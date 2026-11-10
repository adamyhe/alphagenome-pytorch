"""DeepLIFT/SHAP attribution through tangermeme (the stock path).

Runs ``tangermeme.deep_lift_shap.deep_lift_shap`` (tangermeme >= 1.6) on an
AlphaGenome genome-track head. Every nonlinearity in the path from the input to
a track head is a module that tangermeme has a closed-form rule for:

=====================================  ==========================================
Module                                 Rule
=====================================  ==========================================
``layers.GELU``, ``nn.GELU``,          rescale (``_nonlinear``)
``nn.ReLU``, ``nn.Tanh``,
``nn.Softplus``, track unscaling
``nn.MaxPool1d`` (encoder pooling)     ``_maxpool``
``nn.Softmax`` (both attentions)       ``_softmax``
``layers.LayerNorm``                   ``_layernorm`` / ``_rmsnorm`` (by
                                       ``rms_norm``)
``layers.Bilinear`` (q.k, attn.v)      ``_bilinear`` (midpoint product rule)
=====================================  ==========================================

Everything else (eval-mode ``RMSBatchNorm``, ``StandardizedConv1d``, Linear,
RoPE, residuals, U-Net skips, mean pooling, repeats, organism embeddings) is
linear in the input, so its ordinary gradient is exact. Attributions therefore
satisfy summation-to-delta up to floating-point error.

This path keeps tangermeme's full autograd graph over ``2 * batch_size`` rows,
so it is memory-hungry at the full 131 kb context. It is the reference a
faster engine is checked against.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ... import layers
from ...attention import MHABlock
from ...heads import GenomeTracksHead, predictions_scaling
from .types import BASES, AttributionResult


_REDUCTIONS = ("sum", "mean")


def _require_tangermeme():
    try:
        import tangermeme.deep_lift_shap as dls
        from tangermeme import _deep_lift_utils as rules
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "DeepLIFT/SHAP attribution requires tangermeme>=1.6.0. Install it "
            "with `pip install 'alphagenome-pytorch[attribution]'`."
        ) from e
    if not hasattr(rules, "_layernorm") or not hasattr(dls, "BilinearOp"):
        raise ImportError(
            "DeepLIFT/SHAP attribution requires tangermeme>=1.6.0 (LayerNorm, "
            "RMSNorm, softmax and bilinear rules)."
        )
    return dls, rules


class _ExperimentalScale(nn.Module):
    """Elementwise map from model space to experimental space for some tracks.

    ``GenomeTracksHead.unscale`` (soft-clip expansion, optional power law,
    track means x resolution) is elementwise in the prediction, so the rescale
    rule is exact for it. Wrapping it in one module lets that rule replace the
    gradient of the whole map, including its ``torch.where``.
    """

    def __init__(self, head: GenomeTracksHead, resolution: int, organism_index: int,
                 track_indices: Sequence[int]):
        super().__init__()
        slot = 0 if head.num_organisms == 1 else int(organism_index)
        means = head.track_means[slot, list(track_indices)].detach().clone()
        self.register_buffer("track_means", means, persistent=False)  # (T,)
        self.resolution = int(resolution)
        self.apply_squashing = bool(head.apply_squashing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, bins, T)
        means = self.track_means.to(x.dtype).expand(x.shape[0], -1)
        return predictions_scaling(
            x, track_means=means, resolution=self.resolution,
            apply_squashing=self.apply_squashing, channels_last=True,
        )


class TrackTarget(nn.Module):
    """Wrap AlphaGenome as the ``(N, 4, L) -> (N, n_tracks)`` model tangermeme expects.

    Output ``[:, t]`` is the ``reduction`` over the bins in ``target_slice`` of
    track ``track_indices[t]`` of head ``output_type`` at ``resolution``, in
    experimental space if ``experimental_space`` else model space. The organism
    is fixed at construction because tangermeme casts extra ``args`` to the
    input dtype.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        organism_index: int,
        output_type: str,
        resolution: int,
        target_slice: slice,
        track_indices: Sequence[int],
        reduction: str = "sum",
        experimental_space: bool = True,
    ):
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {_REDUCTIONS} for DeepLIFT/SHAP; got "
                f"{reduction!r}. ('max' is not supported: it is a nonlinearity "
                f"with no module to attach a rule to.)"
            )
        heads = getattr(model, "heads", None)
        if heads is None or output_type not in heads:
            raise KeyError(
                f"Model has no genome-track head {output_type!r}. DeepLIFT/SHAP "
                f"supports genome-track heads only."
            )
        head = heads[output_type]
        if not isinstance(head, GenomeTracksHead):
            raise TypeError(f"Head {output_type!r} is not a GenomeTracksHead.")
        if resolution not in head.resolutions:
            raise KeyError(
                f"Head {output_type!r} has no resolution {resolution}; "
                f"available: {sorted(head.resolutions)}."
            )

        self.model = model
        self.organism_index = int(organism_index)
        self.output_type = output_type
        self.resolution = int(resolution)
        self.target_slice = target_slice
        self.track_indices = [int(t) for t in track_indices]
        self.reduction = reduction
        self.scale = (
            _ExperimentalScale(head, resolution, organism_index, self.track_indices)
            if experimental_space else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        onehot = x.transpose(1, 2)  # (N, L, 4)
        org = torch.full((x.shape[0],), self.organism_index, dtype=torch.long,
                         device=x.device)
        out = self.model(
            onehot, org,
            heads=(self.output_type,), resolutions=(self.resolution,),
            channels_last=True, return_scaled_predictions=True,
        )[self.output_type][self.resolution]  # (N, bins, all tracks), model space
        out = out[:, self.target_slice][:, :, self.track_indices]
        if self.scale is not None:
            out = self.scale(out)
        return out.sum(dim=1) if self.reduction == "sum" else out.mean(dim=1)


def attribution_ops() -> dict:
    """Rules for AlphaGenome's own module types, for ``additional_nonlinear_ops``.

    Pass to ``tangermeme.deep_lift_shap.deep_lift_shap`` or
    ``tangermeme.pisa.pisa`` when calling tangermeme directly (inside
    :func:`attributable`). Types tangermeme already covers (``nn.Softmax``,
    ``nn.GELU``, ``nn.ReLU``, ``nn.Tanh``, ``nn.Softplus``, ``nn.MaxPool1d``)
    are not repeated. A fresh dict is returned per call.
    """
    _, rules = _require_tangermeme()

    def _alphagenome_layer_norm(module, grad_input, grad_output):
        rule = rules._rmsnorm if module.rms_norm else rules._layernorm
        return rule(module, grad_input, grad_output)

    _alphagenome_layer_norm._reads = ("input",)

    return {
        layers.GELU: rules._nonlinear,
        layers.LayerNorm: _alphagenome_layer_norm,
        layers.Bilinear: rules._bilinear,
        _ExperimentalScale: rules._nonlinear,
    }


@contextlib.contextmanager
def attributable(model: nn.Module):
    """Put an AlphaGenome model in the state hook-based DeepLIFT needs, then restore it.

    * ``layers.Bilinear`` modules stage their operands for the bilinear rule.
    * In-place RoPE is turned off (module backward hooks reject in-place
      writes to a hooked module's output).
    * Gradient checkpointing is turned off: recomputing a segment would rerun
      the forward hooks and misalign their activation caches.
    * TF32 is turned off for matmuls and cuDNN convolutions. With it on, the
      prediction differences tangermeme checks against are themselves off by
      ~1e-3 relative, and convergence cannot be told apart from TF32 error.
    """
    saved = []
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for m in model.modules():
            if isinstance(m, layers.Bilinear):
                saved.append((m, "stage_operands", m.stage_operands))
                m.stage_operands = True
            elif isinstance(m, MHABlock):
                saved.append((m, "rope_inplace", m.rope_inplace))
                m.rope_inplace = False
            if getattr(m, "gradient_checkpointing", False):
                saved.append((m, "gradient_checkpointing", True))
                m.gradient_checkpointing = False
        yield model
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        for m, name, value in reversed(saved):
            setattr(m, name, value)
        for m in model.modules():
            if isinstance(m, layers.Bilinear):
                m.__dict__.pop("_staged", None)


def _check_fp32(model: nn.Module) -> None:
    policy = getattr(model, "dtype_policy", None)
    compute = getattr(policy, "compute_dtype", torch.float32)
    if compute != torch.float32:
        raise ValueError(
            "DeepLIFT/SHAP runs in float32; the model's dtype policy computes in "
            f"{compute}. Load it with DtypePolicy.full_float32()."
        )


def deep_lift_shap_tracks(
    model: nn.Module,
    *,
    onehot: torch.Tensor,
    organism_index: int,
    output_type: str,
    resolution: int,
    target_slice: slice,
    track_indices: Sequence[int],
    reduction: str = "sum",
    experimental_space: bool = True,
    n_shuffles: int = 20,
    references: Optional[torch.Tensor] = None,
    random_state: Optional[int] = 0,
    batch_size: int = 1,
    hypothetical: bool = True,
    warning_threshold: Optional[float] = None,
    device: Optional[str | torch.device] = None,
    verbose: bool = False,
    engine: str = "shared",
) -> torch.Tensor:
    """DeepLIFT/SHAP attributions over the whole input.

    Args:
        model: An AlphaGenome model with a float32 dtype policy.
        onehot: ``(N, L, 4)`` one-hot input (the model's NLC layout). Every
            position must be A, C, G or T: tangermeme's dinucleotide shuffles
            reject N (all-zero) columns with a ``ValueError``.
        organism_index: 0 (human) or 1 (mouse).
        output_type: Genome-track head name (e.g. ``"dnase"``).
        resolution: Output resolution in bp (1 or 128).
        target_slice: Resolution-bin slice of the attributed window.
        track_indices: Tracks to attribute.
        reduction: ``"sum"`` or ``"mean"`` over the window bins.
        experimental_space: Attribute the experimental-space prediction (what
            ``predict`` returns) rather than the model-space one.
        n_shuffles: Dinucleotide-shuffled references per sequence (ignored when
            ``references`` is given).
        references: Optional ``(N, K, 4, L)`` references (tangermeme layout).
        random_state: Seed for the shuffles, drawn as tangermeme draws them, so
            both engines use the same references. With a seed, every track uses
            the same references. None draws unseeded references (per track with
            ``engine="stock"``, once for all tracks with ``"shared"``).
        batch_size: ``engine="stock"`` only: sequence-reference pairs per
            forward (each pair is two rows).
        hypothetical: Return hypothetical contributions (for TF-MoDISco);
            otherwise contributions projected onto the observed bases.
        warning_threshold: Absolute convergence-delta threshold for
            tangermeme's warning. None (default) turns the absolute check off
            and leaves tangermeme's relative one (deltas above 1% of the
            prediction differences). AlphaGenome's outputs are large enough
            (sums over bins of counts) that float32 rounding alone exceeds
            tangermeme's default of 1e-3; on the pretrained model the deltas
            are ~1e-6 of the prediction differences.
        device: Device to run on (default: the model's).
        verbose: ``engine="stock"`` only: show tangermeme's progress bar.
        engine: ``"shared"`` (default) forwards each sequence once with
            gradients, each reference once without, and runs one one-row
            backward per (reference, track) over the kept graph, applying
            tangermeme's rules (see ``_shared_forward``). ``"stock"`` calls
            ``tangermeme.deep_lift_shap.deep_lift_shap`` once per track. The two
            agree to float rounding for the same references; the stock engine
            is the reference implementation.

    Returns:
        ``(T, N, 4, L)`` float32 tensor on the CPU, ``T = len(track_indices)``.
    """
    dls, _ = _require_tangermeme()
    _check_fp32(model)
    if onehot.dim() != 3 or onehot.shape[2] != 4:
        raise ValueError(f"onehot must have shape (N, L, 4); got {tuple(onehot.shape)}.")
    if not track_indices:
        raise ValueError("track_indices must be non-empty.")
    if device is None:
        device = next(model.parameters()).device

    target = TrackTarget(
        model, organism_index=organism_index, output_type=output_type,
        resolution=resolution, target_slice=target_slice,
        track_indices=track_indices, reduction=reduction,
        experimental_space=experimental_space,
    )
    X = onehot.detach().float().transpose(1, 2).contiguous().cpu()  # (N, 4, L)
    kwargs = dict(
        batch_size=batch_size, hypothetical=hypothetical,
        warning_threshold=float("inf") if warning_threshold is None else warning_threshold,
        additional_nonlinear_ops=attribution_ops(),
        dtype=torch.float32, device=device, random_state=random_state,
        verbose=verbose,
    )
    if references is not None:
        kwargs["references"] = references.float().cpu()
    else:
        kwargs["n_shuffles"] = n_shuffles

    if engine == "shared":
        from tangermeme._deep_lift_utils import _build_nonlinear_ops

        from ._shared_forward import shared_forward_deep_lift_shap

        with attributable(model):
            result = shared_forward_deep_lift_shap(
                target, X, targets=range(len(target.track_indices)),
                nonlinear_ops=_build_nonlinear_ops(attribution_ops()),
                references=kwargs.get("references"), n_shuffles=n_shuffles,
                random_state=random_state, hypothetical=hypothetical,
                device=device, dtype=torch.float32,
            )
        return result.attributions.float()
    if engine != "stock":
        raise ValueError(f"engine must be 'shared' or 'stock'; got {engine!r}.")

    out = []
    with attributable(model):
        for t in range(len(target.track_indices)):
            out.append(dls.deep_lift_shap(target, X, target=t, **kwargs).float())
    return torch.stack(out)


def deep_lift_shap(
    model: nn.Module,
    *,
    onehot: torch.Tensor,
    organism_index: int,
    output_type: str,
    resolution: int,
    target_slice: slice,
    track_indices: Sequence[int],
    reduction: str = "sum",
    n_shuffles: int = 20,
    random_state: Optional[int] = 0,
    batch_size: int = 1,
    sequence: str = "",
    target_start: int = 0,
    target_end: int = 0,
    **kwargs,
) -> AttributionResult:
    """DeepLIFT/SHAP attribution in the ``base_matrix`` format of :func:`gradient_x_input`.

    ``onehot`` is ``(1, L, 4)``. ``values[w, b, t]`` is the contribution at
    position ``w`` of the attributed window (the ``target_slice`` bins, in bp)
    for the observed base; other cells are NaN. ``metadata["hypothetical"]``
    holds the full ``(W, 4, T)`` hypothetical contributions. Extra keyword
    arguments go to :func:`deep_lift_shap_tracks`.
    """
    if onehot.dim() != 3 or onehot.shape[0] != 1 or onehot.shape[2] != 4:
        raise ValueError(f"onehot must have shape (1, L, 4); got {tuple(onehot.shape)}.")
    for unsupported in ("include_raw_gradient", "strand_averaged", "head_selector"):
        if kwargs.pop(unsupported, None):
            raise ValueError(f"{unsupported} is not supported for deep_lift_shap.")
    kwargs.pop("hypothetical", None)

    hyp = deep_lift_shap_tracks(
        model, onehot=onehot, organism_index=organism_index,
        output_type=output_type, resolution=resolution, target_slice=target_slice,
        track_indices=track_indices, reduction=reduction, n_shuffles=n_shuffles,
        random_state=random_state, batch_size=batch_size, hypothetical=True,
        **kwargs,
    )  # (T, 1, 4, L)
    lo, hi = target_slice.start * resolution, target_slice.stop * resolution
    hyp_w = hyp[:, 0, :, lo:hi].permute(2, 1, 0).numpy()  # (W, 4, T)
    window = onehot[0, lo:hi].detach().float().cpu().numpy()  # (W, 4)

    values = np.full(hyp_w.shape, np.nan, dtype=np.float32)
    proj = (hyp_w * window[:, :, None]).sum(axis=1)  # (W, T)
    has_ref = window.sum(axis=1) > 0.5
    ref = window.argmax(axis=1)
    rows = np.nonzero(has_ref)[0]
    values[rows, ref[rows], :] = proj[rows]

    return AttributionResult(
        method="deep_lift_shap",
        kind="base_matrix",
        bases=BASES,
        values=values,
        sequence=sequence,
        target_start=target_start,
        target_end=target_end,
        resolution=resolution,
        track_indices=tuple(int(i) for i in track_indices),
        reduction=reduction,
        metadata={
            "n_shuffles": n_shuffles, "random_state": random_state,
            "hypothetical": hyp_w,
        },
    )
