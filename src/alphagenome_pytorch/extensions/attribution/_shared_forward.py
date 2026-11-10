"""Shared-forward DeepLIFT/SHAP: one forward of each sequence for all references and outputs.

Model-agnostic: ``model`` is any ``(N, 4, L) -> (N, n_outputs)`` module, as for
``tangermeme.deep_lift_shap.deep_lift_shap``, and the multipliers are the ones
tangermeme computes (same rules, same references for the same seeds). Per
sequence, tangermeme runs ``n_shuffles * n_outputs`` forward and backward passes
over two rows each. This runs:

* one forward of the sequence (one row, graph kept),
* one ``no_grad`` forward per reference, recording only rule-site activations,
* one one-row backward per (reference, output), over the kept graph.

So the graph is half as wide (one row rather than a sequence-reference pair),
the sequence forward is shared by every reference and output, and each
reference forward is shared by every output. A candidate to move into tangermeme.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from ._paired_rules import PairedRules


@dataclass
class SharedForwardResult:
    """Outputs of :func:`shared_forward_deep_lift_shap`.

    Attributes:
        attributions: ``(T, N, 4, L)``; hypothetical contributions averaged over
            references if ``hypothetical`` else those times the input, or the raw
            per-reference multipliers ``(T, N, K, 4, L)`` if ``raw_outputs``.
        references: ``(N, K, 4, L)`` references used.
        convergence_deltas: ``(T, N, K)`` ``|f(x) - f(r) - sum((x - r) * m)|``.
        output_diffs: ``(T, N, K)`` ``f(x) - f(r)``.
    """

    attributions: torch.Tensor
    references: torch.Tensor
    convergence_deltas: torch.Tensor
    output_diffs: torch.Tensor


def tangermeme_references(X: torch.Tensor, n_shuffles: int, random_state: Optional[int]) -> torch.Tensor:
    """``(N, K, 4, L)`` dinucleotide shuffles, drawn as tangermeme's ``deep_lift_shap`` draws them.

    With a ``random_state``, pair ``(i, k)`` is seeded ``random_state + k``, so
    the same seed reproduces tangermeme's references exactly.
    """
    from tangermeme.ersatz import dinucleotide_shuffle

    if random_state is None:
        return dinucleotide_shuffle(X, n=n_shuffles)
    refs = [
        dinucleotide_shuffle(X, n=1, random_state=[random_state + k] * X.shape[0])[:, 0]
        for k in range(n_shuffles)
    ]
    return torch.stack(refs, dim=1)


def shared_forward_deep_lift_shap(
    model: nn.Module,
    X: torch.Tensor,
    *,
    targets: Sequence[int],
    nonlinear_ops: dict,
    references: Optional[torch.Tensor] = None,
    n_shuffles: int = 20,
    random_state: Optional[int] = None,
    hypothetical: bool = False,
    raw_outputs: bool = False,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
    relative_warning_threshold: float = 1e-3,
    clone: bool = False,
) -> SharedForwardResult:
    """DeepLIFT/SHAP multipliers for several outputs from one forward per sequence.

    Args:
        model: ``(N, 4, L) -> (N, n_outputs)``; in eval mode, rows independent.
        X: ``(N, 4, L)`` one-hot sequences.
        targets: Output columns to attribute.
        nonlinear_ops: Full rule table (built-ins plus additions), keyed by type.
        references: ``(N, K, 4, L)``; drawn with :func:`tangermeme_references` if None.
        n_shuffles, random_state: Reference count and seed when drawing them.
        hypothetical, raw_outputs: As in tangermeme's ``deep_lift_shap``.
        device, dtype: Where and in what precision to run.
        relative_warning_threshold: Warn when a convergence delta exceeds this
            fraction of the mean ``|f(x) - f(r)|`` for its output.
        clone: Passed to :class:`PairedRules`.
    """
    from tangermeme.deep_lift_shap import hypothetical_attributions

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    if references is None:
        references = tangermeme_references(X, n_shuffles, random_state)
    N, K = references.shape[:2]
    targets = [int(t) for t in targets]
    T = len(targets)

    mult = torch.zeros((T, N, K, *X.shape[1:]), dtype=dtype)
    diffs = torch.zeros((T, N, K), dtype=torch.float64)
    deltas = torch.zeros((T, N, K), dtype=torch.float64)

    with PairedRules(model, nonlinear_ops, clone=clone) as rules:
        for i in range(N):
            x = X[i:i + 1].to(device=device, dtype=dtype).requires_grad_(True)
            with torch.enable_grad(), rules.recording("x"):
                y = model(x)
            if y.ndim != 2:
                y = y.reshape(y.shape[0], -1)

            for k in range(K):
                r = references[i, k:k + 1].to(device=device, dtype=dtype)
                with torch.no_grad(), rules.recording("ref"):
                    y_r = model(r).reshape(1, -1)
                for t_i, t in enumerate(targets):
                    last = k == K - 1 and t_i == T - 1
                    (g,) = torch.autograd.grad(y[0, t], x, retain_graph=not last)
                    diff = float(y[0, t].detach() - y_r[0, t])
                    summed = float(((x.detach() - r) * g).sum())
                    mult[t_i, i, k] = g[0].cpu()
                    diffs[t_i, i, k] = diff
                    deltas[t_i, i, k] = abs(diff - summed)
            rules.clear()
            del y, x

    scale = diffs.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    worst = float((deltas / scale).max()) if deltas.numel() else 0.0
    if worst > relative_warning_threshold:
        warnings.warn(
            f"Convergence deltas reach {worst:.2e} of the mean |f(x) - f(r)|: an "
            "operation may lack a DeepLIFT rule, or numerical precision is limiting.",
            RuntimeWarning,
        )

    if raw_outputs:
        attributions = mult
    else:
        flat = mult.reshape(T * N * K, *X.shape[1:])
        xs = X.to(dtype).unsqueeze(0).unsqueeze(2).expand(T, N, K, *X.shape[1:]).reshape(flat.shape)
        rs = references.to(dtype).unsqueeze(0).expand(T, N, K, *X.shape[1:]).reshape(flat.shape)
        hyp = hypothetical_attributions((flat,), (xs,), (rs,))[0]
        attributions = hyp.reshape(T, N, K, *X.shape[1:]).mean(dim=2)
        if not hypothetical:
            attributions = attributions * X.to(dtype).unsqueeze(0)

    return SharedForwardResult(
        attributions=attributions, references=references,
        convergence_deltas=deltas, output_diffs=diffs,
    )
