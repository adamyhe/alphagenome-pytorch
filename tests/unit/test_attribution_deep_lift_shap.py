"""Unit tests for DeepLIFT/SHAP attribution through tangermeme.

The convergence tests run tangermeme's ``deep_lift_shap`` in float64 on small
probes built from AlphaGenome's real components, and check summation-to-delta:
``sum((x - r) * multipliers) == f(x) - f(r)`` for every sequence-reference pair.
That identity holds only if every nonlinearity between the input and the output
has a DeepLIFT rule, so a missing or wrong rule shows up as a delta far above
float64 rounding (the negative control checks the test can see one).
"""
from __future__ import annotations

import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("tangermeme.deep_lift_shap")
import tangermeme.deep_lift_shap as dls  # noqa: E402

from alphagenome_pytorch import layers  # noqa: E402
from alphagenome_pytorch.attention import MHABlock  # noqa: E402
from alphagenome_pytorch.convolutions import ConvBlock, DownResBlock, UpResBlock  # noqa: E402
from alphagenome_pytorch.embeddings import OutputEmbedder, OutputPair  # noqa: E402
from alphagenome_pytorch.extensions.attribution import (  # noqa: E402
    AttributionResult,
    TrackTarget,
    attributable,
    attribution_ops,
    deep_lift_shap,
    deep_lift_shap_tracks,
)
from alphagenome_pytorch.heads import GenomeTracksHead  # noqa: E402
from alphagenome_pytorch.model import TransformerTower  # noqa: E402


def _onehot(n: int, length: int, seed: int) -> torch.Tensor:
    """(n, 4, length) one-hot in tangermeme's layout."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (n, length), generator=g)
    return F.one_hot(idx, 4).double().transpose(1, 2).contiguous()


@pytest.fixture
def keep_float64(monkeypatch):
    """Make ``Tensor.float()`` a no-op on float64 tensors.

    AlphaGenome hard-codes ``.float()`` casts (attention logits and softmax, the
    track heads' einsum) to match JAX's mixed precision. In a float64 test they
    put a ~1e-5 float32 rounding floor under the convergence deltas, which
    would hide a wrong rule; keeping float64 throughout brings exact rules to
    ~1e-13.
    """
    orig = torch.Tensor.float

    def _float(self, *args, **kwargs):
        if self.dtype == torch.float64:
            return self
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "float", _float)


def _max_relative_delta(model, X, ops, n_shuffles=3, target=0):
    """Largest |f(x) - f(r) - sum((x - r) * m)| over pairs, relative to |f(x) - f(r)|."""
    res = dls.deep_lift_shap(
        model, X, target=target, n_shuffles=n_shuffles, batch_size=n_shuffles,
        raw_outputs=True, return_references=True, additional_nonlinear_ops=ops,
        dtype=torch.float64, device="cpu", random_state=0,
        warning_threshold=float("inf"),
    )
    worst = 0.0
    with torch.no_grad():
        for i in range(X.shape[0]):
            refs, mult = res.references[i], res.attributions[i]  # (K, 4, L)
            fx = model(X[i:i + 1])[:, target]
            fr = model(refs)[:, target]
            diff = fx - fr
            delta = (diff - ((X[i:i + 1] - refs) * mult).sum(dim=(1, 2))).abs()
            worst = max(worst, float((delta / diff.abs().clamp_min(1e-12)).max()))
    return worst


# ---------------------------------------------------------------------------
# Probes: one-hot (N, 4, L) -> real AlphaGenome components -> (N, n_outputs)
# ---------------------------------------------------------------------------


class _ConvProbe(nn.Module):
    """Encoder/decoder path: ConvBlock, max pool, DownResBlock, UpResBlock, embedder."""

    def __init__(self, c: int = 8):
        super().__init__()
        self.stem = nn.Conv1d(4, c, 5, padding=2)
        self.block = ConvBlock(c, c, kernel_size=5)
        self.pool = layers.Pool1d(kernel_size=2)
        self.down = DownResBlock(c)  # c -> c + 128 channels
        self.up = UpResBlock(c + 128, skip_channels=c)
        self.embed = OutputEmbedder(c, 6, num_organisms=1)
        self.out = nn.Linear(6, 2)

    def forward(self, x):
        h = self.stem(x)
        skip = h + self.block(h)
        h = self.down(self.pool(skip))
        h = self.up(h, skip)
        h = self.embed(h, torch.zeros(x.shape[0], dtype=torch.long))  # (N, 6, L)
        return self.out(h.mean(dim=2))


class _TowerProbe(nn.Module):
    """Full TransformerTower: pair updates, attention bias, MHA, MLP, every norm."""

    def __init__(self, d: int = 16, pool: int = 4):
        super().__init__()
        self.pool = pool
        self.stem = nn.Conv1d(4, d, 5, padding=2)
        self.tower = TransformerTower(d_model=d)
        self.pair_out = OutputPair(dim=128, num_organisms=1)
        self.out = nn.Linear(d, 2)
        self.out_pair = nn.Linear(128, 2)

    def forward(self, x):
        h = F.avg_pool1d(self.stem(x), self.pool).transpose(1, 2)  # (N, S, d)
        h, pair = self.tower(h)
        pair = self.pair_out(pair, torch.zeros(x.shape[0], dtype=torch.long))
        return self.out(h.mean(dim=1)) + self.out_pair(pair.mean(dim=(1, 2)))


class _FakeAlphaGenome(nn.Module):
    """Minimal model with AlphaGenome's forward contract and a real GenomeTracksHead."""

    def __init__(self, apply_squashing: bool, c: int = 8, n_tracks: int = 5):
        super().__init__()
        self.stem = nn.Conv1d(4, c, 5, padding=2)
        self.block = ConvBlock(c, c, kernel_size=5)
        means = torch.linspace(0.5, 2.0, 2 * n_tracks).reshape(2, n_tracks)
        self.heads = nn.ModuleDict({
            "rna_seq": GenomeTracksHead(
                in_channels={1: c, 128: c}, num_tracks=n_tracks,
                apply_squashing=apply_squashing, track_means=means,
            ),
        })

    def forward(self, dna_sequence, organism_index, *, heads, resolutions,
                channels_last, return_scaled_predictions):
        h = self.stem(dna_sequence.transpose(1, 2))
        h = h + self.block(h)
        emb = {1: h, 128: F.avg_pool1d(h, 128)}
        return {
            name: self.heads[name](
                emb, organism_index, return_scaled=return_scaled_predictions,
                channels_last=channels_last,
            )
            for name in heads
        }


# ---------------------------------------------------------------------------
# The module rewrite leaves weights and outputs unchanged
# ---------------------------------------------------------------------------


_ADDED = re.compile(r"(^|\.)(act|softplus|max_pool|qk_matmul|av_matmul|logit_cap|qk_einsum|av_einsum)\.")


def test_added_modules_hold_no_state():
    """The hookable modules carry no parameters or buffers, so checkpoints load unchanged."""
    for module in (layers.GELU(), layers.Bilinear(), layers.Bilinear("ij,jk->ik"),
                   layers.Pool1d(2)):
        assert not list(module.state_dict()), type(module).__name__
    tower = TransformerTower(d_model=16)
    head = GenomeTracksHead(in_channels=8, num_tracks=3)
    for m in (tower, head, ConvBlock(8, 8, 5), OutputEmbedder(8, 4), OutputPair(128)):
        assert not [k for k in m.state_dict() if _ADDED.search(k)]


def test_module_forms_match_functions():
    x = torch.randn(2, 6, 32, dtype=torch.float64)
    assert torch.equal(layers.GELU()(x), layers.gelu(x))
    assert torch.equal(layers.Pool1d(2)(x), F.max_pool1d(x, 2, 2))
    a, b = torch.randn(2, 3, 4), torch.randn(2, 4, 5)
    assert torch.equal(layers.Bilinear()(a, b), torch.matmul(a, b))
    assert torch.equal(layers.Bilinear("bij,bjk->bik")(a, b), torch.einsum("bij,bjk->bik", a, b))


def test_bilinear_stages_operands_only_when_asked():
    op = layers.Bilinear()
    op(torch.randn(2, 2), torch.randn(2, 2))
    assert "_staged" not in op.__dict__
    op.stage_operands = True
    a, b = torch.randn(2, 2), torch.randn(2, 2)
    op(a, b)
    assert torch.equal(op._staged["left"], a) and torch.equal(op._staged["right"], b)


def test_attributable_sets_and_restores_state():
    tower = TransformerTower(d_model=16)
    tower.gradient_checkpointing = True
    mhas = [m for m in tower.modules() if isinstance(m, MHABlock)]
    bilinears = [m for m in tower.modules() if isinstance(m, layers.Bilinear)]
    tf32 = torch.backends.cudnn.allow_tf32
    with attributable(tower):
        assert not tower.gradient_checkpointing
        assert not any(m.rope_inplace for m in mhas)
        assert all(m.stage_operands for m in bilinears)
        assert not torch.backends.cudnn.allow_tf32
    assert tower.gradient_checkpointing
    assert all(m.rope_inplace for m in mhas)
    assert not any(m.stage_operands for m in bilinears)
    assert torch.backends.cudnn.allow_tf32 == tf32


# ---------------------------------------------------------------------------
# Summation-to-delta on the real components (float64)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0, 1])
def test_conv_path_converges(target, keep_float64):
    torch.manual_seed(0)
    model = _ConvProbe().double().eval()
    with attributable(model):
        rel = _max_relative_delta(model, _onehot(2, 64, seed=1), attribution_ops(), target=target)
    assert rel < 1e-10


def test_transformer_tower_converges(keep_float64):
    torch.manual_seed(0)
    model = _TowerProbe().double().eval()
    with attributable(model):
        rel = _max_relative_delta(model, _onehot(2, 128, seed=2), attribution_ops())
    assert rel < 1e-10


@pytest.mark.filterwarnings("ignore:Convergence deltas:RuntimeWarning")
def test_missing_rules_are_detected(keep_float64):
    """Negative control: without AlphaGenome's rules (GELU, LayerNorm, Bilinear
    treated as linear) the same check fails by orders of magnitude."""
    torch.manual_seed(0)
    model = _TowerProbe().double().eval()
    with attributable(model):
        rel = _max_relative_delta(model, _onehot(2, 128, seed=2), ops=None)
    assert rel > 1e-4


@pytest.mark.parametrize("apply_squashing", [False, True])
@pytest.mark.parametrize("resolution", [1, 128])
def test_track_target_converges_in_experimental_space(apply_squashing, resolution, keep_float64):
    """Head softplus + experimental-space unscaling (soft clip, power law) converge."""
    torch.manual_seed(0)
    model = _FakeAlphaGenome(apply_squashing).double().eval()
    bins = 512 // resolution
    target = TrackTarget(
        model, organism_index=1, output_type="rna_seq", resolution=resolution,
        target_slice=slice(0, bins), track_indices=[4, 1],
    ).double()
    # Push predictions over the soft-clip threshold so both branches are exercised.
    with torch.no_grad():
        model.heads["rna_seq"].residual_scales[str(resolution)].fill_(30.0)
    with attributable(model):
        rel = _max_relative_delta(target, _onehot(2, 512, seed=3), attribution_ops(), target=1)
    assert rel < 1e-10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_track_target_matches_predict_output():
    torch.manual_seed(0)
    model = _FakeAlphaGenome(apply_squashing=True).eval()
    X = _onehot(2, 256, seed=4).float()
    target = TrackTarget(model, organism_index=0, output_type="rna_seq", resolution=1,
                         target_slice=slice(10, 50), track_indices=[3, 0], reduction="mean")
    with torch.no_grad():
        got = target(X)
        full = model(X.transpose(1, 2), torch.zeros(2, dtype=torch.long), heads=("rna_seq",),
                     resolutions=(1,), channels_last=True,
                     return_scaled_predictions=False)["rna_seq"][1]
    torch.testing.assert_close(got, full[:, 10:50][:, :, [3, 0]].mean(dim=1))


def test_track_target_rejects_max_reduction():
    model = _FakeAlphaGenome(apply_squashing=False)
    with pytest.raises(ValueError, match="max"):
        TrackTarget(model, organism_index=0, output_type="rna_seq", resolution=1,
                    target_slice=slice(0, 4), track_indices=[0], reduction="max")


def test_deep_lift_shap_tracks_shape_and_references_shared():
    torch.manual_seed(0)
    model = _FakeAlphaGenome(apply_squashing=False).eval()
    onehot = _onehot(2, 256, seed=5).float().transpose(1, 2)  # (N, L, 4)
    hyp = deep_lift_shap_tracks(
        model, onehot=onehot, organism_index=0, output_type="rna_seq",
        resolution=128, target_slice=slice(0, 2), track_indices=[0, 2, 2],
        n_shuffles=2, batch_size=2,
    )
    assert hyp.shape == (3, 2, 4, 256)
    # Same track twice -> same seeded references -> identical attributions.
    torch.testing.assert_close(hyp[1], hyp[2])


def test_deep_lift_shap_result_projects_onto_observed_base():
    torch.manual_seed(0)
    model = _FakeAlphaGenome(apply_squashing=False).eval()
    onehot = _onehot(1, 256, seed=6).float().transpose(1, 2)
    res = deep_lift_shap(
        model, onehot=onehot, organism_index=0, output_type="rna_seq",
        resolution=1, target_slice=slice(64, 96), track_indices=[1, 3],
        n_shuffles=2, batch_size=2,
    )
    assert isinstance(res, AttributionResult)
    assert res.method == "deep_lift_shap" and res.values.shape == (32, 4, 2)
    hyp = res.metadata["hypothetical"]
    window = onehot[0, 64:96].numpy()
    for w in range(32):
        b = int(window[w].argmax())
        np.testing.assert_allclose(res.values[w, b], hyp[w, b], rtol=1e-6)
        assert np.isnan(np.delete(res.values[w], b, axis=0)).all()


def test_deep_lift_shap_rejects_n_positions():
    """tangermeme's shuffles need ACGT-only input; an N raises rather than being skipped."""
    model = _FakeAlphaGenome(apply_squashing=False).eval()
    onehot = _onehot(1, 256, seed=6).float().transpose(1, 2)
    onehot[0, 70] = 0.0
    with pytest.raises(ValueError, match="one-hot"):
        deep_lift_shap_tracks(
            model, onehot=onehot, organism_index=0, output_type="rna_seq",
            resolution=128, target_slice=slice(0, 2), track_indices=[0], n_shuffles=2,
        )


def test_deep_lift_shap_requires_float32_policy():
    from alphagenome_pytorch.config import DtypePolicy

    model = _FakeAlphaGenome(apply_squashing=False)
    model.dtype_policy = DtypePolicy.mixed_precision()
    with pytest.raises(ValueError, match="float32"):
        deep_lift_shap_tracks(
            model, onehot=_onehot(1, 256, seed=7).float().transpose(1, 2),
            organism_index=0, output_type="rna_seq", resolution=128,
            target_slice=slice(0, 2), track_indices=[0],
        )


# ---------------------------------------------------------------------------
# Shared-forward engine: same multipliers as stock tangermeme
# ---------------------------------------------------------------------------


def _stock(model, X, targets, n_shuffles, seed, **kw):
    return torch.stack([
        dls.deep_lift_shap(
            model, X, target=t, n_shuffles=n_shuffles, batch_size=n_shuffles,
            additional_nonlinear_ops=attribution_ops(), dtype=X.dtype, device="cpu",
            random_state=seed, warning_threshold=float("inf"), **kw,
        )
        for t in targets
    ])


@pytest.mark.parametrize("probe,length", [(_ConvProbe, 64), (_TowerProbe, 128)])
@pytest.mark.parametrize("hypothetical", [False, True])
def test_shared_forward_matches_stock(probe, length, hypothetical, keep_float64):
    from tangermeme._deep_lift_utils import _build_nonlinear_ops

    from alphagenome_pytorch.extensions.attribution._shared_forward import (
        shared_forward_deep_lift_shap,
    )

    torch.manual_seed(0)
    model = probe().double().eval()
    X = _onehot(2, length, seed=8)
    with attributable(model):
        stock = _stock(model, X, [0, 1], n_shuffles=3, seed=5, hypothetical=hypothetical)
        shared = shared_forward_deep_lift_shap(
            model, X, targets=[0, 1], nonlinear_ops=_build_nonlinear_ops(attribution_ops()),
            n_shuffles=3, random_state=5, hypothetical=hypothetical,
            device="cpu", dtype=torch.float64,
        )
    torch.testing.assert_close(shared.attributions, stock, rtol=1e-10, atol=1e-12)
    rel = shared.convergence_deltas / shared.output_diffs.abs().clamp_min(1e-12)
    assert float(rel.max()) < 1e-10


def test_shared_forward_references_match_tangermeme():
    from alphagenome_pytorch.extensions.attribution._shared_forward import tangermeme_references

    torch.manual_seed(0)
    model = _ConvProbe().double().eval()
    X = _onehot(3, 64, seed=9)
    with attributable(model):
        res = dls.deep_lift_shap(
            model, X, n_shuffles=4, batch_size=5, return_references=True,
            additional_nonlinear_ops=attribution_ops(), dtype=torch.float64,
            device="cpu", random_state=11, warning_threshold=float("inf"),
        )
    torch.testing.assert_close(tangermeme_references(X, 4, 11).double(), res.references)


def test_deep_lift_shap_tracks_engines_agree():
    torch.manual_seed(0)
    model = _FakeAlphaGenome(apply_squashing=True).eval()
    onehot = _onehot(2, 256, seed=10).float().transpose(1, 2)
    kw = dict(onehot=onehot, organism_index=1, output_type="rna_seq", resolution=1,
              target_slice=slice(100, 140), track_indices=[2, 0], n_shuffles=3,
              random_state=3, batch_size=3)
    stock = deep_lift_shap_tracks(model, engine="stock", **kw)
    shared = deep_lift_shap_tracks(model, engine="shared", **kw)
    # float32: rounding only (exact agreement is checked in float64 above).
    torch.testing.assert_close(shared, stock, rtol=0, atol=1e-4 * float(stock.abs().max()))


def test_paired_rules_requires_reference_record():
    from tangermeme._deep_lift_utils import _build_nonlinear_ops

    from alphagenome_pytorch.extensions.attribution._paired_rules import PairedRules

    model = _ConvProbe().double().eval()
    x = _onehot(1, 64, seed=12).requires_grad_(True)
    with PairedRules(model, _build_nonlinear_ops(attribution_ops())) as rules:
        with rules.recording("x"):
            y = model(x)
        with pytest.raises(RuntimeError, match="reference"):
            torch.autograd.grad(y[0, 0], x)
