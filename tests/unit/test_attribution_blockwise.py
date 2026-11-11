"""Unit tests for the blockwise DeepLIFT/SHAP engine.

The engine hand-writes the backward of AlphaGenome's conv path, so it is
checked three ways in float64: its tensor rules against tangermeme's own rule
functions, the whole engine against the shared-forward engine (itself checked
against stock tangermeme in ``test_attribution_deep_lift_shap``) on a
scaled-down AlphaGenome, and the decoder crop against the uncropped decoder.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("tangermeme.deep_lift_shap")

# Some random-init tracks barely move between sequence and reference (|f(x) -
# f(r)| ~ 1e-9), so the engines' relative convergence warning fires on float64
# rounding; the tests check the deltas themselves.
pytestmark = pytest.mark.filterwarnings("ignore:Convergence deltas:RuntimeWarning")
import tangermeme.deep_lift_shap as dls  # noqa: E402
from tangermeme import _deep_lift_utils as rules  # noqa: E402

from alphagenome_pytorch import convolutions, embeddings, heads, layers  # noqa: E402
from alphagenome_pytorch.config import DtypePolicy  # noqa: E402
from alphagenome_pytorch.extensions.attribution import (  # noqa: E402
    TrackTarget,
    attributable,
    attribution_ops,
    deep_lift_shap_tracks,
)
from alphagenome_pytorch.extensions.attribution import _blockwise as bw  # noqa: E402
from alphagenome_pytorch.extensions.attribution.targets import Aggregate, Term  # noqa: E402
from alphagenome_pytorch.extensions.attribution._shared_forward import (  # noqa: E402
    shared_forward_deep_lift_shap,
)
from alphagenome_pytorch.model import (  # noqa: E402
    AlphaGenome,
    SequenceDecoder,
    SequenceEncoder,
    TransformerTower,
)


@pytest.fixture
def keep_float64(monkeypatch):
    """Make ``Tensor.float()`` a no-op on float64 tensors (see test_attribution_deep_lift_shap)."""
    orig = torch.Tensor.float
    monkeypatch.setattr(
        torch.Tensor, "float",
        lambda self, *a, **k: self if self.dtype == torch.float64 else orig(self, *a, **k),
    )


def _onehot(n: int, length: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (n, length), generator=g)
    return F.one_hot(idx, 4).double().transpose(1, 2).contiguous()


class MiniAlphaGenome(AlphaGenome):
    """AlphaGenome's structure (6 down blocks, 9 tower blocks, 7 up blocks, track
    heads) with base width ``c`` instead of 768. Uses AlphaGenome's own forward."""

    def __init__(self, c: int = 8, n_tracks: int = 4, num_organisms: int = 2,
                 dtype: torch.dtype = torch.float64):
        nn.Module.__init__(self)
        torch.manual_seed(0)
        self.num_organisms = num_organisms
        self.dtype_policy = DtypePolicy(dtype, dtype, dtype)
        enc = SequenceEncoder()
        enc.dna_embedder.conv1 = nn.Conv1d(4, c, 15, padding="same")
        enc.dna_embedder.block = convolutions.ConvBlock(c, c, 5)
        enc.down_blocks = nn.ModuleList(convolutions.DownResBlock(c + 128 * i) for i in range(6))
        self.encoder = enc
        d = c + 128 * 6
        self.organism_embed = nn.Embedding(num_organisms, d)
        self.tower = TransformerTower(d_model=d)
        dec = SequenceDecoder()
        skip = {b: c + 128 * i for i, b in enumerate([1, 2, 4, 8, 16, 32, 64])}
        ups, cur = [], d
        for b in dec.bin_sizes:
            ups.append(convolutions.UpResBlock(cur, skip[b]))
            cur = skip[b]
        dec.up_blocks = nn.ModuleList(ups)
        self.decoder = dec
        self.embedder_128bp = embeddings.OutputEmbedder(d, 2 * d, num_organisms)
        self.embedder_1bp = embeddings.OutputEmbedder(c, 16, num_organisms)
        self.embedder_1bp.project_skip = nn.Conv1d(2 * d, 16, 1, bias=False)
        self.embedder_pair = embeddings.OutputPair(dim=128, num_organisms=num_organisms)
        dims = {1: 16, 128: 2 * d}
        self.heads = nn.ModuleDict({
            "dnase": heads.GenomeTracksHead(dims, n_tracks, (1, 128), num_organisms,
                                            track_means=torch.rand(num_organisms, n_tracks) + 0.5),
            "rna_seq": heads.GenomeTracksHead(dims, n_tracks, (1, 128), num_organisms,
                                              apply_squashing=True,
                                              track_means=torch.rand(num_organisms, n_tracks) + 0.5),
        })
        self.contact_maps_head = None
        self.splice_sites_classification_head = None
        self.splice_sites_usage_head = None
        self.splice_sites_junction_head = None
        with torch.no_grad():  # non-trivial norm statistics, so the rules see them
            for m in self.modules():
                if isinstance(m, layers.RMSBatchNorm):
                    m.running_var.uniform_(0.5, 2.0)
                    m.weight.uniform_(0.5, 1.5)
                    m.bias.uniform_(-0.2, 0.2)
        self.to(dtype).eval()


@pytest.fixture(scope="module")
def mini():
    return MiniAlphaGenome()


def _ops():
    return rules._build_nonlinear_ops(attribution_ops())


def _both(model, X, spec, *, margin=2048, n_shuffles=2, **kw):
    with attributable(model):
        shared = shared_forward_deep_lift_shap(
            TrackTarget(model, **spec), X, targets=range(len(spec["track_indices"])),
            nonlinear_ops=_ops(), n_shuffles=n_shuffles, random_state=4, hypothetical=True,
            device="cpu", dtype=torch.float64)
        block = bw.blockwise_deep_lift_shap(
            model, X, nonlinear_ops=_ops(), n_shuffles=n_shuffles, random_state=4,
            hypothetical=True, decoder_margin=margin, device="cpu", dtype=torch.float64,
            **kw, **spec)
    return shared, block


# ---------------------------------------------------------------------------
# Tensor rules against tangermeme's rule functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [64, 63])
def test_maxpool_rule_matches_tangermeme(length):
    torch.manual_seed(0)
    pool = layers.Pool1d(kernel_size=2).double()
    k, T, C = 3, 2, 5
    x = torch.randn(1, C, length, dtype=torch.float64)
    r = torch.randn(k, C, length, dtype=torch.float64)
    r[..., ::5] = x[..., ::5]            # zero deltas: fallback to the plain gradient
    x[..., 1::8] = x[..., 0::8][..., :x[..., 1::8].shape[-1]]  # ties within a window
    Lo = pool(x).shape[-1]
    g = torch.randn(k, T, C, Lo, dtype=torch.float64)

    ours = bw._maxpool_vjp(pool, x, r, g)

    mp = pool.max_pool
    for j in range(k):
        pair = torch.cat([pool.pad(x), pool.pad(r[j:j + 1])]).requires_grad_(True)
        out = mp(pair)
        for t in range(T):
            go = torch.cat([g[j, t:t + 1], g[j, t:t + 1]])
            (gi,) = torch.autograd.grad(out, pair, go, retain_graph=True)
            mp.input, mp.output = pair.detach(), out.detach()
            (ref,) = rules._maxpool(mp, (gi,), (go,))
            torch.testing.assert_close(ours[j, t], ref[0, :, :length], rtol=1e-12, atol=1e-12)
    del mp.input, mp.output


def test_secant_matches_tangermeme_rescale():
    torch.manual_seed(1)
    act = layers.GELU()
    a_x = torch.randn(1, 4, 50, dtype=torch.float64)
    a_r = torch.randn(3, 4, 50, dtype=torch.float64)
    a_r[..., ::7] = a_x[..., ::7]
    a_r[..., 3::11] = a_x[..., 3::11] + 1e-8  # below tangermeme's 1e-6 threshold
    m = bw._secant(act, a_x, a_r)
    for j in range(3):
        pair = torch.cat([a_x, a_r[j:j + 1]]).requires_grad_(True)
        out = act(pair)
        go = torch.ones_like(out)
        (gi,) = torch.autograd.grad(out, pair, go)
        mod = nn.Module()
        mod.input, mod.output = pair.detach(), out.detach()
        (ref,) = rules._nonlinear(mod, (gi,), (go,))
        torch.testing.assert_close(m[j], ref[0], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Whole engine on a scaled-down AlphaGenome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head,resolution,window,reduction,experimental", [
    ("dnase", 128, slice(5, 9), "sum", True),
    ("rna_seq", 128, slice(0, 3), "mean", False),
    ("rna_seq", 1, slice(900, 1300), "mean", True),
    ("dnase", 1, slice(0, 64), "sum", False),
])
def test_blockwise_matches_shared(mini, keep_float64, head, resolution, window,
                                  reduction, experimental):
    X = _onehot(1, 2048, seed=2)
    spec = dict(organism_index=1, output_type=head, resolution=resolution, target_slice=window,
                track_indices=[3, 0, 2], reduction=reduction, experimental_space=experimental)
    shared, block = _both(mini, X, spec, n_shuffles=3, ref_batch_size=2, track_batch_size=2)
    torch.testing.assert_close(block.attributions, shared.attributions, rtol=1e-9,
                               atol=1e-10 * float(shared.attributions.abs().max()))
    torch.testing.assert_close(block.output_diffs, shared.output_diffs, rtol=1e-9, atol=1e-10)
    rel = block.convergence_deltas.max() / block.output_diffs.abs().max()
    assert float(rel) < 1e-10


@pytest.mark.parametrize("fallback", [True, False])
def test_blockwise_matches_stock_tangermeme(mini, keep_float64, monkeypatch, fallback):
    """Against stock tangermeme directly.

    Where ``|delta_in| < 1e-6`` tangermeme's rescale rule falls back to each
    half's own derivative, so the reference half gets ``f'(r)`` rather than
    ``f'(x)``, and max-pool rules further down read that half. The blockwise
    and shared engines use the sequence half for both, so with the fallback
    they agree with stock to ~1e-10 of the largest attribution, and exactly
    without it.
    """
    if not fallback:
        # Fall back only at exact zeros, where f'(x) == f'(r).
        monkeypatch.setattr(rules, "_rescale",
                            lambda di, do, gi, go: torch.where(di == 0, gi, go * (do / di)))
        monkeypatch.setattr(bw, "_RESCALE_EPS", 1e-300)
    X = _onehot(1, 2048, seed=3)
    spec = dict(organism_index=0, output_type="rna_seq", resolution=1,
                target_slice=slice(1000, 1100), track_indices=[1])
    with attributable(mini):
        stock = dls.deep_lift_shap(
            TrackTarget(mini, **spec), X, target=0, n_shuffles=2, batch_size=2,
            hypothetical=True, additional_nonlinear_ops=attribution_ops(),
            dtype=torch.float64, device="cpu", random_state=4,
            warning_threshold=float("inf"))
        block = bw.blockwise_deep_lift_shap(
            mini, X, nonlinear_ops=_ops(), n_shuffles=2, random_state=4, hypothetical=True,
            device="cpu", dtype=torch.float64, **spec)
    tol = 1e-8 if fallback else 1e-12
    torch.testing.assert_close(block.attributions[0], stock, rtol=0,
                               atol=tol * float(stock.abs().max()))


def test_blockwise_decoder_crop_is_exact(mini, keep_float64):
    """A 1 bp window mid-sequence: the decoder runs on the window plus 1 kb only."""
    L = 8192
    X = _onehot(1, L, seed=5)
    spec = dict(organism_index=0, output_type="dnase", resolution=1,
                target_slice=slice(4000, 4200), track_indices=[0])
    plan = bw._make_plan(mini, **spec, reduction="sum", experimental_space=True,
                         length=L, decoder_margin=1024)
    assert plan.crop[1] - plan.crop[0] < L // 2  # the crop is in effect
    shared, block = _both(mini, X, spec, margin=1024)
    torch.testing.assert_close(block.attributions, shared.attributions, rtol=1e-9,
                               atol=1e-10 * float(shared.attributions.abs().max()))
    # A margin below the decoder's receptive radius (762 bp) changes the result.
    _, short = _both(mini, X, spec, margin=128)
    assert not torch.allclose(short.output_diffs, shared.output_diffs, rtol=1e-6)


def test_blockwise_engine_through_deep_lift_shap_tracks():
    model = MiniAlphaGenome(dtype=torch.float32)
    onehot = _onehot(2, 2048, seed=6).float().transpose(1, 2)
    kw = dict(onehot=onehot, organism_index=1, output_type="dnase", resolution=128,
              target_slice=slice(6, 10), track_indices=[1, 2], n_shuffles=3, random_state=1)
    shared = deep_lift_shap_tracks(model, engine="shared", **kw)
    block = deep_lift_shap_tracks(model, engine="blockwise", ref_batch_size=2, **kw)
    assert block.shape == (2, 2, 4, 2048)
    # float32 on a random-init model: secants near tangermeme's 1e-6 threshold
    # are mostly rounding, and stock differs from shared by the same ~1e-4.
    assert float((block - shared).norm() / shared.norm()) < 1e-3


def test_blockwise_rejects_empty_window(mini):
    with pytest.raises(ValueError, match="non-empty"):
        bw._make_plan(mini, organism_index=0, output_type="dnase", resolution=128,
                      target_slice=slice(4, 4), track_indices=[0], reduction="sum",
                      experimental_space=True, length=2048, decoder_margin=1024)


# ---------------------------------------------------------------------------
# Aggregate targets
# ---------------------------------------------------------------------------


def _aggregates(n_bins):
    g = torch.Generator().manual_seed(7)
    profile = torch.rand(3, n_bins, generator=g, dtype=torch.float64)
    return [
        Aggregate.weighted(torch.tensor([1.0, 1.0, 1.0]), name="sum"),
        Aggregate.weighted(torch.tensor([0.5, -1.0, 2.0]), name="contrast"),
        Aggregate.weighted(profile, name="profile"),
        Aggregate.log_sum(torch.tensor([1.0, 0.0, 1.0]), pseudocount=0.5, name="log"),
        Aggregate.log_ratio(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 1.0]),
                            name="lfc"),
        Aggregate((Term(torch.tensor([1.0, 2.0, 0.0]), torch.sqrt, 0.3),
                   Term(profile, None, -0.1)), name="mixed"),
    ]


def _both_agg(model, X, spec, aggs, **kw):
    with attributable(model):
        target = TrackTarget(model, **spec, aggregates=aggs)
        shared = shared_forward_deep_lift_shap(
            target, X, targets=range(target.n_outputs), nonlinear_ops=_ops(), n_shuffles=3,
            random_state=4, hypothetical=True, device="cpu", dtype=torch.float64)
        block = bw.blockwise_deep_lift_shap(
            model, X, nonlinear_ops=_ops(), n_shuffles=3, random_state=4, hypothetical=True,
            device="cpu", dtype=torch.float64, aggregates=aggs, **kw, **spec)
    return shared, block


@pytest.mark.parametrize("head,resolution,window,reduction,experimental", [
    ("dnase", 128, slice(5, 9), "sum", True),
    ("rna_seq", 128, slice(0, 3), "mean", True),
    ("rna_seq", 1, slice(900, 1300), "mean", True),
    ("dnase", 1, slice(0, 64), "sum", False),
])
def test_blockwise_aggregates_match_shared(mini, keep_float64, head, resolution, window,
                                           reduction, experimental):
    X = _onehot(1, 2048, seed=8)
    spec = dict(organism_index=1, output_type=head, resolution=resolution, target_slice=window,
                track_indices=[3, 0, 2], reduction=reduction, experimental_space=experimental)
    aggs = _aggregates(window.stop - window.start)
    shared, block = _both_agg(mini, X, spec, aggs, ref_batch_size=2, track_batch_size=4)
    assert block.attributions.shape == (len(aggs), 1, 4, 2048)
    torch.testing.assert_close(block.output_diffs, shared.output_diffs, rtol=1e-9, atol=1e-12)
    for a in range(len(aggs)):
        torch.testing.assert_close(block.attributions[a], shared.attributions[a], rtol=1e-9,
                                   atol=1e-10 * float(shared.attributions[a].abs().max()))
    # Exact DeepLIFT: attributions sum to f(x) - f(r).
    rel = block.convergence_deltas / block.output_diffs.abs().amax(dim=(1, 2), keepdim=True)
    assert float(rel.max()) < 1e-9


def test_linear_aggregate_is_weighted_sum_of_tracks(mini, keep_float64):
    """A linear aggregate's multipliers are the weighted per-track multipliers."""
    X = _onehot(1, 2048, seed=9)
    spec = dict(organism_index=0, output_type="rna_seq", resolution=1,
                target_slice=slice(300, 400), track_indices=[0, 1, 3])
    w = torch.tensor([0.7, -1.2, 2.0], dtype=torch.float64)
    kw = dict(nonlinear_ops=_ops(), n_shuffles=2, random_state=4, raw_outputs=True,
              device="cpu", dtype=torch.float64, **spec)
    with attributable(mini):
        per_track = bw.blockwise_deep_lift_shap(mini, X, **kw)
        agg = bw.blockwise_deep_lift_shap(mini, X, aggregates=[Aggregate.weighted(w)], **kw)
    expected = torch.einsum("t,tnk...->nk...", w, per_track.attributions)
    torch.testing.assert_close(agg.attributions[0], expected, rtol=1e-10,
                               atol=1e-12 * float(expected.abs().max()))


def test_blockwise_aggregate_matches_stock_tangermeme(mini, keep_float64, monkeypatch):
    """A log fold change through stock tangermeme (link rule fallback off, as above)."""
    monkeypatch.setattr(rules, "_rescale",
                        lambda di, do, gi, go: torch.where(di == 0, gi, go * (do / di)))
    monkeypatch.setattr(bw, "_RESCALE_EPS", 1e-300)
    X = _onehot(1, 2048, seed=10)
    spec = dict(organism_index=0, output_type="dnase", resolution=1,
                target_slice=slice(1000, 1100), track_indices=[1, 2])
    aggs = [Aggregate.log_ratio(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))]
    with attributable(mini):
        stock = dls.deep_lift_shap(
            TrackTarget(mini, **spec, aggregates=aggs), X, target=0, n_shuffles=2,
            batch_size=2, hypothetical=True, additional_nonlinear_ops=attribution_ops(),
            dtype=torch.float64, device="cpu", random_state=4, warning_threshold=float("inf"))
        block = bw.blockwise_deep_lift_shap(
            mini, X, nonlinear_ops=_ops(), n_shuffles=2, random_state=4, hypothetical=True,
            device="cpu", dtype=torch.float64, aggregates=aggs, **spec)
    torch.testing.assert_close(block.attributions[0], stock, rtol=0,
                               atol=1e-12 * float(stock.abs().max()))


def test_aggregates_through_deep_lift_shap_tracks():
    model = MiniAlphaGenome(dtype=torch.float32)
    onehot = _onehot(2, 2048, seed=11).float().transpose(1, 2)
    aggs = [Aggregate.weighted(torch.ones(3)), Aggregate.log_sum(torch.ones(3))]
    kw = dict(onehot=onehot, organism_index=1, output_type="dnase", resolution=128,
              target_slice=slice(6, 10), track_indices=[0, 1, 2], n_shuffles=3,
              random_state=1, aggregates=aggs)
    shared = deep_lift_shap_tracks(model, engine="shared", **kw)
    block = deep_lift_shap_tracks(model, engine="blockwise", **kw)
    assert block.shape == (2, 2, 4, 2048)
    assert float((block - shared).norm() / shared.norm()) < 1e-3


@pytest.mark.parametrize("weights", [torch.ones(2), torch.ones(3, 5), torch.ones(3, 4, 1)])
def test_aggregate_weight_shape_is_checked(mini, weights):
    with pytest.raises(ValueError, match="expected"):
        bw._make_plan(mini, organism_index=0, output_type="dnase", resolution=128,
                      target_slice=slice(4, 8), track_indices=[0, 1, 2], reduction="sum",
                      experimental_space=True, length=2048, decoder_margin=1024,
                      aggregates=[Aggregate.weighted(weights)])
