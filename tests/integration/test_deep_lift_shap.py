"""DeepLIFT/SHAP on the pretrained model: summation-to-delta through every head type.

Run with ``pytest tests/integration/test_deep_lift_shap.py --torch-weights=<.pth|.safetensors>``.
Uses a 4 kb input so the stock (autograd-graph) path fits a consumer GPU.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("tangermeme.deep_lift_shap")
import tangermeme.deep_lift_shap as dls  # noqa: E402

from alphagenome_pytorch import AlphaGenome  # noqa: E402
from alphagenome_pytorch.extensions.attribution import (  # noqa: E402
    TrackTarget,
    attributable,
    attribution_ops,
)

pytestmark = pytest.mark.integration

L = 4096


@pytest.fixture(scope="module")
def model(torch_weights_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AlphaGenome.from_pretrained(torch_weights_path, device=device).eval()


@pytest.mark.parametrize(
    "output_type,resolution,organism",
    [("dnase", 128, 0), ("dnase", 1, 0), ("rna_seq", 1, 0), ("chip_histone", 128, 1)],
)
def test_pretrained_converges(model, output_type, resolution, organism):
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(0)
    X = F.one_hot(torch.randint(0, 4, (1, L), generator=g), 4).float().transpose(1, 2)
    bins = L // resolution
    target = TrackTarget(
        model, organism_index=organism, output_type=output_type, resolution=resolution,
        target_slice=slice(bins // 2 - 4, bins // 2 + 4), track_indices=[0, 7],
    )
    with attributable(model):
        res = dls.deep_lift_shap(
            target, X, target=1, n_shuffles=2, batch_size=1, raw_outputs=True,
            return_references=True, additional_nonlinear_ops=attribution_ops(),
            device=device, random_state=0, warning_threshold=float("inf"),
        )
        refs, mult = res.references[0], res.attributions[0]
        with torch.no_grad():
            diff = (target(X.to(device))[:, 1] - target(refs.to(device))[:, 1]).cpu()
    summed = ((X - refs) * mult).sum(dim=(1, 2))
    # float32 through a 450M-parameter model: ~1e-6 observed.
    rel = ((diff - summed).abs() / diff.abs().clamp_min(1e-6)).max()
    assert rel < 1e-4, (diff, summed)
