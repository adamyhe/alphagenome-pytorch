"""A scaled-down AlphaGenome for attribution tests (shared by several test modules)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from alphagenome_pytorch import convolutions, embeddings, heads, layers
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.model import (
    AlphaGenome,
    SequenceDecoder,
    SequenceEncoder,
    TransformerTower,
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
