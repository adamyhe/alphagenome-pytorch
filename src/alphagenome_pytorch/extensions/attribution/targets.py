"""Aggregate DeepLIFT/SHAP targets: scalar functions of a window of track predictions.

By default each requested track is its own target: the sum (or mean) of its
predictions over the window. An :class:`Aggregate` instead combines the window
predictions ``y[t, w]`` of the requested tracks (``t`` indexes
``track_indices``, ``w`` the window bins) into one scalar::

    f(y) = sum_j coef_j * link_j(s_j),    s_j = reduce_w(sum_t W_j[t, w] * y[t, w])

where ``reduce_w`` is the window ``reduction`` (``"mean"`` divides by the
number of bins). Each term is a weighted sum, then an optional scalar link
(e.g. ``log``), then a coefficient. DeepLIFT is exact for this form: the
weighted sums and the final combination are linear, and the rescale rule on a
scalar link gives its exact secant, so attributions still sum to
``f(x) - f(r)``.

Examples::

    Aggregate.weighted(torch.ones(T))                     # sum of all tracks
    Aggregate.weighted(w_k562 - w_hepg2)                  # a cell-type contrast
    Aggregate.log_sum(torch.ones(T), pseudocount=1.0)     # log(1 + total signal)
    Aggregate.log_ratio(w_alt, w_ref)                     # log fold change
    Aggregate.weighted(torch.ones(T, W) * profile)        # position-weighted

A link must be an elementwise, differentiable torch function of a tensor (its
derivative is the fallback where sequence and reference sums nearly coincide).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from torch import nn

Link = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Term:
    """``coef * link(reduce_w(sum_t weights[t, w] * y[t, w]))``.

    Args:
        weights: ``(T,)`` per-track weights (the same for every window bin) or
            ``(T, W)`` per-track, per-bin weights, aligned with ``track_indices``
            and the window bins.
        link: Elementwise scalar function applied to the weighted sum, or None
            for the identity.
        coef: Multiplies the linked value.
    """

    weights: torch.Tensor
    link: Optional[Link] = None
    coef: float = 1.0


@dataclass(frozen=True)
class Aggregate:
    """One scalar DeepLIFT/SHAP target: a sum of :class:`Term`. See the module docstring."""

    terms: tuple = field(default_factory=tuple)
    name: str = ""

    def __post_init__(self):
        if not self.terms:
            raise ValueError("Aggregate needs at least one term.")
        object.__setattr__(self, "terms", tuple(self.terms))

    @classmethod
    def weighted(cls, weights, *, link: Optional[Link] = None, name: str = "") -> "Aggregate":
        """``link(weighted sum)``; with no link, an exact linear combination."""
        return cls((Term(torch.as_tensor(weights), link),), name)

    @classmethod
    def log_sum(cls, weights, *, pseudocount: float = 1.0, name: str = "") -> "Aggregate":
        """``log(pseudocount + weighted sum)``."""
        return cls.weighted(weights, link=log_link(pseudocount), name=name)

    @classmethod
    def log_ratio(cls, numerator, denominator, *, pseudocount: float = 1.0,
                  name: str = "") -> "Aggregate":
        """``log(pc + numerator sum) - log(pc + denominator sum)``, e.g. a log fold change."""
        link = log_link(pseudocount)
        return cls((Term(torch.as_tensor(numerator), link, 1.0),
                    Term(torch.as_tensor(denominator), link, -1.0)), name)


def log_link(pseudocount: float = 1.0) -> Link:
    """``s -> log(s + pseudocount)``."""
    pc = float(pseudocount)

    def link(s: torch.Tensor) -> torch.Tensor:
        return torch.log(s + pc)

    link.__name__ = f"log(s + {pc:g})"
    return link


@dataclass
class _Prepared:
    """An aggregate's terms as tensors: ``weights`` ``(J, T, W)`` (reduction folded in)."""

    weights: torch.Tensor
    links: list
    coefs: torch.Tensor  # (J,)

    def sums(self, y: torch.Tensor) -> torch.Tensor:
        """``(B, T, W)`` window predictions -> ``(B, J)`` weighted sums."""
        return torch.einsum("btw,jtw->bj", y, self.weights.to(y.dtype))

    def value(self, s: torch.Tensor) -> torch.Tensor:
        """``(B, J)`` sums -> ``(B,)`` aggregate."""
        out = [s[:, j] if f is None else f(s[:, j]) for j, f in enumerate(self.links)]
        return (torch.stack(out, dim=1) * self.coefs.to(s.dtype)).sum(dim=1)


def prepare_aggregates(aggregates: Sequence[Aggregate], n_tracks: int, n_bins: int,
                       reduction: str, *, dtype=torch.float64, device=None) -> list:
    """Validate ``aggregates`` against ``n_tracks`` tracks and ``n_bins`` window bins."""
    out = []
    for a, agg in enumerate(aggregates):
        if not isinstance(agg, Aggregate):
            raise TypeError(f"aggregates[{a}] is {type(agg).__name__}, not an Aggregate.")
        ws = []
        for term in agg.terms:
            w = torch.as_tensor(term.weights).to(dtype=dtype, device=device)
            if w.shape == (n_tracks,):
                w = w[:, None].expand(n_tracks, n_bins)
            elif w.shape != (n_tracks, n_bins):
                raise ValueError(
                    f"aggregates[{a}] term weights have shape {tuple(w.shape)}; expected "
                    f"({n_tracks},) or ({n_tracks}, {n_bins}) for {n_tracks} tracks and "
                    f"{n_bins} window bins."
                )
            if not bool(torch.isfinite(w).all()):
                raise ValueError(f"aggregates[{a}] term weights must be finite.")
            ws.append(w / n_bins if reduction == "mean" else w)
        out.append(_Prepared(
            weights=torch.stack(ws),
            links=[t.link for t in agg.terms],
            coefs=torch.tensor([float(t.coef) for t in agg.terms], dtype=dtype, device=device),
        ))
    return out


class _Link(nn.Module):
    """A term's link as a module, so hook-based DeepLIFT applies the rescale rule to it."""

    def __init__(self, fn: Link):
        super().__init__()
        self.fn = fn

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.fn(s)
