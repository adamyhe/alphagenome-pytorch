"""Blockwise DeepLIFT/SHAP for AlphaGenome genome-track heads.

AlphaGenome-specific, unlike ``_paired_rules`` and ``_shared_forward``. It
computes the multipliers tangermeme computes (same rules, same references),
without keeping an autograd graph over the convolutional parts of the model:

* **Encoder, decoder, output embedders and head** have a hand-written
  backward. Each block's activations are recomputed from its stored input, for
  the sequence (one row) and a chunk of ``k`` references, and turned into
  DeepLIFT multipliers (tangermeme's secant rescale rule for GELU, softplus and
  the experimental-space scaling; its max-pool routing rule for the encoder's
  pooling). The backward then runs over ``k * T`` gradient rows at once (``T``
  tracks). Only block boundaries (the U-Net skips) are kept between blocks.
* **The decoder is cropped.** Its receptive field is local (under 800 bp), so
  for a 1 bp window only the window plus ``decoder_margin`` bp of the decoder
  is computed, forward and backward. The encoder and tower always run in full.
* **The transformer tower** runs on autograd with :class:`PairedRules`. The
  sequence's one-row tower graph is kept for the whole call; each reference's
  rule-site activations are recorded by rerunning the tower on that reference
  (the tower is a small share of the compute), and one backward per
  (reference, track) runs over the kept graph.

Every rule uses the sequence's gradient for both halves of a pair, which is
exact because tangermeme's rules give both halves the same multiplier (see
``_paired_rules``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...convolutions import StandardizedConv1d
from ...heads import predictions_scaling
from ._paired_rules import PairedRules
from ._shared_forward import SharedForwardResult, tangermeme_references

# tangermeme's rescale rule falls back to the plain gradient below this |delta_in|.
_RESCALE_EPS = 1e-6


# -- rules on tensors -------------------------------------------------------


def _secant(f, a_x: torch.Tensor, a_r: torch.Tensor) -> torch.Tensor:
    """Rescale multipliers ``(k, ...)`` of elementwise ``f`` between ``a_x`` (1 row) and ``a_r`` (k rows).

    ``(f(a_x) - f(a_r)) / (a_x - a_r)``, or ``f'(a_x)`` where
    ``|a_x - a_r| < 1e-6``, as in tangermeme's ``_rescale``.
    """
    d_in = a_x - a_r
    m = f(a_x) - f(a_r)
    m.div_(d_in)
    small = d_in.abs() < _RESCALE_EPS
    if bool(small.any()):
        with torch.enable_grad():
            a = a_x.detach().requires_grad_(True)
            (deriv,) = torch.autograd.grad(f(a).sum(), a)
        m = torch.where(small, deriv, m)
    return m


# Elements per channel chunk (summed over rows) when building multipliers, so
# their temporaries stay ~100 MiB however long the input is.
_CHUNK_ELEMS = 1 << 25


def _channel_chunks(n_channels: int, rows: int, length: int):
    step = max(1, _CHUNK_ELEMS // max(1, rows * length))
    return [slice(c, min(n_channels, c + step)) for c in range(0, n_channels, step)]


def _chunked_mult(act, pre, n_channels: int, rows: int, length: int) -> torch.Tensor:
    """Multipliers ``(k, C, L)`` of ``act``, built channel chunk by chunk.

    ``pre(cs)`` returns the pre-activations ``(a_x, a_r)`` of channels ``cs``.
    """
    m = None
    for cs in _channel_chunks(n_channels, rows, length):
        a_x, a_r = pre(cs)
        mc = _secant(act, a_x, a_r)
        if m is None:
            m = mc.new_empty((mc.shape[0], n_channels, mc.shape[-1]))
        m[:, cs] = mc
        del a_x, a_r, mc
    return m


def _norm_slice(norm, x: torch.Tensor, cs: slice) -> torch.Tensor:
    """``norm(x)[:, cs]`` for an NCL eval-mode ``RMSBatchNorm``, computing only ``cs``."""
    inv = norm.weight * torch.rsqrt(norm.running_var + norm.eps).to(x.dtype)
    return x[:, cs] * inv[cs].view(-1, 1) + norm.bias[cs].view(-1, 1)


def _maxpool_vjp(pool, in_x: torch.Tensor, in_r: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """tangermeme's ``_maxpool`` rule through ``layers.Pool1d``.

    ``in_x`` (1, C, L), ``in_r`` (k, C, L), ``g`` (k, T, C, Lo) -> (k, T, C, L).

    For non-overlapping windows (kernel == stride) the rule is, at slot ``j``
    of each window, ``g`` times ``(dx [x wins at j] + dr [r wins at j]) /
    (x_j - r_j)``, with ``dx = max(o_x, o_r) - o_r`` and
    ``dr = o_x - max(o_x, o_r)``, falling back to the plain gradient
    (``[x wins at j]``) where ``x_j == r_j``. The multiplier is built at ``k``
    rows and the ``k * T``-row output is written once.
    """
    mp = pool.max_pool
    ks, st = mp.kernel_size, mp.stride
    ks = ks if isinstance(ks, int) else ks[0]
    st = st if isinstance(st, int) else st[0]
    pd = mp.padding if isinstance(mp.padding, int) else mp.padding[0]
    dl = mp.dilation if isinstance(mp.dilation, int) else mp.dilation[0]
    if ks != st or pd != 0 or dl != 1 or mp.ceil_mode:
        raise NotImplementedError("Blockwise max-pool rule needs kernel == stride, no padding.")

    px, pr = pool.pad(in_x), pool.pad(in_r)
    C, L_pad = px.shape[1], px.shape[-1]
    Lo = L_pad // st
    out = g.new_empty((*g.shape[:-1], Lo, st))
    for cs in _channel_chunks(C, 3 * pr.shape[0], L_pad):
        ox, ix = F.max_pool1d(px[:, cs], ks, st, return_indices=True)
        orr, ir = F.max_pool1d(pr[:, cs], ks, st, return_indices=True)
        off_x, off_r = ix % st, ir % st
        del ix, ir
        mx = torch.maximum(ox, orr)
        dx, dr = mx - orr, ox - mx
        del mx, ox, orr
        d_in = (px[:, cs] - pr[:, cs]).unflatten(-1, (Lo, st))  # (k, c, Lo, st)
        for j in range(st):
            wx = off_x == j
            dj = d_in[..., j]
            coef = dx * wx + dr * (off_r == j)
            zero = dj == 0
            coef.div_(dj.masked_fill(zero, 1))
            coef = torch.where(zero, wx.to(coef.dtype), coef)
            torch.mul(g[:, :, cs], coef.unsqueeze(1), out=out[:, :, cs, :, j])
            del coef, zero, wx, dj
        del d_in, dx, dr, off_x, off_r
    out = out.flatten(-2)
    if L_pad != in_x.shape[-1]:
        left = (L_pad - in_x.shape[-1]) // 2
        out = out[..., left:left + in_x.shape[-1]]
    return out


# -- linear pieces ------------------------------------------------------------


def _conv_vjp(conv: nn.Conv1d, g: torch.Tensor) -> torch.Tensor:
    """Input gradient of a stride-1, odd-kernel, 'same'-padded conv. ``g`` is (..., C_out, L)."""
    k = conv.kernel_size[0]
    if conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1 or k % 2 == 0:
        raise NotImplementedError("Blockwise conv backward needs stride 1, odd kernel, no groups.")
    if isinstance(conv, StandardizedConv1d):
        w = conv.standardized_weight()
        pad = (k - 1) // 2 if conv.pad_mode == "same" else 0
    else:
        w = conv.weight
        pad = (k - 1) // 2 if conv.padding == "same" else conv.padding[0]
    if pad != (k - 1) // 2:
        raise NotImplementedError("Blockwise conv backward needs length-preserving padding.")
    lead = g.shape[:-2]
    out = F.conv_transpose1d(g.reshape(-1, *g.shape[-2:]), w.to(g.dtype), padding=pad)
    return out.view(*lead, *out.shape[-2:])


def _norm_scale(norm, dtype) -> torch.Tensor:
    """Per-channel multiplier ``(C, 1)`` of an NCL eval-mode ``RMSBatchNorm``."""
    inv = norm.weight * torch.rsqrt(norm.running_var + norm.eps).to(dtype)
    return inv.view(-1, 1)


def _convblock_vjp(block, m: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """``ConvBlock`` (norm -> act -> conv) backward with act multipliers ``m`` (k, C_in, L)."""
    gb = _conv_vjp(block.conv, g)
    gb.mul_(m.unsqueeze(1))
    gb.mul_(_norm_scale(block.norm, gb.dtype))
    return gb


def _convblock_mult(block, in_x, in_r):
    """Multipliers ``(k, C_in, L)`` of a ``ConvBlock``'s activation."""
    return _chunked_mult(
        block.act, lambda cs: (_norm_slice(block.norm, in_x, cs), _norm_slice(block.norm, in_r, cs)),
        in_x.shape[1], in_x.shape[0] + in_r.shape[0], in_x.shape[-1])


def _pad_channels_like(g_small: torch.Tensor, channels: int) -> torch.Tensor:
    return F.pad(g_small, (0, 0, 0, channels - g_small.shape[-2]))


# -- blocks ---------------------------------------------------------------------


def _dna_embedder_vjp(emb, x, r, g):
    conv1, norm = emb.conv1, emb.block.norm

    def pre(cs):
        w, b = conv1.weight[cs], conv1.bias[cs]
        cx = F.conv1d(x, w, b, padding=conv1.padding)
        cr = F.conv1d(r, w, b, padding=conv1.padding)
        inv = norm.weight * torch.rsqrt(norm.running_var + norm.eps).to(x.dtype)
        shift = norm.bias[cs].view(-1, 1)
        return cx * inv[cs].view(-1, 1) + shift, cr * inv[cs].view(-1, 1) + shift

    m = _chunked_mult(emb.block.act, pre, conv1.out_channels, 1 + r.shape[0], x.shape[-1])
    g.add_(_convblock_vjp(emb.block, m, g))  # g is owned here; caller discards it
    del m
    return _conv_vjp(emb.conv1, g)


def _down_vjp(block, in_x, in_r, g):
    """``DownResBlock``: out = b1(x) + pad(x); y = out + b2(out)."""
    b1, b2 = block.block1, block.block2
    m1 = _convblock_mult(b1, in_x, in_r)
    c_in = in_x.shape[1]
    out_x = b1(in_x)
    out_x[:, :c_in] += in_x
    out_r = b1(in_r)
    out_r[:, :c_in] += in_r
    m2 = _convblock_mult(b2, out_x, out_r)
    del out_x, out_r
    g.add_(_convblock_vjp(b2, m2, g))  # in place: the caller discards g
    del m2
    g_in = _convblock_vjp(b1, m1, g)
    g_in += g[..., :c_in, :]
    return g_in


def _up_vjp(block, in_x, skip_x, in_r, skip_r, g):
    """``UpResBlock``: o = ci(x) + x[:Cs]; v = rep2(o)*s + pw(skip); y = v + co(v)."""
    ci, pw, co = block.conv_in, block.pointwise, block.conv_out
    rs = block.residual_scale
    c_s = skip_x.shape[1]

    def _v(x, skip):
        o = ci(x) + x[:, :c_s]
        return torch.repeat_interleave(o, 2, dim=2) * rs + pw(skip)

    m_in = _convblock_mult(ci, in_x, in_r)
    m_pw = _convblock_mult(pw, skip_x, skip_r)
    v_x, v_r = _v(in_x, skip_x), _v(in_r, skip_r)
    m_out = _convblock_mult(co, v_x, v_r)
    del v_x, v_r

    g_v = g.add_(_convblock_vjp(co, m_out, g))  # in place: the caller discards g
    del m_out
    g_skip = _convblock_vjp(pw, m_pw, g_v)
    del m_pw
    g_o = g_v.unflatten(-1, (g_v.shape[-1] // 2, 2)).sum(-1) * rs
    del g_v
    g_in = _convblock_vjp(ci, m_in, g_o)
    g_in[..., :c_s, :] += g_o
    return g_in, g_skip


def _embedder_pre(emb, x, org_index: int, skip=None):
    """Pre-activation of ``OutputEmbedder`` (everything before its GELU)."""
    h = emb.project_in(x)
    if skip is not None and emb.project_skip is not None:
        s = emb.project_skip(skip)
        factor = h.shape[2] // s.shape[2]
        if factor > 1:
            s = s.repeat_interleave(factor, dim=2)
        h = h + s
    h = emb.norm(h)
    org = torch.full((x.shape[0],), org_index, dtype=torch.long, device=x.device)
    return h + emb.organism_embed(org).unsqueeze(2)


def _embedder_vjp(emb, a_x, a_r, g, skip_factor: int = 0):
    """``OutputEmbedder`` backward. Returns (grad wrt x, grad wrt skip or None)."""
    g = g * _secant(emb.act, a_x, a_r).unsqueeze(1)
    g.mul_(_norm_scale(emb.norm, g.dtype))
    g_x = _conv_vjp(emb.project_in, g)
    g_skip = None
    if skip_factor:
        g_s = g.unflatten(-1, (g.shape[-1] // skip_factor, skip_factor)).sum(-1)
        g_skip = _conv_vjp(emb.project_skip, g_s)
    return g_x, g_skip


# -- plan and states ------------------------------------------------------------------


@dataclass
class _Plan:
    model: nn.Module
    head: nn.Module
    org: int
    slot: int
    res: int
    lo: int  # window in head bins
    hi: int
    tracks: list
    reduction: str
    experimental: bool
    L: int
    trunk_bin: int
    crop: Optional[tuple] = None  # (c0, c1) bp, 1 bp heads
    win: Optional[tuple] = None   # (w0, w1) bp, window aligned to trunk bins

    @property
    def n_bins(self) -> int:
        return self.hi - self.lo


def _make_plan(model, *, organism_index, output_type, resolution, target_slice,
               track_indices, reduction, experimental_space, length, decoder_margin):
    head = model.heads[output_type]
    slot = 0 if head.num_organisms == 1 else int(organism_index)
    tb = model.encoder.pool.stride ** (len(model.encoder.down_blocks) + 1)
    n_bins = length // resolution
    lo, hi, step = target_slice.indices(n_bins)
    if step != 1 or hi <= lo:
        raise ValueError(f"target_slice must be a non-empty unit-step slice; got {target_slice}.")
    plan = _Plan(model=model, head=head, org=int(organism_index), slot=slot,
                 res=int(resolution), lo=lo, hi=hi, tracks=[int(t) for t in track_indices],
                 reduction=reduction, experimental=experimental_space, L=length, trunk_bin=tb)
    if resolution == 1:
        w0, w1 = (lo // tb) * tb, -(-hi // tb) * tb
        c0 = max(0, ((lo - decoder_margin) // tb) * tb)
        c1 = min(length, -(-(hi + decoder_margin) // tb) * tb)
        plan.win, plan.crop = (w0, w1), (c0, c1)
    elif resolution != tb:
        raise NotImplementedError(f"Resolution {resolution} is not supported by the blockwise engine.")
    return plan


def _encode(enc, x):
    """Encoder forward keeping only the skips. Returns (skips, trunk), NCL."""
    skips = []
    h = enc.dna_embedder(x)
    skips.append(h)
    h = enc.pool(h)
    for block in enc.down_blocks:
        h = block(h)
        skips.append(h)
        h = enc.pool(h)
    return skips, h


def _skip_bins(enc):
    return [1] + list(enc.bin_sizes)


def _tower_input(model, trunk, org_index):
    org = torch.full((trunk.shape[0],), org_index, dtype=torch.long, device=trunk.device)
    return trunk.transpose(1, 2) + model.organism_embed(org).unsqueeze(1)


def _tower(model, tin):
    out, _pair = model.tower(tin, compute_dtype=model.dtype_policy.compute_dtype)
    return out


def _head_states(plan, e):
    """Head forward on window embeddings ``e`` (B, C, W). Returns (z, u, y)."""
    head = plan.head
    res = str(plan.res)
    w = head.convs[res].weight[plan.slot, plan.tracks]  # (T, C)
    b = head.convs[res].bias[plan.slot, plan.tracks]
    z = torch.einsum("bcs,tc->bts", e.float(), w.float()).to(e.dtype) + b[:, None]
    c = F.softplus(head.residual_scales[res][plan.slot, plan.tracks])
    u = head.softplus(z) * c[:, None]
    y = _experimental(plan, u) if plan.experimental else u
    y = y.sum(-1) if plan.reduction == "sum" else y.mean(-1)
    return z, u, y


def _experimental(plan, u):
    means = plan.head.track_means[plan.slot, plan.tracks].to(u.dtype)
    return predictions_scaling(u, track_means=means.expand(u.shape[0], -1),
                               resolution=plan.res, apply_squashing=plan.head.apply_squashing,
                               channels_last=False)


def _post_tower(plan, trunk_ncl, skips):
    """Decoder (cropped), output embedders and head at the window."""
    m = plan.model
    st = {}
    tb = plan.trunk_bin
    if plan.res == 1:
        c0, c1 = plan.crop
        w0, w1 = plan.win
        h = trunk_ncl[..., c0 // tb:c1 // tb]
        bins = dict(zip(_skip_bins(m.encoder), skips))
        dec_in = []
        for b, block in zip(m.decoder.bin_sizes, m.decoder.up_blocks):
            skip = bins[b][..., c0 // b:c1 // b]
            dec_in.append((h, skip))
            h = block(h, skip)
        st["dec_in"] = dec_in
        a128 = _embedder_pre(m.embedder_128bp, trunk_ncl[..., w0 // tb:w1 // tb], plan.org)
        e128 = m.embedder_128bp.act(a128)
        a1 = _embedder_pre(m.embedder_1bp, h[..., w0 - c0:w1 - c0], plan.org, skip=e128)
        del h, e128
        e = m.embedder_1bp.act(a1)[..., plan.lo - w0:plan.hi - w0]
        st["a1"] = a1
    else:
        a128 = _embedder_pre(m.embedder_128bp, trunk_ncl[..., plan.lo:plan.hi], plan.org)
        e = m.embedder_128bp.act(a128)
    st["a128"] = a128
    st["z"], st["u"], st["y"] = _head_states(plan, e)
    return st


# -- backward -------------------------------------------------------------------------


def _post_tower_vjp(plan, sx, sr, ts, S_trunk):
    """Head, embedders and decoder backward for tracks ``ts`` (indices into plan.tracks).

    Returns (grad wrt tower output, NCL (k, T, C, S); skip grads {bin: (offset, grad)}).
    """
    m = plan.model
    head = plan.head
    res = str(plan.res)
    tracks = [plan.tracks[i] for i in ts]
    zx, zr = sx["z"][:, ts], sr["z"][:, ts]
    ux, ur = sx["u"][:, ts], sr["u"][:, ts]
    k, T = zr.shape[0], len(ts)

    G = torch.full_like(zr, 1.0 if plan.reduction == "sum" else 1.0 / plan.n_bins)
    if plan.experimental:
        means = head.track_means[plan.slot, tracks].to(ux.dtype)

        def S(u):
            return predictions_scaling(u, track_means=means.expand(u.shape[0], -1),
                                       resolution=plan.res, apply_squashing=head.apply_squashing,
                                       channels_last=False)

        G = G * _secant(S, ux, ur)
    c = F.softplus(head.residual_scales[res][plan.slot, tracks])
    G = G * c[:, None] * _secant(head.softplus, zx, zr)  # (k, T, W)
    w = head.convs[res].weight[plan.slot, tracks].to(G.dtype)  # (T, C)
    g_e = torch.einsum("tc,ktw->ktcw", w, G)

    C_trunk = m.embedder_128bp.project_in.in_channels
    g_trunk = G.new_zeros((k, T, C_trunk, S_trunk))
    skip_grads = {}
    tb = plan.trunk_bin
    if plan.res == 1:
        c0, c1 = plan.crop
        w0, w1 = plan.win
        g_a = g_e.new_zeros((k, T, g_e.shape[2], w1 - w0))
        g_a[..., plan.lo - w0:plan.hi - w0] = g_e
        del g_e
        g_d, g_e128 = _embedder_vjp(m.embedder_1bp, sx["a1"], sr["a1"], g_a, skip_factor=tb)
        del g_a
        g_t, _ = _embedder_vjp(m.embedder_128bp, sx["a128"], sr["a128"], g_e128)
        g_trunk[..., w0 // tb:w1 // tb] += g_t
        del g_t, g_e128
        g = g_d.new_zeros((k, T, g_d.shape[2], c1 - c0))
        g[..., w0 - c0:w1 - c0] = g_d
        del g_d
        blocks = list(zip(m.decoder.bin_sizes, m.decoder.up_blocks))
        for i in reversed(range(len(blocks))):
            b, block = blocks[i]
            (in_x, skip_x), (in_r, skip_r) = sx["dec_in"][i], sr["dec_in"][i]
            g, g_skip = _up_vjp(block, in_x, skip_x, in_r, skip_r, g)
            skip_grads[b] = (c0 // b, g_skip)
        g_trunk[..., c0 // tb:c1 // tb] += g
    else:
        g_t, _ = _embedder_vjp(m.embedder_128bp, sx["a128"], sr["a128"], g_e)
        g_trunk[..., plan.lo:plan.hi] += g_t
    return g_trunk, skip_grads


def _encoder_vjp(enc, x, r, skips_x, skips_r, g, skip_grads):
    """Encoder backward from the trunk (k, T, C, S) to the one-hot input (k, T, 4, L)."""
    bins = _skip_bins(enc)

    def _add_skip(g, level):
        b = bins[level]
        if b in skip_grads:
            off, sg = skip_grads.pop(b)
            g[..., off:off + sg.shape[-1]] += sg
        return g

    g = _maxpool_vjp(enc.pool, skips_x[-1], skips_r[-1], g)
    for i in reversed(range(len(enc.down_blocks))):
        g = _add_skip(g, i + 1)
        in_x, in_r = enc.pool(skips_x[i]), enc.pool(skips_r[i])
        g = _down_vjp(enc.down_blocks[i], in_x, in_r, g)
        del in_x, in_r
        g = _maxpool_vjp(enc.pool, skips_x[i], skips_r[i], g)
    g = _add_skip(g, 0)
    return _dna_embedder_vjp(enc.dna_embedder, x, r, g)


# -- driver ---------------------------------------------------------------------------


def blockwise_deep_lift_shap(
    model: nn.Module,
    X: torch.Tensor,
    *,
    organism_index: int,
    output_type: str,
    resolution: int,
    target_slice: slice,
    track_indices: Sequence[int],
    nonlinear_ops: dict,
    reduction: str = "sum",
    experimental_space: bool = True,
    references: Optional[torch.Tensor] = None,
    n_shuffles: int = 20,
    random_state: Optional[int] = None,
    hypothetical: bool = False,
    raw_outputs: bool = False,
    ref_batch_size: int = 4,
    track_batch_size: Optional[int] = None,
    decoder_margin: int = 2048,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
    relative_warning_threshold: float = 1e-3,
) -> SharedForwardResult:
    """DeepLIFT/SHAP for one AlphaGenome genome-track head, blockwise.

    Args:
        model: AlphaGenome, inside ``attributable(model)``.
        X: ``(N, 4, L)`` one-hot sequences.
        organism_index, output_type, resolution, target_slice, track_indices,
            reduction, experimental_space: As for ``TrackTarget``.
        nonlinear_ops: Full rule table, used for the transformer tower.
        references, n_shuffles, random_state, hypothetical, raw_outputs: As
            for :func:`shared_forward_deep_lift_shap`.
        ref_batch_size: References per chunk (``k``). Memory grows with
            ``k`` (U-Net skips per reference) and ``k * track_batch_size``
            (gradient rows).
        track_batch_size: Tracks per backward (default: all). Each further
            batch of tracks recomputes the reference chunk's block activations.
        decoder_margin: bp of decoder computed on each side of a 1 bp window.
            Must exceed the decoder's receptive radius (762 bp for AlphaGenome).
        device, dtype: Where and in what precision to run.
        relative_warning_threshold: As for :func:`shared_forward_deep_lift_shap`.
    """
    from tangermeme.deep_lift_shap import hypothetical_attributions

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    if references is None:
        references = tangermeme_references(X, n_shuffles, random_state)
    N, K = references.shape[:2]
    L = X.shape[-1]
    plan = _make_plan(model, organism_index=organism_index, output_type=output_type,
                      resolution=resolution, target_slice=target_slice,
                      track_indices=track_indices, reduction=reduction,
                      experimental_space=experimental_space, length=L,
                      decoder_margin=decoder_margin)
    T = len(plan.tracks)
    tbs = T if track_batch_size is None else max(1, int(track_batch_size))
    track_chunks = [list(range(i, min(T, i + tbs))) for i in range(0, T, tbs)]
    kbs = max(1, int(ref_batch_size))

    mult = torch.zeros((T, N, K, *X.shape[1:]), dtype=dtype) if raw_outputs else None
    hyp_sum = torch.zeros((T, N, *X.shape[1:]), dtype=dtype)
    diffs = torch.zeros((T, N, K), dtype=torch.float64)
    deltas = torch.zeros((T, N, K), dtype=torch.float64)
    enc = model.encoder

    with PairedRules(model.tower, nonlinear_ops) as rules, torch.no_grad():
        for i in range(N):
            x = X[i:i + 1].to(device=device, dtype=dtype)
            skips_x, trunk_x = _encode(enc, x)
            tin_x = _tower_input(model, trunk_x, plan.org).requires_grad_(True)
            with torch.enable_grad(), rules.recording("x"):
                tout_x = _tower(model, tin_x)
            sx = _post_tower(plan, tout_x.detach().transpose(1, 2), skips_x)
            S_trunk = tin_x.shape[1]

            for k0 in range(0, K, kbs):
                r = references[i, k0:k0 + kbs].to(device=device, dtype=dtype)
                k = r.shape[0]
                skips_r, trunk_r = _encode(enc, r)
                tin_r = _tower_input(model, trunk_r, plan.org)
                del trunk_r
                sr = _post_tower(plan, _tower(model, tin_r).transpose(1, 2), skips_r)
                y_diff = (sx["y"] - sr["y"]).double()  # (k, T)

                for ts in track_chunks:
                    g_trunk, skip_grads = _post_tower_vjp(plan, sx, sr, ts, S_trunk)
                    g_tin = torch.empty((k, len(ts), S_trunk, g_trunk.shape[2]),
                                        dtype=dtype, device=device)
                    for j in range(k):
                        with rules.recording("ref"):
                            _tower(model, tin_r[j:j + 1])
                        for t in range(len(ts)):
                            with torch.enable_grad():
                                (gi,) = torch.autograd.grad(
                                    tout_x, tin_x, grad_outputs=g_trunk[j, t].T.unsqueeze(0),
                                    retain_graph=True)
                            g_tin[j, t] = gi[0]
                        rules.clear_reference()
                    del g_trunk
                    g = _encoder_vjp(enc, x, r, skips_x, skips_r,
                                     g_tin.transpose(2, 3), skip_grads)  # (k, t, 4, L)
                    del g_tin

                    summed = ((x - r).unsqueeze(1) * g).sum(dim=(2, 3)).double()  # (k, t)
                    for tt, t_idx in enumerate(ts):
                        diffs[t_idx, i, k0:k0 + k] = y_diff[:, t_idx].cpu()
                        deltas[t_idx, i, k0:k0 + k] = (y_diff[:, t_idx] - summed[:, tt]).abs().cpu()
                    if raw_outputs:
                        mult[ts, i, k0:k0 + k] = g.transpose(0, 1).cpu()
                    else:
                        flat = g.transpose(0, 1).reshape(-1, *X.shape[1:])
                        xs = x.expand(flat.shape[0], -1, -1)
                        rs = r.unsqueeze(0).expand(len(ts), -1, -1, -1).reshape(flat.shape)
                        hyp = hypothetical_attributions((flat,), (xs,), (rs,))[0]
                        hyp_sum[ts, i] += hyp.view(len(ts), k, *X.shape[1:]).sum(1).cpu()
                    del g
                del skips_r, tin_r, sr
            rules.clear()
            del skips_x, tin_x, tout_x, sx

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
        attributions = hyp_sum / K
        if not hypothetical:
            attributions = attributions * X.to(dtype).unsqueeze(0)
    return SharedForwardResult(attributions=attributions, references=references,
                               convergence_deltas=deltas, output_diffs=diffs)
