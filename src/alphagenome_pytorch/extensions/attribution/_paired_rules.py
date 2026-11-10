"""Apply tangermeme's DeepLIFT rules to sequence rows only.

Model-agnostic (nothing here knows about AlphaGenome), and a candidate to move
into tangermeme.

tangermeme forwards each sequence-reference pair as two rows of one batch and
backpropagates through both. Its rules give the two halves the same multiplier,
so the reference half's gradient equals the sequence half's at every layer and
is never needed separately. `PairedRules` exploits that: the sequence is
forwarded with gradients on its own, each reference is forwarded under
``no_grad`` only to record its activations at the rule sites, and during the
backward each hooked module calls tangermeme's own rule on the pair
``[sequence, reference]`` with the sequence gradient standing in for both
halves, keeping the sequence half of the result.

One forward of the sequence then serves every reference and every output,
through repeated ``torch.autograd.grad(..., retain_graph=True)`` calls; only the
reference whose activations are currently recorded is read.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn


_READS_BOTH = ("input", "output")


class PairedRules:
    """Forward and backward hooks implementing sequence-only DeepLIFT backward passes.

    Usage::

        with PairedRules(model, ops) as rules:
            with rules.recording("x"):
                y = model(x.requires_grad_())          # graph kept, 1 row
            for r in references:
                with rules.recording("ref"), torch.no_grad():
                    y_r = model(r)                     # activations only
                g, = torch.autograd.grad(y[0, t], x, retain_graph=True)

    The sequence and reference forwards must call the hooked modules in the
    same order (true for any model without data-dependent control flow). Rows
    are paired one-to-one: a sequence batch of ``n`` rows pairs with a reference
    batch of ``n`` rows.

    Args:
        model: The model to hook.
        nonlinear_ops: Rule table keyed by exact module type, e.g. from
            ``tangermeme._deep_lift_utils._build_nonlinear_ops``.
        clone: Clone recorded activations. Only needed if the model writes in
            place to a hooked module's input or output after the module ran.
    """

    def __init__(self, model: nn.Module, nonlinear_ops: dict, clone: bool = False):
        self.model = model
        self.ops = nonlinear_ops
        self.clone = clone
        self._handles = []
        self._mode = None
        self._caches = {"x": {}, "ref": {}}
        self._counters: dict[int, int] = {}

    # -- context ----------------------------------------------------------

    def __enter__(self):
        for module in self.model.modules():
            if type(module) in self.ops:
                self._handles.append(module.register_forward_hook(self._f_hook))
                self._handles.append(module.register_full_backward_hook(self._b_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.clear()
        for module in self.model.modules():
            module.__dict__.pop("_paired_idx", None)
            module.__dict__.pop("_staged", None)
        return False

    @contextlib.contextmanager
    def recording(self, which: str):
        """Record the hooked modules' activations of one forward pass as ``which``.

        ``"x"``: the sequence pass, whose graph the backward passes run over.
        ``"ref"``: a reference pass; replaces the previous reference's record.
        """
        if which not in self._caches:
            raise ValueError(f"which must be 'x' or 'ref'; got {which!r}")
        self._caches[which] = {}
        self._counters = {}
        self._mode = which
        try:
            yield
        finally:
            self._mode = None

    def clear(self):
        self._caches = {"x": {}, "ref": {}}
        self._counters = {}

    # -- hooks ------------------------------------------------------------

    def _f_hook(self, module, inputs, output):
        staged = module.__dict__.pop("_staged", {})  # Bilinear operands
        if self._mode is None:
            return
        idx = self._counters.get(id(module), 0)
        self._counters[id(module)] = idx + 1

        reads = getattr(self.ops[type(module)], "_reads", _READS_BOTH)
        keep = (lambda t: t.detach().clone()) if self.clone else (lambda t: t.detach())
        entry = {name: keep(t) for name, t in staged.items()}
        if "input" in reads:
            entry["input"] = keep(inputs[0])
        if "output" in reads:
            entry["output"] = keep(output)
        self._caches[self._mode][(id(module), idx)] = entry

        if self._mode == "x" and output.requires_grad:
            def _tag(grad, module=module, idx=idx):
                module._paired_idx = idx
            output.register_hook(_tag)

    def _b_hook(self, module, grad_input, grad_output):
        idx = module.__dict__.pop("_paired_idx", None)
        if idx is None:
            return grad_input
        key = (id(module), idx)
        x_entry, r_entry = self._caches["x"][key], self._caches["ref"].get(key)
        if r_entry is None:
            raise RuntimeError(
                "No reference activations recorded for a hooked module; run a "
                "reference forward inside rules.recording('ref') before backward."
            )

        for name, xt in x_entry.items():
            setattr(module, name, torch.cat([xt, r_entry[name].to(xt.dtype)]))
        try:
            pair = lambda g: None if g is None else torch.cat([g, g])  # noqa: E731
            multipliers = self.ops[type(module)](
                module, tuple(pair(g) for g in grad_input),
                tuple(pair(g) for g in grad_output),
            )
        finally:
            for name in x_entry:
                delattr(module, name)

        return tuple(
            None if m is None or g is None else m[: g.shape[0]].to(g.dtype)
            for m, g in zip(multipliers, grad_input)
        )
