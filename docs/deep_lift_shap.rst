DeepLIFT/SHAP attribution
=========================

``alphagenome_pytorch.extensions.attribution`` computes DeepLIFT/SHAP
attributions for any genome-track head (ATAC, DNase, PRO-cap, CAGE, RNA-seq,
ChIP-TF, ChIP-Histone) through
`tangermeme <https://github.com/jmschrei/tangermeme>`_'s ``deep_lift_shap``.
It needs tangermeme 1.6.0 or later:

.. code-block:: bash

   pip install "alphagenome-pytorch[attribution]"

Every nonlinearity between the input and a track head has a closed-form
DeepLIFT rule, so attributions sum to the difference between the prediction
for the sequence and the prediction for its reference, up to floating-point
error. On the pretrained model the gap is about 1e-6 of the prediction
difference.

Quick start
-----------

.. code-block:: python

   from alphagenome_pytorch import AlphaGenome
   from alphagenome_pytorch.extensions.attribution import deep_lift_shap_tracks

   model = AlphaGenome.from_pretrained("model.pth", device="cuda").eval()

   # onehot: (N, L, 4), ACGT only (no N positions)
   hyp = deep_lift_shap_tracks(
       model,
       onehot=onehot,
       organism_index=0,
       output_type="dnase",
       resolution=128,
       target_slice=slice(508, 516),   # 128 bp bins to attribute
       track_indices=[0, 17],
       reduction="sum",
       n_shuffles=20,
       random_state=0,
   )
   # hyp: (n_tracks, N, 4, L) hypothetical contributions, the input TF-MoDISco expects.
   # Multiply by the one-hot input to get contributions at the observed bases.

Engines
-------

``deep_lift_shap_tracks`` has three engines that compute the same multipliers
from the same references (seeded as tangermeme seeds them). ``engine="auto"``
(the default) picks ``"blockwise"`` for ``AlphaGenome`` models and
``"shared"`` otherwise.

``engine="blockwise"``
    AlphaGenome-specific. Keeps no autograd graph outside the transformer
    tower. The encoder, decoder, output embedders and head have a hand-written
    backward: each block's activations are recomputed from its stored input,
    for the sequence and a chunk of ``ref_batch_size`` references, turned into
    tangermeme's multipliers (its rescale rule for GELU, softplus and the
    experimental-space scaling, its max-pool rule for the encoder's pooling),
    and applied to all references and tracks in the chunk at once. Only the
    U-Net skips are kept between blocks. For a 1 bp head the decoder runs on
    the window plus 2 kb only, since its receptive field is under 800 bp. The
    tower runs on autograd with tangermeme's rules, over the sequence's
    one-row graph.
``engine="shared"``
    Works with any model. Forwards each sequence once with gradients, and each
    reference once without them, recording only the activations the rules
    read. It then runs one one-row backward pass per (reference, track) over
    the kept graph, calling tangermeme's rules on each sequence-reference pair.
``engine="stock"``
    Calls ``tangermeme.deep_lift_shap.deep_lift_shap`` once per track. This is
    the reference implementation.

The blockwise and shared engines use the sequence half's gradient for both
halves of each pair, which is exact because tangermeme's rules give both
halves the same multiplier. The one exception is tangermeme's fallback where
an activation barely changes (``|x - r| < 1e-6``): stock uses each half's own
derivative there, so the engines differ from stock by about 1e-10 of the
largest attribution in float64, and match it to 1e-13 with the fallback
disabled. In float32 on GPU all three agree to about 1e-5, where different
batch shapes choose different kernels.

On an RTX 3080 (10 GiB; pretrained model, 4 references, 2 tracks, a 1 kb
window for 1 bp heads, ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``):

=============================  ===============  ===============  ===============
Input                          stock            shared           blockwise
=============================  ===============  ===============  ===============
4 kb, DNase @ 128 bp           2.1 s, 2.7 GiB   1.3 s, 2.6 GiB   1.1 s, 2.2 GiB
4 kb, DNase @ 1 bp             2.2 s, 4.1 GiB   1.2 s, 3.6 GiB   1.3 s, 2.4 GiB
16 kb, RNA-seq @ 1 bp          does not fit     3.2 s, 7.2 GiB   1.7 s, 3.6 GiB
32 kb, DNase @ 128 bp          3.9 s, 7.3 GiB   2.0 s, 6.7 GiB   2.0 s, 3.6 GiB
131 kb, DNase @ 128 bp         does not fit     does not fit     7.6 s, 7.3 GiB
131 kb, RNA-seq @ 1 bp         does not fit     does not fit     7.5 s, 7.4 GiB
=============================  ===============  ===============  ===============

Peak memory includes 1.7 GiB of weights. The 131 kb rows use
``ref_batch_size=1``. With 20 references (tangermeme's default) and one track,
131 kb RNA-seq @ 1 bp takes 25 s at a 7.2 GiB peak.

``deep_lift_shap`` returns the same ``AttributionResult`` (``base_matrix``) as
``gradient_x_input``, for a single sequence and the attributed window only.
The full hypothetical contributions are in ``metadata["hypothetical"]``.

What is attributed
------------------

Output ``t`` is the ``reduction`` (``"sum"`` or ``"mean"``) over the bins in
``target_slice`` of track ``track_indices[t]``, in experimental space (what
``predict`` returns) by default. Pass ``experimental_space=False`` to
attribute the model-space prediction instead. ``"max"`` is not supported
because it has no DeepLIFT rule. With a fixed ``random_state``, every track
uses the same dinucleotide-shuffled references.

Aggregate targets
~~~~~~~~~~~~~~~~~

Pass ``aggregates`` to attribute scalar combinations of the window predictions
instead of each track separately. Each ``Aggregate`` is a sum of terms, and
each term is a weighted sum of the window predictions ``y[t, w]`` of
``track_indices``, an optional scalar link (such as ``log``), and a
coefficient::

    f(y) = sum_j coef_j * link_j(reduce_w(sum_t W_j[t, w] * y[t, w]))

Weights are ``(T,)`` (per track) or ``(T, W)`` (per track and window bin).
``reduction="mean"`` divides each weighted sum by the number of bins. The
result has one row per aggregate.

.. code-block:: python

   from alphagenome_pytorch.extensions.attribution import Aggregate

   T = model.heads["dnase"].num_tracks
   k562 = torch.zeros(T); k562[k562_tracks] = 1 / len(k562_tracks)
   hepg2 = torch.zeros(T); hepg2[hepg2_tracks] = 1 / len(hepg2_tracks)
   attr = deep_lift_shap_tracks(
       model, onehot=onehot, organism_index=0, output_type="dnase",
       resolution=128, target_slice=slice(508, 516),
       track_indices=list(range(T)),
       aggregates=[
           Aggregate.weighted(torch.ones(T), name="total"),
           Aggregate.weighted(k562 - hepg2, name="contrast"),
           Aggregate.log_ratio(k562, hepg2, pseudocount=1.0, name="lfc"),
       ],
   )  # (3, N, 4, L)

This form is exact DeepLIFT: the sums are linear, and the rescale rule on a
scalar link is its exact secant between the sequence and the reference, so
attributions still sum to ``f(x) - f(r)``. A link can be any elementwise,
differentiable torch function (``Term(weights, link=torch.sqrt)``). Its
derivative is used where the sequence and reference sums differ by less than
1e-6, as for every other rescale rule. An aggregate costs about what one
track costs, however many tracks it combines: on the RTX 3080, two
aggregates over all 768 RNA-seq tracks at 131 kb and 1 bp take 6.2 s, against
5.2 s for one track. All three engines support aggregates.

Using tangermeme directly
-------------------------

To call ``tangermeme.deep_lift_shap.deep_lift_shap`` (or ``tangermeme.pisa``)
yourself, wrap the model with ``TrackTarget``, which gives tangermeme the
``(N, 4, L) -> (N, n_tracks)`` model it expects. Run the call inside
``attributable(model)`` and pass ``attribution_ops()``:

.. code-block:: python

   import tangermeme.deep_lift_shap as dls
   from alphagenome_pytorch.extensions.attribution import (
       TrackTarget, attributable, attribution_ops,
   )

   target = TrackTarget(model, organism_index=0, output_type="rna_seq",
                        resolution=1, target_slice=slice(60000, 61000),
                        track_indices=[3])
   with attributable(model):
       attr = dls.deep_lift_shap(target, X, n_shuffles=20, random_state=0,
                                 additional_nonlinear_ops=attribution_ops())

``attributable`` makes five changes and undoes them on exit:

* ``layers.Bilinear`` modules stage their operands for the bilinear rule.
* In-place RoPE is turned off. Module backward hooks reject in-place writes.
* Gradient checkpointing is turned off. Recomputing a segment would rerun the
  forward hooks.
* TF32 is turned off. With TF32 on, the prediction differences themselves are
  off by about 1e-3 relative.
* Parameters stop requiring gradients, so autograd does not keep the
  activations that only weight gradients need.

Rules
-----

==========================================  ===================================
Module                                      tangermeme rule
==========================================  ===================================
``layers.GELU``, ``nn.GELU``, ``nn.ReLU``,  rescale
``nn.Tanh`` (logit soft-cap),
``nn.Softplus`` (track heads),
experimental-space scaling, aggregate
links
``nn.MaxPool1d`` (encoder pooling)          max-pool
``nn.Softmax`` (sequence and row            softmax
attention)
``layers.LayerNorm``                        LayerNorm or RMSNorm, chosen by
                                            ``rms_norm``
``layers.Bilinear`` (q·k, attention·v)      bilinear (midpoint product)
==========================================  ===================================

Everything else is linear in the input, so its ordinary gradient is exact:
``RMSBatchNorm`` in eval mode (it uses running statistics),
``StandardizedConv1d``, linear layers, RoPE, residuals, U-Net skips, mean
pooling, and organism embeddings. These modules hold no parameters, so they
don't change checkpoints, and the outputs are bit-identical to calling the
functions directly.

Precision and memory
--------------------

Attribution always runs in float32. A model whose dtype policy computes in
bfloat16 is rejected. tangermeme's absolute convergence warning (1e-3) is off
by default (``warning_threshold=None``). AlphaGenome's outputs, such as sums of
counts over bins, are large enough that float32 rounding alone exceeds it.
tangermeme's relative check (warn above 1% of the prediction differences)
stays on.

With the blockwise engine, memory above the weights at the full 131 kb
context is about:

* 0.9 GiB per row (the sequence, plus each reference in a chunk) for the U-Net
  skips,
* 1.7 GiB for the sequence's tower graph and 1.2 GiB for one reference's
  tower activations,
* about 1.5 GiB of transients for the backward of the 1 bp encoder layers,
  growing with ``ref_batch_size * n_tracks``.

So ``ref_batch_size=1`` fits a 10 GiB card at full length, and larger chunks
(fewer, larger kernel launches) pay off on 24 GB or larger cards. Lower
``track_batch_size`` to bound the gradient rows when attributing many tracks;
each further batch of tracks recomputes the chunk's block activations.

Set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` before starting
Python. The engine allocates many large tensors of different sizes, and
without expandable segments PyTorch's caching allocator can reserve 30% more
than it uses. On Windows (including WSL), a GPU that runs out of dedicated
memory spills into system memory instead of raising an error, and runs
several times slower.

The stock engine keeps tangermeme's full autograd graph over
``2 * batch_size`` rows: 2.7 GiB at 4 kb and 7.3 GiB at 32 kb above, and by
linear extrapolation about 30 GiB for a 128 bp head and 80 GiB for a 1 bp head
at 131 kb.
