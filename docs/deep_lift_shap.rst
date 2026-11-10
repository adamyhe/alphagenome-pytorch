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

``deep_lift_shap`` returns the same ``AttributionResult`` (``base_matrix``) as
``gradient_x_input``, for a single sequence and the attributed window only.
The full hypothetical contributions are in ``metadata["hypothetical"]``.

What is attributed
------------------

Output ``t`` is the ``reduction`` (``"sum"`` or ``"mean"``) over the bins in
``target_slice`` of track ``track_indices[t]``, in experimental space (what
``predict`` returns) by default. Pass ``experimental_space=False`` to
attribute the model-space prediction instead. ``"max"`` is not supported
because it has no DeepLIFT rule. Each track is a separate tangermeme call. With
a fixed ``random_state``, every track uses the same dinucleotide-shuffled
references.

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

``attributable`` makes four changes and undoes them on exit:

* ``layers.Bilinear`` modules stage their operands for the bilinear rule.
* In-place RoPE is turned off. Module backward hooks reject in-place writes.
* Gradient checkpointing is turned off. Recomputing a segment would rerun the
  forward hooks.
* TF32 is turned off. With TF32 on, the prediction differences themselves are
  off by about 1e-3 relative.

Rules
-----

==========================================  ===================================
Module                                      tangermeme rule
==========================================  ===================================
``layers.GELU``, ``nn.GELU``, ``nn.ReLU``,  rescale
``nn.Tanh`` (logit soft-cap),
``nn.Softplus`` (track heads),
experimental-space scaling
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

This path keeps tangermeme's full autograd graph for ``2 * batch_size`` rows.
Measured peak GPU memory with ``batch_size=1`` on the pretrained model:

=================  ==================  ============
Input length       Head @ resolution   Peak memory
=================  ==================  ============
4,096 bp           DNase @ 128 bp      3.5 GiB
4,096 bp           DNase @ 1 bp        5.9 GiB
16,384 bp          RNA-seq @ 1 bp      12.2 GiB
32,768 bp          DNase @ 128 bp      9.0 GiB
=================  ==================  ============

Memory grows roughly linearly with length. Extrapolating linearly (not
measured), the full 131,072 bp context needs about 30 GiB for a 128 bp head
and about 80 GiB for a 1 bp head.
