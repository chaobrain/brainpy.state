BrainPy-style Modeling Guide
============================

This guide provides an in-depth tour of the **BrainPy-style modeling API** —
the high-level, composable neuron, synapse, projection, and readout building
blocks shipped under ``brainpy.state``. The BrainPy-style API is the recommended
entry point for building and training spiking neural networks with this library.

.. note::

   This guide covers the **BrainPy-style API only**. For NEST-compatible models
   (``iaf_psc_alpha``, ``aeif_cond_exp``, STDP synapses, etc.), see the
   :doc:`/nest-status/index` page — those models are under active development
   and have their own usage notes.

What this guide covers
----------------------

- **Architecture**: How ``brainpy.state`` composes ``Dynamics``, ``Neuron``,
  ``Synapse``, projections, and readouts on top of `brainstate`'s state
  management.
- **Neurons**: Building blocks for spiking computation — LIF, ALIF, AdEx, HH,
  Izhikevich, and their refractory variants — and how state variables are
  represented.
- **Synapses**: Exponential, alpha, AMPA, GABA, and NMDA dynamics; how
  conductance-based (COBA), current-based (CUBA), and magnesium-block outputs
  attach to a postsynaptic population.
- **Projections**: Network-level wiring — ``AlignPostProj``, delta
  projections, and short-term plasticity hooks — for connecting populations.


Why these concepts matter
-------------------------

Working through this guide will help you:

- Design complex spiking networks with clean, composable code.
- Use surrogate gradients to train BrainPy-style neurons end-to-end.
- Manage simulation state and physical units (``saiunit``) idiomatically.
- Integrate cleanly with the rest of the `BrainX ecosystem
  <https://brainx.chaobrain.com/>`_.

The pages build on one another, so we recommend reading them in order if
you're new to ``brainpy.state``.


.. toctree::
   :hidden:
   :maxdepth: 1

   architecture.ipynb
   neurons.ipynb
   synapses.ipynb
   projections.ipynb
