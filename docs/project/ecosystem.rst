The BrainX Ecosystem
====================

``brainpy.state`` is the point-neuron modeling layer of the `BrainX ecosystem
<https://brainx.chaobrain.com/>`_ — a family of composable, JAX-based packages
for brain modeling and brain-inspired computing. Each package owns one concern,
and ``brainpy.state`` builds on them: state management, physical units,
event-driven operators, training/analysis tooling, and online learning.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: brainstate
      :link: https://github.com/chaobrain/brainstate

      State management for JAX-based brain modeling. Provides the ``State``
      abstraction, ``init_all_states``, ``environ.context`` (``dt``/``t``), the
      ``brainstate.nn`` modules and connectivity, and the ``brainstate.transform``
      primitives (``jit`` / ``for_loop`` / ``scan`` / ``checkpointed_*``) that
      drive every ``brainpy.state`` model.

   .. grid-item-card:: brainunit
      :link: https://github.com/chaobrain/brainunit

      Physical units for neuroscience. Every ``brainpy.state`` parameter carries a
      unit (``mV``, ``ms``, ``nS``, …) so dimensional errors are caught at
      construction time rather than producing silently wrong results.

   .. grid-item-card:: brainevent
      :link: https://github.com/chaobrain/brainevent

      Event-driven sparse operators. Supplies the sparse, spike-event linear
      algebra behind large recurrent and balanced networks, keeping memory and
      compute proportional to the events that actually occur.

   .. grid-item-card:: braintools
      :link: https://github.com/chaobrain/braintools

      Surrogate gradients, initializers, optimizers, metrics, and visualization.
      Home of ``braintools.surrogate`` (passed as ``spk_fun=``),
      ``braintools.init``, ``braintools.optim``, ``braintools.metric``, and
      ``braintools.visualize`` used throughout the training examples.

   .. grid-item-card:: braintrace
      :link: https://github.com/chaobrain/braintrace

      Linear-memory online learning for spiking networks. The published engine
      (BrainScale preprint → BrainTrace) that reformulates real-time recurrent
      learning so memory scales **linearly** with the number of neurons,
      exploiting the same neuron-aligned synaptic state as AlignPre/AlignPost.

Installing the ecosystem
------------------------

The whole stack installs together::

   pip install BrainX

See :doc:`/get-started/installation` for per-backend (CPU / CUDA / TPU) options.

See Also
--------

- :doc:`/concepts/online-learning` — how AlignPre/AlignPost enables the
  linear-memory online learning that ``braintrace`` implements.
- :doc:`/concepts/state-paradigm` — the ``brainstate`` State paradigm and
  ``transform`` primitives in depth.
- :doc:`/project/citing` — how to cite the framework.
