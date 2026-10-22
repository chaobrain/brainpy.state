Positioning and scope
=====================

``brainpy.state`` is a **differentiable point-neuron modeling substrate**: the same
neurons, synapses, plasticity rules, and projections run NEST-validated biological
simulation *and* train with gradients or online learning.

What sets it apart
------------------

- **Validated like a simulator.** The NEST-compatible family keeps NEST's parameter
  names and units and is checked against a live NEST install within documented
  tolerance bands.
- **Differentiable like a training library.** Models compile through JAX/XLA and are
  trainable with surrogate gradients or online learning rules (``braintrace``).
- **One graph, written contracts.** Every network is one ``Network`` graph
  (populations, projections, devices, observers) built from models that *declare*
  their ports, states, parameters and spike rules. The same compiled graph is what a
  simulation runs and what a learner differentiates, so NEST parity validates what
  users train.

Who it is for
-------------

1. Computational neuroscientists who want to run NEST models on accelerators, or train
   them.
2. Brain-inspired-computing researchers who need biologically constrained spiking
   networks.
3. Libraries in the `BrainX ecosystem <https://brainx.chaobrain.com/>`_ that build on it,
   such as ``braintrace`` for online learning.

Model scope
-----------

- **Point neurons only.** Multi-compartment and population/mean-field models are out
  of scope.
- **No single-unit rate neurons.** Rate models (``lin_rate``, ``sigmoid_rate``, …) are
  not carried into the new kernel.

Status: kernel redesign in progress
-----------------------------------

The package is being moved onto a new kernel, described in the architecture
specification (``docs/specs/2026-10-21-architecture-review-and-redesign.md``). The first
milestone (K0) has passed its go/no-go gates; hardening (K1a) is next. While this is
in progress:

- **The current 0.1 API is frozen.** It keeps working, but receives no new features or
  fixes; it will be removed when the new kernel replaces it.
- **The new kernel is private** (not yet exported from ``brainpy.state``). It becomes
  public in a later milestone, together with a migration guide.
- **One front-end.** ``Network`` will be the only way to build and run a model. The
  NEST-style ``Simulator`` (``create`` / ``connect`` / ``simulate``) is kept as a thin
  layer that builds a ``Network``.
- **The BrainPy-style API will be deprecated**: standalone neuron modules driven by hand
  inside a ``brainstate`` module, and the AlignPre/AlignPost projections. Their models
  are ported onto the new contracts and used through ``Network``; the memory-efficient
  post-aligned merging becomes an automatic compiler pass.
- **Performance.** CPU performance of the new kernel has been measured against the
  current simulator on one benchmark (Brunel network); GPU performance has not been
  measured yet, and no GPU numbers are claimed.

Out of scope
------------

- Detailed multi-compartment morphology.
- MPI cluster-scale distributed simulation (single-host multi-device execution is in
  scope).
- MUSIC / SONATA co-simulation (importing SONATA network descriptions is in scope).
- e-prop and other online-learning algorithms, which are implemented in ``braintrace``.
