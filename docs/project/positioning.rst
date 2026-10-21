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
- **Differentiable like a training library.** Models are ``brainstate`` modules,
  compiled through JAX/XLA to CPU, GPU, and TPU, and trainable with surrogate
  gradients or online learning rules.
- **One substrate for both.** The memory-efficient AlignPre/AlignPost projections that
  make large simulations fit are the same ones that make gradient-based and online
  learning memory-efficient.

Who it is for
-------------

1. Computational neuroscientists who want to run NEST models on accelerators, or train
   them.
2. Brain-inspired-computing researchers who need biologically constrained spiking
   networks.
3. Libraries in the `BrainX ecosystem <https://brainx.chaobrain.com/>`_ that build on it,
   such as ``braintrace`` for online learning.

Two front-ends, one kernel
--------------------------

The **BrainPy-style** API (composable ``Module`` + ``State`` + projections) is the
extensible, differentiable core. The **NEST-style** ``Simulator``
(``create`` / ``connect`` / ``simulate``) is an approachable front-end in NEST's
vocabulary. Both build on the same building blocks, and the project is converging them
so that any model can be used from either side.

Out of scope
------------

- Detailed multi-compartment morphology.
- MPI cluster-scale distributed simulation.
- MUSIC / SONATA co-simulation.
- e-prop, which is implemented in ``braintrace``.
