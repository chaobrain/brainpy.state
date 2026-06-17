API Reference
=============

Complete API reference for ``brainpy.state``, organized by model family. The
library exposes two families that share the same state-based substrate:

- **BrainPy-style** — composable, differentiable neurons, synapses, projections,
  outputs, plasticity, readouts, and input generators.
- **NEST-compatible** — JAX re-implementations that preserve NEST parameter
  names and semantics, plus the spatial and network-builder layers.

Both families are usable from the single public ``brainpy.state`` namespace.

.. toctree::
   :hidden:
   :caption: BrainPy-style

   base
   brainpy-neurons
   brainpy-synapses
   brainpy-projections
   brainpy-synouts
   brainpy-plasticity
   brainpy-readouts
   brainpy-inputs

.. toctree::
   :hidden:
   :caption: NEST-compatible

   nest-base
   nest-neurons
   nest-synapses
   nest-plasticity
   nest-devices
   nest-spatial
   nest-network


BrainPy-style
-------------

The native modeling layer: compose neurons, synapses, and projections, and
train them end to end with surrogate gradients.

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`foundation;2em` Base Classes
      :link: base.html

      Abstract base classes shared by all models: ``Dynamics``, ``Neuron``,
      ``Synapse``.

   .. grid-item-card:: :material-regular:`psychology;2em` Neurons
      :link: brainpy-neurons.html

      Spiking neuron models (LIF, ALIF, AdEx, HH, Izhikevich, …).

   .. grid-item-card:: :material-regular:`timeline;2em` Synapses
      :link: brainpy-synapses.html

      Synaptic dynamics (Expon, DualExpon, Alpha, AMPA, GABAa, BioNMDA).

   .. grid-item-card:: :material-regular:`account_tree;2em` Projections
      :link: brainpy-projections.html

      Connect neural populations (``AlignPostProj``, ``DeltaProj``,
      ``CurrentProj``, gap junctions, AlignPre/AlignPost helpers).

   .. grid-item-card:: :material-regular:`output;2em` Synaptic Outputs
      :link: brainpy-synouts.html

      Convert conductances to currents (COBA, CUBA, MgBlock).

   .. grid-item-card:: :material-regular:`psychology_alt;2em` Short-Term Plasticity
      :link: brainpy-plasticity.html

      Short-term synaptic plasticity (STP, STD).

   .. grid-item-card:: :material-regular:`sensors;2em` Readouts
      :link: brainpy-readouts.html

      Readout layers (``LeakyRateReadout``).

   .. grid-item-card:: :material-regular:`input;2em` Input Generators
      :link: brainpy-inputs.html

      Spike and current generators (``PoissonSpike``, ``SpikeTime``,
      ``PoissonInput``).


NEST-compatible
---------------

JAX re-implementations preserving NEST parameter names, plus the spatial and
network-builder layers. Parity with a live NEST install is documented on the
:doc:`validation status page </nest-style/validation-status>`.

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`hub;2em` Base Classes
      :link: nest-base.html

      ``NESTNeuron``, ``NESTSynapse``, ``NESTPlasticity``, ``NESTDevice``.

   .. grid-item-card:: :material-regular:`hub;2em` Neurons
      :link: nest-neurons.html

      IAF, AdEx, GIF, GLIF, HH, Izhikevich, rate, and binary neurons.

   .. grid-item-card:: :material-regular:`sync_alt;2em` Synapses
      :link: nest-synapses.html

      Static synapses, gap junctions, and special connections.

   .. grid-item-card:: :material-regular:`auto_awesome;2em` Plasticity
      :link: nest-plasticity.html

      STDP, Tsodyks-Markram STP, and voltage-based learning rules.

   .. grid-item-card:: :material-regular:`developer_board;2em` Devices
      :link: nest-devices.html

      Generators, recorders, and detectors.

   .. grid-item-card:: :material-regular:`grain;2em` Spatial Networks
      :link: nest-spatial.html

      Spatially-structured layers, distance kernels, masks, and visualization.

   .. grid-item-card:: :material-regular:`schema;2em` Network Builder
      :link: nest-network.html

      Declarative network construction, projections, connection rules, and the
      simulator.
