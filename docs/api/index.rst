API Reference
=============

Complete API reference for ``brainpy.state``.

The API is organized into three categories:

- **Base Models** — abstract base classes that all models inherit from
- **BrainPy-style Models** — high-level, composable building blocks for the `BrainPy <https://brainpy.readthedocs.io/>`_ models
- **NEST-Compatible Models** — faithful re-implementations of `NEST simulator <https://nest-simulator.readthedocs.io/>`_ models


Organization
------------

**Base Models**

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`foundation;2em` Base Classes
      :link: base.html

      Abstract base classes: Dynamics, Neuron, Synapse


**BrainPy-style Models**

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`psychology;2em` Neurons
      :link: brainpy-neurons.html

      Spiking neuron models (LIF, ALIF, Izhikevich, HH, etc.)

   .. grid-item-card:: :material-regular:`timeline;2em` Synapses
      :link: brainpy-synapses.html

      Synaptic dynamics (Expon, Alpha, AMPA, GABA, NMDA)

   .. grid-item-card:: :material-regular:`account_tree;2em` Projections
      :link: brainpy-projections.html

      Connect neural populations (AlignPostProj, DeltaProj, etc.)

   .. grid-item-card:: :material-regular:`output;2em` Synaptic Outputs
      :link: brainpy-synouts.html

      Convert conductances to currents (COBA, CUBA, MgBlock)

   .. grid-item-card:: :material-regular:`psychology_alt;2em` Short-Term Plasticity
      :link: brainpy-plasticity.html

      Short-term synaptic plasticity (STP, STD)

   .. grid-item-card:: :material-regular:`sensors;2em` Readouts
      :link: brainpy-readouts.html

      Readout layers (LeakyRateReadout)

   .. grid-item-card:: :material-regular:`input;2em` Input Generators
      :link: brainpy-inputs.html

      Spike and current generators (PoissonSpike, SpikeTime)


**NEST-Compatible Models**

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`hub;2em` NEST Base Classes
      :link: nest-base.html

      NEST marker bases: NESTNeuron, NESTSynapse, NESTPlasticity, NESTDevice

   .. grid-item-card:: :material-regular:`hub;2em` NEST Neurons
      :link: nest-neurons.html

      IAF, AdEx, GIF, GLIF, HH, Izhikevich, rate, and binary neurons

   .. grid-item-card:: :material-regular:`sync_alt;2em` NEST Synapses
      :link: nest-synapses.html

      Static synapses, gap junctions, and special connections

   .. grid-item-card:: :material-regular:`auto_awesome;2em` NEST Plasticity
      :link: nest-plasticity.html

      STP, STDP, and voltage-based learning rules

   .. grid-item-card:: :material-regular:`developer_board;2em` NEST Devices
      :link: nest-devices.html

      Generators, recorders, and detectors




.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Base Models

   base
   nest-base

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: BrainPy-style Models

   brainpy-neurons.rst
   brainpy-synapses.rst
   brainpy-projections.rst
   brainpy-synouts.rst
   brainpy-plasticity.rst
   brainpy-readouts.rst
   brainpy-inputs.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: NEST-Compatible Models

   nest-neurons
   nest-synapses
   nest-plasticity
   nest-devices
