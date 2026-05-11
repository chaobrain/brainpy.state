:orphan:

API Reference
=============

Complete API reference for ``brainpy.state``. The library exposes two distinct
model families with different maturity levels — see the
:doc:`landing page </index>` for the full maturity table.


Stable API
----------

The Stable API is the recommended entry point for production work and
surrogate-gradient training. Public interfaces are stable for the 0.0.x series.

**Base classes**

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`foundation;2em` Base Classes
      :link: base.html

      Abstract base classes shared by all models: ``Dynamics``, ``Neuron``,
      ``Synapse``.

**BrainPy-style models**

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`psychology;2em` Neurons
      :link: brainpy-neurons.html

      Spiking neuron models (LIF, ALIF, AdEx, HH, Izhikevich, …).

   .. grid-item-card:: :material-regular:`timeline;2em` Synapses
      :link: brainpy-synapses.html

      Synaptic dynamics (Expon, Alpha, AMPA, GABA, NMDA).

   .. grid-item-card:: :material-regular:`account_tree;2em` Projections
      :link: brainpy-projections.html

      Connect neural populations (``AlignPostProj``, ``DeltaProj``, …).

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

      Spike and current generators (``PoissonSpike``, ``SpikeTime``).


Experimental API (NEST-Compatible)
----------------------------------

.. warning::

   The NEST-compatible model family is **under active development**. Parameter
   names, defaults, numerical behavior, and the set of available models may
   change without notice across 0.0.x releases. Start with the
   :doc:`/nest-status/index` for current scope and limitations.

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`info;2em` NEST-style Status
      :link: ../nest-status/index.html

      Scope, limitations, ongoing migration, and what users should not yet
      rely on.

   .. grid-item-card:: :material-regular:`hub;2em` NEST Base Classes
      :link: nest-base.html

      ``NESTNeuron``, ``NESTSynapse``, ``NESTPlasticity``, ``NESTDevice``.

   .. grid-item-card:: :material-regular:`hub;2em` NEST Neurons
      :link: nest-neurons.html

      IAF, AdEx, GIF, GLIF, HH, Izhikevich, rate, and binary neurons.

   .. grid-item-card:: :material-regular:`sync_alt;2em` NEST Synapses
      :link: nest-synapses.html

      Static synapses, gap junctions, and special connections.

   .. grid-item-card:: :material-regular:`auto_awesome;2em` NEST Plasticity
      :link: nest-plasticity.html

      STDP, STP, and voltage-based learning rules.

   .. grid-item-card:: :material-regular:`developer_board;2em` NEST Devices
      :link: nest-devices.html

      Generators, recorders, and detectors.
