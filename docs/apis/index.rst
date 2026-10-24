API Reference
=============

Complete API reference for ``brainspike``: JAX re-implementations of NEST
models that preserve NEST parameter names and semantics, the spatial layer, and
the NEST-style ``Simulator``. Everything is importable from the public
``brainspike`` namespace.

.. toctree::
   :hidden:
   :caption: NEST-compatible

   base
   nest-base
   nest-neurons
   nest-synapses
   nest-plasticity
   nest-devices
   nest-spatial
   nest-network


NEST-compatible
---------------

JAX re-implementations preserving NEST parameter names, plus the spatial layer
and the ``Simulator``. Parity with a live NEST install is documented on the
:doc:`validation status page </nest-style/validation-status>`.

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`foundation;2em` Shared Base Classes
      :link: base.html

      ``Dynamics``, ``Neuron``, ``Synapse``.

   .. grid-item-card:: :material-regular:`hub;2em` NEST Base Classes
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

   .. grid-item-card:: :material-regular:`schema;2em` Simulator
      :link: nest-network.html

      The NEST-style ``Simulator``, population views, connection rules, and
      event projections.
