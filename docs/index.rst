``brainpy.state`` documentation
=====================================

`brainpy.state <https://github.com/chaobrain/brainpy.state>`_ provides comprehensive spiking neural network models built on
`JAX <https://github.com/jax-ml/jax>`_ and the
`brainstate <https://github.com/chaobrain/brainstate>`_ state-management system.
It is the point-neuron modeling layer of the
`BrainX ecosystem <https://brainmodeling.readthedocs.io>`_.

The library ships **167+ models** organized in three tiers:

- **Base classes**: ``Dynamics``, ``Neuron``, ``Synapse``, the abstract foundation every model inherits from.
- **BrainPy-style models (45+)**: high-level, composable neurons (LIF, HH, Izhikevich, ...),
  synapses (Expon, Alpha, AMPA, NMDA, ...), projections, readouts, and input generators
  previously designed in `BrainPy <https://brainpy.readthedocs.io/>`_.
- **NEST-compatible models (119+)**: faithful JAX re-implementations of
  `NEST simulator <https://nest-simulator.readthedocs.io/>`_ neuron, synapse,
  plasticity (STDP, STP), and device models.
- All parameters carry **physical units** via `saiunit <https://github.com/chaobrain/saiunit>`_,
  and every neuron supports surrogate-gradient-based training out of the box.

Compared to ``brainpy.dyn``, ``brainpy.state`` has the following characteristics:

- **Ecosystem compatability**: ``brainpy.state`` is built on `brainstate <https://github.com/chaobrain/brainstate>`_ and fully compatible with `BrainX ecosystem <https://brainmodeling.readthedocs.io>`_.
- **Model scope**: ``brainpy.state`` implements much more models including BrainPy-style models plus a large NEST-compatible model set.
- **Scientific ergonomics**: ``brainpy.state`` uses physical units via ``saiunit`` by default and is designed for surrogate-gradient training.


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainpy.state[cpu]

    .. tab-item:: GPU

       .. code-block:: bash

          pip install -U brainpy.state[cuda12]

          pip install -U brainpy.state[cuda13]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainpy.state[tpu]

    .. tab-item:: Ecosystem

       .. code-block:: bash

          pip install -U BrainX


Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   import brainpy
   import brainstate
   import saiunit as u

   # Create neuron populations
   E = brainpy.state.LIF(3200, V_rest=-60*u.mV, V_th=-50*u.mV, tau=20*u.ms)
   I = brainpy.state.LIF(800,  V_rest=-60*u.mV, V_th=-50*u.mV, tau=20*u.ms)


----

Learn more
^^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` 5-Minute Tutorial
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/5min-tutorial.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Core Concepts
         :class-card: sd-text-black sd-bg-light
         :link: core-concepts/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`explore;2em` Examples Gallery
         :class-card: sd-text-black sd-bg-light
         :link: examples/gallery.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`data_exploration;2em` API Reference
         :class-card: sd-text-black sd-bg-light
         :link: api/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` BrainX Ecosystem
         :class-card: sd-text-black sd-bg-light
         :link: https://brainmodeling.readthedocs.io

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`history;2em` Changelog
         :class-card: sd-text-black sd-bg-light
         :link: changelog.html


----

See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

``brainpy.state`` is one part of the `BrainX ecosystem <https://brainmodeling.readthedocs.io/>`__:

- `brainstate <https://github.com/chaobrain/brainstate>`_ — state management for JAX-based brain modeling
- `saiunit <https://github.com/chaobrain/saiunit>`_ — physical units for neuroscience
- `brainevent <https://github.com/chaobrain/brainevent>`_ — event-driven sparse operators
- `braintools <https://github.com/chaobrain/braintools>`_ — surrogate gradients, analysis, and utilities




.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   quickstart/index.rst
   core-concepts/index.rst
   examples/gallery.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   api/index.rst
   changelog.md
