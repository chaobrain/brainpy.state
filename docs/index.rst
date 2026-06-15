``brainpy.state`` documentation
=====================================

`brainpy.state <https://github.com/chaobrain/brainpy.state>`_ provides spiking
neural network models built on `JAX <https://github.com/jax-ml/jax>`_ and the
`brainstate <https://github.com/chaobrain/brainstate>`_ state-management system.
It is the point-neuron modeling layer of the
`BrainX ecosystem <https://brainx.chaobrain.com>`_.


.. admonition:: API maturity
   :class: important

   ``brainpy.state`` ships two model families with different maturity levels:

   - **Stable** — Base classes (``Dynamics``, ``Neuron``, ``Synapse``) and
     **BrainPy-style models** (45+ neurons, synapses, projections, readouts,
     input generators). Public API is stable for the 0.0.x series and
     recommended for production use and surrogate-gradient training.
   - **Experimental — In Development** — **NEST-compatible models** (119+
     neurons, synapses, plasticity, devices). These are under active
     development; parameter names, defaults, and numerical behavior may change
     without notice. Use them for exploration and validation, but pin your
     dependency version and expect breaking changes.

   See the :doc:`NEST-style status page <nest-status/index>` for what is
   currently available and what users should not rely on yet.


What's in the library
^^^^^^^^^^^^^^^^^^^^^

- **Base classes** — ``Dynamics``, ``Neuron``, ``Synapse``: the abstract
  foundation every model inherits from.
- **BrainPy-style models** (Stable, 45+) — high-level, composable neurons
  (LIF, ALIF, AdEx, HH, Izhikevich, ...), synapses (Expon, Alpha, AMPA, GABA,
  NMDA), projections, readouts, input generators, and short-term plasticity,
  designed in the tradition of `BrainPy <https://brainpy.readthedocs.io/>`_.
- **NEST-compatible models** (Experimental, 119+) — JAX re-implementations
  of `NEST simulator <https://nest-simulator.readthedocs.io/>`_ neuron,
  synapse, plasticity (STDP, STP), and device models with NEST-compatible
  parameter names.
- All parameters carry **physical units** via `brainunit
  <https://github.com/chaobrain/brainunit>`_; BrainPy-style neurons support
  surrogate-gradient training out of the box.


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

A small excitatory–inhibitory balanced network using the Stable BrainPy-style
API. See the :doc:`5-minute tutorial <quickstart/5min-tutorial>` for the full
walkthrough.

.. code-block:: python

   import brainpy
   import brainstate
   import braintools
   import brainunit as u


   class EINet(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.n_exc, self.n_inh = 3200, 800
           self.num = self.n_exc + self.n_inh
           self.N = brainpy.state.LIFRef(
               self.num,
               V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
               tau=20. * u.ms, tau_ref=5. * u.ms,
               V_initializer=braintools.init.Normal(-55., 2., unit=u.mV),
           )
           self.E = brainpy.state.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(self.n_exc, self.num,
                                                  conn_num=0.02, conn_weight=0.6 * u.mS),
               syn=brainpy.state.Expon.desc(self.num, tau=5. * u.ms),
               out=brainpy.state.COBA.desc(E=0. * u.mV),
               post=self.N,
           )
           self.I = brainpy.state.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(self.n_inh, self.num,
                                                  conn_num=0.02, conn_weight=6.7 * u.mS),
               syn=brainpy.state.Expon.desc(self.num, tau=10. * u.ms),
               out=brainpy.state.COBA.desc(E=-80. * u.mV),
               post=self.N,
           )

       def update(self, t, inp):
           with brainstate.environ.context(t=t):
               spk = self.N.get_spike() != 0.
               self.E(spk[:self.n_exc])
               self.I(spk[self.n_exc:])
               self.N(inp)
               return self.N.get_spike()


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

      .. card:: :material-regular:`library_books;2em` BrainPy-style Guide
         :class-card: sd-text-black sd-bg-light
         :link: brainpy-guide/index.html

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

      .. card:: :material-regular:`science;2em` NEST-style Status
         :class-card: sd-text-black sd-bg-light
         :link: nest-status/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` BrainX Ecosystem
         :class-card: sd-text-black sd-bg-light
         :link: https://brainx.chaobrain.com


----

See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

``brainpy.state`` is one part of the `BrainX ecosystem <https://brainx.chaobrain.com/>`__:

- `brainstate <https://github.com/chaobrain/brainstate>`_ — state management for JAX-based brain modeling
- `brainunit <https://github.com/chaobrain/brainunit>`_ — physical units for neuroscience
- `brainevent <https://github.com/chaobrain/brainevent>`_ — event-driven sparse operators
- `braintools <https://github.com/chaobrain/braintools>`_ — surrogate gradients, analysis, and utilities


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   quickstart/index.rst
   brainpy-guide/index.rst
   examples/gallery.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference (Stable)

   api/base
   api/brainpy-neurons
   api/brainpy-synapses
   api/brainpy-projections
   api/brainpy-synouts
   api/brainpy-plasticity
   api/brainpy-readouts
   api/brainpy-inputs


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Experimental (NEST-Compatible)

   nest-status/index
   api/nest-base
   api/nest-neurons
   api/nest-synapses
   api/nest-plasticity
   api/nest-devices


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: NEST Porting Guide

   nest-guide/index


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Project

   changelog.md
