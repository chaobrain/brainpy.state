``brainspike`` documentation
============================

`brainspike <https://github.com/chaobrain/brainspike>`_ is the point-neuron
spiking-network layer of the `BrainX ecosystem <https://brainx.chaobrain.com>`_:
building, simulating, and training spiking neural networks of point neurons. A
new kernel language unifies how models are built, how they are simulated, and
how they are trained. It is built on `JAX <https://github.com/jax-ml/jax>`_ and
`brainstate <https://github.com/chaobrain/brainstate>`_.

``brainspike`` provides **NEST-compatible models** — JAX re-implementations of
`NEST simulator <https://nest-simulator.readthedocs.io/>`_ neuron, synapse,
plasticity (STDP, STP), and device models with NEST parameter names, validated
against live NEST with formal tolerance bands — driven through a NEST-style
``Simulator``. The new kernel (one ``Network`` graph shared by simulation and
learning) is in development and still private.

All parameters carry **physical units** via `brainunit
<https://github.com/chaobrain/brainunit>`_, and every model compiles through JAX
to CPU, GPU, and TPU.


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainspike[cpu]

    .. tab-item:: GPU

       .. code-block:: bash

          pip install -U brainspike[cuda12]

          pip install -U brainspike[cuda13]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainspike[tpu]

    .. tab-item:: Ecosystem

       .. code-block:: bash

          pip install -U BrainX


Quick example
^^^^^^^^^^^^^

A Brunel (2000) sparse balanced random network, built with the explicit
``Simulator``:

.. code-block:: python

   import brainstate
   import brainunit as u
   from brainspike import (Simulator, iaf_psc_alpha, poisson_generator,
                              spike_recorder, all_to_all, fixed_indegree)


   # Brunel (2000) sparse balanced random network, built with the explicit Simulator
   order = 400
   NE, NI = 4 * order, 1 * order              # 1600 excitatory + 400 inhibitory
   CE, CI = int(0.1 * NE), int(0.1 * NI)      # 10% fixed in-degree
   J = 20. * u.pA                             # excitatory PSC; inhibition is -g * J

   neuron = dict(C_m=250. * u.pF, tau_m=20. * u.ms, t_ref=2. * u.ms,
                 E_L=0. * u.mV, V_reset=0. * u.mV, V_th=20. * u.mV)

   sim = Simulator(dt=0.1 * u.ms)
   ne = sim.create(iaf_psc_alpha, NE, params=neuron)
   ni = sim.create(iaf_psc_alpha, NI, params=neuron)
   noise = sim.create(poisson_generator, rate=15000. * u.Hz)
   rec = sim.create(spike_recorder)

   sim.connect(noise, ne + ni, weight=J, delay=1.5 * u.ms, rule=all_to_all)
   sim.connect(ne, ne + ni, weight=J, delay=1.5 * u.ms,
               rule=fixed_indegree(CE), comm='sparse', seed=1)
   sim.connect(ni, ne + ni, weight=-5. * J, delay=1.5 * u.ms,
               rule=fixed_indegree(CI), comm='sparse', seed=2)
   sim.connect(ne[:50], rec)

   res = sim.simulate(1000. * u.ms)
   print(res.rate(rec.segments[0].population))   # excitatory firing rate


----

Learn more
^^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Get Started
         :class-card: sd-text-black sd-bg-light
         :link: get-started/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` NEST-Compatible Hub
         :class-card: sd-text-black sd-bg-light
         :link: nest-style/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`data_exploration;2em` API Reference
         :class-card: sd-text-black sd-bg-light
         :link: apis/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`explore;2em` Examples Gallery
         :class-card: sd-text-black sd-bg-light
         :link: examples/nest-gallery.html


----

See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

``brainspike`` is one part of the `BrainX ecosystem <https://brainx.chaobrain.com/>`__:

- `brainstate <https://github.com/chaobrain/brainstate>`_ — state management for JAX-based brain modeling
- `brainunit <https://github.com/chaobrain/brainunit>`_ — physical units for neuroscience
- `brainevent <https://github.com/chaobrain/brainevent>`_ — event-driven sparse operators
- `braintools <https://github.com/chaobrain/braintools>`_ — surrogate gradients, analysis, and utilities
- `braintrace <https://github.com/chaobrain/braintrace>`_ — linear-memory online learning for spiking networks


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   get-started/index
   nest-style/index


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   apis/index


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples/nest-gallery


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Project

   changelog.md
   project/positioning
   project/citing
   project/ecosystem
