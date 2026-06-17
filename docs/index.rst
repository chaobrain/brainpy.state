``brainpy.state`` documentation
=====================================

`brainpy.state <https://github.com/chaobrain/brainpy.state>`_ is the
point-neuron modeling layer of the `BrainX ecosystem
<https://brainx.chaobrain.com>`_. It provides spiking neural network models
built on `JAX <https://github.com/jax-ml/jax>`_ and the `brainstate
<https://github.com/chaobrain/brainstate>`_ state-management system.

**One differentiable substrate, two worlds.** ``brainpy.state`` is designed so
that the *same* models serve both **brain simulation** (biophysical networks,
spike rasters, conductance dynamics) and **brain-inspired computing**
(surrogate-gradient training, online learning). The bridge is a small set of
ideas — explicit **State**, **physical units**, and the distinctive
**AlignPre / AlignPost** synaptic projection design — that keep memory linear in
the number of neurons while remaining fully differentiable. See
:doc:`concepts/alignpre-alignpost` for the keystone chapter.

.. figure:: /_static/bridging-concept.png
   :alt: A bridge diagram. On the left, a blue "Brain Simulation" panel with a
         network schematic, a spike raster, and a conductance-versus-time trace;
         on the right, an orange "Brain-Inspired Computing" panel with a
         feedforward network, a loss-versus-training-step curve, and a
         surrogate-gradient symbol. An arc across the top links the two panels,
         labelled "surrogate gradients" and "linear-memory online learning". Both
         panels rest on a single platform bar reading "One differentiable
         substrate", whose three pills read "State", "Physical units", and
         "AlignPre / AlignPost".
   :width: 100%
   :align: center

   **Two worlds on one substrate.** The same ``State``-based, unit-aware models —
   wired with AlignPre / AlignPost projections — drive both biophysical **brain
   simulation** and gradient-trained **brain-inspired computing**. Surrogate
   gradients and linear-memory online learning are the bridge between them.


Two model families
^^^^^^^^^^^^^^^^^^^

``brainpy.state`` ships two complementary, production-ready model families on a
shared substrate:

- **BrainPy-style models** — high-level, composable neurons (LIF, ALIF, AdEx,
  HH, Izhikevich, …), synapses (Expon, Alpha, AMPA, GABAa, BioNMDA), projections
  (``AlignPostProj``, ``DeltaProj``, …), readouts, input generators, and
  short-term plasticity, in the tradition of `BrainPy
  <https://brainpy.readthedocs.io/>`_. The idiomatic entry point — start here.
- **NEST-compatible models** — JAX re-implementations of `NEST simulator
  <https://nest-simulator.readthedocs.io/>`_ neuron, synapse, plasticity (STDP,
  STP), and device models with NEST parameter names, validated against live
  NEST with formal tolerance bands. The migration path for NEST users.

All parameters carry **physical units** via `brainunit
<https://github.com/chaobrain/brainunit>`_, and every model compiles through JAX
to CPU, GPU, and TPU.


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


Quick example
^^^^^^^^^^^^^

A small excitatory–inhibitory balanced network (COBA) built from the
BrainPy-style API. See the :doc:`5-minute tour <get-started/5-minute-tour>` for
the full walkthrough.

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


   net = EINet()
   brainstate.nn.init_all_states(net)

   with brainstate.environ.context(dt=0.1 * u.ms):
       times = u.math.arange(0. * u.ms, 1000. * u.ms, brainstate.environ.get_dt())
       spikes = brainstate.transform.for_loop(lambda t: net.update(t, 20. * u.mA), times)


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

      .. card:: :material-regular:`hub;2em` Core Concepts
         :class-card: sd-text-black sd-bg-light
         :link: concepts/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` BrainPy-style Guide
         :class-card: sd-text-black sd-bg-light
         :link: brainpy-style/index.html

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
         :link: examples/brainpy-gallery.html


----

See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

``brainpy.state`` is one part of the `BrainX ecosystem <https://brainx.chaobrain.com/>`__:

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
   concepts/index
   brainpy-style/index
   nest-style/index


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   apis/index


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples/brainpy-gallery
   examples/nest-gallery


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Project

   changelog.md
   project/citing
   project/ecosystem
