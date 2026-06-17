.. _nest-style:

==============================
NEST-Compatible Models
==============================

JAX re-implementations of `NEST simulator <https://nest-simulator.readthedocs.io/>`_
neuron, synapse, plasticity, and device models — preserving NEST's parameter
names, defaults, and unit conventions so a researcher coming from NEST can port a
model with minimal friction. The models are built on
`brainstate <https://github.com/chaobrain/brainstate>`_, use
`brainunit <https://github.com/chaobrain/brainunit>`_ for physical units, compile
through JAX to CPU, GPU, and TPU, and are **differentiable** where the dynamics
allow it.

They are numerically validated against a **live NEST install** within documented
tolerance bands — see :doc:`validation-status` for the per-family parity evidence
and the A–E tolerance categories.

You build NEST-compatible networks with the explicit :class:`Simulator` API, which
mirrors NEST's vocabulary — ``create`` populations and devices, ``connect`` them
with NEST-like rules, ``simulate``:

.. code-block:: python

   import brainunit as u
   from brainpy import state as bp

   sim = bp.Simulator(dt=0.1 * u.ms)
   neuron = sim.create(bp.iaf_psc_alpha, 1, I_e=376. * u.pA)
   vm = sim.create(bp.voltmeter)
   sim.connect(vm, neuron)                     # the voltmeter observes the neuron
   res = sim.simulate(1000. * u.ms)
   v = res.trace(vm, "V_m")                    # (T, 1) membrane-potential trace

This hub is self-contained: it has its own first-steps tutorials, build-your-own-
network guides, and the porting / fidelity material — and it links *up* into the
shared :doc:`/concepts/index` spine (State, physical units, the ``transform``
loops) where the deeper ideas live. We do **not** duplicate NEST's model
*physiology* here — for the biophysics of any model, follow the link out to the
upstream `NEST documentation <https://nest-simulator.readthedocs.io/>`_; our pages
cover the ``brainpy.state`` API, porting parity, semantic divergences, and the
validation showcase.

.. card:: Scaling up: large networks
   :link: /examples/nest-gallery
   :link-type: doc

   Looking for full-scale reference networks — Brunel random balanced networks,
   EI-clustered networks, astrocyte–neuron networks? The
   :doc:`NEST example gallery </examples/nest-gallery>` catalogues 75 ported
   scripts as a NEST → ``brainpy.state`` Rosetta map, each backed by a live-NEST
   parity test.


First steps
===========

Start here. Four short tutorials in the ``brainpy.state`` idiom (State, units,
and the :class:`Simulator` API) — mirroring PyNEST Parts 1–4.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: 1 · One neuron
      :link: tutorials/01-one-neuron
      :link-type: doc

      Create one ``iaf_psc_alpha``, drive it with a current and Poisson noise,
      and record its membrane potential and spikes.

   .. grid-item-card:: 2 · Populations & devices
      :link: tutorials/02-populations
      :link-type: doc

      Build populations, combine and slice them with ``NodeView`` algebra, and
      attach stimulation and recording devices.

   .. grid-item-card:: 3 · Connect a network
      :link: tutorials/03-connect-network
      :link-type: doc

      Wire populations with NEST-like connection rules and assemble a small
      Brunel random balanced network.

   .. grid-item-card:: 4 · Record & analyze
      :link: tutorials/04-record-and-analyze
      :link-type: doc

      Use recorders and detectors, read traces back, and analyze firing rate,
      raster, and ISI statistics.


Build your own network
======================

Reference guides for the building blocks: models, connectivity, devices, and
spatial structure.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Models
      :link: models
      :link-type: doc

      The model directory by family (IAF, AdEx, GIF/GLIF, HH, rate, …), linking
      to the API reference and out to NEST for physiology.

   .. grid-item-card:: Connectivity
      :link: connectivity
      :link-type: doc

      Connection rules and the synapse spec — weights, delays, static and
      plastic synapses, population algebra.

   .. grid-item-card:: Devices
      :link: devices
      :link-type: doc

      Generators, recorders, and detectors — 27 device models and how to read
      their data back.

   .. grid-item-card:: Spatial networks
      :link: spatial
      :link-type: doc

      Spatially-structured networks: grid layouts, distance-dependent
      connectivity, and position queries.

   .. grid-item-card:: Example gallery
      :link: /examples/nest-gallery
      :link-type: doc

      All 75 ported ``nest_like`` scripts as a NEST → ``brainpy.state`` map.


Porting & fidelity
==================

The distinctive material: a side-by-side port, the semantic-divergence catalog,
the validation showcase, and how the integrators work under the hood.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Porting walkthrough
      :link: porting-walkthrough
      :link-type: doc

      Brunel side by side — NEST vs ``brainpy.state`` — with the parameter map
      and the divergences a port must account for.

   .. grid-item-card:: Semantic divergences
      :link: divergences/index
      :link-type: doc

      Where state lives, parameter-location maps, numerical bands, and pairing
      conventions — every entry backed by a live-NEST parity test.

   .. grid-item-card:: Validation status
      :link: validation-status
      :link-type: doc

      Per-family live-NEST parity, the A–E tolerance categories, and how to run
      the parity harness yourself.

   .. grid-item-card:: Integration categories
      :link: integration-categories
      :link-type: doc

      Under the hood: RKF45, analytic propagators, vectorized rate updates, and
      no-ODE devices.


.. toctree::
   :hidden:
   :caption: First steps

   tutorials/01-one-neuron
   tutorials/02-populations
   tutorials/03-connect-network
   tutorials/04-record-and-analyze

.. toctree::
   :hidden:
   :caption: Build your own network

   models
   connectivity
   devices
   spatial

.. toctree::
   :hidden:
   :caption: Porting & fidelity

   porting-walkthrough
   divergences/index
   divergences/stdp
   validation-status
   integration-categories
