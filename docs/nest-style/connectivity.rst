.. _nest-connectivity:

============
Connectivity
============

.. currentmodule:: brainpy.state

Wiring NEST-compatible networks with the :class:`Simulator` API. The vocabulary
mirrors NEST: you ``connect`` a source to a target with a **connection rule** and
a **synapse spec** (weight, delay, and — for plastic synapses — the rule
parameters).

.. code-block:: python

   import brainunit as u
   from brainpy import state as bp

   sim = bp.Simulator(dt=0.1 * u.ms)
   ne = sim.create(bp.iaf_psc_alpha, 800, params=npar)
   ni = sim.create(bp.iaf_psc_alpha, 200, params=npar)

   sim.connect(ne, ne + ni, weight=0.1 * u.pA, delay=1.5 * u.ms,
               rule=bp.fixed_indegree(80), seed=1)


Population algebra (``NodeView``)
=================================

:meth:`Simulator.create` returns a ``NodeView``. Two operators let one
``connect`` call target a combined or partial population:

- **Concatenate** — ``ne + ni`` is the combined population (excitatory followed
  by inhibitory), so a recurrent projection onto "all neurons" is a single call.
- **Slice** — ``ne[:50]`` is the first 50 neurons, e.g. to record a sub-sample.

.. code-block:: python

   sim.connect(ne, ne + ni, ...)     # E → (E and I)
   sim.connect(ne[:50], esr)         # record the first 50 excitatory neurons


Connection rules
================

Pass a rule to ``connect(..., rule=...)``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Rule
     - Meaning
   * - ``all_to_all``
     - Every source connects to every target (the default for a generator
       fanning out).
   * - ``one_to_one``
     - Source ``i`` connects to target ``i`` (equal-size populations).
   * - ``fixed_indegree(K)``
     - Each target draws ``K`` presynaptic partners at random — the Brunel
       wiring rule.

For **spatially-structured** connectivity (distance-dependent probability,
masks, kernels) the rule comes from :doc:`spatial` — e.g.
``bp.spatial.spatial_pairwise_bernoulli(...)`` — and rides the same ``connect``
call.

Sampling options for the random rules:

- ``seed=`` — keys the connectivity draw (so a wiring is reproducible).
- ``allow_multapses=True`` — permit repeated source→target edges (multapses),
  matching NEST's ``fixed_indegree`` default.
- ``comm='sparse'`` — use the sparse event-communication backend for large
  fan-in (keeps memory light at Brunel scale).


The synapse spec
================

Weight and delay are physical quantities:

- **weight** — a synaptic current in ``pA`` (signed: positive excitatory,
  negative inhibitory), delivered as a delayed delta event. For **delta**
  neurons (``iaf_psc_delta``) the weight is an instantaneous membrane-voltage
  jump, so it is a voltage in ``mV`` instead.
- **delay** — a homogeneous axonal delay in ``ms``.
- **per-channel weights** — a multi-segment source (e.g. a 2-channel
  ``poisson_generator``) takes a weight *vector*: ``weight=[1.2, -1.0] * u.pA``
  applies one signed weight per channel.

.. code-block:: python

   sim.connect(noise, neuron, weight=[1.2, -1.0] * u.pA, delay=1.0 * u.ms)

The default synapse is the static synapse (``static_synapse``); see
:doc:`/apis/nest-synapses` for ``static_synapse_hom_w``, ``bernoulli_synapse``,
``cont_delay_synapse``, and the gap-junction / rate-connection family.


Plastic synapses
================

To make an edge plastic, pass a plasticity rule as the ``synapse=`` argument.
The rule carries its own parameters (and, per :doc:`divergences/stdp`, some
parameters NEST keeps on the neuron move onto the synapse spec here):

.. code-block:: python

   import numpy as np
   sg = sim.create(bp.spike_generator, spike_times=np.array([50., 100., 150.]) * u.ms)
   sim.connect(sg, post, synapse=bp.tsodyks2_synapse(
       weight=250. * u.pA, U=0.67, tau_rec=450. * u.ms, tau_fac=0. * u.ms))

Short-term plasticity (``tsodyks_synapse``, ``tsodyks2_synapse``,
``quantal_stp_synapse``), STDP (``stdp_synapse`` and its variants), and
voltage-based rules (``clopath_synapse``, ``urbanczik_synapse``, …) are listed in
:doc:`/apis/nest-plasticity`. Per-edge weights can be recorded and read back with
``res.weight_trace(proj)``.

.. admonition:: Porting note
   :class: seealso

   In NEST a plastic synapse cannot be driven directly by a device, so a
   ``parrot_neuron`` relays the spike train. On the :class:`Simulator` API a
   ``spike_generator`` can drive a plastic edge directly. STDP parameter
   *locations* also differ from NEST — see :doc:`divergences/stdp` before porting
   a learning rule.


Introspecting realized connectivity
===================================

After wiring, enumerate the realized edges with
:meth:`Simulator.get_connections` (NEST's ``GetConnections`` /
``SynapseCollection`` idiom) to read or write per-edge ``weight`` / ``delay`` /
``source`` / ``target`` without holding each projection handle — useful for
plotting weight matrices or auditing a complex network.


See also
========

- :doc:`tutorials/03-connect-network` — connection rules in a runnable Brunel.
- :doc:`spatial` — distance-dependent connection rules.
- :doc:`devices` — generators and recorders that connect into the network.
- :doc:`/apis/nest-synapses`, :doc:`/apis/nest-plasticity` — the synapse and
  plasticity API.
- :doc:`divergences/stdp` — where STDP state lives and the pairing conventions.
