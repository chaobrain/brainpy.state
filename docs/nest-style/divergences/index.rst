.. _nest-divergences:

====================
Semantic divergences
====================

.. currentmodule:: brainpy.state

When you port a NEST model to ``brainpy.state``, the parameter *names* match — but
a few **semantics** differ in ways the source code does not surface at a glance.
This catalog states each divergence exactly. Every entry is backed by a live-NEST
parity test, so the catalog consolidates what those tests proved (see
:doc:`/nest-style/validation-status` for the harness and tolerance categories).

There are five kinds of divergence a port must account for.


1. Where state lives
====================

NEST sometimes keeps learning or trace state on the *neuron*; ``brainpy.state``
moves it onto the *synapse* (or a broadcast node) so a rule runs standalone —
without requiring an archiving-capable postsynaptic neuron. The canonical case is
the STDP post-synaptic trace time constant ``tau_minus``: a **neuron** parameter
in NEST, a **synapse-spec** attribute here.

Setting a parameter on the wrong object relative to NEST is the single most common
porting mistake. The full treatment — with the porting rule and the parameter
table — is on the STDP page:

- :doc:`stdp` — :ref:`stdp-tau-minus` (trace storage) and
  :ref:`stdp-param-location` (the parameter-location map for the whole STDP
  family).


2. Parameter-location maps
==========================

Wherever NEST precomputes a weight change on an archiving neuron,
``brainpy.state`` relocates the governing parameters onto the synapse spec (or, for
the dopamine modulator, onto the ``volume_transmitter``). Parity always sets
identical values on both sides. The exhaustive table — including which Clopath
parameters move to the synapse and which voltage filters *stay* on the neuron — is
in :ref:`stdp-param-location`.


3. Documented numerical bands
=============================

A few rules do not reproduce NEST bit-for-bit. In each case the **direction and
ordering** of the effect are exact; only a small magnitude band remains, and each
band is asserted by a parity test. Pull the numbers from the STDP page, not from a
fresh measurement:

.. list-table::
   :header-rows: 1
   :widths: 36 24 40

   * - Divergence
     - Band
     - Where
   * - Clopath online LTP vs delayed ring buffer
     - ≤ 5 %
     - :ref:`stdp-clopath`
   * - Dopamine online-vs-deferred integration
     - ~0.2 %
     - :ref:`stdp-dopamine`
   * - Nearest-neighbour "phantom pre at 0"
     - not reproduced (documented)
     - :ref:`stdp-phantom-pre`

The online-vs-deferred origin of these bands — ``brainpy.state`` integrates every
step while NEST defers the weight integral to the next pre spike — is explained in
the STDP overview.


4. Device and recording conventions
===================================

The :class:`Simulator` device model differs from NEST's in a few load-bearing,
but easy-to-remember, ways (full detail in :doc:`/nest-style/devices`):

- **Reversed analog tap.** A ``voltmeter`` / ``multimeter`` is connected in NEST's
  *reversed* direction — ``connect(voltmeter, neuron)`` — because the recorder
  *observes* the neuron rather than receiving events from it.
- **Current vs spike sources.** Current devices (``dc_``/``noise_``/``step_``/
  ``ac_generator``) inject through the neuron's current ring buffer (a
  NEST-faithful one-step delay); spike sources (``poisson_generator``,
  ``spike_generator``) deliver delayed delta events.
- **In-memory recording.** There is no file/ascii backend yet, so NEST's
  ``record_to`` backend axis collapses to the in-memory equivalent; data is read
  back after the run with ``res.trace`` / ``res.spikes`` / ``res.rate`` /
  ``res.n_events`` / ``res.weight_trace``.
- **Plastic edges need a spike source.** In NEST a plastic synapse cannot be
  driven by a device, so a ``parrot_neuron`` relays the train; on the
  :class:`Simulator` API a ``spike_generator`` can drive the plastic edge directly
  (with a one-step holder lag the parity tests align to).


5. PRNG-divergent drives → distributional parity
================================================

NEST and JAX draw from independent PRNG streams, so any Poisson drive, random
connectivity, or stochastic neuron diverges sample-by-sample even when the model
is correct. These are validated **distributionally** — the seed-aggregated
statistic (the mean over ``≥ 5`` seeds) is compared, never a per-sample trace.
This is comparison category D; see :doc:`/nest-style/validation-status`.


.. toctree::
   :hidden:

   stdp


See also
========

- :doc:`stdp` — the full STDP trace-storage, parameter-location, numerical-band,
  and nearest-neighbour pairing reference.
- :doc:`/nest-style/validation-status` — the parity harness and the A–E tolerance
  categories that back every divergence here.
- :doc:`/nest-style/porting-walkthrough` — a Brunel port that runs into several of
  these divergences in context.
- `NEST simulator documentation <https://nest-simulator.readthedocs.io/>`_ — the
  authoritative reference for upstream model semantics.
