.. _nest-validation-status:

=================
Validation status
=================

.. currentmodule:: brainpy.state

Why trust the NEST-compatible family? Because each model is checked against a
**live NEST install** and the comparison is asserted in a test you can run
yourself. This page is the trust showcase: how parity is asserted, the tolerance
categories that say *what counts as a match for what kind of model*, the
per-family status, and how to run the harness.

The validation harness lives at ``brainpy_state/_nest_validation/``. Every parity
test imports the **same** comparison engine and the **same** documented
tolerances, so "did it match NEST?" is asked the same way everywhere.


How parity is asserted
======================

NEST and JAX draw from independent PRNG streams, so not everything can be compared
sample-by-sample. The harness has two comparison modes:

.. list-table::
   :header-rows: 1
   :widths: 16 38 30 16

   * - Mode
     - When
     - How it compares
     - Categories
   * - **trace**
     - Deterministic drive — same ``dt``, fixed input, analytic or mean-field
       dynamics.
     - Per-sample / max-abs error, with an optional ±1-step recorder alignment.
     - A, B, C
   * - **distributional**
     - PRNG-divergent drive — Poisson input, random connectivity, stochastic
       neurons.
     - A seed-**aggregated** statistic (the mean over seeds), **never**
       per-sample.
     - D

Anything stochastic is therefore compared distributionally — averaged over seeds,
never spike-by-spike. The pass test is the division-free ``numpy.allclose`` form
``|a − b| ≤ atol + rtol·|reference|``, so a zero reference never divides.


Tolerance categories A–E
========================

The single source of truth for *what tolerance for what kind of model* is
``brainpy_state/_nest_validation/tolerance_conventions.py`` — importable,
unit-aware constants. The five categories span the comparison space the harness
must cover (membrane-voltage trace, firing rate, weight trajectory,
PSC-amplitude train, F–I curve):

.. list-table::
   :header-rows: 1
   :widths: 6 30 26 22 16

   * - Cat
     - Kind
     - Example metric
     - Tolerance
     - Constant
   * - **A**
     - Adaptive numerical integrator (RKF45)
     - ``aeif`` / HH / ``izhikevich`` ``V_m``
     - ``atol 1e-3 mV`` (trace)
     - ``CAT_A``
   * - **B**
     - Analytic exact propagator
     - linear ``iaf_psc_*`` ``V_m`` / PSC
     - ``atol 1e-6 mV`` (trace); ``CAT_B_ALIGNED``: ``5e-2 mV``, ±1 step
     - ``CAT_B``
   * - **C**
     - Conductance / coupled / mean-field
     - ``iaf_cond_*``; ``siegert_neuron`` rate
     - ``1e-3 mV`` trace / ``5 %`` rate
     - ``CAT_C``, ``CAT_C_RATE``
   * - **D**
     - Distributional (PRNG-divergent)
     - network firing rate, ISI CV
     - seed-mean ``5 %``, ``≥ 5`` seeds
     - ``CAT_D``
   * - **E**
     - Spike-time / event-count
     - ``*_ps`` precise spiking, PSC peak timing
     - ``|ΔN| ≤ 2``, ``|Δstep| ≤ 1``
     - ``CAT_E``

The A/B/C numbers are absolute (and unit-aware for voltages). ``CAT_C_RATE`` is
the deterministic mean-field rate fixed point, compared purely relatively (Hz).
Category D's multi-seed protocol runs ``N ≥ 5`` seeds per side and compares the
seed-mean; the NEST side offsets its seeds to decorrelate the two PRNG streams.


Per-family parity status
========================

Each family is validated under the category that fits how it is solved (see
:doc:`integration-categories`). The table lists a representative parity test for
each; browse ``brainpy_state/_nest_validation/`` for the full set.

.. list-table::
   :header-rows: 1
   :widths: 30 12 40

   * - Family
     - Category
     - Representative parity test
   * - Current-based IAF (``iaf_psc_*``)
     - B
     - ``iaf_psc_alpha_parity_test.py``
   * - Conductance-based IAF (``iaf_cond_*``)
     - C
     - ``iaf_cond_alpha_conductance_test.py``
   * - AdEx (``aeif_*``)
     - A
     - ``aeif_cond_exp_conductance_test.py``
   * - Hodgkin–Huxley (``hh_*``, ``ht_neuron``)
     - A / C
     - ``hh_cond_exp_traub_conductance_test.py``
   * - Izhikevich, GIF, GLIF
     - A
     - ``izhikevich_test.py``, ``glif_psc_neuron_test.py``
   * - Rate models, ``siegert_neuron``
     - C / D
     - ``lin_rate_ipn_network_test.py``, ``siegert_diffusion_test.py``
   * - Brunel / balanced networks
     - D
     - ``brunel_alpha_test.py``, ``brunel_delta_test.py``
   * - Precise spiking (``*_ps``)
     - E
     - ``precise_spiking_test.py``
   * - Short-term plasticity (Tsodyks, quantal)
     - A (V_m) / D
     - ``evaluate_tsodyks2_synapse_test.py``, ``quantal_stp_parity_test.py``
   * - STDP family
     - E (+ documented bands)
     - ``stdp_synapse_parity_test.py`` and the ``stdp_*`` siblings
   * - Devices (generators, recorders, detectors)
     - E / D
     - ``device_parity_test.py``, ``recording_demo_test.py``
   * - Spatial connectivity
     - D
     - ``spatial_gaussian_kernel_test.py``, ``spatial_grid_test.py``

The STDP family carries a few **documented numerical bands** (Clopath ≤ 5 %,
dopamine ~0.2 %, and the nearest-neighbour "phantom pre at 0" that is
intentionally not reproduced). The direction and ordering of every weight change
are exact; only a small magnitude band remains. See :doc:`divergences/stdp` for
the exact numbers and the test that asserts each.


Running the harness
===================

The parity tests use a ``requires_nest`` pytest marker (registered in the
harness ``conftest.py``) that **skips cleanly** when ``import nest`` fails — so
you can run them with or without a NEST install.

.. code-block:: bash

   # All live-NEST parity tests (skipped where NEST is absent):
   python -m pytest brainpy_state/_nest_validation/ -m requires_nest -q

   # Everything in the package — harness unit tests always run; the live-NEST
   # tests run if NEST is present, else skip:
   python -m pytest brainpy_state/_nest_validation/ -q

Without NEST installed, the ``@requires_nest`` tests skip and the harness's own
unit tests (the tolerance conventions and the comparison engine) still pass, so
you can verify the machinery is sound before installing NEST.


Known coverage gaps
===================

In the spirit of showing evidence rather than hiding caveats, here is the honest
scope:

- **Multi-compartment models (``_mc``).** ``iaf_cond_alpha_mc`` is not yet
  validated against NEST; ``pp_cond_exp_mc_urbanczik`` *does* have per-compartment
  live-NEST parity (used by the Urbanczik dendritic-prediction example).
- **File-backed recording.** Recording is in-memory only — there is no
  file/ascii backend yet, so device tours that exercise NEST's ``record_to``
  backend axis collapse to the in-memory equivalent.
- **A few legacy single-seed distributional tests.** Some early
  distributional tests use a single realization rather than the full
  ``≥ 5``-seed protocol; this is a documented limitation, not a parity failure.

When you find a numerical mismatch against NEST upstream, open an issue with a
minimal reproducer at https://github.com/chaobrain/brainpy.state/issues.


See also
========

- :doc:`integration-categories` — how each family's ODE (if any) is solved.
- :doc:`divergences/index` — the semantic-divergence catalog the parity tests back.
- :doc:`divergences/stdp` — the STDP numerical bands and pairing conventions.
- :doc:`/apis/nest-neurons`, :doc:`/apis/nest-synapses`,
  :doc:`/apis/nest-plasticity`, :doc:`/apis/nest-devices` — the API reference.
- `NEST simulator documentation <https://nest-simulator.readthedocs.io/>`_ — the
  authoritative reference for upstream model semantics.
