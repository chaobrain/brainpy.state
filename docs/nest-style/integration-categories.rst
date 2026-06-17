.. _nest-integration-categories:

======================
Integration categories
======================

.. currentmodule:: brainpy.state

NEST-compatible models fall into five **integration categories** that differ in
how the underlying ODE (if any) is solved. Knowing a model's category tells you
what numerical method runs under the hood — and, downstream, which validation
tolerance applies to it (see :doc:`validation-status`).

.. note::

   These integration categories (how a model is *solved*) rhyme with the A–E
   validation tolerance categories (how parity is *asserted*) but are distinct.
   A model solved by an adaptive RKF45 integrator (integration category A) is
   validated under the adaptive-integrator tolerance (validation category A);
   the analytic ``iaf_psc`` family (integration category B) is validated
   near-exactly (validation category B). They line up by design, but read each in
   its own page.


Category A — adaptive Runge–Kutta–Fehlberg (RKF45)
==================================================

Models with smooth nonlinear sub-threshold dynamics are integrated with an
adaptive Runge–Kutta–Fehlberg step (``AdaptiveRungeKuttaStep``), which adapts the
internal step to control local error.

- Adaptive exponential IAF — ``aeif_*`` (e.g. ``aeif_cond_exp``, ``aeif_psc_alpha``).
- Generalized IF — ``gif_*``.
- Generalized LIF (Allen Institute) — ``glif_*``.
- Conductance-based IAF — ``iaf_cond_*`` (e.g. ``iaf_cond_alpha``, ``iaf_cond_exp``).


Category B — analytic exact propagators
=======================================

Linear current-based IAF neurons have closed-form solutions over a fixed step, so
they use **exact analytical propagators** — no Runge–Kutta iteration is needed.
This is the most numerically faithful family (it matches NEST to ``~1e-6 mV``).

- Current-based IAF — ``iaf_psc_*`` (e.g. ``iaf_psc_alpha``, ``iaf_psc_exp``,
  ``iaf_psc_delta``).


Category C — Hodgkin–Huxley family (RKF45)
==========================================

The Hodgkin–Huxley neurons carry stiff gating dynamics and are also integrated
with the adaptive Runge–Kutta–Fehlberg step.

- Hodgkin–Huxley — ``hh_psc_*``, ``hh_cond_*`` (e.g. ``hh_cond_exp_traub``).
- ``ht_neuron``.


Category D — vectorized rate updates
====================================

Rate-based models have no spike-resolved ODE; their state evolves with a
vectorized update pattern applied across the population each step.

- Linear / nonlinear rate units — ``lin_rate_*``, ``tanh_rate_*``,
  ``sigmoid_rate_*``, ``threshold_lin_rate_*``.
- ``siegert_neuron`` (mean-field diffusion rate).


Category E — no ODE integration
===============================

Devices and event-driven elements carry no differential equation; they are
updated by discrete rules.

- Devices — generators, recorders, and detectors.
- Static synapses and gap junctions.
- Plasticity rules — STP, STDP, and voltage-based learning rules (the weight
  update is an event-driven kernel, not an integrated ODE; see
  :doc:`divergences/stdp`).


See also
========

- :doc:`validation-status` — the tolerance category and parity test for each family.
- :doc:`models` — the model directory by family.
- :doc:`/apis/nest-neurons`, :doc:`/apis/nest-devices` — the API reference.
- `NEST simulator documentation <https://nest-simulator.readthedocs.io/>`_ — the
  authoritative reference for upstream model semantics.
