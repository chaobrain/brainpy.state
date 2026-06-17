.. _nest-models:

===============
Model directory
===============

.. currentmodule:: brainpy.state

The NEST-compatible neuron models grouped by family. Each family links to the
:doc:`API reference </apis/nest-neurons>` for the exact signature and to the
upstream `NEST documentation <https://nest-simulator.readthedocs.io/>`_ for the
biophysics.

.. admonition:: We link out for physiology
   :class: seealso

   This page is a *directory*, not a model textbook. We deliberately do **not**
   duplicate NEST's model semantics — for the equations, parameter meanings, and
   references of any model, follow the link to the upstream NEST documentation.
   What lives here is the ``brainpy.state`` surface: which models exist, how they
   are grouped, and how each is integrated (see
   :doc:`integration-categories`).

All models are constructed through the :class:`Simulator`:

.. code-block:: python

   import brainunit as u
   from brainpy import state as bp

   sim = bp.Simulator(dt=0.1 * u.ms)
   pop = sim.create(bp.iaf_psc_alpha, 100, I_e=0. * u.pA)   # a population of 100

Parameters use NEST's names and units (``C_m`` in ``pF``, ``tau_m`` in ``ms``,
``V_th`` in ``mV``, ``I_e`` in ``pA``, …). Pass them as keywords, or as a
``params=dict(...)`` mapping for a large parameter set.


Integrate-and-fire — current-based (``psc``)
============================================

Linear current-based IAF neurons; integrated with exact analytic propagators
(:doc:`integration-categories` category B — the most numerically faithful family).

``iaf_psc_delta``, ``iaf_psc_delta_ps``, ``iaf_psc_alpha``,
``iaf_psc_alpha_multisynapse``, ``iaf_psc_alpha_ps``, ``iaf_psc_exp``,
``iaf_psc_exp_multisynapse``, ``iaf_psc_exp_htum``, ``iaf_psc_exp_ps``,
``iaf_psc_exp_ps_lossless``.

The ``_ps`` ("precise spiking") variants resolve the spike time *between* grid
points; the ``_multisynapse`` variants accept multiple receptor ports.


Integrate-and-fire — conductance-based (``cond``)
=================================================

Conductance-based IAF neurons; integrated with the adaptive RKF45 step
(category A/C).

``iaf_cond_alpha``, ``iaf_cond_alpha_mc``, ``iaf_cond_beta``, ``iaf_cond_exp``,
``iaf_cond_exp_sfa_rr``.

Specialized IAF variants: ``iaf_bw_2001``, ``iaf_bw_2001_exact``,
``iaf_chs_2007``, ``iaf_chxk_2008``, ``iaf_tum_2000``.


Adaptive exponential IF (AdEx)
==============================

Brette–Gerstner adaptive exponential neurons; adaptive RKF45 (category A).

``aeif_cond_alpha``, ``aeif_cond_alpha_astro``, ``aeif_cond_alpha_multisynapse``,
``aeif_cond_beta_multisynapse``, ``aeif_cond_exp``, ``aeif_psc_alpha``,
``aeif_psc_delta``, ``aeif_psc_delta_clopath``, ``aeif_psc_exp``.

``aeif_psc_delta_clopath`` exposes the voltage filters the
:class:`clopath_synapse` reads — see :doc:`divergences/stdp`.


Generalized IF (GIF) and Generalized LIF (GLIF)
===============================================

GIF (Mensi/Pozzorini) and Allen-Institute GLIF neurons; adaptive RKF45
(category A).

GIF: ``gif_cond_exp``, ``gif_cond_exp_multisynapse``, ``gif_pop_psc_exp``,
``gif_psc_exp``, ``gif_psc_exp_multisynapse``.

GLIF: ``glif_cond``, ``glif_psc``, ``glif_psc_double_alpha``.


Multi-timescale adaptive threshold (MAT)
========================================

``mat2_psc_exp``, ``amat2_psc_exp``.


Hodgkin–Huxley family
=====================

Biophysical conductance-based neurons; adaptive RKF45 (category C).

``hh_psc_alpha``, ``hh_psc_alpha_clopath``, ``hh_psc_alpha_gap``,
``hh_cond_exp_traub``, ``hh_cond_beta_gap_traub``, ``ht_neuron``.

The ``_gap`` variants participate in gap-junction networks (see
:doc:`connectivity`).


Izhikevich and point-process neurons
====================================

Izhikevich: ``izhikevich`` (adaptive RKF45, category A).

Point process: ``pp_psc_delta``, ``pp_cond_exp_mc_urbanczik`` (the
two-compartment neuron the Urbanczik dendritic-prediction rule reads).


Rate neurons
============

Rate-based units; vectorized update (category D).

``lin_rate_ipn``, ``lin_rate_opn``, ``tanh_rate_ipn``, ``tanh_rate_opn``,
``sigmoid_rate_ipn``, ``sigmoid_rate_gg_1998_ipn``, ``gauss_rate_ipn``,
``threshold_lin_rate_ipn``, ``threshold_lin_rate_opn``, ``rate_neuron_ipn``,
``rate_neuron_opn``, ``rate_transformer_node``, ``siegert_neuron``.


Binary and other neurons
========================

Binary: ``mcculloch_pitts_neuron``, ``ginzburg_neuron``, ``erfc_neuron``.

Relay / utility: ``parrot_neuron`` (relays its input spikes 1:1 — used to drive
plastic edges or to tap a generator), ``ignore_and_fire`` (a fixed-rate source).


See also
========

- :doc:`/apis/nest-neurons` — the full neuron API with signatures and docstrings.
- :doc:`/apis/nest-base` — the ``NESTNeuron`` base class.
- :doc:`integration-categories` — how each family's ODE (if any) is solved.
- :doc:`connectivity`, :doc:`devices`, :doc:`spatial` — wiring, stimulation, and
  spatial structure.
- `NEST simulator documentation <https://nest-simulator.readthedocs.io/>`_ — the
  authoritative reference for upstream model physiology.
