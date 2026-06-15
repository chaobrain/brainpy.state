# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: FACETS/BrainScaleS hardware ``stdp_facetshw_synapse_hom``.

A single plastic edge is driven by deterministic pre/post trains and its per-send
weight trajectory must match NEST's ``stdp_facetshw_synapse_hom`` ``weight_recorder``
step-for-step. Unlike the pair-based models the weight is **discrete**: it changes only
at *readout* events (the first pre past each ``readout_cycle_duration`` boundary), when
the controller quantises the weight to a 4-bit LUT index, compares the two accumulated
charges to thresholds, and applies one of three look-up tables. Because the weight lives
on the LUT grid (``k * weight_per_lut_entry``) both sides agree to machine precision once
their charge evaluations align — so every scenario uses the analytic ``CAT_B`` band, with
thresholds chosen to keep every evaluation comfortably off its boundary.

Scenarios (``a_thresh`` lowered so a single clean pair per cycle is decisive, which
cleanly separates the LUT branches; the threshold *value* does not affect NEST/bp
agreement):

* **potentiation climb** — one causal pair per 15 ms cycle drives ``(T,F)`` -> LUT0; the
  discrete weight ratchets up and saturates at ``Wmax``;
* **depression descent** — one acausal pair per cycle drives ``(F,T)`` -> LUT1; the weight
  ratchets down (a leading post-free pre avoids the phantom-pre-at-0 causal seed);
* **saturation boundary** — start at ``Wmax`` (index 15); LUT0 maps 15->15, so the weight
  is pinned (exercises the index clamp / no out-of-range gather);
* **readout-cadence gap** — a multi-cycle gap with no spikes forces ``next_readout`` to
  advance several cycles at once (NEST's ``while`` loop) while charges persist unchanged;
* **native-threshold no-op** — at the default ``a_thresh=21.835`` a sparse train stays
  sub-threshold, so every readout is ``(F,F)`` and only re-quantises the weight.

Shared drive in :mod:`brainpy_state._nest._validation._stdp_drive`.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import stdp_facetshw_synapse_hom
from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _stdp_drive as drv

WPLE = 100.0 / 15.0          # weight_per_lut_entry for Wmax=100, 16-entry LUT


def _syn(**over):
    s = dict(weight=5 * WPLE, Wmax=100.0, tau_plus=20.0, tau_minus=20.0,
             a_thresh_th=0.7, a_thresh_tl=0.7, drt=15.0, spd=50, wple=WPLE)
    s.update(over)
    return s


def _rule(syn):
    return stdp_facetshw_synapse_hom(
        weight=syn['weight'] * u.pA, delay=drv.DEND_D * u.ms,
        tau_plus=syn['tau_plus'] * u.ms, tau_minus=syn['tau_minus'] * u.ms,
        Wmax=syn['Wmax'], a_thresh_th=syn['a_thresh_th'], a_thresh_tl=syn['a_thresh_tl'],
        weight_per_lut_entry=syn['wple'], driver_readout_time=syn['drt'],
        synapses_per_driver=syn['spd'])


def _per_conn(syn):
    # per-synapse params (the *_hom plasticity params are common, set via _common)
    return {"weight": syn['weight'], "a_thresh_th": syn['a_thresh_th'],
            "a_thresh_tl": syn['a_thresh_tl']}


def _common(syn):
    # homogeneous (common) properties; tau_minus is exposed as 'tau_minus_stdp'
    return {"tau_plus": syn['tau_plus'], "tau_minus_stdp": syn['tau_minus'],
            "Wmax": syn['Wmax'], "weight_per_lut_entry": syn['wple'],
            "driver_readout_time": syn['drt'], "synapses_per_driver": syn['spd']}


@requires_nest
class TestStdpFacetshwSynapseHomParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def _run(self, syn, pre_want, post_want, T, tol, label):
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_facetshw_synapse_hom", _per_conn(syn), syn['tau_minus'],
            pre_want, post_want, T, common=_common(syn))
        bp_all = drv.bp_weight_trace(_rule(syn), pre_fire, post_fire, int(round(T / drv.DT)))
        bp_w = bp_all[drv.steps(wr_t)]
        m = min(len(nest_w), len(bp_w))
        self.assertGreater(m, 0, f"{label}: no weight samples")
        self.assertEqual(len(nest_w), len(pre_fire), f"{label}: one send per pre fire")
        compare_trace(nest_w[:m], bp_w[:m], tol=tol, metric=label).assert_()

    # -- potentiation: one causal pair per cycle -> LUT0, weight climbs --------
    def test_potentiation_climb_matches_nest(self):
        pre = [100.0 + 15.0 * k for k in range(12)]      # one pre per 15 ms readout cycle
        post = [p + 0.5 for p in pre]                    # post effect (q+d) lands just after pre
        self._run(_syn(weight=5 * WPLE), pre, post, T=300.0,
                  tol=tc.CAT_B, label="facetshw potentiation climb")

    # -- depression: one acausal pair per cycle -> LUT1, weight descends -------
    def test_depression_descent_matches_nest(self):
        pre = [100.0 + 15.0 * k for k in range(12)]
        # post 2 ms before each pre after the leading one: effect (q+d) lands 1 ms before
        # the pre (acausal). The leading post-free pre seeds pre_seen, so no phantom causal.
        post = [p - 2.0 for p in pre[1:]]
        self._run(_syn(weight=10 * WPLE), pre, post, T=300.0,
                  tol=tc.CAT_B, label="facetshw depression descent")

    # -- saturation: start at Wmax (index 15); LUT0 pins 15 -> 15 --------------
    def test_saturation_boundary_matches_nest(self):
        pre = [100.0 + 15.0 * k for k in range(10)]
        post = [p + 0.5 for p in pre]
        self._run(_syn(weight=15 * WPLE), pre, post, T=260.0,
                  tol=tc.CAT_B, label="facetshw saturation boundary")

    # -- readout cadence: a multi-cycle gap advances next_readout by >1 cycle --
    def test_readout_cadence_gap_matches_nest(self):
        pre = [100.0, 115.0, 130.0, 145.0, 220.0, 235.0, 250.0, 265.0]   # 75 ms gap
        post = [p + 0.5 for p in pre]
        self._run(_syn(weight=5 * WPLE), pre, post, T=300.0,
                  tol=tc.CAT_B, label="facetshw readout cadence gap")

    # -- native threshold: sparse train stays sub-threshold -> (F,F) re-quantise
    def test_native_threshold_noop_matches_nest(self):
        pre = [100.0, 150.0, 200.0, 250.0]
        post = [p + 5.0 for p in pre]
        self._run(_syn(weight=5 * WPLE, a_thresh_th=21.835, a_thresh_tl=21.835),
                  pre, post, T=320.0, tol=tc.CAT_B, label="facetshw native no-op")


if __name__ == "__main__":
    unittest.main()
