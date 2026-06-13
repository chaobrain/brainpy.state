# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST ``weight_recorder`` send-event audit across the plastic-synapse family.

Every plastic rule already has a ``*_parity_test.py`` that checks the *value* of the
weight at each send. This audit adds, **uniformly across the family and through the
formal send-view seam** (:func:`brainpy_state._network.send_steps_from_pre` /
:func:`brainpy_state._network.weight_recorder_events`), the parts those ad-hoc
per-model tests do not assert together:

* **count** — one send-view event per pre ``send``, equal to NEST's
  ``weight_recorder`` event count;
* **timing** — NEST logs at the pre-spike emission step (``e.get_stamp()``, no delay
  offset); the seam's send mask sits on exactly those steps;
* **value** — the post-update weight (or delivered amplitude for ``ht_synapse``)
  masked at the send steps matches NEST's recorded weights within each rule's band;
* the **edge cases** the goal enumerates (empty train, single send, change strictly
  after the last send, ``Wmax`` clamp, ``dt`` invariance of the event *times*).

A deliberately **modest causal protocol** drives each rule (a few sends, not the 5 s
trains the per-model tests own) — count/timing/value are fully exercised by a short
train, and the family-wide sweep stays fast. Params/bands are copied verbatim from
each rule's existing parity test (not re-derived). The shared drive (decoupled
``iaf_psc_delta`` post, recorded fire times, NEST dendritic-delay shift,
online-vs-deferred sampling at the send steps) lives in
:mod:`brainpy_state._nest._validation._stdp_drive`.
"""
import dataclasses
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import (
    stdp_synapse, stdp_synapse_hom, stdp_pl_synapse_hom, jonke_synapse,
    vogels_sprekeler_synapse, stdp_triplet_synapse, ht_synapse,
    stdp_nn_symm_synapse, stdp_nn_restr_synapse, stdp_nn_pre_centered_synapse,
    stdp_facetshw_synapse_hom, stdp_dopamine_synapse,
)
from brainpy_state._network import send_steps_from_pre, weight_recorder_events
from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _stdp_drive as drv
from brainpy_state._nest._validation import _stdp_dopamine_drive as ddrv

_D = drv.DEND_D * u.ms          # the dendritic delay every rule is built with
WPLE = 100.0 / 15.0             # facetshw weight_per_lut_entry (Wmax=100, 16-entry LUT)

# A modest causal pairing shared by the pair-based rows: four LTP pairs (post 10 ms
# after pre, 50 ms ISI). Exercises facilitation + depression at every send without a
# deep train; the per-model tests own the long trajectories.
_PRE = (100.0, 150.0, 200.0, 250.0)
_POST = (110.0, 160.0, 210.0, 260.0)
_T = 350.0


@dataclasses.dataclass(frozen=True)
class Row:
    """One audited plastic rule: its NEST model, a fresh-rule factory, and protocol.

    Parameters
    ----------
    label : str
        Diagnostic name (the ``subTest`` / metric label).
    model : str
        NEST plastic synapse model name passed to
        :func:`brainpy_state._nest._validation._stdp_drive.nest_pair_run`.
    rule : callable
        Zero-argument factory returning a *fresh* plastic rule spec (the drive mutates
        ``rule.delay`` and initialises state, so each run needs a new instance).
    per_conn : dict
        Per-connection NEST synapse parameters (``weight`` plus any per-synapse params).
    post_tau_minus : float
        ``tau_minus`` (ms) for the postsynaptic archiving node.
    pre, post : tuple of float
        Desired pre / post fire times (ms). ``post`` is empty for presynaptic-only
        rules (``ht_synapse``).
    T : float
        Simulation horizon (ms).
    band : TraceTolerance
        Value tolerance (copied from the rule's parity test).
    common : dict, optional
        Common (homogeneous) properties for ``*_hom`` / common-param models.
    post_params : dict, optional
        Extra postsynaptic node params (e.g. ``tau_minus_triplet``).
    delivered : bool, optional
        Sample the *delivered* amplitude (``w * P``) rather than the stored weight —
        ``True`` only for ``ht_synapse`` (whose recorder logs the delivered value).
    """
    label: str
    model: str
    rule: object
    per_conn: dict
    post_tau_minus: float
    pre: tuple
    post: tuple
    T: float
    band: object
    common: dict = None
    post_params: dict = None
    delivered: bool = False


def _group1_rows():
    """Build the Group-1 audit rows (the 11 ``_stdp_drive`` plastic rules)."""
    rows = [
        # -- canonical pair-based STDP (per-connection params) -----------------
        Row("stdp_synapse", "stdp_synapse",
            lambda: stdp_synapse(weight=5.0 * u.pA, delay=_D, tau_plus=20.0 * u.ms,
                                 tau_minus=20.0 * u.ms, lambda_=0.1, alpha=1.0,
                                 mu_plus=1.0, mu_minus=1.0, Wmax=100.0),
            {"weight": 5.0, "Wmax": 100.0, "lambda": 0.1, "alpha": 1.0,
             "mu_plus": 1.0, "mu_minus": 1.0, "tau_plus": 20.0},
            20.0, _PRE, _POST, _T, tc.CAT_B),
        # -- stdp_synapse_hom: plasticity params are NEST common properties ----
        Row("stdp_synapse_hom", "stdp_synapse_hom",
            lambda: stdp_synapse_hom(weight=5.0 * u.pA, delay=_D, tau_plus=20.0 * u.ms,
                                     tau_minus=20.0 * u.ms, lambda_=0.1, alpha=1.0,
                                     mu_plus=1.0, mu_minus=1.0, Wmax=100.0),
            {"weight": 5.0}, 20.0, _PRE, _POST, _T, tc.CAT_B,
            common={"Wmax": 100.0, "lambda": 0.1, "alpha": 1.0, "mu_plus": 1.0,
                    "mu_minus": 1.0, "tau_plus": 20.0}),
        # -- power-law (common params, unbounded above) ------------------------
        Row("stdp_pl_synapse_hom", "stdp_pl_synapse_hom",
            lambda: stdp_pl_synapse_hom(weight=5.0 * u.pA, delay=_D, tau_plus=20.0 * u.ms,
                                        tau_minus=20.0 * u.ms, lambda_=0.1, alpha=1.0, mu=0.4),
            {"weight": 5.0}, 20.0, _PRE, _POST, _T, tc.CAT_B,
            common={"lambda": 0.1, "alpha": 1.0, "mu": 0.4, "tau_plus": 20.0}),
        # -- jonke exponential-weight (common params; mu>0 path) ---------------
        Row("jonke_synapse", "jonke_synapse",
            lambda: jonke_synapse(weight=10.0 * u.pA, delay=_D, tau_plus=20.0 * u.ms,
                                  tau_minus=20.0 * u.ms, lambda_=0.02, alpha=1.0,
                                  mu_plus=0.05, mu_minus=0.02, beta=0.0, Wmax=100.0),
            {"weight": 10.0}, 20.0, _PRE, _POST, _T, tc.CAT_B,
            common={"Wmax": 100.0, "lambda": 0.02, "alpha": 1.0, "mu_plus": 0.05,
                    "mu_minus": 0.02, "beta": 0.0, "tau_plus": 20.0}),
        # -- vogels-sprekeler symmetric inhibitory (post tau_minus == tau) -----
        Row("vogels_sprekeler_synapse", "vogels_sprekeler_synapse",
            lambda: vogels_sprekeler_synapse(weight=0.5 * u.pA, delay=_D, tau=20.0 * u.ms,
                                             eta=0.01, alpha=0.12, Wmax=1.0),
            {"weight": 0.5}, 20.0, _PRE, _POST, _T, tc.CAT_B,
            common={"Wmax": 1.0, "eta": 0.01, "alpha": 0.12, "tau": 20.0}),
        # -- triplet (multi-trace seam; post carries both K- constants) --------
        Row("stdp_triplet_synapse", "stdp_triplet_synapse",
            lambda: stdp_triplet_synapse(
                weight=5.0 * u.pA, delay=_D, tau_plus=16.8 * u.ms,
                tau_plus_triplet=101.0 * u.ms, tau_minus=20.0 * u.ms,
                tau_minus_triplet=110.0 * u.ms, Aplus=0.005, Aplus_triplet=0.005,
                Aminus=0.005, Aminus_triplet=0.005, Wmax=100.0),
            {"weight": 5.0, "tau_plus": 16.8, "tau_plus_triplet": 101.0, "Aplus": 0.005,
             "Aplus_triplet": 0.005, "Aminus": 0.005, "Aminus_triplet": 0.005, "Wmax": 100.0},
            20.0, _PRE, _POST, _T, tc.CAT_B,
            post_params={"tau_minus_triplet": 110.0}),
        # -- ht_synapse: presynaptic-only; recorder logs the DELIVERED w*P -----
        Row("ht_synapse", "ht_synapse",
            lambda: ht_synapse(weight=100.0 * u.pA, delay=_D, tau_P=300.0 * u.ms,
                               delta_P=0.2, P=1.0),
            {"weight": 100.0, "tau_P": 300.0, "delta_P": 0.2, "P": 1.0},
            20.0, (50.0, 70.0, 95.0, 130.0, 200.0), (), 300.0, tc.CAT_B,
            delivered=True),
        # -- nearest-neighbour symmetric (causal protocol dodges phantom-pre@0) -
        Row("stdp_nn_symm_synapse", "stdp_nn_symm_synapse",
            lambda: stdp_nn_symm_synapse(weight=5.0 * u.pA, delay=_D, tau_plus=20.0 * u.ms,
                                         tau_minus=20.0 * u.ms, lambda_=0.1, alpha=1.0,
                                         mu_plus=1.0, mu_minus=1.0, Wmax=100.0),
            {"weight": 5.0, "Wmax": 100.0, "lambda": 0.1, "alpha": 1.0,
             "mu_plus": 1.0, "mu_minus": 1.0, "tau_plus": 20.0},
            20.0, _PRE, _POST, _T, tc.CAT_B),
        # -- nearest-neighbour restricted -------------------------------------
        Row("stdp_nn_restr_synapse", "stdp_nn_restr_synapse",
            lambda: stdp_nn_restr_synapse(weight=5.0 * u.pA, delay=_D, tau_plus=20.0 * u.ms,
                                          tau_minus=20.0 * u.ms, lambda_=0.1, alpha=1.0,
                                          mu_plus=1.0, mu_minus=1.0, Wmax=100.0),
            {"weight": 5.0, "Wmax": 100.0, "lambda": 0.1, "alpha": 1.0,
             "mu_plus": 1.0, "mu_minus": 1.0, "tau_plus": 20.0},
            20.0, _PRE, _POST, _T, tc.CAT_B),
        # -- presynaptic-centered (per-edge Kplus; immune to phantom-pre@0) ----
        Row("stdp_nn_pre_centered_synapse", "stdp_nn_pre_centered_synapse",
            lambda: stdp_nn_pre_centered_synapse(
                weight=5.0 * u.pA, delay=_D, tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms,
                lambda_=0.1, alpha=1.0, mu_plus=1.0, mu_minus=1.0, Wmax=100.0, Kplus=0.0),
            {"weight": 5.0, "Wmax": 100.0, "lambda": 0.1, "alpha": 1.0, "mu_plus": 1.0,
             "mu_minus": 1.0, "tau_plus": 20.0, "Kplus": 0.0},
            20.0, _PRE, _POST, _T, tc.CAT_B),
        # -- FACETS/BrainScaleS discrete LUT: one causal pair per readout cycle -
        Row("stdp_facetshw_synapse_hom", "stdp_facetshw_synapse_hom",
            lambda: stdp_facetshw_synapse_hom(
                weight=5 * WPLE * u.pA, delay=_D, tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms,
                Wmax=100.0, a_thresh_th=0.7, a_thresh_tl=0.7, weight_per_lut_entry=WPLE,
                driver_readout_time=15.0, synapses_per_driver=50),
            {"weight": 5 * WPLE, "a_thresh_th": 0.7, "a_thresh_tl": 0.7},
            20.0, tuple(100.0 + 15.0 * k for k in range(8)),
            tuple(100.0 + 15.0 * k + 0.5 for k in range(8)), 240.0, tc.CAT_B,
            common={"tau_plus": 20.0, "tau_minus_stdp": 20.0, "Wmax": 100.0,
                    "weight_per_lut_entry": WPLE, "driver_readout_time": 15.0,
                    "synapses_per_driver": 50}),
    ]
    return rows


GROUP1_ROWS = _group1_rows()

# stdp_synapse handles for the edge-case methods (the canonical pair-based rule).
_STDP_PER_CONN = {"weight": 5.0, "Wmax": 100.0, "lambda": 0.1, "alpha": 1.0,
                  "mu_plus": 1.0, "mu_minus": 1.0, "tau_plus": 20.0}


def _stdp_rule(weight=5.0, lambda_=0.1):
    return stdp_synapse(weight=weight * u.pA, delay=_D, tau_plus=20.0 * u.ms,
                        tau_minus=20.0 * u.ms, lambda_=lambda_, alpha=1.0,
                        mu_plus=1.0, mu_minus=1.0, Wmax=100.0)


# Dopamine (Group 3) config — common dopamine properties + the documented weight band
# (the online one-step-n-lag vs NEST deferred-integral residual), copied from
# stdp_dopamine_synapse_parity_test.py.
_DOPA_INIT_W = 50.0
_DOPA_TAU_MINUS = 20.0
_DOPA_COMMON = dict(A_plus=1.0, A_minus=1.5, tau_plus=20.0, tau_c=1000.0, tau_n=200.0,
                    b=0.0, Wmin=-1.0e9, Wmax=1.0e9)
_DOPA_BAND = tc.TraceTolerance(3e-2, 2e-3, label="dopamine",
                               note="online one-step-n-lag vs NEST deferred integral")


def _dopamine_rule():
    return stdp_dopamine_synapse(
        weight=_DOPA_INIT_W * u.pA, A_plus=_DOPA_COMMON['A_plus'],
        A_minus=_DOPA_COMMON['A_minus'], tau_plus=_DOPA_COMMON['tau_plus'] * u.ms,
        tau_minus=_DOPA_TAU_MINUS * u.ms, tau_c=_DOPA_COMMON['tau_c'] * u.ms,
        tau_n=_DOPA_COMMON['tau_n'] * u.ms, b=_DOPA_COMMON['b'],
        Wmin=_DOPA_COMMON['Wmin'], Wmax=_DOPA_COMMON['Wmax'])


@requires_nest
class TestWeightRecorderAudit(unittest.TestCase):
    """Send-triggered ``weight_recorder`` parity across the plastic family."""

    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    # -- the count + timing + value triple, through the send-view seam --------
    def _audit(self, row):
        n_steps = int(round(row.T / drv.DT))
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            row.model, row.per_conn, row.post_tau_minus, list(row.pre), list(row.post),
            row.T, common=row.common, post_params=row.post_params)
        bp = drv.bp_weight_trace(row.rule(), pre_fire, post_fire, n_steps,
                                 delivered=row.delivered)

        # Derive the send mask from the realized pre fire train via the seam, and
        # confirm it lands on the integer pre steps (ties send_steps_from_pre to NEST).
        pre_arr = np.zeros((n_steps, 1))
        pre_arr[drv.steps(pre_fire), 0] = 1.0
        send_steps = send_steps_from_pre(pre_arr)
        np.testing.assert_array_equal(
            send_steps, drv.steps(pre_fire), err_msg=f"{row.label}: send mask != pre steps")

        ev_steps, ev_w = weight_recorder_events(bp, send_steps)

        # (count) one event per pre send == NEST weight_recorder event count.
        self.assertEqual(len(ev_steps), len(pre_fire), f"{row.label}: event count vs pre")
        self.assertEqual(len(nest_w), len(pre_fire), f"{row.label}: NEST count vs pre")
        # (timing) NEST stamps at the pre-spike step; our events sit on the same steps.
        np.testing.assert_array_equal(
            drv.steps(wr_t), drv.steps(pre_fire), err_msg=f"{row.label}: NEST stamp != pre")
        np.testing.assert_array_equal(
            ev_steps, drv.steps(wr_t), err_msg=f"{row.label}: event steps != NEST stamps")
        # (value) the masked send-view reproduces NEST's recorded weights.
        m = min(len(nest_w), len(ev_w))
        self.assertGreater(m, 0, f"{row.label}: no events sampled")
        compare_trace(nest_w[:m], ev_w[:m], tol=row.band, metric=row.label).assert_()

    def test_send_event_count_timing_value_across_family(self):
        for row in GROUP1_ROWS:
            with self.subTest(rule=row.label):
                self._audit(row)

    # -- edge case: no pre spikes -> zero events (recorder logs nothing) -------
    def test_empty_pre_train_zero_events(self):
        n_steps = int(round(200.0 / drv.DT))
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_synapse", _STDP_PER_CONN, 20.0, [], [100.0], 200.0)
        bp = drv.bp_weight_trace(_stdp_rule(), pre_fire, post_fire, n_steps)
        ev_steps, ev_w = weight_recorder_events(bp, send_steps_from_pre(np.zeros((n_steps, 1))))
        self.assertEqual(len(nest_w), 0, "NEST must log no events without a send")
        self.assertEqual(len(ev_steps), 0)
        self.assertEqual(len(ev_w), 0)

    # -- edge case: a single send -> one event, value matches -----------------
    def test_single_send_value_matches(self):
        n_steps = int(round(300.0 / drv.DT))
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_synapse", _STDP_PER_CONN, 20.0, [200.0], [100.0], 300.0)
        bp = drv.bp_weight_trace(_stdp_rule(), pre_fire, post_fire, n_steps)
        ev_steps, ev_w = weight_recorder_events(bp, drv.steps(pre_fire))
        self.assertEqual(len(ev_steps), 1)
        self.assertEqual(len(nest_w), 1)
        compare_trace(nest_w, ev_w, tol=tc.CAT_B, metric="single send").assert_()

    # -- edge case: a change strictly after the last send is invisible --------
    def test_change_after_last_send_is_invisible(self):
        # post@300 fires after the last pre send@200: NEST never logs its (deferred)
        # LTP, and the send-view misses it too; but the eager bp trace tail carries it.
        n_steps = int(round(350.0 / drv.DT))
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_synapse", _STDP_PER_CONN, 20.0, [100.0, 200.0], [110.0, 210.0, 300.0], 350.0)
        bp = drv.bp_weight_trace(_stdp_rule(), pre_fire, post_fire, n_steps)
        ev_steps, ev_w = weight_recorder_events(bp, drv.steps(pre_fire))
        # send-view matches NEST at every send ...
        compare_trace(nest_w, ev_w, tol=tc.CAT_B, metric="tail-invisible").assert_()
        # ... yet the post-last-send LTP is present in the final trace, absent from events.
        self.assertGreater(abs(float(bp[-1]) - float(ev_w[-1])), 1e-5,
                           "the tail change must move the final weight")

    # -- edge case: Wmax clamp -> the recorded (masked) value is clamped -------
    def test_wmax_clamp_recorded_value_matches(self):
        # strong LTP near the bound: the masked send-view saturates at Wmax exactly
        # like NEST's recorder.
        n_steps = int(round(300.0 / drv.DT))
        pre = [100.0, 130.0, 160.0, 190.0, 220.0, 250.0]
        post = [p + 5.0 for p in pre]
        pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
            "stdp_synapse", {**_STDP_PER_CONN, "weight": 80.0, "lambda": 0.5},
            20.0, pre, post, 300.0)
        bp = drv.bp_weight_trace(_stdp_rule(weight=80.0, lambda_=0.5),
                                 pre_fire, post_fire, n_steps)
        ev_steps, ev_w = weight_recorder_events(bp, drv.steps(pre_fire))
        compare_trace(nest_w, ev_w, tol=tc.CAT_B, metric="Wmax clamp").assert_()
        self.assertLessEqual(float(np.max(ev_w)), 100.0 + 1e-9, "must not exceed Wmax")

    # -- edge case: the event TIME (ms) is dt-invariant; the step index scales --
    def test_dt_invariance_of_event_times_live(self):
        seen = []
        for dt in (0.1, 0.05):
            old = drv.DT
            try:
                drv.DT = dt
                brainstate.environ.set(dt=dt * u.ms)
                n_steps = int(round(200.0 / dt))
                pre_fire, post_fire, nest_w, wr_t = drv.nest_pair_run(
                    "stdp_synapse", _STDP_PER_CONN, 20.0, [100.0], [60.0], 200.0)
                bp = drv.bp_weight_trace(_stdp_rule(), pre_fire, post_fire, n_steps)
                ev_steps, _ = weight_recorder_events(bp, drv.steps(pre_fire))
                seen.append((float(wr_t[0]), int(ev_steps[0]), float(ev_steps[0]) * dt))
            finally:
                drv.DT = old
                brainstate.environ.set(dt=old * u.ms)
        # NEST stamp time and our send-step time are ~100 ms on both grids ...
        for wr_ms, _step, ev_ms in seen:
            self.assertAlmostEqual(wr_ms, 100.0, places=6)
            self.assertAlmostEqual(ev_ms, 100.0, places=6)
        # ... while the integer step index doubles when dt halves.
        self.assertEqual(seen[1][1], 2 * seen[0][1])

    # -- Group 3: dopamine-modulated stdp_dopamine_synapse --------------------
    def test_dopamine_send_event_count_timing_value(self):
        # A modest sustained LTP pairing under a steady dopa train: the broadcast n
        # converts the eligibility trace into a rising weight, logged at every pre send.
        pre = list(np.arange(60.0, 400.0, 60.0))
        post = [p + 10.0 for p in pre]
        dopa = list(np.arange(30.0, 400.0, 50.0))
        T = 420.0
        n_steps = int(round(T / ddrv.DT))
        pre_fire, post_fire, nest_w, wr_t, _final = ddrv.nest_dopamine_run(
            dict(weight=_DOPA_INIT_W), _DOPA_COMMON, _DOPA_TAU_MINUS, pre, post, dopa, T)
        w_trace = ddrv.bp_dopamine_weight_trace(
            _dopamine_rule(), pre_fire, post_fire, dopa, n_steps, tau_n=_DOPA_COMMON['tau_n'])

        pre_arr = np.zeros((n_steps, 1))
        pre_arr[ddrv.steps(pre_fire), 0] = 1.0
        send_steps = send_steps_from_pre(pre_arr)
        np.testing.assert_array_equal(
            send_steps, ddrv.steps(pre_fire), err_msg="dopamine: send mask != pre steps")

        ev_steps, ev_w = weight_recorder_events(w_trace, send_steps)
        # (count) one event per pre send == NEST weight_recorder event count.
        self.assertEqual(len(ev_steps), len(pre_fire), "dopamine: event count vs pre")
        self.assertEqual(len(nest_w), len(pre_fire), "dopamine: NEST count vs pre")
        # (timing) NEST stamps at the pre-spike step; events sit on the same steps.
        np.testing.assert_array_equal(
            ddrv.steps(wr_t), ddrv.steps(pre_fire), err_msg="dopamine: NEST stamp != pre")
        np.testing.assert_array_equal(
            ev_steps, ddrv.steps(wr_t), err_msg="dopamine: event steps != NEST stamps")
        # (value) the masked online integral reproduces NEST's recorded weights in-band.
        compare_trace(nest_w, ev_w, tol=_DOPA_BAND, metric="dopamine audit").assert_()
        self.assertGreater(float(ev_w[-1]), _DOPA_INIT_W, "dopamine LTP must raise the weight")


if __name__ == "__main__":
    unittest.main()
