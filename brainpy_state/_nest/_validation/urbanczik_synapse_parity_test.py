# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for the rebuilt dendritic ``urbanczik_synapse`` (cluster-21).

Validates the frozen spec + pure ``update`` rule on the
:class:`VoltageCoupledPlasticProj` substrate against a live NEST
``urbanczik_synapse`` + ``pp_cond_exp_mc_urbanczik``. The validation chain
(see :mod:`brainpy_state._nest._validation._urbanczik_drive`):

1. **Dendrite parity.** ``V_d`` (the rule's only neuron input, through
   ``V_W_star``/``delta_Pi``) matches NEST sample-for-sample (the dendrite is a
   passive RC; identical modulo a recorder offset).
2. **Rule-input wiring.** The neuron's recorded ``V_W_star`` and ``delta_Pi`` equal
   the closed-form NEST functions of ``V_d`` to machine precision — so, with (1),
   they match NEST transitively.
3. **Weight-trajectory parity.** A presynaptic train through one plastic dendritic
   edge depresses the weight; sampled at NEST's ``weight_recorder`` send steps the
   whole trajectory matches NEST (the online every-step weight coincides with NEST's
   every-send weight there — see the drive module docstring).

The regime holds the soma **silent** with a strong hyperpolarising current so
``delta_Pi`` is the deterministic ``-phi(V_W*)*dt*h(V_W*)`` branch (potentiation
needs somatic spikes, which are stochastic in this point-process neuron; the rule's
positive-``delta_Pi`` branch is covered deterministically by the kernel unit tests
in ``urbanczik_synapse_test.py`` and exercised end-to-end by the demo).
"""
import unittest

import jax
import numpy as np
import brainstate
import saiunit as u

# Pin float64 and (in setUpClass) evict the import-time JAX cache, so this test is
# precision-stable regardless of collection order. brainpy_state traces some kernels
# at import; under pytest that happens during collection before x64 is enabled, so
# the kernels get cached in float32. If another test then flips x64 on, the neuron
# runs in a mixed state and the weight trajectory drifts past the band. (The silent
# regime is float32-stable in isolation; this guards the mixed state.)
jax.config.update("jax_enable_x64", True)
brainstate.environ.set(precision=64, platform="cpu")

from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _urbanczik_drive as drv

# Dendrite is a passive RC -> matches NEST to ~0 modulo a (measured) 2-step recorder
# offset on the sharp synaptic onset; align_steps=2 absorbs it.
_VD_BAND = tc.TraceTolerance(2e-2 * u.mV, 1e-3, align_steps=2, label="urbanczik-Vd",
                             note="dendritic V_d; passive RC, matches NEST modulo recorder offset")
# Somatic compartment sanity (secondary: the rule reads only the dendritic delta_Pi).
# A stable ~0.13 mV peak difference in the coupled soma ODE, independent of the clamp.
_VS_BAND = tc.TraceTolerance(0.3 * u.mV, 1e-3, align_steps=2, label="urbanczik-Vs",
                             note="somatic V_s sanity; coupled-ODE peak diff ~0.13 mV")
# Online every-step weight vs NEST every-send weight: they coincide at the send steps
# (observed max |Δ| ~0.016 pA over the 36-send depression trajectory). Bare pA mantissas.
_WEIGHT_BAND = tc.TraceTolerance(0.1, 2e-3, label="urbanczik",
                                 note="online every-step vs NEST every-send; coincide at send steps")
# V_W_star / delta_Pi recorded == closed-form on V_d (same arithmetic; machine-exact).
_CONSISTENCY_ATOL = 1e-6


@requires_nest
class TestUrbanczikSynapseParity(unittest.TestCase):
    """Live-NEST parity for the rebuilt dendritic ``urbanczik_synapse``."""

    @classmethod
    def setUpClass(cls):
        # evict any float32 kernels cached at import (before x64) so the neuron
        # re-traces in float64 regardless of collection order -- see module header.
        jax.clear_caches()
        brainstate.environ.set(dt=drv.DT * u.ms)
        # 1. neuron / dendrite traces (static dendritic drive, soma clamped silent)
        cls.nt, cls.ntr, cls.n_nrn_spk = drv.nest_neuron_trace()
        cls.ot, cls.otr, cls.o_nrn_spk = drv.our_neuron_trace()
        # 2. depression weight trajectory (plastic dendritic edge, soma clamped silent)
        cls.nwt, cls.nw, cls.n_w_spk = drv.nest_weight_traj()
        cls.owt, cls.ow, cls.o_w_spk = drv.our_weight_traj()
        cls.w_samp = drv.sample_at_send_steps(cls.owt, cls.ow, cls.nwt)

    # -- regime preconditions ----------------------------------------------
    def test_neuron_regime_is_silent_and_depolarizing(self):
        # the dendritic drive must depolarize V_d (so delta_Pi is non-trivial) while
        # the soma stays silent and subthreshold (deterministic delta_Pi branch).
        self.assertEqual(self.n_nrn_spk, 0, "NEST soma must stay silent")
        self.assertEqual(self.o_nrn_spk, 0, "our soma must stay silent")
        self.assertGreater(float(self.ntr["V_d"].max()), drv.E_L + 1.0,
                           "dendritic drive must depolarize V_d")
        self.assertLess(float(self.ntr["V_s"].max()), drv.THETA,
                        "soma must stay subthreshold (V_s < theta)")

    def test_weight_regime_is_silent(self):
        self.assertEqual(self.n_w_spk, 0, "NEST soma must stay silent in the weight regime")
        self.assertEqual(self.o_w_spk, 0, "our soma must stay silent in the weight regime")

    # -- 1. dendrite parity (the rule's neuron input) ----------------------
    def test_dendritic_voltage_parity(self):
        compare_trace(self.ntr["V_d"], self.otr["V_d"],
                      tol=_VD_BAND, metric="urbanczik V_d").assert_()

    def test_somatic_voltage_parity(self):
        compare_trace(self.ntr["V_s"], self.otr["V_s"],
                      tol=_VS_BAND, metric="urbanczik V_s").assert_()

    # -- 2. rule-input wiring: V_W_star / delta_Pi == closed-form(V_d) ------
    def test_v_w_star_matches_closed_form(self):
        # our recorded star potential == (E_L g_L + V_d g_sp)/(g_L+g_sp) (NEST's formula);
        # with V_d == NEST, this is V_W_star == NEST transitively.
        np.testing.assert_allclose(self.otr["V_W_star"], drv.v_w_star(self.otr["V_d"]),
                                   atol=_CONSISTENCY_ATOL,
                                   err_msg="recorded V_W_star must equal closed-form on V_d")

    def test_delta_pi_matches_silent_branch(self):
        # silent soma -> delta_Pi == (0 - phi(V_W*)*dt) * h(V_W*) (NEST write_urbanczik_history).
        np.testing.assert_allclose(self.otr["delta_Pi"], drv.delta_pi_silent(self.otr["V_d"]),
                                   atol=_CONSISTENCY_ATOL,
                                   err_msg="recorded delta_Pi must equal the silent-soma branch")

    def test_delta_pi_is_depression_signed(self):
        # a driven dendrite over a silent soma predicts a higher rate than emitted ->
        # delta_Pi <= 0 everywhere (and strictly negative somewhere).
        self.assertLessEqual(float(self.otr["delta_Pi"].max()), 0.0, "delta_Pi must be <= 0")
        self.assertLess(float(self.otr["delta_Pi"].min()), 0.0, "delta_Pi must depress somewhere")

    # -- 3. weight-trajectory parity ---------------------------------------
    def test_weight_depresses(self):
        self.assertLess(float(self.nw[-1]), drv.INIT_W, "NEST sanity: weight depresses")
        self.assertLess(float(self.w_samp[-1]), drv.INIT_W, "ours must depress too")
        self.assertEqual(int(np.sign(self.w_samp[-1] - drv.INIT_W)),
                         int(np.sign(self.nw[-1] - drv.INIT_W)), "depression sign must match NEST")

    def test_weight_trajectory_matches_nest_at_send_steps(self):
        # the full 36-send depression curve, our weight sampled at NEST's send steps.
        compare_trace(self.nw, self.w_samp,
                      tol=_WEIGHT_BAND, metric="urbanczik weight trajectory").assert_()

    def test_weight_monotonic_depression(self):
        # silent-soma depression is monotone (delta_Pi <= 0 throughout) on both sides.
        self.assertTrue(np.all(np.diff(self.nw) <= 1e-9), "NEST weight not monotone down")
        self.assertTrue(np.all(np.diff(self.w_samp) <= 1e-9), "our send-sampled weight not monotone down")


if __name__ == "__main__":
    unittest.main()
