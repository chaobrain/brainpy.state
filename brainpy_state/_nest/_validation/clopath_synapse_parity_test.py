# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: voltage-based ``clopath_synapse`` (primitive #2).

The Clopath rule reads the postsynaptic neuron's analog voltages every step, so
parity is established in three layers (the shared drive lives in
:mod:`brainpy_state._nest._validation._clopath_drive`):

1. **Neuron voltage parity (precondition).** A subthreshold ``I_e`` depolarization
   of the ``aeif_psc_delta_clopath`` post — the analog states the synapse reads
   (``V`` / ``u_bar_plus`` / ``u_bar_minus``) — matches NEST's multimeter
   sample-for-sample (category A, here ``< 1e-3`` mV). Validate the source of the
   reads before trusting the weight trajectory.
2. **Spike-pairing weight parity.** The canonical
   ``clopath_synapse_spike_pairing.py`` protocol (5 post-pre + 5 pre-post trains,
   10–50 Hz). The substrate reads the post State online with a one-step lag where
   NEST defers potentiation and ring-buffers the voltages by ``delay_u_bars``; we
   align ``delay_u_bars`` to one step. The **stored weight** then matches NEST
   within a documented band (pure depression near-exact; potentiation within 5 %,
   growing with pairing frequency), and the **direction** and **frequency-ordering**
   match exactly.
3. **Voltage-clamp LTD sanity.** A post held in the depression band
   (``theta_minus < V < theta_plus``) by a constant current: every presynaptic
   spike depresses and none potentiate — pure voltage-gated LTD, matched to NEST.

The remaining potentiation gap is the structural online-instantaneous-read vs
NEST-deferred-history divergence (characterised in the spec); it is *not* a kernel
error — the same formula on NEST's own recorded voltages reproduces NEST's weight
to within a few percent.
"""
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc
from brainpy_state._nest._validation import _clopath_drive as drv

# Neuron analog-state precondition: subthreshold, smooth, sample-for-sample (cat. A).
_NEURON_TOL = tc.CAT_A
# Documented Clopath stored-weight band (see _clopath_drive module docstring):
# online instantaneous-read vs NEST deferred-history + one-step event-delivery lag.
# Pure depression is near-exact; potentiation stays within 5 % (observed max ~3.3 %,
# growing with pairing frequency). Weights are bare mV mantissas (init 0.5).
_WEIGHT_BAND = tc.TraceTolerance(2e-3, 5e-2, label="clopath",
                                 note="online instantaneous-read vs NEST deferred history")
_PURE_LTD_TOL = tc.TraceTolerance(1e-3, 5e-3, label="clopath-LTD",
                                  note="pure voltage-gated depression path (near-exact)")
_CLAMP_LTD_AMP = 250.0   # pA; holds V ~ -62 mV (in the theta_minus..theta_plus band)
# Net |Δw| below this is the LTD/LTP crossover: a train whose depression and
# frequency-driven potentiation cancel (the post-pre 30 Hz train sits here, net
# Δw ~ 1e-4 on both sides). Its strict sign is meaningless; it is checked by
# magnitude (both neutral) instead, while its weight still matches NEST in-band.
_CROSSOVER_EPS = 3e-3


@requires_nest
class TestClopathSynapseParity(unittest.TestCase):
    """Live-NEST parity for the rebuilt voltage-based ``clopath_synapse``."""

    @classmethod
    def setUpClass(cls):
        brainstate.environ.set(dt=drv.DT * u.ms)
        # 1. neuron analog-state traces (subthreshold dc, no spikes)
        cls.nrn_t_nest, cls.nrn_nest = drv.nest_neuron_trace(I_e=250.0, T=80.0)
        cls.nrn_t_our, cls.nrn_our = drv.our_neuron_trace(I_e=250.0, T=80.0)
        # 2. canonical spike-pairing final weights (per train)
        cls.nest_w = np.array([drv.nest_pairing_weight(sp, sq)
                               for sp, sq in zip(drv.SPIKE_TIMES_PRE, drv.SPIKE_TIMES_POST)])
        cls.our_w = np.array([drv.our_pairing_weight(sp, sq)
                              for sp, sq in zip(drv.SPIKE_TIMES_PRE, drv.SPIKE_TIMES_POST)])
        # 3. voltage-clamp LTD: 5 pre spikes (40 ms ISI) onto a current-held post
        clamp_pre = [20.0, 60.0, 100.0, 140.0, 180.0]
        cls.clamp_nest = drv.nest_clamp_weight(_CLAMP_LTD_AMP, clamp_pre, T=220.0)
        cls.clamp_our = drv.our_clamp_weight(_CLAMP_LTD_AMP, clamp_pre, T=220.0)

    # -- 1. neuron analog-state precondition -------------------------------
    def test_neuron_voltage_parity_subthreshold(self):
        for name in ("V_m", "u_bar_plus", "u_bar_minus"):
            with self.subTest(state=name):
                compare_trace(self.nrn_nest[name], self.nrn_our[name],
                              tol=_NEURON_TOL, metric=f"clopath {name}").assert_()

    def test_neuron_drive_is_actually_subthreshold(self):
        # guards the precondition: the dc drive must depolarize without spiking,
        # so V stays in (theta_minus, theta_plus) and the traces stay smooth.
        vmax = float(np.max(self.nrn_nest["V_m"]))
        self.assertLess(vmax, drv.NRN_PARAMS["theta_plus"], "drive must stay subthreshold")
        self.assertGreater(vmax, drv.NRN_PARAMS["theta_minus"], "drive must depolarize")

    # -- 2a. pure depression train is near-exact ---------------------------
    def test_spike_pairing_pure_ltd_matches_nest(self):
        # lowest-frequency post-before-pre train: a clean depression with no
        # frequency-driven LTP contamination -> the LTD path is near-exact.
        i = drv.LTD_TRAINS[0]
        self.assertLess(self.nest_w[i], drv.INIT_W, "NEST sanity: train depresses")
        self.assertLess(self.our_w[i], drv.INIT_W, "ours must depress too")
        compare_trace(self.nest_w[i], self.our_w[i],
                      tol=_PURE_LTD_TOL, metric="clopath pure-LTD").assert_()

    # -- 2b. every train's stored weight matches NEST in-band ---------------
    def test_spike_pairing_all_trains_within_band(self):
        for i, lab in enumerate(drv.TRAIN_LABELS):
            with self.subTest(train=lab):
                compare_trace(self.nest_w[i], self.our_w[i],
                              tol=_WEIGHT_BAND, metric=f"clopath {lab}").assert_()

    # -- 2c. every clearly-directional train has NEST's sign ---------------
    def test_spike_pairing_direction_matches_nest(self):
        nest_dw = self.nest_w - drv.INIT_W
        our_dw = self.our_w - drv.INIT_W
        for i, lab in enumerate(drv.TRAIN_LABELS):
            with self.subTest(train=lab):
                if abs(nest_dw[i]) <= _CROSSOVER_EPS:
                    # LTD/LTP crossover (net Δw ~ 0): require ours neutral too.
                    self.assertLessEqual(
                        abs(our_dw[i]), _CROSSOVER_EPS,
                        f"{lab}: NEST neutral (Δw={nest_dw[i]:.2e}) but ours Δw={our_dw[i]:.2e}")
                else:
                    self.assertEqual(
                        int(np.sign(our_dw[i])), int(np.sign(nest_dw[i])),
                        f"{lab}: net Δw sign must match NEST "
                        f"(nest {self.nest_w[i]:.6f}, ours {self.our_w[i]:.6f})")

    # -- 2d. potentiation trains potentiate (within the documented band) ---
    def test_spike_pairing_potentiation_within_band(self):
        for i in drv.LTP_TRAINS:
            with self.subTest(train=drv.TRAIN_LABELS[i]):
                self.assertGreater(self.our_w[i], drv.INIT_W, "pre-post train must potentiate")
                self.assertGreater(self.nest_w[i], drv.INIT_W)
                compare_trace(self.nest_w[i], self.our_w[i],
                              tol=_WEIGHT_BAND, metric=f"clopath {drv.TRAIN_LABELS[i]}").assert_()

    # -- 2e. potentiation grows monotonically with pairing frequency -------
    def test_spike_pairing_potentiation_monotonic_in_frequency(self):
        nest_dw = self.nest_w[list(drv.LTP_TRAINS)] - drv.INIT_W
        our_dw = self.our_w[list(drv.LTP_TRAINS)] - drv.INIT_W
        self.assertTrue(np.all(np.diff(nest_dw) > 0), f"NEST LTP not monotone: {nest_dw}")
        self.assertTrue(np.all(np.diff(our_dw) > 0), f"our LTP not monotone: {our_dw}")

    # -- 3. voltage-clamp LTD sanity ---------------------------------------
    def test_voltage_clamp_ltd_matches_nest(self):
        # sustained depolarization in (theta_minus, theta_plus): pre spikes depress,
        # nothing potentiates (V never reaches theta_plus) -> weight monotone down.
        self.assertLess(self.clamp_nest, drv.INIT_W, "NEST: clamp LTD depresses")
        self.assertLess(self.clamp_our, drv.INIT_W, "ours: clamp LTD depresses")
        compare_trace(self.clamp_nest, self.clamp_our,
                      tol=_PURE_LTD_TOL, metric="clopath clamp-LTD").assert_()


if __name__ == "__main__":
    unittest.main()
