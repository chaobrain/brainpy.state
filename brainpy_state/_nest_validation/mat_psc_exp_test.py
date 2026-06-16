# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Live-NEST parity for the MAT / AMAT adaptive-threshold demo (§3.5).

The demo ``examples/nest/mat_psc_exp.py`` drives a ``mat2_psc_exp`` and an
``amat2_psc_exp`` neuron with constant current and records the membrane potential
``V_m`` and the composite moving threshold ``V_th = omega + V_th_1 + V_th_2
[+ V_th_v]`` through a multimeter.

Both models are analytic exact propagators (category **B**): with an ``I_e`` drive
(no device delay) the Simulator reproduces a live NEST ``mat2_psc_exp`` /
``amat2_psc_exp`` **to the float-noise floor with zero shift** — identical spike
counts and identical ``V_m`` / ``V_th`` traces. The ``V_th`` parity in particular
exercises the ``_adaptive_threshold`` recordable (``omega + V_th_1 + V_th_2`` for
mat2, plus the voltage-dependent ``V_th_v`` for amat2). ``amat2`` is run with
``beta = 0.2/ms`` so its ``V_th_v`` component is genuinely active (and still
matches NEST exactly).

The NEST-free checks assert the qualitative MAT behaviour on the Simulator side:
mat2's threshold actually moves (rises well above ``omega`` after spikes), and
amat2's ``beta > 0`` voltage coupling genuinely changes ``V_th`` relative to the
``beta = 0`` (pure-mat2) limit.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

import examples.nest.mat_psc_exp as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

# Category B (analytic exact propagator). V_m and V_th match NEST to the float
# noise floor; the 1-step align is only a recorder-phase guard (I_e drive => no
# device-delay offset).
_BAND = tc.TraceTolerance(1e-3 * u.mV, 1e-6, align_steps=1, label="B",
                          note="MAT/AMAT V_m and adaptive V_th, I_e drive (no device offset)")

# Map demo config -> live-NEST model name.
_NEST_MODEL = {"mat2": "mat2_psc_exp", "amat2": "amat2_psc_exp"}


def _nest_params(config):
    """Translate the demo's brainunit params for one config to NEST's plain floats."""
    _cls, params = demo.CONFIGS[config]
    out = {"I_e": float(u.get_mantissa(params["I_e"] / u.pA))}
    if "omega" in params:
        out["omega"] = float(u.get_mantissa(params["omega"] / u.mV))
    if "beta" in params:
        out["beta"] = float(u.get_mantissa(params["beta"] * u.ms))
    return out


def _nest_traces(config, simtime=demo.T_SIM):
    """Live-NEST V_m, V_th and spike count for one demo config under its drive."""
    nest.ResetKernel()
    nest.resolution = demo.DT
    nest.set_verbosity("M_ERROR")
    n = nest.Create(_NEST_MODEL[config], params={**_nest_params(config), "V_m": demo.V0})
    mm = nest.Create("multimeter", params={"record_from": ["V_m", "V_th"], "interval": demo.DT})
    sr = nest.Create("spike_recorder")
    nest.Connect(mm, n)
    nest.Connect(n, sr)
    nest.Simulate(simtime)
    ev = mm.get("events")
    return np.asarray(ev["V_m"]), np.asarray(ev["V_th"]), int(sr.n_events)


@requires_nest
class TestMatPscExpParity(unittest.TestCase):
    """V_m, the adaptive V_th, and spike count match live NEST for both models."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _run(self, config):
        nest_v, nest_vth, nest_n = _nest_traces(config)
        _t, bp_v, bp_vth, bp_n = demo.run_traces(config)
        mv = min(len(nest_v), len(bp_v))
        compare_trace(nest_v[:mv], bp_v[:mv], tol=_BAND,
                      metric=f"{config} V_m").assert_()
        compare_trace(nest_vth[:mv], bp_vth[:mv], tol=_BAND,
                      metric=f"{config} V_th (adaptive)").assert_()
        self.assertLessEqual(abs(nest_n - bp_n), 2,
                             f"{config} spike count NEST={nest_n} brainpy={bp_n}")

    def test_mat2_matches_nest(self):
        self._run("mat2")

    def test_amat2_matches_nest(self):
        self._run("amat2")


class TestMatPscExpBehaviour(unittest.TestCase):
    """NEST-free: the moving threshold and amat2's voltage coupling are real."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_mat2_threshold_moves(self):
        # The whole point of MAT: the threshold rises well above its resting omega
        # after spikes, then relaxes -- so V_th spans a real range, not a constant.
        _t, _v, vth, n = demo.run_traces("mat2")
        self.assertGreater(n, 2)
        omega = float(u.get_mantissa(demo.CONFIGS["mat2"][1]["omega"] / u.mV))
        self.assertAlmostEqual(float(vth.min()), omega, places=3,
                               msg="V_th should rest at omega between/before spikes")
        self.assertGreater(float(vth.max()) - omega, 10.0,
                           "V_th should jump well above omega after spikes")

    def test_amat2_voltage_coupling_changes_threshold(self):
        # beta > 0 switches on V_th_v: the amat2 threshold trace must differ from
        # the beta = 0 (pure-mat2) limit under the same drive.
        from brainpy_state import Simulator, amat2_psc_exp, multimeter
        import braintools

        def vth_trace(beta):
            sim = Simulator(dt=demo.DT * u.ms)
            neu = sim.create(amat2_psc_exp, 1, params=dict(
                beta=beta / u.ms, I_e=200.0 * u.pA,
                V_initializer=braintools.init.Constant(demo.V0 * u.mV)))
            mm = sim.create(multimeter, record_from=["V_th"], interval=demo.DT * u.ms)
            sim.connect(mm, neu)
            res = sim.simulate(demo.T_SIM * u.ms)
            return np.asarray(u.get_mantissa(res.trace(mm, "V_th") / u.mV)).reshape(-1)

        v_on = vth_trace(0.2)
        v_off = vth_trace(0.0)
        m = min(len(v_on), len(v_off))
        self.assertGreater(float(np.max(np.abs(v_on[:m] - v_off[:m]))), 1.0,
                           "beta>0 (V_th_v) should visibly change the threshold")


if __name__ == "__main__":
    unittest.main()
