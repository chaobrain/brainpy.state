# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Persistent-rollout API for the Simulator: ``cont()`` / ``reset_rollout()``.

``simulate()`` re-initialises all state and runs one window from ``t=0``. ``cont()``
is its non-re-initialising sibling: it continues the rollout for ``duration`` so a
host loop (e.g. the §3.10 pong game) can interleave Python work — read recordings,
set ``host_drive`` schedules, overwrite static weights — between 200 ms chunks while
biological time accumulates.

The keystone oracle is NEST-free and deterministic: a constant-current neuron driven
as **one** ``simulate(T)`` must equal the **same** neuron driven as ``K`` back-to-back
``cont(T/K)`` chunks, to machine precision on the voltage trace and exactly on spikes
(the chunk boundaries fall mid-charge and mid-refractory, so persistence across the
boundary — V, refractory counter, synaptic currents — is genuinely exercised). A
live-NEST test then pins ``cont()`` chunking against NEST's own ``nest.Simulate``
accumulation.
"""
import unittest

import braintools
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import iaf_psc_alpha, voltmeter, spike_recorder
from brainpy_state._nest_network import Simulator
from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

# Constant-current neuron whose V_inf = E_L + I_e*tau_m/C_m = -70 + 500*10/250 = -50 mV
# sits ABOVE V_th = -55 mV, so it charges, spikes, and resets periodically — chunk
# boundaries land mid-charge and mid-refractory.
NPAR = dict(C_m=250. * u.pF, tau_m=10. * u.ms, tau_syn_ex=2. * u.ms,
            tau_syn_in=2. * u.ms, t_ref=2. * u.ms, E_L=-70. * u.mV,
            V_reset=-70. * u.mV, V_th=-55. * u.mV,
            V_initializer=braintools.init.Constant(-70. * u.mV))
I_E = 500. * u.pA
DT = 0.1


def _build():
    """A fresh deterministic 1-neuron sim with a voltmeter + spike recorder."""
    sim = Simulator(dt=DT * u.ms)
    neu = sim.create(iaf_psc_alpha, 1, params={**NPAR, 'I_e': I_E})
    vm = sim.create(voltmeter)
    rec = sim.create(spike_recorder)
    sim.connect(vm, neu)            # reversed: recorder observes neuron
    sim.connect(neu, rec)
    return sim, vm, rec


def _v(res, vm):
    return np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV)).reshape(-1)


class TestContOracle(unittest.TestCase):
    """K chunks of cont() reproduce one long simulate() — NEST-free, deterministic."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def test_cont_chunks_equal_single_run(self):
        # One 30 ms run.
        sim_a, vm_a, rec_a = _build()
        res_a = sim_a.simulate(30. * u.ms)
        v_one = _v(res_a, vm_a)
        spk_one = np.asarray(res_a.spikes(rec_a)).reshape(-1)

        # Same neuron as 3 x 10 ms cont() chunks (explicit fresh rollout first).
        sim_b, vm_b, rec_b = _build()
        sim_b.reset_rollout()
        v_parts, spk_parts = [], []
        for _ in range(3):
            r = sim_b.cont(10. * u.ms)
            v_parts.append(_v(r, vm_b))
            spk_parts.append(np.asarray(r.spikes(rec_b)).reshape(-1))
        v_chunked = np.concatenate(v_parts)
        spk_chunked = np.concatenate(spk_parts)

        self.assertEqual(v_one.shape, v_chunked.shape)
        self.assertGreater(int((spk_one > 0).sum()), 0, 'drive should evoke spikes')
        # Same deterministic update + same init -> machine precision / exact.
        self.assertLess(float(np.max(np.abs(v_one - v_chunked))), 1e-9)
        np.testing.assert_array_equal(spk_one > 0, spk_chunked > 0)

    def test_times_are_continuous_across_chunks(self):
        sim_a, vm_a, _ = _build()
        t_one = np.asarray(u.get_mantissa(sim_a.simulate(30. * u.ms).times / u.ms))

        sim_b, vm_b, _ = _build()
        sim_b.reset_rollout()
        t_parts = [np.asarray(u.get_mantissa(sim_b.cont(10. * u.ms).times / u.ms))
                   for _ in range(3)]
        t_chunked = np.concatenate(t_parts)
        self.assertEqual(t_one.shape, t_chunked.shape)
        np.testing.assert_allclose(t_one, t_chunked, atol=1e-9)
        # No gap/overlap at a boundary: step 100 (chunk 2 start) == 10.0 ms.
        self.assertAlmostEqual(float(t_chunked[100]), 10.0, places=9)

    def test_lazy_init_on_first_cont(self):
        # cont() without an explicit reset_rollout() must still initialise once and
        # then reproduce reset_rollout()+cont().
        sim_a, vm_a, _ = _build()
        sim_a.reset_rollout()
        v_reset = np.concatenate([_v(sim_a.cont(10. * u.ms), vm_a) for _ in range(2)])

        sim_b, vm_b, _ = _build()
        v_lazy = np.concatenate([_v(sim_b.cont(10. * u.ms), vm_b) for _ in range(2)])
        self.assertLess(float(np.max(np.abs(v_reset - v_lazy))), 1e-9)

    def test_reset_rollout_restarts_at_zero(self):
        # After running a rollout, reset_rollout() must restart from the SAME initial
        # condition (trace repeats bit-for-bit).
        sim, vm, _ = _build()
        sim.reset_rollout()
        v_first = _v(sim.cont(15. * u.ms), vm)
        sim.reset_rollout()
        v_again = _v(sim.cont(15. * u.ms), vm)
        np.testing.assert_array_equal(v_first, v_again)

    def test_simulate_after_rollout_reinits(self):
        # simulate() still re-initialises even after a cont() rollout has advanced
        # state: its trace equals a standalone simulate() on a fresh sim.
        sim, vm, _ = _build()
        sim.reset_rollout()
        sim.cont(12. * u.ms)                       # advance state
        v_after = _v(sim.simulate(20. * u.ms), vm)

        sim2, vm2, _ = _build()
        v_fresh = _v(sim2.simulate(20. * u.ms), vm2)
        np.testing.assert_array_equal(v_after, v_fresh)


@requires_nest
class TestContLiveNESTAccumulation(unittest.TestCase):
    """``cont()`` chunking matches NEST's own ``nest.Simulate`` accumulation.

    A deterministic constant-current ``iaf_psc_alpha`` is run as 3 x 10 ms windows
    on both sides — brainpy via ``reset_rollout()`` + 3 x ``cont(10ms)``, NEST via
    3 x ``nest.Simulate(10ms)`` (``biological_time`` accumulates). The concatenated
    ``V_m`` traces agree within ``CAT_B_ALIGNED`` (the standard +/-1-step multimeter
    offset), proving the rollout clock + no-re-init path reproduce NEST's continued
    integration across windows.
    """

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def _nest_chunked_vm(self):
        nest.ResetKernel()
        nest.resolution = DT
        nest.set_verbosity("M_ERROR")
        neu = nest.Create("iaf_psc_alpha", 1, params={
            "C_m": 250.0, "tau_m": 10.0, "tau_syn_ex": 2.0, "tau_syn_in": 2.0,
            "t_ref": 2.0, "E_L": -70.0, "V_reset": -70.0, "V_th": -55.0,
            "V_m": -70.0, "I_e": 500.0})
        mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": DT})
        nest.Connect(mm, neu)
        for _ in range(3):
            nest.Simulate(10.0)
        return np.asarray(mm.events["V_m"]).reshape(-1)

    def test_cont_accumulation_matches_nest(self):
        sim, vm, _ = _build()
        sim.reset_rollout()
        v_bp = np.concatenate([_v(sim.cont(10. * u.ms), vm) for _ in range(3)])
        v_nest = self._nest_chunked_vm()
        compare_trace(v_nest, v_bp, tol=tc.CAT_B_ALIGNED,
                      metric="cont() vs NEST 3x10ms V_m").assert_()


if __name__ == '__main__':
    unittest.main()
