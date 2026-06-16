# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spike-driven conductance parity for ``iaf_chxk_2008`` (goal 25, alpha kinetics, migration).

``iaf_chxk_2008`` (Casti-Hayot-Xiao-Kaplan 2008 retinal-ganglion conductance IAF with an
intrinsic spike-triggered after-hyperpolarisation) carried bespoke ``update(w_ex=, w_in=)``
keyword arguments but was **never enrolled in the Simulator's multi-receptor bridge** -- it had
no ``n_receptors``, so ``connect(receptor_type=...)`` raised ``AttributeError`` and the spike ->
conductance path through the Simulator was dead.

This module validates the migration: the stale ``w_ex``/``w_in`` kwargs are replaced by the
canonical ``w_by_rec`` dual-path arm (17b multi-receptor bridge: ``n_receptors=2``,
``receptor_input_unit=u.nS``), so ``receptor_type=1`` -> ``g_ex``, ``receptor_type=2`` -> ``g_in``
with positive ``nS`` = NEST's weight-**sign** routing. The *alpha* synapse applies
``dg_ex += (e/tau_syn_ex)*w_ex``; only the *source* of ``w_ex``/``w_in`` changed (bridge vs
self-pull) -- the kinetics and the intrinsic AHP are untouched.

Weights are subthreshold (``V_th=-45 mV``; the low-impedance ``g_L=100 nS`` cell stays near
``E_L=-60 mV``) so the AHP/threshold machinery stays inert and the conductance traces match NEST.
"""
import gc
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

try:
    import nest
except Exception:                                   # pragma: no cover - env dependent
    nest = None

from brainpy import state as bp
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

DT = 0.1
E_L = -60.0
VM_TOL = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=3, label='A',
                        note='iaf_chxk_2008 V_m (spike-driven) vs live NEST')
COND_TOL = TraceTolerance(1e-3, 1e-3, align_steps=3, label='A',
                          note='iaf_chxk_2008 g_ex/g_in vs live NEST')

#: Subthreshold weights into the low-impedance (g_L=100 nS) cell: V_m peaks ~-56 mV, well below
#: V_th=-45 mV, so the spike-triggered AHP never engages and the alpha conductance matches NEST.
EXC_SPIKES = [10., 12., 14., 16.]
EXC_W = 8.0          # nS per spike (receptor 1 / NEST +weight); E_ex=+20
INH_SPIKES = [10., 13., 16., 19.]
INH_W = 20.0         # nS per spike (receptor 2 / NEST -weight); E_in=-90
SIM_T = 50.0


def _ms(x):
    """Strip units to a flat float64 ndarray (a recorded trace mantissa)."""
    return np.asarray(u.get_mantissa(x), dtype=float).reshape(-1)


def _bp_drive(spike_times, weight, receptor_type, sim_time=SIM_T):
    """brainpy: spike_generator -> iaf_chxk_2008; return (times, V_m, g_ex, g_in)."""
    brainstate.environ.set(dt=DT * u.ms)
    sim = bp.Simulator(dt=DT * u.ms)
    n = sim.create(bp.iaf_chxk_2008, 1)
    if len(spike_times):
        sg = sim.create(bp.spike_generator, 1, spike_times=np.asarray(spike_times) * u.ms)
        sim.connect(sg, n, weight=weight * u.nS, delay=DT * u.ms, receptor_type=receptor_type)
    mm = sim.create(bp.multimeter, record_from=['V_m', 'g_ex', 'g_in'])
    sim.connect(mm, n)
    res = sim.simulate(sim_time * u.ms)
    return (res.times, res.trace(mm, 'V_m'), res.trace(mm, 'g_ex'), res.trace(mm, 'g_in'))


class TestIafChxk2008ConductanceLaw(unittest.TestCase):
    """Spike-driven conductance invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_excitatory_train_raises_g_ex_and_depolarises(self):
        """``receptor_type=1`` delivers excitatory conductance: g_ex>0, V_m above E_L."""
        _t, v, g_ex, g_in = _bp_drive(EXC_SPIKES, EXC_W, receptor_type=1)
        v, g_ex, g_in = _ms(v), _ms(g_ex), _ms(g_in)
        self.assertGreater(float(np.max(g_ex)), 0.0, 'excitatory conductance is delivered')
        self.assertTrue(np.allclose(g_in, 0.0), 'no inhibitory conductance on receptor 1')
        self.assertGreater(float(np.max(v)), E_L + 1.0, 'the membrane depolarises above E_L')
        self.assertLess(float(np.max(v)), -45.0, 'the drive stays subthreshold (AHP stays inert)')

    def test_inhibitory_train_raises_g_in_and_hyperpolarises(self):
        """``receptor_type=2`` delivers inhibitory conductance: g_in>0, V_m below E_L."""
        _t, v, g_ex, g_in = _bp_drive(INH_SPIKES, INH_W, receptor_type=2)
        v, g_ex, g_in = _ms(v), _ms(g_ex), _ms(g_in)
        self.assertGreater(float(np.max(g_in)), 0.0, 'inhibitory conductance is delivered')
        self.assertTrue(np.allclose(g_ex, 0.0), 'no excitatory conductance on receptor 2')
        self.assertLess(float(np.min(v)), E_L - 1e-3, 'the membrane is pulled toward E_in')

    def test_no_synaptic_input_rests_at_E_L(self):
        """With no spike connection the membrane sits exactly at E_L (the pre-fix null)."""
        _t, v, _g_ex, _g_in = _bp_drive([], EXC_W, receptor_type=1)
        v = _ms(v)
        self.assertTrue(np.allclose(v, E_L, atol=1e-6), 'quiescent neuron rests at E_L')

    def test_loop_lowers_with_stable_trace_shapes(self):
        """The model runs under the Simulator's for_loop with ``(T/dt,)`` traces."""
        _t, v, g_ex, g_in = _bp_drive(EXC_SPIKES, EXC_W, receptor_type=1)
        n = int(round(SIM_T / DT))
        for tr in (_ms(v), _ms(g_ex), _ms(g_in)):
            self.assertEqual(tr.shape, (n,))


# --- Live-NEST parity (deterministic spike drive) ----------------------------------

def _nest_drive(spike_times, weight_signed, sim_time=SIM_T):
    """NEST: spike_generator -> iaf_chxk_2008; return (V_m, g_ex, g_in)."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT})
    n = nest.Create('iaf_chxk_2008')
    sg = nest.Create('spike_generator', params={'spike_times': list(spike_times)})
    nest.Connect(sg, n, syn_spec={'weight': weight_signed, 'delay': DT})
    mm = nest.Create('multimeter', params={'record_from': ['V_m', 'g_ex', 'g_in'], 'interval': DT})
    nest.Connect(mm, n)
    nest.Simulate(sim_time)
    return (np.asarray(mm.events['V_m'], dtype=float),
            np.asarray(mm.events['g_ex'], dtype=float),
            np.asarray(mm.events['g_in'], dtype=float))


@requires_nest
class TestIafChxk2008ConductanceParity(unittest.TestCase):
    """The spike-driven conductance (V_m / g_ex / g_in) matches live NEST."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def _assert_parity(self, spike_times, weight, receptor_type, weight_signed):
        n_v, n_gex, n_gin = _nest_drive(spike_times, weight_signed)
        _t, b_v, b_gex, b_gin = _bp_drive(spike_times, weight, receptor_type)
        b_v, b_gex, b_gin = _ms(b_v), _ms(b_gex), _ms(b_gin)
        nv = min(n_v.size, b_v.size)
        compare_trace(n_v[:nv], b_v[:nv], tol=VM_TOL, metric='V_m').assert_()
        for nm, ref, cand in (('g_ex', n_gex, b_gex), ('g_in', n_gin, b_gin)):
            k = min(ref.size, cand.size)
            compare_trace(ref[:k], cand[:k], tol=COND_TOL, metric=nm).assert_()

    def test_excitatory_conductance_matches_nest(self):
        """Excitatory train (brainpy receptor 1 / NEST +weight): V_m + g_ex track NEST."""
        n_v, n_gex, _n_gin = _nest_drive(EXC_SPIKES, EXC_W)
        self.assertGreater(float(np.max(n_gex)), 0.0)        # the exc channel fired
        self.assertGreater(float(np.max(n_v)), E_L + 1.0)    # NEST also depolarised
        self._assert_parity(EXC_SPIKES, EXC_W, receptor_type=1, weight_signed=EXC_W)

    def test_inhibitory_conductance_matches_nest(self):
        """Inhibitory train (brainpy receptor 2 / NEST -weight): V_m + g_in track NEST."""
        _n_v, _n_gex, n_gin = _nest_drive(INH_SPIKES, -INH_W)
        self.assertGreater(float(np.max(n_gin)), 0.0)        # the inh channel fired
        self._assert_parity(INH_SPIKES, INH_W, receptor_type=2, weight_signed=-INH_W)


if __name__ == '__main__':
    unittest.main()
