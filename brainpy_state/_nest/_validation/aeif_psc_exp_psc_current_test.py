# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spike-driven PSC-current parity for ``aeif_psc_exp`` (goal 28, exp kinetics).

``aeif_psc_exp`` was only ever validated with constant-current (``I_e``) drive. Its
excitatory/inhibitory **spike** path into ``I_ex``/``I_in`` was silently dead: the
Simulator never populated the ``label='w_ex'/'w_in'`` delta channels the model
self-pulls, so a presynaptic spike left ``V_m`` pinned at ``E_L`` with no error.

This validates the fix. ``aeif_psc_exp`` now exposes the multi-receptor bridge
(``n_receptors=2``; bridge default ``runit=u.pA`` -> no ``receptor_input_unit``
override): ``receptor_type=1`` routes a positive ``pA`` weight into ``I_ex`` and
``receptor_type=2`` into ``I_in`` -- NEST's weight-sign routing (``aeif_psc_exp.cpp``
``handle()``: ``weight>0 -> spike_exc_``, else ``spike_inh_``). The *exponential* synapse
applies the jump directly (``I_ex += w_ex``); only the *source* of ``w_ex``/``w_in``
changed (bridge vs self-pull), so this is the exp-kinetics arbiter for the pA bridge.

The shared bridge edge cases (weight=0, convergent scatter-add, seam-zero,
``receptor_type`` out-of-range) live in ``aeif_psc_alpha_psc_current_test.py``.
"""
import gc
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:                                   # pragma: no cover - env dependent
    nest = None

from brainpy import state as bp
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import TraceTolerance

DT = 0.1
E_L = -70.6
VM_TOL = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=3, label='A',
                        note='aeif_psc_exp V_m (spike-driven) vs live NEST')
CUR_TOL = TraceTolerance(1e-3, 1e-3, align_steps=3, label='A',
                         note='aeif_psc_exp I_ex/I_in vs live NEST')

EXC_SPIKES = [10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]
EXC_W = 300.0         # pA per spike (receptor 1 / NEST +weight); subthreshold, depolarises > 1 mV
INH_SPIKES = [10., 12., 14., 16., 18., 20., 22., 24.]
INH_W = 120.0         # pA per spike (receptor 2 / NEST -weight); tau_syn_in=2 ms >> tau_syn_ex so a
                      # smaller weight still hyperpolarises clearly
SIM_T = 60.0


def _ms(x):
    """Strip units to a flat float64 ndarray (a recorded trace mantissa)."""
    return np.asarray(u.get_mantissa(x), dtype=float).reshape(-1)


def _bp_drive(spike_times, weight, receptor_type, sim_time=SIM_T):
    """brainpy: spike_generator -> aeif_psc_exp; return (times, V_m, I_ex, I_in)."""
    brainstate.environ.set(dt=DT * u.ms)
    sim = bp.Simulator(dt=DT * u.ms)
    n = sim.create(bp.aeif_psc_exp, 1)
    if len(spike_times):
        sg = sim.create(bp.spike_generator, 1, spike_times=np.asarray(spike_times) * u.ms)
        sim.connect(sg, n, weight=weight * u.pA, delay=DT * u.ms, receptor_type=receptor_type)
    mm = sim.create(bp.multimeter, record_from=['V_m', 'I_ex', 'I_in'])
    sim.connect(mm, n)
    res = sim.simulate(sim_time * u.ms)
    return (res.times, res.trace(mm, 'V_m'), res.trace(mm, 'I_ex'), res.trace(mm, 'I_in'))


class TestAeifPscExpCurrentLaw(unittest.TestCase):
    """Spike-driven PSC-current invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_excitatory_train_raises_I_ex_and_depolarises(self):
        """``receptor_type=1`` delivers excitatory current: I_ex>0, V_m above E_L."""
        _t, v, i_ex, i_in = _bp_drive(EXC_SPIKES, EXC_W, receptor_type=1)
        v, i_ex, i_in = _ms(v), _ms(i_ex), _ms(i_in)
        self.assertGreater(float(np.max(i_ex)), 0.0, 'excitatory current is delivered')
        self.assertTrue(np.allclose(i_in, 0.0), 'no inhibitory current on receptor 1')
        self.assertGreater(float(np.max(v)), E_L + 1.0, 'the membrane depolarises above E_L')

    def test_inhibitory_train_raises_I_in_and_hyperpolarises(self):
        """``receptor_type=2`` delivers inhibitory current: I_in>0 (subtracted in dV)."""
        _t, v, i_ex, i_in = _bp_drive(INH_SPIKES, INH_W, receptor_type=2)
        v, i_ex, i_in = _ms(v), _ms(i_ex), _ms(i_in)
        self.assertGreater(float(np.max(i_in)), 0.0, 'inhibitory current is delivered')
        self.assertTrue(np.allclose(i_ex, 0.0), 'no excitatory current on receptor 2')
        self.assertLess(float(np.min(v)), E_L - 1.0, 'the membrane is pulled below E_L')

    def test_no_synaptic_input_rests_at_E_L(self):
        """With no spike connection the membrane sits exactly at E_L (the pre-fix null)."""
        _t, v, _ie, _ii = _bp_drive([], EXC_W, receptor_type=1)
        self.assertTrue(np.allclose(_ms(v), E_L, atol=1e-6), 'quiescent neuron rests at E_L')

    def test_loop_lowers_with_stable_trace_shapes(self):
        """The model runs under the Simulator's for_loop with ``(T/dt,)`` traces."""
        _t, v, i_ex, i_in = _bp_drive(EXC_SPIKES, EXC_W, receptor_type=1)
        n = int(round(SIM_T / DT))
        for tr in (_ms(v), _ms(i_ex), _ms(i_in)):
            self.assertEqual(tr.shape, (n,))


# --- Live-NEST parity (deterministic spike drive) ----------------------------------

def _nest_drive(spike_times, weight_signed, sim_time=SIM_T):
    """NEST: spike_generator -> aeif_psc_exp; return (V_m, I_syn_ex, I_syn_in)."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT})
    n = nest.Create('aeif_psc_exp', 1)
    sg = nest.Create('spike_generator', params={'spike_times': list(spike_times)})
    nest.Connect(sg, n, syn_spec={'weight': weight_signed, 'delay': DT})
    mm = nest.Create('multimeter', params={'record_from': ['V_m', 'I_syn_ex', 'I_syn_in'], 'interval': DT})
    nest.Connect(mm, n)
    nest.Simulate(sim_time)
    return (np.asarray(mm.events['V_m'], dtype=float),
            np.asarray(mm.events['I_syn_ex'], dtype=float),
            np.asarray(mm.events['I_syn_in'], dtype=float))


@requires_nest
class TestAeifPscExpCurrentParity(unittest.TestCase):
    """The spike-driven PSC current (V_m / I_syn_ex / I_syn_in) matches live NEST."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def _assert_parity(self, spike_times, weight, receptor_type, weight_signed):
        n_v, n_ie, n_ii = _nest_drive(spike_times, weight_signed)
        _t, b_v, b_ie, b_ii = _bp_drive(spike_times, weight, receptor_type)
        b_v, b_ie, b_ii = _ms(b_v), _ms(b_ie), _ms(b_ii)
        nv = min(n_v.size, b_v.size)
        compare_trace(n_v[:nv], b_v[:nv], tol=VM_TOL, metric='V_m').assert_()
        for nm, ref, cand in (('I_syn_ex', n_ie, b_ie), ('I_syn_in', n_ii, b_ii)):
            k = min(ref.size, cand.size)
            compare_trace(ref[:k], cand[:k], tol=CUR_TOL, metric=nm).assert_()

    def test_excitatory_current_matches_nest(self):
        """Excitatory train (brainpy receptor 1 / NEST +weight): V_m + I_syn_ex track NEST."""
        n_v, n_ie, _n_ii = _nest_drive(EXC_SPIKES, EXC_W)
        self.assertGreater(float(np.max(n_ie)), 0.0)        # the exc channel fired
        self.assertGreater(float(np.max(n_v)), E_L + 1.0)   # NEST also depolarised
        self._assert_parity(EXC_SPIKES, EXC_W, receptor_type=1, weight_signed=EXC_W)

    def test_inhibitory_current_matches_nest(self):
        """Inhibitory train (brainpy receptor 2 / NEST -weight): V_m + I_syn_in track NEST."""
        _n_v, _n_ie, n_ii = _nest_drive(INH_SPIKES, -INH_W)
        self.assertGreater(float(np.max(n_ii)), 0.0)        # the inh channel fired (+ magnitude)
        self._assert_parity(INH_SPIKES, INH_W, receptor_type=2, weight_signed=-INH_W)


if __name__ == '__main__':
    unittest.main()
