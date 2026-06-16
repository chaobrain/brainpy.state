# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spike-driven PSC-current parity for ``hh_psc_alpha_clopath`` (goal 28, stiff HH).

``hh_psc_alpha_clopath`` was only ever validated with constant-current (``I_e``) drive.
Its excitatory/inhibitory **spike** path into ``I_syn_ex``/``I_syn_in`` was silently dead:
the Simulator never populated the ``label='w_ex'/'w_in'`` delta channels the model
self-pulls, so a presynaptic spike left ``V_m`` pinned at rest with no error.

This validates the fix. ``hh_psc_alpha_clopath`` now exposes the multi-receptor bridge
(``n_receptors=2``; bridge default ``runit=u.pA`` -> no ``receptor_input_unit`` override):
``receptor_type=1`` routes a positive ``pA`` weight into ``I_syn_ex`` and ``receptor_type=2``
into ``I_syn_in`` -- NEST's weight-sign routing (``hh_psc_alpha_clopath.cpp`` ``handle()``:
``weight>0 -> spike_exc_``, else ``spike_inh_`` which *keeps the negative weight*). The
alpha kick is unchanged: ``dI_in -= (e/tau)*w_in`` builds a **negative** ``I_syn_in`` (the
NEST convention), so the inhibitory current trace is negative on both sims.

**Stiff cell discipline (25):** HH kinetics need x64 (module-level ``jax_enable_x64`` +
``precision=64``) and ``jax.clear_caches()`` between tests. The neuron is held
**subthreshold** (small weights, no action potential) so the Dormand-Prince vs GSL solvers
do not split on a full spike; the law class compares **driven-vs-baseline** (the cell rests
near -65 mV with ~4e-4 mV/60 ms drift). The shared bridge edge cases live in
``aeif_psc_alpha_psc_current_test.py``.
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
V_REST = -65.0
VM_TOL = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=3, label='A',
                        note='hh_psc_alpha_clopath V_m (spike-driven, subthreshold) vs live NEST')
CUR_TOL = TraceTolerance(1e-3, 1e-3, align_steps=3, label='A',
                         note='hh_psc_alpha_clopath I_syn_ex/I_syn_in vs live NEST')

EXC_SPIKES = [10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]
EXC_W = 200.0         # pA per spike (receptor 1 / NEST +weight); +2.1 mV peak, Vmax=-62.9 (no AP).
INH_SPIKES = [10., 12., 14., 16., 18., 20., 22., 24.]
INH_W = 120.0         # pA per spike (receptor 2 / NEST -weight); tau_syn_in=2 ms >> tau_syn_ex=0.2 ms
                      # so per-pA the inhibitory deflection is far larger (-4.4 mV here).
SIM_T = 60.0


def _ms(x):
    """Strip units to a flat float64 ndarray (a recorded trace mantissa)."""
    return np.asarray(u.get_mantissa(x), dtype=float).reshape(-1)


def _bp_drive(spike_times, weight, receptor_type, sim_time=SIM_T):
    """brainpy: spike_generator -> hh_psc_alpha_clopath; return (times, V_m, I_syn_ex, I_syn_in)."""
    brainstate.environ.set(dt=DT * u.ms)
    sim = bp.Simulator(dt=DT * u.ms)
    n = sim.create(bp.hh_psc_alpha_clopath, 1)
    if len(spike_times):
        sg = sim.create(bp.spike_generator, 1, spike_times=np.asarray(spike_times) * u.ms)
        sim.connect(sg, n, weight=weight * u.pA, delay=DT * u.ms, receptor_type=receptor_type)
    mm = sim.create(bp.multimeter, record_from=['V_m', 'I_syn_ex', 'I_syn_in'])
    sim.connect(mm, n)
    res = sim.simulate(sim_time * u.ms)
    return (res.times, res.trace(mm, 'V_m'),
            res.trace(mm, 'I_syn_ex'), res.trace(mm, 'I_syn_in'))


class TestHhPscAlphaClopathCurrentLaw(unittest.TestCase):
    """Spike-driven PSC-current invariants that need no NEST (always run, subthreshold)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_excitatory_train_raises_I_syn_ex_and_depolarises(self):
        """``receptor_type=1`` delivers excitatory current: I_syn_ex>0, V_m above baseline."""
        _t, vb, _ieb, _iib = _bp_drive([], 0.0, 1)
        _t, v, i_ex, i_in = _bp_drive(EXC_SPIKES, EXC_W, receptor_type=1)
        vb, v, i_ex, i_in = _ms(vb), _ms(v), _ms(i_ex), _ms(i_in)
        self.assertGreater(float(np.max(i_ex)), 0.0, 'excitatory current is delivered')
        self.assertTrue(np.allclose(i_in, 0.0), 'no inhibitory current on receptor 1')
        self.assertGreater(float(np.max(v - vb)), 0.5, 'depolarises clearly above baseline')
        self.assertLess(float(np.max(v)), 0.0, 'and stays subthreshold (no action potential)')

    def test_inhibitory_train_makes_I_syn_in_negative_and_hyperpolarises(self):
        """``receptor_type=2`` delivers inhibitory current: I_syn_in<0 (NEST sign), V_m below baseline."""
        _t, vb, _ieb, _iib = _bp_drive([], 0.0, 1)
        _t, v, i_ex, i_in = _bp_drive(INH_SPIKES, INH_W, receptor_type=2)
        vb, v, i_ex, i_in = _ms(vb), _ms(v), _ms(i_ex), _ms(i_in)
        self.assertLess(float(np.min(i_in)), 0.0, 'inhibitory current is negative (NEST keep-negative-weight)')
        self.assertTrue(np.allclose(i_ex, 0.0), 'no excitatory current on receptor 2')
        self.assertLess(float(np.min(v - vb)), -0.5, 'hyperpolarises clearly below baseline')

    def test_no_synaptic_input_stays_quiescent(self):
        """With no spike connection the membrane rests near -65 mV (no AP, no blow-up)."""
        _t, v, i_ex, i_in = _bp_drive([], 0.0, receptor_type=1)
        v, i_ex, i_in = _ms(v), _ms(i_ex), _ms(i_in)
        self.assertTrue(np.all(np.abs(v - V_REST) < 1.0), 'quiescent neuron rests near -65 mV')
        self.assertTrue(np.allclose(i_ex, 0.0) and np.allclose(i_in, 0.0), 'no synaptic current at rest')

    def test_loop_lowers_with_stable_trace_shapes(self):
        """The model runs under the Simulator's for_loop with ``(T/dt,)`` traces."""
        _t, v, i_ex, i_in = _bp_drive(EXC_SPIKES, EXC_W, receptor_type=1)
        n = int(round(SIM_T / DT))
        for tr in (_ms(v), _ms(i_ex), _ms(i_in)):
            self.assertEqual(tr.shape, (n,))


# --- Live-NEST parity (deterministic spike drive, subthreshold) ---------------------

def _nest_drive(spike_times, weight_signed, sim_time=SIM_T):
    """NEST: spike_generator -> hh_psc_alpha_clopath; return (V_m, I_syn_ex, I_syn_in)."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT})
    n = nest.Create('hh_psc_alpha_clopath', 1)
    sg = nest.Create('spike_generator', params={'spike_times': list(spike_times)})
    nest.Connect(sg, n, syn_spec={'weight': weight_signed, 'delay': DT})
    mm = nest.Create('multimeter', params={'record_from': ['V_m', 'I_syn_ex', 'I_syn_in'], 'interval': DT})
    nest.Connect(mm, n)
    nest.Simulate(sim_time)
    return (np.asarray(mm.events['V_m'], dtype=float),
            np.asarray(mm.events['I_syn_ex'], dtype=float),
            np.asarray(mm.events['I_syn_in'], dtype=float))


@requires_nest
class TestHhPscAlphaClopathCurrentParity(unittest.TestCase):
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
        self.assertLess(float(np.max(n_v)), 0.0)            # NEST also stayed subthreshold
        self._assert_parity(EXC_SPIKES, EXC_W, receptor_type=1, weight_signed=EXC_W)

    def test_inhibitory_current_matches_nest(self):
        """Inhibitory train (brainpy receptor 2 / NEST -weight): V_m + I_syn_in track NEST."""
        _n_v, _n_ie, n_ii = _nest_drive(INH_SPIKES, -INH_W)
        self.assertLess(float(np.min(n_ii)), 0.0)           # the inh channel fired (negative)
        self._assert_parity(INH_SPIKES, INH_W, receptor_type=2, weight_signed=-INH_W)


if __name__ == '__main__':
    unittest.main()
