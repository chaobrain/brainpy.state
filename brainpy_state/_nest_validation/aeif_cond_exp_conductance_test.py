# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spike-driven conductance parity for ``aeif_cond_exp`` (goal 25, exp kinetics).

``aeif_cond_exp`` was only ever validated with constant-current (``I_e``) drive -- never
with excitatory/inhibitory **spike** input into its synaptic conductance. That path was
silently dead: the Simulator never populated the ``label='w_ex'/'w_in'`` delta channels the
model self-pulls, so a presynaptic spike left ``V_m`` pinned at ``E_L`` with no error.

This module validates the fix. ``aeif_cond_exp`` now exposes the 17b multi-receptor bridge
(``n_receptors=2``, ``receptor_input_unit=u.nS``): ``receptor_type=1`` routes a positive
``nS`` weight into the excitatory conductance ``g_ex`` and ``receptor_type=2`` into ``g_in``
-- the Simulator's expression of NEST's weight-**sign** routing (``aeif_cond_exp.cpp``
``handle()``: ``weight>0 -> spike_exc_``, else ``spike_inh_``). The *exponential* synapse
applies the jump directly (``g_ex += w_ex``); only the *source* of ``w_ex``/``w_in`` changed
(bridge vs self-pull), so this is the exp-kinetics arbiter for design A (source-only suffices).

* **Law class** (always runs, no NEST): an excitatory train (``receptor_type=1``) raises
  ``g_ex`` and depolarises ``V_m`` above ``E_L``; an inhibitory train (``receptor_type=2``,
  positive ``nS``) raises ``g_in`` and pulls ``V_m`` below ``E_L`` toward ``E_in``; with no
  synaptic input ``V_m`` rests exactly at ``E_L`` (the pre-fix null / dead-path-gone guard);
  the model lowers under the Simulator's ``for_loop`` with stable trace shapes.
* **Parity class** (``@requires_nest``): a deterministic spike train drives the neuron; NEST
  uses a **signed** static weight (``+W`` excitatory / ``-W`` inhibitory), brainpy uses
  ``receptor_type=1``/``2`` with positive ``W*u.nS``. ``V_m`` (``VM_TOL``) and ``g_ex``/``g_in``
  (``COND_TOL``) track live NEST 3.9.0 sample-for-sample with ``align_steps=3``.
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
E_L = -70.6
#: V_m tolerance: 1e-3 mV with ``align_steps`` to absorb the substrate's integer
#: spike->conductance pipeline offset vs NEST (synaptic input applied after integration +
#: the multimeter buffer alignment; the residual after the clean ~2-sample shift is ~1e-6 mV).
VM_TOL = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=3, label='A',
                        note='aeif_cond_exp V_m (spike-driven) vs live NEST')
#: g_ex/g_in trace tolerance (nS): same ``align_steps`` absorbs the same offset.
COND_TOL = TraceTolerance(1e-3, 1e-3, align_steps=3, label='A',
                          note='aeif_cond_exp g_ex/g_in vs live NEST')

#: A deterministic excitatory train: closely-spaced spikes so the fast (0.2 ms) excitatory
#: conductance accumulates and the membrane clearly departs from rest (subthreshold).
EXC_SPIKES = [10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]
EXC_W = 4.0          # nS per spike (receptor 1 / NEST +weight)
INH_SPIKES = [10., 12., 14., 16., 18., 20., 22., 24.]
INH_W = 30.0         # nS per spike (receptor 2 / NEST -weight)
SIM_T = 60.0


def _ms(x):
    """Strip units to a flat float64 ndarray (a recorded trace mantissa)."""
    return np.asarray(u.get_mantissa(x), dtype=float).reshape(-1)


def _bp_drive(spike_times, weight, receptor_type, sim_time=SIM_T):
    """brainpy: spike_generator -> aeif_cond_exp; return (times, V_m, g_ex, g_in)."""
    brainstate.environ.set(dt=DT * u.ms)
    sim = bp.Simulator(dt=DT * u.ms)
    n = sim.create(bp.aeif_cond_exp, 1)
    if len(spike_times):
        sg = sim.create(bp.spike_generator, 1, spike_times=np.asarray(spike_times) * u.ms)
        sim.connect(sg, n, weight=weight * u.nS, delay=DT * u.ms, receptor_type=receptor_type)
    mm = sim.create(bp.multimeter, record_from=['V_m', 'g_ex', 'g_in'])
    sim.connect(mm, n)
    res = sim.simulate(sim_time * u.ms)
    return (res.times, res.trace(mm, 'V_m'), res.trace(mm, 'g_ex'), res.trace(mm, 'g_in'))


class TestAeifCondExpConductanceLaw(unittest.TestCase):
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


class TestAeifCondExpConductanceEdgeCases(unittest.TestCase):
    """Bridge edge cases (NEST-free): zero weight, convergent sum, seam-zero, out-of-range.

    These characterise the *shared* 17b multi-receptor bridge that all eight conductance
    neurons reuse; ``aeif_cond_exp`` stands in as the representative model.
    """

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_zero_weight_delivers_no_conductance(self):
        """A ``weight=0`` connection routes through the bridge but deposits nothing."""
        _t, v, g_ex, _g_in = _bp_drive(EXC_SPIKES, 0.0, receptor_type=1)
        self.assertTrue(np.allclose(_ms(g_ex), 0.0), 'a zero-weight edge delivers no conductance')
        self.assertTrue(np.allclose(_ms(v), E_L, atol=1e-6), 'and the membrane stays at rest')

    def test_convergent_sources_sum_into_one_receptor(self):
        """Two generators into ``receptor_type=1`` *sum* (scatter-add), not overwrite: g_ex ~ 2x."""
        single = float(np.max(_ms(_bp_drive(EXC_SPIKES, EXC_W, receptor_type=1)[2])))

        brainstate.environ.set(dt=DT * u.ms)
        sim = bp.Simulator(dt=DT * u.ms)
        n = sim.create(bp.aeif_cond_exp, 1)
        sg1 = sim.create(bp.spike_generator, 1, spike_times=np.asarray(EXC_SPIKES) * u.ms)
        sim.connect(sg1, n, weight=EXC_W * u.nS, delay=DT * u.ms, receptor_type=1)
        sg2 = sim.create(bp.spike_generator, 1, spike_times=np.asarray(EXC_SPIKES) * u.ms)
        sim.connect(sg2, n, weight=EXC_W * u.nS, delay=DT * u.ms, receptor_type=1)
        mm = sim.create(bp.multimeter, record_from=['g_ex'])
        sim.connect(mm, n)
        double = float(np.max(_ms(sim.simulate(SIM_T * u.ms).trace(mm, 'g_ex'))))

        self.assertGreater(single, 0.0)
        # Conductance jumps superpose linearly, so two identical sources give exactly 2x.
        self.assertTrue(np.isclose(double, 2.0 * single, rtol=1e-6),
                        f'convergent sources sum: {double} vs 2x{single}')

    def test_self_pull_seam_returns_zero(self):
        """A bare ``update()`` (no ``w_by_rec``, nothing connected) leaves g_ex/g_in at 0."""
        n = bp.aeif_cond_exp(1)
        n.init_state()
        with brainstate.environ.context(t=0. * u.ms):
            n.update()
        self.assertTrue(np.allclose(_ms(n.g_ex.value), 0.0), 'the self-pull seam adds no g_ex')
        self.assertTrue(np.allclose(_ms(n.g_in.value), 0.0), 'the self-pull seam adds no g_in')

    def test_receptor_type_out_of_range_raises(self):
        """``receptor_type=3`` on a 2-receptor neuron raises a clear range error."""
        brainstate.environ.set(dt=DT * u.ms)
        sim = bp.Simulator(dt=DT * u.ms)
        n = sim.create(bp.aeif_cond_exp, 1)
        sg = sim.create(bp.spike_generator, 1, spike_times=np.asarray(EXC_SPIKES) * u.ms)
        with self.assertRaisesRegex(ValueError, 'out of range'):
            sim.connect(sg, n, weight=EXC_W * u.nS, delay=DT * u.ms, receptor_type=3)
            sim.simulate(SIM_T * u.ms)


# --- Live-NEST parity (deterministic spike drive) ----------------------------------

def _nest_drive(spike_times, weight_signed, sim_time=SIM_T):
    """NEST: spike_generator -> aeif_cond_exp; return (V_m, g_ex, g_in)."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT})
    n = nest.Create('aeif_cond_exp', 1)
    sg = nest.Create('spike_generator', params={'spike_times': list(spike_times)})
    nest.Connect(sg, n, syn_spec={'weight': weight_signed, 'delay': DT})
    mm = nest.Create('multimeter', params={'record_from': ['V_m', 'g_ex', 'g_in'], 'interval': DT})
    nest.Connect(mm, n)
    nest.Simulate(sim_time)
    return (np.asarray(mm.events['V_m'], dtype=float),
            np.asarray(mm.events['g_ex'], dtype=float),
            np.asarray(mm.events['g_in'], dtype=float))


@requires_nest
class TestAeifCondExpConductanceParity(unittest.TestCase):
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
