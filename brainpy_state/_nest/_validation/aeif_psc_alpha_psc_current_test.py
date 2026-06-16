# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spike-driven PSC-current parity for ``aeif_psc_alpha`` (goal 28, alpha kinetics).

``aeif_psc_alpha`` was only ever validated with constant-current (``I_e``) drive -- never
with excitatory/inhibitory **spike** input into its synaptic current. That path was
silently dead: the Simulator never populated the ``label='w_ex'/'w_in'`` delta channels
the model self-pulls, so a presynaptic spike left ``V_m`` pinned at ``E_L`` with no error
(the current-based twin of the 17b/25 conductance gap).

This module validates the fix. ``aeif_psc_alpha`` now exposes the multi-receptor bridge
(``n_receptors=2``; the bridge's default ``receptor_input_unit`` is already ``u.pA`` so no
override is needed): ``receptor_type=1`` routes a positive ``pA`` weight into the
excitatory current ``I_ex`` and ``receptor_type=2`` into ``I_in`` -- the Simulator's
expression of NEST's weight-**sign** routing (``aeif_psc_alpha.cpp`` ``handle()``:
``weight>0 -> spike_exc_``, else ``spike_inh_``). The *alpha* kick ``dI += (e/tau)*w`` is
unchanged; only the *source* of ``w_ex``/``w_in`` swaps (bridge vs self-pull).

* **Law class** (always runs, no NEST): an excitatory train (``receptor_type=1``) raises
  ``I_ex`` and depolarises ``V_m`` above ``E_L``; an inhibitory train (``receptor_type=2``,
  positive ``pA``) raises ``I_in`` (a positive magnitude subtracted in ``dV``) and pulls
  ``V_m`` below ``E_L``; with no synaptic input ``V_m`` rests exactly at ``E_L`` (the
  pre-fix null / dead-path-gone guard); the model lowers under the Simulator ``for_loop``.
* **Edge cases** (NEST-free): the *shared* 17b/25 bridge edge cases (weight=0, convergent
  scatter-add -> 2x, self-pull seam-zero, ``receptor_type`` out-of-range) -- folded here as
  ``aeif_psc_alpha`` is the current-based representative (mirrors ``aeif_cond_exp`` for nS).
* **Parity class** (``@requires_nest``): a deterministic train drives the neuron; NEST uses
  a **signed** static weight (``+W`` excitatory / ``-W`` inhibitory), brainpy uses
  ``receptor_type=1``/``2`` with positive ``W*u.pA``. ``V_m`` (``VM_TOL``) and
  ``I_syn_ex``/``I_syn_in`` (``CUR_TOL``) track live NEST 3.9.0 with ``align_steps=3``.
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
#: V_m tolerance: 1e-3 mV with ``align_steps`` to absorb the substrate's integer
#: spike->current pipeline offset vs NEST (synaptic input applied after integration +
#: the multimeter buffer alignment; the residual after the clean ~2-sample shift is tiny).
VM_TOL = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=3, label='A',
                        note='aeif_psc_alpha V_m (spike-driven) vs live NEST')
#: I_ex/I_in trace tolerance (pA): bare-float atol like 25's COND_TOL; the same
#: ``align_steps`` absorbs the same offset. rel=1e-3 governs at O(10-100 pA).
CUR_TOL = TraceTolerance(1e-3, 1e-3, align_steps=3, label='A',
                         note='aeif_psc_alpha I_ex/I_in vs live NEST')

#: A deterministic excitatory train: closely-spaced spikes so the alpha current
#: accumulates and the membrane clearly departs from rest (kept subthreshold).
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
    """brainpy: spike_generator -> aeif_psc_alpha; return (times, V_m, I_ex, I_in)."""
    brainstate.environ.set(dt=DT * u.ms)
    sim = bp.Simulator(dt=DT * u.ms)
    n = sim.create(bp.aeif_psc_alpha, 1)
    if len(spike_times):
        sg = sim.create(bp.spike_generator, 1, spike_times=np.asarray(spike_times) * u.ms)
        sim.connect(sg, n, weight=weight * u.pA, delay=DT * u.ms, receptor_type=receptor_type)
    mm = sim.create(bp.multimeter, record_from=['V_m', 'I_ex', 'I_in'])
    sim.connect(mm, n)
    res = sim.simulate(sim_time * u.ms)
    return (res.times, res.trace(mm, 'V_m'), res.trace(mm, 'I_ex'), res.trace(mm, 'I_in'))


class TestAeifPscAlphaCurrentLaw(unittest.TestCase):
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


class TestAeifPscAlphaCurrentEdgeCases(unittest.TestCase):
    """Bridge edge cases (NEST-free): zero weight, convergent sum, seam-zero, out-of-range.

    These characterise the *shared* multi-receptor bridge that all three current-based
    neurons reuse; ``aeif_psc_alpha`` stands in as the representative model (the pA twin of
    ``aeif_cond_exp`` for the conductance family).
    """

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_zero_weight_delivers_no_current(self):
        """A ``weight=0`` connection routes through the bridge but deposits nothing."""
        _t, v, i_ex, _ii = _bp_drive(EXC_SPIKES, 0.0, receptor_type=1)
        self.assertTrue(np.allclose(_ms(i_ex), 0.0), 'a zero-weight edge delivers no current')
        self.assertTrue(np.allclose(_ms(v), E_L, atol=1e-6), 'and the membrane stays at rest')

    def test_convergent_sources_sum_into_one_receptor(self):
        """Two generators into ``receptor_type=1`` *sum* (scatter-add), not overwrite: I_ex ~ 2x."""
        single = float(np.max(_ms(_bp_drive(EXC_SPIKES, EXC_W, receptor_type=1)[2])))

        brainstate.environ.set(dt=DT * u.ms)
        sim = bp.Simulator(dt=DT * u.ms)
        n = sim.create(bp.aeif_psc_alpha, 1)
        sg1 = sim.create(bp.spike_generator, 1, spike_times=np.asarray(EXC_SPIKES) * u.ms)
        sim.connect(sg1, n, weight=EXC_W * u.pA, delay=DT * u.ms, receptor_type=1)
        sg2 = sim.create(bp.spike_generator, 1, spike_times=np.asarray(EXC_SPIKES) * u.ms)
        sim.connect(sg2, n, weight=EXC_W * u.pA, delay=DT * u.ms, receptor_type=1)
        mm = sim.create(bp.multimeter, record_from=['I_ex'])
        sim.connect(mm, n)
        double = float(np.max(_ms(sim.simulate(SIM_T * u.ms).trace(mm, 'I_ex'))))

        self.assertGreater(single, 0.0)
        # Current jumps superpose linearly, so two identical sources give exactly 2x.
        self.assertTrue(np.isclose(double, 2.0 * single, rtol=1e-6),
                        f'convergent sources sum: {double} vs 2x{single}')

    def test_self_pull_seam_returns_zero(self):
        """A bare ``update()`` (no ``w_by_rec``, nothing connected) leaves I_ex/I_in at 0."""
        n = bp.aeif_psc_alpha(1)
        n.init_state()
        with brainstate.environ.context(t=0. * u.ms):
            n.update()
        self.assertTrue(np.allclose(_ms(n.I_ex.value), 0.0), 'the self-pull seam adds no I_ex')
        self.assertTrue(np.allclose(_ms(n.I_in.value), 0.0), 'the self-pull seam adds no I_in')

    def test_receptor_type_out_of_range_raises(self):
        """``receptor_type=3`` on a 2-receptor neuron raises a clear range error."""
        brainstate.environ.set(dt=DT * u.ms)
        sim = bp.Simulator(dt=DT * u.ms)
        n = sim.create(bp.aeif_psc_alpha, 1)
        sg = sim.create(bp.spike_generator, 1, spike_times=np.asarray(EXC_SPIKES) * u.ms)
        with self.assertRaisesRegex(ValueError, 'out of range'):
            sim.connect(sg, n, weight=EXC_W * u.pA, delay=DT * u.ms, receptor_type=3)
            sim.simulate(SIM_T * u.ms)


# --- Live-NEST parity (deterministic spike drive) ----------------------------------

def _nest_drive(spike_times, weight_signed, sim_time=SIM_T):
    """NEST: spike_generator -> aeif_psc_alpha; return (V_m, I_syn_ex, I_syn_in)."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT})
    n = nest.Create('aeif_psc_alpha', 1)
    sg = nest.Create('spike_generator', params={'spike_times': list(spike_times)})
    nest.Connect(sg, n, syn_spec={'weight': weight_signed, 'delay': DT})
    mm = nest.Create('multimeter', params={'record_from': ['V_m', 'I_syn_ex', 'I_syn_in'], 'interval': DT})
    nest.Connect(mm, n)
    nest.Simulate(sim_time)
    return (np.asarray(mm.events['V_m'], dtype=float),
            np.asarray(mm.events['I_syn_ex'], dtype=float),
            np.asarray(mm.events['I_syn_in'], dtype=float))


@requires_nest
class TestAeifPscAlphaCurrentParity(unittest.TestCase):
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
