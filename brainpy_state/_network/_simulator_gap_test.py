# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for the gap-junction coupling seam (goal 15b).

Gap junctions couple ``hh_psc_alpha_gap`` / ``hh_cond_beta_gap_traub`` by an
explicit-lag *difference deposit* into the post's **current** channel:

    I_gap = G @ V[n-1]  -  D * V[n-1]        D = rowsum(G)   (option (a), full-lag)

The off-diagonal ``G @ V_pre`` reuses the seam-(H) continuous-emission machinery
(the V emission holder, ``_emission_attr='V'``); the per-neuron self term
``-D * V_post`` keeps the rest balance (``I_gap == 0`` when all V are equal). The
deposit rides ``add_current_input`` -- the ``sum_current_inputs(x, V)`` seam the gap
neurons already read -- under the substrate's one-step pipeline lag (the WFR seed,
cluster 15a). No waveform relaxation.

These tests are NEST-free. Live-NEST synchronization parity is covered by
``_nest/_validation/gap_junction_parity_test.py``.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import brainstate
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import (hh_psc_alpha_gap, hh_cond_beta_gap_traub, iaf_psc_alpha,
                           voltmeter, Simulator, all_to_all, one_to_one)
from brainpy_state._nest.gap_junction import gap_junction
from brainpy_state._nest.hh_psc_alpha_gap import _hh_psc_alpha_gap_equilibrium
from brainpy_state._network._simulator import Simulator as _Sim


def _resting_gating():
    """``hh_psc_alpha_gap`` gating at the resting default ``_NEST_V_INIT``.

    NEST's gap demo perturbs a cell's ``V_m`` *after* construction; ``SetStatus`` on
    ``V_m`` does not recompute the gating variables, so the perturbed cell keeps the
    resting (``eq(-69.6 mV)``) gating. The port's default is ``eq(V_m_init)`` per
    neuron, so to reproduce NEST we override the gating to this resting equilibrium.
    """
    m, h, n, p = _hh_psc_alpha_gap_equilibrium(hh_psc_alpha_gap._NEST_V_INIT)
    return dict(Act_m_init=m, Inact_h_init=h, Act_n_init=n, Inact_p_init=p)


def _pop_module(view):
    """The underlying population module of a created NodeView."""
    return view.segments[0].population


def _vm(res, rec):
    return np.asarray(u.get_mantissa(res.trace(rec, 'V_m') / u.mV))


# ---------------------------------------------------------------------------
# Phase 0 -- the V emission attr + holder (reuses 12/22 machinery)
# ---------------------------------------------------------------------------
class TestGapEmissionAttr(unittest.TestCase):
    """Both gap neurons emit their membrane voltage ``V`` for the gap seam."""

    def test_gap_neurons_declare_v_emission(self):
        # The gap difference deposit reads V_pre[n-1] from the emission holder, so
        # both gap-capable neurons must declare ``_emission_attr='V'``.
        self.assertEqual(hh_psc_alpha_gap._emission_attr, 'V')
        self.assertEqual(hh_cond_beta_gap_traub._emission_attr, 'V')

    def test_gap_neurons_are_not_continuous_rate(self):
        # ``_emission_attr='V'`` must NOT flag the neuron as a continuous *rate*
        # emitter: that would route it through the delta-channel rate path + the
        # phi-homogeneity guard (15a), which gap does not use. The gap seam is a
        # current-channel difference deposit, dispatched on the gap synapse model.
        self.assertFalse(getattr(hh_psc_alpha_gap, '_emission_continuous', False))
        self.assertFalse(getattr(hh_cond_beta_gap_traub, '_emission_continuous', False))
        with brainstate.environ.context(dt=0.05 * u.ms):
            self.assertFalse(_Sim._is_continuous_rate(hh_psc_alpha_gap(1)))
            self.assertFalse(_Sim._is_continuous_rate(hh_cond_beta_gap_traub(1)))

    def test_emission_holder_allocated_and_captures_V(self):
        # create() allocates an emission holder for any pop declaring _emission_attr;
        # phase 2 captures getattr(m, 'V').value into it each step. After a short run
        # the holder mirrors the population's (post-update) V.
        sim = Simulator(dt=0.05 * u.ms)
        pop = sim.create(hh_psc_alpha_gap, 2,
                         params={'V_m_init': jnp.array([-10.0, -65.0]) * u.mV,
                                 'I_e': 100.0 * u.pA})
        sim.simulate(0.2 * u.ms)                       # a few steps
        mod = _pop_module(pop)
        holder = getattr(sim, f'_emit_holder_{id(mod)}', None)
        self.assertIsNotNone(holder, 'V emission holder was not allocated')
        np.testing.assert_allclose(
            np.asarray(holder.spk.value),
            np.asarray(u.get_mantissa(mod.V.value)), rtol=0, atol=1e-9)


class TestGapEmissionNonInterference(unittest.TestCase):
    """A plain (non-gap) spike connection from a gap neuron still delivers spikes."""

    def test_plain_connect_from_gap_neuron_builds_spike_projection(self):
        # The gap neuron declares _emission_attr='V' but is NOT continuous; a plain
        # connect (no gap synapse, no receptor) must take the ordinary binary-spike
        # path (the inhibitory-network demo wires the SAME population over chemical
        # static synapses via spikes AND gap junctions via V, simultaneously).
        sim = Simulator(dt=0.05 * u.ms)
        pre = sim.create(hh_psc_alpha_gap, 2, params={'I_e': 100.0 * u.pA})
        post = sim.create(iaf_psc_alpha, 2)
        proj = sim.connect(pre, post, rule=all_to_all, weight=10.0 * u.pA,
                           delay=1.0 * u.ms)
        self.assertIsNotNone(proj)
        # Resolve the emission for a plain (receptor-less) connect: it must read the
        # binary spike holder, NOT the V emission holder. Distinct sentinels make the
        # source unambiguous (the holder States exist only after init_all_states).
        brainstate.nn.init_all_states(sim)
        mod = _pop_module(pre)
        spike_holder = getattr(sim, f'_holder_{id(mod)}')
        emit_holder = getattr(sim, f'_emit_holder_{id(mod)}')
        spike_holder.spk.value = jnp.array([1.0, 0.0])      # sentinel spike pattern
        emit_holder.spk.value = jnp.array([-42.0, -42.0])   # sentinel V emission
        reader, eff_rt = sim._resolve_stp_emission(mod, _pop_module(post), None,
                                                   spike_holder, 'dense')
        self.assertIsNone(eff_rt)
        np.testing.assert_allclose(np.asarray(reader()), [1.0, 0.0])  # spike, not V


# ---------------------------------------------------------------------------
# Phase 1 -- the gap coupler: the difference current G@V - D*V (option (a))
# ---------------------------------------------------------------------------
class TestGapCurrentArithmetic(unittest.TestCase):
    """The pure difference-current helper ``I_gap = G @ V_pre - D * V_post``."""

    def test_two_neuron_difference(self):
        g = 0.5
        G = jnp.array([[0.0, g], [g, 0.0]])
        D = G.sum(axis=1)                         # [g, g]
        v = jnp.array([-10.0, -65.0])             # mV
        i_gap = np.asarray(_Sim._gap_current(G, D, v, v))
        # I_gap = [g(V1-V0), g(V0-V1)] -- equal and opposite (nS*mV = pA).
        np.testing.assert_allclose(i_gap, [g * (v[1] - v[0]), g * (v[0] - v[1])], atol=1e-12)
        # antisymmetric pair sums to zero current (no net charge injection)
        self.assertAlmostEqual(float(i_gap.sum()), 0.0, places=10)

    def test_equal_voltage_is_zero_current(self):
        # The rest balance: when all V are equal, the off-diagonal G@V cancels the
        # self term D*V exactly, so the gap injects no spurious bias current.
        G = jnp.array([[0.0, 0.3, 0.3], [0.3, 0.0, 0.3], [0.3, 0.3, 0.0]])
        D = G.sum(axis=1)
        v = jnp.full((3,), -64.0)
        i_gap = np.asarray(_Sim._gap_current(G, D, v, v))
        np.testing.assert_allclose(i_gap, 0.0, atol=1e-12)

    def test_three_neuron_chain(self):
        # Chain 0-1-2 (no 0-2 edge): I_gap_i = sum_j g_ij (V_j - V_i).
        g = 0.7
        G = jnp.array([[0.0, g, 0.0], [g, 0.0, g], [0.0, g, 0.0]])
        D = G.sum(axis=1)                         # [g, 2g, g]
        v = jnp.array([1.0, 2.0, 4.0])
        i_gap = np.asarray(_Sim._gap_current(G, D, v, v))
        expected = [g * (2 - 1), g * (1 - 2) + g * (4 - 2), g * (2 - 4)]
        np.testing.assert_allclose(i_gap, expected, atol=1e-12)


class TestGapCouplerBuild(unittest.TestCase):
    """``connect(..., synapse=gap_junction)`` builds a symmetric, hollow coupler."""

    def _build(self, n, rule=all_to_all, g=0.5, allow_autapses=False):
        sim = Simulator(dt=0.05 * u.ms)
        pop = sim.create(hh_psc_alpha_gap, n, params={'I_e': 100.0 * u.pA})
        sim.connect(pop, pop, rule=rule, weight=g * u.nS, synapse=gap_junction,
                    comm='dense', allow_autapses=allow_autapses)
        self.assertEqual(len(sim._gap_couplers), 1)
        G, D, reader, post_pop, key = sim._gap_couplers[0]
        return sim, np.asarray(G), np.asarray(D)

    def test_all_to_all_matrix_symmetric_hollow_rowsum(self):
        sim, G, D = self._build(4, g=0.5)
        np.testing.assert_allclose(G, G.T, atol=0)            # symmetric
        np.testing.assert_allclose(np.diag(G), 0.0, atol=0)   # hollow (no autapses)
        # all-to-all, g=0.5: every off-diagonal is 0.5, D_i = (n-1)*g
        self.assertAlmostEqual(float(G[0, 1]), 0.5, places=10)
        np.testing.assert_allclose(D, (4 - 1) * 0.5, atol=1e-10)

    def test_two_neuron_pair(self):
        sim, G, D = self._build(2, g=0.5)
        np.testing.assert_allclose(G, [[0.0, 0.5], [0.5, 0.0]], atol=1e-10)
        np.testing.assert_allclose(D, [0.5, 0.5], atol=1e-10)


class TestGapCouplingGuards(unittest.TestCase):
    """Connect-time enforcement: symmetry domain, no delay, dense-only, scalar g."""

    def _pop(self, sim, n=2):
        return sim.create(hh_psc_alpha_gap, n, params={'I_e': 100.0 * u.pA})

    def test_delay_rejected(self):
        sim = Simulator(dt=0.05 * u.ms)
        pop = self._pop(sim)
        with self.assertRaisesRegex(ValueError, 'delay'):
            sim.connect(pop, pop, rule=all_to_all, weight=0.5 * u.nS,
                        synapse=gap_junction, delay=1.0 * u.ms, allow_autapses=False)

    def test_sparse_rejected(self):
        sim = Simulator(dt=0.05 * u.ms)
        pop = self._pop(sim)
        with self.assertRaisesRegex(ValueError, 'sparse|dense'):
            sim.connect(pop, pop, rule=all_to_all, weight=0.5 * u.nS,
                        synapse=gap_junction, comm='sparse', allow_autapses=False)

    def test_non_recurrent_rejected(self):
        # Gap coupling is recurrent (within one population); pre != post is an error.
        sim = Simulator(dt=0.05 * u.ms)
        a = self._pop(sim)
        b = self._pop(sim)
        with self.assertRaisesRegex(ValueError, 'recurrent|population'):
            sim.connect(a, b, rule=all_to_all, weight=0.5 * u.nS,
                        synapse=gap_junction, allow_autapses=False)

    def test_callable_weight_rejected(self):
        import braintools
        sim = Simulator(dt=0.05 * u.ms)
        pop = self._pop(sim)
        with self.assertRaisesRegex(ValueError, 'scalar|conductance'):
            sim.connect(pop, pop, rule=all_to_all,
                        weight=braintools.init.Uniform(0.1 * u.nS, 0.5 * u.nS),
                        synapse=gap_junction, allow_autapses=False)

    def test_non_scalar_weight_rejected(self):
        # The gap conductance g is one scalar shared by every edge (NEST's symmetric
        # gap weight); a per-neuron / per-edge array weight is out of scope and must be
        # rejected with a clean error (not flow on to a float() cast that TypeErrors).
        sim = Simulator(dt=0.05 * u.ms)
        pop = self._pop(sim, n=3)
        with self.assertRaisesRegex(ValueError, 'scalar|non-scalar'):
            sim.connect(pop, pop, rule=all_to_all,
                        weight=jnp.array([0.1, 0.2, 0.3]) * u.nS,
                        synapse=gap_junction, allow_autapses=False)

    def test_non_emitting_post_rejected(self):
        # Gap coupling reads V_pre[n-1] from the post's V emission holder; a neuron that
        # does not declare _emission_attr='V' (e.g. iaf_psc_alpha, a binary-spike model)
        # has no holder, so a gap connect onto it must be rejected -- not silently wired
        # to read a (nonexistent) voltage emission.
        sim = Simulator(dt=0.05 * u.ms)
        pop = sim.create(iaf_psc_alpha, 2)
        with self.assertRaisesRegex(ValueError, 'emit|emission|_emission_attr'):
            sim.connect(pop, pop, rule=all_to_all, weight=0.5 * u.nS,
                        synapse=gap_junction, comm='dense', allow_autapses=False)


class TestGapBehavior(unittest.TestCase):
    """End-to-end: rest balance, synchronization, for_loop lowering, WFR unused."""

    def setUp(self):
        import jax as _jax
        _jax.clear_caches()
        brainstate.environ.set(precision=64, platform='cpu')

    def _two_neuron_sim(self, v0=-10.0, v1=None, g=0.5, t=150.0, resting_gating=False):
        v1 = hh_psc_alpha_gap._NEST_V_INIT if v1 is None else v1
        sim = Simulator(dt=0.05 * u.ms)
        params = {'V_m_init': jnp.array([v0, v1]) * u.mV, 'I_e': 100.0 * u.pA}
        if resting_gating:                       # NEST-faithful ICs (see _resting_gating)
            params.update(_resting_gating())
        pop = sim.create(hh_psc_alpha_gap, 2, params=params)
        vm = sim.create(voltmeter)
        sim.connect(pop, pop, rule=all_to_all, weight=g * u.nS, synapse=gap_junction,
                    comm='dense', allow_autapses=False)
        sim.connect(vm, pop)
        res = sim.simulate(t * u.ms)
        return _vm(res, vm)

    def _lone_sim(self, v_init, t=20.0):
        # A single uncoupled gap neuron under the same DC drive -- the reference an
        # uncoupled (g=0) neuron in the pair must reproduce exactly.
        sim = Simulator(dt=0.05 * u.ms)
        pop = sim.create(hh_psc_alpha_gap, 1,
                         params={'V_m_init': jnp.array([v_init]) * u.mV,
                                 'I_e': 100.0 * u.pA})
        vm = sim.create(voltmeter)
        sim.connect(vm, pop)
        res = sim.simulate(t * u.ms)
        return _vm(res, vm)

    def test_equal_voltage_pair_stays_identical(self):
        # Identical ICs -> the gap current is zero forever -> the two traces never
        # diverge (a trivial-but-essential guard on the difference deposit).
        v = self._two_neuron_sim(v0=-65.0, v1=-65.0, g=0.5, t=50.0)
        self.assertEqual(v.shape[1], 2)
        np.testing.assert_allclose(v[:, 0], v[:, 1], atol=1e-9)

    def test_decoupled_g0_is_independent(self):
        # g=0 -> the coupler deposits exactly zero current -> each neuron in the pair
        # evolves identically to a lone neuron with the same IC (the seam is off). This
        # is the rigorous decoupling check: not "the voltages stay far apart" (HH from
        # -10 mV fires an AP and the recorded gap shrinks), but "the pair == two
        # independent runs", which holds to machine precision iff the gap injects 0.
        v_pair = self._two_neuron_sim(v0=-10.0, v1=-65.0, g=0.0, t=20.0)
        v_lone0 = self._lone_sim(-10.0, t=20.0)
        v_lone1 = self._lone_sim(-65.0, t=20.0)
        n = min(v_pair.shape[0], v_lone0.shape[0], v_lone1.shape[0])
        np.testing.assert_allclose(v_pair[:n, 0], v_lone0[:n, 0], atol=1e-9)
        np.testing.assert_allclose(v_pair[:n, 1], v_lone1[:n, 0], atol=1e-9)
        # non-vacuous: the two ICs really do drive distinct trajectories.
        self.assertGreater(np.max(np.abs(v_pair[:n, 0] - v_pair[:n, 1])), 10.0)

    def test_gap_pair_converges_to_synchrony(self):
        # One cell perturbed to -10 mV, the other at rest (~-69.6 mV), converge under
        # symmetric gap coupling. This mirrors NEST's gap_junctions_two_neurons demo
        # (resting gating, g=0.5 nS, T=351 ms) and the port reproduces NEST's
        # use_wfr=False synchronization: the last-20 ms RMS membrane gap falls to
        # ~0.5 mV (NEST ~0.54) from a desynchronized ~30 mV start. Live-NEST trace
        # parity is gap_junction_parity_test.py; this is the NEST-free behavioral gate.
        v = self._two_neuron_sim(v0=-10.0, v1=None, g=0.5, t=351.0, resting_gating=True)
        d = np.abs(v[:, 0] - v[:, 1])
        w = 400                                          # 20 ms window (dt=0.05 ms)
        early = float(np.sqrt(np.mean(d[:w] ** 2)))
        late = float(np.sqrt(np.mean(d[-w:] ** 2)))
        self.assertGreater(early, 10.0)                  # genuinely desynchronized start
        self.assertLess(late, 2.0)                       # converged to near-synchrony
        self.assertLess(late, 0.1 * early)               # large relative reduction

    def test_strong_coupling_is_stable_and_synchronizes(self):
        # A large gap conductance (g=50 nS, 5x the leak) must not blow up under the
        # explicit one-step lag: the difference deposit is dissipative (negated graph
        # Laplacian), so the pair locks into tight synchrony and V stays finite/bounded.
        v = self._two_neuron_sim(v0=-10.0, v1=None, g=50.0, t=200.0, resting_gating=True)
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertLess(float(np.max(np.abs(v))), 200.0)        # bounded (no divergence)
        late = float(np.sqrt(np.mean((v[-400:, 0] - v[-400:, 1]) ** 2)))
        self.assertLess(late, 1.0)                              # tight synchrony

    def test_reference_wfr_class_not_used_by_simulator(self):
        # The Simulator builds its own difference-deposit coupler; the reference
        # gap_junction WFR machinery (begin_wfr_cycle/evaluate_gap_current) must
        # never be invoked on the simulation path.
        import unittest.mock as mock
        with mock.patch.object(gap_junction, 'begin_wfr_cycle',
                               side_effect=AssertionError('WFR used')), \
             mock.patch.object(gap_junction, 'evaluate_gap_current',
                               side_effect=AssertionError('WFR used')):
            self._two_neuron_sim(v0=-10.0, v1=-65.0, g=0.5, t=10.0)  # must not raise


if __name__ == '__main__':
    unittest.main()
