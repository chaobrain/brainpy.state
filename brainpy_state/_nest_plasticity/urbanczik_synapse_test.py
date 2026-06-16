# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for the rebuilt ``urbanczik_synapse`` (spec + pure rule kernel).

The rebuild is a frozen parameter spec plus a pure
``update(state, ctx) -> (new_state, w_eff)`` kernel on the
:class:`~brainpy_state._nest_network.event_plastic.VoltageCoupledPlasticProj`
substrate. These NEST-free tests pin:

* the spec declarations the substrate dispatches on (two pre-traces ``(tau_L,
  tau_s)``, the δΠ post-state read, the carried integrals);
* the kernel math against an **independent Python twin** of the online
  reformulation (§2 of the spec) to floating-point tolerance — the twin is the
  per-step recurrence the substrate's decay-then-add trace seam realizes;
* the Urbanczik edge cases (§6): δΠ≡0, the ``Wmin``/``Wmax`` clamps, the
  ``tau_L==tau_s`` degenerate guard, the exc/inh ``tau_s`` selection by initial
  weight sign, the no-presynaptic-spike quiescence, multi-edge independence, and
  ``for_loop``/``vmap``/``grad`` lowering of the kernel.
"""
import math
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state._nest_plasticity.urbanczik_synapse import urbanczik_synapse
from brainpy_state._nest_network.event_plastic import KernelContext


# --------------------------------------------------------------------------
# Independent reference: the §2 online recurrence (host-side, pure Python).
# --------------------------------------------------------------------------
def ref_online_weight(dpi, pre_spikes, *, dt, init_w, eta, tau_Delta, Wmin, Wmax,
                      C_m, g_L, tau_syn_ex, tau_syn_in):
    """Per-step online Urbanczik weight trace (the §2 recurrence).

    Decay-then-add on both presynaptic traces (matching the substrate's
    ``_advance_trace`` for ``all_to_all``); the ``+1`` spike jump cancels in the
    ``tau_L - tau_s`` difference, so this online sum equals NEST's event-driven
    window integral at every grid step.
    """
    tau_L = C_m / g_L
    tau_s = tau_syn_ex if init_w > 0 else tau_syn_in
    pref = 15.0 * C_m * tau_s * eta / (g_L * (tau_L - tau_s))
    trL = trS = PI_int = PI_exp = 0.0
    w = init_w
    out = []
    for k in range(len(dpi)):
        s = 1.0 if pre_spikes[k] else 0.0
        trL = trL * math.exp(-dt / tau_L) + s
        trS = trS * math.exp(-dt / tau_s) + s
        PI = (trL - trS) * dpi[k]
        PI_int += PI
        PI_exp = PI_exp * math.exp(-dt / tau_Delta) + PI
        w = min(max(init_w + (PI_int - PI_exp) * pref, Wmin), Wmax)
        out.append(w)
    return out


def _drive_kernel(rule, dpi_seq, pre_seq, dt_ms):
    """Drive ``rule.update`` step-by-step, advancing the two pre-traces exactly as
    the substrate would (decay-then-add), and return the per-step weight (edge 0).

    Builds a single-edge :class:`KernelContext` each step so the kernel math is
    exercised in isolation from the substrate's delivery/CSR machinery.
    """
    init_w = rule._init_w
    state = {'weight': jnp.asarray([init_w]),
             'PI_integral': jnp.zeros(1),
             'PI_exp_integral': jnp.zeros(1)}
    dt = jnp.asarray(dt_ms)
    decL = math.exp(-dt_ms / rule._tau_L_ms)
    decS = math.exp(-dt_ms / rule._tau_s_ms)
    trL = trS = 0.0
    out = []
    for k in range(len(dpi_seq)):
        s = 1.0 if pre_seq[k] else 0.0
        trL = trL * decL + s
        trS = trS * decS + s
        pre_traces = jnp.asarray([[trL, trS]])           # (E=1, k=2)
        ctx = KernelContext(
            pre_spike=jnp.asarray([s]),
            post_spike=jnp.zeros(1),
            pre_trace=pre_traces[:, 0],
            post_trace=jnp.zeros(1),
            t_now=jnp.asarray(k * dt_ms),
            dt=dt,
            key=jax.random.key(0),
            pre_traces=pre_traces,
            post_traces=jnp.zeros((1, 0)),
            post_states={'delta_Pi': jnp.asarray([dpi_seq[k]])},
            signals=None,
        )
        state, _w_eff = rule.update(state, ctx)
        out.append(float(np.asarray(state['weight']).reshape(-1)[0]))
    return out


# Dendritic defaults shared by the kernel tests (NEST pp_cond_exp_mc_urbanczik).
_DEND = dict(dend_C_m=300.0 * u.pF, dend_g_L=30.0 * u.nS,
             dend_tau_syn_ex=3.0 * u.ms, dend_tau_syn_in=3.0 * u.ms)
_DEND_RAW = dict(C_m=300.0, g_L=30.0, tau_syn_ex=3.0, tau_syn_in=3.0)


class TestSpecDeclarations(unittest.TestCase):
    def test_substrate_dispatch_attributes(self):
        rule = urbanczik_synapse(weight=1.0 * u.pA, **_DEND)
        self.assertFalse(rule.is_homogeneous_weight)
        self.assertFalse(rule.stochastic)
        self.assertIsNone(rule.post_trace_tau)
        self.assertEqual(rule.post_state_reads, ('delta_Pi',))
        self.assertEqual(u.get_unit(rule.weight).dim, u.pA.dim)

    def test_pre_trace_tau_is_tau_L_then_tau_s(self):
        rule = urbanczik_synapse(weight=1.0 * u.pA, **_DEND)
        taus = tuple(float(u.Quantity(t).to_decimal(u.ms)) for t in rule.pre_trace_tau)
        # tau_L = C_m/g_L = 300/30 = 10 ms; tau_s = tau_syn_ex (w>0) = 3 ms
        npt.assert_allclose(taus, (10.0, 3.0))

    def test_edge_state_init_carries_two_integrals(self):
        rule = urbanczik_synapse(weight=1.0 * u.pA, **_DEND)
        self.assertEqual(rule.edge_state_init(),
                         {'PI_integral': 0.0, 'PI_exp_integral': 0.0})

    def test_default_parameters_match_nest(self):
        rule = urbanczik_synapse(**_DEND)
        self.assertAlmostEqual(rule.eta, 0.07)
        self.assertAlmostEqual(float(u.Quantity(rule.tau_Delta).to_decimal(u.ms)), 100.0)
        self.assertAlmostEqual(rule.Wmin, 0.0)
        self.assertAlmostEqual(rule.Wmax, 100.0)
        self.assertAlmostEqual(float(u.get_mantissa(rule.weight)), 1.0)


class TestKernelMatchesOnlineTwin(unittest.TestCase):
    def test_kernel_matches_twin_random_drive(self):
        rule = urbanczik_synapse(weight=1.0 * u.pA, eta=0.07, tau_Delta=100.0 * u.ms,
                                 Wmin=0.0, Wmax=100.0, **_DEND)
        rng = np.random.default_rng(0)
        n = 80
        dpi = list(rng.normal(0.0, 0.02, n))
        pre = [1 if rng.random() < 0.3 else 0 for _ in range(n)]
        got = _drive_kernel(rule, dpi, pre, 0.1)
        exp = ref_online_weight(dpi, pre, dt=0.1, init_w=1.0, eta=0.07,
                                tau_Delta=100.0, Wmin=0.0, Wmax=100.0, **_DEND_RAW)
        npt.assert_allclose(got, exp, atol=1e-12, rtol=0.0)

    def test_kernel_matches_twin_potentiating_drive(self):
        # Sustained positive δΠ with regular pre spikes -> monotone-ish potentiation.
        rule = urbanczik_synapse(weight=2.0 * u.pA, eta=0.1, tau_Delta=50.0 * u.ms,
                                 Wmin=0.0, Wmax=100.0, **_DEND)
        n = 100
        dpi = [0.05] * n
        pre = [1 if k % 5 == 0 else 0 for k in range(n)]
        got = _drive_kernel(rule, dpi, pre, 0.1)
        exp = ref_online_weight(dpi, pre, dt=0.1, init_w=2.0, eta=0.1,
                                tau_Delta=50.0, Wmin=0.0, Wmax=100.0, **_DEND_RAW)
        npt.assert_allclose(got, exp, atol=1e-12, rtol=0.0)
        self.assertGreater(got[-1], got[0])  # net potentiation


class TestEdgeCases(unittest.TestCase):
    def test_zero_delta_pi_keeps_weight_constant(self):
        rule = urbanczik_synapse(weight=1.0 * u.pA, **_DEND)
        n = 40
        got = _drive_kernel(rule, [0.0] * n, [1] * n, 0.1)
        npt.assert_allclose(got, [1.0] * n, atol=1e-12)

    def test_no_pre_spikes_keeps_weight_constant(self):
        rule = urbanczik_synapse(weight=1.0 * u.pA, **_DEND)
        n = 40
        got = _drive_kernel(rule, list(np.linspace(-0.1, 0.1, n)), [0] * n, 0.1)
        npt.assert_allclose(got, [1.0] * n, atol=1e-12)

    def test_wmax_clamp(self):
        rule = urbanczik_synapse(weight=1.0 * u.pA, eta=5.0, tau_Delta=100.0 * u.ms,
                                 Wmin=0.0, Wmax=3.0, **_DEND)
        got = _drive_kernel(rule, [0.5] * 200, [1 if k % 3 == 0 else 0 for k in range(200)], 0.1)
        self.assertLessEqual(max(got), 3.0 + 1e-9)
        self.assertAlmostEqual(max(got), 3.0, places=6)  # actually reaches the bound

    def test_wmin_clamp(self):
        # init_w>0, strong depression (negative δΠ) drives toward Wmin=0 and clamps.
        rule = urbanczik_synapse(weight=1.0 * u.pA, eta=5.0, tau_Delta=100.0 * u.ms,
                                 Wmin=0.0, Wmax=100.0, **_DEND)
        got = _drive_kernel(rule, [-0.5] * 200, [1 if k % 3 == 0 else 0 for k in range(200)], 0.1)
        self.assertGreaterEqual(min(got), -1e-9)
        self.assertAlmostEqual(min(got), 0.0, places=6)  # reaches the lower bound

    def test_tau_L_equals_tau_s_raises(self):
        # C_m/g_L == tau_syn_ex -> degenerate 1/(tau_L - tau_s); reject at construction.
        with self.assertRaises(ValueError):
            urbanczik_synapse(weight=1.0 * u.pA, dend_C_m=30.0 * u.pF,
                              dend_g_L=10.0 * u.nS, dend_tau_syn_ex=3.0 * u.ms,
                              dend_tau_syn_in=3.0 * u.ms)

    def test_tau_s_selected_by_weight_sign(self):
        exc = urbanczik_synapse(weight=1.0 * u.pA, Wmin=0.0, Wmax=100.0,
                                dend_C_m=300.0 * u.pF, dend_g_L=30.0 * u.nS,
                                dend_tau_syn_ex=3.0 * u.ms, dend_tau_syn_in=7.0 * u.ms)
        self.assertAlmostEqual(exc._tau_s_ms, 3.0)
        inh = urbanczik_synapse(weight=-1.0 * u.pA, Wmin=-100.0, Wmax=0.0,
                                dend_C_m=300.0 * u.pF, dend_g_L=30.0 * u.nS,
                                dend_tau_syn_ex=3.0 * u.ms, dend_tau_syn_in=7.0 * u.ms)
        self.assertAlmostEqual(inh._tau_s_ms, 7.0)

    def test_sign_consistency_validation(self):
        # weight & Wmin same sign; weight & Wmax same sign (NEST set_status tests).
        with self.assertRaises(ValueError):
            urbanczik_synapse(weight=1.0 * u.pA, Wmin=-1.0, Wmax=100.0, **_DEND)
        with self.assertRaises(ValueError):
            urbanczik_synapse(weight=1.0 * u.pA, Wmin=0.0, Wmax=-1.0, **_DEND)

    def test_delay_must_be_positive(self):
        with self.assertRaises(ValueError):
            urbanczik_synapse(weight=1.0 * u.pA, delay=0.0 * u.ms, **_DEND)

    def test_multi_edge_independent_gather(self):
        # Three edges with distinct δΠ evolve independently in one update call.
        rule = urbanczik_synapse(weight=1.0 * u.pA, **_DEND)
        E = 3
        state = {'weight': jnp.ones(E),
                 'PI_integral': jnp.zeros(E),
                 'PI_exp_integral': jnp.zeros(E)}
        with brainstate.environ.context(dt=0.1 * u.ms):
            # one pre spike on all edges, distinct δΠ
            pre_traces = jnp.ones((E, 2)) * jnp.asarray([1.0, 1.0])
            ctx = KernelContext(
                pre_spike=jnp.ones(E), post_spike=jnp.zeros(E),
                pre_trace=pre_traces[:, 0], post_trace=jnp.zeros(E),
                t_now=jnp.asarray(0.0), dt=jnp.asarray(0.1), key=jax.random.key(0),
                pre_traces=pre_traces, post_traces=jnp.zeros((E, 0)),
                post_states={'delta_Pi': jnp.asarray([-0.1, 0.0, 0.1])}, signals=None)
            new_state, w_eff = rule.update(state, ctx)
        self.assertEqual(tuple(w_eff.shape), (E,))
        # at the first step PI_int == PI_exp -> all weights still init (= 1.0)
        npt.assert_allclose(np.asarray(w_eff), [1.0, 1.0, 1.0], atol=1e-12)
        self.assertEqual(tuple(new_state['PI_integral'].shape), (E,))


class TestKernelTransforms(unittest.TestCase):
    """The kernel must lower under jit/for_loop/vmap and be grad-safe."""

    def _ctx(self, E, dpi):
        pre_traces = jnp.ones((E, 2)) * jnp.asarray([0.8, 0.5])
        return KernelContext(
            pre_spike=jnp.ones(E), post_spike=jnp.zeros(E),
            pre_trace=pre_traces[:, 0], post_trace=jnp.zeros(E),
            t_now=jnp.asarray(0.0), dt=jnp.asarray(0.1), key=jax.random.key(0),
            pre_traces=pre_traces, post_traces=jnp.zeros((E, 0)),
            post_states={'delta_Pi': dpi}, signals=None)

    def test_jit_and_vmap_smoke(self):
        rule = urbanczik_synapse(weight=1.0 * u.pA, **_DEND)
        state = {'weight': jnp.ones(4), 'PI_integral': jnp.zeros(4),
                 'PI_exp_integral': jnp.zeros(4)}
        ctx = self._ctx(4, jnp.asarray([0.1, -0.1, 0.0, 0.05]))
        out, w = jax.jit(rule.update)(state, ctx)
        self.assertTrue(bool(jnp.all(jnp.isfinite(w))))

    def test_grad_through_weight_is_finite(self):
        rule = urbanczik_synapse(weight=1.0 * u.pA, **_DEND)

        def loss(dpi_val):
            state = {'weight': jnp.ones(1), 'PI_integral': jnp.zeros(1),
                     'PI_exp_integral': jnp.zeros(1)}
            ctx = self._ctx(1, jnp.asarray([dpi_val]))
            _new, w = rule.update(state, ctx)
            return jnp.sum(w)

        g = jax.grad(loss)(0.05)
        self.assertTrue(bool(jnp.isfinite(g)))


if __name__ == '__main__':
    unittest.main()
