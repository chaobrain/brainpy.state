# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for the rebuilt ``stdp_dopamine_synapse`` spec + pure rule kernel (NEST-free).

The kernel is fed a hand-built :class:`KernelContext` (with ``signals={'n': ...}``
for the broadcast dopamine concentration) so the ``c*(n-b)`` weight integral, the
eligibility-trace decay, the STDP facilitate/depress impulses, the ``Wmin``/``Wmax``
clamps, the ``b != 0`` sign-flip, broadcast correctness and jit/vmap/grad are locked
without a live network. The closed-form host reference mirrors NEST's
``update_weight_`` (``stdp_dopamine_synapse.h:427-448``) and ``facilitate_``/
``depress_`` (``.h:507-519``).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state._nest_network.event_plastic import KernelContext  # noqa: E402
from brainpy_state._nest_plasticity.stdp_dopamine_synapse import stdp_dopamine_synapse  # noqa: E402

# NEST defaults (stdp_dopamine_synapse.cpp:45-56)
A_PLUS, A_MINUS = 1.0, 1.5
TAU_PLUS, TAU_C, TAU_N = 20.0, 1000.0, 200.0
B, WMIN, WMAX = 0.0, 0.0, 200.0


def _ctx(E=1, *, pre_spike=0.0, post_spike=0.0, pre_trace=0.0, post_trace=0.0,
         n=0.0, dt=1.0):
    f = lambda v: jnp.full((E,), float(v))
    return KernelContext(
        pre_spike=f(pre_spike), post_spike=f(post_spike),
        pre_trace=f(pre_trace), post_trace=f(post_trace),
        t_now=jnp.asarray(0.0), dt=jnp.asarray(float(dt)), key=jax.random.key(0),
        signals={'n': jnp.asarray(float(n))})


def _ref(w, c, *, n, dt, b=B, A_plus=A_PLUS, A_minus=A_MINUS, tau_c=TAU_C,
         tau_n=TAU_N, Wmin=WMIN, Wmax=WMAX, pre_spike=0.0, post_spike=0.0,
         pre_trace=0.0, post_trace=0.0):
    """Closed-form host reference for one kernel step (NEST update_weight_ + impulses)."""
    taus = (tau_c + tau_n) / (tau_c * tau_n)
    dw = -c * (n / taus * np.expm1(-taus * dt) - b * tau_c * np.expm1(-dt / tau_c))
    w2 = np.clip(w + dw, Wmin, Wmax)
    c2 = c * np.exp(-dt / tau_c)
    if post_spike > 0:
        c2 = c2 + A_plus * (pre_trace - pre_spike)
    if pre_spike > 0:
        c2 = c2 - A_minus * (post_trace - post_spike)
    return w2, c2


def _run(s, w, c, ctx):
    new, w_eff = s.update({'weight': jnp.asarray([w]), 'c': jnp.asarray([c])}, ctx)
    return float(np.asarray(new['weight'])[0]), float(np.asarray(new['c'])[0]), np.asarray(w_eff)


# --------------------------------------------------------------------------
# spec attributes + validation
# --------------------------------------------------------------------------
def test_spec_attributes_and_defaults():
    s = stdp_dopamine_synapse()
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.post_state_reads == ()
    assert s.signal_reads == ('n',)
    assert float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms)) == TAU_PLUS
    assert float(u.Quantity(s.post_trace_tau).to_decimal(u.ms)) == 20.0   # tau_minus
    assert s.edge_state_init() == {'c': 0.0}
    assert float(u.get_mantissa(s.weight)) == 1.0
    assert u.get_unit(s.weight) == u.pA
    assert (s.A_plus, s.A_minus, s.b) == (A_PLUS, A_MINUS, B)
    assert (s.Wmin, s.Wmax) == (WMIN, WMAX)


def test_weight_unit_defaults_to_pa_and_preserves_explicit():
    assert u.get_unit(stdp_dopamine_synapse(weight=2.0).weight) == u.pA
    s = stdp_dopamine_synapse(weight=3.0 * u.pA)
    assert float(u.get_mantissa(s.weight)) == 3.0 and u.get_unit(s.weight) == u.pA


def test_edge_state_init_uses_c_init():
    assert stdp_dopamine_synapse(c=0.4).edge_state_init() == {'c': 0.4}


@pytest.mark.parametrize('kw', ['tau_c', 'tau_n', 'tau_plus', 'tau_minus'])
def test_time_constants_must_be_positive(kw):
    with pytest.raises(ValueError, match=kw):
        stdp_dopamine_synapse(**{kw: 0.0 * u.ms})


def test_nonfinite_param_rejected():
    with pytest.raises(ValueError, match='finite'):
        stdp_dopamine_synapse(A_plus=np.inf)
    with pytest.raises(ValueError, match='finite'):
        stdp_dopamine_synapse(Wmax=np.nan)


def test_delay_must_be_positive():
    with pytest.raises(ValueError, match='delay'):
        stdp_dopamine_synapse(delay=0.0 * u.ms)


# --------------------------------------------------------------------------
# the c*(n-b) weight integral
# --------------------------------------------------------------------------
def test_weight_integral_matches_nest_update_weight():
    s = stdp_dopamine_synapse()
    ctx = _ctx(n=0.5, dt=1.0)
    w, c, w_eff = _run(s, 100.0, 2.0, ctx)
    wr, cr = _ref(100.0, 2.0, n=0.5, dt=1.0)
    assert np.isclose(w, wr) and np.isclose(c, cr)
    assert np.allclose(w_eff, w)           # delivered weight == post-integral stored weight


def test_potentiation_when_n_above_b():
    s = stdp_dopamine_synapse()
    w, c, _ = _run(s, 100.0, 3.0, _ctx(n=0.5, dt=1.0))   # n>b=0, c>0 -> w increases
    assert w > 100.0


def test_no_dopamine_no_weight_change():
    # edge case 1: n=0, b=0 -> integral is 0, only c decays.
    s = stdp_dopamine_synapse()
    w, c, _ = _run(s, 100.0, 5.0, _ctx(n=0.0, dt=1.0))
    assert np.isclose(w, 100.0)
    assert np.isclose(c, 5.0 * np.exp(-1.0 / TAU_C))


def test_no_eligibility_no_weight_change():
    # edge case 2: c=0 -> no weight change regardless of n.
    s = stdp_dopamine_synapse()
    w, c, _ = _run(s, 100.0, 0.0, _ctx(n=0.9, dt=1.0))
    assert np.isclose(w, 100.0) and np.isclose(c, 0.0)


def test_b_nonzero_sign_flip_depresses():
    # edge case 7: n<b -> (n-b)<0 -> facilitation becomes depression (w decreases).
    s = stdp_dopamine_synapse(b=0.5)
    w, _, _ = _run(s, 100.0, 3.0, _ctx(n=0.1, dt=1.0))
    wr, _ = _ref(100.0, 3.0, n=0.1, b=0.5, dt=1.0)
    assert w < 100.0 and np.isclose(w, wr)


# --------------------------------------------------------------------------
# Wmin / Wmax clamps (both signs of n - b)
# --------------------------------------------------------------------------
def test_wmax_clamp_on_strong_potentiation():
    s = stdp_dopamine_synapse(Wmax=100.0)
    w, _, _ = _run(s, 99.9, 50.0, _ctx(n=5.0, dt=1.0))   # large c*n -> overshoot
    assert np.isclose(w, 100.0)


def test_wmin_clamp_on_strong_depression():
    s = stdp_dopamine_synapse(b=1.0, Wmin=0.0)
    w, _, _ = _run(s, 0.1, 50.0, _ctx(n=0.0, dt=1.0))    # n<b -> depress past 0
    assert np.isclose(w, 0.0)


# --------------------------------------------------------------------------
# eligibility-trace decay + STDP facilitate / depress impulses
# --------------------------------------------------------------------------
def test_eligibility_decays_each_step():
    s = stdp_dopamine_synapse()
    _, c, _ = _run(s, 100.0, 4.0, _ctx(n=0.0, dt=1.0))
    assert np.isclose(c, 4.0 * np.exp(-1.0 / TAU_C))


def test_facilitate_on_post_after_pre():
    # post spike adds A_plus * Kplus_prior (the strictly-prior pre-trace) to c.
    s = stdp_dopamine_synapse()
    ctx = _ctx(post_spike=1.0, pre_trace=2.0, n=0.0, dt=1.0)
    _, c, _ = _run(s, 100.0, 1.0, ctx)
    wr, cr = _ref(100.0, 1.0, n=0.0, dt=1.0, post_spike=1.0, pre_trace=2.0)
    assert np.isclose(c, cr)
    assert np.isclose(c, 1.0 * np.exp(-1.0 / TAU_C) + A_PLUS * 2.0)


def test_depress_on_pre_spike():
    # pre spike subtracts A_minus * Kminus_prior (the strictly-prior post-trace) from c.
    s = stdp_dopamine_synapse()
    ctx = _ctx(pre_spike=1.0, post_trace=2.0, n=0.0, dt=1.0)
    _, c, _ = _run(s, 100.0, 1.0, ctx)
    assert np.isclose(c, 1.0 * np.exp(-1.0 / TAU_C) - A_MINUS * 2.0)


def test_strictly_prior_excludes_current_spike():
    # edge case 11: a pre spike coincident with its own trace +1 uses the value
    # *before* this step's increment (Kplus_prior = pre_trace - pre_spike).
    s = stdp_dopamine_synapse()
    # post fires; pre_trace already includes this step's pre +1 (=3), so the
    # facilitation must use 3 - 1 = 2.
    ctx = _ctx(post_spike=1.0, pre_spike=1.0, pre_trace=3.0, post_trace=1.0, n=0.0, dt=1.0)
    _, c, _ = _run(s, 100.0, 1.0, ctx)
    expected = 1.0 * np.exp(-1.0 / TAU_C) + A_PLUS * (3.0 - 1.0) - A_MINUS * (1.0 - 1.0)
    assert np.isclose(c, expected)


def test_integration_uses_pre_impulse_c():
    # NEST order: integrate w with c at t_k (before this step's impulses), then
    # decay c, then apply impulses. So a coincident post spike does NOT inflate
    # this step's weight integral.
    s = stdp_dopamine_synapse()
    w_imp, _, _ = _run(s, 100.0, 2.0, _ctx(n=0.5, post_spike=1.0, pre_trace=9.0, dt=1.0))
    w_plain, _, _ = _run(s, 100.0, 2.0, _ctx(n=0.5, dt=1.0))
    assert np.isclose(w_imp, w_plain)      # impulse affects c (future), not this w


# --------------------------------------------------------------------------
# broadcast correctness + dt closed-form sweep
# --------------------------------------------------------------------------
def test_broadcast_scalar_n_to_all_edges():
    # edge case 4: one scalar n reaches every edge identically.
    s = stdp_dopamine_synapse()
    ctx = _ctx(E=3, n=0.5, dt=1.0)
    new, _ = s.update({'weight': jnp.full((3,), 100.0), 'c': jnp.array([1.0, 2.0, 3.0])}, ctx)
    wr = [_ref(100.0, c, n=0.5, dt=1.0)[0] for c in (1.0, 2.0, 3.0)]
    assert np.allclose(np.asarray(new['weight']), wr)


@pytest.mark.parametrize('dt', [0.1, 0.5, 1.0])
def test_dt_closed_form(dt):
    s = stdp_dopamine_synapse()
    w, c, _ = _run(s, 100.0, 2.0, _ctx(n=0.5, dt=dt))
    wr, cr = _ref(100.0, 2.0, n=0.5, dt=dt)
    assert np.isclose(w, wr) and np.isclose(c, cr)


# --------------------------------------------------------------------------
# jit / vmap / grad smoke (edge case 10)
# --------------------------------------------------------------------------
def test_vmap_over_state_batch():
    s = stdp_dopamine_synapse()
    ctx = _ctx(E=4, n=0.5, dt=1.0)

    def run(w, c):
        return s.update({'weight': w, 'c': c}, ctx)[1]

    out = jax.vmap(run)(jnp.full((5, 4), 100.0), jnp.full((5, 4), 1.0))
    assert out.shape == (5, 4)


def test_grad_flows_through_weight_and_c():
    s = stdp_dopamine_synapse()

    def loss(w, c):
        new, _ = s.update({'weight': w, 'c': c}, _ctx(n=0.5, post_spike=1.0, pre_trace=1.0, dt=1.0))
        return jnp.sum(new['weight']) + jnp.sum(new['c'])

    gw, gc = jax.grad(loss, argnums=(0, 1))(jnp.array([100.0]), jnp.array([1.0]))
    assert np.all(np.isfinite(np.asarray(gw))) and np.all(np.isfinite(np.asarray(gc)))
