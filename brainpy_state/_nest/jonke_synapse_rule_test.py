# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``jonke_synapse``.

Kernel-level checks of the exponential-weight-dependence STDP rule: potentiation
``min(w + lambda*(exp(mu_plus*w)*K+ - beta), Wmax)`` gated on the post spike,
depression ``max(w + lambda*(-alpha*exp(mu_minus*w)*K- - beta), 0)`` gated on the
pre spike, the **one-sided** clips (facilitation upper-only, depression
lower-only), the exponential weight dependence, the beta offset, the lambda=0
early return, the simultaneous-spike exclusion, and per-edge freezing. Live-NEST
equivalence is covered by the parity test.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import saiunit as u

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import jonke_synapse
from brainpy_state._network._event_plastic import KernelContext


def _ctx(pre_spike, post_spike, pre_trace, post_trace, E=1, t=10.0, dt=1.0):
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    return KernelContext(g(pre_spike), g(post_spike), g(pre_trace), g(post_trace),
                         jnp.asarray(t), jnp.asarray(dt), jax.random.key(0))


def _host_fac(w, kplus, lam=0.01, mu=0.0, beta=0.0, Wmax=100.0):
    nw = w + lam * (math.exp(mu * w) * kplus - beta)
    return min(nw, Wmax)


def _host_dep(w, kminus, lam=0.01, alpha=1.0, mu=0.0, beta=0.0):
    nw = w + lam * (-alpha * math.exp(mu * w) * kminus - beta)
    return max(nw, 0.0)


# -- spec contract ---------------------------------------------------------
def test_spec_attributes_and_defaults():
    s = jonke_synapse()
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.edge_state_init() == {}
    assert s.lambda_ == 0.01 and s.alpha == 1.0
    assert s.mu_plus == 0.0 and s.mu_minus == 0.0 and s.beta == 0.0
    assert s.Wmax == 100.0
    assert float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms)) == 20.0
    assert u.get_unit(s.weight) == u.pA


def test_validation():
    with pytest.raises(ValueError):
        jonke_synapse(tau_plus=-1.0 * u.ms)
    with pytest.raises(ValueError):
        jonke_synapse(tau_minus=0.0 * u.ms)


# -- additive limit (mu = 0, beta = 0) -------------------------------------
def test_additive_potentiation_on_post():
    s = jonke_synapse(weight=10.0, lambda_=0.1)
    st, w_eff = s.update({'weight': jnp.array([10.0])}, _ctx(0.0, 1.0, 0.5, 0.0))
    expect = _host_fac(10.0, 0.5, lam=0.1)
    assert np.allclose(np.asarray(st['weight']), [expect])
    assert np.allclose(np.asarray(w_eff), [expect])


def test_additive_depression_on_pre():
    s = jonke_synapse(weight=10.0, lambda_=0.1)
    st, _ = s.update({'weight': jnp.array([10.0])}, _ctx(1.0, 0.0, 0.0, 0.5))
    assert np.allclose(np.asarray(st['weight']), [_host_dep(10.0, 0.5, lam=0.1)])


# -- exponential weight dependence (mu_plus > 0 amplifies at higher w) ------
def test_exponential_weight_dependence():
    s = jonke_synapse(weight=10.0, lambda_=0.1, mu_plus=0.1)
    st_lo, _ = s.update({'weight': jnp.array([10.0])}, _ctx(0.0, 1.0, 0.5, 0.0))
    st_hi, _ = s.update({'weight': jnp.array([20.0])}, _ctx(0.0, 1.0, 0.5, 0.0))
    d_lo = float(np.asarray(st_lo['weight'])[0]) - 10.0
    d_hi = float(np.asarray(st_hi['weight'])[0]) - 20.0
    assert d_hi > d_lo                                      # exp(0.1*20) > exp(0.1*10)
    assert np.allclose(np.asarray(st_lo['weight']), [_host_fac(10.0, 0.5, lam=0.1, mu=0.1)])


# -- beta offset biases each update toward depression ----------------------
def test_beta_offset():
    s = jonke_synapse(weight=10.0, lambda_=0.1, beta=0.3)
    st, _ = s.update({'weight': jnp.array([10.0])}, _ctx(0.0, 1.0, 0.5, 0.0))
    assert np.allclose(np.asarray(st['weight']), [_host_fac(10.0, 0.5, lam=0.1, beta=0.3)])


# -- one-sided clips: facilitate upper-only, depress lower-only ------------
def test_facilitate_clamps_upper_only():
    s = jonke_synapse(weight=99.0, lambda_=5.0, Wmax=100.0)
    st, _ = s.update({'weight': jnp.array([99.0])}, _ctx(0.0, 1.0, 10.0, 0.0))
    assert np.allclose(np.asarray(st['weight']), [100.0])  # clamped to Wmax
    # facilitation does NOT lower-clip: large positive beta drives it below 0
    s2 = jonke_synapse(weight=1.0, lambda_=5.0, beta=10.0, Wmax=100.0)
    st2, _ = s2.update({'weight': jnp.array([1.0])}, _ctx(0.0, 1.0, 0.0, 0.0))
    assert float(np.asarray(st2['weight'])[0]) < 0.0       # no lower clip in facilitate


def test_depress_clamps_lower_only():
    s = jonke_synapse(weight=1.0, lambda_=5.0)
    st, _ = s.update({'weight': jnp.array([1.0])}, _ctx(1.0, 0.0, 0.0, 10.0))
    assert np.allclose(np.asarray(st['weight']), [0.0])    # clamped to 0
    # depression does NOT upper-clip: negative beta drives it above Wmax
    s2 = jonke_synapse(weight=99.0, lambda_=5.0, beta=-10.0, Wmax=100.0)
    st2, _ = s2.update({'weight': jnp.array([99.0])}, _ctx(1.0, 0.0, 0.0, 0.0))
    assert float(np.asarray(st2['weight'])[0]) > 100.0     # no upper clip in depress


# -- lambda = 0 disables learning (returns w unchanged, no clip) -----------
def test_lambda_zero_skips_update_and_clip():
    s = jonke_synapse(weight=150.0, lambda_=0.0, Wmax=100.0)
    st, _ = s.update({'weight': jnp.array([150.0])}, _ctx(0.0, 1.0, 5.0, 0.0))
    assert np.allclose(np.asarray(st['weight']), [150.0])  # unchanged despite w > Wmax


# -- simultaneous pre&post excludes the current step's own spike -----------
def test_simultaneous_excludes_current_step_spike():
    s = jonke_synapse(weight=10.0, lambda_=0.1)
    st, _ = s.update({'weight': jnp.array([10.0])}, _ctx(1.0, 1.0, 1.0, 1.0))
    assert np.allclose(np.asarray(st['weight']), [10.0])   # isolated pair: no net change


def test_frozen_non_firing_edges():
    s = jonke_synapse(weight=10.0, lambda_=0.1)
    ctx = _ctx([1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5], E=2)
    st, _ = s.update({'weight': jnp.array([10.0, 10.0])}, ctx)
    assert np.allclose(np.asarray(st['weight'])[1], 10.0)        # frozen
    assert np.asarray(st['weight'])[0] < 10.0                    # depressed


def test_vmap_grad_smoke():
    s = jonke_synapse(weight=10.0, lambda_=0.1, mu_plus=0.05)

    def run(w):
        st, _ = s.update({'weight': w}, _ctx(0.0, 1.0, 0.5, 0.0))
        return jnp.sum(st['weight'])

    g = jax.grad(run)(jnp.array([10.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    out = jax.vmap(run)(jnp.array([[10.0], [20.0], [30.0]]))
    assert out.shape == (3,)
