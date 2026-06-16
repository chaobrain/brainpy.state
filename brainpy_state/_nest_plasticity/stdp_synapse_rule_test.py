# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``stdp_synapse``.

Kernel-level checks of the online all-to-all STDP rule: potentiation gated on
the post spike (uses ``K+`` = pre trace), depression gated on the pre spike
(uses ``K-`` = post trace), in-update ``[0, Wmax]`` clamp, the mu=0 additive
limit, the simultaneous-spike exclusion convention, and per-edge freezing.
The substrate trace/delay timing + live-NEST equivalence are covered by the
parity test.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import stdp_synapse
from brainpy_state._nest_network._event_plastic import KernelContext


def _ctx(pre_spike, post_spike, pre_trace, post_trace, E=1, t=10.0, dt=1.0):
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    return KernelContext(g(pre_spike), g(post_spike), g(pre_trace), g(post_trace),
                         jnp.asarray(t), jnp.asarray(dt), jax.random.key(0))


def _host_facilitate(w, kplus, Wmax=100., lam=0.01, mu_plus=1.):
    nw = w / Wmax + lam * (1.0 - w / Wmax) ** mu_plus * kplus
    return nw * Wmax if nw < 1.0 else Wmax


def _host_depress(w, kminus, Wmax=100., alpha=1., lam=0.01, mu_minus=1.):
    nw = w / Wmax - alpha * lam * (w / Wmax) ** mu_minus * kminus
    return nw * Wmax if nw > 0.0 else 0.0


# -- spec contract ---------------------------------------------------------
def test_spec_attributes_and_trace_taus():
    s = stdp_synapse(tau_plus=18.0 * u.ms, tau_minus=22.0 * u.ms)
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.edge_state_init() == {}                       # no per-edge aux
    assert float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms)) == 18.0
    assert float(u.Quantity(s.post_trace_tau).to_decimal(u.ms)) == 22.0
    assert u.get_unit(s.weight) == u.pA


def test_validation_weight_wmax_sign_and_kplus():
    with pytest.raises(ValueError):
        stdp_synapse(weight=-1.0, Wmax=100.0)              # opposite sign
    with pytest.raises(ValueError):
        stdp_synapse(Kplus=-1e-3)                          # negative trace
    with pytest.raises(ValueError):
        stdp_synapse(tau_plus=-1.0 * u.ms)                 # non-positive tau_plus
    with pytest.raises(ValueError):
        stdp_synapse(tau_minus=-1.0 * u.ms)               # non-positive tau_minus


# -- potentiation on post spike (K+ = pre trace) ---------------------------
def test_potentiation_on_post_spike():
    s = stdp_synapse(weight=10.0, Wmax=100.0, lambda_=0.1)
    # post fires, no pre this step; pre trace 0.5 (a recent pre spike)
    st, w_eff = s.update({'weight': jnp.array([10.0])}, _ctx(0.0, 1.0, 0.5, 0.0))
    expect = _host_facilitate(10.0, 0.5, lam=0.1)
    assert np.allclose(np.asarray(st['weight']), [expect])
    assert np.allclose(np.asarray(w_eff), [expect])
    assert expect > 10.0                                   # potentiated


def test_depression_on_pre_spike():
    s = stdp_synapse(weight=10.0, Wmax=100.0, lambda_=0.1)
    # pre fires, no post this step; post trace 0.5 (a recent post spike)
    st, w_eff = s.update({'weight': jnp.array([10.0])}, _ctx(1.0, 0.0, 0.0, 0.5))
    expect = _host_depress(10.0, 0.5, lam=0.1)
    assert np.allclose(np.asarray(st['weight']), [expect])
    assert expect < 10.0                                   # depressed


def test_no_spike_no_change():
    s = stdp_synapse(weight=10.0)
    st, _ = s.update({'weight': jnp.array([10.0])}, _ctx(0.0, 0.0, 0.7, 0.7))
    assert np.allclose(np.asarray(st['weight']), [10.0])


# -- simultaneous pre&post: each side excludes the current step's own spike -
def test_simultaneous_excludes_current_step_spike():
    s = stdp_synapse(weight=10.0, lambda_=0.1)
    # pre_trace/post_trace each carry only this step's +1 -> kplus=kminus=0
    st, _ = s.update({'weight': jnp.array([10.0])}, _ctx(1.0, 1.0, 1.0, 1.0))
    assert np.allclose(np.asarray(st['weight']), [10.0])   # isolated pair: no net change
    # with a prior post trace (1.0 + this step's 1.0 = 2.0) depression sees 1.0
    st2, _ = s.update({'weight': jnp.array([10.0])}, _ctx(1.0, 1.0, 1.0, 2.0))
    assert np.allclose(np.asarray(st2['weight']), [_host_depress(10.0, 1.0, lam=0.1)])


# -- clamps ----------------------------------------------------------------
def test_wmax_clamp_from_above():
    s = stdp_synapse(weight=99.0, Wmax=100.0, lambda_=5.0)  # huge step
    st, _ = s.update({'weight': jnp.array([99.0])}, _ctx(0.0, 1.0, 10.0, 0.0))
    assert np.allclose(np.asarray(st['weight']), [100.0])  # clamped to Wmax


def test_depress_clamp_to_zero():
    s = stdp_synapse(weight=1.0, Wmax=100.0, lambda_=5.0)
    st, _ = s.update({'weight': jnp.array([1.0])}, _ctx(1.0, 0.0, 0.0, 10.0))
    assert np.allclose(np.asarray(st['weight']), [0.0])    # clamped to 0


# -- additive limit (mu = 0) ----------------------------------------------
def test_mu_zero_additive_limit():
    s = stdp_synapse(weight=10.0, Wmax=100.0, lambda_=0.1, mu_plus=0.0, mu_minus=0.0)
    st, _ = s.update({'weight': jnp.array([10.0])}, _ctx(0.0, 1.0, 0.5, 0.0))
    expect = _host_facilitate(10.0, 0.5, lam=0.1, mu_plus=0.0)
    assert np.allclose(np.asarray(st['weight']), [expect])


# -- per-edge freezing -----------------------------------------------------
def test_frozen_non_firing_edges():
    s = stdp_synapse(weight=10.0, lambda_=0.1)
    # edge 0 depresses (pre fires), edge 1 untouched
    ctx = _ctx([1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5], E=2)
    st, _ = s.update({'weight': jnp.array([10.0, 10.0])}, ctx)
    assert np.allclose(np.asarray(st['weight'])[1], 10.0)         # frozen
    assert np.asarray(st['weight'])[0] < 10.0                     # depressed


# -- jit/vmap/grad smoke ---------------------------------------------------
def test_vmap_grad_smoke():
    s = stdp_synapse(weight=10.0, lambda_=0.1)

    def run(w):
        st, _ = s.update({'weight': w}, _ctx(0.0, 1.0, 0.5, 0.0))
        return jnp.sum(st['weight'])

    g = jax.grad(run)(jnp.array([10.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    out = jax.vmap(run)(jnp.array([[10.0], [20.0], [30.0]]))
    assert out.shape == (3,)
