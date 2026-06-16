# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``stdp_pl_synapse_hom``.

Kernel-level checks of the power-law STDP rule: additive sub-linear potentiation
``w + lambda*w^mu*K+`` gated on the post spike, linear depression
``max(w - alpha*lambda*w*K-, 0)`` gated on the pre spike, the absence of any
upper bound, the lower clip to zero, the simultaneous-spike exclusion, and
per-edge freezing. Live-NEST equivalence is covered by the parity test.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import stdp_pl_synapse_hom
from brainpy_state._nest_network._event_plastic import KernelContext


def _ctx(pre_spike, post_spike, pre_trace, post_trace, E=1, t=10.0, dt=1.0):
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    return KernelContext(g(pre_spike), g(post_spike), g(pre_trace), g(post_trace),
                         jnp.asarray(t), jnp.asarray(dt), jax.random.key(0))


def _host_fac(w, kplus, lam=0.1, mu=0.4):
    return w + lam * (w ** mu) * kplus


def _host_dep(w, kminus, alpha=1.0, lam=0.1):
    nw = w - alpha * lam * w * kminus
    return nw if nw > 0.0 else 0.0


# -- spec contract ---------------------------------------------------------
def test_spec_attributes_and_defaults():
    s = stdp_pl_synapse_hom()
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.edge_state_init() == {}
    assert s.lambda_ == 0.1 and s.alpha == 1.0 and s.mu == 0.4
    assert not hasattr(s, 'Wmax')                          # power-law: no upper bound
    assert float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms)) == 20.0
    assert float(u.Quantity(s.post_trace_tau).to_decimal(u.ms)) == 20.0
    assert u.get_unit(s.weight) == u.pA


def test_validation():
    with pytest.raises(ValueError):
        stdp_pl_synapse_hom(tau_plus=-1.0 * u.ms)
    with pytest.raises(ValueError):
        stdp_pl_synapse_hom(tau_minus=-1.0 * u.ms)
    with pytest.raises(ValueError):
        stdp_pl_synapse_hom(lambda_=-0.1)
    with pytest.raises(ValueError):
        stdp_pl_synapse_hom(alpha=-0.1)


# -- power-law potentiation on the post spike (K+ = pre trace) -------------
def test_potentiation_power_law_on_post():
    s = stdp_pl_synapse_hom(weight=5.0, lambda_=0.1, mu=0.4)
    st, w_eff = s.update({'weight': jnp.array([5.0])}, _ctx(0.0, 1.0, 0.7, 0.0))
    expect = _host_fac(5.0, 0.7)
    assert np.allclose(np.asarray(st['weight']), [expect])
    assert np.allclose(np.asarray(w_eff), [expect])
    assert expect > 5.0


def test_depression_linear_on_pre():
    s = stdp_pl_synapse_hom(weight=5.0, lambda_=0.1, alpha=1.0)
    st, _ = s.update({'weight': jnp.array([5.0])}, _ctx(1.0, 0.0, 0.0, 0.6))
    expect = _host_dep(5.0, 0.6)
    assert np.allclose(np.asarray(st['weight']), [expect])
    assert expect < 5.0


def test_no_spike_no_change():
    s = stdp_pl_synapse_hom(weight=5.0)
    st, _ = s.update({'weight': jnp.array([5.0])}, _ctx(0.0, 0.0, 0.8, 0.8))
    assert np.allclose(np.asarray(st['weight']), [5.0])


# -- no upper bound: potentiation grows past any nominal cap ---------------
def test_no_upper_bound():
    s = stdp_pl_synapse_hom(weight=90.0, lambda_=0.5, mu=0.4)
    st, _ = s.update({'weight': jnp.array([90.0])}, _ctx(0.0, 1.0, 5.0, 0.0))
    assert float(np.asarray(st['weight'])[0]) > 100.0      # unbounded above


def test_depress_clamp_to_zero():
    s = stdp_pl_synapse_hom(weight=1.0, lambda_=5.0, alpha=2.0)
    st, _ = s.update({'weight': jnp.array([1.0])}, _ctx(1.0, 0.0, 0.0, 10.0))
    assert np.allclose(np.asarray(st['weight']), [0.0])


# -- simultaneous pre&post: exclude the current step's own spike -----------
def test_simultaneous_excludes_current_step_spike():
    s = stdp_pl_synapse_hom(weight=5.0, lambda_=0.1)
    st, _ = s.update({'weight': jnp.array([5.0])}, _ctx(1.0, 1.0, 1.0, 1.0))
    assert np.allclose(np.asarray(st['weight']), [5.0])    # isolated pair: no net change
    st2, _ = s.update({'weight': jnp.array([5.0])}, _ctx(1.0, 1.0, 1.0, 2.0))
    assert np.allclose(np.asarray(st2['weight']), [_host_dep(5.0, 1.0)])


def test_frozen_non_firing_edges():
    s = stdp_pl_synapse_hom(weight=5.0, lambda_=0.1)
    ctx = _ctx([1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5], E=2)
    st, _ = s.update({'weight': jnp.array([5.0, 5.0])}, ctx)
    assert np.allclose(np.asarray(st['weight'])[1], 5.0)         # frozen
    assert np.asarray(st['weight'])[0] < 5.0                     # depressed


def test_vmap_grad_smoke():
    s = stdp_pl_synapse_hom(weight=5.0, lambda_=0.1)

    def run(w):
        st, _ = s.update({'weight': w}, _ctx(0.0, 1.0, 0.5, 0.0))
        return jnp.sum(st['weight'])

    g = jax.grad(run)(jnp.array([5.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    out = jax.vmap(run)(jnp.array([[5.0], [10.0], [20.0]]))
    assert out.shape == (3,)
