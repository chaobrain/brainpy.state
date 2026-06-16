# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``bernoulli_synapse``.

``bernoulli_synapse`` is a *static* synapse with stochastic transmission: each
presynaptic spike is delivered (full ``weight``) with probability ``p_transmit``,
independently per connection, otherwise dropped. No weight state evolves. These
tests cover the rule kernel and the per-edge ``ctx.key`` Bernoulli gate without
NEST (distributional / closed-form), plus parameter validation and jit/vmap/grad.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import bernoulli_synapse
from brainpy_state._nest_network._event_plastic import KernelContext


def _ctx(E, key=None, pre=None):
    """A KernelContext with all edges fired by default and the given key."""
    z = jnp.zeros(E)
    pre = jnp.ones(E) if pre is None else jnp.asarray(pre)
    key = jax.random.key(0) if key is None else key
    return KernelContext(pre, z, z, z, jnp.asarray(1.0), jnp.asarray(0.1), key)


def test_defaults_match_nest():
    s = bernoulli_synapse()
    assert u.get_mantissa(s.weight) == 1.0
    assert s.weight_unit == u.pA
    assert s.is_homogeneous_weight is False
    assert s.stochastic is True                      # drives the ctx.key seam
    assert s.pre_trace_tau is None and s.post_trace_tau is None
    assert s.edge_state_init() == {}                 # memoryless
    assert s.receptor_type == 0
    assert float(s.p_transmit) == 1.0                # NEST default


def test_p_one_equals_static():
    # p_transmit == 1 -> every fired edge delivers the full weight (== static).
    s = bernoulli_synapse(weight=7.0 * u.pA, p_transmit=1.0)
    state = {'weight': jnp.full((5,), 7.0)}
    new_state, w_eff = s.update(state, _ctx(5))
    assert np.allclose(np.asarray(w_eff), 7.0)
    assert np.allclose(np.asarray(new_state['weight']), 7.0)   # weight unchanged


def test_p_zero_no_delivery():
    s = bernoulli_synapse(weight=7.0 * u.pA, p_transmit=0.0)
    _, w_eff = s.update({'weight': jnp.full((5,), 7.0)}, _ctx(5))
    assert np.allclose(np.asarray(w_eff), 0.0)


def test_w_eff_gated_by_pre_spike():
    # An edge whose pre did not fire delivers 0 regardless of the draw (p=1).
    s = bernoulli_synapse(weight=3.0 * u.pA, p_transmit=1.0)
    pre = jnp.array([1.0, 0.0, 1.0, 0.0])
    _, w_eff = s.update({'weight': jnp.full((4,), 3.0)}, _ctx(4, pre=pre))
    assert np.allclose(np.asarray(w_eff), [3.0, 0.0, 3.0, 0.0])


def test_memoryless_state_unchanged():
    s = bernoulli_synapse(weight=2.0 * u.pA, p_transmit=0.5)
    state = {'weight': jnp.full((3,), 2.0)}
    new_state, _ = s.update(state, _ctx(3, key=jax.random.key(11)))
    assert set(new_state) == {'weight'}
    assert np.allclose(np.asarray(new_state['weight']), 2.0)


def test_transmitted_fraction_converges_to_p():
    # Over E*T independent fired draws the transmitted fraction -> p_transmit.
    p = 0.3
    s = bernoulli_synapse(weight=1.0 * u.pA, p_transmit=p)
    E, T = 200, 200                                  # 40k draws
    state = {'weight': jnp.ones((E,))}
    transmitted = 0
    for t in range(T):
        _, w_eff = s.update(state, _ctx(E, key=jax.random.key(t)))
        transmitted += int(np.count_nonzero(np.asarray(w_eff)))
    frac = transmitted / (E * T)
    sigma = np.sqrt(p * (1 - p) / (E * T))
    assert abs(frac - p) < 5 * sigma                 # ~5 sigma band


def test_delivered_count_is_binomial():
    # Per-step delivered count over E edges ~ Binomial(E, p): match mean+variance.
    p, E, T = 0.5, 400, 400
    s = bernoulli_synapse(weight=1.0 * u.pA, p_transmit=p)
    state = {'weight': jnp.ones((E,))}
    counts = np.array([
        int(np.count_nonzero(np.asarray(s.update(state, _ctx(E, key=jax.random.key(t)))[1])))
        for t in range(T)])
    assert abs(counts.mean() - E * p) < 0.05 * E * p                  # mean
    assert abs(counts.var() - E * p * (1 - p)) < 0.20 * E * p * (1 - p)  # variance


def test_per_edge_independence():
    # Two edges from one pre must draw INDEPENDENTLY: joint transmit rate ~ p^2,
    # not p (a shared draw would give p for both-or-neither).
    p, T = 0.5, 8000
    s = bernoulli_synapse(weight=1.0 * u.pA, p_transmit=p)
    state = {'weight': jnp.ones((2,))}
    both = 0
    for t in range(T):
        _, w_eff = s.update(state, _ctx(2, key=jax.random.key(10_000 + t)))
        both += int(np.asarray(w_eff)[0] > 0 and np.asarray(w_eff)[1] > 0)
    joint = both / T
    assert abs(joint - p * p) < 0.03                 # ~p^2 (=0.25), not p (=0.5)


def test_key_determinism_under_jit():
    # Same key -> same mask, eager and under jit.
    s = bernoulli_synapse(weight=1.0 * u.pA, p_transmit=0.4)
    state = {'weight': jnp.ones((16,))}
    ctx = _ctx(16, key=jax.random.key(123))
    _, w1 = s.update(state, ctx)
    _, w2 = s.update(state, ctx)
    _, w3 = jax.jit(s.update)(state, ctx)
    assert np.array_equal(np.asarray(w1), np.asarray(w2))
    assert np.array_equal(np.asarray(w1), np.asarray(w3))


def test_empty_pre_train_no_delivery():
    # No edge fires this step -> all-zero w_eff, no error.
    s = bernoulli_synapse(weight=5.0 * u.pA, p_transmit=0.5)
    _, w_eff = s.update({'weight': jnp.full((4,), 5.0)}, _ctx(4, pre=jnp.zeros(4)))
    assert np.allclose(np.asarray(w_eff), 0.0)


def test_p_transmit_validation():
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        bernoulli_synapse(p_transmit=-0.1)
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        bernoulli_synapse(p_transmit=1.5)


def test_delay_and_receptor_validation():
    with pytest.raises(ValueError):
        bernoulli_synapse(delay=-1.0 * u.ms)
    with pytest.raises(ValueError):
        bernoulli_synapse(delay=0.0 * u.ms)
    with pytest.raises(ValueError, match='finite'):
        bernoulli_synapse(delay=np.inf * u.ms)
    with pytest.raises(ValueError):
        bernoulli_synapse(receptor_type=-1)


def test_bare_weight_coerced_to_pa():
    s = bernoulli_synapse(weight=5.0)
    assert u.get_mantissa(s.weight) == 5.0 and s.weight_unit == u.pA


def test_negative_weight_preserved():
    s = bernoulli_synapse(weight=-2.0 * u.pA, p_transmit=1.0)
    _, w_eff = s.update({'weight': jnp.full((2,), -2.0)}, _ctx(2))
    assert np.allclose(np.asarray(w_eff), -2.0)


def test_vmap_over_keys_smoke():
    # vmap the gate over a batch of keys -> (B, E) masks, shapes preserved.
    s = bernoulli_synapse(weight=1.0 * u.pA, p_transmit=0.5)
    state = {'weight': jnp.ones((8,))}
    keys = jax.random.split(jax.random.key(0), 5)
    fn = jax.vmap(lambda k: s.update(state, _ctx(8, key=k))[1])
    out = fn(keys)
    assert out.shape == (5, 8)


def test_grad_through_weight():
    # w_eff is linear in weight on transmitted edges -> grad flows (p=1).
    s = bernoulli_synapse(weight=1.0 * u.pA, p_transmit=1.0)
    def loss(w):
        _, w_eff = s.update({'weight': w}, _ctx(3))
        return jnp.sum(w_eff)
    g = jax.grad(loss)(jnp.ones((3,)))
    assert np.allclose(np.asarray(g), 1.0)
