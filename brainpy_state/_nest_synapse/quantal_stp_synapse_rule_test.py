# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``quantal_stp_synapse``.

Stochastic model: tests assert *distributional* properties (Monte-Carlo means
over many edges), since the PRNG differs from NEST.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import quantal_stp_synapse, tsodyks2_synapse
from brainpy_state._nest_network._event_plastic import KernelContext


def _ctx(key, t, E):
    z = jnp.zeros(E)
    return KernelContext(jnp.ones(E), z, z, z, jnp.asarray(t), jnp.asarray(0.1), key)


def test_defaults():
    s = quantal_stp_synapse()
    assert s.stochastic is True and s.n == 1
    init = s.edge_state_init()
    assert init['t_lastspike'] == -1.0 and init['a'] == 1.0 and init['u'] == 0.5


def test_validation():
    for kw in [dict(U=1.5), dict(u=1.5), dict(tau_rec=0 * u.ms), dict(tau_fac=-1 * u.ms),
               dict(n=1.5)]:
        with pytest.raises(ValueError):
            quantal_stp_synapse(**kw)


def test_first_spike_mean_release_is_U():
    # large n, weight=1/n -> mean delivered on the first spike ~ U
    n = 50
    s = quantal_stp_synapse(U=0.3, n=n, weight=(1.0 / n) * u.pA)
    E = 4000
    st = {'weight': jnp.full((E,), 1.0 / n), 'u': jnp.full((E,), 0.3),
          'a': jnp.full((E,), float(n)), 't_lastspike': jnp.full((E,), -1.0)}
    st, w = s.update(st, _ctx(jax.random.key(0), 10.0, E))
    assert float(jnp.mean(w)) == pytest.approx(0.3, abs=2e-2)


def test_release_mean_matches_tsodyks2_limit():
    # large n -> mean quantal release tracks the deterministic tsodyks2 train
    n = 50
    s = quantal_stp_synapse(U=0.3, tau_rec=200. * u.ms, tau_fac=0. * u.ms, n=n,
                            weight=(1.0 / n) * u.pA)
    s2 = tsodyks2_synapse(U=0.3, tau_rec=200. * u.ms, tau_fac=0. * u.ms, weight=1.0 * u.pA)
    ts = [10., 20., 30., 40., 50.]
    E = 8000
    st = {'weight': jnp.full((E,), 1.0 / n), 'u': jnp.full((E,), 0.3),
          'a': jnp.full((E,), float(n)), 't_lastspike': jnp.full((E,), -1.0)}
    key = jax.random.key(0)
    means = []
    for t in ts:
        key, sub = jax.random.split(key)
        st, w = s.update(st, _ctx(sub, t, E))
        means.append(float(jnp.mean(w)))
    st2 = {'weight': jnp.array([1.]), 'u': jnp.array([0.3]), 'x': jnp.array([1.]),
           't_lastspike': jnp.array([-1.])}
    ref = []
    z = jnp.zeros(1)
    for t in ts:
        ctx = KernelContext(jnp.array([1.]), z, z, z, jnp.asarray(t), jnp.asarray(0.1),
                            jax.random.key(0))
        st2, w = s2.update(st2, ctx)
        ref.append(float(w[0]))
    assert np.allclose(means, ref, atol=3e-2)


def test_release_depletes_available_sites():
    # after release, mean available 'a' drops below n
    n = 40
    s = quantal_stp_synapse(U=0.8, n=n, weight=(1.0 / n) * u.pA)
    E = 3000
    st = {'weight': jnp.full((E,), 1.0 / n), 'u': jnp.full((E,), 0.8),
          'a': jnp.full((E,), float(n)), 't_lastspike': jnp.full((E,), -1.0)}
    st, _ = s.update(st, _ctx(jax.random.key(1), 10.0, E))
    assert float(jnp.mean(st['a'])) < float(n)
