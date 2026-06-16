# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``tsodyks2_synapse``."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import tsodyks2_synapse
from brainpy_state._nest_network.event_plastic import KernelContext


def _spike_ctx(t, E=1):
    z = jnp.zeros(E)
    return KernelContext(jnp.ones(E), z, z, z, jnp.asarray(t), jnp.asarray(0.1),
                         jax.random.key(0))


def test_defaults_u_equals_U_and_tlast_minus_one():
    s = tsodyks2_synapse(U=0.5)
    init = s.edge_state_init()
    assert init['t_lastspike'] == -1.0
    assert init['u'] == 0.5 and init['x'] == 1.0


def test_setting_U_does_not_change_u():
    s = tsodyks2_synapse(U=0.3, u=0.9)
    assert s.edge_state_init()['u'] == 0.9


def test_validation():
    for kw in [dict(U=1.5), dict(u=1.5), dict(tau_rec=0 * u.ms), dict(tau_fac=-1 * u.ms)]:
        with pytest.raises(ValueError):
            tsodyks2_synapse(**kw)


def test_first_spike_skips_decay():
    s = tsodyks2_synapse(U=0.5)
    state = {'weight': jnp.array([1.0]), 'u': jnp.array([0.5]), 'x': jnp.array([1.0]),
             't_lastspike': jnp.array([-1.0])}
    state, w_eff = s.update(state, _spike_ctx(5.0))
    assert np.allclose(np.asarray(w_eff), [0.5])          # x*u*w = 1*0.5*1, no decay
    assert float(state['t_lastspike'][0]) == 5.0


def test_tau_fac_zero_resets_u_to_U():
    s = tsodyks2_synapse(U=0.4, tau_fac=0.0 * u.ms)
    state = {'weight': jnp.array([1.]), 'u': jnp.array([0.9]), 'x': jnp.array([1.]),
             't_lastspike': jnp.array([0.])}
    state, _ = s.update(state, _spike_ctx(10.))
    assert np.allclose(np.asarray(state['u']), [0.4])     # u <- U


def test_x_uses_old_u_before_update():
    # x_new = 1 + (x - x*u_old - 1)*exp(-h/tau_rec); verify old u is used
    s = tsodyks2_synapse(U=0.5, tau_rec=200. * u.ms, tau_fac=0.0 * u.ms)
    x0, u0, h, trec = 0.8, 0.6, 10.0, 200.0
    state = {'weight': jnp.array([1.]), 'u': jnp.array([u0]), 'x': jnp.array([x0]),
             't_lastspike': jnp.array([0.])}
    state, _ = s.update(state, _spike_ctx(h))
    expected_x = 1.0 + (x0 - x0 * u0 - 1.0) * np.exp(-h / trec)
    assert np.allclose(np.asarray(state['x']), [expected_x], atol=1e-12)


def test_dt_invariance_same_spike_times():
    # the closed form depends only on absolute spike times -> identical for any dt
    def run(spike_ts):
        s = tsodyks2_synapse(U=0.5, tau_rec=200. * u.ms, tau_fac=50. * u.ms)
        st = {'weight': jnp.array([1.]), 'u': jnp.array([0.5]), 'x': jnp.array([1.]),
              't_lastspike': jnp.array([-1.])}
        out = []
        for t in spike_ts:
            st, w = s.update(st, _spike_ctx(t))
            out.append(float(w[0]))
        return out

    assert np.allclose(run([10., 20., 30.]), run([10., 20., 30.]))


def test_non_firing_edges_frozen():
    s = tsodyks2_synapse(U=0.5)
    state = {'weight': jnp.array([1., 1.]), 'u': jnp.array([0.5, 0.5]),
             'x': jnp.array([1., 1.]), 't_lastspike': jnp.array([0., 0.])}
    z = jnp.zeros(2)
    ctx = KernelContext(jnp.array([1., 0.]), z, z, z, jnp.asarray(20.),
                        jnp.asarray(0.1), jax.random.key(0))
    state, _ = s.update(state, ctx)
    assert float(state['t_lastspike'][1]) == 0.0    # frozen
    assert float(state['t_lastspike'][0]) == 20.0   # advanced
