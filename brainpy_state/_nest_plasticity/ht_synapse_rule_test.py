# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``ht_synapse``.

Kernel-level checks of the Hill-Tononi vesicle-pool depression rule: the
recover -> emit -> deplete -> update ordering, the static stored weight with the
delivered ``w_eff = w * P_send`` observable, recovery from ``t = 0`` on the first
spike (NEST's ``t_lastspike_ = 0`` init, unlike ``tsodyks2``'s ``-1`` skip),
multiplicative depletion, ``delta_P`` extremes, and per-edge freezing. Live-NEST
equivalence is covered by the parity test.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import ht_synapse
from brainpy_state._nest_network._event_plastic import KernelContext


def _ctx(pre_spike, t, E=1, dt=0.1):
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    z = jnp.zeros(E)
    return KernelContext(g(pre_spike), z, z, z, jnp.asarray(t), jnp.asarray(dt),
                         jax.random.key(0))


def _ref(spike_times, tau_P, delta_P, w=1.0, P0=1.0):
    """Closed-form event-driven (w_eff, P_send, P_post) per spike (NEST send order)."""
    P, t_last, out = P0, 0.0, []
    for t in spike_times:
        P_send = 1.0 - (1.0 - P) * math.exp(-(t - t_last) / tau_P)
        w_eff = w * P_send
        P = (1.0 - delta_P) * P_send
        t_last = t
        out.append((w_eff, P_send, P))
    return out


def _drive(s, spike_times):
    """Run the kernel event-by-event; return (w_eff, P_post) per spike."""
    state = {'weight': jnp.array([float(u.get_mantissa(s.weight))]), **{
        k: jnp.array([float(v)]) for k, v in s.edge_state_init().items()}}
    out = []
    for t in spike_times:
        state, w_eff = s.update(state, _ctx(1.0, t))
        out.append((float(np.asarray(w_eff)[0]), float(np.asarray(state['P'])[0])))
    return out


# -- spec contract ---------------------------------------------------------
def test_spec_attributes_and_defaults():
    s = ht_synapse()
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.pre_trace_tau is None and s.post_trace_tau is None
    assert s.edge_state_init() == {'P': 1.0, 't_lastspike': 0.0}
    assert s.tau_P == 500.0 and s.delta_P == 0.125
    assert u.get_unit(s.weight) == u.pA


def test_initial_pool_settable():
    s = ht_synapse(P=0.3)
    assert s.edge_state_init()['P'] == 0.3


def test_validation():
    for kw in [dict(tau_P=0.0 * u.ms), dict(tau_P=-1.0 * u.ms),
               dict(delta_P=-0.1), dict(delta_P=1.1), dict(P=-0.1), dict(P=1.1)]:
        with pytest.raises(ValueError):
            ht_synapse(**kw)


# -- recover -> emit -> deplete -> update ----------------------------------
def test_first_spike_full_pool_delivers_weight():
    s = ht_synapse(weight=2.5 * u.pA, tau_P=300.0 * u.ms, delta_P=0.2)
    (w_eff, P_post), = _drive(s, [10.0])
    assert np.isclose(w_eff, 2.5)            # P_send == 1 (pool full)
    assert np.isclose(P_post, 0.8)           # depleted by delta_P


def test_train_matches_closed_form():
    s = ht_synapse(weight=2.0 * u.pA, tau_P=300.0 * u.ms, delta_P=0.2)
    times = [10.0, 20.0, 30.0, 40.0]
    got = _drive(s, times)
    ref = _ref(times, 300.0, 0.2, w=2.0, P0=1.0)
    for (w_g, P_g), (w_r, _, P_r) in zip(got, ref):
        assert np.isclose(w_g, w_r) and np.isclose(P_g, P_r)


def test_depletion_is_multiplicative():
    # rapid train (h << tau_P): pool depresses geometrically toward 0.
    s = ht_synapse(weight=1.0 * u.pA, tau_P=1e6 * u.ms, delta_P=0.25)
    got = _drive(s, [1.0, 2.0, 3.0])
    # negligible recovery: P_send ~ 0.75**k, w_eff ~ same
    assert np.isclose(got[0][0], 1.0, atol=1e-3)
    assert np.isclose(got[1][0], 0.75, atol=1e-3)
    assert np.isclose(got[2][0], 0.75 ** 2, atol=1e-3)


# -- NEST init fidelity: recover from t=0 on the first spike ----------------
def test_partial_initial_pool_recovers_from_zero():
    s = ht_synapse(weight=1.0 * u.pA, tau_P=200.0 * u.ms, delta_P=0.0, P=0.5)
    (w_eff, _), = _drive(s, [100.0])
    expect = 1.0 - (1.0 - 0.5) * math.exp(-100.0 / 200.0)   # recovers from t_last=0
    assert np.isclose(w_eff, expect)
    assert not np.isclose(w_eff, 0.5)                        # NOT the -1 skip behaviour


# -- delta_P extremes ------------------------------------------------------
def test_delta_p_zero_disables_depression():
    s = ht_synapse(weight=1.0 * u.pA, tau_P=1e9 * u.ms, delta_P=0.0)
    got = _drive(s, [1.0, 2.0, 3.0])
    assert all(np.isclose(w, 1.0) for w, _ in got)          # pool never depletes

def test_delta_p_one_fully_depletes():
    s = ht_synapse(weight=1.0 * u.pA, tau_P=1e9 * u.ms, delta_P=1.0)
    got = _drive(s, [1.0, 2.0])
    assert np.isclose(got[0][1], 0.0)                        # P_post == 0 after spike
    assert np.isclose(got[1][0], 0.0, atol=1e-6)             # next delivery ~ 0


# -- freezing: no pre spike -> state held ----------------------------------
def test_no_spike_holds_state():
    s = ht_synapse(weight=1.0 * u.pA, tau_P=300.0 * u.ms, delta_P=0.5)
    state = {'weight': jnp.array([1.0]), 'P': jnp.array([0.6]),
             't_lastspike': jnp.array([5.0])}
    new, w_eff = s.update(state, _ctx(0.0, 50.0))
    assert np.allclose(np.asarray(new['P']), [0.6])          # P frozen
    assert np.allclose(np.asarray(new['t_lastspike']), [5.0])  # t frozen


def test_frozen_non_firing_edges():
    s = ht_synapse(weight=1.0 * u.pA, tau_P=300.0 * u.ms, delta_P=0.5)
    state = {'weight': jnp.array([1.0, 1.0]), 'P': jnp.array([1.0, 1.0]),
             't_lastspike': jnp.array([0.0, 0.0])}
    new, _ = s.update(state, _ctx([1.0, 0.0], 10.0, E=2))
    assert float(np.asarray(new['P'])[0]) < 1.0             # edge 0 fired -> depleted
    assert np.allclose(np.asarray(new['P'])[1], 1.0)        # edge 1 frozen
    assert np.allclose(np.asarray(new['t_lastspike'])[1], 0.0)


def test_vmap_grad_smoke():
    s = ht_synapse(tau_P=300.0 * u.ms, delta_P=0.2)

    def run(w):
        state = {'weight': w, 'P': jnp.array([1.0]), 't_lastspike': jnp.array([0.0])}
        _, w_eff = s.update(state, _ctx(1.0, 10.0))
        return jnp.sum(w_eff)

    g = jax.grad(run)(jnp.array([2.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    out = jax.vmap(run)(jnp.array([[1.0], [2.0], [3.0]]))
    assert out.shape == (3,)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
