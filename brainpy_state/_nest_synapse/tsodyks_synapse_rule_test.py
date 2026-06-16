# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``tsodyks_synapse`` (expm1 form)."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import tsodyks_synapse
from brainpy_state._nest_network.event_plastic import KernelContext


def _ref_tsodyks(seq_h, U=0.5, tpsc=3., tfac=0., trec=800., w=1.0):
    """Inline NEST expm1 propagator reference (``-Pzz*z`` form)."""
    x, y, u_, = 1., 0., 0.
    out = []
    for h in seq_h:
        Puu = 0. if tfac == 0 else math.exp(-h / tfac)
        Pyy = math.exp(-h / tpsc)
        Pzz = math.expm1(-h / trec)
        Pxy = (Pzz * trec - (Pyy - 1) * tpsc) / (tpsc - trec)
        z = 1 - x - y
        u_ *= Puu
        x = x + Pxy * y - Pzz * z
        y *= Pyy
        u_ += U * (1 - u_)
        d = u_ * x
        x -= d
        y += d
        out.append(d * w)
    return out


def _spike_ctx(t):
    z = jnp.zeros(1)
    return KernelContext(jnp.array([1.]), z, z, z, jnp.asarray(t), jnp.asarray(0.1),
                         jax.random.key(0))


def test_defaults():
    s = tsodyks_synapse()
    assert s.is_homogeneous_weight is False and s.stochastic is False
    assert s.pre_trace_tau is None and s.post_trace_tau is None
    init = s.edge_state_init()
    assert init['t_lastspike'] == 0.0
    assert init['x'] == 1.0 and init['y'] == 0.0 and init['u'] == 0.0


def test_validation():
    for kw in [dict(U=1.5), dict(u=1.5), dict(tau_rec=0 * u.ms), dict(tau_psc=0 * u.ms),
               dict(tau_fac=-1 * u.ms), dict(x=0.7, y=0.7)]:
        with pytest.raises(ValueError):
            tsodyks_synapse(**kw)


def test_effective_weight_train_matches_inline_reference():
    s = tsodyks_synapse(U=0.5, weight=1.0 * u.pA)
    init = s.edge_state_init()
    state = {k: jnp.array([float(v)]) for k, v in {**init, 'weight': 1.0}.items()}
    got = []
    for t in [50., 100., 150.]:
        state, w_eff = s.update(state, _spike_ctx(t))
        got.append(float(w_eff[0]))
    assert np.allclose(got, _ref_tsodyks([50., 50., 50.]), atol=1e-9)


def test_depression_train_is_decreasing():
    # depression-dominated (tau_fac=0): repeated spikes deplete -> amplitudes fall
    s = tsodyks_synapse(U=0.67, tau_rec=450. * u.ms, tau_fac=0. * u.ms, weight=250. * u.pA)
    init = s.edge_state_init()
    state = {k: jnp.array([float(v)]) for k, v in {**init, 'weight': 250.0}.items()}
    amps = []
    for t in [10., 30., 50., 70., 90.]:
        state, w_eff = s.update(state, _spike_ctx(t))
        amps.append(float(w_eff[0]))
    assert all(a2 < a1 for a1, a2 in zip(amps, amps[1:]))


def test_weight_sign_passes_through():
    s = tsodyks_synapse(U=0.5, weight=-2.0 * u.pA)
    init = s.edge_state_init()
    state = {k: jnp.array([float(v)]) for k, v in {**init, 'weight': -2.0}.items()}
    _, w_eff = s.update(state, _spike_ctx(50.))
    assert float(w_eff[0]) < 0.0   # inhibitory weight stays negative


def test_non_scalar_initial_state_rejected():
    # initial x/y/u must be scalars (per-edge broadcast happens in the substrate)
    with pytest.raises(ValueError, match='must be scalar'):
        tsodyks_synapse(x=[1.0, 0.5])


def test_bare_number_time_constants_are_ms():
    # to_ms interprets bare floats as milliseconds (no unit attached)
    s = tsodyks_synapse(U=0.5, tau_psc=3.0, tau_rec=800.0, tau_fac=0.0)
    assert s.tau_psc == 3.0 and s.tau_rec == 800.0 and s.tau_fac == 0.0
