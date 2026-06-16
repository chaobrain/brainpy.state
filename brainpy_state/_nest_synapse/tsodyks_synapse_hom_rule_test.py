# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``tsodyks_synapse_hom``."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import tsodyks_synapse_hom
from brainpy_state._nest_network._event_plastic import KernelContext


def _ref_hom(seq_h, U=0.5, tpsc=3., tfac=0., trec=800., w=1.0):
    """Inline NEST ``_hom`` plain-exp propagator reference (``+Pxz*z`` form)."""
    x, y, u_ = 1., 0., 0.
    out = []
    for h in seq_h:
        Puu = 0. if tfac == 0 else math.exp(-h / tfac)
        Pyy = math.exp(-h / tpsc)
        Pzz = math.exp(-h / trec)
        Pxy = ((Pzz - 1) * trec - (Pyy - 1) * tpsc) / (tpsc - trec)
        Pxz = 1 - Pzz
        z = 1 - x - y
        u_ *= Puu
        x = x + Pxy * y + Pxz * z
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


def test_defaults_and_homogeneous_flag():
    s = tsodyks_synapse_hom()
    assert s.is_homogeneous_weight is True
    assert s.stochastic is False
    assert s.pre_trace_tau is None and s.post_trace_tau is None
    assert u.get_mantissa(s.weight) == 1.0
    init = s.edge_state_init()
    assert init['t_lastspike'] == 0.0
    assert init['x'] == 1.0 and init['y'] == 0.0 and init['u'] == 0.0


def test_validation():
    for kw in [dict(U=1.5), dict(tau_rec=0 * u.ms), dict(tau_psc=-1 * u.ms),
               dict(tau_fac=-1 * u.ms), dict(x=0.7, y=0.7)]:
        with pytest.raises(ValueError):
            tsodyks_synapse_hom(**kw)


def test_hom_does_not_validate_u():
    # the _hom variant intentionally does NOT range-check ``u``
    s = tsodyks_synapse_hom(u=1.5)
    assert s.edge_state_init()['u'] == 1.5


def test_effective_weight_train_matches_plain_exp_reference():
    s = tsodyks_synapse_hom(U=0.5, weight=1.0 * u.pA)
    init = s.edge_state_init()
    state = {'weight': jnp.asarray(1.0),
             **{k: jnp.array([float(v)]) for k, v in init.items()}}
    got = []
    for t in [50., 100., 150.]:
        state, w_eff = s.update(state, _spike_ctx(t))
        got.append(float(w_eff[0]))
    assert np.allclose(got, _ref_hom([50., 50., 50.]), atol=1e-9)


def test_non_firing_edges_are_frozen():
    s = tsodyks_synapse_hom(U=0.5)
    init = s.edge_state_init()
    state = {'weight': jnp.asarray(1.0),
             **{k: jnp.array([float(v), float(v)]) for k, v in init.items()}}
    z = jnp.zeros(2)
    # only edge 0 fires
    ctx = KernelContext(jnp.array([1., 0.]), z, z, z, jnp.asarray(50.),
                        jnp.asarray(0.1), jax.random.key(0))
    new_state, w_eff = s.update(state, ctx)
    assert float(new_state['t_lastspike'][1]) == 0.0      # frozen
    assert float(new_state['t_lastspike'][0]) == 50.0     # advanced
    assert float(w_eff[1]) == pytest.approx(0.5)          # delivered "as if fired"
