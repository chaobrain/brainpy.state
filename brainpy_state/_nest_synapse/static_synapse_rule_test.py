# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``static_synapse``."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import static_synapse
from brainpy_state._nest_network.event_plastic import KernelContext


def _ctx(E):
    z = jnp.zeros(E)
    return KernelContext(jnp.ones(E), z, z, z, jnp.asarray(1.0), jnp.asarray(0.1),
                         jax.random.key(0))


def test_defaults_match_nest():
    s = static_synapse()
    assert u.get_mantissa(s.weight) == 1.0
    assert s.weight_unit == u.pA
    assert s.is_homogeneous_weight is False and s.stochastic is False
    assert s.pre_trace_tau is None and s.post_trace_tau is None
    assert s.edge_state_init() == {}
    assert s.receptor_type == 0


def test_constant_weight_no_drift():
    s = static_synapse(weight=3.0 * u.pA)
    state = {'weight': jnp.full((4,), 3.0)}
    for _ in range(5):
        state, w_eff = s.update(state, _ctx(4))
    assert np.allclose(np.asarray(state['weight']), 3.0)
    assert np.allclose(np.asarray(w_eff), 3.0)


def test_bare_weight_coerced_to_pa():
    s = static_synapse(weight=5.0)
    assert u.get_mantissa(s.weight) == 5.0 and s.weight_unit == u.pA


def test_negative_weight_preserved():
    s = static_synapse(weight=-2.0 * u.pA)
    _, w_eff = s.update({'weight': jnp.full((2,), -2.0)}, _ctx(2))
    assert np.allclose(np.asarray(w_eff), -2.0)


def test_delay_validation():
    with pytest.raises(ValueError):
        static_synapse(delay=-1.0 * u.ms)
    with pytest.raises(ValueError):
        static_synapse(delay=0.0 * u.ms)
    with pytest.raises(ValueError, match='finite'):
        static_synapse(delay=np.inf * u.ms)


def test_receptor_type_validation():
    with pytest.raises(ValueError):
        static_synapse(receptor_type=-1)
