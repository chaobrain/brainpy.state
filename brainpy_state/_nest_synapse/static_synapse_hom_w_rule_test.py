# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``static_synapse_hom_w``."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import static_synapse_hom_w
from brainpy_state._nest_network._event_plastic import KernelContext


def _ctx(E):
    z = jnp.zeros(E)
    return KernelContext(jnp.ones(E), z, z, z, jnp.asarray(1.0), jnp.asarray(0.1),
                         jax.random.key(0))


def test_hom_w_is_homogeneous_and_rejects_per_edge_weight():
    s = static_synapse_hom_w(weight=2.0 * u.pA)
    assert s.is_homogeneous_weight is True
    assert s.edge_state_init() == {}
    state = {'weight': jnp.asarray(2.0)}   # 0-d shared
    new_state, w_eff = s.update(state, _ctx(3))
    assert np.allclose(np.asarray(jnp.broadcast_to(w_eff, (3,))), 2.0)
    assert float(new_state['weight']) == 2.0   # unchanged


def test_default_weight_is_one_pa():
    s = static_synapse_hom_w()
    assert u.get_mantissa(s.weight) == 1.0
    assert s.weight_unit == u.pA


def test_check_rejects_weight_in_synspec():
    with pytest.raises(ValueError):
        static_synapse_hom_w.check_synapse_params({'weight': 1.0})
    # delay / receptor_type in the spec are allowed
    static_synapse_hom_w.check_synapse_params({'delay': 1.0})
    static_synapse_hom_w.check_synapse_params(None)


def test_set_weight_rejected():
    with pytest.raises(ValueError):
        static_synapse_hom_w.set_weight(3.0)


def test_delay_validation():
    with pytest.raises(ValueError):
        static_synapse_hom_w(delay=-1.0 * u.ms)
