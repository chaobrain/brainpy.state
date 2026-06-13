# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``cont_delay_synapse``.

``cont_delay_synapse`` is a *static-delivery* synapse with a **continuous
(sub-dt) delay**: no weight state evolves and the rule kernel is the plain
static rule (``w_eff == weight``). The continuous-delay behaviour lives entirely
in the substrate's ``fractional_delay`` output-carry seam (exercised in
``_event_plastic_test.py``); these tests cover the *spec* contract — attributes,
the ``delay >= dt`` resolution floor (NEST: "continuous delays cannot be shorter
than the simulation resolution"), parameter validation, and jit/vmap/grad on the
rule — without NEST.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import saiunit as u

brainstate.environ.set(precision=64, platform='cpu')
brainstate.environ.set(dt=0.1 * u.ms)                    # resolution floor reference

from brainpy_state import cont_delay_synapse
from brainpy_state._network._event_plastic import KernelContext


def _ctx(E, pre=None):
    """A KernelContext with all edges fired by default (deterministic rule)."""
    z = jnp.zeros(E)
    pre = jnp.ones(E) if pre is None else jnp.asarray(pre)
    return KernelContext(pre, z, z, z, jnp.asarray(1.0), jnp.asarray(0.1), jax.random.key(0))


def test_defaults_match_nest():
    s = cont_delay_synapse()
    assert u.get_mantissa(s.weight) == 1.0
    assert s.weight_unit == u.pA
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.pre_trace_tau is None and s.post_trace_tau is None
    assert s.edge_state_init() == {}                     # memoryless
    assert s.receptor_type == 0
    assert float(u.Quantity(s.delay).to_decimal(u.ms)) == 1.0   # NEST default 1 ms


def test_fractional_delay_flag_is_true():
    # The opt-in flag that drives the substrate's sub-dt output-carry seam.
    assert cont_delay_synapse().fractional_delay is True


def test_update_is_the_static_rule():
    # Deterministic delivery: w_eff == weight on EVERY edge (pre-gating is the
    # substrate matmul, not the rule); no state evolves.
    s = cont_delay_synapse(weight=7.0 * u.pA)
    state = {'weight': jnp.full((5,), 7.0)}
    new_state, w_eff = s.update(state, _ctx(5, pre=jnp.array([1., 0., 1., 0., 1.])))
    assert np.allclose(np.asarray(w_eff), 7.0)           # all edges -> weight
    assert set(new_state) == {'weight'}
    assert np.allclose(np.asarray(new_state['weight']), 7.0)     # unchanged


def test_delay_below_resolution_raises():
    # NEST floor: a continuous delay shorter than the resolution (dt) is rejected.
    with pytest.raises(ValueError, match='resolution'):
        cont_delay_synapse(delay=0.05 * u.ms)            # 0.5 * dt, dt = 0.1 ms


def test_delay_equal_to_resolution_ok():
    s = cont_delay_synapse(delay=0.1 * u.ms)             # exactly one step
    assert float(u.Quantity(s.delay).to_decimal(u.ms)) == 0.1


def test_resolution_check_deferred_when_dt_unset(monkeypatch):
    # Best-effort floor: if no resolution is established yet, the >= dt check
    # cannot be evaluated, so construction defers it (no raise) rather than failing.
    def _no_dt(*a, **k):
        raise KeyError('dt')                             # mimic environ with no dt set
    monkeypatch.setattr(brainstate.environ, 'get_dt', _no_dt)
    s = cont_delay_synapse(delay=0.05 * u.ms)            # below dt, but dt unknown -> ok
    assert np.isclose(float(u.Quantity(s.delay).to_decimal(u.ms)), 0.05)


def test_fractional_delay_value_accepted():
    s = cont_delay_synapse(delay=0.17 * u.ms)            # 1.7 * dt (off-grid)
    assert np.isclose(float(u.Quantity(s.delay).to_decimal(u.ms)), 0.17)


def test_integer_multiple_delay_accepted():
    s = cont_delay_synapse(delay=0.2 * u.ms)             # 2 * dt (on-grid)
    assert np.isclose(float(u.Quantity(s.delay).to_decimal(u.ms)), 0.2)


def test_delay_validation():
    with pytest.raises(ValueError):                      # non-positive
        cont_delay_synapse(delay=-1.0 * u.ms)
    with pytest.raises(ValueError):
        cont_delay_synapse(delay=0.0 * u.ms)
    with pytest.raises(ValueError, match='finite'):
        cont_delay_synapse(delay=np.inf * u.ms)


def test_receptor_validation():
    with pytest.raises(ValueError):
        cont_delay_synapse(receptor_type=-1)


def test_bare_weight_coerced_to_pa():
    s = cont_delay_synapse(weight=5.0)
    assert u.get_mantissa(s.weight) == 5.0 and s.weight_unit == u.pA


def test_negative_weight_preserved():
    s = cont_delay_synapse(weight=-2.0 * u.pA)
    _, w_eff = s.update({'weight': jnp.full((2,), -2.0)}, _ctx(2))
    assert np.allclose(np.asarray(w_eff), -2.0)


def test_vmap_over_weights_smoke():
    s = cont_delay_synapse()
    ctx = _ctx(4)
    out = jax.vmap(lambda w: s.update({'weight': w}, ctx)[1])(jnp.ones((3, 4)))
    assert out.shape == (3, 4)


def test_grad_through_weight():
    s = cont_delay_synapse()
    def loss(w):
        _, w_eff = s.update({'weight': w}, _ctx(3))
        return jnp.sum(w_eff)
    g = jax.grad(loss)(jnp.ones((3,)))
    assert np.allclose(np.asarray(g), 1.0)
