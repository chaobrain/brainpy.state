# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``vogels_sprekeler_synapse``.

Kernel-level checks of the symmetric inhibitory-plasticity rule: facilitation on
both sides (post uses ``K+``, pre uses ``K-``) via the sign-aware
``copysign(min(|w|+eta*k, |Wmax|), Wmax)``, a constant per-pre-spike depression
``copysign(max(|w|-alpha*eta, 0), Wmax)``, magnitude saturation at ``±|Wmax|``,
sign preservation, the simultaneous-spike exclusion, and per-edge freezing.
Live-NEST equivalence is covered by the parity test.
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

from brainpy_state import vogels_sprekeler_synapse
from brainpy_state._nest_network._event_plastic import KernelContext


def _ctx(pre_spike, post_spike, pre_trace, post_trace, E=1, t=10.0, dt=1.0):
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    return KernelContext(g(pre_spike), g(post_spike), g(pre_trace), g(post_trace),
                         jnp.asarray(t), jnp.asarray(dt), jax.random.key(0))


def _host_fac(w, k, eta=0.001, Wmax=1.0):
    return math.copysign(min(abs(w) + eta * k, abs(Wmax)), Wmax)


def _host_dep(w, eta=0.001, alpha=0.12, Wmax=1.0):
    return math.copysign(max(abs(w) - alpha * eta, 0.0), Wmax)


# -- spec contract ---------------------------------------------------------
def test_spec_attributes_and_defaults():
    s = vogels_sprekeler_synapse()
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.edge_state_init() == {}
    assert s.eta == 0.001 and s.alpha == 0.12 and s.Wmax == 1.0
    tp = float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms))
    tm = float(u.Quantity(s.post_trace_tau).to_decimal(u.ms))
    assert tp == 20.0 and tm == 20.0                       # symmetric single tau
    assert u.get_unit(s.weight) == u.pA


def test_validation():
    with pytest.raises(ValueError):
        vogels_sprekeler_synapse(tau=-1.0 * u.ms)
    with pytest.raises(ValueError):
        vogels_sprekeler_synapse(weight=0.5, Wmax=-1.0)    # opposite sign


# -- symmetric facilitation: post uses K+, pre uses K- ---------------------
def test_facilitation_on_post_uses_pre_trace():
    s = vogels_sprekeler_synapse(weight=0.5, eta=0.01)
    st, w_eff = s.update({'weight': jnp.array([0.5])}, _ctx(0.0, 1.0, 0.4, 0.0))
    expect = _host_fac(0.5, 0.4, eta=0.01)
    assert np.allclose(np.asarray(st['weight']), [expect])
    assert np.allclose(np.asarray(w_eff), [expect])
    assert expect > 0.5


def test_pre_spike_facilitates_then_constant_depresses():
    s = vogels_sprekeler_synapse(weight=0.5, eta=0.01, alpha=0.12)
    # pre with post trace 0.4: facilitate(K-) then constant depress
    st, _ = s.update({'weight': jnp.array([0.5])}, _ctx(1.0, 0.0, 0.0, 0.4))
    expect = _host_dep(_host_fac(0.5, 0.4, eta=0.01), eta=0.01, alpha=0.12)
    assert np.allclose(np.asarray(st['weight']), [expect])


# -- constant depression on every pre spike (homeostasis) ------------------
def test_isolated_pre_constant_depression():
    s = vogels_sprekeler_synapse(weight=0.5, eta=0.01, alpha=0.12)
    st, _ = s.update({'weight': jnp.array([0.5])}, _ctx(1.0, 0.0, 0.0, 0.0))
    assert np.allclose(np.asarray(st['weight']), [_host_dep(0.5, eta=0.01)])  # < 0.5
    assert float(np.asarray(st['weight'])[0]) < 0.5


def test_no_spike_no_change():
    s = vogels_sprekeler_synapse(weight=0.5)
    st, _ = s.update({'weight': jnp.array([0.5])}, _ctx(0.0, 0.0, 0.7, 0.7))
    assert np.allclose(np.asarray(st['weight']), [0.5])


# -- saturation at +|Wmax| and floor at 0 ----------------------------------
def test_saturates_at_wmax():
    s = vogels_sprekeler_synapse(weight=0.99, eta=1.0, Wmax=1.0)
    st, _ = s.update({'weight': jnp.array([0.99])}, _ctx(0.0, 1.0, 10.0, 0.0))
    assert np.allclose(np.asarray(st['weight']), [1.0])    # saturated at Wmax


def test_depress_floor_at_zero():
    s = vogels_sprekeler_synapse(weight=0.001, eta=1.0, alpha=1.0, Wmax=1.0)
    st, _ = s.update({'weight': jnp.array([0.001])}, _ctx(1.0, 0.0, 0.0, 0.0))
    assert np.allclose(np.asarray(st['weight']), [0.0])    # |w| floored at 0


# -- sign preservation: negative Wmax keeps weights negative ---------------
def test_sign_preserved_for_inhibitory():
    s = vogels_sprekeler_synapse(weight=-0.5, eta=0.01, Wmax=-1.0)
    st, _ = s.update({'weight': jnp.array([-0.5])}, _ctx(0.0, 1.0, 0.4, 0.0))
    w = float(np.asarray(st['weight'])[0])
    assert w < 0.0                                         # stays negative
    assert np.allclose([w], [_host_fac(-0.5, 0.4, eta=0.01, Wmax=-1.0)])


# -- simultaneous pre&post excludes the current step's own spike -----------
def test_simultaneous_excludes_current_step_spike():
    s = vogels_sprekeler_synapse(weight=0.5, eta=0.01, alpha=0.0)
    # alpha=0 (no constant depression) + symmetric exclusion -> isolated pair no-op
    st, _ = s.update({'weight': jnp.array([0.5])}, _ctx(1.0, 1.0, 1.0, 1.0))
    assert np.allclose(np.asarray(st['weight']), [0.5])


def test_frozen_non_firing_edges():
    s = vogels_sprekeler_synapse(weight=0.5, eta=0.01)
    ctx = _ctx([1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], E=2)
    st, _ = s.update({'weight': jnp.array([0.5, 0.5])}, ctx)
    assert np.allclose(np.asarray(st['weight'])[1], 0.5)        # frozen
    assert float(np.asarray(st['weight'])[0]) < 0.5            # constant depression


def test_vmap_grad_smoke():
    s = vogels_sprekeler_synapse(weight=0.5, eta=0.01)

    def run(w):
        st, _ = s.update({'weight': w}, _ctx(0.0, 1.0, 0.5, 0.0))
        return jnp.sum(st['weight'])

    g = jax.grad(run)(jnp.array([0.5]))
    assert np.all(np.isfinite(np.asarray(g)))
    out = jax.vmap(run)(jnp.array([[0.3], [0.5], [0.7]]))
    assert out.shape == (3,)
