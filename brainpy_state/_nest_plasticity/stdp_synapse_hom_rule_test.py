# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``stdp_synapse_hom``.

``stdp_synapse_hom`` is a thin reuse of ``stdp_synapse`` (in NEST the plasticity
parameters are *common* / homogeneous, but the rule math is identical). These
checks pin the reuse: the subclass relationship, the NEST defaults, the spec
contract, and bit-for-bit kernel agreement with ``stdp_synapse`` on the same
inputs. The pair-STDP math itself is exercised by ``stdp_synapse_rule_test`` and
validated against live NEST by the parity test.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import stdp_synapse, stdp_synapse_hom
from brainpy_state._nest_network._event_plastic import KernelContext


def _ctx(pre_spike, post_spike, pre_trace, post_trace, E=1, t=10.0, dt=1.0):
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    return KernelContext(g(pre_spike), g(post_spike), g(pre_trace), g(post_trace),
                         jnp.asarray(t), jnp.asarray(dt), jax.random.key(0))


# -- thin-reuse contract ---------------------------------------------------
def test_is_stdp_synapse_subclass():
    assert issubclass(stdp_synapse_hom, stdp_synapse)
    assert isinstance(stdp_synapse_hom(), stdp_synapse)


def test_nest_defaults_match():
    s = stdp_synapse_hom()
    assert s.lambda_ == 0.01
    assert s.alpha == 1.0
    assert s.mu_plus == 1.0 and s.mu_minus == 1.0
    assert s.Wmax == 100.0
    assert float(u.Quantity(s.tau_plus).to_decimal(u.ms)) == 20.0
    assert float(u.Quantity(s.tau_minus).to_decimal(u.ms)) == 20.0


def test_spec_attributes_and_traces():
    s = stdp_synapse_hom(tau_plus=18.0 * u.ms, tau_minus=22.0 * u.ms)
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.edge_state_init() == {}
    assert float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms)) == 18.0
    assert float(u.Quantity(s.post_trace_tau).to_decimal(u.ms)) == 22.0
    assert u.get_unit(s.weight) == u.pA


def test_validation_inherited():
    with pytest.raises(ValueError):
        stdp_synapse_hom(weight=-1.0, Wmax=100.0)         # sign check inherited
    with pytest.raises(ValueError):
        stdp_synapse_hom(tau_plus=-1.0 * u.ms)


# -- kernel is bit-for-bit identical to stdp_synapse -----------------------
def test_kernel_matches_stdp_synapse():
    kw = dict(weight=12.0, Wmax=80.0, lambda_=0.07, alpha=1.3,
              mu_plus=0.5, mu_minus=0.5, tau_plus=15.0 * u.ms, tau_minus=25.0 * u.ms)
    base = stdp_synapse(**kw)
    hom = stdp_synapse_hom(**kw)
    for ctx in (_ctx(0.0, 1.0, 0.6, 0.0),       # potentiation on post
                _ctx(1.0, 0.0, 0.0, 0.4),       # depression on pre
                _ctx(1.0, 1.0, 1.0, 2.0),       # simultaneous (exclusions)
                _ctx(0.0, 0.0, 0.7, 0.7)):      # no spike
        w0 = {'weight': jnp.array([12.0])}
        sb, wb = base.update(dict(w0), ctx)
        sh, wh = hom.update(dict(w0), ctx)
        assert np.allclose(np.asarray(sb['weight']), np.asarray(sh['weight']))
        assert np.allclose(np.asarray(wb), np.asarray(wh))
