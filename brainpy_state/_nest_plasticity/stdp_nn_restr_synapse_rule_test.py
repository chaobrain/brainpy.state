# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``stdp_nn_restr_synapse``.

The *restricted* symmetric nearest-neighbour scheme (NEST fig. 7C) adds two
per-edge eligibility gates on top of the symmetric kernel: a post spike
facilitates with the nearest preceding pre **only if a pre has occurred since the
previous post**, and a pre spike depresses with the nearest preceding post **only
if a post has occurred since the previous pre** — so each spike participates in at
most one potentiation and one depression pair (``stdp_nn_restr_synapse.h:54-60``).
These tests lock the spec contract, the gate logic (direct kernel), the
single-pair closed forms through the substrate, and — the crux — the
multi-spike trains where the gates make restr diverge from ``stdp_nn_symm``
(only the first post after a pre facilitates; only the first pre after a post
depresses). Live-NEST equivalence is in
``_validation/stdp_nn_restr_synapse_parity_test.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import stdp_nn_restr_synapse, stdp_nn_symm_synapse
from brainpy_state._nest_network._event_plastic import EventPlasticProj, KernelContext


def _ctx(pre_spike, post_spike, pre_trace, post_trace, E=1, t=10.0, dt=1.0):
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    return KernelContext(g(pre_spike), g(post_spike), g(pre_trace), g(post_trace),
                         jnp.asarray(t), jnp.asarray(dt), jax.random.key(0))


def _host_facilitate(w, kplus, Wmax=100., lam=0.1, mu_plus=1.):
    nw = w / Wmax + lam * (1.0 - w / Wmax) ** mu_plus * kplus
    return nw * Wmax if nw < 1.0 else Wmax


def _host_depress(w, kminus, Wmax=100., alpha=1., lam=0.1, mu_minus=1.):
    nw = w / Wmax - alpha * lam * (w / Wmax) ** mu_minus * kminus
    return nw * Wmax if nw > 0.0 else 0.0


def _drive(rule, pre_steps, post_steps, n_steps, dt=1.0):
    """Drive a single 0->0 plastic edge through the substrate; return stored weight."""
    rule.delay = None
    box = {'pre': jnp.zeros(1), 'post': jnp.zeros(1)}
    proj = EventPlasticProj(
        pre_spike=lambda: box['pre'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=None, post_local_idx=jnp.arange(1), n_post_pop=1,
        post_spike=lambda: box['post'],
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    pre_set, post_set = set(pre_steps), set(post_steps)
    for i in range(n_steps):
        box['pre'] = jnp.array([1.0 if i in pre_set else 0.0])
        box['post'] = jnp.array([1.0 if i in post_set else 0.0])
        with brainstate.environ.context(t=(i + 1) * dt * u.ms, dt=dt * u.ms, i=i):
            proj.update()
    return float(proj.weight.value[0])


def _state(w=10.0, pre_avail=0.0, post_avail=0.0):
    return {'weight': jnp.array([w]), 'pre_avail': jnp.array([pre_avail]),
            'post_avail': jnp.array([post_avail])}


# -- spec contract: nearest mode + two per-edge eligibility flags -----------
def test_spec_attributes_nearest_with_eligibility_state():
    s = stdp_nn_restr_synapse(tau_plus=18.0 * u.ms, tau_minus=22.0 * u.ms)
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.pre_trace_mode == 'nearest'
    assert s.post_trace_mode == 'nearest'
    assert s.edge_state_init() == {'pre_avail': 0.0, 'post_avail': 0.0}
    assert float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms)) == 18.0
    assert float(u.Quantity(s.post_trace_tau).to_decimal(u.ms)) == 22.0
    assert u.get_unit(s.weight) == u.pA


def test_validation_weight_wmax_sign_and_taus():
    with pytest.raises(ValueError):
        stdp_nn_restr_synapse(weight=-1.0, Wmax=100.0)
    with pytest.raises(ValueError):
        stdp_nn_restr_synapse(tau_plus=-1.0 * u.ms)
    with pytest.raises(ValueError):
        stdp_nn_restr_synapse(tau_minus=-1.0 * u.ms)


# -- gate logic (direct kernel): facilitation needs pre_avail, depression post_avail
def test_kernel_facilitation_gated_by_pre_avail():
    s = stdp_nn_restr_synapse(weight=10.0, lambda_=0.1)
    # post spike with an available pre -> facilitate
    st, _ = s.update(_state(pre_avail=1.0), _ctx(0.0, 1.0, 0.5, 0.0))
    assert np.allclose(np.asarray(st['weight']), [_host_facilitate(10.0, 0.5)])
    # post spike with NO available pre -> gated off (unchanged)
    st, _ = s.update(_state(pre_avail=0.0), _ctx(0.0, 1.0, 0.5, 0.0))
    assert np.allclose(np.asarray(st['weight']), [10.0])


def test_kernel_depression_gated_by_post_avail():
    s = stdp_nn_restr_synapse(weight=10.0, lambda_=0.1)
    st, _ = s.update(_state(post_avail=1.0), _ctx(1.0, 0.0, 0.0, 0.5))
    assert np.allclose(np.asarray(st['weight']), [_host_depress(10.0, 0.5)])
    st, _ = s.update(_state(post_avail=0.0), _ctx(1.0, 0.0, 0.0, 0.5))
    assert np.allclose(np.asarray(st['weight']), [10.0])


def test_kernel_flag_transitions():
    s = stdp_nn_restr_synapse(weight=10.0)
    flags = lambda st: (float(st['pre_avail'][0]), float(st['post_avail'][0]))
    # pre-only: own side available, opposite consumed
    st, _ = s.update(_state(pre_avail=0.0, post_avail=1.0), _ctx(1.0, 0.0, 0.0, 0.5))
    assert flags(st) == (1.0, 0.0)
    # post-only: symmetric
    st, _ = s.update(_state(pre_avail=1.0, post_avail=0.0), _ctx(0.0, 1.0, 0.5, 0.0))
    assert flags(st) == (0.0, 1.0)
    # simultaneous: both sides become available (each spike sets its own)
    st, _ = s.update(_state(pre_avail=0.0, post_avail=0.0), _ctx(1.0, 1.0, 1.0, 1.0))
    assert flags(st) == (1.0, 1.0)
    # neither: unchanged
    st, _ = s.update(_state(pre_avail=1.0, post_avail=1.0), _ctx(0.0, 0.0, 0.3, 0.3))
    assert flags(st) == (1.0, 1.0)


# -- single alternating pair through the substrate (== symm for one pair) ---
def test_single_pair_ltp_through_substrate():
    tau = 20.0
    w = _drive(stdp_nn_restr_synapse(weight=10.0, lambda_=0.1, tau_plus=tau * u.ms,
                                     tau_minus=tau * u.ms), [0], [3], 5)
    assert np.isclose(w, _host_facilitate(10.0, np.exp(-3.0 / tau)), atol=1e-9)


def test_single_pair_ltd_through_substrate():
    tau = 20.0
    w = _drive(stdp_nn_restr_synapse(weight=10.0, lambda_=0.1, tau_plus=tau * u.ms,
                                     tau_minus=tau * u.ms), [3], [0], 5)
    assert np.isclose(w, _host_depress(10.0, np.exp(-3.0 / tau)), atol=1e-9)


# -- THE restriction: two posts after one pre -> only the FIRST facilitates --
def test_two_posts_one_pre_only_first_facilitates_vs_symm():
    tau = 20.0
    kw = dict(weight=10.0, lambda_=0.1, tau_plus=tau * u.ms, tau_minus=tau * u.ms)
    w_restr = _drive(stdp_nn_restr_synapse(**kw), [0], [2, 4], 6)
    w_symm = _drive(stdp_nn_symm_synapse(**kw), [0], [2, 4], 6)
    # restr: only post@2 pairs with pre@0 (pre_avail consumed); post@4 gated off
    assert np.isclose(w_restr, _host_facilitate(10.0, np.exp(-2.0 / tau)), atol=1e-9)
    # symm: BOTH posts facilitate with pre@0 (nearest), so it climbs higher
    w_symm_expected = _host_facilitate(_host_facilitate(10.0, np.exp(-2.0 / tau)),
                                       np.exp(-4.0 / tau))
    assert np.isclose(w_symm, w_symm_expected, atol=1e-9)
    assert w_symm > w_restr + 1e-6


# -- and two pres after one post -> only the FIRST depresses -----------------
def test_two_pres_one_post_only_first_depresses_vs_symm():
    tau = 20.0
    kw = dict(weight=10.0, lambda_=0.1, tau_plus=tau * u.ms, tau_minus=tau * u.ms)
    w_restr = _drive(stdp_nn_restr_synapse(**kw), [2, 4], [0], 6)
    w_symm = _drive(stdp_nn_symm_synapse(**kw), [2, 4], [0], 6)
    assert np.isclose(w_restr, _host_depress(10.0, np.exp(-2.0 / tau)), atol=1e-9)
    w_symm_expected = _host_depress(_host_depress(10.0, np.exp(-2.0 / tau)),
                                    np.exp(-4.0 / tau))
    assert np.isclose(w_symm, w_symm_expected, atol=1e-9)
    assert w_symm < w_restr - 1e-6                      # symm depresses twice -> lower


# -- jit / vmap / grad smoke ----------------------------------------------
def test_vmap_grad_smoke():
    s = stdp_nn_restr_synapse(weight=10.0, lambda_=0.1)

    def run(w):
        st, _ = s.update({'weight': w, 'pre_avail': jnp.array([1.0]),
                          'post_avail': jnp.array([0.0])}, _ctx(0.0, 1.0, 0.5, 0.0))
        return jnp.sum(st['weight'])

    g = jax.grad(run)(jnp.array([10.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    out = jax.vmap(run)(jnp.array([[10.0], [20.0], [30.0]]))
    assert out.shape == (3,)
