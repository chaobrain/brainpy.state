# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``stdp_nn_pre_centered_synapse``.

The *presynaptic-centered* nearest-neighbour scheme (NEST fig. 7B) keeps a
genuinely **per-edge** ``Kplus`` trace that accumulates ``+1`` per pre spike,
decays at ``tau_plus``, and **resets to 0 on every post spike**
(``stdp_nn_pre_centered_synapse.h:69-74``). A post spike facilitates with the
accumulated ``Kplus`` (all pres since the previous post); a pre spike depresses
with the nearest preceding post (substrate ``'nearest'`` ``K-``). These tests lock
the spec contract, the in-kernel ``Kplus`` decay/accumulate/reset, the single-pair
closed forms through the substrate, and the two divergences: accumulation (vs
``stdp_nn_symm`` which takes only the nearest pre) and reset-on-post (vs all-to-all
``stdp_synapse`` which never forgets). Live-NEST equivalence is in
``_validation/stdp_nn_pre_centered_synapse_parity_test.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import (stdp_nn_pre_centered_synapse, stdp_nn_symm_synapse,
                           stdp_synapse)
from brainpy_state._nest_network.event_plastic import EventPlasticProj, KernelContext

TAU = 20.0


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


def _state(w=10.0, Kplus=0.0):
    return {'weight': jnp.array([w]), 'Kplus': jnp.array([Kplus])}


# -- spec contract: per-edge Kplus, nearest post trace, no substrate pre trace
def test_spec_attributes_per_edge_kplus_nearest_post():
    s = stdp_nn_pre_centered_synapse(tau_plus=18.0 * u.ms, tau_minus=22.0 * u.ms, Kplus=0.0)
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.post_trace_mode == 'nearest'
    assert s.pre_trace_tau is None              # Kplus is per-edge, not a substrate trace
    assert s.edge_state_init() == {'Kplus': 0.0}
    assert float(u.Quantity(s.post_trace_tau).to_decimal(u.ms)) == 22.0
    assert u.get_unit(s.weight) == u.pA


def test_validation_weight_wmax_sign_taus_and_kplus():
    with pytest.raises(ValueError):
        stdp_nn_pre_centered_synapse(weight=-1.0, Wmax=100.0)
    with pytest.raises(ValueError):
        stdp_nn_pre_centered_synapse(tau_plus=-1.0 * u.ms)
    with pytest.raises(ValueError):
        stdp_nn_pre_centered_synapse(tau_minus=-1.0 * u.ms)
    with pytest.raises(ValueError):
        stdp_nn_pre_centered_synapse(Kplus=-0.5)


# -- kernel: facilitation uses (decayed) Kplus then resets it to 0 ----------
def test_kernel_post_facilitates_with_decayed_kplus_then_resets():
    s = stdp_nn_pre_centered_synapse(weight=10.0, lambda_=0.1, tau_plus=TAU * u.ms)
    st, _ = s.update(_state(w=10.0, Kplus=2.0), _ctx(0.0, 1.0, 0.0, 0.0, dt=1.0))
    kd = 2.0 * np.exp(-1.0 / TAU)                # decayed one step before facilitation
    assert np.allclose(np.asarray(st['weight']), [_host_facilitate(10.0, kd)])
    assert np.allclose(np.asarray(st['Kplus']), [0.0])     # post erases the trace


# -- kernel: depression always uses nearest post; pre accumulates Kplus +1 --
def test_kernel_pre_depresses_and_accumulates_kplus():
    s = stdp_nn_pre_centered_synapse(weight=10.0, lambda_=0.1, tau_plus=TAU * u.ms)
    st, _ = s.update(_state(w=10.0, Kplus=1.0), _ctx(1.0, 0.0, 0.0, 0.5, dt=1.0))
    assert np.allclose(np.asarray(st['weight']), [_host_depress(10.0, 0.5)])
    # Kplus: decay one step then +1 for this pre
    assert np.allclose(np.asarray(st['Kplus']), [1.0 * np.exp(-1.0 / TAU) + 1.0])


def test_kernel_simultaneous_excludes_current_pre_from_facilitation():
    # pre&post same step: facilitation sees Kplus BEFORE this pre's +1 (second-latest),
    # then post resets, then pre re-accumulates to 1.
    s = stdp_nn_pre_centered_synapse(weight=10.0, lambda_=0.1, tau_plus=TAU * u.ms)
    st, _ = s.update(_state(w=10.0, Kplus=3.0), _ctx(1.0, 1.0, 0.0, 1.0, dt=1.0))
    kd = 3.0 * np.exp(-1.0 / TAU)
    w_after_fac = _host_facilitate(10.0, kd)
    # depression uses kminus = post_trace - post_spike = 1.0 - 1.0 = 0 -> no depression
    assert np.allclose(np.asarray(st['weight']), [w_after_fac])
    assert np.allclose(np.asarray(st['Kplus']), [1.0])     # reset by post, +1 by pre


# -- single pair through the substrate (LTP and LTD) -----------------------
def test_single_pair_ltp_through_substrate():
    w = _drive(stdp_nn_pre_centered_synapse(weight=10.0, lambda_=0.1, tau_plus=TAU * u.ms,
                                            tau_minus=TAU * u.ms), [0], [3], 5)
    assert np.isclose(w, _host_facilitate(10.0, np.exp(-3.0 / TAU)), atol=1e-9)


def test_single_pair_ltd_through_substrate():
    w = _drive(stdp_nn_pre_centered_synapse(weight=10.0, lambda_=0.1, tau_plus=TAU * u.ms,
                                            tau_minus=TAU * u.ms), [3], [0], 5)
    assert np.isclose(w, _host_depress(10.0, np.exp(-3.0 / TAU)), atol=1e-9)


# -- divergence 1: accumulation. Two pres then a post -> Kplus SUMS pres,
#    whereas symm pairs only the nearest pre ---------------------------------
def test_two_pres_one_post_accumulates_vs_symm():
    kw = dict(weight=10.0, lambda_=0.1, tau_plus=TAU * u.ms, tau_minus=TAU * u.ms)
    w_prec = _drive(stdp_nn_pre_centered_synapse(**kw), [0, 2], [4], 6)
    w_symm = _drive(stdp_nn_symm_synapse(**kw), [0, 2], [4], 6)
    kplus_sum = np.exp(-4.0 / TAU) + np.exp(-2.0 / TAU)     # both pres since last post
    assert np.isclose(w_prec, _host_facilitate(10.0, kplus_sum), atol=1e-9)
    assert np.isclose(w_symm, _host_facilitate(10.0, np.exp(-2.0 / TAU)), atol=1e-9)
    assert w_prec > w_symm + 1e-6


# -- divergence 2: reset-on-post. pre,post,pre,post -> the second post pairs
#    only with the pre AFTER the reset, whereas all-to-all keeps both pres ----
def test_reset_on_post_vs_all_to_all():
    kw = dict(weight=10.0, lambda_=0.1, tau_plus=TAU * u.ms, tau_minus=TAU * u.ms)
    w_prec = _drive(stdp_nn_pre_centered_synapse(**kw), [0, 4], [2, 6], 8)
    w_a2a = _drive(stdp_synapse(**kw), [0, 4], [2, 6], 8)
    # pre_centered: post@2 facil exp(-2/TAU) then Kplus reset; post@6 pairs only p@4
    e2 = np.exp(-2.0 / TAU)
    w1 = _host_facilitate(10.0, e2)             # post@2 with p@0
    w2 = _host_depress(w1, e2)                   # pre@4 with nearest post@2
    w3 = _host_facilitate(w2, e2)               # post@6 with p@4 ONLY (reset erased p@0)
    assert np.isclose(w_prec, w3, atol=1e-9)
    assert w_a2a > w_prec + 1e-6                 # all-to-all also pairs post@6 with p@0


# -- jit / vmap / grad smoke ----------------------------------------------
def test_vmap_grad_smoke():
    s = stdp_nn_pre_centered_synapse(weight=10.0, lambda_=0.1)

    def run(w):
        st, _ = s.update({'weight': w, 'Kplus': jnp.array([2.0])},
                         _ctx(0.0, 1.0, 0.0, 0.0))
        return jnp.sum(st['weight'])

    g = jax.grad(run)(jnp.array([10.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    out = jax.vmap(run)(jnp.array([[10.0], [20.0], [30.0]]))
    assert out.shape == (3,)
