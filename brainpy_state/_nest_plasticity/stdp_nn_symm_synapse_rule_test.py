# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``stdp_nn_symm_synapse``.

The symmetric nearest-neighbour kernel is identical to ``stdp_synapse`` (potentiation
on the post spike, depression on the pre spike, ``[0, Wmax]`` clamp, simultaneous-spike
exclusion); what differs is the substrate **trace mode** — ``pre_trace_mode`` /
``post_trace_mode`` are ``'nearest'`` so each side's trace resets to 1 on a spike. These
tests lock the kernel form, the nearest single-pair closed forms (driven through the
substrate), and the train where nearest pairing diverges from all-to-all. Live-NEST
equivalence is in ``_validation/stdp_nn_symm_parity_test.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import stdp_nn_symm_synapse, stdp_synapse
from brainpy_state._nest_network.event_plastic import EventPlasticProj, KernelContext


def _ctx(pre_spike, post_spike, pre_trace, post_trace, E=1, t=10.0, dt=1.0):
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    return KernelContext(g(pre_spike), g(post_spike), g(pre_trace), g(post_trace),
                         jnp.asarray(t), jnp.asarray(dt), jax.random.key(0))


def _host_facilitate(w, kplus, Wmax=100., lam=0.01, mu_plus=1.):
    nw = w / Wmax + lam * (1.0 - w / Wmax) ** mu_plus * kplus
    return nw * Wmax if nw < 1.0 else Wmax


def _host_depress(w, kminus, Wmax=100., alpha=1., lam=0.01, mu_minus=1.):
    nw = w / Wmax - alpha * lam * (w / Wmax) ** mu_minus * kminus
    return nw * Wmax if nw > 0.0 else 0.0


def _drive(rule, pre_steps, post_steps, n_steps, dt=1.0):
    """Drive a single 0->0 plastic edge through the substrate; return stored weight.

    ``rule.delay`` is forced to ``None`` so the pre trace is built from the raw pre
    spikes (no axonal shift) — the NEST-free analogue of the parity harness.
    """
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


# -- spec contract: nearest mode on BOTH sides -----------------------------
def test_spec_attributes_nearest_both_sides():
    s = stdp_nn_symm_synapse(tau_plus=18.0 * u.ms, tau_minus=22.0 * u.ms)
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.edge_state_init() == {}                       # pure trace-mode, no per-edge state
    assert s.pre_trace_mode == 'nearest'
    assert s.post_trace_mode == 'nearest'
    assert float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms)) == 18.0
    assert float(u.Quantity(s.post_trace_tau).to_decimal(u.ms)) == 22.0
    assert u.get_unit(s.weight) == u.pA


def test_validation_weight_wmax_sign_and_taus():
    with pytest.raises(ValueError):
        stdp_nn_symm_synapse(weight=-1.0, Wmax=100.0)       # opposite sign
    with pytest.raises(ValueError):
        stdp_nn_symm_synapse(tau_plus=-1.0 * u.ms)
    with pytest.raises(ValueError):
        stdp_nn_symm_synapse(tau_minus=-1.0 * u.ms)


# -- kernel form (== stdp_synapse): potentiation/depression/exclusion/clamp -
def test_kernel_potentiation_on_post_and_depression_on_pre():
    s = stdp_nn_symm_synapse(weight=10.0, Wmax=100.0, lambda_=0.1)
    st, _ = s.update({'weight': jnp.array([10.0])}, _ctx(0.0, 1.0, 0.5, 0.0))
    assert np.allclose(np.asarray(st['weight']), [_host_facilitate(10.0, 0.5, lam=0.1)])
    st, _ = s.update({'weight': jnp.array([10.0])}, _ctx(1.0, 0.0, 0.0, 0.5))
    assert np.allclose(np.asarray(st['weight']), [_host_depress(10.0, 0.5, lam=0.1)])


def test_kernel_simultaneous_excludes_current_spike():
    s = stdp_nn_symm_synapse(weight=10.0, lambda_=0.1)
    # both traces carry only this step's +1 -> kplus=kminus=0 -> no net change
    st, _ = s.update({'weight': jnp.array([10.0])}, _ctx(1.0, 1.0, 1.0, 1.0))
    assert np.allclose(np.asarray(st['weight']), [10.0])


def test_kernel_clamps():
    s = stdp_nn_symm_synapse(weight=99.0, Wmax=100.0, lambda_=5.0)
    st, _ = s.update({'weight': jnp.array([99.0])}, _ctx(0.0, 1.0, 10.0, 0.0))
    assert np.allclose(np.asarray(st['weight']), [100.0])
    s = stdp_nn_symm_synapse(weight=1.0, Wmax=100.0, lambda_=5.0)
    st, _ = s.update({'weight': jnp.array([1.0])}, _ctx(1.0, 0.0, 0.0, 10.0))
    assert np.allclose(np.asarray(st['weight']), [0.0])


# -- nearest single pair through the substrate (LTP and LTD) ---------------
def test_nearest_single_pair_ltp_through_substrate():
    # one pre (step 0) then one post (step 3): facilitate with exp(-3 dt/tau).
    tau = 20.0
    w = _drive(stdp_nn_symm_synapse(weight=10.0, lambda_=0.1, tau_plus=tau * u.ms,
                                    tau_minus=tau * u.ms), [0], [3], 5)
    assert np.isclose(w, _host_facilitate(10.0, np.exp(-3.0 / tau), lam=0.1), atol=1e-9)


def test_nearest_single_pair_ltd_through_substrate():
    # one post (step 0) then one pre (step 3): depress with exp(-3 dt/tau).
    tau = 20.0
    w = _drive(stdp_nn_symm_synapse(weight=10.0, lambda_=0.1, tau_plus=tau * u.ms,
                                    tau_minus=tau * u.ms), [3], [0], 5)
    assert np.isclose(w, _host_depress(10.0, np.exp(-3.0 / tau), lam=0.1), atol=1e-9)


# -- THE divergence: 3 pre then 1 post; nearest != all-to-all --------------
def test_three_pre_then_post_diverges_from_all_to_all():
    tau = 20.0
    pre, post, n = [0, 1, 2], [5], 6
    w_nn = _drive(stdp_nn_symm_synapse(weight=10.0, lambda_=0.1, tau_plus=tau * u.ms,
                                       tau_minus=tau * u.ms), pre, post, n)
    w_a2a = _drive(stdp_synapse(weight=10.0, lambda_=0.1, tau_plus=tau * u.ms,
                                tau_minus=tau * u.ms), pre, post, n)
    # nearest pairs only the LAST pre (step 2 -> step 5 = 3 steps):
    kplus_nn = np.exp(-3.0 / tau)
    assert np.isclose(w_nn, _host_facilitate(10.0, kplus_nn, lam=0.1), atol=1e-9)
    # all-to-all sums all three pres -> strictly larger facilitation, so the two differ:
    kplus_a2a = np.exp(-3.0 / tau) + np.exp(-4.0 / tau) + np.exp(-5.0 / tau)
    assert np.isclose(w_a2a, _host_facilitate(10.0, kplus_a2a, lam=0.1), atol=1e-9)
    assert w_a2a > w_nn + 1e-6                              # genuinely divergent


# -- jit / vmap / grad smoke ----------------------------------------------
def test_vmap_grad_smoke():
    s = stdp_nn_symm_synapse(weight=10.0, lambda_=0.1)

    def run(w):
        st, _ = s.update({'weight': w}, _ctx(0.0, 1.0, 0.5, 0.0))
        return jnp.sum(st['weight'])

    g = jax.grad(run)(jnp.array([10.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    out = jax.vmap(run)(jnp.array([[10.0], [20.0], [30.0]]))
    assert out.shape == (3,)
