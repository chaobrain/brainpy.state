# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for the rebuilt JAX-native ``volume_transmitter`` node (NEST-free).

The node maintains the dopamine concentration ``n`` as a broadcast State and
advances it each step by NEST's ``update_dopamine_`` recursion
``n <- n*exp(-dt/tau_n) + count/tau_n`` (``volume_transmitter.{h,cpp}`` +
``stdp_dopamine_synapse.h:419-425``), where ``count`` is the number of bound
dopaminergic spikes delivered this step. These tests lock the recursion,
multi-source summation, ``dt`` sweep, the ``deliver_interval`` no-op, and
jit/vmap/grad without a live network.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest.volume_transmitter import volume_transmitter  # noqa: E402

TAU_N = 200.0


def _host_trace(spikes, *, dt=1.0, tau_n=TAU_N):
    """Reference ``n(t)`` from the NEST update_dopamine_ recursion."""
    n, out = 0.0, []
    decay = math.exp(-dt / tau_n)
    for c in spikes:
        n = n * decay + float(c) / tau_n
        out.append(n)
    return np.asarray(out)


def _build(spikes_holder, local_idx=(0,), *, tau_n=TAU_N * u.ms, deliver_interval=1, size=1):
    vt = volume_transmitter(size, tau_n=tau_n, deliver_interval=deliver_interval)
    vt.bind_dopa(lambda: spikes_holder['s'], jnp.asarray(local_idx))
    brainstate.nn.init_all_states(vt)
    return vt


def _drive(vt, holder, spikes, *, dt=1.0):
    """Step the VT over a 0/1 spike train, returning the captured ``n`` trace."""
    out = []
    for k, sp in enumerate(spikes):
        holder['s'] = jnp.asarray([float(sp)])
        with brainstate.environ.context(t=k * dt * u.ms, i=k):
            vt.update()
        out.append(float(np.asarray(vt.n.value)[0]))
    return np.asarray(out)


# --------------------------------------------------------------------------
# spec attributes + validation
# --------------------------------------------------------------------------
def test_spec_attributes_and_defaults():
    brainstate.environ.set(dt=1.0 * u.ms)
    vt = volume_transmitter()
    assert float(u.Quantity(vt.tau_n).to_decimal(u.ms)) == TAU_N
    assert vt.deliver_interval == 1
    brainstate.nn.init_all_states(vt)
    assert vt.n.value.shape == (1,)
    assert float(np.asarray(vt.n.value)[0]) == 0.0


def test_tau_n_must_be_positive():
    with pytest.raises(ValueError, match='tau_n'):
        volume_transmitter(tau_n=0.0 * u.ms)
    with pytest.raises(ValueError, match='tau_n'):
        volume_transmitter(tau_n=-5.0 * u.ms)


def test_deliver_interval_must_be_geq_one():
    with pytest.raises(ValueError, match='deliver_interval'):
        volume_transmitter(deliver_interval=0)


# --------------------------------------------------------------------------
# the update_dopamine_ recursion
# --------------------------------------------------------------------------
def test_n_follows_update_dopamine_recursion():
    brainstate.environ.set(dt=1.0 * u.ms)
    holder = {'s': jnp.zeros(1)}
    vt = _build(holder)
    spikes = [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
    got = _drive(vt, holder, spikes, dt=1.0)
    assert np.allclose(got, _host_trace(spikes, dt=1.0))


def test_no_dopa_is_pure_decay():
    brainstate.environ.set(dt=1.0 * u.ms)
    holder = {'s': jnp.zeros(1)}
    vt = _build(holder)
    # seed n with one spike, then deliver nothing -> pure exponential decay
    got = _drive(vt, holder, [1, 0, 0, 0, 0], dt=1.0)
    decay = math.exp(-1.0 / TAU_N)
    assert np.allclose(got, [1 / TAU_N * decay ** k for k in range(5)])


def test_unbound_vt_decays_only():
    # a VT with no bound dopa source just decays its (already-seeded) n.
    brainstate.environ.set(dt=1.0 * u.ms)
    vt = volume_transmitter(1, tau_n=TAU_N * u.ms)
    brainstate.nn.init_all_states(vt)
    vt.n.value = jnp.asarray([1.0])
    with brainstate.environ.context(t=0.0 * u.ms, i=0):
        vt.update()
    assert np.allclose(np.asarray(vt.n.value), math.exp(-1.0 / TAU_N))


# --------------------------------------------------------------------------
# multi-source summation + sliced local index
# --------------------------------------------------------------------------
def test_multi_source_count_sums():
    brainstate.environ.set(dt=1.0 * u.ms)
    a, b = {'s': jnp.zeros(1)}, {'s': jnp.zeros(1)}
    vt = volume_transmitter(1, tau_n=TAU_N * u.ms)
    vt.bind_dopa(lambda: a['s'], jnp.array([0]))
    vt.bind_dopa(lambda: b['s'], jnp.array([0]))
    brainstate.nn.init_all_states(vt)
    # both fire -> count = 2 in the first step
    a['s'], b['s'] = jnp.asarray([1.0]), jnp.asarray([1.0])
    with brainstate.environ.context(t=0.0 * u.ms, i=0):
        vt.update()
    assert np.allclose(np.asarray(vt.n.value), 2.0 / TAU_N)


def test_local_idx_selects_subset():
    # the reader exposes a 4-neuron pool; only indices {1, 3} are dopaminergic.
    brainstate.environ.set(dt=1.0 * u.ms)
    holder = {'s': jnp.zeros(4)}
    vt = volume_transmitter(1, tau_n=TAU_N * u.ms)
    vt.bind_dopa(lambda: holder['s'], jnp.array([1, 3]))
    brainstate.nn.init_all_states(vt)
    holder['s'] = jnp.asarray([1.0, 1.0, 1.0, 1.0])   # all fire, only 2 are bound
    with brainstate.environ.context(t=0.0 * u.ms, i=0):
        vt.update()
    assert np.allclose(np.asarray(vt.n.value), 2.0 / TAU_N)


# --------------------------------------------------------------------------
# dt sweep + deliver_interval no-op
# --------------------------------------------------------------------------
@pytest.mark.parametrize('dt', [0.1, 0.5, 1.0])
def test_dt_sweep_recursion(dt):
    brainstate.environ.set(dt=dt * u.ms)
    holder = {'s': jnp.zeros(1)}
    vt = _build(holder)
    spikes = [1, 0, 1, 0, 0, 1, 0, 1]
    got = _drive(vt, holder, spikes, dt=dt)
    assert np.allclose(got, _host_trace(spikes, dt=dt))


def test_deliver_interval_accepted_and_ignored():
    # the online scheme integrates every step; deliver_interval changes nothing.
    brainstate.environ.set(dt=1.0 * u.ms)
    spikes = [1, 0, 1, 1, 0, 1]
    h1, h5 = {'s': jnp.zeros(1)}, {'s': jnp.zeros(1)}
    vt1 = _build(h1, deliver_interval=1)
    vt5 = _build(h5, deliver_interval=5)
    g1 = _drive(vt1, h1, spikes)
    g5 = _drive(vt5, h5, spikes)
    assert np.allclose(g1, g5)
    assert np.allclose(g1, _host_trace(spikes))


# --------------------------------------------------------------------------
# size > 1 (independent broadcast width) + reset
# --------------------------------------------------------------------------
def test_size_sets_n_shape():
    brainstate.environ.set(dt=1.0 * u.ms)
    vt = volume_transmitter(3, tau_n=TAU_N * u.ms)
    brainstate.nn.init_all_states(vt)
    assert vt.n.value.shape == (3,)


def test_reset_state_zeros_n():
    brainstate.environ.set(dt=1.0 * u.ms)
    vt = volume_transmitter(1, tau_n=TAU_N * u.ms)
    brainstate.nn.init_all_states(vt)
    vt.n.value = jnp.asarray([0.5])
    vt.reset_state()
    assert np.allclose(np.asarray(vt.n.value), 0.0)


# --------------------------------------------------------------------------
# the jitted hot path (for_loop) + pure-kernel vmap / grad
# --------------------------------------------------------------------------
def test_n_matches_recursion_in_for_loop():
    import brainstate.transform as transform
    brainstate.environ.set(dt=1.0 * u.ms)
    vt = volume_transmitter(1, tau_n=TAU_N * u.ms)
    holder = brainstate.ShortTermState(jnp.zeros(1))
    vt.bind_dopa(lambda: holder.value, jnp.array([0]))
    brainstate.nn.init_all_states(vt)
    spikes = jnp.asarray([1., 0., 0., 1., 1., 0., 0., 1., 0., 1.])

    def step(i, sp):
        with brainstate.environ.context(t=i * 1.0 * u.ms, i=i):
            holder.value = jnp.asarray([sp])
            vt.update()
            return vt.n.value[0]

    got = transform.for_loop(step, jnp.arange(spikes.size), spikes)
    assert np.allclose(np.asarray(got), _host_trace(np.asarray(spikes)))


def test_advance_kernel_vmap_and_grad():
    adv = volume_transmitter._advance
    out = jax.vmap(lambda n, c: adv(n, c, 1.0, TAU_N))(jnp.arange(5.0), jnp.ones(5))
    assert out.shape == (5,)
    # n=2, count=1: n*exp(-1/200) + 1/200
    assert np.allclose(float(adv(jnp.asarray(2.0), 1.0, 1.0, TAU_N)),
                       2.0 * math.exp(-1.0 / TAU_N) + 1.0 / TAU_N)
    g = jax.grad(lambda n: jnp.sum(adv(n, 1.0, 1.0, TAU_N)))(jnp.asarray([2.0]))
    assert np.all(np.isfinite(np.asarray(g)))
