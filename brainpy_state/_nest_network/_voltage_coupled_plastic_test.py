# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for ``VoltageCoupledPlasticProj`` — primitive #2 (NEST-free).

The voltage-coupled projection is ``EventPlasticProj`` plus a post-neuron
analog-state reader: a rule declares ``post_state_reads`` (a tuple of post-neuron
State attribute names) and the substrate gathers those per-post-neuron State
columns into ``ctx.post_states`` (a ``{name: (E,)}`` dict, CSR sorted-by-pre edge
order, unit-stripped to mantissas) each step. These tests lock the gather
correctness (including sliced/multi-segment post), the construction guards, the
primitive-#1-intact contract (``ctx.post_states is None`` on the base), and the
jit/vmap/grad safety of the new read path.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_network._event_plastic import (  # noqa: E402
    EventPlasticProj,
    VoltageCoupledPlasticProj,
    _StaticTestRule,
)


# --------------------------------------------------------------------------
# post stand-in: named analog State (mV) + add_delta_input sink. Plain object
# (not a brainstate Module) so init_all_states(proj) does not recurse into it,
# mirroring _Sink in _event_plastic_test.py.
# --------------------------------------------------------------------------
class _Val:
    def __init__(self, v):
        self.value = v


class _PostStub:
    """Post population stand-in exposing V / u_bar_plus / u_bar_minus as mV State."""

    def __init__(self, V, u_plus, u_minus):
        self.V = _Val(jnp.asarray(V, dtype=float) * u.mV)
        self.u_bar_plus = _Val(jnp.asarray(u_plus, dtype=float) * u.mV)
        self.u_bar_minus = _Val(jnp.asarray(u_minus, dtype=float) * u.mV)
        self.last = None

    def add_delta_input(self, key, val):
        self.last = val


class _ReaderProbe(_StaticTestRule):
    """Rule that declares post_state_reads and captures the gathered ctx.post_states."""
    post_state_reads = ('u_bar_minus', 'u_bar_plus', 'V')

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = None

    def update(self, state, ctx):
        self.seen = ctx.post_states
        return state, state['weight']


class _CtxProbe(_StaticTestRule):
    """Rule that captures ctx.post_states (used to assert the base is None)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = 'unset'

    def update(self, state, ctx):
        self.seen = ctx.post_states
        return state, state['weight']


# --------------------------------------------------------------------------
# Task 2 — gather correctness in CSR (sorted-by-pre) edge order
# --------------------------------------------------------------------------
def test_post_states_gathered_per_edge_in_csr_order():
    brainstate.environ.set(dt=0.1 * u.ms)
    # 3-neuron post; distinct per-neuron analog values.
    post = _PostStub(V=[-65., -55., -45.],
                     u_plus=[-70., -60., -50.],
                     u_minus=[-71., -61., -51.])
    rule = _ReaderProbe(weight=jnp.array([0., 0., 0.]) * u.pA)
    # edges (pre,post) = (0,2),(1,0),(0,1) -> stable sort by pre -> (0,2),(0,1),(1,0)
    proj = VoltageCoupledPlasticProj(
        pre_spike=lambda: jnp.array([1., 1.]), n_pre_pop=2, pre_local_idx=jnp.arange(2),
        post=post, post_local_idx=jnp.arange(3), n_post_pop=3,
        pre_idx=jnp.array([0, 1, 0]), post_idx=jnp.array([2, 0, 1]), rule=rule)
    brainstate.nn.init_all_states(proj)
    with brainstate.environ.context(t=0.1 * u.ms, i=1):
        proj.update()
    seen = rule.seen
    assert isinstance(seen, dict)
    assert set(seen) == {'u_bar_minus', 'u_bar_plus', 'V'}
    # CSR edge order: post neurons [2, 1, 0]
    assert np.allclose(np.asarray(seen['V']), [-45., -55., -65.])
    assert np.allclose(np.asarray(seen['u_bar_plus']), [-50., -60., -70.])
    assert np.allclose(np.asarray(seen['u_bar_minus']), [-51., -61., -71.])


def test_post_states_respects_sliced_post_population():
    # post_local_idx selects a sub-segment of a larger population; the gather must
    # index the full-population State at the population-local indices.
    brainstate.environ.set(dt=0.1 * u.ms)
    post = _PostStub(V=[0., 1., 2., 3., 4., 5.],
                     u_plus=[0., 10., 20., 30., 40., 50.],
                     u_minus=[0., -1., -2., -3., -4., -5.])
    rule = _ReaderProbe(weight=jnp.array([0., 0.]) * u.pA)
    # target only population neurons {5,3,1}; segment-local post idx 0->5, 1->3, 2->1
    proj = VoltageCoupledPlasticProj(
        pre_spike=lambda: jnp.array([1.]), n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=post, post_local_idx=jnp.array([5, 3, 1]), n_post_pop=6,
        pre_idx=jnp.array([0, 0]), post_idx=jnp.array([0, 2]), rule=rule)
    brainstate.nn.init_all_states(proj)
    with brainstate.environ.context(t=0.1 * u.ms, i=1):
        proj.update()
    # edges target segment-local post {0,2} -> population {5,1}
    assert np.allclose(np.asarray(rule.seen['V']), [5., 1.])
    assert np.allclose(np.asarray(rule.seen['u_bar_plus']), [50., 10.])
    assert np.allclose(np.asarray(rule.seen['u_bar_minus']), [-5., -1.])


def test_reads_are_unit_stripped_mantissas():
    # post State carries a mV unit; ctx.post_states are bare mantissas in mV.
    brainstate.environ.set(dt=0.1 * u.ms)
    post = _PostStub(V=[-42.5], u_plus=[-43.5], u_minus=[-44.5])
    rule = _ReaderProbe(weight=jnp.array([0.]) * u.pA)
    proj = VoltageCoupledPlasticProj(
        pre_spike=lambda: jnp.array([1.]), n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=post, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    with brainstate.environ.context(t=0.1 * u.ms, i=1):
        proj.update()
    assert not isinstance(rule.seen['V'], u.Quantity)
    assert np.allclose(np.asarray(rule.seen['V']), [-42.5])


# --------------------------------------------------------------------------
# Task 2 — construction guards
# --------------------------------------------------------------------------
def test_requires_post_state_reads():
    post = _PostStub(V=[0.], u_plus=[0.], u_minus=[0.])
    with pytest.raises(ValueError, match='post_state_reads'):
        VoltageCoupledPlasticProj(
            pre_spike=lambda: jnp.array([1.]), n_pre_pop=1, pre_local_idx=jnp.arange(1),
            post=post, post_local_idx=jnp.arange(1), n_post_pop=1,
            pre_idx=jnp.array([0]), post_idx=jnp.array([0]),
            rule=_StaticTestRule())   # no post_state_reads


def test_requires_post_population():
    with pytest.raises(ValueError, match='post'):
        VoltageCoupledPlasticProj(
            pre_spike=lambda: jnp.array([1.]), n_pre_pop=1, pre_local_idx=jnp.arange(1),
            post=None, post_local_idx=jnp.arange(1), n_post_pop=1,
            pre_idx=jnp.array([0]), post_idx=jnp.array([0]),
            rule=_ReaderProbe())


class _SignalProbe(_StaticTestRule):
    """Rule that declares a broadcast signal_reads (the cluster-08 dopamine seam)."""
    signal_reads = ('n',)


def test_requires_signal_source_when_declared():
    # a rule declaring signal_reads but no bound source is a construction error
    # (the Simulator binds these from connect(..., vt=...); a bare proj must too).
    post = _PostStub(V=[0.], u_plus=[0.], u_minus=[0.])
    with pytest.raises(ValueError, match='signal_reads'):
        VoltageCoupledPlasticProj(
            pre_spike=lambda: jnp.array([1.]), n_pre_pop=1, pre_local_idx=jnp.arange(1),
            post=post, post_local_idx=jnp.arange(1), n_post_pop=1,
            pre_idx=jnp.array([0]), post_idx=jnp.array([0]),
            rule=_SignalProbe())   # signal_reads=('n',) but signal_sources=None


# --------------------------------------------------------------------------
# Task 1 — primitive #1 intact: base EventPlasticProj exposes ctx.post_states=None
# --------------------------------------------------------------------------
def test_primitive_one_ctx_post_states_is_none():
    brainstate.environ.set(dt=0.1 * u.ms)
    sink = _PostStub(V=[0.], u_plus=[0.], u_minus=[0.])
    rule = _CtxProbe(weight=jnp.array([0.]) * u.pA)
    proj = EventPlasticProj(
        pre_spike=lambda: jnp.array([1.]), n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=sink, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    with brainstate.environ.context(t=0.1 * u.ms, i=1):
        proj.update()
    assert rule.seen is None


# --------------------------------------------------------------------------
# Task 3 — jit / vmap / grad smoke through the post-state read path
# --------------------------------------------------------------------------
class _VoltageKernel(_StaticTestRule):
    """Minimal voltage-coupled kernel: w_eff = w + V mantissa per edge."""
    post_state_reads = ('V',)

    def update(self, state, ctx):
        w = state['weight'] + ctx.post_states['V']
        return {'weight': w}, w


def test_jit_step_runs_with_post_state_reader():
    brainstate.environ.set(dt=0.1 * u.ms)
    post = _PostStub(V=[2.0, 3.0], u_plus=[0., 0.], u_minus=[0., 0.])
    proj = VoltageCoupledPlasticProj(
        pre_spike=lambda: jnp.array([1., 1.]), n_pre_pop=2, pre_local_idx=jnp.arange(2),
        post=post, post_local_idx=jnp.arange(2), n_post_pop=2,
        pre_idx=jnp.array([0, 1]), post_idx=jnp.array([0, 1]),
        rule=_VoltageKernel(weight=jnp.array([1., 1.]) * u.pA))
    brainstate.nn.init_all_states(proj)

    @brainstate.transform.jit   # stateful module -> brainstate's jit (tracks State writes)
    def step():
        with brainstate.environ.context(t=0.1 * u.ms, i=1):
            return proj.update()

    out = step()
    # delivered weight = w(1) + V(2,3) = (3, 4); both pre fired -> delivered as-is
    assert np.allclose(np.asarray(u.get_mantissa(out)), [3.0, 4.0])


def test_grad_flows_through_post_state_reads():
    post = _PostStub(V=[5.0], u_plus=[0.], u_minus=[0.])
    rule = _VoltageKernel()

    def loss(vscale):
        # probe gradient of the read path w.r.t. a scaling of the post voltage read
        from brainpy_state._nest_network._event_plastic import KernelContext
        ctx = KernelContext(
            pre_spike=jnp.ones(1), post_spike=jnp.zeros(1), pre_trace=jnp.zeros(1),
            post_trace=jnp.zeros(1), t_now=jnp.asarray(0.0), dt=jnp.asarray(0.1),
            key=jax.random.key(0), post_states={'V': jnp.asarray([5.0]) * vscale})
        _, w_eff = rule.update({'weight': jnp.zeros(1)}, ctx)
        return jnp.sum(w_eff)

    g = jax.grad(loss)(1.0)
    assert np.allclose(np.asarray(g), 5.0)
