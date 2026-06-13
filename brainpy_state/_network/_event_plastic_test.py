# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for the ``EventPlasticProj`` substrate (NEST-free).

Exercises the rule-kernel contract, rule-declared ``State`` allocation, CSR
event-matmul delivery (vs a dense reference), multapse summation, per-neuron
trace machinery, the zero-edge corner, axonal-delay buffering, and the
jit/vmap/grad safety of the hot path.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import saiunit as u

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._network._event_plastic import (  # noqa: E402
    EventPlasticProj,
    VoltageCoupledPlasticProj,
    KernelContext,
    _StaticTestRule,
)
from brainpy_state._network._rules import all_to_all  # noqa: E402


def _ctx(E, t=1.0, dt=0.1):
    z = jnp.zeros(E)
    return KernelContext(pre_spike=jnp.ones(E), post_spike=z, pre_trace=z,
                         post_trace=z, t_now=jnp.asarray(t), dt=jnp.asarray(dt),
                         key=jax.random.key(0))


class _Sink:
    """Minimal post stand-in capturing ``add_delta_input``."""

    def __init__(self, n):
        self.n = n
        self.last = None

    def add_delta_input(self, key, val):
        self.last = val


# --------------------------------------------------------------------------
# Task 1.1 — contract types + trivial rule
# --------------------------------------------------------------------------
def test_static_rule_delivers_constant_weight():
    rule = _StaticTestRule(weight=2.5)
    state = {'weight': jnp.full((3,), 2.5)}
    new_state, w_eff = rule.update(state, _ctx(3))
    assert np.allclose(np.asarray(w_eff), 2.5)
    assert np.allclose(np.asarray(new_state['weight']), 2.5)   # unchanged


# --------------------------------------------------------------------------
# Task 1.2 — construction + init_state (State shapes, zero-edge)
# --------------------------------------------------------------------------
def _build(n_pre, n_post, rule, delay=None, seed=0):
    holder = {'pre': jnp.zeros(n_pre), 'post': jnp.zeros(n_post)}
    proj = EventPlasticProj(
        pre_spike=lambda: holder['pre'], n_pre_pop=n_pre, pre_local_idx=jnp.arange(n_pre),
        post=None, post_local_idx=jnp.arange(n_post), n_post_pop=n_post,
        post_spike=lambda: holder['post'], rule=rule, conn=all_to_all, seed=seed)
    brainstate.nn.init_all_states(proj)
    return proj, holder


def test_weight_state_shape_matches_edges():
    proj, _ = _build(2, 3, _StaticTestRule(weight=1.0))
    assert proj.weight.value.shape == (6,)            # 2x3 all-to-all
    assert np.allclose(np.asarray(proj.weight.value), 1.0)


def test_zero_edge_projection_allocates_empty():
    proj, _ = _build(2, 0, _StaticTestRule())
    assert proj.weight.value.shape == (0,)


def test_homogeneous_weight_is_scalar_state():
    rule = _StaticTestRule(weight=2.0)
    rule.is_homogeneous_weight = True
    proj, _ = _build(2, 3, rule)
    assert proj.weight.value.shape == ()
    assert float(proj.weight.value) == 2.0


# --------------------------------------------------------------------------
# Task 1.3 — delivery (CSR event-matmul) == dense reference; multapses sum
# --------------------------------------------------------------------------
def test_delivery_equals_dense_reference():
    brainstate.environ.set(dt=0.1 * u.ms)
    sink = _Sink(3)
    rule = _StaticTestRule(weight=jnp.array([1.0, 2.0]) * u.pA)
    proj = EventPlasticProj(
        pre_spike=lambda: jnp.array([1., 0.]), n_pre_pop=2,
        pre_local_idx=jnp.arange(2), post=sink, post_local_idx=jnp.arange(3), n_post_pop=3,
        pre_idx=jnp.array([0, 1]), post_idx=jnp.array([0, 2]), rule=rule)
    brainstate.nn.init_all_states(proj)
    with brainstate.environ.context(t=0.1 * u.ms, i=1):
        proj.update()
    got = u.get_mantissa(sink.last)
    assert np.allclose(np.asarray(got), [1.0, 0.0, 0.0])   # only pre0 fired


def test_multapses_sum():
    brainstate.environ.set(dt=0.1 * u.ms)
    sink = _Sink(1)
    rule = _StaticTestRule(weight=jnp.array([1.0, 3.0]) * u.pA)
    proj = EventPlasticProj(
        pre_spike=lambda: jnp.array([1.]), n_pre_pop=1,
        pre_local_idx=jnp.arange(1), post=sink, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0, 0]), post_idx=jnp.array([0, 0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    with brainstate.environ.context(t=0.1 * u.ms, i=1):
        proj.update()
    assert np.allclose(np.asarray(u.get_mantissa(sink.last)), [4.0])


def test_delivery_scatters_into_subsegment():
    brainstate.environ.set(dt=0.1 * u.ms)
    sink = _Sink(4)
    rule = _StaticTestRule(weight=jnp.array([5.0]) * u.pA)
    # target only post index 2 of a 4-neuron population
    proj = EventPlasticProj(
        pre_spike=lambda: jnp.array([1.]), n_pre_pop=1,
        pre_local_idx=jnp.arange(1), post=sink, post_local_idx=jnp.array([2]), n_post_pop=4,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    with brainstate.environ.context(t=0.1 * u.ms, i=1):
        proj.update()
    assert np.allclose(np.asarray(u.get_mantissa(sink.last)), [0.0, 0.0, 5.0, 0.0])


# --------------------------------------------------------------------------
# Task 1.4 — per-neuron trace machinery (synthetic trace rule)
# --------------------------------------------------------------------------
class _TraceTestRule(_StaticTestRule):
    pre_trace_tau = 10.0 * u.ms

    def update(self, state, ctx):
        # deliver the gathered presynaptic trace as the weight (probe)
        return state, ctx.pre_trace


def test_pre_trace_decays_and_gathers():
    brainstate.environ.set(dt=1.0 * u.ms)
    sink = _Sink(1)
    proj = EventPlasticProj(
        pre_spike=lambda: jnp.array([1.]), n_pre_pop=1,
        pre_local_idx=jnp.arange(1), post=sink, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]),
        rule=_TraceTestRule(weight=jnp.array([0.]) * u.pA))
    brainstate.nn.init_all_states(proj)
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()   # trace -> 1.0 (0*e + 1)
    assert np.allclose(np.asarray(proj.pre_trace.value), [1.0])
    # next step, no spike: trace -> exp(-1/10)
    proj.pre_spike = lambda: jnp.array([0.])
    with brainstate.environ.context(t=2.0 * u.ms, i=2):
        proj.update()
    assert np.allclose(np.asarray(proj.pre_trace.value), [np.exp(-0.1)], atol=1e-9)


class _PostTraceTestRule(_StaticTestRule):
    post_trace_tau = 10.0 * u.ms

    def update(self, state, ctx):
        # probe: deliver the gathered postsynaptic trace as the weight
        return state, ctx.post_trace


def test_post_trace_decays_and_gathers():
    # Locks the post-trace seam of the kernel contract (used by STDP clusters):
    # decay-then-add the full post spike vector, gather per edge.
    brainstate.environ.set(dt=1.0 * u.ms)
    sink = _Sink(1)
    post_box = {'v': jnp.array([1.])}
    proj = EventPlasticProj(
        pre_spike=lambda: jnp.array([1.]), n_pre_pop=1,
        pre_local_idx=jnp.arange(1), post=sink, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]),
        post_spike=lambda: post_box['v'],
        rule=_PostTraceTestRule(weight=jnp.array([0.]) * u.pA))
    brainstate.nn.init_all_states(proj)
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()   # post_trace -> 1.0
    assert np.allclose(np.asarray(proj.post_trace.value), [1.0])
    post_box['v'] = jnp.array([0.])
    with brainstate.environ.context(t=2.0 * u.ms, i=2):
        proj.update()
    assert np.allclose(np.asarray(proj.post_trace.value), [np.exp(-0.1)], atol=1e-9)


# --------------------------------------------------------------------------
# Task 4.A — multi-trace per side (stdp_triplet / clopath seam)
# `pre_trace_tau` / `post_trace_tau` accept a tuple of taus; the substrate
# allocates one per-neuron column per tau and exposes them as ctx.pre_traces /
# ctx.post_traces (E, k). Single-tau (1-D State, (E,) alias) and None stay as
# the degenerate cases — cluster-01 specs are unaffected.
# --------------------------------------------------------------------------
class _MultiTraceProbe(_StaticTestRule):
    pre_trace_tau = (10.0 * u.ms, 20.0 * u.ms)
    post_trace_tau = (5.0 * u.ms, 15.0 * u.ms)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = {}

    def update(self, state, ctx):
        self.seen = dict(pre_traces=ctx.pre_traces, post_traces=ctx.post_traces,
                         pre_trace=ctx.pre_trace, post_trace=ctx.post_trace)
        return state, state['weight']


def _build_probe(rule, post_spike=None):
    sink = _Sink(1)
    proj = EventPlasticProj(
        pre_spike=lambda: jnp.array([1.]), n_pre_pop=1,
        pre_local_idx=jnp.arange(1), post=sink, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]),
        post_spike=(post_spike if post_spike is not None else (lambda: jnp.array([1.]))),
        rule=rule)
    brainstate.nn.init_all_states(proj)
    return proj


def test_multi_trace_allocates_2d_state_and_gathers():
    brainstate.environ.set(dt=1.0 * u.ms)
    rule = _MultiTraceProbe(weight=jnp.array([0.]) * u.pA)
    proj = _build_probe(rule)
    assert proj.pre_trace.value.shape == (1, 2)       # one column per tau
    assert proj.post_trace.value.shape == (1, 2)
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()
    assert np.allclose(np.asarray(proj.pre_trace.value), [[1.0, 1.0]])   # +1 both cols
    assert np.allclose(np.asarray(proj.post_trace.value), [[1.0, 1.0]])
    assert rule.seen['pre_traces'].shape == (1, 2)
    assert rule.seen['post_traces'].shape == (1, 2)
    # the singular (E,) alias is the first column
    assert np.allclose(np.asarray(rule.seen['pre_trace']),
                       np.asarray(rule.seen['pre_traces'][:, 0]))
    assert np.allclose(np.asarray(rule.seen['post_trace']),
                       np.asarray(rule.seen['post_traces'][:, 0]))


def test_multi_trace_columns_decay_with_distinct_taus():
    brainstate.environ.set(dt=1.0 * u.ms)
    rule = _MultiTraceProbe(weight=jnp.array([0.]) * u.pA)
    proj = _build_probe(rule, post_spike=lambda: jnp.array([0.]))
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()                       # pre cols -> 1.0 each
    proj.pre_spike = lambda: jnp.array([0.])
    with brainstate.environ.context(t=2.0 * u.ms, i=2):
        proj.update()                       # distinct decay per column
    assert np.allclose(np.asarray(proj.pre_trace.value),
                       [[np.exp(-0.1), np.exp(-0.05)]], atol=1e-9)


class _SingleTraceProbe(_StaticTestRule):
    pre_trace_tau = 10.0 * u.ms

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = {}

    def update(self, state, ctx):
        self.seen = dict(pre_traces=ctx.pre_traces, pre_trace=ctx.pre_trace)
        return state, state['weight']


def test_single_trace_back_compat_traces_alias():
    # scalar tau keeps the 1-D State (cluster-01 contract) and exposes (E,1).
    brainstate.environ.set(dt=1.0 * u.ms)
    rule = _SingleTraceProbe(weight=jnp.array([0.]) * u.pA)
    proj = _build_probe(rule)
    assert proj.pre_trace.value.shape == (1,)         # unchanged 1-D
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()
    assert rule.seen['pre_traces'].shape == (1, 1)
    assert np.allclose(np.asarray(rule.seen['pre_trace']),
                       np.asarray(rule.seen['pre_traces'][:, 0]))


class _NoTraceProbe(_StaticTestRule):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = {}

    def update(self, state, ctx):
        self.seen = dict(pre_traces=ctx.pre_traces, post_traces=ctx.post_traces)
        return state, state['weight']


def test_no_trace_gives_empty_traces_matrix():
    brainstate.environ.set(dt=1.0 * u.ms)
    rule = _NoTraceProbe(weight=jnp.array([0.]) * u.pA)
    proj = _build_probe(rule)
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()
    assert rule.seen['pre_traces'].shape == (1, 0)
    assert rule.seen['post_traces'].shape == (1, 0)


# --------------------------------------------------------------------------
# Task 0 (cluster 05) — per-side `nearest` trace mode (set-to-1 on spike).
# Nearest-neighbour STDP needs a trace that RESETS to 1 on each spike instead of
# accumulating (decay-then-add). The substrate stores the set-to-1 value but still
# GATHERS `decayed + spike` for the kernel, so the cluster-04 exclusion
# `k = ctx.trace - ctx.spike` is unchanged AND recovers the strictly-prior value
# (NEST's "discard the coinciding pair, use the second-latest preceding partner").
# --------------------------------------------------------------------------
class _NearestPreTraceProbe(_StaticTestRule):
    pre_trace_tau = 10.0 * u.ms
    pre_trace_mode = 'nearest'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = []

    def update(self, state, ctx):
        self.seen.append((np.asarray(ctx.pre_trace), np.asarray(ctx.pre_spike)))
        return state, state['weight']


def test_nearest_mode_stores_reset_to_one_and_gathers_decayed_plus_spike():
    brainstate.environ.set(dt=1.0 * u.ms)            # tau=10 ms -> decay exp(-0.1)/step
    rule = _NearestPreTraceProbe(weight=jnp.array([0.]) * u.pA)
    proj = _build_probe(rule, post_spike=lambda: jnp.array([0.]))
    d = np.exp(-0.1)

    # step 1: pre fires. stored set-to-1; gathered = decayed(0)+spike(1) = 1.
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()
    assert np.allclose(np.asarray(proj.pre_trace.value), [1.0])
    assert np.allclose(rule.seen[0][0], [1.0])

    # step 2: no pre. stored decays to exp(-0.1); gathered = same (spike 0).
    proj.pre_spike = lambda: jnp.array([0.])
    with brainstate.environ.context(t=2.0 * u.ms, i=2):
        proj.update()
    assert np.allclose(np.asarray(proj.pre_trace.value), [d], atol=1e-9)
    assert np.allclose(rule.seen[1][0], [d], atol=1e-9)

    # step 3: pre fires again. NEAREST: stored RESETS to 1 (not exp(-0.2)+1 of
    # all-to-all). Gathered = decayed(exp(-0.2)) + spike(1); the kernel exclusion
    # `gathered - pre_spike` = exp(-0.2) = strictly-prior (the second-latest).
    proj.pre_spike = lambda: jnp.array([1.])
    with brainstate.environ.context(t=3.0 * u.ms, i=3):
        proj.update()
    assert np.allclose(np.asarray(proj.pre_trace.value), [1.0])              # reset, not 1.819
    gathered, pre_spike = rule.seen[2]
    assert np.allclose(gathered, [np.exp(-0.2) + 1.0], atol=1e-9)            # unified gather
    assert np.allclose(gathered - pre_spike, [np.exp(-0.2)], atol=1e-9)      # second-latest


def test_default_trace_mode_is_all_to_all_unchanged():
    # No `pre_trace_mode` declared -> decay-then-add; a re-spike accumulates.
    brainstate.environ.set(dt=1.0 * u.ms)
    rule = _SingleTraceProbe(weight=jnp.array([0.]) * u.pA)   # no mode attr
    proj = _build_probe(rule, post_spike=lambda: jnp.array([0.]))
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()                                         # -> 1.0
    proj.pre_spike = lambda: jnp.array([1.])
    with brainstate.environ.context(t=2.0 * u.ms, i=2):
        proj.update()                                         # decay-then-add
    assert np.allclose(np.asarray(proj.pre_trace.value), [np.exp(-0.1) + 1.0], atol=1e-9)


def test_missing_edges_and_conn_raises():
    # Neither explicit edges nor a connectivity rule -> construction error.
    with pytest.raises(ValueError, match='explicit edges'):
        EventPlasticProj(
            pre_spike=lambda: jnp.zeros(2), n_pre_pop=2, pre_local_idx=jnp.arange(2),
            post=None, post_local_idx=jnp.arange(2), n_post_pop=2,
            rule=_StaticTestRule())


# --------------------------------------------------------------------------
# Task 1.5 — jit / vmap / grad smoke, delay buffer wrap
# --------------------------------------------------------------------------
def test_grad_flows_through_effective_weight():
    rule = _StaticTestRule()

    def loss(w):
        state = {'weight': w}
        _, w_eff = rule.update(state, _ctx(4))
        return jnp.sum(w_eff)

    g = jax.grad(loss)(jnp.ones(4))
    assert np.allclose(np.asarray(g), 1.0)


def test_vmap_over_weight_batch():
    rule = _StaticTestRule()

    def run(w):
        _, w_eff = rule.update({'weight': w}, _ctx(4))
        return w_eff

    out = jax.vmap(run)(jnp.ones((5, 4)) * 2.0)
    assert out.shape == (5, 4) and np.allclose(np.asarray(out), 2.0)


def test_jit_step_runs():
    brainstate.environ.set(dt=0.1 * u.ms)
    sink = _Sink(2)
    proj = EventPlasticProj(
        pre_spike=lambda: jnp.array([1., 0.]), n_pre_pop=2,
        pre_local_idx=jnp.arange(2), post=sink, post_local_idx=jnp.arange(2), n_post_pop=2,
        pre_idx=jnp.array([0, 1]), post_idx=jnp.array([0, 1]),
        rule=_StaticTestRule(weight=jnp.array([1., 1.]) * u.pA))
    brainstate.nn.init_all_states(proj)

    @jax.jit
    def step():
        with brainstate.environ.context(t=0.1 * u.ms, i=1):
            return proj.update()

    out = step()  # compiles + runs without host control-flow errors
    assert np.allclose(np.asarray(u.get_mantissa(out)), [1.0, 0.0])


def test_delay_delivers_after_buffer():
    # delay=0.2 ms at dt=0.1 ms -> a spike at step i arrives 2 steps later.
    brainstate.environ.set(dt=0.1 * u.ms)
    sink = _Sink(1)
    holder = {'pre': jnp.array([1.])}
    rule = _StaticTestRule(weight=jnp.array([7.0]) * u.pA, delay=0.2 * u.ms)
    proj = EventPlasticProj(
        pre_spike=lambda: holder['pre'], n_pre_pop=1,
        pre_local_idx=jnp.arange(1), post=sink, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    delivered = []
    for i in range(1, 5):
        with brainstate.environ.context(t=0.1 * i * u.ms, i=i):
            proj.update()
        delivered.append(float(u.get_mantissa(sink.last)[0]))
        holder['pre'] = jnp.array([0.])   # only the first step has a spike
    # full-delay convention: nothing on the first steps, the 7 pA lands after the buffer
    assert delivered[0] == 0.0
    assert max(delivered) == pytest.approx(7.0)


# --------------------------------------------------------------------------
# Task 08 — broadcast-signal reader seam (signal_reads -> ctx.signals).
# A clean superset of the post-state reader: VoltageCoupledPlasticProj reads a
# named scalar State from a bound third-party node (the volume_transmitter) and
# broadcasts it to ALL edges as ctx.signals[name]. Default-empty so primitive #1
# (base EventPlasticProj) and post-only #2 users (clopath) are untouched.
# --------------------------------------------------------------------------
class _SignalNode:
    """Minimal broadcast-signal source: exposes a scalar State ``.n``."""

    def __init__(self, n=0.7):
        self.n = brainstate.State(jnp.asarray([float(n)]))


class _SignalProbe(_StaticTestRule):
    signal_reads = ('n',)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = {}

    def update(self, state, ctx):
        self.seen = dict(signals=ctx.signals)
        # deliver weight + the broadcast scalar (broadcasts () / (1,) against (E,))
        return state, state['weight'] + ctx.signals['n']


class _CaptureSignalsProbe(_StaticTestRule):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen = 'unset'

    def update(self, state, ctx):
        self.seen = ctx.signals
        return state, state['weight']


def test_base_proj_signals_is_none():
    # primitive #1 reads no signals: ctx.signals is None (default-empty seam).
    brainstate.environ.set(dt=1.0 * u.ms)
    rule = _CaptureSignalsProbe(weight=jnp.array([0.]) * u.pA)
    proj = _build_probe(rule, post_spike=lambda: jnp.array([0.]))
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()
    assert rule.seen is None


def _build_signal_proj(rule, vt, *, n_post=1, pre_idx=None, post_idx=None):
    sink = _Sink(n_post)
    proj = VoltageCoupledPlasticProj(
        pre_spike=lambda: jnp.array([1.]), n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=sink, post_local_idx=jnp.arange(n_post), n_post_pop=n_post,
        pre_idx=(jnp.array([0]) if pre_idx is None else pre_idx),
        post_idx=(jnp.array([0]) if post_idx is None else post_idx),
        rule=rule, signal_sources={'n': (vt, 'n')})
    brainstate.nn.init_all_states(proj)
    return proj, sink


def test_voltage_coupled_broadcasts_signal_to_all_edges():
    # the bound node's scalar `n` reaches every edge identically (broadcast 1->E).
    brainstate.environ.set(dt=1.0 * u.ms)
    vt = _SignalNode(n=0.7)
    rule = _SignalProbe(weight=jnp.array([1.0, 2.0]) * u.pA)
    proj, sink = _build_signal_proj(rule, vt, n_post=2,
                                    pre_idx=jnp.array([0, 0]), post_idx=jnp.array([0, 1]))
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()
    assert np.allclose(np.asarray(rule.seen['signals']['n']), 0.7)     # same scalar both edges
    # delivered weight = w + n: edge0->post0 = 1.7, edge1->post1 = 2.7
    assert np.allclose(np.asarray(u.get_mantissa(sink.last)), [1.7, 2.7])


def test_voltage_coupled_accepts_signal_only_rule():
    # dopamine case: signal_reads non-empty, post_state_reads empty -> no raise.
    brainstate.environ.set(dt=1.0 * u.ms)
    vt = _SignalNode()
    rule = _SignalProbe(weight=jnp.array([1.]) * u.pA)
    proj, _ = _build_signal_proj(rule, vt)
    assert proj is not None


def test_voltage_coupled_signal_read_is_read_only():
    # reading the broadcast State must not mutate the source node.
    brainstate.environ.set(dt=1.0 * u.ms)
    vt = _SignalNode(n=0.5)
    before = float(np.asarray(vt.n.value)[0])
    rule = _SignalProbe(weight=jnp.array([1.]) * u.pA)
    proj, _ = _build_signal_proj(rule, vt)
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()
    assert float(np.asarray(vt.n.value)[0]) == before


def test_voltage_coupled_requires_a_reader():
    # neither post_state_reads nor signal_reads -> construction error (preserved guard).
    with pytest.raises(ValueError, match='post_state_reads'):
        VoltageCoupledPlasticProj(
            pre_spike=lambda: jnp.array([1.]), n_pre_pop=1, pre_local_idx=jnp.arange(1),
            post=_Sink(1), post_local_idx=jnp.arange(1), n_post_pop=1,
            pre_idx=jnp.array([0]), post_idx=jnp.array([0]),
            rule=_StaticTestRule(weight=jnp.array([1.]) * u.pA))


def test_voltage_coupled_signal_only_skips_post_gather():
    # a signal-only rule (empty post_state_reads) gets ctx.post_states is None,
    # while the broadcast signal still flows through to ctx.signals.
    brainstate.environ.set(dt=1.0 * u.ms)
    vt = _SignalNode(n=0.25)

    class _Probe(_SignalProbe):
        def update(self, state, ctx):
            self.seen = dict(signals=ctx.signals, post_states=ctx.post_states)
            return state, state['weight']

    rule = _Probe(weight=jnp.array([1.]) * u.pA)
    proj, _ = _build_signal_proj(rule, vt)
    with brainstate.environ.context(t=1.0 * u.ms, i=1):
        proj.update()
    assert rule.seen['post_states'] is None
    assert np.allclose(np.asarray(rule.seen['signals']['n']), 0.25)
