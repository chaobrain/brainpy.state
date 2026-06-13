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

from brainstate import transform  # noqa: E402

from brainpy_state._network._event_plastic import (  # noqa: E402
    EventPlasticProj,
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
# Cluster 06 — `fractional_delay` two-slot output-carry seam (default-off).
# A rule may declare `fractional_delay = True` (cont_delay_synapse) to honour a
# sub-dt delay on the fixed grid: deliver the binary event at the integer FLOOR
# delay (existing InputDelay + BinaryArray fast path, so ctx.pre_spike stays
# binary), then split the post-segment amplitude across the two bracketing grid
# steps via a 1-step FIR carry `[1-frac, frac]` (frac = d/dt - floor(d/dt)).
# First moment + total charge are exact; only the sub-dt transient is an
# approximation. The seam is additive and DEFAULT-OFF: rules without the flag
# keep the unchanged InputDelay(rule.delay) path. See spec §5.
# --------------------------------------------------------------------------
class _FracDelayRule(_StaticTestRule):
    """Static rule that opts into the two-slot fractional-delay split."""
    fractional_delay = True


class _FracPreProbe(_StaticTestRule):
    """Records ctx.pre_spike each step (to assert it stays binary)."""
    fractional_delay = True

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen_pre = []

    def update(self, state, ctx):
        self.seen_pre.append(np.asarray(ctx.pre_spike))
        return state, state['weight']


def _delivery_series(rule, n_steps, dt_ms=0.1):
    """Per-step delivered amplitude (mantissa) for a single spike at step 0."""
    brainstate.environ.set(dt=dt_ms * u.ms)
    sink = _Sink(1)
    holder = {'pre': jnp.array([1.])}        # spike present only on the first step
    proj = EventPlasticProj(
        pre_spike=lambda: holder['pre'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=sink, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    out = []
    for i in range(n_steps):
        with brainstate.environ.context(t=dt_ms * (i + 1) * u.ms, i=i + 1):
            proj.update()
        out.append(float(u.get_mantissa(sink.last)[0]))
        holder['pre'] = jnp.array([0.])
    return np.array(out)


def test_fractional_delay_splits_amplitude_1p7():
    # d = 1.7*dt -> floor=1, frac=0.7 -> 0.3*w one step after the buffer, 0.7*w next.
    out = _delivery_series(_FracDelayRule(weight=jnp.array([10.]) * u.pA, delay=0.17 * u.ms), 5)
    assert np.allclose(out, [0.0, 3.0, 7.0, 0.0, 0.0], atol=1e-9)


def test_fractional_delay_splits_amplitude_1p5():
    # d = 1.5*dt -> floor=1, frac=0.5 -> even 0.5/0.5 split.
    out = _delivery_series(_FracDelayRule(weight=jnp.array([10.]) * u.pA, delay=0.15 * u.ms), 5)
    assert np.allclose(out, [0.0, 5.0, 5.0, 0.0, 0.0], atol=1e-9)


def test_fractional_delay_sub_dt_floor_zero():
    # d = 0.7*dt -> floor=0 (no buffer), frac=0.7 -> 0.3*w THIS step, 0.7*w next.
    out = _delivery_series(_FracDelayRule(weight=jnp.array([10.]) * u.pA, delay=0.07 * u.ms), 4)
    assert np.allclose(out, [3.0, 7.0, 0.0, 0.0], atol=1e-9)


def test_fractional_delay_charge_conserved_and_centroid():
    # The FIR [1-frac, frac] conserves total weight and places the arrival
    # centroid (in steps, relative to the input step) at exactly d/dt.
    w, d_ms, dt = 10.0, 0.17, 0.1
    out = _delivery_series(_FracDelayRule(weight=jnp.array([w]) * u.pA, delay=d_ms * u.ms), 6)
    assert np.sum(out) == pytest.approx(w, abs=1e-9)                  # charge exact
    centroid = np.sum(out * np.arange(len(out))) / np.sum(out)
    assert centroid == pytest.approx(d_ms / dt, abs=1e-9)            # first moment exact


def test_fractional_delay_integer_delay_is_plain_grid():
    # frac == 0 (integer delay) with the flag set -> NO carry -> identical to a
    # plain grid delay: full w lands at exactly one step (floor==total).
    out = _delivery_series(_FracDelayRule(weight=jnp.array([7.]) * u.pA, delay=0.2 * u.ms), 5)
    assert np.allclose(out, [0.0, 0.0, 7.0, 0.0, 0.0], atol=1e-9)


def test_fractional_delay_default_off_for_plain_rule():
    # A rule WITHOUT the flag must take the unchanged InputDelay path: an integer
    # delay still delivers the full weight at one step (regression guard).
    out = _delivery_series(_StaticTestRule(weight=jnp.array([7.]) * u.pA, delay=0.2 * u.ms), 5)
    assert np.allclose(out, [0.0, 0.0, 7.0, 0.0, 0.0], atol=1e-9)
    # and a plain rule never allocates a carry State
    rule = _StaticTestRule(weight=jnp.array([1.]) * u.pA, delay=0.15 * u.ms)
    brainstate.environ.set(dt=0.1 * u.ms)
    proj, _ = _build(1, 1, rule)
    assert getattr(proj, 'delay_carry', None) is None


def test_fractional_delay_keeps_pre_spike_binary():
    # The kernel must see a BINARY ctx.pre_spike even on the fractional path
    # (the split lives on the output amplitude, not the binary pre vector).
    rule = _FracPreProbe(weight=jnp.array([1.]) * u.pA, delay=0.17 * u.ms)
    _delivery_series(rule, 4)
    seen = np.concatenate(rule.seen_pre)
    assert np.all((seen == 0.0) | (seen == 1.0))


def test_fractional_delay_train_superposition():
    # Two spikes 2 steps apart each split independently -> linear superposition.
    brainstate.environ.set(dt=0.1 * u.ms)
    sink = _Sink(1)
    spikes = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    holder = {'pre': jnp.array([spikes[0]])}
    rule = _FracDelayRule(weight=jnp.array([10.]) * u.pA, delay=0.15 * u.ms)  # 0.5/0.5
    proj = EventPlasticProj(
        pre_spike=lambda: holder['pre'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=sink, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    out = []
    for i in range(len(spikes)):
        holder['pre'] = jnp.array([spikes[i]])
        with brainstate.environ.context(t=0.1 * (i + 1) * u.ms, i=i + 1):
            proj.update()
        out.append(float(u.get_mantissa(sink.last)[0]))
    # spike@0 -> 5@step1,5@step2 ; spike@2 -> 5@step3,5@step4
    assert np.allclose(out, [0.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0], atol=1e-9)
    assert np.sum(out) == pytest.approx(20.0, abs=1e-9)              # 2 spikes * 10 pA


def test_fractional_delay_state_advances_in_for_loop():
    # The delay_carry / floor-delay States must be discovered + threaded by
    # transform.for_loop (they are built in init_state) — drive the same single
    # spike and require the JIT-compiled loop to match the eager Python loop.
    brainstate.environ.set(dt=0.1 * u.ms)
    sink = _Sink(1)
    box = {'v': jnp.zeros(1)}
    rule = _FracDelayRule(weight=jnp.array([10.]) * u.pA, delay=0.17 * u.ms)
    proj = EventPlasticProj(
        pre_spike=lambda: box['v'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=sink, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    n = 5
    spk = jnp.asarray(np.array([[1.], [0.], [0.], [0.], [0.]]))
    times = jnp.arange(n) * 0.1 * u.ms
    idx = jnp.arange(n)

    def step(t, i, x):
        box['v'] = x
        with brainstate.environ.context(t=t, i=i):
            return u.get_mantissa(proj.update()[0])

    out = np.asarray(transform.for_loop(step, times, idx, spk)).reshape(-1)
    assert np.allclose(out, [0.0, 3.0, 7.0, 0.0, 0.0], atol=1e-9)
