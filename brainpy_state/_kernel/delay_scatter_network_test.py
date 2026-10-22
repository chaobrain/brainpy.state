# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Per-edge delays in networks (K1a-4): cost model, plan, grouped == scatter, gradients."""
from __future__ import annotations

import contextlib
import dataclasses
import warnings

import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel import compiler
from brainpy_state._kernel.compiler import DELAY_COST_MODELS, DelayCostModel, delay_cost_model
from brainpy_state._kernel.connectivity import (all_to_all, fixed_indegree, fixed_total_number,
                                                 one_to_one)
from brainpy_state._kernel.delay_scatter import DelayScatter
from brainpy_state._kernel.learn import BPTT, EtraceError, ExternalInput, Online, membrane_mse
from brainpy_state._kernel.models import LIF, dc_generator, iaf_psc_alpha, poisson_generator
from brainpy_state._kernel.network import Network
from brainpy_state._kernel.params import Uniform

DT = 0.1 * u.ms
SIG = braintools.surrogate.Sigmoid()


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


@contextlib.contextmanager
def delivery(mode, **over):
    """Force the cost model's choice: ``'grouped'`` or ``'scatter'`` (or ``'auto'``)."""
    old = dict(DELAY_COST_MODELS)
    m = dataclasses.replace(delay_cost_model(), **over)
    if mode == 'scatter':
        m = dataclasses.replace(m, max_groups=0)
    elif mode == 'grouped':
        m = dataclasses.replace(m, max_groups=10 ** 9, group_step_us=0.0, group_edge_ns=0.0)
    for b in {jax.default_backend(), 'cpu'}:
        DELAY_COST_MODELS[b] = m
    try:
        yield
    finally:
        DELAY_COST_MODELS.clear()
        DELAY_COST_MODELS.update(old)


def _cat_b(ref, got):
    """CAT_B (1e-6 absolute + 1e-6 relative to the reference's maximum)."""
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    assert np.max(np.abs(ref - got)) <= 1e-6 + 1e-6 * np.max(np.abs(ref))


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_choice_and_ceiling(self):
        m = DelayCostModel()
        assert m.choose(200_000, 1, True)[0] == 'grouped'
        kind, est = m.choose(200_000, 200, True)
        assert kind == 'scatter' and est['strategy'] == 'events'
        assert est['grouped_build_s'] == pytest.approx(200 * m.group_build_s)
        # above the hard ceiling grouping is never chosen, even if cheaper
        cheap = dataclasses.replace(m, group_step_us=0.0, group_edge_ns=0.0)
        assert cheap.choose(1000, m.max_groups, True)[0] == 'grouped'
        assert cheap.choose(1000, m.max_groups + 1, True)[0] == 'scatter'

    def test_strategies(self):
        m = DelayCostModel()
        assert m.strategy(10, event=False) == 'dense'
        assert m.strategy(10, event=True) == 'events'
        g = DELAY_COST_MODELS['gpu']
        assert g.strategy(1000, True) == 'dense' and g.strategy(g.dense_max_edges + 1, True) == 'events'
        # value sources pay for every edge
        assert m.estimate(1000, 2, False)['scatter_step_us'] == pytest.approx(
            m.scatter_step_us + 1000 * m.dense_edge_ns / 1e3)

    def test_capacity(self):
        m = DelayCostModel()
        assert m.capacity(10 ** 6) == 0  # CPU default: loop only
        g = dataclasses.replace(m, capacity_factor=4.0, capacity_min=32)
        assert g.capacity(10) == 32
        assert g.capacity(10_000) == 120  # 4 * 0.003 * 10000
        assert g.capacity(10_100) == 128  # 121.2 -> 122 -> multiple of 8
        assert DELAY_COST_MODELS['gpu'].capacity(10_000) >= 120

    def test_backend_lookup(self):
        assert delay_cost_model('cpu') is DELAY_COST_MODELS['cpu']
        assert delay_cost_model('gpu') is DELAY_COST_MODELS['gpu']
        assert delay_cost_model('tpu') is DELAY_COST_MODELS['cpu']  # fallback


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _spiking_sources(net, n, rate=400.0, name='src'):
    return net.add(poisson_generator, n, params=dict(rate=rate * u.Hz), name=name)


class TestPlan:
    def test_scatter_plan_and_describe(self):
        net = Network(dt=DT, seed=3)
        src = _spiking_sources(net, 20)
        tgt = net.add(iaf_psc_alpha, 30, name='tgt')
        p = net.connect(src, tgt, fixed_indegree(10), port='ex', weight=5.0 * u.pA,
                        delay=Uniform(0.0 * u.ms, 3.0 * u.ms), name='st')
        plan = net.compile()
        pp = plan.projection(p)
        assert plan.max_delay_groups == delay_cost_model().max_groups
        assert pp.placement == 'post' and pp.delivery == 'scatter'
        assert pp.delivery_cost['n_groups'] > plan.max_delay_groups
        assert [g.delay for g in pp.groups] == [0, None]  # zero-delay group + one scatter
        sc = pp.groups[1].storage
        assert isinstance(sc, DelayScatter) and sc.strategy == 'events'
        assert sc.length == pp.post_ring.length == pp.layout.max
        text = plan.describe(print_=False)
        assert 'delivery=scatter' in text
        assert f'hard ceiling {plan.max_delay_groups} delay groups' in text
        assert f'scatter[events]@d=1..{pp.layout.max}' in text
        assert plan.memory()['storage'] >= sc.nbytes(4)

    def test_few_delays_are_grouped(self):
        net = Network(dt=DT)
        src = _spiking_sources(net, 4)
        tgt = net.add(LIF, 3, name='tgt')
        p = net.connect(src, tgt, all_to_all(), port='delta', weight=0.1 * u.mV,
                        delay=np.tile([0.1, 0.1, 0.0], 4) * u.ms)
        pp = net.compile().projection(p)
        # one delay value >= 1: a single grouped storage beats the scatter's loop
        assert pp.delivery == 'grouped' and [g.delay for g in pp.groups] == [0, 1]
        assert 'grouped' in net.compile().describe(print_=False)

    def test_value_source_uses_dense_strategy_and_scalar_post_side_is_grouped(self):
        net = Network(dt=DT)
        dc = net.add(dc_generator, 3, params=dict(amplitude=[10.0, 20.0, 30.0] * u.pA), name='dc')
        tgt = net.add(iaf_psc_alpha, 4, name='tgt')
        with delivery('scatter'):
            p = net.connect(dc, tgt, all_to_all(), port='I', delay=Uniform(0.1 * u.ms, 2.0 * u.ms))
            q = net.connect(dc, tgt, all_to_all(), port='I', delay=1.0 * u.ms, delay_buffer='post')
            plan = net.compile()
        assert plan.projection(p).groups[0].storage.strategy == 'dense'
        assert plan.projection(q).delivery == 'scatter'  # forced (ceiling 0)
        assert compiler.delay_cost_model().max_groups > 0  # restored
        plan = net.compile()
        assert plan.projection(q).delivery == 'grouped'  # one delay value: K0 layout


# ---------------------------------------------------------------------------
# grouped == scatter
# ---------------------------------------------------------------------------

def _feedforward(seed, n_src=25, n_tgt=40, w_ff=None):
    """Poisson sources -> iaf_psc_alpha with per-edge delays (delay 0 included, multapses)."""
    net = Network(dt=DT, seed=seed)
    src = _spiking_sources(net, n_src, rate=300.0)
    tgt = net.add(iaf_psc_alpha, n_tgt, params=dict(I_e=300.0 * u.pA), name='tgt')
    w = Uniform(20.0 * u.pA, 60.0 * u.pA) if w_ff is None else w_ff
    net.connect(src, tgt, fixed_total_number(600), port='ex', weight=w,
                delay=Uniform(0.0 * u.ms, 2.5 * u.ms), name='ff')
    net.connect(src, tgt, fixed_indegree(5), port='in_', weight=-30.0 * u.pA,
                delay=Uniform(0.1 * u.ms, 1.3 * u.ms), name='ff_in')
    net.observe(tgt, 'spike', 'V_m', name='rec')
    return net


def _recurrent(seed, n=60, window=False):
    """Recurrent LIF network in an irregular regime (~70 Hz, not saturated)."""
    net = Network(dt=DT, seed=seed)
    src = _spiking_sources(net, n, rate=3000.0)
    A = net.add(LIF, n, params=dict(tau=10.0 * u.ms, V_th=10.0 * u.mV), name='A')
    net.connect(src, A, one_to_one(), port='delta', weight=0.5 * u.mV, delay=0.1 * u.ms,
                name='drive')
    net.connect(A, A, fixed_indegree(12), port='delta', weight=Uniform(-1.5 * u.mV, 1.0 * u.mV),
                delay=Uniform(0.1 * u.ms, 2.0 * u.ms), name='rec_AA')
    net.observe(A, 'spike', 'V', name='rec')
    if window:
        net.observe(A, 'spike', interval=1.0 * u.ms, name='win')
    return net


def _run(build, mode, precision, duration=40.0 * u.ms, **kw):
    with delivery(mode):
        plan = build().compile(precision)
    kinds = {pp.name: pp.delivery for pp in plan.projections if pp.delivery}
    run = plan.runner(**kw)
    res = run.simulate(duration)
    return plan, run, res, kinds


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('seed', [0, 1])
def test_feedforward_grouped_equals_scatter(precision, seed):
    ctx = jax.enable_x64(True) if precision == 'float64' else contextlib.nullcontext()
    with ctx:
        _, run_g, rg, kg = _run(lambda: _feedforward(seed), 'grouped', precision)
        plan_s, run_s, rs, ks = _run(lambda: _feedforward(seed), 'scatter', precision)
        assert set(kg.values()) == {'grouped'} and set(ks.values()) == {'scatter'}
        assert plan_s.projection('ff').layout.has_zero
        sg, ss = np.asarray(rg['rec']['spike']), np.asarray(rs['rec']['spike'])
        np.testing.assert_array_equal(sg, ss)  # counts exact
        assert sg.sum() > 0
        _cat_b(rg['rec']['V_m'].to_decimal(u.mV), rs['rec']['V_m'].to_decimal(u.mV))
        # the arrival rings (inputs still in flight) agree too
        for name in ('ff', 'ff_in'):
            _cat_b(run_g.state['arrivals'][name], run_s.state['arrivals'][name])


@pytest.mark.parametrize('factor,capacity', [(0.0, 0), (1e-9, 8), (1e9, 60)])
def test_recurrent_grouped_equals_scatter(x64, factor, capacity):
    # loop only; a loop-free chunk of 8 (+ the loop); every segment loop-free (60 sources)
    _, _, rg, _ = _run(lambda: _recurrent(4), 'grouped', 'float64', 60.0 * u.ms)
    with delivery('auto', capacity_factor=factor):
        plan_s, _, rs, ks = _run(lambda: _recurrent(4), 'scatter', 'float64', 60.0 * u.ms)
    assert ks == {'rec_AA': 'scatter'}
    assert plan_s.projection('rec_AA').groups[0].storage.capacity == capacity
    assert plan_s.projection('rec_AA').delivery_cost['capacity'] == capacity
    text = plan_s.describe(print_=False)
    assert (f'loop-free capacity {capacity}' in text) == bool(capacity)
    sg = np.asarray(rg['rec']['spike'])
    np.testing.assert_array_equal(sg, np.asarray(rs['rec']['spike']))
    assert sg.sum() > 20
    _cat_b(rg['rec']['V'].to_decimal(u.mV), rs['rec']['V'].to_decimal(u.mV))


def test_value_source_grouped_equals_scatter(x64):
    def build():
        net = Network(dt=DT)
        dc = net.add(dc_generator, 3, params=dict(amplitude=[100.0, 250.0, 400.0] * u.pA,
                                                    start=[1.0, 2.0, 3.0] * u.ms), name='dc')
        tgt = net.add(iaf_psc_alpha, 4, name='tgt')
        net.connect(dc, tgt, all_to_all(), port='I', weight=0.5,
                    delay=Uniform(0.1 * u.ms, 2.0 * u.ms), name='dcI')
        net.observe(tgt, 'V_m', name='rec')
        return net
    _, _, rg, _ = _run(build, 'grouped', 'float64', 10.0 * u.ms)
    plan, _, rs, ks = _run(build, 'scatter', 'float64', 10.0 * u.ms)
    assert plan.projection('dcI').groups[0].storage.strategy == 'dense'
    _cat_b(rg['rec']['V_m'].to_decimal(u.mV), rs['rec']['V_m'].to_decimal(u.mV))


def test_exact_arrival_steps_nest_semantics():
    """One spike over edges with delays 1..L arrives exactly d steps later (and d = L)."""
    from brainpy_state._kernel.models import spike_generator
    net = Network(dt=DT)
    sg = net.add(spike_generator, 1, params=dict(spike_times=[0.5] * u.ms), name='sg')  # step 4
    n = net.add(LIF, 6, params=dict(tau=10.0 * u.ms), name='n')
    d = [0.1, 0.2, 0.5, 0.9, 1.5, 0.0] * u.ms
    net.connect(sg, n, all_to_all(), port='delta', weight=2.0 * u.mV, delay=d)
    rec = net.observe(n, 'spike')
    with delivery('scatter'):
        plan = net.compile()
    assert plan.projections[0].delivery == 'scatter' and plan.projections[0].post_ring.length == 15
    res = plan.runner().simulate(3.0 * u.ms)
    s = np.asarray(res[rec]['spike'])
    assert [int(np.nonzero(s[:, k])[0][0]) for k in range(6)] == [5, 6, 9, 13, 19, 4]


def test_batched_runner_matches_unbatched():
    _, _, r1, _ = _run(lambda: _recurrent(2, n=30), 'scatter', 'float32', 30.0 * u.ms)
    _, _, rb, _ = _run(lambda: _recurrent(2, n=30), 'scatter', 'float32', 30.0 * u.ms, batch=3)
    sb = np.asarray(rb['rec']['spike'])
    assert sb.shape[0] == 3
    np.testing.assert_array_equal(sb[0], np.asarray(r1['rec']['spike']))
    assert not np.array_equal(sb[0], sb[1])  # independent trials
    assert 0 < sb.mean() < 0.05  # irregular regime, not saturated


def test_windowed_observer_and_cont():
    with delivery('scatter'):
        plan = _recurrent(5, n=30, window=True).compile()
    run = plan.runner()
    r1 = run.simulate(20.0 * u.ms)
    r2 = run.cont(20.0 * u.ms)
    full = plan.runner().simulate(40.0 * u.ms)
    np.testing.assert_array_equal(np.concatenate([np.asarray(r1['rec']['spike']),
                                                  np.asarray(r2['rec']['spike'])]),
                                  np.asarray(full['rec']['spike']))
    win = np.asarray(full['win']['spike'])
    assert win.shape[0] == 40 and win.sum() == np.asarray(full['rec']['spike']).sum()


# ---------------------------------------------------------------------------
# edge order
# ---------------------------------------------------------------------------

def test_edges_and_set_weight_exact():
    with delivery('scatter'):
        net = _feedforward(7)
        plan = net.compile()
    proj = net.projections[0]
    es = plan.edges(proj)
    raw = plan.projection(proj).edges.weight
    np.testing.assert_array_equal(np.asarray(es.weight.to_decimal(u.pA)),
                                  np.asarray(raw.to_decimal(u.pA)))
    conns = es.get_connections(source=[0, 3])
    new = (np.arange(len(conns)) + 1.0) * u.pA
    es2 = es.set(weight=new, edges=conns)
    with delivery('scatter'):
        net2 = _feedforward(7, w_ff=es2.weight)  # rebuild with the per-edge weights
        plan2 = net2.compile()
    assert plan2.edges(net2.projections[0]).same_edges(es2)
    got = plan2.edges(net2.projections[0])
    np.testing.assert_array_equal(np.asarray(got.weight.to_decimal(u.pA)),
                                  np.asarray(es2.weight.to_decimal(u.pA)))
    np.testing.assert_array_equal(np.asarray(got.get_connections(source=[0, 3]).weight.to_decimal(u.pA)),
                                  np.asarray(new.to_decimal(u.pA)))


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _ff_bptt_net(steps):
    net = Network(dt=DT)
    x = net.add(ExternalInput, 6, name='x')
    h = net.add(LIF, 20, params=dict(tau=8.0 * u.ms), name='h', surrogate=SIG)
    w = np.linspace(0.5, 1.5, 120) * 0.35 * u.mV
    net.connect(x, h, port='delta', weight=w, delay=np.asarray(steps) * DT, trainable=True, name='xh')
    net.observe(h, 'spike', name='rec_h')
    net.observe(h, 'V', name='v_h')
    return net


def test_bptt_finite_differences_through_scatter(x64):
    """G6 regime (a): hard forward, detach_reset, membrane loss, per-edge delays (scatter)."""
    from brainpy_state._kernel.learn.bptt_test import _check_fd, _xs
    steps = np.random.default_rng(0).integers(1, 12, 120)
    with delivery('scatter'):
        plan = _ff_bptt_net(steps).compile(precision='float64')
    pp = plan.projection('xh')
    assert pp.delivery == 'scatter' and isinstance(pp.groups[0].storage, DelayScatter)
    bptt = BPTT(plan.program(), lambda r, y: membrane_mse(r['v_h']['V'], y), detach_reset=True)
    xs = {'x': _xs(60, 6)}
    rec = bptt.record(None, xs)
    assert float(np.sum(rec['rec_h']['spike'])) > 0
    grads = _check_fd(bptt, None, xs, 0.8)
    assert set(grads) == {'weights/xh/0'}


def test_bptt_gradient_equals_grouped(x64):
    steps = np.random.default_rng(1).integers(0, 6, 120)  # includes zero delays
    out = {}
    for mode in ('grouped', 'scatter'):
        with delivery(mode):
            plan = _ff_bptt_net(steps).compile(precision='float64')
        from brainpy_state._kernel.learn.bptt_test import _xs
        bptt = BPTT(plan.program(), lambda r, y: membrane_mse(r['v_h']['V'], y), detach_reset=True)
        loss, grads = bptt.grad(None, {'x': _xs(50, 6, seed=3)}, 0.8)
        edges_w = {}
        pp = plan.projection('xh')
        g_edge = np.zeros(120)
        for g, grp in enumerate(pp.groups):
            g_edge[grp.edges] = np.asarray(u.get_mantissa(grp.storage.weight_to_edges(
                grads[f'weights/xh/{g}'])))
        out[mode] = (float(loss), g_edge)
    assert out['grouped'][0] == pytest.approx(out['scatter'][0], rel=1e-12)
    np.testing.assert_allclose(out['scatter'][1], out['grouped'][1], rtol=1e-9, atol=1e-14)


# ---------------------------------------------------------------------------
# braintrace
# ---------------------------------------------------------------------------

def _etrace_net(trainable_delayed):
    net = Network(dt=DT)
    x = net.add(ExternalInput, 6, name='x')
    h = net.add(LIF, 12, params=dict(tau=10.0 * u.ms), name='h', surrogate=SIG)
    o = net.add(LIF, 4, params=dict(tau=10.0 * u.ms), name='o', surrogate=SIG)
    net.connect(x, h, port='delta', weight=0.4 * u.mV, delay=DT, trainable=True, name='xh')
    net.connect(h, h, fixed_indegree(4), port='delta', weight=0.1 * u.mV,
                delay=jnp.array([0.1, 0.2, 0.3] * 16) * u.ms, trainable=trainable_delayed, name='hh')
    net.connect(h, o, fixed_indegree(6), port='delta', weight=0.3 * u.mV, delay=0.0 * u.ms,
                trainable=True, name='ho')
    return net


@pytest.mark.parametrize('mode', ['grouped', 'scatter'])
def test_etrace_rejects_trainable_per_edge_delays_clearly(mode):
    with delivery(mode):
        plan = _etrace_net(True).compile()
    assert plan.projection('hh').delivery == mode
    with pytest.raises(EtraceError, match=r"Projections \['hh'\] have per-edge delays"):
        Online(plan.program(), 'd_rtrl', inputs='x', outputs=['h', 'o'])


def test_etrace_with_constant_scatter_projection():
    with delivery('scatter'):
        plan = _etrace_net(False).compile()
    assert plan.projection('hh').delivery == 'scatter'
    online = Online(plan.program(), 'd_rtrl', inputs='x', outputs=['h', 'o'],
                    loss=lambda out, y: jnp.mean((out - y) ** 2))
    assert len(online.report.relations) == 2
    xs = (jax.random.uniform(jax.random.key(0), (60, 6)) < 0.3).astype(jnp.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loss, grads = online.grad(None, xs, jnp.full((60, 16), 0.2))
    assert set(grads) == {'weights/xh/0', 'weights/ho/0'}
    for g in grads.values():
        assert np.all(np.isfinite(np.asarray(u.get_mantissa(g))))


def test_build_brunel_per_edge_delay():
    from brainpy_state._kernel.validation.brunel import build_brunel
    net, _ = build_brunel(20, delay=Uniform(0.1 * u.ms, 2.0 * u.ms))
    plan = net.compile()
    assert plan.projection('E_EI').delivery == 'scatter' == plan.projection('I_EI').delivery
    assert plan.projection('noise_EI').layout.kind == 'scalar'
