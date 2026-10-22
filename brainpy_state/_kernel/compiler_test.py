# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the compiler: schedule, storage/payload/buffer choice, Plan API, describe."""
from __future__ import annotations

import brainstate
import brainunit as u
import numpy as np
import pytest

from brainpy_state._kernel.compiler import Plan, ScheduleError, schedule
from brainpy_state._kernel.connectivity import all_to_all, fixed_indegree, one_to_one
from brainpy_state._kernel.models import LIF, dc_generator, iaf_psc_alpha, poisson_generator, spike_generator
from brainpy_state._kernel.network import Network
from brainpy_state._kernel.params import Normal
from brainpy_state._kernel.validation.brunel import build_brunel

DT = 0.1 * u.ms


class TestSchedule:
    def test_insertion_order_without_constraints(self):
        assert schedule(4, set()) == [0, 1, 2, 3]
        assert schedule(0, set()) == []

    def test_topological_smallest_first(self):
        assert schedule(4, {(3, 0), (2, 1)}) == [2, 1, 3, 0]
        assert schedule(3, {(2, 1), (1, 0), (2, 0)}) == [2, 1, 0]

    def test_cycles(self):
        with pytest.raises(ScheduleError, match='itself'):
            schedule(2, {(1, 1)})
        with pytest.raises(ScheduleError, match=r'\[1, 2\]'):
            schedule(3, {(1, 2), (2, 1)})


def _net():
    net = Network(dt=DT, seed=1)
    sg = net.add(spike_generator, 2, params=dict(spike_times=[1.0] * u.ms), name='sg')
    pg = net.add(poisson_generator, 3, params=dict(rate=10.0 * u.Hz), name='pg')
    dc = net.add(dc_generator, 1, params=dict(amplitude=1.0 * u.pA), name='dc')
    a = net.add(iaf_psc_alpha, 10, name='A')
    return net, sg, pg, dc, a


class TestCompile:
    def test_paths_storages_and_placement(self):
        net, sg, pg, dc, a = _net()
        p1 = net.connect(sg, a, all_to_all(), port='ex', weight=1.0 * u.pA, storage='dense')
        p2 = net.connect(pg, a, fixed_indegree(2), port='ex', weight=1.0 * u.pA, storage='csr',
                         delay=2.0 * u.ms)
        p3 = net.connect(dc, a, port='I')
        p4 = net.connect(a, a, fixed_indegree(3), port='in_', weight=-1.0 * u.pA)
        plan = net.compile()
        pp1, pp2, pp3 = (plan.projection(p) for p in (p1, p2, p3))
        # generators may emit several events per step (float payload); neurons emit 0/1
        assert pp1.path == 'float' and pp2.path == 'float' and pp3.path == 'float'
        assert plan.projection(p4).path == 'binary'
        assert pp1.groups[0].storage.kind == 'dense' and pp2.groups[0].storage.kind == 'csr'
        assert (pp1.placement, pp1.layout.kind, pp1.groups[0].delay) == ('pre', 'scalar', 1)
        assert plan.population('pg').rings['spike'].length == 20
        assert plan.population('sg').rings['spike'].length == 1
        assert pp1.n_edges == 20 and pp1.name == p1.name
        assert plan.population(3).name == 'A' and plan.observers == []
        assert plan.projection(p2.name) is pp2 and plan.projection(0) is pp1
        with pytest.raises(KeyError):
            plan.population('nope')
        with pytest.raises(KeyError):
            plan.projection('nope')
        assert isinstance(plan, brainstate.nn.Module) and 'Plan(' in repr(plan)

    def test_trainable_weights_and_edges_roundtrip(self):
        net, sg, pg, dc, a = _net()
        w = np.linspace(1.0, 2.0, 20) * u.pA
        p = net.connect(sg, a, all_to_all(), port='ex', weight=w, trainable=True)
        q = net.connect(sg, a, all_to_all(), port='ex',
                        weight=Normal(5.0 * u.pA, 1.0 * u.pA),
                        delay=np.tile([0.1, 0.2], 10) * u.ms, delay_buffer='post')
        plan = net.compile()
        assert len(plan.states(brainstate.ParamState)) == 1
        es = plan.edges(p)
        np.testing.assert_allclose(np.asarray(es.weight.to_decimal(u.pA)),
                                   np.asarray(w.to_decimal(u.pA)), rtol=1e-6)
        # post-side groups scatter back into the canonical edge order
        eq = plan.edges(q)
        # two distinct delays -> the cost model picks the single scatter group (K1a-4)
        assert plan.projection(q).placement == 'post' and len(plan.projection(q).groups) == 1
        assert plan.projection(q).delivery == 'scatter'
        raw = plan.projection(q).edges.weight
        np.testing.assert_allclose(np.asarray(eq.weight.to_decimal(u.pA)),
                                   np.asarray(raw.to_decimal(u.pA)), rtol=1e-6)
        # same seed -> same sampled weights
        net2, sg2, _, _, a2 = _net()
        net2.connect(sg2, a2, all_to_all(), port='ex', weight=w)
        net2.connect(sg2, a2, all_to_all(), port='ex', weight=Normal(5.0 * u.pA, 1.0 * u.pA),
                     delay=np.tile([0.1, 0.2], 10) * u.ms)
        np.testing.assert_array_equal(np.asarray(net2.compile().edges(1).weight.to_decimal(u.pA)),
                                      np.asarray(raw.to_decimal(u.pA)))

    def test_errors(self):
        net, sg, pg, dc, a = _net()
        net.connect(sg, a, all_to_all(), port='ex', weight=[1.0, 2.0] * u.pA)
        with pytest.raises(ValueError, match='realised 20 edges'):
            net.compile()
        net, sg, pg, dc, a = _net()
        net.connect(sg, a, all_to_all(), port='ex', weight=1.0 * u.pA, delay=[0.1, 0.2] * u.ms)
        with pytest.raises(ValueError, match='per-edge delays'):
            net.compile()
        net, sg, pg, dc, a = _net()
        net.connect(sg, a, all_to_all(), port='ex', weight=1.0 * u.pA,
                    delay=(np.arange(20) % 3 + 1) * 0.1 * u.ms, delay_buffer='pre')
        with pytest.raises(ValueError, match='pre-side ring'):
            net.compile()

    def test_empty_projection(self):
        net, sg, pg, dc, a = _net()
        p = net.connect(sg, a, fixed_indegree(0), port='ex', weight=1.0 * u.pA)
        plan = net.compile()
        pp = plan.projection(p)
        assert pp.placement == 'none' and pp.groups == [] and plan.edges(p).n_edges == 0
        assert plan.memory()['total'] == 0
        assert 'storage=[none]' in plan.describe(print_=False)

    def test_models_are_built_at_compile_time(self):
        net = Network(dt=DT, seed=3)
        net.add(LIF, 5, name='A')
        net.add(LIF, 2, name='B', spk_reset='hard')
        p1, p2 = net.compile(), net.compile()
        assert [m.name for m in p1.models] == ['A', 'B']
        assert p1.models[0] is not p2.models[0] and p1.models[1].n == 2


class TestDescribe:
    def test_brunel(self, capsys):
        net, info = build_brunel(10)
        plan = net.compile()
        text = plan.describe()
        assert capsys.readouterr().out.strip() == text.strip()
        for s in ('Schedule', 'E -> I -> noise', 'payload=binary', 'payload=float',
                  'pre-side ring E.spike: 15 x 40', 'Memory:', 'Capabilities:', 'BPTT',
                  'rec_E', 'delay=scalar 15 steps', 'buffer=pre-side'):
            assert s in text, s
        mem = plan.memory()
        assert mem['total'] == (mem['storage'] + mem['pre_rings'] + mem['post_rings']
                                + mem['synapses'] + mem['plastic'])
        assert 'Merges: none' in text
        assert mem['pre_rings'] == 15 * (40 + 10 + 50) * 4

    def test_post_side_and_zero_delay(self):
        net = Network(dt=DT)
        sg = net.add(spike_generator, 1, params=dict(spike_times=[1.0] * u.ms), name='sg')
        a = net.add(LIF, 3, name='A')
        net.connect(sg, a, all_to_all(), port='delta', weight=1.0 * u.mV,
                    delay=[0.0, 0.1, 0.3] * u.ms, trainable=True, name='mix')
        text = net.compile().describe(print_=False)
        assert 'post-side ring mix: 3 x 3 float32' in text
        assert 'zero-delay projections (same-step delivery): mix' in text
        assert 'per-edge 0..3 steps' in text and 'dense@d=0, scatter[events]@d=1..3' in text
        # zero-delay group + one scatter group, one weight state each
        assert 'trainable' in text and '2 ParamState' in text
        assert 'mix: 2 distinct delay(s) >= 1 -> scatter (events)' in text
