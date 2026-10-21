# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the Network graph: populations, views, connect-time validation, observers."""
from __future__ import annotations

from typing import ClassVar

import brainunit as u
import numpy as np
import pytest

from brainpy_state._kernel.connectivity import fixed_indegree, one_to_one
from brainpy_state._kernel.integrators import Exact, Integrator
from brainpy_state._kernel.model import Capabilities, NeuronModel
from brainpy_state._kernel.models import LIF, dc_generator, iaf_cond_exp, iaf_psc_alpha, spike_generator
from brainpy_state._kernel.network import Network, PopulationView
from brainpy_state._kernel.params import Init, Normal, States, Uniform
from brainpy_state._kernel.ports import Conductance, Event, PortError, Ports, TimedEvent
from brainpy_state._kernel.units import UnitMismatchError

DT = 0.1 * u.ms


class _Timed(NeuronModel):
    """Minimal model with a timed_event input (T7 hook)."""
    states = States(V=Init(0.0 * u.mV))
    inputs = Ports(precise=TimedEvent(4, unit=u.mV))
    outputs = Ports(spike=Event())
    integrator: ClassVar[Integrator] = Exact()

    def step(self, s, p, inp, ctx):  # pragma: no cover - never run
        return s, {}


class _Gated(NeuronModel):
    """Minimal model with a voltage-gated conductance port (T8)."""
    states = States(V=Init(0.0 * u.mV))
    inputs = Ports(nmda=Conductance(u.nS, E_rev=0.0 * u.mV, gate=lambda V: 1.0))
    integrator: ClassVar[Integrator] = Exact()

    def step(self, s, p, inp, ctx):  # pragma: no cover - never run
        return s, {}


class _HostOnly(NeuronModel):
    capabilities = Capabilities(jit_safe=False)
    integrator: ClassVar[Integrator] = Exact()

    def step(self, s, p, inp, ctx):  # pragma: no cover - never run
        return s, {}


@pytest.fixture
def net():
    return Network(dt=DT, seed=1)


class TestNetwork:
    def test_bad_args(self):
        with pytest.raises(UnitMismatchError):
            Network(dt=0.1 * u.mV)
        with pytest.raises(ValueError):
            Network(dt=0.0 * u.ms)
        with pytest.raises(ValueError):
            Network(dt=DT, seed=-1)
        with pytest.raises(ValueError):
            Network(dt=DT, seed=True)

    def test_add_gids_and_names(self, net):
        a = net.add(LIF, 3)
        z = net.add(LIF, 0, name='Z')  # zero-size population
        b = net.add(iaf_psc_alpha, 2, name='B')
        assert a.gids.tolist() == [0, 1, 2] and len(z) == 0 and b.gids.tolist() == [3, 4]
        assert [p.name for p in net.populations] == ['LIF0', 'Z', 'B']
        assert [p.uid for p in net.populations] == [0, 1, 2]
        assert net.n_neurons == 5 and repr(b) == 'PopulationView(B, n=2)'
        assert b.populations[0].name == 'B' and z.segments == []

    def test_add_errors(self, net):
        with pytest.raises(TypeError):
            net.add(int, 2)
        with pytest.raises(ValueError):
            net.add(LIF, -1)
        with pytest.raises(TypeError, match='compiler'):
            net.add(LIF, 1, precision='float64')
        with pytest.raises(TypeError, match='jit-safe'):
            net.add(_HostOnly, 1)
        net.add(LIF, 1, name='x')
        with pytest.raises(ValueError, match='already used'):
            net.add(LIF, 1, name='x')
        with pytest.raises(TypeError):
            net.add(LIF, 1, name='')


class TestViews:
    def test_indexing(self, net):
        a = net.add(LIF, 5, name='A')
        assert a[1].gids.tolist() == [1] and a[-1].gids.tolist() == [4]
        assert a[1:4].gids.tolist() == [1, 2, 3] and a[::2].gids.tolist() == [0, 2, 4]
        assert a[[4, 0]].gids.tolist() == [4, 0]
        assert a[np.array([True, False, True, False, False])].gids.tolist() == [0, 2]
        assert len(a[[]]) == 0 and a.size == 5
        with pytest.raises(IndexError):
            a[5]
        with pytest.raises(IndexError):
            a[np.array([True])]
        with pytest.raises(TypeError):
            a[[0.5]]
        with pytest.raises(ValueError, match='duplicate'):
            a[[0, 0]]

    def test_concat_and_segments(self, net):
        a = net.add(LIF, 4, name='A')
        b = net.add(LIF, 3, name='B')
        v = b[1:] + a[[3, 1]] + a[:1]
        assert v.gids.tolist() == [5, 6, 3, 1, 0]
        segs = v.segments
        assert [(s.pop, s.pos, s.full) for s in segs] == [
            (1, slice(0, 2), True), (0, slice(2, 5), False)]
        assert segs[0].local == slice(1, 3) and segs[1].local_array().tolist() == [3, 1, 0]
        assert [p.name for p in v.populations] == ['B', 'A']
        with pytest.raises(ValueError, match='overlap'):
            a[:3] + a[2:]
        other = Network(dt=DT)
        c = other.add(LIF, 1)
        with pytest.raises(ValueError, match='different networks'):
            a + c
        assert a.__add__(3) is NotImplemented

    def test_view_validation(self, net):
        net.add(LIF, 2)
        with pytest.raises(ValueError, match='out of range'):
            PopulationView(net, [5])


class TestConnect:
    def test_defaults_and_record(self, net):
        sg = net.add(spike_generator, 2)
        n = net.add(LIF, 2)
        p = net.connect(sg, n, one_to_one(), port='delta', weight=1.0 * u.mV)
        assert p.source_port == 'spike' and p.port == 'delta' and p.uid == 2
        assert p.delay == DT and p.storage == 'auto' and not p.trainable
        assert isinstance(p.src_kind, Event) and p.src_kind.max_multiplicity is None
        q = net.connect(n, n, port='delta', weight=0.5 * u.mV, delay=0.0 * u.ms, name='rec')
        assert q.name == 'rec' and type(q.rule).__name__ == 'AllToAll'

    def test_port_errors(self, net):
        a = net.add(iaf_psc_alpha, 2)
        dc = net.add(dc_generator, 1)
        with pytest.raises(PortError, match='ambiguous'):
            net.connect(a, a, weight=1.0 * u.pA)
        with pytest.raises(PortError, match='does not exist'):
            net.connect(a, a, port='nope', weight=1.0 * u.pA)
        with pytest.raises(PortError, match='cannot feed'):
            net.connect(dc, a, port='ex', weight=1.0)  # current -> event
        with pytest.raises(TypeError):
            net.connect(a, a, port=3)

    def test_weight_checks(self, net):
        a = net.add(iaf_psc_alpha, 2)
        c = net.add(iaf_cond_exp, 2)
        with pytest.raises(UnitMismatchError):
            net.connect(a, a, port='ex', weight=1.0 * u.mV)
        with pytest.raises(UnitMismatchError):
            net.connect(a, a, port='ex')  # default weight 1 is dimensionless
        with pytest.raises(PortError, match='non-negative'):
            net.connect(a, c, port='ex', weight=-1.0 * u.nS)
        with pytest.raises(PortError, match='negative'):
            net.connect(a, c, port='ex', weight=Normal(1.0 * u.nS, 1.0 * u.nS))
        net.connect(a, c, port='ex', weight=Normal(1.0 * u.nS, 1.0 * u.nS, low=0.0 * u.nS))

    def test_mixed_targets_need_same_weight_dimension(self, net):
        a = net.add(spike_generator, 1)
        lif = net.add(LIF, 1)
        cond = net.add(iaf_cond_exp, 1)
        with pytest.raises(PortError):
            net.connect(a, lif + cond, port='nope_on_lif', weight=1.0 * u.mV)

    def test_delay_checks(self, net):
        a = net.add(LIF, 2)
        kw = dict(port='delta', weight=1.0 * u.mV)
        with pytest.raises(UnitMismatchError):
            net.connect(a, a, delay=1.0 * u.mV, **kw)
        with pytest.raises(ValueError):
            net.connect(a, a, delay=-1.0 * u.ms, **kw)
        with pytest.raises(ValueError):
            net.connect(a, a, delay=np.ones((2, 2)) * u.ms, **kw)
        with pytest.raises(ValueError, match='clipped'):
            net.connect(a, a, delay=Normal(1.0 * u.ms, 0.5 * u.ms), **kw)
        with pytest.raises(UnitMismatchError):
            net.connect(a, a, delay=Uniform(0.0 * u.mV, 1.0 * u.mV), **kw)
        net.connect(a, a, delay=Uniform(0.1 * u.ms, 1.0 * u.ms), **kw)

    def test_option_checks(self, net):
        a = net.add(LIF, 2)
        kw = dict(port='delta', weight=1.0 * u.mV)
        with pytest.raises(ValueError, match='storage'):
            net.connect(a, a, storage='coo', **kw)
        with pytest.raises(ValueError, match='delay_buffer'):
            net.connect(a, a, delay_buffer='both', **kw)
        with pytest.raises(TypeError, match='ConnRule'):
            net.connect(a, a, 'all_to_all', **kw)
        with pytest.raises(TypeError, match='PopulationView'):
            net.connect(a.gids, a, **kw)
        other = Network(dt=DT).add(LIF, 1)
        with pytest.raises(ValueError, match='another network'):
            net.connect(a, other, **kw)

    def test_extension_hooks_raise(self, net):
        a = net.add(LIF, 2)
        kw = dict(port='delta', weight=1.0 * u.mV)
        with pytest.raises(TypeError, match='SynapseModel'):  # T6 hooks are live
            net.connect(a, a, port='I', weight=1.0 * u.mA, synapse=object())
        with pytest.raises(TypeError, match='PlasticityRule'):
            net.connect(a, a, plasticity=object(), **kw)
        t = net.add(_Timed, 1)
        # T7: timed_event ports are supported (grid events arrive at offset dt)
        p = net.connect(a, t, port='precise', weight=1.0 * u.mV)
        assert p.strict_capacity
        g = net.add(_Gated, 1)
        # T8: gated ports are supported; an event source still needs a graded
        # source (e.g. the NMDA receptor) or a synapse model in between
        with pytest.raises(PortError, match='cannot feed'):
            net.connect(a, g, port='nmda', weight=1.0 * u.nS)

    def test_empty_pre_view(self, net):
        a = net.add(LIF, 2)
        z = net.add(LIF, 0)
        with pytest.raises(PortError, match='ambiguous'):
            net.connect(z, a, port='delta', weight=1.0 * u.mV)
        p = net.connect(z, a, port='delta', weight=1.0 * u.mV, source_port='spike')
        assert p.src_kind.max_multiplicity == 1

    def test_mixed_source_kinds(self, net):
        a = net.add(LIF, 1)
        dc = net.add(dc_generator, 1)
        b = net.add(LIF, 1)
        with pytest.raises(PortError, match='ambiguous'):
            net.connect(a + dc, b, port='delta', weight=1.0 * u.mV)


class TestObserve:
    def test_create(self, net):
        a = net.add(iaf_psc_alpha, 3)
        sg = net.add(spike_generator, 1)
        o = net.observe(a[:2] + sg, 'spike', interval=1.0 * u.ms)
        assert o.interval == 10 and o.events == ('spike',) and o.size == 3 and o.name == 'obs0'
        assert o.n_samples(20) == 2
        with pytest.raises(ValueError, match='multiple'):
            o.n_samples(15)
        v = net.observe(a, 'V_m', 'refractory_count', name='v')
        assert v.interval == 1 and v.events == ()

    def test_errors(self, net):
        a = net.add(iaf_psc_alpha, 3)
        dc = net.add(dc_generator, 1)
        with pytest.raises(ValueError):
            net.observe(a)
        with pytest.raises(ValueError, match='duplicate'):
            net.observe(a, 'V_m', 'V_m')
        with pytest.raises(TypeError):
            net.observe(a, 3)
        with pytest.raises(KeyError, match='no recordable'):
            net.observe(a, 'w')
        with pytest.raises(ValueError, match='multiple of dt'):
            net.observe(a, 'V_m', interval=0.15 * u.ms)
        with pytest.raises(ValueError, match='multiple of dt'):
            net.observe(a, 'V_m', interval=0.0 * u.ms)
        with pytest.raises(UnitMismatchError):
            net.observe(a, 'V_m', interval=1.0 * u.mV)
        # 'I' is an output (current) of dc_generator, a state elsewhere: no event mix-up
        with pytest.raises(KeyError):
            net.observe(a + dc, 'I')
