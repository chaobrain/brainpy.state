# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Per-port latch of ``current`` / ``graded`` inputs (decision D-a, K1a-2).

Covers the port flag, the generated and hand-written steps, the zero-delay chain of
the post-K0 review, the schedule (same-step cycles rejected, latched ports break
cycles through deferred delivery), external input, batching and ``Plan.describe``.
"""
import inspect
import math

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.compiler import ScheduleError, schedule_ports
from brainspike._kernel.connectivity import one_to_one
from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import RK, ExpEuler
from brainspike._kernel.model import NeuronModel, latch_inputs, latch_name, latched_ports
from brainspike._kernel.models import (
    LIF,
    aeif_cond_alpha,
    dc_generator,
    iaf_cond_exp,
    iaf_psc_alpha,
    iaf_psc_exp_ps,
    spike_generator,
)
from brainspike._kernel.network import Network
from brainspike._kernel.params import Init, Params, Record, States
from brainspike._kernel.ports import Current, Event, Graded, Ports, is_latched
from brainspike._kernel.runner import Program
from brainspike._kernel.synapse import Expon

DT = 0.1 * u.ms


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


class LatchedLIF(LIF):
    """The BrainPy ``LIF`` with a NEST-style latched ``I`` (the K0 behaviour)."""

    inputs = Ports(I=Current(u.mA, latch=True), delta=Event(unit=u.mV))


class Leak(NeuronModel):
    """Generated-step toy: ``tau dV/dt = x - V`` with ``x`` from a graded or current port."""

    params = Params(tau=1.0 * u.ms, R=1.0 * u.Gohm)
    states = States(V=Init(0.0 * u.mV))
    inputs = Ports(x=Graded(u.mV, latch=True), y=Graded(u.mV, latch=False),
                   I=Current(u.pA, latch=False))
    outputs = Ports()
    integrator = RK(order=4, substeps=4)

    def derivatives(self, s, p, inp, ctx):
        return {'V': (inp.x + inp.y + p.R * inp.I - s.V) / p.tau}


def _ctx(k=0):
    return StepContext.create(k * DT, DT, k)


# --------------------------------------------------------------------------- ports

class TestPortFlag:
    def test_defaults_and_flag(self):
        assert Current().latch is True and Graded(u.mV).latch is True
        assert Current(latch=False).latch is False and Graded(None, latch=False).latch is False
        assert is_latched(Current()) and not is_latched(Current(latch=False))
        assert not is_latched(Event()) and not is_latched(Graded(u.mV, latch=False))
        assert 'latch=False' in repr(Ports(I=Current(latch=False), e=Event()))
        with pytest.raises(TypeError, match='latch must be a bool'):
            Current(latch=1)
        with pytest.raises(TypeError, match='latch must be a bool'):
            Graded(u.mV, latch=None)
        with pytest.raises(TypeError):
            Current(u.pA, False, True)  # latch is keyword-only

    def test_equality_includes_latch(self):
        assert Current(u.pA) == Current(u.pA, latch=True)
        assert Current(u.pA) != Current(u.pA, latch=False)

    def test_model_declarations(self):
        # NEST models keep the K0 latch and declare it explicitly; LIF opts out (D-a)
        for m in (iaf_psc_alpha, iaf_cond_exp, aeif_cond_alpha, iaf_psc_exp_ps):
            assert m.inputs['I'].latch is True, m.__name__
            assert 'I=Current(u.pA, latch=True)' in inspect.getsource(m), m.__name__
            assert latch_name('I') in m.all_state_decl()
        assert LIF.inputs['I'].latch is False
        assert latched_ports(LIF.inputs) == () and latched_ports(LatchedLIF.inputs) == ('I',)
        assert latched_ports(Leak.inputs) == ('x',)
        assert set(Leak.all_state_decl()) == {'V', 'latch_x'}

    def test_latch_inputs_helper(self):
        s = Record(V=0.0 * u.mV, latch_x=jnp.array([2.0]) * u.mV)
        inp = Record(x=jnp.array([5.0]) * u.mV, y=jnp.array([1.0]) * u.mV,
                     I=jnp.array([3.0]) * u.pA)
        acting, new = latch_inputs(Leak.inputs, s, inp)
        assert u.math.allclose(acting.x, 2.0 * u.mV) and u.math.allclose(acting.y, 1.0 * u.mV)
        assert u.math.allclose(acting.I, 3.0 * u.pA)
        assert list(new) == ['latch_x'] and u.math.allclose(new['latch_x'], 5.0 * u.mV)


# --------------------------------------------------------------------------- model step

class TestGeneratedStep:
    def test_graded_both_ways(self, x64):
        # unlatched graded y acts in the step it arrives; latched x one step later
        a = math.exp(-0.1)
        m = Leak(1, precision='float64')
        m.update(_ctx(0), {'y': 10.0 * u.mV})
        v0 = float(m.dyn_states['V'].value.to_decimal(u.mV)[0])
        assert v0 == pytest.approx(10.0 * (1 - a), rel=1e-8)  # RK4 x 4 substeps
        m2 = Leak(1, precision='float64')
        m2.update(_ctx(0), {'x': 10.0 * u.mV})
        assert float(m2.dyn_states['V'].value.to_decimal(u.mV)[0]) == 0.0
        m2.update(_ctx(1), {})
        assert float(m2.dyn_states['V'].value.to_decimal(u.mV)[0]) == pytest.approx(v0, rel=1e-13)

    def test_unlatched_current_constant_over_substeps(self, x64):
        # same value at every substep: RK4 with more substeps converges to the exact step
        exact = 10.0 * (1 - math.exp(-0.1))  # R I = 1 Gohm * 10 pA = 10 mV
        for integ, rtol in ((RK(4, 1), 1e-6), (RK(4, 8), 1e-9), (ExpEuler(), 1e-13)):
            m = Leak(1, precision='float64', integrator=integ)
            m.update(_ctx(0), {'I': 10.0 * u.pA})
            assert float(m.dyn_states['V'].value.to_decimal(u.mV)[0]) == pytest.approx(exact, rel=rtol)


# --------------------------------------------------------------------------- networks

def _chain(model, mode, L=4, batch=None, precision='float32'):
    """Review chain (/tmp/k0/review/latency.py): one input spike, L layers, zero delays."""
    net = Network(dt=DT, seed=0)
    prev = net.add(spike_generator, 1, params=dict(spike_times=np.array([1.0]) * u.ms))
    layers = []
    for _ in range(L):
        p = net.add(model, 1, params=dict(V_th=0.001 * u.mV))
        if mode == 'delta':
            net.connect(prev, p, one_to_one(), port='delta', weight=5 * u.mV, delay=0 * u.ms)
        else:
            net.connect(prev, p, one_to_one(), port='I', weight=5 * u.mA, delay=0 * u.ms,
                        synapse=Expon(2 * u.ms))
        layers.append(p)
        prev = p
    obs = [net.observe(p, 'spike') for p in layers]
    plan = net.compile(precision=precision)
    res = plan.runner(batch=batch).simulate(5 * u.ms)
    spk = [np.asarray(res[o].data['spike']) for o in obs]
    first = lambda a: int(np.argmax(a > 0))
    if batch is None:
        return [first(a[:, 0]) for a in spk], plan
    return [[first(a[b, :, 0]) for a in spk] for b in range(batch)], plan


class TestZeroDelayChain:
    def test_unlatched_lif_current_same_step(self):
        steps, _ = _chain(LIF, 'expon')
        assert steps == [9, 9, 9, 9]  # K0 (latched): 10, 11, 12, 13

    def test_event_port_unchanged(self):
        assert _chain(LIF, 'delta')[0] == [9, 9, 9, 9]

    def test_latched_port_one_step_per_layer(self):
        assert _chain(LatchedLIF, 'expon')[0] == [10, 11, 12, 13]  # the K0 numbers

    def test_batched(self):
        steps, _ = _chain(LIF, 'expon', batch=2)
        assert steps == [[9, 9, 9, 9]] * 2

    def test_unlatched_port_behind_unit_delay(self):
        net = Network(dt=DT, seed=0)
        sg = net.add(spike_generator, 1, params=dict(spike_times=np.array([1.0]) * u.ms))
        a = net.add(LIF, 1, params=dict(V_th=0.001 * u.mV), name='a')
        b = net.add(LIF, 1, params=dict(V_th=0.001 * u.mV), name='b')
        net.connect(sg, a, port='delta', weight=5 * u.mV, delay=0 * u.ms)
        net.connect(a, b, port='I', weight=5 * u.mA, delay=DT, synapse=Expon(2 * u.ms))
        oa, ob = net.observe(a, 'spike'), net.observe(b, 'spike')
        res = net.compile().runner().simulate(3 * u.ms)
        first = lambda o: int(np.argmax(np.asarray(res[o].data['spike'])[:, 0] > 0))
        assert (first(oa), first(ob)) == (9, 10)


class TestSchedule:
    def test_schedule_ports_unit(self):
        # acyclic: the K0 order over all edges, nothing deferred
        assert schedule_ports(3, {(1, 2)}, {(2, 0, 'I')}) == ([1, 2, 0], {})
        # cycle through a latched port: same-step order, latched port deferred
        assert schedule_ports(2, {(1, 0)}, {(0, 1, 'I')}) == ([1, 0], {1: frozenset({'I'})})
        # zero-delay self-connection into a latched port
        assert schedule_ports(1, set(), {(0, 0, 'I'), (0, 0, 'J')}) == ([0], {0: frozenset({'I', 'J'})})
        with pytest.raises(ScheduleError):
            schedule_ports(2, {(0, 1), (1, 0)}, set())

    def test_unlatched_zero_delay_cycle_rejected(self):
        net = Network(dt=DT)
        a, b = net.add(LIF, 1, name='A'), net.add(LIF, 1, name='B')
        net.connect(a, b, port='I', weight=1 * u.mA, delay=0 * u.ms, synapse=Expon(2 * u.ms))
        net.connect(b, a, port='I', weight=1 * u.mA, delay=0 * u.ms, synapse=Expon(2 * u.ms))
        with pytest.raises(ScheduleError, match='latched port'):
            net.compile()

    def test_unlatched_zero_delay_self_loop_rejected(self):
        net = Network(dt=DT)
        a = net.add(LIF, 1, name='A')
        net.connect(a, a, port='I', weight=1 * u.mA, delay=0 * u.ms, synapse=Expon(2 * u.ms))
        with pytest.raises(ScheduleError, match='feeds itself'):
            net.compile()

    def _cycle(self, ab_weight, ba=True):
        net = Network(dt=DT, seed=0)
        sg = net.add(spike_generator, 1, params=dict(spike_times=np.array([1.0, 2.0]) * u.ms))
        a = net.add(LIF, 1, params=dict(V_th=0.5 * u.mV), name='A')
        b = net.add(LatchedLIF, 1, params=dict(V_th=0.5 * u.mV), name='B')
        net.connect(sg, a, port='delta', weight=5 * u.mV, delay=0 * u.ms)
        if ba:
            net.connect(b, a, port='delta', weight=0.3 * u.mV, delay=0 * u.ms)  # same-step: B first
        net.connect(a, b, port='I', weight=ab_weight, delay=0 * u.ms,  # latched: deferred
                    synapse=Expon(2 * u.ms))
        obs = net.observe(a + b, 'V')
        return net, obs

    def test_latched_port_breaks_zero_delay_cycle(self):
        net, obs = self._cycle(5 * u.mA)
        plan = net.compile()
        names = [p.name for p in plan.populations]
        assert [names[i] for i in plan.order][-2:] == ['B', 'A']
        assert {names[b]: set(p) for b, p in plan.deferred.items()} == {'B': {'I'}}
        assert 'deferred latched ports (gathered after all updates, act next step): B.I' in \
            plan.describe(print_=False)
        V = np.asarray(plan.runner().simulate(4 * u.ms)[obs]['V'].to_decimal(u.mV))
        # A fires at step 9 (input spike, same step); B's latched I acts from step 10
        assert V[9, 1] == 0.0 and V[10, 1] > 0.0

    def test_deferred_equals_ordered(self, x64):
        """Deferring a latched port gives the same numbers as updating its source first."""
        # without the same-step edge B -> A the graph is acyclic: the K0 order puts A first
        net, obs = self._cycle(5 * u.mA, ba=False)
        p1, p2 = net.compile('float64'), net.compile('float64')
        names = [p.name for p in p1.populations]
        a, b = names.index('A'), names.index('B')
        assert p1.deferred == {} and p1.order.index(a) < p1.order.index(b)
        # force B before A with B.I deferred, as a cycle through B.I would need
        p2.order = [i for i in p2.order if i not in (a, b)] + [b, a]
        p2.deferred = {b: frozenset({'I'})}
        prog1, prog2 = Program(p1), Program(p2)
        s1, s2 = prog1.init_state(), prog2.init_state()
        par1, par2 = prog1.read_params(), prog2.read_params()
        step1, step2 = jax.jit(prog1.step), jax.jit(prog2.step)
        ext = {'B': {'I': 0.2 * u.mA}, 'A': {'I': 0.1 * u.mA}}
        spikes = 0.0
        for n in range(60):
            e = ext if n % 7 == 3 else {}
            s1, o1 = step1(par1, s1, n, jnp.uint32(0), ext=e)
            s2, o2 = step2(par2, s2, n, jnp.uint32(0), ext=e)
            spikes += float(o1[b]['spike'].sum())
            for x, y in zip(jax.tree.leaves(s1), jax.tree.leaves(s2)):
                np.testing.assert_array_equal(np.asarray(u.get_mantissa(x)),
                                              np.asarray(u.get_mantissa(y)))
        assert spikes > 0  # B was driven through the deferred port


class TestExternalInput:
    def _program(self, model):
        net = Network(dt=DT)
        net.add(model, 1, name='n', params=dict(V_th=1e3 * u.mV))
        plan = net.compile('float64')
        return Program(plan)

    def test_ext_into_latched_and_unlatched_ports(self, x64):
        for model, lag in ((LIF, 0), (LatchedLIF, 1)):
            prog = self._program(model)
            s, par = prog.init_state(), prog.read_params()
            Vs = []
            for n in range(3):
                ext = {'n': {'I': 1.0 * u.mA}} if n == 0 else None
                s, _ = prog.step(par, s, n, jnp.uint32(0), ext=ext)
                Vs.append(float(u.get_mantissa(s['pops']['n']['V'])[0]))
            assert Vs[lag] > 0.0 and all(v == 0.0 for v in Vs[:lag]), (model.__name__, Vs)


class TestDescribe:
    def test_latch_per_port(self):
        net = Network(dt=DT)
        net.add(LIF, 1, name='L')
        net.add(iaf_psc_alpha, 1, name='N')
        net.add(dc_generator, 1, name='D')
        text = net.compile().describe(print_=False)
        assert 'L: LIF n=1 gids=[0, 1) inputs: I=current(same-step) delta=event' in text
        assert 'inputs: ex=event(into I_syn_ex) in_=event(into I_syn_in) I=current(latch)' in text
        line = [l for l in text.splitlines() if l.startswith('  D: dc_generator')][0]
        assert 'inputs' not in line  # devices have no inputs
        assert 'deferred latched ports' not in text
