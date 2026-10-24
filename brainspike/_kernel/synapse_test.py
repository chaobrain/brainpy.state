# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of synapse models: Expon numerics, connect-time checks, the merge pass."""
from __future__ import annotations

from typing import ClassVar

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.connectivity import all_to_all, fixed_indegree, one_to_one
from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import Exact, Integrator
from brainspike._kernel.model import NeuronModel
from brainspike._kernel.models import LIF, dc_generator, iaf_cond_exp, iaf_psc_alpha, spike_generator
from brainspike._kernel.network import Network
from brainspike._kernel.params import Init, Params, States
from brainspike._kernel.ports import Conductance, Current, Event, Graded, PortError, Ports
from brainspike._kernel.synapse import (
    Expon,
    SynapseModel,
    check_synapse_connection,
    plan_synapse_groups,
)
from brainspike._kernel.units import UnitMismatchError, from_mantissa, to_mantissa

DT = 0.1


class _CondCell(NeuronModel):
    """Test cell with an **unfiltered** conductance port: dV = dt * g (E - V) / C."""
    params = Params(C=100.0 * u.pF)
    states = States(V=Init(-70.0 * u.mV))
    inputs = Ports(syn=Conductance(u.nS, E_rev=0.0 * u.mV))
    outputs = Ports()
    integrator: ClassVar[Integrator] = Exact()

    def step(self, s, p, inp, ctx):
        dt = to_mantissa(ctx.dt, u.ms)
        V = to_mantissa(s.V, u.mV)
        g = to_mantissa(inp['syn'], u.nS)
        V = V + dt * g * (0.0 - V) / to_mantissa(p.C, u.pF)
        return s.replace(V=from_mantissa(V, u.mV)), {}


def rollout_states(plan, n_steps):
    """Per-step synapse states and population states (pure Program + scan)."""
    prog = plan.runner().program
    params = prog.read_params()

    def body(st, n):
        st, _ = prog.step(params, st, n, jnp.zeros((), jnp.uint32))
        return st, (st['syn'], st['pops'])

    f = brainstate.transform.jit(
        lambda st: brainstate.transform.scan(body, st, jnp.arange(n_steps, dtype=jnp.int32)))
    return f(prog.init_state())[1]


# --------------------------------------------------------------------------- Expon unit

class TestExponModel:
    def test_validation(self):
        with pytest.raises(ValueError, match='positive'):
            Expon(0.0 * u.ms)
        with pytest.raises(ValueError, match='positive'):
            Expon(-1.0 * u.ms)
        with pytest.raises(ValueError, match='scalar'):
            Expon([1.0, 2.0] * u.ms)
        with pytest.raises(UnitMismatchError):
            Expon(5.0 * u.mV)
        with pytest.raises(TypeError, match='bool'):
            Expon(5.0 * u.ms, merge='yes')

    def test_merge_key_and_repr(self):
        a, b = Expon(5.0 * u.ms), Expon(5.0 * u.ms)
        assert a.merge_key() == b.merge_key()
        assert a.merge_key() != Expon(6.0 * u.ms).merge_key()
        assert Expon(5.0 * u.ms, merge=False).merge_key() is None
        assert repr(a) == 'Expon(tau=5 ms)' == a.describe()
        assert a.alignment == 'post' and a.linear and a.filters

    def test_output_kind(self):
        syn = Expon(5.0 * u.ms)
        assert isinstance(syn.output_kind(Current(u.pA)), Current)
        assert syn.output_kind(Conductance(u.nS)).unit == u.nS
        with pytest.raises(PortError, match='current or a conductance'):
            syn.output_kind(Event())

    def test_psc_matches_closed_form(self):
        with jax.enable_x64(True):
            syn = Expon(3.0 * u.ms)
            ctx = StepContext.create(0.0 * u.ms, DT * u.ms)
            x = np.zeros(50)
            x[[3, 10]] = [2.0, -1.0]

            def body(s, xi):
                s, g = syn.step(s, jnp.asarray([xi]), ctx)
                return s, g[0]

            _, g = brainstate.transform.scan(body, syn.init_state(1, jnp.float64), jnp.asarray(x))
            n = np.arange(50)
            b = np.exp(-DT / 3.0)
            ref = np.where(n >= 3, 2.0 * b ** (n - 3), 0.0) - np.where(n >= 10, b ** (n - 10), 0.0)
            np.testing.assert_allclose(np.asarray(g), ref, rtol=1e-13, atol=1e-15)

    def test_matches_old_brainpy_expon(self):
        from brainpy.state import Expon as OldExpon
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 4)) * (rng.random((200, 4)) < 0.1)
        with jax.enable_x64(True):
            # Scoped, not ``environ.set``: a global set/reset here leaked
            # ``precision=32`` into the legacy tests (which set ``precision=64`` at
            # import) that ran after this one on the same xdist worker.
            with brainstate.environ.context(precision=64, dt=DT * u.ms):
                old = OldExpon(4, tau=4.0 * u.ms)
                old.init_state()

                def old_step(xi):
                    return old.update(xi * u.mS).to_decimal(u.mS)

                g_old = np.asarray(brainstate.transform.for_loop(old_step, jnp.asarray(x)))
            syn = Expon(4.0 * u.ms)
            ctx = StepContext.create(0.0 * u.ms, DT * u.ms)

            def body(s, xi):
                s, g = syn.step(s, xi, ctx)
                return s, g

            _, g_new = brainstate.transform.scan(body, syn.init_state(4, jnp.float64), jnp.asarray(x))
        np.testing.assert_allclose(np.asarray(g_new), g_old, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- connect checks

class TestConnectChecks:
    def test_single_owner(self):
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 1)
        c = net.add(iaf_cond_exp, 2)
        with pytest.raises(PortError, match=r'spike_generator\.spike -> iaf_cond_exp\.ex: the target '
                                            r"is a neuron-owned receptor port.*'g_ex'"):
            net.connect(sg, c, port='ex', weight=1.0 * u.nS, synapse=Expon(2.0 * u.ms))
        # any synapse model, even one without kinetics of its own
        class _NoKinetics(Expon):
            filters = False

        with pytest.raises(PortError, match='receptor'):
            net.connect(sg, c, port='ex', weight=1.0 * u.nS, synapse=_NoKinetics(2.0 * u.ms))
        with pytest.raises(PortError, match='receptor'):
            check_synapse_connection(Expon(2.0 * u.ms), Event(), Event(unit=u.nS, into='g_ex'),
                                     1.0 * u.nS)
        # without a synapse the receptor port takes events directly (neuron owns kinetics)
        net.connect(sg, c, port='ex', weight=1.0 * u.nS)

    def test_event_into_unfiltered_current_needs_synapse(self):
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 1)
        a = net.add(LIF, 2)
        with pytest.raises(PortError, match='cannot feed'):
            net.connect(sg, a, port='I', weight=1.0 * u.mA)
        p = net.connect(sg, a, port='I', weight=1.0 * u.mA, synapse=Expon(2.0 * u.ms))
        assert isinstance(p.synapse, Expon)

    def test_bad_targets_and_sources(self):
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 1)
        dc = net.add(dc_generator, 1)
        a = net.add(LIF, 2)
        cell = net.add(_CondCell, 2)
        syn = Expon(2.0 * u.ms)
        with pytest.raises(PortError, match='cannot feed'):
            net.connect(sg, a, port='delta', weight=1.0 * u.mV, synapse=syn)
        with pytest.raises(PortError, match='receives spike events'):
            net.connect(dc, a, port='I', weight=1.0, synapse=syn)
        with pytest.raises(UnitMismatchError):
            net.connect(sg, a, port='I', weight=1.0 * u.mV, synapse=syn)
        with pytest.raises(PortError, match='non-negative'):
            net.connect(sg, cell, port='syn', weight=-1.0 * u.nS, synapse=syn)
        with pytest.raises(TypeError, match='SynapseModel'):
            net.connect(sg, a, port='I', weight=1.0 * u.mA, synapse=object())

    def test_check_function_direct(self):
        syn = Expon(2.0 * u.ms)
        check_synapse_connection(syn, Event(), Current(u.pA), None)
        with pytest.raises(PortError, match='cannot feed'):
            check_synapse_connection(syn, Event(), Graded(u.mV), 1.0 * u.mV, where='x')

    def test_custom_synapse_targets(self):
        class _CurrentOnly(Expon):
            targets = (Current,)

        with pytest.raises(PortError, match='allowed'):
            check_synapse_connection(_CurrentOnly(1.0 * u.ms), Event(), Conductance(u.nS), None)


# --------------------------------------------------------------------------- merge pass

class TestMergePass:
    def test_plan_synapse_groups(self):
        e5, e2 = Expon(5.0 * u.ms), Expon(2.0 * u.ms)
        t = [(0, 0, 'I', e5, 'a'), (1, 0, 'I', Expon(5.0 * u.ms), 'b'), (2, 0, 'I', e2, 'c'),
             (3, 1, 'I', e5, 'd'), (4, 0, 'J', e5, 'e'), (5, 0, 'I', Expon(5.0 * u.ms, merge=False), 'f')]
        groups, route = plan_synapse_groups(t)
        assert [g.members for g in groups] == [[0, 1], [2], [3], [4], [5]]
        assert groups[0].merged and groups[0].key.startswith('merged[') and groups[1].key == 'c'
        assert route[(1, 0)] == 0 and route[(5, 0)] == 4
        groups, _ = plan_synapse_groups(t, merge=False)
        assert len(groups) == 6 and not any(g.merged for g in groups)

    @staticmethod
    def _net(tau, weights, n_post=3, extra=True):
        net = Network(dt=DT * u.ms, seed=3)
        sg = net.add(spike_generator, 3, name='sg', params=dict(spike_times=[
            [1.0, 2.0, 5.5] * u.ms, [1.0, 3.3] * u.ms, [0.4, 2.0, 2.1, 7.0] * u.ms]))
        a = net.add(LIF, n_post, name='A', params=dict(V_th=1e3 * u.mV))
        for k, w in enumerate(weights):
            net.connect(sg[k:k + 1], a, all_to_all(), port='I', weight=w * u.mA,
                        delay=(0.1 + 0.2 * k) * u.ms, synapse=Expon(tau * u.ms), name=f'p{k}')
        if extra:
            net.connect(sg, a, fixed_indegree(2), port='I', weight=0.5 * u.mA,
                        synapse=Expon(2 * tau * u.ms), name='other')
        return net, net.observe(a, 'V')

    def test_merged_equals_unmerged_to_rounding(self):
        with jax.enable_x64(True):
            net, rec = self._net(4.0, [1.0, 0.7, -0.3])
            merged = net.compile('float64')
            unmerged = net.compile('float64', merge=False)
            assert [len(g.members) for g in merged.synapses] == [3, 1]
            assert len(unmerged.synapses) == 4
            text = merged.describe(print_=False)
            assert 'Merges: 4 synapse target(s) -> 2 state(s)' in text
            assert 'A.I <- [p0, p1, p2] Expon(tau=4 ms): one state' in text
            assert '(merge pass disabled)' in unmerged.describe(print_=False)
            v1 = merged.runner().simulate(10.0 * u.ms)[rec]['V'].to_decimal(u.mV)
            v2 = unmerged.runner().simulate(10.0 * u.ms)[rec]['V'].to_decimal(u.mV)
        assert np.ptp(np.asarray(v1)) > 0.01
        np.testing.assert_allclose(np.asarray(v1), np.asarray(v2), rtol=1e-12, atol=1e-13)
        assert merged.memory()['synapses'] == 2 * 3 * 8
        assert unmerged.memory()['synapses'] == 4 * 3 * 8

    def test_merged_equals_unmerged_exactly(self):
        # decay factor exp(-dt/tau) = 1/2 and integer weights: every value is dyadic
        tau = DT / np.log(2.0)
        with jax.enable_x64(True):
            net, rec = self._net(tau, [1.0, 2.0, -3.0], extra=False)
            plans = [net.compile('float64'), net.compile('float64', merge=False)]
            assert [len(p.synapses) for p in plans] == [1, 3]
            syn = [rollout_states(p, 100) for p in plans]
            g_merged = np.asarray(next(iter(syn[0][0].values()))['g'])
            g_split = sum(np.asarray(v['g']) for v in syn[1][0].values())
            np.testing.assert_array_equal(g_merged, g_split)
            v = [np.asarray(s[1]['A']['V'].to_decimal(u.mV)) for s in syn]
        assert np.abs(g_merged).max() > 0
        np.testing.assert_array_equal(v[0], v[1])

    def test_opt_out_and_view_spanning_populations(self):
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 1, params=dict(spike_times=[1.0] * u.ms))
        a = net.add(LIF, 2, name='A')
        b = net.add(LIF, 3, name='B')
        net.connect(sg, a + b, port='I', weight=1.0 * u.mA, synapse=Expon(2.0 * u.ms), name='x')
        net.connect(sg, a, port='I', weight=1.0 * u.mA, synapse=Expon(2.0 * u.ms, merge=False),
                    name='y')
        plan = net.compile()
        assert [(g.pop, g.members) for g in plan.synapses] == [(1, [0]), (2, [0]), (1, [1])]
        assert 'none merged' in plan.describe(print_=False)
        res = plan.runner().simulate(3.0 * u.ms)
        assert plan.memory()['synapses'] == (2 + 3 + 2) * 4

    def test_no_synapses_describe(self):
        net = Network(dt=DT * u.ms)
        net.add(LIF, 1)
        assert 'Merges: none (no synapse models)' in net.compile().describe(print_=False)


# --------------------------------------------------------------------------- end to end

class TestEndToEnd:
    def test_psc_and_psp_closed_form(self):
        """Expon PSC in the state, LIF PSP = closed form of the exact discrete recurrence."""
        tau_s, tau_m, w, d, t_spk = 2.0, 10.0, 3.0, 5, 1.0
        with jax.enable_x64(True):
            net = Network(dt=DT * u.ms)
            sg = net.add(spike_generator, 1, params=dict(spike_times=[t_spk] * u.ms))
            a = net.add(LIF, 1, name='A', params=dict(V_th=1e3 * u.mV, tau=tau_m * u.ms))
            net.connect(sg, a, port='I', weight=w * u.mA, delay=d * DT * u.ms,
                        synapse=Expon(tau_s * u.ms), name='p')
            plan = net.compile('float64')
            syn, pops = rollout_states(plan, 400)
            g = np.asarray(syn['p@A']['g'])[:, 0]
            V = np.asarray(pops['A']['V'].to_decimal(u.mV))[:, 0]
        k0 = int(round(t_spk / DT)) - 1 + d   # emitted at stamp - 1, delivered d steps later
        n = np.arange(400)
        a_, b_ = np.exp(-DT / tau_m), np.exp(-DT / tau_s)
        np.testing.assert_allclose(g, np.where(n >= k0, w * b_ ** (n - k0), 0.0), rtol=1e-12, atol=1e-15)
        # LIF.I is latch=False (D-a): g of step m acts during step m (R = 1 ohm, mA -> mV)
        N = np.where(n >= k0, n - k0 + 1, 0)
        V_ref = (1 - a_) * w * (a_ ** N - b_ ** N) / (a_ - b_)
        np.testing.assert_allclose(V, V_ref, rtol=1e-11, atol=1e-13)
        # and close to the continuous-time PSP (O(dt) discretisation)
        t = N * DT  # V of step n is the state at its end, (n - k0 + 1) dt after the onset
        V_cont = np.where(t > 0, w * tau_s / (tau_s - tau_m) * (np.exp(-t / tau_s) - np.exp(-t / tau_m)), 0)
        assert np.abs(V - V_cont).max() < 0.05 * V_cont.max()

    def test_conductance_port(self):
        with jax.enable_x64(True):
            net = Network(dt=DT * u.ms)
            sg = net.add(spike_generator, 1, params=dict(spike_times=[0.5] * u.ms))
            c = net.add(_CondCell, 2, name='C')
            net.connect(sg, c, port='syn', weight=[2.0, 4.0] * u.nS, synapse=Expon(1.0 * u.ms),
                        name='p')
            syn, pops = rollout_states(net.compile('float64'), 50)
            g = np.asarray(syn['p@C']['g'])
            V = np.asarray(pops['C']['V'].to_decimal(u.mV))
        # conductance ports are not latched: the step uses this step's g
        V_ref = np.full(2, -70.0)
        for k in range(50):
            V_ref = V_ref + DT * g[k] * (0.0 - V_ref) / 100.0
            np.testing.assert_allclose(V[k], V_ref, rtol=1e-12)
        assert g.max() == pytest.approx(4.0)

    def test_batched_and_trainable_weights(self):
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 1, params=dict(spike_times=[0.5] * u.ms))
        a = net.add(LIF, 2, name='A', params=dict(V_th=1e3 * u.mV))
        net.connect(sg, a, port='I', weight=1.0 * u.mA, synapse=Expon(2.0 * u.ms), trainable=True)
        rec = net.observe(a, 'V')
        plan = net.compile()
        assert len(plan.states(brainstate.ParamState)) == 1
        r1 = plan.runner().simulate(5.0 * u.ms)[rec]['V']
        r2 = plan.runner(batch=2).simulate(5.0 * u.ms)[rec]['V']
        np.testing.assert_allclose(np.asarray(r2.to_decimal(u.mV))[0], np.asarray(r1.to_decimal(u.mV)))

    def test_etrace_network_and_online_learning(self):
        """Synapse states are HiddenStates of the braintrace module (T9 integration)."""
        import braintools
        import warnings
        from brainspike._kernel.learn import EtraceNetwork, ExternalInput, Online
        net = Network(dt=DT * u.ms, seed=2)
        x = net.add(ExternalInput, 3, name='x')
        h = net.add(LIF, 4, name='h', surrogate=braintools.surrogate.Sigmoid())
        net.connect(x, h, port='I', weight=4.0 * u.mA, synapse=Expon(2.0 * u.ms),
                    trainable=True, name='xh')
        net.connect(x, h, port='I', weight=2.0 * u.mA, synapse=Expon(2.0 * u.ms), name='xh2')
        rec = net.observe(h, 'V')
        plan = net.compile()
        assert len(plan.synapses) == 1 and plan.synapses[0].merged
        xs = (jax.random.uniform(jax.random.key(0), (60, 3)) < 0.3).astype(jnp.float32)
        prog = plan.runner().program
        params = prog.read_params()

        def step(st, inp):
            n, x_n = inp
            st, _ = prog.step(params, st, n, jnp.zeros((), jnp.uint32), ext={'x': {'x': x_n}})
            return st, st['pops']['h']['V'].to_decimal(u.mV)

        _, V = brainstate.transform.scan(step, prog.init_state(), (jnp.arange(60, dtype=jnp.int32), xs))
        en = EtraceNetwork(plan, inputs='x', outputs=[('h', 'V')])
        assert len(en.syn) == 1 and isinstance(next(iter(en.syn.values())), brainstate.HiddenState)
        out = brainstate.transform.for_loop(en.update, xs)
        np.testing.assert_allclose(np.asarray(out), np.asarray(V), rtol=1e-5, atol=1e-6)
        assert np.ptp(np.asarray(out)) > 0
        online = Online(plan.program(), 'eprop', inputs='x', outputs='h')
        model = online.model
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            grads = model.etrace_grad(xs, jnp.full((60, 4), 0.2),
                                      step_fn=lambda a, y: jnp.mean((model(a) - y) ** 2))
        g = [np.asarray(u.get_mantissa(v)) for v in grads.values()]
        assert len(g) == 1 and np.all(np.isfinite(g[0]))

    def test_synapse_into_iaf_psc_alpha_current_port(self):
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 1, params=dict(spike_times=[0.5] * u.ms))
        n = net.add(iaf_psc_alpha, 1, name='N')
        net.connect(sg, n, one_to_one(), port='I', weight=100.0 * u.pA, synapse=Expon(5.0 * u.ms))
        rec = net.observe(n, 'V_m')
        V = np.asarray(net.compile().runner().simulate(5.0 * u.ms)[rec]['V_m'].to_decimal(u.mV))
        assert V[-1, 0] > V[0, 0]
        assert issubclass(Expon, SynapseModel)
