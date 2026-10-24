# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of plasticity rules: protocol, NEST lazy timing with the delay split, both paths,
gradients through the reference path, network integration (NEST parity: ``validation/stdp_nest_test.py``)."""
from __future__ import annotations

from typing import ClassVar

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.connectivity import all_to_all, fixed_indegree, one_to_one
from brainspike._kernel.integrators import Exact, Integrator
from brainspike._kernel.model import NeuronModel
from brainspike._kernel.models import LIF, dc_generator, spike_generator
from brainspike._kernel.network import Network
from brainspike._kernel.params import Init, States
from brainspike._kernel.plasticity import (
    FastForm,
    PlasticContext,
    PlasticityRule,
    static_synapse,
    stdp_synapse,
)
from brainspike._kernel.ports import Event, PortError, Ports
from brainspike._kernel.units import UnitMismatchError

DT = 0.1


class _Silent(NeuronModel):
    """Post model without a spike output (the post trace has nothing to read)."""
    states = States(V=Init(0.0 * u.mV))
    inputs = Ports(delta=Event(unit=u.mV))
    outputs = Ports(out=Event())
    integrator: ClassVar[Integrator] = Exact()

    def step(self, s, p, inp, ctx):  # pragma: no cover - never run
        return s, {'out': s.V}


def pair_net(pre_times, post_times, *, rule, delay=1.0, weight=5.0, n=1, rule_kw=None,
             **connect_kw):
    """``n`` independent units: pre generator -> LIF post (forced by a driver), plastic."""
    net = Network(dt=DT * u.ms)
    pre = net.add(spike_generator, n, name='pre', params=dict(spike_times=pre_times))
    drv = net.add(spike_generator, n, name='drv',
                  params=dict(spike_times=post_times if post_times is not None else [] * u.ms))
    post = net.add(LIF, n, name='post', params=dict(V_th=1e4 * u.mV), spk_reset='hard')
    net.connect(drv, post, one_to_one(), port='delta', weight=1e6 * u.mV, delay=DT * u.ms)
    proj = net.connect(pre, post, one_to_one(), port='delta', weight=weight * u.mV,
                       delay=delay * u.ms, plasticity=rule, name='stdp', **connect_kw)
    rec = net.observe(post, 'V', 'spike')
    return net, proj, rec


def traces(plan, n_steps, params=None):
    """Per-step observable weights, working weights and post V (pure Program + scan)."""
    prog = plan.runner().program
    params = prog.read_params() if params is None else params
    pp = plan.projection('stdp')
    i_post = plan.population('post').pop.index

    def body(st, n):
        st, cur = prog.step(params, st, n, jnp.zeros((), jnp.uint32))
        ps = st['plastic']['stdp']
        return st, (pp.plastic.weight(ps), ps['edge']['weight'], st['pops']['post']['V'],
                    cur[i_post]['spike'])

    f = brainstate.transform.jit(
        lambda st: brainstate.transform.scan(body, st, jnp.arange(n_steps, dtype=jnp.int32)))
    return [np.asarray(u.get_mantissa(x)) for x in f(prog.init_state())[1]]


def nest_lazy_reference(pre_stamps, post_stamps, d, w, p, mu_plus=1.0, mu_minus=1.0):
    """Event-driven NEST ``stdp_synapse::send`` (whole delay dendritic) -> weights per send."""
    out, kplus, t_last, hist = [], 0.0, 0.0, []
    kminus, last_post = 0.0, None
    for q in post_stamps:  # ArchivingNode history (t, Kminus at t)
        kminus = (kminus * np.exp((last_post - q) / p['tau_minus']) if last_post is not None else 0.0) + 1.0
        last_post = q
        hist.append((q, kminus))
    for t in pre_stamps:
        for q, _ in hist:
            if t_last - d < q - 1e-9 and q <= t - d + 1e-9:
                k = kplus * np.exp((t_last - (q + d)) / p['tau_plus'])
                nw = w / p['Wmax'] + p['lambda_'] * (1 - w / p['Wmax']) ** mu_plus * k
                w = nw * p['Wmax'] if nw < 1 else p['Wmax']
        km = 0.0
        for q, kq in hist:
            if (t - d) - q > 1e-9:
                km = kq * np.exp((q - (t - d)) / p['tau_minus'])
        nw = w / p['Wmax'] - p['alpha'] * p['lambda_'] * (w / p['Wmax']) ** mu_minus * km
        w = nw * p['Wmax'] if nw > 0 else 0.0
        out.append(w)
        kplus = kplus * np.exp((t_last - t) / p['tau_plus']) + 1.0
        t_last = t
    return np.asarray(out)


P = dict(tau_plus=20.0, tau_minus=15.0, lambda_=0.1, alpha=1.1, Wmax=100.0)


def rule_of(**kw):
    p = {**P, **kw.pop('p', {})}
    return stdp_synapse(tau_plus=p['tau_plus'] * u.ms, tau_minus=p['tau_minus'] * u.ms,
                        lambda_=p['lambda_'], alpha=p['alpha'], Wmax=p['Wmax'], **kw)


# --------------------------------------------------------------------------- rule objects

class TestRules:
    def test_validation(self):
        with pytest.raises(ValueError, match='unknown trainable'):
            stdp_synapse(trainable=('beta',))
        with pytest.raises(ValueError, match='timing'):
            stdp_synapse(timing='soon')
        with pytest.raises(ValueError, match='path'):
            stdp_synapse(path='gpu')
        with pytest.raises(ValueError, match='axonal_delay'):
            stdp_synapse(axonal_delay=-1.0 * u.ms)
        with pytest.raises(UnitMismatchError):
            stdp_synapse(axonal_delay=1.0 * u.mV)
        with pytest.raises(UnitMismatchError):
            stdp_synapse(tau_plus=1.0)
        with pytest.raises(ValueError, match='tau_minus must be positive'):
            stdp_synapse(tau_minus=0.0 * u.ms)
        with pytest.raises(ValueError, match='non-zero'):
            stdp_synapse(Wmax=0.0)

    def test_attributes(self):
        r = stdp_synapse(trainable='lambda_')
        assert r.trainable == ('lambda_',) and r.timing == 'lazy' and r.path == 'auto'
        assert r.traces == {'pre': 'tau_plus', 'post': 'tau_minus'} and r.post_port == 'spike'
        assert 'stdp_synapse(tau_plus=20' in repr(r)
        assert r.edge_state_init() == {} and r.weight_interval(r.params) == (0.0, 100.0)
        assert stdp_synapse(Wmax=-5.0).weight_interval({'Wmax': -5.0}) == (-5.0, 0.0)
        assert static_synapse().is_static and not r.is_static
        assert PlasticityRule().fast_form({}) is None
        assert PlasticityRule().weight_interval({}) is None

    def test_fast_form(self):
        r = stdp_synapse(lambda_=0.1, alpha=2.0, Wmax=-4.0, mu_plus=0.0, mu_minus=0.0)
        ff = r.fast_form(r.resolve(u.mV))
        assert isinstance(ff, FastForm)
        assert float(ff.on_post) == pytest.approx(-0.4) and float(ff.on_pre) == pytest.approx(0.8)
        assert (float(ff.w_min), float(ff.w_max)) == (-4.0, 0.0)
        assert stdp_synapse(mu_plus=0.0).fast_form(r.params) is None

    def test_update_kernel_order_and_bounds(self):
        r = stdp_synapse(mu_plus=0.0, mu_minus=0.0)
        p = dict(lambda_=0.5, alpha=1.0, Wmax=10.0)
        one, zero = jnp.ones(3), jnp.zeros(3)
        ctx = PlasticContext(pre_spike=jnp.asarray([1., 0., 1.]), post_spike=jnp.asarray([1., 1., 0.]),
                             pre_trace=jnp.asarray([1., 3., 0.]), post_trace=jnp.asarray([2., 0., 1.]),
                             params=p, t=0.0, dt=0.1)
        st, w = r.update({'weight': jnp.asarray([5., 5., 5.])}, ctx)
        # edge 0: +5 (fac) then -10 (dep) -> 0; edge 1: +15 -> clipped at Wmax; edge 2: -5
        np.testing.assert_allclose(np.asarray(w), [0.0, 10.0, 0.0])
        np.testing.assert_allclose(np.asarray(st['weight']), np.asarray(w))
        r_static = static_synapse()
        st2, w2 = r_static.update({'weight': one}, ctx._replace(pre_spike=zero))
        assert st2['weight'] is one and w2 is one

    def test_resolve_quantity_wmax_and_sign(self):
        r = stdp_synapse(Wmax=0.05 * u.volt)
        assert r.resolve(u.mV)['Wmax'] == pytest.approx(50.0)
        with pytest.raises(UnitMismatchError):
            r.resolve(u.pA)
        with pytest.raises(ValueError, match='same sign'):
            stdp_synapse(Wmax=10.0).check_weights(np.asarray([1.0, -1.0]), {'Wmax': 10.0})
        stdp_synapse(Wmax=-10.0).check_weights(np.asarray([-1.0, -2.0]), {'Wmax': -10.0})
        with pytest.raises(ValueError, match='non-zero'):
            stdp_synapse(Wmax=0.0 * u.mV + 0.0 * u.mV)


# --------------------------------------------------------------------------- connect / compile

class TestConnectCompile:
    def test_connect_errors(self):
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 2)
        dc = net.add(dc_generator, 1)
        a = net.add(LIF, 2)
        s = net.add(_Silent, 2)
        kw = dict(port='delta', weight=1.0 * u.mV)
        with pytest.raises(TypeError, match='PlasticityRule'):
            net.connect(sg, a, plasticity=object(), **kw)
        with pytest.raises(NotImplementedError, match='L4'):
            net.connect(sg, a, plasticity=stdp_synapse(), trainable=True, **kw)
        with pytest.raises(ValueError, match='per edge'):
            net.connect(sg, a, plasticity=stdp_synapse(), storage='dense', **kw)
        with pytest.raises(PortError, match='spike events'):
            net.connect(dc, a, port='I', weight=1.0, plasticity=stdp_synapse())
        with pytest.raises(PortError, match='does not exist'):
            net.connect(sg, s, plasticity=stdp_synapse(), **kw)

        class _Graded(stdp_synapse):
            post_port = 'out'

        net.connect(sg, s, plasticity=_Graded(), **kw)  # an Event 'out' port is accepted

    def test_compile_errors(self):
        rule = stdp_synapse(axonal_delay=1.0 * u.ms)
        net, _, _ = pair_net([1.0] * u.ms, None, rule=rule, delay=1.0)
        with pytest.raises(ValueError, match='dendritic delay'):
            net.compile()
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 2)
        a = net.add(LIF, 1)
        net.connect(sg, a, port='delta', weight=1.0 * u.mV, delay=[1.0, 2.0] * u.ms,
                    plasticity=stdp_synapse())
        with pytest.raises(ValueError, match='one delay'):
            net.compile()
        net, _, _ = pair_net([1.0] * u.ms, None, rule=stdp_synapse(Wmax=-1.0))
        with pytest.raises(ValueError, match='same sign'):
            net.compile()
        net, _, _ = pair_net([1.0] * u.ms, None, rule=stdp_synapse(path='fast'))
        with pytest.raises(ValueError, match='add trace, clip'):
            net.compile()
        net, _, _ = pair_net([1.0] * u.ms, None, weight=500.0,
                             rule=stdp_synapse(mu_plus=0.0, mu_minus=0.0, path='fast'))
        with pytest.raises(ValueError, match='outside the clip interval'):
            net.compile()
        net, _, _ = pair_net([1.0] * u.ms, None,
                             rule=stdp_synapse(mu_plus=0.0, mu_minus=0.0, path='fast',
                                               trainable=('lambda_',)))
        with pytest.raises(ValueError, match='trainable'):
            net.compile()

    def test_path_resolution_and_describe(self):
        add = dict(mu_plus=0.0, mu_minus=0.0)
        cases = [(dict(), 5.0, 'reference'), (add, 5.0, 'fast'), (add, 500.0, 'reference'),
                 ({**add, 'path': 'reference'}, 5.0, 'reference'),
                 ({**add, 'trainable': ('alpha',)}, 5.0, 'reference')]
        for kw, w, path in cases:
            net, proj, _ = pair_net([1.0] * u.ms, None, rule=stdp_synapse(**kw), weight=w)
            plan = net.compile()
            assert plan.projection(proj).plastic.path == path, kw
        text = plan.describe(print_=False)
        assert 'plasticity=stdp_synapse(path=reference, timing=lazy, d_ax=0, d_den=10)' in text
        assert 'pre-side ring post.spike: 10 x 1' in text
        mem = plan.memory()
        assert mem['plastic'] == (2 * 1 + 2) * 4 and mem['total'] >= mem['plastic']
        assert len(plan.states(brainstate.ParamState)) == 1

    def test_static_synapse_is_an_ordinary_projection(self):
        def run(plasticity):
            net = Network(dt=DT * u.ms)
            sg = net.add(spike_generator, 3, params=dict(spike_times=[1.0, 2.0] * u.ms))
            a = net.add(LIF, 4, params=dict(V_th=1e3 * u.mV))
            p = net.connect(sg, a, fixed_indegree(2), port='delta', weight=0.5 * u.mV,
                            delay=0.5 * u.ms, plasticity=plasticity)
            rec = net.observe(a, 'V')
            plan = net.compile()
            assert plan.projection(p).plastic is None and p.plasticity is None
            return np.asarray(plan.runner().simulate(5.0 * u.ms)[rec]['V'].to_decimal(u.mV))

        np.testing.assert_array_equal(run(static_synapse()), run(None))

    def test_empty_projection(self):
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 0)
        a = net.add(LIF, 2)
        p = net.connect(sg, a, port='delta', weight=1.0 * u.mV, plasticity=stdp_synapse(),
                        source_port='spike')
        plan = net.compile()
        assert plan.projection(p).placement == 'none' and plan.projection(p).plastic is None
        plan.runner().simulate(1.0 * u.ms)


# --------------------------------------------------------------------------- timing

class TestTiming:
    def test_lazy_weights_match_event_driven_nest_send(self):
        """Weights at every send equal an event-driven re-implementation of NEST's send()."""
        rng = np.random.default_rng(1)
        pre = np.round(np.sort(rng.choice(np.arange(5.0, 300.0, 0.1), 25, replace=False)), 1)
        post = np.round(np.sort(rng.choice(np.arange(5.0, 300.0, 0.1), 30, replace=False)), 1)
        for delay in (0.1, 1.0, 3.7):
            with jax.enable_x64(True):
                net, _, _ = pair_net(pre * u.ms, np.round(post - DT, 6) * u.ms, rule=rule_of(),
                                     delay=delay, weight=20.0)
                W, Wwork, V, S = traces(net.compile('float64'), 3100)
            post_stamps = (np.nonzero(S[:, 0])[0] + 1) * DT
            np.testing.assert_allclose(post_stamps, post, atol=1e-9)
            ref = nest_lazy_reference(pre, post, delay, 20.0, P)
            steps = np.rint(pre / DT).astype(int) - 1
            np.testing.assert_allclose(W[steps, 0], ref, rtol=1e-10)
            # lazy: the observable weight only changes on pre sends
            changed = np.nonzero(np.diff(W[:, 0]))[0] + 1
            assert set(changed) <= set(steps)
            # the working weight already moved at post arrivals (eager facilitation)
            assert not np.array_equal(W[:, 0], Wwork[:, 0])

    def test_delivery_uses_the_weight_of_its_own_send(self):
        """Two pre spikes inside one delay: each EPSP carries the weight of its send."""
        delay = 3.0
        with jax.enable_x64(True):
            # post fires once in between, so the second send facilitates and depresses
            net, _, _ = pair_net([10.0, 11.0] * u.ms, [7.9] * u.ms, rule=rule_of(), delay=delay,
                                 weight=20.0)
            W, _, V, S = traces(net.compile('float64'), 200)
        s1, s2 = 99, 109                         # send steps (stamps 10.0, 11.0)
        assert W[s1, 0] != W[s2, 0]
        tau = 5.0  # LIF default
        jump1 = V[s1 + 30, 0] - V[s1 + 29, 0] * np.exp(-DT / tau)
        jump2 = V[s2 + 30, 0] - V[s2 + 29, 0] * np.exp(-DT / tau)
        np.testing.assert_allclose([jump1, jump2], [W[s1, 0], W[s2, 0]], rtol=1e-9)
        # nothing arrives before the delay
        np.testing.assert_allclose(V[s1 + 1:s1 + 30, 0], V[s1 + 1, 0] * np.exp(
            -DT / tau * np.arange(29)), rtol=1e-12, atol=1e-12)

    def test_axonal_split_shifts_the_synapse_time(self):
        """d = d_ax + d_den: same delivery time; the pairing uses synapse-local times."""
        pre = np.asarray([10.0, 40.0])
        post = np.asarray([14.0])
        res = {}
        for ax in (0.0, 0.8):
            with jax.enable_x64(True):
                net, proj, _ = pair_net(pre * u.ms, (post - DT) * u.ms,
                                        rule=rule_of(axonal_delay=ax * u.ms), delay=2.0,
                                        weight=20.0)
                plan = net.compile('float64')
                pl = plan.projection(proj).plastic
                assert (pl.d_ax, pl.d_den) == (int(round(ax / DT)), int(round((2.0 - ax) / DT)))
                res[ax] = traces(plan, 600)
        # pre at the synapse at t_pre + d_ax, post at t_post + d_den: the pairing interval
        # is (t_post + d_den) - (t_pre + d_ax); an axonal shift a equals a dendritic delay
        # of d - 2a in the event-driven reference (with the pre times shifted by a).
        for ax, (W, _, V, S) in res.items():
            d_den = 2.0 - ax
            ref = nest_lazy_reference(pre + ax, post, d_den, 20.0, P)
            steps = np.rint((pre + ax) / DT).astype(int) - 1
            np.testing.assert_allclose(W[steps, 0], ref, rtol=1e-10)
        # delivery time does not depend on the split
        dV = [np.nonzero(np.abs(np.diff(r[2][:, 0])) > 1.0)[0][:1] for r in res.values()]
        assert dV[0].tolist() == dV[1].tolist()

    def test_eager_timing(self):
        with jax.enable_x64(True):
            out = {}
            for timing in ('lazy', 'eager'):
                net, _, _ = pair_net([10.0, 30.0] * u.ms, [14.9, 20.0] * u.ms,
                                     rule=rule_of(timing=timing), delay=1.0, weight=20.0)
                plan = net.compile('float64')
                out[timing] = traces(plan, 400)
            assert plan.memory()['plastic'] == (1 + 2) * 8
        Wl, Wwl, Vl, _ = out['lazy']
        We, Wwe, Ve, _ = out['eager']
        np.testing.assert_array_equal(We, Wwe)          # eager observable == working weight
        np.testing.assert_array_equal(Wwl, Wwe)         # same working weights
        np.testing.assert_array_equal(Vl, Ve)           # identical deliveries
        assert not np.array_equal(Wl, We)

    def test_multiplicity_counts_once_for_the_trace(self):
        """A pre count of 2 delivers 2 w but increments K+ by one (NEST send)."""
        with jax.enable_x64(True):
            out = {}
            for times in ([10.0, 10.0] * u.ms, [10.0] * u.ms):
                net, _, _ = pair_net(times, [12.0] * u.ms, rule=rule_of(), delay=1.0, weight=20.0)
                out[len(times)] = traces(net.compile('float64'), 300)
        W2, _, V2, _ = out[2]
        W1, _, V1, _ = out[1]
        np.testing.assert_allclose(W2, W1, rtol=1e-12)
        s = 99 + 10   # delivery step of the send at stamp 10.0
        jump = lambda V: V[s, 0] - V[s - 1, 0] * np.exp(-DT / 5.0)
        np.testing.assert_allclose(jump(V2), 2 * jump(V1), rtol=1e-12)


# --------------------------------------------------------------------------- paths

def random_net(rule, n_pre=12, n_post=8, T=400.0, seed=0, weight=None, observe=False):
    rng = np.random.default_rng(seed)
    pre_t = [np.round(np.sort(rng.choice(np.arange(1.0, T, 0.1), 15, replace=False)), 1) * u.ms
             for _ in range(n_pre)]
    post_t = [np.round(np.sort(rng.choice(np.arange(1.0, T, 0.1), 12, replace=False)), 1) * u.ms
              for _ in range(n_post)]
    net = Network(dt=DT * u.ms, seed=seed)
    pre = net.add(spike_generator, n_pre, name='pre', params=dict(spike_times=pre_t))
    drv = net.add(spike_generator, n_post, name='drv', params=dict(spike_times=post_t))
    post = net.add(LIF, n_post, name='post', params=dict(V_th=1e4 * u.mV), spk_reset='hard')
    net.connect(drv, post, one_to_one(), port='delta', weight=1e6 * u.mV, delay=DT * u.ms)
    w = weight if weight is not None else u.math.asarray(
        rng.uniform(5.0, 60.0, size=n_post * 5)) * u.mV
    proj = net.connect(pre, post, fixed_indegree(5, allow_multapses=True), port='delta', weight=w,
                       delay=1.3 * u.ms, plasticity=rule, name='stdp')
    if observe:
        net.observe(post, 'V', name='rec')
    return net, proj


class TestPaths:
    # brainevent#200 (fixed in 0.3.0): csr_to_csc_index no longer warns under x64
    @pytest.mark.filterwarnings('error:.*cannot safely cast:FutureWarning')
    def test_fast_equals_reference(self):
        add = dict(mu_plus=0.0, mu_minus=0.0, p=dict(lambda_=0.02, Wmax=80.0))
        with jax.enable_x64(True):
            Ws = {}
            for path in ('fast', 'reference'):
                net, proj = random_net(rule_of(path=path, **{k: v for k, v in add.items()}))
                plan = net.compile('float64')
                assert plan.projection(proj).plastic.path == path
                Ws[path] = traces(plan, 4000)
        for a, b in zip(Ws['fast'], Ws['reference']):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-10)
        W = Ws['fast'][0]
        assert np.any(W[-1] != W[0]) and W.min() >= 0.0 and W.max() <= 80.0
        assert np.any(W == 80.0) or np.any(W == 0.0) or np.ptp(W) > 1.0

    # brainevent#200 (fixed in 0.3.0): csr_to_csc_index no longer warns under x64
    @pytest.mark.filterwarnings('error:.*cannot safely cast:FutureWarning')
    def test_fast_path_float32_and_batched(self):
        rule = rule_of(mu_plus=0.0, mu_minus=0.0)
        net, proj = random_net(rule, T=100.0)
        plan = net.compile()
        assert plan.projection(proj).plastic.path == 'fast'
        run = plan.runner(batch=2, rng='common')
        run.simulate(100.0 * u.ms)
        e0, e1 = run.edges(proj, trial=0), run.edges(proj, trial=1)
        np.testing.assert_array_equal(np.asarray(e0.weight.mantissa), np.asarray(e1.weight.mantissa))
        single = plan.runner()
        single.simulate(100.0 * u.ms)
        np.testing.assert_allclose(np.asarray(single.edges(proj).weight.mantissa),
                                   np.asarray(e0.weight.mantissa), rtol=1e-6)
        assert not np.array_equal(np.asarray(e0.weight.mantissa),
                                  np.asarray(plan.edges(proj).weight.mantissa))

    def test_runner_edges_order_and_non_plastic(self):
        net, proj = random_net(rule_of(timing='eager'), T=50.0)
        sg = net.add(spike_generator, 1)
        other = net.connect(sg, net.add(LIF, 1), port='delta', weight=1.0 * u.mV)
        plan = net.compile()
        run = plan.runner()
        e = run.edges(proj)
        assert e.same_edges(plan.edges(proj))
        np.testing.assert_allclose(np.asarray(e.weight.mantissa),
                                   np.asarray(plan.edges(proj).weight.mantissa))
        assert run.edges(other).n_edges == 1

    def test_concatenated_views(self):
        """(A + B) -> (P + Q) all-to-all equals the four separate projections edge by edge."""
        rng = np.random.default_rng(5)
        T = 150.0

        def trains(n):
            return [np.round(np.sort(rng.choice(np.arange(1.0, T, 0.1), 8, replace=False)), 1) * u.ms
                    for _ in range(n)]

        tA, tB, dP, dQ = trains(2), trains(3), trains(2), trains(1)

        def build(split):
            net = Network(dt=DT * u.ms)
            A = net.add(spike_generator, 2, name='A', params=dict(spike_times=tA))
            B = net.add(spike_generator, 3, name='B', params=dict(spike_times=tB))
            P_ = net.add(LIF, 2, name='P', params=dict(V_th=1e4 * u.mV), spk_reset='hard')
            Q = net.add(LIF, 1, name='Q', params=dict(V_th=1e4 * u.mV), spk_reset='hard')
            dp = net.add(spike_generator, 2, params=dict(spike_times=dP))
            dq = net.add(spike_generator, 1, params=dict(spike_times=dQ))
            net.connect(dp, P_, one_to_one(), port='delta', weight=1e6 * u.mV)
            net.connect(dq, Q, one_to_one(), port='delta', weight=1e6 * u.mV)
            kw = dict(port='delta', weight=30.0 * u.mV, delay=0.7 * u.ms, plasticity=rule_of())
            projs = ([net.connect(x, y, all_to_all(), **kw) for x in (A, B) for y in (P_, Q)]
                     if split else [net.connect(A + B, P_ + Q, all_to_all(), **kw)])
            run = net.compile('float64').runner()
            run.simulate(T * u.ms)
            out = {}
            for p in projs:
                c = run.edges(p).get_connections()
                for s, t, w in zip(np.asarray(c.source), np.asarray(c.target),
                                   np.asarray(u.get_mantissa(c.weight))):
                    out[(int(s), int(t))] = float(w)
            return out

        with jax.enable_x64(True):
            joint, split = build(False), build(True)
        assert joint.keys() == split.keys() and len(joint) == 15
        for k in joint:
            assert joint[k] == pytest.approx(split[k], rel=1e-12), k
        assert len({round(v, 9) for v in joint.values()}) > 3


# --------------------------------------------------------------------------- gradients

class TestGradients:
    def test_grad_wrt_rule_params_matches_finite_differences(self):
        rule = rule_of(trainable=('lambda_', 'alpha', 'tau_plus'))
        with jax.enable_x64(True):
            net, proj = random_net(rule, n_pre=4, n_post=3, T=80.0, seed=2)
            plan = net.compile('float64')
            assert plan.projection(proj).plastic.path == 'reference'
            pstates = plan.states(brainstate.ParamState)
            assert len(pstates) == 3
            prog = plan.runner().program
            params = prog.read_params()
            st0 = prog.init_state()

            def loss(theta):
                p = {**params, 'plasticity': {'stdp': {**params['plasticity']['stdp'], **theta}}}
                st, _ = prog.rollout(p, st0, 0, jnp.zeros((), jnp.uint32), 800)
                w = plan.projection(proj).plastic.weight(st['plastic']['stdp'])
                # (the post V is left out: its spikes are forced, and the surrogate gradient
                # of a forced reset is not the derivative finite differences see)
                return jnp.sum(w ** 2) / 100.0

            theta = {k: params['plasticity']['stdp'][k] for k in ('lambda_', 'alpha', 'tau_plus')}
            g = jax.jit(jax.grad(loss))(theta)
            f = jax.jit(loss)
            for k, v in theta.items():
                h = 1e-6 * max(1.0, abs(float(v)))
                fd = (f({**theta, k: v + h}) - f({**theta, k: v - h})) / (2 * h)
                assert float(g[k]) != 0.0, k
                np.testing.assert_allclose(float(g[k]), float(fd), rtol=1e-5, err_msg=k)


# --------------------------------------------------------------------------- learners (T9)

class TestLearners:
    def test_online_learning_rejects_plastic_projections(self):
        from brainspike._kernel.learn import EtraceError, Online
        net, proj = random_net(rule_of(), T=20.0)
        sg = net.add(spike_generator, 2, params=dict(spike_times=[1.0] * u.ms))
        net.connect(sg, net.add(LIF, 2), port='delta', weight=0.5 * u.mV, trainable=True)
        with pytest.raises(EtraceError, match='plasticity'):
            Online(net.compile().program(), 'eprop')

    def test_static_synapse_is_not_rejected_by_the_online_check(self):
        from brainspike._kernel.learn.etrace import check_plan
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 2)
        a = net.add(LIF, 2)
        net.connect(sg, a, port='delta', weight=0.5 * u.mV, trainable=True,
                    plasticity=static_synapse())
        check_plan(net.compile())

    def test_bptt_learns_rule_parameters(self):
        from brainspike._kernel.learn import BPTT
        from brainspike._kernel.learn.bptt import trainable_states
        net, proj = random_net(rule_of(trainable=('lambda_',)), n_pre=4, n_post=3, T=60.0,
                               observe=True)
        plan = net.compile()
        assert sorted(trainable_states(plan)) == ['plasticity/stdp/lambda_']
        bptt = BPTT(plan.program(), lambda r, y: jnp.mean(r['rec']['V']))
        loss, grads = bptt.grad(duration=60.0 * u.ms)
        assert set(grads) == {'plasticity/stdp/lambda_'}
        assert np.isfinite(float(grads['plasticity/stdp/lambda_']))
        assert float(grads['plasticity/stdp/lambda_']) != 0.0
