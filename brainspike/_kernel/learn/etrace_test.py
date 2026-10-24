# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the braintrace compatibility checker and the online learner (gate G7)."""
from __future__ import annotations

import warnings

import braintools
import brainstate
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.connectivity import fixed_indegree
from brainspike._kernel.learn import (
    EtraceError,
    EtraceNetwork,
    ExternalInput,
    Online,
    check_etrace,
    check_learner,
    check_plan,
)
from brainspike._kernel.learn.etrace import EtraceReport
from brainspike._kernel.model import Capabilities
from brainspike._kernel.models import LIF, iaf_psc_alpha, poisson_generator
from brainspike._kernel.network import Network

DT = 0.1 * u.ms
SIG = braintools.surrogate.Sigmoid()
N_IN, N_H, N_O = 6, 12, 4


def lif_net(storage='dense', delay_steps=1, trainable=True, scale=1.0):
    """x -> h (recurrent) -> o, the G7 network (``scale`` multiplies every weight)."""
    net = Network(dt=DT)
    x = net.add(ExternalInput, N_IN, name='x')
    h = net.add(LIF, N_H, params=dict(tau=10.0 * u.ms), name='h', surrogate=SIG)
    o = net.add(LIF, N_O, params=dict(tau=10.0 * u.ms), name='o', surrogate=SIG)
    net.connect(x, h, port='delta', weight=0.4 * scale * u.mV, delay=delay_steps * DT,
                storage=storage, trainable=trainable, name='xh')
    net.connect(h, h, fixed_indegree(4), port='delta', weight=0.1 * scale * u.mV,
                delay=delay_steps * DT, storage=storage, trainable=trainable, name='hh')
    net.connect(h, o, fixed_indegree(6), port='delta', weight=0.3 * scale * u.mV,
                delay=0.0 * u.ms, storage=storage, trainable=trainable, name='ho')
    net.observe(o, 'spike', name='rec')
    return net


def _data(n_steps=80, seed=0):
    xs = (jax.random.uniform(jax.random.key(seed), (n_steps, N_IN)) < 0.3).astype(jnp.float32)
    ys = jnp.full((n_steps, N_H + N_O), 0.2)
    return xs, ys


# ---------------------------------------------------------------------------
# G7
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('algorithm', ['eprop', 'd_rtrl'])
@pytest.mark.parametrize('delay_steps', [1, 2])
def test_g7_compile_and_nonzero_gradients(storage, algorithm, delay_steps):
    plan = lif_net(storage, delay_steps).compile()
    online = Online(plan.program(), algorithm, inputs='x', outputs=['h', 'o'])
    assert online.plan is plan and online.program.etrace
    rep = online.report
    assert isinstance(rep, EtraceReport)
    assert len(rep.relations) == 3 and len(rep.weights) == 3
    prim = {r[0] for r in rep.relations}
    assert prim == ({'etp_mv'} if storage == 'dense' else {'etp_sp_mv'})
    model = online.model
    xs, ys = _data()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grads = model.etrace_grad(xs, ys, step_fn=lambda x, y: jnp.mean((model(x) - y) ** 2))
    leaves = {k: np.asarray(u.get_mantissa(v)) for k, v in grads.items()}
    assert len(leaves) == 3
    for k, g in leaves.items():
        assert np.all(np.isfinite(g)), k
        assert np.abs(g).sum() > 0, k  # every weight receives a gradient


def test_braintrace_compile_directly_on_network():
    """``braintrace.compile(net, ...)`` on the module of a Network-built plan."""
    plan = lif_net('csr').compile()
    net = EtraceNetwork(plan, inputs='x', outputs='o')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = braintrace.compile(net, 'eprop', net.example_input())
        xs, _ = _data(40)
        out = brainstate.transform.for_loop(model, xs)
    assert out.shape == (40, N_O)


@pytest.mark.parametrize('model,port,w', [(LIF, 'delta', 0.2 * u.mV), (iaf_psc_alpha, 'ex', 50.0 * u.pA)])
def test_online_with_per_edge_delay_scatter(model, port, w):
    """A non-trainable per-edge-delay (DelayScatter) projection next to a trainable one.

    braintrace 0.2.7 reported OUTPUT_BYPASS_DROPPED on this network while the #184
    workaround (``new + 0 * old``) was applied: consumers read ``new`` and braintrace
    saw ``new + 0 * old`` as the hidden value. Without it the learner compiles cleanly.
    """
    net = Network(dt=DT, seed=5)
    x = net.add(ExternalInput, N_IN, name='x')
    h = net.add(model, 10, name='h', surrogate=SIG)
    net.connect(x, h, port=port, weight=w, trainable=True, name='xh')
    net.connect(h, h, fixed_indegree(3), port=port, weight=w,
                delay=jnp.array([0.1, 0.3, 0.2] * 10) * u.ms, name='hh')
    plan = net.compile()
    assert type(plan.projection('hh').groups[0].storage).__name__ == 'DelayScatter'
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)  # no braintrace WARNING diagnostic either
        online = Online(plan.program(), 'd_rtrl', inputs='x', outputs='h', loss=_mse)
    kinds = {d.kind for d in online.model.report.diagnostics}
    assert braintrace.DiagnosticKind.OUTPUT_BYPASS_DROPPED not in kinds
    loss, grads = online.grad(None, _data(20)[0], jnp.full((20, 10), 0.2))
    assert np.isfinite(float(loss)) and set(grads) == {'weights/xh/0'}


def test_etrace_network_matches_runner():
    """The one-step module reproduces the runner's rollout (same step contract)."""
    net = Network(dt=DT, seed=5)
    noise = net.add(poisson_generator, 8, params=dict(rate=800.0 * u.Hz), name='noise')
    h = net.add(LIF, 10, name='h')
    net.connect(noise, h, port='delta', weight=0.4 * u.mV, trainable=True, name='nh')
    net.connect(h, h, fixed_indegree(3), port='delta', weight=0.2 * u.mV,
                delay=jnp.array([0.1, 0.3, 0.2] * 10) * u.ms, trainable=True, name='hh')
    rec = net.observe(h, 'spike', 'V')
    plan = net.compile()
    res = plan.runner().simulate(10.0 * u.ms)
    assert plan.projection('hh').placement == 'post'  # per-edge delays: arrival ring
    en = EtraceNetwork(plan, outputs=[('h', 'spike'), ('h', 'V')])
    assert en.example_input() is None
    out = brainstate.transform.for_loop(lambda i: en.update(), jnp.arange(100))
    np.testing.assert_array_equal(np.asarray(out[:, :10]), np.asarray(res[rec]['spike']))
    np.testing.assert_allclose(np.asarray(out[:, 10:]), np.asarray(res[rec]['V'].to_decimal(u.mV)),
                               rtol=1e-6)
    assert int(en.n.value) == 100
    assert float(np.sum(out[:, :10])) > 0


def test_multiple_inputs_and_port_pairs():
    net = Network(dt=DT)
    a = net.add(ExternalInput, 2, name='a')
    b = net.add(ExternalInput, 3, name='b')
    h = net.add(LIF, 4, name='h', surrogate=SIG)
    net.connect(a, h, port='delta', weight=0.3 * u.mV, delay=0.0 * u.ms, trainable=True)
    net.connect(b, h, port='delta', weight=0.3 * u.mV, delay=0.0 * u.ms, trainable=True)
    plan = net.compile()
    en = EtraceNetwork(plan, inputs=['a', ('b', 'x')], outputs='h')
    x0 = en.example_input()
    assert isinstance(x0, tuple) and [v.shape for v in x0] == [(2,), (3,)]
    out = en.update((jnp.ones(2), jnp.ones(3) * 2))
    np.testing.assert_array_equal(np.asarray(out), np.ones(4))  # 0.6 + 1.8 mV > 1 mV
    with pytest.raises(ValueError, match='expected 2 inputs'):
        en.update((jnp.ones(2),))
    report = check_etrace(en, *(x0,))
    assert len(report.relations) == 2


# ---------------------------------------------------------------------------
# the checker rejects non-compliant models
# ---------------------------------------------------------------------------

class _Base(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = brainstate.ParamState(jnp.ones((3, 4)) * 0.5)
        self.f = braintools.surrogate.Sigmoid()

    def init_state(self, *args, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(4))


class _Good(_Base):
    def update(self, x):
        V = 0.9 * self.V.value + braintrace.matmul(x, self.W.value)
        self.V.value = V - jax.lax.stop_gradient(self.f(V - 1.0))
        return self.V.value


class _RawMatmul(_Base):
    def update(self, x):
        self.V.value = 0.9 * self.V.value + x @ self.W.value
        return self.V.value


class _WritesWeight(_Good):
    def update(self, x):
        out = super().update(x)
        self.W.value = self.W.value + 0.01 * jnp.outer(x, out)  # local plasticity
        return out


class _InWhile(_Base):
    def update(self, x):
        I = jax.lax.while_loop(lambda c: c[0] < 2,
                               lambda c: (c[0] + 1, c[1] + braintrace.matmul(x, self.W.value)),
                               (0, jnp.zeros(4)))[1]
        self.V.value = 0.9 * self.V.value + I
        return self.V.value


class _OutputNotHidden(_Base):
    def update(self, x):
        V = 0.9 * self.V.value + braintrace.matmul(x, self.W.value)
        s = self.f(V - 1.0)
        self.V.value = V - jax.lax.stop_gradient(s)
        return s  # computed from the pre-reset value, not from the HiddenState


class _InputOnlyHidden(_Good):
    def init_state(self, *args, **kwargs):
        super().init_state()
        self.latch = brainstate.HiddenState(jnp.zeros(3))

    def update(self, x):
        self.latch.value = x * 2.0  # depends only on the input
        return super().update(x)


class _RawToHiddenAndOutput(_Good):
    """A raw weight that reaches both a hidden state and the output: not a pure readout."""

    def __init__(self):
        super().__init__()
        self.U = brainstate.ParamState(jnp.ones((3, 4)) * 0.1)

    def update(self, x):
        V = 0.9 * self.V.value + braintrace.matmul(x, self.W.value) + x @ self.U.value
        self.V.value = V
        return self.V.value + x @ self.U.value


class _WeightIntoShortTerm(_Good):
    """A weight that only feeds a ``ShortTermState`` (like a per-edge-delay arrival ring)."""

    def __init__(self):
        super().__init__()
        self.U = brainstate.ParamState(jnp.ones((3, 4)))

    def init_state(self, *args, **kwargs):
        super().init_state()
        self.ring = brainstate.ShortTermState(jnp.zeros(4))

    def update(self, x):
        self.ring.value = x @ self.U.value
        return super().update(x)


class _Readout(_Good):
    def __init__(self):
        super().__init__()
        self.w_out = brainstate.ParamState(jnp.ones((4, 2)) * 0.3)

    def update(self, x):
        return jnp.tanh(super().update(x)) @ self.w_out.value


class _BypassThroughWhile(_Base):
    def update(self, x):
        V = 0.9 * self.V.value + braintrace.matmul(x, self.W.value)
        s = self.f(V - 1.0)
        self.V.value = V - jax.lax.stop_gradient(s)
        # weight-free while on the (pre-reset) output path: braintrace drops that gradient
        return jax.lax.while_loop(lambda c: c[0] < 2, lambda c: (c[0] + 1, 0.5 * c[1]), (0, s))[1]


@pytest.mark.parametrize('cls,match', [
    (_RawMatmul, 'not applied through a braintrace primitive'),
    (_RawToHiddenAndOutput, r"not applied through a braintrace primitive .*\['U'\]"),
    (_WritesWeight, 'written inside the step'),
    (_InWhile, 'inside a while_loop body'),
])
def test_checker_rejects_non_compliant(cls, match):
    m = cls()
    brainstate.nn.init_all_states(m)
    with pytest.raises(EtraceError, match=match):
        check_etrace(m, jnp.ones(3))


def test_checker_rejects_weight_reaching_no_hidden_state_or_output():
    m = _WeightIntoShortTerm()
    brainstate.nn.init_all_states(m)
    with pytest.raises(EtraceError, match=r"\['U'\]"):
        check_etrace(m, jnp.ones(3))


def test_rule4_message_points_to_supported_patterns():
    m = _WritesWeight()
    brainstate.nn.init_all_states(m)
    with pytest.raises(EtraceError, match='LongTermState'):
        check_etrace(m, jnp.ones(3))


@pytest.mark.parametrize('cls,bypass,readouts', [
    (_OutputNotHidden, [0], []),      # spike read from the pre-reset value (braintrace#185)
    (_InputOnlyHidden, [], []),       # input-only hidden state (braintrace#184)
    (_Readout, [0], ['w_out']),       # readout weight applied in the step
    (_BypassThroughWhile, [0], []),   # accepted here, rejected by check_learner
])
def test_checker_accepts_braintrace_027_cases(cls, bypass, readouts):
    m = cls()
    brainstate.nn.init_all_states(m)
    rep = check_etrace(m, jnp.ones(3))
    assert rep.bypass == bypass and rep.readouts == readouts
    assert ('etp_mv', 'W', ['V']) in rep.relations


@pytest.mark.parametrize('cls', [_OutputNotHidden, _InputOnlyHidden, _Readout, _Good])
@pytest.mark.parametrize('algorithm', ['d_rtrl', 'eprop'])
def test_braintrace_027_cases_match_bptt(cls, algorithm):
    """Models the 0.2.6 checker rejected compile, and their online gradient is BPTT."""
    with jax.enable_x64(True):
        xs = jax.random.uniform(jax.random.key(3), (10, 3), dtype=jnp.float64) * 2.0
        m = cls()
        brainstate.nn.init_all_states(m)
        n_out = jnp.shape(m.update(xs[0]))[-1]
        ys = jnp.full((10, n_out), 0.3)

        def fresh():
            m = cls()
            brainstate.nn.init_all_states(m)
            return m, {k: v for k, v in m.states(brainstate.ParamState).items()}

        m, params = fresh()
        ref = brainstate.transform.grad(lambda: brainstate.transform.for_loop(
            lambda x, y: jnp.mean((m(x) - y) ** 2), xs, ys).mean(), grad_states=params)()
        m, params = fresh()
        rep = check_etrace(m, xs[0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            learner = braintrace.compile(m, algorithm, xs[0])
            check_learner(learner, rep)
            g = learner.etrace_grad(xs, ys, step_fn=lambda x, y: jnp.mean((learner(x) - y) ** 2),
                                    weights=params)
        for k in ref:
            np.testing.assert_allclose(np.asarray(g[k]), np.asarray(ref[k]), rtol=1e-10, atol=1e-14)
            assert np.abs(np.asarray(ref[k])).max() > 0


def test_check_learner_rejects_dropped_bypass():
    m = _BypassThroughWhile()
    brainstate.nn.init_all_states(m)
    rep = check_etrace(m, jnp.ones(3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        learner = braintrace.compile(m, 'd_rtrl', jnp.ones(3))
    with pytest.raises(EtraceError, match='OUTPUT_BYPASS_DROPPED'):
        check_learner(learner, rep)
    with pytest.raises(EtraceError, match='OUTPUT_BYPASS_DROPPED'):
        check_learner(learner)  # the diagnostic alone suffices


@pytest.mark.parametrize('cls', [_OutputNotHidden, _Readout])
def test_check_learner_rejects_bypass_under_modulatory_signal(cls):
    m = cls()
    brainstate.nn.init_all_states(m)
    rep = check_etrace(m, jnp.ones(3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        learner = braintrace.compile(m, 'three_factor', jnp.ones(3))
    with pytest.raises(EtraceError, match='modulatory'):
        check_learner(learner, rep)
    check_learner(braintrace.compile(m, 'd_rtrl', jnp.ones(3)), rep)  # bypass rewrite: fine


def test_check_learner_accepts_modulatory_without_bypass():
    m = _Good()
    brainstate.nn.init_all_states(m)
    rep = check_etrace(m, jnp.ones(3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        check_learner(braintrace.compile(m, 'three_factor', jnp.ones(3)), rep)


def test_checker_accepts_compliant():
    m = _Good()
    brainstate.nn.init_all_states(m)
    rep = check_etrace(m, jnp.ones(3))
    assert rep.weights == ['W'] and rep.hidden == ['V']
    assert rep.relations == [('etp_mv', 'W', ['V'])]


class _NoEtrace(LIF):
    capabilities = Capabilities(etrace=False)


def test_plan_level_rejections():
    with pytest.raises(EtraceError, match='no trainable weights'):
        check_plan(lif_net(trainable=False).compile())
    net = lif_net()
    net.add(LIF, 2, name='z', trainable=('tau',))
    with pytest.raises(EtraceError, match='trainable neuron parameters'):
        check_plan(net.compile())
    net = Network(dt=DT)
    x = net.add(ExternalInput, 2, name='x')
    z = net.add(_NoEtrace, 2, name='z')
    net.connect(x, z, port='delta', weight=0.3 * u.mV, trainable=True)
    with pytest.raises(EtraceError, match='not etrace-compatible'):
        check_plan(net.compile())
    plan = lif_net().compile()
    object.__setattr__(plan.projections[1].proj, 'plasticity', object())  # T6 attaches rules here
    with pytest.raises(EtraceError, match='local plasticity'):
        check_plan(plan)
    with pytest.raises(EtraceError, match='local plasticity'):
        Online(plan.program(), 'eprop', inputs='x', outputs='o')


def test_network_and_learner_argument_errors():
    plan = lif_net().compile()
    with pytest.raises(ValueError, match='at least one output'):
        EtraceNetwork(plan, inputs='x')
    with pytest.raises(ValueError, match="has ports"):
        EtraceNetwork(plan, inputs='h', outputs='o')
    with pytest.raises(ValueError, match='no state'):
        EtraceNetwork(plan, inputs='x', outputs=('o', 'nope'))
    net = Network(dt=DT)
    x = net.add(ExternalInput, 2, name='x')
    a = net.add(iaf_psc_alpha, 2, name='a')
    net.connect(x, a, port='ex', weight=10.0 * u.pA, trainable=True)
    with pytest.raises(EtraceError, match='not a HiddenState'):
        EtraceNetwork(net.compile(), inputs='x', outputs=('a', 'refractory_count'))
    with pytest.raises(ValueError, match='no spike rule'):
        EtraceNetwork(plan, inputs='x', outputs='x')
    with pytest.raises(TypeError, match='Program'):
        Online(lif_net().compile(), inputs='x', outputs='o')  # a plan is not a program
    with pytest.raises(TypeError, match='callable'):
        Online(lif_net().compile().program(), inputs='x', outputs='o', loss=3.0)
    plan = lif_net().compile()
    online = Online(plan.program(), inputs='x', outputs='o')
    xs, ys = _data(10)
    with pytest.raises(RuntimeError, match='without a loss'):
        online.grad(None, xs, ys[:, :N_O])
    with pytest.raises(RuntimeError, match='without an optimizer'):
        online.run(None, xs, ys[:, :N_O])
    with pytest.raises(ValueError, match='unbatched'):
        online.load(plan.program().init_state(2))
    with pytest.raises(ValueError, match='not a state'):
        online.load({'pops': {}})


def test_checker_rejects_k0_network_with_raw_weights():
    """A K0 network whose weights bypass the braintrace primitives is rejected."""
    en = EtraceNetwork(lif_net('dense').compile(), inputs='x', outputs='o')
    en.program.etrace = False  # storages' plain forward: braintrace cannot see the weights
    with pytest.raises(EtraceError, match='not applied through a braintrace primitive'):
        check_etrace(en, en.example_input())


# ---------------------------------------------------------------------------
# K1a-5: relaxed rule 2 (static scan / fori_loop allowed, while rejected)
# ---------------------------------------------------------------------------

_K_SUB = 3


class _Substeps(_Base):
    """``V <- 0.8 V + 0.1 x @ W`` applied ``_K_SUB`` times, the weight inside the loop."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def update(self, x):
        step = lambda v: 0.8 * v + 0.1 * braintrace.matmul(x, self.W.value)
        V = self.V.value
        if self.mode == 'python':
            for _ in range(_K_SUB):
                V = step(V)
        elif self.mode == 'fori':
            V = jax.lax.fori_loop(0, _K_SUB, lambda i, v: step(v), V)
        else:
            V = brainstate.transform.scan(lambda v, _: (step(v), None), V, None, length=_K_SUB)[0]
        self.V.value = V
        return jnp.tanh(self.V.value)


def _bptt_reference(W, xs, ys):
    """Exact gradient of the mean per-step loss by reverse mode through the unrolled rollout."""
    def loss(W):
        def body(V, xy):
            x, y = xy
            for _ in range(_K_SUB):
                V = 0.8 * V + 0.1 * (x @ W)
            return V, jnp.mean((jnp.tanh(V) - y) ** 2)
        _, ls = jax.lax.scan(body, jnp.zeros(4), (xs, ys))
        return jnp.mean(ls)
    return jax.grad(loss)(W)


@pytest.mark.parametrize('algorithm', ['d_rtrl', 'eprop'])
def test_checker_accepts_braintrace_weight_in_static_loops(algorithm):
    xs = jax.random.uniform(jax.random.key(1), (12, 3))
    ys = jnp.full((12, 4), 0.3)
    grads = {}
    for mode in ('python', 'fori', 'scan'):
        m = _Substeps(mode)
        brainstate.nn.init_all_states(m)
        rep = check_etrace(m, jnp.ones(3))
        assert rep.weights == ['W']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            learner = braintrace.compile(m, algorithm, xs[0])
            g = learner.etrace_grad(xs, ys, step_fn=lambda x, y: jnp.mean((learner(x) - y) ** 2))
        grads[mode] = np.asarray(next(iter(g.values())))
    np.testing.assert_allclose(grads['fori'], grads['python'], rtol=1e-6)
    np.testing.assert_allclose(grads['scan'], grads['python'], rtol=1e-6)
    # diagonal hidden dynamics: D-RTRL / e-prop are exact RTRL here, i.e. BPTT
    ref = np.asarray(_bptt_reference(jnp.ones((3, 4)) * 0.5, xs, ys))
    np.testing.assert_allclose(grads['scan'], ref, rtol=1e-5)
    assert np.abs(grads['scan']).sum() > 0


# ---------------------------------------------------------------------------
# K1a-5: Online over (Program, state0)
# ---------------------------------------------------------------------------

def _mse(out, y):
    return jnp.mean((out - y) ** 2)


def test_online_grad_and_run_from_any_state():
    plan = lif_net('csr', 1, scale=0.3).compile()
    prog = plan.program()
    online = Online(prog, 'd_rtrl', inputs='x', outputs=['h', 'o'], loss=_mse,
                    optimizer=braintools.optim.SGD(lr=0.5))
    assert set(online.params) == {'weights/xh/0', 'weights/hh/0', 'weights/ho/0'}
    xs, ys = _data(60)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # warm up the network itself (40 steps of input), then learn from that state
        warm_x = _data(40, seed=3)[0]
        online.load(None)
        brainstate.transform.for_loop(online.network.update, warm_x)
        state1, n1 = online.state()
        assert int(n1) == 40
        l_cold, g_cold = online.grad(None, xs, ys)
        l_warm, g_warm = online.grad(state1, xs, ys, n0=40)
        # same episode twice: identical (traces are reset on load)
        l_again, g_again = online.grad(state1, xs, ys, n0=40)
    assert float(l_again) == float(l_warm)
    for k in g_warm:
        np.testing.assert_array_equal(np.asarray(u.get_mantissa(g_again[k])),
                                      np.asarray(u.get_mantissa(g_warm[k])))
        assert np.all(np.isfinite(np.asarray(u.get_mantissa(g_warm[k]))))
    assert any(not np.allclose(u.get_mantissa(g_warm[k]), u.get_mantissa(g_cold[k]))
               for k in g_warm)
    assert any(np.abs(np.asarray(u.get_mantissa(g))).sum() > 0 for g in g_warm.values())
    w0 = np.asarray(u.get_mantissa(online.params['weights/ho/0'].value)).copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loss, final = online.run(state1, xs, ys, n0=40)
    assert float(loss) == pytest.approx(float(l_warm), rel=1e-6)
    assert int(online.state()[1]) == 100
    assert jax.tree.structure(final) == jax.tree.structure(prog.init_state())
    assert not np.array_equal(np.asarray(u.get_mantissa(online.params['weights/ho/0'].value)), w0)


def test_online_state_matches_runner():
    """Loading a runner's state continues the runner's trajectory (no input)."""
    net = Network(dt=DT, seed=5)
    noise = net.add(poisson_generator, 8, params=dict(rate=800.0 * u.Hz), name='noise')
    x = net.add(ExternalInput, 2, name='x')
    h = net.add(LIF, 10, name='h', surrogate=SIG)
    net.connect(noise, h, port='delta', weight=0.4 * u.mV, trainable=True, name='nh')
    net.connect(x, h, port='delta', weight=0.4 * u.mV, trainable=True, name='xh')
    rec = net.observe(h, 'spike')
    plan = net.compile()
    run = plan.runner()
    run.simulate(5.0 * u.ms)
    online = Online(plan.program(), 'eprop', inputs='x', outputs='h')
    online.load(run.state, run.n)
    out = brainstate.transform.for_loop(online.network.update, jnp.zeros((30, 2)))
    res = run.cont(3.0 * u.ms)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(res[rec]['spike']))
    assert float(np.sum(out)) > 0


def test_etrace_network_keeps_program_options():
    plan = lif_net().compile()
    prog = plan.program(spikes='relaxed', detach_reset=True)
    en = EtraceNetwork(prog, inputs='x', outputs='o')
    assert en.program.etrace and en.program.spikes == 'relaxed' and en.program.detach_reset
    assert not prog.etrace  # the caller's program is not mutated
    out = en.update(jnp.ones(N_IN))
    assert out.shape == (N_O,)
