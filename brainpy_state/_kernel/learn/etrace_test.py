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

from brainpy_state._kernel.connectivity import fixed_indegree
from brainpy_state._kernel.learn import (
    EtraceError,
    EtraceNetwork,
    ExternalInput,
    Online,
    check_etrace,
    check_plan,
)
from brainpy_state._kernel.learn.etrace import EtraceReport
from brainpy_state._kernel.model import Capabilities
from brainpy_state._kernel.models import LIF, iaf_psc_alpha, poisson_generator
from brainpy_state._kernel.network import Network

DT = 0.1 * u.ms
SIG = braintools.surrogate.Sigmoid()
N_IN, N_H, N_O = 6, 12, 4


def lif_net(storage='dense', delay_steps=1, trainable=True):
    """x -> h (recurrent) -> o, the G7 network."""
    net = Network(dt=DT)
    x = net.add(ExternalInput, N_IN, name='x')
    h = net.add(LIF, N_H, params=dict(tau=10.0 * u.ms), name='h', surrogate=SIG)
    o = net.add(LIF, N_O, params=dict(tau=10.0 * u.ms), name='o', surrogate=SIG)
    net.connect(x, h, port='delta', weight=0.4 * u.mV, delay=delay_steps * DT, storage=storage,
                trainable=trainable, name='xh')
    net.connect(h, h, fixed_indegree(4), port='delta', weight=0.1 * u.mV, delay=delay_steps * DT,
                storage=storage, trainable=trainable, name='hh')
    net.connect(h, o, fixed_indegree(6), port='delta', weight=0.3 * u.mV, delay=0.0 * u.ms,
                storage=storage, trainable=trainable, name='ho')
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
    online = Online(algorithm, inputs='x', outputs=['h', 'o'])
    run = plan.runner(learner=online)
    assert run.learner is online and online.runner is run
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


class _InLoop(_Base):
    def update(self, x):
        I = jax.lax.fori_loop(0, 2, lambda i, a: a + braintrace.matmul(x, self.W.value), jnp.zeros(4))
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


@pytest.mark.parametrize('cls,match', [
    (_RawMatmul, 'not applied through a braintrace primitive'),
    (_WritesWeight, 'written inside the step'),
    (_InLoop, 'inside a scan/while/fori_loop'),
    (_OutputNotHidden, 'not new HiddenState values'),
    (_InputOnlyHidden, 'depends on no hidden state'),
])
def test_checker_rejects_non_compliant(cls, match):
    m = cls()
    brainstate.nn.init_all_states(m)
    with pytest.raises(EtraceError, match=match):
        check_etrace(m, jnp.ones(3))


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
        plan.runner(learner=Online('eprop', inputs='x', outputs='o'))


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
    with pytest.raises(NotImplementedError, match='K1'):
        lif_net().compile().runner(batch=2, learner=Online(inputs='x', outputs='o'))
    online = Online(inputs='x', outputs='o')
    run = lif_net().compile().runner(learner=online)
    with pytest.raises(RuntimeError, match='already attached'):
        online.attach(run)


def test_checker_rejects_k0_network_with_raw_weights():
    """A K0 network whose weights bypass the braintrace primitives is rejected."""
    en = EtraceNetwork(lif_net('dense').compile(), inputs='x', outputs='o')
    en.program.etrace = False  # storages' plain forward: braintrace cannot see the weights
    with pytest.raises(EtraceError, match='not applied through a braintrace primitive'):
        check_etrace(en, en.example_input())
