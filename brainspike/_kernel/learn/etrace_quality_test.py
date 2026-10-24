# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Update quality of the braintrace online learners (K1-L spec, K1L-0): gates G6 and G7.

The reference for every online gradient is BPTT through the same forward:
``brainstate.transform.grad`` of the mean per-step loss over a rollout of the same
:class:`EtraceNetwork` from the same state (``etrace_grad``'s default
``reduction='mean'`` divides by ``T``). The harness is validated first on cases with
a known answer: a hand-computed single step, element-wise single-layer cells (where
D-RTRL is exact RTRL, i.e. BPTT) and an independent BPTT (:class:`BPTT` over the
program, through the runner's storages).

* **G6 (OSTL exact, feed-forward).** ``ExternalInput -> LIF`` / ``-> iaf_psc_alpha``,
  one trainable layer, zero delay, Dense and CSR, T = 50, per-step loss, both spike
  modes with matched ``detach_reset``: ``OSTLRecurrent`` (= D-RTRL) and e-prop equal
  BPTT (``max|g - g_bptt| / max|g_bptt| <= 1e-6``, float64). On a two-layer
  feed-forward net the last layer meets the bound; the first-layer error (D-RTRL
  drops the cross-layer temporal terms) is reported, not gated.
* **G7 (D-RTRL vs BPTT on one step).** T = 1 from a warm state (>= 200 steps, traces
  reset), the G7 nets (``x -> h`` (recurrent) ``-> o`` with ``LIF``,
  ``iaf_psc_alpha``, ``iaf_cond_exp``), Dense and CSR, hard and relaxed spikes,
  outputs with the (pre-reset) spike reads, membrane and synaptic states, and a
  readout weight: ``<= 1e-6`` (float64), ``<= 1e-4`` (float32), every weight.

Without the braintrace#184 workaround (``new + 0 * old``, removed in K1L-0) these
gates hold to rounding; with it the LIF G7 gradients of ``xh`` / ``hh`` were off by
9-10 % (``outputs='o'``: the ``xh`` path was lost entirely), see the K1-L spec
decision log. :func:`test_pinned_online_gradients` pins the (approximate) multi-step
online gradients of the recurrent net so that such a change cannot pass unnoticed.
"""
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
    BPTT,
    EtraceNetwork,
    ExternalInput,
    Online,
    check_etrace,
    check_learner,
)
from brainspike._kernel.latch_test import LatchedLIF
from brainspike._kernel.models import (
    LIF,
    dc_generator,
    iaf_cond_exp,
    iaf_psc_alpha,
    poisson_generator,
    spike_generator,
)
from brainspike._kernel.network import Network
from brainspike._kernel.synapse import Expon

DT = 0.1 * u.ms
SIG = braintools.surrogate.Sigmoid()
N_IN, N_H, N_O = 6, 12, 4
TOL = {'float64': 1e-6, 'float32': 1e-4}

# model, input port, xh / hh / ho base weights, params, G7 output states
MODELS = {
    'LIF': (LIF, 'delta', (0.03 * u.mV, 0.05 * u.mV, 0.1 * u.mV), dict(tau=10.0 * u.ms),
            ('spike', 'V')),
    'iaf_psc_alpha': (iaf_psc_alpha, 'ex', (3.0 * u.pA, 30.0 * u.pA, 150.0 * u.pA),
                      dict(I_e=230.0 * u.pA), ('spike', 'V_m', 'I_syn_ex', 'dI_syn_ex')),
    'iaf_cond_exp': (iaf_cond_exp, 'ex', (0.6 * u.nS, 8.0 * u.nS, 30.0 * u.nS),
                     dict(I_e=240.0 * u.pA), ('spike', 'V_m', 'g_ex')),
}


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# ---------------------------------------------------------------------------
# networks and harness
# ---------------------------------------------------------------------------

def _jitter(w, n, seed):
    r = np.random.default_rng(seed).uniform(0.5, 1.5, n)
    return u.Quantity(r * u.get_mantissa(w), unit=u.get_unit(w))


def g7_net(model, storage, kind='rec', drive=False, observe=()):
    """``x -> h``: ``kind='ff'`` one layer (zero delay); ``'rec'`` adds ``h -> h`` and ``h -> o``.

    ``'rec'`` is the G7 net: ``xh`` and ``hh`` with one step of delay, ``ho`` zero
    delay. ``drive=True`` adds a non-trainable Poisson drive into ``h``; ``observe``
    names populations whose spikes are recorded (observer ``rec_<name>``).
    """
    m, port, (wxh, whh, who), params, _ = MODELS[model]
    net = Network(dt=DT, seed=3)
    x = net.add(ExternalInput, N_IN, name='x')
    h = net.add(m, N_H, params=params, name='h', surrogate=SIG)
    d = 0.0 if kind == 'ff' else 1.0
    net.connect(x, h, port=port, weight=_jitter(wxh, N_IN * N_H, 0), delay=d * DT, storage=storage,
                trainable=True, name='xh')
    if kind == 'rec':
        net.connect(h, h, fixed_indegree(4), port=port, weight=_jitter(whh, 4 * N_H, 1), delay=DT,
                    storage=storage, trainable=True, name='hh')
        o = net.add(m, N_O, params=params, name='o', surrogate=SIG)
        net.connect(h, o, fixed_indegree(6), port=port, weight=_jitter(who, 6 * N_O, 2),
                    delay=0.0 * u.ms, storage=storage, trainable=True, name='ho')
    if drive:
        noise = net.add(poisson_generator, 5, params=dict(rate=500.0 * u.Hz), name='noise')
        net.connect(noise, h, port=port, weight=wxh * 10, name='drive')
    pops = {'h': h, **({'o': o} if kind == 'rec' else {})}
    for p in observe:
        net.observe(pops[p], 'spike', name=f'rec_{p}')
    return net


def two_layer_ff(storage):
    """``x -> h1 -> h2`` (LIF, zero delays, both layers trainable)."""
    net = Network(dt=DT, seed=4)
    x = net.add(ExternalInput, N_IN, name='x')
    h1 = net.add(LIF, N_H, params=dict(tau=10.0 * u.ms), name='h1', surrogate=SIG)
    h2 = net.add(LIF, N_O, params=dict(tau=10.0 * u.ms), name='h2', surrogate=SIG)
    net.connect(x, h1, port='delta', weight=_jitter(0.3 * u.mV, N_IN * N_H, 5), delay=0.0 * u.ms,
                storage=storage, trainable=True, name='x1')
    net.connect(h1, h2, port='delta', weight=_jitter(0.3 * u.mV, N_H * N_O, 6), delay=0.0 * u.ms,
                storage=storage, trainable=True, name='12')
    return net


def _mse(out, y):
    return jnp.mean((out - y) ** 2)


def _params(net):
    return {f'{pp.name}/{g}': grp.weight for pp in net.plan.projections
            for g, grp in enumerate(pp.groups) if isinstance(grp.weight, brainstate.ParamState)}


def _mant(g):
    return {k: np.asarray(u.get_mantissa(v), dtype=np.float64) for k, v in g.items()}


class Readout(EtraceNetwork):
    """An :class:`EtraceNetwork` whose step also applies a trainable readout ``tanh(y) @ w_out``."""

    def __init__(self, program, inputs, outputs, n_out, n_read=3):
        super().__init__(program, inputs, outputs)
        w = jax.random.normal(jax.random.key(4), (n_out, n_read)) * 0.3
        self.w_out = brainstate.ParamState(w.astype(self.program.dtype))

    def update(self, x=None):
        y = super().update(x)
        return jnp.concatenate([y, jnp.tanh(y) @ self.w_out.value])


class Case:
    """One network + spike mode: warm states, BPTT and online gradients of an episode."""

    def __init__(self, plan, outputs, spikes='hard', detach_reset=False, readout=False,
                 netcls=EtraceNetwork):
        self.prog = plan.program(spikes=spikes, detach_reset=detach_reset)
        self.outputs = outputs
        self.readout = readout
        self.netcls = netcls

    def net(self):
        if self.readout:
            n_out = EtraceNetwork(self.prog, 'x', self.outputs).update(
                jnp.zeros(N_IN, self.prog.dtype)).size
            net = Readout(self.prog, 'x', self.outputs, n_out)
        else:
            net = self.netcls(self.prog, 'x', self.outputs)
        params = _params(net)
        if self.readout:
            params['w_out'] = net.w_out
        return net, params

    def warm(self, n_min=200, live='h'):
        """State after >= ``n_min`` steps of random input whose last step has a ``live`` spike."""
        net = EtraceNetwork(self.prog, 'x', live)
        xs = (jax.random.uniform(jax.random.key(7), (n_min + 200, N_IN)) < 0.3).astype(self.prog.dtype)
        spikes = np.asarray(brainstate.transform.for_loop(net.update, xs)).sum(1)
        n = next(i + 1 for i in range(n_min - 1, len(spikes)) if spikes[i] > 0)
        net.init_state()
        brainstate.transform.for_loop(net.update, xs[:n])
        return net.read(), n

    def bptt(self, state0, n0, xs, ys):
        net, params = self.net()
        net.load(state0, n0)

        def total():
            return brainstate.transform.for_loop(lambda x, y: _mse(net.update(x), y), xs, ys).mean()
        return _mant(brainstate.transform.grad(total, grad_states=params)())

    def online(self, algorithm, state0, n0, xs, ys):
        net, params = self.net()
        report = check_etrace(net, net.example_input())
        learner = braintrace.compile(net, algorithm, net.example_input())
        check_learner(learner, report)
        net.load(state0, n0)
        learner.reset_state()
        g = learner.etrace_grad(xs, ys, step_fn=lambda x, y: _mse(learner(x), y), weights=params)
        return _mant(g)


def rel_err(g, ref):
    """Per weight ``max|g - ref| / max|ref|`` (the absolute error where ``ref`` is all zero)."""
    out = {}
    for k in ref:
        scale = float(np.max(np.abs(ref[k])))
        err = float(np.max(np.abs(g[k] - ref[k])))
        out[k] = err / scale if scale > 0 else err
    return out


def _xs(T, dtype, seed=1, p=0.5):
    return (jax.random.uniform(jax.random.key(seed), (T, N_IN)) < p).astype(dtype)


# ---------------------------------------------------------------------------
# harness validation (cases with a known answer)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('detach_reset', [False, True])
def test_harness_hand_computed_single_step(x64, detach_reset):
    """One LIF step from rest: ``V_pre = x @ W``, gradient by hand (soft reset, Sigmoid surrogate)."""
    n = 5
    W = np.linspace(0.05, 0.5, N_IN * n).reshape(N_IN, n)
    net = Network(dt=DT)
    x = net.add(ExternalInput, N_IN, name='x')
    h = net.add(LIF, n, params=dict(tau=10.0 * u.ms), name='h', surrogate=SIG)
    net.connect(x, h, port='delta', weight=W.reshape(-1) * u.mV, delay=0.0 * u.ms, trainable=True,
                name='xh')
    plan = net.compile(precision='float64')
    W = np.asarray(u.get_mantissa(plan.projection('xh').groups[0].weight.value))  # (pre, post)
    case = Case(plan, [('h', 'spike'), ('h', 'V')], detach_reset=detach_reset)
    xv = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    y = np.concatenate([np.full(n, 0.3), np.full(n, 0.2)])
    # hand computation: V_pre = x @ W (V0 = V_rest = 0, no current), margin m = V_pre - 1
    V_pre = xv @ W
    m = V_pre - 1.0
    s = (m >= 0).astype(float)
    sig = 1.0 / (1.0 + np.exp(-SIG.alpha * m))
    ds = SIG.alpha * sig * (1.0 - sig)                    # d spike / d V_pre (V_th - V_reset = 1)
    V1 = V_pre - s                                        # soft reset by 1 mV
    dV1 = np.ones(n) if detach_reset else 1.0 - ds        # d V1 / d V_pre
    e_s, e_v = s - y[:n], V1 - y[n:]
    g_hand = np.outer(xv, (2.0 / (2 * n)) * (e_s * ds + e_v * dV1))
    assert 0 < s.sum() < n                                # both branches of the threshold
    xs, ys = jnp.asarray(xv)[None], jnp.asarray(y)[None]
    state0 = case.prog.init_state()
    g_bptt = case.bptt(state0, 0, xs, ys)['xh/0']
    np.testing.assert_allclose(g_bptt, g_hand, rtol=1e-12, atol=1e-15)
    for algorithm in ('d_rtrl', 'eprop', 'ostl_recurrent'):
        g = case.online(algorithm, state0, 0, xs, ys)['xh/0']
        np.testing.assert_allclose(g, g_hand, rtol=1e-12, atol=1e-15, err_msg=algorithm)


@pytest.mark.parametrize('kind', ['ff', 'rec'])
def test_harness_bptt_equals_program_bptt(x64, kind):
    """The harness's BPTT (through the etrace forward) equals :class:`BPTT` over the program."""
    pops = ['h'] if kind == 'ff' else ['h', 'o']
    plan = g7_net('LIF', 'csr', kind, observe=pops).compile(precision='float64')
    case = Case(plan, pops, spikes='relaxed')
    state0, n0 = case.warm(live='h')
    T = 8
    xs = _xs(T, jnp.float64)
    ys = jnp.full((T, sum(plan.population(p).pop.n for p in pops)), 0.2)
    g_harness = case.bptt(state0, n0, xs, ys)

    def loss(rec, y):
        out = jnp.concatenate([rec[f'rec_{p}']['spike'] for p in pops], axis=-1)
        return jnp.mean((out - y) ** 2)
    bptt = BPTT(case.prog, loss, detach_reset=False)
    _, g_prog = bptt.grad(state0, {'x': xs}, ys, n0=n0)
    g_prog = {k.split('/', 1)[1]: v for k, v in _mant(g_prog).items()}
    for k, err in rel_err(g_harness, g_prog).items():
        assert err <= 1e-12, (k, err)
        assert np.abs(g_prog[k]).max() > 0, k


@pytest.mark.parametrize('mode', ['hidden', 'pre_reset', 'readout'])
@pytest.mark.parametrize('T', [1, 12])
def test_harness_elementwise_cells_exact(x64, mode, T):
    """Element-wise single-layer cells: D-RTRL / e-prop equal BPTT, incl. bypass outputs."""

    class Cell(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(jax.random.normal(jax.random.key(0), (3, 5)))
            self.w_out = brainstate.ParamState(jax.random.normal(jax.random.key(1), (5, 2)))

        def init_state(self, *args, **kwargs):
            self.V = brainstate.HiddenState(jnp.zeros(5))

        def update(self, x):
            V = 0.9 * self.V.value + braintrace.matmul(x, self.w.value)
            spk = SIG(V - 1.0)
            self.V.value = V - jax.lax.stop_gradient(spk)
            if mode == 'pre_reset':
                return spk                       # read before the reset (bypasses V)
            if mode == 'readout':
                return jnp.tanh(self.V.value) @ self.w_out.value
            return self.V.value

    xs = jax.random.uniform(jax.random.key(2), (T, 3)) * 2.0
    ys = jnp.full((T, 2 if mode == 'readout' else 5), 0.3)

    def fresh():
        cell = Cell()
        brainstate.nn.init_all_states(cell)
        params = {'w': cell.w, 'w_out': cell.w_out} if mode == 'readout' else {'w': cell.w}
        return cell, params

    cell, params = fresh()
    ref = _mant(brainstate.transform.grad(lambda: brainstate.transform.for_loop(
        lambda x, y: _mse(cell(x), y), xs, ys).mean(), grad_states=params)())
    for algorithm in ('d_rtrl', 'eprop'):
        cell, params = fresh()
        report = check_etrace(cell, xs[0])
        assert report.bypass == ([] if mode == 'hidden' else [0])
        assert report.readouts == (['w_out'] if mode == 'readout' else [])
        learner = braintrace.compile(cell, algorithm, xs[0])
        check_learner(learner, report)
        g = _mant(learner.etrace_grad(xs, ys, step_fn=lambda x, y: _mse(learner(x), y),
                                      weights=params))
        for k, err in rel_err(g, ref).items():
            assert err <= 1e-12, (algorithm, k, err)
            assert np.abs(ref[k]).max() > 0


def test_harness_detects_inexact_gradients(x64):
    """Not vacuous: D-RTRL is approximate over many steps of the recurrent net."""
    plan = g7_net('LIF', 'dense', 'rec').compile(precision='float64')
    case = Case(plan, ['h', 'o'], spikes='relaxed')
    state0, n0 = case.warm()
    T = 20
    xs, ys = _xs(T, jnp.float64), jnp.full((T, N_H + N_O), 0.2)
    err = rel_err(case.online('d_rtrl', state0, n0, xs, ys), case.bptt(state0, n0, xs, ys))
    assert err['ho/0'] <= 1e-6          # the last layer has no temporal cross-terms
    assert err['xh/0'] > 1e-2 and err['hh/0'] > 1e-2


# ---------------------------------------------------------------------------
# G6: OSTL exact on a feed-forward net
# ---------------------------------------------------------------------------

G6_MODES = [('hard', False), ('hard', True), ('relaxed', False)]


@pytest.mark.parametrize('spikes,detach_reset', G6_MODES)
@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('model', ['LIF', 'iaf_psc_alpha'])
def test_g6_ostl_exact_feedforward(x64, model, storage, spikes, detach_reset):
    plan = g7_net(model, storage, 'ff').compile(precision='float64')
    case = Case(plan, [('h', s) for s in MODELS[model][4]], spikes, detach_reset)
    T = 50
    state0 = case.prog.init_state()
    xs = _xs(T, jnp.float64)
    n_out = len(MODELS[model][4]) * N_H
    ys = jnp.zeros((T, n_out))
    ref = case.bptt(state0, 0, xs, ys)
    assert np.abs(ref['xh/0']).max() > 0
    for algorithm in ('ostl_recurrent', 'd_rtrl', 'eprop'):
        err = rel_err(case.online(algorithm, state0, 0, xs, ys), ref)['xh/0']
        assert err <= TOL['float64'], (algorithm, err)


@pytest.mark.parametrize('storage', ['dense', 'csr'])
def test_g6_two_layer_last_layer_exact(x64, storage, capsys):
    plan = two_layer_ff(storage).compile(precision='float64')
    case = Case(plan, ['h1', 'h2'], spikes='relaxed')
    T = 50
    state0 = case.prog.init_state()
    xs, ys = _xs(T, jnp.float64), jnp.full((T, N_H + N_O), 0.2)
    ref = case.bptt(state0, 0, xs, ys)
    err = rel_err(case.online('ostl_recurrent', state0, 0, xs, ys), ref)
    assert err['12/0'] <= TOL['float64']
    assert np.isfinite(err['x1/0'])
    with capsys.disabled():  # reported, not gated (D-RTRL drops cross-layer temporal terms)
        print(f'\n[K1L-G6] two-layer {storage}: first-layer rel err {err["x1/0"]:.3e}, '
              f'last-layer {err["12/0"]:.3e}')


# ---------------------------------------------------------------------------
# G7: D-RTRL vs BPTT on one step
# ---------------------------------------------------------------------------

def _g7(model, storage, spikes, precision, readout=False):
    plan = g7_net(model, storage, 'rec').compile(precision=precision)
    outs = [(p, s) for p in ('h', 'o') for s in MODELS[model][4]]
    case = Case(plan, outs, spikes, readout=readout)
    state0, n0 = case.warm()
    dtype = case.prog.dtype
    xs = _xs(1, dtype)
    n_out = len(MODELS[model][4]) * (N_H + N_O)
    ys = jnp.zeros((1, n_out + (3 if readout else 0)), dtype)
    ref = case.bptt(state0, n0, xs, ys)
    return case, state0, n0, xs, ys, ref


@pytest.mark.parametrize('spikes', ['hard', 'relaxed'])
@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('model', list(MODELS))
def test_g7_drtrl_equals_bptt_one_step_f64(x64, model, storage, spikes):
    case, state0, n0, xs, ys, ref = _g7(model, storage, spikes, 'float64')
    assert set(ref) == {'xh/0', 'hh/0', 'ho/0'}
    if spikes == 'relaxed':  # dense surrogate spikes: every weight has a gradient
        assert all(np.abs(v).max() > 0 for v in ref.values())
    for algorithm in ('d_rtrl', 'eprop'):
        for k, err in rel_err(case.online(algorithm, state0, n0, xs, ys), ref).items():
            assert err <= TOL['float64'], (algorithm, k, err)


@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('model', list(MODELS))
def test_g7_drtrl_equals_bptt_one_step_f32(model, storage):
    case, state0, n0, xs, ys, ref = _g7(model, storage, 'relaxed', 'float32')
    assert all(np.abs(v).max() > 0 for v in ref.values())
    for k, err in rel_err(case.online('d_rtrl', state0, n0, xs, ys), ref).items():
        assert err <= TOL['float32'], (k, err)


@pytest.mark.parametrize('spikes', ['hard', 'relaxed'])
@pytest.mark.parametrize('storage', ['dense', 'csr'])
def test_g7_readout_weight(x64, storage, spikes):
    """A readout applied in the step (0.2.6 rule 3 rejected it) is trained exactly."""
    case, state0, n0, xs, ys, ref = _g7('LIF', storage, spikes, 'float64', readout=True)
    net, _ = case.net()
    report = check_etrace(net, net.example_input())
    assert report.readouts == ['w_out'] and report.bypass
    assert np.abs(ref['w_out']).max() > 0
    for algorithm in ('d_rtrl', 'eprop'):
        for k, err in rel_err(case.online(algorithm, state0, n0, xs, ys), ref).items():
            assert err <= TOL['float64'], (algorithm, k, err)


class KeepDependency(EtraceNetwork):
    """The removed braintrace#184 workaround: every new hidden value stored as ``new + 0 * old``.

    This step's consumers (zero-delay targets, rings) read ``new``, while braintrace
    sees ``new + 0 * old`` as the hidden value, so the same-step ``h -> o`` path looks
    like a bypass from the previous ``h`` and is lost from the learning signal.
    """

    def update(self, x=None):
        hidden = [s for s in self.states(brainstate.HiddenState).values()]
        old = [s.value for s in hidden]
        super().update(x)
        for s, v in zip(hidden, old):
            s.value = s.value + 0 * v
        outs = [u.get_mantissa(pplan.model.dyn_states[st].value) for pplan, st in self._outputs]
        return jnp.concatenate(outs)


def test_g7_fails_with_the_keep_dependency_workaround(x64):
    """The #184 decision by evidence: the gate rejects the old variant (K1L-0)."""
    plan = g7_net('LIF', 'dense', 'rec').compile(precision='float64')
    outs = [(p, s) for p in ('h', 'o') for s in MODELS['LIF'][4]]
    kept = Case(plan, outs, 'relaxed', netcls=KeepDependency)
    removed = Case(plan, outs, 'relaxed')
    state0, n0 = removed.warm()
    xs, ys = _xs(1, jnp.float64), jnp.zeros((1, 2 * (N_H + N_O)))
    ref = removed.bptt(state0, n0, xs, ys)
    np.testing.assert_array_equal(np.asarray(kept.bptt(state0, n0, xs, ys)['xh/0']), ref['xh/0'])
    err_kept = rel_err(kept.online('d_rtrl', state0, n0, xs, ys), ref)
    err_removed = rel_err(removed.online('d_rtrl', state0, n0, xs, ys), ref)
    assert max(err_removed.values()) <= TOL['float64']
    assert err_kept['xh/0'] > 1e-2 and err_kept['hh/0'] > 1e-2  # measured: 9.9e-2, 9.9e-2
    assert err_kept['ho/0'] <= TOL['float64']


def test_g7_online_learner_matches_bptt(x64):
    """:class:`Online` (the learner API) gives the gated one-step gradient."""
    plan = g7_net('LIF', 'csr', 'rec').compile(precision='float64')
    outs = [('h', 'spike'), ('o', 'spike'), ('o', 'V')]
    case = Case(plan, outs, 'relaxed')
    state0, n0 = case.warm()
    xs, ys = _xs(1, jnp.float64), jnp.full((1, N_H + 2 * N_O), 0.1)
    ref = case.bptt(state0, n0, xs, ys)
    online = Online(case.prog, 'd_rtrl', inputs='x', outputs=outs, loss=_mse)
    _, grads = online.grad(state0, xs, ys, n0=n0)
    grads = {k.split('/', 1)[1]: v for k, v in _mant(grads).items()}
    for k, err in rel_err(grads, ref).items():
        assert err <= TOL['float64'], (k, err)


# ---------------------------------------------------------------------------
# pinned values of the approximate multi-step gradients
# ---------------------------------------------------------------------------

# Frobenius norms of the T = 20 online gradients of the recurrent LIF net (CSR, hard
# spikes, float64) from the warm state; d_rtrl and e-prop coincide on this net.
PINNED = {'xh/0': 0.05579055991130147, 'hh/0': 0.020204825983898743, 'ho/0': 0.013105603558411788}


def test_pinned_online_gradients(x64):
    plan = g7_net('LIF', 'csr', 'rec').compile(precision='float64')
    case = Case(plan, ['h', 'o'])
    state0, n0 = case.warm()
    assert n0 == 201
    T = 20
    xs, ys = _xs(T, jnp.float64), jnp.full((T, N_H + N_O), 0.2)
    for algorithm in ('d_rtrl', 'eprop'):
        g = case.online(algorithm, state0, n0, xs, ys)
        norms = {k: float(np.linalg.norm(v)) for k, v in g.items()}
        for k, v in PINNED.items():
            np.testing.assert_allclose(norms[k], v, rtol=1e-9, err_msg=f'{algorithm} {k}')


# ---------------------------------------------------------------------------
# forward: EtraceNetwork equals the runner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('model', list(MODELS))
def test_etrace_forward_equals_runner_bitwise(model, precision):
    with jax.enable_x64(precision == 'float64'):
        plan = g7_net(model, 'csr', 'rec', drive=True).compile(precision=precision)
        run = plan.runner()
        run.simulate(20.0 * u.ms)
        en = EtraceNetwork(plan, inputs='x', outputs=['h', 'o'])
        out = brainstate.transform.for_loop(en.update, jnp.zeros((200, N_IN), plan.program().dtype))
        assert float(jnp.sum(out)) > 0
        for a, b in zip(jax.tree.leaves(run.state), jax.tree.leaves(en.read())):
            a, b = np.asarray(u.get_mantissa(a)), np.asarray(u.get_mantissa(b))
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _latched_net(model):
    """Poisson drive, a ``dc_generator`` into the latched ``I`` port, delayed recurrence."""
    w, drive = (80.0 * u.pA, 150.0 * u.pA) if model is iaf_psc_alpha else (3.0 * u.nS, 2.0 * u.nS)
    net = Network(dt=DT, seed=9)
    noise = net.add(poisson_generator, 8, params=dict(rate=900.0 * u.Hz), name='noise')
    dc = net.add(dc_generator, 1, params=dict(amplitude=300.0 * u.pA), name='dc')
    h = net.add(model, 10, name='h', surrogate=SIG)
    net.connect(noise, h, port='ex', weight=drive, trainable=True, name='nh')
    net.connect(dc, h, port='I', weight=1.0, name='dch')
    net.connect(h, h, fixed_indegree(3), port='ex', weight=w, delay=0.2 * u.ms, trainable=True,
                name='hh')
    return net


def _deferred_cycle():
    """A zero-delay cycle through a latched port (deferred delivery, K1a-2)."""
    net = Network(dt=DT, seed=0)
    sg = net.add(spike_generator, 1, params=dict(spike_times=np.array([1.0, 2.0, 3.3]) * u.ms))
    a = net.add(LIF, 1, params=dict(V_th=0.5 * u.mV), name='A')
    b = net.add(LatchedLIF, 1, params=dict(V_th=0.5 * u.mV), name='B')
    net.connect(sg, a, port='delta', weight=5 * u.mV, delay=0 * u.ms, trainable=True)
    net.connect(b, a, port='delta', weight=0.3 * u.mV, delay=0 * u.ms)
    net.connect(a, b, port='I', weight=5 * u.mA, delay=0 * u.ms, synapse=Expon(2 * u.ms))
    return net


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('name', ['iaf_psc_alpha', 'iaf_cond_exp', 'cycle'])
def test_etrace_forward_equals_runner_latched(name, precision):
    """Latched ports (incl. deferred ones) carry no workaround: the forward is the runner's."""
    with jax.enable_x64(precision == 'float64'):
        net = _deferred_cycle() if name == 'cycle' else _latched_net(
            {'iaf_psc_alpha': iaf_psc_alpha, 'iaf_cond_exp': iaf_cond_exp}[name])
        plan = net.compile(precision=precision)
        run = plan.runner()
        run.simulate(5.0 * u.ms)
        hidden = [(p.name, k) for p in plan.populations for k, s in p.model.dyn_states.items()
                  if isinstance(s, brainstate.HiddenState)]
        en = EtraceNetwork(plan, outputs=hidden)
        brainstate.transform.for_loop(lambda i: en.update(), jnp.arange(50))
        flat_r = jax.tree_util.tree_leaves_with_path(run.state)
        flat_e = jax.tree.leaves(en.read())
        bad = []
        for (path, a), b in zip(flat_r, flat_e):
            a, b = np.asarray(u.get_mantissa(a)), np.asarray(u.get_mantissa(b))
            assert a.dtype == b.dtype
            if a.tobytes() != b.tobytes():
                bad.append((jax.tree_util.keystr(path), a, b))
        if name == 'iaf_psc_alpha' and precision == 'float64':
            # Known, independent of the K1L-0 change (identical before it): with two Dense
            # projections and the dc input, XLA (CPU) fuses the float64 synaptic sums of
            # the etrace step (braintrace.matmul) differently from the runner's: ~1 ulp in
            # I_syn_ex / dI_syn_ex. Spikes and every other state stay bit-identical.
            for key, a, b in bad:
                np.testing.assert_allclose(b, a, rtol=1e-14, atol=0, err_msg=key)
            assert len(bad) <= 2
            st_r, st_e = run.state['pops']['h'], en.read()['pops']['h']
            for k in ('spike', 'V_m', 'refractory_count', 'latch_I'):
                assert np.asarray(u.get_mantissa(st_r[k])).tobytes() == \
                    np.asarray(u.get_mantissa(st_e[k])).tobytes(), k
        else:
            assert not bad, [k for k, _, _ in bad]
