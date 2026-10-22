# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the BPTT learner, including gate G6 (gradients vs central finite differences).

G6 is checked in two regimes (see ``docs/specs/2026-10-22-k0-t9-learners.md``):

* **hard forward** (the production pass: exact spike counts, surrogate backward) on
  feed-forward nets with ``detach_reset=True`` and a membrane loss -- no gradient
  flows through a spike, so the surrogate BPTT gradient *is* the exact derivative of
  the piecewise-smooth loss and must match finite differences;
* **relaxed forward** (``Program(plan, spikes='relaxed')``: spikes = the smooth
  surrogate primitive, soft jumps) on two-layer recurrent nets with every weight and
  a neuron parameter trainable and ``detach_reset=False`` -- this exercises the spike
  payload VJPs through Dense / CSR storages, delay rings and the reset path.

K1a-5 adds the same checks from a **warmed-up** ``state0`` (200 steps of a
``Runner``), and the learner API over ``(Program, state0)``.
"""
from __future__ import annotations

import braintools
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.connectivity import fixed_indegree
from brainpy_state._kernel.learn import BPTT, ExternalInput, membrane_mse, rate_mse, spike_count_mse
from brainpy_state._kernel.learn.bptt import _cut, trainable_states
from brainpy_state._kernel.model import Capabilities
from brainpy_state._kernel.models import LIF, iaf_psc_alpha, poisson_generator
from brainpy_state._kernel.network import Network
from brainpy_state._kernel.runner import Program, Runner

DT = 0.1 * u.ms
SIG = braintools.surrogate.Sigmoid()


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


# ---------------------------------------------------------------------------
# network builders
# ---------------------------------------------------------------------------

MODELS = {
    # model, input port, input weight, recurrent weight, params, trainable neuron param
    'LIF': (LIF, 'delta', 0.35 * u.mV, 0.15 * u.mV, dict(tau=8.0 * u.ms), 'tau'),
    'iaf_psc_alpha': (iaf_psc_alpha, 'ex', 300.0 * u.pA, 120.0 * u.pA,
                      dict(I_e=300.0 * u.pA), 'tau_m'),
}


def _xs(n_steps, n_in, seed=0, p=0.3):
    k = jax.random.key(seed)
    return (jax.random.uniform(k, (n_steps, n_in)) < p).astype(float)


def _drive(net, model_name, target, weight_scale=0.5):
    """A non-trainable Poisson drive (keeps a warmed-up state and the RNG streams busy)."""
    _, port, w_in, _, _, _ = MODELS[model_name]
    noise = net.add(poisson_generator, 4, params=dict(rate=400.0 * u.Hz), name='noise')
    net.connect(noise, target, port=port, weight=w_in * weight_scale, name='drive')


def feedforward(model_name, storage, delay_steps, n=20, n_in=6, drive=False):
    """ExternalInput -> 20 neurons (the G6 hard-forward net)."""
    model, port, w_in, _, params, tparam = MODELS[model_name]
    net = Network(dt=DT, seed=11)
    x = net.add(ExternalInput, n_in, name='x')
    h = net.add(model, n, params=params, name='h', surrogate=SIG, trainable=(tparam,))
    w = u.Quantity(np.linspace(0.5, 1.5, n_in * n) * u.get_mantissa(w_in), unit=u.get_unit(w_in))
    net.connect(x, h, port=port, weight=w, delay=delay_steps * DT, storage=storage,
                trainable=True, name='xh')
    if drive:
        _drive(net, model_name, h)
    net.observe(h, 'spike', name='rec_h')
    net.observe(h, 'V' if model is LIF else 'V_m', name='v_h')
    return net


def recurrent(model_name, storage, delay_steps, n_in=6, drive=False):
    """x -> A (10) -> B (10) plus A -> A (the G6 relaxed net, 20 neurons)."""
    model, port, w_in, w_rec, params, tparam = MODELS[model_name]
    net = Network(dt=DT, seed=12)
    x = net.add(ExternalInput, n_in, name='x')
    A = net.add(model, 10, params=params, name='A', surrogate=SIG, trainable=(tparam,))
    B = net.add(model, 10, params=params, name='B', surrogate=SIG)
    unit = u.get_unit(w_in)
    net.connect(x, A, port=port, weight=w_in, delay=delay_steps * DT, storage=storage,
                trainable=True, name='xA')
    net.connect(A, A, fixed_indegree(4), port=port,
                weight=u.Quantity(np.linspace(0.3, 1.0, 40) * u.get_mantissa(w_rec), unit=unit),
                delay=max(delay_steps, 1) * DT, storage=storage, trainable=True, name='AA')
    net.connect(A, B, fixed_indegree(5), port=port, weight=w_rec * 2, delay=delay_steps * DT,
                storage=storage, trainable=True, name='AB')
    if drive:
        _drive(net, model_name, A)
    net.observe(A, 'V' if model is LIF else 'V_m', name='v_A')
    net.observe(B, 'spike', name='s_B')
    return net


# ---------------------------------------------------------------------------
# finite differences
# ---------------------------------------------------------------------------

def _shift(states, direction, eps):
    for k, s in states.items():
        v = s.value
        m = u.get_mantissa(v) + eps * direction[k]
        s.value = u.Quantity(m, unit=u.get_unit(v)) if isinstance(v, u.Quantity) else m


def _directional_fd(bptt, args, kw, direction, eps):
    states = bptt.params
    base = {k: s.value for k, s in states.items()}
    _shift(states, direction, eps)
    lp = float(bptt.loss(*args, **kw))
    for k, s in states.items():
        s.value = base[k]
    _shift(states, direction, -eps)
    lm = float(bptt.loss(*args, **kw))
    for k, s in states.items():
        s.value = base[k]
    return (lp - lm) / (2 * eps)


def _directions(bptt, grads, seed=0):
    """A random direction scaled by each parameter's magnitude, plus the largest coordinates."""
    rng = np.random.default_rng(seed)
    out = []
    d = {}
    for k, s in bptt.params.items():
        m = np.asarray(u.get_mantissa(s.value))
        scale = np.mean(np.abs(m)) or 1.0
        d[k] = rng.standard_normal(m.shape) * scale
    out.append(d)
    for k, s in bptt.params.items():
        g = np.asarray(u.get_mantissa(grads[k]))
        i = np.unravel_index(np.argmax(np.abs(g)), g.shape)
        e = {kk: np.zeros(np.shape(u.get_mantissa(ss.value))) for kk, ss in bptt.params.items()}
        e[k][i] = 1.0
        out.append(e)
    return out


def _check_fd(bptt, state0, xs, targets, eps=1e-6, rtol=1e-4, **kw):
    args = (state0, xs, targets)
    loss, grads = bptt.grad(*args, **kw)
    assert np.isfinite(float(loss))
    for d in _directions(bptt, grads):
        ad = sum(float(np.sum(np.asarray(u.get_mantissa(grads[k])) * d[k])) for k in d)
        fd = _directional_fd(bptt, args, kw, d, eps)
        assert ad != 0.0
        assert abs(ad - fd) <= rtol * abs(fd), (ad, fd)
    return grads


def _membrane_loss(name):
    return lambda r, y: membrane_mse(r[name][list(r[name])[0]], y)


def _relaxed_loss(model_name):
    v_target = -62.0 if model_name == 'iaf_psc_alpha' else 0.5

    def loss(r, y):
        v = r['v_A'][list(r['v_A'])[0]]
        return spike_count_mse(r['s_B']['spike'], y) + 0.1 * membrane_mse(v, v_target)
    return loss


# ---------------------------------------------------------------------------
# G6
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model_name', list(MODELS))
@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('delay_steps', [0, 2])
def test_g6_hard_forward_vs_finite_differences(x64, model_name, storage, delay_steps):
    net = feedforward(model_name, storage, delay_steps)
    target = -60.0 if model_name == 'iaf_psc_alpha' else 0.8
    plan = net.compile(precision='float64')
    assert plan.projection('xh').groups[0].storage.kind == storage
    bptt = BPTT(plan.program(), _membrane_loss('v_h'), detach_reset=True)
    xs = {'x': _xs(60, 6)}
    rec = bptt.record(None, xs)
    assert rec['rec_h']['spike'].shape == (60, 20)
    assert float(np.sum(rec['rec_h']['spike'])) > 0  # the nets spike: gradients cross resets
    _check_fd(bptt, None, xs, target)


@pytest.mark.parametrize('model_name', list(MODELS))
@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('delay_steps', [0, 2])
def test_g6_relaxed_vs_finite_differences(x64, model_name, storage, delay_steps):
    net = recurrent(model_name, storage, delay_steps)
    plan = net.compile(precision='float64')
    # relaxed spikes are fractional: the relaxed program uses the exact float path itself
    prog = plan.program(spikes='relaxed')
    bptt = BPTT(prog, _relaxed_loss(model_name), detach_reset=False)
    assert bptt.program.spikes == 'relaxed' and bptt.program.detach_reset is False
    assert set(bptt.params) == {f'pops/A/{MODELS[model_name][5]}', 'weights/xA/0',
                                'weights/AA/0', 'weights/AB/0'}
    grads = _check_fd(bptt, None, {'x': _xs(60, 6)}, jnp.full(10, 2.0))
    for k, g in grads.items():
        assert np.any(np.asarray(u.get_mantissa(g)) != 0), k


# ---------------------------------------------------------------------------
# K1a-5: episodes from a warmed-up state0
# ---------------------------------------------------------------------------

def _warm(plan, steps=200, batch=None, rng='independent'):
    run = Runner(plan.program(), batch=batch, rng=rng)
    run.simulate(steps * DT)
    assert run.n == steps
    return run


@pytest.mark.parametrize('model_name', list(MODELS))
def test_g6_hard_forward_from_warm_state(x64, model_name):
    plan = feedforward(model_name, 'csr', 2, drive=True).compile(precision='float64')
    run = _warm(plan)
    target = -60.0 if model_name == 'iaf_psc_alpha' else 0.8
    bptt = BPTT(plan.program(), _membrane_loss('v_h'), detach_reset=True)
    xs = {'x': _xs(60, 6, seed=1)}
    kw = dict(n0=run.n)
    assert float(np.sum(bptt.record(run.state, xs, **kw)['rec_h']['spike'])) > 0
    g_warm = _check_fd(bptt, run.state, xs, target, **kw)
    _, g_cold = bptt.grad(None, xs, target)
    assert any(not np.allclose(u.get_mantissa(g_warm[k]), u.get_mantissa(g_cold[k])) for k in g_cold)


@pytest.mark.parametrize('model_name', list(MODELS))
def test_g6_relaxed_from_warm_state(x64, model_name):
    plan = recurrent(model_name, 'dense', 1, drive=True).compile(precision='float64')
    run = _warm(plan)
    bptt = BPTT(plan.program(), _relaxed_loss(model_name), detach_reset=False, spikes='relaxed')
    grads = _check_fd(bptt, run.state, {'x': _xs(60, 6, seed=2)}, jnp.full(10, 2.0), n0=run.n)
    for k, g in grads.items():
        assert np.any(np.asarray(u.get_mantissa(g)) != 0), k


def _noise_net(seed=4):
    net = Network(dt=DT, seed=seed)
    noise = net.add(poisson_generator, 10, params=dict(rate=600.0 * u.Hz), name='noise')
    h = net.add(LIF, 6, name='h', surrogate=SIG, trainable=('tau',))
    net.connect(noise, h, fixed_indegree(4), port='delta', weight=0.45 * u.mV, trainable=True,
                name='w')
    net.connect(h, h, fixed_indegree(2), port='delta', weight=0.2 * u.mV, delay=0.3 * u.ms,
                trainable=True, name='rec')
    net.observe(h, 'spike', 'V', name='r')
    return net


def test_episode_forward_equals_runner_cont(x64):
    """The episode from a runner's state is the runner's continuation (same RNG streams)."""
    plan = _noise_net().compile(precision='float64')
    run = _warm(plan)
    state0, n0 = run.state, run.n
    bptt = BPTT(plan.program(), lambda r, y: rate_mse(r['r']['spike'], y, DT))
    rec = bptt.record(state0, duration=8.0 * u.ms, n0=n0)
    res = run.cont(8.0 * u.ms)
    np.testing.assert_array_equal(rec['r']['spike'], np.asarray(res['r']['spike']))
    np.testing.assert_allclose(rec['r']['V'], np.asarray(res['r']['V'].to_decimal(u.mV)),
                               rtol=1e-12, atol=1e-12)
    assert float(np.sum(rec['r']['spike'])) > 0
    # a learner's final state continues the runner exactly
    _, _, final = bptt._jit_grad(80, state0, jnp.int32(n0), jnp.zeros((), jnp.uint32), None, 0.0)
    for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(run.state)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


def test_batched_runner_state(x64):
    """``state0`` from a batched runner (trials = runner.trials) = per-trial continuation."""
    plan = _noise_net().compile(precision='float64')
    run = _warm(plan, steps=50, batch=2)
    bptt = BPTT(plan.program(), lambda r, y: rate_mse(r['r']['spike'], y, DT))
    rec = bptt.record(run.state, duration=5.0 * u.ms, n0=run.n, trials=run.trials)
    res = run.cont(5.0 * u.ms)
    assert rec['r']['spike'].shape == (2, 50, 6)
    np.testing.assert_array_equal(rec['r']['spike'], np.asarray(res['r']['spike']))
    assert not np.array_equal(rec['r']['spike'][0], rec['r']['spike'][1])  # independent trials
    loss, grads = bptt.grad(run.state, None, 100.0 * u.Hz, duration=5.0 * u.ms, n0=run.n,
                            trials=run.trials)
    assert np.isfinite(float(loss)) and set(grads) == {'pops/h/tau', 'weights/w/0', 'weights/rec/0'}
    with pytest.raises(ValueError, match='trials=runner.trials'):
        bptt.grad(run.state, duration=5.0 * u.ms)
    with pytest.raises(ValueError, match='leading axis'):
        bptt.grad(run.state, duration=5.0 * u.ms, trials=jnp.arange(3))
    with pytest.raises(ValueError, match='1-D'):
        bptt.grad(run.state, duration=5.0 * u.ms, trials=0)


def test_run_continues_episodes(x64):
    plan = _noise_net().compile(precision='float64')
    bptt = BPTT(plan.program(), lambda r, y: rate_mse(r['r']['spike'], y, DT),
                braintools.optim.SGD(lr=1e-6))
    ref = Runner(plan.program())
    ref.simulate(5.0 * u.ms)
    w0 = np.asarray(u.get_mantissa(bptt.params['weights/w/0'].value)).copy()
    loss1, s1 = bptt.run(None, None, 100.0 * u.Hz, duration=5.0 * u.ms)
    for a, b in zip(jax.tree.leaves(s1), jax.tree.leaves(ref.state)):  # pre-update forward
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)
    assert not np.array_equal(np.asarray(u.get_mantissa(bptt.params['weights/w/0'].value)), w0)
    loss2, s2 = bptt.run(s1, None, 100.0 * u.Hz, duration=5.0 * u.ms, n0=50)
    assert np.isfinite(float(loss2)) and float(loss2) != float(loss1)


# ---------------------------------------------------------------------------
# checkpointing, truncation, training
# ---------------------------------------------------------------------------

def _rec_bptt(storage='dense', **kw):
    net = recurrent('LIF', storage, 1)
    return BPTT(net.compile(precision='float64').program(),
                lambda r, y: spike_count_mse(r['s_B']['spike'], y) + membrane_mse(r['v_A']['V'], 0.5),
                **kw)


def test_checkpointed_equals_plain(x64):
    xs, y = {'x': _xs(64, 6, p=0.5)}, jnp.full(10, 3.0)
    l0, g0 = _rec_bptt().grad(None, xs, y)
    for base in (2, 4, 16):
        l1, g1 = _rec_bptt(checkpoint=base).grad(None, xs, y)
        assert float(l1) == float(l0)
        for k in g0:
            np.testing.assert_allclose(u.get_mantissa(g1[k]), u.get_mantissa(g0[k]), rtol=1e-12,
                                       atol=1e-14)
    assert any(np.any(np.asarray(u.get_mantissa(g)) != 0) for g in g0.values())


def test_checkpointed_equals_plain_from_warm_state(x64):
    plan = _noise_net().compile(precision='float64')
    run = _warm(plan, steps=100)
    loss = lambda r, y: rate_mse(r['r']['spike'], y, DT) + membrane_mse(r['r']['V'], 0.5)
    kw = dict(duration=6.4 * u.ms, n0=run.n)
    l0, g0 = BPTT(plan.program(), loss).grad(run.state, None, 100.0 * u.Hz, **kw)
    l1, g1 = BPTT(plan.program(), loss, checkpoint=4).grad(run.state, None, 100.0 * u.Hz, **kw)
    assert float(l1) == float(l0)
    for k in g0:
        np.testing.assert_allclose(u.get_mantissa(g1[k]), u.get_mantissa(g0[k]), rtol=1e-12,
                                   atol=1e-14)


def test_truncation(x64):
    xs, y = {'x': _xs(40, 6, p=0.5)}, jnp.full(10, 3.0)
    l0, g0 = _rec_bptt(detach_reset=False).grad(None, xs, y)
    l1, g1 = _rec_bptt(detach_reset=False, truncation=40).grad(None, xs, y)
    l2, g2 = _rec_bptt(detach_reset=False, truncation=3).grad(None, xs, y)
    assert float(l0) == float(l1) == float(l2)  # forward unchanged
    for k in g0:
        np.testing.assert_allclose(u.get_mantissa(g1[k]), u.get_mantissa(g0[k]), rtol=1e-12)
    diff = [not np.allclose(u.get_mantissa(g2[k]), u.get_mantissa(g0[k])) for k in g0]
    assert any(diff)


def test_tiny_training_run_decreases_loss():
    net = Network(dt=DT)
    x = net.add(ExternalInput, 8, name='x')
    out = net.add(LIF, 5, name='out', surrogate=SIG)
    net.connect(x, out, port='delta', weight=0.2 * u.mV, trainable=True, name='w')
    net.observe(out, 'spike', name='rec')
    bptt = BPTT(net.compile().program(), lambda r, y: spike_count_mse(r['rec']['spike'], y),
                braintools.optim.Adam(lr=0.02))
    xs = {'x': _xs(100, 8, p=0.4)}
    target = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
    losses = np.asarray(bptt.fit(None, xs, target, n_epochs=40))
    assert losses.shape == (40,)
    final = float(bptt.loss(None, xs, target))
    assert final < 0.25 * losses[0]
    loss_step, _ = bptt.run(None, xs, target)
    assert float(loss_step) == pytest.approx(final)


def test_trains_neuron_parameter_with_poisson_drive():
    net = Network(dt=DT, seed=3)
    noise = net.add(poisson_generator, 10, params=dict(rate=500.0 * u.Hz), name='noise')
    out = net.add(LIF, 4, name='out', surrogate=SIG, trainable=('tau',))
    net.connect(noise, out, port='delta', weight=0.5 * u.mV, name='w')
    net.observe(out, 'spike', name='rec')
    bptt = BPTT(net.compile().program(), lambda r, y: rate_mse(r['rec']['spike'], y, DT),
                braintools.optim.Adam(lr=0.5))
    assert list(bptt.params) == ['pops/out/tau']
    l0, g = bptt.grad(None, None, 200.0 * u.Hz, duration=20.0 * u.ms)
    assert u.get_unit(g['pops/out/tau']) == u.ms
    assert np.all(np.asarray(u.get_mantissa(g['pops/out/tau'])) != 0)
    losses = bptt.fit(None, None, 200.0 * u.Hz, n_epochs=10, duration=20.0 * u.ms)
    assert float(losses[-1]) < float(l0)


def test_batched_episode_matches_unbatched(x64):
    xs1 = {'x': _xs(30, 6, p=0.5)}
    y = jnp.full(10, 2.0)
    b1 = _rec_bptt()
    l1, g1 = b1.grad(None, xs1, y)
    b2 = _rec_bptt()
    trials = Runner(b2.program, batch=2, rng='common').trials
    xs2 = {'x': jnp.stack([xs1['x'], xs1['x']])}
    l2, g2 = b2.grad(b2.program.init_state(2), xs2, y, trials=trials)
    assert float(l2) == pytest.approx(float(l1), rel=1e-12)
    for k in g1:
        np.testing.assert_allclose(u.get_mantissa(g2[k]), u.get_mantissa(g1[k]), rtol=1e-10)
    l3, _ = b2.grad(None, xs2, y, trials=trials)  # state0=None -> batched initial state
    assert float(l3) == float(l2)
    with pytest.raises(ValueError, match='leading shape'):
        b2.grad(None, xs1, y, trials=trials)


def test_learner_options_do_not_mutate_program_or_models():
    plan = recurrent('LIF', 'dense', 1).compile()
    prog = plan.program()
    b = BPTT(prog, lambda r, y: 0.0 * jnp.sum(r['s_B']['spike']))
    assert b.program is not prog and b.program.detach_reset is True and b.plan is plan
    assert prog.detach_reset is None and prog.spikes == 'hard'
    assert all(m.detach_reset is None for m in plan.models)
    b2 = BPTT(prog, lambda r, y: jnp.sum(r['s_B']['spike']), detach_reset=None)
    assert b2.program is prog  # nothing to change
    b3 = BPTT(prog, lambda r, y: jnp.sum(r['s_B']['spike']), detach_reset=None, spikes='relaxed')
    assert b3.program.spikes == 'relaxed' and b3.program.detach_reset is None


def test_detach_reset_option_changes_gradients(x64):
    xs, y = {'x': _xs(40, 6, p=0.5)}, jnp.full(10, 3.0)
    _, g_det = _rec_bptt(detach_reset=True).grad(None, xs, y)
    _, g_att = _rec_bptt(detach_reset=False).grad(None, xs, y)
    assert any(not np.allclose(u.get_mantissa(g_det[k]), u.get_mantissa(g_att[k])) for k in g_det)


def test_quantity_inputs_and_port_mapping():
    """``data`` given per port as a Quantity reaches a neuron's input directly."""
    net = Network(dt=DT)
    h = net.add(LIF, 3, name='h', surrogate=SIG, trainable=('tau',))
    net.observe(h, 'V', name='v')
    bptt = BPTT(net.compile().program(), lambda r, y: membrane_mse(r['v']['V'], y))
    xs = {'h': {'I': jnp.ones((20, 3)) * 2.0 * u.mA}}
    l, g = bptt.grad(None, xs, 0.0)
    assert float(l) > 0 and np.all(np.asarray(u.get_mantissa(g['pops/h/tau'])) != 0)
    l2, _ = bptt.grad(None, xs, 0.0, duration=2.0 * u.ms)  # consistent duration accepted
    assert float(l2) == float(l)
    with pytest.raises(ValueError, match='duration is 10 steps'):
        bptt.grad(None, xs, 0.0, duration=1.0 * u.ms)
    with pytest.raises(ValueError, match='input ports'):
        bptt.grad(None, {'h': jnp.ones((20, 3))}, 0.0)
    with pytest.raises(KeyError):
        bptt.grad(None, {'nope': jnp.ones((20, 3))}, 0.0)
    with pytest.raises(TypeError, match='mapping'):
        bptt.grad(None, [1, 2], 0.0)
    with pytest.raises(ValueError, match='leading shape'):
        bptt.grad(None, {'h': {'I': jnp.ones((5,)) * u.mA}}, 0.0, trials=jnp.arange(2))
    with pytest.raises(ValueError, match='other inputs'):
        net2 = Network(dt=DT)
        a = net2.add(ExternalInput, 2, name='a')
        b = net2.add(ExternalInput, 2, name='b')
        o = net2.add(LIF, 2, name='o', surrogate=SIG)
        net2.connect(a, o, port='delta', weight=1.0 * u.mV, trainable=True)
        net2.connect(b, o, port='delta', weight=1.0 * u.mV)
        net2.observe(o, 'V', name='v')
        BPTT(net2.compile().program(), lambda r, y: 0.0).grad(
            None, {'a': jnp.ones((5, 2)), 'b': jnp.ones((6, 2))})


# ---------------------------------------------------------------------------
# errors and helpers
# ---------------------------------------------------------------------------

class _NotDifferentiable(LIF):
    capabilities = Capabilities(differentiable=False)


def test_construction_errors():
    net = Network(dt=DT)
    h = net.add(LIF, 2, name='h')
    net.observe(h, 'spike', name='r')
    with pytest.raises(ValueError, match='no trainable'):
        BPTT(net.compile().program(), lambda r, y: 0.0)

    net = Network(dt=DT)
    net.add(LIF, 2, name='h', trainable=('tau',))
    with pytest.raises(ValueError, match='observers'):
        BPTT(net.compile().program(), lambda r, y: 0.0)

    net = Network(dt=DT)
    x = net.add(ExternalInput, 2, name='x')
    h = net.add(_NotDifferentiable, 2, name='h')
    net.connect(x, h, port='delta', weight=1.0 * u.mV, trainable=True)
    net.observe(h, 'spike', name='r')
    with pytest.raises(ValueError, match='not differentiable'):
        BPTT(net.compile().program(), lambda r, y: 0.0)

    plan = recurrent('LIF', 'dense', 1).compile()
    with pytest.raises(TypeError, match='Program'):
        BPTT(plan, lambda r, y: 0.0)  # a plan is not a program
    b = BPTT(plan.program(), lambda r, y: jnp.sum(r['s_B']['spike']))
    with pytest.raises(RuntimeError, match='optimizer'):
        b.run(None, {'x': _xs(10, 6)})
    with pytest.raises(RuntimeError, match='optimizer'):
        b.fit(None, {'x': _xs(10, 6)}, n_epochs=2)
    with pytest.raises(ValueError, match='at least one step'):
        b.grad(None, duration=0.0 * u.ms)
    with pytest.raises(ValueError, match='data or a duration'):
        b.grad(None)
    with pytest.raises(ValueError, match='n0'):
        b.grad(None, {'x': _xs(10, 6)}, n0=jnp.arange(2))
    with pytest.raises(ValueError, match='not a state of this program'):
        b.grad({'pops': {}}, {'x': _xs(10, 6)})


def test_constructor_and_loss_validation():
    prog = recurrent('LIF', 'dense', 1).compile().program()
    with pytest.raises(TypeError, match='callable'):
        BPTT(prog, 3.0)
    with pytest.raises(ValueError, match='checkpoint'):
        BPTT(prog, lambda r, y: 0.0, checkpoint=1)
    with pytest.raises(ValueError, match='truncation'):
        BPTT(prog, lambda r, y: 0.0, truncation=0)
    with pytest.raises(ValueError, match='truncation'):
        BPTT(prog, lambda r, y: 0.0, truncation=True)
    with pytest.raises(TypeError, match='detach_reset'):
        BPTT(prog, lambda r, y: 0.0, detach_reset='yes')
    with pytest.raises(ValueError, match='spikes'):
        BPTT(prog, lambda r, y: 0.0, spikes='soft')
    b = BPTT(prog, lambda r, y: r['s_B']['spike'])  # not a scalar
    with pytest.raises(ValueError, match='scalar'):
        b.loss(None, {'x': _xs(10, 6)})
    b2 = BPTT(recurrent('LIF', 'dense', 1).compile().program(),
              lambda r, y: jnp.sum(r['s_B']['spike']) * u.UNITLESS, braintools.optim.SGD(lr=0.1))
    assert np.isfinite(float(b2.loss(None, {'x': _xs(10, 6)})))
    with pytest.raises(ValueError, match='n_epochs'):
        b2.fit(None, {'x': _xs(10, 6)}, n_epochs=0)
    with pytest.raises(ValueError, match='n_epochs'):
        b2.fit(None, {'x': _xs(10, 6)}, n_epochs=None)


def test_relaxed_program_needs_surrogates():
    net = Network(dt=DT)
    x = net.add(ExternalInput, 2, name='x')
    h = net.add(LIF, 2, name='h', surrogate=None, trainable=('tau',))
    net.connect(x, h, port='delta', weight=1.0 * u.mV)
    net.observe(h, 'V', name='v')
    plan = net.compile()
    with pytest.raises(ValueError, match=r"spikes='relaxed'.*h \(LIF"):
        Program(plan, spikes='relaxed')
    with pytest.raises(ValueError, match='surrogate'):
        BPTT(plan.program(), lambda r, y: 0.0, spikes='relaxed')


def test_trainable_states_keys():
    net = recurrent('LIF', 'csr', 2)
    keys = sorted(trainable_states(net.compile()))
    assert keys == ['pops/A/tau', 'weights/AA/0', 'weights/AB/0', 'weights/xA/0']
    # per-edge delays >= 1 -> one scatter group, hence one ParamState (K1a-4)
    net = Network(dt=DT)
    a = net.add(LIF, 4, name='a')
    net.connect(a, a, fixed_indegree(2), port='delta', weight=0.1 * u.mV, trainable=True, name='p',
                delay=jnp.array([0.1, 0.2] * 4) * u.ms)
    assert sorted(trainable_states(net.compile())) == ['weights/p/0']


def test_loss_helpers():
    s = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # (T=3, n=2): counts [2, 2]
    assert float(spike_count_mse(s, jnp.array([2.0, 0.0]))) == pytest.approx(2.0)
    # rates: 2 spikes in 3 steps of 0.1 ms = 6666.67 Hz
    assert float(rate_mse(s, 6666.666666 * u.Hz, DT)) == pytest.approx(0.0, abs=1e-4)
    assert float(rate_mse(s, jnp.array([0.0, 0.0]), 1.0 * u.second)) == pytest.approx(4.0 / 9.0)
    assert float(membrane_mse(s * u.mV, 1.0 * u.mV)) == pytest.approx(2.0 / 6.0)
    batched = jnp.stack([s, s])
    assert float(spike_count_mse(batched, jnp.array([2.0, 2.0]))) == 0.0


def test_program_ext_rejects_unknown_population():
    net = Network(dt=DT)
    net.add(LIF, 2, name='h')
    prog = net.compile().program()
    with pytest.raises(KeyError, match='unknown population'):
        prog.step(prog.read_params(), prog.init_state(), 0, jnp.uint32(0), ext={'nope': {}})
    st, cur = prog.step(prog.read_params(), prog.init_state(), 0, jnp.uint32(0),
                        ext={'h': {'delta': 2.0 * u.mV}})
    np.testing.assert_array_equal(np.asarray(cur[0]['spike']), [1.0, 1.0])
    assert isinstance(brainstate.ParamState, type)


def test_cut_stops_float_gradients_only():
    tree = {'v': jnp.array([1.0, 2.0]), 'n': jnp.array([1, 2], jnp.int32)}
    out = _cut(tree, jnp.bool_(True))
    assert out['n'].dtype == jnp.int32
    g = jax.grad(lambda v: jnp.sum(_cut({'v': v}, jnp.bool_(True))['v']))(tree['v'])
    np.testing.assert_array_equal(g, [0.0, 0.0])
    g = jax.grad(lambda v: jnp.sum(_cut({'v': v}, jnp.bool_(False))['v']))(tree['v'])
    np.testing.assert_array_equal(g, [1.0, 1.0])
