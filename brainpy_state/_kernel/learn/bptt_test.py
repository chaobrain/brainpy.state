# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the BPTT learner, including gate G6 (gradients vs central finite differences).

G6 is checked in two regimes (see ``docs/specs/2026-10-22-k0-t9-learners.md``):

* **hard forward** (the production pass: exact spike counts, surrogate backward) on
  feed-forward nets with ``detach_reset=True`` and a membrane loss -- no gradient
  flows through a spike, so the surrogate BPTT gradient *is* the exact derivative of
  the piecewise-smooth loss and must match finite differences;
* **relaxed forward** (:func:`~brainpy_state._kernel.spiking.relaxed_spikes`:
  spikes = the smooth surrogate primitive, soft jumps) on two-layer recurrent nets
  with every weight and a neuron parameter trainable and ``detach_reset=False`` --
  this exercises the spike payload VJPs through Dense / CSR storages, delay rings and
  the reset path.
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
from brainpy_state._kernel.spiking import relaxed_spikes

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


def feedforward(model_name, storage, delay_steps, n=20, n_in=6):
    """ExternalInput -> 20 neurons (the G6 hard-forward net)."""
    model, port, w_in, _, params, tparam = MODELS[model_name]
    net = Network(dt=DT)
    x = net.add(ExternalInput, n_in, name='x')
    h = net.add(model, n, params=params, name='h', surrogate=SIG, trainable=(tparam,))
    w = u.Quantity(np.linspace(0.5, 1.5, n_in * n) * u.get_mantissa(w_in), unit=u.get_unit(w_in))
    net.connect(x, h, port=port, weight=w, delay=delay_steps * DT, storage=storage,
                trainable=True, name='xh')
    net.observe(h, 'spike', name='rec_h')
    net.observe(h, 'V' if model is LIF else 'V_m', name='v_h')
    return net


def recurrent(model_name, storage, delay_steps, n_in=6):
    """x -> A (10) -> B (10) plus A -> A (the G6 relaxed net, 20 neurons)."""
    model, port, w_in, w_rec, params, tparam = MODELS[model_name]
    net = Network(dt=DT)
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


def _directional_fd(bptt, duration, xs, targets, direction, eps):
    states = bptt.params
    base = {k: s.value for k, s in states.items()}
    _shift(states, direction, eps)
    lp = float(bptt.loss(duration, xs=xs, targets=targets))
    for k, s in states.items():
        s.value = base[k]
    _shift(states, direction, -eps)
    lm = float(bptt.loss(duration, xs=xs, targets=targets))
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


def _check_fd(bptt, duration, xs, targets, eps=1e-6, rtol=1e-4):
    loss, grads = bptt.grad(duration, xs=xs, targets=targets)
    assert np.isfinite(float(loss))
    for d in _directions(bptt, grads):
        ad = sum(float(np.sum(np.asarray(u.get_mantissa(grads[k])) * d[k])) for k in d)
        fd = _directional_fd(bptt, duration, xs, targets, d, eps)
        assert ad != 0.0
        assert abs(ad - fd) <= rtol * abs(fd), (ad, fd)
    return grads


# ---------------------------------------------------------------------------
# G6
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model_name', list(MODELS))
@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('delay_steps', [0, 2])
def test_g6_hard_forward_vs_finite_differences(x64, model_name, storage, delay_steps):
    net = feedforward(model_name, storage, delay_steps)
    vname = 'v_h'
    target = -60.0 if model_name == 'iaf_psc_alpha' else 0.8
    bptt = BPTT(lambda r, y: membrane_mse(r[vname][list(r[vname])[0]], y), detach_reset=True)
    plan = net.compile(precision='float64')
    assert plan.projection('xh').groups[0].storage.kind == storage
    plan.runner(learner=bptt)
    xs = {'x': _xs(60, 6)}
    rec = bptt.record(6.0 * u.ms, xs=xs)
    assert rec['rec_h']['spike'].shape == (60, 20)
    assert float(np.sum(rec['rec_h']['spike'])) > 0  # the nets spike: gradients cross resets
    _check_fd(bptt, 6.0 * u.ms, xs, target)


@pytest.mark.parametrize('model_name', list(MODELS))
@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('delay_steps', [0, 2])
def test_g6_relaxed_vs_finite_differences(x64, model_name, storage, delay_steps):
    net = recurrent(model_name, storage, delay_steps)
    vname = 'v_A'
    v_target = -62.0 if model_name == 'iaf_psc_alpha' else 0.5

    def loss(r, y):
        v = r[vname][list(r[vname])[0]]
        return spike_count_mse(r['s_B']['spike'], y) + 0.1 * membrane_mse(v, v_target)

    bptt = BPTT(loss, detach_reset=False)
    plan = net.compile(precision='float64')
    for pp in plan.projections:  # relaxed spikes are fractional: use the exact float path
        pp.path = 'float'
    plan.runner(learner=bptt)
    assert set(bptt.params) == {f'pops/A/{MODELS[model_name][5]}', 'weights/xA/0',
                                'weights/AA/0', 'weights/AB/0'}
    with relaxed_spikes():
        grads = _check_fd(bptt, 6.0 * u.ms, {'x': _xs(60, 6)}, jnp.full(10, 2.0))
    for k, g in grads.items():
        assert np.any(np.asarray(u.get_mantissa(g)) != 0), k


# ---------------------------------------------------------------------------
# checkpointing, truncation, training
# ---------------------------------------------------------------------------

def _rec_bptt(storage='dense', **kw):
    net = recurrent('LIF', storage, 1)
    bptt = BPTT(lambda r, y: spike_count_mse(r['s_B']['spike'], y)
                + membrane_mse(r['v_A']['V'], 0.5), **kw)
    net.compile(precision='float64').runner(learner=bptt)
    return bptt


def test_checkpointed_equals_plain(x64):
    xs, y = {'x': _xs(64, 6, p=0.5)}, jnp.full(10, 3.0)
    l0, g0 = _rec_bptt().grad(6.4 * u.ms, xs=xs, targets=y)
    for base in (2, 4, 16):
        l1, g1 = _rec_bptt(checkpoint=base).grad(6.4 * u.ms, xs=xs, targets=y)
        assert float(l1) == float(l0)
        for k in g0:
            np.testing.assert_allclose(u.get_mantissa(g1[k]), u.get_mantissa(g0[k]), rtol=1e-12,
                                       atol=1e-14)
    assert any(np.any(np.asarray(u.get_mantissa(g)) != 0) for g in g0.values())


def test_truncation(x64):
    xs, y = {'x': _xs(40, 6, p=0.5)}, jnp.full(10, 3.0)
    l0, g0 = _rec_bptt(detach_reset=False).grad(4.0 * u.ms, xs=xs, targets=y)
    l1, g1 = _rec_bptt(detach_reset=False, truncation=40).grad(4.0 * u.ms, xs=xs, targets=y)
    l2, g2 = _rec_bptt(detach_reset=False, truncation=3).grad(4.0 * u.ms, xs=xs, targets=y)
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
    bptt = BPTT(lambda r, y: spike_count_mse(r['rec']['spike'], y), braintools.optim.Adam(lr=0.02))
    net.compile().runner(learner=bptt)
    xs = {'x': _xs(100, 8, p=0.4)}
    target = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
    losses = np.asarray(bptt.fit(10.0 * u.ms, 40, xs=xs, targets=target))
    assert losses.shape == (40,)
    final = float(bptt.loss(10.0 * u.ms, xs=xs, targets=target))
    assert final < 0.25 * losses[0]
    loss_step = float(bptt.step(10.0 * u.ms, xs=xs, targets=target))
    assert loss_step == pytest.approx(final)


def test_trains_neuron_parameter_with_poisson_drive():
    net = Network(dt=DT, seed=3)
    noise = net.add(poisson_generator, 10, params=dict(rate=500.0 * u.Hz), name='noise')
    out = net.add(LIF, 4, name='out', surrogate=SIG, trainable=('tau',))
    net.connect(noise, out, port='delta', weight=0.5 * u.mV, name='w')
    net.observe(out, 'spike', name='rec')
    bptt = BPTT(lambda r, y: rate_mse(r['rec']['spike'], y, DT), braintools.optim.Adam(lr=0.5))
    net.compile().runner(learner=bptt)
    assert list(bptt.params) == ['pops/out/tau']
    l0, g = bptt.grad(20.0 * u.ms, targets=200.0 * u.Hz)
    assert u.get_unit(g['pops/out/tau']) == u.ms
    assert np.all(np.asarray(u.get_mantissa(g['pops/out/tau'])) != 0)
    losses = bptt.fit(20.0 * u.ms, 10, targets=200.0 * u.Hz)
    assert float(losses[-1]) < float(l0)


def test_batched_runner_matches_unbatched(x64):
    xs1 = {'x': _xs(30, 6, p=0.5)}
    y = jnp.full(10, 2.0)
    b1 = _rec_bptt()
    l1, g1 = b1.grad(3.0 * u.ms, xs=xs1, targets=y)
    net = recurrent('LIF', 'dense', 1)
    b2 = BPTT(lambda r, t: spike_count_mse(r['s_B']['spike'], t) + membrane_mse(r['v_A']['V'], 0.5))
    run = net.compile(precision='float64').runner(batch=2, rng='common', learner=b2)
    assert run.learner is b2
    xs2 = {'x': jnp.stack([xs1['x'], xs1['x']])}
    l2, g2 = b2.grad(3.0 * u.ms, xs=xs2, targets=y)
    assert float(l2) == pytest.approx(float(l1), rel=1e-12)
    for k in g1:
        np.testing.assert_allclose(u.get_mantissa(g2[k]), u.get_mantissa(g1[k]), rtol=1e-10)
    with pytest.raises(ValueError, match='leading shape'):
        b2.grad(3.0 * u.ms, xs=xs1, targets=y)


def test_detach_reset_is_set_on_models():
    net = recurrent('LIF', 'dense', 1)
    plan = net.compile()
    plan.runner(learner=BPTT(lambda r, y: 0.0 * jnp.sum(r['s_B']['spike'])))
    assert all(m.detach_reset is True for m in plan.models)
    plan2 = recurrent('LIF', 'dense', 1).compile()
    plan2.runner(learner=BPTT(lambda r, y: jnp.sum(r['s_B']['spike']), detach_reset=None))
    assert all(m.detach_reset is None for m in plan2.models)


def test_quantity_inputs_and_port_mapping():
    """``xs`` given per port as a Quantity reaches a neuron's input directly."""
    net = Network(dt=DT)
    h = net.add(LIF, 3, name='h', surrogate=SIG, trainable=('tau',))
    net.observe(h, 'V', name='v')
    bptt = BPTT(lambda r, y: membrane_mse(r['v']['V'], y))
    net.compile().runner(learner=bptt)
    xs = {'h': {'I': jnp.ones((20, 3)) * 2.0 * u.mA}}
    l, g = bptt.grad(2.0 * u.ms, xs=xs, targets=0.0)
    assert float(l) > 0 and np.all(np.asarray(u.get_mantissa(g['pops/h/tau'])) != 0)
    with pytest.raises(ValueError, match='input ports'):
        bptt.grad(2.0 * u.ms, xs={'h': jnp.ones((20, 3))}, targets=0.0)
    with pytest.raises(KeyError):
        bptt.grad(2.0 * u.ms, xs={'nope': jnp.ones((20, 3))}, targets=0.0)
    with pytest.raises(TypeError, match='mapping'):
        bptt.grad(2.0 * u.ms, xs=[1, 2], targets=0.0)
    with pytest.raises(ValueError, match='leading shape'):
        bptt.grad(2.0 * u.ms, xs={'h': {'I': jnp.ones((5, 3)) * u.mA}}, targets=0.0)


# ---------------------------------------------------------------------------
# errors and helpers
# ---------------------------------------------------------------------------

class _NotDifferentiable(LIF):
    capabilities = Capabilities(differentiable=False)


def test_attach_errors():
    net = Network(dt=DT)
    h = net.add(LIF, 2, name='h')
    net.observe(h, 'spike', name='r')
    with pytest.raises(ValueError, match='no trainable'):
        net.compile().runner(learner=BPTT(lambda r, y: 0.0))

    net = Network(dt=DT)
    net.add(LIF, 2, name='h', trainable=('tau',))
    with pytest.raises(ValueError, match='observers'):
        net.compile().runner(learner=BPTT(lambda r, y: 0.0))

    net = Network(dt=DT)
    x = net.add(ExternalInput, 2, name='x')
    h = net.add(_NotDifferentiable, 2, name='h')
    net.connect(x, h, port='delta', weight=1.0 * u.mV, trainable=True)
    net.observe(h, 'spike', name='r')
    with pytest.raises(ValueError, match='not differentiable'):
        net.compile().runner(learner=BPTT(lambda r, y: 0.0))

    b = BPTT(lambda r, y: jnp.sum(r['s_B']['spike']))
    run = recurrent('LIF', 'dense', 1).compile().runner(learner=b)
    with pytest.raises(RuntimeError, match='already attached'):
        b.attach(run)
    with pytest.raises(RuntimeError, match='optimizer'):
        b.step(1.0 * u.ms, xs={'x': _xs(10, 6)})
    with pytest.raises(ValueError, match='at least one step'):
        b.grad(0.0 * u.ms)
    with pytest.raises(RuntimeError, match='attach'):
        BPTT(lambda r, y: 0.0).loss(1.0 * u.ms)


def test_constructor_and_loss_validation():
    with pytest.raises(TypeError, match='callable'):
        BPTT(3.0)
    with pytest.raises(ValueError, match='checkpoint'):
        BPTT(lambda r, y: 0.0, checkpoint=1)
    with pytest.raises(ValueError, match='truncation'):
        BPTT(lambda r, y: 0.0, truncation=0)
    with pytest.raises(ValueError, match='truncation'):
        BPTT(lambda r, y: 0.0, truncation=True)
    with pytest.raises(TypeError, match='detach_reset'):
        BPTT(lambda r, y: 0.0, detach_reset='yes')
    b = BPTT(lambda r, y: r['s_B']['spike'])  # not a scalar
    recurrent('LIF', 'dense', 1).compile().runner(learner=b)
    with pytest.raises(ValueError, match='scalar'):
        b.loss(1.0 * u.ms, xs={'x': _xs(10, 6)})
    b2 = BPTT(lambda r, y: jnp.sum(r['s_B']['spike']) * u.UNITLESS, braintools.optim.SGD(lr=0.1))
    recurrent('LIF', 'dense', 1).compile().runner(learner=b2)
    assert np.isfinite(float(b2.loss(1.0 * u.ms, xs={'x': _xs(10, 6)})))
    with pytest.raises(ValueError, match='n_epochs'):
        b2.fit(1.0 * u.ms, 0, xs={'x': _xs(10, 6)})
    with pytest.raises(ValueError, match='n_epochs'):
        b2.fit(1.0 * u.ms, None, xs={'x': _xs(10, 6)})


def test_trainable_states_keys():
    net = recurrent('LIF', 'csr', 2)
    keys = sorted(trainable_states(net.compile()))
    assert keys == ['pops/A/tau', 'weights/AA/0', 'weights/AB/0', 'weights/xA/0']
    # per-edge delays -> one group (and one ParamState) per delay value
    net = Network(dt=DT)
    a = net.add(LIF, 4, name='a')
    net.connect(a, a, fixed_indegree(2), port='delta', weight=0.1 * u.mV, trainable=True, name='p',
                delay=jnp.array([0.1, 0.2] * 4) * u.ms)
    assert sorted(trainable_states(net.compile())) == ['weights/p/0', 'weights/p/1']


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
    prog = net.compile().runner().program
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
