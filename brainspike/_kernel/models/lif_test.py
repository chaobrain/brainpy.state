# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the K0 ``LIF``: parity with ``_brainpy.LIF``, timing, gradients, braintrace."""
import inspect

import brainstate
import braintools
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._brainpy.lif import LIF as OldLIF
from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import Exact
from brainspike._kernel.models import LIF
from brainspike._kernel.models._testing import rollout
from brainspike._kernel.model import latch_name
from brainspike._kernel.ports import Current

DT = 0.1 * u.ms


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _inputs(n_steps, n, seed=0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    I = jax.random.uniform(k1, (n_steps, n), minval=0.0, maxval=3.0)       # mA
    delta = jnp.where(jax.random.uniform(k2, (n_steps, n)) < 0.05, 0.4, 0.0)  # mV
    return I, delta


def _old_run(n_steps, n, I_seen, delta, spk_reset='soft', **params):
    with brainstate.environ.context(dt=DT):
        old = OldLIF(n, spk_reset=spk_reset, **params)
        old.init_state()

        def body(i):
            old.add_delta_input('d', delta[i] * u.mV)
            spk = old.update(I_seen[i] * u.mA)
            return spk, old.V.value

        return brainstate.transform.for_loop(body, jnp.arange(n_steps))


def _new_run(n_steps, n, I, delta, **kw):
    m = LIF(n, **kw)
    out, tr = rollout(m, n_steps, DT, {'I': I * u.mA, 'delta': delta * u.mV}, record=('V',))
    return out['spike'], tr['V']


class TestParityWithBrainPyLIF:
    @pytest.mark.parametrize('spk_reset', ['soft', 'hard'])
    @pytest.mark.parametrize('precision', ['float32', 'float64'])
    def test_trajectory_and_spikes(self, spk_reset, precision):
        n_steps, n = 600, 16
        params = dict(tau=[5.0, 8.0, 12.0, 20.0] * 4 * u.ms, V_th=1.0 * u.mV, R=1.0 * u.ohm)
        with jax.enable_x64(precision == 'float64'):
            I, delta = _inputs(n_steps, n)
            # LIF.I is latch=False (D-a): both models see this step's input
            old_spk, old_V = _old_run(n_steps, n, I, delta, spk_reset, tau=params['tau'])
            new_spk, new_V = _new_run(n_steps, n, I, delta, params=dict(tau=params['tau']),
                                      spk_reset=spk_reset, precision=precision)
        old_spk = np.asarray(old_spk)
        new_spk = np.asarray(new_spk)
        assert new_spk.sum() > 20
        np.testing.assert_array_equal(new_spk, old_spk)
        # old V is pre-reset at spike steps; ours is post-reset
        old_V = np.asarray(old_V.to_decimal(u.mV))
        if spk_reset == 'soft':
            old_post = old_V - 1.0 * old_spk
        else:
            old_post = np.where(old_spk > 0, 0.0, old_V)
        rtol = 1e-5 if precision == 'float32' else 1e-12
        np.testing.assert_allclose(np.asarray(new_V.to_decimal(u.mV)), old_post, rtol=rtol, atol=rtol)

    def test_defaults_match(self):
        old = OldLIF(1)
        for k in ('R', 'tau', 'V_th', 'V_reset', 'V_rest'):
            assert u.math.allclose(LIF.params[k].default, getattr(old, k))
        assert LIF.state_decl['V'].value == 0.0 * u.mV


class TestTiming:
    def test_delta_before_threshold(self):
        m = LIF(2)
        out = m.update(StepContext.create(0.0 * u.ms, DT), {'delta': [0.5, 1.5] * u.mV})
        np.testing.assert_array_equal(out['spike'], [0.0, 1.0])
        np.testing.assert_allclose(m.dyn_states['V'].value.to_decimal(u.mV), [0.5, 0.5], rtol=1e-6)

    def test_current_port_declared_same_step(self):
        # D-a / K1a-2: the BrainPy-family LIF declares latch=False explicitly
        kind = LIF.inputs['I']
        assert isinstance(kind, Current) and kind.latch is False
        assert 'latch=False' in inspect.getsource(LIF)
        assert latch_name('I') not in LIF.all_state_decl()
        assert latch_name('I') not in LIF(1).dyn_states

    def test_current_acts_in_same_step(self, x64):
        m = LIF(1, precision='float64')
        m.update(StepContext.create(0.0 * u.ms, DT), {'I': 1.0 * u.mA})
        expected = 1.0 * (1 - np.exp(-0.1 / 5.0))
        assert float(m.dyn_states['V'].value.to_decimal(u.mV)[0]) == pytest.approx(expected, rel=1e-13)
        # no input in the next step: pure decay, nothing carried over
        m.update(StepContext.create(DT, DT, 1), {})
        assert float(m.dyn_states['V'].value.to_decimal(u.mV)[0]) == pytest.approx(
            expected * np.exp(-0.1 / 5.0), rel=1e-13)

    def test_hard_reset_and_validation(self):
        m = LIF(1, spk_reset='hard', params=dict(V_reset=-0.5 * u.mV))
        m.update(StepContext.create(0.0 * u.ms, DT), {'delta': 3.0 * u.mV})
        assert float(m.dyn_states['V'].value.to_decimal(u.mV)[0]) == -0.5
        assert LIF.integrator == Exact()
        with pytest.raises(ValueError, match='spk_reset'):
            LIF(1, spk_reset='zero')

    def test_heterogeneous_params_and_zero_size(self):
        m = LIF(3, params=dict(V_th=[0.5, 1.0, 2.0] * u.mV))
        out = m.update(StepContext.create(0.0 * u.ms, DT), {'delta': 0.9 * u.mV})
        np.testing.assert_array_equal(out['spike'], [1.0, 0.0, 0.0])
        assert LIF(0).update(StepContext.create(0.0 * u.ms, DT))['spike'].shape == (0,)

    def test_jit_vmap(self):
        m = LIF(4)
        s, p = m.read()
        ctx = StepContext.create(0.0 * u.ms, DT)

        @jax.jit
        def f(d):
            inp = m.prepare_inputs({'delta': d * u.mV})
            return m.step(s, p, inp, ctx)[1]['spike']

        out = jax.vmap(f)(jnp.array([0.5, 1.5]))
        np.testing.assert_array_equal(out, [[0.0] * 4, [1.0] * 4])


class TestGradients:
    def _loss(self, theta, n_steps=300):
        """Sum of V over a spiking rollout as a function of (tau, I) -- detach_reset."""
        m = LIF(1, precision='float64', detach_reset=True, surrogate=braintools.surrogate.Sigmoid())
        m.param_states['tau'].value = jnp.reshape(theta[0], (1,)) * u.ms

        def body(i):
            m.update(StepContext(t=i * DT, dt=DT, step=i), {'I': theta[1] * u.mA})
            return m.dyn_states['V'].value.to_decimal(u.mV)[0]

        return jnp.sum(brainstate.transform.for_loop(body, jnp.arange(n_steps)))

    def test_bptt_vs_finite_differences(self, x64):
        theta = jnp.array([5.0, 1.7])
        g = jax.grad(self._loss)(theta)
        eps = np.array([1e-5, 1e-6])
        fd = [(self._loss(theta.at[i].add(eps[i])) - self._loss(theta.at[i].add(-eps[i]))) / (2 * eps[i])
              for i in range(2)]
        assert np.all(np.isfinite(np.asarray(g)))
        np.testing.assert_allclose(g, fd, rtol=1e-4)

    def test_surrogate_spike_gradient(self, x64):
        def n_spikes(I):
            m = LIF(1, precision='float64', surrogate=braintools.surrogate.Sigmoid())

            def body(i):
                return m.update(StepContext(t=i * DT, dt=DT, step=i), {'I': I * u.mA})['spike'][0]

            return jnp.sum(brainstate.transform.for_loop(body, jnp.arange(400)))

        assert float(n_spikes(1.7)) >= 1.0
        assert float(jax.grad(n_spikes)(1.7)) > 0.0

    def test_hard_surrogate_none_gives_zero_gradient(self):
        m = LIF(1, surrogate=None)
        assert m.spiking.surrogate is None and LIF.spiking.surrogate is not None

        def f(d):
            return m.step(*m.read(), m.prepare_inputs({'delta': d * u.mV}),
                          StepContext.create(0.0 * u.ms, DT))[1]['spike'][0]

        assert float(jax.grad(f)(1.5)) == 0.0


class _Net(brainstate.nn.Module):
    def __init__(self, n_in, n):
        super().__init__()
        self.W = brainstate.ParamState(jax.random.uniform(jax.random.key(0), (n_in, n), minval=0.5, maxval=1.5))
        self.neu = LIF(n, params=dict(tau=10.0 * u.ms), init=dict(V=0.5 * u.mV),
                       surrogate=braintools.surrogate.Sigmoid())

    def init_state(self, *args, **kwargs):
        self.neu.init_state()

    def update(self, x):
        I = braintrace.matmul(x, self.W.value) * u.mA
        return self.neu.update(StepContext(t=0.0 * u.ms, dt=DT, step=0), {'I': I})['spike']


class TestBraintrace:
    @pytest.mark.parametrize('algorithm', ['eprop', 'd_rtrl'])
    def test_compile_nonzero_weight_gradients(self, algorithm):
        net = _Net(3, 4)
        learner = braintrace.compile(net, algorithm, jnp.ones(3))
        xs = jax.random.uniform(jax.random.key(1), (60, 3)) * 4.0
        ys = jnp.ones((60, 4))
        grads = learner.etrace_grad(xs, ys, step_fn=lambda x, y: jnp.mean((learner(x) - y) ** 2))
        g = jax.tree.leaves(grads)
        assert len(g) == 1 and g[0].shape == (3, 4)
        assert np.all(np.isfinite(np.asarray(g[0])))
        assert float(jnp.abs(g[0]).sum()) > 0.0


def test_replace_rule_validates_fields():
    from brainspike._kernel.models import replace_rule
    r = replace_rule(LIF.spiking, detach_reset=True)
    assert r.detach_reset is True and r.condition is LIF.spiking.condition
    with pytest.raises(TypeError, match='unknown SpikeRule field'):
        replace_rule(LIF.spiking, threshold=1.0)
