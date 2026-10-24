# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""End-to-end tests of the NeuronModel contract with toy models.

The toy LIF validates the generated step (§7.1 timing, analytic trajectory and
spike timing, rollouts, gradients, braintrace); an aeif-like toy validates
multi-spike steps and multi-state jumps; a conductance toy validates gates
evaluated inside the substeps.
"""
import math

import brainstate
import braintools
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import RK, Exact, ExpEuler
from brainspike._kernel.model import (
    Capabilities,
    Device,
    NeuronModel,
    conductance_current,
    latch_name,
)
from brainspike._kernel.params import Init, Param, Params, Record, States
from brainspike._kernel.ports import (
    Conductance,
    Current,
    Event,
    PortError,
    Ports,
    TimedEvent,
    TimedEventPayload,
)
from brainspike._kernel.spiking import Block, Clamp, SpikeRule
from brainspike._kernel.units import UnitMismatchError

DT = 0.1 * u.ms


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


class ToyLIF(NeuronModel):
    """tau dV/dt = E_L - V + R I; V >= V_th -> V_reset, clamp t_ref."""

    params = Params(tau=10.0 * u.ms, V_th=-50.0 * u.mV, V_reset=-70.0 * u.mV,
                    E_L=-70.0 * u.mV, R=0.1 * u.Gohm, t_ref=2.0 * u.ms)
    states = States(V=Init(lambda p: p.E_L))
    inputs = Ports(I=Current(), w=Event(unit=u.mV))
    outputs = Ports(spike=Event())
    spiking = SpikeRule(condition=lambda s, p: (s.V - p.V_th) / (p.V_th - p.V_reset),
                        jump=lambda s, p: s.replace(V=p.V_reset),
                        refractory=Clamp('V', 't_ref'),
                        surrogate=braintools.surrogate.Sigmoid())
    integrator = ExpEuler()

    def derivatives(self, s, p, inp, ctx):
        return {'V': (p.E_L - s.V + p.R * inp.I) / p.tau}

    def on_input(self, s, p, inp, ctx):
        return s.replace(V=s.V + inp.w)


class ToyAdEx(NeuronModel):
    """aeif-like: strongly driven, several spikes per step, jump w += b."""

    params = Params(tau=5.0 * u.ms, V_th=-50.0 * u.mV, V_reset=-60.0 * u.mV, E_L=-70.0 * u.mV,
                    tau_w=50.0 * u.ms, b=10.0 * u.mV, I_e=500.0 * u.mV)
    states = States(V=Init(-70.0 * u.mV), w=Init(0.0 * u.mV))
    outputs = Ports(spike=Event(max_multiplicity=None))
    spiking = SpikeRule(condition=lambda s, p: (s.V - p.V_th) / (p.V_th - p.V_reset),
                        jump=lambda s, p: s.replace(V=p.V_reset, w=s.w + p.b))
    integrator = RK(order=2, substeps=20)

    def derivatives(self, s, p, inp, ctx):
        return {'V': (p.E_L - s.V + p.I_e - s.w) / p.tau, 'w': -s.w / p.tau_w}


GATE_CALLS = []


def _gate(V):
    GATE_CALLS.append(1)
    return 1.0 / (1.0 + jnp.exp(-(V.to_decimal(u.mV) + 60.0) / 5.0))


class ToyCond(NeuronModel):
    params = Params(C=100.0 * u.pF, g_L=10.0 * u.nS, E_L=-70.0 * u.mV, E_ex=0.0 * u.mV)
    states = States(V=Init(-70.0 * u.mV))
    inputs = Ports(ex=Conductance(E_rev='E_ex', gate=_gate), nmda=Conductance(E_rev=0.0 * u.mV))
    integrator = RK(order=4, substeps=2)

    def derivatives(self, s, p, inp, ctx):
        I = conductance_current(self.inputs['ex'], inp.ex, s.V, p)
        I = I + conductance_current(self.inputs['nmda'], inp.nmda, s.V, p)
        return {'V': (p.g_L * (p.E_L - s.V) + I) / p.C}


def _ctx(i=0, dt=DT):
    return StepContext.create(i * dt, dt, i)


def _run(model, n_steps, inputs_fn):
    def body(i):
        ctx = StepContext(t=i * DT, dt=DT, step=i)
        out = model.update(ctx, inputs_fn(i))
        return out['spike'], model.dyn_states['V'].value

    return brainstate.transform.for_loop(body, jnp.arange(n_steps))


# ---------------------------------------------------------------- declaration

class TestDeclaration:
    def test_states_moved_and_module_method_kept(self):
        assert isinstance(ToyLIF.state_decl, States) and list(ToyLIF.state_decl) == ['V']
        m = ToyLIF(2)
        st = m.states(brainstate.HiddenState)
        names = {k[-1] for k in st}
        assert names == {'V', latch_name('I'), 'spike'}
        short = {k[-1] for k, v in m.states(brainstate.ShortTermState).items()
                 if not isinstance(v, brainstate.HiddenState)}
        assert short == {'refractory_count'}
        assert len(m.states(brainstate.ParamState)) == 0

    def test_all_state_decl(self):
        d = ToyLIF.all_state_decl()
        assert list(d) == ['V', 'latch_I', 'spike', 'refractory_count']

    def test_trainable_and_overrides(self):
        m = ToyLIF(3, params=dict(tau=[5.0, 10.0, 20.0] * u.ms), init=dict(V=-65.0 * u.mV),
                   trainable=['tau'])
        assert isinstance(m.param_states['tau'], brainstate.ParamState)
        assert u.math.allclose(m.dyn_states['V'].value, -65.0 * u.mV)
        m.dyn_states['V'].value = jnp.zeros(3) * u.mV
        m.reset_state()
        assert u.math.allclose(m.dyn_states['V'].value, -65.0 * u.mV)

    def test_capabilities_default(self):
        assert ToyLIF.capabilities == Capabilities()
        assert Device.capabilities.differentiable is False

    def test_bad_declarations(self):
        with pytest.raises(TypeError, match='Exact'):
            class A(NeuronModel):
                integrator = Exact()
        with pytest.raises(TypeError, match='not a declared output'):
            class B(NeuronModel):
                spiking = SpikeRule(lambda s, p: s.V > 0, lambda s, p: s)
        with pytest.raises(TypeError, match='Clamp var'):
            class C(NeuronModel):
                outputs = Ports(spike=Event())
                spiking = SpikeRule(lambda s, p: s.V > 0, lambda s, p: s, refractory=Clamp('U', 1 * u.ms))
        with pytest.raises(TypeError, match='clash with the spike rule'):
            class D(NeuronModel):
                states = States(spike=Init(0.0))
                outputs = Ports(spike=Event())
                spiking = SpikeRule(lambda s, p: s.spike > 0, lambda s, p: s)
        with pytest.raises(TypeError, match='latches'):
            class E(NeuronModel):
                states = States(latch_I=Init(0.0 * u.pA))
                inputs = Ports(I=Current())
        with pytest.raises(TypeError, match='must be a States'):
            class F(NeuronModel):
                states = 3
        with pytest.raises(TypeError, match='must be a Ports'):
            class G(NeuronModel):
                inputs = {'I': Current()}
        with pytest.raises(TypeError, match='Integrator'):
            class H(NeuronModel):
                integrator = 'rk4'
        with pytest.raises(TypeError, match='SpikeRule'):
            class K(NeuronModel):
                spiking = 'threshold'

    def test_exact_model_overriding_step_is_allowed(self):
        class Ex(NeuronModel):
            states = States(V=Init(0.0 * u.mV))
            integrator = Exact()

            def step(self, s, p, inp, ctx):
                return s.replace(V=s.V + 1.0 * u.mV), {}

        m = Ex(2)
        assert m.update(_ctx()) == {}
        assert u.math.allclose(m.dyn_states['V'].value, 1.0 * u.mV)

    def test_instance_integrator_override(self):
        m = ToyLIF(1, integrator=RK(4, 3))
        assert m.integrator == RK(4, 3) and ToyLIF.integrator == ExpEuler()
        with pytest.raises(TypeError, match='Exact'):
            ToyLIF(1, integrator=Exact())

    def test_bad_construction(self):
        with pytest.raises(ValueError, match='non-negative'):
            ToyLIF(-1)
        with pytest.raises(NotImplementedError, match='vmap'):
            ToyLIF(1).init_state(batch_size=2)
        with pytest.raises(NotImplementedError, match='derivatives'):
            class NoDeriv(NeuronModel):
                states = States(V=Init(0.0 * u.mV))
            NoDeriv(1).update(_ctx())

    def test_zero_size_population(self):
        m = ToyLIF(0)
        out = m.update(_ctx(), {'I': 100 * u.pA})
        assert out['spike'].shape == (0,)


# ---------------------------------------------------------------- inputs

class TestInputs:
    def test_prepare_inputs(self):
        m = ToyLIF(3)
        inp = m.prepare_inputs({'I': 0.1 * u.nA})
        assert u.math.allclose(inp.I, [100.0, 100.0, 100.0] * u.pA)
        assert u.math.allclose(inp.w, jnp.zeros(3) * u.mV)
        with pytest.raises(PortError, match='unknown input'):
            m.prepare_inputs({'J': 1 * u.pA})
        with pytest.raises(UnitMismatchError):
            m.prepare_inputs({'I': 1 * u.mV})

    def test_timed_event_passthrough(self):
        class T(NeuronModel):
            states = States(V=Init(0.0 * u.mV))
            inputs = Ports(te=TimedEvent(capacity=2, unit=u.mV))
            integrator = Exact()

            def step(self, s, p, inp, ctx):
                return s.replace(V=s.V + u.math.sum(inp.te.weights, axis=-1)), {}

        m = T(2)
        payload = TimedEventPayload.empty(2, 2, weight_unit=u.mV)
        assert m.prepare_inputs({'te': payload}).te is payload
        assert isinstance(m.prepare_inputs().te, TimedEventPayload)
        m.update(_ctx())

    def test_current_latched_one_step(self, x64):
        m = ToyLIF(1, precision='float64')
        p = m.read()[1]
        m.update(_ctx(0), {'I': 100 * u.pA})
        # the current arriving at step 0 acts during step 1 only
        assert u.math.allclose(m.dyn_states['V'].value, p.E_L)
        assert u.math.allclose(m.dyn_states['latch_I'].value, 100 * u.pA)
        m.update(_ctx(1), {})
        expected = -70.0 + 10.0 * (1 - math.exp(-0.01))
        assert float(m.dyn_states['V'].value.to_decimal(u.mV)[0]) == pytest.approx(expected, rel=1e-13)

    def test_event_jump_after_propagation(self, x64):
        m = ToyLIF(1, precision='float64', init=dict(V=-60.0 * u.mV))
        m.update(_ctx(), {'w': 2.0 * u.mV})
        # propagate -60 towards -70 over dt, then jump by +2 (not propagated)
        expected = -70.0 + 10.0 * math.exp(-0.01) + 2.0
        assert float(m.dyn_states['V'].value.to_decimal(u.mV)[0]) == pytest.approx(expected, rel=1e-13)

    def test_conductance_gate_inside_substeps(self, x64):
        GATE_CALLS.clear()
        m = ToyCond(1, precision='float64')
        ctx = StepContext.create(0 * u.ms, 1.0 * u.ms)
        m.update(ctx, {'ex': 5.0 * u.nS})
        assert len(GATE_CALLS) == 8  # 2 substeps x 4 RK4 stages: gate sees substep V
        v_coarse = m.dyn_states['V'].value
        fine = ToyCond(1, precision='float64', integrator=RK(4, 64))
        fine.update(ctx, {'ex': 5.0 * u.nS})
        assert u.math.allclose(v_coarse, fine.dyn_states['V'].value, rtol=1e-7)
        # V moved: conductance is not latched (acts in the same step)
        assert float(v_coarse.to_decimal(u.mV)[0]) > -70.0

    def test_conductance_current_helper(self):
        p = Record(E=0.0 * u.mV)
        I = conductance_current(Conductance(E_rev='E'), 2 * u.nS, -50 * u.mV, p)
        assert u.math.allclose(I, 100 * u.pA)
        with pytest.raises(PortError, match='E_rev'):
            conductance_current(Conductance(), 2 * u.nS, -50 * u.mV, p)

    def test_derivative_errors(self):
        class Bad(NeuronModel):
            states = States(V=Init(0.0 * u.mV))

            def derivatives(self, s, p, inp, ctx):
                return {'U': 1.0 * u.mV / u.ms}

        with pytest.raises(KeyError, match='non-integrated'):
            Bad(1).update(_ctx())

        class BadUnit(NeuronModel):
            states = States(V=Init(0.0 * u.mV))

            def derivatives(self, s, p, inp, ctx):
                return {'V': 1.0 * u.mV}

        with pytest.raises(UnitMismatchError):
            BadUnit(1).update(_ctx())

    def test_missing_derivative_is_constant(self):
        class Two(NeuronModel):
            states = States(V=Init(0.0 * u.mV), c=Init(3.0 * u.mV))

            def derivatives(self, s, p, inp, ctx):
                return {'V': s.c / (1.0 * u.ms)}

        m = Two(1)
        m.update(_ctx())
        assert u.math.allclose(m.dyn_states['V'].value, 0.3 * u.mV)
        assert u.math.allclose(m.dyn_states['c'].value, 3.0 * u.mV)


# ---------------------------------------------------------------- dynamics

def _charge_time(tau, RI, dV):
    return tau * math.log(RI / (RI - dV))


class TestToyLIFDynamics:
    def test_analytic_subthreshold_trajectory(self, x64):
        m = ToyLIF(2, params=dict(tau=[5.0, 20.0] * u.ms), precision='float64')
        _, V = _run(m, 200, lambda i: {'I': 150.0 * u.pA})  # R I = 15 mV < 20 mV
        n = np.arange(200)
        tau = np.array([5.0, 20.0])
        # step 0 sees no current (latch); V_n = E_L + RI (1 - exp(-n dt / tau))
        ref = -70.0 + 15.0 * (1 - np.exp(-(n[:, None]) * 0.1 / tau[None]))
        np.testing.assert_allclose(V.to_decimal(u.mV), ref, rtol=1e-12)

    def test_spike_timing_and_refractory(self, x64):
        m = ToyLIF(1, precision='float64')
        RI = 25.0
        spikes, V = _run(m, 1000, lambda i: {'I': 250.0 * u.pA})
        steps = np.nonzero(np.asarray(spikes[:, 0]))[0]
        tc = _charge_time(10.0, RI, 20.0)  # 16.094 ms
        # first spike: 1 latch step + ceil(tc / dt) integration steps, detected at step end
        assert steps[0] == math.ceil(tc / 0.1)
        # ISI: 20 clamped steps (t_ref) + recharge from V_reset
        isi = np.diff(steps)
        assert np.all(isi == 20 + math.ceil(tc / 0.1))
        # clamped at V_reset during refractoriness
        np.testing.assert_allclose(V[steps[0] + 1: steps[0] + 21, 0].to_decimal(u.mV), -70.0)
        assert np.all(np.asarray(spikes) <= 1.0)

    def test_jit_and_vmap_functional_step(self):
        m = ToyLIF(3)
        s, p = m.read()
        ctx = _ctx()

        @jax.jit
        def one(I):
            inp = m.prepare_inputs({'I': I * u.pA})
            s1, _ = m.step(s.replace(latch_I=inp.I), p, inp, ctx)
            return s1.V.to_decimal(u.mV)

        out = jax.vmap(one)(jnp.array([0.0, 100.0, 300.0]))
        assert out.shape == (3, 3)
        assert np.all(np.diff(np.asarray(out[:, 0])) > 0)


class TestMultiSpike:
    def test_several_spikes_in_one_step(self):
        m = ToyAdEx(1)
        ctx = StepContext.create(0 * u.ms, 1.0 * u.ms)
        out = m.update(ctx)
        n = float(out['spike'][0])
        assert n >= 2.0 and n == int(n)
        # w jumped by b per spike (and decayed a little in between)
        w = float(m.dyn_states['w'].value.to_decimal(u.mV)[0])
        assert 0.9 * 10.0 * n < w <= 10.0 * n

    def test_step_check_caps_at_one(self):
        class OncePerStep(ToyAdEx):
            spiking = SpikeRule(condition=ToyAdEx.spiking.condition, jump=ToyAdEx.spiking.jump, when='step')

        out = OncePerStep(1).update(StepContext.create(0 * u.ms, 1.0 * u.ms))
        assert float(out['spike'][0]) <= 1.0

    def test_block_refractory_substep_granularity(self):
        class Blocked(ToyAdEx):
            params = ToyAdEx.params.extend(t_ref=Param(0.3 * u.ms))
            spiking = SpikeRule(condition=ToyAdEx.spiking.condition, jump=ToyAdEx.spiking.jump,
                                refractory=Block('t_ref', 'substep'))

        free = float(ToyAdEx(1).update(StepContext.create(0 * u.ms, 1.0 * u.ms))['spike'][0])
        blocked = float(Blocked(1).update(StepContext.create(0 * u.ms, 1.0 * u.ms))['spike'][0])
        assert 1.0 <= blocked <= free


class TestNoSpiking:
    def test_outputs_empty(self):
        m = ToyCond(2)
        assert m.update(_ctx(), {'ex': 1 * u.nS}) == {}
        assert m.output_states() == {}


# ---------------------------------------------------------------- gradients

class TestGradients:
    def _loss(self, theta, detach=True):
        """Sum of V over a 300-step rollout (spiking), as a function of (tau, I)."""
        m = ToyLIF(1, precision='float64', detach_reset=detach)
        m.param_states['tau'].value = jnp.reshape(theta[0], (1,)) * u.ms

        def body(i):
            ctx = StepContext(t=i * DT, dt=DT, step=i)
            m.update(ctx, {'I': theta[1] * u.pA})
            return m.dyn_states['V'].value.to_decimal(u.mV)[0]

        return jnp.sum(brainstate.transform.for_loop(body, jnp.arange(300)))

    def test_grad_through_rollout_vs_finite_differences(self, x64):
        theta = jnp.array([10.0, 260.0])
        g = jax.grad(self._loss)(theta)
        eps = np.array([1e-5, 1e-4])
        fd = [(self._loss(theta.at[i].add(eps[i])) - self._loss(theta.at[i].add(-eps[i]))) / (2 * eps[i])
              for i in range(2)]
        assert np.all(np.isfinite(np.asarray(g)))
        np.testing.assert_allclose(g, fd, rtol=1e-4)

    def test_surrogate_gradient_of_spike_count(self, x64):
        def n_spikes(I):
            m = ToyLIF(1, precision='float64')

            def body(i):
                return m.update(StepContext(t=i * DT, dt=DT, step=i), {'I': I * u.pA})['spike'][0]

            return jnp.sum(brainstate.transform.for_loop(body, jnp.arange(400)))

        assert float(n_spikes(260.0)) >= 1.0
        assert float(jax.grad(n_spikes)(260.0)) > 0.0
        assert float(jax.grad(n_spikes)(50.0)) >= 0.0  # far below threshold: ~0, finite

    def test_detach_reset_changes_gradient_not_value(self, x64):
        theta = jnp.array([10.0, 260.0])
        assert float(self._loss(theta, True)) == pytest.approx(float(self._loss(theta, False)), rel=1e-14)
        g1 = jax.grad(lambda t: self._loss(t, True))(theta)
        g2 = jax.grad(lambda t: self._loss(t, False))(theta)
        assert not np.allclose(g1, g2)


# ---------------------------------------------------------------- braintrace

class _Net(brainstate.nn.Module):
    def __init__(self, n_in, n):
        super().__init__()
        self.W = brainstate.ParamState(
            jax.random.uniform(jax.random.key(0), (n_in, n), minval=0.5, maxval=1.5))
        self.neu = ToyLIF(n, params=dict(t_ref=0.5 * u.ms), init=dict(V=-55.0 * u.mV))

    def init_state(self, *args, **kwargs):
        self.neu.init_state()

    def update(self, x):
        I = braintrace.matmul(x, self.W.value) * u.nA
        return self.neu.update(StepContext(t=0.0 * u.ms, dt=DT, step=0), {'I': I})['spike']


class TestBraintrace:
    @pytest.mark.parametrize('algorithm', ['eprop', 'd_rtrl'])
    def test_compile_and_nonzero_weight_gradients(self, algorithm):
        net = _Net(3, 4)
        x0 = jnp.ones(3)
        learner = braintrace.compile(net, algorithm, x0)
        xs = jax.random.uniform(jax.random.key(1), (40, 3)) * 2.0
        ys = jnp.ones((40, 4))

        def step_loss(x, y):
            return jnp.mean((learner(x) - y) ** 2)

        grads = learner.etrace_grad(xs, ys, step_fn=step_loss)
        g = jax.tree.leaves(grads)
        assert len(g) == 1 and g[0].shape == (3, 4)
        assert np.all(np.isfinite(np.asarray(g[0])))
        assert float(jnp.abs(g[0]).sum()) > 0.0


# ---------------------------------------------------------------- devices

class DC(Device):
    params = Params(amplitude=Param(100.0 * u.pA))
    outputs = Ports(I=Current())

    def step(self, s, p, inp, ctx):
        return s, {'I': p.amplitude}


class TestDevice:
    def test_device_outputs(self):
        d = DC(2, params=dict(amplitude=50.0 * u.pA))
        out = d.update(_ctx())
        assert u.math.allclose(out['I'], [50.0, 50.0] * u.pA)

    def test_device_rules(self):
        with pytest.raises(TypeError, match='no inputs'):
            class Bad(Device):
                inputs = Ports(I=Current())
        with pytest.raises(TypeError, match='override step'):
            class NoStep(Device):
                outputs = Ports(spike=Event())
        with pytest.raises(TypeError, match='not an output port'):
            class BadSpike(Device):
                spiking = SpikeRule(lambda s, p: s.x > 0, lambda s, p: s)

                def step(self, s, p, inp, ctx):
                    return s, {}
        with pytest.raises(NotImplementedError):
            Device.step(None, None, None, None, None)
