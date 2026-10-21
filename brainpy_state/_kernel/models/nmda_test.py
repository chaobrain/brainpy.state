# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the saturating NMDA kinetics (K0 T8)."""
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.model import NeuronModel
from brainpy_state._kernel.models import NMDA, nmda_kinetics
from brainpy_state._kernel.models._testing import rollout
from brainpy_state._kernel.ports import Event, Graded

DT = 0.1 * u.ms


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _spikes(n_steps, steps, n=1, value=1.0):
    a = np.zeros((n_steps, n))
    a[np.asarray(steps, int)] = value
    return jnp.asarray(a)


def _run(n_steps, steps, params=None, precision='float64', dt=DT, value=1.0, init=None):
    m = NMDA(1, params=params, precision=precision, init=init)
    _, tr = rollout(m, n_steps, dt, {'spike': _spikes(n_steps, steps, value=value)},
                    record=('x', 's', 'pulse'))
    return {k: np.asarray(v)[:, 0] for k, v in tr.items()}


def _old_bionmda(n_steps, steps, T_dur, dt=DT):
    from brainpy_state._brainpy.synapse import BioNMDA
    old = BioNMDA(1, T_dur=T_dur, g_initializer=braintools.init.Constant(0.))  # unitless g
    with brainstate.environ.context(dt=dt):
        old.init_state()

        def body(i, sp):
            with brainstate.environ.context(t=i * dt):
                return old.update(sp > 0), old.x.value

        g, x = brainstate.transform.for_loop(body, jnp.arange(n_steps), _spikes(n_steps, steps))
    return np.asarray(g)[:, 0], np.asarray(x)[:, 0]


class TestDeclaration:
    def test_ports_and_states(self):
        assert isinstance(NMDA.inputs['spike'], Event)
        assert isinstance(NMDA.outputs['s'], Graded)
        assert u.get_dim(NMDA.outputs['s'].unit) == u.get_dim(u.UNITLESS)
        assert list(NMDA.all_state_decl()) == ['x', 's', 'pulse']
        assert not NMDA.capabilities.differentiable
        assert issubclass(NMDA, NeuronModel)

    def test_states_dimensionless(self):
        m = NMDA(3)
        assert not isinstance(m.dyn_states['s'].value, u.Quantity)
        assert m.dyn_states['pulse'].value.dtype == jnp.int32


class TestKinetics:
    def test_matches_old_bionmda_unitless_g(self, x64):
        # same scheme (exp-Euler of the linear x, then s with the new x): round-off level
        n = 600
        steps = [10, 12, 100, 101, 102, 103, 104, 105, 106, 300]
        g_old, x_old = _old_bionmda(n, steps, 0.45 * u.ms)
        tr = _run(n, steps, params=dict(T_dur=0.45 * u.ms))
        np.testing.assert_allclose(tr['s'], g_old, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(tr['x'], x_old, rtol=1e-12, atol=1e-15)

    def test_closed_form_single_step(self, x64):
        p = NMDA(1, precision='float64').read()[1]
        x, s, pulse = nmda_kinetics(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0], jnp.int32),
                                    jnp.array([True]), p, DT)
        k = 1.0 * 1.0 + 0.5  # alpha2 T + beta2 (1/ms)
        x_ref = (1.0 / k) * (1 - np.exp(-k * 0.1))
        ks = 2.0 * x_ref + 0.01
        s_ref = (2.0 * x_ref / ks) * (1 - np.exp(-ks * 0.1))
        assert float(x[0]) == pytest.approx(x_ref, rel=1e-12)
        assert float(s[0]) == pytest.approx(s_ref, rel=1e-12)
        assert int(pulse[0]) == 4  # 0.5 ms = 5 steps, one used

    def test_pulse_length_rounds_up(self):
        for T_dur, n_steps in ((0.5, 5), (0.45, 5), (0.05, 1), (0.0, 0), (1.0, 10)):
            tr = _run(30, [3], params=dict(T_dur=T_dur * u.ms), precision='float32')
            # x rises exactly while the transmitter is present (steps 3 .. 3 + n_steps - 1)
            rising = np.flatnonzero(np.diff(np.concatenate([[0.0], tr['x']])) > 0)
            np.testing.assert_array_equal(rising, np.arange(3, 3 + n_steps), err_msg=str(T_dur))
            assert int(tr['pulse'][3]) == max(n_steps - 1, 0), T_dur

    def test_no_transmitter_gives_no_opening(self):
        tr = _run(50, [5], params=dict(T_dur=0.0 * u.ms), precision='float32')
        np.testing.assert_array_equal(tr['s'], 0.0)

    def test_arrival_during_pulse_restarts_it(self):
        tr = _run(40, [5, 8], precision="float32")
        assert tr['pulse'][8] == 4 and tr['pulse'][12] == 0 and tr['pulse'][11] == 1

    def test_multiplicity_and_weight_saturate(self, x64):
        one = _run(200, [5], value=1.0)
        many = _run(200, [5], value=3.0)
        np.testing.assert_array_equal(one['s'], many['s'])
        none = _run(200, [5], value=-1.0)  # non-positive payload is not an arrival
        np.testing.assert_array_equal(none['s'], 0.0)

    def test_saturation_bounded_by_one(self, x64):
        # a long burst saturates s below 1, independent of any conductance unit
        tr = _run(3000, np.arange(10, 2000, 2), params=dict(T=10.0 * u.mM))
        assert tr['s'].max() < 1.0 and tr['s'].max() > 0.95
        assert np.all(tr['s'] >= 0) and np.all(tr['x'] <= 1.0)

    def test_decay_after_release(self, x64):
        tr = _run(4000, [10])
        peak = int(np.argmax(tr['s']))
        assert 10 < peak < 400
        tail = tr['s'][peak + 1000:]
        assert np.all(np.diff(tail) < 0)
        # late decay rate -> beta1 (x has died out): s ~ exp(-0.01 t/ms)
        rate = -np.log(tail[-1] / tail[-500]) / (499 * 0.1)
        assert rate == pytest.approx(0.01, rel=2e-2)


class TestUnitsAndPrecision:
    def test_rate_units_do_not_matter(self, x64):
        ref = _run(800, [10, 50, 51])
        alt = _run(800, [10, 50, 51], params=dict(alpha1=2000.0 / u.second,
                                                  beta1=10.0 / u.second,
                                                  alpha2=1.0 / (u.second * u.molar) * 1e6,
                                                  beta2=500.0 / u.second, T=1e-3 * u.molar,
                                                  T_dur=0.5e-3 * u.second))
        np.testing.assert_allclose(alt['s'], ref['s'], rtol=1e-10, atol=1e-14)

    def test_dt_unit_does_not_matter(self, x64):
        ref = _run(300, [10])
        alt = _run(300, [10], dt=1e-4 * u.second)
        np.testing.assert_allclose(alt['s'], ref['s'], rtol=1e-10, atol=1e-14)

    def test_float32_close_to_float64(self, x64):
        ref = _run(1000, [10, 200, 201])
        f32 = _run(1000, [10, 200, 201], precision='float32')
        assert f32['s'].dtype == np.float32 and ref['s'].dtype == np.float64
        np.testing.assert_allclose(f32['s'], ref['s'], atol=2e-6)

    def test_heterogeneous_params_and_vmap(self, x64):
        m = NMDA(2, params=dict(alpha1=jnp.array([2.0, 4.0]) / u.ms), precision='float64')
        s, p = m.read()
        ctx = StepContext.create(0.0 * u.ms, DT)
        new, out = m.step(s, p, m.prepare_inputs({'spike': jnp.array([1.0, 1.0])}), ctx)
        assert float(out['s'][1]) > float(out['s'][0]) > 0
        # vmap over a trial axis of the state and the input
        xs = jnp.stack([s.x, s.x])
        f = lambda x, sp: nmda_kinetics(x, s.s, s.pulse, sp > 0, p, DT)[1]
        got = jax.vmap(f)(xs, jnp.array([[1.0, 0.0], [0.0, 1.0]]))
        assert float(got[0, 1]) == 0.0 and float(got[1, 0]) == 0.0 and float(got[1, 1]) > 0

    def test_jit_step(self, x64):
        m = NMDA(1, precision='float64')
        ctx = StepContext.create(0.0 * u.ms, DT)
        out = brainstate.transform.jit(lambda: m.update(ctx, {'spike': 1.0})['s'])()
        assert float(out[0]) > 0 and int(m.dyn_states['pulse'].value[0]) == 4
