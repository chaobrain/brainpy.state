# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the K0 ``iaf_psc_alpha`` (exact propagators); NEST parity lives in nest_parity_test."""
import math

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.context import StepContext
from brainspike._kernel.models import iaf_psc_alpha
from brainspike._kernel.models._testing import rollout, step_inputs
from brainspike._nest_neuron.iaf_psc_alpha import iaf_psc_alpha as OldIaf

DT = 0.1 * u.ms


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _ctx(i=0):
    return StepContext.create(i * DT, DT, i)


def _drive(n_steps, n, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    w = jax.random.normal(k1, (n_steps, n)) * 300.0
    w = jnp.where(jax.random.uniform(k2, (n_steps, n)) < 0.08, w, 0.0)
    I = jax.random.uniform(k3, (n_steps, n), minval=0.0, maxval=600.0)
    return w, I


class TestParityWithExistingModel:
    """Same numerics as ``_nest_neuron.iaf_psc_alpha`` (NumPy float64 propagators)."""

    @pytest.mark.parametrize('t_ref', [2.0, 0.0, 0.3])
    def test_traces_and_spikes(self, x64, t_ref):
        n_steps, n = 800, 6
        tau_ex = [0.5, 2.0, 10.0, 2.0, 1.0, 3.0] * u.ms  # 10.0 == tau_m: singular propagator
        w, I = _drive(n_steps, n)
        with brainstate.environ.context(dt=DT, precision=64):
            old = OldIaf(n, tau_syn_ex=tau_ex, t_ref=t_ref * u.ms, I_e=100.0 * u.pA)
            old.init_state()

            def body(i):
                old.add_delta_input('w', w[i] * u.pA)
                with brainstate.environ.context(t=i * DT):
                    spk = old.update(I[i] * u.pA)
                return spk, old.V.value, old.I_syn_ex.value, old.I_syn_in.value

            o_spk, o_V, o_Iex, o_Iin = brainstate.transform.for_loop(body, jnp.arange(n_steps))
        m = iaf_psc_alpha(n, params=dict(tau_syn_ex=tau_ex, t_ref=t_ref * u.ms, I_e=100.0 * u.pA),
                          precision='float64')
        inputs = {'ex': jnp.where(w > 0, w, 0.0) * u.pA, 'in_': jnp.where(w < 0, w, 0.0) * u.pA,
                  'I': I * u.pA}
        out, tr = rollout(m, n_steps, DT, inputs, record=('V_m', 'I_syn_ex', 'I_syn_in'))
        np.testing.assert_array_equal(np.asarray(out['spike']), np.asarray(o_spk) > 0)
        assert float(out['spike'].sum()) > 10
        for a, b in ((tr['V_m'], o_V), (tr['I_syn_ex'], o_Iex), (tr['I_syn_in'], o_Iin)):
            np.testing.assert_allclose(np.asarray(u.get_mantissa(a)), np.asarray(u.get_mantissa(b)),
                                       rtol=1e-10, atol=1e-9)


class TestDynamics:
    def test_psp_peak_time_analytic(self, x64):
        """Alpha PSC peaks at tau_syn after arrival; the jump is applied after propagation."""
        m = iaf_psc_alpha(1, params=dict(tau_syn_ex=2.0 * u.ms), precision='float64')
        inputs = step_inputs(200, 1, {'ex': ([0], [100.0], u.pA)}, dtype=jnp.float64)
        _, tr = rollout(m, 200, DT, inputs, record=('V_m', 'I_syn_ex'))
        I = np.asarray(tr['I_syn_ex'].to_decimal(u.pA))[:, 0]
        assert I[0] == 0.0  # the jump goes to dI: I is still 0 after the arrival step
        # I(t) = w * (t / tau) * e^{1 - t / tau}, t measured from the end of step 0
        t = np.arange(200) * 0.1
        np.testing.assert_allclose(I, 100.0 * t / 2.0 * np.exp(1 - t / 2.0), rtol=1e-12, atol=1e-12)
        assert np.argmax(I) == 20
        V = np.asarray(tr['V_m'].to_decimal(u.mV))[:, 0]
        assert V[0] == -70.0 and V[1] > -70.0  # V feels the jump one step later

    def test_current_latched_and_I_e_immediate(self, x64):
        m = iaf_psc_alpha(1, precision='float64')
        m.update(_ctx(0), {'I': 100.0 * u.pA})
        assert float(m.dyn_states['V_m'].value.to_decimal(u.mV)[0]) == -70.0
        m2 = iaf_psc_alpha(1, params=dict(I_e=100.0 * u.pA), precision='float64')
        m2.update(_ctx(0))
        expected = -70.0 + 100.0 * 10.0 / 250.0 * (1 - math.exp(-0.01))
        assert float(m2.dyn_states['V_m'].value.to_decimal(u.mV)[0]) == pytest.approx(expected, rel=1e-13)

    @pytest.mark.parametrize('t_ref, steps', [(2.0, 20), (0.25, 3), (0.05, 1), (0.0, 0), (0.1, 1)])
    def test_refractory_steps_nest_round_up(self, x64, t_ref, steps):
        m = iaf_psc_alpha(1, params=dict(t_ref=t_ref * u.ms, I_e=1e5 * u.pA), precision='float64')
        out, tr = rollout(m, 60, DT, record=('V_m',))
        spk = np.nonzero(np.asarray(out['spike'])[:, 0])[0]
        # very strong drive: spikes as soon as refractoriness ends
        assert np.all(np.diff(spk) == steps + 1)
        V = np.asarray(tr['V_m'].to_decimal(u.mV))[:, 0]
        for k in range(1, steps + 1):
            assert V[spk[0] + k] == -70.0  # held at V_reset

    def test_v_min_clip(self, x64):
        m = iaf_psc_alpha(1, params=dict(V_min=-72.0 * u.mV), precision='float64')
        _, tr = rollout(m, 100, DT, step_inputs(100, 1, {'in_': ([0], [-5e4], u.pA)}, dtype=jnp.float64),
                        record=('V_m',))
        V = np.asarray(tr['V_m'].to_decimal(u.mV))[:, 0]
        assert V.min() == -72.0

    def test_float32_close_to_float64(self):
        w, I = _drive(400, 4)
        inputs = {'ex': jnp.where(w > 0, w, 0.0) * u.pA, 'in_': jnp.where(w < 0, w, 0.0) * u.pA,
                  'I': I * u.pA}
        out32, tr32 = rollout(iaf_psc_alpha(4), 400, DT, inputs, record=('V_m',))
        with jax.enable_x64(True):
            inputs64 = {k: u.Quantity(jnp.asarray(u.get_mantissa(v), jnp.float64), unit=v.unit)
                        for k, v in inputs.items()}
            out64, tr64 = rollout(iaf_psc_alpha(4, precision='float64'), 400, DT, inputs64, record=('V_m',))
        assert tr32['V_m'].dtype == jnp.float32 and tr64['V_m'].dtype == jnp.float64
        np.testing.assert_array_equal(np.asarray(out32['spike']), np.asarray(out64['spike']))
        np.testing.assert_allclose(np.asarray(tr32['V_m'].mantissa), np.asarray(tr64['V_m'].mantissa),
                                   atol=2e-4)

    def test_heterogeneous_params_and_distributions(self):
        from brainspike._kernel.params import Uniform
        m = iaf_psc_alpha(5, params=dict(tau_m=Uniform(5.0 * u.ms, 20.0 * u.ms), I_e=500.0 * u.pA),
                          key=jax.random.key(0))
        tau = np.asarray(m.param_states['tau_m'].value.to_decimal(u.ms))
        assert len(np.unique(tau)) == 5
        out, _ = rollout(m, 300, DT)
        counts = np.asarray(out['spike']).sum(0)
        # V_inf = E_L + I_e tau_m / C_m rises with tau_m: non-decreasing spike counts
        assert np.all(np.diff(counts[np.argsort(tau)]) >= 0) and counts.max() > counts.min()

    def test_jit_vmap_over_params(self):
        m = iaf_psc_alpha(2)
        s, p = m.read()

        @jax.jit
        def V_after(I_e):
            pp = p.replace(I_e=jnp.full(2, I_e) * u.pA)
            return m.step(s, pp, m.prepare_inputs(), _ctx())[0].V_m.to_decimal(u.mV)

        out = jax.vmap(V_after)(jnp.array([0.0, 100.0, 200.0]))
        assert out.shape == (3, 2) and np.all(np.diff(np.asarray(out[:, 0])) > 0)


class TestGradients:
    def _loss(self, theta, n_steps=400):
        m = iaf_psc_alpha(1, precision='float64', detach_reset=True,
                          surrogate=braintools.surrogate.Sigmoid())
        m.param_states['tau_m'].value = jnp.reshape(theta[0], (1,)) * u.ms
        m.param_states['tau_syn_ex'].value = jnp.reshape(theta[1], (1,)) * u.ms
        w = jnp.zeros(n_steps).at[jnp.arange(5, n_steps, 37)].set(theta[2])

        def body(i):
            m.update(StepContext(t=i * DT, dt=DT, step=i), {'ex': w[i] * u.pA, 'I': 380.0 * u.pA})
            return m.dyn_states['V_m'].value.to_decimal(u.mV)[0]

        return jnp.sum(brainstate.transform.for_loop(body, jnp.arange(n_steps)))

    def test_bptt_vs_finite_differences(self, x64):
        theta = jnp.array([10.0, 2.0, 150.0])
        assert float(self._loss(theta)) != 0.0
        g = jax.grad(self._loss)(theta)
        eps = np.array([1e-5, 1e-5, 1e-4])
        fd = [(self._loss(theta.at[i].add(eps[i])) - self._loss(theta.at[i].add(-eps[i]))) / (2 * eps[i])
              for i in range(3)]
        assert np.all(np.isfinite(np.asarray(g)))
        np.testing.assert_allclose(g, fd, rtol=1e-4)

    def test_spiking_rollout_is_spiking(self, x64):
        m = iaf_psc_alpha(1, precision='float64')
        w = jnp.zeros(400).at[jnp.arange(5, 400, 37)].set(150.0)
        out, _ = rollout(m, 400, DT, input_fn=lambda i: {'ex': w[i] * u.pA, 'I': 380.0 * u.pA})
        assert float(out['spike'].sum()) >= 2.0  # the gradient test crosses resets

    def test_surrogate_gradient_wrt_weight(self, x64):
        def n_spikes(wt):
            m = iaf_psc_alpha(1, precision='float64', surrogate=braintools.surrogate.Sigmoid())

            def body(i):
                return m.update(StepContext(t=i * DT, dt=DT, step=i),
                                {'ex': jnp.where(i % 50 == 0, wt, 0.0) * u.pA, 'I': 370.0 * u.pA})['spike'][0]

            return jnp.sum(brainstate.transform.for_loop(body, jnp.arange(500)))

        assert float(jax.grad(n_spikes)(200.0)) > 0.0
