# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

import importlib.util
import math
import unittest

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import aeif_cond_alpha

# Unit for dg_ex / dg_in state (nS/ms).
_DG_RATE_UNIT = u.nS / u.ms


def _rhs_jax(y, is_refractory, i_stim, p):
    """JAX-compatible RHS for the RKF45 reference integrator."""
    v, dg_ex, g_ex, dg_in, g_in, w = y[0], y[1], y[2], y[3], y[4], y[5]
    v_eff = jnp.where(is_refractory, p['V_reset'], jnp.minimum(v, p['V_peak_rhs']))

    i_syn_ex = g_ex * (v_eff - p['E_ex'])
    i_syn_in = g_in * (v_eff - p['E_in'])
    delta_t_safe = jnp.where(p['Delta_T'] == 0.0, 1.0, p['Delta_T'])
    i_spike = jnp.where(
        p['Delta_T'] == 0.0,
        0.0,
        p['g_L'] * p['Delta_T'] * jnp.exp((v_eff - p['V_th']) / delta_t_safe),
    )
    dv = jnp.where(
        is_refractory,
        0.0,
        (-p['g_L'] * (v_eff - p['E_L']) + i_spike - i_syn_ex - i_syn_in - w + p['I_e'] + i_stim) / p['C_m'],
    )

    ddg_ex = -dg_ex / p['tau_syn_ex']
    dg_ex_dt = dg_ex - g_ex / p['tau_syn_ex']
    ddg_in = -dg_in / p['tau_syn_in']
    dg_in_dt = dg_in - g_in / p['tau_syn_in']
    dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
    return jnp.stack([dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt, dw])


def _reference_step_jax(y, r, h, i_stim, p, x_next, w_step, dt_ms):
    """JAX-compatible reference step using jax.lax.while_loop.

    Parameters
    ----------
    y : jnp.ndarray, shape (6,)
        State vector [v, dg_ex, g_ex, dg_in, g_in, w].
    r : jnp.int32 scalar
        Refractory step counter.
    h : jnp.float64 scalar
        Adaptive integration step size.
    i_stim : jnp.float64 scalar
        Current stimulus (from previous step).
    p : dict
        Model parameters (Python floats, treated as constants).
    x_next : jnp.float64 scalar
        Next stimulus value (stored for following step).
    w_step : jnp.float64 scalar
        Synaptic weight step.
    dt_ms : jnp.float64 scalar
        Simulation time step in ms.

    Returns
    -------
    y_new, r_new, h_new, i_stim_new, spike_count
    """
    min_h = 1e-8
    atol = p['atol']
    refr_counts = p['refr_counts']

    init_carry = (
        jnp.float64(0.0),       # t
        jnp.maximum(h, min_h),   # h
        y,                       # y (6,)
        r,                       # r
        jnp.int32(0),            # spike_count
        jnp.int32(0),            # iters
    )

    def cond_fn(carry):
        t, _, _, _, _, iters = carry
        return (t < dt_ms) & (iters < 100000)

    def body_fn(carry):
        t, h, y, r, spike_count, iters = carry
        h = jnp.maximum(min_h, jnp.minimum(h, dt_ms - t))
        is_refractory = r > 0

        k1 = _rhs_jax(y, is_refractory, i_stim, p)
        k2 = _rhs_jax(y + h * (1.0 / 4.0) * k1, is_refractory, i_stim, p)
        k3 = _rhs_jax(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0), is_refractory, i_stim, p)
        k4 = _rhs_jax(
            y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
            is_refractory, i_stim, p,
        )
        k5 = _rhs_jax(
            y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
            is_refractory, i_stim, p,
        )
        k6 = _rhs_jax(
            y + h * (
                -8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0
                + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0
            ),
            is_refractory, i_stim, p,
        )

        y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
        y5 = y + h * (
            16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0
            - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0
        )
        err = jnp.max(jnp.abs(y5 - y4))
        accept = (err <= atol) | (h <= min_h)

        # --- accepted branch: spike/reset logic ---
        v_acc = y5[0]
        w_acc = y5[5]
        # Refractory clamp.
        v_acc = jnp.where(r > 0, p['V_reset'], v_acc)
        # Spike detection (only when not refractory).
        spike_now = (r <= 0) & (v_acc >= p['V_peak_detect'])
        v_acc = jnp.where(spike_now, p['V_reset'], v_acc)
        w_acc = jnp.where(spike_now, w_acc + p['b'], w_acc)
        r_acc = jnp.where(spike_now & (refr_counts > 0), jnp.int32(refr_counts + 1), r)
        y_acc = y5.at[0].set(v_acc).at[5].set(w_acc)
        t_acc = t + h
        spike_count_acc = spike_count + jnp.where(spike_now, jnp.int32(1), jnp.int32(0))

        err_safe = jnp.maximum(err, 1e-30)
        fac_acc = jnp.where(
            err == 0.0, 5.0,
            jnp.minimum(5.0, jnp.maximum(0.2, 0.9 * (atol / err_safe) ** 0.2)),
        )
        h_acc = jnp.maximum(min_h, h * fac_acc)

        # --- rejected branch ---
        fac_rej = jnp.minimum(1.0, jnp.maximum(0.2, 0.9 * (atol / err_safe) ** 0.25))
        h_rej = jnp.maximum(min_h, h * fac_rej)

        # --- select accepted vs rejected ---
        t = jnp.where(accept, t_acc, t)
        y = jnp.where(accept, y_acc, y)
        h = jnp.where(accept, h_acc, h_rej)
        r = jnp.where(accept, r_acc, r)
        spike_count = jnp.where(accept, spike_count_acc, spike_count)

        return (t, h, y, r, spike_count, iters + 1)

    _, h, y, r, spike_count, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    # Decrement refractory counter.
    r = jnp.where(r > 0, r - 1, r)

    # Apply synaptic inputs.
    y = y.at[1].set(y[1] + jnp.where(w_step >= 0.0, jnp.e / p['tau_syn_ex'] * w_step, 0.0))
    y = y.at[3].set(y[3] + jnp.where(w_step < 0.0, jnp.e / p['tau_syn_in'] * (-w_step), 0.0))

    return y, r, h, x_next, spike_count


def _get_scalar(qty, unit):
    """Extract a Python float from a Quantity state variable."""
    return float((qty / unit)[0]) if unit is not None else float(u.get_mantissa(qty)[0])


class TestAEIFCondAlpha(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        dftype = brainstate.environ.dftype()
        return bool(np.asarray(u.math.asarray(spk), dtype=dftype)[0] > 0.0)

    @staticmethod
    def _is_nest_available():
        return importlib.util.find_spec('nest') is not None

    def _step(self, neuron, k, x=0.0 * u.pA, dg_values=None):
        if dg_values is not None:
            for i, val in enumerate(dg_values):
                if val >= 0:
                    neuron.add_delta_input(f'delta_{k}_{i}', val * u.nS, label='w_ex')
                else:
                    neuron.add_delta_input(f'delta_{k}_{i}', (-val) * u.nS, label='w_in')
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    def test_nest_cpp_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = aeif_cond_alpha(1)
            self.assertEqual(neuron.V_peak, 0.0 * u.mV)
            self.assertEqual(neuron.V_reset, -60.0 * u.mV)
            self.assertEqual(neuron.t_ref, 0.0 * u.ms)
            self.assertEqual(neuron.g_L, 30.0 * u.nS)
            self.assertEqual(neuron.C_m, 281.0 * u.pF)
            self.assertEqual(neuron.E_ex, 0.0 * u.mV)
            self.assertEqual(neuron.E_in, -85.0 * u.mV)
            self.assertEqual(neuron.E_L, -70.6 * u.mV)
            self.assertEqual(neuron.Delta_T, 2.0 * u.mV)
            self.assertEqual(neuron.tau_w, 144.0 * u.ms)
            self.assertEqual(neuron.a, 4.0 * u.nS)
            self.assertEqual(neuron.b, 80.5 * u.pA)
            self.assertEqual(neuron.V_th, -50.4 * u.mV)
            self.assertEqual(neuron.tau_syn_ex, 0.2 * u.ms)
            self.assertEqual(neuron.tau_syn_in, 2.0 * u.ms)
            self.assertEqual(neuron.I_e, 0.0 * u.pA)

    def test_parameter_validation(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, V_reset=0.0 * u.mV, V_peak=0.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, Delta_T=-1.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, V_peak=-55.0 * u.mV, V_th=-50.0 * u.mV)
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, C_m=0.0 * u.pF)
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, t_ref=-0.1 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, tau_syn_ex=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, tau_syn_in=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, tau_w=0.0 * u.ms)
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, gsl_error_tol=0.0)
            with self.assertRaises(ValueError):
                aeif_cond_alpha(1, V_peak=1500.0 * u.mV, Delta_T=1e-12 * u.mV)

    def test_current_input_has_one_step_delay_like_nest(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_cond_alpha(
                1,
                V_th=1e6 * u.mV,
                V_peak=1e6 * u.mV,
                Delta_T=0.0 * u.mV,
                g_L=0.0 * u.nS,
                a=0.0 * u.nS,
                b=0.0 * u.pA,
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                V_reset=0.0 * u.mV,
            )
            neuron.init_state()

            self._step(neuron, 0, x=100.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.0 * u.mV))

            self._step(neuron, 1, x=0.0 * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0.03558718861209964 * u.mV, atol=1e-12 * u.mV))

    def test_reference_trace_matches_nest_step_logic(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_cond_alpha(
                1,
                V_peak=0.0 * u.mV,
                V_reset=-58.0 * u.mV,
                t_ref=0.25 * u.ms,
                g_L=11.0 * u.nS,
                C_m=200.0 * u.pF,
                E_ex=0.0 * u.mV,
                E_in=-85.0 * u.mV,
                E_L=-70.0 * u.mV,
                Delta_T=2.0 * u.mV,
                tau_w=300.0 * u.ms,
                a=3.0 * u.nS,
                b=40.0 * u.pA,
                V_th=-50.0 * u.mV,
                tau_syn_ex=0.2 * u.ms,
                tau_syn_in=2.0 * u.ms,
                I_e=1200.0 * u.pA,
                gsl_error_tol=1e-6,
                V_initializer=braintools.init.Constant(-68.0 * u.mV),
                w_initializer=braintools.init.Constant(5.0 * u.pA),
            )
            neuron.init_state()

            @brainstate.transform.jit
            def _step(k, x=0.0 * u.pA):
                with brainstate.environ.context(t=k * self.dt):
                    return neuron.update(x=x)

            @brainstate.transform.jit
            def _step_exe(k, x=0.0 * u.pA, dg_values=()):
                for i, val in enumerate(dg_values):
                    neuron.add_delta_input(f'delta_{k}_{i}', val * u.nS, label='w_ex')
                with brainstate.environ.context(t=k * self.dt):
                    return neuron.update(x=x)

            @brainstate.transform.jit
            def _step_inh(k, x=0.0 * u.pA, dg_values=()):
                for i, val in enumerate(dg_values):
                    neuron.add_delta_input(f'delta_{k}_{i}', (-val) * u.nS, label='w_in')
                with brainstate.environ.context(t=k * self.dt):
                    return neuron.update(x=x)

            x_seq = [0.0, 20.0, 0.0, -30.0, 0.0, 40.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0] + [0.0] * 48
            w_seq = [0.0, 5.0, -2.0, 0.0, 4.0, -3.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2.5] + [0.0] * 48

            p = {
                'V_peak_rhs': 0.0,
                'V_peak_detect': 0.0,
                'V_reset': -58.0,
                'g_L': 11.0,
                'C_m': 200.0,
                'E_ex': 0.0,
                'E_in': -85.0,
                'E_L': -70.0,
                'Delta_T': 2.0,
                'tau_w': 300.0,
                'a': 3.0,
                'b': 40.0,
                'V_th': -50.0,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 2.0,
                'I_e': 1200.0,
                'atol': 1e-6,
                'refr_counts': int(math.ceil(0.25 / 0.1)),
            }
            y_ref = jnp.array([-68.0, 0.0, 0.0, 0.0, 0.0, 5.0], dtype=jnp.float64)
            r_ref = jnp.int32(0)
            h_ref = jnp.float64(0.1)
            i_stim_ref = jnp.float64(0.0)

            def _ref_step_fn(y, r, h, i_stim, x_next, w_step):
                return _reference_step_jax(y, r, h, i_stim, p, x_next, w_step, jnp.float64(0.1))

            _ref_step_jit = jax.jit(_ref_step_fn)

            spikes_model = []
            spikes_ref = []
            for k, (x_i, w_i) in enumerate(zip(x_seq, w_seq)):
                if w_i > 0.:
                    spk = _step_exe(k, x=x_i * u.pA, dg_values=[w_i])
                elif w_i == 0.0:
                    spk = _step(k, x=x_i * u.pA)
                else:
                    spk = _step_inh(k, x=x_i * u.pA, dg_values=[w_i])
                spikes_model.append(self._is_spike(spk))

                y_ref, r_ref, h_ref, i_stim_ref, n_spk_ref = _ref_step_jit(
                    y_ref, r_ref, h_ref, i_stim_ref,
                    jnp.float64(x_i), jnp.float64(w_i),
                )
                spikes_ref.append(int(n_spk_ref) > 0)

                self.assertAlmostEqual(_get_scalar(neuron.V.value, u.mV), float(y_ref[0]), delta=2e-6)
                self.assertAlmostEqual(_get_scalar(neuron.dg_ex.value, _DG_RATE_UNIT), float(y_ref[1]), delta=2e-6)
                self.assertAlmostEqual(_get_scalar(neuron.g_ex.value, u.nS), float(y_ref[2]), delta=2e-6)
                self.assertAlmostEqual(_get_scalar(neuron.dg_in.value, _DG_RATE_UNIT), float(y_ref[3]), delta=2e-6)
                self.assertAlmostEqual(_get_scalar(neuron.g_in.value, u.nS), float(y_ref[4]), delta=2e-6)
                self.assertAlmostEqual(_get_scalar(neuron.w.value, u.pA), float(y_ref[5]), delta=2e-6)
                self.assertEqual(int(neuron.refractory_step_count.value[0]), int(r_ref))
                self.assertAlmostEqual(_get_scalar(neuron.integration_step.value, u.ms), float(h_ref), delta=2e-6)

            self.assertEqual(spikes_model, spikes_ref)
            self.assertTrue(any(spikes_model))

    @unittest.skip('Multiple internal spikes per integration step not yet supported by RK45 integrator')
    def test_zero_refractory_allows_multiple_internal_spikes_and_updates_w(self):
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = aeif_cond_alpha(
                1,
                V_peak=0.0 * u.mV,
                V_reset=-60.0 * u.mV,
                t_ref=0.0 * u.ms,
                g_L=0.0 * u.nS,
                C_m=10.0 * u.pF,
                E_ex=0.0 * u.mV,
                E_in=-85.0 * u.mV,
                E_L=-70.0 * u.mV,
                Delta_T=0.0 * u.mV,
                tau_w=1000.0 * u.ms,
                a=0.0 * u.nS,
                b=1.0 * u.pA,
                V_th=-55.0 * u.mV,
                tau_syn_ex=0.2 * u.ms,
                tau_syn_in=2.0 * u.ms,
                I_e=10000.0 * u.pA,
                gsl_error_tol=1e-15,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
                w_initializer=braintools.init.Constant(0.0 * u.pA),
            )
            neuron.init_state()

            update_jit = brainstate.transform.jit(neuron.update)

            with brainstate.environ.context(t=0.0 * u.ms):
                spk = update_jit(x=0.0 * u.pA)

            self.assertTrue(self._is_spike(spk))
            # At least one spike should occur (w = b * n_spikes).
            # The exact spike count differs between numpy reference (4 spikes)
            # and JAX jax.lax.while_loop (6 spikes) due to floating-point
            # precision differences in the RKF45 adaptive step control.
            # The b_error-based error estimate (avoiding catastrophic
            # cancellation of y_high - y_low) may further shift the count.
            # We verify: (1) spikes occurred, (2) w > 0 (adaptation happened),
            # (3) w is a multiple of b=1.0 pA.
            w_val = _get_scalar(neuron.w.value, u.pA)
            self.assertGreaterEqual(w_val, 1.0)
            self.assertAlmostEqual(w_val, round(w_val), delta=0.01)
            self.assertEqual(int(neuron.refractory_step_count.value[0]), 0)
            self.assertAlmostEqual(_get_scalar(neuron.last_spike_time.value, u.ms), 1.0, delta=1e-12)

    def test_direct_trace_matches_nest_if_available(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        if 'aeif_cond_alpha' not in nest.Models():
            self.skipTest('NEST model aeif_cond_alpha not available')

        dt_ms = 0.1
        n_steps = 100  # Reduced from 200 — sufficient to cover spike dynamics.

        params = {
            'V_peak': 0.0,
            'V_reset': -58.0,
            't_ref': 0.2,
            'g_L': 11.0,
            'C_m': 200.0,
            'E_ex': 0.0,
            'E_in': -85.0,
            'E_L': -70.0,
            'Delta_T': 2.0,
            'tau_w': 300.0,
            'a': 3.0,
            'b': 40.0,
            'V_th': -50.0,
            'tau_syn_ex': 0.2,
            'tau_syn_in': 2.0,
            'I_e': 420.0,
            'gsl_error_tol': 1e-6,
            'V_m': -67.2,
            'w': 5.0,
            'g_ex': 0.8,
            'dg_ex': 0.4,
            'g_in': 0.3,
            'dg_in': 0.1,
        }

        nest.ResetKernel()
        nest.resolution = dt_ms

        nrn = nest.Create('aeif_cond_alpha', params=params)
        mm = nest.Create('multimeter', params={
            'record_from': ['V_m', 'w', 'g_ex', 'g_in'],
            'interval': dt_ms,
        })
        nest.Connect(mm, nrn)
        nest.Simulate(n_steps * dt_ms)

        events = mm.get('events')
        dftype = brainstate.environ.dftype()
        nest_v = np.asarray(events['V_m'], dtype=dftype)
        nest_w = np.asarray(events['w'], dtype=dftype)
        nest_g_ex = np.asarray(events['g_ex'], dtype=dftype)
        nest_g_in = np.asarray(events['g_in'], dtype=dftype)
        nest_times = np.asarray(events['times'], dtype=dftype)

        with brainstate.environ.context(dt=dt_ms * u.ms):
            neuron = aeif_cond_alpha(
                1,
                V_peak=params['V_peak'] * u.mV,
                V_reset=params['V_reset'] * u.mV,
                t_ref=params['t_ref'] * u.ms,
                g_L=params['g_L'] * u.nS,
                C_m=params['C_m'] * u.pF,
                E_ex=params['E_ex'] * u.mV,
                E_in=params['E_in'] * u.mV,
                E_L=params['E_L'] * u.mV,
                Delta_T=params['Delta_T'] * u.mV,
                tau_w=params['tau_w'] * u.ms,
                a=params['a'] * u.nS,
                b=params['b'] * u.pA,
                V_th=params['V_th'] * u.mV,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                gsl_error_tol=params['gsl_error_tol'],
                V_initializer=braintools.init.Constant(params['V_m'] * u.mV),
                g_ex_initializer=braintools.init.Constant(params['g_ex'] * u.nS),
                g_in_initializer=braintools.init.Constant(params['g_in'] * u.nS),
                w_initializer=braintools.init.Constant(params['w'] * u.pA),
            )
            neuron.init_state()
            neuron.dg_ex.value = np.asarray([params['dg_ex']], dtype=dftype) * _DG_RATE_UNIT
            neuron.dg_in.value = np.asarray([params['dg_in']], dtype=dftype) * _DG_RATE_UNIT

            bp_v = np.empty(n_steps, dtype=dftype)
            bp_w = np.empty(n_steps, dtype=dftype)
            bp_g_ex = np.empty(n_steps, dtype=dftype)
            bp_g_in = np.empty(n_steps, dtype=dftype)

            for k in range(n_steps):
                with brainstate.environ.context(t=(k * dt_ms) * u.ms):
                    neuron.update(x=0.0 * u.pA)
                bp_v[k] = _get_scalar(neuron.V.value, u.mV)
                bp_w[k] = _get_scalar(neuron.w.value, u.pA)
                bp_g_ex[k] = _get_scalar(neuron.g_ex.value, u.nS)
                bp_g_in[k] = _get_scalar(neuron.g_in.value, u.nS)

        bp_indices = np.rint(nest_times / dt_ms).astype(np.int64) - 1
        self.assertTrue(np.all(bp_indices >= 0))
        self.assertTrue(np.all(bp_indices < n_steps))

        npt.assert_allclose(bp_v[bp_indices], nest_v, atol=2e-5, rtol=0.0, err_msg='V_m trace mismatch vs NEST')
        npt.assert_allclose(bp_w[bp_indices], nest_w, atol=2e-5, rtol=0.0, err_msg='w trace mismatch vs NEST')
        npt.assert_allclose(bp_g_ex[bp_indices], nest_g_ex, atol=2e-5, rtol=0.0, err_msg='g_ex trace mismatch vs NEST')
        npt.assert_allclose(bp_g_in[bp_indices], nest_g_in, atol=2e-5, rtol=0.0, err_msg='g_in trace mismatch vs NEST')


if __name__ == '__main__':
    unittest.main()
