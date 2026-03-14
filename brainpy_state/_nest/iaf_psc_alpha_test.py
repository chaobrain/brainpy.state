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

import math
import unittest

import brainstate
import braintools
import saiunit as u
import jax
import numpy as np
from brainpy.state import iaf_psc_alpha


def _alpha_propagator_p31_p32(tau_syn, tau_m, c_m, h):
    r"""Match NEST `IAFPropagatorAlpha::evaluate` including singular fallback."""
    beta = tau_syn * tau_m / (tau_m - tau_syn)
    gamma = beta / c_m
    inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)

    exp_h_tau_syn = math.exp(-h / tau_syn)
    expm1_h_tau = math.expm1(h * inv_beta)

    p32 = gamma * exp_h_tau_syn * expm1_h_tau
    if (not math.isfinite(p32)) or (abs(p32) < np.finfo(np.float64).tiny) or (p32 <= 0.0):
        exp_h_tau_m = math.exp(-h / tau_m)
        p32 = h / c_m * exp_h_tau_m
    else:
        exp_h_tau_m = math.nan

    h_min_regular = 1e-7 * tau_m * tau_m / abs(tau_m - tau_syn)
    if h > h_min_regular:
        p31 = gamma * exp_h_tau_syn * (beta * expm1_h_tau - h)
    else:
        if math.isnan(exp_h_tau_m):
            exp_h_tau_m = math.exp(-h / tau_m)
        p31 = 0.5 * h * h / c_m * exp_h_tau_m

    return p31, p32


def _reference_step(state, params, x_next, w_step, dt):
    r"""Single-step NEST-ordered reference update for scalar iaf_psc_alpha."""
    y0 = state['y0']
    dI_ex = state['dI_ex']
    I_ex = state['I_ex']
    dI_in = state['dI_in']
    I_in = state['I_in']
    y3 = state['y3']
    r = state['r']

    tau_m = params['tau_m']
    c_m = params['C_m']
    tau_ex = params['tau_syn_ex']
    tau_in = params['tau_syn_in']
    i_e = params['I_e']
    theta = params['V_th'] - params['E_L']
    v_reset = params['V_reset'] - params['E_L']
    lower = -math.inf if params['V_min'] is None else (params['V_min'] - params['E_L'])

    p11_ex = math.exp(-dt / tau_ex)
    p22_ex = p11_ex
    p21_ex = dt * p11_ex

    p11_in = math.exp(-dt / tau_in)
    p22_in = p11_in
    p21_in = dt * p11_in

    expm1_tau_m = math.expm1(-dt / tau_m)
    p30 = -tau_m / c_m * expm1_tau_m
    p31_ex, p32_ex = _alpha_propagator_p31_p32(tau_ex, tau_m, c_m, dt)
    p31_in, p32_in = _alpha_propagator_p31_p32(tau_in, tau_m, c_m, dt)

    epsc_init = math.e / tau_ex
    ipsc_init = math.e / tau_in

    if r == 0:
        y3 = (
            p30 * (y0 + i_e)
            + p31_ex * dI_ex
            + p32_ex * I_ex
            + p31_in * dI_in
            + p32_in * I_in
            + expm1_tau_m * y3
            + y3
        )
        y3 = max(y3, lower)
    else:
        r -= 1

    I_ex = p21_ex * dI_ex + p22_ex * I_ex
    dI_ex *= p11_ex
    dI_ex += epsc_init * max(w_step, 0.0)

    I_in = p21_in * dI_in + p22_in * I_in
    dI_in *= p11_in
    dI_in += ipsc_init * min(w_step, 0.0)

    spike = y3 >= theta
    if spike:
        r = int(math.ceil(params['t_ref'] / dt))
        y3 = v_reset

    state['y0'] = x_next
    state['dI_ex'] = dI_ex
    state['I_ex'] = I_ex
    state['dI_in'] = dI_in
    state['I_in'] = I_in
    state['y3'] = y3
    state['r'] = r

    return spike


class TestIAFPscAlpha(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Match NEST's double-precision CPU behavior as closely as possible.
        jax.config.update('jax_enable_x64', True)
        brainstate.environ.set(precision=64, platform='cpu')

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def _step(self, neuron, step_idx, x=0.0 * u.pA, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        with brainstate.environ.context(t=step_idx * self.dt):
            return neuron.update(x=x)

    def test_nest_default_parameters(self):
        neuron = iaf_psc_alpha(1)
        self.assertEqual(neuron.E_L, -70. * u.mV)
        self.assertEqual(neuron.C_m, 250. * u.pF)
        self.assertEqual(neuron.tau_m, 10. * u.ms)
        self.assertEqual(neuron.t_ref, 2. * u.ms)
        self.assertEqual(neuron.V_th, -55. * u.mV)
        self.assertEqual(neuron.V_reset, -70. * u.mV)
        self.assertEqual(neuron.tau_syn_ex, 2. * u.ms)
        self.assertEqual(neuron.tau_syn_in, 2. * u.ms)
        self.assertEqual(neuron.I_e, 0. * u.pA)
        self.assertIsNone(neuron.V_min)
        self.assertEqual(neuron.spk_reset, 'hard')

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            iaf_psc_alpha(1, C_m=0.0 * u.pF)
        with self.assertRaises(ValueError):
            iaf_psc_alpha(1, tau_m=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_alpha(1, tau_syn_ex=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_alpha(1, tau_syn_in=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_alpha(1, t_ref=-1.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_alpha(1, V_reset=-55. * u.mV, V_th=-55. * u.mV)

    def test_current_input_is_one_step_delayed(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_alpha(
                1,
                E_L=0. * u.mV,
                I_e=0. * u.pA,
                C_m=250. * u.pF,
                tau_m=10. * u.ms,
                V_th=1e9 * u.mV,
                V_reset=0. * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV),
            )
            neuron.init_state()

            self._step(neuron, 0, x=100. * u.pA)
            self.assertTrue(u.math.allclose(neuron.V.value, 0. * u.mV))

            self._step(neuron, 1, x=0. * u.pA)
            expected = (100. * u.pA) * (10. * u.ms) / (250. * u.pF) * (1. - u.math.exp(-self.dt / (10. * u.ms)))
            self.assertTrue(u.math.allclose(neuron.V.value, expected))

    def test_matches_nest_reference_i_e_trace(self):
        # Reference from NEST testsuite/pytests/sli2py_neurons/test_iaf_psc_alpha.py
        reference = {
            0.1: -69.6020,
            0.2: -69.2079,
            0.3: -68.8178,
            0.4: -68.4316,
            0.5: -68.0492,
            4.3: -56.0204,
            4.4: -55.7615,
            4.5: -55.5051,
            4.6: -55.2513,
            4.7: -55.0001,
            4.8: -70.0000,
            4.9: -70.0000,
            5.0: -70.0000,
        }

        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_alpha(
                1,
                I_e=1000. * u.pA,
                V_initializer=braintools.init.Constant(-70. * u.mV),
            )
            neuron.init_state()

            first_spike = None
            for k in range(80):
                spk = self._step(neuron, k)
                t_ms = round((k + 1) * 0.1, 1)
                if t_ms in reference:
                    vm = float((neuron.V.value / u.mV)[0])
                    self.assertAlmostEqual(vm, reference[t_ms], delta=8e-4)
                if first_spike is None and self._is_spike(spk):
                    first_spike = float((neuron.last_spike_time.value / u.ms)[0])

            self.assertIsNotNone(first_spike)
            self.assertAlmostEqual(first_spike, 4.8, delta=1e-12)

    def test_subthreshold_state_matches_reference_equations(self):
        with brainstate.environ.context(dt=self.dt):
            params = dict(
                E_L=-70.0,
                C_m=250.0,
                tau_m=10.0,
                t_ref=0.3,
                V_th=1e9,
                V_reset=-70.0,
                tau_syn_ex=2.0,
                tau_syn_in=3.0,
                I_e=45.0,
                V_min=-80.0,
            )
            neuron = iaf_psc_alpha(
                1,
                E_L=params['E_L'] * u.mV,
                C_m=params['C_m'] * u.pF,
                tau_m=params['tau_m'] * u.ms,
                t_ref=params['t_ref'] * u.ms,
                V_th=params['V_th'] * u.mV,
                V_reset=params['V_reset'] * u.mV,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                V_min=params['V_min'] * u.mV,
                V_initializer=braintools.init.Constant(-67.0 * u.mV),
            )
            neuron.init_state()

            x_seq = [10.0, 30.0, 0.0, -20.0, 50.0, 0.0, 0.0, 10.0, 0.0, 0.0]
            w_seq = [0.0, 100.0, -20.0, 0.0, 0.0, 50.0, -30.0, 0.0, 0.0, 0.0]

            ref_state = {
                'y0': 0.0,
                'dI_ex': 0.0,
                'I_ex': 0.0,
                'dI_in': 0.0,
                'I_in': 0.0,
                'y3': -67.0 - params['E_L'],
                'r': 0,
            }

            for k, (x_i, w_i) in enumerate(zip(x_seq, w_seq)):
                spk = self._step(neuron, k, x=x_i * u.pA, delta=w_i * u.pA)
                self.assertFalse(self._is_spike(spk))
                _reference_step(ref_state, params, x_i, w_i, 0.1)

                self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), ref_state['y3'] + params['E_L'], delta=1e-12)
                self.assertAlmostEqual(float((neuron.I_syn_ex.value / u.pA)[0]), ref_state['I_ex'], delta=1e-12)
                self.assertAlmostEqual(float((neuron.I_syn_in.value / u.pA)[0]), ref_state['I_in'], delta=1e-12)
                self.assertAlmostEqual(float(neuron.dI_syn_ex.value[0]), ref_state['dI_ex'], delta=1e-12)
                self.assertAlmostEqual(float(neuron.dI_syn_in.value[0]), ref_state['dI_in'], delta=1e-12)
                self.assertEqual(int(neuron.refractory_step_count.value[0]), ref_state['r'])

    def test_spike_refractory_trace_matches_reference_equations(self):
        with brainstate.environ.context(dt=self.dt):
            params = dict(
                E_L=-70.0,
                C_m=250.0,
                tau_m=10.0,
                t_ref=0.25,
                V_th=-55.0,
                V_reset=-70.0,
                tau_syn_ex=2.0,
                tau_syn_in=2.0,
                I_e=1000.0,
                V_min=None,
            )
            neuron = iaf_psc_alpha(
                1,
                E_L=params['E_L'] * u.mV,
                C_m=params['C_m'] * u.pF,
                tau_m=params['tau_m'] * u.ms,
                t_ref=params['t_ref'] * u.ms,
                V_th=params['V_th'] * u.mV,
                V_reset=params['V_reset'] * u.mV,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                V_initializer=braintools.init.Constant(-70.0 * u.mV),
            )
            neuron.init_state()

            x_seq = [0.0] * 60
            w_seq = [0.0] * 60
            # Add alpha-shaped excit/inhib input around refractory phases.
            w_seq[8] = 200.0
            w_seq[9] = -150.0
            w_seq[30] = 100.0

            ref_state = {
                'y0': 0.0,
                'dI_ex': 0.0,
                'I_ex': 0.0,
                'dI_in': 0.0,
                'I_in': 0.0,
                'y3': -70.0 - params['E_L'],
                'r': 0,
            }

            spike_times_model = []
            spike_times_ref = []
            for k, (x_i, w_i) in enumerate(zip(x_seq, w_seq)):
                spk = self._step(neuron, k, x=x_i * u.pA, delta=w_i * u.pA)
                spk_ref = _reference_step(ref_state, params, x_i, w_i, 0.1)
                if self._is_spike(spk):
                    spike_times_model.append(round((k + 1) * 0.1, 12))
                if spk_ref:
                    spike_times_ref.append(round((k + 1) * 0.1, 12))

                self.assertAlmostEqual(float((neuron.V.value / u.mV)[0]), ref_state['y3'] + params['E_L'], delta=1e-11)
                self.assertEqual(int(neuron.refractory_step_count.value[0]), ref_state['r'])

            self.assertEqual(spike_times_model, spike_times_ref)
            self.assertGreaterEqual(len(spike_times_model), 1)

    def test_singularity_is_stable_for_tau_syn_around_tau_m(self):
        with brainstate.environ.context(dt=self.dt):
            tau_m = 10.0
            delta_tau = np.hstack((-np.logspace(-10, -1, 5), [0.0], np.logspace(-10, -1, 5)))
            tau_syn = tau_m + delta_tau

            neuron = iaf_psc_alpha(
                len(tau_syn),
                tau_m=tau_m * u.ms,
                tau_syn_ex=tau_syn * u.ms,
                tau_syn_in=tau_syn * u.ms,
                V_th=np.inf * u.mV,
                V_initializer=braintools.init.Constant(-70. * u.mV),
            )
            neuron.init_state()

            trace = []
            for k in range(500):
                delta = None
                if k == 10:
                    delta = np.full_like(tau_syn, 1000.0) * u.pA
                if k == 20:
                    delta = np.full_like(tau_syn, -1000.0) * u.pA
                self._step(neuron, k, delta=delta)
                dftype = brainstate.environ.dftype()
                trace.append(np.asarray(u.math.asarray(neuron.V.value / u.mV), dtype=dftype))

            trace = np.asarray(trace)
            self.assertFalse(np.any(~np.isfinite(trace)))

            v_range = trace.max(axis=0) - trace.min(axis=0)
            self.assertTrue(np.all(v_range > 1.0))
            d_v = np.diff(trace, axis=0)
            bound = 3.0 * v_range * 0.1
            self.assertTrue(np.all(np.abs(d_v).max(axis=0) < bound))


if __name__ == '__main__':
    unittest.main()
