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

import numpy as np
import numpy.testing as npt

import brainstate
import brainunit as u

from ._utils import (
    to_numpy,
    broadcast_to_state,
    refractory_counts,
    get_spike_scaled,
    check_positive,
    check_non_negative,
    check_reset_below_threshold,
    propagator_exp,
    alpha_propagator_p31_p32,
    rkf45_integrate,
    sum_signed_delta_inputs,
    time_window_gate,
)


class TestToNumpy(unittest.TestCase):
    def test_scalar(self):
        result = to_numpy(-70.0 * u.mV, u.mV)
        npt.assert_allclose(result, -70.0)
        self.assertEqual(result.dtype, np.float64)

    def test_array(self):
        x = np.array([-70.0, -55.0]) * u.mV
        result = to_numpy(x, u.mV)
        npt.assert_allclose(result, [-70.0, -55.0])

    def test_unit_conversion(self):
        result = to_numpy(250.0 * u.pF, u.pF)
        npt.assert_allclose(result, 250.0)


class TestBroadcastToState(unittest.TestCase):
    def test_scalar_to_shape(self):
        x = np.array(1.0)
        result = broadcast_to_state(x, (3,))
        self.assertEqual(result.shape, (3,))
        npt.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_already_correct_shape(self):
        x = np.array([1.0, 2.0, 3.0])
        result = broadcast_to_state(x, (3,))
        npt.assert_allclose(result, x)


class TestRefractoryCounts(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_exact_division(self):
        result = refractory_counts(2.0 * u.ms)
        self.assertEqual(int(result), 20)

    def test_ceil_rounding(self):
        result = refractory_counts(2.05 * u.ms)
        self.assertEqual(int(result), 21)

    def test_zero(self):
        result = refractory_counts(0.0 * u.ms)
        self.assertEqual(int(result), 0)


class TestGetSpikeScaled(unittest.TestCase):
    def test_above_threshold(self):
        V = -50.0 * u.mV
        V_th = -55.0 * u.mV
        V_reset = -70.0 * u.mV
        spk_fun = lambda x: x  # identity for testing
        result = get_spike_scaled(V, V_th, V_reset, spk_fun)
        # ((-50) - (-55)) / ((-55) - (-70)) = 5 / 15 = 1/3
        npt.assert_allclose(float(u.math.asarray(result)), 1.0 / 3.0, rtol=1e-6)

    def test_at_threshold(self):
        V = -55.0 * u.mV
        V_th = -55.0 * u.mV
        V_reset = -70.0 * u.mV
        spk_fun = lambda x: x
        result = get_spike_scaled(V, V_th, V_reset, spk_fun)
        npt.assert_allclose(float(u.math.asarray(result)), 0.0, atol=1e-10)


class TestCheckPositive(unittest.TestCase):
    def test_positive_passes(self):
        check_positive(250.0 * u.pF, u.pF, 'C_m')

    def test_zero_fails(self):
        with self.assertRaises(ValueError):
            check_positive(0.0 * u.pF, u.pF, 'C_m')

    def test_negative_fails(self):
        with self.assertRaises(ValueError):
            check_positive(-1.0 * u.pF, u.pF, 'C_m')


class TestCheckNonNegative(unittest.TestCase):
    def test_positive_passes(self):
        check_non_negative(2.0 * u.ms, u.ms, 't_ref')

    def test_zero_passes(self):
        check_non_negative(0.0 * u.ms, u.ms, 't_ref')

    def test_negative_fails(self):
        with self.assertRaises(ValueError):
            check_non_negative(-1.0 * u.ms, u.ms, 't_ref')


class TestCheckResetBelowThreshold(unittest.TestCase):
    def test_valid(self):
        check_reset_below_threshold(-70.0 * u.mV, -55.0 * u.mV)

    def test_equal_fails(self):
        with self.assertRaises(ValueError):
            check_reset_below_threshold(-55.0 * u.mV, -55.0 * u.mV)

    def test_above_fails(self):
        with self.assertRaises(ValueError):
            check_reset_below_threshold(-50.0 * u.mV, -55.0 * u.mV)


class TestPropagatorExp(unittest.TestCase):
    def test_known_values(self):
        tau_syn = np.array([2.0])
        tau_m = np.array([10.0])
        c_m = np.array([250.0])
        h = 0.1
        result = propagator_exp(tau_syn, tau_m, c_m, h)
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_singular_fallback(self):
        # tau_syn == tau_m triggers fallback
        tau = np.array([10.0])
        c_m = np.array([250.0])
        h = 0.1
        result = propagator_exp(tau, tau, c_m, h)
        expected = h / c_m[0] * math.exp(-h / tau[0])
        npt.assert_allclose(result, expected, rtol=1e-10)


class TestAlphaPropagator(unittest.TestCase):
    def test_returns_two_arrays(self):
        tau_syn = np.array([2.0])
        tau_m = np.array([10.0])
        c_m = np.array([250.0])
        h = 0.1
        p31, p32 = alpha_propagator_p31_p32(tau_syn, tau_m, c_m, h)
        self.assertTrue(np.all(np.isfinite(p31)))
        self.assertTrue(np.all(np.isfinite(p32)))

    def test_singular_fallback(self):
        tau = np.array([10.0])
        c_m = np.array([250.0])
        h = 0.1
        p31, p32 = alpha_propagator_p31_p32(tau, tau, c_m, h)
        exp_h = math.exp(-h / tau[0])
        npt.assert_allclose(p32, h / c_m[0] * exp_h, rtol=1e-10)
        npt.assert_allclose(p31, 0.5 * h * h / c_m[0] * exp_h, rtol=1e-10)


class TestRKF45Integrate(unittest.TestCase):
    def test_exponential_decay(self):
        # dy/dt = -y, y(0) = 1, exact solution: y(dt) = exp(-dt)
        def dynamics(y):
            return (-y,)

        dt = 1.0
        y_final, h_final = rkf45_integrate(dynamics, (1.0,), dt, 0.1)
        npt.assert_allclose(y_final[0], math.exp(-1.0), rtol=1e-3)

    def test_two_variable_system(self):
        # dy1/dt = -y1/2, dy2/dt = -y2/5
        def dynamics(y1, y2):
            return (-y1 / 2.0, -y2 / 5.0)

        dt = 0.1
        y_final, h_final = rkf45_integrate(dynamics, (1.0, 1.0), dt, 0.01)
        npt.assert_allclose(y_final[0], math.exp(-0.05), rtol=1e-4)
        npt.assert_allclose(y_final[1], math.exp(-0.02), rtol=1e-4)


class TestSumSignedDeltaInputs(unittest.TestCase):
    def test_none_inputs(self):
        zero = u.math.zeros((3,)) * u.nS
        g_ex, g_in = sum_signed_delta_inputs(None, zero, zero)
        npt.assert_allclose(u.math.asarray(g_ex / u.nS), 0.0)
        npt.assert_allclose(u.math.asarray(g_in / u.nS), 0.0)

    def test_positive_negative_split(self):
        zero = u.math.zeros((1,)) * u.nS
        inputs = {
            'exc': lambda: 5.0 * u.nS,
            'inh': lambda: -3.0 * u.nS,
        }
        g_ex, g_in = sum_signed_delta_inputs(inputs, zero, zero)
        npt.assert_allclose(np.asarray(u.math.asarray(g_ex / u.nS)), 5.0)
        npt.assert_allclose(np.asarray(u.math.asarray(g_in / u.nS)), 3.0)


class TestTimeWindowGate(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_inside_window(self):
        with brainstate.environ.context(t=5.0 * u.ms):
            result = time_window_gate(
                100.0 * u.pA,
                0.0 * u.ms,
                2.0 * u.ms,
                10.0 * u.ms,
            )
        npt.assert_allclose(float(u.math.asarray(result / u.pA)), 100.0)

    def test_before_window(self):
        with brainstate.environ.context(t=1.0 * u.ms):
            result = time_window_gate(
                100.0 * u.pA,
                0.0 * u.ms,
                2.0 * u.ms,
                10.0 * u.ms,
            )
        npt.assert_allclose(float(u.math.asarray(result / u.pA)), 0.0)

    def test_at_stop_excluded(self):
        with brainstate.environ.context(t=10.0 * u.ms):
            result = time_window_gate(
                100.0 * u.pA,
                0.0 * u.ms,
                2.0 * u.ms,
                10.0 * u.ms,
            )
        npt.assert_allclose(float(u.math.asarray(result / u.pA)), 0.0)

    def test_no_stop(self):
        with brainstate.environ.context(t=1000.0 * u.ms):
            result = time_window_gate(
                100.0 * u.pA,
                0.0 * u.ms,
                2.0 * u.ms,
                None,
            )
        npt.assert_allclose(float(u.math.asarray(result / u.pA)), 100.0)


if __name__ == '__main__':
    unittest.main()
