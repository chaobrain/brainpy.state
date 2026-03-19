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

from brainpy_state._nest._utils import (
    propagator_exp,
    alpha_propagator_p31_p32,
)


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


if __name__ == '__main__':
    unittest.main()
