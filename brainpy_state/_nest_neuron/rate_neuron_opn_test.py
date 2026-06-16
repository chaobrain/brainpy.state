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
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from brainpy.state import lin_rate_opn, rate_neuron_opn

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class TestRateNeuronOPN(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(**kwargs)

    def test_nest_default_parameters(self):
        opn = rate_neuron_opn(1)
        self.assertEqual(opn.tau, 10.0 * u.ms)
        self.assertEqual(opn.sigma, 1.0)
        self.assertEqual(opn.mu, 0.0)
        self.assertEqual(opn.g, 1.0)
        self.assertEqual(opn.mult_coupling, False)
        self.assertEqual(opn.g_ex, 1.0)
        self.assertEqual(opn.g_in, 1.0)
        self.assertEqual(opn.theta_ex, 0.0)
        self.assertEqual(opn.theta_in, 0.0)
        self.assertEqual(opn.linear_summation, True)
        self.assertEqual(opn.recordables, ['rate', 'noise', 'noisy_rate'])
        self.assertEqual(opn.receptor_types, {'RATE': 0})

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            rate_neuron_opn(1, tau=0.0 * u.ms)
        with self.assertRaises(ValueError):
            rate_neuron_opn(1, sigma=-1e-3)


if __name__ == '__main__':
    unittest.main()
