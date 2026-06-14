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
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from brainpy.state import siegert_neuron

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class TestSiegertNeuron(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(**kwargs)

    def test_nest_default_parameters(self):
        nrn = siegert_neuron(1)
        self.assertEqual(nrn.tau, 1.0 * u.ms)
        self.assertEqual(nrn.tau_m, 5.0 * u.ms)
        self.assertEqual(nrn.tau_syn, 0.0 * u.ms)
        self.assertEqual(nrn.t_ref, 2.0 * u.ms)
        self.assertEqual(nrn.mean, 0.0)
        self.assertEqual(nrn.theta, 15.0)
        self.assertEqual(nrn.V_reset, 0.0)
        self.assertEqual(nrn.recordables, ['rate'])
        self.assertEqual(nrn.receptor_types, {'DIFFUSION': 1})

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            siegert_neuron(1, tau=0.0 * u.ms)
        with self.assertRaises(ValueError):
            siegert_neuron(1, tau_m=0.0 * u.ms)
        with self.assertRaises(ValueError):
            siegert_neuron(1, tau_syn=-1e-3 * u.ms)
        with self.assertRaises(ValueError):
            siegert_neuron(1, t_ref=-1e-3 * u.ms)
        with self.assertRaises(ValueError):
            siegert_neuron(1, theta=1.0, V_reset=1.0)


    def test_matches_nest_reference_value_at_threshold(self):
        nrn = siegert_neuron(
            1,
            tau_m=10.0 * u.ms,
            t_ref=2.0 * u.ms,
            theta=15.0,
            V_reset=0.0,
        )
        mu = 15.0
        sigma = np.sqrt(0.1 * mu)
        pred = float(np.asarray(nrn.siegert_rate(np.asarray([mu]), np.asarray([sigma * sigma]))).reshape(-1)[0])

        # Hard-coded reference used by NEST's test_siegert_neuron.py
        self.assertAlmostEqual(pred, 27.1095934379, delta=3e-6)

    def test_noisefree_limit_matches_analytic(self):
        nrn = siegert_neuron(
            1,
            tau_m=10.0 * u.ms,
            t_ref=2.0 * u.ms,
            theta=15.0,
            V_reset=0.0,
        )

        def noisefree(mu):
            if mu > 15.0:
                return 1e3 / (2.0 + 10.0 * np.log((mu - 0.0) / (mu - 15.0)))
            return 0.0

        sigma = 1e-3 * 15.0
        sigma_square = sigma * sigma

        mu_sup = 22.5
        pred_sup = float(np.asarray(nrn.siegert_rate(np.asarray([mu_sup]), np.asarray([sigma_square]))).reshape(-1)[0])
        self.assertTrue(np.isclose(pred_sup, noisefree(mu_sup), rtol=5e-6, atol=5e-6))

        mu_sub = 13.5
        pred_sub = float(np.asarray(nrn.siegert_rate(np.asarray([mu_sub]), np.asarray([sigma_square]))).reshape(-1)[0])
        self.assertLess(abs(pred_sub - noisefree(mu_sub)), 1e-6)


if __name__ == '__main__':
    unittest.main()
