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
import unittest

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainpy.state import lin_rate_ipn, rate_connection_instantaneous

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


class TestRateConnectionInstantaneous(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    @staticmethod
    def _to_scalar(x):
        dftype = brainstate.environ.dftype()
        return float(np.asarray(x, dtype=dftype).reshape(-1)[0])

    def _step(self, model, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return model.update(**kwargs)

    def test_nest_default_parameters_and_properties(self):
        syn = rate_connection_instantaneous()
        self.assertAlmostEqual(syn.weight, 1.0, delta=0.0)
        self.assertEqual(syn.delay, 1)
        self.assertFalse(syn.HAS_DELAY)
        self.assertTrue(syn.SUPPORTS_WFR)
        self.assertEqual(syn.properties['has_delay'], False)
        self.assertEqual(syn.properties['supports_wfr'], True)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 1.0, delta=0.0)
        self.assertEqual(status['delay'], 1)
        self.assertEqual(status['has_delay'], False)
        self.assertEqual(status['supports_wfr'], True)

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            defaults = nest.GetDefaults('rate_connection_instantaneous')
            self.assertAlmostEqual(syn.weight, float(defaults['weight']), delta=0.0)
            self.assertIn('delay', defaults)
            self.assertGreaterEqual(float(defaults['delay']), float(nest.resolution))

    def test_set_status_and_delay_validation(self):
        syn = rate_connection_instantaneous()

        syn.set_status({'weight': -2.5})
        self.assertAlmostEqual(syn.weight, -2.5, delta=0.0)

        with self.assertRaisesRegex(ValueError, 'has no delay'):
            syn.set_status(delay=2)
        with self.assertRaisesRegex(ValueError, 'has no delay'):
            syn.set_status(delay_steps=2)
        with self.assertRaisesRegex(ValueError, 'has no delay'):
            syn.set_delay(2)
        with self.assertRaisesRegex(ValueError, 'has no delay'):
            syn.set_delay_steps(2)

        syn = rate_connection_instantaneous(weight=1.25)
        with self.assertRaisesRegex(ValueError, 'has no delay'):
            syn.set_status(weight=3.0, delay=2)
        self.assertAlmostEqual(syn.weight, 1.25, delta=0.0)

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            src = nest.Create('lin_rate_ipn')
            tgt = nest.Create('lin_rate_ipn')
            with self.assertRaisesRegex(nest.kernel.NESTError, 'has no delay'):
                nest.Connect(
                    src,
                    tgt,
                    syn_spec={'synapse_model': 'rate_connection_instantaneous', 'delay': 2.0},
                )


if __name__ == '__main__':
    unittest.main()
