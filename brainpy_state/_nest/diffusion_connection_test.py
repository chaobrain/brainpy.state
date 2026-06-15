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
import numpy as np
import saiunit as u
import jax
from brainpy.state import diffusion_connection

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


class TestDiffusionConnection(unittest.TestCase):
    """``diffusion_connection`` status/parameter parity with NEST.

    The dual-channel network routing (drift -> mu, diffusion -> sigma^2) and its
    live-NEST trace parity are validated in
    ``brainpy_state/_nest/_validation/siegert_diffusion_test.py``; this module
    covers the connection object's NEST-compatible status spec.
    """

    def test_nest_default_parameters_and_properties(self):
        syn = diffusion_connection()
        self.assertAlmostEqual(syn.drift_factor, 1.0, delta=0.0)
        self.assertAlmostEqual(syn.diffusion_factor, 1.0, delta=0.0)
        self.assertTrue(syn.SUPPORTS_WFR)
        self.assertFalse(syn.HAS_DELAY)
        self.assertEqual(syn.properties['supports_wfr'], True)
        self.assertEqual(syn.properties['has_delay'], False)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 1.0, delta=0.0)
        self.assertIsNone(status['delay'])
        self.assertAlmostEqual(status['drift_factor'], 1.0, delta=0.0)
        self.assertAlmostEqual(status['diffusion_factor'], 1.0, delta=0.0)

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            defaults = nest.GetDefaults('diffusion_connection')
            self.assertAlmostEqual(syn.drift_factor, float(defaults['drift_factor']), delta=0.0)
            self.assertAlmostEqual(syn.diffusion_factor, float(defaults['diffusion_factor']), delta=0.0)

    def test_set_status_and_weight_delay_restrictions(self):
        syn = diffusion_connection()
        syn.set_status({'drift_factor': 2.5})
        syn.set_status(diffusion_factor=3.2)
        self.assertAlmostEqual(syn.drift_factor, 2.5, delta=0.0)
        self.assertAlmostEqual(syn.diffusion_factor, 3.2, delta=0.0)

        with self.assertRaisesRegex(ValueError, 'drift_factor and diffusion_factor'):
            syn.set_status({'weight': 2.0})
        with self.assertRaisesRegex(ValueError, 'drift_factor and diffusion_factor'):
            syn.set_weight(2.0)

        with self.assertRaisesRegex(ValueError, 'diffusion_connection has no delay'):
            syn.set_status({'delay': 2.0})
        with self.assertRaisesRegex(ValueError, 'diffusion_connection has no delay'):
            syn.set_delay(2.0)

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            src = nest.Create('siegert_neuron')
            tgt = nest.Create('siegert_neuron')

            with self.assertRaisesRegex(nest.kernel.NESTError, 'drift_factor and diffusion_factor'):
                nest.Connect(
                    src,
                    tgt,
                    syn_spec={'synapse_model': 'diffusion_connection', 'weight': 2.0},
                )

            with self.assertRaisesRegex(nest.kernel.NESTError, 'diffusion_connection has no delay'):
                nest.Connect(
                    src,
                    tgt,
                    syn_spec={'synapse_model': 'diffusion_connection', 'delay': 2.0},
                )

    def test_is_thin_diffusion_spec(self):
        # Goal 15c: diffusion_connection is a thin NEST-parity status spec dispatched
        # by the Simulator; the host dict-queue event emulator is retired.
        syn = diffusion_connection()
        self.assertIs(syn._IS_DIFFUSION, True)
        for gone in ('prepare_secondary_event', 'project_coeffarray',
                     'to_siegert_event', 'coeffarray_to_step_events'):
            self.assertFalse(hasattr(syn, gone), f'{gone} should be removed')

    def test_get_accessor_and_scalar_coercion(self):
        # The retained NEST-parity accessors after the de-queue: get() and the
        # scalar coercion that set_drift_factor / set_diffusion_factor share.
        syn = diffusion_connection(drift_factor=1.5, diffusion_factor=2.5)
        self.assertEqual(syn.get('status'), syn.get_status())
        self.assertAlmostEqual(syn.get('drift_factor'), 1.5, delta=0.0)
        self.assertAlmostEqual(syn.get('diffusion_factor'), 2.5, delta=0.0)
        with self.assertRaisesRegex(KeyError, 'invalid_key'):
            syn.get('invalid_key')

        # A (dimensionless) Quantity is accepted -- its mantissa is taken ...
        syn.set_drift_factor(u.Quantity(0.25))
        self.assertAlmostEqual(syn.drift_factor, 0.25, delta=0.0)
        # ... and a non-scalar factor is rejected.
        with self.assertRaisesRegex(ValueError, 'must be scalar'):
            syn.set_diffusion_factor(np.array([1.0, 2.0]))


if __name__ == '__main__':
    unittest.main()
