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
import brainunit as u
import jax

from brainpy_state import sic_connection, aeif_cond_alpha_astro, astrocyte_lr_1994

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


class TestSICConnection(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def test_nest_default_parameters_and_properties(self):
        syn = sic_connection()
        self.assertAlmostEqual(syn.weight, 1.0, delta=0.0)
        self.assertEqual(syn.delay_steps, 1)
        self.assertTrue(syn.HAS_DELAY)
        self.assertFalse(syn.SUPPORTS_WFR)
        self.assertEqual(syn.properties['has_delay'], True)
        self.assertEqual(syn.properties['supports_wfr'], False)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 1.0, delta=0.0)
        self.assertEqual(status['delay_steps'], 1)
        self.assertEqual(status['delay'], 1)
        self.assertIn('size_of', status)
        self.assertEqual(status['supported_sources'], ('astrocyte_lr_1994',))
        self.assertEqual(status['supported_targets'], ('aeif_cond_alpha_astro',))

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            defaults = nest.GetDefaults('sic_connection')
            self.assertAlmostEqual(syn.weight, float(defaults['weight']), delta=0.0)
            self.assertIn('delay', defaults)
            self.assertGreaterEqual(float(defaults['delay']), float(nest.resolution))

    def test_set_status_delay_validation_and_supported_pairs(self):
        syn = sic_connection()
        syn.set_status({'weight': 2.5, 'delay_steps': 7})
        self.assertAlmostEqual(syn.weight, 2.5, delta=0.0)
        self.assertEqual(syn.delay_steps, 7)

        syn.set_status(delay=4)
        self.assertEqual(syn.delay_steps, 4)

        with self.assertRaisesRegex(ValueError, 'must be >= 1'):
            syn.set_delay_steps(0)
        with self.assertRaisesRegex(ValueError, 'must be >= 1'):
            syn.set_delay(-1)
        with self.assertRaisesRegex(ValueError, 'integer-valued'):
            syn.set_status(delay=2.5)
        with self.assertRaisesRegex(ValueError, 'must be identical'):
            syn.set_status(delay=2, delay_steps=3)

        self.assertTrue(syn.supports_connection('astrocyte_lr_1994', 'aeif_cond_alpha_astro'))
        self.assertFalse(syn.supports_connection('iaf_psc_exp', 'aeif_cond_alpha_astro'))
        self.assertFalse(syn.supports_connection('astrocyte_lr_1994', 'iaf_psc_exp'))
        with self.assertRaisesRegex(ValueError, 'Unsupported sic_connection pair'):
            syn.check_connection('iaf_psc_exp', 'aeif_cond_alpha_astro')

        if _is_nest_available():
            import nest

            models = ['astrocyte_lr_1994', 'aeif_cond_alpha_astro', 'iaf_psc_exp']
            if not set(models).issubset(set(nest.Models())):
                self.skipTest('Required NEST models for sic_connection pair checks are not available.')

            for src_model in models:
                for tgt_model in models:
                    expected = syn.supports_connection(src_model, tgt_model)
                    nest.ResetKernel()
                    src = nest.Create(src_model)
                    tgt = nest.Create(tgt_model)
                    if expected:
                        nest.Connect(src, tgt, syn_spec={'synapse_model': 'sic_connection'})
                    else:
                        with self.assertRaises(nest.kernel.NESTError):
                            nest.Connect(src, tgt, syn_spec={'synapse_model': 'sic_connection'})

    def test_host_queue_api_removed(self):
        # The bucket-3 de-queue: the host-side coefficient-array/event-emulator API
        # (prepare_secondary_event / to_aeif_sic_event / to_sic_event /
        # coeffarray_to_step_events) is removed. The Simulator builds the routing
        # from the spec's weight / delay_steps + sender/receiver enforcement instead.
        syn = sic_connection()
        for gone in ('prepare_secondary_event', 'to_aeif_sic_event', 'to_sic_event',
                     'coeffarray_to_step_events', '_to_local_delay_steps', '_to_coeff_array'):
            self.assertFalse(hasattr(syn, gone), f'{gone} should be removed')
        # The thin NEST-parity spec survives: weight, delay_steps, status, enforcement.
        self.assertEqual(syn.weight, 1.0)
        self.assertEqual(syn.delay_steps, 1)
        self.assertIn('weight', syn.get_status())


class TestSICConnectionStatusAndCoercion(unittest.TestCase):
    """The kept NEST-parity spec: status getter, scalar coercion, model-name resolution."""

    def test_get_returns_status_value_or_raises(self):
        syn = sic_connection(weight=0.8, delay_steps=2)
        self.assertEqual(syn.get('status'), syn.get_status())
        self.assertAlmostEqual(syn.get('weight'), 0.8, delta=0.0)
        self.assertEqual(syn.get('delay_steps'), 2)
        with self.assertRaisesRegex(KeyError, 'Unsupported key'):
            syn.get('not_a_field')

    def test_set_status_delay_steps_key(self):
        # The 'delay_steps' branch of set_status routes through set_delay_steps.
        syn = sic_connection()
        syn.set_status({'delay_steps': 5})
        self.assertEqual(syn.delay_steps, 5)
        # Supplying delay and delay_steps together is accepted when they agree.
        syn.set_status(delay=4, delay_steps=4)
        self.assertEqual(syn.delay_steps, 4)

    def test_scalar_coercion_strips_units_and_rejects_bad_shapes(self):
        # _to_float_scalar / _to_int_scalar strip arbitrary units and validate shape.
        self.assertAlmostEqual(sic_connection._to_float_scalar(2.5 * u.mV, 'weight'), 2.5)
        self.assertEqual(sic_connection._to_int_scalar(3.0 * u.ms, 'delay'), 3)
        with self.assertRaisesRegex(ValueError, 'must be scalar'):
            sic_connection._to_float_scalar(np.array([1.0, 2.0]), 'weight')
        with self.assertRaisesRegex(ValueError, 'must be scalar'):
            sic_connection._to_int_scalar(np.array([1, 2]), 'delay')
        with self.assertRaisesRegex(ValueError, 'must be finite'):
            sic_connection._to_int_scalar(float('inf'), 'delay')

    def test_model_name_resolves_class_and_instance(self):
        # supports_connection accepts string names, classes, and instances; the
        # sender/receiver contract holds however the model is identified.
        self.assertTrue(
            sic_connection.supports_connection(astrocyte_lr_1994, aeif_cond_alpha_astro))
        # An instance routes through __class__.__name__ (here a non-supported source).
        self.assertFalse(
            sic_connection.supports_connection(sic_connection(), aeif_cond_alpha_astro))


if __name__ == '__main__':
    unittest.main()
