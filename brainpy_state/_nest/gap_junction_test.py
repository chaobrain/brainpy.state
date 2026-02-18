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
import brainunit as u
import jax
import numpy as np
import numpy.testing as npt
from brainpy.state import gap_junction

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _nest_gap_current_reference(sumj_g_ij, interpolation_coefficients, lag, interpolation_order, v_m, t):
    dftype = brainstate.environ.dftype()
    v = np.asarray(v_m, dtype=dftype)
    tt = np.asarray(t, dtype=dftype)
    coeff = np.asarray(interpolation_coefficients, dtype=dftype)
    base = -float(sumj_g_ij) * v

    if interpolation_order == 0:
        return base + coeff[lag]
    if interpolation_order == 1:
        i0 = lag * 2
        return base + coeff[i0] + coeff[i0 + 1] * tt
    if interpolation_order == 3:
        i0 = lag * 4
        t2 = tt * tt
        return base + coeff[i0] + coeff[i0 + 1] * tt + coeff[i0 + 2] * t2 + coeff[i0 + 3] * t2 * tt
    raise ValueError('Interpolation order must be 0, 1, or 3.')


class TestGapJunction(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_default_parameters_and_properties(self):
        syn = gap_junction()
        self.assertAlmostEqual(syn.weight, 1.0, delta=0.0)
        self.assertTrue(syn.REQUIRES_SYMMETRIC)
        self.assertTrue(syn.SUPPORTS_WFR)
        self.assertEqual(syn.SUPPORTED_WFR_INTERPOLATION_ORDERS, (0, 1, 3))
        self.assertEqual(syn.properties['requires_symmetric'], True)
        self.assertEqual(syn.properties['supports_wfr'], True)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 1.0, delta=0.0)
        self.assertIsNone(status['delay'])

    def test_set_status_and_delay_restriction(self):
        syn = gap_junction(weight=2.5)
        syn.set_status({'weight': 3.0})
        self.assertAlmostEqual(syn.weight, 3.0, delta=0.0)

        with self.assertRaisesRegex(ValueError, 'gap_junction connection has no delay'):
            syn.set_status({'delay': 2.0})
        with self.assertRaisesRegex(ValueError, 'gap_junction connection has no delay'):
            syn.set_delay(2.0)

    def test_handle_gap_event_matches_nest_accumulation_logic(self):
        syn = gap_junction(weight=9.0)
        syn.begin_wfr_cycle(min_delay_steps=3, interpolation_order=1)

        dftype = brainstate.environ.dftype()
        c0 = np.asarray([0.2, -0.3, 1.2, 0.7, -0.6, 0.9], dtype=dftype)
        c1 = np.asarray([-0.5, 0.4, -0.1, 2.0, 0.2, -1.0], dtype=dftype)

        syn.handle_gap_event(c0, weight=0.5)
        syn.handle_gap_event(c1, weight=1.2)

        expected_sum = 0.5 + 1.2
        expected_coeff = 0.5 * c0 + 1.2 * c1

        self.assertAlmostEqual(syn.sumj_g_ij, expected_sum, delta=1e-15)
        npt.assert_allclose(syn.interpolation_coefficients, expected_coeff, atol=1e-15, rtol=0.0)

        with self.assertRaisesRegex(ValueError, 'Coefficient size mismatch'):
            syn.handle_gap_event(np.asarray([1.0, 2.0], dtype=dftype), weight=1.0)

    def test_evaluate_gap_current_orders(self):
        v_m = -61.3
        lag = 1
        t = 0.37
        sumj_g_ij = 2.4

        dftype = brainstate.environ.dftype()
        for order, coeff in [
            (0, np.asarray([0.5, -0.2], dtype=dftype)),
            (1, np.asarray([0.1, 0.7, -0.3, 0.2], dtype=dftype)),
            (3, np.asarray([1.0, 0.1, -0.2, 0.3, 0.5, -0.4, 0.2, -0.1], dtype=dftype)),
        ]:
            syn = gap_junction()
            syn.begin_wfr_cycle(min_delay_steps=2, interpolation_order=order)
            syn.sumj_g_ij = sumj_g_ij
            syn.interpolation_coefficients[:] = coeff

            got = syn.evaluate_gap_current(v_m, lag=lag, t=t, interpolation_order=order)
            ref = _nest_gap_current_reference(sumj_g_ij, coeff, lag, order, v_m, t)
            self.assertAlmostEqual(float(np.asarray(got).reshape(-1)[0]), float(np.asarray(ref).reshape(-1)[0]),
                                   delta=1e-15)

        syn = gap_junction()
        syn.begin_wfr_cycle(min_delay_steps=1, interpolation_order=0)
        with self.assertRaisesRegex(ValueError, 'Interpolation order must be 0, 1, or 3'):
            syn.evaluate_gap_current(-70.0, lag=0, interpolation_order=2)

    def test_reference_trace_matches_nest_update_order(self):
        order = 3
        min_delay_steps = 2
        n_coeff = min_delay_steps * (order + 1)

        dftype = brainstate.environ.dftype()
        events_by_cycle = [
            [
                (0.8, np.asarray([0.2, -0.1, 0.3, 0.5, -0.4, 0.1, 0.2, -0.2], dtype=dftype)),
                (1.1, np.asarray([-0.3, 0.2, 0.1, -0.5, 0.7, -0.2, 0.1, 0.3], dtype=dftype)),
            ],
            [
                (0.4, np.asarray([0.6, 0.0, -0.2, 0.4, 0.2, -0.3, 0.1, -0.1], dtype=dftype)),
            ],
            [
                (1.5, np.asarray([-0.5, 0.7, -0.3, 0.2, -0.1, 0.6, -0.4, 0.3], dtype=dftype)),
                (0.3, np.asarray([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4], dtype=dftype)),
            ],
        ]
        vm_by_cycle_lag = np.asarray(
            [
                [-68.0, -66.5],
                [-64.2, -62.8],
                [-61.0, -59.7],
            ],
            dtype=dftype,
        )
        t_values = np.asarray([0.0, 0.25, 0.5, 0.9], dtype=dftype)

        syn = gap_junction(weight=1.0)

        for cycle, events in enumerate(events_by_cycle):
            syn.begin_wfr_cycle(min_delay_steps=min_delay_steps, interpolation_order=order)
            expected_sum = 0.0
            expected_coeff = np.zeros((n_coeff,), dtype=dftype)

            for w, coeff in events:
                syn.handle_gap_event(coeff, weight=w)
                expected_sum += w
                expected_coeff += w * coeff

            self.assertAlmostEqual(syn.sumj_g_ij, expected_sum, delta=1e-15)
            npt.assert_allclose(syn.interpolation_coefficients, expected_coeff, atol=1e-15, rtol=0.0)

            for lag in range(min_delay_steps):
                v_m = vm_by_cycle_lag[cycle, lag]
                for t in t_values:
                    got = syn.evaluate_gap_current(v_m, lag=lag, t=t, interpolation_order=order)
                    ref = _nest_gap_current_reference(expected_sum, expected_coeff, lag, order, v_m, t)
                    self.assertAlmostEqual(
                        float(np.asarray(got).reshape(-1)[0]),
                        float(np.asarray(ref).reshape(-1)[0]),
                        delta=1e-14,
                    )

            syn.reset_runtime_state()
            self.assertAlmostEqual(syn.sumj_g_ij, 0.0, delta=0.0)
            npt.assert_allclose(syn.interpolation_coefficients, np.zeros_like(syn.interpolation_coefficients), atol=0.0)

    def test_matches_nest_defaults_and_delay_error_if_available(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        nest.ResetKernel()
        defaults = nest.GetDefaults('gap_junction')

        syn = gap_junction()
        self.assertAlmostEqual(syn.weight, float(defaults['weight']), delta=0.0)

        src = nest.Create('hh_psc_alpha_gap')
        tgt = nest.Create('hh_psc_alpha_gap')
        with self.assertRaisesRegex(nest.kernel.NESTError, 'gap_junction connection has no delay'):
            nest.Connect(
                src,
                tgt,
                conn_spec={'rule': 'one_to_one', 'make_symmetric': True},
                syn_spec={'synapse_model': 'gap_junction', 'delay': 2.0},
            )

        with self.assertRaisesRegex(ValueError, 'gap_junction connection has no delay'):
            syn.set_status({'delay': 2.0})
