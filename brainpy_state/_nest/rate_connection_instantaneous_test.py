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
import numpy as np
import numpy.testing as npt
from brainpy.state import lin_rate_ipn, rate_connection_instantaneous

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _run_nest_two_rate_trace(dt_ms, simtime_ms, source_params, target_params, weight):
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = True
    nest.local_num_threads = 1

    src = nest.Create('lin_rate_ipn', params=source_params)
    tgt = nest.Create('lin_rate_ipn', params=target_params)

    nest.Connect(
        src,
        tgt,
        syn_spec={
            'synapse_model': 'rate_connection_instantaneous',
            'weight': weight,
        },
    )

    mm_src = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt_ms})
    mm_tgt = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt_ms})
    nest.Connect(mm_src, src, syn_spec={'delay': dt_ms})
    nest.Connect(mm_tgt, tgt, syn_spec={'delay': dt_ms})

    nest.Simulate(simtime_ms)
    dftype = brainstate.environ.dftype()
    src_rate = np.asarray(mm_src.events['rate'], dtype=dftype)
    tgt_rate = np.asarray(mm_tgt.events['rate'], dtype=dftype)
    return src_rate, tgt_rate


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

    def test_event_helpers_match_nest_instantaneous_receiver_mapping(self):
        syn = rate_connection_instantaneous(weight=-0.75)
        dftype = brainstate.environ.dftype()
        coeff = np.asarray([0.2, -0.3, 1.0], dtype=dftype)

        sec = syn.prepare_secondary_event(coeff)
        npt.assert_allclose(sec['coeffarray'], coeff, atol=0.0, rtol=0.0)
        self.assertAlmostEqual(sec['weight'], -0.75, delta=0.0)

        event = syn.to_rate_event(rate=np.asarray([1.2], dtype=dftype), multiplicity=2.0)
        self.assertAlmostEqual(event['rate'], 1.2, delta=0.0)
        self.assertAlmostEqual(event['weight'], -0.75, delta=0.0)
        self.assertEqual(event['delay_steps'], 0)
        self.assertAlmostEqual(event['multiplicity'], 2.0, delta=0.0)

        mapped = syn.coeffarray_to_step_events(coeff, first_delay_steps=1, multiplicity=4.0)
        self.assertEqual(len(mapped), coeff.size)
        for i, ev in enumerate(mapped):
            self.assertAlmostEqual(ev['rate'], float(coeff[i]), delta=0.0)
            self.assertAlmostEqual(ev['weight'], -0.75, delta=0.0)
            self.assertEqual(ev['delay_steps'], 1 + i)
            self.assertAlmostEqual(ev['multiplicity'], 4.0, delta=0.0)

        with self.assertRaisesRegex(ValueError, 'must be 0'):
            syn.to_rate_event(rate=1.0, delay_steps=1)
        with self.assertRaisesRegex(ValueError, 'must be >= 0'):
            syn.coeffarray_to_step_events(coeff, first_delay_steps=-1)

    def test_two_population_trace_matches_nest(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        weight = 0.5
        source_params = {
            'tau': 4.0,
            'lambda': 1.0,
            'mu': 1.5,
            'sigma': 0.0,
            'rate': 0.0,
        }
        target_params = {
            'tau': 5.0,
            'lambda': 1.0,
            'mu': 0.0,
            'sigma': 0.0,
            'rate': 0.0,
        }

        steps = 500
        simtime_ms = steps * self.dt_ms
        nest_src, nest_tgt = _run_nest_two_rate_trace(
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            source_params=source_params,
            target_params=target_params,
            weight=weight,
        )

        replay_steps = min(nest_src.size, nest_tgt.size)
        dftype = brainstate.environ.dftype()
        bp_src = np.zeros((replay_steps,), dtype=dftype)
        bp_tgt = np.zeros((replay_steps,), dtype=dftype)

        with brainstate.environ.context(dt=self.dt):
            src = lin_rate_ipn(
                1,
                tau=source_params['tau'] * u.ms,
                lambda_=source_params['lambda'],
                mu=source_params['mu'],
                sigma=source_params['sigma'],
                rate_initializer=braintools.init.Constant(source_params['rate']),
            )
            tgt = lin_rate_ipn(
                1,
                tau=target_params['tau'] * u.ms,
                lambda_=target_params['lambda'],
                mu=target_params['mu'],
                sigma=target_params['sigma'],
                rate_initializer=braintools.init.Constant(target_params['rate']),
            )
            src.init_state()
            tgt.init_state()
            syn = rate_connection_instantaneous(weight=weight)

            for k in range(replay_steps):
                self._step(src, k, noise=0.0)
                # NEST event delivery aligns instantaneous rate communication
                # with the next scheduler slice (min-delay cycle).
                event = syn.to_rate_event(rate=src.delayed_rate.value)
                self._step(tgt, k, instant_rate_events=[event], noise=0.0)
                bp_src[k] = self._to_scalar(src.rate.value)
                bp_tgt[k] = self._to_scalar(tgt.rate.value)

        npt.assert_allclose(bp_src, nest_src[:replay_steps], atol=1e-12, rtol=0.0)
        npt.assert_allclose(bp_tgt, nest_tgt[:replay_steps], atol=1e-10, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
