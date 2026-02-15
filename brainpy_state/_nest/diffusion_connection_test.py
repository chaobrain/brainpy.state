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

from brainpy.state import diffusion_connection, siegert_neuron

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _run_nest_two_siegert_trace(dt_ms, simtime_ms, src_params, tgt_params, drift_factor, diffusion_factor):
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = False
    nest.local_num_threads = 1

    src = nest.Create('siegert_neuron', params=src_params)
    tgt = nest.Create('siegert_neuron', params=tgt_params)

    nest.Connect(
        src,
        tgt,
        syn_spec={
            'synapse_model': 'diffusion_connection',
            'drift_factor': drift_factor,
            'diffusion_factor': diffusion_factor,
        },
    )

    mm_src = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt_ms})
    mm_tgt = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt_ms})
    nest.Connect(mm_src, src, syn_spec={'delay': dt_ms})
    nest.Connect(mm_tgt, tgt, syn_spec={'delay': dt_ms})

    nest.Simulate(simtime_ms)
    src_rate = np.asarray(mm_src.events['rate'], dtype=np.float64)
    tgt_rate = np.asarray(mm_tgt.events['rate'], dtype=np.float64)
    return src_rate, tgt_rate


class TestDiffusionConnection(unittest.TestCase):
    def setUp(self):
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(**kwargs)

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

    def test_coeff_projection_and_event_helpers_match_nest_semantics(self):
        syn = diffusion_connection(drift_factor=1.3, diffusion_factor=-0.4)
        coeff = np.asarray([0.5, -2.0, 1.25, 0.0], dtype=np.float64)

        event = syn.prepare_secondary_event(coeff)
        npt.assert_allclose(event['coeffarray'], coeff, atol=0.0, rtol=0.0)
        self.assertAlmostEqual(event['drift_factor'], 1.3, delta=0.0)
        self.assertAlmostEqual(event['diffusion_factor'], -0.4, delta=0.0)

        drift, diffusion = syn.project_coeffarray(coeff)
        npt.assert_allclose(drift, 1.3 * coeff, atol=1e-15, rtol=0.0)
        npt.assert_allclose(diffusion, -0.4 * coeff, atol=1e-15, rtol=0.0)

        mapped = syn.coeffarray_to_step_events(coeff, first_delay_steps=2, multiplicity=3.0)
        self.assertEqual(len(mapped), coeff.size)
        for i, ev in enumerate(mapped):
            self.assertAlmostEqual(ev['coeff'], float(coeff[i]), delta=0.0)
            self.assertEqual(ev['delay_steps'], 2 + i)
            self.assertAlmostEqual(ev['drift_factor'], 1.3, delta=0.0)
            self.assertAlmostEqual(ev['diffusion_factor'], -0.4, delta=0.0)
            self.assertAlmostEqual(ev['multiplicity'], 3.0, delta=0.0)

    def test_two_population_step_logic_matches_manual_reference(self):
        drift_factor = 7.5
        diffusion_factor = 1.2
        source_params = dict(
            tau=2.2,
            mean=0.9,
            tau_m=10.0,
            tau_syn=0.3,
            t_ref=2.0,
            theta=15.0,
            V_reset=0.0,
            rate0=0.35,
        )
        target_params = dict(
            tau=1.7,
            mean=-0.15,
            tau_m=10.0,
            tau_syn=0.4,
            t_ref=2.0,
            theta=15.0,
            V_reset=0.0,
            rate0=0.05,
        )

        with brainstate.environ.context(dt=self.dt):
            src = siegert_neuron(
                1,
                tau=source_params['tau'] * u.ms,
                mean=source_params['mean'],
                tau_m=source_params['tau_m'] * u.ms,
                tau_syn=source_params['tau_syn'] * u.ms,
                t_ref=source_params['t_ref'] * u.ms,
                theta=source_params['theta'],
                V_reset=source_params['V_reset'],
                rate_initializer=braintools.init.Constant(source_params['rate0']),
            )
            tgt = siegert_neuron(
                1,
                tau=target_params['tau'] * u.ms,
                mean=target_params['mean'],
                tau_m=target_params['tau_m'] * u.ms,
                tau_syn=target_params['tau_syn'] * u.ms,
                t_ref=target_params['t_ref'] * u.ms,
                theta=target_params['theta'],
                V_reset=target_params['V_reset'],
                rate_initializer=braintools.init.Constant(target_params['rate0']),
            )
            src.init_state()
            tgt.init_state()

            syn = diffusion_connection(
                drift_factor=drift_factor,
                diffusion_factor=diffusion_factor,
            )

            src_rate_ref = source_params['rate0']
            tgt_rate_ref = target_params['rate0']
            queue_drift = {}
            queue_diffusion = {}

            p1_src = np.exp(-self.dt_ms / source_params['tau'])
            p2_src = -np.expm1(-self.dt_ms / source_params['tau'])
            p1_tgt = np.exp(-self.dt_ms / target_params['tau'])
            p2_tgt = -np.expm1(-self.dt_ms / target_params['tau'])

            for k in range(80):
                src_drive = float(src.siegert_rate(np.asarray([0.0]), np.asarray([0.0]))[0])
                src_rate_ref = p1_src * src_rate_ref + p2_src * (source_params['mean'] + src_drive)

                self._step(src, k)
                src_rate_model = float(np.asarray(src.rate.value).reshape(-1)[0])
                self.assertAlmostEqual(src_rate_model, src_rate_ref, delta=1e-12)

                queue_drift[k + 1] = queue_drift.get(k + 1, 0.0) + drift_factor * src_rate_ref
                queue_diffusion[k + 1] = queue_diffusion.get(k + 1, 0.0) + diffusion_factor * src_rate_ref

                drift_now = queue_drift.pop(k, 0.0)
                diffusion_now = queue_diffusion.pop(k, 0.0)
                tgt_drive = float(tgt.siegert_rate(np.asarray([drift_now]), np.asarray([diffusion_now]))[0])
                tgt_rate_ref = p1_tgt * tgt_rate_ref + p2_tgt * (target_params['mean'] + tgt_drive)

                event = syn.to_siegert_event(coeff=src_rate_model, delay_steps=1)
                self._step(tgt, k, delayed_diffusion_events=[event])
                tgt_rate_model = float(np.asarray(tgt.rate.value).reshape(-1)[0])
                self.assertAlmostEqual(tgt_rate_model, tgt_rate_ref, delta=1e-12)

    def test_two_population_trace_matches_nest(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        drift_factor = 5.2
        diffusion_factor = 0.9
        src_params = {
            'tau': 2.5,
            'tau_m': 10.0,
            'tau_syn': 0.2,
            't_ref': 2.0,
            'theta': 15.0,
            'V_reset': 0.0,
            'mean': 0.8,
            'rate': 0.45,
        }
        tgt_params = {
            'tau': 1.4,
            'tau_m': 10.0,
            'tau_syn': 0.5,
            't_ref': 2.0,
            'theta': 15.0,
            'V_reset': 0.0,
            'mean': -0.1,
            'rate': 0.2,
        }

        nominal_steps = 500
        simtime_ms = nominal_steps * self.dt_ms
        nest_src, nest_tgt = _run_nest_two_siegert_trace(
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            src_params=src_params,
            tgt_params=tgt_params,
            drift_factor=drift_factor,
            diffusion_factor=diffusion_factor,
        )

        replay_steps = min(nest_src.size, nest_tgt.size)
        bp_src = np.zeros((replay_steps,), dtype=np.float64)
        bp_tgt = np.zeros((replay_steps,), dtype=np.float64)

        with brainstate.environ.context(dt=self.dt):
            src = siegert_neuron(
                1,
                tau=src_params['tau'] * u.ms,
                tau_m=src_params['tau_m'] * u.ms,
                tau_syn=src_params['tau_syn'] * u.ms,
                t_ref=src_params['t_ref'] * u.ms,
                theta=src_params['theta'],
                V_reset=src_params['V_reset'],
                mean=src_params['mean'],
                rate_initializer=braintools.init.Constant(src_params['rate']),
            )
            tgt = siegert_neuron(
                1,
                tau=tgt_params['tau'] * u.ms,
                tau_m=tgt_params['tau_m'] * u.ms,
                tau_syn=tgt_params['tau_syn'] * u.ms,
                t_ref=tgt_params['t_ref'] * u.ms,
                theta=tgt_params['theta'],
                V_reset=tgt_params['V_reset'],
                mean=tgt_params['mean'],
                rate_initializer=braintools.init.Constant(tgt_params['rate']),
            )
            src.init_state()
            tgt.init_state()

            syn = diffusion_connection(
                drift_factor=drift_factor,
                diffusion_factor=diffusion_factor,
            )

            for k in range(replay_steps):
                self._step(src, k)
                src_rate = float(np.asarray(src.rate.value).reshape(-1)[0])
                event = syn.to_siegert_event(coeff=src_rate, delay_steps=1)
                self._step(tgt, k, delayed_diffusion_events=[event])

                bp_src[k] = src_rate
                bp_tgt[k] = float(np.asarray(tgt.rate.value).reshape(-1)[0])

        npt.assert_allclose(bp_src, nest_src[:replay_steps], atol=2e-6, rtol=1e-8)
        npt.assert_allclose(bp_tgt, nest_tgt[:replay_steps], atol=2e-6, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
