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
import numpy as np
import numpy.testing as npt

from brainpy.state import siegert_neuron

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _run_nest_siegert_trace(dt_ms, simtime_ms, model_params, drift_factor, diffusion_factor):
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = False
    nest.local_num_threads = 1

    target = nest.Create('siegert_neuron', params=model_params)
    drive = nest.Create('siegert_neuron', params={'mean': 1.0, 'rate': 1.0})

    nest.Connect(
        drive,
        target,
        syn_spec={
            'synapse_model': 'diffusion_connection',
            'drift_factor': drift_factor,
            'diffusion_factor': diffusion_factor,
        },
    )

    mm = nest.Create('multimeter', params={
        'record_from': ['rate'],
        'interval': dt_ms,
    })
    nest.Connect(mm, target, syn_spec={'delay': dt_ms})

    nest.Simulate(simtime_ms)
    return np.asarray(mm.events['rate'], dtype=np.float64)


class TestSiegertNeuron(unittest.TestCase):
    def setUp(self):
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

    def test_step_equations_match_nest_update_ordering(self):
        params = dict(
            tau=2.0,
            mean=0.7,
            tau_m=10.0,
            tau_syn=0.6,
            t_ref=2.0,
            theta=15.0,
            V_reset=0.0,
            rate0=0.25,
        )

        drift_direct = np.asarray([14.0, 14.5, 15.0, 15.5, 14.3, 13.7], dtype=np.float64)
        diffusion_direct = np.asarray([1.8, 1.5, 2.2, 1.9, 1.3, 1.0], dtype=np.float64)

        instant_events_seq = [
            [{'coeff': 1.0, 'drift_factor': 0.3, 'diffusion_factor': 0.2}],
            [{'coeff': 0.8, 'drift_factor': -0.4, 'diffusion_factor': 0.6}],
            [],
            [{'coeff': 0.5, 'drift_factor': 0.2, 'diffusion_factor': 0.3}],
            [],
            [],
        ]
        delayed_events_seq = [
            [{'coeff': 1.2, 'drift_factor': 0.5, 'diffusion_factor': 0.7, 'delay_steps': 2}],
            [{'coeff': -0.5, 'drift_factor': 0.4, 'diffusion_factor': 0.5, 'delay_steps': 1}],
            [],
            [{'coeff': 0.9, 'drift_factor': 0.1, 'diffusion_factor': 0.2, 'delay_steps': 0}],
            [],
            [],
        ]

        with brainstate.environ.context(dt=self.dt):
            nrn = siegert_neuron(
                1,
                tau=params['tau'] * u.ms,
                mean=params['mean'],
                tau_m=params['tau_m'] * u.ms,
                tau_syn=params['tau_syn'] * u.ms,
                t_ref=params['t_ref'] * u.ms,
                theta=params['theta'],
                V_reset=params['V_reset'],
                rate_initializer=braintools.init.Constant(params['rate0']),
            )
            nrn.init_state()

            queue_drift = {}
            queue_diff = {}
            rate_ref = params['rate0']

            p1 = math.exp(-self.dt_ms / params['tau'])
            p2 = -math.expm1(-self.dt_ms / params['tau'])

            for k in range(drift_direct.size):
                drift_total = queue_drift.pop(k, 0.0)
                diffusion_total = queue_diff.pop(k, 0.0)

                for ev in delayed_events_seq[k]:
                    coeff = float(ev.get('coeff', 0.0))
                    weight = float(ev.get('weight', 1.0))
                    multiplicity = float(ev.get('multiplicity', 1.0))
                    delay = int(ev.get('delay_steps', 1))

                    weighted_coeff = coeff * weight * multiplicity
                    d = float(ev.get('drift_factor', 1.0)) * weighted_coeff
                    s = float(ev.get('diffusion_factor', 0.0)) * weighted_coeff

                    target = k + delay
                    if target == k:
                        drift_total += d
                        diffusion_total += s
                    else:
                        queue_drift[target] = queue_drift.get(target, 0.0) + d
                        queue_diff[target] = queue_diff.get(target, 0.0) + s

                for ev in instant_events_seq[k]:
                    coeff = float(ev.get('coeff', 0.0))
                    weight = float(ev.get('weight', 1.0))
                    multiplicity = float(ev.get('multiplicity', 1.0))
                    weighted_coeff = coeff * weight * multiplicity
                    drift_total += float(ev.get('drift_factor', 1.0)) * weighted_coeff
                    diffusion_total += float(ev.get('diffusion_factor', 0.0)) * weighted_coeff

                drift_total += drift_direct[k]
                diffusion_total += diffusion_direct[k]

                drive = float(nrn.siegert_rate(np.asarray([drift_total]), np.asarray([diffusion_total]))[0])
                rate_new = p1 * rate_ref + p2 * (params['mean'] + drive)

                out = self._step(
                    nrn,
                    k,
                    drift_input=drift_direct[k],
                    diffusion_input=diffusion_direct[k],
                    instant_diffusion_events=instant_events_seq[k],
                    delayed_diffusion_events=delayed_events_seq[k],
                )

                self.assertAlmostEqual(float(np.asarray(out).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(nrn.rate.value).reshape(-1)[0]), rate_new, delta=1e-12)
                # NEST non-WFR output semantics: outgoing coeff is the updated rate.
                self.assertAlmostEqual(float(np.asarray(nrn.delayed_rate.value).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(nrn.instant_rate.value).reshape(-1)[0]), rate_new, delta=1e-12)

                rate_ref = rate_new

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

    def test_matches_nest_trace_with_constant_diffusion_drive(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        params = {
            'tau': 1.0,
            'tau_m': 10.0,
            'tau_syn': 0.3,
            't_ref': 2.0,
            'theta': 15.0,
            'V_reset': 0.0,
            'mean': 0.4,
            'rate': -0.2,
        }
        drift_factor = 14.0
        diffusion_factor = 1.5

        nominal_steps = 400
        simtime_ms = nominal_steps * self.dt_ms

        nest_rate = _run_nest_siegert_trace(
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            model_params=params,
            drift_factor=drift_factor,
            diffusion_factor=diffusion_factor,
        )

        replay_steps = nest_rate.size

        with brainstate.environ.context(dt=self.dt):
            nrn = siegert_neuron(
                1,
                tau=params['tau'] * u.ms,
                tau_m=params['tau_m'] * u.ms,
                tau_syn=params['tau_syn'] * u.ms,
                t_ref=params['t_ref'] * u.ms,
                theta=params['theta'],
                V_reset=params['V_reset'],
                mean=params['mean'],
                rate_initializer=braintools.init.Constant(params['rate']),
            )
            nrn.init_state()

            bp_rate = np.zeros((replay_steps,), dtype=np.float64)
            for k in range(replay_steps):
                self._step(
                    nrn,
                    k,
                    delayed_diffusion_events=[{
                        'coeff': 1.0,
                        'drift_factor': drift_factor,
                        'diffusion_factor': diffusion_factor,
                        'delay_steps': 1,
                    }],
                )
                bp_rate[k] = float(np.asarray(nrn.rate.value).reshape(-1)[0])

        n_cmp = min(bp_rate.size, nest_rate.size)
        npt.assert_allclose(bp_rate[:n_cmp], nest_rate[:n_cmp], atol=2e-6, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
