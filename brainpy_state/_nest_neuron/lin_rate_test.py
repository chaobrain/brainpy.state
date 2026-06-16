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
from brainpy.state import lin_rate_ipn, lin_rate_opn

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _run_nest_trace(model_name, params, record_from, simtime_ms, dt_ms):
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = False

    neuron = nest.Create(model_name, params=params)
    mm = nest.Create('multimeter', params={
        'record_from': list(record_from),
        'interval': dt_ms,
    })
    nest.Connect(mm, neuron, syn_spec={'delay': dt_ms})
    nest.Simulate(simtime_ms)

    ev = mm.events
    dftype = brainstate.environ.dftype()
    out = {'times': np.asarray(ev['times'], dtype=dftype)}
    for key in record_from:
        out[key] = np.asarray(ev[key], dtype=dftype)
    return out


class TestLinRate(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(**kwargs)

    def test_nest_default_parameters(self):
        ipn = lin_rate_ipn(1)
        self.assertEqual(ipn.tau, 10.0 * u.ms)
        self.assertEqual(ipn.lambda_, 1.0)
        self.assertEqual(ipn.sigma, 1.0)
        self.assertEqual(ipn.mu, 0.0)
        self.assertEqual(ipn.g, 1.0)
        self.assertEqual(ipn.mult_coupling, False)
        self.assertEqual(ipn.g_ex, 1.0)
        self.assertEqual(ipn.g_in, 1.0)
        self.assertEqual(ipn.theta_ex, 0.0)
        self.assertEqual(ipn.theta_in, 0.0)
        self.assertEqual(ipn.linear_summation, True)
        self.assertEqual(ipn.rectify_rate, 0.0)
        self.assertEqual(ipn.rectify_output, False)
        self.assertEqual(ipn.recordables, ['rate', 'noise'])

        opn = lin_rate_opn(1)
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

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            lin_rate_ipn(1, tau=0.0 * u.ms)
        with self.assertRaises(ValueError):
            lin_rate_ipn(1, lambda_=-1e-3)
        with self.assertRaises(ValueError):
            lin_rate_ipn(1, sigma=-1e-3)
        with self.assertRaises(ValueError):
            lin_rate_ipn(1, rectify_rate=-1e-3)

        with self.assertRaises(ValueError):
            lin_rate_opn(1, tau=0.0 * u.ms)
        with self.assertRaises(ValueError):
            lin_rate_opn(1, sigma=-1e-3)


    def test_matches_nest_deterministic_trace_ipn_and_opn(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        steps = 200
        simtime_ms = steps * self.dt_ms

        ipn_nest = _run_nest_trace(
            model_name='lin_rate_ipn',
            params={
                'tau': 5.0,
                'lambda': 1.2,
                'sigma': 0.0,
                'mu': 1.5,
                'rate': -0.3,
            },
            record_from=['rate', 'noise'],
            simtime_ms=simtime_ms,
            dt_ms=self.dt_ms,
        )

        with brainstate.environ.context(dt=self.dt):
            ipn_bp = lin_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.2,
                sigma=0.0,
                mu=1.5,
                rate_initializer=braintools.init.Constant(-0.3),
            )
            ipn_bp.init_state()

            def run_ipn(_):
                ipn_bp.update(noise=0.0)
                return (
                    ipn_bp.rate.value.reshape(-1)[0],
                    ipn_bp.noise.value.reshape(-1)[0],
                )

            ipn_results = brainstate.transform.for_loop(run_ipn, np.zeros(steps))
            bp_rate = np.asarray(ipn_results[0])
            bp_noise = np.asarray(ipn_results[1])

        n_cmp = min(bp_rate.size, ipn_nest['rate'].size)
        npt.assert_allclose(bp_rate[:n_cmp], ipn_nest['rate'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noise[:n_cmp], ipn_nest['noise'][:n_cmp], atol=1e-12)

        opn_nest = _run_nest_trace(
            model_name='lin_rate_opn',
            params={
                'tau': 7.0,
                'sigma': 0.0,
                'mu': -0.8,
                'rate': 0.4,
            },
            record_from=['rate', 'noise', 'noisy_rate'],
            simtime_ms=simtime_ms,
            dt_ms=self.dt_ms,
        )

        with brainstate.environ.context(dt=self.dt):
            opn_bp = lin_rate_opn(
                1,
                tau=7.0 * u.ms,
                sigma=0.0,
                mu=-0.8,
                rate_initializer=braintools.init.Constant(0.4),
                noisy_rate_initializer=braintools.init.Constant(0.4),
            )
            opn_bp.init_state()

            def run_opn(_):
                opn_bp.update(noise=0.0)
                return (
                    opn_bp.rate.value.reshape(-1)[0],
                    opn_bp.noise.value.reshape(-1)[0],
                    opn_bp.noisy_rate.value.reshape(-1)[0],
                )

            opn_results = brainstate.transform.for_loop(run_opn, np.zeros(steps))
            bp_rate = np.asarray(opn_results[0])
            bp_noise = np.asarray(opn_results[1])
            bp_noisy = np.asarray(opn_results[2])

        n_cmp = min(bp_rate.size, opn_nest['rate'].size)
        npt.assert_allclose(bp_rate[:n_cmp], opn_nest['rate'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noise[:n_cmp], opn_nest['noise'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noisy[:n_cmp], opn_nest['noisy_rate'][:n_cmp], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
