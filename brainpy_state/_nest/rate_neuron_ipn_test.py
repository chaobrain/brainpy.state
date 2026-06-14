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
import numpy as np
import numpy.testing as npt
from brainpy.state import rate_neuron_ipn

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _run_nest_trace(params, record_from, simtime_ms, dt_ms):
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = False

    neuron = nest.Create('lin_rate_ipn', params=params)
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


class TestRateNeuronIPN(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(**kwargs)

    def test_nest_default_parameters(self):
        ipn = rate_neuron_ipn(1)
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
        self.assertEqual(ipn.receptor_types, {'RATE': 0})

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            rate_neuron_ipn(1, tau=0.0 * u.ms)
        with self.assertRaises(ValueError):
            rate_neuron_ipn(1, lambda_=-1e-3)
        with self.assertRaises(ValueError):
            rate_neuron_ipn(1, sigma=-1e-3)
        with self.assertRaises(ValueError):
            rate_neuron_ipn(1, rectify_rate=-1e-3)


    def test_matches_nest_lin_rate_trace_with_default_linear_template(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        steps = 200
        simtime_ms = steps * self.dt_ms
        nest_out = _run_nest_trace(
            params={
                'tau': 5.0,
                'lambda': 1.2,
                'sigma': 0.0,
                'mu': 1.5,
                'rate': -0.3,
                'g': 1.3,
                'mult_coupling': True,
                'g_ex': 0.8,
                'g_in': 1.2,
                'theta_ex': 0.6,
                'theta_in': -0.1,
                'linear_summation': True,
                'rectify_output': True,
                'rectify_rate': 0.0,
            },
            record_from=['rate', 'noise'],
            simtime_ms=simtime_ms,
            dt_ms=self.dt_ms,
        )

        with brainstate.environ.context(dt=self.dt):
            bp = rate_neuron_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.2,
                sigma=0.0,
                mu=1.5,
                g=1.3,
                mult_coupling=True,
                g_ex=0.8,
                g_in=1.2,
                theta_ex=0.6,
                theta_in=-0.1,
                linear_summation=True,
                rectify_output=True,
                rectify_rate=0.0,
                rate_initializer=braintools.init.Constant(-0.3),
            )
            bp.init_state()
            dftype = brainstate.environ.dftype()
            bp_rate = np.zeros((steps,), dtype=dftype)
            bp_noise = np.zeros((steps,), dtype=dftype)
            for k in range(steps):
                self._step(bp, k)
                bp_rate[k] = float(np.asarray(bp.rate.value).reshape(-1)[0])
                bp_noise[k] = float(np.asarray(bp.noise.value).reshape(-1)[0])

        n_cmp = min(bp_rate.size, nest_out['rate'].size)
        npt.assert_allclose(bp_rate[:n_cmp], nest_out['rate'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noise[:n_cmp], nest_out['noise'][:n_cmp], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
