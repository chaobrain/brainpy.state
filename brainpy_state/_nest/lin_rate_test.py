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

from brainpy.state import lin_rate_ipn, lin_rate_opn

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


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
    out = {'times': np.asarray(ev['times'], dtype=np.float64)}
    for key in record_from:
        out[key] = np.asarray(ev[key], dtype=np.float64)
    return out


def _run_nest_ipn_driven_trace(mode, dt_ms, simtime_ms, drive, weight, delay_ms):
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = True

    source = nest.Create('lin_rate_ipn', params={
        'rate': drive,
        'mu': drive,
        'sigma': 0.0,
    })
    target = nest.Create('lin_rate_ipn', params={
        'tau': 5.0,
        'lambda': 1.0,
        'mu': 0.0,
        'sigma': 0.0,
        'rate': 0.0,
    })

    mm = nest.Create('multimeter', params={
        'record_from': ['rate'],
        'interval': dt_ms,
    })
    nest.Connect(mm, target, syn_spec={'delay': dt_ms})

    if mode == 'instantaneous':
        syn_spec = {'synapse_model': 'rate_connection_instantaneous', 'weight': weight}
    elif mode == 'delayed':
        syn_spec = {'synapse_model': 'rate_connection_delayed', 'weight': weight, 'delay': delay_ms}
    else:
        raise ValueError(f'Unknown mode: {mode}')
    nest.Connect(source, target, syn_spec=syn_spec)

    nest.Simulate(simtime_ms)
    return np.asarray(mm.events['rate'], dtype=np.float64)


class TestLinRate(unittest.TestCase):
    def setUp(self):
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

    def test_ipn_step_equations_match_nest_update_ordering(self):
        params = dict(
            tau=5.0,
            lambda_=1.3,
            sigma=0.6,
            mu=0.8,
            g=1.2,
            mult_coupling=True,
            g_ex=1.1,
            g_in=0.9,
            theta_ex=0.5,
            theta_in=-0.2,
            linear_summation=True,
            rectify_output=True,
            rectify_rate=0.05,
            rate0=0.3,
        )

        noise_seq = np.asarray([0.2, -1.0, 0.4, -0.3, 1.1, 0.0], dtype=np.float64)
        instant_events_seq = [
            [{'rate': 1.0, 'weight': 0.7}, {'rate': 0.5, 'weight': -0.4}],
            [{'rate': 0.2, 'weight': 0.1}],
            [{'rate': 1.5, 'weight': -0.2}],
            [],
            [{'rate': 0.9, 'weight': 0.3}, {'rate': -1.1, 'weight': -0.2}],
            [],
        ]
        delayed_events_seq = [
            [{'rate': 1.2, 'weight': 0.5, 'delay_steps': 2}],
            [{'rate': 0.8, 'weight': -0.3, 'delay_steps': 1}],
            [],
            [{'rate': 1.0, 'weight': 0.2, 'delay_steps': 0}],
            [],
            [],
        ]

        with brainstate.environ.context(dt=self.dt):
            neuron = lin_rate_ipn(
                1,
                tau=params['tau'] * u.ms,
                lambda_=params['lambda_'],
                sigma=params['sigma'],
                mu=params['mu'],
                g=params['g'],
                mult_coupling=params['mult_coupling'],
                g_ex=params['g_ex'],
                g_in=params['g_in'],
                theta_ex=params['theta_ex'],
                theta_in=params['theta_in'],
                linear_summation=params['linear_summation'],
                rectify_output=params['rectify_output'],
                rectify_rate=params['rectify_rate'],
                rate_initializer=braintools.init.Constant(params['rate0']),
            )
            neuron.init_state()

            queue_ex = {}
            queue_in = {}
            rate_ref = params['rate0']

            h = self.dt_ms
            P1 = math.exp(-params['lambda_'] * h / params['tau'])
            P2 = -math.expm1(-params['lambda_'] * h / params['tau']) / params['lambda_']
            noise_fac = math.sqrt(-0.5 * math.expm1(-2.0 * params['lambda_'] * h / params['tau']) / params['lambda_'])

            for k in range(noise_seq.size):
                delayed_ex = queue_ex.pop(k, 0.0)
                delayed_in = queue_in.pop(k, 0.0)

                for ev in delayed_events_seq[k]:
                    val = float(ev['rate']) * float(ev['weight']) * float(ev.get('multiplicity', 1.0))
                    target = k + int(ev.get('delay_steps', 1))
                    if target == k:
                        if ev['weight'] >= 0.0:
                            delayed_ex += val
                        else:
                            delayed_in += val
                    else:
                        if ev['weight'] >= 0.0:
                            queue_ex[target] = queue_ex.get(target, 0.0) + val
                        else:
                            queue_in[target] = queue_in.get(target, 0.0) + val

                instant_ex = 0.0
                instant_in = 0.0
                for ev in instant_events_seq[k]:
                    val = float(ev['rate']) * float(ev['weight']) * float(ev.get('multiplicity', 1.0))
                    if ev['weight'] >= 0.0:
                        instant_ex += val
                    else:
                        instant_in += val

                noise_ref = params['sigma'] * noise_seq[k]
                rate_new = P1 * rate_ref + P2 * params['mu'] + noise_fac * noise_ref

                H_ex = params['g_ex'] * (params['theta_ex'] - rate_ref)
                H_in = params['g_in'] * (params['theta_in'] + rate_ref)
                rate_new += P2 * H_ex * (params['g'] * (delayed_ex + instant_ex))
                rate_new += P2 * H_in * (params['g'] * (delayed_in + instant_in))
                rate_new = max(rate_new, params['rectify_rate'])

                out = self._step(
                    neuron,
                    k,
                    noise=noise_seq[k],
                    instant_rate_events=instant_events_seq[k],
                    delayed_rate_events=delayed_events_seq[k],
                )

                self.assertAlmostEqual(float(np.asarray(out).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.rate.value).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.noise.value).reshape(-1)[0]), noise_ref, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.delayed_rate.value).reshape(-1)[0]), rate_ref, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.instant_rate.value).reshape(-1)[0]), rate_new, delta=1e-12)

                rate_ref = rate_new

    def test_opn_step_equations_match_nest_update_ordering(self):
        params = dict(
            tau=7.0,
            sigma=0.4,
            mu=-0.3,
            g=1.1,
            mult_coupling=True,
            g_ex=0.8,
            g_in=1.2,
            theta_ex=0.2,
            theta_in=0.7,
            linear_summation=False,
            rate0=-0.2,
        )

        noise_seq = np.asarray([1.0, -0.5, 0.2, 0.0, -1.3, 0.7], dtype=np.float64)
        instant_events_seq = [
            [{'rate': 1.0, 'weight': 0.2}],
            [{'rate': 0.8, 'weight': -0.4}],
            [],
            [{'rate': -0.5, 'weight': -0.3}],
            [],
            [],
        ]
        delayed_events_seq = [
            [{'rate': 1.1, 'weight': 0.3, 'delay_steps': 2}],
            [],
            [{'rate': 0.4, 'weight': -0.6, 'delay_steps': 1}],
            [],
            [],
            [],
        ]

        with brainstate.environ.context(dt=self.dt):
            neuron = lin_rate_opn(
                1,
                tau=params['tau'] * u.ms,
                sigma=params['sigma'],
                mu=params['mu'],
                g=params['g'],
                mult_coupling=params['mult_coupling'],
                g_ex=params['g_ex'],
                g_in=params['g_in'],
                theta_ex=params['theta_ex'],
                theta_in=params['theta_in'],
                linear_summation=params['linear_summation'],
                rate_initializer=braintools.init.Constant(params['rate0']),
                noisy_rate_initializer=braintools.init.Constant(params['rate0']),
            )
            neuron.init_state()

            queue_ex = {}
            queue_in = {}
            rate_ref = params['rate0']

            h = self.dt_ms
            P1 = math.exp(-h / params['tau'])
            P2 = -math.expm1(-h / params['tau'])
            noise_fac = math.sqrt(params['tau'] / h)

            for k in range(noise_seq.size):
                delayed_ex = queue_ex.pop(k, 0.0)
                delayed_in = queue_in.pop(k, 0.0)

                for ev in delayed_events_seq[k]:
                    val = float(ev['rate']) * float(ev['weight']) * float(ev.get('multiplicity', 1.0))
                    target = k + int(ev.get('delay_steps', 1))
                    if target == k:
                        if ev['weight'] >= 0.0:
                            delayed_ex += val
                        else:
                            delayed_in += val
                    else:
                        if ev['weight'] >= 0.0:
                            queue_ex[target] = queue_ex.get(target, 0.0) + val
                        else:
                            queue_in[target] = queue_in.get(target, 0.0) + val

                instant_ex = 0.0
                instant_in = 0.0
                for ev in instant_events_seq[k]:
                    val = float(ev['rate']) * float(ev['weight']) * float(ev.get('multiplicity', 1.0))
                    if ev['weight'] >= 0.0:
                        instant_ex += val
                    else:
                        instant_in += val

                noise_ref = params['sigma'] * noise_seq[k]
                noisy_ref = rate_ref + noise_fac * noise_ref
                rate_new = P1 * rate_ref + P2 * params['mu']

                H_ex = params['g_ex'] * (params['theta_ex'] - noisy_ref)
                H_in = params['g_in'] * (params['theta_in'] + noisy_ref)
                rate_new += P2 * H_ex * (params['g'] * (delayed_ex + instant_ex))
                rate_new += P2 * H_in * (params['g'] * (delayed_in + instant_in))

                out = self._step(
                    neuron,
                    k,
                    noise=noise_seq[k],
                    instant_rate_events=instant_events_seq[k],
                    delayed_rate_events=delayed_events_seq[k],
                )

                self.assertAlmostEqual(float(np.asarray(out).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.rate.value).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.noise.value).reshape(-1)[0]), noise_ref, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.noisy_rate.value).reshape(-1)[0]), noisy_ref, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.delayed_rate.value).reshape(-1)[0]), noisy_ref, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.instant_rate.value).reshape(-1)[0]), noisy_ref, delta=1e-12)

                rate_ref = rate_new

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
            bp_rate = np.zeros((steps,), dtype=np.float64)
            bp_noise = np.zeros((steps,), dtype=np.float64)
            for k in range(steps):
                self._step(ipn_bp, k)
                bp_rate[k] = float(np.asarray(ipn_bp.rate.value).reshape(-1)[0])
                bp_noise[k] = float(np.asarray(ipn_bp.noise.value).reshape(-1)[0])

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
            bp_rate = np.zeros((steps,), dtype=np.float64)
            bp_noise = np.zeros((steps,), dtype=np.float64)
            bp_noisy = np.zeros((steps,), dtype=np.float64)
            for k in range(steps):
                self._step(opn_bp, k)
                bp_rate[k] = float(np.asarray(opn_bp.rate.value).reshape(-1)[0])
                bp_noise[k] = float(np.asarray(opn_bp.noise.value).reshape(-1)[0])
                bp_noisy[k] = float(np.asarray(opn_bp.noisy_rate.value).reshape(-1)[0])

        n_cmp = min(bp_rate.size, opn_nest['rate'].size)
        npt.assert_allclose(bp_rate[:n_cmp], opn_nest['rate'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noise[:n_cmp], opn_nest['noise'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noisy[:n_cmp], opn_nest['noisy_rate'][:n_cmp], atol=1e-12)

    def test_ipn_instantaneous_and_delayed_driven_trace_matches_nest(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        drive = 1.5
        weight = 0.5
        delay_ms = 2.0
        delay_steps = int(round(delay_ms / self.dt_ms))
        steps = 300
        simtime_ms = steps * self.dt_ms

        nest_instant = _run_nest_ipn_driven_trace(
            mode='instantaneous',
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            delay_ms=delay_ms,
        )
        nest_delayed = _run_nest_ipn_driven_trace(
            mode='delayed',
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            delay_ms=delay_ms,
        )

        with brainstate.environ.context(dt=self.dt):
            bp_instant_model = lin_rate_ipn(1, tau=5.0 * u.ms, lambda_=1.0, mu=0.0, sigma=0.0)
            bp_delayed_model = lin_rate_ipn(1, tau=5.0 * u.ms, lambda_=1.0, mu=0.0, sigma=0.0)
            bp_instant_model.init_state()
            bp_delayed_model.init_state()

            bp_instant = np.zeros((steps,), dtype=np.float64)
            bp_delayed = np.zeros((steps,), dtype=np.float64)
            for k in range(steps):
                self._step(
                    bp_instant_model,
                    k,
                    instant_rate_events=[{'rate': drive, 'weight': weight}],
                )
                self._step(
                    bp_delayed_model,
                    k,
                    delayed_rate_events=[{'rate': drive, 'weight': weight, 'delay_steps': delay_steps}],
                )
                bp_instant[k] = float(np.asarray(bp_instant_model.rate.value).reshape(-1)[0])
                bp_delayed[k] = float(np.asarray(bp_delayed_model.rate.value).reshape(-1)[0])

        n_cmp = min(bp_instant.size, nest_instant.size)
        npt.assert_allclose(bp_instant[:n_cmp], nest_instant[:n_cmp], atol=1e-10)
        n_cmp = min(bp_delayed.size, nest_delayed.size)
        npt.assert_allclose(bp_delayed[:n_cmp], nest_delayed[:n_cmp], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
