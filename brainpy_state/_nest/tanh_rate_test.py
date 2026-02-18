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
from brainpy.state import tanh_rate_ipn, tanh_rate_opn

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

    neuron = nest.Create(model_name)
    neuron.set(params)
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


def _run_nest_tanh_driven_trace(linear_summation, dt_ms, simtime_ms, drive, weight, g, theta):
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = True

    source = nest.Create('lin_rate_ipn')
    source.set({'rate': drive, 'mu': drive, 'sigma': 0.0})
    target = nest.Create('tanh_rate_ipn')
    target.set({
        'tau': 5.0,
        'lambda': 1.0,
        'mu': 0.0,
        'sigma': 0.0,
        'g': g,
        'theta': theta,
        'linear_summation': linear_summation,
        'rate': 0.0,
    })

    mm = nest.Create('multimeter', params={
        'record_from': ['rate'],
        'interval': dt_ms,
    })
    nest.Connect(mm, target, syn_spec={'delay': dt_ms})
    nest.Connect(
        source,
        target,
        syn_spec={'synapse_model': 'rate_connection_instantaneous', 'weight': weight},
    )

    nest.Simulate(simtime_ms)
    dftype = brainstate.environ.dftype()
    return np.asarray(mm.events['rate'], dtype=dftype)


class TestTanhRate(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(**kwargs)

    def test_nest_default_parameters(self):
        ipn = tanh_rate_ipn(1)
        self.assertEqual(ipn.tau, 10.0 * u.ms)
        self.assertEqual(ipn.lambda_, 1.0)
        self.assertEqual(ipn.sigma, 1.0)
        self.assertEqual(ipn.mu, 0.0)
        self.assertEqual(ipn.g, 1.0)
        self.assertEqual(ipn.theta, 0.0)
        self.assertEqual(ipn.mult_coupling, False)
        self.assertEqual(ipn.linear_summation, True)
        self.assertEqual(ipn.rectify_rate, 0.0)
        self.assertEqual(ipn.rectify_output, False)
        self.assertEqual(ipn.recordables, ['rate', 'noise'])

        opn = tanh_rate_opn(1)
        self.assertEqual(opn.tau, 10.0 * u.ms)
        self.assertEqual(opn.sigma, 1.0)
        self.assertEqual(opn.mu, 0.0)
        self.assertEqual(opn.g, 1.0)
        self.assertEqual(opn.theta, 0.0)
        self.assertEqual(opn.mult_coupling, False)
        self.assertEqual(opn.linear_summation, True)
        self.assertEqual(opn.recordables, ['rate', 'noise', 'noisy_rate'])

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            tanh_rate_ipn(1, tau=0.0 * u.ms)
        with self.assertRaises(ValueError):
            tanh_rate_ipn(1, lambda_=-1e-3)
        with self.assertRaises(ValueError):
            tanh_rate_ipn(1, sigma=-1e-3)
        with self.assertRaises(ValueError):
            tanh_rate_ipn(1, rectify_rate=-1e-3)

        with self.assertRaises(ValueError):
            tanh_rate_opn(1, tau=0.0 * u.ms)
        with self.assertRaises(ValueError):
            tanh_rate_opn(1, sigma=-1e-3)

    def test_ipn_step_equations_match_nest_update_ordering(self):
        params = dict(
            tau=5.0,
            lambda_=1.3,
            sigma=0.6,
            mu=0.8,
            g=1.7,
            theta=0.2,
            mult_coupling=True,
            linear_summation=False,
            rectify_output=True,
            rectify_rate=0.05,
            rate0=0.3,
        )

        dftype = brainstate.environ.dftype()
        noise_seq = np.asarray([0.2, -1.0, 0.4, -0.3, 1.1, 0.0], dtype=dftype)
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
            neuron = tanh_rate_ipn(
                1,
                tau=params['tau'] * u.ms,
                lambda_=params['lambda_'],
                sigma=params['sigma'],
                mu=params['mu'],
                g=params['g'],
                theta=params['theta'],
                mult_coupling=params['mult_coupling'],
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
                    val = float(ev['weight']) * math.tanh(params['g'] * (float(ev['rate']) - params['theta']))
                    val *= float(ev.get('multiplicity', 1.0))
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
                    val = float(ev['weight']) * math.tanh(params['g'] * (float(ev['rate']) - params['theta']))
                    val *= float(ev.get('multiplicity', 1.0))
                    if ev['weight'] >= 0.0:
                        instant_ex += val
                    else:
                        instant_in += val

                noise_ref = params['sigma'] * noise_seq[k]
                rate_new = P1 * rate_ref + P2 * params['mu'] + noise_fac * noise_ref
                rate_new += P2 * (delayed_ex + instant_ex)
                rate_new += P2 * (delayed_in + instant_in)
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
                self.assertAlmostEqual(float(np.asarray(neuron.delayed_rate.value).reshape(-1)[0]), rate_ref,
                                       delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.instant_rate.value).reshape(-1)[0]), rate_new,
                                       delta=1e-12)

                rate_ref = rate_new

    def test_opn_step_equations_match_nest_update_ordering(self):
        params = dict(
            tau=7.0,
            sigma=0.4,
            mu=-0.3,
            g=1.3,
            theta=-0.1,
            mult_coupling=True,
            linear_summation=False,
            rate0=-0.2,
        )

        dftype = brainstate.environ.dftype()
        noise_seq = np.asarray([1.0, -0.5, 0.2, 0.0, -1.3, 0.7], dtype=dftype)
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
            neuron = tanh_rate_opn(
                1,
                tau=params['tau'] * u.ms,
                sigma=params['sigma'],
                mu=params['mu'],
                g=params['g'],
                theta=params['theta'],
                mult_coupling=params['mult_coupling'],
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
                    val = float(ev['weight']) * math.tanh(params['g'] * (float(ev['rate']) - params['theta']))
                    val *= float(ev.get('multiplicity', 1.0))
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
                    val = float(ev['weight']) * math.tanh(params['g'] * (float(ev['rate']) - params['theta']))
                    val *= float(ev.get('multiplicity', 1.0))
                    if ev['weight'] >= 0.0:
                        instant_ex += val
                    else:
                        instant_in += val

                noise_ref = params['sigma'] * noise_seq[k]
                noisy_ref = rate_ref + noise_fac * noise_ref
                rate_new = P1 * rate_ref + P2 * params['mu']
                rate_new += P2 * (delayed_ex + instant_ex)
                rate_new += P2 * (delayed_in + instant_in)

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
                self.assertAlmostEqual(float(np.asarray(neuron.noisy_rate.value).reshape(-1)[0]), noisy_ref,
                                       delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.delayed_rate.value).reshape(-1)[0]), noisy_ref,
                                       delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(neuron.instant_rate.value).reshape(-1)[0]), noisy_ref,
                                       delta=1e-12)

                rate_ref = rate_new

    def test_mult_coupling_flag_is_noop(self):
        steps = 32
        event = [{'rate': 0.8, 'weight': 0.5}]

        with brainstate.environ.context(dt=self.dt):
            n0 = tanh_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.1,
                sigma=0.0,
                mu=0.2,
                g=1.4,
                theta=0.1,
                mult_coupling=False,
                linear_summation=False,
            )
            n1 = tanh_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.1,
                sigma=0.0,
                mu=0.2,
                g=1.4,
                theta=0.1,
                mult_coupling=True,
                linear_summation=False,
            )
            n0.init_state()
            n1.init_state()

            dftype = brainstate.environ.dftype()
            y0 = np.zeros(steps, dtype=dftype)
            y1 = np.zeros(steps, dtype=dftype)
            for k in range(steps):
                self._step(n0, k, instant_rate_events=event)
                self._step(n1, k, instant_rate_events=event)
                y0[k] = float(np.asarray(n0.rate.value).reshape(-1)[0])
                y1[k] = float(np.asarray(n1.rate.value).reshape(-1)[0])

        npt.assert_allclose(y0, y1, atol=1e-12)

    def test_matches_nest_deterministic_trace_ipn_and_opn(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        steps = 200
        simtime_ms = steps * self.dt_ms

        ipn_nest = _run_nest_trace(
            model_name='tanh_rate_ipn',
            params={
                'tau': 5.0,
                'lambda': 1.2,
                'sigma': 0.0,
                'mu': 1.5,
                'g': 1.7,
                'theta': 0.2,
                'rate': -0.3,
            },
            record_from=['rate', 'noise'],
            simtime_ms=simtime_ms,
            dt_ms=self.dt_ms,
        )

        with brainstate.environ.context(dt=self.dt):
            ipn_bp = tanh_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.2,
                sigma=0.0,
                mu=1.5,
                g=1.7,
                theta=0.2,
                rate_initializer=braintools.init.Constant(-0.3),
            )
            ipn_bp.init_state()
            dftype = brainstate.environ.dftype()
            bp_rate = np.zeros((steps,), dtype=dftype)
            bp_noise = np.zeros((steps,), dtype=dftype)
            for k in range(steps):
                self._step(ipn_bp, k)
                bp_rate[k] = float(np.asarray(ipn_bp.rate.value).reshape(-1)[0])
                bp_noise[k] = float(np.asarray(ipn_bp.noise.value).reshape(-1)[0])

        n_cmp = min(bp_rate.size, ipn_nest['rate'].size)
        npt.assert_allclose(bp_rate[:n_cmp], ipn_nest['rate'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noise[:n_cmp], ipn_nest['noise'][:n_cmp], atol=1e-12)

        opn_nest = _run_nest_trace(
            model_name='tanh_rate_opn',
            params={
                'tau': 7.0,
                'sigma': 0.0,
                'mu': -0.8,
                'g': 0.9,
                'theta': -0.1,
                'rate': 0.4,
            },
            record_from=['rate', 'noise', 'noisy_rate'],
            simtime_ms=simtime_ms,
            dt_ms=self.dt_ms,
        )

        with brainstate.environ.context(dt=self.dt):
            opn_bp = tanh_rate_opn(
                1,
                tau=7.0 * u.ms,
                sigma=0.0,
                mu=-0.8,
                g=0.9,
                theta=-0.1,
                rate_initializer=braintools.init.Constant(0.4),
                noisy_rate_initializer=braintools.init.Constant(0.4),
            )
            opn_bp.init_state()
            bp_rate = np.zeros((steps,), dtype=dftype)
            bp_noise = np.zeros((steps,), dtype=dftype)
            bp_noisy = np.zeros((steps,), dtype=dftype)
            for k in range(steps):
                self._step(opn_bp, k)
                bp_rate[k] = float(np.asarray(opn_bp.rate.value).reshape(-1)[0])
                bp_noise[k] = float(np.asarray(opn_bp.noise.value).reshape(-1)[0])
                bp_noisy[k] = float(np.asarray(opn_bp.noisy_rate.value).reshape(-1)[0])

        n_cmp = min(bp_rate.size, opn_nest['rate'].size)
        npt.assert_allclose(bp_rate[:n_cmp], opn_nest['rate'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noise[:n_cmp], opn_nest['noise'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noisy[:n_cmp], opn_nest['noisy_rate'][:n_cmp], atol=1e-12)

    def test_linear_summation_modes_match_nest(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        drive = 1.5
        weight = 0.5
        g = 1.8
        theta = 0.2
        steps = 300
        simtime_ms = steps * self.dt_ms

        nest_linear_sum = _run_nest_tanh_driven_trace(
            linear_summation=True,
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            g=g,
            theta=theta,
        )
        nest_event_sum = _run_nest_tanh_driven_trace(
            linear_summation=False,
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            g=g,
            theta=theta,
        )

        with brainstate.environ.context(dt=self.dt):
            bp_linear_sum = tanh_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.0,
                mu=0.0,
                sigma=0.0,
                g=g,
                theta=theta,
                linear_summation=True,
            )
            bp_event_sum = tanh_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.0,
                mu=0.0,
                sigma=0.0,
                g=g,
                theta=theta,
                linear_summation=False,
            )
            bp_linear_sum.init_state()
            bp_event_sum.init_state()

            dftype = brainstate.environ.dftype()
            y_linear_sum = np.zeros((steps,), dtype=dftype)
            y_event_sum = np.zeros((steps,), dtype=dftype)
            for k in range(steps):
                self._step(
                    bp_linear_sum,
                    k,
                    instant_rate_events=[{'rate': drive, 'weight': weight}],
                )
                self._step(
                    bp_event_sum,
                    k,
                    instant_rate_events=[{'rate': drive, 'weight': weight}],
                )
                y_linear_sum[k] = float(np.asarray(bp_linear_sum.rate.value).reshape(-1)[0])
                y_event_sum[k] = float(np.asarray(bp_event_sum.rate.value).reshape(-1)[0])

        n_cmp = min(y_linear_sum.size, nest_linear_sum.size)
        npt.assert_allclose(y_linear_sum[:n_cmp], nest_linear_sum[:n_cmp], atol=1e-10)
        n_cmp = min(y_event_sum.size, nest_event_sum.size)
        npt.assert_allclose(y_event_sum[:n_cmp], nest_event_sum[:n_cmp], atol=1e-10)

        # Ensure both modes are distinct for this setup.
        self.assertFalse(np.allclose(y_linear_sum, y_event_sum))


if __name__ == '__main__':
    unittest.main()
