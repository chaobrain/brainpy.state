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
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from brainpy.state import gauss_rate_ipn

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _gauss(h, g, mu, sigma):
    return g * np.exp(-((h - mu) ** 2.0) / (2.0 * (sigma ** 2.0)))


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


def _run_nest_gauss_driven_trace(linear_summation, dt_ms, simtime_ms, drive, weight, g, mu, sigma):
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = True

    source = nest.Create('lin_rate_ipn')
    source.set({'rate': drive, 'mu': drive, 'sigma': 0.0})
    target = nest.Create('gauss_rate_ipn')
    target.set({
        'tau': 5.0,
        'lambda': 1.0,
        'mu': mu,
        'sigma': sigma,
        'g': g,
        'linear_summation': linear_summation,
        'rate': 0.0,
    })

    mm = nest.Create('multimeter', params={
        'record_from': ['rate', 'noise'],
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
    return {
        'rate': np.asarray(mm.events['rate'], dtype=dftype),
        'noise': np.asarray(mm.events['noise'], dtype=dftype),
    }


class TestGaussRate(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(**kwargs)

    def test_nest_default_parameters(self):
        ipn = gauss_rate_ipn(1)
        self.assertEqual(ipn.tau, 10.0 * u.ms)
        self.assertEqual(ipn.lambda_, 1.0)
        self.assertEqual(ipn.sigma, 0.0)
        self.assertEqual(ipn.mu, 0.0)
        self.assertEqual(ipn.g, 1.0)
        self.assertEqual(ipn.mult_coupling, False)
        self.assertEqual(ipn.linear_summation, True)
        self.assertEqual(ipn.rectify_rate, 0.0)
        self.assertEqual(ipn.rectify_output, False)
        self.assertEqual(ipn.recordables, ['rate', 'noise'])

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            gauss_rate_ipn(1, tau=0.0 * u.ms)
        with self.assertRaises(ValueError):
            gauss_rate_ipn(1, lambda_=-1e-3)
        with self.assertRaises(ValueError):
            gauss_rate_ipn(1, sigma=-1e-3)
        with self.assertRaises(ValueError):
            gauss_rate_ipn(1, rectify_rate=-1e-3)

    def test_ipn_step_equations_match_nest_update_ordering(self):
        params = dict(
            tau=5.0,
            lambda_=1.3,
            sigma=0.6,
            mu=0.3,
            g=1.7,
            mult_coupling=True,
            linear_summation=False,
            rectify_output=True,
            rectify_rate=0.05,
            rate0=0.4,
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
            neuron = gauss_rate_ipn(
                1,
                tau=params['tau'] * u.ms,
                lambda_=params['lambda_'],
                sigma=params['sigma'],
                mu=params['mu'],
                g=params['g'],
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
                    val = float(ev['weight']) * float(_gauss(
                        float(ev['rate']),
                        params['g'],
                        params['mu'],
                        params['sigma'],
                    ))
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
                    val = float(ev['weight']) * float(_gauss(
                        float(ev['rate']),
                        params['g'],
                        params['mu'],
                        params['sigma'],
                    ))
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

    def test_mult_coupling_flag_is_noop(self):
        steps = 32
        event = [{'rate': 0.8, 'weight': 0.5}]

        with brainstate.environ.context(dt=self.dt):
            n0 = gauss_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.1,
                sigma=0.5,
                mu=0.2,
                g=1.4,
                mult_coupling=False,
                linear_summation=False,
            )
            n1 = gauss_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.1,
                sigma=0.5,
                mu=0.2,
                g=1.4,
                mult_coupling=True,
                linear_summation=False,
            )
            n0.init_state()
            n1.init_state()

            def body(_):
                n0.update(instant_rate_events=event, noise=0.0)
                n1.update(instant_rate_events=event, noise=0.0)
                return (n0.rate.value.reshape(-1)[0], n1.rate.value.reshape(-1)[0])

            results = brainstate.transform.for_loop(body, np.zeros(steps))
            y0 = np.asarray(results[0])
            y1 = np.asarray(results[1])

        npt.assert_allclose(y0, y1, atol=1e-12)

    def test_matches_nest_trace_with_replayed_noise(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        nominal_steps = 200
        simtime_ms = nominal_steps * self.dt_ms
        sigma = 0.45
        nest_out = _run_nest_trace(
            model_name='gauss_rate_ipn',
            params={
                'tau': 5.0,
                'lambda': 1.2,
                'sigma': sigma,
                'mu': 0.25,
                'g': 1.7,
                'rate': -0.3,
            },
            record_from=['rate', 'noise'],
            simtime_ms=simtime_ms,
            dt_ms=self.dt_ms,
        )
        replay_steps = nest_out['noise'].size

        with brainstate.environ.context(dt=self.dt):
            ipn_bp = gauss_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.2,
                sigma=sigma,
                mu=0.25,
                g=1.7,
                rate_initializer=braintools.init.Constant(-0.3),
            )
            ipn_bp.init_state()
            dftype = brainstate.environ.dftype()
            noise_seq = jnp.asarray(nest_out['noise'] / sigma, dtype=dftype)

            def body(noise_k):
                ipn_bp.update(noise=noise_k)
                return (ipn_bp.rate.value.reshape(-1)[0], ipn_bp.noise.value.reshape(-1)[0])

            results = brainstate.transform.for_loop(body, noise_seq)
            bp_rate = np.asarray(results[0])
            bp_noise = np.asarray(results[1])

        n_cmp = min(bp_rate.size, nest_out['rate'].size)
        npt.assert_allclose(bp_rate[:n_cmp], nest_out['rate'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noise[:n_cmp], nest_out['noise'][:n_cmp], atol=1e-12)

    def test_linear_summation_modes_match_nest(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        drive = 1.5
        weight = 0.5
        g = 1.8
        mu = 0.2
        sigma = 0.35
        nominal_steps = 300
        simtime_ms = nominal_steps * self.dt_ms

        nest_linear_sum = _run_nest_gauss_driven_trace(
            linear_summation=True,
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            g=g,
            mu=mu,
            sigma=sigma,
        )
        nest_event_sum = _run_nest_gauss_driven_trace(
            linear_summation=False,
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            g=g,
            mu=mu,
            sigma=sigma,
        )
        replay_steps = min(nest_linear_sum['noise'].size, nest_event_sum['noise'].size)

        with brainstate.environ.context(dt=self.dt):
            bp_linear_sum = gauss_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.0,
                mu=mu,
                sigma=sigma,
                g=g,
                linear_summation=True,
            )
            bp_event_sum = gauss_rate_ipn(
                1,
                tau=5.0 * u.ms,
                lambda_=1.0,
                mu=mu,
                sigma=sigma,
                g=g,
                linear_summation=False,
            )
            bp_linear_sum.init_state()
            bp_event_sum.init_state()

            dftype = brainstate.environ.dftype()
            event_spec = [{'rate': drive, 'weight': weight}]
            noise_linear = jnp.asarray(nest_linear_sum['noise'] / sigma, dtype=dftype)
            noise_event = jnp.asarray(nest_event_sum['noise'] / sigma, dtype=dftype)

            def body_linear(noise_k):
                bp_linear_sum.update(noise=noise_k, instant_rate_events=event_spec)
                return (bp_linear_sum.rate.value.reshape(-1)[0], bp_linear_sum.noise.value.reshape(-1)[0])

            def body_event(noise_k):
                bp_event_sum.update(noise=noise_k, instant_rate_events=event_spec)
                return (bp_event_sum.rate.value.reshape(-1)[0], bp_event_sum.noise.value.reshape(-1)[0])

            res_linear = brainstate.transform.for_loop(body_linear, noise_linear)
            res_event = brainstate.transform.for_loop(body_event, noise_event)
            y_linear_sum = np.asarray(res_linear[0])
            n_linear_sum = np.asarray(res_linear[1])
            y_event_sum = np.asarray(res_event[0])
            n_event_sum = np.asarray(res_event[1])

        n_cmp = min(y_linear_sum.size, nest_linear_sum['rate'].size)
        npt.assert_allclose(y_linear_sum[:n_cmp], nest_linear_sum['rate'][:n_cmp], atol=1e-10)
        npt.assert_allclose(n_linear_sum[:n_cmp], nest_linear_sum['noise'][:n_cmp], atol=1e-12)

        n_cmp = min(y_event_sum.size, nest_event_sum['rate'].size)
        npt.assert_allclose(y_event_sum[:n_cmp], nest_event_sum['rate'][:n_cmp], atol=1e-10)
        npt.assert_allclose(n_event_sum[:n_cmp], nest_event_sum['noise'][:n_cmp], atol=1e-12)

        # Ensure both modes are distinct for this setup.
        self.assertFalse(np.allclose(y_linear_sum, y_event_sum))


if __name__ == '__main__':
    unittest.main()
