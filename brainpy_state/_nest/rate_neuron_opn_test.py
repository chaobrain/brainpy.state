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
from brainpy.state import lin_rate_opn, rate_neuron_opn

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
    dftype = brainstate.environ.dftype()
    out = {'times': np.asarray(ev['times'], dtype=dftype)}
    for key in record_from:
        out[key] = np.asarray(ev[key], dtype=dftype)
    return out


def _run_nest_opn_driven_trace(mode, dt_ms, simtime_ms, drive, weight, delay_ms):
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
    target = nest.Create('lin_rate_opn', params={
        'tau': 5.0,
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
    dftype = brainstate.environ.dftype()
    return np.asarray(mm.events['rate'], dtype=dftype)


class TestRateNeuronOPN(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(**kwargs)

    def test_nest_default_parameters(self):
        opn = rate_neuron_opn(1)
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
        self.assertEqual(opn.receptor_types, {'RATE': 0})

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            rate_neuron_opn(1, tau=0.0 * u.ms)
        with self.assertRaises(ValueError):
            rate_neuron_opn(1, sigma=-1e-3)

    def test_step_equations_match_template_update_order_linear_summation_true(self):
        params = dict(
            tau=7.0,
            sigma=0.4,
            mu=-0.3,
            mult_coupling=True,
            linear_summation=True,
            rate0=-0.2,
        )

        def input_nl(h):
            return 0.7 * h + 0.1 * np.square(h)

        def mult_ex(rate):
            return 0.9 * (0.6 - rate)

        def mult_in(rate):
            return 1.1 * (-0.2 + rate)

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
            neuron = rate_neuron_opn(
                1,
                tau=params['tau'] * u.ms,
                sigma=params['sigma'],
                mu=params['mu'],
                mult_coupling=params['mult_coupling'],
                linear_summation=params['linear_summation'],
                input_nonlinearity=input_nl,
                mult_coupling_ex_fn=mult_ex,
                mult_coupling_in_fn=mult_in,
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
                rate_new += P2 * float(mult_ex(noisy_ref)) * float(input_nl(delayed_ex + instant_ex))
                rate_new += P2 * float(mult_in(noisy_ref)) * float(input_nl(delayed_in + instant_in))

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

    def test_step_equations_match_template_update_order_linear_summation_false(self):
        params = dict(
            tau=9.0,
            sigma=0.6,
            mu=0.8,
            mult_coupling=True,
            linear_summation=False,
            rate0=0.3,
        )

        def input_nl(h):
            return np.tanh(1.2 * h) + 0.1 * h

        def mult_ex(rate):
            return 1.0 + 0.2 * rate

        def mult_in(rate):
            return 0.8 - 0.3 * rate

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
            neuron = rate_neuron_opn(
                1,
                tau=params['tau'] * u.ms,
                sigma=params['sigma'],
                mu=params['mu'],
                mult_coupling=params['mult_coupling'],
                linear_summation=params['linear_summation'],
                input_nonlinearity=input_nl,
                mult_coupling_ex_fn=mult_ex,
                mult_coupling_in_fn=mult_in,
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
                    val = float(ev['weight']) * float(input_nl(float(ev['rate'])))
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
                    val = float(ev['weight']) * float(input_nl(float(ev['rate'])))
                    val *= float(ev.get('multiplicity', 1.0))
                    if ev['weight'] >= 0.0:
                        instant_ex += val
                    else:
                        instant_in += val

                noise_ref = params['sigma'] * noise_seq[k]
                noisy_ref = rate_ref + noise_fac * noise_ref
                rate_new = P1 * rate_ref + P2 * params['mu']
                rate_new += P2 * float(mult_ex(noisy_ref)) * (delayed_ex + instant_ex)
                rate_new += P2 * float(mult_in(noisy_ref)) * (delayed_in + instant_in)

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

    def test_default_linear_template_matches_lin_rate_opn(self):
        steps = 64
        dftype = brainstate.environ.dftype()
        noise_seq = np.asarray(
            [0.2, -1.0, 0.4, -0.3, 1.1, 0.0, -0.8, 0.7] * 8,
            dtype=dftype,
        )
        # Events used every step:
        #   instant_ev: {rate=0.7, weight=0.3} -> ex; {rate=-0.5, weight=-0.2} -> in
        #   delayed_ev: {rate=0.4, weight=0.6, delay=2} -> arrives 2 steps later -> ex

        with brainstate.environ.context(dt=self.dt):
            opn_template = rate_neuron_opn(
                1,
                tau=6.0 * u.ms,
                sigma=0.5,
                mu=-0.2,
                g=1.3,
                mult_coupling=True,
                g_ex=0.9,
                g_in=1.1,
                theta_ex=0.2,
                theta_in=-0.3,
                linear_summation=False,
                rate_initializer=braintools.init.Constant(0.1),
                noisy_rate_initializer=braintools.init.Constant(0.1),
            )
            opn_linear = lin_rate_opn(
                1,
                tau=6.0 * u.ms,
                sigma=0.5,
                mu=-0.2,
                g=1.3,
                mult_coupling=True,
                g_ex=0.9,
                g_in=1.1,
                theta_ex=0.2,
                theta_in=-0.3,
                linear_summation=False,
                rate_initializer=braintools.init.Constant(0.1),
                noisy_rate_initializer=braintools.init.Constant(0.1),
            )
            opn_template.init_state()
            opn_linear.init_state()

            # --- Pre-compute per-step inputs for opn_template (rate_neuron_opn) ---
            # For linear_summation=False in rate_neuron_opn, _event_to_ex_in applies
            # the input nonlinearity g(h)=g*h during event processing.
            g_val = float(np.asarray(u.get_mantissa(opn_template.g)))  # 1.3
            # instant event {rate=0.7, weight=0.3}: _input_transform(0.7)=1.3*0.7=0.91
            #   weighted_value = 0.91*0.3 = 0.273; weight>0 -> ex=0.273
            # instant event {rate=-0.5, weight=-0.2}: _input_transform(-0.5)=1.3*(-0.5)=-0.65
            #   weighted_value = -0.65*(-0.2) = 0.13; weight<0 -> ex=0, in=0.13
            # delayed {rate=0.4, weight=0.6, delay=2}: _input_transform(0.4)=1.3*0.4=0.52
            #   weighted_value = 0.52*0.6 = 0.312; weight>0 -> ex=0.312; arrives at k+2
            opn_instant_ex_val = g_val * 0.7 * 0.3   # 0.273
            opn_instant_in_val = abs(g_val * (-0.5) * (-0.2))  # 0.13
            opn_delayed_ex_val = g_val * 0.4 * 0.6   # 0.312
            precomputed_ex_opn = np.full((steps, 1), opn_instant_ex_val, dtype=dftype)
            precomputed_in_opn = np.full((steps, 1), opn_instant_in_val, dtype=dftype)
            precomputed_ex_opn[2:] += opn_delayed_ex_val

            # --- Pre-compute per-step inputs for opn_linear (lin_rate_opn) ---
            # For lin_rate_opn, _parse_event does rate*weight (no nonlinearity during buffering).
            # The g factor is applied inside update() as: rate_new += P2*H*g*(ex+instant_ex).
            # instant event {rate=0.7, weight=0.3}: ex = 0.7*0.3 = 0.21
            # instant event {rate=-0.5, weight=-0.2}: weighted=(-0.5)*(-0.2)=0.1; weight<0->in=0.1
            # delayed {rate=0.4, weight=0.6, delay=2}: ex = 0.4*0.6 = 0.24; arrives at k+2
            lin_instant_ex_val = 0.7 * 0.3    # 0.21
            lin_instant_in_val = 0.5 * 0.2    # 0.10
            lin_delayed_ex_val = 0.4 * 0.6    # 0.24
            precomputed_ex_lin = np.full((steps, 1), lin_instant_ex_val, dtype=dftype)
            precomputed_in_lin = np.full((steps, 1), lin_instant_in_val, dtype=dftype)
            precomputed_ex_lin[2:] += lin_delayed_ex_val

            noise_input = jnp.asarray(noise_seq, dtype=dftype).reshape(steps, 1)
            ex_opn_j = jnp.asarray(precomputed_ex_opn)
            in_opn_j = jnp.asarray(precomputed_in_opn)
            ex_lin_j = jnp.asarray(precomputed_ex_lin)
            in_lin_j = jnp.asarray(precomputed_in_lin)

            def body_opn(inputs):
                noise_k, ex_k, in_k = inputs
                opn_template.update(noise=noise_k, _precomputed_ex=ex_k, _precomputed_in=in_k)
                return (
                    opn_template.rate.value,
                    opn_template.noise.value,
                    opn_template.noisy_rate.value,
                    opn_template.delayed_rate.value,
                    opn_template.instant_rate.value,
                )

            def body_lin(inputs):
                noise_k, ex_k, in_k = inputs
                opn_linear.update(noise=noise_k, _precomputed_ex=ex_k, _precomputed_in=in_k)
                return (
                    opn_linear.rate.value,
                    opn_linear.noise.value,
                    opn_linear.noisy_rate.value,
                    opn_linear.delayed_rate.value,
                    opn_linear.instant_rate.value,
                )

            res_opn = brainstate.transform.for_loop(body_opn, (noise_input, ex_opn_j, in_opn_j))
            res_lin = brainstate.transform.for_loop(body_lin, (noise_input, ex_lin_j, in_lin_j))

        npt.assert_allclose(res_opn[0], res_lin[0], atol=1e-12)
        npt.assert_allclose(res_opn[1], res_lin[1], atol=1e-12)
        npt.assert_allclose(res_opn[2], res_lin[2], atol=1e-12)
        npt.assert_allclose(res_opn[3], res_lin[3], atol=1e-12)
        npt.assert_allclose(res_opn[4], res_lin[4], atol=1e-12)

    def test_matches_nest_lin_rate_trace_with_default_linear_template(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        steps = 200
        simtime_ms = steps * self.dt_ms
        nest_out = _run_nest_trace(
            model_name='lin_rate_opn',
            params={
                'tau': 7.0,
                'sigma': 0.0,
                'mu': -0.8,
                'rate': 0.4,
                'g': 1.3,
                'mult_coupling': True,
                'g_ex': 0.8,
                'g_in': 1.2,
                'theta_ex': 0.6,
                'theta_in': -0.1,
                'linear_summation': True,
            },
            record_from=['rate', 'noise', 'noisy_rate'],
            simtime_ms=simtime_ms,
            dt_ms=self.dt_ms,
        )

        with brainstate.environ.context(dt=self.dt):
            bp = rate_neuron_opn(
                1,
                tau=7.0 * u.ms,
                sigma=0.0,
                mu=-0.8,
                g=1.3,
                mult_coupling=True,
                g_ex=0.8,
                g_in=1.2,
                theta_ex=0.6,
                theta_in=-0.1,
                linear_summation=True,
                rate_initializer=braintools.init.Constant(0.4),
                noisy_rate_initializer=braintools.init.Constant(0.4),
            )
            bp.init_state()
            dftype = brainstate.environ.dftype()
            # sigma=0 -> no noise; no events -> pre-computed inputs are zeros.
            zeros = jnp.zeros((1,), dtype=dftype)

            def _body(_k):
                bp.update(_precomputed_ex=zeros, _precomputed_in=zeros, noise=zeros)
                return (bp.rate.value[0], bp.noise.value[0], bp.noisy_rate.value[0])

            _res = brainstate.transform.for_loop(_body, jnp.arange(steps))
            bp_rate = np.array(_res[0])
            bp_noise = np.array(_res[1])
            bp_noisy = np.array(_res[2])

        n_cmp = min(bp_rate.size, nest_out['rate'].size)
        npt.assert_allclose(bp_rate[:n_cmp], nest_out['rate'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noise[:n_cmp], nest_out['noise'][:n_cmp], atol=1e-12)
        npt.assert_allclose(bp_noisy[:n_cmp], nest_out['noisy_rate'][:n_cmp], atol=1e-12)

    def test_default_linear_template_instantaneous_and_delayed_driven_trace_matches_nest(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        drive = 1.5
        weight = 0.5
        delay_ms = 2.0
        delay_steps = int(round(delay_ms / self.dt_ms))
        steps = 300
        simtime_ms = steps * self.dt_ms

        nest_instant = _run_nest_opn_driven_trace(
            mode='instantaneous',
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            delay_ms=delay_ms,
        )
        nest_delayed = _run_nest_opn_driven_trace(
            mode='delayed',
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            delay_ms=delay_ms,
        )

        with brainstate.environ.context(dt=self.dt):
            bp_instant = rate_neuron_opn(1, tau=5.0 * u.ms, mu=0.0, sigma=0.0)
            bp_delayed = rate_neuron_opn(1, tau=5.0 * u.ms, mu=0.0, sigma=0.0)
            bp_instant.init_state()
            bp_delayed.init_state()

            dftype = brainstate.environ.dftype()
            # Pre-compute per-step inputs (linear_summation=True, mult_coupling=False).
            # For linear_summation=True, _event_to_ex_in uses rate*weight (pre-nonlinearity).
            # The nonlinearity g(h)=g*h is applied inside update() to the accumulated inputs.
            # default g=1.0 for these neurons, so ex = drive * weight = 1.5 * 0.5 = 0.75.
            inst_ex_val = np.float64(drive * weight)   # 0.75 per step
            zeros1 = jnp.zeros((1,), dtype=dftype)

            # Instantaneous: ex=0.75 every step; no delayed events.
            ex_instant = jnp.full((steps, 1), inst_ex_val, dtype=dftype)
            in_instant = jnp.zeros((steps, 1), dtype=dftype)

            # Delayed with delay_steps steps: ex=0.75 arrives starting from step delay_steps.
            ex_delayed = jnp.where(
                jnp.arange(steps, dtype=dftype).reshape(steps, 1) >= np.float64(delay_steps),
                inst_ex_val,
                0.0,
            )
            in_delayed = jnp.zeros((steps, 1), dtype=dftype)

            def _body_instant(inputs):
                ex_k, in_k = inputs
                bp_instant.update(_precomputed_ex=ex_k, _precomputed_in=in_k, noise=zeros1)
                return bp_instant.rate.value[0]

            def _body_delayed(inputs):
                ex_k, in_k = inputs
                bp_delayed.update(_precomputed_ex=ex_k, _precomputed_in=in_k, noise=zeros1)
                return bp_delayed.rate.value[0]

            trace_instant = np.array(
                brainstate.transform.for_loop(_body_instant, (ex_instant, in_instant))
            )
            trace_delayed = np.array(
                brainstate.transform.for_loop(_body_delayed, (ex_delayed, in_delayed))
            )

        n_cmp = min(trace_instant.size, nest_instant.size)
        npt.assert_allclose(trace_instant[:n_cmp], nest_instant[:n_cmp], atol=1e-10)
        n_cmp = min(trace_delayed.size, nest_delayed.size)
        npt.assert_allclose(trace_delayed[:n_cmp], nest_delayed[:n_cmp], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
