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

from brainpy.state import rate_transformer_node

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
    out = {'times': np.asarray(ev['times'], dtype=np.float64)}
    for key in record_from:
        out[key] = np.asarray(ev[key], dtype=np.float64)
    return out


def _run_nest_tanh_transformer_driven_trace(linear_summation, dt_ms, simtime_ms, drive, weight, g, theta):
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = True

    source = nest.Create('lin_rate_ipn')
    source.set({'rate': drive, 'mu': drive, 'sigma': 0.0})
    target = nest.Create('rate_transformer_tanh')
    target.set({
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
    return np.asarray(mm.events['rate'], dtype=np.float64)


class TestRateTransformerNode(unittest.TestCase):
    def setUp(self):
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, **kwargs):
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(**kwargs)

    def test_nest_default_parameters(self):
        node = rate_transformer_node(1)
        self.assertEqual(node.linear_summation, True)
        self.assertEqual(node.g, 1.0)
        self.assertEqual(node.recordables, ['rate'])
        self.assertEqual(node.receptor_types, {'RATE': 0})

    def test_step_equations_match_nest_update_order_linear_summation_true(self):
        def input_nl(h):
            return 0.7 * h + 0.3 * np.square(h)

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
            node = rate_transformer_node(
                1,
                linear_summation=True,
                input_nonlinearity=input_nl,
                rate_initializer=braintools.init.Constant(0.3),
            )
            node.init_state()

            delayed_queue = {}
            rate_ref = 0.3

            for k in range(len(instant_events_seq)):
                delayed_total = delayed_queue.pop(k, 0.0)

                for ev in delayed_events_seq[k]:
                    val = float(ev['rate']) * float(ev['weight']) * float(ev.get('multiplicity', 1.0))
                    target = k + int(ev.get('delay_steps', 1))
                    if target == k:
                        delayed_total += val
                    else:
                        delayed_queue[target] = delayed_queue.get(target, 0.0) + val

                instant_total = 0.0
                for ev in instant_events_seq[k]:
                    val = float(ev['rate']) * float(ev['weight']) * float(ev.get('multiplicity', 1.0))
                    instant_total += val

                rate_new = float(input_nl(delayed_total + instant_total))

                out = self._step(
                    node,
                    k,
                    instant_rate_events=instant_events_seq[k],
                    delayed_rate_events=delayed_events_seq[k],
                )

                self.assertAlmostEqual(float(np.asarray(out).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(node.rate.value).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(node.delayed_rate.value).reshape(-1)[0]), rate_ref, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(node.instant_rate.value).reshape(-1)[0]), rate_new, delta=1e-12)

                rate_ref = rate_new

    def test_step_equations_match_nest_update_order_linear_summation_false(self):
        def input_nl(h):
            return np.tanh(1.2 * h) + 0.1 * h

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
            node = rate_transformer_node(
                1,
                linear_summation=False,
                input_nonlinearity=input_nl,
                rate_initializer=braintools.init.Constant(-0.2),
            )
            node.init_state()

            delayed_queue = {}
            rate_ref = -0.2

            for k in range(len(instant_events_seq)):
                delayed_total = delayed_queue.pop(k, 0.0)

                for ev in delayed_events_seq[k]:
                    val = float(ev['weight']) * float(input_nl(float(ev['rate'])))
                    val *= float(ev.get('multiplicity', 1.0))
                    target = k + int(ev.get('delay_steps', 1))
                    if target == k:
                        delayed_total += val
                    else:
                        delayed_queue[target] = delayed_queue.get(target, 0.0) + val

                instant_total = 0.0
                for ev in instant_events_seq[k]:
                    val = float(ev['weight']) * float(input_nl(float(ev['rate'])))
                    val *= float(ev.get('multiplicity', 1.0))
                    instant_total += val

                rate_new = delayed_total + instant_total

                out = self._step(
                    node,
                    k,
                    instant_rate_events=instant_events_seq[k],
                    delayed_rate_events=delayed_events_seq[k],
                )

                self.assertAlmostEqual(float(np.asarray(out).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(node.rate.value).reshape(-1)[0]), rate_new, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(node.delayed_rate.value).reshape(-1)[0]), rate_ref, delta=1e-12)
                self.assertAlmostEqual(float(np.asarray(node.instant_rate.value).reshape(-1)[0]), rate_new, delta=1e-12)

                rate_ref = rate_new

    def test_event_delay_validation(self):
        with brainstate.environ.context(dt=self.dt):
            node = rate_transformer_node(1)
            node.init_state()
            with self.assertRaises(ValueError):
                self._step(node, 0, instant_rate_events=[{'rate': 1.0, 'weight': 1.0, 'delay_steps': 1}])
            with self.assertRaises(ValueError):
                self._step(node, 0, delayed_rate_events=[{'rate': 1.0, 'weight': 1.0, 'delay_steps': -1}])

    def test_matches_nest_tanh_transformer_trace(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        steps = 200
        simtime_ms = steps * self.dt_ms

        nest_out = _run_nest_trace(
            model_name='rate_transformer_tanh',
            params={
                'g': 1.7,
                'theta': 0.2,
                'linear_summation': True,
                'rate': -0.3,
            },
            record_from=['rate'],
            simtime_ms=simtime_ms,
            dt_ms=self.dt_ms,
        )

        with brainstate.environ.context(dt=self.dt):
            node = rate_transformer_node(
                1,
                linear_summation=True,
                input_nonlinearity=lambda h: np.tanh(1.7 * (h - 0.2)),
                rate_initializer=braintools.init.Constant(-0.3),
            )
            node.init_state()
            bp_rate = np.zeros((steps,), dtype=np.float64)
            for k in range(steps):
                self._step(node, k)
                bp_rate[k] = float(np.asarray(node.rate.value).reshape(-1)[0])

        n_cmp = min(bp_rate.size, nest_out['rate'].size)
        npt.assert_allclose(bp_rate[:n_cmp], nest_out['rate'][:n_cmp], atol=1e-12)

    def test_linear_summation_modes_match_nest_tanh_transformer(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        drive = 1.5
        weight = 0.5
        g = 1.8
        theta = 0.2
        steps = 300
        simtime_ms = steps * self.dt_ms

        nest_linear_sum = _run_nest_tanh_transformer_driven_trace(
            linear_summation=True,
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            g=g,
            theta=theta,
        )
        nest_event_sum = _run_nest_tanh_transformer_driven_trace(
            linear_summation=False,
            dt_ms=self.dt_ms,
            simtime_ms=simtime_ms,
            drive=drive,
            weight=weight,
            g=g,
            theta=theta,
        )

        with brainstate.environ.context(dt=self.dt):
            bp_linear_sum = rate_transformer_node(
                1,
                linear_summation=True,
                input_nonlinearity=lambda h: np.tanh(g * (h - theta)),
            )
            bp_event_sum = rate_transformer_node(
                1,
                linear_summation=False,
                input_nonlinearity=lambda h: np.tanh(g * (h - theta)),
            )
            bp_linear_sum.init_state()
            bp_event_sum.init_state()

            y_linear_sum = np.zeros((steps,), dtype=np.float64)
            y_event_sum = np.zeros((steps,), dtype=np.float64)
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

        self.assertFalse(np.allclose(y_linear_sum, y_event_sum))


if __name__ == '__main__':
    unittest.main()
