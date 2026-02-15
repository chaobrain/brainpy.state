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
from dataclasses import dataclass

import brainstate
import jax
import numpy as np
import numpy.testing as npt

from brainpy.state import jonke_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


@dataclass
class _HistEntry:
    t_: float
    access_counter_: int = 0


class _FakeJonkeTarget:
    def __init__(self, post_spike_times_ms, tau_minus, stdp_eps=1.0e-6):
        times = np.asarray(post_spike_times_ms, dtype=np.float64).reshape(-1)
        self.history = [_HistEntry(float(t), 0) for t in np.sort(times)]
        self.tau_minus = float(tau_minus)
        self.stdp_eps = float(stdp_eps)

    def get_history(self, t1, t2):
        out = []
        t1 = float(t1)
        t2 = float(t2)
        for e in self.history:
            if e.t_ - self.stdp_eps <= t1:
                continue
            if e.t_ - self.stdp_eps <= t2:
                e.access_counter_ += 1
                out.append(e)
        return out

    def get_K_value(self, t):
        t = float(t)
        if t < 0.0:
            return 0.0
        kminus = 0.0
        for e in self.history:
            if e.t_ <= t + self.stdp_eps:
                kminus += math.exp((e.t_ - t) / self.tau_minus)
        return kminus


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _facilitate_ref(w, kplus, p):
    if p['lambda'] == 0.0:
        return w
    dw = p['lambda'] * (math.exp(p['mu_plus'] * w) * kplus - p['beta'])
    w_new = w + dw
    return w_new if w_new < p['Wmax'] else p['Wmax']


def _depress_ref(w, kminus, p):
    if p['lambda'] == 0.0:
        return w
    dw = p['lambda'] * (-p['alpha'] * math.exp(p['mu_minus'] * w) * kminus - p['beta'])
    w_new = w + dw
    return w_new if w_new > 0.0 else 0.0


def _jonke_reference_weight_trace(pre_spike_times_ms, post_spike_times_ms, params, stdp_eps=1.0e-6):
    pre = np.asarray(pre_spike_times_ms, dtype=np.float64).reshape(-1)
    post = np.asarray(post_spike_times_ms, dtype=np.float64).reshape(-1)

    w = float(params['weight'])
    kplus = float(params['Kplus'])
    t_last = float(params['t_last_spike_ms'])
    delay = float(params['delay'])
    tau_minus = float(params['tau_minus'])
    tau_plus = float(params['tau_plus'])

    p = {
        'alpha': float(params['alpha']),
        'beta': float(params['beta']),
        'lambda': float(params['lambda']),
        'mu_plus': float(params['mu_plus']),
        'mu_minus': float(params['mu_minus']),
        'Wmax': float(params['Wmax']),
    }

    weights = np.empty((pre.size,), dtype=np.float64)
    kplus_state = np.empty((pre.size,), dtype=np.float64)

    for i, t_pre in enumerate(pre):
        t_hist_lo = t_last - delay
        t_hist_hi = float(t_pre) - delay

        for t_post in post:
            if t_post - stdp_eps <= t_hist_lo:
                continue
            if t_post - stdp_eps <= t_hist_hi:
                minus_dt = t_last - (float(t_post) + delay)
                w = _facilitate_ref(w, kplus * math.exp(minus_dt / tau_plus), p)

        kminus = 0.0
        if t_hist_hi >= 0.0:
            for t_post in post:
                if t_post <= t_hist_hi + stdp_eps:
                    kminus += math.exp((float(t_post) - t_hist_hi) / tau_minus)

        w = _depress_ref(w, kminus, p)
        weights[i] = w

        kplus = kplus * math.exp((t_last - float(t_pre)) / tau_plus) + 1.0
        kplus_state[i] = kplus
        t_last = float(t_pre)

    return weights, kplus, t_last, kplus_state


class TestJonkeSynapse(unittest.TestCase):
    def test_nest_default_parameters_and_properties(self):
        syn = jonke_synapse()

        self.assertAlmostEqual(syn.weight, 1.0, delta=0.0)
        self.assertAlmostEqual(syn.delay, 1.0, delta=0.0)
        self.assertEqual(syn.delay_steps, 1)
        self.assertAlmostEqual(syn.Kplus, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.t_last_spike_ms, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.alpha, 1.0, delta=0.0)
        self.assertAlmostEqual(syn.beta, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.lambda_, 0.01, delta=0.0)
        self.assertAlmostEqual(syn.mu_plus, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.mu_minus, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.tau_plus, 20.0, delta=0.0)
        self.assertAlmostEqual(syn.Wmax, 100.0, delta=0.0)

        self.assertTrue(syn.HAS_DELAY)
        self.assertTrue(syn.IS_PRIMARY)
        self.assertTrue(syn.SUPPORTS_HPC)
        self.assertTrue(syn.SUPPORTS_LBL)
        self.assertFalse(syn.SUPPORTS_WFR)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 1.0, delta=0.0)
        self.assertAlmostEqual(status['delay'], 1.0, delta=0.0)
        self.assertEqual(status['delay_steps'], 1)
        self.assertAlmostEqual(status['Kplus'], 0.0, delta=0.0)
        self.assertAlmostEqual(status['alpha'], 1.0, delta=0.0)
        self.assertAlmostEqual(status['beta'], 0.0, delta=0.0)
        self.assertAlmostEqual(status['lambda'], 0.01, delta=0.0)
        self.assertAlmostEqual(status['mu_plus'], 0.0, delta=0.0)
        self.assertAlmostEqual(status['mu_minus'], 0.0, delta=0.0)
        self.assertAlmostEqual(status['tau_plus'], 20.0, delta=0.0)
        self.assertAlmostEqual(status['Wmax'], 100.0, delta=0.0)
        self.assertIn('size_of', status)

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            defaults = nest.GetDefaults('jonke_synapse')
            self.assertAlmostEqual(syn.weight, float(defaults['weight']), delta=0.0)
            self.assertAlmostEqual(syn.alpha, float(defaults['alpha']), delta=0.0)
            self.assertAlmostEqual(syn.beta, float(defaults['beta']), delta=0.0)
            self.assertAlmostEqual(syn.lambda_, float(defaults['lambda']), delta=0.0)
            self.assertAlmostEqual(syn.mu_plus, float(defaults['mu_plus']), delta=0.0)
            self.assertAlmostEqual(syn.mu_minus, float(defaults['mu_minus']), delta=0.0)
            self.assertAlmostEqual(syn.tau_plus, float(defaults['tau_plus']), delta=0.0)
            self.assertAlmostEqual(syn.Wmax, float(defaults['Wmax']), delta=0.0)
            self.assertIn('delay', defaults)

    def test_set_status_and_validation(self):
        syn = jonke_synapse()
        syn.set_status(
            {
                'weight': 2.5,
                'delay': 0.2,
                'delay_steps': 3,
                'Kplus': 0.4,
                'alpha': 1.2,
                'beta': 0.01,
                'lambda': 0.03,
                'mu_plus': -0.5,
                'mu_minus': 0.2,
                'tau_plus': 30.0,
                'Wmax': 50.0,
            }
        )

        self.assertAlmostEqual(syn.weight, 2.5, delta=0.0)
        self.assertAlmostEqual(syn.delay, 0.2, delta=0.0)
        self.assertEqual(syn.delay_steps, 3)
        self.assertAlmostEqual(syn.Kplus, 0.4, delta=0.0)
        self.assertAlmostEqual(syn.alpha, 1.2, delta=0.0)
        self.assertAlmostEqual(syn.beta, 0.01, delta=0.0)
        self.assertAlmostEqual(syn.lambda_, 0.03, delta=0.0)
        self.assertAlmostEqual(syn.mu_plus, -0.5, delta=0.0)
        self.assertAlmostEqual(syn.mu_minus, 0.2, delta=0.0)
        self.assertAlmostEqual(syn.tau_plus, 30.0, delta=0.0)
        self.assertAlmostEqual(syn.Wmax, 50.0, delta=0.0)

        syn.set_status(lambda_=0.04)
        self.assertAlmostEqual(syn.lambda_, 0.04, delta=0.0)

        with self.assertRaisesRegex(ValueError, 'Kplus must be non-negative'):
            syn.set_status(Kplus=-1e-3)
        with self.assertRaisesRegex(ValueError, 'delay must be > 0'):
            syn.set_status(delay=0.0)
        with self.assertRaisesRegex(ValueError, 'delay_steps must be >= 1'):
            syn.set_status(delay_steps=0)
        with self.assertRaisesRegex(ValueError, 'must be identical'):
            syn.set_status(**{'lambda': 0.1, 'lambda_': 0.2})

        target = _FakeJonkeTarget(post_spike_times_ms=[10.0], tau_minus=20.0)
        with self.assertRaisesRegex(ValueError, 'multiplicity must be >= 0'):
            syn.send(t_spike_ms=11.0, target=target, multiplicity=-1.0)

    def test_send_ordering_matches_reference_trace(self):
        pre = np.asarray([10.0, 18.0, 25.0, 37.0, 50.0], dtype=np.float64)
        post = np.asarray([5.0, 12.0, 20.0, 23.0, 40.0], dtype=np.float64)

        params = {
            'weight': 2.0,
            'delay': 1.5,
            'delay_steps': 2,
            'Kplus': 0.4,
            't_last_spike_ms': 0.0,
            'alpha': 0.7,
            'beta': 0.03,
            'lambda': 0.05,
            'mu_plus': -0.2,
            'mu_minus': 0.1,
            'tau_plus': 30.0,
            'Wmax': 5.0,
            'tau_minus': 25.0,
        }

        syn = jonke_synapse(
            weight=params['weight'],
            delay=params['delay'],
            delay_steps=params['delay_steps'],
            Kplus=params['Kplus'],
            t_last_spike_ms=params['t_last_spike_ms'],
            alpha=params['alpha'],
            beta=params['beta'],
            lambda_=params['lambda'],
            mu_plus=params['mu_plus'],
            mu_minus=params['mu_minus'],
            tau_plus=params['tau_plus'],
            Wmax=params['Wmax'],
        )
        target = _FakeJonkeTarget(post, tau_minus=params['tau_minus'])

        events = syn.simulate_pre_spike_train(pre, target=target)
        got_weights = np.asarray([e['weight'] for e in events], dtype=np.float64)
        got_kplus = np.asarray([e['Kplus_post'] for e in events], dtype=np.float64)

        ref_w, ref_kplus, ref_t_last, ref_kplus_trace = _jonke_reference_weight_trace(pre, post, params)

        npt.assert_allclose(got_weights, ref_w, atol=1e-15, rtol=0.0)
        npt.assert_allclose(got_kplus, ref_kplus_trace, atol=1e-15, rtol=0.0)
        self.assertAlmostEqual(syn.Kplus, ref_kplus, delta=1e-15)
        self.assertAlmostEqual(syn.t_last_spike_ms, ref_t_last, delta=0.0)

    def test_matches_nest_weight_trace_if_available(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        resolution = 0.1
        pre_spike_times = [10.0, 40.0, 75.0, 130.0, 210.0, 330.0, 500.0]
        post_spike_times = [20.0, 58.0, 90.0, 170.0, 260.0, 410.0]

        synapse_parameters = {
            'weight': 2.0,
            'delay': resolution,
        }
        synapse_constants = {
            'lambda': 0.1 * np.e,
            'alpha': 1.0 / np.e,
            'beta': 0.0,
            'mu_plus': -1.0,
            'mu_minus': 0.0,
            'tau_plus': 36.8,
            'Wmax': 100.0,
        }
        neuron_parameters = {
            'tau_minus': 33.7,
        }

        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()
        nest.local_num_threads = 1
        nest.resolution = resolution

        neurons = nest.Create('parrot_neuron', 2, params=neuron_parameters)
        pre_neuron = neurons[0]
        post_neuron = neurons[1]

        sg_pre = nest.Create('spike_generator', 1, params={'spike_times': pre_spike_times})
        sg_post = nest.Create('spike_generator', 1, params={'spike_times': post_spike_times})
        spike_recorder = nest.Create('spike_recorder')
        weight_recorder = nest.Create('weight_recorder')

        nest.Connect(sg_pre, pre_neuron, syn_spec={'synapse_model': 'static_synapse'})
        nest.Connect(sg_post, post_neuron, syn_spec={'synapse_model': 'static_synapse'})
        nest.Connect(pre_neuron + post_neuron, spike_recorder, syn_spec={'synapse_model': 'static_synapse'})

        nest.SetDefaults('jonke_synapse', {**synapse_constants, 'weight_recorder': weight_recorder})
        nest.Connect(pre_neuron, post_neuron, syn_spec={'synapse_model': 'jonke_synapse', **synapse_parameters})

        nest.Simulate(600.0)

        all_spikes = spike_recorder.events
        pre_gid = pre_neuron.tolist()[0]
        post_gid = post_neuron.tolist()[0]
        pre_spikes = np.asarray(all_spikes['times'][all_spikes['senders'] == pre_gid], dtype=np.float64)
        post_spikes = np.asarray(all_spikes['times'][all_spikes['senders'] == post_gid], dtype=np.float64)

        wr_events = nest.GetStatus(weight_recorder)[0]['events']
        nest_weights = np.asarray(wr_events['weights'], dtype=np.float64)

        syn = jonke_synapse(
            weight=synapse_parameters['weight'],
            delay=synapse_parameters['delay'],
            delay_steps=1,
            Kplus=0.0,
            t_last_spike_ms=0.0,
            alpha=synapse_constants['alpha'],
            beta=synapse_constants['beta'],
            lambda_=synapse_constants['lambda'],
            mu_plus=synapse_constants['mu_plus'],
            mu_minus=synapse_constants['mu_minus'],
            tau_plus=synapse_constants['tau_plus'],
            Wmax=synapse_constants['Wmax'],
        )
        target = _FakeJonkeTarget(post_spikes, tau_minus=neuron_parameters['tau_minus'])
        local_events = syn.simulate_pre_spike_train(pre_spikes, target=target)
        local_weights = np.asarray([e['weight'] for e in local_events], dtype=np.float64)

        self.assertEqual(local_weights.shape, nest_weights.shape)
        npt.assert_allclose(local_weights, nest_weights, atol=1e-12, rtol=0.0)

        conn = nest.GetConnections(source=pre_neuron, target=post_neuron, synapse_model='jonke_synapse')
        nest_final_weight = float(np.asarray(conn.get('weight'), dtype=np.float64).reshape(-1)[0])
        self.assertAlmostEqual(syn.weight, nest_final_weight, delta=1e-12)


if __name__ == '__main__':
    unittest.main()
