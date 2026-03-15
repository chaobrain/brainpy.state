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
import saiunit as u
import jax
import numpy as np
import numpy.testing as npt
from brainpy.state import vogels_sprekeler_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


@dataclass
class _HistEntry:
    t_: float
    access_counter_: int = 0


class _FakeVogelsTarget:
    def __init__(self, post_spike_times_ms, tau, stdp_eps=1.0e-6):
        dftype = brainstate.environ.dftype()
        times = np.asarray(post_spike_times_ms, dtype=dftype).reshape(-1)
        self.history = [_HistEntry(float(t), 0) for t in np.sort(times)]
        self.tau = float(tau)
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
                kminus += math.exp((e.t_ - t) / self.tau)
        return kminus


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _facilitate_ref(w, kplus, eta, wmax):
    new_w = abs(float(w)) + float(eta) * float(kplus)
    return math.copysign(min(new_w, abs(float(wmax))), float(wmax))


def _depress_ref(w, alpha, eta, wmax):
    new_w = abs(float(w)) - float(alpha) * float(eta)
    return math.copysign(max(new_w, 0.0), float(wmax))


def _vogels_reference_weight_trace(pre_spike_times_ms, post_spike_times_ms, params, stdp_eps=1.0e-6):
    dftype = brainstate.environ.dftype()
    pre = np.asarray(pre_spike_times_ms, dtype=dftype).reshape(-1)
    post = np.asarray(post_spike_times_ms, dtype=dftype).reshape(-1)

    w = float(params['weight'])
    kplus = float(params['Kplus'])
    t_last = float(params['t_last_spike_ms'])
    delay = float(params['delay'])
    tau = float(params['tau'])
    eta = float(params['eta'])
    alpha = float(params['alpha'])
    wmax = float(params['Wmax'])

    weights = np.empty((pre.size,), dtype=dftype)
    kplus_trace = np.empty((pre.size,), dtype=dftype)

    for i, t_pre in enumerate(pre):
        t_hist_lo = t_last - delay
        t_hist_hi = float(t_pre) - delay

        for t_post in post:
            if t_post - stdp_eps <= t_hist_lo:
                continue
            if t_post - stdp_eps <= t_hist_hi:
                minus_dt = t_last - (float(t_post) + delay)
                w = _facilitate_ref(w, kplus * math.exp(minus_dt / tau), eta, wmax)

        kminus = 0.0
        if t_hist_hi >= 0.0:
            for t_post in post:
                if t_post <= t_hist_hi + stdp_eps:
                    kminus += math.exp((float(t_post) - t_hist_hi) / tau)

        w = _facilitate_ref(w, kminus, eta, wmax)
        w = _depress_ref(w, alpha, eta, wmax)
        weights[i] = w

        kplus = kplus * math.exp((t_last - float(t_pre)) / tau) + 1.0
        kplus_trace[i] = kplus
        t_last = float(t_pre)

    return weights, kplus, t_last, kplus_trace


class TestVogelsSprekelerSynapse(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_default_parameters_and_properties(self):
        syn = vogels_sprekeler_synapse()

        self.assertAlmostEqual(syn.weight, 0.5, delta=0.0)
        self.assertAlmostEqual(syn.delay, 1.0, delta=0.0)
        self.assertEqual(syn.delay_steps, 1)
        self.assertAlmostEqual(syn.tau, 20.0, delta=0.0)
        self.assertAlmostEqual(syn.alpha, 0.12, delta=0.0)
        self.assertAlmostEqual(syn.eta, 0.001, delta=0.0)
        self.assertAlmostEqual(syn.Wmax, 1.0, delta=0.0)
        self.assertAlmostEqual(syn.Kplus, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.t_last_spike_ms, 0.0, delta=0.0)

        self.assertTrue(syn.HAS_DELAY)
        self.assertTrue(syn.IS_PRIMARY)
        self.assertTrue(syn.SUPPORTS_HPC)
        self.assertTrue(syn.SUPPORTS_LBL)
        self.assertTrue(syn.SUPPORTS_WFR)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 0.5, delta=0.0)
        self.assertAlmostEqual(status['tau'], 20.0, delta=0.0)
        self.assertAlmostEqual(status['alpha'], 0.12, delta=0.0)
        self.assertAlmostEqual(status['eta'], 0.001, delta=0.0)
        self.assertAlmostEqual(status['Wmax'], 1.0, delta=0.0)
        self.assertAlmostEqual(status['Kplus'], 0.0, delta=0.0)
        self.assertIn('size_of', status)

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            defaults = nest.GetDefaults('vogels_sprekeler_synapse')
            self.assertAlmostEqual(syn.weight, float(defaults['weight']), delta=0.0)
            self.assertAlmostEqual(syn.tau, float(defaults['tau']), delta=0.0)
            self.assertAlmostEqual(syn.alpha, float(defaults['alpha']), delta=0.0)
            self.assertAlmostEqual(syn.eta, float(defaults['eta']), delta=0.0)
            self.assertAlmostEqual(syn.Wmax, float(defaults['Wmax']), delta=0.0)
            self.assertIn('delay', defaults)

    def test_set_status_and_validation(self):
        syn = vogels_sprekeler_synapse()
        syn.set_status(
            {
                'weight': 2.5,
                'delay': 0.2,
                'delay_steps': 3,
                'tau': 30.0,
                'alpha': 0.1,
                'eta': 0.01,
                'Wmax': 10.0,
                'Kplus': 0.4,
            }
        )

        self.assertAlmostEqual(syn.weight, 2.5, delta=0.0)
        self.assertAlmostEqual(syn.delay, 0.2, delta=0.0)
        self.assertEqual(syn.delay_steps, 3)
        self.assertAlmostEqual(syn.tau, 30.0, delta=0.0)
        self.assertAlmostEqual(syn.alpha, 0.1, delta=0.0)
        self.assertAlmostEqual(syn.eta, 0.01, delta=0.0)
        self.assertAlmostEqual(syn.Wmax, 10.0, delta=0.0)
        self.assertAlmostEqual(syn.Kplus, 0.4, delta=0.0)

        with self.assertRaisesRegex(ValueError, 'State Kplus must be positive'):
            syn.set_status(Kplus=-1e-3)
        with self.assertRaisesRegex(ValueError, 'delay must be > 0'):
            vogels_sprekeler_synapse().set_status(delay=0.0)
        with self.assertRaisesRegex(ValueError, 'delay_steps must be >= 1'):
            vogels_sprekeler_synapse().set_status(delay_steps=0)
        with self.assertRaisesRegex(ValueError, 'tau must be > 0'):
            vogels_sprekeler_synapse().set_status(tau=0.0)
        with self.assertRaisesRegex(ValueError, 'Weight and Wmax must have same sign'):
            vogels_sprekeler_synapse().set_status(weight=1.0, Wmax=-1.0)

        target = _FakeVogelsTarget(post_spike_times_ms=[10.0], tau=20.0)
        with self.assertRaisesRegex(ValueError, 'multiplicity must be >= 0'):
            syn.send(t_spike_ms=11.0, target=target, multiplicity=-1.0)

    def test_send_ordering_matches_reference_trace(self):
        dftype = brainstate.environ.dftype()
        pre = np.asarray([10.0, 18.0, 25.0, 37.0, 50.0], dtype=dftype)
        post = np.asarray([5.0, 12.0, 20.0, 23.0, 40.0], dtype=dftype)

        params = {
            'weight': 2.0,
            'delay': 1.5,
            'delay_steps': 2,
            'Kplus': 0.4,
            't_last_spike_ms': 0.0,
            'tau': 30.0,
            'alpha': 0.2,
            'eta': 0.03,
            'Wmax': 5.0,
        }

        syn = vogels_sprekeler_synapse(
            weight=params['weight'],
            delay=params['delay'],
            delay_steps=params['delay_steps'],
            Kplus=params['Kplus'],
            t_last_spike_ms=params['t_last_spike_ms'],
            tau=params['tau'],
            alpha=params['alpha'],
            eta=params['eta'],
            Wmax=params['Wmax'],
        )
        target = _FakeVogelsTarget(post, tau=params['tau'])

        events = syn.simulate_pre_spike_train(pre, target=target)
        got_weights = np.asarray([e['weight'] for e in events], dtype=dftype)
        got_kplus = np.asarray([e['Kplus_post'] for e in events], dtype=dftype)

        ref_w, ref_kplus, ref_t_last, ref_kplus_trace = _vogels_reference_weight_trace(pre, post, params)

        npt.assert_allclose(got_weights, ref_w, atol=1e-15, rtol=0.0)
        npt.assert_allclose(got_kplus, ref_kplus_trace, atol=1e-15, rtol=0.0)
        self.assertAlmostEqual(syn.Kplus, ref_kplus, delta=1e-15)
        self.assertAlmostEqual(syn.t_last_spike_ms, ref_t_last, delta=0.0)

    def test_signed_wmax_semantics_match_nest_helpers(self):
        syn = vogels_sprekeler_synapse(weight=-0.5, Wmax=-1.0, eta=0.1, alpha=0.5)

        f = syn._facilitate(-0.5, 4.0)
        d = syn._depress(f)

        self.assertLessEqual(f, 0.0)
        self.assertLessEqual(d, 0.0)
        self.assertAlmostEqual(abs(f), min(1.0, 0.5 + 0.4), delta=0.0)
        self.assertAlmostEqual(abs(d), max(abs(f) - 0.05, 0.0), delta=0.0)

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
            'tau': 20.0,
            'alpha': 0.1,
            'eta': 0.01,
            'Wmax': 15.0,
        }

        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()
        nest.local_num_threads = 1
        nest.resolution = resolution

        pre_neuron = nest.Create('parrot_neuron', 1)
        post_neuron = nest.Create('parrot_neuron', 1)

        sg_pre = nest.Create('spike_generator', 1, params={'spike_times': pre_spike_times})
        sg_post = nest.Create('spike_generator', 1, params={'spike_times': post_spike_times})
        spike_recorder = nest.Create('spike_recorder')
        weight_recorder = nest.Create('weight_recorder')

        nest.Connect(sg_pre, pre_neuron, syn_spec={'synapse_model': 'static_synapse'})
        nest.Connect(sg_post, post_neuron, syn_spec={'synapse_model': 'static_synapse'})
        nest.Connect(pre_neuron + post_neuron, spike_recorder, syn_spec={'synapse_model': 'static_synapse'})

        nest.SetDefaults('vogels_sprekeler_synapse', {**synapse_constants, 'weight_recorder': weight_recorder})
        nest.Connect(
            pre_neuron,
            post_neuron,
            syn_spec={'synapse_model': 'vogels_sprekeler_synapse', **synapse_parameters},
        )

        nest.Simulate(600.0)

        all_spikes = spike_recorder.events
        pre_gid = pre_neuron.tolist()[0]
        post_gid = post_neuron.tolist()[0]
        dftype = brainstate.environ.dftype()
        pre_spikes = np.asarray(all_spikes['times'][all_spikes['senders'] == pre_gid], dtype=dftype)
        post_spikes = np.asarray(all_spikes['times'][all_spikes['senders'] == post_gid], dtype=dftype)

        wr_events = nest.GetStatus(weight_recorder)[0]['events']
        nest_weights = np.asarray(wr_events['weights'], dtype=dftype)

        syn = vogels_sprekeler_synapse(
            weight=synapse_parameters['weight'],
            delay=synapse_parameters['delay'],
            delay_steps=1,
            tau=synapse_constants['tau'],
            alpha=synapse_constants['alpha'],
            eta=synapse_constants['eta'],
            Wmax=synapse_constants['Wmax'],
            Kplus=0.0,
            t_last_spike_ms=0.0,
        )
        target = _FakeVogelsTarget(post_spikes, tau=synapse_constants['tau'])
        local_events = syn.simulate_pre_spike_train(pre_spikes, target=target)
        local_weights = np.asarray([e['weight'] for e in local_events], dtype=dftype)

        self.assertEqual(local_weights.shape, nest_weights.shape)
        npt.assert_allclose(local_weights, nest_weights, atol=1e-12, rtol=0.0)

        conn = nest.GetConnections(
            source=pre_neuron,
            target=post_neuron,
            synapse_model='vogels_sprekeler_synapse',
        )
        nest_final_weight = float(np.asarray(conn.get('weight'), dtype=dftype).reshape(-1)[0])
        self.assertAlmostEqual(syn.weight, nest_final_weight, delta=1e-12)


if __name__ == '__main__':
    unittest.main()
