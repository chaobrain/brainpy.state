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
import jax
import numpy as np
import numpy.testing as npt

from brainpy.state import ht_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _ht_reference_weight_trace(spike_times_ms, tau_P, delta_P, p0=1.0):
    p = float(p0)
    t_last = 0.0
    w = np.empty((len(spike_times_ms),), dtype=np.float64)
    p_send = np.empty_like(w)
    p_post = np.empty_like(w)

    for i, t in enumerate(spike_times_ms):
        t = float(t)
        p = 1.0 - (1.0 - p) * math.exp(-(t - t_last) / float(tau_P))
        p_send[i] = p
        w[i] = p
        p *= (1.0 - float(delta_P))
        p_post[i] = p
        t_last = t

    return w, p_send, p_post


class TestHTSynapse(unittest.TestCase):
    def test_nest_default_parameters_and_properties(self):
        syn = ht_synapse()

        self.assertAlmostEqual(syn.weight, 1.0, delta=0.0)
        self.assertEqual(syn.delay_steps, 1)
        self.assertAlmostEqual(syn.tau_P, 500.0, delta=0.0)
        self.assertAlmostEqual(syn.delta_P, 0.125, delta=0.0)
        self.assertAlmostEqual(syn.P, 1.0, delta=0.0)
        self.assertAlmostEqual(syn.t_last_spike_ms, 0.0, delta=0.0)

        self.assertTrue(syn.HAS_DELAY)
        self.assertFalse(syn.SUPPORTS_WFR)
        self.assertTrue(syn.IS_PRIMARY)
        self.assertTrue(syn.SUPPORTS_HPC)
        self.assertTrue(syn.SUPPORTS_LBL)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 1.0, delta=0.0)
        self.assertEqual(status['delay_steps'], 1)
        self.assertEqual(status['delay'], 1)
        self.assertAlmostEqual(status['tau_P'], 500.0, delta=0.0)
        self.assertAlmostEqual(status['delta_P'], 0.125, delta=0.0)
        self.assertAlmostEqual(status['P'], 1.0, delta=0.0)
        self.assertIn('size_of', status)

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            defaults = nest.GetDefaults('ht_synapse')
            self.assertAlmostEqual(syn.weight, float(defaults['weight']), delta=0.0)
            self.assertAlmostEqual(syn.tau_P, float(defaults['tau_P']), delta=0.0)
            self.assertAlmostEqual(syn.delta_P, float(defaults['delta_P']), delta=0.0)
            self.assertAlmostEqual(syn.P, float(defaults['P']), delta=0.0)
            self.assertIn('delay', defaults)
            self.assertGreaterEqual(float(defaults['delay']), float(nest.resolution))

    def test_set_status_and_validation(self):
        syn = ht_synapse()
        syn.set_status({'weight': 2.5, 'delay_steps': 3, 'tau_P': 700.0, 'delta_P': 0.4, 'P': 0.9})
        self.assertAlmostEqual(syn.weight, 2.5, delta=0.0)
        self.assertEqual(syn.delay_steps, 3)
        self.assertAlmostEqual(syn.tau_P, 700.0, delta=0.0)
        self.assertAlmostEqual(syn.delta_P, 0.4, delta=0.0)
        self.assertAlmostEqual(syn.P, 0.9, delta=0.0)

        syn.set_status(delay=5)
        self.assertEqual(syn.delay_steps, 5)
        syn.set_delay_steps(4)
        self.assertEqual(syn.delay_steps, 4)
        syn.set_delay(6)
        self.assertEqual(syn.delay_steps, 6)

        with self.assertRaisesRegex(ValueError, 'tau_P > 0 required'):
            syn.set_status(tau_P=0.0)
        with self.assertRaisesRegex(ValueError, '0 <= delta_P <= 1 required'):
            syn.set_status(delta_P=1.1)
        with self.assertRaisesRegex(ValueError, '0 <= P <= 1 required'):
            syn.set_status(P=-0.01)
        with self.assertRaisesRegex(ValueError, 'must be >= 1'):
            syn.set_status(delay_steps=0)
        with self.assertRaisesRegex(ValueError, 'must be identical'):
            syn.set_status(delay=2, delay_steps=3)
        with self.assertRaisesRegex(ValueError, 'multiplicity must be >= 0'):
            syn.send(t_spike_ms=1.0, multiplicity=-1.0)

    def test_spike_ordering_matches_nest_reference_sequence(self):
        spike_times = np.asarray([10.0, 12.0, 20.0, 20.5, 100.0, 200.0, 1000.0], dtype=np.float64)
        syn = ht_synapse(weight=1.0, tau_P=500.0, delta_P=0.125, P=1.0)

        events = syn.simulate_spike_train(spike_times)
        got_weights = np.asarray([ev['weight'] for ev in events], dtype=np.float64)
        got_p_send = np.asarray([ev['P_send'] for ev in events], dtype=np.float64)
        got_p_post = np.asarray([ev['P_post'] for ev in events], dtype=np.float64)

        ref_weights, ref_p_send, ref_p_post = _ht_reference_weight_trace(
            spike_times_ms=spike_times,
            tau_P=500.0,
            delta_P=0.125,
            p0=1.0,
        )

        npt.assert_allclose(got_weights, ref_weights, atol=1e-15, rtol=0.0)
        npt.assert_allclose(got_p_send, ref_p_send, atol=1e-15, rtol=0.0)
        npt.assert_allclose(got_p_post, ref_p_post, atol=1e-15, rtol=0.0)

        self.assertAlmostEqual(syn.P, ref_p_post[-1], delta=1e-15)
        self.assertAlmostEqual(syn.t_last_spike_ms, float(spike_times[-1]), delta=0.0)

    def test_to_spike_event_payload_fields(self):
        syn = ht_synapse(weight=2.0, delay_steps=3, tau_P=500.0, delta_P=0.125, P=1.0)
        ev = syn.to_spike_event(t_spike_ms=12.5, receptor_type=2, multiplicity=4.0, delay_steps=7)

        self.assertEqual(ev['receptor_type'], 2)
        self.assertAlmostEqual(ev['multiplicity'], 4.0, delta=0.0)
        self.assertEqual(ev['delay_steps'], 7)
        self.assertEqual(ev['delay'], 7)
        self.assertAlmostEqual(ev['t_spike_ms'], 12.5, delta=0.0)

        self.assertAlmostEqual(ev['weight'], 2.0 * ev['P_send'], delta=1e-15)
        self.assertAlmostEqual(ev['P_post'], (1.0 - syn.delta_P) * ev['P_send'], delta=1e-15)

    def test_matches_nest_weight_recorder_trace_if_available(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        spike_times = [10.0, 12.0, 20.0, 20.5, 100.0, 200.0, 1000.0]
        weight = 1.0
        tau_P = 500.0
        delta_P = 0.125

        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()
        nest.local_num_threads = 1

        sg = nest.Create('spike_generator', params={'spike_times': spike_times})
        n = nest.Create('parrot_neuron', 2)
        wr = nest.Create('weight_recorder')

        nest.SetDefaults(
            'ht_synapse',
            {
                'weight_recorder': wr,
                'weight': weight,
                'tau_P': tau_P,
                'delta_P': delta_P,
                'P': 1.0,
            },
        )
        nest.Connect(sg, n[:1])
        nest.Connect(n[:1], n[1:], syn_spec='ht_synapse')
        nest.Simulate(1200.0)

        nest_w = np.asarray(wr.get('events', 'weights'), dtype=np.float64)
        self.assertEqual(nest_w.size, len(spike_times))

        syn = ht_synapse(weight=weight, tau_P=tau_P, delta_P=delta_P, P=1.0)
        local_events = syn.simulate_spike_train(spike_times)
        local_w = np.asarray([ev['weight'] for ev in local_events], dtype=np.float64)

        npt.assert_allclose(local_w, nest_w, atol=1e-15, rtol=0.0)

        conn = nest.GetConnections(source=n[:1], target=n[1:])
        nest_final_P = float(np.asarray(conn.get('P'), dtype=np.float64).reshape(-1)[0])
        self.assertAlmostEqual(syn.P, nest_final_P, delta=1e-15)


if __name__ == '__main__':
    unittest.main()
