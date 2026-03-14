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

import os

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import math
import unittest

import brainstate
import saiunit as u
import jax
import numpy as np

from brainpy.state import stdp_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

_STDP_EPS = 1.0e-6


class _MockReceiver:
    def __init__(self):
        self.delta_events = []

    def add_delta_input(self, key, inp, label=None):
        self.delta_events.append((key, inp, label))


def _spike_step_counts_from_times(spike_times_ms, dt_ms):
    dftype = brainstate.environ.dftype()
    spike_times = np.asarray(spike_times_ms, dtype=dftype).reshape(-1)
    counts = {}
    for t_spike in spike_times:
        step = int(round((float(t_spike) - dt_ms) / dt_ms))
        counts[step] = counts.get(step, 0) + 1
    return counts


def _record_post_ref(*, t_spike, tau_minus, post_kminus, last_post, post_hist_t, post_hist_kminus):
    post_kminus = post_kminus * math.exp((last_post - t_spike) / tau_minus) + 1.0
    last_post = float(t_spike)
    post_hist_t.append(last_post)
    post_hist_kminus.append(float(post_kminus))
    return post_kminus, last_post


def _history_window_ref(post_hist_t, t1, t2):
    t1_lim = t1 + _STDP_EPS
    t2_lim = t2 + _STDP_EPS
    return [t_post for t_post in post_hist_t if t_post >= t1_lim and t_post < t2_lim]


def _get_k_value_ref(post_hist_t, post_hist_kminus, t, tau_minus):
    for idx in range(len(post_hist_t) - 1, -1, -1):
        t_post = post_hist_t[idx]
        if (t - t_post) > _STDP_EPS:
            return post_hist_kminus[idx] * math.exp((t_post - t) / tau_minus)
    return 0.0


def _facilitate_ref(*, w, kplus, lambda_, mu_plus, Wmax):
    norm_w = (w / Wmax) + (lambda_ * math.pow(1.0 - (w / Wmax), mu_plus) * kplus)
    return norm_w * Wmax if norm_w < 1.0 else Wmax


def _depress_ref(*, w, kminus, alpha, lambda_, mu_minus, Wmax):
    norm_w = (w / Wmax) - (alpha * lambda_ * math.pow(w / Wmax, mu_minus) * kminus)
    return norm_w * Wmax if norm_w > 0.0 else 0.0


def _stdp_send_ref(
    *,
    weight,
    Kplus,
    t_lastspike,
    t_spike,
    multiplicity,
    delay,
    tau_plus,
    tau_minus,
    lambda_,
    alpha,
    mu_plus,
    mu_minus,
    Wmax,
    post_hist_t,
    post_hist_kminus,
):
    for t_post in _history_window_ref(post_hist_t, t_lastspike - delay, t_spike - delay):
        minus_dt = t_lastspike - (t_post + delay)
        assert minus_dt < (-1.0 * _STDP_EPS)
        kplus_term = Kplus * math.exp(minus_dt / tau_plus)
        weight = _facilitate_ref(w=weight, kplus=kplus_term, lambda_=lambda_, mu_plus=mu_plus, Wmax=Wmax)

    kminus = _get_k_value_ref(post_hist_t, post_hist_kminus, t_spike - delay, tau_minus)
    weight = _depress_ref(
        w=weight,
        kminus=kminus,
        alpha=alpha,
        lambda_=lambda_,
        mu_minus=mu_minus,
        Wmax=Wmax,
    )

    payload = float(multiplicity) * float(weight)
    Kplus = Kplus * math.exp((t_lastspike - t_spike) / tau_plus) + 1.0
    t_lastspike = float(t_spike)
    return weight, Kplus, t_lastspike, payload


def _run_bp_weight_trace(
    *,
    pre_spikes_ms,
    post_spikes_ms,
    sim_duration_ms,
    dt_ms,
    delay_ms,
    tau_plus,
    tau_minus,
    lambda_,
    alpha,
    mu_plus,
    mu_minus,
    weight,
    Wmax,
):
    dt = dt_ms * u.ms
    pre_counts = _spike_step_counts_from_times(pre_spikes_ms, dt_ms)
    post_counts = _spike_step_counts_from_times(post_spikes_ms, dt_ms)
    sim_steps = 1 + int(np.ceil(sim_duration_ms / dt_ms))

    recv = _MockReceiver()

    with brainstate.environ.context(dt=dt):
        syn = stdp_synapse(
            delay=delay_ms * u.ms,
            tau_plus=tau_plus * u.ms,
            tau_minus=tau_minus * u.ms,
            lambda_=lambda_,
            alpha=alpha,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            weight=weight,
            Wmax=Wmax,
            receptor_type=2,
            post=recv,
        )
        syn.init_state()

        send_times = []
        for step in range(sim_steps):
            pre_count = pre_counts.get(step, 0)
            post_count = post_counts.get(step, 0)
            with brainstate.environ.context(t=step * dt):
                syn.update(pre_spike=float(pre_count), post_spike=float(post_count))
            if pre_count > 0:
                send_times.append((step + 1) * dt_ms)

    dftype = brainstate.environ.dftype()
    payloads = np.asarray(
        [
            float(np.asarray(u.math.asarray(value), dtype=dftype).reshape(()))
            for _key, value, _label in recv.delta_events
        ],
        dtype=dftype,
    )
    labels = [label for _key, _value, label in recv.delta_events]
    return np.asarray(send_times, dtype=dftype), payloads, labels


def _run_reference_weight_trace(
    *,
    pre_spikes_ms,
    post_spikes_ms,
    sim_duration_ms,
    dt_ms,
    delay_ms,
    tau_plus,
    tau_minus,
    lambda_,
    alpha,
    mu_plus,
    mu_minus,
    weight,
    Wmax,
):
    pre_counts = _spike_step_counts_from_times(pre_spikes_ms, dt_ms)
    post_counts = _spike_step_counts_from_times(post_spikes_ms, dt_ms)
    sim_steps = 1 + int(np.ceil(sim_duration_ms / dt_ms))

    t_lastspike = 0.0
    Kplus = 0.0
    post_kminus = 0.0
    last_post = -1.0
    post_hist_t = []
    post_hist_kminus = []
    current_weight = float(weight)

    send_times = []
    payloads = []

    for step in range(sim_steps):
        t_spike = (step + 1) * dt_ms

        post_count = post_counts.get(step, 0)
        for _ in range(post_count):
            post_kminus, last_post = _record_post_ref(
                t_spike=t_spike,
                tau_minus=tau_minus,
                post_kminus=post_kminus,
                last_post=last_post,
                post_hist_t=post_hist_t,
                post_hist_kminus=post_hist_kminus,
            )

        pre_count = pre_counts.get(step, 0)
        if pre_count > 0:
            current_weight, Kplus, t_lastspike, payload = _stdp_send_ref(
                weight=current_weight,
                Kplus=Kplus,
                t_lastspike=t_lastspike,
                t_spike=t_spike,
                multiplicity=pre_count,
                delay=delay_ms,
                tau_plus=tau_plus,
                tau_minus=tau_minus,
                lambda_=lambda_,
                alpha=alpha,
                mu_plus=mu_plus,
                mu_minus=mu_minus,
                Wmax=Wmax,
                post_hist_t=post_hist_t,
                post_hist_kminus=post_hist_kminus,
            )
            send_times.append(t_spike)
            payloads.append(payload)

    dftype = brainstate.environ.dftype()
    return np.asarray(send_times, dtype=dftype), np.asarray(payloads, dtype=dftype)


class TestSTDPSynapseParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = stdp_synapse()
            syn.init_state()
            syn.update(pre_spike=0.0, post_spike=0.0)
            params = syn.get()

        self.assertEqual(params['weight'], 1.0)
        self.assertEqual(params['delay'], 1.0)
        self.assertEqual(params['delay_steps'], 1)
        self.assertEqual(params['receptor_type'], 0)
        self.assertEqual(params['event_type'], 'spike')
        self.assertEqual(params['tau_plus'], 20.0)
        self.assertEqual(params['tau_minus'], 20.0)
        self.assertEqual(params['lambda'], 0.01)
        self.assertEqual(params['alpha'], 1.0)
        self.assertEqual(params['mu_plus'], 1.0)
        self.assertEqual(params['mu_minus'], 1.0)
        self.assertEqual(params['Wmax'], 100.0)
        self.assertEqual(params['Kplus'], 0.0)
        self.assertEqual(params['synapse_model'], 'stdp_synapse')
        self.assertAlmostEqual(syn.t_lastspike, 0.0, delta=1e-12)

    def test_parameter_validation_matches_nest_semantics(self):
        with self.assertRaisesRegex(ValueError, 'Weight and Wmax must have same sign'):
            stdp_synapse(weight=-1.0, Wmax=100.0)
        with self.assertRaisesRegex(ValueError, 'Kplus must be non-negative'):
            stdp_synapse(Kplus=-1e-3)

        syn = stdp_synapse(weight=2.0, Wmax=5.0)
        with self.assertRaisesRegex(ValueError, 'Weight and Wmax must have same sign'):
            syn.set(weight=-1.0)
        with self.assertRaisesRegex(ValueError, 'Kplus must be non-negative'):
            syn.set(Kplus=-1.0)
        with self.assertRaisesRegex(ValueError, 'post_spike must be an integer spike count'):
            with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
                syn.update(pre_spike=0.0, post_spike=0.25)


class TestSTDPSynapseOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_weight_update_order_matches_independent_reference(self):
        dt_ms = 0.1
        delay_ms = 0.3
        sim_duration_ms = 10.0

        params = dict(
            tau_plus=16.8,
            tau_minus=33.7,
            lambda_=0.01,
            alpha=0.85,
            mu_plus=0.0,
            mu_minus=0.0,
            weight=0.5,
            Wmax=15.0,
        )

        dftype = brainstate.environ.dftype()
        pre_spikes = np.asarray([1.0, 2.5, 3.0, 5.2, 7.4], dtype=dftype)
        post_spikes = np.asarray([0.7, 1.6, 2.2, 2.9, 4.7, 7.1], dtype=dftype)

        send_t_ref, payload_ref = _run_reference_weight_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            sim_duration_ms=sim_duration_ms,
            dt_ms=dt_ms,
            delay_ms=delay_ms,
            **params,
        )
        send_t_bp, payload_bp, labels = _run_bp_weight_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            sim_duration_ms=sim_duration_ms,
            dt_ms=dt_ms,
            delay_ms=delay_ms,
            **params,
        )

        np.testing.assert_allclose(send_t_bp, send_t_ref, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(payload_bp, payload_ref, atol=1e-12, rtol=0.0)
        self.assertTrue(all(label == 'receptor_2' for label in labels))


class TestSTDPSynapseDynamics(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_dynamics_match_nest_reference_logic(self):
        # Mirrors the hardcoded edge-case spike trains from NEST
        # testsuite/pytests/test_stdp_synapse.py.
        dftype = brainstate.environ.dftype()
        pre_spikes = np.asarray(
            [1.0, 5.0, 6.0, 7.0, 9.0, 11.0, 12.0, 13.0, 14.5, 16.1, 21.0, 25.0, 26.0, 27.0, 29.0, 31.0, 32.0, 33.0,
             34.5, 36.1],
            dtype=dftype,
        )
        post_spikes = np.asarray(
            [2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 12.0, 12.2, 14.1, 15.4, 22.0, 23.0, 24.0, 28.0, 29.0, 30.0, 32.0, 33.2,
             35.1, 36.4],
            dtype=dftype,
        )

        dt_ms = 0.1
        sim_duration_ms = 45.0
        common = dict(
            tau_plus=16.8,
            tau_minus=33.7,
            lambda_=0.01,
            alpha=0.85,
            mu_plus=0.0,
            mu_minus=0.0,
            weight=0.5,
            Wmax=15.0,
        )

        for delay_ms in (0.1, 1.0):
            with self.subTest(delay_ms=delay_ms):
                send_t_ref, payload_ref = _run_reference_weight_trace(
                    pre_spikes_ms=pre_spikes,
                    post_spikes_ms=post_spikes,
                    sim_duration_ms=sim_duration_ms,
                    dt_ms=dt_ms,
                    delay_ms=delay_ms,
                    **common,
                )
                send_t_bp, payload_bp, _labels = _run_bp_weight_trace(
                    pre_spikes_ms=pre_spikes,
                    post_spikes_ms=post_spikes,
                    sim_duration_ms=sim_duration_ms,
                    dt_ms=dt_ms,
                    delay_ms=delay_ms,
                    **common,
                )

                np.testing.assert_allclose(send_t_bp, send_t_ref, atol=1e-12, rtol=0.0)
                np.testing.assert_allclose(payload_bp, payload_ref, atol=1e-12, rtol=0.0)

    def test_inhibitory_case_matches_nest_reference_logic(self):
        dftype = brainstate.environ.dftype()
        pre_spikes = np.asarray([5.0, 12.0, 20.0, 29.0, 41.0], dtype=dftype)
        post_spikes = np.asarray([7.0, 9.0, 16.0, 21.0, 35.0], dtype=dftype)

        dt_ms = 0.1
        sim_duration_ms = 60.0
        params = dict(
            tau_plus=20.0,
            tau_minus=30.0,
            lambda_=0.01,
            alpha=1.1,
            mu_plus=1.0,
            mu_minus=1.0,
            weight=-45.0,
            Wmax=-90.0,
        )

        send_t_ref, payload_ref = _run_reference_weight_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            sim_duration_ms=sim_duration_ms,
            dt_ms=dt_ms,
            delay_ms=1.0,
            **params,
        )
        send_t_bp, payload_bp, _labels = _run_bp_weight_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            sim_duration_ms=sim_duration_ms,
            dt_ms=dt_ms,
            delay_ms=1.0,
            **params,
        )

        np.testing.assert_allclose(send_t_bp, send_t_ref, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(payload_bp, payload_ref, atol=1e-12, rtol=0.0)


class TestSTDPSynapseVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest
            if hasattr(nest, 'synapse_models'):
                return 'stdp_synapse' in nest.synapse_models
            return 'stdp_synapse' in nest.Models()
        except Exception:
            return False

    @staticmethod
    def _run_nest_weight_trace(
        *,
        pre_spikes_ms,
        post_spikes_ms,
        dt_ms,
        delay_ms,
        tau_plus,
        tau_minus,
        lambda_,
        alpha,
        mu_plus,
        mu_minus,
        weight,
        Wmax,
    ):
        import nest

        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()
        nest.SetKernelStatus(
            {
                'resolution': float(dt_ms),
                'min_delay': float(dt_ms),
                'max_delay': float(max(delay_ms, dt_ms)),
                'local_num_threads': 1,
            }
        )

        pre = nest.Create('parrot_neuron')
        post = nest.Create('parrot_neuron', params={'tau_minus': float(tau_minus)})
        dftype = brainstate.environ.dftype()
        sg_pre = nest.Create(
            'spike_generator',
            params={'spike_times': list(np.asarray(pre_spikes_ms, dtype=dftype)), 'precise_times': False},
        )
        sg_post = nest.Create(
            'spike_generator',
            params={'spike_times': list(np.asarray(post_spikes_ms, dtype=dftype)), 'precise_times': False},
        )
        wr = nest.Create('weight_recorder')

        model_name = f'stdp_synapse_bpstate_{np.random.randint(1_000_000_000)}'
        nest.CopyModel(
            'stdp_synapse',
            model_name,
            {
                'weight_recorder': wr,
                'weight': float(weight),
                'delay': float(delay_ms),
                'tau_plus': float(tau_plus),
                'lambda': float(lambda_),
                'alpha': float(alpha),
                'mu_plus': float(mu_plus),
                'mu_minus': float(mu_minus),
                'Wmax': float(Wmax),
            },
        )

        # ``spike_generator`` -> ``parrot_neuron`` introduces one static-synapse
        # delay, so the effective pre/post spike times on the STDP connection are
        # shifted by ``dt_ms``.
        nest.Connect(sg_pre, pre, syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': float(dt_ms)})
        nest.Connect(sg_post, post, syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': float(dt_ms)})
        nest.Connect(pre, post, syn_spec={'synapse_model': model_name, 'receptor_type': 1})

        sim_duration_ms = float(max(np.max(pre_spikes_ms), np.max(post_spikes_ms)) + 2.0 * delay_ms + 2.0)
        nest.Simulate(sim_duration_ms)

        events = wr.get('events')
        return (
            np.asarray(events['times'], dtype=dftype),
            np.asarray(events['weights'], dtype=dftype),
        )

    def test_weight_trace_matches_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        params = dict(
            dt_ms=0.1,
            delay_ms=1.0,
            tau_plus=20.0,
            tau_minus=30.0,
            lambda_=0.01,
            alpha=1.1,
            mu_plus=1.0,
            mu_minus=1.0,
            weight=45.0,
            Wmax=90.0,
        )

        dftype = brainstate.environ.dftype()
        pre_spikes = np.asarray([10.0, 30.0, 41.0, 57.0, 90.0], dtype=dftype)
        post_spikes = np.asarray([15.0, 26.0, 50.0, 73.0, 88.0], dtype=dftype)

        nest_times, nest_weights = self._run_nest_weight_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            **params,
        )

        # Effective spike times at the STDP synapse are shifted by dt due to
        # spike_generator -> parrot static-synapse transmission.
        send_t_bp, payload_bp, _labels = _run_bp_weight_trace(
            pre_spikes_ms=pre_spikes + params['dt_ms'],
            post_spikes_ms=post_spikes + params['dt_ms'],
            sim_duration_ms=float(max(np.max(pre_spikes), np.max(post_spikes)) + 5.0),
            **params,
        )

        np.testing.assert_allclose(send_t_bp, nest_times, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(payload_bp, nest_weights, atol=1e-12, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
