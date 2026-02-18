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
import brainunit as u
import jax
import numpy as np

from brainpy.state import stdp_dopamine_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


_STDP_EPS = 1.0e-6


class _MockReceiver:
    def __init__(self):
        self.delta_events = []

    def add_delta_input(self, key, inp, label=None):
        self.delta_events.append((key, inp, label))


def _spike_step_counts_from_times(spike_times_ms, dt_ms):
    spike_times = np.asarray(spike_times_ms, dtype=np.float64).reshape(-1)
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


def _get_post_history_times_ref(post_hist_t, t1, t2):
    t1_lim = t1 + _STDP_EPS
    t2_lim = t2 + _STDP_EPS
    return [float(t_post) for t_post in post_hist_t if t_post >= t1_lim and t_post < t2_lim]


def _get_k_value_ref(post_hist_t, post_hist_kminus, t, tau_minus):
    for idx in range(len(post_hist_t) - 1, -1, -1):
        t_post = post_hist_t[idx]
        if (t - t_post) > _STDP_EPS:
            return post_hist_kminus[idx] * math.exp((t_post - t) / tau_minus)
    return 0.0


def _update_weight_ref(*, weight, c0, n0, minus_dt, tau_c, tau_n, b, Wmin, Wmax):
    taus = (tau_c + tau_n) / (tau_c * tau_n)
    weight = weight - c0 * (n0 / taus * math.expm1(taus * minus_dt) - b * tau_c * math.expm1(minus_dt / tau_c))
    if weight < Wmin:
        weight = Wmin
    if weight > Wmax:
        weight = Wmax
    return weight


def _update_dopamine_ref(*, dopa_spikes, dopa_idx, n, tau_n):
    minus_dt = dopa_spikes[dopa_idx][0] - dopa_spikes[dopa_idx + 1][0]
    dopa_idx += 1
    n = n * math.exp(minus_dt / tau_n) + dopa_spikes[dopa_idx][1] / tau_n
    return n, dopa_idx


def _process_dopa_spikes_ref(*, dopa_spikes, t0, t1, dopa_idx, weight, c, n, tau_c, tau_n, b, Wmin, Wmax):
    if (
        len(dopa_spikes) > dopa_idx + 1
        and (t1 - dopa_spikes[dopa_idx + 1][0] > -1.0 * _STDP_EPS)
    ):
        n0 = n * math.exp((dopa_spikes[dopa_idx][0] - t0) / tau_n)
        weight = _update_weight_ref(
            weight=weight,
            c0=c,
            n0=n0,
            minus_dt=t0 - dopa_spikes[dopa_idx + 1][0],
            tau_c=tau_c,
            tau_n=tau_n,
            b=b,
            Wmin=Wmin,
            Wmax=Wmax,
        )
        n, dopa_idx = _update_dopamine_ref(dopa_spikes=dopa_spikes, dopa_idx=dopa_idx, n=n, tau_n=tau_n)

        while (
            len(dopa_spikes) > dopa_idx + 1
            and (t1 - dopa_spikes[dopa_idx + 1][0] > -1.0 * _STDP_EPS)
        ):
            cd = c * math.exp((t0 - dopa_spikes[dopa_idx][0]) / tau_c)
            weight = _update_weight_ref(
                weight=weight,
                c0=cd,
                n0=n,
                minus_dt=dopa_spikes[dopa_idx][0] - dopa_spikes[dopa_idx + 1][0],
                tau_c=tau_c,
                tau_n=tau_n,
                b=b,
                Wmin=Wmin,
                Wmax=Wmax,
            )
            n, dopa_idx = _update_dopamine_ref(dopa_spikes=dopa_spikes, dopa_idx=dopa_idx, n=n, tau_n=tau_n)

        cd = c * math.exp((t0 - dopa_spikes[dopa_idx][0]) / tau_c)
        weight = _update_weight_ref(
            weight=weight,
            c0=cd,
            n0=n,
            minus_dt=dopa_spikes[dopa_idx][0] - t1,
            tau_c=tau_c,
            tau_n=tau_n,
            b=b,
            Wmin=Wmin,
            Wmax=Wmax,
        )
    else:
        n0 = n * math.exp((dopa_spikes[dopa_idx][0] - t0) / tau_n)
        weight = _update_weight_ref(
            weight=weight,
            c0=c,
            n0=n0,
            minus_dt=t0 - t1,
            tau_c=tau_c,
            tau_n=tau_n,
            b=b,
            Wmin=Wmin,
            Wmax=Wmax,
        )

    c = c * math.exp((t0 - t1) / tau_c)
    return weight, c, n, dopa_idx


def _send_ref(
    *,
    weight,
    Kplus,
    c,
    n,
    t_last_update,
    t_lastspike,
    t_spike,
    multiplicity,
    delay,
    tau_plus,
    tau_minus,
    A_plus,
    A_minus,
    tau_c,
    tau_n,
    b,
    Wmin,
    Wmax,
    post_hist_t,
    post_hist_kminus,
    dopa_spikes,
    dopa_idx,
):
    t0 = t_last_update
    for t_post in _get_post_history_times_ref(post_hist_t, t_last_update - delay, t_spike - delay):
        weight, c, n, dopa_idx = _process_dopa_spikes_ref(
            dopa_spikes=dopa_spikes,
            t0=t0,
            t1=t_post + delay,
            dopa_idx=dopa_idx,
            weight=weight,
            c=c,
            n=n,
            tau_c=tau_c,
            tau_n=tau_n,
            b=b,
            Wmin=Wmin,
            Wmax=Wmax,
        )
        t0 = t_post + delay
        minus_dt = t_last_update - t0
        if (t_spike - t_post) > _STDP_EPS:
            c = c + A_plus * (Kplus * math.exp(minus_dt / tau_plus))

    weight, c, n, dopa_idx = _process_dopa_spikes_ref(
        dopa_spikes=dopa_spikes,
        t0=t0,
        t1=t_spike,
        dopa_idx=dopa_idx,
        weight=weight,
        c=c,
        n=n,
        tau_c=tau_c,
        tau_n=tau_n,
        b=b,
        Wmin=Wmin,
        Wmax=Wmax,
    )
    c = c - A_minus * _get_k_value_ref(post_hist_t, post_hist_kminus, t_spike - delay, tau_minus)

    payload = float(multiplicity) * float(weight)
    Kplus = Kplus * math.exp((t_last_update - t_spike) / tau_plus) + 1.0
    t_last_update = float(t_spike)
    t_lastspike = float(t_spike)

    return weight, Kplus, c, n, t_last_update, t_lastspike, dopa_idx, payload


def _trigger_update_weight_ref(
    *,
    Kplus,
    c,
    n,
    t_last_update,
    t_trig,
    delay,
    A_plus,
    tau_plus,
    tau_c,
    tau_n,
    b,
    Wmin,
    Wmax,
    post_hist_t,
    dopa_spikes,
    dopa_idx,
    weight,
):
    t0 = t_last_update
    for t_post in _get_post_history_times_ref(post_hist_t, t_last_update - delay, t_trig - delay):
        weight, c, n, dopa_idx = _process_dopa_spikes_ref(
            dopa_spikes=dopa_spikes,
            t0=t0,
            t1=t_post + delay,
            dopa_idx=dopa_idx,
            weight=weight,
            c=c,
            n=n,
            tau_c=tau_c,
            tau_n=tau_n,
            b=b,
            Wmin=Wmin,
            Wmax=Wmax,
        )
        t0 = t_post + delay
        minus_dt = t_last_update - t0
        c = c + A_plus * (Kplus * math.exp(minus_dt / tau_plus))

    weight, c, n, dopa_idx = _process_dopa_spikes_ref(
        dopa_spikes=dopa_spikes,
        t0=t0,
        t1=t_trig,
        dopa_idx=dopa_idx,
        weight=weight,
        c=c,
        n=n,
        tau_c=tau_c,
        tau_n=tau_n,
        b=b,
        Wmin=Wmin,
        Wmax=Wmax,
    )
    n = n * math.exp((dopa_spikes[dopa_idx][0] - t_trig) / tau_n)
    Kplus = Kplus * math.exp((t_last_update - t_trig) / tau_plus)
    t_last_update = float(t_trig)
    dopa_idx = 0
    dopa_spikes = [(float(t_trig), 0.0)]
    return weight, Kplus, c, n, t_last_update, dopa_spikes, dopa_idx


def _run_reference_trace(
    *,
    pre_spikes_ms,
    post_spikes_ms,
    dopa_spikes_ms,
    sim_duration_ms,
    dt_ms,
    delay_ms,
    tau_plus,
    tau_minus,
    A_plus,
    A_minus,
    tau_c,
    tau_n,
    b,
    Wmin,
    Wmax,
    weight,
):
    pre_counts = _spike_step_counts_from_times(pre_spikes_ms, dt_ms)
    post_counts = _spike_step_counts_from_times(post_spikes_ms, dt_ms)
    dopa_counts = _spike_step_counts_from_times(dopa_spikes_ms, dt_ms)
    sim_steps = 1 + int(np.ceil(sim_duration_ms / dt_ms))

    Kplus = 0.0
    c = 0.0
    n = 0.0
    t_last_update = 0.0
    t_lastspike = 0.0
    current_weight = float(weight)

    dopa_spikes = [(0.0, 0.0)]
    dopa_idx = 0

    post_kminus = 0.0
    last_post = -1.0
    post_hist_t = []
    post_hist_kminus = []

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

        dopa_count = dopa_counts.get(step, 0)
        if dopa_count > 0:
            if abs(dopa_spikes[-1][0] - t_spike) <= _STDP_EPS:
                dopa_spikes[-1] = (t_spike, dopa_spikes[-1][1] + float(dopa_count))
            else:
                dopa_spikes.append((t_spike, float(dopa_count)))

        pre_count = pre_counts.get(step, 0)
        if pre_count > 0:
            current_weight, Kplus, c, n, t_last_update, t_lastspike, dopa_idx, payload = _send_ref(
                weight=current_weight,
                Kplus=Kplus,
                c=c,
                n=n,
                t_last_update=t_last_update,
                t_lastspike=t_lastspike,
                t_spike=t_spike,
                multiplicity=pre_count,
                delay=delay_ms,
                tau_plus=tau_plus,
                tau_minus=tau_minus,
                A_plus=A_plus,
                A_minus=A_minus,
                tau_c=tau_c,
                tau_n=tau_n,
                b=b,
                Wmin=Wmin,
                Wmax=Wmax,
                post_hist_t=post_hist_t,
                post_hist_kminus=post_hist_kminus,
                dopa_spikes=dopa_spikes,
                dopa_idx=dopa_idx,
            )
            send_times.append(t_spike)
            payloads.append(payload)

        current_weight, Kplus, c, n, t_last_update, dopa_spikes, dopa_idx = _trigger_update_weight_ref(
            Kplus=Kplus,
            c=c,
            n=n,
            t_last_update=t_last_update,
            t_trig=t_spike,
            delay=delay_ms,
            A_plus=A_plus,
            tau_plus=tau_plus,
            tau_c=tau_c,
            tau_n=tau_n,
            b=b,
            Wmin=Wmin,
            Wmax=Wmax,
            post_hist_t=post_hist_t,
            dopa_spikes=dopa_spikes,
            dopa_idx=dopa_idx,
            weight=current_weight,
        )

    return (
        np.asarray(send_times, dtype=np.float64),
        np.asarray(payloads, dtype=np.float64),
        dict(weight=float(current_weight), Kplus=float(Kplus), c=float(c), n=float(n)),
    )


def _run_bp_trace(
    *,
    pre_spikes_ms,
    post_spikes_ms,
    dopa_spikes_ms,
    sim_duration_ms,
    dt_ms,
    delay_ms,
    tau_plus,
    tau_minus,
    A_plus,
    A_minus,
    tau_c,
    tau_n,
    b,
    Wmin,
    Wmax,
    weight,
):
    dt = dt_ms * u.ms
    pre_counts = _spike_step_counts_from_times(pre_spikes_ms, dt_ms)
    post_counts = _spike_step_counts_from_times(post_spikes_ms, dt_ms)
    dopa_counts = _spike_step_counts_from_times(dopa_spikes_ms, dt_ms)
    sim_steps = 1 + int(np.ceil(sim_duration_ms / dt_ms))

    recv = _MockReceiver()
    with brainstate.environ.context(dt=dt):
        syn = stdp_dopamine_synapse(
            delay=delay_ms * u.ms,
            tau_plus=tau_plus * u.ms,
            tau_minus=tau_minus * u.ms,
            A_plus=A_plus,
            A_minus=A_minus,
            tau_c=tau_c * u.ms,
            tau_n=tau_n * u.ms,
            b=b,
            Wmin=Wmin,
            Wmax=Wmax,
            weight=weight,
            receptor_type=2,
            post=recv,
            volume_transmitter=object(),
        )
        syn.init_state()

        send_times = []
        for step in range(sim_steps):
            pre_count = pre_counts.get(step, 0)
            post_count = post_counts.get(step, 0)
            dopa_count = dopa_counts.get(step, 0)
            with brainstate.environ.context(t=step * dt):
                syn.update(
                    pre_spike=float(pre_count),
                    post_spike=float(post_count),
                    dopa_spike=float(dopa_count),
                )
            if pre_count > 0:
                send_times.append((step + 1) * dt_ms)

    payloads = np.asarray(
        [
            float(np.asarray(u.math.asarray(value), dtype=np.float64).reshape(()))
            for _key, value, _label in recv.delta_events
        ],
        dtype=np.float64,
    )
    labels = [label for _key, _value, label in recv.delta_events]
    return (
        np.asarray(send_times, dtype=np.float64),
        payloads,
        labels,
        dict(weight=float(syn.weight), Kplus=float(syn.Kplus), c=float(syn.c), n=float(syn.n)),
    )


class TestSTDPDopamineSynapseParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = stdp_dopamine_synapse(volume_transmitter=object())
            syn.init_state()
            syn.update(pre_spike=0.0, post_spike=0.0, dopa_spike=0.0)
            params = syn.get()

        self.assertEqual(params['weight'], 1.0)
        self.assertEqual(params['delay'], 1.0)
        self.assertEqual(params['delay_steps'], 1)
        self.assertEqual(params['receptor_type'], 0)
        self.assertEqual(params['event_type'], 'spike')
        self.assertEqual(params['A_plus'], 1.0)
        self.assertEqual(params['A_minus'], 1.5)
        self.assertEqual(params['tau_plus'], 20.0)
        self.assertEqual(params['tau_minus'], 20.0)
        self.assertEqual(params['tau_c'], 1000.0)
        self.assertEqual(params['tau_n'], 200.0)
        self.assertEqual(params['b'], 0.0)
        self.assertEqual(params['Wmin'], 0.0)
        self.assertEqual(params['Wmax'], 200.0)
        self.assertEqual(params['Kplus'], 0.0)
        self.assertEqual(params['c'], 0.0)
        self.assertEqual(params['n'], 0.0)
        self.assertEqual(params['synapse_model'], 'stdp_dopamine_synapse')

    def test_common_properties_rejected_in_connect_syn_spec(self):
        syn = stdp_dopamine_synapse(volume_transmitter=object())
        for key in ('vt', 'volume_transmitter', 'A_minus', 'A_plus', 'Wmax', 'Wmin', 'b', 'tau_c', 'tau_n', 'tau_plus'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f'{key} cannot be specified'):
                    syn.check_synapse_params({'synapse_model': 'stdp_dopamine_synapse', key: 1.0})

        syn.check_synapse_params({'synapse_model': 'stdp_dopamine_synapse', 'weight': 3.0})
        syn.check_synapse_params({'synapse_model': 'stdp_dopamine_synapse', 'c': 2.0})
        syn.check_synapse_params({'synapse_model': 'stdp_dopamine_synapse', 'n': -1.0})

    def test_parameter_validation_matches_nest_semantics(self):
        with self.assertRaisesRegex(ValueError, 'Kplus must be non-negative'):
            stdp_dopamine_synapse(volume_transmitter=object(), Kplus=-1e-3)

        syn = stdp_dopamine_synapse(volume_transmitter=object())
        syn.set(c=-3.0, n=-2.5)
        params = syn.get()
        self.assertEqual(params['c'], -3.0)
        self.assertEqual(params['n'], -2.5)

        with self.assertRaisesRegex(ValueError, 'tau_c must be > 0'):
            syn.set(tau_c=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, 'post_spike must be an integer spike count'):
            with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
                syn.update(pre_spike=0.0, post_spike=0.2, dopa_spike=0.0)

    def test_missing_volume_transmitter_raises(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = stdp_dopamine_synapse(volume_transmitter=None)
            syn.init_state()
            with self.assertRaisesRegex(ValueError, 'No volume transmitter'):
                syn.update(pre_spike=0.0, post_spike=0.0, dopa_spike=0.0)


class TestSTDPDopamineSynapseDynamics(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_weight_update_order_matches_independent_reference(self):
        params = dict(
            dt_ms=0.1,
            delay_ms=0.3,
            tau_plus=20.0,
            tau_minus=33.7,
            A_plus=0.05,
            A_minus=0.07,
            tau_c=1000.0,
            tau_n=250.0,
            b=0.12,
            Wmin=0.0,
            Wmax=200.0,
            weight=10.0,
        )
        pre_spikes = np.asarray([1.0, 2.5, 4.0, 7.2, 9.4], dtype=np.float64)
        post_spikes = np.asarray([0.8, 1.6, 3.1, 5.3, 8.5], dtype=np.float64)
        dopa_spikes = np.asarray([0.9, 2.2, 2.7, 5.0, 8.0, 9.1], dtype=np.float64)

        sim_duration_ms = 12.0
        send_t_ref, payload_ref, state_ref = _run_reference_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            dopa_spikes_ms=dopa_spikes,
            sim_duration_ms=sim_duration_ms,
            **params,
        )
        send_t_bp, payload_bp, labels, state_bp = _run_bp_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            dopa_spikes_ms=dopa_spikes,
            sim_duration_ms=sim_duration_ms,
            **params,
        )

        np.testing.assert_allclose(send_t_bp, send_t_ref, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(payload_bp, payload_ref, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(state_bp['weight'], state_ref['weight'], atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(state_bp['Kplus'], state_ref['Kplus'], atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(state_bp['c'], state_ref['c'], atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(state_bp['n'], state_ref['n'], atol=1e-12, rtol=0.0)
        self.assertTrue(all(label == 'receptor_2' for label in labels))

    def test_weight_constant_without_presynaptic_spikes(self):
        params = dict(
            dt_ms=0.1,
            delay_ms=1.0,
            tau_plus=20.0,
            tau_minus=20.0,
            A_plus=0.05,
            A_minus=0.05,
            tau_c=1000.0,
            tau_n=200.0,
            b=0.0,
            Wmin=0.0,
            Wmax=200.0,
            weight=1.0,
        )
        pre_spikes = np.asarray([], dtype=np.float64)
        post_spikes = np.asarray([0.5, 1.1, 3.4], dtype=np.float64)
        dopa_spikes = np.asarray([1.4, 2.3, 4.6], dtype=np.float64)

        _send_t_ref, _payload_ref, state_ref = _run_reference_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            dopa_spikes_ms=dopa_spikes,
            sim_duration_ms=10.0,
            **params,
        )
        send_t_bp, payload_bp, _labels, state_bp = _run_bp_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            dopa_spikes_ms=dopa_spikes,
            sim_duration_ms=10.0,
            **params,
        )

        self.assertEqual(send_t_bp.size, 0)
        self.assertEqual(payload_bp.size, 0)
        self.assertAlmostEqual(state_bp['weight'], 1.0, delta=1e-13)
        self.assertAlmostEqual(state_ref['weight'], 1.0, delta=1e-13)


class TestSTDPDopamineSynapseVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest

            if hasattr(nest, 'synapse_models'):
                return 'stdp_dopamine_synapse' in nest.synapse_models
            return 'stdp_dopamine_synapse' in nest.Models()
        except Exception:
            return False

    @staticmethod
    def _run_nest_weight_trace(
        *,
        pre_spikes_ms,
        post_spikes_ms,
        dopa_spikes_ms,
        dt_ms,
        delay_ms,
        tau_plus,
        tau_minus,
        A_plus,
        A_minus,
        tau_c,
        tau_n,
        b,
        Wmin,
        Wmax,
        weight,
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
        dopa = nest.Create('parrot_neuron')
        vt = nest.Create('volume_transmitter')
        sg_pre = nest.Create(
            'spike_generator',
            params={'spike_times': list(np.asarray(pre_spikes_ms, dtype=np.float64)), 'precise_times': False},
        )
        sg_post = nest.Create(
            'spike_generator',
            params={'spike_times': list(np.asarray(post_spikes_ms, dtype=np.float64)), 'precise_times': False},
        )
        sg_dopa = nest.Create(
            'spike_generator',
            params={'spike_times': list(np.asarray(dopa_spikes_ms, dtype=np.float64)), 'precise_times': False},
        )
        wr = nest.Create('weight_recorder')

        model_name = f'stdp_dopamine_synapse_bpstate_{np.random.randint(1_000_000_000)}'
        nest.CopyModel(
            'stdp_dopamine_synapse',
            model_name,
            {
                'weight_recorder': wr,
                'volume_transmitter': vt,
                'weight': float(weight),
                'delay': float(delay_ms),
                'tau_plus': float(tau_plus),
                'A_plus': float(A_plus),
                'A_minus': float(A_minus),
                'tau_c': float(tau_c),
                'tau_n': float(tau_n),
                'b': float(b),
                'Wmin': float(Wmin),
                'Wmax': float(Wmax),
            },
        )

        nest.Connect(sg_pre, pre, syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': float(dt_ms)})
        nest.Connect(sg_post, post, syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': float(dt_ms)})
        nest.Connect(sg_dopa, dopa, syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': float(dt_ms)})
        nest.Connect(dopa, vt, syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': float(dt_ms)})
        nest.Connect(pre, post, syn_spec={'synapse_model': model_name, 'receptor_type': 1})

        sim_duration_ms = float(
            max(np.max(pre_spikes_ms), np.max(post_spikes_ms), np.max(dopa_spikes_ms))
            + 2.0 * delay_ms
            + 2.0
        )
        nest.Simulate(sim_duration_ms)

        events = wr.get('events')
        return (
            np.asarray(events['times'], dtype=np.float64),
            np.asarray(events['weights'], dtype=np.float64),
        )

    def test_weight_trace_matches_nest(self):
        if not self._is_nest_available():
            self.skipTest('NEST simulator not available')

        params = dict(
            dt_ms=0.1,
            delay_ms=1.0,
            tau_plus=10.0,
            tau_minus=10.0,
            A_plus=0.05,
            A_minus=0.05,
            tau_c=1.0,
            tau_n=100.0,
            b=45.45,
            Wmin=0.0,
            Wmax=1000.0,
            weight=500.0,
        )

        pre_spikes = np.asarray([10.0, 30.0, 41.0, 57.0, 90.0], dtype=np.float64)
        post_spikes = np.asarray([15.0, 26.0, 50.0, 73.0, 88.0], dtype=np.float64)
        dopa_spikes = np.asarray([12.0, 32.0, 43.0, 67.0, 92.0], dtype=np.float64)

        nest_times, nest_weights = self._run_nest_weight_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            dopa_spikes_ms=dopa_spikes,
            **params,
        )

        send_t_bp, payload_bp, _labels, _state_bp = _run_bp_trace(
            pre_spikes_ms=pre_spikes + params['dt_ms'],
            post_spikes_ms=post_spikes + params['dt_ms'],
            dopa_spikes_ms=dopa_spikes + 2.0 * params['dt_ms'],
            sim_duration_ms=float(max(np.max(pre_spikes), np.max(post_spikes), np.max(dopa_spikes)) + 5.0),
            **params,
        )

        np.testing.assert_allclose(send_t_bp, nest_times, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(payload_bp, nest_weights, atol=5e-11, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
