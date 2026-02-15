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

from brainpy.state import stdp_facetshw_synapse_hom

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


_STDP_EPS = 1.0e-6
_DEFAULT_LUT_0 = np.asarray([2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15], dtype=np.int64)
_DEFAULT_LUT_1 = np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13], dtype=np.int64)
_DEFAULT_LUT_2 = np.arange(16, dtype=np.int64)
_DEFAULT_CONFIG_0 = np.asarray([0, 0, 1, 0], dtype=np.int64)
_DEFAULT_CONFIG_1 = np.asarray([0, 1, 0, 0], dtype=np.int64)
_DEFAULT_RESET_PATTERN = np.asarray([1, 1, 1, 1, 1, 1], dtype=np.int64)


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


def _round_half_away_from_zero(value):
    if value >= 0.0:
        return int(math.floor(float(value) + 0.5))
    return int(math.ceil(float(value) - 0.5))


def _calc_readout_cycle_duration_ref(no_synapses, synapses_per_driver, driver_readout_time):
    return int((float(no_synapses) - 1.0) / float(synapses_per_driver) + 1.0) * float(driver_readout_time)


def _eval_function_ref(a_causal, a_acausal, a_thresh_th, a_thresh_tl, configbit):
    return (
        (a_thresh_tl + configbit[2] * a_causal + configbit[1] * a_acausal)
        / (1 + configbit[2] + configbit[1])
        > (a_thresh_th + configbit[0] * a_causal + configbit[3] * a_acausal)
        / (1 + configbit[0] + configbit[3])
    )


def _history_window_ref(post_hist_t, t1, t2):
    t1_lim = t1 + _STDP_EPS
    t2_lim = t2 + _STDP_EPS
    return [t_post for t_post in post_hist_t if t_post >= t1_lim and t_post < t2_lim]


def _facetshw_send_ref(state, *, multiplicity, t_spike, delay, post_hist_t):
    if not state['init_flag']:
        state['synapse_id'] = int(state['no_synapses'])
        state['no_synapses'] += 1
        state['readout_cycle_duration'] = _calc_readout_cycle_duration_ref(
            state['no_synapses'],
            state['synapses_per_driver'],
            state['driver_readout_time'],
        )
        state['next_readout_time'] = int(state['synapse_id'] / state['synapses_per_driver']) * state['driver_readout_time']
        state['init_flag'] = True

    if t_spike > state['next_readout_time']:
        discrete_weight = _round_half_away_from_zero(state['weight'] / state['weight_per_lut_entry'])

        eval_0 = _eval_function_ref(
            state['a_causal'],
            state['a_acausal'],
            state['a_thresh_th'],
            state['a_thresh_tl'],
            state['configbit_0'],
        )
        eval_1 = _eval_function_ref(
            state['a_causal'],
            state['a_acausal'],
            state['a_thresh_th'],
            state['a_thresh_tl'],
            state['configbit_1'],
        )

        if eval_0 and not eval_1:
            discrete_weight = state['lookuptable_0'][discrete_weight]
            if state['reset_pattern'][0]:
                state['a_causal'] = 0.0
            if state['reset_pattern'][1]:
                state['a_acausal'] = 0.0
        elif (not eval_0) and eval_1:
            discrete_weight = state['lookuptable_1'][discrete_weight]
            if state['reset_pattern'][2]:
                state['a_causal'] = 0.0
            if state['reset_pattern'][3]:
                state['a_acausal'] = 0.0
        elif eval_0 and eval_1:
            discrete_weight = state['lookuptable_2'][discrete_weight]
            if state['reset_pattern'][4]:
                state['a_causal'] = 0.0
            if state['reset_pattern'][5]:
                state['a_acausal'] = 0.0

        while t_spike > state['next_readout_time']:
            state['next_readout_time'] += state['readout_cycle_duration']

        state['weight'] = float(discrete_weight * state['weight_per_lut_entry'])

    hist = _history_window_ref(post_hist_t, state['t_lastspike'] - delay, t_spike - delay)
    if hist:
        minus_dt_causal = state['t_lastspike'] - (hist[0] + delay)
        assert minus_dt_causal < (-1.0 * _STDP_EPS)
        state['a_causal'] += math.exp(minus_dt_causal / state['tau_plus'])

        minus_dt_acausal = (hist[-1] + delay) - t_spike
        state['a_acausal'] += math.exp(minus_dt_acausal / state['tau_minus_stdp'])

    payload = float(multiplicity) * float(state['weight'])
    state['t_lastspike'] = float(t_spike)
    return payload


def _run_bp_trace(
    *,
    pre_spikes_ms,
    post_spikes_ms,
    sim_duration_ms,
    dt_ms,
    delay_ms,
    params,
):
    dt = dt_ms * u.ms
    pre_counts = _spike_step_counts_from_times(pre_spikes_ms, dt_ms)
    post_counts = _spike_step_counts_from_times(post_spikes_ms, dt_ms)
    sim_steps = 1 + int(np.ceil(sim_duration_ms / dt_ms))
    recv = _MockReceiver()

    with brainstate.environ.context(dt=dt):
        syn = stdp_facetshw_synapse_hom(
            delay=delay_ms * u.ms,
            receptor_type=2,
            post=recv,
            **params,
        )
        syn.init_state()

        send_times = []
        payloads = []
        weight_trace = []
        a_causal_trace = []
        a_acausal_trace = []

        for step in range(sim_steps):
            pre_count = pre_counts.get(step, 0)
            post_count = post_counts.get(step, 0)
            with brainstate.environ.context(t=step * dt):
                syn.update(pre_spike=float(pre_count), post_spike=float(post_count))
            if pre_count > 0:
                send_times.append((step + 1) * dt_ms)
                payloads.append(float(pre_count) * float(syn.weight))
                weight_trace.append(float(syn.weight))
                a_causal_trace.append(float(syn.a_causal))
                a_acausal_trace.append(float(syn.a_acausal))

    return (
        np.asarray(send_times, dtype=np.float64),
        np.asarray(payloads, dtype=np.float64),
        np.asarray(weight_trace, dtype=np.float64),
        np.asarray(a_causal_trace, dtype=np.float64),
        np.asarray(a_acausal_trace, dtype=np.float64),
    )


def _run_reference_trace(
    *,
    pre_spikes_ms,
    post_spikes_ms,
    sim_duration_ms,
    dt_ms,
    delay_ms,
    params,
):
    pre_counts = _spike_step_counts_from_times(pre_spikes_ms, dt_ms)
    post_counts = _spike_step_counts_from_times(post_spikes_ms, dt_ms)
    sim_steps = 1 + int(np.ceil(sim_duration_ms / dt_ms))

    state = dict(
        weight=float(params.get('weight', 1.0)),
        tau_plus=float(params.get('tau_plus', 20.0)),
        tau_minus_stdp=float(params.get('tau_minus_stdp', 20.0)),
        Wmax=float(params.get('Wmax', 100.0)),
        no_synapses=int(params.get('no_synapses', 0)),
        synapses_per_driver=int(params.get('synapses_per_driver', 50)),
        driver_readout_time=float(params.get('driver_readout_time', 15.0)),
        lookuptable_0=list(params.get('lookuptable_0', _DEFAULT_LUT_0)),
        lookuptable_1=list(params.get('lookuptable_1', _DEFAULT_LUT_1)),
        lookuptable_2=list(params.get('lookuptable_2', _DEFAULT_LUT_2)),
        configbit_0=list(params.get('configbit_0', _DEFAULT_CONFIG_0)),
        configbit_1=list(params.get('configbit_1', _DEFAULT_CONFIG_1)),
        reset_pattern=list(params.get('reset_pattern', _DEFAULT_RESET_PATTERN)),
        a_causal=float(params.get('a_causal', 0.0)),
        a_acausal=float(params.get('a_acausal', 0.0)),
        a_thresh_th=float(params.get('a_thresh_th', 21.835)),
        a_thresh_tl=float(params.get('a_thresh_tl', 21.835)),
        init_flag=bool(params.get('init_flag', False)),
        synapse_id=int(params.get('synapse_id', 0)),
        next_readout_time=float(params.get('next_readout_time', 0.0)),
        t_lastspike=0.0,
    )
    if 'weight_per_lut_entry' in params:
        state['weight_per_lut_entry'] = float(params['weight_per_lut_entry'])
    else:
        state['weight_per_lut_entry'] = state['Wmax'] / (len(state['lookuptable_0']) - 1)
    if 'readout_cycle_duration' in params:
        state['readout_cycle_duration'] = float(params['readout_cycle_duration'])
    else:
        state['readout_cycle_duration'] = _calc_readout_cycle_duration_ref(
            state['no_synapses'],
            state['synapses_per_driver'],
            state['driver_readout_time'],
        )

    post_hist_t = []
    send_times = []
    payloads = []
    weight_trace = []
    a_causal_trace = []
    a_acausal_trace = []

    for step in range(sim_steps):
        t_spike = (step + 1) * dt_ms

        post_count = post_counts.get(step, 0)
        for _ in range(post_count):
            post_hist_t.append(float(t_spike))

        pre_count = pre_counts.get(step, 0)
        if pre_count > 0:
            payload = _facetshw_send_ref(
                state,
                multiplicity=pre_count,
                t_spike=t_spike,
                delay=delay_ms,
                post_hist_t=post_hist_t,
            )
            send_times.append(float(t_spike))
            payloads.append(float(payload))
            weight_trace.append(float(state['weight']))
            a_causal_trace.append(float(state['a_causal']))
            a_acausal_trace.append(float(state['a_acausal']))

    return (
        np.asarray(send_times, dtype=np.float64),
        np.asarray(payloads, dtype=np.float64),
        np.asarray(weight_trace, dtype=np.float64),
        np.asarray(a_causal_trace, dtype=np.float64),
        np.asarray(a_acausal_trace, dtype=np.float64),
    )


class TestSTDPFACETSHWSynapseHomParameters(unittest.TestCase):
    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = stdp_facetshw_synapse_hom()
            syn.init_state()
            syn.update(pre_spike=0.0, post_spike=0.0)
            params = syn.get()

        self.assertEqual(params['weight'], 1.0)
        self.assertEqual(params['delay'], 1.0)
        self.assertEqual(params['delay_steps'], 1)
        self.assertEqual(params['receptor_type'], 0)
        self.assertEqual(params['event_type'], 'spike')
        self.assertEqual(params['tau_plus'], 20.0)
        self.assertEqual(params['tau_minus_stdp'], 20.0)
        self.assertEqual(params['Wmax'], 100.0)
        self.assertAlmostEqual(params['weight_per_lut_entry'], 100.0 / 15.0, delta=1e-12)
        self.assertEqual(params['no_synapses'], 0)
        self.assertEqual(params['synapses_per_driver'], 50)
        self.assertEqual(params['driver_readout_time'], 15.0)
        self.assertEqual(params['readout_cycle_duration'], 0.0)
        self.assertEqual(params['lookuptable_0'], list(_DEFAULT_LUT_0))
        self.assertEqual(params['lookuptable_1'], list(_DEFAULT_LUT_1))
        self.assertEqual(params['lookuptable_2'], list(_DEFAULT_LUT_2))
        self.assertEqual(params['configbit_0'], list(_DEFAULT_CONFIG_0))
        self.assertEqual(params['configbit_1'], list(_DEFAULT_CONFIG_1))
        self.assertEqual(params['reset_pattern'], list(_DEFAULT_RESET_PATTERN))
        self.assertEqual(params['a_causal'], 0.0)
        self.assertEqual(params['a_acausal'], 0.0)
        self.assertEqual(params['a_thresh_th'], 21.835)
        self.assertEqual(params['a_thresh_tl'], 21.835)
        self.assertEqual(params['init_flag'], False)
        self.assertEqual(params['synapse_id'], 0)
        self.assertEqual(params['next_readout_time'], 0.0)
        self.assertEqual(params['synapse_model'], 'stdp_facetshw_synapse_hom')

    def test_common_properties_rejected_in_connect_syn_spec(self):
        syn = stdp_facetshw_synapse_hom()
        for key in (
            'tau_plus',
            'tau_minus_stdp',
            'Wmax',
            'weight_per_lut_entry',
            'no_synapses',
            'synapses_per_driver',
            'driver_readout_time',
            'readout_cycle_duration',
            'lookuptable_0',
            'lookuptable_1',
            'lookuptable_2',
            'configbit_0',
            'configbit_1',
            'reset_pattern',
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f'{key} cannot be specified'):
                    syn.check_synapse_params({'synapse_model': 'stdp_facetshw_synapse_hom', key: 1.0})

        syn.check_synapse_params({'synapse_model': 'stdp_facetshw_synapse_hom', 'weight': 3.0})
        syn.check_synapse_params({'synapse_model': 'stdp_facetshw_synapse_hom', 'delay': 2.0})
        syn.check_synapse_params({'synapse_model': 'stdp_facetshw_synapse_hom', 'a_causal': 1.0})
        syn.check_synapse_params({'synapse_model': 'stdp_facetshw_synapse_hom', 'a_thresh_th': 2.0})

    def test_lut_and_vector_validation(self):
        with self.assertRaisesRegex(ValueError, 'lookuptable_0 must contain exactly 16 entries'):
            stdp_facetshw_synapse_hom(lookuptable_0=[0] * 15)
        with self.assertRaisesRegex(ValueError, 'lookuptable_0 entries must be in \\[0,15\\]'):
            stdp_facetshw_synapse_hom(lookuptable_0=[16] + [0] * 15)
        with self.assertRaisesRegex(ValueError, 'configbit_0 must contain exactly 4 entries'):
            stdp_facetshw_synapse_hom(configbit_0=[0, 1, 0])
        with self.assertRaisesRegex(ValueError, 'reset_pattern must contain exactly 6 entries'):
            stdp_facetshw_synapse_hom(reset_pattern=[1, 1, 1, 1, 1])

    def test_status_assignment_matches_nest_loose_weight_semantics(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = stdp_facetshw_synapse_hom(weight=-3.0, a_causal=-1.0)
            syn.init_state()
            syn.update(pre_spike=0.0, post_spike=0.0)
            params = syn.get()
        self.assertEqual(params['weight'], -3.0)
        self.assertEqual(params['a_causal'], -1.0)

        syn.set(weight=2.0, Wmax=90.0, a_acausal=7.0, synapse_id=4, init_flag=1)
        syn.set_weight(-1.5)
        params = syn.get()
        self.assertEqual(params['weight'], -1.5)
        self.assertEqual(params['Wmax'], 90.0)
        self.assertEqual(params['a_acausal'], 7.0)
        self.assertEqual(params['synapse_id'], 4)
        self.assertTrue(params['init_flag'])


class TestSTDPFACETSHWSynapseHomOrdering(unittest.TestCase):
    def test_weight_update_order_matches_independent_reference(self):
        dt_ms = 0.1
        delay_ms = 0.3
        sim_duration_ms = 40.0

        params = dict(
            tau_plus=12.0,
            tau_minus_stdp=18.0,
            Wmax=90.0,
            no_synapses=0,
            synapses_per_driver=3,
            driver_readout_time=2.0,
            lookuptable_0=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15],
            lookuptable_1=[0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            lookuptable_2=list(range(16)),
            configbit_0=[0, 0, 1, 0],
            configbit_1=[0, 1, 0, 0],
            reset_pattern=[1, 1, 1, 1, 1, 1],
            a_thresh_th=1.0,
            a_thresh_tl=1.0,
            weight=0.0,
        )

        pre_spikes = np.asarray([2.0, 4.5, 6.2, 9.0, 14.2, 20.6, 31.0], dtype=np.float64)
        post_spikes = np.asarray([1.2, 3.7, 5.1, 8.8, 12.0, 19.2, 28.0], dtype=np.float64)

        ref = _run_reference_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            sim_duration_ms=sim_duration_ms,
            dt_ms=dt_ms,
            delay_ms=delay_ms,
            params=params,
        )
        got = _run_bp_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            sim_duration_ms=sim_duration_ms,
            dt_ms=dt_ms,
            delay_ms=delay_ms,
            params=params,
        )

        for got_arr, ref_arr in zip(got, ref):
            np.testing.assert_allclose(got_arr, ref_arr, atol=1e-12, rtol=0.0)


class TestSTDPFACETSHWSynapseHomDynamics(unittest.TestCase):
    def test_dynamics_match_nest_reference_logic(self):
        # Mirrors NEST testsuite/pytests/test_facetshw_stdp.py with shorter
        # runtime while keeping the 36-pair LUT update structure.
        Wmax = 100.0
        tau = 20.0
        delay = 5.0
        time_between_pairs = 100.0
        start_weight_index = 0
        start_weight = float(start_weight_index) / 15.0 * Wmax

        spikes_in = np.arange(10.0, 7600.0, time_between_pairs, dtype=np.float64)
        pre_spikes = spikes_in
        post_spikes = spikes_in + delay
        sim_duration_ms = float(np.max(post_spikes) + 10.0)

        params = dict(
            tau_plus=tau,
            tau_minus_stdp=tau,
            Wmax=Wmax,
            synapses_per_driver=50,
            driver_readout_time=15.0,
            lookuptable_0=_DEFAULT_LUT_0.tolist(),
            lookuptable_1=_DEFAULT_LUT_1.tolist(),
            lookuptable_2=_DEFAULT_LUT_2.tolist(),
            configbit_0=_DEFAULT_CONFIG_0.tolist(),
            configbit_1=_DEFAULT_CONFIG_1.tolist(),
            reset_pattern=_DEFAULT_RESET_PATTERN.tolist(),
            a_thresh_th=21.835,
            a_thresh_tl=21.835,
            weight=start_weight,
        )

        _send_t, _payloads, weights, a_causal, a_acausal = _run_bp_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            sim_duration_ms=sim_duration_ms,
            dt_ms=0.1,
            delay_ms=delay,
            params=params,
        )

        # NEST test records after an initial warm-up period.
        weights_eval = weights[1:]
        a_causal_eval = a_causal[1:]
        a_acausal_eval = a_acausal[1:]

        weight_mod36_pre = weights_eval[35::36]
        weight_mod36 = weights_eval[::36]

        weight_index = int(start_weight_index)
        for value in weight_mod36_pre:
            self.assertTrue(np.allclose(value, (weight_index / 15.0) * Wmax, atol=1e-10))
            weight_index = int(_DEFAULT_LUT_0[weight_index])

        weight_index = int(start_weight_index)
        expected_causal = math.exp(-2.0 * delay / tau)
        for i in range(len(weight_mod36)):
            self.assertTrue(np.allclose(weight_mod36[i], (weight_index / 15.0) * Wmax, atol=1e-10))
            self.assertTrue(np.allclose(a_causal_eval[i * 36], expected_causal, atol=1e-10))
            weight_index = int(_DEFAULT_LUT_0[weight_index])

        acausal_scale = math.exp(-(time_between_pairs - 2.0 * delay) / tau)
        for i in range(len(a_acausal_eval) - 1):
            expected = ((i % 36) + 1) * acausal_scale
            self.assertTrue(np.allclose(a_acausal_eval[i], expected, atol=1e-10))


class TestSTDPFACETSHWSynapseHomVsNEST(unittest.TestCase):
    @staticmethod
    def _is_nest_available():
        try:
            import nest

            if hasattr(nest, 'synapse_models'):
                return 'stdp_facetshw_synapse_hom' in nest.synapse_models
            return 'stdp_facetshw_synapse_hom' in nest.Models()
        except Exception:
            return False

    @staticmethod
    def _run_nest_weight_trace(
        *,
        pre_spikes_ms,
        post_spikes_ms,
        dt_ms,
        delay_ms,
        params,
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
        post = nest.Create('parrot_neuron')
        sg_pre = nest.Create(
            'spike_generator',
            params={'spike_times': list(np.asarray(pre_spikes_ms, dtype=np.float64)), 'precise_times': False},
        )
        sg_post = nest.Create(
            'spike_generator',
            params={'spike_times': list(np.asarray(post_spikes_ms, dtype=np.float64)), 'precise_times': False},
        )
        wr = nest.Create('weight_recorder')

        model_name = f'stdp_facetshw_synapse_hom_bpstate_{np.random.randint(1_000_000_000)}'
        nest.CopyModel(
            'stdp_facetshw_synapse_hom',
            model_name,
            {
                'weight_recorder': wr,
                'weight': float(params['weight']),
                'delay': float(delay_ms),
                'tau_plus': float(params['tau_plus']),
                'tau_minus_stdp': float(params['tau_minus_stdp']),
                'Wmax': float(params['Wmax']),
                'a_thresh_th': float(params['a_thresh_th']),
                'a_thresh_tl': float(params['a_thresh_tl']),
                'synapses_per_driver': int(params['synapses_per_driver']),
                'driver_readout_time': float(params['driver_readout_time']),
                'lookuptable_0': list(params['lookuptable_0']),
                'lookuptable_1': list(params['lookuptable_1']),
                'lookuptable_2': list(params['lookuptable_2']),
                'configbit_0': list(params['configbit_0']),
                'configbit_1': list(params['configbit_1']),
                'reset_pattern': list(params['reset_pattern']),
            },
        )

        nest.Connect(sg_pre, pre, syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': float(dt_ms)})
        nest.Connect(sg_post, post, syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': float(dt_ms)})
        nest.Connect(pre, post, syn_spec={'synapse_model': model_name, 'receptor_type': 1})

        sim_duration_ms = float(max(np.max(pre_spikes_ms), np.max(post_spikes_ms)) + 2.0 * delay_ms + 2.0)
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
            weight=0.0,
            tau_plus=20.0,
            tau_minus_stdp=20.0,
            Wmax=100.0,
            synapses_per_driver=50,
            driver_readout_time=15.0,
            lookuptable_0=_DEFAULT_LUT_0.tolist(),
            lookuptable_1=_DEFAULT_LUT_1.tolist(),
            lookuptable_2=_DEFAULT_LUT_2.tolist(),
            configbit_0=_DEFAULT_CONFIG_0.tolist(),
            configbit_1=_DEFAULT_CONFIG_1.tolist(),
            reset_pattern=_DEFAULT_RESET_PATTERN.tolist(),
            a_thresh_th=21.835,
            a_thresh_tl=21.835,
        )

        dt_ms = 0.1
        delay_ms = 5.0
        pre_spikes = np.arange(10.0, 3610.0, 100.0, dtype=np.float64)
        post_spikes = pre_spikes + delay_ms

        nest_times, nest_weights = self._run_nest_weight_trace(
            pre_spikes_ms=pre_spikes,
            post_spikes_ms=post_spikes,
            dt_ms=dt_ms,
            delay_ms=delay_ms,
            params=params,
        )

        send_t_bp, payload_bp, _weights, _a_causal, _a_acausal = _run_bp_trace(
            pre_spikes_ms=pre_spikes + dt_ms,
            post_spikes_ms=post_spikes + dt_ms,
            sim_duration_ms=float(max(np.max(pre_spikes), np.max(post_spikes)) + 10.0),
            dt_ms=dt_ms,
            delay_ms=delay_ms,
            params=params,
        )

        np.testing.assert_allclose(send_t_bp, nest_times, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(payload_bp, nest_weights, atol=1e-12, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
