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

import copy
import importlib.util
import math
import unittest
from dataclasses import dataclass

import brainstate
import jax
import numpy as np
import numpy.testing as npt

from brainpy.state import clopath_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


@dataclass
class _HistEntry:
    t_: float
    dw_: float
    access_counter_: int = 0


class _FakeClopathTarget:
    def __init__(self, ltp_history=None, ltd_history=None, stdp_eps=1e-6):
        self.ltp_history = [] if ltp_history is None else list(ltp_history)
        self.ltd_history = [] if ltd_history is None else list(ltd_history)
        self.stdp_eps = float(stdp_eps)

    def get_LTP_history(self, t1, t2):
        out = []
        for e in self.ltp_history:
            # Match NEST ClopathArchivingNode::get_LTP_history() discretization.
            if e.t_ - 1.0e-6 < float(t1):
                continue
            if e.t_ - 1.0e-6 < float(t2):
                e.access_counter_ += 1
                out.append(e)
        return out

    def get_LTD_value(self, t):
        if not self.ltd_history or float(t) < 0.0:
            return 0.0
        for e in self.ltd_history:
            if abs(float(t) - e.t_) < self.stdp_eps:
                e.access_counter_ += 1
                return e.dw_
        return 0.0


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _reference_weight_trace(spike_times_ms, target, params):
    w = float(params['weight'])
    x_bar = float(params['x_bar'])
    tau_x = float(params['tau_x'])
    wmin = float(params['Wmin'])
    wmax = float(params['Wmax'])
    t_last = float(params['t_last_spike_ms'])
    delay = float(params['delay'])

    weights = np.empty((len(spike_times_ms),), dtype=np.float64)

    for i, t_spike in enumerate(spike_times_ms):
        t_spike = float(t_spike)

        ltp = target.get_LTP_history(t_last - delay, t_spike - delay)
        for e in ltp:
            minus_dt = t_last - (float(e.t_) + delay)
            w = min(wmax, w + float(e.dw_) * (x_bar * math.exp(minus_dt / tau_x)))

        ltd_dw = float(target.get_LTD_value(t_spike - delay))
        w = max(wmin, w - ltd_dw)

        weights[i] = w

        x_bar = x_bar * math.exp((t_last - t_spike) / tau_x) + 1.0 / tau_x
        t_last = t_spike

    return weights, x_bar, t_last


def _build_histories_from_nest_multimeter(events, resolution_ms, params):
    times = np.asarray(events['times'], dtype=np.float64)
    vm = np.asarray(events['V_m'], dtype=np.float64)
    u_plus = np.asarray(events['u_bar_plus'], dtype=np.float64)
    u_minus = np.asarray(events['u_bar_minus'], dtype=np.float64)
    u_bar = np.asarray(events['u_bar_bar'], dtype=np.float64)

    delay_steps = int(np.rint(float(params['delay_u_bars']) / float(resolution_ms))) + 1
    plus_buf = np.zeros((delay_steps,), dtype=np.float64)
    minus_buf = np.zeros((delay_steps,), dtype=np.float64)
    idx = 0

    ltp_hist = []
    ltd_hist = []

    for t_ms, u_i, u_p, u_m, u_b in zip(times, vm, u_plus, u_minus, u_bar):
        plus_buf[idx] = u_p
        minus_buf[idx] = u_m

        idx = (idx + 1) % delay_steps

        delayed_plus = plus_buf[idx]
        delayed_minus = minus_buf[idx]

        if u_i > params['theta_plus'] and delayed_plus > params['theta_minus']:
            dw_ltp = params['A_LTP'] * (u_i - params['theta_plus']) * (delayed_plus - params['theta_minus']) * resolution_ms
            ltp_hist.append(_HistEntry(float(t_ms), float(dw_ltp), 0))

        if delayed_minus > params['theta_minus']:
            if params['A_LTD_const']:
                dw_ltd = params['A_LTD'] * (delayed_minus - params['theta_minus'])
            else:
                dw_ltd = params['A_LTD'] * (u_b * u_b) * (delayed_minus - params['theta_minus']) / params['u_ref_squared']
            ltd_hist.append(_HistEntry(float(t_ms), float(dw_ltd), 0))

    return ltp_hist, ltd_hist


def _run_nest_clopath_pairing(spike_times_pre, spike_times_post):
    import nest

    resolution = 0.1
    init_w = 0.5

    nrn_params = {
        'V_m': -70.6,
        'E_L': -70.6,
        'V_peak': 33.0,
        'C_m': 281.0,
        'theta_minus': -70.6,
        'theta_plus': -45.3,
        'A_LTD': 14.0e-5,
        'A_LTP': 8.0e-5,
        'tau_u_bar_minus': 10.0,
        'tau_u_bar_plus': 7.0,
        'delay_u_bars': 4.0,
        'a': 4.0,
        'b': 0.0805,
        'V_reset': -70.6 + 21.0,
        'V_clamp': 33.0,
        't_clamp': 2.0,
        't_ref': 0.0,
        'A_LTD_const': True,
        'u_ref_squared': 60.0,
    }

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.local_num_threads = 1
    nest.resolution = resolution

    nrn = nest.Create('aeif_psc_delta_clopath', 1, nrn_params)
    parrot = nest.Create('parrot_neuron', 1)

    sg_pre = nest.Create('spike_generator', 1, {'spike_times': spike_times_pre})
    sg_post = nest.Create('spike_generator', 1, {'spike_times': spike_times_post})

    nest.Connect(sg_pre, parrot, syn_spec={'delay': resolution})
    nest.Connect(sg_post, nrn, syn_spec={'delay': resolution, 'weight': 80.0})

    mm = nest.Create('multimeter', params={
        'record_from': ['V_m', 'u_bar_plus', 'u_bar_minus', 'u_bar_bar'],
        'interval': resolution,
    })
    nest.Connect(mm, nrn)

    wr = nest.Create('weight_recorder', 1)
    nest.CopyModel('clopath_synapse', 'clopath_synapse_rec', {'weight_recorder': wr})

    nest.Connect(parrot, nrn, syn_spec={'synapse_model': 'clopath_synapse_rec', 'weight': init_w, 'delay': resolution})

    simulation_time = 10.0 + max(float(spike_times_pre[-1]), float(spike_times_post[-1]))
    nest.Simulate(simulation_time)

    wr_events = nest.GetStatus(wr)[0]['events']
    mm_events = nest.GetStatus(mm)[0]['events']

    wr_times = np.asarray(wr_events['times'], dtype=np.float64)
    wr_weights = np.asarray(wr_events['weights'], dtype=np.float64)

    return {
        'resolution': resolution,
        'init_w': init_w,
        'wr_times': wr_times,
        'wr_weights': wr_weights,
        'mm_events': mm_events,
        'nrn_params': nrn_params,
    }


class TestClopathSynapse(unittest.TestCase):
    def test_nest_default_parameters_and_properties(self):
        syn = clopath_synapse()

        self.assertAlmostEqual(syn.weight, 1.0, delta=0.0)
        self.assertAlmostEqual(syn.delay, 1.0, delta=0.0)
        self.assertEqual(syn.delay_steps, 1)
        self.assertAlmostEqual(syn.x_bar, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.tau_x, 15.0, delta=0.0)
        self.assertAlmostEqual(syn.Wmin, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.Wmax, 100.0, delta=0.0)
        self.assertAlmostEqual(syn.t_last_spike_ms, 0.0, delta=0.0)

        self.assertTrue(syn.HAS_DELAY)
        self.assertTrue(syn.IS_PRIMARY)
        self.assertTrue(syn.REQUIRES_CLOPATH_ARCHIVING)
        self.assertTrue(syn.SUPPORTS_HPC)
        self.assertTrue(syn.SUPPORTS_LBL)
        self.assertTrue(syn.SUPPORTS_WFR)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 1.0, delta=0.0)
        self.assertAlmostEqual(status['delay'], 1.0, delta=0.0)
        self.assertEqual(status['delay_steps'], 1)
        self.assertAlmostEqual(status['x_bar'], 0.0, delta=0.0)
        self.assertAlmostEqual(status['tau_x'], 15.0, delta=0.0)
        self.assertAlmostEqual(status['Wmin'], 0.0, delta=0.0)
        self.assertAlmostEqual(status['Wmax'], 100.0, delta=0.0)

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            defaults = nest.GetDefaults('clopath_synapse')

            self.assertAlmostEqual(syn.weight, float(defaults['weight']), delta=0.0)
            self.assertAlmostEqual(syn.tau_x, float(defaults['tau_x']), delta=0.0)
            self.assertAlmostEqual(syn.Wmin, float(defaults['Wmin']), delta=0.0)
            self.assertAlmostEqual(syn.Wmax, float(defaults['Wmax']), delta=0.0)
            self.assertIn('delay', defaults)

    def test_set_status_and_validation(self):
        syn = clopath_synapse()
        syn.set_status({'weight': 2.5, 'delay': 0.3, 'delay_steps': 3, 'x_bar': 0.2, 'tau_x': 20.0, 'Wmin': 0.0, 'Wmax': 200.0})

        self.assertAlmostEqual(syn.weight, 2.5, delta=0.0)
        self.assertAlmostEqual(syn.delay, 0.3, delta=0.0)
        self.assertEqual(syn.delay_steps, 3)
        self.assertAlmostEqual(syn.x_bar, 0.2, delta=0.0)
        self.assertAlmostEqual(syn.tau_x, 20.0, delta=0.0)
        self.assertAlmostEqual(syn.Wmax, 200.0, delta=0.0)

        with self.assertRaisesRegex(ValueError, 'Weight and Wmin must have same sign'):
            syn.set_status(weight=1.0, Wmin=-1.0)

        syn.set_status(weight=2.5, Wmin=0.0, Wmax=200.0)
        with self.assertRaisesRegex(ValueError, 'Weight and Wmax must have same sign'):
            syn.set_status(weight=1.0, Wmax=-2.0)

        with self.assertRaisesRegex(ValueError, 'delay must be > 0'):
            syn.set_status(delay=0.0)

        with self.assertRaisesRegex(ValueError, 'delay_steps must be >= 1'):
            syn.set_status(delay_steps=0)

        target = _FakeClopathTarget()
        with self.assertRaisesRegex(ValueError, 'multiplicity must be >= 0'):
            syn.send(t_spike_ms=1.0, target=target, multiplicity=-1.0)

    def test_send_ordering_matches_reference_trace(self):
        ltp = [
            _HistEntry(5.0, 0.15, 0),
            _HistEntry(8.0, 0.05, 0),
            _HistEntry(12.0, 0.2, 0),
            _HistEntry(16.0, 0.1, 0),
            _HistEntry(20.0, 0.4, 0),
        ]
        ltd = [
            _HistEntry(10.0, 0.03, 0),
            _HistEntry(17.0, 0.04, 0),
            _HistEntry(24.0, 0.02, 0),
        ]

        target = _FakeClopathTarget(copy.deepcopy(ltp), copy.deepcopy(ltd))
        target_ref = _FakeClopathTarget(copy.deepcopy(ltp), copy.deepcopy(ltd))

        syn = clopath_synapse(
            weight=0.5,
            delay=1.0,
            delay_steps=1,
            x_bar=0.2,
            tau_x=15.0,
            Wmin=0.0,
            Wmax=1.0,
            t_last_spike_ms=0.0,
        )

        spike_times = np.asarray([11.0, 18.0, 25.0], dtype=np.float64)

        events = syn.simulate_pre_spike_train(spike_times, target=target)
        got_weights = np.asarray([ev['weight'] for ev in events], dtype=np.float64)

        ref_weights, ref_x, ref_t_last = _reference_weight_trace(
            spike_times,
            target_ref,
            {
                'weight': 0.5,
                'x_bar': 0.2,
                'tau_x': 15.0,
                'Wmin': 0.0,
                'Wmax': 1.0,
                't_last_spike_ms': 0.0,
                'delay': 1.0,
            },
        )

        npt.assert_allclose(got_weights, ref_weights, atol=1e-15, rtol=0.0)
        self.assertAlmostEqual(syn.x_bar, ref_x, delta=1e-15)
        self.assertAlmostEqual(syn.t_last_spike_ms, ref_t_last, delta=0.0)

    def test_matches_nest_weights_from_recorded_histories_if_available(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        pre_post_pairs = [
            ([29.0, 129.0, 229.0, 329.0, 429.0], [19.0, 119.0, 219.0, 319.0, 419.0]),
            ([62.3, 95.6, 129.0, 162.3, 195.6, 229.0], [72.3, 105.6, 139.0, 172.3, 205.6, 239.0]),
        ]

        for pre_times, post_times in pre_post_pairs:
            run = _run_nest_clopath_pairing(pre_times, post_times)

            ltp_hist, ltd_hist = _build_histories_from_nest_multimeter(
                run['mm_events'],
                resolution_ms=run['resolution'],
                params=run['nrn_params'],
            )

            target = _FakeClopathTarget(ltp_hist, ltd_hist, stdp_eps=1e-6)

            syn = clopath_synapse(
                weight=run['init_w'],
                delay=run['resolution'],
                delay_steps=1,
                x_bar=0.0,
                tau_x=15.0,
                Wmin=0.0,
                Wmax=100.0,
                t_last_spike_ms=0.0,
            )

            local_weights = []
            for t_spike in run['wr_times']:
                ev = syn.send(t_spike_ms=float(t_spike), target=target)
                local_weights.append(ev['weight'])

            local_weights = np.asarray(local_weights, dtype=np.float64)

            self.assertEqual(local_weights.shape, run['wr_weights'].shape)
            npt.assert_allclose(local_weights, run['wr_weights'], atol=5e-9, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
