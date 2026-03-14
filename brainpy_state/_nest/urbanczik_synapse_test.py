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
import saiunit as u
import jax
import numpy as np
import numpy.testing as npt
from brainpy.state import urbanczik_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


@dataclass
class _HistEntry:
    t_: float
    dw_: float
    access_counter_: int = 0


class _FakeUrbanczikTarget:
    def __init__(
        self,
        history_entries=None,
        g_L=30.0,
        tau_L=10.0,
        C_m=300.0,
        tau_syn_ex=3.0,
        tau_syn_in=7.0,
        stdp_eps=1.0e-6,
    ):
        self.history = [] if history_entries is None else list(history_entries)
        self.g_L = float(g_L)
        self.tau_L = float(tau_L)
        self.C_m = float(C_m)
        self.tau_syn_ex = float(tau_syn_ex)
        self.tau_syn_in = float(tau_syn_in)
        self.stdp_eps = float(stdp_eps)

    def get_urbanczik_history(self, t1, t2, comp=1):
        del comp
        out = []
        t1 = float(t1)
        t2 = float(t2)
        for e in self.history:
            # Match NEST UrbanczikArchivingNode::get_urbanczik_history().
            if e.t_ - self.stdp_eps < t1:
                continue
            if e.t_ - self.stdp_eps < t2:
                e.access_counter_ += 1
                out.append(e)
        return out

    def get_g_L(self, comp):
        del comp
        return self.g_L

    def get_tau_L(self, comp):
        del comp
        return self.tau_L

    def get_C_m(self, comp):
        del comp
        return self.C_m

    def get_tau_syn_ex(self, comp):
        del comp
        return self.tau_syn_ex

    def get_tau_syn_in(self, comp):
        del comp
        return self.tau_syn_in


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


def _reference_urbanczik_weight_trace(spike_times_ms, history_entries, params, target_params):
    t_last = float(params['t_last_spike_ms'])
    delay = float(params['delay'])
    tau_delta = float(params['tau_Delta'])
    eta = float(params['eta'])
    w = float(params['weight'])
    init_w = float(params['weight'])
    wmin = float(params['Wmin'])
    wmax = float(params['Wmax'])
    pi_integral = float(params['PI_integral'])
    pi_exp_integral = float(params['PI_exp_integral'])
    tau_l_trace = float(params['tau_L_trace'])
    tau_s_trace = float(params['tau_s_trace'])

    g_L = float(target_params['g_L'])
    tau_L = float(target_params['tau_L'])
    C_m = float(target_params['C_m'])
    tau_syn_ex = float(target_params['tau_syn_ex'])
    tau_syn_in = float(target_params['tau_syn_in'])
    stdp_eps = float(target_params['stdp_eps'])

    dftype = brainstate.environ.dftype()
    weights = np.empty((len(spike_times_ms),), dtype=dftype)
    tau_s_used = np.empty((len(spike_times_ms),), dtype=dftype)

    hist = list(history_entries)
    for i, t_spike in enumerate(spike_times_ms):
        t_spike = float(t_spike)
        tau_s = tau_syn_ex if w > 0.0 else tau_syn_in
        tau_s_used[i] = tau_s

        d_pi_exp_integral = 0.0

        t1 = t_last - delay
        t2 = t_spike - delay
        for e in hist:
            if e.t_ - stdp_eps < t1:
                continue
            if e.t_ - stdp_eps >= t2:
                continue

            t_up = float(e.t_) + delay
            minus_delta_t_up = t_last - t_up
            minus_t_down = t_up - t_spike
            pi = (
                     tau_l_trace * math.exp(minus_delta_t_up / tau_L)
                     - tau_s_trace * math.exp(minus_delta_t_up / tau_s)
                 ) * float(e.dw_)
            pi_integral += pi
            d_pi_exp_integral += math.exp(minus_t_down / tau_delta) * pi

        pi_exp_integral = math.exp((t_last - t_spike) / tau_delta) * pi_exp_integral + d_pi_exp_integral
        w = pi_integral - pi_exp_integral
        w = init_w + w * 15.0 * C_m * tau_s * eta / (g_L * (tau_L - tau_s))
        if w > wmax:
            w = wmax
        elif w < wmin:
            w = wmin
        weights[i] = w

        tau_l_trace = tau_l_trace * math.exp((t_last - t_spike) / tau_L) + 1.0
        tau_s_trace = tau_s_trace * math.exp((t_last - t_spike) / tau_s) + 1.0
        t_last = t_spike

    return {
        'weights': weights,
        'tau_s_used': tau_s_used,
        'tau_L_trace': tau_l_trace,
        'tau_s_trace': tau_s_trace,
        'PI_integral': pi_integral,
        'PI_exp_integral': pi_exp_integral,
        't_last_spike_ms': t_last,
    }


def _run_nest_urbanczik_case():
    import nest

    resolution = 0.1
    nrn_model = 'pp_cond_exp_mc_urbanczik'
    nrn_params = {
        't_ref': 3.0,
        'g_sp': 600.0,
        'soma': {
            'V_m': -70.0,
            'C_m': 300.0,
            'E_L': -70.0,
            'g_L': 30.0,
            'E_ex': 0.0,
            'E_in': -75.0,
            'tau_syn_ex': 3.0,
            'tau_syn_in': 3.0,
        },
        'dendritic': {
            'V_m': -70.0,
            'C_m': 300.0,
            'E_L': -70.0,
            'g_L': 30.0,
            'tau_syn_ex': 3.0,
            'tau_syn_in': 3.0,
        },
        'phi_max': 0.15,
        'rate_slope': 0.5,
        'beta': 1.0 / 3.0,
        'theta': -55.0,
    }

    dftype = brainstate.environ.dftype()
    pre_syn_spike_times = np.array([1.0, 98.0], dtype=dftype)
    init_w = 100.0

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.local_num_threads = 1
    nest.resolution = resolution

    nest.SetDefaults(nrn_model, nrn_params)
    nrn = nest.Create(nrn_model)
    prrt_nrn = nest.Create('parrot_neuron')

    sg_prox = nest.Create('spike_generator', params={'spike_times': pre_syn_spike_times})
    spike_times_soma_inp = np.arange(10.0, 50.0, resolution)
    spike_weights_soma = 10.0 * np.ones_like(spike_times_soma_inp)
    sg_soma_exc = nest.Create(
        'spike_generator',
        params={'spike_times': spike_times_soma_inp, 'spike_weights': spike_weights_soma},
    )

    rqs = nest.GetDefaults(nrn_model)['recordables']
    mm = nest.Create('multimeter', params={'record_from': rqs, 'interval': resolution})
    wr = nest.Create('weight_recorder')
    sr_soma = nest.Create('spike_recorder')

    syns = nest.GetDefaults(nrn_model)['receptor_types']
    syn_params = {
        'synapse_model': 'urbanczik_synapse_wr',
        'receptor_type': syns['dendritic_exc'],
        'tau_Delta': 100.0,
        'eta': 0.75,
        'weight': init_w,
        'Wmax': 4.5 * nrn_params['dendritic']['C_m'],
        'delay': resolution,
    }

    nest.Connect(sg_prox, prrt_nrn, syn_spec={'delay': resolution})
    nest.CopyModel('urbanczik_synapse', 'urbanczik_synapse_wr', {'weight_recorder': wr[0]})
    nest.Connect(prrt_nrn, nrn, syn_spec=syn_params)
    nest.Connect(
        sg_soma_exc,
        nrn,
        syn_spec={'receptor_type': syns['soma_exc'], 'weight': 10.0 * resolution, 'delay': resolution},
    )
    nest.Connect(mm, nrn, syn_spec={'delay': resolution})
    nest.Connect(nrn, sr_soma, syn_spec={'delay': resolution})

    nest.Simulate(100.0)

    mm_events = nest.GetStatus(mm)[0]['events']
    wr_events = nest.GetStatus(wr)[0]['events']
    sr_events = nest.GetStatus(sr_soma)[0]['events']

    return {
        'resolution': float(resolution),
        'nrn_params': nrn_params,
        'syn_params': syn_params,
        'init_weight': float(init_w),
        'pre_syn_spike_times': pre_syn_spike_times,
        'mm_events': mm_events,
        'wr_times': np.asarray(wr_events['times'], dtype=dftype),
        'wr_weights': np.asarray(wr_events['weights'], dtype=dftype),
        'soma_spike_times': np.asarray(sr_events['times'], dtype=dftype),
    }


def _build_history_from_nest_trace(run):
    resolution = float(run['resolution'])
    nrn_params = run['nrn_params']
    mm_events = run['mm_events']

    dftype = brainstate.environ.dftype()
    t = np.asarray(mm_events['times'], dtype=dftype)
    v_w = np.asarray(mm_events['V_m.p'], dtype=dftype)

    g_D = float(nrn_params['g_sp'])
    g_L_soma = float(nrn_params['soma']['g_L'])
    E_L = float(nrn_params['soma']['E_L'])
    v_w_star = (g_L_soma * E_L + g_D * v_w) / (g_L_soma + g_D)

    phi_max = float(nrn_params['phi_max'])
    k = float(nrn_params['rate_slope'])
    beta = float(nrn_params['beta'])
    theta = float(nrn_params['theta'])

    rate = phi_max / (1.0 + k * np.exp(beta * (theta - v_w_star)))
    h = 15.0 * beta / (1.0 + np.exp(-beta * (theta - v_w_star)) / k)

    n_spikes = np.zeros_like(t, dtype=dftype)
    soma_spike_times = np.asarray(run['soma_spike_times'], dtype=dftype)
    if soma_spike_times.size > 0:
        idx = np.nonzero(np.isin(np.around(t, 4), np.around(soma_spike_times, 4)))[0]
        np.add.at(n_spikes, idx, 1.0)

    d_pi = (n_spikes - rate * resolution) * h
    return [_HistEntry(float(ti), float(dwi), 0) for ti, dwi in zip(t, d_pi)]


class TestUrbanczikSynapse(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_default_parameters_and_properties(self):
        syn = urbanczik_synapse()

        self.assertAlmostEqual(syn.weight, 1.0, delta=0.0)
        self.assertAlmostEqual(syn.delay, 1.0, delta=0.0)
        self.assertEqual(syn.delay_steps, 1)
        self.assertAlmostEqual(syn.tau_Delta, 100.0, delta=0.0)
        self.assertAlmostEqual(syn.eta, 0.07, delta=0.0)
        self.assertAlmostEqual(syn.Wmin, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.Wmax, 100.0, delta=0.0)
        self.assertAlmostEqual(syn.PI_integral, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.PI_exp_integral, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.tau_L_trace, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.tau_s_trace, 0.0, delta=0.0)
        self.assertAlmostEqual(syn.t_last_spike_ms, -1.0, delta=0.0)

        self.assertTrue(syn.HAS_DELAY)
        self.assertTrue(syn.IS_PRIMARY)
        self.assertTrue(syn.REQUIRES_URBANCZIK_ARCHIVING)
        self.assertTrue(syn.SUPPORTS_HPC)
        self.assertTrue(syn.SUPPORTS_LBL)
        self.assertTrue(syn.SUPPORTS_WFR)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 1.0, delta=0.0)
        self.assertAlmostEqual(status['tau_Delta'], 100.0, delta=0.0)
        self.assertAlmostEqual(status['eta'], 0.07, delta=0.0)
        self.assertAlmostEqual(status['Wmin'], 0.0, delta=0.0)
        self.assertAlmostEqual(status['Wmax'], 100.0, delta=0.0)
        self.assertIn('size_of', status)

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            defaults = nest.GetDefaults('urbanczik_synapse')

            self.assertAlmostEqual(syn.weight, float(defaults['weight']), delta=0.0)
            self.assertAlmostEqual(syn.tau_Delta, float(defaults['tau_Delta']), delta=0.0)
            self.assertAlmostEqual(syn.eta, float(defaults['eta']), delta=0.0)
            self.assertAlmostEqual(syn.Wmin, float(defaults['Wmin']), delta=0.0)
            self.assertAlmostEqual(syn.Wmax, float(defaults['Wmax']), delta=0.0)
            self.assertIn('delay', defaults)

    def test_set_status_and_validation(self):
        syn = urbanczik_synapse()
        syn.set_status(
            {
                'weight': 2.5,
                'delay': 0.3,
                'delay_steps': 3,
                'tau_Delta': 80.0,
                'eta': 0.2,
                'Wmin': 0.0,
                'Wmax': 200.0,
                'PI_integral': 1.2,
                'PI_exp_integral': 0.4,
                'tau_L_trace': 0.7,
                'tau_s_trace': 0.5,
                't_last_spike_ms': 11.0,
            }
        )

        self.assertAlmostEqual(syn.weight, 2.5, delta=0.0)
        self.assertAlmostEqual(syn.delay, 0.3, delta=0.0)
        self.assertEqual(syn.delay_steps, 3)
        self.assertAlmostEqual(syn.tau_Delta, 80.0, delta=0.0)
        self.assertAlmostEqual(syn.eta, 0.2, delta=0.0)
        self.assertAlmostEqual(syn.Wmax, 200.0, delta=0.0)
        self.assertAlmostEqual(syn.init_weight, 2.5, delta=0.0)
        self.assertAlmostEqual(syn.PI_integral, 1.2, delta=0.0)
        self.assertAlmostEqual(syn.PI_exp_integral, 0.4, delta=0.0)
        self.assertAlmostEqual(syn.tau_L_trace, 0.7, delta=0.0)
        self.assertAlmostEqual(syn.tau_s_trace, 0.5, delta=0.0)
        self.assertAlmostEqual(syn.t_last_spike_ms, 11.0, delta=0.0)

        with self.assertRaisesRegex(ValueError, 'Weight and Wmin must have same sign'):
            urbanczik_synapse(weight=1.0, Wmax=-2.0)

        with self.assertRaisesRegex(ValueError, 'Weight and Wmax must have same sign'):
            urbanczik_synapse(weight=1.0, Wmax=0.0)

        with self.assertRaisesRegex(ValueError, 'delay must be > 0'):
            syn.set_status(delay=0.0)

        with self.assertRaisesRegex(ValueError, 'delay_steps must be >= 1'):
            syn.set_status(delay_steps=0)

        target = _FakeUrbanczikTarget()
        with self.assertRaisesRegex(ValueError, 'multiplicity must be >= 0'):
            syn.send(t_spike_ms=1.0, target=target, multiplicity=-1.0)

    def test_send_ordering_matches_reference_trace(self):
        hist = [
            _HistEntry(0.9, 0.03, 0),
            _HistEntry(1.5, -0.01, 0),
            _HistEntry(2.8, 0.05, 0),
            _HistEntry(4.2, -0.02, 0),
            _HistEntry(6.5, 0.06, 0),
            _HistEntry(8.9, 0.04, 0),
        ]
        target = _FakeUrbanczikTarget(
            history_entries=copy.deepcopy(hist),
            g_L=35.0,
            tau_L=11.0,
            C_m=280.0,
            tau_syn_ex=3.5,
            tau_syn_in=6.5,
        )
        target_ref_hist = copy.deepcopy(hist)

        params = {
            'weight': 1.3,
            'delay': 1.1,
            'delay_steps': 2,
            'tau_Delta': 60.0,
            'eta': 0.12,
            'Wmin': 0.0,
            'Wmax': 4.0,
            'PI_integral': 0.0,
            'PI_exp_integral': 0.0,
            'tau_L_trace': 0.0,
            'tau_s_trace': 0.0,
            't_last_spike_ms': -1.0,
        }
        target_params = {
            'g_L': 35.0,
            'tau_L': 11.0,
            'C_m': 280.0,
            'tau_syn_ex': 3.5,
            'tau_syn_in': 6.5,
            'stdp_eps': 1.0e-6,
        }
        dftype = brainstate.environ.dftype()
        pre_spikes = np.asarray([2.0, 5.5, 9.0], dtype=dftype)

        syn = urbanczik_synapse(
            weight=params['weight'],
            delay=params['delay'],
            delay_steps=params['delay_steps'],
            tau_Delta=params['tau_Delta'],
            eta=params['eta'],
            Wmin=params['Wmin'],
            Wmax=params['Wmax'],
            PI_integral=params['PI_integral'],
            PI_exp_integral=params['PI_exp_integral'],
            tau_L_trace=params['tau_L_trace'],
            tau_s_trace=params['tau_s_trace'],
            t_last_spike_ms=params['t_last_spike_ms'],
        )

        events = syn.simulate_pre_spike_train(pre_spikes, target=target)
        got_weights = np.asarray([ev['weight'] for ev in events], dtype=dftype)
        got_tau_s = np.asarray([ev['tau_s_ms'] for ev in events], dtype=dftype)

        ref = _reference_urbanczik_weight_trace(pre_spikes, target_ref_hist, params, target_params)

        npt.assert_allclose(got_weights, ref['weights'], atol=1e-15, rtol=0.0)
        npt.assert_allclose(got_tau_s, ref['tau_s_used'], atol=0.0, rtol=0.0)
        self.assertAlmostEqual(syn.tau_L_trace, ref['tau_L_trace'], delta=1e-15)
        self.assertAlmostEqual(syn.tau_s_trace, ref['tau_s_trace'], delta=1e-15)
        self.assertAlmostEqual(syn.PI_integral, ref['PI_integral'], delta=1e-15)
        self.assertAlmostEqual(syn.PI_exp_integral, ref['PI_exp_integral'], delta=1e-15)
        self.assertAlmostEqual(syn.t_last_spike_ms, ref['t_last_spike_ms'], delta=0.0)

    def test_tau_syn_branch_uses_weight_sign(self):
        hist = [_HistEntry(0.5, 0.2, 0)]

        syn_pos = urbanczik_synapse(weight=1.0, Wmin=0.0, Wmax=100.0)
        target_pos = _FakeUrbanczikTarget(copy.deepcopy(hist), tau_syn_ex=3.0, tau_syn_in=9.0)
        ev_pos = syn_pos.send(t_spike_ms=2.0, target=target_pos)

        syn_neg = urbanczik_synapse(weight=-1.0, Wmin=-100.0, Wmax=-0.1)
        target_neg = _FakeUrbanczikTarget(copy.deepcopy(hist), tau_syn_ex=3.0, tau_syn_in=9.0)
        ev_neg = syn_neg.send(t_spike_ms=2.0, target=target_neg)

        self.assertAlmostEqual(ev_pos['tau_s_ms'], 3.0, delta=0.0)
        self.assertAlmostEqual(ev_neg['tau_s_ms'], 9.0, delta=0.0)

    def test_matches_nest_weight_trace_if_available(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        have_gsl = bool(nest.ll_api.sli_func('statusdict/have_gsl ::'))
        if not have_gsl:
            self.skipTest('NEST built without GSL')

        run = _run_nest_urbanczik_case()
        history = _build_history_from_nest_trace(run)

        target = _FakeUrbanczikTarget(
            history_entries=history,
            g_L=float(run['nrn_params']['dendritic']['g_L']),
            tau_L=float(run['nrn_params']['dendritic']['C_m']) / float(run['nrn_params']['dendritic']['g_L']),
            C_m=float(run['nrn_params']['dendritic']['C_m']),
            tau_syn_ex=float(run['nrn_params']['dendritic']['tau_syn_ex']),
            tau_syn_in=float(run['nrn_params']['dendritic']['tau_syn_in']),
            stdp_eps=1.0e-6,
        )

        syn = urbanczik_synapse(
            weight=float(run['syn_params']['weight']),
            delay=float(run['syn_params']['delay']),
            delay_steps=1,
            tau_Delta=float(run['syn_params']['tau_Delta']),
            eta=float(run['syn_params']['eta']),
            Wmin=0.0,
            Wmax=float(run['syn_params']['Wmax']),
            t_last_spike_ms=-1.0,
        )

        local_events = syn.simulate_pre_spike_train(run['wr_times'], target=target)
        dftype = brainstate.environ.dftype()
        local_weights = np.asarray([e['weight'] for e in local_events], dtype=dftype)

        self.assertEqual(local_weights.shape, run['wr_weights'].shape)
        npt.assert_allclose(local_weights, run['wr_weights'], atol=1e-12, rtol=0.0)

        denom = float(run['wr_weights'][-1] - run['init_weight'])
        if abs(denom) > 1.0e-12:
            rel_err = float((local_weights[-1] - run['wr_weights'][-1]) / denom)
            self.assertLess(abs(rel_err), 1e-12)


if __name__ == '__main__':
    unittest.main()
