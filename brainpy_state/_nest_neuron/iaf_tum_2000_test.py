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

import math
import unittest

import brainstate
import braintools
import brainunit as bu
import jax
import numpy as np
from brainpy.state import iaf_psc_exp, iaf_tum_2000
from brainpy_state._nest_base._utils import propagator_exp

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)


def _is_spike(spk):
    dftype = brainstate.environ.dftype()
    return bool(np.asarray(bu.math.asarray(spk), dtype=dftype).reshape(-1)[0] > 0.0)


def _event_effective_weight_pA(event):
    sender_model = 'iaf_tum_2000'
    multiplicity = 1.0
    offset = 1.0
    dftype = brainstate.environ.dftype()

    if isinstance(event, dict):
        receptor = int(event.get('receptor_type', event.get('receptor', 0)))
        weight = float(np.asarray(bu.math.asarray(event.get('weight', 0.0 * bu.pA) / bu.pA), dtype=dftype))
        offset = float(np.asarray(bu.math.asarray(event.get('offset', 1.0)), dtype=dftype))
        multiplicity = float(np.asarray(bu.math.asarray(event.get('multiplicity', 1.0)), dtype=dftype))
        sender_model = event.get('sender_model', 'iaf_tum_2000')
    else:
        if len(event) == 2:
            receptor, weight = event
        elif len(event) == 3:
            receptor, weight, offset = event
        elif len(event) == 4:
            receptor, weight, offset, multiplicity = event
        else:
            receptor, weight, offset, multiplicity, sender_model = event
        receptor = int(receptor)
        weight = float(np.asarray(bu.math.asarray(weight / bu.pA), dtype=dftype))
        offset = float(np.asarray(bu.math.asarray(offset), dtype=dftype))
        multiplicity = float(np.asarray(bu.math.asarray(multiplicity), dtype=dftype))

    if receptor == 1 and sender_model != 'iaf_tum_2000':
        raise ValueError(
            'For receptor_type 1 in iaf_tum_2000, pre-synaptic neuron must also be of type iaf_tum_2000.'
        )
    s = weight * multiplicity
    if receptor == 1:
        s *= offset
    return receptor, s


def _reference_step(state, p, x0_next, x1_next, spike_events, dt_ms, t_ms):
    if state['r'] == 0:
        state['v'] = (
            state['v'] * p['P22']
            + state['i_syn_ex'] * p['P21ex']
            + state['i_syn_in'] * p['P21in']
            + (p['I_e'] + state['i0']) * p['P20']
        )
    else:
        state['r'] -= 1

    state['i_syn_ex'] *= p['P11ex']
    state['i_syn_in'] *= p['P11in']
    state['i_syn_ex'] += (1.0 - p['P11ex']) * state['i1']

    w_ex = 0.0
    w_in = 0.0
    for ev in spike_events:
        _, s = _event_effective_weight_pA(ev)
        if s > 0.0:
            w_ex += s
        else:
            w_in += s
    state['i_syn_ex'] += w_ex
    state['i_syn_in'] += w_in

    spike = state['v'] >= p['theta']
    if spike:
        state['r'] = p['refractory_counts']
        state['v'] = p['V_reset_rel']

        t_last = state['last_spike']
        if t_last < 0.0:
            t_last = 0.0
        t_spike = t_ms + dt_ms
        h_tsodyks = t_spike - t_last

        Puu = 0.0 if p['tau_fac'] == 0.0 else math.exp(-h_tsodyks / p['tau_fac'])
        Pyy = math.exp(-h_tsodyks / p['tau_psc'])
        Pzz = math.expm1(-h_tsodyks / p['tau_rec'])
        Pxy = (Pzz * p['tau_rec'] - (Pyy - 1.0) * p['tau_psc']) / (p['tau_psc'] - p['tau_rec'])

        z = 1.0 - state['x'] - state['y']
        u_prop = state['u'] * Puu
        x_prop = state['x'] + Pxy * state['y'] - Pzz * z
        y_prop = state['y'] * Pyy

        u_jump = u_prop + p['U'] * (1.0 - u_prop)
        delta_y = u_jump * x_prop

        state['x'] = x_prop - delta_y
        state['y'] = y_prop + delta_y
        state['u'] = u_jump
        state['spike_offset'] = delta_y
        state['last_spike'] = t_spike
    else:
        state['spike_offset'] = 0.0

    state['i0'] = x0_next
    state['i1'] = x1_next
    return spike


def _tsodyks_connection_spike_jump(state, p, t_spike):
    t_last = state['last_spike']
    if t_last < 0.0:
        t_last = 0.0
    h = t_spike - t_last

    Puu = 0.0 if p['tau_fac'] == 0.0 else math.exp(-h / p['tau_fac'])
    Pyy = math.exp(-h / p['tau_psc'])
    Pzz = math.expm1(-h / p['tau_rec'])
    Pxy = (Pzz * p['tau_rec'] - (Pyy - 1.0) * p['tau_psc']) / (p['tau_psc'] - p['tau_rec'])

    z = 1.0 - state['x'] - state['y']
    u_prop = state['u'] * Puu
    x_prop = state['x'] + Pxy * state['y'] - Pzz * z
    y_prop = state['y'] * Pyy

    u_jump = u_prop + p['U'] * (1.0 - u_prop)
    delta_y = u_jump * x_prop

    state['x'] = x_prop - delta_y
    state['y'] = y_prop + delta_y
    state['u'] = u_jump
    state['last_spike'] = t_spike
    return delta_y


class TestIAFTUM2000(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * bu.ms)
        self.dt = 0.1 * bu.ms
        self.dt_ms = 0.1

    def _step(self, neuron, step_idx, x=0.0 * bu.pA, x_filtered=0.0 * bu.pA, spike_events=None):
        with brainstate.environ.context(t=step_idx * self.dt):
            if spike_events is None:
                try:
                    return neuron.update(x=x, x_filtered=x_filtered)
                except TypeError:
                    return neuron.update(x=x)
            return neuron.update(x=x, x_filtered=x_filtered, spike_events=spike_events)

    def test_nest_cpp_default_parameters_and_metadata(self):
        neuron = iaf_tum_2000(1)

        self.assertEqual(neuron.E_L, -70.0 * bu.mV)
        self.assertEqual(neuron.C_m, 250.0 * bu.pF)
        self.assertEqual(neuron.tau_m, 10.0 * bu.ms)
        self.assertEqual(neuron.t_ref, 2.0 * bu.ms)
        self.assertEqual(neuron.V_th, -55.0 * bu.mV)
        self.assertEqual(neuron.V_reset, -70.0 * bu.mV)
        self.assertEqual(neuron.tau_syn_ex, 2.0 * bu.ms)
        self.assertEqual(neuron.tau_syn_in, 2.0 * bu.ms)
        self.assertEqual(neuron.I_e, 0.0 * bu.pA)
        self.assertEqual(neuron.rho, 0.01 / bu.second)
        self.assertEqual(neuron.delta, 0.0 * bu.mV)
        self.assertEqual(neuron.tau_fac, 1000.0 * bu.ms)
        self.assertEqual(neuron.tau_psc, 2.0 * bu.ms)
        self.assertEqual(neuron.tau_rec, 400.0 * bu.ms)
        self.assertEqual(neuron.U, 0.5)

        self.assertEqual(neuron.receptor_types, {'DEFAULT': 0, 'TSODYKS': 1})
        self.assertEqual(
            neuron.recordables,
            ['V_m', 'I_syn_ex', 'I_syn_in', 'x', 'y', 'u', 'spike_offset'],
        )

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, V_reset=-55.0 * bu.mV, V_th=-55.0 * bu.mV)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, C_m=0.0 * bu.pF)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, tau_m=0.0 * bu.ms)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, tau_syn_ex=0.0 * bu.ms)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, tau_syn_in=0.0 * bu.ms)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, tau_psc=0.0 * bu.ms)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, tau_rec=0.0 * bu.ms)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, tau_fac=-0.1 * bu.ms)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, t_ref=-0.1 * bu.ms)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, U=1.1)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, rho=-1.0 / bu.second)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, delta=-1.0 * bu.mV)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, x=0.8, y=0.3)
        with self.assertRaises(ValueError):
            iaf_tum_2000(1, u=1.2)

    def test_receptor_type_1_requires_iaf_tum_sender(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_tum_2000(
                1,
                E_L=0.0 * bu.mV,
                V_reset=0.0 * bu.mV,
                V_th=1000.0 * bu.mV,
                V_initializer=braintools.init.Constant(0.0 * bu.mV),
            )
            neuron.init_state()

            with self.assertRaisesRegex(ValueError, 'pre-synaptic neuron'):
                self._step(
                    neuron,
                    0,
                    spike_events=[{
                        'receptor_type': 1,
                        'weight': 1.0 * bu.pA,
                        'offset': 0.5,
                        'sender_model': 'iaf_psc_exp',
                    }],
                )

    def test_step_equations_match_nest_reference(self):
        with brainstate.environ.context(dt=self.dt):
            p = dict(
                E_L=-70.0,
                C_m=250.0,
                tau_m=10.0,
                t_ref=0.3,
                V_th=-55.0,
                V_reset=-70.0,
                tau_syn_ex=2.0,
                tau_syn_in=3.0,
                I_e=1200.0,
                rho=0.01,
                delta=0.0,
                tau_fac=1000.0,
                tau_psc=2.0,
                tau_rec=400.0,
                U=0.5,
            )
            neuron = iaf_tum_2000(
                1,
                E_L=p['E_L'] * bu.mV,
                C_m=p['C_m'] * bu.pF,
                tau_m=p['tau_m'] * bu.ms,
                t_ref=p['t_ref'] * bu.ms,
                V_th=p['V_th'] * bu.mV,
                V_reset=p['V_reset'] * bu.mV,
                tau_syn_ex=p['tau_syn_ex'] * bu.ms,
                tau_syn_in=p['tau_syn_in'] * bu.ms,
                I_e=p['I_e'] * bu.pA,
                rho=p['rho'] / bu.second,
                delta=p['delta'] * bu.mV,
                tau_fac=p['tau_fac'] * bu.ms,
                tau_psc=p['tau_psc'] * bu.ms,
                tau_rec=p['tau_rec'] * bu.ms,
                U=p['U'],
                x=0.0,
                y=0.0,
                u=0.0,
                V_initializer=braintools.init.Constant(-67.0 * bu.mV),
            )
            neuron.init_state()

            x0_seq = [0.0, 20.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            x1_seq = [0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            events_seq = [
                [(0, 50.0 * bu.pA)],
                [{'receptor_type': 1, 'weight': 200.0 * bu.pA, 'offset': 0.25, 'multiplicity': 2.0}],
                [(1, -150.0 * bu.pA, 0.4)],
                [{'receptor_type': 0, 'weight': -20.0 * bu.pA, 'multiplicity': 3.0}],
                [],
                [(0, 10.0 * bu.pA)],
                [],
                [(1, 80.0 * bu.pA, 0.125, 4.0)],
                [],
                [],
            ]

            ref = dict(
                v=-67.0 - p['E_L'],
                i0=0.0,
                i1=0.0,
                i_syn_ex=0.0,
                i_syn_in=0.0,
                r=0,
                x=0.0,
                y=0.0,
                u=0.0,
                spike_offset=0.0,
                last_spike=-1e7,
            )
            ref_par = dict(
                P11ex=math.exp(-self.dt_ms / p['tau_syn_ex']),
                P11in=math.exp(-self.dt_ms / p['tau_syn_in']),
                P22=math.exp(-self.dt_ms / p['tau_m']),
                P21ex=float(propagator_exp(np.asarray(p['tau_syn_ex']), np.asarray(p['tau_m']),
                                                        np.asarray(p['C_m']), self.dt_ms)),
                P21in=float(propagator_exp(np.asarray(p['tau_syn_in']), np.asarray(p['tau_m']),
                                                        np.asarray(p['C_m']), self.dt_ms)),
                P20=p['tau_m'] / p['C_m'] * (1.0 - math.exp(-self.dt_ms / p['tau_m'])),
                theta=p['V_th'] - p['E_L'],
                V_reset_rel=p['V_reset'] - p['E_L'],
                refractory_counts=int(math.ceil(p['t_ref'] / self.dt_ms)),
                tau_fac=p['tau_fac'],
                tau_psc=p['tau_psc'],
                tau_rec=p['tau_rec'],
                U=p['U'],
                I_e=p['I_e'],
            )

            for k, (x0, x1, evs) in enumerate(zip(x0_seq, x1_seq, events_seq)):
                spk = self._step(
                    neuron,
                    k,
                    x=x0 * bu.pA,
                    x_filtered=x1 * bu.pA,
                    spike_events=evs,
                )

                spike_ref = _reference_step(
                    ref,
                    ref_par,
                    x0_next=x0,
                    x1_next=x1,
                    spike_events=evs,
                    dt_ms=self.dt_ms,
                    t_ms=k * self.dt_ms,
                )

                self.assertEqual(_is_spike(spk), spike_ref)
                self.assertAlmostEqual(float((neuron.V.value / bu.mV)[0]), ref['v'] + p['E_L'], delta=1e-11)
                self.assertAlmostEqual(float((neuron.i_syn_ex.value / bu.pA)[0]), ref['i_syn_ex'], delta=1e-11)
                self.assertAlmostEqual(float((neuron.i_syn_in.value / bu.pA)[0]), ref['i_syn_in'], delta=1e-11)
                self.assertAlmostEqual(float((neuron.i_0.value / bu.pA)[0]), ref['i0'], delta=1e-11)
                self.assertAlmostEqual(float((neuron.i_1.value / bu.pA)[0]), ref['i1'], delta=1e-11)
                self.assertEqual(int(neuron.refractory_step_count.value[0]), ref['r'])
                self.assertAlmostEqual(float(neuron.x.value[0]), ref['x'], delta=1e-12)
                self.assertAlmostEqual(float(neuron.y.value[0]), ref['y'], delta=1e-12)
                self.assertAlmostEqual(float(neuron.u.value[0]), ref['u'], delta=1e-12)
                self.assertAlmostEqual(float(neuron.spike_offset.value[0]), ref['spike_offset'], delta=1e-12)
                self.assertAlmostEqual(float((neuron.last_spike_time.value / bu.ms)[0]), ref['last_spike'], delta=1e-12)

    def test_matches_iaf_psc_exp_plus_tsodyks_synapse_equivalence(self):
        with brainstate.environ.context(dt=self.dt):
            common = dict(
                E_L=-70.0 * bu.mV,
                C_m=250.0 * bu.pF,
                tau_m=10.0 * bu.ms,
                t_ref=2.0 * bu.ms,
                V_th=-55.0 * bu.mV,
                V_reset=-70.0 * bu.mV,
                tau_syn_ex=2.0 * bu.ms,
                tau_syn_in=2.0 * bu.ms,
                I_e=0.0 * bu.pA,
                rho=0.01 / bu.second,
                delta=0.0 * bu.mV,
            )
            stp = dict(
                tau_psc=2.0,
                tau_rec=400.0,
                tau_fac=1000.0,
                U=0.5,
            )

            pre_tum = iaf_tum_2000(
                1,
                **common,
                tau_psc=stp['tau_psc'] * bu.ms,
                tau_rec=stp['tau_rec'] * bu.ms,
                tau_fac=stp['tau_fac'] * bu.ms,
                U=stp['U'],
                x=0.0,
                y=0.0,
                u=0.0,
            )
            post_tum = iaf_tum_2000(
                1,
                **common,
                tau_psc=stp['tau_psc'] * bu.ms,
                tau_rec=stp['tau_rec'] * bu.ms,
                tau_fac=stp['tau_fac'] * bu.ms,
                U=stp['U'],
            )
            pre_exp = iaf_psc_exp(1, **common)
            post_exp = iaf_psc_exp(1, **common)

            pre_tum.init_state()
            post_tum.init_state()
            pre_exp.init_state()
            post_exp.init_state()

            delay_steps = 10
            weight = 250.0
            dc_amp = 500.0
            stim_start = 5.0
            stim_end = 45.0
            n_steps = int(round(60.0 / self.dt_ms))

            # Pre-compute DC drive array as JAX quantities.
            dc_arr = jax.numpy.array(
                [dc_amp if (stim_start <= k * self.dt_ms < stim_end) else 0.0
                 for k in range(n_steps)],
                dtype=jax.numpy.float64,
            ) * bu.pA

            # Phase 1: JIT-compiled run of both pre-synaptic neurons.
            def _pre_step(k):
                x_i = dc_arr[k]
                with brainstate.environ.context(t=k * self.dt):
                    tum_spk = pre_tum.update(x=x_i)
                    exp_spk = pre_exp.update(x=x_i)
                return tum_spk, pre_tum.spike_offset.value, exp_spk

            pre_results = brainstate.transform.for_loop(_pre_step, jax.numpy.arange(n_steps))
            tum_spks = jax.numpy.asarray(pre_results[0])[:, 0]      # (n_steps,)
            tum_offsets = jax.numpy.asarray(pre_results[1])[:, 0]   # (n_steps,)
            exp_spks = jax.numpy.asarray(pre_results[2])[:, 0]      # (n_steps,)

            # Verify pre-synaptic spike equivalence.
            np.testing.assert_array_equal(
                np.asarray(tum_spks) > 0.0,
                np.asarray(exp_spks) > 0.0,
            )

            # Phase 2: Compute delayed effective weights as a JAX array.
            # w_ex_per_step[k] = weight * tum_offsets[k - delay_steps]
            #                     if tum_spks[k - delay_steps] > 0, else 0.
            padded_spks = jax.numpy.concatenate(
                [jax.numpy.zeros(delay_steps, dtype=jax.numpy.float64), tum_spks]
            )
            padded_offsets = jax.numpy.concatenate(
                [jax.numpy.zeros(delay_steps, dtype=jax.numpy.float64), tum_offsets]
            )
            w_ex_per_step = jax.numpy.where(
                padded_spks[:n_steps] > 0.0,
                weight * padded_offsets[:n_steps],
                0.0,
            )  # shape: (n_steps,)

            # Phase 3: JIT-compiled run of both post-synaptic neurons.
            def _post_step(k):
                w_ex_k = w_ex_per_step[k]
                # post_tum: pass pre-computed JAX weight arrays directly.
                w_ex_arr = jax.numpy.broadcast_to(
                    w_ex_k, post_tum.varshape
                )
                w_in_arr = jax.numpy.zeros(post_tum.varshape, dtype=jax.numpy.float64)
                # post_exp: inject via add_delta_input (consumed by sum_delta_inputs in update).
                post_exp.add_delta_input('_fl', w_ex_k * bu.pA)
                with brainstate.environ.context(t=k * self.dt):
                    post_tum.update(_w_ex_jnp=w_ex_arr, _w_in_jnp=w_in_arr)
                    post_exp.update()
                return (
                    post_tum.V.value / bu.mV,
                    post_tum.i_syn_ex.value / bu.pA,
                    post_exp.V.value / bu.mV,
                    post_exp.i_syn_ex.value / bu.pA,
                )

            post_results = brainstate.transform.for_loop(_post_step, jax.numpy.arange(n_steps))
            post_tum_v = np.asarray(post_results[0])[:, 0]
            post_tum_i = np.asarray(post_results[1])[:, 0]
            post_exp_v = np.asarray(post_results[2])[:, 0]
            post_exp_i = np.asarray(post_results[3])[:, 0]

            np.testing.assert_allclose(post_tum_v, post_exp_v, atol=1e-10)
            np.testing.assert_allclose(post_tum_i, post_exp_i, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
