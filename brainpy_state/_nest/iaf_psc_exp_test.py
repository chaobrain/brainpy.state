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
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u
from brainpy.state import iaf_psc_exp

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _propagator_exp(tau_syn, tau_m, c_m, h):
    beta = tau_syn * tau_m / (tau_m - tau_syn)
    gamma = beta / c_m
    inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)
    p = gamma * math.exp(-h / tau_syn) * math.expm1(h * inv_beta)
    if (not math.isfinite(p)) or (abs(p) < np.finfo(np.float64).tiny) or (p <= 0.0):
        p = h / c_m * math.exp(-h / tau_m)
    return p


class TestIAFPscExp(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    @staticmethod
    def _is_spike(spk):
        return bool(u.math.all(spk > 0.0))

    def test_nest_default_parameters(self):
        neuron = iaf_psc_exp(1)
        self.assertEqual(neuron.E_L, -70. * u.mV)
        self.assertEqual(neuron.C_m, 250. * u.pF)
        self.assertEqual(neuron.tau_m, 10. * u.ms)
        self.assertEqual(neuron.t_ref, 2. * u.ms)
        self.assertEqual(neuron.V_th, -55. * u.mV)
        self.assertEqual(neuron.V_reset, -70. * u.mV)
        self.assertEqual(neuron.tau_syn_ex, 2. * u.ms)
        self.assertEqual(neuron.tau_syn_in, 2. * u.ms)
        self.assertEqual(neuron.I_e, 0. * u.pA)
        self.assertEqual(neuron.delta, 0. * u.mV)
        self.assertEqual(neuron.spk_reset, 'hard')

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            iaf_psc_exp(1, V_reset=-55. * u.mV, V_th=-55. * u.mV)
        with self.assertRaises(ValueError):
            iaf_psc_exp(1, C_m=0.0 * u.pF)
        with self.assertRaises(ValueError):
            iaf_psc_exp(1, tau_m=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_exp(1, tau_syn_ex=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_exp(1, tau_syn_in=0.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_exp(1, t_ref=-1.0 * u.ms)
        with self.assertRaises(ValueError):
            iaf_psc_exp(1, rho=-1.0 / u.second)
        with self.assertRaises(ValueError):
            iaf_psc_exp(1, delta=-1.0 * u.mV)

    def test_dc_input_matches_nest_reference(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_exp(
                1,
                I_e=1000. * u.pA,
                V_initializer=braintools.init.Constant(-70. * u.mV),
                delta=0.0 * u.mV,
            )
            neuron.init_state()

            def _run_step(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0.0 * u.pA, x_filtered=0.0 * u.pA)
                return neuron.last_spike_time.value / u.ms

            brainstate.transform.for_loop(_run_step, jnp.arange(80))

            # Same first-spike time as in NEST iaf_psc_exp DC test: 4.8 ms.
            t_spike = float((neuron.last_spike_time.value / u.ms)[0])
            self.assertAlmostEqual(t_spike, 4.8, delta=1e-12)

    def test_step_equations_match_reference(self):
        with brainstate.environ.context(dt=self.dt):
            params = dict(
                E_L=-70.0,
                C_m=250.0,
                tau_m=10.0,
                t_ref=0.3,
                V_th=-55.0,
                V_reset=-70.0,
                tau_syn_ex=2.0,
                tau_syn_in=3.0,
                I_e=40.0,
            )
            neuron = iaf_psc_exp(
                1,
                E_L=params['E_L'] * u.mV,
                C_m=params['C_m'] * u.pF,
                tau_m=params['tau_m'] * u.ms,
                t_ref=params['t_ref'] * u.ms,
                V_th=params['V_th'] * u.mV,
                V_reset=params['V_reset'] * u.mV,
                tau_syn_ex=params['tau_syn_ex'] * u.ms,
                tau_syn_in=params['tau_syn_in'] * u.ms,
                I_e=params['I_e'] * u.pA,
                V_initializer=braintools.init.Constant(-67.0 * u.mV),
                delta=0.0 * u.mV,
            )
            neuron.init_state()

            x0_seq = [10.0, 20.0, 0.0, 0.0, -5.0, 0.0, 0.0]
            x1_seq = [0.0, 40.0, 0.0, 10.0, 0.0, 0.0, 0.0]
            w_seq = [0.0, 30.0, -15.0, 0.0, 0.0, 20.0, -10.0]
            n_steps = len(x0_seq)

            # Pre-compute per-step inputs as JAX arrays.
            x0_arr = jnp.array(x0_seq)
            x1_arr = jnp.array(x1_seq)
            w_arr = jnp.array(w_seq)

            def _run_step(k):
                neuron.add_delta_input('w', w_arr[k] * u.pA)
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=x0_arr[k] * u.pA, x_filtered=x1_arr[k] * u.pA)
                return (
                    spk[0],
                    (neuron.V.value / u.mV)[0],
                    (neuron.i_syn_ex.value / u.pA)[0],
                    (neuron.i_syn_in.value / u.pA)[0],
                    neuron.refractory_step_count.value[0],
                )

            results = brainstate.transform.for_loop(_run_step, jnp.arange(n_steps))
            spk_all = np.asarray(results[0])
            vm_all = np.asarray(results[1])
            iex_all = np.asarray(results[2])
            iin_all = np.asarray(results[3])
            r_all = np.asarray(results[4])

            # Compute reference values using Python math (same order as NEST update).
            v = -67.0 - params['E_L']
            i0 = 0.0
            i1 = 0.0
            iex = 0.0
            iin = 0.0
            r = 0
            refr = int(math.ceil(params['t_ref'] / 0.1))

            p11ex = math.exp(-0.1 / params['tau_syn_ex'])
            p11in = math.exp(-0.1 / params['tau_syn_in'])
            p22 = math.exp(-0.1 / params['tau_m'])
            p21ex = _propagator_exp(params['tau_syn_ex'], params['tau_m'], params['C_m'], 0.1)
            p21in = _propagator_exp(params['tau_syn_in'], params['tau_m'], params['C_m'], 0.1)
            p20 = params['tau_m'] / params['C_m'] * (1.0 - p22)
            th = params['V_th'] - params['E_L']
            reset_v = params['V_reset'] - params['E_L']

            for k, (x0, x1, w) in enumerate(zip(x0_seq, x1_seq, w_seq)):
                if r == 0:
                    v = v * p22 + iex * p21ex + iin * p21in + (params['I_e'] + i0) * p20
                else:
                    r -= 1

                iex *= p11ex
                iin *= p11in
                iex += (1.0 - p11ex) * i1
                iex += max(w, 0.0)
                iin += min(w, 0.0)

                spike_ref = v >= th
                if spike_ref:
                    r = refr
                    v = reset_v

                i0 = x0
                i1 = x1

                self.assertEqual(bool(spk_all[k] > 0.0), spike_ref)
                self.assertAlmostEqual(float(vm_all[k]), v + params['E_L'], delta=1e-11)
                self.assertAlmostEqual(float(iex_all[k]), iex, delta=1e-11)
                self.assertAlmostEqual(float(iin_all[k]), iin, delta=1e-11)
                self.assertEqual(int(r_all[k]), r)


if __name__ == '__main__':
    unittest.main()
