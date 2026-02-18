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

r"""Tests for glif_cond neuron model.

Tests verify that the glif_cond implementation matches NEST simulator dynamics
for all five GLIF model variants. Test strategy:

1. Default parameter verification against NEST C++ defaults.
2. Parameter validation (bad combinations, out-of-range values).
3. Step-by-step reference simulation for each GLIF variant, comparing membrane
   potential, threshold, ASC, and synaptic conductance traces.
4. Spike timing, refractory behavior, and reset rule verification.
5. Conductance-based alpha-function synapse verification.

Environment: JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
"""

import math
import os
import unittest

# Set JAX environment BEFORE importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest.glif_cond import glif_cond


# ---------------------------------------------------------------------------
# Reference RKF45 integrator for test comparisons
# ---------------------------------------------------------------------------

def _glif_cond_dynamics(v_rel, dg_vals, g_vals, is_refractory,
                        i_stim, asc_sum, p):
    r"""Reference ODE right-hand side matching NEST glif_cond_dynamics."""
    V = p['V_reset_rel'] if is_refractory else v_rel

    # Synaptic current
    I_syn = 0.0
    n_rec = len(g_vals)
    for k in range(n_rec):
        I_syn += g_vals[k] * (V + p['E_L'] - p['E_rev'][k])

    I_leak = p['G'] * V

    dv = 0.0 if is_refractory else (-I_leak - I_syn + i_stim + asc_sum) / p['C_m']

    ddg = [-dg_vals[k] / p['tau_syn'][k] for k in range(n_rec)]
    dg_out = [dg_vals[k] - g_vals[k] / p['tau_syn'][k] for k in range(n_rec)]

    return [dv] + ddg + dg_out


def _rkf45_ref_step(v_rel, dg_vals, g_vals, is_refractory,
                    i_stim, asc_sum, dt, h0, p, atol=1e-3):
    r"""Reference RKF45 integrator."""
    min_h = 1e-8
    n_rec = len(g_vals)
    n = 1 + 2 * n_rec
    t = 0.0
    h = max(h0, min_h)

    y = [v_rel] + list(dg_vals) + list(g_vals)

    def f(state):
        return _glif_cond_dynamics(
            state[0], state[1:1+n_rec], state[1+n_rec:],
            is_refractory, i_stim, asc_sum, p
        )

    max_iters = 10000
    iters = 0
    while t < dt and iters < max_iters:
        iters += 1
        h = max(min_h, min(h, dt - t))

        k1 = f(y)
        y2 = [y[i] + h * k1[i] / 4.0 for i in range(n)]
        k2 = f(y2)
        y3 = [y[i] + h * (3.0 * k1[i] / 32.0 + 9.0 * k2[i] / 32.0) for i in range(n)]
        k3 = f(y3)
        y4 = [y[i] + h * (1932.0 * k1[i] / 2197.0 - 7200.0 * k2[i] / 2197.0 + 7296.0 * k3[i] / 2197.0)
              for i in range(n)]
        k4 = f(y4)
        y5 = [y[i] + h * (439.0 * k1[i] / 216.0 - 8.0 * k2[i] + 3680.0 * k3[i] / 513.0
                           - 845.0 * k4[i] / 4104.0) for i in range(n)]
        k5 = f(y5)
        y6 = [y[i] + h * (-8.0 * k1[i] / 27.0 + 2.0 * k2[i] - 3544.0 * k3[i] / 2565.0
                           + 1859.0 * k4[i] / 4104.0 - 11.0 * k5[i] / 40.0) for i in range(n)]
        k6 = f(y6)

        y4_sol = [y[i] + h * (25.0 * k1[i] / 216.0 + 1408.0 * k3[i] / 2565.0
                               + 2197.0 * k4[i] / 4104.0 - k5[i] / 5.0) for i in range(n)]
        y5_sol = [y[i] + h * (16.0 * k1[i] / 135.0 + 6656.0 * k3[i] / 12825.0
                               + 28561.0 * k4[i] / 56430.0 - 9.0 * k5[i] / 50.0
                               + 2.0 * k6[i] / 55.0) for i in range(n)]

        err = max(abs(y5_sol[i] - y4_sol[i]) for i in range(n))

        if err <= atol or h <= min_h:
            y = y5_sol
            t += h
            fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
            h = max(min_h, h * fac)
        else:
            fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
            h = max(min_h, h * fac)

    return y[0], y[1:1+n_rec], y[1+n_rec:], h


# ---------------------------------------------------------------------------
# Reference full-step simulation matching NEST update() loop
# ---------------------------------------------------------------------------

def _ref_glif_step(state, params, dt, spike_weights=None):
    r"""Execute one reference NEST glif_cond update step.

    state: dict with keys V_rel, dg, g, r, i_stim, h, v_old,
           threshold_spike, threshold_voltage, ASCurrents, ASCurrents_sum,
           threshold
    params: dict with model parameters (I_e is always-on constant current)
    spike_weights: list of floats per receptor (raw weight in nS)
    """
    n_rec = len(params['tau_syn'])
    n_asc = len(params['asc_decay'])
    v_old = state['v_old']

    # Total external current = I_e + buffered I_stim (matches NEST B_.I_)
    i_ext = params.get('I_e', 0.0) + state['i_stim']

    # Pre-compute decay rates
    p = {
        'G': params['G'],
        'E_L': params['E_L'],
        'C_m': params['C_m'],
        'V_reset_rel': params['V_reset_rel'],
        'tau_syn': params['tau_syn'],
        'E_rev': params['E_rev'],
    }

    # Step 2: Integrate ODE
    is_refractory = state['r'] > 0
    v_i, dg_i, g_i, h_i = _rkf45_ref_step(
        state['V_rel'], list(state['dg']), list(state['g']),
        is_refractory, i_ext, state['ASCurrents_sum'],
        dt, state['h'], p
    )

    spiked = False

    if not is_refractory:
        # Decay spike threshold
        if params['has_theta_spike']:
            state['threshold_spike'] *= params['theta_spike_decay_rate']

        # ASC: compute time-averaged sum, then decay
        asc_sum_new = 0.0
        if params['has_asc']:
            for a in range(n_asc):
                asc_sum_new += params['asc_stable_coeff'][a] * state['ASCurrents'][a]
                state['ASCurrents'][a] *= params['asc_decay_rates'][a]
        state['ASCurrents_sum'] = asc_sum_new

        # Voltage-dependent threshold
        if params['has_theta_voltage']:
            beta = (i_ext + asc_sum_new) / params['G']
            phi = params['phi']
            state['threshold_voltage'] = (
                phi * (v_old - beta) * params['potential_decay_rate']
                + params['theta_voltage_decay_rate_inverse'] * (
                    state['threshold_voltage']
                    - phi * (v_old - beta)
                    - params['abpara_ratio_voltage'] * beta
                )
                + params['abpara_ratio_voltage'] * beta
            )

        # Update total threshold
        state['threshold'] = (
            state['threshold_spike'] + state['threshold_voltage'] + params['th_inf']
        )

        # Check for spike
        if v_i > state['threshold']:
            spiked = True
            state['r'] = params['refr_counts']

            # Reset ASC
            if params['has_asc']:
                for a in range(n_asc):
                    state['ASCurrents'][a] = (
                        params['asc_amps'][a]
                        + state['ASCurrents'][a] * params['asc_refractory_decay_rates'][a]
                    )

            # Reset voltage
            if not params['has_theta_spike']:
                v_i = params['V_reset_rel']
            else:
                v_i = params['voltage_reset_fraction'] * v_old + params['voltage_reset_add']
                state['threshold_spike'] = (
                    state['threshold_spike'] * params['theta_spike_refractory_decay_rate']
                    + params['th_spike_add']
                )
                state['threshold'] = (
                    state['threshold_spike'] + state['threshold_voltage'] + params['th_inf']
                )
    else:
        state['r'] -= 1
        v_i = v_old
        state['threshold'] = (
            state['threshold_spike'] + state['threshold_voltage'] + params['th_inf']
        )

    state['V_rel'] = v_i
    state['dg'] = list(dg_i)
    state['g'] = list(g_i)
    state['h'] = h_i

    # Add spike inputs
    if spike_weights is not None:
        for k in range(n_rec):
            state['dg'][k] += spike_weights[k] * params['cond_init_vals'][k]

    # Update i_stim for next step (new current input stored here)
    # (caller is responsible for setting state['i_stim'] = new value)

    state['v_old'] = state['V_rel']

    return spiked


def _make_ref_params(G=9.43, E_L=-78.85, th_inf_abs=-51.68, C_m=58.72,
                     t_ref=3.75, V_reset_abs=-78.85,
                     th_spike_add=0.37, th_spike_decay=0.009,
                     voltage_reset_fraction=0.20, voltage_reset_add=18.51,
                     th_voltage_index=0.005, th_voltage_decay=0.09,
                     asc_decay=(0.003, 0.1), asc_amps=(-9.18, -198.94),
                     asc_r=(1.0, 1.0),
                     tau_syn=(0.2, 2.0), E_rev=(0.0, -85.0),
                     has_theta_spike=False, has_asc=False,
                     has_theta_voltage=False, dt=0.01):
    r"""Build reference parameter dict with pre-computed decay rates."""
    th_inf = th_inf_abs - E_L
    V_reset_rel = V_reset_abs - E_L

    n_asc = len(asc_decay)
    n_rec = len(tau_syn)

    params = {
        'G': G, 'E_L': E_L, 'C_m': C_m, 't_ref': t_ref,
        'V_reset_rel': V_reset_rel, 'th_inf': th_inf,
        'th_spike_add': th_spike_add, 'th_spike_decay': th_spike_decay,
        'voltage_reset_fraction': voltage_reset_fraction,
        'voltage_reset_add': voltage_reset_add,
        'th_voltage_index': th_voltage_index,
        'th_voltage_decay': th_voltage_decay,
        'asc_decay': list(asc_decay), 'asc_amps': list(asc_amps),
        'asc_r': list(asc_r),
        'tau_syn': list(tau_syn), 'E_rev': list(E_rev),
        'has_theta_spike': has_theta_spike, 'has_asc': has_asc,
        'has_theta_voltage': has_theta_voltage,
        'I_e': 0.0,
    }

    # Pre-compute (matching NEST pre_run_hook)
    if has_theta_spike:
        params['theta_spike_decay_rate'] = math.exp(-th_spike_decay * dt)
        params['theta_spike_refractory_decay_rate'] = math.exp(-th_spike_decay * t_ref)

    if has_asc:
        params['asc_decay_rates'] = [math.exp(-asc_decay[a] * dt) for a in range(n_asc)]
        params['asc_stable_coeff'] = [
            ((1.0 / asc_decay[a]) / dt) * (1.0 - params['asc_decay_rates'][a])
            for a in range(n_asc)
        ]
        params['asc_refractory_decay_rates'] = [
            asc_r[a] * math.exp(-asc_decay[a] * t_ref)
            for a in range(n_asc)
        ]

    if has_theta_voltage:
        params['potential_decay_rate'] = math.exp(-G * dt / C_m)
        params['theta_voltage_decay_rate_inverse'] = 1.0 / math.exp(th_voltage_decay * dt)
        params['phi'] = th_voltage_index / (th_voltage_decay - G / C_m)
        params['abpara_ratio_voltage'] = th_voltage_index / th_voltage_decay

    params['refr_counts'] = math.ceil(t_ref / dt)
    params['cond_init_vals'] = [math.e / tau_syn[k] for k in range(n_rec)]

    return params


def _make_ref_state(params, V_abs=None, asc_init=(0.0, 0.0)):
    r"""Create initial reference state dict."""
    E_L = params['E_L']
    if V_abs is None:
        V_abs = E_L
    V_rel = V_abs - E_L

    n_rec = len(params['tau_syn'])
    n_asc = len(params['asc_decay'])

    asc = list(asc_init[:n_asc]) if params['has_asc'] else [0.0] * n_asc
    asc_sum = sum(asc) if params['has_asc'] else 0.0

    return {
        'V_rel': V_rel,
        'dg': [0.0] * n_rec,
        'g': [0.0] * n_rec,
        'r': 0,
        'i_stim': 0.0,
        'h': params.get('dt', 0.01),
        'v_old': V_rel,
        'threshold_spike': 0.0,
        'threshold_voltage': 0.0,
        'ASCurrents': asc,
        'ASCurrents_sum': asc_sum,
        'threshold': params['th_inf'],
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestGlifCond(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.01 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dg_inputs=None):
        r"""Execute one simulation step with optional per-receptor spike inputs.

        dg_inputs: list of (receptor_index, weight_nS) tuples
        """
        if dg_inputs is not None:
            for port, weight in dg_inputs:
                neuron.add_delta_input(
                    f'receptor_{port}_step{k}', weight * u.nS
                )
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    # ------------------------------------------------------------------
    # 1. Default parameter tests
    # ------------------------------------------------------------------

    def test_nest_default_parameters(self):
        r"""Verify default parameters match NEST C++ source."""
        neuron = glif_cond(1)
        self.assertEqual(neuron.g_m, 9.43 * u.nS)
        self.assertEqual(neuron.E_L, -78.85 * u.mV)
        self.assertEqual(neuron.V_th, -51.68 * u.mV)
        self.assertEqual(neuron.C_m, 58.72 * u.pF)
        self.assertEqual(neuron.t_ref, 3.75 * u.ms)
        self.assertEqual(neuron.V_reset, -78.85 * u.mV)
        self.assertEqual(neuron.th_spike_add, 0.37)
        self.assertEqual(neuron.th_spike_decay, 0.009)
        self.assertEqual(neuron.voltage_reset_fraction, 0.20)
        self.assertEqual(neuron.voltage_reset_add, 18.51)
        self.assertEqual(neuron.th_voltage_index, 0.005)
        self.assertEqual(neuron.th_voltage_decay, 0.09)
        self.assertEqual(neuron.asc_init, (0.0, 0.0))
        self.assertEqual(neuron.asc_decay, (0.003, 0.1))
        self.assertEqual(neuron.asc_amps, (-9.18, -198.94))
        self.assertEqual(neuron.asc_r, (1.0, 1.0))
        self.assertEqual(neuron.tau_syn, (0.2, 2.0))
        self.assertEqual(neuron.E_rev, (0.0, -85.0))
        self.assertFalse(neuron.has_theta_spike)
        self.assertFalse(neuron.has_asc)
        self.assertFalse(neuron.has_theta_voltage)

    # ------------------------------------------------------------------
    # 2. Parameter validation tests
    # ------------------------------------------------------------------

    def test_invalid_model_combination(self):
        r"""Invalid mechanism combinations should raise."""
        with self.assertRaises(ValueError):
            glif_cond(1, adapting_threshold=True, spike_dependent_threshold=False)

    def test_v_reset_ge_v_th_raises(self):
        with self.assertRaises(ValueError):
            glif_cond(1, V_reset=-50.0 * u.mV, V_th=-51.68 * u.mV)

    def test_capacitance_positive(self):
        with self.assertRaises(ValueError):
            glif_cond(1, C_m=0.0 * u.pF)

    def test_conductance_positive(self):
        with self.assertRaises(ValueError):
            glif_cond(1, g=0.0 * u.nS)

    def test_t_ref_positive(self):
        with self.assertRaises(ValueError):
            glif_cond(1, t_ref=0.0 * u.ms)

    def test_asc_size_mismatch(self):
        with self.assertRaises(ValueError):
            glif_cond(1, after_spike_currents=True,
                      asc_init=(0.0,), asc_decay=(0.003, 0.1),
                      asc_amps=(-9.18, -198.94), asc_r=(1.0, 1.0))

    def test_tau_syn_e_rev_size_mismatch(self):
        with self.assertRaises(ValueError):
            glif_cond(1, tau_syn=(0.2, 2.0), E_rev=(0.0,))

    # ------------------------------------------------------------------
    # 3. GLIF1 (LIF) subthreshold and spike test
    # ------------------------------------------------------------------

    def test_glif1_subthreshold_dynamics_match_reference(self):
        r"""GLIF1: subthreshold membrane potential matches reference."""
        dt_val = 0.01
        n_steps = 200

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=0.0 * u.pA,
            )
            neuron.init_state()

            # Inject current via the buffered I_stim mechanism
            params = _make_ref_params(
                has_theta_spike=False, has_asc=False, has_theta_voltage=False,
                dt=dt_val,
            )
            state = _make_ref_state(params, V_abs=-78.85)
            state['h'] = dt_val

            v_model = []
            v_ref = []

            for k in range(n_steps):
                # Apply external current at step 50
                if k == 50:
                    x_val = 300.0
                else:
                    x_val = 0.0

                spk = self._step(neuron, k, x=x_val * u.pA)
                v_m = float((neuron.V.value / u.mV)[0])
                v_model.append(v_m)

                # Reference
                _ref_glif_step(state, params, dt_val)
                state['i_stim'] = x_val
                v_ref.append(state['V_rel'] + params['E_L'])

            for k in range(n_steps):
                self.assertAlmostEqual(v_model[k], v_ref[k], places=6,
                                       msg=f"Step {k}: model={v_model[k]}, ref={v_ref[k]}")

    def test_glif1_spike_and_refractory(self):
        r"""GLIF1: spike generation and refractory period."""
        dt_val = 0.01
        n_steps = 1500

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params(
                has_theta_spike=False, has_asc=False, has_theta_voltage=False,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state(params, V_abs=-78.85)
            state['h'] = dt_val

            model_spikes = []
            ref_spikes = []

            for k in range(n_steps):
                spk = self._step(neuron, k)
                if float(spk[0]) > 0:
                    model_spikes.append(k)

                spiked = _ref_glif_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)

            self.assertEqual(model_spikes, ref_spikes,
                             msg="Spike times should match reference")
            self.assertGreater(len(model_spikes), 0, "Should have at least one spike")

    # ------------------------------------------------------------------
    # 4. GLIF2 (LIF_R) spike-dependent threshold test
    # ------------------------------------------------------------------

    def test_glif2_spike_dependent_threshold(self):
        r"""GLIF2: biologically defined reset with spike-dependent threshold."""
        dt_val = 0.01
        n_steps = 2000

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                spike_dependent_threshold=True,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params(
                has_theta_spike=True, has_asc=False, has_theta_voltage=False,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state(params, V_abs=-78.85)
            state['h'] = dt_val

            model_spikes = []
            ref_spikes = []

            for k in range(n_steps):
                spk = self._step(neuron, k)
                if float(spk[0]) > 0:
                    model_spikes.append(k)

                spiked = _ref_glif_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)

            self.assertEqual(model_spikes, ref_spikes)
            self.assertGreater(len(model_spikes), 0)

            # Verify threshold is tracked
            v_m = float((neuron.V.value / u.mV)[0])
            E_L = -78.85
            v_rel = v_m - E_L
            self.assertAlmostEqual(v_rel, state['V_rel'], places=5)

    # ------------------------------------------------------------------
    # 5. GLIF3 (LIF_ASC) after-spike currents test
    # ------------------------------------------------------------------

    def test_glif3_after_spike_currents(self):
        r"""GLIF3: after-spike currents match reference."""
        dt_val = 0.01
        n_steps = 2000

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                spike_dependent_threshold=False,
                after_spike_currents=True,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params(
                has_theta_spike=False, has_asc=True, has_theta_voltage=False,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state(params, V_abs=-78.85)
            state['h'] = dt_val

            model_spikes = []
            ref_spikes = []

            for k in range(n_steps):
                spk = self._step(neuron, k)
                if float(spk[0]) > 0:
                    model_spikes.append(k)

                spiked = _ref_glif_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)

            self.assertEqual(model_spikes, ref_spikes)
            self.assertGreater(len(model_spikes), 0)

    # ------------------------------------------------------------------
    # 6. GLIF4 (LIF_R_ASC) combined test
    # ------------------------------------------------------------------

    def test_glif4_combined_reset_and_asc(self):
        r"""GLIF4: spike-dependent threshold + after-spike currents."""
        dt_val = 0.01
        n_steps = 2000

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                spike_dependent_threshold=True,
                after_spike_currents=True,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params(
                has_theta_spike=True, has_asc=True, has_theta_voltage=False,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state(params, V_abs=-78.85)
            state['h'] = dt_val

            model_spikes = []
            ref_spikes = []

            for k in range(n_steps):
                spk = self._step(neuron, k)
                if float(spk[0]) > 0:
                    model_spikes.append(k)

                spiked = _ref_glif_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)

            self.assertEqual(model_spikes, ref_spikes)
            self.assertGreater(len(model_spikes), 0)

    # ------------------------------------------------------------------
    # 7. GLIF5 (LIF_R_ASC_A) full model test
    # ------------------------------------------------------------------

    def test_glif5_full_model(self):
        r"""GLIF5: all mechanisms enabled — voltage trace matches reference."""
        dt_val = 0.01
        n_steps = 2000

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                spike_dependent_threshold=True,
                after_spike_currents=True,
                adapting_threshold=True,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params(
                has_theta_spike=True, has_asc=True, has_theta_voltage=True,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state(params, V_abs=-78.85)
            state['h'] = dt_val

            model_spikes = []
            ref_spikes = []
            v_model = []
            v_ref = []

            for k in range(n_steps):
                spk = self._step(neuron, k)
                if float(spk[0]) > 0:
                    model_spikes.append(k)
                v_model.append(float((neuron.V.value / u.mV)[0]))

                spiked = _ref_glif_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)
                v_ref.append(state['V_rel'] + params['E_L'])

            self.assertEqual(model_spikes, ref_spikes)
            self.assertGreater(len(model_spikes), 0)

            # Check voltage trace agreement
            for k in range(n_steps):
                self.assertAlmostEqual(
                    v_model[k], v_ref[k], places=4,
                    msg=f"Step {k}: model={v_model[k]:.8f}, ref={v_ref[k]:.8f}"
                )

    # ------------------------------------------------------------------
    # 8. Alpha-function synapse test
    # ------------------------------------------------------------------

    def test_alpha_synapse_conductance_response(self):
        r"""Verify alpha-function synapse produces correct conductance waveform."""
        dt_val = 0.01
        n_steps = 500

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                tau_syn=(2.0,),
                E_rev=(0.0,),
                V_th=100.0 * u.mV,  # very high threshold to prevent spiking
                V_reset=-78.85 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            # Inject one spike of weight 1.0 at step 10
            g_trace = []
            for k in range(n_steps):
                if k == 10:
                    self._step(neuron, k, dg_inputs=[(0, 1.0)])
                else:
                    self._step(neuron, k)
                g_trace.append(float((neuron.g_syn[0].value / u.nS)[0]))

            # Peak should be approximately 1 nS at t = tau_syn = 2.0 ms
            # i.e., at step 10 + 2.0/0.01 = 210
            peak_idx = np.argmax(g_trace)
            peak_time_ms = (peak_idx - 10) * dt_val
            self.assertAlmostEqual(peak_time_ms, 2.0, delta=0.1,
                                   msg=f"Peak at {peak_time_ms} ms, expected ~2.0 ms")

            # Peak conductance should be approximately 1 nS
            self.assertAlmostEqual(g_trace[peak_idx], 1.0, delta=0.05,
                                   msg=f"Peak g={g_trace[peak_idx]} nS, expected ~1.0 nS")

    def test_synaptic_input_affects_voltage(self):
        r"""Excitatory input should depolarize, inhibitory should hyperpolarize."""
        dt_val = 0.01

        with brainstate.environ.context(dt=self.dt):
            # Baseline neuron
            base = glif_cond(
                1,
                tau_syn=(0.2, 2.0), E_rev=(0.0, -85.0),
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
            )
            base.init_state()

            # Excitatory input neuron
            exc = glif_cond(
                1,
                tau_syn=(0.2, 2.0), E_rev=(0.0, -85.0),
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
            )
            exc.init_state()

            # Inhibitory input neuron
            inh = glif_cond(
                1,
                tau_syn=(0.2, 2.0), E_rev=(0.0, -85.0),
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
            )
            inh.init_state()

            # Step 0: inject spikes
            self._step(base, 0)
            self._step(exc, 0, dg_inputs=[(0, 50.0)])  # excitatory receptor
            self._step(inh, 0, dg_inputs=[(1, 50.0)])   # inhibitory receptor

            # Step 1-5: let dynamics evolve
            for k in range(1, 6):
                self._step(base, k)
                self._step(exc, k)
                self._step(inh, k)

            v_base = float((base.V.value / u.mV)[0])
            v_exc = float((exc.V.value / u.mV)[0])
            v_inh = float((inh.V.value / u.mV)[0])

            # Excitatory (E_rev=0 mV > V_m) should depolarize
            self.assertGreater(v_exc, v_base)
            # Inhibitory (E_rev=-85 mV < V_m) should hyperpolarize
            self.assertLess(v_inh, v_base)

    # ------------------------------------------------------------------
    # 9. Refractory period test
    # ------------------------------------------------------------------

    def test_refractory_period_holds_voltage(self):
        r"""During refractory period, voltage should be held at pre-spike value."""
        dt_val = 0.01

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=800.0 * u.pA,
                t_ref=1.0 * u.ms,
            )
            neuron.init_state()

            # Run until first spike
            spike_step = None
            for k in range(2000):
                spk = self._step(neuron, k)
                if float(spk[0]) > 0:
                    spike_step = k
                    break

            self.assertIsNotNone(spike_step, "Should have spiked")

            # After spike, neuron should be refractory
            self.assertGreater(
                int(neuron.refractory_step_count.value[0]),
                0,
                "Should be refractory after spike"
            )

            # Run a few more steps — V should remain constant (at reset value)
            v_reset = float((neuron.V.value / u.mV)[0])
            for k in range(spike_step + 1, spike_step + 5):
                self._step(neuron, k)
                v_now = float((neuron.V.value / u.mV)[0])
                self.assertAlmostEqual(
                    v_now, v_reset, places=10,
                    msg=f"Voltage should be held during refractory at step {k}"
                )

    # ------------------------------------------------------------------
    # 10. Multi-receptor test
    # ------------------------------------------------------------------

    def test_three_receptors(self):
        r"""Model with three receptor ports initializes and runs correctly."""
        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                tau_syn=(0.5, 2.0, 5.0),
                E_rev=(0.0, -70.0, -85.0),
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            self.assertEqual(neuron.n_receptors, 3)

            # Inject spike to each receptor
            self._step(neuron, 0, dg_inputs=[(0, 10.0), (1, 5.0), (2, 3.0)])

            # All receptors should have non-zero dg
            for k_rec in range(3):
                dg_val = float((neuron.dg_syn[k_rec].value / u.nS)[0])
                self.assertGreater(abs(dg_val), 0.0,
                                   msg=f"Receptor {k_rec} should have non-zero dg")

    # ------------------------------------------------------------------
    # 11. Initial membrane potential test
    # ------------------------------------------------------------------

    def test_initial_v_m(self):
        r"""Initial V_m should match the initializer."""
        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()
            v_m = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v_m, -78.85, places=10)

    # ------------------------------------------------------------------
    # 12. Voltage-dependent threshold trace test (GLIF5)
    # ------------------------------------------------------------------

    def test_glif5_threshold_trace_matches_reference(self):
        r"""GLIF5: threshold components (spike + voltage) match reference trace."""
        dt_val = 0.01
        n_steps = 1000

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_cond(
                1,
                spike_dependent_threshold=True,
                after_spike_currents=True,
                adapting_threshold=True,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params(
                has_theta_spike=True, has_asc=True, has_theta_voltage=True,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state(params, V_abs=-78.85)
            state['h'] = dt_val

            for k in range(n_steps):
                self._step(neuron, k)
                _ref_glif_step(state, params, dt_val)

            # Compare threshold components
            E_L = -78.85
            th_model = float(neuron._threshold[0])
            th_ref = state['threshold']
            self.assertAlmostEqual(th_model, th_ref, places=4,
                                   msg=f"Threshold: model={th_model}, ref={th_ref}")

            th_spk_model = float(neuron._threshold_spike[0])
            th_spk_ref = state['threshold_spike']
            self.assertAlmostEqual(th_spk_model, th_spk_ref, places=4,
                                   msg=f"Threshold spike: model={th_spk_model}, ref={th_spk_ref}")

            th_vlt_model = float(neuron._threshold_voltage[0])
            th_vlt_ref = state['threshold_voltage']
            self.assertAlmostEqual(th_vlt_model, th_vlt_ref, places=4,
                                   msg=f"Threshold voltage: model={th_vlt_model}, ref={th_vlt_ref}")


if __name__ == '__main__':
    unittest.main()
