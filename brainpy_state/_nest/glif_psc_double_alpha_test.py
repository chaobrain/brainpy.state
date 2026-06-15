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

r"""Tests for glif_psc_double_alpha neuron model.

Tests verify that the glif_psc_double_alpha implementation matches NEST
simulator dynamics for all five GLIF model variants. The model uses double
alpha-function shaped synaptic currents (fast + slow components).

Test strategy:

1. Default parameter verification against NEST C++ defaults.
2. Parameter validation (bad combinations, out-of-range values).
3. Step-by-step reference simulation for each GLIF variant, comparing membrane
   potential, threshold, ASC, and synaptic current traces.
4. Spike timing, refractory behavior, and reset rule verification.
5. Double alpha-function PSC current waveform verification (fast + slow).
6. Comparison with glif_psc when tau_syn_fast == tau_syn_slow and amp_slow == 1.

The reference simulation uses the same exact integration propagator matrices
as NEST's C++ implementation.

Environment: JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
"""

import math
import os
import unittest

# Set JAX environment BEFORE importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as np
import jax.numpy as jnp

import brainstate
import braintools
import brainunit as u

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest.glif_psc_double_alpha import glif_psc_double_alpha
from brainpy_state._nest._utils import alpha_propagator_p31_p32


# ---------------------------------------------------------------------------
# Reference exact integration for test comparisons
# ---------------------------------------------------------------------------

def _compute_propagators_double(tau_syn_fast_list, tau_syn_slow_list, amp_slow_list,
                                tau_m, c_m, dt):
    r"""Pre-compute all propagator matrix elements for double alpha function."""
    n_rec = len(tau_syn_fast_list)
    P33 = math.exp(-dt / tau_m)
    P30 = (1.0 / c_m) * (1.0 - P33) * tau_m

    # Fast component
    P11_fast = []
    P21_fast = []
    P22_fast = []
    P31_fast = []
    P32_fast = []
    PSCInitialValues_fast = []

    # Slow component
    P11_slow = []
    P21_slow = []
    P22_slow = []
    P31_slow = []
    P32_slow = []
    PSCInitialValues_slow = []

    for i in range(n_rec):
        # Fast
        p11_fast = math.exp(-dt / tau_syn_fast_list[i])
        P11_fast.append(p11_fast)
        P22_fast.append(p11_fast)
        P21_fast.append(dt * p11_fast)
        p31_fast, p32_fast = alpha_propagator_p31_p32(tau_syn_fast_list[i], tau_m, c_m, dt)
        P31_fast.append(p31_fast)
        P32_fast.append(p32_fast)
        PSCInitialValues_fast.append(math.e / tau_syn_fast_list[i])

        # Slow
        p11_slow = math.exp(-dt / tau_syn_slow_list[i])
        P11_slow.append(p11_slow)
        P22_slow.append(p11_slow)
        P21_slow.append(dt * p11_slow)
        p31_slow, p32_slow = alpha_propagator_p31_p32(tau_syn_slow_list[i], tau_m, c_m, dt)
        P31_slow.append(p31_slow)
        P32_slow.append(p32_slow)
        PSCInitialValues_slow.append(math.e / tau_syn_slow_list[i] * amp_slow_list[i])

    return {
        'P33': P33, 'P30': P30,
        'P11_fast': P11_fast, 'P21_fast': P21_fast, 'P22_fast': P22_fast,
        'P31_fast': P31_fast, 'P32_fast': P32_fast,
        'PSCInitialValues_fast': PSCInitialValues_fast,
        'P11_slow': P11_slow, 'P21_slow': P21_slow, 'P22_slow': P22_slow,
        'P31_slow': P31_slow, 'P32_slow': P32_slow,
        'PSCInitialValues_slow': PSCInitialValues_slow,
    }


def _ref_glif_psc_double_alpha_step(state, params, dt, spike_weights=None):
    r"""Execute one reference NEST glif_psc_double_alpha update step.

    state: dict with keys V_rel, y1_fast, y2_fast, y1_slow, y2_slow, r, i_stim,
           v_old, threshold_spike, threshold_voltage, ASCurrents, ASCurrents_sum,
           threshold, I_syn_fast, I_syn_slow
    params: dict with model parameters
    spike_weights: list of floats per receptor (raw weight in pA)
    """
    n_rec = len(params['tau_syn_fast'])
    n_asc = len(params['asc_decay'])
    v_old = state['v_old']
    prop = params['prop']

    # Total external current = I_e + buffered I_stim
    i_ext = params.get('I_e', 0.0) + state['i_stim']

    spiked = False

    if state['r'] == 0:
        # neuron not refractory

        # Decay spike threshold component
        if params['has_theta_spike']:
            state['threshold_spike'] *= params['theta_spike_decay_rate']

        # ASC: compute time-averaged sum, then decay
        asc_sum = 0.0
        if params['has_asc']:
            for a in range(n_asc):
                asc_sum += params['asc_stable_coeff'][a] * state['ASCurrents'][a]
                state['ASCurrents'][a] *= params['asc_decay_rates'][a]
        state['ASCurrents_sum'] = asc_sum

        # Voltage dynamics (exact integration)
        v_new = v_old * prop['P33'] + (i_ext + asc_sum) * prop['P30']

        # Add fast synapse component
        I_syn_fast = 0.0
        for i in range(n_rec):
            v_new += prop['P31_fast'][i] * state['y1_fast'][i] + prop['P32_fast'][i] * state['y2_fast'][i]
            I_syn_fast += state['y2_fast'][i]
        state['I_syn_fast'] = I_syn_fast

        # Add slow synapse component
        I_syn_slow = 0.0
        for i in range(n_rec):
            v_new += prop['P31_slow'][i] * state['y1_slow'][i] + prop['P32_slow'][i] * state['y2_slow'][i]
            I_syn_slow += state['y2_slow'][i]
        state['I_syn_slow'] = I_syn_slow

        state['I_syn'] = I_syn_fast + I_syn_slow

        # Voltage-dependent threshold
        if params['has_theta_voltage']:
            beta = (i_ext + asc_sum) / params['G']
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
        if v_new > state['threshold']:
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
                v_new = params['V_reset_rel']
            else:
                v_new = params['voltage_reset_fraction'] * v_old + params['voltage_reset_add']
                state['threshold_spike'] = (
                    state['threshold_spike'] * params['theta_spike_refractory_decay_rate']
                    + params['th_spike_add']
                )
                state['threshold'] = (
                    state['threshold_spike'] + state['threshold_voltage'] + params['th_inf']
                )

        state['V_rel'] = v_new
    else:
        # neuron is refractory
        state['r'] -= 1
        state['V_rel'] = v_old
        state['threshold'] = (
            state['threshold_spike'] + state['threshold_voltage'] + params['th_inf']
        )

    # Update synaptic current state variables (alpha shape PSCs)
    y1_fast_old = list(state['y1_fast'])
    y2_fast_old = list(state['y2_fast'])
    y1_slow_old = list(state['y1_slow'])
    y2_slow_old = list(state['y2_slow'])

    for i in range(n_rec):
        # Fast component
        state['y2_fast'][i] = prop['P21_fast'][i] * y1_fast_old[i] + prop['P22_fast'][i] * y2_fast_old[i]
        state['y1_fast'][i] = prop['P11_fast'][i] * y1_fast_old[i]

        # Slow component
        state['y2_slow'][i] = prop['P21_slow'][i] * y1_slow_old[i] + prop['P22_slow'][i] * y2_slow_old[i]
        state['y1_slow'][i] = prop['P11_slow'][i] * y1_slow_old[i]

    # Add spike inputs
    if spike_weights is not None:
        for i in range(n_rec):
            state['y1_fast'][i] += spike_weights[i] * prop['PSCInitialValues_fast'][i]
            state['y1_slow'][i] += spike_weights[i] * prop['PSCInitialValues_slow'][i]

    # Update external current (caller sets state['i_stim'])
    state['v_old'] = state['V_rel']

    return spiked


def _make_ref_params_double(G=9.43, E_L=-78.85, th_inf_abs=-51.68, C_m=58.72,
                            t_ref=3.75, V_reset_abs=-78.85,
                            th_spike_add=0.37, th_spike_decay=0.009,
                            voltage_reset_fraction=0.20, voltage_reset_add=18.51,
                            th_voltage_index=0.005, th_voltage_decay=0.09,
                            asc_decay=(0.003, 0.1), asc_amps=(-9.18, -198.94),
                            asc_r=(1.0, 1.0),
                            tau_syn_fast=(2.0,), tau_syn_slow=(6.0,), amp_slow=(0.3,),
                            has_theta_spike=False, has_asc=False,
                            has_theta_voltage=False, dt=0.01):
    r"""Build reference parameter dict with pre-computed decay rates."""
    th_inf = th_inf_abs - E_L
    V_reset_rel = V_reset_abs - E_L
    Tau = C_m / G  # membrane time constant

    n_asc = len(asc_decay)
    n_rec = len(tau_syn_fast)

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
        'tau_syn_fast': list(tau_syn_fast),
        'tau_syn_slow': list(tau_syn_slow),
        'amp_slow': list(amp_slow),
        'has_theta_spike': has_theta_spike, 'has_asc': has_asc,
        'has_theta_voltage': has_theta_voltage,
        'I_e': 0.0,
    }

    # Pre-compute propagator matrices
    params['prop'] = _compute_propagators_double(
        tau_syn_fast, tau_syn_slow, amp_slow, Tau, C_m, dt
    )

    # Pre-compute decay rates (matching NEST pre_run_hook)
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

    return params


def _make_ref_state_double(params, V_abs=None, asc_init=(0.0, 0.0)):
    r"""Create initial reference state dict for double alpha."""
    E_L = params['E_L']
    if V_abs is None:
        V_abs = E_L
    V_rel = V_abs - E_L

    n_rec = len(params['tau_syn_fast'])
    n_asc = len(params['asc_decay'])

    asc = list(asc_init[:n_asc]) if params['has_asc'] else [0.0] * n_asc
    asc_sum = sum(asc) if params['has_asc'] else 0.0

    return {
        'V_rel': V_rel,
        'y1_fast': [0.0] * n_rec,
        'y2_fast': [0.0] * n_rec,
        'y1_slow': [0.0] * n_rec,
        'y2_slow': [0.0] * n_rec,
        'r': 0,
        'i_stim': 0.0,
        'v_old': V_rel,
        'threshold_spike': 0.0,
        'threshold_voltage': 0.0,
        'ASCurrents': asc,
        'ASCurrents_sum': asc_sum,
        'threshold': params['th_inf'],
        'I_syn_fast': 0.0,
        'I_syn_slow': 0.0,
        'I_syn': 0.0,
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestGlifPscDoubleAlpha(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.01 * u.ms

    def _step(self, neuron, k, x=0.0 * u.pA, dy_inputs=None):
        r"""Execute one simulation step with optional per-receptor spike inputs.

        dy_inputs: list of (receptor_index, weight_pA) tuples
        """
        if dy_inputs is not None:
            for port, weight in dy_inputs:
                neuron.add_delta_input(
                    f'receptor_{port}_step{k}', weight * u.pA,
                    label=f'receptor_{port}'
                )
        with brainstate.environ.context(t=k * self.dt):
            return neuron.update(x=x)

    # ------------------------------------------------------------------
    # 1. Default parameter tests
    # ------------------------------------------------------------------

    def test_nest_default_parameters(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Verify default parameters match NEST C++ source."""
            neuron = glif_psc_double_alpha(1)
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
            self.assertEqual(neuron.tau_syn_fast, (2.0,))
            self.assertEqual(neuron.tau_syn_slow, (6.0,))
            self.assertEqual(neuron.amp_slow, (0.3,))
            self.assertFalse(neuron.has_theta_spike)
            self.assertFalse(neuron.has_asc)
            self.assertFalse(neuron.has_theta_voltage)

    # ------------------------------------------------------------------
    # 2. Parameter validation tests
    # ------------------------------------------------------------------

    def test_invalid_model_combination(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Invalid mechanism combinations should raise."""
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, adapting_threshold=True, spike_dependent_threshold=False)

    def test_v_reset_ge_v_th_raises(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, V_reset=-50.0 * u.mV, V_th=-51.68 * u.mV)

    def test_capacitance_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, C_m=0.0 * u.pF)

    def test_conductance_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, g=0.0 * u.nS)

    def test_t_ref_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, t_ref=0.0 * u.ms)

    def test_asc_size_mismatch(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, after_spike_currents=True,
                                      asc_init=(0.0,), asc_decay=(0.003, 0.1),
                                      asc_amps=(-9.18, -198.94), asc_r=(1.0, 1.0))

    def test_tau_syn_fast_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, tau_syn_fast=(0.0,))

    def test_tau_syn_slow_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, tau_syn_slow=(0.0,))

    def test_amp_slow_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, amp_slow=(0.0,))

    def test_tau_syn_size_mismatch(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""tau_syn_fast, tau_syn_slow, amp_slow must have same length."""
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, tau_syn_fast=(2.0, 3.0), tau_syn_slow=(6.0,))
            with self.assertRaises(ValueError):
                glif_psc_double_alpha(1, tau_syn_fast=(2.0,), amp_slow=(0.3, 0.5))

    # ------------------------------------------------------------------
    # 3. IAFPropagatorAlpha tests
    # ------------------------------------------------------------------

    def test_propagator_alpha_regular(self):
        r"""IAFPropagatorAlpha regular case (tau_m != tau_syn)."""
        tau_syn = 2.0
        tau_m = 6.226  # = 58.72 / 9.43
        c_m = 58.72
        h = 0.01

        P31, P32 = alpha_propagator_p31_p32(tau_syn, tau_m, c_m, h)

        self.assertTrue(math.isfinite(P31))
        self.assertTrue(math.isfinite(P32))
        self.assertGreater(P32, 0)

    def test_propagator_alpha_singular(self):
        r"""IAFPropagatorAlpha singular case (tau_m ≈ tau_syn)."""
        tau_syn = 6.226
        tau_m = 6.226
        c_m = 58.72
        h = 0.01

        P31, P32 = alpha_propagator_p31_p32(tau_syn, tau_m, c_m, h)

        expected_P32 = h / c_m * math.exp(-h / tau_m)
        expected_P31 = 0.5 * h * h / c_m * math.exp(-h / tau_m)
        self.assertAlmostEqual(P32, expected_P32, places=15)
        self.assertAlmostEqual(P31, expected_P31, places=15)

    # ------------------------------------------------------------------
    # 4. GLIF1 (LIF) subthreshold and spike test
    # ------------------------------------------------------------------

    def test_glif1_subthreshold_dynamics_match_reference(self):
        r"""GLIF1: subthreshold membrane potential matches reference."""
        dt_val = 0.01
        n_steps = 200

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=0.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params_double(
                has_theta_spike=False, has_asc=False, has_theta_voltage=False,
                dt=dt_val,
            )
            state = _make_ref_state_double(params, V_abs=-78.85)

            x_vals = jnp.zeros(n_steps, dtype=jnp.float64).at[50].set(300.0)

            def _body(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=x_vals[k] * u.pA)
                return neuron.V.value / u.mV

            v_model_all = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            v_model = [float(v_model_all[k, 0]) for k in range(n_steps)]

            v_ref = []
            for k in range(n_steps):
                _ref_glif_psc_double_alpha_step(state, params, dt_val)
                state['i_stim'] = float(x_vals[k])
                v_ref.append(state['V_rel'] + params['E_L'])

            for k in range(n_steps):
                self.assertAlmostEqual(v_model[k], v_ref[k], places=6,
                                       msg=f"Step {k}: model={v_model[k]}, ref={v_ref[k]}")

    def test_glif1_spike_and_refractory(self):
        r"""GLIF1: spike generation and refractory period."""
        dt_val = 0.01
        n_steps = 1500

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params_double(
                has_theta_spike=False, has_asc=False, has_theta_voltage=False,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state_double(params, V_abs=-78.85)

            def _body(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0.0 * u.pA)
                return spk

            spk_trace = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            model_spikes = [k for k in range(n_steps) if float(spk_trace[k, 0]) > 0]

            ref_spikes = []
            for k in range(n_steps):
                spiked = _ref_glif_psc_double_alpha_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)

            self.assertEqual(model_spikes, ref_spikes,
                             msg="Spike times should match reference")
            self.assertGreater(len(model_spikes), 0, "Should have at least one spike")

    # ------------------------------------------------------------------
    # 5. GLIF2 (LIF_R) spike-dependent threshold test
    # ------------------------------------------------------------------

    def test_glif2_spike_dependent_threshold(self):
        r"""GLIF2: biologically defined reset with spike-dependent threshold."""
        dt_val = 0.01
        n_steps = 500

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                spike_dependent_threshold=True,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params_double(
                has_theta_spike=True, has_asc=False, has_theta_voltage=False,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state_double(params, V_abs=-78.85)

            def _body(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0.0 * u.pA)
                return spk

            spk_trace = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            model_spikes = [k for k in range(n_steps) if float(spk_trace[k, 0]) > 0]

            ref_spikes = []
            for k in range(n_steps):
                spiked = _ref_glif_psc_double_alpha_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)

            self.assertEqual(model_spikes, ref_spikes)
            self.assertGreater(len(model_spikes), 0)

    # ------------------------------------------------------------------
    # 6. GLIF3 (LIF_ASC) after-spike currents test
    # ------------------------------------------------------------------

    def test_glif3_after_spike_currents(self):
        r"""GLIF3: after-spike currents match reference."""
        dt_val = 0.01
        n_steps = 500

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                spike_dependent_threshold=False,
                after_spike_currents=True,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params_double(
                has_theta_spike=False, has_asc=True, has_theta_voltage=False,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state_double(params, V_abs=-78.85)

            def _body(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0.0 * u.pA)
                return spk

            spk_trace = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            model_spikes = [k for k in range(n_steps) if float(spk_trace[k, 0]) > 0]

            ref_spikes = []
            for k in range(n_steps):
                spiked = _ref_glif_psc_double_alpha_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)

            self.assertEqual(model_spikes, ref_spikes)
            self.assertGreater(len(model_spikes), 0)

    # ------------------------------------------------------------------
    # 7. GLIF4 (LIF_R_ASC) combined test
    # ------------------------------------------------------------------

    def test_glif4_combined_reset_and_asc(self):
        r"""GLIF4: spike-dependent threshold + after-spike currents."""
        dt_val = 0.01
        n_steps = 500

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                spike_dependent_threshold=True,
                after_spike_currents=True,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params_double(
                has_theta_spike=True, has_asc=True, has_theta_voltage=False,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state_double(params, V_abs=-78.85)

            def _body(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0.0 * u.pA)
                return spk

            spk_trace = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            model_spikes = [k for k in range(n_steps) if float(spk_trace[k, 0]) > 0]

            ref_spikes = []
            for k in range(n_steps):
                spiked = _ref_glif_psc_double_alpha_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)

            self.assertEqual(model_spikes, ref_spikes)
            self.assertGreater(len(model_spikes), 0)

    # ------------------------------------------------------------------
    # 8. GLIF5 (LIF_R_ASC_A) full model test
    # ------------------------------------------------------------------

    def test_glif5_full_model(self):
        r"""GLIF5: all mechanisms enabled — voltage trace matches reference."""
        dt_val = 0.01
        n_steps = 500

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                spike_dependent_threshold=True,
                after_spike_currents=True,
                adapting_threshold=True,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params_double(
                has_theta_spike=True, has_asc=True, has_theta_voltage=True,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state_double(params, V_abs=-78.85)

            def _body(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0.0 * u.pA)
                return (spk, neuron.V.value / u.mV)

            results = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            spk_trace = results[0]
            v_trace = results[1]
            model_spikes = [k for k in range(n_steps) if float(spk_trace[k, 0]) > 0]
            v_model = [float(v_trace[k, 0]) for k in range(n_steps)]

            ref_spikes = []
            v_ref = []
            for k in range(n_steps):
                spiked = _ref_glif_psc_double_alpha_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)
                v_ref.append(state['V_rel'] + params['E_L'])

            self.assertEqual(model_spikes, ref_spikes)
            self.assertGreater(len(model_spikes), 0)

            for k in range(n_steps):
                self.assertAlmostEqual(
                    v_model[k], v_ref[k], places=6,
                    msg=f"Step {k}: model={v_model[k]:.8f}, ref={v_ref[k]:.8f}"
                )

    # ------------------------------------------------------------------
    # 9. Double alpha-function PSC current test
    # ------------------------------------------------------------------

    def test_double_alpha_psc_current_response(self):
        r"""Verify double alpha-function PSC produces correct current waveform.

        A spike of weight 1.0 should produce:
        - Fast component: peak current of 1 pA at t = tau_syn_fast
        - Slow component: peak current of amp_slow pA at t = tau_syn_slow
        """
        dt_val = 0.01
        n_steps = 1000
        tau_syn_fast_val = 2.0  # ms
        tau_syn_slow_val = 6.0  # ms
        amp_slow_val = 0.3

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                tau_syn_fast=(tau_syn_fast_val,),
                tau_syn_slow=(tau_syn_slow_val,),
                amp_slow=(amp_slow_val,),
                V_th=100.0 * u.mV,  # very high threshold to prevent spiking
                V_reset=-78.85 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            w_vals = jnp.zeros(n_steps, dtype=jnp.float64).at[10].set(1.0)

            def _body(k):
                neuron.add_delta_input('fl', w_vals[k] * u.pA, label='receptor_0')
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0.0 * u.pA)
                return (neuron.y2_fast[0].value / u.pA, neuron.y2_slow[0].value / u.pA)

            results = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            y2_fast_trace = [float(results[0][k, 0]) for k in range(n_steps)]
            y2_slow_trace = [float(results[1][k, 0]) for k in range(n_steps)]
            y2_total_trace = [y2_fast_trace[k] + y2_slow_trace[k] for k in range(n_steps)]

            # Peak of fast component should be approximately 1 pA at t = tau_syn_fast
            fast_peak_idx = np.argmax(y2_fast_trace)
            fast_peak_time_ms = (fast_peak_idx - 10) * dt_val
            self.assertAlmostEqual(fast_peak_time_ms, tau_syn_fast_val, delta=0.2,
                                   msg=f"Fast peak at {fast_peak_time_ms} ms, expected ~{tau_syn_fast_val} ms")
            self.assertAlmostEqual(y2_fast_trace[fast_peak_idx], 1.0, delta=0.05,
                                   msg=f"Fast peak I={y2_fast_trace[fast_peak_idx]} pA, expected ~1.0 pA")

            # Peak of slow component should be approximately amp_slow pA at t = tau_syn_slow
            slow_peak_idx = np.argmax(y2_slow_trace)
            slow_peak_time_ms = (slow_peak_idx - 10) * dt_val
            self.assertAlmostEqual(slow_peak_time_ms, tau_syn_slow_val, delta=0.2,
                                   msg=f"Slow peak at {slow_peak_time_ms} ms, expected ~{tau_syn_slow_val} ms")
            self.assertAlmostEqual(y2_slow_trace[slow_peak_idx], amp_slow_val, delta=0.02,
                                   msg=f"Slow peak I={y2_slow_trace[slow_peak_idx]} pA, expected ~{amp_slow_val} pA")

    def test_double_alpha_same_tau_doubles_amplitude(self):
        r"""When tau_syn_fast == tau_syn_slow and amp_slow == 1.0, amplitude doubles.

        This matches the NEST test: when both components are identical, the total
        peak should be 2x the single component peak.
        """
        dt_val = 0.1  # coarser dt to match NEST test resolution
        n_steps = 500
        tau_syn_val = 2.0

        with brainstate.environ.context(dt=dt_val * u.ms):
            neuron = glif_psc_double_alpha(
                1,
                tau_syn_fast=(tau_syn_val,),
                tau_syn_slow=(tau_syn_val,),  # same as fast
                amp_slow=(1.0,),  # same amplitude
                V_th=100.0 * u.mV,
                V_reset=-78.85 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            w_vals = jnp.zeros(n_steps, dtype=jnp.float64).at[10].set(1.0)

            def _body(k):
                neuron.add_delta_input('fl', w_vals[k] * u.pA, label='receptor_0')
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0.0 * u.pA)
                return neuron.get_I_syn() / u.pA

            I_syn_all = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            I_syn_trace = [float(I_syn_all[k, 0]) for k in range(n_steps)]

            # Peak should be approximately 2.0 pA (double the single alpha)
            peak_I_syn = max(I_syn_trace)
            self.assertAlmostEqual(peak_I_syn, 2.0, delta=0.01,
                                   msg=f"Peak I_syn={peak_I_syn} pA, expected ~2.0 pA")

    def test_double_alpha_different_amp_slow(self):
        r"""When amp_slow = 0.5, total peak is 1.5x the fast component peak."""
        dt_val = 0.1
        n_steps = 500
        tau_syn_val = 2.0

        with brainstate.environ.context(dt=dt_val * u.ms):
            neuron = glif_psc_double_alpha(
                1,
                tau_syn_fast=(tau_syn_val,),
                tau_syn_slow=(tau_syn_val,),
                amp_slow=(0.5,),
                V_th=100.0 * u.mV,
                V_reset=-78.85 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            w_vals = jnp.zeros(n_steps, dtype=jnp.float64).at[10].set(1.0)

            def _body(k):
                neuron.add_delta_input('fl', w_vals[k] * u.pA, label='receptor_0')
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0.0 * u.pA)
                return neuron.get_I_syn() / u.pA

            I_syn_all = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            I_syn_trace = [float(I_syn_all[k, 0]) for k in range(n_steps)]

            # Peak should be approximately 1.5 pA
            peak_I_syn = max(I_syn_trace)
            self.assertAlmostEqual(peak_I_syn, 1.5, delta=0.01,
                                   msg=f"Peak I_syn={peak_I_syn} pA, expected ~1.5 pA")

    def test_double_alpha_different_tau_shifts_peak(self):
        r"""When tau_syn_slow > tau_syn_fast, the combined peak shifts right."""
        dt_val = 0.1
        n_steps = 500
        tau_syn_fast_val = 2.0
        tau_syn_slow_val = 4.0
        amp_slow_val = 0.5

        with brainstate.environ.context(dt=dt_val * u.ms):
            neuron = glif_psc_double_alpha(
                1,
                tau_syn_fast=(tau_syn_fast_val,),
                tau_syn_slow=(tau_syn_slow_val,),
                amp_slow=(amp_slow_val,),
                V_th=100.0 * u.mV,
                V_reset=-78.85 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            w_vals = jnp.zeros(n_steps, dtype=jnp.float64).at[10].set(1.0)

            def _body(k):
                neuron.add_delta_input('fl', w_vals[k] * u.pA, label='receptor_0')
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0.0 * u.pA)
                return (neuron.y2_fast[0].value / u.pA, neuron.get_I_syn() / u.pA)

            results = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            y2_fast_trace = [float(results[0][k, 0]) for k in range(n_steps)]
            I_syn_trace = [float(results[1][k, 0]) for k in range(n_steps)]

            # The combined peak should be to the right of the fast peak
            fast_peak_idx = np.argmax(y2_fast_trace)
            total_peak_idx = np.argmax(I_syn_trace)
            self.assertGreater(total_peak_idx, fast_peak_idx,
                               msg="Combined peak should be shifted right from fast peak")

    # ------------------------------------------------------------------
    # 10. Refractory period test
    # ------------------------------------------------------------------

    def test_refractory_period_holds_voltage(self):
        r"""During refractory period, voltage should be held at reset value."""
        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=800.0 * u.pA,
                t_ref=1.0 * u.ms,
            )
            neuron.init_state()

            # I_e=800 pA is strong; spike well before 300 steps
            n_steps = 300

            def _body(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0.0 * u.pA)
                return (spk, neuron.V.value / u.mV, neuron.refractory_step_count.value)

            results = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            spk_trace, v_trace, refr_trace = results

            spike_steps = [k for k in range(n_steps) if float(spk_trace[k, 0]) > 0]
            self.assertGreater(len(spike_steps), 0, "Should have spiked")
            spike_step = spike_steps[0]

            self.assertGreater(
                int(refr_trace[spike_step, 0]),
                0,
                "Should be refractory after spike"
            )

            v_reset = float(v_trace[spike_step, 0])
            for k in range(spike_step + 1, spike_step + 5):
                v_now = float(v_trace[k, 0])
                self.assertAlmostEqual(
                    v_now, v_reset, places=10,
                    msg=f"Voltage should be held during refractory at step {k}"
                )

    # ------------------------------------------------------------------
    # 11. Multi-receptor test
    # ------------------------------------------------------------------

    def test_three_receptors(self):
        r"""Model with three receptor ports initializes and runs correctly."""
        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                tau_syn_fast=(0.5, 2.0, 5.0),
                tau_syn_slow=(1.0, 4.0, 10.0),
                amp_slow=(0.2, 0.3, 0.5),
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            self.assertEqual(neuron.n_receptors, 3)

            # Inject spike to each receptor
            self._step(neuron, 0, dy_inputs=[(0, 10.0), (1, 5.0), (2, 3.0)])

            # All receptors should have non-zero y1_fast and y1_slow
            for k_rec in range(3):
                y1_fast_val = float((neuron.y1_fast[k_rec].value / (u.pA / u.ms))[0])
                y1_slow_val = float((neuron.y1_slow[k_rec].value / (u.pA / u.ms))[0])
                self.assertGreater(abs(y1_fast_val), 0.0,
                                   msg=f"Receptor {k_rec} should have non-zero y1_fast")
                self.assertGreater(abs(y1_slow_val), 0.0,
                                   msg=f"Receptor {k_rec} should have non-zero y1_slow")

    # ------------------------------------------------------------------
    # 12. Initial membrane potential test
    # ------------------------------------------------------------------

    def test_initial_v_m(self):
        r"""Initial V_m should match the initializer."""
        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()
            v_m = float((neuron.V.value / u.mV)[0])
            self.assertAlmostEqual(v_m, -78.85, places=10)

    # ------------------------------------------------------------------
    # 13. Voltage-dependent threshold trace test (GLIF5)
    # ------------------------------------------------------------------

    def test_glif5_threshold_trace_matches_reference(self):
        r"""GLIF5: threshold components (spike + voltage) match reference trace."""
        dt_val = 0.01
        n_steps = 1000

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                spike_dependent_threshold=True,
                after_spike_currents=True,
                adapting_threshold=True,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=500.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params_double(
                has_theta_spike=True, has_asc=True, has_theta_voltage=True,
                dt=dt_val,
            )
            params['I_e'] = 500.0
            state = _make_ref_state_double(params, V_abs=-78.85)

            def _body(k):
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0.0 * u.pA)
                return None

            brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            for k in range(n_steps):
                _ref_glif_psc_double_alpha_step(state, params, dt_val)

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

    # ------------------------------------------------------------------
    # 14. Synaptic current y1/y2 trace against reference
    # ------------------------------------------------------------------

    def test_synaptic_y1_y2_traces_match_reference(self):
        r"""Verify y1_fast/y2_fast and y1_slow/y2_slow traces match reference."""
        dt_val = 0.01
        n_steps = 300
        tau_syn_fast_val = 2.0
        tau_syn_slow_val = 6.0
        amp_slow_val = 0.3

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                tau_syn_fast=(tau_syn_fast_val,),
                tau_syn_slow=(tau_syn_slow_val,),
                amp_slow=(amp_slow_val,),
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            params = _make_ref_params_double(
                tau_syn_fast=(tau_syn_fast_val,),
                tau_syn_slow=(tau_syn_slow_val,),
                amp_slow=(amp_slow_val,),
                has_theta_spike=False, has_asc=False, has_theta_voltage=False,
                dt=dt_val,
            )
            state = _make_ref_state_double(params, V_abs=-78.85)

            w_vals = jnp.zeros(n_steps, dtype=jnp.float64).at[10].set(1.0)

            def _body(k):
                neuron.add_delta_input('fl', w_vals[k] * u.pA, label='receptor_0')
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0.0 * u.pA)
                return (
                    neuron.y1_fast[0].value / (u.pA / u.ms),
                    neuron.y2_fast[0].value / u.pA,
                    neuron.y1_slow[0].value / (u.pA / u.ms),
                    neuron.y2_slow[0].value / u.pA,
                )

            results = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            y1_fast_model = [float(results[0][k, 0]) for k in range(n_steps)]
            y2_fast_model = [float(results[1][k, 0]) for k in range(n_steps)]
            y1_slow_model = [float(results[2][k, 0]) for k in range(n_steps)]
            y2_slow_model = [float(results[3][k, 0]) for k in range(n_steps)]

            y1_fast_ref = []
            y2_fast_ref = []
            y1_slow_ref = []
            y2_slow_ref = []
            for k in range(n_steps):
                if k == 10:
                    _ref_glif_psc_double_alpha_step(state, params, dt_val, spike_weights=[1.0])
                else:
                    _ref_glif_psc_double_alpha_step(state, params, dt_val)
                y1_fast_ref.append(state['y1_fast'][0])
                y2_fast_ref.append(state['y2_fast'][0])
                y1_slow_ref.append(state['y1_slow'][0])
                y2_slow_ref.append(state['y2_slow'][0])

            for k in range(n_steps):
                self.assertAlmostEqual(
                    y1_fast_model[k], y1_fast_ref[k], places=10,
                    msg=f"y1_fast step {k}: model={y1_fast_model[k]}, ref={y1_fast_ref[k]}"
                )
                self.assertAlmostEqual(
                    y2_fast_model[k], y2_fast_ref[k], places=10,
                    msg=f"y2_fast step {k}: model={y2_fast_model[k]}, ref={y2_fast_ref[k]}"
                )
                self.assertAlmostEqual(
                    y1_slow_model[k], y1_slow_ref[k], places=10,
                    msg=f"y1_slow step {k}: model={y1_slow_model[k]}, ref={y1_slow_ref[k]}"
                )
                self.assertAlmostEqual(
                    y2_slow_model[k], y2_slow_ref[k], places=10,
                    msg=f"y2_slow step {k}: model={y2_slow_model[k]}, ref={y2_slow_ref[k]}"
                )

    # ------------------------------------------------------------------
    # 15. Multiple synaptic time constants with different inputs
    # ------------------------------------------------------------------

    def test_multi_receptor_voltage_trace(self):
        r"""Two receptors with different tau_syn and inputs match reference."""
        dt_val = 0.01
        n_steps = 500

        tau_syn_fast_list = (1.0, 2.0)
        tau_syn_slow_list = (2.0, 4.0)
        amp_slow_list = (0.5, 0.3)

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                tau_syn_fast=tau_syn_fast_list,
                tau_syn_slow=tau_syn_slow_list,
                amp_slow=amp_slow_list,
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            params = _make_ref_params_double(
                tau_syn_fast=tau_syn_fast_list,
                tau_syn_slow=tau_syn_slow_list,
                amp_slow=amp_slow_list,
                has_theta_spike=False, has_asc=False, has_theta_voltage=False,
                dt=dt_val,
            )
            state = _make_ref_state_double(params, V_abs=-78.85)

            w_vals_rec0 = jnp.zeros(n_steps, dtype=jnp.float64).at[20].set(5.0)
            w_vals_rec1 = jnp.zeros(n_steps, dtype=jnp.float64).at[50].set(3.0)

            def _body(k):
                neuron.add_delta_input('fl0', w_vals_rec0[k] * u.pA, label='receptor_0')
                neuron.add_delta_input('fl1', w_vals_rec1[k] * u.pA, label='receptor_1')
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0.0 * u.pA)
                return neuron.V.value / u.mV

            v_model_all = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            v_model = [float(v_model_all[k, 0]) for k in range(n_steps)]

            v_ref = []
            for k in range(n_steps):
                if k == 20:
                    _ref_glif_psc_double_alpha_step(state, params, dt_val, spike_weights=[5.0, 0.0])
                elif k == 50:
                    _ref_glif_psc_double_alpha_step(state, params, dt_val, spike_weights=[0.0, 3.0])
                else:
                    _ref_glif_psc_double_alpha_step(state, params, dt_val)
                v_ref.append(state['V_rel'] + params['E_L'])

            for k in range(n_steps):
                self.assertAlmostEqual(
                    v_model[k], v_ref[k], places=6,
                    msg=f"Step {k}: model={v_model[k]}, ref={v_ref[k]}"
                )

    # ------------------------------------------------------------------
    # 16. GLIF1 voltage trace matches reference exactly
    # ------------------------------------------------------------------

    def test_glif1_voltage_trace_with_constant_current(self):
        r"""GLIF1 with constant current: full voltage trace matches reference."""
        dt_val = 0.01
        n_steps = 800

        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
                I_e=400.0 * u.pA,
            )
            neuron.init_state()

            params = _make_ref_params_double(
                has_theta_spike=False, has_asc=False, has_theta_voltage=False,
                dt=dt_val,
            )
            params['I_e'] = 400.0
            state = _make_ref_state_double(params, V_abs=-78.85)

            def _body(k):
                with brainstate.environ.context(t=k * self.dt):
                    spk = neuron.update(x=0.0 * u.pA)
                return (spk, neuron.V.value / u.mV)

            results = brainstate.transform.for_loop(_body, jnp.arange(n_steps))
            spk_trace = results[0]
            v_trace = results[1]
            model_spikes = [k for k in range(n_steps) if float(spk_trace[k, 0]) > 0]
            v_model = [float(v_trace[k, 0]) for k in range(n_steps)]

            ref_spikes = []
            v_ref = []
            for k in range(n_steps):
                spiked = _ref_glif_psc_double_alpha_step(state, params, dt_val)
                if spiked:
                    ref_spikes.append(k)
                v_ref.append(state['V_rel'] + params['E_L'])

            self.assertEqual(model_spikes, ref_spikes)

            for k in range(n_steps):
                self.assertAlmostEqual(
                    v_model[k], v_ref[k], places=8,
                    msg=f"Step {k}: model={v_model[k]:.10f}, ref={v_ref[k]:.10f}"
                )

    # ------------------------------------------------------------------
    # 17. get_I_syn methods test
    # ------------------------------------------------------------------

    def test_get_I_syn_methods(self):
        r"""Test get_I_syn, get_I_syn_fast, get_I_syn_slow methods."""
        with brainstate.environ.context(dt=self.dt):
            neuron = glif_psc_double_alpha(
                1,
                tau_syn_fast=(2.0,),
                tau_syn_slow=(6.0,),
                amp_slow=(0.3,),
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-78.85 * u.mV),
            )
            neuron.init_state()

            # Inject spike at step 0, then run steps 0..49 via for_loop
            w_vals = jnp.zeros(50, dtype=jnp.float64).at[0].set(1.0)

            def _body(k):
                neuron.add_delta_input('fl', w_vals[k] * u.pA, label='receptor_0')
                with brainstate.environ.context(t=k * self.dt):
                    neuron.update(x=0.0 * u.pA)
                return None

            brainstate.transform.for_loop(_body, jnp.arange(50))

            I_syn = neuron.get_I_syn()
            I_syn_fast = neuron.get_I_syn_fast()
            I_syn_slow = neuron.get_I_syn_slow()

            # Verify I_syn = I_syn_fast + I_syn_slow
            I_syn_val = float((I_syn / u.pA)[0])
            I_syn_fast_val = float((I_syn_fast / u.pA)[0])
            I_syn_slow_val = float((I_syn_slow / u.pA)[0])

            self.assertAlmostEqual(
                I_syn_val, I_syn_fast_val + I_syn_slow_val, places=10,
                msg="I_syn should equal I_syn_fast + I_syn_slow"
            )

    # ------------------------------------------------------------------
    # 18. Synaptic current depolarizes membrane
    # ------------------------------------------------------------------

    def test_synaptic_current_depolarizes(self):
        r"""Positive synaptic current input should depolarize membrane."""
        with brainstate.environ.context(dt=self.dt):
            base = glif_psc_double_alpha(
                1,
                tau_syn_fast=(2.0,),
                tau_syn_slow=(6.0,),
                amp_slow=(0.3,),
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
            )
            base.init_state()

            exc = glif_psc_double_alpha(
                1,
                tau_syn_fast=(2.0,),
                tau_syn_slow=(6.0,),
                amp_slow=(0.3,),
                V_th=100.0 * u.mV,
                spike_dependent_threshold=False,
                after_spike_currents=False,
                adapting_threshold=False,
                V_initializer=braintools.init.Constant(-60.0 * u.mV),
            )
            exc.init_state()

            # exc gets a synaptic spike at step 0; base does not — run all 101 steps in for_loop
            w_vals = jnp.zeros(101, dtype=jnp.float64).at[0].set(50.0)

            def _body(k):
                exc.add_delta_input('fl', w_vals[k] * u.pA, label='receptor_0')
                with brainstate.environ.context(t=k * self.dt):
                    base.update(x=0.0 * u.pA)
                    exc.update(x=0.0 * u.pA)
                return None

            brainstate.transform.for_loop(_body, jnp.arange(101))

            v_base = float((base.V.value / u.mV)[0])
            v_exc = float((exc.V.value / u.mV)[0])

            self.assertGreater(v_exc, v_base)


if __name__ == '__main__':
    unittest.main()
