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

r"""
Tests for the ``astrocyte_lr_1994`` model.

This test module validates the brainpy.state implementation against:
1. NEST default parameter values
2. SciPy ODEINT reference solution (same approach as NEST test suite)
3. NEST simulator output (when available on the platform)
4. Subthreshold and input-driven dynamics

All tests use float64 precision and CPU backend.
"""

import math
import unittest

import brainstate
import brainunit as u
import numpy as np
from scipy.integrate import odeint

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_neuron.astrocyte_lr_1994 import astrocyte_lr_1994


def _reference_dynamics(y, t, params, spike_times=None, spike_weights=None):
    r"""ODE RHS for the astrocyte model, matching NEST's astrocyte_lr_1994_dynamics.

    Parameters
    ----------
    y : array [IP3, Ca, h_IP3R]
    t : float (ms)
    params : dict of model parameters
    spike_times : list of spike arrival times (ms)
    spike_weights : list of corresponding weights

    Returns
    -------
    dydt : array [dIP3/dt, dCa/dt, dh/dt]
    """
    ip3, ca, h_ip3r = y

    Ca_tot = params['Ca_tot']
    IP3_0 = params['IP3_0']
    Kd_IP3_1 = params['Kd_IP3_1']
    Kd_IP3_2 = params['Kd_IP3_2']
    Kd_act = params['Kd_act']
    Kd_inh = params['Kd_inh']
    Km_SERCA = params['Km_SERCA']
    k_IP3R = params['k_IP3R']
    rate_IP3R = params['rate_IP3R']
    rate_L = params['rate_L']
    rate_SERCA = params['rate_SERCA']
    ratio_ER_cyt = params['ratio_ER_cyt']
    tau_IP3 = params['tau_IP3']

    calc = max(0.0, min(ca, Ca_tot))

    alpha_h = k_IP3R * Kd_inh * (ip3 + Kd_IP3_1) / (ip3 + Kd_IP3_2)
    beta_h = k_IP3R * calc

    J_pump = rate_SERCA * calc ** 2 / (Km_SERCA ** 2 + calc ** 2)
    m_inf = ip3 / (ip3 + Kd_IP3_1)
    n_inf = calc / (calc + Kd_act)
    calc_ER = (Ca_tot - calc) / ratio_ER_cyt
    J_leak = ratio_ER_cyt * rate_L * (calc_ER - calc)
    J_channel = (ratio_ER_cyt * rate_IP3R
                 * m_inf ** 3 * n_inf ** 3 * h_ip3r ** 3
                 * (calc_ER - calc))

    dip3 = (IP3_0 - ip3) / tau_IP3
    dca = J_channel - J_pump + J_leak  # J_noise = 0 for continuous ODE
    dh = alpha_h * (1.0 - h_ip3r) - beta_h * h_ip3r

    return [dip3, dca, dh]


def _generate_reference_trace(params, y0, simtime_ms, dt_ms,
                              spike_times=None, spike_weights=None):
    r"""Generate reference traces using SciPy ODEINT with spike injection.

    This matches NEST's approach:
    - Integrate ODEs continuously between spikes
    - Apply instantaneous IP3 jumps at spike times
    - Clamp calcium to [0, Ca_tot] after each step

    Parameters
    ----------
    params : dict
    y0 : array [IP3_0, Ca_0, h_0]
    simtime_ms : float
    dt_ms : float
    spike_times : list of float (ms), times when spikes arrive
    spike_weights : list of float, corresponding weights

    Returns
    -------
    times : array of shape (N,)
    IP3 : array of shape (N,)
    Ca : array of shape (N,)
    h_IP3R : array of shape (N,)
    """
    if spike_times is None:
        spike_times = []
    if spike_weights is None:
        spike_weights = []

    n_steps = int(round(simtime_ms / dt_ms))
    times = np.zeros(n_steps)
    IP3_trace = np.zeros(n_steps)
    Ca_trace = np.zeros(n_steps)
    h_trace = np.zeros(n_steps)

    dftype = brainstate.environ.dftype()
    y = np.array(y0, dtype=dftype)
    t_current = 0.0

    for i in range(n_steps):
        # Integrate one step
        t_span = [t_current, t_current + dt_ms]
        sol = odeint(_reference_dynamics, y, t_span, args=(params,),
                     rtol=1e-8, atol=1e-10)
        y = sol[-1].copy()

        # Clamp calcium
        y[1] = max(0.0, min(y[1], params['Ca_tot']))

        t_current += dt_ms

        # Apply spikes that arrive at this time step
        for st, sw in zip(spike_times, spike_weights):
            if abs(t_current - st) < dt_ms * 0.5:
                y[0] += params['delta_IP3'] * sw

        times[i] = t_current
        IP3_trace[i] = y[0]
        Ca_trace[i] = y[1]
        h_trace[i] = y[2]

    return times, IP3_trace, Ca_trace, h_trace


# Default NEST parameters
NEST_DEFAULTS = {
    'Ca_tot': 2.0,
    'IP3_0': 0.16,
    'Kd_IP3_1': 0.13,
    'Kd_IP3_2': 0.9434,
    'Kd_act': 0.08234,
    'Kd_inh': 1.049,
    'Km_SERCA': 0.1,
    'SIC_scale': 1.0,
    'SIC_th': 0.19669,
    'delta_IP3': 0.0002,
    'k_IP3R': 0.0002,
    'rate_IP3R': 0.006,
    'rate_L': 0.00011,
    'rate_SERCA': 0.0009,
    'ratio_ER_cyt': 0.185,
    'tau_IP3': 7142.0,
}


class TestAstrocyteLR1994DefaultParameters(unittest.TestCase):
    r"""Test that default parameters match NEST."""

    def test_nest_defaults(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            astro = astrocyte_lr_1994(1)
            self.assertEqual(astro.Ca_tot, 2.0)
            self.assertEqual(astro.IP3_0, 0.16)
            self.assertEqual(astro.Kd_IP3_1, 0.13)
            self.assertEqual(astro.Kd_IP3_2, 0.9434)
            self.assertEqual(astro.Kd_act, 0.08234)
            self.assertEqual(astro.Kd_inh, 1.049)
            self.assertEqual(astro.Km_SERCA, 0.1)
            self.assertEqual(astro.SIC_scale, 1.0)
            self.assertEqual(astro.SIC_th, 0.19669)
            self.assertEqual(astro.delta_IP3, 0.0002)
            self.assertEqual(astro.k_IP3R, 0.0002)
            self.assertEqual(astro.rate_IP3R, 0.006)
            self.assertEqual(astro.rate_L, 0.00011)
            self.assertEqual(astro.rate_SERCA, 0.0009)
            self.assertEqual(astro.ratio_ER_cyt, 0.185)
            self.assertEqual(astro.tau_IP3, 7142.0)

    def test_default_initial_state(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""IP3 initialized to IP3_0, Ca to 0.073, h to 0.793 (NEST defaults)."""
            astro = astrocyte_lr_1994(1)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state()
            np.testing.assert_allclose(
                float(np.asarray(astro.IP3.value).squeeze()), 0.16, rtol=1e-12
            )
            np.testing.assert_allclose(
                float(np.asarray(astro.Ca.value).squeeze()), 0.073, rtol=1e-12
            )
            np.testing.assert_allclose(
                float(np.asarray(astro.h_IP3R.value).squeeze()), 0.793, rtol=1e-12
            )


class TestAstrocyteLR1994ParameterValidation(unittest.TestCase):
    r"""Test parameter validation constraints."""

    def test_Ca_tot_must_be_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                astrocyte_lr_1994(1, Ca_tot=0.0)
            with self.assertRaises(ValueError):
                astrocyte_lr_1994(1, Ca_tot=-1.0)

    def test_IP3_0_must_be_nonnegative(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                astrocyte_lr_1994(1, IP3_0=-0.1)

    def test_tau_IP3_must_be_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                astrocyte_lr_1994(1, tau_IP3=0.0)

    def test_SIC_scale_must_be_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                astrocyte_lr_1994(1, SIC_scale=0.0)

    def test_ratio_ER_cyt_must_be_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                astrocyte_lr_1994(1, ratio_ER_cyt=-0.1)

    def test_Kd_IP3_1_must_be_positive(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            with self.assertRaises(ValueError):
                astrocyte_lr_1994(1, Kd_IP3_1=0.0)


class TestAstrocyteLR1994Dynamics(unittest.TestCase):
    r"""Test astrocyte dynamics against SciPy ODEINT reference solution."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1  # ms
        self.dt_q = 0.1 * u.ms

    def test_free_decay_matches_odeint(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Test passive dynamics (no input) against ODEINT reference.

            Uses NEST test initial conditions: IP3=1.0, Ca=1.0, h=1.0.
            """
            simtime = 50.0  # ms
            y0 = [1.0, 1.0, 1.0]

            # Generate reference
            ref_t, ref_ip3, ref_ca, ref_h = _generate_reference_trace(
                NEST_DEFAULTS, y0, simtime, self.dt
            )

            # Run brainpy model
            astro = astrocyte_lr_1994(
                1,
                IP3_initializer=1.0,
                Ca_initializer=1.0,
                h_IP3R_initializer=1.0,
            )
            with brainstate.environ.context(dt=self.dt_q):
                astro.init_state()

            n_steps = int(round(simtime / self.dt))

            def body(_):
                astro.update()
                return astro.IP3.value, astro.Ca.value, astro.h_IP3R.value

            results = brainstate.transform.for_loop(body, np.zeros(n_steps))
            bp_ip3 = np.asarray(results[0]).squeeze(axis=-1)
            bp_ca = np.asarray(results[1]).squeeze(axis=-1)
            bp_h = np.asarray(results[2]).squeeze(axis=-1)

            np.testing.assert_allclose(bp_ip3, ref_ip3, rtol=1e-4, atol=1e-8,
                                       err_msg="IP3 mismatch in free decay")
            np.testing.assert_allclose(bp_ca, ref_ca, rtol=1e-4, atol=1e-8,
                                       err_msg="Ca mismatch in free decay")
            np.testing.assert_allclose(bp_h, ref_h, rtol=1e-4, atol=1e-8,
                                       err_msg="h_IP3R mismatch in free decay")

    def test_spike_input_matches_odeint(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Test dynamics with a spike at t=10ms, matching NEST test conditions.

            NEST test uses: IP3=1.0, Ca=1.0, h=1.0, spike at t=10ms with weight=1.0.
            """
            simtime = 100.0  # ms
            y0 = [1.0, 1.0, 1.0]
            spike_time = 10.0  # ms
            spike_weight = 1.0

            # Generate reference
            ref_t, ref_ip3, ref_ca, ref_h = _generate_reference_trace(
                NEST_DEFAULTS, y0, simtime, self.dt,
                spike_times=[spike_time],
                spike_weights=[spike_weight]
            )

            # Run brainpy model
            astro = astrocyte_lr_1994(
                1,
                IP3_initializer=1.0,
                Ca_initializer=1.0,
                h_IP3R_initializer=1.0,
            )
            with brainstate.environ.context(dt=self.dt_q):
                astro.init_state()

            n_steps = int(round(simtime / self.dt))
            spike_step = int(round(spike_time / self.dt))

            spike_weights_all = np.zeros(n_steps)
            spike_weights_all[spike_step - 1] = spike_weight

            def body(sw):
                astro.update(spike_weights=sw)
                return astro.IP3.value, astro.Ca.value, astro.h_IP3R.value

            results = brainstate.transform.for_loop(body, spike_weights_all)
            bp_ip3 = np.asarray(results[0]).squeeze(axis=-1)
            bp_ca = np.asarray(results[1]).squeeze(axis=-1)
            bp_h = np.asarray(results[2]).squeeze(axis=-1)

            np.testing.assert_allclose(bp_ip3, ref_ip3, rtol=1e-4, atol=1e-8,
                                       err_msg="IP3 mismatch with spike input")
            np.testing.assert_allclose(bp_ca, ref_ca, rtol=1e-4, atol=1e-8,
                                       err_msg="Ca mismatch with spike input")
            np.testing.assert_allclose(bp_h, ref_h, rtol=1e-4, atol=1e-8,
                                       err_msg="h_IP3R mismatch with spike input")

    def test_default_state_steady(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Default initial state should be near steady state."""
            simtime = 100.0  # ms
            y0 = [0.16, 0.073, 0.793]

            astro = astrocyte_lr_1994(1)
            with brainstate.environ.context(dt=self.dt_q):
                astro.init_state()

            n_steps = int(round(simtime / self.dt))

            def body(_):
                astro.update()
                return astro.IP3.value

            brainstate.transform.for_loop(body, np.zeros(n_steps))

            # Should not change much from initial (near equilibrium)
            np.testing.assert_allclose(
                float(np.asarray(astro.IP3.value).squeeze()), 0.16, rtol=1e-3,
                err_msg="IP3 should remain near baseline"
            )
            np.testing.assert_allclose(
                float(np.asarray(astro.Ca.value).squeeze()), 0.073, atol=0.01,
                err_msg="Ca should remain near initial"
            )
            np.testing.assert_allclose(
                float(np.asarray(astro.h_IP3R.value).squeeze()), 0.793, atol=0.01,
                err_msg="h_IP3R should remain near initial"
            )


class TestAstrocyteLR1994SIC(unittest.TestCase):
    r"""Test SIC output generation."""

    def test_sic_zero_below_threshold(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""SIC should be zero when Ca < SIC_th."""
            astro = astrocyte_lr_1994(1)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state()
                # Default Ca=0.073 << SIC_th=0.19669
                astro.update()
            self.assertEqual(float(np.asarray(astro.SIC.value).squeeze()), 0.0)

    def test_sic_positive_above_threshold(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""SIC should be positive when Ca > SIC_th + 1/1000 (nM conversion)."""
            # Set Ca above threshold such that (Ca - SIC_th)*1000 > 1
            # Need Ca > SIC_th + 0.001 = 0.19769
            astro = astrocyte_lr_1994(
                1,
                Ca_initializer=0.5,  # well above threshold
            )
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state()
                sic = astro.update()
            self.assertGreater(float(np.asarray(sic).squeeze()), 0.0)

    def test_sic_formula_exact(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Verify SIC = SIC_scale * log((Ca - SIC_th) * 1000) for Ca above threshold."""
            ca_val = 0.5
            sic_th = 0.19669
            sic_scale = 1.0
            expected = sic_scale * math.log((ca_val - sic_th) * 1000.0)

            astro = astrocyte_lr_1994(
                1,
                Ca_initializer=ca_val,
                IP3_initializer=0.16,
                h_IP3R_initializer=0.793,
            )
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state()
                # After one tiny step, Ca is close to initial
                sic = astro.update()
            # SIC computed from post-integration Ca, which may differ slightly
            actual_ca = float(np.asarray(astro.Ca.value).squeeze())
            actual_expected = sic_scale * math.log(
                max(1.0, (actual_ca - sic_th) * 1000.0)
            )
            np.testing.assert_allclose(
                float(np.asarray(sic).squeeze()), actual_expected, rtol=1e-10,
                err_msg="SIC formula mismatch"
            )

    def test_sic_scales_with_parameter(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""SIC should scale with SIC_scale."""
            ca_val = 0.5
            astro1 = astrocyte_lr_1994(1, Ca_initializer=ca_val, SIC_scale=1.0)
            astro2 = astrocyte_lr_1994(1, Ca_initializer=ca_val, SIC_scale=2.0)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro1.init_state()
                astro2.init_state()
                sic1 = astro1.update()
                sic2 = astro2.update()
            # Both have same dynamics, so Ca at read time is same
            # SIC_scale ratio should hold approximately
            self.assertGreater(float(np.asarray(sic2).squeeze()), float(np.asarray(sic1).squeeze()))


class TestAstrocyteLR1994SpikeInput(unittest.TestCase):
    r"""Test spike input handling."""

    def test_spike_increases_ip3(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""A spike should instantaneously increase IP3 by delta_IP3 * weight."""
            astro = astrocyte_lr_1994(1, delta_IP3=0.5)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state()
                ip3_before = float(np.asarray(astro.IP3.value).squeeze())
                astro.update(spike_weights=2.0)
                ip3_after = float(np.asarray(astro.IP3.value).squeeze())

            # IP3 increase should include delta_IP3 * weight = 0.5 * 2.0 = 1.0
            # plus ODE evolution over one dt step
            # The ODE term is small, so the jump should be approximately 1.0
            ip3_jump = ip3_after - ip3_before
            # Should be close to 1.0 (minus small ODE decay)
            self.assertGreater(ip3_jump, 0.9)

    def test_multiple_spikes_accumulate(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Multiple spikes in same step accumulate additively."""
            astro = astrocyte_lr_1994(1, delta_IP3=0.1)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state()
                ip3_before = float(np.asarray(astro.IP3.value).squeeze())
                # Total weight = 3.0 -> IP3 jump = 0.1 * 3.0 = 0.3
                astro.update(spike_weights=3.0)
                ip3_after = float(np.asarray(astro.IP3.value).squeeze())

            ip3_jump = ip3_after - ip3_before
            self.assertGreater(ip3_jump, 0.2)


class TestAstrocyteLR1994Emission(unittest.TestCase):
    r"""Seam-(H) continuous emission declaration + the ``sum_delta_inputs`` spike path.

    The astrocyte is the *emitter* of the slow-inward current (SIC) on the JAX
    substrate: it declares ``_emission_attr='SIC'`` so the Simulator allocates an
    emission holder and captures ``astro.SIC.value`` each step (the value the
    ``sic_connection`` projection reads). Its presynaptic spike input now flows
    through ``sum_delta_inputs`` so the Simulator's neuron->astrocyte delta deposits
    reach IP3, while a bare ``update(spike_weights=w)`` call stays backward-compatible.
    """

    def test_declares_continuous_sic_emission_markers(self):
        # Seam-(H) class metadata the Simulator/connection wiring keys off.
        self.assertEqual(astrocyte_lr_1994._emission_attr, 'SIC')
        self.assertIs(astrocyte_lr_1994._emission_continuous, True)
        self.assertIs(astrocyte_lr_1994._emission_current, True)
        self.assertEqual(astrocyte_lr_1994._emission_current_label, 'I_SIC')

    def test_registered_delta_input_reaches_ip3(self):
        # A delta input deposited externally (as the Simulator's neuron->astro
        # projection does) must reach IP3 via sum_delta_inputs, even when the
        # update() kwarg is 0.0. Compare against a no-input control so the ODE
        # evolution cancels and only the delta_IP3 * weight jump remains.
        with brainstate.environ.context(dt=0.1 * u.ms):
            ctrl = astrocyte_lr_1994(1, delta_IP3=0.5)
            ctrl.init_state()
            ctrl.update(spike_weights=0.0)
            ip3_ctrl = float(np.asarray(ctrl.IP3.value).squeeze())

            driven = astrocyte_lr_1994(1, delta_IP3=0.5)
            driven.init_state()
            driven.add_delta_input('pre_spike', 2.0)  # weight 2.0 spike deposit
            driven.update(spike_weights=0.0)
            ip3_driven = float(np.asarray(driven.IP3.value).squeeze())

        # delta_IP3 * weight = 0.5 * 2.0 = 1.0 over the no-input control.
        self.assertGreater(ip3_driven - ip3_ctrl, 0.9)

    def test_kwarg_and_delta_input_sum_additively(self):
        # sum_delta_inputs(spike_weights) = spike_weights + registered deposits.
        with brainstate.environ.context(dt=0.1 * u.ms):
            base = astrocyte_lr_1994(1, delta_IP3=0.5)
            base.init_state()
            base.update(spike_weights=0.0)
            ip3_base = float(np.asarray(base.IP3.value).squeeze())

            both = astrocyte_lr_1994(1, delta_IP3=0.5)
            both.init_state()
            both.add_delta_input('pre_spike', 1.0)
            both.update(spike_weights=1.0)  # kwarg 1.0 + deposit 1.0 -> 2.0
            ip3_both = float(np.asarray(both.IP3.value).squeeze())

        # (1.0 + 1.0) * delta_IP3 = 1.0 over the no-input control.
        self.assertGreater(ip3_both - ip3_base, 0.9)


class TestAstrocyteLR1994CurrentInput(unittest.TestCase):
    r"""Test current (J_noise) input handling."""

    def test_current_input_affects_calcium(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""External current input should affect calcium concentration."""
            astro = astrocyte_lr_1994(1)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state()
                # First step with no current
                astro.update(J_ext=0.0)
                ca_no_input = float(np.asarray(astro.Ca.value).squeeze())

            astro2 = astrocyte_lr_1994(1)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro2.init_state()
                # First step sets J_noise for next step (one-step delay)
                astro2.update(J_ext=1.0)
                # Second step uses J_noise=1.0
                astro2.update(J_ext=0.0)
                ca_with_input = float(np.asarray(astro2.Ca.value).squeeze())

            # The current should have pushed calcium up
            self.assertGreater(ca_with_input, ca_no_input)


class TestAstrocyteLR1994BatchShape(unittest.TestCase):
    r"""Test that the model handles different population shapes."""

    def test_scalar_shape(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            astro = astrocyte_lr_1994(1)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state()
                sic = astro.update()
            self.assertEqual(astro.IP3.value.shape, (1,))
            self.assertEqual(sic.shape, (1,))

    def test_vector_shape(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            astro = astrocyte_lr_1994(5)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state()
                sic = astro.update()
            self.assertEqual(astro.IP3.value.shape, (5,))
            self.assertEqual(sic.shape, (5,))

    def test_batch_shape(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            astro = astrocyte_lr_1994(3)
            with brainstate.environ.context(dt=0.1 * u.ms):
                astro.init_state(batch_size=2)
                sic = astro.update()
            self.assertEqual(astro.IP3.value.shape, (2, 3))
            self.assertEqual(sic.shape, (2, 3))


class TestAstrocyteLR1994NestComparison(unittest.TestCase):
    r"""Compare against NEST simulator if available."""

    def test_matches_nest_reference_data(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Compare against NEST test reference data (test_astrocyte.dat).

            The reference data was generated using SciPy ODEINT with:
            - simtime = 100 ms, dt = 0.1 ms
            - Initial: IP3=1.0, Ca=1.0, h=1.0
            - Spike at t=10ms with weight=1.0
            """
            import os
            ref_path = os.path.join(
                '/mnt/d/codes/githubs/nest-simulator',
                'testsuite', 'pytests', 'test_astrocyte.dat'
            )
            if not os.path.exists(ref_path):
                self.skipTest("NEST reference data not found")

            ref_data = np.loadtxt(ref_path)
            ref_t = ref_data[:, 0]
            ref_ip3 = ref_data[:, 1]
            ref_ca = ref_data[:, 2]
            ref_h = ref_data[:, 3]

            dt = 0.1  # ms
            dt_q = 0.1 * u.ms
            simtime = 100.0  # ms
            n_steps = len(ref_t)

            astro = astrocyte_lr_1994(
                1,
                IP3_initializer=1.0,
                Ca_initializer=1.0,
                h_IP3R_initializer=1.0,
            )

            with brainstate.environ.context(dt=dt_q):
                astro.init_state()

            # Spike at t=10ms -> arrives at step index 99 (t=10.0 after 100 steps of 0.1ms)
            spike_step = int(round(10.0 / dt))  # step 100

            spike_weights_all = np.zeros(n_steps)
            spike_weights_all[spike_step - 1] = 1.0

            def body(sw):
                astro.update(spike_weights=sw)
                return astro.IP3.value, astro.Ca.value, astro.h_IP3R.value

            results = brainstate.transform.for_loop(body, spike_weights_all)
            bp_ip3 = np.asarray(results[0]).squeeze(axis=-1)
            bp_ca = np.asarray(results[1]).squeeze(axis=-1)
            bp_h = np.asarray(results[2]).squeeze(axis=-1)

            np.testing.assert_allclose(
                bp_ip3, ref_ip3, rtol=1e-3, atol=1e-6,
                err_msg="IP3 mismatch vs NEST reference data"
            )
            np.testing.assert_allclose(
                bp_ca, ref_ca, rtol=1e-3, atol=1e-6,
                err_msg="Ca mismatch vs NEST reference data"
            )
            np.testing.assert_allclose(
                bp_h, ref_h, rtol=1e-3, atol=1e-6,
                err_msg="h_IP3R mismatch vs NEST reference data"
            )

    def test_matches_nest_simulator(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            r"""Run identical simulation in NEST and compare.

            Skip if NEST is not installed.
            """
            try:
                import nest
            except ImportError:
                self.skipTest("NEST simulator not available")

            nest.ResetKernel()
            nest.resolution = 0.1

            simtime = 50.0  # ms
            dt = 0.1
            dt_q = 0.1 * u.ms

            astro_nest = nest.Create(
                "astrocyte_lr_1994",
                params={"IP3": 0.5, "Ca_astro": 0.5, "h_IP3R": 0.5}
            )
            mm = nest.Create(
                "multimeter",
                {"interval": dt, "record_from": ["IP3", "Ca_astro", "h_IP3R"]}
            )
            nest.Connect(mm, astro_nest)
            nest.Simulate(simtime)

            nest_ip3 = mm.events["IP3"]
            nest_ca = mm.events["Ca_astro"]
            nest_h = mm.events["h_IP3R"]

            # Run brainpy model
            astro_bp = astrocyte_lr_1994(
                1,
                IP3_initializer=0.5,
                Ca_initializer=0.5,
                h_IP3R_initializer=0.5,
            )
            with brainstate.environ.context(dt=dt_q):
                astro_bp.init_state()

            n_steps = len(nest_ip3)

            def body(_):
                astro_bp.update()
                return astro_bp.IP3.value, astro_bp.Ca.value, astro_bp.h_IP3R.value

            results = brainstate.transform.for_loop(body, np.zeros(n_steps))
            bp_ip3 = np.asarray(results[0]).squeeze(axis=-1)
            bp_ca = np.asarray(results[1]).squeeze(axis=-1)
            bp_h = np.asarray(results[2]).squeeze(axis=-1)

            np.testing.assert_allclose(
                bp_ip3, nest_ip3, rtol=1e-3, atol=1e-6,
                err_msg="IP3 mismatch vs NEST simulator"
            )
            np.testing.assert_allclose(
                bp_ca, nest_ca, rtol=1e-3, atol=1e-6,
                err_msg="Ca mismatch vs NEST simulator"
            )
            np.testing.assert_allclose(
                bp_h, nest_h, rtol=1e-3, atol=1e-6,
                err_msg="h_IP3R mismatch vs NEST simulator"
            )


if __name__ == '__main__':
    unittest.main()
