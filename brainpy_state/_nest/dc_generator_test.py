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
Tests for the dc_generator stimulation device.

Validates the brainpy.state ``dc_generator`` against:
  1. Expected timing and amplitude behavior (unit tests).
  2. Analytical solutions for the iaf_psc_delta membrane potential
     driven by constant DC current.
  3. The NEST simulator ``dc_generator`` + ``iaf_psc_delta`` reference
     (skipped when NEST is not installed).
"""

import math
import unittest

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import dc_generator, iaf_psc_delta


# ---------------------------------------------------------------------------
# Helper: run a brainpy.state simulation with dc_generator + iaf_psc_delta
# ---------------------------------------------------------------------------

def _run_bp_simulation(dt_ms, simtime_ms, amplitude_pA, start_ms=0.0,
                       stop_ms=None, origin_ms=0.0, neuron_params=None):
    r"""Run a simulation and return the V_m trace (one value per step).

    Parameters
    ----------
    dt_ms : float
        Time step in ms.
    simtime_ms : float
        Total simulation time in ms.
    amplitude_pA : float
        DC amplitude in pA.
    start_ms : float
        DC start time in ms.
    stop_ms : float or None
        DC stop time in ms (None = infinite).
    origin_ms : float
        DC origin in ms.
    neuron_params : dict or None
        Extra keyword arguments forwarded to iaf_psc_delta.

    Returns
    -------
    vm : np.ndarray, shape (n_steps,)
        Membrane potential in mV after each step.
    """
    if neuron_params is None:
        neuron_params = {}

    n_steps = int(round(simtime_ms / dt_ms))
    dt = dt_ms * u.ms

    with brainstate.environ.context(dt=dt):
        dc = dc_generator(
            amplitude=amplitude_pA * u.pA,
            start=start_ms * u.ms,
            stop=stop_ms * u.ms if stop_ms is not None else None,
            origin=origin_ms * u.ms,
        )
        neuron = iaf_psc_delta(1, **neuron_params)
        neuron.init_state()

        vm = np.empty(n_steps, dtype=np.float64)
        for step in range(n_steps):
            t = step * dt
            with brainstate.environ.context(t=t):
                current = dc.update()
                neuron.update(x=current)
                vm[step] = float(neuron.V.value[0] / u.mV)
    return vm


# ---------------------------------------------------------------------------
# Helper: analytical V_m trace for iaf_psc_delta with piecewise-constant I
# ---------------------------------------------------------------------------

def _analytical_vm_trace(dt_ms, n_steps, amplitude_pA, start_ms=0.0,
                         stop_ms=None, origin_ms=0.0,
                         E_L=-70.0, C_m=250.0, tau_m=10.0,
                         V_th=-55.0, V_reset=-70.0, t_ref_ms=2.0):
    r"""Compute the exact discrete-time V_m trace for iaf_psc_delta + DC.

    Uses the propagator (exact integration) update rule matching
    iaf_psc_delta's implementation:

        V[n+1] = E_L + (V[n] - E_L) * P33 + I * P30

    where P33 = exp(-dt/tau_m), P30 = tau_m/C_m * (1 - P33).

    Returns
    -------
    vm : np.ndarray, shape (n_steps,)
    spk_steps : list of int
        Step indices where spikes occurred.
    """
    dt = dt_ms  # ms
    P33 = math.exp(-dt / tau_m)
    P30 = (tau_m / C_m) * (1.0 - P33)
    refr_total = math.ceil(t_ref_ms / dt)

    t_start = origin_ms + start_ms
    t_stop = origin_ms + stop_ms if stop_ms is not None else float('inf')

    V = E_L  # initial V_m = E_L (NEST default)
    ref_count = 0
    vm = np.empty(n_steps, dtype=np.float64)
    spk_steps = []

    for step in range(n_steps):
        t = step * dt
        # Determine current amplitude at this step
        if t_start <= t < t_stop:
            I = amplitude_pA
        else:
            I = 0.0

        if ref_count == 0:
            # subthreshold propagation
            V = E_L + (V - E_L) * P33 + I * P30
        else:
            # refractory: V stays at V_reset
            ref_count -= 1

        # Spike check
        if V >= V_th and ref_count == 0:
            spk_steps.append(step)
            ref_count = refr_total
            V = V_reset

        vm[step] = V

    return vm, spk_steps


# ===================================================================
# Test Suite 1: dc_generator device in isolation
# ===================================================================

class TestDCGenerator(unittest.TestCase):
    r"""Unit tests for dc_generator output values and timing."""

    def setUp(self):
        self.dt = 0.1 * u.ms

    def test_default_parameters(self):
        r"""Default amplitude is 0 pA, start is 0 ms, stop is None."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator()
        self.assertTrue(u.math.allclose(dc.amplitude, 0. * u.pA))
        self.assertTrue(u.math.allclose(dc.start, 0. * u.ms))
        self.assertIsNone(dc.stop)
        self.assertTrue(u.math.allclose(dc.origin, 0. * u.ms))

    def test_output_always_active(self):
        r"""With default start=0 and stop=None, always outputs amplitude."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=500. * u.pA)
            for t_val in [0., 1., 10., 100., 1000.]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = dc.update()
                self.assertTrue(u.math.allclose(out, 500. * u.pA),
                                f"Failed at t={t_val} ms")

    def test_output_before_start_is_zero(self):
        r"""Before start time, output is zero."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=500. * u.pA, start=10. * u.ms)
            for t_val in [0., 5., 9.9]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = dc.update()
                self.assertTrue(u.math.allclose(out, 0. * u.pA),
                                f"Should be 0 at t={t_val} ms")

    def test_output_during_active_window(self):
        r"""During [start, stop), output equals amplitude."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=500. * u.pA,
                              start=10. * u.ms, stop=20. * u.ms)
            for t_val in [10., 15., 19.9]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = dc.update()
                self.assertTrue(u.math.allclose(out, 500. * u.pA),
                                f"Should be 500 at t={t_val} ms")

    def test_output_after_stop_is_zero(self):
        r"""At and after stop time, output is zero."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=500. * u.pA,
                              start=10. * u.ms, stop=20. * u.ms)
            for t_val in [20., 25., 100.]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = dc.update()
                self.assertTrue(u.math.allclose(out, 0. * u.pA),
                                f"Should be 0 at t={t_val} ms")

    def test_start_boundary_inclusive(self):
        r"""At t=start, output is amplitude (start is inclusive)."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=123. * u.pA, start=10. * u.ms)
            with brainstate.environ.context(t=10. * u.ms):
                out = dc.update()
            self.assertTrue(u.math.allclose(out, 123. * u.pA))

    def test_stop_boundary_exclusive(self):
        r"""At t=stop, output is zero (stop is exclusive)."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=123. * u.pA,
                              start=0. * u.ms, stop=10. * u.ms)
            with brainstate.environ.context(t=10. * u.ms):
                out = dc.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA))

    def test_origin_shifts_window(self):
        r"""Origin shifts both start and stop by the same amount."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=500. * u.pA,
                              start=10. * u.ms, stop=20. * u.ms,
                              origin=100. * u.ms)
            # Before effective start (100 + 10 = 110 ms)
            with brainstate.environ.context(t=105. * u.ms):
                self.assertTrue(u.math.allclose(dc.update(), 0. * u.pA))
            # During effective window (110 to 120 ms)
            with brainstate.environ.context(t=115. * u.ms):
                self.assertTrue(u.math.allclose(dc.update(), 500. * u.pA))
            # After effective stop (120 ms)
            with brainstate.environ.context(t=125. * u.ms):
                self.assertTrue(u.math.allclose(dc.update(), 0. * u.pA))

    def test_negative_amplitude(self):
        r"""Negative amplitude produces negative (inhibitory) current."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=-800. * u.pA)
            with brainstate.environ.context(t=0. * u.ms):
                out = dc.update()
            self.assertTrue(u.math.allclose(out, -800. * u.pA))

    def test_zero_amplitude(self):
        r"""Zero amplitude always outputs zero, even when active."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=0. * u.pA)
            with brainstate.environ.context(t=50. * u.ms):
                out = dc.update()
            self.assertTrue(u.math.allclose(out, 0. * u.pA))

    def test_output_shape(self):
        r"""Output shape matches in_size."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(in_size=5, amplitude=100. * u.pA)
            with brainstate.environ.context(t=0. * u.ms):
                out = dc.update()
            self.assertEqual(out.shape, (5,))

    def test_stop_before_start_always_zero(self):
        r"""If stop <= start, the device is never active."""
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=500. * u.pA,
                              start=20. * u.ms, stop=10. * u.ms)
            for t_val in [0., 10., 15., 20., 25.]:
                with brainstate.environ.context(t=t_val * u.ms):
                    out = dc.update()
                self.assertTrue(u.math.allclose(out, 0. * u.pA),
                                f"Should be 0 at t={t_val} ms (stop < start)")


# ===================================================================
# Test Suite 2: dc_generator driving iaf_psc_delta
# ===================================================================

class TestDCGeneratorWithNeuron(unittest.TestCase):
    r"""Integration tests: dc_generator + iaf_psc_delta."""

    def setUp(self):
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms
        # NEST-default iaf_psc_delta parameters
        self.E_L = -70.0
        self.C_m = 250.0
        self.tau_m = 10.0
        self.V_th = -55.0
        self.V_reset = -70.0
        self.t_ref = 2.0  # ms

    def test_dc_vs_I_e_subthreshold(self):
        r"""dc_generator matches neuron's I_e for subthreshold current.

        This mirrors NEST's test_dcgen_versus_I_e.py: a neuron driven by
        dc_generator should reach the same steady-state V_m as a neuron
        using the built-in I_e parameter.
        """
        amplitude = 300.0  # pA, subthreshold
        simtime = 200.0  # ms (20 time constants — well past transient)

        # Run with dc_generator
        vm_dc = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude,
        )

        # Run with I_e
        n_steps = int(round(simtime / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta(1, I_e=amplitude * u.pA)
            neuron.init_state()
            vm_ie = np.empty(n_steps, dtype=np.float64)
            for step in range(n_steps):
                with brainstate.environ.context(t=step * self.dt):
                    neuron.update()
                    vm_ie[step] = float(neuron.V.value[0] / u.mV)

        # Traces should be identical (same numerical computation)
        npt.assert_allclose(vm_dc, vm_ie, atol=1e-12,
                            err_msg="dc_generator trace differs from I_e trace")

    def test_dc_vs_I_e_suprathreshold(self):
        r"""dc_generator matches I_e for suprathreshold (spiking) current.

        Both should produce identical spike times and V_m traces.
        """
        amplitude = 500.0  # pA, suprathreshold
        simtime = 100.0

        vm_dc = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude,
        )

        n_steps = int(round(simtime / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            neuron = iaf_psc_delta(1, I_e=amplitude * u.pA)
            neuron.init_state()
            vm_ie = np.empty(n_steps, dtype=np.float64)
            for step in range(n_steps):
                with brainstate.environ.context(t=step * self.dt):
                    neuron.update()
                    vm_ie[step] = float(neuron.V.value[0] / u.mV)

        npt.assert_allclose(vm_dc, vm_ie, atol=1e-12,
                            err_msg="dc_generator trace differs from I_e trace (spiking)")

    def test_subthreshold_analytical_trace(self):
        r"""V_m trace matches analytical solution for subthreshold DC.

        The iaf_psc_delta update rule with constant current I is:
            V[n+1] = E_L + (V[n] - E_L) * P33 + I * P30
        where P33 = exp(-dt/tau_m), P30 = (tau_m/C_m) * (1 - P33).
        """
        amplitude = 300.0  # pA, subthreshold
        simtime = 100.0
        n_steps = int(round(simtime / self.dt_ms))

        vm_bp = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude,
        )

        vm_ref, _ = _analytical_vm_trace(
            dt_ms=self.dt_ms, n_steps=n_steps,
            amplitude_pA=amplitude,
        )

        npt.assert_allclose(vm_bp, vm_ref, atol=1e-10,
                            err_msg="V_m trace does not match analytical solution")

    def test_suprathreshold_analytical_trace(self):
        r"""V_m trace (with spikes and reset) matches analytical solution."""
        amplitude = 500.0  # pA, suprathreshold
        simtime = 100.0
        n_steps = int(round(simtime / self.dt_ms))

        vm_bp = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude,
        )

        vm_ref, _ = _analytical_vm_trace(
            dt_ms=self.dt_ms, n_steps=n_steps,
            amplitude_pA=amplitude,
        )

        npt.assert_allclose(vm_bp, vm_ref, atol=1e-10,
                            err_msg="V_m trace does not match analytical solution (spiking)")

    def test_dc_with_start_stop_analytical(self):
        r"""V_m trace with start/stop matches analytical solution."""
        amplitude = 300.0
        start = 20.0
        stop = 60.0
        simtime = 100.0
        n_steps = int(round(simtime / self.dt_ms))

        vm_bp = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude, start_ms=start, stop_ms=stop,
        )

        vm_ref, _ = _analytical_vm_trace(
            dt_ms=self.dt_ms, n_steps=n_steps,
            amplitude_pA=amplitude, start_ms=start, stop_ms=stop,
        )

        npt.assert_allclose(vm_bp, vm_ref, atol=1e-10,
                            err_msg="V_m trace with start/stop window "
                                    "does not match analytical solution")

    def test_dc_with_origin_analytical(self):
        r"""V_m trace with origin offset matches analytical solution."""
        amplitude = 300.0
        start = 10.0
        stop = 50.0
        origin = 30.0  # effective window: 40–80 ms
        simtime = 100.0
        n_steps = int(round(simtime / self.dt_ms))

        vm_bp = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude, start_ms=start, stop_ms=stop,
            origin_ms=origin,
        )

        vm_ref, _ = _analytical_vm_trace(
            dt_ms=self.dt_ms, n_steps=n_steps,
            amplitude_pA=amplitude, start_ms=start, stop_ms=stop,
            origin_ms=origin,
        )

        npt.assert_allclose(vm_bp, vm_ref, atol=1e-10,
                            err_msg="V_m trace with origin offset "
                                    "does not match analytical solution")

    def test_membrane_returns_to_rest_after_stop(self):
        r"""After DC stops, V_m should decay back towards E_L."""
        amplitude = 300.0
        stop = 50.0
        simtime = 200.0

        vm = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude, stop_ms=stop,
        )

        # After many time constants past stop, V_m should be near E_L
        final_vm = vm[-1]
        self.assertAlmostEqual(final_vm, self.E_L, places=5,
                               msg="V_m did not return to E_L after DC stop")

    def test_steady_state_voltage(self):
        r"""Steady-state V_m matches E_L + I * tau_m / C_m for subthreshold I."""
        amplitude = 300.0  # pA
        V_ss_expected = self.E_L + amplitude * self.tau_m / self.C_m
        # V_ss = -70 + 300 * 10 / 250 = -70 + 12 = -58 mV
        simtime = 200.0  # 20 time constants

        vm = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude,
        )

        self.assertAlmostEqual(vm[-1], V_ss_expected, places=6,
                               msg=f"Steady-state V_m should be {V_ss_expected} mV")

    def test_two_dc_generators_additive(self):
        r"""Two dc_generators with different windows produce additive currents."""
        simtime = 100.0
        n_steps = int(round(simtime / self.dt_ms))

        with brainstate.environ.context(dt=self.dt):
            dc1 = dc_generator(amplitude=200. * u.pA,
                               start=0. * u.ms, stop=50. * u.ms)
            dc2 = dc_generator(amplitude=100. * u.pA,
                               start=25. * u.ms, stop=75. * u.ms)
            neuron = iaf_psc_delta(1)
            neuron.init_state()

            vm = np.empty(n_steps, dtype=np.float64)
            for step in range(n_steps):
                t = step * self.dt
                with brainstate.environ.context(t=t):
                    I = dc1.update() + dc2.update()
                    neuron.update(x=I)
                    vm[step] = float(neuron.V.value[0] / u.mV)

        # Build analytical reference with piecewise current:
        #   [0,25): 200 pA, [25,50): 300 pA, [50,75): 100 pA, [75,100): 0 pA
        P33 = math.exp(-self.dt_ms / self.tau_m)
        P30 = (self.tau_m / self.C_m) * (1.0 - P33)
        V = self.E_L
        ref_vm = np.empty(n_steps, dtype=np.float64)
        ref_count = 0
        for step in range(n_steps):
            t = step * self.dt_ms
            I = 0.0
            if 0.0 <= t < 50.0:
                I += 200.0
            if 25.0 <= t < 75.0:
                I += 100.0

            if ref_count == 0:
                V = self.E_L + (V - self.E_L) * P33 + I * P30
            else:
                ref_count -= 1
            if V >= self.V_th and ref_count == 0:
                ref_count = math.ceil(self.t_ref / self.dt_ms)
                V = self.V_reset
            ref_vm[step] = V

        npt.assert_allclose(vm, ref_vm, atol=1e-10,
                            err_msg="Additive DC generators do not match "
                                    "analytical piecewise-current solution")


# ===================================================================
# Test Suite 3: comparison against NEST simulator
# ===================================================================

class TestDCGeneratorVsNEST(unittest.TestCase):
    r"""Compare dc_generator + iaf_psc_delta against NEST.

    These tests are skipped when NEST is not installed.
    """

    def setUp(self):
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    @staticmethod
    def _is_nest_available():
        try:
            import nest  # noqa: F401
            return True
        except ImportError:
            return False

    def _run_nest_dc_simulation(self, simtime_ms, amplitude_pA,
                                start_ms=0.0, stop_ms=None,
                                origin_ms=0.0):
        r"""Run NEST dc_generator + iaf_psc_delta and return V_m trace."""
        import nest

        nest.ResetKernel()
        nest.resolution = self.dt_ms

        neuron = nest.Create("iaf_psc_delta")
        dc_params = {"amplitude": amplitude_pA}
        dc_params["start"] = start_ms
        if stop_ms is not None:
            dc_params["stop"] = stop_ms
        dc_params["origin"] = origin_ms
        dc = nest.Create("dc_generator", params=dc_params)

        mm = nest.Create("multimeter", params={
            "record_from": ["V_m"],
            "interval": self.dt_ms,
        })

        nest.Connect(dc, neuron)
        nest.Connect(mm, neuron)
        nest.Simulate(simtime_ms)

        events = mm.get("events")
        return np.array(events["V_m"])

    def test_subthreshold_vs_nest(self):
        r"""Subthreshold DC: compare V_m trace against NEST."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        amplitude = 300.0
        simtime = 100.0

        nest_vm = self._run_nest_dc_simulation(
            simtime_ms=simtime, amplitude_pA=amplitude,
        )
        bp_vm = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude,
        )

        # Both should have the same number of steps
        self.assertEqual(len(bp_vm), len(nest_vm),
                         "Trace lengths differ between brainpy.state and NEST")
        npt.assert_allclose(bp_vm, nest_vm, atol=1e-10,
                            err_msg="Subthreshold V_m differs from NEST")

    def test_suprathreshold_vs_nest(self):
        r"""Suprathreshold (spiking) DC: compare V_m trace against NEST."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        amplitude = 500.0
        simtime = 100.0

        nest_vm = self._run_nest_dc_simulation(
            simtime_ms=simtime, amplitude_pA=amplitude,
        )
        bp_vm = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude,
        )

        self.assertEqual(len(bp_vm), len(nest_vm))
        npt.assert_allclose(bp_vm, nest_vm, atol=1e-10,
                            err_msg="Suprathreshold V_m differs from NEST")

    def test_start_stop_vs_nest(self):
        r"""DC with start/stop window: compare V_m trace against NEST."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        amplitude = 300.0
        start = 20.0
        stop = 60.0
        simtime = 100.0

        nest_vm = self._run_nest_dc_simulation(
            simtime_ms=simtime, amplitude_pA=amplitude,
            start_ms=start, stop_ms=stop,
        )
        bp_vm = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude, start_ms=start, stop_ms=stop,
        )

        self.assertEqual(len(bp_vm), len(nest_vm))
        npt.assert_allclose(bp_vm, nest_vm, atol=1e-10,
                            err_msg="V_m with start/stop differs from NEST")

    def test_dc_vs_I_e_matches_nest_test(self):
        r"""Reproduce NEST's test_dcgen_versus_I_e.py logic.

        A neuron driven by dc_generator from near the end of the first
        simulation block should converge to the same V_m as a neuron
        started fresh with I_e.
        """
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        import nest

        amp = 123.456

        # --- NEST reference ---
        nest.ResetKernel()
        nest.resolution = self.dt_ms
        n1 = nest.Create("iaf_psc_delta")
        dc_gen = nest.Create("dc_generator", params={"start": 99.0})
        nest.Connect(dc_gen, n1)
        dc_gen.set(amplitude=amp)
        nest.Simulate(100.0)

        n2 = nest.Create("iaf_psc_delta")
        n2.set(I_e=amp)
        nest.Simulate(300.0)

        v1_nest = n1.get("V_m")
        v2_nest = n2.get("V_m")

        # NEST test: v1 ≈ v2 (steady-state convergence)
        self.assertAlmostEqual(v1_nest, v2_nest, places=10,
                               msg="NEST dc_gen vs I_e mismatch")

        # --- brainpy.state: same logic ---
        # Phase 1: dc_generator from 99 ms, simulate 100 ms
        n_steps_1 = int(round(100.0 / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            dc = dc_generator(amplitude=amp * u.pA, start=99.0 * u.ms)
            neuron1 = iaf_psc_delta(1)
            neuron1.init_state()
            for step in range(n_steps_1):
                with brainstate.environ.context(t=step * self.dt):
                    neuron1.update(x=dc.update())

        # Phase 2: continue neuron1 with dc, create neuron2 with I_e
        n_steps_2 = int(round(300.0 / self.dt_ms))
        with brainstate.environ.context(dt=self.dt):
            neuron2 = iaf_psc_delta(1, I_e=amp * u.pA)
            neuron2.init_state()
            for step in range(n_steps_2):
                t_abs = (n_steps_1 + step) * self.dt
                with brainstate.environ.context(t=t_abs):
                    neuron1.update(x=dc.update())
                with brainstate.environ.context(t=step * self.dt):
                    neuron2.update()

        v1_bp = float(neuron1.V.value[0] / u.mV)
        v2_bp = float(neuron2.V.value[0] / u.mV)

        self.assertAlmostEqual(v1_bp, v2_bp, places=10,
                               msg="brainpy.state dc_gen vs I_e mismatch")

        # Also verify brainpy.state matches NEST
        self.assertAlmostEqual(v1_bp, v1_nest, places=10,
                               msg="brainpy.state V_m differs from NEST V_m")

    def test_negative_current_vs_nest(self):
        r"""Negative (inhibitory) DC current: compare against NEST."""
        if not self._is_nest_available():
            self.skipTest("NEST simulator not available")

        amplitude = -200.0
        simtime = 50.0

        nest_vm = self._run_nest_dc_simulation(
            simtime_ms=simtime, amplitude_pA=amplitude,
        )
        bp_vm = _run_bp_simulation(
            dt_ms=self.dt_ms, simtime_ms=simtime,
            amplitude_pA=amplitude,
        )

        self.assertEqual(len(bp_vm), len(nest_vm))
        npt.assert_allclose(bp_vm, nest_vm, atol=1e-10,
                            err_msg="Negative-current V_m differs from NEST")


if __name__ == '__main__':
    unittest.main()
