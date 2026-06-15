# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import unittest

import brainstate
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import brainunit as u

from brainpy_state._exp_euler import exp_euler_step


def _analytic_exp_euler(state, jacobian_per_time, drift, dt):
    """Reference one-step exponential Euler: x + dt * phi(dt*J) * f, with phi
    computed on plain dimensionless mantissas to avoid relying on the code path
    under test."""
    z = float(u.get_mantissa(dt * jacobian_per_time))
    phi = (np.expm1(z) / z) if z != 0.0 else 1.0
    return state + dt * phi * drift


class TestExpEulerStep(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_membrane_voltage_step_does_not_raise(self):
        # Regression: a dimensional (mV) linear ODE must integrate one step
        # without exprel rejecting a dimensional argument.
        V_rest, tau = -65.0 * u.mV, 10.0 * u.ms
        V = jnp.array([-60.0, -55.0, -65.0]) * u.mV

        def dv(v):
            return (-(v - V_rest)) / tau

        out = exp_euler_step(dv, V)

        # Correct unit preserved.
        self.assertEqual(u.get_unit(out), u.get_unit(V))
        # Correct numerics vs. analytic exp-euler (per element).
        for i in range(V.shape[0]):
            j = -1.0 / tau  # dv/dV
            ref = _analytic_exp_euler(V[i], j, dv(V[i]), self.dt)
            npt.assert_allclose(
                u.get_mantissa(out[i]), u.get_mantissa(ref), rtol=1e-5
            )

    def test_current_unit_step(self):
        # Currents (mA) appear in several failing models; must also work.
        tau = 5.0 * u.ms
        I = jnp.array([1.0, -2.0]) * u.mA

        def di(i):
            return -i / tau

        out = exp_euler_step(di, I)
        self.assertEqual(u.get_unit(out), u.get_unit(I))
        for k in range(I.shape[0]):
            j = -1.0 / tau
            ref = _analytic_exp_euler(I[k], j, di(I[k]), self.dt)
            npt.assert_allclose(
                u.get_mantissa(out[k]), u.get_mantissa(ref), rtol=1e-5
            )

    def test_dimensionless_state_still_works(self):
        # Plain dimensionless arrays must continue to integrate. Per the
        # documented usage, a dimensionless state pairs with a dimensionless dt
        # (otherwise dt*J carries the dt unit and exprel rightly rejects it).
        brainstate.environ.set(dt=0.1)
        x = jnp.array([1.0, 0.5])

        def drift(v):
            return -v

        out = exp_euler_step(drift, x)
        ref = np.exp(-0.1) * np.asarray(x)
        npt.assert_allclose(np.asarray(out), ref, rtol=1e-4)

    def test_runs_under_jit(self):
        # The whole point of the rebuild is JAX-conformance; ensure it jits.
        V_rest, tau = -65.0 * u.mV, 10.0 * u.ms

        def dv(v):
            return (-(v - V_rest)) / tau

        @brainstate.transform.jit
        def step(v):
            return exp_euler_step(dv, v)

        out = step(jnp.array([-60.0]) * u.mV)
        self.assertEqual(u.get_unit(out), u.mV)


class TestExpEulerPatch(unittest.TestCase):
    """The import-time patch must make the public ``brainstate.nn.exp_euler_step``
    accept dimensional input, so user/test/docstring code that calls it directly
    works on a buggy brainstate."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_patch_makes_public_api_accept_dimensional_input(self):
        import brainpy_state  # noqa: F401  -- ensures install_exp_euler_patch() ran

        V_rest, tau = -65.0 * u.mV, 10.0 * u.ms

        def dv(v):
            return (-(v - V_rest)) / tau

        # This is exactly the call pattern that fails on an unpatched brainstate.
        out = brainstate.nn.exp_euler_step(dv, jnp.array([-60.0]) * u.mV)
        self.assertEqual(u.get_unit(out), u.mV)
        npt.assert_allclose(u.get_mantissa(out)[0], -60.04975128, rtol=1e-5)

    def test_patch_is_idempotent(self):
        from brainpy_state._exp_euler import install_exp_euler_patch

        # Repeated installs are safe and report already-applied / unnecessary.
        self.assertIn(install_exp_euler_patch(), (True, False))
        self.assertIn(install_exp_euler_patch(), (True, False))


if __name__ == "__main__":
    unittest.main()
