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

import jax

jax.config.update('jax_enable_x64', True)
import brainstate
import saiunit as u
import jax.numpy as jnp
import numpy as np

from brainpy_state._nest.erfc_neuron import erfc_neuron


class TestErfcNeuron(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _step(self, neuron, step_idx, x=0.0 * u.mV, delta=None):
        if delta is not None:
            neuron.add_delta_input(f'delta_{step_idx}', delta)
        dt = brainstate.environ.get_dt()
        with brainstate.environ.context(t=step_idx * dt):
            return neuron.update(x=x)

    def test_nest_default_parameters(self):
        r"""Defaults should match NEST erfc_neuron."""
        neuron = erfc_neuron(1)
        self.assertTrue(u.math.allclose(neuron.tau_m, 10.0 * u.ms))
        self.assertTrue(u.math.allclose(neuron.theta, 0.0 * u.mV))
        self.assertTrue(u.math.allclose(neuron.sigma, 1.0 * u.mV))
        self.assertTrue(neuron.stochastic_update)

    def test_gain_formula_matches_nest_equation(self):
        r"""g(h)=0.5*erfc(-(h-theta)/(sqrt(2)*sigma))."""
        with brainstate.environ.context(dt=self.dt):
            neuron = erfc_neuron(
                1,
                theta=0.4 * u.mV,
                sigma=1.3 * u.mV,
                stochastic_update=False,
            )
            neuron.init_state()

            dftype = brainstate.environ.dftype()
            h = jnp.array([0.7], dtype=dftype) * u.mV
            got = neuron._gain_probability(h)
            expected = 0.5 * math.erfc(-(0.7 - 0.4) / (math.sqrt(2.0) * 1.3))
            self.assertAlmostEqual(float(got[0]), expected, places=12)

    def test_matches_nest_binary_communication_h_trace(self):
        r"""Replicate NEST test_binary.py expected h trace for binary encoding."""
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = erfc_neuron(
                1,
                theta=0.0 * u.mV,
                sigma=1.0 * u.mV,
                stochastic_update=False,
                rng_seed=11,
            )
            neuron.init_state()

            h_trace = []
            for step in range(19):
                if step == 9:
                    delta = 1.0 * u.mV
                elif step == 14:
                    delta = -1.0 * u.mV
                else:
                    delta = None
                self._step(neuron, step, delta=delta)
                h_trace.append(float((neuron.h.value / u.mV)[0]))

            expected = [0.0] * 9 + [1.0] * 5 + [0.0] * 5
            np.testing.assert_allclose(h_trace, expected, atol=1e-12, rtol=0.0)

    def test_strict_time_inequality_matches_nest(self):
        r"""Update happens only when t+dt > t_next (strict >, not >=)."""
        dt = 0.125 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = erfc_neuron(
                1,
                tau_m=1.0 * u.ms,
                theta=-10.0 * u.mV,
                sigma=1.0 * u.mV,
                stochastic_update=True,
                rng_seed=5,
            )
            neuron.init_state()
            dftype = brainstate.environ.dftype()
            neuron.t_next.value = jnp.array([0.375], dtype=dftype) * u.ms

            out = self._step(neuron, 1)
            self.assertEqual(float(out[0]), 0.0)

            out = self._step(neuron, 2)
            self.assertEqual(float(out[0]), 0.0)

            out = self._step(neuron, 3)
            self.assertEqual(float(out[0]), 1.0)

    def test_reference_step_trace_with_controlled_rng(self):
        r"""Reference regression for NEST update ordering and timing."""
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = erfc_neuron(
                1,
                tau_m=0.1 * u.ms,
                theta=0.0 * u.mV,
                sigma=1.0 * u.mV,
                stochastic_update=True,
                rng_seed=0,
            )

            dftype = brainstate.environ.dftype()
            exp_samples = iter([
                jnp.array([1.0], dtype=dftype),
                jnp.array([2.0], dtype=dftype),
                jnp.array([1.0], dtype=dftype),
                jnp.array([1.0], dtype=dftype),
            ])
            uni_samples = iter([
                jnp.array([0.2], dtype=dftype),
                jnp.array([0.9], dtype=dftype),
                jnp.array([0.1], dtype=dftype),
            ])
            neuron._sample_exponential = lambda shape: next(exp_samples)
            neuron._sample_uniform = lambda shape: next(uni_samples)
            neuron.init_state()

            h = 0.0
            y = 0.0
            t_next = 0.1
            ref_exp = iter([2.0, 1.0, 1.0])
            ref_uni = iter([0.2, 0.9, 0.1])
            deltas = [0.0, 0.5, 0.0, -0.3, 0.0]

            for step, d_h in enumerate(deltas):
                h += d_h
                current_time = (step + 1) * 0.1
                if current_time > t_next:
                    p = 0.5 * math.erfc(-(h - 0.0) / (math.sqrt(2.0) * 1.0))
                    y = 1.0 if next(ref_uni) < p else 0.0
                    t_next += next(ref_exp) * 0.1

                delta = None if d_h == 0.0 else d_h * u.mV
                out = self._step(neuron, step, delta=delta)
                self.assertAlmostEqual(float(out[0]), y, places=12)

            self.assertAlmostEqual(float((neuron.t_next.value / u.ms)[0]), t_next, places=12)

    def test_activity_matches_erfc_theory(self):
        r"""Long-run mean activity at h=0 follows NEST theory."""
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            dftype = brainstate.environ.dftype()
            sigma_vals = np.logspace(-1.0, 1.0, 3, dtype=dftype)
            theta_vals = np.linspace(-4.0, 4.0, 5, dtype=dftype)
            sigma_grid, theta_grid = np.meshgrid(sigma_vals, theta_vals)
            sigma = sigma_grid.reshape(-1)
            theta = theta_grid.reshape(-1)
            n = sigma.shape[0]

            neuron = erfc_neuron(
                n,
                tau_m=1.0 * u.ms,
                sigma=sigma * u.mV,
                theta=theta * u.mV,
                stochastic_update=False,
                rng_seed=1234,
            )
            neuron.init_state()

            n_steps = 6000
            y_sum = np.zeros(n, dtype=dftype)

            for step in range(n_steps):
                out = self._step(neuron, step, x=0.0 * u.mV)
                y_sum += np.asarray(out, dtype=dftype)

            mean_activity = y_sum / float(n_steps)
            theory = np.array(
                [
                    0.5 * math.erfc(th / (math.sqrt(2.0) * sg))
                    for sg, th in zip(sigma, theta)
                ],
                dtype=dftype,
            )
            delta = np.maximum(2e-1 * theory * (1.0 - theory), 1e-2)
            np.testing.assert_array_less(np.abs(mean_activity - theory), delta)


if __name__ == '__main__':
    unittest.main()
