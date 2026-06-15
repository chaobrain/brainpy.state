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
os.environ['JAX_ENABLE_X64'] = '1'

import unittest

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import brainstate
import brainunit as u
import jax.numpy as jnp

from brainpy_state._nest.ginzburg_neuron import ginzburg_neuron


class TestGinzburgNeuron(unittest.TestCase):
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
        r"""Defaults should match NEST ginzburg_neuron."""
        neuron = ginzburg_neuron(1)
        self.assertTrue(u.math.allclose(neuron.tau_m, 10.0 * u.ms))
        self.assertTrue(u.math.allclose(neuron.theta, 0.0 * u.mV))
        self.assertTrue(u.math.allclose(neuron.c_1, 0.0 / u.mV))
        self.assertTrue(u.math.allclose(neuron.c_2, 1.0))
        self.assertTrue(u.math.allclose(neuron.c_3, 1.0 / u.mV))
        self.assertTrue(neuron.stochastic_update)

    def test_gain_formula_matches_nest_equation(self):
        r"""g(h)=c1*h + c2*(1+tanh(c3*(h-theta)))/2."""
        with brainstate.environ.context(dt=self.dt):
            neuron = ginzburg_neuron(
                1,
                theta=0.4 * u.mV,
                c_1=0.25 / u.mV,
                c_2=0.8,
                c_3=1.2 / u.mV,
            )
            neuron.init_state()
            dftype = brainstate.environ.dftype()
            h = jnp.array([0.7], dtype=dftype) * u.mV

            got = neuron._gain_probability(h)
            expected = (
                0.25 / u.mV * h
                + 0.8 * 0.5 * (1.0 + u.math.tanh(1.2 / u.mV * (h - 0.4 * u.mV)))
            )
            self.assertTrue(u.math.allclose(got, expected))

    def test_probability_outside_unit_interval_behaves_like_nest(self):
        r"""Comparing U in [0,1) against p implements effective clipping."""
        with brainstate.environ.context(dt=self.dt):
            neuron = ginzburg_neuron(
                1,
                c_1=1.0 / u.mV,
                c_2=0.0,
                c_3=0.0 / u.mV,
                stochastic_update=False,
                rng_seed=17,
            )
            neuron.init_state()

            # h=2 -> p=2, always active.
            out = self._step(neuron, 0, delta=2.0 * u.mV)
            self.assertEqual(float(out[0]), 1.0)

            # h=-1 -> p=-1, always inactive.
            out = self._step(neuron, 1, delta=-3.0 * u.mV)
            self.assertEqual(float(out[0]), 0.0)

    def test_matches_nest_binary_communication_h_trace(self):
        r"""Replicate NEST test_binary.py expected h trace for binary encoding."""
        dt = 1.0 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = ginzburg_neuron(
                1,
                c_1=0.0 / u.mV,
                c_2=2.0,
                c_3=0.0 / u.mV,
                stochastic_update=False,
                rng_seed=3,
            )
            neuron.init_state()

            dftype = brainstate.environ.dftype()
            # Pre-compute per-step delta inputs (dimensionless mV mantissa).
            delta_vals = jnp.zeros(19, dtype=dftype)
            delta_vals = delta_vals.at[9].set(1.0)    # up-transition
            delta_vals = delta_vals.at[14].set(-1.0)  # down-transition

            def body(delta_raw):
                # Apply delta directly to h (bypasses add_delta_input / sum_delta_inputs).
                neuron.h.value = neuron.h.value + delta_raw * u.mV
                neuron.update(x=0.0 * u.mV)
                return (neuron.h.value / u.mV)[0]

            h_trace = brainstate.transform.for_loop(body, delta_vals)

            expected = [0.0] * 9 + [1.0] * 5 + [0.0] * 5
            np.testing.assert_allclose(h_trace, expected, atol=1e-12, rtol=0.0)

    def test_strict_time_inequality_matches_nest(self):
        r"""Update happens only when t+dt > t_next (strict >, not >=)."""
        dt = 0.125 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = ginzburg_neuron(
                1,
                tau_m=1.0 * u.ms,
                c_1=0.0 / u.mV,
                c_2=2.0,  # p=1 exactly
                c_3=0.0 / u.mV,
                stochastic_update=True,
                rng_seed=7,
            )
            neuron.init_state()
            dftype = brainstate.environ.dftype()
            neuron.t_next.value = jnp.array([0.375], dtype=dftype) * u.ms

            # t=0.125 -> t+dt=0.25 <= 0.375: no update
            out = self._step(neuron, 1)
            self.assertEqual(float(out[0]), 0.0)

            # t=0.25 -> t+dt=0.375 == 0.375: no update
            out = self._step(neuron, 2)
            self.assertEqual(float(out[0]), 0.0)

            # t=0.375 -> t+dt=0.5 > 0.375: update
            out = self._step(neuron, 3)
            self.assertEqual(float(out[0]), 1.0)

    def test_reference_step_trace_with_controlled_rng(self):
        r"""Reference regression for NEST update order with deterministic RNG."""
        dt = 0.1 * u.ms
        with brainstate.environ.context(dt=dt):
            neuron = ginzburg_neuron(
                1,
                tau_m=0.1 * u.ms,
                theta=0.0 * u.mV,
                c_1=0.0 / u.mV,
                c_2=1.0,
                c_3=2.0 / u.mV,
                stochastic_update=True,
                rng_seed=0,
            )

            # ``update`` now draws one uniform and one exponential sample on
            # *every* step and masks the state with ``should_update`` (so it is
            # jit-safe). The reference below mirrors that cadence; the first
            # exponential draw is consumed by ``init_state`` to seed ``t_next``.
            dftype = brainstate.environ.dftype()
            step_exp = [2.0, 1.0, 1.0, 1.0, 1.0]
            step_uni = [0.2, 0.9, 0.1, 0.5, 0.4]
            exp_samples = iter([jnp.array([1.0], dtype=dftype)]
                               + [jnp.array([v], dtype=dftype) for v in step_exp])
            uni_samples = iter([jnp.array([v], dtype=dftype) for v in step_uni])
            neuron._sample_exponential = lambda shape: next(exp_samples)
            neuron._sample_uniform = lambda shape: next(uni_samples)
            neuron.init_state()

            # Manual reference from NEST equations.
            h = 0.0
            y = 0.0
            t_next = 0.1
            deltas = [0.0, 0.5, 0.0, -0.5, 0.0]

            for step, d_h in enumerate(deltas):
                h += d_h
                current_time = (step + 1) * 0.1
                should_update = current_time > t_next
                p = 0.5 * (1.0 + np.tanh(2.0 * h))
                new_y = 1.0 if step_uni[step] < p else 0.0
                if should_update:
                    y = new_y
                    t_next += step_exp[step] * 0.1

                delta = None if d_h == 0.0 else d_h * u.mV
                out = self._step(neuron, step, delta=delta)
                self.assertAlmostEqual(float(out[0]), y, places=12)

            self.assertAlmostEqual(float((neuron.t_next.value / u.ms)[0]), t_next, places=12)


if __name__ == '__main__':
    unittest.main()
