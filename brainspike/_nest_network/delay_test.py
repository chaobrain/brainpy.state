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

import unittest

import brainstate
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import brainunit as u

from brainspike._nest_network.delay import InputDelay


class TestInputDelaySeam(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=1.0 * u.ms)

    def test_none_is_identity(self):
        seam = InputDelay(3, delay=None)
        brainstate.nn.init_all_states(seam)
        x = jnp.array([1., 2., 3.])
        out = seam.update(x)
        npt.assert_array_equal(np.asarray(out), np.asarray(x))

    def test_global_delay_shifts_whole_frame(self):
        # delay = 2 steps (dt = 1 ms); output at step t is the input at step t-2.
        seam = InputDelay(2, delay=2.0 * u.ms)
        brainstate.nn.init_all_states(seam)
        seq = [jnp.array([1., 10.]), jnp.array([2., 20.]),
               jnp.array([3., 30.]), jnp.array([4., 40.])]
        outs = [np.asarray(seam.update(x)) for x in seq]
        npt.assert_allclose(outs[0], [0., 0.], atol=1e-6)
        npt.assert_allclose(outs[1], [0., 0.], atol=1e-6)
        npt.assert_allclose(outs[2], [1., 10.], atol=1e-6)
        npt.assert_allclose(outs[3], [2., 20.], atol=1e-6)

    def test_axonal_delay_per_element(self):
        # element 0 delayed 1 step, element 1 delayed 3 steps.
        seam = InputDelay(2, delay=jnp.array([1.0, 3.0]) * u.ms)
        brainstate.nn.init_all_states(seam)
        seq = [jnp.array([1., 10.]), jnp.array([2., 20.]), jnp.array([3., 30.]),
               jnp.array([4., 40.]), jnp.array([5., 50.])]
        outs = [np.asarray(seam.update(x)) for x in seq]
        npt.assert_allclose(outs[1][0], 1., atol=1e-6)
        npt.assert_allclose(outs[2][0], 2., atol=1e-6)
        npt.assert_allclose(outs[3], [3., 10.], atol=1e-6)
        npt.assert_allclose(outs[4], [4., 20.], atol=1e-6)

    def test_fractional_delay_interpolates(self):
        # delay = 1.5 steps -> out[t] = 0.5*in[t-1] + 0.5*in[t-2].
        seam = InputDelay(1, delay=1.5 * u.ms)
        brainstate.nn.init_all_states(seam)
        seq = [jnp.array([10.]), jnp.array([20.]), jnp.array([30.]), jnp.array([40.])]
        outs = [float(np.asarray(seam.update(x))[0]) for x in seq]
        npt.assert_allclose(outs[0], 0., atol=1e-6)
        npt.assert_allclose(outs[1], 5., atol=1e-6)    # 0.5*10 + 0.5*0
        npt.assert_allclose(outs[2], 15., atol=1e-6)   # 0.5*20 + 0.5*10
        npt.assert_allclose(outs[3], 25., atol=1e-6)   # 0.5*30 + 0.5*20

    def test_runs_under_jitted_for_loop(self):
        seam = InputDelay(3, delay=2.0 * u.ms)
        brainstate.nn.init_all_states(seam)
        seq = jnp.arange(5 * 3, dtype=float).reshape(5, 3)
        outs = brainstate.transform.for_loop(lambda x: seam.update(x), seq)
        npt.assert_allclose(np.asarray(outs[2]), np.asarray(seq[0]), atol=1e-6)
        npt.assert_allclose(np.asarray(outs[4]), np.asarray(seq[2]), atol=1e-6)

    def test_heterogeneous_per_connection_diagonal_gather(self):
        # 3 connections from 3 distinct pre-neurons, delays [1, 5, 10] steps.
        # A single impulse delivers three pulses offset by 1, 5, 10 steps,
        # directly exercising retrieve_at_step(steps, pre_ids).
        n_pre, T = 3, 12
        pre_ids = jnp.array([0, 1, 2])
        delays = jnp.array([1., 5., 10.]) * u.ms
        seam = InputDelay(n_pre, delay=delays, indices=pre_ids)
        brainstate.nn.init_all_states(seam)
        seq = np.zeros((T, n_pre)); seq[0, :] = 1.0
        out = np.asarray([np.asarray(seam.update(jnp.asarray(seq[k]))) for k in range(T)])  # (T, 3)
        for k, d in enumerate([1, 5, 10]):
            self.assertEqual(int(np.argmax(out[:, k])), d)
            npt.assert_allclose(out[d, k], 1.0, atol=1e-6)
            npt.assert_allclose(out[:, k].sum(), 1.0, atol=1e-6)  # exactly one pulse

    def test_per_connection_length_mismatch_raises(self):
        seam = InputDelay(3, delay=jnp.array([1., 5.]) * u.ms, indices=jnp.array([0, 1, 2]))
        with self.assertRaises(ValueError):
            brainstate.nn.init_all_states(seam)


if __name__ == '__main__':
    unittest.main()
