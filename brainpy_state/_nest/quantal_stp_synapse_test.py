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

import brainstate
import brainunit as u
import jax
import numpy as np

from brainpy.state import quantal_stp_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class _MockReceiver:
    def __init__(self):
        self.delta_events = []

    def add_delta_input(self, key, inp, label=None):
        self.delta_events.append((key, inp, label))


def _spike_steps_from_times(spike_times_ms, dt_ms):
    return [int(round((float(t_ms) - dt_ms) / dt_ms)) for t_ms in spike_times_ms]


def _quantal_reference_step(
    *,
    U,
    u_state,
    n,
    a_state,
    tau_rec,
    tau_fac,
    t_last,
    t_spike,
    draws,
    cursor,
):
    if t_last >= 0.0:
        h = t_spike - t_last
        p_decay = math.exp(-h / tau_rec)
        u_decay = 0.0 if tau_fac < 1.0e-10 else math.exp(-h / tau_fac)
        u_state = U + u_state * (1.0 - U) * u_decay

        recovery_prob = 1.0 - p_decay
        depleted_sites = n - a_state
        for _ in range(depleted_sites if depleted_sites > 0 else 0):
            if draws[cursor] < recovery_prob:
                a_state += 1
            cursor += 1

    n_release = 0
    for _ in range(a_state if a_state > 0 else 0):
        if draws[cursor] < u_state:
            n_release += 1
        cursor += 1

    if n_release > 0:
        a_state -= n_release

    return u_state, a_state, t_spike, n_release, cursor


def _tsodyks2_reference_weights(*, spike_times_ms, U, u0, x0, tau_rec, tau_fac, weight):
    x = float(x0)
    u_state = float(u0)
    t_last = -1.0
    out = []
    for t_spike in spike_times_ms:
        t_spike = float(t_spike)
        if t_last >= 0.0:
            h = t_spike - t_last
            x_decay = math.exp(-h / tau_rec)
            u_decay = 0.0 if tau_fac == 0.0 else math.exp(-h / tau_fac)
            x = 1.0 + (x - x * u_state - 1.0) * x_decay
            u_state = U + u_state * (1.0 - U) * u_decay
        out.append(x * u_state * weight)
        t_last = t_spike
    return np.asarray(out, dtype=np.float64)


def _run_quantal_spike_weights(
    *,
    spike_times_ms,
    dt_ms,
    U,
    u0,
    tau_rec,
    tau_fac,
    n_sites,
    weight,
    seed,
):
    np.random.seed(int(seed))
    dt = dt_ms * u.ms
    spike_steps = _spike_steps_from_times(spike_times_ms, dt_ms)
    recv = _MockReceiver()

    with brainstate.environ.context(dt=dt):
        syn = quantal_stp_synapse(
            U=U,
            u=u0,
            n=n_sites,
            a=n_sites,
            tau_rec=tau_rec * u.ms,
            tau_fac=tau_fac * u.ms,
            delay=dt,
            weight=weight,
            post=recv,
        )
        syn.init_state()

        weights = []
        for step in spike_steps:
            with brainstate.environ.context(t=step * dt):
                sent = syn.send(1.0)
                if sent:
                    delivery_step = int(syn._curr_step(dt_ms) + int(syn._delay_steps))
                    payload = syn._queue[delivery_step][-1][1]
                    payload_f = float(np.asarray(u.math.asarray(payload), dtype=np.float64).reshape(()))
                    weights.append(payload_f)
                else:
                    weights.append(0.0)

    return np.asarray(weights, dtype=np.float64)


class TestQuantalSTPSynapseParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = quantal_stp_synapse()
            syn.init_state()
            syn.update(pre_spike=0.0)
            params = syn.get()

        self.assertEqual(params['weight'], 1.0)
        self.assertEqual(params['delay'], 1.0)
        self.assertEqual(params['delay_steps'], 1)
        self.assertEqual(params['receptor_type'], 0)
        self.assertEqual(params['event_type'], 'spike')
        self.assertEqual(params['U'], 0.5)
        self.assertEqual(params['u'], 0.5)
        self.assertEqual(params['tau_rec'], 800.0)
        self.assertEqual(params['tau_fac'], 0.0)
        self.assertEqual(params['n'], 1)
        self.assertEqual(params['a'], 1)
        self.assertEqual(params['synapse_model'], 'quantal_stp_synapse')
        self.assertAlmostEqual(syn.t_lastspike, -1.0, delta=1e-12)

    def test_parameter_validation_matches_nest(self):
        with self.assertRaisesRegex(ValueError, r"'U' must be in \[0,1\]\."):
            quantal_stp_synapse(U=1.1)
        with self.assertRaisesRegex(ValueError, r"'U' must be in \[0,1\]\."):
            quantal_stp_synapse(U=-0.1)
        with self.assertRaisesRegex(ValueError, r"'u' must be in \[0,1\]\."):
            quantal_stp_synapse(u=1.1)
        with self.assertRaisesRegex(ValueError, r"'u' must be in \[0,1\]\."):
            quantal_stp_synapse(u=-0.1)
        with self.assertRaisesRegex(ValueError, r"'tau_rec' must be > 0\."):
            quantal_stp_synapse(tau_rec=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, r"'tau_fac' must be >= 0\."):
            quantal_stp_synapse(tau_fac=-1.0 * u.ms)

        syn = quantal_stp_synapse()
        with self.assertRaisesRegex(ValueError, r"'U' must be in \[0,1\]\."):
            syn.set(U=1.1)
        with self.assertRaisesRegex(ValueError, r"'u' must be in \[0,1\]\."):
            syn.set(u=-0.1)
        with self.assertRaisesRegex(ValueError, r"'tau_rec' must be > 0\."):
            syn.set(tau_rec=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, r"'tau_fac' must be >= 0\."):
            syn.set(tau_fac=-1.0 * u.ms)

    def test_n_and_a_must_be_scalar_integers(self):
        with self.assertRaisesRegex(ValueError, r"'n' must be an integer\."):
            quantal_stp_synapse(n=2.5)
        with self.assertRaisesRegex(ValueError, r"'a' must be an integer\."):
            quantal_stp_synapse(a=1.5)

    def test_set_U_does_not_implicitly_change_u(self):
        syn = quantal_stp_synapse(U=0.4, u=0.25)
        syn.set(U=0.9)
        params = syn.get()
        self.assertAlmostEqual(params['U'], 0.9, delta=1e-12)
        self.assertAlmostEqual(params['u'], 0.25, delta=1e-12)


class TestQuantalSTPSynapseOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_state_update_order_and_effective_weight(self):
        recv = _MockReceiver()
        dt_ms = 1.0
        dt = dt_ms * u.ms

        U = 0.4
        u0 = 0.4
        n_sites = 5
        a0 = 2
        tau_rec = 100.0
        tau_fac = 50.0
        weight = 1.5
        spike_steps = [1, 4]

        draws = [
            0.3, 0.8,  # first spike release trials (a=2): one release
            0.01, 0.4, 0.02, 0.8,  # second spike recovery trials (n-a=4): two recoveries
            0.2, 0.7, 0.1,  # second spike release trials (a=3): two releases
        ]
        draw_iter = iter(draws)

        with brainstate.environ.context(dt=dt):
            syn = quantal_stp_synapse(
                U=U,
                u=u0,
                n=n_sites,
                a=a0,
                tau_rec=tau_rec * u.ms,
                tau_fac=tau_fac * u.ms,
                delay=1.0 * u.ms,
                weight=weight,
                receptor_type=2,
                post=recv,
            )
            syn.init_state()
            syn._sample_uniform = lambda: float(next(draw_iter))

            u_ref = u0
            a_ref = a0
            t_last_ref = -1.0
            cursor = 0
            release_ref = []
            delivered = []

            for step in range(7):
                pre_spike = 1.0 if step in spike_steps else 0.0
                with brainstate.environ.context(t=step * dt):
                    delivered.append(syn.update(pre_spike=pre_spike))

                if step in spike_steps:
                    t_spike = (step + 1) * dt_ms
                    u_ref, a_ref, t_last_ref, n_release_ref, cursor = _quantal_reference_step(
                        U=U,
                        u_state=u_ref,
                        n=n_sites,
                        a_state=a_ref,
                        tau_rec=tau_rec,
                        tau_fac=tau_fac,
                        t_last=t_last_ref,
                        t_spike=t_spike,
                        draws=draws,
                        cursor=cursor,
                    )
                    params = syn.get()
                    self.assertAlmostEqual(params['u'], u_ref, delta=1e-12)
                    self.assertEqual(params['a'], a_ref)
                    self.assertAlmostEqual(syn.t_lastspike, t_last_ref, delta=1e-12)
                    release_ref.append(n_release_ref)

            self.assertEqual(cursor, len(draws))

        self.assertEqual(delivered, [0, 0, 1, 0, 0, 1, 0])
        self.assertEqual(len(recv.delta_events), 2)

        expected_payloads = [float(r * weight) for r in release_ref]
        for idx, (_key, value, label) in enumerate(recv.delta_events):
            value_f = float(np.asarray(u.math.asarray(value), dtype=np.float64).reshape(()))
            self.assertAlmostEqual(value_f, expected_payloads[idx], delta=1e-12)
            self.assertEqual(label, 'receptor_2')


class TestQuantalSTPSynapseDynamics(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_mean_converges_to_deterministic_tsodyks2_equivalent(self):
        # Mirrors NEST testsuite/pytests/test_quantal_stp_synapse.py idea:
        # the stochastic model converges to tsodyks2 dynamics in expectation.
        spike_times = np.asarray(
            [30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0, 390.0, 900.0],
            dtype=np.float64,
        )
        dt_ms = 0.1
        n_sites = 12
        n_trials = 600

        U = 0.03
        u0 = 0.03
        tau_fac = 500.0
        tau_rec = 200.0

        quantal_weight = 1.0 / float(n_sites)
        deterministic_weight = 1.0

        deterministic = _tsodyks2_reference_weights(
            spike_times_ms=spike_times,
            U=U,
            u0=u0,
            x0=1.0,
            tau_rec=tau_rec,
            tau_fac=tau_fac,
            weight=deterministic_weight,
        )

        trial_weights = np.asarray(
            [
                _run_quantal_spike_weights(
                    spike_times_ms=spike_times,
                    dt_ms=dt_ms,
                    U=U,
                    u0=u0,
                    tau_rec=tau_rec,
                    tau_fac=tau_fac,
                    n_sites=n_sites,
                    weight=quantal_weight,
                    seed=1000 + trial,
                )
                for trial in range(n_trials)
            ],
            dtype=np.float64,
        )
        mean_weights = np.mean(trial_weights, axis=0)

        # Sampling uncertainty bound for 600 trials and n=12 release sites.
        np.testing.assert_allclose(mean_weights, deterministic, atol=1.5e-2, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
