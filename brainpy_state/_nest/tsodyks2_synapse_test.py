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

from brainpy.state import tsodyks2_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class _MockReceiver:
    def __init__(self):
        self.delta_events = []

    def add_delta_input(self, key, inp, label=None):
        self.delta_events.append((key, inp, label))


def _spike_steps_from_times(spike_times_ms, dt_ms):
    return {int(round((float(t_ms) - dt_ms) / dt_ms)) for t_ms in spike_times_ms}


def _tsodyks2_reference_step(*, x, u_state, t_last, t_spike, U, tau_rec, tau_fac, weight):
    if t_last >= 0.0:
        h = t_spike - t_last
        x_decay = math.exp(-h / tau_rec)
        u_decay = 0.0 if tau_fac == 0.0 else math.exp(-h / tau_fac)
        x = 1.0 + (x - x * u_state - 1.0) * x_decay
        u_state = U + u_state * (1.0 - U) * u_decay

    w_eff = x * u_state * weight
    return x, u_state, t_spike, w_eff


def _run_tsodyks2_weight_trace(
    *,
    pre_spikes_ms,
    sim_duration_ms,
    dt_ms,
    U,
    u0,
    x0,
    tau_rec,
    tau_fac,
    weight,
):
    recv = _MockReceiver()
    dt = dt_ms * u.ms
    spike_steps = _spike_steps_from_times(pre_spikes_ms, dt_ms)
    sim_steps = 1 + int(np.ceil(sim_duration_ms / dt_ms))

    with brainstate.environ.context(dt=dt):
        syn = tsodyks2_synapse(
            U=U,
            u=u0,
            x=x0,
            tau_rec=tau_rec * u.ms,
            tau_fac=tau_fac * u.ms,
            delay=dt,
            weight=weight,
            post=recv,
        )
        syn.init_state()

        for step in range(sim_steps):
            pre_spike = 1.0 if step in spike_steps else 0.0
            with brainstate.environ.context(t=step * dt):
                syn.update(pre_spike=pre_spike)

    return [
        float(np.asarray(u.math.asarray(value), dtype=np.float64).reshape(()))
        for _key, value, _label in recv.delta_events
    ]


class TestTsodyks2SynapseParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = tsodyks2_synapse()
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
        self.assertEqual(params['x'], 1.0)
        self.assertEqual(params['tau_rec'], 800.0)
        self.assertEqual(params['tau_fac'], 0.0)
        self.assertEqual(params['synapse_model'], 'tsodyks2_synapse')

    def test_u_defaults_to_U_and_state_timestamp_matches_nest(self):
        syn = tsodyks2_synapse(U=0.2)
        self.assertAlmostEqual(syn.get()['u'], 0.2, delta=1e-12)
        self.assertAlmostEqual(syn.t_lastspike, -1.0, delta=1e-12)

    def test_parameter_validation_matches_nest(self):
        with self.assertRaisesRegex(ValueError, r"'U' must be in \[0,1\]\."):
            tsodyks2_synapse(U=1.1)
        with self.assertRaisesRegex(ValueError, r"'U' must be in \[0,1\]\."):
            tsodyks2_synapse(U=-0.1)
        with self.assertRaisesRegex(ValueError, r"'u' must be in \[0,1\]\."):
            tsodyks2_synapse(u=1.1)
        with self.assertRaisesRegex(ValueError, r"'u' must be in \[0,1\]\."):
            tsodyks2_synapse(u=-0.1)
        with self.assertRaisesRegex(ValueError, r"'tau_rec' must be > 0\."):
            tsodyks2_synapse(tau_rec=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, r"'tau_fac' must be >= 0\."):
            tsodyks2_synapse(tau_fac=-1.0 * u.ms)

        syn = tsodyks2_synapse()
        with self.assertRaisesRegex(ValueError, r"'U' must be in \[0,1\]\."):
            syn.set(U=1.1)
        with self.assertRaisesRegex(ValueError, r"'u' must be in \[0,1\]\."):
            syn.set(u=-0.1)
        with self.assertRaisesRegex(ValueError, r"'tau_rec' must be > 0\."):
            syn.set(tau_rec=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, r"'tau_fac' must be >= 0\."):
            syn.set(tau_fac=-1.0 * u.ms)

    def test_set_U_does_not_implicitly_change_u(self):
        syn = tsodyks2_synapse(U=0.4, u=0.25)
        syn.set(U=0.9)
        params = syn.get()
        self.assertAlmostEqual(params['U'], 0.9, delta=1e-12)
        self.assertAlmostEqual(params['u'], 0.25, delta=1e-12)


class TestTsodyks2SynapseOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_state_update_order_and_effective_weight(self):
        recv = _MockReceiver()
        dt_ms = 1.0
        dt = dt_ms * u.ms

        U = 0.6
        u0 = 0.6
        x0 = 1.0
        tau_rec = 100.0
        tau_fac = 20.0
        weight = 2.5
        spike_steps = [4, 11, 23]

        with brainstate.environ.context(dt=dt):
            syn = tsodyks2_synapse(
                U=U,
                u=u0,
                x=x0,
                tau_rec=tau_rec * u.ms,
                tau_fac=tau_fac * u.ms,
                delay=1.0 * u.ms,
                weight=weight,
                receptor_type=2,
                post=recv,
            )
            syn.init_state()

            x_ref, u_ref, t_last_ref = x0, u0, -1.0
            payloads_ref = []
            for step in range(30):
                pre_spike = 1.0 if step in spike_steps else 0.0
                with brainstate.environ.context(t=step * dt):
                    syn.update(pre_spike=pre_spike)

                if step in spike_steps:
                    t_spike = (step + 1) * dt_ms
                    x_ref, u_ref, t_last_ref, w_eff_ref = _tsodyks2_reference_step(
                        x=x_ref,
                        u_state=u_ref,
                        t_last=t_last_ref,
                        t_spike=t_spike,
                        U=U,
                        tau_rec=tau_rec,
                        tau_fac=tau_fac,
                        weight=weight,
                    )
                    params = syn.get()
                    self.assertAlmostEqual(params['x'], x_ref, delta=1e-12)
                    self.assertAlmostEqual(params['u'], u_ref, delta=1e-12)
                    self.assertAlmostEqual(syn.t_lastspike, t_last_ref, delta=1e-12)
                    payloads_ref.append(w_eff_ref)

        self.assertEqual(len(recv.delta_events), len(spike_steps))
        for idx, (_key, value, label) in enumerate(recv.delta_events):
            value_f = float(np.asarray(u.math.asarray(value), dtype=np.float64).reshape(()))
            self.assertAlmostEqual(value_f, payloads_ref[idx], delta=1e-12)
            self.assertEqual(label, 'receptor_2')


class TestTsodyks2SynapseDynamics(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_weight_drift_matches_independent_reference(self):
        # Mirrors NEST testsuite/pytests/test_tsodyks2_synapse.py logic,
        # with deterministic presynaptic spike times.
        pre_spikes = np.asarray([5.0, 8.6, 14.2, 24.0, 31.1, 47.7, 70.3], dtype=np.float64)
        dt_ms = 0.1
        sim_duration_ms = 100.0

        U = 1.0
        u0 = 1.0
        x0 = 1.0
        tau_rec = 100.0
        weight = 1.0

        for tau_fac in (0.0, 10.0):
            with self.subTest(tau_fac=tau_fac):
                simulated_weights = _run_tsodyks2_weight_trace(
                    pre_spikes_ms=pre_spikes,
                    sim_duration_ms=sim_duration_ms,
                    dt_ms=dt_ms,
                    U=U,
                    u0=u0,
                    x0=x0,
                    tau_rec=tau_rec,
                    tau_fac=tau_fac,
                    weight=weight,
                )

                x_ref, u_ref, t_last_ref = x0, u0, -1.0
                expected_weights = []
                for t_spike in pre_spikes:
                    x_ref, u_ref, t_last_ref, w_eff_ref = _tsodyks2_reference_step(
                        x=x_ref,
                        u_state=u_ref,
                        t_last=t_last_ref,
                        t_spike=float(t_spike),
                        U=U,
                        tau_rec=tau_rec,
                        tau_fac=tau_fac,
                        weight=weight,
                    )
                    expected_weights.append(w_eff_ref)

                np.testing.assert_allclose(
                    np.asarray(simulated_weights, dtype=np.float64),
                    np.asarray(expected_weights, dtype=np.float64),
                    atol=1e-12,
                    rtol=0.0,
                )


if __name__ == '__main__':
    unittest.main()
