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
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy.state import iaf_psc_exp_htum, tsodyks_synapse

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


class _MockReceiver:
    def __init__(self):
        self.delta_events = []

    def add_delta_input(self, key, inp, label=None):
        self.delta_events.append((key, inp, label))


def _spike_steps_from_times(spike_times_ms, dt_ms):
    return {int(round((float(t_ms) - dt_ms) / dt_ms)) for t_ms in spike_times_ms}


def _run_tsodyks_vm_trace(
    *,
    h_ms: float,
    Tau: float,
    Theta: float,
    E_L: float,
    TauR: float,
    Tau_psc: float,
    Tau_rec: float,
    Tau_fac: float,
    U: float,
    A: float,
    input_train,
    T_sim: float,
    vm_interval_ms: float = 25.0,
    source_to_synapse_delay_ms: float = 1.0,
):
    dt = h_ms * u.ms
    # NEST reference test routes spikes through ``parrot_neuron`` because
    # devices cannot project to plastic synapses. With default synapse settings
    # this adds 1.0 ms before spikes reach ``tsodyks_synapse``.
    dftype = brainstate.environ.dftype()
    synapse_spike_times = np.asarray(input_train, dtype=dftype) + float(source_to_synapse_delay_ms)
    spike_steps = _spike_steps_from_times(synapse_spike_times, h_ms)
    sim_steps = int(round(T_sim / h_ms))
    C = Tau

    with brainstate.environ.context(dt=dt):
        dftype = brainstate.environ.dftype()

        # Pre-simulate synapse STP dynamics in pure Python to collect the per-step
        # delta input that will be delivered to the neuron at each simulation step.
        # tsodyks_synapse stores state as plain Python floats (not JAX arrays),
        # so this fast Python loop replaces the JAX-un-traceable synapse calls.
        delay_steps_int = int(round(0.1 / h_ms))
        x_py, y_py, u_py, t_last_py = 1.0, 0.0, 0.0, 0.0
        syn_queue: dict = {}
        delta_inputs_arr = np.zeros(sim_steps, dtype=dftype)
        for step in range(sim_steps):
            delta_inputs_arr[step] = syn_queue.pop(step, 0.0)
            if step in spike_steps:
                t_spike = (step + 1) * h_ms
                x_py, y_py, u_py, t_last_py, delta_y = _tsodyks_reference_step(
                    x=x_py,
                    y=y_py,
                    u_state=u_py,
                    t_last=t_last_py,
                    t_spike=t_spike,
                    tau_psc=Tau_psc,
                    tau_fac=Tau_fac,
                    tau_rec=Tau_rec,
                    U=U,
                )
                ds = step + delay_steps_int
                if ds < sim_steps:
                    syn_queue[ds] = syn_queue.get(ds, 0.0) + delta_y * A
        # pA-valued JAX array: delta_inputs_jax[k] is the current injected at step k
        delta_inputs_jax = jnp.asarray(delta_inputs_arr, dtype=dftype) * u.pA

        neuron = iaf_psc_exp_htum(
            1,
            tau_m=Tau * u.ms,
            t_ref_tot=TauR * u.ms,
            t_ref_abs=TauR * u.ms,
            tau_syn_ex=Tau_psc * u.ms,
            tau_syn_in=Tau_psc * u.ms,
            C_m=C * u.pF,
            V_th=Theta * u.mV,
            V_reset=E_L * u.mV,
            E_L=E_L * u.mV,
            I_e=0.0 * u.pA,
            V_initializer=braintools.init.Constant(E_L * u.mV),
        )
        neuron.init_state()

        def _step_body(k):
            neuron.add_delta_input('syn', delta_inputs_jax[k])
            with brainstate.environ.context(t=k * dt):
                neuron.update(x=0.0 * u.pA)
            return (neuron.V.value / u.mV)[0]

        v_all = brainstate.transform.for_loop(_step_body, jnp.arange(sim_steps))

    # Extract sampled time points (every vm_interval_ms, excluding the final step)
    sample_steps = [
        step
        for step in range(sim_steps)
        if (
            (step + 1) * h_ms < T_sim - 1e-12
            and math.isclose(
                (step + 1) * h_ms / vm_interval_ms,
                round((step + 1) * h_ms / vm_interval_ms),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    ]
    times = np.array([(step + 1) * h_ms for step in sample_steps], dtype=dftype)
    vm = np.array([float(np.asarray(v_all[step])) for step in sample_steps], dtype=dftype)
    return np.vstack([times, vm]).T


def _tsodyks_reference_step(*, x, y, u_state, t_last, t_spike, tau_psc, tau_fac, tau_rec, U):
    h = t_spike - t_last
    puu = 0.0 if tau_fac == 0.0 else math.exp(-h / tau_fac)
    pyy = math.exp(-h / tau_psc)
    pzz = math.expm1(-h / tau_rec)
    pxy = (pzz * tau_rec - (pyy - 1.0) * tau_psc) / (tau_psc - tau_rec)

    z = 1.0 - x - y
    u_state = u_state * puu
    x = x + pxy * y - pzz * z
    y = y * pyy
    u_state = u_state + U * (1.0 - u_state)

    delta_y = u_state * x
    x = x - delta_y
    y = y + delta_y
    return x, y, u_state, t_spike, delta_y


class TestTsodyksSynapseParameters(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_nest_like_defaults(self):
        with brainstate.environ.context(dt=1.0 * u.ms, t=0.0 * u.ms):
            syn = tsodyks_synapse()
            syn.init_state()
            syn.update(pre_spike=0.0)
            params = syn.get()

        self.assertEqual(params['weight'], 1.0)
        self.assertEqual(params['delay'], 1.0)
        self.assertEqual(params['delay_steps'], 1)
        self.assertEqual(params['receptor_type'], 0)
        self.assertEqual(params['event_type'], 'spike')
        self.assertEqual(params['tau_psc'], 3.0)
        self.assertEqual(params['tau_fac'], 0.0)
        self.assertEqual(params['tau_rec'], 800.0)
        self.assertEqual(params['U'], 0.5)
        self.assertEqual(params['x'], 1.0)
        self.assertEqual(params['y'], 0.0)
        self.assertEqual(params['u'], 0.0)
        self.assertEqual(params['synapse_model'], 'tsodyks_synapse')

    def test_parameter_validation_matches_nest(self):
        with self.assertRaisesRegex(ValueError, "'U' must be in \\[0,1\\]"):
            tsodyks_synapse(U=1.1)
        with self.assertRaisesRegex(ValueError, "'U' must be in \\[0,1\\]"):
            tsodyks_synapse(U=-0.1)
        with self.assertRaisesRegex(ValueError, "'tau_psc' must be > 0."):
            tsodyks_synapse(tau_psc=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "'tau_rec' must be > 0."):
            tsodyks_synapse(tau_rec=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "'tau_fac' must be >= 0."):
            tsodyks_synapse(tau_fac=-1.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "'u' must be in \\[0,1\\]"):
            tsodyks_synapse(u=1.1)
        with self.assertRaisesRegex(ValueError, 'x \\+ y must be <= 1.0'):
            tsodyks_synapse(x=0.8, y=0.3)

        syn = tsodyks_synapse()
        with self.assertRaisesRegex(ValueError, "'tau_psc' must be > 0."):
            syn.set(tau_psc=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "'tau_rec' must be > 0."):
            syn.set(tau_rec=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "'tau_fac' must be >= 0."):
            syn.set(tau_fac=-1.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "'U' must be in \\[0,1\\]"):
            syn.set(U=1.1)
        with self.assertRaisesRegex(ValueError, "'u' must be in \\[0,1\\]"):
            syn.set(u=-0.1)
        with self.assertRaisesRegex(ValueError, 'x \\+ y must be <= 1.0'):
            syn.set(x=0.7, y=0.4)


class TestTsodyksSynapseOrdering(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_state_update_order_and_effective_weight(self):
        recv = _MockReceiver()
        dt_ms = 1.0
        dt = dt_ms * u.ms

        tau_psc = 3.0
        tau_fac = 0.0
        tau_rec = 800.0
        U = 0.5
        weight = 250.0
        spike_steps = [97, 147, 197]

        with brainstate.environ.context(dt=dt):
            syn = tsodyks_synapse(
                tau_psc=tau_psc * u.ms,
                tau_fac=tau_fac * u.ms,
                tau_rec=tau_rec * u.ms,
                U=U,
                x=1.0,
                y=0.0,
                u=0.0,
                delay=1.0 * u.ms,
                weight=weight,
                receptor_type=2,
                post=recv,
            )
            syn.init_state()

            x_ref, y_ref, u_ref, t_last_ref = 1.0, 0.0, 0.0, 0.0
            payloads_ref = []
            for step in range(202):
                pre_spike = 1.0 if step in spike_steps else 0.0
                with brainstate.environ.context(t=step * dt):
                    syn.update(pre_spike=pre_spike)

                if step in spike_steps:
                    t_spike = (step + 1) * dt_ms
                    x_ref, y_ref, u_ref, t_last_ref, delta_y = _tsodyks_reference_step(
                        x=x_ref,
                        y=y_ref,
                        u_state=u_ref,
                        t_last=t_last_ref,
                        t_spike=t_spike,
                        tau_psc=tau_psc,
                        tau_fac=tau_fac,
                        tau_rec=tau_rec,
                        U=U,
                    )
                    params = syn.get()
                    self.assertAlmostEqual(params['x'], x_ref, delta=1e-12)
                    self.assertAlmostEqual(params['y'], y_ref, delta=1e-12)
                    self.assertAlmostEqual(params['u'], u_ref, delta=1e-12)
                    payloads_ref.append(delta_y * weight)

        self.assertEqual(len(recv.delta_events), len(spike_steps))
        for idx, (_key, value, label) in enumerate(recv.delta_events):
            dftype = brainstate.environ.dftype()
            value_f = float(np.asarray(u.math.asarray(value), dtype=dftype).reshape(()))
            self.assertAlmostEqual(value_f, payloads_ref[idx], delta=1e-12)
            self.assertEqual(label, 'receptor_2')


class TestTsodyksSynapseDynamics(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_depressing_trace_matches_nest_reference(self):
        dftype = brainstate.environ.dftype()
        times_vm_sim = _run_tsodyks_vm_trace(
            h_ms=0.1,
            Tau=40.0,
            Theta=15.0,
            E_L=0.0,
            TauR=2.0,
            Tau_psc=3.0,
            Tau_rec=800.0,
            Tau_fac=0.0,
            U=0.5,
            A=250.0,
            input_train=np.arange(98.0, 1048.1, 50.0, dtype=dftype),
            T_sim=1200.0,
        )

        times_vm_expected = np.array(
            [
                [25, 0],
                [50, 0],
                [75, 0],
                [100, 2.401350000000000e00],
                [125, 5.302440000000000e00],
                [150, 4.108330000000000e00],
                [175, 4.322170000000000e00],
                [200, 3.053390000000000e00],
                [225, 2.871240000000000e00],
                [250, 2.028640000000000e00],
                [275, 1.908020000000000e00],
                [300, 1.396960000000000e00],
                [325, 1.375850000000000e00],
                [350, 1.057780000000000e00],
                [375, 1.103490000000000e00],
                [400, 8.865690000000001e-01],
                [425, 9.693530000000000e-01],
                [450, 8.028760000000000e-01],
                [475, 9.046710000000000e-01],
                [500, 7.626880000000000e-01],
                [525, 8.738550000000000e-01],
                [550, 7.435880000000000e-01],
                [575, 8.592780000000000e-01],
                [600, 7.345670000000000e-01],
                [625, 8.524119999999999e-01],
                [650, 7.303210000000000e-01],
                [675, 8.491860000000000e-01],
                [700, 7.283280000000000e-01],
                [725, 8.476730000000000e-01],
                [750, 7.273930000000000e-01],
                [775, 8.469640000000001e-01],
                [800, 7.269550000000000e-01],
                [825, 8.466320000000001e-01],
                [850, 7.267500000000000e-01],
                [875, 8.464760000000000e-01],
                [900, 7.266540000000000e-01],
                [925, 8.464030000000000e-01],
                [950, 7.266089999999999e-01],
                [975, 8.463690000000000e-01],
                [1000, 7.265880000000000e-01],
                [1025, 8.463530000000000e-01],
                [1050, 7.265779999999999e-01],
                [1075, 8.463460000000000e-01],
                [1100, 4.531260000000000e-01],
                [1125, 2.425410000000000e-01],
                [1150, 1.298230000000000e-01],
                [1175, 6.948920000000000e-02],
            ],
            dtype=dftype,
        )

        np.testing.assert_allclose(times_vm_sim, times_vm_expected, atol=1e-5, rtol=0.0)

    def test_facilitating_trace_matches_nest_reference(self):
        dftype = brainstate.environ.dftype()
        times_vm_sim = _run_tsodyks_vm_trace(
            h_ms=0.1,
            Tau=60.0,
            Theta=15.0,
            E_L=0.0,
            TauR=2.0,
            Tau_psc=1.5,
            Tau_rec=130.0,
            Tau_fac=530.0,
            U=0.03,
            A=1540.0,
            input_train=np.arange(98.1, 1051.0, 50.1, dtype=dftype),
            T_sim=1200.0,
        )

        times_vm_expected = np.array(
            [
                [25, 0],
                [50, 0],
                [75, 0],
                [100, 4.739750000000000e-01],
                [125, 7.706030000000000e-01],
                [150, 1.297120000000000e00],
                [175, 1.757990000000000e00],
                [200, 2.114420000000000e00],
                [225, 2.714490000000000e00],
                [250, 2.785530000000000e00],
                [275, 3.546150000000000e00],
                [300, 3.272710000000000e00],
                [325, 4.233240000000000e00],
                [350, 3.583070000000000e00],
                [375, 4.788130000000000e00],
                [400, 3.739560000000000e00],
                [425, 5.233850000000000e00],
                [450, 3.767230000000000e00],
                [475, 5.593890000000000e00],
                [500, 3.687720000000000e00],
                [525, 5.888270000000000e00],
                [550, 3.881790000000000e00],
                [575, 6.132580000000000e00],
                [600, 4.042850000000000e00],
                [625, 6.338430000000000e00],
                [650, 4.178550000000000e00],
                [675, 6.514250000000000e00],
                [700, 4.294460000000000e00],
                [725, 6.666160000000000e00],
                [750, 4.394600000000000e00],
                [775, 6.798620000000000e00],
                [800, 4.481930000000000e00],
                [825, 6.914990000000000e00],
                [850, 4.558640000000000e00],
                [875, 7.017830000000000e00],
                [900, 4.626440000000000e00],
                [925, 7.109160000000000e00],
                [950, 4.686650000000000e00],
                [975, 7.190620000000000e00],
                [1000, 4.740350000000000e00],
                [1025, 7.263560000000000e00],
                [1050, 4.788430000000000e00],
                [1075, 7.329110000000000e00],
                [1100, 4.831650000000000e00],
                [1125, 3.185220000000000e00],
                [1150, 2.099830000000000e00],
                [1175, 1.384290000000000e00],
            ],
            dtype=dftype,
        )

        np.testing.assert_allclose(times_vm_sim, times_vm_expected, atol=1e-5, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
