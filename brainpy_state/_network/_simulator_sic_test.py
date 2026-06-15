# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Simulator wiring of the one-way astrocyte->neuron SIC connection (15d).

NEST-free integration tests of ``connect(astro, neuron,
synapse=sic_connection(weight=w))``: the Simulator dispatches a ``sic_connection``
to an ``as_current`` :class:`EventProjection` that reads the astrocyte's emission
holder (the per-step graded ``SIC``) and deposits ``weight·SIC`` into the
neuron's labelled ``'I_SIC'`` current channel. The connection is one-way and
sender/receiver-enforced (``astrocyte_lr_1994`` -> ``aeif_cond_alpha_astro``).

The neuron->astrocyte direction is the ordinary delta path (presynaptic spikes ->
``Δ_IP3·w`` IP3 increment via ``sum_delta_inputs``); both directions carry the
substrate's intrinsic one-step pipeline lag.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import (aeif_cond_alpha_astro, astrocyte_lr_1994,
                           sic_connection, multimeter, iaf_psc_alpha)
from brainpy_state._network import Simulator
from brainpy_state._network._event_proj import EventProjection


def _trace(res, rec, name):
    return np.asarray(u.get_mantissa(res.trace(rec, name)))


class TestSICConnectionWiring(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_connect_builds_as_current_sic_projection(self):
        sim = Simulator(dt=0.1 * u.ms)
        astro = sim.create(astrocyte_lr_1994, 1)
        neuron = sim.create(aeif_cond_alpha_astro, 1)
        proj = sim.connect(astro, neuron, synapse=sic_connection(weight=0.5))
        self.assertIsInstance(proj, EventProjection)
        self.assertTrue(proj._as_current)
        self.assertEqual(proj._channel_label, 'I_SIC')

    def test_invalid_sender_raises(self):
        # A sic_connection source must be an astrocyte_lr_1994 (sends SICEvent).
        sim = Simulator(dt=0.1 * u.ms)
        bad = sim.create(iaf_psc_alpha, 1)
        neuron = sim.create(aeif_cond_alpha_astro, 1)
        with self.assertRaises(ValueError):
            sim.connect(bad, neuron, synapse=sic_connection())

    def test_invalid_receiver_raises(self):
        # A sic_connection target must be an aeif_cond_alpha_astro (handles SICEvent).
        sim = Simulator(dt=0.1 * u.ms)
        astro = sim.create(astrocyte_lr_1994, 1)
        bad = sim.create(iaf_psc_alpha, 1)
        with self.assertRaises(ValueError):
            sim.connect(astro, bad, synapse=sic_connection())

    def test_sparse_sic_rejected(self):
        # Graded current needs the dense matmul; sparse binarises the pre value.
        sim = Simulator(dt=0.1 * u.ms)
        astro = sim.create(astrocyte_lr_1994, 1)
        neuron = sim.create(aeif_cond_alpha_astro, 1)
        with self.assertRaises(ValueError):
            sim.connect(astro, neuron, synapse=sic_connection(), comm='sparse')

    def _run_sic_loop(self, sic_weight, duration=40.0 * u.ms):
        # Astrocyte initialised with Ca above SIC_th (0.5 >> 0.19669) so it emits
        # SIC > 0 immediately; the only driver of the resting neuron is the SIC
        # current, so V reflects exactly what the projection delivers.
        sim = Simulator(dt=0.1 * u.ms)
        astro = sim.create(astrocyte_lr_1994, 1, params={'Ca_initializer': 0.5})
        neuron = sim.create(aeif_cond_alpha_astro, 1)
        sim.connect(astro, neuron, synapse=sic_connection(weight=sic_weight))
        mm = sim.create(multimeter, record_from=['V_m', 'I_SIC'])
        sim.connect(mm, neuron)
        res = sim.simulate(duration)
        v = _trace(res, mm, 'V_m').reshape(-1)
        i_sic = _trace(res, mm, 'I_SIC').reshape(-1)
        return v, i_sic

    def test_sic_modulates_postsynaptic_membrane(self):
        v_on, i_sic_on = self._run_sic_loop(sic_weight=5.0)
        v_off, i_sic_off = self._run_sic_loop(sic_weight=0.0)
        # SIC current is delivered (max > 0) only when the connection is live.
        self.assertGreater(float(np.max(i_sic_on)), 0.0)
        self.assertEqual(float(np.max(np.abs(i_sic_off))), 0.0)
        # A positive SIC current depolarises the otherwise-resting neuron.
        self.assertGreater(float(v_on[-1]), float(v_off[-1]) + 1e-4)

    def test_weight_zero_decouples_loop(self):
        _v, i_sic = self._run_sic_loop(sic_weight=0.0)
        npt_max = float(np.max(np.abs(i_sic)))
        self.assertEqual(npt_max, 0.0)

    def test_neuron_to_astro_delta_raises_ip3(self):
        # The neuron->astrocyte delta path: presynaptic spikes increment IP3 via
        # sum_delta_inputs. Drive the neuron above rheobase and record astro IP3.
        sim = Simulator(dt=0.1 * u.ms)
        neuron = sim.create(aeif_cond_alpha_astro, 1, params={'I_e': 800.0 * u.pA})
        astro = sim.create(astrocyte_lr_1994, 1, params={'delta_IP3': 2.0})
        sim.connect(neuron, astro, weight=1.0)
        mm = sim.create(multimeter, record_from=['IP3'])
        sim.connect(mm, astro)
        res = sim.simulate(100.0 * u.ms)
        ip3 = _trace(res, mm, 'IP3').reshape(-1)
        self.assertGreater(float(np.max(ip3)), float(ip3[0]) + 0.05)

    def test_simulate_lowers_with_stable_trace_shapes(self):
        # The whole bidirectional loop runs under the Simulator's for_loop with
        # stable (n_steps, 1) trace shapes (no carry-shape collapse).
        v, i_sic = self._run_sic_loop(sic_weight=1.0, duration=20.0 * u.ms)
        self.assertEqual(v.shape, (200,))
        self.assertEqual(i_sic.shape, (200,))
        self.assertTrue(np.all(np.isfinite(v)))


if __name__ == '__main__':
    unittest.main()
