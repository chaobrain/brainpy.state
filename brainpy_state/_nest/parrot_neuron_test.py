# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for ``parrot_neuron`` — the 1:1 spike relay.

NEST's ``parrot_neuron`` "emits one spike for every incoming spike"; weights on
connections *to* it are ignored (a spike is a spike), weights *from* it are
honored normally. These tests pin that contract on the Simulator seam:

* every incoming spike is re-emitted exactly once (count parity);
* a silent input yields no output;
* the relayed train drives a downstream multi-receptor neuron through
  ``connect(parrot, neuron, receptor_type=k)`` — i.e. the parrot is a valid
  spiking *source*, the role it plays for the Poisson window in the GLIF demos.
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import saiunit as u

from brainpy_state import (Simulator, parrot_neuron, spike_generator,
                           spike_recorder, multimeter, glif_cond)


class TestParrotRelay(unittest.TestCase):
    """The parrot re-emits each incoming spike exactly once."""

    def test_relays_each_spike_one_to_one(self):
        # three well-separated input spikes -> three relayed spikes
        spike_ms = np.asarray([10.0, 20.0, 30.0])
        sim = Simulator(dt=0.1 * u.ms)
        gen = sim.create(spike_generator, spike_times=spike_ms * u.ms)
        parrot = sim.create(parrot_neuron, 1)
        rec = sim.create(spike_recorder)
        # weight TO the parrot is a gate (ignored value), per NEST semantics
        sim.connect(gen, parrot, weight=1.0, delay=1.0 * u.ms)
        sim.connect(parrot, rec)
        res = sim.simulate(60.0 * u.ms)
        self.assertEqual(res.n_events(rec), spike_ms.size)

    def test_silent_input_yields_no_spikes(self):
        sim = Simulator(dt=0.1 * u.ms)
        gen = sim.create(spike_generator, spike_times=np.asarray([]) * u.ms)
        parrot = sim.create(parrot_neuron, 1)
        rec = sim.create(spike_recorder)
        sim.connect(gen, parrot, weight=1.0, delay=1.0 * u.ms)
        sim.connect(parrot, rec)
        res = sim.simulate(50.0 * u.ms)
        self.assertEqual(res.n_events(rec), 0)

    def test_parrot_drives_downstream_receptor_port(self):
        # parrot is a valid spiking *source*: relay one spike into a glif_cond
        # receptor port and confirm that port's conductance rises.
        sim = Simulator(dt=0.1 * u.ms)
        gen = sim.create(spike_generator, spike_times=np.asarray([10.0]) * u.ms)
        parrot = sim.create(parrot_neuron, 1)
        neuron = sim.create(glif_cond, 1, params=dict(tau_syn=(2.0, 8.0),
                                                       E_rev=(0.0, -85.0)))
        mm = sim.create(multimeter, record_from=['g_1', 'g_2'], interval=0.1 * u.ms)
        sim.connect(gen, parrot, weight=1.0, delay=1.0 * u.ms)
        sim.connect(parrot, neuron, receptor_type=1, weight=20.0 * u.nS,
                    delay=1.0 * u.ms)
        sim.connect(mm, neuron)
        res = sim.simulate(80.0 * u.ms)
        g1 = np.asarray(u.get_mantissa(res.trace(mm, 'g_1') / u.nS)).reshape(-1)
        g2 = np.asarray(u.get_mantissa(res.trace(mm, 'g_2') / u.nS)).reshape(-1)
        self.assertGreater(g1.max(), 0.9 * 20.0)   # driven port rises near its weight
        self.assertLess(g2.max(), 1e-9)            # the undriven port stays at zero


if __name__ == '__main__':
    unittest.main()
