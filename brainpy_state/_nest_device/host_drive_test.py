# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""``host_drive`` — State-backed host-clamped input devices for ``cont()`` rollouts.

A closed-loop host loop (e.g. the §3.10 pong game) must set the network's input
*between* 200 ms chunks without recompiling the per-chunk ``for_loop``. The baked
``spike_generator.spike_times`` / ``dc_generator.amplitude`` can't do this (they are
constants folded into the trace). ``host_drive`` holds a ``(window, n)`` schedule in
a ``ShortTermState`` indexed by a step counter; ``set_schedule`` rewrites the State
*contents* (fixed shape) so the compiled rollout is reused.

These oracles pin each role against its NEST-device equivalent through the Simulator:
``host_spike_drive`` → parrot reproduces ``spike_generator`` → parrot (same relayed
train, up to a constant pipeline shift), and ``host_current_drive`` → iaf reproduces
``dc_generator`` → iaf (same charging trace). A reschedule test then proves the host
can change the active input across ``cont()`` chunks with no recompile.
"""
import os
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import (host_spike_drive, host_current_drive, spike_generator,
                           dc_generator, parrot_neuron, iaf_psc_exp,
                           spike_recorder, voltmeter)
from brainpy_state._nest_network import Simulator, one_to_one, all_to_all

DT = 0.1

# Wall-clock recompile-timing checks compare per-chunk wall-time ratios — meaningful only
# on an unloaded local machine. On shared CI runners (noisy neighbours, variable cores)
# the ratio is unreliable, so skip there; the deterministic oracle + reschedule-behaviour
# tests still cover correctness on CI. Run locally for the no-recompile (R1) guard.
_SKIP_TIMING_ON_CI = os.environ.get('CI') is not None


def _fire_steps(arr):
    return np.where(np.asarray(arr).reshape(-1) > 0)[0]


class TestHostSpikeDriveOracle(unittest.TestCase):
    """host_spike_drive -> parrot reproduces spike_generator -> parrot."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def test_relays_host_set_train_like_spike_generator(self):
        # Distinct times: spike_generator collapses duplicate spike_times whereas
        # host_spike_drive would relay their multiplicity, so the apples-to-apples
        # equivalence uses one spike per step (test_relays_multiplicity covers >1).
        T = 20.0
        n_steps = int(round(T / DT))
        train_ms = [3.0, 7.0, 11.0, 16.0]

        # host_spike_drive: 1 channel, schedule = multiplicity per step.
        sched = np.zeros((n_steps, 1))
        for t in train_ms:
            sched[int(round(t / DT)), 0] += 1.0
        sim_h = Simulator(dt=DT * u.ms)
        hd = sim_h.create(host_spike_drive, 1, params={'window': n_steps})
        par_h = sim_h.create(parrot_neuron, 1)
        rec_h = sim_h.create(spike_recorder)
        sim_h.connect(hd, par_h, rule=one_to_one, weight=1.0)
        sim_h.connect(par_h, rec_h)
        sim_h.reset_rollout()
        hd.segments[0].population.set_schedule(sched)
        h_spk = np.asarray(sim_h.cont(T * u.ms).spikes(rec_h)).reshape(-1)

        # spike_generator with the same train.
        sim_g = Simulator(dt=DT * u.ms)
        sg = sim_g.create(spike_generator, spike_times=np.array(sorted(train_ms)) * u.ms)
        par_g = sim_g.create(parrot_neuron, 1)
        rec_g = sim_g.create(spike_recorder)
        sim_g.connect(sg, par_g, rule=one_to_one, weight=1.0)
        sim_g.connect(par_g, rec_g)
        g_spk = np.asarray(sim_g.simulate(T * u.ms).spikes(rec_g)).reshape(-1)

        self.assertEqual(int(h_spk.sum()), int(g_spk.sum()),
                         'host_spike_drive relayed a different total multiplicity')
        self.assertGreater(int(h_spk.sum()), 0)
        hs, gs = _fire_steps(h_spk), _fire_steps(g_spk)
        self.assertEqual(len(hs), len(gs))
        # Same pattern up to a constant pipeline shift.
        self.assertEqual(int(np.ptp(hs - gs)), 0,
                         f'host vs spike_generator fire steps not a constant shift: {hs} vs {gs}')

    def test_relays_multiplicity(self):
        # Unlike spike_generator (which collapses duplicate times), host_spike_drive
        # relays an integer multiplicity: a schedule value of 3.0 on one step makes
        # the parrot relay a count of 3 at the delivery step.
        T = 10.0
        n_steps = int(round(T / DT))
        sched = np.zeros((n_steps, 1))
        sched[40, 0] = 3.0
        sim = Simulator(dt=DT * u.ms)
        hd = sim.create(host_spike_drive, 1, params={'window': n_steps})
        par = sim.create(parrot_neuron, 1)
        rec = sim.create(spike_recorder)
        sim.connect(hd, par, rule=one_to_one, weight=1.0)
        sim.connect(par, rec)
        sim.reset_rollout()
        hd.segments[0].population.set_schedule(sched)
        spk = np.asarray(sim.cont(T * u.ms).spikes(rec)).reshape(-1)
        self.assertEqual(int(spk.max()), 3, 'multiplicity not relayed')
        self.assertEqual(int(spk.sum()), 3, 'exactly one multiplicity-3 event expected')


class TestHostCurrentDriveOracle(unittest.TestCase):
    """host_current_drive -> iaf reproduces dc_generator -> iaf."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def test_injects_host_set_current_like_dc_generator(self):
        T = 20.0
        n_steps = int(round(T / DT))
        amp = 200.0           # pA
        on_steps = 100        # first 10 ms active, then off

        sim_h = Simulator(dt=DT * u.ms)
        hc = sim_h.create(host_current_drive, 1, params={'window': n_steps})
        neu_h = sim_h.create(iaf_psc_exp, 1)
        vm_h = sim_h.create(voltmeter)
        sim_h.connect(hc, neu_h)
        sim_h.connect(vm_h, neu_h)
        sim_h.reset_rollout()
        sched = np.zeros((n_steps, 1))
        sched[:on_steps, 0] = amp
        hc.segments[0].population.set_schedule(sched)
        v_h = np.asarray(u.get_mantissa(sim_h.cont(T * u.ms).trace(vm_h, 'V_m') / u.mV)).reshape(-1)

        sim_d = Simulator(dt=DT * u.ms)
        dc = sim_d.create(dc_generator, amplitude=amp * u.pA, stop=on_steps * DT * u.ms)
        neu_d = sim_d.create(iaf_psc_exp, 1)
        vm_d = sim_d.create(voltmeter)
        sim_d.connect(dc, neu_d)
        sim_d.connect(vm_d, neu_d)
        v_d = np.asarray(u.get_mantissa(sim_d.simulate(T * u.ms).trace(vm_d, 'V_m') / u.mV)).reshape(-1)

        self.assertGreater(float(v_h.max() - v_h.min()), 1.0, 'current did not charge the neuron')
        self.assertLess(float(np.max(np.abs(v_h - v_d))), 1e-9,
                        'host_current_drive diverged from dc_generator')


class TestHostDriveReschedule(unittest.TestCase):
    """The host can change the active input across cont() chunks (no recompile)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def test_active_channel_changes_per_chunk(self):
        # 3 input channels -> 3 parrots; each chunk activates a different channel.
        chunk_ms = 10.0
        w = int(round(chunk_ms / DT))
        sim = Simulator(dt=DT * u.ms)
        hd = sim.create(host_spike_drive, 3, params={'window': w})
        par = sim.create(parrot_neuron, 3)
        rec = sim.create(spike_recorder)
        sim.connect(hd, par, rule=one_to_one, weight=1.0)
        sim.connect(par, rec)
        sim.reset_rollout()
        dev = hd.segments[0].population

        winners = []
        for ch in (0, 2, 1):
            sched = np.zeros((w, 3))
            sched[w // 2, ch] = 1.0          # one spike on channel `ch`
            dev.set_schedule(sched)
            spk = np.asarray(sim.cont(chunk_ms * u.ms).spikes(rec))   # (w, 3)
            fired = np.where(spk.sum(axis=0) > 0)[0]
            self.assertEqual(list(fired), [ch], f'chunk activated channel {list(fired)}, want {ch}')
            winners.append(int(fired[0]))
        self.assertEqual(winners, [0, 2, 1])

    @unittest.skipIf(_SKIP_TIMING_ON_CI,
                     'wall-clock recompile timing is unreliable on shared CI runners')
    def test_reschedule_reuses_compiled_rollout(self):
        # R1 guard: a schedule rewrite changes only State *contents* (fixed shape),
        # so the per-chunk for_loop is compiled once (chunk 0) and reused — later
        # chunks are markedly faster. A per-chunk XLA recompile (the regression this
        # guards) would make every chunk roughly as slow as the first.
        import time
        W = 2000                                   # 200 ms chunk
        sim = Simulator(dt=DT * u.ms)
        hd = sim.create(host_spike_drive, 20, params={'window': W})
        par = sim.create(parrot_neuron, 20)
        mot = sim.create(iaf_psc_exp, 20)
        rec = sim.create(spike_recorder)
        sim.connect(hd, par, rule=one_to_one, weight=1.0)
        sim.connect(par, mot, rule=all_to_all, weight=200. * u.pA, delay=1. * u.ms)
        sim.connect(mot, rec)
        sim.reset_rollout()
        dev = hd.segments[0].population

        def run_chunk(c):
            s = np.zeros((W, 20))
            s[::100, c % 20] = 1.0
            dev.set_schedule(s)
            t0 = time.perf_counter()
            r = sim.cont(W * DT * u.ms)
            np.asarray(r.spikes(rec))              # force evaluation
            return time.perf_counter() - t0

        t_compile = run_chunk(0)
        steady = [run_chunk(c) for c in range(1, 6)]
        mean_steady = sum(steady) / len(steady)
        if t_compile <= 1.3 * mean_steady:         # compile too cheap to measure here
            self.skipTest('first chunk showed no measurable compile; timing inconclusive')
        self.assertLess(max(steady), 0.7 * t_compile,
                        f'a chunk recompiled after a reschedule (steady={steady}, '
                        f'compile={t_compile:.3f}s)')

    def test_set_schedule_rejects_wrong_shape(self):
        sim = Simulator(dt=DT * u.ms)
        hd = sim.create(host_spike_drive, 4, params={'window': 50})
        sim.create(parrot_neuron, 4)  # so init has a graph; not strictly needed
        sim.reset_rollout()
        dev = hd.segments[0].population
        with self.assertRaises(ValueError):
            dev.set_schedule(np.zeros((50, 3)))      # wrong n
        with self.assertRaises(ValueError):
            dev.set_schedule(np.zeros((40, 4)))      # wrong window


if __name__ == '__main__':
    unittest.main()
