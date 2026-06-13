# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Simulator device→compartment routing for ``iaf_cond_alpha_mc`` (NEST-free).

``iaf_cond_alpha_mc`` has 9 receptors: spike receptors 1-6 map to six labeled
conductance channels (soma/proximal/distal × exc/inh) and current receptors 7-9
map to the three compartments' injected current. These tests pin that
``connect(device, mc, receptor_type=k)`` lands on the right channel/compartment
and nowhere else, and that mis-typed connections are rejected eagerly.

* spike 1-6 → only the matching ``g_ex.{s,p,d}`` / ``g_in.{s,p,d}`` rises.
* current 7-9 → the directly driven compartment's ``V_m`` departs rest far more
  than a non-adjacent compartment (the pre-fix bug broadcast the current into all
  three compartments at once).
"""
import unittest

import brainstate
import jax
import numpy as np
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import (Simulator, iaf_cond_alpha_mc, spike_generator,
                           dc_generator, multimeter)

DT = 0.1


def _trace(res, mm, name):
    return np.asarray(u.get_mantissa(res.trace(mm, name) / u.nS)).reshape(-1)


def _vtrace(res, mm, name):
    return np.asarray(u.get_mantissa(res.trace(mm, name) / u.mV)).reshape(-1)


class TestSpikeReceptorRouting(unittest.TestCase):
    """Each spike receptor 1-6 drives exactly one conductance channel."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    # receptor_type -> (recordable that must rise, the five that must stay ~0)
    _CASES = {
        1: 'g_ex.s', 2: 'g_in.s', 3: 'g_ex.p',
        4: 'g_in.p', 5: 'g_ex.d', 6: 'g_in.d',
    }
    _ALL = ('g_ex.s', 'g_in.s', 'g_ex.p', 'g_in.p', 'g_ex.d', 'g_in.d')

    def _run(self, receptor_type):
        sim = Simulator(dt=DT * u.ms)
        mc = sim.create(iaf_cond_alpha_mc, 1)
        sg = sim.create(spike_generator, spike_times=np.asarray([5.0]) * u.ms)
        sim.connect(sg, mc, receptor_type=receptor_type, weight=5.0 * u.nS)
        mm = sim.create(multimeter, record_from=list(self._ALL), interval=DT * u.ms)
        sim.connect(mm, mc)
        res = sim.simulate(40.0 * u.ms)
        return {name: _trace(res, mm, name) for name in self._ALL}

    def test_each_spike_receptor_drives_only_its_channel(self):
        for rt, active in self._CASES.items():
            traces = self._run(rt)
            self.assertGreater(float(traces[active].max()), 1e-3,
                               f"rt={rt}: channel {active} should rise")
            for name in self._ALL:
                if name == active:
                    continue
                self.assertLess(float(np.abs(traces[name]).max()), 1e-9,
                                f"rt={rt}: channel {name} must stay zero, "
                                f"got max {np.abs(traces[name]).max()}")


class TestCurrentReceptorRouting(unittest.TestCase):
    """Each current receptor 7-9 injects into exactly one compartment."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def _run(self, receptor_type, amplitude=200.0):
        sim = Simulator(dt=DT * u.ms)
        mc = sim.create(iaf_cond_alpha_mc, 1)
        dc = sim.create(dc_generator, amplitude=amplitude * u.pA,
                        start=10.0 * u.ms, stop=60.0 * u.ms)
        sim.connect(dc, mc, receptor_type=receptor_type)
        mm = sim.create(multimeter, record_from=['V_m.s', 'V_m.p', 'V_m.d'],
                        interval=DT * u.ms)
        sim.connect(mm, mc)
        res = sim.simulate(70.0 * u.ms)
        rest = -70.0
        return {c: np.abs(_vtrace(res, mm, f'V_m.{c}') - rest).max()
                for c in ('s', 'p', 'd')}

    def test_soma_current_moves_soma_more_than_distal(self):
        # rt=7 (soma_curr): soma is driven directly; distal is two hops away
        # (soma<->proximal<->distal), so it should barely move by comparison.
        dev = self._run(7)
        self.assertGreater(dev['s'], 1.0, "soma should depolarize under soma current")
        self.assertLess(dev['d'], dev['s'] * 0.5,
                        f"distal moved too much for a soma-only drive: {dev}")

    def test_distal_current_moves_distal_more_than_soma(self):
        # rt=9 (distal_curr): symmetric — distal driven, soma two hops away.
        dev = self._run(9)
        self.assertGreater(dev['d'], 1.0, "distal should depolarize under distal current")
        self.assertLess(dev['s'], dev['d'] * 0.5,
                        f"soma moved too much for a distal-only drive: {dev}")

    def test_proximal_current_moves_proximal(self):
        # rt=8 (proximal_curr): proximal is driven; it is adjacent to both others,
        # so just assert it is the most-deflected compartment.
        dev = self._run(8)
        self.assertGreater(dev['p'], 1.0, "proximal should depolarize")
        self.assertGreaterEqual(dev['p'], dev['s'])
        self.assertGreaterEqual(dev['p'], dev['d'])


class TestReceptorRoutingErrors(unittest.TestCase):
    """Mis-typed device↔receptor connections are rejected eagerly at connect()."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def _sim_mc(self):
        sim = Simulator(dt=DT * u.ms)
        mc = sim.create(iaf_cond_alpha_mc, 1)
        return sim, mc

    def test_spike_source_with_current_receptor_raises(self):
        sim, mc = self._sim_mc()
        sg = sim.create(spike_generator, spike_times=np.asarray([5.0]) * u.ms)
        with self.assertRaisesRegex(ValueError, 'current receptor'):
            sim.connect(sg, mc, receptor_type=7, weight=5.0 * u.nS)

    def test_current_source_with_spike_receptor_raises(self):
        sim, mc = self._sim_mc()
        dc = sim.create(dc_generator, amplitude=100.0 * u.pA)
        with self.assertRaisesRegex(ValueError, 'spike receptor'):
            sim.connect(dc, mc, receptor_type=1)

    def test_current_source_without_receptor_raises(self):
        sim, mc = self._sim_mc()
        dc = sim.create(dc_generator, amplitude=100.0 * u.pA)
        with self.assertRaisesRegex(ValueError, 'receptor_type'):
            sim.connect(dc, mc)

    def test_out_of_range_receptor_raises(self):
        sim, mc = self._sim_mc()
        sg = sim.create(spike_generator, spike_times=np.asarray([5.0]) * u.ms)
        with self.assertRaisesRegex(ValueError, 'receptor'):
            sim.connect(sg, mc, receptor_type=99, weight=5.0 * u.nS)


if __name__ == '__main__':
    unittest.main()
