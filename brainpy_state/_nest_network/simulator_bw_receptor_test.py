# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for the ``iaf_bw_2001`` receptor + presynaptic-emission seam.

These pin the recurrent-NMDA substrate (design question A, goal 22):

* ``iaf_bw_2001`` declares the graded-emission attributes and a
  ``delta_label_for_receptor`` resolver so ``Simulator.connect(receptor_type=k)``
  routes AMPA/GABA/NMDA into the named delta channels the model reads.
* ``Simulator._resolve_stp_emission`` delivers ``weight * spike_offset`` over the
  NMDA receptor (graded, dense) while preserving the receptor routing, and rejects
  the binarizing sparse path; the ``iaf_tum_2000`` TSODYKS path is unchanged.
"""
import unittest

import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import brainunit as u
import braintools


import numpy as np

from brainpy_state import (iaf_bw_2001, iaf_tum_2000, Simulator,
                           spike_generator, all_to_all)

#: Shared neuron params (Wang family); pin V_m IC to E_L (the -70 trap).
_BW = dict(V_th=-50.0 * u.mV, V_reset=-55.0 * u.mV,
           V_initializer=braintools.init.Constant(-70.0 * u.mV))


def _bw_pop(sim, n):
    return sim.create(iaf_bw_2001, n, params=dict(_BW))


class TestBwDeltaLabel(unittest.TestCase):
    """``iaf_bw_2001`` exposes the receptor->channel-label resolver + emission attrs."""

    def _neuron(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            return iaf_bw_2001(2)

    def test_label_mapping(self):
        n = self._neuron()
        self.assertEqual(n.delta_label_for_receptor(1), 'AMPA')
        self.assertEqual(n.delta_label_for_receptor(2), 'GABA')
        self.assertEqual(n.delta_label_for_receptor(3), 'NMDA')
        self.assertEqual(n.delta_label_for_receptor('NMDA'), 'NMDA')

    def test_label_out_of_range_raises(self):
        n = self._neuron()
        with self.assertRaises(ValueError):
            n.delta_label_for_receptor(4)

    def test_emission_attrs_declared(self):
        self.assertEqual(iaf_bw_2001._emission_attr, 'spike_offset')
        self.assertEqual(iaf_bw_2001._emission_receptor, iaf_bw_2001.NMDA)


class TestBwReceptorRouting(unittest.TestCase):
    """A Simulator ``connect(receptor_type=k)`` reaches the right delta channel."""

    def _drive_single(self, rt):
        """Drive one neuron over receptor ``rt`` from a spike generator; return the neuron."""
        sim = Simulator(dt=0.1 * u.ms)
        pop = _bw_pop(sim, 1)
        gen = sim.create(spike_generator, 1, spike_times=[1.0] * u.ms)
        sim.connect(gen, pop, weight=5.0 * u.nS, delay=0.1 * u.ms, receptor_type=rt)
        sim.simulate(2.0 * u.ms)
        return pop.segments[0].population

    def test_ampa_routes_to_s_ampa_only(self):
        n = self._drive_single(1)
        self.assertGreater(float(n.s_AMPA.value[0] / u.nS), 0.0)
        self.assertEqual(float(n.s_GABA.value[0] / u.nS), 0.0)
        self.assertEqual(float(n.s_NMDA.value[0] / u.nS), 0.0)

    def test_gaba_routes_to_s_gaba_only(self):
        n = self._drive_single(2)
        self.assertGreater(float(n.s_GABA.value[0] / u.nS), 0.0)
        self.assertEqual(float(n.s_AMPA.value[0] / u.nS), 0.0)
        self.assertEqual(float(n.s_NMDA.value[0] / u.nS), 0.0)


class TestEmissionResolution(unittest.TestCase):
    """``_resolve_stp_emission`` generalization + back-compat."""

    def test_sparse_emission_rejected(self):
        sim = Simulator(dt=0.1 * u.ms)
        pool = _bw_pop(sim, 3)
        with self.assertRaises(ValueError):
            sim.connect(pool, pool, weight=1.0 * u.nS, delay=0.5 * u.ms,
                        rule=all_to_all, receptor_type=iaf_bw_2001.NMDA,
                        comm='sparse', allow_autapses=False)

    def test_tum_tsodyks_path_unchanged(self):
        """iaf_tum_2000 TSODYKS connect still builds a projection (collapse-to-None)."""
        sim = Simulator(dt=0.1 * u.ms)
        pre = sim.create(iaf_tum_2000, 2)
        post = sim.create(iaf_tum_2000, 2)
        proj = sim.connect(pre, post, weight=1.0 * u.pA, delay=0.1 * u.ms,
                           rule=all_to_all,
                           receptor_type=iaf_tum_2000.RECEPTOR_TYPES['TSODYKS'])
        self.assertIsNotNone(proj)


class TestRecurrentNmdaLowers(unittest.TestCase):
    """A recurrent-NMDA pool builds and lowers in one ``Simulator.simulate``."""

    def test_pool_runs_in_for_loop_and_accumulates_s_nmda(self):
        """pool->pool NMDA (graded, dense) lowers under scan; s_NMDA accumulates."""
        sim = Simulator(dt=0.1 * u.ms)
        pool = _bw_pop(sim, 3)
        # A shared generator fires the whole pool via strong AMPA so every neuron
        # spikes; each spike's graded NMDA offset then feeds the others recurrently.
        gen = sim.create(spike_generator, 1, spike_times=[2.0, 3.0, 4.0] * u.ms)
        sim.connect(gen, pool, weight=400.0 * u.nS, delay=0.1 * u.ms,
                    rule=all_to_all, receptor_type=iaf_bw_2001.AMPA)
        sim.connect(pool, pool, weight=2.0 * u.nS, delay=0.5 * u.ms,
                    rule=all_to_all, receptor_type=iaf_bw_2001.NMDA,
                    comm='dense', allow_autapses=False)
        sim.simulate(30.0 * u.ms)  # must not raise a carry-shape error
        node = pool.segments[0].population
        s_nmda_max = float((node.s_NMDA.value / u.nS).max())
        self.assertGreater(s_nmda_max, 0.0)

    def test_recurrent_ampa_from_bw_pre_stays_binary_on_ampa(self):
        """A bw->bw edge over AMPA stays binary and never feeds the graded NMDA gate.

        ``iaf_bw_2001`` is an *emitting* pre (``_emission_receptor == NMDA``), but a
        recurrent connection over the *AMPA* receptor takes the ``receptor_type !=
        emit_receptor`` branch: it must deliver the binary spike into ``s_AMPA`` (the
        Wang network wires bw->bw over both AMPA and NMDA), not the graded NMDA
        emission. With no NMDA edge present, ``s_NMDA`` must stay exactly zero.
        """
        sim = Simulator(dt=0.1 * u.ms)
        pool = _bw_pop(sim, 2)
        # Fire neuron 0 only; its spike rides the recurrent AMPA edge onto neuron 1.
        gen = sim.create(spike_generator, 1, spike_times=[2.0, 3.0, 4.0] * u.ms)
        sim.connect(gen, pool[0], weight=400.0 * u.nS, delay=0.1 * u.ms,
                    rule=all_to_all, receptor_type=iaf_bw_2001.AMPA)
        sim.connect(pool, pool, weight=3.0 * u.nS, delay=0.5 * u.ms,
                    rule=all_to_all, receptor_type=iaf_bw_2001.AMPA,
                    allow_autapses=False)
        sim.simulate(30.0 * u.ms)
        node = pool.segments[0].population
        s_ampa = node.s_AMPA.value / u.nS
        s_nmda = node.s_NMDA.value / u.nS
        # The recurrent AMPA spike reached neuron 1's AMPA channel...
        self.assertGreater(float(s_ampa[1]), 0.0)
        # ...and no NMDA edge exists, so the emitting pre did NOT leak a graded NMDA
        # deposit (would be non-zero if the AMPA edge wrongly used the NMDA emission).
        self.assertEqual(float((s_nmda).max()), 0.0)


if __name__ == '__main__':
    unittest.main()
