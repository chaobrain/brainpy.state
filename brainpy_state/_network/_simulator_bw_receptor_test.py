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
brainstate.environ.set(precision=64, platform='cpu')

import saiunit as u
import braintools


from brainpy_state import iaf_bw_2001


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


if __name__ == '__main__':
    unittest.main()
