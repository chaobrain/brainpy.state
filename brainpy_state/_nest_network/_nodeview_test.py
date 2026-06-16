# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import brainunit as u

from brainpy_state import iaf_psc_alpha
from brainpy_state._nest_network import NodeView


class TestNodeView(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_size_concat_slice(self):
        a = iaf_psc_alpha(4)
        b = iaf_psc_alpha(2)
        va, vb = NodeView.of(a), NodeView.of(b)
        self.assertEqual(va.size, 4)
        self.assertEqual(vb.size, 2)
        self.assertEqual((va + vb).size, 6)
        self.assertEqual(len((va + vb).segments), 2)
        self.assertEqual(va[:2].size, 2)
        self.assertIs(va[:2].segments[0].population, a)

    def test_concat_then_slice_rejected_multisegment(self):
        a = iaf_psc_alpha(4)
        b = iaf_psc_alpha(2)
        with self.assertRaises(NotImplementedError):
            _ = (NodeView.of(a) + NodeView.of(b))[:3]
