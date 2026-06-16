# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import brainunit as u

from brainpy_state import LIF, Expon, COBA
from brainpy_state._nest_network._builder import Builder
from brainpy_state._nest_network._base import Network
from brainpy_state._nest_network._projections import OneToOneProj


class TestBuilder(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_builder_is_a_network(self):
        b = Builder()
        self.assertIsInstance(b, Network)

    def test_add_sets_attribute_and_returns_module(self):
        b = Builder()
        pop = LIF(5)
        ret = b.add('exc', pop)
        self.assertIs(ret, pop)
        self.assertIs(b.exc, pop)

    def test_connect_by_reference(self):
        b = Builder()
        pre = b.add('pre', LIF(5))
        post = b.add('post', LIF(5))
        proj = b.connect(
            pre, post, rule=OneToOneProj,
            weight=0.1*u.nS,
            syn=Expon.desc(5, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        self.assertIs(proj.pre, pre)
        self.assertIs(proj.post, post)

    def test_connect_by_string_name(self):
        b = Builder()
        b.add('pre', LIF(5))
        b.add('post', LIF(5))
        proj = b.connect(
            'pre', 'post', rule=OneToOneProj,
            weight=0.1*u.nS,
            syn=Expon.desc(5, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        self.assertIs(proj.pre, b.pre)
        self.assertIs(proj.post, b.post)

    def test_duplicate_add_raises(self):
        b = Builder()
        b.add('exc', LIF(5))
        with self.assertRaises(ValueError):
            b.add('exc', LIF(5))

    def test_simulate_works(self):
        b = Builder()
        b.add('pop', LIF(5))
        brainstate.nn.init_all_states(b)
        out = b.simulate(0.5 * u.ms)
        self.assertEqual(out, {})
