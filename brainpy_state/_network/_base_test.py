# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest
import brainstate
import saiunit as u

from brainpy_state._network._base import Network
from brainpy_state import LIF


class TestNetworkSkeleton(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_network_is_a_brainstate_module(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(10)

        net = Net()
        self.assertIsInstance(net, brainstate.nn.Module)
        self.assertIs(net.pop, net.nodes()[('pop',)])

    def test_module_attribute_is_brainpy_state(self):
        self.assertEqual(Network.__module__, 'brainpy.state')
