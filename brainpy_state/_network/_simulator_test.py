# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest


class TestNetworkPublicAPI(unittest.TestCase):
    def test_public_names_importable(self):
        from brainpy_state._network import (
            Simulator, SimulationResult, NodeView,
            all_to_all, one_to_one, fixed_indegree,
        )
        self.assertTrue(callable(fixed_indegree))
        self.assertIsNotNone(Simulator)
        self.assertIsNotNone(SimulationResult)
        self.assertIsNotNone(NodeView)
        self.assertIsNotNone(all_to_all)
        self.assertIsNotNone(one_to_one)
