# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""The spatial submodule is re-exported as the ``brainpy.state.spatial`` namespace."""
import unittest

import brainpy_state as B
from brainpy_state import _nest_spatial


class TestSpatialNamespace(unittest.TestCase):
    def test_spatial_is_reexported_submodule(self):
        self.assertTrue(hasattr(B, 'spatial'))
        self.assertIs(B.spatial, _nest_spatial)

    def test_spatial_in_package_all(self):
        self.assertIn('spatial', B.__all__)

    def test_public_surface_reachable_through_namespace(self):
        for name in ('grid', 'free', 'distance', 'gaussian', 'circular', 'spherical',
                     'box', 'spatial_pairwise_bernoulli', 'center_element', 'Distance',
                     'target_nodes', 'target_positions', 'pairwise_distance',
                     'displacement', 'Layer', 'SpatialConnRule'):
            self.assertTrue(hasattr(B.spatial, name), name)


if __name__ == '__main__':
    unittest.main()
