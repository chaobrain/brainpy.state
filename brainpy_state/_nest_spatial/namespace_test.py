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

    def test_cluster27_surface_reachable_through_namespace(self):
        # Group A-F additions (positions, distributions, masks, queries, dump, plot).
        for name in ('pos', 'source_pos', 'target_pos',
                     'exponential', 'gamma', 'gabor', 'gaussian2D',
                     'rectangular', 'doughnut', 'elliptical', 'ellipsoidal',
                     'nearest_element', 'select_nodes_by_mask',
                     'dump_layer_nodes', 'dump_layer_connections',
                     'plot_layer', 'plot_targets', 'plot_sources',
                     'plot_probability_parameter'):
            self.assertTrue(hasattr(B.spatial, name), name)
            self.assertIn(name, B.spatial.__all__, name)


if __name__ == '__main__':
    unittest.main()
