# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for spatial helpers (center_element, Distance)."""
import unittest

import numpy as np
import brainunit as u

from brainpy_state._nest_spatial._layers import grid
from brainpy_state._nest_spatial._helpers import center_element, Distance


class TestHelpers(unittest.TestCase):
    def test_center_element_matches_nest_grid_4x3(self):
        # Pinned against live NEST: FindCenterElement(grid([4,3],[2,1.5])) -> local idx 4.
        self.assertEqual(center_element(grid([4, 3], extent=[2.0, 1.5])), 4)

    def test_center_element_ties_lowest_index(self):
        # grid([2,2]) centroid is the origin; all four corners equidistant -> lowest idx 0.
        self.assertEqual(center_element(grid([2, 2], extent=[1.0, 1.0])), 0)

    def test_center_element_3d(self):
        idx = center_element(grid([3, 3, 3], extent=[1.0, 1.0, 1.0]))
        # the true center cell of a 3x3x3 grid (col=row=dep=1) -> idx 1*9+1*3+1 = 13
        self.assertEqual(idx, 13)

    def test_distance_query_shape_and_diagonal(self):
        a = grid([2, 2], extent=[1.0, 1.0])
        b = grid([2, 2], extent=[1.0, 1.0])
        D = u.get_magnitude(Distance(a, b).to(u.um))
        self.assertEqual(D.shape, (4, 4))
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)

    def test_distance_query_values(self):
        a = grid([2, 2], extent=[2.0, 2.0])   # corners at +/-0.5
        D = u.get_magnitude(Distance(a, a).to(u.um))
        # node0 (-0.5, 0.5) to node3 (0.5,-0.5): sqrt(1+1) = sqrt(2)
        self.assertAlmostEqual(D[0, 3], np.sqrt(2.0), places=6)


if __name__ == '__main__':
    unittest.main()
