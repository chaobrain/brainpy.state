# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for spatial helpers (center_element, Distance, target queries)."""
import unittest

import numpy as np
import brainunit as u
import brainstate

from brainpy_state import Simulator, iaf_psc_alpha
from brainpy_state._nest_spatial._layers import grid
from brainpy_state._nest_spatial._distance import pairwise_distance
from brainpy_state._nest_spatial._masks import circular
from brainpy_state._nest_spatial._rule import spatial_pairwise_bernoulli
from brainpy_state._nest_spatial._helpers import (
    center_element, Distance, target_nodes, target_positions,
)


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


class TestTargetQueries(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(platform='cpu')

    def _sim_with_local_connectivity(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_alpha, positions=grid([5, 5], extent=[4.0, 4.0]))
        # p=1.0 within circular(1.0): node 12 (center, (0,0)) -> itself + 4 orthogonal
        # neighbours at d=0.8; diagonals at 0.8*sqrt2 are cut.
        sim.connect(pop, pop, rule=spatial_pairwise_bernoulli(p=1.0, mask=circular(1.0)),
                    weight=1.0 * u.pA, delay=1.0 * u.ms)
        return sim, pop

    def test_target_nodes_per_source_and_center_footprint(self):
        sim, pop = self._sim_with_local_connectivity()
        tn = target_nodes(sim, pop, pop)
        self.assertEqual(len(tn), 25)                       # one entry per source node
        self.assertEqual(sorted(int(i) for i in tn[12]), [7, 11, 12, 13, 17])

    def test_target_positions_within_mask_and_shape(self):
        sim, pop = self._sim_with_local_connectivity()
        coords = sim._positions[id(pop.segments[0].population)]
        tn = target_nodes(sim, pop, pop)
        tp = target_positions(sim, pop, pop)
        self.assertEqual(len(tp), 25)
        self.assertEqual(tp[12].shape, (len(tn[12]), 2))
        # every target of the centre node lies within the mask radius of it
        d = u.get_magnitude(pairwise_distance(coords[12][None], tp[12]).to(u.um))
        self.assertTrue(bool((d <= 1.0 + 1e-6).all()))
        # the returned coords are exactly the target nodes' coords
        np.testing.assert_allclose(
            u.get_magnitude(tp[12].to(u.um)), u.get_magnitude(coords[tn[12]].to(u.um)))


if __name__ == '__main__':
    unittest.main()
