# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free seam tests: Simulator.create(positions=) + the spatial coord-bind in connect."""
import unittest

import jax.numpy as jnp
import numpy as np
import brainunit as u
import brainstate

from brainpy_state import Simulator, iaf_psc_alpha
from brainpy_state._dist import Uniform
from brainpy_state._nest_spatial import (
    grid, free, circular, gaussian, distance, spatial_pairwise_bernoulli,
)


class TestCreatePositions(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(platform='cpu')

    def test_grid_size_derived_and_coords_stored(self):
        sim = Simulator(dt=0.1 * u.ms)
        view = sim.create(iaf_psc_alpha, positions=grid([4, 3], extent=[2.0, 1.5]))
        self.assertEqual(view.size, 12)
        pop = view.segments[0].population
        xy = u.get_magnitude(sim._positions[id(pop)].to(u.um))
        self.assertEqual(xy.shape, (12, 2))
        np.testing.assert_allclose(xy[0], [-0.75, 0.5], atol=1e-9)
        np.testing.assert_allclose(xy[11], [0.75, -0.5], atol=1e-9)

    def test_free_explicit_array_size(self):
        sim = Simulator(dt=0.1 * u.ms)
        pos = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]]) * u.um
        view = sim.create(iaf_psc_alpha, positions=free(pos))
        self.assertEqual(view.size, 3)
        pop = view.segments[0].population
        self.assertEqual(sim._positions[id(pop)].shape, (3, 2))

    def test_free_deferred_uses_explicit_size(self):
        sim = Simulator(dt=0.1 * u.ms)
        view = sim.create(iaf_psc_alpha, 1000, positions=free(Uniform(-0.5, 0.5), extent=[1.5, 1.5, 1.5]))
        self.assertEqual(view.size, 1000)
        pop = view.segments[0].population
        self.assertEqual(sim._positions[id(pop)].shape, (1000, 3))
        m = u.get_magnitude(sim._positions[id(pop)].to(u.um))
        self.assertTrue(m.min() >= -0.5 - 1e-6 and m.max() <= 0.5 + 1e-6)

    def test_no_positions_leaves_registry_empty(self):
        sim = Simulator(dt=0.1 * u.ms)
        view = sim.create(iaf_psc_alpha, 5)
        pop = view.segments[0].population
        self.assertNotIn(id(pop), sim._positions)


if __name__ == '__main__':
    unittest.main()
