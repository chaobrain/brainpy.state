# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free seam tests: Simulator.create(positions=) + the spatial coord-bind in connect."""
import unittest

import jax.numpy as jnp
import numpy as np
import brainunit as u
import brainstate

from brainpy_state import Simulator, iaf_psc_alpha, poisson_generator
from brainpy_state._dist import Uniform
from brainpy_state._nest_spatial import (
    grid, free, circular, gaussian, distance, pairwise_distance,
    spatial_pairwise_bernoulli,
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


class TestSpatialConnectSeam(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(platform='cpu')

    def _grid_pop(self, sim, shape=(5, 5), extent=(4.0, 4.0)):
        return sim.create(iaf_psc_alpha, positions=grid(list(shape), extent=list(extent)))

    def test_mask_cutoff_hard_circular(self):
        # p=1.0 within circular(1.0): every in-mask ordered pair connects deterministically,
        # every out-of-mask pair is dropped. Grid spacing 0.8 => orthogonal neighbours (0.8)
        # are kept, diagonals (0.8*sqrt2 ~ 1.13) are cut.
        sim = Simulator(dt=0.1 * u.ms)
        pop = self._grid_pop(sim)
        coords = sim._positions[id(pop.segments[0].population)]
        sim.connect(pop, pop, rule=spatial_pairwise_bernoulli(p=1.0, mask=circular(1.0)),
                    weight=1.0 * u.pA, delay=1.0 * u.ms)
        sc = sim.get_connections(source=pop, target=pop)
        D = u.get_magnitude(pairwise_distance(coords, coords).to(u.um))
        expected = int((D <= 1.0 + 1e-9).sum())
        self.assertEqual(len(sc), expected)
        realized = D[sc.source, sc.target]
        self.assertLessEqual(float(realized.max()), 1.0 + 1e-6)
        self.assertLess(expected, coords.shape[0] ** 2)        # the cutoff excluded diagonals

    def test_no_positions_raises(self):
        sim = Simulator(dt=0.1 * u.ms)
        a = sim.create(iaf_psc_alpha, 4)
        b = sim.create(iaf_psc_alpha, 4)
        with self.assertRaises(ValueError):
            sim.connect(a, b, rule=spatial_pairwise_bernoulli(0.5),
                        weight=1.0 * u.pA, delay=1.0 * u.ms)

    def test_generator_pre_raises(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = self._grid_pop(sim, shape=(3, 3), extent=(2.0, 2.0))
        gen = sim.create(poisson_generator, params={'rate': 100.0 * u.Hz})
        with self.assertRaises(ValueError):
            sim.connect(gen, pop, rule=spatial_pairwise_bernoulli(0.5),
                        weight=1.0 * u.pA, delay=1.0 * u.ms)

    def test_seeded_reproducible_and_seed_sensitive(self):
        def run(seed):
            sim = Simulator(dt=0.1 * u.ms)
            pop = self._grid_pop(sim, shape=(6, 6), extent=(3.0, 3.0))
            sim.connect(pop, pop, rule=spatial_pairwise_bernoulli(p=gaussian(distance, std=1.0)),
                        weight=1.0 * u.pA, delay=1.0 * u.ms, seed=seed)
            sc = sim.get_connections(source=pop, target=pop)
            return np.asarray(sc.source), np.asarray(sc.target)

        s0, t0 = run(7)
        s0b, t0b = run(7)
        np.testing.assert_array_equal(s0, s0b)                 # same seed -> identical adjacency
        np.testing.assert_array_equal(t0, t0b)
        s1, t1 = run(123)
        same = s0.shape == s1.shape and np.array_equal(s0, s1) and np.array_equal(t0, t1)
        self.assertFalse(same)                                 # different seed -> different draw


if __name__ == '__main__':
    unittest.main()
