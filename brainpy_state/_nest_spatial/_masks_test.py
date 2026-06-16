# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for spatial masks (circular / spherical / box)."""
import unittest

import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy_state._nest_spatial._masks import circular, spherical, box


class TestMasks(unittest.TestCase):
    def test_circular_cutoff_inclusive(self):
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.0, 0.0], [0.4, 0.0], [0.5, 0.0], [0.6, 0.0]]) * u.um
        m = np.asarray(circular(0.5).contains(pre, post))
        self.assertEqual(m.shape, (1, 4))
        np.testing.assert_array_equal(m[0], [True, True, True, False])

    def test_circular_radius_quantity(self):
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.49, 0.0]]) * u.um
        self.assertTrue(bool(circular(0.5 * u.um).contains(pre, post)[0, 0]))

    def test_box_anchored_on_source_3d(self):
        pre = jnp.array([[1.0, 1.0, 1.0]]) * u.um
        # displacement target-source: 0.5 (in) and 1.0 (out) along x.
        post = jnp.array([[1.5, 1.0, 1.0], [2.0, 1.0, 1.0]]) * u.um
        m = np.asarray(box([-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]).contains(pre, post))
        np.testing.assert_array_equal(m[0], [True, False])

    def test_box_2d(self):
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.2, 0.2], [0.2, -0.9]]) * u.um
        m = np.asarray(box([-0.5, -0.5], [0.5, 0.5]).contains(pre, post))
        np.testing.assert_array_equal(m[0], [True, False])

    def test_spherical_is_distance_cutoff_3d(self):
        pre = jnp.array([[0.0, 0.0, 0.0]]) * u.um
        post = jnp.array([[0.7, 0.0, 0.0], [0.8, 0.0, 0.0]]) * u.um
        m = np.asarray(spherical(0.75).contains(pre, post))
        np.testing.assert_array_equal(m[0], [True, False])


class TestDoughnut(unittest.TestCase):
    def test_annulus_boundary_inner_exclusive_outer_inclusive(self):
        from brainpy_state._nest_spatial._masks import doughnut
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.2, 0.0], [0.3, 0.0], [0.5, 0.0], [0.7, 0.0], [0.8, 0.0]]) * u.um
        m = np.asarray(doughnut(0.3, 0.7).contains(pre, post))
        # 0.3 < d <= 0.7 : d=0.3 excluded, d=0.7 included
        np.testing.assert_array_equal(m[0], [False, False, True, True, False])

    def test_inner_zero_excludes_only_center(self):
        from brainpy_state._nest_spatial._masks import doughnut, circular
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.0, 0.0], [0.5, 0.0]]) * u.um
        d = np.asarray(doughnut(0.0, 1.0).contains(pre, post))
        c = np.asarray(circular(1.0).contains(pre, post))
        np.testing.assert_array_equal(d[0], [False, True])     # center (d=0) excluded
        np.testing.assert_array_equal(c[0], [True, True])      # circular includes center

    def test_inner_equals_outer_is_empty(self):
        from brainpy_state._nest_spatial._masks import doughnut
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.0]]) * u.um
        m = np.asarray(doughnut(0.5, 0.5).contains(pre, post))
        self.assertFalse(bool(m.any()))


class TestRectangular(unittest.TestCase):
    def test_unrotated_box_on_displacement(self):
        from brainpy_state._nest_spatial._masks import rectangular
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.2, 0.2], [0.6, 0.0]]) * u.um
        m = np.asarray(rectangular([-0.5, -0.5], [0.5, 0.5]).contains(pre, post))
        np.testing.assert_array_equal(m[0], [True, False])

    def test_azimuth_90_rotates_acceptance(self):
        from brainpy_state._nest_spatial._masks import rectangular
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.8, 0.0]]) * u.um               # disp (0.8, 0)
        tall = rectangular([-0.2, -1.0], [0.2, 1.0])        # x:+-0.2 -> excludes (0.8,0)
        rot = rectangular([-0.2, -1.0], [0.2, 1.0], azimuth_angle=90.0)  # becomes wide -> includes
        self.assertFalse(bool(np.asarray(tall.contains(pre, post))[0, 0]))
        self.assertTrue(bool(np.asarray(rot.contains(pre, post))[0, 0]))


class TestElliptical(unittest.TestCase):
    def test_axis_aligned(self):
        from brainpy_state._nest_spatial._masks import elliptical
        pre = jnp.array([[0.0, 0.0]]) * u.um
        # semi-major 1 (x), semi-minor 0.5 (y): in iff dx^2 + 4 dy^2 <= 1
        post = jnp.array([[0.9, 0.0], [1.1, 0.0], [0.0, 0.4], [0.0, 0.6]]) * u.um
        m = np.asarray(elliptical(2.0, 1.0).contains(pre, post))
        np.testing.assert_array_equal(m[0], [True, False, True, False])

    def test_major_equals_minor_is_circular(self):
        from brainpy_state._nest_spatial._masks import elliptical, circular
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.9, 0.0], [0.0, 0.9], [0.8, 0.8]]) * u.um
        e = np.asarray(elliptical(2.0, 2.0).contains(pre, post))   # semi 1 == circular(1)
        c = np.asarray(circular(1.0).contains(pre, post))
        np.testing.assert_array_equal(e[0], c[0])

    def test_rotated_90_swaps_axes(self):
        from brainpy_state._nest_spatial._masks import elliptical
        pre = jnp.array([[0.0, 0.0]]) * u.um
        post = jnp.array([[0.9, 0.0], [0.0, 0.9]]) * u.um
        aligned = np.asarray(elliptical(2.0, 1.0).contains(pre, post))
        rot = np.asarray(elliptical(2.0, 1.0, azimuth_angle=90.0).contains(pre, post))
        np.testing.assert_array_equal(aligned[0], [True, False])
        np.testing.assert_array_equal(rot[0], [False, True])


class TestEllipsoidal(unittest.TestCase):
    def test_axis_aligned_3d(self):
        from brainpy_state._nest_spatial._masks import ellipsoidal
        pre = jnp.array([[0.0, 0.0, 0.0]]) * u.um
        # semi 1,1,0.5: in iff dx^2 + dy^2 + 4 dz^2 <= 1
        post = jnp.array([[0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.4], [0, 0, 0.6]]) * u.um
        m = np.asarray(ellipsoidal(2.0, 2.0, 1.0).contains(pre, post))
        np.testing.assert_array_equal(m[0], [True, True, True, False])

    def test_equal_axes_is_spherical(self):
        from brainpy_state._nest_spatial._masks import ellipsoidal, spherical
        pre = jnp.array([[0.0, 0.0, 0.0]]) * u.um
        post = jnp.array([[0.9, 0, 0], [0, 0, 0.9], [0.7, 0.7, 0.0], [0.8, 0.8, 0.0]]) * u.um
        e = np.asarray(ellipsoidal(2.0, 2.0, 2.0).contains(pre, post))   # sphere radius 1
        s = np.asarray(spherical(1.0).contains(pre, post))
        np.testing.assert_array_equal(e[0], s[0])


if __name__ == '__main__':
    unittest.main()
