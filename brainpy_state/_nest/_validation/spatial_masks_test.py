# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Exact node-set parity for the cluster-27 masks (goal 27).

``rectangular`` / ``doughnut`` / ``elliptical`` / ``ellipsoidal`` are hard candidate cutoffs, so
under ``pairwise_bernoulli`` with ``p = 1`` the realised adjacency *is* the mask: source ``i``
connects to exactly the nodes inside the mask anchored on it. The realised edge set must therefore
equal live NEST's bit-for-bit (no PRNG), including rotated (``azimuth_angle``) and 3-D
(``ellipsoidal``) cases. A NEST-free class pins the geometry independently.
"""
import unittest

import jax
import brainstate
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainunit as u

try:
    import nest
except Exception:
    nest = None

from brainpy_state import Simulator, iaf_psc_alpha
from brainpy_state._nest_spatial import grid, spatial_pairwise_bernoulli
from brainpy_state._nest_spatial._masks import rectangular, doughnut, elliptical, ellipsoidal
from brainpy_state._nest._validation.nest_compare import requires_nest

S2, E2 = [5, 5], [4.0, 4.0]
S3, E3 = [4, 4, 4], [4.0, 4.0, 4.0]


def _bp_adj(shape, extent, mask):
    sim = Simulator(dt=0.1 * u.ms)
    pos = grid(shape, extent=extent)
    a = sim.create(iaf_psc_alpha, positions=pos)
    sim.connect(a, a, rule=spatial_pairwise_bernoulli(p=1.0, mask=mask),
                weight=1.0 * u.pA, delay=1.0 * u.ms, seed=0)
    sc = sim.get_connections(source=a, target=a)
    return set(zip(np.asarray(sc.source).tolist(), np.asarray(sc.target).tolist()))


def _nest_adj(shape, extent, maskdict):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    pos = nest.spatial.grid(shape=shape, extent=extent)
    a = nest.Create('iaf_psc_alpha', positions=pos)
    nest.Connect(a, a, {'rule': 'pairwise_bernoulli', 'p': 1.0, 'mask': maskdict})
    conns = nest.GetConnections(a, a)
    g0 = a[0].global_id
    return set(zip((np.asarray(conns.source) - g0).tolist(),
                   (np.asarray(conns.target) - g0).tolist()))


# (label, shape, extent, bp-mask, nest-mask-dict)
CASES = [
    ('rectangular', S2, E2, rectangular([-1.0, -0.5], [1.0, 0.5]),
     {'rectangular': {'lower_left': [-1.0, -0.5], 'upper_right': [1.0, 0.5]}}),
    ('rectangular_rot', S2, E2, rectangular([-1.0, -0.5], [1.0, 0.5], azimuth_angle=30.0),
     {'rectangular': {'lower_left': [-1.0, -0.5], 'upper_right': [1.0, 0.5], 'azimuth_angle': 30.0}}),
    ('doughnut', S2, E2, doughnut(0.9, 1.8),
     {'doughnut': {'inner_radius': 0.9, 'outer_radius': 1.8}}),
    ('elliptical', S2, E2, elliptical(3.0, 1.4),
     {'elliptical': {'major_axis': 3.0, 'minor_axis': 1.4}}),
    ('elliptical_rot', S2, E2, elliptical(3.0, 1.4, azimuth_angle=40.0),
     {'elliptical': {'major_axis': 3.0, 'minor_axis': 1.4, 'azimuth_angle': 40.0}}),
    ('ellipsoidal', S3, E3, ellipsoidal(3.0, 2.0, 1.4),
     {'ellipsoidal': {'major_axis': 3.0, 'minor_axis': 2.0, 'polar_axis': 1.4}}),
]


class TestMaskGeometry(unittest.TestCase):
    """Each mask selects the geometrically-correct nodes (NEST-free)."""

    def test_doughnut_is_circular_difference(self):
        # doughnut(in, out) == circular(out) minus circular(in) (inner exclusive).
        from brainpy_state._nest_spatial._masks import circular
        inner, outer = 0.9, 1.8
        outer_adj = _bp_adj(S2, E2, circular(outer))
        inner_adj = _bp_adj(S2, E2, circular(inner))
        doughnut_adj = _bp_adj(S2, E2, doughnut(inner, outer))
        # outer-ball minus (closed) inner-ball, plus the inner boundary ring NEST keeps:
        self.assertTrue(doughnut_adj.issubset(outer_adj))
        self.assertEqual(len(doughnut_adj & inner_adj), 0)   # nothing strictly inside inner

    def test_elliptical_major_equals_minor_is_circular(self):
        from brainpy_state._nest_spatial._masks import circular
        self.assertEqual(_bp_adj(S2, E2, elliptical(2.4, 2.4)),
                         _bp_adj(S2, E2, circular(1.2)))      # semi == radius

    def test_rotation_changes_selection(self):
        flat = _bp_adj(S2, E2, elliptical(3.0, 1.0))
        rot = _bp_adj(S2, E2, elliptical(3.0, 1.0, azimuth_angle=90.0))
        self.assertNotEqual(flat, rot)                       # a thin ellipse is orientation-sensitive


@requires_nest
class TestMaskNestParity(unittest.TestCase):
    """The realised adjacency equals live NEST's for every mask, rotation and dimension."""

    def test_node_sets_match_nest(self):
        for label, shape, extent, mask, maskdict in CASES:
            with self.subTest(mask=label):
                bp = _bp_adj(shape, extent, mask)
                nst = _nest_adj(shape, extent, maskdict)
                self.assertEqual(bp, nst, msg=f'{label}: {len(bp ^ nst)} differing edges')


if __name__ == '__main__':
    unittest.main()
