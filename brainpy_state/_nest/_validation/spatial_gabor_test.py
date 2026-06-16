# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Parity for the anisotropic gabor / gaussian2D distance distributions (goal 27).

Both kernels consume the per-axis distances ``distance.x`` / ``distance.y`` (absolute, like NEST)
and combine them with an orientation/correlation. They are read back deterministically as
ordered-pair weight matrices and compared to live NEST's ``spatial_distributions.gabor`` /
``gaussian2D`` to machine precision. A NEST-free class pins the closed forms and smoke-runs the
``spatial_gabor`` example; a final check confirms the example's realised footprint stays inside
its elliptical mask.
"""
import math
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

from brainpy_state._nest_spatial import grid, distance, gabor, gaussian2D
from brainpy_state._nest._validation.nest_compare import requires_nest

SHAPE = (4, 4)
EXTENT = (3.0, 3.0)


def _bp_matrix(kernel, layer):
    return np.asarray(u.get_magnitude(kernel._eval_pair(layer.coords, layer.coords)))


def _nest_weight_matrix(param, n, shape=SHAPE, extent=EXTENT):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    pos = nest.spatial.grid(shape=list(shape), extent=list(extent))
    a = nest.Create('iaf_psc_alpha', positions=pos)
    nest.Connect(a, a, {'rule': 'pairwise_bernoulli', 'p': 1.0, 'allow_autapses': False},
                 {'synapse_model': 'static_synapse', 'weight': param})
    conns = nest.GetConnections(a, a)
    g0 = a[0].global_id
    weights = np.full((n, n), np.nan)
    for s, t, w in zip(conns.source, conns.target, conns.get('weight')):
        weights[s - g0, t - g0] = w
    return weights


class TestAnisotropicStructure(unittest.TestCase):
    """The gabor / gaussian2D kernels equal their closed forms; the example runs."""

    def test_gabor_matches_formula(self):
        pre = (np.array([[0.0, 0.0]]) * u.um)
        post = (np.array([[1.0, 2.0]]) * u.um)
        k = gabor(distance.x, distance.y, theta=30.0, gamma=1.5, std=0.8, lam=1.2, psi=20.0)
        got = float(u.get_magnitude(k._eval_pair(pre, post)[0, 0]))
        X, Y, th = 1.0, 2.0, math.radians(30.0)
        xp = X * math.cos(th) + Y * math.sin(th)
        yp = -X * math.sin(th) + Y * math.cos(th)
        ref = (max(math.cos(2 * math.pi * yp / 1.2 + math.radians(20.0)), 0.0)
               * math.exp(-(1.5 ** 2 * xp ** 2 + yp ** 2) / (2 * 0.8 ** 2)))
        self.assertAlmostEqual(got, ref, places=10)

    def test_gaussian2D_matches_formula(self):
        pre = (np.array([[0.0, 0.0]]) * u.um)
        post = (np.array([[1.0, 1.5]]) * u.um)
        k = gaussian2D(distance.x, distance.y, mean_x=0.1, mean_y=0.2,
                       std_x=0.6, std_y=0.9, rho=0.3)
        got = float(u.get_magnitude(k._eval_pair(pre, post)[0, 0]))
        dx, dy, rho = 1.0 - 0.1, 1.5 - 0.2, 0.3
        denom = 2 * (1 - rho ** 2)
        ref = math.exp(-(dx ** 2 / 0.6 ** 2 + dy ** 2 / 0.9 ** 2 - 2 * rho * dx * dy / (0.6 * 0.9))
                       / denom)
        self.assertAlmostEqual(got, ref, places=10)

    def test_example_run_smoke(self):
        from examples.nest.spatial_gabor import run
        coords, ctr, tgt = run(shape=(12, 12))
        self.assertEqual(coords.shape, (144, 2))
        self.assertGreaterEqual(tgt.shape[0], 1)

    def test_example_targets_inside_ellipse(self):
        # every realised target of the centre lies in the tilted ellipse (mask is a hard cutoff).
        from examples.nest.spatial_gabor import run
        major, minor, theta = 3.0, 1.0, 45.0
        coords, ctr, tgt = run(shape=(20, 20), major=major, minor=minor, theta=theta)
        c = np.asarray(u.get_magnitude(coords.to(u.um)))[ctr]
        t = np.asarray(u.get_magnitude(tgt.to(u.um)))
        dx, dy = t[:, 0] - c[0], t[:, 1] - c[1]
        az = math.radians(theta)
        nx = dx * math.cos(az) + dy * math.sin(az)
        ny = dx * math.sin(az) - dy * math.cos(az)
        inside = nx ** 2 * (4 / major ** 2) + ny ** 2 * (4 / minor ** 2)
        self.assertLessEqual(float(inside.max()), 1.0 + 1e-9)


@requires_nest
class TestAnisotropicNestParity(unittest.TestCase):
    """gabor / gaussian2D match live NEST element-by-element across parameter sets."""

    def test_gabor_weight_readback_matches_nest(self):
        layer = grid(list(SHAPE), extent=list(EXTENT))
        for kw in [dict(theta=0.0, gamma=1.0, std=1.0, lam=1.0, psi=0.0),
                   dict(theta=30.0, gamma=1.5, std=0.8, lam=1.2, psi=20.0),
                   dict(theta=90.0, gamma=0.7, std=1.1, lam=2.0, psi=45.0)]:
            bp = _bp_matrix(gabor(distance.x, distance.y, **kw), layer)
            nst = _nest_weight_matrix(
                nest.spatial_distributions.gabor(nest.spatial.distance.x,
                                                 nest.spatial.distance.y, **kw), layer.n)
            off = ~np.eye(layer.n, dtype=bool)
            self.assertLess(float(np.nanmax(np.abs(bp - nst)[off])), 1e-9,
                            msg=f'gabor({kw}) diverges from NEST')

    def test_gaussian2D_weight_readback_matches_nest(self):
        layer = grid(list(SHAPE), extent=list(EXTENT))
        for kw in [dict(std_x=1.0, std_y=2.0, rho=0.0),
                   dict(mean_x=0.1, mean_y=0.2, std_x=0.6, std_y=0.9, rho=0.3),
                   dict(std_x=0.8, std_y=0.8, rho=-0.4)]:
            bp = _bp_matrix(gaussian2D(distance.x, distance.y, **kw), layer)
            nst = _nest_weight_matrix(
                nest.spatial_distributions.gaussian2D(nest.spatial.distance.x,
                                                      nest.spatial.distance.y, **kw), layer.n)
            off = ~np.eye(layer.n, dtype=bool)
            self.assertLess(float(np.nanmax(np.abs(bp - nst)[off])), 1e-9,
                            msg=f'gaussian2D({kw}) diverges from NEST')


if __name__ == '__main__':
    unittest.main()
