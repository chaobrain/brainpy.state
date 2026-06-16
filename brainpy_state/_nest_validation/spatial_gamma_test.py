# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Parity for the gamma distance distribution (goal 27).

NEST's ``spatial_distributions.gamma(distance, kappa, theta)`` is the (unnormalised-at-peak)
gamma density :math:`p(d) = d^{\kappa-1} e^{-d/\theta} / (\theta^{\kappa}\,\Gamma(\kappa))`.
As with the other distributions the kernel is read back deterministically as an ordered-pair
weight matrix and compared to live NEST to machine precision; a NEST-free class pins the closed
form (including the ``kappa = 1`` reduction to a scaled exponential).
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

from brainpy_state import Simulator, iaf_psc_alpha
from brainpy_state._nest_spatial import grid, distance, gamma, spatial_pairwise_bernoulli
from brainpy_state._nest_validation.nest_compare import requires_nest

SHAPE = (3, 3)
EXTENT = (2.0, 2.0)
KAPPA = 2.0
THETA = 0.5


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


class TestGammaStructure(unittest.TestCase):
    """The kernel equals the gamma density; ``kappa=1`` reduces to ``e^{-d/theta}/theta``."""

    def test_kernel_matches_gamma_density(self):
        layer = grid([1, 4], extent=[1.0, 4.0])             # distances 0,1,2,3 along y
        got = _bp_matrix(gamma(distance, kappa=KAPPA, theta=THETA), layer)[0]
        d = np.abs(np.asarray(u.get_magnitude((layer.coords[:, 1] - layer.coords[0, 1]).to(u.um))))
        ref = d ** (KAPPA - 1) * np.exp(-d / THETA) / (THETA ** KAPPA * math.gamma(KAPPA))
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    def test_kappa_one_is_scaled_exponential(self):
        layer = grid([1, 3], extent=[1.0, 3.0])             # distances 0,1,2
        got = _bp_matrix(gamma(distance, kappa=1.0, theta=THETA), layer)[0]
        d = np.abs(np.asarray(u.get_magnitude((layer.coords[:, 1] - layer.coords[0, 1]).to(u.um))))
        np.testing.assert_allclose(got, np.exp(-d / THETA) / THETA, rtol=1e-10, atol=1e-12)


@requires_nest
class TestGammaNestParity(unittest.TestCase):
    """The kernel matches live NEST element-by-element for several shape/scale pairs."""

    def test_weight_readback_matches_nest(self):
        layer = grid(list(SHAPE), extent=list(EXTENT))
        for kappa, theta in [(2.0, 0.5), (1.0, 1.3), (3.5, 0.4)]:
            bp = _bp_matrix(gamma(distance, kappa=kappa, theta=theta), layer)
            nst = _nest_weight_matrix(
                nest.spatial_distributions.gamma(nest.spatial.distance, kappa=kappa, theta=theta),
                layer.n)
            off = ~np.eye(layer.n, dtype=bool)
            self.assertLess(float(np.nanmax(np.abs(bp - nst)[off])), 1e-9,
                            msg=f'gamma(kappa={kappa}, theta={theta}) diverges from NEST')


if __name__ == '__main__':
    unittest.main()
