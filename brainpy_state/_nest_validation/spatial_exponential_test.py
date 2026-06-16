# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Parity for the exponential distance distribution (goal 27).

NEST's ``spatial_distributions.exponential(distance, beta)`` is :math:`p(d) = e^{-d/\beta}`.
The arbiter here is the **kernel itself**, read back deterministically: connecting a layer to
itself with ``weight`` set to the spatial parameter materialises ``p(d)`` per ordered pair, so
brainpy.state's kernel must match live NEST's to machine precision (no PRNG involved). A second,
NEST-free class checks the realised connection fraction follows the exponential law.
"""
import unittest

import jax
import brainstate
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import brainunit as u

try:
    import nest
except Exception:
    nest = None

from brainpy_state import Simulator, iaf_psc_alpha
from brainpy_state._nest_spatial import grid, distance, exponential, spatial_pairwise_bernoulli
from brainpy_state._nest_validation.nest_compare import requires_nest

SHAPE = (3, 3)
EXTENT = (2.0, 2.0)
BETA = 0.7


def _bp_matrix(kernel, layer):
    return np.asarray(u.get_magnitude(kernel._eval_pair(layer.coords, layer.coords)))


def _nest_weight_matrix(param, n, shape=SHAPE, extent=EXTENT):
    """Materialise a spatial parameter as the (n, n) ordered-pair weight matrix in NEST."""
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


def _realized_fraction(seed, shape=(20, 20)):
    sim = Simulator(dt=0.1 * u.ms)
    pos = grid(list(shape), extent=[3.0, 3.0])
    a = sim.create(iaf_psc_alpha, positions=pos)
    b = sim.create(iaf_psc_alpha, positions=pos)
    sim.connect(a, b, rule=spatial_pairwise_bernoulli(p=exponential(distance, beta=BETA)),
                weight=1.0 * u.pA, delay=1.0 * u.ms, seed=seed)
    sc = sim.get_connections(source=a, target=b)
    coords = np.asarray(u.get_magnitude(sim.get_position(a).to(u.um)))
    n = coords.shape[0]
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    realized = np.zeros((n, n), dtype=bool)
    realized[np.asarray(sc.source), np.asarray(sc.target)] = True
    return D.ravel(), realized.ravel()


class TestExponentialStructure(unittest.TestCase):
    """The kernel equals ``exp(-d/beta)`` and drives an exponentially-decaying footprint."""

    def test_kernel_matches_formula(self):
        layer = grid([1, 4], extent=[1.0, 4.0])             # distances 0,1,2,3 along y
        got = _bp_matrix(exponential(distance, beta=BETA), layer)[0]
        d = np.asarray(u.get_magnitude((layer.coords[:, 1] - layer.coords[0, 1]).to(u.um)))
        np.testing.assert_allclose(got, np.exp(-np.abs(d) / BETA), atol=1e-12)

    def test_realized_fraction_follows_law(self):
        d, r = _realized_fraction(seed=0)
        near = (d > 0.0) & (d <= 0.4)
        far = (d >= 1.2) & (d < 1.6)
        f_near, f_far = r[near].mean(), r[far].mean()
        self.assertGreater(f_near, f_far)                   # decreasing with distance
        # each bin's empirical fraction tracks the analytic mean p(d)
        self.assertLess(abs(f_near - np.exp(-d[near] / BETA).mean()), 0.05)
        self.assertLess(abs(f_far - np.exp(-d[far] / BETA).mean()), 0.05)


@requires_nest
class TestExponentialNestParity(unittest.TestCase):
    """The kernel matches live NEST element-by-element; realised edges agree on average."""

    def test_weight_readback_matches_nest(self):
        layer = grid(list(SHAPE), extent=list(EXTENT))
        bp = _bp_matrix(exponential(distance, beta=BETA), layer)
        nst = _nest_weight_matrix(
            nest.spatial_distributions.exponential(nest.spatial.distance, beta=BETA), layer.n)
        off = ~np.eye(layer.n, dtype=bool)
        self.assertLess(float(np.nanmax(np.abs(bp - nst)[off])), 1e-9)

    def test_realized_edges_seed_mean_matches_nest(self):
        bp_mean = np.mean([int(r.sum()) for _, r in (_realized_fraction(s) for s in range(4))])
        nest_means = []
        for rng in (11, 22, 33, 44):
            nest.ResetKernel()
            nest.set_verbosity('M_ERROR')
            nest.SetKernelStatus({'rng_seed': int(rng)})
            pos = nest.spatial.grid(shape=[20, 20], extent=[3.0, 3.0])
            a = nest.Create('iaf_psc_alpha', positions=pos)
            b = nest.Create('iaf_psc_alpha', positions=pos)
            nest.Connect(a, b, {'rule': 'pairwise_bernoulli',
                                'p': nest.spatial_distributions.exponential(
                                    nest.spatial.distance, beta=BETA)})
            nest_means.append(len(nest.GetConnections(a, b)))
        self.assertLess(abs(bp_mean - np.mean(nest_means)) / np.mean(nest_means), 0.05)


if __name__ == '__main__':
    unittest.main()
