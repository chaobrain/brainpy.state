# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Distributional parity for the Gaussian distance kernel (goal 20, ``spatial/gaussex.py``).

Two grid populations are connected with a distance-dependent pairwise-Bernoulli rule whose
probability is a Gaussian of the source->target distance,
:math:`p(d) = \exp(-d^2 / (2\,\mathrm{std}^2))`, clipped to a circular mask. Individual edge
draws diverge sample-by-sample between simulators (independent PRNGs), so the parity arbiter
is the **law**, not the sample: the empirical connection fraction binned by distance must
track the Gaussian, and brainpy.state's curve must match live NEST's within a band.

Two classes (cluster-16 house style):

* ``TestGaussianKernelStructure`` -- NEST-free, always runs: with a fixed seed the empirical
  ``p(d)`` tracks the analytic Gaussian per distance bin, the total edge count sits near the
  analytic expectation, no edge exceeds the mask radius, and the example runs standalone.
* ``TestGaussianKernelNestParity`` (``@requires_nest``) -- the same grid + kernel vs live NEST
  (``pairwise_bernoulli`` + ``spatial_distributions.gaussian`` + circular mask): the empirical
  per-distance curves agree bin-by-bin and the mean edge count agrees across seeds.
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
from brainpy_state._nest_spatial import (
    grid, gaussian, distance, circular, center_element, spatial_pairwise_bernoulli,
)
from brainpy_state._nest_validation.nest_compare import requires_nest

SHAPE = (20, 20)
EXTENT = (3.0, 3.0)
STD = 0.5
RADIUS = 3.0
MAXD = 1.5            # bins beyond this carry p < exp(-4.5) ~ 0.01 (negligible)
NBINS = 15
MIN_COUNT = 100      # ignore distance bins with too few ordered pairs to be statistical


def _pairwise_d(coords_um):
    d = coords_um[:, None, :] - coords_um[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def _curve(coords_um, src, tgt):
    """Empirical and analytic connection fraction per distance bin.

    Returns ``(centers, empirical, analytic_mean, counts)`` where ``analytic_mean`` is the
    mean Gaussian ``p(d)`` over the ordered pairs falling in each bin (the exact expected
    fraction), so the comparison is bin-resolution-independent.
    """
    n = coords_um.shape[0]
    D = _pairwise_d(coords_um)
    realized = np.zeros((n, n), dtype=bool)
    realized[np.asarray(src), np.asarray(tgt)] = True
    dflat = D.ravel()
    rflat = realized.ravel()
    edges = np.linspace(0.0, MAXD, NBINS + 1)
    dig = np.digitize(dflat, edges)
    centers, emp, ana, counts = [], [], [], []
    for b in range(1, NBINS + 1):
        sel = dig == b
        tot = int(sel.sum())
        if tot < MIN_COUNT:
            continue
        centers.append(0.5 * (edges[b - 1] + edges[b]))
        emp.append(float(rflat[sel].mean()))
        ana.append(float(np.exp(-(dflat[sel] ** 2) / (2 * STD ** 2)).mean()))
        counts.append(tot)
    return (np.array(centers), np.array(emp), np.array(ana), np.array(counts))


def _expected_total_edges(coords_um):
    """Analytic expected edge count = sum over ordered pairs of masked ``p(d)``."""
    D = _pairwise_d(coords_um)
    p = np.exp(-(D ** 2) / (2 * STD ** 2))
    p[D > RADIUS] = 0.0
    return float(p.sum())


def _build_bp(seed, shape=SHAPE):
    sim = Simulator(dt=0.1 * u.ms)
    pos = grid(list(shape), extent=list(EXTENT))
    a = sim.create(iaf_psc_alpha, positions=pos)
    b = sim.create(iaf_psc_alpha, positions=pos)
    sim.connect(a, b, rule=spatial_pairwise_bernoulli(
        p=gaussian(distance, std=STD), mask=circular(RADIUS)),
        weight=1.0 * u.pA, delay=1.0 * u.ms, seed=seed)
    sc = sim.get_connections(source=a, target=b)
    coords = np.asarray(u.get_magnitude(sim.get_position(a).to(u.um)))
    return coords, np.asarray(sc.source), np.asarray(sc.target)


class TestGaussianKernelStructure(unittest.TestCase):
    """The realized connectivity follows the Gaussian distance law (NEST-free, fixed seed)."""

    def test_probability_follows_gaussian_law(self):
        coords, src, tgt = _build_bp(seed=0)
        centers, emp, ana, counts = _curve(coords, src, tgt)
        self.assertGreaterEqual(len(centers), 6)              # several statistical bins
        # the empirical fraction tracks the analytic Gaussian per bin
        self.assertLess(float(np.max(np.abs(emp - ana))), 0.06,
                        msg=f'p(d) deviates: emp={emp}, ana={ana}')
        self.assertLess(float(np.mean(np.abs(emp - ana))), 0.025)
        # monotone decreasing with distance (the kernel is decreasing)
        self.assertTrue(np.all(np.diff(emp) <= 0.02))

    def test_total_edges_near_expected(self):
        coords, src, tgt = _build_bp(seed=0)
        expected = _expected_total_edges(coords)
        self.assertLess(abs(len(src) - expected) / expected, 0.05,
                        msg=f'edges={len(src)} vs expected~{expected:.0f}')

    def test_no_edge_beyond_mask(self):
        coords, src, tgt = _build_bp(seed=1)
        D = _pairwise_d(coords)
        self.assertLessEqual(float(D[np.asarray(src), np.asarray(tgt)].max()), RADIUS + 1e-9)

    def test_example_run_smoke(self):
        from examples.nest_like.spatial_gaussex import run
        coords, ctr, tgt = run(shape=(10, 10))
        self.assertEqual(coords.shape, (100, 2))
        self.assertGreaterEqual(tgt.shape[0], 1)

    def test_csa_spatial_example_run_smoke(self):
        # csa_spatial_example ported natively: same Gaussian-kernel family as gaussex.
        from examples.nest_like.spatial_csa import run
        coords, ctr, tgt = run(shape=(8, 8))
        self.assertEqual(coords.shape, (64, 2))
        self.assertGreaterEqual(tgt.shape[0], 1)

    def test_csa_example_native_density(self):
        # csa_example's csa.random(0.1) maps to pairwise_bernoulli(0.1); density ~ p.
        from examples.nest_like.csa_example import run
        self.assertTrue(0.08 < run(n=200, p=0.1) < 0.12)


# --- NEST side --------------------------------------------------------------------

def _build_nest(rng_seed, shape=SHAPE):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'rng_seed': int(rng_seed)})
    pos = nest.spatial.grid(shape=list(shape), extent=list(EXTENT))
    a = nest.Create('iaf_psc_alpha', positions=pos)
    b = nest.Create('iaf_psc_alpha', positions=pos)
    nest.Connect(a, b, {
        'rule': 'pairwise_bernoulli',
        'p': nest.spatial_distributions.gaussian(nest.spatial.distance, std=STD),
        'mask': {'circular': {'radius': RADIUS}},
    })
    conns = nest.GetConnections(a, b)
    src = np.asarray(conns.source) - a[0].global_id
    tgt = np.asarray(conns.target) - b[0].global_id
    coords = np.asarray(nest.GetPosition(a))
    return coords, src, tgt


@requires_nest
class TestGaussianKernelNestParity(unittest.TestCase):
    """The empirical per-distance curve and edge count match live NEST."""

    def test_empirical_curve_matches_nest(self):
        cb, sb, tb = _build_bp(seed=0)
        cn, sn, tn = _build_nest(rng_seed=12345)
        _, emp_b, ana_b, _ = _curve(cb, sb, tb)
        _, emp_n, ana_n, _ = _curve(cn, sn, tn)
        k = min(len(emp_b), len(emp_n))
        # both track the same analytic law, hence each other, bin-by-bin
        self.assertLess(float(np.max(np.abs(emp_b[:k] - emp_n[:k]))), 0.07,
                        msg=f'bp={emp_b[:k]} nest={emp_n[:k]}')

    def test_total_edges_seed_mean_matches_nest(self):
        bp_mean = np.mean([len(_build_bp(seed=s)[1]) for s in range(4)])
        nest_mean = np.mean([len(_build_nest(rng_seed=s)[1]) for s in (11, 22, 33, 44)])
        self.assertLess(abs(bp_mean - nest_mean) / nest_mean, 0.05,
                        msg=f'bp mean edges {bp_mean:.0f} vs NEST {nest_mean:.0f}')


if __name__ == '__main__':
    unittest.main()
