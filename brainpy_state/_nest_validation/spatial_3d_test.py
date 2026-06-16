# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Distributional + structural parity for the 3D Gaussian demo (goal 20, ``spatial/test_3d_gauss.py``).

1000 neurons sit at uniformly-random 3D positions and connect to themselves with a Gaussian
distance kernel, **no autapses**, clipped to a cubic box mask anchored on the source. The
random layer positions diverge sample-by-sample from NEST (independent PRNGs), so the parity
arbiter is position-independent: the box cutoff is a **hard** per-axis bound, autapses are
absent, and the empirical ``p(d)`` and edge count match live NEST distributionally.

Two classes (cluster-16 house style):

* ``TestSpatial3DStructure`` -- NEST-free, always runs: the box mask is a hard per-axis cutoff
  (verified with a binding box), no edge is an autapse, the empirical ``p(d)`` tracks the
  analytic Gaussian in 3D (fixed seed), and the example runs standalone.
* ``TestSpatial3DNestParity`` (``@requires_nest``) -- the same 3D free layer + kernel + box vs
  live NEST: empirical per-distance curves agree bin-by-bin and the mean edge count agrees
  across seeds (positions differ, the law does not).
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
from brainpy_state._dist import Uniform
from brainpy_state._nest_spatial import (
    grid, free, gaussian, distance, box, spatial_pairwise_bernoulli,
)
from brainpy_state._nest_validation.nest_compare import requires_nest

EXTENT = (1.5, 1.5, 1.5)
STD = 0.25
BOX = 0.75
MAXD = 0.8           # p < exp(-5.1) ~ 0.006 beyond this for std=0.25
NBINS = 16
MIN_COUNT = 100


def _pairwise_d(coords_um):
    d = coords_um[:, None, :] - coords_um[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def _curve(coords_um, src, tgt, std=STD):
    n = coords_um.shape[0]
    D = _pairwise_d(coords_um)
    realized = np.zeros((n, n), dtype=bool)
    realized[np.asarray(src), np.asarray(tgt)] = True
    # autapses are structurally disallowed, so the diagonal is not a candidate pair --
    # excluding it keeps the d~0 bin from being diluted by n unconnectable self-pairs.
    offdiag = ~np.eye(n, dtype=bool)
    dflat, rflat = D[offdiag], realized[offdiag]
    edges = np.linspace(0.0, MAXD, NBINS + 1)
    dig = np.digitize(dflat, edges)
    centers, emp, ana = [], [], []
    for b in range(1, NBINS + 1):
        sel = dig == b
        if int(sel.sum()) < MIN_COUNT:
            continue
        centers.append(0.5 * (edges[b - 1] + edges[b]))
        emp.append(float(rflat[sel].mean()))
        ana.append(float(np.exp(-(dflat[sel] ** 2) / (2 * std ** 2)).mean()))
    return np.array(centers), np.array(emp), np.array(ana)


def _build_bp(n=1000, std=STD, box_half=BOX, seed=0):
    sim = Simulator(dt=0.1 * u.ms)
    pos = free(Uniform(-0.5, 0.5), extent=list(EXTENT))
    l1 = sim.create(iaf_psc_alpha, n, positions=pos)
    sim.connect(l1, l1, rule=spatial_pairwise_bernoulli(
        p=gaussian(distance, std=std),
        mask=box(lower_left=[-box_half] * 3, upper_right=[box_half] * 3)),
        weight=1.0 * u.pA, delay=1.0 * u.ms, allow_autapses=False, seed=seed)
    sc = sim.get_connections(source=l1, target=l1)
    coords = np.asarray(u.get_magnitude(sim.get_position(l1).to(u.um)))
    return coords, np.asarray(sc.source), np.asarray(sc.target)


class TestSpatial3DStructure(unittest.TestCase):
    """The 3D realized connectivity obeys the box, autapse, and Gaussian-law invariants."""

    def test_box_mask_hard_cutoff(self):
        # large std so the kernel alone would connect far pairs; a tight box must cut every
        # realized edge whose per-axis displacement exceeds the box half-width.
        coords, src, tgt = _build_bp(n=400, std=1.0, box_half=0.4, seed=0)
        disp = coords[np.asarray(tgt)] - coords[np.asarray(src)]
        self.assertLessEqual(float(np.abs(disp).max()), 0.4 + 1e-9)
        # the box is actually binding: many pairs lie beyond it
        full = _pairwise_d(coords)
        beyond = (np.abs(coords[:, None, :] - coords[None, :, :]).max(-1) > 0.4)
        self.assertGreater(int(beyond.sum()), 1000)

    def test_no_autapses(self):
        coords, src, tgt = _build_bp(n=500, seed=2)
        self.assertEqual(int(np.sum(np.asarray(src) == np.asarray(tgt))), 0)

    def test_probability_follows_gaussian_law_3d(self):
        coords, src, tgt = _build_bp(seed=0)
        centers, emp, ana = _curve(coords, src, tgt)
        self.assertGreaterEqual(len(centers), 6)
        self.assertLess(float(np.max(np.abs(emp - ana))), 0.06,
                        msg=f'p(d) deviates: emp={emp}, ana={ana}')
        self.assertLess(float(np.mean(np.abs(emp - ana))), 0.025)

    def test_example_run_smoke(self):
        from examples.nest.spatial_3d_gauss import run
        coords, ctr, tgt, dist = run(n=300)
        self.assertEqual(coords.shape, (300, 3))
        self.assertEqual(dist.shape[0], tgt.shape[0])


# --- NEST side --------------------------------------------------------------------

def _build_nest(n=1000, rng_seed=12345):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'rng_seed': int(rng_seed)})
    pos = nest.spatial.free(nest.random.uniform(-0.5, 0.5), extent=list(EXTENT))
    l1 = nest.Create('iaf_psc_alpha', n, positions=pos)
    nest.Connect(l1, l1, {
        'rule': 'pairwise_bernoulli',
        'p': nest.spatial_distributions.gaussian(nest.spatial.distance, std=STD),
        'allow_autapses': False,
        'mask': {'box': {'lower_left': [-BOX] * 3, 'upper_right': [BOX] * 3}},
    })
    conns = nest.GetConnections(l1, l1)
    src = np.asarray(conns.source) - l1[0].global_id
    tgt = np.asarray(conns.target) - l1[0].global_id
    coords = np.asarray(nest.GetPosition(l1))
    return coords, src, tgt


@requires_nest
class TestSpatial3DNestParity(unittest.TestCase):
    """The 3D empirical curve and edge count match live NEST distributionally."""

    def test_empirical_curve_matches_nest(self):
        cb, sb, tb = _build_bp(seed=0)
        cn, sn, tn = _build_nest(rng_seed=12345)
        _, emp_b, _ = _curve(cb, sb, tb)
        _, emp_n, _ = _curve(cn, sn, tn)
        k = min(len(emp_b), len(emp_n))
        self.assertLess(float(np.max(np.abs(emp_b[:k] - emp_n[:k]))), 0.07,
                        msg=f'bp={emp_b[:k]} nest={emp_n[:k]}')

    def test_total_edges_seed_mean_matches_nest(self):
        bp_mean = np.mean([len(_build_bp(seed=s)[1]) for s in range(3)])
        nest_mean = np.mean([len(_build_nest(rng_seed=s)[1]) for s in (11, 22, 33)])
        self.assertLess(abs(bp_mean - nest_mean) / nest_mean, 0.06,
                        msg=f'bp mean edges {bp_mean:.0f} vs NEST {nest_mean:.0f}')


if __name__ == '__main__':
    unittest.main()
