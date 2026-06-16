# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Parity for the plot_weight_matrices demo (§3.4 connection introspection).

The demo ``examples/nest_like/plot_weight_matrices.py`` builds an E/I network with
``fixed_indegree`` connectivity (excitatory weights ``Normal(20, 0.5)`` pA,
inhibitory ``-g`` times as large) and extracts the four ``EE / EI / IE / II``
weight matrices via :meth:`Simulator.get_connections` — the
``GetConnections``/``SynapseCollection`` idiom.

**What is exact vs distributional.** The *connectivity structure* is exact on both
sides: ``fixed_indegree(K)`` gives every post neuron exactly ``K`` incoming edges,
so edge counts and per-target in-degrees match NEST bit-for-bit. The *weights* are
PRNG draws from the same distribution, so they agree only **distributionally**
(category D): the seed-mean excitatory / inhibitory weight matches NEST's, and the
inhibitory mean is ``-g`` times the excitatory mean.

The NEST-free tests pin the structure (shapes, in-degree, weight signs, the ``-g``
scaling). The ``@requires_nest`` tests confirm the seed-mean weights match live
NEST.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state._nest_validation.nest_compare import compare_distributional, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

import examples.nest_like.plot_weight_matrices as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

_SEEDS = (0, 1, 2, 3, 4)


def _pooled_weights(sim, source, target):
    """Per-edge weights (pA mantissa) for one population pairing."""
    conns = sim.get_connections(source=source, target=target)
    return np.asarray(u.Quantity(conns.get('weight')).to_decimal(u.pA))


class TestPlotWeightMatricesStructure(unittest.TestCase):
    """NEST-free: connectivity structure, weight signs, and the -g scaling."""

    def test_matrix_shapes(self):
        sim, E, I = demo.build()
        W = demo.weight_matrices(sim, E, I)
        self.assertEqual(W['EE'].shape, (demo.NE, demo.NE))
        self.assertEqual(W['EI'].shape, (demo.NI, demo.NE))   # I -> E (post-pre)
        self.assertEqual(W['IE'].shape, (demo.NE, demo.NI))   # E -> I (post-pre)
        self.assertEqual(W['II'].shape, (demo.NI, demo.NI))

    def test_edge_counts_match_fixed_indegree(self):
        # fixed_indegree(K) gives every post neuron exactly K incoming edges, so the
        # total edge count is K * n_post (multapses included).
        sim, E, I = demo.build()
        self.assertEqual(len(sim.get_connections(source=E, target=E)), demo.CE * demo.NE)
        self.assertEqual(len(sim.get_connections(source=E, target=I)), demo.CE * demo.NI)
        self.assertEqual(len(sim.get_connections(source=I, target=E)), demo.CI * demo.NE)
        self.assertEqual(len(sim.get_connections(source=I, target=I)), demo.CI * demo.NI)

    def test_each_post_has_fixed_indegree(self):
        # Every excitatory post neuron receives exactly CE edges from E and CI from I.
        sim, E, I = demo.build()
        ee = sim.get_connections(source=E, target=E)
        ie = sim.get_connections(source=I, target=E)
        ee_indeg = np.bincount(np.asarray(ee.target), minlength=demo.NE)
        ie_indeg = np.bincount(np.asarray(ie.target), minlength=demo.NE)
        self.assertTrue(np.all(ee_indeg == demo.CE))
        self.assertTrue(np.all(ie_indeg == demo.CI))

    def test_weight_signs(self):
        # Excitatory edges (E source) are positive; inhibitory (I source) negative.
        sim, E, I = demo.build()
        exc = _pooled_weights(sim, E, E + I)
        inh = _pooled_weights(sim, I, E + I)
        self.assertTrue(np.all(exc > 0.0))
        self.assertTrue(np.all(inh < 0.0))

    def test_inhibitory_is_minus_g_times_excitatory(self):
        # Distributional fact (same demo seed): mean(w_in) ~ -g * mean(w_ex).
        sim, E, I = demo.build()
        exc_mean = float(_pooled_weights(sim, E, E + I).mean())
        inh_mean = float(_pooled_weights(sim, I, E + I).mean())
        self.assertAlmostEqual(exc_mean, demo.W_EX_MEAN, delta=0.2)         # ~20 pA
        self.assertAlmostEqual(inh_mean, -demo.G * demo.W_EX_MEAN, delta=1.0)  # ~-100 pA

    def test_reproducible_given_seed(self):
        sim_a, E_a, I_a = demo.build(seed=3)
        sim_b, E_b, I_b = demo.build(seed=3)
        wa = _pooled_weights(sim_a, E_a, E_a + I_a)
        wb = _pooled_weights(sim_b, E_b, E_b + I_b)
        self.assertTrue(np.array_equal(wa, wb))            # same seed -> identical draws


def _bp_weight_means(seed):
    """brainpy seed run -> (mean excitatory weight, mean inhibitory weight) in pA."""
    sim, E, I = demo.build(seed=seed)
    return (float(_pooled_weights(sim, E, E + I).mean()),
            float(_pooled_weights(sim, I, E + I).mean()))


def _nest_weight_means(seed):
    """Live-NEST run mirroring the demo's wiring -> (mean w_ex, mean w_in) in pA."""
    nest.ResetKernel()
    nest.resolution = demo.DT
    nest.rng_seed = seed + 1
    nest.set_verbosity("M_ERROR")
    E = nest.Create("iaf_psc_alpha", demo.NE)
    I = nest.Create("iaf_psc_alpha", demo.NI)
    w_ex = nest.random.normal(mean=demo.W_EX_MEAN, std=demo.W_EX_STD)
    w_in = -demo.G * w_ex
    nest.Connect(E, E + I, {"rule": "fixed_indegree", "indegree": demo.CE},
                 {"synapse_model": "static_synapse", "weight": w_ex, "delay": demo.DELAY})
    nest.Connect(I, E + I, {"rule": "fixed_indegree", "indegree": demo.CI},
                 {"synapse_model": "static_synapse", "weight": w_in, "delay": demo.DELAY})
    exc = np.asarray(nest.GetConnections(E, E + I).weight, dtype=float)
    inh = np.asarray(nest.GetConnections(I, E + I).weight, dtype=float)
    return float(exc.mean()), float(inh.mean())


@requires_nest
class TestPlotWeightMatricesParity(unittest.TestCase):
    """The seed-mean per-edge weights match live NEST (category D)."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_NEST:
            return
        cls._nest = [_nest_weight_means(s) for s in _SEEDS]
        cls._bp = [_bp_weight_means(s) for s in _SEEDS]

    def test_excitatory_weight_mean_matches_nest(self):
        compare_distributional([r[0] for r in self._nest], [r[0] for r in self._bp],
                               tol=tc.CAT_D, metric="plot_weight_matrices w_ex",
                               statistic="mean").assert_()

    def test_inhibitory_weight_mean_matches_nest(self):
        compare_distributional([r[1] for r in self._nest], [r[1] for r in self._bp],
                               tol=tc.CAT_D, metric="plot_weight_matrices w_in",
                               statistic="mean").assert_()

    def test_indegree_structure_matches_nest(self):
        # Connectivity is exact on both sides: every post gets exactly CE/CI inputs.
        nest.ResetKernel()
        nest.resolution = demo.DT
        nest.rng_seed = 1
        nest.set_verbosity("M_ERROR")
        E = nest.Create("iaf_psc_alpha", demo.NE)
        I = nest.Create("iaf_psc_alpha", demo.NI)
        nest.Connect(E, E + I, {"rule": "fixed_indegree", "indegree": demo.CE},
                     {"weight": 1.0})
        nest.Connect(I, E + I, {"rule": "fixed_indegree", "indegree": demo.CI},
                     {"weight": 1.0})
        n_ee = len(nest.GetConnections(E, E))
        n_ie = len(nest.GetConnections(I, E))
        self.assertEqual(n_ee, demo.CE * demo.NE)
        self.assertEqual(n_ie, demo.CI * demo.NE)


if __name__ == "__main__":
    unittest.main()
