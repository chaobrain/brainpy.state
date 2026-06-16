# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Parity for the synapsecollection demo (§3.4 connection introspection).

The demo ``examples/nest_like/synapsecollection.py`` exercises the
``GetConnections``/``SynapseCollection`` API across several rules, synapse models
and weight distributions: ``get(['source','target','weight'])``,
``set('weight', ...)`` round-trips, and ``get_connections`` filtered by
source/target slice or by synapse model.

**What is exact vs distributional.** Deterministic-count rules (``one_to_one``,
``all_to_all``, ``fixed_total_number(N)``) realize the *same number* of edges as
NEST, and a per-edge weight ``set`` produces the *same* weights on both sides, so
those are compared exactly. The ``Uniform`` weight *values* are PRNG draws that
agree only **distributionally** (category D): the seed-mean weight matches NEST's.
The random topology of ``pairwise_bernoulli`` / ``fixed_total_number`` differs
between the PRNGs, so only counts (not the realized pairs) are compared there.

The NEST-free tests pin the introspection idioms (identity, set round-trips,
source/target and synapse-model filters). The ``@requires_nest`` tests confirm the
deterministic counts and set-weights match live NEST and the ``Uniform`` weight
mean matches distributionally.
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

import examples.nest_like.synapsecollection as demo

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

_SEEDS = (0, 1, 2, 3, 4)


def _weights_pA(conns):
    return np.asarray(u.Quantity(conns.get('weight')).to_decimal(u.pA))


class TestSynapseCollectionIntrospection(unittest.TestCase):
    """NEST-free: the introspection idioms the demo demonstrates."""

    def test_one_to_one_identity_and_set_roundtrip(self):
        sim, _nrns = demo.build_one_to_one(10)
        conns = sim.get_connections()
        srcs, tgts, w0, _ = demo.collection_arrays(conns)
        self.assertEqual(len(conns), 10)
        self.assertTrue(np.array_equal(srcs, tgts))            # identity
        self.assertTrue(np.allclose(w0, 1.0))                  # uniform init
        conns.set('weight', demo.ramp_weights(10))             # per-edge set
        self.assertTrue(np.allclose(_weights_pA(conns), np.arange(1, 11)))

    def test_all_to_all_count_and_uniform_range(self):
        sim, _pre, _post = demo.build_all_to_all(10, 5, seed=0)
        conns = sim.get_connections()
        self.assertEqual(len(conns), 50)                       # 10 x 5
        w = _weights_pA(conns)
        self.assertTrue(np.all(w >= 0.5) and np.all(w <= 4.5))  # Uniform(0.5, 4.5)

    def test_complex_filters(self):
        sim, nrns = demo.build_complex(15, seed=0)
        all_conns = sim.get_connections()
        stdp = sim.get_connections(synapse='stdp_synapse')
        ftn = sim.get_connections(source=nrns[5:10], target=nrns[:5])
        # stdp count is deterministic: one_to_one(5) + all_to_all(5x12=60) = 65.
        self.assertEqual(len(stdp), 65)
        self.assertEqual(len(ftn), 5)                          # fixed_total_number(5)
        self.assertEqual(len(all_conns), max(len(all_conns), 65))  # all >= stdp subset
        self.assertGreaterEqual(len(all_conns), len(stdp))

    def test_complex_subset_set_roundtrip(self):
        sim, nrns = demo.build_complex(15, seed=0)
        ftn = sim.get_connections(source=nrns[5:10], target=nrns[:5])
        ftn.set('weight', demo.ramp_weights(len(ftn)))
        self.assertTrue(np.allclose(_weights_pA(ftn), np.arange(1, len(ftn) + 1)))

    def test_subset_excludes_out_of_range_sources(self):
        sim, nrns = demo.build_complex(15, seed=0)
        subset = sim.get_connections(source=nrns[:10], target=nrns[:10])
        self.assertTrue(np.all(np.asarray(subset.source) < 10))
        self.assertTrue(np.all(np.asarray(subset.target) < 10))

    def test_model_filter_selects_only_stdp(self):
        sim, _nrns = demo.build_complex(15, seed=0)
        stdp = sim.get_connections(synapse='stdp_synapse')
        # The stdp connects use a 5 pA (one_to_one) and 4 pA (all_to_all) init.
        w = _weights_pA(stdp)
        self.assertTrue(np.all(np.isin(np.round(w, 6), [5.0, 4.0])))


def _bp_all_to_all_mean(seed):
    sim, _pre, _post = demo.build_all_to_all(10, 5, seed=seed)
    return float(_weights_pA(sim.get_connections()).mean())


def _nest_all_to_all_mean(seed):
    nest.ResetKernel()
    nest.resolution = demo.DT
    nest.rng_seed = seed + 1
    nest.set_verbosity("M_ERROR")
    pre = nest.Create("iaf_psc_alpha", 10)
    post = nest.Create("iaf_psc_alpha", 5)
    nest.Connect(pre, post, "all_to_all",
                 {"weight": nest.random.uniform(min=0.5, max=4.5)})
    return float(np.asarray(nest.GetConnections().weight, dtype=float).mean())


@requires_nest
class TestSynapseCollectionParity(unittest.TestCase):
    """Deterministic counts / set-weights and the Uniform weight mean match NEST."""

    def test_one_to_one_set_matches_nest(self):
        # Per-edge set on a one_to_one connection -> identical [1..10] on both sides.
        sim, _nrns = demo.build_one_to_one(10)
        conns = sim.get_connections()
        conns.set('weight', demo.ramp_weights(10))
        bp_w = np.sort(_weights_pA(conns))

        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")
        nrns = nest.Create("iaf_psc_alpha", 10)
        nest.Connect(nrns, nrns, "one_to_one")
        nconns = nest.GetConnections(nrns, nrns)
        nconns.set([{"weight": float(x)} for x in range(1, 11)])
        nest_w = np.sort(np.asarray(nconns.weight, dtype=float))
        self.assertTrue(np.allclose(bp_w, nest_w))

    def test_all_to_all_count_matches_nest(self):
        sim, _pre, _post = demo.build_all_to_all(10, 5, seed=0)
        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")
        pre = nest.Create("iaf_psc_alpha", 10)
        post = nest.Create("iaf_psc_alpha", 5)
        nest.Connect(pre, post, "all_to_all")
        self.assertEqual(len(sim.get_connections()), len(nest.GetConnections()))

    def test_all_to_all_uniform_weight_mean_matches_nest(self):
        bp = [_bp_all_to_all_mean(s) for s in _SEEDS]
        nst = [_nest_all_to_all_mean(s) for s in _SEEDS]
        compare_distributional(nst, bp, tol=tc.CAT_D,
                               metric="synapsecollection all_to_all Uniform weight",
                               statistic="mean").assert_()

    def test_stdp_model_filter_count_matches_nest(self):
        # Deterministic synapse-model counts: one_to_one(5) + all_to_all(5x12) = 65.
        sim, nrns = demo.build_complex(15, seed=0)
        bp_stdp = len(sim.get_connections(synapse='stdp_synapse'))

        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")
        n = nest.Create("iaf_psc_alpha", 15)
        nest.Connect(n[:5], n[:5], "one_to_one",
                     {"synapse_model": "stdp_synapse", "weight": 5.0})
        nest.Connect(n[10:], n[:12], "all_to_all",
                     {"synapse_model": "stdp_synapse", "weight": 4.0})
        nest_stdp = len(nest.GetConnections(synapse_model="stdp_synapse"))
        self.assertEqual(bp_stdp, nest_stdp)
        self.assertEqual(bp_stdp, 65)


if __name__ == "__main__":
    unittest.main()
