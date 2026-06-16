# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST micro-parity for ``tripartite_connect`` (NEST ``TripartiteConnect``, goal 24).

The **design-A arbiter**. The Simulator's ``tripartite_connect`` samples ONE primary
``pre->post`` edge set and shares it across three arms -- primary (``pre->post``),
``third_in`` (``pre->astro``), and ``third_out`` (``astro->post``, a
``sic_connection``) -- pairing each primary edge with one pooled astrocyte via the
``third_factor_bernoulli_with_pool`` rule. This test validates that shared-sample
connectivity against live NEST's ``nest.TripartiteConnect`` at micro scale, along
two pillars (each pairs a NEST-free structural law that always runs with a
``@requires_nest`` edge parity):

* **Exact block** -- ``p_primary=1``, ``p_third=1``, ``pool_size=1``,
  ``pool_type='block'``: the connectivity is deterministic, so the three realized
  edge **sets** (n2n, n2a, a2n) must be *identical* between brainpy and NEST.
* **Distributional random** -- ``pool_type='random'``, ``p<1``: PRNG streams
  diverge, so compare seed-mean distinct-edge counts (category D, 5 %) and assert
  the hard pool invariant (each target's astrocytes are drawn from a pool of at
  most ``pool_size``) on both sides.

The arms use ``static_synapse`` (primary, ``third_in``) + ``sic_connection``
(``third_out``) on **both** sides. NEST's demo uses ``tsodyks_synapse`` for the
primary/third_in arms, but edge *connectivity* is synapse-agnostic and the
15d-validated SIC loop uses static synapses (goal 24 spec §3); the connection
**set** is identical either way, so this is a documented, parity-neutral divergence.

Realized numbers (measured, recorded in the cluster spec §8): block edge sets are
bit-identical (n2n 24, n2a 12, a2n 6 for the 4x6x3 micro net). Random seed-mean
distinct counts land within category D of NEST (see ``RANDOM_*`` below).
"""
import gc
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

try:
    import nest
except Exception:                                         # pragma: no cover - env dependent
    nest = None

from brainpy_state import (Simulator, aeif_cond_alpha_astro, astrocyte_lr_1994,
                           sic_connection, pairwise_bernoulli,
                           third_factor_bernoulli_with_pool)
from brainpy_state._nest_validation.nest_compare import (
    requires_nest, compare_distributional)
from brainpy_state._nest_validation.tolerance_conventions import CAT_D

DT = 0.1

# --- micro configurations ----------------------------------------------------------

#: Exact block net: pre x post = 4 x 6, 3 astrocytes; block factor n_post/n_third = 2.
#: post_j is paired with astrocyte j // 2, deterministically (p_primary=p_third=1).
BLK_PRE, BLK_POST, BLK_THIRD = 4, 6, 3
BLK_FACTOR = BLK_POST // BLK_THIRD                        # 2 post neurons per astrocyte

#: Random net: larger so seed-mean counts are stable; pool of 3 of 10 astrocytes.
RND_PRE, RND_POST, RND_THIRD = 30, 30, 15
RND_POOL = 3
RND_P_PRIMARY = 0.5
RND_P_THIRD = 0.6
RND_SEEDS = (1, 2, 3, 4, 5, 6)


def _syn_specs():
    """Static primary / static third_in / sic third_out (goal 24, parity-neutral)."""
    return {
        'primary': {'weight': 1.0 * u.nS, 'delay': DT * u.ms, 'receptor_type': 1},
        'third_in': {'weight': 1.0, 'delay': DT * u.ms},
        'third_out': {'synapse': sic_connection(weight=1.0, delay_steps=10)},
    }


def _edges(proj):
    """Unique (source, target) population-local pairs of a realized projection."""
    if proj is None:
        return set()
    pe = proj.realized_edges()
    s = np.asarray(pe.source); t = np.asarray(pe.target)
    return set(zip(s.tolist(), t.tolist()))


def _pool_sizes_by_target(a2n_edges):
    """Map each target -> number of distinct source astrocytes feeding it (a2n)."""
    by_t = {}
    for s, t in a2n_edges:
        by_t.setdefault(t, set()).add(s)
    return {t: len(srcs) for t, srcs in by_t.items()}


# --- brainpy builders --------------------------------------------------------------

def _bp_tripartite(n_pre, n_post, n_third, *, conn_spec, p_third, pool_size,
                   pool_type, seed):
    """Build a brainpy tripartite net; return (n2n, n2a, a2n) unique edge sets."""
    sim = Simulator(dt=DT * u.ms)
    pre = sim.create(aeif_cond_alpha_astro, n_pre)
    post = sim.create(aeif_cond_alpha_astro, n_post)
    astro = sim.create(astrocyte_lr_1994, n_third)
    prim, tin, tout = sim.tripartite_connect(
        pre, post, astro, conn_spec=conn_spec,
        third_factor_conn_spec=third_factor_bernoulli_with_pool(
            p=p_third, pool_size=pool_size, pool_type=pool_type),
        syn_specs=_syn_specs(), seed=seed)
    return _edges(prim), _edges(tin), _edges(tout)


# --- NEST builders -----------------------------------------------------------------

def _nest_edges(conn, src0, tgt0):
    """Live-NEST GetConnections -> a set of 0-indexed (source, target) local pairs."""
    d = conn.get(['source', 'target'])
    s = np.atleast_1d(np.asarray(d['source']))
    t = np.atleast_1d(np.asarray(d['target']))
    if s.size == 0:
        return set()
    return set(zip((s - src0).tolist(), (t - tgt0).tolist()))


def _nest_tripartite(n_pre, n_post, n_third, *, primary_conn, p_third, pool_size,
                     pool_type, seed):
    """Build the same net in live NEST; return (n2n, n2a, a2n) unique edge sets."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT, 'rng_seed': int(seed)})
    pre = nest.Create('aeif_cond_alpha_astro', n_pre)
    post = nest.Create('aeif_cond_alpha_astro', n_post)
    astro = nest.Create('astrocyte_lr_1994', n_third)
    nest.TripartiteConnect(
        pre, post, astro,
        conn_spec=primary_conn,
        third_factor_conn_spec={'rule': 'third_factor_bernoulli_with_pool',
                                'p': p_third, 'pool_size': pool_size,
                                'pool_type': pool_type},
        syn_specs={'primary': {'synapse_model': 'static_synapse'},
                   'third_in': {'synapse_model': 'static_synapse'},
                   'third_out': {'synapse_model': 'sic_connection'}})
    p0, q0, a0 = pre.tolist()[0], post.tolist()[0], astro.tolist()[0]
    n2n = _nest_edges(nest.GetConnections(pre, post), p0, q0)
    n2a = _nest_edges(nest.GetConnections(pre, astro), p0, a0)
    a2n = _nest_edges(nest.GetConnections(astro, post), a0, q0)
    return n2n, n2a, a2n


# --- Pillar 1: exact block ---------------------------------------------------------

class TestTripartiteBlockLaw(unittest.TestCase):
    """The deterministic block connectivity matches its closed form (no NEST)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_block_edge_sets_are_analytic(self):
        """p=1 block, pool_size=1: n2n=all pairs, a2n={(j//f, j)}, n2a={(i, j//f)}."""
        n2n, n2a, a2n = _bp_tripartite(
            BLK_PRE, BLK_POST, BLK_THIRD, conn_spec=pairwise_bernoulli(1.0),
            p_third=1.0, pool_size=1, pool_type='block', seed=0)
        f = BLK_FACTOR
        self.assertEqual(n2n, {(i, j) for i in range(BLK_PRE) for j in range(BLK_POST)})
        self.assertEqual(a2n, {(j // f, j) for j in range(BLK_POST)})
        self.assertEqual(n2a, {(i, j // f) for i in range(BLK_PRE) for j in range(BLK_POST)})

    def test_block_pool_invariant(self):
        """Each target draws from a pool of exactly one astrocyte (pool_size=1)."""
        _n2n, _n2a, a2n = _bp_tripartite(
            BLK_PRE, BLK_POST, BLK_THIRD, conn_spec=pairwise_bernoulli(1.0),
            p_third=1.0, pool_size=1, pool_type='block', seed=0)
        sizes = _pool_sizes_by_target(a2n)
        self.assertEqual(set(sizes.keys()), set(range(BLK_POST)))
        self.assertTrue(all(v == 1 for v in sizes.values()))


@requires_nest
class TestTripartiteBlockParity(unittest.TestCase):
    """The exact block edge sets are bit-identical to live NEST (design-A arbiter)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_block_edges_match_nest(self):
        """n2n, n2a, a2n edge sets are identical between brainpy and NEST.

        The block config (p_primary=p_third=1) is fully deterministic, so the seed
        is irrelevant to the result; ``seed=1`` satisfies NEST's ``rng_seed`` range
        constraint ``(0, 2^32-1)`` (NEST rejects 0, brainpy accepts it).
        """
        b_n2n, b_n2a, b_a2n = _bp_tripartite(
            BLK_PRE, BLK_POST, BLK_THIRD, conn_spec=pairwise_bernoulli(1.0),
            p_third=1.0, pool_size=1, pool_type='block', seed=1)
        n_n2n, n_n2a, n_a2n = _nest_tripartite(
            BLK_PRE, BLK_POST, BLK_THIRD,
            primary_conn={'rule': 'pairwise_bernoulli', 'p': 1.0},
            p_third=1.0, pool_size=1, pool_type='block', seed=1)
        self.assertEqual(b_n2n, n_n2n, 'primary n2n edge set must match NEST exactly')
        self.assertEqual(b_a2n, n_a2n, 'third_out a2n edge set must match NEST exactly')
        self.assertEqual(b_n2a, n_n2a, 'third_in n2a edge set must match NEST exactly')


# --- Pillar 2: distributional random -----------------------------------------------

class TestTripartiteRandomLaw(unittest.TestCase):
    """Random-pool structural invariants + reproducibility (no NEST)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_random_pool_invariant(self):
        """Each target's distinct astrocytes never exceed pool_size."""
        _n2n, _n2a, a2n = _bp_tripartite(
            RND_PRE, RND_POST, RND_THIRD, conn_spec=pairwise_bernoulli(RND_P_PRIMARY),
            p_third=RND_P_THIRD, pool_size=RND_POOL, pool_type='random', seed=1)
        sizes = _pool_sizes_by_target(a2n)
        self.assertTrue(sizes, 'some targets must receive astrocyte input')
        self.assertTrue(all(v <= RND_POOL for v in sizes.values()),
                        f'distinct astrocytes per target must be <= {RND_POOL}')

    def test_third_in_and_out_share_astrocytes(self):
        """The astrocytes used by third_in (as targets) and third_out (as sources) match.

        Each paired primary edge selects ONE astrocyte used for both arms, so the
        set of astrocytes appearing as n2a targets equals the set appearing as a2n
        sources.
        """
        _n2n, n2a, a2n = _bp_tripartite(
            RND_PRE, RND_POST, RND_THIRD, conn_spec=pairwise_bernoulli(RND_P_PRIMARY),
            p_third=RND_P_THIRD, pool_size=RND_POOL, pool_type='random', seed=2)
        astro_in = {t for _s, t in n2a}
        astro_out = {s for s, _t in a2n}
        self.assertEqual(astro_in, astro_out)

    def test_seeded_reproducible(self):
        """Same seed -> identical edge sets; different seed -> generally different."""
        a = _bp_tripartite(RND_PRE, RND_POST, RND_THIRD,
                           conn_spec=pairwise_bernoulli(RND_P_PRIMARY),
                           p_third=RND_P_THIRD, pool_size=RND_POOL,
                           pool_type='random', seed=7)
        b = _bp_tripartite(RND_PRE, RND_POST, RND_THIRD,
                           conn_spec=pairwise_bernoulli(RND_P_PRIMARY),
                           p_third=RND_P_THIRD, pool_size=RND_POOL,
                           pool_type='random', seed=7)
        c = _bp_tripartite(RND_PRE, RND_POST, RND_THIRD,
                           conn_spec=pairwise_bernoulli(RND_P_PRIMARY),
                           p_third=RND_P_THIRD, pool_size=RND_POOL,
                           pool_type='random', seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a[2], c[2])


@requires_nest
class TestTripartiteRandomParity(unittest.TestCase):
    """Seed-mean distinct-edge counts match live NEST within category D (5 %)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_seed_mean_edge_counts_match_nest(self):
        """Distinct n2n / n2a / a2n counts (seed-mean) land within CAT_D of NEST."""
        bp_counts = {'n2n': [], 'n2a': [], 'a2n': []}
        nest_counts = {'n2n': [], 'n2a': [], 'a2n': []}
        for s in RND_SEEDS:
            b = _bp_tripartite(RND_PRE, RND_POST, RND_THIRD,
                               conn_spec=pairwise_bernoulli(RND_P_PRIMARY),
                               p_third=RND_P_THIRD, pool_size=RND_POOL,
                               pool_type='random', seed=s)
            n = _nest_tripartite(RND_PRE, RND_POST, RND_THIRD,
                                 primary_conn={'rule': 'pairwise_bernoulli', 'p': RND_P_PRIMARY},
                                 p_third=RND_P_THIRD, pool_size=RND_POOL,
                                 pool_type='random', seed=s)
            for key, idx in (('n2n', 0), ('n2a', 1), ('a2n', 2)):
                bp_counts[key].append(len(b[idx]))
                nest_counts[key].append(len(n[idx]))
            # Hard invariant must hold on the NEST side too.
            sizes = _pool_sizes_by_target(n[2])
            self.assertTrue(all(v <= RND_POOL for v in sizes.values()),
                            'NEST distinct astrocytes per target must be <= pool_size')
        for key in ('n2n', 'n2a', 'a2n'):
            self.assertGreater(float(np.mean(nest_counts[key])), 0.0)
            compare_distributional(nest_counts[key], bp_counts[key], tol=CAT_D,
                                   metric=f'{key} distinct-edge count').assert_()


if __name__ == '__main__':
    unittest.main()
