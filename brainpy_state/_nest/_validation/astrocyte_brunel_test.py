# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Marker tests: ``astrocyte_brunel_{bernoulli,fixed_indegree}`` blocked placeholders (17b).

Both NEST Brunel-with-astrocytes demos wire the excitatory population to neurons +
astrocytes with ``nest.TripartiteConnect(...)`` and the
``third_factor_bernoulli_with_pool`` astrocyte-pool rule (``pool_size=10``,
``pool_type='random'``); they differ only in the *primary* neuron-neuron rule
(``pairwise_bernoulli`` vs ``fixed_indegree``). The Simulator API has no
astrocyte-pool connectivity rule yet (no new connectivity rule this cluster --
cluster-15d spec §7), so both ship as ``NotImplementedError`` placeholders. The
per-edge SIC physics is validated (15d; ``astrocyte_single``/``astrocyte_interaction``).
"""
import unittest


class TestAstrocyteBrunelBlocked(unittest.TestCase):
    """Both Brunel placeholders declare the same astrocyte-pool blocker."""

    def test_bernoulli_placeholder_declares_block(self):
        from examples.nest.astrocyte_brunel_bernoulli import main, BLOCKED_REASON
        with self.assertRaises(NotImplementedError) as ctx:
            main()
        self.assertIn("TripartiteConnect", str(ctx.exception))
        self.assertIn("TripartiteConnect", BLOCKED_REASON)
        self.assertIn("third_factor_bernoulli_with_pool", BLOCKED_REASON)
        self.assertIn("network-api-gap.md", BLOCKED_REASON)

    def test_fixed_indegree_placeholder_declares_block(self):
        from examples.nest.astrocyte_brunel_fixed_indegree import main, BLOCKED_REASON
        with self.assertRaises(NotImplementedError) as ctx:
            main()
        self.assertIn("TripartiteConnect", str(ctx.exception))
        self.assertIn("TripartiteConnect", BLOCKED_REASON)
        self.assertIn("third_factor_bernoulli_with_pool", BLOCKED_REASON)
        self.assertIn("network-api-gap.md", BLOCKED_REASON)

    def test_nest_parity_blocked(self):
        self.skipTest(
            "blocked on nest.TripartiteConnect / third_factor_bernoulli_with_pool "
            "astrocyte-pool connectivity (pool_size=10, pool_type='random') -- no "
            "equivalent Simulator rule yet; see network-api-gap.md, examples-gap.md §3.8")


if __name__ == '__main__':
    unittest.main()
