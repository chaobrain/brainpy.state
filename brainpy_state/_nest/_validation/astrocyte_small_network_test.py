# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Marker test: ``astrocyte_small_network`` is a documented blocked placeholder (17b).

NEST's ``astrocyte_small_network`` wires neurons and astrocytes with
``nest.TripartiteConnect(...)`` and the ``third_factor_bernoulli_with_pool``
astrocyte-pool rule (``pool_size=1``, ``pool_type='block'``). The Simulator API has
no astrocyte-pool connectivity rule yet (no new connectivity rule this cluster --
cluster-15d spec §7), so the demo ships as a ``NotImplementedError`` placeholder.
The per-edge SIC loop physics it relies on is already validated (15d,
``astrocyte_sic_test.py``; demos ``astrocyte_single``/``astrocyte_interaction``).
"""
import unittest


class TestAstrocyteSmallNetworkBlocked(unittest.TestCase):
    """The placeholder declares its blocker and refuses to run."""

    def test_placeholder_declares_block(self):
        from examples.nest.astrocyte_small_network import main, BLOCKED_REASON
        with self.assertRaises(NotImplementedError) as ctx:
            main()
        # main() raises the recorded reason verbatim.
        self.assertIn("TripartiteConnect", str(ctx.exception))
        self.assertIn("TripartiteConnect", BLOCKED_REASON)
        self.assertIn("third_factor_bernoulli_with_pool", BLOCKED_REASON)
        self.assertIn("network-api-gap.md", BLOCKED_REASON)

    def test_nest_parity_blocked(self):
        self.skipTest(
            "blocked on nest.TripartiteConnect / third_factor_bernoulli_with_pool "
            "astrocyte-pool connectivity (pool_size=1, pool_type='block') -- no "
            "equivalent Simulator rule yet; see network-api-gap.md, examples-gap.md §3.8")


if __name__ == '__main__':
    unittest.main()
