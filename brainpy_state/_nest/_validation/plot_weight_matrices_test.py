# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Blocked-demo marker for ``examples/nest/plot_weight_matrices.py``.

NEST's ``plot_weight_matrices`` cannot yet be ported: it requires post-hoc
connection-weight introspection (``GetConnections`` / ``SynapseCollection``),
which ``brainpy.state`` does not expose (network-api-gap.md §3.1, §3.8). This
module verifies the placeholder declares the block and skips the (currently
impossible) live-NEST parity until a ``nest_compat`` facade lands.
"""
import unittest


class TestPlotWeightMatricesBlocked(unittest.TestCase):
    def test_placeholder_declares_block(self):
        from examples.nest.plot_weight_matrices import main, BLOCKED_REASON
        with self.assertRaises(NotImplementedError) as ctx:
            main()
        self.assertIn("network-api-gap.md", str(ctx.exception))
        self.assertIn("GetConnections", BLOCKED_REASON)

    def test_nest_parity_blocked(self):
        self.skipTest(
            "blocked on GetConnections (network-api-gap.md §3.1): connection-weight "
            "introspection absent — no SynapseCollection to enumerate the realized "
            "synapses between two populations and read per-edge weights")


if __name__ == "__main__":
    unittest.main()
