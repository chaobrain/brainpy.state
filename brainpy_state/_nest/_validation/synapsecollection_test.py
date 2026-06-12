# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Blocked-demo marker for ``examples/nest/synapsecollection.py``.

NEST's ``synapsecollection`` demo is entirely about the ``SynapseCollection``
object returned by ``GetConnections`` (read/set per-edge source/target/weight),
and additionally uses named connection rules and ``Parameter`` weight
expressions — none of which ``brainpy.state`` exposes (network-api-gap.md §3.8,
§3.1, §3.9, §3.11). This module verifies the placeholder declares the block and
skips the (currently impossible) live-NEST parity until a ``nest_compat`` facade
lands.
"""
import unittest


class TestSynapseCollectionBlocked(unittest.TestCase):
    def test_placeholder_declares_block(self):
        from examples.nest.synapsecollection import main, BLOCKED_REASON
        with self.assertRaises(NotImplementedError) as ctx:
            main()
        self.assertIn("network-api-gap.md", str(ctx.exception))
        self.assertIn("SynapseCollection", BLOCKED_REASON)

    def test_nest_parity_blocked(self):
        self.skipTest(
            "blocked on SynapseCollection (network-api-gap.md §3.8): GetConnections "
            "introspection, named connection rules, and Parameter weight expressions "
            "are all absent")


if __name__ == "__main__":
    unittest.main()
