# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Blocked-demo marker for ``examples/nest/urbanczik_synapse_example.py``.

NEST's Urbanczik-Senn demo trains plastic ``urbanczik_synapse`` inputs onto the
*dendritic* compartment of a two-compartment ``pp_cond_exp_mc_urbanczik`` neuron
to predict a somatically-imposed firing rate. The ``Simulator`` API cannot host
it yet: ``urbanczik_synapse`` still extends the legacy ``NESTSynapse`` base, and
the plastic post-state reader (``VoltageCoupledPlasticProj``) exposes only the
somatic ``V`` — not a named dendritic compartment plus the dendritic prediction
error — while ``pp_cond_exp_mc_urbanczik`` is itself an unvalidated
multi-compartment point-process neuron (synapses-plasticity-gap.md §3,
neurons-gap.md §3, examples-gap.md §3.3). This module verifies the placeholder
declares the block and skips the (currently impossible) live-NEST parity until
the dendritic-compartment reader lands and the post neuron is validated.
"""
import unittest


class TestUrbanczikExampleBlocked(unittest.TestCase):
    def test_placeholder_declares_block(self):
        from examples.nest.urbanczik_synapse_example import main, BLOCKED_REASON
        with self.assertRaises(NotImplementedError) as ctx:
            main()
        self.assertIn("synapses-plasticity-gap.md", str(ctx.exception))
        self.assertIn("VoltageCoupledPlasticProj", BLOCKED_REASON)
        self.assertIn("dendritic", BLOCKED_REASON)

    def test_nest_parity_blocked(self):
        self.skipTest(
            "blocked on the Urbanczik dendritic seam (synapses-plasticity-gap.md "
            "§3, neurons-gap.md §3): urbanczik_synapse is on the legacy base and "
            "VoltageCoupledPlasticProj reads only the somatic V, and the "
            "pp_cond_exp_mc_urbanczik multi-compartment point-process post is "
            "unvalidated on the Simulator API")


if __name__ == "__main__":
    unittest.main()
