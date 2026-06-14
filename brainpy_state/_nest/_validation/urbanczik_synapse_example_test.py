# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Learning smoke test for ``examples/nest/urbanczik_synapse_example.py``.

The demo trains plastic ``urbanczik_synapse`` edges onto the dendrite of a
``pp_cond_exp_mc_urbanczik`` neuron whose soma is driven by a conductance teacher
(cluster-21). This exercises the rebuilt rule end-to-end on the ``Simulator`` API
-- including the potentiating (positive-``delta_Pi``) branch that the silent-soma
live-NEST parity test cannot reach -- and asserts the Urbanczik-Senn outcome:

* the soma fires (the teacher's supervised target is present),
* dendritic weights adapt **bidirectionally** from the initial weight (some edges
  potentiate, some depress), and
* the dendritic prediction improves -- ``RMS|U_M - V_W*|`` and the rate prediction
  error ``|phi(U) - phi(V_W*)|`` both shrink from the first to the last fifth of
  the driven window.

A small, fixed-seed configuration keeps it deterministic and fast; thresholds
carry margin over the measured values (RMS ratio ~0.71, rate-error ratio ~0.87,
final weights straddling the 90 pA init).

**float64 is required.** Unlike the silent-soma parity regime, the *driven* soma
here is stiff and diverges in float32 (``V_s`` blows up). ``brainpy_state`` traces
some module-level kernels at import; under pytest that import happens during
collection, before x64 is enabled, so those kernels get cached in float32 and a
fresh simulation would reuse them. ``setUpClass`` enables x64/precision and evicts
the JAX cache so the simulation re-traces in float64.
"""
import unittest

import brainstate
import jax

jax.config.update("jax_enable_x64", True)
brainstate.environ.set(precision=64, platform="cpu")

from examples.nest.urbanczik_synapse_example import INIT_W, run


class TestUrbanczikExampleLearns(unittest.TestCase):
    """End-to-end learning smoke test (no NEST required)."""

    @classmethod
    def setUpClass(cls):
        # Evict any float32 kernels cached when brainpy_state was imported during
        # collection (before x64 was on); the driven-soma neuron needs float64 to
        # stay stable. See the module docstring.
        jax.clear_caches()
        # small fixed-seed run: deterministic and fast, yet clearly learns.
        cls.res = run(n_pg=40, n_pattern_rep=12, seed=1)

    def test_soma_fires_as_teacher_target(self):
        # the conductance teacher must make the soma spike -- that is the supervised
        # signal the dendrite learns to predict (and the source of potentiation).
        self.assertGreater(self.res["soma_spikes"], 0,
                           "soma must fire under the conductance teacher")

    def test_weights_adapt_bidirectionally(self):
        # delta_Pi has both signs across edges/time -> weights move both ways from init.
        wf = self.res["weights"][-1]
        self.assertGreater(float(wf.max()), INIT_W, "some dendritic edges must potentiate")
        self.assertLess(float(wf.min()), INIT_W, "some dendritic edges must depress")

    def test_dendritic_prediction_improves(self):
        # the headline Urbanczik-Senn result: V_W* tracks the somatic matching
        # potential U_M better over training (RMS error shrinks; measured ratio ~0.71).
        self.assertLess(self.res["rms_last"], 0.9 * self.res["rms_first"],
                        "RMS|U_M - V_W*| must decrease over training")

    def test_rate_prediction_error_decreases(self):
        # the quantity the rule minimizes: |phi(U) - phi(V_W*)| over the driven window.
        self.assertLess(self.res["rate_err_last"], self.res["rate_err_first"],
                        "rate prediction error |phi(U) - phi(V_W*)| must decrease")


if __name__ == "__main__":
    unittest.main()
