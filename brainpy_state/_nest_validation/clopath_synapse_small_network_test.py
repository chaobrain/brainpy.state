# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for the ``clopath_synapse_small_network`` example (§3.3 demo).

The example builds a small all-to-all recurrent ``aeif_psc_delta_clopath``
population (no autapses) on the ``Simulator`` API and clamps the neurons to fire
in a fixed staggered order (``0 -> 1 -> 2``) every cycle, so each directed
recurrent edge sees the canonical Clopath spike-pairing protocol: forward edges
(``pre`` fires before ``post``, i.e. ``i < j``) potentiate, backward edges
(``i > j``) depress. It records the recurrent weight matrix with
``res.weight_trace``.

Per edge this is exactly the cluster-07 pairing protocol, so this test reuses the
shared drive :mod:`brainpy_state._nest_validation._clopath_drive` (neuron
parameters, ``INIT_W``, ``DRIVE_W``, ``RELAY_D``, ``DELAY_UBARS``) and the same
frozen clopath weight band the synapse-level ``clopath_synapse_parity_test.py``
proved. The NEST reference rebuilds the identical network and reads every final
recurrent weight from ``GetConnections``; the assertions are:

* every directed edge's final weight matches NEST within the documented clopath
  band (``atol 2e-3`` mV, ``rtol 5 %``) -- online instantaneous-read vs NEST
  deferred-history (backward/LTD edges near-exact, forward/LTP within 5 %);
* each clearly-directional edge has NEST's sign (neutral at the LTD/LTP
  crossover, where a backward edge's cross-cycle LTP nearly cancels its
  within-cycle LTD);
* the structural claim the demo exists to show -- forward edges potentiate,
  backward edges depress -- holds on **both** the NEST and ``Simulator`` sides.

The weight is a bare mV mantissa (``aeif_psc_delta_clopath`` is a delta neuron),
init ``0.5``. The drive is fully deterministic (spike-clamped, no PRNG), so the
weights are reproducible run-to-run.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc
from brainpy_state._nest_validation import _clopath_drive as drv

# Documented Clopath stored-weight band (see _clopath_drive module docstring):
# online instantaneous-read vs NEST deferred-history + one-step event-delivery lag.
_WEIGHT_BAND = tc.TraceTolerance(2e-3, 5e-2, label="clopath",
                                 note="online instantaneous-read vs NEST deferred history")
# Net |Δw| below this is the LTD/LTP crossover (its strict sign is meaningless): a
# backward edge whose cross-cycle potentiation nearly cancels its within-cycle
# depression sits here.
_CROSSOVER_EPS = 3e-3


def _nest_final_weights(trains, t_sim):
    """Final recurrent Clopath weights of the reproduced network, in live NEST.

    Builds the same network as the example -- ``len(trains)``
    ``aeif_psc_delta_clopath`` neurons (canonical :data:`_clopath_drive.NRN_PARAMS`),
    all-to-all recurrent ``clopath_synapse`` (no autapses, init
    :data:`_clopath_drive.INIT_W`), each neuron clamped by an 80 mV
    ``static_synapse`` from its own ``spike_generator`` train -- and reads every
    stored recurrent weight from ``GetConnections`` after the run.

    Parameters
    ----------
    trains : sequence of sequence of float
        Per-neuron clamp spike times (ms); ``trains[i]`` drives neuron ``i``.
    t_sim : float
        Simulation horizon (ms).

    Returns
    -------
    dict of tuple to float
        ``(pre_local, post_local) -> final weight`` (mV) for every directed edge,
        with local indices ``0 .. n-1`` matching the example's neuron order.
    """
    import nest
    nest.ResetKernel()
    nest.resolution = drv.DT
    nest.set_verbosity("M_ERROR")
    n = len(trains)
    pe = nest.Create("aeif_psc_delta_clopath", n, params=drv.NRN_PARAMS)
    nest.Connect(pe, pe,
                 conn_spec={"rule": "all_to_all", "allow_autapses": False},
                 syn_spec={"synapse_model": "clopath_synapse",
                           "weight": drv.INIT_W, "delay": drv.RELAY_D})
    for i in range(n):
        sg = nest.Create("spike_generator", 1, {"spike_times": list(trains[i])})
        nest.Connect(sg, pe[i], syn_spec={"synapse_model": "static_synapse",
                                          "weight": drv.DRIVE_W, "delay": drv.RELAY_D})
    nest.Simulate(t_sim)
    base = pe[0].global_id
    conns = nest.GetConnections(pe, pe)
    st = conns.get(["source", "target", "weight"])
    src = np.atleast_1d(st["source"])
    tgt = np.atleast_1d(st["target"])
    wgt = np.atleast_1d(st["weight"])
    return {(int(s - base), int(t - base)): float(w)
            for s, t, w in zip(src, tgt, wgt)}


@requires_nest
class TestClopathSmallNetworkExampleParity(unittest.TestCase):
    """Live-NEST parity for the small-network example's recurrent weight matrix."""

    @classmethod
    def setUpClass(cls):
        from examples.nest_like.clopath_synapse_small_network import run, TRAINS, T_SIM
        brainstate.environ.set(dt=drv.DT * u.ms)
        cls.times, cls.weights, cls.edges = run()
        cls.our_final = {e: float(cls.weights[-1, c]) for c, e in enumerate(cls.edges)}
        cls.nest_final = _nest_final_weights(TRAINS, T_SIM)

    # -- every directed edge's final weight tracks NEST within the band ----
    def test_all_edges_within_band(self):
        for e in self.edges:
            with self.subTest(edge=e):
                compare_trace(self.nest_final[e], self.our_final[e],
                              tol=_WEIGHT_BAND, metric=f"clopath edge {e}").assert_()

    # -- every clearly-directional edge has NEST's sign --------------------
    def test_direction_matches_nest(self):
        for e in self.edges:
            with self.subTest(edge=e):
                nest_dw = self.nest_final[e] - drv.INIT_W
                our_dw = self.our_final[e] - drv.INIT_W
                if abs(nest_dw) <= _CROSSOVER_EPS:
                    self.assertLessEqual(
                        abs(our_dw), _CROSSOVER_EPS,
                        f"{e}: NEST neutral (Δw={nest_dw:.2e}) but ours Δw={our_dw:.2e}")
                else:
                    self.assertEqual(
                        int(np.sign(our_dw)), int(np.sign(nest_dw)),
                        f"{e}: net Δw sign must match NEST "
                        f"(nest {self.nest_final[e]:.6f}, ours {self.our_final[e]:.6f})")

    # -- the demo's structural claim holds on the NEST side too ------------
    def test_forward_potentiates_backward_depresses_in_nest(self):
        # firing order 0->1->2 => edge (i->j) is pre-before-post (LTP) iff i<j,
        # post-before-pre (LTD) iff i>j. NEST must show exactly that structure.
        for (i, j), w in self.nest_final.items():
            with self.subTest(edge=(i, j)):
                if j > i:
                    self.assertGreater(w, drv.INIT_W,
                                       f"NEST forward edge {(i, j)} must potentiate")
                else:
                    self.assertLess(w, drv.INIT_W,
                                    f"NEST backward edge {(i, j)} must depress")


class TestClopathSmallNetworkExampleBehavior(unittest.TestCase):
    """Example headline, provable without a live NEST install."""

    def setUp(self):
        brainstate.environ.set(dt=drv.DT * u.ms)

    def test_forward_potentiates_backward_depresses(self):
        # the demo exists to show directional structure emerging: with the
        # 0->1->2 firing order, forward edges (i<j) potentiate above the 0.5
        # baseline and backward edges (i>j) depress below it.
        from examples.nest_like.clopath_synapse_small_network import run
        _, weights, edges = run()
        final = weights[-1]
        for c, (i, j) in enumerate(edges):
            with self.subTest(edge=(i, j)):
                if j > i:
                    self.assertGreater(final[c], drv.INIT_W,
                                       f"forward edge {(i, j)} must potentiate")
                else:
                    self.assertLess(final[c], drv.INIT_W,
                                    f"backward edge {(i, j)} must depress")

    def test_weight_trace_records_evolution(self):
        # the recorded matrix has one column per directed edge, starts at INIT_W,
        # and actually moves (the rule is doing something).
        from examples.nest_like.clopath_synapse_small_network import run, N
        times, weights, edges = run()
        self.assertEqual(len(edges), N * (N - 1))
        self.assertEqual(weights.shape[1], N * (N - 1))
        self.assertEqual(weights.shape[0], times.shape[0])
        np.testing.assert_allclose(weights[0], drv.INIT_W, atol=1e-9)
        self.assertGreater(float(np.abs(weights[-1] - weights[0]).max()), 1e-3)

    def test_weight_matrix_assembler(self):
        # weight_matrix scatters the per-edge vector into an n x n matrix with a
        # NaN (absent) diagonal -- there are no autapses.
        from examples.nest_like.clopath_synapse_small_network import run, weight_matrix, N
        _, weights, edges = run()
        M = weight_matrix(weights[-1], edges)
        self.assertEqual(M.shape, (N, N))
        self.assertTrue(np.all(np.isnan(np.diag(M))))
        for c, (i, j) in enumerate(edges):
            self.assertAlmostEqual(M[i, j], float(weights[-1][c]), places=9)


if __name__ == "__main__":
    unittest.main()
