# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for the ``evaluate_tsodyks2_synapse`` example (§3.3 demo).

The example drives a single ``tsodyks2_synapse`` edge with a 50 ms-ISI burst plus a
recovery pair onto a linear ``iaf_psc_exp`` post (``V_th = 1e4`` mV); the post V_m is
the PSC-amplitude train. This test reproduces the train in live NEST through the
cluster-01 routing (``spike_generator -> parrot_neuron -> tsodyks2_synapse ->
iaf_psc_exp`` -- a device cannot drive a plastic synapse) and compares the post V_m
in both the depression and facilitation regimes.

``tsodyks2_synapse`` is a deterministic analytic-propagator STP rule, so the trains
match within category B. The ``Simulator`` ``spike_generator`` relays the train into
the plastic edge with a **one-step (0.1 ms) holder lag**, so the NEST reference's
``parrot`` relay is given the matching ``delay = 0.1`` (the ``RELAY_D`` convention
of the cluster-07 clopath drive) -- otherwise NEST's default-1.0 ms relay would
delivery-shift the whole train by ~8 steps. Once delivery-aligned, only the
multimeter-recorder one-step offset remains (cluster-03 Lesson), and the trains
agree to machine precision; the tolerance keeps a two-step alignment search
(``align_steps=2``) to absorb that recorder step.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

# Linear, never-spiking post; tau_syn_ex == tau_psc so the PSC shape is the synapse's.
_NPAR = dict(C_m=250., tau_m=20., tau_syn_ex=3.0, tau_syn_in=3.0,
             t_ref=2., E_L=0., V_reset=0., V_m=0., V_th=1e4)
_D1 = 0.1   # parrot relay delay == Simulator spike_generator holder lag (RELAY_D)
_D2 = 1.0   # parrot -> synapse axonal delay (ms); == tsodyks2_synapse default delay
# tsodyks2 is deterministic (category B); with the relay delay matched to the
# holder lag only the multimeter recorder step remains (cluster-03), so allow a
# 2-step alignment.
_BAND = tc.TraceTolerance(5e-2 * u.mV, 1e-3, align_steps=2, label="B",
                          note="tsodyks2 PSC-amplitude train, generator+recorder alignment")


def _nest_vm(regime, weight):
    """Post V_m for one regime via spike_generator -> parrot -> tsodyks2 -> iaf."""
    from examples.nest_like.evaluate_tsodyks2_synapse import TRAIN, T_SIM, REGIMES, DT
    p = REGIMES[regime]
    nest.ResetKernel()
    nest.resolution = DT
    nest.set_verbosity("M_ERROR")
    nrn = nest.Create("iaf_psc_exp", 1, params=_NPAR)
    pn = nest.Create("parrot_neuron")
    sg = nest.Create("spike_generator", params={"spike_times": list(TRAIN)})
    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": DT})
    nest.Connect(sg, pn, syn_spec={"delay": _D1})
    nest.Connect(pn, nrn, syn_spec={"synapse_model": "tsodyks2_synapse", "delay": _D2,
                                    "weight": weight, "U": p["U"], "u": p["u"],
                                    "x": p["x"], "tau_rec": p["tau_rec"],
                                    "tau_fac": p["tau_fac"]})
    nest.Connect(mm, nrn)
    nest.Simulate(T_SIM)
    return np.asarray(mm.get("events")["V_m"])


@requires_nest
class TestTsodyks2Example(unittest.TestCase):
    """Live-NEST parity for the tsodyks2 example's depression / facilitation trains."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _run(self, regime):
        from examples.nest_like.evaluate_tsodyks2_synapse import run, WEIGHT
        nest_v = _nest_vm(regime, WEIGHT)
        _, bp_v = run(regime)
        m = min(len(nest_v), len(bp_v))
        compare_trace(nest_v[:m], bp_v[:m], tol=_BAND,
                      metric=f"tsodyks2 {regime} V_m").assert_()

    def test_depression(self):
        self._run("depression")

    def test_facilitation(self):
        self._run("facilitation")

    def test_regimes_are_distinct(self):
        # the example must actually select different dynamics per regime: a
        # depressing burst shrinks, a facilitating burst grows.
        from examples.nest_like.evaluate_tsodyks2_synapse import run, burst_peak_ratio
        _, dep = run("depression")
        _, fac = run("facilitation")
        self.assertLess(burst_peak_ratio(dep), 1.0, "depression burst must shrink")
        self.assertGreater(burst_peak_ratio(fac), 1.0, "facilitation burst must grow")


if __name__ == "__main__":
    unittest.main()
