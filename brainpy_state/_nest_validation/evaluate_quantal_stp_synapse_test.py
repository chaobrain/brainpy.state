# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for the ``evaluate_quantal_stp_synapse`` example (§3.3 demo).

``quantal_stp_synapse`` releases an integer number of quanta per spike (binomial
recovery of depleted sites, then binomial release of available sites). NEST draws
one Bernoulli per release site while the rebuilt kernel draws a single
``jax.random.binomial`` -- distributionally identical but with *independent* PRNG
streams, so the example's post V_m is **never** compared per-sample to NEST.
Instead several seeds are run per side and the seed-mean post depolarization (mean
``V_m``) is compared -- the category-D protocol (relative tolerance 5 %).

The example forwards its per-run ``seed`` to ``connect``; the runtime release PRNG
is keyed from it (and survives ``simulate``'s ``init_all_states``), so distinct
seeds give independent realizations whose mean tracks the deterministic
``tsodyks2`` limit. Routing reproduces the train in live NEST through the
cluster-01 path (``spike_generator -> parrot_neuron -> quantal_stp_synapse ->
iaf_psc_exp``); the parrot relay delay is set to the ``Simulator``
``spike_generator``'s one-step holder lag (0.1 ms), the synapse delay to the
example's default (1.0 ms).

The NEST reference neutralises two ``set_status`` footguns so its initial state
matches the rebuilt kernel: ``a = n`` (sites start available) and ``u = U`` (NEST
leaves ``u_`` at the *old* 0.5 constructor default when only ``U`` is set, which
otherwise biases the facilitation regime ~4 %; the kernel defaults ``u0 = U``).
The example raises the site count to 100 (``n*w`` held fixed) so the seed-mean is
a tight estimate -- the kernel then matches NEST within category D at 8 seeds.

The behavior tests (ungated) pin the example's headline -- seeds control the
realization, the seed-mean converges to the deterministic limit, and the two
regimes are distinct -- without requiring a live NEST install.
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._nest_validation.nest_compare import compare_distributional, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

try:
    import nest
    _HAS_NEST = True
except Exception:                                         # pragma: no cover - env dependent
    _HAS_NEST = False

# Linear, never-spiking post (matches the example's _post).
_NPAR = dict(C_m=250., tau_m=20., tau_syn_ex=3.0, tau_syn_in=3.0,
             t_ref=2., E_L=0., V_reset=0., V_m=0., V_th=1e4)
_DT = 0.1
_D1 = 0.1   # parrot relay delay == Simulator spike_generator holder lag
_D2 = 1.0   # parrot -> synapse axonal delay == quantal_stp_synapse default delay


def _nest_mean_vm(regime, seed):
    """Seed-mean post V_m via spike_generator -> parrot -> quantal_stp -> iaf."""
    from examples.nest.evaluate_quantal_stp_synapse import (
        TRAIN, T_SIM, REGIMES, WEIGHT, N_SITES)
    p = REGIMES[regime]
    nest.ResetKernel()
    nest.resolution = _DT
    nest.rng_seed = int(seed)
    nest.set_verbosity("M_ERROR")
    neuron = nest.Create("iaf_psc_exp", 1, params=_NPAR)
    pn = nest.Create("parrot_neuron")
    sg = nest.Create("spike_generator", params={"spike_times": list(TRAIN)})
    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": _DT})
    nest.Connect(sg, pn, syn_spec={"delay": _D1})
    # Two NEST set_status footguns must be neutralised to match the rebuilt
    # kernel's defaults (which start at the physically-correct baseline):
    #   * 'a' stays at the constructor default (=1) unless given -> set a = n so
    #     all sites start available (the kernel's ``a`` default).
    #   * 'u' stays at the *old* constructor default (=0.5, from u_(U_) with the
    #     0.5 default U_) unless given -- it is NOT re-derived when 'U' is set --
    #     so set u = U. Without this the first release uses u=0.5 and the
    #     facilitation regime (U=0.15) diverges ~4 % (the kernel defaults u0=U).
    nest.Connect(pn, neuron, syn_spec={"synapse_model": "quantal_stp_synapse",
                                       "delay": _D2, "n": N_SITES, "a": N_SITES,
                                       "u": p["U"], "weight": WEIGHT, "U": p["U"],
                                       "tau_rec": p["tau_rec"], "tau_fac": p["tau_fac"]})
    nest.Connect(mm, neuron)
    nest.Simulate(T_SIM)
    return float(np.mean(np.asarray(mm.get("events")["V_m"])))


@requires_nest
class TestQuantalStpExampleParity(unittest.TestCase):
    """Distributional (category-D) parity of the example's seed-mean V_m."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _run(self, regime, label):
        from examples.nest.evaluate_quantal_stp_synapse import mean_vm, SEEDS
        ref = [_nest_mean_vm(regime, s) for s in SEEDS]
        cand = [mean_vm(regime, s) for s in SEEDS]
        compare_distributional(ref, cand, tol=tc.CAT_D, metric=label).assert_()

    def test_quantal_depression_distribution(self):
        self._run("depression", "quantal dep <V_m>")

    def test_quantal_facilitation_distribution(self):
        self._run("facilitation", "quantal fac <V_m>")


class TestQuantalStpExampleBehavior(unittest.TestCase):
    """Example headline, provable without a live NEST install."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_seed_controls_realization(self):
        # the per-run seed must reach the release PRNG through the Simulator: a
        # fixed seed reproduces exactly, different seeds give different draws.
        from examples.nest.evaluate_quantal_stp_synapse import run
        _, a1 = run("depression", seed=1)
        _, a1b = run("depression", seed=1)
        _, a2 = run("depression", seed=2)
        np.testing.assert_array_equal(a1, a1b)
        self.assertFalse(np.allclose(a1, a2))

    def test_seed_mean_tracks_deterministic_limit(self):
        # the upstream's point: the stochastic seed-mean converges to the
        # deterministic tsodyks2 envelope (weight = n * w).
        from examples.nest.evaluate_quantal_stp_synapse import (
            seed_mean_trace, deterministic_reference, REGIMES)
        for regime in REGIMES:
            _, mean_tr = seed_mean_trace(regime)
            _, det = deterministic_reference(regime)
            q, d = float(mean_tr.mean()), float(det.mean())
            self.assertLess(abs(q - d) / d, 0.10,
                            f"{regime}: seed-mean {q:.3f} vs deterministic {d:.3f}")

    def test_regimes_are_distinct(self):
        # high-U depression releases far more on the first spike than low-U
        # facilitation, so the seed-mean first EPSP peaks differ clearly.
        from examples.nest.evaluate_quantal_stp_synapse import seed_mean_trace
        t, dep = seed_mean_trace("depression")
        _, fac = seed_mean_trace("facilitation")
        win = (t >= 50.0) & (t < 64.0)        # the first EPSP, before spike 2 (65 ms)
        dep_first, fac_first = float(dep[win].max()), float(fac[win].max())
        self.assertGreater(dep_first, 1.3 * fac_first,
                           f"depression first EPSP {dep_first:.3f} not >> "
                           f"facilitation {fac_first:.3f}")


if __name__ == "__main__":
    unittest.main()
