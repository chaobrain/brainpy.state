# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel-alpha evolution-strategies parity: brainpy.state vs live NEST.

Phase 5 layers a separable Natural Evolution Strategies optimizer (Wierstra et
al. 2014) on top of the Phase-1 Brunel alpha network. Two things are checked:

1. ``simulate(g, eta)`` — the network/analysis path the optimizer evaluates —
   produces a population rate within 5 % of a live NEST run at a fixed, balanced
   operating point (``g=5, eta=2``). This exercises the real ``iaf_psc_alpha``
   network inside the ES harness; the optimizer is only as faithful as the
   objective it samples.
2. ``optimize`` — the pure-NumPy ES core (a verbatim port of the NEST reference)
   ascends the natural gradient to the maximizer of a deterministic analytic
   objective. No NEST needed; this guards the optimizer math itself.

The NEST run is skipped when ``nest`` is unavailable; the optimizer test always
runs.
"""
import unittest

import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

TOL = 0.05

# Fixed, balanced operating point (matches the validated Phase-1 alpha network:
# N=1000, gamma=0.8 -> NE=800/NI=200, CE=80/CI=20).
PARAMS = {
    "seed": 12, "dt": 0.1, "sim_time": 1000.0, "warmup_time": 100.0,
    "delay": 1.5, "g": 5.0, "eta": 2.0, "epsilon": 0.1, "N": 1000,
    "gamma": 0.8, "N_rec": 50,
}


def _nest_rate(parameters):
    import scipy.special as sp

    def LambertWm1(x):
        return sp.lambertw(x, k=-1 if x < 0 else 0).real

    def ComputePSPnorm(tauMem, CMem, tauSyn):
        a = tauMem / tauSyn
        b = 1.0 / tauSyn - 1.0 / tauMem
        t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)
        return (np.exp(1.0) / (tauSyn * CMem * b)
                * ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b
                   - t_max * np.exp(-t_max / tauSyn)))

    NE = int(parameters["gamma"] * parameters["N"])
    NI = parameters["N"] - NE
    CE = int(parameters["epsilon"] * NE)
    CI = int(parameters["epsilon"] * NI)
    tauSyn, tauMem, CMem, theta = 0.5, 20.0, 250.0, 20.0
    neuron_parameters = {"C_m": CMem, "tau_m": tauMem, "tau_syn_ex": tauSyn,
                         "tau_syn_in": tauSyn, "t_ref": 2.0, "E_L": 0.0,
                         "V_reset": 0.0, "V_m": 0.0, "V_th": theta}
    J = 0.1
    J_ex = J / ComputePSPnorm(tauMem, CMem, tauSyn)
    J_in = -parameters["g"] * J_ex
    nu_th = (theta * CMem) / (J_ex * CE * np.exp(1) * tauMem * tauSyn)
    p_rate = 1000.0 * (parameters["eta"] * nu_th) * CE

    nest.ResetKernel()
    nest.rng_seed = parameters["seed"]
    nest.resolution = parameters["dt"]
    nodes_ex = nest.Create("iaf_psc_alpha", NE, params=neuron_parameters)
    nodes_in = nest.Create("iaf_psc_alpha", NI, params=neuron_parameters)
    noise = nest.Create("poisson_generator", params={"rate": p_rate})
    espikes = nest.Create("spike_recorder")
    nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex, "delay": parameters["delay"]})
    nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": parameters["delay"]})
    nest.Connect(noise, nodes_ex, syn_spec="excitatory")
    nest.Connect(noise, nodes_in, syn_spec="excitatory")
    nest.Connect(nodes_ex[: parameters["N_rec"]], espikes)
    nest.Connect(nodes_ex, nodes_ex + nodes_in, {"rule": "fixed_indegree", "indegree": CE}, "excitatory")
    nest.Connect(nodes_in, nodes_ex + nodes_in, {"rule": "fixed_indegree", "indegree": CI}, "inhibitory")
    nest.Simulate(parameters["sim_time"])

    ev = espikes.events
    times = ev["times"][ev["times"] > parameters["warmup_time"]]
    return 1.0 * len(times) / parameters["N_rec"] / parameters["sim_time"] * 1e3


class TestEvolutionStrategiesOptimizer(unittest.TestCase):
    def test_optimizer_ascends_to_analytic_optimum(self):
        # Verbatim-ported NES must maximize a concave quadratic toward its peak.
        from examples.nest_like.brunel_alpha_evolution_strategies import optimize
        opt = np.array([1.5, 2.5])

        def func(g, eta):
            return -((g - opt[0]) ** 2 + (eta - opt[1]) ** 2)

        np.random.seed(0)
        start = np.array([1.0, 3.0])
        res = optimize(func, start.copy(), np.array([0.3, 0.3]),
                       max_generations=300, record_history=False)
        d_start = np.linalg.norm(start - opt)
        d_end = np.linalg.norm(res["mu"] - opt)
        self.assertLess(d_end, 0.1, f"ES did not converge: |mu-opt|={d_end:.3f}")
        self.assertLess(d_end, d_start)


@unittest.skipUnless(_HAS_NEST, "live NEST not importable")
class TestBrunelAlphaESNetworkParity(unittest.TestCase):
    def test_simulate_rate_within_5pct_of_nest(self):
        from examples.nest_like.brunel_alpha_evolution_strategies import (
            simulate, cut_warmup_time, compute_rate, compute_cv, sort_spikes,
        )
        espikes, ispikes = simulate(dict(PARAMS))
        espikes = cut_warmup_time(espikes, PARAMS["warmup_time"])
        bp_rate = compute_rate(espikes, PARAMS["N_rec"], PARAMS["sim_time"])
        bp_cv = compute_cv(sort_spikes(espikes)[1])

        nest_rate = _nest_rate(PARAMS)
        self.assertGreater(nest_rate, 0.0)
        self.assertGreater(bp_rate, 0.0)
        self.assertGreater(bp_cv, 0.0)          # cv path runs and is sane
        rel = abs(bp_rate - nest_rate) / nest_rate
        self.assertLess(rel, TOL,
                        f"exc rate brainpy={bp_rate:.3f} nest={nest_rate:.3f} rel={rel:.3f} > {TOL}")
