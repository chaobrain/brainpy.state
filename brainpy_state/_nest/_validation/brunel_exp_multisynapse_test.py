# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel multi-receptor firing-rate parity: brainpy.state Simulator vs live NEST.

Drives the faithful ``examples/nest/brunel_exp_multisynapse.py`` port at a small
order and asserts the excitatory rate lands within 5 % of a live NEST run built
from identical parameters (100 receptor ports, uniformly-routed synapses).

Unlike the homogeneous alpha/delta variants, here each neuron's external drive
lands on a single, randomly-drawn receptor port, and the firing rate is a steep
function of that port's time constant (≈0 below tau≈0.5 ms, ≈70 spks/s at the
longest). The per-neuron rate distribution is therefore highly heterogeneous, so
the *recorded* sample must be the full excitatory population (not 50 neurons) for
the population mean to be a low-variance estimator comparable across the two
independent RNG streams. Skipped when ``nest`` is unavailable.
"""
import unittest

import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainunit as u

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest._validation.tolerance_conventions import CAT_D

ORDER = 200          # NE=800, NI=200 -> small/fast
SIMTIME = 1000.0
N_SEEDS = 4          # average rate over independent realizations (variance reduction)


def _nest_rates(order, simtime, seed):
    nest.ResetKernel()
    nest.resolution = 0.1
    nest.rng_seed = seed
    g, eta, epsilon, delay = 5.0, 2.0, 0.1, 1.5
    NE, NI = 4 * order, order
    CE, CI = int(epsilon * NE), int(epsilon * NI)
    tauMem, theta = 20.0, 20.0
    nr_ports = 100
    tau_syn = [0.1 + 0.01 * i for i in range(nr_ports)]
    npar = {"C_m": 1.0, "tau_m": tauMem, "t_ref": 2.0, "E_L": 0.0,
            "V_reset": 0.0, "V_m": 0.0, "V_th": theta, "tau_syn": tau_syn}

    J = 0.1
    J_ex = J
    J_in = -g * J_ex
    nu_th = theta / (J * CE * tauMem)
    p_rate = 1000.0 * eta * nu_th * CE

    ne = nest.Create("iaf_psc_exp_multisynapse", NE, params=npar)
    ni = nest.Create("iaf_psc_exp_multisynapse", NI, params=npar)
    noise = nest.Create("poisson_generator", params={"rate": p_rate})
    esr = nest.Create("spike_recorder")
    syn_ex = {"synapse_model": "static_synapse", "weight": J_ex, "delay": delay,
              "receptor_type": nest.random.uniform_int(max=nr_ports - 1) + 1}
    syn_in = {"synapse_model": "static_synapse", "weight": J_in, "delay": delay,
              "receptor_type": nest.random.uniform_int(max=nr_ports - 1) + 1}
    nest.Connect(noise, ne, syn_spec=syn_ex)
    nest.Connect(noise, ni, syn_spec=syn_ex)
    nest.Connect(ne, esr)                       # record the full exc population
    nest.Connect(ne, ne + ni, {"rule": "fixed_indegree", "indegree": CE}, syn_ex)
    nest.Connect(ni, ne + ni, {"rule": "fixed_indegree", "indegree": CI}, syn_in)
    nest.Simulate(simtime)
    return esr.n_events / simtime * 1000.0 / NE


@requires_nest
class TestBrunelExpMultisynapseParity(unittest.TestCase):
    def test_excitatory_rate_within_5pct_of_nest(self):
        import numpy as np
        from examples.nest.brunel_exp_multisynapse import build
        bp, ns = [], []
        for seed in range(N_SEEDS):
            sim, esr, _isr, _n, _t = build(order=ORDER, simtime=SIMTIME,
                                           n_rec=4 * ORDER, seed=seed)
            res = sim.simulate(SIMTIME * u.ms)
            bp.append(res.rate(esr.segments[0].population))
            ns.append(_nest_rates(ORDER, SIMTIME, seed=seed + 1))
        self.assertGreater(float(np.mean(ns)), 0.0)
        # PRNG-divergent: average per-seed rates, compare the seed-mean (category D).
        compare_distributional(ns, bp, tol=CAT_D, metric="exc rate").assert_()
