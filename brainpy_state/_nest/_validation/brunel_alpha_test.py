# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel-network firing-rate parity: brainpy.state Simulator vs live NEST.

Drives the faithful ``examples/nest/brunel_alpha.py`` port at a small order and
asserts the recorded excitatory rate lands within 5 % of a live NEST run built
from identical parameters. Skipped when ``nest`` is unavailable.
"""
import unittest

import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import saiunit as u

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

ORDER = 200          # NE=800, NI=200 -> dense-feasible, fast
SIMTIME = 1000.0
TOL = 0.05           # 5% mean-rate parity


def _nest_rates(order, simtime):
    import scipy.special as sp
    nest.ResetKernel()
    nest.resolution = 0.1
    g, eta, epsilon, delay = 5.0, 2.0, 0.1, 1.5
    NE, NI = 4 * order, order
    CE, CI = int(epsilon * NE), int(epsilon * NI)
    N_rec = 50
    tauSyn, tauMem, CMem, theta = 0.5, 20.0, 250.0, 20.0
    npar = {"C_m": CMem, "tau_m": tauMem, "tau_syn_ex": tauSyn, "tau_syn_in": tauSyn,
            "t_ref": 2.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": theta}

    def psp(tauMem, CMem, tauSyn):
        a = tauMem / tauSyn
        b = 1.0 / tauSyn - 1.0 / tauMem
        tmax = 1.0 / b * (-sp.lambertw(-np.exp(-1.0 / a) / a, k=-1).real - 1.0 / a)
        return (np.exp(1.0) / (tauSyn * CMem * b)
                * ((np.exp(-tmax / tauMem) - np.exp(-tmax / tauSyn)) / b
                   - tmax * np.exp(-tmax / tauSyn)))

    J_ex = 0.1 / psp(tauMem, CMem, tauSyn)
    J_in = -g * J_ex
    nu_th = (theta * CMem) / (J_ex * CE * np.exp(1) * tauMem * tauSyn)
    p_rate = 1000.0 * eta * nu_th * CE

    ne = nest.Create("iaf_psc_alpha", NE, params=npar)
    ni = nest.Create("iaf_psc_alpha", NI, params=npar)
    noise = nest.Create("poisson_generator", params={"rate": p_rate})
    esr = nest.Create("spike_recorder")
    nest.CopyModel("static_synapse", "exc", {"weight": J_ex, "delay": delay})
    nest.CopyModel("static_synapse", "inh", {"weight": J_in, "delay": delay})
    nest.Connect(noise, ne, syn_spec="exc")
    nest.Connect(noise, ni, syn_spec="exc")
    nest.Connect(ne[:N_rec], esr, syn_spec="exc")
    nest.Connect(ne, ne + ni, {"rule": "fixed_indegree", "indegree": CE}, "exc")
    nest.Connect(ni, ne + ni, {"rule": "fixed_indegree", "indegree": CI}, "inh")
    nest.Simulate(simtime)
    return esr.n_events / simtime * 1000.0 / N_rec


@unittest.skipUnless(_HAS_NEST, "live NEST not importable")
class TestBrunelAlphaParity(unittest.TestCase):
    def test_excitatory_rate_within_5pct_of_nest(self):
        from examples.nest.brunel_alpha import build
        sim, esr, _isr, _n, _t = build(order=ORDER, simtime=SIMTIME)
        res = sim.simulate(SIMTIME * u.ms)
        bp_rate = res.rate(esr.segments[0].population)
        nest_rate = _nest_rates(ORDER, SIMTIME)
        self.assertGreater(nest_rate, 0.0)
        rel = abs(bp_rate - nest_rate) / nest_rate
        self.assertLess(rel, TOL,
                        f"exc rate brainpy={bp_rate:.2f} nest={nest_rate:.2f} "
                        f"rel={rel:.3f} > {TOL}")
