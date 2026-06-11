# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel mean-field (Siegert) rate parity: brainpy.state vs live NEST.

Drives the faithful ``examples/nest/brunel_siegert.py`` port and asserts the
asymptotic excitatory/inhibitory rates land within 5 % of a live NEST run built
from identical parameters. Both solve the same self-consistent mean-field
equation (Hahne et al. 2017, eqs. 27-30) for the spiking ``brunel_delta``
network, so — unlike the spiking variants — the comparison is a deterministic
fixed-point match, not a statistical one. Skipped when ``nest`` is unavailable.
"""
import unittest

import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import CAT_C_RATE

ORDER = 2500         # the real Brunel order (mean-field cost is O(1) in N)
SIMTIME = 50.0


def _nest_rates(order, simtime):
    nest.ResetKernel()
    nest.resolution = 0.1
    g, eta, epsilon = 5.0, 2.0, 0.1
    NE, NI = 4 * order, order
    CE, CI = int(epsilon * NE), int(epsilon * NI)
    tauMem, theta = 20.0, 20.0
    neuron_params = {"tau_m": tauMem, "t_ref": 2.0, "theta": theta, "V_reset": 0.0}

    J = 0.1
    J_ex = J
    J_in = -g * J_ex
    pref = tauMem * 1e-3
    drift_factor_ext = pref * J_ex
    drift_factor_ex = pref * CE * J_ex
    drift_factor_in = pref * CI * J_in
    diffusion_factor_ext = pref * J_ex ** 2
    diffusion_factor_ex = pref * CE * J_ex ** 2
    diffusion_factor_in = pref * CI * J_in ** 2

    nu_th = theta / (J * CE * tauMem)
    p_rate = 1000.0 * (eta * nu_th) * CE

    siegert_ex = nest.Create("siegert_neuron", params=neuron_params)
    siegert_in = nest.Create("siegert_neuron", params=neuron_params)
    siegert_drive = nest.Create("siegert_neuron", params={"mean": p_rate})
    mm = nest.Create("multimeter", params={"record_from": ["rate"], "interval": 0.1})

    nest.Connect(siegert_drive, siegert_ex + siegert_in, "all_to_all",
                 {"drift_factor": drift_factor_ext, "diffusion_factor": diffusion_factor_ext,
                  "synapse_model": "diffusion_connection"})
    nest.Connect(siegert_ex, siegert_ex + siegert_in, "all_to_all",
                 {"drift_factor": drift_factor_ex, "diffusion_factor": diffusion_factor_ex,
                  "synapse_model": "diffusion_connection"})
    nest.Connect(siegert_in, siegert_ex + siegert_in, "all_to_all",
                 {"drift_factor": drift_factor_in, "diffusion_factor": diffusion_factor_in,
                  "synapse_model": "diffusion_connection"})
    nest.Connect(mm, siegert_ex + siegert_in)
    nest.Simulate(simtime)

    import numpy as np
    data = mm.events
    rex = data["rate"][np.where(data["senders"] == siegert_ex.global_id)][-1]
    rin = data["rate"][np.where(data["senders"] == siegert_in.global_id)][-1]
    return float(rex), float(rin)


@requires_nest
class TestBrunelSiegertParity(unittest.TestCase):
    def test_meanfield_rate_within_5pct_of_nest(self):
        from examples.nest.brunel_siegert import run
        erate, irate, *_ = run(order=ORDER, simtime=SIMTIME)
        nest_ex, nest_in = _nest_rates(ORDER, SIMTIME)
        self.assertGreater(nest_ex, 0.0)
        self.assertGreater(nest_in, 0.0)
        # Deterministic mean-field fixed point -> trace mode (category C, rate).
        for label, bp, ns in (("exc", erate, nest_ex), ("inh", irate, nest_in)):
            compare_trace(ns, bp, tol=CAT_C_RATE, metric=f"{label} mean-field rate").assert_()
