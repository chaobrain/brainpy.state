# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel mean-field (Siegert) rate parity: closed form and live NEST.

The rewritten ``examples/nest/brunel_siegert.py`` relaxes three ``siegert_neuron``
nodes -- excitatory, inhibitory, and a constant driving node that replaces the
Poisson background -- coupled by ``diffusion_connection`` end-to-end through the
:class:`~brainpy.state.Simulator` (a single compiled ``for_loop``, no Python step
loop). Two oracles validate the asymptotic rates:

* a NEST-independent closed-form fixed point -- the self-consistent Siegert
  equation solved with the SciPy transfer-function oracle (always runs), and
* a live NEST run built from identical parameters (skipped when ``nest`` is
  absent).

Both solve the same mean-field equation (Hahne et al. 2017, eqs. 27-30) for the
spiking ``brunel_delta`` network, so -- unlike the spiking variants -- the
comparison is a deterministic fixed-point match, not a statistical one. The
example wires *six* convergent ``diffusion_connection`` edges (drive/ex/in into
each of ex/in, including the ex->ex and in->in population self-coupling), so this
also exercises the Simulator's accumulation of multiple labelled drift/diffusion
deposits into one target -- a path the single-source micro-parity test does not
cover.
"""
import importlib.util
import pathlib
import unittest

import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import numpy.testing as npt
import brainunit as u

from brainpy_state import siegert_neuron
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import CAT_C_RATE

ORDER = 2500         # the real Brunel order (mean-field cost is O(1) in N)
SIMTIME = 50.0       # 50 ms = 50 relaxation tau (tau=1 ms) -> fully converged


def _load_run():
    """Import ``run`` from the example by file path (no ``sys.path`` dependency).

    ``examples`` is not an importable package (no top-level ``__init__``), so the
    example is loaded directly from its file under the repository root.
    """
    path = (pathlib.Path(__file__).resolve().parents[2]
            / 'examples' / 'nest' / 'brunel_siegert.py')
    spec = importlib.util.spec_from_file_location('brunel_siegert_example', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def _brunel_params(order):
    """The Brunel mean-field coupling factors, drive rate, and neuron params.

    Shared by the closed-form oracle and the NEST reference so all three (example,
    closed form, NEST) are built from one definition.
    """
    g, eta, epsilon = 5.0, 2.0, 0.1
    NE, NI = 4 * order, order
    CE, CI = int(epsilon * NE), int(epsilon * NI)
    tauMem, theta = 20.0, 20.0
    J = 0.1
    J_ex = J
    J_in = -g * J_ex
    pref = tauMem * 1e-3
    factors = dict(
        drift_ext=pref * J_ex, drift_ex=pref * CE * J_ex, drift_in=pref * CI * J_in,
        diff_ext=pref * J_ex ** 2, diff_ex=pref * CE * J_ex ** 2,
        diff_in=pref * CI * J_in ** 2,
    )
    nu_th = theta / (J * CE * tauMem)
    p_rate = 1000.0 * (eta * nu_th) * CE
    npar = dict(tau_m=tauMem * u.ms, t_ref=2.0 * u.ms, theta=theta, V_reset=0.0)
    return factors, p_rate, npar


def _closed_form_fixed_point(order, n_iter=300):
    """Self-consistent Siegert fixed point via the SciPy oracle (NEST-independent).

    By the symmetry of the Brunel reduction both populations receive identical
    drift and diffusion, so ``r_ex = r_in = r`` solves ``r = Phi(mu(r), s2(r))``
    with ``mu(r) = p_rate*drift_ext + r*(drift_ex + drift_in)`` and
    ``s2(r) = p_rate*diff_ext + r*(diff_ex + diff_in)``. Plain fixed-point
    iteration converges (this is test-oracle code, not a model rollout, so a SciPy
    loop is fine).
    """
    f, p_rate, npar = _brunel_params(order)
    with brainstate.environ.context(dt=0.1 * u.ms):
        ref = siegert_neuron(1, **npar)
        ref.init_state()
        r = 0.0
        for _ in range(n_iter):
            mu = p_rate * f['drift_ext'] + r * (f['drift_ex'] + f['drift_in'])
            s2 = p_rate * f['diff_ext'] + r * (f['diff_ex'] + f['diff_in'])
            r = float(np.asarray(ref.siegert_rate(mu, s2)).reshape(-1)[0])
    return r


def _nest_rates(order, simtime):
    import nest

    nest.ResetKernel()
    nest.resolution = 0.1
    f, p_rate, npar = _brunel_params(order)
    neuron_params = {"tau_m": 20.0, "t_ref": 2.0, "theta": 20.0, "V_reset": 0.0}

    siegert_ex = nest.Create("siegert_neuron", params=neuron_params)
    siegert_in = nest.Create("siegert_neuron", params=neuron_params)
    siegert_drive = nest.Create("siegert_neuron", params={"mean": p_rate})
    mm = nest.Create("multimeter", params={"record_from": ["rate"], "interval": 0.1})

    nest.Connect(siegert_drive, siegert_ex + siegert_in, "all_to_all",
                 {"drift_factor": f['drift_ext'], "diffusion_factor": f['diff_ext'],
                  "synapse_model": "diffusion_connection"})
    nest.Connect(siegert_ex, siegert_ex + siegert_in, "all_to_all",
                 {"drift_factor": f['drift_ex'], "diffusion_factor": f['diff_ex'],
                  "synapse_model": "diffusion_connection"})
    nest.Connect(siegert_in, siegert_ex + siegert_in, "all_to_all",
                 {"drift_factor": f['drift_in'], "diffusion_factor": f['diff_in'],
                  "synapse_model": "diffusion_connection"})
    nest.Connect(mm, siegert_ex + siegert_in)
    nest.Simulate(simtime)

    data = mm.events
    rex = data["rate"][np.where(data["senders"] == siegert_ex.global_id)][-1]
    rin = data["rate"][np.where(data["senders"] == siegert_in.global_id)][-1]
    return float(rex), float(rin)


class TestBrunelSiegertClosedForm(unittest.TestCase):
    """The Simulator relaxation reproduces the closed-form mean-field rate.

    NEST-independent: validates the full ``Simulator`` end-to-end path (six
    convergent ``diffusion_connection`` edges, including population self-coupling)
    against the analytic Siegert fixed point.
    """

    def test_meanfield_rate_matches_closed_form(self):
        run = _load_run()
        erate, irate, *_ = run(order=ORDER, simtime=SIMTIME)
        r_star = _closed_form_fixed_point(ORDER)

        self.assertGreater(r_star, 1.0)          # a nontrivial fixed point
        # Symmetric reduction: ex and in receive identical input every step and
        # start from the same rate, so they relax to the *same* value.
        npt.assert_allclose(erate, irate, rtol=0.0, atol=1e-9)
        # 50 ms = 50 relaxation tau -> fully converged; matches the Phi fixed
        # point. A dropped ex->ex/in->in self-coupling would land near ~63 (vs
        # ~32), and a dropped convergent deposit would shift mu/sigma^2 -- both
        # far outside this band.
        npt.assert_allclose(erate, r_star, rtol=1e-3, atol=1e-2)
        npt.assert_allclose(irate, r_star, rtol=1e-3, atol=1e-2)


@requires_nest
class TestBrunelSiegertParity(unittest.TestCase):
    """The example's asymptotic rates land within 5 % of a live NEST run."""

    def test_meanfield_rate_within_5pct_of_nest(self):
        run = _load_run()
        erate, irate, *_ = run(order=ORDER, simtime=SIMTIME)
        nest_ex, nest_in = _nest_rates(ORDER, SIMTIME)
        self.assertGreater(nest_ex, 0.0)
        self.assertGreater(nest_in, 0.0)
        # Deterministic mean-field fixed point -> trace mode (category C, rate).
        for label, bp_rate, ns in (("exc", erate, nest_ex), ("inh", irate, nest_in)):
            compare_trace(ns, bp_rate, tol=CAT_C_RATE,
                          metric=f"{label} mean-field rate").assert_()


if __name__ == '__main__':
    unittest.main()
