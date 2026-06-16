# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/multimeter_file.py``.

NEST's ``multimeter_file`` records several analog variables from a spike-driven
neuron. This in-memory port records ``V_m``/``I_syn_ex``/``I_syn_in`` from an
``iaf_psc_exp`` driven by two ``spike_generator``s (excitatory + inhibitory).
All three traces are deterministic analytic-propagator curves; we compare each
against live NEST at category B with two-step generator-delivery alignment
(generator spike-holder + recorder offset; see ``CAT_B_GEN``).
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

SIMTIME = 100.0

# A generator-driven analog trace carries a constant *two*-step delivery offset
# vs NEST: one step from the Simulator's generator spike-holder (the device
# output is captured one step before the projection delivers it) plus the usual
# one-step recorder offset (``CAT_B_ALIGNED``). The shapes are otherwise
# bit-identical (exact propagator), so a plain, tight tolerance with
# ``align_steps=2`` matches all three recordables to machine precision.
CAT_B_GEN = TraceTolerance(1e-3, 1e-6, align_steps=2, label="B",
                           note="analytic trace, generator two-step delivery alignment")


def _nest_traces(simtime):
    """Record V_m/I_syn_ex/I_syn_in from a NEST iaf_psc_exp under the same drive."""
    from examples.nest_like.multimeter_file import (
        SPIKE_TIMES_EX, SPIKE_TIMES_IN, W_EX, W_IN, DELAY, RECORD_FROM)
    nest.ResetKernel()
    nest.resolution = 0.1
    n = nest.Create("iaf_psc_exp")
    m = nest.Create("multimeter", params={"interval": 0.1,
                                          "record_from": list(RECORD_FROM)})
    s_ex = nest.Create("spike_generator", params={"spike_times": list(SPIKE_TIMES_EX)})
    s_in = nest.Create("spike_generator", params={"spike_times": list(SPIKE_TIMES_IN)})
    nest.Connect(s_ex, n, syn_spec={"weight": W_EX, "delay": DELAY})
    nest.Connect(s_in, n, syn_spec={"weight": W_IN, "delay": DELAY})
    nest.Connect(m, n)
    nest.Simulate(simtime)
    ev = nest.GetStatus(m, "events")[0]
    return {name: np.asarray(ev[name]) for name in RECORD_FROM}


class TestMultimeterFileStructural(unittest.TestCase):
    """The multi-recordable recording payload — NEST-free."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_main_smoke(self):
        import io
        import contextlib
        from unittest import mock
        from examples.nest_like.multimeter_file import main

        # Suppress the plot artifact: patch savefig if matplotlib is present,
        # otherwise main() takes its own graceful no-matplotlib branch.
        try:
            import matplotlib  # noqa: F401
            plot_patch = mock.patch("matplotlib.pyplot.savefig")
        except ImportError:
            plot_patch = contextlib.nullcontext()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), plot_patch:
            main()
        out = buf.getvalue()
        self.assertIn("multimeter_file", out)
        self.assertIn("V_m", out)
        self.assertIn("I_syn_ex", out)
        self.assertIn("I_syn_in", out)


@requires_nest
class TestMultimeterFileParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_all_recordables_match_nest(self):
        from examples.nest_like.multimeter_file import build, RECORD_FROM
        sim, mm, _neuron, _t = build(simtime=SIMTIME)
        res = sim.simulate(SIMTIME * u.ms)

        bp = {
            'V_m': np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV)).reshape(-1),
            'I_syn_ex': np.asarray(u.get_mantissa(res.trace(mm, 'I_syn_ex') / u.pA)).reshape(-1),
            'I_syn_in': np.asarray(u.get_mantissa(res.trace(mm, 'I_syn_in') / u.pA)).reshape(-1),
        }
        ref = _nest_traces(SIMTIME)
        units = {'V_m': 'mV', 'I_syn_ex': 'pA', 'I_syn_in': 'pA'}
        for name in RECORD_FROM:
            compare_trace(ref[name], bp[name], tol=CAT_B_GEN,
                          metric=f"{name} ({units[name]})").assert_()

        # Each recordable must carry a real signal (not a trivially-flat match).
        self.assertGreater(bp['I_syn_ex'].max(), 40.0)   # excitatory PSC present
        self.assertLess(bp['I_syn_in'].min(), -20.0)     # inhibitory PSC present
        self.assertGreater(float(bp['V_m'].max() - bp['V_m'].min()), 0.05)


if __name__ == "__main__":
    unittest.main()
