# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/precise_spiking.py``.

NEST's ``precise_spiking`` drives a grid model (``iaf_psc_exp``) and its precise
twin (``iaf_psc_exp_ps``) with the same DC current at several resolutions and
contrasts their spike timing. The brainpy port runs the grid model on the
``Simulator`` and the precise model **eagerly** (a plain Python loop with
concrete ``t = k*dt``), reading the off-grid spike times from
``last_spike_time``.

Both models are deterministic, so parity is category E (spike-time / event
count). The DC drive reaches the neuron after an arbitrary connection-delay
onset (NEST's default 1 ms vs the Simulator/eager one-step convention), a
constant time shift identical for every spike. Parity is therefore asserted on
the **onset-aligned** spike sequence — the spike *count* (exact) and the spike
times *relative to the first spike* within ``max_peak_step_diff`` (one step).
The relative sequence is the physically meaningful content (firing period and
its resolution dependence); the absolute onset is a documented convention.
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

from brainpy_state._nest_validation.nest_compare import requires_nest
from brainpy_state._nest_validation.tolerance_conventions import CAT_E


def _nest_spike_times(model, dt, simtime, stim_current):
    """Spike times (ms) of a NEST grid/precise neuron under a DC drive."""
    nest.ResetKernel()
    nest.resolution = dt
    neuron = nest.Create(model)
    dc = nest.Create("dc_generator", params={"amplitude": stim_current})
    sr = nest.Create("spike_recorder")
    nest.Connect(dc, neuron)
    nest.Connect(neuron, sr)
    nest.Simulate(simtime)
    return np.asarray(sr.events["times"], dtype=float)


def _relative(times):
    """Spike times shifted so the first spike sits at 0 (onset-aligned)."""
    times = np.asarray(times, dtype=float)
    return times - times[0] if times.size else times


def _assert_spike_parity(self, bp_times, ne_times, dt, label):
    """Count + onset-aligned relative-time parity at category E."""
    self.assertLessEqual(
        abs(len(bp_times) - len(ne_times)), CAT_E.max_count_diff,
        f"{label}: count brainpy={len(bp_times)} NEST={len(ne_times)}")
    n = min(len(bp_times), len(ne_times))
    self.assertGreater(n, 0, f"{label}: no spikes")
    bp_rel, ne_rel = _relative(bp_times)[:n], _relative(ne_times)[:n]
    max_diff = float(np.abs(bp_rel - ne_rel).max())
    bound = CAT_E.max_peak_step_diff * dt + 1e-9
    self.assertLessEqual(
        max_diff, bound,
        f"{label}: relative spike-time max|Δ|={max_diff:.4g} ms > {bound:.4g} ms")


class TestPreciseSpikingStructural(unittest.TestCase):
    """The grid-vs-precise contrast — the demo's payload, NEST-free."""

    def test_grid_fires_on_grid(self):
        from examples.nest_like.precise_spiking import run_grid
        out = run_grid(dt=0.1)
        steps = out["spike_steps"]
        self.assertGreater(steps.size, 0)
        self.assertTrue(np.issubdtype(steps.dtype, np.integer))
        # Steady DC drive -> regular firing: near-constant inter-spike step gap.
        if steps.size > 2:
            isi = np.diff(steps)
            self.assertLessEqual(int(isi.max() - isi.min()), 1)
        # The voltmeter trace spans the run at the resolution grid.
        self.assertEqual(out["vm"].shape[0], out["times"].shape[0])

    def test_precise_fires_off_grid(self):
        from examples.nest_like.precise_spiking import run_grid, run_precise
        out = run_precise(dt=0.1)
        st = out["spike_times"]
        self.assertGreater(st.size, 0)
        # The precise model resolves spikes between grid points: at least one
        # spike time is not an integer multiple of dt.
        frac = np.abs(st / 0.1 - np.round(st / 0.1))
        self.assertGreater(float(frac.max()), 1e-6)
        # It fires about as often as the grid model on the same drive.
        grid = run_grid(dt=0.1)
        self.assertLessEqual(abs(out["spike_steps"].size - grid["spike_steps"].size),
                             CAT_E.max_count_diff)

    def test_precise_time_is_resolution_robust(self):
        from examples.nest_like.precise_spiking import run_precise
        # Onset-aligned precise firing period is set by the continuous dynamics,
        # so it barely moves with the integration step (unlike the grid model).
        p1 = _relative(run_precise(dt=0.1)["spike_times"])
        p2 = _relative(run_precise(dt=0.5)["spike_times"])
        n = min(p1.size, p2.size)
        self.assertGreater(n, 1)
        self.assertLess(float(np.abs(p1[:n] - p2[:n]).max()), 0.5)   # < one 0.5 ms step

    def test_main_smoke(self):
        import io
        import contextlib
        from examples.nest_like.precise_spiking import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main()
        out = buf.getvalue()
        self.assertIn("grid", out)
        self.assertIn("precise", out)
        self.assertIn("resolution 0.1 ms", out)


@requires_nest
class TestPreciseSpikingParity(unittest.TestCase):
    def test_grid_matches_nest(self):
        from examples.nest_like.precise_spiking import run_grid, RESOLUTIONS, SIMTIME, STIM_CURRENT
        for dt in RESOLUTIONS:
            bp = run_grid(dt=dt)
            bp_times = bp["spike_steps"] * dt
            ne_times = _nest_spike_times("iaf_psc_exp", dt, SIMTIME, STIM_CURRENT)
            _assert_spike_parity(self, bp_times, ne_times, dt, f"grid dt={dt}")

    def test_precise_matches_nest(self):
        from examples.nest_like.precise_spiking import run_precise, RESOLUTIONS, SIMTIME, STIM_CURRENT
        for dt in RESOLUTIONS:
            bp = run_precise(dt=dt)
            ne_times = _nest_spike_times("iaf_psc_exp_ps", dt, SIMTIME, STIM_CURRENT)
            _assert_spike_parity(self, bp["spike_times"], ne_times, dt, f"precise dt={dt}")


if __name__ == "__main__":
    unittest.main()
