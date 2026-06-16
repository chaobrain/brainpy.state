# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest_like/pulsepacket.py``.

NEST's §3.7 ``pulsepacket`` demo emits Gaussian-jittered synchronous spike packets
from a ``pulsepacket_generator`` and compares the neuron-averaged membrane
excursion to the analytical Diesmann solution. The headline checked against live
NEST is the **packet shape**: the pooled spike-time standard deviation equals the
jitter ``sdev``, and the per-step population spike-count profile matches NEST
distributionally (category D — seed-aggregated, since NEST's per-thread RNG and
the host NumPy RNG diverge sample-by-sample). The membrane excursion is verified
**NEST-free** against the analytical Gaussian⊛PSP convolution.

The generator is host-side (NumPy RNG + ``deque`` queues), so there is no
for_loop-lowering test here; the eager host loop is the contract.
"""
import gc
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest_validation.tolerance_conventions import CAT_D

from examples.nest_like.pulsepacket import (
    generate_packet, packet_stats, packet_psth, averaged_membrane,
    analytical_excursion, excursion_window,
    ACTIVITY, SDEV, PULSE_T, N_NEURONS, SIMTIME, DT, PST_BIN)

SEEDS = (0, 1, 2, 3, 4)


def _nest_events(seed, *, pulse_times, activity, sdev, n_trains, simtime, dt):
    """Spike (senders, times) from a NEST ``pulsepacket_generator`` bank."""
    nest.ResetKernel()
    nest.resolution = dt
    nest.local_num_threads = 1
    nest.rng_seed = seed + 1
    params = {'pulse_times': list(np.asarray(pulse_times, dtype=float)),
              'activity': int(activity), 'sdev': float(sdev)}
    gens = nest.Create('pulsepacket_generator', n_trains, params=params)
    sr = nest.Create('spike_recorder')
    nest.Connect(gens, sr)
    nest.Simulate(simtime)
    ev = sr.get('events')
    return np.asarray(ev['senders']), np.asarray(ev['times'], dtype=float)


def _nest_per_step_counts(times, simtime, dt):
    """Per-step population spike counts from recorded NEST spike times."""
    n_steps = int(round(simtime / dt))
    if times.size == 0:
        return np.zeros(n_steps, dtype=float)
    steps = np.rint(times / dt).astype(np.int64)
    counts = np.bincount(steps, minlength=n_steps + 2).astype(float)
    return counts[1:n_steps + 1]


class TestPulsepacketStructural(unittest.TestCase):
    """Packet-shape and membrane invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def tearDown(self):
        # generate_packet is host-side, but averaged_membrane compiles a Simulator
        # rollout per call; clear caches so artifacts do not accumulate across tests.
        jax.clear_caches()
        gc.collect()

    def test_total_count_exact(self):
        mat = generate_packet(seed=0, simtime=600.0, pulse_times=(300.0,))
        self.assertEqual(packet_stats(mat)['total'], N_NEURONS * ACTIVITY)

    def test_width_tracks_sdev(self):
        # The pooled spike-time std estimates the jitter sdev, and grows with it.
        widths = []
        for sd in (5.0, 10.0, 20.0):
            mat = generate_packet(seed=0, simtime=700.0, pulse_times=(350.0,), sdev=sd)
            widths.append(packet_stats(mat)['std_ms'])
            self.assertAlmostEqual(widths[-1], sd, delta=0.15 * sd)   # width ≈ sdev
        self.assertTrue(np.all(np.diff(widths) > 0.0))               # monotone in sdev

    def test_center_at_pulse(self):
        mat = generate_packet(seed=0, simtime=600.0, pulse_times=(300.0,))
        self.assertAlmostEqual(packet_stats(mat)['mean_ms'], 300.0, delta=1.0)

    def test_sdev_zero_is_synchronous(self):
        # sdev=0 ⇒ every spike at the pulse step ⇒ a single nonzero bin of height N·a.
        mat = generate_packet(seed=0, simtime=200.0, pulse_times=(100.0,), sdev=0.0)
        per_step = mat.sum(axis=1)
        self.assertEqual(int((per_step > 0).sum()), 1)
        self.assertEqual(int(per_step.max()), N_NEURONS * ACTIVITY)

    def test_multi_packet_two_bumps(self):
        mat = generate_packet(seed=0, simtime=600.0, pulse_times=(200.0, 400.0), sdev=8.0)
        centers, counts = packet_psth(mat, bin_ms=5.0)
        self.assertEqual(int(mat.sum()), 2 * N_NEURONS * ACTIVITY)
        # Two separated Gaussian bumps: mass concentrated near both centers, sparse between.
        near = ((np.abs(centers - 200.0) < 25.0) | (np.abs(centers - 400.0) < 25.0))
        between = (np.abs(centers - 300.0) < 25.0)
        self.assertGreater(counts[near].sum(), 20.0 * counts[between].sum() + 1.0)

    def test_activity_zero_is_empty(self):
        mat = generate_packet(seed=0, simtime=200.0, pulse_times=(100.0,), activity=0)
        self.assertEqual(int(mat.sum()), 0)

    def test_membrane_matches_analytical(self):
        mat = generate_packet(seed=0)
        t_sim, exc = averaged_membrane(mat)
        _t_an, u_an = analytical_excursion()
        win = excursion_window(t_sim)
        corr = np.corrcoef(exc[win], u_an[win])[0, 1]
        self.assertGreater(corr, 0.99)                               # shape matches theory
        ratio = exc[win].max() / u_an[win].max()
        self.assertTrue(0.85 < ratio < 1.15)                         # amplitude within 15%

    def test_membrane_peak_latency(self):
        mat = generate_packet(seed=0)
        t_sim, exc = averaged_membrane(mat)
        latency = t_sim[int(np.argmax(exc))] - PULSE_T
        self.assertTrue(8.0 < latency < 16.0)                        # PSP rise-time after packet

    def test_main_smoke(self):
        import io
        import contextlib
        from examples.nest_like.pulsepacket import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main()
        out = buf.getvalue()
        self.assertIn('pulsepacket', out)
        self.assertIn('width', out)
        self.assertIn('excursion', out)


@requires_nest
class TestPulsepacketParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_packet_width_matches_nest_distributional(self):
        # Headline: the pooled spike-time std (the packet width) tracks sdev in both,
        # compared as a seed-aggregated mean (category D).
        def bp_width(seed):
            return packet_stats(generate_packet(seed=seed))['std_ms']

        def nest_width(seed):
            _s, times = _nest_events(seed, pulse_times=(PULSE_T,), activity=ACTIVITY,
                                     sdev=SDEV, n_trains=N_NEURONS, simtime=SIMTIME, dt=DT)
            return float(np.std(times))

        bp = [bp_width(s) for s in SEEDS]
        ns = [nest_width(s) for s in SEEDS]
        # Both must estimate sdev before we compare them to each other.
        self.assertAlmostEqual(float(np.mean(bp)), SDEV, delta=0.1 * SDEV)
        self.assertAlmostEqual(float(np.mean(ns)), SDEV, delta=0.1 * SDEV)
        compare_distributional(ns, bp, tol=CAT_D, metric='pulsepacket width (ms)',
                               statistic='mean').assert_()

    def test_packet_profile_matches_nest(self):
        # The per-step population count profile (a Gaussian bump) matches NEST: same
        # total, aligned peak, per-window mass, and high smoothed correlation.
        mat = generate_packet(seed=0)
        bp_counts = mat.sum(axis=1).astype(float)
        # Local send-step counts lead the NEST recorder timestamps by one step.
        bp_aligned = np.zeros_like(bp_counts)
        bp_aligned[1:] = bp_counts[:-1]

        _s, times = _nest_events(0, pulse_times=(PULSE_T,), activity=ACTIVITY,
                                 sdev=SDEV, n_trains=N_NEURONS, simtime=SIMTIME, dt=DT)
        nest_counts = _nest_per_step_counts(times, SIMTIME, DT)

        total = N_NEURONS * ACTIVITY
        self.assertEqual(int(bp_aligned.sum()), total)
        self.assertEqual(int(nest_counts.sum()), total)

        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel = kernel / kernel.sum()
        bp_sm = np.convolve(bp_aligned, kernel, mode='same')
        ns_sm = np.convolve(nest_counts, kernel, mode='same')

        lo, hi = int((PULSE_T - 60.0) / DT), int((PULSE_T + 60.0) / DT)
        corr = np.corrcoef(bp_sm[lo:hi], ns_sm[lo:hi])[0, 1]
        self.assertGreater(corr, 0.93)

        bp_mass = float(bp_aligned[lo:hi].sum())
        ns_mass = float(nest_counts[lo:hi].sum())
        self.assertAlmostEqual(bp_mass, ns_mass, delta=0.08 * ns_mass)

        # Centers align: compare the count-weighted centroid (a robust moment), not
        # the argmax — the mode of a finite-sample Gaussian histogram wobbles a few ms
        # between independent RNG realizations while the centroid is stable to <0.5 ms.
        idx = np.arange(lo, hi)
        bp_centroid = float((idx * bp_aligned[lo:hi]).sum() / bp_aligned[lo:hi].sum()) * DT
        ns_centroid = float((idx * nest_counts[lo:hi]).sum() / nest_counts[lo:hi].sum()) * DT
        self.assertLessEqual(abs(bp_centroid - ns_centroid), 1.0)    # packet centers aligned


if __name__ == '__main__':
    unittest.main()
