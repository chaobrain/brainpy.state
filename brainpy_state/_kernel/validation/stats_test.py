# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the G2/G3 spike statistics and the agreement rule."""
import numpy as np
import pytest

from brainpy_state._kernel.validation import stats as S


def poisson_trains(n, rate_hz, t1_ms, seed=0):
    rng = np.random.default_rng(seed)
    times, senders = [], []
    for i in range(n):
        k = rng.poisson(rate_hz * t1_ms * 1e-3)
        times.append(np.sort(rng.uniform(0, t1_ms, k)))
        senders.append(np.full(k, i))
    return np.concatenate(times), np.concatenate(senders)


class TestStatistics:
    def test_poisson_values(self):
        t, s = poisson_trains(400, 20.0, 10_000.0)
        st = S.network_stats(t, s, n_e=300, n_i=100, t0=0.0, t1=10_000.0)
        assert st['rate_E'] == pytest.approx(20.0, rel=0.02)
        assert st['rate_I'] == pytest.approx(20.0, rel=0.03)
        assert st['cv_isi_E'] == pytest.approx(1.0, abs=0.03)   # exponential ISIs
        assert st['fano_E'] == pytest.approx(1.0, abs=0.03)     # Poisson counts
        assert abs(st['corr_E']) < 0.01                         # independent trains

    def test_regular_and_synchronous(self):
        # identical regular trains: CV 0, one spike in every bin (Fano 0); bursts: correlation 1
        t = np.tile(np.arange(5.0, 1000.0, 10.0), 4)
        s = np.repeat(np.arange(4), 100)
        counts = S.binned_counts(t, s, 4, 0.0, 1000.0)
        assert counts.shape == (4, 100) and counts.sum() == 400
        assert S.cv_isi(t, s, 4) == 0.0 and S.fano(counts) == 0.0
        bursts = np.concatenate([np.full(4, 50.0), np.full(4, 420.0), np.full(4, 777.0)])
        bs = np.tile(np.arange(4), 3)
        assert S.mean_correlation(S.binned_counts(bursts, bs, 4, 0.0, 1000.0)) == pytest.approx(1.0)

    def test_silent_neurons_and_windows(self):
        t = np.array([1.0, 2.0, 3.0, 1500.0])
        s = np.array([0, 0, 0, 1])
        tw, sw = S.window(t, s, 0.0, 1000.0)
        assert tw.tolist() == [1.0, 2.0, 3.0] and sw.tolist() == [0, 0, 0]
        assert np.isnan(S.cv_isi([], [], 3))
        assert np.isnan(S.fano(np.zeros((3, 10))))
        assert np.isnan(S.mean_correlation(np.zeros((3, 10))))
        st = S.network_stats(t, s, n_e=2, n_i=1, t0=0.0, t1=1000.0)
        assert st['rate_E'] == 1.5 and st['rate_I'] == 0.0
        with pytest.raises(ValueError):
            S.window([1.0], [0, 1], 0.0, 1.0)
        with pytest.raises(ValueError):
            S.rate([], [], 0, 1.0)
        with pytest.raises(ValueError):
            S.binned_counts(t, s, 2, 0.0, 1.0)


class TestG2Rule:
    def test_relative_and_sem_parts(self):
        r = S.g2_rule([100.0, 101.0, 99.0], [104.0, 104.5, 103.5], 'x')
        assert r.passed and r.bound == pytest.approx(0.05 * 104.0)  # relative part wins
        noisy = S.g2_rule([1.0, 3.0, 5.0], [6.0, 2.0, 7.0])           # SEM part wins
        assert noisy.bound == pytest.approx(2 * np.hypot(np.std([1, 3, 5], ddof=1) / np.sqrt(3),
                                                         np.std([6, 2, 7], ddof=1) / np.sqrt(3)))
        assert noisy.passed
        bad = S.g2_rule([12.0, 12.1, 11.9], [10.0, 10.1, 9.9], 'rate')
        assert not bad.passed and bad.diff == pytest.approx(2.0)

    def test_nan_fails_and_needs_two_seeds(self):
        assert not S.g2_rule([np.nan, 1.0], [1.0, 1.0]).passed
        with pytest.raises(ValueError):
            S.g2_rule([1.0], [1.0, 2.0])

    def test_table(self):
        txt = S.format_g2_table([S.g2_rule([1.0, 1.1], [1.0, 1.05], 'rate_E'),
                                 S.g2_rule([5.0, 5.1], [1.0, 1.05], 'cv')])
        assert '| rate_E |' in txt and 'FAIL' in txt and 'pass' in txt
