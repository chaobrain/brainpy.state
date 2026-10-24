# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests of the K1a-G4 microcircuit builder (small scale; no NEST)."""
from __future__ import annotations

import brainunit as u
import jax
import numpy as np
import pytest
import scipy.stats

from brainspike._kernel.validation.microcircuit import (
    POPULATIONS,
    TruncatedNormal,
    build_microcircuit,
    kernel_microcircuit,
    pd14_params,
    population_rates,
)


class TestParams:
    def test_full_scale_sizes(self):
        p = pd14_params()
        assert int(p['n'].sum()) == 77169
        # ~3e8 synapses (spec), reference formula
        assert 2.9e8 < p['n_syn'].sum() < 3.1e8
        assert p['n_syn'][0, 5] == 0 and p['n_syn'][2, 7] == 0  # zero probabilities

    def test_weights_signs_and_l4e_to_l23e(self):
        p = pd14_params()
        w = p['w_mean']
        assert np.all(w[:, 0::2] > 0) and np.all(w[:, 1::2] < 0)
        assert w[0, 2] == pytest.approx(2.0 * w[0, 0])
        assert w[1, 1] == pytest.approx(-4.0 * w[1, 0])
        # 0.15 mV PSP with tau_m 10, tau_syn 0.5, C_m 250 -> ~87.8 pA (reference value)
        assert p['w_ext'] == pytest.approx(87.81, abs=0.01)
        assert np.all(p['d_mean'][:, 0::2] == 1.5) and np.all(p['d_mean'][:, 1::2] == 0.75)

    def test_scaled(self):
        p = pd14_params(0.1)
        assert np.array_equal(p['n'], np.round(np.array([20683, 5834, 21915, 5479, 4850, 1065,
                                                           14395, 2948]) * 0.1).astype(int))
        assert p['n_syn'].sum() == pytest.approx(pd14_params()['n_syn'].sum() * 0.1, rel=1e-3)

    def test_dc_compensation(self):
        p = pd14_params()
        assert p['dc'] == pytest.approx(8.0 * p['k_ext'] * p['w_ext'] * 0.5e-3)

    def test_population_rates(self):
        r = population_rates(dict.fromkeys(POPULATIONS, 10.0), np.full(8, 5), 1000.0)
        assert r == dict.fromkeys(POPULATIONS, 2.0)


class TestTruncatedNormal:
    def test_bounds_and_mean_match_scipy(self):
        d = TruncatedNormal(1.5 * u.ms, 0.75 * u.ms, low=0.05 * u.ms)
        x = np.asarray(d.sample(jax.random.PRNGKey(0), (200000,)).to_decimal(u.ms))
        assert x.min() >= 0.05
        a = (0.05 - 1.5) / 0.75
        ref = scipy.stats.truncnorm(a, np.inf, loc=1.5, scale=0.75)
        assert x.mean() == pytest.approx(ref.mean(), abs=5e-3)
        assert x.std() == pytest.approx(ref.std(), abs=5e-3)
        # redraw, not clip: no mass on the bound
        assert np.mean(x < 0.06) < 0.005

    def test_upper_bound(self):
        d = TruncatedNormal(-10.0 * u.pA, 1.0 * u.pA, high=0.0 * u.pA)
        x = np.asarray(d.sample(jax.random.PRNGKey(1), (1000,)).to_decimal(u.pA))
        assert x.max() <= 0.0
        assert d.bounds() == (-np.inf, 0.0)

    def test_validation(self):
        with pytest.raises(ValueError):
            TruncatedNormal(1.0 * u.ms, 0.0 * u.ms)
        with pytest.raises(Exception):
            TruncatedNormal(1.0 * u.ms, 1.0 * u.pA)


class TestBuild:
    def test_bad_bg_input(self):
        with pytest.raises(ValueError, match='bg_input'):
            build_microcircuit(0.01, bg_input='noise')

    @pytest.mark.parametrize('bg', ['poisson', 'dc'])
    def test_tiny_network_runs(self, bg):
        r = kernel_microcircuit(1, 10.0, 20.0, scale=0.005, bg_input=bg, chunk_ms=10.0)
        assert set(r['rates']) == set(POPULATIONS)
        assert all(np.isfinite(v) and v >= 0 for v in r['rates'].values())
        assert len(r['chunk_s']) == 2 and r['ms_per_step'] > 0
        assert r['n_edges'] > 0

    def test_no_recording(self):
        net, info = build_microcircuit(0.005, record_interval=None)
        assert info['obs'] is None
        assert len(info['pops']) == 8

    def test_kernel_run_argument_checks(self):
        with pytest.raises(ValueError, match='chunk_ms'):
            kernel_microcircuit(1, 50.0, 100.0, scale=0.005, chunk_ms=100.0)
        with pytest.raises(ValueError, match='chunk_ms'):
            kernel_microcircuit(1, 100.0, 150.0, scale=0.005, chunk_ms=100.0)
