# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST parity of the validation-local :class:`iaf_psc_exp` and a scaled microcircuit check.

Single neuron: exact model, ``CAT_B`` on ``V_m`` and the PSCs, equal spike times (the
alignment of :mod:`brainpy_state._kernel.models.nest_parity_test`). Network: the
microcircuit at ``scale = 0.02`` (both sides, 3 seeds) gives every population a rate
within the G2 rule's 5 % or 2 SEM bound or 1 Hz (small-network smoke test; the gate
itself is full scale, ``benchmarks/k1a_microcircuit.py``).
"""
from __future__ import annotations

import unittest

import brainunit as u
import jax
import numpy as np

from brainpy_state._kernel.models import dc_generator
from brainpy_state._kernel.models._testing import rollout
from brainpy_state._kernel.models.nest_parity_test import (
    DT,
    _psc_port,
    _psc_spikes,
    _with_units,
    nest_run,
    ours_run,
)
from brainpy_state._kernel.validation.microcircuit import iaf_psc_exp
from brainpy_state._nest_validation import tolerance_conventions as tc
from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest


@requires_nest
class TestIafPscExpVsNest(unittest.TestCase):
    """Exact model: CAT_B (1e-6 mV) with align_steps = 0."""

    def _check(self, params, spikes=(), t_sim=100.0, current=None, dc=None):
        record = ('V_m', 'I_syn_ex', 'I_syn_in')
        # NEST keeps V_m = -70 mV at Create when E_L changes; the kernel starts at E_L
        nest_params = dict(params, V_m=params.get('E_L', -70.0))
        ref, ref_spk = nest_run('iaf_psc_exp', nest_params, t_sim, spikes=spikes, dc=dc,
                                record=record)
        with jax.enable_x64(True):
            tr, spk = ours_run(iaf_psc_exp, _with_units(iaf_psc_exp, params), t_sim,
                               spikes=spikes, port_of=_psc_port, record=record, current=current)
        for r in record:
            tol = tc.CAT_B if r == 'V_m' else tc.TraceTolerance(1e-6 * u.pA, 1e-6)
            compare_trace(ref[r], tr[r], tol=tol, metric=r).assert_()
        np.testing.assert_allclose(spk, ref_spk, atol=1e-9)
        return ref, ref_spk

    def test_dc_trace_and_spike_times(self):
        _, spk = self._check({'I_e': 420.0}, t_sim=200.0)
        self.assertGreater(len(spk), 5)

    def test_psp_traces(self):
        ref, spk = self._check({'tau_syn_ex': 0.5, 'tau_syn_in': 3.0}, spikes=_psc_spikes())
        self.assertEqual(len(spk), 0)
        self.assertGreater(np.max(ref['V_m']), -69.5)

    def test_psp_with_tau_m_equal_tau_syn(self):
        self._check({'tau_syn_ex': 10.0, 'tau_syn_in': 10.0}, spikes=_psc_spikes())

    def test_spiking_with_input_and_off_grid_refractory(self):
        ex = (np.arange(5.0, 95.0, 0.7).round(1), 600.0)
        _, spk = self._check({'t_ref': 0.25, 'I_e': 200.0}, spikes=[ex])
        self.assertGreater(len(spk), 3)

    def test_microcircuit_neuron_parameters(self):
        p = {'E_L': -65.0, 'V_th': -50.0, 'V_reset': -65.0, 'C_m': 250.0, 'tau_m': 10.0,
             'tau_syn_ex': 0.5, 'tau_syn_in': 0.5, 't_ref': 2.0, 'I_e': 500.0}
        _, spk = self._check(p, spikes=_psc_spikes(), t_sim=200.0)
        self.assertGreater(len(spk), 3)

    def test_dc_generator_onset_through_current_latch(self):
        t_sim = 30.0
        n_steps = int(round(t_sim / DT))
        dc = dc_generator(1, params=dict(amplitude=300.0 * u.pA, start=5.0 * u.ms, stop=20.0 * u.ms))
        out, _ = rollout(dc, n_steps, DT * u.ms)
        emitted = np.asarray(out['I'].to_decimal(u.pA))[:, 0]
        delivered = np.concatenate([[0.0], emitted[:-1]])
        self._check({}, t_sim=t_sim, current=delivered,
                    dc={'amplitude': 300.0, 'start': 5.0, 'stop': 20.0})


SMALL_SCALE = 0.02
SMALL_SEEDS = (1, 2, 3, 4)


@requires_nest
def test_scaled_microcircuit_rates_match_nest():
    """Scale 0.02 (1543 neurons, in-degrees kept), 4 seeds each side, 500 ms after 100 ms.

    Per population: the G2 rule, or ``|Δ| <= 1 Hz`` (the smallest populations have
    ~20 neurons here, so the seed spread alone is not a usable bound).
    """
    from brainpy_state._kernel.validation.microcircuit import (
        POPULATIONS,
        kernel_microcircuit,
        nest_microcircuit,
    )
    from brainpy_state._kernel.validation.stats import g2_rule

    ref = [nest_microcircuit(s, 100.0, 500.0, scale=SMALL_SCALE, threads=4)['rates']
           for s in SMALL_SEEDS]
    ours = [kernel_microcircuit(s, 100.0, 500.0, scale=SMALL_SCALE)['rates'] for s in SMALL_SEEDS]
    bad = []
    for k in POPULATIONS:
        r = g2_rule([o[k] for o in ours], [x[k] for x in ref], k)
        print(k, round(r.ours_mean, 3), round(r.ref_mean, 3), round(r.bound, 3))
        if not (r.passed or abs(r.diff) <= 1.0):
            bad.append(r)
    assert not bad, bad
    assert all(np.mean([x[k] for x in ref]) > 0.1 for k in POPULATIONS)  # network is active
