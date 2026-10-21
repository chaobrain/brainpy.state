# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Network-level delivery timing against live NEST (``requires_nest``), no alignment shift.

Both sides are built as networks: ``spike_generator -> iaf_psc_alpha`` (excitatory
and inhibitory) and ``dc_generator -> iaf_psc_alpha`` with the *same* delays in
NEST and in the kernel, recorded every step. Our step ``n`` ends at ``(n + 1) dt``,
which is the NEST multimeter sample at ``(n + 1) dt``: ``align_steps = 0``.
"""
from __future__ import annotations

import unittest

import brainunit as u
import jax
import numpy as np

from brainpy_state._kernel.connectivity import one_to_one
from brainpy_state._kernel.delays import delay_steps
from brainpy_state._kernel.models import dc_generator, iaf_psc_alpha, spike_generator
from brainpy_state._kernel.network import Network
from brainpy_state._nest_validation import tolerance_conventions as tc
from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest

DT = 0.1
T_SIM = 60.0
EX = [5.0, 5.0, 12.3, 20.0, 31.0]
IN = [8.0, 25.5]
DC = dict(amplitude=300.0, start=10.0, stop=40.0)


def nest_network(delay):
    import nest
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = DT
    n = nest.Create('iaf_psc_alpha', 2)
    sg_ex = nest.Create('spike_generator', params={'spike_times': EX})
    sg_in = nest.Create('spike_generator', params={'spike_times': IN})
    dc = nest.Create('dc_generator', params=DC)
    mm = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': DT})
    sr = nest.Create('spike_recorder')
    nest.Connect(sg_ex, n[0], syn_spec={'weight': 300.0, 'delay': delay})
    nest.Connect(sg_in, n[0], syn_spec={'weight': -200.0, 'delay': delay})
    nest.Connect(dc, n[1], syn_spec={'delay': delay})
    nest.Connect(mm, n)
    nest.Connect(n, sr)
    nest.Simulate(T_SIM + DT)
    ev = mm.events
    steps = np.rint(np.asarray(ev['times']) / DT).astype(int) - 1
    senders = np.asarray(ev['senders']) - n[0].global_id
    n_steps = int(round(T_SIM / DT))
    V = np.full((n_steps, 2), np.nan)
    keep = steps < n_steps
    V[steps[keep], senders[keep]] = np.asarray(ev['V_m'])[keep]
    spk = sr.events
    return V, np.asarray(spk['times']), np.asarray(spk['senders']) - n[0].global_id


def our_network(delay):
    net = Network(dt=DT * u.ms)
    n = net.add(iaf_psc_alpha, 2, name='n')
    sg_ex = net.add(spike_generator, 1, params=dict(spike_times=EX * u.ms), name='sg_ex')
    sg_in = net.add(spike_generator, 1, params=dict(spike_times=IN * u.ms), name='sg_in')
    dc = net.add(dc_generator, 1, params=dict(amplitude=DC['amplitude'] * u.pA,
                                              start=DC['start'] * u.ms, stop=DC['stop'] * u.ms))
    d = delay * u.ms
    net.connect(sg_ex, n[0], port='ex', weight=300.0 * u.pA, delay=d)
    net.connect(sg_in, n[0], port='in_', weight=-200.0 * u.pA, delay=d)
    net.connect(dc, n[1], one_to_one(), port='I', delay=d)
    rec = net.observe(n, 'V_m', 'spike')
    res = net.compile('float64').runner().simulate(T_SIM * u.ms)
    V = np.asarray(res[rec]['V_m'].to_decimal(u.mV))
    t, gids, _ = res[rec].spike_events()
    return V, np.asarray(t.to_decimal(u.ms)), gids


@requires_nest
class TestNetworkDeliveryVsNest(unittest.TestCase):
    def _check(self, delay):
        V_ref, t_ref, s_ref = nest_network(delay)
        with jax.enable_x64(True):
            V, t, s = our_network(delay)
        self.assertFalse(np.isnan(V_ref).any())
        for col in range(2):
            r = compare_trace(V_ref[:, col] * u.mV, V[:, col] * u.mV, tol=tc.CAT_B)
            self.assertTrue(r.passed, f'delay {delay} neuron {col}: {r}')
        np.testing.assert_allclose(t, t_ref, atol=1e-9)
        np.testing.assert_array_equal(s, s_ref)
        # the inputs did something on both neurons
        self.assertGreater(np.ptp(V[:, 0]), 1.0)
        self.assertGreater(np.ptp(V[:, 1]), 1.0)

    def test_delay_min(self):
        self._check(0.1)

    def test_delay_1_5(self):
        self._check(1.5)

    def test_delay_off_grid(self):
        # NEST rounds 0.25 ms to 3 steps (ld_round); our network must match exactly
        self._check(0.25)
        self._check(0.35)  # float64 here; float32 rounding is covered in delays_test

    def test_delay_rounding_matches_nest(self):
        import nest
        delays = [0.1, 0.149, 0.15, 0.25, 0.35, 1.04, 1.05, 2.0]
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        nest.resolution = DT
        a = nest.Create('iaf_psc_alpha')
        b = nest.Create('iaf_psc_alpha', len(delays))
        for i, d in enumerate(delays):
            nest.Connect(a, b[i], syn_spec={'delay': d})
        got = np.asarray(nest.GetConnections(a).get('delay'))
        ours = delay_steps(delays * u.ms, DT * u.ms)
        np.testing.assert_array_equal(np.rint(got / DT).astype(int), ours)


if __name__ == '__main__':
    unittest.main()
