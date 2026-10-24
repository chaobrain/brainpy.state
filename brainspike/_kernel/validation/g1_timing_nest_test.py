# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate G1, ``conductance`` port: ``iaf_cond_exp`` PSPs through the network vs live NEST.

The other port kinds of G1 are tested where they were built: ``event`` and
``current`` in ``network_nest_test.py`` (T5), ``timed_event`` in
``models/precise_nest_test.py`` (T7); see ``validation/gates.py``.

Both sides are networks with the **same** delays: two ``spike_generator`` (excitatory
6 nS, inhibitory 8 nS; NEST routes the negative weight to ``g_in``) onto two
``iaf_cond_exp`` neurons, one subthreshold, one driven to spike by ``I_e``. Our step
``n`` ends at ``(n + 1) dt`` = the NEST multimeter sample, so ``align_steps = 0``.
``iaf_cond_exp`` integrates with ``RK(4, 2)`` (NEST: adaptive RKF45), so the trace
criterion is the ``CAT_A`` absolute bound, ``max |ΔV_m| <= 1e-3 mV`` (and ``1e-3 nS``
for the conductances), with identical spike times.
"""
from __future__ import annotations

import brainunit as u
import jax
import numpy as np
import pytest

from brainspike._kernel.models import iaf_cond_exp, spike_generator
from brainspike._kernel.network import Network
from brainspike._nest_validation.nest_compare import requires_nest

DT = 0.1
T_SIM = 200.0
I_E = [0.0, 262.0]  # pA; rheobase g_L (V_th - E_L) = 250 pA
W_EX, W_IN = 6.0, 8.0  # nS
TOL_V, TOL_G = 1e-3, 1e-3  # mV, nS


def _spike_times():
    rng = np.random.default_rng(3)
    ex = np.round(np.sort(rng.uniform(5.0, 195.0, 60)), 1)
    inh = np.round(np.sort(rng.uniform(5.0, 195.0, 20)), 1)
    return ex.tolist(), inh.tolist()


def nest_network(delay):
    import nest
    ex, inh = _spike_times()
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = DT
    n = nest.Create('iaf_cond_exp', 2, params=[{'I_e': i} for i in I_E])
    sg_ex = nest.Create('spike_generator', params={'spike_times': ex})
    sg_in = nest.Create('spike_generator', params={'spike_times': inh})
    mm = nest.Create('multimeter', params={'record_from': ['V_m', 'g_ex', 'g_in'], 'interval': DT})
    sr = nest.Create('spike_recorder')
    nest.Connect(sg_ex, n, syn_spec={'weight': W_EX, 'delay': delay})
    nest.Connect(sg_in, n, syn_spec={'weight': -W_IN, 'delay': delay})
    nest.Connect(mm, n)
    nest.Connect(n, sr)
    nest.Simulate(T_SIM + 1.0)  # past T_SIM by more than min_delay (NEST slices)
    ev = mm.events
    steps = np.rint(np.asarray(ev['times']) / DT).astype(int) - 1
    snd = np.asarray(ev['senders']) - n[0].global_id
    n_steps = int(round(T_SIM / DT))
    keep = steps < n_steps
    out = {}
    for r in ('V_m', 'g_ex', 'g_in'):
        tr = np.full((n_steps, 2), np.nan)
        tr[steps[keep], snd[keep]] = np.asarray(ev[r])[keep]
        out[r] = tr
    t = np.asarray(sr.events['times'])
    s = np.asarray(sr.events['senders']) - n[0].global_id
    keep = t <= T_SIM + DT / 2
    return out, t[keep], s[keep]


def our_network(delay):
    ex, inh = _spike_times()
    net = Network(dt=DT * u.ms)
    n = net.add(iaf_cond_exp, 2, params=dict(I_e=np.asarray(I_E) * u.pA), name='n')
    sg_ex = net.add(spike_generator, 1, params=dict(spike_times=ex * u.ms), name='sg_ex')
    sg_in = net.add(spike_generator, 1, params=dict(spike_times=inh * u.ms), name='sg_in')
    d = delay * u.ms
    net.connect(sg_ex, n, port='ex', weight=W_EX * u.nS, delay=d)
    net.connect(sg_in, n, port='in_', weight=W_IN * u.nS, delay=d)
    rec = net.observe(n, 'V_m', 'g_ex', 'g_in', 'spike')
    res = net.compile('float64').runner().simulate(T_SIM * u.ms)
    out = {'V_m': np.asarray(res[rec]['V_m'].to_decimal(u.mV)),
           'g_ex': np.asarray(res[rec]['g_ex'].to_decimal(u.nS)),
           'g_in': np.asarray(res[rec]['g_in'].to_decimal(u.nS))}
    t, gids, _ = res[rec].spike_events()
    return out, np.asarray(t.to_decimal(u.ms)), np.asarray(gids) - int(n.gids[0])


def g1_conductance_errors(delay):
    """Max abs errors (V_m mV, g nS) and spike agreement for one delay (used by the report)."""
    ref, t_ref, s_ref = nest_network(delay)
    with jax.enable_x64(True):
        ours, t, s = our_network(delay)
    err = {r: float(np.abs(ours[r] - ref[r]).max()) for r in ours}
    spikes_equal = (t.shape == t_ref.shape and np.allclose(t, t_ref, atol=1e-9)
                    and np.array_equal(s, s_ref))
    return err, spikes_equal, ref, ours, t_ref


@requires_nest
class TestConductancePortTimingVsNest:
    @pytest.mark.parametrize('delay', [0.1, 1.5, 0.25])
    def test_iaf_cond_exp_psp_through_network(self, delay):
        err, spikes_equal, ref, ours, t_ref = g1_conductance_errors(delay)
        assert not np.isnan(ref['V_m']).any()
        assert err['V_m'] <= TOL_V, err
        assert err['g_ex'] <= TOL_G and err['g_in'] <= TOL_G, err
        assert spikes_equal
        # the drive did something: the subthreshold neuron moved, the other one spiked
        assert np.ptp(ours['V_m'][:, 0]) > 1.0 and len(t_ref) >= 3
        assert ours['g_ex'].max() > W_EX * 0.5 and ours['g_in'].max() > W_IN * 0.5

    def test_one_step_shift_is_detected(self):
        # align_steps = 0 is meaningful: shifting our trace by one step breaks the bound
        ref, _, _ = nest_network(1.5)
        with jax.enable_x64(True):
            ours, _, _ = our_network(1.5)
        shifted = np.abs(ours['V_m'][1:, 0] - ref['V_m'][:-1, 0]).max()
        assert shifted > 100 * TOL_V
