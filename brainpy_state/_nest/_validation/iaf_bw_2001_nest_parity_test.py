# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST single-cell parity for ``iaf_bw_2001`` (Wang 2002 decision neuron).

The existing ``brainpy_state/_nest/iaf_bw_2001_test.py`` validates the update against
a *Python twin* of NEST's algorithm. This module closes the remaining gap: a direct
cross-check against the **live** NEST ``iaf_bw_2001`` C++ model, on two fronts —

* **AMPA + GABA (single neuron).** Both Ohmic receptors are driven from spike
  generators (receptor 1 = AMPA, 2 = GABA) and ``V_m`` / ``s_AMPA`` / ``s_GABA`` /
  ``I_AMPA`` / ``I_GABA`` are matched per sample. brainpy and NEST agree to machine
  precision with the *direct* alignment ``bp[i] == nest[i]`` (no recorder-offset
  drop — unlike the RKF45 cond models, ``iaf_bw_2001``'s sample stream lines up with
  NEST's multimeter exactly).

* **NMDA (two neurons).** NEST forbids driving the NMDA receptor (3) from a plain
  spike generator — the spike must carry a presynaptic ``spike_offset`` computed by an
  ``iaf_bw_2001`` *sender*. A sender is forced to fire (strong AMPA drive) and projects
  NMDA onto a receiver; brainpy reproduces this feed-forward by running the sender,
  reading its per-step ``spike_offset``, and depositing ``weight * offset`` onto the
  receiver's NMDA gate one delay step later. The receiver's ``s_NMDA`` / ``I_NMDA`` /
  ``V_m`` match NEST to machine precision — validating the full NMDA jump approximation
  (the ``k0``/``k1`` constants, ``s_NMDA_pre`` recurrence, and Mg2+ block).

Together these establish that the neuron — including its distinctive slow,
voltage-gated NMDA coupling — is faithful to NEST, the prerequisite for any
Wang-style decision-making network built on it.
"""
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import saiunit as u
import braintools

try:
    import nest
except Exception:
    nest = None

from brainpy_state import iaf_bw_2001
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import TraceTolerance

#: Wang/Brunel-style neuron parameters shared by both sides.
BW = dict(E_L=-70.0, E_ex=0.0, E_in=-70.0, V_th=-55.0, V_reset=-60.0, C_m=500.0,
          g_L=25.0, t_ref=2.0, tau_AMPA=2.0, tau_GABA=5.0, tau_decay_NMDA=100.0,
          tau_rise_NMDA=2.0, alpha=0.5, conc_Mg2=1.0)

#: Category C-class tolerances; the measured agreement is exact (0.0), so these give
#: large margin while staying robust across NEST builds. ``align_steps=0`` — the
#: ``iaf_bw_2001`` sample stream aligns with NEST's multimeter directly (bp[i]==nest[i]).
BW_V = TraceTolerance(1e-3 * u.mV, 1e-3, label='C', note='iaf_bw_2001 V_m vs live NEST')
BW_S = TraceTolerance(1e-3, 1e-3, label='C', note='iaf_bw_2001 gating/current vs live NEST')

DT = 0.1


def _bp_neuron():
    """Construct a brainpy ``iaf_bw_2001`` with the shared BW parameters."""
    return iaf_bw_2001(
        1, E_L=BW['E_L'] * u.mV, E_ex=BW['E_ex'] * u.mV, E_in=BW['E_in'] * u.mV,
        V_th=BW['V_th'] * u.mV, V_reset=BW['V_reset'] * u.mV, C_m=BW['C_m'] * u.pF,
        g_L=BW['g_L'] * u.nS, t_ref=BW['t_ref'] * u.ms, tau_AMPA=BW['tau_AMPA'] * u.ms,
        tau_GABA=BW['tau_GABA'] * u.ms, tau_decay_NMDA=BW['tau_decay_NMDA'] * u.ms,
        tau_rise_NMDA=BW['tau_rise_NMDA'] * u.ms, alpha=BW['alpha'] / u.ms,
        conc_Mg2=BW['conc_Mg2'] * u.mM,
        V_initializer=braintools.init.Constant(BW['E_L'] * u.mV))


def _nest_neuron():
    n = nest.Create('iaf_bw_2001', params={**BW, 'V_m': BW['E_L']})
    return n


# --- single neuron, AMPA + GABA ---------------------------------------------------

def _nest_single(ampa_t, w_ampa, gaba_t, w_gaba, *, T, dt=DT):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    nrn = _nest_neuron()
    ga = nest.Create('spike_generator', params={'spike_times': np.array([ampa_t])})
    gg = nest.Create('spike_generator', params={'spike_times': np.array([gaba_t])})
    mm = nest.Create('multimeter', params={
        'record_from': ['V_m', 's_AMPA', 's_GABA', 'I_AMPA', 'I_GABA'], 'interval': dt})
    nest.Connect(ga, nrn, syn_spec={'receptor_type': 1, 'weight': w_ampa, 'delay': dt})
    nest.Connect(gg, nrn, syn_spec={'receptor_type': 2, 'weight': w_gaba, 'delay': dt})
    nest.Connect(mm, nrn)
    nest.Simulate(T)
    ev = mm.events
    return {k: np.asarray(ev[k]) for k in ('V_m', 's_AMPA', 's_GABA', 'I_AMPA', 'I_GABA')}


def _bp_single(ampa_t, w_ampa, gaba_t, w_gaba, *, T, dt=DT):
    n = int(round(T / dt))
    ka, kg = int(round(ampa_t / dt)), int(round(gaba_t / dt))
    with brainstate.environ.context(dt=dt * u.ms):
        nrn = _bp_neuron()
        nrn.init_state()

        def body(k):
            nrn.add_delta_input('a', jnp.where(k == ka, w_ampa, 0.0) * u.nS, label='AMPA')
            nrn.add_delta_input('g', jnp.where(k == kg, w_gaba, 0.0) * u.nS, label='GABA')
            with brainstate.environ.context(t=k * dt * u.ms):
                nrn.update(x=0.0 * u.pA)
            return (nrn.V.value / u.mV, nrn.s_AMPA.value / u.nS, nrn.s_GABA.value / u.nS,
                    nrn.I_AMPA.value / u.pA, nrn.I_GABA.value / u.pA)

        r = brainstate.transform.for_loop(body, jnp.arange(n))
    keys = ('V_m', 's_AMPA', 's_GABA', 'I_AMPA', 'I_GABA')
    return {k: np.asarray(r[i][:, 0]) for i, k in enumerate(keys)}


# --- two neurons, NMDA ------------------------------------------------------------

def _nest_two_neuron_nmda(drive_times, w_drive, w_nmda, *, T, dt=DT):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    snd = _nest_neuron()
    rcv = _nest_neuron()
    drv = nest.Create('spike_generator', params={'spike_times': np.array(drive_times)})
    mm = nest.Create('multimeter', params={
        'record_from': ['s_NMDA', 'I_NMDA', 'V_m'], 'interval': dt})
    nest.Connect(drv, snd, syn_spec={'receptor_type': 1, 'weight': w_drive, 'delay': dt})
    nest.Connect(snd, rcv, syn_spec={'receptor_type': 3, 'weight': w_nmda, 'delay': dt})
    nest.Connect(mm, rcv)
    nest.Simulate(T)
    ev = mm.events
    return {k: np.asarray(ev[k]) for k in ('s_NMDA', 'I_NMDA', 'V_m')}


def _bp_two_neuron_nmda(drive_times, w_drive, w_nmda, *, T, dt=DT):
    n = int(round(T / dt))
    dsteps = jnp.array(sorted({int(round(t / dt)) for t in drive_times}))
    with brainstate.environ.context(dt=dt * u.ms):
        snd = _bp_neuron()
        snd.init_state()

        def sbody(k):
            aw = jnp.where(jnp.any(k == dsteps), w_drive, 0.0)
            snd.add_delta_input('a', aw * u.nS, label='AMPA')
            with brainstate.environ.context(t=k * dt * u.ms):
                spk = snd.update(x=0.0 * u.pA)
            return spk[0], snd.spike_offset.value[0]

        spk_arr, off_arr = brainstate.transform.for_loop(sbody, jnp.arange(n))
        spk_arr = np.asarray(spk_arr)
        off_arr = np.asarray(off_arr)
        # NMDA arrives one delay step (dt) after the sender fires: weight * offset.
        nmda_in = np.zeros(n)
        nmda_in[1:] = w_nmda * off_arr[:-1] * (spk_arr[:-1] > 0)
        nmda_in = jnp.asarray(nmda_in)

        rcv = _bp_neuron()
        rcv.init_state()

        def rbody(k):
            rcv.add_delta_input('n', nmda_in[k] * u.nS, label='NMDA')
            with brainstate.environ.context(t=k * dt * u.ms):
                rcv.update(x=0.0 * u.pA)
            return rcv.s_NMDA.value / u.nS, rcv.I_NMDA.value / u.pA, rcv.V.value / u.mV

        rr = brainstate.transform.for_loop(rbody, jnp.arange(n))
    return {k: np.asarray(rr[i][:, 0]) for i, k in enumerate(('s_NMDA', 'I_NMDA', 'V_m'))}


@requires_nest
class TestIafBw2001SingleCellNestParity(unittest.TestCase):
    """Single ``iaf_bw_2001``: AMPA + GABA drive matches live NEST per sample."""

    def test_ampa_gaba_Vm_and_gating_match_nest(self):
        kw = dict(ampa_t=10.0, w_ampa=40.0, gaba_t=20.0, w_gaba=15.0, T=60.0)
        bp = _bp_single(**kw)
        ns = _nest_single(**kw)
        n = min(len(ns['V_m']), len(bp['V_m']))
        # Sanity: the AMPA EPSP actually moved V_m (not a trivially-flat match).
        self.assertGreater(ns['V_m'][:n].max(), BW['E_L'] + 1.0)
        compare_trace(ns['V_m'][:n], bp['V_m'][:n], tol=BW_V, metric='bw2001 V_m').assert_()
        for g in ('s_AMPA', 's_GABA', 'I_AMPA', 'I_GABA'):
            compare_trace(ns[g][:n], bp[g][:n], tol=BW_S, metric=f'bw2001 {g}').assert_()


@requires_nest
class TestIafBw2001NmdaNestParity(unittest.TestCase):
    """Two ``iaf_bw_2001``: the NMDA presynaptic-offset coupling matches live NEST."""

    def test_two_neuron_nmda_matches_nest(self):
        kw = dict(drive_times=[10.0, 11.0, 12.0, 30.0, 31.0, 32.0],
                  w_drive=350.0, w_nmda=1.2, T=80.0)
        bp = _bp_two_neuron_nmda(**kw)
        ns = _nest_two_neuron_nmda(**kw)
        n = min(len(ns['s_NMDA']), len(bp['s_NMDA']))
        # Sanity: NMDA actually accumulated on the receiver.
        self.assertGreater(ns['s_NMDA'][:n].max(), 0.5)
        compare_trace(ns['s_NMDA'][:n], bp['s_NMDA'][:n], tol=BW_S, metric='bw2001 s_NMDA').assert_()
        compare_trace(ns['I_NMDA'][:n], bp['I_NMDA'][:n], tol=BW_S, metric='bw2001 I_NMDA').assert_()
        compare_trace(ns['V_m'][:n], bp['V_m'][:n], tol=BW_V, metric='bw2001 NMDA V_m').assert_()


if __name__ == '__main__':
    unittest.main()
