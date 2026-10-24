# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Single-neuron parity of the K0 models against live NEST (``requires_nest``).

Alignment (documented, no search): our step ``n`` ends at ``(n + 1) dt``, the NEST
multimeter sample at ``(n + 1) dt`` is the state after the same step, and a spike
emitted in step ``n`` has the NEST stamp ``(n + 1) dt``. Spike input is injected
in NEST by a ``spike_generator`` with ``delay = dt`` (the minimal delay); a spike
at ``T`` then arrives at ``T + dt`` and NEST applies it in the step ending at
``T + dt``, i.e. our step ``round((T + dt) / dt) - 1``, where we give it to the
``event`` port. DC input uses ``I_e`` (acts immediately on both sides) or a NEST
``dc_generator`` vs our :class:`dc_generator` routed with a one-step delay into
the latched ``current`` port. The network-level delivery gate (G1,
``align_steps=0`` through the T5 runner) is T10's.

Run as a script (``python -m brainspike._kernel.models.nest_parity_test``) to
print the G5 error-vs-substeps table.
"""
from __future__ import annotations

import unittest

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.integrators import RK
from brainspike._kernel.models import (
    AEIF_COND_ALPHA_SUBSTEPS,
    IAF_COND_EXP_SUBSTEPS,
    aeif_cond_alpha,
    dc_generator,
    iaf_cond_exp,
    iaf_psc_alpha,
    iaf_psc_exp,
    spike_generator,
)
from brainspike._kernel.models._testing import rollout, step_inputs
from brainspike._nest_validation import tolerance_conventions as tc
from brainspike._nest_validation.nest_compare import compare_trace, requires_nest

DT = 0.1
SUBSTEPS = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------- NEST side

def nest_run(model, params, t_sim, *, spikes=(), dc=None, record=('V_m',), dt=DT):
    """Simulate one NEST neuron; returns ``(trace dict (n_steps,), spike times)``.

    ``spikes`` is a list of ``(times, weight)``; each becomes a ``spike_generator``
    connected with ``delay = dt``. ``dc`` is a ``dc_generator`` parameter dict
    connected with ``delay = dt``.
    """
    import nest
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    n = nest.Create(model, params=params)
    mm = nest.Create('multimeter', params={'record_from': list(record), 'interval': dt})
    sr = nest.Create('spike_recorder')
    nest.Connect(mm, n)
    nest.Connect(n, sr)
    for times, w in spikes:
        sg = nest.Create('spike_generator', params={'spike_times': list(times)})
        nest.Connect(sg, n, syn_spec={'weight': w, 'delay': dt})
    if dc is not None:
        g = nest.Create('dc_generator', params=dc)
        nest.Connect(g, n, syn_spec={'delay': dt})
    nest.Simulate(t_sim + dt)  # the sample at t_sim is written one step later
    ev = mm.events
    idx = np.rint(np.asarray(ev['times']) / dt).astype(int) - 1
    n_steps = int(round(t_sim / dt))
    out = {}
    for r in record:
        tr = np.full(n_steps, np.nan)
        tr[idx[idx < n_steps]] = np.asarray(ev[r])[idx < n_steps]
        out[r] = tr
    spk = np.asarray(sr.events['times'])
    return out, spk[spk <= t_sim + dt / 2]


# ---------------------------------------------------------------- our side

def _arrival_step(t, dt=DT):
    """Our step receiving a NEST spike_generator spike at ``t`` sent with delay ``dt``."""
    return int(round((t + dt) / dt)) - 1


def ours_run(cls, params, t_sim, *, spikes=(), port_of=None, record=('V_m',), dt=DT,
             integrator=None, current=None, precision='float64'):
    """Run our model; ``spikes`` as for :func:`nest_run`; ``current`` is a ``(n_steps,)`` pA payload."""
    n_steps = int(round(t_sim / dt))
    kw = {} if integrator is None else dict(integrator=integrator)
    m = cls(1, params=params, precision=precision, **kw)
    events = {}
    for times, w in spikes:
        port, unit, value = port_of(w)
        steps = [_arrival_step(t, dt) for t in times]
        prev = events.get(port, ([], [], unit))
        events[port] = (prev[0] + steps, prev[1] + [value] * len(steps), unit)
    inputs = step_inputs(n_steps, 1, events)
    if current is not None:
        inputs['I'] = u.Quantity(jnp.asarray(current)[:, None], unit=u.pA)
    out, tr = rollout(m, n_steps, dt * u.ms, inputs, record=record)
    trace = {r: np.asarray(u.get_mantissa(tr[r]))[:, 0] for r in record}
    spk = np.asarray(out['spike'])[:, 0]
    steps = np.nonzero(spk)[0]
    times = np.repeat((steps + 1) * dt, spk[steps].astype(int))
    return trace, times


def _psc_port(w):
    # NEST routes by sign; the PSC weight keeps its sign (pA)
    return ('ex' if w >= 0 else 'in_'), u.pA, w


def _cond_port(w):
    # NEST routes by sign; the conductance jump is |w| (nS)
    return ('ex' if w >= 0 else 'in_'), u.nS, abs(w)


def _cond_spikes():
    rng = np.random.default_rng(3)
    ex = np.round(np.sort(rng.uniform(5.0, 195.0, 60)), 1)
    inh = np.round(np.sort(rng.uniform(5.0, 195.0, 20)), 1)
    return [(ex, 6.0), (inh, -8.0)]


def _psc_spikes():
    return [([10.0, 10.0, 25.3, 40.0, 41.2, 70.0], 250.0), ([30.0, 55.5], -400.0)]


SCENARIOS = {
    # name: (NEST model, our class, params, t_sim, spikes, port_of)
    'iaf_cond_exp/psp': ('iaf_cond_exp', iaf_cond_exp, {}, 200.0, _cond_spikes(), _cond_port),
    'iaf_cond_exp/dc': ('iaf_cond_exp', iaf_cond_exp, {'I_e': 400.0}, 200.0, (), _cond_port),
    'aeif_cond_alpha/psp': ('aeif_cond_alpha', aeif_cond_alpha, {}, 200.0, _cond_spikes(), _cond_port),
    'aeif_cond_alpha/dc': ('aeif_cond_alpha', aeif_cond_alpha, {'I_e': 800.0}, 200.0, (), _cond_port),
}


def _with_units(cls, params):
    return {k: v * cls.params[k].unit for k, v in params.items()}


def spike_mask(V_ref, V, V_th):
    """Steps in a spike upswing (either trace above ``V_th``) and the step after.

    Fixed-substep integration locates the threshold crossing to within one
    substep (first order), so the reset and the steep upswing differ from NEST's
    adaptive RKF45 by ``O(h)`` in time; G5 reports the error with these steps
    excluded separately.
    """
    up = (V_ref > V_th) | (V > V_th)
    return up | np.roll(up, 1)


def g5_table(substeps=SUBSTEPS, scenarios=None):
    """Error vs NEST (mV) per scenario and substep count, plus spike agreement."""
    rows = []
    for name in scenarios or SCENARIOS:
        nm, cls, params, t_sim, spikes, port_of = SCENARIOS[name]
        ref, ref_spk = nest_run(nm, params, t_sim, spikes=spikes)
        V_th = params.get('V_th', float(cls.params['V_th'].default.mantissa))
        for k in substeps:
            with jax.enable_x64(True):
                tr, spk = ours_run(cls, _with_units(cls, params), t_sim, spikes=spikes,
                                   port_of=port_of, integrator=RK(4, k))
            err = np.abs(tr['V_m'] - ref['V_m'])
            mask = spike_mask(ref['V_m'], tr['V_m'], V_th)
            rows.append(dict(
                scenario=name, substeps=k, err=float(err.max()),
                err_masked=float(err[~mask].max()),
                cat_a=compare_trace(ref['V_m'], tr['V_m'], tol=tc.CAT_A).passed,
                n_spikes=(len(spk), len(ref_spk)),
                spikes_equal=len(spk) == len(ref_spk) and bool(np.allclose(spk, ref_spk))))
    return rows


def format_table(rows):
    lines = ['| scenario | substeps | max abs err (mV) | excl. spike steps (mV) | CAT_A full trace | '
             'spikes ours/NEST | spike times equal |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['substeps']} | {r['err']:.3e} | {r['err_masked']:.3e} | "
                     f"{'pass' if r['cat_a'] else 'FAIL'} | {r['n_spikes'][0]}/{r['n_spikes'][1]} | "
                     f"{r['spikes_equal']} |")
    return '\n'.join(lines)


# ---------------------------------------------------------------- tests

@requires_nest
class TestIafPscAlphaVsNest(unittest.TestCase):
    """Exact model: CAT_B (1e-6 mV) with align_steps = 0."""

    def _check(self, params, spikes=(), t_sim=100.0, record=('V_m', 'I_syn_ex', 'I_syn_in'),
               current=None, dc=None):
        ref, ref_spk = nest_run('iaf_psc_alpha', params, t_sim, spikes=spikes, dc=dc, record=record)
        with jax.enable_x64(True):
            tr, spk = ours_run(iaf_psc_alpha, _with_units(iaf_psc_alpha, params), t_sim,
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
        # duplicate times (two events in one step), mixed signs, ex and in kernels differ
        ref, spk = self._check({'tau_syn_ex': 0.5, 'tau_syn_in': 3.0}, spikes=_psc_spikes())
        self.assertEqual(len(spk), 0)
        self.assertGreater(np.max(ref['V_m']), -69.0)

    def test_psp_with_tau_m_equal_tau_syn(self):
        self._check({'tau_syn_ex': 10.0, 'tau_syn_in': 10.0}, spikes=_psc_spikes())

    def test_spiking_with_input_and_off_grid_refractory(self):
        # t_ref = 0.25 ms: NEST rounds up to 3 steps
        ex = (np.arange(5.0, 95.0, 0.7).round(1), 600.0)
        _, spk = self._check({'t_ref': 0.25, 'I_e': 200.0}, spikes=[ex])
        self.assertGreater(len(spk), 3)

    def test_dc_generator_onset_through_current_latch(self):
        """NEST dc_generator (delay dt) == our dc_generator -> 1-step delay -> latched port."""
        t_sim = 30.0
        n_steps = int(round(t_sim / DT))
        dc = dc_generator(1, params=dict(amplitude=300.0 * u.pA, start=5.0 * u.ms, stop=20.0 * u.ms))
        out, _ = rollout(dc, n_steps, DT * u.ms)
        emitted = np.asarray(out['I'].to_decimal(u.pA))[:, 0]
        delivered = np.concatenate([[0.0], emitted[:-1]])  # projection delay: one step
        self._check({}, t_sim=t_sim, current=delivered,
                    dc={'amplitude': 300.0, 'start': 5.0, 'stop': 20.0})


@requires_nest
class TestIafPscExpVsNest(unittest.TestCase):
    """Exact model: CAT_B (1e-6 mV) with align_steps = 0 (promoted from the K1a-G4 microcircuit)."""

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


@requires_nest
class TestConductanceModelsVsNest(unittest.TestCase):
    """Fixed-substep RK4 at the default substeps vs NEST RKF45."""

    def _run(self, name):
        nm, cls, params, t_sim, spikes, port_of = SCENARIOS[name]
        ref, ref_spk = nest_run(nm, params, t_sim, spikes=spikes)
        with jax.enable_x64(True):
            tr, spk = ours_run(cls, _with_units(cls, params), t_sim, spikes=spikes, port_of=port_of)
        np.testing.assert_allclose(spk, ref_spk, atol=1e-9)  # identical spike steps
        return ref['V_m'], tr['V_m'], ref_spk

    def test_iaf_cond_exp_psp(self):
        self.assertEqual(iaf_cond_exp.integrator, RK(4, IAF_COND_EXP_SUBSTEPS))
        V_ref, V, _ = self._run('iaf_cond_exp/psp')
        compare_trace(V_ref, V, tol=tc.CAT_A).assert_()
        self.assertLessEqual(float(np.max(np.abs(V - V_ref))), 1e-3)

    def test_iaf_cond_exp_dc_spiking(self):
        V_ref, V, spk = self._run('iaf_cond_exp/dc')
        self.assertGreater(len(spk), 5)
        self.assertLessEqual(float(np.max(np.abs(V - V_ref))), 1e-3)

    def test_aeif_psp(self):
        self.assertEqual(aeif_cond_alpha.integrator, RK(4, AEIF_COND_ALPHA_SUBSTEPS))
        V_ref, V, _ = self._run('aeif_cond_alpha/psp')
        self.assertLessEqual(float(np.max(np.abs(V - V_ref))), 1e-3)

    def test_aeif_dc_spiking(self):
        """G5 for spiking aeif: spike times equal NEST; CAT_A off the upswing/reset steps.

        On the upswing / reset steps the O(h) crossing location gives errors of
        order 1 mV (see the G5 table): the full-trace CAT_A criterion is not met.
        """
        V_ref, V, spk = self._run('aeif_cond_alpha/dc')  # spike times equal (same step)
        self.assertGreater(len(spk), 3)
        # G5 (K0 spec §6): CAT_A on every step except the upswing (either trace above
        # V_th) and the step after (the reset)
        mask = spike_mask(V_ref, V, -50.4)
        compare_trace(V_ref[~mask], V[~mask], tol=tc.CAT_A, metric='aeif V_m off spikes').assert_()

    def test_g5_convergence(self):
        rows = g5_table()
        print('\n' + format_table(rows))
        err = {(r['scenario'], r['substeps']): r for r in rows}
        # subthreshold: 4th order until NEST's own error floor
        for name in ('iaf_cond_exp/psp', 'aeif_cond_alpha/psp'):
            self.assertGreater(err[(name, 1)]['err'] / err[(name, 2)]['err'], 8.0)
        # spiking aeif: the error off the spike steps shrinks with the substeps
        self.assertLess(err[('aeif_cond_alpha/dc', 16)]['err_masked'],
                        err[('aeif_cond_alpha/dc', 4)]['err_masked'])


@requires_nest
class TestSpikeGeneratorMultiplicityVsNest(unittest.TestCase):
    def test_duplicates_deliver_counts(self):
        """Three spikes at 2.0 ms: NEST delivers three events; ours a count of 3."""
        times = [2.0, 2.0, 2.0, 4.0]
        ref, _ = nest_run('iaf_psc_alpha', {}, 10.0, spikes=[(times, 100.0)],
                          record=('V_m', 'I_syn_ex'))
        n_steps = 100
        sg = spike_generator(1, params=dict(spike_times=times * u.ms))
        out, _ = rollout(sg, n_steps, DT * u.ms)
        counts = np.asarray(out['spike'])[:, 0]
        self.assertEqual(counts[19], 3.0)
        self.assertEqual(counts[39], 1.0)
        delivered = np.concatenate([[0.0], counts[:-1]]) * 100.0  # delay = one step
        with jax.enable_x64(True):
            neu = iaf_psc_alpha(1, precision='float64')
            _, tr = rollout(neu, n_steps, DT * u.ms,
                            {'ex': u.Quantity(jnp.asarray(delivered)[:, None], unit=u.pA)},
                            record=('V_m', 'I_syn_ex'))
        compare_trace(ref['V_m'], np.asarray(tr['V_m'].to_decimal(u.mV))[:, 0], tol=tc.CAT_B).assert_()
        np.testing.assert_allclose(np.asarray(tr['I_syn_ex'].to_decimal(u.pA))[:, 0], ref['I_syn_ex'],
                                   atol=1e-9)


if __name__ == '__main__':  # pragma: no cover
    print(format_table(g5_table()))
