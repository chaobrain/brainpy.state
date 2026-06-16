# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brette et al. 2007 (Vogels-Abbott) COBA/CUBA benchmark parity vs live NEST.

Two FACETS simulator-review benchmarks (Brette et al. 2007), both the
Vogels & Abbott (2005) sparse self-sustained E/I network (NE=3200, NI=800,
``fixed_indegree`` at ε=0.02, a brief 50 ms Poisson kick that ignites
self-sustained asynchronous-irregular activity):

* **CUBA** — current-based ``iaf_psc_exp`` (benchmark 2). Excitatory/inhibitory
  PSCs are split inside the neuron by weight sign, so it needs no seam work.
* **COBA** — conductance-based ``iaf_cond_exp`` (benchmark 1). This required a
  seam fix: ``iaf_cond_exp`` now exposes the ``n_receptors`` / ``w_by_rec``
  multi-receptor bridge so the Simulator's ``connect(receptor_type=1|2)`` routes
  excitatory input to ``g_ex`` (receptor 1) and inhibitory input to ``g_in``
  (receptor 2) — previously connections to it were silently dropped (its update
  only read the legacy ``label='w_ex'/'w_in'`` delta inputs).

Parity is the population firing-rate band (the benchmark's reported observable),
plus a single-cell V_m parity that validates the COBA routing seam against NEST.

The HH variant of this benchmark (NEST ``hh_coba.py``) already ships as
``examples/brainpy_like/106_COBA_HH_2007.py``.
"""
import unittest

import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u
import braintools

try:
    import nest
except Exception:
    nest = None

from brainpy_state import (
    Simulator, iaf_cond_exp, spike_generator, multimeter,
)
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

from examples.nest.brette_et_al_2007 import (
    simulate_coba, simulate_cuba, connection_counts, population_rate, NE, NI)

#: iaf_cond_exp parameters (NEST COBA benchmark defaults).
COND = dict(E_L=-60.0, V_th=-50.0, V_reset=-60.0, t_ref=5.0, E_ex=0.0, E_in=-80.0,
            C_m=200.0, g_L=10.0, tau_syn_ex=5.0, tau_syn_in=10.0)

#: Category C (conductance/coupled RKF45 trace) with the one-step multimeter
#: offset alignment used across the adaptive-integrator parity tests: brainpy
#: samples the t=0 initial state (dropped here), align_steps=1 absorbs the rest.
CAT_C_V = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=1, label="C",
                         note="iaf_cond_exp V_m, one-step recorder alignment")
CAT_C_G = TraceTolerance(1e-3, 1e-3, align_steps=1, label="C",
                         note="iaf_cond_exp conductance g_ex/g_in, one-step aligned")


def _drive_cond(receptor, weight_nS, t_spike=20.0, *, T=100.0, dt=0.1):
    """Drive one iaf_cond_exp with a single spike into ``receptor`` and record."""
    sim = Simulator(dt=dt * u.ms)
    nrn = sim.create(iaf_cond_exp, 1, params=dict(
        E_L=COND['E_L'] * u.mV, V_th=COND['V_th'] * u.mV, V_reset=COND['V_reset'] * u.mV,
        t_ref=COND['t_ref'] * u.ms, E_ex=COND['E_ex'] * u.mV, E_in=COND['E_in'] * u.mV,
        C_m=COND['C_m'] * u.pF, g_L=COND['g_L'] * u.nS, tau_syn_ex=COND['tau_syn_ex'] * u.ms,
        tau_syn_in=COND['tau_syn_in'] * u.ms,
        V_initializer=braintools.init.Constant(COND['E_L'] * u.mV)))
    gen = sim.create(spike_generator, 1, spike_times=[t_spike] * u.ms)
    mm = sim.create(multimeter, record_from=['V_m', 'g_ex', 'g_in'], interval=dt * u.ms)
    sim.connect(gen, nrn, receptor_type=receptor, weight=weight_nS * u.nS, delay=dt * u.ms)
    sim.connect(mm, nrn)
    res = sim.simulate(T * u.ms)
    v = np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV))[:, 0]
    g_ex = np.asarray(u.get_mantissa(res.trace(mm, 'g_ex') / u.nS))[:, 0]
    g_in = np.asarray(u.get_mantissa(res.trace(mm, 'g_in') / u.nS))[:, 0]
    return v, g_ex, g_in


class TestIafCondExpReceptorRouting(unittest.TestCase):
    """The COBA seam fix: connect(receptor_type=1|2) routes ex→g_ex, in→g_in."""

    def test_excitatory_receptor_routes_to_g_ex(self):
        v, g_ex, g_in = _drive_cond(receptor=1, weight_nS=3.0)
        self.assertGreater(g_ex.max(), 2.5)              # g_ex jumps ~ the weight (3 nS)
        self.assertLess(g_ex.max(), 3.5)
        self.assertLess(g_in.max(), 1e-9)                # g_in untouched
        self.assertGreater(v.max(), COND['E_L'] + 1.0)   # depolarizes toward E_ex = 0

    def test_inhibitory_receptor_routes_to_g_in(self):
        v, g_ex, g_in = _drive_cond(receptor=2, weight_nS=3.0)
        self.assertGreater(g_in.max(), 2.5)              # g_in jumps ~ the weight (3 nS)
        self.assertLess(g_in.max(), 3.5)
        self.assertLess(g_ex.max(), 1e-9)                # g_ex untouched
        self.assertLess(v.min(), COND['E_L'] - 0.5)      # hyperpolarizes toward E_in = -80


def _bp_cond_both(*, t_ex=20.0, w_ex=1.0, t_in=50.0, w_in=5.0, T=100.0, dt=0.1):
    """brainpy iaf_cond_exp driven subthreshold: one EPSP (receptor 1) + one IPSP
    (receptor 2). Returns the (V_m, g_ex, g_in) traces including the t=0 sample."""
    sim = Simulator(dt=dt * u.ms)
    nrn = sim.create(iaf_cond_exp, 1, params=dict(
        E_L=COND['E_L'] * u.mV, V_th=COND['V_th'] * u.mV, V_reset=COND['V_reset'] * u.mV,
        t_ref=COND['t_ref'] * u.ms, E_ex=COND['E_ex'] * u.mV, E_in=COND['E_in'] * u.mV,
        C_m=COND['C_m'] * u.pF, g_L=COND['g_L'] * u.nS, tau_syn_ex=COND['tau_syn_ex'] * u.ms,
        tau_syn_in=COND['tau_syn_in'] * u.ms,
        V_initializer=braintools.init.Constant(COND['E_L'] * u.mV)))
    gen_ex = sim.create(spike_generator, 1, spike_times=[t_ex] * u.ms)
    gen_in = sim.create(spike_generator, 1, spike_times=[t_in] * u.ms)
    mm = sim.create(multimeter, record_from=['V_m', 'g_ex', 'g_in'], interval=dt * u.ms)
    sim.connect(gen_ex, nrn, receptor_type=1, weight=w_ex * u.nS, delay=dt * u.ms)
    sim.connect(gen_in, nrn, receptor_type=2, weight=w_in * u.nS, delay=dt * u.ms)
    sim.connect(mm, nrn)
    res = sim.simulate(T * u.ms)
    return {
        'V_m': np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV))[:, 0],
        'g_ex': np.asarray(u.get_mantissa(res.trace(mm, 'g_ex') / u.nS))[:, 0],
        'g_in': np.asarray(u.get_mantissa(res.trace(mm, 'g_in') / u.nS))[:, 0],
    }


def _nest_cond_both(*, t_ex=20.0, w_ex=1.0, t_in=50.0, w_in=5.0, T=100.0, dt=0.1):
    """NEST iaf_cond_exp with the same drive. NEST splits ex/in by weight SIGN
    (positive → g_ex, negative → g_in), so the inhibitory weight is negated."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    neu = nest.Create('iaf_cond_exp', params={
        'V_m': COND['E_L'],  # NEST defaults V_m to -70; pin it to E_L to match brainpy's V_initializer
        'E_L': COND['E_L'], 'V_th': COND['V_th'], 'V_reset': COND['V_reset'],
        't_ref': COND['t_ref'], 'E_ex': COND['E_ex'], 'E_in': COND['E_in'],
        'C_m': COND['C_m'], 'g_L': COND['g_L'], 'tau_syn_ex': COND['tau_syn_ex'],
        'tau_syn_in': COND['tau_syn_in']})
    gen_ex = nest.Create('spike_generator', params={'spike_times': np.array([t_ex])})
    gen_in = nest.Create('spike_generator', params={'spike_times': np.array([t_in])})
    mm = nest.Create('multimeter', params={
        'record_from': ['V_m', 'g_ex', 'g_in'], 'interval': dt})
    nest.Connect(gen_ex, neu, syn_spec={'weight': w_ex, 'delay': dt})    # +w → g_ex
    nest.Connect(gen_in, neu, syn_spec={'weight': -w_in, 'delay': dt})   # -w → g_in
    nest.Connect(mm, neu)
    nest.Simulate(T)
    ev = mm.events
    return {k: np.asarray(ev[k]) for k in ('V_m', 'g_ex', 'g_in')}


@requires_nest
class TestIafCondExpNestParity(unittest.TestCase):
    """Single-cell V_m / g_ex / g_in parity vs live NEST through the COBA seam.

    Drives one EPSP (receptor 1) and one IPSP (receptor 2) subthreshold and checks
    the brainpy trace matches NEST's ``iaf_cond_exp`` per sample — proof the seam
    reproduces NEST-faithful conductance dynamics, not merely the right sign."""

    def test_subthreshold_Vm_and_g_match_nest(self):
        bp = _bp_cond_both()
        ns = _nest_cond_both()
        # No spikes: a clean passive + synaptic parity (validate the drive stayed subthreshold).
        self.assertLess(bp['V_m'].max(), COND['V_th'])
        # brainpy includes the t=0 initial sample; drop it, align_steps=1 absorbs
        # the remaining one-step multimeter offset (same protocol as aeif parity).
        compare_trace(ns['V_m'], bp['V_m'][1:], tol=CAT_C_V, metric='iaf_cond_exp V_m').assert_()
        compare_trace(ns['g_ex'], bp['g_ex'][1:], tol=CAT_C_G, metric='iaf_cond_exp g_ex').assert_()
        compare_trace(ns['g_in'], bp['g_in'][1:], tol=CAT_C_G, metric='iaf_cond_exp g_in').assert_()


# ----------------------------------------------------------------------------------
# Network rate-band parity (category D, distributional). Observable = the second-half
# (steady-state) self-sustained population rate; the full window is dominated by the
# ignition transient whose magnitude differs between sims. Seeds compared by mean.
NET_SEEDS = (1, 2, 3)
NET_SIMTIME = 1000.0              # shared test window (demo main() keeps NEST's 1000/10000)
#: Documented relative bands — margin over the measured late-rate gap + 3-seed noise.
COBA_BAND = TraceTolerance(0.0, 0.15, label='D', note='COBA steady-state rate (meas E 8.9%)')
CUBA_BAND = TraceTolerance(0.0, 0.12, label='D', note='CUBA steady-state rate (meas E 1.5%)')


def _nest_run_brette(model, neuron_params, w_ex, w_in, *, seed, simtime=NET_SIMTIME,
                     ne=NE, ni=NI, dt=0.1):
    """Run a Vogels-Abbott network in live NEST; return (E, I) second-half rates [Hz].

    ``model`` splits ex/in by weight sign (``iaf_cond_exp`` and ``iaf_psc_exp`` both do),
    so ``w_in`` is the signed (negative) weight. The spike recorders start at
    ``simtime/2`` to measure only the steady-state self-sustained activity."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    nest.rng_seed = seed
    ce, ci = connection_counts(ne, ni)
    nest.SetDefaults(model, neuron_params)
    E = nest.Create(model, ne)
    I = nest.Create(model, ni)
    kick = nest.Create('poisson_generator', params={'rate': 300.0, 'start': 1.0, 'stop': 51.0})
    half = simtime / 2.0
    esr = nest.Create('spike_recorder', params={'start': half})
    isr = nest.Create('spike_recorder', params={'start': half})
    nest.CopyModel('static_synapse', 'syn_ex', {'weight': w_ex, 'delay': 0.1})
    nest.CopyModel('static_synapse', 'syn_in', {'weight': w_in, 'delay': 0.1})
    nest.Connect(E, E + I, {'rule': 'fixed_indegree', 'indegree': ce}, 'syn_ex')
    nest.Connect(I, E + I, {'rule': 'fixed_indegree', 'indegree': ci}, 'syn_in')
    nest.Connect(kick, E[:50], 'all_to_all', 'syn_ex')
    nest.Connect(E[:500], esr)
    nest.Connect(I[:500], isr)
    nest.Simulate(simtime)
    return (esr.n_events / (500 * half) * 1000.0, isr.n_events / (500 * half) * 1000.0)


def _nest_coba(seed, **kw):
    return _nest_run_brette(
        'iaf_cond_exp',
        dict(V_m=-60.0, E_L=-60.0, V_th=-50.0, V_reset=-60.0, t_ref=5.0, E_ex=0.0,
             E_in=-80.0, C_m=200.0, g_L=10.0, tau_syn_ex=5.0, tau_syn_in=10.0),
        6.0, -67.0, seed=seed, **kw)


def _nest_cuba(seed, **kw):
    return _nest_run_brette(
        'iaf_psc_exp',
        dict(E_L=-49.0, V_m=-49.0, V_th=-50.0, V_reset=-60.0, C_m=200.0, tau_m=20.0,
             tau_syn_ex=5.0, tau_syn_in=10.0, t_ref=5.0),
        16.2, -139.5, seed=seed, **kw)


def _bp_seed_mean_late(sim_fn, seeds=NET_SEEDS, **kw):
    rs = [sim_fn(s, simtime=NET_SIMTIME, **kw) for s in seeds]
    return (float(np.mean([r['e_rate_late'] for r in rs])),
            float(np.mean([r['i_rate_late'] for r in rs])))


def _nest_seed_mean_late(nest_fn, seeds=NET_SEEDS):
    rs = [nest_fn(s) for s in seeds]
    return (float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs])))


@requires_nest
class TestBretteCobaParity(unittest.TestCase):
    """COBA (iaf_cond_exp) self-sustained rate matches NEST within the documented band."""

    @classmethod
    def setUpClass(cls):
        cls.bp_e, cls.bp_i = _bp_seed_mean_late(simulate_coba)
        cls.ns_e, cls.ns_i = _nest_seed_mean_late(_nest_coba)

    def test_e_rate_matches_nest(self):
        compare_trace(self.ns_e, self.bp_e, tol=COBA_BAND, metric='COBA E rate').assert_()

    def test_i_rate_matches_nest(self):
        compare_trace(self.ns_i, self.bp_i, tol=COBA_BAND, metric='COBA I rate').assert_()

    def test_self_sustained_both(self):
        # Activity persists into the second half (long after the 51 ms kick) in both sims.
        self.assertGreater(self.bp_e, 1.0)
        self.assertGreater(self.ns_e, 1.0)


@requires_nest
class TestBretteCubaParity(unittest.TestCase):
    """CUBA (iaf_psc_exp) self-sustained rate matches NEST within the documented band."""

    @classmethod
    def setUpClass(cls):
        cls.bp_e, cls.bp_i = _bp_seed_mean_late(simulate_cuba)
        cls.ns_e, cls.ns_i = _nest_seed_mean_late(_nest_cuba)

    def test_e_rate_matches_nest(self):
        compare_trace(self.ns_e, self.bp_e, tol=CUBA_BAND, metric='CUBA E rate').assert_()

    def test_i_rate_matches_nest(self):
        compare_trace(self.ns_i, self.bp_i, tol=CUBA_BAND, metric='CUBA I rate').assert_()

    def test_self_sustained_both(self):
        self.assertGreater(self.bp_e, 1.0)
        self.assertGreater(self.ns_e, 1.0)


class TestBretteNetworkCompanion(unittest.TestCase):
    """NEST-free checks: pure helpers + the network builds, is active, and is deterministic.

    Uses a tiny ne=200/ni=50, simtime=150 ms config (~9 s/run) so CI without NEST still
    exercises the COBA receptor seam and the CUBA sign-routing end to end."""

    SMALL = dict(ne=200, ni=50, simtime=150.0)

    def test_connection_counts(self):
        self.assertEqual(connection_counts(3200, 800, 0.02), (64, 16))
        self.assertEqual(connection_counts(800, 200, 0.02), (16, 4))

    def test_population_rate_formula(self):
        raster = np.zeros((1000, 2)); raster[::100] = 1   # 10 spikes/neuron over 100 ms
        self.assertAlmostEqual(population_rate(raster, 2, 100.0), 100.0)

    def test_coba_active_and_deterministic(self):
        a = simulate_coba(7, **self.SMALL)
        b = simulate_coba(7, **self.SMALL)
        self.assertGreater(a['e_rate'], 0.0)              # the kick ignites activity
        self.assertTrue(np.isfinite(a['e_rate']) and np.isfinite(a['i_rate']))
        self.assertEqual(a['e_rate'], b['e_rate'])        # same seed -> identical
        self.assertEqual(a['i_rate'], b['i_rate'])

    def test_cuba_active_and_deterministic(self):
        a = simulate_cuba(7, **self.SMALL)
        b = simulate_cuba(7, **self.SMALL)
        self.assertGreater(a['e_rate'], 0.0)
        self.assertTrue(np.isfinite(a['e_rate']) and np.isfinite(a['i_rate']))
        self.assertEqual(a['e_rate'], b['e_rate'])
        self.assertEqual(a['i_rate'], b['i_rate'])


if __name__ == '__main__':
    unittest.main()
