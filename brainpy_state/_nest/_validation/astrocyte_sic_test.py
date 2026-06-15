# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for the neuron<->astrocyte SIC loop (goal 15d-astro).

The bidirectional tripartite loop on the JAX substrate, validated against live
NEST 3.9.0 along three pillars (each pillar pairs a NEST-free law that always
runs with a ``@requires_nest`` per-sample/seed parity):

* **SIC-response micro-parity** -- a deterministic ``spike_generator`` drives one
  ``astrocyte_lr_1994`` (IP3 -> Ca -> SIC); the SIC reaches a postsynaptic
  ``aeif_cond_alpha_astro`` through one ``sic_connection``. Compares the IP3, Ca,
  and delivered ``I_SIC`` traces.
* **Neuron<->astro loop** -- a current-driven ``aeif_cond_alpha_astro`` spikes
  into the astrocyte (delta -> IP3), whose Ca crosses the SIC threshold and feeds
  ``I_SIC`` back to a downstream neuron. Compares IP3/Ca/I_SIC plus the driver V_m.
* **Astro-network distributional** -- a population of current-driven neurons each
  receives ``I_SIC`` from a Poisson-driven astrocyte. Compares the seed-mean post
  firing rate (category D) and asserts the qualitative law that switching the SIC
  arm on raises mean firing.

Substrate timing. NEST's ``sic_connection`` default ``delay`` is ``1.0 ms`` =
``delay_steps=10`` at ``dt=0.1 ms``; the brainpy spec's ``delay_steps=10`` rides
the same 10-step latency (I_SIC onsets at step 10 with value
``ln((Ca-SIC_th)*1000)``). The neuron->astro delta path and the spike->IP3 path
carry the substrate's intrinsic one-step pipeline lag, absorbed by ``align_steps``
in :func:`~brainpy_state._nest._validation.nest_compare.compare_trace`.

Realized aligned ``max|Δ|`` (measured, recorded in the cluster spec §8): IP3
``2.4e-5``, Ca ``1.9e-4``, I_SIC ``2.3e-4`` (micro); IP3 ``0.0``, Ca ``9.9e-7``,
I_SIC ``6.0e-4``, V_pre ``3.7e-4 mV`` (loop); seed-mean rate ``9.0=9.0`` (off),
``14.0=14.0`` (on) (network).
"""
import gc
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:                                         # pragma: no cover - env dependent
    nest = None

from brainpy_state import (Simulator, aeif_cond_alpha_astro, astrocyte_lr_1994,
                           sic_connection, multimeter, spike_generator,
                           spike_recorder, poisson_generator, one_to_one)
from brainpy_state._nest._validation.nest_compare import (
    requires_nest, compare_trace, compare_distributional)
from brainpy_state._nest._validation.tolerance_conventions import (
    TraceTolerance, CAT_A, CAT_D)

DT = 0.1
#: NEST sic_connection default delay (1.0 ms) -> 10 substrate steps at dt=0.1.
SIC_DELAY_STEPS = 10

#: Astrocyte RKF45 state / delivered-current trace tolerance. IP3/Ca are
#: dimensionless (µM), I_SIC is pA, so atol is a plain float (not unit-bound mV).
#: align_steps=3 absorbs the spike->IP3 + sic-delivery integer pipeline offset.
ASTRO_TOL = TraceTolerance(1e-3, 1e-3, align_steps=3, label='A',
                           note='astrocyte IP3/Ca/I_SIC vs live NEST (one-step pipeline align)')


def _ms(x):
    """Strip units to a flat float64 ndarray (a recorded trace mantissa)."""
    return np.asarray(u.get_mantissa(x), dtype=float).reshape(-1)


# --- Pillar 1: SIC-response micro-parity -------------------------------------------

#: Deterministic spike train into the astrocyte (ms); each spike adds ΔIP3·w to IP3.
SPK_TIMES = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
SPK_WEIGHT = 2.0
MICRO_T = 60.0


def _bp_sic_response():
    """brainpy: spike_generator -> astro -> sic -> post; (IP3, Ca, I_SIC) traces."""
    sim = Simulator(dt=DT * u.ms)
    astro = sim.create(astrocyte_lr_1994, 1,
                       params={'delta_IP3': 0.5, 'IP3_initializer': 1.0,
                               'Ca_initializer': 1.0, 'h_IP3R_initializer': 1.0})
    post = sim.create(aeif_cond_alpha_astro, 1)
    sg = sim.create(spike_generator, 1, spike_times=np.asarray(SPK_TIMES) * u.ms)
    sim.connect(sg, astro, weight=SPK_WEIGHT, delay=DT * u.ms)
    sim.connect(astro, post,
                synapse=sic_connection(weight=1.0, delay_steps=SIC_DELAY_STEPS))
    mm_a = sim.create(multimeter, record_from=['IP3', 'Ca'])
    mm_p = sim.create(multimeter, record_from=['I_SIC'])
    sim.connect(mm_a, astro)
    sim.connect(mm_p, post)
    res = sim.simulate(MICRO_T * u.ms)
    return (_ms(res.trace(mm_a, 'IP3')), _ms(res.trace(mm_a, 'Ca')),
            _ms(res.trace(mm_p, 'I_SIC')))


def _nest_sic_response():
    """NEST: spike_generator -> astro -> sic -> post; (IP3, Ca, I_SIC) traces."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT})
    astro = nest.Create('astrocyte_lr_1994',
                        params={'delta_IP3': 0.5, 'IP3': 1.0, 'Ca_astro': 1.0, 'h_IP3R': 1.0})
    post = nest.Create('aeif_cond_alpha_astro', 1)
    sg = nest.Create('spike_generator',
                     params={'spike_times': SPK_TIMES, 'spike_weights': [SPK_WEIGHT] * len(SPK_TIMES)})
    mm_a = nest.Create('multimeter', params={'record_from': ['IP3', 'Ca_astro'], 'interval': DT})
    mm_p = nest.Create('multimeter', params={'record_from': ['I_SIC'], 'interval': DT})
    nest.Connect(sg, astro, syn_spec={'weight': 1.0, 'delay': DT})
    nest.Connect(astro, post, syn_spec={'synapse_model': 'sic_connection', 'weight': 1.0})
    nest.Connect(mm_a, astro)
    nest.Connect(mm_p, post)
    nest.Simulate(MICRO_T)
    return (np.asarray(mm_a.events['IP3'], dtype=float),
            np.asarray(mm_a.events['Ca_astro'], dtype=float),
            np.asarray(mm_p.events['I_SIC'], dtype=float))


class TestAstrocyteSICResponseLaw(unittest.TestCase):
    """SIC-response invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        # Each builder is a fresh trace+compile; bound the JAX cache per test.
        jax.clear_caches()
        gc.collect()

    def test_sic_threshold_and_onset(self):
        """I_SIC is zero until step 10 (the 10-step sic delay), then ``ln((Ca-θ)·1000)``.

        With Ca initialised at 1.0 µM (>> SIC_th = 0.19669), the astrocyte emits a
        graded SIC every step; the ``delay_steps=10`` connection delivers the first
        nonzero current at step 10, matching ``ln((1.0-0.19669)*1000) ≈ 6.6887``.
        """
        _ip3, _ca, isic = _bp_sic_response()
        nz = np.flatnonzero(isic > 0.0)
        self.assertTrue(nz.size > 0, 'SIC must be delivered for Ca above threshold')
        self.assertEqual(int(nz[0]), SIC_DELAY_STEPS,
                         'first nonzero I_SIC lands at the sic-connection delay')
        self.assertAlmostEqual(float(isic[nz[0]]),
                               float(np.log((1.0 - 0.19669) * 1000.0)), places=2)

    def test_spikes_raise_ip3_monotone_then_decay(self):
        """Each presynaptic spike steps IP3 up; IP3 relaxes back after the train.

        Six spikes of weight 2 with ΔIP3 = 0.5 add ~1.0 to IP3 each, so IP3 climbs
        from its 1.0 baseline to ~7 during the train, then decays toward IP3_0.
        """
        ip3, _ca, _isic = _bp_sic_response()
        self.assertAlmostEqual(float(ip3[0]), 1.0, places=2)
        peak = int(np.argmax(ip3))
        self.assertGreater(float(ip3[peak]), 6.0, 'IP3 climbs with the spike train')
        self.assertLess(float(ip3[-1]), float(ip3[peak]), 'IP3 relaxes after the train')
        self.assertTrue(np.all(np.isfinite(ip3)))


@requires_nest
class TestAstrocyteSICResponseParity(unittest.TestCase):
    """Deterministic SIC pathway (IP3/Ca/I_SIC) matches live NEST per-sample."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_ip3_ca_isic_traces_match_nest(self):
        """IP3, Ca, and the delivered ``I_SIC`` track NEST within ``ASTRO_TOL``."""
        n_ip3, n_ca, n_isic = _nest_sic_response()
        b_ip3, b_ca, b_isic = _bp_sic_response()
        # Sanity: the SIC pathway is actually exercised (Ca above threshold -> SIC).
        self.assertGreater(float(np.max(n_isic)), 1.0)
        for nm, ref, cand in (('IP3', n_ip3, b_ip3), ('Ca', n_ca, b_ca),
                              ('I_SIC', n_isic, b_isic)):
            n = min(ref.size, cand.size)
            compare_trace(ref[:n], cand[:n], tol=ASTRO_TOL, metric=nm).assert_()


# --- Pillar 2: neuron<->astro bidirectional loop -----------------------------------

LOOP_IE = 1000.0      # pA on the driver neuron -> sustained spiking
LOOP_DIP3 = 2.0       # ΔIP3 per driver spike -> Ca crosses SIC_th
LOOP_T = 400.0


def _bp_loop():
    """brainpy: aeif driver -> astro -> sic -> post; (V_pre, IP3, Ca, I_SIC)."""
    sim = Simulator(dt=DT * u.ms)
    pre = sim.create(aeif_cond_alpha_astro, 1, params={'I_e': LOOP_IE * u.pA})
    astro = sim.create(astrocyte_lr_1994, 1, params={'delta_IP3': LOOP_DIP3})
    post = sim.create(aeif_cond_alpha_astro, 1)
    sim.connect(pre, astro, weight=1.0, delay=DT * u.ms)
    sim.connect(astro, post,
                synapse=sic_connection(weight=1.0, delay_steps=SIC_DELAY_STEPS))
    mm_pre = sim.create(multimeter, record_from=['V_m'])
    mm_a = sim.create(multimeter, record_from=['IP3', 'Ca'])
    mm_p = sim.create(multimeter, record_from=['I_SIC'])
    sim.connect(mm_pre, pre)
    sim.connect(mm_a, astro)
    sim.connect(mm_p, post)
    res = sim.simulate(LOOP_T * u.ms)
    return (_ms(res.trace(mm_pre, 'V_m')), _ms(res.trace(mm_a, 'IP3')),
            _ms(res.trace(mm_a, 'Ca')), _ms(res.trace(mm_p, 'I_SIC')))


def _nest_loop():
    """NEST: aeif driver -> astro -> sic -> post; (V_pre, IP3, Ca, I_SIC)."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT})
    pre = nest.Create('aeif_cond_alpha_astro', 1, params={'I_e': LOOP_IE})
    astro = nest.Create('astrocyte_lr_1994', params={'delta_IP3': LOOP_DIP3})
    post = nest.Create('aeif_cond_alpha_astro', 1)
    nest.Connect(pre, astro, syn_spec={'weight': 1.0, 'delay': DT})
    nest.Connect(astro, post, syn_spec={'synapse_model': 'sic_connection', 'weight': 1.0})
    mm_pre = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': DT})
    mm_a = nest.Create('multimeter', params={'record_from': ['IP3', 'Ca_astro'], 'interval': DT})
    mm_p = nest.Create('multimeter', params={'record_from': ['I_SIC'], 'interval': DT})
    nest.Connect(mm_pre, pre)
    nest.Connect(mm_a, astro)
    nest.Connect(mm_p, post)
    nest.Simulate(LOOP_T)
    return (np.asarray(mm_pre.events['V_m'], dtype=float),
            np.asarray(mm_a.events['IP3'], dtype=float),
            np.asarray(mm_a.events['Ca_astro'], dtype=float),
            np.asarray(mm_p.events['I_SIC'], dtype=float))


class TestNeuronAstroLoopLaw(unittest.TestCase):
    """Bidirectional-loop invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_driver_spikes_drive_astro_and_close_the_sic_loop(self):
        """Driver spikes raise IP3, push Ca past SIC_th, and feed I_SIC downstream.

        Both arms of the loop must be live: the delta path (spikes -> IP3 -> Ca)
        and the SIC path (Ca above threshold -> nonzero delivered current).
        """
        v_pre, ip3, ca, isic = _bp_loop()
        self.assertGreater(float(np.max(ip3)), 5.0, 'driver spikes accumulate IP3')
        self.assertGreater(float(np.max(ca)), 0.19669, 'Ca crosses the SIC threshold')
        self.assertGreater(float(np.max(isic)), 0.0, 'SIC current is delivered downstream')
        self.assertTrue(np.all(np.isfinite(v_pre)) and np.all(np.isfinite(isic)))

    def test_loop_lowers_with_stable_trace_shapes(self):
        """The whole loop runs under the Simulator's for_loop with (T/dt,) traces."""
        v_pre, ip3, ca, isic = _bp_loop()
        n = int(round(LOOP_T / DT))
        for tr in (v_pre, ip3, ca, isic):
            self.assertEqual(tr.shape, (n,))


@requires_nest
class TestNeuronAstroLoopParity(unittest.TestCase):
    """The driven neuron<->astro loop (V/IP3/Ca/I_SIC) matches live NEST."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_loop_traces_match_nest(self):
        """V_pre (CAT_A) and IP3/Ca/I_SIC (ASTRO_TOL) all track NEST."""
        n_v, n_ip3, n_ca, n_isic = _nest_loop()
        b_v, b_ip3, b_ca, b_isic = _bp_loop()
        # Sanity: the SIC arm fired (Ca crossed threshold inside the loop).
        self.assertGreater(float(np.max(n_isic)), 1.0)
        nv = min(n_v.size, b_v.size)
        compare_trace(n_v[:nv], b_v[:nv], tol=CAT_A, metric='V_pre').assert_()
        for nm, ref, cand in (('IP3', n_ip3, b_ip3), ('Ca', n_ca, b_ca),
                              ('I_SIC', n_isic, b_isic)):
            n = min(ref.size, cand.size)
            compare_trace(ref[:n], cand[:n], tol=ASTRO_TOL, metric=nm).assert_()


# --- Pillar 3: astro-network distributional ----------------------------------------

NET_N = 30
NET_IE = 700.0        # pA -> ~9 Hz base firing (sensitive f-I region)
NET_PRATE = 800.0     # Hz Poisson drive into each astrocyte
NET_W_P2A = 5.0       # poisson->astro weight (delta -> IP3)
NET_W_A2N = 10.0      # astro->post sic weight (on-condition)
NET_T = 1000.0
NET_SEEDS = (1, 2, 3)


def _bp_net_rate(seed, w_a2n):
    """brainpy seed-mean post firing rate (Hz) with the SIC arm on/off."""
    brainstate.random.seed(seed)
    sim = Simulator(dt=DT * u.ms)
    post = sim.create(aeif_cond_alpha_astro, NET_N, params={'I_e': NET_IE * u.pA})
    astro = sim.create(astrocyte_lr_1994, NET_N,
                       params={'delta_IP3': 1.0, 'Ca_initializer': 0.15})
    pg = sim.create(poisson_generator, NET_N, rate=NET_PRATE * u.Hz)
    sr = sim.create(spike_recorder)
    sim.connect(pg, astro, rule=one_to_one, weight=NET_W_P2A, delay=1.0 * u.ms)
    if w_a2n > 0:
        sim.connect(astro, post, rule=one_to_one,
                    synapse=sic_connection(weight=w_a2n, delay_steps=SIC_DELAY_STEPS))
    sim.connect(post, sr)
    res = sim.simulate(NET_T * u.ms)
    return float(res.rate(sr.segments[0].population))


def _nest_net_rate(seed, w_a2n):
    """NEST seed-mean post firing rate (Hz) with the SIC arm on/off."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT, 'rng_seed': int(seed)})
    post = nest.Create('aeif_cond_alpha_astro', NET_N, params={'I_e': NET_IE})
    astro = nest.Create('astrocyte_lr_1994', NET_N, params={'delta_IP3': 1.0, 'Ca_astro': 0.15})
    pg = nest.Create('poisson_generator', NET_N, params={'rate': NET_PRATE})
    sr = nest.Create('spike_recorder')
    nest.Connect(pg, astro, 'one_to_one', syn_spec={'weight': NET_W_P2A, 'delay': 1.0})
    if w_a2n > 0:
        nest.Connect(astro, post, 'one_to_one',
                     syn_spec={'synapse_model': 'sic_connection', 'weight': w_a2n})
    nest.Connect(post, sr)
    nest.Simulate(NET_T)
    return sr.n_events / (NET_N * NET_T / 1000.0)


class TestAstroNetworkLaw(unittest.TestCase):
    """The emergent SIC-raises-firing law needs no NEST (always runs)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_sic_arm_raises_mean_firing(self):
        """Switching the astrocyte->neuron SIC arm on raises seed-mean post firing.

        With the post neurons clamped just above rheobase, the extra SIC current
        from the Poisson-driven astrocytes lifts the population firing rate.
        """
        off = np.array([_bp_net_rate(s, 0.0) for s in NET_SEEDS])
        on = np.array([_bp_net_rate(s, NET_W_A2N) for s in NET_SEEDS])
        self.assertGreater(float(on.mean()), float(off.mean()) + 1.0,
                           'the SIC arm must measurably raise mean firing')
        self.assertTrue(np.all(off > 0.0), 'the post population fires at baseline')


@requires_nest
class TestAstroNetworkParity(unittest.TestCase):
    """Seed-mean post firing rate matches live NEST with and without the SIC arm."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_seed_mean_rate_matches_nest_both_arms(self):
        """SIC-off and SIC-on seed-mean post rates land within CAT_D (5 %) of NEST."""
        for w, tag in ((0.0, 'SIC-off'), (NET_W_A2N, 'SIC-on')):
            nest_rates = [_nest_net_rate(s, w) for s in NET_SEEDS]
            bp_rates = [_bp_net_rate(s, w) for s in NET_SEEDS]
            self.assertGreater(float(np.mean(nest_rates)), 0.0)
            compare_distributional(nest_rates, bp_rates, tol=CAT_D,
                                   metric=f'post rate ({tag})').assert_()


if __name__ == '__main__':
    unittest.main()
