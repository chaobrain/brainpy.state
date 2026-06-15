# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Goal 15c — Siegert mean-field node + dual-channel ``diffusion_connection``.

Validates the two 15c designs against their arbiters:

* **B (JAX quadrature).** The jnp Siegert transfer ``_siegert_phi_jax`` (leggauss-64 +
  erfcx/Dawson + asymptotic expansions) matches the SciPy quadrature oracle
  (``siegert_rate``) across a (μ, σ²) grid to a documented tolerance, and the model's
  ``update`` now lowers under ``brainstate.transform.for_loop`` (the 15a eager
  exception is retired).
* **A (dual-channel deposit).** One ``siegert_neuron`` driven by one
  ``diffusion_connection`` through the Simulator (drift→μ ``'diffusion_mu'`` channel,
  diffusion→σ² ``'diffusion_sigma2'`` channel) reproduces a live-NEST two-Siegert
  trace; a population relaxes to the self-consistent mean-field fixed point.

NEST-gated groups use ``@requires_nest`` with a NEST-free companion. SciPy is the
quadrature oracle (never NEST for the transfer grid).
"""
import unittest

import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u
import braintools

from brainpy_state import (siegert_neuron, diffusion_connection, Simulator, multimeter,
                           poisson_generator, iaf_psc_alpha)

from scipy import special as _sp_special  # the oracle for the special-function ports
from brainpy_state._nest._validation.nest_compare import requires_nest


def _run_nest_two_siegert_trace(dt_ms, simtime_ms, src_params, tgt_params,
                                drift_factor, diffusion_factor):
    """Live-NEST src -> tgt diffusion_connection trace (the design-A oracle)."""
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.resolution = dt_ms
    nest.use_wfr = False
    nest.local_num_threads = 1

    src = nest.Create('siegert_neuron', params=src_params)
    tgt = nest.Create('siegert_neuron', params=tgt_params)
    nest.Connect(src, tgt, syn_spec={
        'synapse_model': 'diffusion_connection',
        'drift_factor': drift_factor,
        'diffusion_factor': diffusion_factor,
    })
    mm_src = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt_ms})
    mm_tgt = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt_ms})
    nest.Connect(mm_src, src, syn_spec={'delay': dt_ms})
    nest.Connect(mm_tgt, tgt, syn_spec={'delay': dt_ms})

    nest.Simulate(simtime_ms)
    dftype = brainstate.environ.dftype()
    return (np.asarray(mm_src.events['rate'], dtype=dftype),
            np.asarray(mm_tgt.events['rate'], dtype=dftype))


def _best_aligned_max_abs(ref, cand, max_shift=2):
    """Min over integer shifts [0, max_shift] of the max-abs residual (cand >= ref latency).

    Absorbs the Simulator's fixed one-step holder lag relative to NEST's recording
    stamp (cluster-11 align_steps convention). Returns (best_shift, max_abs, rel).
    """
    best = (0, np.inf, np.inf)
    for s in range(max_shift + 1):
        a = ref[:len(ref) - s] if s else ref
        b = cand[s:] if s else cand
        n = min(len(a), len(b))
        diff = np.abs(a[:n] - b[:n])
        rel = float(np.max(diff / np.maximum(np.abs(a[:n]), 1e-12)))
        m = float(np.max(diff))
        if m < best[1]:
            best = (s, m, rel)
    return best


def _nrn(tau_syn_ms=0.0):
    return siegert_neuron(1, tau=1.0 * u.ms, tau_m=10.0 * u.ms,
                          tau_syn=tau_syn_ms * u.ms, t_ref=2.0 * u.ms,
                          theta=15.0, V_reset=0.0)


class TestSpecialFunctionsJax(unittest.TestCase):
    """The jnp ``erfcx`` / Dawson ports match SciPy across their domains."""

    def test_erfcx_jax_matches_scipy(self):
        xs = np.concatenate([np.linspace(-3.0, 25.0, 200), np.array([30.0, 60.0, 120.0])])
        got = np.asarray(siegert_neuron._erfcx_jax(jax.numpy.asarray(xs)))
        ref = _sp_special.erfcx(xs)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)

    def test_dawsn_jax_matches_scipy(self):
        # Siegert only evaluates Dawson at non-negative arguments.
        xs = np.concatenate([np.linspace(0.0, 8.0, 400), np.array([8.5, 12.0, 30.0])])
        got = np.asarray(siegert_neuron._dawsn_jax(jax.numpy.asarray(xs)))
        ref = _sp_special.dawsn(xs)
        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)


class TestQuadratureOracle(unittest.TestCase):
    """``_siegert_phi_jax`` (JAX leggauss-64) matches the SciPy quadrature oracle."""

    def _grid(self):
        mu = np.linspace(-5.0, 30.0, 40)
        sig2 = np.array([0.1, 0.5, 1.5, 4.0, 9.0, 25.0])
        MU, SIG2 = np.meshgrid(mu, sig2, indexing='ij')
        return MU.reshape(-1), SIG2.reshape(-1)

    def test_phi_jax_matches_scipy_grid(self):
        nrn = _nrn()
        mu, sig2 = self._grid()
        ref = np.asarray(nrn.siegert_rate(mu, sig2)).reshape(-1)            # SciPy oracle
        got = np.asarray(nrn._siegert_phi_jax(jax.numpy.asarray(mu),
                                              jax.numpy.asarray(sig2))).reshape(-1)
        self.assertTrue(np.all(np.isfinite(got)))
        # Documented tolerance: leggauss-64 + erfcx/Dawson vs SciPy quad/special.
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)

    def test_phi_jax_matches_scipy_colored_noise(self):
        nrn = _nrn(tau_syn_ms=0.5)  # finite tau_syn -> colored-noise threshold shift
        mu, sig2 = self._grid()
        ref = np.asarray(nrn.siegert_rate(mu, sig2)).reshape(-1)
        got = np.asarray(nrn._siegert_phi_jax(jax.numpy.asarray(mu),
                                              jax.numpy.asarray(sig2))).reshape(-1)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)

    def test_phi_jax_deterministic_and_zero_fastpaths(self):
        nrn = _nrn()
        # sigma^2 = 0 -> deterministic LIF branch; deep subthreshold -> 0.
        mu = np.array([-2.0, 5.0, 14.999, 16.0, 25.0])
        sig2 = np.zeros_like(mu)
        ref = np.asarray(nrn.siegert_rate(mu, sig2)).reshape(-1)
        got = np.asarray(nrn._siegert_phi_jax(jax.numpy.asarray(mu),
                                              jax.numpy.asarray(sig2))).reshape(-1)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)

    def test_phi_jax_matches_nest_reference_point(self):
        # The canonical NEST operating point (mu at threshold).
        nrn = _nrn()
        got = float(np.asarray(nrn._siegert_phi_jax(
            jax.numpy.asarray([15.0]), jax.numpy.asarray([1.5]))).reshape(-1)[0])
        self.assertAlmostEqual(got, 27.1095934379, delta=1e-4)


class TestForLoopLowering(unittest.TestCase):
    """The Siegert ``update`` lowers under ``for_loop`` (15a eager exception retired)."""

    def test_update_lowers_under_for_loop(self):
        nrn = _nrn()  # tau = 1 ms
        brainstate.nn.init_all_states(nrn)
        n_steps = 300  # dt=0.1 ms, tau=1 ms -> 30 tau, fully relaxed
        with brainstate.environ.context(dt=0.1 * u.ms):
            def step(i):
                return nrn.update(drift_input=12.0, diffusion_input=4.0)

            out = brainstate.transform.for_loop(step, jax.numpy.arange(n_steps))
        out = np.asarray(out)
        self.assertEqual(out.shape, (n_steps, 1))
        self.assertTrue(np.all(np.isfinite(out)))
        # The exact-exp relaxation converges to the Siegert fixed point r* = mean + Phi.
        target = float(np.asarray(nrn.siegert_rate(np.array([12.0]), np.array([4.0]))).reshape(-1)[0])
        self.assertAlmostEqual(float(out[-1, 0]), target, places=5)
        # Monotone approach from rest (no overshoot/oscillation).
        self.assertTrue(np.all(np.diff(out[:, 0]) >= -1e-9))


class TestDualChannelMicroParity(unittest.TestCase):
    """Design A arbiter: one siegert -> one siegert via ``diffusion_connection``
    through the Simulator reproduces a live-NEST two-Siegert trace."""

    DT_MS = 0.1
    SIMTIME_MS = 50.0
    DRIFT_FACTOR = 5.2
    DIFFUSION_FACTOR = 0.9
    SRC = dict(tau=2.5, tau_m=10.0, tau_syn=0.2, t_ref=2.0, theta=15.0,
               V_reset=0.0, mean=0.8, rate0=0.45)
    TGT = dict(tau=1.4, tau_m=10.0, tau_syn=0.5, t_ref=2.0, theta=15.0,
               V_reset=0.0, mean=-0.1, rate0=0.2)

    def _build_sim(self):
        sim = Simulator(dt=self.DT_MS * u.ms)

        def _p(d):
            return dict(tau=d['tau'] * u.ms, tau_m=d['tau_m'] * u.ms,
                        tau_syn=d['tau_syn'] * u.ms, t_ref=d['t_ref'] * u.ms,
                        theta=d['theta'], V_reset=d['V_reset'], mean=d['mean'],
                        rate_initializer=braintools.init.Constant(d['rate0']))

        src = sim.create(siegert_neuron, 1, params=_p(self.SRC))
        tgt = sim.create(siegert_neuron, 1, params=_p(self.TGT))
        mm_src = sim.create(multimeter, record_from=['rate'])
        mm_tgt = sim.create(multimeter, record_from=['rate'])
        sim.connect(src, tgt, synapse=diffusion_connection(
            drift_factor=self.DRIFT_FACTOR, diffusion_factor=self.DIFFUSION_FACTOR))
        sim.connect(mm_src, src)
        sim.connect(mm_tgt, tgt)
        return sim, mm_src, mm_tgt

    @requires_nest
    def test_two_siegert_vs_nest(self):
        def _nest_params(d):
            return {'tau': d['tau'], 'tau_m': d['tau_m'], 'tau_syn': d['tau_syn'],
                    't_ref': d['t_ref'], 'theta': d['theta'], 'V_reset': d['V_reset'],
                    'mean': d['mean'], 'rate': d['rate0']}

        nest_src, nest_tgt = _run_nest_two_siegert_trace(
            self.DT_MS, self.SIMTIME_MS, _nest_params(self.SRC), _nest_params(self.TGT),
            self.DRIFT_FACTOR, self.DIFFUSION_FACTOR)

        sim, mm_src, mm_tgt = self._build_sim()
        res = sim.simulate(self.SIMTIME_MS * u.ms)
        bp_src = np.asarray(res.trace(mm_src, 'rate')).reshape(-1)
        bp_tgt = np.asarray(res.trace(mm_tgt, 'rate')).reshape(-1)

        s_src, m_src, r_src = _best_aligned_max_abs(nest_src, bp_src)
        s_tgt, m_tgt, r_tgt = _best_aligned_max_abs(nest_tgt, bp_tgt)
        self.assertLessEqual(s_src, 1, f'src latency offset {s_src} steps')
        self.assertLessEqual(s_tgt, 1, f'tgt latency offset {s_tgt} steps')
        # Tight micro-parity: the substrate reproduces NEST's exact arithmetic.
        self.assertLess(m_src, 2e-6, f'src trace diverged (max_abs={m_src}, rel={r_src})')
        self.assertLess(m_tgt, 2e-6, f'tgt trace diverged (max_abs={m_tgt}, rel={r_tgt})')


class TestDiffusionNoNest(unittest.TestCase):
    """NEST-free companions: headless relaxation, channel isolation, comm guard."""

    DT_MS = 0.1
    SIMTIME_MS = 200.0  # >> tau so the feed-forward fixed point is reached
    SRC = dict(tau=2.5 * u.ms, tau_m=10.0 * u.ms, t_ref=2.0 * u.ms, theta=15.0,
               V_reset=0.0, mean=0.8)
    TGT = dict(tau=1.4 * u.ms, tau_m=10.0 * u.ms, t_ref=2.0 * u.ms, theta=15.0,
               V_reset=0.0, mean=-0.1)

    def _run(self, drift_factor, diffusion_factor, comm='dense'):
        sim = Simulator(dt=self.DT_MS * u.ms)
        src = sim.create(siegert_neuron, 1, params=dict(self.SRC))
        tgt = sim.create(siegert_neuron, 1, params=dict(self.TGT))
        mm_src = sim.create(multimeter, record_from=['rate'])
        mm_tgt = sim.create(multimeter, record_from=['rate'])
        sim.connect(src, tgt, synapse=diffusion_connection(
            drift_factor=drift_factor, diffusion_factor=diffusion_factor), comm=comm)
        sim.connect(mm_src, src)
        sim.connect(mm_tgt, tgt)
        res = sim.simulate(self.SIMTIME_MS * u.ms)
        return (np.asarray(res.trace(mm_src, 'rate')).reshape(-1),
                np.asarray(res.trace(mm_tgt, 'rate')).reshape(-1))

    def _oracle_tgt_fp(self, mu, sigma2):
        oracle = siegert_neuron(1, **self.TGT)
        phi = float(np.asarray(oracle.siegert_rate(np.array([mu]), np.array([sigma2]))).reshape(-1)[0])
        return self.TGT['mean'] + phi  # r* = mean + Phi(mu, sigma^2)

    def test_headless_relaxation(self):
        # mu=25*0.8=20 (suprathreshold), sigma^2=10*0.8=8 -> a nonzero noisy rate.
        drift, diffusion = 25.0, 10.0
        r_src, r_tgt = self._run(drift, diffusion)
        self.assertTrue(np.all(np.isfinite(r_src)) and np.all(np.isfinite(r_tgt)))
        r_src_fp = self.SRC['mean']  # Phi(0,0)=0 -> src relaxes to its mean
        self.assertAlmostEqual(float(r_src[-1]), r_src_fp, places=4)
        expected = self._oracle_tgt_fp(drift * r_src_fp, diffusion * r_src_fp)
        self.assertGreater(expected, 0.5)  # a meaningful (firing) fixed point
        self.assertAlmostEqual(float(r_tgt[-1]), expected, places=4)

    def test_diffusion_factor_zero_is_drift_only(self):
        # diffusion_factor=0 -> the sigma^2 channel is silent; tgt sees mu only
        # (mu=25*0.8=20 suprathreshold -> deterministic LIF rate).
        drift = 25.0
        _, r_tgt = self._run(drift_factor=drift, diffusion_factor=0.0)
        expected = self._oracle_tgt_fp(drift * self.SRC['mean'], 0.0)
        self.assertGreater(expected, 0.5)
        self.assertAlmostEqual(float(r_tgt[-1]), expected, places=4)

    def test_drift_factor_zero_is_variance_only(self):
        # drift_factor=0 -> the mu channel is silent; tgt sees sigma^2 only
        # (sigma^2=100*0.8=80 drives noise-induced firing despite mu=0).
        diffusion = 100.0
        _, r_tgt = self._run(drift_factor=0.0, diffusion_factor=diffusion)
        expected = self._oracle_tgt_fp(0.0, diffusion * self.SRC['mean'])
        self.assertGreater(expected, 0.5)
        self.assertAlmostEqual(float(r_tgt[-1]), expected, places=4)

    def test_channels_do_not_cross_contaminate(self):
        # Adding the variance channel on top of a fixed drift channel changes the
        # fixed point -> sigma^2 is routed independently of mu (no cross-talk).
        _, r_drift_only = self._run(drift_factor=25.0, diffusion_factor=0.0)
        _, r_both = self._run(drift_factor=25.0, diffusion_factor=100.0)
        self.assertGreater(abs(float(r_drift_only[-1]) - float(r_both[-1])), 0.5)

    def test_sparse_rejected(self):
        with self.assertRaisesRegex(ValueError, 'comm="sparse"|sparse'):
            self._run(drift_factor=1.0, diffusion_factor=1.0, comm='sparse')

    def _connect_only(self, **connect_kw):
        # Build a src -> tgt diffusion_connection without simulating; the parameter
        # guards in _build_siegert_diffusion fire at connect time.
        sim = Simulator(dt=self.DT_MS * u.ms)
        src = sim.create(siegert_neuron, 1, params=dict(self.SRC))
        tgt = sim.create(siegert_neuron, 1, params=dict(self.TGT))
        sim.connect(src, tgt, synapse=diffusion_connection(
            drift_factor=1.0, diffusion_factor=1.0), **connect_kw)

    def test_delay_rejected(self):
        # NEST parity: diffusion_connection carries no delay (the one-step seam
        # lag is implicit, matching min_delay=1).
        with self.assertRaisesRegex(ValueError, 'no delay'):
            self._connect_only(delay=1.0 * u.ms)

    def test_weight_rejected(self):
        # NEST parity: no weight; the coupling lives in drift / diffusion factors.
        with self.assertRaisesRegex(ValueError, 'no weight'):
            self._connect_only(weight=1.0)

    def test_generator_source_rejected(self):
        # A generator emits spikes/current, not a continuous rate to deposit.
        sim = Simulator(dt=self.DT_MS * u.ms)
        gen = sim.create(poisson_generator, 1, params=dict(rate=10.0))
        tgt = sim.create(siegert_neuron, 1, params=dict(self.TGT))
        with self.assertRaisesRegex(ValueError, 'not a generator'):
            sim.connect(gen, tgt, synapse=diffusion_connection(
                drift_factor=1.0, diffusion_factor=1.0))

    def test_spiking_source_rejected(self):
        # A spiking neuron has no continuous-rate emission to drive mu / sigma^2.
        sim = Simulator(dt=self.DT_MS * u.ms)
        spk = sim.create(iaf_psc_alpha, 1)
        tgt = sim.create(siegert_neuron, 1, params=dict(self.TGT))
        with self.assertRaisesRegex(ValueError, 'continuous-rate source'):
            sim.connect(spk, tgt, synapse=diffusion_connection(
                drift_factor=1.0, diffusion_factor=1.0))


if __name__ == '__main__':
    unittest.main()
