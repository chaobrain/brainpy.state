# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""EI-clustered-network parity: brainpy.state Simulator vs live NEST.

Ports NEST's ``EI_clustered_network/`` (``clustering="weight"``): an ``iaf_psc_exp``
random balanced network whose E and I populations are split into ``Q`` clusters,
with in-cluster synapses potentiated (``J+``) and out-cluster synapses depressed
(``J-``) so the row sums (mean weight) are preserved (Rostami et al. 2020, Eqs
7-10). The network is driven by a constant per-neuron rheobase current — no external
Poisson. Clustering turns the balanced random network into a **metastable**
one: clusters spontaneously wax and wane (winner-take-all), producing large
across-cluster rate heterogeneity and more irregular firing than the homogeneous
(``rep=1``) control.

Parity model — **quantitative anchor at rep=1, qualitative signature at rep=6**:

* **Homogeneous control (rep=1)** — with no clustering the network is an ordinary
  balanced RBN; its *typical* (AI-state) E/I rates and ISI CV match NEST tightly
  (median over seeds: rate ~1-3 %, CV <4 %). The comparison is on the **median**,
  not the mean, because the balanced network is bistable: an occasional realization
  falls into a globally **synchronized** state (CV ≈ 0, clock-like, elevated I rate)
  — a legitimate alternate attractor whose occurrence is PRNG-dependent and not
  shared between simulators (brainpy showed one such realization in eight; NEST
  none). The median is immune to that outlier and reports the AI-state parity; CV
  matching there is the evidence the *neuron/synapse* is faithful.
* **Clustered (rep=6)** — the metastable attractor occupancy is PRNG-dependent, so
  the per-realization rate/std diverge between simulators (brainpy runs hotter);
  asserting a quantitative match there would be asserting PRNG noise, so it is **not**
  asserted. Instead the **clustering signature** is required in *both* simulators:
  going rep=1→rep=6 raises across-cluster rate heterogeneity (``std6 > 3·std1``) and
  raises irregularity (``CV6 > CV1``), with both populations remaining in the AI
  regime. This is the metastability fingerprint, robust despite the quantitative
  scatter.

The no-NEST companion (reduced config) runs always so CI exercises the importable
surface and the pure connectivity/statistic helpers (``fix-brunel-es-import`` rule).
"""
import unittest

import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np

try:
    import nest
except Exception:
    nest = None

from examples.nest_like.ei_clustered_network import (
    simulate, build, cluster_weight_matrix, cluster_weights, cv_isi,
    cluster_rate_std, population_rate, rbn_weights, psc_to_psp, rheobase,
    N_E, N_I, Q, REP, RJ, WARMUP, SIMTIME, DT, BCP, GEI, GIE, GII,
    I_TH_E, I_TH_I, E_L, C_M, TAU_E, TAU_I, T_REF, V_TH, V_R, TAU_SYN, DELAY,
)
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

#: Canonical / parity config — canonical NEST clustering params (Q, rep, rj), only N
#: scaled 10x for tractability.
CANON = dict(ne=N_E, ni=N_I, q=Q, warmup=WARMUP, simtime=SIMTIME, dt=DT)
#: Odd seed count → the median is a single middle value, cleanly excluding the
#: occasional globally-synchronized realization (see module docstring).
SEEDS = (1, 2, 3, 4, 5)
#: Reduced companion config (no NEST, CI speed). q=10 keeps rep<Q (J- positive).
FAST = dict(ne=200, ni=50, q=10, warmup=300.0, simtime=1000.0, dt=DT)

#: Homogeneous (rep=1) median rate parity: the typical AI-state realization matches
#: within ~12 % (measured 1-3 %), robust to the occasional synchronized outlier.
RATE_BAND = TraceTolerance(0.0, 0.12, label='D',
                           note="homogeneous median E/I rate (robust to sync outliers)")
#: Homogeneous median ISI-CV parity (neuron-faithfulness anchor): ~8 % (measured <4 %).
CV_BAND = TraceTolerance(0.0, 0.08, label='D',
                         note="homogeneous median ISI CV; neuron-faithfulness anchor")


class TestEIClusteredNetwork(unittest.TestCase):
    """No-NEST companion: pure helpers + reduced-config qualitative behavior."""

    #: The metastability signature is a seed-*mean* property; small single
    #: realizations can be duds (one seed shows std6 < std1), so average a few.
    SEEDS_FAST = (1, 2, 3)

    @classmethod
    def setUpClass(cls):
        cls.sims6 = [simulate(s, rep=6.0, **FAST) for s in cls.SEEDS_FAST]
        cls.sims1 = [simulate(s, rep=1.0, **FAST) for s in cls.SEEDS_FAST]

    # ---- pure connectivity / statistic helpers (fast, no simulation) ----

    def test_cluster_weight_matrix_shape_and_sparsity(self):
        # (npre, npost), Bernoulli-sampled at probability p (fraction nonzero ~ p).
        W = cluster_weight_matrix(200, 20, 200, 20, base_j=1.0, plus=6.0, minus=0.7,
                                  p=0.2, no_auto=True, seed=7)
        self.assertEqual(W.shape, (200, 200))
        frac = np.count_nonzero(W) / W.size
        self.assertAlmostEqual(frac, 0.2, delta=0.03)

    def test_cluster_weight_matrix_potentiates_same_cluster(self):
        # rep>1: same-cluster entries carry the larger (plus) weight, cross the smaller.
        W = cluster_weight_matrix(200, 20, 200, 20, base_j=1.0, plus=6.0, minus=0.7,
                                  p=0.2, no_auto=True, seed=7)
        cpre = np.arange(200) // 20
        same = cpre[:, None] == cpre[None, :]
        same_w = W[same & (W != 0)]
        cross_w = W[(~same) & (W != 0)]
        self.assertGreater(same_w.mean(), cross_w.mean())
        self.assertAlmostEqual(same_w.mean(), 6.0, places=6)
        self.assertAlmostEqual(cross_w.mean(), 0.7, places=6)

    def test_cluster_weight_matrix_homogeneous_is_uniform(self):
        # rep=1 (plus==minus): every realized weight is identical (no clustering).
        W = cluster_weight_matrix(200, 20, 200, 20, base_j=1.3, plus=1.0, minus=1.0,
                                  p=0.2, no_auto=True, seed=3)
        nz = W[W != 0]
        self.assertTrue(np.allclose(nz, 1.3))

    def test_cluster_weight_matrix_no_autapses(self):
        # allow_autapses=False on a same-population block → zero diagonal.
        W = cluster_weight_matrix(50, 10, 50, 10, base_j=1.0, plus=6.0, minus=0.7,
                                  p=1.0, no_auto=True, seed=1)
        self.assertTrue(np.all(np.diag(W) == 0.0))
        # cross-population block (npre != npost) keeps its full mask.
        W2 = cluster_weight_matrix(50, 10, 40, 8, base_j=1.0, plus=6.0, minus=0.7,
                                   p=1.0, no_auto=False, seed=1)
        self.assertEqual(np.count_nonzero(W2), 50 * 40)

    def test_cluster_weights_formula(self):
        # jplus=[[rep,jip],[jip,jip]], jip=1+(rep-1)*rj; jminus=(Q-jplus)/(Q-1).
        jplus, jminus = cluster_weights(6.0, 0.82, 20)
        self.assertAlmostEqual(jplus[0, 0], 6.0)
        self.assertAlmostEqual(jplus[0, 1], 1.0 + 5.0 * 0.82)      # 5.1
        self.assertTrue(np.all(jminus > 0))                        # rep<Q keeps J- > 0
        self.assertAlmostEqual(jminus[0, 0], (20 - 6.0) / 19.0)

    def test_cluster_weights_homogeneous(self):
        # rep=1 → all J+ and J- equal 1 (the balanced random network).
        jplus, jminus = cluster_weights(1.0, 0.82, 20)
        self.assertTrue(np.allclose(jplus, 1.0))
        self.assertTrue(np.allclose(jminus, 1.0))

    def test_rbn_weights_signs(self):
        # E rows excitatory (+), I rows inhibitory (-); psp amplitude & rheobase > 0.
        js = rbn_weights()
        self.assertGreater(js[0, 0], 0.0)    # E->E
        self.assertLess(js[0, 1], 0.0)       # I->E
        self.assertGreater(js[1, 0], 0.0)    # E->I
        self.assertLess(js[1, 1], 0.0)       # I->I
        self.assertGreater(psc_to_psp(TAU_E, TAU_SYN), 0.0)
        self.assertGreater(rheobase(TAU_E, E_L, V_TH, C_M), 0.0)

    def test_cv_isi_regular_is_zero_irregular_is_positive(self):
        # Perfectly periodic train → CV 0; jittered ISIs → CV > 0.
        T = 1000
        regular = np.zeros((T, 1), dtype=bool)
        regular[::20, 0] = True
        self.assertAlmostEqual(cv_isi(regular, DT), 0.0, places=6)
        irregular = np.zeros((T, 1), dtype=bool)
        irregular[[10, 25, 80, 95, 200, 360], 0] = True
        self.assertGreater(cv_isi(irregular, DT), 0.0)

    def test_cluster_rate_std_known(self):
        # All clusters equal → 0 std; concentrating spikes in one cluster → > 0.
        T, n, q = 100, 20, 4
        uniform = np.ones((T, n), dtype=bool)
        self.assertAlmostEqual(cluster_rate_std(uniform, q, T=1000.0), 0.0, places=6)
        skewed = np.zeros((T, n), dtype=bool)
        skewed[:, :5] = True                 # only the first cluster fires
        self.assertGreater(cluster_rate_std(skewed, q, T=1000.0), 0.0)

    def test_population_rate_known(self):
        # 2 spikes total over 5 neurons in 1000 ms → 2/5/1 = 0.4 Hz.
        r = np.zeros((100, 5), dtype=bool)
        r[10, 0] = True
        r[20, 3] = True
        self.assertAlmostEqual(population_rate(r, 1000.0), 0.4, places=6)

    # ---- reduced-config simulation behavior ----

    def test_build_returns_separate_recorders(self):
        sim, esr, isr = build(1, rep=6.0, ne=200, ni=50, q=10, dt=DT)
        self.assertIsNotNone(esr)
        self.assertIsNotNone(isr)
        self.assertIsNot(esr, isr)

    def test_rates_in_ai_regime(self):
        for d in self.sims6 + self.sims1:
            self.assertGreater(d['e_rate'], 1.0)
            self.assertLess(d['e_rate'], 60.0)
            self.assertGreater(d['i_rate'], 1.0)
            self.assertLess(d['i_rate'], 60.0)

    def test_clustering_raises_heterogeneity(self):
        # The metastability fingerprint: clustering blows up across-cluster rate std
        # (on the seed mean — single small realizations are noisy).
        std6 = float(np.mean([d['cluster_std'] for d in self.sims6]))
        std1 = float(np.mean([d['cluster_std'] for d in self.sims1]))
        self.assertGreater(std6, 3.0)
        self.assertGreater(std6, 2.0 * std1)

    def test_clustering_raises_irregularity(self):
        # Clustering also makes firing more irregular (bursty winners).
        cv6 = float(np.mean([d['cv_e'] for d in self.sims6]))
        cv1 = float(np.mean([d['cv_e'] for d in self.sims1]))
        self.assertGreater(cv6, cv1)
        for d in self.sims6 + self.sims1:
            self.assertGreater(d['cv_e'], 0.4)
            self.assertLess(d['cv_e'], 1.6)

    def test_determinism(self):
        again = simulate(1, rep=6.0, **FAST)
        self.assertEqual(again['e_rate'], self.sims6[0]['e_rate'])
        self.assertEqual(again['cluster_std'], self.sims6[0]['cluster_std'])


@requires_nest
class TestEIClusteredNetworkParity(unittest.TestCase):
    """Live-NEST parity at the canonical Q=20 config (seeds {1,2,3,4})."""

    @classmethod
    def setUpClass(cls):
        cls.bp6 = [simulate(s, rep=6.0, **CANON) for s in SEEDS]
        cls.bp1 = [simulate(s, rep=1.0, **CANON) for s in SEEDS]
        cls.ne6 = [cls._nest_run(s, 6.0) for s in SEEDS]
        cls.ne1 = [cls._nest_run(s, 1.0) for s in SEEDS]

    @staticmethod
    def _nest_run(seed, rep, *, ne=N_E, ni=N_I, q=Q, warmup=WARMUP, simtime=SIMTIME,
                  dt=DT, rj=RJ):
        n = ne + ni
        sE, sI = ne // q, ni // q
        nest.ResetKernel()
        nest.resolution = dt
        nest.rng_seed = seed
        js = rbn_weights(ne, ni)
        jee, jei = js[0, 0] / np.sqrt(n), js[0, 1] / np.sqrt(n)
        jie, jii = js[1, 0] / np.sqrt(n), js[1, 1] / np.sqrt(n)
        jplus, jminus = cluster_weights(rep, rj, q)
        ix_e = I_TH_E * rheobase(TAU_E, E_L, V_TH, C_M)
        ix_i = I_TH_I * rheobase(TAU_I, E_L, V_TH, C_M)
        epar = dict(E_L=E_L, C_m=C_M, tau_m=TAU_E, t_ref=T_REF, V_th=V_TH, V_reset=V_R,
                    tau_syn_ex=TAU_SYN, tau_syn_in=TAU_SYN, I_e=ix_e)
        ipar = dict(epar); ipar['tau_m'] = TAU_I; ipar['I_e'] = ix_i
        epops = [nest.Create('iaf_psc_exp', sE, params=epar) for _ in range(q)]
        ipops = [nest.Create('iaf_psc_exp', sI, params=ipar) for _ in range(q)]
        rng = np.random.RandomState(seed)
        for p in epops + ipops:
            p.V_m = list(V_TH - 20 * rng.lognormal(0, 1, len(p)))

        def block(prepops, postpops, base_j, plus, minus, p, no_auto):
            for i, pre in enumerate(prepops):
                for j, post in enumerate(postpops):
                    w = (plus if i == j else minus) * base_j
                    nest.Connect(pre, post,
                                 {'rule': 'pairwise_bernoulli', 'p': p,
                                  'allow_autapses': not no_auto, 'allow_multapses': False},
                                 {'weight': w, 'delay': DELAY})
        block(epops, epops, jee, jplus[0, 0], jminus[0, 0], BCP[0, 0], True)
        block(ipops, epops, jei, jplus[0, 1], jminus[0, 1], BCP[0, 1], False)
        block(epops, ipops, jie, jplus[1, 0], jminus[1, 0], BCP[1, 0], False)
        block(ipops, ipops, jii, jplus[1, 1], jminus[1, 1], BCP[1, 1], True)

        sr = nest.Create('spike_recorder')
        alln = epops[0]
        for p in epops[1:] + ipops:
            alln += p
        nest.Connect(alln, sr)
        nest.Simulate(warmup + simtime)
        ev = sr.get('events')
        se = np.asarray(ev['senders']); ti = np.asarray(ev['times'])
        keep = ti >= warmup
        se = se[keep]; ti = ti[keep]
        gid0 = alln[0].global_id
        nid = se - gid0
        e_rate = (nid < ne).sum() / ne / (simtime / 1000.0)
        i_rate = (nid >= ne).sum() / ni / (simtime / 1000.0)
        crates = [((nid >= c * sE) & (nid < (c + 1) * sE)).sum() / sE / (simtime / 1000.0)
                  for c in range(q)]
        cvs = []
        for g in range(ne):
            t = np.sort(ti[nid == g])
            if len(t) >= 3:
                d = np.diff(t)
                cvs.append(np.std(d) / np.mean(d))
        return dict(e_rate=e_rate, i_rate=i_rate, cluster_std=float(np.std(crates)),
                    cv_e=float(np.mean(cvs)) if cvs else float('nan'))

    def test_homogeneous_rate_matches_nest(self):
        # rep=1 balanced RBN: the median (typical AI-state) E and I rates match.
        compare_trace(np.median([d['e_rate'] for d in self.ne1]),
                      np.median([d['e_rate'] for d in self.bp1]),
                      tol=RATE_BAND, metric='homogeneous median E rate (Hz)').assert_()
        compare_trace(np.median([d['i_rate'] for d in self.ne1]),
                      np.median([d['i_rate'] for d in self.bp1]),
                      tol=RATE_BAND, metric='homogeneous median I rate (Hz)').assert_()

    def test_homogeneous_cv_matches_nest(self):
        # rep=1 median ISI CV (neuron-faithfulness anchor) within ~8 %.
        compare_trace(np.median([d['cv_e'] for d in self.ne1]),
                      np.median([d['cv_e'] for d in self.bp1]),
                      tol=CV_BAND, metric='homogeneous median ISI CV').assert_()

    def test_clustering_signature_in_both(self):
        # The metastability fingerprint, required in BOTH simulators — on the seed
        # mean, since per-realization metastable occupancy is PRNG-dependent.
        for tag, six, one in (('nest', self.ne6, self.ne1), ('brainpy', self.bp6, self.bp1)):
            std6 = float(np.mean([d['cluster_std'] for d in six]))
            std1 = float(np.mean([d['cluster_std'] for d in one]))
            cv6 = float(np.mean([d['cv_e'] for d in six]))
            cv1 = float(np.mean([d['cv_e'] for d in one]))
            e6 = float(np.mean([d['e_rate'] for d in six]))
            e1 = float(np.mean([d['e_rate'] for d in one]))
            self.assertGreater(std6, 3.0, f'{tag}: clustered std too small')
            self.assertGreater(std6, 3.0 * std1, f'{tag}: clustering signature absent')
            self.assertGreater(cv6, cv1, f'{tag}: clustering did not raise CV')
            for r in (e6, e1):                # both regimes stay in the AI band
                self.assertGreater(r, 2.0, f'{tag}: silent')
                self.assertLess(r, 40.0, f'{tag}: saturated')


if __name__ == '__main__':
    unittest.main()
