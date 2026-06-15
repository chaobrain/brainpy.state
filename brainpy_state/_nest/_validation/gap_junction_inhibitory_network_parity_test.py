# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST **inhibitory gap network** distributional parity (goal 15b, design B/C scale).

Ports NEST's ``gap_junctions_inhibitory_network.py``: a recurrent inhibitory network of
``hh_psc_alpha_gap`` cells (random ``fixed_indegree`` static inhibition ``-50 pA``, an
all-to-all excitatory Poisson drive ``300 pA @ 500 Hz``, random initial ``V_m ~
U[-80,-40] mV``) with a symmetric random gap-junction graph (``~24`` gap edges / neuron).
Hahne et al. (2015, Fig. 9/10) and the NEST example establish the **synchronization law**:
without gap junctions the balanced network is asynchronous-irregular; raising the gap
weight drives it toward a synchronous state.

This test reproduces that law *and* matches NEST quantitatively, **distributionally**
(cluster-14 style: per-seed ensembles, seed-mean parity -- the realized random graphs
differ between simulators, only their statistics are compared). The synchrony metric is
the Golomb-Rinzel population coherence

    ``chi = sqrt( Var_t<V>_pop / mean_i Var_t V_i )``   (chi -> 0 async, chi -> 1 sync),

measured over the steady state (the first 150 ms transient is dropped). N is reduced to
``200`` (the demo's ``500`` is the ``main()`` scale) so the parity runs in CI; the law
and the bands are unchanged. The NEST side runs ``use_wfr=False`` -- the apples-to-apples
reference for the substrate's one-step pipeline lag (cluster 15a; there is no waveform
relaxation in the port). The port's gating is overridden to the resting equilibrium so
the random ``V_m`` perturbations reproduce NEST's frozen-gating ICs (cf. the 2-neuron
parity).

**Result.** ``gap_weight 0.0 -> ~0.14`` (async) and ``0.7 -> ~0.36`` (synchronized) on
*both* simulators (seed-mean over 4 seeds), a ``~2.6x`` synchrony increase; the brainpy
seed-mean tracks NEST to within a few percent at each weight. The gap junction's
synchronizing effect is therefore reproduced both qualitatively (the law) and
quantitatively (the coherence magnitude).

With NEST present the comparison runs and PASSES; without NEST it SKIPs.
"""
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import saiunit as u

try:
    import nest
except Exception:
    nest = None

from brainpy_state import (Simulator, hh_psc_alpha_gap, voltmeter, poisson_generator,
                           all_to_all, fixed_indegree)
from brainpy_state._nest.gap_junction import gap_junction
from brainpy_state._nest.hh_psc_alpha_gap import _hh_psc_alpha_gap_equilibrium
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest._validation.tolerance_conventions import DistributionalTolerance

N = 200                 # neurons (demo's main() runs 500; reduced for CI, law unchanged)
INH = 20                # inhibitory in-degree per neuron (static_synapse)
GAPK = 12               # gap in-degree pre-symmetrization (~24 gap edges/neuron)
DT = 0.05               # ms
T = 501.0               # ms (matches the NEST example)
J_EXC = 300.0           # pA (Poisson drive weight)
J_INH = -50.0           # pA (recurrent inhibition weight)
RATE = 500.0            # Hz (Poisson rate)
DELAY = 1.0             # ms (chemical-synapse delay)
SKIP_MS = 150.0         # ms transient dropped before measuring chi
N_SEEDS = 4
GAP_ASYNC, GAP_SYNC = 0.0, 0.7          # nS (the demo's async vs synchronous regimes)
VR = hh_psc_alpha_gap._NEST_V_INIT

#: Distributional band on the seed-mean coherence (measured per-weight diff ~1-3 %; the
#: 0.25 relative band absorbs cross-build network-realization scatter while still pinning
#: the synchrony magnitude). chi is the cluster-14 "rate-like" scalar -> ``rate_rtol``.
CHI_TOL = DistributionalTolerance(
    rate_rtol=0.25, mean_diff_pct=0.25, autocorr_max_diff=0.0, n_seeds=N_SEEDS,
    label='D', note='gap-network Golomb coherence vs live NEST (use_wfr=False)')


def _chi(V, skip):
    r"""Golomb-Rinzel population coherence of ``V`` (``(samples, N)``), transient dropped."""
    V = V[skip:]
    den = float(V.var(axis=0).mean())
    return float(np.sqrt(V.mean(axis=1).var() / den)) if den > 0 else 0.0


# --- NEST side --------------------------------------------------------------------

def _nest_net(gap_w, seed):
    """NEST inhibitory gap network (``use_wfr=False``); ``(samples, N)`` V_m at 0.5 ms."""
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = DT
    nest.rng_seed = seed + 1
    nest.use_wfr = False
    np.random.seed(seed)
    nrn = nest.Create('hh_psc_alpha_gap', N)
    nest.SetStatus(nrn, {'I_e': 0.0})
    pg = nest.Create('poisson_generator', params={'rate': RATE})
    nest.Connect(nrn, nrn,
                 {'rule': 'fixed_indegree', 'indegree': INH,
                  'allow_autapses': False, 'allow_multapses': True},
                 {'synapse_model': 'static_synapse', 'weight': J_INH, 'delay': DELAY})
    nest.Connect(pg, nrn, 'all_to_all',
                 syn_spec={'synapse_model': 'static_synapse', 'weight': J_EXC, 'delay': DELAY})
    nrn.V_m = nest.random.uniform(min=-80.0, max=-40.0)
    if gap_w > 0:
        n_conn = int(N * GAPK)                       # ~GAPK symmetric edges per neuron
        conns = np.random.choice(nrn.tolist(), [n_conn, 2])
        for s_, t_ in conns:
            nest.Connect(nest.NodeCollection([int(s_)]), nest.NodeCollection([int(t_)]),
                         {'rule': 'one_to_one', 'make_symmetric': True},
                         {'synapse_model': 'gap_junction', 'weight': gap_w})
    mm = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': 0.5})
    nest.Connect(mm, nrn)
    nest.Simulate(T)
    ev = mm.events
    sid = np.asarray(ev['senders'])
    tt = np.asarray(ev['times'])
    cols = []
    for nid in nrn.tolist():
        m = sid == nid
        order = np.argsort(tt[m], kind='stable')
        cols.append(np.asarray(ev['V_m'])[m][order])
    return np.stack(cols, axis=1)


# --- brainpy side (through the Simulator API -- the path under test) ---------------

def _bp_net(gap_w, seed):
    """The same inhibitory gap network via the Simulator; ``(samples, N)`` V_m (mV)."""
    m_eq, h_eq, n_eq, p_eq = _hh_psc_alpha_gap_equilibrium(VR)   # NEST's frozen resting gating
    sim = Simulator(dt=DT * u.ms)
    v_init = jax.random.uniform(jax.random.PRNGKey(seed), (N,), minval=-80.0, maxval=-40.0)
    nrn = sim.create(hh_psc_alpha_gap, N, params={
        'V_m_init': v_init * u.mV, 'I_e': 0.0 * u.pA,
        'Act_m_init': m_eq, 'Inact_h_init': h_eq, 'Act_n_init': n_eq, 'Inact_p_init': p_eq})
    pg = sim.create(poisson_generator, rate=RATE * u.Hz)
    sim.connect(nrn, nrn, weight=J_INH * u.pA, delay=DELAY * u.ms,
                rule=fixed_indegree(INH), allow_multapses=True, seed=seed + 7)
    sim.connect(pg, nrn, weight=J_EXC * u.pA, delay=DELAY * u.ms, rule=all_to_all)
    if gap_w > 0:
        sim.connect(nrn, nrn, weight=gap_w * u.nS, synapse=gap_junction, comm='dense',
                    rule=fixed_indegree(GAPK), allow_autapses=False, seed=seed + 99)
    vm = sim.create(voltmeter)
    sim.connect(vm, nrn)
    res = sim.simulate(T * u.ms)
    return np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV))


@requires_nest
class TestGapInhibitoryNetworkParity(unittest.TestCase):
    """Inhibitory gap network synchronization via the Simulator matches live NEST."""

    @classmethod
    def setUpClass(cls):
        jax.clear_caches()                                   # stiff-HH x64 hygiene (21)
        brainstate.environ.set(precision=64, platform='cpu')
        skip_n = int(SKIP_MS / 0.5)                          # NEST multimeter at 0.5 ms
        skip_b = int(SKIP_MS / DT)                           # brainpy voltmeter at dt
        cls.chi_nest = {GAP_ASYNC: [], GAP_SYNC: []}
        cls.chi_bp = {GAP_ASYNC: [], GAP_SYNC: []}
        for gap_w in (GAP_ASYNC, GAP_SYNC):
            for seed in range(N_SEEDS):
                cls.chi_nest[gap_w].append(_chi(_nest_net(gap_w, seed), skip_n))
                cls.chi_bp[gap_w].append(_chi(_bp_net(gap_w, seed), skip_b))

    def test_gap_junctions_increase_synchrony_on_both(self):
        """The synchronization law holds on BOTH sims: gap coupling raises coherence.

        Without gaps the balanced network is asynchronous (low chi); the symmetric gap
        graph drives it toward synchrony (clearly higher chi). This is the qualitative
        gate -- the gap junction's defining network effect, reproduced by the port.
        """
        for label, chi in (('NEST', self.chi_nest), ('brainpy', self.chi_bp)):
            asyn = float(np.mean(chi[GAP_ASYNC]))
            sync = float(np.mean(chi[GAP_SYNC]))
            self.assertLess(asyn, 0.22, f'{label}: async chi {asyn:.3f} not low')
            self.assertGreater(sync, 0.28, f'{label}: sync chi {sync:.3f} not high')
            self.assertGreater(sync, 1.6 * asyn,
                               f'{label}: gap junctions did not raise synchrony '
                               f'({asyn:.3f} -> {sync:.3f})')

    def test_async_coherence_distributional_parity(self):
        """Without gaps, the brainpy seed-mean coherence matches NEST (balanced AI state)."""
        compare_distributional(self.chi_nest[GAP_ASYNC], self.chi_bp[GAP_ASYNC],
                               tol=CHI_TOL, metric='chi_async', statistic='mean').assert_()

    def test_sync_coherence_distributional_parity(self):
        """With gaps, the brainpy seed-mean coherence matches NEST (synchronous state)."""
        compare_distributional(self.chi_nest[GAP_SYNC], self.chi_bp[GAP_SYNC],
                               tol=CHI_TOL, metric='chi_sync', statistic='mean').assert_()


if __name__ == '__main__':
    unittest.main()
