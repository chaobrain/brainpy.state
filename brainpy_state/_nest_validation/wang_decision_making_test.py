# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Distributional winner-take-all parity for the Wang (2002) decision network.

Builds the *same* reduced, mean-field-preserving network (``ne=200, ni=50``;
recurrent weights scaled by ``N_full / N``) in **brainpy.state** (the ported
``examples/nest/wang_decision_making.py``) and in **live NEST** (the upstream
``iaf_bw_2001`` model wired by hand here), then compares the *decision behaviour*
over a handful of seeds.

Why distributional, not per-sample
----------------------------------
The recurrent-NMDA *coupling* matches NEST to machine precision (see
``iaf_bw_2001_recurrent_nmda_parity_test.py``). The decision *network*, however,
is a winner-take-all attractor: its positive NMDA feedback amplifies the tiny
per-neuron differences between the two integrators (brainpy's adaptive RKF45 vs
NEST's solver) and the divergent PRNG streams (JAX vs NEST Poisson). So the
*winner's absolute steady rate* legitimately differs between the simulators
(realized seed-means at this scale: brainpy A~12 Hz vs NEST A~7 Hz on +bias),
while the *decision* — which population wins, and that it wins decisively — is
robust and agrees. The assertions therefore target distributional/behavioural
invariants, never a per-sample rate match:

* **Direction.** Strong +coherence selects A and strong -coherence selects B, on
  >= 2/3 seeds on *both* simulators, with the dominant side agreeing across sims.
* **WTA structure.** The winner's late-window rate exceeds the loser's by > 2.5x
  and the loser is suppressed (< 4 Hz) on both sims.
* **Unbiased at zero coherence.** The seed-mean A-vs-B bias at zero coherence is a
  small fraction (< 1/2) of the strong-coherence winner-loser gap on both sims —
  the choice is noise-driven, not wired.

Realized numbers (ne=200, ni=50, seeds 1-3, +102.4): brainpy A wins 3/3, late
A~12 / B~1.4 Hz; NEST A wins 3/3, late A~7 / B~1.2 Hz. Mirror image at -102.4.

Skips when NEST is unavailable.
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import requires_nest
from examples.nest.wang_decision_making import (
    EPOP, IPOP, G, F, NE_FULL, NI_FULL, W_PLUS, W_MINUS, DELAY, DELAY_EXT,
    SIGNAL_START, SIGNAL_DUR, RATE_BG, _signal_rates, decision_from_rates,
    run_decision)

#: Reduced, mean-field-preserving config and the distributional sample.
NE, NI, T, DT = 200, 50, 2500.0, 0.1
SEEDS = (1, 2, 3)
COH = dict(pos=102.4, neg=-102.4, zero=0.0)
LATE = slice(int(2000 / DT), int(2500 / DT))   # post-signal attractor window


def _nest_rates(coherence, seed, ne, ni, T, dt=DT):
    """Run the reduced Wang network in live NEST; return (rate_a, rate_b) in Hz.

    Mirrors ``examples.nest.wang_decision_making.build`` connection-for-connection
    (same block-structured WTA weights, same N_full/N scaling, the same per-interval
    signal envelope via the shared ``_signal_rates``) so any decision difference is
    dynamics, not wiring.
    """
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.set(resolution=dt, rng_seed=int(seed))
    sE, sI = NE_FULL / ne, NI_FULL / ni
    nA = int(F * ne)
    E = nest.Create('iaf_bw_2001', ne, params={**EPOP, 'V_m': EPOP['E_L']})
    I = nest.Create('iaf_bw_2001', ni, params={**IPOP, 'V_m': IPOP['E_L']})
    selA, selB, NS = E[:nA], E[nA:2 * nA], E[2 * nA:]
    rt = E[0].receptor_types

    def syn(factor, g, scale, receptor):
        return {'synapse_model': 'static_synapse', 'weight': factor * g * scale,
                'delay': DELAY, 'receptor_type': rt[receptor]}

    def conn_exc(src, tgt, factor):
        nest.Connect(src, tgt, 'all_to_all', syn(factor, G['AMPA_ex'], sE, 'AMPA'))
        nest.Connect(src, tgt, 'all_to_all', syn(factor, G['NMDA_ex'], sE, 'NMDA'))

    conn_exc(E, NS, 1.0)
    conn_exc(selA, selA, W_PLUS)
    conn_exc(selB, selB, W_PLUS)
    conn_exc(selA, selB, W_MINUS)
    conn_exc(selB, selA, W_MINUS)
    conn_exc(NS, selA, W_MINUS)
    conn_exc(NS, selB, W_MINUS)
    nest.Connect(E, I, 'all_to_all', syn(1.0, G['AMPA_in'], sE, 'AMPA'))
    nest.Connect(E, I, 'all_to_all', syn(1.0, G['NMDA_in'], sE, 'NMDA'))
    nest.Connect(I, E, 'all_to_all', syn(1.0, G['GABA_ex'], sI, 'GABA'))
    nest.Connect(I, I, 'all_to_all', syn(1.0, G['GABA_in'], sI, 'GABA'))

    ext_e = {'synapse_model': 'static_synapse', 'weight': G['AMPA_ext_ex'],
             'delay': DELAY_EXT, 'receptor_type': rt['AMPA']}
    ext_i = {'synapse_model': 'static_synapse', 'weight': G['AMPA_ext_in'],
             'delay': DELAY_EXT, 'receptor_type': rt['AMPA']}
    bg = nest.Create('poisson_generator', params={'rate': RATE_BG})
    nest.Connect(bg, E, 'all_to_all', ext_e)
    nest.Connect(bg, I, 'all_to_all', ext_i)

    times, ra, rb = _signal_rates(coherence, seed)
    pa = nest.Create('inhomogeneous_poisson_generator',
                     params={'rate_times': times, 'rate_values': ra,
                             'start': SIGNAL_START - 0.1, 'stop': SIGNAL_START + SIGNAL_DUR})
    pb = nest.Create('inhomogeneous_poisson_generator',
                     params={'rate_times': times, 'rate_values': rb,
                             'start': SIGNAL_START - 0.1, 'stop': SIGNAL_START + SIGNAL_DUR})
    nest.Connect(pa, selA, 'all_to_all', ext_e)
    nest.Connect(pb, selB, 'all_to_all', ext_e)

    sa = nest.Create('spike_recorder')
    sb = nest.Create('spike_recorder')
    nest.Connect(selA, sa)
    nest.Connect(selB, sb)
    nest.Simulate(T)

    nbins = int(round(T / dt))
    edges = np.arange(nbins + 1) * dt
    w = int(round(50.0 / dt))

    def pop_rate(sr):
        t = np.asarray(sr.events['times'])
        count, _ = np.histogram(t, bins=edges)
        smooth = np.convolve(count.astype(float), np.ones(w) / w, mode='same')
        return smooth / nA / (dt / 1000.0)

    return pop_rate(sa), pop_rate(sb)


def _summary(rate_a, rate_b):
    """Decision summary from two rate traces: winner + late-window mean rates."""
    dec = decision_from_rates(rate_a, rate_b, DT)
    return dict(winner=dec['winner'],
                lateA=float(np.mean(rate_a[LATE])),
                lateB=float(np.mean(rate_b[LATE])))


@requires_nest
class TestWangDecisionParity(unittest.TestCase):
    """The ported Wang network reproduces NEST's WTA decision distributionally."""

    @classmethod
    def setUpClass(cls):
        # Run the (coherence x seed) grid once on each simulator; cache summaries.
        cls.bp, cls.ns = {}, {}
        for key, coh in COH.items():
            for sd in SEEDS:
                out = run_decision(coh, sd, ne=NE, ni=NI, T=T, dt=DT)
                cls.bp[(key, sd)] = _summary(out['rate_a'], out['rate_b'])
                ra, rb = _nest_rates(coh, sd, NE, NI, T)
                cls.ns[(key, sd)] = _summary(ra, rb)

    def _win_fraction(self, store, key, side):
        wins = [store[(key, sd)]['winner'] == side for sd in SEEDS]
        return sum(wins) / len(SEEDS)

    def _seed_mean(self, store, key, field):
        return float(np.mean([store[(key, sd)][field] for sd in SEEDS]))

    def test_strong_coherence_selects_biased_population_on_both_sims(self):
        """+coherence -> A, -coherence -> B, on >= 2/3 seeds, agreeing across sims."""
        for store, label in ((self.bp, 'brainpy'), (self.ns, 'NEST')):
            self.assertGreaterEqual(self._win_fraction(store, 'pos', 'A'), 2 / 3,
                                    f'{label}: +coherence did not select A')
            self.assertGreaterEqual(self._win_fraction(store, 'neg', 'B'), 2 / 3,
                                    f'{label}: -coherence did not select B')
            # The biased side must not lose to the other.
            self.assertLess(self._win_fraction(store, 'pos', 'B'), 2 / 3)
            self.assertLess(self._win_fraction(store, 'neg', 'A'), 2 / 3)

    def test_winner_take_all_contrast_on_both_sims(self):
        """Winner late rate >> loser (> 2.5x) and loser suppressed (< 4 Hz)."""
        for store, label in ((self.bp, 'brainpy'), (self.ns, 'NEST')):
            # +coherence: A is the winner.
            a, b = self._seed_mean(store, 'pos', 'lateA'), self._seed_mean(store, 'pos', 'lateB')
            self.assertGreater(a, 2.5 * b, f'{label}: +coh A not >> B (A={a:.2f} B={b:.2f})')
            self.assertLess(b, 4.0, f'{label}: +coh loser B not suppressed ({b:.2f} Hz)')
            # -coherence: B is the winner.
            a, b = self._seed_mean(store, 'neg', 'lateA'), self._seed_mean(store, 'neg', 'lateB')
            self.assertGreater(b, 2.5 * a, f'{label}: -coh B not >> A (A={a:.2f} B={b:.2f})')
            self.assertLess(a, 4.0, f'{label}: -coh loser A not suppressed ({a:.2f} Hz)')

    def test_zero_coherence_is_unbiased_on_both_sims(self):
        """At zero coherence the seed-mean A/B bias is small vs the biased gap."""
        for store, label in ((self.bp, 'brainpy'), (self.ns, 'NEST')):
            bias0 = abs(self._seed_mean(store, 'zero', 'lateA')
                        - self._seed_mean(store, 'zero', 'lateB'))
            gap = self._seed_mean(store, 'pos', 'lateA') - self._seed_mean(store, 'pos', 'lateB')
            self.assertLess(bias0, 0.5 * gap,
                            f'{label}: zero-coherence bias {bias0:.2f} not << gap {gap:.2f}')


if __name__ == '__main__':
    unittest.main()
