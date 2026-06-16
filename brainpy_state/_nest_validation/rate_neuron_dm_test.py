# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Winner-take-all decision parity for the ``rate_neuron_dm`` demo (goal 17).

The NEST demo wires two ``lin_rate_ipn`` decision units (``lambda=0.1``, ``tau=1 ms``,
``rectify_output=True``) in **mutual instantaneous inhibition** (``weight=-0.2``). The
coupling matrix ``lambda*I - W`` is indefinite, so the rectified dynamics form a
winner-take-all bistable attractor: with evidence ``mu_win`` the winner relaxes to
``mu_win / lambda = 10*mu_win`` and the loser is rectified to ``0``. A positive evidence
bias ``dE`` selects the higher-``mu`` unit; at ``dE=0`` the input noise breaks the tie.

Two arbiters, matching the cluster-16 house style:

* ``TestRateNeuronDmStructure`` -- NEST-free, always runs (the no-NEST companion). The
  deterministic (``sigma=0``) strong-bias run is the closed-form anchor **and** the goal-17
  R1 arbiter: this is the first *recurrent rectified* rate net relaxed through the
  :class:`~brainpy.state.Simulator`, so a misbehaving ``rectify_output`` in the recurrent
  path would surface here. The symmetric run hits the interior fixed point
  ``1 / (lambda - w_inh) = 3.33``; the noisy zero-bias run is unbiased (noise breaks the
  tie in *both* directions).
* ``TestRateNeuronDmNestParity`` (``@requires_nest``) -- the same net vs live NEST
  (``lin_rate_ipn`` + ``rate_connection_instantaneous``, ``use_wfr=False``). The WTA
  amplifies any PRNG divergence, so the noisy runs are compared **distributionally**
  (decision direction, winner/loser contrast, zero-bias balance) over seeds, never
  per-sample; the deterministic strong-bias run is matched tightly.
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainunit as u  # noqa: F401  (kept for parity with sibling test modules / x64 env)

try:
    import nest
except Exception:
    nest = None

from brainpy_state._nest_validation.nest_compare import requires_nest
from examples.nest.rate_neuron_dm import run_scenario, W_INH

LAM = 0.1                              # decision-unit passive decay lambda
N_SEEDS = 5                            # distributional sample (bounds live-NEST wall-clock)
N_UNBIAS = 9                           # larger sample for the NEST-free unbias check (fast)
SYMMETRIC_FP = 1.0 / (LAM - W_INH)     # interior fixed point 1 / 0.3 = 3.333...
STRONG_GAP = 10.0                      # winner-loser scale at unit evidence (mu/lambda)


def _winner_loser(r1, r2, tail=0.2):
    """Mean winner/loser rate over the last ``tail`` fraction of the two traces."""
    k = int(len(r1) * (1.0 - tail))
    a, b = float(np.mean(r1[k:])), float(np.mean(r2[k:]))
    return (a, b) if a >= b else (b, a)


def _d1_wins(r1, r2, tail=0.2):
    """``True`` when D1's tail-mean rate exceeds D2's (D1 is the decision winner)."""
    k = int(len(r1) * (1.0 - tail))
    return float(np.mean(r1[k:])) > float(np.mean(r2[k:]))


class TestRateNeuronDmStructure(unittest.TestCase):
    """Rectified WTA decision dynamics relaxed through the Simulator (NEST-free)."""

    def test_deterministic_wta_fixed_point(self):
        """sigma=0, strong bias: winner -> 10*(1+|dE|), loser -> 0, bias picks the winner.

        Doubles as the goal-17 **R1 arbiter** for ``rectify_output`` in the recurrent path.
        """
        for dE, d1_should_win in ((0.1, True), (-0.1, False)):
            with self.subTest(dE=dE):
                r1, r2, _ = run_scenario(sigma=0.0, dE=dE, T_each=200.0)
                win, lose = _winner_loser(r1, r2)
                self.assertAlmostEqual(win, STRONG_GAP * (1.0 + abs(dE)), delta=1e-2,
                                       msg=f'winner {win} != 10*(1+|dE|)')
                self.assertLess(lose, 1e-6, msg=f'loser {lose} not rectified to 0')
                self.assertEqual(_d1_wins(r1, r2), d1_should_win)

    def test_no_decision_when_symmetric(self):
        """sigma=0, dE=0: the symmetric interior fixed point 1/(lambda - w_inh) = 3.33."""
        r1, r2, _ = run_scenario(sigma=0.0, dE=0.0, T_each=200.0)
        self.assertAlmostEqual(r1[-1], r2[-1], delta=1e-3)
        self.assertAlmostEqual(r1[-1], SYMMETRIC_FP, delta=1e-2)

    def test_zero_bias_noise_is_unbiased(self):
        """sigma>0, dE=0: noise breaks the tie in BOTH directions; seeds stay balanced."""
        outcomes = [run_scenario(sigma=0.1, dE=0.0, T_each=150.0, seed=s)
                    for s in range(N_UNBIAS)]
        d1_won = [_d1_wins(r1, r2) for r1, r2, _ in outcomes]
        self.assertTrue(any(d1_won), 'D1 never won across seeds (biased)')
        self.assertTrue(not all(d1_won), 'D2 never won across seeds (biased)')
        # seed-averaged signed rates stay far below the strong-bias winner-loser gap.
        m1 = float(np.mean([np.mean(r1[-200:]) for r1, _, _ in outcomes]))
        m2 = float(np.mean([np.mean(r2[-200:]) for _, r2, _ in outcomes]))
        self.assertLess(abs(m1 - m2), 0.6 * STRONG_GAP)

    def test_dm_lowers_under_for_loop(self):
        """The whole two-phase protocol relaxes through ``simulate`` with finite output."""
        r1, r2, t = run_scenario(sigma=0.0, dE=0.05, T_each=20.0)
        self.assertGreater(len(r1), 0)
        self.assertTrue(np.all(np.isfinite(r1)) and np.all(np.isfinite(r2)))
        self.assertEqual(len(r1), len(t))


# --- NEST side --------------------------------------------------------------------

def _nest_scenario(sigma, dE, dt=0.1, T_each=100.0, seed=0):
    """The two-unit WTA decision net in live NEST (``use_wfr=False``); ``(r1, r2)``.

    Mirrors :func:`examples.nest.rate_neuron_dm.run_scenario`: a no-evidence phase
    (``mu=0``) then an evidence phase (``mu = 1 +/- dE``). NEST continues the state across
    the two ``Simulate`` calls; for ``sigma=0`` phase 1 leaves both units at the initial
    ``rate=0``, so the evidence phase starts from the same state brainpy resets to.
    """
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt, 'use_wfr': False, 'rng_seed': seed + 1})
    par = {'lambda': LAM, 'sigma': sigma, 'tau': 1.0, 'rectify_output': True, 'mu': 0.0}
    d1 = nest.Create('lin_rate_ipn', 1, params=par)
    d2 = nest.Create('lin_rate_ipn', 1, params=par)
    inh = {'synapse_model': 'rate_connection_instantaneous', 'weight': W_INH}
    nest.Connect(d1, d2, 'all_to_all', inh)
    nest.Connect(d2, d1, 'all_to_all', inh)
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    nest.Connect(mm, d1 + d2)
    nest.Simulate(T_each)
    d1.mu = 1.0 + dE
    d2.mu = 1.0 - dE
    nest.Simulate(T_each)
    ev = mm.events
    senders = np.asarray(ev['senders'])
    times = np.asarray(ev['times'])
    rate = np.asarray(ev['rate'])

    def _trace(nid):
        m = senders == nid
        order = np.argsort(times[m], kind='stable')
        return rate[m][order]

    return _trace(d1.tolist()[0]), _trace(d2.tolist()[0])


def _bp_run(sigma, dE, seed, T_each=150.0):
    """brainpy ``(sigma, dE, seed) -> (r1, r2)`` adapter for the distributional helpers."""
    r1, r2, _ = run_scenario(sigma=sigma, dE=dE, T_each=T_each, seed=seed)
    return r1, r2


def _nest_run(sigma, dE, seed, T_each=150.0):
    """NEST ``(sigma, dE, seed) -> (r1, r2)`` adapter for the distributional helpers."""
    return _nest_scenario(sigma=sigma, dE=dE, T_each=T_each, seed=seed)


def _decision_summary(runner, sigma, dE, n_seeds=N_SEEDS):
    """Per-simulator decision statistics over ``n_seeds`` seeds, in a single pass.

    Returns
    -------
    n_d1_wins : int
        Number of seeds for which D1 is the decision winner.
    seed_mean_bias : float
        Seed-averaged ``|mean(r1_tail) - mean(r2_tail)|`` -- the balance magnitude.
        With ``n_seeds=5`` an unbiased process yields only ``{2, 6, 10}`` (3:2, 4:1, 5:0
        splits), so the fully-one-sided case is ``~10`` and a 4:1 sampling lean is ``~6``.
    """
    m1, m2, n_d1_wins = [], [], 0
    for s in range(n_seeds):
        r1, r2 = runner(sigma, dE, s)
        k = int(len(r1) * 0.8)
        a, b = float(np.mean(r1[k:])), float(np.mean(r2[k:]))
        n_d1_wins += int(a > b)
        m1.append(a)
        m2.append(b)
    return n_d1_wins, abs(float(np.mean(m1)) - float(np.mean(m2)))


@requires_nest
class TestRateNeuronDmNestParity(unittest.TestCase):
    """The two-unit WTA decision net matches live NEST (``use_wfr=False``)."""

    def test_deterministic_wta_matches_nest(self):
        """sigma=0 strong bias: same winner, matching winner rate, both losers ~ 0."""
        for dE in (0.1, -0.1):
            with self.subTest(dE=dE):
                br1, br2 = _bp_run(0.0, dE, 0, T_each=200.0)
                nr1, nr2 = _nest_run(0.0, dE, 0, T_each=200.0)
                self.assertEqual(_d1_wins(br1, br2), _d1_wins(nr1, nr2))   # same winner
                bw, bl = _winner_loser(br1, br2)
                nw, nl = _winner_loser(nr1, nr2)
                self.assertAlmostEqual(bw, nw, delta=1e-2)
                self.assertLess(bl, 1e-6)
                self.assertLess(nl, 1e-2)

    def test_decision_direction_matches_nest(self):
        """Strong +/-bias drives the decision the same way on both simulators."""
        bp_pos, _ = _decision_summary(_bp_run, 0.1, 0.1)
        ns_pos, _ = _decision_summary(_nest_run, 0.1, 0.1)
        self.assertGreaterEqual(bp_pos, (N_SEEDS + 1) // 2, f'brainpy +bias D1 wins {bp_pos}')
        self.assertGreaterEqual(ns_pos, (N_SEEDS + 1) // 2, f'NEST +bias D1 wins {ns_pos}')
        bp_neg, _ = _decision_summary(_bp_run, 0.1, -0.1)
        ns_neg, _ = _decision_summary(_nest_run, 0.1, -0.1)
        self.assertLessEqual(bp_neg, N_SEEDS // 2, f'brainpy -bias D1 wins {bp_neg}')
        self.assertLessEqual(ns_neg, N_SEEDS // 2, f'NEST -bias D1 wins {ns_neg}')

    def test_decision_contrast_matches_nest(self):
        """A biased noisy decision is winner-take-all (winner >> loser) on both sims."""
        for runner in (_bp_run, _nest_run):
            with self.subTest(runner=runner.__name__):
                win, lose = _winner_loser(*runner(0.1, 0.1, 0))
                self.assertGreater(win, 2.5 * max(lose, 1e-9))
                self.assertLess(lose, 1.0)
                self.assertGreater(win, 0.5 * STRONG_GAP)

    def test_zero_bias_balance_matches_nest(self):
        """At dE=0 neither simulator is locked to one unit (both win; no gross lean).

        With ``N_SEEDS=5`` an unbiased process can still split 4:1 by sampling (seed-mean
        bias ~6), so the magnitude guard rejects only the fully-one-sided ~10 case; the
        both-win check is the defining unbiased property.
        """
        for runner, name in ((_bp_run, 'brainpy'), (_nest_run, 'NEST')):
            with self.subTest(sim=name):
                n_d1, bias = _decision_summary(runner, 0.1, 0.0)
                self.assertGreater(n_d1, 0, f'{name}: D1 never won at dE=0 (biased)')
                self.assertLess(n_d1, N_SEEDS, f'{name}: D2 never won at dE=0 (biased)')
                self.assertLess(bias, 0.85 * STRONG_GAP, f'{name}: gross lean {bias:.2f}')


if __name__ == '__main__':
    unittest.main()
