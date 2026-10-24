# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST comparison of connectivity rule semantics and degree statistics.

Exact checks: edge counts, in-degrees and autapse exclusion on global ids for
overlapping and concatenated views (identical in NEST and here). Statistical
checks: multapse counts, out-degree variance and Bernoulli edge counts, averaged
over seeds on both sides, agree within the G2-style rule
``|Δmean| <= max(5 % of NEST mean, 2 x combined SEM)``.
"""
import unittest

import numpy as np

from brainspike._kernel import rng
from brainspike._kernel.connectivity import rules as R
from brainspike._nest_validation.nest_compare import requires_nest

N = 100
SEEDS = range(1, 9)


def _nest_edges(pre_slices, post_slice, conn_spec, seed):
    import nest
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.rng_seed = seed
    pop = nest.Create('iaf_psc_alpha', N)
    pre = pop[pre_slices[0]]
    for s in pre_slices[1:]:
        pre = pre + pop[s]
    post = pop[post_slice]
    nest.Connect(pre, post, conn_spec)
    c = nest.GetConnections(pre, post)
    first = pop[0].global_id
    return np.asarray(c.source) - first, np.asarray(c.target) - first


def _ours(pre_slices, post_slice, rule, seed):
    gids = np.arange(N)
    pre = np.concatenate([gids[s] for s in pre_slices])
    es = rule.build(pre, gids[post_slice], rng.object_key(seed, 0))
    return np.asarray(es.pre), np.asarray(es.post)


def _stats(src, tgt):
    pairs = src * N + tgt
    n_multapse = pairs.size - np.unique(pairs).size
    outdeg = np.bincount(src, minlength=N)
    return dict(n=src.size, autapses=int((src == tgt).sum()), multapses=n_multapse,
                outdeg_var=float(outdeg[outdeg > 0].var()) if src.size else 0.0)


def _agree(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sem = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return abs(a.mean() - b.mean()) <= max(0.05 * abs(b.mean()), 2 * sem)


@requires_nest
class TestRulesVsNest(unittest.TestCase):
    def _compare(self, pre_slices, post_slice, conn_spec, rule):
        ours = [_stats(*_ours(pre_slices, post_slice, rule, s)) for s in SEEDS]
        nest_ = [_stats(*_nest_edges(pre_slices, post_slice, conn_spec, s)) for s in SEEDS]
        return ours, nest_

    def test_fixed_indegree_overlap_no_autapses(self):
        ours, nest_ = self._compare([slice(0, 60)], slice(40, 100),
                                    {'rule': 'fixed_indegree', 'indegree': 10, 'allow_autapses': False},
                                    R.fixed_indegree(10, allow_autapses=False))
        for o, n in zip(ours, nest_):
            self.assertEqual(o['n'], n['n'])
            self.assertEqual(o['n'], 600)
            self.assertEqual(o['autapses'], 0)
            self.assertEqual(n['autapses'], 0)
        self.assertTrue(_agree([o['multapses'] for o in ours], [n['multapses'] for n in nest_]))
        self.assertTrue(_agree([o['outdeg_var'] for o in ours], [n['outdeg_var'] for n in nest_]))

    def test_fixed_indegree_concatenated_pre(self):
        # E + I composite: K per target in total (not K per part), no autapses
        spec = {'rule': 'fixed_indegree', 'indegree': 8, 'allow_autapses': False,
                'allow_multapses': False}
        rule = R.fixed_indegree(8, allow_autapses=False, allow_multapses=False)
        ours, nest_ = self._compare([slice(0, 30), slice(70, 100)], slice(20, 80), spec, rule)
        for o, n in zip(ours, nest_):
            self.assertEqual(o['n'], n['n'])
            self.assertEqual(o['n'], 60 * 8)
            self.assertEqual((o['autapses'], n['autapses']), (0, 0))
            self.assertEqual((o['multapses'], n['multapses']), (0, 0))

    def test_fixed_total_number_overlap(self):
        ours, nest_ = self._compare([slice(0, 60)], slice(40, 100),
                                    {'rule': 'fixed_total_number', 'N': 500, 'allow_autapses': False},
                                    R.fixed_total_number(500, allow_autapses=False))
        for o, n in zip(ours, nest_):
            self.assertEqual((o['n'], n['n']), (500, 500))
            self.assertEqual((o['autapses'], n['autapses']), (0, 0))
        self.assertTrue(_agree([o['multapses'] for o in ours], [n['multapses'] for n in nest_]))

    def test_fixed_total_number_refuses_no_multapses(self):
        import nest
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        pop = nest.Create('iaf_psc_alpha', 4)
        with self.assertRaises(Exception):
            nest.Connect(pop, pop, {'rule': 'fixed_total_number', 'N': 3, 'allow_multapses': False})
        with self.assertRaises(NotImplementedError):
            R.fixed_total_number(3, allow_multapses=False)

    def test_pairwise_bernoulli_overlap(self):
        ours, nest_ = self._compare([slice(0, 60)], slice(40, 100),
                                    {'rule': 'pairwise_bernoulli', 'p': 0.1, 'allow_autapses': False},
                                    R.pairwise_bernoulli(0.1, allow_autapses=False))
        for o, n in zip(ours, nest_):
            self.assertEqual((o['autapses'], n['autapses']), (0, 0))
            self.assertEqual((o['multapses'], n['multapses']), (0, 0))
        self.assertTrue(_agree([o['n'] for o in ours], [n['n'] for n in nest_]))

    def test_all_to_all_disjoint_slices(self):
        # goal-01 bug 5 against NEST: disjoint slices lose no edges
        ours, nest_ = self._compare([slice(0, 5)], slice(5, 10),
                                    {'rule': 'all_to_all', 'allow_autapses': False},
                                    R.all_to_all(allow_autapses=False))
        self.assertEqual(ours[0]['n'], 25)
        self.assertEqual(nest_[0]['n'], 25)
        ours, nest_ = self._compare([slice(0, 5)], slice(2, 7),
                                    {'rule': 'all_to_all', 'allow_autapses': False},
                                    R.all_to_all(allow_autapses=False))
        self.assertEqual(ours[0]['n'], nest_[0]['n'])
        self.assertEqual(ours[0]['n'], 22)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
