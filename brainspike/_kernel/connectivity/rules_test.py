# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import jax
import numpy as np
import pytest

from brainspike._kernel import rng
from brainspike._kernel.connectivity import rules as R
from brainspike._kernel.connectivity.edgeset import EdgeSet

KEY = rng.object_key(0, 4)


def pairs(es):
    return np.asarray(es.pre), np.asarray(es.post)


def n_autapses(es):
    p, q = pairs(es)
    return int((p == q).sum())


def indegree(es):
    return np.bincount(np.asarray(es.post_index), minlength=es.n_post)


def canonical(es):
    pi, qi = np.asarray(es.pre_index), np.asarray(es.post_index)
    return np.array_equal(np.lexsort((pi, qi)), np.arange(pi.size))


POP = np.arange(10)  # one population of 10 neurons, gids 0..9


class TestValidation:
    def test_flags(self):
        with pytest.raises(TypeError, match='allow_autapses'):
            R.AllToAll(allow_autapses=1)
        with pytest.raises(TypeError, match='allow_multapses'):
            R.AllToAll(allow_multapses=None)

    def test_counts(self):
        with pytest.raises(TypeError):
            R.fixed_indegree(2.0)
        with pytest.raises(TypeError):
            R.fixed_indegree(True)
        with pytest.raises(ValueError):
            R.fixed_indegree(-1)
        with pytest.raises(ValueError):
            R.fixed_total_number(-3)

    @pytest.mark.parametrize('p', [-0.1, 1.5])
    def test_probability(self, p):
        with pytest.raises(ValueError, match='p must be'):
            R.pairwise_bernoulli(p)

    def test_key_required(self):
        with pytest.raises(ValueError, match='needs a key'):
            R.fixed_indegree(2).build(POP, POP)
        # deterministic rules do not need one
        assert R.all_to_all().build(POP, POP).n_edges == 100

    def test_duplicate_gids_rejected(self):
        with pytest.raises(ValueError, match='duplicate'):
            R.all_to_all().build([0, 1, 1], POP)

    def test_fixed_total_number_multapses_refused(self):
        # mirrors NEST: NotImplemented for allow_multapses=False
        with pytest.raises(NotImplementedError, match='multapses'):
            R.fixed_total_number(5, allow_multapses=False)


class TestAllToAll:
    def test_full(self):
        es = R.all_to_all().build(POP[:3], POP[:4])
        assert es.n_edges == 12 and canonical(es) and es.max_multiplicity == 1

    def test_autapses_on_disjoint_slices(self):
        # goal-01 bug 5: pop[0:5] -> pop[5:10] without autapses must keep all 25 edges
        es = R.all_to_all(allow_autapses=False).build(POP[0:5], POP[5:10])
        assert es.n_edges == 25

    def test_autapses_on_overlapping_slices(self):
        es = R.all_to_all(allow_autapses=False).build(POP[0:5], POP[2:7])
        assert es.n_edges == 25 - 3 and n_autapses(es) == 0

    def test_concatenated_views(self):
        pre = np.concatenate([POP[0:3], POP[7:10]])   # gids 0,1,2,7,8,9
        post = np.concatenate([POP[8:10], POP[1:3]])  # gids 8,9,1,2 (unsorted concatenation)
        es = R.all_to_all(allow_autapses=False).build(pre, post)
        assert es.n_edges == 6 * 4 - 4 and n_autapses(es) == 0
        es = R.all_to_all().build(pre, post)
        assert es.n_edges == 24 and n_autapses(es) == 4

    def test_empty(self):
        assert R.all_to_all().build([], POP).n_edges == 0
        assert R.all_to_all().build(POP, []).n_edges == 0


class TestOneToOne:
    def test_basic(self):
        es = R.one_to_one().build(POP[:4], POP[4:8])
        assert list(np.asarray(es.pre)) == [0, 1, 2, 3] and list(np.asarray(es.post)) == [4, 5, 6, 7]

    def test_autapses_global(self):
        es = R.one_to_one(allow_autapses=False).build([0, 1, 2], [0, 5, 2])
        assert list(np.asarray(es.pre)) == [1] and list(np.asarray(es.post)) == [5]
        assert R.one_to_one().build([0, 1, 2], [0, 5, 2]).n_edges == 3

    def test_size_mismatch(self):
        with pytest.raises(ValueError, match='equal sizes'):
            R.one_to_one().build(POP[:3], POP[:4])


class TestFixedIndegree:
    @pytest.mark.parametrize('multapses', [True, False])
    def test_exact_indegree_no_autapses_on_overlap(self, multapses):
        pre, post = np.arange(0, 60), np.arange(40, 100)
        es = R.fixed_indegree(10, allow_autapses=False, allow_multapses=multapses).build(pre, post, KEY)
        assert es.n_edges == 600 and canonical(es)
        assert (indegree(es) == 10).all()
        assert n_autapses(es) == 0
        if not multapses:
            assert es.max_multiplicity == 1

    def test_concatenated_pre_is_one_population(self):
        # goal-01: E + I as one view -> K edges per target, not K per part
        E, I = np.arange(0, 80), np.arange(80, 100)
        es = R.fixed_indegree(8).build(np.concatenate([E, I]), E, KEY)
        assert (indegree(es) == 8).all()
        src = np.asarray(es.pre)
        assert (src >= 80).any() and (src < 80).any()

    def test_topk_and_floyd_branches(self):
        pre = np.arange(30)
        for k in (5, 29):  # 25 <= 120 -> Floyd; 841 > 120 -> top_k
            es = R.fixed_indegree(k, allow_autapses=False, allow_multapses=False).build(pre, pre, KEY)
            assert (indegree(es) == k).all() and n_autapses(es) == 0 and es.max_multiplicity == 1
        # k == available sources: every other neuron exactly once
        es = R.fixed_indegree(29, allow_autapses=False, allow_multapses=False).build(pre, pre, KEY)
        assert es.n_edges == 30 * 29

    @pytest.mark.parametrize('multapses,k', [(True, 4), (False, 3), (False, 40)])
    def test_uniform_marginal(self, multapses, k):
        # sources are uniform over the 49 available pre neurons (chi-square, 6 sigma)
        # targets 0..49 overlap the pre view (one excluded source each); targets
        # 1000.. do not, so their sources are uniform over all 50
        pre = np.arange(50)
        es = R.fixed_indegree(k, allow_autapses=False, allow_multapses=multapses).build(
            pre, np.concatenate([pre, np.arange(1000, 3000)]), KEY)
        assert (indegree(es) == k).all() and n_autapses(es) == 0
        src = np.asarray(es.pre)[np.asarray(es.post) >= 1000]
        counts = np.bincount(src, minlength=50)
        expect = src.size / 50
        chi2 = ((counts - expect) ** 2 / expect).sum()
        # without replacement the counts are less dispersed (chi2 below 49), never above
        assert chi2 < 49 + 6 * np.sqrt(2 * 49)
        if multapses:
            assert chi2 > 49 - 6 * np.sqrt(2 * 49)

    def test_post_view_independence(self):
        # a post neuron's sources depend only on its gid, not on the rest of the view
        pre = np.arange(100)
        a = R.fixed_indegree(7, allow_autapses=False).build(pre, np.array([3, 50, 2000]), KEY)
        b = R.fixed_indegree(7, allow_autapses=False).build(pre, np.array([2000, 3]), KEY)
        for g in (3, 2000):
            sa = np.sort(np.asarray(a.pre)[np.asarray(a.post) == g])
            sb = np.sort(np.asarray(b.pre)[np.asarray(b.post) == g])
            assert np.array_equal(sa, sb)

    def test_deterministic_and_key_dependent(self):
        r = R.fixed_indegree(5)
        a, b = r.build(POP, POP, KEY), r.build(POP, POP, KEY)
        assert a.same_edges(b)
        c = r.build(POP, POP, rng.object_key(0, 5))
        assert not a.same_edges(c)

    def test_x64_invariance(self):
        r = R.fixed_indegree(5, allow_multapses=False)
        a = r.build(np.arange(40), np.arange(40), KEY)
        with jax.enable_x64(True):
            b = r.build(np.arange(40), np.arange(40), KEY)
        assert a.same_edges(b)

    def test_errors(self):
        with pytest.raises(ValueError, match='exceeds'):
            R.fixed_indegree(10, allow_autapses=False, allow_multapses=False).build(POP, POP, KEY)
        with pytest.raises(ValueError, match='no available source'):
            R.fixed_indegree(1, allow_autapses=False).build([3], [3], KEY)
        with pytest.raises(ValueError, match='no available source'):
            R.fixed_indegree(1).build([], [3], KEY)

    def test_trivial(self):
        assert R.fixed_indegree(0).build(POP, POP, KEY).n_edges == 0
        assert R.fixed_indegree(3).build(POP, [], KEY).n_edges == 0
        # 10 == n_pre without autapse exclusion and multapses off: every source once
        es = R.fixed_indegree(10, allow_multapses=False).build(POP, POP[:2], KEY)
        assert es.n_edges == 20 and es.max_multiplicity == 1


class TestPairwiseBernoulli:
    def test_density_and_autapses(self):
        pre = post = np.arange(300)
        es = R.pairwise_bernoulli(0.1, allow_autapses=False).build(pre, post, KEY)
        n = 300 * 299
        assert abs(es.n_edges - 0.1 * n) < 6 * np.sqrt(n * 0.1 * 0.9)
        assert n_autapses(es) == 0 and es.max_multiplicity == 1 and canonical(es)

    def test_extremes(self):
        assert R.pairwise_bernoulli(0.0).build(POP, POP, KEY).n_edges == 0
        full = R.pairwise_bernoulli(1.0, allow_autapses=False).build(POP[0:5], POP[2:7], KEY)
        ref = R.all_to_all(allow_autapses=False).build(POP[0:5], POP[2:7])
        assert full.same_edges(ref)
        assert R.pairwise_bernoulli(0.5).build([], POP, KEY).n_edges == 0

    def test_block_size_invariance(self, monkeypatch):
        # the realised edges do not depend on how targets are blocked
        pre, post = np.arange(97), np.arange(50, 180)
        rule = R.pairwise_bernoulli(0.2, allow_autapses=False)
        a = rule.build(pre, post, KEY)
        monkeypatch.setattr(R, '_BLOCK_ELEMS', 97 * 7)  # 7 targets per block, ragged last block
        b = rule.build(pre, post, KEY)
        assert a.same_edges(b)

    def test_multapse_flag_ignored(self):
        a = R.pairwise_bernoulli(0.3).build(POP, POP, KEY)
        b = R.pairwise_bernoulli(0.3, allow_multapses=False).build(POP, POP, KEY)
        assert a.same_edges(b)


class TestFixedTotalNumber:
    def test_exact_count_no_autapses(self):
        es = R.fixed_total_number(500, allow_autapses=False).build(np.arange(0, 30), np.arange(20, 50), KEY)
        assert es.n_edges == 500 and n_autapses(es) == 0 and canonical(es)

    def test_multiple_rounds(self, monkeypatch):
        monkeypatch.setattr(R, '_BLOCK_ELEMS', 16)
        es = R.fixed_total_number(100, allow_autapses=False).build(POP[:3], POP[:3], KEY)
        assert es.n_edges == 100 and n_autapses(es) == 0

    def test_uniform_over_valid_pairs(self):
        # 2 pre x 2 post with pre == post and no autapses -> only (0,1), (1,0)
        es = R.fixed_total_number(4000, allow_autapses=False).build([0, 1], [0, 1], KEY)
        p, q = pairs(es)
        assert set(zip(p.tolist(), q.tolist())) == {(0, 1), (1, 0)}
        assert abs((p == 0).sum() - 2000) < 6 * np.sqrt(1000)

    def test_trivial_and_errors(self):
        es = R.fixed_total_number(5, allow_autapses=False).build([3], [4, 5], KEY)
        assert es.n_edges == 5
        assert R.fixed_total_number(0).build(POP, POP, KEY).n_edges == 0
        with pytest.raises(ValueError, match='no valid'):
            R.fixed_total_number(1, allow_autapses=False).build([4], [4], KEY)
        with pytest.raises(ValueError, match='no valid'):
            R.fixed_total_number(1, allow_autapses=False).build([], [4], KEY)


def test_aliases():
    assert R.all_to_all is R.AllToAll and R.one_to_one is R.OneToOne
    assert R.fixed_indegree(3).indegree == 3
    assert R.fixed_total_number(3).N == 3
    assert R.pairwise_bernoulli(0.25).p == 0.25
    assert isinstance(R.all_to_all().build(POP, POP), EdgeSet)


class TestPostKeysArgument:
    """K1a-6: the compiler passes per-post-neuron keys (population name + local id)."""

    @pytest.mark.parametrize('rule', [R.fixed_indegree(3), R.pairwise_bernoulli(0.3)],
                             ids=['indegree', 'bernoulli'])
    def test_explicit_keys_equal_default_derivation(self, rule):
        pre, post = np.arange(20), np.arange(20, 50)
        a = rule.build(pre, post, KEY)
        b = rule.build(pre, post, KEY, post_keys=rng.post_keys(KEY, post))
        assert a.same_edges(b)

    @pytest.mark.parametrize('rule', [R.fixed_indegree(3), R.pairwise_bernoulli(0.3)],
                             ids=['indegree', 'bernoulli'])
    def test_keys_decide_the_edges_not_the_gids(self, rule):
        # shifting the post gids with the same per-neuron keys gives the same edges
        pk = rng.view_post_keys(KEY, [('P', np.arange(30))])
        a = rule.build(np.arange(20), np.arange(20, 50), KEY, post_keys=pk)
        b = rule.build(np.arange(20), np.arange(120, 150), KEY, post_keys=pk)
        np.testing.assert_array_equal(np.asarray(a.pre_index), np.asarray(b.pre_index))
        np.testing.assert_array_equal(np.asarray(a.post_index), np.asarray(b.post_index))

    def test_wrong_number_of_keys(self):
        with pytest.raises(ValueError, match='post_keys'):
            R.fixed_indegree(2).build(np.arange(5), np.arange(5, 9), KEY,
                                      post_keys=rng.post_keys(KEY, np.arange(3)))
