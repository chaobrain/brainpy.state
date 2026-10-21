# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel import rng

fold = jax.random.fold_in


def same(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


class TestDeriveKey:
    def test_formula(self):
        k = rng.derive_key(7, 3, 2, 11)
        expect = fold(fold(fold(jax.random.PRNGKey(7), 3), 2), 11)
        assert same(k, expect)

    def test_composition(self):
        ok = rng.object_key(1, 5)
        assert same(rng.step_key(rng.block_key(ok, 4), 9), rng.derive_key(1, 5, 4, 9))

    def test_every_counter_matters(self):
        base = rng.derive_key(0, 0, 0, 0)
        for args in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]:
            assert not same(rng.derive_key(*args), base)

    def test_counters_not_interchangeable(self):
        assert not same(rng.derive_key(0, 1, 2, 3), rng.derive_key(0, 2, 1, 3))

    def test_traced_step(self):
        f = jax.jit(lambda s: rng.population_key(0, 2, s))
        assert same(f(jnp.int32(5)), rng.derive_key(0, 2, 0, 5))
        steps = jnp.arange(3)
        ks = jax.vmap(lambda s: rng.population_key(0, 2, s))(steps)
        assert same(ks[2], rng.derive_key(0, 2, 0, 2))

    def test_connectivity_key_reserved_step(self):
        assert same(rng.connectivity_key(0, 1, 3), rng.derive_key(0, 1, 3, rng.CONNECT_STEP))
        assert not same(rng.connectivity_key(0, 1, 0), rng.population_key(0, 1, 0))

    @pytest.mark.parametrize('bad', [-1, 2 ** 32])
    def test_counter_range(self, bad):
        with pytest.raises(ValueError):
            rng.derive_key(0, bad)
        with pytest.raises(ValueError):
            rng.root_key(bad)

    @pytest.mark.parametrize('bad', [True, 1.5, jnp.float32(1.0)])
    def test_counter_type(self, bad):
        with pytest.raises(TypeError):
            rng.derive_key(0, 0, 0, bad)

    def test_numpy_int_counter(self):
        assert same(rng.derive_key(np.int64(0), np.int32(1)), rng.derive_key(0, 1))


class TestPostKeys:
    def test_matches_derivation(self):
        ok = rng.object_key(0, 9)
        gids = np.array([0, 5, 1023, 1024, 5000])
        ks = rng.post_keys(ok, gids)
        assert ks.shape == (5, 2)
        for i, g in enumerate(gids):
            expect = fold(rng.connectivity_key(0, 9, g // rng.GID_BLOCK), g % rng.GID_BLOCK)
            assert same(ks[i], expect)

    def test_independent_of_view(self):
        ok = rng.object_key(3, 1)
        a = rng.post_keys(ok, np.array([10, 20, 30]))
        b = rng.post_keys(ok, np.array([30, 99, 10]))
        assert same(a[0], b[2]) and same(a[2], b[0])

    def test_distinct(self):
        ks = np.asarray(rng.post_keys(rng.object_key(0, 0), np.arange(3000)))
        assert np.unique(ks, axis=0).shape[0] == 3000

    def test_empty_and_errors(self):
        assert rng.post_keys(rng.object_key(0, 0), np.zeros(0, int)).shape == (0, 2)
        with pytest.raises(ValueError):
            rng.post_keys(rng.object_key(0, 0), np.array([-1]))


class TestTrialKeys:
    def test_unbatched_is_trial0(self):
        k = rng.population_key(0, 1, 4)
        for mode in rng.RNG_MODES:
            assert same(rng.trial_keys(k, None, mode), fold(k, 0))
            assert same(rng.trial_keys(k, 1, mode)[0], rng.trial_keys(k))
            assert same(rng.trial_keys(k, 4, mode)[0], rng.trial_keys(k))

    def test_independent(self):
        k = rng.root_key(0)
        ks = rng.trial_keys(k, 4, 'independent')
        assert ks.shape == (4, 2)
        assert np.unique(np.asarray(ks), axis=0).shape[0] == 4
        assert same(ks[3], fold(k, 3))
        # batch 2 is a prefix of batch 4
        assert same(rng.trial_keys(k, 2), ks[:2])

    def test_common(self):
        k = rng.root_key(0)
        ks = rng.trial_keys(k, 3, 'common')
        assert ks.shape == (3, 2)
        assert all(same(ks[i], ks[0]) for i in range(3))

    def test_draws_under_vmap(self):
        # independent trials draw different numbers; common trials identical ones,
        # and trial 0 equals the unbatched draw in both modes.
        k = rng.population_key(0, 2, 7)
        draw = lambda kk: jax.random.uniform(kk, (5,))
        single = draw(rng.trial_keys(k))
        ind = jax.vmap(draw)(rng.trial_keys(k, 4, 'independent'))
        com = jax.vmap(draw)(rng.trial_keys(k, 4, 'common'))
        assert same(ind[0], single) and same(com[0], single)
        assert not same(ind[1], ind[0])
        assert all(same(com[i], single) for i in range(4))

    @pytest.mark.parametrize('bad', [0, -1, 1.5, True])
    def test_bad_batch(self, bad):
        with pytest.raises(ValueError):
            rng.trial_keys(rng.root_key(0), bad)

    def test_bad_mode(self):
        with pytest.raises(ValueError, match='mode'):
            rng.trial_keys(rng.root_key(0), 2, 'shared')


class TestUidAllocator:
    def test_sequence(self):
        a = rng.UidAllocator()
        assert [a.next() for _ in range(3)] == [0, 1, 2]
        assert a.count == 3

    def test_start(self):
        a = rng.UidAllocator(10)
        assert a.next() == 10 and a.count == 1

    def test_deterministic_across_instances(self):
        a, b = rng.UidAllocator(), rng.UidAllocator()
        assert [a.next() for _ in range(5)] == [b.next() for _ in range(5)]

    def test_exhausted(self):
        a = rng.UidAllocator(2 ** 32 - 1)
        assert a.next() == 2 ** 32 - 1
        with pytest.raises(OverflowError):
            a.next()
