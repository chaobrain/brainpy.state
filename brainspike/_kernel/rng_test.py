# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel import rng

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


class TestNameKeys:
    def test_name_id_is_stable_blake2b(self):
        import hashlib
        d = hashlib.blake2b(b'E', digest_size=8).digest()
        assert rng.name_id('E') == (int.from_bytes(d[:4], 'big'), int.from_bytes(d[4:], 'big'))
        assert rng.name_id('E') != rng.name_id('I')
        assert all(0 <= w < 2 ** 32 for w in rng.name_id('\u00e9\u4e2d'))

    @pytest.mark.parametrize('bad', ['', None, 3])
    def test_name_id_rejects(self, bad):
        with pytest.raises(TypeError):
            rng.name_id(bad)

    def test_object_key_from_name(self):
        hi, lo = rng.name_id('pop')
        assert same(rng.object_key(5, 'pop'), fold(fold(jax.random.PRNGKey(5), hi), lo))
        assert not same(rng.object_key(5, 'pop'), rng.object_key(5, 'pop2'))
        assert not same(rng.object_key(5, 'pop'), rng.object_key(6, 'pop'))
        # derive_key / population_key / connectivity_key accept names
        assert same(rng.derive_key(1, 'a', 2, 3),
                    rng.step_key(rng.block_key(rng.object_key(1, 'a'), 2), 3))
        assert same(rng.population_key(1, 'a', 3, 2), rng.derive_key(1, 'a', 2, 3))
        assert same(rng.connectivity_key(1, 'a', 2), rng.derive_key(1, 'a', 2, rng.CONNECT_STEP))


class TestViewPostKeys:
    def test_local_ids_and_population_name(self):
        ok = rng.object_key(0, 'proj')
        parts = [('E', np.array([0, 1, 1500])), ('I', np.array([0, 7]))]
        ks = rng.view_post_keys(ok, parts)
        assert ks.shape == (5, 2)
        for (name, local), sl in zip(parts, [slice(0, 3), slice(3, 5)]):
            hi, lo = rng.name_id(name)
            assert same(ks[sl], rng.post_keys(fold(fold(ok, hi), lo), local))
        # same local id in different populations -> different keys
        assert not same(ks[0], ks[3])

    def test_view_order_and_empty(self):
        ok = rng.object_key(0, 'p')
        a = rng.view_post_keys(ok, [('E', np.array([3, 4]))])
        b = rng.view_post_keys(ok, [('I', np.array([0])), ('E', np.array([4, 3]))])
        assert same(a[0], b[2]) and same(a[1], b[1])
        assert rng.view_post_keys(ok, []).shape == (0, 2)


class TestBlockedDraws:
    B = rng.GID_BLOCK
    uni = staticmethod(lambda k, s: jax.random.uniform(k, s))

    @pytest.mark.parametrize('n, nb', [(0, 0), (1, 1), (1024, 1), (1025, 2), (5000, 5)])
    def test_n_blocks(self, n, nb):
        assert rng.n_blocks(n) == nb
        assert rng.population_block_keys(0, 'x', n).shape == (nb, 2)

    def test_block_and_step_keys_formula(self):
        bk = rng.population_block_keys(3, 'E', 2100)
        ks = jax.jit(rng.step_block_keys)(bk, jnp.int32(9), jnp.uint32(2))
        for b in range(3):
            assert same(bk[b], rng.block_key(rng.object_key(3, 'E'), b))
            assert same(ks[b], fold(rng.population_key(3, 'E', 9, b), 2))
        assert same(rng.step_block_keys(bk, 9)[1], fold(rng.population_key(3, 'E', 9, 1), 0))

    def test_extension_invariance(self):
        # a neuron's draw does not depend on the population size
        big = rng.step_block_keys(rng.population_block_keys(0, 'P', 3 * self.B + 5), 4)
        small = rng.step_block_keys(rng.population_block_keys(0, 'P', 700), 4)
        a = rng.draw_blocked(big, 3 * self.B + 5, self.uni)
        b = rng.draw_blocked(small, 700, self.uni)
        assert same(a[:700], b)

    def test_shard_invariance_block_aligned(self):
        # drawing as one array == drawing block-aligned pieces with their own keys
        n = 3 * self.B + 17
        ks = rng.step_block_keys(rng.population_block_keys(1, 'P', n), 0)
        whole = rng.draw_blocked(ks, n, self.uni)
        pieces = [rng.draw_blocked(ks[0:1], self.B, self.uni),
                  rng.draw_blocked(ks[1:3], 2 * self.B, self.uni),
                  rng.draw_blocked(ks[3:4], 17, self.uni)]
        assert same(whole, jnp.concatenate(pieces))

    def test_lane_args_and_tail(self):
        n = self.B + 3
        ks = rng.step_block_keys(rng.population_block_keys(0, 'P', n), 0)
        lam = jnp.arange(n, dtype=jnp.float32)
        out = rng.draw_blocked(ks, n, lambda k, s, l: l + 0 * jax.random.uniform(k, s), lam)
        assert same(out, lam)
        two = rng.draw_blocked(ks, n, lambda k, s: jax.random.normal(k, s), tail=(2,))
        assert two.shape == (n, 2)

    @pytest.mark.parametrize('typed', [False, True])
    def test_single_key(self, typed):
        k = jax.random.key(4) if typed else jax.random.PRNGKey(4)
        out = rng.draw_blocked(k, self.B + 1, self.uni)
        ks = jax.vmap(lambda b: fold(k, b))(jnp.arange(2, dtype=jnp.uint32))
        assert same(out, rng.draw_blocked(ks, self.B + 1, self.uni))
        assert rng.draw_blocked(k, 0, self.uni).shape == (0,)

    def test_wrong_number_of_block_keys(self):
        ks = rng.population_block_keys(0, 'P', 10)
        with pytest.raises(ValueError, match='block keys'):
            rng.draw_blocked(ks, 2000, self.uni)

    def test_vmap_trials(self):
        bk = rng.population_block_keys(0, 'P', 50)
        f = lambda t: rng.draw_blocked(rng.step_block_keys(bk, 3, t), 50, self.uni)
        out = jax.vmap(f)(jnp.arange(3, dtype=jnp.uint32))
        assert same(out[0], f(0)) and not same(out[1], out[0])
