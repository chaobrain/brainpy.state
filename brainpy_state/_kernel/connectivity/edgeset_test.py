# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.connectivity.edgeset import Connections, EdgeSet, as_gids
from brainpy_state._kernel.params import Normal, Uniform
from brainpy_state._kernel.units import UnitMismatchError


def small():
    # views: pre gids [10, 11, 12], post gids [20, 21]; a multapse (11 -> 20) twice
    return EdgeSet.create([0, 1, 1, 2], [0, 0, 0, 1], [10, 11, 12], [20, 21])


class TestAsGids:
    def test_ok(self):
        a = as_gids([3, 1, 2])
        assert a.dtype == np.int64 and list(a) == [3, 1, 2]
        assert as_gids([]).size == 0

    def test_errors(self):
        with pytest.raises(ValueError, match='1-D'):
            as_gids([[1, 2]])
        with pytest.raises(TypeError):
            as_gids([1.5, 2.0])
        with pytest.raises(ValueError, match='non-negative'):
            as_gids([-1, 2])
        with pytest.raises(ValueError, match='duplicate'):
            as_gids([1, 2, 1])


class TestCreate:
    def test_fields(self):
        es = small()
        assert (es.n_pre, es.n_post, es.n_edges, len(es)) == (3, 2, 4, 4)
        assert list(np.asarray(es.pre)) == [10, 11, 11, 12]
        assert list(np.asarray(es.post)) == [20, 20, 20, 21]
        assert es.pre_index.dtype == jnp.int32 and es.pre_gids.dtype == jnp.int32
        assert isinstance(es.pre_index, jax.Array)
        assert 'n_edges=4' in repr(es)

    def test_multiplicity(self):
        es = small()
        assert list(es.multiplicity()) == [1, 2, 2, 1]
        assert es.max_multiplicity == 2
        empty = EdgeSet.create([], [], [0], [1])
        assert empty.n_edges == 0 and empty.max_multiplicity == 0
        assert empty.multiplicity().size == 0

    def test_validation(self):
        with pytest.raises(ValueError, match='equal length'):
            EdgeSet.create([0, 1], [0], [0, 1], [0])
        with pytest.raises(ValueError, match='equal length'):
            EdgeSet.create(0, 0, [0], [0])
        with pytest.raises(ValueError, match='pre_index out of range'):
            EdgeSet.create([3], [0], [0, 1], [0])
        with pytest.raises(ValueError, match='post_index out of range'):
            EdgeSet.create([0], [-1], [0, 1], [0])
        with pytest.raises(TypeError, match='integers'):
            EdgeSet.create([0.0], [0], [0], [0])

    def test_from_gids(self):
        es = EdgeSet.from_gids([12, 10], [21, 20], [10, 11, 12], [21, 20])
        assert list(np.asarray(es.pre_index)) == [2, 0]
        assert list(np.asarray(es.post_index)) == [0, 1]
        with pytest.raises(ValueError, match='not in the pre view'):
            EdgeSet.from_gids([13], [20], [10, 11, 12], [20])
        with pytest.raises(ValueError, match='not in the post view'):
            EdgeSet.from_gids([10], [20], [10], [])
        assert EdgeSet.from_gids([], [], [1], [2]).n_edges == 0
        es = EdgeSet.from_gids([10], [20], [10], [20], weight=1. * u.pA, delay=2. * u.ms)
        assert es.weight.unit == u.pA and es.delay.unit == u.ms

    def test_pytree(self):
        es = small().with_weight(jnp.arange(4.) * u.pA)
        leaves, tree = jax.tree_util.tree_flatten(es)
        back = jax.tree_util.tree_unflatten(tree, leaves)
        assert back.same_edges(es)
        out = jax.jit(lambda e: e.weight * 2)(es)
        assert u.math.allclose(out, jnp.arange(4.) * 2 * u.pA)


class TestValues:
    def test_scalar_broadcast(self):
        es = small().with_weight(0.5 * u.nS)
        assert es.weight.shape == (4,) and es.weight.unit == u.nS
        es = small().with_weight(2)
        assert jnp.issubdtype(es.weight.dtype, jnp.floating)

    def test_array_and_dtype(self):
        es = small().with_weight(np.arange(4), dtype=jnp.float32)
        assert es.weight.dtype == jnp.float32 and list(np.asarray(es.weight)) == [0, 1, 2, 3]
        with pytest.raises(ValueError, match='shape'):
            small().with_weight(np.arange(3))

    def test_distribution(self):
        key = jax.random.PRNGKey(0)
        es = small().with_weight(Normal(1. * u.pA, 0.1 * u.pA), key=key)
        assert es.weight.shape == (4,) and es.weight.unit == u.pA
        es2 = small().with_weight(Normal(1. * u.pA, 0.1 * u.pA), key=key)
        assert u.math.allclose(es.weight, es2.weight)
        with pytest.raises(ValueError, match='key'):
            small().with_weight(Normal(1. * u.pA, 0.1 * u.pA))

    def test_delay(self):
        es = small().with_delay(1.5 * u.ms)
        assert es.delay.unit == u.ms
        es = small().with_delay(Uniform(1. * u.ms, 2. * u.ms), key=jax.random.PRNGKey(1))
        assert bool(u.math.all((es.delay >= 1. * u.ms) & (es.delay <= 2. * u.ms)))
        with pytest.raises(UnitMismatchError):
            small().with_delay(1. * u.pA)
        assert small().with_delay(1. * u.ms).set(delay=2. * u.ms).delay.unit == u.ms
        with pytest.raises(UnitMismatchError):
            small().with_delay(1. * u.ms).set(delay=2. * u.pA)


class TestGetConnections:
    def test_zero_weights_and_multapses_kept(self):
        # goal-01 bug 6: dense get_connections dropped zero weights and merged multapses
        es = small().with_weight(0. * u.pA)
        c = es.get_connections()
        assert isinstance(c, Connections) and len(c) == 4
        assert list(c.source) == [10, 11, 11, 12]
        assert u.math.allclose(c.weight, jnp.zeros(4) * u.pA)
        assert c.delay is None

    def test_filters(self):
        es = small().with_weight(jnp.arange(4.) * u.pA)
        c = es.get_connections(source=[11])
        assert list(c.edge) == [1, 2] and list(c.target) == [20, 20]
        c = es.get_connections(target=[21])
        assert list(c.edge) == [3] and list(c.source) == [12]
        c = es.get_connections(source=[10, 12], target=[20])
        assert list(c.edge) == [0]
        assert len(es.get_connections(source=[99])) == 0


class TestSet:
    def test_set_all(self):
        es = small().set(weight=jnp.array([1., 2., 3., 4.]) * u.pA)
        assert u.math.allclose(es.weight, jnp.array([1., 2., 3., 4.]) * u.pA)

    def test_set_aligned_subset(self):
        # goal-01: set('weight') must be aligned to the selected edges
        es = small().with_weight(jnp.zeros(4) * u.pA)
        c = es.get_connections(source=[11])
        es2 = es.set(weight=jnp.array([5., 7.]) * u.pA, edges=c)
        assert u.math.allclose(es2.weight, jnp.array([0., 5., 7., 0.]) * u.pA)
        # unit conversion into the stored unit
        es3 = es.set(weight=1. * u.nA, edges=[3])
        assert u.math.allclose(es3.weight, jnp.array([0., 0., 0., 1000.]) * u.pA)
        # bool mask
        es4 = es.set(weight=2. * u.pA, edges=np.array([True, False, False, True]))
        assert u.math.allclose(es4.weight, jnp.array([2., 0., 0., 2.]) * u.pA)
        # the original is unchanged (immutable)
        assert u.math.allclose(es.weight, jnp.zeros(4) * u.pA)

    def test_set_distribution(self):
        es = small().with_weight(jnp.zeros(4) * u.pA)
        es2 = es.set(weight=Uniform(1. * u.pA, 2. * u.pA), edges=[0, 3], key=jax.random.PRNGKey(0))
        w = np.asarray(es2.weight.mantissa)
        assert w[1] == 0 and w[2] == 0 and 1 <= w[0] <= 2 and 1 <= w[3] <= 2

    def test_set_errors(self):
        es = small()
        with pytest.raises(ValueError, match='before it is assigned'):
            es.set(weight=1. * u.pA, edges=[0])
        es = es.with_weight(jnp.zeros(4) * u.pA)
        with pytest.raises(UnitMismatchError):
            es.set(weight=1. * u.mV, edges=[0])
        with pytest.raises(UnitMismatchError):
            es.set(weight=1. * u.mV)
        with pytest.raises(ValueError, match='shape'):
            es.set(weight=jnp.ones(3) * u.pA, edges=[0, 1])
        with pytest.raises(ValueError, match='out of range'):
            es.set(weight=1. * u.pA, edges=[4])
        with pytest.raises(ValueError, match='mask'):
            es.set(weight=1. * u.pA, edges=np.array([True, False]))
        assert es.set().same_edges(es)


class TestSameEdges:
    def test_compare(self):
        assert small().same_edges(small())
        other = EdgeSet.create([0, 1, 1, 2], [0, 0, 1, 1], [10, 11, 12], [20, 21])
        assert not small().same_edges(other)
