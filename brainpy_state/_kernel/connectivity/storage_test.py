# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import warnings

import brainstate
import braintools
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from brainpy_state._kernel import rng
from brainpy_state._kernel.connectivity import rules as R
from brainpy_state._kernel.connectivity import storage as S
from brainpy_state._kernel.connectivity.edgeset import EdgeSet
from brainpy_state._kernel.ports import Event

STORAGES = [S.Dense, S.CSR]


def edges_with_multapses():
    # 5 pre x 4 post; pre 1 -> post 2 three times, pre 0 -> post 0 twice; post 3 has no input
    pre = [0, 0, 1, 1, 1, 2, 4, 3]
    post = [0, 0, 2, 2, 2, 1, 1, 0]
    return EdgeSet.create(pre, post, np.arange(5), np.arange(10, 14))


def reference(x, w_edges, es):
    """Exact per-target sum over edges, as a dense jnp computation."""
    pre = jnp.asarray(es.pre_index)
    post = jnp.asarray(es.post_index)
    contrib = x[..., pre] * w_edges
    return jnp.zeros(x.shape[:-1] + (es.n_post,), contrib.dtype).at[..., post].add(contrib)


W_E = jnp.array([0.5, -1.0, 2.0, 0.25, 0.0, 1.5, -0.75, 3.0])
COUNTS = jnp.array([2., 1., 3., 0., 1.])  # counts in {0, 1, 2, 3}
GOUT = jnp.array([1.0, -2.0, 0.5, 4.0])


class TestPolicy:
    def test_payload_path(self):
        assert S.payload_path(Event()) == 'binary'
        assert S.payload_path(Event(max_multiplicity=2)) == 'float'
        assert S.payload_path(Event(max_multiplicity=None)) == 'float'
        assert S.payload_path(1) == 'binary'
        assert S.payload_path(None) == 'float'
        assert S.payload_path(np.int64(4)) == 'float'
        for bad in (0, True, 1.0, 'one'):
            with pytest.raises(TypeError):
                S.payload_path(bad)

    def test_csr_index_dtypes(self):
        assert S.csr_index_dtypes(10, 10, 100) == (np.int32, np.int32)
        assert S.csr_index_dtypes(10, 10, 2 ** 31 - 1) == (np.int32, np.int32)
        with jax.enable_x64(False):
            with pytest.raises(RuntimeError, match='x64'):
                S.csr_index_dtypes(10, 10, 2 ** 31)
        with jax.enable_x64(True):
            assert S.csr_index_dtypes(10, 10, 2 ** 31) == (np.int32, np.int64)
            assert S.csr_index_dtypes(2 ** 31, 10, 5) == (np.int64, np.int32)

    def test_csr_structure_dtypes(self):
        s = S.CSR(edges_with_multapses())
        assert s.indices.dtype == jnp.int32 and s.indptr.dtype == jnp.int32
        assert list(np.asarray(s.indptr)) == [0, 2, 5, 6, 7, 8]

    def test_make_storage(self):
        es = edges_with_multapses()
        assert isinstance(S.make_storage(es, 'dense'), S.Dense)
        assert isinstance(S.make_storage(es, 'csr'), S.CSR)
        assert isinstance(S.make_storage(es), S.Dense)  # density 8/20
        assert isinstance(S.make_storage(es, dense_threshold=0.5), S.CSR)
        assert isinstance(S.make_storage(es, dense_max_elems=10), S.CSR)
        assert isinstance(S.make_storage(EdgeSet.create([], [], [], [0])), S.CSR)
        with pytest.raises(ValueError, match='kind'):
            S.make_storage(es, 'jitc')
        with pytest.raises(TypeError, match='EdgeSet'):
            S.Dense('edges')

    def test_nbytes_and_repr(self):
        es = edges_with_multapses()
        assert S.Dense(es).nbytes() == 20 * 4 + 20
        assert S.Dense(R.all_to_all().build(np.arange(3), np.arange(3))).nbytes(8) == 9 * 8
        assert S.CSR(es).nbytes() == 8 * (4 + 4) + 6 * 4
        assert 'n_edges=8' in repr(S.CSR(es))


class TestWeightLayout:
    def test_csr_round_trip_exact(self):
        es = edges_with_multapses()
        s = S.CSR(es)
        w = s.weight_from_edges(W_E * u.pA)
        assert w.unit == u.pA and w.shape == (8,)
        assert u.math.allclose(s.weight_to_edges(w), W_E * u.pA)

    def test_dense_sums_multapses(self):
        es = edges_with_multapses()
        s = S.Dense(es)
        W = s.weight_from_edges(W_E)
        assert W.shape == (5, 4)
        assert float(W[0, 0]) == pytest.approx(-0.5) and float(W[1, 2]) == pytest.approx(2.25)
        back = s.weight_to_edges(W)
        # multapse groups get an equal split, which preserves their sum
        assert float(back[0]) == pytest.approx(-0.25) and float(back[2]) == pytest.approx(0.75)
        es1 = R.all_to_all().build(np.arange(3), np.arange(4)).with_weight(jnp.arange(12.))
        s1 = S.Dense(es1)
        assert jnp.allclose(s1.weight_to_edges(s1.weight_from_edges(es1.weight)), es1.weight)

    def test_shape_errors(self):
        es = edges_with_multapses()
        for cls in STORAGES:
            with pytest.raises(ValueError, match='per-edge'):
                cls(es).weight_from_edges(jnp.ones(3))
        with pytest.raises(ValueError, match='dense weights'):
            S.Dense(es).weight_to_edges(jnp.ones((4, 4)))
        with pytest.raises(ValueError, match='CSR weights'):
            S.CSR(es).weight_to_edges(jnp.ones(3))
        with pytest.raises(ValueError, match='CSR weights'):
            S.CSR(es).forward(COUNTS, jnp.ones(3))


@pytest.mark.parametrize('cls', STORAGES)
class TestForward:
    def test_float_path_exact_counts(self, cls):
        es = edges_with_multapses()
        s = cls(es)
        y = s.forward(COUNTS, s.weight_from_edges(W_E * u.pA))
        assert y.unit == u.pA
        assert jnp.allclose(y.mantissa, reference(COUNTS, W_E, es))

    def test_binary_path_matches_float_for_binary(self, cls):
        es = edges_with_multapses()
        s = cls(es)
        w = s.weight_from_edges(W_E)
        x = (COUNTS > 0).astype(jnp.float32)
        assert jnp.allclose(s.forward(x, w, 'binary'), reference(x, W_E, es))
        assert jnp.allclose(s.forward(x > 0, w, 'binary'), reference(x, W_E, es))

    def test_batched_and_vmap(self, cls):
        es = edges_with_multapses()
        s = cls(es)
        w = s.weight_from_edges(W_E)
        xb = jnp.stack([COUNTS, COUNTS[::-1], jnp.zeros(5), jnp.full(5, 3.)])
        ref = reference(xb, W_E, es)
        assert jnp.allclose(s.forward(xb, w), ref)
        assert jnp.allclose(jax.vmap(lambda x: s.forward(x, w))(xb), ref)
        xbin = (xb > 1).astype(jnp.float32)
        rb = reference(xbin, W_E, es)
        assert jnp.allclose(jax.vmap(lambda x: s.forward(x, w, 'binary'))(xbin), rb)
        assert jnp.allclose(s.forward(xbin, w, 'binary'), rb)

    def test_jit(self, cls):
        es = edges_with_multapses()
        s = cls(es)
        w = s.weight_from_edges(W_E)
        f = jax.jit(lambda x, w: s.forward(x, w))
        assert jnp.allclose(f(COUNTS, w), reference(COUNTS, W_E, es))

    def test_vjp_float_path(self, cls):
        # gradients w.r.t. per-edge weights (through the native layout) and payload
        es = edges_with_multapses()
        s = cls(es)
        loss = lambda x, we: jnp.sum(s.forward(x, s.weight_from_edges(we)) * GOUT)
        loss_ref = lambda x, we: jnp.sum(reference(x, we, es) * GOUT)
        gx, gw = jax.grad(loss, (0, 1))(COUNTS, W_E)
        rx, rw = jax.grad(loss_ref, (0, 1))(COUNTS, W_E)
        assert jnp.allclose(gx, rx) and jnp.allclose(gw, rw)
        # d/dw_e = count[pre_e] * g[post_e]: counts > 1 must scale the gradient
        assert float(gw[2]) == pytest.approx(float(COUNTS[1] * GOUT[2]))

    def test_vjp_native_weights(self, cls):
        es = edges_with_multapses()
        s = cls(es)
        w = s.weight_from_edges(W_E)
        g = jax.grad(lambda w: jnp.sum(s.forward(COUNTS, w) * GOUT))(w)
        gref = jax.grad(lambda we: jnp.sum(reference(COUNTS, we, es) * GOUT))(W_E)
        if cls is S.CSR:
            assert jnp.allclose(s.weight_to_edges(g), gref)
        else:
            # dense: gradient at every edge entry, exactly zero at pairs without an edge
            assert jnp.allclose(g[np.asarray(es.pre_index), np.asarray(es.post_index)], gref)
            mask = np.zeros((5, 4), bool)
            mask[np.asarray(es.pre_index), np.asarray(es.post_index)] = True
            assert np.all(np.asarray(g)[~mask] == 0)

    def test_vjp_binary_path(self, cls):
        es = edges_with_multapses()
        s = cls(es)
        x = jnp.array([1., 0., 1., 1., 0.])
        loss = lambda x, we: jnp.sum(s.forward(x, s.weight_from_edges(we), 'binary') * GOUT)
        loss_ref = lambda x, we: jnp.sum(reference(x, we, es) * GOUT)
        gx, gw = jax.grad(loss, (0, 1))(x, W_E)
        rx, rw = jax.grad(loss_ref, (0, 1))(x, W_E)
        assert jnp.allclose(gx, rx) and jnp.allclose(gw, rw)

    def test_vjp_under_vmap(self, cls):
        es = edges_with_multapses()
        s = cls(es)
        xb = jnp.stack([COUNTS, jnp.array([3., 3., 0., 2., 1.])])
        loss = lambda we: jnp.sum(jax.vmap(lambda x: s.forward(x, s.weight_from_edges(we)))(xb) * GOUT)
        loss_ref = lambda we: jnp.sum(reference(xb, we, es) * GOUT)
        assert jnp.allclose(jax.grad(loss)(W_E), jax.grad(loss_ref)(W_E))

    def test_float64_check_grads(self, cls):
        with jax.enable_x64(True):
            es = edges_with_multapses()
            s = cls(es)
            we = jnp.asarray(np.asarray(W_E), jnp.float64)
            x = jnp.asarray(np.asarray(COUNTS), jnp.float64)
            f = lambda x, we: s.forward(x, s.weight_from_edges(we))
            check_grads(f, (x, we), order=1, modes=['rev'])
            assert f(x, we).dtype == jnp.float64

    def test_empty(self, cls):
        es = EdgeSet.create([], [], np.arange(3), np.arange(2))
        s = cls(es)
        w = s.weight_from_edges(jnp.zeros((0,)))
        assert jnp.array_equal(s.forward(jnp.ones(3), w), jnp.zeros(2))
        assert s.forward(jnp.ones((4, 3)), w).shape == (4, 2)
        g = jax.grad(lambda x: jnp.sum(s.forward(x, w)))(jnp.ones(3))
        assert jnp.array_equal(g, jnp.zeros(3))

    def test_errors(self, cls):
        s = cls(edges_with_multapses())
        w = s.weight_from_edges(W_E)
        with pytest.raises(ValueError, match='path'):
            s.forward(COUNTS, w, 'event')
        with pytest.raises(ValueError, match='payload'):
            s.forward(jnp.ones(4), w)
        with pytest.raises(ValueError, match='payload'):
            s.forward(jnp.ones((2, 2, 5)), w)
        with pytest.raises(TypeError, match='dimensionless'):
            s.forward(COUNTS * u.mV, w)
        assert jnp.allclose(s.forward(COUNTS * u.UNITLESS, w), s.forward(COUNTS, w))

    def test_etrace_forward_matches(self, cls):
        es = edges_with_multapses()
        s = cls(es)
        w = s.weight_from_edges(W_E * u.pA)
        y = s.forward_etrace(COUNTS, w)
        assert u.math.allclose(y, s.forward(COUNTS, w))
        g = jax.grad(lambda we: jnp.sum(s.forward_etrace(COUNTS, s.weight_from_edges(we)) * GOUT))(W_E)
        gref = jax.grad(lambda we: jnp.sum(reference(COUNTS, we, es) * GOUT))(W_E)
        assert jnp.allclose(g, gref)


# ---------------------------------------------------------------------------
# braintrace: online learning sees the projection weight (output read from a HiddenState)
# ---------------------------------------------------------------------------

class _TinyNet(brainstate.nn.Module):
    """Input -> projection (storage) -> leaky units; output read from the HiddenState."""

    def __init__(self, storage, w0):
        super().__init__()
        self.storage = storage
        self.w = brainstate.ParamState(w0)
        self.f = braintools.surrogate.Sigmoid()

    def init_state(self, *args, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros((self.storage.n_post,)))

    def update(self, x):
        V = 0.9 * self.V.value + self.storage.forward_etrace(x, self.w.value)
        self.V.value = V - jax.lax.stop_gradient(self.f(V - 1.0))
        return self.f(self.V.value - 0.5)


@pytest.mark.parametrize('cls', STORAGES)
@pytest.mark.parametrize('algorithm', ['d_rtrl', 'eprop'])
def test_braintrace_nonzero_weight_gradients(cls, algorithm):
    n_in, n_out = 6, 5
    es = R.fixed_indegree(3).build(np.arange(n_in), np.arange(100, 100 + n_out), rng.object_key(0, 1))
    s = cls(es)
    w0 = s.weight_from_edges(jnp.linspace(0.2, 0.8, es.n_edges))
    net = _TinyNet(s, w0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        learner = braintrace.compile(net, algorithm, jnp.ones((n_in,)))
        xs = (jax.random.uniform(jax.random.PRNGKey(0), (20, n_in)) < 0.5).astype(jnp.float32) * 2.0
        ys = jnp.full((20, n_out), 0.3)
        grads = learner.etrace_grad(xs, ys, step_fn=lambda x, y: jnp.mean((learner(x) - y) ** 2))
    leaves = jax.tree.leaves(grads)
    assert len(leaves) == 1
    g = np.asarray(leaves[0])
    assert g.shape == np.shape(w0)
    if cls is S.CSR:
        assert np.all(g != 0)  # every edge receives a gradient
    else:
        mask = np.zeros((n_in, n_out), bool)
        mask[np.asarray(es.pre_index), np.asarray(es.post_index)] = True
        assert np.all(g[mask] != 0) and np.all(g[~mask] == 0)


def _matching_edges(kind):
    if kind == 'identity':  # one_to_one onto the same size
        return EdgeSet.create(np.arange(6), np.arange(6), np.arange(6), np.arange(10, 16))
    if kind == 'permutation':
        return EdgeSet.create(np.arange(6), [3, 0, 5, 1, 4, 2], np.arange(6), np.arange(10, 16))
    # partial, with a multapse and a target receiving two sources: 4 edges <= max(3, 5)
    return EdgeSet.create([0, 0, 2, 1], [4, 4, 1, 1], np.arange(3), np.arange(5))


@pytest.mark.parametrize('kind', ['identity', 'permutation', 'partial'])
class TestGatherPath:
    """CSR with n_edges <= max(n_pre, n_post) bypasses the sparse kernel (exact)."""

    def _setup(self, kind):
        es = _matching_edges(kind)
        s = S.CSR(es)
        we = jnp.linspace(-1.0, 2.0, es.n_edges)
        x = jnp.arange(1.0, es.n_pre + 1.0)  # counts > 1
        return es, s, we, x

    def test_flags(self, kind):
        es, s, _, _ = self._setup(kind)
        assert s.gather and s.identity == (kind == 'identity')
        assert (s.rows is None) == (kind == 'identity')
        assert set(s.arrays()) == ({'indices'} if kind == 'identity' else {'indices', 'rows'})
        assert s.nbytes() > 0
        big = EdgeSet.create(np.repeat(np.arange(4), 3), np.tile(np.arange(3), 4),
                             np.arange(4), np.arange(3))
        assert not S.CSR(big).gather and S.CSR(big).rows is None

    def test_forward_matches_reference(self, kind):
        es, s, we, x = self._setup(kind)
        w = s.weight_from_edges(we * u.pA)
        y = s.forward(x, w)
        assert y.unit == u.pA
        np.testing.assert_allclose(y.mantissa, reference(x, we, es), rtol=1e-6)
        xb = (x > 2).astype(jnp.float32)
        np.testing.assert_allclose(s.forward(xb, w, 'binary').mantissa, reference(xb, we, es),
                                   rtol=1e-6)
        xs = jnp.stack([x, -x, jnp.zeros_like(x)])
        np.testing.assert_allclose(s.forward(xs, w).mantissa, reference(xs, we, es), rtol=1e-6)
        np.testing.assert_allclose(jax.vmap(lambda v: s.forward(v, w).mantissa)(xs),
                                   reference(xs, we, es), rtol=1e-6)

    def test_gradients_and_etrace(self, kind):
        es, s, we, x = self._setup(kind)
        g = jnp.arange(1.0, es.n_post + 1.0)
        loss = lambda x, we: jnp.sum(s.forward(x, s.weight_from_edges(we)) * g)
        ref = lambda x, we: jnp.sum(reference(x, we, es) * g)
        for a, b in zip(jax.grad(loss, (0, 1))(x, we), jax.grad(ref, (0, 1))(x, we)):
            np.testing.assert_allclose(a, b, rtol=1e-6)
        with jax.enable_x64(True):
            check_grads(lambda w: s.forward(x.astype(jnp.float64), w),
                        (s.weight_from_edges(we.astype(jnp.float64)),), order=1, modes=['rev'])
        w = s.weight_from_edges(we)
        np.testing.assert_allclose(s.forward_etrace(x, w), s.forward(x, w), rtol=1e-6)


class TestStructureBinding:
    def test_bound_replaces_and_restores(self):
        es = edges_with_multapses()
        s = S.CSR(es)
        orig = s.indices
        with s.bound({'indices': orig + 0}) as t:
            assert t is s and s.indices is not orig
        assert s.indices is orig
        with pytest.raises(KeyError, match='structure arrays'):
            with s.bound({'indptr': s.indptr}):
                pass  # pragma: no cover
        d = S.Dense(es)
        assert set(d.arrays()) == {'mask'}
        full = S.Dense(EdgeSet.create([0, 1], [0, 0], np.arange(2), np.arange(1)))
        assert full.arrays() == {}

    @pytest.mark.parametrize('cls', STORAGES)
    def test_jit_with_structure_as_argument(self, cls):
        es = edges_with_multapses()
        s = cls(es)
        w = s.weight_from_edges(W_E)

        @jax.jit
        def f(x, w, arrays):
            with s.bound(arrays):
                return s.forward(x, w)

        np.testing.assert_allclose(f(COUNTS, w, s.arrays()), reference(COUNTS, W_E, es), rtol=1e-6)
        if cls is S.CSR:  # the braintrace template stays concrete
            with s.bound({'indices': jnp.zeros_like(s.indices)}):
                assert s.template(jnp.float32).indices is s._indices0

    def test_host_index_dtype(self):
        assert S._host_index(np.arange(3, dtype=np.int64)).dtype == np.int32
        assert S._host_index(np.array([2 ** 31], np.int64)).dtype == np.int64
        assert S._host_index(np.zeros((0,), np.int64)).dtype == np.int32


def _spike_edges(n_pre=40, n_post=30, k=6, seed=0):
    """Random edges without multapses (binary path), pre-major."""
    r = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n_pre), k)
    post = np.concatenate([r.choice(n_post, k, replace=False) for _ in range(n_pre)])
    return EdgeSet.create(pre, post, np.arange(n_pre), np.arange(100, 100 + n_post)), r


class TestFloat64BinaryPayload:
    """brainevent#204 / #199: float64 payloads on the binary path (K1a-10)."""

    def test_events_dtype(self):
        with jax.enable_x64(True):
            assert S._events(jnp.ones(3, jnp.float64)).dtype == jnp.float32
            for dt in (jnp.float32, jnp.bool_, jnp.float16):
                x = jnp.ones(3, dt)
                assert S._events(x) is x

    @pytest.mark.parametrize('cls', STORAGES)
    @pytest.mark.parametrize('batch', [None, 3])
    def test_binary_matches_float_path_and_keeps_w_dtype(self, cls, batch):
        with jax.enable_x64(True):
            es, r = _spike_edges()
            st = cls(es)
            w = st.weight_from_edges(jnp.asarray(r.normal(size=es.n_edges)))
            shape = (es.n_pre,) if batch is None else (batch, es.n_pre)
            x = jnp.asarray(r.random(shape) < 0.3, jnp.float64)
            yb = st.forward(x, w, 'binary')
            yf = st.forward(x, w, 'float')
            assert yb.dtype == jnp.float64 and yf.dtype == jnp.float64
            assert yb.shape == shape[:-1] + (es.n_post,)
            np.testing.assert_allclose(np.asarray(yb), np.asarray(yf), rtol=1e-13, atol=1e-13)

    def test_dense_forward_bit_identical_to_uncast_on_cpu(self):
        # the cast changes nothing where brainevent reads float64 correctly (CPU)
        with jax.enable_x64(True):
            r = np.random.default_rng(1)
            x = jnp.asarray(r.random((2, 50)) < 0.2, jnp.float64)
            W = jnp.asarray(r.normal(size=(50, 20)))
            with jax.default_device(jax.devices('cpu')[0]):
                np.testing.assert_array_equal(np.asarray(S._binary_dense(x, W)),
                                              np.asarray(brainevent_matmul(x, W)))

    @pytest.mark.parametrize('cls', STORAGES)
    def test_gradients_match_float_path(self, cls):
        with jax.enable_x64(True):
            es, r = _spike_edges(seed=2)
            st = cls(es)
            w = st.weight_from_edges(jnp.asarray(r.normal(size=es.n_edges)))
            x = jnp.asarray(r.random(es.n_pre) < 0.3, jnp.float64)
            c = jnp.asarray(r.normal(size=es.n_post))

            def loss(path):
                return lambda xx, ww: jnp.sum(st.forward(xx, ww, path) * c)

            gb = jax.grad(loss('binary'), argnums=(0, 1))(x, w)
            gf = jax.grad(loss('float'), argnums=(0, 1))(x, w)
            for a, b in zip(gb, gf):
                assert a.dtype == jnp.float64
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)

    def test_dense_forward_mode_and_check_grads(self):
        with jax.enable_x64(True):
            r = np.random.default_rng(3)
            x = jnp.asarray(r.random(12) < 0.4, jnp.float64)
            W = jnp.asarray(r.normal(size=(12, 7)))
            jf = jax.jacfwd(S._binary_dense, argnums=1)(x, W)
            jr = jax.jacrev(S._binary_dense, argnums=1)(x, W)
            np.testing.assert_allclose(np.asarray(jf), np.asarray(jr), atol=1e-14)
            # weights only: finite differences in x see the binarisation (zero), while the
            # payload cotangent is the straight-through ``ct @ W.T`` of the float path
            check_grads(lambda ww: S._binary_dense(x, ww), (W,), order=1, modes=('fwd', 'rev'))
            gx = jax.grad(lambda xx: jnp.sum(S._binary_dense(xx, W)))(x)
            np.testing.assert_allclose(np.asarray(gx), np.asarray(W.sum(1)), rtol=1e-14)


def brainevent_matmul(x, W):
    import brainevent
    return brainevent.BinaryArray(x) @ W
