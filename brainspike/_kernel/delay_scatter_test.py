# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests of the single-op per-edge-delay delivery (K1a-4)."""
from __future__ import annotations

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.connectivity.edgeset import EdgeSet
from brainspike._kernel.delay_scatter import DelayScatter, segment_length


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _edges(n_pre=7, n_post=5, n_edges=60, L=6, seed=0, hub=False):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n_pre, n_edges)
    if hub:  # one source with most of the edges
        pre[: n_edges // 2] = 0
    post = rng.integers(0, n_post, n_edges)
    d = rng.integers(1, L + 1, n_edges)
    d[0] = L  # a delay equal to the ring length
    es = EdgeSet.create(pre, post, np.arange(n_pre), np.arange(100, 100 + n_post))
    return es, pre, post, d


def _reference(ring, x, w_e, n, pre, post, d):
    """Loop over edges: edge e adds x[pre_e] * w_e to slot (n + d_e) % L."""
    out = np.array(ring, dtype=np.float64)
    L = out.shape[0]
    for e in range(len(pre)):
        out[(n + d[e]) % L, post[e]] += float(x[pre[e]]) * float(w_e[e])
    return out


# (strategy, chunk, seg_len, capacity); capacity 2 < active segments (loop after the
# loop-free chunk), 64 >= all segments (loop-free only)
CONFIGS = [('events', 1, None, 0), ('events', 3, None, 0), ('events', 2, 2, 0),
           ('events', 2, None, 2), ('events', 3, 2, 3), ('events', 1, None, 64),
           ('dense', 1, None, 0)]


@pytest.mark.parametrize('strategy,chunk,seg,cap', CONFIGS)
@pytest.mark.parametrize('hub', [False, True])
def test_deposit_matches_reference(x64, strategy, chunk, seg, cap, hub):
    L = 6
    es, pre, post, d = _edges(L=L, hub=hub)
    rng = np.random.default_rng(1)
    st = DelayScatter(es, d, L, strategy=strategy, chunk=chunk, seg_len=seg, capacity=cap)
    w_e = rng.normal(size=len(pre))
    w = st.weight_from_edges(jnp.asarray(w_e))
    ring = jnp.asarray(rng.normal(size=(L, 5)))
    # counts > 1, zeros, a fractional payload
    x = jnp.asarray([2.0, 0.0, 1.0, 0.0, 1.0, 0.5, 3.0])
    f = jax.jit(st.deposit)
    for n in (0, 4, 5, 6, 13):
        np.testing.assert_allclose(np.asarray(f(ring, n, x, w)),
                                   _reference(ring, x, w_e, n, pre, post, d), atol=1e-12)
    # no events: the ring is unchanged
    np.testing.assert_array_equal(np.asarray(f(ring, 3, jnp.zeros(7), w)), np.asarray(ring))


def test_weights_round_trip_exact():
    es, pre, post, d = _edges()
    st = DelayScatter(es, d, 6)
    w_e = jnp.arange(len(pre), dtype=jnp.float32) * 0.37
    np.testing.assert_array_equal(np.asarray(st.weight_to_edges(st.weight_from_edges(w_e))),
                                  np.asarray(w_e))
    q = st.weight_to_edges(st.weight_from_edges(w_e * u.pA))
    assert isinstance(q, u.Quantity) and q.unit == u.pA
    # storage order: rows ascending, then ring offset within a row
    rows = np.asarray(pre)[st.order]
    assert np.all(np.diff(rows) >= 0)
    rel = np.asarray(st.rel)
    same_row = np.diff(rows) == 0
    assert np.all(np.diff(rel)[same_row] >= 0)


def _dense_fn(st, pre_sorted, n):
    L, n_post = st.length, st.n_post
    ix = (jnp.asarray(st.rel) + (n % L) * n_post) % (L * n_post)

    def f(ring, x, w):
        return ring.reshape(-1).at[ix].add(x[pre_sorted] * w).reshape(ring.shape)
    return f


@pytest.mark.parametrize('strategy,cap', [('events', 0), ('events', 1), ('dense', 0)])
def test_gradients_and_jvp_equal_the_dense_scatter(x64, strategy, cap):
    """The custom JVP is the exact derivative of the scatter, also for silent sources."""
    L = 6
    es, pre, post, d = _edges(L=L)
    st = DelayScatter(es, d, L, strategy=strategy, chunk=2, capacity=cap)
    rng = np.random.default_rng(2)
    w = st.weight_from_edges(jnp.asarray(rng.normal(size=len(pre))))
    ring = jnp.asarray(rng.normal(size=(L, 5)))
    x = jnp.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0])  # many silent sources
    ref = _dense_fn(st, jnp.asarray(np.asarray(pre)[st.order]), 5)
    ct = jnp.asarray(rng.normal(size=(L, 5)))
    g1 = jax.grad(lambda r, a, b: jnp.sum(st.deposit(r, 5, a, b) * ct), argnums=(0, 1, 2))(ring, x, w)
    g2 = jax.grad(lambda r, a, b: jnp.sum(ref(r, a, b) * ct), argnums=(0, 1, 2))(ring, x, w)
    for a, b in zip(g1, g2):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)
    assert np.all(np.asarray(g1[1]) != 0)  # silent sources get their payload gradient
    t = (0.3 * ring, jnp.ones(7), 2.0 * w)
    _, j1 = jax.jvp(lambda r, a, b: st.deposit(r, 5, a, b), (ring, x, w), t)
    _, j2 = jax.jvp(ref, (ring, x, w), t)
    np.testing.assert_allclose(np.asarray(j1), np.asarray(j2), atol=1e-12)


@pytest.mark.parametrize('strategy,cap', [('events', 0), ('events', 2), ('dense', 0)])
def test_vmap_over_trials_and_steps(x64, strategy, cap):
    # with cap=2 some trials overflow the loop-free chunk and some do not
    L = 6
    es, pre, post, d = _edges(L=L)
    st = DelayScatter(es, d, L, strategy=strategy, chunk=2, capacity=cap)
    rng = np.random.default_rng(3)
    w_e = rng.normal(size=len(pre))
    w = st.weight_from_edges(jnp.asarray(w_e))
    xb = jnp.asarray((rng.random((4, 7)) < 0.4) * rng.integers(1, 3, (4, 7)), jnp.float64)
    rb = jnp.asarray(rng.normal(size=(4, L, 5)))
    nb = jnp.asarray([0, 1, 5, 9])
    out = jax.vmap(lambda r, a, n: st.deposit(r, n, a, w))(rb, xb, nb)
    for b in range(4):
        np.testing.assert_allclose(np.asarray(out[b]),
                                   _reference(rb[b], xb[b], w_e, int(nb[b]), pre, post, d), atol=1e-12)
    # batched gradients
    g = jax.vmap(jax.grad(lambda a, r: jnp.sum(st.deposit(r, 2, a, w) ** 2)))(xb, rb)
    assert g.shape == (4, 7) and np.all(np.isfinite(np.asarray(g)))


def test_float32_matches_float64():
    L = 6
    es, pre, post, d = _edges(L=L)
    st = DelayScatter(es, d, L)
    rng = np.random.default_rng(4)
    w_e = rng.normal(size=len(pre)).astype(np.float32)
    x = jnp.asarray([1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0], jnp.float32)
    out = st.deposit(jnp.zeros((L, 5), jnp.float32), 7, x, st.weight_from_edges(jnp.asarray(w_e)))
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(out), _reference(np.zeros((L, 5)), x, w_e, 7, pre, post, d),
                               rtol=1e-6, atol=1e-6)


def test_segments_split_hubs():
    assert segment_length(np.array([0, 0])) == 1
    assert segment_length(np.array([3, 4, 5])) == 5
    assert segment_length(np.array([1, 1, 1, 100])) == 52
    es, pre, post, d = _edges(n_edges=60, hub=True)
    st = DelayScatter(es, d, 6)
    counts = np.bincount(pre, minlength=7)
    assert st.M < counts.max() and st.n_segments == int(np.sum(-(-counts // st.M)))
    starts, lens = np.asarray(st.seg_start), np.asarray(st.seg_len)
    covered = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lens)])
    np.testing.assert_array_equal(covered, np.arange(60))  # segments tile the edges in order
    assert 'segments=' in repr(st) and 'capacity=0' in repr(st)
    assert DelayScatter(es, d, 6, capacity=10 ** 6).capacity == st.n_segments  # clipped


def test_empty_and_single_edge():
    es = EdgeSet.create(np.zeros(0, int), np.zeros(0, int), np.arange(3), np.arange(2))
    st = DelayScatter(es, np.zeros(0, int), 4)
    ring = jnp.ones((4, 2))
    assert st.n_segments == 0
    np.testing.assert_array_equal(np.asarray(st.deposit(ring, 1, jnp.ones(3), jnp.zeros(0))), 1.0)
    es1 = EdgeSet.create([2], [1], np.arange(3), np.arange(2))
    st1 = DelayScatter(es1, [4], 4, chunk=3)
    out = st1.deposit(jnp.zeros((4, 2)), 2, jnp.array([0.0, 0.0, 1.0]), jnp.array([1.5]))
    assert float(out[(2 + 4) % 4, 1]) == 1.5 and float(out.sum()) == 1.5


def test_nbytes():
    es, pre, post, d = _edges()
    ev = DelayScatter(es, d, 6)
    dn = DelayScatter(es, d, 6, strategy='dense')
    assert ev.rows is None and dn.rows is not None
    assert dn.nbytes(4) == ev.nbytes(4) + 60 * 4
    assert ev.nbytes(8) - ev.nbytes(4) == 60 * 4


def test_validation_errors():
    es, pre, post, d = _edges()
    with pytest.raises(ValueError, match='strategy'):
        DelayScatter(es, d, 6, strategy='fused')
    with pytest.raises(ValueError, match='chunk'):
        DelayScatter(es, d, 6, chunk=0)
    with pytest.raises(ValueError, match='ring length'):
        DelayScatter(es, d, 0)
    with pytest.raises(ValueError, match='shape'):
        DelayScatter(es, d[:-1], 6)
    with pytest.raises(ValueError, match=r'\[1, 6\]'):
        DelayScatter(es, np.where(np.arange(60) == 3, 0, d), 6)
    with pytest.raises(ValueError, match=r'\[1, 5\]'):
        DelayScatter(es, d, 5)  # d[0] == 6 > L
    with pytest.raises(ValueError, match='capacity'):
        DelayScatter(es, d, 6, capacity=-1)
    with pytest.raises(ValueError, match='seg_len'):
        DelayScatter(es, d, 6, seg_len=0)
    st = DelayScatter(es, d, 6)
    w = jnp.zeros(60)
    with pytest.raises(ValueError, match='ring'):
        st.deposit(jnp.zeros((5, 5)), 0, jnp.zeros(7), w)
    with pytest.raises(ValueError, match='weights'):
        st.deposit(jnp.zeros((6, 5)), 0, jnp.zeros(7), jnp.zeros(59))
    with pytest.raises(ValueError, match='payload'):
        st.deposit(jnp.zeros((6, 5)), 0, jnp.zeros(6), w)
    with pytest.raises(ValueError, match='weights'):
        st.weight_to_edges(jnp.zeros(3))
    with pytest.raises(TypeError, match='deposit'):
        st.forward(jnp.zeros(7), w)
    with pytest.raises(TypeError, match='braintrace'):
        st.forward_etrace(jnp.zeros(7), w)
