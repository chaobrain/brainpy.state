# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate G4 (invariance) checks owned by T4: storage, batch size and RNG semantics."""
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel import rng
from brainspike._kernel.connectivity import rules as R
from brainspike._kernel.connectivity import storage as S

SEED, UID = 11, 3
PRE = np.concatenate([np.arange(0, 40), np.arange(70, 90)])  # concatenated view
POST = np.arange(30, 80)                                      # overlaps PRE partially

RULES = [
    R.all_to_all(allow_autapses=False),
    R.fixed_indegree(7, allow_autapses=False),
    R.fixed_indegree(7, allow_autapses=False, allow_multapses=False),
    R.fixed_total_number(300, allow_autapses=False),
    R.pairwise_bernoulli(0.15, allow_autapses=False),
]


def build(rule):
    return rule.build(PRE, POST, rng.object_key(SEED, UID))


@pytest.mark.parametrize('rule', RULES, ids=lambda r: type(r).__name__ + str(r.allow_multapses))
def test_edgeset_identical_for_dense_and_csr(rule):
    # the rule never sees the storage: two builds give the same edges, and both
    # storages reproduce them exactly (same per-target sums, same per-edge weights)
    a, b = build(rule), build(rule)
    assert a.same_edges(b)
    w = jnp.linspace(-1.0, 1.0, a.n_edges)
    dense, csr = S.Dense(a), S.CSR(b)
    counts = jax.random.randint(jax.random.PRNGKey(0), (a.n_pre,), 0, 4).astype(jnp.float32)
    yd = dense.forward(counts, dense.weight_from_edges(w))
    yc = csr.forward(counts, csr.weight_from_edges(w))
    assert jnp.allclose(yd, yc, atol=1e-5)
    assert jnp.array_equal(csr.weight_to_edges(csr.weight_from_edges(w)), w)
    # the dense mask is exactly the edge pattern
    mask = np.zeros((a.n_pre, a.n_post), bool)
    mask[np.asarray(a.pre_index), np.asarray(a.post_index)] = True
    assert mask.all() if dense.mask is None else np.array_equal(mask, np.asarray(dense.mask))
    assert a.n_edges == 0 or (np.asarray(a.pre) != np.asarray(a.post)).all()


def _run(storage, w, batch, mode, n_steps=12, rate=0.3):
    """Poisson-like sources driven by per-step population keys, through a storage.

    Returns the post input of every step and trial, shape ``(n_steps, batch, n_post)``.
    """
    pop_uid = 0

    def trial(key):
        return (jax.random.uniform(key, (storage.n_pre,)) < rate).astype(jnp.float32)

    def step(i):
        k = rng.trial_keys(rng.population_key(SEED, pop_uid, i), batch, mode)
        x = jax.vmap(trial)(k)
        return jax.vmap(lambda xx: storage.forward(xx, w, 'binary'))(x)

    return brainstate.transform.for_loop(step, jnp.arange(n_steps))


@pytest.mark.parametrize('cls', [S.Dense, S.CSR])
@pytest.mark.parametrize('mode', rng.RNG_MODES)
def test_batch1_vs_batch4_trial0(cls, mode):
    es = build(R.fixed_indegree(7, allow_autapses=False))
    s = cls(es)
    w = s.weight_from_edges(jnp.linspace(0.1, 1.0, es.n_edges))
    y1 = _run(s, w, 1, mode)
    y4 = _run(s, w, 4, mode)
    assert y1.shape[1] == 1 and y4.shape[1] == 4
    assert jnp.array_equal(y1[:, 0], y4[:, 0])
    if mode == 'common':
        assert all(jnp.array_equal(y4[:, b], y4[:, 0]) for b in range(4))
    else:
        assert not any(jnp.array_equal(y4[:, b], y4[:, 0]) for b in range(1, 4))


def test_trial0_identical_across_storages_and_modes():
    es = build(R.pairwise_bernoulli(0.2))
    outs = []
    for cls in (S.Dense, S.CSR):
        s = cls(es)
        w = s.weight_from_edges(jnp.ones(es.n_edges))
        for mode in rng.RNG_MODES:
            outs.append(_run(s, w, 3, mode)[:, 0])
    assert all(jnp.allclose(o, outs[0]) for o in outs[1:])

