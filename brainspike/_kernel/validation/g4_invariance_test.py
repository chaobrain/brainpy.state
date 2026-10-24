# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gate G4, end to end: the goal-01 connectivity repros through ``Network`` -> plan -> runner.

The unit-level invariance checks (Dense vs CSR edge sets, batch 1 vs 4, RNG modes) are
T4's ``connectivity/invariance_test.py`` and T5's runner tests (see
``validation/gates.py``). Here every repro goes through the compiler and a simulation,
and delivery is checked against the plan's realised edges, so a storage/edge-order
misalignment or a slice-local autapse test would show in the voltages:

* autapse exclusion on **global** ids for disjoint slices, overlapping slices and
  concatenated views, for every random rule, Dense and CSR (goal-01 A5);
* zero-weight edges stay in ``get_connections`` (goal-01 A6);
* ``EdgeSet.set(weight=..., edges=get_connections(...))`` stays aligned with the
  delivered weights (multapses included);
* ``fixed_total_number``: exact count on global ids, refuses ``allow_multapses=False``;
* ``spike_generator`` duplicates / multiplicities deliver ``count * w`` (goal-01 A4).

Exact equality is required everywhere except float sums of several weights
(``rtol = 1e-6``, summation order).
"""
from __future__ import annotations

import brainunit as u
import numpy as np
import pytest

from brainspike._kernel.connectivity import (
    all_to_all,
    fixed_indegree,
    fixed_total_number,
    one_to_one,
    pairwise_bernoulli,
)
from brainspike._kernel.models import LIF, spike_generator
from brainspike._kernel.network import Network
from brainspike._kernel.params import Uniform

DT = 0.1 * u.ms
N = 12
#: LIF that integrates delta inputs without leak over the test window and never fires
#: on them (weights <= 0.01 mV); hard reset to 0.
LIF_P = dict(tau=1e9 * u.ms, V_th=1.0 * u.mV, V_reset=0.0 * u.mV, V_rest=0.0 * u.mV)
W = Uniform(0.001 * u.mV, 0.01 * u.mV)

VIEWS = {
    'disjoint': (lambda p: p[0:5], lambda p: p[5:10]),
    'overlap': (lambda p: p[0:5], lambda p: p[2:7]),
    'concat': (lambda p: p[0:3] + p[7:10], lambda p: p[2:9]),
    'concat_both': (lambda p: p[8:12] + p[0:2], lambda p: p[1:4] + p[9:11]),
}
RULES = {
    'all_to_all': lambda: all_to_all(allow_autapses=False),
    'fixed_indegree': lambda: fixed_indegree(3, allow_autapses=False),
    'fixed_indegree_nomulti': lambda: fixed_indegree(3, allow_autapses=False, allow_multapses=False),
    'fixed_total_number': lambda: fixed_total_number(20, allow_autapses=False),
    'pairwise_bernoulli': lambda: pairwise_bernoulli(0.6, allow_autapses=False),
}


def _recurrent(view, rule, storage, seed=4):
    """``N`` LIF neurons; the pre view fires once at step 0 (initial V above threshold)."""
    pre_f, post_f = VIEWS[view]
    net = Network(dt=DT, seed=seed)
    probe = Network(dt=DT)
    P0 = probe.add(LIF, N)
    pre_gids = np.asarray(pre_f(P0).gids)
    V0 = np.zeros(N)
    V0[pre_gids] = 2.0
    P = net.add(LIF, N, params=LIF_P, init=dict(V=V0 * u.mV), spk_reset='hard', name='P')
    proj = net.connect(pre_f(P), post_f(P), RULES[rule](), port='delta', weight=W, delay=DT,
                       storage=storage, name='rec')
    rec = net.observe(P, 'V', 'spike')
    return net, proj, rec, pre_gids, np.asarray(post_f(P).gids)


@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('rule', list(RULES))
@pytest.mark.parametrize('view', list(VIEWS))
def test_autapses_on_global_ids_end_to_end(view, rule, storage):
    net, proj, rec, pre_gids, post_gids = _recurrent(view, rule, storage)
    plan = net.compile()
    es = plan.edges(proj)
    pre, post = np.asarray(es.pre), np.asarray(es.post)
    assert not np.any(pre == post)                                # global-id autapses excluded
    assert set(pre) <= set(pre_gids) and set(post) <= set(post_gids)
    overlap = len(set(pre_gids) & set(post_gids))
    if rule == 'all_to_all':
        assert es.n_edges == len(pre_gids) * len(post_gids) - overlap
    elif rule.startswith('fixed_indegree'):
        assert np.array_equal(np.bincount(post, minlength=N)[post_gids], np.full(len(post_gids), 3))
        if rule.endswith('nomulti'):
            assert len(set(zip(pre.tolist(), post.tolist()))) == es.n_edges
    elif rule == 'fixed_total_number':
        assert es.n_edges == 20
    # Dense and CSR realise the same edges (same seed)
    other = 'csr' if storage == 'dense' else 'dense'
    es2 = _recurrent(view, rule, other)[0].compile().edges('rec')
    assert es.same_edges(es2)
    # delivery: every pre fired at step 0; at step 1 each neuron holds the sum of its
    # incoming edge weights (a fired neuron was reset to 0 first)
    res = plan.runner().simulate(0.3 * u.ms)
    spk = np.asarray(res[rec]['spike'])
    np.testing.assert_array_equal(np.nonzero(spk[0])[0], np.sort(pre_gids))
    assert spk[1:].sum() == 0
    w = np.asarray(es.weight.to_decimal(u.mV))
    expect = np.zeros(N)
    np.add.at(expect, post, w)
    V1 = np.asarray(res[rec]['V'].to_decimal(u.mV))[1]
    np.testing.assert_allclose(V1, expect, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize('storage', ['dense', 'csr'])
def test_zero_weight_edges_retained(storage):
    net = Network(dt=DT)
    P = net.add(LIF, 3, name='P')
    proj = net.connect(P, P, all_to_all(allow_autapses=False), port='delta', weight=0.0 * u.mV,
                       storage=storage)
    plan = net.compile()
    es = plan.edges(proj)
    conns = es.get_connections()
    assert len(conns) == 6 and es.n_edges == 6
    assert np.all(np.asarray(conns.weight.to_decimal(u.mV)) == 0.0)
    assert plan.projection(proj).groups[0].storage.kind == storage
    run = plan.runner()
    run.simulate(1.0 * u.ms)
    assert len(run.edges(proj).get_connections()) == 6
    assert len(es.get_connections(source=[0])) == 2 and len(es.get_connections(target=[0])) == 2


def _driven(storage, weight, seed=8, n_pre=6, n_post=5):
    """``spike_generator`` unit ``i`` fires at ``(i + 1)`` ms; ``fixed_total_number`` with multapses."""
    net = Network(dt=DT, seed=seed)
    sg = net.add(spike_generator, n_pre,
                 params=dict(spike_times=[[float(i + 1)] for i in range(n_pre)] * u.ms), name='sg')
    post = net.add(LIF, n_post, params=LIF_P, name='post')
    proj = net.connect(sg, post, fixed_total_number(40), port='delta', weight=weight, delay=DT,
                       storage=storage, name='p')
    rec = net.observe(post, 'V')
    return net, proj, rec


def _delivered_matrix(res, rec, n_pre):
    V = np.asarray(res[rec]['V'].to_decimal(u.mV))
    # unit i (stamp 10 (i + 1)) is emitted at step 10 (i + 1) - 1 and arrives one step later
    steps = [10 * (i + 1) for i in range(n_pre)]
    return np.stack([V[s] - V[s - 1] for s in steps])


@pytest.mark.parametrize('storage', ['dense', 'csr'])
def test_set_weight_alignment_end_to_end(storage):
    net, proj, _ = _driven(storage, W)
    es = net.compile().edges(proj)
    assert es.n_edges == 40 and es.max_multiplicity > 1  # multapses present
    # new weights for the edges of sources {1, 4} only, aligned with get_connections order
    conns = es.get_connections(source=[1, 4])
    new = (np.arange(len(conns)) + 1) * 1e-3
    es2 = es.set(weight=new * u.mV, edges=conns)
    got = es2.get_connections(source=[1, 4])
    w_got = np.asarray(got.weight.to_decimal(u.mV))
    np.testing.assert_array_equal(w_got, new.astype(w_got.dtype))  # exactly, in the weight dtype
    np.testing.assert_array_equal(got.edge, conns.edge)
    untouched = np.setdiff1d(np.arange(es.n_edges), conns.edge)
    np.testing.assert_array_equal(np.asarray(es2.weight.to_decimal(u.mV))[untouched],
                                  np.asarray(es.weight.to_decimal(u.mV))[untouched])
    # rebuild with the per-edge weights (canonical edge order) and check delivery
    net2, proj2, rec2 = _driven(storage, es2.weight)
    plan2 = net2.compile()
    assert plan2.edges(proj2).same_edges(es2)
    w = np.asarray(es2.weight.to_decimal(u.mV))
    pre, post = np.asarray(es2.pre_index), np.asarray(es2.post_index)
    pair = np.zeros((6, 5))
    np.add.at(pair, (pre, post), w)
    if storage == 'csr':  # per-edge weights survive the storage round trip exactly
        np.testing.assert_allclose(np.asarray(plan2.edges(proj2).weight.to_decimal(u.mV)), w,
                                   rtol=1e-7)
    res = plan2.runner().simulate(8.0 * u.ms)
    np.testing.assert_allclose(_delivered_matrix(res, rec2, 6), pair, rtol=1e-5, atol=1e-8)


def test_fixed_total_number_exact_and_refuses_no_multapses():
    for storage in ('dense', 'csr'):
        net, proj, _ = _driven(storage, 0.001 * u.mV)
        es = net.compile().edges(proj)
        assert es.n_edges == 40
    with pytest.raises(Exception):
        fixed_total_number(10, allow_multapses=False)


@pytest.mark.parametrize('storage', ['dense', 'csr'])
def test_spike_generator_multiplicity_end_to_end(storage):
    net = Network(dt=DT)
    sg = net.add(spike_generator, 2, params=dict(
        spike_times=[[1.0, 1.0, 1.0, 2.0], [1.5, 2.0, 2.0, 2.0]] * u.ms,
        spike_multiplicities=[[1, 1, 1, 2], [1, 1, 1, 4]]), name='sg')
    post = net.add(LIF, 2, params=LIF_P, name='post')
    proj = net.connect(sg, post, one_to_one(), port='delta', weight=0.01 * u.mV, delay=DT,
                       storage=storage)
    rec = net.observe(post, 'V')
    plan = net.compile()
    assert plan.projection(proj).path == 'float'  # counts > 1: exact float payload path
    V = np.asarray(plan.runner().simulate(3.0 * u.ms)[rec]['V'].to_decimal(u.mV))
    jumps = np.round(np.diff(np.concatenate([np.zeros((1, 2)), V]), axis=0) / 0.01, 6)
    # unit 0: 3 spikes at 1.0 ms, 2 at 2.0 ms; unit 1: 1 at 1.5 ms, 1 + 1 + 4 at 2.0 ms
    np.testing.assert_array_equal(np.nonzero(jumps[:, 0])[0], [10, 20])
    np.testing.assert_array_equal(jumps[[10, 20], 0], [3, 2])
    np.testing.assert_array_equal(np.nonzero(jumps[:, 1])[0], [15, 20])
    np.testing.assert_array_equal(jumps[[15, 20], 1], [1, 6])
