# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free unit tests for the send-view weight_recorder helper."""
import numpy as np
import pytest

from brainpy_state._network._weight_recorder_view import (
    send_steps_from_pre,
    weight_recorder_events,
)


# -- weight_recorder_events: the thin send-view (post-update sampling) ---------

def test_samples_post_update_value_at_send_step():
    # weight_trace[s] is the POST-send weight; the event must read the trace AT s,
    # not s-1 (the cluster-08 set_weight ordering: NEST logs the post-send weight).
    trace = np.array([5.0, 5.0, 4.0, 4.0, 3.0])   # sends at steps 2 and 4 (depress)
    steps, w = weight_recorder_events(trace, np.array([2, 4]))
    assert list(steps) == [2, 4]
    assert list(w) == [4.0, 3.0]                  # post-update, NOT [5.0, 4.0]


def test_empty_train_zero_events():
    steps, w = weight_recorder_events(np.arange(5.0), np.array([], dtype=int))
    assert steps.size == 0 and w.size == 0


def test_single_send():
    steps, w = weight_recorder_events(np.array([5.0, 4.0, 4.0]), np.array([1]))
    assert list(steps) == [1] and list(w) == [4.0]


def test_change_after_last_send_is_invisible():
    # A weight change strictly after the last send is absent from the events
    # (NEST's recorder misses it too) but present in the final weight.
    trace = np.array([5.0, 4.0, 4.0, 9.9])        # last send at step 1; 9.9 is a tail change
    steps, w = weight_recorder_events(trace, np.array([1]))
    assert list(w) == [4.0]
    assert trace[-1] == 9.9


def test_multapse_per_edge_csr_order():
    # (T, E=2): two edges (one shared pre firing at steps 1, 3), independent weights.
    trace = np.array([[5., 7.], [4., 6.5], [4., 6.5], [3., 6.0]])
    ev = weight_recorder_events(trace, [np.array([1, 3]), np.array([1, 3])])
    assert list(ev[0][1]) == [4.0, 3.0]
    assert list(ev[1][1]) == [6.5, 6.0]


def test_shared_steps_on_2d_trace_gives_n_by_E():
    trace = np.array([[5., 7.], [4., 6.5], [3., 6.0]])
    steps, w = weight_recorder_events(trace, np.array([1, 2]))
    assert w.shape == (2, 2)
    assert list(w[:, 0]) == [4.0, 3.0]
    assert list(w[:, 1]) == [6.5, 6.0]


def test_per_edge_list_requires_2d_trace():
    with pytest.raises(ValueError):
        weight_recorder_events(np.arange(5.0), [np.array([1]), np.array([2])])


def test_per_edge_list_length_must_match_columns():
    trace = np.zeros((4, 2))
    with pytest.raises(ValueError):
        weight_recorder_events(trace, [np.array([1])])   # 1 edge list, 2 columns


# -- send_steps_from_pre: derive the send mask from a pre spike train ----------

def test_single_pre_train_to_steps():
    pre = np.zeros(10)
    pre[[2, 5, 9]] = 1
    assert list(send_steps_from_pre(pre)) == [2, 5, 9]


def test_lag_shifts_and_clips():
    pre = np.zeros(5)
    pre[[1, 4]] = 1
    # lag=1: 1->2 kept, 4->5 clipped out of [0, 5)
    assert list(send_steps_from_pre(pre, lag=1)) == [2]


def test_multi_pre_gathers_per_edge_csr():
    pre = np.zeros((6, 2))
    pre[[1, 4], 0] = 1
    pre[[2], 1] = 1
    out = send_steps_from_pre(pre, pre_of_edge=np.array([0, 1, 0]))   # 3 edges, CSR
    assert [list(x) for x in out] == [[1, 4], [2], [1, 4]]


def test_single_column_2d_train_shared():
    pre = np.zeros((6, 1))
    pre[[1, 3]] = 1
    assert list(send_steps_from_pre(pre)) == [1, 3]


def test_multi_pre_requires_mapping():
    with pytest.raises(ValueError):
        send_steps_from_pre(np.zeros((4, 3)))


# -- composition + dt invariance + jit/vmap/grad smoke ------------------------

def test_compose_pre_then_events():
    pre = np.zeros(6)
    pre[[1, 3]] = 1
    trace = np.array([5., 4., 4., 3., 3., 3.])
    steps, w = weight_recorder_events(trace, send_steps_from_pre(pre))
    assert list(steps) == [1, 3]
    assert list(w) == [4.0, 3.0]


def test_dt_invariance_of_event_times():
    # Same protocol on two dt grids: the event TIME (ms) is invariant, steps scale.
    for dt in (0.05, 0.1):
        T = int(round(20.0 / dt))
        pre = np.zeros(T)
        pre[int(round(10.0 / dt))] = 1
        steps = send_steps_from_pre(pre)
        assert np.isclose(steps[0] * dt, 10.0)


def test_masked_view_under_vmap():
    import jax
    import jax.numpy as jnp
    sel = jnp.array([1, 3])
    g = jax.vmap(lambda tr: tr[sel])(jnp.arange(12.).reshape(2, 6))
    assert g.shape == (2, 2)
    assert list(np.asarray(g[0])) == [1.0, 3.0]


def test_masked_view_under_grad():
    import jax
    import jax.numpy as jnp
    sel = jnp.array([1, 3])
    f = lambda tr: jnp.asarray(tr)[sel].sum()        # masking is pure indexing
    grad = np.asarray(jax.grad(f)(jnp.arange(6.)))
    assert list(grad) == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
