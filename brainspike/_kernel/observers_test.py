# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests of observer declarations, window accumulators and result containers."""
from __future__ import annotations

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.models import LIF, iaf_psc_alpha, spike_generator
from brainspike._kernel.network import Network
from brainspike._kernel.observers import (
    Observer,
    Recording,
    Results,
    interval_steps,
    window_init,
    window_update,
)

DT = 0.1 * u.ms


def test_interval_steps():
    assert interval_steps(None, DT) == 1
    assert interval_steps(0.5 * u.ms, DT) == 5
    assert interval_steps(0.001 * u.second, DT) == 10
    for bad in (0.15 * u.ms, 0.0 * u.ms, -1.0 * u.ms):
        with pytest.raises(ValueError, match='multiple'):
            interval_steps(bad, DT)
    with pytest.raises(Exception):
        interval_steps(1.0 * u.mV, DT)


@pytest.mark.parametrize('event', [True, False])
def test_window_update_matches_reshape(event):
    k, n_s, n = 4, 3, 2
    xs = jnp.arange(k * n_s * n, dtype=jnp.float32).reshape(k * n_s, n)

    def body(buf, i):
        return window_update(buf, i, k, xs[i], event), None

    buf, _ = jax.lax.scan(body, window_init(n_s, n, jnp.float32), jnp.arange(k * n_s))
    windows = np.asarray(xs).reshape(n_s, k, n)
    expected = windows.sum(1) if event else windows[:, -1]
    np.testing.assert_array_equal(np.asarray(buf), expected)


def test_observer_declaration():
    net = Network(dt=DT)
    a = net.add(LIF, 3, name='A')
    b = net.add(iaf_psc_alpha, 2, name='B')
    sg = net.add(spike_generator, 1, params=dict(spike_times=[1.0] * u.ms), name='sg')
    o = net.observe(a + b + sg, 'spike', interval=0.5 * u.ms)
    assert isinstance(o, Observer) and o.events == ('spike',) and o.interval == 5 and o.size == 6
    assert o.n_samples(20) == 4
    with pytest.raises(ValueError, match='multiple'):
        o.n_samples(12)
    v = net.observe(a, 'V', 'spike')
    assert v.events == ('spike',) and v.interval == 1
    with pytest.raises(ValueError, match='at least one'):
        Observer.create(net, a, (), None, 'x')
    with pytest.raises(ValueError, match='duplicate'):
        Observer.create(net, a, ('V', 'V'), None, 'x')
    with pytest.raises(TypeError):
        Observer.create(net, a, (1,), None, 'x')
    with pytest.raises(KeyError, match='no recordable'):
        Observer.create(net, a + b, ('V',), None, 'x')


def test_recording_and_results_containers():
    net = Network(dt=DT)
    a = net.add(LIF, 2, name='A')
    b = net.add(LIF, 1, name='B')
    o = net.observe(b + a, 'spike', 'V')
    x = np.zeros((2, 3, 3), np.float32)  # batch 2, 3 samples, 3 columns
    x[1, 2, 0] = 1
    x[0, 0, 2] = 2
    rec = Recording(observer=o, times=[0.1, 0.2, 0.3] * u.ms,
                    data={'spike': jnp.asarray(x), 'V': jnp.asarray(x) * u.mV},
                    gids=np.array([2, 0, 1]), columns={'B': np.array([0]), 'A': np.array([1, 2])},
                    batched=True)
    parts = rec.by_population('V')
    assert parts['A'].shape == (2, 3, 2) and parts['B'].shape == (2, 3, 1)
    t, gids, counts = rec.spike_events(trial=1)
    np.testing.assert_allclose(np.asarray(t.to_decimal(u.ms)), [0.3])
    assert gids.tolist() == [2] and counts.tolist() == [1.0]
    t, gids, counts = rec.spike_events(trial=0)
    assert gids.tolist() == [1] and counts.tolist() == [2.0]
    with pytest.raises(ValueError, match='not an event'):
        rec.spike_events('V', trial=0)
    res = Results(t_start=0.0 * u.ms, t_stop=0.3 * u.ms, n_steps=3, recordings={o.name: rec})
    assert res[o] is rec and res[o.name] is rec and o in res and len(res) == 1
    with pytest.raises(KeyError, match='available'):
        res['zzz']
