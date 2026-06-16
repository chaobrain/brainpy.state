# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for ``EventProjection``'s ``as_current`` current-deposit seam.

These are NEST-free unit tests of the one substrate addition the 15d astrocyte
loop needs: a default-off ``as_current`` flag that makes an ``EventProjection``
deposit its ``(n_post,)`` contribution into the post's **current** input channel
(``add_current_input``) instead of the delta channel (``add_delta_input``).

The slow-inward current (SIC) the astrocyte emits is a *current* (pA) entering
``dV/dt``, not a delta/conductance — so the projection that delivers it must land
in ``sum_current_inputs(label='I_SIC')``, exactly where the neuron already reads
``I_stim``. Everything else (dense ``x @ W`` matmul, ``InputDelay``, segment
scatter) is unchanged; the graded value rides ``comm='dense'`` (``'sparse'``
binarises the presynaptic value and is rejected, as in clusters 22 / 15a).

Pinned here:

* ``as_current=True`` routes to ``add_current_input`` under the channel label,
  NOT ``add_delta_input`` (and the unlabelled variant lands in the current dict);
* the default (``as_current=False``) is unchanged — still a delta deposit;
* ``comm='sparse'`` + ``as_current`` raises;
* the deposit lowers under ``brainstate.transform.for_loop`` with a stable
  ``(n,)`` carry (the cluster-12 ``(N,)->()`` collapse guard).
"""
import unittest

import brainstate
import jax
import jax.numpy as jnp
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._base import Dynamics
from brainpy_state._nest_network import all_to_all, one_to_one
from brainpy_state._nest_network._event_proj import EventProjection


class _Box:
    def __init__(self, val):
        self.val = val


class _RecordingPost:
    """Plain post capturing every ``add_current_input`` / ``add_delta_input`` call.

    Deliberately not a ``brainstate`` Module so ``init_all_states(proj)`` does not
    recurse into it; the projection only needs ``varshape`` plus the two deposit
    methods. Each method records ``(key, label, value)`` into its own list so a
    test can assert *which* channel a deposit landed in.
    """

    def __init__(self, n):
        self.varshape = (n,)
        self.current_calls = []  # list of (key, label, value)
        self.delta_calls = []

    def add_current_input(self, key, value, label=None):
        self.current_calls.append((key, label, value))

    def add_delta_input(self, key, value, label=None):
        self.delta_calls.append((key, label, value))

    def update(self, x=0.0 * u.pA):  # signature is the contract (no w_by_rec)
        pass


class _CurrentSink(Dynamics):
    """Minimal real post: each step reads the labelled SIC current channel and
    accumulates it, so the ``as_current`` deposit lowers under ``for_loop`` with a
    real ``State`` carry (catches the cluster-12 ``(N,)->()`` shape collapse)."""

    def init_state(self, *args, **kwargs):
        self.acc = brainstate.ShortTermState(jnp.zeros(self.varshape))

    def update(self):
        i_sic = self.sum_current_inputs(jnp.zeros(self.acc.value.shape), label='I_SIC')
        self.acc.value = self.acc.value + i_sic
        return self.acc.value


def _sum_mantissa(value):
    return float(jnp.sum(u.get_mantissa(value)))


class TestAsCurrentDeposit(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _build(self, post, *, as_current, channel_label, comm='dense', n_pre=2):
        box = _Box(jnp.ones(n_pre))
        proj = EventProjection(
            pre_spike=lambda: box.val, n_pre_pop=n_pre, pre_local_idx=jnp.arange(n_pre),
            post=post, post_local_idx=jnp.arange(post.varshape[0]), rule=all_to_all,
            weight=2.0, delay=None, comm=comm,
            as_current=as_current, channel_label=channel_label, seed=0,
        )
        return proj, box

    def _run_once(self, proj):
        with brainstate.environ.context(t=0.0 * u.ms, i=0):
            proj.update()

    def test_as_current_routes_to_current_channel_under_label(self):
        # all_to_all, 2 pre (ones) -> 3 post, weight 2.0 -> x @ W = [4, 4, 4].
        post = _RecordingPost(n=3)
        proj, _box = self._build(post, as_current=True, channel_label='I_SIC')
        self._run_once(proj)

        # Exactly one current deposit, no delta deposit.
        self.assertEqual(len(post.current_calls), 1)
        self.assertEqual(len(post.delta_calls), 0)
        key, label, value = post.current_calls[0]
        self.assertEqual(key, proj._delta_key)
        self.assertEqual(label, 'I_SIC')
        self.assertEqual(tuple(value.shape), (3,))
        self.assertAlmostEqual(_sum_mantissa(value), 12.0, places=9)  # 3 * (2 * 1 + 2 * 1)

    def test_as_current_without_label_lands_in_current_dict(self):
        # No channel label: still a current deposit (unlabelled), never a delta.
        post = _RecordingPost(n=3)
        proj, _box = self._build(post, as_current=True, channel_label=None)
        self._run_once(proj)
        self.assertEqual(len(post.current_calls), 1)
        self.assertEqual(len(post.delta_calls), 0)
        _key, label, _value = post.current_calls[0]
        self.assertIsNone(label)

    def test_default_is_delta_deposit_unchanged(self):
        # as_current defaults False -> the existing seam: a delta deposit, no current.
        post = _RecordingPost(n=3)
        proj, _box = self._build(post, as_current=False, channel_label=None)
        self._run_once(proj)
        self.assertEqual(len(post.delta_calls), 1)
        self.assertEqual(len(post.current_calls), 0)

    def test_sparse_as_current_rejected(self):
        # Graded current needs the dense matmul; sparse binarises the pre value.
        post = _RecordingPost(n=3)
        with self.assertRaisesRegex(ValueError, "dense"):
            self._build(post, as_current=True, channel_label='I_SIC', comm='sparse')

    def test_as_current_update_lowers_under_for_loop(self):
        # one_to_one, graded pre [0.5, 1.0, 1.5], weight 2.0 -> contrib [1, 2, 3]
        # deposited each step; the sink accumulates it. Under for_loop the body
        # traces ONCE and the (3,) carry shape stays stable across 5 steps.
        n = 3
        pre_val = jnp.array([0.5, 1.0, 1.5])
        sink = _CurrentSink(n)
        box = _Box(pre_val)
        proj = EventProjection(
            pre_spike=lambda: box.val, n_pre_pop=n, pre_local_idx=jnp.arange(n),
            post=sink, post_local_idx=jnp.arange(n), rule=one_to_one,
            weight=2.0, delay=None, as_current=True, channel_label='I_SIC',
        )
        brainstate.nn.init_all_states(sink)
        brainstate.nn.init_all_states(proj)

        dt = 0.1 * u.ms

        def _run_step(k):
            with brainstate.environ.context(t=k * dt):
                proj.update()
                return sink.update()

        out = brainstate.transform.for_loop(_run_step, jnp.arange(5))
        self.assertEqual(tuple(out.shape), (5, 3))               # stable carry, one trace
        # acc after k+1 steps = (k+1) * [1, 2, 3]
        npt = jnp.asarray([[1., 2., 3.], [2., 4., 6.], [3., 6., 9.],
                           [4., 8., 12.], [5., 10., 15.]])
        self.assertTrue(bool(jnp.allclose(out, npt)), f'got {out}')


if __name__ == '__main__':
    unittest.main()
