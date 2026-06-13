# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for ``connect(receptor_type=k)`` specific-port routing + dual-mode deposit.

These are NEST-free unit tests of :class:`EventProjection`'s receptor branch. They
pin the two routing modes the multi-receptor demos depend on:

* **blob** — models that expose ``w_by_rec`` in their ``update`` signature
  (``iaf``/``aeif``/``gif_cond_exp_multisynapse``) receive one
  ``add_delta_input(delta_key, (N, n_receptors))`` deposit, later assembled by the
  Simulator bridge.
* **label-keyed** — models without ``w_by_rec`` (the 3 GLIF models, which self-pull
  via ``sum_delta_inputs(label='receptor_k')``) receive one
  ``add_delta_input(delta_key, col_k, label=f'receptor_{k}')`` deposit per port.

A 1-based ``receptor_type=k`` integer routes *all* edges to internal port index
``k - 1``; ``'uniform'`` keeps the existing random draw; out-of-range integers raise.
"""
import inspect
import unittest

import brainstate
import jax
import jax.numpy as jnp
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._network import all_to_all, one_to_one
from brainpy_state._network._event_proj import EventProjection


class _Box:
    def __init__(self, val):
        self.val = val


class _RecordingPost:
    """Minimal multi-receptor post that captures every ``add_delta_input`` call.

    Deliberately a plain object (not a ``brainstate`` Module) so that
    ``init_all_states(proj)`` does not recurse into it; the projection only needs
    ``varshape``, ``n_receptors`` and ``add_delta_input`` from its post.
    """

    def __init__(self, n, n_receptors):
        self.varshape = (n,)
        self.n_receptors = n_receptors
        self.deposits = []  # list of (key, label, value)

    def add_delta_input(self, key, value, label=None):
        self.deposits.append((key, label, value))


class _BlobPost(_RecordingPost):
    """w_by_rec-aware post -> blob deposit branch."""

    def update(self, x=0.0 * u.pA, w_by_rec=None):  # noqa: D401 - signature is the contract
        pass


class _KeyedPost(_RecordingPost):
    """No w_by_rec -> label-keyed deposit branch (GLIF-style self-pull)."""

    def update(self, x=0.0 * u.pA):  # noqa: D401 - signature is the contract
        pass


def _sum_mantissa(value):
    return float(jnp.sum(u.get_mantissa(value)))


class TestReceptorRouting(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _build(self, post, receptor_type):
        n = post.varshape[0]
        proj = EventProjection(
            pre_spike=lambda: box.val, n_pre_pop=n, pre_local_idx=jnp.arange(n),
            post=post, post_local_idx=jnp.arange(n), rule=all_to_all,
            weight=2.0 * u.nS, delay=None, receptor_type=receptor_type, seed=0,
        )
        box = _Box(jnp.ones(n))
        return proj, box

    def _run_once(self, proj):
        with brainstate.environ.context(t=0.0 * u.ms, i=0):
            proj.update()

    # ------------------------------------------------------------------ (a)+(d)
    def test_keyed_model_int_receptor_routes_to_internal_port_with_labels(self):
        post = _KeyedPost(n=2, n_receptors=4)
        proj, box = self._build(post, receptor_type=2)  # 1-based -> internal port idx 1
        self._run_once(proj)

        # One deposit per receptor port, each keyed by the projection's unique
        # delta key and tagged with label 'receptor_{k}' (matches GLIF's
        # sum_delta_inputs(label='receptor_k') filter).
        self.assertEqual(len(post.deposits), 4)
        keys = [k for (k, _lbl, _v) in post.deposits]
        labels = [lbl for (_k, lbl, _v) in post.deposits]
        self.assertEqual(set(keys), {proj._delta_key})
        self.assertEqual(labels, [f'receptor_{k}' for k in range(4)])

        # Only internal port 1 carries conductance; all other ports are zero.
        by_label = {lbl: v for (_k, lbl, v) in post.deposits}
        self.assertGreater(_sum_mantissa(by_label['receptor_1']), 0.0)
        for k in (0, 2, 3):
            self.assertEqual(_sum_mantissa(by_label[f'receptor_{k}']), 0.0)

    # ---------------------------------------------------------------------- (c)
    def test_blob_model_int_receptor_single_blob_deposit(self):
        post = _BlobPost(n=2, n_receptors=4)
        proj, box = self._build(post, receptor_type=2)
        self._run_once(proj)

        # Blob models receive exactly one (N, n_receptors) deposit, no label.
        self.assertEqual(len(post.deposits), 1)
        key, label, value = post.deposits[0]
        self.assertEqual(key, proj._delta_key)
        self.assertIsNone(label)
        self.assertEqual(tuple(value.shape), (2, 4))
        # Internal port 1 nonzero, the rest zero.
        col = u.get_mantissa(value)
        self.assertGreater(float(jnp.sum(col[:, 1])), 0.0)
        for k in (0, 2, 3):
            self.assertEqual(float(jnp.sum(col[:, k])), 0.0)

    # ---------------------------------------------------------------------- (b)
    def test_int_receptor_out_of_range_raises(self):
        for bad in (0, 5):  # valid 1-based range for 4 ports is [1, 4]
            with self.assertRaisesRegex(ValueError, 'range'):
                self._build(_KeyedPost(n=2, n_receptors=4), receptor_type=bad)

    # ----------------------------------------------------------- uniform regress
    def test_uniform_still_single_blob_deposit_on_blob_model(self):
        post = _BlobPost(n=3, n_receptors=4)
        proj, box = self._build(post, receptor_type='uniform')
        self._run_once(proj)
        self.assertEqual(len(post.deposits), 1)
        key, label, value = post.deposits[0]
        self.assertEqual(key, proj._delta_key)
        self.assertIsNone(label)
        self.assertEqual(tuple(value.shape), (3, 4))

    # ---------------------------------------------------------------------- (e)
    def test_receptor_type_none_is_ordinary_single_delta(self):
        # No receptor routing: ordinary one_to_one delta path, single unlabeled deposit.
        post = _KeyedPost(n=2, n_receptors=4)
        box = _Box(jnp.ones(2))
        proj = EventProjection(
            pre_spike=lambda: box.val, n_pre_pop=2, pre_local_idx=jnp.arange(2),
            post=post, post_local_idx=jnp.arange(2), rule=one_to_one,
            weight=2.0 * u.nS, delay=None, receptor_type=None,
        )
        with brainstate.environ.context(t=0.0 * u.ms, i=0):
            proj.update()
        self.assertEqual(len(post.deposits), 1)
        key, label, value = post.deposits[0]
        self.assertEqual(key, proj._delta_key)
        self.assertIsNone(label)
        self.assertEqual(tuple(value.shape), (2,))


if __name__ == '__main__':
    unittest.main()
