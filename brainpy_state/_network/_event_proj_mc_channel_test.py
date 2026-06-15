# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for ``connect(receptor_type=k)`` *labeled single-channel* routing.

A multi-compartment / named-channel post (``iaf_cond_alpha_mc``) is **not** a
stacked-``n_receptors`` model: each ``connect(device, post, receptor_type=k)``
feeds exactly ONE named delta-input channel (e.g. ``'w_ex_s'``), which the model
reads back with ``sum_delta_inputs(label='w_ex_s')``. Such a post exposes
``delta_label_for_receptor(rt) -> label`` and has NO ``n_receptors`` attribute.

These NEST-free tests pin :class:`EventProjection`'s labeled-channel branch:
``receptor_type=k`` resolves to a label at construction and the per-step deposit
is the ordinary plain-path contribution ``(n_post,)`` tagged with that label
(NOT a stacked ``(n_post, n_receptors)`` blob, and NOT the GLIF ``receptor_k``
self-pull). An unresolvable receptor type raises at construction.
"""
import unittest

import brainstate
import jax
import jax.numpy as jnp
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._network import all_to_all, one_to_one
from brainpy_state._network._event_proj import EventProjection


class _Box:
    def __init__(self, val):
        self.val = val


class _LabeledChannelPost:
    """Minimal mc-style named-channel post: resolves receptor_type -> a delta label,
    has NO ``n_receptors``, and captures every ``add_delta_input`` call.

    A plain object (not a ``brainstate`` Module) so ``init_all_states`` does not
    recurse; the projection only needs ``varshape``, ``delta_label_for_receptor``
    and ``add_delta_input``.
    """

    _LABELS = {1: 'w_ex_s', 2: 'w_in_s', 3: 'w_ex_p',
               4: 'w_in_p', 5: 'w_ex_d', 6: 'w_in_d'}

    def __init__(self, n):
        self.varshape = (n,)
        self.deposits = []  # list of (key, label, value)

    def delta_label_for_receptor(self, receptor_type):
        rt = int(receptor_type)
        if rt in self._LABELS:
            return self._LABELS[rt]
        raise ValueError(f'invalid spike receptor_type {rt}; valid 1-6')

    def update(self, x=0.0 * u.pA):  # signature is the contract (no w_by_rec)
        pass

    def add_delta_input(self, key, value, label=None):
        self.deposits.append((key, label, value))


def _sum_mantissa(value):
    return float(jnp.sum(u.get_mantissa(value)))


class TestLabeledChannelRouting(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _build(self, post, receptor_type, rule=all_to_all, post_idx=None, weight=2.0 * u.nS):
        n = post.varshape[0]
        box = _Box(jnp.ones(n))
        post_local_idx = jnp.arange(n) if post_idx is None else jnp.asarray(post_idx)
        proj = EventProjection(
            pre_spike=lambda: box.val, n_pre_pop=n, pre_local_idx=jnp.arange(n),
            post=post, post_local_idx=post_local_idx, rule=rule,
            weight=weight, delay=None, receptor_type=receptor_type, seed=0,
        )
        return proj, box

    def _run_once(self, proj):
        with brainstate.environ.context(t=0.0 * u.ms, i=0):
            proj.update()

    def test_int_receptor_routes_to_single_named_channel(self):
        # receptor_type=1 -> a SINGLE labeled deposit into 'w_ex_s' (not a stacked
        # blob, not a per-port GLIF self-pull): one deposit, plain (n_post,) shape.
        post = _LabeledChannelPost(n=2)
        proj, _box = self._build(post, receptor_type=1)
        self._run_once(proj)

        self.assertEqual(len(post.deposits), 1)
        key, label, value = post.deposits[0]
        self.assertEqual(key, proj._delta_key)
        self.assertEqual(label, 'w_ex_s')
        self.assertEqual(tuple(value.shape), (2,))   # plain path, not (2, n_receptors)
        self.assertGreater(_sum_mantissa(value), 0.0)

    def test_each_receptor_maps_to_its_channel(self):
        # All 6 spike receptors land on their distinct compartment+syntype channel.
        expected = {1: 'w_ex_s', 2: 'w_in_s', 3: 'w_ex_p',
                    4: 'w_in_p', 5: 'w_ex_d', 6: 'w_in_d'}
        for rt, lbl in expected.items():
            post = _LabeledChannelPost(n=1)
            proj, _box = self._build(post, receptor_type=rt)
            self._run_once(proj)
            self.assertEqual(len(post.deposits), 1, f"rt={rt}")
            _k, label, value = post.deposits[0]
            self.assertEqual(label, lbl, f"rt={rt}")
            self.assertEqual(tuple(value.shape), (1,), f"rt={rt}")

    def test_weight_unit_is_preserved_nS(self):
        # The conductance weight (nS) must survive the comm so the named channel,
        # which the model reads as nS, sums consistently.
        post = _LabeledChannelPost(n=1)
        proj, _box = self._build(post, receptor_type=1, rule=one_to_one,
                                 weight=5.0 * u.nS)
        self._run_once(proj)
        _k, _label, value = post.deposits[0]
        self.assertEqual(u.get_unit(value).dim, u.nS.dim)
        self.assertAlmostEqual(float(u.get_mantissa(value).reshape(-1)[0]), 5.0, places=9)

    def test_one_to_one_named_channel(self):
        # rule=one_to_one with a named channel: pre i -> post i, single labeled deposit.
        post = _LabeledChannelPost(n=3)
        proj, _box = self._build(post, receptor_type=4, rule=one_to_one)
        self._run_once(proj)
        self.assertEqual(len(post.deposits), 1)
        _k, label, value = post.deposits[0]
        self.assertEqual(label, 'w_in_p')
        self.assertEqual(tuple(value.shape), (3,))

    def test_partial_population_scatter_keeps_label(self):
        # Targeting a subset of post rows exercises the plain _scatter lift while
        # still tagging the deposit with the channel label.
        post = _LabeledChannelPost(n=3)
        proj, _box = self._build(post, receptor_type=5, post_idx=[0, 2])
        # 2 pre all-to-all onto rows [0, 2] of a 3-row post -> scatter runs.
        proj = EventProjection(
            pre_spike=lambda: jnp.ones(2), n_pre_pop=2, pre_local_idx=jnp.arange(2),
            post=post, post_local_idx=jnp.asarray([0, 2]), rule=all_to_all,
            weight=2.0 * u.nS, delay=None, receptor_type=5, seed=0,
        )
        self._run_once(proj)
        self.assertEqual(len(post.deposits), 1)
        _k, label, value = post.deposits[0]
        self.assertEqual(label, 'w_ex_d')
        col = u.get_mantissa(value)
        self.assertEqual(tuple(col.shape), (3,))
        self.assertGreater(float(col[0]), 0.0)
        self.assertGreater(float(col[2]), 0.0)
        self.assertEqual(float(col[1]), 0.0)   # untargeted row stays zero

    def test_invalid_receptor_type_raises_at_construction(self):
        # The post's resolver rejects an out-of-range receptor type; the projection
        # surfaces it eagerly at build time (not silently at the first step).
        with self.assertRaisesRegex(ValueError, 'receptor'):
            self._build(_LabeledChannelPost(n=1), receptor_type=9)

    def test_receptor_none_is_unlabeled_single_delta(self):
        # No receptor routing on a named-channel post: ordinary unlabeled deposit.
        post = _LabeledChannelPost(n=2)
        box = _Box(jnp.ones(2))
        proj = EventProjection(
            pre_spike=lambda: box.val, n_pre_pop=2, pre_local_idx=jnp.arange(2),
            post=post, post_local_idx=jnp.arange(2), rule=one_to_one,
            weight=2.0 * u.nS, delay=None, receptor_type=None,
        )
        self._run_once(proj)
        self.assertEqual(len(post.deposits), 1)
        _k, label, value = post.deposits[0]
        self.assertIsNone(label)
        self.assertEqual(tuple(value.shape), (2,))


if __name__ == '__main__':
    unittest.main()
