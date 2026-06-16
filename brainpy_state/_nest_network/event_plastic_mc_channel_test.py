# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for *plastic* ``receptor_type=k`` labeled single-channel routing.

The plastic projections (:class:`EventPlasticProj` and its voltage-coupled
subclass) must route into a named-channel post exactly like the static
:class:`EventProjection` does: ``receptor_type=k`` resolves — once, at
construction — to a delta-input channel label via the post's
``delta_label_for_receptor(rt) -> label``, and every per-step CSR deposit is
tagged with that label so the model reads it back with
``sum_delta_inputs(label=...)``. Without this seam the plastic weight is
delivered to an *unlabeled* key and the named-channel post silently drops it
(the cluster-21 blocker: the Urbanczik dendritic weight would never reach the
dendrite compartment).

These NEST-free tests pin that seam for both plastic primitives, mirroring
``_event_proj_mc_channel_test.py`` for the static path.
"""
import unittest

import brainstate
import jax
import jax.numpy as jnp
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state._nest_network.event_plastic import (
    EventPlasticProj, VoltageCoupledPlasticProj, _StaticTestRule)


class _State:
    """A minimal ``.value``-carrying stand-in for a post-neuron State."""

    def __init__(self, value):
        self.value = value


class _LabeledChannelPost:
    """Minimal mc-style named-channel post for the plastic path.

    Resolves ``receptor_type`` to a delta label (Urbanczik convention), has NO
    ``n_receptors``, optionally exposes a ``delta_Pi`` State (so the
    voltage-coupled reader has something to gather), and captures every
    ``add_delta_input`` call. A plain object (not a ``brainstate`` Module) so
    ``init_all_states`` does not recurse into it.
    """

    _LABELS = {1: 'soma_exc', 2: 'soma_inh', 3: 'dend_exc', 4: 'dend_inh'}

    def __init__(self, n, delta_pi=None):
        self.varshape = (n,)
        self.deposits = []  # list of (key, label, value)
        if delta_pi is not None:
            self.delta_Pi = _State(jnp.asarray(delta_pi))

    def delta_label_for_receptor(self, receptor_type):
        rt = int(receptor_type)
        if rt in self._LABELS:
            return self._LABELS[rt]
        raise ValueError(f'invalid spike receptor_type {rt}; valid 1-4')

    def add_delta_input(self, key, value, label=None):
        self.deposits.append((key, label, value))


class _DeltaPiReadRule(_StaticTestRule):
    """Voltage-coupled test rule: declares a per-edge δΠ post-state read."""
    post_state_reads = ('delta_Pi',)


def _sum_mantissa(value):
    return float(jnp.sum(u.get_mantissa(value)))


class TestEventPlasticLabeledRouting(unittest.TestCase):
    """Routing seam on the event-driven base projection."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _build(self, post, receptor_type, weight=2.0 * u.pA):
        n = post.varshape[0]
        proj = EventPlasticProj(
            pre_spike=lambda: jnp.ones(n), n_pre_pop=n, pre_local_idx=jnp.arange(n),
            post=post, post_local_idx=jnp.arange(n), n_post_pop=n,
            pre_idx=jnp.arange(n), post_idx=jnp.arange(n),
            rule=_StaticTestRule(weight=weight), receptor_type=receptor_type,
        )
        return proj

    def _run_once(self, proj):
        brainstate.nn.init_all_states(proj)
        with brainstate.environ.context(t=0.1 * u.ms, i=1):
            proj.update()

    def test_int_receptor_routes_to_single_named_channel(self):
        # receptor_type=3 -> a SINGLE labeled deposit into 'dend_exc'.
        post = _LabeledChannelPost(n=2)
        proj = self._build(post, receptor_type=3)
        self._run_once(proj)
        self.assertEqual(len(post.deposits), 1)
        key, label, value = post.deposits[0]
        self.assertEqual(key, proj._delta_key)
        self.assertEqual(label, 'dend_exc')
        self.assertEqual(tuple(value.shape), (2,))
        self.assertGreater(_sum_mantissa(value), 0.0)

    def test_each_receptor_maps_to_its_channel(self):
        expected = {1: 'soma_exc', 2: 'soma_inh', 3: 'dend_exc', 4: 'dend_inh'}
        for rt, lbl in expected.items():
            post = _LabeledChannelPost(n=1)
            proj = self._build(post, receptor_type=rt)
            self._run_once(proj)
            self.assertEqual(len(post.deposits), 1, f'rt={rt}')
            _k, label, _v = post.deposits[0]
            self.assertEqual(label, lbl, f'rt={rt}')

    def test_receptor_none_is_unlabeled_single_delta(self):
        # Regression: no receptor routing -> ordinary unlabeled deposit.
        post = _LabeledChannelPost(n=2)
        proj = self._build(post, receptor_type=None)
        self._run_once(proj)
        self.assertEqual(len(post.deposits), 1)
        _k, label, value = post.deposits[0]
        self.assertIsNone(label)
        self.assertEqual(tuple(value.shape), (2,))

    def test_invalid_receptor_type_raises_at_construction(self):
        with self.assertRaisesRegex(ValueError, 'receptor'):
            self._build(_LabeledChannelPost(n=1), receptor_type=9)


class TestVoltageCoupledPlasticLabeledRouting(unittest.TestCase):
    """Routing seam on the voltage-coupled projection (the Urbanczik path):
    the post-state read AND the labeled delivery must both work together."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _build(self, post, receptor_type, weight=2.0 * u.pA):
        n = post.varshape[0]
        proj = VoltageCoupledPlasticProj(
            pre_spike=lambda: jnp.ones(n), n_pre_pop=n, pre_local_idx=jnp.arange(n),
            post=post, post_local_idx=jnp.arange(n), n_post_pop=n,
            pre_idx=jnp.arange(n), post_idx=jnp.arange(n),
            rule=_DeltaPiReadRule(weight=weight), receptor_type=receptor_type,
        )
        return proj

    def _run_once(self, proj):
        brainstate.nn.init_all_states(proj)
        with brainstate.environ.context(t=0.1 * u.ms, i=1):
            proj.update()

    def test_dend_exc_routing_with_post_state_read(self):
        # The Urbanczik use case: dendritic plastic weight routes to 'dend_exc'
        # while the rule simultaneously reads δΠ per edge from the post.
        post = _LabeledChannelPost(n=2, delta_pi=jnp.array([0.5, -0.5]))
        proj = self._build(post, receptor_type=3)
        # the gather must see the post's δΠ State (no exception) ...
        gathered = proj._gather_post_states()
        self.assertIn('delta_Pi', gathered)
        self.assertEqual(tuple(gathered['delta_Pi'].shape), (2,))
        # ... and delivery must be labeled 'dend_exc'
        self._run_once(proj)
        self.assertEqual(len(post.deposits), 1)
        _k, label, value = post.deposits[0]
        self.assertEqual(label, 'dend_exc')
        self.assertEqual(tuple(value.shape), (2,))

    def test_receptor_none_is_unlabeled(self):
        post = _LabeledChannelPost(n=1, delta_pi=jnp.array([0.0]))
        proj = self._build(post, receptor_type=None)
        self._run_once(proj)
        _k, label, _v = post.deposits[0]
        self.assertIsNone(label)


if __name__ == '__main__':
    unittest.main()
