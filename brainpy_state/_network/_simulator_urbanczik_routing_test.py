# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Simulator threading of ``receptor_type`` through the *plastic* connect path.

A plastic ``connect(pre, post, synapse=spec, receptor_type=k)`` into a
named-channel multi-compartment post (``pp_cond_exp_mc_urbanczik``) must forward
``receptor_type`` to the plastic projection so the per-step weight deposit is
tagged with the resolved compartment label. Before this seam the plastic
branches of ``_connect_pair`` dropped ``receptor_type`` and delivered to an
unlabeled key, which the named-channel post silently discards — the cluster-21
blocker (the dendritic Urbanczik weight would never reach the dendrite).

These NEST-free tests pin the wiring at *both* plastic call sites
(generator-pre and population-pre) and for *both* plastic primitives
(``EventPlasticProj`` and the voltage-coupled ``VoltageCoupledPlasticProj``).
"""
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import (Simulator, pp_cond_exp_mc_urbanczik, spike_generator,
                           parrot_neuron, static_synapse)
from brainpy_state._nest.pp_cond_exp_mc_urbanczik import SOMA_EXC, DEND_EXC
from brainpy_state._network._event_plastic import (
    EventPlasticProj, VoltageCoupledPlasticProj)

DT = 0.1


class _DeltaPiPlasticSpec:
    """Minimal voltage-coupled plastic spec reading the post's δΠ per edge.

    Stands in for the (Phase-2) Urbanczik spec so the Phase-0 routing wiring can
    be tested against the voltage-coupled branch: it declares the δΠ post-state
    read that promotes the projection to :class:`VoltageCoupledPlasticProj`.
    """
    is_homogeneous_weight = False
    stochastic = False
    pre_trace_tau = None
    post_trace_tau = None
    post_state_reads = ('delta_Pi',)
    weight_unit = u.pA

    def __init__(self, weight=80.0 * u.pA, delay=None):
        self.weight = weight
        self.delay = delay

    def edge_state_init(self):
        return {}

    def update(self, state, ctx):
        return state, state['weight']


class TestPlasticReceptorRoutingThreading(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def test_generator_pre_event_plastic_routes_to_named_channel(self):
        # spike_generator -> urbanczik via primitive #1 (EventPlasticProj),
        # receptor_type=DEND_EXC must resolve to the 'dend_exc' channel label.
        sim = Simulator(dt=DT * u.ms)
        post = sim.create(pp_cond_exp_mc_urbanczik, 1)
        sg = sim.create(spike_generator, spike_times=np.asarray([5.0]) * u.ms)
        proj = sim.connect(sg, post, synapse=static_synapse(weight=80.0 * u.pA),
                           receptor_type=DEND_EXC)
        self.assertIsInstance(proj, EventPlasticProj)
        self.assertEqual(proj._channel_label, 'dend_exc')

    def test_population_pre_event_plastic_routes_to_named_channel(self):
        # population-pre branch (parrot relay -> urbanczik): the other plastic
        # call site in _connect_pair must thread receptor_type too.
        sim = Simulator(dt=DT * u.ms)
        post = sim.create(pp_cond_exp_mc_urbanczik, 1)
        relay = sim.create(parrot_neuron, 1)
        sg = sim.create(spike_generator, spike_times=np.asarray([5.0]) * u.ms)
        sim.connect(sg, relay)
        proj = sim.connect(relay, post, synapse=static_synapse(weight=80.0 * u.pA),
                           receptor_type=DEND_EXC)
        self.assertIsInstance(proj, EventPlasticProj)
        self.assertEqual(proj._channel_label, 'dend_exc')

    def test_voltage_coupled_plastic_routes_to_named_channel(self):
        # The Urbanczik path: a δΠ-reading spec promotes to VoltageCoupledPlasticProj
        # and must still route the deposit to the resolved channel label.
        sim = Simulator(dt=DT * u.ms)
        post = sim.create(pp_cond_exp_mc_urbanczik, 1)
        sg = sim.create(spike_generator, spike_times=np.asarray([5.0]) * u.ms)
        proj = sim.connect(sg, post, synapse=_DeltaPiPlasticSpec(weight=80.0 * u.pA),
                           receptor_type=SOMA_EXC)
        self.assertIsInstance(proj, VoltageCoupledPlasticProj)
        self.assertEqual(proj._channel_label, 'soma_exc')

    def test_no_receptor_type_is_unlabeled(self):
        # Regression: omitting receptor_type keeps the historical unlabeled deposit.
        sim = Simulator(dt=DT * u.ms)
        post = sim.create(pp_cond_exp_mc_urbanczik, 1)
        sg = sim.create(spike_generator, spike_times=np.asarray([5.0]) * u.ms)
        proj = sim.connect(sg, post, synapse=static_synapse(weight=80.0 * u.pA))
        self.assertIsNone(proj._channel_label)


if __name__ == '__main__':
    unittest.main()
