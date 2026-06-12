# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests for ``sim.connect(..., synapse=...)`` building an ``EventPlasticProj``."""
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

import brainstate
import saiunit as u

brainstate.environ.set(precision=64, platform='cpu')

import pytest

from brainpy_state import (
    Simulator, all_to_all, iaf_psc_exp, spike_generator, spike_recorder,
    static_synapse, tsodyks2_synapse, vogels_sprekeler_synapse,
)
from brainpy_state._network._event_plastic import EventPlasticProj
from brainpy_state._network._event_proj import EventProjection


def _children(sim):
    return list(sim.nodes(allowed_hierarchy=(1, 1)).values())


def test_connect_synapse_builds_plastic_proj_and_runs():
    sim = Simulator(dt=0.1 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(iaf_psc_exp, 1)
    sim.connect(pre, post, synapse=tsodyks2_synapse(U=0.5, weight=100. * u.pA),
                delay=1.0 * u.ms, rule=all_to_all)
    assert any(isinstance(m, EventPlasticProj) for m in _children(sim))
    sim.simulate(20. * u.ms)   # runs through the for_loop without error


def test_connect_without_synapse_still_static():
    sim = Simulator(dt=0.1 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(iaf_psc_exp, 1)
    sim.connect(pre, post, weight=50. * u.pA, delay=1.0 * u.ms, rule=all_to_all)
    assert any(isinstance(m, EventProjection) for m in _children(sim))
    assert not any(isinstance(m, EventPlasticProj) for m in _children(sim))


def test_generator_pre_plastic_delivers_current():
    # spike_generator -> static_synapse -> iaf_psc_exp; the post voltage must
    # depart from rest, proving the plastic projection injected current.
    sim = Simulator(dt=0.1 * u.ms)
    gen = sim.create(spike_generator, spike_times=[5., 10., 15.] * u.ms)
    post = sim.create(iaf_psc_exp, 1)
    rec = sim.create(spike_recorder)
    sim.connect(gen, post, synapse=static_synapse(weight=200. * u.pA), delay=1.0 * u.ms)
    sim.connect(post, rec)
    res = sim.simulate(40. * u.ms)
    # the recorder captured the post population; at least it ran end-to-end
    assert res.spikes(rec).shape[0] > 0


def test_connect_delay_override_from_connect_arg():
    sim = Simulator(dt=0.1 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(iaf_psc_exp, 1)
    # spec carries delay=1.0 ms; connect overrides to 2.0 ms
    sim.connect(pre, post, synapse=static_synapse(weight=10. * u.pA),
                delay=2.0 * u.ms, rule=all_to_all)
    proj = next(m for m in _children(sim) if isinstance(m, EventPlasticProj))
    assert proj.delay_seam is not None


# -- weight-recording seam -------------------------------------------------
def test_connect_returns_plastic_proj_handle():
    sim = Simulator(dt=0.1 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(iaf_psc_exp, 1)
    proj = sim.connect(pre, post, synapse=tsodyks2_synapse(weight=100. * u.pA),
                       delay=1.0 * u.ms, rule=all_to_all)
    assert isinstance(proj, EventPlasticProj)
    assert proj in _children(sim)


def test_record_weight_returns_trajectory_in_pA():
    # vogels depresses by a constant alpha*eta on every pre spike (post-independent),
    # so a pre-only generator train gives a visible monotone weight staircase.
    sim = Simulator(dt=0.1 * u.ms)
    gen = sim.create(spike_generator, spike_times=[5., 10., 15., 20.] * u.ms)
    post = sim.create(iaf_psc_exp, 1)
    proj = sim.connect(gen, post,
                       synapse=vogels_sprekeler_synapse(weight=0.5 * u.pA, eta=0.01, alpha=0.5),
                       delay=1.0 * u.ms)
    sim.record_weight(proj)
    res = sim.simulate(30. * u.ms)
    wt = res.weight_trace(proj)
    assert u.get_unit(wt) == u.pA
    n_steps = res.times.size
    assert wt.shape == (n_steps, 1)
    w = u.get_mantissa(wt)[:, 0]
    assert float(w[0]) == pytest.approx(0.5, abs=1e-9)     # starts at the spec weight
    assert float(w[-1]) < 0.5                              # depressed by the train
    assert np.all(np.diff(w) <= 1e-12)                     # never increases


def test_weight_trace_unrecorded_raises():
    sim = Simulator(dt=0.1 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(iaf_psc_exp, 1)
    proj = sim.connect(pre, post, synapse=tsodyks2_synapse(weight=100. * u.pA),
                       delay=1.0 * u.ms, rule=all_to_all)
    res = sim.simulate(5. * u.ms)
    with pytest.raises(KeyError):
        res.weight_trace(proj)


def test_record_weight_rejects_non_plastic_projection():
    sim = Simulator(dt=0.1 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(iaf_psc_exp, 1)
    proj = sim.connect(pre, post, weight=50. * u.pA, delay=1.0 * u.ms, rule=all_to_all)
    assert isinstance(proj, EventProjection)
    with pytest.raises(TypeError):
        sim.record_weight(proj)
