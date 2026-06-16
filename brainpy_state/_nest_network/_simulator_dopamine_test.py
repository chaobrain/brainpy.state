# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Simulator wiring for ``volume_transmitter`` + ``stdp_dopamine_synapse`` (cluster-08).

Covers the additive seams of spec §6: VT node registration (no holder), the
``connect(dopa_pool, vt)`` bind branch, the phase-0 update advancing ``n`` with the
substrate's one-step lag, VT independence, the ``signal_reads`` dispatch to
``VoltageCoupledPlasticProj``, the ``connect(..., vt=...)`` requirement, an
end-to-end dopamine-gated weight change, and that non-dopamine sims are untouched.
"""
import math

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64, platform='cpu')

import pytest

from brainpy_state import (
    Simulator, all_to_all, iaf_psc_exp, spike_generator, static_synapse,
    stdp_dopamine_synapse, tsodyks2_synapse, volume_transmitter,
)
from brainpy_state._nest_device.volume_transmitter import volume_transmitter as _vt_cls
from brainpy_state._nest_network._event_plastic import EventPlasticProj, VoltageCoupledPlasticProj


def _children(sim):
    return list(sim.nodes(allowed_hierarchy=(1, 1)).values())


def _advance(n, count, *, dt=1.0, tau_n=200.0):
    return n * math.exp(-dt / tau_n) + count / tau_n


# --------------------------------------------------------------------------
# node creation + dopa binding
# --------------------------------------------------------------------------
def test_create_volume_transmitter_registers_node_without_holder():
    sim = Simulator(dt=1.0 * u.ms)
    vt = sim.create(volume_transmitter, tau_n=200.0 * u.ms)
    mod = vt.segments[0].population
    assert isinstance(mod, _vt_cls)
    assert mod in sim._vt_nodes
    assert getattr(sim, f'_holder_{id(mod)}', None) is None    # VT emits no spikes


def test_connect_dopa_pool_to_vt_binds_source_and_returns_none():
    sim = Simulator(dt=1.0 * u.ms)
    dopa = sim.create(spike_generator, spike_times=[3., 6.] * u.ms)
    vt = sim.create(volume_transmitter, tau_n=200.0 * u.ms)
    out = sim.connect(dopa, vt)
    assert out is None
    assert len(vt.segments[0].population._dopa_sources) == 1


def test_phase0_advances_n_with_one_step_lag():
    # drive a dopa generator; the VT's n at step k must use the dopa holder value
    # captured at step k-1 (phase 0 runs before the generators in phase 2).
    sim = Simulator(dt=1.0 * u.ms)
    dopa = sim.create(spike_generator, spike_times=[2., 3., 6.] * u.ms)
    vt = sim.create(volume_transmitter, tau_n=200.0 * u.ms)
    sim.connect(dopa, vt)
    vt_mod = vt.segments[0].population
    brainstate.nn.init_all_states(sim)

    reader, idx = vt_mod._dopa_sources[0]
    n_prev, prev_count = 0.0, 0.0
    for k in range(10):
        with brainstate.environ.context(t=k * 1.0 * u.ms, i=k):
            sim.update(k * 1.0 * u.ms)
            n_now = float(np.asarray(vt_mod.n.value)[0])
            # n advanced from the PREVIOUS step's captured dopa count (one-step lag)
            assert np.isclose(n_now, _advance(n_prev, prev_count))
            n_prev = n_now
            prev_count = float(np.sum(np.asarray(reader())[np.asarray(idx)]))


def test_two_volume_transmitters_stay_independent():
    sim = Simulator(dt=1.0 * u.ms)
    da = sim.create(spike_generator, spike_times=[2.] * u.ms)
    db = sim.create(spike_generator, spike_times=[2., 4.] * u.ms)
    va = sim.create(volume_transmitter, tau_n=200.0 * u.ms)
    vb = sim.create(volume_transmitter, tau_n=200.0 * u.ms)
    sim.connect(da, va)
    sim.connect(db, vb)
    va_mod, vb_mod = va.segments[0].population, vb.segments[0].population
    brainstate.nn.init_all_states(sim)
    for k in range(8):
        with brainstate.environ.context(t=k * 1.0 * u.ms, i=k):
            sim.update(k * 1.0 * u.ms)
    # vb received strictly more dopa spikes -> larger concentration
    assert float(np.asarray(vb_mod.n.value)[0]) > float(np.asarray(va_mod.n.value)[0])


# --------------------------------------------------------------------------
# forward connect: signal_reads dispatch + vt requirement
# --------------------------------------------------------------------------
def test_connect_dopamine_routes_to_voltage_coupled():
    sim = Simulator(dt=1.0 * u.ms)
    pre = sim.create(spike_generator, spike_times=[5.] * u.ms)
    post = sim.create(iaf_psc_exp, 1)
    dopa = sim.create(spike_generator, spike_times=[5.] * u.ms)
    vt = sim.create(volume_transmitter, tau_n=200.0 * u.ms)
    sim.connect(dopa, vt)
    proj = sim.connect(pre, post, synapse=stdp_dopamine_synapse(weight=50. * u.pA),
                       delay=1.0 * u.ms, vt=vt)
    assert isinstance(proj, VoltageCoupledPlasticProj)
    assert isinstance(proj, EventPlasticProj)        # superset of primitive #1


def test_connect_dopamine_without_vt_raises():
    sim = Simulator(dt=1.0 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(iaf_psc_exp, 1)
    with pytest.raises(ValueError, match='volume_transmitter'):
        sim.connect(pre, post, synapse=stdp_dopamine_synapse(weight=50. * u.pA),
                    delay=1.0 * u.ms, rule=all_to_all)


def test_connect_dopamine_population_pre_routes_and_binds_signal():
    # population-pre branch also dispatches + binds the VT n signal source
    sim = Simulator(dt=1.0 * u.ms)
    pre = sim.create(iaf_psc_exp, 2)
    post = sim.create(iaf_psc_exp, 3)
    dopa = sim.create(spike_generator, spike_times=[5.] * u.ms)
    vt = sim.create(volume_transmitter, tau_n=200.0 * u.ms)
    sim.connect(dopa, vt)
    proj = sim.connect(pre, post, synapse=stdp_dopamine_synapse(weight=10. * u.pA),
                       delay=1.0 * u.ms, rule=all_to_all, vt=vt)
    assert isinstance(proj, VoltageCoupledPlasticProj)
    assert proj._signal_sources['n'][0] is vt.segments[0].population


# --------------------------------------------------------------------------
# end-to-end dopamine-gated weight change
# --------------------------------------------------------------------------
def test_dopamine_potentiates_preloaded_eligibility():
    # preload c>0; with b=0 the weight is flat until dopamine arrives, then the
    # c*(n-b) integral drives w UP (n>b). Isolates the broadcast modulation.
    sim = Simulator(dt=1.0 * u.ms)
    pre = sim.create(spike_generator, spike_times=[100.] * u.ms)   # silent during run
    post = sim.create(iaf_psc_exp, 1)
    dopa = sim.create(spike_generator, spike_times=[5.] * u.ms)
    vt = sim.create(volume_transmitter, tau_n=200.0 * u.ms)
    sim.connect(dopa, vt)
    proj = sim.connect(pre, post,
                       synapse=stdp_dopamine_synapse(weight=50. * u.pA, c=2.0, Wmax=200.0),
                       delay=1.0 * u.ms, vt=vt)
    sim.record_weight(proj)
    res = sim.simulate(40. * u.ms)
    w = u.get_mantissa(res.weight_trace(proj))[:, 0]
    assert float(w[0]) == pytest.approx(50.0, abs=1e-9)   # starts at spec weight
    assert np.all(w[:4] == pytest.approx(50.0, abs=1e-9))  # flat before any dopamine
    assert float(w[-1]) > 50.0                             # potentiated after dopamine
    assert np.all(np.isfinite(w))


def test_non_dopamine_sim_unaffected_by_phase0():
    # a normal plastic sim with no VT still runs (phase 0 is a no-op)
    sim = Simulator(dt=1.0 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(iaf_psc_exp, 1)
    sim.connect(pre, post, synapse=tsodyks2_synapse(weight=100. * u.pA),
                delay=1.0 * u.ms, rule=all_to_all)
    assert sim._vt_nodes == []
    sim.simulate(20. * u.ms)    # runs end-to-end without error
