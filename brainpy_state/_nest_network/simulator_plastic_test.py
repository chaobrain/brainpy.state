# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests for ``sim.connect(..., synapse=...)`` building an ``EventPlasticProj``."""
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

import pytest

from brainpy_state import (
    Simulator, aeif_psc_delta_clopath, all_to_all, clopath_synapse, dc_generator,
    iaf_psc_exp, multimeter, quantal_stp_synapse, spike_generator, spike_recorder,
    static_synapse, tsodyks2_synapse, vogels_sprekeler_synapse,
)
from brainpy_state._nest_network.event_plastic import EventPlasticProj, VoltageCoupledPlasticProj
from brainpy_state._nest_network.event_proj import EventProjection


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


# -- voltage-coupled dispatch (primitive #2 / clopath_synapse) --------------
def test_connect_clopath_builds_voltage_coupled_proj():
    # a synapse spec declaring post_state_reads dispatches to VoltageCoupledPlasticProj
    sim = Simulator(dt=0.1 * u.ms)
    gen = sim.create(spike_generator, spike_times=[5., 10.] * u.ms)
    post = sim.create(aeif_psc_delta_clopath, 1)
    proj = sim.connect(gen, post, synapse=clopath_synapse(weight=50. * u.mV),
                       delay=1.0 * u.ms)
    assert isinstance(proj, VoltageCoupledPlasticProj)
    assert isinstance(proj, EventPlasticProj)          # superset of primitive #1


def test_connect_clopath_population_pre_builds_voltage_coupled_proj():
    # the population-pre branch of _connect_pair dispatches too
    sim = Simulator(dt=0.1 * u.ms)
    pre = sim.create(iaf_psc_exp, 2)
    post = sim.create(aeif_psc_delta_clopath, 3)
    proj = sim.connect(pre, post, synapse=clopath_synapse(weight=20. * u.mV),
                       delay=1.0 * u.ms, rule=all_to_all)
    assert isinstance(proj, VoltageCoupledPlasticProj)


def test_connect_non_voltage_synapse_is_plain_plastic():
    # a spec WITHOUT post_state_reads stays primitive #1 (no over-dispatch)
    sim = Simulator(dt=0.1 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(iaf_psc_exp, 1)
    proj = sim.connect(pre, post, synapse=tsodyks2_synapse(weight=100. * u.pA),
                       delay=1.0 * u.ms, rule=all_to_all)
    assert isinstance(proj, EventPlasticProj)
    assert not isinstance(proj, VoltageCoupledPlasticProj)


def test_clopath_connect_records_depressing_weight_trace():
    # end-to-end: spike_generator -> clopath_synapse -> aeif_psc_delta_clopath post,
    # held depolarized (>theta_minus) by a DC bias so LTD fires on every pre spike.
    # A_LTP=0 isolates depression -> the recorded weight is monotone non-increasing
    # and ends below the start, proving the post-V read reaches the kernel.
    sim = Simulator(dt=0.1 * u.ms)
    gen = sim.create(spike_generator, spike_times=[6., 9., 12., 15., 18., 21.] * u.ms)
    post = sim.create(aeif_psc_delta_clopath, 1)
    bias = sim.create(dc_generator, amplitude=800. * u.pA)
    sim.connect(bias, post)
    proj = sim.connect(gen, post,
                       synapse=clopath_synapse(weight=80. * u.mV, A_LTP=0.0),
                       delay=1.0 * u.ms)
    sim.record_weight(proj)
    res = sim.simulate(30. * u.ms)
    wt = res.weight_trace(proj)
    assert u.get_unit(wt) == u.mV                         # delta-model weight unit
    n_steps = res.times.size
    assert wt.shape == (n_steps, 1)
    w = u.get_mantissa(wt)[:, 0]
    assert float(w[0]) == pytest.approx(80.0, abs=1e-9)   # starts at the spec weight
    assert np.all(np.isfinite(w))
    assert np.all(np.diff(w) <= 1e-12)                    # LTD-only -> never increases
    assert float(w[-1]) < 80.0                            # net depression occurred


def test_clopath_connect_level_bare_weight_override_stays_mv():
    # a bare connect-level weight override preserves the clopath spec's mV unit
    # (not coerced to pA), so it still drives the delta-model post correctly.
    sim = Simulator(dt=0.1 * u.ms)
    pre = sim.create(iaf_psc_exp, 1)
    post = sim.create(aeif_psc_delta_clopath, 1)
    proj = sim.connect(pre, post, synapse=clopath_synapse(), weight=5.0,
                       delay=1.0 * u.ms, rule=all_to_all)
    assert proj.rule.weight_unit == u.mV
    assert float(u.get_mantissa(proj.rule.weight)) == 5.0


# -- stochastic per-run seed seam (quantal_stp) ----------------------------
# A stochastic plastic rule draws from the projection's ``rng`` State each step.
# ``simulate()`` re-inits every State through ``init_all_states`` before the run,
# so the per-run seed must be threaded into ``init_state`` -- otherwise the rng is
# reset to a hard-coded key and ``connect(seed=)`` is silently ignored, making
# every seed produce the identical realization (and seed-mean parity meaningless).
def _quantal_vm(seed):
    sim = Simulator(dt=0.1 * u.ms)
    post = sim.create(iaf_psc_exp, 1, V_th=1e4 * u.mV)
    gen = sim.create(spike_generator, spike_times=np.arange(20., 200., 5.) * u.ms)
    sim.connect(gen, post,
                synapse=quantal_stp_synapse(weight=60. * u.pA, n=30, U=0.5,
                                            tau_rec=150. * u.ms, tau_fac=0. * u.ms),
                delay=1.0 * u.ms, seed=seed)
    mm = sim.create(multimeter, record_from=['V_m'], interval=0.1 * u.ms)
    sim.connect(mm, post)
    res = sim.simulate(250. * u.ms)
    return np.asarray(u.get_mantissa(res.trace(mm, 'V_m') / u.mV)).reshape(-1)


def test_simulator_stochastic_seed_is_reproducible_and_distinct():
    # Regression: init_state hard-seeded jax.random.key(0), ignoring connect(seed=),
    # so every seed produced the identical V_m trace. The per-run seed must survive
    # simulate()'s init_all_states.
    a1 = _quantal_vm(1)
    a1_again = _quantal_vm(1)
    a2 = _quantal_vm(2)
    np.testing.assert_array_equal(a1, a1_again)     # a fixed seed reproduces exactly
    assert not np.allclose(a1, a2)                  # different seeds -> different draws


def test_plastic_proj_seed_threads_into_runtime_rng():
    # Unit-level pin of the seam: after init_all_states, the runtime ``rng`` key
    # reflects the constructor seed (and seed=None stays key(0) for back-compat,
    # so the low-level drivers that set proj.rng.value after init are unaffected).
    import jax.numpy as jnp
    brainstate.environ.set(dt=0.1 * u.ms)

    def _rng_keydata(seed):
        proj = EventPlasticProj(
            pre_spike=lambda: jnp.zeros(1), n_pre_pop=1, pre_local_idx=jnp.arange(1),
            post=iaf_psc_exp(1), post_local_idx=jnp.arange(1), n_post_pop=1,
            pre_idx=jnp.array([0]), post_idx=jnp.array([0]),
            rule=quantal_stp_synapse(weight=60. * u.pA, n=10),
            **({} if seed is None else {'seed': seed}))
        brainstate.nn.init_all_states(proj)
        return jax.random.key_data(proj.rng.value)

    assert jnp.array_equal(_rng_keydata(None), jax.random.key_data(jax.random.key(0)))
    assert jnp.array_equal(_rng_keydata(7), jax.random.key_data(jax.random.key(7)))
    assert not jnp.array_equal(_rng_keydata(7), jax.random.key_data(jax.random.key(0)))
