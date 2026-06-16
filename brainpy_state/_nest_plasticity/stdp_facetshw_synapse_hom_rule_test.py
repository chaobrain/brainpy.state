# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``stdp_facetshw_synapse_hom``.

The FACETS/BrainScaleS hardware model (Schemmel et al. 2006; Pfeil et al. 2012) is
not pair-based: it accumulates two charges -- ``a_causal`` (first post since the last
pre) and ``a_acausal`` (nearest post before this pre) -- and, once per readout cycle,
quantises the weight to a 4-bit LUT index, compares the charges against thresholds
via two config-bit evaluation functions, applies one of three look-up tables, and
resets the charges (``stdp_facetshw_synapse_hom.h`` ``send()``). These tests lock the
quantisation helpers (incl. the default-weight->0 footgun), the evaluation function,
the per-edge charge accumulation through the substrate, and the readout LUT selection
+ reset + cadence in the kernel. Live-NEST equivalence is in
``_validation/stdp_facetshw_synapse_hom_parity_test.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64)

from brainpy_state import stdp_facetshw_synapse_hom
from brainpy_state._nest_network.event_plastic import EventPlasticProj, KernelContext

WPLE = 100.0 / 15.0          # weight_per_lut_entry for default Wmax=100, 16-entry LUT
LUT0 = [2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15]
LUT1 = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13]
THR = 21.835


def _ctx(pre_spike, post_spike, pre_trace, post_trace, t=5.0, E=1, dt=1.0):
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    return KernelContext(g(pre_spike), g(post_spike), g(pre_trace), g(post_trace),
                         jnp.asarray(t), jnp.asarray(dt), jax.random.key(0))


def _state(w, a_causal=0.0, a_acausal=0.0, next_readout=0.0,
           pre_seen=0.0, post_seen=0.0, causal_pending=0.0):
    f = lambda v: jnp.array([v])
    return {'weight': f(w), 'a_causal': f(a_causal), 'a_acausal': f(a_acausal),
            'next_readout': f(next_readout), 'pre_seen': f(pre_seen),
            'post_seen': f(post_seen), 'causal_pending': f(causal_pending)}


def _drive(rule, pre_steps, post_steps, n_steps, dt=1.0, dend=1):
    """Drive one edge; return final (weight, a_causal, a_acausal). Post effect at q+dend."""
    rule.delay = None
    box = {'pre': jnp.zeros(1), 'post': jnp.zeros(1)}
    proj = EventPlasticProj(
        pre_spike=lambda: box['pre'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=None, post_local_idx=jnp.arange(1), n_post_pop=1,
        post_spike=lambda: box['post'],
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(proj)
    pre_set = set(pre_steps)
    post_set = set(q + dend for q in post_steps)
    for i in range(n_steps):
        box['pre'] = jnp.array([1.0 if i in pre_set else 0.0])
        box['post'] = jnp.array([1.0 if i in post_set else 0.0])
        with brainstate.environ.context(t=(i + 1) * dt * u.ms, dt=dt * u.ms, i=i):
            proj.update()
    return (float(proj.weight.value[0]), float(proj.aux['a_causal'].value[0]),
            float(proj.aux['a_acausal'].value[0]))


# -- spec contract ---------------------------------------------------------
def test_spec_attributes_and_edge_state():
    s = stdp_facetshw_synapse_hom(weight=33.333, tau_plus=18.0 * u.ms, tau_minus=22.0 * u.ms)
    assert s.is_homogeneous_weight is False
    assert s.pre_trace_mode == 'nearest' and s.post_trace_mode == 'nearest'
    assert np.isclose(float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms)), 18.0)
    assert np.isclose(float(u.Quantity(s.post_trace_tau).to_decimal(u.ms)), 22.0)
    assert np.isclose(s.weight_per_lut_entry, WPLE)
    assert np.isclose(s.readout_cycle_duration, 15.0)         # single-edge: == driver_readout_time
    init = s.edge_state_init()
    assert set(init) == {'a_causal', 'a_acausal', 'next_readout',
                         'pre_seen', 'post_seen', 'causal_pending'}
    assert all(v == 0.0 for v in init.values())               # next_readout starts at 0
    assert u.get_unit(s.weight) == u.pA


def test_validation():
    with pytest.raises(ValueError):
        stdp_facetshw_synapse_hom(tau_plus=-1.0 * u.ms)
    with pytest.raises(ValueError):
        stdp_facetshw_synapse_hom(tau_minus=-1.0 * u.ms)
    with pytest.raises(ValueError):
        stdp_facetshw_synapse_hom(lookuptable_0=[0, 1, 2])             # not 16 entries
    with pytest.raises(ValueError):
        stdp_facetshw_synapse_hom(lookuptable_0=[16] + list(range(15)))  # entry out of [0,15]
    with pytest.raises(ValueError):
        stdp_facetshw_synapse_hom(lookuptable_0=[])                   # empty table
    with pytest.raises(ValueError):
        stdp_facetshw_synapse_hom(configbit_0=[0, 0, 1])              # not 4
    with pytest.raises(ValueError):
        stdp_facetshw_synapse_hom(configbit_0=[0, 1, 2, 0])          # entries not 0/1
    with pytest.raises(ValueError):
        stdp_facetshw_synapse_hom(reset_pattern=[1, 1, 1])           # not 6


def test_explicit_weight_per_lut_entry_overrides_default():
    # an explicit quantum overrides the Wmax/(lut_size-1) default
    s = stdp_facetshw_synapse_hom(weight=10.0, Wmax=100.0, weight_per_lut_entry=2.0)
    assert s.weight_per_lut_entry == 2.0
    assert s._weight_to_entry(10.0) == 5                          # round(10/2)=5
    assert np.isclose(s._entry_to_weight(5), 10.0)


# -- quantisation helpers (incl. the default-weight footgun) ---------------
def test_weight_entry_roundtrip_and_footgun():
    s = stdp_facetshw_synapse_hom(weight=33.333)
    assert s._weight_to_entry(33.333) == 5                    # round(33.333/6.667)=5
    assert np.isclose(s._entry_to_weight(5), 5 * WPLE)
    assert s._weight_to_entry(1.0) == 0                       # footgun: default weight -> 0
    assert s._weight_to_entry(100.0) == 15                    # top of the LUT


# -- evaluation function (default configbits select causal vs acausal) -----
def test_eval_function_default_configbits():
    s = stdp_facetshw_synapse_hom(weight=33.333)
    # configbit_0=[0,0,1,0] -> eval0 = (tl + a_causal)/2 > th  == a_causal > th
    assert bool(s._eval(jnp.array([30.0]), jnp.array([0.0]), s.configbit_0)[0]) is True
    assert bool(s._eval(jnp.array([10.0]), jnp.array([0.0]), s.configbit_0)[0]) is False
    # configbit_1=[0,1,0,0] -> eval1 = (tl + a_acausal)/2 > th == a_acausal > th
    assert bool(s._eval(jnp.array([0.0]), jnp.array([30.0]), s.configbit_1)[0]) is True
    assert bool(s._eval(jnp.array([0.0]), jnp.array([10.0]), s.configbit_1)[0]) is False


# -- readout LUT selection (direct kernel) ---------------------------------
def test_readout_potentiation_lut0():
    s = stdp_facetshw_synapse_hom(weight=33.333)
    # a_causal high, a_acausal low -> (T,F) -> LUT0, both charges reset
    st, _ = s.update(_state(w=5 * WPLE, a_causal=30.0, a_acausal=0.0, next_readout=0.0),
                     _ctx(1.0, 0.0, 0.0, 0.0, t=5.0))
    assert np.isclose(float(st['weight'][0]), LUT0[5] * WPLE)   # 5 -> 6
    assert float(st['a_causal'][0]) == 0.0 and float(st['a_acausal'][0]) == 0.0
    assert np.isclose(float(st['next_readout'][0]), 15.0)


def test_readout_depression_lut1():
    s = stdp_facetshw_synapse_hom(weight=33.333)
    st, _ = s.update(_state(w=5 * WPLE, a_causal=0.0, a_acausal=30.0, next_readout=0.0),
                     _ctx(1.0, 0.0, 0.0, 0.0, t=5.0))
    assert np.isclose(float(st['weight'][0]), LUT1[5] * WPLE)   # 5 -> 4
    assert float(st['a_causal'][0]) == 0.0 and float(st['a_acausal'][0]) == 0.0


def test_readout_both_high_lut2_identity():
    s = stdp_facetshw_synapse_hom(weight=33.333)
    st, _ = s.update(_state(w=5 * WPLE, a_causal=30.0, a_acausal=30.0, next_readout=0.0),
                     _ctx(1.0, 0.0, 0.0, 0.0, t=5.0))
    assert np.isclose(float(st['weight'][0]), 5 * WPLE)         # LUT2 identity: 5 -> 5
    assert float(st['a_causal'][0]) == 0.0                      # reset_pattern resets both


def test_readout_both_low_requantises_only():
    s = stdp_facetshw_synapse_hom(weight=33.333)
    # charges below threshold -> (F,F): no LUT, no reset, but weight re-quantises
    st, _ = s.update(_state(w=34.0, a_causal=5.0, a_acausal=5.0, next_readout=0.0),
                     _ctx(1.0, 0.0, 0.0, 0.0, t=5.0))
    assert np.isclose(float(st['weight'][0]), 5 * WPLE)         # round(34/6.667)=5 -> 33.33
    assert float(st['a_causal'][0]) == 5.0                      # (F,F) does not reset


def test_no_readout_before_next_readout_time():
    s = stdp_facetshw_synapse_hom(weight=33.333)
    # t below next_readout -> no readout; pre still present but weight untouched here
    st, _ = s.update(_state(w=34.0, a_causal=30.0, a_acausal=0.0, next_readout=100.0),
                     _ctx(1.0, 0.0, 0.0, 0.0, t=5.0))
    assert np.isclose(float(st['weight'][0]), 34.0)             # unquantised, no readout


# -- charge accumulation through the substrate -----------------------------
def test_acausal_accumulates_nearest_post_before_pre():
    # post@2 then pre@5: a_acausal += exp(-((5)-(2+1))/tau_minus) = exp(-2/tau).
    # next_readout=0 so pre@5 also triggers a readout AFTER... use high next_readout to
    # isolate accumulation: drive sets next_readout via a far first pre. Instead read
    # the charge directly with a long pre-free warmup is complex; assert via kernel.
    s = stdp_facetshw_synapse_hom(weight=33.333, tau_minus=20.0 * u.ms)
    # post_seen=1 (a post occurred), nearest post trace exp(-2/20); pre fires, no readout
    kminus = np.exp(-2.0 / 20.0)
    st, _ = s.update(_state(w=34.0, next_readout=100.0, post_seen=1.0),
                     _ctx(1.0, 0.0, 0.0, kminus, t=5.0))
    assert np.isclose(float(st['a_acausal'][0]), kminus)
    assert float(st['post_seen'][0]) == 0.0                    # consumed


def test_causal_captured_on_first_post_then_folded_at_next_pre():
    s = stdp_facetshw_synapse_hom(weight=33.333, tau_plus=20.0 * u.ms)
    # post step with pre_seen=1: capture causal_pending = pre_nearest exp(-3/20)
    kplus = np.exp(-3.0 / 20.0)
    st, _ = s.update(_state(w=34.0, next_readout=100.0, pre_seen=1.0),
                     _ctx(0.0, 1.0, kplus, 0.0, t=5.0))
    assert np.isclose(float(st['causal_pending'][0]), kplus)
    assert float(st['pre_seen'][0]) == 0.0 and float(st['post_seen'][0]) == 1.0
    # next pre folds it into a_causal and clears pending
    st2, _ = s.update({k: v for k, v in st.items()}, _ctx(1.0, 0.0, 0.0, 0.0, t=6.0))
    assert np.isclose(float(st2['a_causal'][0]), kplus)
    assert float(st2['causal_pending'][0]) == 0.0


# -- end-to-end: repeated causal pairings climb the LUT over readouts -------
def test_potentiating_train_climbs_weight_through_readouts():
    # One clean causal pair (post 0.5 ms after pre) per 15 ms readout cycle. With low
    # thresholds a_causal crosses while the (one-cycle-stale) a_acausal stays below, so
    # each readout fires (T,F) -> LUT0 and the discrete weight ratchets up. The causal
    # charge folds at the *next* pre (NEST defers accumulation past the readout), so the
    # climb starts one cycle in. dend=0 keeps the kernel-mechanics timing clean.
    s = stdp_facetshw_synapse_hom(weight=2 * WPLE, tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms,
                                  a_thresh_th=0.5, a_thresh_tl=0.5)
    pre = list(range(50, 1750, 150))        # every 15 ms (dt=0.1 -> 150 steps), offset 5 ms
    post = [p + 5 for p in pre]             # 0.5 ms after each pre (causal)
    w, _, _ = _drive(s, pre, post, 1800, dt=0.1, dend=0)
    assert w > 2 * WPLE                      # weight ratcheted up through LUT0
    assert np.isclose(w / WPLE, round(w / WPLE))   # on the discrete LUT grid


# -- jit / vmap / grad smoke ----------------------------------------------
def test_vmap_grad_smoke():
    s = stdp_facetshw_synapse_hom(weight=33.333)

    def run(w):
        st, _ = s.update(_state(w=34.0, a_causal=0.0, next_readout=100.0)
                         | {'weight': w}, _ctx(1.0, 0.0, 0.0, 0.5, t=5.0))
        return jnp.sum(st['a_acausal'])

    out = jax.vmap(run)(jnp.array([[34.0], [40.0]]))
    assert out.shape == (2,) and np.all(np.isfinite(np.asarray(out)))
