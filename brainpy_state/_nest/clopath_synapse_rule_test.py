# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for the rebuilt ``clopath_synapse`` spec + pure rule kernel (NEST-free).

The kernel is fed a hand-built :class:`KernelContext` (with ``post_states`` for the
post-neuron voltage reads) so the LTD/LTP math, rectifier boundaries, ``Wmax``/``Wmin``
clamps, ``x_bar`` scaling and ``dt``-invariance are locked without a live network.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state._network._event_plastic import KernelContext  # noqa: E402
from brainpy_state._nest.clopath_synapse import clopath_synapse  # noqa: E402

# NEST defaults
A_LTP, A_LTD = 8.0e-5, 14.0e-5
THETA_PLUS, THETA_MINUS = -45.3, -70.6
TAU_X = 15.0


def _ctx(E=1, *, pre_spike=0.0, pre_trace=0.0, V=0.0, u_plus=0.0, u_minus=0.0, dt=0.1):
    f = lambda v: jnp.full((E,), float(v))
    return KernelContext(
        pre_spike=f(pre_spike), post_spike=jnp.zeros(E), pre_trace=f(pre_trace),
        post_trace=jnp.zeros(E), t_now=jnp.asarray(0.0), dt=jnp.asarray(float(dt)),
        key=jax.random.key(0),
        post_states={'V': f(V), 'u_bar_plus': f(u_plus), 'u_bar_minus': f(u_minus)})


# --------------------------------------------------------------------------
# spec attributes + validation
# --------------------------------------------------------------------------
def test_spec_attributes_and_defaults():
    s = clopath_synapse()
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.post_state_reads == ('u_bar_minus', 'u_bar_plus', 'V')
    assert s.post_trace_tau is None
    assert float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms)) == TAU_X
    assert s.edge_state_init() == {}
    assert float(u.get_mantissa(s.weight)) == 1.0
    assert u.get_unit(s.weight) == u.mV     # delta-model default (aeif_psc_delta_clopath)
    assert s.Wmax == 100.0 and s.Wmin == 0.0
    assert s.A_LTP == A_LTP and s.A_LTD == A_LTD


def test_weight_unit_defaults_to_mv_and_preserves_explicit():
    # bare weight -> mV (voltage-jump delta synapse, aeif_psc_delta_clopath)
    assert u.get_unit(clopath_synapse(weight=2.0).weight) == u.mV
    # explicit pA honored (current-based hh_psc_alpha_clopath)
    s = clopath_synapse(weight=3.0 * u.pA)
    assert u.get_unit(s.weight) == u.pA
    assert float(u.get_mantissa(s.weight)) == 3.0
    # explicit mV honored
    assert u.get_unit(clopath_synapse(weight=4.0 * u.mV).weight) == u.mV


def test_validation_tau_x_positive():
    with pytest.raises(ValueError, match='tau_x'):
        clopath_synapse(tau_x=0.0 * u.ms)
    with pytest.raises(ValueError, match='tau_x'):
        clopath_synapse(tau_x=-5.0 * u.ms)


def test_validation_weight_sign_consistency():
    # NEST asymmetric sign tests: sign(weight)==sign(Wmin) and ==sign(Wmax).
    with pytest.raises(ValueError, match='Wmax'):
        clopath_synapse(weight=1.0, Wmax=-5.0)
    with pytest.raises(ValueError, match='Wmin'):
        clopath_synapse(weight=-1.0, Wmin=0.0, Wmax=-5.0)
    # all-positive is fine
    clopath_synapse(weight=1.0, Wmin=0.0, Wmax=5.0)


def test_validation_nonfinite_param_rejected():
    # any non-finite amplitude/threshold/bound is rejected (NEST-style guard).
    with pytest.raises(ValueError, match='finite'):
        clopath_synapse(A_LTP=np.inf)
    with pytest.raises(ValueError, match='finite'):
        clopath_synapse(Wmax=np.nan)


def test_bare_thresholds_interpreted_as_mv():
    # theta_plus/theta_minus passed as bare numbers default to mV and drive the
    # kernel exactly like the equivalent Quantities (the _to_mv bare-number path).
    bare = clopath_synapse(weight=50.0, theta_plus=-45.3, theta_minus=-70.6)
    assert u.get_unit(bare.theta_plus) == u.mV
    assert u.get_unit(bare.theta_minus) == u.mV
    assert np.isclose(bare._theta_plus, THETA_PLUS)
    assert np.isclose(bare._theta_minus, THETA_MINUS)
    # same LTD as the Quantity-thresholds spec on an identical context
    quant = clopath_synapse(weight=50.0)
    ctx = _ctx(pre_spike=1.0, V=-80.0, u_minus=-60.0)
    wb = bare.update({'weight': jnp.array([50.0])}, ctx)[0]['weight']
    wq = quant.update({'weight': jnp.array([50.0])}, ctx)[0]['weight']
    assert np.allclose(np.asarray(wb), np.asarray(wq))


# --------------------------------------------------------------------------
# LTD on pre spike (post not depolarized -> pure depression)
# --------------------------------------------------------------------------
def test_pure_ltd_on_pre_spike():
    s = clopath_synapse(weight=50.0)
    # V well below theta_plus -> relu(V-theta_plus)=0 -> no LTP; u_minus above theta_minus
    ctx = _ctx(pre_spike=1.0, V=-80.0, u_plus=-30.0, u_minus=-60.0)
    new, w_eff = s.update({'weight': jnp.array([50.0])}, ctx)
    expected = 50.0 - A_LTD * (-60.0 - THETA_MINUS)        # 50 - 14e-5*10.6
    assert np.allclose(np.asarray(new['weight']), expected)
    assert np.allclose(np.asarray(w_eff), expected)


def test_no_pre_spike_no_ltd():
    s = clopath_synapse(weight=50.0)
    ctx = _ctx(pre_spike=0.0, V=-80.0, u_minus=-60.0)   # depolarization absent, no pre
    new, _ = s.update({'weight': jnp.array([50.0])}, ctx)
    assert np.allclose(np.asarray(new['weight']), 50.0)


def test_ltd_below_theta_minus_is_zero():
    s = clopath_synapse(weight=50.0)
    # u_minus below theta_minus -> relu = 0 -> no depression even on a pre spike
    ctx = _ctx(pre_spike=1.0, V=-80.0, u_minus=-80.0)
    new, _ = s.update({'weight': jnp.array([50.0])}, ctx)
    assert np.allclose(np.asarray(new['weight']), 50.0)


# --------------------------------------------------------------------------
# LTP on post activity (continuous accumulation, no pre spike)
# --------------------------------------------------------------------------
def test_ltp_accumulates_while_depolarized():
    s = clopath_synapse(weight=50.0)
    # x_bar = pre_trace/tau_x = 15/15 = 1; V above theta_plus; u_plus above theta_minus.
    # NEST's LTP dw carries a * resolution (dt) factor (write_LTP_history).
    ctx = _ctx(pre_spike=0.0, pre_trace=TAU_X, V=-40.0, u_plus=-60.0, dt=0.1)
    new, _ = s.update({'weight': jnp.array([50.0])}, ctx)
    x_bar = 1.0
    expected = 50.0 + A_LTP * x_bar * (-40.0 - THETA_PLUS) * (-60.0 - THETA_MINUS) * 0.1
    assert np.allclose(np.asarray(new['weight']), expected)


def test_ltp_zero_below_theta_plus():
    s = clopath_synapse(weight=50.0)
    # V below theta_plus -> relu(V-theta_plus)=0 -> no LTP
    ctx = _ctx(pre_spike=0.0, pre_trace=TAU_X, V=-50.0, u_plus=-60.0)
    new, _ = s.update({'weight': jnp.array([50.0])}, ctx)
    assert np.allclose(np.asarray(new['weight']), 50.0)


def test_ltp_scales_with_x_bar():
    s = clopath_synapse(weight=50.0)
    # x_bar = pre_trace/tau_x; double the pre_trace -> double the LTP increment
    base = s.update({'weight': jnp.array([50.0])},
                    _ctx(pre_trace=TAU_X, V=-40.0, u_plus=-60.0))[0]['weight']
    dbl = s.update({'weight': jnp.array([50.0])},
                   _ctx(pre_trace=2 * TAU_X, V=-40.0, u_plus=-60.0))[0]['weight']
    assert np.allclose(np.asarray(dbl - 50.0), 2.0 * np.asarray(base - 50.0))


# --------------------------------------------------------------------------
# clamps
# --------------------------------------------------------------------------
def test_wmax_clamp_on_ltp():
    s = clopath_synapse(weight=99.95, Wmax=100.0)
    # ltp = 8e-5*1*95.3*120.6*0.1 = 0.09195 -> 99.95+0.092 = 100.042 overshoots Wmax
    # -> clamp to 100. (With the dt factor the per-step increment is ~0.092.)
    ctx = _ctx(pre_trace=TAU_X, V=50.0, u_plus=50.0, dt=0.1)
    new, _ = s.update({'weight': jnp.array([99.95])}, ctx)
    assert np.allclose(np.asarray(new['weight']), 100.0)


def test_wmin_clamp_on_ltd():
    s = clopath_synapse(weight=0.001, Wmin=0.0)
    ctx = _ctx(pre_spike=1.0, V=-80.0, u_minus=0.0)   # large LTD -> clamp to 0
    new, _ = s.update({'weight': jnp.array([0.001])}, ctx)
    assert np.allclose(np.asarray(new['weight']), 0.0)


# --------------------------------------------------------------------------
# rectifier boundaries (V exactly at theta_plus, u_minus exactly at theta_minus)
# --------------------------------------------------------------------------
def test_theta_plus_boundary_no_ltp():
    s = clopath_synapse(weight=50.0)
    ctx = _ctx(pre_trace=TAU_X, V=THETA_PLUS, u_plus=-60.0)   # relu(0)=0
    new, _ = s.update({'weight': jnp.array([50.0])}, ctx)
    assert np.allclose(np.asarray(new['weight']), 50.0)


def test_theta_minus_boundary_no_ltd():
    s = clopath_synapse(weight=50.0)
    ctx = _ctx(pre_spike=1.0, V=-80.0, u_minus=THETA_MINUS)   # relu(0)=0
    new, _ = s.update({'weight': jnp.array([50.0])}, ctx)
    assert np.allclose(np.asarray(new['weight']), 50.0)


# --------------------------------------------------------------------------
# dt invariance of the LTP time-integral: NEST's LTP dw carries a * resolution
# factor, so accumulating over a fixed physical duration is dt-invariant.
# --------------------------------------------------------------------------
def test_dt_invariance_of_ltp_integral():
    s = clopath_synapse(weight=50.0)

    def accumulate(dt, nsteps):
        w = jnp.array([50.0])
        for _ in range(nsteps):
            w = s.update({'weight': w},
                         _ctx(pre_trace=TAU_X, V=-40.0, u_plus=-60.0, dt=dt))[0]['weight']
        return float(np.asarray(w)[0])

    w_fine = accumulate(0.1, 10)     # T = 1.0 ms
    w_coarse = accumulate(0.2, 5)    # T = 1.0 ms
    assert np.isclose(w_fine, w_coarse, atol=1e-12)
    assert w_fine > 50.0             # there *is* potentiation


# --------------------------------------------------------------------------
# jit / vmap / grad smoke
# --------------------------------------------------------------------------
def test_vmap_over_weight_batch():
    s = clopath_synapse()
    ctx = _ctx(E=4, pre_spike=1.0, V=-80.0, u_minus=-60.0)

    def run(w):
        return s.update({'weight': w}, ctx)[1]

    out = jax.vmap(run)(jnp.full((5, 4), 50.0))
    assert out.shape == (5, 4)


def test_grad_flows_through_weight():
    s = clopath_synapse()

    def loss(w):
        ctx = _ctx(pre_spike=1.0, V=-80.0, u_minus=-60.0)
        return jnp.sum(s.update({'weight': w}, ctx)[1])

    g = jax.grad(loss)(jnp.array([50.0]))
    assert np.all(np.isfinite(np.asarray(g)))
