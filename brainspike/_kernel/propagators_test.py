# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the JAX exact-integration propagators.

References: the NumPy float64 originals in ``_nest_base/utils.py`` and
``iaf_psc_alpha._precompute_propagators`` (tight, where they are accurate) and a
50-digit mpmath evaluation (near the singularity, where the originals lose
accuracy). Float32 tolerance: relative error <= 1e-6 (measured <= 1.5e-7).
"""
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.propagators import (
    alpha_propagator_p31_p32,
    exp_decay,
    iaf_psc_alpha_propagators,
    phi1,
    phi2,
    propagator_exp,
)
from brainspike._nest_base.utils import alpha_propagator_p31_p32 as np_alpha
from brainspike._nest_base.utils import propagator_exp as np_exp

F32_RTOL = 1e-6
TAU_M, C_M, H = 10.0, 250.0, 0.1

# well separated from tau_m: the NumPy originals are accurate here
REGULAR_TAU_SYN = [0.5, 2.0, 5.0, 9.0, 11.0, 20.0, 100.0]
# near / at the singularity tau_syn == tau_m
NEAR_TAU_SYN = [10.0 - 1e-3, 10.0 - 1e-5, 10.0 - 1e-7, 10.0 - 1e-10, 10.0, 10.0 + 1e-9, 10.0 + 1e-6]


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _mp_ref(ts, tm, c, h):
    mp = pytest.importorskip('mpmath')
    mp.mp.dps = 50
    ts, tm, c, h = (mp.mpf(float(v)) for v in (ts, tm, c, h))
    if ts == tm:
        return float(0.5 * h * h / c * mp.e ** (-h / tm)), float(h / c * mp.e ** (-h / tm))
    beta = ts * tm / (tm - ts)
    g = beta / c
    p32 = g * mp.e ** (-h / ts) * mp.expm1(h / beta)
    p31 = g * mp.e ** (-h / ts) * (beta * mp.expm1(h / beta) - h)
    return float(p31), float(p32)


# ---------------------------------------------------------------- phi functions

class TestPhi:
    def test_values_x64(self, x64):
        x = jnp.array([-30.0, -2.0, -0.6, -0.4, -1e-3, 0.0, 1e-8, 0.3, 0.5, 3.0])
        xs = np.asarray(x)
        with np.errstate(invalid='ignore', divide='ignore'):
            ref1 = np.where(xs == 0, 1.0, np.expm1(xs) / np.where(xs == 0, 1, xs))
        np.testing.assert_allclose(phi1(x), ref1, rtol=1e-14)
        # phi2 reference by mpmath (closed form cancels for small x)
        mp = pytest.importorskip('mpmath')
        mp.mp.dps = 40
        ref2 = [0.5 if v == 0 else float((mp.expm1(mp.mpf(v)) - v) / mp.mpf(v) ** 2) for v in xs]
        np.testing.assert_allclose(phi2(x), ref2, rtol=1e-14)

    @pytest.mark.parametrize('fn', [phi1, phi2])
    def test_grad_finite_at_zero_and_branch_cut(self, x64, fn):
        g = jax.vmap(jax.grad(fn))(jnp.array([0.0, 0.5 - 1e-12, 0.5, -0.5, 1e-9]))
        assert np.all(np.isfinite(g))
        # derivative continuous across the series/closed-form switch at |x| = 0.5
        np.testing.assert_allclose(g[1], g[2], rtol=1e-9)
        assert float(jax.grad(phi1)(0.0)) == pytest.approx(0.5)
        assert float(jax.grad(phi2)(0.0)) == pytest.approx(1 / 6)

    def test_float32_dtype(self):
        assert phi1(jnp.float32(0.1)).dtype == jnp.float32
        assert phi2(1).dtype == jnp.float32  # ints promote to float32


# ---------------------------------------------------------------- vs NumPy originals

class TestAgainstNumPy:
    @pytest.mark.parametrize('ts', REGULAR_TAU_SYN)
    def test_alpha_regular_float64(self, x64, ts):
        r31, r32 = np_alpha(ts, TAU_M, C_M, H)
        p31, p32 = alpha_propagator_p31_p32(np.float64(ts), TAU_M, C_M, H)
        assert p31.dtype == jnp.float64
        # NumPy P31 itself carries ~1e-13 cancellation error at ts = 9, 11
        np.testing.assert_allclose(p31, r31, rtol=1e-11)
        np.testing.assert_allclose(p32, r32, rtol=1e-13)

    @pytest.mark.parametrize('ts', REGULAR_TAU_SYN)
    def test_exp_regular_float64(self, x64, ts):
        np.testing.assert_allclose(propagator_exp(np.float64(ts), TAU_M, C_M, H),
                                   np_exp(ts, TAU_M, C_M, H), rtol=1e-13)

    def test_exact_singularity_matches_numpy_fallback(self, x64):
        r31, r32 = np_alpha(TAU_M, TAU_M, C_M, H)
        p31, p32 = alpha_propagator_p31_p32(np.float64(TAU_M), TAU_M, C_M, H)
        np.testing.assert_allclose(p31, r31, rtol=1e-15)
        np.testing.assert_allclose(p32, r32, rtol=1e-15)
        np.testing.assert_allclose(propagator_exp(np.float64(TAU_M), TAU_M, C_M, H),
                                   np_exp(TAU_M, TAU_M, C_M, H), rtol=1e-15)

    @pytest.mark.parametrize('ts', REGULAR_TAU_SYN + NEAR_TAU_SYN)
    def test_float32_vs_numpy(self, ts):
        r31, r32 = np_alpha(ts, TAU_M, C_M, H)
        f = np.float32
        p31, p32 = alpha_propagator_p31_p32(f(ts), f(TAU_M), f(C_M), f(H))
        assert p31.dtype == jnp.float32 and p32.dtype == jnp.float32
        # float32 inputs round ts itself (10 - 1e-7 == 10 in float32), so compare with
        # a tolerance that also covers the input rounding
        np.testing.assert_allclose(p31, r31, rtol=2e-6)
        np.testing.assert_allclose(p32, r32, rtol=2e-6)

    def test_heterogeneous_broadcast(self, x64):
        ts = np.array([2.0, 10.0, 10.0 + 1e-12, 30.0])
        tm = np.array([10.0, 10.0, 10.0, 15.0])
        r31, r32 = np_alpha(ts, tm, C_M, H)
        p31, p32 = alpha_propagator_p31_p32(ts, tm, C_M, H)
        assert p31.shape == (4,)
        np.testing.assert_allclose(p32, r32, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(p31)[[0, 1, 3]], r31[[0, 1, 3]], rtol=1e-11)


# ---------------------------------------------------------------- near-singular accuracy

class TestNearSingular:
    @pytest.mark.parametrize('ts', NEAR_TAU_SYN)
    def test_float64_vs_mpmath(self, x64, ts):
        r31, r32 = _mp_ref(ts, TAU_M, C_M, H)
        p31, p32 = alpha_propagator_p31_p32(np.float64(ts), TAU_M, C_M, H)
        np.testing.assert_allclose(p31, r31, rtol=1e-14)
        np.testing.assert_allclose(p32, r32, rtol=1e-14)
        np.testing.assert_allclose(propagator_exp(np.float64(ts), TAU_M, C_M, H), r32, rtol=1e-14)

    def test_numpy_original_loses_accuracy_but_we_do_not(self, x64):
        # documents why the near-singular reference is mpmath, not the NumPy port
        ts = 10.0 - 1e-5
        r31, _ = _mp_ref(ts, TAU_M, C_M, H)
        n31, _ = np_alpha(ts, TAU_M, C_M, H)
        p31, _ = alpha_propagator_p31_p32(np.float64(ts), TAU_M, C_M, H)
        assert abs(n31 - r31) / r31 > 1e-10
        assert abs(float(p31) - r31) / r31 < 1e-14

    @pytest.mark.parametrize('ts', NEAR_TAU_SYN + REGULAR_TAU_SYN)
    def test_float32_vs_mpmath(self, ts):
        f = np.float32
        ts32 = f(ts)
        r31, r32 = _mp_ref(float(ts32), TAU_M, C_M, H)  # reference at the rounded input
        p31, p32 = alpha_propagator_p31_p32(ts32, f(TAU_M), f(C_M), f(H))
        np.testing.assert_allclose(p31, r31, rtol=F32_RTOL)
        np.testing.assert_allclose(p32, r32, rtol=F32_RTOL)

    def test_tiny_tau_syn_no_overflow(self, x64):
        # exp(-h/tau_syn) underflows: NumPy original returns nan / a wrong fallback
        for dtype in (np.float32, np.float64):
            r31, r32 = _mp_ref(1e-4, TAU_M, C_M, H)
            p31, p32 = alpha_propagator_p31_p32(dtype(1e-4), dtype(TAU_M), dtype(C_M), dtype(H))
            np.testing.assert_allclose(p31, r31, rtol=F32_RTOL)
            np.testing.assert_allclose(p32, r32, rtol=F32_RTOL)
        n31, n32 = np_alpha(1e-4, TAU_M, C_M, H)
        assert np.isnan(n31) and n32 > 100 * r32


# ---------------------------------------------------------------- gradients

class TestGradients:
    @pytest.mark.parametrize('ts', [10.0, 10.0 + 1e-9, 9.0, 0.5])
    def test_grad_finite_and_matches_fd(self, x64, ts):
        def f(params):
            ts_, tm_, c_, h_ = params
            p31, p32 = alpha_propagator_p31_p32(ts_, tm_, c_, h_)
            return p31 * 1e3 + p32 * 1e2 + propagator_exp(ts_, tm_, c_, h_) * 10

        x = jnp.array([ts, TAU_M, C_M, H])
        g = jax.grad(f)(x)
        assert np.all(np.isfinite(g))
        eps = 1e-6 * np.maximum(np.abs(np.asarray(x)), 1.0)
        fd = [(f(x.at[i].add(eps[i])) - f(x.at[i].add(-eps[i]))) / (2 * eps[i]) for i in range(4)]
        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-14)

    def test_float32_grad_no_nan_at_singularity(self):
        f = np.float32
        g = jax.grad(lambda ts: sum(alpha_propagator_p31_p32(ts, f(TAU_M), f(C_M), f(H))))(f(TAU_M))
        assert np.isfinite(g)

    def test_jit_vmap_traced_dt(self):
        fn = jax.jit(jax.vmap(lambda h: alpha_propagator_p31_p32(2.0, 10.0, 250.0, h)[1]))
        hs = jnp.array([0.05, 0.1, 0.2], jnp.float32)
        np.testing.assert_allclose(fn(hs), [np_alpha(2.0, 10.0, 250.0, float(h))[1] for h in hs],
                                   rtol=F32_RTOL)


# ---------------------------------------------------------------- iaf_psc_alpha set

class _Holder:
    pass


def _reference_iaf_psc_alpha(tau_m, ts_ex, ts_in, c_m, h):
    """Re-run ``iaf_psc_alpha._precompute_propagators`` on a stand-in object."""
    import brainstate
    from brainspike._nest_neuron.iaf_psc_alpha import iaf_psc_alpha
    obj = _Holder()
    obj.tau_syn_ex = ts_ex * u.ms
    obj.tau_syn_in = ts_in * u.ms
    obj.tau_m = tau_m * u.ms
    obj.C_m = c_m * u.pF
    obj._alpha_propagator_p31_p32 = iaf_psc_alpha._alpha_propagator_p31_p32
    with brainstate.environ.context(dt=h * u.ms):
        iaf_psc_alpha._precompute_propagators(obj)
    return obj


class TestIafPscAlphaSet:
    @pytest.mark.parametrize('ts_ex,ts_in', [(2.0, 2.0), (0.5, 3.0), (10.0, 9.0)])
    def test_matches_precompute(self, x64, ts_ex, ts_in):
        ref = _reference_iaf_psc_alpha(TAU_M, ts_ex, ts_in, C_M, H)
        out = iaf_psc_alpha_propagators(*map(np.float64, (TAU_M, ts_ex, ts_in, C_M, H)))
        for k in ['P11_ex', 'P22_ex', 'P21_ex', 'P11_in', 'P22_in', 'P21_in', 'expm1_tau_m', 'P30',
                  'P32_ex', 'P32_in']:
            np.testing.assert_allclose(out[k], getattr(ref, '_' + k), rtol=1e-13, err_msg=k)
        for k in ['P31_ex', 'P31_in']:
            np.testing.assert_allclose(out[k], getattr(ref, '_' + k), rtol=1e-10, err_msg=k)
        np.testing.assert_allclose(out['psc_init_ex'], np.e / ts_ex, rtol=1e-15)
        np.testing.assert_allclose(out['psc_init_in'], np.e / ts_in, rtol=1e-15)

    def test_units_and_mantissas_agree(self):
        a = iaf_psc_alpha_propagators(10 * u.ms, 2 * u.ms, 3 * u.ms, 0.25 * u.nF, 100 * u.us)
        b = iaf_psc_alpha_propagators(10.0, 2.0, 3.0, 250.0, 0.1)
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-6, err_msg=k)
        assert a['P30'].dtype == jnp.float32

    def test_exp_decay(self):
        np.testing.assert_allclose(exp_decay(2.0 * u.ms, 0.1 * u.ms), np.exp(-0.05), rtol=1e-7)
