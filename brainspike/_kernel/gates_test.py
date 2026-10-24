# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the Mg-block gate (K0 T8): analytic Jahr-Stevens curve with units."""
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.gates import MgBlock
from brainspike._kernel.ports import Conductance
from brainspike._kernel.units import UnitMismatchError


def _jahr_stevens(v_mV, mg_mM=1.2):
    # Jahr & Stevens (1990): B(V) = 1 / (1 + [Mg]/3.57 mM * exp(-0.062 V/mV))
    return 1.0 / (1.0 + mg_mM / 3.57 * np.exp(-0.062 * v_mV))


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


class TestMgBlockCurve:
    def test_matches_analytic_formula_in_mV(self, x64):
        v = np.linspace(-100.0, 40.0, 141)
        got = np.asarray(MgBlock()(jnp.asarray(v) * u.mV))
        np.testing.assert_allclose(got, _jahr_stevens(v), rtol=1e-13)

    def test_unit_of_V_does_not_matter(self, x64):
        v = np.linspace(-100.0, 40.0, 29)
        in_mV = np.asarray(MgBlock()(jnp.asarray(v) * u.mV))
        in_V = np.asarray(MgBlock()(jnp.asarray(v / 1000.0) * u.volt))
        np.testing.assert_allclose(in_V, in_mV, rtol=1e-12)

    def test_constants_in_other_units(self, x64):
        v = jnp.linspace(-90.0, 10.0, 11) * u.mV
        ref = np.asarray(MgBlock()(v))
        alt = MgBlock(Mg=1.2e-3 * u.molar, alpha=62.0 / u.volt, beta=3.57e-3 * u.molar,
                      V_offset=0.0 * u.volt)
        np.testing.assert_allclose(np.asarray(alt(v)), ref, rtol=1e-12)
        assert alt == MgBlock() and hash(alt) == hash(MgBlock())

    def test_mg_concentration_and_offset(self, x64):
        v = np.array([-80.0, -40.0, 0.0])
        got = np.asarray(MgBlock(Mg=2.0 * u.mM)(jnp.asarray(v) * u.mV))
        np.testing.assert_allclose(got, _jahr_stevens(v, 2.0), rtol=1e-13)
        shifted = np.asarray(MgBlock(V_offset=10.0 * u.mV)(jnp.asarray(v + 10.0) * u.mV))
        np.testing.assert_allclose(shifted, _jahr_stevens(v), rtol=1e-12)

    def test_mg_free_is_unblocked(self):
        np.testing.assert_array_equal(
            np.asarray(MgBlock(Mg=0.0 * u.mM)(jnp.array([-120.0, -65.0, 30.0]) * u.mV)), 1.0)

    def test_monotone_and_bounded(self):
        b = np.asarray(MgBlock()(jnp.linspace(-150.0, 100.0, 501) * u.mV))
        assert np.all(np.diff(b) > 0) and np.all((b > 0) & (b < 1))

    def test_extreme_voltages_are_finite_float32(self):
        b = MgBlock()(jnp.array([-2000.0, 2000.0], jnp.float32) * u.mV)
        assert b.dtype == jnp.float32 and np.all(np.isfinite(np.asarray(b)))
        np.testing.assert_allclose(np.asarray(b), [0.0, 1.0], atol=1e-6)

    @pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
    def test_dtype_follows_V(self, dtype):
        with jax.enable_x64(dtype == jnp.float64):
            b = MgBlock()(jnp.zeros(3, dtype) * u.mV)
            assert b.dtype == dtype

    def test_integer_V_promotes_to_float(self):
        b = MgBlock()(jnp.array([0, -65]) * u.mV)
        assert jnp.issubdtype(b.dtype, jnp.floating)

    def test_jit_and_vmap(self):
        B = MgBlock()
        v = jnp.linspace(-80.0, 0.0, 6).reshape(2, 3) * u.mV
        direct = np.asarray(B(v))
        np.testing.assert_allclose(np.asarray(jax.jit(B)(v)), direct, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(jax.vmap(B)(v)), direct, rtol=1e-6)


class TestMgBlockValidation:
    def test_unitless_V_rejected(self):
        # goal-01 bug 2: the old unitless MgBlock silently mixed mV mantissas
        with pytest.raises(UnitMismatchError, match='MgBlock V'):
            MgBlock()(jnp.array([-65.0]))

    def test_wrong_V_dimension_rejected(self):
        with pytest.raises(UnitMismatchError):
            MgBlock()(jnp.array([1.0]) * u.nS)

    @pytest.mark.parametrize('kw', [dict(alpha=0.062), dict(Mg=1.2), dict(beta=3.57 * u.mV),
                                    dict(V_offset=0.0)])
    def test_unitless_constants_rejected(self, kw):
        with pytest.raises(UnitMismatchError):
            MgBlock(**kw)

    def test_value_checks(self):
        with pytest.raises(ValueError, match='Mg must be >= 0'):
            MgBlock(Mg=-1.0 * u.mM)
        with pytest.raises(ValueError, match='beta must be > 0'):
            MgBlock(beta=0.0 * u.mM)
        with pytest.raises(ValueError, match='scalar'):
            MgBlock(Mg=jnp.array([1.0, 2.0]) * u.mM)

    def test_quantity_properties(self):
        B = MgBlock(Mg=1.0 * u.mM, alpha=0.05 / u.mV, beta=3.0 * u.mM, V_offset=-5.0 * u.mV)
        assert B.Mg.to_decimal(u.mM) == 1.0 and B.beta.to_decimal(u.mM) == 3.0
        assert float(B.alpha.to_decimal(u.get_unit(1 / u.mV))) == pytest.approx(0.05)
        assert B.V_offset.to_decimal(u.mV) == -5.0

    def test_usable_as_port_gate(self):
        port = Conductance(u.nS, E_rev=0.0 * u.mV, gate=MgBlock())
        assert port.gate == MgBlock()
