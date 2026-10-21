# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel import units as U


class TestResolveDtype:
    def test_float32_default(self):
        assert U.resolve_dtype() == np.float32
        assert U.resolve_dtype(None) == np.float32
        assert U.resolve_dtype('float32') == np.float32
        assert U.resolve_dtype(jnp.float32) == np.float32

    def test_float64_requires_x64(self):
        with jax.enable_x64(False):
            with pytest.raises(RuntimeError, match='jax_enable_x64'):
                U.resolve_dtype('float64')

    def test_float64_with_x64(self):
        with jax.enable_x64(True):
            assert U.resolve_dtype('float64') == np.float64
            assert U.resolve_dtype(np.float64) == np.float64

    @pytest.mark.parametrize('bad', ['float16', 'int32', 'bogus', object()])
    def test_unknown(self, bad):
        with pytest.raises(ValueError, match='unknown precision'):
            U.resolve_dtype(bad)


class TestIndexDtype:
    def test_small(self):
        assert U.index_dtype(0) == np.int32
        assert U.index_dtype(2 ** 31 - 1) == np.int32

    def test_large_needs_x64(self):
        with jax.enable_x64(False):
            with pytest.raises(RuntimeError, match='int64'):
                U.index_dtype(2 ** 31)
        with jax.enable_x64(True):
            assert U.index_dtype(2 ** 31) == np.int64

    def test_negative(self):
        with pytest.raises(ValueError):
            U.index_dtype(-1)


class TestUnits:
    def test_as_unit(self):
        assert U.as_unit(None) == u.UNITLESS
        assert U.as_unit(u.mV) == u.mV
        with pytest.raises(TypeError, match='Unit'):
            U.as_unit(2 * u.mV)

    def test_unit_of_and_same_dim(self):
        assert U.unit_of(1.0) == u.UNITLESS
        assert U.unit_of(3 * u.mV) == u.mV
        assert U.same_dim(3 * u.mV, u.volt)
        assert U.same_dim(1.0, None)
        assert not U.same_dim(u.mV, u.ms)

    def test_check_unit(self):
        U.check_unit(1 * u.volt, u.mV)
        U.check_unit(jnp.ones(3), None)
        with pytest.raises(U.UnitMismatchError, match='tau'):
            U.check_unit(1 * u.mV, u.ms, 'tau')
        with pytest.raises(U.UnitMismatchError):
            U.check_unit(1.0, u.mV)
        assert issubclass(U.UnitMismatchError, ValueError)

    def test_to_mantissa_converts_scale(self):
        m = U.to_mantissa(jnp.array([1.0, 2.0]) * u.volt, u.mV)
        np.testing.assert_allclose(m, [1000.0, 2000.0])
        assert isinstance(m, jax.Array)

    def test_to_mantissa_dtype_and_unitless(self):
        m = U.to_mantissa(3, None, jnp.float32)
        assert m.dtype == jnp.float32 and float(m) == 3.0
        m = U.to_mantissa(np.arange(3), None)
        np.testing.assert_array_equal(m, [0, 1, 2])

    def test_to_mantissa_mismatch(self):
        with pytest.raises(U.UnitMismatchError):
            U.to_mantissa(1 * u.mV, u.nS)
        with pytest.raises(U.UnitMismatchError):
            U.to_mantissa(1.0, u.mV)

    def test_to_mantissa_traced(self):
        f = jax.jit(lambda x: U.to_mantissa(u.Quantity(x, unit=u.volt), u.mV))
        np.testing.assert_allclose(f(jnp.ones(2)), [1000.0, 1000.0])

    def test_from_mantissa(self):
        q = U.from_mantissa(jnp.ones(2), u.mV)
        assert isinstance(q, u.Quantity) and q.unit == u.mV
        a = U.from_mantissa([1.0, 2.0], None)
        assert isinstance(a, jax.Array) and not isinstance(a, u.Quantity)

    def test_roundtrip(self):
        x = jnp.array([0.5, -1.0]) * u.nS
        y = U.from_mantissa(U.to_mantissa(x, u.psiemens), u.psiemens)
        assert u.math.allclose(x, y)


class TestGridSteps:
    def test_rounds_up_like_nest(self):
        t = jnp.array([2.0, 0.25, 0.15, 0.04, 0.0, 0.3, jnp.inf])
        np.testing.assert_array_equal(U.grid_steps(t, 0.1), [20, 3, 2, 1, 0, 3, jnp.inf])

    def test_float32_snap(self):
        # 2.0 / 0.1 in float32 is not exactly 20: it must not become 21
        x = U.grid_steps(jnp.float32(2.0), jnp.float32(0.1))
        assert float(x) == 20.0
        assert float(U.grid_steps(jnp.float32(0.7), jnp.float32(0.1))) == 7.0

    def test_traced(self):
        f = jax.jit(U.grid_steps)
        assert float(f(1.05, 0.1)) == 11.0

    def test_integer_inputs(self):
        assert float(U.grid_steps(jnp.array(3), jnp.array(1))) == 3.0
