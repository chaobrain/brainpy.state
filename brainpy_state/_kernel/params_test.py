# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel import (
    ConstParamState,
    Init,
    LogNormal,
    Normal,
    Param,
    Params,
    Record,
    States,
    Uniform,
    UnitMismatchError,
    name_key,
    read_states,
    write_states,
)

KEY = jax.random.key(42)


# ---------------------------------------------------------------- Record

class TestRecord:
    def test_access_and_mapping(self):
        r = Record(a=1, b=2 * u.mV)
        assert r.a == 1 and r['b'] == 2 * u.mV
        assert list(r) == ['a', 'b'] and len(r) == 2 and 'a' in r
        assert 'Record(a=1' in repr(r)

    def test_missing_attr(self):
        with pytest.raises(AttributeError, match='fields are'):
            Record(a=1).zz

    def test_immutable(self):
        r = Record(a=1)
        with pytest.raises(AttributeError, match='immutable'):
            r.a = 2

    def test_replace(self):
        r = Record(a=1, b=2)
        r2 = r.replace(b=3)
        assert r2.b == 3 and r.b == 2
        with pytest.raises(KeyError, match='unknown field'):
            r.replace(c=1)

    def test_pytree(self):
        r = Record(V=jnp.ones(2) * u.mV, c=jnp.zeros(2, jnp.int32))
        out = jax.jit(lambda r: r.replace(V=r.V * 2))(r)
        assert isinstance(out, Record) and list(out) == ['V', 'c']
        assert u.math.allclose(out.V, 2 * u.mV)
        leaves = jax.tree_util.tree_leaves(r)
        assert len(leaves) == 2


# ---------------------------------------------------------------- distributions

class TestNormal:
    def test_units_and_std_conversion(self):
        d = Normal(-55 * u.mV, 0.002 * u.volt)  # std given in volts
        assert d.unit == u.mV and d.std == pytest.approx(2.0)
        x = d.sample(KEY, (20000,))
        assert x.unit == u.mV and x.shape == (20000,)
        m = x.mantissa
        assert float(jnp.mean(m)) == pytest.approx(-55, abs=0.1)
        assert float(jnp.std(m)) == pytest.approx(2.0, rel=0.05)

    def test_unitless(self):
        x = Normal(0.0, 1.0).sample(KEY, (3,))
        assert not isinstance(x, u.Quantity)

    def test_mismatch(self):
        with pytest.raises(UnitMismatchError):
            Normal(1 * u.mV, 1 * u.ms)
        with pytest.raises(UnitMismatchError):
            Normal(1 * u.mV, 1.0)
        with pytest.raises(UnitMismatchError):
            Normal(1 * u.mV, 1 * u.mV, low=0 * u.nS)

    def test_negative_std(self):
        with pytest.raises(ValueError, match='non-negative'):
            Normal(0.0, -1.0)

    def test_unit_aware_clipping(self):
        d = Normal(1.0 * u.nS, 5.0 * u.nS, low=0 * u.psiemens, high=0.002 * u.uS)
        assert d.low == 0.0 and d.high == pytest.approx(2.0)
        x = d.sample(KEY, (5000,)).to_decimal(u.nS)
        assert float(x.min()) == 0.0 and float(x.max()) == pytest.approx(2.0)
        assert d.bounds() == (0.0, pytest.approx(2.0))

    def test_inverted_clip(self):
        with pytest.raises(ValueError, match='inverted'):
            Normal(0.0, 1.0, low=1.0, high=0.0)

    def test_bounds(self):
        assert Normal(0.0, 1.0).bounds() == (-np.inf, np.inf)
        assert Normal(3.0, 0.0).bounds() == (3.0, 3.0)

    def test_dtype_and_zero_size(self):
        x = Normal(0 * u.mV, 1 * u.mV).sample(KEY, (0,), jnp.float32)
        assert x.shape == (0,) and x.mantissa.dtype == jnp.float32
        with jax.enable_x64(True):
            x = Normal(0 * u.mV, 1 * u.mV).sample(KEY, (2,), jnp.float64)
            assert x.mantissa.dtype == jnp.float64

    def test_deterministic(self):
        d = Normal(0.0, 1.0)
        np.testing.assert_array_equal(d.sample(KEY, (4,)), d.sample(KEY, (4,)))


class TestUniform:
    def test_sample(self):
        d = Uniform(1 * u.mV, 0.003 * u.volt)
        x = d.sample(KEY, (1000,)).to_decimal(u.mV)
        assert float(x.min()) >= 1.0 and float(x.max()) <= 3.0
        assert d.bounds() == (1.0, pytest.approx(3.0))

    def test_errors(self):
        with pytest.raises(UnitMismatchError):
            Uniform(1 * u.mV, 3 * u.ms)
        with pytest.raises(ValueError):
            Uniform(3.0, 1.0)


class TestLogNormal:
    def test_unit_and_positive(self):
        d = LogNormal(0.0, 0.5, unit=u.nS)
        x = d.sample(KEY, (10000,))
        assert x.unit == u.nS
        m = x.mantissa
        assert float(m.min()) > 0
        assert float(jnp.median(m)) == pytest.approx(1.0, rel=0.05)
        assert d.bounds() == (0.0, np.inf)

    def test_clip_with_other_unit(self):
        d = LogNormal(0.0, 1.0, unit=u.nS, high=1500 * u.psiemens)
        x = d.sample(KEY, (5000,)).mantissa
        assert float(x.max()) == pytest.approx(1.5)
        assert d.bounds() == (0.0, pytest.approx(1.5))

    def test_unitless_default(self):
        x = LogNormal(0.0, 1.0).sample(KEY, (3,))
        assert not isinstance(x, u.Quantity)

    def test_errors(self):
        with pytest.raises(UnitMismatchError, match='log-space'):
            LogNormal(1 * u.mV, 1.0, unit=u.mV)
        with pytest.raises(UnitMismatchError):
            LogNormal(0.0, 1.0, unit=u.nS, low=0 * u.mV)
        with pytest.raises(ValueError, match='non-negative'):
            LogNormal(0.0, -1.0)
        with pytest.raises(TypeError):
            LogNormal(0.0, 1.0, unit=1 * u.nS)


# ---------------------------------------------------------------- Param

class TestParam:
    def test_infer_unit(self):
        assert Param(10 * u.ms).unit == u.ms
        assert Param(1.0).unit == u.UNITLESS
        assert Param(dist=Normal(1 * u.mV, 1 * u.mV)).unit == u.mV
        p = Param(Normal(1 * u.mV, 1 * u.mV))  # distribution as default
        assert p.dist is not None and p.default is None

    def test_explicit_unit_mismatch(self):
        with pytest.raises(UnitMismatchError):
            Param(10 * u.ms, u.mV)
        with pytest.raises(UnitMismatchError):
            Param(dist=Normal(1 * u.mV, 1 * u.mV), unit=u.ms)
        with pytest.raises(TypeError):
            Param(1.0, dist=3.0)

    def test_materialize_per_neuron_broadcast_and_convert(self):
        p = Param(-70 * u.mV)
        v = p.materialize(3, value=-0.06 * u.volt)
        assert v.unit == u.mV and v.shape == (3,)
        np.testing.assert_allclose(v.mantissa, -60.0)

    def test_array_override(self):
        v = Param(1 * u.ms).materialize(3, value=jnp.array([1., 2., 3.]) * u.ms)
        np.testing.assert_allclose(v.mantissa, [1, 2, 3])
        with pytest.raises(ValueError, match='shape'):
            Param(1 * u.ms).materialize(3, value=jnp.ones(4) * u.ms)

    def test_shared(self):
        p = Param(2 * u.ms, per_neuron=False)
        assert p.materialize(5).shape == ()
        assert p.materialize(5, value=jnp.ones((2, 2)) * u.ms).shape == (2, 2)
        assert p.materialize(5, KEY, value=Normal(1 * u.ms, 1 * u.ms)).shape == ()

    def test_required(self):
        with pytest.raises(ValueError, match="'rate' is required"):
            Param(unit=u.Hz).materialize(2, name='rate')
        assert Param(unit=u.Hz).materialize(2, value=5 * u.Hz).shape == (2,)

    def test_dist_default_needs_key(self):
        p = Param(1 * u.nS, dist=Uniform(0 * u.nS, 1 * u.nS))
        with pytest.raises(ValueError, match='key is required'):
            p.materialize(3)
        x = p.materialize(3, KEY)
        assert x.unit == u.nS and float(x.mantissa.max()) <= 1.0

    def test_dist_unit_mismatch_override(self):
        with pytest.raises(UnitMismatchError):
            Param(1 * u.nS).materialize(3, KEY, value=Normal(1 * u.mV, 1 * u.mV))

    def test_dist_converted_to_param_unit(self):
        x = Param(1 * u.nS).materialize(3, KEY, value=Uniform(1 * u.psiemens, 2 * u.psiemens))
        assert x.unit == u.nS
        assert float(x.mantissa.max()) <= 0.002

    def test_zero_size(self):
        assert Param(1 * u.ms).materialize(0).shape == (0,)
        assert Param(dist=Normal(0 * u.mV, 1 * u.mV)).materialize(0, KEY).shape == (0,)

    def test_value_override_unit_mismatch(self):
        with pytest.raises(UnitMismatchError, match="'tau'"):
            Param(1 * u.ms).materialize(2, value=1 * u.mV, name='tau')


# ---------------------------------------------------------------- Params

def lif_params():
    return Params(
        V_th=-50 * u.mV,
        tau_m=Param(10 * u.ms, trainable=True),
        w=Param(1 * u.nS, dist=LogNormal(0.0, 0.3, unit=u.nS, high=2 * u.nS)),
        g=1.5,
    )


class TestParams:
    def test_mapping(self):
        ps = lif_params()
        assert list(ps) == ['V_th', 'tau_m', 'w', 'g'] and len(ps) == 4
        assert isinstance(ps['g'], Param)
        assert 'Params(V_th' in repr(ps)

    def test_extend(self):
        ps = lif_params().extend(V_th=-45 * u.mV, b=Param(0 * u.pA))
        assert list(ps) == ['V_th', 'tau_m', 'w', 'g', 'b']
        assert ps['V_th'].default == -45 * u.mV

    def test_state_types(self):
        st = lif_params().materialize(5, KEY)
        assert type(st['tau_m']) is brainstate.ParamState
        for k in ('V_th', 'w', 'g'):
            assert type(st[k]) is ConstParamState
            assert not isinstance(st[k], brainstate.ParamState)
        assert st['V_th'].value.shape == (5,)
        assert st['V_th'].value.mantissa.dtype == jnp.float32

    def test_overrides_and_trainable(self):
        st = lif_params().materialize(2, KEY, overrides=dict(V_th=-40 * u.mV), trainable=['g'])
        np.testing.assert_allclose(st['V_th'].value.mantissa, -40.0)
        assert type(st['g']) is brainstate.ParamState

    def test_unknown_names(self):
        with pytest.raises(KeyError, match='unknown parameter'):
            lif_params().materialize(2, KEY, overrides=dict(nope=1.0))
        with pytest.raises(KeyError, match='unknown parameter'):
            lif_params().materialize(2, KEY, trainable=['nope'])

    def test_no_key_for_dist(self):
        with pytest.raises(ValueError, match="'w'"):
            lif_params().materialize(2)

    def test_heterogeneous_by_default(self):
        w = lif_params().materialize(100, KEY)['w'].value.mantissa
        assert float(jnp.std(w)) > 0
        assert float(w.max()) <= 2.0

    def test_name_keyed_streams_are_stable(self):
        a = lif_params().materialize(10, KEY)['w'].value
        b = lif_params().extend(zz=Param(dist=Normal(0.0, 1.0))).materialize(10, KEY)['w'].value
        assert u.math.array_equal(a, b)
        assert not jnp.array_equal(jax.random.key_data(name_key(KEY, 'a')),
                                   jax.random.key_data(name_key(KEY, 'b')))

    def test_zero_size_population(self):
        st = lif_params().materialize(0, KEY)
        assert all(s.value.shape == (0,) for s in st.values())

    def test_float64_tier(self):
        with jax.enable_x64(True):
            st = lif_params().materialize(3, KEY, precision='float64')
            assert st['V_th'].value.mantissa.dtype == jnp.float64
            assert st['w'].value.mantissa.dtype == jnp.float64
        with jax.enable_x64(False):
            with pytest.raises(RuntimeError):
                lif_params().materialize(3, KEY, precision='float64')

    def test_raw_uint32_key(self):
        st = lif_params().materialize(3, jax.random.PRNGKey(0))
        assert st['w'].value.shape == (3,)


# ---------------------------------------------------------------- traced params

class TestTracedParams:
    def test_const_param_change_does_not_recompile(self):
        st = lif_params().materialize(4, KEY)
        traces = []

        @brainstate.transform.jit
        def f(x):
            traces.append(1)
            return (st['V_th'].value.to_decimal(u.mV) + x).sum()

        assert float(f(1.0)) == pytest.approx(4 * -49.0)
        st['V_th'].value = jnp.full(4, -40.0) * u.mV
        assert float(f(1.0)) == pytest.approx(4 * -39.0)
        assert len(traces) == 1

    def test_grad_only_trainable(self):
        st = lif_params().materialize(3, KEY)

        def loss():
            tau = st['tau_m'].value.to_decimal(u.ms)
            vth = st['V_th'].value.to_decimal(u.mV)
            return jnp.sum(tau ** 2 * vth)

        trainable = [s for s in st.values() if isinstance(s, brainstate.ParamState)]
        assert len(trainable) == 1
        grads = brainstate.transform.grad(loss, grad_states=trainable)()
        g = u.get_mantissa(grads[0])
        np.testing.assert_allclose(g, 2 * 10.0 * -50.0)

    def test_vmap_sweep_over_const_param(self):
        # a parameter sweep is a batched ConstParamState value: one trace, no recompile
        st = lif_params().materialize(2, KEY)
        sweep = jnp.array([-50.0, -45.0, -40.0])
        st['V_th'].value = jnp.broadcast_to(sweep[:, None], (3, 2)) * u.mV

        def run():
            return st['V_th'].value.to_decimal(u.mV).sum()

        out = brainstate.transform.vmap(run, in_states=st['V_th'], axis_size=3)()
        np.testing.assert_allclose(out, 2 * sweep)


# ---------------------------------------------------------------- States

class TestInit:
    def test_infer_unit(self):
        assert Init(-70 * u.mV).unit == u.mV
        assert Init(Normal(0 * u.mV, 1 * u.mV)).unit == u.mV
        assert Init(lambda p: p.E_L).unit is None
        with pytest.raises(UnitMismatchError):
            Init(-70 * u.mV, u.ms)

    def test_callable_needs_params(self):
        with pytest.raises(ValueError, match='params are required'):
            Init(lambda p: p.E_L).materialize(2, name='V')
        v = Init(lambda p: p.E_L).materialize(2, params=Record(E_L=jnp.full(2, -65.) * u.mV))
        assert v.unit == u.mV
        np.testing.assert_allclose(v.mantissa, -65)

    def test_callable_with_unit(self):
        v = Init(lambda p: p.E_L, u.volt).materialize(1, params=Record(E_L=-65 * u.mV))
        assert v.unit == u.volt
        np.testing.assert_allclose(v.mantissa, -0.065)

    def test_int_dtype(self):
        v = Init(0, dtype=jnp.int32).materialize(3)
        assert v.dtype == jnp.int32 and v.shape == (3,)

    def test_int_dtype_from_dist(self):
        v = Init(Uniform(0.0, 5.0), dtype=jnp.int32).materialize(3, KEY)
        assert v.dtype == jnp.int32


class TestStates:
    def decl(self):
        return States(
            V_m=Init(lambda p: p.E_L),
            I=0 * u.pA,
            w=Init(Normal(0 * u.pA, 1 * u.pA)),
            ref=Init(0, dtype=jnp.int32),
        )

    def test_types(self):
        st = self.decl().materialize(4, KEY, params=Record(E_L=-70 * u.mV))
        assert type(st['V_m']) is brainstate.HiddenState
        assert type(st['I']) is brainstate.HiddenState
        assert type(st['ref']) is brainstate.ShortTermState
        assert st['V_m'].value.shape == (4,)
        assert list(self.decl()) == ['V_m', 'I', 'w', 'ref'] and len(self.decl()) == 4
        assert isinstance(self.decl()['I'], Init)
        assert 'States(V_m' in repr(self.decl())

    def test_overrides(self):
        st = self.decl().materialize(2, KEY, params=Record(E_L=-70 * u.mV),
                                     overrides=dict(V_m=Uniform(-70 * u.mV, -60 * u.mV)))
        v = st['V_m'].value.mantissa
        assert float(v.min()) >= -70 and float(v.max()) <= -60
        with pytest.raises(KeyError, match='unknown state'):
            self.decl().materialize(2, KEY, overrides=dict(nope=0))

    def test_extend_and_zero_size(self):
        d = self.decl().extend(x=Init(0.0))
        st = d.materialize(0, KEY, params=Record(E_L=-70 * u.mV))
        assert list(st) == ['V_m', 'I', 'w', 'ref', 'x']
        assert all(s.value.shape == (0,) for s in st.values())

    def test_float64(self):
        with jax.enable_x64(True):
            st = self.decl().materialize(2, KEY, params=Record(E_L=-70 * u.mV), precision='float64')
            assert st['V_m'].value.mantissa.dtype == jnp.float64
            assert st['ref'].value.dtype == jnp.int32


class TestReadWrite:
    def test_roundtrip_with_conversion(self):
        st = States(V=-70 * u.mV, c=Init(0, dtype=jnp.int32)).materialize(3)
        r = read_states(st)
        assert isinstance(r, Record) and list(r) == ['V', 'c']
        write_states(st, r.replace(V=jnp.full(3, -0.05) * u.volt))
        assert st['V'].value.unit == u.mV
        np.testing.assert_allclose(st['V'].value.mantissa, -50.0)
        write_states(st, dict(c=jnp.array([1.0, 2.0, 3.9])))
        assert st['c'].value.dtype == jnp.int32
        write_states(st, dict(V=-60 * u.mV))  # scalar broadcast
        np.testing.assert_allclose(st['V'].value.mantissa, -60.0)

    def test_errors(self):
        st = States(V=-70 * u.mV).materialize(3)
        with pytest.raises(KeyError):
            write_states(st, dict(U=1.0))
        with pytest.raises(UnitMismatchError):
            write_states(st, dict(V=1 * u.ms))
        with pytest.raises(ValueError, match='shape'):
            write_states(st, dict(V=jnp.ones(4) * u.mV))

    def test_inside_for_loop(self):
        st = States(V=0 * u.mV).materialize(2)

        def body(i):
            s = read_states(st)
            write_states(st, s.replace(V=s.V + 1 * u.mV))
            return st['V'].value

        out = brainstate.transform.for_loop(body, jnp.arange(5))
        np.testing.assert_allclose(st['V'].value.mantissa, 5.0)
        assert out.shape == (5, 2)
