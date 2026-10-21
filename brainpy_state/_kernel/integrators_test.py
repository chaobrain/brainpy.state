# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.integrators import RK, Exact, ExpEuler, Integrator, integrate


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


# ODEs with analytic solutions: (f(t, y, aux), y0, exact(T))
def _linear_forced():
    f = lambda t, y, aux: {'y': jnp.cos(t) - y['y']}
    exact = lambda T: 0.5 * (np.cos(T) + np.sin(T)) + 0.5 * np.exp(-T)
    return f, {'y': jnp.array(1.0)}, exact


def _logistic():
    f = lambda t, y, aux: {'y': y['y'] * (1.0 - y['y'])}
    exact = lambda T: 1.0 / (1.0 + 9.0 * np.exp(-T))
    return f, {'y': jnp.array(0.1)}, exact


def _oscillator():
    f = lambda t, y, aux: {'x': y['v'], 'v': -y['x']}
    exact = lambda T: np.cos(T)
    return f, {'x': jnp.array(1.0), 'v': jnp.array(0.0)}, exact


def _error(integ, problem, T=1.0):
    f, y0, exact = problem()
    y, _ = integrate(integ, f, y0, 0.0, T)
    key = 'x' if 'x' in y else 'y'
    return abs(float(y[key]) - exact(T))


class TestConvergenceOrder:
    @pytest.mark.parametrize('problem', [_linear_forced, _logistic, _oscillator])
    @pytest.mark.parametrize('order,subs', [(1, (16, 32, 64)), (2, (8, 16, 32)), (4, (4, 8, 16))])
    def test_rk_order(self, x64, problem, order, subs):
        errs = [_error(RK(order, s), problem) for s in subs]
        rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        for r in rates:
            assert r == pytest.approx(order, abs=0.25), (errs, rates)

    @pytest.mark.parametrize('problem,order', [(_linear_forced, 1), (_logistic, 2)])
    def test_exp_euler_order(self, x64, problem, order):
        # order 1 in general (time-dependent forcing); order 2 for autonomous scalar
        # ODEs, where it is the exponential Rosenbrock-Euler method
        errs = [_error(ExpEuler(s), problem) for s in (16, 32, 64)]
        rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
        for r in rates:
            assert r == pytest.approx(order, abs=0.25), (errs, rates)

    def test_exp_euler_exact_for_linear(self, x64):
        tau, a = 7.0, 3.0
        f = lambda t, y, aux: {'v': (a - y['v']) / tau}
        y, _ = integrate(ExpEuler(1), f, {'v': jnp.array(-1.0)}, 0.0, 5.0)
        assert float(y['v']) == pytest.approx(a + (-1.0 - a) * np.exp(-5.0 / tau), rel=1e-14)

    def test_exp_euler_zero_jacobian_is_euler(self, x64):
        f = lambda t, y, aux: {'v': jnp.ones_like(y['v']) * 2.0}
        y, _ = integrate(ExpEuler(3), f, {'v': jnp.zeros(4)}, 0.0, 1.5)
        np.testing.assert_allclose(y['v'], 3.0, rtol=1e-14)

    def test_exp_euler_per_neuron_diagonal(self, x64):
        taus = jnp.array([1.0, 2.0, 5.0])
        f = lambda t, y, aux: {'v': -y['v'] / taus, 'w': -2.0 * y['w'] + y['v']}
        y, _ = integrate(ExpEuler(1), f, {'v': jnp.ones(3), 'w': jnp.zeros(3)}, 0.0, 0.3)
        np.testing.assert_allclose(y['v'], np.exp(-0.3 / np.asarray(taus)), rtol=1e-14)


class TestHookAndAux:
    def test_hook_called_after_every_substep(self, x64):
        calls = []

        def hook(k, t, y, aux):
            calls.append(k)
            return y, aux + 1

        f = lambda t, y, aux: {'y': jnp.zeros(())}
        y, aux = integrate(RK(2, 4), f, {'y': jnp.array(0.0)}, 1.0, 0.4, hook=hook, aux=0)
        assert calls == [0, 1, 2, 3] and aux == 4

    def test_hook_time_and_reset(self, x64):
        # y' = 1, reset to 0 whenever y >= 0.25 inside the step: counts 3 crossings
        f = lambda t, y, aux: {'y': jnp.ones(())}
        times = []

        def hook(k, t, y, n):
            times.append(float(t))
            hit = y['y'] >= 0.25 - 1e-12
            return {'y': jnp.where(hit, 0.0, y['y'])}, n + hit

        y, n = integrate(RK(1, 8), f, {'y': jnp.array(0.0)}, 2.0, 1.0, hook=hook, aux=0)
        assert int(n) == 4 and float(y['y']) == pytest.approx(0.0)
        np.testing.assert_allclose(times, 2.0 + np.arange(1, 9) / 8)

    def test_aux_reaches_vector_field(self, x64):
        f = lambda t, y, aux: {'y': aux * jnp.ones(())}
        y, _ = integrate(RK(4, 2), f, {'y': jnp.array(0.0)}, 0.0, 2.0, aux=3.0)
        assert float(y['y']) == pytest.approx(6.0)


class TestTransforms:
    def _final(self, k, c, integ):
        f = lambda t, y, aux: {'y': -k * y['y'] + c * jnp.sin(t)}
        y, _ = integrate(integ, f, {'y': jnp.ones_like(k)}, 0.0, 1.0)
        return y['y']

    @pytest.mark.parametrize('integ', [RK(4, 3), RK(2, 5), RK(1, 4), ExpEuler(4)])
    def test_grad_vs_finite_difference(self, x64, integ):
        g = jax.grad(lambda kc: jnp.sum(self._final(kc[0], kc[1], integ)))(jnp.array([0.7, 1.3]))
        eps = 1e-6
        fd = []
        for i in range(2):
            e = np.zeros(2)
            e[i] = eps
            fp = self._final(*(jnp.array([0.7, 1.3]) + e), integ)
            fm = self._final(*(jnp.array([0.7, 1.3]) - e), integ)
            fd.append(float((fp - fm) / (2 * eps)))
        np.testing.assert_allclose(g, fd, rtol=1e-7)

    def test_grad_wrt_traced_dt(self, x64):
        def final(dt):
            y, _ = integrate(RK(4, 4), lambda t, y, a: {'y': -y['y']}, {'y': jnp.array(1.0)}, 0.0, dt)
            return y['y']
        assert float(jax.grad(final)(0.5)) == pytest.approx(-np.exp(-0.5), rel=2e-5)

    def test_jit_and_vmap(self):
        ks = jnp.array([0.5, 1.0, 2.0], jnp.float32)
        integ = RK(4, 16)
        out = jax.jit(jax.vmap(lambda k: self._final(k, 0.0, integ)))(ks)
        assert out.dtype == jnp.float32
        np.testing.assert_allclose(out, np.exp(-np.asarray(ks)), rtol=1e-5)

    def test_no_while_loop_in_jaxpr(self):
        jaxpr = jax.make_jaxpr(lambda k: self._final(k, 1.0, RK(4, 8)))(1.0)
        assert 'while' not in str(jaxpr)


class TestValidation:
    @pytest.mark.parametrize('order', [0, 3, 5, '4'])
    def test_bad_order(self, order):
        with pytest.raises(ValueError, match='order'):
            RK(order)

    @pytest.mark.parametrize('s', [0, -1, 1.5, True])
    def test_bad_substeps(self, s):
        with pytest.raises(ValueError, match='substeps'):
            RK(4, s)
        with pytest.raises(ValueError, match='substeps'):
            ExpEuler(s)

    def test_exact_marker(self):
        e = Exact()
        assert e.exact and e.substeps == 1 and e.order is None
        assert isinstance(e, Integrator) and not RK().exact
        with pytest.raises(TypeError, match='override'):
            integrate(e, lambda t, y, a: y, {'y': jnp.zeros(())}, 0.0, 1.0)

    def test_frozen_and_hashable(self):
        assert RK(2, 3) == RK(2, 3) and hash(RK(2, 3)) == hash(RK(2, 3))
        assert ExpEuler(2) == ExpEuler(2) and RK(2, 3) != RK(2, 4)
        with pytest.raises(Exception):
            RK(2, 3).order = 4
