# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the JAX precise-timing models (K0 T7): ``iaf_psc_exp_ps``,
``poisson_generator_ps``, ``spike_generator_ps`` and the Illinois root finder.

NEST parity lives in ``precise_nest_test.py``; these tests check the models against
closed forms, invariances and statistics.
"""
import math

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.models import iaf_psc_exp_ps, poisson_generator_ps, spike_generator_ps
from brainpy_state._kernel.models.precise import illinois_root
from brainpy_state._kernel.params import read_states
from brainpy_state._kernel.ports import TimedEvent, TimedEventPayload

DT = 0.1


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def event_table(n_steps, n, events, dt=DT):
    """Host ``(n_steps, n, M)`` offsets / weights / valid from ``(t_arrival, w, neuron)``."""
    per = {}
    for t, w, i in events:
        step = int(math.ceil(round(t / dt, 9))) - 1
        if 0 <= step < n_steps:
            per.setdefault((step, i), []).append((t - step * dt, w))
    m = max([len(v) for v in per.values()], default=1)
    off = np.full((n_steps, n, m), np.inf)
    w = np.zeros((n_steps, n, m))
    for (step, i), evs in per.items():
        for k, (o, ww) in enumerate(evs):
            off[step, i, k] = o
            w[step, i, k] = ww
    return off, w, np.isfinite(off)


def run_neuron(neu, n_steps, events=(), I=None, params=None, capacity=None, sort=True,
               record=('V_m', 'I_syn_ex', 'I_syn_in', 'spike', 'spike_offset', 'overflow')):
    """Pure scan of ``neu.step`` with timed inputs; returns recorded mantissas."""
    cap = capacity or neu.inputs['syn'].capacity
    off, w, valid = event_table(n_steps, neu.n, events)
    I = np.zeros((n_steps, neu.n)) if I is None else np.broadcast_to(I, (n_steps, neu.n))
    s0 = read_states(neu.dyn_states)
    p = read_states(neu.param_states) if params is None else params
    dtype = u.get_mantissa(s0.V_m).dtype

    def body(s, x):
        i, o, ww, vv, cur = x
        ctx = StepContext(t=i * DT * u.ms, dt=DT * u.ms, step=i, key=None)
        if sort:
            pay = TimedEventPayload.pack(o * u.ms, ww * u.pA, vv, cap)
        else:  # raw slots: the receiver must sort
            pay = TimedEventPayload(offsets=jnp.where(vv, o, jnp.inf) * u.ms, weights=ww * u.pA,
                                    valid=vv, overflow=jnp.zeros((neu.n,), jnp.int32))
        s2, out = neu.step(s, p, neu.prepare_inputs({'syn': pay, 'I': cur * u.pA}), ctx)
        rec = {k: u.get_mantissa(s2[k]) for k in record}
        return s2, rec

    xs = (jnp.arange(n_steps), jnp.asarray(off, dtype), jnp.asarray(w, dtype), jnp.asarray(valid),
          jnp.asarray(I, dtype))
    _, rec = brainstate.transform.scan(body, s0, xs)
    return {k: np.asarray(v) for k, v in rec.items()}


def spike_times(rec, col=0):
    sp = rec['spike'][:, col]
    idx = np.nonzero(sp)[0]
    return idx * DT + rec['spike_offset'][idx, col]


def psc_response(t, w, tau_s, tau_m=10.0, c_m=250.0):
    """Membrane response (mV) to one exponential PSC of weight ``w`` pA at ``t = 0``."""
    t = np.maximum(t, 0)
    return w * tau_s * tau_m / (c_m * (tau_m - tau_s)) * (np.exp(-t / tau_m) - np.exp(-t / tau_s))


# ---------------------------------------------------------------------------
# root finder
# ---------------------------------------------------------------------------

class TestIllinoisRoot:
    def test_converges_to_machine_precision(self, x64):
        h = jnp.array([2.0, 1.0, 0.1])
        a = jnp.array([2.0, 0.3, 0.001])
        r = illinois_root(lambda t: t * t - a, h)
        np.testing.assert_allclose(r, np.sqrt(a), rtol=1e-14)

    def test_convex_exponential(self, x64):
        # f(t) = exp(5 t) - 2 is strongly convex: plain regula falsi stalls, Illinois does not
        r = illinois_root(lambda t: jnp.exp(5 * t) - 2.0, jnp.array([1.0]))
        np.testing.assert_allclose(r, [math.log(2.0) / 5], rtol=1e-14)

    def test_exact_hit_and_no_sign_change_stay_in_bracket(self, x64):
        r = illinois_root(lambda t: t - 0.5, jnp.array([1.0]))  # first secant hits exactly
        assert float(r[0]) == 0.5
        r = illinois_root(lambda t: t - 2.0, jnp.array([1.0]))
        assert float(r[0]) == 1.0                                # f(h) <= 0: bracket end h
        r = illinois_root(lambda t: t + 1.0, jnp.array([1.0]))
        assert float(r[0]) == 0.0                                # f(0) > 0: bracket end 0
        r = illinois_root(lambda t: t, jnp.array([1.0]))
        assert float(r[0]) == 0.0                                # f(0) >= 0: bracket end 0

    def test_float32_threshold_function_regression(self):
        # a float32 membrane-threshold function whose f(c) is ~1e-6 at the root: the
        # bracket must keep a consistent sign decision (was: midpoint of a collapsed bracket)
        from brainpy_state._kernel.models.precise import _ps_propagators
        F = lambda x: jnp.array([x], jnp.float32)
        h = F(0.00813559)

        def f(t):
            P20, P21e, P21i, em, _, _, _ = _ps_propagators(t, F(10.), F(2.), F(2.), F(250.))
            return P20 * 600.0 + P21e * F(230.93419) + P21i * F(-127.113335) + F(14.9948015) * em - 15.0

        for r in (illinois_root(f, h), jax.jit(lambda h: illinois_root(f, h))(h)):
            assert r.dtype == jnp.float32
            np.testing.assert_allclose(r, [0.0039527], atol=2e-6)


# ---------------------------------------------------------------------------
# iaf_psc_exp_ps
# ---------------------------------------------------------------------------

class TestIafPscExpPsSubthreshold:
    def test_single_event_closed_form(self, x64):
        neu = iaf_psc_exp_ps(1, precision='float64')
        t_e, w = 0.537, 100.0
        rec = run_neuron(neu, 100, [(t_e, w, 0)])
        t_end = (np.arange(100) + 1) * DT
        expected = -70.0 + psc_response(t_end - t_e, w, 2.0)
        np.testing.assert_allclose(rec['V_m'][:, 0], expected, atol=1e-12)
        I_expected = np.where(t_end >= t_e, w * np.exp(-(t_end - t_e) / 2.0), 0)
        np.testing.assert_allclose(rec['I_syn_ex'][:, 0], I_expected, atol=1e-12)

    def test_inhibitory_events_use_tau_syn_in(self, x64):
        neu = iaf_psc_exp_ps(1, params=dict(tau_syn_in=5.0 * u.ms), precision='float64')
        rec = run_neuron(neu, 60, [(0.25, -80.0, 0)])
        t_end = (np.arange(60) + 1) * DT
        np.testing.assert_allclose(rec['V_m'][:, 0], -70.0 + psc_response(t_end - 0.25, -80.0, 5.0),
                                   atol=1e-12)
        assert np.all(rec['I_syn_ex'] == 0)
        assert rec['I_syn_in'][-1, 0] < 0

    def test_superposition_of_events_in_one_step(self, x64):
        # three events in the same step, two of them at the same offset
        evs = [(0.43, 60.0, 0), (0.47, 25.0, 0), (0.47, 15.0, 0), (1.21, -40.0, 0)]
        neu = iaf_psc_exp_ps(1, precision='float64')
        rec = run_neuron(neu, 50, evs)
        t_end = (np.arange(50) + 1) * DT
        expected = -70.0 + sum(psc_response(t_end - t, w, 2.0) for t, w, _ in evs)
        np.testing.assert_allclose(rec['V_m'][:, 0], expected, atol=1e-12)

    def test_receiver_sorts_unsorted_slots(self, x64):
        evs = [(0.48, 30.0, 0), (0.41, 70.0, 0), (0.45, -20.0, 0)]
        a = run_neuron(iaf_psc_exp_ps(1, precision='float64'), 30, evs, sort=True)
        b = run_neuron(iaf_psc_exp_ps(1, precision='float64'), 30, evs, sort=False)
        np.testing.assert_array_equal(a['V_m'], b['V_m'])

    def test_current_port_is_latched_and_I_e_is_immediate(self, x64):
        n_steps = 20
        I = np.zeros((n_steps, 1))
        I[5] = 100.0
        rec = run_neuron(iaf_psc_exp_ps(1, precision='float64'), n_steps, I=I)
        assert np.all(rec['V_m'][:6, 0] == -70.0)          # acts during step 6 only
        assert rec['V_m'][6, 0] > -70.0
        rec2 = run_neuron(iaf_psc_exp_ps(1, params=dict(I_e=100.0 * u.pA), precision='float64'), 1)
        np.testing.assert_allclose(rec2['V_m'][0, 0], rec['V_m'][6, 0], rtol=1e-14)

    def test_v_min_applies_only_on_steps_without_events(self, x64):
        neu = iaf_psc_exp_ps(1, params=dict(V_min=-70.5 * u.mV), precision='float64')
        rec = run_neuron(neu, 40, [(0.05, -2000.0, 0), (0.15, -10.0, 0)])
        V = rec['V_m'][:, 0]
        assert V[1] < -70.5          # step 1 has an event: no clamp (NEST)
        assert np.all(V[2:] >= -70.5)
        assert np.any(V[2:] == -70.5)

    def test_overflow_counter_accumulates_payload_overflow(self, x64):
        neu = iaf_psc_exp_ps.with_capacity(2)(1, precision='float64')
        evs = [(0.42, 10.0, 0), (0.43, 10.0, 0), (0.44, 10.0, 0), (0.45, 10.0, 0), (0.95, 10.0, 0)]
        rec = run_neuron(neu, 12, evs)
        np.testing.assert_array_equal(rec['overflow'][:, 0], [0] * 4 + [2] * 8)
        ref = run_neuron(iaf_psc_exp_ps(1, precision='float64'), 12,
                         [(0.42, 10.0, 0), (0.43, 10.0, 0), (0.95, 10.0, 0)])
        np.testing.assert_allclose(rec['V_m'], ref['V_m'], atol=1e-13)  # the latest are dropped


class TestIafPscExpPsSpiking:
    I_E = 600.0
    params = dict(I_e=I_E * u.pA)

    def t_cross(self, v0=0.0):
        # V(t) - E_L = U_inf (1 - e^{-t/tau}) + v0 e^{-t/tau}; crossing at 15 mV
        u_inf = self.I_E * 10.0 / 250.0
        return -10.0 * math.log((u_inf - 15.0) / (u_inf - v0))

    def test_dc_crossing_time_exact(self, x64):
        rec = run_neuron(iaf_psc_exp_ps(1, params=self.params, precision='float64'), 400)
        t = spike_times(rec)
        tc = self.t_cross()
        # spike, then refractory 2 ms (V frozen), then the same crossing time again
        np.testing.assert_allclose(t[:3], [tc, 2 * tc + 2.0, 3 * tc + 4.0], atol=1e-11)
        assert rec['spike'].max() == 1.0

    @pytest.mark.parametrize('t_ref, ref_ms', [(2.0, 2.0), (0.25, 0.3), (0.05, 0.1), (0.0, 0.1)])
    def test_refractory_rounding_and_short_periods(self, x64, t_ref, ref_ms):
        # NEST: ceil(t_ref / dt) steps; t_ref < dt (NEST raises) is clamped to one step
        rec = run_neuron(iaf_psc_exp_ps(1, params=dict(self.params, t_ref=t_ref * u.ms),
                                        precision='float64'), 500)
        t = spike_times(rec)
        tc = self.t_cross()
        np.testing.assert_allclose(np.diff(t[:4]), tc + ref_ms, atol=1e-11)

    def test_refractory_return_mid_step_and_input_during_refractory(self, x64):
        tc = self.t_cross()
        # an excitatory event during refractoriness charges I_syn_ex but V stays at V_reset
        t_in = tc + 1.0
        rec = run_neuron(iaf_psc_exp_ps(1, params=self.params, precision='float64'), 200,
                         [(t_in, 200.0, 0)])
        k_spike = int(np.nonzero(rec['spike'][:, 0])[0][0])
        k_in = int(math.ceil(round(t_in / DT, 9))) - 1
        k_back = int(math.ceil(round((tc + 2.0) / DT, 9))) - 1
        assert np.all(rec['V_m'][k_spike:k_back, 0] == -70.0)
        assert rec['I_syn_ex'][k_in, 0] > 150.0
        # return in the middle of step k_back: only the remainder of the step integrates
        rest = (k_back + 1) * DT - (tc + 2.0)
        assert 0 < rest < DT
        I0 = 200.0 * math.exp(-((tc + 2.0) - t_in) / 2.0)
        v = self.I_E * 10.0 / 250.0 * (1 - math.exp(-rest / 10.0)) + psc_response(np.array(rest), I0, 2.0)
        np.testing.assert_allclose(rec['V_m'][k_back, 0], -70.0 + float(v), atol=1e-11)

    def test_instant_spike_from_superthreshold_initial_state(self, x64):
        neu = iaf_psc_exp_ps(2, init=dict(V_m=[-50.0, -60.0] * u.mV), precision='float64')
        rec = run_neuron(neu, 3)
        assert rec['spike'][0].tolist() == [1.0, 0.0]
        assert rec['spike_offset'][0, 0] == 0.0
        assert rec['V_m'][0, 0] == -70.0

    def test_output_payload_and_state(self, x64):
        neu = iaf_psc_exp_ps(1, params=self.params, precision='float64')
        ctx = StepContext.create(0.0 * u.ms, DT * u.ms)
        out = neu.update(ctx)
        assert isinstance(out['spike'], TimedEventPayload)
        assert out['spike'].capacity == 1 and not bool(out['spike'].valid[0, 0])
        assert np.isinf(float(neu.dyn_states['spike_offset'].value.to_decimal(u.ms)[0]))

    def test_heterogeneous_population_is_vectorised(self, x64):
        I_e = np.array([300.0, 400.0, 500.0, 0.0])
        neu = iaf_psc_exp_ps(4, params=dict(I_e=I_e * u.pA), precision='float64')
        rec = run_neuron(neu, 200, [(3.33, 300.0, 3), (3.34, 300.0, 3)])
        for i in range(3):
            single = run_neuron(iaf_psc_exp_ps(1, params=dict(I_e=I_e[i] * u.pA), precision='float64'), 200)
            np.testing.assert_array_equal(rec['V_m'][:, i], single['V_m'][:, 0])
        assert rec['V_m'][40, 3] > -70.0

    def test_float32_tolerance(self):
        evs = [(t, w, 0) for t, w in zip(np.linspace(0.33, 38.0, 60), np.tile([150.0, -60.0], 30))]
        with jax.enable_x64(True):
            r64 = run_neuron(iaf_psc_exp_ps(1, params=self.params, precision='float64'), 400, evs)
        r32 = run_neuron(iaf_psc_exp_ps(1, params=self.params, precision='float32'), 400, evs)
        assert r32['V_m'].dtype == np.float32
        np.testing.assert_allclose(r32['V_m'], r64['V_m'], atol=1e-3)     # CAT_A
        t32, t64 = spike_times(r32), spike_times(r64)
        assert len(t32) == len(t64) > 3
        np.testing.assert_allclose(t32, t64, atol=1e-4)


class TestIafPscExpPsGradients:
    def _fn(self, which, **fixed):
        neu = iaf_psc_exp_ps(1, params=dict(I_e=500.0 * u.pA), precision='float64')
        p0 = read_states(neu.param_states)

        def f(x):
            p = p0.replace(**{which: u.Quantity(jnp.reshape(x, (1,)), unit=u.get_unit(p0[which]))}) \
                if which != 'w' else p0
            w = x if which == 'w' else 120.0
            s0 = read_states(neu.dyn_states)

            def body(s, i):
                ctx = StepContext(t=i * DT * u.ms, dt=DT * u.ms, step=i, key=None)
                hit = i == 20
                pay = TimedEventPayload.pack(jnp.full((1, 1), 0.0437) * u.ms,
                                             jnp.reshape(jnp.where(hit, w, 0.0), (1, 1)) * u.pA,
                                             jnp.reshape(hit, (1, 1)), 16)
                s2, _ = neu.step(s, p, neu.prepare_inputs({'syn': pay}), ctx)
                return s2, (u.get_mantissa(s2.V_m)[0], u.get_mantissa(s2.spike_offset)[0], s2.spike[0])

            _, (V, off, sp) = brainstate.transform.scan(body, s0, jnp.arange(250))
            return V, off, sp

        return f

    @pytest.mark.parametrize('which, x0', [('w', 120.0), ('tau_m', 10.0), ('I_e', 500.0)])
    def test_subthreshold_voltage_gradient(self, x64, which, x0):
        f = self._fn(which)
        g = lambda x: f(x)[0][25]  # before the first spike
        x0 = jnp.asarray(x0, jnp.float64)
        grad = jax.grad(g)(x0)
        eps = 1e-6 * max(1.0, abs(float(x0)))
        fd = (g(x0 + eps) - g(x0 - eps)) / (2 * eps)
        np.testing.assert_allclose(grad, fd, rtol=1e-6)

    @pytest.mark.parametrize('which, x0', [('w', 120.0), ('I_e', 500.0), ('tau_m', 10.0)])
    def test_spike_time_gradient(self, x64, which, x0):
        f = self._fn(which)
        x0 = jnp.asarray(x0, jnp.float64)
        _, _, sp = f(x0)
        k = int(np.nonzero(np.asarray(sp))[0][0])

        def t_spike(x):
            _, off, _ = f(x)
            return k * DT + off[k]

        grad = jax.grad(t_spike)(x0)
        eps = 1e-5 * max(1.0, abs(float(x0)))
        fd = (t_spike(x0 + eps) - t_spike(x0 - eps)) / (2 * eps)
        assert np.isfinite(grad) and abs(float(grad)) > 0
        np.testing.assert_allclose(grad, fd, rtol=1e-5)

    def test_jit_and_vmap_over_parameters(self, x64):
        f = self._fn('I_e')
        xs = jnp.array([450.0, 500.0, 620.0])
        V_b = jax.jit(jax.vmap(lambda x: f(x)[0]))(xs)
        for i, x in enumerate(xs):
            np.testing.assert_allclose(V_b[i], f(x)[0], rtol=0, atol=1e-12)


class TestCapacityVariants:
    def test_with_capacity_is_cached_and_named(self):
        A = iaf_psc_exp_ps.with_capacity(32)
        assert A is iaf_psc_exp_ps.with_capacity(32)
        assert A.__name__ == 'iaf_psc_exp_ps' and issubclass(A, iaf_psc_exp_ps)
        assert A.capacity() == 32 and iaf_psc_exp_ps.capacity() == 16
        assert iaf_psc_exp_ps.with_capacity(16) is iaf_psc_exp_ps
        assert A.inputs['syn'].unit == u.pA and isinstance(A.inputs['I'], type(iaf_psc_exp_ps.inputs['I']))
        assert poisson_generator_ps.with_capacity(3).outputs['spike'] == TimedEvent(3)
        a = A(2)
        assert a.inputs['syn'].capacity == 32


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

def run_generator(gen, n_steps, key=0):
    s0 = read_states(gen.dyn_states)
    p = read_states(gen.param_states)
    root = jax.random.key(key)

    def body(s, i):
        ctx = StepContext(t=i * DT * u.ms, dt=DT * u.ms, step=i, key=jax.random.fold_in(root, i))
        s2, out = gen.step(s, p, {}, ctx)
        return s2, (u.get_mantissa(out['spike'].offsets), out['spike'].weights, s2.overflow, s2.spike)

    s, (off, w, ovf, cnt) = brainstate.transform.scan(body, s0, jnp.arange(n_steps))
    return np.asarray(off), np.asarray(w), np.asarray(ovf), np.asarray(cnt)


def train_times(off):
    """Absolute spike times per unit from ``(n_steps, n, K)`` offsets."""
    steps = np.arange(off.shape[0])[:, None, None]
    t = steps * DT + off
    return [np.sort(t[:, i][np.isfinite(t[:, i])]) for i in range(off.shape[1])]


class TestPoissonGeneratorPs:
    def test_rate_isi_and_offsets(self, x64):
        stats = pytest.importorskip('scipy.stats')
        rate, n, n_steps = 200.0, 400, 2500
        gen = poisson_generator_ps(n, params=dict(rate=rate * u.Hz), precision='float64')
        off, w, ovf, cnt = run_generator(gen, n_steps)
        T = n_steps * DT
        total = int(cnt.sum())
        expected = rate * n * T / 1000.0
        assert abs(total - expected) < 4 * math.sqrt(expected)
        assert ovf.max() == 0
        isis = np.concatenate([np.diff(t) for t in train_times(off)])
        assert stats.kstest(isis, 'expon', args=(0, 1000.0 / rate)).pvalue > 1e-3
        rel = off[np.isfinite(off)] / DT
        assert rel.min() > 0 and rel.max() <= 1
        assert stats.kstest(rel, 'uniform').pvalue > 1e-3
        np.testing.assert_array_equal(w[np.isfinite(off)], 1.0)

    def test_dead_time(self, x64):
        stats = pytest.importorskip('scipy.stats')
        rate, dead = 150.0, 3.0
        gen = poisson_generator_ps(300, params=dict(rate=rate * u.Hz, dead_time=dead * u.ms),
                                   precision='float64')
        off, *_ = run_generator(gen, 3000, key=3)
        isis = np.concatenate([np.diff(t) for t in train_times(off)])
        assert isis.min() >= dead - 1e-9
        np.testing.assert_allclose(isis.mean(), 1000.0 / rate, rtol=0.03)
        assert stats.kstest(isis - dead, 'expon', args=(0, 1000.0 / rate - dead)).pvalue > 1e-3
        # equilibrium initialisation: the first spike has the stationary forward recurrence
        first = np.array([t[0] for t in train_times(off) if len(t)])
        # NEST: uniform in [0, dead) with probability dead * rate, else dead + Exp
        assert np.mean(first < dead) == pytest.approx(dead * rate / 1000.0, abs=0.06)

    def test_activity_window(self, x64):
        gen = poisson_generator_ps(50, params=dict(rate=2000.0 * u.Hz, start=10.0 * u.ms,
                                                   stop=20.0 * u.ms, origin=5.0 * u.ms),
                                   precision='float64')
        off, *_ = run_generator(gen, 400)
        t = np.concatenate(train_times(off))
        assert t.size > 500
        assert t.min() > 15.0 and t.max() <= 25.0

    def test_overflow_counted_never_silent(self, x64):
        rate, n, n_steps, K = 40000.0, 200, 200, 2
        gen = poisson_generator_ps.with_capacity(K)(n, params=dict(rate=rate * u.Hz), precision='float64')
        off, w, ovf, cnt = run_generator(gen, n_steps)
        assert cnt.max() == K
        expected = rate * n * n_steps * DT / 1000.0
        emitted = cnt.sum()
        dropped = ovf[-1].sum()
        assert dropped > 0
        np.testing.assert_allclose(emitted + dropped, expected, rtol=0.05)

    def test_zero_rate_and_determinism(self, x64):
        gen = poisson_generator_ps(10, params=dict(rate=[0.0, 500.0] * 5 * u.Hz), precision='float64')
        a = run_generator(gen, 300, key=7)
        b = run_generator(gen, 300, key=7)
        c = run_generator(gen, 300, key=8)
        np.testing.assert_array_equal(a[0], b[0])
        assert not np.array_equal(a[0], c[0])
        assert np.all(a[3][:, 0::2] == 0) and a[3][:, 1::2].sum() > 0

    def test_needs_key(self):
        gen = poisson_generator_ps(1, params=dict(rate=10.0 * u.Hz))
        with pytest.raises(ValueError, match='ctx.key'):
            gen.update(StepContext.create(0.0 * u.ms, DT * u.ms))

    def test_float32(self):
        gen = poisson_generator_ps(100, params=dict(rate=500.0 * u.Hz))
        off, w, ovf, cnt = run_generator(gen, 500)
        assert off.dtype == np.float32
        fin = off[np.isfinite(off)]
        assert fin.min() > 0 and fin.max() <= DT + 1e-7
        assert abs(cnt.sum() - 2500) < 4 * 50


class TestSpikeGeneratorPs:
    def _out(self, gen, n_steps):
        off, w, ovf, cnt = run_generator(gen, n_steps)
        return off, w, ovf, cnt

    def test_offsets_multiplicity_and_grid_times(self, x64):
        times = [0.03, 0.1, 1.07, 1.07, 2.551]
        gen = spike_generator_ps(1, params=dict(spike_times=times * u.ms,
                                                spike_multiplicities=[1, 2, 1, 3, 1]),
                                 precision='float64')
        off, w, ovf, cnt = self._out(gen, 30)
        assert off[0, 0, 0] == pytest.approx(0.03) and w[0, 0, 0] == 1
        assert off[0, 0, 1] == pytest.approx(0.1) and w[0, 0, 1] == 2   # on-grid: offset dt
        np.testing.assert_allclose(off[10, 0, :2], [0.07, 0.07], atol=1e-12)
        assert sorted(w[10, 0, :2].tolist()) == [1.0, 3.0]
        np.testing.assert_allclose(off[25, 0, 0], 0.051, atol=1e-12)
        assert cnt[10, 0] == 4 and cnt.sum() == 8
        np.testing.assert_allclose(np.concatenate(train_times(off)), sorted(times), atol=1e-12)

    def test_per_unit_trains_origin_window_and_overflow(self, x64):
        gen = spike_generator_ps.with_capacity(2)(
            2, params=dict(spike_times=[[0.51, 0.52, 0.53, 3.0] * u.ms, [0.2, 4.05] * u.ms],
                           origin=1.0 * u.ms, stop=4.0 * u.ms), precision='float64')
        off, w, ovf, cnt = self._out(gen, 60)
        t0, t1 = train_times(off)
        np.testing.assert_allclose(t0, [1.51, 1.52, 4.0], atol=1e-12)  # 1.53 dropped (capacity 2)
        np.testing.assert_allclose(t1, [1.2], atol=1e-12)               # 5.05 > origin + stop
        assert ovf[-1].tolist() == [1, 0]
