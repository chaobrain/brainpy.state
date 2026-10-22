# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.integrators import RK, integrate
from brainpy_state._kernel.params import Record
from brainpy_state._kernel.spiking import (
    Block,
    Clamp,
    Refractory,
    SpikeRule,
    count_ste,
)
from brainpy_state._kernel.units import UnitMismatchError


def _margin(s, p):
    return (s.V - p.V_th) / (p.V_th - p.V_reset)


def _reset(s, p):
    return s.replace(V=p.V_reset)


P = Record(V_th=1.0, V_reset=0.0, t_ref=0.25, b=0.5)


def _drive(rule, I, dt=1.0, substeps=8, steps=1, V0=0.0, detach=False):
    """Integrate V' = I (plus w' = 0) with the rule in the substep hook (like NeuronModel)."""
    n = jnp.shape(I)[0] if jnp.ndim(I) else 1
    s = Record(V=jnp.full((n,), V0), w=jnp.zeros(n), spike=jnp.zeros(n),
               refractory_count=jnp.zeros(n, jnp.int32))
    f = lambda t, y, c: {'V': jnp.where(rule.refractory_mask(c), 0.0, I + 0 * y['V']),
                         'w': jnp.zeros_like(y['w'])}
    counts = []
    for _ in range(steps):  # a handful of steps only: unrolled on purpose in tests
        carry = rule.begin(s, P, dt, substeps)
        holder = [s]

        def hook(k, t, y, c):
            st = holder[0].replace(**y)
            st = rule.clamp(st, P, c)
            c = rule.tick(c, 'substep')
            if rule.when == 'substep' or k == substeps - 1:
                st, c = rule.check(st, P, c, detach_reset=detach)
            holder[0] = st
            return {'V': st.V, 'w': st.w}, c

        y, carry = integrate(RK(1, substeps), f, {'V': s.V, 'w': s.w}, 0.0, dt, hook=hook, aux=carry)
        spike, counter = rule.end(carry)
        s = holder[0].replace(**y, spike=spike)
        if counter is not None:
            s = s.replace(refractory_count=counter)
        counts.append(spike)
    return s, jnp.stack(counts)


# ---------------------------------------------------------------- count_ste

def test_count_ste_forward_and_gradient():
    s = jnp.array([0.2, 0.9, 0.4])
    c = jnp.array([0.0, 2.0, 1.0])
    np.testing.assert_array_equal(count_ste(c, s), c)
    g = jax.grad(lambda s: jnp.sum(count_ste(c, s) * jnp.array([1.0, 2.0, 3.0])))(s)
    np.testing.assert_array_equal(g, [1.0, 2.0, 3.0])
    gc = jax.grad(lambda c: jnp.sum(count_ste(c, s)))(c)
    np.testing.assert_array_equal(gc, 0.0)


# ---------------------------------------------------------------- refractory specs

class TestRefractorySpecs:
    def test_step_counts_nest(self):
        p = Record(t_ref=jnp.array([2.0, 0.04, 0.0, 0.06]))
        # ceil(t_ref/dt) + 1, or 0 when t_ref == 0: NEST Time(ms).get_steps() rounds
        # up, so any t_ref > 0 is at least one step (NEST aeif / iaf counters)
        np.testing.assert_array_equal(Block('t_ref').counts(p, 0.1, 4), [21, 2, 0, 2])
        p = Record(t_ref=jnp.array([0.25, 0.15, 0.1, 0.3]))
        np.testing.assert_array_equal(Block('t_ref').counts(p, 0.1, 1), [4, 3, 2, 4])

    def test_substep_counts(self):
        p = Record(t_ref=jnp.array([0.05, 2.0]))
        np.testing.assert_array_equal(Block('t_ref', 'substep').counts(p, 0.1, 4), [2, 80])

    def test_quantity_t_ref(self):
        r = Clamp('V', 0.2 * u.ms)
        assert int(r.counts(Record(), 0.1, 1)) == 3
        assert float(r.duration_ms(Record())) == pytest.approx(0.2)

    def test_validation(self):
        with pytest.raises(ValueError, match='granularity'):
            Block('t_ref', 'tick')
        with pytest.raises(UnitMismatchError):
            Block(2.0 * u.mV)
        with pytest.raises(TypeError, match='state name'):
            Clamp('', 't_ref')
        with pytest.raises(TypeError, match='callable'):
            Clamp('V', 't_ref', value=1.0)
        c = Clamp('V', 't_ref', value=lambda s, p: p.V_reset, granularity='substep')
        assert c.var == 'V' and c.granularity == 'substep' and isinstance(c, Refractory)


# ---------------------------------------------------------------- SpikeRule validation

class TestRuleValidation:
    def test_bad_arguments(self):
        with pytest.raises(TypeError, match='callables'):
            SpikeRule(1, _reset)
        with pytest.raises(TypeError, match='refractory'):
            SpikeRule(_margin, _reset, refractory='t_ref')
        with pytest.raises(TypeError, match='surrogate'):
            SpikeRule(_margin, _reset, surrogate=3)
        with pytest.raises(TypeError, match='detach_reset'):
            SpikeRule(_margin, _reset, detach_reset=1)
        with pytest.raises(ValueError, match='when'):
            SpikeRule(_margin, _reset, when='never')
        with pytest.raises(ValueError, match='differ'):
            SpikeRule(_margin, _reset, output='x', counter='x')

    def test_state_decl(self):
        d = SpikeRule(_margin, _reset).state_decl()
        assert list(d) == ['spike']
        d = SpikeRule(_margin, _reset, refractory=Block('t_ref'), output='spk', counter='rc').state_decl()
        assert list(d) == ['spk', 'rc'] and d['rc'].dtype == np.int32

    def test_resolve_detach(self):
        assert SpikeRule(_margin, _reset).resolve_detach() is False
        assert SpikeRule(_margin, _reset).resolve_detach(True) is True
        assert SpikeRule(_margin, _reset, detach_reset=False).resolve_detach(True) is False

    def test_clamp_var(self):
        assert SpikeRule(_margin, _reset, refractory=Clamp('V', 1 * u.ms)).clamp_var == 'V'
        assert SpikeRule(_margin, _reset, refractory=Block(1 * u.ms)).clamp_var is None


# ---------------------------------------------------------------- condition forms

class TestCondition:
    S = Record(V=jnp.array([0.5, 1.0, 1.5]), w=jnp.zeros(3), spike=jnp.zeros(3))

    def _check(self, rule, s=None):
        s = self.S if s is None else s
        return rule.check(s, P, rule.begin(s, P, 0.1))

    def test_margin_threshold_inclusive(self):
        s, c = self._check(SpikeRule(_margin, _reset))
        np.testing.assert_array_equal(c.count, [0, 1, 1])
        np.testing.assert_array_equal(s.V, [0.5, 0.0, 0.0])

    def test_bool_condition(self):
        s, c = self._check(SpikeRule(lambda s, p: s.V > p.V_th, _reset))
        np.testing.assert_array_equal(c.count, [0, 0, 1])
        with pytest.raises(TypeError, match='bool'):
            self._check(SpikeRule(lambda s, p: s.V > p.V_th, _reset,
                                  surrogate=braintools.surrogate.Sigmoid()))

    def test_united_margin(self):
        s = Record(V=jnp.array([-60.0, -50.0]) * u.mV, spike=jnp.zeros(2))
        p = Record(V_th=-50.0 * u.mV, V_reset=-70.0 * u.mV)
        rule = SpikeRule(lambda s, p: s.V - p.V_th, lambda s, p: s.replace(V=p.V_reset))
        s2, c = rule.check(s, p, rule.begin(s, p, 0.1))
        np.testing.assert_array_equal(c.count, [0, 1])
        assert u.math.allclose(s2.V, [-60.0, -70.0] * u.mV)
        bad = SpikeRule(lambda s, p: s.V - p.V_th, lambda s, p: s.replace(V=p.V_reset),
                        surrogate=braintools.surrogate.Sigmoid())
        with pytest.raises(TypeError, match='dimensionless'):
            bad.check(s, p, bad.begin(s, p, 0.1))
        # a dimensionless Quantity ratio is accepted with a surrogate
        ok = SpikeRule(lambda s, p: (s.V - p.V_th) / (1.0 * u.volt), lambda s, p: s.replace(V=p.V_reset),
                       surrogate=braintools.surrogate.Sigmoid())
        _, c = ok.check(s, p, ok.begin(s, p, 0.1))
        np.testing.assert_array_equal(c.count, [0, 1])

    def test_multi_state_jump_and_dict_jump(self):
        rule = SpikeRule(_margin, lambda s, p: {'V': p.V_reset, 'w': s.w + p.b})
        s, c = self._check(rule)
        np.testing.assert_array_equal(s.w, [0.0, 0.5, 0.5])
        np.testing.assert_array_equal(s.V, [0.5, 0.0, 0.0])

    def test_integer_state_in_jump(self):
        s0 = self.S.replace(spike=jnp.zeros(3))
        s0 = Record(**s0, n=jnp.zeros(3, jnp.int32))
        rule = SpikeRule(_margin, lambda s, p: s.replace(V=p.V_reset, n=s.n + 1),
                         surrogate=braintools.surrogate.Sigmoid())
        s, _ = rule.check(s0, P, rule.begin(s0, P, 0.1))
        np.testing.assert_array_equal(s.n, [0, 1, 1])
        assert s.n.dtype == jnp.int32


# ---------------------------------------------------------------- in-step dynamics

class TestInStep:
    def test_multiple_spikes_counted_in_one_step(self):
        # strongly driven aeif-like toy: V' = 3.5, threshold 1, reset to 0, w += b,
        # no refractoriness: crossings at t = 2/7, 4/7, 6/7 -> 3 spikes in one step
        rule = SpikeRule(_margin, lambda s, p: s.replace(V=p.V_reset, w=s.w + p.b))
        s, counts = _drive(rule, jnp.array([3.5, 0.5]), substeps=14)
        np.testing.assert_array_equal(counts[0], [3.0, 0.0])
        np.testing.assert_allclose(s.w, [1.5, 0.0])
        np.testing.assert_allclose(s.V, [0.5, 0.5], atol=1e-6)

    def test_step_check_counts_at_most_one(self):
        rule = SpikeRule(_margin, _reset, when='step')
        _, counts = _drive(rule, jnp.array([3.5]), substeps=14)
        np.testing.assert_array_equal(counts[0], [1.0])

    def test_refractory_shorter_than_dt(self):
        # substep counter: t_ref = 0.25 = 2 substeps of 1/8; drive 8/unit -> crossing
        # after 1 substep; then 2 refractory substeps and 1 to recharge: a period of
        # 3 substeps -> spikes at substeps 0, 3, ..., 21 of 24 -> 3, 3, 2 per step
        rule = SpikeRule(_margin, _reset, refractory=Clamp('V', 't_ref', granularity='substep'))
        s, counts = _drive(rule, jnp.array([8.0]), substeps=8, steps=3)
        np.testing.assert_array_equal(counts[:, 0], [3.0, 3.0, 2.0])
        # without refractoriness: 8 spikes per step
        _, c2 = _drive(SpikeRule(_margin, _reset), jnp.array([8.0]), substeps=8)
        assert float(c2[0, 0]) == 8.0

    def test_step_granularity_nest_semantics(self):
        # t_ref = 0 -> no refractoriness (NEST aeif): a spike every substep
        rule = SpikeRule(_margin, _reset, refractory=Clamp('V', 0.0 * u.ms))
        _, counts = _drive(rule, jnp.array([8.0]), substeps=8)
        assert float(counts[0, 0]) == 8.0
        # t_ref = 0.25 < dt = 1 rounds UP to one step (NEST): spike, rest of the
        # step + one full step clamped
        rule = SpikeRule(_margin, _reset, refractory=Clamp('V', 't_ref'))
        _, counts = _drive(rule, jnp.array([8.0]), substeps=8, steps=4)
        np.testing.assert_array_equal(counts[:, 0], [1, 0, 1, 0])
        # t_ref = 2 steps: one spike, then the rest of the step + 2 full steps clamped
        rule = SpikeRule(_margin, _reset, refractory=Clamp('V', 2.0 * u.ms))
        s, counts = _drive(rule, jnp.array([8.0]), substeps=8, steps=5)
        np.testing.assert_array_equal(counts[:, 0], [1, 0, 0, 1, 0])
        assert int(s.refractory_count[0]) == 1  # 2 after the spike step, then -1

    def test_block_does_not_clamp(self):
        rule = SpikeRule(_margin, _reset, refractory=Block('t_ref', 'substep'))
        # drive in _drive is zeroed while refractory, so emulate Block: V keeps rising
        s = Record(V=jnp.array([1.2]), spike=jnp.zeros(1), refractory_count=jnp.array([3], jnp.int32))
        c = rule.begin(s, P, 1.0, 8)
        s2, c2 = rule.check(s, P, c)
        assert float(c2.count[0]) == 0.0 and float(s2.V[0]) == pytest.approx(1.2)
        assert rule.clamp(s, P, c) is s

    def test_clamp_value_applied_while_refractory(self):
        rule = SpikeRule(_margin, _reset, refractory=Clamp('V', 't_ref', value=lambda s, p: p.V_reset - 1))
        s = Record(V=jnp.array([0.3, 0.3]), spike=jnp.zeros(2), refractory_count=jnp.array([0, 2], jnp.int32))
        c = rule.begin(s, P, 1.0, 8)
        np.testing.assert_allclose(rule.clamp(s, P, c).V, [0.3, -1.0], rtol=1e-7)

    def test_end_without_refractory(self):
        rule = SpikeRule(_margin, _reset)
        s = Record(V=jnp.zeros(2), spike=jnp.zeros(2))
        c = rule.begin(s, P, 0.1)
        assert c.counter is None and not bool(jnp.any(rule.refractory_mask(c)))
        spike, counter = rule.end(c)
        assert counter is None and spike.shape == (2,)


# ---------------------------------------------------------------- gradients

class TestGradients:
    def _spike_out(self, x, rule, counter=0):
        s = Record(V=x, spike=jnp.zeros_like(x), refractory_count=jnp.full(x.shape, counter, jnp.int32))
        if rule.refractory is None:
            s = Record(V=x, spike=jnp.zeros_like(x))
        c = rule.begin(s, P, 0.1)
        _, c = rule.check(s, P, c)
        return rule.end(c)[0]

    def test_surrogate_gradient_nonzero_near_threshold(self):
        rule = SpikeRule(_margin, _reset, surrogate=braintools.surrogate.Sigmoid())
        g = jax.grad(lambda x: jnp.sum(self._spike_out(x, rule)))(jnp.array([0.9, 1.1]))
        assert np.all(np.asarray(g) > 0)

    def test_hard_spike_zero_gradient(self):
        rule = SpikeRule(_margin, _reset)
        g = jax.grad(lambda x: jnp.sum(self._spike_out(x, rule)))(jnp.array([0.9, 1.1]))
        np.testing.assert_array_equal(g, 0.0)

    def test_refractory_zero_gradient(self):
        rule = SpikeRule(_margin, _reset, refractory=Block('t_ref'), surrogate=braintools.surrogate.Sigmoid())
        x = jnp.array([1.1])
        g_ref = jax.grad(lambda x: jnp.sum(self._spike_out(x, rule, counter=2)))(x)
        g_free = jax.grad(lambda x: jnp.sum(self._spike_out(x, rule, counter=0)))(x)
        assert float(g_ref[0]) == 0.0 and float(g_free[0]) > 0
        assert float(self._spike_out(x, rule, counter=2)[0]) == 0.0

    @pytest.mark.parametrize('detach', [True, False])
    def test_detach_reset(self, detach):
        rule = SpikeRule(_margin, _reset, surrogate=braintools.surrogate.Sigmoid(), detach_reset=detach)

        def v_after(x):
            s = Record(V=x, spike=jnp.zeros_like(x))
            s, _ = rule.check(s, P, rule.begin(s, P, 0.1), detach_reset=rule.resolve_detach())
            return s.V

        x = jnp.array([1.01, 0.99])
        np.testing.assert_allclose(v_after(x), [0.0, 0.99])  # forward identical
        g = jax.vmap(jax.grad(lambda xi: v_after(xi[None])[0]))(x)
        if detach:
            # where-reset: spiking neuron gets 0, non-spiking gets 1
            np.testing.assert_allclose(g, [0.0, 1.0])
        else:
            sur_grad = jax.vmap(jax.grad(braintools.surrogate.Sigmoid()))(x - 1.0)
            # + d spike/dx * (V_reset - V)
            np.testing.assert_allclose(g, [0.0 - 1.01 * sur_grad[0], 1.0 - 0.99 * sur_grad[1]], rtol=1e-6)

    def test_jit_vmap(self):
        rule = SpikeRule(_margin, _reset, refractory=Clamp('V', 't_ref', granularity='substep'),
                         surrogate=braintools.surrogate.Sigmoid())
        f = jax.jit(jax.vmap(lambda I: _drive(rule, I, substeps=8)[1]))
        out = f(jnp.array([[8.0], [0.5]]))
        np.testing.assert_array_equal(out[:, 0, 0], [3.0, 0.0])


# ---------------------------------------------------------------- relaxed (gradient-check) mode

class TestRelaxed:
    SIG = braintools.surrogate.Sigmoid()

    def _check(self, rule, x, counter=None, detach=False, relaxed=True):
        s = Record(V=x, n=jnp.zeros(x.shape, jnp.int32), spike=jnp.zeros_like(x))
        if counter is not None:
            s = Record(**s, refractory_count=jnp.full(x.shape, counter, jnp.int32))
        c = rule.begin(s, P, 0.1)
        s, c = rule.check(s, P, c, detach_reset=detach, relaxed=relaxed)
        return s, c, rule.end(c)

    def test_explicit_flag_no_global_state(self):
        rule = SpikeRule(_margin, _reset, surrogate=self.SIG)
        x = jnp.array([1.0])
        assert float(self._check(rule, x, relaxed=False)[2][0][0]) == 1.0
        assert float(self._check(rule, x, relaxed=True)[2][0][0]) == pytest.approx(0.5)
        assert float(self._check(rule, x, relaxed=False)[2][0][0]) == 1.0
        assert rule.supports_relaxed
        assert not SpikeRule(_margin, _reset).supports_relaxed
        assert not SpikeRule(_margin, _reset, surrogate=lambda m: m).supports_relaxed

    def test_smooth_spike_and_soft_jump(self):
        rule = SpikeRule(_margin, lambda s, p: s.replace(V=p.V_reset, n=s.n + 1), surrogate=self.SIG)
        x = jnp.array([0.5, 1.0, 1.5])
        s, c, (spike, _) = self._check(rule, x)
        sig = np.asarray(self.SIG.surrogate_fun(np.asarray(x) - 1.0))
        np.testing.assert_allclose(spike, sig, rtol=1e-6)
        np.testing.assert_allclose(c.count, sig, rtol=1e-6)
        np.testing.assert_allclose(s.V, np.asarray(x) * (1 - sig), rtol=1e-6)  # V + spk * (0 - V)
        np.testing.assert_array_equal(s.n, [0, 1, 1])  # integer states use the hard condition

    def test_relaxed_is_smooth_and_grad_matches_fd(self):
        rule = SpikeRule(_margin, _reset, surrogate=self.SIG)

        def f(x):
            s, _, (spike, _) = self._check(rule, x)
            return jnp.sum(spike * 2.0 + s.V)

        x = jnp.array([0.7, 1.0, 1.3])
        g = jax.grad(f)(x)
        eps = 1e-3
        fd = [(f(x.at[i].add(eps)) - f(x.at[i].add(-eps))) / (2 * eps) for i in range(3)]
        np.testing.assert_allclose(g, fd, rtol=1e-3)

    def test_detach_and_refractory(self):
        rule = SpikeRule(_margin, _reset, refractory=Block('t_ref'), surrogate=self.SIG)
        x = jnp.array([1.2])
        g = jax.grad(lambda v: jnp.sum(self._check(rule, v, counter=0, detach=True)[0].V))(x)
        _, c, (spike, counter) = self._check(rule, x, counter=0)
        _, c2, (spike_ref, _) = self._check(rule, x, counter=3)
        sig = float(self.SIG.surrogate_fun(0.2))
        np.testing.assert_allclose(g, [1.0 - sig], rtol=1e-6)  # spike weight detached
        assert int(counter[0]) > 0  # hard condition starts the refractory period
        assert float(spike_ref[0]) == 0.0  # refractory neurons emit nothing

    def test_needs_smooth_primitive(self):
        rule = SpikeRule(_margin, _reset, surrogate=lambda m: (m >= 0).astype(float))
        with pytest.raises(TypeError, match='surrogate_fun'):
            self._check(rule, jnp.array([1.2]))

    def test_hard_rules_unaffected(self):
        rule = SpikeRule(_margin, _reset)  # no surrogate: hard spikes even when relaxed
        _, c, _ = self._check(rule, jnp.array([0.5, 1.5]))
        np.testing.assert_array_equal(c.count, [0.0, 1.0])
