# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import dataclasses

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import pytest

from brainspike._kernel import StepContext, UnitMismatchError


def make(**kw):
    args = dict(t=0.0 * u.ms, dt=0.1 * u.ms, step=0, key=jax.random.key(0))
    args.update(kw)
    return StepContext.create(**args)


class TestStepContext:
    def test_create_normalises_step(self):
        ctx = make(step=3)
        assert ctx.step.dtype == jnp.int32 and int(ctx.step) == 3

    def test_frozen(self):
        ctx = make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.dt = 1 * u.ms

    @pytest.mark.parametrize('field', ['t', 'dt'])
    def test_time_units_checked(self, field):
        with pytest.raises(UnitMismatchError, match=field):
            make(**{field: 0.1 * u.mV})
        with pytest.raises(UnitMismatchError):
            make(**{field: 0.1})

    def test_step_must_be_scalar(self):
        with pytest.raises(ValueError, match='scalar'):
            make(step=jnp.arange(2))

    def test_key_optional(self):
        ctx = StepContext.create(0 * u.ms, 0.1 * u.ms)
        assert ctx.key is None

    def test_replace_and_advance(self):
        ctx = make(t=1.0 * u.ms, step=10)
        nxt = ctx.advance()
        assert u.math.allclose(nxt.t, 1.1 * u.ms)
        assert int(nxt.step) == 11
        assert nxt.key is ctx.key
        assert int(ctx.step) == 10  # original untouched
        r = ctx.replace(step=jnp.int32(5))
        assert int(r.step) == 5 and r.dt is ctx.dt

    def test_is_pytree_through_jit(self):
        ctx = make(t=2.0 * u.ms, step=20)

        @jax.jit
        def f(c):
            return c.advance()

        out = f(ctx)
        assert isinstance(out, StepContext)
        assert int(out.step) == 21
        assert u.math.allclose(out.t, 2.1 * u.ms)

    def test_for_loop_carries_context(self):
        # repeated stepping goes through brainstate.transform, never a Python loop
        ctx0 = make()

        def body(c, _):
            c = c.advance()
            return c, c.step

        ctx, steps = brainstate.transform.scan(body, ctx0, None, length=5)
        assert int(ctx.step) == 5
        assert steps.tolist() == [1, 2, 3, 4, 5]
        assert u.math.allclose(ctx.t, 0.5 * u.ms)

    def test_keys_in_vmap(self):
        ctxs = jax.vmap(lambda s: make(step=0).replace(step=s))(jnp.arange(3, dtype=jnp.int32))
        assert ctxs.step.shape == (3,)

    def test_spike_options_are_static_fields(self):
        ctx = make()
        assert ctx.spikes == 'hard' and not ctx.relaxed and ctx.detach_reset is None
        rel = make(spikes='relaxed', detach_reset=True)
        assert rel.relaxed and rel.detach_reset is True
        # static: not leaves, part of the tree structure (separate jit traces)
        assert len(jax.tree.leaves(rel)) == len(jax.tree.leaves(ctx))
        assert jax.tree.structure(rel) != jax.tree.structure(ctx)
        assert jax.jit(lambda c: c.advance())(rel).spikes == 'relaxed'
        assert rel.replace(step=jnp.int32(3)).detach_reset is True
        assert ctx.detach_default(False) is False and rel.detach_default(False) is True
        with pytest.raises(ValueError, match='spikes'):
            make(spikes='soft')
        with pytest.raises(TypeError, match='detach_reset'):
            make(detach_reset='yes')
