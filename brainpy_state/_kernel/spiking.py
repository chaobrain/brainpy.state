# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Spike rules: threshold condition, jump map, refractoriness, surrogate output.

One shared implementation replaces the per-model ``get_spike`` / refractory
blocks (architecture doc §7 L1, K0 spec §4.3):

* ``condition(s, p)`` decides when a neuron spikes. It returns either a **signed
  margin** (spike iff ``margin >= 0``; dimensionless when a surrogate is used,
  e.g. ``(s.V - p.V_th) / (p.V_th - p.V_reset)``) or a boolean array (hard
  threshold, no surrogate gradient).
* ``jump(s, p)`` maps the state at a spike to the state just after it, over
  **all** states (``V -> V_reset`` *and* ``w -> w + b`` for ``aeif``). It returns
  a :class:`~brainpy_state._kernel.params.Record` (``s.replace(...)``) or a dict
  of the changed entries.
* ``refractory`` is :class:`Clamp` (hold one variable, typically ``V``, and block
  spikes), :class:`Block` (only block spikes) or ``None``.
* Spikes are **counted**: several spikes in one step (several substeps, short or
  no refractoriness) give a count > 1. The differentiable output is
  :func:`count_ste` of the count and the summed surrogate spikes.

The rule is evaluated by the generated ``NeuronModel.step`` after every
integrator substep, and can be driven by hand-written exact ``step`` methods
through :meth:`SpikeRule.begin`, :meth:`SpikeRule.check` and
:meth:`SpikeRule.end`.

Refractory granularity
----------------------
``granularity='step'`` (default, NEST semantics): the counter holds whole steps.
With ``R = ceil(t_ref / dt)`` (NEST ``Time(ms(t_ref)).get_steps()`` rounds **up**;
see :func:`~brainpy_state._kernel.units.grid_steps`), a spike sets it to ``R + 1``
(or 0 when ``R == 0``), it is
decremented once at the end of every step, so the rest of the spiking step plus
``R`` full steps are refractory -- identical to NEST's
``iaf_*`` (spike checked at the step end) and ``aeif_*`` (spike inside the step)
counters. ``granularity='substep'``: the counter holds substeps of length
``h = dt / substeps``; a spike sets it to ``round(t_ref / h)`` and every elapsed
substep decrements it, so refractory periods shorter than ``dt`` are resolved
inside the step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import brainunit as u
import jax
import jax.numpy as jnp

from brainpy_state._kernel.params import Init, Record, States
from brainpy_state._kernel.units import grid_steps, to_mantissa

__all__ = [
    'count_ste',
    'Refractory',
    'Clamp',
    'Block',
    'SpikeRule',
    'SpikeCarry',
]

_GRANULARITIES = ('step', 'substep')
_WHEN = ('substep', 'step')


def count_ste(count: Any, s: Any) -> Any:
    r"""Straight-through spike count: forward ``count``, backward ``d s``.

    .. math:: \mathrm{count\_ste}(c, s) = \mathrm{stop\_gradient}(c - s) + s

    Parameters
    ----------
    count : array_like
        Exact spike counts (forward value), e.g. ``[0, 2, 1]``.
    s : array_like
        Differentiable surrogate spikes carrying the gradient.

    Returns
    -------
    jax.Array
        Equal to ``count`` in value; its gradient is that of ``s``.
    """
    return jax.lax.stop_gradient(count - s) + s


# ---------------------------------------------------------------------------
# Refractoriness
# ---------------------------------------------------------------------------

def _check_granularity(g: str) -> str:
    if g not in _GRANULARITIES:
        raise ValueError(f'granularity must be one of {_GRANULARITIES}, got {g!r}')
    return g


@dataclass(frozen=True)
class Refractory:
    """Base of refractory specifications.

    Parameters
    ----------
    t_ref : str or brainunit.Quantity
        Refractory period: a time Quantity or the name of a model parameter.
    granularity : {'step', 'substep'}, default 'step'
        Counter resolution (see the module docstring).
    """

    t_ref: Any
    granularity: str = 'step'

    def __post_init__(self) -> None:
        _check_granularity(self.granularity)
        if not isinstance(self.t_ref, str):
            to_mantissa(self.t_ref, u.ms)  # raises UnitMismatchError if not a time

    def duration_ms(self, p: Record) -> Any:
        """Refractory period in ms (mantissa), read from ``p`` when given by name.

        Plain numbers are interpreted as ms.
        """
        t = p[self.t_ref] if isinstance(self.t_ref, str) else self.t_ref
        # plain numbers (parameter mantissas) are taken to be in ms
        return to_mantissa(t, u.ms) if isinstance(t, u.Quantity) else jnp.asarray(t)

    def counts(self, p: Record, dt_ms: Any, substeps: int) -> Any:
        """Counter value set by a spike (int32, per neuron).

        Parameters
        ----------
        p : Record
            Parameter values.
        dt_ms : array_like
            Step size in ms.
        substeps : int
            Integrator substeps per step.

        Returns
        -------
        jax.Array of int32
        """
        t = self.duration_ms(p)
        if self.granularity == 'step':
            n = grid_steps(t, dt_ms).astype(jnp.int32)
            return jnp.where(n > 0, n + 1, 0)
        return jnp.round(t / (dt_ms / substeps)).astype(jnp.int32)


@dataclass(frozen=True, init=False)
class Clamp(Refractory):
    """Refractoriness that holds ``var`` and blocks spikes.

    While refractory the derivative of ``var`` is zero (it keeps its post-jump
    value, e.g. ``V_reset``) and the spike condition is ignored. Other states
    keep evolving (NEST: ``g``, ``w`` continue during refractoriness).

    Parameters
    ----------
    var : str
        State to hold (e.g. ``'V_m'``).
    t_ref : str or brainunit.Quantity
        Refractory period or the name of the parameter holding it.
    value : callable, optional
        ``value(s, p)`` to set ``var`` to after every refractory substep instead
        of only holding it (e.g. ``lambda s, p: p.V_reset``).
    granularity : {'step', 'substep'}, default 'step'
        Counter resolution.
    """

    var: str = ''
    value: Callable[[Record, Record], Any] | None = None

    def __init__(self, var: str, t_ref: Any, value: Callable | None = None,
                 granularity: str = 'step') -> None:
        if not isinstance(var, str) or not var:
            raise TypeError(f'Clamp var must be a state name, got {var!r}')
        if value is not None and not callable(value):
            raise TypeError('Clamp value must be callable (s, p) -> value')
        object.__setattr__(self, 'var', var)
        object.__setattr__(self, 't_ref', t_ref)
        object.__setattr__(self, 'value', value)
        object.__setattr__(self, 'granularity', granularity)
        self.__post_init__()


@dataclass(frozen=True)
class Block(Refractory):
    """Refractoriness that only blocks spikes; all states keep integrating.

    Parameters
    ----------
    t_ref : str or brainunit.Quantity
        Refractory period or the name of the parameter holding it.
    granularity : {'step', 'substep'}, default 'step'
        Counter resolution.
    """


# ---------------------------------------------------------------------------
# Spike rule
# ---------------------------------------------------------------------------

class SpikeCarry(NamedTuple):
    """Per-step spike bookkeeping threaded through the substeps.

    Attributes
    ----------
    count : jax.Array
        Spikes emitted so far in this step (float, exact integers).
    sur : jax.Array
        Sum of the surrogate spikes (carries the gradient of ``count``).
    counter : jax.Array or None
        Refractory counter (int32), ``None`` without refractoriness.
    reset_to : jax.Array or None
        Counter value a spike sets (int32), ``None`` without refractoriness.
    """

    count: Any
    sur: Any
    counter: Any
    reset_to: Any


def _is_float(x: Any) -> bool:
    return jnp.issubdtype(u.get_mantissa(x).dtype, jnp.inexact)


@dataclass(frozen=True, init=False)
class SpikeRule:
    """Threshold, jump and refractoriness of a spiking model.

    Parameters
    ----------
    condition : callable
        ``condition(s, p)`` -> signed margin (spike iff ``>= 0``) or bool array.
    jump : callable
        ``jump(s, p)`` -> ``Record`` (e.g. ``s.replace(V=p.V_reset, w=s.w + p.b)``)
        or a dict of changed states.
    refractory : Clamp or Block, optional
        Refractory semantics; ``None`` means none.
    surrogate : callable, optional
        Surrogate spike function applied to the (dimensionless) margin, e.g.
        ``braintools.surrogate.Sigmoid()``. ``None`` gives hard spikes with zero
        gradient.
    detach_reset : bool, optional
        Stop the gradient through the spike in the jump (the reset) when True.
        ``None`` defers to the model / learner (spec §4.3).
    when : {'substep', 'step'}, default 'substep'
        Check the condition after every substep, or only at the end of the step
        (NEST ``iaf_cond_*`` check once per step after integrating).
    output : str, default 'spike'
        Name of the (float, ``HiddenState``) state holding the step's spike
        output; also the default output port name.
    counter : str, default 'refractory_count'
        Name of the (int32) refractory counter state.

    Raises
    ------
    TypeError
        If ``condition`` / ``jump`` / ``surrogate`` are not callable or
        ``refractory`` has the wrong type.
    ValueError
        If ``when`` is invalid or ``output == counter``.
    """

    condition: Callable[[Record, Record], Any]
    jump: Callable[[Record, Record], Any]
    refractory: Refractory | None
    surrogate: Callable[[Any], Any] | None
    detach_reset: bool | None
    when: str
    output: str
    counter: str

    def __init__(self, condition: Callable, jump: Callable, refractory: Refractory | None = None,
                 surrogate: Callable | None = None, detach_reset: bool | None = None,
                 when: str = 'substep', output: str = 'spike',
                 counter: str = 'refractory_count') -> None:
        if not callable(condition) or not callable(jump):
            raise TypeError('condition and jump must be callables (s, p) -> ...')
        if refractory is not None and not isinstance(refractory, Refractory):
            raise TypeError(f'refractory must be Clamp, Block or None, got {type(refractory).__name__}')
        if surrogate is not None and not callable(surrogate):
            raise TypeError('surrogate must be callable')
        if detach_reset is not None and not isinstance(detach_reset, bool):
            raise TypeError(f'detach_reset must be a bool or None, got {detach_reset!r}')
        if when not in _WHEN:
            raise ValueError(f'when must be one of {_WHEN}, got {when!r}')
        if output == counter:
            raise ValueError('output and counter state names must differ')
        for k, v in dict(condition=condition, jump=jump, refractory=refractory,
                         surrogate=surrogate, detach_reset=detach_reset, when=when,
                         output=output, counter=counter).items():
            object.__setattr__(self, k, v)

    # ---------------------------------------------------------------- declaration

    @property
    def clamp_var(self) -> str | None:
        """Name of the clamped state, or ``None``."""
        return self.refractory.var if isinstance(self.refractory, Clamp) else None

    def state_decl(self) -> States:
        """Extra states the rule needs: the spike output and the refractory counter."""
        decl = {self.output: Init(0.0)}
        if self.refractory is not None:
            decl[self.counter] = Init(0, dtype=jnp.int32)
        return States(**decl)

    def resolve_detach(self, default: bool | None = None) -> bool:
        """Effective ``detach_reset``: the rule's value, else ``default``, else False."""
        if self.detach_reset is not None:
            return self.detach_reset
        return bool(default) if default is not None else False

    # ---------------------------------------------------------------- evaluation

    def begin(self, s: Record, p: Record, dt_ms: Any, substeps: int = 1) -> SpikeCarry:
        """Start-of-step carry: zero counts, the stored refractory counter.

        Parameters
        ----------
        s, p : Record
            State and parameter values (``s`` must hold :attr:`counter` when the
            rule is refractory).
        dt_ms : array_like
            Step size in ms.
        substeps : int, default 1
            Integrator substeps per step.
        """
        ref = s[self.output]
        zero = jnp.zeros_like(u.get_mantissa(ref))
        if self.refractory is None:
            return SpikeCarry(zero, zero, None, None)
        counter = jnp.asarray(s[self.counter], jnp.int32)
        reset_to = jnp.broadcast_to(self.refractory.counts(p, dt_ms, substeps), counter.shape)
        return SpikeCarry(zero, zero, counter, reset_to)

    def refractory_mask(self, carry: SpikeCarry) -> Any:
        """Boolean mask of refractory neurons (all False without refractoriness)."""
        if carry.counter is None:
            return jnp.zeros(carry.count.shape, bool)
        return carry.counter > 0

    def _margin(self, s: Record, p: Record) -> tuple[Any, Any]:
        m = self.condition(s, p)
        if isinstance(m, u.Quantity):
            if self.surrogate is not None and not m.is_unitless:
                raise TypeError(
                    f'spike condition returned a margin with unit {m.unit}; a surrogate '
                    f'needs a dimensionless margin (e.g. (V - V_th) / (V_th - V_reset))')
            m = to_mantissa(m, None) if m.is_unitless else m.mantissa
        m = jnp.asarray(m)
        if m.dtype == jnp.bool_:
            if self.surrogate is not None:
                raise TypeError('spike condition returned a bool array; a surrogate needs '
                                'a signed margin (spike iff margin >= 0)')
            return m, None
        return m >= 0, m

    def check(self, s: Record, p: Record, carry: SpikeCarry, *,
              detach_reset: bool = False) -> tuple[Record, SpikeCarry]:
        """Test the condition once; apply the jump and refractoriness where it holds.

        Parameters
        ----------
        s, p : Record
            State (after the substep) and parameter values.
        carry : SpikeCarry
            Current bookkeeping (from :meth:`begin` or a previous call).
        detach_reset : bool, default False
            Stop the gradient through the spike in the jump.

        Returns
        -------
        s : Record
            State after the jump (unchanged where no spike).
        carry : SpikeCarry
            Updated counts, surrogate sum and refractory counter.

        Notes
        -----
        The jump is ``where(spike, jump(s), s)`` in value; with a surrogate and
        ``detach_reset=False`` it additionally carries the gradient
        ``d spike * (jump(s) - s)`` (the soft-reset surrogate path).
        """
        hard, margin = self._margin(s, p)
        allowed = ~self.refractory_mask(carry)
        hard = hard & allowed
        dtype = carry.count.dtype
        hard_f = hard.astype(dtype)
        if margin is not None and self.surrogate is not None:
            sg = jnp.asarray(self.surrogate(margin), dtype) * allowed.astype(dtype)
            spk = count_ste(hard_f, sg)
        else:
            sg = hard_f
            spk = hard_f
        s = self._jump(s, p, hard, spk, detach_reset)
        counter = carry.counter
        if counter is not None:
            counter = jnp.where(hard, carry.reset_to, counter)
        return s, SpikeCarry(carry.count + hard_f, carry.sur + sg, counter, carry.reset_to)

    def _jump(self, s: Record, p: Record, hard: Any, spk: Any, detach: bool) -> Record:
        j = self.jump(s, p)
        if not isinstance(j, Record):
            j = s.replace(**dict(j))
        soft = None if detach else spk - jax.lax.stop_gradient(spk)
        changes = {}
        for k in s:
            a, b = s[k], j[k]
            if a is b:
                continue
            new = u.math.where(hard, b, a)
            if soft is not None and _is_float(a):
                new = new + soft * (b - a)
            changes[k] = new
        return s.replace(**changes)

    def clamp(self, s: Record, p: Record, carry: SpikeCarry) -> Record:
        """Set the clamped variable to ``Clamp.value(s, p)`` where refractory."""
        ref = self.refractory
        if not isinstance(ref, Clamp) or ref.value is None:
            return s
        mask = self.refractory_mask(carry)
        return s.replace(**{ref.var: u.math.where(mask, ref.value(s, p), s[ref.var])})

    def tick(self, carry: SpikeCarry, granularity: str) -> SpikeCarry:
        """Decrement the refractory counter if its granularity is ``granularity``."""
        if carry.counter is None or self.refractory.granularity != granularity:
            return carry
        return carry._replace(counter=jnp.maximum(carry.counter - 1, 0))

    def end(self, carry: SpikeCarry) -> tuple[Any, Any]:
        """Close the step: spike output and the counter after the step-end decrement.

        Returns
        -------
        spike : jax.Array
            ``count_ste(count, sur)`` -- exact counts forward, surrogate gradient.
        counter : jax.Array or None
            Refractory counter to store.
        """
        carry = self.tick(carry, 'step')
        return count_ste(carry.count, carry.sur), carry.counter
