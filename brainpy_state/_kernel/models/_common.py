# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Shared helpers of the K0 models: spike-rule overrides and step utilities."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import brainunit as u

from brainpy_state._kernel.model import NeuronModel, _keep_dependency, latch_name
from brainpy_state._kernel.params import Record
from brainpy_state._kernel.spiking import SpikeRule
from brainpy_state._kernel.units import grid_steps, to_mantissa

__all__ = ['replace_rule', 'SpikingNeuron', 'as_record', 'latch', 'device_window', 'rule_detach']

_RULE_FIELDS = ('condition', 'jump', 'refractory', 'surrogate', 'detach_reset', 'when',
                'output', 'counter')


def replace_rule(rule: SpikeRule, **changes: Any) -> SpikeRule:
    """Copy of ``rule`` with ``changes`` applied (validated by ``SpikeRule``).

    Parameters
    ----------
    rule : SpikeRule
        Rule to copy.
    **changes
        Fields to replace (``surrogate``, ``jump``, ``detach_reset``, ...).

    Returns
    -------
    SpikeRule
    """
    unknown = set(changes) - set(_RULE_FIELDS)
    if unknown:
        raise TypeError(f'unknown SpikeRule field(s) {sorted(unknown)}')
    fields = {k: getattr(rule, k) for k in _RULE_FIELDS}
    fields.update(changes)
    return SpikeRule(**fields)


_UNSET = object()


class SpikingNeuron(NeuronModel):
    """Base of the K0 spiking models: adds a per-instance ``surrogate`` override.

    Parameters
    ----------
    n : int
        Population size.
    params, init : mapping, optional
        Parameter / initial-value overrides (see ``NeuronModel``).
    surrogate : callable or None, optional
        Surrogate spike function replacing the class default (``None`` gives hard
        spikes with zero gradient). Omit to keep the class default.
    **kwargs
        Forwarded to ``NeuronModel`` (``precision``, ``key``, ``trainable``,
        ``integrator``, ``detach_reset``, ``name``).
    """

    def __init__(self, n: int, params: Mapping[str, Any] | None = None,
                 init: Mapping[str, Any] | None = None, *, surrogate: Any = _UNSET,
                 **kwargs: Any) -> None:
        super().__init__(n, params, init, **kwargs)
        if surrogate is not _UNSET:
            self.spiking = replace_rule(type(self).spiking, surrogate=surrogate)


def as_record(inp: Mapping[str, Any]) -> Record:
    """``inp`` as a :class:`Record` (port inputs may be given as a dict)."""
    return inp if isinstance(inp, Record) else Record(**inp)


def latch(s: Record, inp: Record, ports: tuple[str, ...]) -> tuple[dict, dict]:
    """Latched values (acting this step) and the new latches of ``current`` ports.

    Parameters
    ----------
    s : Record
        State values holding ``latch_<port>``.
    inp : Record
        This step's port inputs.
    ports : tuple of str
        Current/graded port names.

    Returns
    -------
    acting : dict
        ``{port: value latched at the previous step}``.
    new : dict
        ``{latch_<port>: this step's value}`` to store (with the braintrace
        dependency workaround of ``model._keep_dependency``).
    """
    acting = {k: s[latch_name(k)] for k in ports}
    # upstream: braintrace#184
    new = {latch_name(k): _keep_dependency(inp[k], s[latch_name(k)]) for k in ports}
    return acting, new


def device_window(p: Record, dt_ms: Any) -> tuple[Any, Any]:
    """NEST device activity bounds ``(t_min, t_max)`` in steps.

    ``t_min = steps(origin + start)`` and ``t_max = steps(origin + stop)`` with
    NEST's round-up conversion (:func:`~brainpy_state._kernel.units.grid_steps`).
    A stamp ``S`` is active iff ``t_min < S <= t_max``.
    """
    origin = to_mantissa(p.origin, u.ms)
    t_min = grid_steps(origin + to_mantissa(p.start, u.ms), dt_ms)
    t_max = grid_steps(origin + to_mantissa(p.stop, u.ms), dt_ms)
    return t_min, t_max


def rule_detach(model: NeuronModel, ctx: Any = None) -> bool:
    """Effective ``detach_reset`` of ``model``'s spike rule.

    The rule's own setting wins; else the running program's option
    (``ctx.detach_reset``); else the model's ``detach_reset``; else False.
    """
    default = model.detach_reset if ctx is None else ctx.detach_default(model.detach_reset)
    return model.spiking.resolve_detach(default)

