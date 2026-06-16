# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
# -*- coding: utf-8 -*-
r"""Host-clamped per-step input devices for persistent (``Simulator.cont``) rollouts.

A closed-loop host loop — the §3.10 pong game is the motivating case — drives the
network in fixed-length chunks and must change the *input* between chunks (which
input neuron fires this turn; how large the reward current is) **without** forcing
the per-chunk ``brainstate.transform.for_loop`` to recompile. NEST's
``spike_generator``/``dc_generator`` cannot serve here: their schedules are baked
constants folded into the compiled trace, so rewriting them per chunk retraces.

``host_drive`` solves this by holding a ``(window, n)`` schedule in a
:class:`brainstate.ShortTermState`, read one row per step via an internal step
counter. :meth:`set_schedule` overwrites the State *contents* (same fixed shape), so
the compiled rollout is reused across chunks — only the values change. Two roles
share the mechanism:

* :class:`host_spike_drive` — emits a per-step spike *multiplicity* (unitless). Wire
  it ``one_to_one`` into ``parrot_neuron`` (weight ``1.0``) exactly as a
  ``spike_generator`` would be; it is a normal holder-backed spike source.
* :class:`host_current_drive` — emits a per-step *current* (pA). It declares
  ``_injects_current`` so the Simulator registers it as a current injector (the same
  ``sum_current_inputs`` ring-buffer path as ``dc_generator``), not a spike source.

Because both are realised as ordinary (non-deferred) populations, the host keeps a
stable handle to the device and calls ``set_schedule`` on it between
``Simulator.cont`` calls.
"""
import brainstate
import brainunit as u
import jax.numpy as jnp
from brainstate.typing import Size

from brainpy_state._nest_base._base import NESTDevice

__all__ = [
    'host_spike_drive',
    'host_current_drive',
]


class _HostDrive(NESTDevice):
    r"""Shared base: a host-settable ``(window, n)`` per-step schedule.

    Parameters
    ----------
    in_size : int or sequence of int
        Number of channels ``n`` (one per downstream input neuron).
    window : int
        Number of steps per chunk; the schedule is ``(window, *varshape)`` and the
        internal counter wraps modulo ``window``.
    name : str, optional
        Module name.
    """
    __module__ = 'brainpy.state'

    #: Subclasses set this unit; ``None`` means a raw unitless count.
    _emit_unit = None

    def __init__(self, in_size: Size = 1, *, window: int, name: str = None):
        super().__init__(in_size=in_size, name=name)
        if int(window) <= 0:
            raise ValueError(f'host_drive window must be a positive int, got {window!r}')
        self.window = int(window)

    @property
    def _sched_shape(self):
        return (self.window,) + tuple(self.varshape)

    def init_state(self, batch_size=None, **kwargs):
        # Host-written per-step schedule (mantissa) + a wrapping step counter. Both
        # are ShortTermStates so a fresh rollout (reset_rollout -> init_all_states)
        # zeroes them and the host then sets the first chunk's schedule.
        self.schedule = brainstate.ShortTermState(jnp.zeros(self._sched_shape))
        self.k = brainstate.ShortTermState(jnp.zeros((), dtype=jnp.int32))

    def set_schedule(self, schedule):
        r"""Install this chunk's ``(window, n)`` schedule and reset the step counter.

        ``schedule`` is a plain array (or a :class:`~brainunit.Quantity`, whose
        mantissa is taken); it must match ``(window, *varshape)`` exactly so the
        State shape — and therefore the compiled rollout — is unchanged.
        """
        mant = u.get_mantissa(schedule) if isinstance(schedule, u.Quantity) else schedule
        arr = jnp.asarray(mant, dtype=float)
        if arr.shape != self._sched_shape:
            raise ValueError(
                f'host_drive schedule must have shape {self._sched_shape} '
                f'(window, *varshape), got {tuple(arr.shape)}')
        self.schedule.value = arr
        self.k.value = jnp.zeros((), dtype=jnp.int32)

    def update(self):
        r"""Emit the current step's schedule row and advance the wrapping counter."""
        row = self.schedule.value[self.k.value]
        self.k.value = (self.k.value + 1) % self.window
        return row if self._emit_unit is None else row * self._emit_unit


class host_spike_drive(_HostDrive):
    r"""Host-settable spike source: per-step multiplicity → ``parrot_neuron``.

    A non-deferred, holder-backed spike source (the substrate captures its per-step
    output like any neuron/generator). Wire it ``one_to_one`` into a
    ``parrot_neuron`` population with the unit gate weight ``1.0``; the parrot relays
    the host-set multiplicity downstream. Set the schedule (1.0 on the active input
    channel at each desired step) between ``Simulator.cont`` chunks.
    """
    __module__ = 'brainpy.state'
    _emit_unit = None                       # unitless spike multiplicity


class host_current_drive(_HostDrive):
    r"""Host-settable current source: per-step pA → a population's current channel.

    Declares ``_injects_current`` so the Simulator registers it as a current
    injector (``sum_current_inputs`` ring buffer, one-step delay — the same path as
    ``dc_generator``) rather than a spike source; it carries no spike holder. The
    schedule values are pA (raw floats). Use it for the dopaminergic reward current
    the pong critic receives, rewritten each chunk from the prior chunk's outcome.
    """
    __module__ = 'brainpy.state'
    _emit_unit = u.pA
    _injects_current = True
