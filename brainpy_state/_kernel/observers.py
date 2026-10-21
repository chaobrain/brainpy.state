# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Observers: dense spike and state recording (K0 spec §4.8; sparse buffers are K1).

An :class:`Observer` declares *what* to sample (``net.observe(view, 'spike', 'V_m',
interval=...)``); the runner samples it in the step's *observe* phase, after every
population has been updated, and returns a :class:`Recording` per observer inside a
:class:`Results`.

Sampling rules
--------------
With ``k = interval / dt`` steps and the run's local step ``i`` (``0, 1, ...``):

* **event outputs** (``'spike'``) are **counts summed** over each window of ``k``
  steps, so no spike is lost at a coarse interval;
* **states** (``'V_m'``, ...) are **sampled** at the last step of each window.

Sample ``j`` covers local steps ``[j k, (j + 1) k)`` and is stamped with the end of the
window, ``t_start + (j + 1) k dt`` -- the NEST time stamp of a spike emitted in the
window's last step and of a multimeter sample taken there. Run durations must be whole
multiples of every observer's interval, so the sample layout does not depend on where
a run starts and ``cont()`` never re-traces.

Values stay differentiable (float arrays, surrogate gradients flow through ``'spike'``),
so learners (T9) can build losses on recordings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.ports import Event
from brainpy_state._kernel.units import check_unit, from_mantissa

__all__ = ['Observer', 'Recording', 'Results', 'window_init', 'window_update']


def interval_steps(interval: Any, dt: Any) -> int:
    """Interval in whole steps (``None`` means one step).

    Raises
    ------
    ValueError
        If the interval is not a positive whole multiple of ``dt``.
    """
    if interval is None:
        return 1
    check_unit(interval, u.ms, 'interval')
    r = float(u.Quantity(interval).to_decimal(u.ms)) / float(u.Quantity(dt).to_decimal(u.ms))
    k = int(round(r))
    if k < 1 or abs(r - k) > 1e-6 * max(1.0, r):
        raise ValueError(f'interval {interval} must be a positive whole multiple of dt={dt}')
    return k


@dataclass(frozen=True)
class Observer:
    """Declaration of one recording (created by ``Network.observe``).

    Attributes
    ----------
    name : str
        Unique name; key of the :class:`Recording` in :class:`Results`.
    view : PopulationView
        Recorded neurons (columns in view order).
    recordables : tuple of str
        Names to record.
    interval : int
        Sampling interval in steps.
    events : tuple of str
        Recordables that are event outputs (summed per window).
    """

    name: str
    view: Any
    recordables: tuple
    interval: int
    events: tuple

    @classmethod
    def create(cls, net: Any, view: Any, recordables: Any, interval: Any, name: str) -> 'Observer':
        """Validate and build an observer (see ``Network.observe``).

        Raises
        ------
        ValueError
            On no / duplicate recordables, or a bad interval.
        KeyError
            If a recordable is not a state of every model in the view.
        """
        recs = tuple(recordables)
        if not recs:
            raise ValueError('observe needs at least one recordable')
        if len(set(recs)) != len(recs):
            raise ValueError(f'duplicate recordables in {recs}')
        for r in recs:
            if not isinstance(r, str):
                raise TypeError(f'recordables must be str, got {r!r}')
        events = []
        for r in recs:
            is_event = None
            for pop in view.populations:
                decl = pop.model.all_state_decl()
                if r not in decl:
                    raise KeyError(f'{pop.name} ({pop.model.__name__}) has no recordable {r!r}; '
                                   f'available: {list(decl)}')
                ev = isinstance(pop.model.outputs.get(r), Event)
                if is_event is not None and ev != is_event:
                    raise ValueError(f'{r!r} is an event output in some models of the view and a '
                                     f'state in others')
                is_event = ev
            if is_event:
                events.append(r)
        return cls(name=name, view=view, recordables=recs,
                   interval=interval_steps(interval, net.dt), events=tuple(events))

    @property
    def size(self) -> int:
        """Number of recorded neurons."""
        return len(self.view)

    def n_samples(self, n_steps: int) -> int:
        """Samples produced by a run of ``n_steps`` (must be a multiple of the interval)."""
        if n_steps % self.interval:
            raise ValueError(f'run of {n_steps} steps is not a multiple of observer '
                             f'{self.name!r} interval ({self.interval} steps)')
        return n_steps // self.interval


def window_init(n_samples: int, n: int, dtype: Any) -> jax.Array:
    """Zero accumulator ``(n_samples, n)`` for an interval observer."""
    return jnp.zeros((int(n_samples), int(n)), dtype)


def window_update(buf: jax.Array, i: Any, k: int, value: jax.Array, event: bool) -> jax.Array:
    """Accumulate local step ``i`` into the window buffer.

    Parameters
    ----------
    buf : jax.Array, shape (n_samples, n)
        Accumulator.
    i : int or jax.Array
        Local step index of the run.
    k : int
        Interval in steps.
    value : jax.Array, shape (n,)
        This step's value.
    event : bool
        Sum (``True``, events) or sample at the window's last step (``False``).

    Returns
    -------
    jax.Array
        Updated accumulator.
    """
    i = jnp.asarray(i, jnp.int32)
    j = i // k
    row = jax.lax.dynamic_index_in_dim(buf, j, axis=0, keepdims=False)
    value = jnp.asarray(value, buf.dtype)
    if event:
        new = row + value
    else:
        new = jnp.where((i + 1) % k == 0, value, row)
    return jax.lax.dynamic_update_index_in_dim(buf, new, j, axis=0)


@dataclass
class Recording:
    """Samples of one observer for one run.

    Attributes
    ----------
    observer : Observer
        The declaration.
    times : brainunit.Quantity, shape (n_samples,)
        End of each sampling window (NEST stamp).
    data : dict of str to array
        ``recordable -> (n_samples, n)`` (or ``(batch, n_samples, n)``), with units.
    gids : numpy.ndarray
        Global ids of the columns.
    columns : dict of str to numpy.ndarray
        Population name -> column positions.
    batched : bool
        Whether a leading trial axis is present.
    """

    observer: Observer
    times: Any
    data: dict
    gids: np.ndarray
    columns: dict
    batched: bool = False

    def __getitem__(self, recordable: str) -> Any:
        try:
            return self.data[recordable]
        except KeyError:
            raise KeyError(f'{recordable!r} was not recorded by {self.observer.name!r}; '
                           f'recorded: {list(self.data)}') from None

    def __contains__(self, recordable: str) -> bool:
        return recordable in self.data

    def by_population(self, recordable: str) -> dict:
        """Split the columns of ``recordable`` per population (name -> array)."""
        x = self[recordable]
        return {k: x[..., jnp.asarray(cols, jnp.int32)] for k, cols in self.columns.items()}

    def spike_events(self, recordable: str = 'spike', trial: int | None = None
                     ) -> tuple[Any, np.ndarray, np.ndarray]:
        """Host-side event list ``(times, gids, counts)`` of an event recordable.

        Times are the window stamps (exact spike stamps at ``interval == dt``).

        Parameters
        ----------
        recordable : str, default 'spike'
            Event recordable.
        trial : int, optional
            Trial to read when batched (required then).
        """
        if recordable not in self.observer.events:
            raise ValueError(f'{recordable!r} is not an event recordable')
        x = np.asarray(u.get_mantissa(self[recordable]))
        if self.batched:
            if trial is None:
                raise ValueError('batched recording: pass trial=')
            x = x[trial]
        rows, cols = np.nonzero(x)
        t = np.asarray(u.Quantity(self.times).to_decimal(u.ms))[rows]
        return from_mantissa(t, u.ms), self.gids[cols], x[rows, cols]


@dataclass
class Results:
    """Output of ``Runner.simulate`` / ``Runner.cont``.

    Attributes
    ----------
    t_start, t_stop : brainunit.Quantity
        Time span of the run.
    n_steps : int
        Steps simulated.
    recordings : dict of str to Recording
        One per observer.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.models import spike_generator
        >>> from brainpy_state._kernel.network import Network
        >>> net = Network(dt=0.1 * u.ms)
        >>> sg = net.add(spike_generator, 1, params=dict(spike_times=[0.3, 0.8] * u.ms))
        >>> rec = net.observe(sg, 'spike', interval=0.5 * u.ms)
        >>> res = net.compile().runner().simulate(1.0 * u.ms)
        >>> res[rec]['spike'].tolist()
        [[1.0], [1.0]]
        >>> t, gids, counts = res[rec].spike_events()
        >>> t.to_decimal(u.ms).tolist(), gids.tolist()
        ([0.5, 1.0], [0, 0])
    """

    t_start: Any
    t_stop: Any
    n_steps: int
    recordings: dict = field(default_factory=dict)

    def __getitem__(self, key: Any) -> Recording:
        name = key.name if isinstance(key, Observer) else key
        try:
            return self.recordings[name]
        except KeyError:
            raise KeyError(f'no recording {name!r}; available: {list(self.recordings)}') from None

    def __contains__(self, key: Any) -> bool:
        name = key.name if isinstance(key, Observer) else key
        return name in self.recordings

    def __len__(self) -> int:
        return len(self.recordings)
