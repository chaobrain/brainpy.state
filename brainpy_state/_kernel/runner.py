# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Runner: compiled rollouts of a :class:`~brainpy_state._kernel.compiler.Plan` (spec §4.8).

The step contract (architecture doc §7 L2), for step ``n``::

    for each population in the plan's order:
        deliver   inputs of every projection into it: the source output of step
                  n - d (pre-side ring), the arrival-ring slot of step n
                  (post-side), or -- for d = 0 -- the source output of this step
        update    (state, outputs) = model.step(state, params, inputs, ctx)
    emit          plastic projections run their synapse step (pre events d_ax and
                  post events d_den steps late, read before the rings are
                  written) and deposit into their arrival ring d_den steps ahead;
                  outputs -> pre-side rings (slot n % L); per-edge projections clear
                  their consumed slot and deposit into slot (n + k) % L
    observe       observers sample the new states

Projections with a synapse model (K0 T6) deposit into the synapse state of their
(possibly merged) group instead of the port; the group's output goes to the port.

    learn         (optional) ``Program.rollout(..., learn=fn, every=k)`` applies
                  ``fn`` after every k-th step (architecture doc §8.2-6)

A :class:`Program` holds this as **pure** functions of explicit pytrees
(``params``, ``state``); it is the L2 contract (architecture doc §7 L2, decision
D-c). Its forward options -- the spike mode (``spikes='hard'|'relaxed'``) and the
default ``detach_reset`` -- are static fields of the program, passed to every model
through the :class:`~brainpy_state._kernel.context.StepContext` (no global state).
The :class:`Runner` is one client of a program (learners are others, built from
``(Program, state0)``); it keeps the state between calls and runs
``brainstate.transform.scan`` inside one ``brainstate.transform.jit``-compiled
function. The step index and the parameter values are traced arguments and the
number of steps is the only static one, so repeated ``cont()`` calls of the same
duration (and parameter changes) reuse the compiled program.

RNG (spec §4.5, D-g): population ``name`` receives in ``StepContext.key`` one key per
block of :data:`~brainpy_state._kernel.rng.GID_BLOCK` local ids, ``key[k] =
fold_in(population_key(seed, name, n, k), b)`` at step ``n`` of trial ``b`` (``b = 0``
for an unbatched run and for every trial under ``rng='common'``; see
:func:`~brainpy_state._kernel.rng.step_block_keys`); models draw through
:func:`~brainpy_state._kernel.rng.draw_blocked`. Streams depend only on the global
step, so ``cont()`` continues them.

Batching runs ``jax.vmap`` over the state pytree and the trial index; parameters and
weights are shared by the trials.
"""
from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Iterator

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.compiler import Plan, ProjectionPlan
from brainpy_state._kernel.context import StepContext, check_spike_options
from brainpy_state._kernel.delay_scatter import DelayScatter
from brainpy_state._kernel.model import Device, _keep_dependency, latch_name
from brainpy_state._kernel.observers import Recording, Results, window_init, window_update
from brainpy_state._kernel.params import Record, read_states
from brainpy_state._kernel.ports import TimedEvent
from brainpy_state._kernel import timed as _timed
from brainpy_state._kernel.rng import RNG_MODES, population_block_keys, step_block_keys
from brainpy_state._kernel.units import check_unit, from_mantissa, to_mantissa, unit_of

__all__ = ['Program', 'Runner', 'duration_steps', 'trial_indices']


def duration_steps(duration: Any, dt: Any) -> int:
    """Number of steps of ``duration`` (must be a non-negative whole multiple of ``dt``)."""
    check_unit(duration, u.ms, 'duration')
    r = float(u.Quantity(duration).to_decimal(u.ms)) / float(u.Quantity(dt).to_decimal(u.ms))
    n = int(round(r))
    if n < 0 or abs(r - n) > 1e-6 * max(1.0, abs(r)):
        raise ValueError(f'duration {duration} must be a non-negative whole multiple of dt={dt}')
    return n


def trial_indices(batch: int | None, rng: str = 'independent') -> jax.Array:
    """Trial indices folded into the population keys (spec §4.5).

    Parameters
    ----------
    batch : int or None
        Number of trials; ``None`` for an unbatched run (trial 0).
    rng : {'independent', 'common'}, default 'independent'
        ``'independent'``: trial ``b`` gets index ``b``; ``'common'``: every trial
        gets index 0 (identical random numbers).

    Returns
    -------
    jax.Array
        uint32 scalar (unbatched) or ``(batch,)`` array.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._kernel.runner import trial_indices
        >>> trial_indices(3).tolist(), trial_indices(2, 'common').tolist(), int(trial_indices(None))
        ([0, 1, 2], [0, 0], 0)
    """
    if rng not in RNG_MODES:
        raise ValueError(f'rng must be one of {RNG_MODES}, got {rng!r}')
    if batch is None:
        return jnp.zeros((), jnp.uint32)
    if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)) or batch < 1:
        raise ValueError(f'batch must be a positive int or None, got {batch!r}')
    if rng == 'independent':
        return jnp.arange(int(batch), dtype=jnp.uint32)
    return jnp.zeros((int(batch),), jnp.uint32)


def _cast_like(new: Any, old: Any, what: str) -> Any:
    """``new`` with the tree structure, dtypes and shapes of ``old`` (learn-phase carry)."""
    if jax.tree.structure(new) != jax.tree.structure(old):
        raise TypeError(f'learn must return {what} with the structure (and units) of its input; '
                        f'got {jax.tree.structure(new)}, expected {jax.tree.structure(old)}')
    return jax.tree.map(lambda n, o: jnp.broadcast_to(jnp.asarray(n, dtype=jnp.asarray(o).dtype),
                                                      jnp.shape(o)), new, old)


def _strong(tree: Any) -> Any:
    """Drop weak types (they would change the jit signature after the first call)."""
    def f(x):
        if getattr(x, 'weak_type', False):
            return jnp.asarray(x, dtype=x.dtype)
        return x
    return jax.tree.map(f, tree)


def _like(new: Any, old: Any) -> Any:
    """``new`` in the unit, dtype and shape of ``old`` (keeps the scan carry type fixed)."""
    unit = unit_of(old)
    m_old = u.get_mantissa(old)
    m = to_mantissa(new, unit).astype(m_old.dtype)
    m = jnp.broadcast_to(m, jnp.shape(m_old))
    return from_mantissa(m, unit) if not unit.is_unitless else m


def _take(x: jax.Array, local: Any) -> jax.Array:
    return x[local] if isinstance(local, slice) else x[jnp.asarray(local, jnp.int32)]


def _concat(pieces: list, dtype: Any) -> jax.Array:
    if not pieces:
        return jnp.zeros((0,), dtype)
    return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)


class Program:
    """Pure step / rollout functions of a plan: the L2 contract (architecture doc §7 L2).

    ``init_state``, ``read_params``, ``step`` and ``rollout`` are pure functions of
    explicit pytrees; the static structure lives in closures. Every client -- the
    :class:`Runner`, :class:`~brainpy_state._kernel.learn.bptt.BPTT`,
    :class:`~brainpy_state._kernel.learn.etrace.Online` -- is built from a program
    and runs from any state (decision D-c).

    Parameters
    ----------
    plan : Plan
        Compiled network.
    spikes : {'hard', 'relaxed'}, default 'hard'
        Forward spike mode (architecture doc §8.2-2). ``'hard'`` emits exact spike
        counts with a surrogate backward pass. ``'relaxed'`` emits smooth spikes
        (``surrogate.surrogate_fun(margin)``) with soft jumps -- the regime in which
        BPTT is the exact derivative of the simulated loss on recurrent nets (gate
        G6); not for simulation. Relaxed spikes are fractional, so every projection
        uses the exact float payload path. A static field: programs with different
        modes coexist and never share a compiled function.
    detach_reset : bool or None, default None
        Default ``detach_reset`` of every population's spike rule (a rule's own
        explicit setting wins); ``None`` defers to each model's ``detach_reset``.
    etrace : bool, default False
        Apply trainable (``ParamState``) weights through the storages' braintrace
        primitives (:meth:`~brainpy_state._kernel.connectivity.storage.Storage.forward_etrace`)
        so that ``braintrace.compile`` sees them (online learners, T9).

    Raises
    ------
    ValueError
        If ``spikes`` is not a mode, or ``spikes='relaxed'`` and a population's
        model has no relaxed form (a spike rule without a surrogate that provides
        ``surrogate_fun``, or a precise-timing neuron).
    TypeError
        If ``detach_reset`` is not a bool or None.

    Examples
    --------
    .. code-block:: python

        >>> import braintools
        >>> import brainunit as u
        >>> from brainpy_state._kernel.network import Network
        >>> from brainpy_state._kernel.models import LIF
        >>> from brainpy_state._kernel.runner import Program
        >>> net = Network(dt=0.1 * u.ms)
        >>> _ = net.add(LIF, 2, name='h', surrogate=braintools.surrogate.Sigmoid())
        >>> plan = net.compile()
        >>> hard, relaxed = Program(plan), Program(plan, spikes='relaxed')
        >>> st = hard.init_state()
        >>> ext = {'h': {'delta': 1.0 * u.mV}}  # exactly at threshold
        >>> float(hard.step(hard.read_params(), st, 0, 0, ext)[1][0]['spike'][0])
        1.0
        >>> float(relaxed.step(relaxed.read_params(), st, 0, 0, ext)[1][0]['spike'][0])
        0.5
    """

    def __init__(self, plan: Plan, *, spikes: str = 'hard', detach_reset: bool | None = None,
                 etrace: bool = False) -> None:
        check_spike_options(spikes, detach_reset)
        if spikes == 'relaxed':
            bad = []
            for p in plan.populations:
                m = p.model
                rule = getattr(m, 'spiking', None)
                if rule is not None and not rule.supports_relaxed:
                    bad.append(f'{p.name} ({type(m).__name__}: spike rule without a surrogate '
                               f'providing surrogate_fun)')
                elif (rule is None and m.capabilities.precise_spike_times
                      and not isinstance(m, Device)):
                    bad.append(f'{p.name} ({type(m).__name__}: precise spike times)')
            if bad:
                raise ValueError(f"spikes='relaxed' needs a smooth spike forward pass in every "
                                 f"spiking population; these have none: {bad}. Give them a "
                                 f"surrogate (e.g. surrogate=braintools.surrogate.Sigmoid()) or "
                                 f"use spikes='hard'")
        self.plan = plan
        self.spikes = spikes
        self.detach_reset = detach_reset
        self.etrace = bool(etrace)
        self.dtype = plan.dtype
        self.seed = plan.net.seed
        self.dt = plan.net.dt
        self.dt_ms = float(u.Quantity(self.dt).to_decimal(u.ms))
        # unit conversion factors (weight unit x source unit -> target port unit)
        self._scale: dict[tuple, float] = {}
        for pp in plan.projections:
            prj = pp.proj
            src_pop = plan.populations[prj.pre.segments[0].pop] if prj.pre.segments else None
            src_unit = src_pop.out_units[prj.source_port] if src_pop is not None else u.UNITLESS
            for seg in prj.post.segments:
                dst_unit = plan.populations[seg.pop].model.inputs[prj.port].unit
                f = float(u.Quantity(1.0, unit=pp.weight_unit * src_unit).to_decimal(dst_unit))
                self._scale[(prj.index, seg.pop)] = f
        # per-block keys of every population (D-g); folded with the step and trial each step
        self._blk_keys = [population_block_keys(self.seed, p.pop.name, p.pop.n)
                          for p in plan.populations]
        self._syn_by_pop: dict[int, list] = {}
        for gi, g in enumerate(plan.synapses):
            self._syn_by_pop.setdefault(g.pop, []).append(gi)
        # initial state captured now: stateful clients (EtraceNetwork) later write the
        # models' State objects, init_state() must not see their values
        self._init = self._init_state()

    def replace(self, **options: Any) -> 'Program':
        """A program of the same plan with some options changed.

        Parameters
        ----------
        **options
            ``spikes``, ``detach_reset`` and/or ``etrace``.

        Returns
        -------
        Program
            ``self`` when nothing changes, else a new program (the plan, its
            models and parameter states are shared).
        """
        unknown = set(options) - {'spikes', 'detach_reset', 'etrace'}
        if unknown:
            raise TypeError(f'unknown Program options {sorted(unknown)}')
        cur = dict(spikes=self.spikes, detach_reset=self.detach_reset, etrace=self.etrace)
        new = {**cur, **options}
        if new == cur:
            return self
        prog = Program(self.plan, **new)
        prog._init = self._init
        return prog

    def __repr__(self) -> str:
        return (f'Program(spikes={self.spikes!r}, detach_reset={self.detach_reset!r}, '
                f'etrace={self.etrace!r}, populations={len(self.plan.populations)})')

    # ------------------------------------------------------------ structure
    def structure(self) -> dict:
        """Device structure arrays of every storage (``{'<proj>/<group>': {name: array}}``).

        :class:`Runner` passes them to its compiled programs as arguments and binds
        them with :meth:`bound`, so large index arrays are not embedded as constants.
        """
        out = {}
        for pp in self.plan.projections:
            for g, group in enumerate(pp.groups):
                arrays = group.storage.arrays()
                if arrays:
                    out[f'{pp.name}/{g}'] = arrays
        return out

    @contextlib.contextmanager
    def bound(self, structure: dict) -> Iterator[None]:
        """Bind :meth:`structure` values (e.g. jit-argument tracers) to the storages."""
        with contextlib.ExitStack() as stack:
            for pp in self.plan.projections:
                for g, group in enumerate(pp.groups):
                    arrays = structure.get(f'{pp.name}/{g}')
                    if arrays:
                        stack.enter_context(group.storage.bound(arrays))
            yield

    # ------------------------------------------------------------- pytrees
    def init_state(self, batch: int | None = None) -> dict:
        """Initial state pytree: model states and zero buffers.

        The model states are the values the models held when the program (or the
        program it was derived from by :meth:`replace`) was created -- the
        compiled initial values -- not whatever a stateful client wrote to them
        since.

        Parameters
        ----------
        batch : int, optional
            Number of trials: every leaf gets a leading ``(batch,)`` axis (copies
            of the one-trial state); ``None`` for one unbatched trial.

        Returns
        -------
        dict
            ``{'pops', 'rings', 'arrivals', 'syn', 'plastic'}``.
        """
        state = self._init
        if batch is None:
            return state
        trial_indices(batch)  # validates batch
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (int(batch),) + jnp.shape(x)), state)

    def _init_state(self) -> dict:
        pops = {p.name: read_states(p.model.dyn_states) for p in self.plan.populations}
        rings = {f'{p.name}.{port}': r.init() for p in self.plan.populations
                 for port, r in p.rings.items()}
        arrivals = {pp.name: pp.post_ring.init() for pp in self.plan.projections
                    if pp.post_ring is not None}
        syn = {g.key: g.synapse.init_state(self.plan.populations[g.pop].pop.n, self.dtype)
               for g in self.plan.synapses}
        plastic = {pp.name: pp.plastic.init_state() for pp in self.plan.projections
                   if pp.plastic is not None}
        return _strong({'pops': pops, 'rings': rings, 'arrivals': arrivals, 'syn': syn,
                        'plastic': plastic})

    def read_params(self) -> dict:
        """Current parameter and weight values (read from their ``State`` objects)."""
        pops = {p.name: read_states(p.model.param_states) for p in self.plan.populations}
        weights = {name: [s.value for s in states] for name, states in self.plan.weights.items()}
        plasticity = {name: {k: s.value for k, s in states.items()}
                      for name, states in self.plan.plastic_params.items()}
        return _strong({'pops': pops, 'weights': weights, 'plasticity': plasticity})

    def write_params(self, params: dict) -> None:
        """Store a :meth:`read_params` pytree into the parameter ``State`` objects.

        The inverse of :meth:`read_params` (host side, outside ``jit``): the next
        :meth:`read_params` -- and every client of the plan -- sees ``params``.

        Parameters
        ----------
        params : dict
            ``{'pops', 'weights', 'plasticity'}`` as returned by :meth:`read_params`
            (e.g. after a learn phase).
        """
        for p in self.plan.populations:
            values = params['pops'][p.name]
            for k, st in p.model.param_states.items():
                st.value = values[k]
        for name, states in self.plan.weights.items():
            for st, v in zip(states, params['weights'][name]):
                st.value = v
        for name, pstates in self.plan.plastic_params.items():
            for k, st in pstates.items():
                st.value = params['plasticity'][name][k]

    # ------------------------------------------------------------ delivery
    def _pre_payload(self, pp: ProjectionPlan, rings: dict, cur: dict, n: Any) -> jax.Array:
        prj = pp.proj
        lay = pp.layout
        pieces = []
        for seg in prj.pre.segments:
            pplan = self.plan.populations[seg.pop]
            port = prj.source_port
            if lay.kind == 'scalar':
                if lay.steps == 0:
                    piece = _take(cur[seg.pop][port], seg.local)
                else:
                    ring = pplan.rings[port]
                    row = ring.read(rings[f'{pplan.name}.{port}'], n, lay.steps)
                    piece = _take(row, seg.local)
            else:
                d = np.asarray(lay.steps)[seg.pos]
                zero = d == 0
                if zero.all():
                    piece = _take(cur[seg.pop][port], seg.local)
                else:
                    ring = pplan.rings[port]
                    piece = ring.read_per_source(rings[f'{pplan.name}.{port}'], n,
                                                 np.where(zero, 1, d), seg.local_array())
                    if zero.any():
                        piece = jnp.where(jnp.asarray(zero), _take(cur[seg.pop][port], seg.local), piece)
            pieces.append(piece)
        return _concat(pieces, self.dtype)

    def _delayed(self, view: Any, port: str, k: int, rings: dict, cur: dict, n: Any) -> jax.Array:
        """Output ``port`` of ``view`` emitted ``k`` steps ago (``k = 0``: this step)."""
        pieces = []
        for seg in view.segments:
            pplan = self.plan.populations[seg.pop]
            if k == 0:
                row = cur[seg.pop][port]
            else:
                row = pplan.rings[port].read(rings[f'{pplan.name}.{port}'], n, k)
            pieces.append(_take(row, seg.local))
        return _concat(pieces, self.dtype)

    def _plastic(self, pp: ProjectionPlan, params: dict, state: dict, cur: dict, n: Any
                 ) -> tuple[dict, jax.Array]:
        """Synapse step of a plastic projection; returns its new state and delivery."""
        pl = pp.plastic
        assert pl is not None  # called for plastic projections only
        prj = pp.proj
        x = self._delayed(prj.pre, prj.source_port, pl.d_ax, state['rings'], cur, n)
        q = self._delayed(prj.post, prj.plasticity.post_port, pl.d_den, state['rings'], cur, n)
        t = n.astype(self.dtype) * self.dt_ms
        new, w_eff = pl.step(state['plastic'][pp.name], params['plasticity'][pp.name], x, q, t,
                             self.dt_ms)
        y = pp.groups[0].storage.forward(x.astype(self.dtype), w_eff, self._path(pp))
        return new, u.get_mantissa(y)

    def _path(self, pp: ProjectionPlan) -> str:
        # relaxed spikes are fractional: the binary event path would binarise them
        return 'float' if self.spikes == 'relaxed' else pp.path

    def _now_payload(self, pp: ProjectionPlan, cur: dict) -> jax.Array:
        prj = pp.proj
        return _concat([_take(cur[s.pop][prj.source_port], s.local) for s in prj.pre.segments],
                       self.dtype)

    def _forward(self, pp: ProjectionPlan, g: int, payload: jax.Array, params: dict) -> jax.Array:
        group = pp.groups[g]
        w = params['weights'][pp.name][g]
        if self.etrace and isinstance(group.weight, brainstate.ParamState):
            y = group.storage.forward_etrace(payload.astype(self.dtype), w)
        else:
            y = group.storage.forward(payload.astype(self.dtype), w, self._path(pp))
        return u.get_mantissa(y)

    def _deliver_timed(self, pp: ProjectionPlan, params: dict, state: dict, cur: dict, n: Any) -> Any:
        g = pp.groups[0]
        w = u.get_mantissa(g.storage.weight_to_edges(params['weights'][pp.name][0]))
        assert pp.timed is not None  # placement == 'timed' builds a TimedDelivery
        rings, objs = {}, {}
        for s in pp.timed.segments:
            pplan = self.plan.populations[s.pop]
            ring = pplan.rings.get(s.port)
            if ring is not None:
                objs[(s.pop, s.port)] = ring
                rings[(s.pop, s.port)] = state['rings'][f'{pplan.name}.{s.port}']
        return pp.timed.deliver(w, rings, objs, cur, n, self.dtype)

    def _deliver(self, pp: ProjectionPlan, params: dict, state: dict, cur: dict, n: Any,
                 arrived: Any = None) -> Any:
        if pp.placement == 'timed':
            return self._deliver_timed(pp, params, state, cur, n)
        if pp.placement == 'pre':
            return self._forward(pp, 0, self._pre_payload(pp, state['rings'], cur, n), params)
        if arrived is not None:  # the arrival-ring slot of this step (PostRing.take)
            y = arrived
        elif pp.post_ring is not None:
            y = pp.post_ring.read(state['arrivals'][pp.name], n)
        else:
            y = jnp.zeros((pp.edges.n_post,), self.dtype)
        for g, group in enumerate(pp.groups):
            if group.delay == 0:
                y = y + self._forward(pp, g, self._now_payload(pp, cur), params)
        return y

    # ---------------------------------------------------------------- step
    def _ctx(self, a: int, n: Any, trial: Any) -> StepContext:
        key = step_block_keys(self._blk_keys[a], n, trial)
        return StepContext(t=from_mantissa(n.astype(self.dtype) * self.dt_ms, u.ms), dt=self.dt,
                           step=n, key=key, spikes=self.spikes, detach_reset=self.detach_reset)

    def _gather(self, a: int, params: dict, state: dict, cur: dict, n: Any, ctx: StepContext,
                delivered: dict, syn_state: dict, ext: dict | None,
                skip: frozenset = frozenset(), only: frozenset | None = None,
                arrived: dict | None = None) -> dict:
        """Input mantissas of population ``a``: deliveries, synapse groups, external input.

        ``skip`` / ``only`` select input ports (deferred latched ports are gathered
        separately at the end of the step). ``delivered`` caches each projection's
        delivery; ``syn_state`` is updated in place. ``arrived`` holds the arrival-ring
        slot of this step per projection (read by :meth:`PostRing.take`).
        """
        plan = self.plan
        pplan = plan.populations[a]
        model = pplan.model

        def wanted(port: str) -> bool:
            return port not in skip and (only is None or port in only)

        acc: dict[Any, jax.Array] = {}
        for j in pplan.incoming:
            pp = plan.projections[j]
            if not wanted(pp.proj.port):
                continue
            if j not in delivered:
                delivered[j] = self._deliver(pp, params, state, cur, n,
                                             None if arrived is None else arrived.get(pp.name))
            y = delivered[j]
            gi = plan.synapse_route.get((j, a))
            port = pp.proj.port if gi is None else ('syn', gi)
            kind = model.inputs[pp.proj.port] if gi is None else None
            for seg in pp.proj.post.segments:
                if seg.pop != a:
                    continue
                if isinstance(kind, TimedEvent):
                    acc[port] = _timed.accumulate(acc.get(port), y, seg, pplan.pop.n,
                                                  self._scale[(j, a)], kind.capacity,
                                                  self.dt_ms, self.dtype)
                    continue
                ys = y[seg.pos]
                f = self._scale[(j, a)]
                if f != 1.0:
                    ys = ys * f
                if port not in acc and seg.full and seg.local == slice(0, pplan.pop.n):
                    acc[port] = ys
                    continue
                base = acc.get(port)
                if base is None:
                    base = jnp.zeros((pplan.pop.n,), self.dtype)
                if seg.full:
                    acc[port] = base.at[seg.local].add(ys)
                else:
                    acc[port] = base.at[jnp.asarray(seg.local, jnp.int32)].add(ys)
        for gi in self._syn_by_pop.get(a, ()):
            g = plan.synapses[gi]
            if not wanted(g.port):
                continue
            x = acc.pop(('syn', gi), None)
            if x is None:  # pragma: no cover - every member delivers into its group
                x = jnp.zeros((pplan.pop.n,), self.dtype)
            syn_state[g.key], out = g.synapse.step(syn_state[g.key], x, ctx)
            acc[g.port] = out if g.port not in acc else acc[g.port] + out
        if ext is not None and pplan.name in ext:
            values = {k: v for k, v in ext[pplan.name].items() if wanted(k)}
            self._ext(pplan, acc, values)
        return acc

    def _ext(self, pplan: Any, acc: dict, values: dict) -> None:
        """Add external port inputs (``{port: value}``; Quantity or mantissa in the port unit)."""
        model = pplan.model
        for port, v in values.items():
            kind = model.inputs.require(port, pplan.name)
            m = to_mantissa(v, kind.unit) if isinstance(v, u.Quantity) else jnp.asarray(v)
            m = jnp.broadcast_to(m.astype(self.dtype), (pplan.pop.n,))
            acc[port] = acc[port] + m if port in acc else m

    def step(self, params: dict, state: dict, n: Any, trial: Any, ext: dict | None = None
             ) -> tuple[dict, dict]:
        """One network step (pure).

        Parameters
        ----------
        params : dict
            :meth:`read_params` pytree.
        state : dict
            :meth:`init_state` pytree.
        n : jax.Array
            Global step index (int32 scalar).
        trial : jax.Array
            Trial index folded into every population key (uint32 scalar).
        ext : dict, optional
            External inputs of this step, ``{population name: {input port: value}}``
            (Quantity, or mantissa in the port unit; ``(n,)`` or scalar), added to the
            projections' input of the port (learners feed data this way).

        Returns
        -------
        state : dict
            New state.
        outputs : dict
            ``{population index: {output port: mantissa}}`` of this step.
        """
        plan = self.plan
        if ext:
            unknown = set(ext) - {p.name for p in plan.populations}
            if unknown:
                raise KeyError(f'external inputs for unknown population(s) {sorted(unknown)}')
        n = jnp.asarray(n, jnp.int32)
        pops = dict(state['pops'])
        syn_state = dict(state['syn'])
        cur: dict[int, dict] = {}
        delivered: dict[int, jax.Array] = {}
        # arrival rings: read this step's slot and clear it in place (PostRing.take)
        arrived, arrivals = {}, dict(state['arrivals'])
        for pp in plan.projections:
            if pp.post_ring is not None:
                arrived[pp.name], arrivals[pp.name] = pp.post_ring.take(arrivals[pp.name], n)
        for a in plan.order:
            pplan = plan.populations[a]
            model = pplan.model
            ctx = self._ctx(a, n, trial)
            acc = self._gather(a, params, state, cur, n, ctx, delivered, syn_state, ext,
                               arrived=arrived,
                               skip=plan.deferred.get(a, frozenset()))
            inp = model.prepare_inputs({
                k: (_timed.with_weight_unit(v, model.inputs[k].unit) if isinstance(model.inputs[k], TimedEvent)
                    else from_mantissa(v, model.inputs[k].unit)) for k, v in acc.items()})
            old = pops[pplan.name]
            new, out = model.step(old, params['pops'][pplan.name], inp, ctx)
            pops[pplan.name] = Record(**{k: _like(new[k], old[k]) for k in old})
            cur[a] = {port: (_timed.payload_mantissa(v, self.dtype)
                             if isinstance(model.outputs[port], TimedEvent)
                             else to_mantissa(v, pplan.out_units[port]).astype(self.dtype))
                      for port, v in out.items()}

        # deferred latched ports (zero-delay cycles through latched ports, D-a): their
        # inputs only fill the latch, so they are gathered once every source updated
        for a, ports in plan.deferred.items():
            pplan = plan.populations[a]
            acc = self._gather(a, params, state, cur, n, self._ctx(a, n, trial), delivered,
                               syn_state, ext, only=ports, arrived=arrived)
            old = state['pops'][pplan.name]
            latched = dict(pops[pplan.name])
            for port in ports:
                kind = pplan.model.inputs[port]
                k = latch_name(port)
                v = from_mantissa(acc[port], kind.unit) if port in acc else kind.zeros(pplan.pop.n, self.dtype)
                # upstream: braintrace#184 (keep the latch's hidden->hidden edge)
                latched[k] = _like(_keep_dependency(v, old[k]), old[k])
            pops[pplan.name] = Record(**latched)

        # plasticity (reads the rings before this step's outputs are written)
        plastic = dict(state['plastic'])
        plastic_y: dict[str, jax.Array] = {}
        for pp in plan.projections:
            if pp.plastic is not None:
                plastic[pp.name], plastic_y[pp.name] = self._plastic(pp, params, state, cur, n)

        # emit
        rings = dict(state['rings'])
        for pplan in plan.populations:
            for port, ring in pplan.rings.items():
                k = f'{pplan.name}.{port}'
                rings[k] = ring.write(rings[k], n, cur[pplan.pop.index][port])
        for pp in plan.projections:
            if pp.post_ring is None:
                continue
            buf = arrivals[pp.name]  # slot n % L already cleared
            if pp.plastic is not None:
                arrivals[pp.name] = pp.post_ring.deposit(buf, n, pp.plastic.d_den, plastic_y[pp.name])
                continue
            payload = self._now_payload(pp, cur)
            for g, group in enumerate(pp.groups):
                if isinstance(group.storage, DelayScatter):  # one scatter for all delays (K1a-4)
                    buf = group.storage.deposit(buf, n, payload.astype(self.dtype),
                                                params['weights'][pp.name][g])
                elif group.delay >= 1:
                    buf = pp.post_ring.deposit(buf, n, group.delay, self._forward(pp, g, payload, params))
            arrivals[pp.name] = buf
        return {'pops': pops, 'rings': rings, 'arrivals': arrivals, 'syn': syn_state,
                'plastic': plastic}, cur

    # ------------------------------------------------------------- observe
    def observe(self, oplan: Any, pops: dict) -> dict:
        """Recorded values (mantissas in the observer's units) of one step."""
        out = {}
        for r in oplan.observer.recordables:
            unit = oplan.units[r]
            pieces = []
            for seg in oplan.observer.view.segments:
                name = self.plan.populations[seg.pop].name
                pieces.append(_take(to_mantissa(pops[name][r], unit), seg.local))
            out[r] = _concat(pieces, self.dtype)
        return out

    def rollout(self, params: dict, state: dict, n0: Any, trial: Any, n_steps: int, *,
                learn: Callable | None = None, every: int = 1) -> tuple:
        """``n_steps`` steps from global step ``n0`` with ``brainstate.transform.scan`` (pure).

        Parameters
        ----------
        params : dict
            :meth:`read_params` pytree.
        state : dict
            State pytree to start from (:meth:`init_state`, or any state a client
            reached).
        n0 : int or jax.Array
            Global index of the first step (time ``n0 * dt``; RNG streams).
        trial : jax.Array
            Trial index (uint32 scalar).
        n_steps : int
            Number of steps (static).
        learn : callable, optional
            Learn phase (architecture doc §8.2-6), run inside the compiled loop
            after the observe phase of every step ``n`` with ``(n + 1) % every ==
            0`` (global step count, so it lines up across consecutive rollouts):
            ``learn(params, state, n_next) -> (params, state)`` with ``n_next =
            n + 1``. It may change parameters (optimizer steps, weight
            normalisation) and the state (trial resets, learning-signal
            broadcast); the returned pytrees must keep the structure, units,
            dtypes and shapes of its inputs. ``State`` objects it writes (e.g. an
            optimizer's moments) are carried by the scan.
        every : int, default 1
            Interval of the learn phase in steps (>= 1).

        Returns
        -------
        state : dict
            Final state.
        recordings : dict
            ``{observer name: {recordable: (n_samples, n) mantissas}}``.
        params : dict
            Final parameters -- only when ``learn`` is given.

        Raises
        ------
        TypeError
            If ``learn`` is not callable or returns pytrees of another structure.
        ValueError
            If ``every`` is not an int >= 1.
        """
        if learn is not None and not callable(learn):
            raise TypeError(f'learn must be callable (params, state, n) -> (params, state), '
                            f'got {learn!r}')
        if isinstance(every, bool) or not isinstance(every, (int, np.integer)) or every < 1:
            raise ValueError(f'every must be an int >= 1, got {every!r}')
        every = int(every)
        dense = [o for o in self.plan.observers if o.observer.interval == 1]
        windowed = [o for o in self.plan.observers if o.observer.interval > 1]
        probe = self.observe  # dtypes of recordables from the initial state
        accs0 = {}
        for o in windowed:
            vals = probe(o, state['pops'])
            accs0[o.name] = {r: window_init(o.observer.n_samples(n_steps), v.shape[0], v.dtype)
                             for r, v in vals.items()}
        n0 = jnp.asarray(n0, jnp.int32)

        def apply(prm, st, n_next):
            new_p, new_s = learn(prm, st, n_next)
            return _cast_like(new_p, prm, 'params'), _cast_like(new_s, st, 'state')

        def learn_phase(prm, st, n_next):
            if every == 1:
                return apply(prm, st, n_next)
            return brainstate.transform.cond(n_next % every == 0, apply,
                                             lambda a, b, c: (a, b), prm, st, n_next)

        def body(carry, i):
            prm, st, accs = carry
            st, _ = self.step(prm, st, n0 + i, trial)
            ys = {o.name: self.observe(o, st['pops']) for o in dense}
            new_accs = {}
            for o in windowed:
                vals = self.observe(o, st['pops'])
                ob = o.observer
                new_accs[o.name] = {r: window_update(accs[o.name][r], i, ob.interval, vals[r],
                                                     r in ob.events) for r in vals}
            if learn is not None:
                prm, st = learn_phase(prm, st, n0 + i + 1)
            return (prm, st, new_accs), ys

        (params, state, accs), ys = brainstate.transform.scan(
            body, (params, state, accs0), jnp.arange(n_steps, dtype=jnp.int32))
        if learn is None:
            return state, {**ys, **accs}
        return state, {**ys, **accs}, params


class Runner:
    """Runs a program: :meth:`simulate`, :meth:`cont`, :meth:`step` (spec §4.8).

    The runner is one client of a :class:`Program` (decision D-c): it keeps the
    state and the global step between calls. Learners are other clients; they take
    ``(program, state0)`` and can start from :attr:`state` (e.g. after a warm-up).

    Parameters
    ----------
    program : Program or Plan
        The program to run (a plan is wrapped in ``Program(plan)``).
    batch : int, optional
        Number of trials (``vmap``); ``None`` for an unbatched run (= trial 0).
    rng : {'independent', 'common'}, default 'independent'
        Per-trial random numbers.

    Attributes
    ----------
    state : dict
        Current state pytree (leading trial axis when batched).
    n : int
        Global index of the next step.
    n_traces : int
        Number of times the rollout has been traced (each trace compiles once;
        repeated calls with the same duration must not add traces).
    trials : jax.Array
        Trial indices (uint32; ``()`` unbatched, ``(batch,)`` batched), passed to
        learners together with :attr:`state` and :attr:`n`.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.network import Network
        >>> from brainpy_state._kernel.models import dc_generator, iaf_psc_alpha
        >>> net = Network(dt=0.1 * u.ms)
        >>> dc = net.add(dc_generator, 1, params=dict(amplitude=500.0 * u.pA))
        >>> neu = net.add(iaf_psc_alpha, 1)
        >>> _ = net.connect(dc, neu, port='I')
        >>> rec = net.observe(neu, 'spike', interval=10.0 * u.ms)
        >>> run = net.compile().runner()
        >>> res = run.simulate(50.0 * u.ms)
        >>> res[rec]['spike'].shape
        (5, 1)
        >>> _ = run.cont(50.0 * u.ms); _ = run.cont(50.0 * u.ms)
        >>> run.n, run.n_traces
        (1500, 1)
    """

    def __init__(self, program: Program | Plan, batch: int | None = None,
                 rng: str = 'independent') -> None:
        trials = trial_indices(batch, rng)
        if isinstance(program, Plan):
            program = Program(program)
        if not isinstance(program, Program):
            raise TypeError(f'Runner needs a Program or a Plan, got {type(program).__name__}')
        self.program = program
        self.plan = program.plan
        self.batch = None if batch is None else int(batch)
        self.rng = rng
        init = self.program.init_state(self.batch)
        self._init = init
        self._trials = trials
        self.state = init
        self.n = 0
        self.n_traces = 0
        self._structure = self.program.structure()
        self._rollout = brainstate.transform.jit(self._run, static_argnums=(4,),
                                                 static_argnames=('learn', 'every'))
        self._one_step = brainstate.transform.jit(self._step)

    # ------------------------------------------------------------ compiled
    def _vmap(self, f):
        if self.batch is None:
            return f
        return jax.vmap(f, in_axes=(None, 0, None, 0))

    def _run(self, params, state, n0, trials, n_steps, structure, learn=None, every=1):
        self.n_traces += 1
        f = functools.partial(self.program.rollout, n_steps=n_steps, learn=learn, every=every)
        with self.program.bound(structure):
            return self._vmap(f)(params, state, n0, trials)

    def _step(self, params, state, n, trials, structure):
        with self.program.bound(structure):
            return self._vmap(self.program.step)(params, state, n, trials)

    # -------------------------------------------------------------- public
    @property
    def trials(self) -> jax.Array:
        """Trial indices of the run (see the class docstring)."""
        return self._trials

    @property
    def t(self) -> Any:
        """Time of the next step's start."""
        return from_mantissa(self.n * self.program.dt_ms, u.ms)

    def reset(self) -> None:
        """Back to the initial state at ``t = 0``."""
        self.state = self._init
        self.n = 0

    def simulate(self, duration: Any) -> Results:
        """Reset, then run ``duration`` (see :meth:`cont`)."""
        self.reset()
        return self.cont(duration)

    def cont(self, duration: Any, *, learn: Callable | None = None, every: int = 1) -> Results:
        """Continue for ``duration`` from the current state and time.

        Parameters
        ----------
        duration : Quantity
            Whole multiple of ``dt`` and of every observer interval.
        learn : callable, optional
            Learn phase inside the compiled rollout (see :meth:`Program.rollout`):
            ``learn(params, state, n_next) -> (params, state)`` after every step
            ``n`` with ``(n + 1) % every == 0``. The final parameters are written
            back to the plan's parameter states (:meth:`Program.write_params`).
            Unbatched runs only.
        every : int, default 1
            Interval of the learn phase in steps.

        Returns
        -------
        Results

        Raises
        ------
        NotImplementedError
            For ``learn`` on a batched runner (per-trial vs shared parameters is
            a K1-L decision).
        """
        n_steps = duration_steps(duration, self.program.dt)
        for o in self.plan.observers:
            o.observer.n_samples(n_steps)
        if learn is not None and self.batch is not None:
            raise NotImplementedError('a learn phase on a batched runner is a K1-L item; run the '
                                      'learn phase on an unbatched runner')
        params = self.program.read_params()
        n0 = jnp.asarray(self.n, jnp.int32)
        out = self._rollout(params, self.state, n0, self._trials, n_steps, self._structure,
                            learn=learn, every=every)
        if learn is None:
            state, rec = out
        else:
            state, rec, params = out
            self.program.write_params(params)
        start = self.n
        self.state = state
        self.n += n_steps
        return self._results(rec, start, n_steps)

    def step(self) -> dict:
        """Advance one step; returns ``{population: {output port: value}}`` (no observers)."""
        params = self.program.read_params()
        n = jnp.asarray(self.n, jnp.int32)
        state, cur = self._one_step(params, self.state, n, self._trials, self._structure)
        self.state = state
        self.n += 1
        out = {}
        for pplan in self.plan.populations:
            out[pplan.name] = {port: (_timed.payload_from_mantissa(v) if isinstance(v, dict)
                                      else from_mantissa(v, pplan.out_units[port]))
                               for port, v in cur[pplan.pop.index].items()}
        return out

    def edges(self, key: Any, trial: int = 0) -> Any:
        """Realised edges of a projection with the weights of the **current** run state.

        Plastic projections report their observable weights (the committed ones
        under lazy timing); other projections are as :meth:`Plan.edges`.

        Parameters
        ----------
        key : Projection, str or int
            The projection.
        trial : int, default 0
            Trial of a batched run.

        Returns
        -------
        EdgeSet
        """
        pp = self.plan.projection(key)
        if pp.plastic is None or not pp.groups:
            return self.plan.edges(key)
        st = self.state['plastic'][pp.name]
        if self.batch is not None:
            st = jax.tree.map(lambda x: x[trial], st)
        w = pp.groups[0].storage.weight_to_edges(pp.plastic.weight(st))
        unit = pp.weight_unit
        return pp.edges.set(weight=w if unit.is_unitless else u.Quantity(w, unit=unit))

    def _results(self, rec: dict, start: int, n_steps: int) -> Results:
        dt = self.program.dt_ms
        recs = {}
        for o in self.plan.observers:
            ob = o.observer
            k = ob.interval
            s = n_steps // k
            times = from_mantissa(np.asarray((start + (np.arange(s) + 1) * k) * dt), u.ms)
            data = {r: from_mantissa(v, o.units[r]) for r, v in rec[ob.name].items()}
            cols: dict[str, list] = {}
            for seg in ob.view.segments:
                name = self.plan.populations[seg.pop].name
                cols.setdefault(name, []).extend(range(seg.pos.start, seg.pos.stop))
            recs[ob.name] = Recording(observer=ob, times=times, data=data, gids=ob.view.gids,
                                      columns={k2: np.asarray(v, np.int64) for k2, v in cols.items()},
                                      batched=self.batch is not None)
        return Results(t_start=from_mantissa(start * dt, u.ms),
                       t_stop=from_mantissa((start + n_steps) * dt, u.ms),
                       n_steps=n_steps, recordings=recs)
