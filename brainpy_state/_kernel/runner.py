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

A :class:`Program` holds this as **pure** functions of explicit pytrees
(``params``, ``state``); the :class:`Runner` keeps the state between calls and runs
``brainstate.transform.scan`` inside one ``brainstate.transform.jit``-compiled
function. The step index and the parameter values are traced arguments and the
number of steps is the only static one, so repeated ``cont()`` calls of the same
duration (and parameter changes) reuse the compiled program.

RNG (spec §4.5): population ``uid`` receives ``fold_in(population_key(seed, uid, n),
b)`` at step ``n`` of trial ``b`` (``b = 0`` for an unbatched run and for every trial
under ``rng='common'``), i.e. ``trial_keys(population_key(...), batch, mode)[b]``.
Streams depend only on the global step, so ``cont()`` continues them.

Batching runs ``jax.vmap`` over the state pytree and the trial index; parameters and
weights are shared by the trials.
"""
from __future__ import annotations

import functools
from typing import Any

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.compiler import Plan, ProjectionPlan
from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.observers import Recording, Results, window_init, window_update
from brainpy_state._kernel.params import Record, read_states
from brainpy_state._kernel.ports import TimedEvent
from brainpy_state._kernel import timed as _timed
from brainpy_state._kernel.rng import RNG_MODES, population_key
from brainpy_state._kernel.units import check_unit, from_mantissa, to_mantissa, unit_of

__all__ = ['Program', 'Runner', 'duration_steps']


def duration_steps(duration: Any, dt: Any) -> int:
    """Number of steps of ``duration`` (must be a non-negative whole multiple of ``dt``)."""
    check_unit(duration, u.ms, 'duration')
    r = float(u.Quantity(duration).to_decimal(u.ms)) / float(u.Quantity(dt).to_decimal(u.ms))
    n = int(round(r))
    if n < 0 or abs(r - n) > 1e-6 * max(1.0, abs(r)):
        raise ValueError(f'duration {duration} must be a non-negative whole multiple of dt={dt}')
    return n


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
    """Pure step / rollout functions of a plan (static structure in closures).

    Parameters
    ----------
    plan : Plan
        Compiled network.
    etrace : bool, default False
        Apply trainable (``ParamState``) weights through the storages' braintrace
        primitives (:meth:`~brainpy_state._kernel.connectivity.storage.Storage.forward_etrace`)
        so that ``braintrace.compile`` sees them (online learners, T9).
    """

    def __init__(self, plan: Plan, etrace: bool = False) -> None:
        self.plan = plan
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
        self._syn_by_pop: dict[int, list] = {}
        for gi, g in enumerate(plan.synapses):
            self._syn_by_pop.setdefault(g.pop, []).append(gi)

    # ------------------------------------------------------------- pytrees
    def init_state(self) -> dict:
        """Initial state pytree (one trial): model states and zero buffers."""
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
        prj = pp.proj
        x = self._delayed(prj.pre, prj.source_port, pl.d_ax, state['rings'], cur, n)
        q = self._delayed(prj.post, prj.plasticity.post_port, pl.d_den, state['rings'], cur, n)
        t = n.astype(self.dtype) * self.dt_ms
        new, w_eff = pl.step(state['plastic'][pp.name], params['plasticity'][pp.name], x, q, t,
                             self.dt_ms)
        y = pp.groups[0].storage.forward(x.astype(self.dtype), w_eff, pp.path)
        return new, u.get_mantissa(y)

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
            y = group.storage.forward(payload.astype(self.dtype), w, pp.path)
        return u.get_mantissa(y)

    def _deliver_timed(self, pp: ProjectionPlan, params: dict, state: dict, cur: dict, n: Any) -> Any:
        g = pp.groups[0]
        w = u.get_mantissa(g.storage.weight_to_edges(params['weights'][pp.name][0]))
        rings, objs = {}, {}
        for s in pp.timed.segments:
            pplan = self.plan.populations[s.pop]
            ring = pplan.rings.get(s.port)
            if ring is not None:
                objs[(s.pop, s.port)] = ring
                rings[(s.pop, s.port)] = state['rings'][f'{pplan.name}.{s.port}']
        return pp.timed.deliver(w, rings, objs, cur, n, self.dtype)

    def _deliver(self, pp: ProjectionPlan, params: dict, state: dict, cur: dict, n: Any) -> Any:
        if pp.placement == 'timed':
            return self._deliver_timed(pp, params, state, cur, n)
        if pp.placement == 'pre':
            return self._forward(pp, 0, self._pre_payload(pp, state['rings'], cur, n), params)
        y = (pp.post_ring.read(state['arrivals'][pp.name], n) if pp.post_ring is not None
             else jnp.zeros((pp.edges.n_post,), self.dtype))
        for g, group in enumerate(pp.groups):
            if group.delay == 0:
                y = y + self._forward(pp, g, self._now_payload(pp, cur), params)
        return y

    # ---------------------------------------------------------------- step
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
        for a in plan.order:
            pplan = plan.populations[a]
            model = pplan.model
            acc: dict[Any, jax.Array] = {}
            key = jax.random.fold_in(population_key(self.seed, pplan.pop.uid, n), trial)
            ctx = StepContext(t=from_mantissa(n.astype(self.dtype) * self.dt_ms, u.ms), dt=self.dt,
                              step=n, key=key)
            for j in pplan.incoming:
                pp = plan.projections[j]
                if j not in delivered:
                    delivered[j] = self._deliver(pp, params, state, cur, n)
                y = delivered[j]
                gi = plan.synapse_route.get((j, a))
                port = pp.proj.port if gi is None else ('syn', gi)
                kind = model.inputs[port] if gi is None else None
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
                x = acc.pop(('syn', gi), None)
                if x is None:  # pragma: no cover - every member delivers into its group
                    x = jnp.zeros((pplan.pop.n,), self.dtype)
                syn_state[g.key], out = g.synapse.step(syn_state[g.key], x, ctx)
                acc[g.port] = out if g.port not in acc else acc[g.port] + out
            if ext is not None and pplan.name in ext:
                self._ext(pplan, acc, ext[pplan.name])
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
        arrivals = dict(state['arrivals'])
        for pp in plan.projections:
            if pp.post_ring is None:
                continue
            buf = pp.post_ring.clear(arrivals[pp.name], n)
            if pp.plastic is not None:
                arrivals[pp.name] = pp.post_ring.deposit(buf, n, pp.plastic.d_den, plastic_y[pp.name])
                continue
            payload = self._now_payload(pp, cur)
            for g, group in enumerate(pp.groups):
                if group.delay >= 1:
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

    def rollout(self, params: dict, state: dict, n0: Any, trial: Any, n_steps: int) -> tuple[dict, dict]:
        """``n_steps`` steps from global step ``n0`` with ``brainstate.transform.scan`` (pure).

        Returns
        -------
        state : dict
            Final state.
        recordings : dict
            ``{observer name: {recordable: (n_samples, n) mantissas}}``.
        """
        dense = [o for o in self.plan.observers if o.observer.interval == 1]
        windowed = [o for o in self.plan.observers if o.observer.interval > 1]
        probe = self.observe  # dtypes of recordables from the initial state
        accs0 = {}
        for o in windowed:
            vals = probe(o, state['pops'])
            accs0[o.name] = {r: window_init(o.observer.n_samples(n_steps), v.shape[0], v.dtype)
                             for r, v in vals.items()}
        n0 = jnp.asarray(n0, jnp.int32)

        def body(carry, i):
            st, accs = carry
            st, _ = self.step(params, st, n0 + i, trial)
            ys = {o.name: self.observe(o, st['pops']) for o in dense}
            new_accs = {}
            for o in windowed:
                vals = self.observe(o, st['pops'])
                ob = o.observer
                new_accs[o.name] = {r: window_update(accs[o.name][r], i, ob.interval, vals[r],
                                                     r in ob.events) for r in vals}
            return (st, new_accs), ys

        (state, accs), ys = brainstate.transform.scan(body, (state, accs0),
                                                      jnp.arange(n_steps, dtype=jnp.int32))
        return state, {**ys, **accs}


class Runner:
    """Runs a plan: :meth:`simulate`, :meth:`cont`, :meth:`step` (spec §4.8).

    Parameters
    ----------
    plan : Plan
        Compiled network.
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
    learner : object or None
        Learner attached by ``plan.runner(learner=...)`` (e.g.
        :class:`~brainpy_state._kernel.learn.bptt.BPTT`).

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

    def __init__(self, plan: Plan, batch: int | None = None, rng: str = 'independent') -> None:
        if rng not in RNG_MODES:
            raise ValueError(f'rng must be one of {RNG_MODES}, got {rng!r}')
        if batch is not None and (isinstance(batch, bool) or not isinstance(batch, (int, np.integer))
                                  or batch < 1):
            raise ValueError(f'batch must be a positive int or None, got {batch!r}')
        self.plan = plan
        self.program = Program(plan)
        self.batch = None if batch is None else int(batch)
        self.rng = rng
        init = self.program.init_state()
        if self.batch is not None:
            init = jax.tree.map(lambda x: jnp.broadcast_to(x, (self.batch,) + jnp.shape(x)), init)
            trials = (jnp.arange(self.batch, dtype=jnp.uint32) if rng == 'independent'
                      else jnp.zeros((self.batch,), jnp.uint32))
        else:
            trials = jnp.zeros((), jnp.uint32)
        self._init = init
        self._trials = trials
        self.state = init
        self.n = 0
        self.n_traces = 0
        self.learner = None
        self._rollout = brainstate.transform.jit(self._run, static_argnums=(4,))
        self._one_step = brainstate.transform.jit(self._step)

    # ------------------------------------------------------------ compiled
    def _vmap(self, f):
        if self.batch is None:
            return f
        return jax.vmap(f, in_axes=(None, 0, None, 0))

    def _run(self, params, state, n0, trials, n_steps):
        self.n_traces += 1
        f = functools.partial(self.program.rollout, n_steps=n_steps)
        return self._vmap(f)(params, state, n0, trials)

    def _step(self, params, state, n, trials):
        return self._vmap(self.program.step)(params, state, n, trials)

    # -------------------------------------------------------------- public
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

    def cont(self, duration: Any) -> Results:
        """Continue for ``duration`` from the current state and time.

        Parameters
        ----------
        duration : Quantity
            Whole multiple of ``dt`` and of every observer interval.

        Returns
        -------
        Results
        """
        n_steps = duration_steps(duration, self.program.dt)
        for o in self.plan.observers:
            o.observer.n_samples(n_steps)
        params = self.program.read_params()
        n0 = jnp.asarray(self.n, jnp.int32)
        state, rec = self._rollout(params, self.state, n0, self._trials, n_steps)
        start = self.n
        self.state = state
        self.n += n_steps
        return self._results(rec, start, n_steps)

    def step(self) -> dict:
        """Advance one step; returns ``{population: {output port: value}}`` (no observers)."""
        params = self.program.read_params()
        n = jnp.asarray(self.n, jnp.int32)
        state, cur = self._one_step(params, self.state, n, self._trials)
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
