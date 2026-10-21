# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Compiler: ``Network`` -> :class:`Plan` (K0 spec §4.8, architecture doc §7 L2).

``compile_network(net, precision)``:

1. **models** -- every population's model is instantiated at the plan precision with
   its init key ``fold_in(object_key(seed, uid), INIT_STREAM)``;
2. **edges** -- every projection's rule is built with ``object_key(seed, uid)``
   (storage/batch independent, spec §4.5); weights and delays are drawn from their own
   sub-streams (``WEIGHT_STREAM`` / ``DELAY_STREAM``) and delays are rounded to steps
   the NEST way (:func:`~brainpy_state._kernel.delays.delay_steps`);
3. **delays and buffers** -- each projection's delays are classified (scalar,
   per-source, per-edge) and placed in a pre-side ring (shared per source port) or a
   post-side arrival ring with the edges grouped by delay (:mod:`.delays`);
4. **storage and payload path** -- one :func:`make_storage` per edge group; the
   payload path is ``'binary'`` only for sources with ``max_multiplicity == 1``
   (:func:`payload_path`);
5. **schedule** -- populations are updated in insertion order, except that every
   zero-delay edge ``A -> B`` forces ``A`` before ``B`` (topological order, decision
   D4a); a zero-delay cycle is a :class:`ScheduleError`;
6. **plasticity** -- a plastic projection (K0 T6) keeps per-edge CSR weights in the
   runner state; its scalar delay is split into axonal and dendritic steps
   (:class:`~brainpy_state._kernel.plasticity.PlasticPlan`): the pre side is read
   ``d_ax`` steps late, the post spikes ``d_den`` steps late (a pre-side ring on the
   post population's spike port) and the delivery goes through a post-side arrival
   ring ``d_den`` steps ahead;
7. **synapse merge pass** -- synapse states are grouped per (target population,
   port, merge key) (:func:`~brainpy_state._kernel.synapse.plan_synapse_groups`);
8. **capabilities** -- models that are not ``jit_safe`` are rejected; the
   differentiable / braintrace summary is reported by :meth:`Plan.describe`.

The plan is a ``brainstate.nn.Module``: ``plan.states(brainstate.ParamState)``
returns every trainable parameter and weight, which is what learners (T9) optimise.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.connectivity.edgeset import EdgeSet
from brainpy_state._kernel.connectivity.storage import Storage, make_storage, payload_path
from brainpy_state._kernel.delays import DelayLayout, PostRing, PreRing, classify_delays, delay_steps
from brainpy_state._kernel.model import NeuronModel
from brainpy_state._kernel.network import Network, Population, Projection, Segment
from brainpy_state._kernel.observers import Observer
from brainpy_state._kernel.params import ConstParamState, Distribution
from brainpy_state._kernel.plasticity import PlasticPlan
from brainpy_state._kernel.ports import Event, TimedEvent
from brainpy_state._kernel.rng import block_key, object_key
from brainpy_state._kernel.synapse import plan_synapse_groups
from brainpy_state._kernel.timed import TimedDelivery, TimedRing, check_capacities
from brainpy_state._kernel.units import resolve_dtype, to_mantissa, unit_of

__all__ = [
    'INIT_STREAM',
    'WEIGHT_STREAM',
    'DELAY_STREAM',
    'ScheduleError',
    'PopulationPlan',
    'EdgeGroup',
    'ProjectionPlan',
    'ObserverPlan',
    'Plan',
    'compile_network',
    'schedule',
]

#: Reserved ``gid_block`` counters of the per-object sub-streams (never a real block).
INIT_STREAM = 0xFFFFFFFE
WEIGHT_STREAM = 0xFFFFFFFD
DELAY_STREAM = 0xFFFFFFFC


class ScheduleError(ValueError):
    """The zero-delay graph has a cycle (a cycle needs a delay of at least one step)."""


@dataclass
class PopulationPlan:
    """Compiled population.

    Attributes
    ----------
    pop : Population
        Declaration.
    model : NeuronModel
        Instantiated model (owns parameter and initial-state ``State`` objects).
    rings : dict of str to PreRing
        Pre-side rings per output port (only ports read with a delay >= 1).
    incoming : list of int
        Indices of the projections delivering into this population.
    out_units : dict of str to brainunit.Unit
        Unit of each output port's payload (dimensionless for events).
    """

    pop: Population
    model: NeuronModel
    rings: dict = field(default_factory=dict)
    incoming: list = field(default_factory=list)
    out_units: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.pop.name


@dataclass
class EdgeGroup:
    """Edges of a projection that share one delay buffer access and one storage.

    Attributes
    ----------
    delay : int or None
        Delay in steps; ``None`` for a per-source pre-side group.
    storage : Storage
        Edge storage of the group.
    edges : numpy.ndarray
        Positions of the group's edges in the projection's :class:`EdgeSet`.
    weight : brainstate.State
        Native-layout weights (``ParamState`` if trainable, else ``ConstParamState``).
    """

    delay: Any
    storage: Storage
    edges: np.ndarray
    weight: Any


@dataclass
class ProjectionPlan:
    """Compiled projection.

    Attributes
    ----------
    proj : Projection
        Declaration.
    edges : EdgeSet
        Realised edges (initial per-edge weights and delays in ``EdgeSet`` order).
    layout : DelayLayout
        Classified delays.
    placement : {'pre', 'post', 'timed', 'none'}
        Delay-buffer placement (``'none'``: no edges, nothing is delivered;
        ``'timed'``: per-edge gather from the source's :class:`TimedRing`, K0 T7).
    path : {'binary', 'float', 'timed'}
        Payload path (``'timed'``: precisely timed events, K0 T7).
    groups : list of EdgeGroup
        One group for pre-side placement; one per delay value for post-side.
    post_ring : PostRing or None
        Arrival ring (post-side placement with delays >= 1).
    weight_unit : brainunit.Unit
        Unit of the stored weights (= target unit per source unit).
    plastic : PlasticPlan or None
        Compiled plasticity (K0 T6); the group weight then holds the *initial*
        weights and the current ones live in the runner state.
    timed : TimedDelivery or None
        Delivery of a projection from a ``timed_event`` source (``placement == 'timed'``).
    """

    proj: Projection
    edges: EdgeSet
    layout: DelayLayout
    placement: str
    path: str
    groups: list
    post_ring: PostRing | None
    weight_unit: Any
    plastic: PlasticPlan | None = None
    timed: TimedDelivery | None = None

    @property
    def name(self) -> str:
        return self.proj.name

    @property
    def n_edges(self) -> int:
        return self.edges.n_edges


@dataclass
class ObserverPlan:
    """Compiled observer: the declaration plus the unit of every recordable."""

    observer: Observer
    units: dict

    @property
    def name(self) -> str:
        return self.observer.name


def schedule(n_pops: int, zero_edges: set) -> list:
    """Update order: insertion order, constrained by zero-delay edges.

    Parameters
    ----------
    n_pops : int
        Number of populations.
    zero_edges : set of (int, int)
        ``(a, b)``: ``a`` feeds ``b`` with a zero delay (``a`` must update first).

    Returns
    -------
    list of int
        Population indices (Kahn's algorithm, smallest index first).

    Raises
    ------
    ScheduleError
        On a zero-delay cycle (including a zero-delay self-connection).
    """
    succ: dict[int, set] = {i: set() for i in range(n_pops)}
    indeg = [0] * n_pops
    for a, b in zero_edges:
        if a == b:
            raise ScheduleError(f'population {a} feeds itself with a zero delay')
        if b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1
    heap = [i for i in range(n_pops) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        a = heapq.heappop(heap)
        order.append(a)
        for b in sorted(succ[a]):
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(heap, b)
    if len(order) != n_pops:
        raise ScheduleError(f'zero-delay cycle among populations '
                            f'{sorted(set(range(n_pops)) - set(order))}')
    return order


def _state(value: Any, trainable: bool) -> brainstate.State:
    return brainstate.ParamState(value) if trainable else ConstParamState(value)


def _fmt_bytes(n: int) -> str:
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if n < 1024 or unit == 'GiB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} GiB'  # pragma: no cover


class Plan(brainstate.nn.Module):
    """A compiled network: models, edges, storages, buffers and schedule.

    Built by :meth:`Network.compile`; run with :meth:`runner`.

    Attributes
    ----------
    net : Network
        Source graph.
    precision : str
        Float tier.
    dtype : numpy.dtype
        Float dtype.
    populations : list of PopulationPlan
        In ``net.populations`` order.
    projections : list of ProjectionPlan
        In ``net.projections`` order.
    observers : list of ObserverPlan
        In ``net.observers`` order.
    order : list of int
        Update order (population indices).
    models : list of NeuronModel
        The models (registered as child modules for ``states()``).
    weights : dict of str to list of brainstate.State
        Weight states per projection (one per edge group); initial weights for
        plastic projections.
    plastic_params : dict of str to dict of brainstate.State
        Rule parameter states per plastic projection.
    synapses : list of SynapseGroup
        Synapse states after the merge pass.
    synapse_route : dict
        ``(projection index, population index) -> index into synapses``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.connectivity import fixed_indegree
        >>> from brainpy_state._kernel.models import LIF
        >>> from brainpy_state._kernel.network import Network
        >>> net = Network(dt=0.1 * u.ms)
        >>> A = net.add(LIF, 20, name='A')
        >>> p = net.connect(A, A, fixed_indegree(5), port='delta', weight=0.1 * u.mV,
        ...                 delay=[0.1, 0.3] * 50 * u.ms)
        >>> plan = net.compile()
        >>> pp = plan.projection(p)
        >>> pp.placement, pp.layout.kind, [g.delay for g in pp.groups]
        ('post', 'per_edge', [1, 3])
        >>> plan.edges(p).n_edges
        100
        >>> 'post-side ring' in plan.describe(print_=False)
        True
    """

    def __init__(self, net: Network, precision: str) -> None:
        super().__init__()
        self.net = net
        self.precision = resolve_dtype(precision).name
        self.dtype = resolve_dtype(precision)
        self.populations: list[PopulationPlan] = []
        self.projections: list[ProjectionPlan] = []
        self.observers: list[ObserverPlan] = []
        self.order: list[int] = []
        self.models: list[NeuronModel] = []
        self.weights: dict[str, list] = {}
        self.plastic_params: dict[str, dict] = {}
        self.synapses: list = []
        self.synapse_route: dict = {}
        self.merge = True

    # ----------------------------------------------------------------- access
    @property
    def dt(self) -> Any:
        return self.net.dt

    def population(self, key: Any) -> PopulationPlan:
        """Population plan by name, index or view of a whole population."""
        if isinstance(key, str):
            for p in self.populations:
                if p.name == key:
                    return p
            raise KeyError(f'no population {key!r}')
        return self.populations[int(key)]

    def projection(self, key: Any) -> ProjectionPlan:
        """Projection plan by name, index or ``Projection``."""
        if isinstance(key, Projection):
            return self.projections[key.index]
        if isinstance(key, str):
            for p in self.projections:
                if p.name == key:
                    return p
            raise KeyError(f'no projection {key!r}')
        return self.projections[int(key)]

    def edges(self, key: Any) -> EdgeSet:
        """Realised edges of a projection with its **current** per-edge weights.

        For a plastic projection these are the initial weights (the evolving ones
        live in the runner state: ``Runner.edges``).

        Parameters
        ----------
        key : Projection, str or int
            The projection.

        Returns
        -------
        EdgeSet
            Edges in canonical order; ``weight`` read back from the weight states,
            ``delay`` the requested (unrounded) per-edge delays.
        """
        pp = self.projection(key)
        es = pp.edges
        if not pp.groups:
            return es
        m, unit = None, pp.weight_unit
        for g in pp.groups:
            w = to_mantissa(g.storage.weight_to_edges(g.weight.value), unit)
            if m is None:
                m = jnp.zeros((es.n_edges,), w.dtype)
            m = m.at[jnp.asarray(g.edges)].set(w)
        return es.set(weight=m if unit.is_unitless else u.Quantity(m, unit=unit))

    def memory(self) -> dict:
        """Device memory (bytes, one trial) of storages, buffers, synapse and plastic state."""
        item = self.dtype.itemsize
        storage = sum(g.storage.nbytes(item) for p in self.projections for g in p.groups)
        pre = sum(r.nbytes() for p in self.populations for r in p.rings.values())
        post = sum(p.post_ring.nbytes() for p in self.projections if p.post_ring is not None)
        syn = sum(self.populations[g.pop].pop.n * item for g in self.synapses)
        plastic = sum(p.plastic.nbytes() for p in self.projections if p.plastic is not None)
        return {'storage': storage, 'pre_rings': pre, 'post_rings': post, 'synapses': syn,
                'plastic': plastic, 'total': storage + pre + post + syn + plastic}

    # ---------------------------------------------------------------- running
    def runner(self, batch: int | None = None, rng: str = 'independent', learner: Any = None):
        """Create a :class:`~brainpy_state._kernel.runner.Runner`.

        Parameters
        ----------
        batch : int, optional
            Number of trials run in parallel with ``vmap`` (``None``: unbatched,
            identical to trial 0 of any batch).
        rng : {'independent', 'common'}, default 'independent'
            Per-trial random numbers (spec §4.5).
        learner : optional
            A learner with an ``attach(runner)`` method
            (:class:`~brainpy_state._kernel.learn.bptt.BPTT`,
            :class:`~brainpy_state._kernel.learn.etrace.Online`); it validates the
            plan (capabilities, braintrace rules) and binds to the runner.

        Returns
        -------
        Runner

        Raises
        ------
        TypeError
            If ``learner`` has no ``attach`` method.
        """
        if learner is not None and not callable(getattr(learner, 'attach', None)):
            raise TypeError(f'learner must provide attach(runner) (BPTT, Online), '
                            f'got {type(learner).__name__}')
        from brainpy_state._kernel.runner import Runner
        run = Runner(self, batch=batch, rng=rng)
        if learner is not None:
            learner.attach(run)
            run.learner = learner
        return run

    # --------------------------------------------------------------- describe
    def describe(self, print_: bool = True) -> str:
        """Human-readable summary of the plan.

        Sections: populations and schedule, projections (edges, storage, payload
        path, delay layout and buffer), buffers and memory, merges, capabilities.

        Parameters
        ----------
        print_ : bool, default True
            Also print the summary.

        Returns
        -------
        str
        """
        dt_ms = float(u.Quantity(self.dt).to_decimal(u.ms))
        L = [f'Plan: precision={self.precision}, dt={dt_ms:g} ms, seed={self.net.seed}, '
             f'{self.net.n_neurons} neurons in {len(self.populations)} populations, '
             f'{len(self.projections)} projections, {len(self.observers)} observers']
        L.append('Schedule (update order per step: deliver -> update; then emit -> observe):')
        L.append('  ' + ' -> '.join(self.populations[i].name for i in self.order) if self.order
                 else '  (empty)')
        zero = [p.name for p in self.projections if p.placement != 'none' and p.layout.has_zero]
        L.append(f'  zero-delay projections (same-step delivery): {", ".join(zero) or "none"}')
        L.append('Populations:')
        for p in self.populations:
            g0, n = p.pop.gid_start, p.pop.n
            L.append(f'  {p.name}: {p.pop.model.__name__} n={n} gids=[{g0}, {g0 + n}) uid={p.pop.uid}')
        L.append('Projections:')
        for pp in self.projections:
            pr = pp.proj
            lay = pp.layout
            if lay.kind == 'scalar':
                d = f'scalar {lay.min} steps'
            else:
                d = f'{lay.kind.replace("_", "-")} {lay.min}..{lay.max} steps'
            st = ', '.join(f'{g.storage.kind}' + (f'@d={g.delay}' if pp.placement == 'post' else '')
                           for g in pp.groups) or 'none'
            L.append(f'  {pp.name}: {pr.pre.label}.{pr.source_port} -> {pr.post.label}.{pr.port} '
                     f'rule={type(pr.rule).__name__} edges={pp.n_edges} storage=[{st}] '
                     f'payload={pp.path} delay={d} buffer={pp.placement}-side'
                     + (' trainable' if pr.trainable else '')
                     + (f' synapse={pr.synapse.describe()}' if pr.synapse is not None else '')
                     + (f' plasticity={pp.plastic.describe()}' if pp.plastic is not None else ''))
        L.append('Buffers (per trial):')
        for p in self.populations:
            for port, r in p.rings.items():
                cap = f' x {r.capacity} (offsets, weights)' if isinstance(r, TimedRing) else ''
                L.append(f'  pre-side ring {p.name}.{port}: {r.length} x {r.n}{cap} {r.dtype.name} '
                         f'= {_fmt_bytes(r.nbytes())}')
        for pp in self.projections:
            if pp.post_ring is not None:
                r = pp.post_ring
                L.append(f'  post-side ring {pp.name}: {r.length} x {r.n} {r.dtype.name} '
                         f'= {_fmt_bytes(r.nbytes())}')
        mem = self.memory()
        L.append(f'Memory: storages {_fmt_bytes(mem["storage"])}, pre-side rings '
                 f'{_fmt_bytes(mem["pre_rings"])}, post-side rings {_fmt_bytes(mem["post_rings"])}, '
                 f'synapse states {_fmt_bytes(mem["synapses"])}, plastic state '
                 f'{_fmt_bytes(mem["plastic"])}, total {_fmt_bytes(mem["total"])}')
        L.extend(self._describe_merges())
        L.append('Observers:')
        for o in self.observers:
            ob = o.observer
            L.append(f'  {ob.name}: {ob.view.label} {list(ob.recordables)} every {ob.interval} step(s)')
        L.append('Capabilities:')
        neurons = [p for p in self.populations if not p.pop.model.capabilities.differentiable]
        for p in self.populations:
            c = p.pop.model.capabilities
            L.append(f'  {p.name}: jit_safe={c.jit_safe} differentiable={c.differentiable} '
                     f'etrace={c.etrace} precise_spike_times={c.precise_spike_times}')
        trainable = len(self.states(brainstate.ParamState))
        L.append(f'  BPTT: differentiable except {[p.name for p in neurons] or "nothing"} '
                 f'(non-differentiable populations act as inputs); {trainable} ParamState(s)')
        L.append('  braintrace: storages route through braintrace primitives (forward_etrace); '
                 'rules checked by learn.etrace (check_plan / check_etrace)')
        text = '\n'.join(L)
        if print_:
            print(text)
        return text

    def _describe_merges(self) -> list:
        merged = [g for g in self.synapses if g.merged]
        n_pairs = sum(len(g.members) for g in self.synapses)
        if not self.synapses:
            return ['Merges: none (no synapse models)']
        L = [f'Merges: {n_pairs} synapse target(s) -> {len(self.synapses)} state(s)'
             + ('' if self.merge else ' (merge pass disabled)')]
        for g in merged:
            pop = self.populations[g.pop].name
            names = ', '.join(self.projections[j].name for j in g.members)
            L.append(f'  {pop}.{g.port} <- [{names}] {g.synapse.describe()}: one state')
        if not merged:
            L.append('  none merged')
        return L

    def __repr__(self) -> str:
        return (f'Plan(precision={self.precision}, populations={len(self.populations)}, '
                f'projections={len(self.projections)})')


# ---------------------------------------------------------------------------
# compilation
# ---------------------------------------------------------------------------

def _instantiate(net: Network, pop: Population, precision: str) -> NeuronModel:
    key = block_key(object_key(net.seed, pop.uid), INIT_STREAM)
    return pop.model(pop.n, pop.params or None, pop.init or None, precision=precision, key=key,
                     name=pop.name, **pop.kwargs)


def _edge_weights(proj: Projection, es: EdgeSet, okey: Any, dtype: Any) -> Any:
    w = proj.weight
    k = block_key(okey, WEIGHT_STREAM)
    try:
        es = es.with_weight(w, key=k if isinstance(w, Distribution) else None, dtype=dtype)
    except ValueError as e:
        raise ValueError(f'{proj.name}: {e} (the rule realised {es.n_edges} edges)') from None
    m = to_mantissa(es.weight, proj.weight_unit, dtype)
    return m if proj.weight_unit.is_unitless else u.Quantity(m, unit=proj.weight_unit)


def _edge_delays(net: Network, proj: Projection, es: EdgeSet, okey: Any) -> tuple:
    d = proj.delay
    if isinstance(d, Distribution):
        q = d.sample(block_key(okey, DELAY_STREAM), (es.n_edges,))
    else:
        m = np.asarray(u.get_mantissa(d))
        if m.ndim == 1 and m.shape != (es.n_edges,):
            raise ValueError(f'{proj.name}: per-edge delays have shape {m.shape} but the rule '
                             f'realised {es.n_edges} edges')
        q = d
    steps = np.broadcast_to(delay_steps(q, net.dt), (es.n_edges,)).astype(np.int64)
    per_edge = q if np.ndim(u.get_mantissa(q)) == 1 else u.Quantity(
        np.full((es.n_edges,), float(u.Quantity(q).to_decimal(u.ms))), unit=u.ms)
    return steps, per_edge


def _sub_edges(es: EdgeSet, pos: np.ndarray) -> EdgeSet:
    if pos.size == es.n_edges:
        return es
    p = jnp.asarray(pos)
    return EdgeSet(pre_index=es.pre_index[p], post_index=es.post_index[p],
                   pre_gids=es.pre_gids, post_gids=es.post_gids)


def _compile_projection(net: Network, proj: Projection, dtype: Any) -> ProjectionPlan:
    okey = object_key(net.seed, proj.uid)
    es = proj.rule.build(proj.pre.gids, proj.post.gids, key=okey)
    w_edges = _edge_weights(proj, es, okey, dtype)
    steps, d_edges = _edge_delays(net, proj, es, okey)
    es = EdgeSet(pre_index=es.pre_index, post_index=es.post_index, pre_gids=es.pre_gids,
                 post_gids=es.post_gids, weight=w_edges, delay=d_edges)
    force = {'auto': None, 'pre': None, 'post': 'per_edge'}[proj.delay_buffer]
    layout = classify_delays(steps, np.asarray(es.pre_index), es.n_pre, force=force)
    if proj.delay_buffer == 'pre' and layout.kind == 'per_edge':
        raise ValueError(f'{proj.name}: delays differ between edges of one source; a pre-side '
                         f"ring cannot hold them (use delay_buffer='auto' or 'post')")
    src = proj.src_kind
    path = payload_path(src) if isinstance(src, Event) else 'float'
    trainable = proj.trainable
    groups = []
    post_ring = None
    if proj.plasticity is not None and es.n_edges:
        return _compile_plastic(net, proj, es, w_edges, layout, path, dtype)
    timed = None
    if es.n_edges == 0:
        placement = 'none'
    elif isinstance(src, TimedEvent):
        # per-edge gather of precisely timed events (K0 T7); CSR keeps per-edge weights exact
        placement = 'timed'
        path = 'timed'
        st = make_storage(es, 'csr')
        groups.append(EdgeGroup(None, st, np.arange(es.n_edges),
                                _state(st.weight_from_edges(w_edges), trainable)))
        timed = TimedDelivery.build(net, proj, es, steps)
    elif layout.kind in ('scalar', 'per_source'):
        placement = 'pre'
        st = make_storage(es, proj.storage)
        delay = layout.steps if layout.kind == 'scalar' else None
        groups.append(EdgeGroup(delay, st, np.arange(es.n_edges),
                                _state(st.weight_from_edges(w_edges), trainable)))
    else:
        placement = 'post'
        for k in layout.values:
            pos = np.nonzero(steps == k)[0]
            sub = _sub_edges(es, pos)
            st = make_storage(sub, proj.storage)
            wk = w_edges[jnp.asarray(pos)]
            groups.append(EdgeGroup(int(k), st, pos, _state(st.weight_from_edges(wk), trainable)))
        if layout.max >= 1:
            post_ring = PostRing(es.n_post, layout.max, dtype)
    return ProjectionPlan(proj=proj, edges=es, layout=layout, placement=placement, path=path,
                          groups=groups, post_ring=post_ring, weight_unit=proj.weight_unit,
                          timed=timed)


def _compile_plastic(net: Network, proj: Projection, es: EdgeSet, w_edges: Any,
                     layout: DelayLayout, path: str, dtype: Any) -> ProjectionPlan:
    rule = proj.plasticity
    if layout.kind != 'scalar':
        raise ValueError(f'{proj.name}: plastic projections need one delay for all edges in K0 '
                         f'(got {layout.kind.replace("_", "-")} delays)')
    d = int(layout.steps)
    ax_ms = float(u.Quantity(rule.axonal_delay).to_decimal(u.ms))
    d_ax = int(delay_steps(rule.axonal_delay, net.dt)) if ax_ms > 0 else 0
    d_den = d - d_ax
    if d_den < 1:
        raise ValueError(f'{proj.name}: the dendritic delay (delay - axonal_delay = {d} - {d_ax} '
                         f'steps) must be at least one step')
    st = make_storage(es, 'csr')
    w_csr = st.weight_from_edges(w_edges)
    plastic = PlasticPlan(rule, st, to_mantissa(w_csr, proj.weight_unit, dtype), proj.weight_unit,
                          d_ax, d_den, dtype)
    group = EdgeGroup(d_den, st, np.arange(es.n_edges), _state(w_csr, False))
    return ProjectionPlan(proj=proj, edges=es, layout=layout, placement='post', path=path,
                          groups=[group], post_ring=PostRing(es.n_post, d_den, dtype),
                          weight_unit=proj.weight_unit, plastic=plastic)


def _zero_delay_pops(pp: ProjectionPlan) -> set:
    """Pre populations with at least one zero-delay edge in the projection."""
    lay = pp.layout
    segs = pp.proj.pre.segments
    if not lay.has_zero:
        return set()
    if lay.kind == 'scalar':
        return {s.pop for s in segs}
    if lay.kind == 'per_source':
        d = np.asarray(lay.steps)
        return {s.pop for s in segs if np.any(d[s.pos] == 0)}
    pre_pos = np.asarray(pp.edges.pre_index)[np.asarray(lay.steps) == 0]
    return {s.pop for s in segs if np.any((pre_pos >= s.pos.start) & (pre_pos < s.pos.stop))}


def compile_network(net: Network, precision: str = 'float32', merge: bool = True) -> Plan:
    """Compile ``net`` (see the module docstring).

    Parameters
    ----------
    net : Network
        The graph.
    precision : {'float32', 'float64'}, default 'float32'
        Float tier.
    merge : bool, default True
        Run the synapse merge pass.

    Returns
    -------
    Plan

    Raises
    ------
    ScheduleError
        On a zero-delay cycle.
    ValueError
        On inconsistent weights/delays (e.g. per-edge arrays of the wrong length).
    """
    plan = Plan(net, precision)
    plan.merge = bool(merge)
    dtype = plan.dtype
    for pop in net.populations:
        model = _instantiate(net, pop, plan.precision)
        plan.models.append(model)
        out_units = {}
        for port, kind in pop.model.outputs.items():
            out_units[port] = u.UNITLESS if isinstance(kind, (Event, TimedEvent)) else kind.unit
        plan.populations.append(PopulationPlan(pop=pop, model=model, out_units=out_units))

    zero_edges = set()
    ring_len: dict[tuple, int] = {}
    timed_len: dict[tuple, int] = {}
    for proj in net.projections:
        pp = _compile_projection(net, proj, dtype)
        plan.projections.append(pp)
        plan.weights[proj.name] = [g.weight for g in pp.groups]
        if pp.plastic is not None:
            plan.plastic_params[proj.name] = pp.plastic.states
        if pp.placement == 'none':
            continue
        pre_pops = {s.pop for s in proj.pre.segments}
        post_pops = {s.pop for s in proj.post.segments}
        for b in post_pops:
            plan.populations[b].incoming.append(proj.index)
        zero_edges |= {(a, b) for a in _zero_delay_pops(pp) for b in post_pops}
        if pp.placement == 'pre' and pp.layout.max >= 1:
            for a in pre_pops:
                k = (a, proj.source_port)
                ring_len[k] = max(ring_len.get(k, 0), pp.layout.max)
        if pp.plastic is not None:
            reads = [(a, proj.source_port, pp.plastic.d_ax) for a in pre_pops]
            reads += [(b, proj.plasticity.post_port, pp.plastic.d_den) for b in post_pops]
            for a, port, k in reads:
                if k >= 1:
                    ring_len[(a, port)] = max(ring_len.get((a, port), 0), k)
        if pp.timed is not None:
            for k, length in pp.timed.ring_lengths().items():
                timed_len[k] = max(timed_len.get(k, 0), length)
    for (a, port), length in ring_len.items():
        pplan = plan.populations[a]
        pplan.rings[port] = PreRing(pplan.pop.n, length, dtype)
    for (a, port), length in timed_len.items():
        pplan = plan.populations[a]
        cap = pplan.pop.model.outputs[port].capacity
        pplan.rings[port] = TimedRing(pplan.pop.n, length, cap, dtype)
    check_capacities(plan)
    targets = []
    for pp in plan.projections:
        prj = pp.proj
        if prj.synapse is None or pp.placement == 'none':
            continue
        for b in dict.fromkeys(s.pop for s in prj.post.segments):
            targets.append((prj.index, b, prj.port, prj.synapse,
                            f'{prj.name}@{plan.populations[b].name}'))
    plan.synapses, plan.synapse_route = plan_synapse_groups(targets, merge=plan.merge)
    try:
        plan.order = schedule(len(net.populations), zero_edges)
    except ScheduleError as e:
        names = [p.name for p in net.populations]
        raise ScheduleError(f'{e} ({names}); a cycle needs a delay of at least one step '
                            f'(decisions D4/D4a)') from None

    for obs in net.observers:
        units = {}
        for r in obs.recordables:
            pop0 = plan.populations[obs.view.segments[0].pop] if obs.view.segments else None
            units[r] = unit_of(pop0.model.dyn_states[r].value) if pop0 is not None else u.UNITLESS
        plan.observers.append(ObserverPlan(obs, units))
    return plan
