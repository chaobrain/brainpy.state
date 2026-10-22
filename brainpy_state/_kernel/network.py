# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""The ``Network`` graph: populations, views, projections and observers (K0 spec §4.8).

A :class:`Network` only *describes* a simulation; :meth:`Network.compile` turns it into
a :class:`~brainpy_state._kernel.compiler.Plan` (models instantiated at the requested
precision, edges realised, buffers placed), and ``plan.runner()`` runs it.

Global ids
----------
``net.add(Model, n)`` reserves the next ``n`` global ids (gids) and returns a
:class:`PopulationView` over them. Views slice (``E[:50]``, ``E[[3, 1]]``), and
concatenate (``E + I``) -- a concatenation is one view, sampled as one population by
connectivity rules (fixes doubled in-degrees). A view never lists a gid twice.

Names and random streams
------------------------
Every population, projection and observer has a unique name (duplicates raise).
Every random stream (model initialisation, per-step noise, connectivity, weights,
delays) is keyed by ``(seed, name, ...)`` -- a stable hash of the name
(:func:`~brainpy_state._kernel.rng.object_key`, D-g), never the insertion index or
Python identity -- so adding, removing or reordering an unrelated object leaves the
other objects' streams unchanged. Unnamed objects are auto-named ``<model>_<k>``,
``k`` counting the unnamed objects of that class (populations: the model class,
e.g. ``iaf_psc_alpha_0``; projections: the connectivity rule class, e.g.
``FixedIndegree_0``; observers: ``Observer_0``), so insertion order matters only
among *unnamed objects of the same class*; name objects to make their streams
independent of it. Renaming an object gives it new streams.

Synapses and plasticity (K0 T6)
-------------------------------
``synapse=`` attaches a :class:`~brainpy_state._kernel.synapse.SynapseModel` (e.g.
``Expon``) between delivery and an unfiltered ``Current`` / ``Conductance`` port;
``plasticity=`` attaches a :class:`~brainpy_state._kernel.plasticity.PlasticityRule`
(per-edge CSR weights, NEST lazy timing).

Gated conductances and precise timing (K0 T8, T7)
-------------------------------------------------
Voltage-gated ``conductance`` ports (T8) are supported: the gate is evaluated by the
receiving model at every substep (``with_conductance_ports``, ``models.nmda``).
``timed_event`` ports (precise spike times) are supported since T7 (:mod:`.timed`).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import brainunit as u
import numpy as np

from brainpy_state._kernel.connectivity.edgeset import as_gids
from brainpy_state._kernel.connectivity.rules import AllToAll, ConnRule
from brainpy_state._kernel.connectivity.storage import STORAGE_KINDS
from brainpy_state._kernel.model import NeuronModel
from brainpy_state._kernel.params import Distribution
from brainpy_state._kernel.plasticity import PlasticityRule
from brainpy_state._kernel.ports import (
    Event,
    PortError,
    PortKind,
    TimedEvent,
    check_connection,
    expected_weight_unit,
)
from brainpy_state._kernel.rng import name_id
from brainpy_state._kernel.synapse import check_synapse_connection
from brainpy_state._kernel.units import check_unit

__all__ = [
    'DELAY_BUFFERS',
    'Network',
    'Population',
    'PopulationView',
    'Segment',
    'Projection',
]

#: Delay-buffer placements accepted by :meth:`Network.connect`.
DELAY_BUFFERS = ('auto', 'pre', 'post')


@dataclass(frozen=True)
class Population:
    """Declaration of one population (created by :meth:`Network.add`).

    Attributes
    ----------
    index : int
        Position in ``net.populations``.
    name : str
        Unique name; keys every random stream of the population (RNG contract).
    model : type
        :class:`~brainpy_state._kernel.model.NeuronModel` subclass.
    n : int
        Size (``0`` allowed).
    gid_start : int
        First global id; the population owns ``[gid_start, gid_start + n)``.
    params, init : dict
        Parameter / initial-value overrides passed to the model.
    kwargs : dict
        Extra model constructor arguments (``integrator``, ``surrogate``,
        ``trainable``, ``detach_reset``, ...).
    """

    index: int
    name: str
    model: type
    n: int
    gid_start: int
    params: dict = field(default_factory=dict)
    init: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)

    @property
    def gids(self) -> np.ndarray:
        """Global ids, ``int64``."""
        return np.arange(self.gid_start, self.gid_start + self.n, dtype=np.int64)


@dataclass(frozen=True)
class Segment:
    """A run of consecutive view positions that belong to one population.

    Attributes
    ----------
    pop : int
        Population index.
    pos : slice
        Positions in the view (contiguous).
    local : slice or numpy.ndarray
        Local indices in the population: a ``slice`` when contiguous and
        increasing, else an ``int64`` index array.
    """

    pop: int
    pos: slice
    local: Any

    @property
    def size(self) -> int:
        """Number of neurons in the run."""
        return self.pos.stop - self.pos.start

    @property
    def full(self) -> bool:
        """Whether the run is a contiguous slice (fast path)."""
        return isinstance(self.local, slice)

    def local_array(self) -> np.ndarray:
        """Local indices as an array."""
        if isinstance(self.local, slice):
            return np.arange(self.local.start, self.local.stop, dtype=np.int64)
        return self.local


class PopulationView:
    """An ordered set of neurons of one network, identified by global ids.

    Returned by :meth:`Network.add`; supports indexing (int, slice, integer array,
    boolean mask), concatenation with ``+`` and ``len``.

    Parameters
    ----------
    net : Network
        Owning network.
    gids : array_like of int
        Global ids in view order (no duplicates).
    label : str, optional
        Human-readable label used in messages and ``describe``.

    Raises
    ------
    ValueError
        If ``gids`` contains duplicates or ids outside the network.
    """

    def __init__(self, net: 'Network', gids: Any, label: str | None = None) -> None:
        self.net = net
        self.gids = as_gids(np.asarray(gids, dtype=np.int64).reshape(-1), 'view gids')
        if self.gids.size and self.gids.max() >= net.n_neurons:
            raise ValueError(f'gids out of range for a network of {net.n_neurons} neurons')
        self.label = label if label is not None else f'view[{self.gids.size}]'
        self._segments: list[Segment] | None = None

    def __len__(self) -> int:
        return int(self.gids.size)

    @property
    def size(self) -> int:
        """Number of neurons."""
        return len(self)

    def __repr__(self) -> str:
        return f'PopulationView({self.label}, n={len(self)})'

    def __getitem__(self, idx: Any) -> 'PopulationView':
        if isinstance(idx, (int, np.integer)) and not isinstance(idx, bool):
            i = int(idx)
            if not -len(self) <= i < len(self):
                raise IndexError(f'index {i} out of range for a view of {len(self)}')
            sel = self.gids[i:i + 1] if i >= 0 else self.gids[len(self) + i:len(self) + i + 1]
            return PopulationView(self.net, sel, f'{self.label}[{i}]')
        if isinstance(idx, slice):
            return PopulationView(self.net, self.gids[idx], f'{self.label}[{_fmt_slice(idx)}]')
        a = np.asarray(idx)
        if a.dtype == bool:
            if a.shape != self.gids.shape:
                raise IndexError(f'boolean mask must have shape {self.gids.shape}, got {a.shape}')
            return PopulationView(self.net, self.gids[a], f'{self.label}[mask]')
        if not np.issubdtype(a.dtype, np.integer) and a.size:
            raise TypeError(f'cannot index a view with {type(idx).__name__}')
        return PopulationView(self.net, self.gids[a.astype(np.int64).reshape(-1)], f'{self.label}[...]')

    def __add__(self, other: 'PopulationView') -> 'PopulationView':
        if not isinstance(other, PopulationView):
            return NotImplemented
        if other.net is not self.net:
            raise ValueError('cannot concatenate views of different networks')
        gids = np.concatenate([self.gids, other.gids])
        if np.unique(gids).size != gids.size:
            raise ValueError(f'views {self.label} and {other.label} overlap; a concatenated '
                             f'view must list each neuron once')
        return PopulationView(self.net, gids, f'{self.label}+{other.label}')

    @property
    def segments(self) -> list[Segment]:
        """Runs of the view per population (cached)."""
        if self._segments is None:
            self._segments = self.net._segments(self.gids)
        return self._segments

    def stream_parts(self) -> list[tuple[str, np.ndarray]]:
        """``(population name, local ids)`` per segment, in view order.

        The RNG identity of the view's neurons (:func:`~brainpy_state._kernel.rng.view_post_keys`):
        independent of global-id offsets, hence of unrelated populations.
        """
        return [(self.net.populations[s.pop].name, s.local_array()) for s in self.segments]

    @property
    def populations(self) -> list[Population]:
        """Distinct populations touched by the view, in first-appearance order."""
        seen: list[int] = []
        for s in self.segments:
            if s.pop not in seen:
                seen.append(s.pop)
        return [self.net.populations[i] for i in seen]


def _fmt_slice(s: slice) -> str:
    a = '' if s.start is None else s.start
    b = '' if s.stop is None else s.stop
    return f'{a}:{b}' + ('' if s.step is None else f':{s.step}')


@dataclass(frozen=True)
class Projection:
    """Declaration of one projection (created by :meth:`Network.connect`).

    Attributes
    ----------
    index : int
        Position in ``net.projections``.
    name : str
        Unique name; keys the projection's connectivity, weight and delay streams.
    pre, post : PopulationView
        Source and target views.
    rule : ConnRule
        Connectivity rule.
    source_port, port : str
        Output port of the sources and input port of the targets.
    src_kind, dst_kind : PortKind
        Port kinds (the source kind with the smallest ``max_multiplicity`` guarantee
        when the pre view mixes models).
    weight : Any
        Weight specification (scalar, Quantity, ``(E,)`` array or Distribution).
    weight_unit : brainunit.Unit
        Unit weights are stored in (:func:`~brainpy_state._kernel.ports.expected_weight_unit`).
    delay : Any
        Delay specification (Quantity scalar / ``(E,)`` / Distribution).
    storage : {'auto', 'dense', 'csr'}
        Storage request.
    delay_buffer : {'auto', 'pre', 'post'}
        Delay-buffer placement request.
    trainable : bool
        Store weights as ``brainstate.ParamState``.
    synapse : SynapseModel or None
        Synapse kinetics between delivery and the port (K0 T6).
    plasticity : PlasticityRule or None
        Plasticity rule (K0 T6); ``static_synapse`` compiles as no rule.
    strict_capacity : bool
        ``timed_event`` targets: raise when the worst-case events per target and step
        exceed the port capacity (else count the overflow at run time).
    """

    index: int
    name: str
    pre: PopulationView
    post: PopulationView
    rule: ConnRule
    source_port: str
    port: str
    src_kind: PortKind
    dst_kind: PortKind
    weight: Any
    weight_unit: Any
    delay: Any
    storage: str
    delay_buffer: str
    trainable: bool
    synapse: Any = None
    plasticity: Any = None
    strict_capacity: bool = True


class Network:
    """Graph of populations, projections and observers (no state, no compilation).

    Parameters
    ----------
    dt : brainunit.Quantity
        Simulation step (time dimension, > 0).
    seed : int, default 0
        Root seed of every random stream (spec §4.5).

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.network import Network
        >>> from brainpy_state._kernel.models import iaf_psc_alpha
        >>> from brainpy_state._kernel.connectivity import fixed_indegree
        >>> net = Network(dt=0.1 * u.ms, seed=0)
        >>> E = net.add(iaf_psc_alpha, 80, name='E')
        >>> I = net.add(iaf_psc_alpha, 20, name='I')
        >>> E.gids[:3].tolist(), I.gids[:3].tolist()
        ([0, 1, 2], [80, 81, 82])
        >>> p = net.connect(E, E + I, rule=fixed_indegree(8), port='ex',
        ...                 weight=10.0 * u.pA, delay=1.5 * u.ms)
        >>> rec = net.observe(E[:5], 'spike', 'V_m')
        >>> len(net.populations), len(net.projections), len(net.observers)
        (2, 1, 1)
    """

    def __init__(self, dt: Any, seed: int = 0) -> None:
        check_unit(dt, u.ms, 'dt')
        if not float(u.Quantity(dt).to_decimal(u.ms)) > 0:
            raise ValueError(f'dt must be positive, got {dt}')
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 32:
            raise ValueError(f'seed must be an int in [0, 2**32), got {seed!r}')
        self.dt = dt
        self.seed = int(seed)
        self.populations: list[Population] = []
        self.projections: list[Projection] = []
        self.observers: list = []
        self._names: set[str] = set()
        self._name_ids: dict[tuple[int, int], str] = {}
        self._auto: dict[str, int] = {}

    # ---------------------------------------------------------------- helpers
    @property
    def n_neurons(self) -> int:
        """Total number of neurons (gids handed out)."""
        if not self.populations:
            return 0
        last = self.populations[-1]
        return last.gid_start + last.n

    def _unique(self, name: str | None, cls: str) -> str:
        """Validate ``name`` or auto-name ``<cls>_<k>`` (``k`` counts unnamed ``cls`` objects)."""
        if name is None:
            k = self._auto.get(cls, 0)
            while f'{cls}_{k}' in self._names:  # skip names taken explicitly
                k += 1
            self._auto[cls] = k + 1
            name = f'{cls}_{k}'
        if not isinstance(name, str) or not name:
            raise TypeError(f'name must be a non-empty str, got {name!r}')
        if name in self._names:
            raise ValueError(f'name {name!r} is already used in this network')
        nid = name_id(name)
        if nid in self._name_ids:  # pragma: no cover - 64-bit hash collision
            raise ValueError(f'names {name!r} and {self._name_ids[nid]!r} hash to the same RNG '
                             f'stream id; rename one of them')
        self._names.add(name)
        self._name_ids[nid] = name
        return name

    def _segments(self, gids: np.ndarray) -> list[Segment]:
        if gids.size == 0:
            return []
        starts = np.asarray([p.gid_start for p in self.populations], np.int64)
        pop = np.searchsorted(starts, gids, side='right') - 1
        local = gids - starts[pop]
        cut = np.nonzero(np.diff(pop) != 0)[0] + 1
        bounds = np.concatenate([[0], cut, [gids.size]])
        out = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            loc = local[a:b]
            if np.array_equal(loc, np.arange(loc[0], loc[0] + loc.size)):
                loc_s: Any = slice(int(loc[0]), int(loc[0] + loc.size))
            else:
                loc_s = loc.astype(np.int64)
            out.append(Segment(int(pop[a]), slice(int(a), int(b)), loc_s))
        return out

    def _check_view(self, view: Any, what: str) -> PopulationView:
        if not isinstance(view, PopulationView):
            raise TypeError(f'{what} must be a PopulationView, got {type(view).__name__}')
        if view.net is not self:
            raise ValueError(f'{what} belongs to another network')
        return view

    # ---------------------------------------------------------------- building
    def add(self, model: type, n: int, params: Mapping[str, Any] | None = None,
            init: Mapping[str, Any] | None = None, *, name: str | None = None,
            **kwargs: Any) -> PopulationView:
        """Add a population of ``n`` neurons of ``model``.

        Parameters
        ----------
        model : type
            A :class:`~brainpy_state._kernel.model.NeuronModel` (or ``Device``)
            subclass; instantiated at compile time with the plan's precision.
        n : int
            Number of neurons (``0`` allowed).
        params, init : mapping, optional
            Parameter / initial-value overrides (see ``NeuronModel``).
        name : str, optional
            Unique name; keys the population's random streams (default
            ``'<model>_<k>'``, ``k`` counting unnamed populations of ``model``, so
            the streams of unnamed populations depend on their order within the
            class -- see the module notes).
        **kwargs
            Extra constructor arguments (``integrator``, ``surrogate``,
            ``trainable``, ``detach_reset``, model-specific options).

        Returns
        -------
        PopulationView
            View over the population's gids.

        Raises
        ------
        TypeError
            If ``model`` is not a ``NeuronModel`` subclass, or ``kwargs`` contains
            ``precision`` / ``key`` (owned by the compiler).
        ValueError
            If ``n`` is negative or the name is taken.
        """
        if not (isinstance(model, type) and issubclass(model, NeuronModel)):
            raise TypeError(f'model must be a NeuronModel subclass, got {model!r}')
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f'n must be a non-negative int, got {n!r}')
        owned = {'precision', 'key'} & set(kwargs)
        if owned:
            raise TypeError(f'{sorted(owned)} are set by the compiler (net.compile / Network seed)')
        cap = model.capabilities
        if not cap.jit_safe:
            raise TypeError(f'{model.__name__} is not jit-safe and cannot run in a compiled plan')
        name = self._unique(name, model.__name__)
        pop = Population(index=len(self.populations), name=name, model=model,
                         n=int(n), gid_start=self.n_neurons, params=dict(params or {}),
                         init=dict(init or {}), kwargs=dict(kwargs))
        self.populations.append(pop)
        return PopulationView(self, pop.gids, name)

    def connect(self, pre: PopulationView, post: PopulationView, rule: ConnRule | None = None, *,
                port: str | None = None, weight: Any = None, delay: Any = None,
                storage: str = 'auto', synapse: Any = None, plasticity: Any = None,
                source_port: str | None = None, delay_buffer: str = 'auto',
                trainable: bool = False, name: str | None = None,
                strict_capacity: bool = True) -> Projection:
        """Connect ``pre`` to ``post`` (validated now, realised at compile).

        Parameters
        ----------
        pre, post : PopulationView
            Source and target views (slices and concatenations allowed).
        rule : ConnRule, optional
            Connectivity rule (default ``all_to_all``).
        port : str, optional
            Target input port; may be omitted when the targets have one input.
        weight : scalar, Quantity, array (E,) or Distribution, optional
            Weight in :func:`~brainpy_state._kernel.ports.expected_weight_unit`
            (event sources: the port unit, e.g. pA; value sources: dimensionless
            for equal units). ``None`` means ``1`` and is only allowed when that
            unit is dimensionless.
        delay : Quantity (scalar or (E,)) or Distribution, optional
            Delay, rounded to whole steps as NEST does
            (:func:`~brainpy_state._kernel.delays.delay_steps`). ``None`` means one
            step (NEST's minimum). ``0`` is allowed on acyclic paths.
        storage : {'auto', 'dense', 'csr'}, default 'auto'
            Edge storage (:func:`~brainpy_state._kernel.connectivity.make_storage`).
        synapse : SynapseModel, optional
            Synapse kinetics (e.g. ``Expon(tau)``) between delivery and an
            **unfiltered** ``Current`` / ``Conductance`` port; event sources only;
            weights in the port unit. A filtered port raises ``PortError`` (single
            owner of kinetics).
        plasticity : PlasticityRule, optional
            Plasticity rule (``static_synapse``, ``stdp_synapse``): per-edge CSR
            weights, scalar delay with ``delay - rule.axonal_delay >= dt``, event
            sources, the post populations must have the rule's ``post_port``.
        source_port : str, optional
            Output port of the sources; may be omitted when they have one output.
        delay_buffer : {'auto', 'pre', 'post'}, default 'auto'
            Delay-buffer placement: ``'pre'`` (ring per source; scalar or
            per-source delays), ``'post'`` (arrival ring per projection; any delays),
            or ``'auto'`` (``'pre'`` when the delays allow it).
        trainable : bool, default False
            Store the weights as ``brainstate.ParamState`` (learners).
        name : str, optional
            Unique name; keys the connectivity, weight and delay streams (default
            ``'<Rule>_<k>'``, e.g. ``'FixedIndegree_0'``, ``k`` counting unnamed
            projections with that rule class).
        strict_capacity : bool, default True
            For ``timed_event`` targets: raise :class:`PortError` if the events that
            can reach one target in one step may exceed the port capacity (checked
            here for one source, and at compile time for the realised in-degrees of
            all projections into the port). ``False`` accepts the risk: excess
            events are dropped **and counted** in the target's ``overflow`` state.
            Delays of timed projections are rounded to the grid; the sub-step
            offsets are preserved (NEST).

        Returns
        -------
        Projection

        Raises
        ------
        NotImplementedError
            For trainable weights on a plastic projection (hybrid weights, decision
            L4, are K1).
        PortError
            On unknown / ambiguous / incompatible ports, a negative weight into a
            conductance port, a filtering synapse onto a filtered port, or a
            ``timed_event`` capacity that may overflow (``strict_capacity=True``).
        UnitMismatchError
            On a weight or delay with the wrong dimension.
        """
        pre = self._check_view(pre, 'pre')
        post = self._check_view(post, 'post')
        if plasticity is not None and not isinstance(plasticity, PlasticityRule):
            raise TypeError(f'plasticity must be a PlasticityRule, got {type(plasticity).__name__}')
        plastic = plasticity is not None and not plasticity.is_static
        if plastic:
            if trainable:
                raise NotImplementedError('trainable weights on a plastic projection (hybrid '
                                          'weights, decision L4) are K1')
            if storage == 'dense':
                raise ValueError('plastic weights are per edge; use storage=\'csr\' or \'auto\' '
                                 '(the dense round trip is lossy with multapses)')
        rule = AllToAll() if rule is None else rule
        if not isinstance(rule, ConnRule):
            raise TypeError(f'rule must be a ConnRule, got {type(rule).__name__}')
        if storage not in STORAGE_KINDS:
            raise ValueError(f'storage must be one of {STORAGE_KINDS}, got {storage!r}')
        if delay_buffer not in DELAY_BUFFERS:
            raise ValueError(f'delay_buffer must be one of {DELAY_BUFFERS}, got {delay_buffer!r}')
        pre_models = [p.model for p in pre.populations]
        post_models = [p.model for p in post.populations]
        source_port = _pick_port(source_port, pre_models, 'outputs', 'source_port')
        port = _pick_port(port, post_models, 'inputs', 'port')
        src_kinds = [m.outputs.require(source_port, m.__name__) for m in pre_models]
        dst_kinds = [m.inputs.require(port, m.__name__) for m in post_models]
        if any(isinstance(k, TimedEvent) for k in src_kinds) and storage == 'dense':
            raise ValueError('projections from timed_event sources use CSR storage (exact per-edge '
                             "weights for every event); use storage='auto' or 'csr'")
        if weight is None:
            weight = 1.0
        if delay is None:
            delay = self.dt
        src_kind = _merge_src_kinds(src_kinds) if src_kinds else None
        unit = None
        for sm, sk in zip(pre_models or [None], src_kinds or [None]):
            for dm, dk in zip(post_models, dst_kinds):
                where = f'{sm.__name__ if sm else "?"}.{source_port} -> {dm.__name__}.{port}'
                if sk is not None:
                    if synapse is not None:
                        check_synapse_connection(synapse, sk, dk, weight, where=where)
                    else:
                        check_connection(sk, dk, weight, where=where,
                                         max_events=sk.capacity if isinstance(sk, TimedEvent) else None,
                                         strict_capacity=bool(strict_capacity))
                    w_unit = expected_weight_unit(sk, dk)
                    if unit is not None and u.get_dim(unit) != u.get_dim(w_unit):
                        raise PortError(f'{where}: targets expect weights of different dimensions '
                                        f'({unit} vs {w_unit})')
                    unit = unit if unit is not None else w_unit
        if plastic:
            for sk in src_kinds:
                if not isinstance(sk, Event):
                    raise PortError(f'{type(plasticity).__name__} needs spike events from the '
                                    f'sources; {source_port!r} is a {sk.kind!r} port')
            for m in post_models:
                k = m.outputs.require(plasticity.post_port, m.__name__)
                if not isinstance(k, Event):
                    raise PortError(f'{type(plasticity).__name__} reads post events from '
                                    f'{m.__name__}.{plasticity.post_port}, a {k.kind!r} port')
        if src_kind is None:  # empty pre view: nothing will be delivered
            src_kind = Event()
        dst_kind = dst_kinds[0] if dst_kinds else Event()
        if unit is None and (not pre_models or not post_models):
            # an empty side: nothing will be delivered, keep the weight's own unit
            unit = weight.unit if isinstance(weight, Distribution) else u.get_unit(weight)
        if unit is None:
            unit = expected_weight_unit(src_kind, dst_kind)
        if u.get_dim(unit) == u.get_dim(u.UNITLESS):
            unit = u.UNITLESS  # value -> value ports: the runner converts src -> dst units
        _check_delay_spec(delay)
        name = self._unique(name, type(rule).__name__)
        proj = Projection(index=len(self.projections), name=name, pre=pre,
                          post=post, rule=rule, source_port=source_port, port=port,
                          src_kind=src_kind, dst_kind=dst_kind, weight=weight, weight_unit=unit,
                          delay=delay, storage=storage, delay_buffer=delay_buffer,
                          trainable=bool(trainable), synapse=synapse,
                          plasticity=plasticity if plastic else None,
                          strict_capacity=bool(strict_capacity))
        self.projections.append(proj)
        return proj

    def observe(self, view: PopulationView, *recordables: str, interval: Any = None,
                name: str | None = None):
        """Record ``recordables`` of ``view`` (dense; see :mod:`.observers`).

        Parameters
        ----------
        view : PopulationView
            Neurons to record (columns in view order).
        *recordables : str
            State names (e.g. ``'V_m'``) or event outputs (``'spike'``: counts
            summed over each interval).
        interval : Quantity, optional
            Sampling interval, a whole multiple of ``dt`` (default ``dt``).
        name : str, optional
            Unique name (default ``'Observer_<k>'``).

        Returns
        -------
        Observer
        """
        from brainpy_state._kernel.observers import Observer
        view = self._check_view(view, 'view')
        obs = Observer.create(self, view, recordables, interval, self._unique(name, 'Observer'))
        self.observers.append(obs)
        return obs

    def compile(self, precision: str = 'float32', merge: bool = True):
        """Validate and compile into a :class:`~brainpy_state._kernel.compiler.Plan`.

        Parameters
        ----------
        precision : {'float32', 'float64'}, default 'float32'
            Float tier of states, weights and buffers (``float64`` needs
            ``jax_enable_x64``).
        merge : bool, default True
            Run the synapse merge pass (linear synapses with equal parameters onto
            the same port share one state).

        Returns
        -------
        Plan
        """
        from brainpy_state._kernel.compiler import compile_network
        return compile_network(self, precision=precision, merge=merge)


def _pick_port(name: str | None, models: Sequence[type], attr: str, what: str) -> str:
    if name is not None:
        if not isinstance(name, str):
            raise TypeError(f'{what} must be a str, got {name!r}')
        return name
    names = {tuple(getattr(m, attr)) for m in models}
    if len(names) == 1:
        only = next(iter(names))
        if len(only) == 1:
            return only[0]
    avail = sorted({p for m in models for p in getattr(m, attr)})
    raise PortError(f'{what} is ambiguous; give it explicitly (available: {avail})')


def _merge_src_kinds(kinds: Sequence[PortKind]) -> PortKind:
    first = kinds[0]
    for k in kinds[1:]:
        if type(k) is not type(first):
            raise PortError(f'pre view mixes source port kinds {type(first).__name__} and {type(k).__name__}')
    if isinstance(first, Event):
        mults = [k.max_multiplicity for k in kinds]
        if any(m is None for m in mults):
            return Event(max_multiplicity=None)
        return Event(max_multiplicity=max(mults))
    if isinstance(first, TimedEvent):
        return TimedEvent(max(k.capacity for k in kinds), unit=first.unit)
    return first


def _check_delay_spec(delay: Any) -> None:
    if isinstance(delay, Distribution):
        check_unit(delay.unit, u.ms, 'delay')
        lo, _ = delay.bounds()
        if lo < 0:
            raise ValueError('delay distributions must be clipped to non-negative values (low=0)')
        return
    check_unit(delay, u.ms, 'delay')
    m = np.asarray(u.get_mantissa(delay), dtype=np.float64)
    if m.ndim > 1:
        raise ValueError(f'delay must be a scalar or per-edge (E,) array, got shape {m.shape}')
    if m.size and (not np.all(np.isfinite(m)) or m.min() < 0):
        raise ValueError('delays must be finite and non-negative')
