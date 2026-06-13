# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Simulator — explicit NEST-flavored network builder and runner.

The :class:`Simulator` builds a flat module graph (populations, generators,
recorders, and delta-event projections) and runs it through a single
``brainstate.transform.for_loop``. Populations expose their per-step spikes via
a Simulator-managed :class:`_SpikeHolder` (NEST models do not persist a
``.spike`` state), so projections read the previous step's spikes — matching the
projection-before-dynamics convention. Recording is collected as a stacked JAX
array (the ``spike_recorder`` device mutates Python lists and cannot run inside
the jitted loop).
"""
from __future__ import annotations

import copy
import inspect
import itertools
from typing import Optional

import brainstate
import jax.numpy as jnp
import saiunit as u

from brainpy_state._base import Neuron
from brainpy_state._nest.ac_generator import ac_generator as _ac_generator
from brainpy_state._nest.dc_generator import dc_generator as _dc_generator
from brainpy_state._nest.multimeter import multimeter as _multimeter
from brainpy_state._nest.noise_generator import noise_generator as _noise_generator
from brainpy_state._nest.spike_recorder import spike_recorder as _spike_recorder
from brainpy_state._nest.step_current_generator import step_current_generator as _step_current_generator
from brainpy_state._nest.volume_transmitter import volume_transmitter as _volume_transmitter
from brainpy_state._network._event_plastic import EventPlasticProj, VoltageCoupledPlasticProj
from brainpy_state._network._event_proj import EventProjection
from brainpy_state._network._nodeview import NodeView, _Segment, _flat_size
from brainpy_state._network._rules import all_to_all, one_to_one

__all__ = ['Simulator', 'SimulationResult']

# NEST recordable name -> ordered candidate brainpy.state model State attributes.
# NEST exposes the membrane potential as ``V_m`` while the models store it on
# ``self.V``. Synaptic currents are spelled ``I_syn_ex``/``I_syn_in`` on the alpha
# family (``iaf_psc_alpha``) but ``i_syn_ex``/``i_syn_in`` on the exp family
# (``iaf_psc_exp``), so each maps to a tuple of candidate attributes tried in
# order. Recordables not listed here (``g_ex``, ``g_in``, ``w``, …) resolve by
# their own name via ``getattr`` (e.g. ``iaf_cond_alpha`` exposes ``g_ex``/``g_in``).
def _asc_sum(pop):
    """Total after-spike current: a precomputed sum state if present, else summed."""
    state = getattr(pop, '_asc_sum_state', None)
    if state is not None:
        return state.value
    return sum(s.value for s in pop._asc_states)


def _psc_sum(pop):
    """Total post-synaptic current ``I_syn`` = sum of per-port PSC states ``y2``.

    NEST ``glif_psc`` reports ``I_syn`` as the sum of every receptor's PSC
    (``glif_psc.cpp``: ``S_.I_syn_ += S_.y2_[i]`` over all receptors); brainpy
    stores those per-port PSCs as the ``y2`` list of States (each in pA).
    """
    return sum(s.value for s in pop.y2)


def _g_port(k):
    """Resolver for NEST per-port conductance ``g_k`` (1-indexed).

    The multi-receptor models lay conductance out three different ways: a
    ``g_syn`` *list* of per-receptor States (``glif_cond*``), a ``g`` *list* of
    per-receptor States (``gif_cond_exp_multisynapse``), or a single ``g`` State
    with the receptor on the last axis (``aeif_cond_beta_multisynapse``). This
    returns a ``pop -> value`` reader that handles all three.
    """
    idx = k - 1

    def read(pop):
        g_syn = getattr(pop, 'g_syn', None)
        if g_syn is not None:                       # glif_cond: list of States
            return g_syn[idx].value
        g = getattr(pop, 'g', None)
        if g is None:
            raise KeyError(f'g_{k}: population exposes neither g_syn nor g')
        if isinstance(g, (list, tuple)):            # gif: list of States
            return g[idx].value
        return g.value[..., idx]                     # aeif: single State, last axis

    return read


# Map NEST recordable names to brainpy.state State attributes. Most current-based
# neurons store ``V`` (NEST ``V_m``); the exp family uses ``i_syn_ex`` where the
# alpha family uses ``I_syn_ex``; so each maps either to a tuple of candidate attrs
# tried in order, or to a callable ``pop -> value`` for derived/indexed recordables
# (per-port conductance, summed adaptation currents). Recordables not listed here
# resolve by their own name via ``getattr`` (e.g. ``iaf_cond_alpha`` exposes ``g_ex``).
_RECORDABLE_ALIAS = {
    'V_m': ('V',),
    # Injected current-generator input (NEST ``I`` = S_.I_ = currents_ ring buffer);
    # the current-based models buffer it on the I_stim ShortTermState.
    'I': ('I_stim',),
    'I_syn_ex': ('I_syn_ex', 'i_syn_ex'),
    'I_syn_in': ('I_syn_in', 'i_syn_in'),
    # HH gating (NEST Act_m/Inact_h/Act_n -> brainpy m/h/n).
    'Act_m': ('m',),
    'Inact_h': ('h',),
    'Act_n': ('n',),
    # GIF adaptation (NEST E_sfa / I_stc).
    'E_sfa': ('_sfa_val_state',),
    'I_stc': ('_stc_val_state',),
    # GLIF threshold components (relative to E_L; demos add E_L test-side).
    'threshold': ('_threshold_state',),
    'threshold_spike': ('_threshold_spike_state',),
    'threshold_voltage': ('_threshold_voltage_state',),
    # Per-port conductance g_k (glif_cond g_syn list, gif g list, aeif g last-axis)
    # and total after-spike current.
    'g_1': _g_port(1),
    'g_2': _g_port(2),
    'g_3': _g_port(3),
    'g_4': _g_port(4),
    'ASCurrents_sum': _asc_sum,
    # glif_psc total post-synaptic current: sum of per-port PSC states (y2).
    'I_syn': _psc_sum,
}


def _read_recordable(pop, name):
    """Read a NEST recordable as the model's State value (Quantity or array).

    Resolves ``name`` via ``_RECORDABLE_ALIAS``: a callable entry is invoked as
    ``entry(pop)`` (for derived/indexed recordables), otherwise ``name`` maps to a
    tuple of candidate attr spellings tried in order (falling back to the recordable
    name itself).
    """
    entry = _RECORDABLE_ALIAS.get(name, (name,))
    if callable(entry):
        return entry(pop)
    for attr in entry:
        state = getattr(pop, attr, None)
        if state is not None:
            return state.value
    raise KeyError(
        f'recordable {name!r} (tried {entry}) is not available on '
        f'{type(pop).__name__}'
    )


class _SpikeHolder(brainstate.nn.Module):
    """Per-population holder for the most recent captured spike/counts vector."""
    __module__ = 'brainpy.state'

    def __init__(self, n: int):
        super().__init__()
        self._n = int(n)

    def init_state(self, *args, **kwargs):
        self.spk = brainstate.ShortTermState(
            jnp.zeros(self._n, dtype=brainstate.environ.dftype())
        )


class _GeneratorSpec:
    """A deferred generator (model class + params), realised per target."""
    def __init__(self, model_cls, params):
        self.model_cls = model_cls
        self.params = params


class _GenSegment:
    """A NodeView segment carrying a deferred generator spec (size unknown)."""
    def __init__(self, spec: _GeneratorSpec):
        self.spec = spec
        self.population = None
        self.indices = jnp.arange(0)


def _holder_reader(holder: _SpikeHolder):
    return lambda: holder.spk.value


def _is_generator(model_cls) -> bool:
    name = getattr(model_cls, '__name__', '')
    return 'generator' in name or 'injector' in name


# Generators that inject a *current* (pA) rather than emitting spike events. These
# wire into the neuron's current-input seam (NEST current ring buffer, one-step
# delay), not the delta-event path used by spike generators.
_CURRENT_GENERATORS = (_noise_generator, _dc_generator, _step_current_generator,
                       _ac_generator)


def _is_current_generator(model_cls) -> bool:
    return isinstance(model_cls, type) and issubclass(model_cls, _CURRENT_GENERATORS)


def _n_channels(size) -> int:
    """Flatten a ``create`` size spec to a scalar channel count."""
    if isinstance(size, (tuple, list)):
        n = 1
        for s in size:
            n *= int(s)
        return n
    return int(size)


def _is_len_vector(val, k: int) -> bool:
    """True if ``val`` is a length-``k`` 1-D vector (Quantity / array / sequence)."""
    if isinstance(val, u.Quantity):
        m = val.mantissa
        return jnp.ndim(m) >= 1 and m.shape[0] == k
    if isinstance(val, (list, tuple)):
        return len(val) == k
    return hasattr(val, 'shape') and jnp.ndim(val) >= 1 and val.shape[0] == k


def _index_channel(val, i: int, k: int):
    """Channel ``i`` of a length-``k`` vector ``val``; broadcast a scalar unchanged.

    Splits a vector-valued generator parameter (e.g. ``rate=[r0, r1] * u.Hz``) or
    a per-segment ``weight`` into one scalar per channel, preserving units.
    """
    if not _is_len_vector(val, k):
        return val
    if isinstance(val, u.Quantity):
        return u.maybe_decimal(val.mantissa[i] * u.get_unit(val))
    return val[i]


class SimulationResult:
    """Recorded spikes and analog traces from a :meth:`Simulator.simulate` run.

    Spike recorders are read with :meth:`spikes` / :meth:`n_events` / :meth:`rate`.
    Analog recorders (``voltmeter`` / ``multimeter``, connected in NEST's reversed
    direction) are read with :meth:`trace`, and the common time axis with
    :attr:`times`.
    """
    __module__ = 'brainpy.state'

    def __init__(self, recordings: dict, duration, dt, *, traces=None, times=None,
                 weights=None):
        self._rec = recordings          # {id(recorder): (T, n_rec) array}
        self._T = duration
        self._dt = dt
        self._traces = dict(traces or {})  # {f'{id(rec)}|{recordable}': (T, n) Quantity}
        self._times = times                # (T,) Quantity, the for_loop time axis
        self._weights = dict(weights or {})  # {id(proj): (T, E) weight trajectory}

    @staticmethod
    def _key(node):
        if isinstance(node, NodeView):
            return id(node.segments[0].population)
        return id(node)

    @staticmethod
    def _trace_key(rid, recordable):
        return f'{rid}|{recordable}'

    def spikes(self, node):
        """Per-step spike matrix ``(n_steps, n_recorded)`` for a recorder/source."""
        return self._rec[self._key(node)]

    def n_events(self, node) -> int:
        return int(jnp.sum(self._rec[self._key(node)] > 0))

    def rate(self, node) -> float:
        """Mean firing rate in spikes/second over all recorded neurons."""
        spk = self._rec[self._key(node)]
        n = spk.shape[1]
        t_s = float(self._T.to_decimal(u.second))
        return float(jnp.sum(spk > 0)) / n / t_s

    def trace(self, recorder, recordable: str = 'V_m'):
        """Analog trace ``(n_steps, n_recorded)`` for an analog recorder.

        Parameters
        ----------
        recorder : NodeView
            The ``voltmeter`` / ``multimeter`` handle returned by
            :meth:`Simulator.create` and connected via ``connect(recorder, pop)``.
        recordable : str, optional
            Recordable name (NEST vocabulary, e.g. ``'V_m'``, ``'g_ex'``).
            Default is ``'V_m'``.

        Returns
        -------
        saiunit.Quantity
            ``(n_steps, n_recorded)`` trace in the model state's natural unit.

        Raises
        ------
        KeyError
            If ``recordable`` was not recorded by this recorder.
        """
        rid = self._key(recorder)
        key = self._trace_key(rid, recordable)
        if key not in self._traces:
            available = sorted(k.split('|', 1)[1] for k in self._traces
                               if k.startswith(f'{rid}|'))
            raise KeyError(
                f'recordable {recordable!r} was not recorded by this recorder; '
                f'recorded: {available}'
            )
        return self._traces[key]

    def weight_trace(self, proj):
        """Per-step weight trajectory ``(n_steps, n_edges)`` for a recorded proj.

        Parameters
        ----------
        proj : EventPlasticProj
            The plastic-projection handle returned by ``connect(..., synapse=spec)``
            and registered via :meth:`Simulator.record_weight` before the run.

        Returns
        -------
        saiunit.Quantity
            ``(n_steps, n_edges)`` weights in the synapse weight unit (pA), in CSR
            (sorted-by-pre) edge order — the same order the rule kernel sees.

        Raises
        ------
        KeyError
            If this projection's weight was not recorded (no
            :meth:`Simulator.record_weight` before :meth:`Simulator.simulate`).
        """
        rid = id(proj)
        if rid not in self._weights:
            raise KeyError(
                "this projection's weight was not recorded; call "
                'sim.record_weight(proj) before simulate()'
            )
        return self._weights[rid]

    @property
    def times(self):
        """The common time axis ``(n_steps,)`` of the run (saiunit Quantity)."""
        return self._times


class Simulator(brainstate.nn.Module):
    """Explicit NEST-flavored network builder and runner.

    Parameters
    ----------
    dt : saiunit.Quantity
        Simulation timestep; set into ``brainstate.environ`` at construction.

    Examples
    --------
    .. code-block:: python

       >>> import saiunit as u
       >>> from brainpy_state import iaf_psc_alpha, poisson_generator, spike_recorder
       >>> from brainpy_state.network import Simulator, all_to_all
       >>> sim = Simulator(dt=0.1 * u.ms)
       >>> pop = sim.create(iaf_psc_alpha, 10)
       >>> noise = sim.create(poisson_generator, rate=8000. * u.Hz)
       >>> rec = sim.create(spike_recorder)
       >>> sim.connect(noise, pop, weight=20. * u.pA, delay=1.5 * u.ms, rule=all_to_all)
       >>> sim.connect(pop, rec)
       >>> res = sim.simulate(100. * u.ms)
       >>> rate = res.rate(rec)
    """
    __module__ = 'brainpy.state'

    def __init__(self, *, dt):
        super().__init__()
        brainstate.environ.set(dt=dt)
        self._dt = dt
        self._taps = {}                       # id(recorder) -> (id(source), idx)
        self._analog_taps = {}                # id(recorder) -> (id(pop), idx, recordables)
        self._weight_taps = {}                # id(proj) -> EventPlasticProj (weight tap)
        self._current_injectors = []          # (device, post_pop, post_idx, weight, key)
        self._vt_nodes = []                   # volume_transmitter nodes (phase-0 update)
        self._proj_counter = itertools.count()

    # -- node creation -----------------------------------------------------
    def create(self, model_cls, size=1, *, params=None, **kw) -> NodeView:
        """Instantiate a population/device and return a :class:`NodeView`.

        Generators are deferred (realised per target at :meth:`connect`) so each
        target receives an independent train, mirroring NEST fan-out.
        """
        p = dict(params or {})
        p.update(kw)
        if _is_generator(model_cls):
            k = _n_channels(size)
            if k > 1:
                # Multi-channel generator (Extension D2): one independent segment
                # per channel, each a scalar-param spec. Vector params (e.g.
                # ``rate=[r0, r1]``) are split per channel; scalars broadcast.
                return NodeView([
                    _GenSegment(_GeneratorSpec(
                        model_cls, {key: _index_channel(v, i, k) for key, v in p.items()}))
                    for i in range(k)
                ])
            return NodeView([_GenSegment(_GeneratorSpec(model_cls, p))])
        mod = model_cls(size, **p)
        setattr(self, f'_node_{id(mod)}', mod)
        # Volume transmitters are driven in phase 0 (before projections) and expose
        # the dopamine concentration ``n`` as State; they emit no spikes, so they
        # get no _SpikeHolder (phase 2 skips them) and are registered for phase 0.
        if isinstance(mod, _volume_transmitter):
            self._vt_nodes.append(mod)
            return NodeView.of(mod)
        # Recorders are tapped, not driven: spike recorders read captured spikes,
        # analog recorders (voltmeter/multimeter) read model State per step. Neither
        # gets a _SpikeHolder.
        if isinstance(mod, (_spike_recorder, _multimeter)):
            return NodeView([_Segment(mod, jnp.arange(1))])
        holder = _SpikeHolder(_flat_size(mod))
        setattr(self, f'_holder_{id(mod)}', holder)
        return NodeView.of(mod)

    # -- connection --------------------------------------------------------
    def connect(self, pre: NodeView, post: NodeView, *, rule=all_to_all,
                weight=None, delay=None, comm: str = 'dense', receptor_type=None,
                synapse=None, vt=None, allow_autapses: bool = True,
                allow_multapses: bool = True, seed: Optional[int] = None):
        """Connect ``pre`` to ``post`` (or register a recorder tap).

        ``comm='sparse'`` routes the connectivity through a sparse CSR event
        matmul (memory-light for large fan-out); ``'dense'`` (default) uses a
        dense weight matrix. Both yield identical results for the same rule/seed.

        ``receptor_type='uniform'`` routes each edge to a uniformly-drawn receptor
        port of a multi-receptor post population (``iaf_psc_exp_multisynapse``).

        ``synapse=<spec>`` builds a plastic :class:`EventPlasticProj` from a
        rebuilt ``_nest`` synapse spec (``static_synapse``, the ``tsodyks*``
        family, ``quantal_stp_synapse``); ``weight``/``delay`` here override the
        spec's defaults. ``synapse=None`` (default) keeps the static
        :class:`EventProjection` path unchanged.

        ``connect(dopa_pool, vt)`` (reverse direction, ``post`` a
        ``volume_transmitter`` view) registers each presynaptic segment as a
        dopaminergic source on the transmitter and builds no projection.
        ``vt=<volume_transmitter view>`` binds a transmitter to a synapse spec that
        reads a broadcast signal (``signal_reads``, e.g. ``stdp_dopamine_synapse``);
        such a spec raises if no ``vt`` is supplied.

        Analog recorders (``voltmeter`` / ``multimeter``) are connected in NEST's
        reversed direction --- ``connect(recorder, pop)`` --- because the recorder
        *observes* the population. This registers a per-step State tap; no
        projection is built.

        Returns
        -------
        EventProjection or EventPlasticProj or list or None
            The projection handle(s) built by this call (a single handle when one
            projection is built, a list for multi-segment fan-out). A plastic
            handle (``synapse=spec``) can be passed to :meth:`record_weight`.
            Recorder-tap connects (and current injectors) return ``None``.
        """
        if len(pre.segments) == 1 and isinstance(pre.segments[0].population, _multimeter):
            rec = pre.segments[0].population
            if len(post.segments) != 1:
                raise NotImplementedError(
                    'a voltmeter/multimeter records a single population segment'
                )
            seg = post.segments[0]
            self._analog_taps[id(rec)] = (id(seg.population), seg.indices,
                                          tuple(rec.record_from))
            return None
        if len(post.segments) == 1 and isinstance(post.segments[0].population, _spike_recorder):
            if len(pre.segments) != 1:
                raise NotImplementedError(
                    'recording a multi-segment view requires one recorder per segment'
                )
            seg = pre.segments[0]
            self._taps[id(post.segments[0].population)] = (id(seg.population), seg.indices)
            return None
        if len(post.segments) == 1 and isinstance(post.segments[0].population, _volume_transmitter):
            # reverse-direction bind: connect(dopa_pool, vt) registers each dopa
            # source on the transmitter (no projection built, like a recorder tap).
            vt_node = post.segments[0].population
            for pre_seg in pre.segments:
                self._bind_dopa_source(pre_seg, vt_node)
            return None
        seg_weights = self._segment_weights(weight, len(pre.segments))
        projs = []
        for pre_seg, w_seg in zip(pre.segments, seg_weights):
            for post_seg in post.segments:
                proj = self._connect_pair(pre_seg, post_seg, rule, w_seg, delay,
                                          allow_autapses, allow_multapses, seed, comm,
                                          receptor_type, synapse, vt)
                if proj is not None:
                    projs.append(proj)
        if not projs:
            return None
        return projs[0] if len(projs) == 1 else projs

    def record_weight(self, proj):
        """Register a per-step weight tap on a plastic projection.

        ``proj`` is the handle returned by ``connect(..., synapse=spec)``. After
        :meth:`simulate`, read the stacked ``(n_steps, n_edges)`` weight trajectory
        (CSR sorted-by-pre edge order) via :meth:`SimulationResult.weight_trace`.
        Mirrors the analog-recorder tap, but reads the projection's ``weight``
        State rather than a population's recordable.

        Returns
        -------
        EventPlasticProj
            The same ``proj``, for chaining.

        Raises
        ------
        TypeError
            If ``proj`` is not a plastic projection (only ``connect(..., synapse=)``
            builds one; the static path has no plastic ``weight`` State to record).
        """
        if not isinstance(proj, EventPlasticProj):
            raise TypeError(
                'record_weight requires a plastic projection handle from '
                f'connect(..., synapse=spec); got {type(proj).__name__}'
            )
        self._weight_taps[id(proj)] = proj
        return proj

    @staticmethod
    def _segment_weights(weight, n_seg: int):
        """One weight per pre-segment (Extension D2).

        A ``weight`` vector whose length equals the number of pre-segments is
        indexed per segment (``weight[i]`` -> segment ``i``); any other ``weight``
        (scalar, or a per-edge vector for a single-segment ``all_to_all``) is
        passed through unchanged to every segment.
        """
        if n_seg > 1 and _is_len_vector(weight, n_seg):
            return [_index_channel(weight, i, n_seg) for i in range(n_seg)]
        return [weight] * n_seg

    @staticmethod
    def _derive_seed(base, ordinal: int) -> int:
        """Distinct, reproducible seed per realized projection/generator.

        Fan-out (one ``connect`` to several post segments, or one generator to
        several targets) must draw independently; ``jax.random`` derives element
        ``j`` from counter ``j`` regardless of array length, so sharing a base
        seed would otherwise duplicate trains/connectivity across segments.
        """
        b = 0 if base is None else int(base)
        return (b * 1_000_003 + ordinal + 1) & 0x7FFFFFFF

    @staticmethod
    def _resolve_synapse(synapse, weight, delay):
        """Shallow-copy a plastic synapse spec, applying connect-level overrides."""
        spec = copy.copy(synapse)
        if weight is not None:
            if isinstance(weight, u.Quantity):
                spec.weight = weight
            else:
                # preserve the spec's own weight unit (pA for current synapses,
                # mV for the delta-model clopath_synapse) instead of assuming pA
                spec.weight = weight * u.get_unit(spec.weight)
            spec.weight_unit = u.get_unit(spec.weight)
        if delay is not None:
            spec.delay = delay
        return spec

    @staticmethod
    def _plastic_proj_cls(synapse):
        """Pick the plastic-projection primitive for a synapse spec.

        A spec declaring a non-empty ``post_state_reads`` (e.g. ``clopath_synapse``,
        per-edge post-State gather) **or** a non-empty ``signal_reads`` (e.g.
        ``stdp_dopamine_synapse``, broadcast modulator) needs the voltage-coupled
        reader (primitive #2); every other plastic spec uses the event-driven
        primitive #1.
        """
        if getattr(synapse, 'post_state_reads', ()) or getattr(synapse, 'signal_reads', ()):
            return VoltageCoupledPlasticProj
        return EventPlasticProj

    @staticmethod
    def _build_signal_sources(synapse, vt):
        """Resolve a spec's ``signal_reads`` names to ``{name: (vt_module, attr)}``.

        Each broadcast signal name resolves to the same-named State attribute on the
        bound :class:`~brainpy_state._nest.volume_transmitter` (``'n'`` -> ``vt.n``).
        Returns ``None`` for a spec that reads no signal (clopath); raises if a
        signal-reading spec is given no transmitter.
        """
        names = tuple(getattr(synapse, 'signal_reads', ()) or ())
        if not names:
            return None
        if vt is None:
            raise ValueError(
                f'{type(synapse).__name__} reads broadcast signal(s) {names} and '
                'requires a bound volume_transmitter; pass '
                'connect(..., vt=<volume_transmitter view>).'
            )
        vt_mod = vt.segments[0].population if isinstance(vt, NodeView) else vt
        if not isinstance(vt_mod, _volume_transmitter):
            raise ValueError('vt= must be a volume_transmitter view.')
        return {name: (vt_mod, name) for name in names}

    def _bind_dopa_source(self, pre_seg, vt):
        """Register one presynaptic segment as a dopaminergic source on ``vt``.

        A population segment binds its captured-spike holder directly; a deferred
        generator segment (e.g. ``spike_generator``) is realized as a single-channel
        dopa pool with its own holder (driven in phase 2), so its one-step holder lag
        plays the role of NEST's ``spike_generator -> parrot -> volume_transmitter``
        relay.
        """
        if isinstance(pre_seg, _GenSegment):
            ordinal = next(self._proj_counter)
            params = dict(pre_seg.spec.params)
            if 'rng_seed' in inspect.signature(pre_seg.spec.model_cls.__init__).parameters:
                params['rng_seed'] = self._derive_seed(params.get('rng_seed'), ordinal)
            gen = pre_seg.spec.model_cls(1, **params)
            setattr(self, f'_node_{id(gen)}', gen)
            holder = _SpikeHolder(1)
            setattr(self, f'_holder_{id(gen)}', holder)
            vt.bind_dopa(_holder_reader(holder), jnp.arange(1))
        else:
            pre_pop = pre_seg.population
            holder = getattr(self, f'_holder_{id(pre_pop)}', None)
            if holder is None:
                raise ValueError(
                    'the dopaminergic source for a volume_transmitter must be a '
                    'spiking population or generator (no captured spikes found).'
                )
            vt.bind_dopa(_holder_reader(holder), pre_seg.indices)

    def _connect_pair(self, pre_seg, post_seg, rule, weight, delay,
                      allow_autapses, allow_multapses, seed, comm='dense',
                      receptor_type=None, synapse=None, vt=None):
        ordinal = next(self._proj_counter)
        post_pop = post_seg.population
        post_holder = getattr(self, f'_holder_{id(post_pop)}', None)
        post_reader = _holder_reader(post_holder) if post_holder is not None else None
        # voltage-coupled reader (#2) also carries broadcast signal sources (the VT n
        # for stdp_dopamine_synapse); primitive #1 and a vt-less spec build neither.
        proj_cls = self._plastic_proj_cls(synapse) if synapse is not None else None
        plastic_extra = {}
        if proj_cls is VoltageCoupledPlasticProj:
            plastic_extra['signal_sources'] = self._build_signal_sources(synapse, vt)
        if isinstance(pre_seg, _GenSegment):
            if _is_current_generator(pre_seg.spec.model_cls):
                self._wire_current_injector(pre_seg, post_seg, weight, ordinal)
                return
            n = int(post_seg.indices.shape[0])
            params = dict(pre_seg.spec.params)
            if 'rng_seed' in inspect.signature(pre_seg.spec.model_cls.__init__).parameters:
                params['rng_seed'] = self._derive_seed(params.get('rng_seed'), ordinal)
            gen = pre_seg.spec.model_cls(n, **params)
            setattr(self, f'_node_{id(gen)}', gen)
            holder = _SpikeHolder(n)
            setattr(self, f'_holder_{id(gen)}', holder)
            if synapse is not None:
                proj = proj_cls(
                    pre_spike=_holder_reader(holder), n_pre_pop=n,
                    pre_local_idx=jnp.arange(n), post=post_pop,
                    post_local_idx=post_seg.indices, n_post_pop=_flat_size(post_pop),
                    post_spike=post_reader, rule=self._resolve_synapse(synapse, weight, delay),
                    conn=one_to_one, seed=seed, **plastic_extra)
            else:
                proj = EventProjection(
                    pre_spike=_holder_reader(holder), n_pre_pop=n,
                    pre_local_idx=jnp.arange(n), post=post_pop,
                    post_local_idx=post_seg.indices, rule=one_to_one, weight=weight,
                    delay=delay, receptor_type=receptor_type, seed=seed)
        else:
            pre_pop = pre_seg.population
            holder = getattr(self, f'_holder_{id(pre_pop)}')
            if synapse is not None:
                proj = proj_cls(
                    pre_spike=_holder_reader(holder), n_pre_pop=_flat_size(pre_pop),
                    pre_local_idx=pre_seg.indices, post=post_pop,
                    post_local_idx=post_seg.indices, n_post_pop=_flat_size(post_pop),
                    post_spike=post_reader, rule=self._resolve_synapse(synapse, weight, delay),
                    conn=rule, pre_is_post=(pre_pop is post_pop),
                    allow_autapses=allow_autapses, allow_multapses=allow_multapses,
                    seed=self._derive_seed(seed, ordinal), **plastic_extra)
            else:
                proj = EventProjection(
                    pre_spike=_holder_reader(holder), n_pre_pop=_flat_size(pre_pop),
                    pre_local_idx=pre_seg.indices, post=post_pop,
                    post_local_idx=post_seg.indices, rule=rule, weight=weight,
                    delay=delay, comm=comm, receptor_type=receptor_type,
                    pre_is_post=(pre_pop is post_pop),
                    allow_autapses=allow_autapses, allow_multapses=allow_multapses,
                    seed=self._derive_seed(seed, ordinal))
        setattr(self, f'_proj_{ordinal}', proj)
        return proj

    def _wire_current_injector(self, pre_seg, post_seg, weight, ordinal):
        """Realize a current generator at the post size and register it as an injector.

        Current generators (``dc_generator`` / ``step_current_generator`` /
        ``noise_generator`` / ``ac_generator``) inject a *current* (pA) into the
        post population's current-input seam each step --- the neuron's own
        ``sum_current_inputs`` ring buffer (one-step delay, matching NEST's
        current ring buffer) --- rather than the delta-event path used by spike
        generators. No spike holder and no projection are built.

        The device is realized at ``n = n_post`` so a single generator fans out
        one independent channel per target (``noise_generator`` draws ``randn(n)``
        each step); a per-connect derived seed keeps separate connects
        independent.
        """
        post_pop = post_seg.population
        n = int(post_seg.indices.shape[0])
        params = dict(pre_seg.spec.params)
        if 'seed' in inspect.signature(pre_seg.spec.model_cls.__init__).parameters:
            params['seed'] = self._derive_seed(params.get('seed'), ordinal)
        device = pre_seg.spec.model_cls(n, **params)
        setattr(self, f'_node_{id(device)}', device)
        key = f'cur_inj_{ordinal}'
        self._current_injectors.append((device, post_pop, post_seg.indices, weight, key))

    @staticmethod
    def _scatter_current(cur, pop, idx):
        """Place a device's ``(n,)`` current into the post population's ``(n_pop,)`` frame.

        ``cur`` is the generator's per-channel current (one entry per target,
        in device order); it is scattered into a zero current vector over the
        full population at ``idx`` so neurons outside the connection receive no
        current. Works for full, partial, and reordered target views.
        """
        n_pop = _flat_size(pop)
        mant = u.get_mantissa(cur)
        base = jnp.zeros(n_pop, dtype=mant.dtype)
        return u.maybe_decimal(base.at[idx].add(mant) * u.get_unit(cur))

    # -- run ---------------------------------------------------------------
    def update(self, t=None):
        dftype = brainstate.environ.dftype()
        children = list(self.nodes(allowed_hierarchy=(1, 1)).values())
        # 0) volume transmitters advance the broadcast dopamine concentration n from
        #    the previous step's captured dopa spikes (the substrate's one-step lag,
        #    matching NEST's +1 delivery stamp), so projections in phase 1 read fresh n.
        for vt in self._vt_nodes:
            vt.update()
        # 1) projections route the previous step's spikes into delta inputs
        for m in children:
            if isinstance(m, (EventProjection, EventPlasticProj)):
                m.update()
        # 1b) current-injecting devices (dc/step/noise/ac) push this step's
        #     current into each post population's current-input seam. The neuron
        #     consumes it in step 2 via ``sum_current_inputs`` (captured into
        #     ``y0`` and applied on the *next* step --- a one-step ring buffer,
        #     matching NEST's current buffer). Non-callable inputs are popped on
        #     consumption, so the contribution is re-added every step.
        for device, pop, idx, weight, key in self._current_injectors:
            cur = device.update()
            if weight is not None:
                cur = cur * weight
            pop.add_current_input(key, self._scatter_current(cur, pop, idx))
        # 2) drive neurons/generators and capture their output into holders
        for m in children:
            if isinstance(m, (EventProjection, EventPlasticProj, _SpikeHolder)):
                continue
            holder = getattr(self, f'_holder_{id(m)}', None)
            if holder is None:
                continue  # recorders / untracked devices have no holder
            if (isinstance(m, Neuron) and hasattr(m, 'n_receptors')
                    and 'w_by_rec' in inspect.signature(type(m).update).parameters):
                # Multi-receptor neuron: gather the per-port delta input and drive
                # the model's JIT-safe ``w_by_rec`` path (its no-arg seam is numpy).
                # ``receptor_input_unit`` scales the gathered mantissa: pA for
                # current-based (iaf), nS for conductance-based (aeif/gif) models.
                runit = getattr(m, 'receptor_input_unit', u.pA)
                init = u.math.zeros(m.varshape + (int(m.n_receptors),)) * runit
                out = m.update(w_by_rec=u.get_mantissa(m.sum_delta_inputs(init) / runit))
            else:
                out = m.update()
            if isinstance(m, Neuron) and not getattr(m, '_relays_multiplicity', False):
                val = (jnp.asarray(u.get_mantissa(out)) >= 0.5).astype(dftype)
            else:
                # Generators and multiplicity-relaying neurons (parrot_neuron)
                # keep their raw per-step count instead of a binarised spike.
                val = jnp.asarray(u.get_mantissa(out), dtype=dftype)
            holder.spk.value = val

    def simulate(self, duration, *, dt=None) -> SimulationResult:
        """Run for ``duration`` and return recorded spikes and analog traces.

        Spike recorders are stacked as ``(n_steps, n_recorded)`` arrays; analog
        recorders (``voltmeter`` / ``multimeter``) tap their target population's
        State each step (after the update) into ``(n_steps, n_recorded)`` traces
        keyed by recordable. The run's time axis is exposed as ``res.times``.
        """
        import brainstate.transform as transform
        if dt is None:
            dt = self._dt
        brainstate.nn.init_all_states(self)
        times = u.math.arange(0.0 * u.get_unit(dt), duration, dt)
        indices = u.math.arange(times.size)
        taps = dict(self._taps)
        analog = dict(self._analog_taps)
        weight_taps = dict(self._weight_taps)

        def step(t, i):
            with brainstate.environ.context(t=t, i=i):
                self.update(t)
                spk_out = {rid: getattr(self, f'_holder_{sid}').spk.value[idx]
                           for rid, (sid, idx) in taps.items()}
                ana_out = {}
                for rid, (sid, idx, names) in analog.items():
                    pop = getattr(self, f'_node_{sid}')
                    for name in names:
                        ana_out[SimulationResult._trace_key(rid, name)] = _read_recordable(pop, name)[idx]
                # weight taps read the projection's (post-update) weight State
                w_out = {rid: proj.weight.value for rid, proj in weight_taps.items()}
                return spk_out, ana_out, w_out

        stacked_spk, stacked_ana, stacked_w = transform.for_loop(step, times, indices)
        recordings = {rid: jnp.asarray(stacked_spk[rid]) for rid in taps}
        traces = {key: stacked_ana[key] for key in stacked_ana}
        weights = {}
        for rid, proj in weight_taps.items():
            arr = jnp.asarray(stacked_w[rid])          # (T, E) mantissa
            unit = getattr(proj, '_w_unit', u.UNITLESS)
            weights[rid] = arr if unit is u.UNITLESS else u.maybe_decimal(arr * unit)
        return SimulationResult(recordings, duration, dt, traces=traces, times=times,
                                weights=weights)
