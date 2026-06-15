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
import jax
import jax.numpy as jnp
import saiunit as u

from brainpy_state._base import Neuron
from brainpy_state._nest.ac_generator import ac_generator as _ac_generator
from brainpy_state._nest.dc_generator import dc_generator as _dc_generator
from brainpy_state._nest.multimeter import multimeter as _multimeter
from brainpy_state._nest.noise_generator import noise_generator as _noise_generator
from brainpy_state._nest.sic_connection import sic_connection as _sic_connection
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
    """Total post-synaptic current ``I_syn``.

    NEST reports ``I_syn`` as the sum of every receptor's PSC. Models expose this
    differently: ``glif_psc`` keeps a single ``y2`` list of per-port PSC States
    (``glif_psc.cpp``: ``S_.I_syn_ += S_.y2_[i]``), while ``glif_psc_double_alpha``
    splits each port into fast/slow components and provides a ``get_I_syn()`` that
    sums them. Prefer the model's own ``get_I_syn()`` when present, else sum ``y2``.
    """
    get = getattr(pop, 'get_I_syn', None)
    if callable(get):
        return get()
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


def _adaptive_threshold(pop):
    """NEST ``V_th`` recordable for the multi-timescale adaptive-threshold family.

    ``mat2_psc_exp``/``amat2_psc_exp`` do not store the firing threshold; it is
    *computed* as ``V_th = omega + V_th_1 + V_th_2`` (mat2_psc_exp.cpp), where
    ``omega`` is a parameter (absolute mV) and ``V_th_1``/``V_th_2`` are States
    that jump by ``alpha_1``/``alpha_2`` on the model's own spikes and decay back.
    ``amat2_psc_exp`` adds the voltage-dependent component ``V_th_v``.

    Models that instead expose the threshold *directly* as a ``V_th`` State (e.g.
    ``aeif_psc_delta_clopath``) fall through to that State, so this single alias
    serves both the computed-threshold and stored-threshold conventions.
    """
    omega = getattr(pop, 'omega', None)
    if omega is not None and getattr(pop, 'V_th_1', None) is not None:
        total = omega + pop.V_th_1.value + pop.V_th_2.value
        v_th_v = getattr(pop, 'V_th_v', None)        # amat2 only
        if v_th_v is not None:
            total = total + v_th_v.value
        return total
    state = getattr(pop, 'V_th', None)               # clopath: threshold is a State
    if state is not None and hasattr(state, 'value'):
        return state.value
    raise KeyError(
        'V_th: population is neither a MAT model (omega + V_th_1 + V_th_2) nor '
        f'exposes a V_th State on {type(pop).__name__}'
    )


def _mc_comp(attr, idx):
    """Resolver for a multi-compartment recordable ``<attr>.<s|p|d>``.

    ``iaf_cond_alpha_mc`` stacks the three compartments on the *last axis* of a
    single State (SOMA=0, PROX=1, DIST=2); NEST spells the per-compartment
    recordables ``V_m.s``/``V_m.p``/``V_m.d``, ``g_ex.s``/…, ``g_in.s``/…. This
    returns a ``pop -> value`` reader selecting one compartment column, preserving
    the leading (neuron) axis so the analog tap can index by neuron ordinal.
    """
    def read(pop):
        state = getattr(pop, attr, None)
        if state is None:
            raise KeyError(
                f'{attr!r} compartment recordable: {type(pop).__name__} has no '
                f'{attr!r} stacked-compartment State'
            )
        return state.value[..., idx]

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
    # Astrocyte slow-inward current (aeif_cond_alpha_astro stores it on I_sic).
    'I_SIC': ('I_sic',),
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
    # izhikevich recovery variable (NEST ``U_m`` -> brainpy ``U``).
    'U_m': ('U',),
    # MAT family adaptive threshold (computed: omega + V_th_1 + V_th_2 [+ V_th_v]);
    # falls back to a directly-stored ``V_th`` State for non-MAT models (clopath).
    'V_th': _adaptive_threshold,
    # iaf_cond_alpha_mc per-compartment recordables (SOMA=0, PROX=1, DIST=2).
    'V_m.s': _mc_comp('V', 0), 'V_m.p': _mc_comp('V', 1), 'V_m.d': _mc_comp('V', 2),
    'g_ex.s': _mc_comp('g_ex', 0), 'g_ex.p': _mc_comp('g_ex', 1), 'g_ex.d': _mc_comp('g_ex', 2),
    'g_in.s': _mc_comp('g_in', 0), 'g_in.p': _mc_comp('g_in', 1), 'g_in.d': _mc_comp('g_in', 2),
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
        self._current_injectors = []          # (device, post_pop, post_idx, weight, key, comp, ncomp)
        self._gap_couplers = []               # (G, D, v_reader, post_pop, key) gap-junction couplers
        self._vt_nodes = []                   # volume_transmitter nodes (phase-0 update)
        self._proj_counter = itertools.count()
        self._connections = []                # (pre_pop, post_pop, model_name, proj), registration order

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
        # A neuron that integrates short-term plasticity *presynaptically* (declares
        # ``_emission_attr`` -- iaf_tum_2000's released efficacy ``spike_offset``)
        # also gets an emission holder. A TSODYKS connection delivers that graded
        # efficacy (captured in phase 2 alongside the binary spike) rather than
        # ``weight * spike``; the binary holder still serves every other connection.
        if getattr(mod, '_emission_attr', None) is not None:
            setattr(self, f'_emit_holder_{id(mod)}', _SpikeHolder(_flat_size(mod)))
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
                    # A ``mult_coupling`` rate pair returns its two sign-split
                    # (rate_ex/rate_in) projections as a list; flatten them in.
                    projs.extend(proj) if isinstance(proj, list) else projs.append(proj)
        if not projs:
            return None
        return projs[0] if len(projs) == 1 else projs

    def get_connections(self, source=None, target=None, synapse=None):
        """Enumerate realized synapses across projections (NEST ``GetConnections``).

        Returns a :class:`~brainpy_state._network._connection_introspection.SynapseCollection`
        over the edges matching the filters, letting you read and write weights /
        delays **without holding each projection handle**. The collection is a lazy
        view: weights and delays are re-read from the live projections on each
        :meth:`~brainpy_state._network._connection_introspection.SynapseCollection.get`,
        so a query made before :meth:`simulate` still reflects post-simulation
        evolved plastic weights.

        Parameters
        ----------
        source : NodeView, optional
            Keep only edges whose presynaptic neuron lies in this view (matched by
            population identity and population-local index). ``None`` keeps all.
        target : NodeView, optional
            Keep only edges whose postsynaptic neuron lies in this view. ``None``
            keeps all.
        synapse : str, optional
            Keep only projections with this synapse-model name (``'static_synapse'``
            for the static event path, else the plastic spec's class name such as
            ``'stdp_synapse'``). ``None`` keeps all.

        Returns
        -------
        SynapseCollection
            A filtered, lazy view over the matching edges (empty if none match).

        See Also
        --------
        connect : Build the projections this enumerates.

        Examples
        --------
        .. code-block:: python

           >>> import saiunit as u
           >>> from brainpy import state as bp
           >>> sim = bp.Simulator(dt=0.1 * u.ms)
           >>> exc = sim.create(bp.iaf_psc_exp, 4)
           >>> _ = sim.connect(exc, exc, rule=bp.all_to_all, weight=20. * u.pA,
           ...                 allow_autapses=False, comm='sparse')
           >>> conns = sim.get_connections(source=exc, target=exc)
           >>> len(conns)
           12
           >>> bool(u.math.allclose(conns.get('weight'), 20. * u.pA))
           True
        """
        from brainpy_state._network._connection_introspection import collect_connections
        return collect_connections(self._connections, source, target, synapse)

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

    def _resolve_stp_emission(self, pre_pop, post_pop, receptor_type, spike_holder,
                              comm='dense'):
        """Choose what a neuron->neuron static connection delivers per step.

        Most neurons emit a binary spike (the spike holder -> ``weight * spike``).
        A neuron that integrates a *presynaptic graded efficacy* declares
        ``_emission_attr`` (the State holding the per-step emission, 0 off spike)
        and ``_emission_receptor`` (the one NEST receptor that carries it). Over
        that receptor the connection delivers ``weight * emission`` instead of
        ``weight * spike``; the DEFAULT/``None`` connection -- and every other
        receptor and every non-emitting model -- still delivers the binary spike
        with its receptor unchanged.

        A third, *receptorless* class emits **continuously**: a rate neuron
        declares ``_emission_continuous=True`` and ``_emission_attr='rate'`` (or
        ``'phi_rate'``). Its static connection delivers ``weight * rate`` into the
        post's DEFAULT delta channel every step (read back via
        ``sum_delta_inputs``) -- there is no NEST receptor, so this branch is
        resolved first and ignores ``receptor_type``. The two receptor-gated
        spike-offset emitters are:

        * ``iaf_tum_2000`` -- released short-term-plasticity efficacy
          ``spike_offset = u * x`` over its TSODYKS receptor. The post is
          single-port, so ``receptor_type`` collapses to ``None`` (a plain
          delta-input delivery; the post integrates ``weight * efficacy`` as its
          excitatory PSC) and the post must be the same model.
        * ``iaf_bw_2001`` -- presynaptic NMDA gate increment
          ``spike_offset = k0 + k1 * s_NMDA_pre`` over its ``NMDA`` receptor. The
          post is multi-port, so ``receptor_type`` is *preserved* and the graded
          deposit is routed into the post's NMDA delta channel (via
          ``delta_label_for_receptor``); AMPA/GABA on other connections stay
          binary and land in their own channels.

        The graded emission rides the **dense** matmul (``x @ W``); the sparse
        event path binarizes the presynaptic value, so ``comm='sparse'`` over an
        emitting receptor is rejected.

        Parameters
        ----------
        pre_pop, post_pop : Neuron
            The presynaptic and postsynaptic populations.
        receptor_type : int or str or None
            The connection's NEST receptor type as passed to :meth:`connect`.
        spike_holder : _SpikeHolder
            The presynaptic population's binary-spike holder (the default source).
        comm : str, default 'dense'
            The connection's communication mode. ``'sparse'`` is rejected for the
            graded-emission path because it binarizes the presynaptic value.

        Returns
        -------
        tuple
            ``(pre_spike_reader, effective_receptor_type)`` -- the callable the
            :class:`EventProjection` reads each step, and the receptor type to
            build it with (``None`` for the single-port efficacy collapse, the
            original receptor for the multi-port routed path).

        Raises
        ------
        ValueError
            If the graded emission would ride ``comm='sparse'`` (binarized); if a
            single-port efficacy connection targets a post of a different model
            (the efficacy is delivered as that model's PSC, so the post must be the
            same model); or if the emission holder is absent.
        """
        emit_attr = getattr(pre_pop, '_emission_attr', None)
        # Continuous graded emitter (rate neurons): the connection delivers
        # ``weight * emission`` (e.g. ``weight * rate``) into the post's DEFAULT
        # delta channel every step -- a *receptorless* continuous coupling that
        # rides the ordinary delta seam (read back via ``sum_delta_inputs``),
        # not a NEST receptor. It is resolved first and independently of
        # ``receptor_type`` (rate connections carry none), so it cannot collide
        # with the receptor-gated STP/NMDA branches below.
        if emit_attr is not None and getattr(pre_pop, '_emission_continuous', False):
            emit_holder = getattr(self, f'_emit_holder_{id(pre_pop)}', None)
            if emit_holder is None:
                raise ValueError(
                    f'{type(pre_pop).__name__} declares _emission_continuous but no '
                    'emission holder was allocated by create().')
            if comm == 'sparse':
                raise ValueError(
                    f'a continuous rate connection from {type(pre_pop).__name__} '
                    f'delivers weight * {emit_attr} and must ride the dense matmul; '
                    'comm="sparse" binarizes the presynaptic value. Use comm="dense".')
            return _holder_reader(emit_holder), None
        emit_receptor = getattr(pre_pop, '_emission_receptor', None)
        if emit_receptor is None:
            emit_receptor = getattr(pre_pop, 'RECEPTOR_TYPES', {}).get('TSODYKS')
        if emit_attr is None or receptor_type is None or receptor_type != emit_receptor:
            return _holder_reader(spike_holder), receptor_type
        emit_holder = getattr(self, f'_emit_holder_{id(pre_pop)}', None)
        if emit_holder is None:
            raise ValueError(
                f'{type(pre_pop).__name__} declares _emission_attr={emit_attr!r} but '
                'no emission holder was allocated by create().')
        if comm == 'sparse':
            raise ValueError(
                f'a graded-emission connection from {type(pre_pop).__name__} over '
                f'receptor_type={receptor_type} delivers weight * {emit_attr} and '
                'must ride the dense matmul; comm="sparse" binarizes the '
                'presynaptic value. Use comm="dense".')
        # Multi-port post routes by receptor into a named delta channel: keep the
        # receptor so EventProjection deposits the graded value into the right
        # channel (e.g. iaf_bw_2001 NMDA).
        if hasattr(post_pop, 'delta_label_for_receptor') or hasattr(post_pop, 'n_receptors'):
            return _holder_reader(emit_holder), receptor_type
        # Single-port post (iaf_tum_2000): the efficacy IS the PSC; collapse the
        # receptor to None and require the same integrate-and-fire-with-STP model.
        if not isinstance(post_pop, type(pre_pop)):
            raise ValueError(
                f'a graded-efficacy (receptor_type={receptor_type}) connection from '
                f'{type(pre_pop).__name__} delivers presynaptic efficacy as the post '
                f'PSC and requires a {type(pre_pop).__name__} post; got '
                f'{type(post_pop).__name__}.')
        return _holder_reader(emit_holder), None

    @staticmethod
    def _is_continuous_rate(pop):
        """Whether ``pop`` couples through seam-(H) continuous graded emission."""
        return (getattr(pop, '_emission_continuous', False)
                and getattr(pop, '_emission_attr', None) is not None)

    def _check_rate_phi_homogeneity(self, pre_pop, post_pop):
        r"""Guard a rate->rate connection against a φ / summation-mode mismatch.

        A continuous rate connection emits the *sender's* per-step value and the
        *receiver* integrates it through its own coupling path. With
        ``linear_summation=True`` the sender emits the raw ``rate`` and the receiver
        applies its own φ to the summed input, so the two φ may differ freely; with
        ``linear_summation=False`` the sender emits ``φ_pre(rate)`` and the receiver
        adds it *as-is* (it would otherwise have applied ``φ_post``), which is exact
        only for a homogeneous φ. The summation modes must also agree --- a raw
        ``rate`` integrated where ``φ(rate)`` was expected (or vice versa) is silently
        wrong. Both conditions are enforced here and raise at ``connect()`` time
        (the user's "guard to homogeneous-φ" decision, spec §3.3).

        A rate -> non-rate connection (a rate neuron driving a spiking/current
        target) carries no φ contract and is left unchecked.
        """
        if not self._is_continuous_rate(post_pop):
            return
        pre_ls = bool(getattr(pre_pop, 'linear_summation', True))
        post_ls = bool(getattr(post_pop, 'linear_summation', True))
        if pre_ls != post_ls:
            raise ValueError(
                f'rate connection {type(pre_pop).__name__} -> {type(post_pop).__name__}: '
                f'linear_summation must match on both sides (pre={pre_ls}, post={post_ls}). '
                'The receiver integrates the raw rate or φ(rate) the sender emits, and the '
                'two summation modes are not interchangeable.')
        if not pre_ls:
            sig_pre = getattr(pre_pop, '_phi_signature', None)
            sig_post = getattr(post_pop, '_phi_signature', None)
            if sig_pre != sig_post:
                raise ValueError(
                    f'rate connection {type(pre_pop).__name__} -> {type(post_pop).__name__} '
                    'with linear_summation=False requires a homogeneous input nonlinearity φ: '
                    'the sender emits φ(rate) and the receiver integrates it directly, which is '
                    f'exact only when both φ agree (got pre φ={sig_pre!r}, post φ={sig_post!r}). '
                    'Use linear_summation=True so the receiver applies its own φ, or match the '
                    'models and their gain parameters.')

    @staticmethod
    def _sign_split_weight(weight):
        """Split a connection weight into ``(max(W,0), min(W,0))`` by sign.

        The two halves sum back to ``W``; they feed the ``mult_coupling`` excitatory
        and inhibitory channels (spec §3.2). The weight must be concrete (a scalar or
        array) --- a random initializer cannot be sign-split, so ``mult_coupling``
        rejects callable / absent weights.
        """
        if weight is None or callable(weight):
            raise ValueError(
                'mult_coupling rate connections sign-split the weight into the '
                'rate_ex/rate_in channels and require a concrete weight (scalar or '
                'array); a callable initializer or an absent weight cannot be split.')
        return jnp.maximum(weight, 0.0), jnp.minimum(weight, 0.0)

    def _build_rate_dual_channel(self, pre_spike, pre_pop, pre_seg, post_pop, post_seg,
                                 rule, weight, delay, comm, seed, ordinal):
        r"""Build the two sign-split labelled projections for ``mult_coupling`` (spec §3.2).

        NEST's multiplicative rate coupling splits the presynaptic drive by weight
        sign --- :math:`\sum_\mathrm{ex} w\,r` over the excitatory (``w>0``) edges and
        :math:`\sum_\mathrm{in} w\,r` over the inhibitory (``w<0``) edges --- and scales
        each partial sum by a receiver-state factor ``H_ex``/``H_in``. The split is
        realised here as two ordinary rate projections reading the *same* presynaptic
        emission: one carrying ``max(W,0)`` into the post's ``'rate_ex'`` delta channel,
        one carrying ``min(W,0)`` into ``'rate_in'``. The receiver reads them back with
        ``sum_delta_inputs(label='rate_ex'|'rate_in')`` and applies ``H_ex``/``H_in``.

        Both halves share one connectivity sample (the same derived seed), so a random
        rule realises the *same* edge set split only by weight sign.
        """
        w_ex, w_in = self._sign_split_weight(weight)
        conn_seed = self._derive_seed(seed, ordinal)
        proj_ords = (ordinal, next(self._proj_counter))
        projs = []
        for (w_part, label), p_ord in zip(((w_ex, 'rate_ex'), (w_in, 'rate_in')), proj_ords):
            proj = EventProjection(
                pre_spike=pre_spike, n_pre_pop=_flat_size(pre_pop),
                pre_local_idx=pre_seg.indices, post=post_pop,
                post_local_idx=post_seg.indices, rule=rule, weight=w_part,
                delay=delay, comm=comm, channel_label=label,
                pre_is_post=(pre_pop is post_pop), seed=conn_seed)
            self._connections.append((pre_pop, post_pop, 'static_synapse', proj))
            setattr(self, f'_proj_{p_ord}', proj)
            projs.append(proj)
        return projs

    def _build_siegert_diffusion(self, pre_seg, post_seg, rule, weight, delay,
                                 synapse, comm, seed, ordinal):
        r"""Route a ``diffusion_connection`` as two labelled seam-(H) deposits (goal 15c).

        NEST delivers a single ``DiffusionConnectionEvent`` carrying the presynaptic
        rate :math:`r_j`, from which the target ``siegert_neuron`` accumulates a drift
        :math:`g_\mu r_j \to \mu` and a variance :math:`g_\sigma r_j \to \sigma^2`. Here
        that one event becomes two ordinary rate projections reading the *same*
        presynaptic emission (the source's seam-(H) ``rate`` holder): one carrying
        ``drift_factor`` into the post's ``'diffusion_mu'`` delta channel, one carrying
        ``diffusion_factor`` into ``'diffusion_sigma2'``. The target reads them back with
        ``sum_delta_inputs(label='diffusion_mu' | 'diffusion_sigma2')`` -- distinct
        labels so :math:`\mu` and :math:`\sigma^2` never cross-contaminate (a default,
        unlabelled read would sum both). Both halves share one connectivity sample.

        The source must be a continuous-rate emitter (e.g. ``siegert_neuron``); the
        deposit rides the dense matmul, so ``comm='sparse'`` is rejected; and the
        connection carries no delay (the one-step coupling lag is the seam holder's,
        matching NEST ``min_delay=1``).
        """
        if isinstance(pre_seg, _GenSegment):
            raise ValueError(
                'a diffusion_connection source must be a continuous-rate neuron '
                '(e.g. siegert_neuron), not a generator.')
        pre_pop = pre_seg.population
        post_pop = post_seg.population
        if not self._is_continuous_rate(pre_pop):
            raise ValueError(
                f'a diffusion_connection requires a continuous-rate source '
                f'(_emission_continuous + _emission_attr); {type(pre_pop).__name__} '
                'does not emit a rate.')
        if comm == 'sparse':
            raise ValueError(
                'a diffusion_connection delivers drift_factor * rate / diffusion_factor '
                '* rate over the dense matmul; comm="sparse" binarizes the rate. '
                'Use comm="dense".')
        if delay is not None:
            raise ValueError('diffusion_connection has no delay.')
        if weight is not None:
            raise ValueError(
                'diffusion_connection has no weight; use drift_factor / diffusion_factor.')
        holder = getattr(self, f'_holder_{id(pre_pop)}')
        pre_spike, _ = self._resolve_stp_emission(pre_pop, post_pop, None, holder, comm)
        drift = float(synapse.drift_factor)
        diffusion = float(synapse.diffusion_factor)
        conn_seed = self._derive_seed(seed, ordinal)
        proj_ords = (ordinal, next(self._proj_counter))
        projs = []
        for (factor, label), p_ord in zip(
                ((drift, 'diffusion_mu'), (diffusion, 'diffusion_sigma2')), proj_ords):
            proj = EventProjection(
                pre_spike=pre_spike, n_pre_pop=_flat_size(pre_pop),
                pre_local_idx=pre_seg.indices, post=post_pop,
                post_local_idx=post_seg.indices, rule=rule, weight=factor,
                delay=None, comm=comm, channel_label=label,
                pre_is_post=(pre_pop is post_pop), seed=conn_seed)
            self._connections.append((pre_pop, post_pop, 'diffusion_connection', proj))
            setattr(self, f'_proj_{p_ord}', proj)
            projs.append(proj)
        return projs
    # -- gap junctions (goal 15b) ------------------------------------------
    @staticmethod
    def _is_gap_synapse(synapse):
        """Whether a ``connect(synapse=...)`` argument selects gap-junction coupling.

        ``gap_junction`` is the only synapse model that requires symmetric
        connectivity (``REQUIRES_SYMMETRIC=True``); the Simulator detects it by that
        marker and builds its own explicit-lag difference-deposit coupler, never
        instantiating the reference model or its waveform-relaxation machinery.
        """
        return bool(getattr(synapse, 'REQUIRES_SYMMETRIC', False))

    @staticmethod
    def _gap_conductance(weight):
        """Resolve a gap connection weight to a scalar conductance (nS mantissa).

        The gap weight is the coupling conductance ``g`` (nS); the difference deposit
        ``g * (V_pre - V_post)`` needs one concrete scalar shared by every gap edge.
        A unit ``Quantity`` is read in nS; a bare scalar is taken as nS. Per-edge /
        random / callable gap weights are out of scope (like sparse gaps).
        """
        if weight is None or callable(weight):
            raise ValueError(
                'gap_junction coupling requires a concrete scalar conductance weight '
                '(nS); a callable initializer or an absent weight cannot build the '
                'gap matrix G.')
        w = weight
        if isinstance(w, u.Quantity):
            w = u.get_mantissa(w / u.nS)       # nS-valued; raises if not a conductance
        w = jnp.asarray(w)
        if w.ndim != 0:
            raise ValueError(
                'gap_junction coupling uses a scalar conductance g (one value for all '
                f'gap edges); got a non-scalar weight of shape {tuple(w.shape)}. '
                'Per-edge gap weights are out of scope.')
        return float(w)

    @staticmethod
    def _gap_current(G, D, v_pre, v_post):
        r"""The explicit-lag gap difference current ``I_gap = G @ V_pre - D * V_post``.

        ``G`` (nS) is the dense symmetric gap-conductance matrix, ``D = rowsum(G)``
        the per-neuron total gap conductance, and ``v_pre`` / ``v_post`` (mV) the
        one-step-lagged pre / post membrane voltages. For a recurrent gap the two
        voltage vectors are identical, giving ``(G - diag(D)) @ V`` --- the negated
        graph Laplacian of the gap graph applied to ``V``. ``nS * mV = pA``, so the
        result is the gap current in pA; at equal voltages it is exactly zero.
        """
        return G @ v_pre - D * v_post

    def _build_gap_coupling(self, pre_seg, post_seg, rule, weight, delay, comm,
                            allow_autapses, allow_multapses, seed, ordinal):
        r"""Register a recurrent gap-junction coupler (the difference deposit, 15b).

        Builds the dense **symmetric** gap-conductance matrix ``G`` (nS) over the
        post population from the rule's realized edges (both directions materialized,
        hollow diagonal), caches ``D = rowsum(G)`` and the post's V emission-holder
        reader, and appends a coupler driven each step in :meth:`update`: it deposits
        ``I_gap = G @ V[n-1] - D * V[n-1]`` into the post's current channel
        (``add_current_input`` -> ``sum_current_inputs``) under the substrate's
        one-step pipeline lag (the WFR seed, cluster 15a). No waveform relaxation;
        the reference ``gap_junction`` WFR class is never used.

        Raises
        ------
        ValueError
            If pre and post are different populations (gap coupling is recurrent), a
            delay is given (gap is instantaneous), ``comm='sparse'`` (the gap ``G@V``
            is a dense matmul), the post does not emit ``V`` (``_emission_attr='V'``),
            or the conductance weight is not a concrete scalar.
        """
        pre_pop = pre_seg.population
        post_pop = post_seg.population
        if pre_pop is not post_pop:
            raise ValueError(
                'gap_junction coupling is recurrent (electrical coupling within one '
                'population); connect a population to itself, got '
                f'{type(pre_pop).__name__} -> {type(post_pop).__name__}.')
        if delay is not None:
            raise ValueError(
                'gap_junction connections carry no delay (instantaneous electrical '
                'coupling); remove the delay= argument.')
        if comm != 'dense':
            raise ValueError(
                "gap_junction coupling rides comm='dense' (the gap G @ V matmul); "
                "comm='sparse' binarizes the presynaptic voltage. Use comm='dense'.")
        emit_holder = getattr(self, f'_emit_holder_{id(post_pop)}', None)
        if emit_holder is None:
            raise ValueError(
                f'{type(post_pop).__name__} does not emit its membrane voltage for '
                "gap coupling (it must declare _emission_attr='V'); no emission "
                'holder was allocated by create().')
        g = self._gap_conductance(weight)
        n_pop = _flat_size(post_pop)
        n_seg = int(post_seg.indices.shape[0])
        spec = rule.sample(n_seg, n_seg,
                           key=jax.random.key(self._derive_seed(seed, ordinal)),
                           pre_is_post=True, allow_autapses=allow_autapses,
                           allow_multapses=allow_multapses)
        # Realized edges -> a hollow, symmetric boolean adjacency over the full post
        # population (both directions materialized: NEST's make_symmetric / the
        # all_to_all bidirectionality), then scaled by the scalar conductance.
        gpre = jnp.asarray(post_seg.indices)[spec.pre_idx]
        gpost = jnp.asarray(post_seg.indices)[spec.post_idx]
        A = jnp.zeros((n_pop, n_pop), dtype=bool).at[gpre, gpost].set(True)
        A = (A | A.T) & ~jnp.eye(n_pop, dtype=bool)
        G = A.astype(brainstate.environ.dftype()) * g            # (n_pop, n_pop) nS
        D = G.sum(axis=1)                                         # rowsum (n_pop,) nS
        self._gap_couplers.append((G, D, _holder_reader(emit_holder), post_pop,
                                   f'gap_inj_{ordinal}'))

    def _connect_pair(self, pre_seg, post_seg, rule, weight, delay,
                      allow_autapses, allow_multapses, seed, comm='dense',
                      receptor_type=None, synapse=None, vt=None):
        ordinal = next(self._proj_counter)
        post_pop = post_seg.population
        # Diffusion coupling (siegert mean-field): a diffusion_connection is not a
        # plastic projection -- the Simulator routes it as a dual-channel seam deposit
        # (drift -> mu, diffusion -> sigma^2). Dispatch before the plastic path.
        if synapse is not None and getattr(synapse, '_IS_DIFFUSION', False):
            return self._build_siegert_diffusion(
                pre_seg, post_seg, rule, weight, delay, synapse, comm, seed, ordinal)
        # Gap-junction coupling (goal 15b) is dispatched on the gap synapse model
        # before the plastic/static paths: it builds its own explicit-lag difference-
        # deposit coupler (no EventProjection, no rule kernel, no WFR).
        if synapse is not None and self._is_gap_synapse(synapse):
            self._build_gap_coupling(pre_seg, post_seg, rule, weight, delay, comm,
                                     allow_autapses, allow_multapses, seed, ordinal)
            return None
        # One-way astrocyte->neuron slow-inward-current edge (NEST sic_connection):
        # dispatched before the plastic path, it builds an ``as_current``
        # EventProjection reading the astrocyte's emission holder.
        if isinstance(synapse, _sic_connection):
            return self._connect_sic(pre_seg, post_seg, rule, weight, delay,
                                     allow_autapses, allow_multapses, seed, ordinal,
                                     synapse, comm)
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
                self._wire_current_injector(pre_seg, post_seg, weight, ordinal,
                                            receptor_type)
                return
            n = int(post_seg.indices.shape[0])
            params = dict(pre_seg.spec.params)
            if 'rng_seed' in inspect.signature(pre_seg.spec.model_cls.__init__).parameters:
                params['rng_seed'] = self._derive_seed(params.get('rng_seed'), ordinal)
            gen = pre_seg.spec.model_cls(n, **params)
            source_pop = gen
            setattr(self, f'_node_{id(gen)}', gen)
            holder = _SpikeHolder(n)
            setattr(self, f'_holder_{id(gen)}', holder)
            if synapse is not None:
                proj = proj_cls(
                    pre_spike=_holder_reader(holder), n_pre_pop=n,
                    pre_local_idx=jnp.arange(n), post=post_pop,
                    post_local_idx=post_seg.indices, n_post_pop=_flat_size(post_pop),
                    post_spike=post_reader, rule=self._resolve_synapse(synapse, weight, delay),
                    conn=one_to_one, seed=seed, receptor_type=receptor_type, **plastic_extra)
            else:
                proj = EventProjection(
                    pre_spike=_holder_reader(holder), n_pre_pop=n,
                    pre_local_idx=jnp.arange(n), post=post_pop,
                    post_local_idx=post_seg.indices, rule=one_to_one, weight=weight,
                    delay=delay, receptor_type=receptor_type, seed=seed)
        else:
            pre_pop = pre_seg.population
            source_pop = pre_pop
            holder = getattr(self, f'_holder_{id(pre_pop)}')
            if synapse is not None:
                proj = proj_cls(
                    pre_spike=_holder_reader(holder), n_pre_pop=_flat_size(pre_pop),
                    pre_local_idx=pre_seg.indices, post=post_pop,
                    post_local_idx=post_seg.indices, n_post_pop=_flat_size(post_pop),
                    post_spike=post_reader, rule=self._resolve_synapse(synapse, weight, delay),
                    conn=rule, pre_is_post=(pre_pop is post_pop),
                    allow_autapses=allow_autapses, allow_multapses=allow_multapses,
                    seed=self._derive_seed(seed, ordinal), receptor_type=receptor_type,
                    **plastic_extra)
            else:
                pre_spike, eff_receptor = self._resolve_stp_emission(
                    pre_pop, post_pop, receptor_type, holder, comm)
                # Continuous rate coupling: enforce the φ / summation-mode contract,
                # and split into the labelled rate_ex/rate_in channels when the
                # receiver uses multiplicative coupling (spec §3.2-3.3).
                if self._is_continuous_rate(pre_pop):
                    self._check_rate_phi_homogeneity(pre_pop, post_pop)
                    if getattr(post_pop, '_use_mult_coupling', False):
                        return self._build_rate_dual_channel(
                            pre_spike, pre_pop, pre_seg, post_pop, post_seg,
                            rule, weight, delay, comm, seed, ordinal)
                proj = EventProjection(
                    pre_spike=pre_spike, n_pre_pop=_flat_size(pre_pop),
                    pre_local_idx=pre_seg.indices, post=post_pop,
                    post_local_idx=post_seg.indices, rule=rule, weight=weight,
                    delay=delay, comm=comm, receptor_type=eff_receptor,
                    pre_is_post=(pre_pop is post_pop),
                    allow_autapses=allow_autapses, allow_multapses=allow_multapses,
                    seed=self._derive_seed(seed, ordinal))
        # Register for get_connections (NEST GetConnections analogue). The static
        # event path is 'static_synapse'; a plastic path takes its spec class name.
        model_name = (type(proj.rule).__name__ if isinstance(proj, EventPlasticProj)
                      else 'static_synapse')
        self._connections.append((source_pop, post_pop, model_name, proj))
        setattr(self, f'_proj_{ordinal}', proj)
        return proj

    def _connect_sic(self, pre_seg, post_seg, rule, weight, delay,
                     allow_autapses, allow_multapses, seed, ordinal, synapse, comm):
        r"""Wire a one-way astrocyte->neuron slow-inward-current (SIC) connection.

        The astrocyte (``astrocyte_lr_1994``) is the sender: phase 2 captures its
        per-step graded ``SIC`` into the emission holder. The ``sic_connection``
        deposits ``weight·SIC`` into the receiver's (``aeif_cond_alpha_astro``)
        labelled ``'I_SIC'`` *current* channel through an ``as_current``
        :class:`EventProjection`. The edge is one-way (a SICEvent, no back-channel),
        and the NEST sender/receiver contract is enforced by
        :meth:`sic_connection.check_connection`.

        Timing follows the deleted host queue's ``base_offset = delay_steps - 1``:
        ``delay_steps=1`` (the NEST default / minimum delay) rides the substrate's
        intrinsic pipeline latency with no extra :class:`InputDelay`; a larger
        ``delay_steps`` adds ``(delay_steps - 1)`` steps. The small residual offset
        vs NEST is absorbed by ``align_steps`` in parity.

        Parameters
        ----------
        pre_seg, post_seg : _Segment
            The presynaptic astrocyte and postsynaptic neuron segments.
        rule : ConnRule
            Connectivity rule (fan-out from astrocytes to neurons).
        weight : ArrayLike or None
            Connect-level weight override (unitless); ``None`` uses the spec's
            ``synapse.weight``.
        delay : Quantity or None
            Connect-level delay override; ``None`` uses ``delay_steps``.
        allow_autapses, allow_multapses : bool
            Passed through to the connectivity sampler.
        seed : int or None
            Base seed for the connectivity draw.
        ordinal : int
            Projection ordinal (registration key).
        synapse : sic_connection
            The connection spec carrying ``weight`` / ``delay_steps`` and the
            sender/receiver enforcement.
        comm : str
            Communication mode; ``'sparse'`` is rejected (a graded current must
            ride the dense matmul).

        Returns
        -------
        EventProjection
            The ``as_current`` projection delivering ``weight·SIC`` into ``'I_SIC'``.

        Raises
        ------
        ValueError
            If the sender/receiver models are not the supported SIC pair, if the
            source is a deferred generator, if ``comm='sparse'``, or if no emission
            holder was allocated for the astrocyte.
        """
        post_pop = post_seg.population
        pre_name = (pre_seg.spec.model_cls.__name__ if isinstance(pre_seg, _GenSegment)
                    else type(pre_seg.population).__name__)
        # Enforce the NEST sender/receiver contract (astrocyte_lr_1994 sends a
        # SICEvent; aeif_cond_alpha_astro handles it). Raises ValueError otherwise.
        synapse.check_connection(pre_name, type(post_pop).__name__)
        if isinstance(pre_seg, _GenSegment):
            raise ValueError(
                'a sic_connection source must be an astrocyte_lr_1994 population, '
                'not a deferred generator.')
        if comm == 'sparse':
            raise ValueError(
                "a sic_connection delivers a graded current and must ride "
                "comm='dense'; comm='sparse' binarises the presynaptic value.")
        pre_pop = pre_seg.population
        emit_holder = getattr(self, f'_emit_holder_{id(pre_pop)}', None)
        if emit_holder is None:
            raise ValueError(
                f'{type(pre_pop).__name__} declares _emission_attr but no emission '
                'holder was allocated by create().')
        # Resolve weight (connect override else the spec weight) and delay (connect
        # override else ``(delay_steps - 1)`` extra steps). The SIC value and weight
        # are unitless; the neuron attaches pA on write-back into I_sic.
        eff_weight = weight if weight is not None else synapse.weight
        if delay is not None:
            eff_delay = delay
        elif int(getattr(synapse, 'delay_steps', 1)) > 1:
            eff_delay = (int(synapse.delay_steps) - 1) * brainstate.environ.get_dt()
        else:
            eff_delay = None
        label = getattr(pre_pop, '_emission_current_label', 'I_SIC')
        proj = EventProjection(
            pre_spike=_holder_reader(emit_holder), n_pre_pop=_flat_size(pre_pop),
            pre_local_idx=pre_seg.indices, post=post_pop,
            post_local_idx=post_seg.indices, rule=rule, weight=eff_weight,
            delay=eff_delay, comm='dense', as_current=True, channel_label=label,
            pre_is_post=False, allow_autapses=allow_autapses,
            allow_multapses=allow_multapses, seed=self._derive_seed(seed, ordinal))
        self._connections.append((pre_pop, post_pop, 'sic_connection', proj))
        setattr(self, f'_proj_{ordinal}', proj)
        return proj

    def _wire_current_injector(self, pre_seg, post_seg, weight, ordinal, receptor_type=None):
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

        ``receptor_type`` selects a target compartment for a multi-compartment post
        (``iaf_cond_alpha_mc``: 7=soma, 8=proximal, 9=distal); the resolved
        ``(comp, ncomp)`` makes the injection land in one compartment instead of
        broadcasting across all of them. It is ``(None, None)`` for an ordinary
        single-compartment post.
        """
        post_pop = post_seg.population
        comp, ncomp = self._resolve_current_compartment(post_pop, receptor_type)
        n = int(post_seg.indices.shape[0])
        params = dict(pre_seg.spec.params)
        if 'seed' in inspect.signature(pre_seg.spec.model_cls.__init__).parameters:
            params['seed'] = self._derive_seed(params.get('seed'), ordinal)
        device = pre_seg.spec.model_cls(n, **params)
        setattr(self, f'_node_{id(device)}', device)
        key = f'cur_inj_{ordinal}'
        self._current_injectors.append(
            (device, post_pop, post_seg.indices, weight, key, comp, ncomp))

    @staticmethod
    def _resolve_current_compartment(post_pop, receptor_type):
        """Resolve a compartmental post's current ``receptor_type`` to ``(comp, ncomp)``.

        A multi-compartment post (``iaf_cond_alpha_mc``) exposes
        ``current_compartment_for_receptor`` and ``NCOMP``; its current generators MUST
        name a target compartment via ``receptor_type`` (7=soma, 8=proximal, 9=distal)
        so the injected current lands in one compartment instead of broadcasting across
        all of them. A non-compartmental post takes no current ``receptor_type``.

        Parameters
        ----------
        post_pop : Dynamics
            The postsynaptic population.
        receptor_type : int or None
            The connection's NEST receptor type.

        Returns
        -------
        tuple
            ``(comp, ncomp)`` --- the target compartment index and compartment count for
            a compartmental post, or ``(None, None)`` for an ordinary single-compartment
            post.

        Raises
        ------
        ValueError
            If a compartmental post is given no ``receptor_type``, a non-compartmental
            post is given one, or the receptor type is not a valid current receptor.
        """
        resolver = getattr(post_pop, 'current_compartment_for_receptor', None)
        if resolver is None:
            if receptor_type is not None:
                raise ValueError(
                    f'receptor_type={receptor_type} was given for current injection into '
                    f'{type(post_pop).__name__}, which has no current receptors.')
            return None, None
        if receptor_type is None:
            raise ValueError(
                f'{type(post_pop).__name__} requires a current receptor_type for current '
                f'input (soma_curr=7, proximal_curr=8, distal_curr=9); none was given.')
        comp = resolver(receptor_type)            # raises for spike / out-of-range
        return comp, int(post_pop.NCOMP)

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

    @staticmethod
    def _scatter_current_compartment(cur, pop, idx, comp, ncomp):
        """Place a device's ``(n,)`` current into one compartment column of ``(n_pop, ncomp)``.

        Like :meth:`_scatter_current`, but for a multi-compartment post: the current is
        scattered into column ``comp`` of an otherwise-zero ``(n_pop, ncomp)`` frame, so
        the neuron's ``sum_current_inputs`` (which reads a ``(*varshape, ncomp)`` array)
        sees the current only in the targeted compartment, never broadcast across all of
        them.
        """
        n_pop = _flat_size(pop)
        mant = u.get_mantissa(cur)
        base = jnp.zeros((n_pop, ncomp), dtype=mant.dtype)
        return u.maybe_decimal(base.at[idx, comp].add(mant) * u.get_unit(cur))

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
        for device, pop, idx, weight, key, comp, ncomp in self._current_injectors:
            cur = device.update()
            if weight is not None:
                cur = cur * weight
            if comp is None:
                pop.add_current_input(key, self._scatter_current(cur, pop, idx))
            else:
                pop.add_current_input(
                    key, self._scatter_current_compartment(cur, pop, idx, comp, ncomp))
        # 1c) gap-junction couplers deposit the explicit-lag difference current
        #     I_gap = G @ V[n-1] - D * V[n-1] (pA) into the post's current channel.
        #     V[n-1] is the previous step's voltage from the V emission holder (the
        #     one-step pipeline lag = NEST's use_wfr=False seed, cluster 15a); the
        #     deposit rides the same current ring buffer as the device injectors.
        #     For a recurrent gap V_pre and V_post are the same population's V.
        for G, D, v_reader, pop, key in self._gap_couplers:
            v = jnp.asarray(v_reader())                  # (n_pop,) mV, one step lagged
            pop.add_current_input(key, self._gap_current(G, D, v, v) * u.pA)
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
            # Presynaptic STP: capture this step's released efficacy (graded, 0 off
            # spike) into the emission holder, time-aligned with the binary spike so a
            # TSODYKS connection delivers it with the same latency as a plain spike.
            emit_attr = getattr(m, '_emission_attr', None)
            if emit_attr is not None:
                emit = getattr(self, f'_emit_holder_{id(m)}', None)
                if emit is not None:
                    emit.spk.value = jnp.asarray(
                        u.get_mantissa(getattr(m, emit_attr).value), dtype=dftype)

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
