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

import itertools
from typing import Optional

import brainstate
import jax.numpy as jnp
import saiunit as u

from brainpy_state._base import Neuron
from brainpy_state._nest.spike_recorder import spike_recorder as _spike_recorder
from brainpy_state._network._event_proj import EventProjection
from brainpy_state._network._nodeview import NodeView, _Segment, _flat_size
from brainpy_state._network._rules import all_to_all, one_to_one

__all__ = ['Simulator', 'SimulationResult']


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


class SimulationResult:
    """Recorded spikes from a :meth:`Simulator.simulate` run."""
    __module__ = 'brainpy.state'

    def __init__(self, recordings: dict, duration, dt):
        self._rec = recordings          # {id(recorder): (T, n_rec) array}
        self._T = duration
        self._dt = dt

    @staticmethod
    def _key(node):
        if isinstance(node, NodeView):
            return id(node.segments[0].population)
        return id(node)

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
            return NodeView([_GenSegment(_GeneratorSpec(model_cls, p))])
        mod = model_cls(size, **p)
        setattr(self, f'_node_{id(mod)}', mod)
        if isinstance(mod, _spike_recorder):
            return NodeView([_Segment(mod, jnp.arange(1))])
        holder = _SpikeHolder(_flat_size(mod))
        setattr(self, f'_holder_{id(mod)}', holder)
        return NodeView.of(mod)

    # -- connection --------------------------------------------------------
    def connect(self, pre: NodeView, post: NodeView, *, rule=all_to_all,
                weight=None, delay=None, allow_autapses: bool = True,
                allow_multapses: bool = True, seed: Optional[int] = None):
        """Connect ``pre`` to ``post`` (or register a recorder tap)."""
        if len(post.segments) == 1 and isinstance(post.segments[0].population, _spike_recorder):
            if len(pre.segments) != 1:
                raise NotImplementedError(
                    'recording a multi-segment view requires one recorder per segment'
                )
            seg = pre.segments[0]
            self._taps[id(post.segments[0].population)] = (id(seg.population), seg.indices)
            return
        for pre_seg in pre.segments:
            for post_seg in post.segments:
                self._connect_pair(pre_seg, post_seg, rule, weight, delay,
                                   allow_autapses, allow_multapses, seed)

    def _connect_pair(self, pre_seg, post_seg, rule, weight, delay,
                      allow_autapses, allow_multapses, seed):
        post_pop = post_seg.population
        if isinstance(pre_seg, _GenSegment):
            n = int(post_seg.indices.shape[0])
            gen = pre_seg.spec.model_cls(n, **pre_seg.spec.params)
            setattr(self, f'_node_{id(gen)}', gen)
            holder = _SpikeHolder(n)
            setattr(self, f'_holder_{id(gen)}', holder)
            proj = EventProjection(
                pre_spike=_holder_reader(holder), n_pre_pop=n,
                pre_local_idx=jnp.arange(n), post=post_pop,
                post_local_idx=post_seg.indices, rule=one_to_one, weight=weight,
                delay=delay, seed=seed)
        else:
            pre_pop = pre_seg.population
            holder = getattr(self, f'_holder_{id(pre_pop)}')
            proj = EventProjection(
                pre_spike=_holder_reader(holder), n_pre_pop=_flat_size(pre_pop),
                pre_local_idx=pre_seg.indices, post=post_pop,
                post_local_idx=post_seg.indices, rule=rule, weight=weight,
                delay=delay, pre_is_post=(pre_pop is post_pop),
                allow_autapses=allow_autapses, allow_multapses=allow_multapses, seed=seed)
        setattr(self, f'_proj_{next(self._proj_counter)}', proj)

    # -- run ---------------------------------------------------------------
    def update(self, t=None):
        dftype = brainstate.environ.dftype()
        children = list(self.nodes(allowed_hierarchy=(1, 1)).values())
        # 1) projections route the previous step's spikes into delta inputs
        for m in children:
            if isinstance(m, EventProjection):
                m.update()
        # 2) drive neurons/generators and capture their output into holders
        for m in children:
            if isinstance(m, (EventProjection, _SpikeHolder)):
                continue
            holder = getattr(self, f'_holder_{id(m)}', None)
            if holder is None:
                continue  # recorders / untracked devices have no holder
            out = m.update()
            if isinstance(m, Neuron):
                val = (jnp.asarray(u.get_mantissa(out)) >= 0.5).astype(dftype)
            else:
                val = jnp.asarray(out, dtype=dftype)
            holder.spk.value = val

    def simulate(self, duration, *, dt=None) -> SimulationResult:
        """Run for ``duration`` and return recorded spikes."""
        import brainstate.transform as transform
        if dt is None:
            dt = self._dt
        brainstate.nn.init_all_states(self)
        times = u.math.arange(0.0 * u.get_unit(dt), duration, dt)
        indices = u.math.arange(times.size)
        taps = dict(self._taps)

        def step(t, i):
            with brainstate.environ.context(t=t, i=i):
                self.update(t)
                return {rid: getattr(self, f'_holder_{sid}').spk.value[idx]
                        for rid, (sid, idx) in taps.items()}

        stacked = transform.for_loop(step, times, indices)
        recordings = {rid: jnp.asarray(stacked[rid]) for rid in taps}
        return SimulationResult(recordings, duration, dt)
