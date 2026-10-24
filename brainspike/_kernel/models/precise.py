# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Precise-timing models on ``timed_event`` ports (K0 task T7), built in JAX.

* :class:`iaf_psc_exp_ps` -- NEST ``iaf_psc_exp_ps``: exact integration between the
  sorted in-step input events, threshold crossings localised by a fixed-iteration
  root finder, precise output spike offsets and refractoriness with precise timing.
* :class:`poisson_generator_ps` -- NEST ``poisson_generator_ps``: continuous-time
  Poisson (optionally dead-time) spike trains, one per unit.
* :class:`spike_generator_ps` -- NEST ``spike_generator`` with ``precise_times=True``.

Time convention (``docs/specs/2026-10-22-k0-t7-precise-timing.md`` §1): step ``n``
covers ``(n dt, (n + 1) dt]``; a timed event carries its offset **from the step
start**, in ``(0, dt]``. NEST describes the same event by the stamp ``n + 1`` and the
backward offset ``dt - offset``.

Everything is vectorised over neurons, ``jit``/``vmap``-able and loop-free at run
time: the ``capacity + 1`` in-step event segments run as one ``lax.scan`` (traced
once, whatever the capacity) followed by the final segment to the step end, and the
root finder runs a fixed number of iterations (also a ``scan``). The host implementations in
``_nest_neuron/iaf_psc_exp_ps.py`` and ``_nest_device/poisson_generator_ps.py`` are
the algorithmic reference (NEST ``models/iaf_psc_exp_ps.cpp``,
``models/poisson_generator_ps.cpp``).
"""
from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import Exact, Integrator
from brainspike._kernel.model import Capabilities, Device, NeuronModel
from brainspike._kernel.models._common import as_record, device_window, latch
from brainspike._kernel.models.devices import _spike_table
from brainspike._kernel.params import Init, Param, Params, Record, States
from brainspike._kernel.ports import Current, Ports, TimedEvent, TimedEventPayload
from brainspike._kernel.propagators import propagator_exp
from brainspike._kernel.rng import draw_blocked
from brainspike._kernel.units import from_mantissa, grid_steps, to_mantissa

__all__ = [
    'ROOT_ITERATIONS',
    'iaf_psc_exp_ps',
    'poisson_generator_ps',
    'spike_generator_ps',
    'illinois_root',
]

#: Fixed number of Illinois (modified regula falsi) iterations of the spike-time search.
ROOT_ITERATIONS = 40

_INF = float('inf')

#: Driver of the in-step segment loop (``jax.lax.scan``). A module attribute so the
#: tests can swap in a Python-unrolled loop as the K0 reference.
_segment_scan = jax.lax.scan


# ---------------------------------------------------------------------------
# capacity variants
# ---------------------------------------------------------------------------

class _CapacityMixin:
    """``Model.with_capacity(K)``: cached subclass whose timed port holds ``K`` events."""

    #: (inputs|outputs, port name) of the timed port whose capacity is configurable.
    _timed_port: ClassVar[tuple[str, str]] = ('outputs', 'spike')

    @classmethod
    def with_capacity(cls, capacity: int) -> type:
        """Subclass whose configurable ``TimedEvent`` port has ``capacity`` slots.

        Parameters
        ----------
        capacity : int
            Events per neuron and step (>= 1).

        Returns
        -------
        type
            A cached subclass (same name); use it in ``Network.add``.
        """
        return _capacity_variant(cls, int(capacity))  # type: ignore[arg-type]  # false positive: classes hash

    @classmethod
    def capacity(cls) -> int:
        """Capacity of the configurable ``TimedEvent`` port."""
        attr, port = cls._timed_port
        return getattr(cls, attr)[port].capacity


@functools.lru_cache(maxsize=None)
def _capacity_variant(base: type[_CapacityMixin], capacity: int) -> type:
    attr, port = base._timed_port
    ports = dict(getattr(base, attr))
    old = ports[port]
    ports[port] = TimedEvent(capacity, unit=old.unit, into=old.into)
    if base.capacity() == capacity:
        return base
    return type(base.__name__, (base,), {attr: Ports(**ports), '__module__': base.__module__,
                                         '__qualname__': base.__qualname__,
                                         '__doc__': base.__doc__})


def _timed_output(offsets: Any, weights: Any, valid: Any, overflow: Any = None) -> TimedEventPayload:
    """Payload with invalid slots at ``+inf`` / weight 0 (no sorting)."""
    n = valid.shape[0]
    return TimedEventPayload(
        offsets=from_mantissa(jnp.where(valid, offsets, jnp.inf), u.ms),
        weights=jnp.where(valid, weights, 0).astype(offsets.dtype),
        valid=valid,
        overflow=jnp.zeros((n,), jnp.int32) if overflow is None else overflow,
    )


# ---------------------------------------------------------------------------
# iaf_psc_exp_ps
# ---------------------------------------------------------------------------

def illinois_root(f: Any, h: Any, iterations: int = ROOT_ITERATIONS) -> Any:
    """Root of ``f`` on ``[0, h]`` by the Illinois method, fixed iteration count.

    Vectorised NEST ``regula_falsi`` (``libnestutil/regula_falsi.h``): regula falsi
    that halves the function value kept at a bracket end chosen twice in a row.
    Requires ``f(0) <= 0 <= f(h)`` element-wise; if rounding loses the sign change,
    the bracket end nearest the root is returned (``h`` if ``f(h) <= 0``, ``0`` if
    ``f(0) >= 0``).

    Parameters
    ----------
    f : callable
        ``f(t) -> array`` evaluated element-wise (shape of ``h``).
    h : jax.Array
        Bracket lengths (``> 0`` where a root is wanted).
    iterations : int
        Number of iterations (static).

    Returns
    -------
    jax.Array
        Root estimates, same shape as ``h``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainspike._kernel.models.precise import illinois_root
        >>> r = illinois_root(lambda t: t * t - 2.0, jnp.array([2.0]))
        >>> bool(abs(float(r[0]) - 2.0 ** 0.5) < 1e-6)
        True
    """
    a = jnp.zeros_like(h)
    b = h
    side = jnp.zeros(h.shape, jnp.int32)

    def body(carry, _):
        a, b, fa, fb, side, _c = carry
        denom = fb - fa
        ok = denom != 0
        c = jnp.where(ok, (a * fb - b * fa) / jnp.where(ok, denom, 1), 0.5 * (a + b))
        c = jnp.clip(c, jnp.minimum(a, b), jnp.maximum(a, b))
        # materialise f(c) once: XLA may otherwise duplicate it into several fusions
        # whose float32 results can differ in sign right at the root
        fc = jax.lax.optimization_barrier(f(c))
        left = fc < 0                # bracket invariant f(a) <= 0 <= f(b): move a
        right = fc > 0               # move b
        exact = ~left & ~right       # fc == 0 (or nan): collapse the bracket
        fb = jnp.where(left & (side == 1), fb / 2, fb)
        fa = jnp.where(right & (side == -1), fa / 2, fa)
        a = jnp.where(left | exact, c, a)
        fa = jnp.where(left, fc, jnp.where(exact, 0, fa))
        b = jnp.where(right | exact, c, b)
        fb = jnp.where(right, fc, jnp.where(exact, 0, fb))
        side = jnp.where(left, 1, jnp.where(right, -1, side))
        return (a, b, fa, fb, side, c), None

    # fixed trip count: lowered to one scan (no while_loop), the body is traced once
    fa0, fb0 = f(a), f(b)
    carry = (a, b, fa0, fb0, side, 0.5 * (a + b))
    carry, _ = jax.lax.scan(body, carry, None, length=iterations)
    # no sign change after rounding (low precision near a bracket end): take that end
    return jnp.where(fb0 <= 0, h, jnp.where(fa0 >= 0, jnp.zeros_like(h), carry[-1]))


def _ps_propagators(h: Any, tau_m: Any, tau_ex: Any, tau_in: Any, c_m: Any) -> tuple:
    """``P20, P21_ex, P21_in, exp(-h/tau_m), expm1(-h/tau_m), exp(-h/tau_ex), exp(-h/tau_in)``."""
    expm1_m = jnp.expm1(-h / tau_m)
    P20 = -tau_m / c_m * expm1_m
    P21e = propagator_exp(tau_ex, tau_m, c_m, h).astype(h.dtype)
    P21i = propagator_exp(tau_in, tau_m, c_m, h).astype(h.dtype)
    return P20, P21e, P21i, jnp.exp(-h / tau_m), expm1_m, jnp.exp(-h / tau_ex), jnp.exp(-h / tau_in)


class iaf_psc_exp_ps(_CapacityMixin, NeuronModel):
    r"""Current-based LIF with exponential PSCs and precise spike times (NEST ``iaf_psc_exp_ps``).

    .. math::

       \frac{dV}{dt} = -\frac{V - E_L}{\tau_m} + \frac{I_{syn,ex} + I_{syn,in} + I_e
       + I_{stim}}{C_m}, \qquad
       \frac{dI_{syn,X}}{dt} = -\frac{I_{syn,X}}{\tau_{syn,X}} .

    Input events arrive with offsets inside the step and are integrated exactly
    between them; a threshold crossing is localised to machine precision and emitted
    as a timed spike. See ``docs/specs/2026-10-22-k0-t7-precise-timing.md`` §2.1.

    Parameters
    ----------
    n : int
        Number of neurons.
    params : mapping, optional
        Overrides of ``E_L`` (-70 mV), ``C_m`` (250 pF), ``tau_m`` (10 ms),
        ``tau_syn_ex`` / ``tau_syn_in`` (2 ms), ``t_ref`` (2 ms), ``V_th``
        (-55 mV), ``V_reset`` (-70 mV), ``V_min`` (-inf mV), ``I_e`` (0 pA). Per
        neuron, traced.
    init : mapping, optional
        Initial values (``V_m`` defaults to ``E_L``; currents to 0).
    **kwargs
        ``precision``, ``key``, ``trainable``, ``name``.

    Notes
    -----
    Ports: ``syn`` -- ``timed_event`` (pA, capacity 16; :meth:`with_capacity`), a
    weight ``w >= 0`` adds to ``I_syn_ex`` and ``w < 0`` to ``I_syn_in`` (NEST);
    ``I`` -- ``current`` (pA), latched (acts during the next step). Output
    ``spike`` -- ``timed_event`` with capacity 1 (refractoriness lasts at least one
    step). States: ``spike`` (count of the step), ``spike_offset`` (ms from the step
    start, ``inf`` without a spike), ``is_refractory``, ``ref_count``,
    ``last_spike_offset``, ``overflow`` (input events dropped because the ``syn``
    capacity was exceeded, cumulative).

    NEST requires ``t_ref >= dt``; here the refractory period is
    ``max(1, ceil(t_ref / dt))`` steps (parameters are traced, so they cannot be
    validated at run time). ``V_min`` is applied only on steps without input events,
    as in NEST.

    Gradients: ``V_m`` and the output spike offset are differentiable (the offset
    by the implicit function theorem: :math:`dt^*/d\theta = -\partial_\theta f /
    f'(t^*)`); the spike count is a hard threshold.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from brainspike._kernel import StepContext, TimedEventPayload
        >>> from brainspike._kernel.models import iaf_psc_exp_ps
        >>> neu = iaf_psc_exp_ps(1, params=dict(I_e=1000.0 * u.pA))
        >>> ctx = StepContext.create(0.0 * u.ms, 0.1 * u.ms)
        >>> ev = TimedEventPayload.pack(jnp.array([[0.05]]) * u.ms, jnp.array([[50.0]]) * u.pA,
        ...                             jnp.array([[True]]), capacity=16)
        >>> out = neu.update(ctx, {'syn': ev})
        >>> float(neu.dyn_states['I_syn_ex'].value.to_decimal(u.pA)[0]) > 48.0
        True
    """

    params = Params(E_L=-70.0 * u.mV, C_m=250.0 * u.pF, tau_m=10.0 * u.ms,
                    tau_syn_ex=2.0 * u.ms, tau_syn_in=2.0 * u.ms, t_ref=2.0 * u.ms,
                    V_th=-55.0 * u.mV, V_reset=-70.0 * u.mV,
                    V_min=Param(-_INF * u.mV), I_e=0.0 * u.pA)
    states = States(V_m=Init(lambda p: p.E_L), I_syn_ex=Init(0.0 * u.pA),
                    I_syn_in=Init(0.0 * u.pA), spike=Init(0.0),
                    spike_offset=Init(_INF * u.ms),
                    is_refractory=Init(0, dtype=np.int32), ref_count=Init(0, dtype=np.int32),
                    last_spike_offset=Init(0.0 * u.ms), overflow=Init(0, dtype=np.int32))
    inputs = Ports(syn=TimedEvent(16, unit=u.pA),
                   I=Current(u.pA, latch=True))  # NEST CurrentEvent: next step (D-a)
    outputs = Ports(spike=TimedEvent(1))
    integrator: Integrator = Exact()
    capabilities: ClassVar[Capabilities] = Capabilities(differentiable=True, etrace=False,
                                                        precise_spike_times=True)
    _timed_port = ('inputs', 'syn')

    def output_states(self) -> dict[str, str]:
        return {}

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Advance one step (NEST ``iaf_psc_exp_ps::update``, vectorised)."""
        inp = as_record(inp)
        acting, new_latch = latch(self, s, inp)
        mV = lambda x: to_mantissa(x, u.mV)
        pA = lambda x: to_mantissa(x, u.pA)
        ms = lambda x: to_mantissa(x, u.ms)
        dtype = mV(s.V_m).dtype
        cast = lambda x: jnp.asarray(x).astype(dtype)
        dt = cast(ms(ctx.dt))
        E_L = cast(mV(p.E_L))
        tau_m, tau_ex, tau_in = cast(ms(p.tau_m)), cast(ms(p.tau_syn_ex)), cast(ms(p.tau_syn_in))
        c_m = cast(to_mantissa(p.C_m, u.pF))
        U_th, U_reset, U_min = cast(mV(p.V_th)) - E_L, cast(mV(p.V_reset)) - E_L, cast(mV(p.V_min)) - E_L
        I0 = cast(pA(p.I_e)) + cast(pA(acting['I']))
        R = jnp.maximum(grid_steps(ms(p.t_ref), ms(ctx.dt)), 1).astype(jnp.int32)
        prop = lambda h: _ps_propagators(h, tau_m, tau_ex, tau_in, c_m)

        y2 = cast(mV(s.V_m)) - E_L
        y1e, y1i = cast(pA(s.I_syn_ex)), cast(pA(s.I_syn_in))
        refr = s.is_refractory != 0

        # refractory return scheduled in this step (from the counters before this step)
        cnt = s.ref_count
        counting = cnt > 0
        cnt = jnp.where(counting, cnt - 1, cnt)
        end_here = counting & (cnt == 0)
        end_off = cast(ms(s.last_spike_offset))

        # instant spike of a supra-threshold (initial) state at the step start
        inst = y2 >= U_th
        y2 = jnp.where(inst, U_reset, y2)
        refr = refr | inst
        spiked = inst

        # merge the input events with the refractory pseudo-event; sort at the receiver
        syn: TimedEventPayload = inp.syn
        off_in = cast(ms(syn.offsets))
        w_in = cast(pA(syn.weights))
        off = jnp.concatenate([end_off[:, None], off_in], axis=1)
        w = jnp.concatenate([jnp.zeros_like(end_off)[:, None], w_in], axis=1)
        valid = jnp.concatenate([end_here[:, None], jnp.asarray(syn.valid, bool)], axis=1)
        is_ref = jnp.zeros(valid.shape, bool).at[:, 0].set(True)
        key = jnp.where(valid, off, jnp.inf)
        order = jnp.argsort(key, axis=1, stable=True)  # pseudo-event first on ties (NEST)
        take = lambda a: jnp.take_along_axis(a, order, axis=1)
        off, w, valid, is_ref = take(off), take(w), take(valid), take(is_ref)
        off = jnp.clip(jnp.where(valid, off, dt), 0, dt)  # double-where: no inf in the graph
        noev = ~jnp.any(valid, axis=1)

        zero = jnp.zeros_like(y2)
        t_last = zero
        br = dict(y2=zero, y1e=zero, y1i=zero, h=zero, t0=zero)  # bracket of the crossing

        def segment(h, y2, y1e, y1i, refr, spiked, t_last, br, final):
            P20, P21e, P21i, em, expm1_m, ee, ei = prop(h)
            y2n = P20 * I0 + P21e * y1e + P21i * y1i + y2 * em
            if final:  # NEST's no-event formula and V_min clamp on steps without events
                y2_noev = jnp.maximum(P20 * I0 + P21e * y1e + P21i * y1i + expm1_m * y2 + y2, U_min)
                y2n = jnp.where(noev, y2_noev, y2n)
            y2n = jnp.where(refr, y2, y2n)
            cross = (h > 0) & ~refr & (y2n >= U_th)
            br = {k: jnp.where(cross, v, br[k]) for k, v in
                  dict(y2=y2, y1e=y1e, y1i=y1i, h=h, t0=t_last).items()}
            y2n = jnp.where(cross, U_reset, y2n)
            return y2n, y1e * ee, y1i * ei, refr | cross, spiked | cross, br

        # one scan over the ``capacity + 1`` sorted slots (traced once, §7.3), then the
        # peeled final segment to the step end (static ``final=True``)
        def slot(carry, ev_i):
            y2, y1e, y1i, refr, spiked, t_last, br = carry
            off_i, w_i, valid_i, is_ref_i = ev_i
            h = jnp.where(valid_i, jnp.maximum(off_i - t_last, 0), 0)
            y2, y1e, y1i, refr, spiked, br = segment(h, y2, y1e, y1i, refr, spiked, t_last, br, False)
            ev = valid_i & ~is_ref_i
            refr = jnp.where(valid_i & is_ref_i, False, refr)
            y1e = y1e + jnp.where(ev & (w_i >= 0), w_i, 0)
            y1i = y1i + jnp.where(ev & (w_i < 0), w_i, 0)
            t_last = jnp.where(valid_i, off_i, t_last)
            return (y2, y1e, y1i, refr, spiked, t_last, br), None

        (y2, y1e, y1i, refr, spiked, t_last, br), _ = _segment_scan(
            slot, (y2, y1e, y1i, refr, spiked, t_last, br), (off.T, w.T, valid.T, is_ref.T))
        h = jnp.maximum(dt - t_last, 0)
        y2, y1e, y1i, refr, spiked, br = segment(h, y2, y1e, y1i, refr, spiked, t_last, br, True)

        crossed = spiked & ~inst
        t_cross = self._spike_time(br, crossed, I0, U_th, tau_m, tau_ex, tau_in, c_m)
        sp_off = jnp.where(inst, 0, jnp.where(crossed, br['t0'] + t_cross, jnp.inf)).astype(dtype)

        ref_count = jnp.where(spiked, R, cnt)
        last_off = jnp.where(spiked, sp_off, end_off)
        s = s.replace(
            V_m=from_mantissa((y2 + E_L).astype(dtype), u.mV),
            I_syn_ex=from_mantissa(y1e.astype(dtype), u.pA),
            I_syn_in=from_mantissa(y1i.astype(dtype), u.pA),
            spike=spiked.astype(dtype),
            spike_offset=from_mantissa(sp_off, u.ms),
            is_refractory=refr.astype(jnp.int32),
            ref_count=ref_count.astype(jnp.int32),
            last_spike_offset=from_mantissa(last_off.astype(dtype), u.ms),
            overflow=s.overflow + jnp.asarray(syn.overflow, jnp.int32),
            **new_latch)
        out = _timed_output(jnp.where(spiked, sp_off, dt)[:, None], jnp.ones_like(sp_off)[:, None],
                            spiked[:, None])
        return s, {'spike': out}

    @staticmethod
    def _spike_time(br: dict, crossed: Any, I0: Any, U_th: Any, tau_m: Any, tau_ex: Any,
                    tau_in: Any, c_m: Any) -> Any:
        """Crossing time from the segment start (NEST ``threshold_distance`` root)."""
        h = jnp.where(crossed, br['h'], 1)  # dummy bracket elsewhere (masked by the caller)

        def f(t, b):
            P20, P21e, P21i, em, _, _, _ = _ps_propagators(t, tau_m, tau_ex, tau_in, c_m)
            return P20 * I0 + P21e * b['y1e'] + P21i * b['y1i'] + b['y2'] * em - U_th

        def search(b):
            b = jax.lax.stop_gradient(b)
            return illinois_root(lambda t: f(t, b), jax.lax.stop_gradient(h))

        t_star = jax.lax.cond(jnp.any(crossed), search, lambda b: jnp.zeros_like(h), br)
        t_star = jax.lax.stop_gradient(t_star)
        # implicit-function correction: value t*, gradient -df/dθ / f'(t*)
        y = f(t_star, br) + U_th
        dV = (I0 + br['y1e'] * jnp.exp(-t_star / tau_ex) + br['y1i'] * jnp.exp(-t_star / tau_in)) / c_m \
            - y / tau_m
        dV_safe = jnp.where(crossed & (dV > 0), dV, 1)
        corr = -(y - U_th) / dV_safe
        corr = jnp.where(crossed, corr, 0)
        return t_star + (corr - jax.lax.stop_gradient(corr))


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------

_WINDOW = dict(start=Param(0.0 * u.ms), stop=Param(_INF * u.ms), origin=Param(0.0 * u.ms))
_DEVICE_CAPS = Capabilities(differentiable=False, etrace=False, precise_spike_times=True)


if TYPE_CHECKING:
    # the mixin is only ever combined with ``_CapacityMixin`` and a ``Device``
    class _OffsetsBase(_CapacityMixin, NeuronModel):
        pass
else:
    _OffsetsBase = object


class _OffsetsStateMixin(_OffsetsBase):
    """Adds the ``(n, K)`` state ``spike_offsets`` (ms, ``inf`` in empty slots)."""

    @classmethod
    def all_state_decl(cls) -> States:
        return super().all_state_decl().extend(spike_offsets=Init(_INF * u.ms))

    def init_state(self, batch_size: int | None = None, **kwargs: Any) -> None:
        super().init_state(batch_size, **kwargs)
        dtype = u.get_mantissa(self.dyn_states['spike'].value).dtype
        k = type(self).capacity()
        self.dyn_states['spike_offsets'] = brainstate.HiddenState(
            from_mantissa(jnp.full((self.n, k), jnp.inf, dtype), u.ms))


class poisson_generator_ps(_OffsetsStateMixin, _CapacityMixin, Device):
    r"""Poisson spike trains in continuous time (NEST ``poisson_generator_ps``).

    Parameters
    ----------
    n : int
        Number of independent trains (NEST draws one train **per target**: give each
        target its own unit and connect ``one_to_one``).
    params : mapping, optional
        ``rate`` (Hz, default 0), ``dead_time`` (ms, default 0; ``dead_time * rate
        <= 1``), ``start`` (0 ms), ``stop`` (inf), ``origin`` (0 ms). Per unit,
        traced.
    **kwargs
        ``precision``, ``name``.

    Notes
    -----
    Output ``spike`` -- ``timed_event`` with capacity 8 per unit and step
    (:meth:`with_capacity`). Spikes are emitted in ``(t_lo, t_hi]`` with ``t_lo =
    max(n dt, origin + start)`` and ``t_hi = min((n + 1) dt, origin + stop)``.
    Inter-spike intervals are ``dead_time + Exp(1) * (1/rate - dead_time)``; the
    first spike after activation is drawn from NEST's equilibrium initialisation.
    Events beyond the capacity are not emitted and increase ``overflow`` by 1 plus
    the expected number of further events (stochastically rounded, unbiased); the next spike is then redrawn from the
    step end (exact for ``dead_time = 0``). States: ``spike`` (count), ``spike_offsets``
    ``(n, K)``, ``next_spike`` (relative to the next step's start), ``started``,
    ``overflow``. Randomness only from ``ctx.key``, drawn per id block
    (:func:`~brainspike._kernel.rng.draw_blocked`, D-g).

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainunit as u
        >>> from brainspike._kernel import StepContext
        >>> from brainspike._kernel.models import poisson_generator_ps
        >>> pg = poisson_generator_ps(2000, params=dict(rate=5000.0 * u.Hz))
        >>> ctx = StepContext.create(0.0 * u.ms, 0.1 * u.ms, key=jax.random.key(0))
        >>> out = pg.update(ctx)['spike']
        >>> 0.4 < float(out.count.mean()) < 0.6
        True
    """

    params = Params(rate=Param(0.0 * u.Hz), dead_time=Param(0.0 * u.ms), **_WINDOW)
    states = States(spike=Init(0.0), next_spike=Init(0.0 * u.ms),
                    started=Init(0, dtype=np.int32), overflow=Init(0, dtype=np.int32))
    outputs = Ports(spike=TimedEvent(8))
    capabilities: ClassVar[Capabilities] = _DEVICE_CAPS

    def output_states(self) -> dict[str, str]:
        return {}

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Emit this step's spikes (see the class notes)."""
        if ctx.key is None:
            raise ValueError('poisson_generator_ps needs ctx.key (a per-population, per-step key)')
        dtype = jnp.asarray(s.spike).dtype
        cast = lambda x: jnp.asarray(x).astype(dtype)
        ms = lambda x: cast(to_mantissa(x, u.ms))
        K = type(self).capacity()
        n_units = self.n
        dt = ms(ctx.dt)
        step = cast(ctx.step)
        rate = cast(to_mantissa(p.rate, u.Hz))
        dead = ms(p.dead_time)
        # window in steps relative to this step (exact for grid-aligned start / stop)
        s_lo = (ms(p.origin) + ms(p.start)) / dt - step
        s_hi = (ms(p.origin) + ms(p.stop)) / dt - step
        lo = jnp.clip(s_lo, 0, 1) * dt
        hi = jnp.clip(s_hi, 0, 1) * dt
        active = (rate > 0) & (lo < hi)
        isi_mean = jnp.where(rate > 0, 1000.0 / jnp.where(rate > 0, rate, 1), jnp.inf)
        inv = isi_mean - dead

        # per id block (D-g): sub-stream j of block b uses fold_in(block key, j)
        def sub(j, draw, tail=()):
            return draw_blocked(ctx.key, n_units,
                                lambda k, sh: draw(jax.random.fold_in(k, j), sh, dtype), tail=tail)

        u_choice = sub(0, jax.random.uniform)
        u_dead = sub(1, jax.random.uniform)
        e_first = sub(2, jax.random.exponential)
        e_isi = sub(3, jax.random.exponential, (K + 1,))

        in_dead = (dead > 0) & (u_choice < dead * rate / 1000.0)
        init_off = jnp.where(in_dead, u_dead * dead, inv * e_first + dead)
        started = s.started != 0
        first = jnp.where(started, ms(s.next_spike), lo + init_off)
        gaps = dead[:, None] + inv[:, None] * e_isi[:, :K]
        times = first[:, None] + jnp.concatenate(
            [jnp.zeros((n_units, 1), dtype), jnp.cumsum(gaps, axis=1)], axis=1)  # (n, K + 1)
        emit = active[:, None] & (times[:, :K] <= hi[:, None])
        over = active & (times[:, K] <= hi)
        after = times > hi[:, None]
        nxt_in = jnp.take_along_axis(times, jnp.argmax(after, axis=1)[:, None], axis=1)[:, 0]
        nxt_over = hi + inv * e_isi[:, K]
        # events beyond capacity: 1 (the (K+1)-th) + the expected number in the rest of
        # the window, stochastically rounded so that the count is unbiased
        u_more = sub(4, jax.random.uniform)
        n_more = jnp.floor(jnp.where(over, (hi - times[:, K]) / isi_mean, 0) + u_more)
        nxt = jnp.where(over, nxt_over, nxt_in)
        started_new = started | active
        nxt = jnp.where(active, nxt, first)
        nxt = jnp.where(started_new, nxt - dt, ms(s.next_spike))

        offsets = jnp.clip(jnp.where(emit, times[:, :K], dt), 0, dt)
        out = _timed_output(offsets, jnp.ones_like(offsets), emit)
        s = s.replace(spike=jnp.sum(emit, axis=1).astype(dtype),
                      spike_offsets=out.offsets,
                      next_spike=from_mantissa(nxt.astype(dtype), u.ms),
                      started=started_new.astype(jnp.int32),
                      overflow=s.overflow + jnp.where(over, 1 + n_more, 0).astype(jnp.int32))
        return s, {'spike': out}


def _precise_stamps(t: Any, dt: Any) -> Any:
    """NEST ``Time::ms_stamp``: the smallest grid step ``S`` with ``S dt >= t``.

    Unlike :func:`~brainspike._kernel.units.grid_steps` (which snaps times within
    ``GRID_SNAP`` steps of the grid, right for grid parameters), a precise time just
    after a grid point keeps its own step; only rounding noise of a few ulp snaps.
    """
    x = jnp.asarray(t) / jnp.asarray(dt)
    r = jnp.round(x)
    snap = 4 * jnp.finfo(x.dtype).eps * jnp.maximum(jnp.abs(r), 1)
    return jnp.where(jnp.abs(x - r) <= snap, r, jnp.ceil(x))


class spike_generator_ps(_OffsetsStateMixin, _CapacityMixin, Device):
    r"""Spikes at precise (off-grid) times (NEST ``spike_generator``, ``precise_times=True``).

    Parameters
    ----------
    n : int
        Number of units.
    params : mapping, optional
        ``spike_times`` -- one train shared by all units or one per unit (ragged
        allowed; see :class:`~brainspike._kernel.models.spike_generator`);
        ``spike_multiplicities`` (default 1, sent as the event weight, NEST
        ``weight * multiplicity``); ``start`` (0 ms), ``stop`` (inf),
        ``origin`` (0 ms).
    **kwargs
        ``precision``, ``name``.

    Notes
    -----
    A spike at ``T`` has the NEST stamp ``S = steps(origin + T)`` (rounded up) and is
    emitted at step ``S - 1`` with offset ``origin + T - (S - 1) dt`` in ``(0, dt]``,
    iff ``t_min < S <= t_max`` (device window). Output ``spike`` -- ``timed_event``
    with capacity 8 (:meth:`with_capacity`); more spikes of one unit in one step are
    counted in ``overflow`` (the latest are dropped). In float32 the offsets carry
    the absolute-time rounding of ``T`` (about ``1e-7 * T``); use float64 for parity.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainspike._kernel import StepContext
        >>> from brainspike._kernel.models import spike_generator_ps
        >>> sg = spike_generator_ps(1, params=dict(spike_times=[1.03, 1.07] * u.ms))
        >>> out = sg.update(StepContext.create(1.0 * u.ms, 0.1 * u.ms, step=10))['spike']
        >>> [round(float(x), 4) for x in out.offsets.to_decimal(u.ms)[0, :2]]
        [0.03, 0.07]
    """

    params = Params(spike_times=Param(None, u.ms, per_neuron=False),
                    spike_multiplicities=Param(None, None, per_neuron=False), **_WINDOW)
    states = States(spike=Init(0.0), overflow=Init(0, dtype=np.int32))
    outputs = Ports(spike=TimedEvent(8))
    capabilities: ClassVar[Capabilities] = _DEVICE_CAPS

    def __init__(self, n: int, params: Mapping[str, Any] | None = None,
                 init: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        params = dict(params or {})
        times, mult = _spike_table(int(n), params.pop('spike_times', None),
                                   params.pop('spike_multiplicities', None))
        params.update(spike_times=times, spike_multiplicities=mult)
        super().__init__(n, params, init, **kwargs)

    def output_states(self) -> dict[str, str]:
        return {}

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Emit the spikes whose stamp ends this step, with their offsets."""
        dtype = jnp.asarray(s.spike).dtype
        dt = jnp.asarray(to_mantissa(ctx.dt, u.ms)).astype(dtype)
        t_min, t_max = device_window(p, dt)
        t = (to_mantissa(p.origin, u.ms)[:, None] + to_mantissa(p.spike_times, u.ms)).astype(dtype)
        stamps = _precise_stamps(t, dt)  # (n, M)
        n = jnp.asarray(ctx.step).astype(stamps.dtype)
        hit = (stamps - 1 == n) & (t_min[:, None] < stamps) & (stamps <= t_max[:, None])
        hit = hit & (jnp.asarray(p.spike_multiplicities) > 0)
        off = jnp.clip(jnp.where(hit, t - (stamps - 1) * dt, dt), 0, dt)
        mult = jnp.asarray(p.spike_multiplicities).astype(dtype)
        packed = TimedEventPayload.pack(from_mantissa(off, u.ms), jnp.broadcast_to(mult, off.shape),
                                        hit, type(self).capacity())
        out = _timed_output(to_mantissa(packed.offsets, u.ms), packed.weights, packed.valid)
        s = s.replace(spike=jnp.sum(out.weights, axis=1).astype(dtype), spike_offsets=out.offsets,
                      overflow=s.overflow + packed.overflow)
        return s, {'spike': out}
