# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``stdp_dopamine_synapse`` — dopamine-modulated STDP spec + pure rule.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on the
:class:`~brainpy_state._nest_network._event_plastic.VoltageCoupledPlasticProj`
substrate (primitive #2). Two ingredients drive the weight:

* a **per-edge eligibility trace** ``c`` (cluster-05 ``edge_state_init`` machinery)
  built from standard STDP pairing (cluster-04): a post-after-pre pairing
  *facilitates* ``c`` by ``A_plus * Kplus`` and a presynaptic spike *depresses* it
  by ``A_minus * Kminus``;
* a **broadcast dopamine concentration** ``n`` read every step from a bound
  :class:`~brainpy_state._nest_device.volume_transmitter` via the cluster-08
  ``signal_reads`` seam (``ctx.signals['n']``).

The weight follows ``dw/dt = c(t) * (n(t) - b)`` (derived from NEST's
``update_weight_``); each step reproduces NEST's exact analytic integral with
``minus_dt = -dt`` (``stdp_dopamine_synapse.h:427-448``). The eager imperative
port (dopa-spike history buffers, lazy event-time integration) is retired; the
online per-step integral coincides with NEST's deferred trajectory at the
send/trigger sampling times (the cluster-04 "online <-> deferred equality").
"""
from __future__ import annotations
from brainpy_state._nest_base._base import NESTPlasticity

import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainstate.typing import ArrayLike

from brainpy_state._nest_base._plastic_base import (
    to_ms, to_scalar_float, unit_of, validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['stdp_dopamine_synapse']


class stdp_dopamine_synapse(NESTPlasticity):
    r"""Dopamine-modulated STDP synapse spec (NEST ``stdp_dopamine_synapse``).

    The weight is driven by the product of a slow per-edge eligibility trace ``c``
    (set by spike-timing pairing) and a global dopamine concentration ``n``
    (broadcast from a :class:`~brainpy_state._nest_device.volume_transmitter`):

    .. math::

       \frac{dw}{dt} = c(t)\,\big(n(t) - b\big),

    integrated analytically each step with NEST's ``update_weight_`` kernel
    (``minus_dt = -\Delta t``):

    .. math::

       \tau_s = \frac{\tau_c + \tau_n}{\tau_c\,\tau_n}, \qquad
       \Delta w = -c\left(\frac{n}{\tau_s}\,\mathrm{expm1}(-\tau_s\Delta t)
                  - b\,\tau_c\,\mathrm{expm1}(-\Delta t/\tau_c)\right),

    followed by clamping ``w`` to ``[Wmin, Wmax]``. The eligibility trace then
    decays (``c \leftarrow c\,e^{-\Delta t/\tau_c}``) and receives STDP impulses at
    the grid point: a post spike facilitates by ``A_plus`` times the strictly-prior
    presynaptic trace ``Kplus`` (``tau_plus``, ``+1`` per pre spike), a pre spike
    depresses by ``A_minus`` times the strictly-prior postsynaptic trace ``Kminus``
    (``tau_minus``, the post-neuron ``K^-``). The ``b`` baseline's sign flip
    (facilitation :math:`\leftrightarrow` depression when :math:`n<b`) falls out of
    the ``(n-b)`` factor for free.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge weight (current synapse). Bare numbers default to **pA**. Default
        ``1.0`` pA.
    delay : Quantity, optional
        Axonal/dendritic delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    A_plus, A_minus : float, optional
        Facilitation / depression amplitudes (per-pairing increment of ``c``).
        Defaults ``1.0`` / ``1.5``.
    tau_plus : Quantity, optional
        Presynaptic ``Kplus`` trace time constant (> 0). Default ``20.0 ms``.
    tau_minus : Quantity, optional
        Postsynaptic ``Kminus`` trace time constant (> 0). Default ``20.0 ms``.
    tau_c : Quantity, optional
        Eligibility-trace time constant (> 0). Default ``1000.0 ms``.
    tau_n : Quantity, optional
        Dopamine-concentration time constant (> 0). **Must equal the bound**
        ``volume_transmitter``'s ``tau_n``. Default ``200.0 ms``.
    b : float, optional
        Dopaminergic baseline (concentration units). Default ``0.0``.
    Wmin, Wmax : float, optional
        Lower / upper weight clamp. Defaults ``0.0`` / ``200.0``.
    c : float, optional
        Initial per-edge eligibility trace. Default ``0.0``.

    Notes
    -----
    **NEST divergence — parameter location.** In NEST ``A_plus``/``A_minus``/
    ``tau_plus``/``tau_c``/``tau_n``/``b``/``Wmin``/``Wmax`` are *common* properties
    and the ``volume_transmitter`` is bound there; ``tau_minus`` lives on the
    *postsynaptic neuron* (read via ``get_K_value``) and ``n`` lives on the
    *synapse* (integrated from the relayed dopa train). Here the spec is
    self-contained: ``tau_minus`` is a spec attribute (the cluster-04 ``stdp_synapse``
    convention — set identically on the NEST neuron for parity) and ``n`` is moved
    onto the broadcast :class:`~brainpy_state._nest_device.volume_transmitter`
    (``signal_reads=('n',)``), so ``tau_n`` must match between spec and transmitter.

    **Online vs deferred.** NEST integrates the weight lazily (only at pre ``send``
    and VT ``trigger_update_weight`` times); this kernel integrates every step with
    the broadcast ``n`` (one-step lag, ``O(dt/tau_n)``). The trajectories coincide
    at the send/trigger times where NEST's ``weight_recorder`` samples.

    **Parity note.** The ``n``/``tau_n``-on-``volume_transmitter`` move, the
    online-vs-deferred band, and the parity test are documented in
    :doc:`/nest-guide/stdp-divergences` (:ref:`stdp-dopamine`).

    References
    ----------
    .. [1] Izhikevich (2007). Solving the distal reward problem through linkage of
       STDP and dopamine signaling. *Cereb. Cortex* 17(10):2443-2452.
    .. [2] Potjans, Morrison, Diesmann (2010). Enabling functional neural circuit
       simulations with distributed computing of neuromodulated plasticity.
       *Front. Comput. Neurosci.* 4:141.
    .. [3] NEST ``models/stdp_dopamine_synapse.{h,cpp}`` + ``volume_transmitter``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state import stdp_dopamine_synapse
        >>> s = stdp_dopamine_synapse(weight=1.0, b=0.0)
        >>> s.is_homogeneous_weight, s.signal_reads, s.post_state_reads
        (False, ('n',), ())
        >>> s.edge_state_init()
        {'c': 0.0}
        >>> float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms))
        20.0
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False
    # primitive #2 reader: no per-edge post-State gather, one broadcast signal (n)
    post_state_reads = ()
    signal_reads = ('n',)

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        A_plus: ArrayLike = 1.0,
        A_minus: ArrayLike = 1.5,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        tau_c: ArrayLike = 1000.0 * u.ms,
        tau_n: ArrayLike = 200.0 * u.ms,
        b: ArrayLike = 0.0,
        Wmin: ArrayLike = 0.0,
        Wmax: ArrayLike = 200.0,
        c: ArrayLike = 0.0,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.A_plus = to_scalar_float(A_plus, name='A_plus')
        self.A_minus = to_scalar_float(A_minus, name='A_minus')
        self.b = to_scalar_float(b, name='b')
        self.Wmin = to_scalar_float(Wmin, name='Wmin')
        self.Wmax = to_scalar_float(Wmax, name='Wmax')
        self.c_init = to_scalar_float(c, name='c')

        self._tau_plus_ms = to_ms(tau_plus, name='tau_plus')
        self._tau_minus_ms = to_ms(tau_minus, name='tau_minus')
        self._tau_c_ms = to_ms(tau_c, name='tau_c')
        self._tau_n_ms = to_ms(tau_n, name='tau_n')
        for v, name in ((self._tau_plus_ms, 'tau_plus'), (self._tau_minus_ms, 'tau_minus'),
                        (self._tau_c_ms, 'tau_c'), (self._tau_n_ms, 'tau_n')):
            if v <= 0.0:
                raise ValueError(f"'{name}' must be > 0.")
        self.tau_plus = self._tau_plus_ms * u.ms
        self.tau_minus = self._tau_minus_ms * u.ms
        self.tau_c = self._tau_c_ms * u.ms
        self.tau_n = self._tau_n_ms * u.ms

        w0 = float(u.get_mantissa(self.weight))
        for v, name in ((w0, 'weight'), (self.A_plus, 'A_plus'), (self.A_minus, 'A_minus'),
                        (self.b, 'b'), (self.Wmin, 'Wmin'), (self.Wmax, 'Wmax'),
                        (self.c_init, 'c')):
            if not np.isfinite(v):
                raise ValueError(f"'{name}' must be finite.")

        # per-side trace seam: substrate maintains Kplus (tau_plus) + Kminus (tau_minus)
        self.pre_trace_tau = self.tau_plus
        self.post_trace_tau = self.tau_minus
        # precomputed inverse-time-constant sum for the weight integral
        self._taus = (self._tau_c_ms + self._tau_n_ms) / (self._tau_c_ms * self._tau_n_ms)

    def edge_state_init(self) -> dict:
        # per-edge eligibility trace c (cluster-05 edge-State machinery)
        return {'c': float(self.c_init)}

    # -- rule kernel -------------------------------------------------------
    def update(self, state, ctx):
        w = state['weight']
        c = state['c']
        n = ctx.signals['n']                                    # broadcast dopamine scalar
        dt = ctx.dt

        # (i) integrate the weight over the step: NEST update_weight_(c, n, -dt).
        # c is at t_{k-1} (pre-impulse, left-continuous); n at t_k (one-step lag).
        dw = -c * (n / self._taus * jnp.expm1(-self._taus * dt)
                   - self.b * self._tau_c_ms * jnp.expm1(-dt / self._tau_c_ms))
        w = jnp.clip(w + dw, self.Wmin, self.Wmax)

        # (ii) decay the eligibility trace over the step (NEST .h:504)
        c = c * jnp.exp(-dt / self._tau_c_ms)

        # (iii) STDP impulses at the grid point (cluster-04 strictly-prior idiom:
        # subtract this step's own spike so the trace is the value just *before* t_k)
        kplus_prior = ctx.pre_trace - ctx.pre_spike            # Kplus(t_pre^-)
        kminus_prior = ctx.post_trace - ctx.post_spike         # Kminus(t_post^-)
        c = c + self.A_plus * jnp.where(ctx.post_spike > 0, kplus_prior, 0.0)   # facilitate
        c = c - self.A_minus * jnp.where(ctx.pre_spike > 0, kminus_prior, 0.0)  # depress
        return {'weight': w, 'c': c}, w
