# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST ``iaf_psc_alpha`` on the K0 contracts (exact propagator integration).

Numerics follow ``brainspike/_nest_neuron/iaf_psc_alpha.py`` (NEST
``models/iaf_psc_alpha.cpp``) with the float32-safe JAX propagators of
:func:`~brainspike._kernel.propagators.iaf_psc_alpha_propagators`, computed
inside ``step`` from traced parameters and ``ctx.dt``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import braintools
import brainunit as u
import jax.numpy as jnp

from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import Exact, Integrator
from brainspike._kernel.models._common import SpikingNeuron, as_record, latch, rule_detach
from brainspike._kernel.params import Init, Param, Params, Record, States
from brainspike._kernel.ports import Current, Event, Ports
from brainspike._kernel.propagators import iaf_psc_alpha_propagators
from brainspike._kernel.spiking import Clamp, SpikeRule
from brainspike._kernel.units import from_mantissa, to_mantissa

__all__ = ['iaf_psc_alpha']

_PA_MS = u.pA / u.ms


def _margin(s: Record, p: Record) -> Any:
    return (s.V_m - p.V_th) / (p.V_th - p.V_reset)


def _reset(s: Record, p: Record) -> Record:
    return s.replace(V_m=p.V_reset)


class iaf_psc_alpha(SpikingNeuron):
    r"""Leaky integrate-and-fire neuron with alpha-shaped PSCs (NEST ``iaf_psc_alpha``).

    .. math::

       C_m \frac{dV}{dt} = -\frac{C_m}{\tau_m}(V - E_L) + I_{syn,ex} + I_{syn,in}
       + I_e + I_{stim}, \qquad
       \frac{d^2 I_{syn,X}}{dt^2} + \frac{2}{\tau_X}\frac{dI_{syn,X}}{dt}
       + \frac{I_{syn,X}}{\tau_X^2} = 0 .

    One step (NEST order): if not refractory, ``V`` is propagated exactly with
    the PSC states and the current latched at the previous step (``I_e`` acts
    immediately), clipped below at ``V_min``; the PSC states are propagated;
    ``event`` inputs jump ``dI_syn += (e / tau_syn) * w``; then the threshold
    check, reset to ``V_reset`` and refractoriness of ``ceil(t_ref / dt)`` steps.

    Parameters
    ----------
    n : int
        Number of neurons.
    params : mapping, optional
        Overrides of ``E_L`` (-70 mV), ``C_m`` (250 pF), ``tau_m`` (10 ms),
        ``t_ref`` (2 ms), ``V_th`` (-55 mV), ``V_reset`` (-70 mV),
        ``tau_syn_ex`` / ``tau_syn_in`` (2 ms), ``I_e`` (0 pA), ``V_min``
        (-inf mV). All per neuron, traced.
    init : mapping, optional
        Initial values (``V_m`` defaults to ``E_L``; PSC states to 0).
    surrogate : callable or None, optional
        Surrogate spike function (default ``braintools.surrogate.ReluGrad()``)
        applied to ``(V_m - V_th) / (V_th - V_reset)``.
    **kwargs
        ``precision``, ``key``, ``trainable``, ``detach_reset``, ``name``.

    Notes
    -----
    Ports: ``ex`` / ``in_`` -- receptor ``event`` ports (decision D-b),
    ``Event(unit=u.pA, into='I_syn_ex')`` / ``into='I_syn_in'``, weights in pA
    (the peak of the PSC), each filtered by its own alpha kernel
    (``tau_syn_ex`` / ``tau_syn_in``; the jump enters ``dI_syn_*``); the weight
    sign is kept
    (NEST convention: inhibitory weights are negative; the NEST front-end routes
    ``w > 0`` to ``ex`` and ``w < 0`` to ``in_``). ``I`` -- ``current`` in pA,
    latched. Output ``spike`` -- ``event``, ``max_multiplicity=1``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainspike._kernel import StepContext
        >>> from brainspike._kernel.models import iaf_psc_alpha
        >>> neu = iaf_psc_alpha(1)
        >>> ctx = StepContext.create(0.0 * u.ms, 0.1 * u.ms)
        >>> _ = neu.update(ctx, {'ex': 100.0 * u.pA})
        >>> dI = neu.dyn_states['dI_syn_ex'].value.to_decimal(u.pA / u.ms)
        >>> round(float(dI[0]), 3)  # e / (2 ms) * 100 pA
        135.914
    """

    params = Params(E_L=-70.0 * u.mV, C_m=250.0 * u.pF, tau_m=10.0 * u.ms, t_ref=2.0 * u.ms,
                    V_th=-55.0 * u.mV, V_reset=-70.0 * u.mV, tau_syn_ex=2.0 * u.ms,
                    tau_syn_in=2.0 * u.ms, I_e=0.0 * u.pA,
                    V_min=Param(-float('inf') * u.mV))
    states = States(V_m=Init(lambda p: p.E_L), I_syn_ex=Init(0.0 * u.pA),
                    dI_syn_ex=Init(0.0 * _PA_MS), I_syn_in=Init(0.0 * u.pA),
                    dI_syn_in=Init(0.0 * _PA_MS))
    inputs = Ports(ex=Event(unit=u.pA, into='I_syn_ex'),  # neuron-owned receptor ports (D-b)
                   in_=Event(unit=u.pA, into='I_syn_in'),
                   I=Current(u.pA, latch=True))  # NEST CurrentEvent: next step (D-a)
    outputs = Ports(spike=Event())
    spiking = SpikeRule(condition=_margin, jump=_reset, refractory=Clamp('V_m', 't_ref'),
                        surrogate=braintools.surrogate.ReluGrad())
    integrator: Integrator = Exact()

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Advance one step (NEST ``iaf_psc_alpha::update`` order)."""
        inp = as_record(inp)
        acting, new_latch = latch(self, s, inp)
        mV = lambda x: to_mantissa(x, u.mV)
        pA = lambda x: to_mantissa(x, u.pA)
        dpA = lambda x: to_mantissa(x, _PA_MS)
        dtype = mV(s.V_m).dtype
        dt = to_mantissa(ctx.dt, u.ms)
        P = {k: jnp.asarray(v).astype(dtype) for k, v in iaf_psc_alpha_propagators(p.tau_m, p.tau_syn_ex, p.tau_syn_in, p.C_m,
                                                   dt).items()}
        rule = self.spiking
        carry = rule.begin(s, p, dt)
        refractory = rule.refractory_mask(carry)

        E_L = mV(p.E_L)
        y3 = mV(s.V_m) - E_L
        I_ex, dI_ex = pA(s.I_syn_ex), dpA(s.dI_syn_ex)
        I_in, dI_in = pA(s.I_syn_in), dpA(s.dI_syn_in)
        # 1. membrane (old PSC states, latched current, I_e)
        y3_new = (P['P30'] * (pA(acting['I']) + pA(p.I_e))
                  + P['P31_ex'] * dI_ex + P['P32_ex'] * I_ex
                  + P['P31_in'] * dI_in + P['P32_in'] * I_in
                  + P['expm1_tau_m'] * y3 + y3)
        y3_new = jnp.maximum(y3_new, mV(p.V_min) - E_L)
        y3 = jnp.where(refractory, y3, y3_new)
        # 2. alpha PSCs, then the event jumps of this step
        I_ex_new = P['P21_ex'] * dI_ex + P['P22_ex'] * I_ex
        dI_ex_new = P['P11_ex'] * dI_ex + P['psc_init_ex'] * pA(inp.ex)
        I_in_new = P['P21_in'] * dI_in + P['P22_in'] * I_in
        dI_in_new = P['P11_in'] * dI_in + P['psc_init_in'] * pA(inp['in_'])
        s = s.replace(V_m=from_mantissa((y3 + E_L).astype(dtype), u.mV),
                      I_syn_ex=from_mantissa(I_ex_new.astype(dtype), u.pA),
                      dI_syn_ex=from_mantissa(dI_ex_new.astype(dtype), _PA_MS),
                      I_syn_in=from_mantissa(I_in_new.astype(dtype), u.pA),
                      dI_syn_in=from_mantissa(dI_in_new.astype(dtype), _PA_MS),
                      **new_latch)
        # 3. threshold, reset, refractoriness
        s, carry = rule.check(s, p, carry, detach_reset=rule_detach(self, ctx),
                              relaxed=ctx.relaxed)
        spike, counter = rule.end(carry)
        s = s.replace(spike=spike.astype(dtype), refractory_count=counter)
        return s, {'spike': s.spike}
