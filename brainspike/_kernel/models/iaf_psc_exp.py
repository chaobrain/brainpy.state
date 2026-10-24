# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST ``iaf_psc_exp`` on the kernel contracts (exact propagator integration).

Numerics follow ``brainspike/_nest_neuron/iaf_psc_exp.py`` (NEST
``models/iaf_psc_exp.cpp``, deterministic threshold) with the float32-safe
:func:`~brainspike._kernel.propagators.propagator_exp`, computed inside ``step``
from traced parameters and ``ctx.dt``.
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
from brainspike._kernel.params import Init, Params, Record, States
from brainspike._kernel.ports import Current, Event, Ports
from brainspike._kernel.propagators import propagator_exp
from brainspike._kernel.spiking import Clamp, SpikeRule
from brainspike._kernel.units import from_mantissa, to_mantissa

__all__ = ['iaf_psc_exp']


def _margin(s: Record, p: Record) -> Any:
    return (s.V_m - p.V_th) / (p.V_th - p.V_reset)


def _reset(s: Record, p: Record) -> Record:
    return s.replace(V_m=p.V_reset)


class iaf_psc_exp(SpikingNeuron):
    r"""Leaky integrate-and-fire neuron with exponential PSCs (NEST ``iaf_psc_exp``).

    .. math::

       C_m \frac{dV}{dt} = -\frac{C_m}{\tau_m}(V - E_L) + I_{syn,ex} + I_{syn,in}
       + I_e + I_{stim}, \qquad \tau_X \frac{dI_{syn,X}}{dt} = -I_{syn,X} .

    One step (NEST ``iaf_psc_exp::update`` order, ``delta = 0``): if not
    refractory, ``V`` is propagated exactly with the PSCs and the current latched at
    the previous step plus ``I_e``; the PSCs decay; the ``event`` inputs of this
    step add to them (weights in pA, NEST sign convention); then the threshold
    check, reset to ``V_reset`` and ``ceil(t_ref / dt)`` refractory steps.

    Numerics follow NEST ``models/iaf_psc_exp.cpp`` with ``delta = 0``
    (deterministic threshold) and the float32-safe ``P21`` of
    :func:`~brainspike._kernel.propagators.propagator_exp` (singular
    ``tau_syn == tau_m`` handled), computed inside ``step`` from traced parameters
    and ``ctx.dt``. Promoted from the K1a-G4 microcircuit (K1a-7).

    Parameters
    ----------
    n : int
        Number of neurons.
    params : mapping, optional
        Overrides of ``E_L`` (-70 mV), ``C_m`` (250 pF), ``tau_m`` (10 ms),
        ``t_ref`` (2 ms), ``V_th`` (-55 mV), ``V_reset`` (-70 mV),
        ``tau_syn_ex`` / ``tau_syn_in`` (2 ms), ``I_e`` (0 pA) -- NEST defaults.
        All per neuron, traced.
    init : mapping, optional
        Initial values (``V_m`` defaults to ``E_L``, PSCs to 0).
    surrogate : callable or None, optional
        Surrogate spike function (default ``braintools.surrogate.ReluGrad()``)
        applied to ``(V_m - V_th) / (V_th - V_reset)``.
    **kwargs
        ``precision``, ``key``, ``trainable``, ``detach_reset``, ``name``.

    Notes
    -----
    Ports: ``ex`` / ``in_`` -- receptor ``event`` ports (decision D-b),
    ``Event(unit=u.pA, into='I_syn_ex')`` / ``into='I_syn_in'``, weights in pA
    (the PSC amplitude), each decaying with its own ``tau_syn_ex`` /
    ``tau_syn_in``; the weight sign is kept (NEST convention: inhibitory weights
    are negative; the NEST front-end routes ``w > 0`` to ``ex`` and ``w < 0`` to
    ``in_``). ``I`` -- ``current`` in pA, latched (NEST receptor 0). Output
    ``spike`` -- ``event``, ``max_multiplicity=1``.

    Not modelled (K1b follow-ups): NEST's receptor-1 current, which enters
    ``I_syn_ex`` through the excitatory filter (``(1 - P11ex) * i_1``), and the
    escape-noise threshold (``rho`` / ``delta > 0``).

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainspike._kernel import StepContext
        >>> from brainspike._kernel.models import iaf_psc_exp
        >>> neu = iaf_psc_exp(1)
        >>> ctx = StepContext.create(0.0 * u.ms, 0.1 * u.ms)
        >>> _ = neu.update(ctx, {'ex': 100.0 * u.pA})
        >>> float(neu.dyn_states['I_syn_ex'].value.to_decimal(u.pA)[0])
        100.0
    """

    params = Params(E_L=-70.0 * u.mV, C_m=250.0 * u.pF, tau_m=10.0 * u.ms, t_ref=2.0 * u.ms,
                    V_th=-55.0 * u.mV, V_reset=-70.0 * u.mV, tau_syn_ex=2.0 * u.ms,
                    tau_syn_in=2.0 * u.ms, I_e=0.0 * u.pA)
    states = States(V_m=Init(lambda p: p.E_L), I_syn_ex=Init(0.0 * u.pA), I_syn_in=Init(0.0 * u.pA))
    inputs = Ports(ex=Event(unit=u.pA, into='I_syn_ex'),  # neuron-owned receptor ports (D-b)
                   in_=Event(unit=u.pA, into='I_syn_in'),
                   I=Current(u.pA, latch=True))  # NEST CurrentEvent (i_0_): next step (D-a)
    outputs = Ports(spike=Event())
    spiking = SpikeRule(condition=_margin, jump=_reset, refractory=Clamp('V_m', 't_ref'),
                        surrogate=braintools.surrogate.ReluGrad())
    integrator: Integrator = Exact()

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Advance one step (NEST ``iaf_psc_exp::update`` order)."""
        inp = as_record(inp)
        acting, new_latch = latch(self, s, inp)
        mV = lambda x: to_mantissa(x, u.mV)
        pA = lambda x: to_mantissa(x, u.pA)
        dtype = mV(s.V_m).dtype
        dt = to_mantissa(ctx.dt, u.ms)
        tau_m, c_m = to_mantissa(p.tau_m, u.ms), to_mantissa(p.C_m, u.pF)
        tau_ex, tau_in = to_mantissa(p.tau_syn_ex, u.ms), to_mantissa(p.tau_syn_in, u.ms)
        P22 = jnp.exp(-dt / tau_m).astype(dtype)
        P20 = (-tau_m / c_m * jnp.expm1(-dt / tau_m)).astype(dtype)
        P11ex = jnp.exp(-dt / tau_ex).astype(dtype)
        P11in = jnp.exp(-dt / tau_in).astype(dtype)
        P21ex = jnp.asarray(propagator_exp(tau_ex, tau_m, c_m, dt)).astype(dtype)
        P21in = jnp.asarray(propagator_exp(tau_in, tau_m, c_m, dt)).astype(dtype)
        rule = self.spiking
        carry = rule.begin(s, p, dt)
        refractory = rule.refractory_mask(carry)

        E_L = mV(p.E_L)
        y = mV(s.V_m) - E_L
        I_ex, I_in = pA(s.I_syn_ex), pA(s.I_syn_in)
        y_new = y * P22 + I_ex * P21ex + I_in * P21in + (pA(p.I_e) + pA(acting['I'])) * P20
        y = jnp.where(refractory, y, y_new)
        I_ex = I_ex * P11ex + pA(inp.ex)
        I_in = I_in * P11in + pA(inp['in_'])
        s = s.replace(V_m=from_mantissa((y + E_L).astype(dtype), u.mV),
                      I_syn_ex=from_mantissa(I_ex.astype(dtype), u.pA),
                      I_syn_in=from_mantissa(I_in.astype(dtype), u.pA),
                      **new_latch)
        s, carry = rule.check(s, p, carry, detach_reset=rule_detach(self, ctx),
                              relaxed=ctx.relaxed)
        spike, counter = rule.end(carry)
        s = s.replace(spike=spike.astype(dtype), refractory_count=counter)
        return s, {'spike': s.spike}
