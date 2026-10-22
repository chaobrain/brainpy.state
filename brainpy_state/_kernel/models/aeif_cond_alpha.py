# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST ``aeif_cond_alpha`` on the K0 contracts (generated step, in-substep spikes).

Numerics follow ``brainpy_state/_nest_neuron/aeif_cond_alpha.py`` (NEST
``models/aeif_cond_alpha.cpp``); NEST's adaptive RKF45 with spike handling
after every accepted GSL step is replaced by ``RK(4, k)`` with the spike check,
the multi-state jump (``V_m = V_reset``, ``w += b``) and the refractory clamp
after every substep. The default substep count comes from the G5 convergence
study (``docs/specs/2026-10-21-k0-t3-models.md``).
"""
from __future__ import annotations

import math
from typing import Any, ClassVar

import braintools
import brainunit as u

from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.integrators import RK, Integrator
from brainpy_state._kernel.models._common import SpikingNeuron
from brainpy_state._kernel.params import Init, Params, Record, States
from brainpy_state._kernel.ports import Current, Event, Ports
from brainpy_state._kernel.spiking import Clamp, SpikeRule

__all__ = ['aeif_cond_alpha', 'AEIF_COND_ALPHA_SUBSTEPS']

#: Default RK4 substeps per step at dt = 0.1 ms (G5 table in docs/specs/2026-10-21-k0-t3-models.md):
#: subthreshold traces within 1e-3 mV of NEST, spike times equal to NEST; post-reset
#: error is first order in the substep (event location), ~2e-2 mV.
AEIF_COND_ALPHA_SUBSTEPS = 16

_NS_MS = u.nS / u.ms


def _v_peak(p: Record) -> Any:
    # NEST V_.V_peak: V_peak for Delta_T > 0, V_th for the IAF limit Delta_T == 0
    return u.math.where(p.Delta_T > 0 * u.mV, p.V_peak, p.V_th)


def _margin(s: Record, p: Record) -> Any:
    vp = _v_peak(p)
    return (s.V_m - vp) / (vp - p.V_reset)


def _reset(s: Record, p: Record) -> Record:
    return s.replace(V_m=p.V_reset, w=s.w + p.b)


class aeif_cond_alpha(SpikingNeuron):
    r"""Adaptive exponential integrate-and-fire neuron, alpha conductances (NEST ``aeif_cond_alpha``).

    .. math::

       C_m \frac{dV}{dt} = -g_L(\tilde V - E_L)
         + g_L \Delta_T e^{(\tilde V - V_{th})/\Delta_T}
         - g_{ex}(\tilde V - E_{ex}) - g_{in}(\tilde V - E_{in}) - w + I_e + I_{stim},

       \tau_w \frac{dw}{dt} = a(\tilde V - E_L) - w, \qquad
       \frac{d\,dg_X}{dt} = -\frac{dg_X}{\tau_X}, \quad
       \frac{dg_X}{dt} = dg_X - \frac{g_X}{\tau_X},

    with :math:`\tilde V = \min(V, V_{peak})`. A spike (``V >= V_peak``, or
    ``V_th`` when ``Delta_T == 0``) is detected after every RK substep, sets
    ``V = V_reset``, ``w += b`` and starts ``ceil(t_ref / dt)`` refractory steps
    (``V`` held); with short or no refractoriness several spikes per step are
    counted. ``event`` jumps ``dg_X += (e / tau_X) w`` follow the propagation.

    Parameters
    ----------
    n : int
        Number of neurons.
    params : mapping, optional
        Overrides of ``V_peak`` (0 mV), ``V_reset`` (-60 mV), ``t_ref`` (0 ms),
        ``g_L`` (30 nS), ``C_m`` (281 pF), ``E_ex`` (0 mV), ``E_in`` (-85 mV),
        ``E_L`` (-70.6 mV), ``Delta_T`` (2 mV), ``tau_w`` (144 ms), ``a``
        (4 nS), ``b`` (80.5 pA), ``V_th`` (-50.4 mV), ``tau_syn_ex`` (0.2 ms),
        ``tau_syn_in`` (2 ms), ``I_e`` (0 pA).
    init : mapping, optional
        Initial values (``V_m`` defaults to ``E_L``, the rest to 0).
    surrogate : callable or None, optional
        Surrogate spike function (default ``braintools.surrogate.ReluGrad()``).
    **kwargs
        ``precision``, ``key``, ``trainable``, ``integrator``, ``detach_reset``,
        ``name``.

    Notes
    -----
    Ports: ``ex`` / ``in_`` -- receptor ``event`` ports (decision D-b),
    ``Event(unit=u.nS, into='g_ex')`` / ``into='g_in'``: the neuron owns the
    alpha kinetics; a delivered weight (nS, non-negative, the peak of ``g``) is
    scaled by ``e / tau_syn`` into ``dg`` (``on_input``); ``I`` -- ``current`` (pA),
    latched. Output ``spike`` -- ``event`` with ``max_multiplicity=None``
    (several spikes per step are possible), so projections use the exact count
    payload path.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel import StepContext
        >>> from brainpy_state._kernel.models import aeif_cond_alpha
        >>> neu = aeif_cond_alpha(1, params=dict(I_e=50000.0 * u.pA))
        >>> ctx = StepContext.create(0.0 * u.ms, 1.0 * u.ms)
        >>> n_spikes = float(neu.update(ctx)['spike'][0])
        >>> n_spikes >= 2
        True
    """

    params = Params(V_peak=0.0 * u.mV, V_reset=-60.0 * u.mV, t_ref=0.0 * u.ms, g_L=30.0 * u.nS,
                    C_m=281.0 * u.pF, E_ex=0.0 * u.mV, E_in=-85.0 * u.mV, E_L=-70.6 * u.mV,
                    Delta_T=2.0 * u.mV, tau_w=144.0 * u.ms, a=4.0 * u.nS, b=80.5 * u.pA,
                    V_th=-50.4 * u.mV, tau_syn_ex=0.2 * u.ms, tau_syn_in=2.0 * u.ms,
                    I_e=0.0 * u.pA)
    states = States(V_m=Init(lambda p: p.E_L), dg_ex=Init(0.0 * _NS_MS), g_ex=Init(0.0 * u.nS),
                    dg_in=Init(0.0 * _NS_MS), g_in=Init(0.0 * u.nS), w=Init(0.0 * u.pA))
    inputs = Ports(ex=Event(unit=u.nS, into='g_ex'),  # neuron-owned receptor ports (D-b)
                   in_=Event(unit=u.nS, into='g_in'),
                   I=Current(u.pA, latch=True))  # NEST CurrentEvent: next step (D-a)
    outputs = Ports(spike=Event(max_multiplicity=None))
    spiking = SpikeRule(condition=_margin, jump=_reset, refractory=Clamp('V_m', 't_ref'),
                        surrogate=braintools.surrogate.ReluGrad())
    integrator: ClassVar[Integrator] = RK(order=4, substeps=AEIF_COND_ALPHA_SUBSTEPS)
    accepts_conductance_ports: ClassVar[bool] = True

    def derivatives(self, s: Record, p: Record, inp: Record, ctx: StepContext) -> dict:
        """NEST ``aeif_cond_alpha_dynamics`` (refractory ``dV = 0`` is applied by the step)."""
        V = u.math.minimum(s.V_m, p.V_peak)
        pos = p.Delta_T > 0 * u.mV
        dT = u.math.where(pos, p.Delta_T, 1.0 * u.mV)  # double-where: no 0/0 at Delta_T = 0
        I_spike = u.math.where(pos, p.g_L * dT * u.math.exp((V - p.V_th) / dT), 0.0 * u.pA)
        I_syn = s.g_ex * (V - p.E_ex) + s.g_in * (V - p.E_in)
        I_g = self.conductance_port_current(inp, V, p)  # added (gated) ports, per substep
        dV = (-p.g_L * (V - p.E_L) + I_spike - I_syn + I_g - s.w + p.I_e + inp.I) / p.C_m
        return {'V_m': dV,
                'dg_ex': -s.dg_ex / p.tau_syn_ex,
                'g_ex': s.dg_ex - s.g_ex / p.tau_syn_ex,
                'dg_in': -s.dg_in / p.tau_syn_in,
                'g_in': s.dg_in - s.g_in / p.tau_syn_in,
                'w': (p.a * (V - p.E_L) - s.w) / p.tau_w}

    def on_input(self, s: Record, p: Record, inp: Record, ctx: StepContext) -> Record:
        """Alpha-conductance jumps after the propagation (NEST ``dg += g0 * w``)."""
        return s.replace(dg_ex=s.dg_ex + math.e / p.tau_syn_ex * inp.ex,
                         dg_in=s.dg_in + math.e / p.tau_syn_in * inp['in_'])
