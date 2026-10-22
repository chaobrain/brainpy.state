# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST ``iaf_cond_exp`` on the K0 contracts (generated step, fixed-substep RK4).

Numerics follow ``brainpy_state/_nest_neuron/iaf_cond_exp.py`` (NEST
``models/iaf_cond_exp.cpp``); NEST's adaptive RKF45 is replaced by ``RK(4, k)``
with a fixed number of substeps (decision I1), chosen by the G5 convergence
study (``docs/specs/2026-10-21-k0-t3-models.md``).
"""
from __future__ import annotations

from typing import Any, ClassVar

import braintools
import brainunit as u

from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.integrators import RK, Integrator
from brainpy_state._kernel.models._common import SpikingNeuron
from brainpy_state._kernel.params import Init, Params, Record, States
from brainpy_state._kernel.ports import Current, Event, Ports
from brainpy_state._kernel.spiking import Clamp, SpikeRule

__all__ = ['iaf_cond_exp', 'IAF_COND_EXP_SUBSTEPS']

#: Default RK4 substeps per step (G5: max |V_m - NEST| < 1e-4 mV at dt = 0.1 ms,
#: table in docs/specs/2026-10-21-k0-t3-models.md).
IAF_COND_EXP_SUBSTEPS = 2


def _margin(s: Record, p: Record) -> Any:
    return (s.V_m - p.V_th) / (p.V_th - p.V_reset)


def _reset(s: Record, p: Record) -> Record:
    return s.replace(V_m=p.V_reset)


class iaf_cond_exp(SpikingNeuron):
    r"""Conductance-based leaky integrate-and-fire neuron with exponential conductances.

    .. math::

       C_m \frac{dV}{dt} = -g_L(\tilde V - E_L) - g_{ex}(\tilde V - E_{ex})
       - g_{in}(\tilde V - E_{in}) + I_e + I_{stim}, \qquad
       \frac{dg_X}{dt} = -\frac{g_X}{\tau_{syn,X}},

    with :math:`\tilde V = \min(V, V_{th})` (NEST bounds ``V`` inside the vector
    field). One step: integrate with ``RK(4, substeps)`` (``dV/dt = 0`` while
    refractory) using the current latched at the previous step; the threshold is
    checked once at the end of the step (``when='step'``, NEST); the receptor
    jumps ``g_X += w`` are applied after the propagation (the default
    ``on_input``).

    Parameters
    ----------
    n : int
        Number of neurons.
    params : mapping, optional
        Overrides of ``V_th`` (-55 mV), ``V_reset`` (-60 mV), ``t_ref`` (2 ms),
        ``g_L`` (16.6667 nS), ``C_m`` (250 pF), ``E_ex`` (0 mV), ``E_in``
        (-85 mV), ``E_L`` (-70 mV), ``tau_syn_ex`` (0.2 ms), ``tau_syn_in``
        (2 ms), ``I_e`` (0 pA).
    init : mapping, optional
        Initial values (``V_m`` defaults to ``E_L``, conductances to 0 nS).
    surrogate : callable or None, optional
        Surrogate spike function (default ``braintools.surrogate.ReluGrad()``).
    **kwargs
        ``precision``, ``key``, ``trainable``, ``integrator`` (e.g.
        ``RK(4, 8)``), ``detach_reset``, ``name``.

    Notes
    -----
    Ports: ``ex`` / ``in_`` -- receptor ``event`` ports (decision D-b),
    ``Event(unit=u.nS, into='g_ex')`` / ``into='g_in'``: the neuron owns the
    exponential kinetics, a delivered weight (nS, non-negative) jumps ``g_ex`` /
    ``g_in``, and projections into them carry no synapse model; ``I`` --
    ``current`` (pA), latched. Output ``spike`` -- ``event``, ``max_multiplicity=1``. Extra
    (e.g. NMDA Mg-gated) ``Conductance`` ports can be added with
    :func:`~brainpy_state._kernel.model.with_conductance_ports`; their current
    uses the bounded :math:`\tilde V` like the built-in ones.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel import StepContext
        >>> from brainpy_state._kernel.models import iaf_cond_exp
        >>> neu = iaf_cond_exp(1)
        >>> ctx = StepContext.create(0.0 * u.ms, 0.1 * u.ms)
        >>> _ = neu.update(ctx, {'ex': 5.0 * u.nS})
        >>> neu.dyn_states['g_ex'].value.to_decimal(u.nS).tolist()
        [5.0]
    """

    params = Params(V_th=-55.0 * u.mV, V_reset=-60.0 * u.mV, t_ref=2.0 * u.ms,
                    g_L=16.6667 * u.nS, C_m=250.0 * u.pF, E_ex=0.0 * u.mV, E_in=-85.0 * u.mV,
                    E_L=-70.0 * u.mV, tau_syn_ex=0.2 * u.ms, tau_syn_in=2.0 * u.ms,
                    I_e=0.0 * u.pA)
    states = States(V_m=Init(lambda p: p.E_L), g_ex=Init(0.0 * u.nS), g_in=Init(0.0 * u.nS))
    inputs = Ports(ex=Event(unit=u.nS, into='g_ex'),  # neuron-owned receptor ports (D-b)
                   in_=Event(unit=u.nS, into='g_in'),
                   I=Current(u.pA, latch=True))  # NEST CurrentEvent: next step (D-a)
    outputs = Ports(spike=Event())
    spiking = SpikeRule(condition=_margin, jump=_reset, refractory=Clamp('V_m', 't_ref'),
                        surrogate=braintools.surrogate.ReluGrad(), when='step')
    integrator: ClassVar[Integrator] = RK(order=4, substeps=IAF_COND_EXP_SUBSTEPS)
    accepts_conductance_ports: ClassVar[bool] = True

    def derivatives(self, s: Record, p: Record, inp: Record, ctx: StepContext) -> dict:
        """NEST ``iaf_cond_exp_dynamics`` (refractory ``dV = 0`` is applied by the step)."""
        V = u.math.minimum(s.V_m, p.V_th)
        I_syn = s.g_ex * (V - p.E_ex) + s.g_in * (V - p.E_in)
        I_L = p.g_L * (V - p.E_L)
        I_g = self.conductance_port_current(inp, V, p)  # added (gated) ports, per substep
        return {'V_m': (-I_L - I_syn + I_g + p.I_e + inp.I) / p.C_m,
                'g_ex': -s.g_ex / p.tau_syn_ex,
                'g_in': -s.g_in / p.tau_syn_in}

    # on_input: the default receptor jumps g_ex += w_ex, g_in += w_in after the
    # propagation (NEST spike_exc_ / spike_inh_, decision D-b)
