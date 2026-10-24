# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""BrainPy-style leaky integrate-and-fire neuron on the K0 contracts.

Numerics follow ``brainspike/_brainpy/lif.py`` (``LIF``): exponential-Euler
(exact for this linear ODE) propagation, ``delta`` voltage jumps added before the
threshold check, soft reset by default, no refractoriness.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import braintools
import brainunit as u

from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import Exact, Integrator
from brainspike._kernel.models._common import SpikingNeuron, as_record, latch, replace_rule, rule_detach
from brainspike._kernel.params import Init, Params, Record, States
from brainspike._kernel.ports import Current, Event, Ports
from brainspike._kernel.propagators import phi1
from brainspike._kernel.spiking import SpikeRule
from brainspike._kernel.units import from_mantissa, to_mantissa

__all__ = ['LIF']


def _margin(s: Record, p: Record) -> Any:
    return (s.V - p.V_th) / (p.V_th - p.V_reset)


def _soft_reset(s: Record, p: Record) -> Record:
    return s.replace(V=s.V - (p.V_th - p.V_reset))


def _hard_reset(s: Record, p: Record) -> Record:
    return s.replace(V=p.V_reset)


class LIF(SpikingNeuron):
    r"""Leaky integrate-and-fire neuron (BrainPy ``LIF`` numerics), surrogate-gradient ready.

    .. math::

       \tau \frac{dV}{dt} = -(V - V_{rest}) + R\, I(t), \qquad
       V \ge V_{th} \;\Rightarrow\; \text{spike},\; V \leftarrow V - (V_{th} - V_{reset})
       \;(\text{soft}) \text{ or } V_{reset}\;(\text{hard}).

    One step (``dt``): ``V <- V + dt * phi1(-dt / tau) * dV/dt`` with the current
    delivered in this step (exact for constant input), then ``V += delta``
    (event jumps, same step), then the threshold check and the reset. The spike
    output is the exact count (0/1) forward and the surrogate gradient backward.

    Compared with ``brainspike._brainpy.lif.LIF``: identical spikes and
    voltage for the same membrane input (the ``I`` port is declared
    ``latch=False``, decision D-a: like the BrainPy model it acts in the step it
    is delivered, so zero-delay feed-forward chains propagate within one step),
    except that the reset is applied in the spiking step, so the stored ``V`` at a
    spike step is the post-reset value (the old model stores the pre-reset value
    and resets at the start of the next step).

    Parameters
    ----------
    n : int
        Number of neurons.
    params : mapping, optional
        Overrides of ``R`` (1 ohm), ``tau`` (5 ms), ``V_th`` (1 mV), ``V_reset``
        (0 mV), ``V_rest`` (0 mV); scalars, ``(n,)`` arrays or distributions.
    init : mapping, optional
        Initial ``V`` (default 0 mV, as the BrainPy model).
    spk_reset : {'soft', 'hard'}, default 'soft'
        Soft reset subtracts ``V_th - V_reset``; hard reset sets ``V_reset``.
    surrogate : callable or None, optional
        Surrogate spike function (default ``braintools.surrogate.ReluGrad()``).
    **kwargs
        ``precision``, ``key``, ``trainable``, ``detach_reset``, ``name``.

    Notes
    -----
    Ports: ``I`` -- ``current`` in mA (``R * I`` in mV with the default units),
    ``latch=False`` (acts in the step it arrives, no latch state); ``delta`` -- ``event`` in mV, a jump of ``V`` in the step it arrives.
    Output ``spike`` -- ``event`` with ``max_multiplicity=1``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainspike._kernel import StepContext
        >>> from brainspike._kernel.models import LIF
        >>> lif = LIF(2)
        >>> ctx = StepContext.create(0.0 * u.ms, 0.1 * u.ms)
        >>> out = lif.update(ctx, {'delta': [0.5, 1.5] * u.mV})
        >>> out['spike'].tolist()
        [0.0, 1.0]
    """

    params = Params(R=1.0 * u.ohm, tau=5.0 * u.ms, V_th=1.0 * u.mV, V_reset=0.0 * u.mV,
                    V_rest=0.0 * u.mV)
    states = States(V=Init(0.0 * u.mV))
    # BrainPy family: the current acts in the step it arrives in (D-a)
    inputs = Ports(I=Current(u.mA, latch=False), delta=Event(unit=u.mV))
    outputs = Ports(spike=Event())
    spiking = SpikeRule(condition=_margin, jump=_soft_reset,
                        surrogate=braintools.surrogate.ReluGrad())
    integrator: Integrator = Exact()

    def __init__(self, n: int, params: Mapping[str, Any] | None = None,
                 init: Mapping[str, Any] | None = None, *, spk_reset: str = 'soft',
                 **kwargs: Any) -> None:
        if spk_reset not in ('soft', 'hard'):
            raise ValueError(f"spk_reset must be 'soft' or 'hard', got {spk_reset!r}")
        super().__init__(n, params, init, **kwargs)
        self.spk_reset = spk_reset
        if spk_reset == 'hard':
            self.spiking = replace_rule(self.spiking, jump=_hard_reset)

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Advance one step; see the class docstring for the order of operations."""
        inp = as_record(inp)
        acting, new_latch = latch(self, s, inp)
        dtype = to_mantissa(s.V, u.mV).dtype
        dt = to_mantissa(ctx.dt, u.ms)
        V = to_mantissa(s.V, u.mV)
        tau = to_mantissa(p.tau, u.ms)
        drive = to_mantissa(p.V_rest, u.mV) + to_mantissa(p.R, u.ohm) * to_mantissa(acting['I'], u.mA)
        V = V + dt * phi1(-dt / tau) * ((drive - V) / tau)
        V = V + to_mantissa(inp.delta, u.mV)
        s = s.replace(V=from_mantissa(V.astype(dtype), u.mV), **new_latch)
        rule = self.spiking
        carry = rule.begin(s, p, dt)
        s, carry = rule.check(s, p, carry, detach_reset=rule_detach(self, ctx),
                              relaxed=ctx.relaxed)
        spike, _ = rule.end(carry)
        s = s.replace(spike=spike.astype(dtype))
        return s, {'spike': s.spike}
