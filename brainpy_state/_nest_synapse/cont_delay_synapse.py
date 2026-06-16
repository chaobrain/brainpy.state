# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``cont_delay_synapse`` — static synapse with a continuous (sub-dt) delay.

Rebuilt as a frozen parameter spec plus the plain static ``update(state, ctx)``
rule kernel on
:class:`~brainpy_state._nest_network._event_plastic.EventPlasticProj`. The synapse is
non-plastic: no per-edge state evolves and the effective weight is just the
constant (per-edge) ``weight`` — identical to :class:`static_synapse`. Its one
distinguishing feature, a delay that need *not* be an integer multiple of the
simulation step, is realised by the substrate's ``fractional_delay`` output-carry
seam, opted into here via the class attribute ``fractional_delay = True``.
"""
from __future__ import annotations
from brainpy_state._nest_base._base import NESTSynapse

import brainstate
import brainunit as u
from brainstate.typing import ArrayLike

from brainpy_state._nest_base._plastic_base import unit_of, validate_delay, validate_receptor_type, weight_to_pa

__all__ = ['cont_delay_synapse']


def _validate_min_delay(delay) -> None:
    r"""Reject a delay shorter than the resolution (NEST cont_delay floor).

    NEST requires a continuous delay ``>= h`` (the simulation resolution); a
    shorter delay has no on-grid floor to deliver at. The check is *best-effort*:
    it fires only when ``dt`` is already established in the environment — the
    normal :class:`~brainpy_state.Simulator` workflow sets ``dt`` before building
    synapses, and every parity/rule test sets it explicitly. If no ``dt`` is set
    yet the floor cannot be evaluated, so the check is deferred (the substrate
    decomposes the delay against the live ``dt`` in ``init_state``).
    """
    try:
        dt = brainstate.environ.get_dt()
    except Exception:
        return
    dt_ms = float(u.Quantity(dt).to_decimal(u.ms))
    d_ms = (float(u.Quantity(delay).to_decimal(u.ms))
            if isinstance(delay, u.Quantity) else float(delay))
    if d_ms < dt_ms - 1e-9:
        raise ValueError(
            'cont_delay_synapse: continuous delays cannot be shorter than the '
            'simulation resolution (dt).'
        )


class cont_delay_synapse(NESTSynapse):
    r"""Static synapse with a continuous, sub-timestep delay (NEST ``cont_delay_synapse``).

    Delivery is identical to :class:`static_synapse` — each presynaptic spike on
    edge ``e`` delivers the constant amplitude ``weight[e]`` (pA), no per-edge
    state evolves — except that ``delay`` may fall *between* grid steps. The
    substrate honours the sub-step part by delivering at the integer floor delay
    and splitting the delivered amplitude across the two bracketing grid steps
    (the ``fractional_delay`` output-carry seam); an integer-multiple delay
    reduces exactly to :class:`static_synapse`.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge synaptic weight (pA; bare numbers interpreted as pA, sign
        preserved). Default ``1.0`` pA.
    delay : Quantity, optional
        Homogeneous continuous delay; must be finite, strictly positive and
        ``>= dt`` (the NEST resolution floor). Need not be a multiple of ``dt``.
        Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (non-negative integer). Default ``0``.

    Raises
    ------
    ValueError
        If ``delay`` is non-finite, non-positive, or shorter than the simulation
        resolution ``dt`` (NEST floor; checked when ``dt`` is known), or if
        ``receptor_type < 0``.

    See Also
    --------
    static_synapse : Fixed grid-delay delivery (the integer-multiple-delay limit).
    bernoulli_synapse : Static delivery with stochastic (Bernoulli) transmission.

    Notes
    -----
    NEST ``models/cont_delay_synapse.h`` decomposes a continuous delay :math:`d`
    into an integer step count and a sub-step offset. With resolution :math:`h`,
    ``set_status`` (``cont_delay_synapse_impl.h`` lines 72-87) computes
    :math:`\mathrm{frac} = \operatorname{modf}(d/h)`: an on-grid delay
    (:math:`\mathrm{frac}=0`) keeps ``delay_steps = d/h``, ``delay_offset_ = 0``;
    an off-grid delay sets ``delay_steps = \lfloor d/h\rfloor + 1`` and
    ``delay_offset_ = h(1-\mathrm{frac})``, so the realised delay is
    :math:`\mathrm{delay\_steps}\cdot h - \mathrm{delay\_offset\_} = d`.
    ``send()`` (lines 218-246) emits one spike carrying that offset; only a
    *precise* (``*_ps``) postsynaptic neuron integrates the true off-grid arrival,
    whereas a *grid* neuron ignores the offset and receives the event at
    :math:`\lceil d/h\rceil`.

    **Grid-faithful first-moment scheme.** :class:`EventPlasticProj` is a grid
    integrator and cannot place an event between steps. Instead, because this spec
    sets ``fractional_delay = True``, the substrate delivers the (binary) event at
    the integer floor delay :math:`k_{lo} = \lfloor d/h\rfloor` and splits the
    delivered amplitude across the two bracketing grid steps with a one-step FIR
    :math:`[\,1-\mathrm{frac},\ \mathrm{frac}\,]`,
    :math:`\mathrm{frac} = d/h - k_{lo}`. This conserves total charge exactly
    (:math:`(1-\mathrm{frac}) + \mathrm{frac} = 1`, so the time-integrated
    postsynaptic current — hence the integrated depolarization — is preserved) and
    places the arrival *centroid* exactly at :math:`d` (first moment exact).
    Measured against NEST's precise ``iaf_psc_exp_ps`` (which integrates the true
    off-grid arrival at :math:`t+d`), the integrated depolarization
    :math:`\int V_m` and the broad EPSP peak amplitude therefore agree to
    :math:`\sim 10^{-4}` relative and the peak *timing* to within one step; the
    sole residual is a bounded sub-step *onset transient* (a ripple at the PSC
    onset discontinuity, :math:`\mathrm{frac}`-dependent and first order in
    :math:`h/\tau` there, vanishing as :math:`h\to 0`) that a grid integrator
    cannot avoid. ``frac == 0`` (an integer-multiple delay) collapses to a plain
    grid delay, byte-identical to :class:`static_synapse`. The seam is
    delivery-side only: the event matmul still sees a binary presynaptic vector
    (``ctx.pre_spike`` stays binary), and the rule kernel itself is the trivial
    static rule.

    References
    ----------
    .. [1] NEST ``models/cont_delay_synapse.h`` ``send()`` (lines 218-246) and
           ``cont_delay_synapse_impl.h`` ``set_status`` (lines 72-87).
    .. [2] Morrison A, Straube S, Plesser HE, Diesmann M (2007). Exact subthreshold
           integration with continuous spike times in discrete-time neural network
           simulations. Neural Computation, 19(1):47-79.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state import cont_delay_synapse
       >>> brainstate.environ.set(dt=0.1 * u.ms)
       >>> s = cont_delay_synapse(weight=20.0 * u.pA, delay=0.17 * u.ms)
       >>> s.fractional_delay
       True
       >>> s.stochastic
       False
       >>> s.edge_state_init()
       {}
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False
    pre_trace_tau = None
    post_trace_tau = None
    fractional_delay = True              # drives the substrate sub-dt output-carry seam

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        _validate_min_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

    def edge_state_init(self) -> dict:
        return {}

    def update(self, state, ctx):
        # Static delivery: the effective weight is just the constant per-edge
        # weight (no state evolves). The continuous-delay behaviour lives in the
        # substrate's fractional_delay output-carry seam, not in this kernel.
        return state, state['weight']
