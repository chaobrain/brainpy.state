# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``urbanczik_synapse`` — dendritic prediction-error plastic spec + pure rule.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on the
:class:`~brainpy_state._network._event_plastic.VoltageCoupledPlasticProj`
substrate (primitive #2). The Urbanczik-Senn rule (Urbanczik & Senn, 2014) makes
dendritic synapses learn so the dendritic potential predicts the somatically
imposed firing rate: each synapse integrates the product of its presynaptic trace
and the post neuron's **dendritic prediction error** δΠ, low-pass filtered over
``tau_Delta``.

The substrate maintains the **two presynaptic traces** the rule declares
(``pre_trace_tau = (tau_L, tau_s)``, one per-neuron column each, gathered per
edge as ``ctx.pre_traces[:, 0]`` / ``[:, 1]``) and, via the post-state reader,
samples the post neuron's δΠ per edge each step
(``post_state_reads = ('delta_Pi',)``). The kernel accumulates the per-step
plasticity integrand ``PI = (tau_L_trace - tau_s_trace) · δΠ`` and its
``tau_Delta`` low-pass, anchoring the weight on the spec's initial value (NEST
``w = init_weight + (PI_integral - PI_exp_integral) · PREFACTOR``, clipped to
``[Wmin, Wmax]``).

Notes
-----
**Online reformulation.** NEST applies the rule event-driven: on each presynaptic
spike ``send()`` integrates δΠ over the dendritic-delay window
``(t_last - d, t_spike - d]``. The substrate instead runs the pure kernel **every
grid step**. Accumulating ``PI`` every step equals NEST's window integral, because
the windows tile every step after the first presynaptic spike and both traces are
zero before it (so ``PI = 0``). The ``+1`` trace jump on a spike cancels in the
``tau_L - tau_s`` difference, and the running low-pass ``PI_exp_integral`` equals
NEST's at presynaptic-spike times. The substrate **gates delivery** by the actual
presynaptic spikes (CSR event matmul), so the *delivered* weight matches NEST at
presynaptic-spike steps; the weight State itself evolves continuously between
spikes (it re-synchronises with NEST at every spike).

**Dendritic-parameter location.** In NEST the prefactor reads the dendritic
compartment's ``C_m``, ``g_L`` and ``tau_syn`` from the *post neuron* at send
time. Here the spec+rule is frozen before ``connect``, so these are
**synapse-spec** constructor arguments defaulting to the
``pp_cond_exp_mc_urbanczik`` dendritic defaults (``C_m=300 pF``, ``g_L=30 nS``,
``tau_syn_ex=tau_syn_in=3 ms``); for parity the drive passes values matching the
post neuron. ``tau_L = C_m / g_L`` and ``tau_s`` is the excitatory or inhibitory
dendritic synaptic time constant selected by the **initial weight sign** (NEST's
``weight > 0 ? tau_syn_ex : tau_syn_in``; the sign-consistency constraints below
keep the weight from crossing zero, so ``tau_s`` is fixed).

References
----------
.. [1] Urbanczik, R., & Senn, W. (2014). Learning by the dendritic prediction of
   somatic spiking. *Neuron*, 81(3), 521-528.
.. [2] NEST ``models/urbanczik_synapse.h`` +
   ``nestkernel/urbanczik_archiving_node_impl.h``.
"""
from __future__ import annotations
from ._base import NESTPlasticity

import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainstate.typing import ArrayLike

from ._plastic_base import (
    to_ms, to_scalar_float, unit_of, validate_delay, validate_receptor_type,
    weight_to_pa,
)

__all__ = ['urbanczik_synapse']


def _sign_like_wmin(x: float) -> int:
    # NEST set_status sign test for the *lower* bound: (x >= 0) - (x < 0).
    return int(x >= 0.0) - int(x < 0.0)


def _sign_like_wmax(x: float) -> int:
    # NEST set_status sign test for the *upper* bound: (x > 0) - (x <= 0).
    return int(x > 0.0) - int(x <= 0.0)


def _to_unit_mantissa(value, unit, *, name: str) -> float:
    """Return ``value`` as a scalar mantissa in ``unit`` (bare numbers assume ``unit``)."""
    if isinstance(value, u.Quantity):
        return float(value.to_decimal(unit))
    return to_scalar_float(value, name=name)


class urbanczik_synapse(NESTPlasticity):
    r"""Dendritic prediction-error plasticity synapse spec (NEST ``urbanczik_synapse``).

    Implements the Urbanczik-Senn rule: with :math:`\bar{s}_L`, :math:`\bar{s}_s`
    the presynaptic traces at the dendritic membrane (``tau_L``) and synaptic
    (``tau_s``) time constants, and :math:`\delta\Pi` the post neuron's dendritic
    prediction error read per edge each step,

    .. math::

       \mathrm{PI}(t) &= (\bar{s}_L(t) - \bar{s}_s(t))\,\delta\Pi(t) \\
       \mathrm{PI_{int}}(t) &= \textstyle\sum_{s \le t} \mathrm{PI}(s) \\
       \mathrm{PI_{exp}}(t) &= \textstyle\sum_{s \le t}
           e^{-(t-s)/\tau_\Delta}\,\mathrm{PI}(s) \\
       w(t) &= \mathrm{clip}\big(w_0 + (\mathrm{PI_{int}} - \mathrm{PI_{exp}})\,
           P,\; W_{\min},\, W_{\max}\big)

    with the NEST prefactor :math:`P = 15\,C_m\,\tau_s\,\eta /
    (g_L (\tau_L - \tau_s))` (dendritic ``C_m`` [pF], ``g_L`` [nS],
    ``tau_L = C_m/g_L`` [ms], ``tau_s`` [ms]) and :math:`w_0` the spec's initial
    weight. The two sums are carried per edge (``edge_state_init``); the
    presynaptic traces are the substrate's two-column per-pre trace seam
    (``pre_trace_tau = (tau_L, tau_s)``).

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Initial per-edge weight; bare numbers default to **pA** (the dendritic
        current port). Its sign selects ``tau_s`` and must agree with
        ``Wmin``/``Wmax`` under NEST's sign tests. Default ``1.0 pA``.
    delay : Quantity, optional
        Axonal/dendritic delay (> 0). Default ``1.0 ms``. For live-NEST parity the
        drive sets ``delay = dt`` (one grid step) to align the online reader's
        one-step lag to NEST's minimal dendritic-delay window.
    eta : float, optional
        Learning rate :math:`\eta`. Default ``0.07``.
    tau_Delta : Quantity, optional
        Plasticity low-pass time constant :math:`\tau_\Delta` (> 0). Default
        ``100.0 ms``.
    Wmin, Wmax : float, optional
        Lower / upper weight clamp. Defaults ``0.0`` / ``100.0``.
    receptor_type : int, optional
        Informational default target port (``3`` = dendritic excitatory). The
        actual routing is driven by ``connect(..., receptor_type=...)``; the post
        neuron resolves it to a delta-input channel label. Default ``3``.
    dend_C_m, dend_g_L : Quantity, optional
        Dendritic compartment capacitance / leak conductance for the prefactor and
        ``tau_L = C_m/g_L``. Defaults ``300.0 pF`` / ``30.0 nS``
        (``pp_cond_exp_mc_urbanczik`` dendrite).
    dend_tau_syn_ex, dend_tau_syn_in : Quantity, optional
        Dendritic excitatory / inhibitory synaptic time constants; ``tau_s`` is the
        one selected by the initial weight sign. Defaults ``3.0 ms`` / ``3.0 ms``.

    Notes
    -----
    See the module docstring for the online-vs-event-driven equivalence, the
    dendritic-parameter-location divergence, and the parity posture.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state import urbanczik_synapse
        >>> s = urbanczik_synapse(weight=1.0 * u.pA)
        >>> s.is_homogeneous_weight, s.post_state_reads
        (False, ('delta_Pi',))
        >>> s.post_trace_tau, s.edge_state_init()
        (None, {'PI_integral': 0.0, 'PI_exp_integral': 0.0})
        >>> [float(u.Quantity(t).to_decimal(u.ms)) for t in s.pre_trace_tau]
        [10.0, 3.0]
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False
    post_trace_tau = None
    # primitive #2 reader: the post neuron's dendritic prediction error per edge
    post_state_reads = ('delta_Pi',)

    def __init__(
        self,
        weight: ArrayLike = 1.0 * u.pA,
        delay: ArrayLike = 1.0 * u.ms,
        eta: ArrayLike = 0.07,
        tau_Delta: ArrayLike = 100.0 * u.ms,
        Wmin: ArrayLike = 0.0,
        Wmax: ArrayLike = 100.0,
        receptor_type: int = 3,
        dend_C_m: ArrayLike = 300.0 * u.pF,
        dend_g_L: ArrayLike = 30.0 * u.nS,
        dend_tau_syn_ex: ArrayLike = 3.0 * u.ms,
        dend_tau_syn_in: ArrayLike = 3.0 * u.ms,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.eta = to_scalar_float(eta, name='eta')
        self.tau_Delta = to_ms(tau_Delta, name='tau_Delta') * u.ms
        self._tau_Delta_ms = to_ms(tau_Delta, name='tau_Delta')
        self.Wmin = to_scalar_float(Wmin, name='Wmin')
        self.Wmax = to_scalar_float(Wmax, name='Wmax')

        # Dendritic-compartment params (NEST units: pF, nS, ms) -> prefactor / taus.
        self.dend_C_m = dend_C_m if isinstance(dend_C_m, u.Quantity) else dend_C_m * u.pF
        self.dend_g_L = dend_g_L if isinstance(dend_g_L, u.Quantity) else dend_g_L * u.nS
        self.dend_tau_syn_ex = (dend_tau_syn_ex if isinstance(dend_tau_syn_ex, u.Quantity)
                                else dend_tau_syn_ex * u.ms)
        self.dend_tau_syn_in = (dend_tau_syn_in if isinstance(dend_tau_syn_in, u.Quantity)
                                else dend_tau_syn_in * u.ms)
        C_m_pF = _to_unit_mantissa(dend_C_m, u.pF, name='dend_C_m')
        g_L_nS = _to_unit_mantissa(dend_g_L, u.nS, name='dend_g_L')
        tau_syn_ex_ms = _to_unit_mantissa(dend_tau_syn_ex, u.ms, name='dend_tau_syn_ex')
        tau_syn_in_ms = _to_unit_mantissa(dend_tau_syn_in, u.ms, name='dend_tau_syn_in')

        self._init_w = float(u.get_mantissa(self.weight))
        # tau_L = C_m/g_L; tau_s selected by the initial weight sign (NEST send()).
        self._tau_L_ms = C_m_pF / g_L_nS
        self._tau_s_ms = tau_syn_ex_ms if self._init_w > 0.0 else tau_syn_in_ms

        # Finiteness (NEST-style) before the degenerate / sign checks.
        for v, name in ((self._init_w, 'weight'), (self.eta, 'eta'),
                        (self._tau_Delta_ms, 'tau_Delta'), (self.Wmin, 'Wmin'),
                        (self.Wmax, 'Wmax'), (C_m_pF, 'dend_C_m'), (g_L_nS, 'dend_g_L'),
                        (self._tau_L_ms, 'tau_L'), (self._tau_s_ms, 'tau_s')):
            if not np.isfinite(v):
                raise ValueError(f"'{name}' must be finite.")
        if g_L_nS <= 0.0 or C_m_pF <= 0.0:
            raise ValueError("'dend_C_m' and 'dend_g_L' must be > 0.")
        if self._tau_Delta_ms <= 0.0:
            raise ValueError("'tau_Delta' must be > 0.")
        if self._tau_s_ms <= 0.0:
            raise ValueError("dendritic 'tau_syn' must be > 0.")
        # Degenerate prefactor 1/(tau_L - tau_s).
        if abs(self._tau_L_ms - self._tau_s_ms) < 1e-12:
            raise ValueError(
                'Degenerate Urbanczik prefactor: tau_L (= dend_C_m/dend_g_L) must '
                f'differ from tau_s; got tau_L = tau_s = {self._tau_L_ms} ms.')
        # NEST asymmetric sign tests (lower bound first, then upper bound).
        if _sign_like_wmin(self._init_w) != _sign_like_wmin(self.Wmin):
            raise ValueError('Weight and Wmin must have same sign.')
        if _sign_like_wmax(self._init_w) != _sign_like_wmax(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')

        self._pref = (15.0 * C_m_pF * self._tau_s_ms * self.eta
                      / (g_L_nS * (self._tau_L_ms - self._tau_s_ms)))

        # Two-column per-pre trace seam: column 0 = tau_L trace, column 1 = tau_s.
        self.pre_trace_tau = (self._tau_L_ms * u.ms, self._tau_s_ms * u.ms)

    def edge_state_init(self) -> dict:
        # Running plasticity integral and its tau_Delta low-pass, per edge.
        return {'PI_integral': 0.0, 'PI_exp_integral': 0.0}

    # -- rule kernel -------------------------------------------------------
    def update(self, state, ctx):
        dpi = ctx.post_states['delta_Pi']
        pre_L = ctx.pre_traces[:, 0]                          # tau_L trace
        pre_s = ctx.pre_traces[:, 1]                          # tau_s trace
        PI = (pre_L - pre_s) * dpi
        PI_int = state['PI_integral'] + PI
        PI_exp = state['PI_exp_integral'] * jnp.exp(-ctx.dt / self._tau_Delta_ms) + PI
        # NEST resets the weight from its initial value each send (not incrementally).
        w = jnp.clip(self._init_w + (PI_int - PI_exp) * self._pref, self.Wmin, self.Wmax)
        return {'weight': w, 'PI_integral': PI_int, 'PI_exp_integral': PI_exp}, w
