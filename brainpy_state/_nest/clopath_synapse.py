# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``clopath_synapse`` — voltage-based STDP spec + pure rule kernel.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on the
:class:`~brainpy_state._network._event_plastic.VoltageCoupledPlasticProj`
substrate (primitive #2). The substrate maintains the per-pre-neuron presynaptic
trace ``x_bar`` (``pre_trace_tau=tau_x``) and, via the **post-state reader**,
samples the post neuron's membrane and low-pass filtered voltages per edge each
step (``post_state_reads=('u_bar_minus','u_bar_plus','V')``). The kernel applies
**depression on the pre spike** (LTD from ``u_bar_minus``) and **potentiation
continuously while the post is depolarized** (LTP from ``V`` and ``u_bar_plus``),
the voltage-based rule of Clopath et al. (2010).

The previous imperative (NEST history-buffer) implementation lives in
:mod:`brainpy_state._nest._legacy_clopath_synapse`.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import saiunit as u
from brainstate.typing import ArrayLike

from ._plastic_base import (
    frozen, to_ms, to_scalar_float, unit_of,
    validate_delay, validate_receptor_type,
)

__all__ = ['clopath_synapse']


def _sign_like_wmin(x: float) -> int:
    # NEST set_status sign test for the *lower* bound: (x >= 0) - (x < 0).
    return int(x >= 0.0) - int(x < 0.0)


def _sign_like_wmax(x: float) -> int:
    # NEST set_status sign test for the *upper* bound: (x > 0) - (x <= 0).
    return int(x > 0.0) - int(x <= 0.0)


def _to_mv(value, *, name: str) -> float:
    """Return ``value`` in millivolts (bare numbers are interpreted as mV)."""
    if isinstance(value, u.Quantity):
        return float(value.to_decimal(u.mV))
    return to_scalar_float(value, name=name)


class clopath_synapse:
    r"""Voltage-based spike-timing-dependent plasticity synapse spec (NEST ``clopath_synapse``).

    Weight updates follow Clopath et al. (2010): a presynaptic spike depresses the
    weight by the post neuron's low-pass filtered voltage above ``theta_minus``,
    and sustained postsynaptic depolarization potentiates it. With ``x_bar`` the
    presynaptic trace and ``V`` / ``u_bar_plus`` / ``u_bar_minus`` the post neuron's
    (filtered) voltages read per edge each step:

    .. math::

       w \leftarrow \min\!\big(W_{\max},\; w + A_\text{LTP}\, \bar{x}\,
         [V - \theta_+]_+ \, [\bar{u}_+ - \theta_-]_+ \, \Delta t\big)
         \quad\text{(every step)}

       w \leftarrow \max\!\big(W_{\min},\; w - A_\text{LTD}\,
         [\bar{u}_- - \theta_-]_+\big) \quad\text{(pre spike)}

    where :math:`[x]_+ = \max(x, 0)`. **LTP carries a** :math:`\Delta t` **factor,
    LTD does not.** NEST's ``write_LTP_history`` multiplies its per-step ``dw`` by
    ``Time::get_resolution()`` (so the accumulated potentiation is a
    resolution-independent time integral), whereas the depression ``dw`` applied at
    each presynaptic spike has no such factor. The product of rectifiers reproduces
    NEST's ``u > theta_plus AND u_bar_plus > theta_minus`` AND-gate, and the strict
    NEST inequalities at equality give :math:`[0]_+ = 0` (no update) — see the
    boundary edge cases. ``x_bar`` is the substrate's per-pre trace divided by
    ``tau_x`` (NEST increments ``x_bar`` by :math:`1/\tau_x` per spike, the
    substrate by 1).

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge weight. Bare numbers default to **mV** — the reference neuron
        ``aeif_psc_delta_clopath`` is a delta model whose input is a voltage jump;
        pass an explicit pA ``Quantity`` for the current-based ``hh_psc_alpha_clopath``.
        Same sign as ``Wmax``/``Wmin`` under NEST's sign tests. Default ``1.0`` mV.
    delay : Quantity, optional
        Axonal/dendritic delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    tau_x : Quantity, optional
        Presynaptic trace (``x_bar``) time constant (> 0); drives the substrate's
        per-pre trace seam. Default ``15.0 ms``.
    Wmax, Wmin : float, optional
        Upper / lower weight clamp (pA). Defaults ``100.0`` / ``0.0``.
    A_LTP, A_LTD : float, optional
        Potentiation / depression amplitudes. Defaults ``8.0e-5`` / ``14.0e-5``.
    theta_plus, theta_minus : Quantity, optional
        Potentiation / depression voltage thresholds (mV). Defaults ``-45.3`` /
        ``-70.6`` mV.

    Notes
    -----
    **NEST divergence — parameter location.** In NEST, ``A_LTD``/``A_LTP``/
    ``theta_plus``/``theta_minus`` are parameters of the *postsynaptic neuron*
    (``ClopathArchivingNode``), which precomputes the weight change; the synapse
    stores only ``weight``/``x_bar``/``tau_x``/``Wmin``/``Wmax``. Here the spec+rule
    is self-contained, so these four amplitudes/thresholds are **synapse-spec**
    attributes that the kernel reads against the post neuron's filtered voltages
    (mirrors ``stdp_synapse`` moving ``tau_minus`` onto the synapse). For parity,
    the drive sets identical values on the NEST neuron and this spec; the
    filter time constants ``tau_u_bar_*`` remain post-neuron parameters.

    **Online vs deferred.** NEST defers potentiation (it accumulates an LTP history
    and applies it at the next pre ``send``, weighted by the decayed ``x_bar``);
    this kernel potentiates eagerly every step. The cumulative weight at each pre
    spike — where NEST's ``weight_recorder`` samples — coincides.

    **Delayed reads (``delay_u_bars``).** NEST's archiving node gates and evaluates
    LTP/LTD against the post voltages **delayed by** ``delay_u_bars`` (a ring buffer):
    LTP uses the instantaneous ``u`` with the delayed ``u_bar_plus``; LTD uses the
    delayed ``u_bar_minus``. This kernel reads the post neuron's *current* State with
    the substrate's intrinsic one-step lag (projections run before neurons), so live
    parity aligns NEST's ``delay_u_bars`` to one resolution step.

    References
    ----------
    .. [1] Clopath, Büsing, Vasilaki, Gerstner (2010). Connectivity reflects coding:
       a model of voltage-based STDP with homeostasis. *Nat. Neurosci.* 13(3):344-352.
    .. [2] NEST ``models/clopath_synapse.h`` + ``nestkernel/clopath_archiving_node``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> from brainpy_state import clopath_synapse
        >>> s = clopath_synapse(weight=0.5, tau_x=15.0 * u.ms)
        >>> s.is_homogeneous_weight, s.post_state_reads
        (False, ('u_bar_minus', 'u_bar_plus', 'V'))
        >>> s.post_trace_tau, s.edge_state_init()
        (None, {})
        >>> float(u.Quantity(s.pre_trace_tau).to_decimal(u.ms))
        15.0
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False
    post_trace_tau = None
    # primitive #2 reader: post-neuron State variables sampled per edge each step
    post_state_reads = ('u_bar_minus', 'u_bar_plus', 'V')

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_x: ArrayLike = 15.0 * u.ms,
        Wmax: ArrayLike = 100.0,
        Wmin: ArrayLike = 0.0,
        A_LTP: ArrayLike = 8.0e-5,
        A_LTD: ArrayLike = 14.0e-5,
        theta_plus: ArrayLike = -45.3 * u.mV,
        theta_minus: ArrayLike = -70.6 * u.mV,
    ):
        # Clopath's reference neuron aeif_psc_delta_clopath is a *delta* model
        # whose input seam is a voltage jump (mV), so a bare weight defaults to mV
        # (unlike the pA current synapses). An explicit Quantity is honored as-is,
        # e.g. pA for the current-based hh_psc_alpha_clopath.
        self.weight = weight if isinstance(weight, u.Quantity) else weight * u.mV
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self.tau_x = to_ms(tau_x, name='tau_x') * u.ms
        self._tau_x_ms = to_ms(tau_x, name='tau_x')
        self.Wmax = to_scalar_float(Wmax, name='Wmax')
        self.Wmin = to_scalar_float(Wmin, name='Wmin')
        self.A_LTP = to_scalar_float(A_LTP, name='A_LTP')
        self.A_LTD = to_scalar_float(A_LTD, name='A_LTD')
        self.theta_plus = theta_plus if isinstance(theta_plus, u.Quantity) else theta_plus * u.mV
        self.theta_minus = theta_minus if isinstance(theta_minus, u.Quantity) else theta_minus * u.mV
        self._theta_plus = _to_mv(theta_plus, name='theta_plus')
        self._theta_minus = _to_mv(theta_minus, name='theta_minus')

        if self._tau_x_ms <= 0.0:
            raise ValueError("'tau_x' must be > 0.")
        w0 = float(u.get_mantissa(self.weight))
        for v, name in ((w0, 'weight'), (self.Wmax, 'Wmax'), (self.Wmin, 'Wmin'),
                        (self.A_LTP, 'A_LTP'), (self.A_LTD, 'A_LTD'),
                        (self._theta_plus, 'theta_plus'), (self._theta_minus, 'theta_minus')):
            if not np.isfinite(v):
                raise ValueError(f"'{name}' must be finite.")
        # NEST asymmetric sign tests (lower bound first, then upper bound).
        if _sign_like_wmin(w0) != _sign_like_wmin(self.Wmin):
            raise ValueError('Weight and Wmin must have same sign.')
        if _sign_like_wmax(w0) != _sign_like_wmax(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')

        # per-side trace seam (substrate allocates + decays x_bar)
        self.pre_trace_tau = self.tau_x

    def edge_state_init(self) -> dict:
        # x_bar is the substrate's per-pre trace; no extra per-edge State needed.
        return {}

    # -- rule kernel -------------------------------------------------------
    def update(self, state, ctx):
        w = state['weight']
        ps = ctx.post_states
        V = ps['V']
        u_plus = ps['u_bar_plus']
        u_minus = ps['u_bar_minus']
        x_bar = ctx.pre_trace / self._tau_x_ms                # NEST 1/tau_x increment

        # LTP: time-integrated potentiation. NEST's write_LTP_history multiplies the
        # per-step dw by the resolution (Time::get_resolution().get_ms()), so the
        # accumulated LTP is a resolution-independent integral. Hence the * ctx.dt.
        ltp = (self.A_LTP * x_bar
               * jnp.maximum(V - self._theta_plus, 0.0)
               * jnp.maximum(u_plus - self._theta_minus, 0.0)
               * ctx.dt)
        w = jnp.minimum(w + ltp, self.Wmax)

        # LTD: on the pre spike
        ltd = self.A_LTD * jnp.maximum(u_minus - self._theta_minus, 0.0)
        w = frozen(ctx.pre_spike > 0, jnp.maximum(w - ltd, self.Wmin), w)
        return {'weight': w}, w
