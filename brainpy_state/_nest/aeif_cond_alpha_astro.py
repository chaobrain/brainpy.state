# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

from typing import Callable

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, validate_aeif_overflow, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'aeif_cond_alpha_astro',
]


class aeif_cond_alpha_astro(NESTNeuron):
    r"""NEST-compatible ``aeif_cond_alpha_astro`` neuron model.

    Short description
    -----------------

    Conductance-based adaptive exponential integrate-and-fire neuron with
    alpha-shaped synapses and support for astrocyte slow inward current (SIC).

    Description
    -----------

    This model follows NEST ``models/aeif_cond_alpha_astro.{h,cpp}`` and is a
    direct extension of :class:`aeif_cond_alpha` with an additional SIC current
    term in the membrane equation.

    **1. Continuous dynamics**

    Let :math:`V` be membrane voltage and :math:`w` adaptation current.

    .. math::

       C_m \frac{dV}{dt}
       =
       -g_L (V - E_L)
       + g_L \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)
       - g_{ex}(V - E_{ex})
       - g_{in}(V - E_{in})
       - w + I_e + I_{stim} + I_{SIC}.

    Adaptation dynamics:

    .. math::

       \tau_w \frac{dw}{dt} = a (V - E_L) - w.

    Alpha conductance states (two states per channel):

    .. math::

       \frac{d\,dg_{ex}}{dt} = -\frac{dg_{ex}}{\tau_{syn,ex}},
       \qquad
       \frac{d g_{ex}}{dt} = dg_{ex} - \frac{g_{ex}}{\tau_{syn,ex}},

    .. math::

       \frac{d\,dg_{in}}{dt} = -\frac{dg_{in}}{\tau_{syn,in}},
       \qquad
       \frac{d g_{in}}{dt} = dg_{in} - \frac{g_{in}}{\tau_{syn,in}}.

    Incoming spike weights are interpreted in nS and split by sign:

    .. math::

       dg_{ex} \leftarrow dg_{ex} + \frac{e}{\tau_{syn,ex}} w_+,
       \qquad
       dg_{in} \leftarrow dg_{in} + \frac{e}{\tau_{syn,in}} |w_-|.

    **2. Refractory and spike handling**

    During refractory integration, effective voltage is clamped to ``V_reset``
    and :math:`dV/dt = 0`. Otherwise the RHS uses :math:`\min(V, V_{peak})`
    as in NEST.

    Threshold detection uses:

    - ``V_peak`` if ``Delta_T > 0``
    - ``V_th`` if ``Delta_T == 0``

    On a detected spike (inside RKF45 substeps):

    - ``V <- V_reset``
    - ``w <- w + b``
    - refractory counter ``r <- refractory_counts + 1`` if ``t_ref > 0``

    **3. Update order per simulation step (NEST semantics)**

    1. Integrate ODEs on :math:`(t, t+dt]` with adaptive RKF45.
    2. Inside integration loop: refractory clamp and spike/reset/adaptation.
    3. Decrement refractory counter once.
    4. Apply arriving spike weights to ``dg_ex`` / ``dg_in``.
    5. Store new external current into one-step delayed ``I_stim``.
    6. Store SIC ring-buffer value for next step in ``I_SIC``.

    **4. SIC event semantics**

    ``sic_events`` passed to :meth:`update` emulate NEST ``SICEvent`` handling.
    For an event with ``weight``, coefficient series ``coeffs`` and
    ``delay_steps``:

    - effective ring-buffer offset is ``delay_steps - 1``,
    - each coefficient is added to future SIC buffer entries as
      ``weight * coeffs[i]``.

    This matches NEST ``handle(SICEvent&)`` where contributions are written to
    ``sic_currents_`` and become available through ``I_SIC`` with one-step
    delayed application in membrane dynamics.

    Parameters
    ----------
    in_size : Size
        Population shape. States are broadcast/initialized over
        ``self.varshape`` derived from ``in_size``.
    V_peak, V_reset, V_th, E_ex, E_in, E_L, Delta_T : ArrayLike
        Voltage-like parameters in mV, each broadcastable to ``self.varshape``.
    t_ref, tau_w, tau_syn_ex, tau_syn_in : ArrayLike
        Time constants in ms, broadcastable to ``self.varshape``.
    g_L, a : ArrayLike
        Conductances in nS, broadcastable to ``self.varshape``.
    C_m : ArrayLike
        Membrane capacitance in pF, broadcastable to ``self.varshape``.
    b, I_e : ArrayLike
        Currents in pA, broadcastable to ``self.varshape``.
    gsl_error_tol : ArrayLike
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
    V_initializer, g_ex_initializer, g_in_initializer, w_initializer : Callable
        Initializer callables used by :meth:`init_state` and :meth:`reset_state`.
    spk_fun : Callable
        Surrogate spike function used by :meth:`get_spike`.
    spk_reset : str
        Reset mode inherited from :class:`~brainpy_state._base.Neuron`.
    ref_var : bool
        If ``True``, allocate and expose ``self.refractory`` state.
    name : str | None
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 17 25 15 20 43

       * - Parameter
         - Type / shape / unit
         - Default
         - Math symbol
         - Semantics
       * - ``in_size``
         - :class:`~brainstate.typing.Size`; scalar or tuple
         - required
         - --
         - Population shape defining ``self.varshape``.
       * - ``V_peak``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``0.0 * u.mV``
         - :math:`V_\mathrm{peak}`
         - Spike detection threshold when ``Delta_T > 0`` and RHS clamp limit
           via :math:`\min(V, V_{peak})`.
       * - ``V_reset``
         - ArrayLike, broadcastable (mV)
         - ``-60.0 * u.mV``
         - :math:`V_\mathrm{reset}`
         - Membrane reset value and refractory clamp voltage.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms)
         - ``0.0 * u.ms``
         - :math:`t_\mathrm{ref}`
         - Absolute refractory duration converted to integer step counts using
           ``ceil(t_ref / dt)``.
       * - ``g_L``, ``C_m``
         - ArrayLike, broadcastable (nS, pF)
         - ``30.0 * u.nS``, ``281.0 * u.pF``
         - :math:`g_L`, :math:`C_m`
         - Leak conductance and membrane capacitance in the AdEx membrane ODE.
       * - ``E_ex``, ``E_in``, ``E_L``
         - ArrayLike, broadcastable (mV)
         - ``0.0 * u.mV``, ``-85.0 * u.mV``, ``-70.6 * u.mV``
         - :math:`E_\mathrm{ex}`, :math:`E_\mathrm{in}`, :math:`E_L`
         - Excitatory, inhibitory, and leak reversal potentials.
       * - ``Delta_T``, ``V_th``
         - ArrayLike, broadcastable (mV)
         - ``2.0 * u.mV``, ``-50.4 * u.mV``
         - :math:`\Delta_T`, :math:`V_\mathrm{th}`
         - Exponential spike-initiation slope and soft-threshold location.
       * - ``tau_w``, ``a``, ``b``
         - ArrayLike, broadcastable (ms, nS, pA)
         - ``144.0 * u.ms``, ``4.0 * u.nS``, ``80.5 * u.pA``
         - :math:`\tau_w`, :math:`a`, :math:`b`
         - Adaptation time constant, subthreshold coupling, and spike-triggered
           jump amplitude.
       * - ``tau_syn_ex``, ``tau_syn_in``
         - ArrayLike, broadcastable (ms)
         - ``0.2 * u.ms``, ``2.0 * u.ms``
         - :math:`\tau_{\mathrm{syn,ex}}`, :math:`\tau_{\mathrm{syn,in}}`
         - Alpha conductance time constants for excitatory/inhibitory channels.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0.0 * u.pA``
         - :math:`I_e`
         - Constant injected current added every RKF45-accepted substep.
       * - ``gsl_error_tol``
         - ArrayLike, broadcastable, unitless, ``> 0``
         - ``1e-6``
         - --
         - Local absolute tolerance for the embedded RKF45 error estimate.
       * - ``V_initializer``
         - Callable
         - ``Constant(-70.6 * u.mV)``
         - --
         - Initializer for membrane state ``V``.
       * - ``g_ex_initializer``, ``g_in_initializer``
         - Callable
         - ``Constant(0.0 * u.nS)``
         - --
         - Initializers for ``g_ex`` and ``g_in``; ``dg_ex`` and ``dg_in``
           are always reset to zero.
       * - ``w_initializer``
         - Callable
         - ``Constant(0.0 * u.pA)``
         - --
         - Initializer for adaptation current ``w``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate spike nonlinearity used by :meth:`get_spike`.
       * - ``spk_reset``
         - str
         - ``'hard'``
         - --
         - Reset policy inherited from :class:`~brainpy_state._base.Neuron`;
           hard reset matches NEST behavior.
       * - ``ref_var``
         - bool
         - ``False``
         - --
         - If ``True``, expose boolean state ``self.refractory``.
       * - ``name``
         - str | None
         - ``None``
         - --
         - Optional node name.

    Raises
    ------
    ValueError
        If parameters violate NEST-compatible constraints:
        ``V_reset < V_peak``, ``V_peak >= V_th``, ``Delta_T >= 0``,
        ``C_m > 0``, ``t_ref >= 0``, all time constants strictly positive,
        and ``gsl_error_tol > 0``. Also raised when the exponential threshold
        expression can overflow at spike time, for invalid SIC event tuples
        or non-positive SIC delays, and for runtime instability guards in
        :meth:`update`.
    TypeError
        If incompatible unitful/unitless values are passed and arithmetic
        fails during parameter broadcasting, SIC event coercion, or updates.

    Attributes
    ----------
    V : HiddenState
        Membrane potential :math:`V_m` (mV).
    dg_ex, dg_in : ShortTermState
        Alpha auxiliary states stored as numeric values representing
        :math:`\mathrm{nS}/\mathrm{ms}`.
    g_ex, g_in : HiddenState
        Excitatory and inhibitory conductances (nS).
    w : HiddenState
        Adaptation current (pA).
    I_stim : ShortTermState
        One-step delayed injected current buffer (pA).
    I_sic : ShortTermState
        One-step delayed SIC current (pA) loaded from the SIC queue.
    refractory_step_count : ShortTermState
        Remaining refractory grid steps (``int32``).
    integration_step : ShortTermState
        Persistent RKF45 substep size estimate (ms).
    last_spike_time : ShortTermState
        Last emitted spike time (ms); written as ``t + dt`` on spike.
    refractory : ShortTermState
        Optional boolean refractory indicator, available only when
        ``ref_var=True``.

    Recordables
    -----------
    Dynamic recordables follow NEST naming:

    - ``V_m``
    - ``g_ex``
    - ``g_in``
    - ``w``
    - ``I_SIC``

    See Also
    --------
    aeif_cond_alpha : AdEx alpha-conductance model without SIC support.
    aeif_cond_alpha_multisynapse : AdEx alpha-conductance model with multiple
        receptor ports.
    sic_connection : NEST-style SIC connection model for astrocyte-mediated
        currents.

    Notes
    -----
    The alpha-synapse subsystem is identical to :class:`aeif_cond_alpha`; for
    an event of effective conductance weight :math:`w` injected via
    ``dg += e w / \tau``, the resulting conductance kernel is:

    .. math::

       g(t) = w \cdot \frac{t}{\tau} \exp\!\left(1-\frac{t}{\tau}\right),\quad t \ge 0.

    SIC handling is discrete-time queue based. If an SIC event ``e`` arrives at
    simulation step :math:`s_e` with delay ``d_e`` (steps), weight ``w_e``,
    and coefficients :math:`c_{e,i}`, this implementation enqueues:

    .. math::

       Q[s_e + d_e - 1 + i] \mathrel{+}= w_e c_{e,i}.

    The membrane ODE at step :math:`k` uses stored :math:`I_{SIC}^{(k)}` from
    the previous update call, so a delay of 1 step first affects dynamics on
    step :math:`s_e + 1`, matching NEST ring-buffer semantics.

    Additional implementation implications:

    - ``sic_events`` accepts dict, tuple/list, scalar, or iterable forms;
      tuple/list forms must have length 2 or 3.
    - Event ``coeffs`` can be scalar, state-shaped, or leading-time-array;
      each coefficient creates one future queue entry.
    - Queue memory cost scales with active future SIC bins and state size.
    - As with ``aeif_cond_alpha``, ``t_ref=0`` can allow multiple in-loop
      spikes within one simulation step.

    References
    ----------
    .. [1] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
           model as an effective description of neuronal activity.
           Journal of Neurophysiology, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [2] NEST source: ``models/aeif_cond_alpha_astro.h`` and
           ``models/aeif_cond_alpha_astro.cpp``.
    .. [3] NEST source: ``models/sic_connection.h`` and
           ``models/sic_connection.cpp``.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> neuron = brainpy.state.aeif_cond_alpha_astro(in_size=2)
       >>> neuron.init_state()
       >>> sic = {'weight': 20.0 * u.pA, 'coefficients': [1.0, 0.5], 'delay_steps': 2}
       >>> with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
       ...     spikes = neuron.update(x=80.0 * u.pA, sic_events=[sic])
       >>> spikes.shape
       (2,)
    """

    __module__ = 'brainpy.state'

    RECORDABLES = (
        'V_m',
        'g_ex',
        'g_in',
        'w',
        'I_SIC',
    )

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        V_peak: ArrayLike = 0.0 * u.mV,
        V_reset: ArrayLike = -60.0 * u.mV,
        t_ref: ArrayLike = 0.0 * u.ms,
        g_L: ArrayLike = 30.0 * u.nS,
        C_m: ArrayLike = 281.0 * u.pF,
        E_ex: ArrayLike = 0.0 * u.mV,
        E_in: ArrayLike = -85.0 * u.mV,
        E_L: ArrayLike = -70.6 * u.mV,
        Delta_T: ArrayLike = 2.0 * u.mV,
        tau_w: ArrayLike = 144.0 * u.ms,
        a: ArrayLike = 4.0 * u.nS,
        b: ArrayLike = 80.5 * u.pA,
        V_th: ArrayLike = -50.4 * u.mV,
        tau_syn_ex: ArrayLike = 0.2 * u.ms,
        tau_syn_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0.0 * u.pA,
        gsl_error_tol: ArrayLike = 1e-6,
        V_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        g_ex_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        g_in_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        w_initializer: Callable = braintools.init.Constant(0.0 * u.pA),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.V_peak = braintools.init.param(V_peak, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.Delta_T = braintools.init.param(Delta_T, self.varshape)
        self.tau_w = braintools.init.param(tau_w, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
        self.w_initializer = w_initializer
        self.ref_var = ref_var

        self._validate_parameters()

        self._sic_queue = {}
        self._sic_step = 0

        self.integrator = AdaptiveRungeKuttaStep(
            method='RKF45',
            vf=self._vector_field,
            event_fn=self._event_fn,
            min_h=self._MIN_H,
            max_iters=self._MAX_ITERS,
            atol=self.gsl_error_tol,
            dt=brainstate.environ.get_dt()
        )

        # other variable
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    @property
    def recordables(self):
        return list(self.RECORDABLES)

    @staticmethod
    def _to_numpy(x, unit):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _to_numpy_unitless(x):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x), dtype=dftype)

    @staticmethod
    def _to_numpy_pA_or_unitless(x):
        try:
            dftype = brainstate.environ.dftype()
            return np.asarray(u.math.asarray(x / u.pA), dtype=dftype)
        except Exception:
            return np.asarray(u.math.asarray(x), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        r"""Validate model parameters against NEST constraints.

        Raises
        ------
        ValueError
            If parameter inequalities or positivity constraints are violated,
            or if the exponential term can overflow at spike time for the
            configured ``V_peak``, ``V_th``, and ``Delta_T``.
        """
        v_reset = self.V_reset
        v_peak = self.V_peak
        v_th = self.V_th
        delta_t = self.Delta_T / u.mV

        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (v_reset, v_peak, v_th, delta_t)):
            return

        if cond_any(v_reset >= v_peak):
            raise ValueError('Ensure that: V_reset < V_peak .')
        if cond_any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if cond_any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_w <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        # Mirror NEST overflow guard for exponential term at spike time.
        validate_aeif_overflow(v_peak, v_th, delta_t)

    def init_state(self, **kwargs):
        r"""Initialize persistent and short-term state variables.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Raises
        ------
        ValueError
            If an initializer cannot be broadcast to requested shape.
        TypeError
            If initializer outputs have incompatible units/dtypes for the
            corresponding state variables.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape)
        V = braintools.init.param(self.V_initializer, self.varshape)
        zeros = u.math.zeros(self.varshape, dtype=V.dtype) * (u.nS / u.ms)
        w = braintools.init.param(self.w_initializer, self.varshape)

        self.dg_ex = brainstate.ShortTermState(zeros)
        self.dg_in = brainstate.ShortTermState(zeros)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)
        self.V = brainstate.HiddenState(V)
        self.w = brainstate.HiddenState(w)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))
        self.I_sic = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        self._sic_queue = {}
        self._sic_step = 0

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Evaluate surrogate spike output from membrane voltage.

        Parameters
        ----------
        V : ArrayLike, optional
            Voltage values with shape broadcastable to ``self.varshape`` and
            units compatible with mV. If ``None``, uses current state
            ``self.V.value``.

        Returns
        -------
        ArrayLike
            Surrogate spike activation produced by
            ``spk_fun((V - V_th) / (V_th - V_reset))``.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    @staticmethod
    def _coerce_sic_events(sic_events):
        if sic_events is None:
            return []
        if isinstance(sic_events, dict):
            return [sic_events]
        if isinstance(sic_events, tuple) and len(sic_events) in (2, 3):
            return [sic_events]
        if np.isscalar(sic_events):
            return [sic_events]
        return list(sic_events)

    def _queue_sic_value(self, step_index: int, value: np.ndarray):
        prev = self._sic_queue.get(step_index, None)
        if prev is None:
            self._sic_queue[step_index] = value.copy()
        else:
            self._sic_queue[step_index] = prev + value

    def _enqueue_sic_events(self, sic_events, state_shape):
        r"""Convert and enqueue SIC events into the future-step queue.

        Parameters
        ----------
        sic_events : object
            SIC event description or iterable of descriptions. Accepted forms:
            ``dict`` with keys ``weight``, ``coefficients``/``coeffs``/``coeff``/``values``,
            optional ``delay_steps``/``delay``, optional ``multiplicity``;
            tuple/list ``(weight, coeffs)`` or ``(weight, coeffs, delay_steps)``;
            scalar interpreted as coefficients with default ``weight=1`` and
            ``delay_steps=1``.
        state_shape : tuple[int, ...]
            Target neuron state shape used to broadcast event values.

        Raises
        ------
        ValueError
            If a tuple/list event has invalid length, ``delay_steps <= 0``, or
            SIC coefficients cannot be broadcast to ``state_shape``.
        TypeError
            If event payload cannot be converted to numeric arrays.
        """
        events = self._coerce_sic_events(sic_events)
        if len(events) == 0:
            return

        for ev in events:
            weight = 1.0
            coeffs = 0.0
            delay_steps = 1

            if isinstance(ev, dict):
                weight = ev.get('weight', 1.0)
                coeffs = ev.get('coefficients', ev.get('coeffs', ev.get('coeff', ev.get('values', 0.0))))
                delay_steps = ev.get('delay_steps', ev.get('delay', 1))
                multiplicity = ev.get('multiplicity', 1.0)
                weight = weight * multiplicity
            elif isinstance(ev, tuple) or isinstance(ev, list):
                if len(ev) == 2:
                    weight, coeffs = ev
                elif len(ev) == 3:
                    weight, coeffs, delay_steps = ev
                else:
                    raise ValueError('SIC event tuples must have length 2 or 3.')
            else:
                coeffs = ev

            delay_steps = int(delay_steps)
            if delay_steps <= 0:
                raise ValueError('SIC event delay_steps must be a positive integer.')

            weight_np = self._broadcast_to_state(self._to_numpy_pA_or_unitless(weight), state_shape)
            coeffs_np = self._to_numpy_pA_or_unitless(coeffs)

            if coeffs_np.ndim == 0 or coeffs_np.shape == state_shape:
                coeff_iter = [self._broadcast_to_state(coeffs_np, state_shape)]
            else:
                coeff_iter = [self._broadcast_to_state(coeffs_np[i], state_shape) for i in range(coeffs_np.shape[0])]

            base_offset = delay_steps - 1
            for i, coeff_i in enumerate(coeff_iter):
                self._queue_sic_value(self._sic_step + base_offset + i, weight_np * coeff_i)

    def _pop_sic_current(self, state_shape):
        dftype = brainstate.environ.dftype()
        current = self._sic_queue.pop(self._sic_step, None)
        if current is None:
            return np.zeros(state_shape, dtype=dftype)
        return np.asarray(current, dtype=dftype)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, dg_ex, g_ex, dg_in, g_in, w -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, i_sic, v_peak_detect --
            mutable auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, u.math.minimum(state.V, self.V_peak))

        i_syn_exc = state.g_ex * (v_eff - self.E_ex)
        i_syn_inh = state.g_in * (v_eff - self.E_in)

        delta_t_safe = u.math.where(self.Delta_T == 0.0 * u.mV, 1.0 * u.mV, self.Delta_T)
        exp_arg = u.math.clip((v_eff - self.V_th) / delta_t_safe, -500.0, 500.0)
        i_spike = self.g_L * self.Delta_T * u.math.exp(exp_arg)

        dV_raw = (
                     -self.g_L * (v_eff - self.E_L) + i_spike
                     - i_syn_exc - i_syn_inh - state.w + self.I_e + extra.i_stim + extra.i_sic
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        ddg_ex = -state.dg_ex / self.tau_syn_ex
        dg_ex_dt = state.dg_ex - state.g_ex / self.tau_syn_ex
        ddg_in = -state.dg_in / self.tau_syn_in
        dg_in_dt = state.dg_in - state.g_in / self.tau_syn_in
        dw = (self.a * (v_eff - self.E_L) - state.w) / self.tau_w

        return DotDict(V=dV, dg_ex=ddg_ex, g_ex=dg_ex_dt, dg_in=ddg_in, g_in=dg_in_dt, w=dw)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, dg_ex, g_ex, dg_in, g_in, w -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, i_sic, v_peak_detect.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & ((state.V < -1e3 * u.mV) | (state.w < -1e6 * u.pA) | (state.w > 1e6 * u.pA))
        )

        refr_accept = accept & (extra.r > 0)
        new_V = u.math.where(refr_accept, self.V_reset, state.V)

        spike_now = accept & (extra.r <= 0) & (new_V >= extra.v_peak_detect)
        spike_mask = extra.spike_mask | spike_now
        new_V = u.math.where(spike_now, self.V_reset, new_V)
        new_w = u.math.where(spike_now, state.w + self.b, state.w)
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict({**state, 'V': new_V, 'w': new_w})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

    def update(self, x=0.0 * u.pA, sic_events=None):
        r"""Advance the neuron by one simulation step with optional SIC events.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous external current input in pA, broadcastable to
            ``self.varshape``. This value is stored into ``I_stim`` and applied
            at the next simulation step (one-step delay).
        sic_events : object, optional
            SIC event payload consumed by :meth:`_enqueue_sic_events`.
            Enqueued values are popped for the current queue step and stored
            into ``I_sic`` for the next update call.

        Returns
        -------
        jax.Array
            Binary spike tensor with dtype ``jnp.float64`` and shape
            ``self.V.value.shape``. A value of ``1.0`` indicates at least one
            internal spike event occurred during the integrated interval
            :math:`(t, t+dt]`.

        Raises
        ------
        ValueError
            If RKF45 integration enters a guarded unstable regime
            (``V < -1e3 mV`` or ``|w| > 1e6 pA``), if SIC tuple/list events have
            unsupported lengths, or if SIC ``delay_steps`` is not positive.
        TypeError
            If SIC event values cannot be interpreted as numeric arrays.

        Notes
        -----
        Integration is performed with an adaptive vectorized RKF45 loop,
        including in-loop spike/reset/adaptation events and optional
        multiple spikes per step. All arithmetic is unit-aware via
        ``saiunit.math``. SIC events are enqueued after ODE integration,
        then current-step SIC queue values are popped and written to
        ``I_sic`` for use in the next call.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        dg_ex = self.dg_ex.value  # nS/ms
        g_ex = self.g_ex.value  # nS
        dg_in = self.dg_in.value  # nS/ms
        g_in = self.g_in.value  # nS
        w = self.w.value  # pA
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        i_sic = self.I_sic.value  # pA
        h = self.integration_step.value  # ms

        # Spike detection threshold: V_peak if Delta_T > 0, else V_th.
        v_peak_detect = u.math.where(self.Delta_T > 0.0 * u.mV, self.V_peak, self.V_th)

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, dg_ex=dg_ex, g_ex=g_ex, dg_in=dg_in, g_in=g_in, w=w)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            i_sic=i_sic,
            v_peak_detect=v_peak_detect,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, dg_ex, g_ex = ode_state.V, ode_state.dg_ex, ode_state.g_ex
        dg_in, g_in, w = ode_state.dg_in, ode_state.g_in, ode_state.w
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in aeif_cond_alpha_astro dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration).
        w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
        w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')
        pscon_ex = np.e / self.tau_syn_ex  # 1/ms
        pscon_in = np.e / self.tau_syn_in  # 1/ms

        # Apply synaptic spike inputs.
        dg_ex = dg_ex + pscon_ex * w_ex  # nS/ms + 1/ms * nS = nS/ms
        dg_in = dg_in + pscon_in * w_in  # nS/ms + 1/ms * nS = nS/ms

        # SIC event handling: enqueue new events and pop current step value.
        v_shape = self.V.value.shape
        self._enqueue_sic_events(sic_events, v_shape)
        new_i_sic = self._pop_sic_current(v_shape)

        # Write back state.
        self.V.value = V
        self.dg_ex.value = dg_ex
        self.g_ex.value = g_ex
        self.dg_in.value = dg_in
        self.g_in.value = g_in
        self.w.value = w
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        self.I_sic.value = new_i_sic * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        self._sic_step += 1

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
