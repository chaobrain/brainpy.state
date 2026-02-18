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

import math
from typing import Callable

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'aeif_cond_alpha_astro',
]


class aeif_cond_alpha_astro(Neuron):
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
       >>> import brainunit as u
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

    _MIN_H = 1e-8  # ms
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
        self.gsl_error_tol = braintools.init.param(gsl_error_tol, self.varshape)

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
        self.w_initializer = w_initializer
        self.ref_var = ref_var

        self._validate_parameters()

        self._sic_queue = {}
        self._sic_step = 0

    @property
    def recordables(self):
        return list(self.RECORDABLES)

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _to_numpy_unitless(x):
        return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _to_numpy_pA_or_unitless(x):
        try:
            return np.asarray(u.math.asarray(x / u.pA), dtype=np.float64)
        except Exception:
            return np.asarray(u.math.asarray(x), dtype=np.float64)

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
        v_reset = self._to_numpy(self.V_reset, u.mV)
        v_peak = self._to_numpy(self.V_peak, u.mV)
        v_th = self._to_numpy(self.V_th, u.mV)
        delta_t = self._to_numpy(self.Delta_T, u.mV)

        if np.any(v_reset >= v_peak):
            raise ValueError('Ensure that: V_reset < V_peak .')
        if np.any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if np.any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_w, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy_unitless(self.gsl_error_tol) <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        positive_dt = delta_t > 0.0
        if np.any(positive_dt):
            max_exp_arg = np.log(np.finfo(np.float64).max / 1e20)
            ratio = (v_peak - v_th) / np.where(positive_dt, delta_t, 1.0)
            if np.any(ratio[positive_dt] >= max_exp_arg):
                raise ValueError(
                    'The current combination of V_peak, V_th and Delta_T will lead to numerical overflow at spike '
                    'time; try for instance to increase Delta_T or to reduce V_peak to avoid this problem.'
                )

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize persistent and short-term state variables.

        Parameters
        ----------
        batch_size : int, optional
            If provided, state tensors are created with leading batch axis
            ``(batch_size, *self.varshape)``. Otherwise shapes are
            ``self.varshape``.
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
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        w = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))

        self.V = brainstate.HiddenState(V)
        self.dg_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.g_ex = brainstate.HiddenState(g_ex)
        self.dg_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.g_in = brainstate.HiddenState(g_in)
        self.w = brainstate.HiddenState(w)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = self._safe_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )
        self.I_sic = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        self._sic_queue = {}
        self._sic_step = 0

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset model state and SIC queue to initial conditions.

        Parameters
        ----------
        batch_size : int, optional
            Optional batch size used to rebuild values with shape
            ``(batch_size, *self.varshape)``. If omitted, keeps non-batched
            ``self.varshape`` layout.
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Raises
        ------
        ValueError
            If initializer outputs cannot be broadcast to state shape.
        TypeError
            If unit arithmetic fails while constructing reset values.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.g_ex.value = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        self.g_in.value = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(self.V.value / u.mV))
        self.dg_ex.value = np.asarray(zeros, dtype=np.float64)
        self.dg_in.value = np.asarray(zeros, dtype=np.float64)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        dt = self._safe_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )
        self.I_sic.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

        self._sic_queue = {}
        self._sic_step = 0

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

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

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _sum_signed_delta_inputs(self):
        w_ex = u.math.zeros_like(self.g_ex.value)
        w_in = u.math.zeros_like(self.g_in.value)
        if self.delta_inputs is None:
            return w_ex, w_in

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            zero = u.math.zeros_like(out)
            w_ex = w_ex + u.math.maximum(out, zero)
            w_in = w_in + u.math.maximum(-out, zero)
        return w_ex, w_in

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
        current = self._sic_queue.pop(self._sic_step, None)
        if current is None:
            return np.zeros(state_shape, dtype=np.float64)
        return np.asarray(current, dtype=np.float64)

    @staticmethod
    def _dynamics_scalar(v, dg_ex, g_ex, dg_in, g_in, w, is_refractory, i_stim, i_sic, p):
        v_eff = p['V_reset'] if is_refractory else min(v, p['V_peak_rhs'])

        i_syn_exc = g_ex * (v_eff - p['E_ex'])
        i_syn_inh = g_in * (v_eff - p['E_in'])
        i_spike = 0.0 if p['Delta_T'] == 0.0 else (
            p['g_L'] * p['Delta_T'] * math.exp((v_eff - p['V_th']) / p['Delta_T'])
        )

        dv = 0.0 if is_refractory else (
            -p['g_L'] * (v_eff - p['E_L']) + i_spike - i_syn_exc - i_syn_inh - w + p['I_e'] + i_stim + i_sic
        ) / p['C_m']

        ddg_ex = -dg_ex / p['tau_syn_ex']
        dg_ex_dt = dg_ex - (g_ex / p['tau_syn_ex'])
        ddg_in = -dg_in / p['tau_syn_in']
        dg_in_dt = dg_in - (g_in / p['tau_syn_in'])
        dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
        return dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt, dw

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
        Integration is performed with an adaptive scalar RKF45 loop per neuron
        index to preserve NEST update ordering. SIC events are enqueued after
        ODE integration, then current-step SIC queue values are popped and
        written to ``I_sic`` for use in the next call.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        dg_ex = self._broadcast_to_state(np.asarray(self.dg_ex.value, dtype=np.float64), v_shape)
        g_ex = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape)
        dg_in = self._broadcast_to_state(np.asarray(self.dg_in.value, dtype=np.float64), v_shape)
        g_in = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape)
        w = self._broadcast_to_state(self._to_numpy(self.w.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            v_shape,
        )
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        i_sic = self._broadcast_to_state(self._to_numpy(self.I_sic.value, u.pA), v_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape)

        p = {
            'V_peak_rhs': self._broadcast_to_state(self._to_numpy(self.V_peak, u.mV), v_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'E_ex': self._broadcast_to_state(self._to_numpy(self.E_ex, u.mV), v_shape),
            'E_in': self._broadcast_to_state(self._to_numpy(self.E_in, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'Delta_T': self._broadcast_to_state(self._to_numpy(self.Delta_T, u.mV), v_shape),
            'tau_w': self._broadcast_to_state(self._to_numpy(self.tau_w, u.ms), v_shape),
            'a': self._broadcast_to_state(self._to_numpy(self.a, u.nS), v_shape),
            'b': self._broadcast_to_state(self._to_numpy(self.b, u.pA), v_shape),
            'V_th': self._broadcast_to_state(self._to_numpy(self.V_th, u.mV), v_shape),
            'tau_syn_ex': self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape),
            'tau_syn_in': self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
            'atol': self._broadcast_to_state(self._to_numpy_unitless(self.gsl_error_tol), v_shape),
        }

        v_peak_detect = np.where(
            p['Delta_T'] > 0.0,
            p['V_peak_rhs'],
            p['V_th'],
        )
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )

        w_ex_q, w_in_q = self._sum_signed_delta_inputs()
        w_ex = self._broadcast_to_state(self._to_numpy(w_ex_q, u.nS), v_shape)
        w_in = self._broadcast_to_state(self._to_numpy(w_in_q, u.nS), v_shape)
        pscon_ex = self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        pscon_in = self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        spike_mask = np.zeros(v_shape, dtype=bool)
        V_next = np.empty_like(V)
        dg_ex_next = np.empty_like(dg_ex)
        g_ex_next = np.empty_like(g_ex)
        dg_in_next = np.empty_like(dg_in)
        g_in_next = np.empty_like(g_in)
        w_next = np.empty_like(w)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            y = np.asarray([V[idx], dg_ex[idx], g_ex[idx], dg_in[idx], g_in[idx], w[idx]], dtype=np.float64)
            r_i = int(r[idx])
            h_i = float(max(h_int[idx], self._MIN_H))
            t_local = 0.0
            iters = 0
            local_spike = False

            while t_local < dt and iters < self._MAX_ITERS:
                iters += 1
                h_i = max(self._MIN_H, min(h_i, dt - t_local))
                is_refractory = r_i > 0

                def f(y_):
                    return np.asarray(
                        self._dynamics_scalar(
                            y_[0], y_[1], y_[2], y_[3], y_[4], y_[5],
                            is_refractory, i_stim[idx], i_sic[idx], local_p
                        ),
                        dtype=np.float64,
                    )

                k1 = f(y)
                k2 = f(y + h_i * (1.0 / 4.0) * k1)
                k3 = f(y + h_i * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
                k4 = f(y + h_i * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0))
                k5 = f(y + h_i * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0))
                k6 = f(
                    y
                    + h_i
                    * (
                        -8.0 * k1 / 27.0
                        + 2.0 * k2
                        - 3544.0 * k3 / 2565.0
                        + 1859.0 * k4 / 4104.0
                        - 11.0 * k5 / 40.0
                    )
                )

                y4 = y + h_i * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
                y5 = y + h_i * (
                    16.0 * k1 / 135.0
                    + 6656.0 * k3 / 12825.0
                    + 28561.0 * k4 / 56430.0
                    - 9.0 * k5 / 50.0
                    + 2.0 * k6 / 55.0
                )
                err = float(np.max(np.abs(y5 - y4)))
                atol = float(local_p['atol'])

                if err <= atol or h_i <= self._MIN_H:
                    y = y5
                    t_local += h_i
                    fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
                    h_i = max(self._MIN_H, h_i * fac)

                    if y[0] < -1e3 or y[5] < -1e6 or y[5] > 1e6:
                        raise ValueError('Numerical instability in aeif_cond_alpha_astro dynamics.')

                    if r_i > 0:
                        y[0] = local_p['V_reset']
                    elif y[0] >= v_peak_detect[idx]:
                        local_spike = True
                        y[0] = local_p['V_reset']
                        y[5] += local_p['b']
                        r_i = int(refr_counts[idx]) + 1 if int(refr_counts[idx]) > 0 else 0
                else:
                    fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
                    h_i = max(self._MIN_H, h_i * fac)

            if r_i > 0:
                r_i -= 1

            y[1] += pscon_ex[idx] * w_ex[idx]
            y[3] += pscon_in[idx] * w_in[idx]

            spike_mask[idx] = local_spike
            V_next[idx] = y[0]
            dg_ex_next[idx] = y[1]
            g_ex_next[idx] = y[2]
            dg_in_next[idx] = y[3]
            g_in_next[idx] = y[4]
            w_next[idx] = y[5]
            r_next[idx] = r_i
            h_next[idx] = h_i

        self._enqueue_sic_events(sic_events, v_shape)
        new_i_sic = self._pop_sic_current(v_shape)

        self.V.value = V_next * u.mV
        self.dg_ex.value = dg_ex_next
        self.g_ex.value = g_ex_next * u.nS
        self.dg_in.value = dg_in_next
        self.g_in.value = g_in_next * u.nS
        self.w.value = w_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.I_sic.value = new_i_sic * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        self._sic_step += 1

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=jnp.float64)
