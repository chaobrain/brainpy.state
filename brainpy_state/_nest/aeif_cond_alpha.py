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

from ._base import NESTNeuron
from ._utils import is_tracer, validate_aeif_overflow

__all__ = [
    'aeif_cond_alpha',
]


class aeif_cond_alpha(NESTNeuron):
    r"""NEST-compatible ``aeif_cond_alpha`` neuron model.

    Conductance-based adaptive exponential integrate-and-fire neuron with
    alpha-shaped synaptic conductances.

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
         - Spike detection threshold when ``Delta_T > 0`` and RHS clamp limit \
           via :math:`\min(V, V_{\mathrm{peak}})`.
       * - ``V_reset``
         - ArrayLike, broadcastable (mV)
         - ``-60.0 * u.mV``
         - :math:`V_\mathrm{reset}`
         - Membrane reset value and refractory clamp voltage.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms)
         - ``0.0 * u.ms``
         - :math:`t_\mathrm{ref}`
         - Absolute refractory duration converted to integer step counts using \
           ``ceil(t_ref / dt)``.
       * - ``g_L`` and ``C_m``
         - ArrayLike, broadcastable (nS, pF)
         - ``30.0 * u.nS``, ``281.0 * u.pF``
         - :math:`g_L`, :math:`C_m`
         - Leak conductance and membrane capacitance in the AdEx membrane ODE.
       * - ``E_ex``, ``E_in``, and ``E_L``
         - ArrayLike, broadcastable (mV)
         - ``0.0 * u.mV``, ``-85.0 * u.mV``, ``-70.6 * u.mV``
         - :math:`E_\mathrm{ex}`, :math:`E_\mathrm{in}`, :math:`E_L`
         - Excitatory, inhibitory, and leak reversal potentials.
       * - ``Delta_T`` and ``V_th``
         - ArrayLike, broadcastable (mV)
         - ``2.0 * u.mV``, ``-50.4 * u.mV``
         - :math:`\Delta_T`, :math:`V_\mathrm{th}`
         - Exponential spike-initiation slope and soft-threshold location.
       * - ``tau_w``, ``a``, and ``b``
         - ArrayLike, broadcastable (ms, nS, pA)
         - ``144.0 * u.ms``, ``4.0 * u.nS``, ``80.5 * u.pA``
         - :math:`\tau_w`, :math:`a`, :math:`b`
         - Adaptation time constant, subthreshold coupling, and spike-triggered \
           jump amplitude.
       * - ``tau_syn_ex`` and ``tau_syn_in``
         - ArrayLike, broadcastable (ms)
         - ``0.2 * u.ms``, ``2.0 * u.ms``
         - :math:`\tau_{\mathrm{syn,ex}}`, :math:`\tau_{\mathrm{syn,in}}`
         - Alpha conductance time constants for excitatory/inhibitory channels.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0.0 * u.pA``
         - :math:`I_e`
         - Constant injected current added every substep.
       * - ``gsl_error_tol``
         - ArrayLike, broadcastable, unitless, ``> 0``
         - ``1e-6``
         - --
         - Local absolute tolerance for the embedded RKF45 error estimate.
       * - ``V_initializer``
         - Callable
         - ``Constant(-70.6 * u.mV)``
         - --
         - Initializer for membrane potential state ``V``.
       * - ``g_ex_initializer`` and ``g_in_initializer``
         - Callable
         - ``Constant(0.0 * u.nS)``
         - --
         - Initializers for ``g_ex`` and ``g_in``; ``dg_ex`` and ``dg_in`` \
           always start at zero.
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
         - Reset policy inherited from :class:`~brainpy_state._base.Neuron`; \
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

    Returns
    -------
    out : Any
        Configured neuron node. Each :meth:`update` call returns a binary spike
        tensor (dtype ``float64``) with shape ``self.V.value.shape``.


    Description
    -----------

    ``aeif_cond_alpha`` follows NEST ``models/aeif_cond_alpha.{h,cpp}``.
    The model combines:

    - exponential spike-initiation current (AdEx),
    - spike-triggered and subthreshold adaptation current ``w``,
    - alpha-shaped excitatory/inhibitory conductances.

    **1. Membrane, synapse, and adaptation dynamics**

    Let :math:`V` be membrane voltage and :math:`w` adaptation current.

    .. math::

       C_m \frac{dV}{dt}
       =
       -g_L (V - E_L)
       + g_L \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)
       - g_{ex}(V - E_{ex})
       - g_{in}(V - E_{in})
       - w + I_e + I_{stim}.

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

    **2. Refractory and spike handling (NEST semantics)**

    During refractory integration, NEST clamps effective membrane voltage to
    ``V_reset`` and sets :math:`dV/dt=0`. Otherwise the RHS uses
    :math:`\min(V, V_{peak})` as effective voltage.

    Threshold detection uses:

    - ``V_peak`` if ``Delta_T > 0``,
    - ``V_th`` if ``Delta_T == 0`` (iaf-like limit).

    On each detected spike:

    - ``V`` is reset to ``V_reset``,
    - adaptation jump ``w <- w + b`` is applied immediately,
    - refractory counter is set to ``refractory_counts + 1`` if refractory is enabled.

    Spike handling occurs *inside* the adaptive RKF45 substep loop. Therefore,
    with ``t_ref = 0`` multiple spikes can occur inside one simulation step,
    matching NEST behavior.

    **3. Update order per simulation step**

    1. Integrate ODEs on :math:`(t, t+dt]` via adaptive RKF45.
    2. Inside integration loop: apply refractory clamp and spike/reset/adaptation.
    3. After loop: decrement refractory counter once.
    4. Apply arriving spike weights to ``dg_ex``/``dg_in``.
    5. Store external current input ``x`` into one-step delayed ``I_stim``.

    Raises
    ------
    ValueError
        If parameters violate NEST-compatible constraints:
        ``V_reset < V_peak``, ``V_peak >= V_th``, ``Delta_T >= 0``,
        ``C_m > 0``, ``t_ref >= 0``, all time constants strictly positive,
        and ``gsl_error_tol > 0``. Also raised when
        ``(V_peak - V_th) / Delta_T`` can overflow the exponential term, or if
        runtime states exceed stability guards in :meth:`update`.
    TypeError
        If incompatible unitful/unitless values are passed and arithmetic
        fails during parameter broadcasting or updates.

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
    refractory_step_count : ShortTermState
        Remaining refractory grid steps (``int32``).
    integration_step : ShortTermState
        Persistent RKF45 substep size estimate (ms).
    I_stim : ShortTermState
        One-step delayed injected current buffer (pA).
    last_spike_time : ShortTermState
        Last emitted spike time (ms); written as ``t + dt`` on spike.
    refractory : ShortTermState
        Optional boolean refractory indicator, available only when
        ``ref_var=True``.

    See Also
    --------
    aeif_cond_exp : AdEx conductance model with exponential (single-state)
        synaptic kernels.
    aeif_cond_alpha_multisynapse : AdEx alpha-conductance model with
        receptor-indexed ports.
    aeif_psc_alpha : Current-based AdEx model with alpha PSCs.

    Notes
    -----

    The two-state alpha formulation is equivalent to a causal alpha kernel.
    With an event of effective conductance weight :math:`w` applied at
    :math:`t=0` through ``dg += e w / \tau``, the resulting conductance is:

    .. math::

       g(t) = w \cdot \frac{t}{\tau} \exp\!\left(1-\frac{t}{\tau}\right),\quad t \ge 0.

    Hence the kernel peaks at :math:`t=\tau` with amplitude exactly :math:`w`,
    matching NEST's interpretation of weight magnitudes in nS.

    Additional implementation implications:

    - ``t_ref=0`` (default) allows multiple in-loop spikes within one grid step.
    - Current input ``x`` is delayed by one step via ``I_stim`` (ring-buffer
      semantics), while spike events are applied after ODE integration.
    - Runtime is dominated by per-neuron adaptive RKF45 loops and therefore
      scales with both population size and accepted substeps.
    - Spike output is binary per simulation step even though multiple internal
      spike/reset events can occur during a single ``dt`` integration window.

    References
    ----------
    .. [1] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
           model as an effective description of neuronal activity.
           Journal of Neurophysiology, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [2] NEST source: ``models/aeif_cond_alpha.h`` and
           ``models/aeif_cond_alpha.cpp``.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> neuron = brainpy.state.aeif_cond_alpha(
       ...     in_size=3,
       ...     V_peak=0.0 * u.mV,
       ...     t_ref=2.0 * u.ms,
       ... )
       >>> neuron.init_state()
       >>> with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
       ...     spikes = neuron.update(x=120.0 * u.pA)
       >>> spikes.shape
       (3,)
    """

    __module__ = 'brainpy.state'

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

        if np.any(v_reset >= v_peak):
            raise ValueError('Ensure that: V_reset < V_peak .')
        if np.any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if np.any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if np.any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self.tau_syn_ex <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self.tau_w <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self.gsl_error_tol <= 0.0):
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
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()

        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape)
        V = braintools.init.param(self.V_initializer, self.varshape)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
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
        self.I_stim = brainstate.ShortTermState.init(braintools.init.Constant(0.0 * u.pA), self.varshape)

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

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        ditype = brainstate.environ.ditype()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

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

    def _dynamics_vector(self, V, dg_ex, g_ex, dg_in, g_in, w, is_refractory, i_stim):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        V : Quantity, mV
        dg_ex : Quantity, nS/ms
        g_ex : Quantity, nS
        dg_in : Quantity, nS/ms
        g_in : Quantity, nS
        w : Quantity, pA
        is_refractory : array, bool
        i_stim : Quantity, pA

        Returns
        -------
        tuple of 6 Quantities (dV, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt, dw)
        """
        v_eff = u.math.where(is_refractory, self.V_reset, u.math.minimum(V, self.V_peak))

        i_syn_exc = g_ex * (v_eff - self.E_ex)
        i_syn_inh = g_in * (v_eff - self.E_in)

        delta_t_safe = u.math.where(self.Delta_T == 0.0 * u.mV, 1.0 * u.mV, self.Delta_T)
        exp_arg = u.math.clip((v_eff - self.V_th) / delta_t_safe, -500.0, 500.0)
        i_spike = self.g_L * self.Delta_T * u.math.exp(exp_arg)

        dV_raw = (
                     -self.g_L * (v_eff - self.E_L) + i_spike
                     - i_syn_exc - i_syn_inh - w + self.I_e + i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        ddg_ex = -dg_ex / self.tau_syn_ex
        dg_ex_dt = dg_ex - g_ex / self.tau_syn_ex
        ddg_in = -dg_in / self.tau_syn_in
        dg_in_dt = dg_in - g_in / self.tau_syn_in
        dw = (self.a * (v_eff - self.E_L) - w) / self.tau_w

        return dV, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt, dw

    def update(self, x=0.0 * u.pA):
        r"""Advance the neuron by one simulation step.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous external current input in pA, broadcastable to
            ``self.varshape``. This value is stored into ``I_stim`` and applied
            at the next simulation step (one-step delay).

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
            (``V < -1e3 mV`` or ``|w| > 1e6 pA``), indicating divergent
            dynamics for the current parameter/input regime.

        Notes
        -----
        Integration is performed with an adaptive vectorized RKF45 loop,
        including in-loop spike/reset/adaptation events and optional
        multiple spikes per step. All arithmetic is unit-aware via
        ``saiunit.math``.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        dg_rate_unit = u.nS / u.ms
        V = self.V.value  # mV
        dg_ex = self.dg_ex.value * dg_rate_unit  # dimensionless -> nS/ms
        g_ex = self.g_ex.value  # nS
        dg_in = self.dg_in.value * dg_rate_unit  # dimensionless -> nS/ms
        g_in = self.g_in.value  # nS
        w = self.w.value  # pA
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        v_shape = u.get_mantissa(V).shape

        # Spike detection threshold: V_peak if Delta_T > 0, else V_th.
        v_peak_detect = u.math.where(self.Delta_T > 0.0 * u.mV, self.V_peak, self.V_th)
        refr_counts = self._refractory_counts()

        # Synaptic spike inputs (applied after integration).
        w_ex, w_in = self._sum_signed_delta_inputs()  # nS, nS
        pscon_ex = np.e / self.tau_syn_ex  # 1/ms
        pscon_in = np.e / self.tau_syn_in  # 1/ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via jax.lax.while_loop.
        MIN_H = self._MIN_H  # 1e-8 ms
        t_local = jnp.zeros(v_shape, dtype=dftype) * u.ms
        h = u.math.maximum(h, MIN_H)
        spike_mask = jnp.zeros(v_shape, dtype=jnp.bool_)
        atol = self.gsl_error_tol

        init_carry = (
            (V, dg_ex, g_ex, dg_in, g_in, w),  # state tuple
            t_local, h, spike_mask, r,
            jnp.array(0, dtype=jnp.int32),  # n_iters
            jnp.array(False),  # unstable flag
        )

        def _cond_fn(carry):
            _, t_loc, _, _, _, n_iters, unstable = carry
            return (jnp.any(u.get_mantissa(t_loc) < u.get_mantissa(dt))
                    & (n_iters < self._MAX_ITERS) & ~unstable)

        def _body_fn(carry):
            state, t_loc, h, spk_mask, r, n_iters, unstable = carry

            active = u.get_mantissa(t_loc) < u.get_mantissa(dt)

            # Clamp step size to remaining integration time.
            h = u.math.where(
                active,
                u.math.maximum(MIN_H, u.math.minimum(h, dt - t_loc)),
                h,
            )
            is_refractory = r > 0

            # RKF45 stages (coefficient ordering matches NEST reference).
            k1 = list(self._dynamics_vector(*state, is_refractory, i_stim))

            k2 = list(self._dynamics_vector(
                *[s + h * (1.0 / 4.0 * ki)
                  for s, ki in zip(state, k1)],
                is_refractory, i_stim,
            ))

            k3 = list(self._dynamics_vector(
                *[s + h * (3.0 * k1i / 32.0 + 9.0 * k2i / 32.0)
                  for s, k1i, k2i in zip(state, k1, k2)],
                is_refractory, i_stim,
            ))

            k4 = list(self._dynamics_vector(
                *[s + h * (1932.0 * k1i / 2197.0 - 7200.0 * k2i / 2197.0
                           + 7296.0 * k3i / 2197.0)
                  for s, k1i, k2i, k3i in zip(state, k1, k2, k3)],
                is_refractory, i_stim,
            ))

            k5 = list(self._dynamics_vector(
                *[s + h * (439.0 * k1i / 216.0 - 8.0 * k2i
                           + 3680.0 * k3i / 513.0
                           - 845.0 * k4i / 4104.0)
                  for s, k1i, k2i, k3i, k4i
                  in zip(state, k1, k2, k3, k4)],
                is_refractory, i_stim,
            ))

            k6 = list(self._dynamics_vector(
                *[s + h * (-8.0 * k1i / 27.0 + 2.0 * k2i
                           - 3544.0 * k3i / 2565.0
                           + 1859.0 * k4i / 4104.0
                           - 11.0 * k5i / 40.0)
                  for s, k1i, k2i, k3i, k4i, k5i
                  in zip(state, k1, k2, k3, k4, k5)],
                is_refractory, i_stim,
            ))

            # 4th and 5th order solutions.
            y4 = [s + h * (25.0 * k1i / 216.0 + 1408.0 * k3i / 2565.0
                           + 2197.0 * k4i / 4104.0 - k5i / 5.0)
                  for s, k1i, k3i, k4i, k5i
                  in zip(state, k1, k3, k4, k5)]
            y5 = [s + h * (16.0 * k1i / 135.0 + 6656.0 * k3i / 12825.0
                           + 28561.0 * k4i / 56430.0 - 9.0 * k5i / 50.0
                           + 2.0 * k6i / 55.0)
                  for s, k1i, k3i, k4i, k5i, k6i
                  in zip(state, k1, k3, k4, k5, k6)]

            # Error: max absolute difference across state dims (unitless).
            err_components = [u.get_mantissa(u.math.abs(y5i - y4i))
                              for y5i, y4i in zip(y5, y4)]
            err = err_components[0]
            for ec in err_components[1:]:
                err = jnp.maximum(err, ec)

            # Accept where error within tolerance or step at minimum.
            accept = active & (
                (err <= atol)
                | (u.get_mantissa(h) <= u.get_mantissa(MIN_H))
            )
            reject = active & ~accept

            # Update state for accepted neurons.
            new_state = tuple(
                u.math.where(accept, y5i, si)
                for y5i, si in zip(y5, state)
            )
            t_loc = u.math.where(accept, t_loc + h, t_loc)

            # Stability guard (deferred to post-loop check).
            V_m = u.get_mantissa(new_state[0])
            w_m = u.get_mantissa(new_state[5])
            unstable = unstable | jnp.any(
                accept & ((V_m < -1e3) | (w_m < -1e6) | (w_m > 1e6))
            )

            # Refractory voltage clamp.
            refr_accept = accept & (r > 0)
            new_V = u.math.where(
                refr_accept, self.V_reset, new_state[0],
            )

            # Spike detection: accepted, non-refractory, V >= threshold.
            spike_now = accept & (r <= 0) & (new_V >= v_peak_detect)
            spk_mask = spk_mask | spike_now
            new_V = u.math.where(spike_now, self.V_reset, new_V)
            new_w = u.math.where(
                spike_now, new_state[5] + self.b, new_state[5],
            )
            r = u.math.where(
                spike_now & (refr_counts > 0), refr_counts + 1, r,
            )

            # Adaptive step size.
            err_safe = jnp.maximum(err, 1e-30)
            fac_accept = jnp.where(
                err == 0.0, 5.0,
                jnp.minimum(5.0, jnp.maximum(
                    0.2, 0.9 * (atol / err_safe) ** 0.2)),
            )
            fac_reject = jnp.minimum(
                1.0, jnp.maximum(
                    0.2, 0.9 * (atol / err_safe) ** 0.25),
            )
            h = u.math.where(
                accept, u.math.maximum(MIN_H, h * fac_accept), h,
            )
            h = u.math.where(
                reject, u.math.maximum(MIN_H, h * fac_reject), h,
            )

            final_state = (
                new_V, new_state[1], new_state[2],
                new_state[3], new_state[4], new_w,
            )
            return (final_state, t_loc, h, spk_mask, r, n_iters + 1, unstable)

        carry_out = jax.lax.while_loop(_cond_fn, _body_fn, init_carry)
        state, _, h, spike_mask, r, _, unstable = carry_out
        V, dg_ex, g_ex, dg_in, g_in, w = state

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in aeif_cond_alpha dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Apply synaptic spike inputs.
        dg_ex = dg_ex + pscon_ex * w_ex  # nS/ms + 1/ms * nS = nS/ms
        dg_in = dg_in + pscon_in * w_in  # nS/ms + 1/ms * nS = nS/ms

        # Write back state.
        self.V.value = V
        self.dg_ex.value = u.get_mantissa(dg_ex / dg_rate_unit)
        self.g_ex.value = g_ex
        self.dg_in.value = u.get_mantissa(dg_in / dg_rate_unit)
        self.g_in.value = g_in
        self.w.value = w
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
