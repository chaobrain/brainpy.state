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
    'aeif_psc_delta',
]


class aeif_psc_delta(NESTNeuron):
    r"""NEST-compatible ``aeif_psc_delta`` neuron model.

    Current-based adaptive exponential integrate-and-fire neuron with
    delta-shaped synaptic input. Implements NEST ``models/aeif_psc_delta.{h,cpp}``
    semantics with adaptive RKF45 integration, in-loop spike handling, and optional
    refractory input buffering.

    **1. Mathematical Formulation**

    The model combines exponential spike-initiation current (AdEx), spike-triggered
    and subthreshold adaptation current :math:`w`, and delta-function synaptic input
    that directly jumps the membrane voltage.

    Membrane dynamics:

    .. math::

       C_m \frac{dV}{dt}
       =
       -g_L (V - E_L)
       + g_L \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)
       - w + I_e + I_{\mathrm{stim}}.

    Adaptation dynamics:

    .. math::

       \tau_w \frac{dw}{dt} = a (V - E_L) - w.

    Incoming delta spikes are interpreted as instantaneous voltage jumps:

    .. math::

       V \leftarrow V + J \sum_k \delta(t - t_k).

    Here :math:`J` is the synaptic weight in millivolts.

    **2. Refractory and Spike Handling**

    During refractory integration (when ``refractory_step_count > 0``), NEST clamps
    the effective membrane voltage to ``V_reset`` and sets :math:`dV/dt = 0`. Outside
    refractory, the RHS uses :math:`\min(V, V_{\mathrm{peak}})` as the effective voltage
    to prevent unbounded exponential growth.

    Threshold detection uses:

    - ``V_peak`` if ``Delta_T > 0`` (exponential regime),
    - ``V_th`` if ``Delta_T == 0`` (IAF-like limit).

    On each detected spike:

    1. ``V`` is reset to ``V_reset``,
    2. adaptation jump ``w <- w + b`` is applied immediately,
    3. refractory counter is set to ``ceil(t_ref / dt) + 1`` if ``t_ref > 0``.

    Spike handling occurs *inside* the adaptive RKF45 substep loop. With ``t_ref = 0``,
    multiple spikes can occur within one simulation step, matching NEST behavior.

    **3. Refractory Input Buffering**

    If ``refractory_input=True`` (NEST ``refractory_input`` flag), delta spikes arriving
    during refractory are accumulated into ``refractory_spike_buffer`` with NEST's
    exponential discount factor:

    .. math::

       \mathrm{buffer} \leftarrow \mathrm{buffer} + J \cdot \exp(-r \cdot \Delta t / \tau_m),

    where :math:`r` is the current refractory step count. The buffered input is applied
    when the neuron exits refractory.

    **4. Update Order per Simulation Step**

    1. Integrate ODEs on interval :math:`(t, t+\Delta t]` via adaptive RKF45.
    2. **Inside integration loop**:
        a. Apply arriving delta jump to ``V`` (if not refractory).
        b. Apply refractory clamp (``V <- V_reset`` if refractory).
        c. Apply refractory buffering logic if ``refractory_input=True``.
        d. Threshold detection and spike/reset/adaptation handling.
    3. **After loop**: Decrement refractory counter once.
    4. Apply arriving spike weights directly to ``V`` as delta-function pulses.
    5. Store external current input ``x`` into one-step delayed ``I_stim``.

    **5. Numerical Integration Details**

    The model uses adaptive Runge-Kutta-Fehlberg 4(5) (RKF45) with local error control.
    The step size ``integration_step`` is adjusted per neuron to satisfy the tolerance
    ``gsl_error_tol``, matching NEST's GSL solver behavior. Minimum step size is clamped
    to ``1e-8 ms`` to prevent infinite loops.

    The exponential term is computed using the effective voltage :math:`V_{\mathrm{eff}}`:

    .. math::

       V_{\mathrm{eff}} = \begin{cases}
           V_{\mathrm{reset}} & \text{if refractory}, \\
           \min(V, V_{\mathrm{peak}}) & \text{otherwise}.
       \end{cases}

    Overflow protection: the model validates that :math:`(V_{\mathrm{peak}} - V_{\mathrm{th}}) / \Delta_T`
    does not exceed ``log(DBL_MAX / 1e20)`` to prevent numerical overflow at spike time.

    **6. Differences from NEST**

    - **Spike output format**: NEST emits spike events; brainpy.state returns a binary
      array (0/1) per simulation step. Internal dynamics (adaptation increments, refractory
      handling) match NEST exactly.
    - **Surrogate gradients**: brainpy.state uses ``spk_fun`` (e.g., ``ReluGrad()``) for
      differentiable spike generation; NEST does not support gradient-based learning.
    - **Spike reset mode**: ``spk_reset='hard'`` (default) matches NEST; ``'soft'`` is
      available but non-canonical.

    Parameters
    ----------
    in_size : int, tuple of int
        Shape of the neuron population. Can be an integer (1D) or tuple (multi-dimensional).
    V_peak : ArrayLike, optional
        Spike detection threshold in millivolts. Used when ``Delta_T > 0``. Default: ``0.0 * u.mV``.
        Can be scalar or array matching ``in_size``.
    V_reset : ArrayLike, optional
        Reset potential in millivolts. Default: ``-60.0 * u.mV``. Must satisfy ``V_reset < V_peak``.
    t_ref : ArrayLike, optional
        Absolute refractory period in milliseconds. Default: ``0.0 * u.ms`` (no refractoriness).
        Must be non-negative.
    g_L : ArrayLike, optional
        Leak conductance in nanosiemens. Default: ``30.0 * u.nS``. Must be positive.
    C_m : ArrayLike, optional
        Membrane capacitance in picofarads. Default: ``281.0 * u.pF``. Must be positive.
    E_L : ArrayLike, optional
        Leak reversal potential in millivolts. Default: ``-70.6 * u.mV``.
    Delta_T : ArrayLike, optional
        Exponential slope factor in millivolts. Default: ``2.0 * u.mV``. Set to ``0.0`` for
        IAF-like limit. Must be non-negative.
    tau_w : ArrayLike, optional
        Adaptation time constant in milliseconds. Default: ``144.0 * u.ms``. Must be positive.
    a : ArrayLike, optional
        Subthreshold adaptation coupling in nanosiemens. Default: ``4.0 * u.nS``.
    b : ArrayLike, optional
        Spike-triggered adaptation increment in picoamperes. Default: ``80.5 * u.pA``.
    V_th : ArrayLike, optional
        Spike initiation threshold in millivolts (used in exponential term). Default: ``-50.4 * u.mV``.
        Must satisfy ``V_th <= V_peak``.
    I_e : ArrayLike, optional
        Constant external current in picoamperes. Default: ``0.0 * u.pA``.
    gsl_error_tol : ArrayLike, optional
        RKF45 local error tolerance (unitless). Default: ``1e-6``. Must be positive.
    refractory_input : bool, optional
        If True, accumulate delta spikes arriving during refractory with NEST's exponential
        discount factor and apply when refractory ends. Default: ``False``.
    V_initializer : Callable, optional
        Membrane potential initializer. Default: ``braintools.init.Constant(-70.6 * u.mV)``.
    w_initializer : Callable, optional
        Adaptation current initializer. Default: ``braintools.init.Constant(0.0 * u.pA)``.
    spk_fun : Callable, optional
        Surrogate gradient function for differentiable spike generation. Default: ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Spike reset mode. Options: ``'hard'`` (stop gradient), ``'soft'`` (V -= V_th). Default: ``'hard'``.
    ref_var : bool, optional
        If True, expose ``self.refractory`` as a boolean state variable. Default: ``False``.
    name : str, optional
        Name of the neuron group.

    Parameter Mapping
    -----------------

    ==================== ================== ========================================== =====================================================
    **Parameter**        **Default**        **Math equivalent**                        **Description**
    ==================== ================== ========================================== =====================================================
    ``in_size``          (required)         ---                                          Population shape
    ``V_peak``           0 mV               :math:`V_{\mathrm{peak}}`                  Spike detection threshold (if :math:`\Delta_T > 0`)
    ``V_reset``          -60 mV             :math:`V_{\mathrm{reset}}`                 Reset potential
    ``t_ref``            0 ms               :math:`t_{\mathrm{ref}}`                   Absolute refractory duration
    ``g_L``              30 nS              :math:`g_{\mathrm{L}}`                     Leak conductance
    ``C_m``              281 pF             :math:`C_{\mathrm{m}}`                     Membrane capacitance
    ``E_L``              -70.6 mV           :math:`E_{\mathrm{L}}`                     Leak reversal potential
    ``Delta_T``          2 mV               :math:`\Delta_T`                           Exponential slope factor
    ``tau_w``            144 ms             :math:`\tau_w`                             Adaptation time constant
    ``a``                4 nS               :math:`a`                                  Subthreshold adaptation coupling
    ``b``                80.5 pA            :math:`b`                                  Spike-triggered adaptation increment
    ``V_th``             -50.4 mV           :math:`V_{\mathrm{th}}`                    Spike initiation threshold (in exponential term)
    ``I_e``              0 pA               :math:`I_{\mathrm{e}}`                     Constant external current
    ``gsl_error_tol``    1e-6               ---                                          RKF45 local error tolerance
    ``refractory_input`` ``False``          ---                                          If True, buffer spikes during refractory with NEST discounting
    ``V_initializer``    Constant(-70.6 mV) ---                                          Membrane initializer
    ``w_initializer``    Constant(0 pA)     ---                                          Adaptation current initializer
    ``spk_fun``          ReluGrad()         ---                                          Surrogate spike function
    ``spk_reset``        ``'hard'``         ---                                          Reset mode; hard reset matches NEST behavior
    ``ref_var``          ``False``          ---                                          If True, expose boolean refractory indicator
    ==================== ================== ========================================== =====================================================

    State Variables
    ---------------

    V : brainstate.HiddenState
        Membrane potential :math:`V_m` in millivolts. Shape: ``(*in_size,)``.
    w : brainstate.HiddenState
        Adaptation current in picoamperes. Shape: ``(*in_size,)``.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory grid steps (int32). Shape: ``(*in_size,)``.
    integration_step : brainstate.ShortTermState
        Persistent RKF45 internal step size in milliseconds. Shape: ``(*in_size,)``.
    I_stim : brainstate.ShortTermState
        One-step delayed current buffer in picoamperes. Shape: ``(*in_size,)``.
    last_spike_time : brainstate.ShortTermState
        Last emitted spike time in milliseconds (:math:`t+\Delta t` on spike). Shape: ``(*in_size,)``.
    refractory : brainstate.ShortTermState, optional
        Boolean refractory indicator. Only present if ``ref_var=True``. Shape: ``(*in_size,)``.

    Raises
    ------
    ValueError
        If ``V_reset >= V_peak``.
    ValueError
        If ``Delta_T < 0``.
    ValueError
        If ``V_peak < V_th``.
    ValueError
        If ``C_m <= 0``.
    ValueError
        If ``t_ref < 0``.
    ValueError
        If ``tau_w <= 0``.
    ValueError
        If ``gsl_error_tol <= 0``.
    ValueError
        If ``(V_peak - V_th) / Delta_T`` exceeds ``log(DBL_MAX / 1e20)`` (overflow protection).
    ValueError
        During integration: if ``V < -1e3`` or ``|w| > 1e6`` (numerical instability).

    Examples
    --------
    **Basic usage with delta-function input:**

    .. code-block:: python

        >>> from brainpy import state as bst
        >>> import saiunit as u
        >>> import brainstate as bs
        >>> import jax.numpy as jnp
        >>>
        >>> # Create 100 AdEx neurons
        >>> neu = bst.aeif_psc_delta(100, V_peak=0.*u.mV, V_reset=-60.*u.mV, t_ref=2.*u.ms)
        >>>
        >>> # Initialize state
        >>> with bs.environ.context(dt=0.1*u.ms):
        ...     neu.init_all_states()
        ...
        ...     # Simulate with delta input
        ...     for i in range(100):
        ...         # Delta spike input (+1 mV jump)
        ...         neu.add_delta_input('external', lambda: jnp.ones(100) * 1.0 * u.mV)
        ...         spk = neu.step_run(i, 0.0*u.pA)
        ...         print(f"t={i*0.1:.1f}ms: {spk.sum():.0f} spikes")

    **With refractory input buffering:**

    .. code-block:: python

        >>> # Enable refractory buffering
        >>> neu = bst.aeif_psc_delta(
        ...     100,
        ...     V_peak=0.*u.mV,
        ...     V_reset=-60.*u.mV,
        ...     t_ref=5.*u.ms,
        ...     refractory_input=True
        ... )
        >>>
        >>> with bs.environ.context(dt=0.1*u.ms):
        ...     neu.init_all_states()
        ...
        ...     # Spikes arriving during refractory are buffered and discounted
        ...     for i in range(100):
        ...         neu.add_delta_input('external', lambda: jnp.ones(100) * 2.0 * u.mV)
        ...         spk = neu.step_run(i, 0.0*u.pA)

    **IAF-like limit (Delta_T = 0):**

    .. code-block:: python

        >>> # Delta_T=0 disables exponential term
        >>> neu = bst.aeif_psc_delta(
        ...     100,
        ...     Delta_T=0.0*u.mV,
        ...     V_th=-55.*u.mV,
        ...     V_peak=-55.*u.mV,  # Must equal V_th when Delta_T=0
        ...     a=0.0*u.nS,        # No subthreshold adaptation
        ...     b=0.0*u.pA         # No spike-triggered adaptation
        ... )

    See Also
    --------
    aeif_psc_alpha : AdEx with alpha-function synaptic currents
    aeif_psc_exp : AdEx with exponential synaptic currents
    aeif_cond_alpha : AdEx with conductance-based synapses

    Notes
    -----
    - The default ``t_ref=0`` matches NEST and allows multiple spikes per simulation step.
    - Returned spike tensor is binary per simulation step (spike/no-spike), while internal
      adaptation dynamics follow NEST in-loop spike/reset behavior.
    - With ``Delta_T > 0``, the exponential term can cause rapid voltage growth near spike
      threshold. The adaptive RKF45 integrator automatically reduces step size to maintain
      accuracy.
    - For gradient-based learning, use surrogate functions like ``ReluGrad()``, ``SigmoidGrad()``,
      or ``SuperSpike()`` via the ``spk_fun`` parameter.

    References
    ----------
    .. [1] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
           model as an effective description of neuronal activity.
           Journal of Neurophysiology, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [2] NEST source: ``models/aeif_psc_delta.h`` and ``models/aeif_psc_delta.cpp``.
           https://github.com/nest/nest-simulator
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
        E_L: ArrayLike = -70.6 * u.mV,
        Delta_T: ArrayLike = 2.0 * u.mV,
        tau_w: ArrayLike = 144.0 * u.ms,
        a: ArrayLike = 4.0 * u.nS,
        b: ArrayLike = 80.5 * u.pA,
        V_th: ArrayLike = -50.4 * u.mV,
        I_e: ArrayLike = 0.0 * u.pA,
        gsl_error_tol: ArrayLike = 1e-6,
        refractory_input: bool = False,
        V_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
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
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.Delta_T = braintools.init.param(Delta_T, self.varshape)
        self.tau_w = braintools.init.param(tau_w, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        self.refractory_input = refractory_input
        self.V_initializer = V_initializer
        self.w_initializer = w_initializer
        self.ref_var = ref_var

        self._validate_parameters()

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
            raise ValueError('Ensure that V_reset < V_peak .')
        if cond_any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if cond_any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Ensure that C_m > 0')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if cond_any(self.tau_w <= 0.0 * u.ms):
            raise ValueError('tau_w must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        # Mirror NEST overflow guard for exponential term at spike time.
        validate_aeif_overflow(v_peak, v_th, delta_t)

    def init_state(self, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Creates and initializes state variables: membrane potential ``V``, adaptation
        current ``w``, refractory counter, integration step size, delayed current
        buffer, and spike timing. Optionally creates boolean refractory indicator
        if ``ref_var=True``.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments (ignored, for API compatibility).

        Notes
        -----
        - ``V`` and ``w`` are initialized using ``V_initializer`` and ``w_initializer``.
        - ``last_spike_time`` starts at ``-1e7 ms`` (far past, indicating no recent spikes).
        - ``refractory_step_count`` starts at 0 (not refractory).
        - ``integration_step`` starts at the global simulation timestep ``dt``.
        - ``I_stim`` (delayed current buffer) starts at 0 pA.
        - If ``ref_var=True``, ``refractory`` boolean array starts at ``False``.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        w = braintools.init.param(self.w_initializer, self.varshape)

        self.V = brainstate.HiddenState(V)
        self.w = brainstate.HiddenState(w)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Generate differentiable spike output using surrogate gradient function.

        Computes spike probability via the surrogate function ``spk_fun`` applied to
        normalized membrane potential. Used for gradient-based learning.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential in millivolts. If None, uses ``self.V.value``.
            Shape: ``(*in_size,)``.

        Returns
        -------
        spike : ArrayLike
            Differentiable spike output in range [0, 1]. Shape matches ``V``.
            Forward pass: approximately binary (0 or 1).
            Backward pass: uses surrogate gradient from ``spk_fun``.

        Notes
        -----
        The membrane potential is normalized as:

        .. math::

            v_{\mathrm{scaled}} = \frac{V - V_{\mathrm{th}}}{V_{\mathrm{th}} - V_{\mathrm{reset}}}.

        The surrogate function is then applied: ``spk_fun(v_scaled)``.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, w -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_peak_detect -- mutable
            auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, u.math.minimum(state.V, self.V_peak))

        delta_t_safe = u.math.where(self.Delta_T == 0.0 * u.mV, 1.0 * u.mV, self.Delta_T)
        exp_arg = u.math.clip((v_eff - self.V_th) / delta_t_safe, -500.0, 500.0)
        i_spike = self.g_L * self.Delta_T * u.math.exp(exp_arg)

        dV_raw = (
                     -self.g_L * (v_eff - self.E_L) + i_spike
                     - state.w + self.I_e + extra.i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        dw = (self.a * (v_eff - self.E_L) - state.w) / self.tau_w

        return DotDict(V=dV, w=dw)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, w -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_peak_detect.
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

    def update(self, x=0.0 * u.pA):
        r"""Advance neuron state by one simulation timestep using adaptive RKF45 integration.

        Integrates membrane and adaptation dynamics over interval :math:`(t, t+\Delta t]`
        using adaptive Runge-Kutta-Fehlberg 4(5) with per-neuron step size control. Handles
        delta-function input, refractory clamping, spike detection, and reset within the
        integration loop to match NEST semantics.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input in picoamperes. Shape: ``(*in_size,)`` or
            broadcastable. Default: ``0.0 * u.pA``.

        Returns
        -------
        spike : ArrayLike
            Binary spike indicator (0 or 1) for this timestep. Shape: ``(*in_size,)``.
            Value is 1 if any spike occurred during the integration interval, 0 otherwise.

        Raises
        ------
        ValueError
            If membrane potential drops below -1e3 mV (numerical instability).
        ValueError
            If adaptation current magnitude exceeds 1e6 pA (numerical instability).

        Notes
        -----
        **Integration algorithm:**

        1. For each neuron, iterate RKF45 substeps until ``t_local`` reaches ``dt``.
        2. At each substep:
            a. Compute RKF45 stages using ``_vector_field``.
            b. Compute higher-order and error estimates.
            c. Accept or reject substep based on local error vs ``gsl_error_tol``.
            d. On acceptance: apply spike detection, reset, and refractory handling
               via ``_event_fn``.
        3. After integration loop, decrement refractory counter once.
        4. Apply arriving spike weights directly to ``V`` as delta-function pulses.
        5. Store current input ``x`` into ``I_stim`` for next timestep (one-step delay).

        **Delta input handling:**

        Delta inputs (accumulated via ``sum_delta_inputs``) are applied as instantaneous
        voltage jumps after integration. Spike weights go directly into ``V`` as
        delta function pulses, not through synaptic state variables.

        **Refractory clamping:**

        During refractory (``refractory_step_count > 0``), the effective voltage in the
        RHS is clamped to ``V_reset`` and :math:`dV/dt = 0`. The adaptation current ``w``
        continues to evolve normally.

        **Multiple spikes per step:**

        With ``t_ref = 0``, multiple spikes can occur within one simulation step. The
        returned binary spike indicator is 1 if *any* spike occurred, but internal state
        (adaptation increments, refractory handling) reflects all spikes that occurred
        during integration.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        w = self.w.value  # pA
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Spike detection threshold: V_peak if Delta_T > 0, else V_th.
        v_peak_detect = u.math.where(self.Delta_T > 0.0 * u.mV, self.V_peak, self.V_th)

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, w=w)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            v_peak_detect=v_peak_detect,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, w = ode_state.V, ode_state.w
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in aeif_psc_delta dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Delta spike inputs: applied directly to V as instantaneous voltage jumps.
        w_delta = self.sum_delta_inputs(u.math.zeros_like(self.V.value), label='w_delta')
        V = V + w_delta

        # Write back state.
        self.V.value = V
        self.w.value = w
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
