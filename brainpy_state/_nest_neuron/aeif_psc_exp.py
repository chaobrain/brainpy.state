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
import brainunit as u
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, validate_aeif_overflow, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'aeif_psc_exp',
]


class aeif_psc_exp(NESTNeuron):
    r"""NEST-compatible adaptive exponential integrate-and-fire neuron with exponential synapses.

    Current-based adaptive exponential integrate-and-fire neuron with exponentially
    decaying synaptic currents. Implements the AdEx model of Brette & Gerstner (2005)
    with spike-triggered adaptation, subthreshold adaptation coupling, and separate
    excitatory/inhibitory exponential current synapses. Follows NEST
    ``models/aeif_psc_exp.{h,cpp}`` implementation exactly.

    **1. Mathematical Model**

    **Membrane and adaptation dynamics:**

    The membrane potential :math:`V` and adaptation current :math:`w` evolve as:

    .. math::

       C_m \frac{dV}{dt}
       =
       -g_L (V - E_L)
       + g_L \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)
       - w + I_{ex} - I_{in} + I_e + I_{stim}

    .. math::

       \tau_w \frac{dw}{dt} = a (V - E_L) - w

    where :math:`C_m` is membrane capacitance, :math:`g_L` is leak conductance,
    :math:`E_L` is leak reversal, :math:`\Delta_T` is the exponential slope factor,
    :math:`V_{th}` is the spike threshold, :math:`a` couples subthreshold voltage to
    adaptation, and :math:`\tau_w` is the adaptation time constant.

    **Synaptic current dynamics:**

    Excitatory and inhibitory currents decay exponentially:

    .. math::

       \frac{d I_{ex}}{dt} = -\frac{I_{ex}}{\tau_{syn,ex}},
       \qquad
       \frac{d I_{in}}{dt} = -\frac{I_{in}}{\tau_{syn,in}}

    Incoming spike weights (in pA) are split by sign and applied instantaneously:

    .. math::

       I_{ex} \leftarrow I_{ex} + \max(w, 0),
       \qquad
       I_{in} \leftarrow I_{in} + \max(-w, 0)

    **2. Refractory and Spike Handling (NEST Semantics)**

    During refractory period (:math:`r > 0` steps remaining), the effective voltage
    used in the RHS is clamped to :math:`V_{\text{reset}}` and :math:`dV/dt = 0`.
    Outside refractory, the effective voltage is :math:`\min(V, V_{\text{peak}})`.

    Spike detection threshold:
      - :math:`V_{\text{peak}}` if :math:`\Delta_T > 0` (exponential regime)
      - :math:`V_{th}` if :math:`\Delta_T = 0` (integrate-and-fire limit)

    On each detected spike:
      1. :math:`V \leftarrow V_{\text{reset}}`
      2. :math:`w \leftarrow w + b` (spike-triggered adaptation increment)
      3. Refractory counter set to ``refractory_counts + 1`` (if ``t_ref > 0``)

    Spike detection/reset occurs *inside* the RKF45 substep loop. With ``t_ref = 0``,
    multiple spikes can occur within one simulation step, matching NEST behavior.

    **3. Update Order Per Simulation Step**

    1. Integrate ODEs on :math:`(t, t+dt]` via adaptive RKF45 (Runge-Kutta-Fehlberg 4(5))
    2. Inside integration loop: apply refractory clamp, detect spike, reset, adapt
    3. After integration: decrement refractory counter by 1
    4. Apply arriving spike weights to :math:`I_{ex}`, :math:`I_{in}`
    5. Store external current input :math:`x` into one-step delayed buffer :math:`I_{\text{stim}}`

    **4. Numerical Integration**

    Uses adaptive RKF45 with local error control. Step size :math:`h` is adjusted
    to keep error below ``gsl_error_tol``. Integration step size is persistent across
    simulation steps for efficiency.

    Parameters
    ----------
    in_size : int or tuple of int
        Population shape. Scalar for 1D population, tuple for multi-dimensional.
    V_peak : ArrayLike, optional
        Spike detection threshold (if ``Delta_T > 0``). Units: mV. Default: 0.0 mV.
        Scalar or broadcastable to ``in_size``.
    V_reset : ArrayLike, optional
        Reset potential after spike. Units: mV. Default: -60.0 mV.
        Scalar or broadcastable to ``in_size``. Must satisfy ``V_reset < V_peak``.
    t_ref : ArrayLike, optional
        Absolute refractory period duration. Units: ms. Default: 0.0 ms.
        Scalar or broadcastable to ``in_size``. Zero allows multiple spikes per step.
    g_L : ArrayLike, optional
        Leak conductance. Units: nS. Default: 30.0 nS.
        Scalar or broadcastable to ``in_size``. Must be positive.
    C_m : ArrayLike, optional
        Membrane capacitance. Units: pF. Default: 281.0 pF.
        Scalar or broadcastable to ``in_size``. Must be positive.
    E_L : ArrayLike, optional
        Leak reversal potential. Units: mV. Default: -70.6 mV.
        Scalar or broadcastable to ``in_size``.
    Delta_T : ArrayLike, optional
        Exponential slope factor. Units: mV. Default: 2.0 mV.
        Scalar or broadcastable to ``in_size``. Zero recovers integrate-and-fire limit.
        Must be non-negative. Large values relative to ``V_peak - V_th`` may cause overflow.
    tau_w : ArrayLike, optional
        Adaptation time constant. Units: ms. Default: 144.0 ms.
        Scalar or broadcastable to ``in_size``. Must be positive.
    a : ArrayLike, optional
        Subthreshold adaptation coupling. Units: nS. Default: 4.0 nS.
        Scalar or broadcastable to ``in_size``. Couples voltage deviation to adaptation.
    b : ArrayLike, optional
        Spike-triggered adaptation increment. Units: pA. Default: 80.5 pA.
        Scalar or broadcastable to ``in_size``. Added to ``w`` on each spike.
    V_th : ArrayLike, optional
        Spike initiation threshold (in exponential term). Units: mV. Default: -50.4 mV.
        Scalar or broadcastable to ``in_size``. Must satisfy ``V_th <= V_peak``.
    tau_syn_ex : ArrayLike, optional
        Excitatory synaptic current time constant. Units: ms. Default: 0.2 ms.
        Scalar or broadcastable to ``in_size``. Must be positive.
    tau_syn_in : ArrayLike, optional
        Inhibitory synaptic current time constant. Units: ms. Default: 2.0 ms.
        Scalar or broadcastable to ``in_size``. Must be positive.
    I_e : ArrayLike, optional
        Constant external current. Units: pA. Default: 0.0 pA.
        Scalar or broadcastable to ``in_size``.
    gsl_error_tol : ArrayLike, optional
        RKF45 local error tolerance. Dimensionless. Default: 1e-6.
        Scalar or broadcastable to ``in_size``. Must be positive. Smaller values
        increase accuracy at the cost of smaller integration steps.
    V_initializer : Callable, optional
        Membrane potential initializer. Default: Constant(-70.6 mV).
        Must return quantity with mV units when called with ``(shape,)``.
    I_ex_initializer : Callable, optional
        Excitatory current initializer. Default: Constant(0.0 pA).
        Must return quantity with pA units when called with ``(shape,)``.
    I_in_initializer : Callable, optional
        Inhibitory current initializer. Default: Constant(0.0 pA).
        Must return quantity with pA units when called with ``(shape,)``.
    w_initializer : Callable, optional
        Adaptation current initializer. Default: Constant(0.0 pA).
        Must return quantity with pA units when called with ``(shape,)``.
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation. Default: ReluGrad().
        Must be a differentiable spike function from ``braintools.surrogate``.
    spk_reset : str, optional
        Spike reset mode. Default: 'hard'.
        - 'hard': Stop gradient through reset (matches NEST behavior)
        - 'soft': Allow gradient through reset
    ref_var : bool, optional
        If True, expose boolean refractory state variable. Default: False.
        When True, creates ``self.refractory`` indicating refractory status.
    name : str, optional
        Model instance name. Default: None (auto-generated).

    Parameter Mapping
    -----------------

    ==================== ================== ========================================== =======================================================
    **Parameter**        **Default**        **Math equivalent**                        **Description**
    ==================== ================== ========================================== =======================================================
    ``in_size``          (required)                                                    Population shape
    ``V_peak``           0 mV               :math:`V_\mathrm{peak}`                    Spike detection threshold (if ``Delta_T > 0``)
    ``V_reset``          -60 mV             :math:`V_\mathrm{reset}`                   Reset potential
    ``t_ref``            0 ms               :math:`t_\mathrm{ref}`                     Absolute refractory duration
    ``g_L``              30 nS              :math:`g_\mathrm{L}`                       Leak conductance
    ``C_m``              281 pF             :math:`C_\mathrm{m}`                       Membrane capacitance
    ``E_L``              -70.6 mV           :math:`E_\mathrm{L}`                       Leak reversal potential
    ``Delta_T``          2 mV               :math:`\Delta_T`                           Exponential slope factor
    ``tau_w``            144 ms             :math:`\tau_w`                             Adaptation time constant
    ``a``                4 nS               :math:`a`                                  Subthreshold adaptation
    ``b``                80.5 pA            :math:`b`                                  Spike-triggered adaptation increment
    ``V_th``             -50.4 mV           :math:`V_\mathrm{th}`                      Spike initiation threshold (in exponential term)
    ``tau_syn_ex``       0.2 ms             :math:`\tau_{\mathrm{syn,ex}}`             Excitatory exponential time constant
    ``tau_syn_in``       2.0 ms             :math:`\tau_{\mathrm{syn,in}}`             Inhibitory exponential time constant
    ``I_e``              0 pA               :math:`I_\mathrm{e}`                       Constant external current
    ``gsl_error_tol``    1e-6               (solver tolerance)                         RKF45 local error tolerance
    ``V_initializer``    Constant(-70.6 mV)                                            Membrane initializer
    ``I_ex_initializer`` Constant(0 pA)                                                Excitatory current initializer
    ``I_in_initializer`` Constant(0 pA)                                                Inhibitory current initializer
    ``w_initializer``    Constant(0 pA)                                                Adaptation current initializer
    ``spk_fun``          ReluGrad()                                                    Surrogate spike function
    ``spk_reset``        ``'hard'``                                                    Reset mode; hard reset matches NEST behavior
    ``ref_var``          ``False``                                                     If True, expose boolean refractory indicator
    ==================== ================== ========================================== =======================================================

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential. Shape: ``(*in_size,)``. Units: mV.
    I_ex : brainstate.HiddenState
        Excitatory synaptic current. Shape: ``(*in_size,)``. Units: pA.
    I_in : brainstate.HiddenState
        Inhibitory synaptic current. Shape: ``(*in_size,)``. Units: pA.
    w : brainstate.HiddenState
        Adaptation current. Shape: ``(*in_size,)``. Units: pA.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory steps. Shape: ``(*in_size,)``. Dtype: int32.
    integration_step : brainstate.ShortTermState
        Persistent RKF45 internal step size. Shape: ``(*in_size,)``. Units: ms.
    I_stim : brainstate.ShortTermState
        One-step delayed current buffer. Shape: ``(*in_size,)``. Units: pA.
    last_spike_time : brainstate.ShortTermState
        Last emitted spike time. Shape: ``(*in_size,)``. Units: ms.
        Updated to ``t + dt`` on spike emission.
    refractory : brainstate.ShortTermState, optional
        Boolean refractory indicator. Only exists if ``ref_var=True``.
        Shape: ``(*in_size,)``. Dtype: bool.

    Raises
    ------
    ValueError
        - If ``V_reset >= V_peak``
        - If ``Delta_T < 0``
        - If ``V_peak < V_th``
        - If ``C_m <= 0``
        - If ``t_ref < 0``
        - If any time constant ``<= 0``
        - If ``gsl_error_tol <= 0``
        - If ``(V_peak - V_th) / Delta_T`` is too large (overflow risk in exponential term)
        - If numerical instability detected (``V < -1e3`` or ``|w| > 1e6``)

    Notes
    -----
    **Implementation Details:**

    - **Adaptive integration:** RKF45 adjusts step size :math:`h` dynamically to meet error
      tolerance. Step size is persistent across simulation steps for efficiency.
    - **Refractory semantics:** During refractory, voltage is clamped to ``V_reset`` in the
      ODE RHS and ``dV/dt = 0``. This matches NEST exactly.
    - **Multiple spikes per step:** With ``t_ref = 0``, multiple spikes can occur within
      one simulation step. Each spike triggers reset and adaptation increment.
    - **Overflow protection:** Parameter validation checks that the exponential term
      :math:`\exp((V_{\text{peak}} - V_{th}) / \Delta_T)` does not overflow.
    - **Surrogate gradients:** For backpropagation, spike generation uses ``spk_fun``
      (default: ReLU gradient). Hard reset (``spk_reset='hard'``) stops gradient through
      reset, matching biological discontinuity.

    **Differences from other models:**

    - ``aeif_cond_exp``: Uses conductance-based synapses instead of current-based.
    - ``aeif_psc_alpha``: Uses alpha-function synapses instead of exponential.
    - ``aeif_psc_delta``: Uses delta-function (instantaneous) synapses.

    Examples
    --------
    Basic usage with constant input current:

    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> import brainstate
        >>>
        >>> # Create population of 100 AdEx neurons
        >>> neurons = bp.aeif_psc_exp(100, I_e=200 * u.pA)
        >>>
        >>> # Initialize states
        >>> with brainstate.environ.context(dt=0.1 * u.ms):
        ...     neurons.init_all_states()
        ...
        ...     # Run for 100 ms
        ...     spikes = []
        ...     for _ in range(1000):
        ...         spike = neurons.update()
        ...         spikes.append(spike)

    With synaptic input and refractory period:

    .. code-block:: python

        >>> # Create neurons with 2 ms refractory period
        >>> neurons = bp.aeif_psc_exp(
        ...     in_size=100,
        ...     t_ref=2.0 * u.ms,
        ...     tau_syn_ex=5.0 * u.ms,
        ...     tau_syn_in=10.0 * u.ms
        ... )
        >>>
        >>> with brainstate.environ.context(dt=0.1 * u.ms):
        ...     neurons.init_all_states()
        ...
        ...     # Add excitatory input (positive weights)
        ...     neurons.add_delta_input('exc', lambda: 100 * u.pA)
        ...
        ...     # Simulation step
        ...     spike = neurons.update(x=50 * u.pA)  # External current

    References
    ----------
    .. [1] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
           model as an effective description of neuronal activity.
           Journal of Neurophysiology, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [2] NEST source: ``models/aeif_psc_exp.h`` and
           ``models/aeif_psc_exp.cpp``.
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
        tau_syn_ex: ArrayLike = 0.2 * u.ms,
        tau_syn_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0.0 * u.pA,
        gsl_error_tol: ArrayLike = 1e-6,
        V_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        I_ex_initializer: Callable = braintools.init.Constant(0.0 * u.pA),
        I_in_initializer: Callable = braintools.init.Constant(0.0 * u.pA),
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
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        self.V_initializer = V_initializer
        self.I_ex_initializer = I_ex_initializer
        self.I_in_initializer = I_in_initializer
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
        r"""Validate parameter consistency and numerical stability.

        Checks parameter constraints following NEST validation rules:
          - ``V_reset < V_peak``
          - ``Delta_T >= 0``
          - ``V_peak >= V_th``
          - ``C_m > 0``
          - ``t_ref >= 0``
          - All time constants ``> 0``
          - ``gsl_error_tol > 0``
          - Exponential term overflow guard: ``(V_peak - V_th) / Delta_T < log(max_float / 1e20)``

        Raises
        ------
        ValueError
            If any parameter constraint is violated.
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

        # Enroll in the Simulator multi-receptor spike->current bridge (goal 28,
        # design A, pA variant): receptor_type=1 -> I_ex, =2 -> I_in. The bridge's
        # default receptor_input_unit is already u.pA, so no override is needed.
        self.n_receptors = 2

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

        V = braintools.init.param(self.V_initializer, self.varshape)
        I_ex = braintools.init.param(self.I_ex_initializer, self.varshape)
        I_in = braintools.init.param(self.I_in_initializer, self.varshape)
        w = braintools.init.param(self.w_initializer, self.varshape)

        self.V = brainstate.HiddenState(V)
        self.I_ex = brainstate.HiddenState(I_ex)
        self.I_in = brainstate.HiddenState(I_in)
        self.w = brainstate.HiddenState(w)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output using surrogate gradient.

        Applies the surrogate gradient function ``spk_fun`` to a scaled voltage.
        The scaling maps the threshold region to a suitable range for the surrogate.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential. Units: mV. If None, uses ``self.V.value``.
            Shape: ``(*in_size,)``.

        Returns
        -------
        ArrayLike
            Differentiable spike output in [0, 1]. Shape: ``(*in_size,)``.
            Dtype: float. Values close to 1 indicate spike, close to 0 indicate silence.

        Notes
        -----
        The voltage is scaled by ``(V - V_th) / (V_th - V_reset)`` before passing
        to the surrogate function. This normalizes the threshold crossing region
        for the surrogate gradient approximation.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, I_ex, I_in, w — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_peak_detect — mutable
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
                     + state.I_ex - state.I_in - state.w + self.I_e + extra.i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        dI_ex = -state.I_ex / self.tau_syn_ex
        dI_in = -state.I_in / self.tau_syn_in
        dw = (self.a * (v_eff - self.E_L) - state.w) / self.tau_w

        return DotDict(V=dV, I_ex=dI_ex, I_in=dI_in, w=dw)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, I_ex, I_in, w — ODE state variables.
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

    def update(self, x=0.0 * u.pA, w_by_rec=None):
        r"""Advance neuron state by one simulation timestep.

        Performs one simulation step of the adaptive exponential integrate-and-fire
        neuron using adaptive RKF45 integration. Handles refractory clamping, spike
        detection, reset, adaptation increment, and synaptic input application.

        The update sequence follows NEST semantics:

        1. **Integrate ODEs** over :math:`[t, t+dt]` using adaptive RKF45:

           - Vectorized integration with adaptive step size
           - Inside integration: apply refractory clamp, detect spikes, reset voltage,
             increment adaptation, update refractory counter
           - Step size persists across simulation steps for efficiency

        2. **Post-integration processing:**

           - Decrement refractory counter by 1
           - Apply delta inputs (spike weights) to :math:`I_{ex}`, :math:`I_{in}`
           - Store external current :math:`x` into one-step delayed buffer :math:`I_{\text{stim}}`
           - Update ``last_spike_time`` for neurons that spiked

        3. **Return spike tensor:**

           - Binary array indicating which neurons spiked during :math:`[t, t+dt]`

        Parameters
        ----------
        x : ArrayLike, optional
            External current input. Units: pA. Default: 0.0 pA.
            Shape: scalar or broadcastable to ``(*in_size,)``.
            Combined with ``current_inputs`` and stored in ``I_stim`` for next step.
        w_by_rec : ArrayLike or None, optional
            Per-receptor synaptic current jumps supplied by the Simulator's
            multi-receptor bridge, shape ``(*varshape, n_receptors)`` as a
            dimensionless ``pA`` mantissa. Column 0 (``receptor_type=1``) is added to
            the excitatory current ``I_ex`` and column 1 (``receptor_type=2``) to the
            inhibitory current ``I_in``. When ``None`` (the legacy path), the jumps are
            self-pulled from the ``label='w_ex'/'w_in'`` delta inputs instead.

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
            If numerical instability detected: ``V < -1e3`` or ``|w| > 1e6``.

        Notes
        -----
        Integration is performed with an adaptive vectorized RKF45 loop,
        including in-loop spike/reset/adaptation events and optional
        multiple spikes per step. All arithmetic is unit-aware via
        ``brainunit.math``.

        See Also
        --------
        init_state : Initialize state variables
        get_spike : Compute differentiable spike output
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        I_ex = self.I_ex.value  # pA
        I_in = self.I_in.value  # pA
        w = self.w.value  # pA
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Spike detection threshold: V_peak if Delta_T > 0, else V_th.
        v_peak_detect = u.math.where(self.Delta_T > 0.0 * u.mV, self.V_peak, self.V_th)

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, I_ex=I_ex, I_in=I_in, w=w)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            v_peak_detect=v_peak_detect,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, I_ex, I_in, w = ode_state.V, ode_state.I_ex, ode_state.I_in, ode_state.w
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in aeif_psc_exp dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration). Source-only bridge (goal
        # 28): the Simulator gathers the per-port pA deposit (shape
        # ``(*varshape, n_receptors)``) and passes it via ``w_by_rec`` -- column 0 ->
        # I_ex, column 1 -> I_in; the legacy path self-pulls the ``label='w_ex'/'w_in'``
        # delta inputs. Only the *source* of w_ex/w_in swaps; the exp jump is untouched.
        if w_by_rec is not None:
            w_by_rec = jnp.asarray(w_by_rec, dtype=dftype)
            w_ex = w_by_rec[..., 0] * u.pA
            w_in = w_by_rec[..., 1] * u.pA
        else:
            w_ex = self.sum_delta_inputs(u.math.zeros_like(self.I_ex.value), label='w_ex')
            w_in = self.sum_delta_inputs(u.math.zeros_like(self.I_in.value), label='w_in')

        # Apply synaptic spike inputs (current-based: direct addition in pA).
        I_ex = I_ex + w_ex
        I_in = I_in + w_in

        # Write back state.
        self.V.value = V
        self.I_ex.value = I_ex
        self.I_in.value = I_in
        self.w.value = w
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
