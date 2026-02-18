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

from brainpy_state._nest._base import NESTNeuron

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
        Must return quantity with mV units when called with ``(shape, batch_size)``.
    I_ex_initializer : Callable, optional
        Excitatory current initializer. Default: Constant(0.0 pA).
        Must return quantity with pA units when called with ``(shape, batch_size)``.
    I_in_initializer : Callable, optional
        Inhibitory current initializer. Default: Constant(0.0 pA).
        Must return quantity with pA units when called with ``(shape, batch_size)``.
    w_initializer : Callable, optional
        Adaptation current initializer. Default: Constant(0.0 pA).
        Must return quantity with pA units when called with ``(shape, batch_size)``.
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
        Membrane potential. Shape: ``(*in_size, *batch_size)``. Units: mV.
    I_ex : brainstate.HiddenState
        Excitatory synaptic current. Shape: ``(*in_size, *batch_size)``. Units: pA.
    I_in : brainstate.HiddenState
        Inhibitory synaptic current. Shape: ``(*in_size, *batch_size)``. Units: pA.
    w : brainstate.HiddenState
        Adaptation current. Shape: ``(*in_size, *batch_size)``. Units: pA.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory steps. Shape: ``(*in_size, *batch_size)``. Dtype: int32.
    integration_step : brainstate.ShortTermState
        Persistent RKF45 internal step size. Shape: ``(*in_size, *batch_size)``. Units: ms.
    I_stim : brainstate.ShortTermState
        One-step delayed current buffer. Shape: ``(*in_size, *batch_size)``. Units: pA.
    last_spike_time : brainstate.ShortTermState
        Last emitted spike time. Shape: ``(*in_size, *batch_size)``. Units: ms.
        Updated to ``t + dt`` on spike emission.
    refractory : brainstate.ShortTermState, optional
        Boolean refractory indicator. Only exists if ``ref_var=True``.
        Shape: ``(*in_size, *batch_size)``. Dtype: bool.

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

        >>> import brainpy.state as bp
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
        self.gsl_error_tol = braintools.init.param(gsl_error_tol, self.varshape)

        self.V_initializer = V_initializer
        self.I_ex_initializer = I_ex_initializer
        self.I_in_initializer = I_in_initializer
        self.w_initializer = w_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        r"""Convert quantity to unitless NumPy array.

        Parameters
        ----------
        x : ArrayLike
            Input quantity with units.
        unit : brainunit.Unit
            Unit to divide by before conversion.

        Returns
        -------
        np.ndarray
            Unitless float64 array.
        """
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _to_numpy_unitless(x):
        r"""Convert dimensionless quantity to NumPy array.

        Parameters
        ----------
        x : ArrayLike
            Input dimensionless quantity.

        Returns
        -------
        np.ndarray
            Float64 array.
        """
        return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast array to target shape.

        Parameters
        ----------
        x_np : np.ndarray
            Input array.
        shape : tuple
            Target shape.

        Returns
        -------
        np.ndarray
            Broadcasted array with shape ``shape``.
        """
        return np.broadcast_to(x_np, shape)

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
        v_reset = self._to_numpy(self.V_reset, u.mV)
        v_peak = self._to_numpy(self.V_peak, u.mV)
        v_th = self._to_numpy(self.V_th, u.mV)
        delta_t = self._to_numpy(self.Delta_T, u.mV)

        if np.any(v_reset >= v_peak):
            raise ValueError('Ensure that V_reset < V_peak .')
        if np.any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if np.any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Ensure that C_m > 0')
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

        # Mirror NEST overflow guard for exponential term at spike time.
        positive_dt = delta_t > 0.0
        if np.any(positive_dt):
            max_exp_arg = np.log(np.finfo(np.float64).max / 1e20)
            ratio = (v_peak - v_th) / np.where(positive_dt, delta_t, 1.0)
            if np.any(ratio[positive_dt] >= max_exp_arg):
                raise ValueError(
                    'The current combination of V_peak, V_th and Delta_T will lead to numerical overflow at spike '
                    'time; try for instance to increase Delta_T or to reduce V_peak to avoid this problem.'
                )

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Creates and initializes state variables:
          - ``V``: membrane potential (HiddenState)
          - ``I_ex``, ``I_in``: synaptic currents (HiddenState)
          - ``w``: adaptation current (HiddenState)
          - ``last_spike_time``: last spike time (ShortTermState)
          - ``refractory_step_count``: refractory counter (ShortTermState)
          - ``integration_step``: RKF45 step size (ShortTermState)
          - ``I_stim``: delayed current buffer (ShortTermState)
          - ``refractory``: boolean refractory flag (ShortTermState, if ``ref_var=True``)

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If None, no batch dimension is added.
            State shape will be ``(*in_size, batch_size)``.
        **kwargs
            Additional keyword arguments (unused).

        Notes
        -----
        Must be called within a ``brainstate.environ.context()`` with ``dt`` set.
        ``last_spike_time`` is initialized to -1e7 ms (long before simulation start).
        ``refractory_step_count`` is initialized to 0.
        ``integration_step`` is initialized to the simulation timestep ``dt``.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        I_ex = braintools.init.param(self.I_ex_initializer, self.varshape, batch_size)
        I_in = braintools.init.param(self.I_in_initializer, self.varshape, batch_size)
        w = braintools.init.param(self.w_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.I_ex = brainstate.HiddenState(I_ex)
        self.I_in = brainstate.HiddenState(I_in)
        self.w = brainstate.HiddenState(w)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = brainstate.environ.get_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to initial values.

        Re-initializes all state variables using the configured initializers.
        Equivalent to calling ``init_state`` but reuses existing state objects.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If None, no batch dimension is added.
        **kwargs
            Additional keyword arguments (unused).

        Notes
        -----
        Must be called within a ``brainstate.environ.context()`` with ``dt`` set.
        Resets all states to the same initial values as ``init_state``.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.I_ex.value = braintools.init.param(self.I_ex_initializer, self.varshape, batch_size)
        self.I_in.value = braintools.init.param(self.I_in_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        dt = brainstate.environ.get_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )
        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output using surrogate gradient.

        Applies the surrogate gradient function ``spk_fun`` to a scaled voltage.
        The scaling maps the threshold region to a suitable range for the surrogate.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential. Units: mV. If None, uses ``self.V.value``.
            Shape: ``(*in_size, *batch_size)``.

        Returns
        -------
        ArrayLike
            Differentiable spike output in [0, 1]. Shape: ``(*in_size, *batch_size)``.
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

    def _refractory_counts(self):
        r"""Compute number of refractory steps from refractory duration.

        Returns
        -------
        ArrayLike
            Number of refractory steps. Shape: ``(*in_size,)``. Dtype: int32.
            Computed as ``ceil(t_ref / dt)``.

        Notes
        -----
        Must be called within a ``brainstate.environ.context()`` with ``dt`` set.
        """
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _sum_signed_delta_inputs(self):
        r"""Collect and split delta inputs by sign into excitatory/inhibitory currents.

        Iterates over all registered delta inputs, evaluates callables, and splits
        contributions by sign. Positive weights accumulate in excitatory current,
        negative weights (absolute value) accumulate in inhibitory current.

        Returns
        -------
        w_ex : ArrayLike
            Total excitatory delta input. Shape: ``(*in_size, *batch_size)``. Units: pA.
            Sum of all positive delta inputs.
        w_in : ArrayLike
            Total inhibitory delta input. Shape: ``(*in_size, *batch_size)``. Units: pA.
            Sum of absolute values of all negative delta inputs.

        Notes
        -----
        Delta inputs are registered via ``add_delta_input(label, input_fn)``.
        If an input is callable, it is evaluated and then removed from the dict.
        This implements the NEST convention: excitatory weights are positive,
        inhibitory weights are negative, and each is integrated separately.
        """
        w_ex = u.math.zeros_like(self.I_ex.value)
        w_in = u.math.zeros_like(self.I_in.value)
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
    def _dynamics_scalar(v, I_ex, I_in, w, is_refractory, i_stim, p):
        r"""Compute RHS of ODEs for scalar state (single neuron).

        Implements the AdEx dynamics equations for membrane potential, synaptic
        currents, and adaptation current. Handles refractory clamping and exponential
        spike current.

        Parameters
        ----------
        v : float
            Membrane potential (mV, unitless).
        I_ex : float
            Excitatory synaptic current (pA, unitless).
        I_in : float
            Inhibitory synaptic current (pA, unitless).
        w : float
            Adaptation current (pA, unitless).
        is_refractory : bool
            True if neuron is in refractory period.
        i_stim : float
            External stimulus current (pA, unitless).
        p : dict
            Parameter dictionary with keys:
              - 'V_reset': reset potential (mV)
              - 'V_peak_rhs': spike detection threshold (mV)
              - 'E_L': leak reversal (mV)
              - 'C_m': capacitance (pF)
              - 'g_L': leak conductance (nS)
              - 'Delta_T': exponential slope (mV)
              - 'V_th': threshold in exponential (mV)
              - 'I_e': constant external current (pA)
              - 'tau_syn_ex': excitatory time constant (ms)
              - 'tau_syn_in': inhibitory time constant (ms)
              - 'tau_w': adaptation time constant (ms)
              - 'a': subthreshold adaptation coupling (nS)

        Returns
        -------
        dv : float
            Time derivative of membrane potential (mV/ms). Zero if refractory.
        dI_ex : float
            Time derivative of excitatory current (pA/ms).
        dI_in : float
            Time derivative of inhibitory current (pA/ms).
        dw : float
            Time derivative of adaptation current (pA/ms).

        Notes
        -----
        During refractory period, effective voltage is clamped to ``V_reset`` and
        ``dV/dt = 0``. Outside refractory, effective voltage is ``min(v, V_peak_rhs)``
        to prevent exponential overflow.

        If ``Delta_T == 0``, the exponential spike current is disabled (IAF limit).
        """
        v_eff = p['V_reset'] if is_refractory else min(v, p['V_peak_rhs'])

        i_spike = 0.0 if p['Delta_T'] == 0.0 else (
            p['g_L'] * p['Delta_T'] * math.exp((v_eff - p['V_th']) / p['Delta_T'])
        )

        dv = 0.0 if is_refractory else (
            -p['g_L'] * (v_eff - p['E_L']) + i_spike + I_ex - I_in - w + p['I_e'] + i_stim
        ) / p['C_m']

        dI_ex = -I_ex / p['tau_syn_ex']
        dI_in = -I_in / p['tau_syn_in']
        dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
        return dv, dI_ex, dI_in, dw

    def update(self, x=0.0 * u.pA):
        r"""Advance neuron state by one simulation timestep.

        Performs one simulation step of the adaptive exponential integrate-and-fire
        neuron using adaptive RKF45 integration. Handles refractory clamping, spike
        detection, reset, adaptation increment, and synaptic input application.

        The update sequence follows NEST semantics:

        1. **Integrate ODEs** over :math:`[t, t+dt]` using adaptive RKF45:

           - Per-neuron loop with scalar integration
           - Adaptive step size :math:`h` adjusted to meet ``gsl_error_tol``
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
            Shape: scalar or broadcastable to ``(*in_size, *batch_size)``.
            Combined with ``current_inputs`` and stored in ``I_stim`` for next step.

        Returns
        -------
        ArrayLike
            Binary spike array. Shape: ``(*in_size, *batch_size)``. Dtype: float.
            Value 1.0 indicates spike during this timestep, 0.0 indicates silence.
            For neurons with ``t_ref=0``, multiple internal spikes collapse to single
            output spike (binary per step).

        Raises
        ------
        ValueError
            If numerical instability detected: ``V < -1e3`` or ``|w| > 1e6``.

        Notes
        -----
        **Adaptive RKF45 Integration:**

        Uses Runge-Kutta-Fehlberg 4(5) with local error control. For each neuron,
        the integration loop runs until ``t_local >= dt`` or maximum iterations exceeded.
        Step size :math:`h` is adjusted to keep local error below ``gsl_error_tol``:

        - If error :math:`\leq` tolerance: accept step, increase :math:`h`
        - If error :math:`>` tolerance: reject step, decrease :math:`h`
        - Minimum step size: ``_MIN_H = 1e-8`` ms
        - Maximum iterations: ``_MAX_ITERS = 100000``

        The integration step size ``h`` is persistent (stored in ``integration_step``)
        to avoid re-initialization overhead on each simulation step.

        **Refractory Handling:**

        During refractory period (``refractory_step_count > 0``):
          - Effective voltage in ODE RHS is clamped to ``V_reset``
          - ``dV/dt = 0``
          - Adaptation and synaptic currents continue to evolve
          - Spike detection is suppressed

        After integration loop, refractory counter is decremented by 1.

        **Spike Detection and Reset:**

        Inside the integration loop, after each accepted RKF45 step:
          1. Check if ``V >= spike_threshold`` (``V_peak`` or ``V_th`` depending on ``Delta_T``)
          2. If spike detected:
             - Set ``V = V_reset``
             - Add spike-triggered adaptation: ``w = w + b``
             - Set refractory counter to ``refractory_counts + 1`` (if ``t_ref > 0``)
             - Mark neuron as spiked for output

        With ``t_ref = 0``, multiple spikes can occur within one simulation step.
        All internal spikes collapse to a single binary output spike.

        **Synaptic Input Handling:**

        Delta inputs (spike weights) are collected via ``_sum_signed_delta_inputs()``,
        split by sign, and added to ``I_ex`` / ``I_in`` after integration completes.
        This implements instantaneous synaptic current jumps.

        Current inputs (continuous currents) are summed via ``sum_current_inputs()``
        and stored in ``I_stim`` for use in the *next* simulation step (one-step delay).

        **Numerical Stability:**

        - Overflow protection: Parameter validation ensures exponential term does not
          overflow at spike time.
        - Instability detection: Raises ``ValueError`` if ``V < -1e3`` or ``|w| > 1e6``.
        - Minimum step size: Integration cannot proceed below ``_MIN_H = 1e-8`` ms.

        **Performance:**

        Per-neuron scalar loop in NumPy is used for adaptive integration. This is
        necessary because each neuron has independent adaptive step size and refractory
        state. For large populations, this is slower than vectorized Euler integration
        but provides higher accuracy and NEST-compatible semantics.

        See Also
        --------
        init_state : Initialize state variables
        get_spike : Compute differentiable spike output
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        I_ex = self._broadcast_to_state(self._to_numpy(self.I_ex.value, u.pA), v_shape)
        I_in = self._broadcast_to_state(self._to_numpy(self.I_in.value, u.pA), v_shape)
        w = self._broadcast_to_state(self._to_numpy(self.w.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            v_shape,
        )
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape)

        p = {
            'V_peak_rhs': self._broadcast_to_state(self._to_numpy(self.V_peak, u.mV), v_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
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
        w_ex = self._broadcast_to_state(self._to_numpy(w_ex_q, u.pA), v_shape)
        w_in = self._broadcast_to_state(self._to_numpy(w_in_q, u.pA), v_shape)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        spike_mask = np.zeros(v_shape, dtype=bool)
        V_next = np.empty_like(V)
        I_ex_next = np.empty_like(I_ex)
        I_in_next = np.empty_like(I_in)
        w_next = np.empty_like(w)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            y = np.asarray([V[idx], I_ex[idx], I_in[idx], w[idx]], dtype=np.float64)
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
                            y_[0], y_[1], y_[2], y_[3], is_refractory, i_stim[idx], local_p
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

                    if y[0] < -1e3 or y[3] < -1e6 or y[3] > 1e6:
                        raise ValueError('Numerical instability in aeif_psc_exp dynamics.')

                    if r_i > 0:
                        y[0] = local_p['V_reset']
                    elif y[0] >= v_peak_detect[idx]:
                        local_spike = True
                        y[0] = local_p['V_reset']
                        y[3] += local_p['b']
                        r_i = int(refr_counts[idx]) + 1 if int(refr_counts[idx]) > 0 else 0
                else:
                    fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
                    h_i = max(self._MIN_H, h_i * fac)

            if r_i > 0:
                r_i -= 1

            y[1] += w_ex[idx]
            y[2] += w_in[idx]

            spike_mask[idx] = local_spike
            V_next[idx] = y[0]
            I_ex_next[idx] = y[1]
            I_in_next[idx] = y[2]
            w_next[idx] = y[3]
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.I_ex.value = I_ex_next * u.pA
        self.I_in.value = I_in_next * u.pA
        self.w.value = w_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=jnp.float64)
