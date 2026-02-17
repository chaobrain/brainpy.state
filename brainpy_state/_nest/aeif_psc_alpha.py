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
    'aeif_psc_alpha',
]


class aeif_psc_alpha(Neuron):
    r"""NEST-compatible adaptive exponential integrate-and-fire neuron with alpha-shaped postsynaptic currents.

    This model implements the adaptive exponential integrate-and-fire (AdEx) neuron with
    current-based synapses following alpha-function kinetics. It replicates the behavior of
    NEST's ``aeif_psc_alpha`` model, including adaptive Runge-Kutta-Fehlberg (RKF45) numerical
    integration, in-loop spike detection and reset, and NEST-compatible refractory handling.

    **1. Mathematical Model**

    **Membrane Dynamics**

    The subthreshold membrane potential :math:`V` evolves according to:

    .. math::

       C_m \frac{dV}{dt} = -g_L (V - E_L) + g_L \Delta_T \exp\left(\frac{V - V_{th}}{\Delta_T}\right)
                           - w + I_{ex} - I_{in} + I_e + I_{stim}

    where:

    - :math:`C_m` -- membrane capacitance
    - :math:`g_L` -- leak conductance
    - :math:`E_L` -- leak reversal potential
    - :math:`\Delta_T` -- exponential slope factor (spike sharpness)
    - :math:`V_{th}` -- spike initiation threshold
    - :math:`w` -- adaptation current
    - :math:`I_{ex}`, :math:`I_{in}` -- excitatory and inhibitory synaptic currents
    - :math:`I_e` -- constant external current
    - :math:`I_{stim}` -- time-varying external input (one-step delayed)

    The exponential term :math:`g_L \Delta_T \exp((V - V_{th})/\Delta_T)` causes rapid
    voltage acceleration near :math:`V_{th}`, producing spike initiation. Setting
    :math:`\Delta_T = 0` recovers the leaky integrate-and-fire (LIF) limit.

    **Adaptation Dynamics**

    The adaptation current :math:`w` provides spike-frequency adaptation and subthreshold
    coupling:

    .. math::

       \tau_w \frac{dw}{dt} = a(V - E_L) - w

    - Subthreshold adaptation: parameter :math:`a` couples :math:`w` to membrane potential
    - Spike-triggered adaptation: at each spike, :math:`w \leftarrow w + b`

    **Alpha-Function Synaptic Currents**

    Excitatory and inhibitory currents are modeled as alpha functions, each requiring
    two state variables:

    .. math::

       \frac{d\,dI_{ex}}{dt} = -\frac{dI_{ex}}{\tau_{syn,ex}}, \quad
       \frac{dI_{ex}}{dt} = dI_{ex} - \frac{I_{ex}}{\tau_{syn,ex}}

    .. math::

       \frac{d\,dI_{in}}{dt} = -\frac{dI_{in}}{\tau_{syn,in}}, \quad
       \frac{dI_{in}}{dt} = dI_{in} - \frac{I_{in}}{\tau_{syn,in}}

    Incoming spike weights :math:`w_j` (in pA) are split by sign and delivered as:

    .. math::

       dI_{ex} \leftarrow dI_{ex} + \frac{e}{\tau_{syn,ex}} \max(w_j, 0)

    .. math::

       dI_{in} \leftarrow dI_{in} + \frac{e}{\tau_{syn,in}} \max(-w_j, 0)

    where :math:`e = \exp(1)` provides the alpha-function normalization.

    **2. Spike Detection and Reset**

    **Threshold Crossing**

    Spike detection threshold depends on :math:`\Delta_T`:

    - If :math:`\Delta_T > 0`: spike when :math:`V \geq V_{peak}`
    - If :math:`\Delta_T = 0`: spike when :math:`V \geq V_{th}` (LIF-like)

    **Reset Mechanism**

    Upon spike detection:

    1. :math:`V \leftarrow V_{reset}`
    2. :math:`w \leftarrow w + b` (spike-triggered adaptation)
    3. Refractory counter set to :math:`\lceil t_{ref}/dt \rceil + 1` (if :math:`t_{ref} > 0`)

    Spike detection and reset occur *inside* the RKF45 integration substeps, allowing
    multiple spikes per simulation time step when :math:`t_{ref} = 0`.

    **3. Refractory Period Handling**

    During the refractory period (:math:`r > 0` steps remaining):

    - Membrane potential clamped: :math:`V_{eff} = V_{reset}`
    - Voltage derivative forced: :math:`dV/dt = 0`
    - Alpha currents and adaptation continue evolving normally

    After each time step, the refractory counter is decremented: :math:`r \leftarrow r - 1`.

    **4. Numerical Integration**

    The model uses adaptive Runge-Kutta-Fehlberg (RKF45) with local error control:

    - **Order**: 5th-order accurate solution with 4th-order error estimate
    - **Error tolerance**: controlled by ``gsl_error_tol`` (default :math:`10^{-6}`)
    - **Step size adaptation**: :math:`h_{new} = h \cdot \min(5, \max(0.2, 0.9 (\epsilon/\text{err})^{0.2}))`
    - **Minimum step**: :math:`h_{min} = 10^{-8}` ms to prevent stalling
    - **Persistent step size**: each neuron maintains its own integration step size across time

    The RKF45 Butcher tableau coefficients follow the standard formulation with stages
    :math:`k_1` through :math:`k_6`.

    **5. Update Sequence**

    Each simulation step processes state updates in this order:

    1. **Integration loop**: Integrate ODEs from :math:`t` to :math:`t + dt` using RKF45
       substeps, checking for spikes and applying resets within the loop
    2. **Refractory decrement**: After integration, decrement refractory counter once
    3. **Synaptic input delivery**: Add spike weights to :math:`dI_{ex}` and :math:`dI_{in}`
    4. **External current update**: Store current input :math:`x` into one-step-delayed buffer
       :math:`I_{stim}` (to be used in next step)

    Parameters
    ----------
    in_size : int, tuple of int
        Shape of the neuron population. Can be an integer (1D) or tuple (multi-dimensional).
    V_peak : ArrayLike, optional
        Spike detection threshold voltage. Default: ``0.0 * u.mV``.
        Used for threshold detection when :math:`\Delta_T > 0`.
    V_reset : ArrayLike, optional
        Reset potential after spike. Default: ``-60.0 * u.mV``.
    t_ref : ArrayLike, optional
        Absolute refractory period duration. Default: ``0.0 * u.ms``.
        During refractory period, :math:`V` is clamped to :math:`V_{reset}` and :math:`dV/dt = 0`.
    g_L : ArrayLike, optional
        Leak conductance. Default: ``30.0 * u.nS``.
    C_m : ArrayLike, optional
        Membrane capacitance. Default: ``281.0 * u.pF``.
        Determines membrane time constant :math:`\tau_m = C_m / g_L`.
    E_L : ArrayLike, optional
        Leak reversal potential. Default: ``-70.6 * u.mV``.
    Delta_T : ArrayLike, optional
        Exponential slope factor. Default: ``2.0 * u.mV``.
        Controls spike sharpness; set to 0 for LIF-like behavior.
    tau_w : ArrayLike, optional
        Adaptation time constant. Default: ``144.0 * u.ms``.
    a : ArrayLike, optional
        Subthreshold adaptation coupling. Default: ``4.0 * u.nS``.
        Couples adaptation current to membrane potential deviation from :math:`E_L`.
    b : ArrayLike, optional
        Spike-triggered adaptation increment. Default: ``80.5 * u.pA``.
        Added to :math:`w` at each spike.
    V_th : ArrayLike, optional
        Spike initiation threshold. Default: ``-50.4 * u.mV``.
        Appears in exponential term and as fallback spike threshold when :math:`\Delta_T = 0`.
    tau_syn_ex : ArrayLike, optional
        Excitatory synaptic alpha time constant. Default: ``0.2 * u.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory synaptic alpha time constant. Default: ``2.0 * u.ms``.
    I_e : ArrayLike, optional
        Constant external current. Default: ``0.0 * u.pA``.
    gsl_error_tol : ArrayLike, optional
        RKF45 local error tolerance. Default: ``1e-6``.
        Smaller values increase accuracy but require smaller integration steps.
    V_initializer : Callable, optional
        Membrane potential initializer. Default: ``Constant(-70.6 * u.mV)``.
    I_ex_initializer : Callable, optional
        Excitatory current initializer. Default: ``Constant(0.0 * u.pA)``.
    I_in_initializer : Callable, optional
        Inhibitory current initializer. Default: ``Constant(0.0 * u.pA)``.
    w_initializer : Callable, optional
        Adaptation current initializer. Default: ``Constant(0.0 * u.pA)``.
    spk_fun : Callable, optional
        Surrogate gradient function for differentiable spike generation.
        Default: ``ReluGrad()``.
    spk_reset : str, optional
        Spike reset mode: ``'soft'`` (subtract threshold) or ``'hard'`` (stop gradient).
        Default: ``'hard'`` (matches NEST behavior).
    ref_var : bool, optional
        If ``True``, expose a boolean ``refractory`` state variable indicating refractory status.
        Default: ``False``.
    name : str, optional
        Name of the neuron population.

    Parameter Mapping
    -----------------

    ==================== ================== ========================================== =====================================================
    **Parameter**        **Default**        **Math equivalent**                        **Description**
    ==================== ================== ========================================== =====================================================
    ``in_size``          (required)         —                                          Population shape
    ``V_peak``           0 mV               :math:`V_\mathrm{peak}`                    Spike detection threshold (if :math:`\Delta_T > 0`)
    ``V_reset``          -60 mV             :math:`V_\mathrm{reset}`                   Reset potential
    ``t_ref``            0 ms               :math:`t_\mathrm{ref}`                     Absolute refractory duration
    ``g_L``              30 nS              :math:`g_\mathrm{L}`                       Leak conductance
    ``C_m``              281 pF             :math:`C_\mathrm{m}`                       Membrane capacitance
    ``E_L``              -70.6 mV           :math:`E_\mathrm{L}`                       Leak reversal potential
    ``Delta_T``          2 mV               :math:`\Delta_T`                           Exponential slope factor
    ``tau_w``            144 ms             :math:`\tau_w`                             Adaptation time constant
    ``a``                4 nS               :math:`a`                                  Subthreshold adaptation coupling
    ``b``                80.5 pA            :math:`b`                                  Spike-triggered adaptation increment
    ``V_th``             -50.4 mV           :math:`V_\mathrm{th}`                      Spike initiation threshold
    ``tau_syn_ex``       0.2 ms             :math:`\tau_{\mathrm{syn,ex}}`             Excitatory alpha time constant
    ``tau_syn_in``       2.0 ms             :math:`\tau_{\mathrm{syn,in}}`             Inhibitory alpha time constant
    ``I_e``              0 pA               :math:`I_\mathrm{e}`                       Constant external current
    ``gsl_error_tol``    1e-6               —                                          RKF45 local error tolerance
    ``V_initializer``    Constant(-70.6 mV) —                                          Membrane initializer
    ``I_ex_initializer`` Constant(0 pA)     —                                          Excitatory current initializer
    ``I_in_initializer`` Constant(0 pA)     —                                          Inhibitory current initializer
    ``w_initializer``    Constant(0 pA)     —                                          Adaptation current initializer
    ``spk_fun``          ReluGrad()         —                                          Surrogate spike function
    ``spk_reset``        ``'hard'``         —                                          Reset mode (hard matches NEST)
    ``ref_var``          ``False``          —                                          Expose boolean refractory indicator
    ==================== ================== ========================================== =====================================================

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential, shape ``(*in_size, *batch_size)`` with unit mV.
    dI_ex : brainstate.ShortTermState
        Excitatory alpha auxiliary state (derivative of :math:`I_{ex}`), unitless array.
    I_ex : brainstate.HiddenState
        Excitatory synaptic current, unit pA.
    dI_in : brainstate.ShortTermState
        Inhibitory alpha auxiliary state (derivative of :math:`I_{in}`), unitless array.
    I_in : brainstate.HiddenState
        Inhibitory synaptic current, unit pA.
    w : brainstate.HiddenState
        Adaptation current, unit pA.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory time steps (int32 array).
    integration_step : brainstate.ShortTermState
        Current RKF45 integration step size, unit ms. Persists across simulation steps.
    I_stim : brainstate.ShortTermState
        One-step-delayed external current buffer, unit pA.
    last_spike_time : brainstate.ShortTermState
        Time of last spike emission, unit ms. Updated to :math:`t + dt` on spike.
    refractory : brainstate.ShortTermState, optional
        Boolean refractory indicator (only if ``ref_var=True``).

    Returns
    -------
    spike : Array
        Binary spike indicator array with shape ``(*in_size, *batch_size)``, dtype float.
        Value is 1.0 where spikes occurred in the current step, 0.0 otherwise.

    Raises
    ------
    ValueError
        - If :math:`V_{reset} \geq V_{peak}`
        - If :math:`\Delta_T < 0`
        - If :math:`V_{peak} < V_{th}`
        - If :math:`C_m \leq 0`
        - If :math:`t_{ref} < 0`
        - If any time constant :math:`\leq 0`
        - If ``gsl_error_tol`` :math:`\leq 0`
        - If :math:`(V_{peak} - V_{th})/\Delta_T` would cause exponential overflow
        - If numerical instability detected during integration (:math:`V < -1000` mV or
          :math:`|w| > 10^6` pA)

    Notes
    -----
    **NEST Compatibility**

    - Replicates NEST ``aeif_psc_alpha`` dynamics including RKF45 integration and in-loop
      spike handling
    - Default parameter values match NEST defaults (converted to SI units)
    - Refractory clamping follows NEST semantics: :math:`V_{eff} = V_{reset}` during
      refractory, with :math:`dV/dt = 0`

    **Numerical Considerations**


    - Maximum iteration limit: 100,000 substeps per time step (prevents infinite loops)
    - Minimum step size: :math:`h_{min} = 10^{-8}` ms
    - Voltage capping during integration: :math:`V_{eff} = \min(V, V_{peak})` outside
      refractory period to prevent exponential overflow
    - Overflow protection: validates that :math:`\exp((V_{peak} - V_{th})/\Delta_T)`
      remains within floating-point range

    **Multiple Spikes Per Step**

    With :math:`t_{ref} = 0` (default), a neuron can spike multiple times within a single
    simulation step. The internal adaptation :math:`w` accumulates all spike-triggered
    increments :math:`b`, but the returned spike tensor is binary (0 or 1) per step.

    **Surrogate Gradients**

    The ``spk_fun`` parameter controls backpropagation through spikes for gradient-based
    learning. The surrogate function approximates the derivative of the Heaviside step
    function during backward passes.

    Examples
    --------
    Basic usage with default parameters:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
        >>> import brainstate as bs
        >>>
        >>> # Create a population of 100 AdEx neurons
        >>> neuron = bst.aeif_psc_alpha(100)
        >>>
        >>> # Initialize states
        >>> with bs.environ.context(dt=0.1 * u.ms):
        ...     neuron.init_all_states()
        >>>
        >>> # Simulate one step with external current
        >>> with bs.environ.context(dt=0.1 * u.ms):
        ...     spikes = neuron.update(x=500 * u.pA)
        >>> spikes.shape
        (100,)

    Custom parameters for fast-spiking interneuron:

    .. code-block:: python

        >>> # Fast-spiking configuration
        >>> fs_neuron = bst.aeif_psc_alpha(
        ...     in_size=50,
        ...     C_m=150 * u.pF,
        ...     g_L=20 * u.nS,
        ...     tau_w=30 * u.ms,
        ...     a=0 * u.nS,  # Minimal subthreshold adaptation
        ...     b=20 * u.pA,  # Small spike-triggered adaptation
        ...     V_th=-52 * u.mV,
        ...     Delta_T=1.5 * u.mV,
        ...     tau_syn_ex=0.5 * u.ms,
        ...     tau_syn_in=1.0 * u.ms,
        ... )

    Connecting to upstream spike sources:

    .. code-block:: python

        >>> import brainevent as be
        >>>
        >>> # Create presynaptic spike generator
        >>> spike_gen = bst.PoissonSpike(100, freq=10 * u.Hz)
        >>>
        >>> # Create postsynaptic AdEx neurons
        >>> neurons = bst.aeif_psc_alpha(50)
        >>>
        >>> # Create projection with alpha synapses
        >>> proj = be.nn.FixedProb(
        ...     pre=spike_gen,
        ...     post=neurons,
        ...     prob=0.2,
        ...     weight=50.0,  # pA per spike
        ... )

    See Also
    --------
    aeif_cond_alpha : Conductance-based AdEx with alpha synapses
    aeif_psc_exp : AdEx with exponential postsynaptic currents
    aeif_psc_delta : AdEx with delta-function synaptic currents
    iaf_psc_alpha : Leaky integrate-and-fire with alpha currents (set ``Delta_T=0``)

    References
    ----------
    .. [1] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
           model as an effective description of neuronal activity.
           Journal of Neurophysiology, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [2] Gerstner W, Kistler WM, Naud R, Paninski L (2014). Neuronal Dynamics:
           From Single Neurons to Networks and Models of Cognition.
           Cambridge University Press. Chapter 6.
    .. [3] NEST Simulator Documentation. ``aeif_psc_alpha`` model.
           https://nest-simulator.readthedocs.io/
    .. [4] NEST source code: ``models/aeif_psc_alpha.h`` and ``models/aeif_psc_alpha.cpp``.
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
        r"""Initialize the aeif_psc_alpha neuron model.

        See class docstring for detailed parameter descriptions.
        """
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
        r"""Convert brainunit quantity to NumPy float64 array by dividing out unit.

        Parameters
        ----------
        x : ArrayLike
            Input quantity with physical units.
        unit : brainunit.Unit
            Unit to divide out (e.g., ``u.mV``, ``u.pA``).

        Returns
        -------
        np.ndarray
            Unitless float64 array with numerical values in the specified unit.
        """
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _to_numpy_unitless(x):
        r"""Convert unitless or unit-bearing input to NumPy float64 array.

        Parameters
        ----------
        x : ArrayLike
            Input value (unitless scalar or array).

        Returns
        -------
        np.ndarray
            Float64 array.
        """
        return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast NumPy array to target shape.

        Parameters
        ----------
        x_np : np.ndarray
            Input array to broadcast.
        shape : tuple of int
            Target shape.

        Returns
        -------
        np.ndarray
            Broadcasted array with shape ``shape``, sharing memory with ``x_np``
            where possible.
        """
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        r"""Validate parameter constraints and check for numerical overflow conditions.

        Raises
        ------
        ValueError
            If any of the following conditions are violated:

            - :math:`V_{reset} < V_{peak}` (reset must be below spike threshold)
            - :math:`\Delta_T \geq 0` (slope factor must be non-negative)
            - :math:`V_{peak} \geq V_{th}` (detection threshold must exceed initiation)
            - :math:`C_m > 0` (capacitance must be positive)
            - :math:`t_{ref} \geq 0` (refractory time cannot be negative)
            - :math:`\tau_{syn,ex}, \tau_{syn,in}, \tau_w > 0` (time constants must be positive)
            - ``gsl_error_tol`` :math:`> 0` (tolerance must be positive)
            - :math:`\exp((V_{peak} - V_{th})/\Delta_T)` within float64 range
              (prevents overflow at spike time)

        Notes
        -----
        The overflow check mirrors NEST's validation: it ensures the exponential term
        :math:`g_L \Delta_T \exp((V_{peak} - V_{th})/\Delta_T)` remains computable
        in float64 precision. Uses threshold :math:`\log(\text{float64}_{\max} / 10^{20})`
        to provide safety margin.
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

    def _safe_dt(self):
        r"""Retrieve current simulation time step with fallback.

        Returns
        -------
        Quantity
            Simulation time step with unit ms. Returns ``0.1 * u.ms`` if no context is set.

        Notes
        -----
        Used during state initialization when ``brainstate.environ.context`` may not yet
        be active. The fallback value is arbitrary and will be overwritten when the first
        update occurs within a proper context.
        """
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Creates and initializes the membrane potential, synaptic currents, adaptation
        current, refractory counters, integration step size, and stimulus buffer.

        Parameters
        ----------
        batch_size : int, optional
            Number of batched simulations. If provided, states have shape
            ``(*in_size, batch_size)``. If ``None``, states have shape ``(*in_size,)``.
        **kwargs : dict
            Additional keyword arguments (unused, for API compatibility).

        Notes
        -----
        State variables created:

        - ``V``: membrane potential (mV)
        - ``dI_ex``, ``I_ex``: excitatory alpha current states (unitless aux, pA current)
        - ``dI_in``, ``I_in``: inhibitory alpha current states (unitless aux, pA current)
        - ``w``: adaptation current (pA)
        - ``last_spike_time``: last spike time (ms), initialized to -10^7 ms
        - ``refractory_step_count``: remaining refractory steps (int32), initialized to 0
        - ``integration_step``: RKF45 step size (ms), initialized to current ``dt``
        - ``I_stim``: one-step-delayed external current buffer (pA), initialized to 0
        - ``refractory``: boolean refractory indicator (only if ``ref_var=True``)
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        I_ex = braintools.init.param(self.I_ex_initializer, self.varshape, batch_size)
        I_in = braintools.init.param(self.I_in_initializer, self.varshape, batch_size)
        w = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))

        self.V = brainstate.HiddenState(V)
        self.dI_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.I_ex = brainstate.HiddenState(I_ex)
        self.dI_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.I_in = brainstate.HiddenState(I_in)
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

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to their initial values.

        Reinitializes membrane potential, synaptic currents, adaptation current,
        refractory counters, integration step size, and stimulus buffer without
        creating new state objects.

        Parameters
        ----------
        batch_size : int, optional
            Number of batched simulations. If provided, states are reset to shape
            ``(*in_size, batch_size)``. If ``None``, states have shape ``(*in_size,)``.
        **kwargs : dict
            Additional keyword arguments (unused, for API compatibility).

        Notes
        -----
        This method is more efficient than ``init_state`` when states already exist,
        as it reuses existing state containers and only updates their values.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.I_ex.value = braintools.init.param(self.I_ex_initializer, self.varshape, batch_size)
        self.I_in.value = braintools.init.param(self.I_in_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(self.V.value / u.mV))
        self.dI_ex.value = np.asarray(zeros, dtype=np.float64)
        self.dI_in.value = np.asarray(zeros, dtype=np.float64)
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
        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output using surrogate gradient.

        Applies the surrogate spike function to the scaled membrane potential for
        gradient-based learning. This method is used for backpropagation and does
        not affect the internal spike detection logic (which uses hard threshold
        crossing during integration).

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential array with unit mV. If ``None``, uses the current
            ``self.V.value``. Shape: ``(*in_size, *batch_size)``.

        Returns
        -------
        spike : Array
            Differentiable spike output with shape matching ``V``. Values are continuous
            in the forward pass (soft spikes) but use surrogate gradients in the backward
            pass. Typically in range [0, 1] depending on surrogate function.

        Notes
        -----
        The voltage is scaled before applying the surrogate function:

        .. math::

            v_{scaled} = \\frac{V - V_{th}}{V_{th} - V_{reset}}

        This normalization ensures the surrogate function operates in a consistent range
        regardless of the specific voltage parameters.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        r"""Compute refractory period duration in discrete time steps.

        Returns
        -------
        Array
            Number of time steps for refractory period, computed as
            :math:`\lceil t_{ref} / dt \rceil` and cast to int32.
            Shape matches ``self.varshape``.

        Notes
        -----
        The ceiling operation ensures that partial time steps are counted as full steps,
        matching NEST's discrete-time refractory handling. For example, with
        :math:`t_{ref} = 2.5` ms and :math:`dt = 1.0` ms, the result is 3 steps.
        """
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _sum_signed_delta_inputs(self):
        r"""Collect and split synaptic spike inputs by sign.

        Retrieves all registered delta inputs (from projections), evaluates callables,
        and separates positive (excitatory) and negative (inhibitory) contributions.

        Returns
        -------
        w_ex : Array
            Total excitatory input (sum of positive components), unit pA.
            Shape: ``(*in_size, *batch_size)``.
        w_in : Array
            Total inhibitory input (sum of absolute values of negative components), unit pA.
            Shape: ``(*in_size, *batch_size)``.

        Notes
        -----
        - Positive spike weights contribute to ``w_ex``
        - Negative spike weights contribute to ``w_in`` (with sign flipped to positive)
        - Callable inputs are evaluated and then removed from the delta_inputs dict
        - Non-callable inputs are removed immediately (assumed to be stale)

        This splitting allows the model to apply different synaptic time constants
        (``tau_syn_ex`` vs ``tau_syn_in``) to excitatory and inhibitory inputs.
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
    def _dynamics_scalar(v, dI_ex, I_ex, dI_in, I_in, w, is_refractory, i_stim, p):
        r"""Compute time derivatives for all state variables (single neuron, scalar computation).

        This function implements the right-hand side of the ODE system for one neuron,
        used within the RKF45 integration loop.

        Parameters
        ----------
        v : float
            Membrane potential (mV, unitless scalar).
        dI_ex : float
            Excitatory alpha auxiliary state (unitless).
        I_ex : float
            Excitatory synaptic current (pA, unitless scalar).
        dI_in : float
            Inhibitory alpha auxiliary state (unitless).
        I_in : float
            Inhibitory synaptic current (pA, unitless scalar).
        w : float
            Adaptation current (pA, unitless scalar).
        is_refractory : bool
            ``True`` if neuron is in refractory period.
        i_stim : float
            External current input (pA, unitless scalar).
        p : dict
            Parameter dictionary with unitless float values for all model parameters.
            Required keys: ``'V_reset'``, ``'V_peak_rhs'``, ``'E_L'``, ``'C_m'``, ``'g_L'``,
            ``'Delta_T'``, ``'V_th'``, ``'tau_syn_ex'``, ``'tau_syn_in'``, ``'tau_w'``,
            ``'a'``, ``'I_e'``.

        Returns
        -------
        tuple of float
            Six derivatives: ``(dv, ddI_ex, dI_ex_dt, ddI_in, dI_in_dt, dw)``

            - ``dv``: :math:`dV/dt` (mV/ms, unitless)
            - ``ddI_ex``: :math:`d^2I_{ex}/dt^2` (unitless)
            - ``dI_ex_dt``: :math:`dI_{ex}/dt` (unitless)
            - ``ddI_in``: :math:`d^2I_{in}/dt^2` (unitless)
            - ``dI_in_dt``: :math:`dI_{in}/dt` (unitless)
            - ``dw``: :math:`dw/dt` (pA/ms, unitless)

        Notes
        -----
        **Effective Voltage**

        - If ``is_refractory``: :math:`v_{eff} = V_{reset}`
        - Otherwise: :math:`v_{eff} = \min(v, V_{peak})` (caps voltage to prevent overflow)

        **Membrane Dynamics**

        - Leak current: :math:`I_{leak} = g_L (v_{eff} - E_L)`
        - Spike current: :math:`I_{spike} = g_L \Delta_T \exp((v_{eff} - V_{th})/\Delta_T)`
          (zero if :math:`\Delta_T = 0`)
        - Total current: :math:`I_{total} = -I_{leak} + I_{spike} + I_{ex} - I_{in} - w + I_e + i_{stim}`
        - Derivative: :math:`dV/dt = I_{total} / C_m` (zero if refractory)

        **Alpha Currents**

        Second-order system for each channel:

        - :math:`d^2I/dt^2 = -dI/\tau`
        - :math:`dI/dt = dI - I/\tau`

        **Adaptation**

        - :math:`dw/dt = (a \cdot (v_{eff} - E_L) - w) / \tau_w`

        This function uses pure Python arithmetic for compatibility with NumPy's scalar
        operations within the RKF45 loop.
        """
        v_eff = p['V_reset'] if is_refractory else min(v, p['V_peak_rhs'])

        i_spike = 0.0 if p['Delta_T'] == 0.0 else (
            p['g_L'] * p['Delta_T'] * math.exp((v_eff - p['V_th']) / p['Delta_T'])
        )
        dv = 0.0 if is_refractory else (
            -p['g_L'] * (v_eff - p['E_L']) + i_spike + I_ex - I_in - w + p['I_e'] + i_stim
        ) / p['C_m']

        ddI_ex = -dI_ex / p['tau_syn_ex']
        dI_ex_dt = dI_ex - (I_ex / p['tau_syn_ex'])
        ddI_in = -dI_in / p['tau_syn_in']
        dI_in_dt = dI_in - (I_in / p['tau_syn_in'])
        dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
        return dv, ddI_ex, dI_ex_dt, ddI_in, dI_in_dt, dw

    def update(self, x=0.0 * u.pA):
        r"""Advance the neuron state by one simulation time step.

        Performs adaptive RKF45 integration of membrane, synaptic, and adaptation dynamics
        over the interval :math:`[t, t+dt]`, with in-loop spike detection, reset, and
        refractory handling matching NEST semantics.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input at the current time step, with unit pA.
            Shape must be broadcastable to ``(*in_size, *batch_size)``.
            Default: ``0.0 * u.pA``.

            This input is stored in the one-step-delayed buffer ``I_stim`` and will be
            used in the *next* time step's dynamics (matching NEST input handling).

        Returns
        -------
        spike : Array
            Binary spike indicator with shape ``(*in_size, *batch_size)``, dtype float.
            Value is ``1.0`` where at least one spike occurred during the integration
            interval, ``0.0`` otherwise.

            Note: With ``t_ref=0``, neurons may spike multiple times within the step,
            but the returned tensor is binary per neuron per step. Internal adaptation
            dynamics accumulate all spike-triggered increments.

        Notes
        -----
        **Integration Process**

        1. **Adaptive RKF45 loop**: Starting from current state at time :math:`t`, integrate
           ODEs using RKF45 with adaptive step sizing until reaching :math:`t + dt`.

           - Each substep computes 6 stages (:math:`k_1` through :math:`k_6`)
           - Error estimate: :math:`err = \max|y_5 - y_4|`
           - Step acceptance: if :math:`err \leq atol` or :math:`h \leq h_{min}`
           - Step size update: :math:`h_{new} = h \cdot \min(5, \max(0.2, 0.9(atol/err)^{0.2}))`

        2. **In-loop spike handling**: After each accepted substep, check if
           :math:`V \geq V_{peak}` (or :math:`V \geq V_{th}` if :math:`\Delta_T=0`).
           If spike detected:

           - Reset: :math:`V \leftarrow V_{reset}`
           - Adaptation jump: :math:`w \leftarrow w + b`
           - Refractory counter: :math:`r \leftarrow \lceil t_{ref}/dt \rceil + 1` (if enabled)

        3. **Post-integration cleanup**:

           - Decrement refractory counter: :math:`r \leftarrow r - 1` (if :math:`r > 0`)
           - Deliver synaptic inputs: add spike weights to :math:`dI_{ex}` and :math:`dI_{in}`
           - Store external input: :math:`I_{stim} \leftarrow x` (for next step)
           - Update spike time: :math:`t_{spike} \leftarrow t + dt` (where spikes occurred)

        **Refractory Clamping**

        During refractory period (:math:`r > 0`):

        - Effective voltage: :math:`V_{eff} = V_{reset}`
        - Voltage derivative: :math:`dV/dt = 0`
        - All other state variables evolve normally

        **Voltage Capping**

        Outside refractory period, effective voltage is capped to prevent exponential
        overflow: :math:`V_{eff} = \min(V, V_{peak})`.

        **Numerical Stability**

        - Raises ``ValueError`` if :math:`V < -1000` mV (indicates divergence)
        - Raises ``ValueError`` if :math:`|w| > 10^6` pA (adaptation overflow)
        - Maximum iteration limit: 100,000 substeps per time step

        **Performance Considerations**

        The integration loop processes each neuron independently using pure NumPy operations.
        For large populations, this can be slow compared to vectorized solvers, but it
        ensures exact NEST compatibility including adaptive step sizing and in-loop spike
        handling.

        Examples
        --------
        Single step update:

        .. code-block:: python

            >>> import brainpy.state as bst
            >>> import brainunit as u
            >>> import brainstate as bs
            >>>
            >>> neuron = bst.aeif_psc_alpha(10)
            >>> with bs.environ.context(dt=0.1 * u.ms):
            ...     neuron.init_all_states()
            ...     spike = neuron.update(x=400 * u.pA)
            >>> spike.shape
            (10,)

        Multi-step simulation:

        .. code-block:: python

            >>> import jax.numpy as jnp
            >>>
            >>> neuron = bst.aeif_psc_alpha(5)
            >>> n_steps = 100
            >>> spike_train = []
            >>>
            >>> with bs.environ.context(dt=0.1 * u.ms):
            ...     neuron.init_all_states()
            ...     for i in range(n_steps):
            ...         # Inject step current
            ...         current = 500 * u.pA if i > 20 else 0 * u.pA
            ...         spike = neuron.update(x=current)
            ...         spike_train.append(spike)
            >>>
            >>> spike_train = jnp.array(spike_train)  # shape: (100, 5)
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        dI_ex = self._broadcast_to_state(np.asarray(self.dI_ex.value, dtype=np.float64), v_shape)
        I_ex = self._broadcast_to_state(self._to_numpy(self.I_ex.value, u.pA), v_shape)
        dI_in = self._broadcast_to_state(np.asarray(self.dI_in.value, dtype=np.float64), v_shape)
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
        i0_ex = self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        i0_in = self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        spike_mask = np.zeros(v_shape, dtype=bool)
        V_next = np.empty_like(V)
        dI_ex_next = np.empty_like(dI_ex)
        I_ex_next = np.empty_like(I_ex)
        dI_in_next = np.empty_like(dI_in)
        I_in_next = np.empty_like(I_in)
        w_next = np.empty_like(w)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            y = np.asarray([V[idx], dI_ex[idx], I_ex[idx], dI_in[idx], I_in[idx], w[idx]], dtype=np.float64)
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
                            y_[0], y_[1], y_[2], y_[3], y_[4], y_[5], is_refractory, i_stim[idx], local_p
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
                        raise ValueError('Numerical instability in aeif_psc_alpha dynamics.')

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

            y[1] += i0_ex[idx] * w_ex[idx]
            y[3] += i0_in[idx] * w_in[idx]

            spike_mask[idx] = local_spike
            V_next[idx] = y[0]
            dI_ex_next[idx] = y[1]
            I_ex_next[idx] = y[2]
            dI_in_next[idx] = y[3]
            I_in_next[idx] = y[4]
            w_next[idx] = y[5]
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.dI_ex.value = dI_ex_next
        self.I_ex.value = I_ex_next * u.pA
        self.dI_in.value = dI_in_next
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
