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
from ._utils import is_tracer, validate_aeif_overflow, AdaptiveRungeKuttaStep

__all__ = [
    'aeif_psc_alpha',
]


class aeif_psc_alpha(NESTNeuron):
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
        Membrane potential, shape ``(*in_size,)`` with unit mV.
    dI_ex : brainstate.ShortTermState
        Excitatory alpha auxiliary state (derivative of :math:`I_{ex}`), unit pA/ms.
    I_ex : brainstate.HiddenState
        Excitatory synaptic current, unit pA.
    dI_in : brainstate.ShortTermState
        Inhibitory alpha auxiliary state (derivative of :math:`I_{in}`), unit pA/ms.
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
        >>> import saiunit as u
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
        r"""Initialize all state variables.

        Creates and initializes the membrane potential, synaptic currents, adaptation
        current, refractory counters, integration step size, and stimulus buffer.

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

        I_ex = braintools.init.param(self.I_ex_initializer, self.varshape)
        I_in = braintools.init.param(self.I_in_initializer, self.varshape)
        V = braintools.init.param(self.V_initializer, self.varshape)
        zeros = u.math.zeros(self.varshape, dtype=V.dtype) * (u.pA / u.ms)
        w = braintools.init.param(self.w_initializer, self.varshape)

        self.dI_ex = brainstate.ShortTermState(zeros)
        self.dI_in = brainstate.ShortTermState(zeros)
        self.I_ex = brainstate.HiddenState(I_ex)
        self.I_in = brainstate.HiddenState(I_in)
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
        r"""Compute differentiable spike output using surrogate gradient.

        Applies the surrogate spike function to the scaled membrane potential for
        gradient-based learning. This method is used for backpropagation and does
        not affect the internal spike detection logic (which uses hard threshold
        crossing during integration).

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential array with unit mV. If ``None``, uses the current
            ``self.V.value``. Shape: ``(*in_size,)``.

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

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, dI_ex, I_ex, dI_in, I_in, w -- ODE state variables.
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
                     - state.I_ex - state.I_in - state.w + self.I_e + extra.i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        ddI_ex = -state.dI_ex / self.tau_syn_ex
        dI_ex_dt = state.dI_ex - state.I_ex / self.tau_syn_ex
        ddI_in = -state.dI_in / self.tau_syn_in
        dI_in_dt = state.dI_in - state.I_in / self.tau_syn_in
        dw = (self.a * (v_eff - self.E_L) - state.w) / self.tau_w

        return DotDict(V=dV, dI_ex=ddI_ex, I_ex=dI_ex_dt, dI_in=ddI_in, I_in=dI_in_dt, w=dw)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, dI_ex, I_ex, dI_in, I_in, w -- ODE state variables.
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
        r"""Advance the neuron state by one simulation time step.

        Performs adaptive RKF45 integration of membrane, synaptic, and adaptation dynamics
        over the interval :math:`[t, t+dt]`, with in-loop spike detection, reset, and
        refractory handling matching NEST semantics.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input at the current time step, with unit pA.
            Shape must be broadcastable to ``(*in_size,)``.
            Default: ``0.0 * u.pA``.

            This input is stored in the one-step-delayed buffer ``I_stim`` and will be
            used in the *next* time step's dynamics (matching NEST input handling).

        Returns
        -------
        spike : Array
            Binary spike indicator with shape ``(*in_size,)``, dtype float.
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
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        dI_ex = self.dI_ex.value  # pA/ms
        I_ex = self.I_ex.value  # pA
        dI_in = self.dI_in.value  # pA/ms
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
        ode_state = DotDict(V=V, dI_ex=dI_ex, I_ex=I_ex, dI_in=dI_in, I_in=I_in, w=w)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            v_peak_detect=v_peak_detect,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, dI_ex, I_ex = ode_state.V, ode_state.dI_ex, ode_state.I_ex
        dI_in, I_in, w = ode_state.dI_in, ode_state.I_in, ode_state.w
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in aeif_psc_alpha dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration).
        w_ex = self.sum_delta_inputs(u.math.zeros_like(self.I_ex.value), label='w_ex')
        w_in = self.sum_delta_inputs(u.math.zeros_like(self.I_in.value), label='w_in')
        pscon_ex = np.e / self.tau_syn_ex  # 1/ms
        pscon_in = np.e / self.tau_syn_in  # 1/ms

        # Apply synaptic spike inputs.
        dI_ex = dI_ex + pscon_ex * w_ex  # pA/ms + 1/ms * pA = pA/ms
        dI_in = dI_in + pscon_in * w_in  # pA/ms + 1/ms * pA = pA/ms

        # Write back state.
        self.V.value = V
        self.dI_ex.value = dI_ex
        self.I_ex.value = I_ex
        self.dI_in.value = dI_in
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
