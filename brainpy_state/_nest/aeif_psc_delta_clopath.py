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

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

__all__ = [
    'aeif_psc_delta_clopath',
]


class aeif_psc_delta_clopath(NESTNeuron):
    r"""Adaptive exponential integrate-and-fire neuron with delta-shaped synaptic input and Clopath voltage traces.

    This model extends the standard adaptive exponential integrate-and-fire (AdEx) neuron with additional
    state variables required for voltage-based Clopath plasticity. It implements delta-function postsynaptic
    currents (instantaneous voltage jumps), spike afterpotential dynamics, adaptive threshold, post-spike
    voltage clamping, and three low-pass filtered voltage traces (``u_bar_plus``, ``u_bar_minus``, ``u_bar_bar``)
    used by the Clopath learning rule.

    **1. Membrane and Adaptation Dynamics**

    The subthreshold membrane potential evolves according to:

    .. math::

       C_m \frac{dV}{dt} = -g_L (V - E_L) + g_L \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)
                           - w + z + I_e + I_{stim},

    where :math:`V` is the membrane potential, :math:`w` is the adaptation current, :math:`z` is the spike
    afterpotential current, :math:`V_{th}` is the adaptive threshold, :math:`I_e` is constant external current,
    and :math:`I_{stim}` is the one-step delayed synaptic input. The exponential term provides the spike
    upstroke when :math:`\Delta_T > 0`.

    Three auxiliary currents evolve as:

    .. math::

       \tau_w \frac{dw}{dt} &= a (V - E_L) - w, \\
       \tau_z \frac{dz}{dt} &= -z, \\
       \tau_{V_{th}} \frac{dV_{th}}{dt} &= -(V_{th} - V_{th,rest}).

    The adaptation current :math:`w` provides subthreshold coupling and spike-frequency adaptation.
    The afterpotential :math:`z` creates a depolarizing transient following each spike.
    The adaptive threshold :math:`V_{th}` relaxes toward :math:`V_{th,rest}` between spikes and jumps
    to :math:`V_{th,max}` upon spike emission.

    **2. Clopath Low-Pass Voltage Traces**

    Three filtered voltage variables are maintained for plasticity:

    .. math::

       \tau_{u+} \frac{du_{bar+}}{dt} &= -u_{bar+} + V, \\
       \tau_{u-} \frac{du_{bar-}}{dt} &= -u_{bar-} + V, \\
       \tau_{u\bar{}} \frac{du_{bar\bar{}}}{dt} &= -u_{bar\bar{}} + u_{bar-}.

    These traces are first-order low-pass filters of the membrane voltage with different time constants.
    ``u_bar_plus`` and ``u_bar_minus`` filter :math:`V` directly; ``u_bar_bar`` is a second-order filter
    (filters ``u_bar_minus``). Delayed versions (delayed by ``delay_u_bars``) are stored in ring buffers
    for use by Clopath synaptic plasticity rules.

    **3. Delta-Function Synaptic Input**

    Incoming synaptic spikes cause instantaneous voltage jumps:

    .. math::

       V \leftarrow V + \sum_k J_k \delta(t - t_k^{\mathrm{spike}}),

    where :math:`J_k` is the synaptic weight from presynaptic neuron :math:`k`. Delta inputs are summed
    from the ``delta_inputs`` dictionary and applied at the beginning of each accepted RKF45 substep
    (but only when the neuron is neither refractory nor clamped).

    **4. Spike Detection and Reset**

    Spike detection threshold depends on :math:`\Delta_T`:

    - If :math:`\Delta_T > 0`: threshold is ``V_peak`` (exponential blowup detector).
    - If :math:`\Delta_T = 0`: threshold is the dynamic ``V_th`` (standard IF threshold).

    Upon threshold crossing, the following spike-triggered updates occur:

    .. math::

       V &\leftarrow V_{\mathrm{clamp}}, \\
       w &\leftarrow w + b, \\
       z &\leftarrow I_{sp}, \\
       V_{th} &\leftarrow V_{th,max}, \\
       \text{clamp\_step\_count} &\leftarrow \lceil t_{\mathrm{clamp}} / dt \rceil + 1.

    **5. Post-Spike Clamping and Refractory Period**

    The model implements a two-stage reset:

    1. **Clamping stage** (duration ``t_clamp``): voltage is held at ``V_clamp``, and adaptation dynamics
       are frozen (``dw/dt = 0``). At the end of clamping (when ``clamp_step_count`` reaches 1 during
       substep integration), voltage is reset to ``V_reset`` and the refractory period begins.

    2. **Refractory stage** (duration ``t_ref``): voltage is clamped to ``V_reset``, but adaptation
       dynamics continue (``dw/dt != 0``). Spike detection is disabled during both clamping and refractory.

    This two-stage mechanism reproduces NEST's spike handling order and allows modeling of realistic
    action potential waveforms with controlled overshoot.

    **6. Numerical Integration**

    The continuous-time dynamics are integrated using an adaptive Runge-Kutta-Fehlberg 4(5) solver (RKF45)
    with local error control. The integrator maintains a persistent step size (``integration_step``) that
    adapts based on local truncation error estimates. During refractory or clamping, the effective voltage
    used in the right-hand side is replaced with ``V_reset`` or ``V_clamp``, but state integration continues.

    Parameters
    ----------
    in_size : int or tuple of int
        Population shape. Scalar for 1D populations, tuple for multi-dimensional arrays.
    V_peak : ArrayLike, default: 33.0 * u.mV
        Spike detection threshold when ``Delta_T > 0``. Must satisfy ``V_peak > V_th_rest``.
        Shape: scalar or broadcastable to ``in_size``.
    V_reset : ArrayLike, default: -60.0 * u.mV
        Reset potential after clamping ends. Must satisfy ``V_reset < V_peak``.
        Shape: scalar or broadcastable to ``in_size``.
    t_ref : ArrayLike, default: 0.0 * u.ms
        Absolute refractory period duration (non-negative). Default of 0 ms matches NEST defaults.
        Shape: scalar or broadcastable to ``in_size``.
    g_L : ArrayLike, default: 30.0 * u.nS
        Leak conductance (must be positive). Shape: scalar or broadcastable to ``in_size``.
    C_m : ArrayLike, default: 281.0 * u.pF
        Membrane capacitance (must be positive). Shape: scalar or broadcastable to ``in_size``.
    E_L : ArrayLike, default: -70.6 * u.mV
        Leak reversal potential. Shape: scalar or broadcastable to ``in_size``.
    Delta_T : ArrayLike, default: 2.0 * u.mV
        Exponential slope factor (non-negative). Set to 0 for non-exponential IF model.
        Shape: scalar or broadcastable to ``in_size``.
    tau_w : ArrayLike, default: 144.0 * u.ms
        Adaptation current time constant (must be positive). Shape: scalar or broadcastable to ``in_size``.
    tau_z : ArrayLike, default: 40.0 * u.ms
        Spike afterpotential time constant (must be positive). Shape: scalar or broadcastable to ``in_size``.
    tau_V_th : ArrayLike, default: 50.0 * u.ms
        Adaptive threshold time constant (must be positive). Shape: scalar or broadcastable to ``in_size``.
    V_th_max : ArrayLike, default: 30.4 * u.mV
        Threshold value immediately after spike. Must satisfy ``V_th_max >= V_th_rest``.
        Shape: scalar or broadcastable to ``in_size``.
    V_th_rest : ArrayLike, default: -50.4 * u.mV
        Resting threshold value (asymptotic value between spikes). Must satisfy ``V_th_rest <= V_peak``.
        Shape: scalar or broadcastable to ``in_size``.
    tau_u_bar_plus : ArrayLike, default: 7.0 * u.ms
        Time constant for ``u_bar_plus`` trace (must be positive). Shape: scalar or broadcastable to ``in_size``.
    tau_u_bar_minus : ArrayLike, default: 10.0 * u.ms
        Time constant for ``u_bar_minus`` trace (must be positive). Shape: scalar or broadcastable to ``in_size``.
    tau_u_bar_bar : ArrayLike, default: 500.0 * u.ms
        Time constant for ``u_bar_bar`` trace (must be positive). Shape: scalar or broadcastable to ``in_size``.
    a : ArrayLike, default: 4.0 * u.nS
        Subthreshold adaptation coupling strength. Shape: scalar or broadcastable to ``in_size``.
    b : ArrayLike, default: 80.5 * u.pA
        Spike-triggered adaptation increment. Shape: scalar or broadcastable to ``in_size``.
    I_sp : ArrayLike, default: 400.0 * u.pA
        Spike afterpotential current reset value (sets ``z`` on spike). Shape: scalar or broadcastable to ``in_size``.
    I_e : ArrayLike, default: 0.0 * u.pA
        Constant external current. Shape: scalar or broadcastable to ``in_size``.
    A_LTD : ArrayLike, default: 1.4e-4
        Clopath depression amplitude (dimensionless). Used in delayed-buffer bookkeeping for compatibility.
        Shape: scalar or broadcastable to ``in_size``.
    A_LTP : ArrayLike, default: 8.0e-5
        Clopath potentiation amplitude (dimensionless). Used in delayed-buffer bookkeeping for compatibility.
        Shape: scalar or broadcastable to ``in_size``.
    theta_plus : ArrayLike, default: -45.3 * u.mV
        Clopath potentiation voltage threshold. Shape: scalar or broadcastable to ``in_size``.
    theta_minus : ArrayLike, default: -70.6 * u.mV
        Clopath depression voltage threshold. Shape: scalar or broadcastable to ``in_size``.
    A_LTD_const : bool, default: True
        If True, LTD amplitude is constant. If False, LTD scales with ``u_bar_bar**2 / u_ref_squared`` (homeostatic).
    delay_u_bars : ArrayLike, default: 5.0 * u.ms
        Delay for Clopath u-bar traces (ring buffer delay). Rounded to nearest integer multiple of ``dt``.
        Shape: scalar or broadcastable to ``in_size``.
    u_ref_squared : ArrayLike, default: 60.0
        Clopath LTD homeostatic reference (dimensionless, must be positive). Only used when ``A_LTD_const=False``.
        Shape: scalar or broadcastable to ``in_size``.
    gsl_error_tol : ArrayLike, default: 1e-6
        RKF45 local error tolerance (must be positive). Smaller values increase accuracy and decrease step size.
        Shape: scalar or broadcastable to ``in_size``.
    t_clamp : ArrayLike, default: 2.0 * u.ms
        Spike clamping duration (non-negative). Shape: scalar or broadcastable to ``in_size``.
    V_clamp : ArrayLike, default: 33.0 * u.mV
        Clamped voltage immediately after spike. Shape: scalar or broadcastable to ``in_size``.
    V_initializer : Callable, default: braintools.init.Constant(-70.6 * u.mV)
        Initializer for membrane potential. Must return values with voltage units.
    w_initializer : Callable, default: braintools.init.Constant(0.0 * u.pA)
        Initializer for adaptation current. Must return values with current units.
    z_initializer : Callable, default: braintools.init.Constant(0.0 * u.pA)
        Initializer for spike afterpotential current. Must return values with current units.
    V_th_initializer : Callable, default: braintools.init.Constant(-50.4 * u.mV)
        Initializer for adaptive threshold. Must return values with voltage units.
    u_bar_plus_initializer : Callable, default: braintools.init.Constant(-70.6 * u.mV)
        Initializer for ``u_bar_plus`` trace. Must return values with voltage units.
    u_bar_minus_initializer : Callable, default: braintools.init.Constant(-70.6 * u.mV)
        Initializer for ``u_bar_minus`` trace. Must return values with voltage units.
    u_bar_bar_initializer : Callable, default: braintools.init.Constant(-70.6 * u.mV)
        Initializer for ``u_bar_bar`` trace. Must return values with voltage units.
    spk_fun : Callable, default: braintools.surrogate.ReluGrad()
        Surrogate gradient function for differentiable spike generation during training.
    spk_reset : str, default: 'hard'
        Spike reset mode. 'hard' (stop_gradient) matches NEST behavior; 'soft' (V -= V_th) preserves gradients.
    ref_var : bool, default: False
        If True, expose ``refractory`` state variable indicating whether neuron is refractory or clamped.
    name : str, optional
        Name for this neuron instance.

    Parameter Mapping
    -----------------
    The table below maps BrainPy parameters to their mathematical symbols and NEST equivalents:

    ========================== ================== ================================ ================================================================
    **Parameter**              **Default**        **Math Symbol**                  **Description**
    ========================== ================== ================================ ================================================================
    ``in_size``                (required)         —                                Population shape
    ``V_peak``                 33 mV              :math:`V_\mathrm{peak}`          Spike detection threshold for :math:`\Delta_T > 0`
    ``V_reset``                -60 mV             :math:`V_\mathrm{reset}`         Reset potential
    ``t_ref``                  0 ms               :math:`t_\mathrm{ref}`           Absolute refractory duration
    ``g_L``                    30 nS              :math:`g_\mathrm{L}`             Leak conductance
    ``C_m``                    281 pF             :math:`C_\mathrm{m}`             Membrane capacitance
    ``E_L``                    -70.6 mV           :math:`E_\mathrm{L}`             Leak reversal potential
    ``Delta_T``                2 mV               :math:`\Delta_T`                 Exponential slope factor
    ``tau_w``                  144 ms             :math:`\tau_w`                   Adaptation time constant
    ``tau_z``                  40 ms              :math:`\tau_z`                   Spike afterpotential time constant
    ``tau_V_th``               50 ms              :math:`\tau_{V_{th}}`            Adaptive threshold time constant
    ``V_th_max``               30.4 mV            :math:`V_{th,\mathrm{max}}`      Threshold value immediately after spike
    ``V_th_rest``              -50.4 mV           :math:`V_{th,\mathrm{rest}}`     Resting threshold value
    ``tau_u_bar_plus``         7 ms               :math:`\tau_{u+}`                Time constant of ``u_bar_plus``
    ``tau_u_bar_minus``        10 ms              :math:`\tau_{u-}`                Time constant of ``u_bar_minus``
    ``tau_u_bar_bar``          500 ms             :math:`\tau_{u\bar{}}`           Time constant of ``u_bar_bar``
    ``a``                      4 nS               :math:`a`                        Subthreshold adaptation strength
    ``b``                      80.5 pA            :math:`b`                        Spike-triggered adaptation increment
    ``I_sp``                   400 pA             :math:`I_{sp}`                   Spike afterpotential current reset value
    ``I_e``                    0 pA               :math:`I_\mathrm{e}`             Constant external current
    ``A_LTD``                  1.4e-4             :math:`A_\mathrm{LTD}`           Clopath depression amplitude
    ``A_LTP``                  8.0e-5             :math:`A_\mathrm{LTP}`           Clopath potentiation amplitude
    ``theta_plus``             -45.3 mV           :math:`\theta_+`                 Clopath potentiation threshold
    ``theta_minus``            -70.6 mV           :math:`\theta_-`                 Clopath depression threshold
    ``A_LTD_const``            ``True``           —                                If False, homeostatic LTD scaling
    ``delay_u_bars``           5 ms               —                                Delay for Clopath u-bar buffers
    ``u_ref_squared``          60                 :math:`u_\mathrm{ref}^2`         Clopath LTD homeostatic reference
    ``gsl_error_tol``          1e-6               —                                RKF45 local error tolerance
    ``t_clamp``                2 ms               :math:`t_\mathrm{clamp}`         Spike clamping duration
    ``V_clamp``                33 mV              :math:`V_\mathrm{clamp}`         Clamped voltage after spike
    ``V_initializer``          Constant(E_L)      —                                Membrane voltage initializer
    ``w_initializer``          Constant(0 pA)     —                                Adaptation current initializer
    ``z_initializer``          Constant(0 pA)     —                                Spike afterpotential initializer
    ``V_th_initializer``       Constant(-50.4 mV) —                                Adaptive threshold initializer
    ``u_bar_plus_initializer`` Constant(-70.6 mV) —                                ``u_bar_plus`` initializer
    `u_bar_minus_initializer`  Constant(-70.6 mV) —                                ``u_bar_minus`` initializer
    ``u_bar_bar_initializer``  Constant(-70.6 mV) —                                ``u_bar_bar`` initializer
    ``spk_fun``                ReluGrad()         —                                Surrogate spike function
    ``spk_reset``              ``'hard'``         —                                Reset mode (``'hard'`` or ``'soft'``)
    ``ref_var``                ``False``          —                                If True, expose refractory indicator
    ========================== ================== ================================ ================================================================

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential (mV). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    w : brainstate.HiddenState
        Adaptation current (pA). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    z : brainstate.HiddenState
        Spike afterpotential current (pA). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    V_th : brainstate.HiddenState
        Adaptive threshold (mV). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    u_bar_plus : brainstate.HiddenState
        Clopath low-pass filtered voltage trace (mV). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    u_bar_minus : brainstate.HiddenState
        Clopath low-pass filtered voltage trace (mV). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    u_bar_bar : brainstate.HiddenState
        Clopath second-order filtered voltage trace (mV). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory time steps (int32). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    clamp_step_count : brainstate.ShortTermState
        Remaining clamping time steps (int32). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    integration_step : brainstate.ShortTermState
        Current RKF45 adaptive step size (ms). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    I_stim : brainstate.ShortTermState
        One-step delayed synaptic current (pA). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    delayed_u_bar_plus_buffer : brainstate.ShortTermState
        Ring buffer for delayed ``u_bar_plus`` (mV). Shape: ``(delay_steps, *in_size)`` or ``(delay_steps, *in_size, batch_size)``.
    delayed_u_bar_minus_buffer : brainstate.ShortTermState
        Ring buffer for delayed ``u_bar_minus`` (mV). Shape: ``(delay_steps, *in_size)`` or ``(delay_steps, *in_size, batch_size)``.
    delayed_u_bars_idx : brainstate.ShortTermState
        Current ring buffer write index (int32). Scalar.
    delayed_u_bars_steps : brainstate.ShortTermState
        Total ring buffer size (int32). Scalar.
    last_spike_time : brainstate.ShortTermState
        Last spike time (ms). Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.
    refractory : brainstate.ShortTermState, optional
        Boolean indicator: True if neuron is refractory or clamped. Only present if ``ref_var=True``.
        Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.

    Raises
    ------
    ValueError

        - If ``V_reset >= V_peak``.
        - If ``Delta_T < 0``.
        - If ``V_th_max < V_th_rest`` or ``V_peak < V_th_rest``.
        - If ``C_m <= 0``, ``t_ref < 0``, ``t_clamp < 0``, or any time constant <= 0.
        - If ``u_ref_squared <= 0`` or ``gsl_error_tol <= 0``.
        - If ``(V_peak - V_th_rest) / Delta_T`` exceeds overflow limit (when ``Delta_T > 0``).
        - If ``delay_u_bars`` maps to fewer than 1 delay buffer entry.
        - If ``delay_u_bars`` is spatially heterogeneous (delay steps must be uniform).
        - If numerical instability is detected during integration (voltage or adaptation out of bounds).

    Notes
    -----
    **Implementation Details:**

    - **RKF45 integration:** Uses adaptive-step Runge-Kutta-Fehlberg 4(5) with error control. Step size
      is persisted across time steps to improve stability. Minimum step size is 1e-8 ms; maximum iteration
      count is 100000 per ``dt`` to prevent infinite loops.

    - **Refractory/clamping precedence:** During integration, if ``clamp_step_count > 0``, voltage is clamped
      to ``V_clamp`` and adaptation dynamics freeze. If ``refractory_step_count > 0`` (and not clamped),
      voltage is clamped to ``V_reset`` but adaptation continues. Both conditions disable spike detection.

    - **Delta input timing:** Delta voltage jumps are applied at the start of each accepted substep, but
      only when the neuron is neither refractory nor clamped. This matches NEST's per-substep spike delivery.

    - **Spike timing convention:** ``last_spike_time`` is set to ``t + dt`` upon spike emission (end of
      current time step), matching NEST's convention.

    - **Clopath buffer bookkeeping:** This implementation maintains delayed ``u_bar_plus`` and ``u_bar_minus``
      buffers even without a dedicated Clopath synapse model, ensuring state-level compatibility with NEST
      for future plasticity extensions. The delayed traces are updated at the end of each ``update()`` call.

    - **Overflow protection:** The exponential term is guarded against overflow when ``Delta_T > 0``. If
      ``(V_peak - V_th_rest) / Delta_T`` would cause ``exp(...)`` to exceed ``max(float64) / 1e20``, an
      error is raised during initialization.

    - **Batch support:** All state variables support an optional batch dimension (via ``batch_size`` in
      ``init_state``/``reset_state``). Batched states have shape ``(*in_size, batch_size)``.

    **Usage:**

    This model is designed for voltage-based plasticity studies and detailed spike waveform modeling.
    Use ``Delta_T > 0`` for exponential IF dynamics (rapid spike upstroke) or ``Delta_T = 0`` for standard
    IF with dynamic threshold. The ``t_clamp`` and ``V_clamp`` parameters control the spike overshoot and
    allow modeling realistic action potential shapes. For basic AdEx simulations without Clopath plasticity,
    consider using the simpler ``aeif_psc_delta`` or ``aeif_psc_exp`` models (if available).

    See Also
    --------
    aeif_psc_delta : Simplified AdEx without Clopath traces or clamping.
    aeif_psc_exp : AdEx with exponential postsynaptic currents.
    clopath_synapse : Voltage-based STDP synapse (NEST reference).

    References
    ----------
    .. [1] Clopath C, Büsing L, Vasilaki E, Gerstner W (2010). Connectivity reflects coding: a model of
           voltage-based STDP with homeostasis. *Nature Neuroscience*, 13(3):344-352.
           DOI: https://doi.org/10.1038/nn.2479
    .. [2] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire model as an effective
           description of neuronal activity. *Journal of Neurophysiology*, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [3] NEST Simulator documentation: ``aeif_psc_delta_clopath`` model.
           https://nest-simulator.readthedocs.io/
    .. [4] NEST source code: ``models/aeif_psc_delta_clopath.h`` and ``models/aeif_psc_delta_clopath.cpp``.

    Examples
    --------
    Simulate a single neuron with step current input:

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import brainstate as bs
       >>> import brainunit as u
       >>> import matplotlib.pyplot as plt
       >>>
       >>> # Create neuron population
       >>> neuron = bst.aeif_psc_delta_clopath(in_size=1, I_e=300*u.pA)
       >>>
       >>> # Simulate for 100 ms
       >>> with bs.environ.context(dt=0.1*u.ms):
       ...     neuron.init_state()
       ...     times, voltages = [], []
       ...     for t in range(1000):
       ...         spike = neuron.update()
       ...         times.append(t * 0.1)
       ...         voltages.append(float(neuron.V.value / u.mV))
       >>>
       >>> # Plot membrane potential
       >>> plt.plot(times, voltages)
       >>> plt.xlabel('Time (ms)')
       >>> plt.ylabel('Voltage (mV)')
       >>> plt.show()

    Network simulation with delta-function synaptic connections:

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import brainstate as bs
       >>> import brainunit as u
       >>>
       >>> # Create excitatory and inhibitory populations
       >>> exc = bst.aeif_psc_delta_clopath(in_size=100, I_e=200*u.pA)
       >>> inh = bst.aeif_psc_delta_clopath(in_size=25, I_e=150*u.pA)
       >>>
       >>> # Create delta-function projection (instantaneous voltage jump)
       >>> # Note: Requires appropriate projection class that adds delta inputs
       >>> # exc_to_inh = bst.DeltaProj(exc, inh, weight=0.5*u.mV, prob=0.1)
       >>>
       >>> # Simulate network
       >>> with bs.environ.context(dt=0.1*u.ms):
       ...     exc.init_state()
       ...     inh.init_state()
       ...     for t in range(10000):  # 1 second
       ...         exc_spikes = exc.update()
       ...         inh_spikes = inh.update()
    """

    __module__ = 'brainpy.state'

    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        V_peak: ArrayLike = 33.0 * u.mV,
        V_reset: ArrayLike = -60.0 * u.mV,
        t_ref: ArrayLike = 0.0 * u.ms,
        g_L: ArrayLike = 30.0 * u.nS,
        C_m: ArrayLike = 281.0 * u.pF,
        E_L: ArrayLike = -70.6 * u.mV,
        Delta_T: ArrayLike = 2.0 * u.mV,
        tau_w: ArrayLike = 144.0 * u.ms,
        tau_z: ArrayLike = 40.0 * u.ms,
        tau_V_th: ArrayLike = 50.0 * u.ms,
        V_th_max: ArrayLike = 30.4 * u.mV,
        V_th_rest: ArrayLike = -50.4 * u.mV,
        tau_u_bar_plus: ArrayLike = 7.0 * u.ms,
        tau_u_bar_minus: ArrayLike = 10.0 * u.ms,
        tau_u_bar_bar: ArrayLike = 500.0 * u.ms,
        a: ArrayLike = 4.0 * u.nS,
        b: ArrayLike = 80.5 * u.pA,
        I_sp: ArrayLike = 400.0 * u.pA,
        I_e: ArrayLike = 0.0 * u.pA,
        A_LTD: ArrayLike = 14.0e-5,
        A_LTP: ArrayLike = 8.0e-5,
        theta_plus: ArrayLike = -45.3 * u.mV,
        theta_minus: ArrayLike = -70.6 * u.mV,
        A_LTD_const: bool = True,
        delay_u_bars: ArrayLike = 5.0 * u.ms,
        u_ref_squared: ArrayLike = 60.0,
        gsl_error_tol: ArrayLike = 1e-6,
        t_clamp: ArrayLike = 2.0 * u.ms,
        V_clamp: ArrayLike = 33.0 * u.mV,
        V_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        w_initializer: Callable = braintools.init.Constant(0.0 * u.pA),
        z_initializer: Callable = braintools.init.Constant(0.0 * u.pA),
        V_th_initializer: Callable = braintools.init.Constant(-50.4 * u.mV),
        u_bar_plus_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        u_bar_minus_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        u_bar_bar_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
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
        self.tau_z = braintools.init.param(tau_z, self.varshape)
        self.tau_V_th = braintools.init.param(tau_V_th, self.varshape)
        self.V_th_max = braintools.init.param(V_th_max, self.varshape)
        self.V_th_rest = braintools.init.param(V_th_rest, self.varshape)
        self.tau_u_bar_plus = braintools.init.param(tau_u_bar_plus, self.varshape)
        self.tau_u_bar_minus = braintools.init.param(tau_u_bar_minus, self.varshape)
        self.tau_u_bar_bar = braintools.init.param(tau_u_bar_bar, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.I_sp = braintools.init.param(I_sp, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        # Clopath-related parameters kept for source-level compatibility.
        self.A_LTD = braintools.init.param(A_LTD, self.varshape)
        self.A_LTP = braintools.init.param(A_LTP, self.varshape)
        self.theta_plus = braintools.init.param(theta_plus, self.varshape)
        self.theta_minus = braintools.init.param(theta_minus, self.varshape)
        self.A_LTD_const = bool(A_LTD_const)
        self.delay_u_bars = braintools.init.param(delay_u_bars, self.varshape)
        self.u_ref_squared = braintools.init.param(u_ref_squared, self.varshape)

        self.gsl_error_tol = braintools.init.param(gsl_error_tol, self.varshape)
        self.t_clamp = braintools.init.param(t_clamp, self.varshape)
        self.V_clamp = braintools.init.param(V_clamp, self.varshape)

        self.V_initializer = V_initializer
        self.w_initializer = w_initializer
        self.z_initializer = z_initializer
        self.V_th_initializer = V_th_initializer
        self.u_bar_plus_initializer = u_bar_plus_initializer
        self.u_bar_minus_initializer = u_bar_minus_initializer
        self.u_bar_bar_initializer = u_bar_bar_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _to_numpy_unitless(x):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x), dtype=dftype)

    @staticmethod
    def _to_numpy_time_ms(x):
        try:
            dftype = brainstate.environ.dftype()
            return np.asarray(u.math.asarray(x / u.ms), dtype=dftype)
        except Exception:
            return np.asarray(u.math.asarray(x), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        v_reset = self._to_numpy(self.V_reset, u.mV)
        v_peak = self._to_numpy(self.V_peak, u.mV)
        v_th_rest = self._to_numpy(self.V_th_rest, u.mV)
        v_th_max = self._to_numpy(self.V_th_max, u.mV)
        delta_t = self._to_numpy(self.Delta_T, u.mV)

        if np.any(v_reset >= v_peak):
            raise ValueError('Ensure that V_reset < V_peak .')
        if np.any(delta_t < 0.0):
            raise ValueError('Delta_T must be greater than or equal to zero.')
        if np.any(v_th_max < v_th_rest):
            raise ValueError('V_th_max >= V_th_rest required.')
        if np.any(v_peak < v_th_rest):
            raise ValueError('V_peak >= V_th_rest required.')

        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Ensure that C_m > 0')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.t_clamp, u.ms) < 0.0):
            raise ValueError('Ensure that t_clamp >= 0')

        if np.any(self._to_numpy(self.tau_w, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_z, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_V_th, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_u_bar_plus, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_u_bar_minus, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_u_bar_bar, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')

        if np.any(self._to_numpy_unitless(self.u_ref_squared) <= 0.0):
            raise ValueError('Ensure that u_ref_squared > 0')
        if np.any(self._to_numpy_unitless(self.gsl_error_tol) <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        # Mirror NEST overflow guard for exponential term at spike time.
        positive_dt = delta_t > 0.0
        if np.any(positive_dt):
            max_exp_arg = np.log(np.finfo(np.float64).max / 1e20)
            ratio = (v_peak - v_th_rest) / np.where(positive_dt, delta_t, 1.0)
            if np.any(ratio[positive_dt] >= max_exp_arg):
                raise ValueError(
                    'The current combination of V_peak, V_th_rest and Delta_T will lead to numerical overflow at '
                    'spike time; try for instance to increase Delta_T or to reduce V_peak to avoid this problem.'
                )

    def _delay_u_bars_steps(self, dt_q):
        dt_ms = float(u.math.asarray(dt_q / u.ms))
        delay_ms = self._to_numpy_time_ms(self.delay_u_bars)
        ditype = brainstate.environ.ditype()
        delay_steps = np.asarray(np.rint(delay_ms / dt_ms), dtype=ditype) + 1

        if np.any(delay_steps < 1):
            raise ValueError('delay_u_bars must map to at least one delay-buffer entry.')
        if np.any(delay_steps != delay_steps.flat[0]):
            raise ValueError(
                'delay_u_bars must map to a uniform number of delay steps across the neuron state shape.'
            )
        return int(delay_steps.flat[0])

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        ditype = brainstate.environ.ditype()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    def _clamp_counts(self):
        dt = brainstate.environ.get_dt()
        ditype = brainstate.environ.ditype()
        return u.math.asarray(u.math.ceil(self.t_clamp / dt), dtype=ditype)

    def _allocate_clopath_delay_buffers(self, state_shape, dt_q):
        delay_steps = self._delay_u_bars_steps(dt_q)
        ditype = brainstate.environ.ditype()
        self.delayed_u_bars_steps = brainstate.ShortTermState(np.asarray(delay_steps, dtype=ditype))
        self.delayed_u_bars_idx = brainstate.ShortTermState(np.asarray(0, dtype=ditype))

        buf_shape = (delay_steps,) + tuple(state_shape)
        dftype = brainstate.environ.dftype()
        self.delayed_u_bar_plus_buffer = brainstate.ShortTermState(np.zeros(buf_shape, dtype=dftype))
        self.delayed_u_bar_minus_buffer = brainstate.ShortTermState(np.zeros(buf_shape, dtype=dftype))

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Allocates and initializes all neuron state variables using the configured initializers. This includes
        membrane dynamics states (V, w, z, V_th), Clopath voltage traces (u_bar_plus, u_bar_minus, u_bar_bar),
        refractory/clamping counters, RKF45 integration state, and delayed-buffer bookkeeping for Clopath plasticity.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If provided, all states will have shape ``(*in_size, batch_size)``.
            If None, states have shape ``(*in_size,)``. Default: None.
        **kwargs
            Reserved for future extensions. Currently unused.

        Notes
        -----
        - ``last_spike_time`` is initialized to -1e7 ms (far in the past) to indicate no prior spike.
        - ``refractory_step_count`` and ``clamp_step_count`` are initialized to 0 (not refractory/clamped).
        - ``integration_step`` is initialized to the current simulation time step (``dt``).
        - Clopath delay buffers are allocated with size ``ceil(delay_u_bars / dt) + 1``.
        - If ``ref_var=True``, an additional ``refractory`` boolean state is created.

        See Also
        --------
        reset_state : Reset existing states to initial values.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        w = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        z = braintools.init.param(self.z_initializer, self.varshape, batch_size)

        v_th = braintools.init.param(self.V_th_initializer, self.varshape, batch_size)
        u_plus = braintools.init.param(self.u_bar_plus_initializer, self.varshape, batch_size)
        u_minus = braintools.init.param(self.u_bar_minus_initializer, self.varshape, batch_size)
        u_bar = braintools.init.param(self.u_bar_bar_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.w = brainstate.HiddenState(w)
        self.z = brainstate.HiddenState(z)
        self.V_th = brainstate.HiddenState(v_th)
        self.u_bar_plus = brainstate.HiddenState(u_plus)
        self.u_bar_minus = brainstate.HiddenState(u_minus)
        self.u_bar_bar = brainstate.HiddenState(u_bar)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        ditype = brainstate.environ.ditype()
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=ditype))
        self.clamp_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=ditype))

        dt = brainstate.environ.get_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        dftype = brainstate.environ.dftype()
        v_shape = np.asarray(u.math.asarray(V / u.mV), dtype=dftype).shape
        self._allocate_clopath_delay_buffers(v_shape, dt)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to initial values.

        Resets all existing state variables to their initial values using the configured initializers.
        This is useful for running multiple trials or resetting network state during training.
        Clopath delay buffers are reallocated and cleared.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If provided, all states will be reset with shape ``(*in_size, batch_size)``.
            If None, states will have shape ``(*in_size,)``. Default: None.
        **kwargs
            Reserved for future extensions. Currently unused.

        Notes
        -----
        - All dynamic states (V, w, z, V_th, u_bar_*, counters, buffers) are reset.
        - Clopath delay buffers are reallocated, clearing all history.
        - Integration step size is reset to the current simulation ``dt``.
        - If batch size changes from initialization, state shapes will be updated accordingly.

        See Also
        --------
        init_state : Initial state allocation.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        self.z.value = braintools.init.param(self.z_initializer, self.varshape, batch_size)
        self.V_th.value = braintools.init.param(self.V_th_initializer, self.varshape, batch_size)
        self.u_bar_plus.value = braintools.init.param(self.u_bar_plus_initializer, self.varshape, batch_size)
        self.u_bar_minus.value = braintools.init.param(self.u_bar_minus_initializer, self.varshape, batch_size)
        self.u_bar_bar.value = braintools.init.param(self.u_bar_bar_initializer, self.varshape, batch_size)

        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )

        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        ditype = brainstate.environ.ditype()
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=ditype)
        self.clamp_step_count.value = u.math.asarray(ref_steps, dtype=ditype)

        dt = brainstate.environ.get_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

        dftype = brainstate.environ.dftype()
        v_shape = np.asarray(u.math.asarray(self.V.value / u.mV), dtype=dftype).shape
        self._allocate_clopath_delay_buffers(v_shape, dt)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output using surrogate gradient function.

        Applies the surrogate gradient function to a scaled voltage relative to the dynamic threshold.
        This produces a continuous approximation of discrete spikes, enabling gradient-based learning.
        The scaling factor ``(v_th - V_reset)`` normalizes the voltage range for the surrogate function.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (mV). If None, uses current ``self.V.value``. Shape: ``(*in_size,)``
            or ``(*in_size, batch_size)``.

        Returns
        -------
        spike : ArrayLike
            Differentiable spike signal (dimensionless, approximately in [0, 1] for most surrogate functions).
            Shape matches input ``V``.

        Notes
        -----
        - This method is primarily used during training with surrogate gradient descent.
        - During inference with ``update()``, spikes are detected via hard threshold crossing (not this function).
        - The threshold used is the dynamic ``V_th`` (if available) or the resting ``V_th_rest`` otherwise.
        - The surrogate function is configured via the ``spk_fun`` parameter (default: ``ReluGrad``).

        See Also
        --------
        update : Hard spike detection and state integration.
        """
        V = self.V.value if V is None else V
        if hasattr(self, 'V_th'):
            v_th = self.V_th.value
        else:
            v_th = self.V_th_rest
        v_scaled = (V - v_th) / (v_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _sum_delta_inputs(self):
        delta_v = u.math.zeros_like(self.V.value)
        if self.delta_inputs is None:
            return delta_v

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)
            delta_v = delta_v + out

        return delta_v

    @staticmethod
    def _dynamics_scalar(v, w, z, v_th, u_plus, u_minus, u_bar, is_refractory, is_clamped, i_stim, p):
        if is_refractory or is_clamped:
            v_eff = p['V_clamp'] if is_clamped else p['V_reset']
        else:
            v_eff = min(v, p['V_peak_rhs'])

        i_spike = 0.0 if p['Delta_T'] == 0.0 else (
            p['g_L'] * p['Delta_T'] * math.exp((v_eff - v_th) / p['Delta_T'])
        )

        dv = 0.0 if (is_refractory or is_clamped) else (
                                                           -p['g_L'] * (v_eff - p['E_L']) + i_spike - w + z + p[
                                                           'I_e'] + i_stim
                                                       ) / p['C_m']

        # NEST sets dw/dt = 0 while clamped, but not during pure refractory.
        dw = 0.0 if is_clamped else (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
        dz = -z / p['tau_z']
        dv_th = -(v_th - p['V_th_rest']) / p['tau_V_th']

        du_plus = (-u_plus + v_eff) / p['tau_u_bar_plus']
        du_minus = (-u_minus + v_eff) / p['tau_u_bar_minus']
        du_bar = (-u_bar + u_minus) / p['tau_u_bar_bar']

        return dv, dw, dz, dv_th, du_plus, du_minus, du_bar

    def _write_clopath_history(self, V_m, u_plus, u_minus, u_bar, p):
        dftype = brainstate.environ.dftype()
        plus_buf = np.asarray(self.delayed_u_bar_plus_buffer.value, dtype=dftype)
        minus_buf = np.asarray(self.delayed_u_bar_minus_buffer.value, dtype=dftype)

        ditype = brainstate.environ.ditype()
        delay_steps = int(np.asarray(self.delayed_u_bars_steps.value, dtype=ditype))
        idx = int(np.asarray(self.delayed_u_bars_idx.value, dtype=ditype))

        plus_buf[idx] = u_plus
        minus_buf[idx] = u_minus

        idx = (idx + 1) % delay_steps

        del_u_plus = plus_buf[idx]
        del_u_minus = minus_buf[idx]

        # Keep same delayed-buffer and threshold gating behavior as NEST.
        # The resulting dw traces are used by Clopath synapses in NEST.
        if self.A_LTD_const:
            _ = np.where(
                del_u_minus > p['theta_minus'],
                p['A_LTD'] * (del_u_minus - p['theta_minus']),
                0.0,
            )
        else:
            _ = np.where(
                del_u_minus > p['theta_minus'],
                p['A_LTD'] * (u_bar * u_bar) * (del_u_minus - p['theta_minus']) / p['u_ref_squared'],
                0.0,
            )

        _ = np.where(
            (V_m > p['theta_plus']) & (del_u_plus > p['theta_minus']),
            p['A_LTP'] * (V_m - p['theta_plus']) * (del_u_plus - p['theta_minus']) * p['dt'],
            0.0,
        )

        self.delayed_u_bar_plus_buffer.value = plus_buf
        self.delayed_u_bar_minus_buffer.value = minus_buf
        self.delayed_u_bars_idx.value = np.asarray(idx, dtype=ditype)

    def update(self, x=0.0 * u.pA):
        r"""Advance neuron state by one time step using adaptive RKF45 integration.

        Integrates the neuron dynamics over the current simulation time step ``dt`` using an adaptive
        Runge-Kutta-Fehlberg 4(5) solver with local error control. Handles spike detection, post-spike
        reset, refractory period, voltage clamping, delta-function synaptic inputs, and Clopath trace
        updates. Returns binary spike output for the current time step.

        Parameters
        ----------
        x : ArrayLike, default: 0.0 * u.pA
            External input current for the current time step (pA). This is combined with synaptic currents
            from ``current_inputs`` dictionary. Shape: scalar or broadcastable to ``(*in_size,)`` or
            ``(*in_size, batch_size)``.

        Returns
        -------
        spike : ArrayLike
            Binary spike indicator (1.0 if neuron spiked during this time step, 0.0 otherwise).
            Shape: ``(*in_size,)`` or ``(*in_size, batch_size)``.

        Raises
        ------
        ValueError
            If numerical instability is detected (voltage < -1000 mV or abs(adaptation) > 1e6 pA).

        Notes
        -----
        **Integration algorithm:**

        - Uses Runge-Kutta-Fehlberg 4(5) with adaptive step size control.
        - Each time step ``dt`` is subdivided into substeps with sizes determined by local error estimates.
        - Step size is adjusted based on error ratio: ``h_new = h * min(5, max(0.2, 0.9 * (tol/err)^(1/5)))``.
        - Minimum substep size: 1e-8 ms. Maximum iterations per ``dt``: 100000.
        - Step size is persisted across time steps (``integration_step``) for stability.

        **Update order within each accepted substep:**

        1. Apply delta voltage jumps (from ``delta_inputs``) if not refractory/clamped.
        2. Integrate ODEs using RKF45.
        3. Check for threshold crossing (``V >= V_peak`` or ``V >= V_th``).
        4. On spike: apply spike-triggered updates (V, w, z, V_th, clamp counter).
        5. On clamp expiry: transition to refractory (set V to V_reset, start refractory counter).
        6. During refractory: clamp voltage to V_reset.

        **Update order at the end of full ``dt``:**

        1. Write Clopath delayed-buffer bookkeeping (update ring buffers with current u_bar traces).
        2. Decrement clamp counter (if > 0).
        3. Decrement refractory counter (if > 0).
        4. Store new input current in delayed buffer (``I_stim``), to be used in next time step.
        5. Update ``last_spike_time`` for neurons that spiked.
        6. If ``ref_var=True``, update ``refractory`` indicator.

        **Input handling:**

        - Current-based inputs: summed from ``current_inputs`` dictionary and added to external ``x``.
        - Delta-based inputs: summed from ``delta_inputs`` dictionary as instantaneous voltage jumps.
        - The combined current input is delayed by one time step (``I_stim``), matching NEST's behavior.

        **Spike timing:**

        - Spikes are detected when voltage crosses threshold within a substep.
        - ``last_spike_time`` is set to ``t + dt`` (end of current time step), not the exact crossing time.
        - Multiple spikes within a single ``dt`` are possible if parameters allow (e.g., very short ``t_clamp``).

        **Computational cost:**

        - Integration is performed element-wise using Python loops over ``np.ndindex(in_size)``.
        - For large populations, this implementation may be slow (O(N) Python-level iterations).
        - Batch dimension does NOT increase cost (iteration is over spatial dimensions only).

        See Also
        --------
        init_state : Initialize state variables before first update.
        reset_state : Reset states between trials.
        get_spike : Differentiable spike output for training.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        w = self._broadcast_to_state(self._to_numpy(self.w.value, u.pA), v_shape)
        z = self._broadcast_to_state(self._to_numpy(self.z.value, u.pA), v_shape)
        v_th = self._broadcast_to_state(self._to_numpy(self.V_th.value, u.mV), v_shape)
        u_plus = self._broadcast_to_state(self._to_numpy(self.u_bar_plus.value, u.mV), v_shape)
        u_minus = self._broadcast_to_state(self._to_numpy(self.u_bar_minus.value, u.mV), v_shape)
        u_bar = self._broadcast_to_state(self._to_numpy(self.u_bar_bar.value, u.mV), v_shape)

        ditype = brainstate.environ.ditype()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=ditype),
            v_shape,
        )
        clamp_r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.clamp_step_count.value), dtype=ditype),
            v_shape,
        )

        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape)

        dftype = brainstate.environ.dftype()
        p = {
            'V_peak_rhs': self._broadcast_to_state(self._to_numpy(self.V_peak, u.mV), v_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'Delta_T': self._broadcast_to_state(self._to_numpy(self.Delta_T, u.mV), v_shape),
            'tau_w': self._broadcast_to_state(self._to_numpy(self.tau_w, u.ms), v_shape),
            'tau_z': self._broadcast_to_state(self._to_numpy(self.tau_z, u.ms), v_shape),
            'tau_V_th': self._broadcast_to_state(self._to_numpy(self.tau_V_th, u.ms), v_shape),
            'V_th_max': self._broadcast_to_state(self._to_numpy(self.V_th_max, u.mV), v_shape),
            'V_th_rest': self._broadcast_to_state(self._to_numpy(self.V_th_rest, u.mV), v_shape),
            'tau_u_bar_plus': self._broadcast_to_state(self._to_numpy(self.tau_u_bar_plus, u.ms), v_shape),
            'tau_u_bar_minus': self._broadcast_to_state(self._to_numpy(self.tau_u_bar_minus, u.ms), v_shape),
            'tau_u_bar_bar': self._broadcast_to_state(self._to_numpy(self.tau_u_bar_bar, u.ms), v_shape),
            'a': self._broadcast_to_state(self._to_numpy(self.a, u.nS), v_shape),
            'b': self._broadcast_to_state(self._to_numpy(self.b, u.pA), v_shape),
            'I_sp': self._broadcast_to_state(self._to_numpy(self.I_sp, u.pA), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
            'V_clamp': self._broadcast_to_state(self._to_numpy(self.V_clamp, u.mV), v_shape),
            'theta_plus': self._broadcast_to_state(self._to_numpy(self.theta_plus, u.mV), v_shape),
            'theta_minus': self._broadcast_to_state(self._to_numpy(self.theta_minus, u.mV), v_shape),
            'A_LTD': self._broadcast_to_state(self._to_numpy_unitless(self.A_LTD), v_shape),
            'A_LTP': self._broadcast_to_state(self._to_numpy_unitless(self.A_LTP), v_shape),
            'u_ref_squared': self._broadcast_to_state(self._to_numpy_unitless(self.u_ref_squared), v_shape),
            'atol': self._broadcast_to_state(self._to_numpy_unitless(self.gsl_error_tol), v_shape),
            'dt': self._broadcast_to_state(np.asarray(dt, dtype=dftype), v_shape),
        }

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=ditype),
            v_shape,
        )
        clamp_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._clamp_counts()), dtype=ditype),
            v_shape,
        )

        delta_v_q = self._sum_delta_inputs()
        delta_v = self._broadcast_to_state(self._to_numpy(delta_v_q, u.mV), v_shape)

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        spike_mask = np.zeros(v_shape, dtype=bool)

        V_next = np.empty_like(V)
        w_next = np.empty_like(w)
        z_next = np.empty_like(z)
        v_th_next = np.empty_like(v_th)
        u_plus_next = np.empty_like(u_plus)
        u_minus_next = np.empty_like(u_minus)
        u_bar_next = np.empty_like(u_bar)

        r_next = np.empty_like(r)
        clamp_r_next = np.empty_like(clamp_r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            y = np.asarray([
                V[idx], w[idx], z[idx], v_th[idx], u_plus[idx], u_minus[idx], u_bar[idx]
            ], dtype=dftype)

            r_i = int(r[idx])
            clamp_i = int(clamp_r[idx])
            h_i = float(max(h_int[idx], self._MIN_H))
            t_local = 0.0
            iters = 0
            local_spike = False

            pending_delta = float(delta_v[idx])

            while t_local < dt and iters < self._MAX_ITERS:
                iters += 1
                h_i = max(self._MIN_H, min(h_i, dt - t_local))

                is_refractory = r_i > 0
                is_clamped = clamp_i > 0

                def f(y_):
                    dftype = brainstate.environ.dftype()
                    return np.asarray(
                        self._dynamics_scalar(
                            y_[0],
                            y_[1],
                            y_[2],
                            y_[3],
                            y_[4],
                            y_[5],
                            y_[6],
                            is_refractory,
                            is_clamped,
                            i_stim[idx],
                            local_p,
                        ),
                        dtype=dftype,
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

                y4 = y + h_i * (
                    25.0 * k1 / 216.0
                    + 1408.0 * k3 / 2565.0
                    + 2197.0 * k4 / 4104.0
                    - k5 / 5.0
                )
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

                    if y[0] < -1e3 or y[1] < -1e6 or y[1] > 1e6:
                        raise ValueError('Numerical instability in aeif_psc_delta_clopath dynamics.')

                    if r_i == 0 and clamp_i == 0:
                        y[0] += pending_delta
                    pending_delta = 0.0

                    v_peak_detect_i = local_p['V_peak_rhs'] if local_p['Delta_T'] > 0.0 else y[3]

                    if y[0] >= v_peak_detect_i and clamp_i == 0:
                        local_spike = True

                        y[0] = local_p['V_clamp']
                        y[1] += local_p['b']
                        y[2] = local_p['I_sp']
                        y[3] = local_p['V_th_max']

                        clamp_i = int(clamp_counts[idx]) + 1 if int(clamp_counts[idx]) > 0 else 0
                    elif clamp_i == 1:
                        y[0] = local_p['V_reset']
                        clamp_i = 0
                        r_i = int(refr_counts[idx]) + 1 if int(refr_counts[idx]) > 0 else 0

                    if r_i > 0:
                        y[0] = local_p['V_reset']
                else:
                    fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
                    h_i = max(self._MIN_H, h_i * fac)

            # Decrement counters after full dt integration, matching NEST.
            if clamp_i > 0:
                clamp_i -= 1
            if r_i > 0:
                r_i -= 1

            spike_mask[idx] = local_spike

            V_next[idx] = y[0]
            w_next[idx] = y[1]
            z_next[idx] = y[2]
            v_th_next[idx] = y[3]
            u_plus_next[idx] = y[4]
            u_minus_next[idx] = y[5]
            u_bar_next[idx] = y[6]

            r_next[idx] = r_i
            clamp_r_next[idx] = clamp_i
            h_next[idx] = h_i

        self._write_clopath_history(
            V_next,
            u_plus_next,
            u_minus_next,
            u_bar_next,
            p,
        )

        self.V.value = V_next * u.mV
        self.w.value = w_next * u.pA
        self.z.value = z_next * u.pA
        self.V_th.value = v_th_next * u.mV
        self.u_bar_plus.value = u_plus_next * u.mV
        self.u_bar_minus.value = u_minus_next * u.mV
        self.u_bar_bar.value = u_bar_next * u.mV

        self.refractory_step_count.value = jnp.asarray(r_next, dtype=ditype)
        self.clamp_step_count.value = jnp.asarray(clamp_r_next, dtype=ditype)

        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA

        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(
                (self.refractory_step_count.value > 0) | (self.clamp_step_count.value > 0)
            )

        return u.math.asarray(spike_mask, dtype=dftype)
