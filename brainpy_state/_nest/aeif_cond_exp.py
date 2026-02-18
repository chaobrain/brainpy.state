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
    'aeif_cond_exp',
]


class aeif_cond_exp(NESTNeuron):
    r"""NEST-compatible ``aeif_cond_exp`` neuron model.

    Conductance-based adaptive exponential integrate-and-fire neuron with
    exponential synaptic conductances.

    This implementation follows NEST ``models/aeif_cond_exp.{h,cpp}`` and combines
    exponential spike-initiation current (AdEx), spike-triggered and subthreshold
    adaptation current, and exponentially decaying excitatory/inhibitory conductances.

    **1. Membrane, Synapse, and Adaptation Dynamics**

    The membrane potential :math:`V` evolves according to:

    .. math::

       C_m \frac{dV}{dt} = -g_L (V - E_L)
                          + g_L \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)
                          - g_{ex}(V - E_{ex})
                          - g_{in}(V - E_{in})
                          - w + I_e + I_{stim}

    where the first term is the leak current, the second term is the exponential
    spike-initiation current (the hallmark of the AdEx model), the third and fourth
    terms are excitatory and inhibitory synaptic currents, :math:`w` is the adaptation
    current, :math:`I_e` is constant external current, and :math:`I_{stim}` is the
    time-varying stimulation current.

    The adaptation current :math:`w` follows:

    .. math::

       \tau_w \frac{dw}{dt} = a (V - E_L) - w

    where :math:`a` controls subthreshold adaptation (coupling between :math:`V` and :math:`w`)
    and :math:`\tau_w` is the adaptation time constant.

    Excitatory and inhibitory conductances decay exponentially:

    .. math::

       \frac{d g_{ex}}{dt} = -\frac{g_{ex}}{\tau_{syn,ex}}, \qquad
       \frac{d g_{in}}{dt} = -\frac{g_{in}}{\tau_{syn,in}}

    Incoming spike weights (in nS) are split by sign and added to the respective conductances:

    .. math::

       g_{ex} \leftarrow g_{ex} + w_+, \qquad
       g_{in} \leftarrow g_{in} + |w_-|

    **2. Refractory Period and Spike Handling (NEST Semantics)**

    During refractory integration (when ``refractory_step_count > 0``), the effective
    membrane voltage is clamped to ``V_reset`` and :math:`dV/dt = 0`. Outside refractory
    periods, the right-hand side uses :math:`\min(V, V_{peak})` as the effective voltage
    to prevent numerical overflow in the exponential term.

    Spike detection threshold:

    - If ``Delta_T > 0``: spike when :math:`V \geq V_{peak}`
    - If ``Delta_T == 0`` (IAF-like limit): spike when :math:`V \geq V_{th}`

    Upon spike detection:

    1. :math:`V` is reset to ``V_reset``
    2. Adaptation jump :math:`w \leftarrow w + b` is applied immediately
    3. Refractory counter is set to ``ceil(t_ref / dt) + 1`` if ``t_ref > 0``

    Spike detection and reset occur *inside* the adaptive RKF45 integration substep loop.
    Therefore, with ``t_ref = 0``, multiple spikes can occur within one simulation step,
    matching NEST behavior.

    **3. Update Order per Simulation Step**

    Each call to ``update(x)`` performs the following sequence:

    1. Integrate ODEs on :math:`(t, t+dt]` via adaptive RKF45 with local error control
    2. Inside integration loop: apply refractory clamp and spike/reset/adaptation as needed
    3. After integration loop: decrement refractory counter once (if > 0)
    4. Apply arriving spike weights (from ``delta_inputs``) to ``g_ex`` / ``g_in``
    5. Store external current input ``x`` into one-step delayed buffer ``I_stim``
       (for use in the next time step)

    **4. Numerical Integration Details**

    The model uses an adaptive Runge-Kutta-Fehlberg 4(5) integrator (RKF45) with local
    error control. Step size is dynamically adjusted based on ``gsl_error_tol``. The
    integration step size is stored in ``integration_step`` and persists across time steps
    for efficiency. Minimum step size is clamped to ``_MIN_H = 1e-8 ms`` to prevent
    stalling. Maximum iterations per time step is ``_MAX_ITERS = 100000``. If membrane
    potential drops below -1000 mV or adaptation current exceeds ±1e6 pA, a numerical
    instability error is raised.

    **5. Computational Constraints and Assumptions**

    - **Overflow guard**: The exponential term can overflow if ``(V_peak - V_th) / Delta_T``
      is too large. The model validates that this ratio stays below ``log(max_float64 / 1e20)``
      at initialization, mirroring NEST's safeguard.
    - **Refractory clamp**: During refractory period, :math:`V` is clamped to ``V_reset``
      and :math:`dV/dt = 0`, but all other variables (``g_ex``, ``g_in``, ``w``) continue
      to evolve normally.
    - **Hard spike reset**: By default, ``spk_reset='hard'`` uses ``jax.lax.stop_gradient``
      to prevent gradient flow through spike times, matching typical neuroscience practice.
    - **Delayed input**: The current input ``x`` from time :math:`t` is stored in ``I_stim``
      and used during integration from :math:`t+dt` to :math:`t+2dt`. This one-step delay
      matches NEST's input handling.

    Parameters
    ----------
    in_size : Size (int, tuple of int, or callable returning shape)
        Neuron population shape. Supports integer (1D), tuple (multi-dimensional), or
        callable returning shape.
    V_peak : ArrayLike, optional
        Spike detection threshold (mV). Used when ``Delta_T > 0``. Default: 0 mV.
        Must satisfy ``V_peak >= V_th`` and ``V_peak > V_reset``.
    V_reset : ArrayLike, optional
        Reset potential (mV) after spike. Default: -60 mV. Must satisfy ``V_reset < V_peak``.
    t_ref : ArrayLike, optional
        Absolute refractory period (ms). Default: 0 ms. When 0, multiple spikes per
        simulation step are possible. Must be non-negative.
    g_L : ArrayLike, optional
        Leak conductance (nS). Default: 30 nS. Must be positive.
    C_m : ArrayLike, optional
        Membrane capacitance (pF). Default: 281 pF. Must be positive.
    E_ex : ArrayLike, optional
        Excitatory reversal potential (mV). Default: 0 mV.
    E_in : ArrayLike, optional
        Inhibitory reversal potential (mV). Default: -85 mV.
    E_L : ArrayLike, optional
        Leak reversal potential (mV). Default: -70.6 mV.
    Delta_T : ArrayLike, optional
        Exponential slope factor (mV) controlling sharpness of spike initiation.
        Default: 2 mV. Must be non-negative. Set to 0 to recover IAF-like behavior.
    tau_w : ArrayLike, optional
        Adaptation time constant (ms). Default: 144 ms. Must be positive.
    a : ArrayLike, optional
        Subthreshold adaptation coupling (nS). Default: 4 nS. Controls how strongly
        membrane potential drives adaptation current.
    b : ArrayLike, optional
        Spike-triggered adaptation increment (pA). Default: 80.5 pA. Added to ``w``
        on each spike.
    V_th : ArrayLike, optional
        Spike initiation threshold (mV) appearing in exponential term. Default: -50.4 mV.
        Must satisfy ``V_th <= V_peak``.
    tau_syn_ex : ArrayLike, optional
        Excitatory conductance decay time constant (ms). Default: 0.2 ms. Must be positive.
    tau_syn_in : ArrayLike, optional
        Inhibitory conductance decay time constant (ms). Default: 2.0 ms. Must be positive.
    I_e : ArrayLike, optional
        Constant external current (pA). Default: 0 pA.
    gsl_error_tol : ArrayLike, optional
        RKF45 local error tolerance (unitless). Default: 1e-6. Smaller values increase
        accuracy but slow integration. Must be positive.
    V_initializer : Callable, optional
        Initializer for membrane potential. Default: Constant(-70.6 mV).
    g_ex_initializer : Callable, optional
        Initializer for excitatory conductance. Default: Constant(0 nS).
    g_in_initializer : Callable, optional
        Initializer for inhibitory conductance. Default: Constant(0 nS).
    w_initializer : Callable, optional
        Initializer for adaptation current. Default: Constant(0 pA).
    spk_fun : Callable, optional
        Surrogate gradient function for differentiable spike generation. Default: ReluGrad().
        Used in ``get_spike()`` for gradient-based learning.
    spk_reset : str, optional
        Spike reset mode. Default: ``'hard'`` (stop gradient). Use ``'soft'`` to allow
        gradient flow through spike times.
    ref_var : bool, optional
        If True, expose boolean ``refractory`` state variable indicating whether neuron
        is in refractory period. Default: False.
    name : str, optional
        Name of the neuron group. Default: None.

    Parameter Mapping
    -----------------

    This table shows the correspondence between brainpy.state parameters, NEST parameters,
    and mathematical notation:

    ===================== ===================== ========================================= ==========================================
    **brainpy.state**     **NEST**              **Math Symbol**                           **Description**
    ===================== ===================== ========================================= ==========================================
    ``in_size``           (N/A)                 —                                         Population shape
    ``V_peak``            ``V_peak``            :math:`V_\mathrm{peak}`                   Spike detection threshold (if ``Delta_T > 0``)
    ``V_reset``           ``V_reset``           :math:`V_\mathrm{reset}`                  Reset potential
    ``t_ref``             ``t_ref``             :math:`t_\mathrm{ref}`                    Absolute refractory duration
    ``g_L``               ``g_L``               :math:`g_\mathrm{L}`                      Leak conductance
    ``C_m``               ``C_m``               :math:`C_\mathrm{m}`                      Membrane capacitance
    ``E_ex``              ``E_ex``              :math:`E_\mathrm{ex}`                     Excitatory reversal potential
    ``E_in``              ``E_in``              :math:`E_\mathrm{in}`                     Inhibitory reversal potential
    ``E_L``               ``E_L``               :math:`E_\mathrm{L}`                      Leak reversal potential
    ``Delta_T``           ``Delta_T``           :math:`\Delta_T`                          Exponential slope factor
    ``tau_w``             ``tau_w``             :math:`\tau_w`                            Adaptation time constant
    ``a``                 ``a``                 :math:`a`                                 Subthreshold adaptation coupling
    ``b``                 ``b``                 :math:`b`                                 Spike-triggered adaptation increment
    ``V_th``              ``V_th``              :math:`V_\mathrm{th}`                     Spike initiation threshold
    ``tau_syn_ex``        ``tau_syn_ex``        :math:`\tau_{\mathrm{syn,ex}}`            Excitatory conductance time constant
    ``tau_syn_in``        ``tau_syn_in``        :math:`\tau_{\mathrm{syn,in}}`            Inhibitory conductance time constant
    ``I_e``               ``I_e``               :math:`I_\mathrm{e}`                      Constant external current
    ``gsl_error_tol``     ``gsl_error_tol``     —                                         RKF45 solver tolerance
    ===================== ===================== ========================================= ==========================================

    Attributes
    ----------
    V : HiddenState
        Membrane potential (mV). Shape: ``(batch_size,) + varshape``.
    g_ex : HiddenState
        Excitatory conductance (nS). Shape: ``(batch_size,) + varshape``.
    g_in : HiddenState
        Inhibitory conductance (nS). Shape: ``(batch_size,) + varshape``.
    w : HiddenState
        Adaptation current (pA). Shape: ``(batch_size,) + varshape``.
    refractory_step_count : ShortTermState
        Remaining refractory time steps (int32). Shape: ``(batch_size,) + varshape``.
    integration_step : ShortTermState
        Persistent RKF45 internal step size (ms). Shape: ``(batch_size,) + varshape``.
    I_stim : ShortTermState
        One-step delayed current buffer (pA). Shape: ``(batch_size,) + varshape``.
    last_spike_time : ShortTermState
        Last emitted spike time (ms). Updated to :math:`t + dt` on spike. Shape: ``(batch_size,) + varshape``.
    refractory : ShortTermState (optional)
        Boolean refractory indicator. Only present if ``ref_var=True``. Shape: ``(batch_size,) + varshape``.

    Raises
    ------
    ValueError
        If ``V_peak < V_th``, ``Delta_T < 0``, ``V_reset >= V_peak``, ``C_m <= 0``,
        ``t_ref < 0``, any time constant ``<= 0``, ``gsl_error_tol <= 0``, or if
        ``(V_peak - V_th) / Delta_T`` would cause exponential overflow.
    ValueError
        During integration, if membrane potential drops below -1000 mV or adaptation
        current exceeds ±1e6 pA, indicating numerical instability.

    Notes
    -----
    - **Default refractory period**: ``t_ref = 0`` matches NEST and can allow multiple
      spikes per simulation step. Set ``t_ref > 0`` to enforce absolute refractory period.
    - **Spike output**: The returned spike tensor is binary per step (0 or 1), even if
      multiple spikes occur internally. Use ``last_spike_time`` to track precise spike timing.
    - **Gradient-based learning**: Use ``get_spike()`` method for differentiable spike
      generation with surrogate gradients, suitable for gradient-based learning.
    - **NEST compatibility**: This implementation closely follows NEST's C++ source,
      including refractory clamping, spike detection logic, and overflow guards.

    Examples
    --------
    Create and simulate a population of AdEx neurons:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
        >>> import brainstate
        >>> # Create 100 AdEx neurons
        >>> neurons = bst.aeif_cond_exp(100)
        >>> # Initialize states
        >>> neurons.init_all_states()
        >>> # Simulate with constant current input
        >>> with brainstate.environ.context(dt=0.1 * u.ms):
        ...     for _ in range(1000):
        ...         spikes = neurons.update(x=500 * u.pA)

    Create with custom parameters matching cortical pyramidal cells:

    .. code-block:: python

        >>> neurons = bst.aeif_cond_exp(
        ...     in_size=100,
        ...     V_peak=0.0 * u.mV,
        ...     V_reset=-70.0 * u.mV,
        ...     t_ref=2.0 * u.ms,
        ...     g_L=30.0 * u.nS,
        ...     C_m=281.0 * u.pF,
        ...     Delta_T=2.0 * u.mV,
        ...     tau_w=144.0 * u.ms,
        ...     a=4.0 * u.nS,
        ...     b=80.5 * u.pA,
        ... )

    Access state variables:

    .. code-block:: python

        >>> neurons.init_all_states()
        >>> print(neurons.V.value.shape)  # Membrane potential
        >>> print(neurons.g_ex.value.shape)  # Excitatory conductance
        >>> print(neurons.w.value.shape)  # Adaptation current
        >>> print(neurons.refractory_step_count.value.shape)  # Refractory counter

    References
    ----------
    .. [1] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
           model as an effective description of neuronal activity.
           Journal of Neurophysiology, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [2] NEST source: ``models/aeif_cond_exp.h`` and
           ``models/aeif_cond_exp.cpp``.
           https://github.com/nest/nest-simulator
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

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _to_numpy_unitless(x):
        return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        v_reset = self._to_numpy(self.V_reset, u.mV)
        v_peak = self._to_numpy(self.V_peak, u.mV)
        v_th = self._to_numpy(self.V_th, u.mV)
        delta_t = self._to_numpy(self.Delta_T, u.mV)

        if np.any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if np.any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if np.any(v_reset >= v_peak):
            raise ValueError('Ensure that: V_reset < V_peak .')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Ensure that C_m >0')
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

        Creates and initializes hidden states (V, g_ex, g_in, w) and short-term states
        (last_spike_time, refractory_step_count, integration_step, I_stim). If ``ref_var=True``,
        also creates boolean refractory indicator.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for vectorized simulation. If None, states have shape ``varshape``.
            If provided, states have shape ``(batch_size,) + varshape``.
        **kwargs : dict
            Additional keyword arguments (ignored).

        Notes
        -----
        - ``V`` initialized to ``V_initializer`` (default: -70.6 mV)
        - ``g_ex``, ``g_in`` initialized to 0 nS
        - ``w`` initialized to 0 pA
        - ``last_spike_time`` initialized to -1e7 ms (far in past)
        - ``refractory_step_count`` initialized to 0
        - ``integration_step`` initialized to current simulation dt
        - ``I_stim`` initialized to 0 pA
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        w = braintools.init.param(self.w_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)
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

        Resets all states to their initialized values without recreating state objects.
        Useful for running multiple trials without reinitializing the model.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for vectorized simulation. If None, states have shape ``varshape``.
            If provided, states have shape ``(batch_size,) + varshape``.
        **kwargs : dict
            Additional keyword arguments (ignored).

        Notes
        -----
        All states are reset to the same initial values as in ``init_state()``.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.g_ex.value = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        self.g_in.value = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
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
        r"""Generate differentiable spike signal using surrogate gradient.

        Computes a continuous spike probability using the surrogate gradient function
        (``spk_fun``) applied to scaled membrane potential. This enables gradient-based
        learning through spike generation.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (mV). If None, uses current ``self.V.value``. Shape: arbitrary,
            but typically ``(batch_size,) + varshape``.

        Returns
        -------
        spike_prob : ArrayLike
            Continuous spike signal in [0, 1]. Shape matches input ``V``. Values near 0
            indicate no spike, values near 1 indicate spike. Exact range depends on
            ``spk_fun`` (e.g., ``ReluGrad`` returns values in [0, 1]).

        Notes
        -----
        - The membrane potential is scaled as ``(V - V_th) / (V_th - V_reset)`` before
          applying the surrogate function.
        - This method is primarily used for gradient-based learning and does NOT affect
          the hard spike detection used in ``update()``.
        - For binary spike output matching NEST semantics, use the return value of
          ``update()``.
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
    def _dynamics_scalar(v, g_ex, g_in, w, is_refractory, i_stim, p):
        r"""Compute ODE derivatives for a single neuron.

        Evaluates the right-hand side of the AdEx ODE system with refractory clamping
        and voltage overflow protection. Used internally by RKF45 integrator.

        Parameters
        ----------
        v : float
            Membrane potential (mV, unitless).
        g_ex : float
            Excitatory conductance (nS, unitless).
        g_in : float
            Inhibitory conductance (nS, unitless).
        w : float
            Adaptation current (pA, unitless).
        is_refractory : bool
            Whether neuron is in refractory period.
        i_stim : float
            Stimulation current (pA, unitless).
        p : dict
            Parameter dictionary with keys: 'V_peak_rhs', 'V_reset', 'E_L', 'E_ex', 'E_in',
            'C_m', 'g_L', 'Delta_T', 'tau_w', 'a', 'b', 'V_th', 'tau_syn_ex', 'tau_syn_in', 'I_e'.

        Returns
        -------
        dv : float
            Membrane potential derivative (mV/ms, unitless). Zero during refractory period.
        dg_ex : float
            Excitatory conductance derivative (nS/ms, unitless).
        dg_in : float
            Inhibitory conductance derivative (nS/ms, unitless).
        dw : float
            Adaptation current derivative (pA/ms, unitless).

        Notes
        -----
        - During refractory period, ``v_eff`` is clamped to ``V_reset`` and ``dv = 0``.
        - Outside refractory period, ``v_eff = min(v, V_peak_rhs)`` to prevent exponential overflow.
        - If ``Delta_T == 0``, the exponential spike current is disabled (IAF-like behavior).
        - All units are implicit (stripped before calling this function).
        """
        v_eff = p['V_reset'] if is_refractory else min(v, p['V_peak_rhs'])

        i_syn_exc = g_ex * (v_eff - p['E_ex'])
        i_syn_inh = g_in * (v_eff - p['E_in'])
        i_spike = 0.0 if p['Delta_T'] == 0.0 else (
            p['g_L'] * p['Delta_T'] * math.exp((v_eff - p['V_th']) / p['Delta_T'])
        )

        dv = 0.0 if is_refractory else (
            -p['g_L'] * (v_eff - p['E_L']) + i_spike - i_syn_exc - i_syn_inh - w + p['I_e'] + i_stim
        ) / p['C_m']

        dg_ex = -g_ex / p['tau_syn_ex']
        dg_in = -g_in / p['tau_syn_in']
        dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
        return dv, dg_ex, dg_in, dw

    def update(self, x=0.0 * u.pA):
        r"""Advance neuron state by one simulation time step.

        Integrates the AdEx ODE system over interval :math:`(t, t+dt]` using adaptive
        RKF45 with local error control. Handles spike detection, reset, adaptation jumps,
        refractory clamping, and synaptic input processing following NEST semantics.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input (pA) for the current time step. Default: 0 pA.
            Shape: scalar, ``varshape``, or ``(batch_size,) + varshape``. This input
            is stored in ``I_stim`` and will be used during the *next* time step
            (one-step delay, matching NEST).

        Returns
        -------
        spike : ArrayLike
            Binary spike indicator (0 or 1, dtype float64). Shape: ``(batch_size,) + varshape``.
            Value is 1 if at least one spike occurred during this time step, 0 otherwise.
            Multiple spikes within one step (when ``t_ref = 0``) are compressed to a single
            binary flag.

        Notes
        -----
        **Update sequence**:

        1. **ODE integration**: Integrate :math:`(t, t+dt]` via adaptive RKF45. Inside the
           integration loop:

           - Apply refractory clamp if ``refractory_step_count > 0``
           - Check for spike when :math:`V \geq V_{peak}` (or :math:`V \geq V_{th}` if ``Delta_T = 0``)
           - On spike: reset :math:`V \leftarrow V_{reset}`, jump :math:`w \leftarrow w + b`,
             set ``refractory_step_count = ceil(t_ref / dt) + 1``

        2. **Post-integration**: Decrement ``refractory_step_count`` once (if > 0)

        3. **Synaptic input**: Process ``delta_inputs`` (spike weights from projections), split
           by sign, and add to ``g_ex`` / ``g_in``

        4. **Delayed input buffer**: Store current external input ``x`` in ``I_stim`` for use
           in the next time step

        5. **Spike time tracking**: Update ``last_spike_time`` to :math:`t + dt` for neurons
           that spiked

        **Numerical integration details**:

        - Uses Runge-Kutta-Fehlberg 4(5) with embedded error estimation
        - Step size is adaptive based on ``gsl_error_tol``
        - Minimum step size: ``_MIN_H = 1e-8 ms``
        - Maximum iterations: ``_MAX_ITERS = 100000`` per simulation step
        - Step size is persistent across time steps (stored in ``integration_step``)

        **Failure modes**:

        - Raises ``ValueError`` if membrane potential drops below -1000 mV or adaptation
          current exceeds ±1e6 pA, indicating numerical instability (typically from bad
          parameters or extreme inputs)
        - Does NOT raise error if max iterations exceeded; instead completes integration
          with accumulated error (silent degradation)

        **Computational cost**:

        - Per-neuron scalar integration (no vectorization across neurons)
        - Cost scales with ``1/gsl_error_tol`` (smaller tolerance = more substeps)
        - Typical: 1-10 substeps per simulation step for standard parameters

        Examples
        --------
        Basic usage:

        .. code-block:: python

            >>> import brainpy.state as bst
            >>> import brainunit as u
            >>> import brainstate
            >>> neurons = bst.aeif_cond_exp(100)
            >>> neurons.init_all_states()
            >>> with brainstate.environ.context(dt=0.1 * u.ms):
            ...     spikes = neurons.update(x=300 * u.pA)
            >>> print(spikes.sum())  # Number of neurons that spiked

        Access updated states:

        .. code-block:: python

            >>> spikes = neurons.update(x=300 * u.pA)
            >>> print(neurons.V.value)  # Updated membrane potential
            >>> print(neurons.w.value)  # Updated adaptation current
            >>> print(neurons.last_spike_time.value)  # Spike times
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        g_ex = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape)
        g_in = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape)
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
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        spike_mask = np.zeros(v_shape, dtype=bool)
        V_next = np.empty_like(V)
        g_ex_next = np.empty_like(g_ex)
        g_in_next = np.empty_like(g_in)
        w_next = np.empty_like(w)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            y = np.asarray([V[idx], g_ex[idx], g_in[idx], w[idx]], dtype=np.float64)
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
                        raise ValueError('Numerical instability in aeif_cond_exp dynamics.')

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
            g_ex_next[idx] = y[1]
            g_in_next[idx] = y[2]
            w_next[idx] = y[3]
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.g_ex.value = g_ex_next * u.nS
        self.g_in.value = g_in_next * u.nS
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
