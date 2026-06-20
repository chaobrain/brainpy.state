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

from collections.abc import Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from brainpy_state._nest_base.utils import is_tracer, validate_aeif_overflow, AdaptiveRungeKuttaStep, cond_any
from brainpy_state._nest_base.base import NESTNeuron

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

        >>> from brainpy import state as bst
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

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    #: Unit the multi-receptor ``connect(receptor_type=k)`` bridge uses to scale the
    #: gathered per-port deposit into the ``w_by_rec`` mantissa (conductance -> nS).
    receptor_input_unit = u.nS

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
        name: str | None = None,
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

        # Two delivery ports for the Simulator's multi-receptor bridge: receptor 1 ->
        # g_ex (excitatory), receptor 2 -> g_in (inhibitory). NEST routes excitation /
        # inhibition by weight *sign* (aeif_cond_exp.cpp handle(): weight>0 -> spike_exc_,
        # else spike_inh_); the Simulator expresses that as receptor_type=1/2 with
        # positive nS weights.
        self.n_receptors = 2

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
        v_reset = self.V_reset
        v_peak = self.V_peak
        v_th = self.V_th
        delta_t = self.Delta_T / u.mV

        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (v_reset, v_peak, v_th, delta_t)):
            return

        if cond_any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if cond_any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if cond_any(v_reset >= v_peak):
            raise ValueError('Ensure that: V_reset < V_peak .')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Ensure that C_m >0')
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
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape)
        V = braintools.init.param(self.V_initializer, self.varshape)
        w = braintools.init.param(self.w_initializer, self.varshape)

        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)
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

    def _vector_field(self, state, extra):
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, u.math.minimum(state.V, self.V_peak))

        i_syn_exc = state.g_ex * (v_eff - self.E_ex)
        i_syn_inh = state.g_in * (v_eff - self.E_in)

        delta_t_safe = u.math.where(self.Delta_T == 0.0 * u.mV, 1.0 * u.mV, self.Delta_T)
        exp_arg = u.math.clip((v_eff - self.V_th) / delta_t_safe, -500.0, 500.0)
        i_spike = self.g_L * self.Delta_T * u.math.exp(exp_arg)

        dV_raw = (
                     -self.g_L * (v_eff - self.E_L) + i_spike
                     - i_syn_exc - i_syn_inh - state.w + self.I_e + extra.i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        dg_ex = -state.g_ex / self.tau_syn_ex
        dg_in = -state.g_in / self.tau_syn_in
        dw = (self.a * (v_eff - self.E_L) - state.w) / self.tau_w

        return DotDict(V=dV, g_ex=dg_ex, g_in=dg_in, w=dw)

    def _event_fn(self, state, extra, accept):
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
        w_by_rec : ArrayLike or None, optional
            Per-receptor synaptic conductance jumps supplied by the Simulator's
            multi-receptor bridge, shape ``(*varshape, n_receptors)`` as a
            dimensionless ``nS`` mantissa. Column 0 (``receptor_type=1``) is added to
            the excitatory conductance ``g_ex`` and column 1 (``receptor_type=2``) to
            the inhibitory conductance ``g_in``. When ``None`` (the legacy path), the
            jumps are self-pulled from the ``label='w_ex'/'w_in'`` delta inputs instead.

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
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value
        g_ex = self.g_ex.value
        g_in = self.g_in.value
        w = self.w.value
        r = self.refractory_step_count.value
        i_stim = self.I_stim.value
        h = self.integration_step.value

        # Spike detection threshold: V_peak if Delta_T > 0, else V_th.
        v_peak_detect = u.math.where(self.Delta_T > 0.0 * u.mV, self.V_peak, self.V_th)

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, g_ex=g_ex, g_in=g_in, w=w)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            v_peak_detect=v_peak_detect,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, g_ex, g_in, w = ode_state.V, ode_state.g_ex, ode_state.g_in, ode_state.w
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in aeif_cond_exp dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration). Two delivery paths: the
        # Simulator's multi-receptor bridge pre-gathers per-receptor conductance jumps
        # (nS mantissa, shape ``(*varshape, n_receptors)``) and passes them via
        # ``w_by_rec`` -- column 0 -> g_ex, column 1 -> g_in; the legacy BrainPy-style
        # path self-pulls the ``label='w_ex'/'w_in'`` delta inputs. Only the *source* of
        # ``w_ex``/``w_in`` changes; the exponential kinetics below consume them unchanged.
        if w_by_rec is not None:
            w_by_rec = jnp.asarray(w_by_rec, dtype=dftype)
            w_ex = w_by_rec[..., 0] * u.nS
            w_in = w_by_rec[..., 1] * u.nS
        else:
            w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
            w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')
        # Exponential synapses: direct addition (no pscon factor unlike alpha).
        g_ex = g_ex + w_ex
        g_in = g_in + w_in

        # Write back state.
        self.V.value = V
        self.g_ex.value = g_ex
        self.g_in.value = g_in
        self.w.value = w
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
