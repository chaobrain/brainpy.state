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

import numpy as np
from typing import Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'iaf_cond_exp',
]


class iaf_cond_exp(Neuron):
    r"""Leaky integrate-and-fire model with exponential conductance synapses.

    This is a conductance-based leaky integrate-and-fire neuron with hard threshold,
    fixed absolute refractory period, exponentially decaying excitatory and inhibitory
    synaptic conductances, and no adaptation variables.

    This implementation follows NEST ``iaf_cond_exp`` dynamics and update order,
    using NEST C++ model behavior as the source of truth.

    **1. Membrane Potential and Synaptic Conductances**

    The membrane potential evolves according to

    .. math::

       \frac{dV_\mathrm{m}}{dt} =
       \frac{-g_\mathrm{L}(V_\mathrm{m}-E_\mathrm{L})
             - I_\mathrm{syn}
             + I_\mathrm{e}
             + I_\mathrm{stim}}
            {C_\mathrm{m}}

    with the total synaptic current given by

    .. math::

       I_\mathrm{syn}
       = I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}}
       = g_\mathrm{ex}(V_\mathrm{m}-E_\mathrm{ex})
       + g_\mathrm{in}(V_\mathrm{m}-E_\mathrm{in}) .

    Synaptic conductances decay exponentially:

    .. math::

       \frac{dg_\mathrm{ex}}{dt} = -\frac{g_\mathrm{ex}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{dg_\mathrm{in}}{dt} = -\frac{g_\mathrm{in}}{\tau_{\mathrm{syn,in}}}.

    A presynaptic spike with weight :math:`w` causes an instantaneous jump at
    the end of the simulation step:

    .. math::

       w > 0 \Rightarrow g_\mathrm{ex} \leftarrow g_\mathrm{ex} + w,
       \qquad
       w < 0 \Rightarrow g_\mathrm{in} \leftarrow g_\mathrm{in} + |w|.

    **2. Spike Emission and Refractory Mechanism**

    A spike is emitted when :math:`V_\mathrm{m} \ge V_\mathrm{th}` at the end of
    a simulation step. On spike:

    * :math:`V_\mathrm{m}` is reset to :math:`V_\mathrm{reset}`,
    * refractory counter is set to :math:`\lceil t_\mathrm{ref}/dt \rceil`,
    * spike time is recorded as :math:`t + dt`.

    During absolute refractory period:

    * membrane potential is clamped to :math:`V_\mathrm{reset}`,
    * :math:`dV_\mathrm{m}/dt = 0`,
    * conductances continue to decay.

    **3. Numerical Integration and Update Order**

    NEST integrates this model with adaptive RKF45. This implementation mirrors
    that behavior with an RKF45(4,5) integrator and persistent internal step size.
    The discrete-time update order is:

    1. Integrate continuous dynamics on :math:`(t, t+dt]`.
    2. Add synaptic conductance jumps from spike inputs arriving this step.
    3. Apply refractory countdown / threshold check / reset and spike emission.
    4. Store external current input as :math:`I_\mathrm{stim}` for the next step.

    The one-step delayed application of current input (``I_stim`` buffer) is
    intentional and matches NEST's ring-buffer update semantics.

    Parameters
    ----------
    in_size : int, tuple of int
        Shape of the neuron population. Can be an integer for 1D population or
        tuple for multi-dimensional populations.
    E_L : float, ArrayLike, optional
        Leak reversal potential. Must have unit of voltage (mV).
        Default: -70 mV
    C_m : float, ArrayLike, optional
        Membrane capacitance. Must be strictly positive with unit of capacitance (pF).
        Default: 250 pF
    t_ref : float, ArrayLike, optional
        Absolute refractory period duration. Must be non-negative with unit of time (ms).
        Default: 2 ms
    V_th : float, ArrayLike, optional
        Spike threshold voltage. Must be greater than ``V_reset`` with unit of voltage (mV).
        Default: -55 mV
    V_reset : float, ArrayLike, optional
        Reset potential after spike. Must be less than ``V_th`` with unit of voltage (mV).
        Default: -60 mV
    E_ex : float, ArrayLike, optional
        Excitatory reversal potential. Must have unit of voltage (mV).
        Default: 0 mV
    E_in : float, ArrayLike, optional
        Inhibitory reversal potential. Must have unit of voltage (mV).
        Default: -85 mV
    g_L : float, ArrayLike, optional
        Leak conductance. Must be strictly positive with unit of conductance (nS).
        Default: 16.6667 nS
    tau_syn_ex : float, ArrayLike, optional
        Excitatory synaptic conductance time constant. Must be strictly positive
        with unit of time (ms). Default: 0.2 ms
    tau_syn_in : float, ArrayLike, optional
        Inhibitory synaptic conductance time constant. Must be strictly positive
        with unit of time (ms). Default: 2.0 ms
    I_e : float, ArrayLike, optional
        Constant external input current. Must have unit of current (pA).
        Default: 0 pA
    V_initializer : Callable, optional
        Initializer function for membrane potential state. Must return values with
        voltage units. Default: ``braintools.init.Constant(-70 * u.mV)``
    g_ex_initializer : Callable, optional
        Initializer function for excitatory conductance state. Must return values
        with conductance units. Default: ``braintools.init.Constant(0 * u.nS)``
    g_in_initializer : Callable, optional
        Initializer function for inhibitory conductance state. Must return values
        with conductance units. Default: ``braintools.init.Constant(0 * u.nS)``
    spk_fun : Callable, optional
        Surrogate gradient function for differentiable spike generation. Must be
        a callable with signature ``(x: ArrayLike) -> ArrayLike``.
        Default: ``braintools.surrogate.ReluGrad()``
    spk_reset : str, optional
        Spike reset mode. Options: ``'hard'`` (gradient blocking, matches NEST),
        ``'soft'`` (gradient-friendly subtraction). Default: ``'hard'``
    ref_var : bool, optional
        If True, expose a boolean ``refractory`` state variable indicating whether
        each neuron is currently in the refractory period. Default: False
    name : str, optional
        Name of the neuron population. If None, an automatic name is generated.

    **Parameter Mapping**

    ==================== ================== ==========================================
    **Parameter**        **Default**        **Math equivalent**
    ==================== ================== ==========================================
    ``in_size``          (required)         —
    ``E_L``              -70 mV             :math:`E_\mathrm{L}`
    ``C_m``              250 pF             :math:`C_\mathrm{m}`
    ``t_ref``            2 ms               :math:`t_\mathrm{ref}`
    ``V_th``             -55 mV             :math:`V_\mathrm{th}`
    ``V_reset``          -60 mV             :math:`V_\mathrm{reset}`
    ``E_ex``             0 mV               :math:`E_\mathrm{ex}`
    ``E_in``             -85 mV             :math:`E_\mathrm{in}`
    ``g_L``              16.6667 nS         :math:`g_\mathrm{L}`
    ``tau_syn_ex``       0.2 ms             :math:`\tau_{\mathrm{syn,ex}}`
    ``tau_syn_in``       2.0 ms             :math:`\tau_{\mathrm{syn,in}}`
    ``I_e``              0 pA               :math:`I_\mathrm{e}`
    ``V_initializer``    Constant(-70 mV)   —
    ``g_ex_initializer`` Constant(0 nS)     —
    ``g_in_initializer`` Constant(0 nS)     —
    ``spk_fun``          ReluGrad()         —
    ``spk_reset``        ``'hard'``         —
    ``ref_var``          ``False``          —
    ==================== ================== ==========================================

    State Variables
    ---------------
    V : brainstate.HiddenState
        Membrane potential :math:`V_\mathrm{m}` in mV, shape ``(\*in_size, \*batch_shape)``.
    g_ex : brainstate.HiddenState
        Excitatory synaptic conductance :math:`g_\mathrm{ex}` in nS,
        shape ``(\*in_size, \*batch_shape)``.
    g_in : brainstate.HiddenState
        Inhibitory synaptic conductance :math:`g_\mathrm{in}` in nS,
        shape ``(\*in_size, \*batch_shape)``.
    last_spike_time : brainstate.ShortTermState
        Last spike emission time in ms, shape ``(\*in_size, \*batch_shape)``.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory time steps (int32), shape ``(\*in_size, \*batch_shape)``.
    integration_step : brainstate.ShortTermState
        Internal RKF45 adaptive step size in ms, shape ``(\*in_size, \*batch_shape)``.
    I_stim : brainstate.ShortTermState
        Buffered external current (one-step delayed) in pA, shape ``(\*in_size, \*batch_shape)``.
    refractory : brainstate.ShortTermState, optional
        Boolean refractory state indicator, shape ``(\*in_size, \*batch_shape)``.
        Only present if ``ref_var=True``.

    Raises
    ------
    ValueError
        If ``V_reset >= V_th`` (reset must be below threshold).
    ValueError
        If ``C_m <= 0`` (capacitance must be strictly positive).
    ValueError
        If ``t_ref < 0`` (refractory time cannot be negative).
    ValueError
        If ``tau_syn_ex <= 0`` or ``tau_syn_in <= 0`` (time constants must be positive).

    Notes
    -----
    * Defaults follow NEST C++ source for ``iaf_cond_exp``.
    * In NEST docs, some printed default values may differ from the source for
      specific releases; source code behavior is used here for parity.
    * Synaptic spike weights are interpreted in conductance units (nS), with
      positive/negative sign selecting excitatory/inhibitory channel.
    * The RKF45 integrator uses absolute error tolerance of 1e-3 with minimum
      step size of 1e-8 ms and maximum iteration count of 10000 per simulation step.
    * Integration may fall back to minimum step size if adaptive control fails,
      potentially degrading accuracy for stiff dynamics.

    Examples
    --------
    Create a population of 100 conductance-based LIF neurons:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
        >>> neurons = bst.iaf_cond_exp(100, V_th=-50*u.mV, t_ref=5*u.ms)

    Simulate with external current input:

    .. code-block:: python

        >>> with bst.environ.context(dt=0.1*u.ms):
        ...     neurons.init_all_states()
        ...     for t in range(1000):
        ...         spike = neurons.update(x=500*u.pA)

    References
    ----------
    .. [1] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. Journal of Computational Neuroscience
           16:159-175. DOI: https://doi.org/10.1023/B:JCNS.0000014108.03012.81
    .. [2] NEST Simulator ``iaf_cond_exp`` model documentation and C++ source:
           ``models/iaf_cond_exp.h`` and ``models/iaf_cond_exp.cpp``.

    See Also
    --------
    iaf_psc_delta : Current-based LIF with delta synapses
    iaf_psc_exp : Current-based LIF with exponential synapses
    iaf_cond_alpha : Conductance-based LIF with alpha-function synapses
    """
    __module__ = 'brainpy.state'

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 10000

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 250. * u.pF,
        t_ref: ArrayLike = 2. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -60. * u.mV,
        E_ex: ArrayLike = 0. * u.mV,
        E_in: ArrayLike = -85. * u.mV,
        g_L: ArrayLike = 16.6667 * u.nS,
        tau_syn_ex: ArrayLike = 0.2 * u.ms,
        tau_syn_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        g_ex_initializer: Callable = braintools.init.Constant(0. * u.nS),
        g_in_initializer: Callable = braintools.init.Constant(0. * u.nS),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive.')

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Creates and registers state variables for membrane potential, synaptic
        conductances, refractory tracking, RKF45 integration, and buffered currents.
        All states are initialized using the configured initializer functions.

        Parameters
        ----------
        batch_size : int, optional
            Number of batches for parallelized simulation. If None, creates states
            without batch dimension. Default: None
        **kwargs
            Additional keyword arguments (reserved for future extensions).

        Notes
        -----
        * State variables are registered as ``brainstate.HiddenState`` (continuous
          dynamics) or ``brainstate.ShortTermState`` (discrete/reset behavior).
        * ``last_spike_time`` is initialized to -1e7 ms (far past) to indicate no
          prior spikes.
        * ``integration_step`` is initialized to the simulation timestep ``dt``.
        * If ``ref_var=True``, an additional boolean ``refractory`` state is created.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = self._safe_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0. * u.pA), self.varshape, batch_size)
        )

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to their initial values.

        Re-initializes existing state variables without creating new state objects.
        Used to restart simulations while preserving state object references.

        Parameters
        ----------
        batch_size : int, optional
            Number of batches for parallelized simulation. If None, resets states
            without batch dimension. Default: None
        **kwargs
            Additional keyword arguments (reserved for future extensions).

        Notes
        -----
        * This method modifies ``.value`` attributes of existing state objects.
        * All states are reset using the same initializers as ``init_state``.
        * Integration step size is reset to the current simulation timestep.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.g_ex.value = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        self.g_in.value = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
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
            braintools.init.Constant(0. * u.pA), self.varshape, batch_size
        )
        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output using surrogate gradient.

        Transforms membrane potential into a continuous spike signal suitable for
        gradient-based learning. Uses the configured surrogate gradient function
        (``spk_fun``) applied to normalized voltage distance from threshold.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential values to evaluate (with voltage units). If None,
            uses current ``self.V.value``. Default: None

        Returns
        -------
        ArrayLike
            Spike signal with same shape as input ``V``. Values are continuous
            (not binary) to support gradient flow. Typically near 0 below threshold
            and near 1 above threshold, with smooth transition determined by ``spk_fun``.

        Notes
        -----
        * Voltage is normalized as ``(V - V_th) / (V_th - V_reset)`` before applying
          the surrogate function.
        * The normalization ensures consistent surrogate behavior across different
          threshold/reset voltage configurations.
        * This method is used internally by ``update`` but can also be called
          externally for spike extraction.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _sum_signed_delta_inputs(self):
        g_ex = u.math.zeros_like(self.g_ex.value)
        g_in = u.math.zeros_like(self.g_in.value)
        if self.delta_inputs is None:
            return g_ex, g_in

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            zero = u.math.zeros_like(out)
            g_ex = g_ex + u.math.maximum(out, zero)
            g_in = g_in + u.math.maximum(-out, zero)
        return g_ex, g_in

    @staticmethod
    def _dynamics_scalar(v, g_ex, g_in, is_refractory, i_stim, p):
        r"""Compute time derivatives for a single neuron.

        Evaluates the ODE system for membrane potential and synaptic conductances.
        During refractory period, voltage derivative is clamped to zero while
        conductances continue decaying.

        Parameters
        ----------
        v : float
            Membrane potential in mV.
        g_ex : float
            Excitatory conductance in nS.
        g_in : float
            Inhibitory conductance in nS.
        is_refractory : bool
            Whether neuron is in absolute refractory period.
        i_stim : float
            External stimulus current in pA.
        p : dict
            Dictionary of model parameters in SI-derived units (mV, nS, pF, ms, pA).
            Required keys: 'V_th', 'E_ex', 'E_in', 'E_L', 'g_L', 'C_m', 'I_e',
            'tau_syn_ex', 'tau_syn_in'.

        Returns
        -------
        tuple of (float, float, float)
            Time derivatives (dV/dt, dg_ex/dt, dg_in/dt) in (mV/ms, nS/ms, nS/ms).

        Notes
        -----
        * Voltage is clamped to threshold during integration (``v_eff = min(v, V_th)``)
          to prevent numerical overshoot beyond spike detection.
        * During refractory period, dV/dt = 0 (voltage clamped externally to V_reset).
        * Sign convention: excitatory and inhibitory synaptic currents are positive
          when driving voltage toward their respective reversal potentials.
        """
        if is_refractory:
            return 0.0, -g_ex / p['tau_syn_ex'], -g_in / p['tau_syn_in']

        v_eff = min(v, p['V_th'])
        i_syn_exc = g_ex * (v_eff - p['E_ex'])
        i_syn_inh = g_in * (v_eff - p['E_in'])
        i_l = p['g_L'] * (v_eff - p['E_L'])

        dv = (-i_l + i_stim + p['I_e'] - i_syn_exc - i_syn_inh) / p['C_m']
        dg_ex = -g_ex / p['tau_syn_ex']
        dg_in = -g_in / p['tau_syn_in']
        return dv, dg_ex, dg_in

    def _rkf45_integrate_scalar(self, v0, ge0, gi0, is_refractory, i_stim, h0, dt, p):
        r"""Integrate ODE system for one timestep using adaptive RKF45.

        Implements Runge-Kutta-Fehlberg 4(5) method with embedded error estimation
        and automatic step size control. Integrates from time 0 to ``dt`` with
        initial step size ``h0``, adapting step size based on local truncation error.

        Parameters
        ----------
        v0 : float
            Initial membrane potential in mV.
        ge0 : float
            Initial excitatory conductance in nS.
        gi0 : float
            Initial inhibitory conductance in nS.
        is_refractory : bool
            Whether neuron is in absolute refractory period.
        i_stim : float
            External stimulus current in pA (constant during integration).
        h0 : float
            Initial/previous step size in ms (adaptive control persists across calls).
        dt : float
            Target integration interval in ms.
        p : dict
            Model parameters (see ``_dynamics_scalar`` for details).

        Returns
        -------
        v : float
            Final membrane potential in mV.
        ge : float
            Final excitatory conductance in nS.
        gi : float
            Final inhibitory conductance in nS.
        h : float
            Final adaptive step size in ms (for next integration call).

        Notes
        -----
        **Algorithm**: RKF45 uses 6 function evaluations per step to compute
        4th-order and 5th-order solutions. Error is estimated as
        ``err = max(|y5 - y4|)`` across all variables.

        **Step Size Control**:

        * Accept step if ``err <= ATOL`` (1e-3) or ``h <= MIN_H`` (1e-8 ms)
        * On accept: increase ``h`` by factor ``0.9 * (ATOL/err)^0.2`` (capped at 5x)
        * On reject: decrease ``h`` by factor ``0.9 * (ATOL/err)^0.25`` (min 0.2x)
        * Step size is clamped to ``[MIN_H, dt - t]``

        **Failure Handling**:

        * If error exceeds tolerance but step is at minimum size, step is accepted
          anyway (silent accuracy degradation)
        * If iteration count exceeds ``MAX_ITERS`` (10000), integration stops at
          current time (may not reach ``t=dt``)
        * No warnings or exceptions are raised (matches NEST behavior)

        **Computational Cost**: Typically 1-10 RKF45 steps per simulation timestep
        for standard neuron parameters. Stiff dynamics may require more steps.

        References
        ----------
        .. [1] Fehlberg E (1969). Low-order classical Runge-Kutta formulas with
               stepsize control. NASA Technical Report R-315.
        """
        t = 0.0
        h = max(h0, self._MIN_H)
        v, ge, gi = v0, ge0, gi0
        iters = 0

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = min(h, dt - t)
            h = max(h, self._MIN_H)

            def f(y1, y2, y3):
                return self._dynamics_scalar(y1, y2, y3, is_refractory, i_stim, p)

            k1 = f(v, ge, gi)
            y2 = (v + h * k1[0] / 4.0, ge + h * k1[1] / 4.0, gi + h * k1[2] / 4.0)
            k2 = f(*y2)
            y3 = (
                v + h * (3.0 * k1[0] / 32.0 + 9.0 * k2[0] / 32.0),
                ge + h * (3.0 * k1[1] / 32.0 + 9.0 * k2[1] / 32.0),
                gi + h * (3.0 * k1[2] / 32.0 + 9.0 * k2[2] / 32.0),
            )
            k3 = f(*y3)
            y4 = (
                v + h * (1932.0 * k1[0] / 2197.0 - 7200.0 * k2[0] / 2197.0 + 7296.0 * k3[0] / 2197.0),
                ge + h * (1932.0 * k1[1] / 2197.0 - 7200.0 * k2[1] / 2197.0 + 7296.0 * k3[1] / 2197.0),
                gi + h * (1932.0 * k1[2] / 2197.0 - 7200.0 * k2[2] / 2197.0 + 7296.0 * k3[2] / 2197.0),
            )
            k4 = f(*y4)
            y5 = (
                v + h * (439.0 * k1[0] / 216.0 - 8.0 * k2[0] + 3680.0 * k3[0] / 513.0 - 845.0 * k4[0] / 4104.0),
                ge + h * (439.0 * k1[1] / 216.0 - 8.0 * k2[1] + 3680.0 * k3[1] / 513.0 - 845.0 * k4[1] / 4104.0),
                gi + h * (439.0 * k1[2] / 216.0 - 8.0 * k2[2] + 3680.0 * k3[2] / 513.0 - 845.0 * k4[2] / 4104.0),
            )
            k5 = f(*y5)
            y6 = (
                v + h * (-8.0 * k1[0] / 27.0 + 2.0 * k2[0] - 3544.0 * k3[0] / 2565.0 + 1859.0 * k4[0] / 4104.0 - 11.0 * k5[0] / 40.0),
                ge + h * (-8.0 * k1[1] / 27.0 + 2.0 * k2[1] - 3544.0 * k3[1] / 2565.0 + 1859.0 * k4[1] / 4104.0 - 11.0 * k5[1] / 40.0),
                gi + h * (-8.0 * k1[2] / 27.0 + 2.0 * k2[2] - 3544.0 * k3[2] / 2565.0 + 1859.0 * k4[2] / 4104.0 - 11.0 * k5[2] / 40.0),
            )
            k6 = f(*y6)

            y4_sol = (
                v + h * (25.0 * k1[0] / 216.0 + 1408.0 * k3[0] / 2565.0 + 2197.0 * k4[0] / 4104.0 - k5[0] / 5.0),
                ge + h * (25.0 * k1[1] / 216.0 + 1408.0 * k3[1] / 2565.0 + 2197.0 * k4[1] / 4104.0 - k5[1] / 5.0),
                gi + h * (25.0 * k1[2] / 216.0 + 1408.0 * k3[2] / 2565.0 + 2197.0 * k4[2] / 4104.0 - k5[2] / 5.0),
            )
            y5_sol = (
                v + h * (16.0 * k1[0] / 135.0 + 6656.0 * k3[0] / 12825.0 + 28561.0 * k4[0] / 56430.0 - 9.0 * k5[0] / 50.0 + 2.0 * k6[0] / 55.0),
                ge + h * (16.0 * k1[1] / 135.0 + 6656.0 * k3[1] / 12825.0 + 28561.0 * k4[1] / 56430.0 - 9.0 * k5[1] / 50.0 + 2.0 * k6[1] / 55.0),
                gi + h * (16.0 * k1[2] / 135.0 + 6656.0 * k3[2] / 12825.0 + 28561.0 * k4[2] / 56430.0 - 9.0 * k5[2] / 50.0 + 2.0 * k6[2] / 55.0),
            )

            err = max(
                abs(y5_sol[0] - y4_sol[0]),
                abs(y5_sol[1] - y4_sol[1]),
                abs(y5_sol[2] - y4_sol[2]),
            )

            if err <= self._ATOL or h <= self._MIN_H:
                v, ge, gi = y5_sol
                t += h
                if err == 0.0:
                    fac = 5.0
                else:
                    fac = 0.9 * (self._ATOL / err) ** 0.2
                    fac = min(5.0, max(0.2, fac))
                h = max(self._MIN_H, h * fac)
            else:
                fac = 0.9 * (self._ATOL / err) ** 0.25
                fac = min(1.0, max(0.2, fac))
                h = max(self._MIN_H, h * fac)

        return v, ge, gi, h

    def update(self, x=0. * u.pA):
        r"""Advance neuron dynamics by one simulation timestep.

        Integrates membrane potential and synaptic conductances using adaptive RKF45,
        applies synaptic input increments, handles spike emission and reset, and
        stores external current for the next step. This method implements the complete
        NEST update cycle.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input for the **next** timestep (one-step delay buffer).
            Must have current units (pA). Can be scalar (broadcast to all neurons)
            or array with shape matching ``in_size``. Default: 0 pA

        Returns
        -------
        ArrayLike
            Differentiable spike output with shape ``(\*in_size, \*batch_shape)``.
            Values are computed via surrogate gradient function from pre-reset
            membrane potentials. Binary spike detection uses ``V >= V_th`` threshold.

        Notes
        -----
        **Update Order (matches NEST)**:

        1. Integrate ODE system on interval :math:`(t, t+dt]` using adaptive RKF45
        2. Add synaptic conductance jumps from ``delta_inputs`` (presynaptic spikes)
        3. Check refractory status:

           * If refractory: decrement counter, clamp voltage to reset
           * If not refractory and V ≥ V_th: emit spike, reset voltage, start refractory
           * Otherwise: continue integration

        4. Store input current ``x`` in buffer ``I_stim`` for next step
        5. Update ``last_spike_time`` for neurons that spiked
        6. Return surrogate spike signal from pre-reset voltages

        **Implementation Details**:

        * RKF45 integrator uses per-neuron adaptive step size stored in ``integration_step``
        * Integration is performed via scalar loop (``np.ndindex``) for per-neuron
          adaptive control
        * Synaptic conductance jumps are split by sign: positive → ``g_ex``, negative → ``g_in``
        * Refractory neurons have clamped voltage but conductances continue decaying
        * Current input delay matches NEST ring-buffer semantics

        **Failure Modes**:

        * Integration may degrade to minimum step size (1e-8 ms) for stiff dynamics
        * Iteration limit (10000) may be reached for extreme parameter combinations
        * No warning is issued on integration failure (matches NEST silent fallback)

        Examples
        --------
        Single timestep update with external current:

        .. code-block:: python

            >>> import brainunit as u
            >>> spike = neuron.update(x=100 * u.pA)

        Simulate over 1 second with constant input:

        .. code-block:: python

            >>> import brainstate as bst
            >>> with bst.environ.context(dt=0.1*u.ms):
            ...     neuron.init_all_states()
            ...     spikes = []
            ...     for _ in range(10000):
            ...         spikes.append(neuron.update(x=200*u.pA))
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        ge = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape)
        gi = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape)

        p = {
            'V_th': self._broadcast_to_state(self._to_numpy(self.V_th, u.mV), v_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'E_ex': self._broadcast_to_state(self._to_numpy(self.E_ex, u.mV), v_shape),
            'E_in': self._broadcast_to_state(self._to_numpy(self.E_in, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'tau_syn_ex': self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape),
            'tau_syn_in': self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
        }
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )

        dg_ex_q, dg_in_q = self._sum_signed_delta_inputs()
        dg_ex = self._broadcast_to_state(self._to_numpy(dg_ex_q, u.nS), v_shape)
        dg_in = self._broadcast_to_state(self._to_numpy(dg_in_q, u.nS), v_shape)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        v_for_spike = np.empty_like(V)
        spike_mask = np.zeros_like(V, dtype=bool)
        V_next = np.empty_like(V)
        ge_next = np.empty_like(ge)
        gi_next = np.empty_like(gi)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            is_refractory = r[idx] > 0
            v_i, ge_i, gi_i, h_i = self._rkf45_integrate_scalar(
                V[idx], ge[idx], gi[idx], is_refractory, i_stim[idx], h_int[idx], dt, local_p
            )

            ge_i += dg_ex[idx]
            gi_i += dg_in[idx]

            if is_refractory:
                v_for_spike[idx] = local_p['V_reset']
                v_i = local_p['V_reset']
                r_i = r[idx] - 1
            else:
                v_for_spike[idx] = v_i
                if v_i >= local_p['V_th']:
                    spike_mask[idx] = True
                    v_i = local_p['V_reset']
                    r_i = refr_counts[idx]
                else:
                    r_i = 0

            V_next[idx] = v_i
            ge_next[idx] = ge_i
            gi_next[idx] = gi_i
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.g_ex.value = ge_next * u.nS
        self.g_in.value = gi_next * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float64) * u.mV)
