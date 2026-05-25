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
from ._utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'iaf_cond_exp',
]


class iaf_cond_exp(NESTNeuron):
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
    gsl_error_tol : ArrayLike
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
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


    Parameter Mapping
    -----------------

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
    ``gsl_error_tol``    1e-3               —
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
        Membrane potential :math:`V_\mathrm{m}` in mV, shape ``(*in_size)``.
    g_ex : brainstate.HiddenState
        Excitatory synaptic conductance :math:`g_\mathrm{ex}` in nS,
        shape ``(*in_size)``.
    g_in : brainstate.HiddenState
        Inhibitory synaptic conductance :math:`g_\mathrm{in}` in nS,
        shape ``(*in_size)``.
    last_spike_time : brainstate.ShortTermState
        Last spike emission time in ms, shape ``(*in_size)``.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory time steps (int32), shape ``(*in_size)``.
    integration_step : brainstate.ShortTermState
        Internal RKF45 adaptive step size in ms, shape ``(*in_size)``.
    I_stim : brainstate.ShortTermState
        Buffered external current (one-step delayed) in pA, shape ``(*in_size)``.
    refractory : brainstate.ShortTermState, optional
        Boolean refractory state indicator, shape ``(*in_size)``.
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
    ValueError
        If ``gsl_error_tol <= 0`` (error tolerance must be strictly positive).

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
        >>> import saiunit as u
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

    _MIN_H = 1e-8 * u.ms  # ms
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
        gsl_error_tol: ArrayLike = 1e-3,
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
        self.gsl_error_tol = gsl_error_tol

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
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
            If parameter inequalities or positivity constraints are violated.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_reset, self.C_m)):
            return
        if cond_any(self.V_reset >= self.V_th):
            raise ValueError('Reset potential must be smaller than threshold.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms) or cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('All synaptic time constants must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Creates and registers state variables for membrane potential, synaptic
        conductances, refractory tracking, RKF45 integration, and buffered currents.
        All states are initialized using the configured initializer functions.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        * State variables are registered as ``brainstate.HiddenState`` (continuous
          dynamics) or ``brainstate.ShortTermState`` (discrete/reset behavior).
        * ``last_spike_time`` is initialized to -1e7 ms (far past) to indicate no
          prior spikes.
        * ``integration_step`` is initialized to the simulation timestep ``dt``.
        * If ``ref_var=True``, an additional boolean ``refractory`` state is created.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape)

        self.V = brainstate.HiddenState(V)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

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

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, g_ex, g_in — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim — mutable
            auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, u.math.minimum(state.V, self.V_th))

        i_syn_exc = state.g_ex * (v_eff - self.E_ex)
        i_syn_inh = state.g_in * (v_eff - self.E_in)
        i_leak = self.g_L * (v_eff - self.E_L)

        dV_raw = (-i_leak - i_syn_exc - i_syn_inh + self.I_e + extra.i_stim) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        dg_ex = -state.g_ex / self.tau_syn_ex
        dg_in = -state.g_in / self.tau_syn_in

        return DotDict(V=dV, g_ex=dg_ex, g_in=dg_in)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, g_ex, g_in — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & (state.V < -1e3 * u.mV)
        )

        refr_accept = accept & (extra.r > 0)
        new_V = u.math.where(refr_accept, self.V_reset, state.V)

        spike_now = accept & (extra.r <= 0) & (new_V >= self.V_th)
        spike_mask = extra.spike_mask | spike_now
        new_V = u.math.where(spike_now, self.V_reset, new_V)
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict({**state, 'V': new_V})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

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
            Binary spike tensor with dtype ``jnp.float64`` and shape
            ``self.V.value.shape``. A value of ``1.0`` indicates at least one
            internal spike event occurred during the integrated interval
            :math:`(t, t+dt]`.

        Raises
        ------
        ValueError
            If RKF45 integration enters a guarded unstable regime
            (``V < -1e3 mV``), indicating divergent dynamics for the
            current parameter/input regime.

        Notes
        -----
        Integration is performed with an adaptive vectorized RKF45 loop,
        including in-loop spike/reset events and optional multiple spikes
        per step. All arithmetic is unit-aware via ``saiunit.math``.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        g_ex = self.g_ex.value  # nS
        g_in = self.g_in.value  # nS
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, g_ex=g_ex, g_in=g_in)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, g_ex, g_in = ode_state.V, ode_state.g_ex, ode_state.g_in
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in iaf_cond_exp dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration).
        w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
        w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')

        # Apply synaptic spike inputs (direct conductance jump for exponential synapses).
        g_ex = g_ex + w_ex
        g_in = g_in + w_in

        # Write back state.
        self.V.value = V
        self.g_ex.value = g_ex
        self.g_in.value = g_in
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
