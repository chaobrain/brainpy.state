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

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'iaf_cond_alpha',
]


class iaf_cond_alpha(Neuron):
    r"""Leaky integrate-and-fire model with alpha-shaped conductance synapses.

    Description
    -----------

    ``iaf_cond_alpha`` is a conductance-based leaky integrate-and-fire neuron with

    * hard threshold,
    * fixed absolute refractory period,
    * alpha-shaped excitatory and inhibitory synaptic conductances (second-order kinetics),
    * no adaptation variables.

    This implementation follows NEST ``iaf_cond_alpha`` dynamics and update order,
    using NEST C++ model behavior as the source of truth.

    **1. Membrane Potential and Synaptic Currents**

    The membrane potential evolves according to

    .. math::

       \frac{dV_\mathrm{m}}{dt} =
       \frac{-g_\mathrm{L}(V_\mathrm{m}-E_\mathrm{L})
             - I_\mathrm{syn}
             + I_\mathrm{e}
             + I_\mathrm{stim}}
            {C_\mathrm{m}}

    with

    .. math::

       I_\mathrm{syn}
       = I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}}
       = g_\mathrm{ex}(V_\mathrm{m}-E_\mathrm{ex})
       + g_\mathrm{in}(V_\mathrm{m}-E_\mathrm{in}) .

    **2. Alpha-Shaped Conductance Kinetics**

    Alpha conductances use two coupled state variables per channel:

    .. math::

       \frac{d\,dg_\mathrm{ex}}{dt} = -\frac{dg_\mathrm{ex}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{d g_\mathrm{ex}}{dt}
       = dg_\mathrm{ex} - \frac{g_\mathrm{ex}}{\tau_{\mathrm{syn,ex}}},

    .. math::

       \frac{d\,dg_\mathrm{in}}{dt} = -\frac{dg_\mathrm{in}}{\tau_{\mathrm{syn,in}}},
       \qquad
       \frac{d g_\mathrm{in}}{dt}
       = dg_\mathrm{in} - \frac{g_\mathrm{in}}{\tau_{\mathrm{syn,in}}}.

    A presynaptic spike with weight :math:`w` causes an instantaneous jump at
    the end of the simulation step. Positive/negative weights map to
    excitatory/inhibitory channels:

    .. math::

       w > 0 \Rightarrow dg_\mathrm{ex} \leftarrow dg_\mathrm{ex} + \frac{e}{\tau_{\mathrm{syn,ex}}} w,

    .. math::

       w < 0 \Rightarrow dg_\mathrm{in} \leftarrow dg_\mathrm{in} + \frac{e}{\tau_{\mathrm{syn,in}}} |w|.

    The normalization factor :math:`e/\tau` ensures the conductance peak matches
    the weight magnitude (in nS).

    **3. Spike Emission and Refractory Mechanism**

    A spike is emitted when :math:`V_\mathrm{m} \ge V_\mathrm{th}` at the end of
    a simulation step. On spike:

    * :math:`V_\mathrm{m}` is reset to :math:`V_\mathrm{reset}`,
    * refractory counter is set to :math:`\lceil t_\mathrm{ref}/dt \rceil`,
    * spike time is recorded as :math:`t + dt`.

    During absolute refractory period:

    * effective membrane potential in current computation is clamped to :math:`V_\mathrm{reset}`,
    * :math:`dV_\mathrm{m}/dt = 0`,
    * conductances continue to decay.

    **4. Numerical Integration and Update Order**

    NEST integrates this model with adaptive RKF45. This implementation mirrors
    that behavior with an RKF45(4,5) integrator and persistent internal step size.
    The discrete-time update order is:

    1. Integrate continuous dynamics on :math:`(t, t+dt]` using RKF45 with adaptive substeps.
    2. Apply refractory countdown / threshold test / reset and spike emission.
    3. Add synaptic conductance jumps from spike inputs arriving this step.
    4. Store external current input as :math:`I_\mathrm{stim}` for the next step.

    The one-step delayed application of current input (``I_stim`` buffer) is
    intentional and matches NEST's ring-buffer update semantics.

    Parameters
    ----------
    in_size : tuple of int or int
        Shape of the neuron population. Can be an integer for 1D populations or
        a tuple for multi-dimensional populations.
    E_L : ArrayLike, optional
        Leak reversal potential :math:`E_\mathrm{L}`. Default: -70 mV.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_\mathrm{m}`. Must be strictly positive.
        Default: 250 pF.
    t_ref : ArrayLike, optional
        Absolute refractory period :math:`t_\mathrm{ref}`. Must be non-negative.
        Default: 2 ms.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_\mathrm{th}`. Must be larger than ``V_reset``.
        Default: -55 mV.
    V_reset : ArrayLike, optional
        Reset potential :math:`V_\mathrm{reset}`. Must be smaller than ``V_th``.
        Default: -60 mV.
    E_ex : ArrayLike, optional
        Excitatory reversal potential :math:`E_\mathrm{ex}`. Default: 0 mV.
    E_in : ArrayLike, optional
        Inhibitory reversal potential :math:`E_\mathrm{in}`. Default: -85 mV.
    g_L : ArrayLike, optional
        Leak conductance :math:`g_\mathrm{L}`. Must be strictly positive.
        Default: 16.6667 nS (yields :math:`\tau_\mathrm{m} = 15` ms with default ``C_m``).
    tau_syn_ex : ArrayLike, optional
        Excitatory alpha time constant :math:`\tau_{\mathrm{syn,ex}}`. Must be
        strictly positive. Default: 0.2 ms.
    tau_syn_in : ArrayLike, optional
        Inhibitory alpha time constant :math:`\tau_{\mathrm{syn,in}}`. Must be
        strictly positive. Default: 2.0 ms.
    I_e : ArrayLike, optional
        Constant external current :math:`I_\mathrm{e}`. Default: 0 pA.
    V_initializer : Callable, optional
        Initializer for membrane potential. Default: Constant(-70 mV).
    g_ex_initializer : Callable, optional
        Initializer for excitatory conductance. Default: Constant(0 nS).
    g_in_initializer : Callable, optional
        Initializer for inhibitory conductance. Default: Constant(0 nS).
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation (differentiable approximation).
        Default: ReluGrad().
    spk_reset : str, optional
        Spike reset mode. ``'hard'`` uses stop_gradient (matches NEST behavior),
        ``'soft'`` allows gradients through reset. Default: ``'hard'``.
    ref_var : bool, optional
        If True, expose ``refractory`` state variable as boolean indicator.
        Default: False.
    name : str, optional
        Name of the neuron group.

    Parameter Mapping
    -----------------

    ==================== ================== ========================================
    **Parameter**        **Default**        **Math equivalent**
    ==================== ================== ========================================
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
    ==================== ================== ========================================

    State Variables
    ---------------

    ========================= ================================================================
    **State variable**        **Description**
    ========================= ================================================================
    ``V``                     Membrane potential :math:`V_\mathrm{m}`
    ``dg_ex``                 Excitatory alpha auxiliary state
    ``g_ex``                  Excitatory conductance :math:`g_\mathrm{ex}`
    ``dg_in``                 Inhibitory alpha auxiliary state
    ``g_in``                  Inhibitory conductance :math:`g_\mathrm{in}`
    ``last_spike_time``       Last spike time (recorded at :math:`t+dt`)
    ``refractory_step_count`` Remaining refractory grid steps
    ``integration_step``      Internal RKF45 step-size state (persistent)
    ``I_stim``                Buffered current applied in next step
    ``refractory``            Optional boolean refractory indicator (if ``ref_var=True``)
    ========================= ================================================================

    Sends
    -----
    ``SpikeEvent`` (conceptually; represented as returned spike tensor from ``update``).

    Receives
    --------
    - Signed spike-weight conductance increments through ``add_delta_input``.
    - External current input through ``x`` in :meth:`update` (one-step delayed).

    Raises
    ------
    ValueError
        If ``V_reset >= V_th``, ``C_m <= 0``, ``t_ref < 0``, or any time constants
        are non-positive.

    Notes
    -----

    - Defaults follow NEST C++ source for ``iaf_cond_alpha`` (``models/iaf_cond_alpha.h/.cpp``).
    - Synaptic spike weights are interpreted in conductance units (nS), with
      positive/negative sign selecting excitatory/inhibitory channel.
    - The alpha shape produces a smoother conductance transient than single exponentials,
      peaking at :math:`t = \tau` after a spike.
    - During refractory period, the effective voltage used for current computation is
      clamped, but the actual ``V`` state continues to be updated (remains at reset value).

    References
    ----------
    .. [1] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. Journal of Computational Neuroscience,
           16:159-175. DOI: https://doi.org/10.1023/B:JCNS.0000014108.03012.81
    .. [2] Bernander O, Douglas RJ, Martin KAC, Koch C (1991). Synaptic
           background activity influences spatiotemporal integration in single
           pyramidal cells. PNAS, 88(24):11569-11573.
           DOI: https://doi.org/10.1073/pnas.88.24.11569
    .. [3] Kuhn A, Rotter S (2004). Neuronal integration of synaptic input in
           the fluctuation-driven regime. Journal of Neuroscience, 24(10):2345-2356.
           DOI: https://doi.org/10.1523/JNEUROSCI.3349-03.2004
    .. [4] NEST Simulator ``iaf_cond_alpha`` model documentation and C++ source:
           ``models/iaf_cond_alpha.h`` and ``models/iaf_cond_alpha.cpp``.

    See Also
    --------
    iaf_cond_exp : Conductance-based LIF with exponential synapses
    iaf_psc_alpha : Current-based LIF with alpha synapses
    iaf_psc_delta : Current-based LIF with delta synapses

    Examples
    --------
    Create a population of 100 conductance-based neurons:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
        >>> neurons = bst.iaf_cond_alpha(
        ...     in_size=100,
        ...     V_th=-50. * u.mV,
        ...     tau_syn_ex=0.5 * u.ms,
        ...     tau_syn_in=2.0 * u.ms
        ... )
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
            raise ValueError('All time constants must be strictly positive.')

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize all state variables.

        Creates and initializes membrane potential, conductance states, refractory
        counters, integration step size, and optional refractory indicator.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for state variables. If None, states are initialized
            without batch dimension.
        **kwargs : dict
            Additional keyword arguments (unused, for API compatibility).

        Notes
        -----
        - ``V``, ``g_ex``, ``g_in`` are initialized using their respective initializers.
        - ``dg_ex``, ``dg_in`` (alpha auxiliary states) are initialized to zero.
        - ``last_spike_time`` is set to large negative value (-1e7 ms).
        - ``refractory_step_count`` starts at 0 (not in refractory period).
        - ``integration_step`` is initialized to the global timestep ``dt``.
        - ``I_stim`` buffer starts at 0 pA.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))

        self.V = brainstate.HiddenState(V)
        self.dg_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.g_ex = brainstate.HiddenState(g_ex)
        self.dg_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
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
        """Reset all state variables to their initial values.

        Re-initializes all states without recreating the state objects themselves.
        Useful for resetting between simulation runs or trials.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for state variables. If None, uses existing shape.
        **kwargs : dict
            Additional keyword arguments (unused, for API compatibility).

        Notes
        -----
        This method reuses the existing state containers (``HiddenState``,
        ``ShortTermState``) and only updates their ``.value`` attributes.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.g_ex.value = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        self.g_in.value = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(self.V.value / u.mV))
        self.dg_ex.value = np.asarray(zeros, dtype=np.float64)
        self.dg_in.value = np.asarray(zeros, dtype=np.float64)
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

        Applies the surrogate spike function to a normalized voltage to produce
        a continuous approximation of spike events suitable for gradient-based learning.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential to evaluate. If None, uses current ``self.V.value``.
            Shape must match neuron population shape.

        Returns
        -------
        ArrayLike
            Spike output in [0, 1], where values close to 1 indicate spike events.
            Shape matches input voltage shape.

        Notes
        -----
        The voltage is normalized to :math:`(V - V_\mathrm{th}) / (V_\mathrm{th} - V_\mathrm{reset})`
        before applying the surrogate function. This makes the surrogate function
        operate in a standardized range regardless of absolute voltage values.
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
    def _dynamics_scalar(v, dg_ex, g_ex, dg_in, g_in, is_refractory, i_stim, p):
        """Compute right-hand side of ODEs for a single neuron.

        Evaluates derivatives for membrane potential and alpha conductance states
        according to NEST semantics (voltage clamping during refractory period,
        voltage threshold clamping otherwise).

        Parameters
        ----------
        v : float
            Membrane potential (mV, unitless).
        dg_ex : float
            Excitatory alpha auxiliary state (nS/ms, unitless).
        g_ex : float
            Excitatory conductance (nS, unitless).
        dg_in : float
            Inhibitory alpha auxiliary state (nS/ms, unitless).
        g_in : float
            Inhibitory conductance (nS, unitless).
        is_refractory : bool
            True if neuron is in absolute refractory period.
        i_stim : float
            External stimulus current (pA, unitless).
        p : dict
            Parameter dictionary with keys: ``V_th``, ``V_reset``, ``E_L``, ``E_ex``,
            ``E_in``, ``C_m``, ``g_L``, ``tau_syn_ex``, ``tau_syn_in``, ``I_e``.
            All values are scalar floats (unitless).

        Returns
        -------
        tuple of float
            (dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt) — derivatives for all five
            state variables.

        Notes
        -----
        - During refractory period: ``v_eff = V_reset``, ``dv = 0``.
        - Otherwise: ``v_eff = min(v, V_th)`` (threshold clamping), ``dv`` computed normally.
        - Conductances always decay regardless of refractory state.
        """
        v_eff = p['V_reset'] if is_refractory else min(v, p['V_th'])

        i_syn_exc = g_ex * (v_eff - p['E_ex'])
        i_syn_inh = g_in * (v_eff - p['E_in'])
        i_leak = p['g_L'] * (v_eff - p['E_L'])
        dv = 0.0 if is_refractory else (
            -i_leak - i_syn_exc - i_syn_inh + i_stim + p['I_e']
        ) / p['C_m']

        ddg_ex = -dg_ex / p['tau_syn_ex']
        dg_ex_dt = dg_ex - (g_ex / p['tau_syn_ex'])
        ddg_in = -dg_in / p['tau_syn_in']
        dg_in_dt = dg_in - (g_in / p['tau_syn_in'])
        return dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt

    def _rkf45_integrate_scalar(self, v0, dg_ex0, g_ex0, dg_in0, g_in0, is_refractory, i_stim, h0, dt, p):
        """Integrate ODEs for one neuron over time interval [0, dt] using RKF45.

        Performs adaptive-step Runge-Kutta-Fehlberg 4(5) integration with error
        control and step size adjustment. Matches NEST's GSL-based integrator behavior.

        Parameters
        ----------
        v0 : float
            Initial membrane potential (mV, unitless).
        dg_ex0 : float
            Initial excitatory alpha auxiliary state (nS/ms, unitless).
        g_ex0 : float
            Initial excitatory conductance (nS, unitless).
        dg_in0 : float
            Initial inhibitory alpha auxiliary state (nS/ms, unitless).
        g_in0 : float
            Initial inhibitory conductance (nS, unitless).
        is_refractory : bool
            True if neuron is in absolute refractory period.
        i_stim : float
            External stimulus current (pA, unitless).
        h0 : float
            Initial candidate step size (ms, unitless).
        dt : float
            Total integration interval (ms, unitless).
        p : dict
            Parameter dictionary (see :meth:`_dynamics_scalar`).

        Returns
        -------
        v_final : float
            Final membrane potential.
        dg_ex_final : float
            Final excitatory auxiliary state.
        g_ex_final : float
            Final excitatory conductance.
        dg_in_final : float
            Final inhibitory auxiliary state.
        g_in_final : float
            Final inhibitory conductance.
        h_final : float
            Final step size (preserved for next call to maintain adaptive behavior).

        Notes
        -----
        - Uses RKF45 embedded pair: 4th-order solution for advancement, 5th-order
          for error estimation.
        - Step size is adjusted based on absolute error tolerance (``_ATOL``).
        - Minimum step size (``_MIN_H``) prevents infinite loops on stiff equations.
        - Maximum iterations (``_MAX_ITERS``) safeguards against non-convergence.
        - The persistent step size ``h_final`` is stored and reused across timesteps
          to maintain adaptive efficiency.
        """
        t = 0.0
        h = max(h0, self._MIN_H)
        y = np.asarray([v0, dg_ex0, g_ex0, dg_in0, g_in0], dtype=np.float64)
        iters = 0

        def f(y_):
            return np.asarray(
                self._dynamics_scalar(
                    y_[0], y_[1], y_[2], y_[3], y_[4], is_refractory, i_stim, p
                ),
                dtype=np.float64
            )

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = max(self._MIN_H, min(h, dt - t))

            k1 = f(y)
            k2 = f(y + h * (1.0 / 4.0) * k1)
            k3 = f(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
            k4 = f(y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0))
            k5 = f(y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0))
            k6 = f(y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0))

            y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
            y5 = y + h * (16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0)
            err = float(np.max(np.abs(y5 - y4)))

            if err <= self._ATOL or h <= self._MIN_H:
                y = y5
                t += h
                fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (self._ATOL / err) ** 0.2))
                h = max(self._MIN_H, h * fac)
            else:
                fac = min(1.0, max(0.2, 0.9 * (self._ATOL / err) ** 0.25))
                h = max(self._MIN_H, h * fac)

        return y[0], y[1], y[2], y[3], y[4], h

    def update(self, x=0. * u.pA):
        """Advance neuron state by one simulation timestep.

        Integrates ODEs, handles refractory period and spike emission, applies
        synaptic conductance jumps, and buffers external current for next step.
        This method implements the full NEST update semantics.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input for this timestep (pA). Broadcasted to population
            shape. This input is buffered and applied in the *next* timestep (one-step
            delay) to match NEST ring-buffer semantics. Default: 0 pA.

        Returns
        -------
        ArrayLike
            Differentiable spike output (values in [0, 1], shape matching population).
            Computed using surrogate gradient on pre-reset membrane potential.

        Notes
        -----
        **Update order** (matching NEST):

        1. **Integrate ODEs**: Use RKF45 to advance ``V``, ``dg_ex``, ``g_ex``,
           ``dg_in``, ``g_in`` over ``(t, t+dt]`` with ``I_stim`` from previous step.
        2. **Refractory/spike handling**:
           - If in refractory period: clamp ``V`` to ``V_reset``, decrement counter.
           - Else if ``V >= V_th``: emit spike, reset ``V`` to ``V_reset``, set
             refractory counter.
        3. **Apply synaptic inputs**: Add conductance jumps from incoming spikes
           (via ``add_delta_input``) to ``dg_ex`` / ``dg_in`` with alpha normalization.
        4. **Buffer current input**: Store ``x`` into ``I_stim`` for next timestep.

        The surrogate spike is computed from the *pre-reset* voltage to allow gradient
        flow through spike events during training.

        **Failure modes**: If integration does not converge within ``_MAX_ITERS``
        iterations, the final state may be inaccurate. Reduce global ``dt`` or check
        for extreme parameter values if this occurs.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        dg_ex = self._broadcast_to_state(np.asarray(self.dg_ex.value, dtype=np.float64), v_shape)
        g_ex = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape)
        dg_in = self._broadcast_to_state(np.asarray(self.dg_in.value, dtype=np.float64), v_shape)
        g_in = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            v_shape
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
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )

        w_ex_q, w_in_q = self._sum_signed_delta_inputs()
        w_ex = self._broadcast_to_state(self._to_numpy(w_ex_q, u.nS), v_shape)
        w_in = self._broadcast_to_state(self._to_numpy(w_in_q, u.nS), v_shape)
        pscon_ex = self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        pscon_in = self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        v_for_spike = np.empty_like(V)
        spike_mask = np.zeros_like(V, dtype=bool)
        V_next = np.empty_like(V)
        dg_ex_next = np.empty_like(dg_ex)
        g_ex_next = np.empty_like(g_ex)
        dg_in_next = np.empty_like(dg_in)
        g_in_next = np.empty_like(g_in)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            is_refractory = r[idx] > 0
            v_i, dg_ex_i, g_ex_i, dg_in_i, g_in_i, h_i = self._rkf45_integrate_scalar(
                V[idx], dg_ex[idx], g_ex[idx], dg_in[idx], g_in[idx], is_refractory,
                i_stim[idx], h_int[idx], dt, local_p
            )

            # NEST ordering: refractory/spike handling before applying spike inputs.
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

            dg_ex_i += pscon_ex[idx] * w_ex[idx]
            dg_in_i += pscon_in[idx] * w_in[idx]

            V_next[idx] = v_i
            dg_ex_next[idx] = dg_ex_i
            g_ex_next[idx] = g_ex_i
            dg_in_next[idx] = dg_in_i
            g_in_next[idx] = g_in_i
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.dg_ex.value = dg_ex_next
        self.g_ex.value = g_ex_next * u.nS
        self.dg_in.value = dg_in_next
        self.g_in.value = g_in_next * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float32) * u.mV)
