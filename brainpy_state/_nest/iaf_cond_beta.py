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
    'iaf_cond_beta',
]


class iaf_cond_beta(NESTNeuron):
    r"""NEST-compatible conductance-based leaky integrate-and-fire neuron with beta-shaped synaptic conductances.

    This model implements a conductance-based LIF neuron with beta-function (dual-exponential)
    synaptic conductances for both excitatory and inhibitory channels. It follows NEST's
    ``iaf_cond_beta`` implementation, including hard threshold crossing, absolute refractory
    period, and one-step delayed external current buffering.

    **1. Mathematical Model**

    The membrane voltage :math:`V_m` evolves according to:

    .. math::

       C_m \frac{dV_m}{dt} = -I_\mathrm{leak} - I_{\mathrm{syn,ex}} - I_{\mathrm{syn,in}} + I_e + I_\mathrm{stim}

    where the currents are defined as:

    .. math::

       I_\mathrm{leak} &= g_L (V_m - E_L) \\
       I_{\mathrm{syn,ex}} &= g_\mathrm{ex}(t) (V_m - E_\mathrm{ex}) \\
       I_{\mathrm{syn,in}} &= g_\mathrm{in}(t) (V_m - E_\mathrm{in})

    During the refractory period, the membrane voltage is clamped to :math:`V_\mathrm{reset}`
    and :math:`dV_m/dt = 0`. Outside the refractory period, the effective voltage for
    synaptic current computation is bounded by :math:`\min(V_m, V_\mathrm{th})`.

    **2. Beta-Function Conductance Dynamics**

    Each synaptic conductance (excitatory and inhibitory) is modeled using two coupled
    state variables to produce a beta-function (rise-decay) waveform:

    .. math::

       \frac{d\,dg_\mathrm{ex}}{dt} &= -\frac{dg_\mathrm{ex}}{\tau_{\mathrm{decay,ex}}} \\
       \frac{d g_\mathrm{ex}}{dt} &= dg_\mathrm{ex} - \frac{g_\mathrm{ex}}{\tau_{\mathrm{rise,ex}}}

    .. math::

       \frac{d\,dg_\mathrm{in}}{dt} &= -\frac{dg_\mathrm{in}}{\tau_{\mathrm{decay,in}}} \\
       \frac{d g_\mathrm{in}}{dt} &= dg_\mathrm{in} - \frac{g_\mathrm{in}}{\tau_{\mathrm{rise,in}}}

    Incoming spikes cause instantaneous jumps in :math:`dg_\mathrm{ex}` or :math:`dg_\mathrm{in}`.
    Positive weights target the excitatory channel; negative weights target the inhibitory channel.
    Each spike weight (in nS) is multiplied by the beta normalization factor
    :math:`\kappa(\tau_\mathrm{rise}, \tau_\mathrm{decay})` to ensure unit weight produces
    a 1 nS peak conductance.

    The normalization factor is computed as:

    .. math::

       \kappa = \frac{1/\tau_\mathrm{rise} - 1/\tau_\mathrm{decay}}{\exp(-t_\mathrm{peak}/\tau_\mathrm{decay}) - \exp(-t_\mathrm{peak}/\tau_\mathrm{rise})}

    where :math:`t_\mathrm{peak} = \frac{\tau_\mathrm{rise} \tau_\mathrm{decay}}{\tau_\mathrm{decay} - \tau_\mathrm{rise}} \ln\left(\frac{\tau_\mathrm{decay}}{\tau_\mathrm{rise}}\right)`.

    **3. Numerical Integration**

    ODEs are integrated using the Runge-Kutta-Fehlberg (RKF45) adaptive-step method with
    embedded error control. The integrator maintains a persistent step size estimate
    (``integration_step``) across simulation steps, adjusting it based on local truncation
    error to satisfy a fixed absolute tolerance (``gsl_error_tol``).

    **4. Update Order (NEST Semantics)**

    Each simulation step executes the following operations in order:

    1. Integrate all ODEs on the interval :math:`(t, t+dt]` using RKF45.
    2. Inside integration loop: apply refractory clamp and spike/reset.
    3. After loop: decrement refractory counter once.
    4. Apply incoming spike weights to :math:`dg_\mathrm{ex}` and :math:`dg_\mathrm{in}`.
    5. Store external current input ``x`` into the delayed buffer ``I_stim`` (affects next step).

    This matches NEST's ring-buffer semantics: external currents applied at time :math:`t`
    take effect at time :math:`t + dt`.

    **5. Design Constraints and Assumptions**

    - **Refractory clamping**: During refractory period, voltage is fixed at :math:`V_\mathrm{reset}`
      and no integration occurs. NEST uses this approach for consistency with exact spike times.
    - **Beta normalization edge case**: When :math:`\tau_\mathrm{rise} \approx \tau_\mathrm{decay}`,
      the normalization factor approaches :math:`e / \tau_\mathrm{decay}` to avoid division by zero.

    Parameters
    ----------
    in_size : Size
        Population shape, specified as an integer (1D), tuple of integers (multi-dimensional),
        or brainstate Size object. Determines the shape of all state variables and parameters.
    E_L : ArrayLike, optional
        Leak reversal potential. Default: ``-70 mV``. Broadcast to ``in_size`` if scalar.
        Must have units of voltage (mV).
    C_m : ArrayLike, optional
        Membrane capacitance. Default: ``250 pF``. Broadcast to ``in_size`` if scalar.
        Must be strictly positive. Determines voltage response timescale :math:`\tau_m = C_m / g_L`.
    t_ref : ArrayLike, optional
        Absolute refractory period duration. Default: ``2 ms``. Broadcast to ``in_size`` if scalar.
        Must be non-negative. Converted to discrete grid steps via :math:`\lceil t_\mathrm{ref} / dt \rceil`.
    V_th : ArrayLike, optional
        Spike threshold voltage. Default: ``-55 mV``. Broadcast to ``in_size`` if scalar.
        Must satisfy :math:`V_\mathrm{reset} < V_\mathrm{th}`.
    V_reset : ArrayLike, optional
        Post-spike reset voltage. Default: ``-60 mV``. Broadcast to ``in_size`` if scalar.
        Must be strictly less than ``V_th``. Neuron is clamped to this value during refractory period.
    E_ex : ArrayLike, optional
        Excitatory reversal potential. Default: ``0 mV``. Broadcast to ``in_size`` if scalar.
        Typically positive (depolarizing).
    E_in : ArrayLike, optional
        Inhibitory reversal potential. Default: ``-85 mV``. Broadcast to ``in_size`` if scalar.
        Typically more negative than :math:`E_L` (hyperpolarizing).
    g_L : ArrayLike, optional
        Leak conductance. Default: ``16.6667 nS`` (yields :math:`\tau_m = 15` ms with default :math:`C_m`).
        Broadcast to ``in_size`` if scalar. Must be strictly positive.
    tau_rise_ex : ArrayLike, optional
        Excitatory conductance rise time constant. Default: ``0.2 ms``. Broadcast to ``in_size`` if scalar.
        Must be strictly positive. Smaller values produce faster rise times.
    tau_decay_ex : ArrayLike, optional
        Excitatory conductance decay time constant. Default: ``0.2 ms``. Broadcast to ``in_size`` if scalar.
        Must be strictly positive. When equal to ``tau_rise_ex``, beta function degenerates to alpha function.
    tau_rise_in : ArrayLike, optional
        Inhibitory conductance rise time constant. Default: ``2.0 ms``. Broadcast to ``in_size`` if scalar.
        Must be strictly positive. Typically slower than excitatory rise for GABA receptors.
    tau_decay_in : ArrayLike, optional
        Inhibitory conductance decay time constant. Default: ``2.0 ms``. Broadcast to ``in_size`` if scalar.
        Must be strictly positive. Determines inhibitory synaptic integration window.
    I_e : ArrayLike, optional
        Constant external current. Default: ``0 pA``. Broadcast to ``in_size`` if scalar.
        Added to membrane current at every time step.
    gsl_error_tol : ArrayLike
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
    V_initializer : Callable, optional
        Initialization function for membrane voltage. Default: ``Constant(-70 mV)``.
        Called as ``V_initializer(varshape)`` during ``init_state()``.
    g_ex_initializer : Callable, optional
        Initialization function for excitatory conductance. Default: ``Constant(0 nS)``.
        Called as ``g_ex_initializer(varshape)`` during ``init_state()``.
    g_in_initializer : Callable, optional
        Initialization function for inhibitory conductance. Default: ``Constant(0 nS)``.
        Called as ``g_in_initializer(varshape)`` during ``init_state()``.
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation. Default: ``ReluGrad()``.
        Applied to scaled voltage :math:`(V - V_\mathrm{th}) / (V_\mathrm{th} - V_\mathrm{reset})`
        to produce differentiable spike output for gradient-based learning.
    spk_reset : str, optional
        Spike reset mode. Default: ``'hard'`` (stop-gradient reset, matches NEST behavior).
        Alternative: ``'soft'`` (subtractive reset :math:`V \leftarrow V - V_\mathrm{th}`).
    ref_var : bool, optional
        If ``True``, create a boolean ``refractory`` state variable indicating refractory status.
        Default: ``False``. Useful for monitoring or conditional computations.
    name : str, optional
        Module name for debugging and visualization. Default: ``None`` (auto-generated).

    Parameter Mapping
    -----------------

    The following table maps constructor parameters to mathematical notation and NEST equivalents:

    ==================== ================== ======================================== ================================================
    **Parameter**        **Default**        **Math equivalent**                      **Description**
    ==================== ================== ======================================== ================================================
    ``in_size``          (required)         —                                        Population shape
    ``E_L``              -70 mV             :math:`E_\mathrm{L}`                     Leak reversal potential
    ``C_m``              250 pF             :math:`C_\mathrm{m}`                     Membrane capacitance
    ``t_ref``            2 ms               :math:`t_\mathrm{ref}`                   Absolute refractory duration
    ``V_th``             -55 mV             :math:`V_\mathrm{th}`                    Spike threshold
    ``V_reset``          -60 mV             :math:`V_\mathrm{reset}`                 Reset potential
    ``E_ex``             0 mV               :math:`E_\mathrm{ex}`                    Excitatory reversal potential
    ``E_in``             -85 mV             :math:`E_\mathrm{in}`                    Inhibitory reversal potential
    ``g_L``              16.6667 nS         :math:`g_\mathrm{L}`                     Leak conductance
    ``tau_rise_ex``      0.2 ms             :math:`\tau_{\mathrm{rise,ex}}`          Excitatory beta rise constant
    ``tau_decay_ex``     0.2 ms             :math:`\tau_{\mathrm{decay,ex}}`         Excitatory beta decay constant
    ``tau_rise_in``      2.0 ms             :math:`\tau_{\mathrm{rise,in}}`          Inhibitory beta rise constant
    ``tau_decay_in``     2.0 ms             :math:`\tau_{\mathrm{decay,in}}`         Inhibitory beta decay constant
    ``I_e``              0 pA               :math:`I_\mathrm{e}`                     Constant external current
    ``gsl_error_tol``    1e-6               —                                        RKF45 error tolerance
    ``V_initializer``    Constant(-70 mV)   —                                        Membrane initializer
    ``g_ex_initializer`` Constant(0 nS)     —                                        Excitatory conductance initializer
    ``g_in_initializer`` Constant(0 nS)     —                                        Inhibitory conductance initializer
    ``spk_fun``          ReluGrad()         —                                        Surrogate spike function
    ``spk_reset``        ``'hard'``         —                                        Reset mode (``'hard'`` matches NEST)
    ``ref_var``          ``False``          —                                        Expose boolean refractory indicator
    ==================== ================== ======================================== ================================================

    Raises
    ------
    ValueError
        If ``V_reset >= V_th`` (reset potential must be below threshold).
    ValueError
        If ``C_m <= 0`` (capacitance must be strictly positive).
    ValueError
        If ``t_ref < 0`` (refractory period cannot be negative).
    ValueError
        If any of ``tau_rise_ex``, ``tau_decay_ex``, ``tau_rise_in``, ``tau_decay_in`` are non-positive.

    Notes
    -----
    **State Variables**


    - ``V`` : brainstate.HiddenState
        Membrane potential :math:`V_m` with shape ``(*in_size,)`` and units mV.
    - ``dg_ex`` : brainstate.ShortTermState
        Excitatory beta auxiliary state (nS/ms).
    - ``g_ex`` : brainstate.HiddenState
        Excitatory synaptic conductance with units nS.
    - ``dg_in`` : brainstate.ShortTermState
        Inhibitory beta auxiliary state (nS/ms).
    - ``g_in`` : brainstate.HiddenState
        Inhibitory synaptic conductance with units nS.
    - ``refractory_step_count`` : brainstate.ShortTermState
        Remaining refractory grid steps (integer, dtype ``int32``). Zero when not refractory.
    - ``integration_step`` : brainstate.ShortTermState
        Persistent RKF45 internal step size with units ms. Adapted automatically for numerical stability.
    - ``I_stim`` : brainstate.ShortTermState
        One-step delayed external current buffer with units pA. Updated after ODE integration.
    - ``last_spike_time`` : brainstate.ShortTermState
        Time of last emitted spike (units ms). Set to :math:`t + dt` when spike occurs.
    - ``refractory`` : brainstate.ShortTermState (optional)
        Boolean refractory indicator. Only created if ``ref_var=True``.

    **Performance Considerations:**

    This model uses per-neuron scalar NumPy integration loops, which are significantly slower
    than vectorized JAX operations. For large populations, consider using ``iaf_cond_exp``
    or ``iaf_cond_alpha`` with vectorized exponential integrators. The RKF45 method is
    primarily intended for high-accuracy validation against NEST rather than production simulations.

    **NEST Compatibility:**

    This implementation matches NEST 3.9+ ``iaf_cond_beta`` semantics, including:

    - Beta normalization factor computation (exact formula match).
    - One-step delayed external current handling.
    - Refractory voltage clamping during integration.
    - Hard threshold crossing and immediate reset.

    Minor differences from NEST:

    - NEST uses GSL's RK integrator; this uses a pure-Python RKF45 implementation.
    - Numerical differences may appear at :math:`O(10^{-6})` due to floating-point rounding.

    Examples
    --------
    **Basic Usage:**

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import saiunit as u
        >>> import brainstate as bstate
        >>>
        >>> with bstate.environ.context(dt=0.1 * u.ms):
        ...     neuron = bst.iaf_cond_beta(10, V_th=-50*u.mV, V_reset=-65*u.mV)
        ...     neuron.init_all_states()
        ...     # Apply excitatory synaptic input (5 nS conductance jump)
        ...     neuron.add_delta_input('syn_input', 5.0 * u.nS)
        ...     spikes = neuron()
        ...     print(neuron.V.value[:3])  # Membrane voltages of first 3 neurons
        [-70. -70. -70.] mV

    **Comparing Excitatory and Inhibitory Time Constants:**

    .. code-block:: python

        >>> import matplotlib.pyplot as plt
        >>> with bstate.environ.context(dt=0.01 * u.ms):
        ...     fast_ex = bst.iaf_cond_beta(1, tau_rise_ex=0.2*u.ms, tau_decay_ex=2.0*u.ms)
        ...     slow_in = bst.iaf_cond_beta(1, tau_rise_in=2.0*u.ms, tau_decay_in=10.0*u.ms)
        ...     fast_ex.init_all_states()
        ...     slow_in.init_all_states()
        ...     # Single excitatory spike at t=1ms
        ...     fast_ex.add_delta_input('spike', 10.0 * u.nS)
        ...     # Record excitatory conductance
        ...     g_ex_trace = []
        ...     for _ in range(500):
        ...         fast_ex()
        ...         g_ex_trace.append(fast_ex.g_ex.value[0])
        ...     plt.plot(g_ex_trace)
        ...     plt.xlabel('Time (0.01 ms steps)')
        ...     plt.ylabel('g_ex (nS)')
        ...     plt.title('Beta-function conductance waveform')

    **Network with Balanced Excitation and Inhibition:**

    .. code-block:: python

        >>> from brainevent.nn import FixedProb
        >>> exc_neurons = bst.iaf_cond_beta(800, E_L=-70*u.mV, V_th=-50*u.mV)
        >>> inh_neurons = bst.iaf_cond_beta(200, E_L=-70*u.mV, V_th=-50*u.mV)
        >>> exc_neurons.init_all_states()
        >>> inh_neurons.init_all_states()
        >>> # Create projections (placeholder - requires brainevent)
        >>> # exc_proj = FixedProb(exc_neurons, exc_neurons, prob=0.1, weight=2.0*u.nS)
        >>> # inh_proj = FixedProb(inh_neurons, exc_neurons, prob=0.2, weight=-5.0*u.nS)

    See Also
    --------
    iaf_cond_alpha : LIF with alpha-function conductances (single time constant).
    iaf_cond_exp : LIF with exponential conductances (simpler, faster).
    iaf_psc_exp : Current-based LIF (no conductance dynamics).

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
    .. [4] Rotter S, Diesmann M (1999). Exact simulation of time-invariant
           linear systems with applications to neuronal modeling.
           Biological Cybernetics, 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    .. [5] Roth A, van Rossum M (2010). Chapter 6: Modeling synapses.
           In De Schutter, Computational Modeling Methods for Neuroscientists.
    """
    __module__ = 'brainpy.state'

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000
    _EPS = np.finfo(np.float64).eps

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
        tau_rise_ex: ArrayLike = 0.2 * u.ms,
        tau_decay_ex: ArrayLike = 0.2 * u.ms,
        tau_rise_in: ArrayLike = 2.0 * u.ms,
        tau_decay_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        gsl_error_tol: ArrayLike = 1e-6,
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
        self.tau_rise_ex = braintools.init.param(tau_rise_ex, self.varshape)
        self.tau_decay_ex = braintools.init.param(tau_decay_ex, self.varshape)
        self.tau_rise_in = braintools.init.param(tau_rise_in, self.varshape)
        self.tau_decay_in = braintools.init.param(tau_decay_in, self.varshape)
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

    @classmethod
    def _beta_normalization_factor_scalar(cls, tau_rise: float, tau_decay: float):
        r"""Compute beta normalization factor for scalar time constants.

        Parameters
        ----------
        tau_rise : float
            Rise time constant in ms (unitless scalar).
        tau_decay : float
            Decay time constant in ms (unitless scalar).

        Returns
        -------
        float
            Normalization factor ensuring unit weight produces 1 nS peak conductance.
        """
        tau_difference = tau_decay - tau_rise
        peak_value = 0.0
        if abs(tau_difference) > cls._EPS:
            t_peak = tau_decay * tau_rise * np.log(tau_decay / tau_rise) / tau_difference
            peak_value = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)

        if abs(peak_value) < cls._EPS:
            return np.e / tau_decay

        return (1.0 / tau_rise - 1.0 / tau_decay) / peak_value

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
        if cond_any(self.tau_rise_ex <= 0.0 * u.ms) or cond_any(
            self.tau_decay_ex <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_rise_in <= 0.0 * u.ms) or cond_any(
            self.tau_decay_in <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

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

        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape)
        V = braintools.init.param(self.V_initializer, self.varshape)
        zeros = u.math.zeros(self.varshape, dtype=V.dtype) * (u.nS / u.ms)

        self.dg_ex = brainstate.ShortTermState(zeros)
        self.dg_in = brainstate.ShortTermState(zeros)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)
        self.V = brainstate.HiddenState(V)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output using surrogate gradients.

        Scales the membrane voltage relative to threshold and reset, then applies the surrogate
        spike function to produce a continuous spike signal suitable for gradient-based learning.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane voltage to evaluate. If ``None`` (default), uses ``self.V.value``.
            Must have compatible shape with ``V_th`` and ``V_reset`` (broadcast-compatible).
            Expected units: mV (or dimensionless if consistent).

        Returns
        -------
        spike : jax.Array
            Spike output with same shape as input ``V``. Values depend on ``spk_fun`` but are
            typically in :math:`[0, 1]` for surrogate gradient functions like ``ReluGrad``.
            Higher values indicate stronger spike activation. Dtype is ``float32``.

        Notes
        -----
        The scaling formula is:

        .. math::

           \mathrm{spike} = \mathrm{spk\_fun}\left(\frac{V - V_\mathrm{th}}{V_\mathrm{th} - V_\mathrm{reset}}\right)

        This normalization ensures that when :math:`V = V_\mathrm{th}`, the scaled input is zero,
        and when :math:`V = V_\mathrm{reset}`, the scaled input is :math:`-1`. The surrogate
        function (e.g., ``ReluGrad``) produces a differentiable approximation to the Heaviside
        step function for backpropagation.

        This method is called internally by ``update()`` to generate spike outputs, but can also
        be called manually for custom spike detection logic.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, dg_ex, g_ex, dg_in, g_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim -- mutable
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

        dV_raw = (
                     -i_leak - i_syn_exc - i_syn_inh + self.I_e + extra.i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        ddg_ex = -state.dg_ex / self.tau_decay_ex
        dg_ex_dt = state.dg_ex - state.g_ex / self.tau_rise_ex
        ddg_in = -state.dg_in / self.tau_decay_in
        dg_in_dt = state.dg_in - state.g_in / self.tau_rise_in

        return DotDict(V=dV, dg_ex=ddg_ex, g_ex=dg_ex_dt, dg_in=ddg_in, g_in=dg_in_dt)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, dg_ex, g_ex, dg_in, g_in -- ODE state variables.
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
        r"""Advance neuron dynamics by one simulation time step.

        Performs the full NEST-compatible update cycle: ODE integration via RKF45, refractory
        countdown, threshold detection, spike emission, reset, synaptic input application, and
        delayed external current buffering.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input for the **next** time step (one-step delayed). Default: ``0 pA``.
            Must have shape compatible with ``(*in_size,)`` (broadcast-compatible).
            Units: pA (picoamperes). This current is stored in ``I_stim`` and takes effect at
            time :math:`t + dt`, matching NEST's ring-buffer semantics.

        Returns
        -------
        spike : jax.Array
            Binary spike output for the current time step. Shape: ``self.V.value.shape``.
            Dtype: ``float64``. Values of ``1.0`` indicate at least one internal spike
            event occurred during the integrated interval :math:`(t, t+dt]`.

        Notes
        -----
        **Update Order (NEST-compatible):**

        1. **ODE Integration**: Integrate all differential equations on :math:`(t, t+dt]` using
           the Runge-Kutta-Fehlberg (RKF45) adaptive-step method.
        2. **Refractory Handling**: Inside integration loop, apply refractory clamp and
           spike/reset events.
        3. **Refractory Decrement**: After loop, decrement refractory counter once.
        4. **Synaptic Input Application**: Sum all incoming delta inputs (spike weights), split
           by sign into excitatory (positive) and inhibitory (negative) channels, multiply by
           beta normalization factors, and add to ``dg_ex`` and ``dg_in`` states.
        5. **External Current Buffering**: Store input ``x`` plus ``sum_current_inputs()`` into
           ``I_stim`` for use in the **next** time step.

        **Spike Weight Handling:**

        - All delta inputs (registered via ``add_delta_input()``) are summed and split by sign.
        - Positive weights :math:`w_\mathrm{ex} = \max(w, 0)` are multiplied by
          :math:`\kappa(\tau_{\mathrm{rise,ex}}, \tau_{\mathrm{decay,ex}})` and added to ``dg_ex``.
        - Negative weights :math:`w_\mathrm{in} = \max(-w, 0)` are multiplied by
          :math:`\kappa(\tau_{\mathrm{rise,in}}, \tau_{\mathrm{decay,in}})` and added to ``dg_in``.
        - The beta normalization factor ensures unit weight produces a 1 nS peak conductance.

        See Also
        --------
        init_state : Initialize state variables before calling ``update()``.
        get_spike : Compute spike output from membrane voltage.
        add_delta_input : Register synaptic spike inputs.
        sum_current_inputs : Aggregate external current sources.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        dg_ex = self.dg_ex.value  # nS/ms
        g_ex = self.g_ex.value  # nS
        dg_in = self.dg_in.value  # nS/ms
        g_in = self.g_in.value  # nS
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, dg_ex=dg_ex, g_ex=g_ex, dg_in=dg_in, g_in=g_in)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, dg_ex, g_ex = ode_state.V, ode_state.dg_ex, ode_state.g_ex
        dg_in, g_in = ode_state.dg_in, ode_state.g_in
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in iaf_cond_beta dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration).
        w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
        w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')

        # Compute beta normalization factors.
        # Extract unitless tau values for the scalar beta normalization computation.
        tau_rise_ex_ms = float(u.get_mantissa(self.tau_rise_ex / u.ms)) if np.ndim(self.tau_rise_ex) == 0 else None
        tau_decay_ex_ms = float(u.get_mantissa(self.tau_decay_ex / u.ms)) if np.ndim(self.tau_decay_ex) == 0 else None
        tau_rise_in_ms = float(u.get_mantissa(self.tau_rise_in / u.ms)) if np.ndim(self.tau_rise_in) == 0 else None
        tau_decay_in_ms = float(u.get_mantissa(self.tau_decay_in / u.ms)) if np.ndim(self.tau_decay_in) == 0 else None

        if tau_rise_ex_ms is not None and tau_decay_ex_ms is not None:
            pscon_ex = self._beta_normalization_factor_scalar(tau_rise_ex_ms, tau_decay_ex_ms) / u.ms
        else:
            # Fallback: use element-wise computation for array taus
            pscon_ex = np.e / self.tau_decay_ex

        if tau_rise_in_ms is not None and tau_decay_in_ms is not None:
            pscon_in = self._beta_normalization_factor_scalar(tau_rise_in_ms, tau_decay_in_ms) / u.ms
        else:
            pscon_in = np.e / self.tau_decay_in

        # Apply synaptic spike inputs.
        dg_ex = dg_ex + pscon_ex * w_ex  # nS/ms + 1/ms * nS = nS/ms
        dg_in = dg_in + pscon_in * w_in  # nS/ms + 1/ms * nS = nS/ms

        # Write back state.
        self.V.value = V
        self.dg_ex.value = dg_ex
        self.g_ex.value = g_ex
        self.dg_in.value = dg_in
        self.g_in.value = g_in
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
