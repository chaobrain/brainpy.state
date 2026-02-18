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
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

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
    error to satisfy a fixed absolute tolerance (``_ATOL = 1e-3``).

    **4. Update Order (NEST Semantics)**

    Each simulation step executes the following operations in order:

    1. Integrate all ODEs on the interval :math:`(t, t+dt]` using RKF45.
    2. Decrement refractory counter; check threshold; emit spike; reset voltage if needed.
    3. Apply incoming spike weights to :math:`dg_\mathrm{ex}` and :math:`dg_\mathrm{in}`.
    4. Store external current input ``x`` into the delayed buffer ``I_stim`` (affects next step).

    This matches NEST's ring-buffer semantics: external currents applied at time :math:`t`
    take effect at time :math:`t + dt`.

    **5. Design Constraints and Assumptions**

    - **Non-differentiable integration**: RKF45 uses Python loops and NumPy arrays,
      preventing JAX gradient backpropagation through the dynamics. Surrogate gradients
      are computed via ``spk_fun`` for spike output only.
    - **Per-element integration**: Each neuron is integrated independently in a Python loop,
      which is slow for large populations. This design prioritizes numerical accuracy and
      exact NEST compatibility over performance.
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
    V_initializer : Callable, optional
        Initialization function for membrane voltage. Default: ``Constant(-70 mV)``.
        Called as ``V_initializer(varshape, batch_size)`` during ``init_state()``.
    g_ex_initializer : Callable, optional
        Initialization function for excitatory conductance. Default: ``Constant(0 nS)``.
        Called as ``g_ex_initializer(varshape, batch_size)`` during ``init_state()``.
    g_in_initializer : Callable, optional
        Initialization function for inhibitory conductance. Default: ``Constant(0 nS)``.
        Called as ``g_in_initializer(varshape, batch_size)`` during ``init_state()``.
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
        Membrane potential :math:`V_m` with shape ``(*in_size, *batch_shape)`` and units mV.
    - ``dg_ex`` : brainstate.ShortTermState
        Excitatory beta auxiliary state (dimensionless, dtype ``float64``).
    - ``g_ex`` : brainstate.HiddenState
        Excitatory synaptic conductance with units nS.
    - ``dg_in`` : brainstate.ShortTermState
        Inhibitory beta auxiliary state (dimensionless, dtype ``float64``).
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
        >>> import brainunit as u
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

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 10000
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

    @classmethod
    def _beta_normalization_factor_scalar(cls, tau_rise: float, tau_decay: float):
        tau_difference = tau_decay - tau_rise
        peak_value = 0.0
        if abs(tau_difference) > cls._EPS:
            t_peak = tau_decay * tau_rise * np.log(tau_decay / tau_rise) / tau_difference
            peak_value = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)

        if abs(peak_value) < cls._EPS:
            return np.e / tau_decay

        return (1.0 / tau_rise - 1.0 / tau_decay) / peak_value

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_rise_ex, u.ms) <= 0.0) or np.any(
            self._to_numpy(self.tau_decay_ex, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_rise_in, u.ms) <= 0.0) or np.any(
            self._to_numpy(self.tau_decay_in, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Creates and initializes state variables for membrane voltage, conductances, refractory
        tracking, RKF45 integration bookkeeping, and delayed current buffering. Called automatically
        by ``init_all_states()`` or manually for custom initialization workflows.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size for vectorized simulation. If ``None`` (default), creates
            state variables with shape ``in_size`` only. If provided, appends batch dimension
            to create shape ``(*in_size, batch_size)``.
        **kwargs
            Reserved for future extensions. Currently unused.

        Notes
        -----
        State variables created:

        - ``V`` : HiddenState, initialized using ``V_initializer``, units mV.
        - ``dg_ex``, ``dg_in`` : ShortTermState, initialized to zero, dimensionless (float64).
        - ``g_ex``, ``g_in`` : HiddenState, initialized using ``g_ex_initializer`` and ``g_in_initializer``, units nS.
        - ``last_spike_time`` : ShortTermState, initialized to ``-1e7 ms`` (effectively :math:`-\infty`), units ms.
        - ``refractory_step_count`` : ShortTermState, initialized to zero, integer (int32).
        - ``integration_step`` : ShortTermState, initialized to current simulation ``dt``, units ms.
        - ``I_stim`` : ShortTermState, initialized to zero, units pA.
        - ``refractory`` : ShortTermState (optional, only if ``ref_var=True``), initialized to ``False``, boolean.

        The ``dg_ex`` and ``dg_in`` states are stored as dimensionless NumPy float64 arrays for
        numerical precision during RKF45 integration.
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

        dt = brainstate.environ.get_dt()
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

        Reinitializes all state variables using the same initializers and logic as ``init_state()``.
        This is useful for restarting simulations or resetting a trained network to initial conditions
        without recreating the neuron object.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size for vectorized simulation. If ``None`` (default), uses the
            shape from the original initialization. If provided, reshapes states to
            ``(*in_size, batch_size)``.
        **kwargs
            Reserved for future extensions. Currently unused.

        Notes
        -----
        All state variables are overwritten with fresh values from their respective initializers.
        The ``integration_step`` is reset to the current simulation ``dt``, which may differ from
        the value at ``init_state()`` if ``dt`` has changed. The refractory counter is reset to zero,
        and ``last_spike_time`` is reset to ``-1e7 ms``.
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
        dt = brainstate.environ.get_dt()
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
        v_eff = p['V_reset'] if is_refractory else min(v, p['V_th'])

        i_syn_exc = g_ex * (v_eff - p['E_ex'])
        i_syn_inh = g_in * (v_eff - p['E_in'])
        i_leak = p['g_L'] * (v_eff - p['E_L'])
        dv = 0.0 if is_refractory else (
                                           -i_leak - i_syn_exc - i_syn_inh + i_stim + p['I_e']
                                       ) / p['C_m']

        ddg_ex = -dg_ex / p['tau_decay_ex']
        dg_ex_dt = dg_ex - (g_ex / p['tau_rise_ex'])
        ddg_in = -dg_in / p['tau_decay_in']
        dg_in_dt = dg_in - (g_in / p['tau_rise_in'])
        return dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt

    def _rkf45_integrate_scalar(self, v0, dg_ex0, g_ex0, dg_in0, g_in0, is_refractory, i_stim, h0, dt, p):
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
            k6 = f(
                y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0))

            y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
            y5 = y + h * (
                    16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0)
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
        r"""Advance neuron dynamics by one simulation time step.

        Performs the full NEST-compatible update cycle: ODE integration via RKF45, refractory
        countdown, threshold detection, spike emission, reset, synaptic input application, and
        delayed external current buffering.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input for the **next** time step (one-step delayed). Default: ``0 pA``.
            Must have shape compatible with ``(*in_size, *batch_shape)`` (broadcast-compatible).
            Units: pA (picoamperes). This current is stored in ``I_stim`` and takes effect at
            time :math:`t + dt`, matching NEST's ring-buffer semantics.

        Returns
        -------
        spike : jax.Array
            Binary spike output for the current time step, computed from the membrane voltage
            **before** reset. Shape: ``(*in_size, *batch_shape)``. Dtype: ``float32``.
            Values are typically in :math:`[0, 1]` when using surrogate gradients.

        Notes
        -----
        **Update Order (NEST-compatible):**

        1. **ODE Integration**: Integrate all differential equations on :math:`(t, t+dt]` using
           the Runge-Kutta-Fehlberg (RKF45) adaptive-step method with per-neuron state.
        2. **Refractory Handling**: If ``refractory_step_count > 0``, decrement counter, clamp
           voltage to ``V_reset``, and skip spike detection.
        3. **Threshold Detection**: If voltage crosses ``V_th`` (and not refractory), emit spike,
           reset voltage to ``V_reset``, and set refractory counter to
           :math:`\lceil t_\mathrm{ref} / dt \rceil`.
        4. **Synaptic Input Application**: Sum all incoming delta inputs (spike weights), split
           by sign into excitatory (positive) and inhibitory (negative) channels, multiply by
           beta normalization factors, and add to ``dg_ex`` and ``dg_in`` states.
        5. **External Current Buffering**: Store input ``x`` plus ``sum_current_inputs()`` into
           ``I_stim`` for use in the **next** time step.

        **Numerical Integration Details:**

        - Each neuron is integrated independently using a scalar RKF45 loop over state vector
          :math:`[V, dg_\mathrm{ex}, g_\mathrm{ex}, dg_\mathrm{in}, g_\mathrm{in}]`.
        - Adaptive step size starts from the previous ``integration_step`` value and adjusts
          dynamically to satisfy ``_ATOL = 1e-3`` absolute error tolerance.
        - If ``refractory_step_count > 0``, the voltage derivative is forced to zero and the
          effective voltage for current computation is clamped to ``V_reset``.
        - Integration step size is capped at :math:`[_MIN_H, dt]` and limited to ``_MAX_ITERS``
          substeps to prevent infinite loops.

        **Spike Weight Handling:**

        - All delta inputs (registered via ``add_delta_input()``) are summed and split by sign.
        - Positive weights :math:`w_\mathrm{ex} = \max(w, 0)` are multiplied by
          :math:`\kappa(\tau_{\mathrm{rise,ex}}, \tau_{\mathrm{decay,ex}})` and added to ``dg_ex``.
        - Negative weights :math:`w_\mathrm{in} = \max(-w, 0)` are multiplied by
          :math:`\kappa(\tau_{\mathrm{rise,in}}, \tau_{\mathrm{decay,in}})` and added to ``dg_in``.
        - The beta normalization factor ensures unit weight produces a 1 nS peak conductance.

        **State Updates:**

        After integration and spike handling, the following states are updated:

        - ``V.value``: Post-integration (and possibly post-reset) membrane voltage.
        - ``dg_ex.value``, ``g_ex.value``: Updated excitatory beta states and conductance.
        - ``dg_in.value``, ``g_in.value``: Updated inhibitory beta states and conductance.
        - ``refractory_step_count.value``: Decremented or set to :math:`\lceil t_\mathrm{ref} / dt \rceil`.
        - ``integration_step.value``: Adapted RKF45 step size for next iteration.
        - ``I_stim.value``: Delayed external current buffer for next step.
        - ``last_spike_time.value``: Set to :math:`t + dt` where spike occurred.
        - ``refractory.value`` (optional): Boolean refractory indicator if ``ref_var=True``.

        **Failure Modes**


        - If RKF45 fails to converge within ``_MAX_ITERS`` substeps, the integration stops at
          the current time and returns potentially inaccurate results. This typically indicates
          stiff dynamics or extreme parameter values. Check for:

          - Very small time constants (< 0.01 ms).
          - Very large conductances or currents causing rapid voltage changes.
          - Refractory period shorter than ``dt``.

        - If ``dt`` changes between calls (via ``brainstate.environ.set_dt()``), the refractory
          counter may become inaccurate because it's computed in grid steps. Reset state after
          changing ``dt``.

        See Also
        --------
        init_state : Initialize state variables before calling ``update()``.
        get_spike : Compute spike output from membrane voltage.
        add_delta_input : Register synaptic spike inputs.
        sum_current_inputs : Aggregate external current sources.
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

        tau_rise_ex = self._broadcast_to_state(self._to_numpy(self.tau_rise_ex, u.ms), v_shape)
        tau_decay_ex = self._broadcast_to_state(self._to_numpy(self.tau_decay_ex, u.ms), v_shape)
        tau_rise_in = self._broadcast_to_state(self._to_numpy(self.tau_rise_in, u.ms), v_shape)
        tau_decay_in = self._broadcast_to_state(self._to_numpy(self.tau_decay_in, u.ms), v_shape)

        p = {
            'V_th': self._broadcast_to_state(self._to_numpy(self.V_th, u.mV), v_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'E_ex': self._broadcast_to_state(self._to_numpy(self.E_ex, u.mV), v_shape),
            'E_in': self._broadcast_to_state(self._to_numpy(self.E_in, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'tau_rise_ex': tau_rise_ex,
            'tau_decay_ex': tau_decay_ex,
            'tau_rise_in': tau_rise_in,
            'tau_decay_in': tau_decay_in,
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
        }
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )

        w_ex_q, w_in_q = self._sum_signed_delta_inputs()
        w_ex = self._broadcast_to_state(self._to_numpy(w_ex_q, u.nS), v_shape)
        w_in = self._broadcast_to_state(self._to_numpy(w_in_q, u.nS), v_shape)

        pscon_ex = np.empty_like(V)
        pscon_in = np.empty_like(V)
        for idx in np.ndindex(v_shape):
            pscon_ex[idx] = self._beta_normalization_factor_scalar(tau_rise_ex[idx], tau_decay_ex[idx])
            pscon_in[idx] = self._beta_normalization_factor_scalar(tau_rise_in[idx], tau_decay_in[idx])

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
