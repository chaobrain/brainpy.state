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

from brainpy_state._nest._base import NESTNeuron

__all__ = [
    'iaf_cond_exp_sfa_rr',
]


class iaf_cond_exp_sfa_rr(NESTNeuron):
    r"""NEST-compatible conductance-based LIF neuron with spike-frequency adaptation and relative refractory mechanisms.

    This model implements a conductance-based leaky integrate-and-fire neuron with exponential
    synaptic conductances, spike-frequency adaptation (SFA), and a relative refractory (RR)
    conductance mechanism. It follows the NEST ``iaf_cond_exp_sfa_rr`` model dynamics and
    update ordering exactly.

    Mathematical Description
    ------------------------

    The model evolves five state variables:

    1. **Synaptic conductances** (exponential decay):

       .. math::

          \frac{dg_{\mathrm{ex}}}{dt} = -\frac{g_{\mathrm{ex}}}{\tau_{\mathrm{syn,ex}}}, \qquad
          \frac{dg_{\mathrm{in}}}{dt} = -\frac{g_{\mathrm{in}}}{\tau_{\mathrm{syn,in}}}

    2. **Adaptation and relative refractory conductances** (exponential decay):

       .. math::

          \frac{dg_{\mathrm{sfa}}}{dt} = -\frac{g_{\mathrm{sfa}}}{\tau_{\mathrm{sfa}}}, \qquad
          \frac{dg_{\mathrm{rr}}}{dt} = -\frac{g_{\mathrm{rr}}}{\tau_{\mathrm{rr}}}

    3. **Membrane potential**:

       .. math::

          \frac{dV}{dt} = \frac{-I_{\mathrm{L}} + I_e + I_{\mathrm{stim}}
                               - I_{\mathrm{syn,ex}} - I_{\mathrm{syn,in}}
                               - I_{\mathrm{sfa}} - I_{\mathrm{rr}}}{C_m}

       where the individual currents are computed as:

       .. math::

          \begin{aligned}
          I_{\mathrm{L}} &= g_{\mathrm{L}} (V_{\mathrm{eff}} - E_{\mathrm{L}}) \\
          I_{\mathrm{syn,ex}} &= g_{\mathrm{ex}} (V_{\mathrm{eff}} - E_{\mathrm{ex}}) \\
          I_{\mathrm{syn,in}} &= g_{\mathrm{in}} (V_{\mathrm{eff}} - E_{\mathrm{in}}) \\
          I_{\mathrm{sfa}} &= g_{\mathrm{sfa}} (V_{\mathrm{eff}} - E_{\mathrm{sfa}}) \\
          I_{\mathrm{rr}} &= g_{\mathrm{rr}} (V_{\mathrm{eff}} - E_{\mathrm{rr}})
          \end{aligned}

       The effective voltage :math:`V_{\mathrm{eff}}` implements NEST voltage clamping:

       * During absolute refractory period: :math:`V_{\mathrm{eff}} = V_{\mathrm{reset}}`
       * Otherwise: :math:`V_{\mathrm{eff}} = \min(V, V_{\mathrm{th}})`

       During absolute refractory period, :math:`dV/dt = 0` while all conductances
       continue decaying.

    **Spike Dynamics**

    When :math:`V \geq V_{\mathrm{th}}` and the neuron is not refractory:

    1. A spike is emitted
    2. :math:`V \leftarrow V_{\mathrm{reset}}`
    3. Absolute refractory period begins (duration :math:`t_{\mathrm{ref}}`)
    4. Adaptation and RR conductances are incremented:

       .. math::

          g_{\mathrm{sfa}} \leftarrow g_{\mathrm{sfa}} + q_{\mathrm{sfa}}, \qquad
          g_{\mathrm{rr}} \leftarrow g_{\mathrm{rr}} + q_{\mathrm{rr}}

    **Numerical Integration**

    The ODEs are integrated using adaptive Runge-Kutta-Fehlberg 4(5) (RKF45) with
    absolute error tolerance ``_ATOL = 1e-3``. Each neuron maintains its own adaptive
    time step size (stored in ``integration_step``), which is adjusted based on local
    error estimates. The minimum step size is ``_MIN_H = 1e-8 ms`` and maximum iterations
    per simulation step is ``_MAX_ITERS = 10000``.

    **Update Ordering (NEST Semantics)**

    Per simulation step at time ``t``:

    1. **Integrate ODEs** over :math:`(t, t+dt]` using RKF45
    2. **Apply spike inputs**: Add incoming delta inputs to ``g_ex`` and ``g_in``
    3. **Refractory countdown**: Decrement refractory counter if neuron is refractory
    4. **Threshold test**: If :math:`V \geq V_{\mathrm{th}}` and not refractory, emit spike
    5. **Reset and adaptation**: On spike, reset voltage and increment ``g_sfa`` and ``g_rr``
    6. **Buffer current**: Store current input ``x`` into ``I_stim`` (one-step delay)

    The one-step delayed current input mirrors NEST's ring-buffer semantics.

    **Biological Interpretation**

    * ``g_sfa``: Models spike-frequency adaptation through a slow potassium-like current that
      accumulates with repeated spiking and gradually decays. This causes firing rate to decrease
      during sustained input.
    * ``g_rr``: Models relative refractoriness through a transient hyperpolarizing current that
      makes the neuron harder to excite immediately after a spike, beyond the absolute refractory
      period. This provides a smooth transition back to normal excitability.

    Parameters
    ----------
    in_size : int, tuple of int
        Population shape. Can be an integer for 1D population or tuple for multi-dimensional.
    E_L : ArrayLike, default: -70 mV
        Leak reversal potential. Must be less than ``V_th``.
    C_m : ArrayLike, default: 289.5 pF
        Membrane capacitance. Must be strictly positive.
    t_ref : ArrayLike, default: 0.5 ms
        Absolute refractory period duration. Must be non-negative. During this period,
        voltage is clamped to ``V_reset`` and no spikes can occur.
    V_th : ArrayLike, default: -57 mV
        Spike threshold potential. Must be greater than ``V_reset``.
    V_reset : ArrayLike, default: -70 mV
        Reset potential after spike. Must be less than ``V_th``.
    E_ex : ArrayLike, default: 0 mV
        Excitatory synaptic reversal potential. Typically set to 0 mV (depolarizing).
    E_in : ArrayLike, default: -75 mV
        Inhibitory synaptic reversal potential. Typically set below ``E_L`` (hyperpolarizing).
    g_L : ArrayLike, default: 28.95 nS
        Leak conductance. Determines membrane time constant :math:`\tau_m = C_m / g_L`.
    tau_syn_ex : ArrayLike, default: 1.5 ms
        Excitatory synaptic conductance decay time constant. Must be strictly positive.
        Fast excitatory synapses (AMPA-like).
    tau_syn_in : ArrayLike, default: 10.0 ms
        Inhibitory synaptic conductance decay time constant. Must be strictly positive.
        Slower inhibitory synapses (GABA-A-like).
    tau_sfa : ArrayLike, default: 110.0 ms
        Spike-frequency adaptation conductance decay time constant. Must be strictly positive.
        Long timescale for slow adaptation (calcium-activated potassium currents).
    tau_rr : ArrayLike, default: 1.97 ms
        Relative refractory conductance decay time constant. Must be strictly positive.
        Short timescale for post-spike transient refractoriness.
    E_sfa : ArrayLike, default: -70 mV
        Adaptation reversal potential. Typically set to or below ``E_L`` for hyperpolarizing effect.
    E_rr : ArrayLike, default: -70 mV
        Relative refractory reversal potential. Typically set to or below ``E_L`` for
        hyperpolarizing effect.
    q_sfa : ArrayLike, default: 14.48 nS
        Spike-triggered adaptation conductance increment. Added to ``g_sfa`` on each spike.
        Controls adaptation strength.
    q_rr : ArrayLike, default: 3214.0 nS
        Spike-triggered relative refractory conductance increment. Added to ``g_rr`` on each spike.
        Controls relative refractoriness strength. Large value creates strong transient
        hyperpolarization.
    I_e : ArrayLike, default: 0 pA
        Constant external current injection. Positive values are depolarizing.
    V_initializer : Callable, default: Constant(-70 mV)
        Initializer for membrane potential. Called as ``V_initializer(shape, batch_size)``.
    g_ex_initializer : Callable, default: Constant(0 nS)
        Initializer for excitatory conductance. Called as ``g_ex_initializer(shape, batch_size)``.
    g_in_initializer : Callable, default: Constant(0 nS)
        Initializer for inhibitory conductance. Called as ``g_in_initializer(shape, batch_size)``.
    g_sfa_initializer : Callable, default: Constant(0 nS)
        Initializer for adaptation conductance. Called as ``g_sfa_initializer(shape, batch_size)``.
    g_rr_initializer : Callable, default: Constant(0 nS)
        Initializer for relative refractory conductance. Called as ``g_rr_initializer(shape, batch_size)``.
    spk_fun : Callable, default: ReluGrad()
        Surrogate gradient function for differentiable spike generation. Maps scaled voltage
        difference :math:`(V - V_{\mathrm{th}}) / (V_{\mathrm{th}} - V_{\mathrm{reset}})`
        to spike probability in :math:`[0, 1]`.
    spk_reset : str, default: 'hard'
        Spike reset mode. ``'hard'`` uses stop_gradient (matches NEST), ``'soft'`` allows
        gradient flow through reset.
    ref_var : bool, default: False
        If True, expose boolean state variable ``refractory`` indicating whether neuron is
        in absolute refractory period.
    name : str, optional
        Name of the neuron group. If None, auto-generated.


    Parameter Mapping
    -----------------

    ==================== ================== ==========================================
    **Parameter**        **Default**        **Math equivalent**
    ==================== ================== ==========================================
    ``in_size``          (required)         —
    ``E_L``              -70 mV             :math:`E_\mathrm{L}`
    ``C_m``              289.5 pF           :math:`C_\mathrm{m}`
    ``t_ref``            0.5 ms             :math:`t_\mathrm{ref}`
    ``V_th``             -57 mV             :math:`V_\mathrm{th}`
    ``V_reset``          -70 mV             :math:`V_\mathrm{reset}`
    ``E_ex``             0 mV               :math:`E_\mathrm{ex}`
    ``E_in``             -75 mV             :math:`E_\mathrm{in}`
    ``g_L``              28.95 nS           :math:`g_\mathrm{L}`
    ``tau_syn_ex``       1.5 ms             :math:`\tau_{\mathrm{syn,ex}}`
    ``tau_syn_in``       10.0 ms            :math:`\tau_{\mathrm{syn,in}}`
    ``tau_sfa``          110.0 ms           :math:`\tau_{\mathrm{sfa}}`
    ``tau_rr``           1.97 ms            :math:`\tau_{\mathrm{rr}}`
    ``E_sfa``            -70 mV             :math:`E_\mathrm{sfa}`
    ``E_rr``             -70 mV             :math:`E_\mathrm{rr}`
    ``q_sfa``            14.48 nS           :math:`q_\mathrm{sfa}`
    ``q_rr``             3214.0 nS          :math:`q_\mathrm{rr}`
    ``I_e``              0 pA               :math:`I_\mathrm{e}`
    ==================== ================== ==========================================

    State Variables
    ---------------

    * ``V``: ``HiddenState`` (float, shape ``in_size``) — Membrane potential in mV
    * ``g_ex``: ``HiddenState`` (float, shape ``in_size``) — Excitatory conductance in nS
    * ``g_in``: ``HiddenState`` (float, shape ``in_size``) — Inhibitory conductance in nS
    * ``g_sfa``: ``HiddenState`` (float, shape ``in_size``) — Adaptation conductance in nS
    * ``g_rr``: ``HiddenState`` (float, shape ``in_size``) — Relative refractory conductance in nS
    * ``refractory_step_count``: ``ShortTermState`` (int32, shape ``in_size``) — Remaining refractory steps
    * ``integration_step``: ``ShortTermState`` (float, shape ``in_size``) — Adaptive RKF45 step size in ms
    * ``I_stim``: ``ShortTermState`` (float, shape ``in_size``) — One-step delayed current buffer in pA
    * ``last_spike_time``: ``ShortTermState`` (float, shape ``in_size``) — Last spike time in ms
    * ``refractory``: ``ShortTermState`` (bool, shape ``in_size``) — Boolean refractory indicator (if ``ref_var=True``)

    Raises
    ------
    ValueError
        If ``V_reset >= V_th`` (reset must be below threshold)
    ValueError
        If ``C_m <= 0`` (capacitance must be positive)
    ValueError
        If ``t_ref < 0`` (refractory period cannot be negative)
    ValueError
        If any time constant (``tau_syn_ex``, ``tau_syn_in``, ``tau_sfa``, ``tau_rr``) is non-positive

    See Also
    --------
    iaf_cond_exp : Simpler conductance-based LIF without adaptation or relative refractoriness
    iaf_cond_alpha : Conductance-based LIF with alpha-function synaptic conductances
    aeif_cond_exp : Exponential integrate-and-fire with conductance-based synapses

    References
    ----------
    .. [1] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for the large,
           fluctuating synaptic conductance state typical of neocortical neurons in vivo.
           Journal of Computational Neuroscience, 16:159-175.
           DOI: https://doi.org/10.1023/B:JCNS.0000014108.03012.81
    .. [2] Dayan P, Abbott LF (2001). Theoretical Neuroscience: Computational and
           Mathematical Modeling of Neural Systems. MIT Press.
    .. [3] NEST Simulator. ``iaf_cond_exp_sfa_rr`` documentation.
           https://nest-simulator.readthedocs.io/

    Examples
    --------
    Create a single neuron with default parameters:

    .. code-block:: python

        >>> import brainstate as bst
        >>> import brainunit as u
        >>> import brainpy.state as bp
        >>> neuron = bp.iaf_cond_exp_sfa_rr(in_size=1)
        >>> with bst.environ.context(dt=0.1 * u.ms):
        ...     neuron.init_all_states()
        ...     print(neuron.V.value)
        [-70.] mV

    Simulate a population with constant current injection:

    .. code-block:: python

        >>> import matplotlib.pyplot as plt
        >>> neuron = bp.iaf_cond_exp_sfa_rr(in_size=10, I_e=500 * u.pA)
        >>> with bst.environ.context(dt=0.1 * u.ms):
        ...     neuron.init_all_states()
        ...     voltages = []
        ...     for _ in range(1000):
        ...         spike = neuron.update()
        ...         voltages.append(neuron.V.value[0])
        >>> # plt.plot(voltages)  # Shows adaptation: decreasing firing rate

    Demonstrate spike-frequency adaptation:

    .. code-block:: python

        >>> # Strong adaptation (large q_sfa)
        >>> neuron_adapt = bp.iaf_cond_exp_sfa_rr(in_size=1, I_e=600*u.pA, q_sfa=50*u.nS)
        >>> # Weak adaptation (small q_sfa)
        >>> neuron_weak = bp.iaf_cond_exp_sfa_rr(in_size=1, I_e=600*u.pA, q_sfa=5*u.nS)
        >>> # neuron_adapt will show stronger decrease in firing rate over time

    Notes
    -----
    * **Computational cost**: This model uses adaptive RKF45 integration, which is more
      expensive than fixed-step exponential Euler used in simpler models like ``iaf_cond_exp``.
      However, it provides better accuracy for stiff dynamics.
    * **NEST compatibility**: This implementation exactly reproduces NEST behavior including
      voltage clamping, update ordering, and one-step delayed current semantics.
    * **Gradient flow**: The ``'hard'`` reset mode (default) uses ``stop_gradient`` on reset,
      which is necessary for NEST compatibility but prevents gradient flow through spike reset.
      Use ``spk_reset='soft'`` for better gradient-based learning, at the cost of deviating
      from NEST semantics.
    * **Parameter tuning**: The default ``q_sfa`` and ``q_rr`` values are taken from NEST
      defaults and produce moderate adaptation. Increase ``q_sfa`` for stronger adaptation
      (more pronounced firing rate decrease). Increase ``q_rr`` for stronger post-spike
      hyperpolarization.
    """

    __module__ = 'brainpy.state'

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 10000

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 289.5 * u.pF,
        t_ref: ArrayLike = 0.5 * u.ms,
        V_th: ArrayLike = -57. * u.mV,
        V_reset: ArrayLike = -70. * u.mV,
        E_ex: ArrayLike = 0. * u.mV,
        E_in: ArrayLike = -75. * u.mV,
        g_L: ArrayLike = 28.95 * u.nS,
        tau_syn_ex: ArrayLike = 1.5 * u.ms,
        tau_syn_in: ArrayLike = 10.0 * u.ms,
        tau_sfa: ArrayLike = 110.0 * u.ms,
        tau_rr: ArrayLike = 1.97 * u.ms,
        E_sfa: ArrayLike = -70. * u.mV,
        E_rr: ArrayLike = -70. * u.mV,
        q_sfa: ArrayLike = 14.48 * u.nS,
        q_rr: ArrayLike = 3214.0 * u.nS,
        I_e: ArrayLike = 0. * u.pA,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        g_ex_initializer: Callable = braintools.init.Constant(0. * u.nS),
        g_in_initializer: Callable = braintools.init.Constant(0. * u.nS),
        g_sfa_initializer: Callable = braintools.init.Constant(0. * u.nS),
        g_rr_initializer: Callable = braintools.init.Constant(0. * u.nS),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        r"""Initialize the iaf_cond_exp_sfa_rr neuron model.

        All parameters are validated to ensure physical consistency. Parameters can be scalars
        (broadcast to all neurons) or arrays matching ``in_size``.
        """
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
        self.tau_sfa = braintools.init.param(tau_sfa, self.varshape)
        self.tau_rr = braintools.init.param(tau_rr, self.varshape)
        self.E_sfa = braintools.init.param(E_sfa, self.varshape)
        self.E_rr = braintools.init.param(E_rr, self.varshape)
        self.q_sfa = braintools.init.param(q_sfa, self.varshape)
        self.q_rr = braintools.init.param(q_rr, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
        self.g_sfa_initializer = g_sfa_initializer
        self.g_rr_initializer = g_rr_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        r"""Convert brainunit quantity to plain NumPy array.

        Parameters
        ----------
        x : ArrayLike
            Input value with or without units.
        unit : brainunit.Unit
            Target unit for conversion (e.g., ``u.mV``, ``u.nS``).

        Returns
        -------
        np.ndarray
            Float64 array with unit removed.
        """
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast parameter array to match state shape.

        Parameters
        ----------
        x_np : np.ndarray
            Parameter array (may be scalar or lower-dimensional).
        shape : tuple
            Target shape (typically ``self.varshape`` or with batch dimension).

        Returns
        -------
        np.ndarray
            Broadcasted array of shape ``shape``.
        """
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        r"""Validate parameter consistency and physical constraints.

        Checks that:

        * Reset potential is below threshold (``V_reset < V_th``)
        * Capacitance is positive (``C_m > 0``)
        * Refractory period is non-negative (``t_ref >= 0``)
        * All time constants are positive (``tau_syn_ex``, ``tau_syn_in``, ``tau_sfa``, ``tau_rr > 0``)

        Raises
        ------
        ValueError
            If any validation check fails with descriptive error message.

        Notes
        -----
        Called automatically during ``__init__`` before state initialization.
        """
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_sfa, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_rr, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Creates and registers state variables for membrane potential, conductances,
        refractory counter, adaptive integration step, delayed current buffer, and
        last spike time. If ``ref_var=True``, also creates boolean refractory indicator.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for vectorized simulation. If None, no batch dimension is added.
        **kwargs : dict
            Additional keyword arguments (unused, for API compatibility).

        Notes
        -----
        * All conductances initialized to 0 nS by default
        * Membrane potential initialized to ``E_L`` (-70 mV) by default
        * Refractory counter initialized to 0 (not refractory)
        * Integration step initialized to simulation time step ``dt``
        * Last spike time initialized to -1e7 ms (far in the past)
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        g_sfa = braintools.init.param(self.g_sfa_initializer, self.varshape, batch_size)
        g_rr = braintools.init.param(self.g_rr_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)
        self.g_sfa = brainstate.HiddenState(g_sfa)
        self.g_rr = brainstate.HiddenState(g_rr)

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
        r"""Reset all state variables to initial values.

        Resets membrane potential, conductances, refractory counter, and other state variables
        to their initial values as specified by the initializers.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for vectorized simulation. If None, uses existing batch size.
        **kwargs : dict
            Additional keyword arguments (unused, for API compatibility).

        Notes
        -----
        This method does not reinitialize the state objects themselves, only their values.
        Use ``init_state`` to create new state objects.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.g_ex.value = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        self.g_in.value = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        self.g_sfa.value = braintools.init.param(self.g_sfa_initializer, self.varshape, batch_size)
        self.g_rr.value = braintools.init.param(self.g_rr_initializer, self.varshape, batch_size)
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
        r"""Compute differentiable spike output using surrogate gradient.

        Applies the surrogate gradient function to the scaled voltage difference. The voltage
        is scaled to :math:`[0, 1]` range where 0 corresponds to ``V_reset`` and 1 corresponds
        to ``V_th``. Values above 1 (threshold crossing) produce spike output near 1.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential in mV. If None, uses current ``self.V.value``.

        Returns
        -------
        ArrayLike
            Differentiable spike indicator in [0, 1], shape matching ``V``.
            Values near 1 indicate spike, near 0 indicate no spike.

        Notes
        -----
        * The surrogate function is specified by ``spk_fun`` parameter
        * The scaling ensures consistent behavior across different voltage ranges
        * During backpropagation, gradients flow through the surrogate function
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        r"""Compute number of simulation steps in absolute refractory period.

        Returns
        -------
        ArrayLike (int32)
            Number of time steps corresponding to ``t_ref``, computed as
            ``ceil(t_ref / dt)``. Shape matches parameter shape.

        Notes
        -----
        * Uses ceiling to ensure refractory period is at least as long as specified
        * Result is converted to int32 for use as counter
        * Called during each update to get refractory duration in discrete steps
        """
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _sum_signed_delta_inputs(self):
        r"""Accumulate delta inputs from synaptic projections into excitatory and inhibitory conductances.

        Processes all entries in ``self.delta_inputs`` dictionary, which is populated by
        synaptic projections during their update phase. Positive values are accumulated into
        excitatory conductance increments, negative values into inhibitory conductance increments.

        Returns
        -------
        tuple of ArrayLike
            (g_ex_delta, g_in_delta) — Conductance increments in nS, shape matching ``self.g_ex``.
            * ``g_ex_delta``: Sum of all positive delta inputs (excitatory)
            * ``g_in_delta``: Sum of absolute values of all negative delta inputs (inhibitory)

        Notes
        -----
        * Each delta input can be a value or a callable returning a value
        * Callable inputs are removed from the dict after retrieval (single-use)
        * If no delta inputs are registered, returns zeros
        * Sign convention: positive = excitatory, negative = inhibitory
        """
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
    def _dynamics_scalar(v, g_ex, g_in, g_sfa, g_rr, is_refractory, i_stim, p):
        r"""Compute ODE right-hand side for a single neuron (scalar implementation).

        This function implements the membrane potential and conductance dynamics for a single
        neuron. It is called repeatedly by the RKF45 integrator. All values are in NEST
        base units (mV, nS, pA, ms).

        Parameters
        ----------
        v : float
            Current membrane potential in mV.
        g_ex : float
            Current excitatory conductance in nS.
        g_in : float
            Current inhibitory conductance in nS.
        g_sfa : float
            Current spike-frequency adaptation conductance in nS.
        g_rr : float
            Current relative refractory conductance in nS.
        is_refractory : bool
            Whether neuron is in absolute refractory period.
        i_stim : float
            Stimulation current in pA.
        p : dict
            Parameter dictionary with keys: 'V_th', 'V_reset', 'E_L', 'E_ex', 'E_in', 'E_sfa',
            'E_rr', 'C_m', 'g_L', 'tau_syn_ex', 'tau_syn_in', 'tau_sfa', 'tau_rr', 'I_e'.
            All values in NEST base units.

        Returns
        -------
        tuple of float
            (dv, dg_ex, dg_in, dg_sfa, dg_rr) — Time derivatives in base units per ms.

        Notes
        -----
        * Effective voltage is clamped: ``V_reset`` during refractory, ``min(v, V_th)`` otherwise
        * During refractory period, ``dv = 0`` but conductances continue decaying
        * All conductances decay exponentially with their respective time constants
        """
        v_eff = p['V_reset'] if is_refractory else min(v, p['V_th'])

        i_syn_exc = g_ex * (v_eff - p['E_ex'])
        i_syn_inh = g_in * (v_eff - p['E_in'])
        i_l = p['g_L'] * (v_eff - p['E_L'])
        i_sfa = g_sfa * (v_eff - p['E_sfa'])
        i_rr = g_rr * (v_eff - p['E_rr'])

        dv = 0.0 if is_refractory else (
            -i_l + i_stim + p['I_e'] - i_syn_exc - i_syn_inh - i_sfa - i_rr
        ) / p['C_m']
        dg_ex = -g_ex / p['tau_syn_ex']
        dg_in = -g_in / p['tau_syn_in']
        dg_sfa = -g_sfa / p['tau_sfa']
        dg_rr = -g_rr / p['tau_rr']
        return dv, dg_ex, dg_in, dg_sfa, dg_rr

    def _rkf45_integrate_scalar(self, v0, ge0, gi0, gsfa0, grr0, is_refractory, i_stim, h0, dt, p):
        r"""Integrate ODEs for one time step using adaptive Runge-Kutta-Fehlberg 4(5) method.

        Performs adaptive step-size integration using RKF45 algorithm. The method computes
        both 4th and 5th order solutions, estimates local error, and adjusts step size
        accordingly to maintain error below ``_ATOL``.

        Parameters
        ----------
        v0 : float
            Initial membrane potential in mV.
        ge0 : float
            Initial excitatory conductance in nS.
        gi0 : float
            Initial inhibitory conductance in nS.
        gsfa0 : float
            Initial adaptation conductance in nS.
        grr0 : float
            Initial relative refractory conductance in nS.
        is_refractory : bool
            Whether neuron is in absolute refractory period.
        i_stim : float
            Stimulation current in pA (constant over integration interval).
        h0 : float
            Initial step size in ms (from previous integration).
        dt : float
            Total integration interval in ms.
        p : dict
            Parameter dictionary (see ``_dynamics_scalar`` for keys).

        Returns
        -------
        tuple of float
            (v_final, ge_final, gi_final, gsfa_final, grr_final, h_final) — Final state
            after integration and updated step size for next integration.

        Notes
        -----
        * Uses absolute error tolerance ``_ATOL = 1e-3``
        * Minimum step size ``_MIN_H = 1e-8 ms`` to prevent stalling
        * Maximum iterations ``_MAX_ITERS = 10000`` per call
        * Step size multiplier clamped to [0.2, 5.0] for stability
        * If error is below tolerance or step size is minimal, step is accepted
        * All computations use float64 for numerical stability
        """
        t = 0.0
        h = max(h0, self._MIN_H)
        y = np.asarray([v0, ge0, gi0, gsfa0, grr0], dtype=np.float64)
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
        r"""Advance the neuron state by one simulation time step.

        Implements the complete NEST update cycle:

        1. **ODE Integration**: Integrate voltage and conductances over [t, t+dt] using RKF45
        2. **Spike Input**: Apply delta inputs from incoming spikes to ``g_ex`` and ``g_in``
        3. **Refractory Logic**: Handle absolute refractory countdown and voltage clamping
        4. **Threshold Test**: Detect threshold crossing and emit spike if not refractory
        5. **Reset and Adaptation**: On spike, reset voltage and increment ``g_sfa`` and ``g_rr``
        6. **Current Buffering**: Store current input ``x`` for next time step (one-step delay)

        Parameters
        ----------
        x : ArrayLike, default: 0 pA
            External current input for the **next** time step in pA, shape matching ``in_size``
            or broadcastable. This input is buffered and applied with one-step delay,
            mirroring NEST ring-buffer semantics.

        Returns
        -------
        ArrayLike
            Differentiable spike output in [0, 1], shape matching ``in_size``. Values near 1
            indicate spike, near 0 indicate no spike. This is the spike probability computed
            by the surrogate gradient function applied to the voltage before reset.

        Notes
        -----
        * **Delta inputs**: Incoming spikes are accumulated via ``delta_inputs`` dict, which is
          populated by synaptic projections. Positive values increment ``g_ex``, negative values
          increment ``g_in``.
        * **Current inputs**: Continuous currents from current-based projections are summed via
          ``sum_current_inputs`` and buffered to ``I_stim`` with one-step delay.
        * **Spike timing**: Spikes are detected at time ``t+dt`` based on voltage at end of
          integration interval. ``last_spike_time`` is set to ``t+dt`` when spike occurs.
        * **Refractory behavior**: During absolute refractory period (``refractory_step_count > 0``),
          voltage is clamped to ``V_reset``, ``dV/dt = 0``, and threshold test is skipped.
        * **Adaptation increments**: ``q_sfa`` and ``q_rr`` are added to ``g_sfa`` and ``g_rr``
          immediately after spike detection, before storing updated conductances.
        * **Vectorization**: This implementation uses scalar RKF45 integration with explicit
          loop over neurons. This ensures exact NEST compatibility but is slower than vectorized
          integration. Each neuron maintains independent adaptive step size.
"""
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        g_ex = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape)
        g_in = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape)
        g_sfa = self._broadcast_to_state(self._to_numpy(self.g_sfa.value, u.nS), v_shape)
        g_rr = self._broadcast_to_state(self._to_numpy(self.g_rr.value, u.nS), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            v_shape,
        )
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape)

        p = {
            'V_th': self._broadcast_to_state(self._to_numpy(self.V_th, u.mV), v_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'E_ex': self._broadcast_to_state(self._to_numpy(self.E_ex, u.mV), v_shape),
            'E_in': self._broadcast_to_state(self._to_numpy(self.E_in, u.mV), v_shape),
            'E_sfa': self._broadcast_to_state(self._to_numpy(self.E_sfa, u.mV), v_shape),
            'E_rr': self._broadcast_to_state(self._to_numpy(self.E_rr, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'tau_syn_ex': self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape),
            'tau_syn_in': self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape),
            'tau_sfa': self._broadcast_to_state(self._to_numpy(self.tau_sfa, u.ms), v_shape),
            'tau_rr': self._broadcast_to_state(self._to_numpy(self.tau_rr, u.ms), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
        }
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )

        dg_ex_q, dg_in_q = self._sum_signed_delta_inputs()
        dg_ex = self._broadcast_to_state(self._to_numpy(dg_ex_q, u.nS), v_shape)
        dg_in = self._broadcast_to_state(self._to_numpy(dg_in_q, u.nS), v_shape)

        q_sfa = self._broadcast_to_state(self._to_numpy(self.q_sfa, u.nS), v_shape)
        q_rr = self._broadcast_to_state(self._to_numpy(self.q_rr, u.nS), v_shape)

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        v_for_spike = np.empty_like(V)
        spike_mask = np.zeros_like(V, dtype=bool)
        V_next = np.empty_like(V)
        ge_next = np.empty_like(g_ex)
        gi_next = np.empty_like(g_in)
        gsfa_next = np.empty_like(g_sfa)
        grr_next = np.empty_like(g_rr)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            is_refractory = r[idx] > 0
            v_i, ge_i, gi_i, gsfa_i, grr_i, h_i = self._rkf45_integrate_scalar(
                V[idx], g_ex[idx], g_in[idx], g_sfa[idx], g_rr[idx],
                is_refractory, i_stim[idx], h_int[idx], dt, local_p
            )

            # NEST ordering: spike input is added immediately after ODE integration.
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
                    gsfa_i += q_sfa[idx]
                    grr_i += q_rr[idx]
                else:
                    r_i = 0

            V_next[idx] = v_i
            ge_next[idx] = ge_i
            gi_next[idx] = gi_i
            gsfa_next[idx] = gsfa_i
            grr_next[idx] = grr_i
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.g_ex.value = ge_next * u.nS
        self.g_in.value = gi_next * u.nS
        self.g_sfa.value = gsfa_next * u.nS
        self.g_rr.value = grr_next * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float64) * u.mV)
