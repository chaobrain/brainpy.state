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
import brainunit as u
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

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
    absolute error tolerance ``gsl_error_tol``. Each neuron maintains its own adaptive
    time step size (stored in ``integration_step``), which is adjusted based on local
    error estimates. The minimum step size is ``_MIN_H = 1e-8 ms`` and maximum iterations
    per simulation step is ``_MAX_ITERS = 100000``.

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
    gsl_error_tol : ArrayLike
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
    V_initializer : Callable, default: Constant(-70 mV)
        Initializer for membrane potential. Called as ``V_initializer(shape)``.
    g_ex_initializer : Callable, default: Constant(0 nS)
        Initializer for excitatory conductance. Called as ``g_ex_initializer(shape)``.
    g_in_initializer : Callable, default: Constant(0 nS)
        Initializer for inhibitory conductance. Called as ``g_in_initializer(shape)``.
    g_sfa_initializer : Callable, default: Constant(0 nS)
        Initializer for adaptation conductance. Called as ``g_sfa_initializer(shape)``.
    g_rr_initializer : Callable, default: Constant(0 nS)
        Initializer for relative refractory conductance. Called as ``g_rr_initializer(shape)``.
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
        >>> from brainpy import state as bp
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

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

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
        gsl_error_tol: ArrayLike = 1e-6,
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
        self.gsl_error_tol = gsl_error_tol

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
        self.g_sfa_initializer = g_sfa_initializer
        self.g_rr_initializer = g_rr_initializer
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
        r"""Validate parameter consistency and physical constraints.

        Checks that:

        * Reset potential is below threshold (``V_reset < V_th``)
        * Capacitance is positive (``C_m > 0``)
        * Refractory period is non-negative (``t_ref >= 0``)
        * All time constants are positive (``tau_syn_ex``, ``tau_syn_in``, ``tau_sfa``, ``tau_rr > 0``)
        * ``gsl_error_tol`` is strictly positive

        Raises
        ------
        ValueError
            If any validation check fails with descriptive error message.

        Notes
        -----
        Called automatically during ``__init__`` before state initialization.
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
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_sfa <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_rr <= 0.0 * u.ms):
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

        V = braintools.init.param(self.V_initializer, self.varshape)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape)
        g_sfa = braintools.init.param(self.g_sfa_initializer, self.varshape)
        g_rr = braintools.init.param(self.g_rr_initializer, self.varshape)

        self.V = brainstate.HiddenState(V)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)
        self.g_sfa = brainstate.HiddenState(g_sfa)
        self.g_rr = brainstate.HiddenState(g_rr)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

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

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, g_ex, g_in, g_sfa, g_rr — ODE state variables.
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
        i_l = self.g_L * (v_eff - self.E_L)
        i_sfa = state.g_sfa * (v_eff - self.E_sfa)
        i_rr = state.g_rr * (v_eff - self.E_rr)

        dV_raw = (
                     -i_l + self.I_e + extra.i_stim
                     - i_syn_exc - i_syn_inh - i_sfa - i_rr
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        dg_ex = -state.g_ex / self.tau_syn_ex
        dg_in = -state.g_in / self.tau_syn_in
        dg_sfa = -state.g_sfa / self.tau_sfa
        dg_rr = -state.g_rr / self.tau_rr

        return DotDict(V=dV, g_ex=dg_ex, g_in=dg_in, g_sfa=dg_sfa, g_rr=dg_rr)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, g_ex, g_in, g_sfa, g_rr — ODE state variables.
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
        new_g_sfa = u.math.where(spike_now, state.g_sfa + self.q_sfa, state.g_sfa)
        new_g_rr = u.math.where(spike_now, state.g_rr + self.q_rr, state.g_rr)
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict({**state, 'V': new_V, 'g_sfa': new_g_sfa, 'g_rr': new_g_rr})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

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
        jax.Array
            Binary spike tensor with dtype ``jnp.float64`` and shape
            ``self.V.value.shape``. A value of ``1.0`` indicates at least one
            internal spike event occurred during the integrated interval
            :math:`(t, t+dt]`.

        Raises
        ------
        ValueError
            If RKF45 integration enters a guarded unstable regime
            (``V < -1e3 mV``), indicating divergent dynamics for the current
            parameter/input regime.

        Notes
        -----
        Integration is performed with an adaptive vectorized RKF45 loop,
        including in-loop spike/reset/adaptation events and optional
        multiple spikes per step. All arithmetic is unit-aware via
        ``brainunit.math``.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        g_ex = self.g_ex.value  # nS
        g_in = self.g_in.value  # nS
        g_sfa = self.g_sfa.value  # nS
        g_rr = self.g_rr.value  # nS
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, g_ex=g_ex, g_in=g_in, g_sfa=g_sfa, g_rr=g_rr)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, g_ex, g_in = ode_state.V, ode_state.g_ex, ode_state.g_in
        g_sfa, g_rr = ode_state.g_sfa, ode_state.g_rr
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in iaf_cond_exp_sfa_rr dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration).
        w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
        w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')

        # Apply synaptic spike inputs.
        g_ex = g_ex + w_ex
        g_in = g_in + w_in

        # Write back state.
        self.V.value = V
        self.g_ex.value = g_ex
        self.g_in.value = g_in
        self.g_sfa.value = g_sfa
        self.g_rr.value = g_rr
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
