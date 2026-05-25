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
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'hh_psc_alpha_clopath',
]


def _hh_psc_alpha_clopath_equilibrium(V):
    r"""Compute equilibrium values of Hodgkin-Huxley gating variables for Clopath model.

    Calculates steady-state activation and inactivation at a given membrane
    potential using the voltage-dependent rate functions from NEST's
    ``hh_psc_alpha_clopath`` model. These equilibria are used for state initialization
    when explicit gating values are not provided.

    Parameters
    ----------
    V : float
        Membrane potential in mV (unitless, not a ``saiunit`` quantity).

    Returns
    -------
    tuple of float
        ``(m_inf, h_inf, n_inf)`` — equilibrium values (dimensionless) for
        Na activation, Na inactivation, and K activation, respectively.
        Each is in [0, 1].

    Notes
    -----
    Uses the Hodgkin-Huxley rate functions with sign conventions matching
    NEST's implementation:

    .. math::

       \alpha_n = \frac{0.01(V + 55)}{1 - e^{-(V+55)/10}}, \quad
       \beta_n = 0.125 e^{-(V+65)/80}

    .. math::

       \alpha_m = \frac{0.1(V + 40)}{1 - e^{-(V+40)/10}}, \quad
       \beta_m = 4 e^{-(V+65)/18}

    .. math::

       \alpha_h = 0.07 e^{-(V+65)/20}, \quad
       \beta_h = \frac{1}{1 + e^{-(V+35)/10}}

    Equilibrium is :math:`x_\infty = \alpha_x / (\alpha_x + \beta_x)`.
    """
    alpha_n = (0.01 * (V + 55.0)) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
    alpha_m = (0.1 * (V + 40.0)) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
    alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    m_inf = alpha_m / (alpha_m + beta_m)
    h_inf = alpha_h / (alpha_h + beta_h)
    n_inf = alpha_n / (alpha_n + beta_n)
    return m_inf, h_inf, n_inf


class hh_psc_alpha_clopath(NESTNeuron):
    r"""NEST-compatible Hodgkin-Huxley neuron with Clopath plasticity support.

    Current-based spiking neuron using the Hodgkin-Huxley formalism with
    voltage-gated sodium and potassium channels, leak conductance, alpha-function
    postsynaptic currents, threshold-and-local-maximum spike detection, and three
    additional low-pass filtered voltage traces for Clopath voltage-based STDP.
    Follows NEST ``models/hh_psc_alpha_clopath.{h,cpp}`` implementation with
    adaptive Runge-Kutta integration (RK45).

    **1. Mathematical Model**

    **Membrane and ionic current dynamics:**

    The membrane potential evolves as

    .. math::

       C_m \frac{dV_m}{dt} = -(I_{Na} + I_K + I_L) + I_{stim} + I_e
                              + I_{syn,ex} + I_{syn,in}

    where

    .. math::

       I_{Na} &= g_{Na}\, m^3\, h\, (V_m - E_{Na})  \\
       I_K    &= g_K\,   n^4\,     (V_m - E_K)       \\
       I_L    &= g_L\,             (V_m - E_L)

    Gating variables :math:`m` (Na activation), :math:`h` (Na inactivation),
    :math:`n` (K activation) obey first-order kinetics:

    .. math::

       \frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)\,x

    with voltage-dependent rate functions (voltage :math:`V` in mV, rates in 1/ms):

    .. math::

       \alpha_n &= \frac{0.01\,(V + 55)}{1 - e^{-(V+55)/10}}, \quad
       \beta_n  = 0.125\,e^{-(V+65)/80}                                   \\
       \alpha_m &= \frac{0.1\,(V + 40)}{1 - e^{-(V+40)/10}}, \quad
       \beta_m  = 4\,e^{-(V+65)/18}                                       \\
       \alpha_h &= 0.07\,e^{-(V+65)/20}, \quad
       \beta_h  = \frac{1}{1 + e^{-(V+35)/10}}

    **Clopath low-pass filtered voltage traces:**

    The model extends standard ``hh_psc_alpha`` with three additional state
    variables for Clopath voltage-based plasticity:

    .. math::

       \frac{d\bar{u}_+}{dt}          &= \frac{-\bar{u}_+ + V_m}{\tau_{\bar{u}_+}} \\
       \frac{d\bar{u}_-}{dt}          &= \frac{-\bar{u}_- + V_m}{\tau_{\bar{u}_-}} \\
       \frac{d\bar{\bar{u}}}{dt}      &= \frac{-\bar{\bar{u}} + \bar{u}_-}{\tau_{\bar{\bar{u}}}}

    - :math:`\bar{u}_+` (``u_bar_plus``) is a slow-filtered voltage with time
      constant :math:`\tau_{\bar{u}_+} = 114` ms, used for LTP induction.
    - :math:`\bar{u}_-` (``u_bar_minus``) is a fast-filtered voltage with time
      constant :math:`\tau_{\bar{u}_-} = 10` ms, used for LTD induction.
    - :math:`\bar{\bar{u}}` (``u_bar_bar``) is a second-stage slow filter of
      :math:`\bar{u}_-` with time constant :math:`\tau_{\bar{\bar{u}}} = 500` ms,
      used for homeostatic sliding threshold in the Clopath rule.

    These traces are integrated as part of the same 11-dimensional ODE system
    and are accessible to connected Clopath synapse models for computing
    voltage-dependent weight updates.

    **Alpha-function synaptic currents:**

    Each synapse type (excitatory/inhibitory) is modelled as a second-order
    linear system producing an alpha-shaped postsynaptic current:

    .. math::

       \frac{dI_{syn}}{dt}  &= dI_{syn} - \frac{I_{syn}}{\tau_{syn}} \\
       \frac{d(dI_{syn})}{dt} &= -\frac{dI_{syn}}{\tau_{syn}}

    A spike arriving with weight :math:`w` (in pA) adds
    :math:`w \cdot e / \tau_{syn}` to :math:`dI_{syn}`, normalizing the
    peak current to :math:`w` for :math:`w = 1`. Incoming spike weights are
    split by sign: positive weights drive excitatory state (:math:`dI_{syn,ex}`),
    negative weights drive inhibitory state (:math:`dI_{syn,in}`).

    **2. Spike Detection and Refractory Handling**

    A spike is detected when the membrane potential crosses 0 mV from below
    **and** a local maximum is detected (i.e., the potential starts decreasing).
    Formally, a spike is emitted when:

    1. ``refractory_step_count == 0`` (not in refractory period), **and**
    2. ``V_m >= 0 mV`` (threshold crossing), **and**
    3. ``V_old > V_m`` (local maximum — potential is now falling).

    Unlike integrate-and-fire models, **no voltage reset occurs**. The potassium
    current naturally repolarizes the membrane after a spike. During the
    refractory period :math:`t_{ref}`, spike emission is suppressed but all
    state variables (including the Clopath filtered voltages) continue evolving
    according to their differential equations.

    **3. Update Order Per Simulation Step**

    The update follows NEST's exact order:

    1. Record pre-integration membrane potential (``V_old``).
    2. Integrate the full 11-dimensional ODE system
       :math:`(V_m, m, h, n, dI_{ex}, I_{ex}, dI_{in}, I_{in}, \bar{u}_+, \bar{u}_-, \bar{\bar{u}})`
       over one time step :math:`[t, t+dt]` using adaptive RK45 (Dormand-Prince).
    3. Add arriving synaptic spike inputs to :math:`dI_{syn,ex}` and
       :math:`dI_{syn,in}`.
    4. Check spike condition: ``V_m >= 0 and V_old > V_m and r == 0``.
    5. Update refractory counter and record spike time.
    6. Store buffered external stimulation current for the next step.

    **4. Numerical Integration**

    Uses a JAX-based adaptive RKF45 integrator via
    :class:`~brainpy_state._nest._utils.AdaptiveRungeKuttaStep` to match
    NEST's GSL RKF45 adaptive integrator. Default tolerance is
    ``gsl_error_tol=1e-6``. All neurons are integrated simultaneously in a
    vectorized fashion.

    **5. Assumptions, Constraints, and Computational Implications**

    - ``C_m > 0``, ``g_Na >= 0``, ``g_K >= 0``, ``g_L >= 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, ``tau_u_bar_plus > 0``,
      ``tau_u_bar_minus > 0``, ``tau_u_bar_bar > 0``, and ``t_ref >= 0``
      are enforced at construction.
    - External current ``update(x=...)`` is buffered for one step, matching
      NEST ring-buffer semantics.
    - The adaptive RKF45 integrator performs vectorized integration across
      all neurons simultaneously, enabling efficient GPU acceleration.
    - Spike detection uses a local maximum criterion rather than threshold
      crossing alone, matching biological action potential dynamics.
    - The three Clopath voltage traces add computational overhead (~27% increase
      in ODE dimensions compared to ``hh_psc_alpha``), but enable voltage-based
      plasticity without requiring additional post-hoc filtering.

    Parameters
    ----------
    in_size : Size
        Population shape specification. All per-neuron parameters are broadcast
        to ``self.varshape`` derived from ``in_size``.
    E_L : ArrayLike, optional
        Leak reversal potential :math:`E_L` in mV; scalar or array broadcastable
        to ``self.varshape``. Determines resting potential. Default is
        ``-54.402 * u.mV``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF; broadcastable to ``self.varshape``
        and strictly positive. Default is ``100. * u.pF``.
    g_Na : ArrayLike, optional
        Sodium peak conductance :math:`g_{Na}` in nS; broadcastable to
        ``self.varshape`` and non-negative. Default is ``12000. * u.nS``.
    g_K : ArrayLike, optional
        Potassium peak conductance :math:`g_K` in nS; broadcastable to
        ``self.varshape`` and non-negative. Default is ``3600. * u.nS``.
    g_L : ArrayLike, optional
        Leak conductance :math:`g_L` in nS; broadcastable to ``self.varshape``
        and non-negative. Default is ``30. * u.nS``.
    E_Na : ArrayLike, optional
        Sodium reversal potential :math:`E_{Na}` in mV; broadcastable to
        ``self.varshape``. Default is ``50. * u.mV``.
    E_K : ArrayLike, optional
        Potassium reversal potential :math:`E_K` in mV; broadcastable to
        ``self.varshape``. Default is ``-77. * u.mV``.
    t_ref : ArrayLike, optional
        Absolute refractory period :math:`t_{ref}` in ms; broadcastable to
        ``self.varshape`` and non-negative. Converted to integer step counts by
        ``ceil(t_ref / dt)``. Default is ``2. * u.ms``.
    tau_syn_ex : ArrayLike, optional
        Excitatory alpha time constant :math:`\tau_{syn,ex}` in ms; broadcastable
        to ``self.varshape`` and strictly positive. Default is ``0.2 * u.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory alpha time constant :math:`\tau_{syn,in}` in ms; broadcastable
        to ``self.varshape`` and strictly positive. Default is ``2. * u.ms``.
    I_e : ArrayLike, optional
        Constant injected current :math:`I_e` in pA; scalar or array broadcastable
        to ``self.varshape``. Default is ``0. * u.pA``.
    tau_u_bar_plus : ArrayLike, optional
        Time constant :math:`\tau_{\bar{u}_+}` in ms for slow voltage filter
        :math:`\bar{u}_+` (used in Clopath LTP); broadcastable to ``self.varshape``
        and strictly positive. Default is ``114. * u.ms``.
    tau_u_bar_minus : ArrayLike, optional
        Time constant :math:`\tau_{\bar{u}_-}` in ms for fast voltage filter
        :math:`\bar{u}_-` (used in Clopath LTD); broadcastable to ``self.varshape``
        and strictly positive. Default is ``10. * u.ms``.
    tau_u_bar_bar : ArrayLike, optional
        Time constant :math:`\tau_{\bar{\bar{u}}}` in ms for second-stage slow
        filter :math:`\bar{\bar{u}}` (used in Clopath homeostatic threshold);
        broadcastable to ``self.varshape`` and strictly positive. Default is
        ``500. * u.ms``.
    V_m_init : ArrayLike, optional
        Initial membrane potential in mV; broadcastable to ``self.varshape``.
        Default is ``-65. * u.mV``.
    Act_m_init : ArrayLike or None, optional
        Initial Na activation gating variable (dimensionless, range [0,1]);
        broadcastable to ``self.varshape``. If ``None``, initialized to
        equilibrium value at ``V_m_init``. Default is ``None``.
    Inact_h_init : ArrayLike or None, optional
        Initial Na inactivation gating variable (dimensionless, range [0,1]);
        broadcastable to ``self.varshape``. If ``None``, initialized to
        equilibrium value at ``V_m_init``. Default is ``None``.
    Act_n_init : ArrayLike or None, optional
        Initial K activation gating variable (dimensionless, range [0,1]);
        broadcastable to ``self.varshape``. If ``None``, initialized to
        equilibrium value at ``V_m_init``. Default is ``None``.
    u_bar_plus_init : ArrayLike, optional
        Initial value for :math:`\bar{u}_+` in mV; broadcastable to
        ``self.varshape``. Default is ``0. * u.mV``.
    u_bar_minus_init : ArrayLike, optional
        Initial value for :math:`\bar{u}_-` in mV; broadcastable to
        ``self.varshape``. Default is ``0. * u.mV``.
    u_bar_bar_init : ArrayLike, optional
        Initial value for :math:`\bar{\bar{u}}` in mV; broadcastable to
        ``self.varshape``. Default is ``0. * u.mV``.
    gsl_error_tol : ArrayLike, optional
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
        Default is ``1e-6``.
    spk_fun : Callable, optional
        Surrogate spike nonlinearity used by :meth:`get_spike` for differentiable
        spike generation. Default is ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy inherited from :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` applies stop-gradient to match NEST hard reset semantics.
        Default is ``'hard'``.
    name : str or None, optional
        Optional node name for debugging and visualization.

    Parameter Mapping
    -----------------

    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 17 27 14 16 36

       * - Parameter
         - Type / shape / unit
         - Default
         - Math symbol
         - Semantics
       * - ``in_size``
         - :class:`~brainstate.typing.Size`; scalar/tuple
         - required
         - --
         - Defines neuron population shape ``self.varshape``.
       * - ``E_L``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``-54.402 * u.mV``
         - :math:`E_L`
         - Leak reversal potential (resting potential).
       * - ``C_m``
         - ArrayLike, broadcastable (pF), ``> 0``
         - ``100. * u.pF``
         - :math:`C_m`
         - Membrane capacitance.
       * - ``g_Na``
         - ArrayLike, broadcastable (nS), ``>= 0``
         - ``12000. * u.nS``
         - :math:`g_{Na}`
         - Sodium peak conductance.
       * - ``g_K``
         - ArrayLike, broadcastable (nS), ``>= 0``
         - ``3600. * u.nS``
         - :math:`g_K`
         - Potassium peak conductance.
       * - ``g_L``
         - ArrayLike, broadcastable (nS), ``>= 0``
         - ``30. * u.nS``
         - :math:`g_L`
         - Leak conductance.
       * - ``E_Na``
         - ArrayLike, broadcastable (mV)
         - ``50. * u.mV``
         - :math:`E_{Na}`
         - Sodium reversal potential.
       * - ``E_K``
         - ArrayLike, broadcastable (mV)
         - ``-77. * u.mV``
         - :math:`E_K`
         - Potassium reversal potential.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), ``>= 0``
         - ``2. * u.ms``
         - :math:`t_{ref}`
         - Absolute refractory period duration.
       * - ``tau_syn_ex``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``0.2 * u.ms``
         - :math:`\tau_{syn,ex}`
         - Excitatory alpha-kernel time constant.
       * - ``tau_syn_in``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``2. * u.ms``
         - :math:`\tau_{syn,in}`
         - Inhibitory alpha-kernel time constant.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant external current added every step.
       * - ``tau_u_bar_plus``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``114. * u.ms``
         - :math:`\tau_{\bar{u}_+}`
         - Time constant for slow voltage filter (Clopath LTP).
       * - ``tau_u_bar_minus``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``10. * u.ms``
         - :math:`\tau_{\bar{u}_-}`
         - Time constant for fast voltage filter (Clopath LTD).
       * - ``tau_u_bar_bar``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``500. * u.ms``
         - :math:`\tau_{\bar{\bar{u}}}`
         - Time constant for second-stage filter (Clopath homeostasis).
       * - ``V_m_init``
         - ArrayLike, broadcastable (mV)
         - ``-65. * u.mV``
         - --
         - Initial membrane potential.
       * - ``Act_m_init``
         - ArrayLike or ``None``, dimensionless
         - ``None``
         - --
         - Initial Na activation; ``None`` uses equilibrium at ``V_m_init``.
       * - ``Inact_h_init``
         - ArrayLike or ``None``, dimensionless
         - ``None``
         - --
         - Initial Na inactivation; ``None`` uses equilibrium at ``V_m_init``.
       * - ``Act_n_init``
         - ArrayLike or ``None``, dimensionless
         - ``None``
         - --
         - Initial K activation; ``None`` uses equilibrium at ``V_m_init``.
       * - ``u_bar_plus_init``
         - ArrayLike, broadcastable (mV)
         - ``0. * u.mV``
         - --
         - Initial value for :math:`\bar{u}_+`.
       * - ``u_bar_minus_init``
         - ArrayLike, broadcastable (mV)
         - ``0. * u.mV``
         - --
         - Initial value for :math:`\bar{u}_-`.
       * - ``u_bar_bar_init``
         - ArrayLike, broadcastable (mV)
         - ``0. * u.mV``
         - --
         - Initial value for :math:`\bar{\bar{u}}`.
       * - ``gsl_error_tol``
         - ArrayLike, broadcastable, unitless, ``> 0``
         - ``1e-6``
         - --
         - Local absolute tolerance for the embedded RKF45 error estimate.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate gradient function for spike generation.
       * - ``spk_reset``
         - str
         - ``'hard'``
         - --
         - Reset mode; ``'hard'`` stops gradient through reset.

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential :math:`V_m`. Shape: ``self.varshape``.
        Units: mV.
    m : brainstate.HiddenState
        Na activation gating variable (dimensionless). Shape: ``self.varshape``.
        Range: [0, 1].
    h : brainstate.HiddenState
        Na inactivation gating variable (dimensionless). Shape: ``self.varshape``.
        Range: [0, 1].
    n : brainstate.HiddenState
        K activation gating variable (dimensionless). Shape: ``self.varshape``.
        Range: [0, 1].
    I_syn_ex : brainstate.ShortTermState
        Excitatory postsynaptic current :math:`I_{syn,ex}`. Shape: ``self.varshape``.
        Units: pA.
    I_syn_in : brainstate.ShortTermState
        Inhibitory postsynaptic current :math:`I_{syn,in}`. Shape: ``self.varshape``.
        Units: pA.
    dI_syn_ex : brainstate.ShortTermState
        Excitatory alpha-kernel derivative state. Shape: ``self.varshape``.
        Units: pA/ms.
    dI_syn_in : brainstate.ShortTermState
        Inhibitory alpha-kernel derivative state. Shape: ``self.varshape``.
        Units: pA/ms.
    u_bar_plus : brainstate.HiddenState
        Slow-filtered voltage :math:`\bar{u}_+` for Clopath LTP. Shape: ``self.varshape``.
        Units: mV.
    u_bar_minus : brainstate.HiddenState
        Fast-filtered voltage :math:`\bar{u}_-` for Clopath LTD. Shape: ``self.varshape``.
        Units: mV.
    u_bar_bar : brainstate.HiddenState
        Second-stage filtered voltage :math:`\bar{\bar{u}}` for Clopath homeostasis.
        Shape: ``self.varshape``. Units: mV.
    I_stim : brainstate.ShortTermState
        One-step delayed external current buffer. Shape: ``self.varshape``.
        Units: pA.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory steps. Shape: ``self.varshape``. Dtype: int32.
    last_spike_time : brainstate.ShortTermState
        Time of most recent spike emission. Shape: ``self.varshape``.
        Units: ms. Updated to ``t + dt`` on spike emission.
    integration_step : brainstate.ShortTermState
        Persistent RKF45 substep size estimate (ms).

    Raises
    ------
    ValueError
        If any of the following conditions are violated:
        - ``C_m <= 0``
        - ``t_ref < 0``
        - ``tau_syn_ex <= 0`` or ``tau_syn_in <= 0``
        - ``tau_u_bar_plus <= 0``, ``tau_u_bar_minus <= 0``, or ``tau_u_bar_bar <= 0``
        - ``g_Na < 0``, ``g_K < 0``, or ``g_L < 0``
        - ``gsl_error_tol <= 0``

    Notes
    -----
    - Unlike IAF models, the HH model does **not** reset the membrane potential
      after a spike. Repolarization occurs naturally through the potassium current.
    - During the refractory period, the neuron's subthreshold dynamics continue
      to evolve freely; only spike emission is suppressed.
    - Spike weights are interpreted as current amplitudes (pA). Positive weights
      are excitatory; negative weights are inhibitory.
    - The three Clopath-related voltage traces (``u_bar_plus``, ``u_bar_minus``,
      ``u_bar_bar``) are integrated as part of the same 11-dimensional ODE system,
      matching NEST's GSL integration. This adds ~27% computational overhead
      compared to ``hh_psc_alpha``.
    - The adaptive RKF45 integrator evaluates the ODE right-hand side multiple
      times per step, so computation cost scales with desired accuracy (controlled
      by ``gsl_error_tol``).
    - Spike detection combines threshold crossing (0 mV) and local maximum
      detection, matching the biological action potential waveform.

    References
    ----------
    .. [1] Hodgkin AL, Huxley AF (1952). A quantitative description of membrane
           current and its application to conduction and excitation in nerve.
           The Journal of Physiology 117:500-544.
           DOI: https://doi.org/10.1113/jphysiol.1952.sp004764
    .. [2] Clopath C, Busing L, Vasilaki E, Gerstner W (2010). Connectivity
           reflects coding: a model of voltage-based STDP with homeostasis.
           Nature Neuroscience 13(3):344-352.
           DOI: https://doi.org/10.1038/nn.2479
    .. [3] Clopath C, Gerstner W (2010). Voltage and spike timing interact
           in STDP -- a unified model. Frontiers in Synaptic Neuroscience 2:25.
           DOI: https://doi.org/10.3389/fnsyn.2010.00025
    .. [4] Gerstner W, Kistler WM (2002). Spiking neuron models: Single neurons,
           populations, plasticity. Cambridge University Press.
    .. [5] Dayan P, Abbott LF (2001). Theoretical neuroscience: Computational
           and mathematical modeling of neural systems. MIT Press.

    See Also
    --------
    hh_psc_alpha : Hodgkin-Huxley neuron without Clopath plasticity support.
    clopath_synapse : Voltage-based STDP synapse model that uses these filtered traces.

    Examples
    --------
    Create a population of HH neurons with Clopath plasticity support:

    .. code-block:: python

        >>> import brainstate as bst
        >>> import brainpy_state as bps
        >>> import saiunit as u
        >>> bst.environ.set(dt=0.1 * u.ms)
        >>> neurons = bps.hh_psc_alpha_clopath(
        ...     in_size=100,
        ...     tau_u_bar_plus=114. * u.ms,  # Slow LTP filter
        ...     tau_u_bar_minus=10. * u.ms,  # Fast LTD filter
        ...     tau_u_bar_bar=500. * u.ms,   # Homeostatic filter
        ... )
        >>> neurons.init_state()
        >>> # Simulate with constant current injection
        >>> spikes = neurons.update(400. * u.pA)
        >>> # Access Clopath voltage traces for plasticity computation
        >>> u_plus = neurons.u_bar_plus.value
        >>> u_minus = neurons.u_bar_minus.value
        >>> u_bar = neurons.u_bar_bar.value
    """

    __module__ = 'brainpy.state'

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -54.402 * u.mV,
        C_m: ArrayLike = 100. * u.pF,
        g_Na: ArrayLike = 12000. * u.nS,
        g_K: ArrayLike = 3600. * u.nS,
        g_L: ArrayLike = 30. * u.nS,
        E_Na: ArrayLike = 50. * u.mV,
        E_K: ArrayLike = -77. * u.mV,
        t_ref: ArrayLike = 2. * u.ms,
        tau_syn_ex: ArrayLike = 0.2 * u.ms,
        tau_syn_in: ArrayLike = 2. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        tau_u_bar_plus: ArrayLike = 114. * u.ms,
        tau_u_bar_minus: ArrayLike = 10. * u.ms,
        tau_u_bar_bar: ArrayLike = 500. * u.ms,
        V_m_init: ArrayLike = -65. * u.mV,
        Act_m_init: ArrayLike = None,
        Inact_h_init: ArrayLike = None,
        Act_n_init: ArrayLike = None,
        u_bar_plus_init: ArrayLike = 0. * u.mV,
        u_bar_minus_init: ArrayLike = 0. * u.mV,
        u_bar_bar_init: ArrayLike = 0. * u.mV,
        gsl_error_tol: ArrayLike = 1e-6,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.g_Na = braintools.init.param(g_Na, self.varshape)
        self.g_K = braintools.init.param(g_K, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.E_Na = braintools.init.param(E_Na, self.varshape)
        self.E_K = braintools.init.param(E_K, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.tau_u_bar_plus = braintools.init.param(tau_u_bar_plus, self.varshape)
        self.tau_u_bar_minus = braintools.init.param(tau_u_bar_minus, self.varshape)
        self.tau_u_bar_bar = braintools.init.param(tau_u_bar_bar, self.varshape)
        self.V_m_init = V_m_init
        self.Act_m_init = Act_m_init
        self.Inact_h_init = Inact_h_init
        self.Act_n_init = Act_n_init
        self.u_bar_plus_init = u_bar_plus_init
        self.u_bar_minus_init = u_bar_minus_init
        self.u_bar_bar_init = u_bar_bar_init
        self.gsl_error_tol = gsl_error_tol

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
        r"""Validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If any parameter violates physical or numerical constraints:
            - ``C_m <= 0``
            - ``t_ref < 0``
            - ``tau_syn_ex <= 0`` or ``tau_syn_in <= 0``
            - ``tau_u_bar_plus <= 0``, ``tau_u_bar_minus <= 0``, or ``tau_u_bar_bar <= 0``
            - ``g_Na < 0``, ``g_K < 0``, or ``g_L < 0``
            - ``gsl_error_tol <= 0``
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.t_ref, self.g_Na)):
            return
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms) or cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_u_bar_plus <= 0.0 * u.ms) or cond_any(
            self.tau_u_bar_minus <= 0.0 * u.ms) or cond_any(
            self.tau_u_bar_bar <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.g_Na < 0.0 * u.nS) or cond_any(self.g_K < 0.0 * u.nS) or cond_any(
            self.g_L < 0.0 * u.nS):
            raise ValueError('All conductances must be non-negative.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all neuron state variables.

        Creates and initializes the 11-dimensional state vector for each neuron:
        membrane potential, three gating variables (m, h, n), two pairs of alpha-kernel
        states (excitatory/inhibitory), three Clopath filtered voltages, and auxiliary
        states for refractory handling and spike timing.

        **Initialization logic:**

        - **Membrane potential** (``V``): set to ``V_m_init``.
        - **Gating variables** (``m``, ``h``, ``n``): if explicit ``Act_m_init``,
          ``Inact_h_init``, ``Act_n_init`` are provided, use those values; otherwise,
          compute equilibrium values at ``V_m_init`` using rate functions.
        - **Alpha-kernel states** (``dI_syn_ex``, ``I_syn_ex``, ``dI_syn_in``,
          ``I_syn_in``): initialized to zero.
        - **Clopath filtered voltages** (``u_bar_plus``, ``u_bar_minus``, ``u_bar_bar``):
          set to ``u_bar_plus_init``, ``u_bar_minus_init``, ``u_bar_bar_init`` (default 0 mV).
        - **Auxiliary states**: ``I_stim`` set to 0 pA, ``refractory_step_count`` set to 0,
          ``last_spike_time`` set to -1e7 ms (far past), ``integration_step`` set to dt.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        - Equilibrium gating variables are computed using
          :func:`_hh_psc_alpha_clopath_equilibrium` at the scalar value of
          ``V_m_init[0]`` (first element if ``V_m_init`` is an array).
        - Initial Clopath filtered voltages default to 0 mV, matching NEST behavior.
          For long-running simulations starting from rest, consider setting these
          to ``V_m_init`` to avoid initial transient artifacts in voltage-based
          plasticity.
        - This method must be called before the first :meth:`update` call.

        See Also
        --------
        _hh_psc_alpha_clopath_equilibrium : Computes equilibrium gating variables.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V_init_mV = np.asarray(u.math.asarray(
            braintools.init.param(self.V_m_init, self.varshape) / u.mV
        ), dtype=dftype)
        V_init_scalar = float(V_init_mV.flat[0]) if V_init_mV.ndim > 0 else float(V_init_mV)

        # Compute equilibrium gating variables at initial V
        m_eq, h_eq, n_eq = _hh_psc_alpha_clopath_equilibrium(V_init_scalar)

        V = braintools.init.param(braintools.init.Constant(self.V_m_init), self.varshape)

        if self.Act_m_init is not None:
            m_init = float(np.asarray(u.math.asarray(self.Act_m_init), dtype=dftype))
        else:
            m_init = m_eq
        if self.Inact_h_init is not None:
            h_init = float(np.asarray(u.math.asarray(self.Inact_h_init), dtype=dftype))
        else:
            h_init = h_eq
        if self.Act_n_init is not None:
            n_init = float(np.asarray(u.math.asarray(self.Act_n_init), dtype=dftype))
        else:
            n_init = n_eq

        # Clopath filtered voltage initial values
        u_bar_plus_init_val = braintools.init.param(
            braintools.init.Constant(self.u_bar_plus_init), self.varshape
        )
        u_bar_minus_init_val = braintools.init.param(
            braintools.init.Constant(self.u_bar_minus_init), self.varshape
        )
        u_bar_bar_init_val = braintools.init.param(
            braintools.init.Constant(self.u_bar_bar_init), self.varshape
        )

        zeros_pA = u.math.zeros(self.varshape, dtype=dftype) * u.pA
        zeros_pA_per_ms = u.math.zeros(self.varshape, dtype=dftype) * (u.pA / u.ms)

        self.V = brainstate.HiddenState(V)
        self.m = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(m_init), self.varshape)
        )
        self.h = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(h_init), self.varshape)
        )
        self.n = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(n_init), self.varshape)
        )
        self.dI_syn_ex = brainstate.ShortTermState(zeros_pA_per_ms)
        self.I_syn_ex = brainstate.ShortTermState(zeros_pA)
        self.dI_syn_in = brainstate.ShortTermState(zeros_pA_per_ms)
        self.I_syn_in = brainstate.ShortTermState(zeros_pA)
        self.u_bar_plus = brainstate.HiddenState(u_bar_plus_init_val)
        self.u_bar_minus = brainstate.HiddenState(u_bar_minus_init_val)
        self.u_bar_bar = brainstate.HiddenState(u_bar_bar_init_val)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

    def get_spike(self, V: ArrayLike = None):
        r"""Generate differentiable spike output via surrogate gradient function.

        Applies the surrogate spike function ``self.spk_fun`` to the membrane
        potential, producing a differentiable approximation of the Heaviside
        step function for gradient-based learning. For HH neurons, the spike
        threshold is 0 mV.

        **Usage in training:**

        - The actual spike detection in :meth:`update` uses the discrete threshold-
          and-local-maximum criterion (non-differentiable).
        - This method provides a **separate**, differentiable spike signal for
          backpropagation through time (BPTT) or surrogate gradient learning.
        - The returned values do **not** affect the neuron dynamics; they are purely
          for gradient computation.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Membrane potential in mV; scalar or array broadcastable to state shape.
            If ``None`` (default), uses the current state ``self.V.value``.

        Returns
        -------
        ArrayLike
            Differentiable spike-like signal with shape matching ``V``. Output
            range depends on ``self.spk_fun``; for ``ReluGrad()``, positive values
            indicate suprathreshold activity, with gradient flowing through the
            ReLU derivative at the threshold (0 mV).

        Notes
        -----
        - The membrane potential ``V`` is scaled to be unitless before applying
          ``self.spk_fun``, as surrogate functions expect dimensionless inputs.
        - Common surrogate functions include:
          - ``braintools.surrogate.ReluGrad()``: piecewise linear, fast.
          - ``braintools.surrogate.Sigmoid()``: smooth, symmetric.
          - ``braintools.surrogate.ATan()``: unbounded, soft.
        - For inference (non-training), use the boolean spike array from
          :meth:`update` thresholded at 0 instead of this method.

        Examples
        --------
        Compute differentiable spike output for a given voltage:

        .. code-block:: python

            >>> import brainpy_state as bps
            >>> import saiunit as u
            >>> neurons = bps.hh_psc_alpha_clopath(in_size=10)
            >>> neurons.init_state()
            >>> V_test = u.math.array([[-70., -10., 0., 5., 20.]]) * u.mV
            >>> spikes_surrogate = neurons.get_spike(V_test)
            >>> print(spikes_surrogate)  # Differentiable approximation
        """
        V = self.V.value if V is None else V
        # For HH neurons, spike threshold is 0 mV. Scale relative to 0 mV.
        v_scaled = V / (1. * u.mV)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Computes the 11-dimensional ODE right-hand side for the Hodgkin-Huxley
        model with Clopath voltage traces.

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, dI_ex, I_ex, dI_in, I_in, u_bar_plus,
            u_bar_minus, u_bar_bar -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, V_old, i_stim -- mutable auxiliary data
            carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        V = state.V
        m_ = state.m
        h_ = state.h
        n_ = state.n

        # Voltage-dependent rate functions (V in mV, rates in 1/ms)
        V_mV = V / u.mV
        alpha_n = (0.01 * (V_mV + 55.0)) / (1.0 - u.math.exp(-(V_mV + 55.0) / 10.0)) / u.ms
        beta_n = 0.125 * u.math.exp(-(V_mV + 65.0) / 80.0) / u.ms
        alpha_m = (0.1 * (V_mV + 40.0)) / (1.0 - u.math.exp(-(V_mV + 40.0) / 10.0)) / u.ms
        beta_m = 4.0 * u.math.exp(-(V_mV + 65.0) / 18.0) / u.ms
        alpha_h = 0.07 * u.math.exp(-(V_mV + 65.0) / 20.0) / u.ms
        beta_h = 1.0 / (1.0 + u.math.exp(-(V_mV + 35.0) / 10.0)) / u.ms

        # Ionic currents
        I_Na = self.g_Na * m_ * m_ * m_ * h_ * (V - self.E_Na)
        I_K = self.g_K * n_ * n_ * n_ * n_ * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # Membrane voltage dynamics
        dV = (-(I_Na + I_K + I_L) + extra.i_stim + self.I_e + state.I_ex + state.I_in) / self.C_m

        # Gating variable dynamics
        dm = alpha_m * (1.0 - m_) - beta_m * m_
        dh = alpha_h * (1.0 - h_) - beta_h * h_
        dn = alpha_n * (1.0 - n_) - beta_n * n_

        # Alpha-kernel synaptic current dynamics
        ddI_ex = -state.dI_ex / self.tau_syn_ex
        dI_ex_dt = state.dI_ex - state.I_ex / self.tau_syn_ex
        ddI_in = -state.dI_in / self.tau_syn_in
        dI_in_dt = state.dI_in - state.I_in / self.tau_syn_in

        # Clopath filtered voltage traces
        du_bar_plus = (-state.u_bar_plus + V) / self.tau_u_bar_plus
        du_bar_minus = (-state.u_bar_minus + V) / self.tau_u_bar_minus
        du_bar_bar = (-state.u_bar_bar + state.u_bar_minus) / self.tau_u_bar_bar

        return DotDict(
            V=dV, m=dm, h=dh, n=dn,
            dI_ex=ddI_ex, I_ex=dI_ex_dt,
            dI_in=ddI_in, I_in=dI_in_dt,
            u_bar_plus=du_bar_plus, u_bar_minus=du_bar_minus, u_bar_bar=du_bar_bar,
        )

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection and V_old tracking.

        For HH neurons there is no voltage reset inside the integration loop.
        This callback records the V_old for post-loop local-maximum spike
        detection and tracks spike occurrences.

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, dI_ex, I_ex, dI_in, I_in, u_bar_plus,
            u_bar_minus, u_bar_bar -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, V_old, i_stim.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated V_old tracking.
        """
        # Update V_old to track the previous accepted voltage for local-max detection
        new_V_old = u.math.where(accept, state.V, extra.V_old)
        new_extra = DotDict({**extra, 'V_old': new_V_old})
        return state, new_extra

    def update(self, x=0. * u.pA):
        r"""Advance the neuron by one simulation step.

        Integrates the full 11-dimensional Hodgkin-Huxley dynamics by one time step,
        including membrane potential, gating variables, synaptic currents, and
        three Clopath filtered voltage traces. Follows NEST's exact update order
        for ``hh_psc_alpha_clopath`` with adaptive RK45 integration.

        **Update sequence:**

        1. Record pre-integration membrane potential (``V_old``) for spike detection.
        2. Integrate the full 11-dimensional ODE system
           :math:`(V_m, m, h, n, dI_{ex}, I_{ex}, dI_{in}, I_{in}, \bar{u}_+, \bar{u}_-, \bar{\bar{u}})`
           over one time step :math:`[t, t+dt]` using adaptive RK45 (Dormand-Prince
           method) with tolerance ``gsl_error_tol``.
        3. Add arriving synaptic spike inputs to :math:`dI_{syn,ex}` and
           :math:`dI_{syn,in}` (spike weights split by sign; positive -> excitatory,
           negative -> inhibitory).
        4. Check spike condition:
           ``V_m >= 0 mV`` **and** ``V_old > V_m`` **and** ``refractory_step_count == 0``.
        5. Update refractory counter: set to ``ceil(t_ref / dt)`` on spike, otherwise
           decrement if positive.
        6. Record spike time as ``t + dt`` if spike detected.
        7. Store buffered external stimulation current ``x`` for the next step
           (one-step delay, matching NEST ring-buffer semantics).

        **Integration details:**

        - Uses :class:`~brainpy_state._nest._utils.AdaptiveRungeKuttaStep` with
          method ``'RKF45'`` for vectorized integration of all neurons simultaneously.
        - The ODE right-hand side includes all 11 state equations with full coupling.
        - Alpha-kernel normalization ensures a weight of 1 pA produces a peak PSC of 1 pA.

        **Spike detection semantics:**

        - **No hard reset**: Unlike IAF models, the membrane potential is not clamped
          after a spike. The potassium current :math:`I_K` naturally repolarizes the cell.
        - **Local maximum criterion**: A spike is only emitted when the voltage
          both exceeds 0 mV **and** starts to fall (``V_old > V_m``), matching
          biological action potential detection.
        - **Refractory suppression**: Spike emission is blocked during the refractory
          period, but all state variables (including Clopath filters) continue evolving.

        Parameters
        ----------
        x : ArrayLike, optional
            External stimulation current in pA; scalar or array broadcastable to
            ``self.varshape``. Added to ``I_e`` and synaptic currents in the membrane
            equation. Buffered for one step (applied in the **next** update call).
            Default is ``0. * u.pA``.

        Returns
        -------
        jax.Array
            Binary spike tensor with dtype ``jnp.float64`` and shape
            ``self.V.value.shape``. A value of ``1.0`` indicates at least one
            internal spike event occurred during the integrated interval
            :math:`(t, t+dt]`.

        Notes
        -----
        - The external current ``x`` is **buffered**: the current passed in step
          :math:`t` affects the dynamics at step :math:`t+1`. This matches NEST's
          ring-buffer semantics for device input.
        - Delta inputs (spike-driven) and current inputs (continuous) are summed via
          :meth:`sum_delta_inputs` and :meth:`sum_current_inputs` from the
          :class:`~brainpy_state._base.Dynamics` base class.
        - Spike weights are interpreted as current amplitudes (pA). To convert from
          conductance-based models, multiply weights by driving force.
        - The Clopath filtered voltages (``u_bar_plus``, ``u_bar_minus``, ``u_bar_bar``)
          are updated automatically as part of the ODE integration. External code
          (e.g., Clopath synapse models) can read these values after :meth:`update`
          completes.
        - Integration is performed with an adaptive vectorized RKF45 loop.
          All arithmetic is unit-aware via ``saiunit.math``.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        m = self.m.value  # dimensionless
        h_val = self.h.value  # dimensionless
        n = self.n.value  # dimensionless
        dI_ex = self.dI_syn_ex.value  # pA/ms
        I_ex = self.I_syn_ex.value  # pA
        dI_in = self.dI_syn_in.value  # pA/ms
        I_in = self.I_syn_in.value  # pA
        u_bp = self.u_bar_plus.value  # mV
        u_bm = self.u_bar_minus.value  # mV
        u_bb = self.u_bar_bar.value  # mV
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Record V_old before integration for spike detection
        V_old = V

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(
            V=V, m=m, h=h_val, n=n,
            dI_ex=dI_ex, I_ex=I_ex,
            dI_in=dI_in, I_in=I_in,
            u_bar_plus=u_bp, u_bar_minus=u_bm, u_bar_bar=u_bb,
        )
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            V_old=V_old,
            i_stim=i_stim,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V = ode_state.V
        m = ode_state.m
        h_val = ode_state.h
        n = ode_state.n
        dI_ex = ode_state.dI_ex
        I_ex = ode_state.I_ex
        dI_in = ode_state.dI_in
        I_in = ode_state.I_in
        u_bp = ode_state.u_bar_plus
        u_bm = ode_state.u_bar_minus
        u_bb = ode_state.u_bar_bar
        r = extra.r

        # Synaptic spike inputs (applied after integration).
        w_ex = self.sum_delta_inputs(u.math.zeros_like(self.I_syn_ex.value), label='w_ex')
        w_in = self.sum_delta_inputs(u.math.zeros_like(self.I_syn_in.value), label='w_in')
        pscon_ex = np.e / self.tau_syn_ex  # 1/ms
        pscon_in = np.e / self.tau_syn_in  # 1/ms

        # Apply synaptic spike inputs.
        # w_ex is positive (excitatory magnitude); w_in is positive (inhibitory magnitude,
        # negated here to produce a negative dI_in, matching the inhibitory convention).
        dI_ex = dI_ex + pscon_ex * w_ex  # pA/ms + 1/ms * pA = pA/ms
        dI_in = dI_in - pscon_in * w_in  # pA/ms - 1/ms * pA = pA/ms (negative = inhibitory)

        # Spike detection: threshold crossing + local maximum
        # Use V_old (pre-integration voltage) not extra.V_old (integrator-internal),
        # matching hh_psc_alpha spike detection logic.
        not_refractory = r == 0
        crossed_threshold = V >= 0.0 * u.mV
        local_max = V_old > V
        spike_cond = not_refractory & crossed_threshold & local_max

        # Refractory update
        r_new = u.math.where(spike_cond, self.ref_count, u.math.where(r > 0, r - 1, r))

        # Write back state.
        self.V.value = V
        self.m.value = m
        self.h.value = h_val
        self.n.value = n
        self.I_syn_ex.value = I_ex
        self.I_syn_in.value = I_in
        self.dI_syn_ex.value = dI_ex
        self.dI_syn_in.value = dI_in
        self.u_bar_plus.value = u_bp
        self.u_bar_minus.value = u_bm
        self.u_bar_bar.value = u_bb
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r_new), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_cond, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return u.math.asarray(spike_cond, dtype=dftype)
