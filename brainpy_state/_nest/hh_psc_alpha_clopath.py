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
from scipy.integrate import solve_ivp

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

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
        Membrane potential in mV (unitless, not a ``brainunit`` quantity).

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


class hh_psc_alpha_clopath(Neuron):
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

    Uses ``scipy.integrate.solve_ivp`` with method ``'RK45'`` (Dormand-Prince)
    to match NEST's GSL RKF45 adaptive integrator. Default tolerances are
    ``rtol=1e-3`` and ``atol=1e-9``. Each neuron is integrated independently,
    allowing heterogeneous parameters but limiting vectorization.

    **5. Assumptions, Constraints, and Computational Implications**

    - ``C_m > 0``, ``g_Na >= 0``, ``g_K >= 0``, ``g_L >= 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, ``tau_u_bar_plus > 0``,
      ``tau_u_bar_minus > 0``, ``tau_u_bar_bar > 0``, and ``t_ref >= 0``
      are enforced at construction.
    - External current ``update(x=...)`` is buffered for one step, matching
      NEST ring-buffer semantics.
    - The adaptive RK45 integrator performs per-neuron integration in a Python
      loop, so vectorization is limited. For large populations, performance is
      lower than fixed-step models.
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
    spk_fun : Callable, optional
        Surrogate spike nonlinearity used by :meth:`get_spike` for differentiable
        spike generation. Default is ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy inherited from :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` applies stop-gradient to match NEST hard reset semantics.
        Default is ``'hard'``.
    rtol : float, optional
        Relative tolerance for the RK45 ODE solver. Smaller values increase
        accuracy at the cost of smaller integration steps. Default is ``1e-3``.
    atol : float, optional
        Absolute tolerance for the RK45 ODE solver. Default is ``1e-9``.
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
       * - ``rtol``
         - float, ``> 0``
         - ``1e-3``
         - --
         - Relative tolerance for RK45 adaptive integration.
       * - ``atol``
         - float, ``> 0``
         - ``1e-9``
         - --
         - Absolute tolerance for RK45 adaptive integration.

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential :math:`V_m`. Shape: ``(*in_size, *batch_size)``.
        Units: mV.
    m : brainstate.HiddenState
        Na activation gating variable (dimensionless). Shape: ``(*in_size, *batch_size)``.
        Range: [0, 1].
    h : brainstate.HiddenState
        Na inactivation gating variable (dimensionless). Shape: ``(*in_size, *batch_size)``.
        Range: [0, 1].
    n : brainstate.HiddenState
        K activation gating variable (dimensionless). Shape: ``(*in_size, *batch_size)``.
        Range: [0, 1].
    I_syn_ex : brainstate.ShortTermState
        Excitatory postsynaptic current :math:`I_{syn,ex}`. Shape: ``(*in_size, *batch_size)``.
        Units: pA.
    I_syn_in : brainstate.ShortTermState
        Inhibitory postsynaptic current :math:`I_{syn,in}`. Shape: ``(*in_size, *batch_size)``.
        Units: pA.
    dI_syn_ex : brainstate.ShortTermState
        Excitatory alpha-kernel derivative state (dimensionless). Shape: ``(*in_size, *batch_size)``.
    dI_syn_in : brainstate.ShortTermState
        Inhibitory alpha-kernel derivative state (dimensionless). Shape: ``(*in_size, *batch_size)``.
    u_bar_plus : brainstate.HiddenState
        Slow-filtered voltage :math:`\bar{u}_+` for Clopath LTP. Shape: ``(*in_size, *batch_size)``.
        Units: mV.
    u_bar_minus : brainstate.HiddenState
        Fast-filtered voltage :math:`\bar{u}_-` for Clopath LTD. Shape: ``(*in_size, *batch_size)``.
        Units: mV.
    u_bar_bar : brainstate.HiddenState
        Second-stage filtered voltage :math:`\bar{\bar{u}}` for Clopath homeostasis.
        Shape: ``(*in_size, *batch_size)``. Units: mV.
    I_stim : brainstate.ShortTermState
        One-step delayed external current buffer. Shape: ``(*in_size, *batch_size)``.
        Units: pA.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory steps. Shape: ``(*in_size, *batch_size)``. Dtype: int32.
    last_spike_time : brainstate.ShortTermState
        Time of most recent spike emission. Shape: ``(*in_size, *batch_size)``.
        Units: ms. Updated to ``t + dt`` on spike emission.

    Raises
    ------
    ValueError
        If any of the following conditions are violated:
        - ``C_m <= 0``
        - ``t_ref < 0``
        - ``tau_syn_ex <= 0`` or ``tau_syn_in <= 0``
        - ``tau_u_bar_plus <= 0``, ``tau_u_bar_minus <= 0``, or ``tau_u_bar_bar <= 0``
        - ``g_Na < 0``, ``g_K < 0``, or ``g_L < 0``

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
    - The adaptive RK45 integrator evaluates the ODE right-hand side multiple
      times per step, so computation cost scales with desired accuracy (controlled
      by ``rtol`` and ``atol``).
    - Spike detection combines threshold crossing (0 mV) and local maximum
      detection, matching the biological action potential waveform.

    References
    ----------
    .. [1] Hodgkin AL, Huxley AF (1952). A quantitative description of membrane
           current and its application to conduction and excitation in nerve.
           The Journal of Physiology 117:500-544.
           DOI: https://doi.org/10.1113/jphysiol.1952.sp004764
    .. [2] Clopath C, Büsing L, Vasilaki E, Gerstner W (2010). Connectivity
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
        >>> import brainunit as u
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
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        rtol: float = 1e-3,
        atol: float = 1e-9,
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
        self.rtol = rtol
        self.atol = atol

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        """Convert brainunit quantity to unitless NumPy float64 array.

        Parameters
        ----------
        x : ArrayLike
            Quantity with physical units (``brainunit.Quantity``).
        unit : brainunit.Unit
            Target unit for conversion.

        Returns
        -------
        numpy.ndarray
            Unitless float64 array with values in the target unit.
        """
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        """Broadcast NumPy array to target state shape.

        Parameters
        ----------
        x_np : numpy.ndarray
            Input array (typically scalar or 1D).
        shape : tuple
            Target shape for broadcasting.

        Returns
        -------
        numpy.ndarray
            Broadcasted array with shape ``shape``, sharing memory with ``x_np``
            if possible.
        """
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        """Validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If any parameter violates physical or numerical constraints:
            - ``C_m <= 0``
            - ``t_ref < 0``
            - ``tau_syn_ex <= 0`` or ``tau_syn_in <= 0``
            - ``tau_u_bar_plus <= 0``, ``tau_u_bar_minus <= 0``, or ``tau_u_bar_bar <= 0``
            - ``g_Na < 0``, ``g_K < 0``, or ``g_L < 0``
        """
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_u_bar_plus, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_u_bar_minus, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_u_bar_bar, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.g_Na, u.nS) < 0.0) or np.any(self._to_numpy(self.g_K, u.nS) < 0.0) or np.any(self._to_numpy(self.g_L, u.nS) < 0.0):
            raise ValueError('All conductances must be non-negative.')

    def _refractory_counts(self):
        """Compute refractory period duration in integer time steps.

        Converts the refractory period ``t_ref`` (in ms) to an integer number
        of simulation steps using the current simulation time step ``dt``. The
        result is rounded up via ``ceil`` to ensure at least ``t_ref`` duration.

        Returns
        -------
        jax.numpy.ndarray
            Integer array with dtype ``int32`` and shape matching ``self.varshape``,
            representing the number of steps to suppress spike emission after each
            spike. If ``t_ref`` is heterogeneous (per-neuron), the result is also
            heterogeneous.

        Notes
        -----
        - Called once during :meth:`update` to determine the refractory counter
          reset value upon spike detection.
        - For ``t_ref = 2 ms`` and ``dt = 0.1 ms``, returns 20 steps.
        """
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize all neuron state variables.

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
          ``last_spike_time`` set to -1e7 ms (far past).

        All states are broadcast to shape ``(*in_size, *batch_size)`` when ``batch_size``
        is specified.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional batch dimension prepended to all state shapes. If ``None``,
            states have shape ``self.varshape`` (derived from ``in_size``). If
            an integer, states have shape ``(*batch_size_tuple, *self.varshape)``.
        **kwargs
            Reserved for future extensions; currently ignored.

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
        V_init_mV = self._to_numpy(self.V_m_init, u.mV)
        V_init_scalar = float(V_init_mV.flat[0]) if V_init_mV.ndim > 0 else float(V_init_mV)

        # Compute equilibrium gating variables at initial V
        m_eq, h_eq, n_eq = _hh_psc_alpha_clopath_equilibrium(V_init_scalar)

        V = braintools.init.param(braintools.init.Constant(self.V_m_init), self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        if self.Act_m_init is not None:
            m_init = self._to_numpy(self.Act_m_init, u.UNITLESS).item()
        else:
            m_init = m_eq
        if self.Inact_h_init is not None:
            h_init = self._to_numpy(self.Inact_h_init, u.UNITLESS).item()
        else:
            h_init = h_eq
        if self.Act_n_init is not None:
            n_init = self._to_numpy(self.Act_n_init, u.UNITLESS).item()
        else:
            n_init = n_eq

        # Clopath filtered voltage initial values
        u_bar_plus_init_mV = float(self._to_numpy(self.u_bar_plus_init, u.mV))
        u_bar_minus_init_mV = float(self._to_numpy(self.u_bar_minus_init, u.mV))
        u_bar_bar_init_mV = float(self._to_numpy(self.u_bar_bar_init, u.mV))

        self.V = brainstate.HiddenState(V)
        self.m = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(m_init), self.varshape, batch_size)
        )
        self.h = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(h_init), self.varshape, batch_size)
        )
        self.n = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(n_init), self.varshape, batch_size)
        )
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.dI_syn_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.dI_syn_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.u_bar_plus = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(u_bar_plus_init_mV * u.mV), self.varshape, batch_size)
        )
        self.u_bar_minus = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(u_bar_minus_init_mV * u.mV), self.varshape, batch_size)
        )
        self.u_bar_bar = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(u_bar_bar_init_mV * u.mV), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(zeros * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

    def get_spike(self, V: ArrayLike = None):
        """Generate differentiable spike output via surrogate gradient function.

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
            >>> import brainunit as u
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

    def update(self, x=0. * u.pA):
        r"""Update neuron state for one simulation step.

        Advances the 11-dimensional Hodgkin-Huxley dynamics by one time step,
        including membrane potential, gating variables, synaptic currents, and
        three Clopath filtered voltage traces. Follows NEST's exact update order
        for ``hh_psc_alpha_clopath`` with adaptive RK45 integration.

        **Update sequence:**

        1. Record pre-integration membrane potential (``V_old``) for spike detection.
        2. Integrate the full 11-dimensional ODE system
           :math:`(V_m, m, h, n, dI_{ex}, I_{ex}, dI_{in}, I_{in}, \bar{u}_+, \bar{u}_-, \bar{\bar{u}})`
           over one time step :math:`[t, t+dt]` using adaptive RK45 (Dormand-Prince
           method) with tolerances ``rtol`` and ``atol``.
        3. Add arriving synaptic spike inputs to :math:`dI_{syn,ex}` and
           :math:`dI_{syn,in}` (spike weights split by sign; positive → excitatory,
           negative → inhibitory).
        4. Check spike condition:
           ``V_m >= 0 mV`` **and** ``V_old > V_m`` **and** ``refractory_step_count == 0``.
        5. Update refractory counter: set to ``ceil(t_ref / dt)`` on spike, otherwise
           decrement if positive.
        6. Record spike time as ``t + dt`` if spike detected.
        7. Store buffered external stimulation current ``x`` for the next step
           (one-step delay, matching NEST ring-buffer semantics).

        **Integration details:**

        - Uses ``scipy.integrate.solve_ivp`` with method ``'RK45'`` for each neuron
          independently (Python loop over flattened neuron array).
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
        ArrayLike
            Spike output array with shape ``(*batch_size, *in_size)``. Values are
            differentiable surrogate gradients produced by ``self.spk_fun`` applied
            to a spike indicator (1e-12 for spiking neurons, -1.0 otherwise). For
            binary spike trains, threshold at 0.

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
        - Computational cost scales roughly as :math:`O(N_{\text{neurons}} \times N_{\text{RK45 steps}})`.
          For large populations with ``rtol=1e-3``, expect 5-10 RK45 steps per
          simulation time step on average.

        Examples
        --------
        Inject constant current and simulate:

        .. code-block:: python

            >>> import brainstate as bst
            >>> import brainpy_state as bps
            >>> import brainunit as u
            >>> bst.environ.set(dt=0.1 * u.ms)
            >>> neurons = bps.hh_psc_alpha_clopath(in_size=10)
            >>> neurons.init_state()
            >>> # Inject 500 pA for 10 ms
            >>> for _ in range(100):
            ...     spikes = neurons.update(500. * u.pA)
            >>> # Check filtered voltages for plasticity
            >>> print(neurons.u_bar_plus.value)  # Slow LTP filter
            >>> print(neurons.u_bar_minus.value)  # Fast LTD filter

        Deliver spike input via projection:

        .. code-block:: python

            >>> # Assuming pre_spikes is a boolean array
            >>> neurons.add_delta_input('syn_exc', pre_spikes * 10. * u.pA)
            >>> spikes = neurons.update()  # No external current
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        # Extract parameters as numpy float64
        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        g_Na = self._broadcast_to_state(self._to_numpy(self.g_Na, u.nS), v_shape)
        g_K = self._broadcast_to_state(self._to_numpy(self.g_K, u.nS), v_shape)
        g_L = self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape)
        E_Na = self._broadcast_to_state(self._to_numpy(self.E_Na, u.mV), v_shape)
        E_K = self._broadcast_to_state(self._to_numpy(self.E_K, u.mV), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        tau_ubp = self._broadcast_to_state(self._to_numpy(self.tau_u_bar_plus, u.ms), v_shape)
        tau_ubm = self._broadcast_to_state(self._to_numpy(self.tau_u_bar_minus, u.ms), v_shape)
        tau_ubb = self._broadcast_to_state(self._to_numpy(self.tau_u_bar_bar, u.ms), v_shape)

        # Current state
        V_m = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        m_val = self._broadcast_to_state(np.asarray(self.m.value, dtype=np.float64), v_shape)
        h_val = self._broadcast_to_state(np.asarray(self.h.value, dtype=np.float64), v_shape)
        n_val = self._broadcast_to_state(np.asarray(self.n.value, dtype=np.float64), v_shape)
        dI_ex = self._broadcast_to_state(np.asarray(self.dI_syn_ex.value, dtype=np.float64), v_shape)
        I_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape)
        dI_in = self._broadcast_to_state(np.asarray(self.dI_syn_in.value, dtype=np.float64), v_shape)
        I_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape)
        ubp = self._broadcast_to_state(self._to_numpy(self.u_bar_plus.value, u.mV), v_shape)
        ubm = self._broadcast_to_state(self._to_numpy(self.u_bar_minus.value, u.mV), v_shape)
        ubb = self._broadcast_to_state(self._to_numpy(self.u_bar_bar.value, u.mV), v_shape)
        I_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )

        # PSC normalization: e / tau ensures peak current = weight for weight=1.
        psc_init_ex = math.e / tau_ex
        psc_init_in = math.e / tau_in

        # Collect spike/current inputs
        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all > 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)
        I_stim_next = self._broadcast_to_state(
            self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape
        )

        # Record V before integration for spike detection
        V_old = V_m.copy()

        # Integrate ODE for each neuron independently
        flat_size = int(np.prod(v_shape)) if len(v_shape) > 0 else 1
        V_new = np.empty(flat_size, dtype=np.float64)
        m_new = np.empty(flat_size, dtype=np.float64)
        h_new = np.empty(flat_size, dtype=np.float64)
        n_new = np.empty(flat_size, dtype=np.float64)
        dI_ex_new = np.empty(flat_size, dtype=np.float64)
        I_ex_new = np.empty(flat_size, dtype=np.float64)
        dI_in_new = np.empty(flat_size, dtype=np.float64)
        I_in_new = np.empty(flat_size, dtype=np.float64)
        ubp_new = np.empty(flat_size, dtype=np.float64)
        ubm_new = np.empty(flat_size, dtype=np.float64)
        ubb_new = np.empty(flat_size, dtype=np.float64)

        V_m_flat = V_m.ravel()
        m_flat = m_val.ravel()
        h_flat = h_val.ravel()
        n_flat = n_val.ravel()
        dI_ex_flat = dI_ex.ravel()
        I_ex_flat = I_ex.ravel()
        dI_in_flat = dI_in.ravel()
        I_in_flat = I_in.ravel()
        ubp_flat = ubp.ravel()
        ubm_flat = ubm.ravel()
        ubb_flat = ubb.ravel()
        I_stim_flat = I_stim.ravel()
        g_Na_flat = g_Na.ravel()
        g_K_flat = g_K.ravel()
        g_L_flat = g_L.ravel()
        E_Na_flat = E_Na.ravel()
        E_K_flat = E_K.ravel()
        E_L_flat = E_L.ravel()
        C_m_flat = C_m.ravel()
        I_e_flat = I_e.ravel()
        tau_ex_flat = tau_ex.ravel()
        tau_in_flat = tau_in.ravel()
        tau_ubp_flat = tau_ubp.ravel()
        tau_ubm_flat = tau_ubm.ravel()
        tau_ubb_flat = tau_ubb.ravel()

        for i in range(flat_size):
            y0 = np.array([
                V_m_flat[i], m_flat[i], h_flat[i], n_flat[i],
                dI_ex_flat[i], I_ex_flat[i], dI_in_flat[i], I_in_flat[i],
                ubp_flat[i], ubm_flat[i], ubb_flat[i]
            ])

            # Capture per-neuron parameters for closure
            _g_Na = g_Na_flat[i]
            _g_K = g_K_flat[i]
            _g_L = g_L_flat[i]
            _E_Na = E_Na_flat[i]
            _E_K = E_K_flat[i]
            _E_L = E_L_flat[i]
            _C_m = C_m_flat[i]
            _I_e = I_e_flat[i]
            _I_stim = I_stim_flat[i]
            _tau_ex = tau_ex_flat[i]
            _tau_in = tau_in_flat[i]
            _tau_ubp = tau_ubp_flat[i]
            _tau_ubm = tau_ubm_flat[i]
            _tau_ubb = tau_ubb_flat[i]

            def rhs(t_local, y,
                    _g_Na=_g_Na, _g_K=_g_K, _g_L=_g_L,
                    _E_Na=_E_Na, _E_K=_E_K, _E_L=_E_L,
                    _C_m=_C_m, _I_e=_I_e, _I_stim=_I_stim,
                    _tau_ex=_tau_ex, _tau_in=_tau_in,
                    _tau_ubp=_tau_ubp, _tau_ubm=_tau_ubm, _tau_ubb=_tau_ubb):
                V = y[0]
                m_ = y[1]
                h_ = y[2]
                n_ = y[3]
                dI_e = y[4]
                I_e_ = y[5]
                dI_i = y[6]
                I_i_ = y[7]
                u_bar_plus_ = y[8]
                u_bar_minus_ = y[9]
                u_bar_bar_ = y[10]

                alpha_n = (0.01 * (V + 55.0)) / (1.0 - math.exp(-(V + 55.0) / 10.0))
                beta_n = 0.125 * math.exp(-(V + 65.0) / 80.0)
                alpha_m = (0.1 * (V + 40.0)) / (1.0 - math.exp(-(V + 40.0) / 10.0))
                beta_m = 4.0 * math.exp(-(V + 65.0) / 18.0)
                alpha_h = 0.07 * math.exp(-(V + 65.0) / 20.0)
                beta_h = 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))

                I_Na = _g_Na * m_ * m_ * m_ * h_ * (V - _E_Na)
                I_K = _g_K * n_ * n_ * n_ * n_ * (V - _E_K)
                I_L = _g_L * (V - _E_L)

                f = np.empty(11)
                f[0] = (-(I_Na + I_K + I_L) + _I_stim + _I_e + I_e_ + I_i_) / _C_m
                f[1] = alpha_m * (1.0 - m_) - beta_m * m_
                f[2] = alpha_h * (1.0 - h_) - beta_h * h_
                f[3] = alpha_n * (1.0 - n_) - beta_n * n_
                f[4] = -dI_e / _tau_ex
                f[5] = dI_e - (I_e_ / _tau_ex)
                f[6] = -dI_i / _tau_in
                f[7] = dI_i - (I_i_ / _tau_in)
                # Clopath filtered voltage traces
                f[8] = (-u_bar_plus_ + V) / _tau_ubp
                f[9] = (-u_bar_minus_ + V) / _tau_ubm
                f[10] = (-u_bar_bar_ + u_bar_minus_) / _tau_ubb
                return f

            sol = solve_ivp(
                rhs,
                [0.0, h],
                y0,
                method='RK45',
                rtol=self.rtol,
                atol=self.atol,
                dense_output=False,
            )
            yf = sol.y[:, -1]
            V_new[i] = yf[0]
            m_new[i] = yf[1]
            h_new[i] = yf[2]
            n_new[i] = yf[3]
            dI_ex_new[i] = yf[4]
            I_ex_new[i] = yf[5]
            dI_in_new[i] = yf[6]
            I_in_new[i] = yf[7]
            ubp_new[i] = yf[8]
            ubm_new[i] = yf[9]
            ubb_new[i] = yf[10]

        V_m = V_new.reshape(v_shape)
        m_val = m_new.reshape(v_shape)
        h_val = h_new.reshape(v_shape)
        n_val = n_new.reshape(v_shape)
        dI_ex = dI_ex_new.reshape(v_shape)
        I_ex = I_ex_new.reshape(v_shape)
        dI_in = dI_in_new.reshape(v_shape)
        I_in = I_in_new.reshape(v_shape)
        ubp_val = ubp_new.reshape(v_shape)
        ubm_val = ubm_new.reshape(v_shape)
        ubb_val = ubb_new.reshape(v_shape)

        # Add arriving spike inputs to dI (after ODE integration, matching NEST)
        dI_ex = dI_ex + w_ex * psc_init_ex
        dI_in = dI_in + w_in * psc_init_in

        # Spike detection: threshold crossing + local maximum
        not_refractory = r == 0
        crossed_threshold = V_m >= 0.0
        local_max = V_old > V_m
        spike_cond = not_refractory & crossed_threshold & local_max

        # Refractory update
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )
        r_new = np.where(spike_cond, refr_counts, np.where(r > 0, r - 1, r))

        # Write back state
        self.V.value = V_m * u.mV
        self.m.value = m_val
        self.h.value = h_val
        self.n.value = n_val
        self.I_syn_ex.value = I_ex * u.pA
        self.I_syn_in.value = I_in * u.pA
        self.dI_syn_ex.value = dI_ex
        self.dI_syn_in.value = dI_in
        self.u_bar_plus.value = ubp_val * u.mV
        self.u_bar_minus.value = ubm_val * u.mV
        self.u_bar_bar.value = ubb_val * u.mV
        self.I_stim.value = I_stim_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r_new, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        # Return spike output: only signal a spike when spike_cond is True
        V_out = np.where(spike_cond, 1e-12, -1.0)
        return self.get_spike(V_out * u.mV)
