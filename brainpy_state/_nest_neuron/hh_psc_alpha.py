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
from brainstate.util import DotDict

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'hh_psc_alpha',
]


def _hh_psc_alpha_equilibrium(V):
    r"""Compute equilibrium values of Hodgkin-Huxley gating variables.

    Calculates steady-state activation and inactivation at a given membrane
    potential using the voltage-dependent rate functions from NEST's
    ``hh_psc_alpha`` model. These equilibria are used for state initialization
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


class hh_psc_alpha(NESTNeuron):
    r"""NEST-compatible Hodgkin-Huxley neuron with alpha-shaped postsynaptic currents.

    Current-based spiking neuron using the Hodgkin-Huxley formalism with
    voltage-gated sodium and potassium channels, leak conductance, alpha-function
    postsynaptic currents, threshold-and-local-maximum spike detection, and an
    explicit refractory period that suppresses spike emission only (subthreshold
    dynamics continue freely). Follows NEST ``models/hh_psc_alpha.{h,cpp}``
    implementation with adaptive Runge-Kutta integration (RK45).

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
    state variables continue evolving according to their differential equations.

    **3. Update Order Per Simulation Step**

    The update follows NEST's exact order:

    1. Record pre-integration membrane potential (``V_old``).
    2. Integrate the full 8-dimensional ODE system
       :math:`(V_m, m, h, n, dI_{ex}, I_{ex}, dI_{in}, I_{in})` over one
       time step :math:`[t, t+dt]` using adaptive RK45 (Dormand-Prince).
    3. Add arriving synaptic spike inputs to :math:`dI_{syn,ex}` and
       :math:`dI_{syn,in}`.
    4. Check spike condition: ``V_m >= 0 and V_old > V_m and r == 0``.
    5. Update refractory counter and record spike time.
    6. Store buffered external stimulation current for the next step.

    **4. Numerical Integration**

    Uses ``AdaptiveRungeKuttaStep`` with method ``'RKF45'`` to match NEST's
    GSL RKF45 adaptive integrator. Default tolerance is ``gsl_error_tol=1e-3``.
    All neurons are integrated simultaneously in a vectorized fashion.

    **5. Assumptions, Constraints, and Computational Implications**

    - ``C_m > 0``, ``g_Na >= 0``, ``g_K >= 0``, ``g_L >= 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, and ``t_ref >= 0`` are enforced
      at construction.
    - External current ``update(x=...)`` is buffered for one step, matching
      NEST ring-buffer semantics.
    - The adaptive RK45 integrator performs vectorized integration across all
      neurons simultaneously using JAX operations.
    - Spike detection uses a local maximum criterion rather than threshold
      crossing alone, matching biological action potential dynamics.

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
    spk_fun : Callable, optional
        Surrogate spike nonlinearity used by :meth:`get_spike` for differentiable
        spike generation. Default is ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy inherited from :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` applies stop-gradient to match NEST hard reset semantics.
        Default is ``'hard'``.
    gsl_error_tol : float, optional
        Unitless local RKF45 error tolerance, strictly positive.
        Default is ``1e-3``.
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
       * - ``gsl_error_tol``
         - float, ``> 0``
         - ``1e-3``
         - --
         - Local absolute tolerance for the embedded RKF45 error estimate.

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential :math:`V_m`. Shape: ``(*in_size,)``.
        Units: mV.
    m : brainstate.HiddenState
        Na activation gating variable (dimensionless). Shape: ``(*in_size,)``.
        Range: [0, 1].
    h : brainstate.HiddenState
        Na inactivation gating variable (dimensionless). Shape: ``(*in_size,)``.
        Range: [0, 1].
    n : brainstate.HiddenState
        K activation gating variable (dimensionless). Shape: ``(*in_size,)``.
        Range: [0, 1].
    I_syn_ex : brainstate.ShortTermState
        Excitatory postsynaptic current :math:`I_{syn,ex}`. Shape: ``(*in_size,)``.
        Units: pA.
    I_syn_in : brainstate.ShortTermState
        Inhibitory postsynaptic current :math:`I_{syn,in}`. Shape: ``(*in_size,)``.
        Units: pA.
    dI_syn_ex : brainstate.ShortTermState
        Excitatory alpha-kernel derivative state. Shape: ``(*in_size,)``.
        Units: pA/ms.
    dI_syn_in : brainstate.ShortTermState
        Inhibitory alpha-kernel derivative state. Shape: ``(*in_size,)``.
        Units: pA/ms.
    I_stim : brainstate.ShortTermState
        One-step delayed external current buffer. Shape: ``(*in_size,)``.
        Units: pA.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory steps. Shape: ``(*in_size,)``. Dtype: int32.
    integration_step : brainstate.ShortTermState
        Persistent RKF45 substep size estimate (ms).
    last_spike_time : brainstate.ShortTermState
        Time of most recent spike emission. Shape: ``(*in_size,)``.
        Units: ms. Updated to ``t + dt`` on spike emission.

    Raises
    ------
    ValueError
        If any of the following conditions are violated:
        - ``C_m <= 0``
        - ``t_ref < 0``
        - ``tau_syn_ex <= 0`` or ``tau_syn_in <= 0``
        - ``g_Na < 0``, ``g_K < 0``, or ``g_L < 0``

    Notes
    -----
    - Unlike IAF models, the HH model does **not** reset the membrane potential
      after a spike. Repolarization occurs naturally through the potassium current.
    - During the refractory period, the neuron's subthreshold dynamics continue
      to evolve freely; only spike emission is suppressed.
    - Spike weights are interpreted as current amplitudes (pA). Positive weights
      are excitatory; negative weights are inhibitory.
    - The adaptive RK45 integrator evaluates the ODE right-hand side multiple
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
    .. [2] Gerstner W, Kistler W (2002). Spiking neuron models: Single neurons,
           populations, plasticity. Cambridge University Press.
    .. [3] Dayan P, Abbott LF (2001). Theoretical neuroscience: Computational
           and mathematical modeling of neural systems. MIT Press.
    .. [4] NEST Simulator Documentation. hh_psc_alpha neuron model.
           https://nest-simulator.readthedocs.io/en/stable/models/hh_psc_alpha.html

    See Also
    --------
    iaf_psc_alpha : Leaky integrate-and-fire with alpha-shaped PSCs.
    hh_psc_alpha_clopath : HH neuron with Clopath voltage-based STDP.
    hh_psc_alpha_gap : HH neuron with gap junction support.

    Examples
    --------
    Create a single Hodgkin-Huxley neuron and observe spiking behavior under
    constant current injection:

    .. code-block:: python

       >>> import brainstate as bst
       >>> import brainunit as u
       >>> from brainpy import state as bps
       >>> import matplotlib.pyplot as plt
       >>> # Initialize simulation context
       >>> bst.environ.set(dt=0.1 * u.ms)
       >>> # Create neuron
       >>> neuron = bps.hh_psc_alpha(in_size=1, I_e=500. * u.pA)
       >>> neuron.init_all_states()
       >>> # Run simulation
       >>> times = []
       >>> voltages = []
       >>> for _ in range(2000):  # 200 ms
       ...     neuron.update()
       ...     times.append(float(bst.environ.get('t') / u.ms))
       ...     voltages.append(float(neuron.V.value / u.mV))
       >>> # Plot results
       >>> plt.plot(times, voltages)
       >>> plt.xlabel('Time (ms)')
       >>> plt.ylabel('Membrane potential (mV)')
       >>> plt.title('Hodgkin-Huxley neuron spiking')
       >>> plt.show()
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
        V_m_init: ArrayLike = -65. * u.mV,
        Act_m_init: ArrayLike = None,
        Inact_h_init: ArrayLike = None,
        Act_n_init: ArrayLike = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        gsl_error_tol: float = 1e-3,
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
        self.V_m_init = V_m_init
        self.Act_m_init = Act_m_init
        self.Inact_h_init = Inact_h_init
        self.Act_n_init = Act_n_init
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
            If ``C_m <= 0``, ``t_ref < 0``, ``tau_syn_ex <= 0``, ``tau_syn_in <= 0``,
            or any conductance is negative.
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
        if cond_any(self.g_Na < 0.0 * u.nS) or cond_any(self.g_K < 0.0 * u.nS) or cond_any(
            self.g_L < 0.0 * u.nS):
            raise ValueError('All conductances must be non-negative.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables.

        Sets initial values for membrane potential, gating variables, synaptic
        currents, refractory counters, and buffers. If gating variable initial
        values are not explicitly provided, they are computed at equilibrium for
        the given initial membrane potential.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments (unused, for compatibility).

        Notes
        -----
        State variables initialized:
            - ``V``: membrane potential (from ``V_m_init``)
            - ``m``, ``h``, ``n``: gating variables (from ``Act_m_init``,
              ``Inact_h_init``, ``Act_n_init`` if provided; otherwise computed
              at equilibrium for ``V_m_init``)
            - ``I_syn_ex``, ``I_syn_in``, ``dI_syn_ex``, ``dI_syn_in``: synaptic
              states (initialized to zero)
            - ``I_stim``: external current buffer (initialized to zero)
            - ``refractory_step_count``: refractory countdown (initialized to zero)
            - ``integration_step``: persistent RKF45 substep size
            - ``last_spike_time``: spike time record (initialized to -1e7 ms)
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V_init_mV = float(u.get_mantissa(u.math.asarray(self.V_m_init / u.mV)))

        # Compute equilibrium gating variables at initial V
        m_eq, h_eq, n_eq = _hh_psc_alpha_equilibrium(V_init_mV)

        V = braintools.init.param(braintools.init.Constant(self.V_m_init), self.varshape)

        if self.Act_m_init is not None:
            m_init = float(u.get_mantissa(u.math.asarray(self.Act_m_init)))
        else:
            m_init = m_eq
        if self.Inact_h_init is not None:
            h_init = float(u.get_mantissa(u.math.asarray(self.Inact_h_init)))
        else:
            h_init = h_eq
        if self.Act_n_init is not None:
            n_init = float(u.get_mantissa(u.math.asarray(self.Act_n_init)))
        else:
            n_init = n_eq

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

        zeros_pA_per_ms = u.math.zeros(self.varshape, dtype=dftype) * (u.pA / u.ms)
        self.dI_syn_ex = brainstate.ShortTermState(zeros_pA_per_ms)
        self.dI_syn_in = brainstate.ShortTermState(zeros_pA_per_ms)
        self.I_syn_ex = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=dftype) * u.pA)
        self.I_syn_in = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=dftype) * u.pA)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))

    def get_spike(self, V: ArrayLike = None):
        r"""Generate differentiable spike output using surrogate gradient.

        Applies the surrogate spike function to the membrane potential scaled
        relative to the 0 mV threshold. This enables gradient-based learning
        through the spike generation process.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Membrane potential in mV. If ``None``, uses ``self.V.value``.
            Shape must broadcast with ``self.varshape``.

        Returns
        -------
        ArrayLike
            Differentiable spike signal with shape ``(*in_size,)``.
            Typically near 0 for subthreshold, near 1 for suprathreshold.

        Notes
        -----
        The spike threshold for HH neurons is 0 mV. The input voltage is
        scaled relative to this threshold before applying the surrogate function.
        """
        V = self.V.value if V is None else V
        # For HH neurons, spike threshold is 0 mV. Scale relative to 0 mV.
        v_scaled = V / (1. * u.mV)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, dI_ex, I_ex, dI_in, I_in — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, V_old, i_stim — mutable auxiliary data
            carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        V = state.V
        m_ = state.m
        h_ = state.h
        n_ = state.n
        dI_ex = state.dI_ex
        I_ex = state.I_ex
        dI_in = state.dI_in
        I_in = state.I_in

        # Voltage in mV for rate functions (unitless computation)
        V_mV = V / u.mV

        alpha_n = (0.01 * (V_mV + 55.0)) / (1.0 - u.math.exp(-(V_mV + 55.0) / 10.0))
        beta_n = 0.125 * u.math.exp(-(V_mV + 65.0) / 80.0)
        alpha_m = (0.1 * (V_mV + 40.0)) / (1.0 - u.math.exp(-(V_mV + 40.0) / 10.0))
        beta_m = 4.0 * u.math.exp(-(V_mV + 65.0) / 18.0)
        alpha_h = 0.07 * u.math.exp(-(V_mV + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + u.math.exp(-(V_mV + 35.0) / 10.0))

        # Ionic currents (nS * mV = pA)
        I_Na = self.g_Na * m_ * m_ * m_ * h_ * (V - self.E_Na)
        I_K = self.g_K * n_ * n_ * n_ * n_ * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # Membrane voltage derivative: dV/dt = (-(I_Na + I_K + I_L) + I_stim + I_e + I_ex + I_in) / C_m
        dV = (-(I_Na + I_K + I_L) + extra.i_stim + self.I_e + I_ex + I_in) / self.C_m

        # Gating variable derivatives (rates are in 1/ms)
        dm = (alpha_m * (1.0 - m_) - beta_m * m_) / u.ms
        dh = (alpha_h * (1.0 - h_) - beta_h * h_) / u.ms
        dn = (alpha_n * (1.0 - n_) - beta_n * n_) / u.ms

        # Alpha-kernel synaptic current derivatives
        ddI_ex = -dI_ex / self.tau_syn_ex
        dI_ex_dt = dI_ex - I_ex / self.tau_syn_ex
        ddI_in = -dI_in / self.tau_syn_in
        dI_in_dt = dI_in - I_in / self.tau_syn_in

        return DotDict(V=dV, m=dm, h=dh, n=dn, dI_ex=ddI_ex, I_ex=dI_ex_dt, dI_in=ddI_in, I_in=dI_in_dt)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection using threshold-and-local-maximum criterion.

        For the HH model, spike detection occurs *after* integration in the
        update method, not inside the integration loop. This event function
        tracks V_old for local maximum detection but does not perform
        spike/reset inside the loop (since HH has no voltage reset).

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, dI_ex, I_ex, dI_in, I_in — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, V_old, i_stim.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated auxiliary info.
        """
        # Update V_old to track the previous accepted voltage for local max detection
        new_V_old = u.math.where(accept, state.V, extra.V_old)
        new_extra = DotDict({**extra, 'V_old': new_V_old})
        return state, new_extra

    def update(self, x=0. * u.pA):
        r"""Update neuron state for one simulation step.

        Integrates the full Hodgkin-Huxley dynamics over one time step :math:`dt`,
        applies synaptic inputs, detects spikes using threshold-and-local-maximum
        criterion, updates refractory state, and buffers external current for the
        next step. Follows NEST ``hh_psc_alpha`` update order exactly.

        **Update Order:**

        1. Record pre-integration membrane potential (``V_old``).
        2. Integrate the 8-dimensional ODE system
           :math:`(V_m, m, h, n, dI_{ex}, I_{ex}, dI_{in}, I_{in})` over
           :math:`[t, t+dt]` using adaptive RK45 (Dormand-Prince).
        3. Add arriving synaptic spike inputs to :math:`dI_{syn,ex}` and
           :math:`dI_{syn,in}`.
        4. Check spike condition:
           ``refractory_step_count == 0 and V_m >= 0 and V_old > V_m``.
        5. Update refractory counter and record spike time.
        6. Store buffered external stimulation current ``x`` for next step.

        Parameters
        ----------
        x : ArrayLike, optional
            External stimulation current input in pA (in addition to ``I_e``).
            Shape must broadcast with ``(*in_size,)``.
            Default is ``0. * u.pA``.

        Returns
        -------
        ArrayLike
            Differentiable spike output with shape ``(*in_size,)``.
            Generated by applying ``self.spk_fun`` to the spike condition.
            Near 1 when spike detected, near 0 otherwise.

        Notes
        -----
        - The external current ``x`` is buffered for one step via ``I_stim``,
          matching NEST's ring-buffer semantics. Current provided at step
          :math:`n` affects dynamics at step :math:`n+1`.
        - Spike weights are collected via ``sum_delta_inputs(0*pA)`` and split
          by sign: positive weights drive excitatory state, negative weights
          drive inhibitory state.
        - During the refractory period, all state variables evolve freely;
          only spike emission is suppressed.
        - Spike detection combines threshold crossing (0 mV) and local maximum
          detection (``V_old > V_m``) to match biological action potential
          characteristics.
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
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h_step = self.integration_step.value  # ms

        # Record V before integration for spike detection
        V_old = V

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, m=m, h=h_val, n=n, dI_ex=dI_ex, I_ex=I_ex, dI_in=dI_in, I_in=I_in)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            V_old=V_old,
            i_stim=i_stim,
        )

        ode_state, h_step, extra = self.integrator(state=ode_state, h=h_step, extra=extra)
        V = ode_state.V
        m = ode_state.m
        h_val = ode_state.h
        n = ode_state.n
        dI_ex = ode_state.dI_ex
        I_ex = ode_state.I_ex
        dI_in = ode_state.dI_in
        I_in = ode_state.I_in

        # Synaptic spike inputs (applied after integration, matching NEST).
        w_all = self.sum_delta_inputs(0. * u.pA)
        w_ex = u.math.where(w_all > 0.0 * u.pA, w_all, 0.0 * u.pA)
        w_in = u.math.where(w_all < 0.0 * u.pA, w_all, 0.0 * u.pA)

        # PSC normalization: e / tau ensures peak current = weight for weight=1.
        pscon_ex = np.e / self.tau_syn_ex  # 1/ms
        pscon_in = np.e / self.tau_syn_in  # 1/ms

        # Apply synaptic spike inputs.
        dI_ex = dI_ex + pscon_ex * w_ex  # pA/ms + 1/ms * pA = pA/ms
        dI_in = dI_in + pscon_in * w_in  # pA/ms + 1/ms * pA = pA/ms

        # Spike detection: threshold crossing + local maximum
        not_refractory = r == 0
        crossed_threshold = V >= 0.0 * u.mV
        local_max = V_old > V
        spike_mask = not_refractory & crossed_threshold & local_max

        # Refractory update
        r_new = u.math.where(spike_mask, self.ref_count, u.math.where(r > 0, r - 1, r))

        # Write back state.
        self.V.value = V
        self.m.value = m
        self.h.value = h_val
        self.n.value = n
        self.dI_syn_ex.value = dI_ex
        self.I_syn_ex.value = I_ex
        self.dI_syn_in.value = dI_in
        self.I_syn_in.value = I_in
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r_new), dtype=ditype)
        self.integration_step.value = h_step
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return u.math.asarray(spike_mask, dtype=dftype)
