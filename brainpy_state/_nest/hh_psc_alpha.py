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


class hh_psc_alpha(Neuron):
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

    Uses ``scipy.integrate.solve_ivp`` with method ``'RK45'`` (Dormand-Prince)
    to match NEST's GSL RKF45 adaptive integrator. Default tolerances are
    ``rtol=1e-3`` and ``atol=1e-9``. Each neuron is integrated independently.

    **5. Assumptions, Constraints, and Computational Implications**

    - ``C_m > 0``, ``g_Na >= 0``, ``g_K >= 0``, ``g_L >= 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, and ``t_ref >= 0`` are enforced
      at construction.
    - External current ``update(x=...)`` is buffered for one step, matching
      NEST ring-buffer semantics.
    - The adaptive RK45 integrator performs per-neuron integration in a Python
      loop, so vectorization is limited. For large populations, performance is
      lower than fixed-step models.
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
        Membrane potential :math:`V_m`. Shape: ``(\*in_size, \*batch_size)``.
        Units: mV.
    m : brainstate.HiddenState
        Na activation gating variable (dimensionless). Shape: ``(\*in_size, \*batch_size)``.
        Range: [0, 1].
    h : brainstate.HiddenState
        Na inactivation gating variable (dimensionless). Shape: ``(\*in_size, \*batch_size)``.
        Range: [0, 1].
    n : brainstate.HiddenState
        K activation gating variable (dimensionless). Shape: ``(\*in_size, \*batch_size)``.
        Range: [0, 1].
    I_syn_ex : brainstate.ShortTermState
        Excitatory postsynaptic current :math:`I_{syn,ex}`. Shape: ``(\*in_size, \*batch_size)``.
        Units: pA.
    I_syn_in : brainstate.ShortTermState
        Inhibitory postsynaptic current :math:`I_{syn,in}`. Shape: ``(\*in_size, \*batch_size)``.
        Units: pA.
    dI_syn_ex : brainstate.ShortTermState
        Excitatory alpha-kernel derivative state (dimensionless). Shape: ``(\*in_size, \*batch_size)``.
    dI_syn_in : brainstate.ShortTermState
        Inhibitory alpha-kernel derivative state (dimensionless). Shape: ``(\*in_size, \*batch_size)``.
    I_stim : brainstate.ShortTermState
        One-step delayed external current buffer. Shape: ``(\*in_size, \*batch_size)``.
        Units: pA.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory steps. Shape: ``(\*in_size, \*batch_size)``. Dtype: int32.
    last_spike_time : brainstate.ShortTermState
        Time of most recent spike emission. Shape: ``(\*in_size, \*batch_size)``.
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
      by ``rtol`` and ``atol``).
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
       >>> import brainpy.state as bps
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
        self.V_m_init = V_m_init
        self.Act_m_init = Act_m_init
        self.Inact_h_init = Inact_h_init
        self.Act_n_init = Act_n_init
        self.rtol = rtol
        self.atol = atol

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        r"""Convert a BrainUnit quantity to a NumPy float64 array.

        Parameters
        ----------
        x : ArrayLike
            BrainUnit quantity or scalar.
        unit : brainunit.Unit
            Unit to divide by before conversion.

        Returns
        -------
        np.ndarray
            Unitless NumPy array with dtype ``float64``.
        """
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast a NumPy array to match state shape.

        Parameters
        ----------
        x_np : np.ndarray
            Input array to broadcast.
        shape : tuple of int
            Target shape.

        Returns
        -------
        np.ndarray
            Broadcasted array with shape ``shape``.
        """
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        r"""Validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If ``C_m <= 0``, ``t_ref < 0``, ``tau_syn_ex <= 0``, ``tau_syn_in <= 0``,
            or any conductance is negative.
        """
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.g_Na, u.nS) < 0.0) or np.any(self._to_numpy(self.g_K, u.nS) < 0.0) or np.any(self._to_numpy(self.g_L, u.nS) < 0.0):
            raise ValueError('All conductances must be non-negative.')

    def _refractory_counts(self):
        r"""Convert refractory period to integer grid step counts.

        Returns
        -------
        jax.Array
            Integer array with shape ``self.varshape``, each element giving
            the number of time steps corresponding to the refractory period
            ``ceil(t_ref / dt)``.
        """
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Sets initial values for membrane potential, gating variables, synaptic
        currents, refractory counters, and buffers. If gating variable initial
        values are not explicitly provided, they are computed at equilibrium for
        the given initial membrane potential.

        Parameters
        ----------
        batch_size : int or None, optional
            Batch dimension size. If ``None``, no batch dimension is added.
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
            - ``last_spike_time``: spike time record (initialized to -1e7 ms)
        """
        V_init_mV = self._to_numpy(self.V_m_init, u.mV)
        V_init_scalar = float(V_init_mV.flat[0]) if V_init_mV.ndim > 0 else float(V_init_mV)

        # Compute equilibrium gating variables at initial V
        m_eq, h_eq, n_eq = _hh_psc_alpha_equilibrium(V_init_scalar)

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
        self.I_stim = brainstate.ShortTermState(zeros * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

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
            Differentiable spike signal with shape ``(\*in_size, \*batch_size)``.
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
            Shape must broadcast with ``(\*in_size, \*batch_size)``.
            Default is ``0. * u.pA``.

        Returns
        -------
        ArrayLike
            Differentiable spike output with shape ``(\*in_size, \*batch_size)``.
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
        - The RK45 integrator is called once per neuron per step with a
          Python loop, limiting vectorization. For large populations, this
          may be slower than fixed-step models.
        - Spike detection combines threshold crossing (0 mV) and local maximum
          detection (``V_old > V_m``) to match biological action potential
          characteristics.
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

        # Current state
        V_m = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        m_val = self._broadcast_to_state(np.asarray(self.m.value, dtype=np.float64), v_shape)
        h_val = self._broadcast_to_state(np.asarray(self.h.value, dtype=np.float64), v_shape)
        n_val = self._broadcast_to_state(np.asarray(self.n.value, dtype=np.float64), v_shape)
        dI_ex = self._broadcast_to_state(np.asarray(self.dI_syn_ex.value, dtype=np.float64), v_shape)
        I_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape)
        dI_in = self._broadcast_to_state(np.asarray(self.dI_syn_in.value, dtype=np.float64), v_shape)
        I_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape)
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

        V_m_flat = V_m.ravel()
        m_flat = m_val.ravel()
        h_flat = h_val.ravel()
        n_flat = n_val.ravel()
        dI_ex_flat = dI_ex.ravel()
        I_ex_flat = I_ex.ravel()
        dI_in_flat = dI_in.ravel()
        I_in_flat = I_in.ravel()
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

        for i in range(flat_size):
            y0 = np.array([
                V_m_flat[i], m_flat[i], h_flat[i], n_flat[i],
                dI_ex_flat[i], I_ex_flat[i], dI_in_flat[i], I_in_flat[i]
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

            def rhs(t_local, y,
                    _g_Na=_g_Na, _g_K=_g_K, _g_L=_g_L,
                    _E_Na=_E_Na, _E_K=_E_K, _E_L=_E_L,
                    _C_m=_C_m, _I_e=_I_e, _I_stim=_I_stim,
                    _tau_ex=_tau_ex, _tau_in=_tau_in):
                V = y[0]
                m_ = y[1]
                h_ = y[2]
                n_ = y[3]
                dI_e = y[4]
                I_e_ = y[5]
                dI_i = y[6]
                I_i_ = y[7]

                alpha_n = (0.01 * (V + 55.0)) / (1.0 - math.exp(-(V + 55.0) / 10.0))
                beta_n = 0.125 * math.exp(-(V + 65.0) / 80.0)
                alpha_m = (0.1 * (V + 40.0)) / (1.0 - math.exp(-(V + 40.0) / 10.0))
                beta_m = 4.0 * math.exp(-(V + 65.0) / 18.0)
                alpha_h = 0.07 * math.exp(-(V + 65.0) / 20.0)
                beta_h = 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))

                I_Na = _g_Na * m_ * m_ * m_ * h_ * (V - _E_Na)
                I_K = _g_K * n_ * n_ * n_ * n_ * (V - _E_K)
                I_L = _g_L * (V - _E_L)

                f = np.empty(8)
                f[0] = (-(I_Na + I_K + I_L) + _I_stim + _I_e + I_e_ + I_i_) / _C_m
                f[1] = alpha_m * (1.0 - m_) - beta_m * m_
                f[2] = alpha_h * (1.0 - h_) - beta_h * h_
                f[3] = alpha_n * (1.0 - n_) - beta_n * n_
                f[4] = -dI_e / _tau_ex
                f[5] = dI_e - (I_e_ / _tau_ex)
                f[6] = -dI_i / _tau_in
                f[7] = dI_i - (I_i_ / _tau_in)
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

        V_m = V_new.reshape(v_shape)
        m_val = m_new.reshape(v_shape)
        h_val = h_new.reshape(v_shape)
        n_val = n_new.reshape(v_shape)
        dI_ex = dI_ex_new.reshape(v_shape)
        I_ex = I_ex_new.reshape(v_shape)
        dI_in = dI_in_new.reshape(v_shape)
        I_in = I_in_new.reshape(v_shape)

        # Add arriving spike inputs to dI (after ODE integration, matching NEST)
        dI_ex = dI_ex + w_ex * psc_init_ex
        dI_in = dI_in + w_in * psc_init_in

        # Spike detection: threshold crossing + local maximum
        V_old_flat = V_old.ravel()
        V_m_flat = V_m.ravel()

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
        self.I_stim.value = I_stim_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r_new, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        # Return spike output: only signal a spike when spike_cond is True
        V_out = np.where(spike_cond, 1e-12, -1.0)
        return self.get_spike(V_out * u.mV)
