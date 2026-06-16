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
    'hh_cond_beta_gap_traub',
]


class hh_cond_beta_gap_traub(NESTNeuron):
    r"""NEST-compatible Hodgkin-Huxley neuron with beta-function synapses and gap junctions.

    Implements a conductance-based Hodgkin-Huxley model with Traub-Miles gating
    kinetics, beta-function (double-exponential) synaptic conductances, and support
    for gap-junction coupling. Based on the NEST ``hh_cond_beta_gap_traub`` model.

    **1. Model Overview**

    This model extends the classical Hodgkin-Huxley formalism to include:

    - **Traub-Miles gating kinetics**: Simplified three-variable (:math:`m`, :math:`h`, :math:`n`)
      sodium and potassium channel dynamics from Traub and Miles (1991) [1]_.
    - **Beta-function synapses**: Double-exponential conductance profiles with separate
      rise and decay time constants for excitatory and inhibitory inputs.
    - **Gap-junction support**: Resistive coupling current that can be supplied externally
      to model electrical synapses between neurons.
    - **Refractory spike detection**: Physiological spike detection based on threshold
      crossing and local maximum detection with refractory period enforcement.

    This is a point neuron model (single compartment) suitable for large-scale network
    simulations where detailed morphology is not required but synaptic dynamics and
    gap-junction coupling are important.

    **2. Membrane Dynamics**

    The membrane potential evolves according to:

    .. math::

       C_m \frac{dV_m}{dt} = -(I_{Na} + I_K + I_L + I_{syn,ex} + I_{syn,in})
                              + I_{stim} + I_e + I_{gap}

    where the ionic and synaptic currents are:

    .. math::

       I_{Na}     &= g_{Na}\, m^3\, h\, (V_m - E_{Na})  \\
       I_K        &= g_K\,   n^4\,     (V_m - E_K)       \\
       I_L        &= g_L\,             (V_m - E_L)        \\
       I_{syn,ex} &= g_{ex}\,          (V_m - E_{ex})     \\
       I_{syn,in} &= g_{in}\,          (V_m - E_{in})

    **Physical interpretation:**

    - :math:`I_{Na}` -- Fast sodium current responsible for spike upstroke.
    - :math:`I_K` -- Delayed rectifier potassium current for repolarization.
    - :math:`I_L` -- Leak current maintaining resting potential.
    - :math:`I_{syn,ex}`, :math:`I_{syn,in}` -- Excitatory and inhibitory synaptic currents.
    - :math:`I_{gap}` -- Gap-junction current from electrically coupled neighbors.

    **3. Gating Variable Dynamics**

    Gating variables :math:`m`, :math:`h`, :math:`n` follow first-order kinetics:

    .. math::

       \frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)\,x

    with Traub-Miles rate functions using voltage-shifted dynamics :math:`V = V_m - V_T`:

    .. math::

       \alpha_n &= \frac{0.032\,(15 - V)}{e^{(15 - V)/5} - 1}, \quad
       \beta_n  = 0.5\,e^{(10 - V)/40}                                \\
       \alpha_m &= \frac{0.32\,(13 - V)}{e^{(13 - V)/4} - 1}, \quad
       \beta_m  = \frac{0.28\,(V - 40)}{e^{(V - 40)/5} - 1}          \\
       \alpha_h &= 0.128\,e^{(17 - V)/18}, \quad
       \beta_h  = \frac{4}{1 + e^{(40 - V)/5}}

    The voltage offset :math:`V_T` (default -50 mV) effectively shifts the spike threshold.

    **Computational note:** Singularities in :math:`\alpha` functions at specific voltages
    are handled via L'Hôpital's rule in the ODE solver.

    **4. Beta-Function Synaptic Conductances**

    Synaptic conductances follow double-exponential (beta-function) dynamics:

    .. math::

       \frac{d(\Delta g_{ex})}{dt} &= -\frac{\Delta g_{ex}}{\tau_{decay,ex}} \\
       \frac{dg_{ex}}{dt}          &= \Delta g_{ex} - \frac{g_{ex}}{\tau_{rise,ex}}

    and analogously for inhibitory conductance :math:`g_{in}`.

    **Spike input handling:**

    - Excitatory spikes (positive weights) increment :math:`\Delta g_{ex}`.
    - Inhibitory spikes (negative weights) increment :math:`\Delta g_{in}` (sign-flipped).
    - Each spike adds :math:`w \times \text{PSConInit}` to :math:`\Delta g`, where
      :math:`\text{PSConInit}` is the beta normalization factor ensuring peak conductance
      of 1 nS for unit weight.

    **Why beta functions?** Unlike simple exponential or alpha functions, beta functions
    provide independent control over rise and decay time scales, critical for accurately
    modeling AMPA (fast), NMDA (slow), and GABA receptors.

    **5. Gap-Junction Current**

    Gap junctions model electrical synapses as resistive couplings:

    .. math::

       I_{gap} = \sum_j g_{gap,ij}\,(V_j - V_i)

    In this single-neuron implementation, :math:`I_{gap}` must be computed externally
    (e.g., by a network simulation framework) and supplied via the ``x`` parameter to
    :meth:`update` or via ``add_current_input``.

    **6. Spike Detection**

    A spike is emitted when **all three conditions** are satisfied:

    1. ``refractory_step_count == 0`` (not in refractory period)
    2. :math:`V_m \geq V_T + 30` mV (threshold crossing)
    3. :math:`V_{old} > V_m` (local maximum detection)

    **No voltage reset** occurs after spike emission (unlike integrate-and-fire models);
    repolarization is driven naturally by the potassium current.

    **Refractory period:** During refractory steps, spike emission is suppressed but
    subthreshold dynamics continue to evolve. This prevents multiple spike detections
    during the falling phase of an action potential.

    **7. Numerical Integration**

    Uses an adaptive Runge-Kutta-Fehlberg (RKF45) integrator implemented in JAX.
    Default absolute tolerance (``gsl_error_tol=1e-6``) matches NEST's GSL RKF45
    integrator settings for numerical correspondence in benchmark comparisons.

    The ODE system has 8 state variables per neuron:
    :math:`[V_m, m, h, n, \Delta g_{ex}, g_{ex}, \Delta g_{in}, g_{in}]`.

    Parameters
    ----------
    in_size : Size
        Shape of the neuron population. Can be int (1D), tuple of ints (multidimensional),
        or None (scalar neuron). Determines state variable array dimensions.
    E_L : ArrayLike, default -60 mV
        Leak reversal potential (resting potential in absence of input).
    C_m : ArrayLike, default 200 pF
        Membrane capacitance. Must be strictly positive. Typical range: 50-500 pF.
    g_Na : ArrayLike, default 20000 nS
        Sodium channel peak conductance. Must be non-negative. Controls spike amplitude.
    g_K : ArrayLike, default 6000 nS
        Potassium channel peak conductance. Must be non-negative. Controls repolarization speed.
    g_L : ArrayLike, default 10 nS
        Leak conductance. Must be non-negative. Determines input resistance and time constant.
    E_Na : ArrayLike, default 50 mV
        Sodium reversal potential. Typically +40 to +60 mV.
    E_K : ArrayLike, default -90 mV
        Potassium reversal potential. Typically -80 to -100 mV.
    V_T : ArrayLike, default -50 mV
        Voltage offset for gating dynamics. Shifts the effective spike threshold.
    E_ex : ArrayLike, default 0 mV
        Excitatory synaptic reversal potential (typical for AMPA/NMDA receptors).
    E_in : ArrayLike, default -80 mV
        Inhibitory synaptic reversal potential (typical for GABA receptors).
    t_ref : ArrayLike, default 2 ms
        Refractory period duration. Must be non-negative. Increase if multiple spikes
        are detected per action potential.
    tau_rise_ex : ArrayLike, default 0.5 ms
        Excitatory synaptic rise time constant. Must be strictly positive.
    tau_decay_ex : ArrayLike, default 5.0 ms
        Excitatory synaptic decay time constant. Must be strictly positive.
        Should be larger than ``tau_rise_ex`` for physiological beta-function shape.
    tau_rise_in : ArrayLike, default 0.5 ms
        Inhibitory synaptic rise time constant. Must be strictly positive.
    tau_decay_in : ArrayLike, default 10.0 ms
        Inhibitory synaptic decay time constant. Must be strictly positive.
    I_e : ArrayLike, default 0 pA
        Constant external input current (bias current). Can be positive (depolarizing)
        or negative (hyperpolarizing).
    gsl_error_tol : ArrayLike, default 1e-6
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
    V_m_init : ArrayLike, optional
        Initial membrane potential. If None, defaults to ``E_L``.
    Act_m_init : ArrayLike, optional
        Initial sodium activation variable. If None, computed from equilibrium at ``V_m_init``.
    Inact_h_init : ArrayLike, optional
        Initial sodium inactivation variable. If None, computed from equilibrium at ``V_m_init``.
    Act_n_init : ArrayLike, optional
        Initial potassium activation variable. If None, computed from equilibrium at ``V_m_init``.
    spk_fun : Callable, default braintools.surrogate.ReluGrad()
        Surrogate gradient function for differentiable spike generation during backpropagation.
        Only affects gradient computation; forward-pass spike detection is always threshold-based.
    spk_reset : str, default 'hard'
        Spike reset mode for gradient computation. ``'hard'`` uses stop_gradient;
        ``'soft'`` allows gradients through spike. Does not affect forward dynamics.
    name : str, optional
        Name identifier for the neuron population.

    Parameter Mapping
    -----------------

    This table maps brainpy.state parameter names to NEST ``hh_cond_beta_gap_traub``
    parameter names and mathematical symbols:

    ==================== ===================== =============================== ==============================================
    **brainpy.state**    **NEST**              **Math**                        **Description**
    ==================== ===================== =============================== ==============================================
    ``in_size``          (population size)                                     Number/shape of neurons
    ``E_L``              ``E_L``               :math:`E_L`                     Leak reversal potential (mV)
    ``C_m``              ``C_m``               :math:`C_m`                     Membrane capacitance (pF)
    ``g_Na``             ``g_Na``              :math:`g_{Na}`                  Sodium conductance (nS)
    ``g_K``              ``g_K``               :math:`g_K`                     Potassium conductance (nS)
    ``g_L``              ``g_L``               :math:`g_L`                     Leak conductance (nS)
    ``E_Na``             ``E_Na``              :math:`E_{Na}`                  Sodium reversal (mV)
    ``E_K``              ``E_K``               :math:`E_K`                     Potassium reversal (mV)
    ``V_T``              ``V_T``               :math:`V_T`                     Voltage offset (mV)
    ``E_ex``             ``E_ex``              :math:`E_{ex}`                  Excitatory reversal (mV)
    ``E_in``             ``E_in``              :math:`E_{in}`                  Inhibitory reversal (mV)
    ``t_ref``            ``t_ref``             :math:`t_{ref}`                 Refractory period (ms)
    ``tau_rise_ex``      ``tau_rise_ex``       :math:`\tau_{rise,ex}`          Excitatory rise time (ms)
    ``tau_decay_ex``     ``tau_decay_ex``      :math:`\tau_{decay,ex}`         Excitatory decay time (ms)
    ``tau_rise_in``      ``tau_rise_in``       :math:`\tau_{rise,in}`          Inhibitory rise time (ms)
    ``tau_decay_in``     ``tau_decay_in``      :math:`\tau_{decay,in}`         Inhibitory decay time (ms)
    ``I_e``              ``I_e``               :math:`I_e`                     External current (pA)
    ``gsl_error_tol``    --                    --                              RKF45 local error tolerance
    ``V_m_init``         (initial ``V_m``)     :math:`V_m(t=0)`                Initial membrane potential (mV)
    ``Act_m_init``       (initial ``Act_m``)   :math:`m(t=0)`                  Initial Na activation (0-1)
    ``Inact_h_init``     (initial ``Inact_h``) :math:`h(t=0)`                  Initial Na inactivation (0-1)
    ``Act_n_init``       (initial ``Act_n``)   :math:`n(t=0)`                  Initial K activation (0-1)
    ==================== ===================== =============================== ==============================================

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential in mV. Shape: ``(*in_size,)``.
    m : brainstate.HiddenState
        Sodium activation gating variable (unitless, 0-1 range).
    h : brainstate.HiddenState
        Sodium inactivation gating variable (unitless, 0-1 range).
    n : brainstate.HiddenState
        Potassium activation gating variable (unitless, 0-1 range).
    dg_ex : brainstate.ShortTermState
        Time derivative of excitatory conductance in nS/ms (beta-function intermediate state).
    g_ex : brainstate.HiddenState
        Excitatory synaptic conductance in nS.
    dg_in : brainstate.ShortTermState
        Time derivative of inhibitory conductance in nS/ms (beta-function intermediate state).
    g_in : brainstate.HiddenState
        Inhibitory synaptic conductance in nS.
    I_stim : brainstate.ShortTermState
        Buffered stimulation current in pA (applied in next time step).
    refractory_step_count : brainstate.ShortTermState
        Integer countdown of remaining refractory steps. Zero means neuron can spike.
    integration_step : brainstate.ShortTermState
        Persistent RKF45 substep size estimate (ms).
    last_spike_time : brainstate.ShortTermState
        Time of most recent spike emission in ms (for recording/analysis).

    Raises
    ------
    ValueError
        If ``C_m <= 0``, ``t_ref < 0``, any time constant ``<= 0``, or any conductance ``< 0``.

    Notes
    -----
    **Usage Considerations:**

    1. **Synaptic weight units**: Spike weights are interpreted in conductance units (nS).
       A weight of 1.0 produces a peak conductance of 1 nS at the synapse's rise time.

    2. **Excitatory vs. inhibitory synapses**: The sign of the synaptic weight determines
       the receptor type:

       - Positive weights drive ``g_ex`` (excitatory, reversal at ``E_ex``).
       - Negative weights drive ``g_in`` (inhibitory, reversal at ``E_in``).

       The sign is automatically handled by :meth:`_sum_signed_delta_inputs`.

    3. **Gap-junction current**: Must be computed externally and provided via the ``x``
       parameter to :meth:`update` or registered with ``add_current_input``. In a network,
       compute as :math:`\sum_j g_{gap,ij}(V_j - V_i)` where :math:`V_j` are neighbor
       potentials and :math:`g_{gap,ij}` are coupling conductances.

    4. **No voltage reset**: Unlike integrate-and-fire models, the membrane potential
       is not reset after spike emission. The potassium current naturally drives
       repolarization and hyperpolarization.

    5. **Refractory period tuning**: If the model emits multiple spikes per action
       potential, increase ``t_ref``. Traub and Miles (1991) used 3 ms; NEST defaults
       to 2 ms.

    6. **Numerical stability**: The adaptive RKF45 integrator handles the stiff HH
       dynamics robustly. If you encounter instability, try reducing ``gsl_error_tol``
       or increasing the simulation time step ``dt``.

    7. **Performance**: All neurons are integrated in a single vectorized adaptive
       RKF45 loop via JAX, providing efficient GPU/TPU execution.

    Examples
    --------
    **Basic single-neuron simulation with step current:**

    .. code-block:: python

       >>> from brainpy import state as bst
       >>> import brainunit as u
       >>> import brainstate
       >>> import matplotlib.pyplot as plt
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neuron = bst.hh_cond_beta_gap_traub(1)
       ...     neuron.init_all_states()
       ...     # Apply 500 pA step current for 100 ms
       ...     times, voltages = [], []
       ...     for t in range(1000):  # 100 ms simulation
       ...         if 200 <= t < 700:  # 20-70 ms
       ...             neuron.update(500 * u.pA)
       ...         else:
       ...             neuron.update(0 * u.pA)
       ...         times.append(brainstate.environ.get('t'))
       ...         voltages.append(neuron.V.value.item())
       >>> plt.plot(times, voltages)
       >>> plt.xlabel('Time (ms)')
       >>> plt.ylabel('Membrane potential (mV)')
       >>> plt.title('HH neuron with step current input')
       >>> plt.show()

    **Network simulation with gap junctions:**

    .. code-block:: python

       >>> # Two coupled neurons with gap junction
       >>> neuron_pop = bst.hh_cond_beta_gap_traub(2, I_e=200 * u.pA)
       >>> neuron_pop.init_all_states()
       >>> g_gap = 50.0  # nS gap conductance
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     for _ in range(1000):
       ...         V = neuron_pop.V.value
       ...         # Compute gap currents: I_gap[i] = g_gap * (V[j] - V[i])
       ...         I_gap = u.math.zeros_like(V)
       ...         I_gap = I_gap.at[0].set(g_gap * u.nS * (V[1] - V[0]))
       ...         I_gap = I_gap.at[1].set(g_gap * u.nS * (V[0] - V[1]))
       ...         neuron_pop.update(I_gap)

    **Beta-function synapse with different time constants:**

    .. code-block:: python

       >>> # Slow NMDA-like synapse (tau_rise=2ms, tau_decay=50ms)
       >>> neuron = bst.hh_cond_beta_gap_traub(
       ...     1,
       ...     tau_rise_ex=2.0 * u.ms,
       ...     tau_decay_ex=50.0 * u.ms,
       ... )
       >>> neuron.init_all_states()
       >>> # Add excitatory spike input at t=10ms
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     for t in range(1000):
       ...         if t == 100:  # t=10ms
       ...             neuron.delta_inputs['spike'] = lambda: 5.0 * u.nS
       ...         neuron.update()

    See Also
    --------
    hh_cond_exp_traub : Hodgkin-Huxley Traub model with single-exponential synapses.
    hh_psc_alpha_gap : Hodgkin-Huxley model with gap junctions and alpha-function PSCs.
    hh_psc_alpha : Classic HH model with current-based alpha-function synapses.

    References
    ----------
    .. [1] Traub RD and Miles R (1991). Neuronal Networks of the Hippocampus.
           Cambridge University Press, Cambridge UK.
    .. [2] Brette R et al. (2007). Simulation of networks of spiking neurons:
           A review of tools and strategies. Journal of Computational
           Neuroscience 23:349-398.
           DOI: https://doi.org/10.1007/s10827-007-0038-6
    .. [3] Hahne J, Helias M, Kunkel S, Igarashi J, Bolten M, Frommer A,
           and Diesmann M (2015). A unified framework for spiking and
           gap-junction interactions in distributed neuronal network
           simulations. Frontiers in Neuroinformatics 9:22.
           DOI: https://doi.org/10.3389/fninf.2015.00022
    .. [4] Rotter S and Diesmann M (1999). Exact digital simulation of
           time-invariant linear systems with applications to neuronal
           modeling. Biological Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    .. [5] Roth A and van Rossum M (2010). Chapter 6: Modeling synapses.
           In: De Schutter E (ed), Computational Modeling Methods for
           Neuroscientists, MIT Press, pp 139-160.
    """

    __module__ = 'brainpy.state'

    # Gap-junction emission (seam H, goal 15b). The membrane voltage ``V`` is the
    # graded quantity a gap junction couples on: the Simulator allocates an emission
    # holder for any model declaring ``_emission_attr`` and captures it each step, so
    # a gap connection reads the (one-step-lagged) ``V_pre`` and deposits the
    # difference current ``g * (V_pre - V_post)`` via ``sum_current_inputs``. This is
    # NOT a continuous *rate* emitter (``_emission_continuous`` stays False): a plain
    # spike connection from this neuron still delivers the binary spike.
    _emission_attr = 'V'

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000
    _EPS = np.finfo(np.float64).eps

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -60. * u.mV,
        C_m: ArrayLike = 200. * u.pF,
        g_Na: ArrayLike = 20000. * u.nS,
        g_K: ArrayLike = 6000. * u.nS,
        g_L: ArrayLike = 10. * u.nS,
        E_Na: ArrayLike = 50. * u.mV,
        E_K: ArrayLike = -90. * u.mV,
        V_T: ArrayLike = -50. * u.mV,
        E_ex: ArrayLike = 0. * u.mV,
        E_in: ArrayLike = -80. * u.mV,
        t_ref: ArrayLike = 2. * u.ms,
        tau_rise_ex: ArrayLike = 0.5 * u.ms,
        tau_decay_ex: ArrayLike = 5. * u.ms,
        tau_rise_in: ArrayLike = 0.5 * u.ms,
        tau_decay_in: ArrayLike = 10. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        gsl_error_tol: ArrayLike = 1e-6,
        V_m_init: ArrayLike = None,
        Act_m_init: ArrayLike = None,
        Inact_h_init: ArrayLike = None,
        Act_n_init: ArrayLike = None,
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
        self.V_T = braintools.init.param(V_T, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.tau_rise_ex = braintools.init.param(tau_rise_ex, self.varshape)
        self.tau_decay_ex = braintools.init.param(tau_decay_ex, self.varshape)
        self.tau_rise_in = braintools.init.param(tau_rise_in, self.varshape)
        self.tau_decay_in = braintools.init.param(tau_decay_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.gsl_error_tol = gsl_error_tol
        self.V_m_init = V_m_init
        self.Act_m_init = Act_m_init
        self.Inact_h_init = Inact_h_init
        self.Act_n_init = Act_n_init

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

    @staticmethod
    def _hh_equilibrium(V):
        r"""Compute Traub HH gating variable equilibrium values at voltage V (mV).

        This matches NEST's ``State_::State_(const Parameters_&)`` initialization,
        which applies the Traub rate equations **without** the V_T offset.  The
        dynamics function uses ``V - V_T`` in its rate equations, but the
        equilibrium initialization in NEST uses the raw voltage ``y_[0]`` (= E_L).

        Parameters
        ----------
        V : float
            Membrane potential in mV (without V_T offset).

        Returns
        -------
        m_inf : float
            Sodium activation gating variable equilibrium value (unitless, 0-1 range).
        h_inf : float
            Sodium inactivation gating variable equilibrium value (unitless, 0-1 range).
        n_inf : float
            Potassium activation gating variable equilibrium value (unitless, 0-1 range).

        Notes
        -----
        The equilibrium values are computed from the rate equations:

        .. math::

            x_{\infty} = \frac{\alpha_x}{\alpha_x + \beta_x}

        where the rate functions match the Traub-Miles formulation with zero V_T shift.
        This differs from the dynamics integration, which applies the voltage shift
        ``V - V_T`` during time evolution.
        """
        alpha_n = 0.032 * (15.0 - V) / (np.exp((15.0 - V) / 5.0) - 1.0)
        beta_n = 0.5 * np.exp((10.0 - V) / 40.0)
        alpha_m = 0.32 * (13.0 - V) / (np.exp((13.0 - V) / 4.0) - 1.0)
        beta_m = 0.28 * (V - 40.0) / (np.exp((V - 40.0) / 5.0) - 1.0)
        alpha_h = 0.128 * np.exp((17.0 - V) / 18.0)
        beta_h = 4.0 / (1.0 + np.exp((40.0 - V) / 5.0))
        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)
        n_inf = alpha_n / (alpha_n + beta_n)
        return m_inf, h_inf, n_inf

    @classmethod
    def _beta_normalization_factor_scalar(cls, tau_rise: float, tau_decay: float):
        r"""Compute the normalization factor for a beta-function synapse.

        This is a Python translation of NEST's ``beta_normalization_factor()``
        from ``libnestutil/beta_normalization_factor.h``.

        The beta function synapse ODE solution is:

        .. math::

           g(t) = \frac{c}{a - b} \left( e^{-bt} - e^{-at} \right)

        where :math:`a = 1/\tau_{rise}` and :math:`b = 1/\tau_{decay}`.
        This function computes the constant :math:`c` such that the peak
        conductance equals 1 nS for unit-weight spike input.

        Parameters
        ----------
        tau_rise : float
            Synaptic rise time constant in milliseconds. Must be positive.
        tau_decay : float
            Synaptic decay time constant in milliseconds. Must be positive.

        Returns
        -------
        float
            Normalization factor (unitless) that scales the synaptic conductance jump
            to ensure peak conductance equals 1 nS for a unit-weight input spike.

        Notes
        -----
        **Mathematical Derivation:**

        1. The beta-function conductance is the solution to the second-order system:

           .. math::

               \frac{d(\Delta g)}{dt} &= -\frac{\Delta g}{\tau_{decay}} \\\\
               \frac{dg}{dt} &= \Delta g - \frac{g}{\tau_{rise}}

        2. For an impulse input at t=0, the analytical solution is:

           .. math::

               g(t) = c \cdot \frac{e^{-t/\tau_{decay}} - e^{-t/\tau_{rise}}}{1/\tau_{rise} - 1/\tau_{decay}}

        3. The peak occurs at time:

           .. math::

               t_{peak} = \frac{\tau_{rise} \tau_{decay}}{\tau_{decay} - \tau_{rise}} \ln\left(\frac{\tau_{decay}}{\tau_{rise}}\right)

        4. The normalization factor ensures :math:`g(t_{peak}) = 1` nS.

        **Special Cases:**

        - When :math:`\tau_{rise} \approx \tau_{decay}`, the beta function degenerates to
          an alpha function with normalization factor :math:`e / \tau_{decay}`.
        - If either time constant is zero or negative (invalid), the function returns 0.

        **Numerical Stability:**

        Uses ``numpy.finfo(np.float64).eps`` to detect near-equality of time constants,
        preventing division by zero or overflow in the log/exponential calculations.

        References
        ----------
        .. [1] Rotter S, Diesmann M (1999). Exact digital simulation of
               time-invariant linear systems with applications to neuronal
               modeling. Biological Cybernetics 81:381.
               DOI: https://doi.org/10.1007/s004220050570
        .. [2] Roth A, van Rossum M (2010). Chapter 6: Modeling synapses.
               in De Schutter, Computational Modeling Methods for
               Neuroscientists, MIT Press.
        """
        tau_difference = tau_decay - tau_rise
        peak_value = 0.0

        if abs(tau_difference) > cls._EPS:
            t_peak = tau_decay * tau_rise * np.log(tau_decay / tau_rise) / tau_difference
            peak_value = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)

        if abs(peak_value) < cls._EPS:
            # rise time ~ decay time -> alpha function fallback
            return np.e / tau_decay
        else:
            return (1.0 / tau_rise - 1.0 / tau_decay) / peak_value

    def _validate_parameters(self):
        r"""Validate parameter constraints at initialization.

        Raises
        ------
        ValueError
            If any parameter violates physical constraints:
            - ``C_m <= 0`` (capacitance must be positive)
            - ``t_ref < 0`` (refractory time must be non-negative)
            - Any time constant ``<= 0`` (must be strictly positive)
            - Any conductance ``< 0`` (must be non-negative)
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.t_ref, self.g_Na)):
            return
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if (
            cond_any(self.tau_rise_ex <= 0.0 * u.ms)
            or cond_any(self.tau_decay_ex <= 0.0 * u.ms)
            or cond_any(self.tau_rise_in <= 0.0 * u.ms)
            or cond_any(self.tau_decay_in <= 0.0 * u.ms)
        ):
            raise ValueError('All time constants must be strictly positive.')
        if (
            cond_any(self.g_Na < 0.0 * u.nS)
            or cond_any(self.g_K < 0.0 * u.nS)
            or cond_any(self.g_L < 0.0 * u.nS)
        ):
            raise ValueError('All conductances must be non-negative.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables to equilibrium or user-specified values.

        Sets up hidden states (membrane potential, gating variables, synaptic conductances)
        and short-term states (refractory counter, spike time buffer). By default, initializes
        to physiologically realistic equilibrium values matching NEST's initialization protocol.

        **Initialization Protocol:**

        1. **Membrane potential**: Defaults to ``E_L`` (resting potential) if ``V_m_init`` is None.
        2. **Gating variables**: If ``Act_m_init``, ``Inact_h_init``, or ``Act_n_init`` are None,
           compute equilibrium values :math:`x_{\infty} = \alpha_x / (\alpha_x + \beta_x)` at
           the initial membrane potential **without** V_T offset (matching NEST).
        3. **Synaptic conductances**: Initialize ``dg_ex``, ``g_ex``, ``dg_in``, ``g_in`` to zero.
        4. **Refractory state**: Set ``refractory_step_count`` to 0 (not refractory).
        5. **Spike time**: Set ``last_spike_time`` to -1e7 ms (no recent spike).

        Parameters
        ----------
        **kwargs : dict, optional
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        **Equilibrium Computation:**

        The equilibrium gating variables are computed using the Traub-Miles rate functions
        evaluated at the **raw** initial voltage (without V_T offset):

        .. math::

            m_{\infty} &= \\frac{\\alpha_m(V_0)}{\\alpha_m(V_0) + \\beta_m(V_0)} \\\\
            h_{\infty} &= \\frac{\\alpha_h(V_0)}{\\alpha_h(V_0) + \\beta_h(V_0)} \\\\
            n_{\infty} &= \\frac{\\alpha_n(V_0)}{\\alpha_n(V_0) + \\beta_n(V_0)}

        where :math:`V_0 =` ``V_m_init`` (or ``E_L`` if None). This matches NEST's
        ``State_::State_(const Parameters_&)`` constructor, which uses ``y_[0]`` (= ``E_L``)
        without applying the V_T shift used during dynamics integration.

        **Why No V_T Offset?**

        The V_T offset is applied during **dynamics** integration (in the ODE right-hand side)
        to shift the effective spike threshold. However, equilibrium initialization uses the
        **absolute** membrane potential to ensure consistency with the model's resting state
        before any dynamics occur.

        **Custom Initialization:**

        To initialize with specific gating variable values (e.g., after depolarization):

        .. code-block:: python

           >>> neuron = bst.hh_cond_beta_gap_traub(
           ...     10,
           ...     V_m_init=-50 * u.mV,      # Depolarized initial state
           ...     Act_m_init=0.3,            # Custom Na activation
           ...     Inact_h_init=0.4,          # Custom Na inactivation
           ...     Act_n_init=0.2,            # Custom K activation
           ... )
           >>> neuron.init_all_states()

        **State Variable Summary:**

        After calling ``init_state``, the following attributes are available:

        - ``V`` (HiddenState): Membrane potential (mV)
        - ``m`` (HiddenState): Sodium activation (0-1)
        - ``h`` (HiddenState): Sodium inactivation (0-1)
        - ``n`` (HiddenState): Potassium activation (0-1)
        - ``dg_ex`` (ShortTermState): Excitatory conductance derivative (nS/ms)
        - ``g_ex`` (HiddenState): Excitatory conductance (nS)
        - ``dg_in`` (ShortTermState): Inhibitory conductance derivative (nS/ms)
        - ``g_in`` (HiddenState): Inhibitory conductance (nS)
        - ``I_stim`` (ShortTermState): Stimulation current buffer (pA)
        - ``refractory_step_count`` (ShortTermState): Refractory countdown (int)
        - ``integration_step`` (ShortTermState): RKF45 substep size (ms)
        - ``last_spike_time`` (ShortTermState): Last spike time (ms)

        Examples
        --------
        **Default equilibrium initialization:**

        .. code-block:: python

           >>> from brainpy import state as bst
           >>> neuron = bst.hh_cond_beta_gap_traub(5)
           >>> neuron.init_all_states()
           >>> print(neuron.V.value)  # Should be E_L = -60 mV
           >>> print(neuron.m.value)  # Equilibrium at -60 mV

        **Custom depolarized initial state:**

        .. code-block:: python

           >>> neuron = bst.hh_cond_beta_gap_traub(
           ...     1,
           ...     V_m_init=-45 * u.mV,  # Near threshold
           ... )
           >>> neuron.init_all_states()
           >>> print(neuron.V.value)  # -45 mV
           >>> print(neuron.m.value)  # Equilibrium at -45 mV (higher than at -60 mV)
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        # Default V_m_init to E_L (matching NEST: y_[0] = p.E_L)
        if self.V_m_init is not None:
            V_init_val = self.V_m_init
        else:
            V_init_val = self.E_L

        V = braintools.init.param(braintools.init.Constant(V_init_val), self.varshape)
        zeros = u.math.zeros(self.varshape, dtype=u.math.asarray(V / u.mV).dtype) * (u.nS / u.ms)

        # Gating variables equilibrate at EACH neuron's own initial voltage (NEST uses
        # the raw per-node V_m, not V_m - V_T). ``_hh_equilibrium`` is elementwise, so a
        # per-neuron voltage array yields per-neuron gating; collapsing to ``.flat[0]``
        # would start a heterogeneously-initialized population (e.g. the gap demo's
        # perturbed cell) far out of equilibrium and fire spurious transients.
        V_arr = np.asarray(u.get_mantissa(V / u.mV), dtype=dftype)
        m_eq, h_eq, n_eq = self._hh_equilibrium(V_arr)

        # An explicit ``*_init`` override (scalar or per-neuron array) wins; otherwise
        # the per-neuron equilibrium. ``braintools.init.param`` broadcasts either.
        m_init = (m_eq if self.Act_m_init is None else
                  np.asarray(u.math.asarray(self.Act_m_init / u.UNITLESS), dtype=dftype))
        h_init = (h_eq if self.Inact_h_init is None else
                  np.asarray(u.math.asarray(self.Inact_h_init / u.UNITLESS), dtype=dftype))
        n_init = (n_eq if self.Act_n_init is None else
                  np.asarray(u.math.asarray(self.Act_n_init / u.UNITLESS), dtype=dftype))

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
        # Beta-function synapse state: derivative (dg) and conductance (g)
        # All initialized to zero (matching NEST: y_[i] = 0 for i > 0)
        self.dg_ex = brainstate.ShortTermState(zeros)
        self.g_ex = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=dftype) * u.nS)
        self.dg_in = brainstate.ShortTermState(zeros)
        self.g_in = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=dftype) * u.nS)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))
        self.V_old = brainstate.ShortTermState(V)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output from membrane potential.

        Applies the surrogate gradient function (``spk_fun``) to the membrane potential
        to generate a differentiable spike signal for gradient-based learning. This is
        used internally by :meth:`update` to compute the return value.

        **Forward Pass vs. Backward Pass:**

        - **Forward pass**: Returns a binary-like spike indicator (1.0 where spike occurred,
          0.0 otherwise) based on the three-condition spike detection in :meth:`update`.
        - **Backward pass**: Gradients flow through the surrogate function (e.g., ``ReluGrad``),
          which provides a smooth approximation of the Heaviside step function.

        **Why Surrogate Gradients?**

        The true spike detection logic (threshold + local maximum + refractory) is
        non-differentiable. Surrogate gradient methods replace the zero-everywhere gradient
        of the Heaviside function with a smooth proxy (e.g., ReLU, sigmoid, exponential)
        during backpropagation, enabling gradient-based optimization of spiking networks.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential in millivolts. If None, uses ``self.V.value`` (current state).
            Shape must match ``(*in_size,)``.

        Returns
        -------
        spike : ArrayLike
            Differentiable spike output with same shape as ``V``. Forward values are
            approximately binary (close to 0 or 1); backward gradients are provided by
            the surrogate function.

        Notes
        -----
        **Voltage Scaling:**

        The membrane potential is divided by 1 mV to convert from physical units to a
        unitless scale before passing to ``spk_fun``. This ensures the surrogate function
        operates on dimensionless voltage values (typically in the range -80 to +50 for
        biological neurons).

        **Surrogate Function Choice:**

        The default ``braintools.surrogate.ReluGrad()`` uses a rectified linear gradient:

        .. math::

            \\text{forward}(V) &= H(V) \quad \\text{(Heaviside step function)} \\\\
            \\frac{d}{dV}\\text{backward}(V) &= \\begin{cases}
                1 & \\text{if } V > 0 \\\\
                0 & \\text{otherwise}
            \\end{cases}

        Other options include:

        - ``Sigmoid()``: Smooth logistic gradient.
        - ``Gaussian()``: Gaussian-shaped gradient.
        - ``PiecewiseQuadratic()``: Quadratic spline gradient.

        See ``braintools.surrogate`` for available functions.

        **Spike Reset Mode:**

        The ``spk_reset`` parameter (``'hard'`` or ``'soft'``) controls whether gradients
        flow through the spike in :meth:`update`:

        - ``'hard'``: Uses ``jax.lax.stop_gradient`` to prevent gradients from propagating
          through the spike event. Gradient flow stops at the spike.
        - ``'soft'``: Allows gradients to flow through the spike (no stop_gradient).
          This can help learning but may be less biologically plausible.

        This method does not directly apply ``spk_reset``; it is handled in :meth:`update`.

        Examples
        --------
        **Direct spike computation from voltage:**

        .. code-block:: python

           >>> from brainpy import state as bst
           >>> import brainunit as u
           >>> neuron = bst.hh_cond_beta_gap_traub(1)
           >>> neuron.init_all_states()
           >>> # Manually set voltage above threshold
           >>> V_test = (-50 + 30 + 1) * u.mV  # V_T + 30 + 1 = -19 mV
           >>> spike = neuron.get_spike(V_test)
           >>> print(f"Spike value: {spike.item():.3f}")

        **Using custom surrogate function:**

        .. code-block:: python

           >>> import braintools
           >>> neuron = bst.hh_cond_beta_gap_traub(
           ...     1,
           ...     spk_fun=braintools.surrogate.Sigmoid(alpha=5.0),
           ... )
           >>> neuron.init_all_states()
           >>> spike = neuron.get_spike(neuron.V.value)

        See Also
        --------
        update : Main update method that uses this function to compute spike output.
        braintools.surrogate : Module containing surrogate gradient functions.
        """
        V = self.V.value if V is None else V
        # For HH neurons with Traub threshold: spike at V_T + 30.
        # Scale relative to 0 mV for the surrogate function.
        v_scaled = V / (1. * u.mV)
        return self.spk_fun(v_scaled)

    def _sum_signed_delta_inputs(self):
        r"""Split delta inputs into excitatory (positive) and inhibitory (negative) conductances.

        Processes all registered delta inputs (spike-triggered conductance jumps) and
        separates them by sign: positive weights drive excitatory conductance, negative
        weights drive inhibitory conductance.

        Returns
        -------
        g_ex : ArrayLike
            Total excitatory conductance jump in nS (sum of all positive delta inputs).
        g_in : ArrayLike
            Total inhibitory conductance jump in nS (sum of absolute values of negative inputs).

        Notes
        -----
        **Delta Input Semantics:**

        Delta inputs are registered via ``add_delta_input(key, func)`` where ``func()``
        returns a conductance value in nS. This method:

        1. Calls each registered delta input function.
        2. Separates positive (excitatory) and negative (inhibitory) contributions.
        3. Sums them into ``g_ex`` and ``g_in`` respectively.
        4. Removes callable entries after invocation (one-time spike inputs).

        **Sign Convention:**

        - Positive weight :math:`w > 0`: Excitatory synapse, adds :math:`w` to :math:`g_{ex}`.
        - Negative weight :math:`w < 0`: Inhibitory synapse, adds :math:`|w|` to :math:`g_{in}`.

        The reversal potentials (``E_ex``, ``E_in``) determine the synaptic current direction:

        .. math::

            I_{syn,ex} &= g_{ex} (V_m - E_{ex}) \\\\
            I_{syn,in} &= g_{in} (V_m - E_{in})

        **Example Usage:**

        .. code-block:: python

           >>> neuron.add_delta_input('synapse1', lambda: 5.0 * u.nS)   # Excitatory
           >>> neuron.add_delta_input('synapse2', lambda: -3.0 * u.nS)  # Inhibitory
           >>> g_ex, g_in = neuron._sum_signed_delta_inputs()
           >>> # g_ex = 5.0 nS, g_in = 3.0 nS
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
            # Inhibitory: negative weight -> positive conductance (sign flipped)
            g_in = g_in + u.math.maximum(-out, zero)
        return g_ex, g_in

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, dg_ex, g_ex, dg_in, g_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, V_old, v_spike_detect --
            mutable auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        V_m = state.V
        m_ = state.m
        h_ = state.h
        n_ = state.n

        # Ionic currents
        I_Na = self.g_Na * m_ * m_ * m_ * h_ * (V_m - self.E_Na)
        I_K = self.g_K * n_ * n_ * n_ * n_ * (V_m - self.E_K)
        I_L = self.g_L * (V_m - self.E_L)

        # Synaptic currents (conductance-based)
        I_syn_exc = state.g_ex * (V_m - self.E_ex)
        I_syn_inh = state.g_in * (V_m - self.E_in)

        # Shifted voltage for gating variable rate equations
        V_shifted = (V_m - self.V_T) / u.mV  # unitless

        # Traub-Miles rate functions (with safe clipping to avoid overflow)
        arg_n = u.math.clip((15.0 - V_shifted) / 5.0, -500.0, 500.0)
        alpha_n = 0.032 * (15.0 - V_shifted) / (u.math.exp(arg_n) - 1.0) / u.ms

        alpha_n = u.math.where(
            u.math.abs(15.0 - V_shifted) < 1e-10,
            0.032 * 5.0 / u.ms,  # L'Hopital limit
            alpha_n
        )
        beta_n = 0.5 * u.math.exp(u.math.clip((10.0 - V_shifted) / 40.0, -500.0, 500.0)) / u.ms

        arg_m = u.math.clip((13.0 - V_shifted) / 4.0, -500.0, 500.0)
        alpha_m = 0.32 * (13.0 - V_shifted) / (u.math.exp(arg_m) - 1.0) / u.ms

        alpha_m = u.math.where(
            u.math.abs(13.0 - V_shifted) < 1e-10,
            0.32 * 4.0 / u.ms,  # L'Hopital limit
            alpha_m
        )

        arg_bm = u.math.clip((V_shifted - 40.0) / 5.0, -500.0, 500.0)
        beta_m = 0.28 * (V_shifted - 40.0) / (u.math.exp(arg_bm) - 1.0) / u.ms

        beta_m = u.math.where(
            u.math.abs(V_shifted - 40.0) < 1e-10,
            0.28 * 5.0 / u.ms,  # L'Hopital limit
            beta_m
        )

        alpha_h = 0.128 * u.math.exp(u.math.clip((17.0 - V_shifted) / 18.0, -500.0, 500.0)) / u.ms
        beta_h = 4.0 / (1.0 + u.math.exp(u.math.clip((40.0 - V_shifted) / 5.0, -500.0, 500.0))) / u.ms

        # Membrane potential derivative
        dV = (-I_Na - I_K - I_L - I_syn_exc - I_syn_inh + extra.i_stim + self.I_e) / self.C_m

        # Gating variable derivatives
        dm = alpha_m * (1.0 - m_) - beta_m * m_
        dh = alpha_h * (1.0 - h_) - beta_h * h_
        dn = alpha_n * (1.0 - n_) - beta_n * n_

        # Beta-function synapse derivatives
        ddg_ex = -state.dg_ex / self.tau_decay_ex
        dg_ex_dt = state.dg_ex - state.g_ex / self.tau_rise_ex
        ddg_in = -state.dg_in / self.tau_decay_in
        dg_in_dt = state.dg_in - state.g_in / self.tau_rise_in

        return DotDict(
            V=dV, m=dm, h=dh, n=dn,
            dg_ex=ddg_ex, g_ex=dg_ex_dt, dg_in=ddg_in, g_in=dg_in_dt
        )

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection and refractory handling.

        For the HH model, spike detection uses threshold crossing + local maximum
        detection. No voltage reset occurs after spike (repolarization is natural).

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, dg_ex, g_ex, dg_in, g_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, V_old, v_spike_detect.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & (
                (state.V < -1e3 * u.mV) |
                (state.V > 1e3 * u.mV)
            )
        )

        # Spike detection: threshold crossing + local maximum (V_old > V)
        # Only for non-refractory neurons where the substep was accepted.
        crossed_threshold = state.V >= extra.v_spike_detect
        local_max = extra.V_old > state.V
        spike_now = accept & (extra.r <= 0) & crossed_threshold & local_max
        spike_mask = extra.spike_mask | spike_now

        # Set refractory counter on spike (no voltage reset for HH).
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count, extra.r)

        # Track V_old for local maximum detection in next substep.
        new_V_old = u.math.where(accept, state.V, extra.V_old)

        new_state = DotDict({**state})
        new_extra = DotDict(
            {**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable, 'V_old': new_V_old}
        )
        return new_state, new_extra

    def update(self, x=0. * u.pA):
        r"""Update neuron state for one simulation time step.

        Advances all state variables by one time step ``dt`` following the NEST
        ``hh_cond_beta_gap_traub`` update protocol. Integrates the 8D ODE system
        using adaptive RKF45, applies synaptic conductance jumps, detects spikes,
        and updates refractory state.

        **Update Protocol (Matching NEST Order):**

        1. **Record pre-integration voltage**: Store :math:`V_{old} = V_m(t)` for
           spike detection (local maximum criterion).

        2. **ODE integration**: Integrate the 8-variable system
           :math:`[V_m, m, h, n, \Delta g_{ex}, g_{ex}, \Delta g_{in}, g_{in}]`
           from :math:`t` to :math:`t + dt` using adaptive RKF45.

        3. **Apply synaptic inputs**: Add arriving spike-triggered conductance jumps:

           .. math::

               \Delta g_{ex} &\leftarrow \Delta g_{ex} + w_{ex} \times \text{PSConInit}_{ex} \\
               \Delta g_{in} &\leftarrow \Delta g_{in} + w_{in} \times \text{PSConInit}_{in}

           where :math:`\text{PSConInit}` is the beta normalization factor ensuring
           peak conductance of 1 nS for unit weight.

        4. **Spike detection**: Emit spike if **all conditions** are met:

           - ``refractory_step_count == 0`` (not refractory)
           - :math:`V_m(t+dt) \geq V_T + 30` mV (threshold crossed)
           - :math:`V_{old} > V_m(t+dt)` (local maximum detected)

        5. **Refractory state update**: If spike detected, set ``refractory_step_count``
           to :math:`\lceil t_{ref} / dt \rceil`; otherwise decrement if nonzero.

        6. **Buffer next stimulation current**: Store ``I_stim`` for next step
           (one-step delay matching NEST buffer semantics).

        Parameters
        ----------
        x : ArrayLike, default 0 pA
            External stimulation current for this time step. This is added to ``I_e``
            and should include:

            - Gap-junction current: :math:`I_{gap} = \sum_j g_{gap,ij}(V_j - V_i)`
            - Any additional bias or time-varying input current

            Shape must broadcast with ``(*in_size,)`` or be scalar.
            Unit: picoamperes (pA).

        Returns
        -------
        spike : ArrayLike
            Binary spike output with shape ``(*in_size,)``.
            Dtype: ``float64``. Values of ``1.0`` indicate at least one spike
            event occurred during the integrated interval :math:`(t, t+dt]`.

        Notes
        -----
        **Numerical Integration Details:**

        - All neurons are integrated simultaneously using a vectorized adaptive
          RKF45 loop implemented in JAX, providing efficient GPU/TPU execution.
        - The RKF45 (Runge-Kutta-Fehlberg) method uses adaptive step-size control
          with error tolerance ``gsl_error_tol``.
        - Integration includes in-loop spike detection with local maximum criterion.

        **Spike Detection Logic:**

        The three-condition spike criterion prevents multiple detections per action potential:

        1. **Refractory guard**: Ensures minimum inter-spike interval.
        2. **Threshold crossing**: Voltage must exceed :math:`V_T + 30` mV.
        3. **Local maximum**: :math:`V_{old} > V_m` ensures detection only at peak,
           not during rising or falling phases.

        This physiological detection method differs from IF models' threshold-reset mechanism.

        **Gap-Junction Current Handling:**

        Gap junctions are **not** computed internally. You must:

        1. Compute neighbor voltage differences externally.
        2. Calculate :math:`I_{gap,i} = \sum_j g_{gap,ij}(V_j - V_i)`.
        3. Pass the result as the ``x`` parameter.

        For networks, this typically requires gathering :math:`V_j` from connected neurons
        before calling :meth:`update`.

        See Also
        --------
        init_state : Initialize state variables before calling ``update()``.
        get_spike : Compute differentiable spike output from voltage.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        m_val = self.m.value  # unitless
        h_val = self.h.value  # unitless
        n_val = self.n.value  # unitless
        dg_ex = self.dg_ex.value  # nS/ms
        g_ex = self.g_ex.value  # nS
        dg_in = self.dg_in.value  # nS/ms
        g_in = self.g_in.value  # nS
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms
        V_old = self.V_old.value  # mV

        # Spike detection threshold: V_T + 30 mV
        v_spike_detect = self.V_T + 30.0 * u.mV

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(
            V=V, m=m_val, h=h_val, n=n_val,
            dg_ex=dg_ex, g_ex=g_ex, dg_in=dg_in, g_in=g_in
        )
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            V_old=V_old,
            v_spike_detect=v_spike_detect,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V = ode_state.V
        m_val, h_val, n_val = ode_state.m, ode_state.h, ode_state.n
        dg_ex, g_ex = ode_state.dg_ex, ode_state.g_ex
        dg_in, g_in = ode_state.dg_in, ode_state.g_in
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in hh_cond_beta_gap_traub dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration).
        dg_ex_q, dg_in_q = self._sum_signed_delta_inputs()

        # Compute beta normalization factors.
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
        dg_ex = dg_ex + pscon_ex * dg_ex_q  # nS/ms + 1/ms * nS = nS/ms
        dg_in = dg_in + pscon_in * dg_in_q  # nS/ms + 1/ms * nS = nS/ms

        # Write back state.
        self.V.value = V
        self.m.value = m_val
        self.h.value = h_val
        self.n.value = n_val
        self.dg_ex.value = dg_ex
        self.g_ex.value = g_ex
        self.dg_in.value = dg_in
        self.g_in.value = g_in
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)
        self.V_old.value = V

        return u.math.asarray(spike_mask, dtype=dftype)
