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
from ._utils import is_tracer, AdaptiveRungeKuttaStep

__all__ = [
    'hh_psc_alpha_gap',
]


def _hh_psc_alpha_gap_equilibrium(V):
    r"""Compute equilibrium gating variables for hh_psc_alpha_gap at given voltage.

    This function computes steady-state values of all four gating variables
    (m, h, n, p) using the voltage-dependent rate functions specific to the
    ``hh_psc_alpha_gap`` model. These rate functions differ from the classic
    Hodgkin-Huxley formulation and are based on the kinetics described in
    Mancilla et al. (2007) for modeling gap-junction-coupled interneurons.

    The equilibrium values are computed as:

    .. math::

       x_\infty = \frac{\alpha_x(V)}{\alpha_x(V) + \beta_x(V)}

    for each gating variable :math:`x \in \{m, h, n, p\}`.

    **Mathematical Details:**

    The rate functions (with voltage V in mV, rates in 1/ms) are:

    .. math::

       \alpha_m &= \frac{40(V - 75.5)}{1 - e^{-(V - 75.5)/13.5}}, \quad
       \beta_m  = \frac{1.2262}{e^{V/42.248}} \\
       \alpha_h &= \frac{0.0035}{e^{V/24.186}}, \quad
       \beta_h  = \frac{0.017(51.25 + V)}{1 - e^{-(51.25 + V)/5.2}} \\
       \alpha_n &= \frac{0.014(V + 44)}{1 - e^{-(V + 44)/2.3}}, \quad
       \beta_n  = \frac{0.0043}{e^{(V + 44)/34}} \\
       \alpha_p &= \frac{V - 95}{1 - e^{-(V - 95)/11.8}}, \quad
       \beta_p  = \frac{0.025}{e^{V/22.222}}

    Numerical Considerations
    ------------------------

    - The rate functions contain exponential terms that may produce division
      by zero or overflow at specific voltages. NumPy's exp function handles
      overflow by returning inf, which propagates correctly through the
      equilibrium calculation.
    - At voltages where the denominator :math:`1 - e^{-x}` approaches zero
      (e.g., when :math:`x \approx 0`), numerical instability may occur.
      However, for typical physiological voltage ranges (-100 to +50 mV),
      these expressions are well-behaved.

    **Usage:**

    This function is primarily used during state initialization to set gating
    variables to their equilibrium values at the initial membrane potential,
    avoiding transient artifacts from arbitrary initial conditions.

    Parameters
    ----------
    V : float
        Membrane potential in millivolts (mV). Typically in the range
        [-100, +50] mV for physiological conditions.

    Returns
    -------
    m_inf : float
        Equilibrium sodium activation (range [0, 1]).
    h_inf : float
        Equilibrium sodium inactivation (range [0, 1]).
    n_inf : float
        Equilibrium potassium Kv1 activation (range [0, 1]).
    p_inf : float
        Equilibrium potassium Kv3 activation (range [0, 1]).

    Notes
    -----
    - This function uses NumPy for computation and is not JIT-compiled. It is
      intended for use during initialization only, not during simulation loops.
    - The returned values are unitless (dimensionless gating variables).
    - For V = -69.604 mV (the NEST default initial voltage), the equilibrium
      values place the neuron near its resting state.

    References
    ----------
    .. [1] Mancilla JG, Lewis TG, Pinto DJ, Rinzel J, Connors BW (2007).
           Synchronization of electrically coupled pairs of inhibitory
           interneurons in neocortex. Journal of Neuroscience, 27:2058-2073.
           DOI: https://doi.org/10.1523/JNEUROSCI.2715-06.2007

    Examples
    --------
    Compute equilibrium values at resting potential:

    .. code-block:: python

       >>> V_rest = -69.604  # mV
       >>> m_inf, h_inf, n_inf, p_inf = _hh_psc_alpha_gap_equilibrium(V_rest)
       >>> print(f"m={m_inf:.4f}, h={h_inf:.4f}, n={n_inf:.4f}, p={p_inf:.4f}")
       m=0.0703, h=0.9541, n=0.1042, p=0.0000

    Compare equilibrium at depolarized voltage:

    .. code-block:: python

       >>> V_depol = -50.0  # mV
       >>> m_inf, h_inf, n_inf, p_inf = _hh_psc_alpha_gap_equilibrium(V_depol)
       >>> print(f"Sodium activation increased: m={m_inf:.4f}")
       Sodium activation increased: m=0.1523
    """
    alpha_m = 40.0 * (V - 75.5) / (1.0 - np.exp(-(V - 75.5) / 13.5))
    beta_m = 1.2262 / np.exp(V / 42.248)
    alpha_h = 0.0035 / np.exp(V / 24.186)
    beta_h = 0.017 * (51.25 + V) / (1.0 - np.exp(-(51.25 + V) / 5.2))
    alpha_n = 0.014 * (V + 44.0) / (1.0 - np.exp(-(V + 44.0) / 2.3))
    beta_n = 0.0043 / np.exp((V + 44.0) / 34.0)
    alpha_p = (V - 95.0) / (1.0 - np.exp(-(V - 95.0) / 11.8))
    beta_p = 0.025 / np.exp(V / 22.222)

    m_inf = alpha_m / (alpha_m + beta_m)
    h_inf = alpha_h / (alpha_h + beta_h)
    n_inf = alpha_n / (alpha_n + beta_n)
    p_inf = alpha_p / (alpha_p + beta_p)
    return m_inf, h_inf, n_inf, p_inf


class hh_psc_alpha_gap(NESTNeuron):
    r"""NEST-compatible Hodgkin-Huxley neuron with alpha PSCs and gap junctions.

    Short Description
    -----------------

    Conductance-based spiking neuron model implementing Hodgkin-Huxley
    dynamics with alpha-function postsynaptic currents and support for
    gap-junction coupling. Uses modified ion-channel kinetics from Mancilla
    et al. (2007) with two distinct potassium conductances (Kv1, Kv3) for
    modeling gap-junction-coupled inhibitory interneurons.

    **Model Overview**

    ``hh_psc_alpha_gap`` extends the classic Hodgkin-Huxley formalism with:

    - **Sodium (Na) conductance:** Activation gate :math:`m`, inactivation
      gate :math:`h`
    - **Two potassium conductances:** Fast Kv3 with :math:`p` gate, slow Kv1
      with :math:`n` gate
    - **Leak conductance:** Passive membrane current
    - **Alpha-function PSCs:** Second-order synaptic current dynamics for
      excitatory and inhibitory inputs
    - **Gap-junction support:** External resistive coupling current
      :math:`I_{gap}`
    - **Hybrid spike detection:** Combines voltage threshold (0 mV) with
      local maximum detection
    - **Explicit refractoriness:** Suppresses spike emission during
      refractory period; subthreshold dynamics continue evolving

    This implementation replicates NEST's ``hh_psc_alpha_gap`` model
    (``models/hh_psc_alpha_gap.{h,cpp}``), using adaptive Runge-Kutta
    integration (RK45/Dormand-Prince) to match NEST's GSL RKF45 solver.

    **1. Membrane Potential Dynamics**

    The membrane voltage evolves according to:

    .. math::

       C_m \frac{dV_m}{dt} = -(I_{Na} + I_K + I_L)
                              + I_{stim} + I_e
                              + I_{syn,ex} + I_{syn,in}
                              + I_{gap}

    where the ionic currents are:

    .. math::

       I_{Na} &= g_{Na}\, m^3\, h\, (V_m - E_{Na})  \\
       I_K    &= (g_{Kv1}\, n^4 + g_{Kv3}\, p^2)\, (V_m - E_K)  \\
       I_L    &= g_L\, (V_m - E_L)

    The potassium current combines contributions from slow Kv1 channels
    (:math:`n^4` gating) and fast Kv3 channels (:math:`p^2` gating), which
    is the key difference from standard HH models.

    **2. Gating Variable Dynamics**

    All four gating variables :math:`x \in \{m, h, n, p\}` follow first-order
    kinetics:

    .. math::

       \frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)\,x

    The voltage-dependent rate functions (voltage :math:`V` in mV, rates in
    1/ms) are:

    .. math::

       \alpha_m &= \frac{40\,(V - 75.5)}{1 - e^{-(V - 75.5)/13.5}}, \quad
       \beta_m  = \frac{1.2262}{e^{V/42.248}}                              \\
       \alpha_h &= \frac{0.0035}{e^{V/24.186}}, \quad
       \beta_h  = \frac{0.017\,(51.25 + V)}{1 - e^{-(51.25 + V)/5.2}}     \\
       \alpha_n &= \frac{0.014\,(V + 44)}{1 - e^{-(V + 44)/2.3}}, \quad
       \beta_n  = \frac{0.0043}{e^{(V + 44)/34}}                           \\
       \alpha_p &= \frac{V - 95}{1 - e^{-(V - 95)/11.8}}, \quad
       \beta_p  = \frac{0.025}{e^{V/22.222}}

    These kinetics differ from the classic Hodgkin-Huxley equations and are
    based on experimental measurements from neocortical interneurons.

    **3. Gap-Junction Current**

    Gap junctions provide resistive electrical coupling:

    .. math::

       I_{gap} = \sum_j g_{ij}\,(V_j - V_i)

    where :math:`g_{ij}` is the gap-junction conductance between neuron *i*
    and neuron *j*, and :math:`V_j` is the membrane potential of the coupled
    neuron. In this single-neuron model, :math:`I_{gap}` must be computed
    externally and provided as input via the ``x`` parameter to ``update()``
    or through ``add_current_input()``.

    **4. Alpha-Function Synaptic Currents**

    Each synapse type (excitatory/inhibitory) uses a second-order system to
    generate alpha-shaped postsynaptic currents:

    .. math::

       \frac{dI_{syn}}{dt}  &= dI_{syn} - \frac{I_{syn}}{\tau_{syn}} \\
       \frac{d(dI_{syn})}{dt} &= -\frac{dI_{syn}}{\tau_{syn}}

    An incoming spike with weight :math:`w` (in pA) increments
    :math:`dI_{syn}` by :math:`w \cdot e / \tau_{syn}`, ensuring the peak
    current reaches :math:`w` pA. The factor :math:`e = \exp(1)` normalizes
    the alpha function.

    **5. Spike Detection Mechanism**

    Spikes are detected using a combined threshold-and-local-maximum criterion:

    1. **Not in refractory period:** ``r == 0``
    2. **Threshold crossing:** :math:`V_m \geq 0` mV
    3. **Local maximum:** :math:`V_{old} > V_m` (voltage is decreasing)

    All three conditions must be satisfied simultaneously. This prevents
    multiple spike detections during the rising and falling phases of the
    action potential. Unlike integrate-and-fire models, **no voltage reset**
    occurs---repolarization happens naturally through activation of potassium
    currents.

    **6. Refractory Period**

    During the refractory period (duration :math:`t_{ref}`), spike emission
    is suppressed, but the neuron's subthreshold dynamics continue to evolve
    according to the differential equations. This differs from models that
    clamp the membrane potential during refractoriness.

    **7. Numerical Integration**

    NEST uses GSL's RKF45 (Runge-Kutta-Fehlberg 4th/5th order) adaptive
    integrator with absolute tolerance 1e-6 and relative tolerance 0. This
    implementation uses a vectorized adaptive RKF45 integrator via
    ``AdaptiveRungeKuttaStep`` with matching tolerances. The 9-dimensional ODE
    system (V, m, h, n, p, dI_ex, I_ex, dI_in, I_in) is integrated
    simultaneously for all neurons over each time step.

    Computational Complexity
    ------------------------

    - **Per neuron, per time step:** One adaptive ODE integration (~10-50
      function evaluations depending on step size control)
    - **Scaling:** Linear in population size (vectorized across neurons)
    - **Memory:** O(population_size) for state storage
    - **JIT-compiled:** Uses JAX-based adaptive RKF45 integrator for
      high performance and GPU compatibility

    Parameters
    ----------
    in_size : int or tuple of int
        Population shape. Can be an integer (1D population) or tuple of
        integers (multidimensional population). Defines the number of neurons
        in the group.
    E_L : ArrayLike, optional
        Leak reversal potential (resting potential). Scalar or array with
        shape broadcastable to ``in_size``. Unit: mV. Default: -70.0 mV.
    C_m : ArrayLike, optional
        Membrane capacitance. Must be strictly positive. Scalar or array with
        shape broadcastable to ``in_size``. Unit: pF. Default: 40.0 pF.
    g_Na : ArrayLike, optional
        Sodium peak conductance. Must be non-negative. Scalar or array with
        shape broadcastable to ``in_size``. Unit: nS. Default: 4500.0 nS.
    g_Kv1 : ArrayLike, optional
        Potassium Kv1 (slow) peak conductance. Must be non-negative. Scalar
        or array with shape broadcastable to ``in_size``. Unit: nS.
        Default: 9.0 nS.
    g_Kv3 : ArrayLike, optional
        Potassium Kv3 (fast) peak conductance. Must be non-negative. Scalar
        or array with shape broadcastable to ``in_size``. Unit: nS.
        Default: 9000.0 nS.
    g_L : ArrayLike, optional
        Leak conductance. Must be non-negative. Scalar or array with shape
        broadcastable to ``in_size``. Unit: nS. Default: 10.0 nS.
    E_Na : ArrayLike, optional
        Sodium reversal potential. Scalar or array with shape broadcastable
        to ``in_size``. Unit: mV. Default: 74.0 mV.
    E_K : ArrayLike, optional
        Potassium reversal potential. Scalar or array with shape
        broadcastable to ``in_size``. Unit: mV. Default: -90.0 mV.
    t_ref : ArrayLike, optional
        Duration of refractory period. Must be non-negative. During this
        period, spike emission is suppressed but dynamics continue evolving.
        Scalar or array with shape broadcastable to ``in_size``. Unit: ms.
        Default: 2.0 ms.
    tau_syn_ex : ArrayLike, optional
        Excitatory synaptic time constant (alpha-function rise time). Must be
        strictly positive. Scalar or array with shape broadcastable to
        ``in_size``. Unit: ms. Default: 0.2 ms.
    tau_syn_in : ArrayLike, optional
        Inhibitory synaptic time constant (alpha-function rise time). Must be
        strictly positive. Scalar or array with shape broadcastable to
        ``in_size``. Unit: ms. Default: 2.0 ms.
    I_e : ArrayLike, optional
        Constant external input current. Positive values are depolarizing.
        Scalar or array with shape broadcastable to ``in_size``. Unit: pA.
        Default: 0.0 pA.
    V_m_init : ArrayLike or None, optional
        Initial membrane potential. If None, uses NEST's default value of
        -69.604012 mV. Scalar or array with shape broadcastable to
        ``in_size``. Unit: mV. Default: None.
    Act_m_init : ArrayLike or None, optional
        Initial sodium activation gating variable. Must be in [0, 1]. If
        None, computed from equilibrium at ``V_m_init``. Scalar or array with
        shape broadcastable to ``in_size``. Unitless. Default: None.
    Inact_h_init : ArrayLike or None, optional
        Initial sodium inactivation gating variable. Must be in [0, 1]. If
        None, computed from equilibrium at ``V_m_init``. Scalar or array with
        shape broadcastable to ``in_size``. Unitless. Default: None.
    Act_n_init : ArrayLike or None, optional
        Initial Kv1 activation gating variable. Must be in [0, 1]. If None,
        computed from equilibrium at ``V_m_init``. Scalar or array with shape
        broadcastable to ``in_size``. Unitless. Default: None.
    Inact_p_init : ArrayLike or None, optional
        Initial Kv3 activation gating variable. Must be in [0, 1]. If None,
        computed from equilibrium at ``V_m_init``. Scalar or array with shape
        broadcastable to ``in_size``. Unitless. Default: None.
    spk_fun : Callable, optional
        Surrogate gradient function for differentiable spike generation.
        Should be a callable from ``braintools.surrogate`` with signature
        ``(ArrayLike) -> ArrayLike``. Used for gradient-based learning.
        Default: ``braintools.surrogate.ReluGrad()``.
    spk_reset : {'hard', 'soft'}, optional
        Spike reset mode. For HH models, this affects surrogate gradient
        computation only (no actual voltage reset occurs). 'hard': stop
        gradient propagation; 'soft': allow gradient flow. Default: 'hard'.
    gsl_error_tol : float, optional
        Absolute tolerance for the embedded RKF45 error estimate. Must be
        strictly positive. Default: 1e-6 (matching NEST).
    name : str or None, optional
        Name of the neuron population for identification. If None, an
        automatic name is generated. Default: None.


    Parameter Mapping
    -----------------

    ==================== ================== =============================== ====================================================
    **Parameter**        **Default**        **Math Symbol**                 **Description**
    ==================== ================== =============================== ====================================================
    ``in_size``          (required)         ---                               Population shape
    ``E_L``              -70.0 mV           :math:`E_L`                     Leak reversal potential (resting potential)
    ``C_m``              40.0 pF            :math:`C_m`                     Membrane capacitance
    ``g_Na``             4500.0 nS          :math:`g_{Na}`                  Sodium peak conductance
    ``g_Kv1``            9.0 nS             :math:`g_{Kv1}`                 Potassium Kv1 (slow) peak conductance
    ``g_Kv3``            9000.0 nS          :math:`g_{Kv3}`                 Potassium Kv3 (fast) peak conductance
    ``g_L``              10.0 nS            :math:`g_L`                     Leak conductance
    ``E_Na``             74.0 mV            :math:`E_{Na}`                  Sodium reversal potential
    ``E_K``              -90.0 mV           :math:`E_K`                     Potassium reversal potential
    ``t_ref``            2.0 ms             :math:`t_{ref}`                 Duration of refractory period
    ``tau_syn_ex``       0.2 ms             :math:`\tau_{syn,ex}`           Excitatory synaptic time constant
    ``tau_syn_in``       2.0 ms             :math:`\tau_{syn,in}`           Inhibitory synaptic time constant
    ``I_e``              0.0 pA             :math:`I_e`                     Constant external input current
    ``V_m_init``         -69.60401 mV       ---                               Initial membrane potential (NEST default)
    ``Act_m_init``       None               ---                               Initial Na activation (None -> equilibrium)
    ``Inact_h_init``     None               ---                               Initial Na inactivation (None -> equilibrium)
    ``Act_n_init``       None               ---                               Initial Kv1 activation (None -> equilibrium)
    ``Inact_p_init``     None               ---                               Initial Kv3 activation (None -> equilibrium)
    ``spk_fun``          ReluGrad()         ---                               Surrogate spike function
    ``spk_reset``        'hard'             ---                               Reset mode for gradient computation
    ``gsl_error_tol``    1e-6               ---                               Absolute tolerance for RKF45 error estimate
    ==================== ================== =============================== ====================================================

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential :math:`V_m`. Shape: ``in_size``.
        Unit: mV.
    m : brainstate.HiddenState
        Sodium activation gating variable. Shape: ``in_size``.
        Range: [0, 1]. Unitless.
    h : brainstate.HiddenState
        Sodium inactivation gating variable. Shape: ``in_size``.
        Range: [0, 1]. Unitless.
    n : brainstate.HiddenState
        Potassium Kv1 activation gating variable. Shape:
        ``in_size``. Range: [0, 1]. Unitless.
    p : brainstate.HiddenState
        Potassium Kv3 activation gating variable. Shape:
        ``in_size``. Range: [0, 1]. Unitless.
    I_syn_ex : brainstate.ShortTermState
        Excitatory postsynaptic current. Shape: ``in_size``.
        Unit: pA.
    I_syn_in : brainstate.ShortTermState
        Inhibitory postsynaptic current. Shape: ``in_size``.
        Unit: pA.
    dI_syn_ex : brainstate.ShortTermState
        Excitatory alpha-kernel derivative state.
        Shape: ``in_size``. Unit: pA/ms.
    dI_syn_in : brainstate.ShortTermState
        Inhibitory alpha-kernel derivative state.
        Shape: ``in_size``. Unit: pA/ms.
    I_stim : brainstate.ShortTermState
        Stimulation current buffer for next time step. Shape:
        ``in_size``. Unit: pA.
    refractory_step_count : brainstate.ShortTermState
        Refractory countdown in discrete time steps. Counts down from
        ``ceil(t_ref / dt)`` to 0. Shape: ``in_size``.
        Unit: steps (integer).
    integration_step : brainstate.ShortTermState
        Persistent RKF45 substep size estimate (ms).
    last_spike_time : brainstate.ShortTermState
        Time of most recent spike. Shape: ``in_size``.
        Unit: ms.

    Raises
    ------
    ValueError
        If ``C_m <= 0`` (capacitance must be strictly positive).
    ValueError
        If ``t_ref < 0`` (refractory time cannot be negative).
    ValueError
        If ``tau_syn_ex <= 0`` or ``tau_syn_in <= 0`` (time constants must be
        strictly positive).
    ValueError
        If any conductance (``g_Na``, ``g_Kv1``, ``g_Kv3``, ``g_L``) is
        negative.

    Notes
    -----
    **Differences from ``hh_psc_alpha``:**

    - Adds gap-junction current :math:`I_{gap}` to membrane equation
    - Uses modified ion-channel kinetics (Mancilla et al. 2007) with two
      potassium channel types (Kv1, Kv3)
    - Different default conductance values optimized for interneuron models

    **Spike Weights and Input Interpretation:**

    - Positive spike weights -> excitatory input (added to ``dI_syn_ex``)
    - Negative spike weights -> inhibitory input (added to ``dI_syn_in``)
    - Weight magnitude in pA determines peak current amplitude
    - Gap-junction current should be provided via ``x`` parameter to
      ``update()`` or registered via ``add_current_input()``

    **Integration Accuracy:**

    - Adaptive step-size control ensures high accuracy but variable
      computational cost per step
    - Default tolerance matches NEST for comparable numerical behavior

    **Gradient-Based Learning:**

    - Surrogate gradients enable backpropagation through spike generation
    - The ``spk_fun`` parameter controls the shape of the surrogate gradient
    - No actual voltage reset occurs, so gradients flow through natural
      action potential dynamics

    References
    ----------
    .. [1] Hodgkin AL, Huxley AF (1952). A quantitative description of
           membrane current and its application to conduction and excitation
           in nerve. The Journal of Physiology 117:500-544.
           DOI: https://doi.org/10.1113/jphysiol.1952.sp004764
    .. [2] Mancilla JG, Lewis TG, Pinto DJ, Rinzel J, Connors BW (2007).
           Synchronization of electrically coupled pairs of inhibitory
           interneurons in neocortex. Journal of Neuroscience, 27:2058-2073.
           DOI: https://doi.org/10.1523/JNEUROSCI.2715-06.2007
    .. [3] Gerstner W, Kistler W (2002). Spiking neuron models: Single
           neurons, populations, plasticity. Cambridge University Press.
    .. [4] Hahne J, Helias M, Kunkel S, Igarashi J, Bolten M, Frommer A,
           Diesmann M (2015). A unified framework for spiking and gap-junction
           interactions in distributed neuronal network simulations. Frontiers
           in Neuroinformatics, 9:22.
           DOI: https://doi.org/10.3389/fninf.2015.00022

    See Also
    --------
    hh_psc_alpha : Hodgkin-Huxley neuron without gap-junction support.
    hh_cond_exp_traub : Alternative HH implementation with exponential PSCs.
    iaf_cond_exp : Simpler integrate-and-fire model with conductance-based
        synapses.

    Examples
    --------
    Create a single gap-junction-coupled HH neuron:

    .. code-block:: python

       >>> import brainpy.state as bs
       >>> import saiunit as u
       >>> neuron = bs.hh_psc_alpha_gap(in_size=1, E_L=-70*u.mV, C_m=40*u.pF)
       >>> neuron.init_all_states()

    Simulate with constant input current:

    .. code-block:: python

       >>> import brainstate as bst
       >>> with bst.environ.context(dt=0.1*u.ms):
       ...     neuron.init_all_states()
       ...     spikes = []
       ...     for i in range(1000):
       ...         spk = neuron.update(x=500*u.pA)  # 500 pA input
       ...         spikes.append(spk.item())

    Create a population with heterogeneous capacitance:

    .. code-block:: python

       >>> import jax.numpy as jnp
       >>> C_m_values = jnp.linspace(30, 50, 10) * u.pF
       >>> neurons = bs.hh_psc_alpha_gap(in_size=10, C_m=C_m_values)
       >>> neurons.init_all_states()

    Add gap-junction coupling between two neurons:

    .. code-block:: python

       >>> neuron1 = bs.hh_psc_alpha_gap(in_size=1)
       >>> neuron2 = bs.hh_psc_alpha_gap(in_size=1)
       >>> neuron1.init_all_states()
       >>> neuron2.init_all_states()
       >>> g_gap = 0.5 * u.nS  # gap-junction conductance
       >>> # In update loop:
       >>> I_gap_1 = g_gap * (neuron2.V.value - neuron1.V.value)
       >>> I_gap_2 = g_gap * (neuron1.V.value - neuron2.V.value)
       >>> spk1 = neuron1.update(x=I_gap_1)
       >>> spk2 = neuron2.update(x=I_gap_2)
    """

    __module__ = 'brainpy.state'

    # NEST default initial membrane potential (mV)
    _NEST_V_INIT = -69.60401191631222

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70.0 * u.mV,
        C_m: ArrayLike = 40.0 * u.pF,
        g_Na: ArrayLike = 4500.0 * u.nS,
        g_Kv1: ArrayLike = 9.0 * u.nS,
        g_Kv3: ArrayLike = 9000.0 * u.nS,
        g_L: ArrayLike = 10.0 * u.nS,
        E_Na: ArrayLike = 74.0 * u.mV,
        E_K: ArrayLike = -90.0 * u.mV,
        t_ref: ArrayLike = 2.0 * u.ms,
        tau_syn_ex: ArrayLike = 0.2 * u.ms,
        tau_syn_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0.0 * u.pA,
        V_m_init: ArrayLike = None,
        Act_m_init: ArrayLike = None,
        Inact_h_init: ArrayLike = None,
        Act_n_init: ArrayLike = None,
        Inact_p_init: ArrayLike = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        gsl_error_tol: ArrayLike = 1e-6,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.g_Na = braintools.init.param(g_Na, self.varshape)
        self.g_Kv1 = braintools.init.param(g_Kv1, self.varshape)
        self.g_Kv3 = braintools.init.param(g_Kv3, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.E_Na = braintools.init.param(E_Na, self.varshape)
        self.E_K = braintools.init.param(E_K, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        if V_m_init is None:
            V_m_init = self._NEST_V_INIT * u.mV
        self.V_m_init = V_m_init
        self.Act_m_init = Act_m_init
        self.Inact_h_init = Inact_h_init
        self.Act_n_init = Act_n_init
        self.Inact_p_init = Inact_p_init

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
        r"""Validate model parameters against NEST constraints.

        Raises
        ------
        ValueError
            If parameter inequalities or positivity constraints are violated.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.t_ref, self.g_Na)):
            return
        if np.any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self.tau_syn_ex <= 0.0 * u.ms) or np.any(
            self.tau_syn_in <= 0.0 * u.ms
        ):
            raise ValueError('All time constants must be strictly positive.')
        if (
            np.any(self.g_Na < 0.0 * u.nS)
            or np.any(self.g_Kv1 < 0.0 * u.nS)
            or np.any(self.g_Kv3 < 0.0 * u.nS)
            or np.any(self.g_L < 0.0 * u.nS)
        ):
            raise ValueError('All conductances must be non-negative.')
        if np.any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Sets up membrane potential, gating variables, synaptic currents, and
        internal state tracking. Gating variables are initialized to their
        equilibrium values at the initial membrane potential unless explicitly
        specified, ensuring the neuron starts in a consistent resting state
        without transient artifacts.

        **Initialization Strategy:**

        1. **Membrane potential:** Set to ``V_m_init`` (default: NEST's
           -69.604012 mV equilibrium value)
        2. **Gating variables:** If ``Act_m_init``, ``Inact_h_init``,
           ``Act_n_init``, or ``Inact_p_init`` are None, compute equilibrium
           values at ``V_m_init`` using ``_hh_psc_alpha_gap_equilibrium()``
        3. **Synaptic currents:** Initialize ``I_syn_ex``, ``I_syn_in`` and
           their derivatives to zero
        4. **Refractory state:** Set refractory counter to 0 (not refractory)
        5. **Spike timing:** Set ``last_spike_time`` to large negative value
           (-1e7 ms)

        **Equilibrium Initialization Rationale:**

        Starting gating variables at their equilibrium values for the given
        initial voltage prevents spurious transient currents during the first
        few time steps. Without this, arbitrary initial values would cause
        artificial spikes or oscillations as the system relaxes to equilibrium.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Attributes Set
        --------------
        This method initializes the following instance attributes:

        V : brainstate.HiddenState
            Membrane potential. Initial value: ``V_m_init``. Unit: mV.
        m : brainstate.HiddenState
            Sodium activation gating variable. Initial value: ``Act_m_init``
            if provided, otherwise equilibrium at ``V_m_init``. Range: [0, 1].
        h : brainstate.HiddenState
            Sodium inactivation gating variable. Initial value:
            ``Inact_h_init`` if provided, otherwise equilibrium at
            ``V_m_init``. Range: [0, 1].
        n : brainstate.HiddenState
            Kv1 activation gating variable. Initial value: ``Act_n_init`` if
            provided, otherwise equilibrium at ``V_m_init``. Range: [0, 1].
        p : brainstate.HiddenState
            Kv3 activation gating variable. Initial value: ``Inact_p_init``
            if provided, otherwise equilibrium at ``V_m_init``. Range: [0, 1].
        I_syn_ex : brainstate.ShortTermState
            Excitatory synaptic current. Initial value: 0. Unit: pA.
        I_syn_in : brainstate.ShortTermState
            Inhibitory synaptic current. Initial value: 0. Unit: pA.
        dI_syn_ex : brainstate.ShortTermState
            Time derivative of excitatory current (alpha kernel state).
            Initial value: 0. Unit: pA/ms.
        dI_syn_in : brainstate.ShortTermState
            Time derivative of inhibitory current (alpha kernel state).
            Initial value: 0. Unit: pA/ms.
        I_stim : brainstate.ShortTermState
            Stimulation current buffer. Initial value: 0. Unit: pA.
        refractory_step_count : brainstate.ShortTermState
            Refractory countdown in time steps. Initial value: 0 (not
            refractory). Unit: steps (int32).
        integration_step : brainstate.ShortTermState
            Persistent RKF45 substep size estimate (ms).
        last_spike_time : brainstate.ShortTermState
            Time of last spike. Initial value: -1e7 ms (far in the past).
            Unit: ms.

        Notes
        -----
        - Must be called before ``update()`` to ensure state variables exist
        - Can be called multiple times to reinitialize (e.g., between trials)
        - For heterogeneous populations with per-neuron initial conditions,
          pass arrays to ``V_m_init``, ``Act_m_init``, etc. during construction
        - The NEST default initial voltage (-69.604012 mV) places the neuron
          near its resting state with minimal initial transients

        **State Variable Types:**

        - ``HiddenState``: For slow variables (V, gating variables) that
          persist across time steps and require gradient tracking
        - ``ShortTermState``: For fast variables (currents, counters) that
          are recomputed each step or have short-term dynamics

        See Also
        --------
        _hh_psc_alpha_gap_equilibrium : Computes equilibrium gating values.
        update : Main simulation step that uses these state variables.

        Examples
        --------
        Basic initialization:

        .. code-block:: python

           >>> neuron = bs.hh_psc_alpha_gap(in_size=10)
           >>> neuron.init_state()
           >>> print(neuron.V.value.shape)
           (10,)

        Custom initial conditions:

        .. code-block:: python

           >>> import jax.numpy as jnp
           >>> V_init = jnp.linspace(-75, -65, 10) * u.mV
           >>> neuron = bs.hh_psc_alpha_gap(in_size=10, V_m_init=V_init)
           >>> neuron.init_state()
           >>> print(neuron.V.value)  # voltage varies across population

        Initialize with custom gating variables:

        .. code-block:: python

           >>> neuron = bs.hh_psc_alpha_gap(
           ...     in_size=1,
           ...     Act_m_init=0.1,  # specific sodium activation
           ...     Inact_h_init=0.9  # specific sodium inactivation
           ... )
           >>> neuron.init_state()
           >>> print(f"m={neuron.m.value}, h={neuron.h.value}")
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        # Compute initial V as numpy for equilibrium calculation
        V_m_init_mV = self.V_m_init / u.mV
        V_init_scalar = float(np.asarray(V_m_init_mV).flat[0]) if np.asarray(V_m_init_mV).ndim > 0 else float(V_m_init_mV)

        # Compute equilibrium gating variables at initial V
        m_eq, h_eq, n_eq, p_eq = _hh_psc_alpha_gap_equilibrium(V_init_scalar)

        V = braintools.init.param(braintools.init.Constant(self.V_m_init), self.varshape)

        if self.Act_m_init is not None:
            m_init = float(np.asarray(self.Act_m_init).flat[0]) if np.asarray(self.Act_m_init).ndim > 0 else float(self.Act_m_init)
        else:
            m_init = m_eq
        if self.Inact_h_init is not None:
            h_init = float(np.asarray(self.Inact_h_init).flat[0]) if np.asarray(self.Inact_h_init).ndim > 0 else float(self.Inact_h_init)
        else:
            h_init = h_eq
        if self.Act_n_init is not None:
            n_init = float(np.asarray(self.Act_n_init).flat[0]) if np.asarray(self.Act_n_init).ndim > 0 else float(self.Act_n_init)
        else:
            n_init = n_eq
        if self.Inact_p_init is not None:
            p_init = float(np.asarray(self.Inact_p_init).flat[0]) if np.asarray(self.Inact_p_init).ndim > 0 else float(self.Inact_p_init)
        else:
            p_init = p_eq

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
        self.p = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(p_init), self.varshape)
        )

        zeros_pA = u.math.zeros(self.varshape, dtype=dftype) * u.pA
        zeros_pA_per_ms = u.math.zeros(self.varshape, dtype=dftype) * (u.pA / u.ms)

        self.I_syn_ex = brainstate.ShortTermState(zeros_pA)
        self.I_syn_in = brainstate.ShortTermState(zeros_pA)
        self.dI_syn_ex = brainstate.ShortTermState(zeros_pA_per_ms)
        self.dI_syn_in = brainstate.ShortTermState(zeros_pA_per_ms)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.V_old = brainstate.ShortTermState(V.copy())

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output from membrane potential.

        Applies the surrogate gradient function (``spk_fun``) to generate a
        continuous, differentiable spike signal from the membrane potential.
        This enables gradient-based learning through the spiking dynamics.

        For the HH model with combined threshold-and-local-maximum spike
        detection, this method is typically called with a specially crafted
        voltage signal (positive for spiking, negative otherwise) rather than
        the raw membrane potential.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Membrane potential or voltage-derived signal. If None, uses the
            current value of ``self.V.value``. Shape: ``in_size``.
            Unit: mV. Default: None.

        Returns
        -------
        spike_signal : ArrayLike
            Differentiable spike output with shape ``in_size``.
            The surrogate function maps the scaled voltage to a continuous
            output (typically in range [0, 1] or [-1, 1] depending on
            ``spk_fun``). Gradients flow through this function during
            backpropagation.

        Notes
        -----
        - The voltage is normalized by dividing by 1 mV before applying
          ``spk_fun`` to ensure dimensionless input
        - For binary spike detection, threshold the returned value at 0
        - The choice of ``spk_fun`` affects gradient magnitudes and learning
          dynamics (e.g., ``ReluGrad``, ``SigmoidGrad``, ``SuperSpike``)

        See Also
        --------
        update : Main simulation step that calls this method.
        braintools.surrogate : Module containing surrogate gradient functions.

        Examples
        --------
        Direct spike computation:

        .. code-block:: python

           >>> neuron = bs.hh_psc_alpha_gap(in_size=1)
           >>> neuron.init_all_states()
           >>> neuron.V.value = 10 * u.mV  # depolarized
           >>> spk = neuron.get_spike()
           >>> print(f"Spike signal: {spk}")

        Using custom voltage signal:

        .. code-block:: python

           >>> V_custom = jnp.array([1e-12, -1.0]) * u.mV  # spike/no-spike
           >>> spk = neuron.get_spike(V=V_custom)
        """
        V = self.V.value if V is None else V
        v_scaled = V / (1.0 * u.mV)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, p, dI_ex, I_ex, dI_in, I_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, V_old -- mutable
            auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        V = state.V
        m_ = state.m
        h_ = state.h
        n_ = state.n
        p_ = state.p

        # Gating variable rate functions (Mancilla et al. 2007 kinetics)
        # All rate arguments are unitless (mV / mV), rates are 1/ms
        V_mV = V / u.mV  # unitless

        alpha_m = 40.0 * (V_mV - 75.5) / (1.0 - u.math.exp(-(V_mV - 75.5) / 13.5)) / u.ms
        beta_m = 1.2262 / u.math.exp(V_mV / 42.248) / u.ms
        alpha_h = 0.0035 / u.math.exp(V_mV / 24.186) / u.ms
        beta_h = 0.017 * (51.25 + V_mV) / (1.0 - u.math.exp(-(51.25 + V_mV) / 5.2)) / u.ms
        alpha_n = 0.014 * (V_mV + 44.0) / (1.0 - u.math.exp(-(V_mV + 44.0) / 2.3)) / u.ms
        beta_n = 0.0043 / u.math.exp((V_mV + 44.0) / 34.0) / u.ms
        alpha_p = (V_mV - 95.0) / (1.0 - u.math.exp(-(V_mV - 95.0) / 11.8)) / u.ms
        beta_p = 0.025 / u.math.exp(V_mV / 22.222) / u.ms

        # Ionic currents
        I_Na = self.g_Na * m_ * m_ * m_ * h_ * (V - self.E_Na)
        I_K = (self.g_Kv1 * n_ * n_ * n_ * n_ + self.g_Kv3 * p_ * p_) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # Membrane voltage dynamics
        dV = (-(I_Na + I_K + I_L) + extra.i_stim + self.I_e + state.I_ex + state.I_in) / self.C_m

        # Gating variable dynamics
        dm = alpha_m * (1.0 - m_) - beta_m * m_
        dh = alpha_h * (1.0 - h_) - beta_h * h_
        dn = alpha_n * (1.0 - n_) - beta_n * n_
        dp = alpha_p * (1.0 - p_) - beta_p * p_

        # Alpha-kernel synaptic current dynamics
        ddI_ex = -state.dI_ex / self.tau_syn_ex
        dI_ex_dt = state.dI_ex - state.I_ex / self.tau_syn_ex
        ddI_in = -state.dI_in / self.tau_syn_in
        dI_in_dt = state.dI_in - state.I_in / self.tau_syn_in

        return DotDict(
            V=dV, m=dm, h=dh, n=dn, p=dp,
            dI_ex=ddI_ex, I_ex=dI_ex_dt, dI_in=ddI_in, I_in=dI_in_dt
        )

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection and refractory handling.

        Spikes are detected using a combined threshold-and-local-maximum
        criterion: (1) not refractory, (2) V >= 0 mV, (3) V_old > V
        (voltage is decreasing, i.e. local maximum has passed).

        Unlike integrate-and-fire models, no voltage reset occurs. The
        V_old tracking is updated on each accepted substep.

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, p, dI_ex, I_ex, dI_in, I_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, V_old.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & (
                (state.V < -1e3 * u.mV) |
                (state.m < -0.5) | (state.m > 1.5) |
                (state.h < -0.5) | (state.h > 1.5)
            )
        )

        # Spike detection: threshold crossing + local maximum
        not_refractory = extra.r <= 0
        crossed_threshold = state.V >= 0.0 * u.mV
        local_max = extra.V_old > state.V
        spike_now = accept & not_refractory & crossed_threshold & local_max

        spike_mask = extra.spike_mask | spike_now

        # Refractory update: set counter on spike
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count, extra.r)

        # Update V_old to track voltage for local maximum detection
        new_V_old = u.math.where(accept, state.V, extra.V_old)

        new_extra = DotDict({
            **extra,
            'spike_mask': spike_mask,
            'r': r,
            'unstable': unstable,
            'V_old': new_V_old,
        })
        return state, new_extra

    def update(self, x=0.0 * u.pA):
        r"""Advance neuron state by one simulation time step.

        Executes the full update cycle following NEST's ``hh_psc_alpha_gap``
        implementation order. This includes integrating the 9-dimensional ODE
        system (membrane potential, gating variables, and synaptic currents),
        processing incoming spikes, detecting output spikes, and managing the
        refractory state.

        **Update Sequence (Matching NEST Order):**

        1. **Record pre-integration voltage:** Save ``V_old`` for spike detection
        2. **Integrate ODEs:** Solve the 9D system over ``[t, t+dt]`` using
           adaptive RKF45
        3. **Process arriving spikes:** Add weighted spike inputs to
           ``dI_syn_ex`` / ``dI_syn_in`` derivative states
        4. **Detect spikes:** Check three-part condition (not refractory,
           threshold crossed, local maximum)
        5. **Update refractory counter:** Reset to ``t_ref/dt`` steps if
           spiking, otherwise decrement
        6. **Store stimulation buffer:** Save ``I_stim`` for next time step
        7. **Return spike output:** Compute surrogate spike signal

        **The 9-Dimensional ODE System:**

        The ODE integrator solves for state vector
        :math:`\mathbf{y} = [V_m, m, h, n, p, dI_{ex}, I_{ex}, dI_{in}, I_{in}]`
        using the dynamics described in the class docstring. All neurons in
        the population are integrated simultaneously (vectorized).

        **Spike Input Processing:**

        - Spike inputs arrive via ``sum_delta_inputs()`` (collects all
          registered delta-function inputs)
        - Positive weights -> excitatory: added to ``dI_syn_ex``
        - Negative weights -> inhibitory: added to ``dI_syn_in``
        - Normalization factor :math:`e/\tau_{syn}` ensures peak current
          equals weight magnitude

        **Current Input Processing:**

        - Continuous current inputs via ``sum_current_inputs()`` (collects
          parameter ``x`` and all registered current inputs)
        - Gap-junction current typically provided through ``x`` parameter
        - Also includes constant bias current ``I_e``

        **Spike Detection Logic:**

        .. code-block:: python

           spike = (r == 0) & (V_m >= 0.0) & (V_old > V_m)

        This ensures only one spike per action potential by requiring:
        (1) not refractory, (2) above threshold, (3) voltage decreasing
        (local maximum has passed).

        Numerical Considerations
        ------------------------

        - All neurons are integrated simultaneously via vectorized adaptive
          RKF45 (JIT-compiled, GPU-compatible)
        - Adaptive step-size control may use 10-50 function evaluations per
          time step depending on dynamics and error tolerances

        **Integration Tolerances:**

        The ODE solver uses ``gsl_error_tol`` as absolute tolerance to control
        step-size adaptation. Smaller values increase accuracy but require more
        function evaluations. Default value (1e-6) matches NEST's GSL settings.

        Parameters
        ----------
        x : ArrayLike, optional
            External input current. Can be scalar or array broadcastable to
            population shape. Typically includes gap-junction current computed
            as :math:`\sum_j g_{ij}(V_j - V_i)` for coupled networks. Also
            accepts stimulation currents from external devices. Unit: pA.
            Default: 0.0 pA.

        Returns
        -------
        spike_output : ArrayLike
            Differentiable spike signal with shape ``in_size``.
            Computed by applying surrogate gradient function ``spk_fun`` to a
            voltage-derived signal: positive when spiking (``V_out = 1e-12``),
            negative otherwise (``V_out = -1.0``). For binary spike detection,
            threshold at 0. For gradient-based learning, use the returned
            analog values with ``spk_fun``'s surrogate gradient.

        Notes
        -----
        **State Update Side Effects:**

        This method modifies the following instance attributes:

        - ``V.value``: Updated membrane potential
        - ``m.value, h.value, n.value, p.value``: Updated gating variables
        - ``I_syn_ex.value, I_syn_in.value``: Updated synaptic currents
        - ``dI_syn_ex.value, dI_syn_in.value``: Updated synaptic derivatives
        - ``I_stim.value``: Buffered stimulation for next step
        - ``refractory_step_count.value``: Updated refractory countdown
        - ``last_spike_time.value``: Spike time when spiking occurs

        **Gap-Junction Usage Example:**

        For a network with gap-junction coupling matrix G and voltage vector V:

        .. code-block:: python

           >>> G = [[0, 0.5], [0.5, 0]] * u.nS  # coupling conductances
           >>> V1, V2 = neuron1.V.value, neuron2.V.value
           >>> I_gap1 = G[0,1] * (V2 - V1)
           >>> I_gap2 = G[1,0] * (V1 - V2)
           >>> spk1 = neuron1.update(x=I_gap1)
           >>> spk2 = neuron2.update(x=I_gap2)

        **Alternative Input Mechanism:**

        Instead of passing gap-junction current via ``x``, you can register it
        as a named current input:

        .. code-block:: python

           >>> neuron.add_current_input('gap', lambda: I_gap)
           >>> spk = neuron.update()  # gap current applied automatically

        Warnings
        --------
        - Do not call ``update()`` before ``init_state()`` or
          ``init_all_states()`` --- state variables must be initialized first
        - Ensure time step ``dt`` is sufficiently small (typically <= 0.1 ms)
          for accurate spike detection and alpha-function dynamics

        See Also
        --------
        init_state : Initialize neuron state variables.
        get_spike : Compute spike output from membrane potential.
        sum_delta_inputs : Collect all registered delta-function inputs.
        sum_current_inputs : Collect all registered current inputs.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        m_ = self.m.value  # unitless
        h_ = self.h.value  # unitless
        n_ = self.n.value  # unitless
        p_ = self.p.value  # unitless
        dI_ex = self.dI_syn_ex.value  # pA/ms
        I_ex = self.I_syn_ex.value  # pA
        dI_in = self.dI_syn_in.value  # pA/ms
        I_in = self.I_syn_in.value  # pA
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h_step = self.integration_step.value  # ms
        V_old = self.V_old.value  # mV

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(
            V=V, m=m_, h=h_, n=n_, p=p_,
            dI_ex=dI_ex, I_ex=I_ex, dI_in=dI_in, I_in=I_in
        )
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            V_old=V_old,
        )

        ode_state, h_step, extra = self.integrator(state=ode_state, h=h_step, extra=extra)
        V = ode_state.V
        m_ = ode_state.m
        h_ = ode_state.h
        n_ = ode_state.n
        p_ = ode_state.p
        dI_ex = ode_state.dI_ex
        I_ex = ode_state.I_ex
        dI_in = ode_state.dI_in
        I_in = ode_state.I_in
        spike_mask = extra.spike_mask
        r = extra.r
        unstable = extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in hh_psc_alpha_gap dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration).
        w_all = self.sum_delta_inputs(0.0 * u.pA)
        w_ex = u.math.where(w_all > 0.0 * u.pA, w_all, 0.0 * u.pA)
        w_in = u.math.where(w_all < 0.0 * u.pA, w_all, 0.0 * u.pA)
        pscon_ex = np.e / self.tau_syn_ex  # 1/ms
        pscon_in = np.e / self.tau_syn_in  # 1/ms

        # Apply synaptic spike inputs.
        dI_ex = dI_ex + pscon_ex * w_ex  # pA/ms + 1/ms * pA = pA/ms
        dI_in = dI_in + pscon_in * w_in  # pA/ms + 1/ms * pA = pA/ms

        # Write back state.
        self.V.value = V
        self.m.value = m_
        self.h.value = h_
        self.n.value = n_
        self.p.value = p_
        self.I_syn_ex.value = I_ex
        self.I_syn_in.value = I_in
        self.dI_syn_ex.value = dI_ex
        self.dI_syn_in.value = dI_in
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h_step
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)
        self.V_old.value = V

        # Return spike output: only signal a spike when spike_mask is True
        V_out = u.math.where(spike_mask, 1e-12 * u.mV, -1.0 * u.mV)
        return self.get_spike(V_out)
