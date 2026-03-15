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

from typing import Callable, Optional

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
    'pp_cond_exp_mc_urbanczik',
]

# Compartment indices
SOMA = 0
DEND = 1
NCOMP = 2


class pp_cond_exp_mc_urbanczik(NESTNeuron):
    r"""Two-compartment point process neuron with conductance-based synapses for Urbanczik-Senn learning.

    ``pp_cond_exp_mc_urbanczik`` implements a two-compartment spiking neuron model
    that combines stochastic point process spike generation with dendritic prediction
    error computation for supervised learning. The soma uses conductance-based synapses
    while the dendrite uses current-based synapses. At each time step, the model
    computes a learning signal (δΠ) based on the mismatch between actual somatic
    spiking and the dendritic prediction, enabling gradient-based synaptic plasticity.

    This is a brainpy.state re-implementation of the NEST simulator model described
    in Urbanczik & Senn (2014) [1]_, using NEST-standard parameterization and
    numerical integration methods.

    Parameters
    ----------
    in_size : Size
        Population shape (tuple of ints or single int). Determines neuron array
        dimensions. Required parameter.
    t_ref : ArrayLike, optional
        Refractory period duration (Quantity, default: 3.0 ms). Neurons cannot
        spike again within this interval after a spike. If 0, no refractory period
        and Poisson spike generation is used.
    phi_max : float, optional
        Maximum firing rate in kHz (dimensionless, default: 0.15). Upper bound
        of the rate function φ(u). Typical range: 0.1-0.2 kHz (100-200 Hz).
    rate_slope : float, optional
        Rate function slope parameter ``k`` (dimensionless, default: 0.5). Controls
        the steepness of the sigmoid rate function. Must be non-negative.
    beta : float, optional
        Rate function steepness in 1/mV (dimensionless, default: 1/3 ≈ 0.333).
        Higher values create sharper transitions around threshold ``theta``.
    theta : float, optional
        Rate function threshold potential in mV (numeric, default: -55.0). Membrane
        potential at which firing rate is approximately half-maximal.
    g_sp : ArrayLike, optional
        Soma-to-dendrite coupling conductance (Quantity, default: 600.0 nS). Forward
        coupling from dendrite voltage to soma dynamics. Typically dominant coupling.
    g_ps : ArrayLike, optional
        Dendrite-to-soma coupling conductance (Quantity, default: 0.0 nS). Backward
        coupling from soma voltage to dendritic dynamics. Usually zero in this model.
    soma_g_L : ArrayLike, optional
        Somatic leak conductance (Quantity, default: 30.0 nS). Controls somatic
        resting potential and membrane time constant.
    soma_C_m : ArrayLike, optional
        Somatic membrane capacitance (Quantity, default: 300.0 pF). Together with
        leak conductance determines somatic time constant τ = C_m / g_L.
    soma_E_L : ArrayLike, optional
        Somatic leak reversal potential (Quantity, default: -70.0 mV). Resting
        potential of the soma in absence of inputs.
    soma_E_ex : ArrayLike, optional
        Somatic excitatory reversal potential (Quantity, default: 0.0 mV). Driving
        force for excitatory conductance-based synapses.
    soma_E_in : ArrayLike, optional
        Somatic inhibitory reversal potential (Quantity, default: -75.0 mV). Driving
        force for inhibitory conductance-based synapses.
    soma_tau_syn_ex : ArrayLike, optional
        Somatic excitatory synaptic time constant (Quantity, default: 3.0 ms). Decay
        time constant for excitatory conductance.
    soma_tau_syn_in : ArrayLike, optional
        Somatic inhibitory synaptic time constant (Quantity, default: 3.0 ms). Decay
        time constant for inhibitory conductance.
    soma_I_e : ArrayLike, optional
        Somatic constant external current (Quantity, default: 0.0 pA). DC bias
        current applied to soma at all times.
    dend_g_L : ArrayLike, optional
        Dendritic leak conductance (Quantity, default: 30.0 nS). Controls dendritic
        resting potential and membrane time constant.
    dend_C_m : ArrayLike, optional
        Dendritic membrane capacitance (Quantity, default: 300.0 pF). Together with
        leak conductance determines dendritic time constant τ = C_m / g_L.
    dend_E_L : ArrayLike, optional
        Dendritic leak reversal potential (Quantity, default: -70.0 mV). Resting
        potential of the dendrite in absence of inputs.
    dend_E_ex : ArrayLike, optional
        Dendritic excitatory reversal potential (Quantity, default: 0.0 mV). Used
        for documentation; current-based synapses don't use reversal potentials.
    dend_E_in : ArrayLike, optional
        Dendritic inhibitory reversal potential (Quantity, default: 0.0 mV, note:
        NOT -75.0 mV). Matches NEST default. Used for documentation only.
    dend_tau_syn_ex : ArrayLike, optional
        Dendritic excitatory synaptic time constant (Quantity, default: 3.0 ms).
        Decay time constant for excitatory current.
    dend_tau_syn_in : ArrayLike, optional
        Dendritic inhibitory synaptic time constant (Quantity, default: 3.0 ms).
        Decay time constant for inhibitory current.
    dend_I_e : ArrayLike, optional
        Dendritic constant external current (Quantity, default: 0.0 pA). DC bias
        current applied to dendrite at all times.
    gsl_error_tol : ArrayLike, optional
        Unitless local RKF45 error tolerance (default: 1e-3). Must be strictly
        positive.
    rng_key : jax.Array, optional
        JAX PRNG key for stochastic spike generation (default: None). If None, a
        default key (PRNGKey(0)) is used. For reproducibility, provide explicit key.
    spk_fun : Callable, optional
        Surrogate gradient function for spike differentiation (default:
        braintools.surrogate.ReluGrad()). Used in backpropagation through spikes.
    spk_reset : str, optional
        Spike reset mode (default: 'hard'). Options: 'hard' (stop_gradient) or
        'soft' (subtract threshold). Note: This model has NO voltage reset after
        spikes, so this parameter has limited effect.
    name : str, optional
        Module name (default: None). Used for logging and identification.

    Parameter Mapping
    -----------------
    This table maps NEST C++ parameter names to brainpy.state constructor arguments:

    ================================ =========================== ===============
    **NEST Parameter**               **brainpy.state Parameter** **Default**
    ================================ =========================== ===============
    ``t_ref``                        ``t_ref``                   3.0 ms
    ``phi_max``                      ``phi_max``                 0.15 kHz
    ``rate_slope``                   ``rate_slope``              0.5
    ``beta``                         ``beta``                    0.333 (1/mV)
    ``theta``                        ``theta``                   -55.0 mV
    ``g_sp``                         ``g_sp``                    600.0 nS
    ``g_ps``                         ``g_ps``                    0.0 nS
    ``g_L`` (soma)                   ``soma_g_L``                30.0 nS
    ``C_m`` (soma)                   ``soma_C_m``                300.0 pF
    ``E_L`` (soma)                   ``soma_E_L``                -70.0 mV
    ``E_ex`` (soma)                  ``soma_E_ex``               0.0 mV
    ``E_in`` (soma)                  ``soma_E_in``               -75.0 mV
    ``tau_syn_ex`` (soma)            ``soma_tau_syn_ex``         3.0 ms
    ``tau_syn_in`` (soma)            ``soma_tau_syn_in``         3.0 ms
    ``I_e`` (soma)                   ``soma_I_e``                0.0 pA
    ``g_L`` (dendrite)               ``dend_g_L``                30.0 nS
    ``C_m`` (dendrite)               ``dend_C_m``                300.0 pF
    ``E_L`` (dendrite)               ``dend_E_L``                -70.0 mV
    ``E_ex`` (dendrite)              ``dend_E_ex``               0.0 mV
    ``E_in`` (dendrite)              ``dend_E_in``               0.0 mV
    ``tau_syn_ex`` (dendrite)        ``dend_tau_syn_ex``         3.0 ms
    ``tau_syn_in`` (dendrite)        ``dend_tau_syn_in``         3.0 ms
    ``I_e`` (dendrite)               ``dend_I_e``                0.0 pA
    ================================ =========================== ===============

    Mathematical Formulation
    ------------------------
    **1. Compartment Structure**

    The neuron consists of two compartments:

    * **Soma (s):** Conductance-based synapses, stochastic spike generation
    * **Dendrite (d, also labeled p for "proximal"):** Current-based synapses,
      predictive signal for learning

    **2. Somatic Dynamics**

    The somatic membrane potential evolves according to:

    .. math::

        C_\mathrm{m}^s \frac{dV^s}{dt} = -g_\mathrm{L}^s (V^s - E_\mathrm{L}^s)
            - g_\mathrm{ex}^s (V^s - E_\mathrm{ex}^s)
            - g_\mathrm{in}^s (V^s - E_\mathrm{in}^s)
            + g_\mathrm{sp} (V^p - V^s)
            + I_\mathrm{stim}^s + I_\mathrm{e}^s

    Somatic synaptic conductances decay exponentially:

    .. math::

        \frac{dg_\mathrm{ex}^s}{dt} = -\frac{g_\mathrm{ex}^s}{\tau_\mathrm{syn,ex}^s},
        \qquad
        \frac{dg_\mathrm{in}^s}{dt} = -\frac{g_\mathrm{in}^s}{\tau_\mathrm{syn,in}^s}

    **3. Dendritic Dynamics**

    The dendritic membrane potential evolves according to:

    .. math::

        C_\mathrm{m}^p \frac{dV^p}{dt} = -g_\mathrm{L}^p (V^p - E_\mathrm{L}^p)
            + I_\mathrm{ex}^p + I_\mathrm{in}^p
            + g_\mathrm{ps} (V^s - V^p)

    Dendritic synaptic currents (note: **current-based**, not conductance) decay
    exponentially:

    .. math::

        \frac{dI_\mathrm{ex}^p}{dt} = -\frac{I_\mathrm{ex}^p}{\tau_\mathrm{syn,ex}^p},
        \qquad
        \frac{dI_\mathrm{in}^p}{dt} = -\frac{I_\mathrm{in}^p}{\tau_\mathrm{syn,in}^p}

    **4. Stochastic Spike Generation**

    Spikes are generated stochastically based on the instantaneous rate function:

    .. math::

        \text{rate}(t) = 1000 \cdot \phi(V^s(t)) \quad [\text{Hz}]

    where:

    .. math::

        \phi(u) = \frac{\phi_\mathrm{max}}{1 + k \cdot \exp(\beta (\theta - u))}

    * **With refractory period** (``t_ref > 0``): At most one spike per time step.
      Spike probability is :math:`P_{\mathrm{spike}} = 1 - \exp(-\text{rate} \cdot dt \cdot 10^{-3})`.
      A uniform random number :math:`r \sim U(0,1)` is compared to this probability.
    * **Without refractory period** (``t_ref == 0``): Number of spikes drawn from
      Poisson distribution with mean :math:`\lambda = \text{rate} \cdot dt \cdot 10^{-3}`.

    **Important:** There is **NO membrane potential reset** after a spike. The voltage
    continues to evolve according to the differential equations.

    **5. Urbanczik-Senn Learning Signal**

    At each time step, the model computes a learning signal for synaptic plasticity.
    The dendritic compartment predicts the somatic potential via:

    .. math::

        V^*_W = \frac{E_\mathrm{L}^s \cdot g_\mathrm{L}^s + V^p \cdot g_\mathrm{sp}}{g_\mathrm{sp} + g_\mathrm{L}^s}

    This represents the steady-state somatic voltage given the current dendritic
    voltage, assuming all synaptic inputs are zero.

    The error signal (prediction error) at time step :math:`t` is:

    .. math::

        \delta\Pi(t) = \left(n_\mathrm{spikes}(t) - \phi(V^*_W(t)) \cdot dt\right) \cdot h(V^*_W(t))

    where:

    * :math:`n_{\mathrm{spikes}}(t)` is the number of actual spikes emitted (0 or 1
      with refractory period, ≥0 without)
    * :math:`\phi(V^*_W(t)) \cdot dt` is the expected spike count based on prediction
    * :math:`h(u)` is the learning modulation function:

    .. math::

        h(u) = \frac{15 \cdot \beta}{1 + \frac{1}{k} \cdot \exp(-\beta (\theta - u))}

    The history of :math:`(t, \delta\Pi)` pairs is stored and accessible via
    ``get_urbanczik_history()`` for use by plasticity rules.

    **6. Receptor Types and Synaptic Input Addressing**

    Synaptic inputs are routed to specific compartments and receptor types via
    labeled input channels:

    =================== ====== ============================================
    Receptor Label       Port   Description
    =================== ====== ============================================
    ``soma_exc``         1      Excitatory conductance input to soma (nS)
    ``soma_inh``         2      Inhibitory conductance input to soma (nS)
    ``dend_exc``         3      Excitatory current input to dendrite (pA)
    ``dend_inh``         4      Inhibitory current input to dendrite (pA)
    ``soma`` (current)   5      Direct current injection to soma (pA)
    ``dend`` (current)   6      Direct current injection to dendrite (pA)
    =================== ====== ============================================

    **Implementation Note:** In brainpy.state, use ``add_delta_input()`` with labels
    ``'soma_exc'``, ``'soma_inh'``, ``'dend_exc'``, ``'dend_inh'`` for synaptic
    spikes. Use ``add_current_input()`` with labels ``'soma'`` and ``'dend'`` for
    current injections. All synaptic weights must be **positive**; excitation vs.
    inhibition is determined by the receptor label.

    **7. Numerical Integration**

    The 6-dimensional ODE system (V_s, g_ex_s, g_in_s, V_d, I_ex_d, I_in_d) is
    integrated using an adaptive RKF45 Runge-Kutta-Fehlberg integrator that is
    fully JAX-compatible and differentiable.

    **Update Order per Time Step:**

    1. Integrate ODEs over interval :math:`(t, t + dt]` using current stimulus
       currents from previous step
    2. Add arriving synaptic spike inputs (conductance/current jumps):

       * Soma: :math:`g_{\mathrm{ex}}^s \mathrel{+}= \Delta g_{\mathrm{ex}}`,
         :math:`g_{\mathrm{in}}^s \mathrel{+}= \Delta g_{\mathrm{in}}`
       * Dendrite: :math:`I_{\mathrm{ex}}^p \mathrel{+}= \Delta I_{\mathrm{ex}}`,
         :math:`I_{\mathrm{in}}^p \mathrel{-}= \Delta I_{\mathrm{in}}` (note sign)

    3. Check refractoriness and generate spikes stochastically if not refractory
    4. Compute and store Urbanczik learning signal :math:`\delta\Pi`
    5. Buffer external current inputs for next time step

    Computational Complexity and Performance
    ----------------------------------------
    **Time Complexity:** :math:`O(N \cdot S)` where :math:`N` is the number of neurons
    and :math:`S` is the number of adaptive ODE solver steps per neuron per time step.
    Typically :math:`S \approx 3-10` depending on dynamics.

    **Space Complexity:** :math:`O(N)` for state variables, plus :math:`O(N \cdot T)`
    for Urbanczik history over :math:`T` time steps.

    **Performance Notes:**

    * This model is **significantly slower** than simple LIF neurons due to:
      (1) element-wise adaptive ODE solving per neuron, (2) stochastic spike
      generation requiring RNG calls, and (3) learning signal computation.
    * Not vectorized across neurons; uses Python loop over ``np.ndindex``.
    * For large networks (>1000 neurons), consider alternative implementations or
      simplified models.
    * History storage grows linearly with simulation time; clear periodically if
      memory is constrained.

    Attributes (State Variables)
    -----------------------------
    V_s : brainstate.HiddenState
        Somatic membrane potential (Quantity, shape: ``varshape``).
        Initialized to ``soma_E_L``. Unit: mV.
    g_ex_s : brainstate.HiddenState
        Somatic excitatory synaptic conductance (Quantity). Initialized to 0. Unit: nS.
    g_in_s : brainstate.HiddenState
        Somatic inhibitory synaptic conductance (Quantity). Initialized to 0. Unit: nS.
    V_d : brainstate.HiddenState
        Dendritic membrane potential (Quantity). Initialized to ``dend_E_L``. Unit: mV.
    I_ex_d : brainstate.HiddenState
        Dendritic excitatory synaptic current (Quantity). Initialized to 0. Unit: pA.
    I_in_d : brainstate.HiddenState
        Dendritic inhibitory synaptic current (Quantity). Initialized to 0. Unit: pA.
    refractory_step_count : brainstate.ShortTermState
        Remaining refractory time steps (int32 array). Counts down to zero. Initialized to 0.
    I_stim_soma : brainstate.ShortTermState
        Buffered soma current for next integration step (Quantity). Unit: pA.
    I_stim_dend : brainstate.ShortTermState
        Buffered dendrite current for next integration step (Quantity). Unit: pA.
    last_spike_time : brainstate.ShortTermState
        Time of last spike emission (Quantity). Initialized to -1e7 ms. Unit: ms.
    integration_step : brainstate.ShortTermState
        Persistent RKF45 substep size estimate (ms).

    Raises
    ------
    ValueError
        If ``rate_slope < 0`` (must be non-negative).
    ValueError
        If ``phi_max < 0`` (must be non-negative).
    ValueError
        If ``t_ref < 0`` (must be non-negative).
    ValueError
        If any capacitance ``C_m`` ≤ 0 (must be strictly positive).
    ValueError
        If any synaptic time constant ≤ 0 (must be strictly positive).
    ValueError
        If ``gsl_error_tol`` ≤ 0 (must be strictly positive).

    Notes
    -----
    * **NEST Compatibility:** All default parameter values match NEST 3.9+ C++ source
      for ``pp_cond_exp_mc_urbanczik``. Notable: dendritic inhibitory reversal
      potential is 0.0 mV (not -75.0 mV).
    * **Stochasticity:** Spike times are non-deterministic unless ``rng_key`` is
      explicitly managed. For reproducibility, provide a fixed PRNG key and re-seed
      appropriately.
    * **No Voltage Reset:** Unlike integrate-and-fire models, there is no discrete
      voltage reset after spiking. The membrane potential evolves continuously.
    * **Urbanczik History:** The learning signal history is stored in a Python dict
      (``_urbanczik_history``) and grows unbounded. For long simulations, periodically
      clear history or implement custom storage.
    * **Surrogate Gradients:** The ``spk_fun`` parameter enables gradient-based
      learning through spike discontinuities, but this model is primarily designed
      for the Urbanczik-Senn rule which uses the stored δΠ signals directly.

    Examples
    --------
    **Basic single neuron simulation:**

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import saiunit as u
        >>> import brainstate
        >>> import numpy as np
        >>> # Create a single neuron
        >>> neuron = bp.pp_cond_exp_mc_urbanczik(in_size=1)
        >>> neuron.init_all_states()
        >>> # Simulate for 100 ms with constant soma current
        >>> dt = 0.1 * u.ms
        >>> with brainstate.environ.context(dt=dt):
        ...     spikes = []
        ...     for i in range(1000):  # 100 ms
        ...         spk = neuron.update(x=300.0 * u.pA)  # Strong depolarizing current
        ...         spikes.append(float(spk[0]))
        >>> print(f"Total spikes: {sum(spikes)}")
        Total spikes: 12

    **Two-compartment voltage monitoring:**

    .. code-block:: python

        >>> neuron = bp.pp_cond_exp_mc_urbanczik(in_size=1, t_ref=0.0*u.ms)
        >>> neuron.init_all_states()
        >>> soma_voltages, dend_voltages = [], []
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     for i in range(500):
        ...         neuron.update(x=200.0 * u.pA)
        ...         soma_voltages.append(float(neuron.V_s.value[0] / u.mV))
        ...         dend_voltages.append(float(neuron.V_d.value[0] / u.mV))
        >>> # Plot soma_voltages and dend_voltages to visualize dynamics

    **Accessing Urbanczik learning signals:**

    .. code-block:: python

        >>> neuron = bp.pp_cond_exp_mc_urbanczik(in_size=2)
        >>> neuron.init_all_states()
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     for i in range(100):
        ...         neuron.update(x=250.0 * u.pA)
        >>> # Retrieve learning signal history for neuron 0
        >>> history_0 = neuron.get_urbanczik_history(neuron_idx=0)
        >>> print(f"History length: {len(history_0)}")
        History length: 100
        >>> # Each entry is (time_ms, delta_PI)
        >>> t, dPI = history_0[-1]
        >>> print(f"Last time: {t:.2f} ms, Last dPI: {dPI:.4f}")
        Last time: 10.00 ms, Last dPI: -0.0234

    References
    ----------
    .. [1] Urbanczik R, Senn W (2014). Learning by the Dendritic Prediction of
           Somatic Spiking. Neuron, 81(3):521-528.
           DOI: https://doi.org/10.1016/j.neuron.2013.11.030

    .. [2] NEST Simulator ``pp_cond_exp_mc_urbanczik`` model documentation:
           https://nest-simulator.readthedocs.io/en/stable/models/pp_cond_exp_mc_urbanczik.html

    .. [3] NEST C++ source code: ``models/pp_cond_exp_mc_urbanczik.h`` and
           ``models/pp_cond_exp_mc_urbanczik.cpp`` in NEST 3.9+ distribution.

    See Also
    --------
    gif_cond_exp : Generalized integrate-and-fire with conductance synapses
    pp_psc_delta : Point process neuron with current synapses
    urbanczik_synapse : Synapse model implementing Urbanczik-Senn plasticity rule
    """
    __module__ = 'brainpy.state'

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        # Global parameters
        t_ref: ArrayLike = 3.0 * u.ms,
        phi_max: float = 0.15,  # kHz
        rate_slope: float = 0.5,  # dimensionless
        beta: float = 1.0 / 3.0,  # 1/mV
        theta: float = -55.0,  # mV
        g_sp: ArrayLike = 600.0 * u.nS,  # soma-dendrite coupling
        g_ps: ArrayLike = 0.0 * u.nS,  # dendrite-soma coupling
        # Soma compartment parameters
        soma_g_L: ArrayLike = 30.0 * u.nS,
        soma_C_m: ArrayLike = 300.0 * u.pF,
        soma_E_L: ArrayLike = -70.0 * u.mV,
        soma_E_ex: ArrayLike = 0.0 * u.mV,
        soma_E_in: ArrayLike = -75.0 * u.mV,
        soma_tau_syn_ex: ArrayLike = 3.0 * u.ms,
        soma_tau_syn_in: ArrayLike = 3.0 * u.ms,
        soma_I_e: ArrayLike = 0.0 * u.pA,
        # Dendritic compartment parameters
        dend_g_L: ArrayLike = 30.0 * u.nS,
        dend_C_m: ArrayLike = 300.0 * u.pF,
        dend_E_L: ArrayLike = -70.0 * u.mV,
        dend_E_ex: ArrayLike = 0.0 * u.mV,
        dend_E_in: ArrayLike = 0.0 * u.mV,
        dend_tau_syn_ex: ArrayLike = 3.0 * u.ms,
        dend_tau_syn_in: ArrayLike = 3.0 * u.ms,
        dend_I_e: ArrayLike = 0.0 * u.pA,
        # Integration tolerance
        gsl_error_tol: ArrayLike = 1e-3,
        # RNG and surrogate
        rng_key: Optional[jax.Array] = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Global parameters
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.phi_max = float(phi_max)
        self.rate_slope = float(rate_slope)
        self.beta = float(beta)
        self.theta = float(theta)
        self.g_sp = braintools.init.param(g_sp, self.varshape)
        self.g_ps = braintools.init.param(g_ps, self.varshape)

        # Soma parameters
        self.soma_g_L = braintools.init.param(soma_g_L, self.varshape)
        self.soma_C_m = braintools.init.param(soma_C_m, self.varshape)
        self.soma_E_L = braintools.init.param(soma_E_L, self.varshape)
        self.soma_E_ex = braintools.init.param(soma_E_ex, self.varshape)
        self.soma_E_in = braintools.init.param(soma_E_in, self.varshape)
        self.soma_tau_syn_ex = braintools.init.param(soma_tau_syn_ex, self.varshape)
        self.soma_tau_syn_in = braintools.init.param(soma_tau_syn_in, self.varshape)
        self.soma_I_e = braintools.init.param(soma_I_e, self.varshape)

        # Dendritic parameters
        self.dend_g_L = braintools.init.param(dend_g_L, self.varshape)
        self.dend_C_m = braintools.init.param(dend_C_m, self.varshape)
        self.dend_E_L = braintools.init.param(dend_E_L, self.varshape)
        self.dend_E_ex = braintools.init.param(dend_E_ex, self.varshape)
        self.dend_E_in = braintools.init.param(dend_E_in, self.varshape)
        self.dend_tau_syn_ex = braintools.init.param(dend_tau_syn_ex, self.varshape)
        self.dend_tau_syn_in = braintools.init.param(dend_tau_syn_in, self.varshape)
        self.dend_I_e = braintools.init.param(dend_I_e, self.varshape)

        # Integration tolerance
        self.gsl_error_tol = gsl_error_tol

        # RNG
        self._rng_key = rng_key

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

        # Precompute refractory step count
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
        if any(is_tracer(v) for v in (self.soma_C_m, self.t_ref)):
            return

        if self.rate_slope < 0:
            raise ValueError('Rate slope cannot be negative.')
        if self.phi_max < 0:
            raise ValueError('Maximum rate cannot be negative.')
        if np.any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        for label, C_m in [('soma', self.soma_C_m), ('dendritic', self.dend_C_m)]:
            if np.any(C_m <= 0.0 * u.pF):
                raise ValueError(f'Capacitance ({label}) must be strictly positive.')
        for label, tse, tsi in [
            ('soma', self.soma_tau_syn_ex, self.soma_tau_syn_in),
            ('dendritic', self.dend_tau_syn_ex, self.dend_tau_syn_in),
        ]:
            if np.any(tse <= 0.0 * u.ms) or np.any(tsi <= 0.0 * u.ms):
                raise ValueError('All time constants must be strictly positive.')
        if np.any(self.gsl_error_tol <= 0.0):
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

        # Membrane potentials initialized to E_L
        self.V_s = brainstate.HiddenState(
            u.math.ones(self.varshape, dtype=dftype) * self.soma_E_L
        )
        self.V_d = brainstate.HiddenState(
            u.math.ones(self.varshape, dtype=dftype) * self.dend_E_L
        )

        # Somatic conductances
        self.g_ex_s = brainstate.HiddenState(
            u.math.zeros(self.varshape, dtype=dftype) * u.nS
        )
        self.g_in_s = brainstate.HiddenState(
            u.math.zeros(self.varshape, dtype=dftype) * u.nS
        )

        # Dendritic currents
        self.I_ex_d = brainstate.HiddenState(
            u.math.zeros(self.varshape, dtype=dftype) * u.pA
        )
        self.I_in_d = brainstate.HiddenState(
            u.math.zeros(self.varshape, dtype=dftype) * u.pA
        )

        # Refractory counter
        self.refractory_step_count = brainstate.ShortTermState(
            u.math.full(self.varshape, 0, dtype=ditype)
        )

        # Buffered stimulus currents (per compartment)
        self.I_stim_soma = brainstate.ShortTermState(
            u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype)
        )
        self.I_stim_dend = brainstate.ShortTermState(
            u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype)
        )

        # Last spike time
        self.last_spike_time = brainstate.ShortTermState(
            u.math.full(self.varshape, -1e7 * u.ms)
        )

        # Integration step size
        self.integration_step = brainstate.ShortTermState.init(
            braintools.init.Constant(dt), self.varshape
        )

        # Urbanczik history: list of (t_ms, dPI) tuples per neuron element
        self._urbanczik_history = {}

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def get_spike(self, V: ArrayLike = None):
        r"""Evaluate surrogate spike output from membrane voltage.

        Parameters
        ----------
        V : ArrayLike, optional
            Voltage values with shape broadcastable to ``self.varshape`` and
            units compatible with mV. If ``None``, uses current state
            ``self.V_s.value``.

        Returns
        -------
        ArrayLike
            Surrogate spike activation produced by
            ``spk_fun(V / (1.0 * u.mV))``.
        """
        V = self.V_s.value if V is None else V
        v_scaled = V / (1.0 * u.mV)
        return self.spk_fun(v_scaled)

    def _collect_receptor_delta_inputs(self):
        r"""Collect delta inputs labeled by receptor type.

        Expected labels: 'soma_exc', 'soma_inh', 'dend_exc', 'dend_inh'.

        Returns
        -------
        soma_exc, soma_inh, dend_exc, dend_inh : Quantity arrays (nS or pA)
        """
        v_shape = self.V_s.value.shape

        soma_exc = u.math.zeros(v_shape) * u.nS
        soma_inh = u.math.zeros(v_shape) * u.nS
        dend_exc = u.math.zeros(v_shape) * u.pA
        dend_inh = u.math.zeros(v_shape) * u.pA

        if self.delta_inputs is None:
            return soma_exc, soma_inh, dend_exc, dend_inh

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            label = key if isinstance(key, str) else str(key)

            if 'soma_exc' in label:
                soma_exc = soma_exc + out
            elif 'soma_inh' in label:
                soma_inh = soma_inh + out
            elif 'dend_exc' in label:
                dend_exc = dend_exc + out
            elif 'dend_inh' in label:
                dend_inh = dend_inh + out

        return soma_exc, soma_inh, dend_exc, dend_inh

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V_s, g_ex_s, g_in_s, V_d, I_ex_d, I_in_d — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, i_stim_soma — mutable auxiliary data carried
            through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        # Soma dynamics
        V_s = state.V_s
        V_d = state.V_d

        # Soma leak current
        I_L_s = self.soma_g_L * (V_s - self.soma_E_L)

        # Soma excitatory synaptic current (conductance-based)
        I_syn_exc = state.g_ex_s * (V_s - self.soma_E_ex)

        # Soma inhibitory synaptic current (conductance-based)
        I_syn_inh = state.g_in_s * (V_s - self.soma_E_in)

        # Coupling current dendrite -> soma
        I_conn_d_s = self.g_sp * (V_d - V_s)

        # Coupling current soma -> dendrite
        I_conn_s_d = self.g_ps * (V_s - V_d)

        # Soma membrane potential derivative
        dV_s = (
            -I_L_s - I_syn_exc - I_syn_inh + I_conn_d_s
            + extra.i_stim_soma + self.soma_I_e
        ) / self.soma_C_m

        # Soma conductance derivatives
        dg_ex_s = -state.g_ex_s / self.soma_tau_syn_ex
        dg_in_s = -state.g_in_s / self.soma_tau_syn_in

        # Dendrite membrane potential derivative
        dV_d = (
            -self.dend_g_L * (V_d - self.dend_E_L)
            + state.I_ex_d + state.I_in_d + I_conn_s_d
        ) / self.dend_C_m

        # Dendrite current derivatives
        dI_ex_d = -state.I_ex_d / self.dend_tau_syn_ex
        dI_in_d = -state.I_in_d / self.dend_tau_syn_in

        return DotDict(
            V_s=dV_s, g_ex_s=dg_ex_s, g_in_s=dg_in_s,
            V_d=dV_d, I_ex_d=dI_ex_d, I_in_d=dI_in_d,
        )

    def _event_fn(self, state, extra, accept):
        """In-loop event callback for the adaptive integrator.

        This model does not perform spike detection or voltage reset inside the
        integration loop (spikes are stochastic and generated after integration).
        The event function is a no-op pass-through.

        Parameters
        ----------
        state : DotDict
            Keys: V_s, g_ex_s, g_in_s, V_d, I_ex_d, I_in_d — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, i_stim_soma.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts unchanged.
        """
        return state, extra

    def update(self, x=0.0 * u.pA):
        r"""Advance neuron state by one simulation time step with ODE integration and stochastic spiking.

        This method performs a complete update cycle for the two-compartment model,
        including numerical integration of differential equations, synaptic input
        processing, stochastic spike generation, and Urbanczik learning signal
        computation. It follows NEST's update order exactly.

        Parameters
        ----------
        x : Quantity, optional
            External current input applied to the soma compartment (unit: pA, default: 0.0 pA).
            This is typically used for direct current injection from external sources.
            Shape must be broadcastable to neuron population shape.

        Returns
        -------
        spike : jax.numpy.ndarray
            Binary spike output array of shape matching neuron population (float32).
            Values are 1.0 where a spike was emitted, 0.0 otherwise. For neurons
            without refractory period (t_ref=0), values can be >1 if multiple
            spikes occurred in one step.

        Notes
        -----

        Update Procedure
        ----------------

        The method executes the following steps in order (per neuron):

        **1. ODE Integration**

        Integrate the 6-dimensional state vector over the interval (t, t+dt] using
        the adaptive RKF45 solver. The integration uses stimulus currents buffered
        from the previous time step.

        **2. Synaptic Input Application**

        Apply instantaneous jumps to synaptic variables from arriving spikes:

        * Soma: g_ex_s += Δg_ex, g_in_s += Δg_in (conductance jumps in nS)
        * Dendrite: I_ex_d += ΔI_ex, I_in_d -= ΔI_in (current jumps in pA; note sign)

        **3. Stochastic Spike Generation**

        If neuron is not refractory:

        * Compute instantaneous rate: rate = 1000 · φ(V_s) [Hz]
        * With t_ref > 0: Draw uniform random r, emit spike if r ≤ 1 - exp(-rate·dt·1e-3)
        * With t_ref = 0: Draw Poisson(rate·dt·1e-3) for spike count
        * If spike(s) emitted: set refractory counter to round(t_ref / dt)

        If neuron is refractory: decrement refractory counter, no spikes.

        **4. Urbanczik Learning Signal**

        Compute dendritic prediction and error signal:

        * :math:`V^*_W = (E_{L,s} \cdot g_{L,s} + V_d \cdot g_{sp}) / (g_{sp} + g_{L,s})`
        * :math:`\delta\Pi = (n_{\text{spikes}} - \phi(V^*_W) \cdot dt) \cdot h(V^*_W)`
        * Store :math:`(t, \delta\Pi)` in history dict

        **5. Current Input Buffering**

        Collect all current inputs (via ``sum_current_inputs()``) and store for use
        in the next time step's ODE integration.

        Computational Complexity
        ------------------------

        * Time: O(N · S) where N is population size, S is adaptive ODE steps per neuron
        * Space: O(N) for state updates, O(N·T) for history accumulation over T steps
        * **Not vectorized:** Uses Python loop over all neuron indices

        Side Effects
        ------------

        * Updates all state variables (V_s, V_d, g_ex_s, g_in_s, I_ex_d, I_in_d)
        * Updates refractory counters and last_spike_time
        * Appends (t, δΠ) to internal ``_urbanczik_history`` dict
        * Advances internal PRNG state (``_rng_state``)
        * Consumes and clears delta_inputs from projections

        Numerical Considerations
        ------------------------

        * The ODE solver is adaptive and may take variable numbers of internal steps
        * For stiff dynamics or large coupling conductances, integration may require
          more steps, increasing computation time
        * Dendritic inhibitory current is **subtracted**, matching NEST
          convention for inhibitory synapses

        .. warning::

           Numerical issues (NaN, Inf) can arise from invalid parameter
           combinations (e.g., zero capacitance), extremely large input
           currents, or ODE solver failure.

        Notes
        -----
        * This is a **slow** model due to per-neuron ODE solving and lack of
          vectorization. For networks >1000 neurons, expect significant runtime.
        * The lack of voltage reset after spikes is intentional and matches the
          original Urbanczik & Senn (2014) formulation.
        * Random number generation state is advanced even if no spikes occur, ensuring
          reproducibility across different input patterns given the same seed.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        dt_ms = float(u.math.asarray(dt / u.ms))  # dt in ms

        # Read state variables with their natural units.
        V_s = self.V_s.value  # mV
        g_ex_s = self.g_ex_s.value  # nS
        g_in_s = self.g_in_s.value  # nS
        V_d = self.V_d.value  # mV
        I_ex_d = self.I_ex_d.value  # pA
        I_in_d = self.I_in_d.value  # pA
        r = self.refractory_step_count.value  # int
        i_stim_soma = self.I_stim_soma.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim_soma = self.sum_current_inputs(x, self.V_s.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(
            V_s=V_s, g_ex_s=g_ex_s, g_in_s=g_in_s,
            V_d=V_d, I_ex_d=I_ex_d, I_in_d=I_in_d,
        )
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            i_stim_soma=i_stim_soma,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V_s = ode_state.V_s
        g_ex_s = ode_state.g_ex_s
        g_in_s = ode_state.g_in_s
        V_d = ode_state.V_d
        I_ex_d = ode_state.I_ex_d
        I_in_d = ode_state.I_in_d
        r = extra.r

        # Collect synaptic spike inputs
        d_soma_exc, d_soma_inh, d_dend_exc, d_dend_inh = self._collect_receptor_delta_inputs()

        # Apply synaptic spike inputs (after integration).
        g_ex_s = g_ex_s + d_soma_exc
        g_in_s = g_in_s + d_soma_inh
        I_ex_d = I_ex_d + d_dend_exc
        I_in_d = I_in_d - d_dend_inh  # Note: inhibitory is subtracted (NEST convention)

        # --- Stochastic spike generation (per-element, Python loop) ---
        v_shape = self.varshape

        # Convert state to numpy for per-element spike generation
        V_s_np = np.asarray(u.math.asarray(V_s / u.mV), dtype=dftype)
        V_d_np = np.asarray(u.math.asarray(V_d / u.mV), dtype=dftype)
        r_np = np.asarray(u.get_mantissa(r), dtype=ditype)
        ref_counts_np = np.asarray(u.get_mantissa(self.ref_count), dtype=ditype)

        # Advance RNG
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(jax.random.uniform(subkey, shape=v_shape), dtype=dftype)

        spike_mask = np.zeros(v_shape, dtype=bool)

        # Compute step time for urbanczik history
        t_ms = float(u.math.asarray(t / u.ms)) + dt_ms

        # Precompute scalar parameters for phi/h functions
        phi_max = self.phi_max
        rate_slope = self.rate_slope
        beta = self.beta
        theta = self.theta

        # Precompute numpy arrays for per-element parameters
        g_sp_np = np.asarray(u.math.asarray(self.g_sp / u.nS), dtype=dftype)
        g_L_soma_np = np.asarray(u.math.asarray(self.soma_g_L / u.nS), dtype=dftype)
        E_L_soma_np = np.asarray(u.math.asarray(self.soma_E_L / u.mV), dtype=dftype)
        t_ref_np = np.asarray(u.math.asarray(self.t_ref / u.ms), dtype=dftype)

        # Broadcast to v_shape
        g_sp_np = np.broadcast_to(g_sp_np, v_shape)
        g_L_soma_np = np.broadcast_to(g_L_soma_np, v_shape)
        E_L_soma_np = np.broadcast_to(E_L_soma_np, v_shape)
        t_ref_np = np.broadcast_to(t_ref_np, v_shape)
        V_s_np = np.broadcast_to(V_s_np, v_shape).copy()
        V_d_np = np.broadcast_to(V_d_np, v_shape).copy()
        r_np = np.broadcast_to(r_np, v_shape).copy()
        ref_counts_np = np.broadcast_to(ref_counts_np, v_shape)

        for idx in np.ndindex(v_shape):
            n_spikes = 0

            if r_np[idx] == 0:
                # Neuron not refractory — no V_m reset after spike
                u_val = V_s_np[idx]
                rate = 1000.0 * phi_max / (1.0 + rate_slope * np.exp(beta * (theta - u_val)))

                if rate > 0.0:
                    t_ref_val = float(t_ref_np[idx])

                    if t_ref_val > 0.0:
                        # With dead time: at most 1 spike
                        if rand_vals[idx] <= -np.expm1(-rate * dt_ms * 1e-3):
                            n_spikes = 1
                    else:
                        # No dead time: Poisson spikes
                        lam = rate * dt_ms * 1e-3
                        n_spikes = int(np.random.RandomState(
                            int(rand_vals[idx] * 2 ** 31)
                        ).poisson(lam))

                    if n_spikes > 0:
                        spike_mask[idx] = True
                        r_np[idx] = ref_counts_np[idx]
            else:
                # Refractory: decrement
                r_np[idx] -= 1

            # --- Urbanczik learning signal ---
            V_d_current = V_d_np[idx]
            g_D = g_sp_np[idx]
            g_L_s = g_L_soma_np[idx]
            E_L_s = E_L_soma_np[idx]
            V_W_star = (E_L_s * g_L_s + V_d_current * g_D) / (g_D + g_L_s)

            phi_val = phi_max / (1.0 + rate_slope * np.exp(beta * (theta - V_W_star)))
            h_val = 15.0 * beta / (1.0 + (1.0 / rate_slope) * np.exp(-beta * (theta - V_W_star)))
            dPI = (n_spikes - phi_val * dt_ms) * h_val

            flat_idx = np.ravel_multi_index(idx, v_shape) if len(idx) > 0 else 0
            if flat_idx not in self._urbanczik_history:
                self._urbanczik_history[flat_idx] = []
            self._urbanczik_history[flat_idx].append((t_ms, dPI))

        # Write back state.
        self.V_s.value = V_s
        self.g_ex_s.value = g_ex_s
        self.g_in_s.value = g_in_s
        self.V_d.value = V_d
        self.I_ex_d.value = I_ex_d
        self.I_in_d.value = I_in_d
        self.refractory_step_count.value = jnp.asarray(r_np, dtype=ditype)
        self.integration_step.value = h
        self.I_stim_soma.value = new_i_stim_soma + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return jnp.asarray(spike_mask, dtype=dftype)

    def get_urbanczik_history(self, neuron_idx=0):
        r"""Retrieve the Urbanczik-Senn learning signal history for a specific neuron.

        This method returns the complete time series of error signals (δΠ) computed
        during simulation, which can be used to implement the Urbanczik-Senn synaptic
        plasticity rule. Each entry contains the simulation time and corresponding
        error signal value.

        Parameters
        ----------
        neuron_idx : int, optional
            Flat (raveled) index of the neuron within the population array (default: 0).
            For a 2D population of shape (N, M), valid indices are 0 to N*M-1.
            Use ``np.ravel_multi_index((i, j), shape)`` to convert multi-dimensional
            indices to flat index.

        Returns
        -------
        history : list of tuple
            List of (time_ms, delta_PI) tuples, where:

            * ``time_ms`` (float): Simulation time in milliseconds when the signal
              was computed. Times are strictly increasing.
            * ``delta_PI`` (float): Learning signal value (dimensionless). Positive
              values indicate the neuron spiked more than predicted (potentiation
              signal); negative values indicate under-spiking (depression signal).

            If the neuron index has not been encountered (no history), returns an
            empty list ``[]``.

        Mathematical Interpretation
        ---------------------------
        Each ``delta_PI`` value represents:

        .. math::

            \delta\Pi(t) = \left(n_{\mathrm{spikes}}(t) - \phi(V^*_W(t)) \cdot dt\right) \cdot h(V^*_W(t))

        where:

        * :math:`n_{\mathrm{spikes}}` is the actual spike count in the time step
        * :math:`\phi(V^*_W) \cdot dt` is the expected spike count from prediction
        * :math:`h(V^*_W)` is the voltage-dependent learning modulation

        **Usage in Plasticity:**

        The Urbanczik-Senn weight update rule for a synapse connecting to this neuron
        involves integrating these error signals with presynaptic activity traces.
        Typically, weights are updated as:

        .. math::

            \Delta w_i = \eta \sum_t \delta\Pi(t) \cdot x_i(t)

        where :math:`x_i(t)` is the presynaptic trace (e.g., filtered spike train).

        Examples
        --------
        **Retrieve and plot learning signals:**

        .. code-block:: python

            >>> import brainpy.state as bp
            >>> import saiunit as u
            >>> import brainstate
            >>> import matplotlib.pyplot as plt
            >>> neuron = bp.pp_cond_exp_mc_urbanczik(in_size=1)
            >>> neuron.init_all_states()
            >>> with brainstate.environ.context(dt=0.1*u.ms):
            ...     for i in range(1000):
            ...         neuron.update(x=250.0 * u.pA)
            >>> history = neuron.get_urbanczik_history(neuron_idx=0)
            >>> times, dPIs = zip(*history)
            >>> plt.plot(times, dPIs)
            >>> plt.xlabel('Time (ms)')
            >>> plt.ylabel('δΠ')
            >>> plt.title('Urbanczik Learning Signal')
            >>> plt.show()

        **Access for multi-dimensional population:**

        .. code-block:: python

            >>> import numpy as np
            >>> neuron_pop = bp.pp_cond_exp_mc_urbanczik(in_size=(10, 10))
            >>> neuron_pop.init_all_states()
            >>> # Simulate...
            >>> # Get history for neuron at position (3, 7)
            >>> flat_idx = np.ravel_multi_index((3, 7), (10, 10))
            >>> history_3_7 = neuron_pop.get_urbanczik_history(neuron_idx=flat_idx)

        **Check if history exists:**

        .. code-block:: python

            >>> history = neuron.get_urbanczik_history(neuron_idx=999)
            >>> if not history:
            ...     print("No history recorded for neuron 999")

        Notes
        -----
        * History is stored in memory and grows linearly with simulation length.
          For long simulations or large populations, consider periodic clearing.
        * History is reset by ``reset_state()`` but persists across ``update()`` calls.
        * The internal storage is a Python dict mapping flat indices to lists,
          which is not JAX-compatible but sufficient for post-simulation analysis.
        * Times are recorded at the **end** of each time step (t + dt), not at
          the beginning (t).

        See Also
        --------
        urbanczik_synapse : Synapse model that uses these signals for plasticity
        """
        return self._urbanczik_history.get(neuron_idx, [])
