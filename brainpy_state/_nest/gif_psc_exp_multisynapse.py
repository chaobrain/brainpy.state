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

r"""Current-based GIF neuron with multiple synaptic time constants.

This module implements ``gif_psc_exp_multisynapse``, the multisynapse
extension of :class:`gif_psc_exp`.  It is a faithful re-implementation of
the identically named NEST model
(``models/gif_psc_exp_multisynapse.{h,cpp}``), preserving update ordering,
exact (analytic) propagator integration, stochastic firing, and all default
parameter values.

The key difference from :class:`gif_psc_exp` is that instead of having two
fixed synaptic channels (excitatory and inhibitory), this model supports an
arbitrary number of receptor ports, each with its own exponential synaptic
time constant.  Incoming spike events specify which receptor port they
target (1-based indexing, as in NEST).

Mathematical model
------------------

Membrane potential ODE:

.. math::

   C_m \frac{dV}{dt} = -g_L (V - E_L)
       - \sum_j \eta_j(t)
       + \sum_k I_{\mathrm{syn},k}(t)
       + I_e + I_{\mathrm{stim}}(t)

Synaptic currents (one per receptor port *k*):

.. math::

   \frac{dI_{\mathrm{syn},k}}{dt} = -\frac{I_{\mathrm{syn},k}}{\tau_{\mathrm{syn},k}}

Spike-triggered currents (STC):

.. math::

   \tau_{\eta_j} \frac{d\eta_j}{dt} = -\eta_j, \qquad
   \eta_j \to \eta_j + q_{\eta_j} \;\text{on spike}

Spike-frequency adaptation (SFA) threshold:

.. math::

   V_T(t) = V_{T^*} + \sum_i \gamma_i(t), \qquad
   \tau_{\gamma_i} \frac{d\gamma_i}{dt} = -\gamma_i, \qquad
   \gamma_i \to \gamma_i + q_{\gamma_i} \;\text{on spike}

Stochastic spiking via exponential escape rate:

.. math::

   \lambda(t) = \lambda_0 \exp\!\bigl((V(t) - V_T(t)) / \Delta_V\bigr),
   \qquad P_{\text{spike}} = 1 - \exp(-\lambda \, dt)

References
----------
.. [1] Mensi S, Naud R, Pozzorini C, Avermann M, Petersen CC, Gerstner W
       (2012). Parameter extraction and classification of three cortical
       neuron types reveals two distinct adaptation mechanisms.
       *J. Neurophysiol.*, 107(6):1756-1775.
.. [2] Pozzorini C, Mensi S, Hagens O, Naud R, Koch C, Gerstner W (2015).
       Automated high-throughput characterization of single neurons by means
       of simplified spiking models. *PLoS Comput. Biol.*, 11(6), e1004275.
.. [3] NEST Simulator ``gif_psc_exp_multisynapse`` model,
       ``models/gif_psc_exp_multisynapse.h`` and
       ``models/gif_psc_exp_multisynapse.cpp``.
"""

import math
from typing import Callable, Iterable, Optional, Sequence

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron
from ._utils import is_tracer
from .gif_psc_exp import gif_psc_exp

__all__ = [
    'gif_psc_exp_multisynapse',
]


class gif_psc_exp_multisynapse(NESTNeuron):
    r"""Current-based generalized integrate-and-fire neuron (GIF) model
    with multiple synaptic time constants.

    This model implements the multisynapse extension of the generalized
    integrate-and-fire neuron according to Mensi et al. (2012) [1]_ and
    Pozzorini et al. (2015) [2]_, with exponential postsynaptic currents
    and an arbitrary number of receptor ports. It is a faithful
    re-implementation of the NEST simulator's ``gif_psc_exp_multisynapse``
    model, preserving exact (analytic) propagator integration, stochastic
    firing dynamics, update ordering, and all default parameter values.

    The model combines four key features:

    1. **Multiple receptor ports**: Each with independent exponential
       synaptic time constants (``tau_syn`` parameter)
    2. **Spike-triggered currents (STC)**: Post-spike current injection
       with multiple time scales (``tau_stc``, ``q_stc`` parameters)
    3. **Spike-frequency adaptation (SFA)**: Dynamic threshold modulation
       after each spike (``tau_sfa``, ``q_sfa`` parameters)
    4. **Stochastic spiking**: Exponential escape-rate firing with
       parameter ``lambda_0`` and threshold noise ``Delta_V``

    Mathematical Model
    ------------------

    **1. Membrane Potential Dynamics**

    The subthreshold membrane potential :math:`V(t)` evolves according to:

    .. math::

       C_m \frac{dV}{dt} = -g_L (V - E_L) - \sum_j \eta_j(t)
           + \sum_k I_{\mathrm{syn},k}(t) + I_e + I_{\mathrm{stim}}(t)

    where:

      - :math:`g_L (V - E_L)` is the passive leak current
      - :math:`\eta_j(t)` are spike-triggered currents (STCs)
      - :math:`I_{\mathrm{syn},k}(t)` are synaptic currents for each receptor port :math:`k`
      - :math:`I_e` is a constant external bias current
      - :math:`I_{\mathrm{stim}}(t)` is time-varying external input

    **2. Synaptic Currents (Multi-Receptor)**

    Each receptor port :math:`k` has an independent exponential synaptic current:

    .. math::

       \frac{dI_{\mathrm{syn},k}}{dt} = -\frac{I_{\mathrm{syn},k}}{\tau_{\mathrm{syn},k}}

    The number of receptor ports is determined by ``len(tau_syn)``. When
    connecting projections, specify ``receptor_type`` (1-based indexing,
    matching NEST convention) to target a specific port. Both excitatory
    and inhibitory connections can target any receptor port (weights can
    be positive or negative).

    **3. Spike-Triggered Currents (STC)**

    Each STC element :math:`\eta_j` evolves as:

    .. math::

       \tau_{\eta_j} \frac{d\eta_j}{dt} = -\eta_j

    Upon spike emission at time :math:`t_{\mathrm{sp}}`:

    .. math::

       \eta_j(t_{\mathrm{sp}}^+) = \eta_j(t_{\mathrm{sp}}^-) + q_{\eta_j}

    The total STC contribution is :math:`\sum_j \eta_j(t)`. STCs can model
    post-spike currents such as afterhyperpolarization (AHP) or
    afterdepolarization (ADP) depending on the sign of ``q_stc``.

    **4. Spike-Frequency Adaptation (SFA)**

    The firing threshold is dynamic, consisting of a base threshold
    :math:`V_{T^*}` plus adaptive components:

    .. math::

       V_T(t) = V_{T^*} + \sum_i \gamma_i(t)

    Each SFA element :math:`\gamma_i` evolves as:

    .. math::

       \tau_{\gamma_i} \frac{d\gamma_i}{dt} = -\gamma_i

    Upon spike emission:

    .. math::

       \gamma_i(t_{\mathrm{sp}}^+) = \gamma_i(t_{\mathrm{sp}}^-) + q_{\gamma_i}

    Positive ``q_sfa`` values increase the threshold after each spike,
    leading to spike-frequency adaptation.

    **5. Stochastic Spiking Mechanism**

    The neuron fires stochastically with an exponential escape-rate
    intensity:

    .. math::

       \lambda(t) = \lambda_0 \exp\!\left(\frac{V(t) - V_T(t)}{\Delta_V}\right)

    The probability of firing within a time step :math:`dt` is:

    .. math::

       P_{\mathrm{spike}}(\Delta t) = 1 - \exp(-\lambda(t) \cdot dt)

    At each non-refractory time step, a uniform random number
    :math:`r \in [0, 1)` is drawn. If :math:`r < P_{\mathrm{spike}}`, a
    spike is emitted. The stochasticity level :math:`\Delta_V` controls
    the sharpness of the firing threshold (smaller values → more
    deterministic).

    **6. Refractory Period**

    After a spike, the neuron enters an absolute refractory period of
    duration :math:`t_{\mathrm{ref}}`. During this period:

    dftype = brainstate.environ.dftype()
    ditype = brainstate.environ.ditype()
    * The membrane potential is clamped to :math:`V_{\mathrm{reset}`
    * No spike can be emitted (firing intensity check is skipped)
    * Synaptic currents continue to evolve and receive inputs
    * STC and SFA elements continue to decay

    **Numerical Integration**

    The model uses **exact (analytic) integration** for all linear ODEs,
    matching NEST's propagator-based integration scheme. For each variable
    with dynamics :math:`\tau \frac{dx}{dt} = -x + f(t)`, the update over
    one time step :math:`h` is:

    .. math::

       x(t + h) = e^{-h/\tau} x(t) + \int_0^h e^{-(h-s)/\tau} f(t+s) \, ds

    For constant :math:`f`, this yields exact propagator coefficients. The
    membrane potential propagator accounts for coupling between :math:`V`
    and synaptic currents with potentially different time constants.

    **Update Order (Matching NEST)**

    Each simulation step follows this exact sequence (matching NEST's
    ``gif_psc_exp_multisynapse::update`` implementation):

    **Step 1: Adaptation Decay**
      - Compute total STC: :math:`\mathrm{stc\_total} = \sum_j \eta_j(t)`
      - Compute total threshold: :math:`V_T(t) = V_{T^*} + \sum_i \gamma_i(t)`
      - Decay all STC elements: :math:`\eta_j \leftarrow \eta_j \cdot e^{-dt/\tau_{\eta_j}}`
      - Decay all SFA elements: :math:`\gamma_i \leftarrow \gamma_i \cdot e^{-dt/\tau_{\gamma_i}}`

    **Step 2: Synaptic Current Processing (per receptor)**
      For each receptor port :math:`k`:
        - Compute propagated contribution to :math:`V`:
          :math:`\Delta V_k = P_{21,k} \cdot I_{\mathrm{syn},k}(t)`
        - Decay synaptic current:
          :math:`I_{\mathrm{syn},k} \leftarrow I_{\mathrm{syn},k} \cdot e^{-dt/\tau_{\mathrm{syn},k}}`
        - Add incoming spike weights:
          :math:`I_{\mathrm{syn},k} \leftarrow I_{\mathrm{syn},k} + w_k`

    **Step 3: Membrane Update and Spike Check**
      If **not refractory**:
        - Update membrane potential using exact propagator:

          .. math::

             V(t+dt) = P_{33} V(t) + P_{31} E_L + P_{30}(I_{\mathrm{stim}}(t) + I_e - \mathrm{stc\_total})
                 + \sum_k \Delta V_k

        - Compute firing intensity: :math:`\lambda = \lambda_0 \exp((V - V_T)/\Delta_V)`
        - Compute spike probability: :math:`p = 1 - \exp(-\lambda \cdot dt)`
        - Draw random number :math:`r \sim \mathrm{Uniform}(0, 1)`
        - If :math:`r < p`:
            * Emit spike
            * Jump STC elements: :math:`\eta_j \leftarrow \eta_j + q_{\eta_j}`
            * Jump SFA elements: :math:`\gamma_i \leftarrow \gamma_i + q_{\gamma_i}`
            * Set refractory counter: :math:`r_{\mathrm{count}} \leftarrow \lceil t_{\mathrm{ref}} / dt \rceil`
      If **refractory**:
        - Decrement refractory counter: :math:`r_{\mathrm{count}} \leftarrow r_{\mathrm{count}} - 1`
        - Clamp membrane potential: :math:`V \leftarrow V_{\mathrm{reset}}`

    **Step 4: Buffer External Current**
      Store :math:`I_{\mathrm{stim}}(t+dt)` for use in the next step
      (NEST ring-buffer semantics: one-step delay).

    **Differences from gif_psc_exp**

    Unlike :class:`gif_psc_exp` which has exactly two fixed synaptic
    channels (excitatory and inhibitory with ``tau_syn_ex``,
    ``tau_syn_in``), this model supports an arbitrary number of receptor
    ports specified by the ``tau_syn`` parameter. This enables:

    * Multi-receptor modeling (AMPA, NMDA, GABA_A, GABA_B, etc.)
    * Heterogeneous synaptic time constants within the same neuron
    * Flexible connectivity patterns with receptor-specific routing

    All spike weights are applied to the receptor port specified in the
    connection's ``receptor_type`` field (1-based indexing). Positive or
    negative weights are both allowed on any receptor.

    Parameters
    ----------
    in_size : int, tuple of int
        Shape of the neuron population. Can be an integer (for 1D
        populations) or a tuple of integers (for multi-dimensional arrays).
    g_L : Quantity, ArrayLike, optional
        Leak conductance (nanosiemens). Determines the passive membrane
        time constant :math:`\tau_m = C_m / g_L`. Must be strictly positive.
        Default: 4.0 nS.
    E_L : Quantity, ArrayLike, optional
        Leak reversal potential (millivolts). Resting potential in the
        absence of inputs. Default: -70.0 mV.
    C_m : Quantity, ArrayLike, optional
        Membrane capacitance (picofarads). Together with ``g_L``, determines
        the membrane time constant. Must be strictly positive. Default:
        80.0 pF (gives :math:`\tau_m = 20` ms with default ``g_L``).
    V_reset : Quantity, ArrayLike, optional
        Reset potential (millivolts). Membrane potential is clamped to this
        value during the refractory period. Default: -55.0 mV.
    Delta_V : Quantity, ArrayLike, optional
        Voltage scale of stochastic firing (millivolts). Controls the
        sharpness of the firing threshold in the exponential escape-rate
        model. Smaller values make spiking more deterministic. Must be
        strictly positive. Default: 0.5 mV.
    V_T_star : Quantity, ArrayLike, optional
        Base firing threshold (millivolts). The resting threshold in the
        absence of spike-frequency adaptation. Default: -35.0 mV.
    lambda_0 : float, optional
        Stochastic firing intensity at threshold (1/s). When :math:`V(t) =
        V_T(t)`, the firing rate is ``lambda_0``. Must be non-negative.
        Internally converted to 1/ms for computation. Default: 1.0 /s.
    t_ref : Quantity, ArrayLike, optional
        Absolute refractory period (milliseconds). Duration during which
        the neuron cannot spike after an emitted spike. Must be
        non-negative. Default: 4.0 ms.
    tau_syn : sequence of float, optional
        Synaptic time constants (milliseconds), one per receptor port.
        Specified as bare floats (not Quantities), matching NEST C++
        parameterization. The number of elements determines the number of
        receptor ports. All values must be strictly positive. Default:
        ``(2.0,)`` (single receptor with 2 ms time constant).
    I_e : Quantity, ArrayLike, optional
        Constant external bias current (picoamperes). Applied at every
        time step. Default: 0.0 pA.
    tau_sfa : sequence of float, optional
        Spike-frequency adaptation time constants (milliseconds). Specified
        as bare floats. Must have the same length as ``q_sfa``. All values
        must be strictly positive. Default: ``()`` (no adaptation).
    q_sfa : sequence of float, optional
        SFA jump values (millivolts). Amount by which each SFA element
        increases upon spike emission. Specified as bare floats. Positive
        values cause threshold elevation (adaptation). Default: ``()`` (no
        adaptation).
    tau_stc : sequence of float, optional
        Spike-triggered current time constants (milliseconds). Specified as
        bare floats. Must have the same length as ``q_stc``. All values
        must be strictly positive. Default: ``()`` (no STC).
    q_stc : sequence of float, optional
        STC jump values (nanoamperes). Amount by which each STC element
        increases upon spike emission. Specified as bare floats. Negative
        values cause hyperpolarizing currents (e.g., AHP). Default: ``()``
        (no STC).
    rng_key : jax.Array, optional
        JAX PRNG key for stochastic spike generation. If ``None``, a
        default key (``jax.random.PRNGKey(0)``) is used. For reproducible
        simulations, provide an explicit key. Default: None.
    V_initializer : Callable, optional
        Initializer function for membrane potential. Should return a
        Quantity in millivolts. Default: ``Constant(-70.0 * u.mV)`` (resting
        potential).
    spk_fun : Callable, optional
        Surrogate gradient function for differentiable spike generation.
        Applied to scaled voltage for gradient computation. Default:
        ``ReluGrad()`` (ReLU-based surrogate).
    spk_reset : str, optional
        Spike reset mode for gradient flow. Options are ``'hard'`` (stop
        gradient at reset, matches NEST) or ``'soft'`` (allow gradient
        through reset). Default: ``'hard'``.
    name : str, optional
        Name of the neuron population. Used for logging and debugging. If
        ``None``, an automatic name is generated. Default: None.


    Parameter Mapping
    -----------------

    Below is the correspondence between brainpy.state and NEST parameter names:

    ==================== =================== =================================== =====================================================
    **brainpy.state**    **NEST**            **Mathematical Symbol**             **Description**
    ==================== =================== =================================== =====================================================
    ``in_size``          n/a                 —                                   Population shape (brainpy.state-specific)
    ``g_L``              ``g_L``             :math:`g_L`                         Leak conductance
    ``E_L``              ``E_L``             :math:`E_L`                         Leak reversal potential
    ``C_m``              ``C_m``             :math:`C_m`                         Membrane capacitance
    ``V_reset``          ``V_reset``         :math:`V_{\mathrm{reset}}`          Reset potential
    ``Delta_V``          ``Delta_V``         :math:`\Delta_V`                    Stochastic firing sharpness
    ``V_T_star``         ``V_T_star``        :math:`V_{T^*}`                     Base firing threshold
    ``lambda_0``         ``lambda_0``        :math:`\lambda_0`                   Firing intensity at threshold (1/s)
    ``t_ref``            ``t_ref``           :math:`t_{\mathrm{ref}}`            Absolute refractory period
    ``tau_syn``          ``tau_syn``         :math:`\tau_{\mathrm{syn},k}`       Synaptic time constants (list/tuple)
    ``I_e``              ``I_e``             :math:`I_e`                         Constant external current
    ``tau_sfa``          ``tau_sfa``         :math:`\tau_{\gamma_i}`             SFA time constants (list/tuple)
    ``q_sfa``            ``q_sfa``           :math:`q_{\gamma_i}`                SFA jump values (list/tuple)
    ``tau_stc``          ``tau_stc``         :math:`\tau_{\eta_j}`               STC time constants (list/tuple)
    ``q_stc``            ``q_stc``           :math:`q_{\eta_j}`                  STC jump values (list/tuple)
    ``rng_key``          RNG state           —                                   JAX PRNG key (brainpy.state-specific)
    ``V_initializer``    ``V_m``             :math:`V(0)`                        Initial membrane potential
    ``spk_fun``          n/a                 —                                   Surrogate gradient (brainpy.state-specific)
    ``spk_reset``        n/a                 —                                   Reset mode (brainpy.state-specific)
    ==================== =================== =================================== =====================================================

    State Variables
    ---------------
    V : HiddenState, shape ``(batch_size, *in_size)``
        Membrane potential :math:`V_m(t)` in millivolts.
    i_syn : ShortTermState, shape ``(batch_size, *in_size, n_receptors)``
        Synaptic currents in picoamperes, one per receptor port.
        :math:`I_{\mathrm{syn},k}(t)` for :math:`k = 0, \ldots, n_{\mathrm{receptors}} - 1`.
    refractory_step_count : ShortTermState, shape ``(batch_size, *in_size)``
        Remaining refractory time steps (int32). Counts down from
        :math:`\lceil t_{\mathrm{ref}} / dt \rceil` to 0 after each spike.
    I_stim : ShortTermState, shape ``(batch_size, *in_size)``
        Buffered external current (picoamperes) applied in the next time
        step. Implements NEST's ring-buffer semantics (one-step delay).
    last_spike_time : ShortTermState, shape ``(batch_size, *in_size)``
        Time of the last emitted spike (milliseconds). Initialized to a
        large negative value (-1e7 ms).

    Notes
    -----
    **Implementation Details:**

    * **Random number generation**: Uses JAX's stateless PRNG. The RNG key
      is split at each time step to ensure reproducibility. Exact
      spike-time reproducibility requires matching the initial ``rng_key``.

    * **Parameter units**: Following NEST's C++ implementation,
      ``tau_syn``, ``tau_sfa``, ``q_sfa``, ``tau_stc``, and ``q_stc`` are
      specified as **bare floats** (not saiunit Quantities) in their
      respective natural units (ms, mV, nA). All other parameters use
      saiunit Quantities for dimensional safety.

    * **Exact integration**: The model uses analytic propagator
      coefficients for exact integration of linear ODEs, matching NEST's
      approach. This ensures numerical accuracy independent of time step
      size (within machine precision).

    * **Singular propagator handling**: If :math:`\tau_m \approx
      \tau_{\mathrm{syn},k}`, the propagator formula is numerically
      singular. The implementation uses a threshold-based switch to a
      limiting formula to avoid instabilities.

    * **GIF toolbox compatibility**: NEST and the GIF toolbox (Pozzorini et
      al.) use different conventions for when STC/SFA jumps occur. NEST
      applies jumps immediately after spike emission; the toolbox applies
      them after the refractory period. Conversion formula:

      .. math::

         q_{\eta,\text{toolbox}} = q_{\eta,\text{NEST}} \cdot
             \left(1 - e^{-t_{\mathrm{ref}} / \tau_\eta}\right)

    * **Receptor indexing**: NEST uses **1-based indexing** for receptor
      types (``receptor_type=1`` for the first receptor). Internally, this
      implementation uses 0-based indexing (Python/NumPy convention).

    * **Default behavior**: With default parameters (single receptor,
      empty STC/SFA), this model behaves as a simple stochastic LIF neuron
      with exponential synaptic currents.

    **Failure Modes**


    * If ``Delta_V`` is too small (< 1e-6 mV), numerical overflow may occur
      in the exponential firing intensity computation.
    * If ``tau_m`` and ``tau_syn[k]`` differ by less than machine epsilon,
      the propagator may become ill-conditioned despite singular handling.
    * Extremely short time steps (< 1e-3 ms) may lead to refractory counter
      overflow (int32 limit).

    Examples
    --------
    **Basic usage with single receptor:**

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import saiunit as u
       >>> # Create a small population
       >>> neurons = bst.gif_psc_exp_multisynapse(10, tau_syn=(2.0,))
       >>> neurons.init_state()
       >>> # Simulate one time step
       >>> spikes = neurons(x=100.0 * u.pA)  # External current input

    **Multi-receptor configuration (AMPA, NMDA, GABA):**

    .. code-block:: python

       >>> # Define three receptor types with different time constants
       >>> neurons = bst.gif_psc_exp_multisynapse(
       ...     100,
       ...     tau_syn=(2.0, 10.0, 5.0),  # AMPA-like, NMDA-like, GABA-like
       ... )
       >>> neurons.init_state()
       >>> # Connect synapses to different receptors
       >>> # (receptor_type=1 for AMPA, =2 for NMDA, =3 for GABA)

    **With spike-frequency adaptation:**

    .. code-block:: python

       >>> neurons = bst.gif_psc_exp_multisynapse(
       ...     50,
       ...     tau_syn=(2.0,),
       ...     tau_sfa=(100.0, 500.0),  # Two SFA time scales
       ...     q_sfa=(5.0, 10.0),       # Threshold increases by 5 and 10 mV
       ... )
       >>> neurons.init_state()

    **With spike-triggered currents (afterhyperpolarization):**

    .. code-block:: python

       >>> neurons = bst.gif_psc_exp_multisynapse(
       ...     50,
       ...     tau_syn=(2.0,),
       ...     tau_stc=(10.0,),
       ...     q_stc=(-50.0,),  # Hyperpolarizing current (-50 pA)
       ... )
       >>> neurons.init_state()

    **Deterministic spike generation (for testing):**

    .. code-block:: python

       >>> import jax
       >>> neurons = bst.gif_psc_exp_multisynapse(
       ...     10,
       ...     tau_syn=(2.0,),
       ...     rng_key=jax.random.PRNGKey(42),  # Fixed seed
       ... )
       >>> neurons.init_state()

    See Also
    --------
    gif_psc_exp : Two-receptor GIF model (excitatory/inhibitory)
    iaf_psc_exp_multisynapse : Multi-receptor IAF model without adaptation
    aeif_psc_exp_multisynapse : Multi-receptor AdEx model

    References
    ----------
    .. [1] Mensi S, Naud R, Pozzorini C, Avermann M, Petersen CC, Gerstner W
           (2012). Parameter extraction and classification of three cortical
           neuron types reveals two distinct adaptation mechanisms. Journal of
           Neurophysiology, 107(6):1756-1775.
           DOI: https://doi.org/10.1152/jn.00408.2011
    .. [2] Pozzorini C, Mensi S, Hagens O, Naud R, Koch C, Gerstner W (2015).
           Automated high-throughput characterization of single neurons by
           means of simplified spiking models. PLoS Computational Biology,
           11(6), e1004275.
           DOI: https://doi.org/10.1371/journal.pcbi.1004275
    .. [3] NEST Simulator ``gif_psc_exp_multisynapse`` model documentation
           and C++ source: ``models/gif_psc_exp_multisynapse.h`` and
           ``models/gif_psc_exp_multisynapse.cpp``.
           https://nest-simulator.readthedocs.io/en/stable/models/gif_psc_exp_multisynapse.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        g_L: ArrayLike = 4.0 * u.nS,
        E_L: ArrayLike = -70.0 * u.mV,
        C_m: ArrayLike = 80.0 * u.pF,
        V_reset: ArrayLike = -55.0 * u.mV,
        Delta_V: ArrayLike = 0.5 * u.mV,
        V_T_star: ArrayLike = -35.0 * u.mV,
        lambda_0: float = 1.0,  # 1/s, as in NEST Python interface
        t_ref: ArrayLike = 4.0 * u.ms,
        tau_syn: Sequence[float] = (2.0,),  # ms values
        I_e: ArrayLike = 0.0 * u.pA,
        tau_sfa: Sequence[float] = (),  # ms values
        q_sfa: Sequence[float] = (),  # mV values
        tau_stc: Sequence[float] = (),  # ms values
        q_stc: Sequence[float] = (),  # nA values
        rng_key: Optional[jax.Array] = None,
        V_initializer: Callable = braintools.init.Constant(-70.0 * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Membrane parameters
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.Delta_V = braintools.init.param(Delta_V, self.varshape)
        self.V_T_star = braintools.init.param(V_T_star, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        # Synaptic time constants (stored as numpy array of ms values)
        self.tau_syn = np.asarray([float(x) for x in tau_syn], dtype=dftype)
        if len(self.tau_syn) == 0:
            raise ValueError("'tau_syn' must have at least one element.")

        # Stochastic spiking: lambda_0 in 1/s, store as 1/ms internally
        self.lambda_0 = lambda_0 / 1000.0  # convert from 1/s to 1/ms

        # Adaptation parameters (stored as plain Python tuples of floats)
        self.tau_sfa = tuple(float(x) for x in tau_sfa)
        self.q_sfa = tuple(float(x) for x in q_sfa)
        self.tau_stc = tuple(float(x) for x in tau_stc)
        self.q_stc = tuple(float(x) for x in q_stc)

        if len(self.tau_sfa) != len(self.q_sfa):
            raise ValueError(
                f"'tau_sfa' and 'q_sfa' must have the same length. "
                f"Got {len(self.tau_sfa)} and {len(self.q_sfa)}."
            )
        if len(self.tau_stc) != len(self.q_stc):
            raise ValueError(
                f"'tau_stc' and 'q_stc' must have the same length. "
                f"Got {len(self.tau_stc)} and {len(self.q_stc)}."
            )

        # RNG key for stochastic spiking
        self._rng_key = rng_key

        # Initializers
        self.V_initializer = V_initializer

        self._validate_parameters()

    @property
    def n_receptors(self):
        r"""Number of synaptic receptor ports.

        Returns
        -------
        int
            Number of receptor ports, equal to ``len(tau_syn)``.
        """
        return int(self.tau_syn.size)

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.g_L, self.Delta_V)):
            return
        if np.any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self.g_L <= 0.0 * u.nS):
            raise ValueError('Membrane conductance must be strictly positive.')
        if np.any(self.Delta_V <= 0.0 * u.mV):
            raise ValueError('Delta_V must be strictly positive.')
        if np.any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time must not be negative.')
        if self.lambda_0 < 0.0:
            raise ValueError('lambda_0 must not be negative.')
        for i, tau in enumerate(self.tau_syn):
            if tau <= 0.0:
                raise ValueError(
                    f'All synaptic time constants must be strictly positive '
                    f'(tau_syn[{i}]={tau}).'
                )
        for i, tau in enumerate(self.tau_sfa):
            if tau <= 0.0:
                raise ValueError(
                    f'All SFA time constants must be strictly positive '
                    f'(tau_sfa[{i}]={tau}).'
                )
        for i, tau in enumerate(self.tau_stc):
            if tau <= 0.0:
                raise ValueError(
                    f'All STC time constants must be strictly positive '
                    f'(tau_stc[{i}]={tau}).'
                )

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Creates and initializes state variables including membrane potential,
        synaptic currents, refractory counters, adaptation elements, and
        buffered currents. Also initializes the internal RNG state for
        stochastic spike generation.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size for vectorized simulation. If ``None``, no
            batch dimension is added. If provided, all state variables will
            have shape ``(batch_size, *in_size, ...)``. Default: None.
        **kwargs
            Additional keyword arguments (reserved for future extensions,
            currently unused).

        Notes
        -----
        **State Variable Initialization:**

        - ``V``: Membrane potential initialized using ``V_initializer``
          (default: -70 mV, resting potential).
        - ``i_syn``: Synaptic currents initialized to zero for all
          receptors. Shape: ``(*v_shape, n_receptors)``.
        - ``refractory_step_count``: Refractory counter initialized to 0
          (not refractory).
        - ``I_stim``: Buffered external current initialized to 0 pA.
        - ``last_spike_time``: Last spike time initialized to -1e7 ms (far
          in the past).
        - STC elements (``_stc_elems``): Initialized to zero if
          ``len(tau_stc) > 0``, otherwise ``None``.
        - SFA elements (``_sfa_elems``): Initialized to zero if
          ``len(tau_sfa) > 0``, otherwise ``None``.
        - Total STC (``_stc_val``): Initialized to zero.
        - Total threshold (``_sfa_val``): Initialized to ``V_T_star``.

        **RNG Initialization:**
        If ``rng_key`` was provided during construction, uses that key.
        Otherwise, uses ``jax.random.PRNGKey(0)``. The RNG state is stored
        internally and updated at each time step.

        This method should be called once before starting simulation, or
        after calling ``reset_state()`` to reinitialize.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        self.V = brainstate.HiddenState(V)

        # Synaptic currents: shape (..., n_receptors)
        syn_zeros = np.zeros(v_shape + (self.n_receptors,), dtype=dftype)
        self.i_syn = brainstate.ShortTermState(syn_zeros * u.pA)

        spk_time = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(
            braintools.init.Constant(0), self.varshape, batch_size
        )
        self.refractory_step_count = brainstate.ShortTermState(
            u.math.asarray(ref_steps, dtype=ditype)
        )

        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(
                braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
            )
        )

        # Adaptation state: stc and sfa element arrays
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=dftype) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=dftype) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=dftype)
        self._sfa_val = np.full(
            v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=dftype
        )

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to initial values.

        Resets membrane potential, synaptic currents, refractory counters,
        adaptation elements, and buffered currents to their initial states.
        Also reinitializes the internal RNG state. This is equivalent to
        calling ``init_state()`` but reuses existing state variable objects.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If ``None``, uses the existing batch size
            (or no batch dimension). If provided and different from the
            current batch size, reshapes all state variables. Default: None.
        **kwargs
            Additional keyword arguments (reserved for future extensions,
            currently unused).

        Notes
        -----
        This method is useful for:

        - Restarting simulations from initial conditions without recreating
          the neuron object.
        - Running multiple trials with the same model configuration.
        - Resetting after parameter changes that require state
          reinitialization.

        The reset follows the same initialization logic as ``init_state()``,
        with all state variables returned to their default values (membrane
        potential from ``V_initializer``, currents to zero, refractory
        counter to zero, etc.).

        If ``batch_size`` is changed, all state arrays are reshaped
        accordingly. This allows switching between single-neuron and
        batched simulations.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        syn_zeros = np.zeros(v_shape + (self.n_receptors,), dtype=dftype)
        self.i_syn.value = syn_zeros * u.pA

        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(
            braintools.init.Constant(0), self.varshape, batch_size
        )
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=ditype)
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=dftype) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=dftype) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=dftype)
        self._sfa_val = np.full(
            v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=dftype
        )

        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute spike output via surrogate gradient function.

        Converts membrane potential to a differentiable spike signal by
        scaling and applying the surrogate gradient function. This method is
        used for gradient-based learning but does **not** affect the
        stochastic spike generation during ``update()``.

        Parameters
        ----------
        V : Quantity, ArrayLike, optional
            Membrane potential (millivolts). If ``None``, uses the current
            state ``self.V.value``. Default: None.

        Returns
        -------
        ndarray
            Spike output (float32). Shape matches ``V``. Typically binary
            (0 or 1) in forward pass, continuous in backward pass for
            gradient computation.

        Notes
        -----
        The scaling formula is :math:`v_{\mathrm{scaled}} = (V -
        V_{\mathrm{reset}}) / \Delta_V`, which normalizes the voltage
        relative to reset potential. The surrogate function then maps this
        to a spike probability-like value.

        This method is separate from the stochastic firing mechanism used in
        ``update()``, which draws random numbers and compares to the
        exponential escape-rate probability.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_reset) / self.Delta_V
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        r"""Parse spike events into per-receptor weight arrays.

        This method accumulates incoming spike weights into a multi-dimensional
        array organized by receptor port. It handles both tuple and dictionary
        event formats, supporting NEST-style 1-based receptor indexing.

        Parameters
        ----------
        spike_events : iterable or None
            Incoming spike events. Each element can be:
            - A ``(receptor_type, weight)`` tuple
            - A dict with keys ``'receptor_type'`` (or ``'receptor'``) and
              ``'weight'``
            where ``receptor_type`` is a 1-based integer (1 to ``n_receptors``)
            and ``weight`` is a current value (Quantity in pA or bare float).
            If ``None``, returns an array of zeros.
        v_shape : tuple of int
            Shape of the neuron population state (excluding receptor dimension).

        Returns
        -------
        ndarray
            Accumulated spike weights per receptor. Shape: ``v_shape +
            (n_receptors,)``. Data type: float64. Units: pA (as bare floats).

        Raises
        ------
        ValueError
            If any ``receptor_type`` is outside the range [1, ``n_receptors``].
        """
        out = np.zeros(v_shape + (self.n_receptors,), dtype=dftype)
        if spike_events is None:
            return out
        for ev in spike_events:
            if isinstance(ev, dict):
                receptor = int(ev.get('receptor_type', ev.get('receptor', 1)))
                weight = ev.get('weight', 0.0)
            else:
                receptor, weight = ev
                receptor = int(receptor)
            if receptor < 1 or receptor > self.n_receptors:
                raise ValueError(
                    f'Receptor type {receptor} out of range '
                    f'[1, {self.n_receptors}].'
                )
            w_np = np.asarray(u.math.asarray(weight / u.pA), dtype=dftype)
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    def update(self, x=0.0 * u.pA, spike_events=None):
        r"""Update neuron state for one simulation step.

        This method advances the neuron state by one time step ``dt``,
        integrating membrane dynamics, synaptic currents, adaptation
        mechanisms, and stochastic spike generation. It follows NEST's exact
        update order for ``gif_psc_exp_multisynapse``.

        The update sequence is:

        1. Decay STC and SFA elements, compute their totals
        2. Process synaptic currents for each receptor (propagate, decay, add spikes)
        3. Update membrane potential (if not refractory) or decrement refractory counter
        4. Stochastic spike check and adaptation updates (if spike emitted)
        5. Buffer external current for next step

        Parameters
        ----------
        x : Quantity, optional
            External current input (picoamperes). This current is
            **buffered by one time step** (NEST ring-buffer semantics): the current
            provided at time :math:`t` is applied at time :math:`t + dt`.
            Default: 0.0 pA.
        spike_events : iterable, optional
            Incoming spike events from presynaptic neurons. Each event
            specifies a receptor port and synaptic weight. Supported formats:

            - ``(receptor_type, weight)`` tuples
            - Dicts with keys ``'receptor_type'`` (or ``'receptor'``) and
              ``'weight'``

            where ``receptor_type`` is 1-based (1 to ``n_receptors``) and
            ``weight`` is in pA (Quantity or float). If ``None``, only delta
            inputs registered via ``add_delta_input()`` are used (mapped to
            receptor 1). Default: None.

        Returns
        -------
        ndarray
            Spike output for this time step. Shape: ``(batch_size, *in_size)``.
            Data type: float32. Values are in [0, 1] via the surrogate
            gradient function (typically binary in forward pass, continuous
            in backward pass for gradient computation).

        Notes
        -----
        **Random Number Generation:**
        At each non-refractory time step, a uniform random number is drawn
        from the JAX PRNG. The internal RNG state is updated automatically.
        For reproducible results, initialize with a fixed ``rng_key``.

        **Current Input Buffering:**
        The external current ``x`` is stored in ``I_stim`` and applied in
        the **next** time step. This matches NEST's ring-buffer semantics,
        where spike events and external currents arrive one step before
        being applied.

        **Receptor Mapping:**

        - Spike events explicitly specify their target receptor via
          ``receptor_type`` (1-based).
        - Delta inputs registered via ``add_delta_input()`` are always
          applied to receptor 1 (the first receptor).
        - Current inputs registered via ``add_current_input()`` are
          accumulated into ``I_stim`` and distributed uniformly (not
          receptor-specific).

        **Failure Modes**


        - If ``receptor_type`` in ``spike_events`` is out of range [1,
          ``n_receptors``], raises ``ValueError``.
        - If ``Delta_V`` is extremely small, the exponential firing
          intensity may overflow (returns ``inf``), causing all neurons to
          spike deterministically.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))  # dt in ms

        v_shape = self.V.value.shape

        # Extract state variables as numpy arrays
        V = self._broadcast_to_state(
            self._to_numpy(self.V.value, u.mV), v_shape
        ).copy()
        i_syn = np.asarray(
            u.math.asarray(self.i_syn.value / u.pA), dtype=dftype
        ).copy()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=ditype),
            v_shape,
        ).copy()
        i_stim = self._broadcast_to_state(
            self._to_numpy(self.I_stim.value, u.pA), v_shape
        ).copy()

        # Extract parameters as numpy arrays
        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        g_L = self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        V_reset = self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape)
        V_T_star = float(self._to_numpy(self.V_T_star, u.mV))
        Delta_V = float(self._to_numpy(self.Delta_V, u.mV))
        lambda_0 = self.lambda_0  # 1/ms

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=ditype),
            v_shape,
        )

        # Compute propagator coefficients (exact integration)
        tau_m = C_m / g_L  # membrane time constant in ms
        P33 = np.exp(-h / tau_m)
        P30 = -1.0 / C_m * np.expm1(-h / tau_m) * tau_m
        P31 = -np.expm1(-h / tau_m)

        # Per-receptor propagator coefficients
        P11_syn = np.exp(-h / self.tau_syn)  # shape: (n_receptors,)
        P21_syn = np.stack([
            gif_psc_exp._propagator_exp(
                tau_s * np.ones(v_shape), tau_m, C_m, h
            )
            for tau_s in self.tau_syn
        ], axis=-1)  # shape: v_shape + (n_receptors,)

        # Adaptation decay factors
        P_stc = [math.exp(-h / tau) for tau in self.tau_stc]
        P_sfa = [math.exp(-h / tau) for tau in self.tau_sfa]

        # Parse spike events into per-receptor arrays
        w_by_rec = self._parse_spike_events(spike_events, v_shape)

        # Map default delta inputs to receptor 1
        w_default = self._broadcast_to_state(
            self._to_numpy(self.sum_delta_inputs(0.0 * u.pA), u.pA), v_shape
        )
        if self.n_receptors > 0:
            w_by_rec[..., 0] += w_default

        # Get external current for NEXT step (NEST ring buffer semantics)
        new_i_stim = self._broadcast_to_state(
            self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape
        )

        # Advance RNG state for this step
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(
            jax.random.uniform(subkey, shape=v_shape), dtype=dftype
        )

        spike_mask = np.zeros_like(V, dtype=bool)

        for idx in np.ndindex(v_shape):
            # ---- Step 1: Decay stc/sfa elements and compute totals ----
            stc_total = 0.0
            if self._stc_elems is not None:
                for i in range(len(self.tau_stc)):
                    stc_total += self._stc_elems[i][idx]
                    self._stc_elems[i][idx] *= P_stc[i]

            sfa_total = V_T_star
            if self._sfa_elems is not None:
                for i in range(len(self.tau_sfa)):
                    sfa_total += self._sfa_elems[i][idx]
                    self._sfa_elems[i][idx] *= P_sfa[i]

            self._stc_val[idx] = stc_total
            self._sfa_val[idx] = sfa_total

            # ---- Step 2: Synaptic currents ----
            # Compute propagated contribution, decay, then add new spikes
            # (matches NEST update order exactly)
            sum_syn_pot = 0.0
            for k in range(self.n_receptors):
                syn_idx = idx + (k,)
                sum_syn_pot += P21_syn[syn_idx] * i_syn[syn_idx]
                i_syn[syn_idx] *= P11_syn[k]
                i_syn[syn_idx] += w_by_rec[syn_idx]

            # ---- Step 3: Refractory / membrane update / spike check ----
            if r[idx] == 0:
                # Not refractory: update membrane potential via exact propagator
                V[idx] = (P30[idx] * (i_stim[idx] + I_e[idx] - stc_total)
                          + P33[idx] * V[idx]
                          + P31[idx] * E_L[idx]
                          + sum_syn_pot)

                # Stochastic spike check
                lam = lambda_0 * math.exp((V[idx] - sfa_total) / Delta_V)
                if lam > 0.0:
                    spike_prob = -math.expm1(-lam * h)
                    if rand_vals[idx] < spike_prob:
                        # Spike!
                        spike_mask[idx] = True

                        # Jump stc elements
                        if self._stc_elems is not None:
                            for i in range(len(self.q_stc)):
                                self._stc_elems[i][idx] += self.q_stc[i]

                        # Jump sfa elements
                        if self._sfa_elems is not None:
                            for i in range(len(self.q_sfa)):
                                self._sfa_elems[i][idx] += self.q_sfa[i]

                        r[idx] = refr_counts[idx]
            else:
                # Refractory: decrement counter, clamp V to V_reset
                r[idx] -= 1
                V[idx] = V_reset[idx]

        # ---- Step 4: Store new I_stim for next step, update state ----
        self.V.value = V * u.mV
        self.i_syn.value = i_syn * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=ditype)
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)
