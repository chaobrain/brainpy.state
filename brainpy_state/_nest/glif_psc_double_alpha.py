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

r"""Current-based generalized leaky integrate-and-fire (GLIF) neuron model
with double alpha-function shaped synaptic currents.

This module implements the ``glif_psc_double_alpha`` neuron model from the
NEST simulator. It extends the ``glif_psc`` model by using a double alpha
function (fast + slow components) for synaptic currents, allowing more
flexible control over the synaptic current waveform shape and tail.

The implementation uses exact integration (propagator matrices) matching
NEST's numerical scheme for linear subthreshold dynamics.

References
----------
.. [1] Teeter C, Iyer R, Menon V, Gouwens N, Feng D, Berg J, Szafer A,
       Cain N, Zeng H, Hawrylycz M, Koch C, & Mihalas S (2018).
       Generalized leaky integrate-and-fire models classify multiple neuron
       types. Nature Communications 9:709.
.. [2] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
       the large, fluctuating synaptic conductance state typical of
       neocortical neurons in vivo. J. Comput. Neurosci. 16:159-175.
.. [3] NEST Simulator ``glif_psc_double_alpha`` model documentation and
       C++ source: ``models/glif_psc_double_alpha.h`` and
       ``models/glif_psc_double_alpha.cpp``.
"""

from typing import Callable, Sequence

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
    'glif_psc_double_alpha',
]


class glif_psc_double_alpha(NESTNeuron):
    r"""Current-based generalized leaky integrate-and-fire (GLIF) neuron model
    with double alpha-function shaped synaptic currents.

    Implements the NEST ``glif_psc_double_alpha`` model, which extends the basic
    GLIF framework [1]_ with dual-component (fast + slow) alpha-function shaped
    postsynaptic currents [2]_. This allows flexible control over synaptic
    waveform shape, including realistic biphasic or long-tailed currents observed
    experimentally. The model provides five GLIF variants (Models 1-5) selectable
    via boolean flags, ranging from simple LIF to adaptive threshold models with
    after-spike currents.

    **Model Family Overview**

    The five GLIF models are hierarchical, each adding biological mechanisms:

    * **GLIF Model 1** (LIF) — Traditional leaky integrate-and-fire
    * **GLIF Model 2** (LIF_R) — LIF with biologically defined reset rules
    * **GLIF Model 3** (LIF_ASC) — LIF with after-spike currents
    * **GLIF Model 4** (LIF_R_ASC) — LIF with reset rules and after-spike currents
    * **GLIF Model 5** (LIF_R_ASC_A) — LIF with reset rules, after-spike currents,
      and a voltage-dependent threshold

    Model selection is determined by three boolean parameters:

    +--------+---------------------------+----------------------+--------------------+
    | Model  | spike_dependent_threshold | after_spike_currents | adapting_threshold |
    +========+===========================+======================+====================+
    | GLIF1  | False                     | False                | False              |
    +--------+---------------------------+----------------------+--------------------+
    | GLIF2  | True                      | False                | False              |
    +--------+---------------------------+----------------------+--------------------+
    | GLIF3  | False                     | True                 | False              |
    +--------+---------------------------+----------------------+--------------------+
    | GLIF4  | True                      | True                 | False              |
    +--------+---------------------------+----------------------+--------------------+
    | GLIF5  | True                      | True                 | True               |
    +--------+---------------------------+----------------------+--------------------+

    **Double Alpha-Function Synaptic Currents**

    Each synaptic receptor port receives inputs shaped by a sum of two alpha
    functions (fast and slow components) [2]_:

    .. math::

        I_\mathrm{syn,k}(t) = \alpha_\mathrm{fast}(t; \tau_{\mathrm{syn,fast},k})
                            + \mathrm{amp\_slow}_k \cdot
                              \alpha_\mathrm{slow}(t; \tau_{\mathrm{syn,slow},k})

    Normalization: A spike of weight 1.0 produces a peak current of 1 pA for the
    fast component at :math:`t = \tau_\mathrm{syn,fast}`. The slow component peaks
    at :math:`\mathrm{amp\_slow}_k` pA at :math:`t = \tau_\mathrm{syn,slow}`.

    Multiple receptor ports are supported by passing arrays to ``tau_syn_fast``,
    ``tau_syn_slow``, and ``amp_slow``. By default, one receptor port is created.
    Projections specify receptor ports via ``receptor_<k>`` labels (0-based indexing).

    **Detailed Mathematical Description**

    **1. Membrane Dynamics**

    The membrane potential :math:`U` (tracked relative to :math:`E_L`) evolves
    via exact integration using propagator matrices:

    .. math::

       U(t+dt) = U(t) \cdot P_{33} + (I_e + I_\mathrm{stim} + I_\mathrm{ASC,sum}) \cdot P_{30}
                 + \sum_k \left( P_{31,k}^\mathrm{fast} \cdot y_{1,k}^\mathrm{fast}
                               + P_{32,k}^\mathrm{fast} \cdot y_{2,k}^\mathrm{fast} \right)
                 + \sum_k \left( P_{31,k}^\mathrm{slow} \cdot y_{1,k}^\mathrm{slow}
                               + P_{32,k}^\mathrm{slow} \cdot y_{2,k}^\mathrm{slow} \right)

    where:

    .. math::

       P_{33} = \exp\left(-\frac{dt}{\tau_m}\right), \quad
       P_{30} = \frac{\tau_m}{C_m} \left(1 - P_{33}\right), \quad
       \tau_m = \frac{C_m}{g}

    The propagators :math:`P_{31,k}`, :math:`P_{32,k}` for each receptor and
    component (fast/slow) are computed using the ``IAFPropagatorAlpha`` algorithm,
    which handles the singularity when :math:`\tau_m \approx \tau_{\mathrm{syn},k}`.

    **2. Synaptic Current Dynamics (Double Alpha Function)**

    Each receptor port :math:`k` maintains **four** state variables: two for the
    fast component and two for the slow component. Each pair :math:`(y_1, y_2)`
    represents an alpha function.

    **Fast component:**

    .. math::

       y_{2,k}^\mathrm{fast}(t+dt) = P_{21,k}^\mathrm{fast} \cdot y_{1,k}^\mathrm{fast}(t)
                                   + P_{22,k}^\mathrm{fast} \cdot y_{2,k}^\mathrm{fast}(t)

    .. math::

       y_{1,k}^\mathrm{fast}(t+dt) = P_{11,k}^\mathrm{fast} \cdot y_{1,k}^\mathrm{fast}(t)

    **Slow component:**

    .. math::

       y_{2,k}^\mathrm{slow}(t+dt) = P_{21,k}^\mathrm{slow} \cdot y_{1,k}^\mathrm{slow}(t)
                                   + P_{22,k}^\mathrm{slow} \cdot y_{2,k}^\mathrm{slow}(t)

    .. math::

       y_{1,k}^\mathrm{slow}(t+dt) = P_{11,k}^\mathrm{slow} \cdot y_{1,k}^\mathrm{slow}(t)

    where:

    .. math::

       P_{11,k}^\mathrm{fast} = P_{22,k}^\mathrm{fast} = \exp\left(-\frac{dt}{\tau_{\mathrm{syn,fast},k}}\right)

       P_{21,k}^\mathrm{fast} = dt \cdot P_{11,k}^\mathrm{fast}

       P_{11,k}^\mathrm{slow} = P_{22,k}^\mathrm{slow} = \exp\left(-\frac{dt}{\tau_{\mathrm{syn,slow},k}}\right)

       P_{21,k}^\mathrm{slow} = dt \cdot P_{11,k}^\mathrm{slow}

    On a presynaptic spike of weight :math:`w` to receptor port :math:`k`:

    .. math::

       y_{1,k}^\mathrm{fast} \leftarrow y_{1,k}^\mathrm{fast} + w \cdot \frac{e}{\tau_{\mathrm{syn,fast},k}}

       y_{1,k}^\mathrm{slow} \leftarrow y_{1,k}^\mathrm{slow} + w \cdot \frac{e}{\tau_{\mathrm{syn,slow},k}} \cdot \mathrm{amp\_slow}_k

    The total synaptic current is:

    .. math::

       I_\mathrm{syn,total} = \sum_k \left( y_{2,k}^\mathrm{fast} + y_{2,k}^\mathrm{slow} \right)

    **3. After-Spike Currents (GLIF3/4/5)**

    After-spike currents (ASC) model slow adaptation via exponentially decaying
    currents triggered by spikes. Each ASC component :math:`I_j` decays with rate
    :math:`k_j`:

    .. math::

       I_j(t+dt) = I_j(t) \cdot \exp(-k_j \cdot dt)

    The time-averaged ASC over a step (used for stable integration) is:

    .. math::

       \bar{I}_j = \frac{1 - \exp(-k_j \cdot dt)}{k_j \cdot dt} \cdot I_j(t)

    On spike, ASC values are reset:

    .. math::

       I_j \leftarrow \Delta I_j + I_j \cdot r_j \cdot \exp(-k_j \cdot t_\mathrm{ref})

    where :math:`\Delta I_j` is the jump amplitude and :math:`r_j` is the fraction
    coefficient.

    **4. Spike-Dependent Threshold (GLIF2/4/5)**

    The spike component of the threshold :math:`\theta_s` decays exponentially:

    .. math::

       \theta_s(t+dt) = \theta_s(t) \cdot \exp(-b_s \cdot dt)

    On spike, after refractory decay:

    .. math::

       \theta_s \leftarrow \theta_s \cdot \exp(-b_s \cdot t_\mathrm{ref}) + \Delta\theta_s

    Voltage reset uses a biologically defined rule:

    .. math::

       U \leftarrow f_v \cdot U_\mathrm{old} + V_\mathrm{add}

    where :math:`f_v` is the voltage fraction coefficient and :math:`V_\mathrm{add}`
    is the additive constant.

    **5. Voltage-Dependent Threshold (GLIF5)**

    The voltage component of the threshold :math:`\theta_v` evolves according to:

    .. math::

       \theta_v(t+dt) = \phi \cdot (U_\mathrm{old} - \beta) \cdot P_\mathrm{decay}
           + \frac{1}{P_{\theta,v}} \cdot \left(\theta_v(t)
               - \phi \cdot (U_\mathrm{old} - \beta)
               - \frac{a_v}{b_v} \cdot \beta \right)
           + \frac{a_v}{b_v} \cdot \beta

    where:

    .. math::

       \phi = \frac{a_v}{b_v - g/C_m}, \quad
       P_\mathrm{decay} = \exp\left(-\frac{g \cdot dt}{C_m}\right), \quad
       P_{\theta,v} = \exp(b_v \cdot dt), \quad
       \beta = \frac{I_e + I_\mathrm{stim} + I_\mathrm{ASC,sum}}{g}

    **6. Overall Threshold and Spike Condition**

    .. math::

       \theta_\mathrm{total} = \theta_\infty + \theta_s + \theta_v

    Spike condition (checked after voltage update):

    .. math::

       \text{spike} = \begin{cases}
       \text{True} & \text{if } U > \theta_\mathrm{total} \\
       \text{False} & \text{otherwise}
       \end{cases}

    **7. Numerical Integration and Update Order**

    The discrete-time update sequence per simulation step is:

    1. Record :math:`U_\mathrm{old}` (relative to :math:`E_L`).
    2. If not refractory:

       a. Decay spike threshold component :math:`\theta_s`.
       b. Compute time-averaged ASC :math:`\bar{I}_j` and decay ASC values.
       c. Update membrane potential :math:`U` (include fast/slow synaptic contributions).
       d. Compute voltage-dependent threshold component :math:`\theta_v` (using :math:`U_\mathrm{old}`).
       e. Update total threshold :math:`\theta_\mathrm{total}`.
       f. If :math:`U > \theta_\mathrm{total}`: emit spike, apply reset rules.

    3. If refractory: decrement refractory counter, hold :math:`U` at :math:`U_\mathrm{old}`.
    4. Update synaptic current state variables for both fast and slow components.
    5. Add incoming spike current jumps (scaled for fast/slow).
    6. Buffer external current input for next step.
    7. Save :math:`U_\mathrm{old}` for next step.

    Parameters
    ----------
    in_size : int, tuple of int
        Population shape (number of neurons). Scalars are interpreted as (n,).
    g : ArrayLike, optional
        Membrane (leak) conductance. Default: 9.43 nS. Must be strictly positive.
        Shape: scalar or broadcastable to ``in_size``.
    E_L : ArrayLike, optional
        Resting (leak) membrane potential (absolute). Default: -78.85 mV.
        Shape: scalar or broadcastable to ``in_size``.
    V_th : ArrayLike, optional
        Instantaneous spike threshold (absolute). Default: -51.68 mV.
        Must be greater than ``V_reset``. Shape: scalar or broadcastable to ``in_size``.
    C_m : ArrayLike, optional
        Membrane capacitance. Default: 58.72 pF. Must be strictly positive.
        Shape: scalar or broadcastable to ``in_size``.
    t_ref : ArrayLike, optional
        Absolute refractory period. Default: 3.75 ms. Must be strictly positive.
        Shape: scalar or broadcastable to ``in_size``.
    V_reset : ArrayLike, optional
        Reset potential (absolute; used in GLIF1/3). Default: -78.85 mV.
        Must be less than ``V_th``. Shape: scalar or broadcastable to ``in_size``.
    th_spike_add : float, optional
        Threshold additive constant after spike (:math:`\Delta\theta_s`).
        Default: 0.37 mV. Used in GLIF2/4/5.
    th_spike_decay : float, optional
        Spike threshold decay rate (:math:`b_s`). Default: 0.009 /ms.
        Must be strictly positive. Used in GLIF2/4/5.
    voltage_reset_fraction : float, optional
        Voltage fraction coefficient after spike (:math:`f_v`).
        Default: 0.20. Must be in [0.0, 1.0]. Used in GLIF2/4/5.
    voltage_reset_add : float, optional
        Voltage additive constant after spike (:math:`V_\mathrm{add}`).
        Default: 18.51 mV. Used in GLIF2/4/5.
    th_voltage_index : float, optional
        Voltage-dependent threshold leak rate (:math:`a_v`). Default: 0.005 /ms.
        Used in GLIF5.
    th_voltage_decay : float, optional
        Voltage-dependent threshold decay rate (:math:`b_v`). Default: 0.09 /ms.
        Must be strictly positive. Used in GLIF5.
    asc_init : Sequence[float], optional
        Initial values of after-spike current components (pA). Default: (0.0, 0.0).
        Length must match ``asc_decay``, ``asc_amps``, ``asc_r``. Used in GLIF3/4/5.
    asc_decay : Sequence[float], optional
        After-spike current decay rates (:math:`k_j`, /ms). Default: (0.003, 0.1).
        All values must be strictly positive. Used in GLIF3/4/5.
    asc_amps : Sequence[float], optional
        After-spike current jump amplitudes (:math:`\Delta I_j`, pA). Default: (-9.18, -198.94).
        Used in GLIF3/4/5.
    asc_r : Sequence[float], optional
        After-spike current fraction coefficients (:math:`r_j`). Default: (1.0, 1.0).
        All values must be in [0.0, 1.0]. Used in GLIF3/4/5.
    tau_syn_fast : Sequence[float], optional
        Fast synaptic alpha-function time constants (ms). Default: (2.0,).
        All values must be strictly positive. Length determines number of receptor ports.
    tau_syn_slow : Sequence[float], optional
        Slow synaptic alpha-function time constants (ms). Default: (6.0,).
        All values must be strictly positive. Length must match ``tau_syn_fast``.
    amp_slow : Sequence[float], optional
        Relative amplitude of slow component (unitless). Default: (0.3,).
        All values must be strictly positive. Length must match ``tau_syn_fast``.
    spike_dependent_threshold : bool, optional
        Enable biologically defined reset rules (GLIF2/4/5). Default: False.
    after_spike_currents : bool, optional
        Enable after-spike currents (GLIF3/4/5). Default: False.
    adapting_threshold : bool, optional
        Enable voltage-dependent threshold (GLIF5). Default: False.
    I_e : ArrayLike, optional
        Constant external current input (pA). Default: 0.0 pA.
        Shape: scalar or broadcastable to ``in_size``.
    V_initializer : Callable, optional
        Membrane potential initializer. Default: ``Constant(E_L)``.
        Should return values in mV when called with shape and batch_size.
    spk_fun : Callable, optional
        Surrogate gradient function for differentiable spike generation.
        Default: ``ReluGrad()``. Must accept scaled voltage and return spike output.
    spk_reset : str, optional
        Spike reset mode. Default: ``'hard'`` (stop gradient). Alternative: ``'soft'``.
    name : str, optional
        Name of this neuron population.


    Parameter Mapping
    -----------------

    =============================== ========================== ========================================== =====================================================
    **Parameter**                   **Default**                **Math equivalent**                        **Description**
    =============================== ========================== ========================================== =====================================================
    ``in_size``                     (required)                 —                                          Population shape
    ``g``                           9.43 nS                    :math:`g`                                  Membrane (leak) conductance
    ``E_L``                         -78.85 mV                  :math:`E_L`                                Resting membrane potential
    ``V_th``                        -51.68 mV                  :math:`V_\mathrm{th}`                      Instantaneous threshold (absolute)
    ``C_m``                         58.72 pF                   :math:`C_m`                                Membrane capacitance
    ``t_ref``                       3.75 ms                    :math:`t_\mathrm{ref}`                     Absolute refractory period
    ``V_reset``                     -78.85 mV                  :math:`V_\mathrm{reset}`                   Reset potential (absolute; GLIF1/3)
    ``th_spike_add``                0.37 mV                    :math:`\Delta\theta_s`                     Threshold additive constant after spike
    ``th_spike_decay``              0.009 /ms                  :math:`b_s`                                Spike threshold decay rate
    ``voltage_reset_fraction``      0.20                       :math:`f_v`                                Voltage fraction after spike
    ``voltage_reset_add``           18.51 mV                   :math:`V_\mathrm{add}`                     Voltage additive after spike
    ``th_voltage_index``            0.005 /ms                  :math:`a_v`                                Voltage-dependent threshold leak
    ``th_voltage_decay``            0.09 /ms                   :math:`b_v`                                Voltage-dependent threshold decay rate
    ``asc_init``                    (0.0, 0.0) pA              :math:`I_j(0)`                             Initial values of ASC
    ``asc_decay``                   (0.003, 0.1) /ms           :math:`k_j`                                ASC decay rates
    ``asc_amps``                    (-9.18, -198.94) pA        :math:`\Delta I_j`                         ASC amplitudes on spike
    ``asc_r``                       (1.0, 1.0)                 :math:`r_j`                                ASC fraction coefficient
    ``tau_syn_fast``                (2.0,) ms                  :math:`\tau_{\mathrm{syn,fast},k}`         Fast synaptic alpha-function time constants
    ``tau_syn_slow``                (6.0,) ms                  :math:`\tau_{\mathrm{syn,slow},k}`         Slow synaptic alpha-function time constants
    ``amp_slow``                    (0.3,)                     :math:`\mathrm{amp\_slow}_k`               Relative amplitude of slow component
    ``spike_dependent_threshold``   False                      —                                          Enable biologically defined reset (GLIF2/4/5)
    ``after_spike_currents``        False                      —                                          Enable after-spike currents (GLIF3/4/5)
    ``adapting_threshold``          False                      —                                          Enable voltage-dependent threshold (GLIF5)
    ``I_e``                         0.0 pA                     :math:`I_e`                                Constant external current
    ``V_initializer``               Constant(E_L)              —                                          Membrane potential initializer
    ``spk_fun``                     ReluGrad()                 —                                          Surrogate spike function
    ``spk_reset``                   ``'hard'``                 —                                          Reset mode (``'hard'`` or ``'soft'``)
    =============================== ========================== ========================================== =====================================================

    Notes
    -----
    - **Default parameters** are from GLIF Model 5 of Cell 490626718 in the
      Allen Cell Type Database (https://celltypes.brain-map.org).
    - **Voltage tracking**: ``V_th`` and ``V_reset`` are specified in absolute mV.
      Internally, membrane potential is stored relative to ``E_L`` (matching NEST).
    - **Stability constraint** for GLIF2/4/5: The reset condition should satisfy:

      .. math::

          E_L + f_v \cdot (V_\mathrm{th} - E_L) + V_\mathrm{add} < V_\mathrm{th} + \Delta\theta_s

      Otherwise, the neuron may spike continuously.
    - **Numerical integration**: Uses exact integration via propagator matrices
      (matching NEST), unlike ``glif_cond`` which uses RKF45 ODE integration.
    - **Singularity handling**: If :math:`\tau_m \approx \tau_{\mathrm{syn,fast}}`
      or :math:`\tau_m \approx \tau_{\mathrm{syn,slow}}`, the model automatically
      applies singularity-safe formulas (see NEST IAF_Integration_Singularity notebook).
    - **Synaptic waveform control**: The double alpha function provides more flexible
      control over synaptic current shape compared to single alpha (``glif_psc``).
      By tuning ``tau_syn_fast``, ``tau_syn_slow``, and ``amp_slow``, experimentally
      observed waveforms can be matched.
    - **Receptor port indexing**: Synaptic inputs are registered via
      ``add_delta_input()`` with labels like ``'receptor_0'``, ``'receptor_1'``, etc.
      Inputs without explicit receptor labels default to receptor 0.
    - **State persistence**: After-spike current values (``_ASCurrents``), threshold
      components (``_threshold_spike``, ``_threshold_voltage``), and total threshold
      (``_threshold``) are stored as NumPy arrays (not JAX arrays) to match NEST's
      state handling and allow in-place updates during the per-neuron loop.

    Examples
    --------
    **1. GLIF Model 1 (basic LIF) with single receptor:**

    .. code-block:: python

        >>> import brainpy.state as st
        >>> import brainstate as bst
        >>> import saiunit as u
        >>> bst.environ.set(dt=0.1 * u.ms)
        >>> neurons = st.glif_psc_double_alpha(
        ...     in_size=100,
        ...     g=10.0 * u.nS,
        ...     E_L=-70.0 * u.mV,
        ...     V_th=-55.0 * u.mV,
        ...     C_m=250.0 * u.pF,
        ...     t_ref=2.0 * u.ms,
        ...     V_reset=-70.0 * u.mV,
        ...     tau_syn_fast=(2.0,) * u.ms,
        ...     tau_syn_slow=(6.0,) * u.ms,
        ...     amp_slow=(0.5,),
        ... )
        >>> neurons.init_all_states()
        >>> spikes = neurons.update(10.0 * u.pA)

    **2. GLIF Model 5 (full model) with multiple receptors:**

    .. code-block:: python

        >>> neurons = st.glif_psc_double_alpha(
        ...     in_size=50,
        ...     spike_dependent_threshold=True,
        ...     after_spike_currents=True,
        ...     adapting_threshold=True,
        ...     tau_syn_fast=(1.0, 3.0) * u.ms,  # Two receptor ports
        ...     tau_syn_slow=(5.0, 10.0) * u.ms,
        ...     amp_slow=(0.3, 0.4),
        ...     asc_decay=(0.01, 0.05) / u.ms,
        ...     asc_amps=(-10.0, -100.0) * u.pA,
        ... )
        >>> neurons.init_all_states()
        >>> # Synaptic inputs can target different receptors
        >>> neurons.add_delta_input('excitatory_receptor_0')
        >>> neurons.add_delta_input('inhibitory_receptor_1')

    **3. Accessing synaptic current components:**

    .. code-block:: python

        >>> I_syn_total = neurons.get_I_syn()
        >>> I_syn_fast = neurons.get_I_syn_fast()
        >>> I_syn_slow = neurons.get_I_syn_slow()

    References
    ----------
    .. [1] Teeter C, Iyer R, Menon V, Gouwens N, Feng D, Berg J, Szafer A,
           Cain N, Zeng H, Hawrylycz M, Koch C, & Mihalas S (2018).
           Generalized leaky integrate-and-fire models classify multiple neuron
           types. Nature Communications 9:709.
           DOI: 10.1038/s41467-017-02717-4
    .. [2] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. J. Comput. Neurosci. 16:159-175.
           DOI: 10.1023/B:JCNS.0000014108.03012.81
    .. [3] NEST Simulator ``glif_psc_double_alpha`` model documentation and C++
           source: ``models/glif_psc_double_alpha.h`` and
           ``models/glif_psc_double_alpha.cpp`` in NEST repository.

    See Also
    --------
    glif_psc : Single alpha-function variant.
    glif_cond : Conductance-based GLIF using ODE integration.
    gif_psc_exp_multisynapse : Generalized IF with exponential PSCs and multisynapse support.
    aeif_psc_alpha : Adaptive exponential IF with alpha PSCs.
    """
    __module__ = 'brainpy.state'

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        g: ArrayLike = 9.43 * u.nS,
        E_L: ArrayLike = -78.85 * u.mV,
        V_th: ArrayLike = -51.68 * u.mV,
        C_m: ArrayLike = 58.72 * u.pF,
        t_ref: ArrayLike = 3.75 * u.ms,
        V_reset: ArrayLike = -78.85 * u.mV,
        th_spike_add: float = 0.37,  # mV
        th_spike_decay: float = 0.009,  # 1/ms
        voltage_reset_fraction: float = 0.20,
        voltage_reset_add: float = 18.51,  # mV
        th_voltage_index: float = 0.005,  # 1/ms
        th_voltage_decay: float = 0.09,  # 1/ms
        asc_init: Sequence[float] = (0.0, 0.0),  # pA
        asc_decay: Sequence[float] = (0.003, 0.1),  # 1/ms
        asc_amps: Sequence[float] = (-9.18, -198.94),  # pA
        asc_r: Sequence[float] = (1.0, 1.0),
        tau_syn_fast: Sequence[float] = (2.0,),  # ms
        tau_syn_slow: Sequence[float] = (6.0,),  # ms
        amp_slow: Sequence[float] = (0.3,),  # unitless
        spike_dependent_threshold: bool = False,
        after_spike_currents: bool = False,
        adapting_threshold: bool = False,
        I_e: ArrayLike = 0.0 * u.pA,
        gsl_error_tol: ArrayLike = 1e-6,
        V_initializer: Callable = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Store membrane parameters
        self.g_m = braintools.init.param(g, self.varshape)
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        # V_th and V_reset are absolute; store th_inf_ relative to E_L (like NEST)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)

        # Scalar GLIF parameters (unitless floats in NEST units)
        self.th_spike_add = float(th_spike_add)
        self.th_spike_decay = float(th_spike_decay)
        self.voltage_reset_fraction = float(voltage_reset_fraction)
        self.voltage_reset_add = float(voltage_reset_add)
        self.th_voltage_index = float(th_voltage_index)
        self.th_voltage_decay = float(th_voltage_decay)

        # ASC parameters (lists of floats)
        self.asc_init = tuple(float(x) for x in asc_init)
        self.asc_decay = tuple(float(x) for x in asc_decay)
        self.asc_amps = tuple(float(x) for x in asc_amps)
        self.asc_r = tuple(float(x) for x in asc_r)

        # Synaptic parameters (double alpha: fast and slow components)
        self.tau_syn_fast = tuple(float(x) for x in tau_syn_fast)
        self.tau_syn_slow = tuple(float(x) for x in tau_syn_slow)
        self.amp_slow = tuple(float(x) for x in amp_slow)

        # Model mechanism flags
        self.has_theta_spike = bool(spike_dependent_threshold)
        self.has_asc = bool(after_spike_currents)
        self.has_theta_voltage = bool(adapting_threshold)

        # Default V_initializer to E_L
        if V_initializer is None:
            V_initializer = braintools.init.Constant(E_L)
        self.V_initializer = V_initializer

        self._n_receptors = len(self.tau_syn_fast)
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

    @property
    def n_receptors(self):
        r"""Number of synaptic receptor ports.

        Returns the number of distinct receptor ports configured for this neuron
        population. Each receptor port has independent fast and slow alpha-function
        current dynamics, allowing modeling of multiple synaptic receptor types
        (e.g., AMPA, NMDA, GABA_A, GABA_B).

        Returns
        -------
        int
            Number of receptor ports, determined by the length of ``tau_syn_fast``
            (which must match the lengths of ``tau_syn_slow`` and ``amp_slow``).

        Notes
        -----
        - Receptor ports are indexed from 0 to ``n_receptors - 1``.
        - Projections target specific receptors via labels like ``'receptor_0'``,
          ``'receptor_1'``, etc.
        - By default (single-element arrays for synaptic parameters), ``n_receptors == 1``.

        See Also
        --------
        _collect_receptor_delta_inputs : Routes synaptic inputs to receptor ports.
        """
        return self._n_receptors

    def _validate_parameters(self):
        # Check valid model mechanism combinations
        s, a, v = self.has_theta_spike, self.has_asc, self.has_theta_voltage
        valid_combos = [
            (False, False, False),  # GLIF1
            (True, False, False),  # GLIF2
            (False, True, False),  # GLIF3
            (True, True, False),  # GLIF4
            (True, True, True),  # GLIF5
        ]
        if (s, a, v) not in valid_combos:
            raise ValueError(
                "Incorrect model mechanism combination. "
                "Valid combinations: GLIF1(FFF), GLIF2(TFF), GLIF3(FTF), "
                "GLIF4(TTF), GLIF5(TTT). Got spike_dependent_threshold=%s, "
                "after_spike_currents=%s, adapting_threshold=%s." % (s, a, v)
            )

        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_reset, self.C_m)):
            return

        # V_reset (relative) < V_th (relative) — both relative to E_L
        E_L_val = self.E_L
        V_reset_rel = self.V_reset - E_L_val
        V_th_rel = self.V_th - E_L_val
        if np.any(V_reset_rel >= V_th_rel):
            raise ValueError("Reset potential must be smaller than threshold.")

        if np.any(self.C_m <= 0.0 * u.pF):
            raise ValueError("Capacitance must be strictly positive.")
        if np.any(self.g_m <= 0.0 * u.nS):
            raise ValueError("Membrane conductance must be strictly positive.")
        if np.any(self.t_ref <= 0.0 * u.ms):
            raise ValueError("Refractory time constant must be strictly positive.")

        if self.has_theta_spike:
            if self.th_spike_decay <= 0.0:
                raise ValueError("Spike induced threshold time constant must be strictly positive.")
            if not (0.0 <= self.voltage_reset_fraction <= 1.0):
                raise ValueError("Voltage fraction coefficient following spike must be within [0.0, 1.0].")

        if self.has_asc:
            n = len(self.asc_decay)
            if not (len(self.asc_init) == n and len(self.asc_amps) == n and len(self.asc_r) == n):
                raise ValueError(
                    "All after spike current parameters (asc_init, asc_decay, asc_amps, asc_r) "
                    "must have the same size."
                )
            for k_val in self.asc_decay:
                if k_val <= 0.0:
                    raise ValueError("After-spike current time constant must be strictly positive.")
            for r_val in self.asc_r:
                if not (0.0 <= r_val <= 1.0):
                    raise ValueError(
                        "After spike current fraction coefficients r must be within [0.0, 1.0]."
                    )

        if self.has_theta_voltage:
            if self.th_voltage_decay <= 0.0:
                raise ValueError("Voltage-induced threshold time constant must be strictly positive.")

        # Check synaptic parameter sizes
        n_rec = len(self.tau_syn_fast)
        if len(self.tau_syn_slow) != n_rec:
            raise ValueError(
                f"tau_syn_slow must have same length as tau_syn_fast ({n_rec}), "
                f"got {len(self.tau_syn_slow)}."
            )
        if len(self.amp_slow) != n_rec:
            raise ValueError(
                f"amp_slow must have same length as tau_syn_fast ({n_rec}), "
                f"got {len(self.amp_slow)}."
            )

        for tau in self.tau_syn_fast:
            if tau <= 0.0:
                raise ValueError("All fast synaptic time constants must be strictly positive.")
        for tau in self.tau_syn_slow:
            if tau <= 0.0:
                raise ValueError("All slow synaptic time constants must be strictly positive.")
        for amp in self.amp_slow:
            if amp <= 0.0:
                raise ValueError("All slow synaptic amplitudes must be strictly positive.")

        if np.any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Creates and initializes all state variables required for GLIF dynamics,
        including membrane potential, synaptic current states (fast and slow
        components for each receptor port), threshold components, after-spike
        current values, refractory counters, and buffered input current.

        **State Variables Created**

        - ``self.V``: Membrane potential (HiddenState, saiunit.mV)
        - ``self.y1_fast[k]``: Fast alpha-function derivative state for receptor k (HiddenState, saiunit.pA)
        - ``self.y2_fast[k]``: Fast alpha-function current for receptor k (HiddenState, saiunit.pA)
        - ``self.y1_slow[k]``: Slow alpha-function derivative state for receptor k (HiddenState, saiunit.pA)
        - ``self.y2_slow[k]``: Slow alpha-function current for receptor k (HiddenState, saiunit.pA)
        - ``self.last_spike_time``: Last spike time (ShortTermState, saiunit.ms)
        - ``self.refractory_step_count``: Remaining refractory steps (ShortTermState, int32)
        - ``self.I_stim``: Buffered external current (ShortTermState, saiunit.pA)
        - ``self._ASCurrents``: After-spike current values (NumPy array, pA)
        - ``self._ASCurrents_sum``: Sum of after-spike currents (NumPy array, pA)
        - ``self._threshold_spike``: Spike component of threshold (NumPy array, mV relative to E_L)
        - ``self._threshold_voltage``: Voltage component of threshold (NumPy array, mV relative to E_L)
        - ``self._threshold``: Total threshold (NumPy array, mV relative to E_L)

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        - **Membrane potential** is initialized using ``self.V_initializer`` (default: ``E_L``).
        - **Synaptic current states** (``y1_fast``, ``y2_fast``, ``y1_slow``, ``y2_slow``)
          are initialized to 0.0 pA for all receptor ports.
        - **After-spike currents** are initialized to ``asc_init`` values.
        - **Threshold components** are initialized to 0.0 (spike/voltage components)
          and ``V_th - E_L`` (baseline threshold).
        - **Refractory counter** is initialized to 0 (not refractory).
        - **Last spike time** is initialized to -1e7 ms (effectively never spiked).
        - **State storage**: Hidden states (membrane potential, synaptic currents)
          are stored as JAX arrays for gradient computation. Threshold and ASC
          values are stored as NumPy arrays to match NEST's state handling.

        See Also
        --------
        reset_state : Reset state variables to initial conditions during simulation.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        self.V = brainstate.HiddenState(V)

        # Per-receptor alpha-function current states: fast component (y1_fast rate pA/ms, y2_fast current pA)
        self.y1_fast = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA / u.ms), self.varshape)
            )
            for _ in range(self._n_receptors)
        ]
        self.y2_fast = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape)
            )
            for _ in range(self._n_receptors)
        ]

        # Per-receptor alpha-function current states: slow component (y1_slow rate pA/ms, y2_slow current pA)
        self.y1_slow = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA / u.ms), self.varshape)
            )
            for _ in range(self._n_receptors)
        ]
        self.y2_slow = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape)
            )
            for _ in range(self._n_receptors)
        ]

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        # GLIF-specific state (stored as plain numpy, matching NEST)
        # ASC values
        v_shape = self.varshape
        n_asc = len(self.asc_decay)
        self._ASCurrents = np.zeros((n_asc, *v_shape), dtype=dftype)
        for a in range(n_asc):
            self._ASCurrents[a] = self.asc_init[a]
        self._ASCurrents_sum = np.sum(self._ASCurrents, axis=0) if n_asc > 0 else np.zeros(v_shape, dtype=dftype)

        # Threshold components (relative to E_L)
        E_L_mV = float(np.asarray(u.math.asarray(self.E_L / u.mV), dtype=dftype))
        th_inf = float(np.asarray(u.math.asarray(self.V_th / u.mV), dtype=dftype)) - E_L_mV
        self._th_inf = th_inf
        self._threshold_spike = np.zeros(v_shape, dtype=dftype)
        self._threshold_voltage = np.zeros(v_shape, dtype=dftype)
        self._threshold = np.full(v_shape, th_inf, dtype=dftype)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute spike output from membrane potential using surrogate gradient function.

        Applies the surrogate gradient function (``self.spk_fun``) to a scaled version
        of the membrane potential to produce a differentiable spike signal. This
        method computes the spike output **without updating state**, making it useful
        for inspection or custom integration schemes.

        **Scaling**

        The membrane potential is scaled to the range where the surrogate function
        is most sensitive:

        .. math::

            v_\mathrm{scaled} = \frac{V - V_\mathrm{th}}{V_\mathrm{th} - V_\mathrm{reset}}

        This normalization ensures that:

        - When :math:`V = V_\mathrm{th}`, :math:`v_\mathrm{scaled} = 0`
        - When :math:`V = V_\mathrm{reset}`, :math:`v_\mathrm{scaled} = -1`

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (absolute, in mV). If ``None`` (default), uses
            the current state ``self.V.value``. If provided, should have shape
            compatible with ``self.varshape`` (or ``(batch_size, *self.varshape)``).
            Unit: ``saiunit.mV`` or dimensionless (interpreted as mV).

        Returns
        -------
        spike : jax.Array
            Spike output computed via surrogate gradient function.
            Shape: same as input ``V``.
            Dtype: same as input ``V`` (typically ``jnp.float32``).
            Values: Continuous in [0, 1] for most surrogate functions (e.g., ``ReluGrad``,
            ``SigmoidGrad``), though exact range depends on ``self.spk_fun``.

        Notes
        -----
        - This method is used internally by ``update()`` to compute spike output
          after the membrane potential update.
        - The surrogate gradient function ensures gradients can flow through spike
          events during backpropagation, enabling gradient-based training of spiking
          neural networks.
        - The scaling factor :math:`(V_\mathrm{th} - V_\mathrm{reset})` normalizes
          the input to the surrogate function, improving numerical stability and
          gradient flow.

        See Also
        --------
        update : Main simulation step, which calls this method internally.
        braintools.surrogate.ReluGrad : Default surrogate gradient function.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, y1_fast_<k>, y2_fast_<k>, y1_slow_<k>, y2_slow_<k>,
            theta_s, theta_v, asc_<j> — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_old — mutable
            auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        # Membrane potential relative to E_L
        V_rel = state.V - self.E_L

        # Compute total synaptic current (from fast and slow components)
        I_syn = u.math.zeros_like(state.V) * u.pA / u.mV  # will accumulate pA
        I_syn = u.math.zeros_like(self.I_e)
        for k in range(self._n_receptors):
            I_syn = I_syn + state[f'y2_fast_{k}'] + state[f'y2_slow_{k}']

        # Compute total ASC current
        I_asc = u.math.zeros_like(self.I_e)
        if self.has_asc:
            n_asc = len(self.asc_decay)
            for j in range(n_asc):
                I_asc = I_asc + state[f'asc_{j}']

        # Membrane dynamics: C_m * dV/dt = -g*(V - E_L) + I_syn + I_e + I_stim + I_asc
        dV_raw = (-self.g_m * V_rel + I_syn + self.I_e + extra.i_stim + I_asc) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        derivs = DotDict(V=dV)

        # Synaptic current derivatives (alpha function ODEs)
        for k in range(self._n_receptors):
            tau_fast = self.tau_syn_fast[k] * u.ms
            tau_slow = self.tau_syn_slow[k] * u.ms

            # Fast: dy1/dt = -y1/tau, dy2/dt = y1 - y2/tau
            dy1_fast = -state[f'y1_fast_{k}'] / tau_fast
            dy2_fast = state[f'y1_fast_{k}'] - state[f'y2_fast_{k}'] / tau_fast

            # Slow: dy1/dt = -y1/tau, dy2/dt = y1 - y2/tau
            dy1_slow = -state[f'y1_slow_{k}'] / tau_slow
            dy2_slow = state[f'y1_slow_{k}'] - state[f'y2_slow_{k}'] / tau_slow

            derivs[f'y1_fast_{k}'] = dy1_fast
            derivs[f'y2_fast_{k}'] = dy2_fast
            derivs[f'y1_slow_{k}'] = dy1_slow
            derivs[f'y2_slow_{k}'] = dy2_slow

        # Threshold spike decay: dtheta_s/dt = -b_s * theta_s
        if self.has_theta_spike:
            derivs['theta_s'] = -self.th_spike_decay / u.ms * state.theta_s
        else:
            derivs['theta_s'] = u.math.zeros_like(state.theta_s) / u.ms

        # Threshold voltage: dtheta_v/dt = a_v*(V_rel/mV) - b_v*theta_v (in mV/ms)
        if self.has_theta_voltage:
            V_rel_mV = V_rel / u.mV
            derivs['theta_v'] = (self.th_voltage_index * V_rel_mV - self.th_voltage_decay * state.theta_v) / u.ms
        else:
            derivs['theta_v'] = u.math.zeros_like(state.theta_v) / u.ms

        # ASC decay: dI_j/dt = -k_j * I_j
        if self.has_asc:
            for j in range(len(self.asc_decay)):
                derivs[f'asc_{j}'] = -self.asc_decay[j] / u.ms * state[f'asc_{j}']
        else:
            for j in range(len(self.asc_decay)):
                derivs[f'asc_{j}'] = u.math.zeros_like(state[f'asc_{j}']) / u.ms

        return derivs

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, y1_fast_<k>, y2_fast_<k>, y1_slow_<k>, y2_slow_<k>,
            theta_s, theta_v, asc_<j> — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_old.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & ((state.V < -1e3 * u.mV) | (state.V > 1e3 * u.mV))
        )

        # Refractory clamp
        refr_accept = accept & (extra.r > 0)
        new_V = u.math.where(refr_accept, extra.v_old, state.V)

        # Compute effective threshold (in mV, relative to E_L)
        # th_inf + theta_s + theta_v, all in mV relative to E_L
        # V_th is absolute, so threshold_abs = V_th + theta_s*mV + theta_v*mV
        threshold_abs = self.V_th + state.theta_s * u.mV + state.theta_v * u.mV

        # Spike detection
        spike_now = accept & (extra.r <= 0) & (new_V >= threshold_abs)
        spike_mask = extra.spike_mask | spike_now

        # Reset voltage
        V_rel_old = extra.v_old - self.E_L
        if not self.has_theta_spike:
            # GLIF1/3: simple reset
            reset_V = self.V_reset
        else:
            # GLIF2/4/5: biologically defined reset
            reset_V = self.voltage_reset_fraction * V_rel_old + self.voltage_reset_add * u.mV + self.E_L

        new_V = u.math.where(spike_now, reset_V, new_V)

        # Reset spike threshold on spike (GLIF2/4/5)
        new_theta_s = state.theta_s
        if self.has_theta_spike:
            t_ref_ms = self.t_ref / u.ms
            theta_s_after_ref = state.theta_s * u.math.exp(-self.th_spike_decay * t_ref_ms) + self.th_spike_add
            new_theta_s = u.math.where(spike_now, theta_s_after_ref, state.theta_s)

        # Reset ASC on spike (GLIF3/4/5)
        new_state_dict = {**state, 'V': new_V, 'theta_s': new_theta_s}
        if self.has_asc:
            t_ref_ms = self.t_ref / u.ms
            for j in range(len(self.asc_decay)):
                asc_refractory_decay = self.asc_r[j] * u.math.exp(-self.asc_decay[j] * t_ref_ms)
                asc_reset = self.asc_amps[j] * u.pA + state[f'asc_{j}'] * asc_refractory_decay
                new_state_dict[f'asc_{j}'] = u.math.where(spike_now, asc_reset, state[f'asc_{j}'])

        # Set refractory counter
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict(new_state_dict)
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

    def update(self, x=0.0 * u.pA):
        r"""Perform a single simulation step for all neurons in the population.

        This method implements the full GLIF update sequence using exact integration
        via precomputed propagator matrices. The update follows NEST's discrete-time
        integration scheme with careful ordering of subthreshold dynamics, threshold
        adaptation, spike detection, and reset rules.

        **Update Sequence**

        1. Extract and preprocess state variables (convert to NumPy float64 arrays).
        2. Precompute propagator matrix elements for membrane and synaptic dynamics.
        3. For each neuron:

           a. Check refractory status.
           b. If not refractory:

              i. Decay spike threshold component.
              ii. Compute time-averaged after-spike currents and decay ASC values.
              iii. Update membrane potential via exact integration (including fast/slow synaptic contributions).
              iv. Compute voltage-dependent threshold component.
              v. Update total threshold.
              vi. Check spike condition and apply reset rules if threshold crossed.

           c. If refractory: decrement counter, hold voltage.

        4. Update synaptic current state variables for all receptor ports (fast and slow components).
        5. Add incoming spike events (delta inputs) to synaptic state variables.
        6. Buffer external current input for next time step.
        7. Write updated state back to ``self.V``, ``self.y1_fast``, etc.

        **Numerical Integration Details**

        - Membrane potential uses exact integration: :math:`U(t+dt) = P_{33} U(t) + \ldots`
        - Synaptic currents use exact integration of alpha-function ODEs.
        - Threshold components (spike-dependent, voltage-dependent) use exact exponential decay.
        - After-spike currents use exact exponential decay with time-averaged contributions.
        - All exact integration coefficients are precomputed before the per-neuron loop.

        **Spike Detection and Reset**

        - Spike condition: :math:`U > \theta_\mathrm{total}` after voltage update.
        - Spike output is computed via surrogate gradient function for differentiability.
        - Reset rules depend on model variant (GLIF1/3: simple reset; GLIF2/4/5: biologically defined).
        - Refractory period is enforced by setting a grid step counter.

        **Performance Notes**

        - Uses explicit NumPy loops over neuron indices (``np.ndindex``) for clarity
          and exact NEST equivalence, rather than vectorized operations.
        - For large populations, consider using GPU-accelerated alternatives (though
          this implementation prioritizes numerical fidelity to NEST).

        Parameters
        ----------
        x : ArrayLike, optional
            External current input (picoamperes). Default: 0.0 pA.
            Applied with **one-step delay** (buffered in ``self.I_stim``).
            Shape: scalar or broadcastable to population shape (excluding batch dimension).
            Unit: ``saiunit.pA`` or dimensionless (interpreted as pA).

        Returns
        -------
        spike : jax.Array
            Spike output for current time step, computed via surrogate gradient function.
            Shape: same as ``self.V.value.shape`` (including batch dimension if present).
            Dtype: ``jnp.float32``.
            Values: Continuous in [0, 1] (for gradient computation), though semantically
            binary (0 = no spike, 1 = spike).

        Notes
        -----
        - **External current delay**: The input ``x`` is buffered for one time step.
          At step ``t``, the voltage update uses the ``x`` value from step ``t-1``.
          This matches NEST's event-driven handling of ``dc_generator`` devices.
        - **Delta inputs** (synaptic spikes) are applied immediately without delay,
          as they represent discrete events occurring at the current time step.
        - **State consistency**: All state variables are updated atomically at the
          end of the function to ensure consistency during gradient computation.
        - **Gradient flow**: The surrogate gradient function (``self.spk_fun``) ensures
          gradients can flow through spike events for training.

        Raises
        ------
        KeyError
            If simulation time ``t`` or timestep ``dt`` is not set in ``brainstate.environ``.

        See Also
        --------
        init_state : Initialize state variables before simulation.
        reset_state : Reset state variables to initial conditions.
        get_spike : Compute spike output from membrane potential (without state update).
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Build ODE state dict
        ode_state = DotDict(V=V)
        for k in range(self._n_receptors):
            ode_state[f'y1_fast_{k}'] = self.y1_fast[k].value
            ode_state[f'y2_fast_{k}'] = self.y2_fast[k].value
            ode_state[f'y1_slow_{k}'] = self.y1_slow[k].value
            ode_state[f'y2_slow_{k}'] = self.y2_slow[k].value

        # Threshold components (unitless mV relative to E_L, stored as plain floats)
        ode_state['theta_s'] = jnp.asarray(self._threshold_spike)
        ode_state['theta_v'] = jnp.asarray(self._threshold_voltage)

        # ASC state
        n_asc = len(self.asc_decay)
        for j in range(n_asc):
            ode_state[f'asc_{j}'] = jnp.asarray(self._ASCurrents[j]) * u.pA

        # Extra (mutable auxiliary data)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            v_old=V,
        )

        # Adaptive RKF45 integration via generic integrator.
        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)

        V = ode_state.V
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in glif_psc_double_alpha dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration).
        # Collect per-receptor delta inputs
        dy_inputs = []
        for k in range(self._n_receptors):
            label = f'receptor_{k}'
            w_k = self.sum_delta_inputs(u.math.zeros(self.varshape, dtype=dftype) * u.pA, label=label)
            dy_inputs.append(w_k)

        # Apply synaptic spike inputs to y1 states.
        y1_fast_vals = []
        y1_slow_vals = []
        for k in range(self._n_receptors):
            pscon_fast = np.e / (self.tau_syn_fast[k] * u.ms)  # 1/ms
            pscon_slow = np.e / (self.tau_syn_slow[k] * u.ms) * self.amp_slow[k]  # 1/ms

            y1_fast_k = ode_state[f'y1_fast_{k}'] + pscon_fast * dy_inputs[k]
            y1_slow_k = ode_state[f'y1_slow_{k}'] + pscon_slow * dy_inputs[k]
            y1_fast_vals.append(y1_fast_k)
            y1_slow_vals.append(y1_slow_k)

        # Write back state.
        self.V.value = V
        for k in range(self._n_receptors):
            self.y1_fast[k].value = y1_fast_vals[k]
            self.y2_fast[k].value = ode_state[f'y2_fast_{k}']
            self.y1_slow[k].value = y1_slow_vals[k]
            self.y2_slow[k].value = ode_state[f'y2_slow_{k}']

        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA

        # Write back threshold and ASC state (numpy arrays)
        self._threshold_spike = np.asarray(u.get_mantissa(ode_state.theta_s))
        self._threshold_voltage = np.asarray(u.get_mantissa(ode_state.theta_v))
        self._threshold = self._threshold_spike + self._threshold_voltage + self._th_inf

        if self.has_asc:
            for j in range(n_asc):
                self._ASCurrents[j] = np.asarray(u.get_mantissa(ode_state[f'asc_{j}'] / u.pA))
            self._ASCurrents_sum = np.sum(self._ASCurrents, axis=0)

        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return u.math.asarray(spike_mask, dtype=dftype)

    def get_I_syn(self):
        r"""Get the total synaptic current summed across all receptor ports and components.

        Computes the instantaneous total synaptic current by summing the fast and
        slow alpha-function current components (``y2_fast`` and ``y2_slow``) across
        all receptor ports. This represents the total postsynaptic current :math:`I_\mathrm{syn}`
        flowing into the membrane at the current time step.

        Returns
        -------
        I_syn : jax.Array
            Total synaptic current across all receptors (fast + slow).
            Shape: same as ``self.V.value.shape`` (including batch dimension if present).
            Unit: ``saiunit.pA`` (picoamperes).

        Notes
        -----
        - This method reads the current state of ``y2_fast[k]`` and ``y2_slow[k]``
          for all receptor ports ``k``, without modifying state.
        - For a population with :math:`N` receptor ports, the total current is:

          .. math::

              I_\mathrm{syn,total} = \sum_{k=0}^{N-1} \left( y_{2,k}^\mathrm{fast} + y_{2,k}^\mathrm{slow} \right)

        See Also
        --------
        get_I_syn_fast : Get only the fast component.
        get_I_syn_slow : Get only the slow component.
        """
        I_syn = 0.0 * u.pA
        for k in range(self._n_receptors):
            I_syn = I_syn + self.y2_fast[k].value + self.y2_slow[k].value
        return I_syn

    def get_I_syn_fast(self):
        r"""Get the fast component of synaptic current summed across all receptor ports.

        Computes the instantaneous fast synaptic current by summing the fast alpha-function
        current components (``y2_fast``) across all receptor ports. The fast component
        corresponds to synaptic currents with time constant ``tau_syn_fast``.

        Returns
        -------
        I_syn_fast : jax.Array
            Fast synaptic current across all receptors.
            Shape: same as ``self.V.value.shape`` (including batch dimension if present).
            Unit: ``saiunit.pA`` (picoamperes).

        Notes
        -----
        - For a population with :math:`N` receptor ports, the fast current is:

          .. math::

              I_\mathrm{syn,fast} = \sum_{k=0}^{N-1} y_{2,k}^\mathrm{fast}

        See Also
        --------
        get_I_syn : Get total synaptic current (fast + slow).
        get_I_syn_slow : Get only the slow component.
        """
        I_syn = 0.0 * u.pA
        for k in range(self._n_receptors):
            I_syn = I_syn + self.y2_fast[k].value
        return I_syn

    def get_I_syn_slow(self):
        r"""Get the slow component of synaptic current summed across all receptor ports.

        Computes the instantaneous slow synaptic current by summing the slow alpha-function
        current components (``y2_slow``) across all receptor ports. The slow component
        corresponds to synaptic currents with time constant ``tau_syn_slow``, scaled
        by amplitude factor ``amp_slow``.

        Returns
        -------
        I_syn_slow : jax.Array
            Slow synaptic current across all receptors.
            Shape: same as ``self.V.value.shape`` (including batch dimension if present).
            Unit: ``saiunit.pA`` (picoamperes).

        Notes
        -----
        - For a population with :math:`N` receptor ports, the slow current is:

          .. math::

              I_\mathrm{syn,slow} = \sum_{k=0}^{N-1} y_{2,k}^\mathrm{slow}

        - The slow component typically models NMDA-like or other slow synaptic processes.

        See Also
        --------
        get_I_syn : Get total synaptic current (fast + slow).
        get_I_syn_fast : Get only the fast component.
        """
        I_syn = 0.0 * u.pA
        for k in range(self._n_receptors):
            I_syn = I_syn + self.y2_slow[k].value
        return I_syn
