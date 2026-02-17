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

import math
from typing import Callable, Sequence

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'glif_psc_double_alpha',
]


def _iaf_propagator_alpha(tau_syn, tau_m, c_m, h):
    r"""Compute exact integration propagator elements P31, P32 for alpha-PSC.

    This function mirrors NEST's ``IAFPropagatorAlpha::evaluate()`` algorithm,
    implementing exact integration coefficients for the subthreshold linear
    membrane dynamics coupled with alpha-function shaped postsynaptic currents.
    It includes singularity handling when :math:`\tau_m \approx \tau_{syn}` to
    avoid numerical instabilities.

    The propagator maps the synaptic current state variables :math:`(y_1, y_2)`
    to membrane voltage changes per time step. The state :math:`y_1` represents
    the derivative of the alpha-function current, and :math:`y_2` is the current
    itself.

    **Mathematical Formulation**

    **1. Regular Case** — When :math:`\tau_m` and :math:`\tau_{syn}` differ
    sufficiently:

    Define intermediate parameters:

    .. math::

        \beta = \frac{\tau_{syn} \cdot \tau_m}{\tau_m - \tau_{syn}}, \quad
        \gamma = \frac{\beta}{C_m}

    Then the propagator elements are:

    .. math::

        P_{32} = \gamma \cdot \exp\left(-\frac{h}{\tau_{syn}}\right) \cdot
                 \left( \exp\left(\frac{h}{\beta}\right) - 1 \right)

    .. math::

        P_{31} = \gamma \cdot \exp\left(-\frac{h}{\tau_{syn}}\right) \cdot
                 \left( \beta \cdot \left( \exp\left(\frac{h}{\beta}\right) - 1 \right) - h \right)

    **2. Singular Case** — When :math:`\tau_m \approx \tau_{syn}`, the regular
    formula becomes numerically unstable. The limiting forms are:

    .. math::

        P_{32,\text{singular}} = \frac{h}{C_m} \cdot \exp\left(-\frac{h}{\tau_m}\right)

    .. math::

        P_{31,\text{singular}} = \frac{h^2}{2 C_m} \cdot \exp\left(-\frac{h}{\tau_m}\right)

    The algorithm automatically detects when the singular case is needed by
    checking if the computed ``P32`` is numerically valid, or if
    :math:`h < h_{\text{min}} = 10^{-7} \cdot \tau_m^2 / |\tau_m - \tau_{syn}|`.

    **Numerical Stability**

    The implementation uses ``expm1(x) = exp(x) - 1`` for improved accuracy
    near zero. Explicit singularity checks prevent division by zero when
    :math:`\tau_m = \tau_{syn}` exactly, or when the result underflows.

    Parameters
    ----------
    tau_syn : float
        Synaptic time constant (milliseconds). Must be strictly positive.
        Determines the rise/decay rate of the alpha-function PSC.
    tau_m : float
        Membrane time constant (milliseconds), defined as :math:`C_m / g`.
        Must be strictly positive.
    c_m : float
        Membrane capacitance (picofarads). Must be strictly positive.
    h : float
        Simulation time step (milliseconds). Must be strictly positive.

    Returns
    -------
    P31 : float
        Propagator element mapping :math:`y_1` (alpha-function derivative,
        in pA/ms) to membrane voltage change (in mV). Unitless in NEST
        conventions (assumes consistent pA/pF/mV/ms units).
    P32 : float
        Propagator element mapping :math:`y_2` (alpha-function current, in pA)
        to membrane voltage change (in mV). Unitless in NEST conventions.

    Notes
    -----
    - This is a low-level utility function used by GLIF models during
      initialization to precompute propagator matrix elements for efficient
      exact integration.
    - The function does not validate inputs; callers must ensure all parameters
      are positive and finite.
    - When :math:`\tau_m = \tau_{syn}` exactly, the function handles the
      singularity by returning the limiting forms analytically.
    - For further details on the singularity handling, see NEST's documentation
      notebook "IAF_Integration_Singularity".

    See Also
    --------
    glif_psc_double_alpha : Uses this function to compute propagators for
                             fast and slow synaptic components.
    glif_psc : Single alpha-function variant.
    """
    NUMERICAL_STABILITY_FACTOR = 1e-7
    inv_tau_syn = 1.0 / tau_syn
    inv_tau_m = 1.0 / tau_m
    inv_c_m = 1.0 / c_m

    diff = tau_m - tau_syn

    # Handle exact singularity (tau_m == tau_syn) to avoid ZeroDivisionError.
    if diff == 0.0:
        exp_h_tau = math.exp(-h * inv_tau_m)
        P32 = h * inv_c_m * exp_h_tau
        P31 = 0.5 * h * h * inv_c_m * exp_h_tau
        return P31, P32

    beta = (tau_syn * tau_m) / diff
    gamma = beta / c_m
    inv_beta = diff / (tau_syn * tau_m)

    h_min_regular = NUMERICAL_STABILITY_FACTOR * tau_m * tau_m / abs(diff)

    # Compute P32 (same as evaluate_P32_)
    exp_h_tau_syn = math.exp(-h * inv_tau_syn)
    expm1_h_tau = math.expm1(h * inv_beta)

    P32 = gamma * exp_h_tau_syn * expm1_h_tau

    # Check if P32 is in the regular regime
    if P32 > 0 and math.isfinite(P32) and P32 != 0.0:
        exp_h_tau = None
    else:
        # Singular case for P32
        exp_h_tau = math.exp(-h * inv_tau_m)
        P32 = h * inv_c_m * exp_h_tau

    # Compute P31
    if h > h_min_regular:
        # Regular case for P31
        P31 = gamma * exp_h_tau_syn * (beta * expm1_h_tau - h)
    else:
        # Singular case for P31
        if exp_h_tau is None:
            exp_h_tau = math.exp(-h * inv_tau_m)
        P31 = 0.5 * h * h * inv_c_m * exp_h_tau

    return P31, P32


class glif_psc_double_alpha(Neuron):
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

    **Parameter Mapping (NEST to brainpy.state)**

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
        >>> import brainunit as u
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

        self._validate_parameters()

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

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        # Check valid model mechanism combinations
        s, a, v = self.has_theta_spike, self.has_asc, self.has_theta_voltage
        valid_combos = [
            (False, False, False),  # GLIF1
            (True, False, False),   # GLIF2
            (False, True, False),   # GLIF3
            (True, True, False),    # GLIF4
            (True, True, True),     # GLIF5
        ]
        if (s, a, v) not in valid_combos:
            raise ValueError(
                "Incorrect model mechanism combination. "
                "Valid combinations: GLIF1(FFF), GLIF2(TFF), GLIF3(FTF), "
                "GLIF4(TTF), GLIF5(TTT). Got spike_dependent_threshold=%s, "
                "after_spike_currents=%s, adapting_threshold=%s." % (s, a, v)
            )

        # V_reset (relative) < V_th (relative) — both relative to E_L
        E_L_mV = self._to_numpy(self.E_L, u.mV)
        V_reset_rel = self._to_numpy(self.V_reset, u.mV) - E_L_mV
        V_th_rel = self._to_numpy(self.V_th, u.mV) - E_L_mV
        if np.any(V_reset_rel >= V_th_rel):
            raise ValueError("Reset potential must be smaller than threshold.")

        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError("Capacitance must be strictly positive.")
        if np.any(self._to_numpy(self.g_m, u.nS) <= 0.0):
            raise ValueError("Membrane conductance must be strictly positive.")
        if np.any(self._to_numpy(self.t_ref, u.ms) <= 0.0):
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

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Creates and initializes all state variables required for GLIF dynamics,
        including membrane potential, synaptic current states (fast and slow
        components for each receptor port), threshold components, after-spike
        current values, refractory counters, and buffered input current.

        **State Variables Created**

        - ``self.V``: Membrane potential (HiddenState, brainunit.mV)
        - ``self.y1_fast[k]``: Fast alpha-function derivative state for receptor k (HiddenState, brainunit.pA)
        - ``self.y2_fast[k]``: Fast alpha-function current for receptor k (HiddenState, brainunit.pA)
        - ``self.y1_slow[k]``: Slow alpha-function derivative state for receptor k (HiddenState, brainunit.pA)
        - ``self.y2_slow[k]``: Slow alpha-function current for receptor k (HiddenState, brainunit.pA)
        - ``self.last_spike_time``: Last spike time (ShortTermState, brainunit.ms)
        - ``self.refractory_step_count``: Remaining refractory steps (ShortTermState, int32)
        - ``self.I_stim``: Buffered external current (ShortTermState, brainunit.pA)
        - ``self._ASCurrents``: After-spike current values (NumPy array, pA)
        - ``self._ASCurrents_sum``: Sum of after-spike currents (NumPy array, pA)
        - ``self._threshold_spike``: Spike component of threshold (NumPy array, mV relative to E_L)
        - ``self._threshold_voltage``: Voltage component of threshold (NumPy array, mV relative to E_L)
        - ``self._threshold``: Total threshold (NumPy array, mV relative to E_L)

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size for parallel simulations. If ``None`` (default),
            no batch dimension is added. If an integer, state variables are
            initialized with shape ``(batch_size, \*self.varshape)``.
        **kwargs
            Additional keyword arguments (currently unused, reserved for future extensions).

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
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.V = brainstate.HiddenState(V)

        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        # Per-receptor alpha-function current states: fast component (y1_fast, y2_fast)
        self.y1_fast = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
            )
            for _ in range(self._n_receptors)
        ]
        self.y2_fast = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
            )
            for _ in range(self._n_receptors)
        ]

        # Per-receptor alpha-function current states: slow component (y1_slow, y2_slow)
        self.y1_slow = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
            )
            for _ in range(self._n_receptors)
        ]
        self.y2_slow = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
            )
            for _ in range(self._n_receptors)
        ]

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        # GLIF-specific state (stored as plain numpy, matching NEST)
        # ASC values
        n_asc = len(self.asc_decay)
        self._ASCurrents = np.zeros((n_asc, *v_shape), dtype=np.float64)
        for a in range(n_asc):
            self._ASCurrents[a] = self.asc_init[a]
        self._ASCurrents_sum = np.sum(self._ASCurrents, axis=0) if n_asc > 0 else np.zeros(v_shape, dtype=np.float64)

        # Threshold components (relative to E_L)
        E_L_mV = float(self._to_numpy(self.E_L, u.mV))
        th_inf = float(self._to_numpy(self.V_th, u.mV)) - E_L_mV
        self._th_inf = th_inf
        self._threshold_spike = np.zeros(v_shape, dtype=np.float64)
        self._threshold_voltage = np.zeros(v_shape, dtype=np.float64)
        self._threshold = np.full(v_shape, th_inf, dtype=np.float64)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to initial conditions.

        Resets all state variables to their initial values as defined in
        ``init_state()``. This is useful for restarting simulations or running
        multiple trials without re-instantiating the neuron population.

        **State Variables Reset**

        All state variables are reset to the same initial conditions as in
        ``init_state()``:

        - Membrane potential → ``V_initializer`` (default: ``E_L``)
        - Synaptic current states (fast/slow, all receptors) → 0.0 pA
        - After-spike currents → ``asc_init``
        - Threshold components → 0.0 (spike/voltage), ``V_th - E_L`` (baseline)
        - Refractory counter → 0
        - Last spike time → -1e7 ms
        - Buffered external current → 0.0 pA

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If ``None`` (default), uses the current batch
            size (or no batch dimension). If an integer, reshapes all states to
            ``(batch_size, \*self.varshape)``, allowing changing batch size during
            simulation.
        **kwargs
            Additional keyword arguments (currently unused, reserved for future extensions).

        Notes
        -----
        - This method can be called mid-simulation to restart dynamics while
          preserving model parameters.
        - Changing ``batch_size`` allows switching between single-run and batched
          simulations without re-creating the model.

        See Also
        --------
        init_state : Initialize state variables (called during model setup).
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        for i in range(self._n_receptors):
            self.y1_fast[i].value = braintools.init.param(
                braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
            )
            self.y2_fast[i].value = braintools.init.param(
                braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
            )
            self.y1_slow[i].value = braintools.init.param(
                braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
            )
            self.y2_slow[i].value = braintools.init.param(
                braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
            )
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        n_asc = len(self.asc_decay)
        self._ASCurrents = np.zeros((n_asc, *v_shape), dtype=np.float64)
        for a in range(n_asc):
            self._ASCurrents[a] = self.asc_init[a]
        self._ASCurrents_sum = np.sum(self._ASCurrents, axis=0) if n_asc > 0 else np.zeros(v_shape, dtype=np.float64)

        E_L_mV = float(self._to_numpy(self.E_L, u.mV))
        th_inf = float(self._to_numpy(self.V_th, u.mV)) - E_L_mV
        self._th_inf = th_inf
        self._threshold_spike = np.zeros(v_shape, dtype=np.float64)
        self._threshold_voltage = np.zeros(v_shape, dtype=np.float64)
        self._threshold = np.full(v_shape, th_inf, dtype=np.float64)

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
            compatible with ``self.varshape`` (or ``(batch_size, \*self.varshape)``).
            Unit: ``brainunit.mV`` or dimensionless (interpreted as mV).

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

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _collect_receptor_delta_inputs(self):
        r"""Collect delta inputs per receptor port from registered delta_inputs.

        This internal method scans all registered delta inputs (typically added via
        ``add_delta_input()`` by projection objects) and routes them to the
        appropriate receptor ports based on label patterns. Delta inputs represent
        discrete spike events (weighted by synaptic strength) that are applied
        instantaneously to the synaptic current state variables.

        **Routing Logic**

        - If a delta input key contains the substring ``'receptor_<k>'`` where ``k``
          is an integer (0-based), the input is routed to receptor port ``k``.
        - If a delta input key does not match any receptor pattern, it is routed
          to receptor port 0 by default.
        - Callable delta inputs (typically lambdas or functions) are evaluated once
          and then removed from the dictionary to prevent re-evaluation.

        **Unit Conversion**

        All delta inputs are converted to picoamperes (pA) as NumPy float64 arrays,
        then broadcast to match the neuron population shape (including batch dimension
        if present).

        Returns
        -------
        dy : list of ndarray
            List of length ``n_receptors``, where ``dy[k]`` is a float64 NumPy array
            of shape ``self.V.value.shape`` (including batch dimension) containing
            the summed delta current inputs (in pA) for receptor port ``k`` at the
            current time step.

        Notes
        -----
        - This method is called internally by ``update()`` before updating synaptic
          current state variables.
        - Delta inputs are cumulative: if multiple projections target the same
          receptor port, their contributions are summed.
        - After collection, callable inputs are removed from ``self.delta_inputs``
          to avoid stale references.

        See Also
        --------
        update : Main simulation step, which calls this method.
        add_delta_input : Method inherited from ``Dynamics`` for registering inputs.
        """
        v_shape = self.V.value.shape
        dy = [np.zeros(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]

        if self.delta_inputs is None:
            return dy

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            out_pA = self._to_numpy(out, u.pA)
            out_pA = self._broadcast_to_state(out_pA, v_shape)

            port = None
            parts = key.split('_')
            for i, part in enumerate(parts):
                if part == 'receptor' and i + 1 < len(parts):
                    try:
                        port = int(parts[i + 1])
                    except ValueError:
                        pass
                    break

            if port is not None and 0 <= port < self._n_receptors:
                dy[port] = dy[port] + out_pA
            else:
                dy[0] = dy[0] + out_pA

        return dy

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
            Unit: ``brainunit.pA`` or dimensionless (interpreted as pA).

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
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        # Extract state as numpy float64
        E_L_mV = float(self._to_numpy(self.E_L, u.mV))
        V_abs = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape).copy()
        V_rel = V_abs - E_L_mV  # relative to E_L

        # Fast component states
        y1_fast_all = [
            self._broadcast_to_state(self._to_numpy(self.y1_fast[k].value, u.pA), v_shape).copy()
            for k in range(self._n_receptors)
        ]
        y2_fast_all = [
            self._broadcast_to_state(self._to_numpy(self.y2_fast[k].value, u.pA), v_shape).copy()
            for k in range(self._n_receptors)
        ]

        # Slow component states
        y1_slow_all = [
            self._broadcast_to_state(self._to_numpy(self.y1_slow[k].value, u.pA), v_shape).copy()
            for k in range(self._n_receptors)
        ]
        y2_slow_all = [
            self._broadcast_to_state(self._to_numpy(self.y2_slow[k].value, u.pA), v_shape).copy()
            for k in range(self._n_receptors)
        ]

        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        ).copy()
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape).copy()

        # Parameters
        G = float(self._to_numpy(self.g_m, u.nS))
        C_m = float(self._to_numpy(self.C_m, u.pF))
        V_reset_rel = float(self._to_numpy(self.V_reset, u.mV)) - E_L_mV
        t_ref_ms = float(self._to_numpy(self.t_ref, u.ms))
        I_e = float(self._to_numpy(self.I_e, u.pA))

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )

        # Pre-compute propagator matrix elements (matching NEST pre_run_hook)
        Tau = C_m / G  # membrane time constant in ms
        P33 = math.exp(-dt / Tau)
        P30 = (1.0 / C_m) * (1.0 - P33) * Tau

        # Fast component propagators
        P11_fast = [0.0] * self._n_receptors
        P21_fast = [0.0] * self._n_receptors
        P22_fast = [0.0] * self._n_receptors
        P31_fast = [0.0] * self._n_receptors
        P32_fast = [0.0] * self._n_receptors
        PSCInitialValues_fast = [0.0] * self._n_receptors

        # Slow component propagators
        P11_slow = [0.0] * self._n_receptors
        P21_slow = [0.0] * self._n_receptors
        P22_slow = [0.0] * self._n_receptors
        P31_slow = [0.0] * self._n_receptors
        P32_slow = [0.0] * self._n_receptors
        PSCInitialValues_slow = [0.0] * self._n_receptors

        for i in range(self._n_receptors):
            # Fast component
            P11_fast[i] = math.exp(-dt / self.tau_syn_fast[i])
            P22_fast[i] = P11_fast[i]
            P21_fast[i] = dt * P11_fast[i]
            P31_fast[i], P32_fast[i] = _iaf_propagator_alpha(self.tau_syn_fast[i], Tau, C_m, dt)
            PSCInitialValues_fast[i] = math.e / self.tau_syn_fast[i]

            # Slow component
            P11_slow[i] = math.exp(-dt / self.tau_syn_slow[i])
            P22_slow[i] = P11_slow[i]
            P21_slow[i] = dt * P11_slow[i]
            P31_slow[i], P32_slow[i] = _iaf_propagator_alpha(self.tau_syn_slow[i], Tau, C_m, dt)
            PSCInitialValues_slow[i] = math.e / self.tau_syn_slow[i] * self.amp_slow[i]

        # Pre-compute GLIF decay rates
        if self.has_theta_spike:
            theta_spike_decay_rate = math.exp(-self.th_spike_decay * dt)
            theta_spike_refractory_decay_rate = math.exp(-self.th_spike_decay * t_ref_ms)

        if self.has_asc:
            n_asc = len(self.asc_decay)
            asc_decay_rates = [math.exp(-self.asc_decay[a] * dt) for a in range(n_asc)]
            asc_stable_coeff = [
                ((1.0 / self.asc_decay[a]) / dt) * (1.0 - asc_decay_rates[a])
                for a in range(n_asc)
            ]
            asc_refractory_decay_rates = [
                self.asc_r[a] * math.exp(-self.asc_decay[a] * t_ref_ms)
                for a in range(n_asc)
            ]

        if self.has_theta_voltage:
            potential_decay_rate = math.exp(-G * dt / C_m)
            theta_voltage_decay_rate_inverse = 1.0 / math.exp(self.th_voltage_decay * dt)
            phi = self.th_voltage_index / (self.th_voltage_decay - G / C_m)
            abpara_ratio_voltage = self.th_voltage_index / self.th_voltage_decay

        # Get per-receptor synaptic spike inputs
        dy_input = self._collect_receptor_delta_inputs()

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        # Output arrays
        spike_mask = np.zeros(v_shape, dtype=bool)
        V_next = np.empty(v_shape, dtype=np.float64)
        y1_fast_next = [np.empty(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]
        y2_fast_next = [np.empty(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]
        y1_slow_next = [np.empty(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]
        y2_slow_next = [np.empty(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]
        r_next = np.empty(v_shape, dtype=np.int32)

        for idx in np.ndindex(v_shape):
            # ---- Step 1: Record v_old (relative) ----
            v_old = V_rel[idx]

            if r[idx] == 0:
                # neuron not refractory

                # ---- Step 2a: Decay spike threshold component ----
                if self.has_theta_spike:
                    self._threshold_spike[idx] *= theta_spike_decay_rate

                # ---- Step 2b: Calculate ASC (exact mean and decay) ----
                asc_sum = 0.0
                if self.has_asc:
                    for a in range(n_asc):
                        asc_sum += asc_stable_coeff[a] * self._ASCurrents[a][idx]
                        self._ASCurrents[a][idx] *= asc_decay_rates[a]
                self._ASCurrents_sum[idx] = asc_sum

                # ---- Step 2c: Voltage dynamics (exact integration) ----
                v_new = v_old * P33 + (I_e + i_stim[idx] + asc_sum) * P30

                # Add fast synapse component
                I_syn_fast = 0.0
                for i in range(self._n_receptors):
                    v_new += P31_fast[i] * y1_fast_all[i][idx] + P32_fast[i] * y2_fast_all[i][idx]
                    I_syn_fast += y2_fast_all[i][idx]

                # Add slow synapse component
                I_syn_slow = 0.0
                for i in range(self._n_receptors):
                    v_new += P31_slow[i] * y1_slow_all[i][idx] + P32_slow[i] * y2_slow_all[i][idx]
                    I_syn_slow += y2_slow_all[i][idx]

                # ---- Step 2d: Voltage-dependent threshold ----
                if self.has_theta_voltage:
                    beta = (I_e + i_stim[idx] + asc_sum) / G
                    self._threshold_voltage[idx] = (
                        phi * (v_old - beta) * potential_decay_rate
                        + theta_voltage_decay_rate_inverse * (
                            self._threshold_voltage[idx]
                            - phi * (v_old - beta)
                            - abpara_ratio_voltage * beta
                        )
                        + abpara_ratio_voltage * beta
                    )

                # ---- Step 2e: Update total threshold ----
                self._threshold[idx] = (
                    self._threshold_spike[idx]
                    + self._threshold_voltage[idx]
                    + self._th_inf
                )

                # ---- Step 2f: Check for spike ----
                if v_new > self._threshold[idx]:
                    spike_mask[idx] = True

                    # Set refractory
                    r_next[idx] = refr_counts[idx]

                    # Reset ASC values
                    if self.has_asc:
                        for a in range(n_asc):
                            self._ASCurrents[a][idx] = (
                                self.asc_amps[a]
                                + self._ASCurrents[a][idx] * asc_refractory_decay_rates[a]
                            )

                    # Reset voltage
                    if not self.has_theta_spike:
                        # GLIF1/3: simple reset
                        v_new = V_reset_rel
                    else:
                        # GLIF2/4/5: biologically defined reset
                        v_new = self.voltage_reset_fraction * v_old + self.voltage_reset_add

                        # Reset spike threshold component
                        self._threshold_spike[idx] = (
                            self._threshold_spike[idx] * theta_spike_refractory_decay_rate
                            + self.th_spike_add
                        )

                        # Update global threshold
                        self._threshold[idx] = (
                            self._threshold_spike[idx]
                            + self._threshold_voltage[idx]
                            + self._th_inf
                        )

                    V_next[idx] = v_new
                else:
                    r_next[idx] = 0
                    V_next[idx] = v_new
            else:
                # ---- Refractory: decrement, hold voltage ----
                r_next[idx] = r[idx] - 1
                V_next[idx] = v_old
                self._threshold[idx] = (
                    self._threshold_spike[idx]
                    + self._threshold_voltage[idx]
                    + self._th_inf
                )

        # ---- Step 4: Update synaptic current state variables (alpha shape PSCs) ----
        for i in range(self._n_receptors):
            for idx in np.ndindex(v_shape):
                # Fast component
                y2_fast_next[i][idx] = P21_fast[i] * y1_fast_all[i][idx] + P22_fast[i] * y2_fast_all[i][idx]
                y1_fast_next[i][idx] = P11_fast[i] * y1_fast_all[i][idx]

                # Slow component
                y2_slow_next[i][idx] = P21_slow[i] * y1_slow_all[i][idx] + P22_slow[i] * y2_slow_all[i][idx]
                y1_slow_next[i][idx] = P11_slow[i] * y1_slow_all[i][idx]

        # ---- Step 5: Add incoming spike current jumps ----
        for i in range(self._n_receptors):
            y1_fast_next[i] = y1_fast_next[i] + dy_input[i] * PSCInitialValues_fast[i]
            y1_slow_next[i] = y1_slow_next[i] + dy_input[i] * PSCInitialValues_slow[i]

        # ---- Step 6: Update external current (buffered for next step) ----
        # ---- Step 7: Write back state ----
        self.V.value = (V_next + E_L_mV) * u.mV  # convert back to absolute
        for k in range(self._n_receptors):
            self.y1_fast[k].value = y1_fast_next[k] * u.pA
            self.y2_fast[k].value = y2_fast_next[k] * u.pA
            self.y1_slow[k].value = y1_slow_next[k] * u.pA
            self.y2_slow[k].value = y2_slow_next[k] * u.pA
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)

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
            Unit: ``brainunit.pA`` (picoamperes).

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
            Unit: ``brainunit.pA`` (picoamperes).

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
            Unit: ``brainunit.pA`` (picoamperes).

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
