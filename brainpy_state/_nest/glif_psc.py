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
    'glif_psc',
]


def _iaf_propagator_alpha(tau_syn, tau_m, c_m, h):
    r"""Compute exact integration propagator elements P31, P32 for alpha-PSC.

    This mirrors NEST's ``IAFPropagatorAlpha::evaluate()`` with singularity
    handling for the case :math:`\tau_m \approx \tau_{syn}`.

    The propagator maps the synaptic current state variables
    :math:`(y_1, y_2)` to the membrane voltage update.

    **Regular case** (when :math:`\tau_m` and :math:`\tau_{syn}` are
    sufficiently different):

    .. math::

        \beta = \frac{\tau_{syn} \cdot \tau_m}{\tau_m - \tau_{syn}}

    .. math::

        \gamma = \frac{\beta}{C_m}

    .. math::

        P_{32} = \gamma \cdot e^{-h/\tau_{syn}} \cdot
                 \left( e^{h / \beta} - 1 \right)

    .. math::

        P_{31} = \gamma \cdot e^{-h/\tau_{syn}} \cdot
                 \left( \beta \cdot \left( e^{h / \beta} - 1 \right) - h \right)

    **Singular case** (when :math:`\tau_m \approx \tau_{syn}`):

    .. math::

        P_{32,\text{singular}} = \frac{h}{C_m} \cdot e^{-h/\tau_m}

    .. math::

        P_{31,\text{singular}} = \frac{h^2}{2 \, C_m} \cdot e^{-h/\tau_m}

    Parameters
    ----------
    tau_syn : float
        Synaptic time constant in ms.
    tau_m : float
        Membrane time constant in ms.
    c_m : float
        Membrane capacitance in pF.
    h : float
        Time step in ms.

    Returns
    -------
    P31 : float
        Propagator element mapping :math:`y_1` (derivative of synaptic
        current) to membrane voltage.
    P32 : float
        Propagator element mapping :math:`y_2` (synaptic current) to
        membrane voltage.
    """
    NUMERICAL_STABILITY_FACTOR = 1e-7
    inv_tau_syn = 1.0 / tau_syn
    inv_tau_m = 1.0 / tau_m
    inv_c_m = 1.0 / c_m

    diff = tau_m - tau_syn

    # Handle exact singularity (tau_m == tau_syn) to avoid ZeroDivisionError.
    # In NEST C++, float division by zero yields inf/nan which is caught later;
    # in Python we must handle it explicitly.
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
        # Regular case
        exp_h_tau = None  # not needed
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


class glif_psc(Neuron):
    r"""Current-based generalized leaky integrate-and-fire (GLIF) neuron model.

    Description
    -----------

    ``glif_psc`` provides five generalized leaky integrate-and-fire (GLIF)
    models [1]_ with alpha-function shaped synaptic currents. Incoming spike
    events induce a postsynaptic change of current modeled by an alpha
    function [2]_. The alpha function is normalized such that an event of
    weight 1.0 results in a peak current of 1 pA at
    :math:`t = \tau_\mathrm{syn}`. By default, ``glif_psc`` has a single
    synapse that is accessible through ``receptor_port`` 1. An arbitrary
    number of synapses with different time constants can be configured by
    setting the desired time constants as ``tau_syn`` array. The resulting
    synapses are addressed through ``receptor_port`` 1, 2, 3, ....

    The five GLIF models are:

    * **GLIF Model 1** (LIF) — Traditional leaky integrate-and-fire
    * **GLIF Model 2** (LIF_R) — LIF with biologically defined reset rules
    * **GLIF Model 3** (LIF_ASC) — LIF with after-spike currents
    * **GLIF Model 4** (LIF_R_ASC) — LIF with reset rules and after-spike
      currents
    * **GLIF Model 5** (LIF_R_ASC_A) — LIF with reset rules, after-spike
      currents, and a voltage-dependent threshold

    Model mechanism selection is based on three boolean parameters:

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

    Membrane dynamics
    .................

    The membrane potential :math:`U` (stored relative to :math:`E_L`) evolves
    according to exact integration (linear dynamics):

    .. math::

       U(t+dt) = U(t) \cdot P_{33} + (I_e + I_\mathrm{ASC,sum}) \cdot P_{30}
                 + \sum_k \left( P_{31,k} \cdot y_{1,k} + P_{32,k} \cdot y_{2,k} \right)

    where the propagator matrix elements are:

    .. math::

       P_{33} = \exp\left(-\frac{dt}{\tau_m}\right), \quad
       P_{30} = \frac{\tau_m}{C_m} \left(1 - P_{33}\right), \quad
       \tau_m = \frac{C_m}{g}

    and :math:`P_{31,k}`, :math:`P_{32,k}` are computed via the
    ``IAFPropagatorAlpha`` algorithm that handles the singularity when
    :math:`\tau_m \approx \tau_{\mathrm{syn},k}`.

    Synaptic currents (alpha function)
    ...................................

    Each receptor port has a current modeled by an alpha function with two
    state variables :math:`y_{1,k}` and :math:`y_{2,k}`:

    .. math::

       y_{2,k}(t+dt) = P_{21,k} \cdot y_{1,k}(t) + P_{22,k} \cdot y_{2,k}(t)

    .. math::

       y_{1,k}(t+dt) = P_{11,k} \cdot y_{1,k}(t)

    where:

    .. math::

       P_{11,k} = P_{22,k} = \exp(-dt / \tau_{\mathrm{syn},k}), \quad
       P_{21,k} = dt \cdot P_{11,k}

    On a presynaptic spike of weight :math:`w`:

    .. math::

       y_{1,k} \leftarrow y_{1,k} + w \cdot \frac{e}{\tau_{\mathrm{syn},k}}

    After-spike currents (GLIF3/4/5)
    .................................

    After-spike currents (ASC) are modeled as exponentially decaying currents
    with exact integration. Each ASC component :math:`I_j` decays with rate
    :math:`k_j`:

    .. math::

       I_j(t+dt) = I_j(t) \cdot \exp(-k_j \cdot dt)

    The time-averaged ASC over a step uses the stable coefficient:

    .. math::

       \bar{I}_j = \frac{1 - \exp(-k_j \cdot dt)}{k_j \cdot dt} \cdot I_j(t)

    On spike, ASC values are reset:

    .. math::

       I_j \leftarrow \Delta I_j + I_j \cdot r_j \cdot \exp(-k_j \cdot t_\mathrm{ref})

    Spike-dependent threshold (GLIF2/4/5)
    ......................................

    The spike component of the threshold decays exponentially:

    .. math::

       \theta_s(t+dt) = \theta_s(t) \cdot \exp(-b_s \cdot dt)

    On spike, after refractory decay:

    .. math::

       \theta_s \leftarrow \theta_s \cdot \exp(-b_s \cdot t_\mathrm{ref})
           + \Delta\theta_s

    Voltage reset (with spike-dependent threshold):

    .. math::

       U \leftarrow f_v \cdot U_\mathrm{old} + V_\mathrm{add}

    Voltage-dependent threshold (GLIF5)
    ....................................

    The voltage component of the threshold evolves according to:

    .. math::

       \theta_v(t+dt) = \phi \cdot (U_\mathrm{old} - \beta) \cdot P_\mathrm{decay}
           + \frac{1}{P_{\theta,v}} \cdot \left(\theta_v(t)
               - \phi \cdot (U_\mathrm{old} - \beta)
               - \frac{a_v}{b_v} \cdot \beta \right)
           + \frac{a_v}{b_v} \cdot \beta

    where :math:`\phi = a_v / (b_v - g/C_m)`,
    :math:`P_\mathrm{decay} = \exp(-g \cdot dt / C_m)`,
    :math:`P_{\theta,v} = \exp(b_v \cdot dt)`,
    and :math:`\beta = (I_e + I_\mathrm{ASC,sum}) / g`.

    Overall threshold:

    .. math::

       \theta = \theta_\infty + \theta_s + \theta_v

    Spike condition (checked after voltage update):

    .. math::

       U > \theta

    Numerical integration and update order
    .......................................

    NEST uses exact integration for the linear subthreshold dynamics (via
    propagator matrices). The discrete-time update order per simulation step
    is:

    1. Record :math:`U_\mathrm{old}` (relative to :math:`E_L`).
    2. If not refractory:

       a. Decay spike threshold component.
       b. Compute time-averaged ASC and decay ASC values.
       c. Update membrane potential:
          :math:`U = U_\mathrm{old} \cdot P_{33} + (I + ASC_\mathrm{sum}) \cdot P_{30} + \sum P_{31} y_1 + P_{32} y_2`.
       d. Compute voltage-dependent threshold component (using :math:`U_\mathrm{old}`).
       e. Update total threshold.
       f. If :math:`U > \theta`: emit spike, apply reset rules.

    3. If refractory: decrement counter, hold U at :math:`U_\mathrm{old}`.
    4. Update synaptic current state variables:
       :math:`y_2 = P_{21} y_1 + P_{22} y_2`, then :math:`y_1 = P_{11} y_1`.
    5. Add incoming spike current jumps (scaled by :math:`e / \tau_\mathrm{syn}`).
    6. Update external current input :math:`I`.
    7. Record and save :math:`U_\mathrm{old}` for next step.

    Parameters
    ----------

    =============================== =================== ========================================== =====================================================
    **Parameter**                   **Default**         **Math equivalent**                        **Description**
    =============================== =================== ========================================== =====================================================
    ``in_size``                     (required)                                                     Population shape
    ``g``                           9.43 nS             :math:`g`                                  Membrane (leak) conductance
    ``E_L``                         -78.85 mV           :math:`E_L`                                Resting membrane potential
    ``V_th``                        -51.68 mV           :math:`V_\mathrm{th}`                      Instantaneous threshold (absolute)
    ``C_m``                         58.72 pF            :math:`C_\mathrm{m}`                       Membrane capacitance
    ``t_ref``                       3.75 ms             :math:`t_\mathrm{ref}`                     Absolute refractory period
    ``V_reset``                     -78.85 mV           :math:`V_\mathrm{reset}`                   Reset potential (absolute; GLIF1/3)
    ``th_spike_add``                0.37 mV             :math:`\Delta\theta_s`                     Threshold additive constant after spike
    ``th_spike_decay``              0.009 /ms           :math:`b_s`                                Spike threshold decay rate
    ``voltage_reset_fraction``      0.20                :math:`f_v`                                Voltage fraction after spike
    ``voltage_reset_add``           18.51 mV            :math:`V_\mathrm{add}`                     Voltage additive after spike
    ``th_voltage_index``            0.005 /ms           :math:`a_v`                                Voltage-dependent threshold leak
    ``th_voltage_decay``            0.09 /ms            :math:`b_v`                                Voltage-dependent threshold decay rate
    ``asc_init``                    (0.0, 0.0) pA                                                  Initial values of ASC
    ``asc_decay``                   (0.003, 0.1) /ms    :math:`k_j`                                ASC time constants (decay rates)
    ``asc_amps``                    (-9.18, -198.94) pA :math:`\Delta I_j`                         ASC amplitudes on spike
    ``asc_r``                       (1.0, 1.0)          :math:`r_j`                                ASC fraction coefficient
    ``tau_syn``                     (2.0,) ms           :math:`\tau_{\mathrm{syn},k}`              Synaptic alpha-function time constants
    ``spike_dependent_threshold``   False                                                          Enable biologically defined reset (GLIF2/4/5)
    ``after_spike_currents``        False                                                          Enable after-spike currents (GLIF3/4/5)
    ``adapting_threshold``          False                                                          Enable voltage-dependent threshold (GLIF5)
    ``I_e``                         0.0 pA              :math:`I_e`                                Constant external current
    ``V_initializer``               Constant(E_L)                                                  Membrane potential initializer
    ``spk_fun``                     ReluGrad()                                                     Surrogate spike function
    ``spk_reset``                   ``'hard'``                                                     Reset mode
    =============================== =================== ========================================== =====================================================

    State Variables
    ---------------

    ============================ ===========================================
    **State variable**           **Description**
    ============================ ===========================================
    ``V``                        Membrane potential :math:`V_\mathrm{m}`
    ``y1``                       Synaptic current derivative state (list per receptor)
    ``y2``                       Synaptic current state (list per receptor)
    ``threshold``                Total threshold (relative to :math:`E_L`)
    ``threshold_spike``          Spike component of threshold
    ``threshold_voltage``        Voltage component of threshold
    ``ASCurrents``               After-spike current values (numpy array)
    ``ASCurrents_sum``           Sum of after-spike currents
    ``refractory_step_count``    Remaining refractory grid steps
    ``I_stim``                   Buffered external current
    ``last_spike_time``          Last spike time
    ============================ ===========================================

    Notes
    -----

    - Default parameter values are from GLIF Model 5 of Cell 490626718 from the
      `Allen Cell Type Database <https://celltypes.brain-map.org>`_.
    - Parameters ``V_th`` and ``V_reset`` are specified in absolute mV. Internally,
      membrane potential is tracked relative to ``E_L``, matching NEST's convention.
    - For models with spike-dependent threshold (GLIF2/4/5), the reset condition
      should satisfy:

      .. math::

          E_L + f_v \cdot (V_{th} - E_L) + V_{add} < V_{th} + \Delta\theta_s

      Otherwise the neuron may spike continuously.
    - Unlike ``glif_cond`` which uses an RKF45 ODE integrator, ``glif_psc`` uses
      exact integration via propagator matrices for the linear subthreshold
      dynamics, matching NEST's implementation.
    - If ``tau_m`` is very close to ``tau_syn``, the model numerically behaves as
      if they are equal, to avoid numerical instabilities (see NEST
      IAF_Integration_Singularity notebook).

    References
    ----------
    .. [1] Teeter C, Iyer R, Menon V, Gouwens N, Feng D, Berg J, Szafer A,
           Cain N, Zeng H, Hawrylycz M, Koch C, & Mihalas S (2018).
           Generalized leaky integrate-and-fire models classify multiple neuron
           types. Nature Communications 9:709.
    .. [2] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. J. Comput. Neurosci. 16:159-175.
    .. [3] NEST Simulator ``glif_psc`` model documentation and C++ source:
           ``models/glif_psc.h`` and ``models/glif_psc.cpp``.

    See Also
    --------
    glif_cond
    gif_psc_exp_multisynapse
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
        tau_syn: Sequence[float] = (2.0,),  # ms
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

        # Synaptic parameters (lists)
        self.tau_syn = tuple(float(x) for x in tau_syn)

        # Model mechanism flags
        self.has_theta_spike = bool(spike_dependent_threshold)
        self.has_asc = bool(after_spike_currents)
        self.has_theta_voltage = bool(adapting_threshold)

        # Default V_initializer to E_L
        if V_initializer is None:
            V_initializer = braintools.init.Constant(E_L)
        self.V_initializer = V_initializer

        self._n_receptors = len(self.tau_syn)

        self._validate_parameters()

    @property
    def n_receptors(self):
        """Number of synaptic receptor ports."""
        return self._n_receptors

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

        for tau in self.tau_syn:
            if tau <= 0.0:
                raise ValueError("All synaptic time constants must be strictly positive.")

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.V = brainstate.HiddenState(V)

        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        # Per-receptor alpha-function current states: y1, y2
        self.y1 = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
            )
            for _ in range(self._n_receptors)
        ]
        self.y2 = [
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
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        for i in range(self._n_receptors):
            self.y1[i].value = braintools.init.param(
                braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
            )
            self.y2[i].value = braintools.init.param(
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
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _collect_receptor_delta_inputs(self):
        """Collect delta inputs per receptor port.

        Delta inputs for receptor port k should be registered with key
        containing 'receptor_<k>' (0-based). Any input not matching a specific
        receptor pattern is added to receptor 0 as default.

        Returns a list of arrays, one per receptor, in pA (numpy float64).
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
        """Perform a single simulation step.

        Parameters
        ----------
        x : ArrayLike
            External current input (pA). Applied with one-step delay (buffered).

        Returns
        -------
        spike : array
            Spike output via surrogate gradient function.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        # Extract state as numpy float64
        E_L_mV = float(self._to_numpy(self.E_L, u.mV))
        V_abs = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape).copy()
        V_rel = V_abs - E_L_mV  # relative to E_L

        y1_all = [
            self._broadcast_to_state(self._to_numpy(self.y1[k].value, u.pA), v_shape).copy()
            for k in range(self._n_receptors)
        ]
        y2_all = [
            self._broadcast_to_state(self._to_numpy(self.y2[k].value, u.pA), v_shape).copy()
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

        P11 = [0.0] * self._n_receptors
        P21 = [0.0] * self._n_receptors
        P22 = [0.0] * self._n_receptors
        P31 = [0.0] * self._n_receptors
        P32 = [0.0] * self._n_receptors
        PSCInitialValues = [0.0] * self._n_receptors

        for i in range(self._n_receptors):
            P11[i] = math.exp(-dt / self.tau_syn[i])
            P22[i] = P11[i]
            P21[i] = dt * P11[i]
            P31[i], P32[i] = _iaf_propagator_alpha(self.tau_syn[i], Tau, C_m, dt)
            PSCInitialValues[i] = math.e / self.tau_syn[i]

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
        y1_next = [np.empty(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]
        y2_next = [np.empty(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]
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

                # Add synapse component
                I_syn = 0.0
                for i in range(self._n_receptors):
                    v_new += P31[i] * y1_all[i][idx] + P32[i] * y2_all[i][idx]
                    I_syn += y2_all[i][idx]

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

        # ---- Step 4: Update synaptic current state variables ----
        for i in range(self._n_receptors):
            for idx in np.ndindex(v_shape):
                y2_next[i][idx] = P21[i] * y1_all[i][idx] + P22[i] * y2_all[i][idx]
                y1_next[i][idx] = P11[i] * y1_all[i][idx]

        # ---- Step 5: Add incoming spike current jumps ----
        for i in range(self._n_receptors):
            y1_next[i] = y1_next[i] + dy_input[i] * PSCInitialValues[i]

        # ---- Step 6: Update external current (buffered for next step) ----
        # ---- Step 7: Write back state ----
        self.V.value = (V_next + E_L_mV) * u.mV  # convert back to absolute
        for k in range(self._n_receptors):
            self.y1[k].value = y1_next[k] * u.pA
            self.y2[k].value = y2_next[k] * u.pA
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)
