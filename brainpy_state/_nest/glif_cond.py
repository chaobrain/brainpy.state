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
    'glif_cond',
]


class glif_cond(Neuron):
    r"""Conductance-based generalized leaky integrate-and-fire (GLIF) neuron model.

    Description
    -----------

    ``glif_cond`` provides five generalized leaky integrate-and-fire (GLIF)
    models [1]_ with conductance-based synapses. Incoming spike events induce a
    postsynaptic change of conductance modeled by an alpha function [2]_. The
    alpha function is normalized such that an event of weight 1.0 results in a
    peak conductance change of 1 nS at :math:`t = \tau_\mathrm{syn}`. On the
    postsynaptic side, there can be arbitrarily many synaptic time constants
    (multiple receptor ports).

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

    The membrane potential :math:`V` (stored relative to :math:`E_L`) evolves
    according to:

    .. math::

       C_\mathrm{m} \frac{dV}{dt} = -g \cdot V
           - \sum_k g_k(t) \left( V + E_L - E_{\mathrm{rev},k} \right)
           + I_\mathrm{e} + I_\mathrm{ASC,sum}

    where :math:`g` is the membrane (leak) conductance, :math:`g_k` is the
    synaptic conductance for receptor port :math:`k`, :math:`E_{\mathrm{rev},k}`
    is the reversal potential for port :math:`k`, :math:`I_\mathrm{e}` is the
    external current input, and :math:`I_\mathrm{ASC,sum}` is the sum of
    after-spike currents.

    Synaptic conductances (alpha function)
    ......................................

    Each receptor port has a conductance modeled by an alpha function with two
    state variables :math:`dg_k` and :math:`g_k`:

    .. math::

       \frac{d(dg_k)}{dt} = -\frac{dg_k}{\tau_{\mathrm{syn},k}}

    .. math::

       \frac{dg_k}{dt} = dg_k - \frac{g_k}{\tau_{\mathrm{syn},k}}

    On a presynaptic spike of weight :math:`w`:

    .. math::

       dg_k \leftarrow dg_k + w \cdot \frac{e}{\tau_{\mathrm{syn},k}}

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

       V \leftarrow f_v \cdot V_\mathrm{old} + V_\mathrm{add}

    Voltage-dependent threshold (GLIF5)
    ....................................

    The voltage component of the threshold evolves according to:

    .. math::

       \theta_v(t+dt) = \phi \cdot (V_\mathrm{old} - \beta) \cdot P_\mathrm{decay}
           + \frac{1}{P_{\theta,v}} \cdot \left(\theta_v(t)
               - \phi \cdot (V_\mathrm{old} - \beta)
               - \frac{a_v}{b_v} \cdot \beta \right)
           + \frac{a_v}{b_v} \cdot \beta

    where :math:`\phi = a_v / (b_v - g/C_m)`,
    :math:`P_\mathrm{decay} = \exp(-g \cdot dt / C_m)`,
    :math:`P_{\theta,v} = \exp(b_v \cdot dt)`,
    and :math:`\beta = (I_e + I_\mathrm{ASC,sum}) / g`.

    Overall threshold:

    .. math::

       \theta = \theta_\infty + \theta_s + \theta_v

    Spike condition (checked after ODE integration):

    .. math::

       V > \theta

    Numerical integration and update order
    ......................................

    NEST integrates the ODE system [V, dg_0, g_0, dg_1, g_1, ...] with
    adaptive RKF45 (GSL). This implementation mirrors that behavior with an
    RKF45(4,5) integrator.

    The discrete-time update order per simulation step is:

    1. Record :math:`V_\mathrm{old}` (relative to :math:`E_L`).
    2. Integrate ODE system over :math:`(t, t+dt]` using RKF45.
    3. If not refractory:

       a. Decay spike threshold component.
       b. Compute time-averaged ASC and decay ASC values.
       c. Compute voltage-dependent threshold component (using :math:`V_\mathrm{old}`).
       d. Update total threshold.
       e. If :math:`V > \theta`: emit spike, apply reset rules.

    4. If refractory: decrement counter, hold V at :math:`V_\mathrm{old}`.
    5. Add incoming spike conductance jumps (scaled by :math:`e/\tau_\mathrm{syn}`).
    6. Update external current input :math:`I_e`.
    7. Record and save :math:`V_\mathrm{old}` for next step.

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
    ``tau_syn``                     (0.2, 2.0) ms       :math:`\tau_{\mathrm{syn},k}`              Synaptic alpha-function time constants
    ``E_rev``                       (0.0, -85.0) mV     :math:`E_{\mathrm{rev},k}`                 Synaptic reversal potentials
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
    ``g_syn``                    Synaptic conductances :math:`g_k` (list per receptor)
    ``dg_syn``                   Synaptic conductance derivatives (list per receptor)
    ``threshold``                Total threshold
    ``threshold_spike``          Spike component of threshold
    ``threshold_voltage``        Voltage component of threshold
    ``ASCurrents``               After-spike current values (numpy array)
    ``ASCurrents_sum``           Sum of after-spike currents
    ``refractory_step_count``    Remaining refractory grid steps
    ``integration_step``         Internal RKF45 step-size state
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

    References
    ----------
    .. [1] Teeter C, Iyer R, Menon V, Gouwens N, Feng D, Berg J, Szafer A,
           Cain N, Zeng H, Hawrylycz M, Koch C, & Mihalas S (2018).
           Generalized leaky integrate-and-fire models classify multiple neuron
           types. Nature Communications 9:709.
    .. [2] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. J. Comput. Neurosci. 16:159-175.
    .. [3] NEST Simulator ``glif_cond`` model documentation and C++ source:
           ``models/glif_cond.h`` and ``models/glif_cond.cpp``.

    See Also
    --------
    iaf_cond_exp
    gif_cond_exp_multisynapse
    """
    __module__ = 'brainpy.state'

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 10000

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
        tau_syn: Sequence[float] = (0.2, 2.0),  # ms
        E_rev: Sequence[float] = (0.0, -85.0),  # mV
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
        self.E_rev = tuple(float(x) for x in E_rev)

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

        if len(self.tau_syn) != len(self.E_rev):
            raise ValueError(
                "tau_syn and E_rev must have the same size. "
                "Got %d and %d." % (len(self.tau_syn), len(self.E_rev))
            )

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

        # Per-receptor alpha-function conductance states: dg and g
        self.g_syn = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.nS), self.varshape, batch_size)
            )
            for _ in range(self._n_receptors)
        ]
        self.dg_syn = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.nS), self.varshape, batch_size)
            )
            for _ in range(self._n_receptors)
        ]

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = self._safe_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
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
            self.g_syn[i].value = braintools.init.param(
                braintools.init.Constant(0.0 * u.nS), self.varshape, batch_size
            )
            self.dg_syn[i].value = braintools.init.param(
                braintools.init.Constant(0.0 * u.nS), self.varshape, batch_size
            )
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        dt = self._safe_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
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

        Returns a list of arrays, one per receptor, in nS (numpy float64).
        """
        v_shape = self.V.value.shape
        dg = [np.zeros(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]

        if self.delta_inputs is None:
            return dg

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            out_nS = self._to_numpy(out, u.nS)
            out_nS = self._broadcast_to_state(out_nS, v_shape)

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
                dg[port] = dg[port] + out_nS
            else:
                dg[0] = dg[0] + out_nS

        return dg

    def _dynamics_scalar(self, v_rel, dg_vals, g_vals, is_refractory, i_ext, asc_sum, p):
        """Compute derivatives for ODE system [V_rel, dg_0, g_0, dg_1, g_1, ...].

        Matches NEST's glif_cond_dynamics() function exactly.
        V is relative to E_L. During refractory period, V is clamped to V_reset_rel.

        Parameters
        ----------
        i_ext : float
            Total external current (I_e + I_stim) in pA, matching NEST's B_.I_.
        """
        V = p['V_reset_rel'] if is_refractory else v_rel

        # Synaptic current: I_syn = sum_k g_k * (V + E_L - E_rev_k)
        # In NEST: V is relative, so (V + E_L) is absolute V_m
        I_syn = 0.0
        for k in range(self._n_receptors):
            I_syn += g_vals[k] * (V + p['E_L'] - self.E_rev[k])

        # Leak current: I_leak = G * V (V is relative to E_L)
        I_leak = p['G'] * V

        # dV/dt: i_ext = I_e + I_stim (matches NEST B_.I_)
        dv = 0.0 if is_refractory else (-I_leak - I_syn + i_ext + asc_sum) / p['C_m']

        # Alpha function dynamics for each receptor
        ddg = []
        dg_out = []
        for k in range(self._n_receptors):
            ddg.append(-dg_vals[k] / self.tau_syn[k])
            dg_out.append(dg_vals[k] - g_vals[k] / self.tau_syn[k])

        return (dv,) + tuple(ddg) + tuple(dg_out)

    def _rkf45_integrate_scalar(self, v0, dg0_vals, g0_vals, is_refractory, i_stim, asc_sum, h0, dt, p):
        """Adaptive RKF45 integration.

        State vector: [V_rel, dg_0, g_0, dg_1, g_1, ...]
        """
        n_rec = self._n_receptors
        n = 1 + 2 * n_rec  # V + (dg + g) per receptor
        t = 0.0
        h = max(h0, self._MIN_H)

        # Pack state: [V, dg_0, dg_1, ..., g_0, g_1, ...]
        y = [v0] + list(dg0_vals) + list(g0_vals)
        iters = 0

        def f(state):
            return self._dynamics_scalar(
                state[0],
                state[1:1+n_rec],
                state[1+n_rec:],
                is_refractory, i_stim, asc_sum, p
            )

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = min(h, dt - t)
            h = max(h, self._MIN_H)

            k1 = f(y)
            y2 = [y[i] + h * k1[i] / 4.0 for i in range(n)]
            k2 = f(y2)
            y3 = [y[i] + h * (3.0 * k1[i] / 32.0 + 9.0 * k2[i] / 32.0) for i in range(n)]
            k3 = f(y3)
            y4 = [y[i] + h * (1932.0 * k1[i] / 2197.0 - 7200.0 * k2[i] / 2197.0 + 7296.0 * k3[i] / 2197.0)
                  for i in range(n)]
            k4 = f(y4)
            y5 = [y[i] + h * (439.0 * k1[i] / 216.0 - 8.0 * k2[i] + 3680.0 * k3[i] / 513.0
                               - 845.0 * k4[i] / 4104.0) for i in range(n)]
            k5 = f(y5)
            y6 = [y[i] + h * (-8.0 * k1[i] / 27.0 + 2.0 * k2[i] - 3544.0 * k3[i] / 2565.0
                               + 1859.0 * k4[i] / 4104.0 - 11.0 * k5[i] / 40.0) for i in range(n)]
            k6 = f(y6)

            y4_sol = [y[i] + h * (25.0 * k1[i] / 216.0 + 1408.0 * k3[i] / 2565.0
                                   + 2197.0 * k4[i] / 4104.0 - k5[i] / 5.0) for i in range(n)]
            y5_sol = [y[i] + h * (16.0 * k1[i] / 135.0 + 6656.0 * k3[i] / 12825.0
                                   + 28561.0 * k4[i] / 56430.0 - 9.0 * k5[i] / 50.0
                                   + 2.0 * k6[i] / 55.0) for i in range(n)]

            err = max(abs(y5_sol[i] - y4_sol[i]) for i in range(n))

            if err <= self._ATOL or h <= self._MIN_H:
                y = y5_sol
                t += h
                if err == 0.0:
                    fac = 5.0
                else:
                    fac = 0.9 * (self._ATOL / err) ** 0.2
                    fac = min(5.0, max(0.2, fac))
                h = max(self._MIN_H, h * fac)
            else:
                fac = 0.9 * (self._ATOL / err) ** 0.25
                fac = min(1.0, max(0.2, fac))
                h = max(self._MIN_H, h * fac)

        v_out = y[0]
        dg_out = y[1:1+n_rec]
        g_out = y[1+n_rec:]
        return v_out, dg_out, g_out, h

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

        dg_all = [
            self._broadcast_to_state(self._to_numpy(self.dg_syn[k].value, u.nS), v_shape).copy()
            for k in range(self._n_receptors)
        ]
        g_all = [
            self._broadcast_to_state(self._to_numpy(self.g_syn[k].value, u.nS), v_shape).copy()
            for k in range(self._n_receptors)
        ]
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        ).copy()
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape).copy()
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape).copy()

        # Parameters dict
        G = float(self._to_numpy(self.g_m, u.nS))
        C_m = float(self._to_numpy(self.C_m, u.pF))
        V_reset_rel = float(self._to_numpy(self.V_reset, u.mV)) - E_L_mV
        t_ref_ms = float(self._to_numpy(self.t_ref, u.ms))
        I_e = float(self._to_numpy(self.I_e, u.pA))

        p = {
            'G': G,
            'E_L': E_L_mV,
            'C_m': C_m,
            'V_reset_rel': V_reset_rel,
            'I_e': I_e,
        }

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )

        # Pre-compute decay rates (matching NEST pre_run_hook)
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

        # CondInitialValues: e / tau_syn (matching NEST)
        cond_init_vals = [math.e / self.tau_syn[k] for k in range(self._n_receptors)]

        # Get per-receptor synaptic inputs
        dg_input = self._collect_receptor_delta_inputs()

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        # Output arrays
        spike_mask = np.zeros(v_shape, dtype=bool)
        V_next = np.empty(v_shape, dtype=np.float64)
        dg_next = [np.empty(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]
        g_next = [np.empty(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]
        r_next = np.empty(v_shape, dtype=np.int32)
        h_next = np.empty(v_shape, dtype=np.float64)

        for idx in np.ndindex(v_shape):
            # ---- Step 1: Record v_old (relative) ----
            v_old = V_rel[idx]

            # ---- Step 2: Integrate ODE ----
            is_refractory = r[idx] > 0
            dg_vals = [dg_all[k][idx] for k in range(self._n_receptors)]
            g_vals = [g_all[k][idx] for k in range(self._n_receptors)]

            # Total external current: I_e + I_stim (matches NEST B_.I_ usage)
            i_ext = I_e + i_stim[idx]

            v_i, dg_i, g_i, h_i = self._rkf45_integrate_scalar(
                V_rel[idx], dg_vals, g_vals,
                is_refractory, i_ext, self._ASCurrents_sum[idx],
                h_int[idx], dt, p
            )

            if not is_refractory:
                # ---- Step 3a: Update threshold spike component ----
                if self.has_theta_spike:
                    self._threshold_spike[idx] *= theta_spike_decay_rate

                # ---- Step 3b: Calculate ASC (exact mean and decay) ----
                asc_sum_new = 0.0
                if self.has_asc:
                    for a in range(n_asc):
                        asc_sum_new += asc_stable_coeff[a] * self._ASCurrents[a][idx]
                        self._ASCurrents[a][idx] *= asc_decay_rates[a]
                self._ASCurrents_sum[idx] = asc_sum_new

                # ---- Step 3c: Voltage-dependent threshold ----
                if self.has_theta_voltage:
                    beta = (i_ext + asc_sum_new) / G
                    self._threshold_voltage[idx] = (
                        phi * (v_old - beta) * potential_decay_rate
                        + theta_voltage_decay_rate_inverse * (
                            self._threshold_voltage[idx]
                            - phi * (v_old - beta)
                            - abpara_ratio_voltage * beta
                        )
                        + abpara_ratio_voltage * beta
                    )

                # ---- Step 3d: Update total threshold ----
                self._threshold[idx] = (
                    self._threshold_spike[idx]
                    + self._threshold_voltage[idx]
                    + self._th_inf
                )

                # ---- Step 3e: Check for spike ----
                if v_i > self._threshold[idx]:
                    spike_mask[idx] = True

                    # Set refractory
                    r_i = refr_counts[idx]

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
                        v_i = V_reset_rel
                    else:
                        # GLIF2/4/5: biologically defined reset
                        v_i = self.voltage_reset_fraction * v_old + self.voltage_reset_add

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

                    r_next[idx] = r_i
                else:
                    r_next[idx] = 0
            else:
                # ---- Refractory: decrement, hold voltage ----
                r_next[idx] = r[idx] - 1
                v_i = v_old
                self._threshold[idx] = (
                    self._threshold_spike[idx]
                    + self._threshold_voltage[idx]
                    + self._th_inf
                )

            V_next[idx] = v_i
            for k in range(self._n_receptors):
                dg_next[k][idx] = dg_i[k]
                g_next[k][idx] = g_i[k]
            h_next[idx] = h_i

        # ---- Step 5: Add incoming spike conductance jumps (after everything else) ----
        for k in range(self._n_receptors):
            dg_next[k] = dg_next[k] + dg_input[k] * cond_init_vals[k]

        # ---- Step 6: Update external current ----
        # ---- Step 7: Write back state ----
        self.V.value = (V_next + E_L_mV) * u.mV  # convert back to absolute
        for k in range(self._n_receptors):
            self.dg_syn[k].value = dg_next[k] * u.nS
            self.g_syn[k].value = g_next[k] * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        # Record v_old for next step (needed for voltage-dependent threshold)
        # NEST stores v_old at end of update loop as: v_old = S_.y_[V_M]
        # This is handled by reading V_rel at beginning of next update call.

        return jnp.asarray(spike_mask, dtype=jnp.float32)
