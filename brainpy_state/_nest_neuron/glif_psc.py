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

from typing import Callable, Sequence

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base._base import NESTNeuron
from brainpy_state._nest_base._utils import is_tracer, alpha_propagator_p31_p32, cond_any

__all__ = [
    'glif_psc',
]


class glif_psc(NESTNeuron):
    r"""Current-based generalized leaky integrate-and-fire (GLIF) neuron model.

    The ``glif_psc`` model implements the five-level GLIF model hierarchy
    from the Allen Institute [1]_, featuring alpha-function shaped synaptic
    currents, after-spike currents (ASC), spike-dependent threshold adaptation,
    and voltage-dependent threshold modulation. Exact integration via
    propagator matrices ensures numerical stability and matches NEST's
    implementation.

    **Model Hierarchy**

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

    Mathematical Formulation
    ------------------------

    **1. Membrane Dynamics**

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

    **2. Synaptic Currents (Alpha Function)**

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

    The alpha function is normalized such that an event of weight 1.0 results
    in a peak current of 1 pA at :math:`t = \tau_\mathrm{syn}`.

    **3. After-Spike Currents (GLIF3/4/5)**

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

    **4. Spike-Dependent Threshold (GLIF2/4/5)**

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

    **5. Voltage-Dependent Threshold (GLIF5)**

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

    **6. Numerical Integration and Update Order**

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
    in_size : Size
        Shape of the neuron population. Can be tuple of ints or single int.
    g : ArrayLike, optional
        Membrane (leak) conductance. Default: 9.43 nS.
    E_L : ArrayLike, optional
        Resting membrane potential. Default: -78.85 mV.
    V_th : ArrayLike, optional
        Instantaneous threshold voltage (absolute). Default: -51.68 mV.
    C_m : ArrayLike, optional
        Membrane capacitance. Default: 58.72 pF.
    t_ref : ArrayLike, optional
        Absolute refractory period. Default: 3.75 ms.
    V_reset : ArrayLike, optional
        Reset potential (absolute; used for GLIF1/3). Default: -78.85 mV.
    th_spike_add : float, optional
        Threshold additive constant after spike (mV). Default: 0.37.
    th_spike_decay : float, optional
        Spike threshold decay rate (/ms). Default: 0.009.
    voltage_reset_fraction : float, optional
        Voltage fraction coefficient after spike. Default: 0.20.
    voltage_reset_add : float, optional
        Voltage additive constant after spike (mV). Default: 18.51.
    th_voltage_index : float, optional
        Voltage-dependent threshold leak rate (/ms). Default: 0.005.
    th_voltage_decay : float, optional
        Voltage-dependent threshold decay rate (/ms). Default: 0.09.
    asc_init : Sequence[float], optional
        Initial values of after-spike currents (pA). Default: (0.0, 0.0).
    asc_decay : Sequence[float], optional
        ASC decay rates (/ms). Default: (0.003, 0.1).
    asc_amps : Sequence[float], optional
        ASC amplitudes added on spike (pA). Default: (-9.18, -198.94).
    asc_r : Sequence[float], optional
        ASC fraction coefficients (dimensionless). Default: (1.0, 1.0).
    tau_syn : Sequence[float], optional
        Synaptic alpha-function time constants (ms), one per receptor port.
        Default: (2.0,).
    spike_dependent_threshold : bool, optional
        Enable biologically defined reset rules (GLIF2/4/5). Default: False.
    after_spike_currents : bool, optional
        Enable after-spike currents (GLIF3/4/5). Default: False.
    adapting_threshold : bool, optional
        Enable voltage-dependent threshold (GLIF5). Default: False.
    I_e : ArrayLike, optional
        Constant external current. Default: 0.0 pA.
    gsl_error_tol : ArrayLike, optional
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
        Default: 1e-6.
    V_initializer : Callable, optional
        Membrane potential initializer. Default: Constant(E_L).
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation. Default: ReluGrad().
    spk_reset : str, optional
        Spike reset mode: 'hard' or 'soft'. Default: 'hard'.
    ref_var : bool, optional
        If ``True``, allocate and expose ``self.refractory`` state.
    name : str, optional
        Name of the neuron group.


    Parameter Mapping
    -----------------

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
    ``gsl_error_tol``               1e-6                --                                         Local absolute tolerance for RKF45 error estimate
    ``V_initializer``               Constant(E_L)                                                  Membrane potential initializer
    ``spk_fun``                     ReluGrad()                                                     Surrogate spike function
    ``spk_reset``                   ``'hard'``                                                     Reset mode
    ``ref_var``                     False                                                          If True, expose boolean refractory state
    =============================== =================== ========================================== =====================================================

    Attributes
    ----------
    V : HiddenState
        Membrane potential :math:`V_\mathrm{m}` (absolute, mV).
    y1 : list of HiddenState
        Synaptic current derivative states (pA), one per receptor port.
    y2 : list of HiddenState
        Synaptic current states (pA), one per receptor port.
    last_spike_time : ShortTermState
        Last spike time for each neuron (ms).
    refractory_step_count : ShortTermState
        Remaining refractory grid steps (int32).
    integration_step : ShortTermState
        Persistent RKF45 substep size estimate (ms).
    I_stim : ShortTermState
        Buffered external current for next step (pA).
    _ASCurrents : numpy.ndarray
        After-spike current values (pA). Shape: (n_asc, \*varshape).
    _ASCurrents_sum : numpy.ndarray
        Sum of after-spike currents (pA). Shape: (\*varshape).
    _threshold : numpy.ndarray
        Total threshold (relative to E_L, in mV). Shape: (\*varshape).
    _threshold_spike : numpy.ndarray
        Spike component of threshold (mV). Shape: (\*varshape).
    _threshold_voltage : numpy.ndarray
        Voltage component of threshold (mV). Shape: (\*varshape).
    refractory : ShortTermState
        Optional boolean refractory indicator, available only when
        ``ref_var=True``.

    Raises
    ------
    ValueError
        If invalid model mechanism combination is specified.
    ValueError
        If V_reset >= V_th (reset must be below threshold).
    ValueError
        If capacitance, conductance, or time constants are not positive.
    ValueError
        If voltage_reset_fraction not in [0, 1].
    ValueError
        If asc_r values not in [0, 1].
    ValueError
        If ASC parameter arrays have mismatched lengths.

    Notes
    -----
    - Default parameter values are from GLIF Model 5 of Cell 490626718 from the
      `Allen Cell Type Database <https://celltypes.brain-map.org>`_.
    - Parameters ``V_th`` and ``V_reset`` are specified in absolute mV.
      Internally, membrane potential is tracked relative to ``E_L``, matching
      NEST's convention.
    - For models with spike-dependent threshold (GLIF2/4/5), the reset
      condition should satisfy:

      .. math::

          E_L + f_v \cdot (V_{th} - E_L) + V_{add} < V_{th} + \Delta\theta_s

      Otherwise the neuron may spike continuously.
    - Unlike ``glif_cond`` which uses an RKF45 ODE integrator, ``glif_psc``
      uses exact integration via propagator matrices for the linear
      subthreshold dynamics, matching NEST's implementation.
    - If ``tau_m`` is very close to ``tau_syn``, the model numerically behaves
      as if they are equal, to avoid numerical instabilities (see NEST
      IAF_Integration_Singularity notebook).
    - Synaptic inputs are delivered to receptor ports starting from port 0.
      Register inputs with keys like 'receptor_0', 'receptor_1', etc., via
      the ``add_delta_input`` method. Inputs without a receptor label default
      to receptor port 0.

    Examples
    --------
    **GLIF Model 1 (Basic LIF)**:

    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> with u.context(dt=0.1 * u.ms):
        ...     model = bp.glif_psc(100, spike_dependent_threshold=False,
        ...                         after_spike_currents=False, adapting_threshold=False)
        ...     model.init_all_states()
        ...     output = model(350 * u.pA)

    **GLIF Model 5 (Full Model with Adaptation)**:

    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> with u.context(dt=0.1 * u.ms):
        ...     model = bp.glif_psc(100, spike_dependent_threshold=True,
        ...                         after_spike_currents=True, adapting_threshold=True)
        ...     model.init_all_states()
        ...     output = model(200 * u.pA)

    **Multi-Receptor Configuration**:

    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> with u.context(dt=0.1 * u.ms):
        ...     model = bp.glif_psc(100, tau_syn=(2.0, 5.0, 10.0))
        ...     model.init_all_states()
        ...     # Register inputs to different receptor ports
        ...     model.add_delta_input('exc_receptor_0', lambda: 10 * u.pA)
        ...     model.add_delta_input('inh_receptor_1', lambda: -5 * u.pA)

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
    glif_cond : Conductance-based GLIF model with RKF45 integration.
    gif_psc_exp_multisynapse : Generalized IF with exponential synapses.
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
        gsl_error_tol: ArrayLike = 1e-6,
        V_initializer: Callable = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
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
        self.gsl_error_tol = gsl_error_tol
        self.ref_var = ref_var

        self._validate_parameters()

        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    @property
    def n_receptors(self):
        r"""Number of synaptic receptor ports.

        Returns
        -------
        int
            Number of independent receptor ports, determined by the length
            of the ``tau_syn`` parameter. Each receptor port has its own
            synaptic time constant and independent alpha-function dynamics.
        """
        return self._n_receptors

    def _validate_parameters(self):
        r"""Validate model parameters against NEST constraints.

        Raises
        ------
        ValueError
            If parameter inequalities or positivity constraints are violated.
        """
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
        if cond_any(V_reset_rel >= V_th_rel):
            raise ValueError("Reset potential must be smaller than threshold.")

        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError("Capacitance must be strictly positive.")
        if cond_any(self.g_m <= 0.0 * u.nS):
            raise ValueError("Membrane conductance must be strictly positive.")
        if cond_any(self.t_ref <= 0.0 * u.ms):
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

        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize persistent and short-term state variables.

        All GLIF-specific state variables are stored as JAX ``HiddenState`` arrays,
        and pre-computed decay constants are stored as Python floats.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()
        dt_ms = float(np.asarray(u.get_mantissa(dt / u.ms)))

        V = braintools.init.param(self.V_initializer, self.varshape)
        self.V = brainstate.HiddenState(V)

        # Per-receptor alpha-function current states: y1 (rate, pA/ms), y2 (current, pA)
        self.y1 = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA / u.ms), self.varshape)
            )
            for _ in range(self._n_receptors)
        ]
        self.y2 = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape)
            )
            for _ in range(self._n_receptors)
        ]

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        # GLIF-specific state as HiddenState (JAX-traceable, compatible with for_loop)
        n_asc = len(self.asc_decay)
        self._asc_states = [
            brainstate.HiddenState(jnp.full(self.varshape, self.asc_init[a], dtype=dftype))
            for a in range(n_asc)
        ]

        # Threshold components (relative to E_L) as HiddenState
        E_L_mV = float(np.asarray(u.get_mantissa(self.E_L / u.mV)))
        th_inf = float(np.asarray(u.get_mantissa(self.V_th / u.mV))) - E_L_mV
        self._th_inf = th_inf
        self._threshold_spike_state = brainstate.HiddenState(
            jnp.zeros(self.varshape, dtype=dftype)
        )
        self._threshold_voltage_state = brainstate.HiddenState(
            jnp.zeros(self.varshape, dtype=dftype)
        )
        self._threshold_state = brainstate.HiddenState(
            jnp.full(self.varshape, th_inf, dtype=dftype)
        )

        # Pre-compute decay rates (Python float constants, computed once per init_state call)
        G = float(np.asarray(u.get_mantissa(self.g_m / u.nS)))
        C_m_val = float(np.asarray(u.get_mantissa(self.C_m / u.pF)))
        t_ref_ms = float(np.asarray(u.get_mantissa(self.t_ref / u.ms)))

        if self.has_theta_spike:
            self._decay_spike = np.exp(-self.th_spike_decay * dt_ms)
            self._decay_spike_refr = np.exp(-self.th_spike_decay * t_ref_ms)

        if self.has_asc:
            self._asc_decay_rates = [np.exp(-self.asc_decay[a] * dt_ms) for a in range(n_asc)]
            self._asc_stable_coeff = [
                ((1.0 / self.asc_decay[a]) / dt_ms) * (1.0 - self._asc_decay_rates[a])
                for a in range(n_asc)
            ]
            self._asc_refr_decay_rates = [
                self.asc_r[a] * np.exp(-self.asc_decay[a] * t_ref_ms)
                for a in range(n_asc)
            ]

        if self.has_theta_voltage:
            self._potential_decay_rate = np.exp(-G * dt_ms / C_m_val)
            self._theta_voltage_decay_rate_inv = 1.0 / np.exp(self.th_voltage_decay * dt_ms)
            self._phi = self.th_voltage_index / (self.th_voltage_decay - G / C_m_val)
            self._abpara_ratio = self.th_voltage_index / self.th_voltage_decay

        # Pre-compute exact propagator matrices (NEST IAFPropagatorAlpha scheme)
        tau_m = C_m_val / G  # membrane time constant in ms
        self._P33 = np.exp(-dt_ms / tau_m)
        self._P30 = (1.0 / C_m_val) * (1.0 - self._P33) * tau_m  # mV/pA

        self._P11 = []
        self._P21 = []
        self._P22 = []
        self._P31 = []
        self._P32 = []
        self._PSCInitialValues = []

        for k in range(self._n_receptors):
            p11 = np.exp(-dt_ms / self.tau_syn[k])
            self._P11.append(p11)
            self._P22.append(p11)
            self._P21.append(dt_ms * p11)
            p31, p32 = alpha_propagator_p31_p32(self.tau_syn[k], tau_m, C_m_val, dt_ms)
            self._P31.append(float(p31))
            self._P32.append(float(p32))
            self._PSCInitialValues.append(np.e / self.tau_syn[k])

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    # Backward-compatible properties for threshold components
    @property
    def _threshold(self):
        return self._threshold_state.value

    @property
    def _threshold_spike(self):
        return self._threshold_spike_state.value

    @property
    def _threshold_voltage(self):
        return self._threshold_voltage_state.value

    def get_spike(self, V: ArrayLike = None):
        r"""Generate spike output via surrogate gradient function.

        Applies the surrogate gradient function to a normalized voltage signal.
        The voltage is linearly scaled such that ``V_th`` maps to 1 and
        ``V_reset`` maps to 0, providing a normalized input for the surrogate
        function.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (with units). If None, uses current ``self.V.value``.

        Returns
        -------
        spike : jax.numpy.ndarray
            Spike output (float32). Shape matches the neuron population.
            Forward pass produces values in [0, 1]; backward pass uses the
            surrogate gradient specified by ``spk_fun``.

        Notes
        -----
        - This method is called internally by the base ``Neuron`` class and
          is typically not invoked directly by users.
        - The surrogate function enables gradient-based learning by providing
          a differentiable approximation to the Heaviside step function.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _collect_receptor_delta_inputs(self):
        r"""Collect delta inputs per receptor port using label-based routing.

        Returns a list of current jumps (pA) for each receptor port, JIT-compatible.
        """
        dftype = brainstate.environ.dftype()
        return [
            self.sum_delta_inputs(
                jnp.zeros(self.varshape, dtype=dftype) * u.pA,
                label=f'receptor_{k}',
            )
            for k in range(self._n_receptors)
        ]

    def update(self, x=0.0 * u.pA):
        r"""Perform a single simulation step using exact propagator matrices.

        Implements the NEST ``glif_psc`` update using the exact
        IAFPropagatorAlpha integration scheme. All GLIF-specific discrete
        updates (threshold decay, ASC, voltage-dependent threshold) are
        applied as vectorised JAX operations, making this method compatible
        with ``brainstate.transform.for_loop``.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input (pA), applied with one-step delay. Default: 0.0 pA.

        Returns
        -------
        spike : jax.Array
            Binary spike tensor (float32), shape ``(*varshape)``.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Python-level constants (concrete, not JAX-traced)
        E_L_mV = float(np.asarray(u.get_mantissa(self.E_L / u.mV)))
        I_e_pA = float(np.asarray(u.get_mantissa(self.I_e / u.pA)))
        V_reset_rel = float(np.asarray(u.get_mantissa(self.V_reset / u.mV))) - E_L_mV
        G_nS = float(np.asarray(u.get_mantissa(self.g_m / u.nS)))

        # JAX state (traced under for_loop)
        r = self.refractory_step_count.value          # int array, varshape
        i_stim_pA = u.get_mantissa(self.I_stim.value / u.pA)  # float array, varshape
        # V_rel (old, before this step's update)
        V_rel = jax.lax.stop_gradient(
            u.get_mantissa(self.V.value / u.mV) - E_L_mV
        )  # plain JAX array, mV relative to E_L

        # Buffer new external current (one-step delay)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)

        is_refractory = r > 0
        i_ext = I_e_pA + i_stim_pA  # pA, plain JAX array

        n_asc = len(self.asc_decay)

        # 1. Spike threshold decay (non-refractory only)
        if self.has_theta_spike:
            tspk = self._threshold_spike_state.value
            tspk = jnp.where(is_refractory, tspk, tspk * self._decay_spike)
        else:
            tspk = jnp.zeros(self.varshape, dtype=dftype)

        # 2. ASC stable-coeff sum + decay (non-refractory only)
        if self.has_asc:
            asc_sum_new = jnp.zeros(self.varshape, dtype=dftype)
            asc_decayed = []
            for a in range(n_asc):
                asc_a = self._asc_states[a].value
                asc_sum_new = asc_sum_new + self._asc_stable_coeff[a] * asc_a
                asc_decayed.append(asc_a * self._asc_decay_rates[a])
            asc_sum = jnp.where(is_refractory, jnp.zeros(self.varshape, dtype=dftype), asc_sum_new)
        else:
            asc_sum = jnp.zeros(self.varshape, dtype=dftype)
            asc_decayed = []

        # 3. Voltage-dependent threshold (non-refractory only, using old V_rel)
        if self.has_theta_voltage:
            tvlt = self._threshold_voltage_state.value
            beta = (i_ext + asc_sum) / G_nS  # pA/nS = mV
            tvlt_new = (
                self._phi * (V_rel - beta) * self._potential_decay_rate
                + self._theta_voltage_decay_rate_inv * (
                    tvlt
                    - self._phi * (V_rel - beta)
                    - self._abpara_ratio * beta
                )
                + self._abpara_ratio * beta
            )
            tvlt = jnp.where(is_refractory, tvlt, tvlt_new)
        else:
            tvlt = jnp.zeros(self.varshape, dtype=dftype)

        # 4. Total threshold
        threshold = tspk + tvlt + self._th_inf

        # 5. V update via exact propagator
        y1_old = [u.get_mantissa(self.y1[k].value / (u.pA / u.ms)) for k in range(self._n_receptors)]
        y2_old = [u.get_mantissa(self.y2[k].value / u.pA) for k in range(self._n_receptors)]

        v_new = V_rel * self._P33 + (i_ext + asc_sum) * self._P30
        for k in range(self._n_receptors):
            v_new = v_new + self._P31[k] * y1_old[k] + self._P32[k] * y2_old[k]
        # Clamp refractory neurons to old V_rel
        v_new = jnp.where(is_refractory, V_rel, v_new)

        # 6. Spike check (non-refractory only)
        spiked = (v_new > threshold) & ~is_refractory

        # 7. ASC reset on spike
        if self.has_asc:
            for a in range(n_asc):
                asc_a = self._asc_states[a].value
                asc_reset = self.asc_amps[a] + asc_decayed[a] * self._asc_refr_decay_rates[a]
                self._asc_states[a].value = jnp.where(
                    spiked, asc_reset,
                    jnp.where(is_refractory, asc_a, asc_decayed[a])
                )

        # 8. Voltage reset on spike
        if not self.has_theta_spike:
            # GLIF1/3: simple reset
            V_final_rel = jnp.where(spiked, V_reset_rel, v_new)
        else:
            # GLIF2/4/5: biologically defined reset
            V_reset_bio = self.voltage_reset_fraction * V_rel + self.voltage_reset_add
            V_final_rel = jnp.where(spiked, V_reset_bio, v_new)
            # 9. Theta_spike reset on spike
            tspk_reset = tspk * self._decay_spike_refr + self.th_spike_add
            tspk = jnp.where(spiked, tspk_reset, tspk)
            threshold = jnp.where(spiked, tspk + tvlt + self._th_inf, threshold)

        # 10. Refractory counter
        r_new = jnp.where(
            spiked, self.ref_count,
            jnp.where(is_refractory, r - 1, r)
        )

        # 11. Y1/Y2 propagator update (unconditional — all neurons, including refractory)
        y1_new = [self._P11[k] * y1_old[k] for k in range(self._n_receptors)]
        y2_new = [self._P21[k] * y1_old[k] + self._P22[k] * y2_old[k]
                  for k in range(self._n_receptors)]

        # 12. Collect and apply synaptic delta inputs to y1
        dy_input = self._collect_receptor_delta_inputs()
        for k in range(self._n_receptors):
            w_k = u.get_mantissa(dy_input[k] / u.pA)  # weight in pA
            y1_new[k] = y1_new[k] + self._PSCInitialValues[k] * w_k

        # ---- Write back all state ----
        self.V.value = (V_final_rel + E_L_mV) * u.mV
        for k in range(self._n_receptors):
            self.y1[k].value = y1_new[k] * (u.pA / u.ms)
            self.y2[k].value = y2_new[k] * u.pA

        self._threshold_spike_state.value = tspk
        self._threshold_voltage_state.value = tvlt
        self._threshold_state.value = threshold
        self.refractory_step_count.value = jnp.asarray(r_new, dtype=ditype)
        self.I_stim.value = new_i_stim_q + u.math.zeros(self.varshape) * u.pA

        last_spike_time = u.math.where(spiked, t + dt_q, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return jnp.asarray(spiked, dtype=jnp.float32)
