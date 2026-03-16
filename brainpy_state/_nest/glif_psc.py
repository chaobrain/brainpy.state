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
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, AdaptiveRungeKuttaStep

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

        >>> import brainpy.state as bp
        >>> import saiunit as u
        >>> with u.context(dt=0.1 * u.ms):
        ...     model = bp.glif_psc(100, spike_dependent_threshold=False,
        ...                         after_spike_currents=False, adapting_threshold=False)
        ...     model.init_all_states()
        ...     output = model(350 * u.pA)

    **GLIF Model 5 (Full Model with Adaptation)**:

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import saiunit as u
        >>> with u.context(dt=0.1 * u.ms):
        ...     model = bp.glif_psc(100, spike_dependent_threshold=True,
        ...                         after_spike_currents=True, adapting_threshold=True)
        ...     model.init_all_states()
        ...     output = model(200 * u.pA)

    **Multi-Receptor Configuration**:

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import saiunit as u
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

        Returns
        -------
        int
            Number of independent receptor ports, determined by the length
            of the ``tau_syn`` parameter. Each receptor port has its own
            synaptic time constant and independent alpha-function dynamics.
        """
        return self._n_receptors

    @staticmethod
    def _to_numpy(x, unit):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

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

        for tau in self.tau_syn:
            if tau <= 0.0:
                raise ValueError("All synaptic time constants must be strictly positive.")

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
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        # GLIF-specific state (stored as plain numpy, matching NEST)
        # ASC values
        n_asc = len(self.asc_decay)
        self._ASCurrents = np.zeros((n_asc, *self.varshape), dtype=dftype)
        for a in range(n_asc):
            self._ASCurrents[a] = self.asc_init[a]
        self._ASCurrents_sum = (
            np.sum(self._ASCurrents, axis=0) if n_asc > 0
            else np.zeros(self.varshape, dtype=dftype)
        )

        # Threshold components (relative to E_L)
        E_L_mV = float(self._to_numpy(self.E_L, u.mV))
        th_inf = float(self._to_numpy(self.V_th, u.mV)) - E_L_mV
        self._th_inf = th_inf
        self._threshold_spike = np.zeros(self.varshape, dtype=dftype)
        self._threshold_voltage = np.zeros(self.varshape, dtype=dftype)
        self._threshold = np.full(self.varshape, th_inf, dtype=dftype)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

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

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Computes the continuous-time derivatives for the membrane voltage
        and alpha-function synaptic current states. The membrane dynamics
        follow the linear LIF equation (no exponential spike initiation).

        Parameters
        ----------
        state : DotDict
            Keys: V, plus y1_k and y2_k for each receptor port k --
            ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_peak_detect,
            asc_sum -- mutable auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, state.V)

        # Total synaptic current from all receptor ports
        I_syn = u.math.zeros_like(state.V) * u.pA / u.mV  # start as 0 pA
        I_syn = jnp.zeros_like(u.get_mantissa(state.V)) * u.pA
        for k in range(self._n_receptors):
            I_syn = I_syn + state['y2_%d' % k]

        # LIF membrane dynamics: C_m * dV/dt = -g_m * (V - E_L) + I_syn + I_asc + I_e + I_stim
        dV_raw = (
            -self.g_m * (v_eff - self.E_L) + I_syn + extra.asc_sum + self.I_e + extra.i_stim
        ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        derivs = DotDict(V=dV)

        # Alpha-function synaptic current derivatives per receptor port
        for k in range(self._n_receptors):
            tau_k = self.tau_syn[k] * u.ms
            y1_k = state['y1_%d' % k]
            y2_k = state['y2_%d' % k]
            # dy1/dt = -y1 / tau_syn
            derivs['y1_%d' % k] = -y1_k / tau_k
            # dy2/dt = y1 - y2 / tau_syn
            derivs['y2_%d' % k] = y1_k - y2_k / tau_k

        return derivs

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Checks for threshold crossing after each accepted RKF45 substep.
        When a spike is detected, applies voltage reset, ASC reset, and
        threshold adaptation according to the configured GLIF model level.

        Parameters
        ----------
        state : DotDict
            Keys: V, plus y1_k and y2_k for each receptor port k --
            ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_peak_detect,
            asc_sum.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & (state.V < -1e3 * u.mV)
        )

        refr_accept = accept & (extra.r > 0)

        spike_now = accept & (extra.r <= 0) & (state.V >= extra.v_peak_detect)
        spike_mask = extra.spike_mask | spike_now

        # Voltage reset depends on model level
        if not self.has_theta_spike:
            # GLIF1/3: simple reset to V_reset during refractory and on spike
            new_V = u.math.where(refr_accept, self.V_reset, state.V)
            new_V = u.math.where(spike_now, self.V_reset, new_V)
        else:
            # GLIF2/4/5: biologically defined reset on spike;
            # during refractory, hold V at its current value (already bio-reset)
            new_V = state.V
            v_reset_bio = (
                self.voltage_reset_fraction * (new_V - self.E_L)
                + self.voltage_reset_add * u.mV
                + self.E_L
            )
            new_V = u.math.where(spike_now, v_reset_bio, new_V)

        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict({**state, 'V': new_V})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

    def _collect_receptor_delta_inputs(self):
        r"""Collect delta inputs per receptor port.

        Scans the ``delta_inputs`` dictionary for input currents registered
        with receptor-specific keys. Input keys containing 'receptor_<k>'
        (0-based indexing) are directed to the corresponding receptor port.
        Any input not matching a specific receptor pattern is added to
        receptor 0 as default.

        Returns
        -------
        list of numpy.ndarray
            List of length ``n_receptors``, where each element is a float64
            array of shape matching the neuron population. Each array contains
            the total delta input current (pA) for that receptor port during
            this time step.

        Notes
        -----
        - Callable inputs are evaluated once and then removed from the
          dictionary.
        - Input values are converted to pA and broadcast to the neuron
          population shape.
        - This method is called during the ``update`` step before synaptic
          state propagation.
        """
        v_shape = self.V.value.shape
        dftype = brainstate.environ.dftype()
        dy = [np.zeros(v_shape, dtype=dftype) for _ in range(self._n_receptors)]

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
        r"""Advance the neuron by one simulation step.

        Performs adaptive RKF45 integration of membrane and synaptic dynamics
        over the interval :math:`(t, t+dt]`, with in-loop spike detection,
        reset, and refractory handling. GLIF-specific discrete-step updates
        (ASC decay, threshold adaptation) are applied before and after
        integration.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input (pA). Applied with one-step delay
            (buffered). Shape must broadcast to the neuron population shape.
            Default: 0.0 pA.

        Returns
        -------
        jax.Array
            Binary spike tensor with dtype ``float64`` and shape
            ``self.V.value.shape``. A value of ``1.0`` indicates at least one
            internal spike event occurred during the integrated interval
            :math:`(t, t+dt]`.

        Notes
        -----
        - The update follows NEST's discrete-time integration order:
          (1) record old voltage, (2) update threshold and ASC, (3) integrate
          voltage, (4) check spike and reset, (5) propagate synaptic states,
          (6) add spike inputs, (7) buffer external current.
        - Propagator matrix elements are pre-computed once per step for all
          neurons, ensuring computational efficiency.
        - Refractory neurons have their voltage clamped to the previous value
          and do not update threshold or ASC.
        - ASC state is stored as NumPy arrays for efficiency and mutated
          in-place during the update.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        v_shape = self.V.value.shape

        # Extract state
        E_L_mV = float(self._to_numpy(self.E_L, u.mV))
        V_abs = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape).copy()
        V_rel = V_abs - E_L_mV  # relative to E_L

        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=ditype), v_shape
        ).copy()
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape).copy()

        # Parameters
        G = float(self._to_numpy(self.g_m, u.nS))
        C_m_val = float(self._to_numpy(self.C_m, u.pF))
        V_reset_rel = float(self._to_numpy(self.V_reset, u.mV)) - E_L_mV
        t_ref_ms = float(self._to_numpy(self.t_ref, u.ms))
        I_e_val = float(self._to_numpy(self.I_e, u.pA))

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.ref_count), dtype=ditype), v_shape
        )

        # Pre-compute GLIF decay rates
        if self.has_theta_spike:
            theta_spike_decay_rate = np.exp(-self.th_spike_decay * dt)
            theta_spike_refractory_decay_rate = np.exp(-self.th_spike_decay * t_ref_ms)

        if self.has_asc:
            n_asc = len(self.asc_decay)
            asc_decay_rates = [np.exp(-self.asc_decay[a] * dt) for a in range(n_asc)]
            asc_stable_coeff = [
                ((1.0 / self.asc_decay[a]) / dt) * (1.0 - asc_decay_rates[a])
                for a in range(n_asc)
            ]
            asc_refractory_decay_rates = [
                self.asc_r[a] * np.exp(-self.asc_decay[a] * t_ref_ms)
                for a in range(n_asc)
            ]

        if self.has_theta_voltage:
            potential_decay_rate = np.exp(-G * dt / C_m_val)
            theta_voltage_decay_rate_inverse = 1.0 / np.exp(self.th_voltage_decay * dt)
            phi = self.th_voltage_index / (self.th_voltage_decay - G / C_m_val)
            abpara_ratio_voltage = self.th_voltage_index / self.th_voltage_decay

        # Get per-receptor synaptic spike inputs
        dy_input = self._collect_receptor_delta_inputs()

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)

        # Pre-integration GLIF updates (per-neuron, numpy)
        spike_mask = np.zeros(v_shape, dtype=bool)
        V_next = np.empty(v_shape, dtype=dftype)
        r_next = np.empty(v_shape, dtype=ditype)

        for idx in np.ndindex(v_shape):
            v_old = V_rel[idx]

            if r[idx] == 0:
                # neuron not refractory

                # Decay spike threshold component
                if self.has_theta_spike:
                    self._threshold_spike[idx] *= theta_spike_decay_rate

                # Calculate ASC (exact mean and decay)
                asc_sum = 0.0
                if self.has_asc:
                    for a in range(n_asc):
                        asc_sum += asc_stable_coeff[a] * self._ASCurrents[a][idx]
                        self._ASCurrents[a][idx] *= asc_decay_rates[a]
                self._ASCurrents_sum[idx] = asc_sum

                # Voltage-dependent threshold (using v_old, before voltage update)
                if self.has_theta_voltage:
                    beta = (I_e_val + i_stim[idx] + asc_sum) / G
                    self._threshold_voltage[idx] = (
                        phi * (v_old - beta) * potential_decay_rate
                        + theta_voltage_decay_rate_inverse * (
                            self._threshold_voltage[idx]
                            - phi * (v_old - beta)
                            - abpara_ratio_voltage * beta
                        )
                        + abpara_ratio_voltage * beta
                    )

                # Update total threshold
                self._threshold[idx] = (
                    self._threshold_spike[idx]
                    + self._threshold_voltage[idx]
                    + self._th_inf
                )
            else:
                # Refractory: just update threshold display
                self._threshold[idx] = (
                    self._threshold_spike[idx]
                    + self._threshold_voltage[idx]
                    + self._th_inf
                )

        # Build ODE state for adaptive RKF45 integration
        V_state = self.V.value  # mV (absolute, with units)
        h = self.integration_step.value  # ms
        r_jax = self.refractory_step_count.value  # int

        # Spike detection threshold (absolute voltage, with units)
        # For GLIF the threshold is: E_L + threshold (where threshold is relative)
        v_peak_detect = self._threshold * u.mV + self.E_L

        # ASC sum with units for the vector field
        asc_sum_pA = self._ASCurrents_sum * u.pA

        ode_state = DotDict(V=V_state)
        for k in range(self._n_receptors):
            ode_state['y1_%d' % k] = self.y1[k].value
            ode_state['y2_%d' % k] = self.y2[k].value

        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r_jax,
            unstable=jnp.array(False),
            i_stim=self.I_stim.value,
            v_peak_detect=v_peak_detect,
            asc_sum=asc_sum_pA,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V_out = ode_state.V
        spike_mask_jax = extra.spike_mask
        r_out = extra.r
        unstable = extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in glif_psc dynamics.'
        )

        # Post-integration spike handling: ASC reset and threshold adaptation
        # These are discrete-step updates that happen when a spike is detected
        spike_np = np.asarray(spike_mask_jax)
        V_out_mV = self._broadcast_to_state(self._to_numpy(V_out, u.mV), v_shape).copy()
        V_out_rel = V_out_mV - E_L_mV

        for idx in np.ndindex(v_shape):
            if spike_np[idx]:
                # Reset ASC values on spike
                if self.has_asc:
                    for a in range(n_asc):
                        self._ASCurrents[a][idx] = (
                            self.asc_amps[a]
                            + self._ASCurrents[a][idx] * asc_refractory_decay_rates[a]
                        )

                # Reset spike threshold component on spike
                if self.has_theta_spike:
                    self._threshold_spike[idx] = (
                        self._threshold_spike[idx] * theta_spike_refractory_decay_rate
                        + self.th_spike_add
                    )

                # Update global threshold after spike
                self._threshold[idx] = (
                    self._threshold_spike[idx]
                    + self._threshold_voltage[idx]
                    + self._th_inf
                )

        # Decrement refractory counter.
        r_out = u.math.where(r_out > 0, r_out - 1, r_out)

        # Synaptic spike inputs (applied after integration).
        PSCInitialValues = [np.e / self.tau_syn[k] for k in range(self._n_receptors)]

        # Write back state.
        self.V.value = V_out
        for k in range(self._n_receptors):
            y1_val = ode_state['y1_%d' % k]
            y2_val = ode_state['y2_%d' % k]
            # Add incoming spike current jumps (y1 has units pA/ms)
            y1_val = y1_val + dy_input[k] * PSCInitialValues[k] * u.pA / u.ms
            self.y1[k].value = y1_val
            self.y2[k].value = y2_val

        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r_out), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim_q + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask_jax, t + dt_q, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask_jax, dtype=dftype)
