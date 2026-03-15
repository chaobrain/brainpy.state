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
    'glif_cond',
]


class glif_cond(NESTNeuron):
    r"""Conductance-based generalized leaky integrate-and-fire (GLIF) neuron model.

    Implements the five-level GLIF model hierarchy from Teeter et al. (2018) [1]_,
    with conductance-based alpha-function synapses and adaptive RKF45 integration.
    Designed for fitting to Allen Institute single-neuron electrophysiology data.
    Supports multiple receptor ports with distinct reversal potentials and synaptic
    time constants.

    **Model Selection**

    The five GLIF variants are:

    1. **GLIF1 (LIF)** — Traditional leaky integrate-and-fire
    2. **GLIF2 (LIF_R)** — LIF with biologically defined voltage reset rules
    3. **GLIF3 (LIF_ASC)** — LIF with after-spike currents (adaptation)
    4. **GLIF4 (LIF_R_ASC)** — LIF with reset rules and after-spike currents
    5. **GLIF5 (LIF_R_ASC_A)** — LIF with reset rules, after-spike currents, and
       voltage-dependent threshold

    Model mechanism selection is controlled by three boolean parameters:

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

    The membrane potential :math:`V` (tracked relative to :math:`E_L` internally)
    evolves according to:

    .. math::

       C_\mathrm{m} \frac{dV}{dt} = -g \cdot V
           - \sum_k g_k(t) \left( V + E_L - E_{\mathrm{rev},k} \right)
           + I_\mathrm{e} + I_\mathrm{ASC,sum}

    where:

    * :math:`g` — membrane (leak) conductance
    * :math:`g_k(t)` — synaptic conductance for receptor port :math:`k`
    * :math:`E_{\mathrm{rev},k}` — reversal potential for port :math:`k`
    * :math:`I_\mathrm{e}` — constant external current
    * :math:`I_\mathrm{ASC,sum}` — sum of after-spike currents (GLIF3/4/5 only)

    **2. Synaptic Conductances (Alpha Function)**

    Each receptor port :math:`k` has a conductance modeled by an alpha function
    with two state variables :math:`dg_k` and :math:`g_k`:

    .. math::

       \frac{d(dg_k)}{dt} = -\frac{dg_k}{\tau_{\mathrm{syn},k}}

    .. math::

       \frac{dg_k}{dt} = dg_k - \frac{g_k}{\tau_{\mathrm{syn},k}}

    On a presynaptic spike with weight :math:`w`, the derivative is incremented:

    .. math::

       dg_k \leftarrow dg_k + w \cdot \frac{e}{\tau_{\mathrm{syn},k}}

    This normalization ensures that a spike of weight 1.0 produces a peak conductance
    of 1 nS at time :math:`t = \tau_{\mathrm{syn},k}`.

    **3. After-Spike Currents (GLIF3/4/5)**

    After-spike currents (ASC) model spike-triggered adaptation as exponentially
    decaying currents. Each ASC component :math:`I_j` decays with rate :math:`k_j`:

    .. math::

       I_j(t+dt) = I_j(t) \cdot \exp(-k_j \cdot dt)

    The time-averaged ASC over a simulation step uses the exact integral (stable
    coefficient method):

    .. math::

       \bar{I}_j = \frac{1 - \exp(-k_j \cdot dt)}{k_j \cdot dt} \cdot I_j(t)

    On spike, ASC values are updated with amplitude and refractory decay:

    .. math::

       I_j \leftarrow \Delta I_j + I_j \cdot r_j \cdot \exp(-k_j \cdot t_\mathrm{ref})

    where :math:`\Delta I_j` is the amplitude jump and :math:`r_j \in [0, 1]` is
    the retention fraction.

    **4. Spike-Dependent Threshold (GLIF2/4/5)**

    The spike component of the threshold :math:`\theta_s` decays exponentially:

    .. math::

       \theta_s(t+dt) = \theta_s(t) \cdot \exp(-b_s \cdot dt)

    On spike, after accounting for refractory decay, it is incremented:

    .. math::

       \theta_s \leftarrow \theta_s \cdot \exp(-b_s \cdot t_\mathrm{ref})
           + \Delta\theta_s

    Voltage reset with spike-dependent threshold uses:

    .. math::

       V \leftarrow f_v \cdot V_\mathrm{old} + V_\mathrm{add}

    where :math:`f_v \in [0, 1]` is the fraction coefficient and :math:`V_\mathrm{add}`
    is the additive term (both in mV, dimensionless in NEST convention).

    **5. Voltage-Dependent Threshold (GLIF5)**

    The voltage component :math:`\theta_v` evolves according to:

    .. math::

       \theta_v(t+dt) = \phi \cdot (V_\mathrm{old} - \beta) \cdot P_\mathrm{decay}
           + \frac{1}{P_{\theta,v}} \cdot \left(\theta_v(t)
               - \phi \cdot (V_\mathrm{old} - \beta)
               - \frac{a_v}{b_v} \cdot \beta \right)
           + \frac{a_v}{b_v} \cdot \beta

    where:

    * :math:`\phi = a_v / (b_v - g/C_m)`
    * :math:`P_\mathrm{decay} = \exp(-g \cdot dt / C_m)`
    * :math:`P_{\theta,v} = \exp(b_v \cdot dt)`
    * :math:`\beta = (I_e + I_\mathrm{ASC,sum}) / g`

    The total threshold is the sum of all components:

    .. math::

       \theta = \theta_\infty + \theta_s + \theta_v

    Spike condition (checked after ODE integration):

    .. math::

       V > \theta

    **Numerical Integration**

    The ODE system :math:`[V, dg_0, g_0, dg_1, g_1, \ldots]` is integrated using
    an adaptive RKF45(4,5) Runge-Kutta-Fehlberg method with error tolerance
    ``ATOL = 1e-3`` and minimum step size ``MIN_H = 1e-8`` ms, matching NEST's
    GSL integrator behavior.

    **Update Order (Per Simulation Step)**

    1. Record :math:`V_\mathrm{old}` (relative to :math:`E_L`)
    2. Integrate ODE system over :math:`(t, t+dt]` using RKF45
    3. If not refractory:

       a. Decay spike threshold component :math:`\theta_s`
       b. Compute time-averaged ASC :math:`\bar{I}_\mathrm{ASC,sum}` and decay ASC values
       c. Compute voltage-dependent threshold :math:`\theta_v` (using :math:`V_\mathrm{old}`)
       d. Update total threshold :math:`\theta = \theta_\infty + \theta_s + \theta_v`
       e. If :math:`V > \theta`: emit spike, apply reset rules

    4. If refractory: decrement counter, clamp :math:`V` to :math:`V_\mathrm{old}`
    5. Add incoming spike conductance jumps (scaled by :math:`e/\tau_\mathrm{syn}`)
    6. Update external current buffer :math:`I_\mathrm{stim}`
    7. Save :math:`V_\mathrm{old}` for next step

    Parameters
    ----------
    in_size : Size
        Shape of the neuron population. Can be an int for 1D or tuple for multi-D.
    g : ArrayLike, optional
        Membrane (leak) conductance in nS. Broadcast to population shape.
        Default: 9.43 nS (from Allen Cell 490626718 GLIF5).
    E_L : ArrayLike, optional
        Resting membrane potential (leak reversal) in mV. Default: -78.85 mV.
    V_th : ArrayLike, optional
        Instantaneous spike threshold (absolute) in mV. Default: -51.68 mV.
        Internally, threshold is tracked relative to ``E_L``.
    C_m : ArrayLike, optional
        Membrane capacitance in pF. Must be strictly positive. Default: 58.72 pF.
    t_ref : ArrayLike, optional
        Absolute refractory period in ms. During this period, voltage is clamped
        and spike detection is disabled. Must be > 0. Default: 3.75 ms.
    V_reset : ArrayLike, optional
        Reset potential (absolute) in mV for GLIF1/3 models. Ignored if
        ``spike_dependent_threshold=True``. Default: -78.85 mV (same as ``E_L``).
    th_spike_add : float, optional
        Threshold additive constant :math:`\Delta\theta_s` after spike (mV,
        dimensionless in NEST units). Only used if ``spike_dependent_threshold=True``.
        Default: 0.37 mV.
    th_spike_decay : float, optional
        Spike threshold decay rate :math:`b_s` in 1/ms. Must be > 0 if
        ``spike_dependent_threshold=True``. Default: 0.009 /ms.
    voltage_reset_fraction : float, optional
        Voltage fraction coefficient :math:`f_v \in [0, 1]` after spike.
        Only used if ``spike_dependent_threshold=True``. Default: 0.20.
    voltage_reset_add : float, optional
        Voltage additive term :math:`V_\mathrm{add}` after spike (mV, dimensionless).
        Only used if ``spike_dependent_threshold=True``. Default: 18.51 mV.
    th_voltage_index : float, optional
        Voltage-dependent threshold leak :math:`a_v` in 1/ms. Only used if
        ``adapting_threshold=True``. Default: 0.005 /ms.
    th_voltage_decay : float, optional
        Voltage-dependent threshold decay rate :math:`b_v` in 1/ms. Must be > 0 if
        ``adapting_threshold=True``. Default: 0.09 /ms.
    asc_init : Sequence[float], optional
        Initial values of after-spike currents in pA. Tuple/list of length ``n_asc``.
        Default: (0.0, 0.0) pA.
    asc_decay : Sequence[float], optional
        ASC decay rates :math:`k_j` in 1/ms. All values must be > 0. Length must
        match ``asc_init``. Default: (0.003, 0.1) /ms.
    asc_amps : Sequence[float], optional
        ASC amplitude jumps :math:`\Delta I_j` on spike, in pA. Length must match
        ``asc_init``. Negative values cause hyperpolarizing adaptation. Default:
        (-9.18, -198.94) pA.
    asc_r : Sequence[float], optional
        ASC retention fraction coefficients :math:`r_j \in [0, 1]`. Length must
        match ``asc_init``. Default: (1.0, 1.0).
    tau_syn : Sequence[float], optional
        Synaptic alpha-function time constants :math:`\tau_{\mathrm{syn},k}` in ms,
        one per receptor port. All values must be > 0. Default: (0.2, 2.0) ms
        (fast excitatory, slow inhibitory).
    E_rev : Sequence[float], optional
        Synaptic reversal potentials :math:`E_{\mathrm{rev},k}` in mV, one per
        receptor port. Must have same length as ``tau_syn``. Default: (0.0, -85.0) mV
        (excitatory, inhibitory).
    spike_dependent_threshold : bool, optional
        Enable biologically defined voltage reset rules (GLIF2/4/5). Default: False.
    after_spike_currents : bool, optional
        Enable after-spike currents (adaptation) (GLIF3/4/5). Default: False.
    adapting_threshold : bool, optional
        Enable voltage-dependent threshold component (GLIF5 only). Requires
        ``spike_dependent_threshold=True`` and ``after_spike_currents=True``.
        Default: False.
    I_e : ArrayLike, optional
        Constant external current in pA. Broadcast to population shape. Default: 0.0 pA.
    V_initializer : Callable, optional
        Initializer for membrane potential. If None, defaults to ``Constant(E_L)``.
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation. Default: ``ReluGrad()``.
    spk_reset : str, optional
        Spike reset mode: ``'hard'`` (stop gradient) or ``'soft'`` (subtract threshold).
        Default: ``'hard'``.
    name : str, optional
        Name of the neuron population.


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

    Attributes
    ----------
    V : HiddenState
        Membrane potential in mV (absolute, broadcast to population shape).
    g_syn : list[HiddenState]
        Synaptic conductances :math:`g_k` in nS, one per receptor port.
    dg_syn : list[HiddenState]
        Synaptic conductance derivatives :math:`dg_k` in nS, one per receptor port.
    last_spike_time : ShortTermState
        Time of last spike in ms.
    refractory_step_count : ShortTermState
        Remaining refractory steps (int32), decremented each step.
    integration_step : ShortTermState
        Internal RKF45 adaptive step size in ms (updated per neuron).
    I_stim : ShortTermState
        Buffered external current in pA (applied with one-step delay).

    Notes
    -----
    **Implementation Details**

    * **Internal state convention**: Membrane potential is tracked relative to ``E_L``
      internally (matching NEST), but exposed as absolute values in mV.
    * **Threshold components**: ``_threshold_spike``, ``_threshold_voltage``, and
      ``_th_inf`` are stored as numpy arrays (not JAX) for exact NEST replication.
    * **After-spike currents**: ``_ASCurrents`` is a numpy array of shape
      ``(n_asc, *in_size)``.
    * **Receptor port routing**: Delta inputs (from projections) with keys containing
      ``'receptor_<k>'`` (0-based) are routed to receptor port ``k``. Inputs without
      a receptor tag default to receptor 0.
    * **Stability constraint**: For GLIF2/4/5, the reset condition must satisfy:

      .. math::

          E_L + f_v \cdot (V_\mathrm{th} - E_L) + V_\mathrm{add} < V_\mathrm{th} + \Delta\theta_s

      Otherwise the neuron may spike continuously.

    * **Valid mechanism combinations**: Only the five combinations listed in the
      parameter table are valid. Other combinations will raise ``ValueError``.
    * **Adaptive integration**: RKF45 step size adapts per-neuron and is preserved
      across simulation steps.

    **Failure Modes**


    * Raises ``ValueError`` if parameter validation fails (invalid model combination,
      non-positive capacitance/conductance/time constants, mismatched sequence lengths).
    * Raises ``ValueError`` if ``V_reset >= V_th`` (relative to ``E_L``).
    * Integration may fail to converge if ``dt`` is too large relative to ``tau_syn``
      or if threshold parameters cause continuous spiking.

    **Default Parameters**

    Default parameter values are from GLIF Model 5 of Cell 490626718 from the
    `Allen Cell Type Database <https://celltypes.brain-map.org>`_, fitted to
    mouse visual cortex layer 5 pyramidal neuron electrophysiology.

    References
    ----------
    .. [1] Teeter C, Iyer R, Menon V, Gouwens N, Feng D, Berg J, Szafer A,
           Cain N, Zeng H, Hawrylycz M, Koch C, & Mihalas S (2018).
           Generalized leaky integrate-and-fire models classify multiple neuron
           types. Nature Communications 9:709.
           DOI: `10.1038/s41467-017-02717-4 <https://doi.org/10.1038/s41467-017-02717-4>`_
    .. [2] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. J. Comput. Neurosci. 16:159-175.
           DOI: `10.1023/B:JCNS.0000014108.03012.81 <https://doi.org/10.1023/B:JCNS.0000014108.03012.81>`_
    .. [3] NEST Simulator ``glif_cond`` model documentation and C++ source:
           ``models/glif_cond.h`` and ``models/glif_cond.cpp``.

    Examples
    --------
    **Example 1: GLIF1 (simple LIF) with dual-receptor synapses**

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import saiunit as u
        >>> import brainstate as bts
        >>> # Create GLIF1 neuron (all mechanisms disabled)
        >>> neuron = bst.glif_cond(
        ...     100,
        ...     spike_dependent_threshold=False,
        ...     after_spike_currents=False,
        ...     adapting_threshold=False,
        ...     tau_syn=(0.2, 2.0),  # fast excitatory, slow inhibitory
        ...     E_rev=(0.0, -85.0)   # mV
        ... )
        >>> neuron.init_all_states()
        >>> # Stimulate with constant current
        >>> with bts.environ.context(dt=0.1 * u.ms):
        ...     for _ in range(100):
        ...         spike = neuron(200.0 * u.pA)

    **Example 2: GLIF5 (full model) with custom parameters**

    .. code-block:: python

        >>> # Create GLIF5 with all mechanisms enabled
        >>> neuron = bst.glif_cond(
        ...     (10, 10),  # 10x10 population
        ...     spike_dependent_threshold=True,
        ...     after_spike_currents=True,
        ...     adapting_threshold=True,
        ...     g=10.0 * u.nS,
        ...     C_m=100.0 * u.pF,
        ...     tau_syn=(0.5, 1.5, 5.0),  # three receptor ports
        ...     E_rev=(0.0, 0.0, -80.0)   # two excitatory, one inhibitory
        ... )
        >>> neuron.init_all_states()
        >>> print(neuron.n_receptors)  # 3

    **Example 3: Multi-receptor input routing**

    .. code-block:: python

        >>> from brainevent.nn import FixedProb
        >>> # Create projection targeting receptor 1
        >>> proj = bst.align_post_projection(
        ...     pre=pre_neurons,
        ...     post=glif_neurons,
        ...     comm=FixedProb(0.1, weight=0.5 * u.nS),
        ...     label='receptor_1'  # route to receptor port 1
        ... )

    See Also
    --------
    iaf_cond_exp : Simpler conductance-based LIF with exponential synapses
    gif_cond_exp_multisynapse : Generalized integrate-and-fire with exponential synapses
    glif_psc : Current-based GLIF variant
    """
    __module__ = 'brainpy.state'

    _ATOL = 1e-3
    _MIN_H = 1e-8 * u.ms  # ms
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

        self.integrator = AdaptiveRungeKuttaStep(
            method='RKF45',
            vf=self._vector_field,
            event_fn=self._event_fn,
            min_h=self._MIN_H,
            max_iters=self._MAX_ITERS,
            atol=self._ATOL,
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
            Number of receptor ports, determined by length of ``tau_syn``.
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

        if len(self.tau_syn) != len(self.E_rev):
            raise ValueError(
                "tau_syn and E_rev must have the same size. "
                "Got %d and %d." % (len(self.tau_syn), len(self.E_rev))
            )

        for tau in self.tau_syn:
            if tau <= 0.0:
                raise ValueError("All synaptic time constants must be strictly positive.")

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

        # Per-receptor alpha-function conductance states: dg and g
        self.g_syn = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.nS), self.varshape)
            )
            for _ in range(self._n_receptors)
        ]
        self.dg_syn = [
            brainstate.HiddenState(
                braintools.init.param(braintools.init.Constant(0.0 * u.nS), self.varshape)
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
        E_L_mV = float(np.asarray(u.math.asarray(self.E_L / u.mV), dtype=dftype))
        th_inf = float(np.asarray(u.math.asarray(self.V_th / u.mV), dtype=dftype)) - E_L_mV
        self._th_inf = th_inf
        self._threshold_spike = np.zeros(self.varshape, dtype=dftype)
        self._threshold_voltage = np.zeros(self.varshape, dtype=dftype)
        self._threshold = np.full(self.varshape, th_inf, dtype=dftype)

    def get_spike(self, V: ArrayLike = None):
        r"""Generate surrogate spike signal from membrane potential.

        Computes a differentiable spike signal by scaling membrane potential
        relative to threshold range and applying the surrogate gradient function.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential in mV. If None, uses ``self.V.value``.
            Shape: ``(*batch_dims, *in_size)``.

        Returns
        -------
        spike : ArrayLike
            Surrogate spike output in [0, 1], same shape as input.
            Values near 1 indicate spiking neurons.

        Notes
        -----
        Scaling: :math:`v_\mathrm{scaled} = (V - V_\mathrm{th}) / (V_\mathrm{th} - V_\mathrm{reset})`

        This method is used internally for gradient computation but does not
        affect the discrete spike logic in ``update()``.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V_rel (mV), plus dg_<k> and g_<k> for each receptor port k
            (nS and nS respectively) — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, V_reset_rel, G, C_m, E_L,
            I_e, asc_sum — mutable auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        V = u.math.where(is_refractory, extra.V_reset_rel, state.V_rel)

        # Synaptic current: I_syn = sum_k g_k * (V + E_L - E_rev_k)
        # Units: nS * mV = pA
        I_syn = jnp.zeros_like(u.get_mantissa(state.V_rel)) * u.nS * u.mV
        for k in range(self._n_receptors):
            g_k = state['g_%d' % k]
            I_syn = I_syn + g_k * (V + extra.E_L - self.E_rev[k] * u.mV)

        # Leak current: I_leak = G * V (V is relative to E_L)
        I_leak = extra.G * V

        # dV/dt
        dV_raw = (-I_leak - I_syn + extra.I_e + extra.i_stim + extra.asc_sum) / extra.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        derivs = DotDict(V_rel=dV)

        # Alpha function dynamics for each receptor
        for k in range(self._n_receptors):
            dg_k = state['dg_%d' % k]
            g_k = state['g_%d' % k]
            tau_k = self.tau_syn[k] * u.ms
            derivs['dg_%d' % k] = -dg_k / tau_k
            derivs['g_%d' % k] = dg_k - g_k / tau_k

        return derivs

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            ODE state variables including V_rel and per-receptor dg/g.
        extra : DotDict
            Auxiliary data including spike_mask, r, unstable, threshold, etc.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & (state.V_rel < -1e3 * u.mV)
        )

        refr_accept = accept & (extra.r > 0)
        new_V_rel = u.math.where(refr_accept, extra.V_reset_rel, state.V_rel)

        spike_now = accept & (extra.r <= 0) & (new_V_rel > extra.threshold * u.mV)
        spike_mask = extra.spike_mask | spike_now

        # For GLIF1/3: simple reset
        # For GLIF2/4/5: biologically defined reset
        if not self.has_theta_spike:
            reset_V = extra.V_reset_rel
        else:
            reset_V = self.voltage_reset_fraction * extra.v_old + self.voltage_reset_add * u.mV

        new_V_rel = u.math.where(spike_now, reset_V, new_V_rel)
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count, extra.r)

        new_state = DotDict({**state, 'V_rel': new_V_rel})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

    def _collect_receptor_delta_inputs(self):
        r"""Collect and route delta inputs to receptor ports.

        Scans ``self.delta_inputs`` dict and routes inputs to receptor ports based
        on key naming convention. Keys containing ``'receptor_<k>'`` (0-based index)
        are routed to port ``k``; all other inputs default to receptor 0.

        Returns
        -------
        dg : list[np.ndarray]
            List of length ``n_receptors``. Each element is a float64 array of
            shape ``(*in_size)`` containing conductance jumps in nS
            for that receptor port. Summed across all matching input keys.

        Notes
        -----
        **Routing Logic**

        - Input key ``'exc_receptor_1'`` → receptor port 1
        - Input key ``'inh_receptor_2'`` → receptor port 2
        - Input key ``'external'`` → receptor port 0 (default)

        Callables in ``delta_inputs`` are invoked and removed from the dict.
        Non-callable values are assumed to be immediate and removed after use.

        **Unit Conversion**

        All inputs are converted to nS and broadcast to the current state shape.
        """
        dftype = brainstate.environ.dftype()
        v_shape = self.V.value.shape
        dg = [np.zeros(v_shape, dtype=dftype) for _ in range(self._n_receptors)]

        if self.delta_inputs is None:
            return dg

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            out_nS = np.asarray(u.math.asarray(out / u.nS), dtype=dftype)
            out_nS = np.broadcast_to(out_nS, v_shape)

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

    def update(self, x=0.0 * u.pA):
        r"""Perform a single simulation step with GLIF dynamics.

        Executes the full GLIF update cycle: ODE integration via RKF45, threshold
        computation (spike/voltage-dependent components if enabled), spike detection,
        reset rules, refractory handling, and synaptic input application.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input in pA. Shape: scalar or ``(*in_size,)``.
            Applied with one-step delay (buffered to ``I_stim`` and used in
            next step). Default: 0.0 pA.

        Returns
        -------
        spike : jax.Array
            Binary spike output (float32), shape ``(*in_size)``.
            Values are 1.0 for spiking neurons, 0.0 otherwise. This is a discrete
            binary signal, not the surrogate gradient output.

        Notes
        -----
        **Update Sequence (Per Simulation Step)**

        1. **State extraction**: Extract current state as numpy float64 arrays
           (matching NEST's double precision).
        2. **Parameter setup**: Prepare decay rates, stable coefficients, and
           integration parameters (computed once per step).
        3. **ODE integration**: Integrate synaptic and membrane dynamics via
           adaptive RKF45 with in-loop spike/reset events.
        4. **Post-integration**: Apply GLIF-specific threshold dynamics,
           ASC decay and update, voltage-dependent threshold (GLIF5).
        5. **Spike check and reset**: If V > threshold, emit spike and apply
           model-specific reset rules.
        6. **Synaptic input application**: Add incoming spike conductance jumps
           (scaled by :math:`e/\tau_\mathrm{syn}`) to all receptor ports.
        7. **Current buffering**: Update ``I_stim`` with new external current ``x``
           (applied in next step).
        8. **State writeback**: Convert numpy results back to JAX arrays with units.

        **Refractory Behavior**

        During refractory period (``refractory_step_count > 0``):

        - Membrane potential is clamped to previous value (no integration of :math:`dV/dt`)
        - Synaptic conductances continue to evolve normally
        - Spike detection is disabled
        - Threshold components continue to decay
        - Refractory counter is decremented each step

        **Receptor Port Routing**

        Delta inputs (from projections) are routed to receptor ports based on key
        naming. See ``_collect_receptor_delta_inputs()`` for details.

        **Numerical Precision**

        All computations use numpy float64 (matching NEST's C++ ``double``) to ensure
        exact replication of NEST results. Final state is converted to JAX arrays
        for gradient computation compatibility.

        **Failure Modes**


        - If RKF45 integration fails to converge, the method proceeds with the
          current state (no exception raised).
        - If threshold parameters cause continuous spiking (violating stability
          constraint), spike output will be 1.0 every step.

        See Also
        --------
        get_spike : Compute surrogate spike signal for gradient computation
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        v_shape = self.V.value.shape

        # Extract state as numpy float64
        E_L_mV = float(np.asarray(u.math.asarray(self.E_L / u.mV), dtype=dftype))
        V_abs = np.broadcast_to(
            np.asarray(u.math.asarray(self.V.value / u.mV), dtype=dftype), v_shape
        ).copy()
        V_rel = V_abs - E_L_mV  # relative to E_L

        dg_all = [
            np.broadcast_to(
                np.asarray(u.math.asarray(self.dg_syn[k].value / u.nS), dtype=dftype), v_shape
            ).copy()
            for k in range(self._n_receptors)
        ]
        g_all = [
            np.broadcast_to(
                np.asarray(u.math.asarray(self.g_syn[k].value / u.nS), dtype=dftype), v_shape
            ).copy()
            for k in range(self._n_receptors)
        ]
        r = np.broadcast_to(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=ditype), v_shape
        ).copy()
        i_stim = np.broadcast_to(
            np.asarray(u.math.asarray(self.I_stim.value / u.pA), dtype=dftype), v_shape
        ).copy()
        h_int = np.broadcast_to(
            np.asarray(u.math.asarray(self.integration_step.value / u.ms), dtype=dftype), v_shape
        ).copy()

        # Parameters
        G = float(np.asarray(u.math.asarray(self.g_m / u.nS), dtype=dftype))
        C_m = float(np.asarray(u.math.asarray(self.C_m / u.pF), dtype=dftype))
        V_reset_rel = float(np.asarray(u.math.asarray(self.V_reset / u.mV), dtype=dftype)) - E_L_mV
        t_ref_ms = float(np.asarray(u.math.asarray(self.t_ref / u.ms), dtype=dftype))
        I_e = float(np.asarray(u.math.asarray(self.I_e / u.pA), dtype=dftype))

        refr_counts = np.broadcast_to(
            np.asarray(u.math.asarray(self.ref_count), dtype=ditype), v_shape
        )

        # Record v_old for voltage-dependent threshold
        v_old = V_rel.copy()

        # Current input for next step (one-step delay).
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)

        # ---- Adaptive RKF45 integration via generic integrator ----
        ode_state = DotDict(V_rel=V_rel * u.mV)
        for k in range(self._n_receptors):
            ode_state['dg_%d' % k] = dg_all[k] * u.nS
            ode_state['g_%d' % k] = g_all[k] * u.nS

        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim * u.pA,
            V_reset_rel=V_reset_rel * u.mV,
            G=G * u.nS,
            C_m=C_m * u.pF,
            E_L=E_L_mV * u.mV,
            I_e=I_e * u.pA,
            asc_sum=self._ASCurrents_sum * u.pA,
            threshold=self._threshold,
            v_old=v_old * u.mV,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h_int * u.ms, extra=extra)

        V_rel_new = ode_state.V_rel
        dg_new = [ode_state['dg_%d' % k] for k in range(self._n_receptors)]
        g_new = [ode_state['g_%d' % k] for k in range(self._n_receptors)]
        spike_mask_jax = extra.spike_mask
        r_new = extra.r
        unstable = extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in glif_cond dynamics.'
        )

        # Convert integrated state back to numpy for GLIF-specific post-processing
        V_rel_np = np.asarray(u.math.asarray(V_rel_new / u.mV), dtype=dftype)
        spike_mask = np.asarray(spike_mask_jax, dtype=bool)
        r_np = np.asarray(r_new, dtype=ditype)

        # ---- GLIF-specific post-integration: threshold dynamics ----
        # Pre-compute decay rates (matching NEST pre_run_hook)
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
            potential_decay_rate = np.exp(-G * dt / C_m)
            theta_voltage_decay_rate_inverse = 1.0 / np.exp(self.th_voltage_decay * dt)
            phi = self.th_voltage_index / (self.th_voltage_decay - G / C_m)
            abpara_ratio_voltage = self.th_voltage_index / self.th_voltage_decay

        # Per-neuron GLIF threshold and ASC processing
        # (These are numpy-based for exact NEST replication)
        spike_mask_final = np.zeros(v_shape, dtype=bool)
        V_final = np.empty(v_shape, dtype=dftype)
        r_final = np.empty(v_shape, dtype=ditype)

        for idx in np.ndindex(v_shape):
            is_refractory = r_np[idx] > 0

            if not is_refractory:
                # Decay spike threshold component
                if self.has_theta_spike:
                    self._threshold_spike[idx] *= theta_spike_decay_rate

                # Calculate ASC (exact mean and decay)
                asc_sum_new = 0.0
                if self.has_asc:
                    for a in range(n_asc):
                        asc_sum_new += asc_stable_coeff[a] * self._ASCurrents[a][idx]
                        self._ASCurrents[a][idx] *= asc_decay_rates[a]
                self._ASCurrents_sum[idx] = asc_sum_new

                # Voltage-dependent threshold
                if self.has_theta_voltage:
                    beta = (I_e + i_stim[idx] + asc_sum_new) / G
                    self._threshold_voltage[idx] = (
                        phi * (v_old[idx] - beta) * potential_decay_rate
                        + theta_voltage_decay_rate_inverse * (
                            self._threshold_voltage[idx]
                            - phi * (v_old[idx] - beta)
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

                # Check for spike
                if V_rel_np[idx] > self._threshold[idx]:
                    spike_mask_final[idx] = True

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
                        V_rel_np[idx] = V_reset_rel
                    else:
                        # GLIF2/4/5: biologically defined reset
                        V_rel_np[idx] = self.voltage_reset_fraction * v_old[idx] + self.voltage_reset_add

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

                    r_final[idx] = r_i
                else:
                    r_final[idx] = 0
                V_final[idx] = V_rel_np[idx]
            else:
                # Refractory: decrement, hold voltage
                r_final[idx] = r_np[idx] - 1
                V_final[idx] = v_old[idx]
                self._threshold[idx] = (
                    self._threshold_spike[idx]
                    + self._threshold_voltage[idx]
                    + self._th_inf
                )

        # Get per-receptor synaptic inputs
        dg_input = self._collect_receptor_delta_inputs()

        # CondInitialValues: e / tau_syn (matching NEST)
        cond_init_vals = [np.e / self.tau_syn[k] for k in range(self._n_receptors)]

        # Extract integrated synaptic state back to numpy for spike input application
        dg_next = [
            np.asarray(u.math.asarray(dg_new[k] / u.nS), dtype=dftype)
            for k in range(self._n_receptors)
        ]
        g_next = [
            np.asarray(u.math.asarray(g_new[k] / u.nS), dtype=dftype)
            for k in range(self._n_receptors)
        ]

        # Add incoming spike conductance jumps (after everything else)
        for k in range(self._n_receptors):
            dg_next[k] = dg_next[k] + dg_input[k] * cond_init_vals[k]

        # Extract h from integrator output
        h_out = np.asarray(u.math.asarray(h / u.ms), dtype=dftype)

        # Write back state
        self.V.value = (V_final + E_L_mV) * u.mV  # convert back to absolute
        for k in range(self._n_receptors):
            self.dg_syn[k].value = dg_next[k] * u.nS
            self.g_syn[k].value = g_next[k] * u.nS
        self.refractory_step_count.value = jnp.asarray(r_final, dtype=ditype)
        self.integration_step.value = h_out * u.ms
        self.I_stim.value = new_i_stim_q + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask_final, t + dt_q, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return jnp.asarray(spike_mask_final, dtype=jnp.float32)
