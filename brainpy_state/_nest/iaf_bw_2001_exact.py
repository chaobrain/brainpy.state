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
from typing import Callable, Hashable, Iterable

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest._base import NESTNeuron

__all__ = [
    'iaf_bw_2001_exact',
]


class iaf_bw_2001_exact(NESTNeuron):
    r"""NEST-compatible conductance-based LIF neuron with exact per-synapse NMDA dynamics.

    This model implements the Brunel-Wang (2001) neuron with exact NMDA kinetics, maintaining
    separate rise and decay variables for each NMDA synapse without presynaptic-jump approximation.
    Each NMDA connection is assigned a unique port with a fixed weight, enforcing NEST's constraint
    that NMDA connections cannot be added after the first simulation step.

    Parameters
    ----------
    in_size : int, tuple of int, Sequence of int
        Population shape. Defines the number and arrangement of neurons.
    E_L : ArrayLike, optional
        Leak reversal potential. Default: -70 mV.
        Determines the resting potential in the absence of input.
    E_ex : ArrayLike, optional
        Excitatory reversal potential. Default: 0 mV.
        Reversal potential for AMPA and NMDA receptors.
    E_in : ArrayLike, optional
        Inhibitory reversal potential. Default: -70 mV.
        Reversal potential for GABA receptors.
    V_th : ArrayLike, optional
        Spike threshold potential. Default: -55 mV.
        Membrane potential at which a spike is emitted.
    V_reset : ArrayLike, optional
        Reset potential. Default: -60 mV.
        Membrane potential immediately after spike emission. Must be < V_th.
    C_m : ArrayLike, optional
        Membrane capacitance. Default: 500 pF.
        Must be strictly positive.
    g_L : ArrayLike, optional
        Leak conductance. Default: 25 nS.
        Conductance through passive leak channels.
    t_ref : ArrayLike, optional
        Absolute refractory period duration. Default: 2 ms.
        Time after spike during which membrane is clamped to V_reset.
    tau_AMPA : ArrayLike, optional
        AMPA decay time constant. Default: 2 ms.
        Governs exponential decay of AMPA conductance. Must be > 0.
    tau_GABA : ArrayLike, optional
        GABA decay time constant. Default: 5 ms.
        Governs exponential decay of GABA conductance. Must be > 0.
    tau_rise_NMDA : ArrayLike, optional
        NMDA rise time constant. Default: 2 ms.
        Time constant for NMDA activation variable x_j. Must be > 0.
    tau_decay_NMDA : ArrayLike, optional
        NMDA decay time constant. Default: 100 ms.
        Time constant for NMDA gating variable s_j. Must be > 0.
    alpha : ArrayLike, optional
        NMDA rise coupling strength. Default: 0.5 / ms.
        Scales the coupling between rise (x_j) and gating (s_j) variables. Must be > 0.
    conc_Mg2 : ArrayLike, optional
        Extracellular magnesium concentration. Default: 1 mM.
        Controls voltage-dependent NMDA blockade. Must be > 0.
    gsl_error_tol : ArrayLike, optional
        RKF45 local error tolerance. Default: 1e-3.
        Controls adaptive step size in Runge-Kutta-Fehlberg integration. Must be > 0.
        Smaller values improve accuracy at the cost of more iterations.
    V_initializer : Callable, optional
        Membrane potential initializer. Default: Constant(-70 mV).
        Function that generates initial V_m values.
    s_AMPA_initializer : Callable, optional
        AMPA conductance state initializer. Default: Constant(0 nS).
        Function that generates initial s_AMPA values.
    s_GABA_initializer : Callable, optional
        GABA conductance state initializer. Default: Constant(0 nS).
        Function that generates initial s_GABA values.
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation. Default: ReluGrad().
        Maps scaled voltage to differentiable spike output.
    spk_reset : str, optional
        Spike reset mode. Default: 'hard'.
        - 'hard': Stop gradient through reset (matches NEST)
        - 'soft': Gradient flows through reset (V -= V_th)
    ref_var : bool, optional
        If True, expose boolean refractory state variable. Default: False.
        Adds a `refractory` attribute for monitoring refractory state.
    name : str, optional
        Module name. Default: None (auto-generated).

    Raises
    ------
    ValueError
        If V_reset >= V_th, or any of C_m, tau_*, alpha, conc_Mg2, gsl_error_tol <= 0.
    ValueError
        If attempting to change NMDA port weights after first registration.
    ValueError
        If attempting to add new NMDA ports after first :meth:`update` call.
    ValueError
        If NMDA port is not hashable.
    ValueError
        If spike event format is invalid.

    See Also
    --------
    iaf_bw_2001 : Approximate version using presynaptic-jump NMDA dynamics
    iaf_cond_exp : Simpler conductance-based LIF without NMDA
    aeif_cond_alpha : Adaptive exponential IF with alpha-shaped conductances

    Parameter Mapping
    -----------------

    ============================ ======================== ============================================
    **NEST Parameter**           **brainpy.state**        **Notes**
    ============================ ======================== ============================================
    ``E_L``                      ``E_L``                  Leak reversal potential (mV)
    ``E_ex``                     ``E_ex``                 Excitatory reversal (mV)
    ``E_in``                     ``E_in``                 Inhibitory reversal (mV)
    ``V_th``                     ``V_th``                 Spike threshold (mV)
    ``V_reset``                  ``V_reset``              Reset potential (mV)
    ``C_m``                      ``C_m``                  Membrane capacitance (pF)
    ``g_L``                      ``g_L``                  Leak conductance (nS)
    ``t_ref``                    ``t_ref``                Refractory period (ms)
    ``tau_AMPA``                 ``tau_AMPA``             AMPA decay time (ms)
    ``tau_GABA``                 ``tau_GABA``             GABA decay time (ms)
    ``tau_rise_NMDA``            ``tau_rise_NMDA``        NMDA rise time (ms)
    ``tau_decay_NMDA``           ``tau_decay_NMDA``       NMDA decay time (ms)
    ``alpha``                    ``alpha``                NMDA coupling (1/ms)
    ``conc_Mg2``                 ``conc_Mg2``             Mg²⁺ concentration (mM)
    ``gsl_error_tol``            ``gsl_error_tol``        RKF45 tolerance (dimensionless)
    ============================ ======================== ============================================

    Mathematical Model
    ------------------

    **1. Membrane Dynamics**

    The subthreshold membrane potential evolves according to:

    .. math::

       C_m \frac{dV_m}{dt} = -g_L(V_m - E_L) - I_{syn} + I_{stim}

    where :math:`I_{syn} = I_{AMPA} + I_{GABA} + I_{NMDA}` is the total synaptic current.

    **2. Synaptic Currents**

    AMPA and GABA currents are ohmic:

    .. math::

       I_{AMPA} &= (V_m - E_{ex}) s_{AMPA} \\
       I_{GABA} &= (V_m - E_{in}) s_{GABA}

    NMDA current includes voltage-dependent Mg²⁺ blockade:

    .. math::

       I_{NMDA} = \frac{(V_m - E_{ex})}{1 + [Mg^{2+}]\exp(-0.062V_m)/3.57} \sum_j w_j s_j

    where :math:`j` indexes individual NMDA synapses, :math:`w_j` is the fixed weight for port :math:`j`,
    and :math:`s_j` is the gating variable for that synapse.

    **3. Synaptic Gating Variables**

    AMPA and GABA conductances decay exponentially:

    .. math::

       \frac{ds_{AMPA}}{dt} &= -\frac{s_{AMPA}}{\tau_{AMPA}} \\
       \frac{ds_{GABA}}{dt} &= -\frac{s_{GABA}}{\tau_{GABA}}

    Each NMDA synapse :math:`j` has dual-timescale kinetics:

    .. math::

       \frac{dx_j}{dt} &= -\frac{x_j}{\tau_{NMDA,rise}} \\
       \frac{ds_j}{dt} &= -\frac{s_j}{\tau_{NMDA,decay}} + \alpha x_j (1-s_j)

    where :math:`x_j` is the rise variable (fast activation) and :math:`s_j` is the gating variable
    (slow inactivation with saturation).

    **4. Spike Generation and Reset**

    When :math:`V_m \geq V_{th}` and the neuron is not refractory:

    - Emit a spike
    - Set :math:`V_m \leftarrow V_{reset}`
    - Enter refractory state for :math:`t_{ref}` ms

    During refractoriness, :math:`V_m` is clamped to :math:`V_{reset}`.

    **5. Numerical Integration**

    The continuous dynamics are integrated using adaptive Runge-Kutta-Fehlberg (RKF45) with:

    - 4th and 5th order embedded methods for error estimation
    - Persistent step size :math:`h` that adapts to maintain local error < ``gsl_error_tol``
    - Minimum step size :math:`h_{min} = 10^{-8}` ms
    - Maximum iterations per simulation step: 10,000

    **NMDA Port Semantics**

    NEST assigns each NMDA connection a unique receptor port at connect time and prohibits adding
    new NMDA connections after the first simulation step. This implementation mirrors that behavior:

    - Each NMDA event requires a ``port`` identifier (any hashable value)
    - The first event for a new port registers that port with the provided weight
    - Subsequent events to the same port must use the same weight (enforced)
    - New ports can only be added before the first :meth:`update` call
    - AMPA/GABA events do not use ports (weights accumulate directly)

    **Spike Event Formats**

    The ``spike_events`` parameter accepts multiple formats:

    **Tuple formats:**

    - ``(receptor, weight)`` — receptor ∈ {1, 2, 3} or {'AMPA', 'GABA', 'NMDA'}
    - ``(receptor, weight, third)`` — ``third`` is multiplicity for AMPA/GABA, port for NMDA
    - ``(receptor, weight, port, multiplicity)`` — full NMDA specification

    **Dict format:**

    - Required keys: ``receptor_type`` or ``receptor`` (1/2/3 or 'AMPA'/'GABA'/'NMDA'), ``weight``
    - Optional keys: ``multiplicity`` (default 1.0), ``port``/``rport``/``synapse_id`` (for NMDA)

    **Update Ordering (matches NEST)**

    Each :meth:`update` call executes in this order:

    1. **Integrate ODEs** on :math:`(t, t+dt]` using RKF45 with persistent step size
    2. **Apply spike jumps**: add to ``s_AMPA``, ``s_GABA``, and ``x_j`` for each NMDA port
    3. **Threshold check and reset**: emit spikes, reset voltage, update refractory countdown
    4. **Store external current**: buffer ``I_stim`` for next step (one-step delay)

    **Recordable Variables**

    - ``V_m`` — Membrane potential (mV)
    - ``s_AMPA`` — AMPA conductance state (nS)
    - ``s_GABA`` — GABA conductance state (nS)
    - ``s_NMDA`` — Weighted sum of NMDA gating variables (nS), :math:`\sum_j w_j s_j`
    - ``I_AMPA`` — AMPA current (pA)
    - ``I_GABA`` — GABA current (pA)
    - ``I_NMDA`` — NMDA current (pA)

    Additional State Variables
    --------------------------

    - ``x_NMDA`` — NMDA rise variables for each port (shape: ``[*in_size, n_ports]``)
    - ``s_NMDA_components`` — NMDA gating variables for each port (shape: ``[*in_size, n_ports]``)
    - ``nmda_weights`` — Fixed weights for each NMDA port (shape: ``[*in_size, n_ports]``)
    - ``refractory_step_count`` — Remaining refractory steps (int32)
    - ``integration_step`` — Persistent RKF45 step size (ms)
    - ``I_stim`` — One-step delayed external current buffer (pA)
    - ``refractory`` — Boolean refractory indicator (only if ``ref_var=True``)

    **Performance Considerations:**

    - RKF45 integration is performed per-neuron in NumPy (not vectorized)
    - Computational cost scales linearly with the number of NMDA ports
    - Large ``gsl_error_tol`` reduces accuracy but improves speed
    - This model is significantly slower than ``iaf_bw_2001`` due to per-synapse state

    **Comparison to iaf_bw_2001:**

    - ``iaf_bw_2001`` approximates all NMDA synapses with a single pair of state variables
    - ``iaf_bw_2001_exact`` tracks rise and decay for each NMDA connection separately
    - Use ``iaf_bw_2001_exact`` when NMDA synapse heterogeneity matters (e.g., detailed working memory models)
    - Use ``iaf_bw_2001`` for large-scale simulations where approximation is acceptable

    References
    ----------
    .. [1] Wang X-J (1999). Synaptic basis of cortical persistent activity:
           The importance of NMDA receptors to working memory.
           Journal of Neuroscience, 19(21):9587-9603.
           DOI: https://doi.org/10.1523/JNEUROSCI.19-21-09587.1999
    .. [2] Brunel N, Wang X-J (2001). Effects of neuromodulation in a cortical
           network model of object working memory dominated by recurrent
           inhibition. Journal of Computational Neuroscience, 11(1):63-85.
           DOI: https://doi.org/10.1023/A:1011204814320
    .. [3] Wang X-J (2002). Probabilistic decision making by slow
           reverberation in cortical circuits. Neuron, 36(5):955-968.
           DOI: https://doi.org/10.1016/S0896-6273(02)01092-9
    .. [4] NEST Simulator. Models: iaf_bw_2001_exact.
           https://nest-simulator.readthedocs.io/en/stable/models/iaf_bw_2001_exact.html

    Examples
    --------
    **Basic usage with AMPA input:**

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.context(dt=0.1 * u.ms)
        >>> net = bp.iaf_bw_2001_exact(in_size=10)
        >>> net.init_all_states()
        >>> # Apply AMPA input spike
        >>> spike = bp.iaf_bw_2001_exact.get_spike(net(spike_events=[(1, 100*u.nS)]))
        >>> print(net.V.value)  # doctest: +SKIP

    **NMDA connections with unique ports:**

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.context(dt=0.1 * u.ms)
        >>> net = bp.iaf_bw_2001_exact(in_size=5)
        >>> net.init_all_states()
        >>> # Register two NMDA ports with different weights
        >>> events = [
        ...     (3, 50*u.nS, 'port_A', 1.0),  # NMDA port A, weight 50 nS
        ...     (3, 75*u.nS, 'port_B', 1.0),  # NMDA port B, weight 75 nS
        ... ]
        >>> spike = net(spike_events=events)
        >>> print(net.s_NMDA_components.value.shape)  # doctest: +SKIP
        (5, 2)  # 5 neurons × 2 NMDA ports

    **Mixing AMPA, GABA, and NMDA:**

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.context(dt=0.1 * u.ms)
        >>> net = bp.iaf_bw_2001_exact(in_size=1, V_th=-50*u.mV)
        >>> net.init_all_states()
        >>> events = [
        ...     {'receptor': 'AMPA', 'weight': 200*u.nS, 'multiplicity': 2.0},
        ...     {'receptor': 'GABA', 'weight': 100*u.nS},
        ...     {'receptor': 'NMDA', 'weight': 50*u.nS, 'port': 0},
        ... ]
        >>> for _ in range(100):
        ...     spike = net(spike_events=events if _ == 10 else None)
        >>> print(net.last_spike_time.value)  # doctest: +SKIP

    **Monitoring refractory state:**

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.context(dt=0.1 * u.ms)
        >>> net = bp.iaf_bw_2001_exact(in_size=3, ref_var=True, t_ref=5*u.ms)
        >>> net.init_all_states()
        >>> net.V.value = net.V_th + 1*u.mV  # Force spike
        >>> spike = net()
        >>> print(net.refractory.value)  # doctest: +SKIP
        [True True True]
    """

    __module__ = 'brainpy.state'

    AMPA = 1
    GABA = 2
    NMDA = 3

    RECEPTOR_TYPES = {
        'AMPA': AMPA,
        'GABA': GABA,
        'NMDA': NMDA,
    }

    RECORDABLES = (
        'V_m',
        's_AMPA',
        's_GABA',
        's_NMDA',
        'I_NMDA',
        'I_AMPA',
        'I_GABA',
    )

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 10000

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        E_ex: ArrayLike = 0. * u.mV,
        E_in: ArrayLike = -70. * u.mV,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -60. * u.mV,
        C_m: ArrayLike = 500. * u.pF,
        g_L: ArrayLike = 25. * u.nS,
        t_ref: ArrayLike = 2. * u.ms,
        tau_AMPA: ArrayLike = 2. * u.ms,
        tau_GABA: ArrayLike = 5. * u.ms,
        tau_rise_NMDA: ArrayLike = 2. * u.ms,
        tau_decay_NMDA: ArrayLike = 100. * u.ms,
        alpha: ArrayLike = 0.5 / u.ms,
        conc_Mg2: ArrayLike = 1.0 * u.mM,
        gsl_error_tol: ArrayLike = 1e-3,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        s_AMPA_initializer: Callable = braintools.init.Constant(0. * u.nS),
        s_GABA_initializer: Callable = braintools.init.Constant(0. * u.nS),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)

        self.tau_AMPA = braintools.init.param(tau_AMPA, self.varshape)
        self.tau_GABA = braintools.init.param(tau_GABA, self.varshape)
        self.tau_rise_NMDA = braintools.init.param(tau_rise_NMDA, self.varshape)
        self.tau_decay_NMDA = braintools.init.param(tau_decay_NMDA, self.varshape)
        self.alpha = braintools.init.param(alpha, self.varshape)
        self.conc_Mg2 = braintools.init.param(conc_Mg2, self.varshape)
        self.gsl_error_tol = braintools.init.param(gsl_error_tol, self.varshape)

        self.V_initializer = V_initializer
        self.s_AMPA_initializer = s_AMPA_initializer
        self.s_GABA_initializer = s_GABA_initializer
        self.ref_var = ref_var

        self._nmda_port_index = {}
        self._updates_started = False

        self._validate_parameters()

    @property
    def receptor_types(self):
        r"""Mapping of receptor names to numeric identifiers.

        Returns
        -------
        dict
            Dictionary mapping {'AMPA': 1, 'GABA': 2, 'NMDA': 3}.
        """
        return dict(self.RECEPTOR_TYPES)

    @property
    def recordables(self):
        r"""List of variables available for recording.

        Returns
        -------
        list of str
            ['V_m', 's_AMPA', 's_GABA', 's_NMDA', 'I_NMDA', 'I_AMPA', 'I_GABA'].
        """
        return list(self.RECORDABLES)

    @staticmethod
    def _value_to_float(x, unit=None):
        r"""Convert quantity with units to float64 NumPy array.

        Parameters
        ----------
        x : ArrayLike
            Input value, possibly with units.
        unit : brainunit.Unit, optional
            Target unit for division. If None, return dimensionless float.

        Returns
        -------
        np.ndarray
            Float64 array, dimensionless if unit is provided (x / unit), else raw conversion.
        """
        if unit is None:
            return np.asarray(u.math.asarray(x), dtype=np.float64)
        try:
            return np.asarray(u.math.asarray(x / unit), dtype=np.float64)
        except Exception:
            return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast array to target state shape.

        Parameters
        ----------
        x_np : np.ndarray
            Input array.
        shape : tuple of int
            Target shape.

        Returns
        -------
        np.ndarray
            Broadcasted view of input array with target shape.
        """
        return np.broadcast_to(x_np, shape)

    @classmethod
    def _normalize_spike_receptor(cls, receptor):
        r"""Normalize receptor identifier to numeric code.

        Parameters
        ----------
        receptor : str or int
            Receptor identifier. Accepts 'AMPA', 'GABA', 'NMDA', or numeric codes 1/2/3.

        Returns
        -------
        int
            Numeric receptor code (1=AMPA, 2=GABA, 3=NMDA).

        Raises
        ------
        ValueError
            If receptor is not recognized or is out of valid range [1, 3].
        """
        if isinstance(receptor, str):
            key = receptor.strip()
            if key in cls.RECEPTOR_TYPES:
                return cls.RECEPTOR_TYPES[key]
            if key.isdigit():
                receptor = int(key)
            else:
                raise ValueError(f'Unknown receptor label: {receptor}')
        receptor = int(receptor)
        if receptor < cls.AMPA or receptor > cls.NMDA:
            raise ValueError(f'Receptor type must be in [1, 3], got {receptor}.')
        return receptor

    @staticmethod
    def _normalize_nmda_port(port) -> Hashable:
        r"""Normalize NMDA port identifier to hashable value.

        Parameters
        ----------
        port : Hashable or None
            NMDA port identifier. Can be int, str, or any hashable type.
            If None, defaults to port 0.

        Returns
        -------
        Hashable
            Normalized port identifier. Numeric strings converted to int,
            None converted to 0, other hashable values returned as-is.

        Raises
        ------
        ValueError
            If port is not hashable.
        """
        if port is None:
            return 0
        if isinstance(port, str):
            p = port.strip()
            if p.isdigit():
                return int(p)
            return p
        try:
            hash(port)
        except TypeError as e:
            raise ValueError(f'NMDA port must be hashable, got {type(port)}.') from e
        return port

    def _validate_parameters(self):
        r"""Validate model parameters at initialization.

        Raises
        ------
        ValueError
            If V_reset >= V_th.
        ValueError
            If C_m, tau_AMPA, tau_GABA, tau_rise_NMDA, tau_decay_NMDA, alpha,
            conc_Mg2, or gsl_error_tol are non-positive.
        ValueError
            If t_ref is negative.
        """
        if np.any(self._value_to_float(self.V_reset, u.mV) >= self._value_to_float(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._value_to_float(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._value_to_float(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._value_to_float(self.tau_AMPA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._value_to_float(self.tau_GABA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._value_to_float(self.tau_rise_NMDA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._value_to_float(self.tau_decay_NMDA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._value_to_float(self.alpha, 1 / u.ms) <= 0.0):
            raise ValueError('alpha > 0 required.')
        if np.any(self._value_to_float(self.conc_Mg2, u.mM) <= 0.0):
            raise ValueError('Mg2 concentration must be strictly positive.')
        if np.any(self._value_to_float(self.gsl_error_tol, None) <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def _nmda_num_ports(self):
        if hasattr(self, 'x_NMDA'):
            return int(np.asarray(self.x_NMDA.value).shape[-1])
        return 0

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Creates and initializes membrane potential, synaptic conductances, currents,
        NMDA port arrays (initially empty), refractory state, and integration step size.
        NMDA port registry is cleared.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size for state variables. Default: None (no batching).
            If provided, adds a leading batch dimension to all state variables.
        **kwargs
            Additional keyword arguments (currently unused).

        Notes
        -----
        - NMDA port arrays (x_NMDA, s_NMDA_components, nmda_weights) start empty (shape: [..., 0])
        - Ports are allocated dynamically when first NMDA spike arrives
        - Clears the internal ``_nmda_port_index`` registry
        - Resets ``_updates_started`` flag to False
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        s_ampa = braintools.init.param(self.s_AMPA_initializer, self.varshape, batch_size)
        s_gaba = braintools.init.param(self.s_GABA_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.s_AMPA = brainstate.HiddenState(s_ampa)
        self.s_GABA = brainstate.HiddenState(s_gaba)

        state_shape = self.V.value.shape
        zeros = np.zeros(state_shape, dtype=np.float64)

        self.s_NMDA = brainstate.ShortTermState(zeros * u.nS)
        self.I_NMDA = brainstate.ShortTermState(zeros * u.pA)
        self.I_AMPA = brainstate.ShortTermState(zeros * u.pA)
        self.I_GABA = brainstate.ShortTermState(zeros * u.pA)

        self.x_NMDA = brainstate.ShortTermState(np.zeros(state_shape + (0,), dtype=np.float64))
        self.s_NMDA_components = brainstate.ShortTermState(np.zeros(state_shape + (0,), dtype=np.float64))
        self.nmda_weights = brainstate.ShortTermState(np.zeros(state_shape + (0,), dtype=np.float64))

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = brainstate.environ.get_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0. * u.pA), self.varshape, batch_size)
        )

        self._nmda_port_index = {}
        self._updates_started = False

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to initial values.

        Unlike :meth:`init_state`, this preserves NMDA port structure (number of ports
        and their weights remain unchanged). Resets voltage, conductances, currents,
        NMDA gating variables, refractory state, and integration step size.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size for state variables. Default: None (no batching).
            If provided, reshapes state variables with a leading batch dimension.
        **kwargs
            Additional keyword arguments (currently unused).

        Notes
        -----
        - NMDA port count and weights are preserved (but x_NMDA and s_NMDA_components are zeroed)
        - Does NOT clear ``_nmda_port_index`` (port registry persists)
        - Does NOT reset ``_updates_started`` flag
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.s_AMPA.value = braintools.init.param(self.s_AMPA_initializer, self.varshape, batch_size)
        self.s_GABA.value = braintools.init.param(self.s_GABA_initializer, self.varshape, batch_size)

        state_shape = self.V.value.shape
        zeros = np.zeros(state_shape, dtype=np.float64)
        self.s_NMDA.value = zeros * u.nS
        self.I_NMDA.value = zeros * u.pA
        self.I_AMPA.value = zeros * u.pA
        self.I_GABA.value = zeros * u.pA

        n_ports = self._nmda_num_ports()
        self.x_NMDA.value = np.zeros(state_shape + (n_ports,), dtype=np.float64)
        self.s_NMDA_components.value = np.zeros(state_shape + (n_ports,), dtype=np.float64)

        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)

        dt = brainstate.environ.get_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0. * u.pA), self.varshape, batch_size
        )

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        r"""Generate differentiable spike output from membrane potential.

        Scales voltage relative to threshold and applies surrogate gradient function
        for gradient-based learning. Voltage is scaled linearly between V_reset (0)
        and V_th (1).

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (mV). Default: None (uses current ``self.V.value``).
            Shape must match ``self.varshape`` or be broadcastable to it.

        Returns
        -------
        ArrayLike
            Differentiable spike output in [0, 1]. Shape matches input voltage.
            Values close to 1 indicate spiking; values close to 0 indicate quiescence.
            Exact output depends on ``self.spk_fun`` (e.g., ReLU, sigmoid, etc.).

        Notes
        -----
        - Used internally during :meth:`update` to compute spike output before reset
        - Scaling formula: :math:`v_{scaled} = (V - V_{th}) / (V_{th} - V_{reset})`
        - For hard reset mode, actual spike detection uses :math:`V \geq V_{th}`
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _ensure_nmda_port(self, port: Hashable, weight_np: np.ndarray, state_shape):
        if port in self._nmda_port_index:
            idx = self._nmda_port_index[port]
            current_weight = np.asarray(self.nmda_weights.value[..., idx], dtype=np.float64)
            if np.any(current_weight != weight_np):
                raise ValueError('iaf_bw_2001_exact requires constant weights per NMDA port.')
            return idx

        if self._updates_started:
            raise ValueError('NMDA ports can only be added before the first call to update().')

        idx = self._nmda_num_ports()
        self._nmda_port_index[port] = idx

        zero_channel = np.zeros(state_shape + (1,), dtype=np.float64)
        x_old = np.asarray(self.x_NMDA.value, dtype=np.float64)
        s_old = np.asarray(self.s_NMDA_components.value, dtype=np.float64)
        w_old = np.asarray(self.nmda_weights.value, dtype=np.float64)

        self.x_NMDA.value = np.concatenate([x_old, zero_channel], axis=-1)
        self.s_NMDA_components.value = np.concatenate([s_old, zero_channel], axis=-1)
        self.nmda_weights.value = np.concatenate([w_old, np.expand_dims(weight_np, axis=-1)], axis=-1)
        return idx

    def _parse_spike_events(self, spike_events: Iterable, state_shape):
        ds_ampa = np.zeros(state_shape, dtype=np.float64)
        ds_gaba = np.zeros(state_shape, dtype=np.float64)
        nmda_mult = np.zeros(state_shape + (self._nmda_num_ports(),), dtype=np.float64)

        if spike_events is None:
            return ds_ampa, ds_gaba, nmda_mult

        for ev in spike_events:
            receptor = 'AMPA'
            weight = 0.0 * u.nS
            multiplicity = 1.0
            port = None

            if isinstance(ev, dict):
                receptor = ev.get('receptor_type', ev.get('receptor', 'AMPA'))
                weight = ev.get('weight', 0.0 * u.nS)
                multiplicity = ev.get('multiplicity', 1.0)
                port = ev.get('port', ev.get('rport', ev.get('synapse_id', None)))
            else:
                if len(ev) == 2:
                    receptor, weight = ev
                elif len(ev) == 3:
                    receptor, weight, third = ev
                    receptor_id = self._normalize_spike_receptor(receptor)
                    if receptor_id == self.NMDA:
                        port = third
                    else:
                        multiplicity = third
                elif len(ev) == 4:
                    receptor, weight, port, multiplicity = ev
                else:
                    raise ValueError('Spike event tuples must have length 2, 3, or 4.')

            receptor_id = self._normalize_spike_receptor(receptor)
            weight_np = self._value_to_float(weight, u.nS)
            weight_np = np.broadcast_to(weight_np, state_shape)
            mult_np = self._value_to_float(multiplicity, None)
            mult_np = np.broadcast_to(mult_np, state_shape)

            if receptor_id == self.AMPA:
                ds_ampa = ds_ampa + weight_np * mult_np
            elif receptor_id == self.GABA:
                ds_gaba = ds_gaba + weight_np * mult_np
            else:
                nmda_port = self._normalize_nmda_port(port)
                nmda_idx = self._ensure_nmda_port(nmda_port, weight_np, state_shape)
                if nmda_idx >= nmda_mult.shape[-1]:
                    pad = np.zeros(state_shape + (nmda_idx + 1 - nmda_mult.shape[-1],), dtype=np.float64)
                    nmda_mult = np.concatenate([nmda_mult, pad], axis=-1)
                nmda_mult[..., nmda_idx] = nmda_mult[..., nmda_idx] + mult_np

        return ds_ampa, ds_gaba, nmda_mult

    def _parse_registered_spike_inputs(self, state_shape):
        ds_ampa = np.zeros(state_shape, dtype=np.float64)
        ds_gaba = np.zeros(state_shape, dtype=np.float64)
        if self.delta_inputs is None:
            return ds_ampa, ds_gaba

        for key in tuple(self.delta_inputs.keys()):
            val = self.delta_inputs[key]
            if callable(val):
                val = val()
            else:
                self.delta_inputs.pop(key)

            label = None
            if ' // ' in key:
                label, _ = key.split(' // ', maxsplit=1)

            if label is None:
                receptor = self.AMPA
            else:
                receptor = self._normalize_spike_receptor(label)

            if receptor == self.NMDA:
                raise ValueError('Use spike_events with NMDA port specification for iaf_bw_2001_exact.')

            val_np = self._value_to_float(val, u.nS)
            val_np = np.broadcast_to(val_np, state_shape)
            if receptor == self.AMPA:
                ds_ampa = ds_ampa + val_np
            else:
                ds_gaba = ds_gaba + val_np

        return ds_ampa, ds_gaba

    @staticmethod
    def _nmda_currents_scalar(v, s_ampa, s_gaba, s_nmda_sum, p):
        r"""Compute synaptic currents for a single neuron (scalar computation).

        Parameters
        ----------
        v : float
            Membrane potential (mV).
        s_ampa : float
            AMPA conductance state (nS).
        s_gaba : float
            GABA conductance state (nS).
        s_nmda_sum : float
            Weighted sum of NMDA gating variables (nS), sum_j w_j s_j.
        p : dict
            Parameter dictionary with keys 'E_ex', 'E_in', 'conc_Mg2' (all floats in base units).

        Returns
        -------
        i_ampa : float
            AMPA current (pA).
        i_gaba : float
            GABA current (pA).
        i_nmda : float
            NMDA current with Mg2+ blockade (pA).

        Notes
        -----
        NMDA Mg2+ blockade: denom = 1 + [Mg2+] * exp(-0.062 * V) / 3.57
        """
        i_ampa = (v - p['E_ex']) * s_ampa
        i_gaba = (v - p['E_in']) * s_gaba
        denom = 1.0 + p['conc_Mg2'] * math.exp(-0.062 * v) / 3.57
        i_nmda = (v - p['E_ex']) / denom * s_nmda_sum
        return i_ampa, i_gaba, i_nmda

    @classmethod
    def _dynamics_scalar(cls, y, i_stim, p, nmda_weights):
        r"""Compute ODE right-hand side for a single neuron (scalar computation).

        Parameters
        ----------
        y : np.ndarray
            State vector with layout: [V_m, s_AMPA, s_GABA, x_0, ..., x_{n-1}, s_0, ..., s_{n-1}]
            where n is the number of NMDA ports. Shape: (3 + 2*n_nmda,).
        i_stim : float
            External input current (pA).
        p : dict
            Parameter dictionary (all values in base units).
        nmda_weights : np.ndarray
            Fixed NMDA weights for each port (nS). Shape: (n_nmda,).

        Returns
        -------
        dy : np.ndarray
            Time derivatives dy/dt with same shape as y.
        i_ampa : float
            AMPA current (pA).
        i_gaba : float
            GABA current (pA).
        i_nmda : float
            NMDA current (pA).
        s_nmda_sum : float
            Weighted NMDA sum (nS).

        Notes
        -----
        ODE system:
            dV/dt = (-g_L(V - E_L) - I_syn + I_stim) / C_m
            ds_AMPA/dt = -s_AMPA / tau_AMPA
            ds_GABA/dt = -s_GABA / tau_GABA
            dx_j/dt = -x_j / tau_rise_NMDA
            ds_j/dt = -s_j / tau_decay_NMDA + alpha * x_j * (1 - s_j)
        """
        n_nmda = int(nmda_weights.shape[0])
        v = float(y[0])
        s_ampa = float(y[1])
        s_gaba = float(y[2])

        if n_nmda > 0:
            x_nmda = y[3:3 + n_nmda]
            s_nmda = y[3 + n_nmda:]
            s_nmda_sum = float(np.dot(s_nmda, nmda_weights))
        else:
            x_nmda = np.zeros((0,), dtype=np.float64)
            s_nmda = np.zeros((0,), dtype=np.float64)
            s_nmda_sum = 0.0

        i_ampa, i_gaba, i_nmda = cls._nmda_currents_scalar(v, s_ampa, s_gaba, s_nmda_sum, p)
        i_syn = i_ampa + i_gaba + i_nmda

        dy = np.zeros_like(y, dtype=np.float64)
        dy[0] = (-p['g_L'] * (v - p['E_L']) - i_syn + i_stim) / p['C_m']
        dy[1] = -s_ampa / p['tau_AMPA']
        dy[2] = -s_gaba / p['tau_GABA']

        if n_nmda > 0:
            dy[3:3 + n_nmda] = -x_nmda / p['tau_rise_NMDA']
            dy[3 + n_nmda:] = -s_nmda / p['tau_decay_NMDA'] + p['alpha'] * x_nmda * (1.0 - s_nmda)

        return dy, i_ampa, i_gaba, i_nmda, s_nmda_sum

    def _rkf45_integrate_scalar(self, y0, i_stim, h0, dt, p, nmda_weights, atol):
        r"""Integrate ODEs for a single neuron using adaptive RKF45 method.

        Uses Runge-Kutta-Fehlberg (RKF45) with embedded 4th and 5th order methods
        for local error estimation and adaptive step size control.

        Parameters
        ----------
        y0 : np.ndarray
            Initial state vector [V_m, s_AMPA, s_GABA, x_0, ..., s_0, ...]. Shape: (3 + 2*n_nmda,).
        i_stim : float
            External input current (pA).
        h0 : float
            Initial step size (ms).
        dt : float
            Total integration interval (ms).
        p : dict
            Parameter dictionary (all values in base units).
        nmda_weights : np.ndarray
            Fixed NMDA weights (nS). Shape: (n_nmda,).
        atol : float
            Absolute local error tolerance (dimensionless).

        Returns
        -------
        y : np.ndarray
            Final state vector after integration.
        h : float
            Final adaptive step size (ms), used as initial step for next time step.
        i_ampa : float
            Final AMPA current (pA).
        i_gaba : float
            Final GABA current (pA).
        i_nmda : float
            Final NMDA current (pA).
        s_nmda_sum : float
            Final weighted NMDA sum (nS).

        Notes
        -----
        **Adaptive step size control:**

        - If local error < atol, accept step and increase h by factor <= 5
        - If local error > atol, reject step and decrease h by factor >= 0.2
        - Step size clipped to [MIN_H, dt - t] where MIN_H = 1e-8 ms
        - Maximum iterations: 10,000 (to prevent infinite loops)

        **RKF45 coefficients:**

        - 4th order: Butcher tableau with (25/216, 0, 1408/2565, 2197/4104, -1/5, 0)
        - 5th order: (16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55)
        - Error estimate: max|y5 - y4|

        **Persistence:**

        - Returned step size h is stored in ``integration_step`` state
        - Provides continuity across time steps for smooth adaptation
        """
        t = 0.0
        h = max(h0, self._MIN_H)
        y = np.asarray(y0, dtype=np.float64)
        iters = 0

        i_ampa = 0.0
        i_gaba = 0.0
        i_nmda = 0.0
        s_nmda_sum = 0.0

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = max(self._MIN_H, min(h, dt - t))

            k1, *_ = self._dynamics_scalar(y, i_stim, p, nmda_weights)
            k2, *_ = self._dynamics_scalar(y + h * (1.0 / 4.0) * k1, i_stim, p, nmda_weights)
            k3, *_ = self._dynamics_scalar(
                y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0),
                i_stim,
                p,
                nmda_weights,
            )
            k4, *_ = self._dynamics_scalar(
                y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
                i_stim,
                p,
                nmda_weights,
            )
            k5, *_ = self._dynamics_scalar(
                y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
                i_stim,
                p,
                nmda_weights,
            )
            k6, *_ = self._dynamics_scalar(
                y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0),
                i_stim,
                p,
                nmda_weights,
            )

            y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
            y5 = y + h * (
                16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0
            )
            err = float(np.max(np.abs(y5 - y4)))

            if err <= atol or h <= self._MIN_H:
                y = y5
                t += h
                fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
                h = max(self._MIN_H, h * fac)
            else:
                fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
                h = max(self._MIN_H, h * fac)

            _, i_ampa, i_gaba, i_nmda, s_nmda_sum = self._dynamics_scalar(y, i_stim, p, nmda_weights)

        return y, h, i_ampa, i_gaba, i_nmda, s_nmda_sum

    def update(self, x=0. * u.pA, spike_events=None):
        r"""Advance neuron state by one simulation time step.

        Performs RKF45 integration of ODEs, applies spike jumps to conductances,
        checks threshold, resets spiking neurons, and updates refractory state.
        External current is buffered with one-step delay (NEST compatibility).

        Parameters
        ----------
        x : ArrayLike, optional
            External input current (pA). Default: 0 pA.
            Shape must match ``self.varshape`` or be broadcastable to it.
            Summed with registered ``current_inputs`` to form total stimulus.
        spike_events : iterable, optional
            Collection of synaptic spike events. Default: None (no spikes).
            Each event can be a tuple or dict specifying receptor, weight, multiplicity, and port.

            **Tuple formats:**

            - ``(receptor, weight)``
            - ``(receptor, weight, third)`` where ``third`` is multiplicity for AMPA/GABA, port for NMDA
            - ``(receptor, weight, port, multiplicity)`` for full NMDA specification

            **Dict format:**

            - ``receptor_type`` or ``receptor``: int (1/2/3) or str ('AMPA'/'GABA'/'NMDA')
            - ``weight``: ArrayLike (nS), synaptic weight
            - ``multiplicity``: float, optional (default 1.0)
            - ``port`` / ``rport`` / ``synapse_id``: Hashable, optional (required for NMDA)

        Returns
        -------
        ArrayLike
            Differentiable spike output for current time step. Shape: ``self.varshape``.
            Computed from voltage before reset using ``self.get_spike()``.

        Raises
        ------
        ValueError
            If attempting to add new NMDA ports after first :meth:`update` call.
        ValueError
            If NMDA port weight changes after initial registration.
        ValueError
            If spike event format is invalid.

        Notes
        -----
        **Update sequence (matches NEST ordering):**

        1. **RKF45 integration**: Integrate V_m, s_AMPA, s_GABA, x_NMDA, s_NMDA on (t, t+dt]
        2. **Spike jumps**: Add to s_AMPA, s_GABA (weight × multiplicity), x_NMDA (multiplicity only)
        3. **Threshold check**: If V_m >= V_th and not refractory, emit spike and reset
        4. **Refractory update**: Decrement refractory countdown or clamp V_m to V_reset
        5. **Buffer stimulus**: Store current input in ``I_stim`` for next step (one-step delay)

        **NMDA port constraints:**

        - New ports can only be added before first :meth:`update` call
        - Port weights are fixed at first registration and cannot change
        - Attempting to violate these constraints raises ``ValueError``

        **Integration details:**

        - Uses adaptive RKF45 with per-neuron step size (not vectorized)
        - Local error tolerance controlled by ``gsl_error_tol``
        - Minimum step size: 1e-8 ms; maximum iterations: 10,000
        - Step size persists across time steps in ``integration_step`` state

        **Refractory behavior:**

        - During refractory period, V_m is clamped to V_reset
        - Refractory countdown decrements each time step
        - Threshold check bypassed while refractory

        Examples
        --------
        **Single neuron with AMPA spike:**

        .. code-block:: python

            >>> import brainpy.state as bp
            >>> import brainunit as u
            >>> import brainstate
            >>> brainstate.environ.context(dt=0.1 * u.ms)
            >>> net = bp.iaf_bw_2001_exact(in_size=1)
            >>> net.init_all_states()
            >>> spike = net(spike_events=[(1, 500*u.nS)])
            >>> print(net.s_AMPA.value)  # doctest: +SKIP

        **Population with mixed input:**

        .. code-block:: python

            >>> import brainpy.state as bp
            >>> import brainunit as u
            >>> import brainstate
            >>> import jax.numpy as jnp
            >>> brainstate.environ.context(dt=0.1 * u.ms)
            >>> net = bp.iaf_bw_2001_exact(in_size=10)
            >>> net.init_all_states()
            >>> # AMPA to all, GABA to subset
            >>> events = [
            ...     (1, jnp.ones(10) * 100*u.nS),  # AMPA
            ...     (2, jnp.array([0,0,0,0,0,50,50,50,50,50])*u.nS),  # GABA
            ... ]
            >>> spike = net(spike_events=events)
            >>> print(spike)  # doctest: +SKIP

        **NMDA port registration:**

        .. code-block:: python

            >>> import brainpy.state as bp
            >>> import brainunit as u
            >>> import brainstate
            >>> brainstate.environ.context(dt=0.1 * u.ms)
            >>> net = bp.iaf_bw_2001_exact(in_size=2)
            >>> net.init_all_states()
            >>> # First update: register ports 'A' and 'B'
            >>> events = [
            ...     (3, 60*u.nS, 'A', 1.0),
            ...     (3, 80*u.nS, 'B', 2.0),
            ... ]
            >>> spike = net(spike_events=events)
            >>> print(net.nmda_weights.value.shape)  # doctest: +SKIP
            (2, 2)  # 2 neurons × 2 ports
            >>> # Subsequent updates: ports A and B exist, weights fixed
            >>> spike = net(spike_events=[(3, 60*u.nS, 'A', 3.0)])  # OK
            >>> spike = net(spike_events=[(3, 99*u.nS, 'A', 1.0)])  # Raises ValueError
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        state_shape = self.V.value.shape

        V = self._broadcast_to_state(self._value_to_float(self.V.value, u.mV), state_shape)
        s_ampa = self._broadcast_to_state(self._value_to_float(self.s_AMPA.value, u.nS), state_shape)
        s_gaba = self._broadcast_to_state(self._value_to_float(self.s_GABA.value, u.nS), state_shape)

        i_ampa_prev = self._broadcast_to_state(self._value_to_float(self.I_AMPA.value, u.pA), state_shape)
        i_gaba_prev = self._broadcast_to_state(self._value_to_float(self.I_GABA.value, u.pA), state_shape)
        i_nmda_prev = self._broadcast_to_state(self._value_to_float(self.I_NMDA.value, u.pA), state_shape)

        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), state_shape
        )
        i_stim = self._broadcast_to_state(self._value_to_float(self.I_stim.value, u.pA), state_shape)
        h_int = self._broadcast_to_state(self._value_to_float(self.integration_step.value, u.ms), state_shape)

        p = {
            'E_L': self._broadcast_to_state(self._value_to_float(self.E_L, u.mV), state_shape),
            'E_ex': self._broadcast_to_state(self._value_to_float(self.E_ex, u.mV), state_shape),
            'E_in': self._broadcast_to_state(self._value_to_float(self.E_in, u.mV), state_shape),
            'V_th': self._broadcast_to_state(self._value_to_float(self.V_th, u.mV), state_shape),
            'V_reset': self._broadcast_to_state(self._value_to_float(self.V_reset, u.mV), state_shape),
            'C_m': self._broadcast_to_state(self._value_to_float(self.C_m, u.pF), state_shape),
            'g_L': self._broadcast_to_state(self._value_to_float(self.g_L, u.nS), state_shape),
            'tau_AMPA': self._broadcast_to_state(self._value_to_float(self.tau_AMPA, u.ms), state_shape),
            'tau_GABA': self._broadcast_to_state(self._value_to_float(self.tau_GABA, u.ms), state_shape),
            'tau_rise_NMDA': self._broadcast_to_state(self._value_to_float(self.tau_rise_NMDA, u.ms), state_shape),
            'tau_decay_NMDA': self._broadcast_to_state(self._value_to_float(self.tau_decay_NMDA, u.ms), state_shape),
            'alpha': self._broadcast_to_state(self._value_to_float(self.alpha, 1 / u.ms), state_shape),
            'conc_Mg2': self._broadcast_to_state(self._value_to_float(self.conc_Mg2, u.mM), state_shape),
            'gsl_error_tol': self._broadcast_to_state(self._value_to_float(self.gsl_error_tol, None), state_shape),
        }

        ds_ampa_ev, ds_gaba_ev, dx_nmda_ev = self._parse_spike_events(spike_events, state_shape)
        ds_ampa_reg, ds_gaba_reg = self._parse_registered_spike_inputs(state_shape)
        ds_ampa = ds_ampa_ev + ds_ampa_reg
        ds_gaba = ds_gaba_ev + ds_gaba_reg

        x_nmda = np.asarray(self.x_NMDA.value, dtype=np.float64)
        s_nmda_components = np.asarray(self.s_NMDA_components.value, dtype=np.float64)
        nmda_weights = np.asarray(self.nmda_weights.value, dtype=np.float64)
        n_nmda = int(x_nmda.shape[-1])

        if dx_nmda_ev.shape[-1] != n_nmda:
            if dx_nmda_ev.shape[-1] < n_nmda:
                pad = np.zeros(state_shape + (n_nmda - dx_nmda_ev.shape[-1],), dtype=np.float64)
                dx_nmda_ev = np.concatenate([dx_nmda_ev, pad], axis=-1)
            else:
                dx_nmda_ev = dx_nmda_ev[..., :n_nmda]

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._value_to_float(new_i_stim_q, u.pA), state_shape)

        v_for_spike = np.empty_like(V)
        V_next = np.empty_like(V)
        s_ampa_next = np.empty_like(s_ampa)
        s_gaba_next = np.empty_like(s_gaba)
        x_nmda_next = np.empty_like(x_nmda)
        s_nmda_next = np.empty_like(s_nmda_components)
        s_nmda_sum_next = np.zeros_like(V)

        i_ampa_next = np.empty_like(i_ampa_prev)
        i_gaba_next = np.empty_like(i_gaba_prev)
        i_nmda_next = np.empty_like(i_nmda_prev)

        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)
        last_spike = self._broadcast_to_state(self._value_to_float(self.last_spike_time.value, u.ms), state_shape)
        last_spike_next = np.empty_like(last_spike)
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            state_shape,
        )

        for idx in np.ndindex(state_shape):
            local_p = {k: p[k][idx] for k in p}
            w_nmda_i = np.asarray(nmda_weights[idx], dtype=np.float64)
            x_i = np.asarray(x_nmda[idx], dtype=np.float64)
            s_i = np.asarray(s_nmda_components[idx], dtype=np.float64)
            y0 = np.concatenate([
                np.asarray([V[idx], s_ampa[idx], s_gaba[idx]], dtype=np.float64),
                x_i,
                s_i,
            ])
            y_i, h_i, ia_i, ig_i, in_i, s_nmda_sum_i = self._rkf45_integrate_scalar(
                y0=y0,
                i_stim=i_stim[idx],
                h0=h_int[idx],
                dt=dt,
                p=local_p,
                nmda_weights=w_nmda_i,
                atol=local_p['gsl_error_tol'],
            )

            sa_i = y_i[1] + ds_ampa[idx]
            sg_i = y_i[2] + ds_gaba[idx]
            if n_nmda > 0:
                x_i = y_i[3:3 + n_nmda] + dx_nmda_ev[idx]
                s_i = y_i[3 + n_nmda:]
            else:
                x_i = np.zeros((0,), dtype=np.float64)
                s_i = np.zeros((0,), dtype=np.float64)

            v_i = y_i[0]
            if r[idx] > 0:
                v_for_spike[idx] = local_p['V_reset']
                v_i = local_p['V_reset']
                r_i = r[idx] - 1
                t_last_i = last_spike[idx]
            else:
                v_for_spike[idx] = v_i
                if v_i >= local_p['V_th']:
                    v_i = local_p['V_reset']
                    r_i = int(refr_counts[idx])
                    t_last_i = float(u.math.asarray((t + dt_q) / u.ms))
                else:
                    r_i = 0
                    t_last_i = last_spike[idx]

            V_next[idx] = v_i
            s_ampa_next[idx] = sa_i
            s_gaba_next[idx] = sg_i
            if n_nmda > 0:
                x_nmda_next[idx] = x_i
                s_nmda_next[idx] = s_i
            s_nmda_sum_next[idx] = s_nmda_sum_i

            i_ampa_next[idx] = ia_i
            i_gaba_next[idx] = ig_i
            i_nmda_next[idx] = in_i

            r_next[idx] = r_i
            h_next[idx] = h_i
            last_spike_next[idx] = t_last_i

        self.V.value = V_next * u.mV
        self.s_AMPA.value = s_ampa_next * u.nS
        self.s_GABA.value = s_gaba_next * u.nS
        self.s_NMDA.value = s_nmda_sum_next * u.nS

        self.x_NMDA.value = x_nmda_next
        self.s_NMDA_components.value = s_nmda_next

        self.I_AMPA.value = i_ampa_next * u.pA
        self.I_GABA.value = i_gaba_next * u.pA
        self.I_NMDA.value = i_nmda_next * u.pA

        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_next * u.ms)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        self._updates_started = True
        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float64) * u.mV)
