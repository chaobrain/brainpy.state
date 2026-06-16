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

from typing import Callable, Hashable, Iterable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from brainpy_state._nest_base._base import NESTNeuron
from brainpy_state._nest_base._utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

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
    ``conc_Mg2``                 ``conc_Mg2``             Mg2+ concentration (mM)
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

    NMDA current includes voltage-dependent Mg2+ blockade:

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

    - ``(receptor, weight)`` --- receptor in {1, 2, 3} or {'AMPA', 'GABA', 'NMDA'}
    - ``(receptor, weight, third)`` --- ``third`` is multiplicity for AMPA/GABA, port for NMDA
    - ``(receptor, weight, port, multiplicity)`` --- full NMDA specification

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

    - ``V_m`` --- Membrane potential (mV)
    - ``s_AMPA`` --- AMPA conductance state (nS)
    - ``s_GABA`` --- GABA conductance state (nS)
    - ``s_NMDA`` --- Weighted sum of NMDA gating variables (nS), :math:`\sum_j w_j s_j`
    - ``I_AMPA`` --- AMPA current (pA)
    - ``I_GABA`` --- GABA current (pA)
    - ``I_NMDA`` --- NMDA current (pA)

    Additional State Variables
    --------------------------

    - ``x_NMDA`` --- NMDA rise variables for each port (shape: ``[*in_size, n_ports]``)
    - ``s_NMDA_components`` --- NMDA gating variables for each port (shape: ``[*in_size, n_ports]``)
    - ``nmda_weights`` --- Fixed weights for each NMDA port (shape: ``[*in_size, n_ports]``)
    - ``refractory_step_count`` --- Remaining refractory steps (int32)
    - ``integration_step`` --- Persistent RKF45 step size (ms)
    - ``I_stim`` --- One-step delayed external current buffer (pA)
    - ``refractory`` --- Boolean refractory indicator (only if ``ref_var=True``)

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

        >>> from brainpy import state as bp
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

        >>> from brainpy import state as bp
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
        (5, 2)  # 5 neurons x 2 NMDA ports

    **Mixing AMPA, GABA, and NMDA:**

    .. code-block:: python

        >>> from brainpy import state as bp
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

        >>> from brainpy import state as bp
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
    _MIN_H = 1e-8 * u.ms  # ms
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
        self.gsl_error_tol = gsl_error_tol

        self.V_initializer = V_initializer
        self.s_AMPA_initializer = s_AMPA_initializer
        self.s_GABA_initializer = s_GABA_initializer
        self.ref_var = ref_var

        self._nmda_port_index = {}
        self._updates_started = False

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
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_reset, self.C_m, self.tau_AMPA)):
            return

        if cond_any(self.V_reset >= self.V_th):
            raise ValueError('Reset potential must be smaller than threshold.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if cond_any(self.tau_AMPA <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_GABA <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_rise_NMDA <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_decay_NMDA <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.alpha <= 0.0 / u.ms):
            raise ValueError('alpha > 0 required.')
        if cond_any(self.conc_Mg2 <= 0.0 * u.mM):
            raise ValueError('Mg2 concentration must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def _nmda_num_ports(self):
        if hasattr(self, 'x_NMDA'):
            return int(self.x_NMDA.value.shape[-1])
        return 0

    def init_state(self, **kwargs):
        r"""Initialize all state variables.

        Creates and initializes membrane potential, synaptic conductances, currents,
        NMDA port arrays (initially empty), refractory state, and integration step size.
        NMDA port registry is cleared.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        - NMDA port arrays (x_NMDA, s_NMDA_components, nmda_weights) start empty (shape: [..., 0])
        - Ports are allocated dynamically when first NMDA spike arrives
        - Clears the internal ``_nmda_port_index`` registry
        - Resets ``_updates_started`` flag to False
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        s_ampa = braintools.init.param(self.s_AMPA_initializer, self.varshape)
        s_gaba = braintools.init.param(self.s_GABA_initializer, self.varshape)

        self.V = brainstate.HiddenState(V)
        self.s_AMPA = brainstate.HiddenState(s_ampa)
        self.s_GABA = brainstate.HiddenState(s_gaba)

        zeros = u.math.zeros(self.varshape, dtype=dftype)

        self.s_NMDA = brainstate.ShortTermState(zeros * u.nS)
        self.I_NMDA = brainstate.ShortTermState(zeros * u.pA)
        self.I_AMPA = brainstate.ShortTermState(zeros * u.pA)
        self.I_GABA = brainstate.ShortTermState(zeros * u.pA)

        self.x_NMDA = brainstate.ShortTermState(np.zeros(self.varshape + (0,), dtype=dftype))
        self.s_NMDA_components = brainstate.ShortTermState(np.zeros(self.varshape + (0,), dtype=dftype))
        self.nmda_weights = brainstate.ShortTermState(np.zeros(self.varshape + (0,), dtype=dftype))

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        self._nmda_port_index = {}
        self._updates_started = False

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
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
        dftype = brainstate.environ.dftype()
        zeros = np.zeros(state_shape, dtype=dftype)
        self.s_NMDA.value = zeros * u.nS
        self.I_NMDA.value = zeros * u.pA
        self.I_AMPA.value = zeros * u.pA
        self.I_GABA.value = zeros * u.pA

        n_ports = self._nmda_num_ports()
        self.x_NMDA.value = np.zeros(state_shape + (n_ports,), dtype=dftype)
        self.s_NMDA_components.value = np.zeros(state_shape + (n_ports,), dtype=dftype)

        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        ditype = brainstate.environ.ditype()
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=ditype)

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

    def _ensure_nmda_port(self, port: Hashable, weight_np: np.ndarray, state_shape):
        dftype = brainstate.environ.dftype()
        if port in self._nmda_port_index:
            idx = self._nmda_port_index[port]
            current_weight = np.asarray(self.nmda_weights.value[..., idx], dtype=dftype)
            if cond_any(current_weight != weight_np):
                raise ValueError('iaf_bw_2001_exact requires constant weights per NMDA port.')
            return idx

        if self._updates_started:
            raise ValueError('NMDA ports can only be added before the first call to update().')

        idx = self._nmda_num_ports()
        self._nmda_port_index[port] = idx

        zero_channel = np.zeros(state_shape + (1,), dtype=dftype)
        x_old = np.asarray(self.x_NMDA.value, dtype=dftype)
        s_old = np.asarray(self.s_NMDA_components.value, dtype=dftype)
        w_old = np.asarray(self.nmda_weights.value, dtype=dftype)

        self.x_NMDA.value = np.concatenate([x_old, zero_channel], axis=-1)
        self.s_NMDA_components.value = np.concatenate([s_old, zero_channel], axis=-1)
        self.nmda_weights.value = np.concatenate([w_old, np.expand_dims(weight_np, axis=-1)], axis=-1)
        return idx

    def _parse_spike_events(self, spike_events: Iterable, state_shape):
        dftype = brainstate.environ.dftype()
        ds_ampa = np.zeros(state_shape, dtype=dftype)
        ds_gaba = np.zeros(state_shape, dtype=dftype)
        nmda_mult = np.zeros(state_shape + (self._nmda_num_ports(),), dtype=dftype)

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
                    pad = np.zeros(state_shape + (nmda_idx + 1 - nmda_mult.shape[-1],), dtype=dftype)
                    nmda_mult = np.concatenate([nmda_mult, pad], axis=-1)
                nmda_mult[..., nmda_idx] = nmda_mult[..., nmda_idx] + mult_np

        return ds_ampa, ds_gaba, nmda_mult

    def _parse_registered_spike_inputs(self, state_shape):
        dftype = brainstate.environ.dftype()
        ds_ampa = np.zeros(state_shape, dtype=dftype)
        ds_gaba = np.zeros(state_shape, dtype=dftype)
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
        dftype = brainstate.environ.dftype()
        if unit is None:
            return np.asarray(u.math.asarray(x), dtype=dftype)
        try:
            return np.asarray(u.math.asarray(x / unit), dtype=dftype)
        except Exception:
            return np.asarray(u.math.asarray(x), dtype=dftype)

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

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        The ODE is integrated freely without in-loop V clamping or spike reset.
        Spike detection and refractory clamping are applied post-integration
        in :meth:`update`, matching NEST's GSL-based integration semantics.

        Parameters
        ----------
        state : DotDict
            Keys: V, s_AMPA, s_GABA, x_NMDA, s_NMDA_components -- ODE state variables.
        extra : DotDict
            Keys: unstable, i_stim, nmda_weights -- auxiliary data.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        v_eff = state.V  # V evolves freely; no refractory clamping in ODE

        # Synaptic currents
        i_ampa = state.s_AMPA * (v_eff - self.E_ex)
        i_gaba = state.s_GABA * (v_eff - self.E_in)

        # NMDA current with Mg2+ blockade
        # nmda_weights shape: (*varshape, n_ports), s_NMDA_components shape: (*varshape, n_ports)
        s_nmda_sum = u.math.sum(extra.nmda_weights * state.s_NMDA_components, axis=-1)
        # Mg2+ voltage-dependent block: denom = 1 + [Mg2+] * exp(-0.062 * V_mV) / 3.57
        v_mV = v_eff / u.mV
        conc_Mg2_mM = self.conc_Mg2 / u.mM
        denom = 1.0 + conc_Mg2_mM * u.math.exp(-0.062 * v_mV) / 3.57
        i_nmda = (v_eff - self.E_ex) / denom * s_nmda_sum * u.nS

        i_syn = i_ampa + i_gaba + i_nmda

        dV = (-self.g_L * (v_eff - self.E_L) - i_syn + extra.i_stim) / self.C_m

        ds_AMPA = -state.s_AMPA / self.tau_AMPA
        ds_GABA = -state.s_GABA / self.tau_GABA

        # NMDA dynamics: dx_j/dt = -x_j / tau_rise_NMDA
        #                ds_j/dt = -s_j / tau_decay_NMDA + alpha * x_j * (1 - s_j)
        # Expand tau/alpha for broadcasting over port dimension
        tau_rise = u.math.expand_dims(self.tau_rise_NMDA, axis=-1)
        tau_decay = u.math.expand_dims(self.tau_decay_NMDA, axis=-1)
        alpha_exp = u.math.expand_dims(self.alpha, axis=-1)

        dx_NMDA = -state.x_NMDA / tau_rise
        ds_NMDA_components = -state.s_NMDA_components / tau_decay + alpha_exp * state.x_NMDA * (1.0 - state.s_NMDA_components)

        return DotDict(
            V=dV,
            s_AMPA=ds_AMPA,
            s_GABA=ds_GABA,
            x_NMDA=dx_NMDA,
            s_NMDA_components=ds_NMDA_components,
        )

    def _event_fn(self, state, extra, accept):
        """Track numerical instability only; no in-ODE spike/reset logic.

        Spike detection and refractory clamping are handled post-integration
        in :meth:`update` to match NEST's semantics (currents recorded from
        freely-evolved, pre-reset V).

        Parameters
        ----------
        state : DotDict
            Keys: V, s_AMPA, s_GABA, x_NMDA, s_NMDA_components.
        extra : DotDict
            Keys: unstable, i_stim, nmda_weights.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (state, new_extra) -- state is unchanged; extra has updated unstable flag.
        """
        unstable = extra.unstable | jnp.any(
            accept & (u.get_mantissa(state.V) < -1e3)
        )
        return state, DotDict({**extra, 'unstable': unstable})

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
        2. **Spike jumps**: Add to s_AMPA, s_GABA (weight x multiplicity), x_NMDA (multiplicity only)
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
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        state_shape = self.V.value.shape

        # Read state variables with their natural units.
        V = self.V.value  # mV
        s_AMPA = self.s_AMPA.value  # nS
        s_GABA = self.s_GABA.value  # nS
        x_NMDA = self.x_NMDA.value  # dimensionless
        s_NMDA_components = self.s_NMDA_components.value  # dimensionless
        nmda_weights_val = self.nmda_weights.value  # nS (dimensionless float)
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Parse spike events (AMPA/GABA weight deltas and NMDA multiplicities).
        ds_ampa_ev, ds_gaba_ev, dx_nmda_ev = self._parse_spike_events(spike_events, state_shape)
        ds_ampa_reg, ds_gaba_reg = self._parse_registered_spike_inputs(state_shape)
        ds_ampa = ds_ampa_ev + ds_ampa_reg
        ds_gaba = ds_gaba_ev + ds_gaba_reg

        n_nmda_pre = int(x_NMDA.shape[-1]) if len(x_NMDA.shape) > len(state_shape) else 0
        # Re-read n_nmda and weights after parsing; new ports may have been registered.
        n_nmda = int(self.x_NMDA.value.shape[-1]) if len(self.x_NMDA.value.shape) > len(state_shape) else 0
        nmda_weights_val = self.nmda_weights.value

        # If new ports were registered during parsing, expand pre-integration arrays with zeros.
        if n_nmda > n_nmda_pre:
            n_new = n_nmda - n_nmda_pre
            x_NMDA = np.concatenate(
                [np.asarray(x_NMDA, dtype=dftype), np.zeros(state_shape + (n_new,), dtype=dftype)], axis=-1
            )
            s_NMDA_components = np.concatenate(
                [np.asarray(s_NMDA_components, dtype=dftype), np.zeros(state_shape + (n_new,), dtype=dftype)], axis=-1
            )

        if n_nmda > 0 and dx_nmda_ev.shape[-1] != n_nmda:
            if dx_nmda_ev.shape[-1] < n_nmda:
                pad = np.zeros(state_shape + (n_nmda - dx_nmda_ev.shape[-1],), dtype=dftype)
                dx_nmda_ev = np.concatenate([dx_nmda_ev, pad], axis=-1)
            else:
                dx_nmda_ev = dx_nmda_ev[..., :n_nmda]

        # Adaptive RKF45 integration via generic integrator.
        # V evolves freely (no in-ODE refractory clamp or spike reset)
        # to match NEST's GSL integration semantics.
        ode_state = DotDict(
            V=V,
            s_AMPA=s_AMPA,
            s_GABA=s_GABA,
            x_NMDA=x_NMDA,
            s_NMDA_components=s_NMDA_components,
        )
        ode_extra = DotDict(
            unstable=jnp.array(False),
            i_stim=i_stim,
            nmda_weights=nmda_weights_val,
        )

        ode_state, h, ode_extra = self.integrator(state=ode_state, h=h, extra=ode_extra)
        V = ode_state.V        # freely-evolved post-ODE V (may exceed V_th)
        s_AMPA = ode_state.s_AMPA
        s_GABA = ode_state.s_GABA
        x_NMDA = ode_state.x_NMDA
        s_NMDA_components = ode_state.s_NMDA_components
        unstable = ode_extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in iaf_bw_2001_exact dynamics.'
        )

        # Compute NMDA weighted sum and synaptic currents for recording.
        # Use the freely-evolved post-ODE V (before any spike reset or refractory clamp),
        # matching NEST's recording semantics where currents are snapshotted pre-reset.
        if n_nmda > 0:
            s_nmda_sum = u.math.sum(nmda_weights_val * s_NMDA_components, axis=-1)
        else:
            s_nmda_sum = u.math.zeros(self.varshape, dtype=dftype)

        v_for_current = V  # pre-reset, freely-evolved V
        i_ampa = s_AMPA * (v_for_current - self.E_ex)
        i_gaba = s_GABA * (v_for_current - self.E_in)
        v_mV = v_for_current / u.mV
        conc_Mg2_mM = self.conc_Mg2 / u.mM
        denom = 1.0 + conc_Mg2_mM * u.math.exp(-0.062 * v_mV) / 3.57
        i_nmda = (v_for_current - self.E_ex) / denom * s_nmda_sum * u.nS

        # Apply synaptic spike inputs (applied after integration).
        s_AMPA = s_AMPA + ds_ampa * u.nS
        s_GABA = s_GABA + ds_gaba * u.nS
        if n_nmda > 0:
            x_NMDA = x_NMDA + dx_nmda_ev

        # Post-ODE spike detection and refractory handling (matches NEST ordering).
        # Refractory neurons: clamp V to V_reset, decrement counter.
        is_refractory = r > 0
        V = u.math.where(is_refractory, self.V_reset, V)
        r = u.math.where(is_refractory, r - 1, r)

        # Non-refractory neurons: check threshold, emit spike, reset V, enter refractoriness.
        spike_mask = (~is_refractory) & (V >= self.V_th)
        V = u.math.where(spike_mask, self.V_reset, V)
        r = u.math.where(spike_mask & (self.ref_count > 0), self.ref_count, r)

        # Write back state.
        self.V.value = V
        self.s_AMPA.value = s_AMPA
        self.s_GABA.value = s_GABA
        self.s_NMDA.value = s_nmda_sum * u.nS
        self.x_NMDA.value = x_NMDA
        self.s_NMDA_components.value = s_NMDA_components
        self.I_AMPA.value = i_ampa
        self.I_GABA.value = i_gaba
        self.I_NMDA.value = i_nmda
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        self._updates_started = True
        return u.math.asarray(spike_mask, dtype=dftype)
