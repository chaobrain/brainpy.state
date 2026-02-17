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
from typing import Callable, Iterable

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'iaf_bw_2001',
]


class iaf_bw_2001(Neuron):
    r"""NEST-compatible ``iaf_bw_2001`` neuron model.

    Conductance-based leaky integrate-and-fire neuron with AMPA, GABA, and
    approximate NMDA synaptic dynamics from Brunel-Wang style cortical models.

    This model implements the NEST ``iaf_bw_2001`` neuron with full compatibility,
    including adaptive RKF45 integration of subthreshold ODEs, receptor-routed
    AMPA/GABA/NMDA spike inputs, one-step delayed external current buffering,
    refractory countdown and reset ordering, and NMDA presynaptic jump approximation
    using spike-event offsets.

    **1. Mathematical Model**

    The continuous-time state vector is:

    .. math::

       y = (V_m, s_{AMPA}, s_{GABA}, s_{NMDA}).

    **Membrane dynamics:**

    .. math::

       C_m \frac{dV_m}{dt} = -g_L(V_m - E_L) - I_{syn} + I_{stim},

    where the total synaptic current is:

    .. math::

       I_{syn} = I_{AMPA} + I_{GABA} + I_{NMDA}.

    **Synaptic currents:**

    AMPA and GABA currents use Ohmic conductance:

    .. math::

       I_{AMPA} = (V_m - E_{ex}) s_{AMPA},
       \quad
       I_{GABA} = (V_m - E_{in}) s_{GABA}.

    NMDA current includes voltage-dependent Mg²⁺ block:

    .. math::

       I_{NMDA} = \frac{(V_m - E_{ex}) s_{NMDA}}
       {1 + [Mg^{2+}]\exp(-0.062 V_m)/3.57}.

    **Synaptic kinetics:**

    All three receptor types decay exponentially:

    .. math::

       \frac{ds_{AMPA}}{dt} = -\frac{s_{AMPA}}{\tau_{AMPA}},
       \quad
       \frac{ds_{GABA}}{dt} = -\frac{s_{GABA}}{\tau_{GABA}},
       \quad
       \frac{ds_{NMDA}}{dt} = -\frac{s_{NMDA}}{\tau_{NMDA,decay}}.

    **2. NMDA Approximation and Spike Offsets**

    NMDA recurrent coupling uses a presynaptic auxiliary variable ``s_NMDA_pre``
    updated only when this neuron spikes. At spike time :math:`t_{spike}`:

    .. math::

       s_{pre} \leftarrow s_{pre}
       \exp\left(-\frac{t_{spike} - t_{last}}{\tau_{NMDA,decay}}\right),

    .. math::

       \Delta s_{NMDA} = k_0 + k_1 s_{pre},
       \quad
       s_{pre} \leftarrow s_{pre} + \Delta s_{NMDA},

    where the jump constants are:

    .. math::

       k_1 = \exp(-\alpha\tau_{NMDA,rise}) - 1,

    .. math::

       k_0 = (\alpha\tau_{NMDA,rise})^{\tau_{NMDA,rise}/\tau_{NMDA,decay}}
       \gamma\Big(1 - \tau_{NMDA,rise}/\tau_{NMDA,decay},
       \alpha\tau_{NMDA,rise}\Big),

    where :math:`\gamma` is the lower incomplete gamma function. The per-spike
    :math:`\Delta s_{NMDA}` is exposed as ``spike_offset`` and used by NMDA
    receptor events as ``weight * spike_offset`` (matching NEST ``SpikeEvent``
    semantics for ``iaf_bw_2001``).

    **3. Update Order (NEST Semantics)**

    Per simulation step:

    1. **Integration**: Integrate ODEs on :math:`(t, t+dt]` using adaptive
       Runge-Kutta-Fehlberg 4(5) with persistent internal step size.
    2. **Spike reception**: Add arriving AMPA/GABA/NMDA spike increments to
       ``s_AMPA``, ``s_GABA``, ``s_NMDA``.
    3. **Threshold/reset**: Apply refractory countdown or check threshold,
       emit spike, and reset if :math:`V_m \geq V_{th}`.
    4. **Current buffering**: Store external current into delayed buffer
       ``I_stim`` for next step (one-step ring-buffer delay).

    Ordering notes:

    - Refractory clamping is applied after integration (as in NEST source).
    - ``I_stim`` uses one-step delay to match NEST's ring-buffer semantics.
    - During refractory period, :math:`V_m` is clamped to :math:`V_{reset}`.

    **4. Receptor Types and Event Semantics**

    Receptor types (matching NEST names and IDs):

    - ``AMPA`` = 1 (excitatory, fast)
    - ``GABA`` = 2 (inhibitory)
    - ``NMDA`` = 3 (excitatory, slow, voltage-dependent)

    The ``spike_events`` parameter passed to :meth:`update` may contain tuples
    or dictionaries:

    - Tuple format: ``(receptor, weight)`` or ``(receptor, weight, offset)``
      or ``(receptor, weight, offset, sender_model)``
    - Dict format: ``{'receptor_type': ..., 'weight': ..., 'offset': ...,
      'sender_model': ...}``

    For NMDA events, ``sender_model`` must be ``'iaf_bw_2001'``; otherwise a
    ``ValueError`` is raised (mirroring NEST's illegal-connection check, as
    only ``iaf_bw_2001`` neurons compute the NMDA spike offset).

    Registered ``add_delta_input`` entries can be receptor-labeled using
    ``label='AMPA'``, ``label='GABA'``, or ``label='NMDA'``. Unlabeled delta
    inputs default to AMPA.

    Parameters
    ----------
    in_size : int, tuple of int
        Population shape (number of neurons). Can be an integer or tuple for
        multi-dimensional populations.
    E_L : brainunit.Quantity, optional
        Leak reversal potential. Default: -70 mV.
    E_ex : brainunit.Quantity, optional
        Excitatory reversal potential (AMPA, NMDA). Default: 0 mV.
    E_in : brainunit.Quantity, optional
        Inhibitory reversal potential (GABA). Default: -70 mV.
    V_th : brainunit.Quantity, optional
        Spike threshold potential. Default: -55 mV.
    V_reset : brainunit.Quantity, optional
        Reset potential after spike. Must be strictly less than ``V_th``.
        Default: -60 mV.
    C_m : brainunit.Quantity, optional
        Membrane capacitance. Must be strictly positive. Default: 500 pF.
    g_L : brainunit.Quantity, optional
        Leak conductance. Default: 25 nS.
    t_ref : brainunit.Quantity, optional
        Absolute refractory period duration. Must be non-negative. Default: 2 ms.
    tau_AMPA : brainunit.Quantity, optional
        AMPA receptor decay time constant. Must be strictly positive. Default: 2 ms.
    tau_GABA : brainunit.Quantity, optional
        GABA receptor decay time constant. Must be strictly positive. Default: 5 ms.
    tau_decay_NMDA : brainunit.Quantity, optional
        NMDA receptor slow decay time constant. Must be strictly positive.
        Default: 100 ms.
    tau_rise_NMDA : brainunit.Quantity, optional
        NMDA receptor fast rise time constant for jump approximation. Must be
        strictly positive. Default: 2 ms.
    alpha : brainunit.Quantity, optional
        NMDA jump-shape parameter (rate constant). Must be strictly positive.
        Default: 0.5 / ms.
    conc_Mg2 : brainunit.Quantity, optional
        Extracellular magnesium concentration for NMDA voltage-dependent block.
        Must be strictly positive. Default: 1 mM.
    gsl_error_tol : float, optional
        RKF45 local error tolerance (analog to NEST's ``gsl_error_tol``).
        Smaller values increase integration accuracy but decrease performance.
        Must be strictly positive. Default: 1e-3.
    V_initializer : callable, optional
        Membrane potential initializer function. Default: Constant(-70 mV).
    s_AMPA_initializer : callable, optional
        AMPA conductance state initializer. Default: Constant(0 nS).
    s_GABA_initializer : callable, optional
        GABA conductance state initializer. Default: Constant(0 nS).
    s_NMDA_initializer : callable, optional
        NMDA conductance state initializer. Default: Constant(0 nS).
    spk_fun : callable, optional
        Surrogate gradient function for spike generation. Default: ReluGrad().
    spk_reset : str, optional
        Spike reset mode. ``'hard'`` (stop gradient) matches NEST behavior;
        ``'soft'`` (subtract threshold) is differentiable. Default: 'hard'.
    ref_var : bool, optional
        If True, expose boolean ``refractory`` state variable. Default: False.
    name : str, optional
        Name of the neuron group.

    Parameter Mapping
    -----------------

    The following table maps brainpy.state parameter names to their NEST equivalents:

    ==================== =================== ===========================================================
    **brainpy.state**    **NEST**            **Description**
    ==================== =================== ===========================================================
    ``E_L``              ``E_L``             Leak reversal potential
    ``E_ex``             ``E_ex``            Excitatory reversal potential
    ``E_in``             ``E_in``            Inhibitory reversal potential
    ``V_th``             ``V_th``            Spike threshold
    ``V_reset``          ``V_reset``         Reset potential
    ``C_m``              ``C_m``             Membrane capacitance
    ``g_L``              ``g_L``             Leak conductance
    ``t_ref``            ``t_ref``           Refractory period
    ``tau_AMPA``         ``tau_AMPA``        AMPA decay time constant
    ``tau_GABA``         ``tau_GABA``        GABA decay time constant
    ``tau_decay_NMDA``   ``tau_decay_NMDA``  NMDA slow decay time constant
    ``tau_rise_NMDA``    ``tau_rise_NMDA``   NMDA fast rise time constant
    ``alpha``            ``alpha``           NMDA jump-shape parameter
    ``conc_Mg2``         ``conc_Mg2``        Extracellular Mg²⁺ concentration
    ``gsl_error_tol``    ``gsl_error_tol``   RKF45 error tolerance
    ==================== =================== ===========================================================

    Recordables
    -----------

    The following state variables can be recorded during simulation:

    - ``V_m`` : membrane potential (mV)
    - ``s_AMPA`` : AMPA conductance state (nS)
    - ``s_GABA`` : GABA conductance state (nS)
    - ``s_NMDA`` : NMDA conductance state (nS)
    - ``I_AMPA`` : AMPA synaptic current (pA)
    - ``I_GABA`` : GABA synaptic current (pA)
    - ``I_NMDA`` : NMDA synaptic current (pA)

    Additional State Variables
    --------------------------

    The following internal state variables are maintained but typically not recorded:

    - ``s_NMDA_pre`` : presynaptic NMDA helper state (unitless)
    - ``spike_offset`` : per-step NMDA offset emitted on spike (unitless)
    - ``refractory_step_count`` : absolute refractory countdown (int)
    - ``integration_step`` : persistent adaptive RKF45 step size (ms)
    - ``I_stim`` : one-step delayed external current buffer (pA)
    - ``last_spike_time`` : time of last spike (ms)
    - ``refractory`` : boolean refractory indicator (only if ``ref_var=True``)

    Raises
    ------
    ValueError
        If ``V_reset >= V_th`` (reset must be below threshold).
    ValueError
        If ``C_m <= 0`` (capacitance must be positive).
    ValueError
        If ``t_ref < 0`` (refractory period cannot be negative).
    ValueError
        If any time constant (``tau_AMPA``, ``tau_GABA``, ``tau_decay_NMDA``,
        ``tau_rise_NMDA``) is non-positive.
    ValueError
        If ``alpha <= 0`` (NMDA shape parameter must be positive).
    ValueError
        If ``conc_Mg2 <= 0`` (Mg²⁺ concentration must be positive).
    ValueError
        If ``gsl_error_tol <= 0`` (error tolerance must be positive).
    ValueError
        If NMDA spike event has ``sender_model != 'iaf_bw_2001'`` (only
        ``iaf_bw_2001`` neurons can compute NMDA spike offsets).

    Examples
    --------
    Create a simple network with AMPA and NMDA recurrent connections:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
        >>> import brainstate
        >>>
        >>> # Create neuron population
        >>> neurons = bst.iaf_bw_2001(100, V_th=-50*u.mV, t_ref=3*u.ms)
        >>>
        >>> # Initialize states
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     neurons.init_all_states()
        >>>
        >>> # Simulate with external input
        >>> with brainstate.environ.context(dt=0.1*u.ms, t=0*u.ms):
        ...     spike = neurons(x=500*u.pA)  # External current input

    Simulate with explicit spike events (receptor-routed):

    .. code-block:: python

        >>> # AMPA spike event (tuple format)
        >>> ampa_event = ('AMPA', 1.0*u.nS)
        >>> spike = neurons(spike_events=[ampa_event])
        >>>
        >>> # NMDA spike event (dict format with offset)
        >>> nmda_event = {
        ...     'receptor_type': 'NMDA',
        ...     'weight': 0.5*u.nS,
        ...     'offset': 0.8,  # Presynaptic NMDA offset from sender
        ...     'sender_model': 'iaf_bw_2001'
        ... }
        >>> spike = neurons(spike_events=[nmda_event])

    Notes
    -----
    - **Integration method**: This model uses adaptive Runge-Kutta-Fehlberg 4(5)
      (RKF45) with local error control, matching NEST's GSL integration. The
      internal step size ``integration_step`` is persistent and adapted per neuron.
    - **NMDA offset computation**: Only ``iaf_bw_2001`` neurons compute the NMDA
      spike offset. If connecting other neuron types, NMDA connections will raise
      a ``ValueError``. Use AMPA for inter-model connectivity.
    - **Surrogate gradients**: Unlike NEST (which is not differentiable), this
      implementation supports gradient-based learning via surrogate spike functions.
    - **Performance**: RKF45 integration is accurate but slow for large populations.
      For performance-critical applications, consider using fixed-step models
      (e.g., ``iaf_cond_exp``, ``iaf_psc_alpha``) when NMDA dynamics are not required.
    - **Refractory semantics**: During refractory period, :math:`V_m` is clamped to
      :math:`V_{reset}`, and threshold crossing is disabled. This matches NEST behavior.

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
    .. [4] NEST source: ``models/iaf_bw_2001.h`` and ``models/iaf_bw_2001.cpp``.

    See Also
    --------
    iaf_cond_exp : Simpler conductance-based LIF without NMDA dynamics.
    iaf_psc_alpha : Current-based LIF with alpha-function PSCs.
    iaf_bw_2001_exact : Exact integration variant (if available).
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
        tau_decay_NMDA: ArrayLike = 100. * u.ms,
        tau_rise_NMDA: ArrayLike = 2. * u.ms,
        alpha: ArrayLike = 0.5 / u.ms,
        conc_Mg2: ArrayLike = 1.0 * u.mM,
        gsl_error_tol: ArrayLike = 1e-3,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        s_AMPA_initializer: Callable = braintools.init.Constant(0. * u.nS),
        s_GABA_initializer: Callable = braintools.init.Constant(0. * u.nS),
        s_NMDA_initializer: Callable = braintools.init.Constant(0. * u.nS),
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
        self.tau_decay_NMDA = braintools.init.param(tau_decay_NMDA, self.varshape)
        self.tau_rise_NMDA = braintools.init.param(tau_rise_NMDA, self.varshape)
        self.alpha = braintools.init.param(alpha, self.varshape)
        self.conc_Mg2 = braintools.init.param(conc_Mg2, self.varshape)
        self.gsl_error_tol = braintools.init.param(gsl_error_tol, self.varshape)

        self.V_initializer = V_initializer
        self.s_AMPA_initializer = s_AMPA_initializer
        self.s_GABA_initializer = s_GABA_initializer
        self.s_NMDA_initializer = s_NMDA_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @property
    def receptor_types(self):
        """Return dictionary of available receptor types.

        Returns
        -------
        dict
            Mapping from receptor name (str) to receptor ID (int).
            Keys: ``'AMPA'``, ``'GABA'``, ``'NMDA'``. Values: 1, 2, 3.
        """
        return dict(self.RECEPTOR_TYPES)

    @property
    def recordables(self):
        """Return list of recordable state variable names.

        Returns
        -------
        list of str
            State variables that can be recorded during simulation:
            ``['V_m', 's_AMPA', 's_GABA', 's_NMDA', 'I_NMDA', 'I_AMPA', 'I_GABA']``.
        """
        return list(self.RECORDABLES)

    @staticmethod
    def _value_to_float(x, unit=None):
        if unit is None:
            return np.asarray(u.math.asarray(x), dtype=np.float64)
        try:
            return np.asarray(u.math.asarray(x / unit), dtype=np.float64)
        except Exception:
            return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @classmethod
    def _normalize_spike_receptor(cls, receptor):
        if isinstance(receptor, str):
            key = receptor.strip()
            if key in cls.RECEPTOR_TYPES:
                return cls.RECEPTOR_TYPES[key]
            if key.isdigit():
                receptor = int(key)
            else:
                raise ValueError(f'Unknown receptor label: {receptor}')

        receptor = int(receptor)
        if receptor < 1 or receptor > 3:
            raise ValueError(f'Receptor type must be in [1, 3], got {receptor}.')
        return receptor

    def _validate_parameters(self):
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
        if np.any(self._value_to_float(self.tau_decay_NMDA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._value_to_float(self.tau_rise_NMDA, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')

        if np.any(self._value_to_float(self.alpha, 1 / u.ms) <= 0.0):
            raise ValueError('alpha > 0 required.')
        if np.any(self._value_to_float(self.conc_Mg2, u.mM) <= 0.0):
            raise ValueError('Mg2 concentration must be strictly positive.')
        if np.any(self._value_to_float(self.gsl_error_tol, None) <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize all state variables for the neuron population.

        Creates and initializes membrane potential, synaptic conductance states
        (AMPA, GABA, NMDA), synaptic currents, refractory counters, NMDA presynaptic
        helper state, adaptive RKF45 step size, and delayed current buffer.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for vectorized simulation. If None, no batch dimension.
        **kwargs
            Additional keyword arguments (unused, for compatibility).

        Notes
        -----
        - All synaptic conductances initialize to 0 nS by default.
        - Membrane potential initializes to -70 mV (near ``E_L``) by default.
        - ``integration_step`` initializes to the simulation timestep ``dt``.
        - ``last_spike_time`` initializes to -1e7 ms (far in the past).
        - If ``ref_var=True``, a boolean ``refractory`` state is also created.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        s_ampa = braintools.init.param(self.s_AMPA_initializer, self.varshape, batch_size)
        s_gaba = braintools.init.param(self.s_GABA_initializer, self.varshape, batch_size)
        s_nmda = braintools.init.param(self.s_NMDA_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.s_AMPA = brainstate.HiddenState(s_ampa)
        self.s_GABA = brainstate.HiddenState(s_gaba)
        self.s_NMDA = brainstate.HiddenState(s_nmda)

        zeros = np.zeros(self.V.value.shape, dtype=np.float64)
        self.I_NMDA = brainstate.ShortTermState(zeros * u.pA)
        self.I_AMPA = brainstate.ShortTermState(zeros * u.pA)
        self.I_GABA = brainstate.ShortTermState(zeros * u.pA)

        self.s_NMDA_pre = brainstate.ShortTermState(zeros.copy())
        self.spike_offset = brainstate.ShortTermState(zeros.copy())

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = self._safe_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0. * u.pA), self.varshape, batch_size)
        )

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        """Reset all state variables to initial values.

        Restores membrane potential, synaptic conductances, currents, refractory
        counters, NMDA helper state, adaptive step size, and delayed current buffer
        to their initial values. Does not reinitialize the state structure itself
        (use :meth:`init_state` for that).

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for vectorized simulation. If None, no batch dimension.
        **kwargs
            Additional keyword arguments (unused, for compatibility).

        Notes
        -----
        This method is typically called between simulation trials to reset the
        population to initial conditions without re-allocating state arrays.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.s_AMPA.value = braintools.init.param(self.s_AMPA_initializer, self.varshape, batch_size)
        self.s_GABA.value = braintools.init.param(self.s_GABA_initializer, self.varshape, batch_size)
        self.s_NMDA.value = braintools.init.param(self.s_NMDA_initializer, self.varshape, batch_size)

        zeros = np.zeros(self.V.value.shape, dtype=np.float64)
        self.I_NMDA.value = zeros * u.pA
        self.I_AMPA.value = zeros * u.pA
        self.I_GABA.value = zeros * u.pA

        self.s_NMDA_pre.value = zeros.copy()
        self.spike_offset.value = zeros.copy()

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
            braintools.init.Constant(0. * u.pA), self.varshape, batch_size
        )

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        """Compute spike output using surrogate gradient function.

        Converts membrane potential to a differentiable spike signal using the
        configured surrogate gradient function (``spk_fun``). The membrane potential
        is scaled relative to threshold and reset before applying the surrogate.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (mV). If None, uses current ``self.V.value``.
            Shape: ``(*in_size,)`` or ``(batch_size, *in_size)``.

        Returns
        -------
        jax.numpy.ndarray
            Spike signal (differentiable). Shape matches input ``V``.
            Values in [0, 1] for typical surrogate functions (e.g., sigmoid-based).
            Hard thresholding (Heaviside) gives binary {0, 1} values.

        Notes
        -----
        - Scaling factor: :math:`(V - V_{th}) / (V_{th} - V_{reset})`.
        - The surrogate function is differentiable during backpropagation but
          appears as a step function during forward pass (for gradient flow).
        - This method is called internally by :meth:`update` after integration
          and threshold checking.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _parse_spike_events(self, spike_events: Iterable, state_shape):
        s_ampa = np.zeros(state_shape, dtype=np.float64)
        s_gaba = np.zeros(state_shape, dtype=np.float64)
        s_nmda = np.zeros(state_shape, dtype=np.float64)

        if spike_events is None:
            return s_ampa, s_gaba, s_nmda

        for ev in spike_events:
            sender_model = 'iaf_bw_2001'
            offset = 1.0

            if isinstance(ev, dict):
                receptor = ev.get('receptor_type', ev.get('receptor', 'AMPA'))
                weight = ev.get('weight', 0.0 * u.nS)
                sender_model = ev.get('sender_model', 'iaf_bw_2001')
                offset = ev.get('offset', ev.get('nmda_offset', 1.0))
            else:
                if len(ev) == 2:
                    receptor, weight = ev
                elif len(ev) == 3:
                    receptor, weight, offset = ev
                elif len(ev) == 4:
                    receptor, weight, offset, sender_model = ev
                else:
                    raise ValueError('Spike event tuples must have length 2, 3, or 4.')

            receptor_id = self._normalize_spike_receptor(receptor)
            w_np = self._value_to_float(weight, u.nS)
            w_np = np.broadcast_to(w_np, state_shape)

            if receptor_id == self.AMPA:
                s_ampa += w_np
            elif receptor_id == self.GABA:
                s_gaba += w_np
            else:
                if sender_model != 'iaf_bw_2001':
                    raise ValueError(
                        'For NMDA synapses in iaf_bw_2001, pre-synaptic neuron must also be of type iaf_bw_2001.'
                    )
                off_np = self._value_to_float(offset, None)
                off_np = np.broadcast_to(off_np, state_shape)
                s_nmda += w_np * off_np

        return s_ampa, s_gaba, s_nmda

    def _parse_registered_spike_inputs(self, state_shape):
        s_ampa = np.zeros(state_shape, dtype=np.float64)
        s_gaba = np.zeros(state_shape, dtype=np.float64)
        s_nmda = np.zeros(state_shape, dtype=np.float64)

        if self.delta_inputs is None:
            return s_ampa, s_gaba, s_nmda

        for key in tuple(self.delta_inputs.keys()):
            val = self.delta_inputs[key]
            if callable(val):
                val = val()
            else:
                self.delta_inputs.pop(key)

            label = None
            if ' // ' in key:
                label, _ = key.split(' // ', maxsplit=1)
            receptor = self.AMPA if label is None else self._normalize_spike_receptor(label)

            val_np = self._value_to_float(val, u.nS)
            val_np = np.broadcast_to(val_np, state_shape)

            if receptor == self.AMPA:
                s_ampa += val_np
            elif receptor == self.GABA:
                s_gaba += val_np
            else:
                s_nmda += val_np

        return s_ampa, s_gaba, s_nmda

    @staticmethod
    def _nmda_currents_scalar(v, s_ampa, s_gaba, s_nmda, p):
        """Compute synaptic currents for a single neuron (scalar values).

        Calculates AMPA, GABA, and NMDA currents given membrane potential,
        conductance states, and parameters. NMDA includes voltage-dependent
        Mg²⁺ block.

        Parameters
        ----------
        v : float
            Membrane potential (mV).
        s_ampa : float
            AMPA conductance state (nS).
        s_gaba : float
            GABA conductance state (nS).
        s_nmda : float
            NMDA conductance state (nS).
        p : dict
            Parameter dictionary with keys: ``'E_ex'``, ``'E_in'``, ``'conc_Mg2'``.

        Returns
        -------
        i_ampa : float
            AMPA current (pA).
        i_gaba : float
            GABA current (pA).
        i_nmda : float
            NMDA current with Mg²⁺ block (pA).
        """
        i_ampa = (v - p['E_ex']) * s_ampa
        i_gaba = (v - p['E_in']) * s_gaba
        denom = 1.0 + p['conc_Mg2'] * math.exp(-0.062 * v) / 3.57
        i_nmda = (v - p['E_ex']) / denom * s_nmda
        return i_ampa, i_gaba, i_nmda

    @classmethod
    def _dynamics_scalar(cls, y, i_stim, p):
        """Compute ODE derivatives for a single neuron (scalar values).

        Evaluates the right-hand side of the ODE system:

        .. math::

           dy/dt = f(y, i_{stim}, p)

        where :math:`y = (V_m, s_{AMPA}, s_{GABA}, s_{NMDA})`.

        Parameters
        ----------
        y : array_like
            State vector [v, s_ampa, s_gaba, s_nmda] (length 4).
            Units: [mV, nS, nS, nS].
        i_stim : float
            External stimulus current (pA).
        p : dict
            Parameter dictionary with keys: ``'E_L'``, ``'E_ex'``, ``'E_in'``,
            ``'C_m'``, ``'g_L'``, ``'tau_AMPA'``, ``'tau_GABA'``,
            ``'tau_decay_NMDA'``, ``'conc_Mg2'``.

        Returns
        -------
        numpy.ndarray
            Derivative vector [dv/dt, ds_ampa/dt, ds_gaba/dt, ds_nmda/dt] (length 4).
            Units: [mV/ms, nS/ms, nS/ms, nS/ms].
        """
        v, s_ampa, s_gaba, s_nmda = y
        i_ampa, i_gaba, i_nmda = cls._nmda_currents_scalar(v, s_ampa, s_gaba, s_nmda, p)
        i_syn = i_ampa + i_gaba + i_nmda

        dv = (-p['g_L'] * (v - p['E_L']) - i_syn + i_stim) / p['C_m']
        ds_ampa = -s_ampa / p['tau_AMPA']
        ds_gaba = -s_gaba / p['tau_GABA']
        ds_nmda = -s_nmda / p['tau_decay_NMDA']
        return np.asarray([dv, ds_ampa, ds_gaba, ds_nmda], dtype=np.float64)

    def _rkf45_integrate_scalar(self, y0, i_stim, h0, dt, p, atol):
        """Integrate ODEs for one timestep using adaptive RKF45 (scalar implementation).

        Performs adaptive Runge-Kutta-Fehlberg 4(5) integration with local error
        control. The step size is adapted based on estimated truncation error,
        matching NEST's GSL RKF45 integrator behavior.

        Parameters
        ----------
        y0 : array_like
            Initial state vector [v, s_ampa, s_gaba, s_nmda] (length 4).
            Units: [mV, nS, nS, nS].
        i_stim : float
            External stimulus current (pA).
        h0 : float
            Initial internal step size (ms). This is adapted during integration.
        dt : float
            Target integration interval (ms). Must integrate from 0 to ``dt``.
        p : dict
            Parameter dictionary (see :meth:`_dynamics_scalar` for required keys).
        atol : float
            Absolute local error tolerance (unitless). Smaller values increase
            accuracy but decrease performance.

        Returns
        -------
        v_final : float
            Final membrane potential (mV).
        s_ampa_final : float
            Final AMPA conductance state (nS).
        s_gaba_final : float
            Final GABA conductance state (nS).
        s_nmda_final : float
            Final NMDA conductance state (nS).
        h_final : float
            Adapted internal step size for next integration (ms).
        i_ampa_final : float
            Final AMPA current (pA).
        i_gaba_final : float
            Final GABA current (pA).
        i_nmda_final : float
            Final NMDA current (pA).

        Notes
        -----
        - RKF45 uses embedded 4th and 5th order Runge-Kutta formulas to estimate
          local truncation error.
        - Step size adaptation: if error > ``atol``, step is rejected and retried
          with smaller step; if error <= ``atol``, step is accepted and step size
          increases for next step.
        - Step size bounds: minimum ``_MIN_H`` (1e-8 ms), maximum ``dt`` (simulation step).
        - Maximum iterations: ``_MAX_ITERS`` (10000). If exceeded, integration stops
          at current time (may not reach ``dt``).
        - Safety factors: 0.9 for error ratio, 0.2-5.0 for step size adjustment.
        """
        t = 0.0
        h = max(h0, self._MIN_H)
        y = np.asarray(y0, dtype=np.float64)
        iters = 0

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = max(self._MIN_H, min(h, dt - t))

            k1 = self._dynamics_scalar(y, i_stim, p)
            k2 = self._dynamics_scalar(y + h * (1.0 / 4.0) * k1, i_stim, p)
            k3 = self._dynamics_scalar(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0), i_stim, p)
            k4 = self._dynamics_scalar(
                y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
                i_stim,
                p,
            )
            k5 = self._dynamics_scalar(
                y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
                i_stim,
                p,
            )
            k6 = self._dynamics_scalar(
                y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0),
                i_stim,
                p,
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

        i_ampa, i_gaba, i_nmda = self._nmda_currents_scalar(y[0], y[1], y[2], y[3], p)
        return y[0], y[1], y[2], y[3], h, i_ampa, i_gaba, i_nmda

    @staticmethod
    def _nmda_jump_constants(alpha, tau_rise, tau_decay):
        """Compute NMDA spike offset jump constants k0 and k1.

        Calculates precomputed constants for NMDA spike offset approximation
        based on alpha-function rise dynamics. These constants are used to
        compute the NMDA conductance jump :math:`\Delta s_{NMDA} = k_0 + k_1 s_{pre}`
        when a neuron spikes.

        Parameters
        ----------
        alpha : float or numpy.ndarray
            NMDA jump-shape parameter (1/ms). Can be scalar or array.
        tau_rise : float or numpy.ndarray
            NMDA rise time constant (ms). Can be scalar or array.
        tau_decay : float or numpy.ndarray
            NMDA decay time constant (ms). Can be scalar or array.

        Returns
        -------
        k0 : float or numpy.ndarray
            Constant term for NMDA offset (unitless). Shape matches inputs.
        k1 : float or numpy.ndarray
            Linear term for NMDA offset (unitless). Shape matches inputs.

        Notes
        -----
        The constants are derived from the integral of the NMDA alpha-function
        kernel:

        .. math::

           k_1 = \exp(-\alpha\tau_{rise}) - 1,

        .. math::

           k_0 = (\alpha\tau_{rise})^{\tau_{rise}/\tau_{decay}}
           \gamma\Big(1 - \tau_{rise}/\tau_{decay}, \alpha\tau_{rise}\Big),

        where :math:`\gamma(a, x)` is the lower incomplete gamma function.

        These constants are precomputed once per update step and reused for all
        neurons that spike during that step.
        """
        alpha_tau = alpha * tau_rise
        tau_ratio = tau_rise / tau_decay
        k1 = np.expm1(-alpha_tau)

        a = 1.0 - tau_ratio
        x = alpha_tau
        a_j = jnp.asarray(a, dtype=jnp.float64)
        x_j = jnp.asarray(x, dtype=jnp.float64)
        lower_gamma = np.asarray(
            jsp.special.gammainc(a_j, x_j) * jnp.exp(jsp.special.gammaln(a_j)),
            dtype=np.float64,
        )
        k0 = np.power(alpha_tau, tau_ratio) * lower_gamma
        return k0, k1

    def update(self, x=0. * u.pA, spike_events=None):
        """Advance the neuron state by one simulation timestep.

        Performs a complete update cycle including: (1) RKF45 integration of
        ODEs, (2) reception of AMPA/GABA/NMDA spike events, (3) threshold
        detection and spike emission, (4) refractory period handling, (5) NMDA
        spike offset computation, and (6) delayed current buffering.

        Parameters
        ----------
        x : brainunit.Quantity, optional
            External input current (pA). Can be scalar or array matching population
            shape. This current is buffered and applied in the **next** timestep
            (one-step delay, matching NEST ring-buffer semantics). Default: 0 pA.
        spike_events : list of tuple or dict, optional
            Incoming spike events from presynaptic neurons. Each event can be:

            - Tuple: ``(receptor, weight)`` or ``(receptor, weight, offset)`` or
              ``(receptor, weight, offset, sender_model)``
            - Dict: ``{'receptor_type': ..., 'weight': ..., 'offset': ...,
              'sender_model': ...}``

            Receptor types: ``'AMPA'`` or ``1``, ``'GABA'`` or ``2``, ``'NMDA'``
            or ``3``. Weight units: nS (conductance). Offset (for NMDA only):
            presynaptic NMDA spike offset (unitless, default 1.0). Sender model
            (for NMDA only): must be ``'iaf_bw_2001'``.

            If None, no spike events are processed. Default: None.

        Returns
        -------
        jax.numpy.ndarray
            Spike output (differentiable). Shape: ``(*in_size,)`` or
            ``(batch_size, *in_size)``. Values in [0, 1] for typical surrogate
            functions.

        Raises
        ------
        ValueError
            If an NMDA spike event has ``sender_model != 'iaf_bw_2001'``. Only
            ``iaf_bw_2001`` neurons compute NMDA spike offsets; other neuron
            types cannot send NMDA spikes to this model.

        Notes
        -----
        **Update order (matching NEST):**

        1. **Integration**: Integrate ODEs using adaptive RKF45 from :math:`t`
           to :math:`t + dt`. The persistent ``integration_step`` is adapted
           per neuron based on local error.
        2. **Spike reception**: Add incoming spike weights (scaled by offset
           for NMDA) to ``s_AMPA``, ``s_GABA``, ``s_NMDA``.
        3. **Refractory/threshold**:
           - If in refractory period (``refractory_step_count > 0``): clamp
             :math:`V_m` to :math:`V_{reset}`, decrement counter.
           - Else: check threshold :math:`V_m \geq V_{th}`. If crossed, emit
             spike, reset :math:`V_m \leftarrow V_{reset}`, set refractory
             counter, compute NMDA spike offset.
        4. **Current buffering**: Store input current ``x`` (plus any registered
           current inputs) into ``I_stim`` buffer for **next** step.

        **NMDA spike offset computation:**

        When this neuron spikes, the NMDA spike offset :math:`\Delta s_{NMDA}`
        is computed using the presynaptic helper state ``s_NMDA_pre``:

        .. math::

           s_{pre} \leftarrow s_{pre} \exp(-\Delta t / \tau_{NMDA,decay}),

        .. math::

           \Delta s_{NMDA} = k_0 + k_1 s_{pre},

        where :math:`\Delta t = t_{spike} - t_{last}` and :math:`k_0, k_1` are
        precomputed constants. The updated ``s_NMDA_pre`` is stored for the
        next spike. The offset :math:`\Delta s_{NMDA}` is exposed as
        ``spike_offset`` and should be passed to downstream NMDA connections.

        **Current delay:**

        The external current ``x`` is stored in ``I_stim`` and applied in the
        **next** timestep. This one-step delay matches NEST's ring-buffer
        semantics. Current inputs registered via ``add_current_input`` are
        summed with ``x`` and delayed together.

        **Integration notes:**

        - RKF45 uses local error tolerance ``gsl_error_tol`` (default 1e-3).
        - The adaptive step size ``integration_step`` is persistent per neuron
          and typically stabilizes after a few milliseconds.
        - Maximum iterations: 10000 per timestep (prevents infinite loops).
        - Minimum step size: 1e-8 ms (prevents numerical instability).

        Examples
        --------
        Simple update with external current:

        .. code-block:: python

            >>> import brainpy.state as bst
            >>> import brainunit as u
            >>> import brainstate
            >>>
            >>> neurons = bst.iaf_bw_2001(10)
            >>> with brainstate.environ.context(dt=0.1*u.ms):
            ...     neurons.init_all_states()
            ...     spike = neurons.update(x=500*u.pA)  # Returns spike array

        Update with spike events (receptor-routed):

        .. code-block:: python

            >>> # Receive AMPA spike with 1 nS weight
            >>> ampa_event = ('AMPA', 1.0*u.nS)
            >>> spike = neurons.update(spike_events=[ampa_event])
            >>>
            >>> # Receive NMDA spike with offset from another iaf_bw_2001 neuron
            >>> nmda_event = {
            ...     'receptor_type': 'NMDA',
            ...     'weight': 0.5*u.nS,
            ...     'offset': neurons.spike_offset.value[0],  # From presynaptic neuron
            ...     'sender_model': 'iaf_bw_2001'
            ... }
            >>> spike = neurons.update(spike_events=[nmda_event])
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))
        t_ms = float(u.math.asarray(t / u.ms))

        state_shape = self.V.value.shape

        V = self._broadcast_to_state(self._value_to_float(self.V.value, u.mV), state_shape)
        s_ampa = self._broadcast_to_state(self._value_to_float(self.s_AMPA.value, u.nS), state_shape)
        s_gaba = self._broadcast_to_state(self._value_to_float(self.s_GABA.value, u.nS), state_shape)
        s_nmda = self._broadcast_to_state(self._value_to_float(self.s_NMDA.value, u.nS), state_shape)

        i_ampa_prev = self._broadcast_to_state(self._value_to_float(self.I_AMPA.value, u.pA), state_shape)
        i_gaba_prev = self._broadcast_to_state(self._value_to_float(self.I_GABA.value, u.pA), state_shape)
        i_nmda_prev = self._broadcast_to_state(self._value_to_float(self.I_NMDA.value, u.pA), state_shape)

        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            state_shape,
        )
        i_stim = self._broadcast_to_state(self._value_to_float(self.I_stim.value, u.pA), state_shape)
        h_int = self._broadcast_to_state(self._value_to_float(self.integration_step.value, u.ms), state_shape)

        s_nmda_pre = self._broadcast_to_state(np.asarray(self.s_NMDA_pre.value, dtype=np.float64), state_shape)
        last_spike = self._broadcast_to_state(self._value_to_float(self.last_spike_time.value, u.ms), state_shape)

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
            'tau_decay_NMDA': self._broadcast_to_state(self._value_to_float(self.tau_decay_NMDA, u.ms), state_shape),
            'tau_rise_NMDA': self._broadcast_to_state(self._value_to_float(self.tau_rise_NMDA, u.ms), state_shape),
            'alpha': self._broadcast_to_state(self._value_to_float(self.alpha, 1 / u.ms), state_shape),
            'conc_Mg2': self._broadcast_to_state(self._value_to_float(self.conc_Mg2, u.mM), state_shape),
            'gsl_error_tol': self._broadcast_to_state(self._value_to_float(self.gsl_error_tol, None), state_shape),
        }

        k0, k1 = self._nmda_jump_constants(p['alpha'], p['tau_rise_NMDA'], p['tau_decay_NMDA'])

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            state_shape,
        )

        ev_ampa, ev_gaba, ev_nmda = self._parse_spike_events(spike_events, state_shape)
        reg_ampa, reg_gaba, reg_nmda = self._parse_registered_spike_inputs(state_shape)
        ds_ampa = ev_ampa + reg_ampa
        ds_gaba = ev_gaba + reg_gaba
        ds_nmda = ev_nmda + reg_nmda

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._value_to_float(new_i_stim_q, u.pA), state_shape)

        v_for_spike = np.empty_like(V)
        spike_mask = np.zeros_like(V, dtype=bool)

        V_next = np.empty_like(V)
        s_ampa_next = np.empty_like(s_ampa)
        s_gaba_next = np.empty_like(s_gaba)
        s_nmda_next = np.empty_like(s_nmda)

        i_ampa_next = np.empty_like(i_ampa_prev)
        i_gaba_next = np.empty_like(i_gaba_prev)
        i_nmda_next = np.empty_like(i_nmda_prev)

        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        s_nmda_pre_next = np.empty_like(s_nmda_pre)
        last_spike_next = np.empty_like(last_spike)
        spike_offset_next = np.zeros_like(s_nmda_pre)

        for idx in np.ndindex(state_shape):
            local_p = {k: p[k][idx] for k in p}
            v_i, sa_i, sg_i, sn_i, h_i, ia_i, ig_i, in_i = self._rkf45_integrate_scalar(
                (V[idx], s_ampa[idx], s_gaba[idx], s_nmda[idx]),
                i_stim[idx],
                h_int[idx],
                dt,
                local_p,
                local_p['gsl_error_tol'],
            )

            sa_i += ds_ampa[idx]
            sg_i += ds_gaba[idx]
            sn_i += ds_nmda[idx]

            if r[idx] > 0:
                v_for_spike[idx] = local_p['V_reset']
                v_i = local_p['V_reset']
                r_i = r[idx] - 1

                s_pre_i = s_nmda_pre[idx]
                t_last_i = last_spike[idx]
                offset_i = 0.0
            else:
                v_for_spike[idx] = v_i
                if v_i >= local_p['V_th']:
                    spike_mask[idx] = True
                    v_i = local_p['V_reset']
                    r_i = refr_counts[idx]

                    t_spike = t_ms + dt
                    s_pre_i = s_nmda_pre[idx] * math.exp(-(t_spike - last_spike[idx]) / local_p['tau_decay_NMDA'])
                    offset_i = k0[idx] + k1[idx] * s_pre_i
                    s_pre_i = s_pre_i + offset_i
                    t_last_i = t_spike
                else:
                    r_i = 0
                    s_pre_i = s_nmda_pre[idx]
                    t_last_i = last_spike[idx]
                    offset_i = 0.0

            V_next[idx] = v_i
            s_ampa_next[idx] = sa_i
            s_gaba_next[idx] = sg_i
            s_nmda_next[idx] = sn_i

            i_ampa_next[idx] = ia_i
            i_gaba_next[idx] = ig_i
            i_nmda_next[idx] = in_i

            r_next[idx] = r_i
            h_next[idx] = h_i
            s_nmda_pre_next[idx] = s_pre_i
            last_spike_next[idx] = t_last_i
            spike_offset_next[idx] = offset_i

        self.V.value = V_next * u.mV
        self.s_AMPA.value = s_ampa_next * u.nS
        self.s_GABA.value = s_gaba_next * u.nS
        self.s_NMDA.value = s_nmda_next * u.nS

        self.I_AMPA.value = i_ampa_next * u.pA
        self.I_GABA.value = i_gaba_next * u.pA
        self.I_NMDA.value = i_nmda_next * u.pA

        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA

        self.s_NMDA_pre.value = s_nmda_pre_next
        self.spike_offset.value = spike_offset_next
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_next * u.ms)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float64) * u.mV)
