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

from typing import Callable, Iterable

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'iaf_bw_2001',
]


class iaf_bw_2001(NESTNeuron):
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
    E_L : saiunit.Quantity, optional
        Leak reversal potential. Default: -70 mV.
    E_ex : saiunit.Quantity, optional
        Excitatory reversal potential (AMPA, NMDA). Default: 0 mV.
    E_in : saiunit.Quantity, optional
        Inhibitory reversal potential (GABA). Default: -70 mV.
    V_th : saiunit.Quantity, optional
        Spike threshold potential. Default: -55 mV.
    V_reset : saiunit.Quantity, optional
        Reset potential after spike. Must be strictly less than ``V_th``.
        Default: -60 mV.
    C_m : saiunit.Quantity, optional
        Membrane capacitance. Must be strictly positive. Default: 500 pF.
    g_L : saiunit.Quantity, optional
        Leak conductance. Default: 25 nS.
    t_ref : saiunit.Quantity, optional
        Absolute refractory period duration. Must be non-negative. Default: 2 ms.
    tau_AMPA : saiunit.Quantity, optional
        AMPA receptor decay time constant. Must be strictly positive. Default: 2 ms.
    tau_GABA : saiunit.Quantity, optional
        GABA receptor decay time constant. Must be strictly positive. Default: 5 ms.
    tau_decay_NMDA : saiunit.Quantity, optional
        NMDA receptor slow decay time constant. Must be strictly positive.
        Default: 100 ms.
    tau_rise_NMDA : saiunit.Quantity, optional
        NMDA receptor fast rise time constant for jump approximation. Must be
        strictly positive. Default: 2 ms.
    alpha : saiunit.Quantity, optional
        NMDA jump-shape parameter (rate constant). Must be strictly positive.
        Default: 0.5 / ms.
    conc_Mg2 : saiunit.Quantity, optional
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

        >>> from brainpy import state as bst
        >>> import saiunit as u
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
        self.gsl_error_tol = gsl_error_tol

        self.V_initializer = V_initializer
        self.s_AMPA_initializer = s_AMPA_initializer
        self.s_GABA_initializer = s_GABA_initializer
        self.s_NMDA_initializer = s_NMDA_initializer
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

        # Pre-compute NMDA jump constants once (parameters are constant throughout simulation).
        _alpha_np = np.asarray(u.get_mantissa(self.alpha * u.ms))
        _tau_rise_np = np.asarray(u.get_mantissa(self.tau_rise_NMDA / u.ms))
        _tau_decay_np = np.asarray(u.get_mantissa(self.tau_decay_NMDA / u.ms))
        _k0, _k1 = self._nmda_jump_constants(_alpha_np, _tau_rise_np, _tau_decay_np)
        self._k0_np = np.asarray(_k0, dtype=np.float64)
        self._k1_np = np.asarray(_k1, dtype=np.float64)

    @property
    def receptor_types(self):
        r"""Return dictionary of available receptor types.

        Returns
        -------
        dict
            Mapping from receptor name (str) to receptor ID (int).
            Keys: ``'AMPA'``, ``'GABA'``, ``'NMDA'``. Values: 1, 2, 3.
        """
        return dict(self.RECEPTOR_TYPES)

    @property
    def recordables(self):
        r"""Return list of recordable state variable names.

        Returns
        -------
        list of str
            State variables that can be recorded during simulation:
            ``['V_m', 's_AMPA', 's_GABA', 's_NMDA', 'I_NMDA', 'I_AMPA', 'I_GABA']``.
        """
        return list(self.RECORDABLES)

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
        if cond_any(self.tau_decay_NMDA <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_rise_NMDA <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')

        if cond_any(self.alpha <= 0.0 / u.ms):
            raise ValueError('alpha > 0 required.')
        if cond_any(self.conc_Mg2 <= 0.0 * u.mM):
            raise ValueError('Mg2 concentration must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Creates and initializes membrane potential, synaptic conductance states
        (AMPA, GABA, NMDA), synaptic currents, refractory counters, NMDA presynaptic
        helper state, adaptive RKF45 step size, and delayed current buffer.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        - All synaptic conductances initialize to 0 nS by default.
        - Membrane potential initializes to -70 mV (near ``E_L``) by default.
        - ``integration_step`` initializes to the simulation timestep ``dt``.
        - ``last_spike_time`` initializes to -1e7 ms (far in the past).
        - If ``ref_var=True``, a boolean ``refractory`` state is also created.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        s_ampa = braintools.init.param(self.s_AMPA_initializer, self.varshape)
        s_gaba = braintools.init.param(self.s_GABA_initializer, self.varshape)
        s_nmda = braintools.init.param(self.s_NMDA_initializer, self.varshape)

        self.V = brainstate.HiddenState(V)
        self.s_AMPA = brainstate.HiddenState(s_ampa)
        self.s_GABA = brainstate.HiddenState(s_gaba)
        self.s_NMDA = brainstate.HiddenState(s_nmda)

        self.I_NMDA = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=dftype) * u.pA)
        self.I_AMPA = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=dftype) * u.pA)
        self.I_GABA = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=dftype) * u.pA)

        self.s_NMDA_pre = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=dftype))
        self.spike_offset = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=dftype))

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute spike output using surrogate gradient function.

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

    @staticmethod
    def _nmda_jump_constants(alpha, tau_rise, tau_decay):
        r"""Compute NMDA spike offset jump constants k0 and k1.

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
        dftype = brainstate.environ.dftype()
        alpha_tau = alpha * tau_rise
        tau_ratio = tau_rise / tau_decay
        k1 = np.expm1(-alpha_tau)

        a = 1.0 - tau_ratio
        x = alpha_tau
        a_j = jnp.asarray(a, dtype=dftype)
        x_j = jnp.asarray(x, dtype=dftype)
        lower_gamma = np.asarray(
            jsp.special.gammainc(a_j, x_j) * jnp.exp(jsp.special.gammaln(a_j)),
            dtype=dftype,
        )
        k0 = np.power(alpha_tau, tau_ratio) * lower_gamma
        return k0, k1

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, s_AMPA, s_GABA, s_NMDA -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim -- mutable
            auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        # AMPA current: I_AMPA = (V - E_ex) * s_AMPA
        i_ampa = state.s_AMPA * (state.V - self.E_ex)
        # GABA current: I_GABA = (V - E_in) * s_GABA
        i_gaba = state.s_GABA * (state.V - self.E_in)
        # NMDA current with Mg2+ block
        v_mV = state.V / u.mV
        conc_mM = self.conc_Mg2 / u.mM
        denom = 1.0 + conc_mM * u.math.exp(-0.062 * v_mV) / 3.57
        i_nmda = state.s_NMDA * (state.V - self.E_ex) / denom

        i_syn = i_ampa + i_gaba + i_nmda

        dV = (-self.g_L * (state.V - self.E_L) - i_syn + extra.i_stim) / self.C_m

        ds_AMPA = -state.s_AMPA / self.tau_AMPA
        ds_GABA = -state.s_GABA / self.tau_GABA
        ds_NMDA = -state.s_NMDA / self.tau_decay_NMDA

        return DotDict(V=dV, s_AMPA=ds_AMPA, s_GABA=ds_GABA, s_NMDA=ds_NMDA)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, s_AMPA, s_GABA, s_NMDA -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, s_nmda_pre, last_spike_time,
            k0, k1, t_spike.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & ((state.V < -1e3 * u.mV) | (state.s_AMPA < -1e6 * u.nS) | (state.s_AMPA > 1e6 * u.nS))
        )

        # Spike detection: non-refractory (r <= 0 at step start), above threshold, not already spiked.
        # The ODE integrates freely; V reset and refractory clamping are applied post-integration.
        spike_now = accept & (extra.r <= 0) & (state.V >= self.V_th) & ~extra.spike_mask
        spike_mask = extra.spike_mask | spike_now

        # NMDA spike offset computation on spike
        dt_since_last = extra.t_spike - extra.last_spike_time
        s_pre_decayed = extra.s_nmda_pre * u.math.exp(-dt_since_last / self.tau_decay_NMDA)
        offset = extra.k0 + extra.k1 * s_pre_decayed
        s_pre_updated = s_pre_decayed + offset

        # Only apply NMDA updates on spike
        new_s_nmda_pre = u.math.where(spike_now, s_pre_updated, extra.s_nmda_pre)
        new_spike_offset = u.math.where(spike_now, offset, extra.spike_offset)
        new_last_spike_time = u.math.where(spike_now, extra.t_spike, extra.last_spike_time)

        new_extra = DotDict({
            **extra,
            'spike_mask': spike_mask,
            'unstable': unstable,
            's_nmda_pre': new_s_nmda_pre,
            'spike_offset': new_spike_offset,
            'last_spike_time': new_last_spike_time,
        })
        return state, new_extra

    def _parse_spike_events(self, spike_events: Iterable, state_shape):
        r"""Parse explicit spike events into per-receptor conductance increments.

        Parameters
        ----------
        spike_events : Iterable or None
            Incoming spike events. Each event is a tuple or dict with receptor
            type, weight, optional offset, and optional sender_model.
        state_shape : tuple
            Shape of the state arrays for broadcasting.

        Returns
        -------
        s_ampa, s_gaba, s_nmda : jax.numpy.ndarray
            Conductance increments for each receptor type (nS).
        """
        dftype = brainstate.environ.dftype()
        s_ampa = jnp.zeros(state_shape, dtype=dftype) * u.nS
        s_gaba = jnp.zeros(state_shape, dtype=dftype) * u.nS
        s_nmda = jnp.zeros(state_shape, dtype=dftype) * u.nS

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

            if receptor_id == self.AMPA:
                s_ampa = s_ampa + weight
            elif receptor_id == self.GABA:
                s_gaba = s_gaba + weight
            else:
                if sender_model != 'iaf_bw_2001':
                    raise ValueError(
                        'For NMDA synapses in iaf_bw_2001, pre-synaptic neuron must also be of type iaf_bw_2001.'
                    )
                s_nmda = s_nmda + weight * offset

        return s_ampa, s_gaba, s_nmda

    def update(self, x=0. * u.pA, spike_events=None):
        r"""Advance the neuron state by one simulation timestep.

        Performs a complete update cycle including: (1) RKF45 integration of
        ODEs, (2) reception of AMPA/GABA/NMDA spike events, (3) threshold
        detection and spike emission, (4) refractory period handling, (5) NMDA
        spike offset computation, and (6) delayed current buffering.

        Parameters
        ----------
        x : saiunit.Quantity, optional
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
            Spike output (differentiable). Shape: ``(*in_size,)``.
            Values in [0, 1] for typical surrogate functions.

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
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        s_AMPA = self.s_AMPA.value  # nS
        s_GABA = self.s_GABA.value  # nS
        s_NMDA = self.s_NMDA.value  # nS
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        s_nmda_pre = self.s_NMDA_pre.value
        last_spike_time = self.last_spike_time.value
        spike_offset_prev = self.spike_offset.value

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Use pre-computed NMDA jump constants (computed once in __init__).
        k0 = jnp.asarray(self._k0_np, dtype=dftype)
        k1 = jnp.asarray(self._k1_np, dtype=dftype)

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, s_AMPA=s_AMPA, s_GABA=s_GABA, s_NMDA=s_NMDA)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            s_nmda_pre=s_nmda_pre,
            spike_offset=jnp.zeros(self.varshape, dtype=dftype),
            last_spike_time=last_spike_time,
            k0=k0,
            k1=k1,
            t_spike=t + dt,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V_ode = ode_state.V  # Free ODE output — no refractory/spike reset applied yet.
        s_AMPA, s_GABA, s_NMDA = ode_state.s_AMPA, ode_state.s_GABA, ode_state.s_NMDA
        spike_mask, r_init, unstable = extra.spike_mask, extra.r, extra.unstable
        s_nmda_pre = extra.s_nmda_pre
        spike_offset_new = extra.spike_offset
        last_spike_time = extra.last_spike_time

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in iaf_bw_2001 dynamics.'
        )

        # Compute synaptic currents from FREE ODE output (before V reset/refractory clamping).
        # This matches NEST semantics: recorded currents reflect the ODE solution.
        v_mV = V_ode / u.mV
        conc_mM = self.conc_Mg2 / u.mM
        denom = 1.0 + conc_mM * u.math.exp(-0.062 * v_mV) / 3.57
        I_AMPA_val = s_AMPA * (V_ode - self.E_ex)
        I_GABA_val = s_GABA * (V_ode - self.E_in)
        I_NMDA_val = s_NMDA * (V_ode - self.E_ex) / denom

        # Apply refractory / spike reset post-integration (matching NEST).
        # V is clamped to V_reset if the neuron spiked this step OR is still refractory.
        V = u.math.where(spike_mask | (r_init > 0), self.V_reset, V_ode)

        # Update refractory counter:
        #   - spike this step and t_ref > 0 → start refractory (ref_count steps)
        #   - already refractory               → decrement
        #   - otherwise                        → keep at 0
        r = u.math.where(
            spike_mask & (self.ref_count > 0),
            self.ref_count,
            u.math.where(r_init > 0, r_init - 1, r_init),
        )

        # Synaptic spike inputs (applied after integration and current recording).
        # Parse explicit spike events.
        ev_ampa, ev_gaba, ev_nmda = self._parse_spike_events(spike_events, self.varshape)

        # Parse registered delta inputs by receptor label.
        w_ampa = self.sum_delta_inputs(u.math.zeros_like(self.s_AMPA.value), label='AMPA')
        w_gaba = self.sum_delta_inputs(u.math.zeros_like(self.s_GABA.value), label='GABA')
        w_nmda = self.sum_delta_inputs(u.math.zeros_like(self.s_NMDA.value), label='NMDA')

        # Apply synaptic spike inputs.
        s_AMPA = s_AMPA + ev_ampa + w_ampa
        s_GABA = s_GABA + ev_gaba + w_gaba
        s_NMDA = s_NMDA + ev_nmda + w_nmda

        # Write back state.
        self.V.value = V
        self.s_AMPA.value = s_AMPA
        self.s_GABA.value = s_GABA
        self.s_NMDA.value = s_NMDA

        self.I_AMPA.value = I_AMPA_val
        self.I_GABA.value = I_GABA_val
        self.I_NMDA.value = I_NMDA_val

        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA

        self.s_NMDA_pre.value = s_nmda_pre
        self.spike_offset.value = spike_offset_new
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
