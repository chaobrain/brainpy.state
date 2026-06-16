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

from typing import Callable, Optional, Sequence

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
    'gif_cond_exp_multisynapse',
]


class gif_cond_exp_multisynapse(NESTNeuron):
    r"""Conductance-based generalized integrate-and-fire neuron (GIF) model
    with multiple synaptic time constants.

    ``gif_cond_exp_multisynapse`` is the generalized integrate-and-fire neuron
    according to Mensi et al. (2012) [1]_ and Pozzorini et al. (2015) [2]_, with
    postsynaptic conductances in the form of truncated exponentials and an
    arbitrary number of synaptic receptor ports.

    This is a brainpy.state re-implementation of the NEST simulator model of the
    same name, using NEST-standard parameterization.

    This model features both an adaptation current and a dynamic threshold for
    spike-frequency adaptation. The membrane potential :math:`V` is described by
    the differential equation:

    .. math::

       C_\mathrm{m} \frac{dV(t)}{dt} = -g_\mathrm{L}(V(t) - E_\mathrm{L})
           - \sum_k g_k(t)(V(t) - E_{\mathrm{rev},k})
           - \eta_1(t) - \eta_2(t) - \ldots - \eta_n(t)
           + I_\mathrm{e} + I_\mathrm{stim}(t)

    where :math:`g_k(t)` are synaptic conductances for receptor port :math:`k`,
    each with its own reversal potential :math:`E_{\mathrm{rev},k}` and
    time constant :math:`\tau_{\mathrm{syn},k}`, and each :math:`\eta_i` is a
    spike-triggered current (stc).

    **1. Synaptic conductances**

    Each synaptic conductance decays exponentially:

    .. math::

       \frac{dg_k}{dt} = -\frac{g_k}{\tau_{\mathrm{syn},k}}

    On the postsynaptic side, there can be arbitrarily many synaptic time
    constants. This is achieved by specifying separate receptor ports, each
    for a different time constant. The number of receptor ports is determined
    by the length of the ``tau_syn`` and ``E_rev`` lists, which must have
    equal length.

    **2. Spike-triggered currents**

    Dynamic of each :math:`\eta_i` is described by:

    .. math::

       \tau_{\eta_i} \cdot \frac{d\eta_i}{dt} = -\eta_i

    and in case of spike emission, its value is increased by a constant:

    .. math::

       \eta_i = \eta_i + q_{\eta_i} \quad \text{(on spike emission)}

    **3. Spike-frequency adaptation**

    The neuron produces spikes stochastically according to a point process with
    the firing intensity:

    .. math::

       \lambda(t) = \lambda_0 \cdot \exp\left(\frac{V(t) - V_T(t)}{\Delta_V}\right)

    where :math:`V_T(t)` is a time-dependent firing threshold:

    .. math::

       V_T(t) = V_{T^*} + \gamma_1(t) + \gamma_2(t) + \ldots + \gamma_m(t)

    where :math:`\gamma_i` is a kernel of spike-frequency adaptation (sfa).
    Dynamic of each :math:`\gamma_i` is described by:

    .. math::

       \tau_{\gamma_i} \cdot \frac{d\gamma_i}{dt} = -\gamma_i

    and in case of spike emission, its value is increased by a constant:

    .. math::

       \gamma_i = \gamma_i + q_{\gamma_i} \quad \text{(on spike emission)}

    **4. Stochastic spiking**

    The probability of firing within a time step :math:`dt` is computed using
    the hazard function:

    .. math::

       P(\text{spike}) = 1 - \exp(-\lambda(t) \cdot dt)

    A random number is drawn each (non-refractory) time step and compared to
    this probability to determine whether a spike occurs.

    **5. Refractory mechanism**

    After a spike, the neuron enters an absolute refractory period of duration
    :math:`t_\mathrm{ref}`. During this period:

    * :math:`V_\mathrm{m}` is clamped to :math:`V_\mathrm{reset}`,
    * :math:`dV_\mathrm{m}/dt = 0`,
    * conductances continue to decay,
    * refractory counter decrements each step.

    **6. Numerical integration and update order**

    NEST integrates this model with adaptive RKF45. This implementation mirrors
    that behavior with an RKF45(4,5) integrator and persistent internal step size.
    The discrete-time update order per simulation step is:

    1. Compute total stc (sum of stc elements) and sfa threshold (V_T_star + sum
       of sfa elements). Then decay all stc and sfa elements by their respective
       exponential factors.
    2. Integrate continuous dynamics :math:`[V_\mathrm{m}, g_0, g_1, \ldots, g_{n-1}]`
       over :math:`(t, t+dt]` using RKF45.
    3. Add synaptic conductance jumps from spike inputs arriving this step
       (per receptor).
    4. If not refractory: compute firing intensity, draw random number,
       potentially emit spike (update stc/sfa elements, set refractory counter).
       If refractory: decrement counter, clamp V to V_reset.
    5. Store external current input as :math:`I_\mathrm{stim}` for the next step.

    **7. Multisynapse differences from gif_cond_exp**

    Unlike ``gif_cond_exp`` which has exactly two fixed synaptic channels
    (excitatory and inhibitory with separate parameters ``tau_syn_ex``,
    ``tau_syn_in``, ``E_ex``, ``E_in``), this model supports an arbitrary
    number of receptor ports specified by the lists ``tau_syn`` and ``E_rev``.

    When connecting to this model, all synaptic weights must be non-negative.
    Each connection specifies its receptor port via ``receptor_type``, which
    indexes into the ``tau_syn`` / ``E_rev`` arrays (1-based in NEST, but
    0-based for the ``add_delta_input`` interface here).

    .. note::

       In the NEST implementation, the stc and sfa element jumps occur immediately
       after spike emission. The GIF toolbox uses a different convention where
       jumps occur after the refractory period. Conversion:

       .. math::

          q_{\eta,\text{toolbox}} = q_{\eta,\text{NEST}} \cdot
              (1 - \exp(-t_\mathrm{ref} / \tau_\eta))

    .. note::

       Because spiking is stochastic (random number drawn each step), exact
       spike-time reproducibility requires matching the random number generator
       state. For deterministic testing, set ``rng_key`` explicitly.

    Parameters
    ----------
    in_size : int, sequence of int
        Population shape (e.g., 100 or (10, 10)). Required.
    g_L : ArrayLike, default: 4.0 nS
        Leak conductance. Must be strictly positive. Shape: scalar or broadcastable to ``in_size``.
    E_L : ArrayLike, default: -70.0 mV
        Leak reversal potential (resting potential). Shape: scalar or broadcastable to ``in_size``.
    C_m : ArrayLike, default: 80.0 pF
        Membrane capacitance. Must be strictly positive. Shape: scalar or broadcastable to ``in_size``.
    V_reset : ArrayLike, default: -55.0 mV
        Reset potential after spike. Shape: scalar or broadcastable to ``in_size``.
    Delta_V : ArrayLike, default: 0.5 mV
        Stochasticity level for exponential firing intensity. Must be strictly positive.
        Shape: scalar or broadcastable to ``in_size``.
    V_T_star : ArrayLike, default: -35.0 mV
        Base (non-adapting) firing threshold. Shape: scalar or broadcastable to ``in_size``.
    lambda_0 : float, default: 1.0
        Stochastic intensity at threshold (in 1/s). Must be non-negative. Internally converted to 1/ms.
    t_ref : ArrayLike, default: 4.0 ms
        Absolute refractory period duration. Must be non-negative. Shape: scalar or broadcastable to ``in_size``.
    tau_syn : Sequence[float], default: (2.0,)
        Synaptic conductance time constants for each receptor port (in ms). Each element must be
        strictly positive. Must have same length as ``E_rev``. At least one element required.
    E_rev : Sequence[float], default: (0.0,)
        Reversal potentials for each receptor port (in mV). Must have same length as ``tau_syn``.
        At least one element required.
    I_e : ArrayLike, default: 0.0 pA
        Constant external current. Shape: scalar or broadcastable to ``in_size``.
    tau_sfa : Sequence[float], default: ()
        Time constants for spike-frequency adaptation (SFA) threshold elements (in ms).
        Each element must be strictly positive. Must have same length as ``q_sfa``.
    q_sfa : Sequence[float], default: ()
        Jump values for SFA threshold elements (in mV). Must have same length as ``tau_sfa``.
    tau_stc : Sequence[float], default: ()
        Time constants for spike-triggered current (STC) elements (in ms).
        Each element must be strictly positive. Must have same length as ``q_stc``.
    q_stc : Sequence[float], default: ()
        Jump values for STC elements (in nA). Must have same length as ``tau_stc``.
    gsl_error_tol : ArrayLike, default: 1e-3
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
    rng_key : jax.Array, optional
        JAX PRNG key for stochastic spiking. If None, defaults to ``jax.random.PRNGKey(0)``.
    V_initializer : Callable, default: Constant(-70.0 mV)
        Initializer for membrane potential. Must return values compatible with ``in_size``.
    spk_fun : Callable, default: ReluGrad()
        Surrogate gradient function for spike generation. Used in gradient-based learning.
    spk_reset : str, default: 'hard'
        Spike reset mode. 'hard' (stop gradient, matches NEST) or 'soft' (subtract threshold).
    name : str, optional
        Module name. If None, auto-generated.

    Parameter Mapping
    -----------------
    Maps brainpy.state parameter names to NEST equivalents for cross-framework compatibility:

    ==================== =================== =================================== ============================================================
    **Parameter**        **Default**         **Math equivalent**                 **Description**
    ==================== =================== =================================== ============================================================
    ``in_size``          (required)                                              Population shape
    ``g_L``              4.0 nS              :math:`g_\mathrm{L}`                Leak conductance
    ``E_L``              -70.0 mV            :math:`E_\mathrm{L}`                Leak reversal potential
    ``C_m``              80.0 pF             :math:`C_\mathrm{m}`                Membrane capacitance
    ``V_reset``          -55.0 mV            :math:`V_\mathrm{reset}`            Reset potential
    ``Delta_V``          0.5 mV              :math:`\Delta_V`                    Stochasticity level
    ``V_T_star``         -35.0 mV            :math:`V_{T^*}`                     Base firing threshold
    ``lambda_0``         1.0 /s              :math:`\lambda_0`                   Stochastic intensity at threshold
    ``t_ref``            4.0 ms              :math:`t_\mathrm{ref}`              Absolute refractory period
    ``tau_syn``          (2.0,) ms           :math:`\tau_{\mathrm{syn},k}`       Synaptic conductance time constants (list)
    ``E_rev``            (0.0,) mV           :math:`E_{\mathrm{rev},k}`          Reversal potentials (list, same size as tau_syn)
    ``I_e``              0.0 pA              :math:`I_\mathrm{e}`                Constant external current
    ``tau_sfa``          () ms               :math:`\tau_{\gamma_i}`             SFA time constants (tuple/list)
    ``q_sfa``            () mV               :math:`q_{\gamma_i}`                SFA jump values (tuple/list)
    ``tau_stc``          () ms               :math:`\tau_{\eta_i}`               STC time constants (tuple/list)
    ``q_stc``            () nA               :math:`q_{\eta_i}`                  STC jump values (tuple/list)
    ``gsl_error_tol``    1e-3                --                                  Local absolute tolerance for RKF45 error estimate
    ``rng_key``          None                                                    JAX PRNG key for stochastic spiking
    ``V_initializer``    Constant(-70 mV)                                        Initializer for membrane potential
    ``spk_fun``          ReluGrad()                                              Surrogate spike function
    ``spk_reset``        ``'hard'``                                              Reset mode; hard reset matches NEST
    ==================== =================== =================================== ============================================================

    State Variables
    ---------------
    After ``init_state()``, the following state variables are available:

    ========================== ==================== ==============================================================
    **State variable**         **Type**             **Description**
    ========================== ==================== ==============================================================
    ``V``                      HiddenState          Membrane potential :math:`V_\mathrm{m}` (mV)
    ``g``                      List[HiddenState]    List of synaptic conductances :math:`g_k` (nS), one per receptor
    ``refractory_step_count``  ShortTermState       Remaining refractory grid steps (int32)
    ``integration_step``       ShortTermState       Internal RKF45 step-size state (ms)
    ``I_stim``                 ShortTermState       Buffered current applied in next step (pA)
    ``last_spike_time``        ShortTermState       Last spike time (ms)
    ========================== ==================== ==============================================================

    Additionally, the following NumPy arrays are maintained internally:

    - ``_stc_elems`` -- shape ``(len(tau_stc), *in_size)`` -- individual stc elements (nA)
    - ``_sfa_elems`` -- shape ``(len(tau_sfa), *in_size)`` -- individual sfa elements (mV)
    - ``_stc_val`` -- shape ``in_size`` -- total spike-triggered current (nA)
    - ``_sfa_val`` -- shape ``in_size`` -- adaptive threshold :math:`V_T(t)` (mV)

    Raises
    ------
    ValueError
        If ``C_m <= 0``, ``g_L <= 0``, ``Delta_V <= 0``, ``t_ref < 0``, ``lambda_0 < 0``,
        any ``tau_syn <= 0``, any ``tau_sfa <= 0``, any ``tau_stc <= 0``,
        ``len(tau_syn) != len(E_rev)``, ``len(tau_syn) == 0``,
        ``len(tau_sfa) != len(q_sfa)``, or ``len(tau_stc) != len(q_stc)``.

    Notes
    -----
    - Defaults follow NEST C++ source for ``gif_cond_exp_multisynapse``.
    - ``lambda_0`` is specified in 1/s (as in NEST's Python interface) and is
      internally converted to 1/ms for computation.
    - All synaptic spike weights must be non-negative (conductance-based model).
    - Delta inputs for synaptic conductances are indexed by receptor port
      (0-based). Use ``add_delta_input(f'receptor_{port}', weight * u.nS)``
      to add a conductance jump to a specific receptor port. If no receptor
      port is specified in the key, the input defaults to receptor 0.
    - RKF45 integration with adaptive step size ensures numerical stability for stiff systems,
      matching NEST's GSL-based integrator behavior.
    - The stochastic spiking mechanism uses JAX PRNG, which is split each time step to ensure
      reproducible randomness under JIT compilation.
    - The model supports an arbitrary number of receptor ports, making it suitable for
      modeling neurons with multiple synaptic receptor types (e.g., AMPA, NMDA, GABA_A, GABA_B).

    References
    ----------
    .. [1] Mensi S, Naud R, Pozzorini C, Avermann M, Petersen CC, Gerstner W
           (2012). Parameter extraction and classification of three cortical
           neuron types reveals two distinct adaptation mechanisms. Journal of
           Neurophysiology, 107(6):1756-1775.
           DOI: https://doi.org/10.1152/jn.00408.2011
    .. [2] Pozzorini C, Mensi S, Hagens O, Naud R, Koch C, Gerstner W (2015).
           Automated high-throughput characterization of single neurons by means
           of simplified spiking models. PLoS Computational Biology, 11(6),
           e1004275.
           DOI: https://doi.org/10.1371/journal.pcbi.1004275
    .. [3] NEST Simulator ``gif_cond_exp_multisynapse`` model documentation and
           C++ source: ``models/gif_cond_exp_multisynapse.h`` and
           ``models/gif_cond_exp_multisynapse.cpp``.

    See Also
    --------
    gif_cond_exp : Two-channel GIF model with fixed excitatory/inhibitory receptors
    iaf_cond_exp : Conductance-based leaky integrate-and-fire
    gif_psc_exp_multisynapse : Current-based GIF with multiple synaptic time constants

    Examples
    --------
    Create a GIF neuron with three receptor ports (e.g., AMPA, NMDA, GABA_A):

    .. code-block:: python

        >>> from brainpy import state as bst
        >>> import brainunit as u
        >>> # AMPA: fast excitatory, NMDA: slow excitatory, GABA_A: fast inhibitory
        >>> neuron = bst.gif_cond_exp_multisynapse(
        ...     in_size=100,
        ...     tau_syn=(2.0, 20.0, 5.0),  # ms
        ...     E_rev=(0.0, 0.0, -85.0),   # mV
        ...     tau_sfa=(100.0,),           # ms
        ...     q_sfa=(5.0,),               # mV
        ...     tau_stc=(50.0,),            # ms
        ...     q_stc=(10.0,)               # nA
        ... )
        >>> neuron.init_all_states()
        >>> # Add AMPA input to receptor 0
        >>> neuron.add_delta_input('receptor_0', 0.5 * u.nS)
        >>> # Add NMDA input to receptor 1
        >>> neuron.add_delta_input('receptor_1', 0.3 * u.nS)
        >>> # Add GABA_A input to receptor 2
        >>> neuron.add_delta_input('receptor_2', 0.8 * u.nS)
    """
    __module__ = 'brainpy.state'

    #: Unit the multi-receptor ``connect(receptor_type=k)`` bridge uses to scale the
    #: gathered per-port deposit into the ``w_by_rec`` mantissa (conductance -> nS).
    receptor_input_unit = u.nS

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 10000

    def __init__(
        self,
        in_size: Size,
        g_L: ArrayLike = 4.0 * u.nS,
        E_L: ArrayLike = -70.0 * u.mV,
        C_m: ArrayLike = 80.0 * u.pF,
        V_reset: ArrayLike = -55.0 * u.mV,
        Delta_V: ArrayLike = 0.5 * u.mV,
        V_T_star: ArrayLike = -35.0 * u.mV,
        lambda_0: float = 1.0,  # 1/s, as in NEST Python interface
        t_ref: ArrayLike = 4.0 * u.ms,
        tau_syn: Sequence[float] = (2.0,),  # ms values
        E_rev: Sequence[float] = (0.0,),  # mV values
        I_e: ArrayLike = 0.0 * u.pA,
        tau_sfa: Sequence[float] = (),  # ms values
        q_sfa: Sequence[float] = (),  # mV values
        tau_stc: Sequence[float] = (),  # ms values
        q_stc: Sequence[float] = (),  # nA values
        gsl_error_tol: ArrayLike = 1e-3,
        rng_key: Optional[jax.Array] = None,
        V_initializer: Callable = braintools.init.Constant(-70.0 * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Membrane parameters
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.Delta_V = braintools.init.param(Delta_V, self.varshape)
        self.V_T_star = braintools.init.param(V_T_star, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        # Synaptic parameters (lists)
        self.tau_syn = tuple(float(x) for x in tau_syn)
        self.E_rev = tuple(float(x) for x in E_rev)

        if len(self.tau_syn) != len(self.E_rev):
            raise ValueError(
                f"'tau_syn' and 'E_rev' must have the same length. "
                f"Got {len(self.tau_syn)} and {len(self.E_rev)}."
            )
        if len(self.tau_syn) == 0:
            raise ValueError("'tau_syn' must have at least one element.")

        # Stochastic spiking: lambda_0 in 1/s, store as 1/ms internally
        self.lambda_0 = lambda_0 / 1000.0  # convert from 1/s to 1/ms

        # Adaptation parameters (stored as plain Python lists of floats in ms/mV/nA)
        self.tau_sfa = tuple(float(x) for x in tau_sfa)
        self.q_sfa = tuple(float(x) for x in q_sfa)
        self.tau_stc = tuple(float(x) for x in tau_stc)
        self.q_stc = tuple(float(x) for x in q_stc)

        if len(self.tau_sfa) != len(self.q_sfa):
            raise ValueError(
                f"'tau_sfa' and 'q_sfa' must have the same length. "
                f"Got {len(self.tau_sfa)} and {len(self.q_sfa)}."
            )
        if len(self.tau_stc) != len(self.q_stc):
            raise ValueError(
                f"'tau_stc' and 'q_stc' must have the same length. "
                f"Got {len(self.tau_stc)} and {len(self.q_stc)}."
            )

        # RNG key for stochastic spiking
        self._rng_key = rng_key

        # Initializer
        self.V_initializer = V_initializer

        # Number of receptor ports
        self._n_receptors = len(self.tau_syn)

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
            Number of receptor ports, equal to ``len(tau_syn)`` and ``len(E_rev)``.
        """
        return self._n_receptors

    def _validate_parameters(self):
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.g_L, self.Delta_V)):
            return
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.g_L <= 0.0 * u.nS):
            raise ValueError('Membrane conductance must be strictly positive.')
        if cond_any(self.Delta_V <= 0.0 * u.mV):
            raise ValueError('Delta_V must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time must not be negative.')
        if self.lambda_0 < 0.0:
            raise ValueError('lambda_0 must not be negative.')
        for i, tau in enumerate(self.tau_syn):
            if tau <= 0.0:
                raise ValueError(f'All synaptic time constants must be strictly positive (tau_syn[{i}]={tau}).')
        for i, tau in enumerate(self.tau_sfa):
            if tau <= 0.0:
                raise ValueError(f'All SFA time constants must be strictly positive (tau_sfa[{i}]={tau}).')
        for i, tau in enumerate(self.tau_stc):
            if tau <= 0.0:
                raise ValueError(f'All STC time constants must be strictly positive (tau_stc[{i}]={tau}).')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        This method initializes:
        - ``V``: membrane potential to ``V_initializer`` values
        - ``g``: list of ``n_receptors`` conductance states, all initialized to 0 nS
        - ``last_spike_time``: initialized to -1e7 ms (far past)
        - ``refractory_step_count``: initialized to 0
        - ``integration_step``: initialized to current dt
        - ``I_stim``: initialized to 0 pA
        - ``_stc_elems``, ``_sfa_elems``: NumPy arrays for adaptation elements, all zeros
        - ``_stc_val``, ``_sfa_val``: computed totals
        - ``_rng_state``: JAX PRNG state
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        self.V = brainstate.HiddenState(V)

        # Initialize per-receptor conductance states
        self.g = [
            brainstate.HiddenState(
                braintools.init.param(
                    braintools.init.Constant(0.0 * u.nS),
                    self.varshape
                )
            )
            for _ in range(self._n_receptors)
        ]

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA))

        # Adaptation state: JAX arrays wrapped in brainstate states for JIT compatibility.
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape
        self._stc_elems_state = (
            brainstate.HiddenState(jnp.zeros((n_stc, *v_shape), dtype=jnp.float64))
            if n_stc > 0 else None
        )
        self._sfa_elems_state = (
            brainstate.HiddenState(jnp.zeros((n_sfa, *v_shape), dtype=jnp.float64))
            if n_sfa > 0 else None
        )
        self._stc_val_state = brainstate.ShortTermState(
            jnp.zeros(v_shape, dtype=jnp.float64)
        )
        V_T_star_mV = float(np.asarray(u.get_mantissa(self.V_T_star / u.mV)))
        self._sfa_val_state = brainstate.ShortTermState(
            jnp.full(v_shape, V_T_star_mV, dtype=jnp.float64)
        )

        # RNG state as a brainstate ShortTermState for JIT compatibility.
        rng_init = self._rng_key if self._rng_key is not None else jax.random.PRNGKey(0)
        self._rng_state_state = brainstate.ShortTermState(rng_init)

    @property
    def _stc_elems(self):
        """Spike-triggered current elements (n_stc, *varshape), float64."""
        return self._stc_elems_state.value if self._stc_elems_state is not None else None

    @property
    def _sfa_elems(self):
        """Spike-frequency adaptation elements (n_sfa, *varshape), float64."""
        return self._sfa_elems_state.value if self._sfa_elems_state is not None else None

    @property
    def _stc_val(self):
        """Total STC current at the start of the last update step (*varshape), float64."""
        return self._stc_val_state.value

    @property
    def _sfa_val(self):
        """Effective firing threshold (V_T_star + sum of sfa elements) (*varshape), float64."""
        return self._sfa_val_state.value

    def get_spike(self, V: ArrayLike = None):
        r"""Compute spike output using surrogate gradient function.

        This method is used for gradient-based learning. The actual spike emission
        during simulation is stochastic and handled in ``update()``.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential. If None, uses ``self.V.value``. Shape: ``(*batch_size, *in_size)`` with unit mV.

        Returns
        -------
        ArrayLike
            Spike values in [0, 1] range. Shape matches input ``V``. Dimensionless.
            Uses ``spk_fun`` to compute differentiable spike output from scaled voltage.

        Notes
        -----
        The spike output is computed as ``spk_fun((V - V_reset) / Delta_V)``, providing
        a differentiable approximation for gradient-based learning. This is separate from
        the stochastic spike mechanism used in ``update()``.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_reset) / (self.Delta_V)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for GIF cond exp multisynapse dynamics.

        Computes derivatives for membrane voltage and per-receptor conductances.
        During refractory period, V is clamped to V_reset and dV/dt=0.
        Conductances always decay regardless of refractory state.

        Parameters
        ----------
        state : DotDict
            Keys: V, and per-receptor g_0..g_{n-1} -- ODE state variables.
        extra : DotDict
            Keys: r, i_stim, stc_total -- mutable auxiliary data carried
            through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, state.V)

        # I_syn = sum_k g_k * (E_rev_k - V)
        i_syn = 0.0 * u.pA
        for k in range(self._n_receptors):
            g_k = state[f'g_{k}']
            E_rev_k = self.E_rev[k] * u.mV
            i_syn = i_syn + g_k * (E_rev_k - v_eff)

        # stc_total carries the unitless numeric value of the spike-triggered
        # current in the same scale as the other currents (pA-equivalent when
        # g_L is nS, V is mV, I_e is pA, etc.).
        stc_current = extra.stc_total * u.pA

        dV_raw = (
                     -self.g_L * (v_eff - self.E_L) + i_syn
                     - stc_current + self.I_e + extra.i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        result = DotDict(V=dV)
        for k in range(self._n_receptors):
            tau_k = self.tau_syn[k] * u.ms
            g_k = state[f'g_{k}']
            result[f'g_{k}'] = -g_k / tau_k

        return result

    def _event_fn(self, state, extra, accept):
        """In-loop refractory handling and stability check.

        For GIF models, spike detection is stochastic and happens outside
        the ODE integration loop. This event function only handles
        refractory voltage clamping and numerical stability monitoring.

        Parameters
        ----------
        state : DotDict
            Keys: V, and per-receptor g_0..g_{n-1}.
        extra : DotDict
            Keys: r, i_stim, stc_total, unstable.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated refractory/stability info.
        """
        unstable = extra.unstable | jnp.any(
            accept & (state.V < -1e3 * u.mV)
        )

        refr_accept = accept & (extra.r > 0)
        new_V = u.math.where(refr_accept, self.V_reset, state.V)

        new_state = DotDict({**state, 'V': new_V})
        new_extra = DotDict({**extra, 'unstable': unstable})
        return new_state, new_extra

    def _collect_receptor_delta_inputs(self):
        r"""Collect delta inputs per receptor port.

        Parses ``delta_inputs`` dictionary to extract conductance jumps for each receptor port.
        Keys containing 'receptor_<k>' are routed to receptor port k (0-based indexing).
        Inputs without a receptor specification default to receptor 0.

        Returns
        -------
        list of np.ndarray
            List of length ``n_receptors``, where each element is a NumPy array of shape
            matching ``V.value.shape``, containing conductance jumps in nS (float64).
            Inputs not matching any receptor pattern are added to receptor 0.

        Notes
        -----
        This method consumes callable delta inputs by invoking them and removing them
        from ``delta_inputs``. Non-callable inputs are removed after first use.
        For non-synaptic currents, use ``add_current_input()`` instead.
        """
        dftype = brainstate.environ.dftype()
        dg = [jnp.zeros(self.varshape, dtype=dftype) for _ in range(self._n_receptors)]

        if self.delta_inputs is None:
            return dg

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            # Determine which receptor this input belongs to
            out_nS = jnp.broadcast_to(
                jnp.asarray(u.get_mantissa(out / u.nS), dtype=dftype),
                self.varshape,
            )

            # Parse receptor index from key
            # Keys should be like "receptor_0", "receptor_1", etc.
            # or "receptor_0_stepK" for unique-per-step keys
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
                # If no receptor specified, add to receptor 0 as default
                dg[0] = dg[0] + out_nS

        return dg

    def update(self, x=0.0 * u.pA, w_by_rec=None):
        r"""Update neuron state for one time step.

        Performs the following operations in order:
        1. Decay adaptation elements (stc, sfa) and compute totals
        2. Integrate membrane and conductance dynamics using RKF45
        3. Add synaptic conductance jumps from spike inputs
        4. Stochastic spike check (if not refractory) or refractory countdown
        5. Store external current for next step

        Parameters
        ----------
        x : ArrayLike, default: 0.0 pA
            External current input for this time step. Shape: scalar or broadcastable to ``in_size``.
            Unit: pA. This is stored as ``I_stim`` for use in the **next** time step.
        w_by_rec : ArrayLike or None, optional
            Pre-gathered per-receptor conductance jumps (nS mantissa), shape
            ``(*in_size, n_receptors)``. When given, it is used directly and the
            legacy per-key ``add_delta_input('receptor_k', ...)`` parser is
            bypassed. This is the JIT-safe path the multi-receptor
            ``connect(receptor_type=k)`` seam drives via the Simulator bridge.
            Default: None.

        Returns
        -------
        jnp.ndarray
            Binary spike output (0 or 1) for this time step. Shape: ``(*batch_size, *in_size)``.
            dtype: float32. Value 1 indicates spike emission.

        Notes
        -----
        The update implements NEST's exact algorithm:

        - Adaptation elements decay exponentially by factor ``exp(-dt/tau)``
        - RKF45 integration uses adaptive step size (stored in ``integration_step`` state)
        - Spike probability computed as ``1 - exp(-lambda * dt)`` where
          ``lambda = lambda_0 * exp((V - V_T) / Delta_V)``
        - Random number generation uses JAX PRNG with automatic state splitting
        - Refractory neurons have V clamped to V_reset and cannot spike
        - All computations performed element-wise in NumPy for each neuron

        The current input ``x`` is buffered (not used in current step) to match NEST's
        one-step delay convention.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        dt = float(u.get_mantissa(dt_q / u.ms))
        n_dims = len(self.varshape)

        # Read state.
        V = self.V.value
        r = self.refractory_step_count.value
        i_stim = self.I_stim.value
        h = self.integration_step.value

        # ---- Step 1: Compute stc/sfa totals (before decay), then decay. ----
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)

        if n_stc > 0:
            stc_elems = self._stc_elems_state.value          # (n_stc, *varshape)
            stc_total = jnp.sum(stc_elems, axis=0)           # (*varshape)
            P_stc_arr = jnp.array(
                [np.exp(-dt / tau) for tau in self.tau_stc], dtype=jnp.float64
            ).reshape(n_stc, *([1] * n_dims))
            stc_elems_decayed = stc_elems * P_stc_arr
        else:
            stc_total = jnp.zeros(self.varshape, dtype=jnp.float64)
            stc_elems_decayed = None

        V_T_star_arr = jnp.full(
            self.varshape,
            float(np.asarray(u.get_mantissa(self.V_T_star / u.mV))),
            dtype=jnp.float64,
        )
        if n_sfa > 0:
            sfa_elems = self._sfa_elems_state.value           # (n_sfa, *varshape)
            sfa_total = V_T_star_arr + jnp.sum(sfa_elems, axis=0)  # (*varshape)
            P_sfa_arr = jnp.array(
                [np.exp(-dt / tau) for tau in self.tau_sfa], dtype=jnp.float64
            ).reshape(n_sfa, *([1] * n_dims))
            sfa_elems_decayed = sfa_elems * P_sfa_arr
        else:
            sfa_total = V_T_star_arr
            sfa_elems_decayed = None

        self._stc_val_state.value = stc_total
        self._sfa_val_state.value = sfa_total

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)

        # ---- Step 2: Integrate ODE [V, g_0, g_1, ...] via adaptive RKF45. ----
        ode_state = DotDict(V=V)
        for k in range(self._n_receptors):
            ode_state[f'g_{k}'] = self.g[k].value

        extra = DotDict(
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            stc_total=stc_total,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V = ode_state.V
        unstable = extra.unstable
        g_list = [ode_state[f'g_{k}'] for k in range(self._n_receptors)]

        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in gif_cond_exp_multisynapse dynamics.'
        )

        # ---- Step 3: Add synaptic conductance jumps from spike inputs. ----
        if w_by_rec is not None:
            # Bridged blob path (``connect(receptor_type=k)`` seam): per-receptor
            # nS-mantissa jumps gathered by the Simulator, used directly. Bypasses
            # the legacy ``key.split('_')`` parser (kept for the per-key API).
            w_by_rec = jnp.asarray(w_by_rec, dtype=brainstate.environ.dftype())
            dg = [w_by_rec[..., k] for k in range(self._n_receptors)]
        else:
            dg = self._collect_receptor_delta_inputs()
        for k in range(self._n_receptors):
            g_list[k] = g_list[k] + dg[k] * u.nS

        # ---- Step 4: Vectorized stochastic spike check. ----
        V_mV = jnp.asarray(u.get_mantissa(V / u.mV), dtype=jnp.float64)
        V_reset_mV = jnp.asarray(u.get_mantissa(self.V_reset / u.mV), dtype=jnp.float64)
        Delta_V_arr = jnp.asarray(u.get_mantissa(self.Delta_V / u.mV), dtype=jnp.float64)
        lambda_0 = self.lambda_0  # 1/ms

        lam = lambda_0 * jnp.exp((V_mV - sfa_total) / Delta_V_arr)
        spike_prob = jnp.where(lam > 0.0, -jnp.expm1(-lam * dt), 0.0)

        new_rng, subkey = jax.random.split(self._rng_state_state.value)
        self._rng_state_state.value = new_rng
        rand_vals = jax.random.uniform(subkey, shape=self.varshape, dtype=jnp.float64)

        # Spike only for non-refractory neurons.
        r_int = jnp.asarray(r, dtype=jnp.int32)
        spike_mask = (rand_vals < spike_prob) & (r_int == 0)

        # Update adaptation elements for spiked neurons.
        if n_stc > 0 or n_sfa > 0:
            spike_mask_f = spike_mask.astype(jnp.float64)
            if n_stc > 0:
                q_stc_arr = jnp.array(self.q_stc, dtype=jnp.float64).reshape(
                    n_stc, *([1] * n_dims)
                )
                self._stc_elems_state.value = stc_elems_decayed + q_stc_arr * spike_mask_f
            if n_sfa > 0:
                q_sfa_arr = jnp.array(self.q_sfa, dtype=jnp.float64).reshape(
                    n_sfa, *([1] * n_dims)
                )
                self._sfa_elems_state.value = sfa_elems_decayed + q_sfa_arr * spike_mask_f

        # Update refractory counter; clamp V for currently refractory neurons.
        ref_count = jnp.asarray(u.get_mantissa(self.ref_count), dtype=jnp.int32)
        r_new = jnp.where(
            r_int > 0,
            r_int - 1,
            jnp.where(spike_mask, ref_count, jnp.zeros_like(r_int)),
        )
        V_mV_new = jnp.where(r_int > 0, V_reset_mV, V_mV)

        # Write back state.
        self.V.value = V_mV_new * u.mV
        for k in range(self._n_receptors):
            self.g[k].value = g_list[k]
        self.refractory_step_count.value = jnp.asarray(r_new, dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return u.math.asarray(spike_mask, dtype=dftype)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to their initial values.

        Resets membrane potential, conductances, refractory counter,
        integration step, stimulus buffer, adaptation elements, and RNG state.

        Parameters
        ----------
        batch_size : int, optional
            Unused; present for API compatibility.
        **kwargs
            Unused compatibility parameters.
        """
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        V = braintools.init.param(self.V_initializer, self.varshape)
        self.V.value = V

        for k in range(self._n_receptors):
            self.g[k].value = braintools.init.param(
                braintools.init.Constant(0.0 * u.nS), self.varshape
            )

        self.last_spike_time.value = u.math.full(self.varshape, -1e7 * u.ms)
        self.refractory_step_count.value = u.math.full(self.varshape, 0, dtype=ditype)
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape
        )
        self.I_stim.value = u.math.full(self.varshape, 0.0 * u.pA)

        if self._stc_elems_state is not None:
            self._stc_elems_state.value = jnp.zeros_like(self._stc_elems_state.value)
        if self._sfa_elems_state is not None:
            self._sfa_elems_state.value = jnp.zeros_like(self._sfa_elems_state.value)

        self._stc_val_state.value = jnp.zeros(self.varshape, dtype=jnp.float64)
        self._sfa_val_state.value = jnp.full(
            self.varshape,
            float(np.asarray(u.get_mantissa(self.V_T_star / u.mV))),
            dtype=jnp.float64,
        )

        rng_init = self._rng_key if self._rng_key is not None else jax.random.PRNGKey(0)
        self._rng_state_state.value = rng_init
