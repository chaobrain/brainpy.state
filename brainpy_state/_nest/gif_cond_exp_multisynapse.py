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
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, AdaptiveRungeKuttaStep

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

        >>> import brainpy.state as bst
        >>> import saiunit as u
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
        if np.any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self.g_L <= 0.0 * u.nS):
            raise ValueError('Membrane conductance must be strictly positive.')
        if np.any(self.Delta_V <= 0.0 * u.mV):
            raise ValueError('Delta_V must be strictly positive.')
        if np.any(self.t_ref < 0.0 * u.ms):
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
        if np.any(self.gsl_error_tol <= 0.0):
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
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        # Adaptation state
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)
        V_T_star_val = float(np.asarray(u.math.asarray(self.V_T_star / u.mV)))
        self._sfa_val = np.full(v_shape, V_T_star_val, dtype=np.float64)

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

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

            # Determine which receptor this input belongs to
            out_nS = np.asarray(u.math.asarray(out / u.nS), dtype=dftype)
            out_nS = np.broadcast_to(out_nS, v_shape)

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

    def update(self, x=0.0 * u.pA):
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
        dt = float(np.asarray(u.math.asarray(dt_q / u.ms)))

        v_shape = self.V.value.shape

        # Read state
        V = self.V.value  # mV
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # ---- Step 1: Decay stc/sfa elements and compute totals ----
        P_stc = [np.exp(-dt / tau) for tau in self.tau_stc]
        P_sfa = [np.exp(-dt / tau) for tau in self.tau_sfa]

        stc_total_np = np.zeros(v_shape, dtype=np.float64)
        if self._stc_elems is not None:
            for i in range(len(self.tau_stc)):
                stc_total_np = stc_total_np + self._stc_elems[i]
                self._stc_elems[i] = self._stc_elems[i] * P_stc[i]

        V_T_star_val = float(np.asarray(u.math.asarray(self.V_T_star / u.mV)))
        sfa_total_np = np.full(v_shape, V_T_star_val, dtype=np.float64)
        if self._sfa_elems is not None:
            for i in range(len(self.tau_sfa)):
                sfa_total_np = sfa_total_np + self._sfa_elems[i]
                self._sfa_elems[i] = self._sfa_elems[i] * P_sfa[i]

        self._stc_val = stc_total_np
        self._sfa_val = sfa_total_np

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # ---- Step 2: Integrate ODE [V, g_0, g_1, ...] via adaptive RKF45 ----
        # Build ODE state DotDict with per-receptor g_k.
        ode_state = DotDict(V=V)
        for k in range(self._n_receptors):
            ode_state[f'g_{k}'] = self.g[k].value

        # stc_total as unitless jax array (nA values)
        stc_total_jax = jnp.asarray(stc_total_np)

        extra = DotDict(
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            stc_total=stc_total_jax,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V = ode_state.V
        unstable = extra.unstable

        # Extract per-receptor conductances back.
        g_list = []
        for k in range(self._n_receptors):
            g_list.append(ode_state[f'g_{k}'])

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in gif_cond_exp_multisynapse dynamics.'
        )

        # ---- Step 3: Add synaptic conductance jumps from spike inputs ----
        dg = self._collect_receptor_delta_inputs()
        for k in range(self._n_receptors):
            g_list[k] = g_list[k] + dg[k] * u.nS

        # ---- Step 4: Refractory / spike check ----
        # Extract V as unitless mV for stochastic spike computation
        V_mV = np.asarray(u.math.asarray(V / u.mV), dtype=np.float64)
        V_mV = np.broadcast_to(V_mV, v_shape).copy()
        V_reset_mV = np.asarray(u.math.asarray(self.V_reset / u.mV), dtype=np.float64)
        V_reset_mV = np.broadcast_to(V_reset_mV, v_shape)
        Delta_V_val = float(np.asarray(u.math.asarray(self.Delta_V / u.mV)))
        lambda_0 = self.lambda_0  # 1/ms

        r_np = np.asarray(u.math.asarray(u.get_mantissa(extra.r)), dtype=np.int32)
        r_np = np.broadcast_to(r_np, v_shape).copy()
        refr_counts = np.broadcast_to(
            np.asarray(u.math.asarray(u.get_mantissa(self.ref_count)), dtype=np.int32), v_shape
        )

        # Advance RNG state
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(jax.random.uniform(subkey, shape=v_shape), dtype=np.float64)

        spike_mask = np.zeros(v_shape, dtype=bool)

        for idx in np.ndindex(v_shape):
            if r_np[idx] == 0:
                # Neuron is not refractory
                lam = lambda_0 * np.exp((V_mV[idx] - sfa_total_np[idx]) / Delta_V_val)
                if lam > 0.0:
                    spike_prob = -np.expm1(-lam * dt)
                    if rand_vals[idx] < spike_prob:
                        # Spike!
                        spike_mask[idx] = True

                        if self._stc_elems is not None:
                            for i in range(len(self.q_stc)):
                                self._stc_elems[i][idx] += self.q_stc[i]

                        if self._sfa_elems is not None:
                            for i in range(len(self.q_sfa)):
                                self._sfa_elems[i][idx] += self.q_sfa[i]

                        r_np[idx] = refr_counts[idx]
            else:
                # Neuron is refractory
                r_np[idx] = r_np[idx] - 1
                V_mV[idx] = V_reset_mV[idx]

        # Write back state
        self.V.value = jnp.asarray(V_mV) * u.mV
        for k in range(self._n_receptors):
            self.g[k].value = g_list[k]
        self.refractory_step_count.value = jnp.asarray(r_np, dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return u.math.asarray(spike_mask, dtype=dftype)
