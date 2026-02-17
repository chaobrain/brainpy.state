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
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'gif_cond_exp',
]


class gif_cond_exp(Neuron):
    r"""Conductance-based generalized integrate-and-fire neuron (GIF) model.

    ``gif_cond_exp`` is the generalized integrate-and-fire neuron according to
    Mensi et al. (2012) [1]_ and Pozzorini et al. (2015) [2]_, with postsynaptic
    conductances in the form of truncated exponentials.

    This is a brainpy.state re-implementation of the NEST simulator model of the
    same name, using NEST-standard parameterization.

    This model features both an adaptation current and a dynamic threshold for
    spike-frequency adaptation. The membrane potential :math:`V` is described by
    the differential equation:

    .. math::

       C_\mathrm{m} \frac{dV(t)}{dt} = -g_\mathrm{L}(V(t) - E_\mathrm{L})
           - g_\mathrm{ex}(t)(V(t) - E_\mathrm{ex})
           - g_\mathrm{in}(t)(V(t) - E_\mathrm{in})
           - \eta_1(t) - \eta_2(t) - \ldots - \eta_n(t)
           + I_\mathrm{e} + I_\mathrm{stim}(t)

    where each :math:`\eta_i` is a spike-triggered current (stc), and the neuron
    model can have an arbitrary number of them.

    Synaptic conductances decay exponentially:

    .. math::

       \frac{dg_\mathrm{ex}}{dt} = -\frac{g_\mathrm{ex}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{dg_\mathrm{in}}{dt} = -\frac{g_\mathrm{in}}{\tau_{\mathrm{syn,in}}}.

    **1. Spike-triggered currents**

    Dynamic of each :math:`\eta_i` is described by:

    .. math::

       \tau_{\eta_i} \cdot \frac{d\eta_i}{dt} = -\eta_i

    and in case of spike emission, its value is increased by a constant:

    .. math::

       \eta_i = \eta_i + q_{\eta_i} \quad \text{(on spike emission)}

    **2. Spike-frequency adaptation**

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

    **3. Stochastic spiking**

    The probability of firing within a time step :math:`dt` is computed using
    the hazard function:

    .. math::

       P(\text{spike}) = 1 - \exp(-\lambda(t) \cdot dt)

    A random number is drawn each (non-refractory) time step and compared to
    this probability to determine whether a spike occurs.

    **4. Refractory mechanism**

    After a spike, the neuron enters an absolute refractory period of duration
    :math:`t_\mathrm{ref}`. During this period:

    * :math:`V_\mathrm{m}` is clamped to :math:`V_\mathrm{reset}`,
    * :math:`dV_\mathrm{m}/dt = 0`,
    * conductances continue to decay,
    * refractory counter decrements each step.

    **5. Numerical integration and update order**

    NEST integrates this model with adaptive RKF45. This implementation mirrors
    that behavior with an RKF45(4,5) integrator and persistent internal step size.
    The discrete-time update order per simulation step is:

    1. Compute total stc (sum of stc elements) and sfa threshold (V_T_star + sum
       of sfa elements). Then decay all stc and sfa elements by their respective
       exponential factors.
    2. Integrate continuous dynamics :math:`[V_\mathrm{m}, g_\mathrm{ex}, g_\mathrm{in}]`
       over :math:`(t, t+dt]` using RKF45.
    3. Add synaptic conductance jumps from spike inputs arriving this step.
    4. If not refractory: compute firing intensity, draw random number,
       potentially emit spike (update stc/sfa elements, set refractory counter).
       If refractory: decrement counter, clamp V to V_reset.
    5. Store external current input as :math:`I_\mathrm{stim}` for the next step.

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
    E_ex : ArrayLike, default: 0.0 mV
        Excitatory reversal potential. Shape: scalar or broadcastable to ``in_size``.
    E_in : ArrayLike, default: -85.0 mV
        Inhibitory reversal potential. Shape: scalar or broadcastable to ``in_size``.
    tau_syn_ex : ArrayLike, default: 2.0 ms
        Excitatory conductance decay time constant. Must be strictly positive.
        Shape: scalar or broadcastable to ``in_size``.
    tau_syn_in : ArrayLike, default: 2.0 ms
        Inhibitory conductance decay time constant. Must be strictly positive.
        Shape: scalar or broadcastable to ``in_size``.
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
    rng_key : jax.Array, optional
        JAX PRNG key for stochastic spiking. If None, defaults to ``jax.random.PRNGKey(0)``.
    V_initializer : Callable, default: Constant(-70.0 mV)
        Initializer for membrane potential. Must return values compatible with ``in_size``.
    g_ex_initializer : Callable, default: Constant(0.0 nS)
        Initializer for excitatory conductance. Must return values compatible with ``in_size``.
    g_in_initializer : Callable, default: Constant(0.0 nS)
        Initializer for inhibitory conductance. Must return values compatible with ``in_size``.
    spk_fun : Callable, default: ReluGrad()
        Surrogate gradient function for spike generation. Used in gradient-based learning.
    spk_reset : str, default: 'hard'
        Spike reset mode. 'hard' (stop gradient, matches NEST) or 'soft' (subtract threshold).
    name : str, optional
        Module name. If None, auto-generated.

    Parameter Mapping
    -----------------
    Maps brainpy.state parameter names to NEST equivalents for cross-framework compatibility:

    ==================== =================== =================================== ======================================================
    **Parameter**        **Default**         **Math equivalent**                 **Description**
    ==================== =================== =================================== ======================================================
    ``in_size``          (required)                                              Population shape
    ``g_L``              4.0 nS              :math:`g_\mathrm{L}`                Leak conductance
    ``E_L``              -70.0 mV            :math:`E_\mathrm{L}`                Leak reversal potential
    ``C_m``              80.0 pF             :math:`C_\mathrm{m}`                Membrane capacitance
    ``V_reset``          -55.0 mV            :math:`V_\mathrm{reset}`            Reset potential
    ``Delta_V``          0.5 mV              :math:`\Delta_V`                    Stochasticity level
    ``V_T_star``         -35.0 mV            :math:`V_{T^*}`                     Base firing threshold
    ``lambda_0``         1.0 /s              :math:`\lambda_0`                   Stochastic intensity at threshold
    ``t_ref``            4.0 ms              :math:`t_\mathrm{ref}`              Absolute refractory period
    ``E_ex``             0.0 mV              :math:`E_\mathrm{ex}`               Excitatory reversal potential
    ``E_in``             -85.0 mV            :math:`E_\mathrm{in}`               Inhibitory reversal potential
    ``tau_syn_ex``       2.0 ms              :math:`\tau_{\mathrm{syn,ex}}`      Excitatory conductance time constant
    ``tau_syn_in``       2.0 ms              :math:`\tau_{\mathrm{syn,in}}`      Inhibitory conductance time constant
    ``I_e``              0.0 pA              :math:`I_\mathrm{e}`                Constant external current
    ``tau_sfa``          () ms               :math:`\tau_{\gamma_i}`             SFA time constants (tuple/list)
    ``q_sfa``            () mV               :math:`q_{\gamma_i}`                SFA jump values (tuple/list)
    ``tau_stc``          () ms               :math:`\tau_{\eta_i}`               STC time constants (tuple/list)
    ``q_stc``            () nA               :math:`q_{\eta_i}`                  STC jump values (tuple/list)
    ``rng_key``          None                                                    JAX PRNG key for stochastic spiking
    ``V_initializer``    Constant(-70 mV)                                        Initializer for membrane potential
    ``g_ex_initializer`` Constant(0 nS)                                          Initializer for excitatory conductance
    ``g_in_initializer`` Constant(0 nS)                                          Initializer for inhibitory conductance
    ``spk_fun``          ReluGrad()                                              Surrogate spike function
    ``spk_reset``        ``'hard'``                                              Reset mode; hard reset matches NEST
    ==================== =================== =================================== ======================================================

    State Variables
    ---------------
    After ``init_state()``, the following state variables are available:

    ========================== =============== =======================================================
    **State variable**         **Type**        **Description**
    ========================== =============== =======================================================
    ``V``                      HiddenState     Membrane potential :math:`V_\mathrm{m}` (mV)
    ``g_ex``                   HiddenState     Excitatory conductance :math:`g_\mathrm{ex}` (nS)
    ``g_in``                   HiddenState     Inhibitory conductance :math:`g_\mathrm{in}` (nS)
    ``refractory_step_count``  ShortTermState  Remaining refractory grid steps (int32)
    ``integration_step``       ShortTermState  Internal RKF45 step-size state (ms)
    ``I_stim``                 ShortTermState  Buffered current applied in next step (pA)
    ``last_spike_time``        ShortTermState  Last spike time (ms)
    ========================== =============== =======================================================

    Additionally, the following NumPy arrays are maintained internally:

    - ``_stc_elems`` -- shape ``(len(tau_stc), \*in_size)`` -- individual stc elements (nA)
    - ``_sfa_elems`` -- shape ``(len(tau_sfa), \*in_size)`` -- individual sfa elements (mV)
    - ``_stc_val`` -- shape ``in_size`` -- total spike-triggered current (nA)
    - ``_sfa_val`` -- shape ``in_size`` -- adaptive threshold :math:`V_T(t)` (mV)

    Raises
    ------
    ValueError
        If ``C_m <= 0``, ``g_L <= 0``, ``Delta_V <= 0``, ``t_ref < 0``, ``lambda_0 < 0``,
        ``tau_syn_ex <= 0``, ``tau_syn_in <= 0``, any ``tau_sfa <= 0``, any ``tau_stc <= 0``,
        ``len(tau_sfa) != len(q_sfa)``, or ``len(tau_stc) != len(q_stc)``.

    Notes
    -----
    - Defaults follow NEST C++ source for ``gif_cond_exp``.
    - ``lambda_0`` is specified in 1/s (as in NEST's Python interface) and is
      internally converted to 1/ms for computation.
    - Synaptic spike weights are interpreted in conductance units (nS), with
      positive/negative sign selecting excitatory/inhibitory channel.
    - RKF45 integration with adaptive step size ensures numerical stability for stiff systems,
      matching NEST's GSL-based integrator behavior.
    - The stochastic spiking mechanism uses JAX PRNG, which is split each time step to ensure
      reproducible randomness under JIT compilation.

    Examples
    --------
    Create a GIF neuron with default parameters:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
        >>> import brainstate as bs
        >>> bs.environ.context(dt=0.1 * u.ms)
        >>> neuron = bst.gif_cond_exp(in_size=10)
        >>> neuron.init_all_states()
        >>> spikes = neuron.update(x=5.0 * u.pA)

    Create a GIF neuron with spike-frequency adaptation:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
        >>> import brainstate as bs
        >>> bs.environ.context(dt=0.1 * u.ms)
        >>> neuron = bst.gif_cond_exp(
        ...     in_size=10,
        ...     tau_sfa=(100.0, 200.0),  # Two SFA time constants (ms)
        ...     q_sfa=(5.0, 10.0),       # SFA jumps (mV)
        ...     tau_stc=(50.0,),         # One STC time constant (ms)
        ...     q_stc=(100.0,),          # STC jump (nA)
        ... )
        >>> neuron.init_all_states()
        >>> spikes = neuron.update(x=50.0 * u.pA)

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
    .. [3] NEST Simulator ``gif_cond_exp`` model documentation and C++ source:
           ``models/gif_cond_exp.h`` and ``models/gif_cond_exp.cpp``.

    See Also
    --------
    gif_psc_exp, gif_cond_exp_multisynapse, iaf_cond_exp
    """
    __module__ = 'brainpy.state'

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
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
        E_ex: ArrayLike = 0.0 * u.mV,
        E_in: ArrayLike = -85.0 * u.mV,
        tau_syn_ex: ArrayLike = 2.0 * u.ms,
        tau_syn_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0.0 * u.pA,
        tau_sfa: Sequence[float] = (),  # ms values
        q_sfa: Sequence[float] = (),  # mV values
        tau_stc: Sequence[float] = (),  # ms values
        q_stc: Sequence[float] = (),  # nA values
        rng_key: Optional[jax.Array] = None,
        V_initializer: Callable = braintools.init.Constant(-70.0 * u.mV),
        g_ex_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        g_in_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
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

        # Synaptic parameters
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)

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

        # Initializers
        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.g_L, u.nS) <= 0.0):
            raise ValueError('Membrane conductance must be strictly positive.')
        if np.any(self._to_numpy(self.Delta_V, u.mV) <= 0.0):
            raise ValueError('Delta_V must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')
        if self.lambda_0 < 0.0:
            raise ValueError('lambda_0 must not be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or \
           np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('Synapse time constants must be strictly positive.')
        for tau in self.tau_sfa:
            if tau <= 0.0:
                raise ValueError('All SFA time constants must be strictly positive.')
        for tau in self.tau_stc:
            if tau <= 0.0:
                raise ValueError('All STC time constants must be strictly positive.')

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables for the GIF neuron.

        Initializes membrane potential (``V``), conductances (``g_ex``, ``g_in``),
        adaptation elements (``_stc_elems``, ``_sfa_elems``), refractory counter,
        integration step size, buffered current, and RNG state.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for vectorized simulation. If None, creates single-neuron states.
        **kwargs
            Additional keyword arguments (unused, for API compatibility).

        Notes
        -----
        - Sets ``V`` using ``V_initializer`` (default: -70 mV).
        - Sets ``g_ex`` and ``g_in`` using respective initializers (default: 0 nS).
        - Initializes all STC and SFA elements to zero.
        - Sets ``refractory_step_count`` to 0 (not refractory).
        - Sets ``integration_step`` to simulation timestep (from ``brainstate.environ.get_dt()``).
        - Initializes RNG state from ``rng_key`` if provided, else uses ``jax.random.PRNGKey(0)``.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)

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

        # Adaptation state: stc and sfa element arrays (unitless floats in nA and mV respectively)
        # Stored as plain numpy arrays, one per neuron element
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        # We store adaptation elements externally as numpy arrays since they are
        # variable-length lists in NEST. Shape: (n_elements, *varshape) or (n_elements, batch, *varshape)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)  # total stc current (nA)
        self._sfa_val = np.full(v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=np.float64)  # adaptive threshold (mV)

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to initial values.

        Resets membrane potential, conductances, adaptation elements, refractory counter,
        integration step size, buffered current, and RNG state to their initialized values.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for vectorized simulation. If None, resets to single-neuron states.
        **kwargs
            Additional keyword arguments (unused, for API compatibility).

        Notes
        -----
        - This method is equivalent to ``init_state()`` but called on an already-initialized module.
        - Resets all STC and SFA elements to zero.
        - Resets ``last_spike_time`` to a very negative value (-1e7 ms).
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.g_ex.value = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        self.g_in.value = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
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

        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)
        self._sfa_val = np.full(v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=np.float64)

        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike signal using surrogate gradient.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (mV). If None, uses current ``self.V.value``.

        Returns
        -------
        ArrayLike
            Differentiable spike signal in [0, 1], computed via surrogate function.
            Shape matches ``V`` or ``self.V.value``.

        Notes
        -----
        - This method is used for gradient-based learning, not for actual spike generation
          in forward simulation (which is stochastic via ``update()``).
        - Spike signal is computed as ``spk_fun((V - V_reset) / Delta_V)``.
        - Default ``spk_fun`` is ``ReluGrad()``, providing a piecewise-linear surrogate.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_reset) / (self.Delta_V)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _sum_signed_delta_inputs(self):
        g_ex = u.math.zeros_like(self.g_ex.value)
        g_in = u.math.zeros_like(self.g_in.value)
        if self.delta_inputs is None:
            return g_ex, g_in

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            zero = u.math.zeros_like(out)
            g_ex = g_ex + u.math.maximum(out, zero)
            g_in = g_in + u.math.maximum(-out, zero)
        return g_ex, g_in

    @staticmethod
    def _dynamics_scalar(v, g_ex, g_in, is_refractory, i_stim, stc, p):
        r"""Compute derivatives for the ODE system [V_m, g_ex, g_in].

        This matches the NEST gif_cond_exp_dynamics() function exactly.
        During refractory period, V is clamped to V_reset and dV/dt=0.

        Parameters
        ----------
        v : float
            Membrane potential (mV).
        g_ex : float
            Excitatory conductance (nS).
        g_in : float
            Inhibitory conductance (nS).
        is_refractory : bool
            Whether the neuron is in refractory period.
        i_stim : float
            External current input (pA).
        stc : float
            Total spike-triggered current (pA, computed from sum of stc elements).
        p : dict
            Parameter dictionary containing 'V_reset', 'E_L', 'E_ex', 'E_in',
            'C_m', 'g_L', 'tau_syn_ex', 'tau_syn_in', 'I_e'.

        Returns
        -------
        tuple of (float, float, float)
            Derivatives (dV/dt, dg_ex/dt, dg_in/dt) in (mV/ms, nS/ms, nS/ms).
        """
        V = p['V_reset'] if is_refractory else v

        I_syn_exc = g_ex * (V - p['E_ex'])
        I_syn_inh = g_in * (V - p['E_in'])
        I_L = p['g_L'] * (V - p['E_L'])

        dv = 0.0 if is_refractory else (-I_L + i_stim + p['I_e'] - I_syn_exc - I_syn_inh - stc) / p['C_m']
        dg_ex = -g_ex / p['tau_syn_ex']
        dg_in = -g_in / p['tau_syn_in']
        return dv, dg_ex, dg_in

    def _rkf45_integrate_scalar(self, v0, ge0, gi0, is_refractory, i_stim, stc, h0, dt, p):
        r"""Adaptive RKF45 integration matching GSL gsl_odeiv_evolve_apply behavior.

        Integrates the ODE system [V, g_ex, g_in] over the interval [0, dt] using
        a Runge-Kutta-Fehlberg 4(5) method with adaptive step size control.

        Parameters
        ----------
        v0 : float
            Initial membrane potential (mV).
        ge0 : float
            Initial excitatory conductance (nS).
        gi0 : float
            Initial inhibitory conductance (nS).
        is_refractory : bool
            Whether the neuron is in refractory period.
        i_stim : float
            External current input (pA).
        stc : float
            Total spike-triggered current (pA).
        h0 : float
            Initial internal step size (ms).
        dt : float
            Simulation timestep to integrate over (ms).
        p : dict
            Parameter dictionary containing 'V_reset', 'E_L', 'E_ex', 'E_in',
            'C_m', 'g_L', 'tau_syn_ex', 'tau_syn_in', 'I_e'.

        Returns
        -------
        tuple of (float, float, float, float)
            Final (V, g_ex, g_in, h_new) after integration.
            ``h_new`` is the updated internal step size for next integration.

        Notes
        -----
        - Uses RKF45 embedded pair with error estimation and step size adaptation.
        - Matches NEST's GSL integrator with absolute tolerance ``_ATOL = 1e-3``.
        - Minimum step size is ``_MIN_H = 1e-8`` ms.
        - Maximum iterations per timestep is ``_MAX_ITERS = 10000``.
        - Step size is adjusted based on local error estimate using 0.9 safety factor.
        """
        t = 0.0
        h = max(h0, self._MIN_H)
        v, ge, gi = v0, ge0, gi0
        iters = 0

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = min(h, dt - t)
            h = max(h, self._MIN_H)

            def f(y1, y2, y3):
                return self._dynamics_scalar(y1, y2, y3, is_refractory, i_stim, stc, p)

            k1 = f(v, ge, gi)
            y2 = (v + h * k1[0] / 4.0, ge + h * k1[1] / 4.0, gi + h * k1[2] / 4.0)
            k2 = f(*y2)
            y3 = (
                v + h * (3.0 * k1[0] / 32.0 + 9.0 * k2[0] / 32.0),
                ge + h * (3.0 * k1[1] / 32.0 + 9.0 * k2[1] / 32.0),
                gi + h * (3.0 * k1[2] / 32.0 + 9.0 * k2[2] / 32.0),
            )
            k3 = f(*y3)
            y4 = (
                v + h * (1932.0 * k1[0] / 2197.0 - 7200.0 * k2[0] / 2197.0 + 7296.0 * k3[0] / 2197.0),
                ge + h * (1932.0 * k1[1] / 2197.0 - 7200.0 * k2[1] / 2197.0 + 7296.0 * k3[1] / 2197.0),
                gi + h * (1932.0 * k1[2] / 2197.0 - 7200.0 * k2[2] / 2197.0 + 7296.0 * k3[2] / 2197.0),
            )
            k4 = f(*y4)
            y5 = (
                v + h * (439.0 * k1[0] / 216.0 - 8.0 * k2[0] + 3680.0 * k3[0] / 513.0 - 845.0 * k4[0] / 4104.0),
                ge + h * (439.0 * k1[1] / 216.0 - 8.0 * k2[1] + 3680.0 * k3[1] / 513.0 - 845.0 * k4[1] / 4104.0),
                gi + h * (439.0 * k1[2] / 216.0 - 8.0 * k2[2] + 3680.0 * k3[2] / 513.0 - 845.0 * k4[2] / 4104.0),
            )
            k5 = f(*y5)
            y6 = (
                v + h * (-8.0 * k1[0] / 27.0 + 2.0 * k2[0] - 3544.0 * k3[0] / 2565.0 + 1859.0 * k4[0] / 4104.0 - 11.0 * k5[0] / 40.0),
                ge + h * (-8.0 * k1[1] / 27.0 + 2.0 * k2[1] - 3544.0 * k3[1] / 2565.0 + 1859.0 * k4[1] / 4104.0 - 11.0 * k5[1] / 40.0),
                gi + h * (-8.0 * k1[2] / 27.0 + 2.0 * k2[2] - 3544.0 * k3[2] / 2565.0 + 1859.0 * k4[2] / 4104.0 - 11.0 * k5[2] / 40.0),
            )
            k6 = f(*y6)

            y4_sol = (
                v + h * (25.0 * k1[0] / 216.0 + 1408.0 * k3[0] / 2565.0 + 2197.0 * k4[0] / 4104.0 - k5[0] / 5.0),
                ge + h * (25.0 * k1[1] / 216.0 + 1408.0 * k3[1] / 2565.0 + 2197.0 * k4[1] / 4104.0 - k5[1] / 5.0),
                gi + h * (25.0 * k1[2] / 216.0 + 1408.0 * k3[2] / 2565.0 + 2197.0 * k4[2] / 4104.0 - k5[2] / 5.0),
            )
            y5_sol = (
                v + h * (16.0 * k1[0] / 135.0 + 6656.0 * k3[0] / 12825.0 + 28561.0 * k4[0] / 56430.0 - 9.0 * k5[0] / 50.0 + 2.0 * k6[0] / 55.0),
                ge + h * (16.0 * k1[1] / 135.0 + 6656.0 * k3[1] / 12825.0 + 28561.0 * k4[1] / 56430.0 - 9.0 * k5[1] / 50.0 + 2.0 * k6[1] / 55.0),
                gi + h * (16.0 * k1[2] / 135.0 + 6656.0 * k3[2] / 12825.0 + 28561.0 * k4[2] / 56430.0 - 9.0 * k5[2] / 50.0 + 2.0 * k6[2] / 55.0),
            )

            err = max(
                abs(y5_sol[0] - y4_sol[0]),
                abs(y5_sol[1] - y4_sol[1]),
                abs(y5_sol[2] - y4_sol[2]),
            )

            if err <= self._ATOL or h <= self._MIN_H:
                v, ge, gi = y5_sol
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

        return v, ge, gi, h

    def update(self, x=0.0 * u.pA):
        r"""Advance the neuron state by one simulation timestep.

        Performs the complete GIF update cycle: decay adaptation elements, integrate
        membrane dynamics via RKF45, add synaptic inputs, evaluate stochastic spike
        condition, handle refractory period, and update all state variables.

        Parameters
        ----------
        x : ArrayLike, default: 0.0 pA
            External current input for this timestep (pA). Shape must be broadcastable to ``in_size``.

        Returns
        -------
        ArrayLike
            Binary spike indicator (0 or 1) for each neuron. Shape matches ``in_size``.
            Value is 1.0 if neuron spiked, 0.0 otherwise.

        Notes
        -----
        **Update order:**

        1. Compute total stc and sfa from element arrays, then decay all elements.
        2. Integrate continuous dynamics [V, g_ex, g_in] using adaptive RKF45.
        3. Add synaptic conductance jumps from ``delta_inputs``.
        4. Evaluate stochastic spike condition (if not refractory):
           - Compute firing intensity: :math:`\\lambda = \\lambda_0 \\exp((V - V_T) / \\Delta_V)`
           - Draw random number, spike if :math:`U < 1 - \\exp(-\\lambda \\cdot dt)`
           - On spike: increment stc/sfa elements, set refractory counter
        5. If refractory: decrement counter, clamp V to V_reset.
        6. Buffer current input for next step.

        **Synaptic input handling:**

        - Conductance inputs are accumulated from ``delta_inputs`` dict.
        - Positive weights → excitatory (``g_ex``), negative weights → inhibitory (``g_in``).
        - Current inputs are summed via ``sum_current_inputs()`` and buffered for next step.

        **Stochastic spiking:**

        - RNG state is advanced each timestep via ``jax.random.split()``.
        - Spike times are not exact (unlike ``*_ps`` models) — spikes occur on grid.
        - For reproducibility, set ``rng_key`` explicitly during initialization.

        **Failure modes:**

        - If RKF45 cannot converge within ``_MAX_ITERS`` iterations, integration may be
          incomplete. This typically occurs only with extreme parameter values or very large dt.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape).copy()
        ge = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape).copy()
        gi = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape).copy()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        ).copy()
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape).copy()
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape).copy()

        p = {
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'E_ex': self._broadcast_to_state(self._to_numpy(self.E_ex, u.mV), v_shape),
            'E_in': self._broadcast_to_state(self._to_numpy(self.E_in, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'tau_syn_ex': self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape),
            'tau_syn_in': self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
        }
        V_T_star = float(self._to_numpy(self.V_T_star, u.mV))
        Delta_V = float(self._to_numpy(self.Delta_V, u.mV))
        lambda_0 = self.lambda_0  # 1/ms
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )

        # Compute exponential decay factors for adaptation
        P_stc = [math.exp(-dt / tau) for tau in self.tau_stc]
        P_sfa = [math.exp(-dt / tau) for tau in self.tau_sfa]

        # Get synaptic inputs
        dg_ex_q, dg_in_q = self._sum_signed_delta_inputs()
        dg_ex = self._broadcast_to_state(self._to_numpy(dg_ex_q, u.nS), v_shape)
        dg_in = self._broadcast_to_state(self._to_numpy(dg_in_q, u.nS), v_shape)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        # Advance RNG state for this step - generate uniform random numbers for all neurons
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(jax.random.uniform(subkey, shape=v_shape), dtype=np.float64)

        v_for_spike = np.empty_like(V)
        spike_mask = np.zeros_like(V, dtype=bool)
        V_next = np.empty_like(V)
        ge_next = np.empty_like(ge)
        gi_next = np.empty_like(gi)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}

            # ---- Step 1: Decay stc/sfa elements and compute totals ----
            # This matches the NEST update order: stc/sfa computed BEFORE ODE integration
            stc_total = 0.0
            if self._stc_elems is not None:
                for i in range(len(self.tau_stc)):
                    stc_total += self._stc_elems[i][idx]
                    self._stc_elems[i][idx] *= P_stc[i]

            sfa_total = V_T_star
            if self._sfa_elems is not None:
                for i in range(len(self.tau_sfa)):
                    sfa_total += self._sfa_elems[i][idx]
                    self._sfa_elems[i][idx] *= P_sfa[i]

            self._stc_val[idx] = stc_total
            self._sfa_val[idx] = sfa_total

            # ---- Step 2: Integrate ODE [V, g_ex, g_in] ----
            is_refractory = r[idx] > 0
            v_i, ge_i, gi_i, h_i = self._rkf45_integrate_scalar(
                V[idx], ge[idx], gi[idx], is_refractory, i_stim[idx], stc_total,
                h_int[idx], dt, local_p
            )

            # ---- Step 3: Add synaptic conductance jumps ----
            ge_i += dg_ex[idx]
            gi_i += dg_in[idx]

            # ---- Step 4: Refractory / spike check ----
            if r[idx] == 0:
                # Neuron is not refractory
                lam = lambda_0 * math.exp((v_i - sfa_total) / Delta_V)
                if lam > 0.0:
                    # Hazard function: P(spike) = 1 - exp(-lambda * dt)
                    # NEST uses: rng->drand() < -expm1(-lambda * dt)
                    spike_prob = -math.expm1(-lam * dt)
                    if rand_vals[idx] < spike_prob:
                        # Spike!
                        spike_mask[idx] = True

                        # Jump stc elements (after decay has already been applied)
                        if self._stc_elems is not None:
                            for i in range(len(self.q_stc)):
                                self._stc_elems[i][idx] += self.q_stc[i]

                        # Jump sfa elements
                        if self._sfa_elems is not None:
                            for i in range(len(self.q_sfa)):
                                self._sfa_elems[i][idx] += self.q_sfa[i]

                        r_i = refr_counts[idx]
                    else:
                        r_i = 0
                else:
                    r_i = 0

                v_for_spike[idx] = v_i
            else:
                # Neuron is refractory
                r_i = r[idx] - 1
                v_i = local_p['V_reset']
                v_for_spike[idx] = local_p['V_reset']

            V_next[idx] = v_i
            ge_next[idx] = ge_i
            gi_next[idx] = gi_i
            r_next[idx] = r_i
            h_next[idx] = h_i

        # ---- Step 5: Store new I_stim for next step ----
        self.V.value = V_next * u.mV
        self.g_ex.value = ge_next * u.nS
        self.g_in.value = gi_next * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)
