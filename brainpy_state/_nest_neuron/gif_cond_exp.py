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

from collections.abc import Callable, Sequence

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'gif_cond_exp',
]


class gif_cond_exp(NESTNeuron):
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
    gsl_error_tol : ArrayLike, default: 1e-6
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
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
    ref_var : bool, default: False
        If ``True``, allocate and expose ``self.refractory`` state.
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
    ``gsl_error_tol``    1e-6                --                                  RKF45 absolute error tolerance
    ``rng_key``          None                                                    JAX PRNG key for stochastic spiking
    ``V_initializer``    Constant(-70 mV)                                        Initializer for membrane potential
    ``g_ex_initializer`` Constant(0 nS)                                          Initializer for excitatory conductance
    ``g_in_initializer`` Constant(0 nS)                                          Initializer for inhibitory conductance
    ``spk_fun``          ReluGrad()                                              Surrogate spike function
    ``spk_reset``        ``'hard'``                                              Reset mode; hard reset matches NEST
    ``ref_var``          ``False``                                               If True, expose boolean refractory state
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
    ``refractory``             ShortTermState  Optional boolean refractory indicator (ref_var=True)
    ========================== =============== =======================================================

    Additionally, the following NumPy arrays are maintained internally:

    - ``_stc_elems`` -- shape ``(len(tau_stc), *in_size)`` -- individual stc elements (nA)
    - ``_sfa_elems`` -- shape ``(len(tau_sfa), *in_size)`` -- individual sfa elements (mV)
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

        >>> from brainpy import state as bst
        >>> import brainunit as u
        >>> import brainstate as bs
        >>> bs.environ.context(dt=0.1 * u.ms)
        >>> neuron = bst.gif_cond_exp(in_size=10)
        >>> neuron.init_all_states()
        >>> spikes = neuron.update(x=5.0 * u.pA)

    Create a GIF neuron with spike-frequency adaptation:

    .. code-block:: python

        >>> from brainpy import state as bst
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

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    #: Unit the multi-receptor ``connect(receptor_type=k)`` bridge uses to scale the
    #: gathered per-port deposit into the ``w_by_rec`` mantissa (conductance -> nS).
    receptor_input_unit = u.nS

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
        gsl_error_tol: ArrayLike = 1e-6,
        rng_key: jax.Array | None = None,
        V_initializer: Callable = braintools.init.Constant(-70.0 * u.mV),
        g_ex_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        g_in_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str | None = None,
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
        self.gsl_error_tol = gsl_error_tol
        self.ref_var = ref_var

        # Two synaptic receptors (g_ex / g_in) addressable by ``connect(receptor_type=k)``:
        # the Simulator's multi-receptor bridge gathers per-port conductance jumps into a
        # ``(*varshape, n_receptors)`` blob and passes it to ``update(w_by_rec=...)``;
        # column 0 -> g_ex (receptor 1), column 1 -> g_in (receptor 2).
        self.n_receptors = 2

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

    def _validate_parameters(self):
        r"""Validate model parameters against NEST constraints.

        Raises
        ------
        ValueError
            If parameter inequalities or positivity constraints are violated.
        """
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
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms) or \
            cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('Synapse time constants must be strictly positive.')
        for tau in self.tau_sfa:
            if tau <= 0.0:
                raise ValueError('All SFA time constants must be strictly positive.')
        for tau in self.tau_stc:
            if tau <= 0.0:
                raise ValueError('All STC time constants must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables for the GIF neuron.

        Initializes membrane potential (``V``), conductances (``g_ex``, ``g_in``),
        adaptation elements (``_stc_elems``, ``_sfa_elems``), refractory counter,
        integration step size, buffered current, and RNG state.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        - Sets ``V`` using ``V_initializer`` (default: -70 mV).
        - Sets ``g_ex`` and ``g_in`` using respective initializers (default: 0 nS).
        - Initializes all STC and SFA elements to zero.
        - Sets ``refractory_step_count`` to 0 (not refractory).
        - Sets ``integration_step`` to simulation timestep (from ``brainstate.environ.get_dt()``).
        - Initializes RNG state from ``rng_key`` if provided, else uses ``jax.random.PRNGKey(0)``.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape)

        self.V = brainstate.HiddenState(V)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA))

        # Adaptation state: JAX arrays wrapped in ShortTermState for JIT compatibility.
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape
        V_T_star_mV = float(np.asarray(u.get_mantissa(self.V_T_star / u.mV)))

        self._stc_elems_state = (
            brainstate.ShortTermState(jnp.zeros((n_stc, *v_shape), dtype=jnp.float64))
            if n_stc > 0 else None
        )
        self._sfa_elems_state = (
            brainstate.ShortTermState(jnp.zeros((n_sfa, *v_shape), dtype=jnp.float64))
            if n_sfa > 0 else None
        )
        self._stc_val_state = brainstate.ShortTermState(
            jnp.zeros(v_shape, dtype=jnp.float64)
        )
        self._sfa_val_state = brainstate.ShortTermState(
            jnp.full(v_shape, V_T_star_mV, dtype=jnp.float64)
        )

        # RNG state as ShortTermState for JIT compatibility.
        rng_init = self._rng_key if self._rng_key is not None else jax.random.PRNGKey(0)
        self._rng_state = brainstate.ShortTermState(rng_init)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

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

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, g_ex, g_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, stc_total -- mutable
            auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, state.V)

        i_syn_exc = state.g_ex * (v_eff - self.E_ex)
        i_syn_inh = state.g_in * (v_eff - self.E_in)
        i_leak = self.g_L * (v_eff - self.E_L)

        dV_raw = (
                     -i_leak - i_syn_exc - i_syn_inh
                     - extra.stc_total + self.I_e + extra.i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        dg_ex = -state.g_ex / self.tau_syn_ex
        dg_in = -state.g_in / self.tau_syn_in

        return DotDict(V=dV, g_ex=dg_ex, g_in=dg_in)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, g_ex, g_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, stc_total, sfa_total,
            lambda_0, Delta_V, rand_vals, dt_ms.
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
        new_V = u.math.where(refr_accept, self.V_reset, state.V)

        # For GIF: stochastic spike check when not refractory
        # Compute firing intensity: lambda = lambda_0 * exp((V - V_T) / Delta_V)
        v_mantissa = u.get_mantissa(new_V / u.mV)
        sfa_mantissa = u.get_mantissa(extra.sfa_total)
        delta_v_mantissa = u.get_mantissa(self.Delta_V / u.mV)
        lam = extra.lambda_0 * jnp.exp(
            jnp.clip((v_mantissa - sfa_mantissa) / delta_v_mantissa, -500.0, 500.0)
        )
        # Hazard function: P(spike) = 1 - exp(-lambda * dt)
        spike_prob = -jnp.expm1(-lam * extra.dt_ms)
        stochastic_spike = accept & (extra.r <= 0) & (extra.rand_vals < spike_prob)

        spike_mask = extra.spike_mask | stochastic_spike
        new_V = u.math.where(stochastic_spike, self.V_reset, new_V)
        r = u.math.where(stochastic_spike & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict({**state, 'V': new_V})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

    def update(self, x=0.0 * u.pA, w_by_rec=None):
        r"""Advance the neuron state by one simulation timestep.

        Performs the complete GIF update cycle: decay adaptation elements, integrate
        membrane dynamics via RKF45, add synaptic inputs, evaluate stochastic spike
        condition, handle refractory period, and update all state variables.

        Parameters
        ----------
        x : ArrayLike, default: 0.0 pA
            External current input for this timestep (pA). Shape must be broadcastable to ``in_size``.
        w_by_rec : ArrayLike or None, optional
            Per-receptor conductance jumps gathered by the Simulator's multi-receptor
            bridge, shape ``(*varshape, n_receptors)`` in nanosiemens -- column 0 is the
            excitatory port (``g_ex``), column 1 the inhibitory port (``g_in``). ``None``
            (the default) selects the legacy self-pull path (``label='w_ex'/'w_in'`` delta
            inputs), so direct calls and the BrainPy-style API are unchanged.

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
        - Positive weights -> excitatory (``g_ex``), negative weights -> inhibitory (``g_in``).
        - Current inputs are summed via ``sum_current_inputs()`` and buffered for next step.

        **Stochastic spiking:**

        - RNG state is advanced each timestep via ``jax.random.split()``.
        - Spike times are not exact (unlike ``*_ps`` models) -- spikes occur on grid.
        - For reproducibility, set ``rng_key`` explicitly during initialization.

        **Failure modes:**

        - If RKF45 cannot converge within ``_MAX_ITERS`` iterations, integration may be
          incomplete. This typically occurs only with extreme parameter values or very large dt.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        dt_ms = float(u.get_mantissa(dt / u.ms))

        # Read state variables with their natural units.
        V = self.V.value  # mV
        g_ex = self.g_ex.value  # nS
        g_in = self.g_in.value  # nS
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        v_shape = self.V.value.shape
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        n_dims = len(v_shape)

        # ---- Step 1: Compute stc/sfa totals and exponential decay ----
        if n_stc > 0:
            stc_elems = self._stc_elems_state.value              # (n_stc, *v_shape), float64
            stc_total = jnp.sum(stc_elems, axis=0)              # (*v_shape)
            P_stc_arr = jnp.array(
                [np.exp(-dt_ms / tau) for tau in self.tau_stc], dtype=jnp.float64
            ).reshape(n_stc, *([1] * n_dims))
            stc_elems_decayed = stc_elems * P_stc_arr
        else:
            stc_total = jnp.zeros(v_shape, dtype=jnp.float64)
            stc_elems_decayed = None

        V_T_star_mV = float(np.asarray(u.get_mantissa(self.V_T_star / u.mV)))
        if n_sfa > 0:
            sfa_elems = self._sfa_elems_state.value              # (n_sfa, *v_shape), float64
            sfa_total = V_T_star_mV + jnp.sum(sfa_elems, axis=0)  # (*v_shape)
            P_sfa_arr = jnp.array(
                [np.exp(-dt_ms / tau) for tau in self.tau_sfa], dtype=jnp.float64
            ).reshape(n_sfa, *([1] * n_dims))
            sfa_elems_decayed = sfa_elems * P_sfa_arr
        else:
            sfa_total = jnp.full(v_shape, V_T_star_mV, dtype=jnp.float64)
            sfa_elems_decayed = None

        self._stc_val_state.value = stc_total
        self._sfa_val_state.value = sfa_total

        # Convert stc_total to physical units for the ODE (cast to dftype to preserve state dtype)
        stc_total_pA = stc_total.astype(dftype) * u.nA

        # Advance RNG state
        new_rng, subkey = jax.random.split(self._rng_state.value)
        self._rng_state.value = new_rng
        rand_vals = jax.random.uniform(subkey, shape=v_shape)

        # ---- Step 2: Adaptive RKF45 integration via generic integrator ----
        ode_state = DotDict(V=V, g_ex=g_ex, g_in=g_in)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            stc_total=stc_total_pA,
            sfa_total=sfa_total,
            lambda_0=self.lambda_0,
            Delta_V=float(np.asarray(u.get_mantissa(self.Delta_V / u.mV))),
            rand_vals=rand_vals,
            dt_ms=dt_ms,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, g_ex, g_in = ode_state.V, ode_state.g_ex, ode_state.g_in
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in gif_cond_exp dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # ---- Step 3: Synaptic spike inputs (applied after integration) ----
        # Two delivery paths: the Simulator's multi-receptor bridge pre-gathers per-receptor
        # conductance jumps (nS mantissa, shape ``(*varshape, n_receptors)``) and passes them
        # via ``w_by_rec`` -- column 0 -> g_ex, column 1 -> g_in; the legacy BrainPy-style path
        # self-pulls the ``label='w_ex'/'w_in'`` delta inputs. Only the *source* of
        # ``w_ex``/``w_in`` changes; the exponential kinetics below consume them unchanged.
        if w_by_rec is not None:
            w_by_rec = jnp.asarray(w_by_rec, dtype=dftype)
            w_ex = w_by_rec[..., 0] * u.nS
            w_in = w_by_rec[..., 1] * u.nS
        else:
            w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
            w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')

        # Apply synaptic spike inputs.
        g_ex = g_ex + w_ex
        g_in = g_in + w_in

        # ---- Step 4: Update stc/sfa elements on spike (JAX-native, JIT-compatible) ----
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

        # ---- Step 5: Write back state ----
        self.V.value = V
        self.g_ex.value = g_ex
        self.g_in.value = g_in
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
