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
from typing import Callable, Optional, Sequence

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'gif_cond_exp_multisynapse',
]


class gif_cond_exp_multisynapse(Neuron):
    r"""Conductance-based generalized integrate-and-fire neuron (GIF) model
    with multiple synaptic time constants.

    Description
    -----------

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

    Synaptic conductances
    .....................

    Each synaptic conductance decays exponentially:

    .. math::

       \frac{dg_k}{dt} = -\frac{g_k}{\tau_{\mathrm{syn},k}}

    On the postsynaptic side, there can be arbitrarily many synaptic time
    constants. This is achieved by specifying separate receptor ports, each
    for a different time constant. The number of receptor ports is determined
    by the length of the ``tau_syn`` and ``E_rev`` lists, which must have
    equal length.

    Spike-triggered currents
    ........................

    Dynamic of each :math:`\eta_i` is described by:

    .. math::

       \tau_{\eta_i} \cdot \frac{d\eta_i}{dt} = -\eta_i

    and in case of spike emission, its value is increased by a constant:

    .. math::

       \eta_i = \eta_i + q_{\eta_i} \quad \text{(on spike emission)}

    Spike-frequency adaptation
    ..........................

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

    Stochastic spiking
    ..................

    The probability of firing within a time step :math:`dt` is computed using
    the hazard function:

    .. math::

       P(\text{spike}) = 1 - \exp(-\lambda(t) \cdot dt)

    A random number is drawn each (non-refractory) time step and compared to
    this probability to determine whether a spike occurs.

    Refractory mechanism
    ....................

    After a spike, the neuron enters an absolute refractory period of duration
    :math:`t_\mathrm{ref}`. During this period:

    * :math:`V_\mathrm{m}` is clamped to :math:`V_\mathrm{reset}`,
    * :math:`dV_\mathrm{m}/dt = 0`,
    * conductances continue to decay,
    * refractory counter decrements each step.

    Numerical integration and update order
    ......................................

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

    Multisynapse differences from gif_cond_exp
    ..........................................

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

    ==================== =================== =================================== =====================================================
    **Parameter**        **Default**         **Math equivalent**                 **Description**
    ==================== =================== =================================== =====================================================
    ``in_size``          (required)                                              Population shape
    ``g_L``              4.0 nS              :math:`g_\mathrm{L}`               Leak conductance
    ``E_L``              -70.0 mV            :math:`E_\mathrm{L}`               Leak reversal potential
    ``C_m``              80.0 pF             :math:`C_\mathrm{m}`               Membrane capacitance
    ``V_reset``          -55.0 mV            :math:`V_\mathrm{reset}`           Reset potential
    ``Delta_V``          0.5 mV              :math:`\Delta_V`                   Stochasticity level
    ``V_T_star``         -35.0 mV            :math:`V_{T^*}`                    Base firing threshold
    ``lambda_0``         1.0 /s              :math:`\lambda_0`                  Stochastic intensity at threshold
    ``t_ref``            4.0 ms              :math:`t_\mathrm{ref}`             Absolute refractory period
    ``tau_syn``          (2.0,) ms           :math:`\tau_{\mathrm{syn},k}`      Synaptic conductance time constants (list)
    ``E_rev``            (0.0,) mV           :math:`E_{\mathrm{rev},k}`         Reversal potentials (list, same size as tau_syn)
    ``I_e``              0.0 pA              :math:`I_\mathrm{e}`               Constant external current
    ``tau_sfa``          () ms               :math:`\tau_{\gamma_i}`            SFA time constants (tuple/list)
    ``q_sfa``            () mV               :math:`q_{\gamma_i}`              SFA jump values (tuple/list)
    ``tau_stc``          () ms               :math:`\tau_{\eta_i}`              STC time constants (tuple/list)
    ``q_stc``            () nA               :math:`q_{\eta_i}`                STC jump values (tuple/list)
    ``rng_key``          None                                                    JAX PRNG key for stochastic spiking
    ``V_initializer``    Constant(-70 mV)                                        Initializer for membrane potential
    ``spk_fun``          ReluGrad()                                              Surrogate spike function
    ``spk_reset``        ``'hard'``                                              Reset mode; hard reset matches NEST
    ==================== =================== =================================== =====================================================

    State Variables
    ---------------

    ========================== ===========================================
    **State variable**         **Description**
    ========================== ===========================================
    ``V``                      Membrane potential :math:`V_\mathrm{m}`
    ``g``                      List of synaptic conductances :math:`g_k`
    ``stc``                    Total spike-triggered current
    ``sfa``                    Adaptive threshold :math:`V_T(t)`
    ``stc_elems``              Individual stc adaptation elements
    ``sfa_elems``              Individual sfa adaptation elements
    ``refractory_step_count``  Remaining refractory grid steps
    ``integration_step``       Internal RKF45 step-size state
    ``I_stim``                 Buffered current applied in next step
    ``last_spike_time``        Last spike time
    ========================== ===========================================

    Notes
    -----

    - Defaults follow NEST C++ source for ``gif_cond_exp_multisynapse``.
    - ``lambda_0`` is specified in 1/s (as in NEST's Python interface) and is
      internally converted to 1/ms for computation.
    - All synaptic spike weights must be non-negative (conductance-based model).
    - Delta inputs for synaptic conductances are indexed by receptor port
      (0-based). Use ``add_delta_input(f'receptor_{port}', weight * u.nS)``
      to add a conductance jump to a specific receptor port.

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
    gif_cond_exp
    iaf_cond_exp
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
        tau_syn: Sequence[float] = (2.0,),  # ms values
        E_rev: Sequence[float] = (0.0,),  # mV values
        I_e: ArrayLike = 0.0 * u.pA,
        tau_sfa: Sequence[float] = (),  # ms values
        q_sfa: Sequence[float] = (),  # mV values
        tau_stc: Sequence[float] = (),  # ms values
        q_stc: Sequence[float] = (),  # nA values
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

    @property
    def n_receptors(self):
        """Number of synaptic receptor ports."""
        return self._n_receptors

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
        for i, tau in enumerate(self.tau_syn):
            if tau <= 0.0:
                raise ValueError(f'All synaptic time constants must be strictly positive (tau_syn[{i}]={tau}).')
        for i, tau in enumerate(self.tau_sfa):
            if tau <= 0.0:
                raise ValueError(f'All SFA time constants must be strictly positive (tau_sfa[{i}]={tau}).')
        for i, tau in enumerate(self.tau_stc):
            if tau <= 0.0:
                raise ValueError(f'All STC time constants must be strictly positive (tau_stc[{i}]={tau}).')

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.V = brainstate.HiddenState(V)

        # Initialize per-receptor conductance states
        self.g = [
            brainstate.HiddenState(
                braintools.init.param(
                    braintools.init.Constant(0.0 * u.nS),
                    self.varshape, batch_size
                )
            )
            for _ in range(self._n_receptors)
        ]

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

        # Adaptation state
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)
        self._sfa_val = np.full(v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=np.float64)

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        for i in range(self._n_receptors):
            self.g[i].value = braintools.init.param(
                braintools.init.Constant(0.0 * u.nS), self.varshape, batch_size
            )
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
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_reset) / (self.Delta_V)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _collect_receptor_delta_inputs(self):
        """Collect delta inputs per receptor port.

        Delta inputs for receptor port k should be registered with key
        containing 'receptor_<k>' (0-based). Any input not matching a specific
        receptor pattern is ignored (use add_current_input for non-synaptic
        currents).

        Returns a list of arrays, one per receptor, in nS (numpy float64).
        """
        v_shape = self.V.value.shape
        dg = [np.zeros(v_shape, dtype=np.float64) for _ in range(self._n_receptors)]

        if self.delta_inputs is None:
            return dg

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            # Determine which receptor this input belongs to
            out_nS = self._to_numpy(out, u.nS)
            out_nS = self._broadcast_to_state(out_nS, v_shape)

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

    def _dynamics_scalar(self, v, g_vals, is_refractory, i_stim, stc, p):
        """Compute derivatives for the ODE system [V_m, g_0, g_1, ..., g_{n-1}].

        This matches the NEST gif_cond_exp_multisynapse_dynamics() function exactly.
        During refractory period, V is clamped to V_reset and dV/dt=0.
        """
        V = p['V_reset'] if is_refractory else v

        # I_syn = -sum_k g_k * (V - E_rev_k)
        I_syn = 0.0
        for k in range(self._n_receptors):
            I_syn += -g_vals[k] * (V - self.E_rev[k])

        I_L = p['g_L'] * (V - p['E_L'])

        dv = 0.0 if is_refractory else (-I_L + i_stim + p['I_e'] + I_syn - stc) / p['C_m']

        dg = []
        for k in range(self._n_receptors):
            dg.append(-g_vals[k] / self.tau_syn[k])

        return (dv,) + tuple(dg)

    def _rkf45_integrate_scalar(self, v0, g0_vals, is_refractory, i_stim, stc, h0, dt, p):
        """Adaptive RKF45 integration matching GSL gsl_odeiv_evolve_apply behavior.

        State vector: [v, g_0, g_1, ..., g_{n-1}]
        """
        n = 1 + self._n_receptors  # total state dimension
        t = 0.0
        h = max(h0, self._MIN_H)

        # Pack state
        y = [v0] + list(g0_vals)
        iters = 0

        def f(state):
            return self._dynamics_scalar(state[0], state[1:], is_refractory, i_stim, stc, p)

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = min(h, dt - t)
            h = max(h, self._MIN_H)

            k1 = f(y)

            y2 = [y[i] + h * k1[i] / 4.0 for i in range(n)]
            k2 = f(y2)

            y3 = [y[i] + h * (3.0 * k1[i] / 32.0 + 9.0 * k2[i] / 32.0) for i in range(n)]
            k3 = f(y3)

            y4 = [y[i] + h * (1932.0 * k1[i] / 2197.0 - 7200.0 * k2[i] / 2197.0 + 7296.0 * k3[i] / 2197.0)
                  for i in range(n)]
            k4 = f(y4)

            y5 = [y[i] + h * (439.0 * k1[i] / 216.0 - 8.0 * k2[i] + 3680.0 * k3[i] / 513.0
                               - 845.0 * k4[i] / 4104.0) for i in range(n)]
            k5 = f(y5)

            y6 = [y[i] + h * (-8.0 * k1[i] / 27.0 + 2.0 * k2[i] - 3544.0 * k3[i] / 2565.0
                               + 1859.0 * k4[i] / 4104.0 - 11.0 * k5[i] / 40.0) for i in range(n)]
            k6 = f(y6)

            y4_sol = [y[i] + h * (25.0 * k1[i] / 216.0 + 1408.0 * k3[i] / 2565.0
                                   + 2197.0 * k4[i] / 4104.0 - k5[i] / 5.0) for i in range(n)]
            y5_sol = [y[i] + h * (16.0 * k1[i] / 135.0 + 6656.0 * k3[i] / 12825.0
                                   + 28561.0 * k4[i] / 56430.0 - 9.0 * k5[i] / 50.0
                                   + 2.0 * k6[i] / 55.0) for i in range(n)]

            err = max(abs(y5_sol[i] - y4_sol[i]) for i in range(n))

            if err <= self._ATOL or h <= self._MIN_H:
                y = y5_sol
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

        return y[0], y[1:], h

    def update(self, x=0.0 * u.pA):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape).copy()
        g_all = [
            self._broadcast_to_state(self._to_numpy(self.g[k].value, u.nS), v_shape).copy()
            for k in range(self._n_receptors)
        ]
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        ).copy()
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape).copy()
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape).copy()

        p = {
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
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

        # Get per-receptor synaptic inputs
        dg = self._collect_receptor_delta_inputs()

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        # Advance RNG state
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(jax.random.uniform(subkey, shape=v_shape), dtype=np.float64)

        spike_mask = np.zeros_like(V, dtype=bool)
        V_next = np.empty_like(V)
        g_next = [np.empty_like(g_all[k]) for k in range(self._n_receptors)]
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}

            # ---- Step 1: Decay stc/sfa elements and compute totals ----
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

            # ---- Step 2: Integrate ODE [V, g_0, g_1, ...] ----
            is_refractory = r[idx] > 0
            g_vals = [g_all[k][idx] for k in range(self._n_receptors)]
            v_i, g_i_list, h_i = self._rkf45_integrate_scalar(
                V[idx], g_vals, is_refractory, i_stim[idx], stc_total,
                h_int[idx], dt, local_p
            )

            # ---- Step 3: Add synaptic conductance jumps ----
            for k in range(self._n_receptors):
                g_i_list[k] += dg[k][idx]

            # ---- Step 4: Refractory / spike check ----
            if r[idx] == 0:
                # Neuron is not refractory
                lam = lambda_0 * math.exp((v_i - sfa_total) / Delta_V)
                if lam > 0.0:
                    spike_prob = -math.expm1(-lam * dt)
                    if rand_vals[idx] < spike_prob:
                        # Spike!
                        spike_mask[idx] = True

                        if self._stc_elems is not None:
                            for i in range(len(self.q_stc)):
                                self._stc_elems[i][idx] += self.q_stc[i]

                        if self._sfa_elems is not None:
                            for i in range(len(self.q_sfa)):
                                self._sfa_elems[i][idx] += self.q_sfa[i]

                        r_i = refr_counts[idx]
                    else:
                        r_i = 0
                else:
                    r_i = 0
            else:
                # Neuron is refractory
                r_i = r[idx] - 1
                v_i = local_p['V_reset']

            V_next[idx] = v_i
            for k in range(self._n_receptors):
                g_next[k][idx] = g_i_list[k]
            r_next[idx] = r_i
            h_next[idx] = h_i

        # ---- Step 5: Store new I_stim for next step ----
        self.V.value = V_next * u.mV
        for k in range(self._n_receptors):
            self.g[k].value = g_next[k] * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)
