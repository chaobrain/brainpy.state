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

r"""Current-based GIF neuron with multiple synaptic time constants.

This module implements ``gif_psc_exp_multisynapse``, the multisynapse
extension of :class:`gif_psc_exp`.  It is a faithful re-implementation of
the identically named NEST model
(``models/gif_psc_exp_multisynapse.{h,cpp}``), preserving update ordering,
exact (analytic) propagator integration, stochastic firing, and all default
parameter values.

The key difference from :class:`gif_psc_exp` is that instead of having two
fixed synaptic channels (excitatory and inhibitory), this model supports an
arbitrary number of receptor ports, each with its own exponential synaptic
time constant.  Incoming spike events specify which receptor port they
target (1-based indexing, as in NEST).

Mathematical model
------------------

Membrane potential ODE:

.. math::

   C_m \frac{dV}{dt} = -g_L (V - E_L)
       - \sum_j \eta_j(t)
       + \sum_k I_{\mathrm{syn},k}(t)
       + I_e + I_{\mathrm{stim}}(t)

Synaptic currents (one per receptor port *k*):

.. math::

   \frac{dI_{\mathrm{syn},k}}{dt} = -\frac{I_{\mathrm{syn},k}}{\tau_{\mathrm{syn},k}}

Spike-triggered currents (STC):

.. math::

   \tau_{\eta_j} \frac{d\eta_j}{dt} = -\eta_j, \qquad
   \eta_j \to \eta_j + q_{\eta_j} \;\text{on spike}

Spike-frequency adaptation (SFA) threshold:

.. math::

   V_T(t) = V_{T^*} + \sum_i \gamma_i(t), \qquad
   \tau_{\gamma_i} \frac{d\gamma_i}{dt} = -\gamma_i, \qquad
   \gamma_i \to \gamma_i + q_{\gamma_i} \;\text{on spike}

Stochastic spiking via exponential escape rate:

.. math::

   \lambda(t) = \lambda_0 \exp\!\bigl((V(t) - V_T(t)) / \Delta_V\bigr),
   \qquad P_{\text{spike}} = 1 - \exp(-\lambda \, dt)

References
----------
.. [1] Mensi S, Naud R, Pozzorini C, Avermann M, Petersen CC, Gerstner W
       (2012). Parameter extraction and classification of three cortical
       neuron types reveals two distinct adaptation mechanisms. *J.
       Neurophysiol.*, 107(6):1756-1775.
.. [2] Pozzorini C, Mensi S, Hagens O, Naud R, Koch C, Gerstner W (2015).
       Automated high-throughput characterization of single neurons by means
       of simplified spiking models. *PLoS Comput. Biol.*, 11(6), e1004275.
.. [3] NEST Simulator ``gif_psc_exp_multisynapse`` model,
       ``models/gif_psc_exp_multisynapse.h`` and
       ``models/gif_psc_exp_multisynapse.cpp``.
"""

import math
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron
from .gif_psc_exp import gif_psc_exp

__all__ = [
    'gif_psc_exp_multisynapse',
]


class gif_psc_exp_multisynapse(Neuron):
    r"""Current-based generalized integrate-and-fire neuron (GIF) model
    with multiple synaptic time constants.

    Description
    -----------

    ``gif_psc_exp_multisynapse`` is the generalized integrate-and-fire neuron
    according to Mensi et al. (2012) [1]_ and Pozzorini et al. (2015) [2]_,
    with exponential shaped postsynaptic currents and an arbitrary number of
    receptor ports.  This is the multisynapse extension of
    :class:`gif_psc_exp`, equivalent to the NEST ``gif_psc_exp_multisynapse``
    model.

    This is a brainpy.state re-implementation of the NEST simulator model of
    the same name, using NEST-standard parameterization and exact integration.

    The membrane potential :math:`V` obeys:

    .. math::

       C_m \frac{dV}{dt} = -g_L (V - E_L)
           - \eta_1(t) - \eta_2(t) - \ldots - \eta_n(t)
           + \sum_k I_{\mathrm{syn},k}(t) + I_e + I_{\mathrm{stim}}(t)

    where each :math:`\eta_j` is a spike-triggered current (stc), and
    :math:`I_{\mathrm{syn},k}` is the synaptic current for receptor port
    :math:`k`.

    Synaptic currents
    .................

    Each receptor port :math:`k` has its own exponential synaptic current:

    .. math::

       \frac{dI_{\mathrm{syn},k}}{dt}
           = -\frac{I_{\mathrm{syn},k}}{\tau_{\mathrm{syn},k}}

    On the postsynaptic side, there can be arbitrarily many synaptic time
    constants.  The number of receptor ports is determined by the length of
    ``tau_syn``.  When connecting, specify ``receptor_type`` (1-based,
    matching NEST convention) to select the target port.

    Spike-triggered currents
    ........................

    Dynamic of each :math:`\eta_j` is described by:

    .. math::

       \tau_{\eta_j} \frac{d\eta_j}{dt} = -\eta_j

    and upon spike emission:

    .. math::

       \eta_j \to \eta_j + q_{\eta_j}

    Spike-frequency adaptation
    ..........................

    The neuron produces spikes stochastically according to a point process
    with firing intensity:

    .. math::

       \lambda(t) = \lambda_0 \exp\!\left(\frac{V(t) - V_T(t)}{\Delta_V}\right)

    where :math:`V_T(t)` is a time-dependent firing threshold:

    .. math::

       V_T(t) = V_{T^*} + \gamma_1(t) + \gamma_2(t) + \ldots + \gamma_m(t)

    Each :math:`\gamma_i` obeys:

    .. math::

       \tau_{\gamma_i} \frac{d\gamma_i}{dt} = -\gamma_i

    and upon spike emission:

    .. math::

       \gamma_i \to \gamma_i + q_{\gamma_i}

    Stochastic spiking
    ..................

    The probability of firing within a time step :math:`dt` is:

    .. math::

       P(\text{spike}) = 1 - \exp(-\lambda(t) \cdot dt)

    A random number is drawn each (non-refractory) time step and compared
    to this probability.

    Refractory mechanism
    ....................

    After a spike, the neuron enters an absolute refractory period of
    duration :math:`t_\mathrm{ref}`.  During this period:

    * the refractory counter decrements each step,
    * :math:`V_m` is clamped to :math:`V_\mathrm{reset}`,
    * synaptic currents continue to decay and receive inputs.

    Numerical integration and update order
    ......................................

    NEST integrates this model with exact (analytic) propagators for the
    linear subthreshold dynamics.  The discrete-time update order per
    simulation step is (matching NEST ``gif_psc_exp_multisynapse::update``):

    1. Compute total stc (sum of stc elements) and sfa threshold
       (:math:`V_{T^*}` + sum of sfa elements).  Then decay all stc and
       sfa elements by their respective exponential factors.
    2. For each receptor *k*: compute propagated synaptic contribution to
       V (``P21_syn[k] * i_syn[k]``), then decay the synaptic current
       (``i_syn[k] *= P11_syn[k]``), then add incoming spike weight
       (``i_syn[k] += spikes[k]``).
    3. If not refractory: update membrane potential via exact propagator.
       Compute firing intensity, draw random number, potentially emit spike
       (update stc/sfa elements, set refractory counter).
       If refractory: decrement counter, clamp V to V_reset.
    4. Store external current input for the next step.

    Multisynapse differences from gif_psc_exp
    ..........................................

    Unlike ``gif_psc_exp`` which has exactly two fixed synaptic channels
    (excitatory and inhibitory with ``tau_syn_ex``, ``tau_syn_in``), this
    model supports an arbitrary number of receptor ports specified by the
    list ``tau_syn``.  All spike weights are applied to the receptor port
    specified in the connection (positive or negative weights are both
    allowed).

    .. note::

       In the NEST implementation, the stc and sfa element jumps occur
       immediately after spike emission.  The GIF toolbox uses a different
       convention where jumps occur after the refractory period.  Conversion:

       .. math::

          q_{\eta,\text{toolbox}} = q_{\eta,\text{NEST}} \cdot
              (1 - \exp(-t_\mathrm{ref} / \tau_\eta))

    .. note::

       If ``tau_m`` is very close to any ``tau_syn[k]``, the model
       will numerically behave as if they are equal, using the singular
       propagator formula to avoid numerical instabilities.

    .. note::

       Because spiking is stochastic (random number drawn each step), exact
       spike-time reproducibility requires matching the random number
       generator state.  For deterministic testing, set ``rng_key``
       explicitly.

    Parameters
    ----------

    ==================== =================== =================================== =====================================================
    **Parameter**        **Default**         **Math equivalent**                 **Description**
    ==================== =================== =================================== =====================================================
    ``in_size``          (required)                                              Population shape
    ``g_L``              4.0 nS              :math:`g_L`                         Leak conductance
    ``E_L``              -70.0 mV            :math:`E_L`                         Leak reversal potential
    ``C_m``              80.0 pF             :math:`C_m`                         Membrane capacitance
    ``V_reset``          -55.0 mV            :math:`V_\mathrm{reset}`           Reset potential
    ``Delta_V``          0.5 mV              :math:`\Delta_V`                   Stochasticity level
    ``V_T_star``         -35.0 mV            :math:`V_{T^*}`                    Base firing threshold
    ``lambda_0``         1.0 /s              :math:`\lambda_0`                  Stochastic intensity at threshold
    ``t_ref``            4.0 ms              :math:`t_\mathrm{ref}`             Absolute refractory period
    ``tau_syn``          (2.0,) ms           :math:`\tau_{\mathrm{syn},k}`      Synaptic time constants (one per receptor port)
    ``I_e``              0.0 pA              :math:`I_e`                         Constant external current
    ``tau_sfa``          () ms               :math:`\tau_{\gamma_i}`            SFA time constants (tuple/list)
    ``q_sfa``            () mV               :math:`q_{\gamma_i}`              SFA jump values (tuple/list)
    ``tau_stc``          () ms               :math:`\tau_{\eta_j}`              STC time constants (tuple/list)
    ``q_stc``            () nA               :math:`q_{\eta_j}`                STC jump values (tuple/list)
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
    ``V``                      Membrane potential :math:`V_m`
    ``i_syn``                  Synaptic currents per receptor (shape: ``(..., n_receptors)``)
    ``stc``                    Total spike-triggered current
    ``sfa``                    Adaptive threshold :math:`V_T(t)`
    ``stc_elems``              Individual stc adaptation elements
    ``sfa_elems``              Individual sfa adaptation elements
    ``refractory_step_count``  Remaining refractory grid steps
    ``I_stim``                 Buffered current applied in next step
    ``last_spike_time``        Last spike time
    ========================== ===========================================

    Notes
    -----

    - Defaults follow NEST C++ source for ``gif_psc_exp_multisynapse``.
    - ``lambda_0`` is specified in 1/s (as in NEST's Python interface) and
      is internally converted to 1/ms for computation.
    - ``tau_syn`` values are specified in ms (bare floats), matching the
      NEST C++ parameterization.  The default ``(2.0,)`` gives a single
      receptor with a 2 ms time constant.
    - Synaptic spike weights are interpreted in current units (pA).
    - The subthreshold dynamics use exact (analytic) integration via
      propagator coefficients, matching NEST's integration scheme.

    References
    ----------
    .. [1] Mensi S, Naud R, Pozzorini C, Avermann M, Petersen CC, Gerstner W
           (2012). Parameter extraction and classification of three cortical
           neuron types reveals two distinct adaptation mechanisms. Journal of
           Neurophysiology, 107(6):1756-1775.
           DOI: https://doi.org/10.1152/jn.00408.2011
    .. [2] Pozzorini C, Mensi S, Hagens O, Naud R, Koch C, Gerstner W (2015).
           Automated high-throughput characterization of single neurons by
           means of simplified spiking models. PLoS Computational Biology,
           11(6), e1004275.
           DOI: https://doi.org/10.1371/journal.pcbi.1004275
    .. [3] NEST Simulator ``gif_psc_exp_multisynapse`` model documentation
           and C++ source: ``models/gif_psc_exp_multisynapse.h`` and
           ``models/gif_psc_exp_multisynapse.cpp``.

    See Also
    --------
    gif_psc_exp, iaf_psc_exp_multisynapse
    """
    __module__ = 'brainpy.state'

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

        # Synaptic time constants (stored as numpy array of ms values)
        self.tau_syn = np.asarray([float(x) for x in tau_syn], dtype=np.float64)
        if len(self.tau_syn) == 0:
            raise ValueError("'tau_syn' must have at least one element.")

        # Stochastic spiking: lambda_0 in 1/s, store as 1/ms internally
        self.lambda_0 = lambda_0 / 1000.0  # convert from 1/s to 1/ms

        # Adaptation parameters (stored as plain Python tuples of floats)
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

        self._validate_parameters()

    @property
    def n_receptors(self):
        """Number of synaptic receptor ports."""
        return int(self.tau_syn.size)

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
                raise ValueError(
                    f'All synaptic time constants must be strictly positive '
                    f'(tau_syn[{i}]={tau}).'
                )
        for i, tau in enumerate(self.tau_sfa):
            if tau <= 0.0:
                raise ValueError(
                    f'All SFA time constants must be strictly positive '
                    f'(tau_sfa[{i}]={tau}).'
                )
        for i, tau in enumerate(self.tau_stc):
            if tau <= 0.0:
                raise ValueError(
                    f'All STC time constants must be strictly positive '
                    f'(tau_stc[{i}]={tau}).'
                )

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        self.V = brainstate.HiddenState(V)

        # Synaptic currents: shape (..., n_receptors)
        syn_zeros = np.zeros(v_shape + (self.n_receptors,), dtype=np.float64)
        self.i_syn = brainstate.ShortTermState(syn_zeros * u.pA)

        spk_time = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(
            braintools.init.Constant(0), self.varshape, batch_size
        )
        self.refractory_step_count = brainstate.ShortTermState(
            u.math.asarray(ref_steps, dtype=jnp.int32)
        )

        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(
                braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
            )
        )

        # Adaptation state: stc and sfa element arrays
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)
        self._sfa_val = np.full(
            v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=np.float64
        )

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        syn_zeros = np.zeros(v_shape + (self.n_receptors,), dtype=np.float64)
        self.i_syn.value = syn_zeros * u.pA

        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(
            braintools.init.Constant(0), self.varshape, batch_size
        )
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)
        self._sfa_val = np.full(
            v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=np.float64
        )

        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_reset) / self.Delta_V
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        """Parse spike events into per-receptor weight arrays.

        Parameters
        ----------
        spike_events : iterable or None
            Each element is either a ``(receptor_type, weight)`` tuple or a
            dict with keys ``'receptor_type'`` (1-based int) and ``'weight'``
            (pA quantity or float).
        v_shape : tuple
            Shape of the neuron population state.

        Returns
        -------
        out : ndarray, shape ``v_shape + (n_receptors,)``
            Accumulated weights in pA per receptor, as float64.
        """
        out = np.zeros(v_shape + (self.n_receptors,), dtype=np.float64)
        if spike_events is None:
            return out
        for ev in spike_events:
            if isinstance(ev, dict):
                receptor = int(ev.get('receptor_type', ev.get('receptor', 1)))
                weight = ev.get('weight', 0.0)
            else:
                receptor, weight = ev
                receptor = int(receptor)
            if receptor < 1 or receptor > self.n_receptors:
                raise ValueError(
                    f'Receptor type {receptor} out of range '
                    f'[1, {self.n_receptors}].'
                )
            w_np = np.asarray(u.math.asarray(weight / u.pA), dtype=np.float64)
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    def update(self, x=0.0 * u.pA, spike_events=None):
        """Update neuron state for one simulation step.

        Parameters
        ----------
        x : Quantity, optional
            External current input (pA).  Buffered by one step (NEST ring
            buffer semantics).  Default is 0.
        spike_events : iterable of (receptor_type, weight) or dicts, optional
            Incoming spike events with receptor port and weight.  If ``None``,
            only delta inputs registered via ``add_delta_input`` are used
            (mapped to receptor 1).

        Returns
        -------
        spike : array
            Spike output (float, via surrogate gradient function).
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))  # dt in ms

        v_shape = self.V.value.shape

        # Extract state variables as numpy arrays
        V = self._broadcast_to_state(
            self._to_numpy(self.V.value, u.mV), v_shape
        ).copy()
        i_syn = np.asarray(
            u.math.asarray(self.i_syn.value / u.pA), dtype=np.float64
        ).copy()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            v_shape,
        ).copy()
        i_stim = self._broadcast_to_state(
            self._to_numpy(self.I_stim.value, u.pA), v_shape
        ).copy()

        # Extract parameters as numpy arrays
        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        g_L = self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        V_reset = self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape)
        V_T_star = float(self._to_numpy(self.V_T_star, u.mV))
        Delta_V = float(self._to_numpy(self.Delta_V, u.mV))
        lambda_0 = self.lambda_0  # 1/ms

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )

        # Compute propagator coefficients (exact integration)
        tau_m = C_m / g_L  # membrane time constant in ms
        P33 = np.exp(-h / tau_m)
        P30 = -1.0 / C_m * np.expm1(-h / tau_m) * tau_m
        P31 = -np.expm1(-h / tau_m)

        # Per-receptor propagator coefficients
        P11_syn = np.exp(-h / self.tau_syn)  # shape: (n_receptors,)
        P21_syn = np.stack([
            gif_psc_exp._propagator_exp(
                tau_s * np.ones(v_shape), tau_m, C_m, h
            )
            for tau_s in self.tau_syn
        ], axis=-1)  # shape: v_shape + (n_receptors,)

        # Adaptation decay factors
        P_stc = [math.exp(-h / tau) for tau in self.tau_stc]
        P_sfa = [math.exp(-h / tau) for tau in self.tau_sfa]

        # Parse spike events into per-receptor arrays
        w_by_rec = self._parse_spike_events(spike_events, v_shape)

        # Map default delta inputs to receptor 1
        w_default = self._broadcast_to_state(
            self._to_numpy(self.sum_delta_inputs(0.0 * u.pA), u.pA), v_shape
        )
        if self.n_receptors > 0:
            w_by_rec[..., 0] += w_default

        # Get external current for NEXT step (NEST ring buffer semantics)
        new_i_stim = self._broadcast_to_state(
            self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape
        )

        # Advance RNG state for this step
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(
            jax.random.uniform(subkey, shape=v_shape), dtype=np.float64
        )

        spike_mask = np.zeros_like(V, dtype=bool)

        for idx in np.ndindex(v_shape):
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

            # ---- Step 2: Synaptic currents ----
            # Compute propagated contribution, decay, then add new spikes
            # (matches NEST update order exactly)
            sum_syn_pot = 0.0
            for k in range(self.n_receptors):
                syn_idx = idx + (k,)
                sum_syn_pot += P21_syn[syn_idx] * i_syn[syn_idx]
                i_syn[syn_idx] *= P11_syn[k]
                i_syn[syn_idx] += w_by_rec[syn_idx]

            # ---- Step 3: Refractory / membrane update / spike check ----
            if r[idx] == 0:
                # Not refractory: update membrane potential via exact propagator
                V[idx] = (P30[idx] * (i_stim[idx] + I_e[idx] - stc_total)
                          + P33[idx] * V[idx]
                          + P31[idx] * E_L[idx]
                          + sum_syn_pot)

                # Stochastic spike check
                lam = lambda_0 * math.exp((V[idx] - sfa_total) / Delta_V)
                if lam > 0.0:
                    spike_prob = -math.expm1(-lam * h)
                    if rand_vals[idx] < spike_prob:
                        # Spike!
                        spike_mask[idx] = True

                        # Jump stc elements
                        if self._stc_elems is not None:
                            for i in range(len(self.q_stc)):
                                self._stc_elems[i][idx] += self.q_stc[i]

                        # Jump sfa elements
                        if self._sfa_elems is not None:
                            for i in range(len(self.q_sfa)):
                                self._sfa_elems[i][idx] += self.q_sfa[i]

                        r[idx] = refr_counts[idx]
            else:
                # Refractory: decrement counter, clamp V to V_reset
                r[idx] -= 1
                V[idx] = V_reset[idx]

        # ---- Step 4: Store new I_stim for next step, update state ----
        self.V.value = V * u.mV
        self.i_syn.value = i_syn * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)
