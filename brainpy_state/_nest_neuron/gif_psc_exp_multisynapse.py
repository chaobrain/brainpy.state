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
       neuron types reveals two distinct adaptation mechanisms.
       *J. Neurophysiol.*, 107(6):1756-1775.
.. [2] Pozzorini C, Mensi S, Hagens O, Naud R, Koch C, Gerstner W (2015).
       Automated high-throughput characterization of single neurons by means
       of simplified spiking models. *PLoS Comput. Biol.*, 11(6), e1004275.
.. [3] NEST Simulator ``gif_psc_exp_multisynapse`` model,
       ``models/gif_psc_exp_multisynapse.h`` and
       ``models/gif_psc_exp_multisynapse.cpp``.
"""

from typing import Callable, Optional, Sequence

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, propagator_exp, cond_any

__all__ = [
    'gif_psc_exp_multisynapse',
]


# ---------------------------------------------------------------------------
# Proxy classes for mutable [i][j] access to ShortTermState arrays
# ---------------------------------------------------------------------------

class _RowProxy:
    """Read/write proxy for row ``row`` of a 2-D ShortTermState array.

    Supports ``proxy[j]`` (read) and ``proxy[j] = val`` (write) where the
    underlying state has shape ``(n_elems, *varshape)`` and units ``unit``.
    """
    __slots__ = ('_state', '_row', '_unit')

    def __init__(self, state, row, unit):
        self._state = state
        self._row = row
        self._unit = unit

    def __getitem__(self, j):
        raw = np.asarray(u.get_mantissa(self._state.value))
        return float(raw[self._row, j])

    def __setitem__(self, j, val):
        raw = np.asarray(u.get_mantissa(self._state.value)).copy()
        raw[self._row, j] = float(val)
        self._state.value = raw * self._unit


class _AdaptProxy:
    """Proxy exposing ``elems[i][j]`` indexing for adaptation state arrays."""
    __slots__ = ('_state', '_unit')

    def __init__(self, state, unit):
        self._state = state
        self._unit = unit

    def __getitem__(self, i):
        return _RowProxy(self._state, i, self._unit)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class gif_psc_exp_multisynapse(NESTNeuron):
    r"""Current-based generalized integrate-and-fire neuron (GIF) model
    with multiple synaptic time constants.

    This model implements the multisynapse extension of the generalized
    integrate-and-fire neuron according to Mensi et al. (2012) [1]_ and
    Pozzorini et al. (2015) [2]_, with exponential postsynaptic currents
    and an arbitrary number of receptor ports. It is a faithful
    re-implementation of the NEST simulator's ``gif_psc_exp_multisynapse``
    model, preserving exact (analytic) propagator integration, stochastic
    firing dynamics, update ordering, and all default parameter values.

    The model combines four key features:

    1. **Multiple receptor ports**: Each with independent exponential
       synaptic time constants (``tau_syn`` parameter)
    2. **Spike-triggered currents (STC)**: Post-spike current injection
       with multiple time scales (``tau_stc``, ``q_stc`` parameters)
    3. **Spike-frequency adaptation (SFA)**: Dynamic threshold modulation
       after each spike (``tau_sfa``, ``q_sfa`` parameters)
    4. **Stochastic spiking**: Exponential escape-rate firing with
       parameter ``lambda_0`` and threshold noise ``Delta_V``

    Mathematical Model
    ------------------

    **1. Membrane Potential Dynamics**

    The subthreshold membrane potential :math:`V(t)` evolves according to:

    .. math::

       C_m \frac{dV}{dt} = -g_L (V - E_L) - \sum_j \eta_j(t)
           + \sum_k I_{\mathrm{syn},k}(t) + I_e + I_{\mathrm{stim}}(t)

    where:

      - :math:`g_L (V - E_L)` is the passive leak current
      - :math:`\eta_j(t)` are spike-triggered currents (STCs)
      - :math:`I_{\mathrm{syn},k}(t)` are synaptic currents for each receptor port :math:`k`
      - :math:`I_e` is a constant external bias current
      - :math:`I_{\mathrm{stim}}(t)` is time-varying external input

    **2. Synaptic Currents (Multi-Receptor)**

    Each receptor port :math:`k` has an independent exponential synaptic current:

    .. math::

       \frac{dI_{\mathrm{syn},k}}{dt} = -\frac{I_{\mathrm{syn},k}}{\tau_{\mathrm{syn},k}}

    The number of receptor ports is determined by ``len(tau_syn)``. When
    connecting projections, specify ``receptor_type`` (1-based indexing,
    matching NEST convention) to target a specific port. Both excitatory
    and inhibitory connections can target any receptor port (weights can
    be positive or negative).

    **3. Spike-Triggered Currents (STC)**

    Each STC element :math:`\eta_j` evolves as:

    .. math::

       \tau_{\eta_j} \frac{d\eta_j}{dt} = -\eta_j

    Upon spike emission at time :math:`t_{\mathrm{sp}}`:

    .. math::

       \eta_j(t_{\mathrm{sp}}^+) = \eta_j(t_{\mathrm{sp}}^-) + q_{\eta_j}

    The total STC contribution is :math:`\sum_j \eta_j(t)`. STCs can model
    post-spike currents such as afterhyperpolarization (AHP) or
    afterdepolarization (ADP) depending on the sign of ``q_stc``.

    **4. Spike-Frequency Adaptation (SFA)**

    The firing threshold is dynamic, consisting of a base threshold
    :math:`V_{T^*}` plus adaptive components:

    .. math::

       V_T(t) = V_{T^*} + \sum_i \gamma_i(t)

    Each SFA element :math:`\gamma_i` evolves as:

    .. math::

       \tau_{\gamma_i} \frac{d\gamma_i}{dt} = -\gamma_i

    Upon spike emission:

    .. math::

       \gamma_i(t_{\mathrm{sp}}^+) = \gamma_i(t_{\mathrm{sp}}^-) + q_{\gamma_i}

    Positive ``q_sfa`` values increase the threshold after each spike,
    leading to spike-frequency adaptation.

    **5. Stochastic Spiking Mechanism**

    The neuron fires stochastically with an exponential escape-rate
    intensity:

    .. math::

       \lambda(t) = \lambda_0 \exp\!\left(\frac{V(t) - V_T(t)}{\Delta_V}\right)

    The probability of firing within a time step :math:`dt` is:

    .. math::

       P_{\mathrm{spike}}(\Delta t) = 1 - \exp(-\lambda(t) \cdot dt)

    At each non-refractory time step, a uniform random number
    :math:`r \in [0, 1)` is drawn. If :math:`r < P_{\mathrm{spike}}`, a
    spike is emitted. The stochasticity level :math:`\Delta_V` controls
    the sharpness of the firing threshold (smaller values → more
    deterministic).

    **6. Refractory Period**

    After a spike, the neuron enters an absolute refractory period of
    duration :math:`t_{\mathrm{ref}}`. During this period:

    * The membrane potential is clamped to :math:`V_{\mathrm{reset}}`
    * No spike can be emitted (firing intensity check is skipped)
    * Synaptic currents continue to evolve and receive inputs
    * STC and SFA elements continue to decay

    **Numerical Integration**

    The model uses **exact (analytic) integration** for all linear ODEs,
    matching NEST's propagator-based integration scheme. For each variable
    with dynamics :math:`\tau \frac{dx}{dt} = -x + f(t)`, the update over
    one time step :math:`h` is:

    .. math::

       x(t + h) = e^{-h/\tau} x(t) + \int_0^h e^{-(h-s)/\tau} f(t+s) \, ds

    For constant :math:`f`, this yields exact propagator coefficients. The
    membrane potential propagator accounts for coupling between :math:`V`
    and synaptic currents with potentially different time constants.

    **Update Order (Matching NEST)**

    Each simulation step follows this exact sequence (matching NEST's
    ``gif_psc_exp_multisynapse::update`` implementation):

    **Step 1: Adaptation Decay**
      - Compute total STC: :math:`\mathrm{stc\_total} = \sum_j \eta_j(t)`
      - Compute total threshold: :math:`V_T(t) = V_{T^*} + \sum_i \gamma_i(t)`
      - Decay all STC elements: :math:`\eta_j \leftarrow \eta_j \cdot e^{-dt/\tau_{\eta_j}}`
      - Decay all SFA elements: :math:`\gamma_i \leftarrow \gamma_i \cdot e^{-dt/\tau_{\gamma_i}}`

    **Step 2: Synaptic Current Processing (per receptor)**
      For each receptor port :math:`k`:
        - Compute propagated contribution to :math:`V`:
          :math:`\Delta V_k = P_{21,k} \cdot I_{\mathrm{syn},k}(t)`
        - Decay synaptic current:
          :math:`I_{\mathrm{syn},k} \leftarrow I_{\mathrm{syn},k} \cdot e^{-dt/\tau_{\mathrm{syn},k}}`
        - Add incoming spike weights:
          :math:`I_{\mathrm{syn},k} \leftarrow I_{\mathrm{syn},k} + w_k`

    **Step 3: Membrane Update and Spike Check**
      If **not refractory**:
        - Update membrane potential using exact propagator:

          .. math::

             V(t+dt) = P_{33} V(t) + P_{31} E_L + P_{30}(I_{\mathrm{stim}}(t) + I_e - \mathrm{stc\_total})
                 + \sum_k \Delta V_k

        - Compute firing intensity: :math:`\lambda = \lambda_0 \exp((V - V_T)/\Delta_V)`
        - Compute spike probability: :math:`p = 1 - \exp(-\lambda \cdot dt)`
        - Draw random number :math:`r \sim \mathrm{Uniform}(0, 1)`
        - If :math:`r < p`:
            * Emit spike
            * Jump STC elements: :math:`\eta_j \leftarrow \eta_j + q_{\eta_j}`
            * Jump SFA elements: :math:`\gamma_i \leftarrow \gamma_i + q_{\gamma_i}`
            * Set refractory counter: :math:`r_{\mathrm{count}} \leftarrow \lceil t_{\mathrm{ref}} / dt \rceil`
      If **refractory**:
        - Decrement refractory counter: :math:`r_{\mathrm{count}} \leftarrow r_{\mathrm{count}} - 1`
        - Clamp membrane potential: :math:`V \leftarrow V_{\mathrm{reset}}`

    **Step 4: Buffer External Current**
      Store :math:`I_{\mathrm{stim}}(t+dt)` for use in the next step
      (NEST ring-buffer semantics: one-step delay).

    **Differences from gif_psc_exp**

    Unlike :class:`gif_psc_exp` which has exactly two fixed synaptic
    channels (excitatory and inhibitory with ``tau_syn_ex``,
    ``tau_syn_in``), this model supports an arbitrary number of receptor
    ports specified by the ``tau_syn`` parameter. This enables:

    * Multi-receptor modeling (AMPA, NMDA, GABA_A, GABA_B, etc.)
    * Heterogeneous synaptic time constants within the same neuron
    * Flexible connectivity patterns with receptor-specific routing

    All spike weights are applied to the receptor port specified in the
    connection's ``receptor_type`` field (1-based indexing). Positive or
    negative weights are both allowed on any receptor.

    Parameters
    ----------
    in_size : int, tuple of int
        Shape of the neuron population.
    g_L : Quantity, ArrayLike, optional
        Leak conductance (nanosiemens). Default: 4.0 nS.
    E_L : Quantity, ArrayLike, optional
        Leak reversal potential (millivolts). Default: -70.0 mV.
    C_m : Quantity, ArrayLike, optional
        Membrane capacitance (picofarads). Default: 80.0 pF.
    V_reset : Quantity, ArrayLike, optional
        Reset potential (millivolts). Default: -55.0 mV.
    Delta_V : Quantity, ArrayLike, optional
        Voltage scale of stochastic firing (millivolts). Default: 0.5 mV.
    V_T_star : Quantity, ArrayLike, optional
        Base firing threshold (millivolts). Default: -35.0 mV.
    lambda_0 : float, optional
        Stochastic firing intensity at threshold (1/s). Default: 1.0 /s.
    t_ref : Quantity, ArrayLike, optional
        Absolute refractory period (milliseconds). Default: 4.0 ms.
    tau_syn : sequence of float, optional
        Synaptic time constants (milliseconds), one per receptor port.
        Specified as bare floats (not Quantities). Default: ``(2.0,)``.
    I_e : Quantity, ArrayLike, optional
        Constant external bias current (picoamperes). Default: 0.0 pA.
    tau_sfa : sequence of float, optional
        SFA time constants (milliseconds). Default: ``()`` (no adaptation).
    q_sfa : sequence of float, optional
        SFA jump values (millivolts). Default: ``()`` (no adaptation).
    tau_stc : sequence of float, optional
        STC time constants (milliseconds). Default: ``()`` (no STC).
    q_stc : sequence of float, optional
        STC jump values (picoamperes). Default: ``()`` (no STC).
    rng_key : jax.Array, optional
        JAX PRNG key for stochastic spike generation. Default: None.
    V_initializer : Callable, optional
        Initializer for membrane potential. Default: ``Constant(-70.0 mV)``.
    spk_fun : Callable, optional
        Surrogate gradient function. Default: ``ReluGrad()``.
    spk_reset : str, optional
        Spike reset mode (``'hard'`` or ``'soft'``). Default: ``'hard'``.
    name : str, optional
        Name of the neuron population. Default: None.

    State Variables
    ---------------
    V : HiddenState, shape ``(*in_size,)``
        Membrane potential in millivolts.
    i_syn : ShortTermState, shape ``(*in_size, n_receptors)``
        Synaptic currents in picoamperes.
    refractory_step_count : ShortTermState, shape ``(*in_size,)``
        Remaining refractory steps (int).
    I_stim : ShortTermState, shape ``(*in_size,)``
        Buffered external current (one-step delay).
    last_spike_time : ShortTermState, shape ``(*in_size,)``
        Time of last spike (milliseconds).

    See Also
    --------
    gif_psc_exp : Two-receptor GIF model
    iaf_psc_exp_multisynapse : Multi-receptor IAF model without adaptation

    References
    ----------
    .. [1] Mensi S et al. (2012). J. Neurophysiol., 107(6):1756-1775.
    .. [2] Pozzorini C et al. (2015). PLoS Comput. Biol., 11(6), e1004275.
    .. [3] NEST Simulator ``gif_psc_exp_multisynapse`` model documentation.
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
        q_sfa: Sequence[float] = (),    # mV values
        tau_stc: Sequence[float] = (),  # ms values
        q_stc: Sequence[float] = (),    # pA values
        rng_key: Optional[jax.Array] = None,
        V_initializer: Callable = braintools.init.Constant(-70.0 * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
        # Accepted for backward compatibility but unused:
        gsl_error_tol: ArrayLike = 1e-6,
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

        # tau_syn: stored as plain numpy array of ms values (no units)
        if len(tau_syn) == 0:
            raise ValueError("'tau_syn' must have at least one element.")
        dftype = brainstate.environ.dftype()
        self.tau_syn = np.asarray([float(x) for x in tau_syn], dtype=dftype)

        # Stochastic spiking: lambda_0 in 1/s → store as 1/ms
        self.lambda_0 = lambda_0 / 1000.0

        # Adaptation parameters (stored as Python tuples of bare floats)
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

        self.n_stc = len(self.tau_stc)
        self.n_sfa = len(self.tau_sfa)

        # RNG key for stochastic spiking
        self._rng_key = rng_key

        # Initializer
        self.V_initializer = V_initializer

        self._validate_parameters()

        # Pre-compute refractory step count
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    @property
    def n_receptors(self):
        r"""Number of synaptic receptor ports."""
        return int(self.tau_syn.shape[0])

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
        for i, tau in enumerate(self.tau_syn):
            if tau <= 0.0:
                raise ValueError(
                    f'All synaptic time constants must be strictly positive '
                    f'(tau_syn[{i}]={tau}).'
                )

    def init_state(self, **kwargs):
        r"""Initialize all state variables.

        Creates membrane potential, synaptic currents, refractory counters,
        adaptation elements, buffered current, and internal RNG state.
        """
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        V = braintools.init.param(self.V_initializer, self.varshape)
        self.V = brainstate.HiddenState(V)
        v_shape = V.shape

        # Synaptic currents: shape (*v_shape, n_receptors) — float64 for precision
        syn_shape = v_shape + (self.n_receptors,)
        self.i_syn = brainstate.ShortTermState(
            np.zeros(syn_shape, dtype=np.float64) * u.pA
        )

        # STC elements: shape (n_stc, *v_shape) in pA — float64 for precision
        if self.n_stc > 0:
            stc_shape = (self.n_stc,) + v_shape
            self.stc_elems = brainstate.ShortTermState(
                np.zeros(stc_shape, dtype=np.float64) * u.pA
            )
        else:
            self.stc_elems = None

        # SFA elements: shape (n_sfa, *v_shape) in mV — float64 for precision
        if self.n_sfa > 0:
            sfa_shape = (self.n_sfa,) + v_shape
            self.sfa_elems = brainstate.ShortTermState(
                np.zeros(sfa_shape, dtype=np.float64) * u.mV
            )
        else:
            self.sfa_elems = None

        self.last_spike_time = brainstate.ShortTermState(
            u.math.full(v_shape, -1e7 * u.ms)
        )
        self.refractory_step_count = brainstate.ShortTermState(
            u.math.full(v_shape, 0, dtype=ditype)
        )
        self.I_stim = brainstate.ShortTermState(
            u.math.full(v_shape, 0.0 * u.pA, dtype=dftype)
        )

        # Caches for pre-decay totals (accessed via _stc_val / _sfa_val)
        V_T_star_mV = float(np.asarray(u.get_mantissa(self.V_T_star)))
        self._stc_val_cache = np.zeros(v_shape, dtype=np.float64)
        self._sfa_val_cache = np.full(v_shape, V_T_star_mV, dtype=np.float64)

        # RNG state — stored as ShortTermState so brainstate.transform.for_loop can track it
        rng_init = self._rng_key if self._rng_key is not None else jax.random.PRNGKey(0)
        self._rng_state = brainstate.ShortTermState(rng_init)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute spike output via surrogate gradient function."""
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_reset) / self.Delta_V
        return self.spk_fun(v_scaled)

    def _parse_spike_events(self, spike_events, v_shape):
        r"""Parse spike event descriptors into a per-receptor weight array.

        Parameters
        ----------
        spike_events : iterable or None
            Events as ``(receptor_type, weight)`` tuples or dicts with
            ``'receptor_type'`` and ``'weight'`` keys. ``None`` → no events.
        v_shape : tuple of int
            Shape of the neuron population state.

        Returns
        -------
        out : np.ndarray, shape ``v_shape + (n_receptors,)``, dtype float64
            Total weight (pA) arriving at each receptor this step.

        Raises
        ------
        ValueError
            If any ``receptor_type`` is outside ``[1, n_receptors]``.
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
                    f'Receptor type {receptor} out of range [1, {self.n_receptors}].'
                )
            w_np = np.asarray(u.math.asarray(weight / u.pA), dtype=np.float64)
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    # ------------------------------------------------------------------
    # Adaptation element proxy properties
    # ------------------------------------------------------------------

    @property
    def _stc_elems(self):
        """Proxy for ``stc_elems[i][j]`` read/write access (units: pA)."""
        if self.stc_elems is None:
            raise AttributeError('No STC elements configured (tau_stc is empty).')
        return _AdaptProxy(self.stc_elems, u.pA)

    @property
    def _sfa_elems(self):
        """Proxy for ``sfa_elems[i][j]`` read/write access (units: mV)."""
        if self.sfa_elems is None:
            raise AttributeError('No SFA elements configured (tau_sfa is empty).')
        return _AdaptProxy(self.sfa_elems, u.mV)

    @property
    def _stc_val(self):
        """Pre-decay STC totals (pA) from the last completed update step."""
        return self._stc_val_cache

    @property
    def _sfa_val(self):
        """Pre-decay SFA threshold totals (mV) from the last completed update step."""
        return self._sfa_val_cache

    # ------------------------------------------------------------------
    # Main update method
    # ------------------------------------------------------------------

    def update(self, x=0.0 * u.pA, spike_events=None, receptor_weights=None):
        r"""Update neuron state for one simulation step.

        Follows NEST's ``gif_psc_exp_multisynapse::update`` exactly:

        1. Compute pre-decay STC/SFA totals, then decay.
        2. Propagate + decay + inject per-receptor synaptic currents.
        3. If not refractory: exact-propagator membrane update + stochastic
           spike check; if spiked, jump STC/SFA and set refractory counter.
           If refractory: decrement counter, clamp V to V_reset.
        4. Buffer external current for next step (one-step delay).

        Parameters
        ----------
        x : Quantity, optional
            External current input (pA), buffered by one step. Default: 0 pA.
        spike_events : iterable or None, optional
            Receptor-indexed spike events. Default: None.
        receptor_weights : jax.Array or None, optional
            Pre-computed per-receptor weight array, shape ``v_shape +
            (n_receptors,)``.  When provided, these weights are added
            directly to the synaptic currents after decay (same semantics
            as ``spike_events``).  Useful inside ``brainstate.transform.for_loop``
            where Python-level spike_events iteration is not traceable.
            Default: None.

        Returns
        -------
        jax.Array
            Binary spike output (float), shape ``self.V.value.shape``.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        h = float(u.get_mantissa(dt / u.ms))           # step in ms (concrete Python float)
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        v_shape = self.V.value.shape

        # ---- Strip units from parameters (concrete Python values via get_mantissa) ----
        tau_m_ms = np.asarray(u.get_mantissa(self.C_m / self.g_L / u.ms))
        C_m_pF = np.asarray(u.get_mantissa(self.C_m / u.pF))
        E_L_mV = np.asarray(u.get_mantissa(self.E_L / u.mV))
        V_reset_mV = np.asarray(u.get_mantissa(self.V_reset / u.mV))
        Delta_V_mV = np.asarray(u.get_mantissa(self.Delta_V / u.mV))
        V_T_star_mV = np.asarray(u.get_mantissa(self.V_T_star / u.mV))
        I_e_pA = np.asarray(u.get_mantissa(self.I_e / u.pA))

        # ---- Read state (JAX arrays, compatible with for_loop tracing) ----
        V_mV = u.get_mantissa(self.V.value / u.mV)
        i_syn_pA = u.get_mantissa(self.i_syn.value / u.pA)
        r = self.refractory_step_count.value
        I_stim_pA = u.get_mantissa(self.I_stim.value / u.pA)

        # ---- Propagator coefficients ----
        P33 = np.exp(-h / tau_m_ms)
        P30 = -1.0 / C_m_pF * np.expm1(-h / tau_m_ms) * tau_m_ms
        P31 = -np.expm1(-h / tau_m_ms)
        P11_syn = np.exp(-h / self.tau_syn)   # shape (n_receptors,)
        P21_syn = np.stack([
            propagator_exp(ts * np.ones(v_shape), tau_m_ms, C_m_pF, h)
            for ts in self.tau_syn
        ], axis=-1)   # shape v_shape + (n_receptors,)

        # ---- Step 1: Adaptation — compute pre-decay totals, then decay ----
        if self.n_stc > 0:
            stc_elems_pA = jnp.asarray(u.get_mantissa(self.stc_elems.value / u.pA))
            stc_total_pA = jnp.sum(stc_elems_pA, axis=0)   # pre-decay total
            P_stc = np.exp(-h / np.array(self.tau_stc, dtype=np.float64))
            P_stc_bc = P_stc.reshape((-1,) + (1,) * len(v_shape))
            stc_elems_pA = stc_elems_pA * P_stc_bc        # decay
        else:
            stc_total_pA = jnp.zeros(v_shape)
            stc_elems_pA = None

        if self.n_sfa > 0:
            sfa_elems_mV = jnp.asarray(u.get_mantissa(self.sfa_elems.value / u.mV))
            sfa_total_mV = V_T_star_mV + jnp.sum(sfa_elems_mV, axis=0)  # pre-decay
            P_sfa = np.exp(-h / np.array(self.tau_sfa, dtype=np.float64))
            P_sfa_bc = P_sfa.reshape((-1,) + (1,) * len(v_shape))
            sfa_elems_mV = sfa_elems_mV * P_sfa_bc        # decay
        else:
            sfa_total_mV = jnp.broadcast_to(jnp.asarray(V_T_star_mV), v_shape)
            sfa_elems_mV = None

        # Cache pre-decay totals for external inspection
        self._stc_val_cache = stc_total_pA
        self._sfa_val_cache = sfa_total_mV

        # ---- Step 2: Synaptic currents (propagate, decay, inject) ----
        # Propagate contribution to V using pre-decay i_syn
        sum_syn_pot = jnp.sum(P21_syn * i_syn_pA, axis=-1)  # shape v_shape

        # Decay each receptor current
        i_syn_pA = i_syn_pA * P11_syn

        # Parse spike events and add delta inputs (both go to receptors)
        w_by_rec = self._parse_spike_events(spike_events, v_shape)
        w_default_pA = u.get_mantissa(self.sum_delta_inputs(0.0 * u.pA) / u.pA)
        w_by_rec[..., 0] = w_by_rec[..., 0] + np.broadcast_to(np.asarray(w_default_pA), v_shape)
        i_syn_pA = i_syn_pA + jnp.asarray(w_by_rec)   # add spike weights AFTER decay
        if receptor_weights is not None:
            i_syn_pA = i_syn_pA + receptor_weights

        # Buffer current for NEXT step (NEST ring-buffer semantics)
        new_I_stim = self.sum_current_inputs(x, self.V.value)

        # ---- Step 3: Membrane update and stochastic spike check ----
        not_refractory = (r == 0)

        # Candidate V for non-refractory neurons (NEST propagator update)
        V_candidate_mV = (
            P30 * (I_stim_pA + I_e_pA - stc_total_pA)
            + P33 * V_mV
            + P31 * E_L_mV
            + sum_syn_pot
        )

        # Stochastic spike check (only for non-refractory)
        new_rng, subkey = jax.random.split(self._rng_state.value)
        self._rng_state.value = new_rng
        rand_vals = jax.random.uniform(subkey, shape=v_shape)

        exp_arg = jnp.clip(
            (V_candidate_mV - sfa_total_mV) / Delta_V_mV, -500.0, 500.0
        )
        lam = self.lambda_0 * jnp.exp(exp_arg)           # 1/ms
        spike_prob = jnp.clip(-jnp.expm1(-lam * h), 0.0, 1.0)
        spike_mask = not_refractory & (rand_vals < spike_prob)

        # Final V: spike or refractory → V_reset; else → V_candidate
        new_V_mV = jnp.where(
            not_refractory & ~spike_mask,
            V_candidate_mV,
            V_reset_mV,
        )

        # STC jumps on spike (applied to already-decayed elements)
        if self.n_stc > 0:
            q_stc_arr = np.array(self.q_stc, dtype=np.float64)
            stc_elems_pA = jnp.asarray(stc_elems_pA)
            for i in range(self.n_stc):
                stc_elems_pA = stc_elems_pA.at[i].set(
                    jnp.where(spike_mask, stc_elems_pA[i] + q_stc_arr[i], stc_elems_pA[i])
                )

        # SFA jumps on spike
        if self.n_sfa > 0:
            q_sfa_arr = np.array(self.q_sfa, dtype=np.float64)
            sfa_elems_mV = jnp.asarray(sfa_elems_mV)
            for i in range(self.n_sfa):
                sfa_elems_mV = sfa_elems_mV.at[i].set(
                    jnp.where(spike_mask, sfa_elems_mV[i] + q_sfa_arr[i], sfa_elems_mV[i])
                )

        # Update refractory counter:
        #   spike       → ref_count
        #   refractory  → r - 1
        #   otherwise   → r (keep 0)
        ref_count_jax = u.get_mantissa(self.ref_count)
        new_r = jnp.where(
            spike_mask,
            ref_count_jax,
            jnp.where(not_refractory, r, r - 1),
        )

        # ---- Write back state ----
        self.V.value = new_V_mV * u.mV
        self.i_syn.value = i_syn_pA * u.pA
        if self.n_stc > 0:
            self.stc_elems.value = stc_elems_pA * u.pA
        if self.n_sfa > 0:
            self.sfa_elems.value = sfa_elems_mV * u.mV
        self.refractory_step_count.value = new_r.astype(ditype)
        self.I_stim.value = new_I_stim + u.math.zeros(v_shape) * u.pA
        last_spike_time = u.math.where(
            spike_mask, t + dt, self.last_spike_time.value
        )
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return jnp.asarray(spike_mask, dtype=dftype)
