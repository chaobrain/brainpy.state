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
    'pp_psc_delta',
]


class pp_psc_delta(Neuron):
    r"""Point process neuron with leaky integration of delta-shaped PSCs.

    Description
    -----------

    ``pp_psc_delta`` is an implementation of a leaky integrator where the
    potential jumps on each spike arrival. It produces spikes stochastically,
    and supports spike-frequency adaptation and other optional features.

    This is a brainpy.state re-implementation of the NEST simulator model of
    the same name, using NEST-standard parameterization and exact integration.

    Spikes are generated randomly according to the current value of the
    transfer function which operates on the membrane potential. Spike
    generation is followed by an optional dead time. Setting ``with_reset``
    to ``True`` will reset the membrane potential after each spike.

    Transfer function
    .................

    The transfer function can be chosen to be linear, exponential or a sum of
    both by adjusting three parameters:

    .. math::

       \text{rate} = \text{Rect}\!\left[
           c_1 \cdot V' + c_2 \cdot \exp(c_3 \cdot V')
       \right]

    where the effective potential :math:`V' = V_\mathrm{m} - E_\mathrm{sfa}`
    and :math:`E_\mathrm{sfa}` is the adaptive threshold. The rectifier
    :math:`\text{Rect}(x) = \max(0, x)` is applied because negative rates are
    not possible.

    By setting ``c_3 = 0``, ``c_2`` can be used as an offset spike rate for an
    otherwise linear rate model.

    Dead time (refractory period)
    .............................

    The dead time enables refractoriness. If ``dead_time`` is 0, the number of
    spikes in one time step may exceed one and is drawn from the Poisson
    distribution. Otherwise, the probability for a spike is given by
    :math:`1 - \exp(-\text{rate} \cdot h)`, where :math:`h` is the simulation
    time step. If ``dead_time`` is smaller than the simulation resolution, it
    is internally set to the resolution.

    Note that even for non-refractory neurons, a small value of ``dead_time``
    (e.g. ``1e-8`` ms) may be preferred since it uses faster uniform random
    numbers rather than Poisson draws. Only for very large spike rates
    (> 1 spike/time_step) will this cause errors.

    If ``dead_time_random`` is ``True``, the dead time after each spike is
    drawn from a gamma distribution with shape ``dead_time_shape`` and
    mean ``dead_time``.

    Spike-frequency adaptation
    ..........................

    If the neuron spikes, the adaptive threshold :math:`E_\mathrm{sfa}`
    increases and the effective membrane potential will be smaller.
    :math:`E_\mathrm{sfa}` jumps by ``q_sfa`` when the neuron fires a spike,
    and decays exponentially with time constant ``tau_sfa``:

    .. math::

       \tau_{\mathrm{sfa},i} \frac{dE_{\mathrm{sfa},i}}{dt} = -E_{\mathrm{sfa},i}

    .. math::

       E_{\mathrm{sfa},i} \to E_{\mathrm{sfa},i} + q_{\mathrm{sfa},i}
       \quad \text{(on spike)}

    .. math::

       E_\mathrm{sfa}(t) = \sum_i E_{\mathrm{sfa},i}(t)

    The adaptation kernel may be the sum of *n* exponential kernels by passing
    ``q_sfa`` and ``tau_sfa`` as lists of *n* values.

    Membrane dynamics
    .................

    The membrane potential evolves according to:

    .. math::

       C_\mathrm{m} \frac{dV_\mathrm{m}}{dt} = -\frac{V_\mathrm{m}}{\tau_\mathrm{m}}
       + I_\mathrm{e} + I_\mathrm{syn}

    Note that :math:`V_\mathrm{m}` is **relative to the resting potential**
    (resting potential = 0 mV in this model). The exact (analytic) integration
    over one time step :math:`h` gives:

    .. math::

       V_\mathrm{m}(t + h) = P_{33} \cdot V_\mathrm{m}(t)
       + P_{30} \cdot (I_0 + I_\mathrm{e})
       + w_\mathrm{syn}

    where :math:`P_{33} = \exp(-h / \tau_\mathrm{m})`,
    :math:`P_{30} = \frac{\tau_\mathrm{m}}{C_\mathrm{m}}(1 - P_{33})`, and
    :math:`w_\mathrm{syn}` is the weighted sum of all incoming spike events
    (delta-shaped post-synaptic potential jumps, in mV).

    Stochastic spike generation
    ...........................

    * **With dead time** (``dead_time > 0``): At most one spike per step.
      A uniform random number is compared to the spike probability
      :math:`P(\text{spike}) = 1 - \exp(-\text{rate} \cdot h \cdot 10^{-3})`.
    * **Without dead time** (``dead_time == 0``): Multiple spikes per step
      drawn from a Poisson distribution with mean
      :math:`\text{rate} \cdot h \cdot 10^{-3}`.

    The factor :math:`10^{-3}` converts from Hz·ms to a dimensionless quantity.

    Numerical integration and update order
    ......................................

    NEST integrates this model with exact (analytic) propagators. The
    discrete-time update order per simulation step is:

    1. Update membrane potential via exact propagator (including external
       current and synaptic delta inputs).
    2. Decay all adaptation elements and compute total :math:`E_\mathrm{sfa}`.
    3. If not refractory: compute effective potential
       :math:`V' = V_\mathrm{m} - E_\mathrm{sfa}`, compute instantaneous
       rate, draw random number and potentially emit spike(s). If spike:
       jump adaptation elements, optionally reset :math:`V_\mathrm{m}`,
       set dead time counter.
       If refractory: decrement dead time counter.
    4. Store external current input for the next step.

    .. note::

       The membrane potential in this model is relative to the resting
       potential (V_m = 0 at rest), unlike ``iaf_psc_delta`` which uses
       absolute potentials.

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
    ``tau_m``            10.0 ms             :math:`\tau_\mathrm{m}`            Membrane time constant
    ``C_m``              250.0 pF            :math:`C_\mathrm{m}`              Membrane capacitance
    ``dead_time``        1.0 ms                                                  Duration of the dead time (refractory period)
    ``dead_time_random`` False                                                   Whether to draw random dead time after each spike
    ``dead_time_shape``  1                                                       Shape parameter of dead time gamma distribution
    ``with_reset``       True                                                    Whether to reset V_m to 0 after each spike
    ``tau_sfa``          () ms               :math:`\tau_{\mathrm{sfa},i}`      Adaptive threshold time constants (list)
    ``q_sfa``            () mV               :math:`q_{\mathrm{sfa},i}`        Adaptive threshold jump sizes (list)
    ``c_1``              0.0 Hz/mV           :math:`c_1`                        Slope of linear part of transfer function
    ``c_2``              1.238 Hz            :math:`c_2`                        Prefactor of exponential part of transfer function
    ``c_3``              0.25 1/mV           :math:`c_3`                        Coefficient of exponential nonlinearity
    ``I_e``              0.0 pA              :math:`I_\mathrm{e}`              Constant external input current
    ``t_ref_remaining``  0.0 ms                                                  Remaining dead time at simulation start
    ``rng_key``          None                                                    JAX PRNG key for stochastic spiking
    ``V_initializer``    Constant(0 mV)                                          Initializer for membrane potential (relative to rest)
    ``spk_fun``          ReluGrad()                                              Surrogate spike function
    ``spk_reset``        ``'hard'``                                              Reset mode; hard reset matches NEST
    ==================== =================== =================================== =====================================================

    State Variables
    ---------------

    ============================== ===========================================
    **State variable**             **Description**
    ============================== ===========================================
    ``V``                          Membrane potential :math:`V_\mathrm{m}` (relative to rest)
    ``E_sfa``                      Adaptive threshold :math:`E_\mathrm{sfa}`
    ``refractory_step_count``      Remaining dead time grid steps
    ``I_stim``                     Buffered current applied in next step
    ``last_spike_time``            Last spike time
    ============================== ===========================================

    Notes
    -----

    - Default parameter values match NEST C++ source for ``pp_psc_delta``,
      which are based on Jolivet et al. (2006) [2]_.
    - ``tau_sfa`` and ``q_sfa`` default to empty tuples (no adaptation).
      In NEST, the C++ defaults of ``tau_sfa=34.0`` and ``q_sfa=0.0`` are
      immediately cleared in the constructor, resulting in empty vectors.
    - The membrane potential is stored relative to the resting potential
      (V_m = 0 at rest). The recordable ``V_m`` in NEST corresponds to
      ``self.V.value``.
    - The recordable ``E_sfa`` corresponds to the sum of all adaptation
      elements.

    References
    ----------

    .. [1] Cardanobile S, Rotter S (2010). Multiplicatively interacting point
           processes and applications to neural modeling. Journal of
           Computational Neuroscience 28(2):267-284.
           DOI: https://doi.org/10.1007/s10827-009-0204-0
    .. [2] Jolivet R, Rauch A, Luescher H-R, Gerstner W (2006). Predicting
           spike timing of neocortical pyramidal neurons by simple threshold
           models. Journal of Computational Neuroscience 21:35-49.
           DOI: https://doi.org/10.1007/s10827-006-7074-5
    .. [3] Pozzorini C, Naud R, Mensi S, Gerstner W (2013). Temporal whitening
           by power-law adaptation in neocortical neurons. Nature Neuroscience
           16:942-948.
           DOI: https://doi.org/10.1038/nn.3431
    .. [4] Grytskyy D, Tetzlaff T, Diesmann M, Helias M (2013). A unified view
           on weakly correlated recurrent networks. Frontiers in Computational
           Neuroscience, 7:131.
           DOI: https://doi.org/10.3389/fncom.2013.00131
    .. [5] Deger M, Schwalger T, Naud R, Gerstner W (2014). Fluctuations and
           information filtering in coupled populations of spiking neurons with
           adaptation. Physical Review E 90:6, 062704.
           DOI: https://doi.org/10.1103/PhysRevE.90.062704
    .. [6] Gerstner W, Kistler WM, Naud R, Paninski L (2014). Neuronal
           Dynamics: From single neurons to networks and models of cognition.
           Cambridge University Press.
    .. [7] NEST Simulator ``pp_psc_delta`` model documentation and C++ source:
           ``models/pp_psc_delta.h`` and ``models/pp_psc_delta.cpp``.

    See Also
    --------
    iaf_psc_delta, gif_psc_exp
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau_m: ArrayLike = 10.0 * u.ms,
        C_m: ArrayLike = 250.0 * u.pF,
        dead_time: float = 1.0,  # ms, plain float as in NEST
        dead_time_random: bool = False,
        dead_time_shape: int = 1,
        with_reset: bool = True,
        tau_sfa: Sequence[float] = (),  # ms values
        q_sfa: Sequence[float] = (),  # mV values
        c_1: float = 0.0,  # Hz/mV
        c_2: float = 1.238,  # Hz
        c_3: float = 0.25,  # 1/mV
        I_e: ArrayLike = 0.0 * u.pA,
        t_ref_remaining: float = 0.0,  # ms
        rng_key: Optional[jax.Array] = None,
        V_initializer: Callable = braintools.init.Constant(0.0 * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Membrane parameters
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        # Dead time parameters (stored as plain Python scalars)
        self.dead_time = float(dead_time)
        self.dead_time_random = bool(dead_time_random)
        self.dead_time_shape = int(dead_time_shape)
        self.with_reset = bool(with_reset)

        # Transfer function coefficients
        self.c_1 = float(c_1)
        self.c_2 = float(c_2)
        self.c_3 = float(c_3)

        # Initial dead time remaining
        self.t_ref_remaining = float(t_ref_remaining)

        # Adaptation parameters (stored as plain Python tuples of floats)
        self.tau_sfa = tuple(float(x) for x in tau_sfa)
        self.q_sfa = tuple(float(x) for x in q_sfa)

        if len(self.tau_sfa) != len(self.q_sfa):
            raise ValueError(
                f"'tau_sfa' and 'q_sfa' must have the same length. "
                f"Got {len(self.tau_sfa)} and {len(self.q_sfa)}."
            )

        # RNG key for stochastic spiking
        self._rng_key = rng_key

        # Initializers
        self.V_initializer = V_initializer

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
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if self.dead_time < 0.0:
            raise ValueError('Dead time must not be negative.')
        if self.dead_time_shape < 1:
            raise ValueError('Shape of the dead time gamma distribution must not be smaller than 1.')
        if self.t_ref_remaining < 0.0:
            raise ValueError('Remaining refractory time must not be negative.')
        if self.c_3 < 0.0:
            raise ValueError('c_3 must not be negative.')
        for tau in self.tau_sfa:
            if tau <= 0.0:
                raise ValueError('All SFA time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.V = brainstate.HiddenState(V)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        # Adaptation state: q_elems array
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self._q_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._q_val = np.zeros(v_shape, dtype=np.float64)  # total E_sfa

        # Initialize remaining dead time from parameter
        if self.t_ref_remaining > 0.0:
            dt_q = brainstate.environ.get_dt()
            h = float(u.math.asarray(dt_q / u.ms))
            r_init = int(round(self.t_ref_remaining / h))
            r_arr = np.full(v_shape, r_init, dtype=np.int32)
            self.refractory_step_count.value = jnp.asarray(r_arr, dtype=jnp.int32)

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self._q_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._q_val = np.zeros(v_shape, dtype=np.float64)

        if self.t_ref_remaining > 0.0:
            dt_q = brainstate.environ.get_dt()
            h = float(u.math.asarray(dt_q / u.ms))
            r_init = int(round(self.t_ref_remaining / h))
            r_arr = np.full(v_shape, r_init, dtype=np.int32)
            self.refractory_step_count.value = jnp.asarray(r_arr, dtype=jnp.int32)

        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        # For a stochastic model, we use V directly scaled by a reasonable factor
        v_scaled = V / (1.0 * u.mV)
        return self.spk_fun(v_scaled)

    def update(self, x=0.0 * u.pA):
        """Update neuron state for one simulation step.

        Parameters
        ----------
        x : Quantity, optional
            External current input (pA). Default is 0.

        Returns
        -------
        spike : array
            Spike output (float array; >0 indicates spike).
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))  # dt in ms as float

        v_shape = self.V.value.shape

        # Extract state variables as numpy arrays (V_m is relative to resting potential)
        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape).copy()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        ).copy()
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape).copy()

        # Extract parameters as numpy arrays
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)

        # Compute propagator coefficients (exact integration)
        P33 = np.exp(-h / tau_m)
        P30 = 1.0 / C_m * (1.0 - P33) * tau_m

        # Compute exponential decay factors for adaptation
        P_sfa = [math.exp(-h / tau) for tau in self.tau_sfa]

        # Dead time parameters
        dead_time = self.dead_time
        # If dead_time > 0 but < h, clamp to h (matching NEST)
        if dead_time != 0.0 and dead_time < h:
            dead_time = h

        if not self.dead_time_random and dead_time > 0.0:
            # Fixed dead time: convert to steps (matching NEST Time::ms -> get_steps)
            dead_time_counts = int(round(dead_time / h))
        else:
            dead_time_counts = 0

        if self.dead_time_random and dead_time > 0.0:
            dt_rate = self.dead_time_shape / dead_time
        else:
            dt_rate = 0.0

        # Get delta (spike) inputs: these are voltage jumps in mV
        delta_v = self._to_numpy(
            self.sum_delta_inputs(u.math.zeros(v_shape) * u.mV), u.mV
        )

        # Get external current for NEXT step (NEST ring buffer semantics)
        new_i_stim = self._broadcast_to_state(
            self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape
        )

        # Advance RNG state for this step
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(jax.random.uniform(subkey, shape=v_shape), dtype=np.float64)

        # For Poisson mode (dead_time == 0), we need Poisson random draws
        # We'll compute them per-element in the loop if needed

        spike_mask = np.zeros(v_shape, dtype=bool)
        n_spikes_arr = np.zeros(v_shape, dtype=np.int64)

        for idx in np.ndindex(v_shape):
            # ---- Step 1: Update membrane potential via exact propagator ----
            V[idx] = P30[idx] * (i_stim[idx] + I_e[idx]) + P33[idx] * V[idx] + delta_v[idx]

            # ---- Step 2: Decay adaptation elements and compute total E_sfa ----
            q_total = 0.0
            if self._q_elems is not None:
                for i in range(len(self.tau_sfa)):
                    self._q_elems[i][idx] *= P_sfa[i]
                    q_total += self._q_elems[i][idx]
            self._q_val[idx] = q_total

            # ---- Step 3: Spike check / refractory ----
            if r[idx] == 0:
                # Neuron not refractory
                V_eff = V[idx] - q_total

                rate = self.c_1 * V_eff + self.c_2 * math.exp(self.c_3 * V_eff)

                if rate > 0.0:
                    n_spikes = 0

                    if dead_time > 0.0:
                        # With dead time: at most 1 spike per step
                        spike_prob = -math.expm1(-rate * h * 1e-3)
                        if rand_vals[idx] <= spike_prob:
                            n_spikes = 1
                    else:
                        # Without dead time: Poisson-distributed spikes
                        # Use numpy for Poisson draws
                        lam_poisson = rate * h * 1e-3
                        # Use a deterministic approach based on JAX random
                        n_spikes = int(np.random.RandomState(
                            int(rand_vals[idx] * 2**31)
                        ).poisson(lam_poisson))

                    if n_spikes > 0:
                        spike_mask[idx] = True
                        n_spikes_arr[idx] = n_spikes

                        # Set dead time
                        if self.dead_time_random:
                            # Gamma-distributed dead time
                            gamma_sample = np.random.RandomState(
                                int(rand_vals[idx] * 2**30 + 1)
                            ).gamma(self.dead_time_shape)
                            r[idx] = max(1, int(round(gamma_sample / dt_rate / h)))
                        elif dead_time > 0.0:
                            r[idx] = dead_time_counts

                        # Jump adaptation elements
                        if self._q_elems is not None:
                            for i in range(len(self.q_sfa)):
                                self._q_elems[i][idx] += self.q_sfa[i] * n_spikes

                        # Reset membrane potential if applicable
                        if self.with_reset:
                            V[idx] = 0.0
            else:
                # Within dead time: decrement counter
                r[idx] -= 1

        # ---- Step 4: Update state ----
        self.V.value = V * u.mV
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)
