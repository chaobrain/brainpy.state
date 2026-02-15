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

from brainpy_state._base import Dynamics

__all__ = [
    'gif_pop_psc_exp',
]


class gif_pop_psc_exp(Dynamics):
    r"""Population of generalized integrate-and-fire neurons (GIF) with
    exponential postsynaptic currents and adaptation.

    Description
    -----------

    ``gif_pop_psc_exp`` simulates a population of spike-response model neurons
    with multi-timescale adaptation and exponential postsynaptic currents.
    It directly models the population activity (sum of all spikes) without
    explicitly representing each individual neuron, following the algorithm of
    Schwalger et al. (2017) [1]_.

    This is a brainpy.state re-implementation of the NEST simulator model of
    the same name, using NEST-standard parameterization.

    The single neuron model is defined by the hazard function:

    .. math::

       h(t) = \lambda_0 \exp\frac{V_m(t) - E_{\mathrm{sfa}}(t)}{\Delta_V}

    After each spike, the membrane potential :math:`V_m` is reset to
    :math:`V_{\mathrm{reset}}`. Spike frequency adaptation is implemented by a
    set of exponentially decaying traces, the sum of which is
    :math:`E_{\mathrm{sfa}}`. Upon a spike, each of the adaptation traces is
    incremented by the respective :math:`q_{\mathrm{sfa}}` and decays with the
    respective time constant :math:`\tau_{\mathrm{sfa}}`.

    Population dynamics algorithm
    .............................

    The model uses the algorithm from Figures 11 and 12 of [1]_ to simulate
    the population activity directly:

    1. **Membrane update**: Exact exponential integration of the subthreshold
       membrane dynamics with exponential postsynaptic currents:

       .. math::

          C_m \frac{dV}{dt} = -\frac{C_m}{\tau_m}(V - E_L) + I_{\mathrm{syn,ex}}
              + I_{\mathrm{syn,in}} + I_e + I_{\mathrm{stim}}

    2. **Adaptation update**: Multi-timescale adaptation state evolves as:

       .. math::

          E_{\mathrm{sfa}} = V_{T^*} + \sum_j Q_{30K,j} \cdot g_j

       where :math:`g_j` tracks the convolution of past spike rates with the
       adaptation kernel.

    3. **Population escape rate**: For each refractory cohort, compute the
       hazard-based escape (firing) probability. Free (non-refractory) neurons
       and neurons emerging from refractoriness are tracked separately.

    4. **Stochastic spike generation**: The expected number of spikes is
       computed from the population escape rates, and the actual spike count
       is drawn from either a binomial or Poisson distribution.

    5. **History buffer rotation**: Spike counts, survival counts, mean
       membrane potentials, and variances of each refractory cohort are
       maintained in rotating circular buffers.

    Synaptic currents
    .................

    Exponential postsynaptic currents follow first-order dynamics:

    .. math::

       \tau_{\mathrm{syn}} \frac{dy}{dt} = -y

    Integrated exactly per time step:

    .. math::

       y(t+h) = y_{\infty} + (y(t) - y_{\infty}) \cdot e^{-h/\tau_{\mathrm{syn}}}

    The membrane contribution from synaptic currents is computed via the exact
    propagator for the coupled linear system of membrane and synaptic dynamics.

    Connecting two population models corresponds to full connectivity of every
    neuron in each population. An approximation of random connectivity can be
    implemented by connecting populations using a ``bernoulli_synapse``.

    .. note::

       As ``gif_pop_psc_exp`` represents many neurons in one node, it may
       generate many spikes per time step. The ``n_spikes`` state variable
       records the number of spikes emitted by the population each step.

    .. note::

       The computational cost of this model is largely independent of the
       number N of neurons represented.

    Parameters
    ----------

    ==================== ======================= =================================== =================================================
    **Parameter**        **Default**             **Math equivalent**                 **Description**
    ==================== ======================= =================================== =================================================
    ``in_size``          (required)                                                  Population shape (scalar size)
    ``N``                100                     :math:`N`                           Number of neurons in the population
    ``tau_m``            20.0 ms                 :math:`\tau_m`                      Membrane time constant
    ``C_m``              250.0 pF                :math:`C_m`                         Membrane capacitance
    ``t_ref``            4.0 ms                  :math:`t_{\mathrm{ref}}`            Absolute refractory period
    ``lambda_0``         10.0 1/s                :math:`\lambda_0`                   Firing rate at threshold
    ``Delta_V``          2.0 mV                  :math:`\Delta_V`                    Noise level of escape rate
    ``E_L``              0.0 mV                  :math:`E_L`                         Resting/leak potential
    ``V_reset``          0.0 mV                  :math:`V_{\mathrm{reset}}`          Reset potential after spike
    ``V_T_star``         15.0 mV                 :math:`V_{T^*}`                     Baseline threshold level
    ``I_e``              0.0 pA                  :math:`I_e`                         Constant external DC current
    ``tau_syn_ex``       3.0 ms                  :math:`\tau_{\mathrm{syn,ex}}`      Excitatory synaptic time constant
    ``tau_syn_in``       6.0 ms                  :math:`\tau_{\mathrm{syn,in}}`      Inhibitory synaptic time constant
    ``tau_sfa``          (300.0,) ms             :math:`\tau_{\mathrm{sfa},j}`       Adaptation time constants (tuple)
    ``q_sfa``            (0.5,) mV               :math:`q_{\mathrm{sfa},j}`         Adaptation kernel amplitudes (tuple)
    ``len_kernel``       -1                                                          History kernel length (-1 = automatic)
    ``BinoRand``         True                                                        Use binomial (True) or Poisson (False) RNG
    ``rng_seed``         0                                                           Random number generator seed
    ==================== ======================= =================================== =================================================

    State Variables
    ---------------

    ========================== ===========================================
    **State variable**         **Description**
    ========================== ===========================================
    ``V_m``                    Mean membrane potential (mV)
    ``I_syn_ex``               Excitatory synaptic current (pA)
    ``I_syn_in``               Inhibitory synaptic current (pA)
    ``n_spikes``               Number of spikes in current step
    ``n_expect``               Expected number of spikes
    ``theta_hat``              Adaptive threshold (mV)
    ``y0``                     External current input state (pA)
    ========================== ===========================================

    Notes
    -----

    - Defaults follow NEST C++ source for ``gif_pop_psc_exp``.
    - ``lambda_0`` is specified in 1/s (as in NEST's Python interface); the
      conversion factor of 0.001 to 1/ms is applied in the update equations
      exactly as in NEST (the factor 0.0005 in the trapezoidal escape rate).
    - Synaptic spike weights are interpreted in current units; positive
      weights go to excitatory, negative to inhibitory channel.
    - The model does not inherit from ``Neuron`` because it represents a
      population, not a single spiking neuron with a surrogate gradient.

    =================== =============  =============================
    **Parameter translation to gif_psc_exp**
    ---------------------------------------------------------
    gif_pop_psc_exp      gif_psc_exp    relation
    tau_m                g_L            tau_m = C_m / g_L
    N                    ---            use N gif_psc_exp neurons
    =================== =============  =============================

    References
    ----------
    .. [1] Schwalger T, Deger M, Gerstner W (2017). Towards a theory of
           cortical columns: From spiking neurons to interacting neural
           populations of finite size. PLoS Computational Biology, 13(4),
           e1005507. https://doi.org/10.1371/journal.pcbi.1005507
    .. [2] NEST Simulator ``gif_pop_psc_exp`` model documentation and C++
           source: ``models/gif_pop_psc_exp.h`` and
           ``models/gif_pop_psc_exp.cpp``.

    See Also
    --------
    gif_cond_exp
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        N: int = 100,
        tau_m: float = 20.0,      # ms
        C_m: float = 250.0,       # pF
        t_ref: float = 4.0,       # ms
        lambda_0: float = 10.0,   # 1/s
        Delta_V: float = 2.0,     # mV
        E_L: float = 0.0,         # mV
        V_reset: float = 0.0,     # mV
        V_T_star: float = 15.0,   # mV
        I_e: float = 0.0,         # pA
        tau_syn_ex: float = 3.0,  # ms
        tau_syn_in: float = 6.0,  # ms
        tau_sfa: Sequence[float] = (300.0,),  # ms
        q_sfa: Sequence[float] = (0.5,),      # mV
        len_kernel: int = -1,
        BinoRand: bool = True,
        rng_seed: int = 0,
        name: str = None,
    ):
        super().__init__(in_size, name=name)

        # Store parameters as plain floats (NEST uses raw doubles, no units)
        self.N = int(N)
        self.tau_m = float(tau_m)
        self.C_m = float(C_m)
        self.t_ref = float(t_ref)
        self.lambda_0 = float(lambda_0)
        self.Delta_V = float(Delta_V)
        self.E_L = float(E_L)
        self.V_reset = float(V_reset)
        self.V_T_star = float(V_T_star)
        self.I_e = float(I_e)
        self.tau_syn_ex = float(tau_syn_ex)
        self.tau_syn_in = float(tau_syn_in)
        self.tau_sfa = tuple(float(x) for x in tau_sfa)
        self.q_sfa = tuple(float(x) for x in q_sfa)
        self.len_kernel = int(len_kernel)
        self.BinoRand = bool(BinoRand)
        self.rng_seed = int(rng_seed)

        self._validate_parameters()

    def _validate_parameters(self):
        if len(self.tau_sfa) != len(self.q_sfa):
            raise ValueError(
                f"'tau_sfa' and 'q_sfa' must have the same length. "
                f"Got {len(self.tau_sfa)} and {len(self.q_sfa)}."
            )
        if self.C_m <= 0:
            raise ValueError('Capacitance must be strictly positive.')
        if self.tau_m <= 0:
            raise ValueError('The membrane time constant must be strictly positive.')
        if self.tau_syn_ex <= 0 or self.tau_syn_in <= 0:
            raise ValueError('The synaptic time constants must be strictly positive.')
        for tau in self.tau_sfa:
            if tau <= 0:
                raise ValueError('All adaptation time constants must be strictly positive.')
        if self.N <= 0:
            raise ValueError('Number of neurons must be positive.')
        if self.lambda_0 < 0:
            raise ValueError('lambda_0 must not be negative.')
        if self.Delta_V <= 0:
            raise ValueError('Delta_V must be strictly positive.')
        if self.t_ref < 0:
            raise ValueError('Absolute refractory period cannot be negative.')

    def _adaptation_kernel(self, k: int, h: float) -> float:
        """Compute the adaptation kernel value at lag k time steps.

        See below Eq. (87) of [1]. No division by tau because the result
        must be in voltage units, matching q_sfa.
        """
        theta_tmp = 0.0
        for j in range(len(self.tau_sfa)):
            theta_tmp += self.q_sfa[j] * math.exp(-k * h / self.tau_sfa[j])
        return theta_tmp

    def _get_history_size(self, h: float) -> int:
        """Automatically determine a suitable history kernel size.

        See [1], Eq. (86) and Fig 11, Procedure GetHistoryLength.
        """
        tmax = 20000.0  # ms, maximum automatic kernel length
        k = int(tmax / h)
        kmin = int(5 * self.tau_m / h)
        while (self._adaptation_kernel(k, h) / self.Delta_V < 0.1) and k > kmin:
            k -= 1
        if k * h <= self.t_ref:
            k = int(self.t_ref / h) + 1
        return k

    def _escrate(self, x: float) -> float:
        """Escape rate (hazard function).

        h(t) = lambda_0 * exp(x / Delta_V)
        """
        return self.lambda_0 * math.exp(x / self.Delta_V)

    def init_state(self, batch_size: int = None, **kwargs):
        dt = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt / u.ms))
        self._h = h

        # Integration constants
        self._R = self.tau_m / self.C_m  # membrane resistance
        self._P22 = math.exp(-h / self.tau_m)
        self._P20 = self.tau_m / self.C_m * (1.0 - self._P22)
        self._P11_ex = math.exp(-h / self.tau_syn_ex)
        self._P11_in = math.exp(-h / self.tau_syn_in)

        # Determine kernel length
        len_kernel = self.len_kernel
        if len_kernel < 1:
            len_kernel = self._get_history_size(h)
        self._len_kernel = len_kernel

        # Refractory period in time steps (NEST uses Time::ms -> get_steps)
        # This matches NEST: V_.k_ref_ = Time( Time::ms( P_.t_ref_ ) ).get_steps()
        self._k_ref = int(round(self.t_ref / h))

        # Initialize population state variables
        self._lambda_free = 0.0

        # History buffers (rotating), length = len_kernel
        self._n = np.zeros(len_kernel, dtype=np.float64)      # spike counts
        self._m = np.zeros(len_kernel, dtype=np.float64)      # survival
        self._v_buf = np.zeros(len_kernel, dtype=np.float64)   # variance of survivors
        self._u = np.zeros(len_kernel, dtype=np.float64)       # mean of survivors
        self._lambda_buf = np.zeros(len_kernel, dtype=np.float64)  # escape rates

        # Adaptation kernel values
        self._theta = np.zeros(len_kernel, dtype=np.float64)
        self._theta_tld = np.zeros(len_kernel, dtype=np.float64)

        # Procedure InitPopulations, see Fig. 11 of [1]
        for k in range(len_kernel):
            theta_tmp = self._adaptation_kernel(len_kernel - k, h)
            self._theta[k] = theta_tmp                                   # line 4
            self._theta_tld[k] = (
                self.Delta_V * (1.0 - math.exp(-theta_tmp / self.Delta_V))
                / float(self.N)
            )  # line 5

        # InitPopulations, line 7: last entry gets N
        self._n[len_kernel - 1] = float(self.N)
        self._m[len_kernel - 1] = float(self.N)

        # InitPopulations, line 8
        self._x = 0.0
        self._z = 0.0
        self._k0 = 0  # rotating index

        # Adaptation variables
        self._Q30 = np.array([math.exp(-h / tau) for tau in self.tau_sfa], dtype=np.float64)
        self._Q30K = np.array([
            self.q_sfa[j] * self.tau_sfa[j] * math.exp(-h * len_kernel / self.tau_sfa[j])
            for j in range(len(self.tau_sfa))
        ], dtype=np.float64)
        self._g = np.zeros(len(self.tau_sfa), dtype=np.float64)

        # Observable state variables
        self._V_m = 0.0        # mV
        self._I_syn_ex = 0.0   # pA
        self._I_syn_in = 0.0   # pA
        self._y0 = 0.0         # pA (external current state)
        self._n_expect = 0.0
        self._theta_hat = 0.0
        self._n_spikes = 0

        # RNG
        self._rng = np.random.RandomState(self.rng_seed)

    def reset_state(self, batch_size: int = None, **kwargs):
        self.init_state(batch_size=batch_size, **kwargs)

    def _draw_binomial(self, n_expect: float) -> int:
        """Draw a binomial random number of spikes, matching NEST."""
        p_bino = n_expect / self.N
        if p_bino >= 1.0:
            return self.N
        elif p_bino <= 0.0:
            return 0
        else:
            return int(self._rng.binomial(self.N, p_bino))

    def _draw_poisson(self, n_expect: float) -> int:
        """Draw a Poisson random number of spikes, matching NEST."""
        min_double = np.finfo(np.float64).tiny
        if n_expect > self.N:
            return self.N
        elif n_expect > min_double:
            # If probability of any spike is indistinguishable from that of
            # one spike, use Bernoulli instead of Poisson
            if 1.0 - (n_expect + 1.0) * math.exp(-n_expect) > min_double:
                n_t = int(self._rng.poisson(n_expect))
            else:
                n_t = int(self._rng.random() < n_expect)
            # Clip to [0, N]
            return max(0, min(n_t, self.N))
        else:
            return 0

    def update(self, x=0.0):
        """Advance the population model by one time step.

        Parameters
        ----------
        x : float
            External input current (pA). This corresponds to the sum of
            weighted spike inputs from other populations. Positive values
            go to excitatory channel, negative to inhibitory.

        Returns
        -------
        int
            Number of spikes emitted by the population in this step.
        """
        h = self._h

        # ================================================================
        # Main update routine, see Fig. 11 of [1]
        # ================================================================

        # line 6: membrane and synapse update
        h_tot = (self.I_e + self._y0) * self._P20 + self.E_L

        # Get input spikes from the ring buffer / external input
        # In NEST, spikes come from ring buffer weighted by synaptic weight.
        # Here we receive the total weighted input directly.
        # Separate excitatory and inhibitory components
        if isinstance(x, (int, float)):
            x_val = float(x)
        else:
            x_val = float(x)

        # Collect inputs: current inputs go to y0 for next step
        # Delta inputs (spikes) go to excitatory/inhibitory channels
        ex_input = 0.0
        in_input = 0.0

        if self._delta_inputs is not None:
            for key in tuple(self._delta_inputs.keys()):
                out = self._delta_inputs[key]
                if callable(out):
                    out = out()
                else:
                    self._delta_inputs.pop(key)
                val = float(out)
                if val > 0.0:
                    ex_input += val
                else:
                    in_input += val  # negative

        JNA_ex = ex_input / h
        JNA_in = in_input / h

        # Rescale inputs to voltage scale used in [1]
        JNA_ex *= self.tau_syn_ex / self.C_m
        JNA_in *= self.tau_syn_in / self.C_m

        # Translate synaptic currents into [1]'s definition
        JNy_ex = self._I_syn_ex / self.C_m
        JNy_in = self._I_syn_in / self.C_m

        # Membrane update (line 10 of [1])
        h_ex_tmpvar = (
            self.tau_syn_ex * self._P11_ex * (JNy_ex - JNA_ex)
            - self._P22 * (self.tau_syn_ex * JNy_ex - self.tau_m * JNA_ex)
        )
        h_in_tmpvar = (
            self.tau_syn_in * self._P11_in * (JNy_in - JNA_in)
            - self._P22 * (self.tau_syn_in * JNy_in - self.tau_m * JNA_in)
        )
        h_ex = self.tau_m * (JNA_ex + h_ex_tmpvar / (self.tau_syn_ex - self.tau_m))
        h_in = self.tau_m * (JNA_in + h_in_tmpvar / (self.tau_syn_in - self.tau_m))
        h_tot += h_ex + h_in

        # Update EPSCs & IPSCs (line 11 of [1])
        JNy_ex = JNA_ex + (JNy_ex - JNA_ex) * self._P11_ex
        JNy_in = JNA_in + (JNy_in - JNA_in) * self._P11_in

        # Store the updated currents, translated back
        self._I_syn_ex = JNy_ex * self.C_m
        self._I_syn_in = JNy_in * self.C_m

        # Set new input current (for next step)
        # Current inputs go to y0
        new_y0 = float(x_val)
        if self._current_inputs is not None:
            for key in tuple(self._current_inputs.keys()):
                out = self._current_inputs[key]
                if callable(out):
                    out = out()
                else:
                    self._current_inputs.pop(key)
                new_y0 += float(out)
        self._y0 = new_y0

        # ================================================================
        # Begin procedure UpdatePopulation, see Fig. 12 of [1]
        # ================================================================

        W_ = 0.0
        X_ = 0.0
        Y_ = 0.0
        Z_ = 0.0  # line 2
        self._theta_hat = self.V_T_star  # line 2, initialize theta

        # line 3: membrane potential update
        self._V_m = (self._V_m - self.E_L) * self._P22 + h_tot

        # Compute free adaptation state (lines 4-6)
        for j in range(len(self.tau_sfa)):
            g_j_tmp = (
                (1.0 - self._Q30[j]) * self._n[self._k0]
                / (float(self.N) * h)
            )
            self._g[j] = self._g[j] * self._Q30[j] + g_j_tmp  # line 5
            self._theta_hat += self._Q30K[j] * self._g[j]      # line 6

        # Compute free escape rate (line 8)
        lambda_tld = self._escrate(self._V_m - self._theta_hat)
        # line 9: trapezoidal escape probability for free neurons
        P_free = 1.0 - math.exp(-0.0005 * (self._lambda_free + lambda_tld) * h)
        self._lambda_free = lambda_tld  # line 10
        self._theta_hat -= self._n[0] * self._theta_tld[0]  # line 11

        # line 12: sum up all surviving neurons
        for k_marked in range(self._len_kernel):
            X_ += self._m[k_marked]

        # Use a local theta_hat to preserve S_.theta_hat_ for recording
        theta_hat_local = self._theta_hat

        # lines 13-27: loop over non-refractory cohorts
        for k_marked in range(self._len_kernel - self._k_ref):
            k = (self._k0 + k_marked) % self._len_kernel  # line 14
            theta = self._theta[k_marked] + theta_hat_local  # line 15
            theta_hat_local += self._n[k] * self._theta_tld[k_marked]  # line 16
            self._u[k] = (self._u[k] - self.E_L) * self._P22 + h_tot  # line 17
            lambda_tld = self._escrate(self._u[k] - theta)  # line 18

            P_lambda_ = 0.0005 * (lambda_tld + self._lambda_buf[k]) * h
            if P_lambda_ > 0.01:
                P_lambda_ = 1.0 - math.exp(-P_lambda_)  # line 20

            self._lambda_buf[k] = lambda_tld  # line 21
            Y_ += P_lambda_ * self._v_buf[k]  # line 22
            Z_ += self._v_buf[k]               # line 23
            W_ += P_lambda_ * self._m[k]       # line 24

            ompl = 1.0 - P_lambda_
            self._v_buf[k] = ompl * ompl * self._v_buf[k] + P_lambda_ * self._m[k]
            self._m[k] = ompl * self._m[k]  # line 26

        # line 28
        if (Z_ + self._z) > 0.0:
            P_Lambda_ = (Y_ + P_free * self._z) / (Z_ + self._z)
        else:
            P_Lambda_ = 0.0

        # line 29: expected number of spikes
        self._n_expect = W_ + P_free * self._x + P_Lambda_ * (self.N - X_ - self._x)

        # Draw random spike count
        if self.BinoRand:
            self._n_spikes = self._draw_binomial(self._n_expect)
        else:
            self._n_spikes = self._draw_poisson(self._n_expect)

        # line 31: update z
        ompf = 1.0 - P_free
        self._z = ompf * ompf * self._z + self._x * P_free + self._v_buf[self._k0]
        # line 32: update x
        self._x = self._x * ompf + self._m[self._k0]

        # lines 33-36: reset buffers at current position
        self._n[self._k0] = float(self._n_spikes)
        self._m[self._k0] = float(self._n_spikes)
        self._v_buf[self._k0] = 0.0
        self._u[self._k0] = self.V_reset
        self._lambda_buf[self._k0] = 0.0

        # End procedure UpdatePopulation

        # Shift rotating index (line 17 of Fig. 11)
        self._k0 = (self._k0 + 1) % self._len_kernel

        return self._n_spikes

    @property
    def V_m(self) -> float:
        """Mean membrane potential (mV)."""
        return self._V_m

    @property
    def I_syn_ex(self) -> float:
        """Excitatory synaptic current (pA)."""
        return self._I_syn_ex

    @property
    def I_syn_in(self) -> float:
        """Inhibitory synaptic current (pA)."""
        return self._I_syn_in

    @property
    def n_spikes(self) -> int:
        """Number of spikes in the current time step."""
        return self._n_spikes

    @property
    def n_expect(self) -> float:
        """Expected number of spikes in the current time step."""
        return self._n_expect

    @property
    def theta_hat(self) -> float:
        """Adaptive threshold for non-refractory neurons (mV)."""
        return self._theta_hat

    @property
    def y0(self) -> float:
        """External current input state (pA)."""
        return self._y0
