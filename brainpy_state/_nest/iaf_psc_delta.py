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

from typing import Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'iaf_psc_delta',
]


class iaf_psc_delta(Neuron):
    r"""Leaky integrate-and-fire neuron model with delta-shaped input currents.

    Description
    -----------

    ``iaf_psc_delta`` is a leaky integrate-and-fire neuron model with

    * a hard threshold,
    * a fixed refractory period,
    * Dirac delta (:math:`\delta`)-shaped synaptic input currents.

    This is a brainpy.state re-implementation of the NEST simulator model of the
    same name, using NEST-standard parameterization.

    Membrane potential evolution, spike emission, and refractoriness
    ................................................................

    The membrane potential evolves according to

    .. math::

       \frac{dV_\text{m}}{dt} = -\frac{V_{\text{m}} - E_\text{L}}{\tau_{\text{m}}}
       + \dot{\Delta}_{\text{syn}}
       + \frac{I_{\text{syn}} + I_\text{e}}{C_{\text{m}}}

    where the derivative of change in voltage due to synaptic input
    :math:`\dot{\Delta}_{\text{syn}}(t)` is discussed below and :math:`I_\text{e}` is
    a constant input current set as a model parameter.

    A spike is emitted at time step :math:`t^*=t_{k+1}` if

    .. math::

       V_\text{m}(t_k) < V_{th} \quad\text{and}\quad V_\text{m}(t_{k+1})\geq V_\text{th} \;.

    Subsequently,

    .. math::

       V_\text{m}(t) = V_{\text{reset}} \quad\text{for}\quad t^* \leq t < t^* + t_{\text{ref}} \;,

    that is, the membrane potential is clamped to :math:`V_{\text{reset}}` during the
    refractory period.

    Synaptic input
    ..............

    The change in membrane potential due to synaptic inputs can be formulated as:

    .. math::

       \dot{\Delta}_{\text{syn}}(t) = \sum_{j} w_j \sum_k \delta(t-t_j^k-d_j) \;,

    where :math:`j` indexes either excitatory (:math:`w_j > 0`) or inhibitory
    (:math:`w_j < 0`) presynaptic neurons, :math:`k` indexes the spike times of
    neuron :math:`j`, :math:`d_j` is the delay from neuron :math:`j`, and
    :math:`\delta` is the Dirac delta distribution. This implies that the jump in
    voltage upon a single synaptic input spike is

    .. math::

       \Delta_{\text{syn}} = w \;,

    where :math:`w` is the corresponding synaptic weight in mV.

    The change in voltage caused by the synaptic input can be interpreted as being
    caused by individual post-synaptic currents (PSCs) given by

    .. math::

       i_{\text{syn}}(t) = C_{\text{m}} \cdot w \cdot \delta(t) \;.

    As a consequence, the total charge :math:`q` transferred by a single PSC is

    .. math::

       q = \int_0^{\infty}  i_{\text{syn}}(t)\, dt = C_{\text{m}} \cdot w \;.

    By default, :math:`V_\text{m}` is not bounded from below. To limit
    hyperpolarization to biophysically plausible values, set parameter
    :math:`V_{\text{min}}` as lower bound of :math:`V_\text{m}`.

    .. note::

       This implementation uses exact integration (exponential Euler) [1]_, [2]_
       to integrate subthreshold membrane dynamics, which for this linear ODE is
       equivalent to the propagator-based approach used in the NEST C++
       implementation.

       Spikes arriving while the neuron is refractory are discarded (matching
       NEST's default ``refractory_input=false``).

    Parameters
    ----------

    The following parameters can be set. Default values match the NEST simulator.

    ==================== ================== =============================== ====================================================
    **Parameter**        **Default**        **Math equivalent**             **Description**
    ==================== ================== =============================== ====================================================
    ``in_size``          (required)                                         Size of the input / number of neurons
    ``E_L``              -70 mV             :math:`E_\text{L}`              Resting membrane potential
    ``C_m``              250 pF             :math:`C_{\text{m}}`            Capacitance of the membrane
    ``tau_m``            10 ms              :math:`\tau_{\text{m}}`         Membrane time constant
    ``t_ref``            2 ms               :math:`t_{\text{ref}}`          Duration of refractory period
    ``V_th``             -55 mV             :math:`V_{\text{th}}`           Spike threshold
    ``V_reset``          -70 mV             :math:`V_{\text{reset}}`        Reset potential of the membrane
    ``I_e``              0 pA               :math:`I_\text{e}`              Constant input current
    ``V_min``            None               :math:`V_{\text{min}}`          Absolute lower value for the membrane potential
    ``V_initializer``    Constant(-70 mV)                                   Initializer for the membrane potential state
    ``spk_fun``          ReluGrad()                                         Surrogate gradient function for spike generation
    ``spk_reset``        ``'hard'``                                         Reset mode. NEST behavior is hard reset at spike.
    ``refractory_input`` ``False``                                          If True, integrate refractory-arriving spikes at refractory end
    ``ref_var``          ``False``                                          If True, expose boolean refractory state variable
    ==================== ================== =============================== ====================================================

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    last_spike_time : ShortTermState
        Time of the last spike, used to implement the refractory period.
    refractory : HiddenState
        Neuron refractory state (only present if ``ref_var=True``).

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an iaf_psc_delta neuron layer with 10 neurons
    >>> neuron = brainpy.state.iaf_psc_delta(10, tau_m=10*u.ms, t_ref=2*u.ms)
    >>>
    >>> # Initialize the state
    >>> neuron.init_state(batch_size=1)
    >>>
    >>> # Apply an input current and update the neuron state
    >>> with brainstate.environ.context(dt=0.1*u.ms, t=0.0*u.ms):
    ...     spikes = neuron.update(x=500.*u.pA)

    References
    ----------
    .. [1] Rotter S, Diesmann M (1999). Exact simulation of time-invariant linear
           systems with applications to neuronal modeling. Biological Cybernetics
           81:381-402. DOI: https://doi.org/10.1007/s004220050570
    .. [2] Diesmann M, Gewaltig M-O, Rotter S, & Aertsen A (2001). State space
           analysis of synchronous spiking in cortical neural networks.
           Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X

    See Also
    --------
    LIF : Leaky integrate-and-fire with current-based synapses
    LIFRef : Leaky integrate-and-fire with refractory period (brainpy parameterization)
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 250. * u.pF,
        tau_m: ArrayLike = 10. * u.ms,
        t_ref: ArrayLike = 2. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -70. * u.mV,
        I_e: ArrayLike = 0. * u.pA,
        V_min: ArrayLike = None,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        refractory_input: bool = False,
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_min = V_min
        self.V_initializer = V_initializer
        self.refractory_input = refractory_input
        self.ref_var = ref_var

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.V = brainstate.HiddenState(V)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.refractory_spike_buffer = brainstate.ShortTermState(u.math.zeros_like(V))
        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        # NEST converts refractory duration to grid steps by rounding up to the
        # next simulation step.
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def update(self, x=0. * u.pA):
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        last_v = self.V.value
        ref_steps = self.refractory_step_count.value

        # Exact subthreshold propagation for one fixed simulation step.
        decay = u.math.exp(-dt / self.tau_m)
        i_total = self.sum_current_inputs(self.I_e + x, last_v)
        v_candidate = self.E_L + (last_v - self.E_L) * decay + (i_total / self.C_m) * self.tau_m * (1. - decay)
        delta_v = self.sum_delta_inputs(u.math.zeros_like(last_v))
        v_candidate = v_candidate + delta_v

        if self.refractory_input:
            v_candidate = v_candidate + self.refractory_spike_buffer.value

        if self.V_min is not None:
            v_candidate = u.math.maximum(v_candidate, self.V_min)

        not_refractory = ref_steps == 0
        v_post = u.math.where(not_refractory, v_candidate, last_v)

        if self.refractory_input:
            refr_decay = u.math.exp(-ref_steps * dt / self.tau_m)
            self.refractory_spike_buffer.value = u.math.where(
                not_refractory,
                u.math.zeros_like(self.refractory_spike_buffer.value),
                self.refractory_spike_buffer.value + delta_v * refr_decay
            )

        ref_steps = u.math.where(not_refractory, ref_steps, ref_steps - 1)

        spike_cond = v_post >= self.V_th
        self.refractory_step_count.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, self._refractory_counts(), ref_steps)
        )
        self.V.value = u.math.where(spike_cond, self.V_reset, v_post)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return self.get_spike(v_post)
