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
    'izhikevich',
]


class izhikevich(Neuron):
    r"""Izhikevich neuron model (NEST-compatible).

    Description
    -----------

    ``izhikevich`` is a brainpy.state re-implementation of the NEST simulator
    model of the same name, using NEST-standard parameterization.  It implements
    the simple spiking neuron model introduced by Izhikevich [1]_, which
    reproduces spiking and bursting behavior of known types of cortical neurons.

    Membrane potential evolution, spike emission, and refractoriness
    ................................................................

    The model is defined by the following differential equations:

    .. math::

       \frac{dV_{\text{m}}}{dt} = 0.04\, V_{\text{m}}^2 + 5\, V_{\text{m}}
       + 140 - U_{\text{m}} + I_{\text{e}}

    .. math::

       \frac{dU_{\text{m}}}{dt} = a\,(b\, V_{\text{m}} - U_{\text{m}})

    where :math:`V_{\text{m}}` is the membrane potential, :math:`U_{\text{m}}`
    is the recovery variable, and :math:`I_{\text{e}}` is the total input
    current (external constant current plus synaptic current).

    A spike is emitted when :math:`V_{\text{m}}` reaches the threshold
    :math:`V_{\text{th}}`.  At this point the state variables are reset:

    .. math::

       &\text{if}\; V_m \geq V_{th}:\\
       &\quad V_m \leftarrow c\\
       &\quad U_m \leftarrow U_m + d

    Each incoming spike adds to :math:`V_{\text{m}}` by the synaptic weight
    associated with the spike (delta-coupling).

    Integration scheme
    ..................

    This model offers two forms of Euler integration, selected by the boolean
    parameter ``consistent_integration``:

    * ``consistent_integration = True`` *(default)* — standard forward Euler.
      Both :math:`V_{\text{m}}` and :math:`U_{\text{m}}` are updated based on
      their values at the *beginning* of the time step.

    * ``consistent_integration = False`` — the numerics published in [1]_.
      The membrane potential is updated in two half-steps of size
      :math:`h/2`, and the recovery variable uses the *updated*
      :math:`V_{\text{m}}`.  Recommended only for replicating published
      results; use ``h = 1.0 ms`` for consistency.

    For a detailed analysis of the numerical differences, see [2]_.

    Synaptic input
    ..............

    Synaptic input enters via two channels:

    * **Spike (delta) input** — delivered through ``add_delta_input`` or the
      ``delta`` keyword; added directly to :math:`V_{\text{m}}` at the
      integration step.
    * **Current input** — delivered through the ``x`` argument of
      :meth:`update`.  Following NEST ring-buffer semantics, the current
      applied at step *k* takes effect at step *k + 1* (one-step delay).

    Parameters
    ----------

    The following parameters can be set. Default values match the NEST simulator.

    ========================== =================== ============================== =================================================================
    **Parameter**              **Default**         **Math equivalent**            **Description**
    ========================== =================== ============================== =================================================================
    ``in_size``                (required)                                         Size of the input / number of neurons
    ``a``                      0.02                :math:`a`                      Time scale of the recovery variable :math:`U_{\text{m}}`
    ``b``                      0.2                 :math:`b`                      Sensitivity of :math:`U_{\text{m}}` to :math:`V_{\text{m}}`
    ``c``                      -65 mV              :math:`c`                      After-spike reset value of :math:`V_{\text{m}}`
    ``d``                      8 mV                :math:`d`                      After-spike increment of :math:`U_{\text{m}}`
    ``I_e``                    0 pA                :math:`I_{\text{e}}`           Constant input current (R=1)
    ``V_th``                   30 mV               :math:`V_{\text{th}}`          Spike threshold
    ``V_min``                  None                :math:`V_{\text{min}}`         Absolute lower bound for :math:`V_{\text{m}}` (``None`` = no bound)
    ``consistent_integration`` ``True``                                           Use standard forward Euler (True) or published numerics (False)
    ``V_initializer``          Constant(-65 mV)                                   Initializer for :math:`V_{\text{m}}`
    ``U_initializer``          ``None``                                           Initializer for :math:`U_{\text{m}}` (default: :math:`b \cdot V_0`)
    ``spk_fun``                ReluGrad()                                         Surrogate gradient function for spike generation
    ``spk_reset``              ``'hard'``                                         Reset mode; NEST behavior is hard reset at spike
    ========================== =================== ============================== =================================================================

    Attributes
    ----------
    V : HiddenState
        Membrane potential :math:`V_{\text{m}}` (mV).
    U : HiddenState
        Recovery variable :math:`U_{\text{m}}` (mV).
    I : ShortTermState
        Buffered input current from the previous step (pA, one-step delayed).

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an izhikevich neuron with regular spiking parameters
    >>> neuron = brainpy.state.izhikevich(1, a=0.02, b=0.2, c=-65*u.mV, d=8*u.mV)
    >>>
    >>> # Initialize the state
    >>> neuron.init_state()
    >>>
    >>> # Apply an input current and update the neuron state
    >>> with brainstate.environ.context(dt=1.0*u.ms, t=0.0*u.ms):
    ...     spikes = neuron.update(x=10.0*u.pA)

    References
    ----------
    .. [1] Izhikevich EM. (2003). Simple model of spiking neurons. IEEE
           Transactions on Neural Networks, 14:1569–1572.
           DOI: https://doi.org/10.1109/TNN.2003.820440
    .. [2] Pauli R, Weidel P, Kunkel S, Morrison A (2018). Reproducing
           polychronization: A guide to maximizing the reproducibility of
           spiking network models. Frontiers in Neuroinformatics, 12.
           DOI: https://doi.org/10.3389/fninf.2018.00046

    See Also
    --------
    iaf_psc_delta : Leaky integrate-and-fire with delta-shaped PSCs
    mat2_psc_exp : Multi-timescale adaptive threshold with exponential PSCs
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        a: ArrayLike = 0.02,
        b: ArrayLike = 0.2,
        c: ArrayLike = -65. * u.mV,
        d: ArrayLike = 8. * u.mV,
        I_e: ArrayLike = 0. * u.pA,
        V_th: ArrayLike = 30. * u.mV,
        V_min: ArrayLike = None,
        consistent_integration: bool = True,
        V_initializer: Callable = braintools.init.Constant(-65. * u.mV),
        U_initializer: Callable = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Parameters (broadcast to varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.c = braintools.init.param(c, self.varshape)
        self.d = braintools.init.param(d, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_min = V_min
        self.consistent_integration = consistent_integration
        self.V_initializer = V_initializer
        self.U_initializer = U_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize state variables.

        Parameters
        ----------
        batch_size : int, optional
            If provided, all state variables will have shape
            ``(batch_size, *varshape)``.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        if self.U_initializer is not None:
            U = braintools.init.param(self.U_initializer, self.varshape, batch_size)
        else:
            # NEST default: u_ = b * v_  (dimensionless b times V in mV)
            U = self.b * V
        self.V = brainstate.HiddenState(V)
        self.U = brainstate.HiddenState(U)
        # Buffered input current (one-step delay, matching NEST ring buffer)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        self.I = brainstate.ShortTermState(zeros * u.pA)

    def get_spike(self, V: ArrayLike = None):
        """Compute spike output using the surrogate gradient function.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential to test. Defaults to ``self.V.value``.

        Returns
        -------
        ArrayLike
            Surrogate-differentiable spike indicator.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.c)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.pA):
        """Advance the neuron state by one simulation step.

        This method mirrors the NEST ``izhikevich::update`` function exactly:

        1. Read current state (V_old, U_old) and buffered current I.
        2. Integrate V and U using forward Euler (or published half-step scheme).
        3. Apply the lower bound ``V_min``.
        4. Detect threshold crossing and apply reset (V → c, U += d).
        5. Buffer the new external current for the next step (one-step delay).
        6. Return surrogate-differentiable spike output.

        Parameters
        ----------
        x : ArrayLike
            External current input in pA.  This current takes effect at the
            *next* simulation step (NEST ring-buffer one-step delay).

        Returns
        -------
        ArrayLike
            Surrogate spike output for this step.
        """
        dt_q = brainstate.environ.get_dt()
        h = u.math.asarray(dt_q / u.ms, dtype=jnp.float64)

        # Read current state
        v_old = self.V.value
        u_old = self.U.value
        I_buf = self.I.value  # current from previous step

        # Strip units for the integration (NEST uses dimensionless arithmetic
        # internally; the quantities are in mV and pA with R=1)
        v = u.math.asarray(v_old / u.mV, dtype=jnp.float64)
        um = u.math.asarray(u_old / u.mV, dtype=jnp.float64)
        I_val = u.math.asarray((I_buf + self.I_e) / u.pA, dtype=jnp.float64)
        a = u.math.asarray(self.a, dtype=jnp.float64)
        b = u.math.asarray(self.b, dtype=jnp.float64)

        # Delta (spike) input — added directly to V
        delta_v = self.sum_delta_inputs(u.math.zeros_like(v_old))
        delta_v_raw = u.math.asarray(delta_v / u.mV, dtype=jnp.float64)

        if self.consistent_integration:
            # Standard forward Euler
            v_new = v + h * (0.04 * v * v + 5.0 * v + 140.0 - um + I_val) + delta_v_raw
            u_new = um + h * a * (b * v - um)
        else:
            # Published Izhikevich (2003) numerics: two half-step V updates,
            # then U update using the *new* V.
            I_syn = delta_v_raw
            v_new = v + h * 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - um + I_val + I_syn)
            v_new = v_new + h * 0.5 * (0.04 * v_new * v_new + 5.0 * v_new + 140.0 - um + I_val + I_syn)
            u_new = um + h * a * (b * v_new - um)

        # Lower bound on membrane potential
        if self.V_min is not None:
            v_min = u.math.asarray(self.V_min / u.mV, dtype=jnp.float64)
            v_new = jnp.maximum(v_new, v_min)

        # Convert back to quantities with units for spike detection
        V_new = v_new * u.mV
        U_new = u_new * u.mV

        # Threshold crossing and reset
        spike_cond = V_new >= self.V_th
        V_post = u.math.where(spike_cond, self.c, V_new)
        U_post = u.math.where(spike_cond, U_new + self.d, U_new)

        # Write back state
        self.V.value = V_post
        self.U.value = U_post

        # Buffer external current for the next step (one-step delay)
        self.I.value = self.sum_current_inputs(x, V_post)

        return self.get_spike(V_new)
