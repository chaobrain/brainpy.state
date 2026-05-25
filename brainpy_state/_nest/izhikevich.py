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
import saiunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

__all__ = [
    'izhikevich',
]


class izhikevich(NESTNeuron):
    r"""Izhikevich neuron model (NEST-compatible).

    This model is a brainpy.state re-implementation of the NEST simulator
    ``izhikevich`` model, using NEST-standard parameterization. It implements
    the simple spiking neuron model introduced by Izhikevich [1]_, which
    reproduces spiking and bursting behavior of known types of cortical neurons
    through a two-dimensional system of ordinary differential equations.

    **1. Mathematical Formulation**

    The model is defined by the following coupled differential equations:

    .. math::

       \frac{dV_{\text{m}}}{dt} = 0.04\, V_{\text{m}}^2 + 5\, V_{\text{m}}
       + 140 - U_{\text{m}} + I_{\text{e}}

    .. math::

       \frac{dU_{\text{m}}}{dt} = a\,(b\, V_{\text{m}} - U_{\text{m}})

    where:

    - :math:`V_{\text{m}}` is the membrane potential (mV)
    - :math:`U_{\text{m}}` is the recovery variable (mV), representing the
      combined effects of sodium channel inactivation and potassium channel
      activation
    - :math:`I_{\text{e}}` is the total input current (pA): external constant
      current plus synaptic current
    - :math:`a` is the time scale of the recovery variable (dimensionless)
    - :math:`b` describes the sensitivity of :math:`U_{\text{m}}` to
      subthreshold fluctuations of :math:`V_{\text{m}}` (dimensionless)

    **2. Spike Emission and Reset**

    A spike is emitted when :math:`V_{\text{m}}` reaches the threshold
    :math:`V_{\text{th}}`. At this point the state variables undergo an
    instantaneous reset:

    .. math::

       &\text{if}\; V_m \geq V_{th}:\\
       &\quad V_m \leftarrow c\\
       &\quad U_m \leftarrow U_m + d

    where:

    - :math:`c` is the after-spike reset value for :math:`V_{\text{m}}` (mV)
    - :math:`d` is the after-spike increment of :math:`U_{\text{m}}` (mV)

    Each incoming spike adds to :math:`V_{\text{m}}` by the synaptic weight
    associated with the spike (delta-coupling, instantaneous PSC).

    **3. Integration Scheme**

    This model offers two forms of Euler integration, selected by the boolean
    parameter ``consistent_integration``:

    - **Standard forward Euler** (``consistent_integration = True``, default):
      Both :math:`V_{\text{m}}` and :math:`U_{\text{m}}` are updated based on
      their values at the *beginning* of the time step:

      .. math::

         V_{n+1} &= V_n + h \cdot f(V_n, U_n, I_n) + \Delta V_{\text{syn}}\\
         U_{n+1} &= U_n + h \cdot a \cdot (b \cdot V_n - U_n)

      where :math:`h` is the time step and :math:`\Delta V_{\text{syn}}` is
      the delta synaptic input.

    - **Published Izhikevich (2003) numerics** (``consistent_integration =
      False``): The membrane potential is updated in two half-steps of size
      :math:`h/2`, and the recovery variable uses the *updated*
      :math:`V_{\text{m}}`:

      .. math::

         V_{\text{mid}} &= V_n + \frac{h}{2} \cdot f(V_n, U_n, I_n)\\
         V_{n+1} &= V_{\text{mid}} + \frac{h}{2} \cdot f(V_{\text{mid}}, U_n, I_n)\\
         U_{n+1} &= U_n + h \cdot a \cdot (b \cdot V_{n+1} - U_n)

      This scheme is recommended only for replicating published results and
      requires :math:`h = 1.0\,\text{ms}` for consistency with the original
      paper. For a detailed analysis of the numerical differences, see [2]_.

    **4. Synaptic Input**

    Synaptic input enters via two channels:

    - **Spike (delta) input** — delivered through ``add_delta_input()`` or the
      ``delta`` keyword; added directly to :math:`V_{\text{m}}` at the
      integration step as an instantaneous voltage jump.

    - **Current input** — delivered through the ``x`` argument of
      :meth:`update`. Following NEST ring-buffer semantics, the current
      applied at simulation step *k* takes effect at step *k + 1* (one-step
      delay). This is stored in the ``I`` state variable.

    **5. Physical Units and Numerical Assumptions**

    The original Izhikevich model uses dimensionless equations with implicit
    units. This implementation follows NEST conventions:

    - Membrane potential :math:`V_{\text{m}}` in mV
    - Recovery variable :math:`U_{\text{m}}` in mV
    - Input current :math:`I_{\text{e}}` in pA (with implicit resistance R=1)
    - Time constants :math:`a`, :math:`b` are dimensionless
    - Time step :math:`h` in ms

    The coefficients 0.04 and 5 in the voltage equation have implicit units
    that make the equation dimensionally consistent when :math:`V_{\text{m}}`
    is in mV and time in ms.

    **6. Computational Considerations**

    - The quadratic voltage term can lead to numerical instability if the time
      step is too large. Use :math:`h \leq 1.0\,\text{ms}` for stability.
    - The ``V_min`` parameter prevents unphysical negative voltage divergence.
    - The model uses ``float64`` precision internally for all integration
      steps to match NEST numerical accuracy.

    Parameters
    ----------
    in_size : int, tuple of int
        Number of neurons or shape of the neuron population. Determines the
        shape of all state variables and parameters (``varshape``).
    a : float, array_like, optional
        Time scale of the recovery variable :math:`U_{\text{m}}`.
        Dimensionless. Default: 0.02.
        Typical values: 0.02 (regular spiking), 0.1 (fast spiking).
    b : float, array_like, optional
        Sensitivity of :math:`U_{\text{m}}` to subthreshold fluctuations of
        :math:`V_{\text{m}}`. Dimensionless. Default: 0.2.
        Typical values: 0.2 (regular spiking), 0.25 (chattering).
    c : Quantity (voltage), array_like, optional
        After-spike reset value of :math:`V_{\text{m}}`. Default: -65 mV.
        Typical values: -65 mV (regular spiking), -50 mV (chattering).
    d : Quantity (voltage), array_like, optional
        After-spike increment of :math:`U_{\text{m}}`. Default: 8 mV.
        Typical values: 8 mV (regular spiking), 2 mV (fast spiking).
    I_e : Quantity (current), array_like, optional
        Constant external input current. Default: 0 pA.
        Positive values provide tonic excitation.
    V_th : Quantity (voltage), array_like, optional
        Spike threshold voltage. Default: 30 mV.
        NEST uses 30 mV as the practical threshold for the Izhikevich model.
    V_min : Quantity (voltage), array_like, optional
        Absolute lower bound for :math:`V_{\text{m}}`. Default: None (no bound).
        When set, prevents unphysical negative voltage divergence.
        Typical value: -100 mV.
    consistent_integration : bool, optional
        Integration scheme selector. Default: True.
        - True: standard forward Euler (recommended).
        - False: published Izhikevich (2003) half-step numerics (requires dt=1ms).
    V_initializer : callable, optional
        Initialization function for :math:`V_{\text{m}}`.
        Default: ``Constant(-65 mV)``.
        Must accept ``(shape, batch_size)`` and return voltage values.
    U_initializer : callable, optional
        Initialization function for :math:`U_{\text{m}}`.
        Default: None (uses :math:`U_0 = b \cdot V_0`, matching NEST).
        Must accept ``(shape, batch_size)`` and return voltage values.
    spk_fun : callable, optional
        Surrogate gradient function for differentiable spike generation.
        Default: ``ReluGrad()``.
        Must map ``(V - V_th) / scale`` to [0, 1] with defined gradient.
    spk_reset : str, optional
        Spike reset mode. Default: 'hard'.
        - 'hard': stop gradient at reset (matches NEST dynamics).
        - 'soft': allow gradient flow through reset.
    name : str, optional
        Name of the neuron population. Default: None.

    Parameter Mapping
    -----------------

    ========================== ========================== ============================== =================================================================
    **NEST Parameter**         **brainpy.state**          **Math Equivalent**            **Description**
    ========================== ========================== ============================== =================================================================
    ``a``                      ``a``                      :math:`a`                      Time scale of recovery variable :math:`U_{\text{m}}`
    ``b``                      ``b``                      :math:`b`                      Sensitivity of :math:`U_{\text{m}}` to :math:`V_{\text{m}}`
    ``c``                      ``c``                      :math:`c`                      After-spike reset value of :math:`V_{\text{m}}` (mV)
    ``d``                      ``d``                      :math:`d`                      After-spike increment of :math:`U_{\text{m}}` (mV)
    ``I_e``                    ``I_e``                    :math:`I_{\text{e}}`           Constant input current (pA)
    ``V_th``                   ``V_th``                   :math:`V_{\text{th}}`          Spike threshold (mV)
    ``V_min``                  ``V_min``                  :math:`V_{\text{min}}`         Lower bound for :math:`V_{\text{m}}` (mV, optional)
    ``consistent_integration`` ``consistent_integration`` --                             Forward Euler (True) vs. published numerics (False)
    ========================== ========================== ============================== =================================================================

    Attributes
    ----------
    V : HiddenState
        Membrane potential :math:`V_{\text{m}}` in mV. Shape: ``(*varshape,)``
        or ``(batch_size, *varshape)``.
    U : HiddenState
        Recovery variable :math:`U_{\text{m}}` in mV. Shape: ``(*varshape,)``
        or ``(batch_size, *varshape)``.
    I : ShortTermState
        Buffered input current from the previous time step in pA (one-step
        delayed ring buffer, matching NEST semantics). Shape: ``(*varshape,)``
        or ``(batch_size, *varshape)``.

    Examples
    --------
    **Example 1: Regular spiking (RS) neuron**

    .. code-block:: python

       >>> from brainpy import state as bp
       >>> import brainstate
       >>> import saiunit as u
       >>>
       >>> # Create a regular spiking neuron
       >>> neuron = bp.izhikevich(1, a=0.02, b=0.2, c=-65*u.mV, d=8*u.mV)
       >>> neuron.init_state()
       >>>
       >>> # Simulate with constant input
       >>> with brainstate.environ.context(dt=1.0*u.ms):
       ...     spikes = []
       ...     for _ in range(100):
       ...         spk = neuron.update(x=10.0*u.pA)
       ...         spikes.append(spk)

    **Example 2: Fast spiking (FS) neuron**

    .. code-block:: python

       >>> # Create a fast spiking neuron
       >>> neuron = bp.izhikevich(1, a=0.1, b=0.2, c=-65*u.mV, d=2*u.mV)
       >>> neuron.init_state()

    **Example 3: Chattering (CH) neuron**

    .. code-block:: python

       >>> # Create a chattering neuron
       >>> neuron = bp.izhikevich(1, a=0.02, b=0.2, c=-50*u.mV, d=2*u.mV)
       >>> neuron.init_state()

    **Example 4: Population with heterogeneous parameters**

    .. code-block:: python

       >>> import jax.numpy as jnp
       >>>
       >>> # Create 100 neurons with random parameter variation
       >>> key = jax.random.PRNGKey(0)
       >>> a_vals = jax.random.uniform(key, (100,), minval=0.01, maxval=0.1)
       >>> neuron = bp.izhikevich(100, a=a_vals, b=0.2, c=-65*u.mV, d=8*u.mV)
       >>> neuron.init_state()

    References
    ----------
    .. [1] Izhikevich EM. (2003). Simple model of spiking neurons. IEEE
           Transactions on Neural Networks, 14:1569–1572.
           DOI: https://doi.org/10.1109/TNN.2003.820440
    .. [2] Pauli R, Weidel P, Kunkel S, Morrison A (2018). Reproducing
           polychronization: A guide to maximizing the reproducibility of
           spiking network models. Frontiers in Neuroinformatics, 12:46.
           DOI: https://doi.org/10.3389/fninf.2018.00046

    See Also
    --------
    iaf_psc_delta : Leaky integrate-and-fire with delta-shaped PSCs
    iaf_psc_exp : Leaky integrate-and-fire with exponential PSCs
    mat2_psc_exp : Multi-timescale adaptive threshold with exponential PSCs
    aeif_psc_exp : Adaptive exponential integrate-and-fire model
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

    def init_state(self, batch_size=None, **kwargs):
        r"""Initialize state variables for the Izhikevich neuron.

        This method initializes the membrane potential :math:`V_{\text{m}}`,
        recovery variable :math:`U_{\text{m}}`, and buffered input current
        :math:`I`. By default, :math:`V_{\text{m}}` is initialized to -65 mV
        and :math:`U_{\text{m}}` is initialized to :math:`b \cdot V_0`
        (matching NEST behavior). The buffered current :math:`I` is initialized
        to zero.

        Parameters
        ----------
        batch_size : int or None, optional
            If provided, states are created with shape
            ``(batch_size, *varshape)``. ``None`` keeps unbatched state.
            Default is ``None``.
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        - If ``U_initializer`` is None (default), :math:`U_{\text{m}}` is
          initialized to :math:`b \cdot V_0` where :math:`V_0` is the initial
          value of :math:`V_{\text{m}}`. This matches NEST's default
          initialization: ``u_ = b * v_``.
        - The buffered current ``I`` is always initialized to zero with units
          of pA, implementing NEST's ring buffer semantic (one-step delay).
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
        batch_shape = ((batch_size,) + tuple(self.varshape)) if batch_size is not None else self.varshape
        self.I = brainstate.ShortTermState(u.math.zeros(batch_shape) * u.pA)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute spike output using the surrogate gradient function.

        This method applies the surrogate gradient function (``spk_fun``) to
        the scaled voltage difference :math:`(V - V_{\text{th}}) / (V_{\text{th}} - c)`,
        producing a differentiable spike indicator for gradient-based learning.
        The scaling factor normalizes the voltage range to approximately [0, 1]
        for typical surrogate functions.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential to test for spike emission (with units of
            voltage, typically mV). Shape: ``(*varshape,)`` or
            ``(batch_size, *varshape)``.
            Default: None (uses ``self.V.value``).

        Returns
        -------
        ArrayLike
            Surrogate-differentiable spike indicator. Shape matches input ``V``.
            Values are in [0, 1] for typical surrogate functions, with gradients
            defined even at the threshold crossing.

        Notes
        -----
        - The scaling uses the voltage reset range :math:`(V_{\text{th}} - c)`
          to normalize the input to the surrogate function.
        - This method is called automatically by ``update()`` but can also be
          used standalone for custom spike detection logic.
        - The returned spike indicator is differentiable for gradient-based
          training, unlike a hard threshold (``V >= V_th``).
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.c)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.pA):
        r"""Advance the neuron state by one simulation step.

        This method implements the NEST ``izhikevich::update`` function,
        integrating the differential equations for one time step and handling
        spike emission and reset. The update follows NEST semantics exactly,
        including the one-step delayed ring buffer for current input.

        **Update Sequence:**

        1. Read current state (:math:`V_{\text{old}}`, :math:`U_{\text{old}}`)
           and buffered current :math:`I` from the previous step.
        2. Integrate :math:`V_{\text{m}}` and :math:`U_{\text{m}}` using
           forward Euler (or published half-step scheme if
           ``consistent_integration=False``).
        3. Add delta (spike) input directly to :math:`V_{\text{m}}`.
        4. Apply the lower bound ``V_min`` if specified.
        5. Detect threshold crossing (:math:`V \geq V_{\text{th}}`) and apply
           reset: :math:`V \leftarrow c`, :math:`U \leftarrow U + d`.
        6. Buffer the new external current ``x`` for the next step (one-step
           delay, NEST ring-buffer semantic).
        7. Return surrogate-differentiable spike output.

        **Integration Details:**

        - **Standard Euler** (``consistent_integration=True``):
          Both :math:`V` and :math:`U` are updated using their values at the
          start of the step.

        - **Published Izhikevich numerics** (``consistent_integration=False``):
          :math:`V` is updated in two half-steps, and :math:`U` uses the final
          :math:`V` value.

        **Current Input Timing:**

        Following NEST conventions, the current ``x`` provided at simulation
        step *k* is buffered and takes effect at step *k + 1*. This one-step
        delay matches NEST's ring buffer implementation for synaptic and
        external currents.

        Parameters
        ----------
        x : Quantity (current), array_like, optional
            External current input in pA (or compatible current unit).
            Shape: scalar, ``(*varshape,)``, or ``(batch_size, *varshape)``.
            Default: 0 pA.
            This current is buffered and applied at the *next* time step.

        Returns
        -------
        ArrayLike
            Surrogate-differentiable spike output for the current time step.
            Shape: ``(*varshape,)`` or ``(batch_size, *varshape)``.
            Values are in [0, 1] for typical surrogate functions, with defined
            gradients for backpropagation.

        Notes
        -----
        - The integration is performed in ``float64`` precision to match NEST
          numerical accuracy.
        - Units are stripped during integration (NEST uses dimensionless
          arithmetic internally) and restored after integration.
        - Delta (spike) inputs are summed via ``sum_delta_inputs()`` and added
          directly to the membrane potential as an instantaneous voltage jump.
        - The spike detection uses the voltage *before* reset (``V_new``) to
          compute the surrogate gradient, while the state variables are updated
          to their post-reset values (``V_post``, ``U_post``).
        - If ``V_min`` is set, it is enforced after integration but before
          spike detection and reset.
        """
        dt_q = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        h = u.math.asarray(dt_q / u.ms, dtype=dftype)

        # Read current state
        v_old = self.V.value
        u_old = self.U.value
        I_buf = self.I.value  # current from previous step

        # Strip units for the integration (NEST uses dimensionless arithmetic
        # internally; the quantities are in mV and pA with R=1)
        v = u.math.asarray(v_old / u.mV, dtype=dftype)
        um = u.math.asarray(u_old / u.mV, dtype=dftype)
        I_val = u.math.asarray((I_buf + self.I_e) / u.pA, dtype=dftype)
        a = u.math.asarray(self.a, dtype=dftype)
        b = u.math.asarray(self.b, dtype=dftype)

        # Delta (spike) input — added directly to V
        delta_v = self.sum_delta_inputs(u.math.zeros_like(v_old))
        delta_v_raw = u.math.asarray(delta_v / u.mV, dtype=dftype)

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
            v_min = u.math.asarray(self.V_min / u.mV, dtype=dftype)
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
