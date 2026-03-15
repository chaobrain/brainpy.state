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
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron
from ._utils import is_tracer

__all__ = [
    'amat2_psc_exp',
]


class amat2_psc_exp(NESTNeuron):
    r"""NEST-compatible ``amat2_psc_exp`` neuron model.

    Non-resetting leaky integrate-and-fire neuron with exponential postsynaptic
    currents, two-timescale adaptive threshold, and a voltage-dependent threshold
    component that tracks the low-pass filtered membrane potential derivative.

    **Model Overview**

    ``amat2_psc_exp`` extends the ``mat2_psc_exp`` model by adding a voltage-dependent
    threshold component :math:`V_{th,v}` that captures the effect of fast voltage
    fluctuations on spike threshold. This mechanism improves the model's ability to
    reproduce diverse firing patterns observed in biological neurons, including
    burst firing and spike-frequency adaptation.

    The model features:

    - **Non-resetting membrane potential**: After spike emission, the membrane
      potential continues to integrate normally without reset
    - **Exponential PSCs**: Postsynaptic currents decay exponentially with separate
      time constants for excitation and inhibition
    - **Multi-timescale adaptation**: Two independent threshold components (fast and
      slow) capture short-term and long-term adaptation
    - **Voltage-dependent threshold**: A third threshold component tracks the
      low-pass filtered derivative of membrane potential, making the threshold
      sensitive to voltage velocity
    - **Absolute refractory period**: Spike emission is blocked for a fixed duration
      after each spike

    When ``beta = 0``, this model reduces to ``mat2_psc_exp``.

    Mathematical Description
    ------------------------

    **1. Subthreshold Membrane Dynamics**

    The membrane potential evolves according to:

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m}
       + \frac{I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}} + I_e + I_0}{C_m}

    where :math:`V_m` is the absolute membrane potential, :math:`E_L` is the resting
    potential, :math:`\tau_m` is the membrane time constant, :math:`C_m` is the
    membrane capacitance, and :math:`I_{\mathrm{syn,ex}}`, :math:`I_{\mathrm{syn,in}}`,
    :math:`I_e`, and :math:`I_0` are excitatory synaptic, inhibitory synaptic,
    constant external, and dynamic external currents, respectively.

    **2. Synaptic Current Dynamics**

    Postsynaptic currents decay exponentially:

    .. math::

       \frac{dI_{\mathrm{syn,ex}}}{dt} = -\frac{I_{\mathrm{syn,ex}}}{\tau_{\mathrm{syn,ex}}}

       \frac{dI_{\mathrm{syn,in}}}{dt} = -\frac{I_{\mathrm{syn,in}}}{\tau_{\mathrm{syn,in}}}

    Incoming spikes cause instantaneous jumps in the corresponding current by the
    synaptic weight.

    **3. Adaptive Threshold Dynamics**

    The total spike threshold is:

    .. math::

       V_{th}(t) = \omega + V_{th,1}(t) + V_{th,2}(t) + V_{th,v}(t)

    where :math:`\omega` is the resting threshold (an absolute voltage), and
    :math:`V_{th,1}`, :math:`V_{th,2}`, :math:`V_{th,v}` are adaptive components.

    The two time-dependent threshold components decay exponentially:

    .. math::

       \frac{dV_{th,1}}{dt} = -\frac{V_{th,1}}{\tau_1}
       \qquad
       \frac{dV_{th,2}}{dt} = -\frac{V_{th,2}}{\tau_2}

    On each spike emission, these components are incremented:

    .. math::

       V_{th,1} \leftarrow V_{th,1} + \alpha_1
       \qquad
       V_{th,2} \leftarrow V_{th,2} + \alpha_2

    **4. Voltage-Dependent Threshold Component**

    The voltage-dependent threshold component is defined as [3]_, Eqs. 16-17:

    .. math::

       V_{th,v}(t) = \beta \int_0^t K(s) \frac{dV_m}{dt}(t-s)\, ds

    where the kernel is:

    .. math::

       K(s) = \frac{s}{\tau_v} \exp\!\left(-\frac{s}{\tau_v}\right)

    This convolution is implemented via two auxiliary state variables
    :math:`V_{th,v}` and :math:`V_{th,dv}`, which are evolved using the exact
    integration scheme. The propagator coefficients for these variables depend
    on :math:`\beta`, :math:`\tau_v`, and all other time constants and are
    computed according to the formulas in the NEST implementation (see ``update``
    method for details).

    **5. Spike Emission and Refractory Period**

    A spike is emitted when:

    .. math::

       V_m \geq V_{th}(t) \quad \text{and} \quad t - t_{\mathrm{last\_spike}} > t_{\mathrm{ref}}

    where :math:`t_{\mathrm{ref}}` is the absolute refractory period. Upon spike
    emission:

    - The threshold components :math:`V_{th,1}` and :math:`V_{th,2}` are incremented
    - The refractory period counter is set to :math:`t_{\mathrm{ref}} / dt`
    - **The membrane potential is NOT reset** but continues to integrate normally

    **6. Numerical Integration**

    The model uses the exact integration scheme for linear ODEs [1]_, computing
    closed-form propagator matrices for one time step. This ensures numerical
    stability and accuracy for arbitrary time step sizes (subject to the constraint
    that all time constants must differ to avoid singularities in the propagator
    computation).

    **Update Order**

    dftype = brainstate.environ.dftype()
    ditype = brainstate.environ.ditype()
    Each simulation step proceeds as follows (matching NEST's update order):

    1. Evolve voltage-dependent threshold component (``V_th_v``, ``V_th_dv``)
       using exact integration propagators
    2. Evolve membrane potential using exact integration
    3. Decay adaptive threshold components (``V_th_1``, ``V_th_2``)
    4. Decay synaptic currents and add incoming spike weights
    5. Check spike condition: if not refractory and :math:`V_m \geq V_{th}`,
       emit spike, increment threshold components, set refractory counter
    6. If refractory, decrement refractory counter
    7. Store buffered external currents for next step

    Implementation Notes
    --------------------

    - All time constants must be strictly positive and pairwise distinct:
      ``tau_m != tau_syn_ex``, ``tau_m != tau_syn_in``, ``tau_m != tau_v``,
      ``tau_v != tau_syn_ex``, ``tau_v != tau_syn_in``. This constraint arises
      from the exact integration scheme, which requires inverting matrices that
      become singular when time constants coincide.
    - Numerics may be unstable if time constants are very close but not exactly
      equal due to ill-conditioning of the propagator matrix computation.
    - Some parameter values in Table 1 of [4]_ are incorrect; see Table 4 of [5]_
      for corrected values.
    - The voltage-dependent threshold component requires significant computational
      overhead (additional propagator coefficients). Set ``beta = 0`` to disable
      this feature and recover ``mat2_psc_exp`` behavior.

    Parameters
    ----------
    in_size : int, tuple of int
        Population shape (number of neurons or spatial dimensions).
    E_L : Quantity, ndarray
        Resting membrane potential. Default: -70 mV.
    C_m : Quantity, ndarray
        Membrane capacitance. Must be strictly positive. Default: 200 pF.
    tau_m : Quantity, ndarray
        Membrane time constant. Must be strictly positive and differ from
        ``tau_syn_ex``, ``tau_syn_in``, and ``tau_v``. Default: 10 ms.
    t_ref : Quantity, ndarray
        Absolute refractory period (duration of spike emission block).
        Must be strictly positive. Default: 2 ms.
    tau_syn_ex : Quantity, ndarray
        Excitatory postsynaptic current time constant. Must be strictly positive
        and differ from ``tau_m``, ``tau_v``, and ``tau_syn_in``. Default: 1 ms.
    tau_syn_in : Quantity, ndarray
        Inhibitory postsynaptic current time constant. Must be strictly positive
        and differ from ``tau_m``, ``tau_v``, and ``tau_syn_ex``. Default: 3 ms.
    I_e : Quantity, ndarray
        Constant external input current. Default: 0 pA.
    tau_1 : Quantity, ndarray
        Time constant for short-timescale adaptive threshold component.
        Must be strictly positive. Default: 10 ms.
    tau_2 : Quantity, ndarray
        Time constant for long-timescale adaptive threshold component.
        Must be strictly positive. Default: 200 ms.
    alpha_1 : Quantity, ndarray
        Increment to ``V_th_1`` on each spike (fast adaptation amplitude).
        Default: 10 mV.
    alpha_2 : Quantity, ndarray
        Increment to ``V_th_2`` on each spike (slow adaptation amplitude).
        Default: 0 mV.
    beta : Quantity, ndarray
        Scaling coefficient for voltage-dependent threshold component.
        Units: 1/ms. Set to 0 to disable voltage-dependent threshold and
        recover ``mat2_psc_exp`` behavior. Default: 0 / ms.
    tau_v : Quantity, ndarray
        Time constant for voltage-dependent threshold component. Must be
        strictly positive and differ from ``tau_m``, ``tau_syn_ex``, and
        ``tau_syn_in``. Default: 5 ms.
    omega : Quantity, ndarray
        Resting spike threshold (absolute voltage, not relative to ``E_L``).
        Default: -65 mV.
    V_initializer : Callable, Quantity
        Initializer for membrane potential. Can be a ``braintools.init``
        initializer or a constant value. Default: Constant(-70 mV).
    spk_fun : Callable
        Surrogate gradient function for differentiable spike generation.
        Default: ``braintools.surrogate.ReluGrad()``.
    spk_reset : str
        Reset mode for surrogate gradient computation. Options: ``'soft'``
        (subtract threshold) or ``'hard'`` (stop gradient). Note: this does
        NOT affect the membrane potential dynamics (no reset occurs). It only
        affects gradient flow through the spike function. Default: ``'hard'``.
    ref_var : bool
        If True, expose a boolean ``refractory`` state variable indicating
        whether each neuron is currently in the refractory period.
        Default: False.
    name : str, optional
        Name of the neuron population.

    Parameter Mapping
    -----------------

    The following table maps BrainPy parameter names to their mathematical symbols
    and NEST equivalents:

    ==================== ================== =============================== ==========================================================
    **Parameter**        **Default**        **Math equivalent**             **Description**
    ==================== ================== =============================== ==========================================================
    ``in_size``          (required)                                         Population shape
    ``E_L``              -70 mV             :math:`E_L`                     Resting membrane potential
    ``C_m``              200 pF             :math:`C_m`                     Membrane capacitance
    ``tau_m``            10 ms              :math:`\tau_m`                  Membrane time constant
    ``t_ref``            2 ms               :math:`t_{ref}`                 Duration of absolute refractory period (no spiking)
    ``tau_syn_ex``       1 ms               :math:`\tau_{\mathrm{syn,ex}}`  Time constant of excitatory postsynaptic current
    ``tau_syn_in``       3 ms               :math:`\tau_{\mathrm{syn,in}}`  Time constant of inhibitory postsynaptic current
    ``I_e``              0 pA               :math:`I_e`                     Constant external input current
    ``tau_1``            10 ms              :math:`\tau_1`                  Short time constant of adaptive threshold
    ``tau_2``            200 ms             :math:`\tau_2`                  Long time constant of adaptive threshold
    ``alpha_1``          10 mV              :math:`\alpha_1`                Amplitude of short time threshold adaption
    ``alpha_2``          0 mV               :math:`\alpha_2`                Amplitude of long time threshold adaption
    ``beta``             0 1/ms             :math:`\beta`                   Scaling coefficient for voltage-dependent threshold
    ``tau_v``            5 ms               :math:`\tau_v`                  Time constant for voltage-dependent threshold component
    ``omega``            -65 mV             :math:`\omega`                  Resting spike threshold (absolute value, not relative to E_L)
    ``V_initializer``    Constant(-70 mV)                                   Membrane potential initializer
    ``spk_fun``          ReluGrad()                                         Surrogate spike function
    ``spk_reset``        ``'hard'``                                         Reset mode (not used for voltage; used in ``get_spike``)
    ``ref_var``          ``False``                                          If True, expose boolean refractory state
    ==================== ================== =============================== ==========================================================

    State Variables
    ---------------

    ========================= ===================== ====================================================
    **Variable**              **Type**              **Description**
    ========================= ===================== ====================================================
    ``V``                     ``HiddenState`` (mV)  Membrane potential (absolute)
    ``V_th_1``                ``ShortTermState``    Short-timescale adaptive threshold component (mV, relative to omega)
    ``V_th_2``                ``ShortTermState``    Long-timescale adaptive threshold component (mV, relative to omega)
    ``V_th_v``                ``ShortTermState``    Voltage-dependent threshold component (mV)
    ``V_th_dv``               ``ShortTermState``    Derivative of voltage-dependent threshold (mV)
    ``i_syn_ex``              ``ShortTermState``    Excitatory postsynaptic current (pA)
    ``i_syn_in``              ``ShortTermState``    Inhibitory postsynaptic current (pA)
    ``i_0``                   ``ShortTermState``    DC input current (pA)
    ``refractory_step_count`` ``ShortTermState``    Refractory countdown (integer steps)
    ``last_spike_time``       ``ShortTermState``    Time of last spike (ms)
    ``refractory``            ``ShortTermState``    Boolean refractory state (only if ``ref_var=True``)
    ========================= ===================== ====================================================

    Raises
    ------
    ValueError
        If ``C_m <= 0``.
    ValueError
        If any time constant is non-positive.
    ValueError
        If ``tau_m`` equals ``tau_syn_ex``, ``tau_syn_in``, or ``tau_v``
        (exact integration propagators become singular).
    ValueError
        If ``tau_v`` equals ``tau_syn_ex`` or ``tau_syn_in``
        (exact integration propagators become singular).

    Examples
    --------
    **Basic usage with voltage-dependent threshold:**

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import saiunit as u
        >>> import brainstate as bstate
        >>>
        >>> # Create a neuron with voltage-dependent threshold
        >>> neuron = bst.amat2_psc_exp(
        ...     in_size=100,
        ...     beta=0.5 / u.ms,  # Enable voltage-dependent threshold
        ...     tau_v=5.0 * u.ms,
        ...     alpha_1=10.0 * u.mV,
        ...     alpha_2=0.5 * u.mV,
        ... )
        >>>
        >>> # Initialize states
        >>> with bstate.environ.context(dt=0.1 * u.ms):
        ...     neuron.init_all_states()
        ...     spike = neuron.update(x=500.0 * u.pA)  # Apply input current

    **Comparing with mat2_psc_exp (beta=0):**

    .. code-block:: python

        >>> # AMAT2 with beta=0 behaves like MAT2
        >>> amat2 = bst.amat2_psc_exp(in_size=10, beta=0.0 / u.ms)
        >>> mat2 = bst.mat2_psc_exp(in_size=10)
        >>>
        >>> # Both should produce similar dynamics
        >>> with bstate.environ.context(dt=0.1 * u.ms):
        ...     amat2.init_all_states()
        ...     mat2.init_all_states()

    **Simulating burst firing with strong voltage-dependent threshold:**

    .. code-block:: python

        >>> neuron = bst.amat2_psc_exp(
        ...     in_size=1,
        ...     beta=1.0 / u.ms,  # Strong voltage dependence
        ...     tau_v=3.0 * u.ms,  # Fast voltage tracking
        ...     alpha_1=15.0 * u.mV,  # Strong fast adaptation
        ...     tau_1=5.0 * u.ms,
        ... )
        >>>
        >>> with bstate.environ.context(dt=0.1 * u.ms):
        ...     neuron.init_all_states()
        ...     spikes = []
        ...     for _ in range(1000):
        ...         spk = neuron.update(x=600.0 * u.pA)
        ...         spikes.append(spk)

    References
    ----------
    .. [1] Rotter S and Diesmann M (1999). Exact simulation of
           time-invariant linear systems with applications to neuronal
           modeling. Biological Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    .. [2] Diesmann M, Gewaltig M-O, Rotter S, Aertsen A (2001). State
           space analysis of synchronous spiking in cortical neural
           networks. Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X
    .. [3] Kobayashi R, Tsubo Y and Shinomoto S (2009). Made-to-order
           spiking neuron model equipped with a multi-timescale adaptive
           threshold. Frontiers in Computational Neuroscience 3:9.
           DOI: https://doi.org/10.3389/neuro.10.009.2009
    .. [4] Yamauchi S, Kim H, Shinomoto S (2011). Elemental spiking neuron
           model for reproducing diverse firing patterns and predicting precise
           firing times. Frontiers in Computational Neuroscience 5:42.
           DOI: https://doi.org/10.3389/fncom.2011.00042
    .. [5] Heiberg T, Kriener B, Tetzlaff T, Einevoll GT, Plesser HE (2018).
           Firing-rate model for neurons with a broad repertoire of spiking
           behaviors. J Comput Neurosci 45:103.
           DOI: https://doi.org/10.1007/s10827-018-0693-9

    See Also
    --------
    mat2_psc_exp : Same model without voltage-dependent threshold component.
    aeif_psc_exp : Adaptive exponential integrate-and-fire with spike reset.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 200. * u.pF,
        tau_m: ArrayLike = 10. * u.ms,
        t_ref: ArrayLike = 2. * u.ms,
        tau_syn_ex: ArrayLike = 1. * u.ms,
        tau_syn_in: ArrayLike = 3. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        tau_1: ArrayLike = 10. * u.ms,
        tau_2: ArrayLike = 200. * u.ms,
        alpha_1: ArrayLike = 10. * u.mV,
        alpha_2: ArrayLike = 0. * u.mV,
        beta: ArrayLike = 0. / u.ms,
        tau_v: ArrayLike = 5. * u.ms,
        omega: ArrayLike = -65. * u.mV,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.tau_1 = braintools.init.param(tau_1, self.varshape)
        self.tau_2 = braintools.init.param(tau_2, self.varshape)
        self.alpha_1 = braintools.init.param(alpha_1, self.varshape)
        self.alpha_2 = braintools.init.param(alpha_2, self.varshape)
        self.beta = braintools.init.param(beta, self.varshape)
        self.tau_v = braintools.init.param(tau_v, self.varshape)
        self.omega = braintools.init.param(omega, self.varshape)

        self.V_initializer = V_initializer
        self.ref_var = ref_var
        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        r"""Convert a quantity to a plain NumPy array in specified units.

        Parameters
        ----------
        x : Quantity, ndarray
            Input value with units.
        unit : Quantity
            Target unit for conversion.

        Returns
        -------
        ndarray
            Plain float64 NumPy array with units stripped.
        """
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast a parameter array to match state variable shape.

        Parameters
        ----------
        x_np : ndarray
            Parameter array (plain NumPy, no units).
        shape : tuple
            Target shape for broadcasting.

        Returns
        -------
        ndarray
            Broadcasted array with shape matching ``shape``.
        """
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        r"""Validate model parameters for physical and numerical constraints.

        This method checks that:
        - Capacitance is strictly positive
        - All time constants are strictly positive
        - Time constants are pairwise distinct (required for exact integration)

        Raises
        ------
        ValueError
            If ``C_m <= 0``.
        ValueError
            If any time constant (``tau_m``, ``tau_syn_ex``, ``tau_syn_in``,
            ``tau_1``, ``tau_2``, ``tau_v``, ``t_ref``) is non-positive.
        ValueError
            If ``tau_m`` equals ``tau_syn_ex``, ``tau_syn_in``, or ``tau_v``
            (causes singularities in propagator matrix).
        ValueError
            If ``tau_v`` equals ``tau_syn_ex`` or ``tau_syn_in``
            (causes singularities in propagator matrix).
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.tau_m)):
            return

        if np.any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        tau_m_val = self.tau_m
        tau_ex_val = self.tau_syn_ex
        tau_in_val = self.tau_syn_in
        tau_v_val = self.tau_v
        if np.any(tau_m_val <= 0.0 * u.ms) or np.any(tau_ex_val <= 0.0 * u.ms) or np.any(tau_in_val <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self.t_ref <= 0.0 * u.ms):
            raise ValueError('Refractory time must be strictly positive.')
        if np.any(self.tau_1 <= 0.0 * u.ms) or np.any(self.tau_2 <= 0.0 * u.ms):
            raise ValueError('Adaptive threshold time constants must be strictly positive.')
        if np.any(tau_v_val <= 0.0 * u.ms):
            raise ValueError('tau_v must be strictly positive.')
        if np.any(tau_m_val == tau_ex_val) or np.any(tau_m_val == tau_in_val) or np.any(tau_m_val == tau_v_val):
            raise ValueError(
                'tau_m must differ from tau_syn_ex, tau_syn_in and tau_v. '
                'See note in documentation.'
            )
        if np.any(tau_v_val == tau_ex_val) or np.any(tau_v_val == tau_in_val):
            raise ValueError(
                'tau_v must differ from tau_syn_ex, tau_syn_in and tau_m. '
                'See note in documentation.'
            )

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Creates and initializes all state variables including membrane potential,
        adaptive threshold components, voltage-dependent threshold components,
        synaptic currents, and refractory state.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size for vectorized simulation. If None, no batch
            dimension is added (shape is ``in_size`` only). If provided, state
            shape becomes ``(batch_size, *in_size)``.
        **kwargs
            Additional initialization arguments (unused, for API compatibility).

        Notes
        -----
        State variables initialized:
        - ``V``: Membrane potential (from ``V_initializer``)
        - ``V_th_1``, ``V_th_2``: Adaptive threshold components (zero)
        - ``V_th_v``, ``V_th_dv``: Voltage-dependent threshold components (zero)
        - ``i_syn_ex``, ``i_syn_in``: Synaptic currents (zero)
        - ``i_0``: External current buffer (zero)
        - ``refractory_step_count``: Refractory counter (zero, not refractory)
        - ``last_spike_time``: Last spike time (large negative value)
        - ``refractory`` (if ``ref_var=True``): Boolean refractory state (False)
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.V_th_1 = brainstate.ShortTermState(zeros * u.mV)
        self.V_th_2 = brainstate.ShortTermState(zeros * u.mV)
        self.V_th_v = brainstate.ShortTermState(zeros * u.mV)
        self.V_th_dv = brainstate.ShortTermState(zeros * u.mV)
        self.i_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.i_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.i_0 = brainstate.ShortTermState(zeros * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=ditype))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(u.math.asarray(ref_steps > 0, dtype=bool))

    def get_spike(self, V: ArrayLike = None, V_th: ArrayLike = None):
        r"""Compute spike output using surrogate gradient function.

        Applies the surrogate gradient function to the scaled distance between
        membrane potential and adaptive threshold. This enables differentiable
        spike generation for gradient-based learning.

        Parameters
        ----------
        V : Quantity, ndarray, optional
            Membrane potential (absolute voltage). If None, uses current
            ``self.V.value``. Shape: ``(*varshape,)`` or ``(batch_size, *varshape)``.
        V_th : Quantity, ndarray, optional
            Total spike threshold (absolute voltage). If None, computed as
            ``omega + V_th_1 + V_th_2 + V_th_v``. Shape: same as ``V``.

        Returns
        -------
        ndarray
            Spike output (pseudo-probability in [0, 1] from surrogate function).
            Shape: same as ``V``.

        Notes
        -----
        The spike function is applied to the scaled voltage distance:

        .. math::

            s = \\mathrm{spk\\_fun}\\left(\\frac{V - V_{th}}{|\\omega - E_L|}\\right)

        The scaling factor ``|omega - E_L|`` normalizes the voltage distance to
        the typical threshold range, improving numerical stability across different
        parameter regimes.
        """
        V = self.V.value if V is None else V
        if V_th is None:
            V_th = self.omega + self.V_th_1.value + self.V_th_2.value + self.V_th_v.value
        v_scaled = (V - V_th) / u.math.abs(self.omega - self.E_L)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        r"""Compute refractory period duration in time steps.

        Returns
        -------
        int32 array
            Number of simulation steps corresponding to the refractory period
            ``t_ref``, computed as ``ceil(t_ref / dt)``. Shape: ``(*varshape,)``.

        Notes
        -----
        Uses the current simulation time step from ``brainstate.environ.get_dt()``.
        The ceiling function ensures at least one step if ``t_ref > 0``.
        """
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    def update(self, x=0. * u.pA):
        r"""Perform one simulation time step.

        Integrates membrane potential, synaptic currents, and adaptive threshold
        components for one time step using the exact integration scheme. Detects
        spike emission and updates refractory state. The membrane potential is
        NOT reset after spikes.

        This method follows the NEST update order for ``amat2_psc_exp``:

        1. Evolve voltage-dependent threshold components (``V_th_v``, ``V_th_dv``)
           using exact propagators that depend on all synaptic and membrane currents
        2. Evolve membrane potential using exact integration
        3. Decay time-dependent threshold components (``V_th_1``, ``V_th_2``)
        4. Decay synaptic currents and add incoming spike weights
        5. Detect spikes: if not refractory and ``V >= omega + V_th_1 + V_th_2 + V_th_v``:

           - Increment threshold components by ``alpha_1`` and ``alpha_2``
           - Set refractory counter to ``ceil(t_ref / dt)``
           - Record spike time
        6. If refractory, decrement refractory counter
        7. Buffer external currents for next step

        Parameters
        ----------
        x : Quantity, ndarray, optional
            External input current for the current time step. This current is
            buffered and applied in the NEXT time step (one-step delay, following
            NEST convention). Shape: scalar, ``(*varshape,)``, or
            ``(batch_size, *varshape)``. Default: 0 pA.

        Returns
        -------
        ndarray
            Spike output from surrogate gradient function. Values in [0, 1]
            represent pseudo-spike probabilities. Actual spike detection (for
            threshold increment and refractory period) uses hard threshold crossing.
            Shape: same as state variables.

        Notes
        -----
        **Exact Integration Propagators**

        The model uses closed-form propagators for linear ODEs [1]_. For a single
        time step of size ``h``, the propagators are:

        **Independent exponential decays:**

        .. math::

            P_{11} &= e^{-h/\\tau_{syn,ex}} \\quad (i\\_syn\\_ex) \\\\
            P_{22} &= e^{-h/\\tau_{syn,in}} \\quad (i\\_syn\\_in) \\\\
            P_{33} &= e^{-h/\\tau_m} \\quad (V_m) \\\\
            P_{44} &= e^{-h/\\tau_1} \\quad (V_{th,1}) \\\\
            P_{55} &= e^{-h/\\tau_2} \\quad (V_{th,2}) \\\\
            P_{66} &= e^{-h/\\tau_v} \\quad (V_{th,dv}) \\\\
            P_{77} &= e^{-h/\\tau_v} \\quad (V_{th,v})

        **Membrane potential coupling to currents:**

        .. math::

            P_{30} &= \\frac{\\tau_m}{C_m}(1 - e^{-h/\\tau_m}) \\\\
            P_{31} &= \\frac{\\tau_m \\tau_{syn,ex}}{C_m(\\tau_{syn,ex} - \\tau_m)}
                       (e^{-h/\\tau_{syn,ex}} - e^{-h/\\tau_m}) \\\\
            P_{32} &= \\frac{\\tau_m \\tau_{syn,in}}{C_m(\\tau_{syn,in} - \\tau_m)}
                       (e^{-h/\\tau_{syn,in}} - e^{-h/\\tau_m})

        **Voltage-dependent threshold (derivative component, ``V_th_dv``):**

        .. math::

            P_{60} &= \\frac{\\beta \\tau_m \\tau_v}{C_m(\\tau_m - \\tau_v)}
                       (e^{-h/\\tau_m} - e^{-h/\\tau_v}) \\\\
            P_{61} &= \\frac{\\beta \\tau_{syn,ex} \\tau_m \\tau_v}
                       {C_m(\\tau_{syn,ex}-\\tau_m)(\\tau_{syn,ex}-\\tau_v)(\\tau_m-\\tau_v)}
                       \\times \\\\
                   &\\quad (e^{-h/\\tau_v}(-\\tau_{syn,ex}+\\tau_m) +
                            e^{-h/\\tau_m}(\\tau_{syn,ex}-\\tau_v) +
                            e^{-h/\\tau_{syn,ex}}(-\\tau_m+\\tau_v)) \\\\
            P_{62} &= \\text{[similar for inhibitory synapse]} \\\\
            P_{63} &= \\frac{\\beta \\tau_v}{\\tau_m - \\tau_v}
                       (e^{-h/\\tau_v} - e^{-h/\\tau_m})

        **Voltage-dependent threshold (integrated component, ``V_th_v``):**

        .. math::

            P_{70} &= \\frac{\\beta \\tau_m \\tau_v}{C_m(\\tau_m-\\tau_v)^2}
                       (e^{-h/\\tau_m} \\tau_m \\tau_v -
                        e^{-h/\\tau_v}(h(\\tau_m-\\tau_v) + \\tau_m \\tau_v)) \\\\
            P_{71} &= \\text{[complex expression, see code]} \\\\
            P_{72} &= \\text{[complex expression, see code]} \\\\
            P_{73} &= \\frac{\\beta \\tau_v}{(\\tau_m-\\tau_v)^2}
                       (e^{-h/\\tau_v}(h(\\tau_m-\\tau_v)+\\tau_m\\tau_v) -
                        e^{-h/\\tau_m}\\tau_m\\tau_v) \\\\
            P_{76} &= h e^{-h/\\tau_v}

        These propagators are recomputed at each time step to accommodate
        spatially-varying parameters (different time constants for different neurons).

        **Update Equations**

        The state update proceeds as:

        .. math::

            V_{th,v}^{new} &= P_{70} (I_e + I_0) + P_{71} I_{syn,ex} + P_{72} I_{syn,in}
                              + P_{73} V_m + P_{76} V_{th,dv} + P_{77} V_{th,v} \\\\
            V_{th,dv}^{new} &= P_{60} (I_e + I_0) + P_{61} I_{syn,ex} + P_{62} I_{syn,in}
                               + P_{63} V_m + P_{66} V_{th,dv} \\\\
            V_m^{new} &= P_{30} (I_e + I_0) + P_{31} I_{syn,ex} + P_{32} I_{syn,in}
                         + P_{33} V_m \\\\
            V_{th,1}^{new} &= P_{44} V_{th,1} \\\\
            V_{th,2}^{new} &= P_{55} V_{th,2} \\\\
            I_{syn,ex}^{new} &= P_{11} I_{syn,ex} + \\Delta I_{ex} \\\\
            I_{syn,in}^{new} &= P_{22} I_{syn,in} + \\Delta I_{in}

        where :math:`\\Delta I_{ex}` and :math:`\\Delta I_{in}` are the summed weights
        of excitatory and inhibitory spikes arriving in the current step.

        **Spike Detection and Threshold Increment**

        Spikes are detected when:

        .. math::

            V_m \\geq \\omega + V_{th,1} + V_{th,2} + V_{th,v}
            \\quad \\text{and} \\quad r = 0

        where :math:`r` is the refractory counter. On spike detection:

        .. math::

            V_{th,1} &\\leftarrow V_{th,1} + \\alpha_1 \\\\
            V_{th,2} &\\leftarrow V_{th,2} + \\alpha_2 \\\\
            r &\\leftarrow \\lceil t_{ref} / dt \\rceil

        **No Membrane Reset**

        Unlike many spiking neuron models, the membrane potential is NOT reset
        after a spike. It continues to integrate according to the differential
        equation. Adaptation is achieved solely through threshold elevation.

        **Input Handling**

        - **Spike inputs**: Accessed via ``self.sum_delta_inputs()`` which aggregates
          weights from all connected projections. Positive weights add to excitatory
          current, negative weights to inhibitory current.
        - **Current inputs**: Accessed via ``self.sum_current_inputs(x, V)`` which
          sums the external current ``x`` and any currents from projections. This
          current is buffered in ``i_0`` and applied in the NEXT time step.

        **Surrogate Gradient**

        The return value uses the surrogate gradient function for differentiability.
        The actual spike condition (hard threshold) is evaluated separately and used
        for threshold increment and refractory logic. This allows gradient-based
        learning while maintaining biological spike semantics.

        Warnings
        --------
        - If time constants are very close but not exactly equal, numerical
          instability may occur in propagator computation due to near-singularities.
        - The one-step delay in external current application (``i_0``) is required
          for consistency with NEST and exact integration numerics.
        - Setting ``beta`` to large values can make the voltage-dependent threshold
          very sensitive to voltage fluctuations, potentially causing numerical issues.

        References
        ----------
        .. [6] Rotter S and Diesmann M (1999). Exact simulation of
               time-invariant linear systems with applications to neuronal
               modeling. Biological Cybernetics 81:381-402.
               DOI: https://doi.org/10.1007/s004220050570
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        # Extract all parameters as plain float64 numpy arrays
        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        taum = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tauE = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tauI = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        tau_1 = self._broadcast_to_state(self._to_numpy(self.tau_1, u.ms), v_shape)
        tau_2 = self._broadcast_to_state(self._to_numpy(self.tau_2, u.ms), v_shape)
        alpha_1 = self._broadcast_to_state(self._to_numpy(self.alpha_1, u.mV), v_shape)
        alpha_2 = self._broadcast_to_state(self._to_numpy(self.alpha_2, u.mV), v_shape)
        beta = self._broadcast_to_state(self._to_numpy(self.beta, 1.0 / u.ms), v_shape)
        tauV = self._broadcast_to_state(self._to_numpy(self.tau_v, u.ms), v_shape)
        # omega is stored as absolute mV; convert to relative to E_L
        omega_rel = self._broadcast_to_state(self._to_numpy(self.omega - self.E_L, u.mV), v_shape)

        # State variables (V_m relative to E_L)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        V_th_1 = self._broadcast_to_state(self._to_numpy(self.V_th_1.value, u.mV), v_shape)
        V_th_2 = self._broadcast_to_state(self._to_numpy(self.V_th_2.value, u.mV), v_shape)
        V_th_v = self._broadcast_to_state(self._to_numpy(self.V_th_v.value, u.mV), v_shape)
        V_th_dv = self._broadcast_to_state(self._to_numpy(self.V_th_dv.value, u.mV), v_shape)
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value, u.pA), v_shape)
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.i_syn_in.value, u.pA), v_shape)
        i_0 = self._broadcast_to_state(self._to_numpy(self.i_0.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=ditype), v_shape
        )

        # --- Compute propagator coefficients (NEST pre_run_hook) ---
        c = C_m

        eE = np.exp(-h / tauE)
        eI = np.exp(-h / tauI)
        em = np.exp(-h / taum)
        e1 = np.exp(-h / tau_1)
        e2 = np.exp(-h / tau_2)
        eV = np.exp(-h / tauV)

        # Independent propagators
        P11 = eE
        P22 = eI
        P33 = em
        P44 = e1
        P55 = e2
        P66 = eV
        P77 = eV

        # Membrane potential propagators
        P30 = (taum - em * taum) / c
        P31 = ((eE - em) * tauE * taum) / (c * (tauE - taum))
        P32 = ((eI - em) * tauI * taum) / (c * (tauI - taum))

        # Voltage-dependent threshold propagators (V_th_dv -> row 6)
        P60 = (beta * (em - eV) * taum * tauV) / (c * (taum - tauV))
        P61 = (beta * tauE * taum * tauV * (eV * (-tauE + taum) + em * (tauE - tauV) + eE * (-taum + tauV))) \
              / (c * (tauE - taum) * (tauE - tauV) * (taum - tauV))
        P62 = (beta * tauI * taum * tauV * (eV * (-tauI + taum) + em * (tauI - tauV) + eI * (-taum + tauV))) \
              / (c * (tauI - taum) * (tauI - tauV) * (taum - tauV))
        P63 = (beta * (-em + eV) * tauV) / (taum - tauV)

        # Voltage-dependent threshold propagators (V_th_v -> row 7)
        P70 = (beta * taum * tauV * (em * taum * tauV - eV * (h * (taum - tauV) + taum * tauV))) \
              / (c * (taum - tauV) ** 2)
        P71 = (beta * tauE * taum * tauV
               * ((em * taum * (tauE - tauV) ** 2 - eE * tauE * (taum - tauV) ** 2) * tauV
                  - eV * (tauE - taum)
                  * (h * (tauE - tauV) * (taum - tauV) + tauE * taum * tauV - tauV ** 3))) \
              / (c * (tauE - taum) * (tauE - tauV) ** 2 * (taum - tauV) ** 2)
        P72 = (beta * tauI * taum * tauV
               * ((em * taum * (tauI - tauV) ** 2 - eI * tauI * (taum - tauV) ** 2) * tauV
                  - eV * (tauI - taum)
                  * (h * (tauI - tauV) * (taum - tauV) + tauI * taum * tauV - tauV ** 3))) \
              / (c * (tauI - taum) * (tauI - tauV) ** 2 * (taum - tauV) ** 2)
        P73 = (beta * tauV * (-(em * taum * tauV) + eV * (h * (taum - tauV) + taum * tauV))) \
              / (taum - tauV) ** 2
        P76 = eV * h

        # --- Get spike inputs ---
        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all > 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)

        # --- Get current inputs (one-step delayed, stored for next step) ---
        i_0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        # === NEST update ordering (amat2_psc_exp.cpp update() lines 375-421) ===

        # Step 1: Evolve voltage-dependent threshold (V_th_v and V_th_dv)
        # IMPORTANT: V_th_v must be computed BEFORE V_th_dv is updated,
        # because V_th_v uses the OLD V_th_dv value.
        V_th_v_new = (I_e + i_0) * P70 + i_syn_ex * P71 + i_syn_in * P72 \
                     + V_rel * P73 + V_th_dv * P76 + V_th_v * P77
        V_th_dv_new = (I_e + i_0) * P60 + i_syn_ex * P61 + i_syn_in * P62 \
                      + V_rel * P63 + V_th_dv * P66
        V_th_v = V_th_v_new
        V_th_dv = V_th_dv_new

        # Step 2: Evolve membrane potential
        V_rel = (I_e + i_0) * P30 + i_syn_ex * P31 + i_syn_in * P32 + V_rel * P33

        # Step 3: Decay adaptive threshold components
        V_th_1 = V_th_1 * P44
        V_th_2 = V_th_2 * P55

        # Step 4: Decay synaptic currents and add incoming spikes
        i_syn_ex = i_syn_ex * P11
        i_syn_in = i_syn_in * P22
        i_syn_ex = i_syn_ex + w_ex
        i_syn_in = i_syn_in + w_in

        # Step 5-6: Spike detection (no voltage reset!)
        not_refractory = r == 0
        spike_cond = not_refractory & (V_rel >= omega_rel + V_th_1 + V_th_2 + V_th_v)

        # On spike: jump threshold components, set refractory counter
        V_th_1 = np.where(spike_cond, V_th_1 + alpha_1, V_th_1)
        V_th_2 = np.where(spike_cond, V_th_2 + alpha_2, V_th_2)
        r = np.where(
            spike_cond,
            self._broadcast_to_state(
                np.asarray(u.math.asarray(self._refractory_counts()), dtype=ditype), v_shape
            ),
            np.where(not_refractory, r, r - 1),
        )

        # Step 7: Store buffered currents for next step

        # --- Write back state variables ---
        self.V.value = (V_rel + E_L) * u.mV
        self.V_th_1.value = V_th_1 * u.mV
        self.V_th_2.value = V_th_2 * u.mV
        self.V_th_v.value = V_th_v * u.mV
        self.V_th_dv.value = V_th_dv * u.mV
        self.i_syn_ex.value = i_syn_ex * u.pA
        self.i_syn_in.value = i_syn_in * u.pA
        self.i_0.value = i_0_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=ditype)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        # Return spike output via surrogate gradient.
        V_th_abs = omega_rel + V_th_1 + V_th_2 + V_th_v + E_L
        V_out = np.where(spike_cond, V_th_abs + 1e-12, V_th_abs - 1e-12)
        return self.get_spike(V_out * u.mV, V_th_abs * u.mV)
