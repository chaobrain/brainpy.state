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
    'mat2_psc_exp',
]


class mat2_psc_exp(NESTNeuron):
    r"""NEST-compatible ``mat2_psc_exp`` neuron model.

    Non-resetting leaky integrate-and-fire neuron model with exponential
    postsynaptic currents and a two-timescale adaptive threshold.

    **1. Model Overview**

    ``mat2_psc_exp`` implements a leaky integrate-and-fire model with exponential
    shaped postsynaptic currents (PSCs) and a multi-timescale adaptive threshold
    (MAT) [3]_. Key features:

    - **No voltage reset**: The membrane potential continues to integrate through
      spikes, providing biological realism for high-firing-rate regimes.
    - **Two-timescale threshold adaptation**: Separate fast (τ₁) and slow (τ₂)
      threshold components capture short-term spike frequency adaptation and
      long-term gain control.
    - **Absolute refractory period**: Prevents multiple spikes within ``t_ref``
      without clamping the membrane potential.
    - **Exact integration**: Subthreshold dynamics use the exponential Euler
      propagator [1]_ for numerical stability.

    **2. Mathematical Formulation**

    **2.1 Subthreshold Membrane Dynamics**

    The membrane potential evolves according to:

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m}
       + \frac{I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}} + I_e + I_0}{C_m}

    where:

    - :math:`V_m` -- membrane potential (absolute voltage)
    - :math:`E_L` -- resting potential
    - :math:`\tau_m = C_m / g_L` -- membrane time constant
    - :math:`I_{\mathrm{syn,ex}}, I_{\mathrm{syn,in}}` -- synaptic currents
    - :math:`I_e` -- constant external current
    - :math:`I_0` -- buffered step current input (updated each time step)

    **2.2 Synaptic Currents**

    Exponentially decaying currents with infinitely fast rise:

    .. math::

       \frac{dI_{\mathrm{syn,ex}}}{dt} = -\frac{I_{\mathrm{syn,ex}}}{\tau_{\mathrm{syn,ex}}}
       \qquad
       \frac{dI_{\mathrm{syn,in}}}{dt} = -\frac{I_{\mathrm{syn,in}}}{\tau_{\mathrm{syn,in}}}

    Incoming spike weights are added instantaneously: :math:`I_{\mathrm{syn}} \leftarrow I_{\mathrm{syn}} + w`.

    **2.3 Adaptive Threshold**

    The effective spike threshold is the sum of a baseline and two decaying components:

    .. math::

       V_{th}(t) = \omega + V_{th,1}(t) + V_{th,2}(t)

    where:

    .. math::

       \frac{dV_{th,1}}{dt} = -\frac{V_{th,1}}{\tau_1}
       \qquad
       \frac{dV_{th,2}}{dt} = -\frac{V_{th,2}}{\tau_2}

    On each spike at time :math:`t_{\text{spike}}`:

    .. math::

       V_{th,1}(t_{\text{spike}}^+) = V_{th,1}(t_{\text{spike}}^-) + \alpha_1
       \qquad
       V_{th,2}(t_{\text{spike}}^+) = V_{th,2}(t_{\text{spike}}^-) + \alpha_2

    **2.4 Spike Emission**

    A spike is emitted when:

    .. math::

       V_m \geq \omega + V_{th,1} + V_{th,2}
       \quad \text{and} \quad
       t - t_{\text{last\_spike}} \geq t_{\text{ref}}

    After spiking:

    - Threshold components jump by :math:`\alpha_1, \alpha_2`
    - Refractory counter is set to :math:`\lceil t_{\text{ref}} / \Delta t \rceil`
    - **Membrane potential is NOT reset** (continues integrating)

    **3. Numerical Integration**

    The model uses exact integration for the linear subthreshold system [1]_.
    For a time step :math:`h = \Delta t`:

    **3.1 Membrane Potential Propagators**

    .. math::

       V_m(t+h) &= V_m(t) e^{-h/\tau_m} + E_L (1 - e^{-h/\tau_m}) \\
                &\quad + P_{21}^{\text{ex}} I_{\text{syn,ex}}(t)
                + P_{21}^{\text{in}} I_{\text{syn,in}}(t)
                + P_{20} (I_e + I_0)

    where:

    .. math::

       P_{21}^{\text{ex}} &= -\frac{\tau_m}{C_m (1 - \tau_m/\tau_{\text{syn,ex}})}
                            e^{-h/\tau_{\text{syn,ex}}}
                            (e^{h(1/\tau_{\text{syn,ex}} - 1/\tau_m)} - 1) \\
       P_{21}^{\text{in}} &= -\frac{\tau_m}{C_m (1 - \tau_m/\tau_{\text{syn,in}})}
                            e^{-h/\tau_{\text{syn,in}}}
                            (e^{h(1/\tau_{\text{syn,in}} - 1/\tau_m)} - 1) \\
       P_{20} &= -\frac{\tau_m}{C_m} (e^{-h/\tau_m} - 1)

    **3.2 Synaptic and Threshold Propagators**

    .. math::

       I_{\text{syn}}(t+h) &= I_{\text{syn}}(t) e^{-h/\tau_{\text{syn}}} + w_{\text{spike}} \\
       V_{th,1}(t+h) &= V_{th,1}(t) e^{-h/\tau_1} \\
       V_{th,2}(t+h) &= V_{th,2}(t) e^{-h/\tau_2}

    **3.3 Numerical Stability Constraint**

    The propagators become ill-conditioned when :math:`\tau_m \approx \tau_{\text{syn,ex}}`
    or :math:`\tau_m \approx \tau_{\text{syn,in}}` due to division by
    :math:`(1 - \tau_m/\tau_{\text{syn}})`. The constructor enforces strict inequality.

    **4. Update Order (NEST-Compatible)**

    For each time step (matching NEST's ``mat2_psc_exp.cpp``):

    1. **Integrate membrane potential** using exact propagators
    2. **Decay adaptive threshold components** (:math:`V_{th,1}`, :math:`V_{th,2}`)
    3. **Decay synaptic currents** and add incoming spike weights
    4. **Detect spikes**: if not refractory and :math:`V_m \geq V_{th}`, emit spike,
       jump threshold components, reset refractory counter
    5. **Update refractory state**: decrement counter if active
    6. **Buffer current inputs** for next step (:math:`I_0`)

    **5. Surrogate Gradient Handling**

    For differentiable training, the output spike signal passes through a surrogate
    gradient function (``spk_fun``). The voltage is scaled relative to the adaptive
    threshold:

    .. math::

       v_{\text{scaled}} = \frac{V_m - V_{th}}{|\omega - E_L|}

    where the denominator provides a characteristic voltage scale (~19 mV with defaults).

    Parameters
    ----------
    in_size : int, tuple of int
        Shape of the neuron population. Can be an integer (1D) or tuple (multi-dimensional).
    E_L : Quantity, ArrayLike, optional
        Resting membrane potential (default: -70 mV). Broadcastable to ``in_size``.
    C_m : Quantity, ArrayLike, optional
        Membrane capacitance (default: 100 pF). Must be strictly positive.
    tau_m : Quantity, ArrayLike, optional
        Membrane time constant (default: 5 ms). Must be strictly positive and differ
        from ``tau_syn_ex`` and ``tau_syn_in`` to avoid numerical degeneracy.
    t_ref : Quantity, ArrayLike, optional
        Duration of absolute refractory period (default: 2 ms). Must be strictly positive.
    tau_syn_ex : Quantity, ArrayLike, optional
        Time constant of excitatory postsynaptic current (default: 1 ms). Must be
        strictly positive and differ from ``tau_m``.
    tau_syn_in : Quantity, ArrayLike, optional
        Time constant of inhibitory postsynaptic current (default: 3 ms). Must be
        strictly positive and differ from ``tau_m``.
    I_e : Quantity, ArrayLike, optional
        Constant external input current (default: 0 pA). Broadcastable to ``in_size``.
    tau_1 : Quantity, ArrayLike, optional
        Short time constant of adaptive threshold (default: 10 ms). Must be strictly positive.
    tau_2 : Quantity, ArrayLike, optional
        Long time constant of adaptive threshold (default: 200 ms). Must be strictly positive.
    alpha_1 : Quantity, ArrayLike, optional
        Amplitude of short-timescale threshold jump on spike (default: 37 mV).
    alpha_2 : Quantity, ArrayLike, optional
        Amplitude of long-timescale threshold jump on spike (default: 2 mV).
    omega : Quantity, ArrayLike, optional
        Resting spike threshold (default: -51 mV). This is an **absolute voltage**,
        not relative to ``E_L``. With defaults, the threshold is 19 mV above resting.
    V_initializer : Callable, optional
        Initializer for membrane potential (default: Constant(-70 mV)). Called as
        ``V_initializer(shape, batch_size)`` to produce initial voltages.
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation (default: ReluGrad()).
        Maps scaled voltage to differentiable spike signal.
    spk_reset : str, optional
        Reset mode for spike output (default: ``'hard'``). Options: ``'hard'`` (stop gradient)
        or ``'soft'`` (preserve gradient). Does NOT affect membrane voltage reset
        (which never occurs in this model).
    ref_var : bool, optional
        If True, expose a boolean ``refractory`` state variable (default: False).
    name : str, optional
        Name of the neuron population. If None, auto-generated.

    Parameter Mapping
    -----------------
    Correspondence between constructor parameters and mathematical symbols:

    ==================== ================== =============================== ==========================================================
    **Parameter**        **Default**        **Math Symbol**                 **Description**
    ==================== ================== =============================== ==========================================================
    ``in_size``          (required)         —                               Population shape
    ``E_L``              -70 mV             :math:`E_L`                     Resting membrane potential
    ``C_m``              100 pF             :math:`C_m`                     Membrane capacitance
    ``tau_m``            5 ms               :math:`\tau_m`                  Membrane time constant
    ``t_ref``            2 ms               :math:`t_{\text{ref}}`          Duration of absolute refractory period
    ``tau_syn_ex``       1 ms               :math:`\tau_{\text{syn,ex}}`    Time constant of excitatory PSC
    ``tau_syn_in``       3 ms               :math:`\tau_{\text{syn,in}}`    Time constant of inhibitory PSC
    ``I_e``              0 pA               :math:`I_e`                     Constant external input current
    ``tau_1``            10 ms              :math:`\tau_1`                  Short time constant of adaptive threshold
    ``tau_2``            200 ms             :math:`\tau_2`                  Long time constant of adaptive threshold
    ``alpha_1``          37 mV              :math:`\alpha_1`                Amplitude of short-timescale threshold jump
    ``alpha_2``          2 mV               :math:`\alpha_2`                Amplitude of long-timescale threshold jump
    ``omega``            -51 mV             :math:`\omega`                  Resting spike threshold (absolute voltage)
    ``V_initializer``    Constant(-70 mV)   —                               Membrane potential initializer
    ``spk_fun``          ReluGrad()         —                               Surrogate spike function
    ``spk_reset``        ``'hard'``         —                               Reset mode (for gradient handling)
    ``ref_var``          ``False``          —                               If True, expose ``refractory`` boolean state
    ==================== ================== =============================== ==========================================================

    State Variables
    ---------------
    After ``init_state()``, the following state variables are available:

    ========================= ===================== ====================================================
    **Variable**              **Type**              **Description**
    ========================= ===================== ====================================================
    ``V``                     ``HiddenState`` (mV)  Membrane potential (absolute voltage)
    ``V_th_1``                ``ShortTermState``    Short-timescale adaptive threshold component (mV)
    ``V_th_2``                ``ShortTermState``    Long-timescale adaptive threshold component (mV)
    ``i_syn_ex``              ``ShortTermState``    Excitatory postsynaptic current (pA)
    ``i_syn_in``              ``ShortTermState``    Inhibitory postsynaptic current (pA)
    ``i_0``                   ``ShortTermState``    Buffered DC input current (pA, one-step delayed)
    ``refractory_step_count`` ``ShortTermState``    Refractory countdown (integer steps remaining)
    ``last_spike_time``       ``ShortTermState``    Time of last spike (ms)
    ``refractory``            ``ShortTermState``    Boolean refractory flag (only if ``ref_var=True``)
    ========================= ===================== ====================================================

    Raises
    ------
    ValueError
        If ``C_m <= 0``, ``tau_m <= 0``, ``tau_syn_ex <= 0``, ``tau_syn_in <= 0``,
        ``t_ref <= 0``, ``tau_1 <= 0``, or ``tau_2 <= 0``.
    ValueError
        If ``tau_m == tau_syn_ex`` or ``tau_m == tau_syn_in`` (numerical degeneracy).

    See Also
    --------
    amat2_psc_exp : Variant with exponential synaptic currents and additional state variables.
    iaf_psc_exp : Standard LIF with voltage reset (no adaptive threshold).
    aeif_psc_exp : Adaptive exponential IF with spike-triggered adaptation current.

    Notes
    -----
    **Biological Interpretation:**
    The MAT model captures spike frequency adaptation without explicit adaptation currents.
    The fast threshold component (τ₁ ~ 10 ms) models sodium channel inactivation,
    while the slow component (τ₂ ~ 200 ms) models calcium-dependent potassium currents.

    **Comparison to NEST:**
    This implementation matches NEST's ``mat2_psc_exp`` update order (see NEST 3.7+
    ``mat2_psc_exp.cpp``). Key differences:

    - **Surrogate gradients**: brainpy.state adds differentiable spike signals via ``spk_fun``
      for gradient-based learning; NEST uses exact spike times.
    - **Batch dimension**: brainpy.state supports batch processing for parallel simulations;
      NEST operates on single neuron instances.
    - **Precision**: brainpy.state uses float32 (JAX default); NEST uses float64. Minor
      numerical differences may occur for long simulations.

    **Performance Notes:**

    - Propagator computation (exponentials, ``expm1``) dominates runtime for small populations.
    - For large populations (>10k neurons), vectorized operations amortize this cost.
    - Use ``jax.jit`` compilation for optimal performance.

    References
    ----------
    .. [1] Rotter S and Diesmann M (1999). Exact simulation of time-invariant linear
           systems with applications to neuronal modeling. Biological Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    .. [2] Diesmann M, Gewaltig M-O, Rotter S, Aertsen A (2001). State space analysis of
           synchronous spiking in cortical neural networks. Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X
    .. [3] Kobayashi R, Tsubo Y and Shinomoto S (2009). Made-to-order spiking neuron model
           equipped with a multi-timescale adaptive threshold. Frontiers in Computational
           Neuroscience 3:9. DOI: https://doi.org/10.3389/neuro.10.009.2009

    Examples
    --------
    **Basic Usage:**

    .. code-block:: python

       >>> import brainpy.state as bp
       >>> import saiunit as u
       >>> # Create a population of 100 MAT neurons
       >>> neurons = bp.mat2_psc_exp(100, tau_1=10*u.ms, tau_2=200*u.ms)
       >>> neurons.init_all_states()
       >>> # Inject step current and simulate
       >>> with brainstate.environ.context(dt=0.1*u.ms):
       ...     spikes = neurons.update(500*u.pA)  # 500 pA step current

    **Demonstrating Adaptive Threshold:**

    .. code-block:: python

       >>> # Single neuron with strong adaptation
       >>> neuron = bp.mat2_psc_exp(1, alpha_1=50*u.mV, alpha_2=5*u.mV)
       >>> neuron.init_all_states()
       >>> with brainstate.environ.context(dt=0.1*u.ms):
       ...     V_trace = []
       ...     for _ in range(1000):  # 100 ms simulation
       ...         spk = neuron.update(800*u.pA)
       ...         V_trace.append(neuron.V.value)
       >>> # Plot V_trace to observe spike frequency adaptation

    **Network with Excitatory/Inhibitory Synapses:**

    .. code-block:: python

       >>> exc = bp.mat2_psc_exp(800)
       >>> inh = bp.mat2_psc_exp(200, tau_syn_ex=0.5*u.ms)
       >>> exc.init_all_states()
       >>> inh.init_all_states()
       >>> # Connect populations via projections (see brainpy.state.Projection)
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 100. * u.pF,
        tau_m: ArrayLike = 5. * u.ms,
        t_ref: ArrayLike = 2. * u.ms,
        tau_syn_ex: ArrayLike = 1. * u.ms,
        tau_syn_in: ArrayLike = 3. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        tau_1: ArrayLike = 10. * u.ms,
        tau_2: ArrayLike = 200. * u.ms,
        alpha_1: ArrayLike = 37. * u.mV,
        alpha_2: ArrayLike = 2. * u.mV,
        omega: ArrayLike = -51. * u.mV,
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
        self.omega = braintools.init.param(omega, self.varshape)

        self.V_initializer = V_initializer
        self.ref_var = ref_var
        self._validate_parameters()

        # Precompute refractory step count
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    @staticmethod
    def _to_numpy(x, unit):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.tau_m)):
            return

        if np.any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self.tau_m <= 0.0 * u.ms):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self.tau_syn_ex <= 0.0 * u.ms) or np.any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('Synaptic time constants must be strictly positive.')
        if np.any(self.t_ref <= 0.0 * u.ms):
            raise ValueError('Refractory time must be strictly positive.')
        if np.any(self.tau_1 <= 0.0 * u.ms) or np.any(self.tau_2 <= 0.0 * u.ms):
            raise ValueError('Adaptive threshold time constants must be strictly positive.')
        if np.any(self.tau_m == self.tau_syn_ex) or np.any(self.tau_m == self.tau_syn_in):
            raise ValueError(
                'Membrane and synapse time constant(s) must differ. '
                'See note in documentation.'
            )

    def init_state(self, **kwargs):
        ditype = brainstate.environ.ditype()

        V = braintools.init.param(self.V_initializer, self.varshape)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))

        self.V = brainstate.HiddenState(V)
        self.V_th_1 = brainstate.ShortTermState(zeros * u.mV)
        self.V_th_2 = brainstate.ShortTermState(zeros * u.mV)
        self.i_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.i_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.i_0 = brainstate.ShortTermState(zeros * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None, V_th: ArrayLike = None):
        r"""Compute surrogate gradient spike signal.

        Parameters
        ----------
        V : Quantity, ArrayLike, optional
            Membrane potential (mV). If None, uses current state ``self.V.value``.
        V_th : Quantity, ArrayLike, optional
            Effective threshold (mV). If None, computes as ``omega + V_th_1 + V_th_2``.

        Returns
        -------
        spike : ArrayLike
            Differentiable spike signal from surrogate function. Shape matches ``V``.

        Notes
        -----
        The voltage is scaled relative to the adaptive threshold before passing through
        the surrogate function, providing a normalized input that improves gradient stability.
        """
        V = self.V.value if V is None else V
        if V_th is None:
            V_th = self.omega + self.V_th_1.value + self.V_th_2.value
        # Scale relative to the effective adaptive threshold.
        v_scaled = (V - V_th) / u.math.abs(self.omega - self.E_L)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.pA):
        r"""Advance the neuron state by one time step.

        Implements the NEST-compatible update order for the MAT2 model with exact
        integration of subthreshold dynamics.

        Parameters
        ----------
        x : Quantity, ArrayLike, optional
            External input current (pA) for this time step (default: 0 pA).
            Broadcastable to population shape. This current is buffered and applied
            in the **next** time step (one-step delay).

        Returns
        -------
        spike : ArrayLike
            Differentiable spike signal for this time step. Shape matches population size.

        Notes
        -----
        **Update sequence (NEST-compatible):**

        1. Integrate membrane potential using exact propagators
        2. Decay adaptive threshold components (V_th_1, V_th_2)
        3. Decay synaptic currents and add incoming spike weights
        4. Detect spikes: if not refractory and V_m >= V_th, emit spike
        5. On spike: jump threshold components, reset refractory counter
        6. Buffer external current for next step

        **Key implementation details:**

        - Membrane potential is **never reset** (non-resetting LIF)
        - Spike detection compares V_m against the adaptive threshold V_th = ω + V_th_1 + V_th_2
        - Refractory period is implemented as an integer countdown; no voltage clamping
        - External current ``x`` is stored in ``i_0`` and applied in the **next** time step

        **Numerical stability:**

        The exact integration scheme requires tau_m ≠ tau_syn_ex and tau_m ≠ tau_syn_in.
        Violations of this constraint are caught during initialization.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        # Extract all parameters as plain float64 numpy arrays
        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        tau_1 = self._broadcast_to_state(self._to_numpy(self.tau_1, u.ms), v_shape)
        tau_2 = self._broadcast_to_state(self._to_numpy(self.tau_2, u.ms), v_shape)
        alpha_1 = self._broadcast_to_state(self._to_numpy(self.alpha_1, u.mV), v_shape)
        alpha_2 = self._broadcast_to_state(self._to_numpy(self.alpha_2, u.mV), v_shape)
        # omega is stored as absolute mV; convert to relative to E_L
        omega_rel = self._broadcast_to_state(self._to_numpy(self.omega - self.E_L, u.mV), v_shape)

        # State variables (all relative to E_L for V_m, relative to omega for thresholds)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        V_th_1 = self._broadcast_to_state(self._to_numpy(self.V_th_1.value, u.mV), v_shape)
        V_th_2 = self._broadcast_to_state(self._to_numpy(self.V_th_2.value, u.mV), v_shape)
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value, u.pA), v_shape)
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.i_syn_in.value, u.pA), v_shape)
        i_0 = self._broadcast_to_state(self._to_numpy(self.i_0.value, u.pA), v_shape)
        ditype = brainstate.environ.ditype()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=ditype), v_shape
        )

        # --- Compute propagator coefficients ---
        # Membrane potential propagators (exact integration)
        P11ex = np.exp(-h / tau_ex)
        P11in = np.exp(-h / tau_in)
        P22_expm1 = np.expm1(-h / tau_m)

        P21ex = -tau_m / (C_m * (1.0 - tau_m / tau_ex)) * P11ex * np.expm1(h * (1.0 / tau_ex - 1.0 / tau_m))
        P21in = -tau_m / (C_m * (1.0 - tau_m / tau_in)) * P11in * np.expm1(h * (1.0 / tau_in - 1.0 / tau_m))
        P20 = -tau_m / C_m * P22_expm1

        # Adaptive threshold propagators
        P11th = np.exp(-h / tau_1)
        P22th = np.exp(-h / tau_2)

        # --- Get spike inputs ---
        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all > 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)

        # --- Get current inputs (one-step delayed, stored for next step) ---
        i_0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        # === NEST update ordering (mat2_psc_exp.cpp lines 316-358) ===

        # Step 1: Evolve membrane potential
        V_rel = V_rel * P22_expm1 + V_rel + i_syn_ex * P21ex + i_syn_in * P21in + (I_e + i_0) * P20

        # Step 2: Evolve adaptive threshold
        V_th_1 = V_th_1 * P11th
        V_th_2 = V_th_2 * P22th

        # Step 3: Decay synaptic currents and add incoming spikes
        i_syn_ex = i_syn_ex * P11ex
        i_syn_in = i_syn_in * P11in
        i_syn_ex = i_syn_ex + w_ex
        i_syn_in = i_syn_in + w_in

        # Step 4-5: Spike detection (no voltage reset!)
        not_refractory = r == 0
        spike_cond = not_refractory & (V_rel >= omega_rel + V_th_1 + V_th_2)

        # On spike: jump threshold components, set refractory counter
        V_th_1 = np.where(spike_cond, V_th_1 + alpha_1, V_th_1)
        V_th_2 = np.where(spike_cond, V_th_2 + alpha_2, V_th_2)
        r = np.where(
            spike_cond,
            self._broadcast_to_state(
                np.asarray(u.math.asarray(self.ref_count), dtype=ditype), v_shape
            ),
            np.where(not_refractory, r, r - 1),
        )

        # Step 6: Store buffered currents for next step
        # (i_0 is updated with the new current input)

        # --- Write back state variables ---
        self.V.value = (V_rel + E_L) * u.mV
        self.V_th_1.value = V_th_1 * u.mV
        self.V_th_2.value = V_th_2 * u.mV
        self.i_syn_ex.value = i_syn_ex * u.pA
        self.i_syn_in.value = i_syn_in * u.pA
        self.i_0.value = i_0_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=ditype)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        # Return spike output: for surrogate gradient, emit above-threshold signal.
        # The effective threshold (absolute) is E_L + omega_rel + V_th_1 + V_th_2.
        # When spiking, produce V slightly above threshold; otherwise, slightly below.
        V_th_abs = omega_rel + V_th_1 + V_th_2 + E_L
        V_out = np.where(spike_cond, V_th_abs + 1e-12, V_th_abs - 1e-12)
        return self.get_spike(V_out * u.mV, V_th_abs * u.mV)
