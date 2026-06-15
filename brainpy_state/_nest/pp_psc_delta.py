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

from typing import Callable, Optional, Sequence

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron
from ._utils import is_tracer, cond_any

__all__ = [
    'pp_psc_delta',
]


class pp_psc_delta(NESTNeuron):
    r"""Point process neuron with leaky integration of delta-shaped PSCs.

    ``pp_psc_delta`` is an implementation of a leaky integrator where the
    potential jumps on each spike arrival. It produces spikes stochastically
    according to a transfer function operating on the membrane potential, and
    supports spike-frequency adaptation with multiple exponential kernels.

    This is a brainpy.state re-implementation of the NEST simulator model of
    the same name, using NEST-standard parameterization and exact integration.

    Parameters
    ----------
    in_size : int, tuple of int
        Population shape. Defines the number of neurons in the population.
    tau_m : Quantity, optional
        Membrane time constant. Must be a positive quantity with time units.
        Default: 10.0 ms.
    C_m : Quantity, optional
        Membrane capacitance. Must be a positive quantity with capacitance units.
        Default: 250.0 pF.
    dead_time : float, optional
        Duration of the dead time (absolute refractory period) in milliseconds.
        If set to 0, the model operates in Poisson mode with potentially multiple
        spikes per time step. If ``dead_time`` is nonzero but smaller than the
        simulation resolution, it is clamped to the resolution. Must be non-negative.
        Default: 1.0 ms.
    dead_time_random : bool, optional
        Whether to draw random dead time after each spike from a gamma distribution.
        If True, ``dead_time`` becomes the mean of the gamma distribution with
        shape parameter ``dead_time_shape``. Default: False.
    dead_time_shape : int, optional
        Shape parameter of the gamma distribution for random dead times. Must be
        at least 1. Default: 1.
    with_reset : bool, optional
        Whether to reset the membrane potential to 0 after each spike. Default: True.
    tau_sfa : tuple of float, optional
        Adaptive threshold time constants in milliseconds. Each element defines
        the decay time constant of one adaptation kernel. Must be a sequence of
        positive values with the same length as ``q_sfa``. Default: () (no adaptation).
    q_sfa : tuple of float, optional
        Adaptive threshold jump sizes in millivolts. Each element defines the
        increment added to the corresponding adaptation kernel on each spike.
        Must be a sequence with the same length as ``tau_sfa``. Default: () (no adaptation).
    c_1 : float, optional
        Slope of the linear part of the transfer function in Hz/mV. Default: 0.0.
    c_2 : float, optional
        Prefactor of the exponential part of the transfer function in Hz. Can be
        used as an offset spike rate when ``c_3 = 0``. Default: 1.238 Hz.
    c_3 : float, optional
        Coefficient of exponential nonlinearity in 1/mV. Must be non-negative.
        Set to 0 for purely linear transfer function. Default: 0.25 1/mV.
    I_e : Quantity, optional
        Constant external input current. Must be a quantity with current units.
        Default: 0.0 pA.
    t_ref_remaining : float, optional
        Remaining dead time at simulation start in milliseconds. Must be non-negative.
        Default: 0.0 ms.
    rng_key : jax.Array, optional
        JAX PRNG key for stochastic spike generation. If None, a default key is
        used. For reproducible results, provide an explicit key. Default: None.
    V_initializer : Callable, optional
        Initializer for the membrane potential (relative to resting potential).
        Default: ``Constant(0.0 * u.mV)``.
    spk_fun : Callable, optional
        Surrogate spike function for differentiable spike generation. Default:
        ``ReluGrad()``.
    spk_reset : str, optional
        Reset mode. Options: ``'hard'`` (stop gradient), ``'soft'`` (V -= V_th).
        Default: ``'hard'`` (matches NEST behavior).
    name : str, optional
        Name of the neuron population. Default: None.

    Raises
    ------
    ValueError
        If ``C_m <= 0`` (capacitance must be strictly positive).
    ValueError
        If ``tau_m <= 0`` (membrane time constant must be strictly positive).
    ValueError
        If ``dead_time < 0`` (dead time must be non-negative).
    ValueError
        If ``dead_time_shape < 1`` (gamma shape parameter must be at least 1).
    ValueError
        If ``t_ref_remaining < 0`` (remaining refractory time must be non-negative).
    ValueError
        If ``c_3 < 0`` (exponential coefficient must be non-negative).
    ValueError
        If any element of ``tau_sfa <= 0`` (adaptation time constants must be positive).
    ValueError
        If ``len(tau_sfa) != len(q_sfa)`` (adaptation parameter lists must match).

    See Also
    --------
    iaf_psc_delta : Integrate-and-fire neuron with delta PSCs
    gif_psc_exp : Generalized integrate-and-fire with exponential PSCs

    Parameter Mapping
    -----------------

    ========================= ====================== ===============================================
    **NEST Parameter**        **brainpy.state**      **Notes**
    ========================= ====================== ===============================================
    ``tau_m``                 ``tau_m``              Membrane time constant
    ``C_m``                   ``C_m``                Membrane capacitance
    ``dead_time``             ``dead_time``          Refractory period duration
    ``dead_time_random``      ``dead_time_random``   Enable random dead time
    ``dead_time_shape``       ``dead_time_shape``    Gamma distribution shape parameter
    ``with_reset``            ``with_reset``         Reset ``V_m`` after spike
    ``tau_sfa``               ``tau_sfa``            Adaptation time constants (list)
    ``q_sfa``                 ``q_sfa``              Adaptation jump sizes (list)
    ``c_1``                   ``c_1``                Linear transfer function coefficient
    ``c_2``                   ``c_2``                Exponential transfer function prefactor
    ``c_3``                   ``c_3``                Exponential transfer function exponent
    ``I_e``                   ``I_e``                External input current
    ``t_ref_remaining``       ``t_ref_remaining``    Initial refractory time
    ``V_m``                   ``V.value``            Membrane potential (relative to rest)
    ``E_sfa``                 ``_q_val``             Sum of all adaptation elements
    ========================= ====================== ===============================================

    **1. Mathematical Model**

    **1.1. Membrane Dynamics**

    The membrane potential :math:`V_\mathrm{m}` (relative to resting potential)
    evolves according to a leaky integrator:

    .. math::

       C_\mathrm{m} \frac{dV_\mathrm{m}}{dt} = -\frac{V_\mathrm{m}}{\tau_\mathrm{m}}
       + I_\mathrm{e} + I_\mathrm{syn}(t)

    where:

    - :math:`C_\mathrm{m}` is the membrane capacitance
    - :math:`\tau_\mathrm{m}` is the membrane time constant
    - :math:`I_\mathrm{e}` is the constant external input current
    - :math:`I_\mathrm{syn}(t)` is the synaptic input current

    The exact (analytic) integration over one time step :math:`h` gives:

    .. math::

       V_\mathrm{m}(t + h) = P_{33} \cdot V_\mathrm{m}(t)
       + P_{30} \cdot (I_0 + I_\mathrm{e})
       + w_\mathrm{syn}

    where:

    - :math:`P_{33} = \exp(-h / \tau_\mathrm{m})`
    - :math:`P_{30} = \frac{\tau_\mathrm{m}}{C_\mathrm{m}}(1 - P_{33})`
    - :math:`I_0` is the buffered current from the previous step (ring buffer)
    - :math:`w_\mathrm{syn}` is the sum of all incoming delta-shaped PSP jumps (in mV)

    **1.2. Transfer Function**

    The instantaneous firing rate is computed from the effective membrane potential
    :math:`V' = V_\mathrm{m} - E_\mathrm{sfa}` using a flexible transfer function:

    .. math::

       \text{rate}(t) = \text{Rect}\!\left[
           c_1 \cdot V'(t) + c_2 \cdot \exp(c_3 \cdot V'(t))
       \right]

    where :math:`\text{Rect}(x) = \max(0, x)` ensures non-negative rates.

    By adjusting ``c_1``, ``c_2``, and ``c_3``, the transfer function can be:

    - Linear: Set ``c_3 = 0``, ``c_1 > 0`` -- :math:`\text{rate} = c_1 V' + c_2`
    - Exponential: Set ``c_1 = 0`` -- :math:`\text{rate} = c_2 \exp(c_3 V')`
    - Mixed: All coefficients nonzero -- linear + exponential

    **1.3. Spike-Frequency Adaptation**

    The adaptive threshold :math:`E_\mathrm{sfa}` is the sum of multiple exponential
    kernels, each with its own time constant and jump size:

    .. math::

       \tau_{\mathrm{sfa},i} \frac{dE_{\mathrm{sfa},i}}{dt} = -E_{\mathrm{sfa},i}

    .. math::

       E_{\mathrm{sfa},i}(t) \to E_{\mathrm{sfa},i}(t) + q_{\mathrm{sfa},i}
       \quad \text{(on spike)}

    .. math::

       E_\mathrm{sfa}(t) = \sum_{i=1}^{n} E_{\mathrm{sfa},i}(t)

    The adaptation kernels decay exponentially with exact propagators:

    .. math::

       E_{\mathrm{sfa},i}(t + h) = E_{\mathrm{sfa},i}(t) \exp(-h / \tau_{\mathrm{sfa},i})

    **1.4. Stochastic Spike Generation**

    - With dead time (``dead_time > 0``): At most one spike per time step.
      A uniform random number :math:`u \sim \mathcal{U}(0,1)` is compared to
      the spike probability:

      .. math::

         P(\text{spike}) = 1 - \exp(-\text{rate} \cdot h \cdot 10^{-3})

      A spike is generated if :math:`u \le P(\text{spike})`.

    - Without dead time (``dead_time = 0``): Multiple spikes per step are
      possible. The number of spikes is drawn from a Poisson distribution:

      .. math::

         n_{\text{spikes}} \sim \text{Poisson}(\text{rate} \cdot h \cdot 10^{-3})

    The factor :math:`10^{-3}` converts from Hz*ms to a dimensionless rate.

    **1.5. Dead Time (Refractory Period)**

    After each spike, the neuron enters a dead time during which it cannot spike:

    - Fixed dead time: ``dead_time_random = False``. The neuron is refractory
      for exactly ``dead_time`` milliseconds, converted to grid steps.
    - Random dead time: ``dead_time_random = True``. The dead time is drawn
      from a gamma distribution with shape ``dead_time_shape`` and mean ``dead_time``.

    If ``dead_time`` is nonzero but smaller than the simulation resolution :math:`h`,
    it is clamped to :math:`h`.

    **2. Numerical Integration and Update Order**

    The discrete-time update per simulation step follows this order:

    1. **Update membrane potential** via exact propagator (including external
       current and synaptic delta inputs).
    2. **Decay adaptation elements** and compute total :math:`E_\mathrm{sfa}`.
    3. **Spike check**:

       - If not refractory: compute effective potential
         :math:`V' = V_\mathrm{m} - E_\mathrm{sfa}`,
         compute instantaneous rate, draw random number and potentially emit spike(s).
         If spike occurs:

         - Jump all adaptation elements by ``q_sfa``
         - Optionally reset :math:`V_\mathrm{m}` to 0 (if ``with_reset = True``)
         - Set dead time counter

       - If refractory: decrement dead time counter

    4. **Buffer external current** for the next step (ring buffer semantics).

    **3. Important Implementation Notes**

    - Relative membrane potential: The membrane potential :math:`V_\mathrm{m}`
      is stored relative to the resting potential (resting potential = 0 mV).
      This differs from ``iaf_psc_delta``, which uses absolute potentials.
    - Stochastic reproducibility: Because spiking is stochastic (random number
      drawn each step), exact spike-time reproducibility requires matching the
      random number generator state. For deterministic testing, set ``rng_key``
      explicitly.
    - Dead time < dt clamping: If ``dead_time`` is nonzero but smaller than
      the simulation resolution, it is internally clamped to the resolution to
      match NEST behavior.
    - Poisson mode performance: For non-refractory neurons (``dead_time = 0``),
      Poisson random draws are used, which are slower than uniform random draws.
      For typical firing rates (<1 spike/time_step), setting a small ``dead_time``
      (e.g., 1e-8 ms) is faster and nearly equivalent.

    **4. State Variables**

    ============================== ================= ==========================================
    **State Variable**             **Type**          **Description**
    ============================== ================= ==========================================
    ``V``                          HiddenState       Membrane potential (relative to rest)
    ``refractory_step_count``      ShortTermState    Remaining dead time grid steps
    ``I_stim``                     ShortTermState    Buffered current applied in next step
    ``last_spike_time``            ShortTermState    Last spike time (for recording)
    ``_q_elems``                   HiddenState       Adaptation kernel elements (internal)
    ``_q_val``                     ShortTermState    Total :math:`E_\mathrm{sfa}` (internal)
    ``_rng_state``                 JAX PRNG key      Random number generator state (internal)
    ============================== ================= ==========================================

    - Default parameter values match NEST C++ source for ``pp_psc_delta``,
      which are based on Jolivet et al. (2006) [2]_.
    - ``tau_sfa`` and ``q_sfa`` default to empty tuples (no adaptation).
      In NEST, the C++ defaults of ``tau_sfa=34.0`` and ``q_sfa=0.0`` are
      immediately cleared in the constructor, resulting in empty vectors.
    - The recordable ``V_m`` in NEST corresponds to ``self.V.value`` in brainpy.state.
    - The recordable ``E_sfa`` in NEST corresponds to ``self._q_val`` (the sum of
      all adaptation elements).

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

    Examples
    --------
    Basic usage with default parameters:

    .. code-block:: python

       >>> from brainpy import state as bst
       >>> import brainunit as u
       >>> neurons = bst.pp_psc_delta(100)
       >>> neurons.init_all_states()

    Exponential transfer function (default):

    .. code-block:: python

       >>> neurons = bst.pp_psc_delta(
       ...     100,
       ...     c_1=0.0,      # no linear part
       ...     c_2=1.238,    # exponential prefactor
       ...     c_3=0.25      # exponential coefficient
       ... )

    Linear transfer function with offset:

    .. code-block:: python

       >>> neurons = bst.pp_psc_delta(
       ...     100,
       ...     c_1=10.0,     # linear slope (Hz/mV)
       ...     c_2=5.0,      # offset rate (Hz)
       ...     c_3=0.0       # disable exponential
       ... )

    With spike-frequency adaptation:

    .. code-block:: python

       >>> neurons = bst.pp_psc_delta(
       ...     100,
       ...     tau_sfa=(100.0, 1000.0),  # two adaptation kernels
       ...     q_sfa=(5.0, 10.0)          # jump sizes in mV
       ... )

    Poisson mode (no dead time):

    .. code-block:: python

       >>> neurons = bst.pp_psc_delta(
       ...     100,
       ...     dead_time=0.0  # multiple spikes per step possible
       ... )

    Random dead time:

    .. code-block:: python

       >>> neurons = bst.pp_psc_delta(
       ...     100,
       ...     dead_time=2.0,           # mean dead time (ms)
       ...     dead_time_random=True,   # enable random dead time
       ...     dead_time_shape=2        # gamma distribution shape
       ... )

    Reproducible stochastic behavior:

    .. code-block:: python

       >>> import jax
       >>> key = jax.random.PRNGKey(42)
       >>> neurons = bst.pp_psc_delta(100, rng_key=key)
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

    def _validate_parameters(self):
        r"""Validate all model parameters.

        Raises
        ------
        ValueError
            If any parameter is outside its valid range.

        Notes
        -----
        Validation checks:

        - ``C_m > 0`` (capacitance must be positive)
        - ``tau_m > 0`` (membrane time constant must be positive)
        - ``dead_time >= 0`` (dead time must be non-negative)
        - ``dead_time_shape >= 1`` (gamma shape must be at least 1)
        - ``t_ref_remaining >= 0`` (remaining refractory time must be non-negative)
        - ``c_3 >= 0`` (exponential coefficient must be non-negative)
        - All elements of ``tau_sfa > 0`` (adaptation time constants must be positive)
        - ``len(tau_sfa) == len(q_sfa)`` (adaptation parameter lists must match)
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.tau_m)):
            return

        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.tau_m <= 0.0 * u.ms):
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

    def _precompute_constants(self, state_shape):
        r"""Pre-compute dt-dependent propagator constants for JIT compatibility.

        Called from :meth:`init_state` while ``dt`` is still a concrete Python
        value (not a JAX abstract tracer).  Storing these as plain Python floats
        or static JAX arrays avoids ``ConcretizationTypeError`` inside
        ``jax.lax.scan`` / ``brainstate.transform.for_loop``.

        Parameters
        ----------
        state_shape : tuple
            Shape of the membrane-potential state array (``V.shape`` after
            initialization).  Used to pre-shape the adaptation-decay and
            q_sfa-jump arrays for broadcasting against ``(n_sfa, *state_shape)``.
        """
        dt_q = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # ---- h in ms as a concrete Python float ----
        # brainstate.environ.get_dt() returns a Python brainunit.Quantity, so
        # dividing by u.ms yields a plain Python float — no JAX array created.
        self._h_ms = float(dt_q / u.ms)

        # ---- Membrane propagator (computed once, reused every step) ----
        self._P33 = u.math.exp(-dt_q / self.tau_m)
        self._P30 = (1.0 / self.C_m) * (1.0 - self._P33) * self.tau_m

        # ---- Effective dead time (clamped to dt if nonzero but smaller) ----
        dead_time = self.dead_time
        if dead_time != 0.0 and dead_time < self._h_ms:
            dead_time = self._h_ms
        self._dead_time_eff = dead_time  # Python float constant

        # ---- Dead time in grid steps (Python int constant) ----
        if self._dead_time_eff > 0.0:
            self._dead_time_counts = int(round(self._dead_time_eff / self._h_ms))
        else:
            self._dead_time_counts = 0

        # ---- Adaptation decay factors (pre-shaped for broadcasting) ----
        n_sfa = len(self.tau_sfa)
        if n_sfa > 0:
            P_sfa_1d = jnp.array(
                [np.exp(-self._h_ms / tau) for tau in self.tau_sfa], dtype=dftype
            )
            # Reshape to (n_sfa,) + (1,) * len(state_shape) so it broadcasts
            # against q_elems of shape (n_sfa, *state_shape).
            P_sfa = P_sfa_1d
            for _ in range(len(state_shape)):
                P_sfa = jnp.expand_dims(P_sfa, axis=-1)
            self._P_sfa = P_sfa
        else:
            self._P_sfa = None

        # ---- q_sfa jump array (pre-shaped for broadcasting) ----
        if n_sfa > 0:
            q_sfa_arr = jnp.array(self.q_sfa, dtype=dftype)
            for _ in range(len(state_shape)):
                q_sfa_arr = jnp.expand_dims(q_sfa_arr, axis=-1)
            self._q_sfa_arr = q_sfa_arr
        else:
            self._q_sfa_arr = None

    def init_state(self, batch_size=None, **kwargs):
        r"""Initialize all state variables.

        Allocates and initializes membrane potential, spike times, refractory
        counters, buffered currents, adaptation kernels, and random number
        generator state.

        Parameters
        ----------
        batch_size : int or None, optional
            If provided, states are created with shape ``(batch_size, *varshape)``
            to support batched simulation. If None, states have shape ``varshape``.
        **kwargs : dict, optional
            Additional keyword arguments (ignored).

        Notes
        -----
        - Membrane potential is initialized using ``V_initializer``.
        - Last spike time is initialized to -1e7 ms (sufficiently in the past).
        - Refractory counter is initialized based on ``t_ref_remaining``.
        - Adaptation kernels (``_q_elems``) are initialized to zero.
        - Random number generator state is initialized from ``rng_key`` or
          a default key.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()

        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        state_shape = V.shape
        self.V = brainstate.HiddenState(V)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(state_shape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(state_shape, 0, dtype=ditype))
        self.I_stim = brainstate.ShortTermState(u.math.full(state_shape, 0.0 * u.pA, dtype=dftype))

        # Adaptation state: q_elems array stored as JAX arrays (mV units)
        n_sfa = len(self.tau_sfa)
        if n_sfa > 0:
            self._q_elems = brainstate.HiddenState(
                u.math.zeros((n_sfa, *state_shape), dtype=dftype) * u.mV
            )
        else:
            self._q_elems = None
        self._q_val = brainstate.ShortTermState(
            u.math.zeros(state_shape, dtype=dftype) * u.mV
        )

        # Pre-compute dt-dependent propagator constants (must happen after
        # state_shape is known, and while dt is still a concrete Python value).
        self._precompute_constants(state_shape)

        # Initialize remaining dead time from parameter (uses _h_ms from above)
        if self.t_ref_remaining > 0.0:
            r_init = int(round(self.t_ref_remaining / self._h_ms))
            self.refractory_step_count.value = u.math.full(state_shape, r_init, dtype=ditype)

        # RNG state wrapped in ShortTermState so for_loop carries it correctly
        if self._rng_key is not None:
            rng = self._rng_key
        else:
            rng = jax.random.PRNGKey(0)
        self._rng_state = brainstate.ShortTermState(rng)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute surrogate gradient spike output for backpropagation.

        This method is used for computing differentiable spike outputs during
        training. For a stochastic point process neuron, the true spike output
        is random and computed in ``update()``. This method provides a surrogate
        gradient based on the membrane potential.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (with units). If None, uses the current state
            ``self.V.value``. Default: None.

        Returns
        -------
        spike : jax.Array
            Differentiable spike output. Shape matches ``V``.

        Notes
        -----
        - This method is primarily used for gradient-based optimization.
        - The surrogate gradient is computed by scaling the membrane potential
          and passing it through ``spk_fun`` (e.g., ``ReluGrad``).
        - The true stochastic spike output is computed in ``update()`` and is
          not directly differentiable.
        """
        V = self.V.value if V is None else V
        # For a stochastic model, we use V directly scaled by a reasonable factor
        v_scaled = V / (1.0 * u.mV)
        return self.spk_fun(v_scaled)

    def update(self, x=0.0 * u.pA):
        r"""Update neuron state for one simulation step.

        Performs the complete update sequence: (1) updates membrane potential
        via exact propagator, (2) decays adaptation kernels, (3) computes
        instantaneous firing rate and stochastically generates spikes, (4) buffers
        input current for the next step.

        Parameters
        ----------
        x : Quantity, optional
            External current input (with current units). This input is added to
            the sum of all registered current inputs via projections. Default: 0.0 pA.

        Returns
        -------
        spike : jax.Array
            Binary spike output array. Shape: ``in_size``. Values are 1.0 where
            spikes occurred, 0.0 otherwise. In Poisson mode (``dead_time = 0``),
            values can be integers > 1 representing multiple spikes per step.

        Notes
        -----
        **Update order per time step:**

        1. **Membrane potential update**: Apply exact propagator to update
           :math:`V_\mathrm{m}` using buffered current from the previous step,
           constant external current, and delta-shaped synaptic inputs.

        2. **Adaptation decay**: Decay all adaptation kernel elements using
           exponential propagators. Compute total :math:`E_\mathrm{sfa}`.

        3. **Spike generation**:

           - If not refractory: compute effective potential :math:`V' = V_\mathrm{m} - E_\mathrm{sfa}`,
             compute instantaneous rate from transfer function, draw random
             number(s), and potentially emit spike(s).
           - If spike occurs: jump adaptation elements by ``q_sfa``, optionally
             reset :math:`V_\mathrm{m}` to 0, set dead time counter.
           - If refractory: decrement dead time counter.

        4. **Buffer input**: Store external current input for the next step
           (ring buffer semantics, matching NEST).

        **Spike generation modes:**

        - With dead time (``dead_time > 0``): At most one spike per step.
          Uses uniform random numbers and spike probability.
        - Without dead time (``dead_time = 0``): Poisson-distributed spikes.
          Multiple spikes per step are possible.

        **Failure modes:**

        - If ``C_m`` or ``tau_m`` contain invalid values (NaN, Inf), membrane
          potential update will fail silently (produces NaN).
        - If ``c_3 * V'`` causes overflow in ``exp()``, the exponential term
          will saturate to infinity. The rectifier ensures the rate remains
          non-negative.
        - If random number generator state is corrupted, spike generation will
          produce undefined results.

        **Performance considerations:**

        - Poisson mode (``dead_time = 0``) is slower due to Poisson random draws.
        - Setting a small ``dead_time`` (e.g., 1e-8 ms) uses faster uniform
          random numbers and is nearly equivalent for typical firing rates.
        - Random dead time (``dead_time_random = True``) requires additional
          gamma distribution samples per spike.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Use pre-computed h_ms (Python float, safe inside JIT/scan).
        # _precompute_constants() stored this in init_state() when dt was concrete.
        h_ms = self._h_ms

        # Read state variables
        V = self.V.value  # mV
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        state_shape = V.shape  # (batch_size, *varshape) or varshape

        # ---- Step 1: Update membrane potential via exact propagator ----
        # Use pre-computed propagator coefficients (P33, P30 from init_state).
        delta_v = self.sum_delta_inputs(u.math.zeros(self.varshape) * u.mV)
        V = self._P30 * (i_stim + self.I_e) + self._P33 * V + delta_v

        # ---- Step 2: Decay adaptation elements and compute total E_sfa ----
        n_sfa = len(self.tau_sfa)
        if n_sfa > 0 and self._q_elems is not None:
            q_elems = self._q_elems.value  # (n_sfa, *state_shape) in mV
            # Use pre-shaped P_sfa (shape: (n_sfa,) + (1,)*len(state_shape))
            q_elems = q_elems * self._P_sfa
            q_total = u.math.sum(q_elems, axis=0)  # shape: state_shape, in mV
        else:
            q_elems = None
            q_total = u.math.zeros(self.varshape) * u.mV

        # ---- Step 3: Spike check / refractory ----
        not_refractory = r == 0

        # Compute effective potential and transfer function rate
        V_eff = V - q_total  # mV
        V_eff_raw = V_eff / u.mV  # unitless

        # Transfer function: rate = rect(c_1 * V_eff + c_2 * exp(c_3 * V_eff))
        # Clip c_3 * V_eff to prevent overflow
        exp_arg = jnp.clip(self.c_3 * V_eff_raw, -500.0, 500.0)
        rate = self.c_1 * V_eff_raw + self.c_2 * jnp.exp(exp_arg)
        rate = jnp.maximum(rate, 0.0)  # rectifier

        # Advance RNG state for this step
        rng_state, subkey = jax.random.split(self._rng_state.value)
        self._rng_state.value = rng_state

        # Use pre-computed effective dead time and grid-step count.
        dead_time = self._dead_time_eff  # Python float constant

        if dead_time > 0.0:
            # With dead time: at most 1 spike per step
            # spike_prob = 1 - exp(-rate * h * 1e-3)  = -expm1(-rate * h * 1e-3)
            spike_prob = -jnp.expm1(-rate * h_ms * 1e-3)
            rand_vals = jax.random.uniform(subkey, shape=state_shape, dtype=dftype)

            spike_now = not_refractory & (rate > 0.0) & (rand_vals <= spike_prob)

            # Set dead time counter
            if self.dead_time_random:
                # Gamma-distributed dead time
                _, gamma_key = jax.random.split(subkey)
                gamma_samples = jax.random.gamma(
                    gamma_key, self.dead_time_shape, shape=state_shape, dtype=dftype
                )
                dt_rate = self.dead_time_shape / dead_time
                new_r_random = jnp.maximum(1, jnp.round(gamma_samples / dt_rate / h_ms).astype(ditype))
                new_r = jnp.where(spike_now, new_r_random, r)
            else:
                new_r = jnp.where(spike_now, self._dead_time_counts, r)

            n_spikes = jnp.where(spike_now, 1, 0).astype(ditype)
        else:
            # Without dead time (Poisson mode): multiple spikes per step possible
            lam_poisson = rate * h_ms * 1e-3
            n_spikes_raw = jax.random.poisson(subkey, lam_poisson, shape=state_shape, dtype=ditype)
            n_spikes = jnp.where(not_refractory & (rate > 0.0), n_spikes_raw, 0)
            spike_now = n_spikes > 0
            new_r = r  # no dead time to set

        # Decrement refractory counter for neurons that did NOT spike
        # (neurons still in refractory period)
        refractory_and_no_spike = (r > 0) & ~spike_now
        new_r = jnp.where(refractory_and_no_spike, r - 1, new_r)

        # Jump adaptation elements on spike (use pre-shaped q_sfa_arr)
        if n_sfa > 0 and q_elems is not None:
            n_spikes_float = jnp.expand_dims(n_spikes.astype(dftype), axis=0)
            q_elems = q_elems + (self._q_sfa_arr * n_spikes_float) * u.mV
            q_total = u.math.sum(q_elems, axis=0)

        # Reset membrane potential if applicable
        if self.with_reset:
            V = u.math.where(spike_now, 0.0 * u.mV, V)

        # ---- Step 4: Get external current for NEXT step (NEST ring buffer semantics) ----
        new_i_stim = self.sum_current_inputs(x, self.V.value)

        # ---- Write back state ----
        self.V.value = V
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(new_r), dtype=ditype)
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_now, t + dt_q, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if n_sfa > 0 and self._q_elems is not None:
            self._q_elems.value = q_elems
        self._q_val.value = q_total

        spike_mask = spike_now if dead_time > 0.0 else (n_spikes > 0)
        return u.math.asarray(spike_mask, dtype=dftype)
