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

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

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
    ``_q_elems``                   NumPy array       Adaptation kernel elements (internal)
    ``_q_val``                     NumPy array       Total :math:`E_\mathrm{sfa}` (internal)
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

       >>> import brainpy.state as bst
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

    @staticmethod
    def _to_numpy(x, unit):
        r"""Convert brainunit quantity to NumPy array in specified units.

        Parameters
        ----------
        x : Quantity or array-like
            Input value (with or without units).
        unit : Quantity
            Target unit for conversion (e.g., ``u.mV``, ``u.ms``, ``u.pA``).

        Returns
        -------
        numpy.ndarray
            NumPy array with values in the specified unit, dtype=float64.

        Notes
        -----
        This method strips units and converts to float64 for use in NumPy-based
        numerical integration loops where brainunit operations would be too slow.
        """
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast NumPy array to match state variable shape.

        Parameters
        ----------
        x_np : numpy.ndarray
            Input array (scalar or array).
        shape : tuple
            Target shape (including batch dimension if present).

        Returns
        -------
        numpy.ndarray
            Broadcasted array with shape matching ``shape``.

        Notes
        -----
        This method ensures that scalar parameters (e.g., ``tau_m``) are
        broadcast to match the full population shape for element-wise operations.
        """
        return np.broadcast_to(x_np, shape)

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
        r"""Initialize all state variables.

        Allocates and initializes membrane potential, spike times, refractory
        counters, buffered currents, adaptation kernels, and random number
        generator state.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for vectorized simulation. If None, no batch dimension
            is added. Default: None.
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
        r"""Reset all state variables to initial values.

        Resets membrane potential, spike times, refractory counters, buffered
        currents, adaptation kernels, and random number generator state to
        their initial configurations.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for vectorized simulation. If None, no batch dimension
            is added. Default: None.
        **kwargs : dict, optional
            Additional keyword arguments (ignored).

        Notes
        -----
        This method performs the same initialization as ``init_state``. It is
        useful for resetting the model state during a simulation without
        recreating the model instance.
        """
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
            Binary spike output array. Shape: ``(batch_size, *in_size)`` if
            batched, ``in_size`` otherwise. Values are 1.0 where spikes occurred,
            0.0 otherwise. In Poisson mode (``dead_time = 0``), values can be
            integers > 1 representing multiple spikes per step.

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
                            int(rand_vals[idx] * 2 ** 31)
                        ).poisson(lam_poisson))

                    if n_spikes > 0:
                        spike_mask[idx] = True
                        n_spikes_arr[idx] = n_spikes

                        # Set dead time
                        if self.dead_time_random:
                            # Gamma-distributed dead time
                            gamma_sample = np.random.RandomState(
                                int(rand_vals[idx] * 2 ** 30 + 1)
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
