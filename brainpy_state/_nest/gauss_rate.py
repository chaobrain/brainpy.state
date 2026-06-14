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
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest.lin_rate import _lin_rate_base
from ._utils import is_tracer

__all__ = [
    'gauss_rate_ipn',
]


class _gauss_rate_base(_lin_rate_base):
    __module__ = 'brainpy.state'

    #: φ(h) = g·exp(−(h−μ)²/2σ²) is fixed by the gain, centre and width; these
    #: identify it for the ``linear_summation=False`` homogeneity guard.
    _phi_param_names = ('g', 'mu', 'sigma')

    def _activation(self, h):
        """Gaussian gain ``φ(h) = g·exp(−(h−μ)²/2σ²)`` (JAX; reads ``self``).

        ``μ`` (gain centre) and ``σ`` (gain width) are NEST's dual-role ``mu`` /
        ``sigma`` parameters. As in NEST, ``σ=0`` leaves ``φ`` undefined at
        ``h=μ`` (a ``0/0`` → ``NaN``).
        """
        g = u.get_mantissa(self.g)
        mu = u.get_mantissa(self.mu)
        sigma = u.get_mantissa(self.sigma)
        return g * jnp.exp(-jnp.square(h - mu) / (2.0 * jnp.square(sigma)))

    def _mult_factors(self, rate):
        one = jnp.ones_like(rate)
        return one, one

    def _common_parameters_gauss(self, state_shape):
        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        sigma = self._broadcast_to_state(self._to_numpy(self.sigma), state_shape)
        mu = self._broadcast_to_state(self._to_numpy(self.mu), state_shape)
        g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
        return tau, sigma, mu, g


class gauss_rate_ipn(_gauss_rate_base):
    r"""NEST-compatible ``gauss_rate_ipn`` nonlinear rate neuron with input noise.

    Implements a stochastic rate-based neuron with Gaussian gain function and input
    noise, matching NEST's ``gauss_rate_ipn`` model. The dynamics combine passive
    decay, mean drive, network input (processed through a Gaussian nonlinearity),
    and additive Brownian noise.

    **1. Model equations**

    The stochastic differential equation governing the rate dynamics is:

    .. math::

       \tau\,dX(t)=
       \left[-\lambda X(t)+\mu+I_{\mathrm{net}}(t)\right]dt
       +\left[\sqrt{\tau}\,\sigma\right]dW(t),

    where :math:`W(t)` is a standard Wiener process and :math:`I_{\mathrm{net}}(t)`
    is the effective network input after applying the Gaussian gain function
    :math:`\phi(h)`:

    .. math::

       \phi(h)=g\exp\left(-\frac{(h-\mu)^2}{2\sigma^2}\right).

    The gain function produces a bell-shaped response centered at :math:`\mu` with
    width controlled by :math:`\sigma` and amplitude scaled by :math:`g`.

    **2. NEST parameter coupling (critical implementation detail)**

    NEST's ``gauss_rate_ipn`` model uses the same parameter names ``mu`` and
    ``sigma`` for two distinct purposes:

    1. **SDE parameters**: ``mu`` is the mean drive in the drift term;
       ``sigma`` scales the diffusion coefficient (input noise strength).
    2. **Gain-function parameters**: ``mu`` is the Gaussian center (location of
       peak response); ``sigma`` is the Gaussian width (standard deviation of
       the bell curve).

    This implementation preserves NEST's dual-role design. Consequently:

    - The default ``sigma=0.0`` from NEST (no input noise) is retained.
    - When ``sigma=0``, the gain function becomes undefined at ``h=mu`` (0/0 form),
      potentially producing ``NaN`` values, matching NEST behavior.
    - Both roles share the same parameter instance, so changes affect both the
      SDE noise term and the gain-function shape.

    **3. Update ordering (matching NEST ``rate_neuron_ipn_impl.h``)**

    Per simulation step of duration ``dt``:

    1. **Store outgoing delayed value**: Current ``rate`` becomes ``delayed_rate``.
    2. **Draw noise sample**: Compute ``noise = sigma * xi`` where ``xi ~ N(0,1)``.
    3. **Propagate intrinsic dynamics**: Apply stochastic exponential Euler
       (reduces to Euler-Maruyama when ``lambda=0``):

       .. math::

          X_{\mathrm{new}} = e^{-\lambda h/\tau}X_{\mathrm{prev}}
          + \frac{1-e^{-\lambda h/\tau}}{\lambda}(\mu + \mu_{\mathrm{ext}})
          + \sqrt{\frac{1-e^{-2\lambda h/\tau}}{2\lambda}}\,\sigma\xi,

       where :math:`h=dt` and special handling applies when ``lambda=0``.

    4. **Drain delayed queues**: Retrieve and sum delayed excitatory/inhibitory
       contributions scheduled for the current step.
    5. **Process instantaneous events**: Parse and accumulate ``instant_rate_events``
       and zero-delay entries from ``delayed_rate_events``.
    6. **Apply Gaussian gain function**:

       - ``linear_summation=True``: Sum all network inputs, then apply :math:`\phi`.
       - ``linear_summation=False``: Apply :math:`\phi` to each event value before
         summation (nonlinearity applied during event buffering).

    7. **Include multiplicative coupling** (if enabled): Scale excitatory/inhibitory
       branches by state-dependent factors (trivially ``1.0`` for this model).
    8. **Apply rectification** (if enabled): Clamp ``rate >= rectify_rate``.
    9. **Store outgoing instantaneous value**: Updated ``rate`` becomes
       ``instant_rate`` for immediate event transmission.

    **4. Assumptions and constraints**

    Mathematical validity:

    - ``tau > 0`` (time constant must be positive).
    - ``lambda >= 0`` (passive decay rate must be non-negative).
    - ``sigma >= 0`` (noise/gain width cannot be negative).
    - When ``sigma=0``, the gain function is undefined at ``h=mu``, matching NEST's
      potential NaN generation.

    Event semantics:

    - Events are specified as ``(rate, weight)`` tuples, ``(rate, weight, delay_steps)``
      triples, ``(rate, weight, delay_steps, multiplicity)`` 4-tuples, or dicts with
      ``'rate'``, ``'weight'``, ``'delay_steps'``, ``'multiplicity'`` keys.
    - ``instant_rate_events`` must have ``delay_steps=0`` (enforced with exception).
    - ``delayed_rate_events`` support integer ``delay_steps >= 0``.
    - Negative weights create inhibitory contributions (sign-based routing).

    **5. Computational implications**

    Integration method: Stochastic exponential Euler is exact for linear drift with
    additive noise (Ornstein-Uhlenbeck process) but approximate when network input
    is present. Accuracy degrades if ``dt`` is not sufficiently small relative to
    ``tau/lambda``.

    Delay queue management: Each delayed event is stored in a dictionary keyed by
    target step index. Memory scales with the number of active delayed events.
    Unbounded delays can lead to memory growth.

    Gaussian evaluation: Computing :math:`\exp(-(h-\mu)^2/(2\sigma^2))` per event
    (when ``linear_summation=False``) or per step (when ``linear_summation=True``)
    is vectorized via NumPy. For ``sigma=0``, evaluations at ``h=mu`` produce NaN.

    Parameters
    ----------
    in_size : Size
        Population shape specification. Determines ``self.varshape`` and the shape
        of state variables ``rate``, ``noise``, etc. Can be an integer (1D population)
        or tuple of integers (multi-dimensional population).
    tau : Quantity[ms], optional
        Time constant :math:`\tau` of rate dynamics. Must be positive. Controls the
        temporal scale of both drift and diffusion terms. Default ``10 ms``.
    lambda_ : float, optional
        Passive decay rate :math:`\lambda \ge 0`. When ``lambda=0``, dynamics reduce
        to driftless Brownian motion with external drive. Larger values produce
        stronger relaxation toward the mean drive. Default ``1.0``.
    sigma : float, optional
        Shared dual-role parameter (matching NEST):

        1. **Diffusion coefficient**: Scales input noise as :math:`\sqrt{\tau}\sigma dW(t)`.
        2. **Gaussian width**: Standard deviation of the gain function :math:`\phi(h)`.

        Must be non-negative. NEST default ``0.0`` (no noise, but gain function
        becomes undefined at ``h=mu``). Default ``0.0``.
    mu : float, optional
        Shared dual-role parameter (matching NEST):

        1. **Mean drive**: Constant drift term in the SDE.
        2. **Gaussian center**: Location of peak response in :math:`\phi(h)`.

        Default ``0.0``.
    g : float, optional
        Gain amplitude parameter. Scales the maximum value of the Gaussian nonlinearity
        :math:`\phi(h)`. When ``g=1``, peak response is 1.0 at ``h=mu``.
        Default ``1.0``.
    mult_coupling : bool, optional
        Enable multiplicative coupling (state-dependent input scaling). For
        ``gauss_rate_ipn``, the coupling factors are trivially ``1.0`` (no effect),
        but the parameter is retained for NEST API compatibility. Default ``False``.
    linear_summation : bool, optional
        NEST switch controlling where the Gaussian nonlinearity is applied:

        - ``True`` (default): Sum all network inputs first, then apply :math:`\phi`
          to the total. Results in :math:`\phi(\sum h_i w_i)`.
        - ``False``: Apply :math:`\phi` to each event's rate value during buffering,
          then sum the transformed contributions. Results in :math:`\sum \phi(h_i) w_i`.

        Default ``True``.
    rectify_rate : float, optional
        Lower bound for output clamping when ``rectify_output=True``. Must be
        non-negative. Default ``0.0``.
    rectify_output : bool, optional
        If ``True``, apply rectification ``rate = max(rate, rectify_rate)`` after
        all updates. Prevents negative firing rates. Default ``False``.
    rate_initializer : Callable, optional
        Initializer for the ``rate`` state variable. Called with ``(shape, batch_size)``
        to produce initial firing rates. Default ``braintools.init.Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for the ``noise`` state variable (stores last noise sample).
        Default ``braintools.init.Constant(0.0)``.
    name : str or None, optional
        Module identifier. Default ``None``.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to NEST and model symbols
       :header-rows: 1
       :widths: 20 18 18 44

       * - Parameter
         - Default
         - Math symbol
         - Semantics / NEST correspondence
       * - ``tau``
         - ``10 ms``
         - :math:`\tau`
         - Time constant of rate dynamics (NEST: ``tau``).
       * - ``lambda_``
         - ``1.0``
         - :math:`\lambda`
         - Passive decay rate (NEST: ``lambda`` in template parameters).
       * - ``sigma``
         - ``0.0``
         - :math:`\sigma`
         - Dual role: input-noise scale in SDE **and** Gaussian width in :math:`\phi(h)` (NEST: ``sigma``).
       * - ``mu``
         - ``0.0``
         - :math:`\mu`
         - Dual role: mean drive in SDE **and** Gaussian center in :math:`\phi(h)` (NEST: ``mu``).
       * - ``g``
         - ``1.0``
         - :math:`g`
         - Gain amplitude of Gaussian nonlinearity (NEST: ``g`` in template nonlinearity).
       * - ``rectify_rate``
         - ``0.0``
         - :math:`r_{\mathrm{min}}`
         - Lower bound for output rectification (NEST: ``rectify_rate``).
       * - ``rectify_output``
         - ``False``
         - —
         - Enable output clamping to :math:`\ge r_{\mathrm{min}}` (NEST: ``rectify_output``).
       * - ``linear_summation``
         - ``True``
         - —
         - Apply :math:`\phi` to sum (``True``) vs. per-event (``False``) (NEST: ``linear_summation``).
       * - ``mult_coupling``
         - ``False``
         - —
         - Enable multiplicative coupling (no-op for this model, NEST compatibility only).

    Attributes
    ----------
    rate : brainstate.ShortTermState
        Current firing rate of shape ``self.varshape``. Updated each step.
    noise : brainstate.ShortTermState
        Last noise sample :math:`\sigma\xi` of shape ``self.varshape``.
    instant_rate : brainstate.ShortTermState
        Copy of ``rate`` after update, used for zero-delay event transmission.
    delayed_rate : brainstate.ShortTermState
        Copy of ``rate`` before update, used for non-zero delay event transmission.

    Notes
    -----
    **Runtime event semantics**:

    - ``instant_rate_events``: Applied in the current step with zero delay.
      Format: scalar, ``(rate, weight)``, ``(rate, weight, 0)``,
      ``(rate, weight, 0, multiplicity)``, or dict with keys
      ``'rate'``, ``'weight'``, ``'delay_steps'`` (must be 0), ``'multiplicity'``.
    - ``delayed_rate_events``: Scheduled for future delivery based on ``delay_steps``.
      Format: same as above, but ``delay_steps`` can be any non-negative integer.
    - ``x``: External current input (additive to ``mu``), summed via
      ``sum_current_inputs(x, rate)``.

    **Failure modes**:

    - **NaN generation**: When ``sigma=0`` and network input ``h`` exactly equals
      ``mu``, the Gaussian :math:`\phi(h) = g \exp(0/0)` is undefined. NEST also
      produces NaN in this case.
    - **Non-increasing ``amplitude_times``**: Raises ``ValueError`` during construction
      if delay queues are misconfigured (internal logic error).
    - **Invalid event delays**: ``instant_rate_events`` with non-zero ``delay_steps``
      raise ``ValueError``. Negative delays in ``delayed_rate_events`` also raise
      ``ValueError``.

    **Relationship to other models**:

    - ``gauss_rate_ipn`` is the NEST input-noise template instantiated with Gaussian
      nonlinearities. The base template ``rate_neuron_ipn`` supports arbitrary input
      nonlinearities and multiplicative-coupling functions.
    - ``gauss_rate_opn`` is the output-noise variant (noise added after nonlinearity).
    - For linear gain (``g * h``), use ``lin_rate_ipn`` instead.

    Examples
    --------
    Minimal usage with default parameters:

    .. code-block:: python

        >>> import brainpy_state as bpst
        >>> import saiunit as u
        >>> import brainstate
        >>> brainstate.environ.set_dt(0.1 * u.ms)
        >>> neuron = bpst.gauss_rate_ipn(in_size=100)
        >>> neuron.init_all_states()
        >>> # Simulate 10 steps with no input
        >>> for _ in range(10):
        ...     rate = neuron.update()

    With network events and external drive:

    .. code-block:: python

        >>> neuron = bpst.gauss_rate_ipn(
        ...     in_size=50,
        ...     tau=20.0 * u.ms,
        ...     lambda_=1.5,
        ...     sigma=0.5,
        ...     mu=0.0,
        ...     g=2.0,
        ...     linear_summation=True,
        ...     rectify_output=True,
        ...     rectify_rate=0.0,
        ... )
        >>> neuron.init_all_states()
        >>> # Apply instantaneous rate input and delayed event
        >>> rate = neuron.update(
        ...     x=1.0,  # external drive
        ...     instant_rate_events=(0.5, 1.0),  # (rate, weight)
        ...     delayed_rate_events=(1.0, 2.0, 5),  # (rate, weight, delay_steps)
        ... )

    References
    ----------
    .. [1] Hahne J, Dahmen D, Schuecker J, Frommer A, Bolten M, Helias M,
           Diesmann M (2017). Integration of continuous-time dynamics in a
           spiking neural network simulator. *Frontiers in Neuroinformatics*, 11:34.
           DOI: `10.3389/fninf.2017.00034 <https://doi.org/10.3389/fninf.2017.00034>`_.
    .. [2] Hahne J, Helias M, Kunkel S, Igarashi J, Bolten M, Frommer A,
           Diesmann M (2015). A unified framework for spiking and gap-junction
           interactions in distributed neuronal network simulations.
           *Frontiers in Neuroinformatics*, 9:22.
           DOI: `10.3389/fninf.2015.00022 <https://doi.org/10.3389/fninf.2015.00022>`_.
    .. [3] NEST Documentation: ``gauss_rate_ipn`` model.
           https://nest-simulator.readthedocs.io/en/stable/models/gauss_rate_ipn.html
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau: ArrayLike = 10.0 * u.ms,
        lambda_: ArrayLike = 1.0,
        sigma: ArrayLike = 0.0,
        mu: ArrayLike = 0.0,
        g: ArrayLike = 1.0,
        mult_coupling: bool = False,
        linear_summation: bool = True,
        rectify_rate: ArrayLike = 0.0,
        rectify_output: bool = False,
        rate_initializer: Callable = braintools.init.Constant(0.0),
        noise_initializer: Callable = braintools.init.Constant(0.0),
        name: str = None,
    ):
        super().__init__(
            in_size=in_size,
            tau=tau,
            sigma=sigma,
            mu=mu,
            g=g,
            mult_coupling=mult_coupling,
            g_ex=1.0,
            g_in=1.0,
            theta_ex=0.0,
            theta_in=0.0,
            linear_summation=linear_summation,
            rate_initializer=rate_initializer,
            noise_initializer=noise_initializer,
            name=name,
        )
        self.lambda_ = braintools.init.param(lambda_, self.varshape)
        self.rectify_rate = braintools.init.param(rectify_rate, self.varshape)
        self.rectify_output = bool(rectify_output)
        self._validate_parameters()

    @property
    def recordables(self):
        r"""List of state variable names that can be recorded during simulation.

        Returns
        -------
        list of str
            ``['rate', 'noise']`` — firing rate and noise sample state variables.

        Notes
        -----
        These names correspond to attributes that can be monitored by recording
        devices or logged during simulation. Accessing ``neuron.rate.value`` and
        ``neuron.noise.value`` retrieves the current values.
        """
        return ['rate', 'noise']

    @property
    def receptor_types(self):
        r"""Dictionary mapping receptor port names to integer indices.

        Returns
        -------
        dict
            ``{'RATE': 0}`` — single receptor type for rate-based input.

        Notes
        -----
        NEST uses receptor types to distinguish synaptic input channels (e.g.,
        AMPA, NMDA, GABA). For ``gauss_rate_ipn``, only one generic ``'RATE'``
        receptor is defined. This is used for NEST API compatibility but has no
        functional effect in this implementation (excitatory/inhibitory routing
        is based on weight sign, not receptor type).
        """
        return {'RATE': 0}

    def _validate_parameters(self):
        r"""Check parameter validity and raise exceptions for invalid configurations.

        Enforces mathematical and physical constraints on model parameters:

        - ``tau > 0`` (time constant must be positive)
        - ``lambda >= 0`` (passive decay rate must be non-negative)
        - ``sigma >= 0`` (noise/gain width must be non-negative)
        - ``rectify_rate >= 0`` (lower rectification bound must be non-negative)

        Raises
        ------
        ValueError
            If any parameter violates its constraint, with a descriptive message
            indicating which parameter is invalid.

        Notes
        -----
        This method is called automatically during ``__init__``. It does not validate
        the ``sigma=0`` special case (undefined gain function at ``h=mu``), as NEST
        permits this configuration despite potential NaN generation.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.tau, self.sigma)):
            return
        if np.any(self.tau <= 0.0 * u.ms):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self.lambda_ < 0.0):
            raise ValueError('Passive decay rate lambda must be >= 0.')
        if np.any(self.sigma < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')
        if np.any(self.rectify_rate < 0.0):
            raise ValueError('Rectifying rate must be >= 0.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables and internal delay queues.

        Creates and initializes firing rate, noise, and auxiliary state variables
        required for event-driven simulation with delayed transmission.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        This method initializes:

        - ``self.rate``: Current firing rate, initialized via ``rate_initializer``.
        - ``self.noise``: Last noise sample, initialized via ``noise_initializer``.
        - ``self.instant_rate``: Copy of ``rate`` for zero-delay transmission.
        - ``self.delayed_rate``: Copy of ``rate`` for delayed transmission.
        - ``self._step_count``: Internal step counter (int64 scalar).
        - ``self._delayed_ex_queue``: Dictionary ``{step_idx: excitatory_contribution}``
          for delayed excitatory events.
        - ``self._delayed_in_queue``: Dictionary ``{step_idx: inhibitory_contribution}``
          for delayed inhibitory events.

        All queues are empty at initialization. The step counter starts at 0.

        Examples
        --------
        .. code-block:: python

            >>> neuron = bpst.gauss_rate_ipn(in_size=100)
            >>> neuron.init_state()
            >>> neuron.rate.value.shape
            (100,)
        """
        rate = braintools.init.param(self.rate_initializer, self.varshape)
        noise = braintools.init.param(self.noise_initializer, self.varshape)
        rate_np = self._to_numpy(rate)
        noise_np = self._to_numpy(noise)

        self.rate = brainstate.ShortTermState(rate_np)
        self.noise = brainstate.ShortTermState(noise_np)
        dftype = brainstate.environ.dftype()
        self.instant_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))
        self._alloc_phi_rate(rate_np)

    def update(self, x=0.0, noise=None):
        r"""Advance dynamics by one time step using stochastic exponential Euler.

        Implements the complete NEST ``gauss_rate_ipn`` update cycle: store delayed
        output, draw noise, integrate SDE, process events, apply Gaussian nonlinearity,
        and update firing rate.

        Parameters
        ----------
        x : ArrayLike, optional
            External additive drive (current input), broadcast to ``self.varshape``.
            Added to ``mu`` in the drift term. Can be scalar, array-like, or have
            ``saiunit`` units (automatically converted). Default ``0.0``.
        instant_rate_events : scalar, tuple, list of tuples, or None, optional
            Rate events applied in the current step with zero delay. Each event can be:

            - Scalar: Interpreted as ``(value, weight=1.0)``.
            - ``(rate, weight)``: Rate value and synaptic weight.
            - ``(rate, weight, delay_steps)``: Must have ``delay_steps=0`` (raises
              ``ValueError`` otherwise).
            - ``(rate, weight, delay_steps, multiplicity)``: 4-tuple with multiplicity
              factor.
            - Dict with keys ``'rate'``, ``'weight'``, ``'delay_steps'``,
              ``'multiplicity'``.

            Weights are signed: positive for excitatory, negative for inhibitory.
            Default ``None`` (no events).
        delayed_rate_events : scalar, tuple, list of tuples, or None, optional
            Rate events scheduled for future delivery based on ``delay_steps``.
            Format is the same as ``instant_rate_events``, but ``delay_steps`` can be
            any non-negative integer. Zero-delay events are applied immediately.
            Negative delays raise ``ValueError``. Default ``None``.
        noise : ArrayLike or None, optional
            Optional external noise sample :math:`\xi` to use instead of drawing from
            :math:`N(0,1)`. Must be broadcast-compatible with ``self.varshape``.
            When ``None``, standard normal noise is drawn internally. Useful for
            reproducible testing. Default ``None``.

        Returns
        -------
        rate_new : ndarray
            Updated firing rate of shape matching ``self.rate.value.shape``, after
            applying all dynamics, network input, Gaussian nonlinearity, multiplicative
            coupling, and optional rectification.

        Raises
        ------
        ValueError
            - If any ``instant_rate_events`` entry specifies non-zero ``delay_steps``.
            - If any ``delayed_rate_events`` entry has negative ``delay_steps``.

        Notes
        -----
        **Integration method**: Stochastic exponential Euler for the linear part of
        the SDE, with network input and Gaussian nonlinearity applied as an additive
        perturbation scaled by the integration factor ``P2``.

        **Update propagation coefficients**:

        - ``P1 = exp(-lambda * dt / tau)``: State persistence factor.
        - ``P2 = (1 - exp(-lambda * dt / tau)) / lambda``: Input integration factor
          (reduces to ``dt / tau`` when ``lambda=0``).
        - ``input_noise_factor = sqrt((1 - exp(-2*lambda*dt/tau)) / (2*lambda))``:
          Diffusion coefficient (reduces to ``sqrt(dt / tau)`` when ``lambda=0``).

        **Gaussian nonlinearity application**:

        - ``linear_summation=True``: Compute ``phi(sum(excitatory) + sum(inhibitory))``.
        - ``linear_summation=False``: Each event's rate is transformed during buffering,
          so summed values already include ``phi`` applied per event.

        **Multiplicative coupling**: For ``gauss_rate_ipn``, factors ``H_ex`` and
        ``H_in`` are trivially ``1.0`` (no-op), but the code path is present for
        NEST compatibility.

        **Rectification**: If ``rectify_output=True``, the final rate is clamped to
        ``max(rate_new, rectify_rate)``.

        **State side effects**: Updates ``self.rate``, ``self.noise``,
        ``self.delayed_rate``, ``self.instant_rate``, ``self._step_count``, and
        modifies delay queues ``self._delayed_ex_queue`` and ``self._delayed_in_queue``.
        """
        h = float(u.get_mantissa(brainstate.environ.get_dt() / u.ms))
        dftype = brainstate.environ.dftype()
        state_shape = self.rate.value.shape

        tau, sigma, mu, g = self._common_parameters_gauss(state_shape)
        lambda_ = self._broadcast_to_state(self._to_numpy(self.lambda_), state_shape)
        rectify_rate = self._broadcast_to_state(self._to_numpy(self.rectify_rate), state_shape)

        rate_prev = jnp.broadcast_to(jnp.asarray(self.rate.value, dtype=dftype), state_shape)

        mu_ext, h_a, h_b = self._read_coupling(x)

        if noise is None:
            xi = brainstate.random.randn(*state_shape)
        else:
            xi = jnp.broadcast_to(jnp.asarray(noise, dtype=dftype), state_shape)
        noise_now = sigma * xi

        if np.any(lambda_ > 0.0):
            P1 = np.exp(-lambda_ * h / tau)
            P2 = -np.expm1(-lambda_ * h / tau) / np.where(lambda_ == 0.0, 1.0, lambda_)
            input_noise_factor = np.sqrt(
                -0.5 * np.expm1(-2.0 * lambda_ * h / tau) / np.where(lambda_ == 0.0, 1.0, lambda_)
            )
            zero_lambda = lambda_ == 0.0
            if np.any(zero_lambda):
                P1 = np.where(zero_lambda, 1.0, P1)
                P2 = np.where(zero_lambda, h / tau, P2)
                input_noise_factor = np.where(zero_lambda, np.sqrt(h / tau), input_noise_factor)
        else:
            P1 = np.ones_like(lambda_)
            P2 = h / tau
            input_noise_factor = np.sqrt(h / tau)

        mu_total = mu + mu_ext
        rate_new = P1 * rate_prev + P2 * mu_total + input_noise_factor * noise_now
        rate_new = rate_new + P2 * self._coupling_increment(rate_prev, h_a, h_b)

        if self.rectify_output:
            rate_new = jnp.where(rate_new < rectify_rate, rectify_rate, rate_new)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.delayed_rate.value = rate_prev
        self.instant_rate.value = rate_new
        self._store_phi_rate(rate_new)
        return rate_new
