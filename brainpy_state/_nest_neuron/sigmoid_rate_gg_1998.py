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
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_neuron.lin_rate import _lin_rate_base
from brainpy_state._nest_base.utils import is_tracer

__all__ = [
    'sigmoid_rate_gg_1998_ipn',
]


class _sigmoid_rate_gg_1998_base(_lin_rate_base):
    r"""Base class for Gancarz-Grossberg (1998) quartic gain function rate neurons.

    Provides the quartic gain function :math:`\phi(h) = (gh)^4 / (0.1^4 + (gh)^4)`
    and event/input processing infrastructure specific to ``sigmoid_rate_gg_1998``
    variants. Inherits common rate neuron machinery from ``_lin_rate_base``.
    """
    __module__ = 'brainpy.state'

    def _activation(self, h):
        """Gancarz-Grossberg quartic gain ``φ(h) = (g·h)^4 / (0.1^4 + (g·h)^4)`` (JAX)."""
        gh4 = jnp.power(u.get_mantissa(self.g) * h, 4.0)
        return gh4 / (0.1 ** 4 + gh4)

    def _mult_factors(self, rate):
        one = jnp.ones_like(rate)
        return one, one

    def _common_parameters_sigmoid_gg_1998(self, state_shape):
        r"""Broadcast model parameters to state shape.

        Converts time constant to milliseconds and broadcasts all parameters to
        match the current state array shape.

        Parameters
        ----------
        state_shape : tuple of int
            Target broadcast shape (``rate.value.shape``).

        Returns
        -------
        tau : ndarray
            Time constant in milliseconds (shape ``state_shape``).
        sigma : ndarray
            Noise amplitude (shape ``state_shape``).
        mu : ndarray
            Constant drive (shape ``state_shape``).
        g : ndarray
            Gain parameter (shape ``state_shape``).
        """
        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        sigma = self._broadcast_to_state(self._to_numpy(self.sigma), state_shape)
        mu = self._broadcast_to_state(self._to_numpy(self.mu), state_shape)
        g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
        return tau, sigma, mu, g


class sigmoid_rate_gg_1998_ipn(_sigmoid_rate_gg_1998_base):
    r"""NEST-compatible ``sigmoid_rate_gg_1998_ipn`` nonlinear rate neuron with input noise.

    Description
    -----------

    ``sigmoid_rate_gg_1998_ipn`` implements NEST's ``sigmoid_rate_gg_1998_ipn`` model:
    a continuous-time rate neuron driven by stochastic input noise and shaped by the
    historical Gancarz-Grossberg (1998) quartic gain function. The model corresponds
    to NEST's ``rate_neuron_ipn`` template instantiated with ``sigmoid_rate_gg_1998``
    nonlinearity. Multiplicative coupling factors are fixed to one for this model variant
    (the flag is kept for API compatibility).

    **1. Continuous-time stochastic dynamics**

    The neuron rate :math:`X(t)` evolves under the Itô stochastic differential equation:

    .. math::

       \tau\,dX(t) = \left[-\lambda X(t) + \mu + \phi(\cdot)\right]dt
       + \left[\sqrt{\tau}\,\sigma\right]dW(t),

    where :math:`\tau > 0` is the intrinsic time constant (ms), :math:`\lambda \ge 0`
    is the passive decay rate, :math:`\mu` is the constant mean drive, :math:`\sigma \ge 0`
    is the input noise amplitude, and :math:`W(t)` is a standard Wiener process.

    **2. Gancarz-Grossberg quartic gain function**

    The gain function :math:`\phi(h)` applies a steep, saturating nonlinearity to
    summed synaptic input :math:`h`:

    .. math::

       \phi(h) = \frac{(g\,h)^4}{0.1^4 + (g\,h)^4},

    where :math:`g > 0` is the gain parameter controlling horizontal scaling. This
    form, introduced by Gancarz & Grossberg (1998) for saccade generation modeling,
    provides faster saturation than standard sigmoid or Hill functions and approaches
    a binary step near threshold. The constant denominator term :math:`0.1^4 = 10^{-4}`
    sets the half-activation point at :math:`h = 0.1/g` when :math:`g=1`.

    Compared with the standard sigmoid :math:`g/(1+e^{-\beta(h-\theta)})`, the quartic
    gain exhibits steeper rise and faster approach to asymptotic bounds, reflecting
    winner-take-all dynamics in reticular formation circuits.

    **3. Numerical integration scheme**

    At each time step :math:`h = dt`, the model applies the stochastic exponential
    Euler (Euler-Maruyama) scheme:

    .. math::

       X_{n+1} = P_1\,X_n + P_2\,(\mu + \mu_{ext} + \phi(\cdot))
       + \sqrt{\frac{\tau\,(-0.5\,\mathrm{expm1}(-2\lambda h/\tau))}{\lambda}}\,\xi_n,

    where :math:`P_1 = e^{-\lambda h/\tau}`,
    :math:`P_2 = -\mathrm{expm1}(-\lambda h/\tau)/\lambda` for :math:`\lambda > 0`,
    and :math:`\xi_n \sim \mathcal{N}(0, 1)`. For :math:`\lambda = 0`, the propagators
    reduce to :math:`P_1 = 1`, :math:`P_2 = h/\tau`, and the noise factor becomes
    :math:`\sqrt{h/\tau}`.

    The external drive :math:`\mu_{ext}` aggregates continuous current inputs and
    delta inputs via ``sum_current_inputs()`` and ``sum_delta_inputs()``.

    **4. Synaptic input routing and delayed events**

    When ``linear_summation=True`` (default), the gain :math:`\phi` applies to the
    total summed synaptic input across all sources:

    .. math::

       \phi(\cdot) = \phi(h_{ex,\mathrm{inst}} + h_{in,\mathrm{inst}}
       + h_{ex,\mathrm{del}} + h_{in,\mathrm{del}}),

    where subscripts denote instantaneous vs. delayed, excitatory vs. inhibitory
    branch sums. When ``linear_summation=False``, the gain applies to each incoming
    rate event **before** weighted summation, matching NEST's per-event nonlinearity
    mode.

    Delayed events are stored in per-step queues (``_delayed_ex_queue``,
    ``_delayed_in_queue``) indexed by target step. Event polarity (excitatory or
    inhibitory) is determined by the sign of the weight field. Events support
    dict, tuple, or scalar formats:

    - ``{'rate': r, 'weight': w, 'delay_steps': d, 'multiplicity': m}``
    - ``(r, w, d, m)`` or shorter tuples (defaults applied)
    - Scalar ``r`` (weight=1, delay_steps=0, multiplicity=1)

    **5. Update ordering**

    Per simulation step (matching NEST ``rate_neuron_ipn`` with ``sigmoid_rate_gg_1998``):

    1. Store outgoing delayed value as current ``rate`` (``delayed_rate``).
    2. Draw noise realization :math:`\xi_n`.
    3. Propagate intrinsic dynamics :math:`-\lambda X` with stochastic exponential Euler.
    4. Read delayed and instantaneous event buffers, schedule new delayed events.
    5. Apply gain function to branch sums (if ``linear_summation=True``) or per event
       (if ``linear_summation=False``).
    6. Apply rectification: if ``rectify_output=True``, clamp ``rate`` to
       :math:`\ge` ``rectify_rate``.
    7. Store outgoing instantaneous value as updated ``rate`` (``instant_rate``).

    **6. Assumptions, constraints, and failure modes**

    - Construction-time validation ensures ``tau > 0``, ``lambda >= 0``, ``sigma >= 0``,
      ``rectify_rate >= 0``.
    - Parameters are scalar or broadcastable to ``self.varshape``.
    - Delayed event ``delay_steps`` must be non-negative integers; instantaneous events
      (``instant_rate_events``) must have ``delay_steps=0`` or omit the field.
    - Multiplicative coupling factors are identically one (``_mult_coupling_ex/in``
      return ``ones_like``). Enabling ``mult_coupling=True`` has no effect for this model.
    - Per-step complexity is :math:`O(|\mathrm{state}| \cdot K)` for ``K`` events per step.

    Parameters
    ----------
    in_size : Size
        Population shape. Supports integer ``n`` (1D with ``n`` neurons), tuple ``(n, m)``
        (2D grid), or higher-dimensional tuples.
    tau : Quantity[ms], optional
        Intrinsic time constant :math:`\tau` of rate dynamics (ms). Must be positive.
        Default ``10 ms``.
    lambda_ : float or array_like, optional
        Passive decay rate :math:`\lambda` (dimensionless). Must be non-negative.
        Default ``1.0``.
    sigma : float or array_like, optional
        Input noise amplitude :math:`\sigma` (dimensionless). Must be non-negative.
        Zero disables stochastic forcing. Default ``1.0``.
    mu : float or array_like, optional
        Constant mean drive :math:`\mu` (dimensionless). Default ``0.0``.
    g : float or array_like, optional
        Gain parameter :math:`g` of the quartic nonlinearity (dimensionless).
        Larger values steepen the gain curve and shift the half-activation point.
        Must be positive. Default ``1.0``.
    mult_coupling : bool, optional
        Multiplicative coupling flag kept for NEST compatibility. For this model
        variant, multiplicative factors are identically one regardless of this
        setting. Default ``False``.
    linear_summation : bool, optional
        If ``True`` (default), apply gain :math:`\phi` to total summed synaptic input.
        If ``False``, apply :math:`\phi` to each event before weighted summation
        (per-event nonlinearity mode). Default ``True``.
    rectify_rate : float or array_like, optional
        Lower bound for rectified output when ``rectify_output=True``.
        Must be non-negative. Default ``0.0``.
    rectify_output : bool, optional
        If ``True``, clamp updated ``rate`` to :math:`\ge` ``rectify_rate`` at each step.
        Default ``False``.
    rate_initializer : Callable[[shape, batch_size], Array], optional
        Initializer for ``rate`` state. Default ``Constant(0.0)``.
    noise_initializer : Callable[[shape, batch_size], Array], optional
        Initializer for ``noise`` state. Default ``Constant(0.0)``.
    name : str, optional
        Module name for identification. Auto-generated if not provided.

    State Variables
    ---------------
    rate : ShortTermState
        Current neuron rate :math:`X(t)` (dimensionless, shape ``varshape + (batch_size,)``).
    noise : ShortTermState
        Last noise realization :math:`\sigma\,\xi_n` (dimensionless).
    instant_rate : ShortTermState
        Instantaneous outgoing rate (alias of ``rate`` after update).
    delayed_rate : ShortTermState
        Delayed outgoing rate (stored from previous step).
    _step_count : ShortTermState
        Internal step counter (int64 scalar) for delayed event scheduling.

    Parameter Mapping
    -----------------

    ===============================  ===============================  =======================
    brainpy.state parameter          NEST parameter                   Notes
    ===============================  ===============================  =======================
    ``tau``                          ``tau``                          Time constant (ms)
    ``lambda_``                      ``lambda``                       Passive decay rate
    ``sigma``                        ``sigma``                        Input noise amplitude
    ``mu``                           ``mu``                           Constant drive
    ``g``                            ``g``                            Gain parameter
    ``mult_coupling``                ``mult_coupling``                (Always 1.0 for this model)
    ``linear_summation``             ``linear_summation``             Gain application mode
    ``rectify_rate``                 ``rectify_rate``                 Lower rectification bound
    ``rectify_output``               ``rectify_output``               Rectification flag
    ``rate_initializer``             ``rate`` (initialization)        Initial rate value
    ===============================  ===============================  =======================

    Recordables
    -----------
    - ``rate`` : Current neuron rate (main output)
    - ``noise`` : Last noise realization

    Receptor Types
    --------------
    - ``RATE`` : index 0 (single receptor port for all rate inputs)

    Notes
    -----
    **Runtime event API**:

    - ``instant_rate_events``: Applied in the current step with ``delay_steps=0``.
      Raises ``ValueError`` if non-zero delay is specified.
    - ``delayed_rate_events``: Queued for delivery after ``delay_steps`` steps
      (default ``delay_steps=1`` if omitted).

    **Event format flexibility**:

    - Dict: ``{'rate': r, 'weight': w, 'delay_steps': d, 'multiplicity': m}``
      (all fields optional except ``rate``).
    - Tuple: ``(rate, weight)``, ``(rate, weight, delay_steps)``, or
      ``(rate, weight, delay_steps, multiplicity)``.
    - Scalar: Interpreted as ``rate`` with weight=1, delay_steps=0, multiplicity=1.
    - Lists of any of the above are flattened and processed sequentially.

    **Weight polarity**: Positive weights contribute to excitatory branch, negative
    to inhibitory branch. Branch separation only affects ``linear_summation=False`` mode
    with ``mult_coupling=True`` (which has no effect for this model).

    Examples
    --------
    Create a population and drive it with noisy input:

    .. code-block:: python

       >>> from brainpy import state as bst
       >>> import brainunit as u
       >>> import brainstate as bs
       >>> pop = bst.sigmoid_rate_gg_1998_ipn(
       ...     in_size=100, tau=20*u.ms, lambda_=0.5, sigma=0.1, g=2.0
       ... )
       >>> pop.init_all_states()
       >>> with bs.environ.context(dt=0.1*u.ms):
       ...     for _ in range(1000):
       ...         r = pop.update(x=0.5)

    Apply delayed rate events:

    .. code-block:: python

       >>> with bs.environ.context(dt=0.1*u.ms):
       ...     # Event at t+5 steps with weight 0.8
       ...     r = pop.update(delayed_rate_events=[{'rate': 1.0, 'weight': 0.8, 'delay_steps': 5}])

    References
    ----------
    .. [1] Gancarz G, Grossberg S (1998). A neural model of the saccade generator
       in the reticular formation. Neural Networks, 11(7):1159-1174.
       DOI: 10.1016/S0893-6080(98)00096-3.
    .. [2] Hahne J, Dahmen D, Schuecker J, Frommer A, Bolten M, Helias M,
       Diesmann M (2017). Integration of continuous-time dynamics in a spiking
       neural network simulator. Frontiers in Neuroinformatics, 11:34.
       DOI: 10.3389/fninf.2017.00034.

    See Also
    --------
    sigmoid_rate_ipn : Standard sigmoid gain function
    lin_rate_ipn : Linear rate neuron variant
    tanh_rate_ipn : Hyperbolic tangent gain function
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau: ArrayLike = 10.0 * u.ms,
        lambda_: ArrayLike = 1.0,
        sigma: ArrayLike = 1.0,
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
        r"""List of state variables available for recording.

        Returns
        -------
        list of str
            ``['rate', 'noise']`` — current rate and last noise realization.
        """
        return ['rate', 'noise']

    @property
    def receptor_types(self):
        r"""Receptor port name-to-index mapping for synaptic input routing.

        Returns
        -------
        dict[str, int]
            ``{'RATE': 0}`` — single receptor port accepting all rate events.
        """
        return {'RATE': 0}

    def _validate_parameters(self):
        r"""Validate construction-time parameter constraints.

        Checks that time constant, decay rate, noise amplitude, and rectification
        rate satisfy physical and numerical consistency requirements.

        Raises
        ------
        ValueError
            If ``tau <= 0`` (non-positive time constant).
        ValueError
            If ``lambda_ < 0`` (negative passive decay rate).
        ValueError
            If ``sigma < 0`` (negative noise amplitude).
        ValueError
            If ``rectify_rate < 0`` (negative rectification bound).
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
        r"""Initialize neuron state variables and delayed event queues.

        Allocates ``rate``, ``noise``, ``instant_rate``, ``delayed_rate``, internal
        step counter, and per-step delayed event dictionaries for excitatory and
        inhibitory branches.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        Called automatically by ``init_all_states()``. Initializes:

        - ``rate`` : from ``rate_initializer``
        - ``noise`` : from ``noise_initializer``
        - ``instant_rate``, ``delayed_rate`` : copies of initial ``rate``
        - ``_step_count`` : int64 scalar starting at 0
        - ``_delayed_ex_queue``, ``_delayed_in_queue`` : empty dicts
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
        r"""Advance rate neuron dynamics by one time step with stochastic integration.

        Implements the full update cycle: drain delayed events, schedule new delayed
        events, aggregate instantaneous events and external inputs, propagate intrinsic
        dynamics with stochastic exponential Euler, apply quartic gain function, and
        apply optional rectification.

        Parameters
        ----------
        x : float or array_like, optional
            External continuous current input (dimensionless). Broadcasts to ``varshape``.
            Aggregated via ``sum_current_inputs()``. Default ``0.0``.
        instant_rate_events : None or list or dict or tuple, optional
            Instantaneous rate events applied in the current step. Each event must have
            ``delay_steps=0`` or omit the field. Supports nested lists, dicts, tuples,
            or scalars. Default ``None`` (no instantaneous events).
        delayed_rate_events : None or list or dict or tuple, optional
            Delayed rate events scheduled for future delivery. Each event specifies
            ``delay_steps >= 0`` (default 1 if omitted). Format matches
            ``instant_rate_events``. Default ``None`` (no delayed events).
        noise : float or array_like, optional
            External noise realization :math:`\xi_n` to override internal random draw.
            If provided, broadcasts to ``varshape``. If ``None`` (default), draws
            :math:`\xi_n \sim \mathcal{N}(0, 1)` internally.

        Returns
        -------
        rate_new : ndarray
            Updated rate :math:`X_{n+1}` (shape ``varshape`` or ``varshape + (batch,)``).

        Raises
        ------
        ValueError
            If ``instant_rate_events`` contain non-zero ``delay_steps``.
        ValueError
            If ``delayed_rate_events`` contain negative ``delay_steps``.
        ValueError
            If event tuples have unsupported length (must be 2, 3, or 4).

        Notes
        -----
        **Computational sequence**:

        1. Convert time step to milliseconds: :math:`h = dt`.
        2. Broadcast parameters (``tau``, ``sigma``, ``mu``, ``g``, ``lambda_``,
           ``rectify_rate``) to ``state_shape``.
        3. Drain delayed queues for current step index, schedule new delayed events
           from ``delayed_rate_events``.
        4. Accumulate ``instant_rate_events`` and delta inputs, separating by weight
           polarity (excitatory/inhibitory).
        5. Aggregate external input ``x`` via ``sum_current_inputs()``.
        6. Compute propagators :math:`P_1`, :math:`P_2`, and noise factor based on
           :math:`\lambda`.
        7. Draw or use provided noise :math:`\xi_n`.
        8. Propagate :math:`X_{n+1} = P_1 X_n + P_2 (\mu + \mu_{ext}) +
           \text{noise_factor} \cdot \sigma \xi_n`.
        9. Apply gain :math:`\phi` to synaptic inputs (mode depends on
           ``linear_summation`` and ``mult_coupling``).
        10. Add weighted synaptic contributions to :math:`X_{n+1}`.
        11. Apply rectification if ``rectify_output=True``.
        12. Update state variables and increment step counter.

        **Gain application modes**:

        - ``linear_summation=True``, ``mult_coupling=False`` (default):
          :math:`\phi(h_{ex,\mathrm{inst}} + h_{in,\mathrm{inst}} +
          h_{ex,\mathrm{del}} + h_{in,\mathrm{del}})`.
        - ``linear_summation=False``: Gain already applied per event before summation.
        - ``mult_coupling=True``: Separate excitatory/inhibitory branches with
          multiplicative factors (always 1.0 for this model).

        **Delayed event scheduling**: Events with ``delay_steps=d`` are delivered
        at global step ``current_step + d``. Zero-delay events in ``delayed_rate_events``
        are applied immediately (no queueing).

        **Random number generation**: Uses NumPy's default RNG
        (``np.random.normal``). For reproducibility, seed the global RNG before
        simulation.
        """
        h = float(u.get_mantissa(brainstate.environ.get_dt() / u.ms))
        dftype = brainstate.environ.dftype()
        state_shape = self.rate.value.shape

        tau, sigma, mu, g = self._common_parameters_sigmoid_gg_1998(state_shape)
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
