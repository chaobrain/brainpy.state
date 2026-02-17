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

from __future__ import annotations

from typing import Callable

import numpy as np

import brainstate
import braintools
import brainunit as u
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest.lin_rate import _lin_rate_base

__all__ = [
    'tanh_rate_ipn',
    'tanh_rate_opn',
]


class _tanh_rate_base(_lin_rate_base):
    """Base class for tanh-rate neurons with shared input transformation logic.

    Provides hyperbolic-tangent nonlinearity and fixed multiplicative coupling
    factors (always 1) for both ``tanh_rate_ipn`` and ``tanh_rate_opn``.
    """
    __module__ = 'brainpy.state'

    def _input(self, h, g, theta):
        """Apply hyperbolic-tangent nonlinearity to input.

        Parameters
        ----------
        h : ndarray
            Input value(s) to transform (dimensionless).
        g : ndarray
            Gain parameter (dimensionless).
        theta : ndarray
            Threshold/shift parameter (dimensionless).

        Returns
        -------
        out : ndarray
            Transformed input :math:`\tanh(g(h - \theta))`.
        """
        return np.tanh(g * (h - theta))

    @staticmethod
    def _mult_coupling_ex(rate):
        """Multiplicative coupling factor for excitatory inputs (always 1).

        Parameters
        ----------
        rate : ndarray
            Current rate values (unused for tanh_rate).

        Returns
        -------
        out : ndarray
            Array of ones with same shape and dtype as ``rate``.
        """
        return np.ones_like(rate, dtype=np.float64)

    @staticmethod
    def _mult_coupling_in(rate):
        """Multiplicative coupling factor for inhibitory inputs (always 1).

        Parameters
        ----------
        rate : ndarray
            Current rate values (unused for tanh_rate).

        Returns
        -------
        out : ndarray
            Array of ones with same shape and dtype as ``rate``.
        """
        return np.ones_like(rate, dtype=np.float64)

    def _extract_event_fields(self, ev, default_delay_steps: int):
        """Extract event fields from flexible event representation.

        Parse rate events in dict, tuple, or scalar format and return
        normalized components.

        Parameters
        ----------
        ev : dict or tuple or list or scalar
            Rate event specification. Supported formats:

            - dict: ``{'rate': val, 'weight': w, 'delay_steps': d,
              'multiplicity': m}``
            - tuple/list: ``(rate, weight)``, ``(rate, weight, delay_steps)``,
              or ``(rate, weight, delay_steps, multiplicity)``
            - scalar: interpreted as rate with default weight, multiplicity,
              and delay

        default_delay_steps : int
            Default delay in simulation steps when not specified.

        Returns
        -------
        rate : Any
            Rate value (not yet converted to array).
        weight : float
            Synaptic weight.
        multiplicity : float
            Event multiplicity factor.
        delay_steps : int
            Delay in simulation steps.

        Raises
        ------
        ValueError
            If tuple/list has invalid length (not 2, 3, or 4).
        """
        if isinstance(ev, dict):
            rate = ev.get('rate', ev.get('coeff', ev.get('value', 0.0)))
            weight = ev.get('weight', 1.0)
            multiplicity = ev.get('multiplicity', 1.0)
            delay_steps = ev.get('delay_steps', ev.get('delay', default_delay_steps))
        elif isinstance(ev, (tuple, list)):
            if len(ev) == 2:
                rate, weight = ev
                delay_steps = default_delay_steps
                multiplicity = 1.0
            elif len(ev) == 3:
                rate, weight, delay_steps = ev
                multiplicity = 1.0
            elif len(ev) == 4:
                rate, weight, delay_steps, multiplicity = ev
            else:
                raise ValueError('Rate event tuples must have length 2, 3, or 4.')
        else:
            rate = ev
            weight = 1.0
            multiplicity = 1.0
            delay_steps = default_delay_steps

        delay_steps = self._to_int_scalar(delay_steps, name='delay_steps')
        return rate, weight, multiplicity, delay_steps

    def _event_to_ex_in(self, ev, default_delay_steps: int, state_shape, g, theta):
        """Convert event to excitatory and inhibitory contributions.

        Extract event components, broadcast to state shape, apply nonlinearity
        if needed, and split by weight sign.

        Parameters
        ----------
        ev : dict or tuple or list or scalar
            Rate event specification.
        default_delay_steps : int
            Default delay when not specified in event.
        state_shape : tuple
            Target shape for broadcasting.
        g : ndarray
            Gain parameter for tanh nonlinearity.
        theta : ndarray
            Threshold parameter for tanh nonlinearity.

        Returns
        -------
        ex : ndarray
            Excitatory (positive-weight) contribution with shape
            ``state_shape``.
        inh : ndarray
            Inhibitory (negative-weight) contribution with shape
            ``state_shape``.
        delay_steps : int
            Extracted delay in simulation steps.

        Notes
        -----
        When ``linear_summation=False``, tanh is applied per event before
        weighting. When ``True``, raw rates are weighted (tanh applied later
        to sums).
        """
        rate, weight, multiplicity, delay_steps = self._extract_event_fields(ev, default_delay_steps)

        rate_np = self._broadcast_to_state(self._to_numpy(rate), state_shape)
        weight_np = self._broadcast_to_state(self._to_numpy(weight), state_shape)
        multiplicity_np = self._broadcast_to_state(self._to_numpy(multiplicity), state_shape)
        weight_sign = self._broadcast_to_state(
            np.asarray(u.math.asarray(weight), dtype=np.float64) >= 0.0,
            state_shape,
        )

        if self.linear_summation:
            weighted_value = rate_np * weight_np * multiplicity_np
        else:
            weighted_value = self._input(rate_np, g, theta) * weight_np * multiplicity_np

        ex = np.where(weight_sign, weighted_value, 0.0)
        inh = np.where(weight_sign, 0.0, weighted_value)
        return ex, inh, delay_steps

    def _accumulate_instant_events_tanh(self, events, state_shape, g, theta):
        """Accumulate instant rate events (zero-delay).

        Sum excitatory and inhibitory contributions from all instant events.

        Parameters
        ----------
        events : list or None
            List of rate events to process, or None.
        state_shape : tuple
            Shape for output arrays.
        g : ndarray
            Gain parameter for tanh nonlinearity.
        theta : ndarray
            Threshold parameter for tanh nonlinearity.

        Returns
        -------
        ex : ndarray
            Total excitatory input with shape ``state_shape``.
        inh : ndarray
            Total inhibitory input with shape ``state_shape``.

        Raises
        ------
        ValueError
            If any event specifies non-zero ``delay_steps``.
        """
        ex = np.zeros(state_shape, dtype=np.float64)
        inh = np.zeros(state_shape, dtype=np.float64)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._event_to_ex_in(
                ev,
                default_delay_steps=0,
                state_shape=state_shape,
                g=g,
                theta=theta,
            )
            if delay_steps != 0:
                raise ValueError('instant_rate_events must not specify non-zero delay_steps.')
            ex += ex_i
            inh += inh_i
        return ex, inh

    def _schedule_delayed_events_tanh(self, events, step_idx: int, state_shape, g, theta):
        """Schedule delayed rate events and return zero-delay contributions.

        Queue events with positive delay into internal buffers and accumulate
        zero-delay events for immediate application.

        Parameters
        ----------
        events : list or None
            List of rate events to schedule, or None.
        step_idx : int
            Current simulation step index.
        state_shape : tuple
            Shape for output arrays.
        g : ndarray
            Gain parameter for tanh nonlinearity.
        theta : ndarray
            Threshold parameter for tanh nonlinearity.

        Returns
        -------
        ex_now : ndarray
            Excitatory contribution from zero-delay events with shape
            ``state_shape``.
        inh_now : ndarray
            Inhibitory contribution from zero-delay events with shape
            ``state_shape``.

        Raises
        ------
        ValueError
            If any event specifies negative ``delay_steps``.

        Notes
        -----
        Events with ``delay_steps > 0`` are queued in
        ``self._delayed_ex_queue`` and ``self._delayed_in_queue`` for delivery
        at ``step_idx + delay_steps``.
        """
        ex_now = np.zeros(state_shape, dtype=np.float64)
        inh_now = np.zeros(state_shape, dtype=np.float64)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._event_to_ex_in(
                ev,
                default_delay_steps=1,
                state_shape=state_shape,
                g=g,
                theta=theta,
            )
            if delay_steps < 0:
                raise ValueError('delay_steps for delayed_rate_events must be >= 0.')
            if delay_steps == 0:
                ex_now += ex_i
                inh_now += inh_i
            else:
                target_step = step_idx + delay_steps
                self._queue_add(self._delayed_ex_queue, target_step, ex_i)
                self._queue_add(self._delayed_in_queue, target_step, inh_i)
        return ex_now, inh_now

    def _common_inputs_tanh(self, x, instant_rate_events, delayed_rate_events, g, theta):
        """Collect all input contributions for current simulation step.

        Process delayed queues, schedule new delayed events, accumulate instant
        events, and gather external current/delta inputs.

        Parameters
        ----------
        x : Any
            External current input.
        instant_rate_events : list or None
            Rate events to apply immediately.
        delayed_rate_events : list or None
            Rate events to schedule with delay.
        g : ndarray
            Gain parameter for tanh nonlinearity.
        theta : ndarray
            Threshold parameter for tanh nonlinearity.

        Returns
        -------
        state_shape : tuple
            Shape of state variables.
        step_idx : int
            Current simulation step index.
        delayed_ex : ndarray
            Total delayed excitatory input.
        delayed_in : ndarray
            Total delayed inhibitory input.
        instant_ex : ndarray
            Total instant excitatory input (includes delta_inputs).
        instant_in : ndarray
            Total instant inhibitory input (includes delta_inputs).
        mu_ext : ndarray
            External current input (broadcast).

        Notes
        -----
        Delta inputs (from projections) are split by sign and added to instant
        excitatory/inhibitory branches.
        """
        state_shape = self.rate.value.shape
        step_idx = int(np.asarray(self._step_count.value, dtype=np.int64).reshape(-1)[0])

        delayed_ex, delayed_in = self._drain_delayed_queue(step_idx, state_shape)
        delayed_ex_now, delayed_in_now = self._schedule_delayed_events_tanh(
            delayed_rate_events,
            step_idx=step_idx,
            state_shape=state_shape,
            g=g,
            theta=theta,
        )
        delayed_ex = delayed_ex + delayed_ex_now
        delayed_in = delayed_in + delayed_in_now

        instant_ex, instant_in = self._accumulate_instant_events_tanh(
            instant_rate_events,
            state_shape=state_shape,
            g=g,
            theta=theta,
        )

        delta_input = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0.0)), state_shape)
        instant_ex += np.where(delta_input > 0.0, delta_input, 0.0)
        instant_in += np.where(delta_input < 0.0, delta_input, 0.0)

        mu_ext = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.rate.value)), state_shape)

        return state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext

    def _common_parameters_tanh(self, state_shape):
        """Broadcast model parameters to state shape.

        Convert parameters to plain NumPy arrays and broadcast to match state
        dimensions.

        Parameters
        ----------
        state_shape : tuple
            Target shape for broadcasting.

        Returns
        -------
        tau : ndarray
            Time constant in ms, broadcast to ``state_shape``.
        sigma : ndarray
            Noise scale, broadcast to ``state_shape``.
        mu : ndarray
            Mean drive, broadcast to ``state_shape``.
        g : ndarray
            Gain parameter, broadcast to ``state_shape``.
        theta : ndarray
            Threshold parameter, broadcast to ``state_shape``.
        """
        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        sigma = self._broadcast_to_state(self._to_numpy(self.sigma), state_shape)
        mu = self._broadcast_to_state(self._to_numpy(self.mu), state_shape)
        g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.theta), state_shape)
        return tau, sigma, mu, g, theta


class tanh_rate_ipn(_tanh_rate_base):
    r"""NEST-compatible ``tanh_rate_ipn`` nonlinear rate neuron with input noise.

    Stochastic rate model with hyperbolic-tangent nonlinearity applied to
    network inputs, matching NEST's ``rate_neuron_ipn`` template instantiated
    with ``tanh_rate`` gain function.

    **1. Model equations**

    The state :math:`X(t)` evolves according to the Langevin equation

    .. math::

       \tau\,dX(t)=
       \left[-\lambda X(t)+\mu+\phi(\cdot)\right]dt
       +\left[\sqrt{\tau}\,\sigma\right]dW(t),

    where the input nonlinearity is

    .. math::

       \phi(h)=\tanh(g(h-\theta)).

    Here :math:`W(t)` is a standard Brownian motion, :math:`\lambda` is the
    passive decay rate, :math:`\mu` is a constant drive, :math:`g` is the
    gain, and :math:`\theta` is the horizontal shift of the tanh function.

    **2. Numerical integration and noise implementation**

    For :math:`\lambda > 0`, integration uses the stochastic exponential Euler
    (exact for the Ornstein-Uhlenbeck process):

    .. math::

       P_1 &= \exp(-\lambda h / \tau), \\
       P_2 &= \frac{1 - P_1}{\lambda}, \\
       X_{n+1} &= P_1 X_n + P_2 \left[\mu + \phi(\cdot)\right]
                  + \sqrt{\frac{1-P_1^2}{2\lambda}} \sigma \xi_n,

    with :math:`\xi_n \sim \mathcal{N}(0,1)`. For :math:`\lambda = 0`, it
    reduces to Euler-Maruyama:

    .. math::

       X_{n+1} = X_n + \frac{h}{\tau} \left[\mu + \phi(\cdot)\right]
                 + \sqrt{\frac{h}{\tau}} \sigma \xi_n.

    **3. Update ordering (matching NEST ``rate_neuron_ipn`` with tanh)**

    Each simulation step proceeds as follows:

    1. Store outgoing delayed value as current ``rate``.
    2. Draw ``noise = sigma * xi`` from the standard normal distribution.
    3. Propagate intrinsic dynamics with stochastic exponential Euler
       (Euler-Maruyama for ``lambda=0``).
    4. Read delayed and instantaneous input buffers.
    5. Apply input contributions:

       - ``linear_summation=True``: apply tanh to summed branch inputs.
       - ``linear_summation=False``: apply tanh per event before summation.

    6. Apply rectification when ``rectify_output=True`` (clamp to
       ``>= rectify_rate``).
    7. Store outgoing instantaneous value as updated ``rate``.

    **4. Timing semantics, assumptions, and constraints**

    - Noise term is white (independent at each step, variance scales with
      :math:`dt`).
    - For :math:`\lambda > 0`, integration exactly preserves stationary
      variance of the OU process.
    - For :math:`\lambda = 0`, the process is non-stationary (variance grows
      linearly with time).
    - Multiplicative coupling factors are fixed to 1 for tanh_rate models
      (``mult_coupling`` has no effect; kept for NEST API compatibility).

    **5. Computational implications**

    Per :meth:`update` call:

    - Random number generation: :math:`O(\prod \mathrm{varshape})`.
    - Exponential operations for :math:`P_1, P_2` when :math:`\lambda > 0`.
    - Event processing is linear in number of events per step.
    - Broadcasting parameters and inputs over ``self.varshape``.

    Parameters
    ----------
    in_size : Size
        Population shape specification consumed by
        :class:`brainstate.nn.Dynamics`. Determines output rate array shape.
    tau : ArrayLike, optional
        Time constant :math:`\tau` of rate dynamics. Must be positive.
        Accepts scalar or array broadcast to ``self.varshape``. Unitful values
        are converted to ms; unitless are interpreted as ms. Default
        ``10.0 * u.ms``.
    lambda_ : ArrayLike, optional
        Passive decay rate :math:`\lambda` (dimensionless). Must be
        non-negative. For :math:`\lambda > 0`, uses stochastic exponential
        Euler; for :math:`\lambda = 0`, uses Euler-Maruyama. Default ``1.0``.
    sigma : ArrayLike, optional
        Input noise scale (dimensionless). Must be non-negative. Scales the
        Wiener increment. Broadcast to ``self.varshape``. Default ``1.0``.
    mu : ArrayLike, optional
        Mean drive :math:`\mu` (dimensionless). Constant additive input.
        Broadcast to ``self.varshape``. Default ``0.0``.
    g : ArrayLike, optional
        Gain of tanh nonlinearity (dimensionless). Controls steepness of tanh.
        Broadcast to ``self.varshape``. Default ``1.0``.
    theta : ArrayLike, optional
        Threshold (horizontal shift) of tanh nonlinearity (dimensionless).
        Shifts the input :math:`h` before applying tanh. Broadcast to
        ``self.varshape``. Default ``0.0``.
    mult_coupling : bool, optional
        Kept for NEST compatibility. For ``tanh_rate`` models, multiplicative
        coupling factors are identically 1, so this switch has no effect.
        Default ``False``.
    linear_summation : bool, optional
        Controls nonlinearity application order. If ``True``, sum inputs then
        apply tanh. If ``False``, apply tanh per event then sum weighted
        results. Default ``True``.
    rectify_rate : ArrayLike, optional
        Lower bound for output rate when ``rectify_output=True``
        (dimensionless). Must be non-negative. Broadcast to
        ``self.varshape``. Default ``0.0``.
    rectify_output : bool, optional
        If ``True``, clamp updated rate to ``>= rectify_rate`` after all
        dynamics. Default ``False``.
    rate_initializer : Callable, optional
        Initializer for state variable ``rate``. Called as
        ``rate_initializer(self.varshape, batch_size)`` in
        :meth:`init_state`. Default ``braintools.init.Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for state variable ``noise`` (recording). Default
        ``braintools.init.Constant(0.0)``.
    name : str or None, optional
        Module name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 22 18 22 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``tau``
         - ``10.0 * u.ms``
         - :math:`\tau`
         - Time constant of rate dynamics (ms).
       * - ``lambda_``
         - ``1.0``
         - :math:`\lambda`
         - Passive decay rate (dimensionless, >= 0).
       * - ``sigma``
         - ``1.0``
         - :math:`\sigma`
         - Input noise scale (dimensionless, >= 0).
       * - ``mu``
         - ``0.0``
         - :math:`\mu`
         - Constant mean drive (dimensionless).
       * - ``g``
         - ``1.0``
         - :math:`g`
         - Gain of tanh nonlinearity (dimensionless).
       * - ``theta``
         - ``0.0``
         - :math:`\theta`
         - Horizontal shift of tanh nonlinearity (dimensionless).
       * - ``rectify_rate``
         - ``0.0``
         - :math:`r_{\min}`
         - Lower clamp bound when ``rectify_output=True``.

    Returns
    -------
    out : Any
        Dynamics node. Calling :meth:`update` returns updated rate array with
        shape ``self.varshape`` and dimensionless dtype (float64).

    Raises
    ------
    ValueError
        If ``tau <= 0``, ``lambda_ < 0``, ``sigma < 0``, or
        ``rectify_rate < 0``.
    ValueError
        If ``instant_rate_events`` specify non-zero ``delay_steps``, or if
        ``delayed_rate_events`` specify negative ``delay_steps``.

    See Also
    --------
    tanh_rate_opn : Output-noise variant of tanh_rate.
    sigmoid_rate_ipn : Sigmoid nonlinearity with input noise.
    lin_rate_ipn : Linear (identity) nonlinearity with input noise.
    gauss_rate_ipn : Gaussian nonlinearity with input noise.

    Notes
    -----
    **Runtime events:**

    - ``instant_rate_events`` : applied in the current step (``delay_steps=0``).
    - ``delayed_rate_events`` : scheduled with integer ``delay_steps >= 0``.

    **Event format** supports dict or tuple:

    - Dict: ``{'rate': value, 'weight': w, 'delay_steps': d,
      'multiplicity': m}``
    - Tuple: ``(rate, weight)``, ``(rate, weight, delay_steps)``, or
      ``(rate, weight, delay_steps, multiplicity)``

    References
    ----------
    .. [1] NEST Simulator documentation for ``rate_neuron_ipn``:
           https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_ipn.html
    .. [2] Hahne, J., Dahmen, D., Schuecker, J., Frommer, A., Bolten, M.,
           Helias, M., & Diesmann, M. (2017). Integration of continuous-time
           dynamics in a spiking neural network simulator. *Frontiers in
           Neuroinformatics*, 11, 34.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> import jax.numpy as jnp
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     net = brainpy.state.tanh_rate_ipn(
       ...         in_size=10,
       ...         tau=10.0 * u.ms,
       ...         lambda_=1.0,
       ...         sigma=0.5,
       ...         mu=0.0,
       ...         g=1.0,
       ...         theta=0.0,
       ...         rectify_output=True,
       ...         rectify_rate=0.0,
       ...     )
       ...     net.init_all_states()
       ...     rate = net.update(x=0.1)
       ...     _ = rate.shape  # (10,)
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
        theta: ArrayLike = 0.0,
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
        self.theta = braintools.init.param(theta, self.varshape)
        self.lambda_ = braintools.init.param(lambda_, self.varshape)
        self.rectify_rate = braintools.init.param(rectify_rate, self.varshape)
        self.rectify_output = bool(rectify_output)
        self._validate_parameters()

    @property
    def recordables(self):
        """List of recordable state variables.

        Returns
        -------
        list of str
            ``['rate', 'noise']``.
        """
        return ['rate', 'noise']

    @property
    def receptor_types(self):
        """Mapping of receptor type names to indices.

        Returns
        -------
        dict
            ``{'RATE': 0}`` for rate-based connections.
        """
        return {'RATE': 0}

    def _validate_parameters(self):
        """Validate parameter constraints at initialization.

        Raises
        ------
        ValueError
            If ``tau <= 0``, ``lambda_ < 0``, ``sigma < 0``, or
            ``rectify_rate < 0``.
        """
        if np.any(self._to_numpy_ms(self.tau) <= 0.0):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self._to_numpy(self.lambda_) < 0.0):
            raise ValueError('Passive decay rate lambda must be >= 0.')
        if np.any(self._to_numpy(self.sigma) < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')
        if np.any(self._to_numpy(self.rectify_rate) < 0.0):
            raise ValueError('Rectifying rate must be >= 0.')

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize state variables and internal buffers.

        Create ``rate``, ``noise``, ``instant_rate``, ``delayed_rate``, and
        step counter states. Initialize delay queues.

        Parameters
        ----------
        batch_size : int or None, optional
            Batch dimension for state variables. If None, no batch dimension
            is added.
        **kwargs
            Additional keyword arguments (unused).

        Notes
        -----
        State variables are initialized from ``rate_initializer`` and
        ``noise_initializer``. Both ``instant_rate`` and ``delayed_rate`` are
        initialized as copies of ``rate``.
        """
        rate = braintools.init.param(self.rate_initializer, self.varshape, batch_size)
        noise = braintools.init.param(self.noise_initializer, self.varshape, batch_size)
        rate_np = self._to_numpy(rate)
        noise_np = self._to_numpy(noise)

        self.rate = brainstate.ShortTermState(rate_np)
        self.noise = brainstate.ShortTermState(noise_np)
        self.instant_rate = brainstate.ShortTermState(np.array(rate_np, dtype=np.float64, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(rate_np, dtype=np.float64, copy=True))
        self._step_count = brainstate.ShortTermState(np.asarray(0, dtype=np.int64))

        self._delayed_ex_queue = {}
        self._delayed_in_queue = {}

    def update(self, x=0.0, instant_rate_events=None, delayed_rate_events=None, noise=None):
        r"""Advance rate dynamics by one simulation step.

        Execute stochastic exponential Euler integration with input noise,
        process delayed and instant events, apply tanh nonlinearity, and
        optionally rectify output.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input (dimensionless). Broadcast to state shape
            and summed with ``mu``. Default ``0.0``.
        instant_rate_events : list or None, optional
            Rate events to apply in the current step (``delay_steps=0``). Each
            event can be dict, tuple, or scalar. Default None.
        delayed_rate_events : list or None, optional
            Rate events to schedule with integer delays (``delay_steps >= 0``).
            Default None.
        noise : ArrayLike or None, optional
            Custom noise samples :math:`\xi` drawn from :math:`\mathcal{N}(0,1)`.
            If None, random samples are drawn internally. Useful for
            reproducibility or testing. Default None.

        Returns
        -------
        rate : ndarray
            Updated rate values with shape ``self.varshape`` (float64).

        Notes
        -----
        **Update sequence:**

        1. Extract current step index and state shape.
        2. Drain queued delayed inputs scheduled for this step.
        3. Schedule new delayed events and accumulate zero-delay events.
        4. Accumulate instant events and split delta inputs by sign.
        5. Draw noise (or use provided ``noise``).
        6. Compute stochastic exponential Euler step:

           - For :math:`\lambda > 0`: use OU-exact propagators.
           - For :math:`\lambda = 0`: use Euler-Maruyama.

        7. Apply tanh nonlinearity to summed inputs (if
           ``linear_summation=True``) or use pre-transformed per-event values
           (if ``False``).
        8. Rectify output if ``rectify_output=True``.
        9. Update state variables and increment step counter.
        """
        h = float(u.math.asarray(brainstate.environ.get_dt() / u.ms))
        state_shape = self.rate.value.shape

        tau, sigma, mu, g, theta = self._common_parameters_tanh(state_shape)
        lambda_ = self._broadcast_to_state(self._to_numpy(self.lambda_), state_shape)
        rectify_rate = self._broadcast_to_state(self._to_numpy(self.rectify_rate), state_shape)

        state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext = self._common_inputs_tanh(
            x=x,
            instant_rate_events=instant_rate_events,
            delayed_rate_events=delayed_rate_events,
            g=g,
            theta=theta,
        )

        rate_prev = self._broadcast_to_state(self._to_numpy(self.rate.value), state_shape)

        if noise is None:
            xi = np.random.normal(size=state_shape)
        else:
            xi = self._broadcast_to_state(self._to_numpy(noise), state_shape)
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

        H_ex = np.ones_like(rate_prev)
        H_in = np.ones_like(rate_prev)
        if self.mult_coupling:
            H_ex = self._mult_coupling_ex(rate_prev)
            H_in = self._mult_coupling_in(rate_prev)

        if self.linear_summation:
            if self.mult_coupling:
                rate_new += P2 * H_ex * self._input(delayed_ex + instant_ex, g, theta)
                rate_new += P2 * H_in * self._input(delayed_in + instant_in, g, theta)
            else:
                rate_new += P2 * self._input(delayed_ex + instant_ex + delayed_in + instant_in, g, theta)
        else:
            # Nonlinear transform has already been applied per event in buffer handling.
            rate_new += P2 * H_ex * (delayed_ex + instant_ex)
            rate_new += P2 * H_in * (delayed_in + instant_in)

        if self.rectify_output:
            rate_new = np.where(rate_new < rectify_rate, rectify_rate, rate_new)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.delayed_rate.value = rate_prev
        self.instant_rate.value = rate_new
        self._step_count.value = np.asarray(step_idx + 1, dtype=np.int64)
        return rate_new


class tanh_rate_opn(_tanh_rate_base):
    r"""NEST-compatible ``tanh_rate_opn`` nonlinear rate neuron with output noise.

    Deterministic rate model with output-coupled additive noise and
    hyperbolic-tangent nonlinearity applied to network inputs, matching NEST's
    ``rate_neuron_opn`` template instantiated with ``tanh_rate`` gain function.

    **1. Model equations**

    The internal state :math:`X(t)` evolves deterministically

    .. math::

       \tau\frac{dX(t)}{dt}=-X(t)+\mu+\phi(\cdot),

    where the input nonlinearity is

    .. math::

       \phi(h)=\tanh(g(h-\theta)).

    The observed rate includes white noise added at the output:

    .. math::

       X_\mathrm{noisy}(t)=X(t)+\sqrt{\frac{\tau}{h}}\sigma\xi(t),

    with :math:`\xi(t) \sim \mathcal{N}(0,1)` and :math:`h=dt` the simulation
    step size. The noise is scaled by :math:`\sqrt{\tau/h}` so that its
    variance is independent of the step size, matching NEST's implementation.

    **2. Numerical integration**

    Deterministic exponential Euler integration:

    .. math::

       P_1 &= \exp(-h / \tau), \\
       P_2 &= 1 - P_1, \\
       X_{n+1} &= P_1 X_n + P_2 \left[\mu + \phi(\cdot)\right].

    The noisy rate for outgoing communication is computed as

    .. math::

       X_{\mathrm{noisy},n} = X_n + \sqrt{\frac{\tau}{h}} \sigma \xi_n.

    **3. Update ordering (matching NEST ``rate_neuron_opn`` with tanh)**

    Each simulation step proceeds as follows:

    1. Draw ``noise = sigma * xi`` from the standard normal distribution.
    2. Build ``noisy_rate`` from the current ``rate`` by adding scaled noise.
    3. Store ``noisy_rate`` as delayed outgoing value (for delayed
       connections).
    4. Propagate deterministic intrinsic dynamics with exponential Euler.
    5. Add event-driven input contributions:

       - ``linear_summation=True``: apply tanh to summed branch inputs.
       - ``linear_summation=False``: apply tanh per event before summation.

    6. Store ``noisy_rate`` as instantaneous outgoing value (for instant
       connections).

    **4. Timing semantics, assumptions, and constraints**

    - Noise is added to the output only (internal state :math:`X` remains
      deterministic).
    - Noise variance is independent of step size :math:`h` due to
      :math:`\sqrt{\tau/h}` scaling.
    - Multiplicative coupling factors are fixed to 1 for tanh_rate models
      (``mult_coupling`` has no effect; kept for NEST API compatibility).
    - Unlike input-noise models, there is no rectification option for
      output-noise models.

    **5. Computational implications**

    Per :meth:`update` call:

    - Random number generation: :math:`O(\prod \mathrm{varshape})`.
    - Exponential operations for :math:`P_1, P_2` (single exp call).
    - Event processing is linear in number of events per step.
    - Broadcasting parameters and inputs over ``self.varshape``.

    Parameters
    ----------
    in_size : Size
        Population shape specification consumed by
        :class:`brainstate.nn.Dynamics`. Determines output rate array shape.
    tau : ArrayLike, optional
        Time constant :math:`\tau` of rate dynamics. Must be positive.
        Accepts scalar or array broadcast to ``self.varshape``. Unitful values
        are converted to ms; unitless are interpreted as ms. Default
        ``10.0 * u.ms``.
    sigma : ArrayLike, optional
        Output noise scale (dimensionless). Must be non-negative. Scales the
        Gaussian white noise added to the output. Broadcast to
        ``self.varshape``. Default ``1.0``.
    mu : ArrayLike, optional
        Mean drive :math:`\mu` (dimensionless). Constant additive input.
        Broadcast to ``self.varshape``. Default ``0.0``.
    g : ArrayLike, optional
        Gain of tanh nonlinearity (dimensionless). Controls steepness of tanh.
        Broadcast to ``self.varshape``. Default ``1.0``.
    theta : ArrayLike, optional
        Threshold (horizontal shift) of tanh nonlinearity (dimensionless).
        Shifts the input :math:`h` before applying tanh. Broadcast to
        ``self.varshape``. Default ``0.0``.
    mult_coupling : bool, optional
        Kept for NEST compatibility. For ``tanh_rate`` models, multiplicative
        coupling factors are identically 1, so this switch has no effect.
        Default ``False``.
    linear_summation : bool, optional
        Controls nonlinearity application order. If ``True``, sum inputs then
        apply tanh. If ``False``, apply tanh per event then sum weighted
        results. Default ``True``.
    rate_initializer : Callable, optional
        Initializer for state variable ``rate``. Called as
        ``rate_initializer(self.varshape, batch_size)`` in
        :meth:`init_state`. Default ``braintools.init.Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for state variable ``noise`` (recording). Default
        ``braintools.init.Constant(0.0)``.
    noisy_rate_initializer : Callable, optional
        Initializer for state variable ``noisy_rate`` (output with noise).
        Default ``braintools.init.Constant(0.0)``.
    name : str or None, optional
        Module name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 22 18 22 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``tau``
         - ``10.0 * u.ms``
         - :math:`\tau`
         - Time constant of rate dynamics (ms).
       * - ``sigma``
         - ``1.0``
         - :math:`\sigma`
         - Output noise scale (dimensionless, >= 0).
       * - ``mu``
         - ``0.0``
         - :math:`\mu`
         - Constant mean drive (dimensionless).
       * - ``g``
         - ``1.0``
         - :math:`g`
         - Gain of tanh nonlinearity (dimensionless).
       * - ``theta``
         - ``0.0``
         - :math:`\theta`
         - Horizontal shift of tanh nonlinearity (dimensionless).

    Returns
    -------
    out : Any
        Dynamics node. Calling :meth:`update` returns updated rate array with
        shape ``self.varshape`` and dimensionless dtype (float64).

    Raises
    ------
    ValueError
        If ``tau <= 0`` or ``sigma < 0``.
    ValueError
        If ``instant_rate_events`` specify non-zero ``delay_steps``, or if
        ``delayed_rate_events`` specify negative ``delay_steps``.

    See Also
    --------
    tanh_rate_ipn : Input-noise variant of tanh_rate.
    sigmoid_rate_opn : Sigmoid nonlinearity with output noise.
    lin_rate_opn : Linear (identity) nonlinearity with output noise.
    gauss_rate_opn : Gaussian nonlinearity with output noise.

    Notes
    -----
    **Runtime events:**

    - ``instant_rate_events`` : applied in the current step (``delay_steps=0``).
    - ``delayed_rate_events`` : scheduled with integer ``delay_steps >= 0``.

    **Event format** supports dict or tuple (identical to
    :class:`tanh_rate_ipn`):

    - Dict: ``{'rate': value, 'weight': w, 'delay_steps': d,
      'multiplicity': m}``
    - Tuple: ``(rate, weight)``, ``(rate, weight, delay_steps)``, or
      ``(rate, weight, delay_steps, multiplicity)``

    References
    ----------
    .. [1] NEST Simulator documentation for ``rate_neuron_opn``:
           https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_opn.html
    .. [2] Hahne, J., Dahmen, D., Schuecker, J., Frommer, A., Bolten, M.,
           Helias, M., & Diesmann, M. (2017). Integration of continuous-time
           dynamics in a spiking neural network simulator. *Frontiers in
           Neuroinformatics*, 11, 34.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> import jax.numpy as jnp
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     net = brainpy.state.tanh_rate_opn(
       ...         in_size=10,
       ...         tau=10.0 * u.ms,
       ...         sigma=0.5,
       ...         mu=0.0,
       ...         g=1.0,
       ...         theta=0.0,
       ...     )
       ...     net.init_all_states()
       ...     rate = net.update(x=0.1)
       ...     _ = rate.shape  # (10,)
       ...     _ = net.noisy_rate.value.shape  # (10,)
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau: ArrayLike = 10.0 * u.ms,
        sigma: ArrayLike = 1.0,
        mu: ArrayLike = 0.0,
        g: ArrayLike = 1.0,
        theta: ArrayLike = 0.0,
        mult_coupling: bool = False,
        linear_summation: bool = True,
        rate_initializer: Callable = braintools.init.Constant(0.0),
        noise_initializer: Callable = braintools.init.Constant(0.0),
        noisy_rate_initializer: Callable = braintools.init.Constant(0.0),
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
        self.theta = braintools.init.param(theta, self.varshape)
        self.noisy_rate_initializer = noisy_rate_initializer
        self._validate_parameters()

    @property
    def recordables(self):
        """List of recordable state variables.

        Returns
        -------
        list of str
            ``['rate', 'noise', 'noisy_rate']``.
        """
        return ['rate', 'noise', 'noisy_rate']

    @property
    def receptor_types(self):
        """Mapping of receptor type names to indices.

        Returns
        -------
        dict
            ``{'RATE': 0}`` for rate-based connections.
        """
        return {'RATE': 0}

    def _validate_parameters(self):
        """Validate parameter constraints at initialization.

        Raises
        ------
        ValueError
            If ``tau <= 0`` or ``sigma < 0``.
        """
        if np.any(self._to_numpy_ms(self.tau) <= 0.0):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self._to_numpy(self.sigma) < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize state variables and internal buffers.

        Create ``rate``, ``noise``, ``noisy_rate``, ``instant_rate``,
        ``delayed_rate``, and step counter states. Initialize delay queues.

        Parameters
        ----------
        batch_size : int or None, optional
            Batch dimension for state variables. If None, no batch dimension
            is added.
        **kwargs
            Additional keyword arguments (unused).

        Notes
        -----
        State variables are initialized from ``rate_initializer``,
        ``noise_initializer``, and ``noisy_rate_initializer``. Both
        ``instant_rate`` and ``delayed_rate`` are initialized as copies of
        ``noisy_rate``.
        """
        rate = braintools.init.param(self.rate_initializer, self.varshape, batch_size)
        noise = braintools.init.param(self.noise_initializer, self.varshape, batch_size)
        noisy_rate = braintools.init.param(self.noisy_rate_initializer, self.varshape, batch_size)
        rate_np = self._to_numpy(rate)
        noise_np = self._to_numpy(noise)
        noisy_rate_np = self._to_numpy(noisy_rate)

        self.rate = brainstate.ShortTermState(rate_np)
        self.noise = brainstate.ShortTermState(noise_np)
        self.noisy_rate = brainstate.ShortTermState(noisy_rate_np)
        self.instant_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=np.float64, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=np.float64, copy=True))
        self._step_count = brainstate.ShortTermState(np.asarray(0, dtype=np.int64))

        self._delayed_ex_queue = {}
        self._delayed_in_queue = {}

    def update(self, x=0.0, instant_rate_events=None, delayed_rate_events=None, noise=None):
        r"""Advance rate dynamics by one simulation step.

        Execute deterministic exponential Euler integration, add output noise,
        process delayed and instant events, and apply tanh nonlinearity.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input (dimensionless). Broadcast to state shape
            and summed with ``mu``. Default ``0.0``.
        instant_rate_events : list or None, optional
            Rate events to apply in the current step (``delay_steps=0``). Each
            event can be dict, tuple, or scalar. Default None.
        delayed_rate_events : list or None, optional
            Rate events to schedule with integer delays (``delay_steps >= 0``).
            Default None.
        noise : ArrayLike or None, optional
            Custom noise samples :math:`\xi` drawn from :math:`\mathcal{N}(0,1)`.
            If None, random samples are drawn internally. Useful for
            reproducibility or testing. Default None.

        Returns
        -------
        rate : ndarray
            Updated deterministic rate values with shape ``self.varshape``
            (float64). Note: the communicated output is ``noisy_rate``, not
            ``rate``.

        Notes
        -----
        **Update sequence:**

        1. Draw noise (or use provided ``noise``).
        2. Compute ``noisy_rate`` from current ``rate`` by adding scaled
           noise: :math:`X_{\mathrm{noisy}} = X + \sqrt{\tau/h} \sigma \xi`.
        3. Store ``noisy_rate`` as delayed outgoing value.
        4. Drain queued delayed inputs scheduled for this step.
        5. Schedule new delayed events and accumulate zero-delay events.
        6. Accumulate instant events and split delta inputs by sign.
        7. Compute deterministic exponential Euler step:
           :math:`X_{n+1} = P_1 X_n + P_2 [\mu + \phi(\cdot)]`.
        8. Apply tanh nonlinearity to summed inputs (if
           ``linear_summation=True``) or use pre-transformed per-event values
           (if ``False``).
        9. Store ``noisy_rate`` as instantaneous outgoing value.
        10. Update state variables and increment step counter.
        """
        h = float(u.math.asarray(brainstate.environ.get_dt() / u.ms))
        state_shape = self.rate.value.shape

        tau, sigma, mu, g, theta = self._common_parameters_tanh(state_shape)
        state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext = self._common_inputs_tanh(
            x=x,
            instant_rate_events=instant_rate_events,
            delayed_rate_events=delayed_rate_events,
            g=g,
            theta=theta,
        )

        rate_prev = self._broadcast_to_state(self._to_numpy(self.rate.value), state_shape)

        if noise is None:
            xi = np.random.normal(size=state_shape)
        else:
            xi = self._broadcast_to_state(self._to_numpy(noise), state_shape)
        noise_now = sigma * xi

        P1 = np.exp(-h / tau)
        P2 = -np.expm1(-h / tau)
        output_noise_factor = np.sqrt(tau / h)
        noisy_rate = rate_prev + output_noise_factor * noise_now

        mu_total = mu + mu_ext
        rate_new = P1 * rate_prev + P2 * mu_total

        H_ex = np.ones_like(rate_prev)
        H_in = np.ones_like(rate_prev)
        if self.mult_coupling:
            H_ex = self._mult_coupling_ex(noisy_rate)
            H_in = self._mult_coupling_in(noisy_rate)

        if self.linear_summation:
            if self.mult_coupling:
                rate_new += P2 * H_ex * self._input(delayed_ex + instant_ex, g, theta)
                rate_new += P2 * H_in * self._input(delayed_in + instant_in, g, theta)
            else:
                rate_new += P2 * self._input(delayed_ex + instant_ex + delayed_in + instant_in, g, theta)
        else:
            # Nonlinear transform has already been applied per event in buffer handling.
            rate_new += P2 * H_ex * (delayed_ex + instant_ex)
            rate_new += P2 * H_in * (delayed_in + instant_in)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.noisy_rate.value = noisy_rate
        self.delayed_rate.value = noisy_rate
        self.instant_rate.value = noisy_rate
        self._step_count.value = np.asarray(step_idx + 1, dtype=np.int64)
        return rate_new
