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

import numpy as np

import brainstate
import braintools
import brainunit as u
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest.lin_rate import _lin_rate_base

__all__ = [
    'threshold_lin_rate_ipn',
    'threshold_lin_rate_opn',
]


class _threshold_lin_rate_base(_lin_rate_base):
    __module__ = 'brainpy.state'

    def _input(self, h, g, theta, alpha):
        return np.minimum(np.maximum(g * (h - theta), 0.0), alpha)

    @staticmethod
    def _mult_coupling_ex(rate):
        return np.ones_like(rate, dtype=np.float64)

    @staticmethod
    def _mult_coupling_in(rate):
        return np.ones_like(rate, dtype=np.float64)

    def _extract_event_fields(self, ev, default_delay_steps: int):
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

    def _event_to_ex_in(self, ev, default_delay_steps: int, state_shape, g, theta, alpha):
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
            weighted_value = self._input(rate_np, g, theta, alpha) * weight_np * multiplicity_np

        ex = np.where(weight_sign, weighted_value, 0.0)
        inh = np.where(weight_sign, 0.0, weighted_value)
        return ex, inh, delay_steps

    def _accumulate_instant_events_threshold(self, events, state_shape, g, theta, alpha):
        ex = np.zeros(state_shape, dtype=np.float64)
        inh = np.zeros(state_shape, dtype=np.float64)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._event_to_ex_in(
                ev,
                default_delay_steps=0,
                state_shape=state_shape,
                g=g,
                theta=theta,
                alpha=alpha,
            )
            if delay_steps != 0:
                raise ValueError('instant_rate_events must not specify non-zero delay_steps.')
            ex += ex_i
            inh += inh_i
        return ex, inh

    def _schedule_delayed_events_threshold(self, events, step_idx: int, state_shape, g, theta, alpha):
        ex_now = np.zeros(state_shape, dtype=np.float64)
        inh_now = np.zeros(state_shape, dtype=np.float64)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._event_to_ex_in(
                ev,
                default_delay_steps=1,
                state_shape=state_shape,
                g=g,
                theta=theta,
                alpha=alpha,
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

    def _common_inputs_threshold(self, x, instant_rate_events, delayed_rate_events, g, theta, alpha):
        state_shape = self.rate.value.shape
        step_idx = int(np.asarray(self._step_count.value, dtype=np.int64).reshape(-1)[0])

        delayed_ex, delayed_in = self._drain_delayed_queue(step_idx, state_shape)
        delayed_ex_now, delayed_in_now = self._schedule_delayed_events_threshold(
            delayed_rate_events,
            step_idx=step_idx,
            state_shape=state_shape,
            g=g,
            theta=theta,
            alpha=alpha,
        )
        delayed_ex = delayed_ex + delayed_ex_now
        delayed_in = delayed_in + delayed_in_now

        instant_ex, instant_in = self._accumulate_instant_events_threshold(
            instant_rate_events,
            state_shape=state_shape,
            g=g,
            theta=theta,
            alpha=alpha,
        )

        delta_input = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0.0)), state_shape)
        instant_ex += np.where(delta_input > 0.0, delta_input, 0.0)
        instant_in += np.where(delta_input < 0.0, delta_input, 0.0)

        mu_ext = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.rate.value)), state_shape)

        return state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext

    def _common_parameters_threshold(self, state_shape):
        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        sigma = self._broadcast_to_state(self._to_numpy(self.sigma), state_shape)
        mu = self._broadcast_to_state(self._to_numpy(self.mu), state_shape)
        g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.theta), state_shape)
        alpha = self._broadcast_to_state(self._to_numpy(self.alpha), state_shape)
        return tau, sigma, mu, g, theta, alpha


class threshold_lin_rate_ipn(_threshold_lin_rate_base):
    r"""NEST-compatible input-noise threshold-linear rate neuron.

    Implements the NEST ``threshold_lin_rate_ipn`` model, an input-noise rate neuron
    with threshold-linear gain function. This model provides a piecewise-linear
    activation with lower and upper saturation bounds, commonly used for modeling
    neural populations with firing rate constraints and additive stochastic drive.

    Mathematical Description
    ------------------------

    **1. Continuous-Time Stochastic Dynamics**

    The rate state :math:`X(t)` evolves according to the Langevin equation:

    .. math::

       \tau\,dX(t) = [-\lambda X(t) + \mu + I_\mathrm{net}(t)]\,dt
       + \sqrt{\tau}\,\sigma\,dW(t),

    where:

    - :math:`\tau > 0` is the time constant (ms).
    - :math:`\lambda \ge 0` is the passive decay rate (dimensionless). Controls
      exponential relaxation; :math:`\lambda=0` yields driftless diffusion.
    - :math:`\mu` is the mean drive (dimensionless, external constant input).
    - :math:`\sigma \ge 0` is the input-noise strength (dimensionless).
    - :math:`W(t)` is a standard Wiener process.
    - :math:`I_\mathrm{net}(t)` is the network input (see below).

    **2. Threshold-Linear Gain Function**

    The input nonlinearity :math:`\phi(h)` is a threshold-linear function with
    saturation:

    .. math::

       \phi(h) = \min(\max(g(h-\theta), 0), \alpha),

    where:

    - :math:`g > 0` is the gain slope (dimensionless).
    - :math:`\theta` is the activation threshold (dimensionless).
    - :math:`\alpha > 0` is the saturation level (dimensionless).

    This function is zero for :math:`h < \theta`, linear with slope :math:`g` for
    :math:`\theta \le h < \theta + \alpha/g`, and saturates at :math:`\alpha` for
    :math:`h \ge \theta + \alpha/g`.

    **3. Network Input Structure**

    The network input :math:`I_\mathrm{net}(t)` is computed according to:

    .. math::

       I_\mathrm{net}(t) = \phi(I_\mathrm{ex}(t) + I_\mathrm{in}(t))
       \quad\text{(if linear\_summation=True)},

    or:

    .. math::

       I_\mathrm{net}(t) = \phi(I_\mathrm{ex}(t)) + \phi(I_\mathrm{in}(t))
       \quad\text{(if linear\_summation=False)},

    where :math:`I_\mathrm{ex}(t)` and :math:`I_\mathrm{in}(t)` are excitatory and
    inhibitory branches (sign-separated by event weight).

    **Note**: Unlike the base ``rate_neuron_ipn`` model, multiplicative coupling
    :math:`H_\mathrm{ex}(X)`, :math:`H_\mathrm{in}(X)` is **not** supported for
    threshold-linear neurons in NEST. The ``mult_coupling`` parameter is accepted
    for API compatibility but has no effect on dynamics (coupling factors are
    constant 1.0).

    **4. Discrete-Time Integration (Stochastic Exponential Euler)**

    For time step :math:`h=dt` (in ms), the model uses exact Ornstein-Uhlenbeck
    integration for the linear part:

    .. math::

       X_{n+1} = P_1 X_n + P_2 (\mu + I_\mathrm{net,n}) + N\,\xi_n,

    where :math:`\xi_n\sim\mathcal{N}(0,1)` is standard Gaussian noise.

    **For** :math:`\lambda > 0`:

    .. math::

       P_1 = \exp\left(-\frac{\lambda h}{\tau}\right), \quad
       P_2 = \frac{1-P_1}{\lambda}, \quad
       N = \sigma\sqrt{\frac{1-P_1^2}{2\lambda}}.

    **For** :math:`\lambda = 0` (Euler-Maruyama):

    .. math::

       P_1=1, \quad P_2=\frac{h}{\tau}, \quad N=\sigma\sqrt{\frac{h}{\tau}}.

    **5. Update Ordering (Matching NEST ``rate_neuron_ipn_impl.h``)**

    Per simulation step:

    1. **Store outgoing delayed value**: current ``rate`` is recorded as
       ``delayed_rate``.
    2. **Draw noise**: sample :math:`\xi_n\sim\mathcal{N}(0,1)`, compute
       :math:`\mathrm{noise}_n=\sigma\,\xi_n`.
    3. **Propagate intrinsic dynamics**: apply stochastic exponential Euler to
       :math:`X_n` with external drive and noise.
    4. **Read event buffers**: drain delayed events arriving at current step;
       accumulate instantaneous events.
    5. **Apply network input with threshold-linear gain**:

       - ``linear_summation=True``: nonlinearity applied to summed branch input
         during update: :math:`I_\mathrm{net}=\phi(I_\mathrm{ex}+I_\mathrm{in})`.
       - ``linear_summation=False``: nonlinearity applied per event during
         buffering: :math:`I_\mathrm{net}=\phi(I_\mathrm{ex})+\phi(I_\mathrm{in})`.

    6. **Rectification** (optional): if ``rectify_output=True``, clamp
       :math:`X_{n+1}\gets\max(X_{n+1},\,\mathrm{rectify\_rate})`.
    7. **Update state variables**: ``rate``, ``noise``, ``delayed_rate``,
       ``instant_rate``, ``_step_count``.

    **6. Numerical Stability and Computational Complexity**

    - Construction enforces :math:`\tau>0`, :math:`\lambda\ge 0`,
      :math:`\sigma\ge 0`, :math:`\mathrm{rectify\_rate}\ge 0`.
    - The threshold-linear gain is evaluated using ``np.minimum`` and ``np.maximum``
      for numerically stable clipping.
    - Per-call cost is :math:`O(\prod\mathrm{varshape})` with vectorized NumPy
      operations in float64.
    - The exponential Euler scheme is numerically stable for all :math:`h>0`.

    Parameters
    ----------
    in_size : Size
        Population shape (tuple or int). All per-neuron parameters are broadcast
        to ``self.varshape``.
    tau : ArrayLike, optional
        Time constant :math:`\tau` (ms). Scalar or array broadcastable to
        ``self.varshape``. Must be :math:`>0`. Default: ``10.0 * u.ms``.
    lambda_ : ArrayLike, optional
        Passive decay rate :math:`\lambda` (dimensionless). Scalar or array
        broadcastable to ``self.varshape``. Must be :math:`\ge 0`. Controls
        exponential relaxation (:math:`\lambda=0` yields driftless diffusion).
        Default: ``1.0``.
    sigma : ArrayLike, optional
        Input-noise scale :math:`\sigma` (dimensionless). Scalar or array
        broadcastable to ``self.varshape``. Must be :math:`\ge 0`. Default:
        ``1.0``.
    mu : ArrayLike, optional
        Mean drive :math:`\mu` (dimensionless). Scalar or array broadcastable to
        ``self.varshape``. External constant input to the rate dynamics. Default:
        ``0.0``.
    g : ArrayLike, optional
        Gain slope :math:`g` (dimensionless) for the threshold-linear function
        :math:`\phi(h)=\min(\max(g(h-\theta),0),\alpha)`. Scalar or array
        broadcastable to ``self.varshape``. Default: ``1.0``.
    theta : ArrayLike, optional
        Activation threshold :math:`\theta` (dimensionless). The gain function is
        zero for :math:`h<\theta`. Scalar or array broadcastable to
        ``self.varshape``. Default: ``0.0``.
    alpha : ArrayLike, optional
        Saturation level :math:`\alpha` (dimensionless). The gain function
        saturates at :math:`\alpha` for large inputs. Scalar or array broadcastable
        to ``self.varshape``. Default: ``np.inf`` (no saturation).
    mult_coupling : bool, optional
        API compatibility flag. Has **no effect** on dynamics for threshold-linear
        neurons (multiplicative coupling factors are constant 1.0). Default:
        ``False``.
    linear_summation : bool, optional
        Controls where the threshold-linear gain is applied. If ``True``, the gain
        is applied to the sum of excitatory and inhibitory inputs. If ``False``,
        the gain is applied separately to each input branch (matching NEST event
        semantics). Default: ``True``.
    rectify_rate : ArrayLike, optional
        Lower bound :math:`X_\mathrm{min}` for the rate when
        ``rectify_output=True`` (dimensionless). Scalar or array broadcastable to
        ``self.varshape``. Must be :math:`\ge 0`. Default: ``0.0``.
    rectify_output : bool, optional
        If ``True``, clamp the rate output to
        :math:`X\ge\mathrm{rectify\_rate}` after each update step. Default:
        ``False``.
    rate_initializer : Callable, optional
        Initializer for the ``rate`` state variable :math:`X_0`. Callable
        compatible with ``braintools.init`` API. Default:
        ``braintools.init.Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for the ``noise`` state variable (records last noise sample
        :math:`\sigma\,\xi_{n-1}`). Callable compatible with ``braintools.init``
        API. Default: ``braintools.init.Constant(0.0)``.
    name : str or None, optional
        Module name for identification in hierarchies. If ``None``, an
        auto-generated name is used. Default: ``None``.

    Parameter Mapping
    -----------------

    The following table maps NEST ``threshold_lin_rate_ipn`` parameters to
    brainpy.state equivalents:

    =============================== ===================== =========
    NEST Parameter                  brainpy.state         Default
    =============================== ===================== =========
    ``tau``                         ``tau``               10 ms
    ``lambda``                      ``lambda_``           1.0
    ``sigma``                       ``sigma``             1.0
    ``mu``                          ``mu``                0.0
    ``g`` (gain slope)              ``g``                 1.0
    ``theta`` (threshold)           ``theta``             0.0
    ``alpha`` (saturation)          ``alpha``             inf
    ``mult_coupling``               ``mult_coupling``     False
                                    (no effect)
    ``linear_summation``            ``linear_summation``  True
    ``rectify_rate``                ``rectify_rate``      0.0
    ``rectify_output``              ``rectify_output``    False
    =============================== ===================== =========

    Attributes
    ----------
    rate : brainstate.ShortTermState
        Current rate state :math:`X_n` (float64 array of shape ``self.varshape``
        or ``(batch_size,) + self.varshape``).
    noise : brainstate.ShortTermState
        Last noise sample :math:`\sigma\,\xi_{n-1}` (float64 array, same shape as
        ``rate``).
    instant_rate : brainstate.ShortTermState
        Rate value after instantaneous event application (float64 array, same
        shape as ``rate``).
    delayed_rate : brainstate.ShortTermState
        Rate value before current update, used for delayed projections (float64
        array, same shape as ``rate``).
    _step_count : brainstate.ShortTermState
        Internal step counter for delayed event scheduling (int64 scalar).
    _delayed_ex_queue : dict
        Internal queue mapping ``step_idx`` to accumulated excitatory delayed
        events.
    _delayed_in_queue : dict
        Internal queue mapping ``step_idx`` to accumulated inhibitory delayed
        events.

    Raises
    ------
    ValueError
        If ``tau <= 0``, ``lambda_ < 0``, ``sigma < 0``, or
        ``rectify_rate < 0``.
    ValueError
        If ``instant_rate_events`` contain non-zero ``delay_steps``.
    ValueError
        If ``delayed_rate_events`` contain negative ``delay_steps``.
    ValueError
        If event tuples have length other than 2, 3, or 4.

    Notes
    -----
    **Runtime Event Semantics**

    - ``instant_rate_events``: Applied in the current step without delay. Each
      event can be:

      - A scalar (treated as ``rate`` value with ``weight=1.0``).
      - A tuple ``(rate, weight)`` or ``(rate, weight, delay_steps)`` or
        ``(rate, weight, delay_steps, multiplicity)``.
      - A dict with keys ``'rate'``/``'coeff'``/``'value'``, ``'weight'``,
        ``'delay_steps'``/``'delay'``, ``'multiplicity'``.

    - ``delayed_rate_events``: Scheduled with integer ``delay_steps`` (units of
      simulation time step). Same format as ``instant_rate_events``.

    - Sign convention: events with ``weight >= 0`` contribute to the excitatory
      branch; events with ``weight < 0`` contribute to the inhibitory branch.

    - For ``linear_summation=False``, event values are transformed by the
      threshold-linear gain during buffering (matching NEST event handlers).

    **Comparison to Other Rate Neuron Variants**

    - ``rate_neuron_ipn``: Uses linear or custom gain function with optional
      multiplicative coupling. ``threshold_lin_rate_ipn`` is a special case with
      fixed threshold-linear gain and no multiplicative coupling.
    - ``threshold_lin_rate_opn``: Output-noise variant (noise applied after
      nonlinearity) vs. input noise (applied before dynamics propagation).

    **Failure Modes**


    - No automatic failure handling. Negative time constants, decay rates, or
      noise parameters are caught at construction by ``_validate_parameters``.
    - Invalid event formats raise ``ValueError`` during update.
    - Numerical instability is unlikely due to exact OU integration and stable
      clipping operations, but extreme parameter combinations (very large
      :math:`\sigma`, very small :math:`\tau`) may lead to rate explosions
      without ``rectify_output=True``.

    Examples
    --------
    **Example 1**: Minimal threshold-linear rate neuron with external drive.

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import brainunit as u
       >>> model = bst.threshold_lin_rate_ipn(
       ...     in_size=10, tau=20*u.ms, sigma=0.5, g=2.0, theta=1.0
       ... )
       >>> model.init_all_states(batch_size=1)
       >>> rate = model(x=0.5)  # external drive
       >>> print(rate.shape)
       (1, 10)

    **Example 2**: Saturating threshold-linear neuron with rectification.

    .. code-block:: python

       >>> model = bst.threshold_lin_rate_ipn(
       ...     in_size=5,
       ...     tau=10*u.ms,
       ...     lambda_=2.0,
       ...     g=1.0, theta=0.5, alpha=5.0,
       ...     rectify_rate=0.0, rectify_output=True
       ... )
       >>> model.init_all_states()

    **Example 3**: Update with instantaneous and delayed events.

    .. code-block:: python

       >>> model = bst.threshold_lin_rate_ipn(in_size=3, tau=10*u.ms, sigma=0.1)
       >>> model.init_all_states()
       >>> instant_event = {'rate': 2.0, 'weight': 0.1}
       >>> delayed_event = {'rate': 1.5, 'weight': -0.05, 'delay_steps': 3}
       >>> rate = model.update(
       ...     x=0.2,
       ...     instant_rate_events=instant_event,
       ...     delayed_rate_events=delayed_event
       ... )

    References
    ----------
    .. [1] NEST Simulator Documentation: ``threshold_lin_rate_ipn``
           https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_ipn.html
    .. [2] NEST Simulator Documentation: ``threshold_lin_rate`` nonlinearity
           https://nest-simulator.readthedocs.io/en/stable/models/rate_transformer_node.html
    .. [3] Hahne, J., Dahmen, D., Schuecker, J., Frommer, A., Bolten, M.,
           Helias, M., & Diesmann, M. (2017). Integration of continuous-time
           dynamics in a spiking neural network simulator.
           *Frontiers in Neuroinformatics*, 11, 34.
           https://doi.org/10.3389/fninf.2017.00034

    See Also
    --------
    threshold_lin_rate_opn : Output-noise variant of the threshold-linear rate neuron.
    rate_neuron_ipn : General input-noise rate neuron with custom gain functions.
    lin_rate : Deterministic linear rate neuron (``sigma=0``, no threshold).
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
        alpha: ArrayLike = np.inf,
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
        self.alpha = braintools.init.param(alpha, self.varshape)
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
            ``['rate', 'noise']``. The ``rate`` variable records the current rate
            state :math:`X_n`, and ``noise`` records the last noise sample
            :math:`\sigma\,\xi_{n-1}`.

        Notes
        -----
        These variables can be accessed via recording tools in BrainPy for
        post-simulation analysis of rate dynamics and noise contributions.
        """
        return ['rate', 'noise']

    @property
    def receptor_types(self):
        r"""Receptor type dictionary for projection compatibility.

        Returns
        -------
        dict[str, int]
            ``{'RATE': 0}``. Rate neurons have a single unified receptor port
            for all rate-based inputs. Excitatory vs. inhibitory separation is
            handled internally via event weight signs.

        Notes
        -----
        This property is used by projection objects to validate connection targets.
        Unlike spiking neurons with separate AMPA/GABA receptor ports, rate neurons
        use sign-based branch routing (``weight >= 0`` → excitatory branch,
        ``weight < 0`` → inhibitory branch).
        """
        return {'RATE': 0}

    def _validate_parameters(self):
        r"""Validate model parameters at construction time.

        Raises
        ------
        ValueError
            If ``tau <= 0``, ``lambda_ < 0``, ``sigma < 0``, or
            ``rectify_rate < 0``.

        Notes
        -----
        This method is called automatically during ``__init__``.
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
        r"""Initialize all state variables for simulation.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If ``None`` (default), state shape is
            ``self.varshape``. If ``int``, state shape is
            ``(batch_size,) + self.varshape``.
        **kwargs
            Additional keyword arguments (reserved for future use).

        Notes
        -----
        This method initializes:

        - ``rate``: Current rate state :math:`X_n`.
        - ``noise``: Last noise sample :math:`\sigma\,\xi_{n-1}`.
        - ``instant_rate``: Rate after instantaneous event application.
        - ``delayed_rate``: Rate before current update (for delayed projections).
        - ``_step_count``: Internal step counter for delay scheduling.
        - ``_delayed_ex_queue``, ``_delayed_in_queue``: Delay queues.

        All state arrays are initialized as float64 NumPy arrays using the
        provided initializers.
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
        r"""Perform one simulation step of stochastic threshold-linear rate dynamics.

        Parameters
        ----------
        x : ArrayLike, optional
            External drive (scalar or array broadcastable to ``self.varshape``).
            Added to ``mu`` as constant forcing. Default is ``0.0``.
        instant_rate_events : None, dict, tuple, list, or iterable, optional
            Instantaneous rate events applied in the current step without delay.
            See class docstring for event format. Default is ``None``.
        delayed_rate_events : None, dict, tuple, list, or iterable, optional
            Delayed rate events scheduled with integer ``delay_steps`` (units of
            simulation time step). See class docstring for event format. Default
            is ``None``.
        noise : ArrayLike, optional
            Externally supplied noise sample :math:`\xi_n` (scalar or array
            broadcastable to state shape). If ``None`` (default), draws
            :math:`\xi_n\sim\mathcal{N}(0,1)` internally.

        Returns
        -------
        rate_new : np.ndarray
            Updated rate state :math:`X_{n+1}` (float64 array of shape
            ``self.rate.value.shape``).

        Notes
        -----
        **Update algorithm**:

        1. Collect input contributions:

           - Delayed events arriving at current step (from internal queues).
           - Newly scheduled delayed events with ``delay_steps=0``.
           - Instantaneous events.
           - Delta inputs (sign-separated into excitatory/inhibitory).
           - Current inputs via ``sum_current_inputs(x, rate)``.

        2. Compute propagator coefficients:

           For :math:`\lambda>0`:

           .. math::

              P_1 = \exp(-\lambda h/\tau), \quad
              P_2 = (1-P_1)/\lambda, \quad
              N = \sigma\sqrt{(1-P_1^2)/(2\lambda)}.

           For :math:`\lambda=0`: :math:`P_1=1`, :math:`P_2=h/\tau`,
           :math:`N=\sigma\sqrt{h/\tau}`.

        3. Propagate intrinsic dynamics:

           .. math::

              X' = P_1 X_n + P_2(\mu + \mu_\mathrm{ext}) + N\,\xi_n.

        4. Apply network input with threshold-linear gain:

           - ``linear_summation=True``:
             :math:`X' \gets X' + P_2\,\phi(I_\mathrm{ex}+I_\mathrm{in})`.
           - ``linear_summation=False``:
             :math:`X' \gets X' + P_2\,[\phi(I_\mathrm{ex})+\phi(I_\mathrm{in})]`.

           where :math:`\phi(h)=\min(\max(g(h-\theta),0),\alpha)`.

        5. Apply optional output rectification:
           :math:`X_{n+1}\gets\max(X',\,\mathrm{rectify\_rate})`.

        6. Update state variables: ``rate``, ``noise``, ``delayed_rate``,
           ``instant_rate``, ``_step_count``.

        **Numerical stability**: The threshold-linear gain uses ``np.minimum`` and
        ``np.maximum`` for stable clipping. The exponential Euler scheme uses
        ``np.expm1`` for numerically stable evaluation of :math:`1-e^{-x}` and
        handles the :math:`\lambda=0` limit explicitly.

        **Failure modes**: No automatic failure handling. Negative time constants,
        decay rates, or noise parameters are caught at construction by
        ``_validate_parameters``. Invalid event formats raise ``ValueError``.

        Examples
        --------
        Single update step with external drive:

        .. code-block:: python

           >>> import brainpy.state as bst
           >>> import brainunit as u
           >>> model = bst.threshold_lin_rate_ipn(
           ...     in_size=5, tau=10*u.ms, g=2.0, theta=1.0
           ... )
           >>> model.init_all_states()
           >>> rate = model.update(x=0.5)

        Update with instantaneous event:

        .. code-block:: python

           >>> event = {'rate': 2.0, 'weight': 0.1}
           >>> rate = model.update(instant_rate_events=event)

        Update with delayed event (arrives 3 steps later):

        .. code-block:: python

           >>> event = {'rate': 1.5, 'weight': 0.1, 'delay_steps': 3}
           >>> rate = model.update(delayed_rate_events=event)
        """
        h = float(u.math.asarray(brainstate.environ.get_dt() / u.ms))
        state_shape = self.rate.value.shape

        tau, sigma, mu, g, theta, alpha = self._common_parameters_threshold(state_shape)
        lambda_ = self._broadcast_to_state(self._to_numpy(self.lambda_), state_shape)
        rectify_rate = self._broadcast_to_state(self._to_numpy(self.rectify_rate), state_shape)

        state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext = self._common_inputs_threshold(
            x=x,
            instant_rate_events=instant_rate_events,
            delayed_rate_events=delayed_rate_events,
            g=g,
            theta=theta,
            alpha=alpha,
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
                rate_new += P2 * H_ex * self._input(delayed_ex + instant_ex, g, theta, alpha)
                rate_new += P2 * H_in * self._input(delayed_in + instant_in, g, theta, alpha)
            else:
                rate_new += P2 * self._input(delayed_ex + instant_ex + delayed_in + instant_in, g, theta, alpha)
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


class threshold_lin_rate_opn(_threshold_lin_rate_base):
    r"""NEST-compatible output-noise threshold-linear rate neuron.

    Implements the NEST ``threshold_lin_rate_opn`` model, an output-noise rate
    neuron with threshold-linear gain function. Unlike the input-noise variant
    (``threshold_lin_rate_ipn``), noise is applied to the output after deterministic
    dynamics, leading to different stationary distributions and noise scaling.

    Mathematical Description
    ------------------------

    **1. Continuous-Time Deterministic Dynamics with Output Noise**

    The rate state :math:`X(t)` evolves according to the deterministic ODE:

    .. math::

       \tau\frac{dX(t)}{dt} = -X(t) + \mu + I_\mathrm{net}(t),

    where:

    - :math:`\tau > 0` is the time constant (ms).
    - :math:`\mu` is the mean drive (dimensionless, external constant input).
    - :math:`I_\mathrm{net}(t)` is the network input (see below).

    The **output** rate :math:`X_\mathrm{noisy}(t)` is obtained by adding noise to
    the deterministic state:

    .. math::

       X_\mathrm{noisy}(t) = X(t) + \sqrt{\frac{\tau}{h}}\,\sigma\,\xi(t),

    where:

    - :math:`\sigma \ge 0` is the output-noise scale (dimensionless).
    - :math:`\xi(t)\sim\mathcal{N}(0,1)` is standard Gaussian white noise.
    - :math:`h=dt` is the simulation time step (ms).

    The :math:`\sqrt{\tau/h}` scaling ensures the noise variance is independent of
    the time step for small :math:`h`.

    **2. Threshold-Linear Gain Function**

    The input nonlinearity :math:`\phi(h)` is identical to the input-noise variant:

    .. math::

       \phi(h) = \min(\max(g(h-\theta), 0), \alpha),

    where:

    - :math:`g > 0` is the gain slope (dimensionless).
    - :math:`\theta` is the activation threshold (dimensionless).
    - :math:`\alpha > 0` is the saturation level (dimensionless).

    **3. Network Input Structure**

    The network input :math:`I_\mathrm{net}(t)` is computed according to:

    .. math::

       I_\mathrm{net}(t) = \phi(I_\mathrm{ex}(t) + I_\mathrm{in}(t))
       \quad\text{(if linear\_summation=True)},

    or:

    .. math::

       I_\mathrm{net}(t) = \phi(I_\mathrm{ex}(t)) + \phi(I_\mathrm{in}(t))
       \quad\text{(if linear\_summation=False)},

    where :math:`I_\mathrm{ex}(t)` and :math:`I_\mathrm{in}(t)` are excitatory and
    inhibitory branches (sign-separated by event weight).

    **Note**: Multiplicative coupling is **not** supported (``mult_coupling``
    parameter is accepted for API compatibility but has no effect).

    **4. Discrete-Time Integration (Exponential Euler)**

    For time step :math:`h=dt` (in ms), the deterministic dynamics are integrated
    using exponential Euler:

    .. math::

       X_{n+1} = P_1 X_n + P_2 (\mu + I_\mathrm{net,n}),

    where:

    .. math::

       P_1 = \exp\left(-\frac{h}{\tau}\right), \quad
       P_2 = 1 - P_1 = -\mathrm{expm1}\left(-\frac{h}{\tau}\right).

    The noisy output is:

    .. math::

       X_\mathrm{noisy,n} = X_n + \sqrt{\frac{\tau}{h}}\,\sigma\,\xi_n,

    where :math:`\xi_n\sim\mathcal{N}(0,1)`.

    **5. Update Ordering (Matching NEST ``rate_neuron_opn_impl.h``)**

    Per simulation step:

    1. **Draw noise**: sample :math:`\xi_n\sim\mathcal{N}(0,1)`, compute
       :math:`\mathrm{noise}_n=\sigma\,\xi_n`.
    2. **Build noisy output**: compute
       :math:`X_\mathrm{noisy,n}=X_n+\sqrt{\tau/h}\,\mathrm{noise}_n` and store
       as both ``delayed_rate`` and ``instant_rate`` (outgoing values for
       projections).
    3. **Propagate deterministic dynamics**: apply exponential Euler to update
       :math:`X_n`.
    4. **Read event buffers**: drain delayed events arriving at current step;
       accumulate instantaneous events.
    5. **Apply network input with threshold-linear gain**:

       - ``linear_summation=True``:
         :math:`X_{n+1} \gets X_{n+1} + P_2\,\phi(I_\mathrm{ex}+I_\mathrm{in})`.
       - ``linear_summation=False``:
         :math:`X_{n+1} \gets X_{n+1} + P_2\,[\phi(I_\mathrm{ex})+\phi(I_\mathrm{in})]`.

    6. **Update state variables**: ``rate``, ``noise``, ``noisy_rate``,
       ``delayed_rate``, ``instant_rate``, ``_step_count``.

    **Note**: Unlike input-noise variant, there is **no** rectification option for
    output-noise neurons. The noise is applied to the output only and does not
    affect the internal deterministic state.

    **6. Numerical Stability and Computational Complexity**

    - Construction enforces :math:`\tau>0`, :math:`\sigma\ge 0`.
    - The threshold-linear gain is evaluated using ``np.minimum`` and ``np.maximum``
      for numerically stable clipping.
    - Per-call cost is :math:`O(\prod\mathrm{varshape})` with vectorized NumPy
      operations in float64.
    - The exponential Euler scheme is numerically stable for all :math:`h>0`.

    Parameters
    ----------
    in_size : Size
        Population shape (tuple or int). All per-neuron parameters are broadcast
        to ``self.varshape``.
    tau : ArrayLike, optional
        Time constant :math:`\tau` (ms). Scalar or array broadcastable to
        ``self.varshape``. Must be :math:`>0`. Default: ``10.0 * u.ms``.
    sigma : ArrayLike, optional
        Output-noise scale :math:`\sigma` (dimensionless). Scalar or array
        broadcastable to ``self.varshape``. Must be :math:`\ge 0`. Default:
        ``1.0``.
    mu : ArrayLike, optional
        Mean drive :math:`\mu` (dimensionless). Scalar or array broadcastable to
        ``self.varshape``. External constant input to the rate dynamics. Default:
        ``0.0``.
    g : ArrayLike, optional
        Gain slope :math:`g` (dimensionless) for the threshold-linear function
        :math:`\phi(h)=\min(\max(g(h-\theta),0),\alpha)`. Scalar or array
        broadcastable to ``self.varshape``. Default: ``1.0``.
    theta : ArrayLike, optional
        Activation threshold :math:`\theta` (dimensionless). The gain function is
        zero for :math:`h<\theta`. Scalar or array broadcastable to
        ``self.varshape``. Default: ``0.0``.
    alpha : ArrayLike, optional
        Saturation level :math:`\alpha` (dimensionless). The gain function
        saturates at :math:`\alpha` for large inputs. Scalar or array broadcastable
        to ``self.varshape``. Default: ``np.inf`` (no saturation).
    mult_coupling : bool, optional
        API compatibility flag. Has **no effect** on dynamics for threshold-linear
        neurons (multiplicative coupling factors are constant 1.0). Default:
        ``False``.
    linear_summation : bool, optional
        Controls where the threshold-linear gain is applied. If ``True``, the gain
        is applied to the sum of excitatory and inhibitory inputs. If ``False``,
        the gain is applied separately to each input branch (matching NEST event
        semantics). Default: ``True``.
    rate_initializer : Callable, optional
        Initializer for the ``rate`` state variable :math:`X_0`. Callable
        compatible with ``braintools.init`` API. Default:
        ``braintools.init.Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for the ``noise`` state variable (records last noise sample
        :math:`\sigma\,\xi_{n-1}`). Callable compatible with ``braintools.init``
        API. Default: ``braintools.init.Constant(0.0)``.
    noisy_rate_initializer : Callable, optional
        Initializer for the ``noisy_rate`` state variable :math:`X_\mathrm{noisy,0}`
        (initial noisy output). Callable compatible with ``braintools.init`` API.
        Default: ``braintools.init.Constant(0.0)``.
    name : str or None, optional
        Module name for identification in hierarchies. If ``None``, an
        auto-generated name is used. Default: ``None``.

    Parameter Mapping
    -----------------

    The following table maps NEST ``threshold_lin_rate_opn`` parameters to
    brainpy.state equivalents:

    =============================== ===================== =========
    NEST Parameter                  brainpy.state         Default
    =============================== ===================== =========
    ``tau``                         ``tau``               10 ms
    ``sigma``                       ``sigma``             1.0
    ``mu``                          ``mu``                0.0
    ``g`` (gain slope)              ``g``                 1.0
    ``theta`` (threshold)           ``theta``             0.0
    ``alpha`` (saturation)          ``alpha``             inf
    ``mult_coupling``               ``mult_coupling``     False
                                    (no effect)
    ``linear_summation``            ``linear_summation``  True
    =============================== ===================== =========

    **Note**: Unlike ``threshold_lin_rate_ipn``, this model does **not** have
    ``lambda`` (passive decay is fixed at 1.0), ``rectify_rate``, or
    ``rectify_output`` parameters.

    Attributes
    ----------
    rate : brainstate.ShortTermState
        Current deterministic rate state :math:`X_n` (float64 array of shape
        ``self.varshape`` or ``(batch_size,) + self.varshape``).
    noise : brainstate.ShortTermState
        Last noise sample :math:`\sigma\,\xi_{n-1}` (float64 array, same shape as
        ``rate``).
    noisy_rate : brainstate.ShortTermState
        Noisy output rate :math:`X_\mathrm{noisy,n}=X_n+\sqrt{\tau/h}\,\sigma\,\xi_n`
        (float64 array, same shape as ``rate``).
    instant_rate : brainstate.ShortTermState
        Noisy rate value used for instantaneous projections (float64 array, same
        shape as ``rate``).
    delayed_rate : brainstate.ShortTermState
        Noisy rate value used for delayed projections (float64 array, same shape
        as ``rate``).
    _step_count : brainstate.ShortTermState
        Internal step counter for delayed event scheduling (int64 scalar).
    _delayed_ex_queue : dict
        Internal queue mapping ``step_idx`` to accumulated excitatory delayed
        events.
    _delayed_in_queue : dict
        Internal queue mapping ``step_idx`` to accumulated inhibitory delayed
        events.

    Raises
    ------
    ValueError
        If ``tau <= 0`` or ``sigma < 0``.
    ValueError
        If ``instant_rate_events`` contain non-zero ``delay_steps``.
    ValueError
        If ``delayed_rate_events`` contain negative ``delay_steps``.
    ValueError
        If event tuples have length other than 2, 3, or 4.

    Notes
    -----
    **Runtime Event Semantics**

    Event formats are identical to :class:`threshold_lin_rate_ipn`:

    - ``instant_rate_events``: Applied in the current step without delay.
    - ``delayed_rate_events``: Scheduled with integer ``delay_steps``.
    - Sign convention: ``weight >= 0`` → excitatory, ``weight < 0`` → inhibitory.

    **Comparison to Input-Noise Variant**

    The key differences between ``threshold_lin_rate_opn`` (output noise) and
    ``threshold_lin_rate_ipn`` (input noise) are:

    - **Noise location**: Output noise is added after nonlinearity; input noise is
      integrated before nonlinearity.
    - **Stationary distribution**: Output noise does not affect the mean of the
      deterministic attractor; input noise shifts the effective drive.
    - **Dynamics**: Output-noise model has simpler deterministic dynamics
      (:math:`\lambda=1.0` fixed) with additive output corruption.
    - **Rectification**: Input-noise variant supports ``rectify_output``; output-
      noise variant does not (noise is on output only).

    **Failure Modes**


    - No automatic failure handling. Negative time constants or noise parameters
      are caught at construction by ``_validate_parameters``.
    - Invalid event formats raise ``ValueError`` during update.
    - The noise scaling :math:`\sqrt{\tau/h}` can become large for small time
      steps, but this is by design to ensure correct variance scaling.

    Examples
    --------
    **Example 1**: Minimal output-noise threshold-linear neuron.

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import brainunit as u
       >>> model = bst.threshold_lin_rate_opn(
       ...     in_size=10, tau=20*u.ms, sigma=0.5, g=2.0, theta=1.0
       ... )
       >>> model.init_all_states(batch_size=1)
       >>> rate = model(x=0.5)  # deterministic state
       >>> noisy_rate = model.noisy_rate.value  # noisy output

    **Example 2**: Saturating threshold-linear neuron with output noise.

    .. code-block:: python

       >>> model = bst.threshold_lin_rate_opn(
       ...     in_size=5,
       ...     tau=10*u.ms,
       ...     sigma=0.2,
       ...     g=1.5, theta=0.5, alpha=5.0
       ... )
       >>> model.init_all_states()

    **Example 3**: Update with events (identical to input-noise variant).

    .. code-block:: python

       >>> model = bst.threshold_lin_rate_opn(in_size=3, tau=10*u.ms, sigma=0.1)
       >>> model.init_all_states()
       >>> instant_event = {'rate': 2.0, 'weight': 0.1}
       >>> delayed_event = {'rate': 1.5, 'weight': -0.05, 'delay_steps': 3}
       >>> rate = model.update(
       ...     x=0.2,
       ...     instant_rate_events=instant_event,
       ...     delayed_rate_events=delayed_event
       ... )

    References
    ----------
    .. [1] NEST Simulator Documentation: ``threshold_lin_rate_opn``
           https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_opn.html
    .. [2] NEST Simulator Documentation: ``threshold_lin_rate`` nonlinearity
           https://nest-simulator.readthedocs.io/en/stable/models/rate_transformer_node.html
    .. [3] Hahne, J., Dahmen, D., Schuecker, J., Frommer, A., Bolten, M.,
           Helias, M., & Diesmann, M. (2017). Integration of continuous-time
           dynamics in a spiking neural network simulator.
           *Frontiers in Neuroinformatics*, 11, 34.
           https://doi.org/10.3389/fninf.2017.00034

    See Also
    --------
    threshold_lin_rate_ipn : Input-noise variant of the threshold-linear rate neuron.
    rate_neuron_opn : General output-noise rate neuron with custom gain functions.
    lin_rate : Deterministic linear rate neuron (``sigma=0``, no threshold).
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
        alpha: ArrayLike = np.inf,
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
        self.alpha = braintools.init.param(alpha, self.varshape)
        self.noisy_rate_initializer = noisy_rate_initializer
        self._validate_parameters()

    @property
    def recordables(self):
        r"""List of state variable names that can be recorded during simulation.

        Returns
        -------
        list of str
            ``['rate', 'noise', 'noisy_rate']``. The ``rate`` variable records the
            deterministic rate state :math:`X_n`, ``noise`` records the last noise
            sample :math:`\sigma\,\xi_{n-1}`, and ``noisy_rate`` records the noisy
            output :math:`X_\mathrm{noisy,n}`.

        Notes
        -----
        These variables can be accessed via recording tools in BrainPy for
        post-simulation analysis. The ``noisy_rate`` is the value transmitted to
        downstream neurons via projections.
        """
        return ['rate', 'noise', 'noisy_rate']

    @property
    def receptor_types(self):
        r"""Receptor type dictionary for projection compatibility.

        Returns
        -------
        dict[str, int]
            ``{'RATE': 0}``. Rate neurons have a single unified receptor port
            for all rate-based inputs. Excitatory vs. inhibitory separation is
            handled internally via event weight signs.

        Notes
        -----
        This property is used by projection objects to validate connection targets.
        Unlike spiking neurons with separate AMPA/GABA receptor ports, rate neurons
        use sign-based branch routing (``weight >= 0`` → excitatory branch,
        ``weight < 0`` → inhibitory branch).
        """
        return {'RATE': 0}

    def _validate_parameters(self):
        r"""Validate model parameters at construction time.

        Raises
        ------
        ValueError
            If ``tau <= 0`` or ``sigma < 0``.

        Notes
        -----
        This method is called automatically during ``__init__``. Unlike the input-
        noise variant, this model does not have ``lambda_`` or ``rectify_rate``
        parameters to validate.
        """
        if np.any(self._to_numpy_ms(self.tau) <= 0.0):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self._to_numpy(self.sigma) < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables for simulation.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If ``None`` (default), state shape is
            ``self.varshape``. If ``int``, state shape is
            ``(batch_size,) + self.varshape``.
        **kwargs
            Additional keyword arguments (reserved for future use).

        Notes
        -----
        This method initializes:

        - ``rate``: Deterministic rate state :math:`X_n`.
        - ``noise``: Last noise sample :math:`\sigma\,\xi_{n-1}`.
        - ``noisy_rate``: Noisy output :math:`X_\mathrm{noisy,n}`.
        - ``instant_rate``: Noisy rate for instantaneous projections.
        - ``delayed_rate``: Noisy rate for delayed projections.
        - ``_step_count``: Internal step counter for delay scheduling.
        - ``_delayed_ex_queue``, ``_delayed_in_queue``: Delay queues.

        All state arrays are initialized as float64 NumPy arrays using the
        provided initializers. Both ``instant_rate`` and ``delayed_rate`` are
        initialized to ``noisy_rate`` (outgoing values are noisy).
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
        r"""Perform one simulation step of deterministic threshold-linear rate dynamics with output noise.

        Parameters
        ----------
        x : ArrayLike, optional
            External drive (scalar or array broadcastable to ``self.varshape``).
            Added to ``mu`` as constant forcing. Default is ``0.0``.
        instant_rate_events : None, dict, tuple, list, or iterable, optional
            Instantaneous rate events applied in the current step without delay.
            See class docstring for event format. Default is ``None``.
        delayed_rate_events : None, dict, tuple, list, or iterable, optional
            Delayed rate events scheduled with integer ``delay_steps`` (units of
            simulation time step). See class docstring for event format. Default
            is ``None``.
        noise : ArrayLike, optional
            Externally supplied noise sample :math:`\xi_n` (scalar or array
            broadcastable to state shape). If ``None`` (default), draws
            :math:`\xi_n\sim\mathcal{N}(0,1)` internally.

        Returns
        -------
        rate_new : np.ndarray
            Updated deterministic rate state :math:`X_{n+1}` (float64 array of
            shape ``self.rate.value.shape``).

        Notes
        -----
        **Update algorithm**:

        1. **Draw noise and compute noisy output**:

           .. math::

              \mathrm{noise}_n = \sigma\,\xi_n, \quad
              X_\mathrm{noisy,n} = X_n + \sqrt{\frac{\tau}{h}}\,\mathrm{noise}_n.

           Store :math:`X_\mathrm{noisy,n}` as ``delayed_rate`` and
           ``instant_rate`` (outgoing values for projections).

        2. **Collect input contributions**:

           - Delayed events arriving at current step (from internal queues).
           - Newly scheduled delayed events with ``delay_steps=0``.
           - Instantaneous events.
           - Delta inputs (sign-separated into excitatory/inhibitory).
           - Current inputs via ``sum_current_inputs(x, rate)``.

        3. **Compute propagator coefficients** (deterministic exponential Euler):

           .. math::

              P_1 = \exp(-h/\tau), \quad P_2 = 1 - P_1 = -\mathrm{expm1}(-h/\tau).

        4. **Propagate deterministic dynamics**:

           .. math::

              X_{n+1} = P_1 X_n + P_2(\mu + \mu_\mathrm{ext}).

        5. **Apply network input with threshold-linear gain**:

           - ``linear_summation=True``:
             :math:`X_{n+1} \gets X_{n+1} + P_2\,\phi(I_\mathrm{ex}+I_\mathrm{in})`.
           - ``linear_summation=False``:
             :math:`X_{n+1} \gets X_{n+1} + P_2\,[\phi(I_\mathrm{ex})+\phi(I_\mathrm{in})]`.

           where :math:`\phi(h)=\min(\max(g(h-\theta),0),\alpha)`.

        6. **Update state variables**: ``rate``, ``noise``, ``noisy_rate``,
           ``delayed_rate``, ``instant_rate``, ``_step_count``.

        **Key difference from input-noise variant**: Noise is added to the output
        *before* the deterministic update, not during the stochastic integration.
        This means the internal state :math:`X_n` evolves deterministically, and
        only the transmitted rate is noisy.

        **Numerical stability**: The threshold-linear gain uses ``np.minimum`` and
        ``np.maximum`` for stable clipping. The exponential Euler scheme uses
        ``np.expm1`` for numerically stable evaluation of :math:`1-e^{-x}`. The
        noise scaling :math:`\sqrt{\tau/h}` ensures correct variance scaling as
        :math:`h\to 0`.

        **Failure modes**: No automatic failure handling. Negative time constants
        or noise parameters are caught at construction by ``_validate_parameters``.
        Invalid event formats raise ``ValueError``.

        Examples
        --------
        Single update step with external drive:

        .. code-block:: python

           >>> import brainpy.state as bst
           >>> import brainunit as u
           >>> model = bst.threshold_lin_rate_opn(
           ...     in_size=5, tau=10*u.ms, g=2.0, theta=1.0
           ... )
           >>> model.init_all_states()
           >>> rate = model.update(x=0.5)  # deterministic state
           >>> noisy_output = model.noisy_rate.value  # noisy transmitted value

        Update with instantaneous event:

        .. code-block:: python

           >>> event = {'rate': 2.0, 'weight': 0.1}
           >>> rate = model.update(instant_rate_events=event)

        Update with delayed event (arrives 3 steps later):

        .. code-block:: python

           >>> event = {'rate': 1.5, 'weight': 0.1, 'delay_steps': 3}
           >>> rate = model.update(delayed_rate_events=event)
        """
        h = float(u.math.asarray(brainstate.environ.get_dt() / u.ms))
        state_shape = self.rate.value.shape

        tau, sigma, mu, g, theta, alpha = self._common_parameters_threshold(state_shape)
        state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext = self._common_inputs_threshold(
            x=x,
            instant_rate_events=instant_rate_events,
            delayed_rate_events=delayed_rate_events,
            g=g,
            theta=theta,
            alpha=alpha,
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
                rate_new += P2 * H_ex * self._input(delayed_ex + instant_ex, g, theta, alpha)
                rate_new += P2 * H_in * self._input(delayed_in + instant_in, g, theta, alpha)
            else:
                rate_new += P2 * self._input(delayed_ex + instant_ex + delayed_in + instant_in, g, theta, alpha)
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
