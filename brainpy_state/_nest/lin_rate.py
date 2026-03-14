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


from typing import Callable, Tuple

import brainstate
import braintools
import saiunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron
from ._utils import is_tracer

__all__ = [
    'lin_rate_ipn',
    'lin_rate_opn',
]


class _lin_rate_base(NESTNeuron):
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau: ArrayLike,
        sigma: ArrayLike,
        mu: ArrayLike,
        g: ArrayLike,
        mult_coupling: bool,
        g_ex: ArrayLike,
        g_in: ArrayLike,
        theta_ex: ArrayLike,
        theta_in: ArrayLike,
        linear_summation: bool,
        rate_initializer: Callable,
        noise_initializer: Callable,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.tau = braintools.init.param(tau, self.varshape)
        self.sigma = braintools.init.param(sigma, self.varshape)
        self.mu = braintools.init.param(mu, self.varshape)
        self.g = braintools.init.param(g, self.varshape)
        self.mult_coupling = bool(mult_coupling)
        self.g_ex = braintools.init.param(g_ex, self.varshape)
        self.g_in = braintools.init.param(g_in, self.varshape)
        self.theta_ex = braintools.init.param(theta_ex, self.varshape)
        self.theta_in = braintools.init.param(theta_in, self.varshape)
        self.linear_summation = bool(linear_summation)

        self.rate_initializer = rate_initializer
        self.noise_initializer = noise_initializer

        self._delayed_ex_queue = {}
        self._delayed_in_queue = {}

    @staticmethod
    def _to_numpy(x):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x), dtype=dftype)

    @staticmethod
    def _to_numpy_ms(x):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x / u.ms), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @staticmethod
    def _to_int_scalar(x, name: str):
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(x), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return int(arr[0])

    def _input(self, h, g):
        return g * h

    def _mult_coupling_ex(self, rate, g_ex, theta_ex):
        return g_ex * (theta_ex - rate)

    def _mult_coupling_in(self, rate, g_in, theta_in):
        return g_in * (theta_in + rate)

    @staticmethod
    def _coerce_events(events):
        if events is None:
            return []
        if isinstance(events, dict):
            return [events]
        if isinstance(events, tuple):
            if len(events) == 0:
                return []
            if isinstance(events[0], (dict, tuple, list)):
                return list(events)
            if len(events) in (2, 3, 4):
                return [events]
        if isinstance(events, list):
            if len(events) == 0:
                return []
            if isinstance(events[0], (dict, tuple, list)):
                return events
            if len(events) in (2, 3, 4):
                return [tuple(events)]
        return [events]

    def _parse_event(self, ev, default_delay_steps: int) -> Tuple[np.ndarray, np.ndarray, int]:
        if isinstance(ev, dict):
            rate = ev.get('rate', ev.get('coeff', ev.get('value', 0.0)))
            weight = ev.get('weight', 1.0)
            multiplicity = ev.get('multiplicity', 1.0)
            delay_steps = ev.get('delay_steps', ev.get('delay', default_delay_steps))
            dftype = brainstate.environ.dftype()
            weight_sign = np.asarray(u.math.asarray(weight), dtype=dftype) >= 0.0
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
            weight_sign = np.asarray(u.math.asarray(weight), dtype=dftype) >= 0.0
        else:
            rate = ev
            weight = 1.0
            multiplicity = 1.0
            delay_steps = default_delay_steps
            weighted_value = self._to_numpy(rate)
            ex = np.where(weighted_value >= 0.0, weighted_value, 0.0)
            inh = np.where(weighted_value < 0.0, weighted_value, 0.0)
            return ex, inh, self._to_int_scalar(delay_steps, name='delay_steps')

        weighted_value = self._to_numpy(rate) * self._to_numpy(weight) * self._to_numpy(multiplicity)
        ex = np.where(weight_sign, weighted_value, 0.0)
        inh = np.where(weight_sign, 0.0, weighted_value)
        return ex, inh, self._to_int_scalar(delay_steps, name='delay_steps')

    @staticmethod
    def _queue_add(queue: dict, step_idx: int, value: np.ndarray):
        if step_idx in queue:
            queue[step_idx] = queue[step_idx] + value
        else:
            dftype = brainstate.environ.dftype()
            queue[step_idx] = np.array(value, dtype=dftype, copy=True)

    def _drain_delayed_queue(self, step_idx: int, state_shape):
        ex = self._delayed_ex_queue.pop(step_idx, None)
        inh = self._delayed_in_queue.pop(step_idx, None)
        if ex is None:
            dftype = brainstate.environ.dftype()
            ex = np.zeros(state_shape, dtype=dftype)
        else:
            ex = np.array(self._broadcast_to_state(np.asarray(ex, dtype=dftype), state_shape), copy=True)
        if inh is None:
            inh = np.zeros(state_shape, dtype=dftype)
        else:
            inh = np.array(self._broadcast_to_state(np.asarray(inh, dtype=dftype), state_shape), copy=True)
        return ex, inh

    def _accumulate_instant_events(self, events, state_shape):
        dftype = brainstate.environ.dftype()
        ex = np.zeros(state_shape, dtype=dftype)
        inh = np.zeros(state_shape, dtype=dftype)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._parse_event(ev, default_delay_steps=0)
            if delay_steps != 0:
                raise ValueError('instant_rate_events must not specify non-zero delay_steps.')
            ex += self._broadcast_to_state(ex_i, state_shape)
            inh += self._broadcast_to_state(inh_i, state_shape)
        return ex, inh

    def _schedule_delayed_events(self, events, step_idx: int, state_shape):
        dftype = brainstate.environ.dftype()
        ex_now = np.zeros(state_shape, dtype=dftype)
        inh_now = np.zeros(state_shape, dtype=dftype)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._parse_event(ev, default_delay_steps=1)
            if delay_steps < 0:
                raise ValueError('delay_steps for delayed_rate_events must be >= 0.')
            ex_i = self._broadcast_to_state(ex_i, state_shape)
            inh_i = self._broadcast_to_state(inh_i, state_shape)
            if delay_steps == 0:
                ex_now += ex_i
                inh_now += inh_i
            else:
                target_step = step_idx + delay_steps
                self._queue_add(self._delayed_ex_queue, target_step, ex_i)
                self._queue_add(self._delayed_in_queue, target_step, inh_i)
        return ex_now, inh_now

    def _common_inputs(self, x, instant_rate_events, delayed_rate_events):
        state_shape = self.rate.value.shape
        ditype = brainstate.environ.ditype()
        step_idx = int(np.asarray(self._step_count.value, dtype=ditype).reshape(-1)[0])

        delayed_ex, delayed_in = self._drain_delayed_queue(step_idx, state_shape)
        delayed_ex_now, delayed_in_now = self._schedule_delayed_events(
            delayed_rate_events,
            step_idx=step_idx,
            state_shape=state_shape,
        )
        delayed_ex = delayed_ex + delayed_ex_now
        delayed_in = delayed_in + delayed_in_now

        instant_ex, instant_in = self._accumulate_instant_events(instant_rate_events, state_shape)

        delta_input = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0.0)), state_shape)
        instant_ex += np.where(delta_input > 0.0, delta_input, 0.0)
        instant_in += np.where(delta_input < 0.0, delta_input, 0.0)

        mu_ext = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.rate.value)), state_shape)

        return state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext

    def _common_parameters(self, state_shape):
        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        sigma = self._broadcast_to_state(self._to_numpy(self.sigma), state_shape)
        mu = self._broadcast_to_state(self._to_numpy(self.mu), state_shape)
        g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
        g_ex = self._broadcast_to_state(self._to_numpy(self.g_ex), state_shape)
        g_in = self._broadcast_to_state(self._to_numpy(self.g_in), state_shape)
        theta_ex = self._broadcast_to_state(self._to_numpy(self.theta_ex), state_shape)
        theta_in = self._broadcast_to_state(self._to_numpy(self.theta_in), state_shape)
        return tau, sigma, mu, g, g_ex, g_in, theta_ex, theta_in


class lin_rate_ipn(_lin_rate_base):
    r"""NEST-compatible ``lin_rate_ipn`` linear rate neuron with input noise.

    Description
    -----------

    ``lin_rate_ipn`` implements NEST's linear rate neuron with **input noise**:

    .. math::

       \tau\, dX(t) =
       \left[-\lambda X(t) + \mu + \phi(\cdot)\right] dt
       + \left[\sqrt{\tau}\,\sigma\right] dW(t),

    where :math:`\phi(h)=g\,h`.

    The model supports:

    - additive mean drive ``mu`` (plus optional runtime input ``x``),
    - Gaussian input noise (``sigma``),
    - optional multiplicative coupling,
    - linear/nonlinear summation mode (``linear_summation``),
    - optional output rectification (``rectify_output``).

    **Update ordering (matching NEST ``rate_neuron_ipn``)**

    For each simulation step:

    1. Compute noise sample ``noise = sigma * xi``.
    2. Propagate intrinsic dynamics with stochastic exponential Euler
       (or Euler-Maruyama when ``lambda=0``).
    3. Read delayed and instantaneous rate-event buffers.
    4. Apply linear input nonlinearity and optional multiplicative coupling.
    5. Apply output rectification (if enabled).
    6. Store outputs analogous to NEST events:
       ``delayed_rate`` (pre-update rate), ``instant_rate`` (post-update rate).

    Parameters
    ----------
    in_size : Size
        Population shape.
    tau : Quantity[ms], optional
        Time constant of rate dynamics. Default ``10 ms``.
    lambda\_ : float, optional
        Passive decay rate :math:`\lambda`. Default ``1.0``.
    sigma : float, optional
        Input noise scale. Default ``1.0``.
    mu : float, optional
        Mean drive. Default ``0.0``.
    g : float, optional
        Input gain :math:`g`. Default ``1.0``.
    mult_coupling : bool, optional
        Enable multiplicative coupling. Default ``False``.
    g_ex, g_in, theta_ex, theta_in : float, optional
        Parameters of multiplicative coupling factors
        ``g_ex * (theta_ex - rate)`` and ``g_in * (theta_in + rate)``.
    linear_summation : bool, optional
        If ``True`` apply input nonlinearity to summed input;
        if ``False`` to each input branch before coupling.
        For linear nonlinearity both are mathematically equivalent.
    rectify_rate : float, optional
        Lower bound used when ``rectify_output=True``. Default ``0.0``.
    rectify_output : bool, optional
        If ``True`` clamp output rate to ``>= rectify_rate``.
    rate_initializer : Callable, optional
        Initializer for ``rate``. Default ``Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for ``noise``. Default ``Constant(0.0)``.
    name : str, optional
        Module name.

    Notes
    -----
    Runtime events:

    - ``instant_rate_events`` are applied in the current step.
    - ``delayed_rate_events`` are scheduled by integer ``delay_steps``:
      value ``1`` means next step, ``2`` means two steps later, etc.
    - Event format can be dict or tuple:
      ``(rate, weight)``, ``(rate, weight, delay_steps)``,
      ``(rate, weight, delay_steps, multiplicity)``.
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
        g_ex: ArrayLike = 1.0,
        g_in: ArrayLike = 1.0,
        theta_ex: ArrayLike = 0.0,
        theta_in: ArrayLike = 0.0,
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
            g_ex=g_ex,
            g_in=g_in,
            theta_ex=theta_ex,
            theta_in=theta_in,
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
        return ['rate', 'noise']

    @property
    def receptor_types(self):
        return {'RATE': 0}

    def _validate_parameters(self):
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

    def init_state(self, batch_size: int = None, **kwargs):
        rate = braintools.init.param(self.rate_initializer, self.varshape, batch_size)
        noise = braintools.init.param(self.noise_initializer, self.varshape, batch_size)
        rate_np = self._to_numpy(rate)
        noise_np = self._to_numpy(noise)

        self.rate = brainstate.ShortTermState(rate_np)
        self.noise = brainstate.ShortTermState(noise_np)
        dftype = brainstate.environ.dftype()
        self.instant_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))
        ditype = brainstate.environ.ditype()
        self._step_count = brainstate.ShortTermState(np.asarray(0, dtype=ditype))

        self._delayed_ex_queue = {}
        self._delayed_in_queue = {}

    def update(self, x=0.0, instant_rate_events=None, delayed_rate_events=None, noise=None):
        h = float(u.math.asarray(brainstate.environ.get_dt() / u.ms))

        state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext = self._common_inputs(
            x=x,
            instant_rate_events=instant_rate_events,
            delayed_rate_events=delayed_rate_events,
        )
        tau, sigma, mu, g, g_ex, g_in, theta_ex, theta_in = self._common_parameters(state_shape)
        lambda_ = self._broadcast_to_state(self._to_numpy(self.lambda_), state_shape)
        rectify_rate = self._broadcast_to_state(self._to_numpy(self.rectify_rate), state_shape)

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
            H_ex = self._mult_coupling_ex(rate_prev, g_ex, theta_ex)
            H_in = self._mult_coupling_in(rate_prev, g_in, theta_in)

        if self.linear_summation:
            if self.mult_coupling:
                rate_new += P2 * H_ex * self._input(delayed_ex + instant_ex, g)
                rate_new += P2 * H_in * self._input(delayed_in + instant_in, g)
            else:
                rate_new += P2 * self._input(delayed_ex + instant_ex + delayed_in + instant_in, g)
        else:
            rate_new += P2 * H_ex * self._input(delayed_ex + instant_ex, g)
            rate_new += P2 * H_in * self._input(delayed_in + instant_in, g)

        if self.rectify_output:
            rate_new = np.where(rate_new < rectify_rate, rectify_rate, rate_new)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.delayed_rate.value = rate_prev
        self.instant_rate.value = rate_new
        ditype = brainstate.environ.ditype()
        self._step_count.value = np.asarray(step_idx + 1, dtype=ditype)
        return rate_new


class lin_rate_opn(_lin_rate_base):
    r"""NEST-compatible ``lin_rate_opn`` linear rate neuron with output noise.

    Description
    -----------

    ``lin_rate_opn`` implements NEST's linear rate neuron with **output noise**:

    .. math::

       \tau \frac{dX(t)}{dt} = -X(t) + \mu + \phi(\cdot), \qquad
       X_\mathrm{noisy}(t)=X(t)+\sqrt{\frac{\tau}{h}}\sigma\xi(t)

    with :math:`\phi(h)=g\,h` and piecewise-constant Gaussian noise.

    **Update ordering (matching NEST ``rate_neuron_opn``)**

    For each simulation step:

    1. Draw ``noise = sigma * xi`` and build ``noisy_rate`` from current rate.
    2. Propagate deterministic intrinsic dynamics.
    3. Read delayed and instantaneous rate-event buffers.
    4. Apply linear input nonlinearity and optional multiplicative coupling.
    5. Store outputs analogous to NEST events:
       both ``delayed_rate`` and ``instant_rate`` carry ``noisy_rate``.

    Parameters
    ----------
    Same as :class:`lin_rate_ipn`, except:

    - no ``lambda_`` parameter (fixed leak form),
    - no output rectification parameters.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau: ArrayLike = 10.0 * u.ms,
        sigma: ArrayLike = 1.0,
        mu: ArrayLike = 0.0,
        g: ArrayLike = 1.0,
        mult_coupling: bool = False,
        g_ex: ArrayLike = 1.0,
        g_in: ArrayLike = 1.0,
        theta_ex: ArrayLike = 0.0,
        theta_in: ArrayLike = 0.0,
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
            g_ex=g_ex,
            g_in=g_in,
            theta_ex=theta_ex,
            theta_in=theta_in,
            linear_summation=linear_summation,
            rate_initializer=rate_initializer,
            noise_initializer=noise_initializer,
            name=name,
        )
        self.noisy_rate_initializer = noisy_rate_initializer
        self._validate_parameters()

    @property
    def recordables(self):
        return ['rate', 'noise', 'noisy_rate']

    @property
    def receptor_types(self):
        return {'RATE': 0}

    def _validate_parameters(self):
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.tau, self.sigma)):
            return
        if np.any(self.tau <= 0.0 * u.ms):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self.sigma < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')

    def init_state(self, batch_size: int = None, **kwargs):
        rate = braintools.init.param(self.rate_initializer, self.varshape, batch_size)
        noise = braintools.init.param(self.noise_initializer, self.varshape, batch_size)
        noisy_rate = braintools.init.param(self.noisy_rate_initializer, self.varshape, batch_size)
        rate_np = self._to_numpy(rate)
        noise_np = self._to_numpy(noise)
        noisy_rate_np = self._to_numpy(noisy_rate)

        self.rate = brainstate.ShortTermState(rate_np)
        self.noise = brainstate.ShortTermState(noise_np)
        self.noisy_rate = brainstate.ShortTermState(noisy_rate_np)
        dftype = brainstate.environ.dftype()
        self.instant_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=dftype, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=dftype, copy=True))
        ditype = brainstate.environ.ditype()
        self._step_count = brainstate.ShortTermState(np.asarray(0, dtype=ditype))

        self._delayed_ex_queue = {}
        self._delayed_in_queue = {}

    def update(self, x=0.0, instant_rate_events=None, delayed_rate_events=None, noise=None):
        h = float(u.math.asarray(brainstate.environ.get_dt() / u.ms))

        state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext = self._common_inputs(
            x=x,
            instant_rate_events=instant_rate_events,
            delayed_rate_events=delayed_rate_events,
        )
        tau, sigma, mu, g, g_ex, g_in, theta_ex, theta_in = self._common_parameters(state_shape)

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
            H_ex = self._mult_coupling_ex(noisy_rate, g_ex, theta_ex)
            H_in = self._mult_coupling_in(noisy_rate, g_in, theta_in)

        if self.linear_summation:
            if self.mult_coupling:
                rate_new += P2 * H_ex * self._input(delayed_ex + instant_ex, g)
                rate_new += P2 * H_in * self._input(delayed_in + instant_in, g)
            else:
                rate_new += P2 * self._input(delayed_ex + instant_ex + delayed_in + instant_in, g)
        else:
            rate_new += P2 * H_ex * self._input(delayed_ex + instant_ex, g)
            rate_new += P2 * H_in * self._input(delayed_in + instant_in, g)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.noisy_rate.value = noisy_rate
        self.delayed_rate.value = noisy_rate
        self.instant_rate.value = noisy_rate
        ditype = brainstate.environ.ditype()
        self._step_count.value = np.asarray(step_idx + 1, dtype=ditype)
        return rate_new
