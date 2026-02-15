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
    r"""NEST-compatible ``threshold_lin_rate_ipn`` with threshold-linear gain.

    Description
    -----------

    ``threshold_lin_rate_ipn`` implements NEST's input-noise threshold-linear
    rate neuron:

    .. math::

       \tau\,dX(t)=
       \left[-\lambda X(t)+\mu+\phi(\cdot)\right]dt
       +\left[\sqrt{\tau}\,\sigma\right]dW(t),

    with gain function

    .. math::

       \phi(h)=\min(\max(g(h-\theta),0),\alpha).

    This matches NEST's ``threshold_lin_rate`` nonlinearity class where
    multiplicative-coupling factors are constant one, so ``mult_coupling`` is
    accepted for API compatibility but has no effect on dynamics.

    Update ordering (matching NEST ``rate_neuron_ipn``)
    ...................................................

    Per simulation step:

    1. Store outgoing delayed value as current ``rate``.
    2. Draw ``noise = sigma * xi``.
    3. Propagate intrinsic dynamics with stochastic exponential Euler
       (Euler-Maruyama for ``lambda=0``).
    4. Read delayed and instantaneous event buffers.
    5. Add input contributions:
       - ``linear_summation=True``: apply threshold-linear gain to branch sums.
       - ``linear_summation=False``: apply threshold-linear gain per event before summation.
    6. Apply rectification if ``rectify_output=True``.
    7. Store outgoing instantaneous value as updated ``rate``.

    Parameters
    ----------
    in_size : Size
        Population shape.
    tau : Quantity[ms], optional
        Time constant of rate dynamics. Default ``10 ms``.
    lambda_ : float, optional
        Passive decay rate :math:`\lambda`. Default ``1.0``.
    sigma : float, optional
        Input noise scale. Default ``1.0``.
    mu : float, optional
        Mean drive. Default ``0.0``.
    g : float, optional
        Gain slope before clipping. Default ``1.0``.
    theta : float, optional
        Activation threshold. Default ``0.0``.
    alpha : float, optional
        Saturation threshold. Default ``+inf``.
    mult_coupling : bool, optional
        Compatibility flag; no effect for this model.
    linear_summation : bool, optional
        If ``True`` apply gain function to summed branch inputs; if ``False``
        apply gain function per event before weighted summation.
    rectify_rate : float, optional
        Lower bound when ``rectify_output=True``. Default ``0.0``.
    rectify_output : bool, optional
        If ``True`` clamp updated rate to ``>= rectify_rate``.
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
    - ``delayed_rate_events`` use integer ``delay_steps``.
    - Event format supports dict or tuple:
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
        return ['rate', 'noise']

    @property
    def receptor_types(self):
        return {'RATE': 0}

    def _validate_parameters(self):
        if np.any(self._to_numpy_ms(self.tau) <= 0.0):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self._to_numpy(self.lambda_) < 0.0):
            raise ValueError('Passive decay rate lambda must be >= 0.')
        if np.any(self._to_numpy(self.sigma) < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')
        if np.any(self._to_numpy(self.rectify_rate) < 0.0):
            raise ValueError('Rectifying rate must be >= 0.')

    def init_state(self, batch_size: int = None, **kwargs):
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
    r"""NEST-compatible ``threshold_lin_rate_opn`` with threshold-linear gain.

    Description
    -----------

    ``threshold_lin_rate_opn`` implements NEST's output-noise threshold-linear
    rate neuron:

    .. math::

       \tau\frac{dX(t)}{dt}=-X(t)+\mu+\phi(\cdot), \qquad
       X_\mathrm{noisy}(t)=X(t)+\sqrt{\frac{\tau}{h}}\sigma\xi(t),

    where

    .. math::

       \phi(h)=\min(\max(g(h-\theta),0),\alpha).

    Update ordering (matching NEST ``rate_neuron_opn``)
    ...................................................

    Per simulation step:

    1. Draw ``noise = sigma * xi``.
    2. Build ``noisy_rate`` from current ``rate`` and store as delayed output.
    3. Propagate deterministic intrinsic dynamics.
    4. Add input contributions using threshold-linear gain.
    5. Store ``noisy_rate`` as instantaneous output.

    Parameters
    ----------
    in_size : Size
        Population shape.
    tau : Quantity[ms], optional
        Time constant of rate dynamics. Default ``10 ms``.
    sigma : float, optional
        Output noise scale. Default ``1.0``.
    mu : float, optional
        Mean drive. Default ``0.0``.
    g : float, optional
        Gain slope before clipping. Default ``1.0``.
    theta : float, optional
        Activation threshold. Default ``0.0``.
    alpha : float, optional
        Saturation threshold. Default ``+inf``.
    mult_coupling : bool, optional
        Compatibility flag; no effect for this model.
    linear_summation : bool, optional
        If ``True`` apply gain function to summed branch inputs; if ``False``
        apply gain function per event before weighted summation.
    rate_initializer : Callable, optional
        Initializer for ``rate``. Default ``Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for ``noise``. Default ``Constant(0.0)``.
    noisy_rate_initializer : Callable, optional
        Initializer for ``noisy_rate``. Default ``Constant(0.0)``.
    name : str, optional
        Module name.

    Notes
    -----
    Runtime event formats are identical to :class:`threshold_lin_rate_ipn`.
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
        return ['rate', 'noise', 'noisy_rate']

    @property
    def receptor_types(self):
        return {'RATE': 0}

    def _validate_parameters(self):
        if np.any(self._to_numpy_ms(self.tau) <= 0.0):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self._to_numpy(self.sigma) < 0.0):
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
        self.instant_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=np.float64, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=np.float64, copy=True))
        self._step_count = brainstate.ShortTermState(np.asarray(0, dtype=np.int64))

        self._delayed_ex_queue = {}
        self._delayed_in_queue = {}

    def update(self, x=0.0, instant_rate_events=None, delayed_rate_events=None, noise=None):
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
