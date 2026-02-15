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
    'rate_neuron_opn',
]


class rate_neuron_opn(_lin_rate_base):
    r"""NEST-compatible ``rate_neuron_opn`` output-noise rate-neuron template.

    Description
    -----------

    ``rate_neuron_opn`` implements the NEST template model
    ``rate_neuron_opn<TNonlinearities>`` with output noise:

    .. math::

       \tau \frac{dX(t)}{dt}
       = -X(t) + \mu + I_\mathrm{net}(t),

    .. math::

       X_\mathrm{noisy}(t)
       = X(t) + \sqrt{\frac{\tau}{h}}\,\sigma\,\xi(t),

    where :math:`\xi(t)` is Gaussian white noise approximated as piecewise
    constant over simulation step size :math:`h`.

    The network input :math:`I_\mathrm{net}(t)` follows NEST's template logic:

    - ``linear_summation=True``: nonlinearity is applied during the update to
      summed branch inputs.
    - ``linear_summation=False``: nonlinearity is applied per incoming event in
      event buffering, then summed in the update.
    - ``mult_coupling=True`` enables excitatory/inhibitory branch factors
      evaluated at ``noisy_rate`` (matching NEST ``rate_neuron_opn_impl.h``).

    With default callables, this corresponds to NEST ``lin_rate_opn``:

    - ``input(h) = g * h``
    - ``mult_coupling_ex(rate) = g_ex * (theta_ex - rate)``
    - ``mult_coupling_in(rate) = g_in * (theta_in + rate)``

    Update ordering (matching NEST ``rate_neuron_opn_impl.h``)
    ..........................................................

    Per simulation step:

    1. Draw ``noise = sigma * xi``.
    2. Compute ``noisy_rate = rate + sqrt(tau / h) * noise``.
    3. Propagate deterministic intrinsic dynamics:
       ``rate <- P1 * rate + P2 * (mu + mu_ext)``.
    4. Read delayed and instantaneous event buffers.
    5. Add network contribution according to ``linear_summation`` and
       ``mult_coupling``.
    6. Expose ``noisy_rate`` as outgoing instantaneous and delayed event value
       for this step (consistent with single-step min-delay behavior).

    Parameters
    ----------
    in_size : Size
        Population shape.
    tau : Quantity[ms], optional
        Time constant. Default ``10 ms``.
    sigma : float, optional
        Output-noise scale. Default ``1.0``.
    mu : float, optional
        Mean drive. Default ``0.0``.
    g : float, optional
        Linear gain parameter used by the default input nonlinearity.
        Default ``1.0``.
    mult_coupling : bool, optional
        Enable multiplicative coupling. Default ``False``.
    g_ex, g_in, theta_ex, theta_in : float, optional
        Parameters used by default multiplicative-coupling callables.
    linear_summation : bool, optional
        NEST switch controlling where the input nonlinearity is applied.
        Default ``True``.
    input_nonlinearity : Callable, optional
        Custom input nonlinearity replacing ``input`` from NEST template.
        Callable signature can be ``f(h)`` or ``f(model, h)``.
    mult_coupling_ex_fn : Callable, optional
        Custom excitatory multiplicative coupling function. Signature can be
        ``f(rate)`` or ``f(model, rate)``.
    mult_coupling_in_fn : Callable, optional
        Custom inhibitory multiplicative coupling function. Signature can be
        ``f(rate)`` or ``f(model, rate)``.
    rate_initializer : Callable, optional
        Initializer for internal deterministic state ``rate``.
        Default ``Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for ``noise``. Default ``Constant(0.0)``.
    noisy_rate_initializer : Callable, optional
        Initializer for ``noisy_rate`` and event outputs.
        Default ``Constant(0.0)``.
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
    - For ``linear_summation=False``, event values are transformed by the input
      nonlinearity during buffering (matching NEST event handlers).
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
        input_nonlinearity: Callable | None = None,
        mult_coupling_ex_fn: Callable | None = None,
        mult_coupling_in_fn: Callable | None = None,
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

        self.input_nonlinearity = input_nonlinearity
        self.mult_coupling_ex_fn = mult_coupling_ex_fn
        self.mult_coupling_in_fn = mult_coupling_in_fn
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

    def _call_nl(self, fn: Callable, x: np.ndarray):
        try:
            return fn(self, x)
        except TypeError as first_error:
            try:
                return fn(x)
            except TypeError:
                raise first_error

    def _input_transform(self, h: np.ndarray, state_shape):
        h_np = self._broadcast_to_state(self._to_numpy(h), state_shape)
        if self.input_nonlinearity is None:
            g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
            return g * h_np
        y = self._call_nl(self.input_nonlinearity, h_np)
        return self._broadcast_to_state(self._to_numpy(y), state_shape)

    def _mult_ex_transform(self, rate: np.ndarray, state_shape):
        rate_np = self._broadcast_to_state(self._to_numpy(rate), state_shape)
        if self.mult_coupling_ex_fn is None:
            g_ex = self._broadcast_to_state(self._to_numpy(self.g_ex), state_shape)
            theta_ex = self._broadcast_to_state(self._to_numpy(self.theta_ex), state_shape)
            return g_ex * (theta_ex - rate_np)
        y = self._call_nl(self.mult_coupling_ex_fn, rate_np)
        return self._broadcast_to_state(self._to_numpy(y), state_shape)

    def _mult_in_transform(self, rate: np.ndarray, state_shape):
        rate_np = self._broadcast_to_state(self._to_numpy(rate), state_shape)
        if self.mult_coupling_in_fn is None:
            g_in = self._broadcast_to_state(self._to_numpy(self.g_in), state_shape)
            theta_in = self._broadcast_to_state(self._to_numpy(self.theta_in), state_shape)
            return g_in * (theta_in + rate_np)
        y = self._call_nl(self.mult_coupling_in_fn, rate_np)
        return self._broadcast_to_state(self._to_numpy(y), state_shape)

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

    def _event_to_ex_in(self, ev, default_delay_steps: int, state_shape):
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
            weighted_value = self._input_transform(rate_np, state_shape) * weight_np * multiplicity_np

        ex = np.where(weight_sign, weighted_value, 0.0)
        inh = np.where(weight_sign, 0.0, weighted_value)
        return ex, inh, delay_steps

    def _accumulate_instant_events(self, events, state_shape):
        ex = np.zeros(state_shape, dtype=np.float64)
        inh = np.zeros(state_shape, dtype=np.float64)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._event_to_ex_in(
                ev,
                default_delay_steps=0,
                state_shape=state_shape,
            )
            if delay_steps != 0:
                raise ValueError('instant_rate_events must not specify non-zero delay_steps.')
            ex += ex_i
            inh += inh_i
        return ex, inh

    def _schedule_delayed_events(self, events, step_idx: int, state_shape):
        ex_now = np.zeros(state_shape, dtype=np.float64)
        inh_now = np.zeros(state_shape, dtype=np.float64)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._event_to_ex_in(
                ev,
                default_delay_steps=1,
                state_shape=state_shape,
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

    def _common_inputs_template(self, x, instant_rate_events, delayed_rate_events):
        state_shape = self.rate.value.shape
        step_idx = int(np.asarray(self._step_count.value, dtype=np.int64).reshape(-1)[0])

        delayed_ex, delayed_in = self._drain_delayed_queue(step_idx, state_shape)
        delayed_ex_now, delayed_in_now = self._schedule_delayed_events(
            delayed_rate_events,
            step_idx=step_idx,
            state_shape=state_shape,
        )
        delayed_ex = delayed_ex + delayed_ex_now
        delayed_in = delayed_in + delayed_in_now

        instant_ex, instant_in = self._accumulate_instant_events(
            instant_rate_events,
            state_shape=state_shape,
        )

        delta_input = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0.0)), state_shape)
        instant_ex += np.where(delta_input > 0.0, delta_input, 0.0)
        instant_in += np.where(delta_input < 0.0, delta_input, 0.0)

        mu_ext = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.rate.value)), state_shape)

        return state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext

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

        state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext = self._common_inputs_template(
            x=x,
            instant_rate_events=instant_rate_events,
            delayed_rate_events=delayed_rate_events,
        )

        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        sigma = self._broadcast_to_state(self._to_numpy(self.sigma), state_shape)
        mu = self._broadcast_to_state(self._to_numpy(self.mu), state_shape)

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
            H_ex = self._mult_ex_transform(noisy_rate, state_shape)
            H_in = self._mult_in_transform(noisy_rate, state_shape)

        if self.linear_summation:
            if self.mult_coupling:
                rate_new += P2 * H_ex * self._input_transform(delayed_ex + instant_ex, state_shape)
                rate_new += P2 * H_in * self._input_transform(delayed_in + instant_in, state_shape)
            else:
                rate_new += P2 * self._input_transform(
                    delayed_ex + instant_ex + delayed_in + instant_in,
                    state_shape,
                )
        else:
            rate_new += P2 * H_ex * (delayed_ex + instant_ex)
            rate_new += P2 * H_in * (delayed_in + instant_in)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.noisy_rate.value = noisy_rate
        self.delayed_rate.value = noisy_rate
        self.instant_rate.value = noisy_rate
        self._step_count.value = np.asarray(step_idx + 1, dtype=np.int64)
        return rate_new
