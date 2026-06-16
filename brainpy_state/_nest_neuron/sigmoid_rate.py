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
from brainpy_state._nest_base._utils import is_tracer

__all__ = [
    'sigmoid_rate_ipn',
]


class _sigmoid_rate_base(_lin_rate_base):
    __module__ = 'brainpy.state'

    #: φ(h) = g / (1 + exp(−β(h − θ))) is fixed by the gain, slope and threshold;
    #: these identify it for the ``linear_summation=False`` homogeneity guard.
    _phi_param_names = ('g', 'beta', 'theta')

    def _activation(self, h):
        """Logistic gain ``φ(h) = g / (1 + exp(−β(h − θ)))`` (JAX; reads ``self``)."""
        g = u.get_mantissa(self.g)
        beta = u.get_mantissa(self.beta)
        theta = u.get_mantissa(self.theta)
        return g / (1.0 + jnp.exp(-beta * (h - theta)))

    def _mult_factors(self, rate):
        one = jnp.ones_like(rate)
        return one, one

    def _common_parameters_sigmoid(self, state_shape):
        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        sigma = self._broadcast_to_state(self._to_numpy(self.sigma), state_shape)
        mu = self._broadcast_to_state(self._to_numpy(self.mu), state_shape)
        g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
        beta = self._broadcast_to_state(self._to_numpy(self.beta), state_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.theta), state_shape)
        return tau, sigma, mu, g, beta, theta


class sigmoid_rate_ipn(_sigmoid_rate_base):
    r"""NEST-compatible ``sigmoid_rate_ipn`` nonlinear rate neuron with input noise.

    Description
    -----------

    ``sigmoid_rate_ipn`` implements NEST's ``sigmoid_rate_ipn`` model:

    .. math::

       \tau\,dX(t)=
       \left[-\lambda X(t)+\mu+\phi(\cdot)\right]dt
       +\left[\sqrt{\tau}\,\sigma\right]dW(t),

    where the gain function is

    .. math::

       \phi(h)=\frac{g}{1+\exp[-\beta(h-\theta)]}.

    This model corresponds to NEST's input-noise rate neuron template
    instantiated with ``sigmoid_rate`` nonlinearity. Multiplicative coupling
    factors are fixed to one for this model (the flag is kept for compatibility).

    **Update ordering (matching NEST ``rate_neuron_ipn`` with sigmoid nonlinearity)**

    Per simulation step:

    1. Store outgoing delayed value as current ``rate``.
    2. Draw ``noise = sigma * xi``.
    3. Propagate intrinsic dynamics with stochastic exponential Euler
       (Euler-Maruyama for ``lambda=0``).
    4. Read delayed and instantaneous buffers.
    5. Apply input contributions:
       - ``linear_summation=True``: apply sigmoid to branch sums.
       - ``linear_summation=False``: apply sigmoid per event before summation.
    6. Apply rectification when ``rectify_output=True``.
    7. Store outgoing instantaneous value as updated ``rate``.

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
        Gain (amplitude) of the sigmoid nonlinearity. Default ``1.0``.
    beta : float, optional
        Slope parameter of sigmoid nonlinearity. Default ``1.0``.
    theta : float, optional
        Threshold (horizontal shift) of sigmoid nonlinearity. Default ``0.0``.
    mult_coupling : bool, optional
        Kept for NEST compatibility. For ``sigmoid_rate`` this switch has no
        effect because multiplicative coupling factors are identically 1.
    linear_summation : bool, optional
        If ``True`` apply sigmoid to summed branch inputs; if ``False``
        apply sigmoid to each event before weighted summation.
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
        beta: ArrayLike = 1.0,
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
        self.beta = braintools.init.param(beta, self.varshape)
        self.theta = braintools.init.param(theta, self.varshape)
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

    def init_state(self, **kwargs):
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
        h = float(u.get_mantissa(brainstate.environ.get_dt() / u.ms))
        dftype = brainstate.environ.dftype()
        state_shape = self.rate.value.shape

        tau, sigma, mu, g, beta, theta = self._common_parameters_sigmoid(state_shape)
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
