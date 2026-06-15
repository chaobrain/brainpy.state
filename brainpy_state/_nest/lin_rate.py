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

from ._base import NESTNeuron
from ._utils import is_tracer, cond_any

__all__ = [
    'lin_rate_ipn',
    'lin_rate_opn',
]


class _lin_rate_base(NESTNeuron):
    """Shared JAX-native base for the NEST rate-neuron family.

    Rate neurons couple **continuously**, not by spikes: each step the
    presynaptic ``rate`` (a graded value) is emitted over a receptorless static
    connection and the postsynaptic neuron integrates the weighted sum
    ``h = Σ_pre weight·rate_pre`` as its coupling input. On the
    :class:`~brainpy_state.Simulator` substrate this is realized by the seam-(H)
    *continuous emission* path: the class declares ``_emission_continuous`` and
    ``_emission_attr`` so ``create()`` allocates an emission holder, phase-2
    captures the per-step ``rate`` into it, and the connection deposits
    ``weight·rate`` into the post's default delta channel (``comm='dense'``);
    the neuron reads it back with ``sum_delta_inputs``. The dynamics
    ``τ dX = (−λX + μ + φ(h)) dt + noise`` are integrated with the exact
    exponential-Euler propagators in pure JAX, so the whole simulation lowers
    into one compiled ``for_loop``.
    """

    __module__ = 'brainpy.state'

    #: Rate neurons emit a continuous graded value (the per-step ``rate``) rather
    #: than a binary spike. The Simulator routes ``weight·rate`` into the post's
    #: default delta channel each step (receptorless seam-(H) coupling).
    _emission_continuous = True

    #: Whether ``mult_coupling=True`` has a real effect. Only models with genuine
    #: excitatory/inhibitory coupling factors ``H_ex=g_ex(θ_ex−r)`` /
    #: ``H_in=g_in(θ_in+r)`` (``lin_rate``, the ``rate_neuron`` template) override
    #: this to ``True``; for the fixed-nonlinearity models (``gauss``/``sigmoid``/
    #: ``tanh``/``threshold_lin``/``sigmoid_gg``) the factors are identically 1, so
    #: ``mult_coupling`` is a no-op and the dual-channel split is skipped.
    _supports_mult_coupling = False

    #: φ-defining parameter names compared by :pyattr:`_phi_signature` for the
    #: ``linear_summation=False`` homogeneity guard. The default linear gain φ(h)=g·h
    #: is identified by ``g``; richer nonlinearities extend this.
    _phi_param_names = ('g',)

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

        # Seam-(H) continuous emission. ``linear_summation=True`` emits the raw
        # ``rate`` (the receiver applies φ to the summed input); ``False`` emits
        # ``phi_rate = φ(rate)`` so the receiver integrates ``Σ w·φ(r)`` (exact for
        # a homogeneous φ). For a linear gain the two coincide. ``_emission_attr``
        # must be known at ``create()`` time (before ``init_state``), so it is
        # pinned here; the ``phi_rate`` State itself is allocated in ``init_state``.
        self._emission_attr = 'rate' if self.linear_summation else 'phi_rate'

    @staticmethod
    def _to_numpy(x):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.get_mantissa(x), dtype=dftype)

    @staticmethod
    def _to_numpy_ms(x):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.get_mantissa(x / u.ms), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _activation(self, h):
        """The input gain φ(h) on the summed coupling input (JAX; reads ``self``).

        Linear default ``φ(h) = g·h``. Nonlinear subclasses override this with a
        JAX (``jnp``/``u.math``) gain reading their own parameters, so it lowers
        into the compiled ``for_loop`` (the coupling input ``h`` is a tracer).
        """
        return u.get_mantissa(self.g) * h

    def _mult_factors(self, rate):
        """Multiplicative-coupling factors ``(H_ex, H_in)`` at ``rate`` (JAX).

        Default is the linear-rate form ``H_ex = g_ex·(θ_ex − rate)`` and
        ``H_in = g_in·(θ_in + rate)``; models whose coupling is trivially unity
        (``gauss``/``sigmoid``/``tanh``/``threshold_lin``) override to return ones.
        """
        g_ex = u.get_mantissa(self.g_ex)
        g_in = u.get_mantissa(self.g_in)
        theta_ex = u.get_mantissa(self.theta_ex)
        theta_in = u.get_mantissa(self.theta_in)
        return g_ex * (theta_ex - rate), g_in * (theta_in + rate)

    @property
    def _use_mult_coupling(self):
        """Whether dual-channel multiplicative coupling is active for this neuron.

        ``True`` only when the user requested ``mult_coupling`` **and** the model
        has genuine ``(H_ex, H_in)`` factors (:pyattr:`_supports_mult_coupling`).
        For the fixed-nonlinearity models the factors are identically one, so the
        request is silently a no-op and the single default channel is used — this
        keeps ``mult_coupling=True`` exactly equivalent to ``False`` for them.
        """
        return self.mult_coupling and self._supports_mult_coupling

    @property
    def _phi_signature(self):
        r"""Hashable identity of this neuron's input nonlinearity φ.

        Two rate neurons share a φ iff they are the same model class, agree on
        ``linear_summation``, and carry identical φ-defining gain parameters
        (:pyattr:`_phi_param_names`). A ``linear_summation=False`` rate connection
        emits the **sender's** ``φ(rate)`` but is integrated where the **receiver**
        would have applied **its** φ; the two coincide only for a homogeneous φ, so
        the Simulator compares signatures at ``connect()`` and refuses a mismatch.

        Each parameter is reduced to its *set* of distinct values, so a scalar and
        a uniformly-filled population array compare equal regardless of size.
        """
        params = tuple(
            (name, tuple(sorted(set(
                np.asarray(u.get_mantissa(getattr(self, name))).reshape(-1).tolist()
            ))))
            for name in self._phi_param_names
        )
        return (type(self).__name__, bool(self.linear_summation), params)

    def _emission_state(self, rate_new):
        """The value emitted this step: ``rate`` (linear summation) or ``φ(rate)``."""
        return rate_new if self.linear_summation else self._activation(rate_new)

    def _alloc_phi_rate(self, rate_np):
        """Allocate the ``phi_rate`` emission State (call from ``init_state``).

        Only needed when ``linear_summation=False`` (the neuron emits ``φ(rate)``);
        a no-op otherwise (the ``rate`` State is emitted directly).
        """
        if not self.linear_summation:
            dftype = brainstate.environ.dftype()
            phi0 = np.asarray(u.get_mantissa(self._activation(jnp.asarray(rate_np))), dtype=dftype)
            self.phi_rate = brainstate.ShortTermState(phi0)

    def _store_phi_rate(self, rate_new):
        """Refresh the ``phi_rate`` emission State (call at the end of ``update``)."""
        if not self.linear_summation:
            self.phi_rate.value = self._activation(rate_new)

    def _read_coupling(self, x):
        """Read the per-step JAX coupling inputs (no host event queue).

        Returns the external mean drive ``mu_ext = sum_current_inputs(x, rate)``
        and the rate coupling delivered by the seam-(H) continuous-emission
        connections. Without ``mult_coupling`` a single summed default channel
        ``h = Σ_pre weight·rate_pre`` is read (the second value is ``None``); with
        ``mult_coupling`` the excitatory/inhibitory partial sums are read from the
        labelled ``'rate_ex'``/``'rate_in'`` channels (dual-channel deposit).

        Parameters
        ----------
        x : ArrayLike
            Optional runtime drive forwarded as the ``sum_current_inputs`` init.

        Returns
        -------
        mu_ext : jax.Array
            External mean drive (the summed current inputs).
        h_a : jax.Array
            The default summed rate channel ``h`` (``mult_coupling=False``) or the
            excitatory partial sum ``h_ex`` (``mult_coupling=True``).
        h_b : jax.Array or None
            ``None`` (``mult_coupling=False``) or the inhibitory partial sum
            ``h_in`` (``mult_coupling=True``).
        """
        rate_now = self.rate.value
        mu_ext = u.get_mantissa(self.sum_current_inputs(x, rate_now))
        if self._use_mult_coupling:
            h_ex = u.get_mantissa(self.sum_delta_inputs(0.0, label='rate_ex'))
            h_in = u.get_mantissa(self.sum_delta_inputs(0.0, label='rate_in'))
            return mu_ext, h_ex, h_in
        h = u.get_mantissa(self.sum_delta_inputs(0.0))
        return mu_ext, h, None

    def _coupling_increment(self, rate_for_H, h_a, h_b):
        """The coupling term added each step as ``rate_new += P2 * <this>``.

        Shared by every rate neuron's update; they differ only in ``_activation``
        (φ) and in ``rate_for_H`` (the rate the multiplicative-coupling factors
        are evaluated at: the pre-update rate for ipn, the noisy rate for opn).

        With ``linear_summation=True`` the input nonlinearity is applied to the
        summed channel, ``φ(h_a)``; with ``False`` the receiver integrates the
        already-transformed channel (the pre emitted ``φ(rate)``), so ``h_a`` is
        added directly. ``mult_coupling=False`` uses the single default channel
        (``h_a``); ``mult_coupling=True`` uses the labelled ex/in partial sums
        (``h_a=h_ex``, ``h_b=h_in``) scaled by ``H_ex``/``H_in``.
        """
        a = self._activation(h_a) if self.linear_summation else h_a
        if not self._use_mult_coupling:
            return a
        H_ex, H_in = self._mult_factors(rate_for_H)
        b = self._activation(h_b) if self.linear_summation else h_b
        return H_ex * a + H_in * b

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

    #: Linear rate neurons carry genuine ``(H_ex, H_in)`` factors, so
    #: ``mult_coupling`` splits the deposit into the ``'rate_ex'``/``'rate_in'``
    #: channels (spec §3.2).
    _supports_mult_coupling = True

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
        if cond_any(self.tau <= 0.0 * u.ms):
            raise ValueError('Time constant tau must be > 0.')
        if cond_any(self.lambda_ < 0.0):
            raise ValueError('Passive decay rate lambda must be >= 0.')
        if cond_any(self.sigma < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')
        if cond_any(self.rectify_rate < 0.0):
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

        tau, sigma, mu, g, g_ex, g_in, theta_ex, theta_in = self._common_parameters(state_shape)
        lambda_ = self._broadcast_to_state(self._to_numpy(self.lambda_), state_shape)
        rectify_rate = self._broadcast_to_state(self._to_numpy(self.rectify_rate), state_shape)

        rate_prev = jnp.broadcast_to(jnp.asarray(self.rate.value, dtype=dftype), state_shape)

        mu_ext, h_a, h_b = self._read_coupling(x)

        if noise is None:
            xi = brainstate.random.randn(*state_shape)
        else:
            xi = jnp.broadcast_to(jnp.asarray(noise, dtype=dftype), state_shape)
        noise_now = sigma * xi

        if cond_any(lambda_ > 0.0):
            P1 = np.exp(-lambda_ * h / tau)
            P2 = -np.expm1(-lambda_ * h / tau) / np.where(lambda_ == 0.0, 1.0, lambda_)
            input_noise_factor = np.sqrt(
                -0.5 * np.expm1(-2.0 * lambda_ * h / tau) / np.where(lambda_ == 0.0, 1.0, lambda_)
            )
            zero_lambda = lambda_ == 0.0
            if cond_any(zero_lambda):
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

    #: Linear rate neurons carry genuine ``(H_ex, H_in)`` factors, so
    #: ``mult_coupling`` splits the deposit into the ``'rate_ex'``/``'rate_in'``
    #: channels (spec §3.2).
    _supports_mult_coupling = True

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
        if cond_any(self.tau <= 0.0 * u.ms):
            raise ValueError('Time constant tau must be > 0.')
        if cond_any(self.sigma < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')

    def init_state(self, **kwargs):
        rate = braintools.init.param(self.rate_initializer, self.varshape)
        noise = braintools.init.param(self.noise_initializer, self.varshape)
        noisy_rate = braintools.init.param(self.noisy_rate_initializer, self.varshape)
        rate_np = self._to_numpy(rate)
        noise_np = self._to_numpy(noise)
        noisy_rate_np = self._to_numpy(noisy_rate)

        self.rate = brainstate.ShortTermState(rate_np)
        self.noise = brainstate.ShortTermState(noise_np)
        self.noisy_rate = brainstate.ShortTermState(noisy_rate_np)
        dftype = brainstate.environ.dftype()
        self.instant_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=dftype, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=dftype, copy=True))
        self._alloc_phi_rate(rate_np)

    def update(self, x=0.0, noise=None):
        h = float(u.get_mantissa(brainstate.environ.get_dt() / u.ms))
        dftype = brainstate.environ.dftype()
        state_shape = self.rate.value.shape

        tau, sigma, mu, g, g_ex, g_in, theta_ex, theta_in = self._common_parameters(state_shape)

        rate_prev = jnp.broadcast_to(jnp.asarray(self.rate.value, dtype=dftype), state_shape)

        mu_ext, h_a, h_b = self._read_coupling(x)

        if noise is None:
            xi = brainstate.random.randn(*state_shape)
        else:
            xi = jnp.broadcast_to(jnp.asarray(noise, dtype=dftype), state_shape)
        noise_now = sigma * xi

        P1 = np.exp(-h / tau)
        P2 = -np.expm1(-h / tau)
        output_noise_factor = np.sqrt(tau / h)
        noisy_rate = rate_prev + output_noise_factor * noise_now

        mu_total = mu + mu_ext
        rate_new = P1 * rate_prev + P2 * mu_total
        # opn evaluates the multiplicative-coupling factors at the *noisy* rate.
        rate_new = rate_new + P2 * self._coupling_increment(noisy_rate, h_a, h_b)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.noisy_rate.value = noisy_rate
        self.delayed_rate.value = noisy_rate
        self.instant_rate.value = noisy_rate
        self._store_phi_rate(rate_new)
        return rate_new
