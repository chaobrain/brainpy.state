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
import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron
from ._utils import cond_any

__all__ = [
    'erfc_neuron',
]


class erfc_neuron(NESTNeuron):
    r"""Binary stochastic neuron with complementary error-function gain.

    Description
    -----------

    ``erfc_neuron`` re-implements NEST's binary neuron model of the same name.
    The neuron keeps a persistent synaptic input state :math:`h` and updates
    its binary output :math:`y \in \{0, 1\}` at Poisson-distributed update
    times with mean interval :math:`\tau_m`.

    **1. Gain function and state transition**

    At each scheduled update, the new binary state is sampled as

    .. math::

       y \leftarrow \mathbf{1}[U < g(h + c)], \quad U \sim \mathrm{Uniform}(0, 1),

    with gain function

    .. math::

       g(h) = \frac{1}{2}\,\mathrm{erfc}\!\left(-\frac{h - \theta}{\sqrt{2}\,\sigma}\right).

    This matches the NEST implementation in ``gainfunction_erfc::operator()``.
    The model corresponds to a McCulloch-Pitts threshold unit with additive
    Gaussian noise of standard deviation :math:`\sigma`.

    **2. Interpretation: threshold unit with Gaussian noise**

    The complementary error function gain arises from a threshold model with
    Gaussian noise. Suppose the neuron fires when :math:`h + \xi > \theta`,
    where :math:`\xi \sim \mathcal{N}(0, \sigma^2)`. The activation probability
    is then

    .. math::

       P(\text{fire}) = P(h + \xi > \theta)
                      = P\left(\frac{\xi}{\sigma} > \frac{\theta - h}{\sigma}\right)
                      = \frac{1}{2}\,\mathrm{erfc}\!\left(\frac{\theta - h}{\sqrt{2}\,\sigma}\right).

    This establishes the connection to the McCulloch-Pitts neuron with additive
    Gaussian noise.

    **3. Update order (NEST semantics)**

    Each simulation step follows the same ordering as NEST's
    ``binary_neuron::update()``:

    1. Accumulate delta inputs into persistent :math:`h`.
    2. Read current input :math:`c` for the present step.
    3. If ``t + dt > t_next`` (strict inequality), sample a new binary state
       from :math:`g(h+c)`.
    4. If an update happened, advance ``t_next`` by ``Exp(1) * tau_m``.

    As in NEST, probabilities are not explicitly clipped before comparing
    against uniform random numbers. The comparison with a uniform random number
    implies effective clipping: gain values below 0 yield probability 0, values
    above 1 yield probability 1.

    **4. Assumptions, constraints, and computational implications**

    - The model assumes unit-compatible parameters and broadcast-compatible
      shapes against ``self.varshape``.
    - ``tau_m`` must be strictly positive (enforced in :meth:`__init__`).
    - Per-step compute is :math:`O(\prod \mathrm{varshape})` with vectorized
      elementwise operations plus random sampling overhead.
    - Stochastic update times are sampled from an exponential distribution, so
      the inter-update intervals are memoryless (Poisson process property).
    - When ``stochastic_update=False``, the model updates at every time step
      but retains stochastic state transitions according to the same gain
      function.

    Parameters
    ----------
    in_size : Size
        Population shape specification. All neuron parameters are broadcast to
        ``self.varshape`` derived from ``in_size``.
    tau_m : ArrayLike, optional
        Mean inter-update interval :math:`\tau_m` in ms; scalar or array
        broadcastable to ``self.varshape``. Must be strictly positive. Default
        is ``10.0 * u.ms``.
    theta : ArrayLike, optional
        Threshold :math:`\theta` in mV; scalar or array broadcastable to
        ``self.varshape``. Default is ``0.0 * u.mV``.
    sigma : ArrayLike, optional
        Gain/noise parameter :math:`\sigma` in mV; scalar or array broadcastable
        to ``self.varshape``. Larger values produce smoother gain transitions.
        Default is ``1.0 * u.mV``.
    y_initializer : Callable, optional
        Initializer for initial binary state ``y`` in :meth:`init_state`. Output
        should be float64 values (typically 0.0 or 1.0) shape-compatible with
        ``self.varshape`` (and optional batch prefix). Default is
        ``braintools.init.Constant(0.0)``.
    stochastic_update : bool, optional
        If ``True`` (default), use Poisson update scheduling as in NEST
        (updates occur at intervals sampled from ``Exp(tau_m)``). If ``False``,
        update each time step while retaining stochastic state sampling from
        the same gain function. Default is ``True``.
    rng_seed : int, optional
        Seed for internal random sampling (both for uniform and exponential
        random variables). Different seeds produce different random sequences.
        Default is ``0``.
    name : str or None, optional
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 20 28 14 16 35

       * - Parameter
         - Type / shape / unit
         - Default
         - Math symbol
         - Semantics
       * - ``in_size``
         - :class:`~brainstate.typing.Size`; scalar/tuple
         - required
         - --
         - Defines population/state shape ``self.varshape``.
       * - ``tau_m``
         - ArrayLike, broadcastable to ``self.varshape`` (ms)
         - ``10.0 * u.ms``
         - :math:`\tau_m`
         - Mean Poisson inter-update interval.
       * - ``theta``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``0.0 * u.mV``
         - :math:`\theta`
         - Activation threshold in gain function.
       * - ``sigma``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``1.0 * u.mV``
         - :math:`\sigma`
         - Noise standard deviation / gain slope parameter.
       * - ``y_initializer``
         - Callable
         - ``Constant(0.0)``
         - --
         - Initializes binary output state ``y``.
       * - ``stochastic_update``
         - bool
         - ``True``
         - --
         - Enables Poisson-timed updates vs. every-step updates.
       * - ``rng_seed``
         - int
         - ``0``
         - --
         - Random number generator seed.
       * - ``name``
         - str | None
         - ``None``
         - --
         - Optional node identifier.

    Raises
    ------
    ValueError
        If ``tau_m`` contains any non-positive values (checked in
        :meth:`__init__`), or if parameter initialization or broadcasting fails.
    TypeError
        If provided values are not compatible with expected units/types
        (ms, mV, or callable initializer).
    KeyError
        At runtime, if required simulation context entries (``t`` or ``dt``)
        are missing when :meth:`update` is called (only when
        ``stochastic_update=True``).
    AttributeError
        If :meth:`update` is called before :meth:`init_state` creates required
        state variables.

    Attributes
    ----------
    y : ShortTermState
        Binary output state (float64 values 0.0 or 1.0).
    h : ShortTermState
        Persistent summed synaptic input.
    t_next : ShortTermState
        Next stochastic update time (only if ``stochastic_update=True``).
    rng_key : ShortTermState
        JAX PRNGKey for random sampling (internal state).

    Notes
    -----
    - State variables are ``y``, ``h``, ``rng_key``, and optionally ``t_next``
      (when ``stochastic_update=True``).
    - In NEST, binary-neuron communication encodes state transitions using spike
      multiplicity (double spike for up-transition, single spike for
      down-transition). Here, equivalent effects are represented through delta
      inputs added to :math:`h`.
    - The gain function is evaluated at :math:`h + c`, where :math:`c` is the
      sum of current inputs for the present step.
    - Random sampling uses JAX's functional random number generation with state
      splitting for reproducibility and compatibility with JAX transformations.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.erfc_neuron(in_size=10, tau_m=5.0 * u.ms)
       ...     neu.init_state(batch_size=1)
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         out = neu.update(x=2.0 * u.mV)
       ...     _ = out.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.erfc_neuron(
       ...         in_size=(2, 3),
       ...         theta=1.0 * u.mV,
       ...         sigma=0.5 * u.mV,
       ...         stochastic_update=False
       ...     )
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = neu.update(x=1.5 * u.mV)

    References
    ----------
    .. [1] Ginzburg I, Sompolinsky H (1994). Theory of correlations in
           stochastic neural networks. PRE 50(4):3171.
           DOI: https://doi.org/10.1103/PhysRevE.50.3171
    .. [2] McCulloch W, Pitts W (1943). A logical calculus of the ideas
           immanent in nervous activity. Bulletin of Mathematical Biophysics,
           5:115-133. DOI: https://doi.org/10.1007/BF02478259
    .. [3] Morrison A, Diesmann M (2007). Maintaining causality in discrete
           time neuronal simulations. Lectures in Supercomputational
           Neuroscience. DOI: https://doi.org/10.1007/978-3-540-73159-7_10

    See Also
    --------
    ginzburg_neuron : Binary neuron with sigmoidal/affine gain function
    mcculloch_pitts_neuron : Binary neuron with hard threshold
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau_m: ArrayLike = 10. * u.ms,
        theta: ArrayLike = 0. * u.mV,
        sigma: ArrayLike = 1. * u.mV,
        y_initializer: Callable = braintools.init.Constant(0.0),
        stochastic_update: bool = True,
        rng_seed: int = 0,
        name: str = None,
    ):
        super().__init__(in_size, name=name)

        self.tau_m = braintools.init.param(tau_m, self.varshape)
        if cond_any(self.tau_m <= 0. * u.ms):
            raise ValueError('tau_m must be strictly positive.')

        self.theta = braintools.init.param(theta, self.varshape)
        self.sigma = braintools.init.param(sigma, self.varshape)
        self.y_initializer = y_initializer
        self.stochastic_update = stochastic_update
        self.rng_seed = int(rng_seed)

    def init_state(self, **kwargs):
        r"""Initialize binary state, input accumulator, and update timing.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Raises
        ------
        ValueError
            If initializer outputs cannot be broadcast to target state shape.
        TypeError
            If initializer values are incompatible with required numeric/unit
            conversions.
        """
        shape = self.varshape

        y = braintools.init.param(self.y_initializer, self.varshape)
        dftype = brainstate.environ.dftype()
        self.y = brainstate.ShortTermState(u.math.asarray(y, dtype=dftype))
        self.h = brainstate.ShortTermState(u.math.zeros(shape, dtype=dftype) * u.mV)
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))

        if self.stochastic_update:
            exp0 = self._sample_exponential(self.y.value.shape)
            next_interval = exp0 * u.math.asarray(self.tau_m / u.ms, dtype=dftype) * u.ms
            self.t_next = brainstate.ShortTermState(next_interval)

    def _sample_uniform(self, shape):
        r"""Sample uniform random numbers in [0, 1) with functional RNG state update.

        Parameters
        ----------
        shape : tuple
            Shape of the output random array.

        Returns
        -------
        out : jnp.ndarray
            Uniform random samples with dtype ``jnp.float64``.

        Raises
        ------
        ValueError
            If ``shape`` is not a valid tuple for ``jax.random.uniform``.
        """
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        dftype = brainstate.environ.dftype()
        return jax.random.uniform(subkey, shape=shape, dtype=dftype)

    def _sample_exponential(self, shape):
        r"""Sample exponential random variables with rate 1 (mean 1).

        Parameters
        ----------
        shape : tuple
            Shape of the output random array.

        Returns
        -------
        out : jnp.ndarray
            Exponential random samples (rate=1) with dtype ``jnp.float64``.

        Raises
        ------
        ValueError
            If ``shape`` is not a valid tuple for ``jax.random.exponential``.
        """
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        dftype = brainstate.environ.dftype()
        return jax.random.exponential(subkey, shape=shape, dtype=dftype)

    def _gain_probability(self, h):
        r"""Evaluate complementary error function gain at input ``h``.

        Computes :math:`g(h) = \\frac{1}{2}\\,\\mathrm{erfc}\\!\\left(-\\frac{h - \\theta}{\\sqrt{2}\\,\\sigma}\\right)`.

        Parameters
        ----------
        h : ArrayLike
            Effective input (synaptic state plus current input) in mV,
            broadcast-compatible with ``self.varshape``.

        Returns
        -------
        out : float
            Activation probability with the same shape as ``h`` (unitless float64).

        Raises
        ------
        TypeError
            If ``h`` is not unit-compatible with ``theta`` and ``sigma`` (all
            should be in mV).
        """
        arg = -(h - self.theta) / (jnp.sqrt(jnp.asarray(2.0)) * self.sigma)
        return 0.5 * jspecial.erfc(u.math.asarray(arg))

    def update(self, x=0. * u.mV):
        r"""Advance the binary neuron by one simulation step.

        Follows NEST update ordering:

        1. Integrate delta inputs into persistent ``h``.
        2. Compute total input ``h + c`` where ``c`` is current input.
        3. Evaluate gain function :math:`g(h + c)`.
        4. If Poisson-scheduled update is due (``t + dt > t_next``), sample new
           binary state from :math:`g(h + c)` and schedule next update.
        5. Return updated binary output ``y``.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input in mV for this step. Combined with additional
            current sources from :meth:`sum_current_inputs`. Default is
            ``0.0 * u.mV``.

        Returns
        -------
        out : jax.Array
            Binary output state ``self.y.value`` with shape ``self.varshape``
            (or ``(batch_size,) + self.varshape`` if batched). Values are
            float64 (0.0 or 1.0) wrapped in ``jax.lax.stop_gradient`` to
            prevent gradient flow through stochastic sampling.

        Raises
        ------
        KeyError
            If simulation context does not provide required entries ``t`` or
            ``dt`` when ``stochastic_update=True``.
        AttributeError
            If required states are missing because :meth:`init_state` has not
            been called.
        TypeError
            If input/state values are not unit-compatible with expected mV
            arithmetic.

        Notes
        -----
        - When ``stochastic_update=True``, updates only occur at Poisson-
          distributed times (mean interval ``tau_m``). Between updates, ``y``
          remains constant.
        - When ``stochastic_update=False``, the binary state is resampled at
          every time step according to the same gain function.
        - The gain function is never explicitly clipped; effective clipping
          occurs through comparison with uniform random numbers: if
          :math:`g(h + c) < 0`, firing probability is 0; if :math:`g(h + c) > 1`,
          firing probability is 1.
        - All random sampling uses functional JAX RNG state splitting for
          reproducibility and JAX transformation compatibility.
        """
        # NEST ordering: first integrate binary-event deltas into persistent h.
        delta_h = self.sum_delta_inputs(u.math.zeros_like(self.h.value))
        self.h.value = self.h.value + delta_h

        # Then include current input for this step in gain evaluation.
        c = self.sum_current_inputs(x, self.h.value)
        dftype = brainstate.environ.dftype()
        p = u.math.asarray(self._gain_probability(self.h.value + c), dtype=dftype)

        if self.stochastic_update:
            t = brainstate.environ.get('t')
            dt = brainstate.environ.get_dt()
            current_time = t + dt
            should_update = current_time > self.t_next.value

            # Always compute the candidate update and mask per-neuron with
            # ``where``; gating the whole block behind ``if any(should_update)``
            # would break under jax.jit (Python branch on a traced predicate).
            u_rand = self._sample_uniform(self.y.value.shape)
            new_y = u.math.asarray(u_rand < p, dtype=dftype)
            self.y.value = jax.lax.stop_gradient(u.math.where(should_update, new_y, self.y.value))

            next_interval = (
                self._sample_exponential(self.y.value.shape)
                * u.math.asarray(self.tau_m / u.ms, dtype=dftype)
                * u.ms
            )
            self.t_next.value = u.math.where(
                should_update,
                self.t_next.value + next_interval,
                self.t_next.value
            )
        else:
            u_rand = self._sample_uniform(self.y.value.shape)
            self.y.value = jax.lax.stop_gradient(
                u.math.asarray(u_rand < p, dtype=dftype)
            )

        return self.y.value
