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
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Dynamics

__all__ = [
    'ginzburg_neuron',
]


class ginzburg_neuron(Dynamics):
    r"""Binary stochastic neuron with sigmoidal/affine gain function.

    This model re-implements the NEST ``ginzburg_neuron``, a binary neuron that
    updates its output state :math:`y \in \{0, 1\}` stochastically at Poisson-distributed
    intervals. The transition probability depends on a persistent input state :math:`h`
    via a combined linear-sigmoidal gain function.

    **1. Model Dynamics**

    The neuron maintains a persistent input :math:`h` (in mV) and a binary output
    :math:`y \in \{0, 1\}`. State transitions occur at Poisson-distributed times
    with mean interval :math:`\tau_m`. At each update, the transition probability is:

    .. math::

       g(h) = c_1 h + c_2 \frac{1 + \tanh(c_3 (h - \theta))}{2}

    where:

    - :math:`c_1` (1/mV): linear gain coefficient
    - :math:`c_2` (dimensionless): sigmoidal amplitude prefactor
    - :math:`c_3` (1/mV): sigmoidal slope parameter
    - :math:`\theta` (mV): threshold for sigmoidal activation

    The new binary state is sampled as:

    .. math::

       y \leftarrow \mathbb{1}[U < g(h + c)],

    where :math:`U \sim \mathrm{Uniform}(0, 1)` and :math:`c` is the current input
    for the present time step.

    **2. Update Scheduling**

    When ``stochastic_update=True`` (default), updates occur stochastically:

    1. At initialization, draw :math:`\Delta t_0 \sim \mathrm{Exp}(\tau_m)` and
       set :math:`t_{\text{next}} = \Delta t_0`.
    2. At each time step, check if :math:`t + dt > t_{\text{next}}` (strict inequality).
    3. If true, perform state transition and draw new :math:`\Delta t \sim \mathrm{Exp}(\tau_m)`,
       then update :math:`t_{\text{next}} \leftarrow t_{\text{next}} + \Delta t`.

    When ``stochastic_update=False``, the neuron updates at every time step, but
    transitions remain stochastic according to :math:`g(h+c)`.

    **3. Input Accumulation**

    Following NEST semantics, the update order is:

    1. Accumulate delta inputs (from binary events) into :math:`h`.
    2. Read current input :math:`c` for the present step.
    3. Evaluate gain function :math:`g(h + c)` with total input.
    4. Sample new binary state if scheduled for update.

    Delta inputs represent state-change events from upstream binary neurons: positive
    for up-transitions (0→1), negative for down-transitions (1→0).

    **4. Gain Function Properties**

    The combined linear-sigmoidal gain allows modeling both:

    - **Linear neurons** (:math:`c_2 = 0`, :math:`c_1 \neq 0`): :math:`g(h) = c_1 h`
    - **Sigmoidal neurons** (:math:`c_1 = 0`, :math:`c_2 = 1`): :math:`g(h) = \frac{1 + \tanh(c_3(h - \theta))}{2}`
    - **Hybrid models** (:math:`c_1, c_2 \neq 0`): affine-shifted sigmoid with linear component

    The sigmoidal component saturates between 0 and :math:`c_2`, with steepness
    controlled by :math:`c_3` and center at :math:`\theta`.

    **5. Probability Clipping**

    As in NEST, probabilities :math:`g(h+c)` are not explicitly clipped. The comparison
    :math:`U < g(h+c)` provides implicit clipping:

    - :math:`g < 0` → probability 0 (never transition to 1)
    - :math:`g > 1` → probability 1 (always transition to 1)

    This avoids numerical issues with negative or super-unitary probabilities while
    maintaining mathematical equivalence.

    **6. Numerical Implementation**

    - All state variables use ``float64`` precision for accurate random sampling.
    - Random number generation uses ``jax.random`` with stateful PRNGKey updates.
    - State transitions use ``jax.lax.stop_gradient`` to prevent backpropagation
      through stochastic sampling operations.

    Parameters
    ----------
    in_size : Size
        Number or shape of neurons in the population. Can be an integer (1D array)
        or tuple of integers (multi-dimensional array).
    tau_m : ArrayLike, optional
        Mean inter-update interval :math:`\tau_m` (time units). Must be strictly positive.
        Controls the expected time between state transitions in Poisson update mode.
        Default: ``10.0 * u.ms``.
    theta : ArrayLike, optional
        Threshold parameter :math:`\theta` for sigmoidal component (voltage units).
        Determines the input level at which the sigmoid reaches half-maximum.
        Default: ``0.0 * u.mV``.
    c_1 : ArrayLike, optional
        Linear gain coefficient :math:`c_1` (1/voltage units). Sets the slope of
        the linear component. Use ``0.0 / u.mV`` for purely sigmoidal neurons.
        Default: ``0.0 / u.mV``.
    c_2 : ArrayLike, optional
        Sigmoidal gain prefactor :math:`c_2` (dimensionless). Amplitude of the
        sigmoidal component. Use ``1.0`` for standard sigmoid or ``0.0`` for purely
        linear neurons. Default: ``1.0``.
    c_3 : ArrayLike, optional
        Sigmoidal slope parameter :math:`c_3` (1/voltage units). Controls the steepness
        of the sigmoid. Larger values produce sharper transitions around :math:`\theta`.
        Default: ``1.0 / u.mV``.
    y_initializer : Callable[[Size, Optional[int]], ArrayLike], optional
        Initializer for binary state :math:`y`. Should return array of 0.0 or 1.0 values.
        Default: ``braintools.init.Constant(0.0)`` (all neurons start in state 0).
    stochastic_update : bool, optional
        If ``True`` (default), use Poisson-distributed update times as in NEST.
        If ``False``, update at every time step (synchronous updates), but transitions
        remain stochastic according to gain function. Default: ``True``.
    rng_seed : int, optional
        Seed for internal random number generator. Affects both uniform sampling for
        state transitions and exponential sampling for update intervals. Default: ``0``.
    name : str, optional
        Unique identifier for this module instance. If ``None``, auto-generated.

    Parameter Mapping
    -----------------
    Correspondence with NEST ``ginzburg_neuron``:

    ================================  ================================  ================================
    brainpy.state                     NEST                              Notes
    ================================  ================================  ================================
    ``tau_m``                         ``tau_m``                         Mean update interval
    ``theta``                         ``theta``                         Sigmoid threshold
    ``c_1``                           ``c_1``                           Linear gain
    ``c_2``                           ``c_2``                           Sigmoid amplitude
    ``c_3``                           ``c_3``                           Sigmoid slope
    ``y``                             ``S_`` (state variable)           Binary output (0 or 1)
    ``h``                             ``h_`` (state variable)           Persistent input
    ``stochastic_update=True``        Default NEST behavior             Poisson update times
    ``stochastic_update=False``       Not directly available            Synchronous updates
    ================================  ================================  ================================

    State Variables
    ---------------
    y : ShortTermState, shape=(in_size,), dtype=float64
        Binary output state. Values are 0.0 (inactive) or 1.0 (active). Updated
        stochastically according to gain function.
    h : ShortTermState, shape=(in_size,), dtype=float64, units=mV
        Persistent input state. Accumulates delta inputs from upstream neurons and
        determines transition probability via gain function.
    t_next : ShortTermState, shape=(in_size,), dtype=float64, units=ms
        Next scheduled update time (only present when ``stochastic_update=True``).
        Incremented by exponentially-distributed intervals after each update.
    rng_key : ShortTermState, shape=(2,), dtype=uint32
        JAX PRNG key for random number generation. Automatically split and updated
        on each random sample.

    Notes
    -----
    **Binary Communication**: In NEST, binary neurons communicate state changes (not
    absolute states) via spike multiplicity encoding:

    - 0→1 transition sends +1 event
    - 1→0 transition sends -1 event (represented as 2× outgoing spike)
    - No change sends no event

    In brainpy.state, this is represented via delta inputs: positive delta for
    up-transition, negative for down-transition. Projections connecting binary
    neurons should use ``align_pre_projection`` to properly encode state changes.

    **Gain Function Design**: The mixed linear-sigmoidal form allows flexible
    response properties:

    - Pure sigmoid (:math:`c_1=0, c_2=1`): bounded response, saturates at high inputs
    - Linear (:math:`c_2=0`): unbounded response, no saturation
    - Mixed: linear baseline with sigmoidal nonlinearity

    For biological realism, typical settings might be :math:`c_1=0, c_2=1, c_3>0`,
    producing a graded sigmoidal response. For theoretical work (e.g., mean-field
    analysis), :math:`c_1 \neq 0` can simplify calculations.

    **Stochasticity**: This model introduces two sources of randomness:

    1. **Update timing** (when ``stochastic_update=True``): Poisson process with rate :math:`1/\tau_m`
    2. **State transitions**: Bernoulli trial with probability :math:`g(h+c)`

    These combine to produce rich stochastic dynamics even with constant input.

    **Performance Considerations**: Binary neurons are computationally lightweight
    (no differential equations to integrate), making them suitable for large-scale
    network simulations. The ``stochastic_update=False`` mode eliminates exponential
    sampling overhead while retaining stochastic transitions.

    See Also
    --------
    erfc_neuron : Binary neuron with error-function gain
    mcculloch_pitts_neuron : Deterministic binary threshold neuron

    References
    ----------
    .. [1] Ginzburg I, Sompolinsky H (1994). Theory of correlations in stochastic
           neural networks. Physical Review E 50(4):3171–3191.
           DOI: https://doi.org/10.1103/PhysRevE.50.3171
    .. [2] Hertz J, Krogh A, Palmer RG (1991). Introduction to the theory of neural
           computation. Addison-Wesley Publishing Company, Redwood City, CA.
    .. [3] Morrison A, Diesmann M (2007). Maintaining causality in discrete time
           neuronal network simulations. In: Lectures in Supercomputational
           Neuroscience, pp. 267–278. Springer, Berlin, Heidelberg.
           DOI: https://doi.org/10.1007/978-3-540-73159-7_10

    Examples
    --------
    **Basic usage with default sigmoidal gain:**

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
        >>> import brainstate
        >>>
        >>> # Create population of 100 binary neurons with sigmoidal gain
        >>> neurons = bst.ginzburg_neuron(100, tau_m=10*u.ms, theta=5*u.mV, c_3=0.5/u.mV)
        >>>
        >>> # Initialize and simulate
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     neurons.init_all_states()
        ...     # Apply constant input and observe stochastic transitions
        ...     states = []
        ...     for _ in range(1000):
        ...         y = neurons.update(x=8*u.mV)
        ...         states.append(y.mean())  # Average activity across population

    **Linear neuron (c_2=0):**

    .. code-block:: python

        >>> # Pure linear gain: g(h) = c_1 * h
        >>> linear_neurons = bst.ginzburg_neuron(
        ...     50, c_1=0.1/u.mV, c_2=0.0, tau_m=5*u.ms
        ... )

    **Hybrid linear-sigmoidal neuron:**

    .. code-block:: python

        >>> # Combined gain with linear baseline
        >>> hybrid_neurons = bst.ginzburg_neuron(
        ...     50,
        ...     tau_m=8*u.ms,
        ...     theta=3*u.mV,
        ...     c_1=0.05/u.mV,  # Linear component
        ...     c_2=0.8,         # Sigmoid amplitude
        ...     c_3=0.3/u.mV     # Sigmoid slope
        ... )

    **Synchronous updates (stochastic_update=False):**

    .. code-block:: python

        >>> # Update at every time step instead of Poisson times
        >>> sync_neurons = bst.ginzburg_neuron(
        ...     100, tau_m=10*u.ms, stochastic_update=False
        ... )
        >>>
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     sync_neurons.init_all_states()
        ...     # Transitions occur every step, but stochastically
        ...     for _ in range(100):
        ...         y = sync_neurons.update(x=5*u.mV)

    **Network with binary-binary connections:**

    .. code-block:: python

        >>> import brainevent as be
        >>>
        >>> pre = bst.ginzburg_neuron(100, theta=0*u.mV, c_2=1.0, c_3=1.0/u.mV)
        >>> post = bst.ginzburg_neuron(100, theta=2*u.mV, c_2=1.0, c_3=0.5/u.mV)
        >>>
        >>> # Connect with fixed probability, encoding state changes as delta inputs
        >>> proj = be.nn.align_pre_projection(
        ...     pre=pre, post=post,
        ...     comm=be.nn.FixedProb(100, 100, prob=0.1, weight=0.5*u.mV)
        ... )
        >>>
        >>> net = brainstate.nn.Module([pre, post, proj])
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau_m: ArrayLike = 10. * u.ms,
        theta: ArrayLike = 0. * u.mV,
        c_1: ArrayLike = 0. / u.mV,
        c_2: ArrayLike = 1.0,
        c_3: ArrayLike = 1. / u.mV,
        y_initializer: Callable = braintools.init.Constant(0.0),
        stochastic_update: bool = True,
        rng_seed: int = 0,
        name: str = None,
    ):
        super().__init__(in_size, name=name)

        self.tau_m = braintools.init.param(tau_m, self.varshape)
        if u.math.any(self.tau_m <= 0. * u.ms):
            raise ValueError('tau_m must be strictly positive.')

        self.theta = braintools.init.param(theta, self.varshape)
        self.c_1 = braintools.init.param(c_1, self.varshape)
        self.c_2 = braintools.init.param(c_2, self.varshape)
        self.c_3 = braintools.init.param(c_3, self.varshape)
        self.y_initializer = y_initializer
        self.stochastic_update = stochastic_update
        self.rng_seed = int(rng_seed)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize neuron state variables.

        Creates binary output state :math:`y`, persistent input :math:`h`, PRNG key,
        and (if ``stochastic_update=True``) the next update time :math:`t_{\text{next}}`.

        Parameters
        ----------
        batch_size : int, optional
            Number of batches for parallel simulation. If ``None``, creates unbatched
            states with shape ``in_size``. If specified, prepends batch dimension to
            create shape ``(batch_size, *in_size)``.
        **kwargs
            Additional arguments (ignored, for API compatibility).

        Notes
        -----
        - Binary state :math:`y` initialized using ``y_initializer`` (default: all zeros).
        - Input state :math:`h` initialized to zero.
        - Next update time :math:`t_{\text{next}}` drawn from :math:`\mathrm{Exp}(\tau_m)`
          distribution when ``stochastic_update=True``.
        - All state arrays use ``float64`` dtype for precise random sampling.
        """
        shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        y = braintools.init.param(self.y_initializer, self.varshape, batch_size)
        self.y = brainstate.ShortTermState(u.math.asarray(y, dtype=jnp.float64))

        self.h = brainstate.ShortTermState(u.math.zeros(shape, dtype=jnp.float64) * u.mV)

        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))

        if self.stochastic_update:
            exp0 = self._sample_exponential(self.y.value.shape)
            next_interval = exp0 * u.math.asarray(self.tau_m / u.ms, dtype=jnp.float64) * u.ms
            self.t_next = brainstate.ShortTermState(next_interval)

    def _sample_uniform(self, shape):
        r"""Draw uniform random samples from [0, 1).

        Parameters
        ----------
        shape : tuple of int
            Shape of output array.

        Returns
        -------
        jax.Array
            Uniform random samples with dtype ``float64``.

        Notes
        -----
        Automatically updates internal PRNG key for next call.
        """
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.uniform(subkey, shape=shape, dtype=jnp.float64)

    def _sample_exponential(self, shape):
        r"""Draw exponential random samples with rate=1.

        Parameters
        ----------
        shape : tuple of int
            Shape of output array.

        Returns
        -------
        jax.Array
            Exponential random samples (mean=1.0) with dtype ``float64``.

        Notes
        -----
        Automatically updates internal PRNG key for next call. Multiply by
        :math:`\tau_m` to get inter-update intervals.
        """
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.exponential(subkey, shape=shape, dtype=jnp.float64)

    def _gain_probability(self, h):
        r"""Compute transition probability from input state.

        Evaluates the combined linear-sigmoidal gain function:

        .. math::

           g(h) = c_1 h + c_2 \\frac{1 + \\tanh(c_3 (h - \\theta))}{2}

        Parameters
        ----------
        h : ArrayLike, units=mV
            Input state(s), shape ``(in_size,)`` or ``(batch_size, *in_size)``.

        Returns
        -------
        ArrayLike, dimensionless
            Transition probability :math:`g(h)`, same shape as input. Not clipped;
            comparison with uniform random number provides implicit clipping.

        Notes
        -----
        - Sigmoidal component saturates smoothly between 0 and :math:`c_2`.
        - Linear component can extend probability beyond [0, 1]; implicit clipping
          occurs during Bernoulli sampling.
        - For :math:`c_1=0, c_2=1`, this is a standard sigmoid with range [0, 1].
        """
        return self.c_1 * h + self.c_2 * 0.5 * (1.0 + u.math.tanh(self.c_3 * (h - self.theta)))

    def update(self, x=0. * u.mV):
        r"""Perform one simulation step with stochastic state transition.

        Accumulates inputs, evaluates gain function, and (if scheduled or in synchronous
        mode) performs Bernoulli trial for state transition.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input for this time step (voltage units). Can be scalar
            (broadcast to all neurons) or array with shape matching ``in_size``.
            Default: ``0.0 * u.mV``.

        Returns
        -------
        y : jax.Array, shape=(in_size,), dtype=float64
            Updated binary state (0.0 or 1.0) after stochastic transition.

        Notes
        -----
        **Update sequence (matching NEST):**

        1. Accumulate delta inputs into :math:`h`: :math:`h \leftarrow h + \Delta h`
        2. Compute total input: :math:`h_{\text{total}} = h + c` (current inputs)
        3. Evaluate gain: :math:`p = g(h_{\text{total}})`
        4. If scheduled (``stochastic_update=True``) or always (``stochastic_update=False``):

           - Draw :math:`U \sim \mathrm{Uniform}(0,1)`
           - Set :math:`y \leftarrow \mathbb{1}[U < p]`
           - If ``stochastic_update=True``, update :math:`t_{\text{next}}`

        **Stochastic update timing:**

        When ``stochastic_update=True``, updates occur when :math:`t + dt > t_{\text{next}}`
        (strict inequality). After update, draw :math:`\Delta t \sim \mathrm{Exp}(\tau_m)`
        and set :math:`t_{\text{next}} \leftarrow t_{\text{next}} + \Delta t`.

        **Synchronous mode:**

        When ``stochastic_update=False``, neurons update every time step. Transitions
        are still stochastic (Bernoulli with probability :math:`p`), but no longer
        Poisson-distributed in time.

        **Non-differentiability:**

        State transitions use ``jax.lax.stop_gradient`` to prevent backpropagation
        through stochastic sampling. For gradient-based learning, consider differentiable
        rate-based neurons or surrogate gradient methods.
        """
        # NEST update order: first integrate binary-event deltas into h.
        delta_h = self.sum_delta_inputs(u.math.zeros_like(self.h.value))
        self.h.value = self.h.value + delta_h

        # Then include current input for this step in gain evaluation.
        c = self.sum_current_inputs(x, self.h.value)
        p = u.math.asarray(self._gain_probability(self.h.value + c), dtype=jnp.float64)

        if self.stochastic_update:
            t = brainstate.environ.get('t')
            dt = brainstate.environ.get_dt()
            current_time = t + dt
            should_update = current_time > self.t_next.value

            if bool(u.math.asarray(u.math.any(should_update))):
                u_rand = self._sample_uniform(self.y.value.shape)
                new_y = u.math.asarray(u_rand < p, dtype=jnp.float64)
                self.y.value = jax.lax.stop_gradient(u.math.where(should_update, new_y, self.y.value))

                next_interval = (
                    self._sample_exponential(self.y.value.shape)
                    * u.math.asarray(self.tau_m / u.ms, dtype=jnp.float64)
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
                u.math.asarray(u_rand < p, dtype=jnp.float64)
            )

        return self.y.value
