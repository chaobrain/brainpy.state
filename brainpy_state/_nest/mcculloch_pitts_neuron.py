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

from brainpy_state._nest._base import NESTNeuron

__all__ = [
    'mcculloch_pitts_neuron',
]


class mcculloch_pitts_neuron(NESTNeuron):
    r"""Binary deterministic neuron with Heaviside activation function.

    This model implements a binary neuron that transitions between two discrete states (0 or 1)
    based on a Heaviside threshold function applied to the total synaptic input. It replicates
    the NEST simulator's ``mcculloch_pitts_neuron`` model with NEST-standard parameterization
    and supports both deterministic (synchronous) and stochastic (Poisson-distributed) update
    schedules.

    **1. Mathematical Formulation**

    At each update time point, the neuron evaluates its binary output :math:`y` as:

    .. math::

       y(t) = H\left(h(t) - \theta\right) = \begin{cases}
           1 & \text{if } h(t) > \theta \\
           0 & \text{if } h(t) \leq \theta
       \end{cases}

    where:

      - :math:`h(t)` is the total accumulated synaptic input (in mV)
      - :math:`\theta` is the activation threshold (in mV)
      - :math:`H(\cdot)` is the Heaviside step function (strict inequality)

    The total input :math:`h(t)` is computed as:

    .. math::

       h(t) = h_{\text{prev}} + \sum_{\text{delta}} \Delta h_i + \sum_{\text{current}} I_j(t)

    where :math:`\Delta h_i` are discrete delta inputs (e.g., from binary spike events) and
    :math:`I_j(t)` are continuous current inputs (e.g., from external stimulation).

    **2. Update Timing Modes**

    The model provides two distinct update schedules:

    - **Deterministic mode** (``stochastic_update=False``, default):
      The neuron updates at every simulation time step :math:`dt`. This is equivalent to the
      NEST model with :math:`\tau_m = dt` and provides synchronous, reproducible dynamics
      suitable for discrete-time simulation frameworks.

    - **Stochastic mode** (``stochastic_update=True``):
      The neuron updates at Poisson-distributed random time points with mean inter-update
      interval :math:`\tau_m`. Update times are drawn from an exponential distribution:

      .. math::

         t_{\text{next}} = t_{\text{current}} + \Delta t, \quad \Delta t \sim \text{Exp}(\tau_m)

      This matches the original NEST implementation and models asynchronous, event-driven
      dynamics. Requires a JAX PRNG key in the environment (``brainstate.environ.key``).

    **3. Input Handling and Binary Communication**

    Binary neurons in NEST encode state transitions via spike multiplicity:
      - **Two spikes** (multiplicity = 2) signal an **up-transition** (0 → 1), contributing
        ``+weight`` to :math:`h`
      - **One spike** (multiplicity = 1) signals a **down-transition** (1 → 0), contributing
        ``-weight`` to :math:`h`

    In this brainpy.state implementation:
      - Delta inputs (via ``add_delta_input()``) directly modify :math:`h` and are analogous
        to binary spike events. Positive deltas promote up-transitions; negative deltas promote
        down-transitions.
      - Current inputs (via ``add_current_input()`` or the ``x`` argument) are added to :math:`h`
        at each update point and represent continuous driving currents (e.g., ``dc_generator``
        in NEST).

    **4. Computational Considerations**

    - **Strict inequality**: The Heaviside function uses :math:`h > \theta` (not :math:`\geq`),
      matching NEST's ``gainfunction_mcculloch_pitts::operator()`` implementation. This means
      :math:`h = \theta` produces output 0.
    - **Non-differentiable dynamics**: The binary output is wrapped in ``jax.lax.stop_gradient``
      to prevent gradient flow, as the Heaviside function is discontinuous and not suitable for
      gradient-based optimization without surrogate gradients.
    - **Stochastic update timing**: In stochastic mode, updates may be skipped if
      :math:`t_{\text{current}} \leq t_{\text{next}}`. State remains unchanged between updates.

    **5. Implementation-Specific Differences from NEST**

    - **Binary state representation**: NEST communicates state transitions via spike multiplicity
      (1 or 2 spikes). This implementation returns the binary state (0.0 or 1.0) directly from
      ``update()``, which is more natural for modular composition with other BrainPy models.
    - **Stochastic update default**: Stochastic Poisson-distributed updates are disabled by
      default (``stochastic_update=False``) for deterministic reproducibility. Enable explicitly
      for NEST-equivalent stochastic behavior.
    - **Ring buffer simulation**: NEST uses ring buffers (``spikes_``, ``currents_``) for
      spike delivery delays. This implementation uses ``delta_inputs`` / ``current_inputs``
      dictionaries managed by the ``Dynamics`` base class.

    Parameters
    ----------
    in_size : int, tuple of int
        Number of neurons in the population. Can be an integer for a 1D array or a tuple
        for multi-dimensional populations (e.g., ``(10, 10)`` for a 10×10 grid).
    tau_m : float, ArrayLike, optional
        Membrane time constant (mean inter-update interval for stochastic mode).
        Must have units of time (ms). Scalar or array matching ``in_size`` shape.
        Default: ``10.0 * u.ms``.
    theta : float, ArrayLike, optional
        Threshold for the Heaviside activation function. Must have units of voltage (mV).
        Scalar or array matching ``in_size`` shape. Default: ``0.0 * u.mV``.
    y_initializer : Callable, optional
        Initializer for the binary output state ``y``. Should return values in {0.0, 1.0}.
        Default: ``braintools.init.Constant(0.0)`` (all neurons start in state 0).
    stochastic_update : bool, optional
        If ``True``, use Poisson-distributed update times with mean interval ``tau_m``.
        If ``False`` (default), update deterministically at every time step.
    name : str, optional
        Name of the neuron module for identification in the computational graph.


    Parameter Mapping
    -----------------

    ====================== ================== =============================== ====================================================
    **Parameter**          **Default**        **Math equivalent**             **Description**
    ====================== ================== =============================== ====================================================
    ``in_size``            (required)         —                               Number of neurons (population size)
    ``tau_m``              10 ms              :math:`\tau_m`                  Membrane time constant (mean inter-update interval)
    ``theta``              0.0 mV             :math:`\theta`                  Activation threshold for Heaviside function
    ``y_initializer``      Constant(0.0)      —                               Initial binary state (0.0 or 1.0)
    ``stochastic_update``  ``False``          —                               Enable Poisson-distributed update times (NEST mode)
    ====================== ================== =============================== ====================================================

    Attributes
    ----------
    y : brainstate.ShortTermState
        Binary output state of the neuron. Shape: ``(in_size,)`` or ``(batch_size, *in_size)``.
        Values are float64 in {0.0, 1.0}. Updated at each (scheduled) update point.
    h : brainstate.ShortTermState
        Total accumulated synaptic input (in mV). Shape matches ``y``. Includes contributions
        from delta inputs (spike-like events) and current inputs (continuous driving).
    t_next : brainstate.ShortTermState (only in stochastic mode)
        Next scheduled update time (in ms). Only created if ``stochastic_update=True``.
        Initialized to a large negative value (-1e7 ms) to trigger immediate first update.

    Examples
    --------
    **Example 1: Deterministic binary neuron with threshold crossing**

    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> import brainpy_state as bst
       >>>
       >>> # Create a single McCulloch-Pitts neuron with threshold 0.5 mV
       >>> neuron = bst.mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
       >>> neuron.init_state()
       >>>
       >>> # Simulate with inputs below and above threshold
       >>> with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
       ...     output1 = neuron.update(x=0.3 * u.mV)  # Below threshold -> 0
       ...     output2 = neuron.update(x=0.8 * u.mV)  # Above threshold -> 1
       >>> print(output1, output2)
       [0.] [1.]

    **Example 2: Stochastic update mode with Poisson timing**

    .. code-block:: python

       >>> import jax
       >>> neuron = bst.mcculloch_pitts_neuron(
       ...     100, tau_m=5.0 * u.ms, stochastic_update=True
       ... )
       >>> neuron.init_state()
       >>>
       >>> # Run simulation with PRNG key for stochastic updates
       >>> key = jax.random.PRNGKey(0)
       >>> with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms, key=key):
       ...     for step in range(100):
       ...         output = neuron.update(x=1.0 * u.mV)
       ...         # Neuron updates only at Poisson-distributed time points

    **Example 3: Binary network with delta inputs**

    .. code-block:: python

       >>> # Create a pair of binary neurons
       >>> pre = bst.mcculloch_pitts_neuron(1, theta=0.0 * u.mV)
       >>> post = bst.mcculloch_pitts_neuron(1, theta=1.5 * u.mV)
       >>> pre.init_state()
       >>> post.init_state()
       >>>
       >>> # Connect pre -> post with weight 2.0 mV
       >>> post.add_delta_input('pre_input', pre.y.value * 2.0 * u.mV)
       >>>
       >>> with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
       ...     pre_out = pre.update(x=1.0 * u.mV)  # Pre transitions to 1
       ...     post_out = post.update()             # Post receives delta input 2.0 mV

    Notes
    -----
    - **Non-differentiability**: Binary outputs are non-differentiable and wrapped in
      ``stop_gradient``. For gradient-based learning, use spiking neurons with surrogate
      gradients (e.g., ``LIF`` with ``braintools.surrogate`` spike functions).
    - **Stochastic update caveat**: In stochastic mode, the exponential distribution for
      inter-update intervals uses the JAX PRNG key from ``brainstate.environ.key``. Ensure
      the key is updated between simulation steps for proper randomness (e.g., using
      ``jax.random.split``).
    - **Threshold boundary**: The strict inequality :math:`h > \theta` means that
      :math:`h = \theta` produces output 0. For symmetric threshold behavior, adjust
      ``theta`` by a small epsilon (e.g., ``theta - 1e-10 * u.mV``).

    References
    ----------
    .. [1] McCulloch WS, Pitts W (1943). A logical calculus of the ideas immanent in nervous
           activity. Bulletin of Mathematical Biophysics, 5:115-133.
           DOI: https://doi.org/10.1007/BF02478259
    .. [2] Hertz J, Krogh A, Palmer RG (1991). Introduction to the Theory of Neural Computation.
           Addison-Wesley Publishing Company. ISBN: 978-0201515602.
    .. [3] Morrison A, Diesmann M (2007). Maintaining causality in discrete time neuronal
           simulations. In: Lectures in Supercomputational Neuroscience, pp. 267-278.
           Peter beim Graben, Changsong Zhou, Marco Thiel, Jürgen Kurths (Eds.), Springer.
           DOI: https://doi.org/10.1007/978-3-540-73159-7_10

    See Also
    --------
    ginzburg_neuron : Binary neuron with stochastic activation (Boltzmann distribution)
    erfc_neuron : Binary neuron with erfc activation (smooth approximation to Heaviside)
    iaf_psc_delta : Leaky integrate-and-fire neuron with delta-shaped postsynaptic currents
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau_m: ArrayLike = 10. * u.ms,
        theta: ArrayLike = 0. * u.mV,
        y_initializer: Callable = braintools.init.Constant(0.),
        stochastic_update: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name)

        # parameters
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.theta = braintools.init.param(theta, self.varshape)
        self.y_initializer = y_initializer
        self.stochastic_update = stochastic_update

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize neuron state variables.

        Allocates and initializes the binary output state ``y``, total synaptic input ``h``,
        and (if stochastic updates are enabled) the next scheduled update time ``t_next``.

        Parameters
        ----------
        batch_size : int, optional
            Number of parallel batches for vectorized simulation. If ``None``, state shape
            is ``(in_size,)``. If provided, state shape is ``(batch_size, *in_size)``.
        **kwargs
            Additional keyword arguments (reserved for future use).

        Notes
        -----
        - **Binary state initialization**: ``y`` is initialized using ``y_initializer``.
          Default initializer is ``Constant(0.0)``, so all neurons start in the inactive state.
        - **Input accumulator**: ``h`` is initialized to zero (0.0 mV). Delta and current inputs
          accumulate into this variable during ``update()``.
        - **Stochastic timing**: If ``stochastic_update=True``, ``t_next`` is initialized to
          a large negative value (-1e7 ms) to ensure the first update triggers immediately
          at simulation start.
        """
        # Binary output state y (0.0 or 1.0)
        y = braintools.init.param(self.y_initializer, self.varshape, batch_size)
        self.y = brainstate.ShortTermState(u.math.asarray(y, dtype=jnp.float64))

        # Total synaptic input h
        self.h = brainstate.ShortTermState(
            u.math.zeros(self.varshape if batch_size is None else (batch_size, *self.varshape),
                         dtype=jnp.float64) * u.mV
        )

        # Next update time for stochastic mode
        if self.stochastic_update:
            self.t_next = brainstate.ShortTermState(
                u.math.full(
                    self.varshape if batch_size is None else (batch_size, *self.varshape),
                    -1e7,
                    dtype=jnp.float64
                ) * u.ms
            )

    def _heaviside(self, h):
        r"""Heaviside step function with strict inequality threshold test.

        Computes the binary activation :math:`H(h - \theta)` where:

        .. math::

           H(h - \theta) = \begin{cases}
               1.0 & \text{if } h > \theta \\
               0.0 & \text{if } h \leq \theta
           \end{cases}

        This implements the strict inequality :math:`h > \theta` to match NEST's
        ``gainfunction_mcculloch_pitts::operator()`` implementation. Inputs exactly
        equal to threshold produce output 0.

        Parameters
        ----------
        h : ArrayLike
            Total synaptic input to the neuron (in mV). Shape: ``(in_size,)`` or
            ``(batch_size, *in_size)``. Can be scalar or array.

        Returns
        -------
        ArrayLike
            Binary output values in {0.0, 1.0} (float64). Shape matches input ``h``.
            Returns 1.0 where ``h > theta``, 0.0 elsewhere.

        Notes
        -----
        - The output dtype is explicitly cast to ``float64`` for consistency with
          internal state variables.
        - The strict inequality means :math:`h = \theta` yields 0.0 (inactive state).
        """
        return u.math.asarray(h > self.theta, dtype=jnp.float64)

    def update(self, x=0. * u.mV):
        r"""Update the neuron state for one simulation time step.

        Accumulates delta and current inputs into the total input variable :math:`h`, then
        evaluates the Heaviside activation function :math:`H(h - \theta)` to determine the
        new binary output state :math:`y`. In stochastic mode, updates occur only at
        Poisson-distributed time points; in deterministic mode, updates occur every step.

        **Update Algorithm (Deterministic Mode)**:

        1. Accumulate delta inputs: :math:`h \leftarrow h + \sum_i \Delta h_i`
        2. Add current inputs: :math:`c = x + \sum_j I_j`
        3. Evaluate activation: :math:`y = H(h + c - \theta)`

        **Update Algorithm (Stochastic Mode)**:

        1. Accumulate inputs as above
        2. Check if current time :math:`t > t_{\text{next}}`
        3. If yes: evaluate activation, update :math:`y`, draw next update time from
           :math:`\text{Exp}(\tau_m)`
        4. If no: keep :math:`y` unchanged

        Parameters
        ----------
        x : float, ArrayLike, optional
            External current input (in mV). Shape must be broadcastable to ``(batch_size, *in_size)``
            or ``(in_size,)``. This represents continuous driving current (e.g., from a
            ``dc_generator`` in NEST) and is added to :math:`h` at the update point.
            Default: ``0.0 * u.mV`` (no external input).

        Returns
        -------
        ArrayLike
            Binary output state :math:`y` (0.0 or 1.0) after the update. Shape: ``(in_size,)``
            or ``(batch_size, *in_size)``. Values are wrapped in ``jax.lax.stop_gradient``
            to prevent gradient flow through the discontinuous Heaviside function.

        Notes
        -----
        - **Delta inputs**: Inputs registered via ``add_delta_input()`` are accumulated into
          :math:`h` before the activation check. These represent discrete spike-like events
          from other binary neurons (analogous to NEST's ``spikes_`` ring buffer).
        - **Current inputs**: Inputs registered via ``add_current_input()`` plus the ``x``
          argument are summed into :math:`c` and added to :math:`h` for the activation
          check (analogous to NEST's ``currents_`` ring buffer). Unlike delta inputs,
          current inputs do NOT persist in :math:`h` across time steps.
        - **Stochastic update skipping**: In stochastic mode, if the scheduled update time
          has not been reached, the neuron state remains unchanged and the activation function
          is not evaluated. The next update time is drawn from an exponential distribution
          when an update occurs.
        - **PRNG key requirement**: Stochastic mode requires a JAX PRNG key in the environment
          (``brainstate.environ.key``). If no key is provided, the exponential distribution
          cannot be sampled and update times will not advance correctly.
        - **Non-differentiability**: Output is wrapped in ``stop_gradient`` to prevent
          autodiff from propagating gradients through the discontinuous threshold operation.
"""
        # Accumulate delta inputs into h (analogous to binary spike events
        # modifying h via the spikes_ ring buffer in NEST)
        delta_h = self.sum_delta_inputs(u.math.zeros_like(self.h.value))
        self.h.value = self.h.value + delta_h

        # Current inputs are added at the update point (analogous to
        # currents_ ring buffer in NEST, variable c in the update loop)
        c = self.sum_current_inputs(x, self.h.value)

        if self.stochastic_update:
            # Stochastic update: only update if current time > t_next
            t = brainstate.environ.get('t')
            dt = brainstate.environ.get_dt()
            current_time = t + dt

            should_update = current_time > self.t_next.value

            # Evaluate gain function: new_y = H(h + c - theta)
            new_y = self._heaviside(self.h.value + c)

            # Only apply update where scheduled
            self.y.value = jax.lax.stop_gradient(
                u.math.where(should_update, new_y, self.y.value)
            )

            # Draw next update time from exponential distribution where update happened
            key = brainstate.environ.get('key', default=None)
            if key is not None:
                exp_sample = jax.random.exponential(key, shape=self.y.value.shape)
                next_interval = exp_sample * u.math.asarray(self.tau_m / u.ms, dtype=jnp.float64) * u.ms
                self.t_next.value = u.math.where(
                    should_update,
                    self.t_next.value + next_interval,
                    self.t_next.value
                )
        else:
            # Deterministic update: evaluate gain function every step
            # new_y = H(h + c - theta)
            new_y = self._heaviside(self.h.value + c)
            self.y.value = jax.lax.stop_gradient(new_y)

        return self.y.value
