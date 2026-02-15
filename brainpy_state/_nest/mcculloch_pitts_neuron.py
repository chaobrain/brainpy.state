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
    'mcculloch_pitts_neuron',
]


class mcculloch_pitts_neuron(Dynamics):
    r"""Binary deterministic neuron with Heaviside activation function.

    Description
    -----------

    ``mcculloch_pitts_neuron`` is an implementation of a binary neuron that
    is irregularly updated at Poisson time points [1]_. At each update point,
    the total synaptic input :math:`h` into the neuron is summed up, passed
    through a Heaviside gain function

    .. math::

       g(h) = H(h - \theta)

    whose output is either 1 (if input is above threshold :math:`\theta`) or
    0 (if input is below threshold).

    This is a brainpy.state re-implementation of the NEST simulator model of
    the same name, using NEST-standard parameterization.

    Update scheme
    .............

    In NEST, the neuron is updated at Poisson-distributed random time points
    with mean inter-update-interval :math:`\tau_m`. This implementation provides
    two modes:

    - **Deterministic mode** (default, ``stochastic_update=False``): The neuron
      is updated at every simulation time step. This is equivalent to setting
      :math:`\tau_m = dt` in the original NEST model and is suitable for
      synchronous update simulations.

    - **Stochastic mode** (``stochastic_update=True``): The neuron is updated
      at Poisson-distributed time points with mean interval :math:`\tau_m`,
      matching the original NEST behavior. The next update time is drawn from
      an exponential distribution. This requires a JAX PRNG key to be set in
      the environment.

    Input handling
    ..............

    The neuron accumulates inputs into the total input variable :math:`h`.
    Binary spike inputs from other binary neurons are decoded as:

    - Two spikes at the same time (multiplicity 2) signal an up-transition
      (0 → 1), contributing ``+weight`` to :math:`h`.
    - A single spike signals a down-transition (1 → 0), contributing
      ``-weight`` to :math:`h`.

    Current inputs (e.g., from a ``dc_generator``) are added directly to
    :math:`h` at each update point.

    In this implementation, external inputs are passed via the ``x`` argument
    to ``update()`` and via ``add_current_input()`` / ``add_delta_input()``.
    Delta inputs modify :math:`h` directly (analogous to binary spike events),
    while current inputs are added at each update step.

    Parameters
    ----------

    The following parameters can be set. Default values match the NEST simulator.

    ====================== ================== =============================== ====================================================
    **Parameter**          **Default**        **Math equivalent**             **Description**
    ====================== ================== =============================== ====================================================
    ``in_size``            (required)                                         Size of the input / number of neurons
    ``tau_m``              10 ms              :math:`\tau_m`                  Membrane time constant (mean inter-update-interval)
    ``theta``              0.0 mV             :math:`\theta`                  Threshold for Heaviside activation function
    ``y_initializer``      Constant(0.0)                                      Initializer for the binary output state
    ``stochastic_update``  ``False``                                          If True, use Poisson-distributed update times
    ====================== ================== =============================== ====================================================

    Attributes
    ----------
    y : ShortTermState
        Binary output state of the neuron (0.0 or 1.0).
    h : ShortTermState
        Total synaptic input to the neuron.

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create a McCulloch-Pitts neuron with default parameters
    >>> neuron = mcculloch_pitts_neuron(1, theta=0.5 * u.mV)
    >>>
    >>> # Initialize the state
    >>> neuron.init_state()
    >>>
    >>> # Update with input above threshold -> output becomes 1
    >>> with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
    ...     output = neuron.update(x=1.0 * u.mV)

    .. admonition:: Differences from NEST

       - In NEST, binary neurons communicate state transitions via spike
         multiplicity (2 for up-transition, 1 for down-transition). In this
         implementation, the binary state is returned directly as a float
         (0.0 or 1.0) from ``update()``.
       - The stochastic Poisson update timing is optional and disabled by
         default for deterministic reproducibility.

    References
    ----------
    .. [1] McCulloch W, Pitts W (1943). A logical calculus of the ideas
           immanent in nervous activity. Bulletin of Mathematical Biophysics,
           5:115-133. DOI: https://doi.org/10.1007/BF02478259
    .. [2] Hertz J, Krogh A, Palmer R (1991). Introduction to the theory
           of neural computation. Addison-Wesley Publishing Company.
    .. [3] Morrison A, Diesmann M (2007). Maintaining causality in discrete
           time neuronal simulations. In: Lectures in Supercomputational
           Neuroscience, p. 267. Peter beim Graben, Changsong Zhou, Marco
           Thiel, Juergen Kurths (Eds.), Springer.
           DOI: https://doi.org/10.1007/978-3-540-73159-7_10

    See also
    --------
    iaf_psc_delta : Leaky integrate-and-fire neuron with delta-shaped PSCs
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
        """Heaviside activation function: g(h) = H(h - theta).

        Returns 1.0 if h > theta, 0.0 otherwise. This matches NEST's
        ``gainfunction_mcculloch_pitts::operator()`` which uses strict
        inequality (h > theta).

        Parameters
        ----------
        h : ArrayLike
            Total input to the neuron (in mV).

        Returns
        -------
        ArrayLike
            Binary output (0.0 or 1.0).
        """
        return u.math.asarray(h > self.theta, dtype=jnp.float64)

    def update(self, x=0. * u.mV):
        """Update the neuron state for one simulation time step.

        At each (scheduled) update point, the total input ``h`` is evaluated
        through the Heaviside gain function to determine the new binary state.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input to the neuron (in mV). Default is 0 mV.
            This is added to the accumulated input ``h`` at the update point,
            analogous to NEST's ``CurrentEvent`` handling where currents are
            added to the ``currents_`` ring buffer.

        Returns
        -------
        ArrayLike
            The binary output state (0.0 or 1.0) after the update.
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
