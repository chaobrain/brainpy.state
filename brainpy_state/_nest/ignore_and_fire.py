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

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Dynamics

__all__ = [
    'ignore_and_fire',
]


class ignore_and_fire(Dynamics):
    r"""Ignore-and-fire neuron model for generating spikes at fixed intervals.

    Description
    -----------

    The ``ignore_and_fire`` neuron is a neuron model that generates spikes at
    a predefined ``rate`` with a constant inter-spike interval ("fire"),
    irrespective of its inputs ("ignore"). In this simplest version of the
    ``ignore_and_fire`` neuron, the inputs from other neurons or devices are
    not processed at all.

    This is a brainpy.state re-implementation of the NEST simulator model of the
    same name, using NEST-standard parameterization.

    Dynamics
    ........

    The model's internal state variable, the ``phase``, describes the time to
    the next spike relative to the firing period (the inverse of the ``rate``).

    The firing period (in simulation time steps) is computed as:

    .. math::

        T_{\text{fire}} = \text{round}\!\left(\frac{1}{\text{rate}} \times 1000\right) / dt

    where rate is in spikes/s and the result is expressed in simulation time steps
    (NEST rounds this to the simulation grid via ``Time::get_steps()``).

    The initial phase countdown (in simulation time steps) is computed as:

    .. math::

        N_{\text{phase}} = \text{round}\!\left(\frac{\text{phase}}{\text{rate}}
        \times 1000\right) / dt

    In each update step, the model checks whether the countdown has reached zero:

    - If ``phase_steps == 0``: a spike is emitted and the countdown is reset to
      ``firing_period_steps - 1``.
    - Otherwise: the countdown is decremented by 1.

    To create asynchronous activity for a population of ``ignore_and_fire``
    neurons, the firing phases can be randomly initialized.

    .. note::

        The ``ignore_and_fire`` neuron is primarily used for neuronal-network
        model verification and validation purposes ("benchmarking"), in
        particular, to evaluate the correctness and performance of connectivity
        generation and inter-neuron communication. It permits an easy scaling
        of the network size and/or connectivity without affecting the output
        spike statistics. The amount of network traffic is predefined by the
        user, and therefore fully controllable and predictable, irrespective
        of the network size and structure.

    .. note::

        This model inherits from :class:`Dynamics` rather than :class:`Neuron`
        because it has no membrane potential, no threshold-based spike
        generation, and no subthreshold dynamics. Surrogate gradients and
        spike reset mechanisms are therefore not applicable.

    Parameters
    ----------

    The following parameters can be set. Default values match the NEST simulator.

    ==================== ================== =============================== ====================================================
    **Parameter**        **Default**        **Math equivalent**             **Description**
    ==================== ================== =============================== ====================================================
    ``in_size``          (required)                                         Size of the input / number of neurons
    ``phase``            1.0                                                Phase (relative time to next spike; 0 < phase <= 1)
    ``rate``             10.0 Hz                                            Firing rate in spikes/s
    ==================== ================== =============================== ====================================================

    Attributes
    ----------
    phase_steps : ShortTermState
        Integer countdown to next spike (in simulation time steps).

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> # Create an ignore_and_fire neuron with 10 Hz firing rate
    >>> neuron = brainpy.state.ignore_and_fire(1, rate=10.0 * u.Hz)
    >>>
    >>> # Initialize the state
    >>> neuron.init_state()
    >>>
    >>> # Step the neuron and check for spikes
    >>> with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
    ...     spike = neuron.update()

    References
    ----------
    .. [1] NEST Simulator, ``ignore_and_fire`` model.
           https://nest-simulator.readthedocs.io/en/stable/models/ignore_and_fire.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        phase: ArrayLike = 1.0,
        rate: ArrayLike = 10. * u.Hz,
        name: str = None,
    ):
        super().__init__(in_size, name=name)

        # Store parameters
        self.phase = braintools.init.param(phase, self.varshape)
        self.rate = braintools.init.param(rate, self.varshape)

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        phase = self.phase
        rate = self.rate

        # Convert to raw values for comparison
        phase_val = np.asarray(phase)
        if np.any(phase_val <= 0.0) or np.any(phase_val > 1.0):
            raise ValueError("Phase must be > 0 and <= 1.")

        rate_val = np.asarray(u.get_magnitude(rate))
        if np.any(rate_val <= 0.0):
            raise ValueError("Firing rate must be > 0.")

    def _calc_initial_variables(self, batch_size=None):
        """Compute firing_period_steps and phase_steps matching NEST's
        ``calc_initial_variables_`` method.

        NEST computes these as:
            firing_period_steps = Time(ms(1/rate * 1000)).get_steps()
            phase_steps         = Time(ms(phase/rate * 1000)).get_steps()

        ``Time::get_steps()`` rounds to the nearest simulation time step.
        We replicate this by computing the period/phase in ms and dividing
        by dt, then rounding to the nearest integer.
        """
        dt = brainstate.environ.get_dt()
        dt_ms = u.get_magnitude(u.maybe_decimal(dt / u.ms))

        rate_hz = u.get_magnitude(u.maybe_decimal(self.rate / u.Hz))
        phase_val = np.asarray(self.phase)

        # period in ms = 1/rate * 1000
        period_ms = 1.0 / rate_hz * 1000.0
        # NEST uses Time(Time::ms(...)).get_steps() which rounds to nearest step
        firing_period_steps = np.rint(period_ms / dt_ms).astype(np.int32)

        # phase time in ms = phase/rate * 1000
        phase_ms = phase_val / rate_hz * 1000.0
        phase_steps = np.rint(phase_ms / dt_ms).astype(np.int32)

        return firing_period_steps, phase_steps

    def init_state(self, batch_size: int = None, **kwargs):
        firing_period_steps, phase_steps = self._calc_initial_variables(batch_size)

        if batch_size is not None:
            firing_period_steps = np.broadcast_to(firing_period_steps, (batch_size,) + self.varshape)
            phase_steps = np.broadcast_to(phase_steps, (batch_size,) + self.varshape)

        self.firing_period_steps = brainstate.ShortTermState(
            jnp.asarray(firing_period_steps, dtype=jnp.int32)
        )
        self.phase_steps = brainstate.ShortTermState(
            jnp.asarray(phase_steps, dtype=jnp.int32)
        )

    def update(self, x=None):
        """Update the ignore_and_fire neuron for one simulation time step.

        All inputs are ignored. The neuron fires deterministically based on
        its internal phase counter.

        Parameters
        ----------
        x : optional
            Input (ignored). Accepted for API compatibility with other neuron
            models but has no effect on the dynamics.

        Returns
        -------
        spike : jnp.ndarray
            Float array of shape ``varshape`` (or ``(batch, *varshape)``) with
            1.0 where a spike is emitted this step and 0.0 otherwise.
        """
        phase_steps = self.phase_steps.value
        firing_period_steps = self.firing_period_steps.value

        # Threshold crossing: phase_steps == 0 means fire
        spike = jnp.where(phase_steps == 0, 1.0, 0.0)

        # Update phase_steps:
        # if fired (phase_steps == 0): reset to firing_period_steps - 1
        # else: decrement by 1
        new_phase_steps = jnp.where(
            phase_steps == 0,
            firing_period_steps - 1,
            phase_steps - 1,
        )

        self.phase_steps.value = new_phase_steps

        return spike
