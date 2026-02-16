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

from typing import Sequence

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

__all__ = [
    'spike_generator',
]


class spike_generator(brainstate.nn.Dynamics):
    r"""Spike generator -- NEST-compatible stimulation device.

    Description
    -----------

    ``spike_generator`` generates spikes at specified times. It can be used to
    provide precisely timed spike input to connected neurons.

    This is a brainpy.state re-implementation of the NEST simulator device of
    the same name, using NEST-standard parameterization.

    Spike generation
    ................

    Spikes are emitted at the times specified in ``spike_times``. The spike
    times array must be sorted in non-descending order (earliest spike first).

    At each simulation step, the generator checks whether any spike times fall
    within the current time step's interval. If so, a spike (output value of 1)
    is produced; otherwise, the output is 0.

    Multiple occurrences of the same time indicate that more than one event is
    to be generated at that time (the output will still be 1 in this
    implementation, as spikes are binary in the brainpy.state framework).

    Optionally, ``spike_weights`` can be set. This is an array of the same
    length as ``spike_times``, containing one weight per spike. When set, the
    output at spike times is the corresponding weight instead of 1.

    Timing convention
    .................

    A spike at time :math:`t_s` is emitted during the simulation step where
    the simulation time ``t`` satisfies :math:`t_s - dt < t \le t_s` (i.e.,
    the spike is delivered at the end of the step). In practice, for
    grid-aligned spike times, the spike is emitted at step
    :math:`t_s / dt`.

    The active window ``[origin + start, origin + stop)`` gates the output:
    spikes outside this window are suppressed.

    Parameters
    ----------

    The following parameters can be set. Default values match the NEST simulator.

    ======================= ================== ============================================
    **Parameter**           **Default**        **Description**
    ======================= ================== ============================================
    ``in_size``             1                  Output size (number of independent generators)
    ``spike_times``         ``[]``             Spike times in ms (list, sorted)
    ``spike_weights``       ``[]``             Spike weights (list, same length as spike_times)
    ``start``               0 ms               Activation time relative to ``origin``
    ``stop``                ``None`` (inf)     Deactivation time relative to ``origin``
    ``origin``              0 ms               Global time offset
    ======================= ================== ============================================

    Examples
    --------

    Basic usage with a neuron:

    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    ...     sg = brainpy.state.spike_generator(
    ...         spike_times=[5. * u.ms, 10. * u.ms, 15. * u.ms],
    ...     )
    ...
    ...     for step in range(200):
    ...         with brainstate.environ.context(t=step * 0.1 * u.ms):
    ...             spk = sg.update()

    With spike weights:

    >>> sg = brainpy.state.spike_generator(
    ...     spike_times=[5. * u.ms, 10. * u.ms],
    ...     spike_weights=[2.0, 0.5],
    ... )

    References
    ----------
    .. [1] NEST Simulator, ``spike_generator`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/spike_generator.html

    See Also
    --------
    dc_generator : Constant current generator
    ac_generator : Sinusoidal current generator
    step_current_generator : Piecewise constant current generator
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        spike_times: Sequence = (),
        spike_weights: Sequence = (),
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        # Store spike times in ms
        self._spike_times_ms = []
        for t in spike_times:
            if u.is_unitless(t):
                self._spike_times_ms.append(float(t))
            else:
                self._spike_times_ms.append(float(t / u.ms))

        # Validate non-descending order
        for i in range(1, len(self._spike_times_ms)):
            if self._spike_times_ms[i] < self._spike_times_ms[i - 1]:
                raise ValueError(
                    "spike_times must be sorted in non-descending order. "
                    f"Got {self._spike_times_ms[i - 1]} > {self._spike_times_ms[i]} at index {i}."
                )

        # Store spike weights
        if len(spike_weights) > 0 and len(spike_weights) != len(spike_times):
            raise ValueError(
                "spike_weights must have the same length as spike_times "
                f"or be empty. Got {len(spike_weights)} and {len(spike_times)}."
            )

        self._spike_weights = [float(w) for w in spike_weights]

        self.start = braintools.init.param(start, self.varshape)
        if stop is not None:
            self.stop = braintools.init.param(stop, self.varshape)
        else:
            self.stop = None
        self.origin = braintools.init.param(origin, self.varshape)

    def update(self):
        """Return spike output at the current simulation time.

        Checks if any spike times match the current simulation time. If so,
        returns the spike weight (or 1.0 if no weights are set). Otherwise
        returns 0.0.

        The spike time matching uses a tolerance of ``dt/2`` to handle
        floating-point alignment to the simulation grid.

        Returns
        -------
        spike : array
            Spike output, shaped ``(in_size,)``. 1.0 (or weight) at spike
            times, 0.0 otherwise.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()

        # Get t and dt in ms
        if u.is_unitless(t):
            t_ms = float(t)
        else:
            t_ms = float(t / u.ms)

        if u.is_unitless(dt):
            dt_ms = float(dt)
        else:
            dt_ms = float(dt / u.ms)

        # Check if device is active
        if u.is_unitless(self.start):
            t_start_ms = float(self.origin + self.start)
        else:
            t_start_ms = float((self.origin + self.start) / u.ms)

        if self.stop is not None:
            if u.is_unitless(self.stop):
                t_stop_ms = float(self.origin + self.stop)
            else:
                t_stop_ms = float((self.origin + self.stop) / u.ms)
            active = t_ms >= t_start_ms and t_ms < t_stop_ms
        else:
            active = t_ms >= t_start_ms

        if not active:
            return jnp.zeros(self.varshape)

        # Check for spikes at current time
        # A spike at time t_s fires at the simulation step where t == t_s
        # (grid-aligned). We use a tolerance of dt/2 for matching.
        tol = dt_ms / 2.0
        spike_val = 0.0
        for i in range(len(self._spike_times_ms)):
            spike_t = self._spike_times_ms[i]
            if abs(t_ms - spike_t) < tol:
                if self._spike_weights:
                    spike_val = self._spike_weights[i]
                else:
                    spike_val = 1.0
                # Don't break -- if multiple spikes at same time, use last weight
                # (or accumulate if needed, but NEST uses multiplicity)

        return spike_val * jnp.ones(self.varshape)
