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
    'step_current_generator',
]


class step_current_generator(brainstate.nn.Dynamics):
    r"""Piecewise constant DC current generator -- NEST-compatible stimulation device.

    Description
    -----------

    ``step_current_generator`` provides a piecewise constant DC input to the
    connected neuron(s). The amplitude of the current changes at the specified
    times. The unit of the current is pA.

    This is a brainpy.state re-implementation of the NEST simulator device of
    the same name, using NEST-standard parameterization.

    Current output
    ..............

    The device provides a current that is piecewise constant over time:

    .. math::

        I(t) = \begin{cases}
            0                          & \text{if } t < t_1 \\
            a_k                        & \text{if } t_k \le t < t_{k+1}, \; k = 1, \dots, N-1 \\
            a_N                        & \text{if } t \ge t_N
        \end{cases}

    where :math:`t_k` are the amplitude change times and :math:`a_k` are the
    corresponding amplitude values.

    Timing convention
    .................

    In NEST, the amplitude change is applied one simulation step ahead so that
    the new amplitude takes effect at the specified time. This re-implementation
    follows the same convention: the amplitude at time ``t`` is the most recent
    amplitude value whose corresponding time is ``<= t``.

    The active window ``[origin + start, origin + stop)`` gates the output:
    outside this window, the output is zero regardless of the amplitude schedule.

    .. note::

       ``amplitude_times`` must be strictly increasing and positive (> 0).
       ``amplitude_times`` and ``amplitude_values`` must have the same length.

    Parameters
    ----------

    The following parameters can be set. Default values match the NEST simulator.

    ======================= ================== ============================================
    **Parameter**           **Default**        **Description**
    ======================= ================== ============================================
    ``in_size``             1                  Output size of the generator
    ``amplitude_times``     ``[]``             Times at which current changes (list of ms)
    ``amplitude_values``    ``[]``             Amplitudes of step current (list of pA)
    ``start``               0 ms               Activation time relative to ``origin``
    ``stop``                ``None`` (inf)     Deactivation time relative to ``origin``
    ``origin``              0 ms               Global time offset
    ======================= ================== ============================================

    Examples
    --------

    Basic usage:

    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    ...     scg = brainpy.state.step_current_generator(
    ...         amplitude_times=[10. * u.ms, 50. * u.ms, 80. * u.ms],
    ...         amplitude_values=[200. * u.pA, -100. * u.pA, 500. * u.pA],
    ...     )
    ...     neuron = brainpy.state.iaf_psc_delta(1)
    ...     neuron.init_state()
    ...
    ...     for step in range(1000):
    ...         with brainstate.environ.context(t=step * 0.1 * u.ms):
    ...             current = scg.update()
    ...             spk = neuron.update(x=current)

    References
    ----------
    .. [1] NEST Simulator, ``step_current_generator`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/step_current_generator.html

    See Also
    --------
    dc_generator : Constant current generator
    ac_generator : Sinusoidal current generator
    noise_generator : Gaussian white noise current generator
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        amplitude_times: Sequence = (),
        amplitude_values: Sequence = (),
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        # Validate
        if len(amplitude_times) != len(amplitude_values):
            raise ValueError(
                "amplitude_times and amplitude_values must have the same length. "
                f"Got {len(amplitude_times)} and {len(amplitude_values)}."
            )

        # Store amplitude schedule as plain Python lists for easy indexing
        # Convert to float ms and float pA for internal use
        self._amp_times_ms = []
        for t in amplitude_times:
            if u.is_unitless(t):
                self._amp_times_ms.append(float(t))
            else:
                self._amp_times_ms.append(float(t / u.ms))

        self._amp_values_pA = []
        for a in amplitude_values:
            if u.is_unitless(a):
                self._amp_values_pA.append(float(a))
            else:
                self._amp_values_pA.append(float(a / u.pA))

        # Validate strictly increasing times
        for i in range(1, len(self._amp_times_ms)):
            if self._amp_times_ms[i] <= self._amp_times_ms[i - 1]:
                raise ValueError(
                    "amplitude_times must be strictly increasing. "
                    f"Got {self._amp_times_ms[i - 1]} >= {self._amp_times_ms[i]} at index {i}."
                )

        self.start = braintools.init.param(start, self.varshape)
        if stop is not None:
            self.stop = braintools.init.param(stop, self.varshape)
        else:
            self.stop = None
        self.origin = braintools.init.param(origin, self.varshape)

    def update(self):
        """Return the current amplitude at the current simulation time.

        The output is the most recent amplitude value whose corresponding time
        is ``<= t``. If ``t`` is before all amplitude times, the output is zero.
        The output is gated by the active window ``[origin+start, origin+stop)``.

        Returns
        -------
        current : Quantity[pA]
            The output current, shaped ``(in_size,)``.
        """
        t = brainstate.environ.get('t')

        # Get t in ms
        if u.is_unitless(t):
            t_ms = float(t)
        else:
            t_ms = float(t / u.ms)

        # Find the current amplitude based on time
        # NEST applies amplitude one step ahead: at step where
        # curr_time + 1 == amp_time, so by the time we reach amp_time,
        # the amplitude is already set. This means: at time t, the amplitude
        # is the value for the largest amp_time <= t.
        amp_pA = 0.0
        for i in range(len(self._amp_times_ms)):
            if t_ms >= self._amp_times_ms[i]:
                amp_pA = self._amp_values_pA[i]
            else:
                break

        amplitude = amp_pA * u.pA * jnp.ones(self.varshape)

        # Check if device is active
        t_start = self.origin + self.start
        if self.stop is not None:
            t_stop = self.origin + self.stop
            active = u.math.logical_and(t >= t_start, t < t_stop)
        else:
            active = t >= t_start

        return u.math.where(active, amplitude, u.math.zeros_like(amplitude))
