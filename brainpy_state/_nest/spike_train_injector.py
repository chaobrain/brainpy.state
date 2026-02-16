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
    'spike_train_injector',
]


class spike_train_injector(brainstate.nn.Dynamics):
    r"""Spike train injector -- NEST-compatible neuron device.

    Short description
    -----------------

    Neuron that emits prescribed spike trains.

    Description
    -----------

    The spike train injector neuron emits spikes at prescribed spike times
    which are given as an array. The neuron does not allow incoming connections
    and is thus not able to process incoming spikes or currents.

    This is a brainpy.state re-implementation of the NEST simulator device of
    the same name, using NEST-standard parameterization.

    .. note::

       ``spike_train_injector`` is recommended if the spike trains have a
       similar rate to regular neurons.  For very high rates, use
       :class:`spike_generator`.

    Spike times are given in milliseconds as an array. The ``spike_times``
    array must be sorted with the earliest spike first. All spike times must
    be strictly in the future.  Setting a spike time of 0.0 will result in an
    error (unless ``shift_now_spikes`` is ``True``).

    Multiple occurrences of the same time indicate that more than one event is
    to be generated at this particular time.

    Spike time handling options
    ...........................

    The spike train injector supports spike times that do not coincide with a
    time step, that is, are not falling on the grid defined by the simulation
    resolution. Spike times that do not coincide with a step are handled with
    one of three options:

    **Option 1: ``precise_times``** (default: ``False``)

    If ``False``, spike times will be rounded to simulation steps, i.e.
    multiples of the resolution. The rounding is controlled by the two other
    flags.  If ``True``, spike times will not be rounded but represented
    exactly as a combination of step and offset.  This should only be used if
    all neurons receiving the spike train can handle precise timing
    information. In this case, the other two options are ignored.

    **Option 2: ``allow_offgrid_times``** (default: ``False``)

    If ``False``, spike times will be rounded to the nearest step if they are
    less than ``tic/2`` from the step, otherwise an error is raised.  If
    ``True``, spike times are rounded to the nearest step if within ``tic/2``
    from the step, otherwise they are rounded up to the *end* of the step.
    This setting has no effect if ``precise_times`` is ``True``.

    **Option 3: ``shift_now_spikes``** (default: ``False``)

    This option is mainly for use by the PyNN-NEST interface.  If ``False``,
    spike times rounded down to the current point in time will be considered
    in the past and ignored.  If ``True``, spike times that are rounded down
    to the current time step are shifted one time step into the future.

    .. note::

       In this brainpy.state implementation, the three option flags
       (``precise_times``, ``allow_offgrid_times``, ``shift_now_spikes``) are
       accepted for API compatibility but the actual spike time quantisation
       follows a simplified grid-alignment scheme: spike times are always
       rounded to the nearest simulation step using a tolerance of ``dt/2``.
       For typical use cases where spike times are grid-aligned (multiples of
       ``dt``), this produces identical results to NEST.

    Spike multiplicities
    ....................

    Optionally, ``spike_multiplicities`` can be set.  This is a list of
    integers of the same length as ``spike_times``, giving the number of
    spikes to deliver at each time.  In this brainpy.state implementation the
    multiplicity is encoded by emitting an output equal to the multiplicity
    value (rather than 1) at that step.

    Update semantics
    .................

    At each simulation step, the ``update()`` method checks whether any spike
    times match the current simulation time within a tolerance of ``dt/2``.
    If so, the output is 1 (or the multiplicity, or the accumulated
    multiplicity if several spike times map to the same step).  Otherwise the
    output is 0.  The output is gated by the active device window
    ``[origin + start, origin + stop)``.

    Parameters
    ----------

    The following parameters can be set.  Default values match the NEST
    simulator.

    ========================== ================== =============================================
    **Parameter**              **Default**        **Description**
    ========================== ================== =============================================
    ``in_size``                1                  Output size (number of independent injectors)
    ``spike_times``            ``[]``             Spike times in ms (sorted, non-descending)
    ``spike_multiplicities``   ``[]``             Spike multiplicities (same length or empty)
    ``precise_times``          ``False``          Use precise spike timing (API compat.)
    ``allow_offgrid_times``    ``False``          Allow off-grid spike times (API compat.)
    ``shift_now_spikes``       ``False``          Shift now-spikes into future (API compat.)
    ``start``                  0 ms               Activation time relative to ``origin``
    ``stop``                   ``None`` (inf)     Deactivation time relative to ``origin``
    ``origin``                 0 ms               Global time offset
    ========================== ================== =============================================

    Receives
    --------
    None — This device does not accept incoming connections.

    Sends
    -----
    SpikeEvent (output > 0)

    Examples
    --------

    Basic usage:

    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    ...     inj = brainpy.state.spike_train_injector(
    ...         spike_times=[1. * u.ms, 2. * u.ms, 3. * u.ms],
    ...     )
    ...
    ...     for step in range(50):
    ...         with brainstate.environ.context(t=step * 0.1 * u.ms):
    ...             spk = inj.update()

    With spike multiplicities:

    >>> inj = brainpy.state.spike_train_injector(
    ...     spike_times=[1. * u.ms, 2. * u.ms],
    ...     spike_multiplicities=[3, 5],
    ... )

    References
    ----------
    .. [1] NEST Simulator, ``spike_train_injector`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/spike_train_injector.html

    See Also
    --------
    spike_generator : General-purpose spike generator device
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        spike_times: Sequence = (),
        spike_multiplicities: Sequence = (),
        precise_times: bool = False,
        allow_offgrid_times: bool = False,
        shift_now_spikes: bool = False,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        # ---- Validate option flags ----
        if precise_times and (allow_offgrid_times or shift_now_spikes):
            raise ValueError(
                "Option precise_times cannot be set to True when either "
                "allow_offgrid_times or shift_now_spikes is set to True."
            )
        self.precise_times = precise_times
        self.allow_offgrid_times = allow_offgrid_times
        self.shift_now_spikes = shift_now_spikes

        # ---- Convert spike times to ms (float) ----
        self._spike_times_ms = []
        for t in spike_times:
            if u.is_unitless(t):
                self._spike_times_ms.append(float(t))
            else:
                self._spike_times_ms.append(float(t / u.ms))

        # ---- Validate non-descending order ----
        for i in range(1, len(self._spike_times_ms)):
            if self._spike_times_ms[i] < self._spike_times_ms[i - 1]:
                raise ValueError(
                    "spike_times must be sorted in non-descending order. "
                    f"Got {self._spike_times_ms[i - 1]} > {self._spike_times_ms[i]} at index {i}."
                )

        # ---- Validate and store spike multiplicities ----
        if len(spike_multiplicities) > 0 and len(spike_multiplicities) != len(spike_times):
            raise ValueError(
                "spike_multiplicities must have the same number of elements "
                "as spike_times or 0 elements to clear the property. "
                f"Got {len(spike_multiplicities)} and {len(spike_times)}."
            )
        self._spike_multiplicities = [int(m) for m in spike_multiplicities]

        # ---- Device window parameters ----
        self.start = braintools.init.param(start, self.varshape)
        if stop is not None:
            self.stop = braintools.init.param(stop, self.varshape)
        else:
            self.stop = None
        self.origin = braintools.init.param(origin, self.varshape)

    def update(self):
        """Return spike output at the current simulation time.

        Checks if any spike times match the current simulation time within a
        tolerance of ``dt/2``.  If a match is found, returns the spike
        multiplicity (or 1 if multiplicities are not set).  If multiple spike
        times map to the same step, their multiplicities are accumulated.

        The output is gated by the device active window
        ``[origin + start, origin + stop)``.

        Returns
        -------
        spike : jax.Array
            Spike output, shaped ``(in_size,)``.  The value is the
            (accumulated) multiplicity at spike times, 0 otherwise.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()

        # ---- Get t and dt in ms ----
        if u.is_unitless(t):
            t_ms = float(t)
        else:
            t_ms = float(t / u.ms)

        if u.is_unitless(dt):
            dt_ms = float(dt)
        else:
            dt_ms = float(dt / u.ms)

        # ---- Check if device is active ----
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

        # ---- Check for spikes at current time ----
        # A spike at time t_s fires at the simulation step where
        # |t - t_s| < dt/2 (grid-aligned). Multiplicities are accumulated
        # when multiple spike times map to the same step.
        tol = dt_ms / 2.0
        spike_val = 0.0
        for i in range(len(self._spike_times_ms)):
            spike_t = self._spike_times_ms[i]
            if abs(t_ms - spike_t) < tol:
                if self._spike_multiplicities:
                    spike_val += self._spike_multiplicities[i]
                else:
                    spike_val += 1.0

        return spike_val * jnp.ones(self.varshape)
