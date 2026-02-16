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
    r"""Piecewise-constant current generator -- NEST-compatible stimulation device.

    Generate a deterministic piecewise-constant current trace and gate it with
    a half-open activity window using NEST-compatible time semantics.

    **1. Model equations**

    Let :math:`\{(t_k, a_k)\}_{k=1}^{K}` be the configured change-time/current
    pairs, where :math:`t_k` are times (ms) and :math:`a_k` are currents (pA).
    Define the scheduled amplitude

    .. math::

        A(t) =
        \begin{cases}
            0, & t < t_1, \\
            a_k, & t_k \le t < t_{k+1},\ k=1,\dots,K-1, \\
            a_K, & t \ge t_K.
        \end{cases}

    The output is gated by

    .. math::

        g(t) = \mathbf{1}\!\left[t \ge t_0+t_{\mathrm{start,rel}}\right]
        \cdot
        \mathbf{1}\!\left[t < t_0+t_{\mathrm{stop,rel}}\right],

    with the second indicator omitted when ``stop is None``. Final output:

    .. math::

        I(t) = g(t)\,A(t).

    **2. Timing semantics, assumptions, and constraints**

    NEST timing is matched by selecting, at time ``t``, the most recent change
    point with ``t_k <= t``. In discrete simulation with step ``dt``, this
    corresponds to applying a change exactly from the step whose environment
    time equals the configured change time.

    Enforced constraints in this implementation:

    - ``len(amplitude_times) == len(amplitude_values)``.
    - ``amplitude_times`` are strictly increasing after conversion to float ms.

    Inputs accepted but not explicitly constrained:

    - Unitless ``amplitude_times`` are interpreted as ms.
    - Unitless ``amplitude_values`` are interpreted as pA.
    - Positive-time-only schedules are recommended by NEST, but positivity is
      not explicitly validated here.

    **3. Computational implications**

    Each :meth:`update` call scans ``amplitude_times`` linearly to find the
    active plateau, then broadcasts one scalar current over ``self.varshape``
    and applies one boolean mask. Per-call complexity is
    :math:`O(K + \prod \mathrm{varshape})`, with :math:`K` schedule entries.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification consumed by
        :class:`brainstate.nn.Dynamics`. The emitted current has shape
        ``self.varshape`` derived from ``in_size``. Default is ``1``.
    amplitude_times : Sequence, optional
        Ordered sequence of change times with length ``K``. Entries may be
        unitful times (typically ms) or unitless numerics interpreted as ms.
        Internally, each entry is converted to ``float(t / u.ms)`` (or
        ``float(t)`` when unitless). Must be strictly increasing. Default is
        ``()``.
    amplitude_values : Sequence, optional
        Sequence of current plateaus with length ``K`` matching
        ``amplitude_times`` elementwise. Entries may be unitful currents
        (typically pA) or unitless numerics interpreted as pA. Internally
        converted to plain floats in pA. Default is ``()``.
    start : ArrayLike, optional
        Relative start time :math:`t_{\mathrm{start,rel}}` (typically ms),
        broadcast to ``self.varshape`` via :func:`braintools.init.param`.
        Effective lower bound is ``origin + start`` (inclusive).
        Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative stop time :math:`t_{\mathrm{stop,rel}}` (typically ms),
        broadcast to ``self.varshape`` when provided. Effective upper bound is
        ``origin + stop`` (exclusive). ``None`` means no upper bound.
        Default is ``None``.
    origin : ArrayLike, optional
        Time origin :math:`t_0` (typically ms) added to ``start`` and ``stop``,
        broadcast to ``self.varshape``. Default is ``0. * u.ms``.
    name : str or None, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 22 18 22 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``amplitude_times``
         - ``()``
         - :math:`t_k`
         - Change times (ms) for piecewise-constant plateaus.
       * - ``amplitude_values``
         - ``()``
         - :math:`a_k`
         - Plateau currents (pA) selected at and after corresponding ``t_k``.
       * - ``start``
         - ``0. * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative inclusive lower bound of activity window.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative exclusive upper bound of activity window.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global offset added to ``start`` and ``stop``.

    Returns
    -------
    out : Any
        Dynamics node. Calling :meth:`update` returns a current-like array with
        shape ``self.varshape`` and current units (typically pA), equal to the
        scheduled plateau while active and zero otherwise.

    Raises
    ------
    ValueError
        If ``amplitude_times`` and ``amplitude_values`` lengths differ, or if
        ``amplitude_times`` is not strictly increasing after conversion to ms.
    TypeError
        If unitful/unitless arithmetic is invalid during conversion,
        broadcasting, or time-window comparisons.
    KeyError
        At update time, if simulation time ``'t'`` is missing from
        ``brainstate.environ``.

    See Also
    --------
    dc_generator : Constant current stimulation device.
    ac_generator : Sinusoidal current stimulation device.
    noise_generator : Gaussian white-noise current stimulation device.

    References
    ----------
    .. [1] NEST Simulator documentation for ``step_current_generator``:
           https://nest-simulator.readthedocs.io/en/stable/models/step_current_generator.html

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     stim = brainpy.state.step_current_generator(
       ...         in_size=1,
       ...         amplitude_times=[10.0 * u.ms, 50.0 * u.ms, 80.0 * u.ms],
       ...         amplitude_values=[200.0 * u.pA, -100.0 * u.pA, 500.0 * u.pA],
       ...         start=5.0 * u.ms,
       ...         stop=120.0 * u.ms,
       ...     )
       ...     with brainstate.environ.context(t=60.0 * u.ms):
       ...         current = stim.update()
       ...     _ = current.shape
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
        r"""Compute scheduled current at environment time ``t``.

        Parameters
        ----------
        None
            Uses the current simulation time from ``brainstate.environ['t']``
            and instance parameters initialized in :meth:`__init__`.

        Returns
        -------
        out : Any
            Current-like quantity with shape ``self.varshape``. For each output
            channel, value equals the latest scheduled plateau whose change time
            is ``<= t``; channels outside the active window
            ``[origin + start, origin + stop)`` are set to zero (or
            ``t >= origin + start`` when ``stop is None``).

        Raises
        ------
        KeyError
            If ``brainstate.environ`` has no ``'t'`` entry.
        TypeError
            If provided time values cannot be compared because of incompatible
            units or dtypes.
        ValueError
            If conversion of schedule entries to floating-point ms/pA fails.

        Notes
        -----
        The schedule lookup is linear in ``len(amplitude_times)``. This is
        efficient for short schedules and preserves straightforward NEST-like
        semantics without interpolation.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate
           >>> import brainunit as u
           >>> from brainpy.state import step_current_generator
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     gen = step_current_generator(
           ...         amplitude_times=[2.0 * u.ms, 4.0 * u.ms],
           ...         amplitude_values=[150.0 * u.pA, -50.0 * u.pA],
           ...     )
           ...     with brainstate.environ.context(t=3.0 * u.ms):
           ...         current = gen.update()
           ...     _ = current.shape
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
