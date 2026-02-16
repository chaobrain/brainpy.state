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
    'step_rate_generator',
]


class step_rate_generator(brainstate.nn.Dynamics):
    r"""Piecewise-constant rate generator -- NEST-compatible stimulation device.

    Generate a deterministic piecewise-constant rate trace and gate it with a
    half-open activity window using NEST-compatible parameter semantics.

    **1. Model equations and schedule selection**

    Let :math:`\{(t_k, a_k)\}_{k=1}^{K}` be configured change-time/rate pairs,
    where :math:`t_k` are times in ms and :math:`a_k` are rates in spikes/s
    (Hz). The scheduled rate is

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

        r_{\mathrm{out}}(t) = g(t)\,A(t).

    **2. Timing semantics, assumptions, and constraints**

    This implementation chooses, at environment time ``t``, the latest
    schedule entry satisfying ``t_k <= t``. With discrete simulation time on a
    grid, this reproduces NEST-compatible step semantics where a configured
    change time marks the first step at which the new rate is emitted.

    Enforced constraints:

    - ``len(amplitude_times) == len(amplitude_values)``.
    - ``amplitude_times`` are strictly increasing after conversion to float ms.

    Accepted but not additionally constrained:

    - Unitless ``amplitude_times`` are interpreted as ms.
    - Unitless ``amplitude_values`` are interpreted as spikes/s.
    - NEST documentation recommends positive change times; positivity is not
      explicitly enforced here.

    **3. Computational implications**

    :meth:`update` performs a linear scan over ``amplitude_times`` to locate
    the active plateau, then broadcasts one scalar rate over ``self.varshape``
    and applies one boolean activity mask. Per-call complexity is
    :math:`O(K + \prod \mathrm{varshape})`, where :math:`K` is the number of
    schedule entries.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification consumed by
        :class:`brainstate.nn.Dynamics`. The emitted rate has shape
        ``self.varshape`` derived from ``in_size``. Default is ``1``.
    amplitude_times : Sequence[ArrayLike], optional
        Ordered sequence of change times with length ``K``. Each value may be
        a unitful time (typically ms) or a unitless numeric interpreted as ms.
        Internally converted to plain ``float`` milliseconds and stored as a
        Python list. Must be strictly increasing. Default is ``()``.
    amplitude_values : Sequence[ArrayLike], optional
        Sequence of plateau rates with length ``K`` matching
        ``amplitude_times`` elementwise. Values represent spikes/s (Hz) and
        may be unitful or unitless. Internally converted to plain ``float``
        values and stored as a Python list. Default is ``()``.
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
         - Change times (ms) for piecewise-constant rate plateaus.
       * - ``amplitude_values``
         - ``()``
         - :math:`a_k`
         - Plateau rates (spikes/s) selected at and after each ``t_k``.
       * - ``start``
         - ``0. * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative inclusive lower bound of the active output window.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative exclusive upper bound of the active output window.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global time offset added to ``start`` and ``stop``.

    Returns
    -------
    out : Any
        Dynamics node. Calling :meth:`update` returns a rate-like array with
        shape ``self.varshape`` and values in spikes/s: scheduled plateau while
        active and zeros outside the activity window.

    Raises
    ------
    ValueError
        If ``amplitude_times`` and ``amplitude_values`` lengths differ, or if
        ``amplitude_times`` is not strictly increasing after conversion to ms.
    TypeError
        If unitful/unitless arithmetic is invalid during schedule conversion,
        parameter broadcasting, or time-window comparisons.
    KeyError
        At update time, if simulation time ``'t'`` is missing from
        ``brainstate.environ``.

    See Also
    --------
    step_current_generator : Piecewise-constant current stimulation device.
    dc_generator : Constant current stimulation device.
    inhomogeneous_poisson_generator : Stochastic rate-to-spike generator.

    References
    ----------
    .. [1] NEST Simulator documentation for ``step_rate_generator``:
           https://nest-simulator.readthedocs.io/en/stable/models/step_rate_generator.html

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.step_rate_generator(
       ...         amplitude_times=[10.0 * u.ms, 110.0 * u.ms, 210.0 * u.ms],
       ...         amplitude_values=[400.0, 1000.0, 200.0],
       ...         start=0.0 * u.ms,
       ...         stop=300.0 * u.ms,
       ...     )
       ...     with brainstate.environ.context(t=160.0 * u.ms):
       ...         rate = gen.update()
       ...     _ = rate.shape
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

        # Store amplitude schedule as plain Python lists
        self._amp_times_ms = []
        for t in amplitude_times:
            if u.is_unitless(t):
                self._amp_times_ms.append(float(t))
            else:
                self._amp_times_ms.append(float(t / u.ms))

        self._amp_values = []
        for a in amplitude_values:
            if u.is_unitless(a):
                self._amp_values.append(float(a))
            else:
                # Rate values may have Hz units or be dimensionless
                self._amp_values.append(float(a))

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
        r"""Compute scheduled rate at environment time ``t``.

        Parameters
        ----------
        None
            Uses current simulation time ``t`` from ``brainstate.environ`` and
            parameters initialized in :meth:`__init__`.

        Returns
        -------
        out : Any
            Rate-like quantity with shape ``self.varshape``. For each output
            channel, value equals the latest scheduled plateau whose change
            time is ``<= t``. Channels outside the active window
            ``[origin + start, origin + stop)`` are set to zero (or
            ``t >= origin + start`` when ``stop is None``).

        Raises
        ------
        KeyError
            If ``brainstate.environ`` has no ``'t'`` entry.
        TypeError
            If provided times cannot be compared because of incompatible
            units/dtypes.
        ValueError
            If conversion of ``t`` to milliseconds fails.

        Notes
        -----
        Start is inclusive and stop is exclusive. If ``stop <= start`` (after
        adding ``origin``), the active set is empty and the output is always
        zero regardless of the scheduled plateaus.
        """
        t = brainstate.environ.get('t')

        # Get t in ms
        if u.is_unitless(t):
            t_ms = float(t)
        else:
            t_ms = float(t / u.ms)

        # Find the current rate based on time
        rate = 0.0
        for i in range(len(self._amp_times_ms)):
            if t_ms >= self._amp_times_ms[i]:
                rate = self._amp_values[i]
            else:
                break

        rate_arr = rate * jnp.ones(self.varshape)

        # Check if device is active
        t_start = self.origin + self.start
        if self.stop is not None:
            t_stop = self.origin + self.stop
            active = u.math.logical_and(t >= t_start, t < t_stop)
        else:
            active = t >= t_start

        # For rate, we use dimensionless values
        if u.is_unitless(active):
            return jnp.where(active, rate_arr, jnp.zeros_like(rate_arr))
        else:
            return jnp.where(active, rate_arr, jnp.zeros_like(rate_arr))
