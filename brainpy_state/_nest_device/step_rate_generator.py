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
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base._base import NESTDevice
from brainpy_state._nest_base._utils import stack_schedule_values

__all__ = [
    'step_rate_generator',
]


class step_rate_generator(NESTDevice):
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
    - ``amplitude_times`` are strictly increasing.

    Accepted but not additionally constrained:

    - Unitless ``amplitude_times`` are interpreted as ms.
    - Unitless ``amplitude_values`` are interpreted as spikes/s.
    - NEST documentation recommends positive change times; positivity is not
      explicitly enforced here.

    **3. Computational implications**

    Each :meth:`update` call uses :func:`u.math.searchsorted` to find the
    active plateau, then selects the pre-broadcast rate array for
    ``self.varshape`` and applies one boolean activity mask. Per-call
    complexity is :math:`O(\log K + \prod \mathrm{varshape})`, where
    :math:`K` is the number of schedule entries.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification consumed by
        :class:`brainstate.nn.Dynamics`. The emitted rate has shape
        ``self.varshape`` derived from ``in_size``. Default is ``1``.
    amplitude_times : Sequence, optional
        Ordered sequence of change times with length ``K``. Each value may be
        a unitful time (typically ms) or a unitless numeric. Passed directly to
        :func:`u.math.asarray`, which validates unit consistency across all
        entries. Must be strictly increasing. Default is ``()``.
    amplitude_values : Sequence, optional
        Sequence of plateau rates with length ``K`` matching
        ``amplitude_times`` elementwise. Values represent spikes/s (Hz) and
        may be unitful or unitless. Each entry is converted via
        :func:`u.math.asarray` and expanded to the maximum ndim found across
        all entries (by prepending size-1 axes); the results are stacked to a
        shape that is broadcastable to ``(K, *varshape)``. Default is ``()``.
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
         - Change times for piecewise-constant rate plateaus.
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

    Raises
    ------
    ValueError
        If ``amplitude_times`` and ``amplitude_values`` lengths differ, or if
        ``amplitude_times`` is not strictly increasing.
    TypeError
        If :func:`u.math.asarray` detects unit inconsistency across entries,
        or if unitful/unitless arithmetic is invalid during broadcasting or
        time-window comparisons.
    KeyError
        At update time, if simulation time ``'t'`` is missing from
        ``brainstate.environ``.

    Notes
    -----
    NEST recommends specifying ``amplitude_times`` on a grid of simulation
    resolution ``dt``. Using off-grid change times is allowed but may shift
    the effective change by up to one ``dt`` step depending on floating-point
    rounding when comparing ``t >= amp_time``. Use ``dc_generator``
    when only a constant current drive is needed; use ``step_rate_generator``
    when a rate-coded drive must take different values at different simulation
    intervals. Unlike ``step_current_generator``, the emitted quantity is
    dimensionless (spikes/s) and is not multiplied by a unit before output.

    See Also
    --------
    step_current_generator : Piecewise-constant current stimulation device.
    dc_generator : Constant current stimulation device.
    ac_generator : Sinusoidal current stimulation device.

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

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> gen1 = brainpy.state.step_rate_generator(
       ...     amplitude_times=[0.0 * u.ms, 100.0 * u.ms, 200.0 * u.ms],
       ...     amplitude_values=[50.0, 0.0, 80.0],
       ... )
       >>> gen2 = brainpy.state.step_rate_generator(
       ...     in_size=10,
       ...     amplitude_times=[50.0 * u.ms, 150.0 * u.ms],
       ...     amplitude_values=[120.0, 40.0],
       ...     start=40.0 * u.ms,
       ...     stop=180.0 * u.ms,
       ...     origin=10.0 * u.ms,
       ... )
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

        if len(amplitude_times) != len(amplitude_values):
            raise ValueError(
                "amplitude_times and amplitude_values must have the same length. "
                f"Got {len(amplitude_times)} and {len(amplitude_values)}."
            )
        assert len(amplitude_times) > 0, "At least one schedule entry is required. Got len(amplitude_times) = 0."

        # Store amplitude_times as a Quantity array; u.math.asarray validates
        # that all entries share a consistent unit.
        # Shape: (K,)
        self.amplitude_times = u.math.asarray(amplitude_times)

        # Validate strictly increasing times before storing.
        for i in range(1, len(self.amplitude_times)):
            if self.amplitude_times[i] <= self.amplitude_times[i - 1]:
                raise ValueError(
                    "amplitude_times must be strictly increasing. "
                    f"Got {self.amplitude_times[i - 1]} >= {self.amplitude_times[i]} at index {i}."
                )

        self.amplitude_values = stack_schedule_values(amplitude_values, self.varshape)

        self.start = braintools.init.param(start, self.varshape)
        self.stop = None if stop is None else braintools.init.param(stop, self.varshape)
        self.origin = braintools.init.param(origin, self.varshape)

    def update(self):
        r"""Compute scheduled rate at environment time ``t``.

        The implementation is fully compatible with ``jax.jit``: the schedule
        look-up uses :func:`u.math.searchsorted` on the static
        ``amplitude_times`` array, while ``t`` remains a traced value
        throughout.

        Returns
        -------
        out : jax.Array
            Dimensionless rate array with shape ``self.varshape`` and values in
            spikes/s. For each output channel, value equals the latest
            scheduled plateau whose change time is ``<= t``. Channels outside
            the active window ``[origin + start, origin + stop)`` are set to
            zero (or ``t >= origin + start`` when ``stop is None``).

        Raises
        ------
        KeyError
            If ``brainstate.environ`` has no ``'t'`` entry.

        Notes
        -----
        Both ``amplitude_times`` and ``t`` are divided by ``u.ms`` to obtain
        dimensionless arrays before calling :func:`u.math.searchsorted`.
        ``u.math.searchsorted(..., side='right') - 1`` returns the index of
        the most-recently-passed change point, or ``-1`` when ``t`` precedes
        all change times (zero rate).  :func:`u.math.clip` keeps the index in
        bounds for the gather; :func:`u.math.where` then suppresses the result
        when the index is negative.  Start is inclusive and stop is exclusive,
        matching NEST semantics.

        See Also
        --------
        step_rate_generator : Class-level parameter definitions and model equations.
        step_current_generator.update : Windowed piecewise-constant current update rule.
        dc_generator.update : Windowed constant-current update rule.
        """
        t = brainstate.environ.get('t')
        # zeros has shape varshape so that u.math.where always broadcasts the
        # selected rate value to the full output shape.
        zeros = u.math.zeros(self.varshape, unit=u.get_unit(self.amplitude_values))

        if len(self.amplitude_times) == 0:
            # No schedule entries: output is always zero.
            return zeros

        # Divide both by u.ms to obtain dimensionless arrays for searchsorted.
        # amplitude_times is a static array (compile-time constant under jit);
        # t_dimless is the only traced value in the look-up.
        t_dimless = u.math.asarray(t / u.ms)
        times_dimless = u.math.asarray(self.amplitude_times / u.ms)

        # Last index k such that amplitude_times[k] <= t, or -1 if none.
        idx = u.math.searchsorted(times_dimless, t_dimless, side='right') - 1

        # Clamp to a valid index for the gather (idx=-1 is handled by where).
        safe_idx = u.math.clip(idx, 0, self.amplitude_values.shape[0] - 1)

        # amplitude_values has shape (K, *broadcast_shape); indexing with a scalar
        # safe_idx yields shape (*broadcast_shape,) broadcastable to varshape.
        rate = u.math.where(idx >= 0, self.amplitude_values[safe_idx], zeros)

        # NEST-compatible half-open activity window [origin+start, origin+stop).
        t_start = self.origin + self.start
        if self.stop is not None:
            t_stop = self.origin + self.stop
            active = u.math.logical_and(t >= t_start, t < t_stop)
        else:
            active = t >= t_start

        return u.math.where(active, rate, u.math.zeros_like(rate))
