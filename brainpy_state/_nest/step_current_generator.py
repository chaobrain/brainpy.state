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

from ._base import NESTDevice
from ._utils import stack_schedule_values

__all__ = [
    'step_current_generator',
]


class step_current_generator(NESTDevice):
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
    - ``amplitude_times`` are strictly increasing.

    Inputs accepted but not explicitly constrained:

    - Unitless ``amplitude_times`` are interpreted as ms.
    - Unitless ``amplitude_values`` are interpreted as pA.
    - Positive-time-only schedules are recommended by NEST, but positivity is
      not explicitly validated here.

    **3. Computational implications**

    Each :meth:`update` call uses :func:`u.math.searchsorted` to find the
    active plateau, then selects the pre-broadcast current array for
    ``self.varshape`` and applies one boolean mask. Per-call complexity is
    :math:`O(\log K + \prod \mathrm{varshape})`, with :math:`K` schedule entries.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification consumed by
        :class:`brainstate.nn.Dynamics`. The emitted current has shape
        ``self.varshape`` derived from ``in_size``. Default is ``1``.
    amplitude_times : Sequence, optional
        Ordered sequence of change times with length ``K``. Entries may be
        unitful times (typically ms) or unitless numerics. Passed directly to
        :func:`u.math.asarray`, which validates unit consistency across all
        entries. Must be strictly increasing. Default is ``()``.
    amplitude_values : Sequence, optional
        Sequence of current plateaus with length ``K`` matching
        ``amplitude_times`` elementwise. Entries may be unitful currents
        (typically pA) or unitless numerics. Each entry is converted via
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
         - Change times for piecewise-constant plateaus.
       * - ``amplitude_values``
         - ``()``
         - :math:`a_k`
         - Plateau currents selected at and after corresponding ``t_k``.
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
    resolution ``dt``.  Using off-grid change times is allowed but may shift
    the effective change by up to one ``dt`` step depending on floating-point
    rounding when comparing ``t >= amp_time``.  Use ``dc_generator``
    when only a single constant plateau is needed; ``step_current_generator``
    is the preferred device when the current must take different values at
    different intervals within a single simulation run.

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

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> stim1 = brainpy.state.step_current_generator(
       ...     amplitude_times=[0.0 * u.ms, 100.0 * u.ms, 200.0 * u.ms],
       ...     amplitude_values=[300.0 * u.pA, 0.0 * u.pA, -150.0 * u.pA],
       ... )
       >>> stim2 = brainpy.state.step_current_generator(
       ...     in_size=10,
       ...     amplitude_times=[50.0 * u.ms, 150.0 * u.ms],
       ...     amplitude_values=[400.0 * u.pA, 100.0 * u.pA],
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
        r"""Compute scheduled current at environment time ``t``.

        The implementation is fully compatible with ``jax.jit``: the schedule
        look-up uses :func:`u.math.searchsorted` on the static ``amplitude_times``
        array, while ``t`` remains a traced value throughout.

        Returns
        -------
        out : Quantity
            Current quantity with shape ``self.varshape``. For each output
            channel, value equals the latest scheduled plateau whose change time
            is ``<= t``; channels outside the active window
            ``[origin + start, origin + stop)`` are set to zero (or all
            active when ``stop is None``).

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
        all change times (zero current).  :func:`u.math.clip` keeps the index
        in bounds for the gather; :func:`u.math.where` then suppresses the
        result when the index is negative.  Start is inclusive and stop is
        exclusive, matching NEST semantics.

        See Also
        --------
        step_current_generator : Class-level parameter definitions and model equations.
        dc_generator.update : Windowed constant-current update rule.
        ac_generator.update : Windowed sinusoidal-current update rule.

        """
        t = brainstate.environ.get('t')
        zeros = u.math.zeros(self.amplitude_values.shape[1:], unit=u.get_unit(self.amplitude_values))

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
        amplitude = u.math.where(idx >= 0, self.amplitude_values[safe_idx], zeros)

        # NEST-compatible half-open activity window [origin+start, origin+stop).
        t_start = self.origin + self.start
        if self.stop is not None:
            t_stop = self.origin + self.stop
            active = u.math.logical_and(t >= t_start, t < t_stop)
        else:
            active = t >= t_start

        return u.math.where(active, amplitude, u.math.zeros_like(amplitude))
