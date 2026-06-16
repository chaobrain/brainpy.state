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

from brainpy_state._nest_base.base import NESTDevice

__all__ = [
    'spike_train_injector',
]


class spike_train_injector(NESTDevice):
    r"""Spike train injector -- NEST-compatible event source device.

    Emit deterministic spike events at configured times with optional
    per-time multiplicity, then gate output by a half-open activity window.
    Unlike :class:`spike_generator`, which selects the last matching weight,
    this device *accumulates* all multiplicities that match the current step,
    making it suitable for injecting pre-recorded spike trains where multiple
    events may be scheduled at the same simulation time.

    **1. Model equations**

    Let :math:`\{t_i\}_{i=1}^{K}` be configured spike times in ms after
    conversion from unitful or unitless inputs. Let :math:`m_i` denote
    multiplicity (``spike_multiplicities``) when provided, otherwise
    :math:`m_i = 1`. At simulation time :math:`t` with step :math:`\Delta t`
    (both in ms), define the matching indicator

    .. math::

        q_i(t) = \mathbf{1}\!\left[|t - t_i| < \frac{\Delta t}{2}\right].

    The scalar emitted spike count before window gating is

    .. math::

        a(t) = \sum_{i=1}^{K} m_i\, q_i(t).

    The activity gate is

    .. math::

        g(t) =
        \mathbf{1}\!\left[t \ge t_0 + t_{\mathrm{start,rel}}\right]
        \cdot
        \mathbf{1}\!\left[t < t_0 + t_{\mathrm{stop,rel}}\right],

    where the second indicator is omitted when ``stop is None``.
    The returned output is broadcast to node shape ``self.varshape``:

    .. math::

        y(t) = g(t)\,a(t)\,\mathbf{1}_{\mathrm{varshape}}.

    **2. Timing derivation, assumptions, and constraints**

    The :math:`|t - t_i| < \Delta t / 2` rule corresponds to nearest-grid
    assignment under uniform-step simulation. For exact half-step offsets,
    strict inequality means no match at that boundary. If multiple
    ``spike_times`` entries map to the same step, their multiplicities are
    *summed*, giving :math:`a(t) > 1` for bursts.

    Enforced constraints:

    - ``spike_times`` must be non-descending after conversion.
    - ``spike_multiplicities`` must be empty or have exactly
      ``len(spike_multiplicities) == len(spike_times)`` elements.
    - ``precise_times=True`` cannot be combined with
      ``allow_offgrid_times=True`` or ``shift_now_spikes=True``.

    Implementation-specific constraints:

    - NEST option flags ``precise_times``, ``allow_offgrid_times``, and
      ``shift_now_spikes`` are accepted for API compatibility but the current
      update rule always uses the fixed tolerance test above regardless of
      their values.
    - NEST documentation states spikes should be strictly in the future. This
      implementation does not perform explicit future-time validation in
      :meth:`__init__` and instead relies on runtime matching combined with
      active-window gating.

    **3. Computational implications**

    Each :meth:`update` call uses :func:`u.math.searchsorted` to locate the
    open interval :math:`(t - \Delta t/2,\, t + \Delta t/2)` in the sorted
    ``spike_times`` array, yielding a range :math:`[\textit{idx\_lo},
    \textit{idx\_hi})` of matching indices. A Boolean mask over
    :math:`\{0,\ldots,K-1\}` is then used to sum the multiplicities of all
    matching entries. Per-call complexity is :math:`O(\log K + K + \prod
    \mathrm{varshape})`.  The :meth:`update` method is fully compatible with
    ``jax.jit``: no Python control flow branches on traced values.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape consumed by :class:`brainstate.nn.Dynamics`. The
        emitted array has shape ``self.varshape`` derived from ``in_size``.
        Default is ``1``.
    spike_times : Sequence, optional
        Sequence of spike times with length ``K``. Entries may be unitful
        times (typically ``brainunit`` ms quantities) or bare numerics
        interpreted as ms. Passed directly to :func:`u.math.asarray`, which
        validates unit consistency across all entries. Must be non-descending.
        Duplicate times are allowed and their multiplicities are accumulated.
        Default is ``()``.
    spike_multiplicities : Sequence, optional
        Sequence of integer multiplicities with length ``K`` matching
        ``spike_times``, or empty to use implicit unit multiplicities
        (:math:`m_i = 1`). Entries are converted with ``int(m)`` and stored
        as a dimensionless JAX array; accumulated across all indices matching
        the same step. Default is ``()``.
    precise_times : bool, optional
        NEST compatibility flag for sub-step precise timing. Stored and
        validated against ``allow_offgrid_times`` / ``shift_now_spikes`` but
        not used to alter runtime matching in this implementation.
        Default is ``False``.
    allow_offgrid_times : bool, optional
        NEST compatibility flag permitting off-grid spike times. Stored and
        validated but not used to alter runtime matching in this
        implementation. Default is ``False``.
    shift_now_spikes : bool, optional
        NEST compatibility flag for shifting spikes that would fire at the
        current step to the next. Stored and validated but not used to alter
        runtime matching in this implementation. Default is ``False``.
    start : ArrayLike, optional
        Relative activation time :math:`t_{\mathrm{start,rel}}` (typically
        ms), initialized via :func:`braintools.init.param`. The effective
        inclusive lower bound of the active window is ``origin + start``.
        Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative deactivation time :math:`t_{\mathrm{stop,rel}}` (typically
        ms), initialized via :func:`braintools.init.param` when not ``None``.
        The effective exclusive upper bound is ``origin + stop``. ``None``
        disables the upper bound. Default is ``None``.
    origin : ArrayLike, optional
        Global time origin :math:`t_0` (typically ms) added to both ``start``
        and ``stop`` to obtain absolute window bounds. Default is ``0. * u.ms``.
    name : str or None, optional
        Optional node name forwarded to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 22 18 22 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``spike_times``
         - ``()``
         - :math:`t_i`
         - Spike schedule; matched by ``|t - t_i| < dt/2``.
       * - ``spike_multiplicities``
         - ``()``
         - :math:`m_i`
         - Per-time spike count; empty means implicit :math:`m_i = 1`.
       * - ``start``
         - ``0. * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative inclusive lower bound of active window.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative exclusive upper bound; ``None`` means unbounded.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global offset applied to ``start`` and ``stop``.

    Raises
    ------
    ValueError
        If ``precise_times=True`` is combined with ``allow_offgrid_times=True``
        or ``shift_now_spikes=True``, if ``spike_times`` is not non-descending
        after conversion, or if ``spike_multiplicities`` is non-empty and has
        a different length than ``spike_times``.
    TypeError
        If :func:`u.math.asarray` detects unit inconsistency across entries,
        or if unitful/unitless arithmetic is invalid during time-window
        comparisons.
    KeyError
        At update time, if required simulation context entries (e.g. ``'t'``
        or ``dt``) are absent from ``brainstate.environ``.

    Notes
    -----
    This device does not accept incoming synaptic or current connections; it
    only emits scheduled events. The output is dimensionless (spike count per
    step) and is typically consumed by a downstream synapse model that scales
    by connection weight.

    The key behavioral difference from :class:`spike_generator` is
    *accumulation*: when two entries in ``spike_times`` round to the same
    step, ``spike_train_injector`` sums their multiplicities while
    ``spike_generator`` retains only the last matching weight. Use
    ``spike_train_injector`` when replaying recorded spike trains that may
    contain bursts, and ``spike_generator`` when a single weighted event per
    step is intended.

    Spike times should ideally be aligned to the simulation grid (multiples
    of ``dt``) to avoid off-by-one steps. The tolerance ``dt/2`` covers
    one-ULP rounding for grid-aligned times in typical float64 arithmetic.

    See Also
    --------
    spike_generator : Deterministic spike device with per-spike weights
        (last-match semantics).
    dc_generator : Constant-current stimulation device.
    ac_generator : Sinusoidal current stimulation device.
    step_current_generator : Piecewise-constant current stimulation device.

    References
    ----------
    .. [1] NEST Simulator, ``spike_train_injector`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/spike_train_injector.html

    Examples
    --------
    Inject a burst of five spikes at ``t = 2 ms`` (two entries map to the same
    step, multiplicities are accumulated to give ``a = 2 + 3 = 5``):

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     inj = brainpy.state.spike_train_injector(
       ...         spike_times=[1.0 * u.ms, 2.0 * u.ms, 2.0 * u.ms],
       ...         spike_multiplicities=[1, 2, 3],
       ...         start=0.0 * u.ms,
       ...         stop=5.0 * u.ms,
       ...     )
       ...     with brainstate.environ.context(t=2.0 * u.ms):
       ...         out = inj.update()
       ...     _ = out.shape

    Inject a single spike at ``t = 10 ms`` using NEST's ``precise_times``
    flag for API compatibility (sub-step resolution not enforced here):

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     inj = brainpy.state.spike_train_injector(
       ...         spike_times=[10.0 * u.ms],
       ...         precise_times=True,
       ...     )
       ...     with brainstate.environ.context(t=10.0 * u.ms):
       ...         out = inj.update()
       ...     _ = out.shape
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

        # ---- Store spike times as a Quantity array ----
        # u.math.asarray validates unit consistency across all entries.
        # Plain floats are interpreted as milliseconds.
        if len(spike_times) > 0:
            self.spike_times = u.math.asarray(spike_times)
            if not isinstance(self.spike_times, u.Quantity):
                self.spike_times = self.spike_times * u.ms

            # Validate non-descending order.
            for i in range(1, len(self.spike_times)):
                if self.spike_times[i] < self.spike_times[i - 1]:
                    raise ValueError(
                        "spike_times must be sorted in non-descending order. "
                        f"Got {self.spike_times[i - 1]} > {self.spike_times[i]} at index {i}."
                    )
        else:
            self.spike_times = None

        # ---- Validate and store spike multiplicities as a JAX array ----
        if len(spike_multiplicities) > 0 and len(spike_multiplicities) != len(spike_times):
            raise ValueError(
                "spike_multiplicities must have the same number of elements "
                "as spike_times or 0 elements to clear the property. "
                f"Got {len(spike_multiplicities)} and {len(spike_times)}."
            )
        if len(spike_multiplicities) > 0:
            self.spike_multiplicities = u.math.asarray([int(m) for m in spike_multiplicities], dtype=float)
        else:
            self.spike_multiplicities = None

        # ---- Device window parameters ----
        self.start = braintools.init.param(start, self.varshape)
        self.stop = None if stop is None else braintools.init.param(stop, self.varshape)
        self.origin = braintools.init.param(origin, self.varshape)

    def update(self):
        r"""Compute the accumulated spike output for the current simulation step.

        The implementation is fully compatible with ``jax.jit``: spike-time
        matching uses :func:`u.math.searchsorted` on the static
        ``spike_times`` array while ``t`` and ``dt`` remain traced values
        throughout. The multiplicity sum uses a Boolean mask with no Python
        branching over traced values.

        Returns
        -------
        out : jax.Array
            Float-valued JAX array with shape ``self.varshape``.
            Output semantics:

            - ``0`` when outside ``[origin + start, origin + stop)`` (or
              ``[origin + start, +inf)`` if ``stop is None``),
            - ``0`` when active but no configured spike satisfies
              ``|t - t_i| < dt/2``,
            - accumulated integer multiplicity :math:`a(t) = \sum_i m_i\,
              q_i(t)` when active and one or more spikes match.

        Raises
        ------
        KeyError
            If required simulation context entries are missing from
            ``brainstate.environ`` (e.g. ``'t'`` or ``dt``).

        Notes
        -----
        Matching uses the open interval :math:`(t - \Delta t/2,\, t +
        \Delta t/2)` located via two :func:`u.math.searchsorted` calls:

        - ``idx_lo = searchsorted(times, t - dt/2, side='right')`` — first
          index strictly greater than the lower bound.
        - ``idx_hi = searchsorted(times, t + dt/2, side='left')`` — first
          index at or above the upper bound.

        A Boolean mask ``indices in [idx_lo, idx_hi)`` selects all matching
        entries; their multiplicities (or 1s if none configured) are summed
        to obtain the scalar count :math:`a(t)`. Start is inclusive and stop
        is exclusive, matching NEST semantics.

        Unlike :meth:`spike_generator.update`, which keeps only the last
        matching weight, this method *accumulates* all matching multiplicities.
        A burst of three spikes scheduled at the same time thus returns ``3``
        (or the sum of their individual multiplicities).

        See Also
        --------
        spike_train_injector : Class-level parameter definitions and equations.
        spike_generator.update : Weight-selection (last-match) update rule.
        dc_generator.update : Windowed constant-current update rule.
        step_current_generator.update : Windowed piecewise-constant update rule.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()

        zeros = u.math.zeros(self.varshape)

        if self.spike_times is None:
            return zeros

        # Locate the open interval (t - dt/2, t + dt/2) via two searchsorted calls.
        # idx_lo: first index where spike_times > t - dt/2  (side='right')
        # idx_hi: first index where spike_times >= t + dt/2 (side='left')
        # Matching range is [idx_lo, idx_hi).
        idx_lo = u.math.searchsorted(self.spike_times, t - dt / 2, side='right')
        idx_hi = u.math.searchsorted(self.spike_times, t + dt / 2, side='left')

        # Build a Boolean mask over all K spike indices for the matching range.
        K = self.spike_times.shape[0]
        indices = u.math.arange(K)
        in_range = u.math.logical_and(indices >= idx_lo, indices < idx_hi)

        # Sum multiplicities (or 1s) over all matching indices.
        if self.spike_multiplicities is not None:
            spike_val = u.math.sum(u.math.where(in_range, self.spike_multiplicities, 0.0))
        else:
            spike_val = u.math.sum(in_range.astype(float))

        # NEST-compatible half-open activity window [origin+start, origin+stop).
        t_start = self.origin + self.start
        if self.stop is not None:
            t_stop = self.origin + self.stop
            active = u.math.logical_and(t >= t_start, t < t_stop)
        else:
            active = t >= t_start

        return u.math.where(active, spike_val * u.math.ones(self.varshape), zeros)
