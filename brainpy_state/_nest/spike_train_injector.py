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

from ._base import NESTDevice
import braintools
import brainunit as u
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

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

    - ``spike_times`` must be non-descending after conversion to float ms.
    - ``spike_multiplicities`` must be empty or have exactly
      ``len(spike_multiplicities) == len(spike_times)`` elements.
    - ``precise_times=True`` cannot be combined with
      ``allow_offgrid_times=True`` or ``shift_now_spikes=True``.

    Implementation-specific constraints:

    - NEST option flags ``precise_times``, ``allow_offgrid_times``, and
      ``shift_now_spikes`` are accepted for API compatibility but the current
      update rule always uses the fixed tolerance test above regardless of
      their values.
    - ``start``, ``stop``, and ``origin`` are converted to scalar ms via
      ``float(...)`` inside :meth:`update`; non-scalar or traced JAX arrays
      are not supported by this conversion path.
    - NEST documentation states spikes should be strictly in the future. This
      implementation does not perform explicit future-time validation in
      :meth:`__init__` and instead relies on runtime matching combined with
      active-window gating.

    **3. Computational implications**

    Each :meth:`update` call performs one linear scan over all ``K`` spike
    times and one broadcast over ``self.varshape``. Per-step complexity is
    :math:`O(K + \prod \mathrm{varshape})`. Memory cost is :math:`O(K)` for
    the stored times and optional multiplicity list. For large :math:`K`, a
    binary-search variant over the sorted ``spike_times`` list could reduce
    the scan to :math:`O(\log K)` plus the number of matches, but the current
    implementation favours simplicity.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape consumed by :class:`brainstate.nn.Dynamics`. The
        emitted array has shape ``self.varshape`` derived from ``in_size``.
        Default is ``1``.
    spike_times : Sequence, optional
        Sequence of spike times with length ``K``. Entries may be unitful
        times (typically ``brainunit`` ms quantities) or bare numerics
        interpreted as ms. Internally converted to ``float`` milliseconds and
        required to be non-descending. Duplicate times are allowed and their
        multiplicities are accumulated. Default is ``()``.
    spike_multiplicities : Sequence, optional
        Sequence of integer multiplicities with length ``K`` matching
        ``spike_times``, or empty to use implicit unit multiplicities
        (:math:`m_i = 1`). Entries are converted with ``int(m)`` and
        accumulated across all indices matching the same step. Default is
        ``()``.
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
        Must be scalar-convertible inside :meth:`update`.
        Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative deactivation time :math:`t_{\mathrm{stop,rel}}` (typically
        ms), initialized via :func:`braintools.init.param` when not ``None``.
        The effective exclusive upper bound is ``origin + stop``. ``None``
        disables the upper bound. Must be scalar-convertible when not ``None``.
        Default is ``None``.
    origin : ArrayLike, optional
        Global time origin :math:`t_0` (typically ms) added to both ``start``
        and ``stop`` to obtain absolute window bounds. Must be
        scalar-convertible inside :meth:`update`. Default is ``0. * u.ms``.
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
         - Spike schedule in ms; matched by ``|t - t_i| < dt/2``.
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
        after ms conversion, or if ``spike_multiplicities`` is non-empty and
        has a different length than ``spike_times``.
    TypeError
        If any spike-time or multiplicity entry cannot be converted to
        ``float`` / ``int``, or if ``start``, ``stop``, or ``origin`` are not
        scalar-convertible at update time.
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
        r"""Compute the accumulated spike output for the current simulation step.

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
        TypeError
            If ``t``, ``dt``, ``start``, ``stop``, or ``origin`` cannot be
            converted to scalar ms values due to incompatible shapes or units.
        ValueError
            If downstream unit conversion raises an invalid-value error.

        Notes
        -----
        Matching uses strict inequality ``abs(t_ms - spike_t) < dt_ms / 2``,
        so a spike exactly at half-step distance from ``t`` is *not* emitted
        at that step.

        The activity-window check short-circuits to a zero array before the
        spike scan, so steps outside ``[origin + start, origin + stop)``
        incur only the scalar comparisons, not the :math:`O(K)` scan.

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
