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

from brainpy_state._nest._base import NESTDevice

__all__ = [
    'spike_generator',
]


class spike_generator(NESTDevice):
    r"""Spike generator -- NEST-compatible stimulation device.

    Emit deterministic spike-like outputs at prescribed times with optional
    per-event amplitudes, while respecting a half-open activity window.

    **1. Model equations**

    Let :math:`\{t_i\}_{i=1}^{K}` be configured spike times in ms
    (non-descending after conversion), and :math:`\{w_i\}_{i=1}^{K}` optional
    spike weights. At simulation time :math:`t` with step :math:`\Delta t`
    (both in ms), define the matching indicator

    .. math::

        m_i(t) = \mathbf{1}\!\left[|t - t_i| < \frac{\Delta t}{2}\right].

    The active-window gate is

    .. math::

        g(t) = \mathbf{1}\!\left[t \ge t_0 + t_{\mathrm{start,rel}}\right]
        \cdot
        \mathbf{1}\!\left[t < t_0 + t_{\mathrm{stop,rel}}\right],

    where the second indicator is omitted when ``stop is None``.

    This implementation computes a scalar amplitude :math:`a(t)` as follows:

    - no ``spike_weights``: :math:`a(t)=1` if any :math:`m_i(t)=1`, else
      :math:`a(t)=0`;
    - with ``spike_weights``: :math:`a(t)` equals the weight associated with
      the *last* matching index (iteration order through ``spike_times``).

    The returned output is broadcast to ``self.varshape``:

    .. math::

        y(t) = g(t)\,a(t)\,\mathbf{1}_{\mathrm{varshape}}.

    **2. Timing semantics, assumptions, and constraints**

    A configured spike at :math:`t_s` is intended for the step satisfying
    :math:`t_s-\Delta t < t \le t_s` under grid-aligned simulation. The
    implementation uses :math:`|t-t_s| < \Delta t/2` for robust floating-point
    matching, which is equivalent to :math:`t - \Delta t/2 < t_s < t + \Delta
    t/2`.

    Enforced constraints:

    - ``spike_times`` must be sorted in non-descending order after conversion.
    - ``spike_weights`` must be empty or have exactly
      ``len(spike_times)`` elements.

    Accepted but not additionally constrained:

    - Unitless ``spike_times`` are interpreted as ms.
    - Duplicate spike times are allowed. Without weights, duplicates remain
      binary output. With weights, the last duplicate's weight is used.

    **3. Computational implications**

    Each :meth:`update` call uses :func:`u.math.searchsorted` to locate the
    spike-time range matching the current step, then selects the last matching
    weight with :func:`u.math.clip` and :func:`u.math.where`. Per-call
    complexity is :math:`O(\log K + \prod\mathrm{varshape})`, where :math:`K`
    is the number of configured spike times.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification consumed by
        :class:`brainstate.nn.Dynamics`. The emitted array has shape
        ``self.varshape`` derived from ``in_size``. Default is ``1``.
    spike_times : Sequence, optional
        Sequence of spike times with length ``K``. Entries may be unitful
        times (typically ms) or unitless numerics interpreted as ms. Passed
        directly to :func:`u.math.asarray`, which validates unit consistency
        across all entries. Must be non-descending. Default is ``()``.
    spike_weights : Sequence, optional
        Optional sequence of per-spike amplitudes with length ``K`` matching
        ``spike_times`` exactly, or empty to use binary spikes. Entries are
        passed to :func:`u.math.asarray` (dimensionless). Default is ``()``.
    start : ArrayLike, optional
        Relative activation time :math:`t_{\mathrm{start,rel}}` (typically ms),
        initialized through :func:`braintools.init.param`. Effective lower
        bound is ``origin + start`` (inclusive). Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative deactivation time :math:`t_{\mathrm{stop,rel}}` (typically
        ms), initialized through :func:`braintools.init.param` when provided.
        Effective upper bound is ``origin + stop`` (exclusive). ``None`` means
        no upper bound. Default is ``None``.
    origin : ArrayLike, optional
        Global time origin :math:`t_0` (typically ms) added to ``start`` and
        ``stop``, broadcast to ``self.varshape``. Default is ``0. * u.ms``.
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
       * - ``spike_times``
         - ``()``
         - :math:`t_i`
         - Scheduled spike times in ms, checked by ``|t - t_i| < dt/2``.
       * - ``spike_weights``
         - ``()``
         - :math:`w_i`
         - Per-spike amplitude; when multiple indices match, the last wins.
       * - ``start``
         - ``0. * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative inclusive lower bound of active window.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative exclusive upper bound of active window.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global offset applied to ``start`` and ``stop``.

    Raises
    ------
    ValueError
        If ``spike_times`` is not non-descending, or if
        ``len(spike_weights)`` is non-zero and differs from
        ``len(spike_times)``.
    TypeError
        If :func:`u.math.asarray` detects unit inconsistency across entries,
        or if unitful/unitless arithmetic is invalid during time-window
        comparisons.
    KeyError
        At update time, if simulation context lacks ``'t'`` or ``dt`` in
        ``brainstate.environ``.

    Notes
    -----
    Unlike current generators (``dc_generator``, ``step_current_generator``),
    ``spike_generator`` emits dimensionless impulses (or weighted real values)
    rather than physical current quantities. The output is intended to be
    consumed directly as pre-synaptic spike events or injected into a synapse
    model that scales by connection weight.

    NEST's ``spike_generator`` uses multiplicity to allow multiple spikes per
    time step; this implementation preserves that semantics — the last matching
    weight wins when duplicates exist. The :meth:`update` method is fully
    compatible with ``jax.jit``: both the spike-time lookup and the
    activity-window check use purely functional operations with no Python
    control flow over traced values.

    Spike times should ideally be aligned to the simulation grid (multiples of
    ``dt``) to avoid off-by-one steps due to floating-point comparison. The
    half-open tolerance ``dt/2`` generally covers one-ULP rounding errors for
    grid-aligned times.

    See Also
    --------
    dc_generator : Constant-current stimulation device.
    ac_generator : Sinusoidal current stimulation device.
    step_current_generator : Piecewise-constant current stimulation device.
    spike_train_injector : Inject pre-recorded spike trains into the network.

    References
    ----------
    .. [1] NEST Simulator, ``spike_generator`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/spike_generator.html

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     sg = brainpy.state.spike_generator(
       ...         spike_times=[5.0 * u.ms, 10.0 * u.ms, 15.0 * u.ms],
       ...     )
       ...     with brainstate.environ.context(t=10.0 * u.ms):
       ...         spk = sg.update()
       ...     _ = spk.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     sg = brainpy.state.spike_generator(
       ...         spike_times=[5.0 * u.ms, 5.0 * u.ms, 10.0 * u.ms],
       ...         spike_weights=[0.25, 0.5, 2.0],
       ...     )
       ...     with brainstate.environ.context(t=5.0 * u.ms):
       ...         spk = sg.update()
       ...     _ = spk.shape
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

        if len(spike_weights) > 0 and len(spike_weights) != len(spike_times):
            raise ValueError(
                "spike_weights must have the same length as spike_times "
                f"or be empty. Got {len(spike_weights)} and {len(spike_times)}."
            )

        # Store spike_times as a Quantity array; u.math.asarray validates
        # that all entries share a consistent unit.
        # Shape: (K,)
        if len(spike_times) > 0:
            self.spike_times = u.math.asarray(spike_times)

            # Validate non-descending order.
            for i in range(1, len(self.spike_times)):
                if self.spike_times[i] < self.spike_times[i - 1]:
                    raise ValueError(
                        "spike_times must be sorted in non-descending order. "
                        f"Got {self.spike_times[i - 1]} > {self.spike_times[i]} at index {i}."
                    )
        else:
            self.spike_times = None

        # Store spike weights as a dimensionless array, or None for binary mode.
        self.spike_weights = u.math.asarray(spike_weights) if len(spike_weights) > 0 else None

        self.start = braintools.init.param(start, self.varshape)
        self.stop = None if stop is None else braintools.init.param(stop, self.varshape)
        self.origin = braintools.init.param(origin, self.varshape)

    def update(self):
        r"""Compute spike output for the current simulation step.

        The implementation is fully compatible with ``jax.jit``: spike-time
        matching uses :func:`u.math.searchsorted` on the static
        ``spike_times`` array while ``t`` and ``dt`` remain traced values
        throughout. The activity-window check uses :func:`u.math.logical_and`
        and :func:`u.math.where` with no Python branching over traced values.

        Returns
        -------
        out : jax.Array
            Float-valued JAX array with shape ``self.varshape``.
            Output semantics:

            - ``0`` when outside ``[origin + start, origin + stop)`` (or
              ``[origin + start, +inf)`` if ``stop is None``),
            - ``0`` when active but no configured spike matches
              ``|t - t_i| < dt/2``,
            - ``1`` at a matching spike time without weights,
            - last matching weight when ``spike_weights`` is configured.

        Raises
        ------
        KeyError
            If required simulation context values are missing from
            ``brainstate.environ`` (e.g. ``'t'`` or ``dt``).

        Notes
        -----
        Both ``spike_times`` and ``t`` are divided by ``u.ms`` to obtain
        dimensionless arrays before calling :func:`u.math.searchsorted`.
        The matching condition ``|t - t_s| < dt/2`` is rewritten as the open
        interval ``(t - dt/2, t + dt/2)`` and located with two
        ``searchsorted`` calls:

        - ``idx_lo = searchsorted(times, t - dt/2, side='right')`` — first
          index strictly greater than the lower bound.
        - ``idx_hi = searchsorted(times, t + dt/2, side='left')`` — first
          index at or above the upper bound.

        Any spike exists when ``idx_hi > idx_lo``; the last matching spike
        index is ``idx_hi - 1``, clamped to a valid range for the gather.
        Start is inclusive and stop is exclusive, matching NEST semantics.

        See Also
        --------
        spike_generator : Class-level parameter definitions and model equations.
        dc_generator.update : Windowed constant-current update rule.
        step_current_generator.update : Windowed piecewise-constant update rule.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()

        zeros = u.math.zeros(self.varshape)

        if self.spike_times is None:
            # No spike times configured: output is always zero.
            return zeros

        # Locate the open interval (t - dt/2, t + dt/2) via two searchsorted calls.
        # idx_lo: first index where spike_times > t - dt/2  (side='right')
        # idx_hi: first index where spike_times >= t + dt/2 (side='left')
        # Matching range is [idx_lo, idx_hi).
        idx_lo = u.math.searchsorted(self.spike_times, t - dt / 2, side='right')
        idx_hi = u.math.searchsorted(self.spike_times, t + dt / 2, side='left')

        any_match = idx_hi > idx_lo

        # Last matching spike index; clamped to [0, K-1] for safe gather.
        last_idx = u.math.clip(idx_hi - 1, 0, self.spike_times.shape[0] - 1)

        if self.spike_weights is not None:
            spike_val = u.math.where(any_match, self.spike_weights[last_idx], 0.0)
        else:
            spike_val = u.math.where(any_match, 1.0, 0.0)

        # NEST-compatible half-open activity window [origin+start, origin+stop).
        t_start = self.origin + self.start
        if self.stop is not None:
            t_stop = self.origin + self.stop
            active = u.math.logical_and(t >= t_start, t < t_stop)
        else:
            active = t >= t_start

        return u.math.where(active, spike_val * u.math.ones(self.varshape), zeros)
