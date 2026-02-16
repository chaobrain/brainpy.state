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
    matching.

    Enforced constraints:

    - ``spike_times`` must be sorted in non-descending order after conversion
      to float ms.
    - ``spike_weights`` must be empty or have exactly
      ``len(spike_times)`` elements.

    Practical constraints from the current implementation:

    - ``start``, ``stop`` (if provided), and ``origin`` are converted to scalar
      ms inside :meth:`update`; non-scalar/broadcasted arrays are not supported
      by this conversion path.
    - Duplicate spike times are allowed. Without weights, duplicates remain
      binary output. With weights, the last duplicate's weight is used.

    **3. Computational implications**

    Per :meth:`update` call, complexity is
    :math:`O(K + \prod\mathrm{varshape})`, where :math:`K` is the number of
    configured spike times. The linear scan preserves deterministic ordering
    semantics for duplicate times and weight override behavior.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification consumed by
        :class:`brainstate.nn.Dynamics`. The emitted array has shape
        ``self.varshape`` derived from ``in_size``. Default is ``1``.
    spike_times : Sequence, optional
        Sequence of spike times with length ``K``. Entries may be unitful
        times (typically ms) or unitless numerics interpreted as ms. Internally
        converted to ``float`` milliseconds and required to be non-descending.
        Default is ``()``.
    spike_weights : Sequence, optional
        Optional sequence of per-spike amplitudes with length ``K`` matching
        ``spike_times`` exactly, or empty to use binary spikes. Entries are
        converted to ``float`` without unit conversion. Default is ``()``.
    start : ArrayLike, optional
        Relative activation time :math:`t_{\mathrm{start,rel}}` (typically ms),
        initialized through :func:`braintools.init.param`. Effective lower
        bound is ``origin + start`` (inclusive). Must be scalar-convertible at
        update time. Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative deactivation time :math:`t_{\mathrm{stop,rel}}` (typically
        ms), initialized through :func:`braintools.init.param` when provided.
        Effective upper bound is ``origin + stop`` (exclusive). ``None`` means
        no upper bound. Must be scalar-convertible when not ``None``. Default
        is ``None``.
    origin : ArrayLike, optional
        Global time origin :math:`t_0` (typically ms) added to ``start`` and
        ``stop``. Must be scalar-convertible at update time. Default is
        ``0. * u.ms``.
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

    Returns
    -------
    out : Any
        Dynamics node. Calling :meth:`update` returns a float-valued JAX array
        with shape ``self.varshape``, equal to ``0`` when inactive or when no
        spike matches the current step.

    Raises
    ------
    ValueError
        If ``spike_times`` is not non-descending, or if
        ``len(spike_weights)`` is non-zero and differs from
        ``len(spike_times)``.
    TypeError
        If a time/weight entry cannot be converted to float, or if
        ``start``/``stop``/``origin`` are not scalar-convertible during update.
    KeyError
        At update time, if simulation context lacks required time information
        (for example ``'t'`` or ``dt``), depending on environment behavior.

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
        r"""Compute spike output for the current simulation step.

        Parameters
        ----------
        None
            Uses ``brainstate.environ['t']`` and ``brainstate.environ.get_dt()``
            together with instance parameters initialized in :meth:`__init__`.

        Returns
        -------
        out : Any
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
            ``brainstate.environ``.
        TypeError
            If scalar conversion of time parameters fails due to incompatible
            shapes/dtypes/units.
        ValueError
            If downstream unit conversion raises an invalid-value error.

        Notes
        -----
        The matching tolerance is ``dt/2`` in ms. When multiple entries in
        ``spike_times`` match the same step, this implementation intentionally
        keeps only the last matching weight/value.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate
           >>> import brainunit as u
           >>> from brainpy.state import spike_generator
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     gen = spike_generator(spike_times=[1.0 * u.ms, 2.0 * u.ms])
           ...     with brainstate.environ.context(t=2.0 * u.ms):
           ...         out = gen.update()
           ...     _ = out.shape
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
