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
    r"""Spike train injector -- NEST-compatible event source device.

    Emit deterministic spike events at configured times with optional
    per-time multiplicity, then gate output by an activity window.

    **1. Model equations**

    Let :math:`\{t_i\}_{i=1}^{K}` be configured spike times in ms after
    conversion from unitful or unitless inputs. Let :math:`m_i` denote
    multiplicity (``spike_multiplicities``) when provided, otherwise
    :math:`m_i = 1`. At simulation time :math:`t` with step :math:`\Delta t`
    (both in ms), this implementation uses

    .. math::

        q_i(t) = \mathbf{1}\!\left[|t - t_i| < \frac{\Delta t}{2}\right].

    The scalar emitted spike count before window gating is

    .. math::

        a(t) = \sum_{i=1}^{K} m_i q_i(t).

    The activity gate is

    .. math::

        g(t) =
        \mathbf{1}\!\left[t \ge t_0 + t_{\mathrm{start,rel}}\right]
        \cdot
        \mathbf{1}\!\left[t < t_0 + t_{\mathrm{stop,rel}}\right],

    where the second indicator is omitted when ``stop is None``.
    Returned output is broadcast to node shape ``self.varshape``:

    .. math::

        y(t) = g(t)\,a(t)\,\mathbf{1}_{\mathrm{varshape}}.

    **2. Timing derivation, assumptions, and constraints**

    The :math:`|t - t_i| < \Delta t / 2` rule corresponds to nearest-grid
    assignment under uniform-step simulation. For exact half-step offsets,
    strict inequality means no match at that step. If multiple ``spike_times``
    entries map to the same step, emitted values are summed.

    Enforced constraints:

    - ``spike_times`` must be non-descending after conversion to float ms.
    - ``spike_multiplicities`` must be empty or have
      ``len(spike_multiplicities) == len(spike_times)``.
    - ``precise_times=True`` cannot be combined with
      ``allow_offgrid_times=True`` or ``shift_now_spikes=True``.

    Implementation-specific constraints:

    - NEST option flags ``precise_times``, ``allow_offgrid_times``, and
      ``shift_now_spikes`` are accepted for API compatibility, but the current
      update rule always uses the fixed tolerance test above.
    - ``start``/``stop``/``origin`` are converted through ``float(...)``
      inside :meth:`update`; values must be scalar-convertible in runtime
      context.
    - NEST documentation states spikes should be strictly in the future. This
      implementation does not perform explicit future-time validation in
      ``__init__`` and instead relies on runtime matching and active-window
      gating.

    **3. Computational implications**

    Each :meth:`update` call performs one linear scan over ``spike_times`` and
    one broadcast over ``self.varshape``. Per-step complexity is
    :math:`O(K + \prod \mathrm{varshape})`, where :math:`K` is the number of
    configured spikes. Memory cost is :math:`O(K)` for stored times and
    optional multiplicities.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape consumed by :class:`brainstate.nn.Dynamics`. The
        emitted array has shape ``self.varshape`` derived from ``in_size``.
        Default is ``1``.
    spike_times : Sequence, optional
        Sequence of spike times with length ``K``. Entries can be unitful
        times (typically ms) or unitless numerics interpreted as ms.
        Internally converted to ``float`` milliseconds and required to be
        non-descending. Duplicate times are allowed. Default is ``()``.
    spike_multiplicities : Sequence, optional
        Sequence of multiplicities with length ``K`` matching ``spike_times``,
        or empty to use unit spikes. Entries are converted with ``int(m)`` and
        accumulated when multiple spikes map to the same step. Default is
        ``()``.
    precise_times : bool, optional
        NEST compatibility flag for precise timing representation. Stored and
        validated against other flags but not used to alter runtime matching in
        this implementation. Default is ``False``.
    allow_offgrid_times : bool, optional
        NEST compatibility flag for off-grid handling. Stored and validated but
        not used to alter runtime matching in this implementation. Default is
        ``False``.
    shift_now_spikes : bool, optional
        NEST compatibility flag for shifting spikes that round to current step.
        Stored and validated but not used to alter runtime matching in this
        implementation. Default is ``False``.
    start : ArrayLike, optional
        Relative activation time :math:`t_{\mathrm{start,rel}}` (typically ms),
        initialized via :func:`braintools.init.param`. Effective lower bound is
        ``origin + start`` (inclusive). Must be scalar-convertible inside
        :meth:`update`. Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative deactivation time :math:`t_{\mathrm{stop,rel}}` (typically
        ms), initialized via :func:`braintools.init.param` when provided.
        Effective upper bound is ``origin + stop`` (exclusive). ``None`` means
        no upper bound. Must be scalar-convertible when provided.
        Default is ``None``.
    origin : ArrayLike, optional
        Global origin :math:`t_0` (typically ms) added to ``start`` and
        ``stop`` at runtime. Must be scalar-convertible in :meth:`update`.
        Default is ``0. * u.ms``.
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
         - Spike schedule in ms used by ``|t - t_i| < dt/2`` matching.
       * - ``spike_multiplicities``
         - ``()``
         - :math:`m_i`
         - Per-time spike count; empty means implicit ``m_i = 1``.
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
         - Global offset added to ``start`` and ``stop``.

    Returns
    -------
    out : Any
        Dynamics node. Calling :meth:`update` returns a float-valued JAX array
        with shape ``self.varshape`` equal to accumulated spike multiplicity at
        matching times and ``0`` otherwise.

    Raises
    ------
    ValueError
        If incompatible timing flags are combined, if ``spike_times`` is not
        non-descending, or if ``spike_multiplicities`` has invalid length.
    TypeError
        If spike-time or multiplicity conversion to ``float``/``int`` fails,
        or if time-window values are not scalar-convertible at update time.
    KeyError
        At update time, if required simulation context entries are unavailable,
        depending on ``brainstate.environ`` behavior.

    Notes
    -----
    This device does not accept incoming synaptic/current connections. It only
    emits scheduled spike events (nonzero values in the returned array).

    Examples
    --------
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

    References
    ----------
    .. [1] NEST Simulator, ``spike_train_injector`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/spike_train_injector.html

    See Also
    --------
    spike_generator : General-purpose spike generator device.
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
        r"""Compute spike output for the current simulation step.

        Parameters
        ----------
        None
            Uses ``brainstate.environ['t']`` and
            ``brainstate.environ.get_dt()`` together with the instance state
            initialized in :meth:`__init__`.

        Returns
        -------
        out : Any
            Float-valued JAX array with shape ``self.varshape``. Output is
            ``0`` outside the active window and equals the accumulated
            multiplicity of all indices satisfying ``|t - t_i| < dt/2`` when
            the device is active.

        Raises
        ------
        KeyError
            If required simulation context entries are missing from
            ``brainstate.environ``.
        TypeError
            If ``t``, ``dt``, ``start``, ``stop``, or ``origin`` cannot be
            converted to scalar ms values.
        ValueError
            If downstream unit conversion/comparison raises invalid-value
            errors.

        Notes
        -----
        Matching uses strict inequality ``abs(t_ms - spike_t) < dt_ms / 2``.
        Therefore, a spike exactly at half-step distance from ``t`` is not
        emitted at that step.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate
           >>> import brainunit as u
           >>> from brainpy.state import spike_train_injector
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     gen = spike_train_injector(
           ...         spike_times=[1.0 * u.ms, 1.0 * u.ms],
           ...         spike_multiplicities=[2, 3],
           ...     )
           ...     with brainstate.environ.context(t=1.0 * u.ms):
           ...         out = gen.update()
           ...     _ = out.shape
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
