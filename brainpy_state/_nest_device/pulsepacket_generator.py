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


import math
from collections import deque
from typing import Sequence

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base.base import NESTDevice

__all__ = [
    'pulsepacket_generator',
]

_UNSET = object()


class pulsepacket_generator(NESTDevice):
    r"""Gaussian pulse-packet spike generator compatible with NEST.

    Description
    -----------
    ``pulsepacket_generator`` re-implements NEST's stimulation device with the
    same name and emits integer per-step spike multiplicities.

    **1. Pulse model and grid projection**

    For each configured pulse center :math:`t_c` (ms), this model generates
    exactly ``activity`` sampled spike times per output generator:

    .. math::

       x_{i,j} \sim \mathcal{N}(t_c, \mathrm{sdev}^2),
       \quad i=1,\dots,N,\ j=1,\dots,\mathrm{activity},

    where :math:`N=\prod \mathrm{varshape}` is the number of independent
    generators. For ``sdev == 0``, the Gaussian draw degenerates to the
    deterministic value :math:`x_{i,j}=t_c`.

    Sampled times are converted to NEST-like integer tics and delivery steps:

    .. math::

       \tau = \left\lfloor x \cdot 1000 + 0.5 \right\rfloor,\qquad
       k = \left\lceil \tau / \Delta\tau \right\rceil,

    where :math:`\Delta\tau` is the resolution in tics per simulation step.
    Samples with ``tau < tau_now`` are discarded; the remaining samples are
    queued and emitted as multiplicity counts at their delivery steps.

    **2. NEST update ordering (source-equivalent)**

    This implementation mirrors ``models/pulsepacket_generator.cpp``:

    1. Keep indices ``start_center_idx``/``stop_center_idx`` into sorted
       ``pulse_times`` for a moving window of centers around current time.
    2. At each update step, extend the right edge of that center window while
       ``center_time - t <= tolerance``.
    3. For each newly entered center, sample ``activity`` Gaussian times,
       keep only samples with ``sample_time >= t``, convert them to integer
       steps, and append to a per-generator queue.
    4. Sort each queue.
    5. Emit (pop) all queued spikes whose integer step is in the current
       delivery interval and return per-step multiplicity.

    As in NEST, ``tolerance = sdev * 10`` for ``sdev > 0`` and
    ``tolerance = 1.0 ms`` otherwise.

    **3. Timing semantics (CURRENT_GENERATOR shift)**

    NEST classifies this model as ``CURRENT_GENERATOR`` in
    ``get_type()``. Therefore activity is evaluated with the
    ``StimulationDevice`` current-generator shift:

    .. math::

       t_{\min} < (n + 2) \le t_{\max},

    where ``n`` is the current simulation step and
    ``t_min = origin + start``, ``t_max = origin + stop`` (in steps).

    This differs from regular spike generators and is intentionally preserved
    for behavioral parity.

    **4. Assumptions, constraints, and computational implications**

    Enforced constraints:

    - ``activity`` is an integer scalar with ``activity >= 0``.
    - ``sdev`` is a scalar in ms with ``sdev >= 0``.
    - ``stop >= start`` after scalar conversion.
    - ``sdev_tolerance > 0``.

    Runtime constraints:

    - If ``dt`` is available, finite ``origin``, ``start``, and ``stop`` must
      be exact grid multiples (absolute tolerance ``1e-12`` in ``time / dt``).
    - ``pulse_times`` are flattened to 1-D and sorted ascending before use.

    Computational implications:

    - Let ``C_new`` be newly entered pulse centers in one step. New pulse
      generation costs
      :math:`O(C_{\mathrm{new}} \cdot N \cdot \mathrm{activity})` sampling
      plus per-queue sort when new events are appended.
    - Emission costs :math:`O(N + M_{\mathrm{pop}})` where
      :math:`M_{\mathrm{pop}}` is number of emitted queued spikes in the step.
    - Memory is proportional to the total number of queued future spikes
      across all output generators.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification consumed by
        :class:`brainstate.nn.Dynamics`. ``self.varshape`` derived from this
        value is the exact shape returned by :meth:`update`. Each element is
        one independent output generator. Default is ``1``.
    pulse_times : Sequence[ArrayLike] or ArrayLike or None, optional
        Pulse center times in ms. Accepted inputs are any array-like values
        flattenable to shape ``(K,)`` after conversion, or a
        :class:`brainunit.Quantity` convertible to ``u.ms``.
        ``None`` creates an empty schedule. Values are sorted internally in
        ascending order. Default is ``None``.
    activity : ArrayLike, optional
        Scalar integer count per pulse center, shape ``()`` after conversion.
        Parsed through nearest-integer check with absolute tolerance
        ``1e-12`` and must satisfy ``activity >= 0``. Default is ``0``.
    sdev : ArrayLike, optional
        Scalar standard deviation in ms, shape ``()`` after conversion.
        Accepts unitful time convertible to ``u.ms`` or scalar numeric.
        Must satisfy ``sdev >= 0``. Default is ``0.0 * u.ms``.
    start : ArrayLike, optional
        Scalar relative start time in ms, shape ``()`` after conversion.
        Effective lower bound is ``origin + start`` under current-generator
        semantics and must be grid-representable when ``dt`` is available.
        Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative stop time in ms, shape ``()`` after conversion.
        ``None`` maps to ``+inf``. When finite, must satisfy ``stop >= start``
        and be grid-representable when ``dt`` is available.
        Default is ``None``.
    origin : ArrayLike, optional
        Scalar origin offset in ms, shape ``()`` after conversion.
        Added to ``start`` and ``stop`` to form absolute activity bounds.
        Must be grid-representable when finite and ``dt`` is available.
        Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed used to initialize ``numpy.random.default_rng`` in
        :meth:`init_state`. Default is ``0``.
    sdev_tolerance : float, optional
        Positive multiplicative factor used to compute tolerance window
        ``sdev * sdev_tolerance`` when ``sdev > 0``. NEST default is ``10.0``.
    name : str, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 20 18 24 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``pulse_times``
         - ``None``
         - :math:`t_c`
         - Pulse-center schedule in ms (internally sorted ascending).
       * - ``activity``
         - ``0``
         - :math:`n_{\mathrm{spk}}`
         - Number of sampled spikes generated per center and output train.
       * - ``sdev``
         - ``0.0 * u.ms``
         - :math:`\sigma_t`
         - Temporal jitter standard deviation in ms.
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative lower activity bound (with CURRENT_GENERATOR shift).
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative upper activity bound; ``None`` maps to ``+\infty``.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global offset added to ``start`` and ``stop``.
       * - ``sdev_tolerance``
         - ``10.0``
         - :math:`\kappa`
         - Tolerance factor for center-window inclusion, ``\kappa \sigma_t``.
       * - ``in_size``
         - ``1``
         - -
         - Defines ``self.varshape`` / number of independent generators.
       * - ``rng_seed``
         - ``0``
         - -
         - Seed for NumPy RNG used for Gaussian pulse sampling.

    Raises
    ------
    ValueError
        If ``activity`` is negative or non-integral; if ``sdev`` is negative;
        if ``stop < start``; if ``sdev_tolerance <= 0``; if scalar conversion
        fails due to non-scalar input shape; if backend data has fewer than
        three values; if finite ``origin``/``start``/``stop`` are not grid
        multiples when ``dt`` is available; or if simulation resolution is
        non-positive.
    TypeError
        If time-valued arguments cannot be converted to ``u.ms``-compatible
        values or numeric arrays.
    KeyError
        At runtime, if required simulation context entries (for example
        ``dt`` from ``brainstate.environ.get_dt()``) are unavailable.

    Notes
    -----
    - ``set(activity=...)`` and ``set(sdev=...)`` trigger pulse
      re-generation behavior by clearing queued spikes, matching NEST.
    - Stimulation-backend parameter order in NEST is
      ``[activity, sdev_ms, pulse_time_0_ms, ...]`` and is exposed via
      :meth:`set_data_from_stimulation_backend`.
    - Pulse times that are too far in the past (``sample_time < t``) are
      silently discarded during generation; no error is raised.
    - Outputs are integer multiplicities ``0, 1, 2, ...`` per step,
      matching NEST ``SpikeEvent`` multiplicity semantics rather than
      binary spike flags.

    See Also
    --------
    poisson_generator : Independent Poisson spike trains at fixed rate.
    mip_generator : Correlated spike trains via Multiple Interaction Process.
    inhomogeneous_poisson_generator : Poisson generator with time-varying rate.
    gamma_sup_generator : Superposition of stationary gamma-process trains.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.pulsepacket_generator(
       ...         in_size=(2, 3),
       ...         pulse_times=[10.0 * u.ms, 20.0 * u.ms],
       ...         activity=5,
       ...         sdev=1.5 * u.ms,
       ...         start=0.0 * u.ms,
       ...         stop=40.0 * u.ms,
       ...         rng_seed=7,
       ...     )
       ...     with brainstate.environ.context(t=10.0 * u.ms):
       ...         counts = gen.update()
       ...     _ = counts.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> gen = brainpy.state.pulsepacket_generator(activity=3, sdev=0.5 * u.ms)
       >>> gen.set_data_from_stimulation_backend([4.0, 0.8, 5.0, 15.0, 25.0])
       >>> params = gen.get()
       >>> _ = params['activity'], params['pulse_times']

    References
    ----------
    .. [1] NEST source: ``models/pulsepacket_generator.cpp`` and
           ``models/pulsepacket_generator.h``.
    .. [2] NEST source: ``nestkernel/stimulation_device.cpp``.
    .. [3] NEST model docs:
           https://nest-simulator.readthedocs.io/en/stable/models/pulsepacket_generator.html
    """
    __module__ = 'brainpy.state'

    _TICS_PER_MS = 1000.0

    def __init__(
        self,
        in_size: Size = 1,
        pulse_times: Sequence[ArrayLike] | ArrayLike | None = None,
        activity: ArrayLike = 0,
        sdev: ArrayLike = 0. * u.ms,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        sdev_tolerance: float = 10.0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.activity = self._to_scalar_int(activity, name='activity')
        self.sdev = self._to_scalar_time_ms(sdev)
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self.sdev_tolerance = float(sdev_tolerance)
        if self.sdev_tolerance <= 0.0:
            raise ValueError('sdev_tolerance must be positive.')

        dftype = brainstate.environ.dftype()
        self._pulse_times_ms = np.asarray([], dtype=dftype)
        if pulse_times is not None:
            self._pulse_times_ms = np.sort(self._to_time_array_ms(pulse_times))

        self._validate_parameters(
            activity=self.activity,
            sdev=self.sdev,
            start=self.start,
            stop=self.stop,
        )

        self._num_generators = int(np.prod(self.varshape))
        self._dt_cache_ms = np.nan
        self._dt_tics = 0
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max
        self._tolerance_ms = 1.0

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

    @staticmethod
    def _to_scalar_time_ms(value: ArrayLike) -> float:
        if isinstance(value, u.Quantity):
            dftype = brainstate.environ.dftype()
            arr = np.asarray(value.to_decimal(u.ms), dtype=dftype)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError('Time parameters must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_time_array_ms(values: Sequence[ArrayLike] | ArrayLike) -> np.ndarray:
        dftype = brainstate.environ.dftype()
        if not isinstance(values, u.Quantity):
            arr0 = np.asarray(values)
            if arr0.size == 0:
                return np.asarray([], dtype=dftype)
        if isinstance(values, u.Quantity):
            arr = values.to_decimal(u.ms)
        else:
            arr = u.math.asarray(values, dtype=dftype)
        return np.asarray(arr, dtype=dftype).reshape(-1)

    @staticmethod
    def _to_scalar_int(value: ArrayLike, *, name: str) -> int:
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        scalar = float(arr.reshape(()))
        nearest = np.rint(scalar)
        if not math.isclose(scalar, nearest, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be an integer.')
        return int(nearest)

    @staticmethod
    def _validate_parameters(
        *,
        activity: int,
        sdev: float,
        start: float,
        stop: float,
    ):
        if activity < 0:
            raise ValueError('The activity cannot be negative.')
        if sdev < 0.0:
            raise ValueError('The standard deviation cannot be negative.')
        if stop < start:
            raise ValueError('stop >= start required.')

    @classmethod
    def _ms_to_tics(cls, time_ms: float) -> int:
        # Match NEST Time(ms): static_cast<long>(ms * TICS_PER_MS + 0.5).
        return int(time_ms * cls._TICS_PER_MS + 0.5)

    @staticmethod
    def _assert_grid_time(name: str, time_ms: float, dt_ms: float):
        if not np.isfinite(time_ms):
            return
        ratio = time_ms / dt_ms
        nearest = np.rint(ratio)
        if not math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be a multiple of the simulation resolution.')

    def _dt_ms(self) -> float:
        dt = brainstate.environ.get_dt()
        return self._to_scalar_time_ms(dt)

    def _maybe_dt_ms(self) -> float | None:
        dt = brainstate.environ.get('dt', default=None)
        if dt is None:
            return None
        return self._to_scalar_time_ms(dt)

    def _current_time_ms(self) -> float:
        t = brainstate.environ.get('t', default=0. * u.ms)
        if t is None:
            return 0.0
        return self._to_scalar_time_ms(t)

    def _time_to_step(self, time_ms: float, dt_ms: float) -> int:
        return int(np.rint(time_ms / dt_ms))

    def _time_to_delivery_step(self, time_ms: float) -> int:
        tic = self._ms_to_tics(time_ms)
        if self._dt_tics <= 0:
            return 0
        return int(math.ceil(float(tic) / float(self._dt_tics)))

    def _refresh_runtime_cache(self, dt_ms: float):
        self._assert_grid_time('origin', self.origin, dt_ms)
        self._assert_grid_time('start', self.start, dt_ms)
        self._assert_grid_time('stop', self.stop, dt_ms)

        self._dt_tics = int(np.rint(dt_ms * self._TICS_PER_MS))
        if self._dt_tics <= 0:
            raise ValueError('Simulation resolution must be positive.')

        self._t_min_step = self._time_to_step(self.origin + self.start, dt_ms)
        if np.isfinite(self.stop):
            self._t_max_step = self._time_to_step(self.origin + self.stop, dt_ms)
        else:
            self._t_max_step = np.iinfo(np.int64).max

        if self.sdev > 0.0:
            self._tolerance_ms = self.sdev * self.sdev_tolerance
        else:
            self._tolerance_ms = 1.0

        self._dt_cache_ms = float(dt_ms)

    def _is_active(self, curr_step: int) -> bool:
        shifted_step = curr_step + 2
        return (self._t_min_step < shifted_step) and (shifted_step <= self._t_max_step)

    def _clear_spike_queues(self):
        if hasattr(self, '_spike_queues'):
            for i in range(len(self._spike_queues)):
                self._spike_queues[i].clear()

    def _all_queues_empty(self) -> bool:
        return all(len(q) == 0 for q in self._spike_queues)

    def _pre_run_hook(self, now_ms: float):
        assert self._start_center_idx <= self._stop_center_idx

        self._start_center_idx = 0
        self._stop_center_idx = 0

        now_tic = self._ms_to_tics(now_ms)

        n_centers = self._pulse_times_ms.size
        while self._stop_center_idx < n_centers:
            center_tic = self._ms_to_tics(float(self._pulse_times_ms[self._stop_center_idx]))
            if ((center_tic - now_tic) / self._TICS_PER_MS) > self._tolerance_ms:
                break
            if (abs(center_tic - now_tic) / self._TICS_PER_MS) > self._tolerance_ms:
                self._start_center_idx += 1
            self._stop_center_idx += 1

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize runtime state for stochastic pulse generation.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused by this implementation. Present to match the base-class
            interface. Default is ``None``.
        **kwargs
            Additional unused keyword arguments accepted for interface
            compatibility.


        Raises
        ------
        ValueError
            If ``dt`` is available and finite timing parameters are not grid
            multiples, or if computed simulation resolution is non-positive.
        TypeError
            If environment times cannot be converted to scalar milliseconds.

        Notes
        -----
        Re-initialization resets queues and deterministic random state from
        ``rng_seed``; pending queued spikes are discarded.
        """
        del batch_size, kwargs

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

        self._rng = np.random.default_rng(self.rng_seed)
        self._spike_queues = [deque() for _ in range(self._num_generators)]
        self._start_center_idx = 0
        self._stop_center_idx = 0

        self._pre_run_hook(self._current_time_ms())

    def set(
        self,
        *,
        pulse_times: Sequence[ArrayLike] | ArrayLike | object = _UNSET,
        activity: ArrayLike | object = _UNSET,
        sdev: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        r"""Set public parameters with NEST-compatible update semantics.

        Parameters
        ----------
        pulse_times : Sequence[ArrayLike] or ArrayLike or object, optional
            New pulse-center schedule in ms. Any provided value is converted to
            a flattened ``float64`` array and sorted ascending. Pass ``_UNSET``
            (default) to keep current pulse times.
        activity : ArrayLike or object, optional
            New scalar integer spikes-per-center value, shape ``()`` after
            conversion, with ``activity >= 0``. Pass ``_UNSET`` to keep the
            current value.
        sdev : ArrayLike or object, optional
            New scalar temporal jitter standard deviation in ms, shape ``()``
            after conversion, with ``sdev >= 0``. Pass ``_UNSET`` to keep the
            current value.
        start : ArrayLike or object, optional
            New scalar relative start time in ms, shape ``()`` after
            conversion. Pass ``_UNSET`` to keep the current value.
        stop : ArrayLike or None or object, optional
            New scalar relative stop time in ms, shape ``()`` after conversion.
            ``None`` maps to ``+inf``. Pass ``_UNSET`` to keep the current
            value.
        origin : ArrayLike or object, optional
            New scalar origin offset in ms, shape ``()`` after conversion.
            Pass ``_UNSET`` to keep the current value.


        Raises
        ------
        ValueError
            If integer/scalar validation fails, if ``activity < 0``,
            ``sdev < 0``, ``stop < start``, or if finite time bounds are not
            aligned to the simulation grid when ``dt`` is available.
        TypeError
            If provided values cannot be converted to expected numeric/time
            forms.

        Notes
        -----
        Matching NEST behavior, changing either ``activity`` or ``sdev``
        triggers pulse re-generation state reset by clearing queued spikes.
        """
        new_activity = (
            self.activity
            if activity is _UNSET
            else self._to_scalar_int(activity, name='activity')
        )
        new_sdev = self.sdev if sdev is _UNSET else self._to_scalar_time_ms(sdev)

        new_start = self.start if start is _UNSET else self._to_scalar_time_ms(start)
        if stop is _UNSET:
            new_stop = self.stop
        elif stop is None:
            new_stop = np.inf
        else:
            new_stop = self._to_scalar_time_ms(stop)
        new_origin = self.origin if origin is _UNSET else self._to_scalar_time_ms(origin)

        self._validate_parameters(
            activity=new_activity,
            sdev=new_sdev,
            start=new_start,
            stop=new_stop,
        )

        need_new_pulse = (new_activity != self.activity) or (
            not math.isclose(new_sdev, self.sdev, rel_tol=0.0, abs_tol=0.0))

        if pulse_times is _UNSET:
            new_pulse_times = self._pulse_times_ms.copy()
        else:
            new_pulse_times = self._to_time_array_ms(pulse_times)

        if pulse_times is not _UNSET or need_new_pulse:
            dftype = brainstate.environ.dftype()
            new_pulse_times = np.sort(np.asarray(new_pulse_times, dtype=dftype).reshape(-1))
            self._pulse_times_ms = new_pulse_times
            self._clear_spike_queues()

        self.activity = new_activity
        self.sdev = new_sdev
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

    def get(self) -> dict:
        r"""Return current public parameter values.

        Returns
        -------
        out : dict
            ``dict`` with keys ``pulse_times``, ``activity``, ``sdev``,
            ``start``, ``stop``, and ``origin``. Time values are returned in
            milliseconds as Python ``float`` values, and ``pulse_times`` is a
            Python ``list[float]``.
        """
        return {
            'pulse_times': self._pulse_times_ms.tolist(),
            'activity': int(self.activity),
            'sdev': float(self.sdev),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def set_data_from_stimulation_backend(self, input_param: Sequence[float] | np.ndarray):
        r"""Update parameters from stimulation-backend payload.

        Parameters
        ----------
        input_param : Sequence[float] or numpy.ndarray
            One-dimensional backend payload with shape ``(M,)`` and
            ``M >= 3`` in NEST order:
            ``[activity, sdev_ms, pulse_time_0_ms, ...]``. Entries are parsed
            as ``float64``. ``sdev`` and ``pulse_times`` are interpreted in ms.

        Raises
        ------
        ValueError
            If payload length is between 1 and 2 (inclusive), since at least
            ``activity``, ``sdev``, and one pulse time are required by this
            backend contract.
        TypeError
            If payload cannot be cast to numeric ``float64`` values.
        """
        dftype = brainstate.environ.dftype()
        data = np.asarray(input_param, dtype=dftype).reshape(-1)
        if data.size == 0:
            return
        if data.size < 3:
            raise ValueError(
                'The size of the data for pulsepacket_generator must be at least 3 '
                '[activity, sdev, pulse_times...].'
            )
        self.set(
            activity=data[0],
            sdev=data[1] * u.ms,
            pulse_times=data[2:] * u.ms,
        )

    def _generate_new_pulses(self, curr_tic: int):
        if self._start_center_idx >= self._stop_center_idx or self.activity <= 0:
            return

        need_sort = False

        while self._start_center_idx < self._stop_center_idx:
            center = float(self._pulse_times_ms[self._start_center_idx])

            if self.sdev > 0.0:
                sampled = self._rng.normal(
                    loc=center,
                    scale=self.sdev,
                    size=(self._num_generators, self.activity),
                )
            else:
                dftype = brainstate.environ.dftype()
                sampled = np.full(
                    (self._num_generators, self.activity),
                    center,
                    dtype=dftype,
                )

            for i in range(self._num_generators):
                queue_i = self._spike_queues[i]
                for x in sampled[i]:
                    x_tic = self._ms_to_tics(float(x))
                    if x_tic >= curr_tic:
                        queue_i.append(self._time_to_delivery_step(float(x)))

            need_sort = True
            self._start_center_idx += 1

        if need_sort:
            for i in range(self._num_generators):
                q = self._spike_queues[i]
                if len(q) > 1:
                    self._spike_queues[i] = deque(sorted(q))

    def update(self):
        r"""Advance one simulation step and emit spike multiplicities.

        Returns
        -------
        out : jax.Array
            JAX array of dtype ``int64`` and shape ``self.varshape``.
            Each element is the number of spikes emitted by one output
            generator in the current step. Returns all zeros when inactive or
            when no spikes are due.

        Raises
        ------
        ValueError
            If runtime ``dt`` is non-positive, if finite activity bounds are
            not grid multiples, or if cached time-step conversion becomes
            invalid.
        TypeError
            If runtime time values cannot be converted to scalar milliseconds.
        KeyError
            If required simulation context entries are missing.

        Notes
        -----
        If state has not been initialized explicitly, :meth:`update` performs
        lazy initialization by calling :meth:`init_state`.
        """
        if not hasattr(self, '_rng'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_runtime_cache(dt_ms)

        ditype = brainstate.environ.ditype()
        curr_t_ms = self._current_time_ms()
        curr_step = self._time_to_step(curr_t_ms, dt_ms)

        if (
            (self._start_center_idx == self._pulse_times_ms.size and self._all_queues_empty())
            or (not self._is_active(curr_step))
        ):
            return jnp.zeros(self.varshape, dtype=ditype)

        curr_tic = self._ms_to_tics(curr_t_ms)

        n_centers = self._pulse_times_ms.size
        while self._stop_center_idx < n_centers:
            center_tic = self._ms_to_tics(float(self._pulse_times_ms[self._stop_center_idx]))
            if ((center_tic - curr_tic) / self._TICS_PER_MS) > self._tolerance_ms:
                break
            self._stop_center_idx += 1

        self._generate_new_pulses(curr_tic)

        step_limit = curr_step + 1
        counts = np.zeros(self._num_generators, dtype=ditype)
        for i in range(self._num_generators):
            q = self._spike_queues[i]
            n = 0
            while len(q) > 0 and q[0] < step_limit:
                q.popleft()
                n += 1
            counts[i] = n

        return jnp.asarray(counts.reshape(self.varshape), dtype=ditype)
