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

import brainstate
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base._base import NESTDevice

__all__ = [
    'ppd_sup_generator',
]

_UNSET = object()


class ppd_sup_generator(NESTDevice):
    r"""Superposition of Poisson processes with dead time (NEST-compatible).

    Description
    -----------
    ``ppd_sup_generator`` re-implements NEST's stimulation device with the
    same name. For each output train, it emits the per-step multiplicity of a
    superposition of ``n_proc`` independent Poisson-like component processes
    with absolute dead time.

    **1. State model, derivation, and update equations**

    Let :math:`r=\mathrm{rate}` (Hz), :math:`\tau_d=\mathrm{dead\_time}` (ms),
    and :math:`\Delta t` be the simulation resolution in ms. For each output
    train, the internal state is an age-discretized occupancy model:

    - ``occ_active``: number of currently active component processes.
    - ``occ_refractory[a]`` for
      :math:`a=0,\dots,\lfloor\tau_d/\Delta t\rfloor-1`: number of processes
      in refractory age bin ``a``.
    - ``activate``: rotating pointer indicating the bin whose occupants become
      active at the current step.

    NEST's per-step hazard for one active process is

    .. math::

       h_{\mathrm{step}} =
       \frac{\Delta t}{1000/r-\tau_d}.

    This discretization is valid under NEST's model constraint
    :math:`1000/r>\tau_d` (or ``rate == 0``). If sinusoidal modulation is
    enabled, the instantaneous hazard becomes

    .. math::

       h_t = h_{\mathrm{step}}
       \left(1 + A \sin\left(2\pi f t / 1000\right)\right),

    where :math:`A=\mathrm{relative\_amplitude}\in[0,1]` and
    :math:`f=\mathrm{frequency}` in Hz.

    For each train and step, emitted multiplicity ``n_spikes`` is sampled from
    the active pool using NEST's branch logic:

    - Binomial branch: ``Binomial(occ_active, h_t)``.
    - Poisson approximation branch when
      ``(occ_active >= 100 and h_t <= 0.01)`` or
      ``(occ_active >= 500 and h_t * occ_active <= 0.1)``:
      sample ``Poisson(h_t * occ_active)`` and clip to ``occ_active``.

    State transition for nonzero refractory bins is

    .. math::

       occ\_active' = occ\_active + occ\_refractory[p] - n\_spikes,
       \quad
       occ\_refractory[p]' = n\_spikes,

    with pointer update :math:`p'=(p+1)\bmod B`,
    :math:`B=\lfloor\tau_d/\Delta t\rfloor`.
    If ``B == 0`` (zero dead time), the active pool is not decremented by
    refractory bookkeeping and each component can contribute at most one spike
    per step through the binomial/Poisson draw.

    **2. Timing semantics and activity window**

    Activity follows NEST ``StimulationDevice`` semantics for generators:

    .. math::

       t_{\min} < t \le t_{\max},
       \qquad
       t_{\min} = origin + start,\quad t_{\max} = origin + stop.

    Therefore ``start`` is exclusive and ``stop`` is inclusive. Internally,
    finite times are projected to steps with ``round(time_ms / dt_ms)`` and
    checked as ``t_min_step < curr_step <= t_max_step``.

    **3. Assumptions, constraints, and computational implications**

    All physical parameters are scalarized to host-side ``float64`` or
    ``int`` before simulation. Enforced constraints are:

    - ``dead_time >= 0``.
    - ``n_proc >= 1``.
    - ``relative_amplitude in [0, 1]``.
    - ``stop >= start``.
    - ``1000 / rate > dead_time`` (or ``rate == 0``).

    If ``dt`` is available, finite ``origin``, ``start``, and ``stop`` must be
    exact grid multiples (absolute tolerance ``1e-12`` in ``time/dt`` ratio).
    Runtime of :meth:`update` is
    :math:`O(\prod \mathrm{varshape})` per step; memory is
    :math:`O(\prod \mathrm{varshape} \cdot \lfloor\tau_d/\Delta t\rfloor)`.
    Random draws are produced by ``numpy.random.Generator`` (seeded by
    ``rng_seed``), so stochasticity is NumPy-host based rather than JAX-key
    based.

    Parameters
    ----------
    in_size : Size, optional
        Output size specification consumed by
        :class:`brainstate.nn.Dynamics`. ``self.varshape`` derived from this
        value is the exact shape returned by :meth:`update`, and each element
        corresponds to one independent output train. Default is ``1``.
    rate : ArrayLike, optional
        Scalar component-process rate in spikes/s (Hz), shape ``()`` after
        conversion. Accepts a single-element numeric ``ArrayLike`` or a
        :class:`brainunit.Quantity` convertible to ``u.Hz``.
        Must satisfy ``1000 / rate > dead_time`` when ``rate > 0``.
        Default is ``0.0 * u.Hz``.
    dead_time : ArrayLike, optional
        Scalar absolute refractory time in ms, shape ``()`` after conversion.
        Accepts a single-element numeric ``ArrayLike`` or a
        :class:`brainunit.Quantity` convertible to ``u.ms``.
        Must satisfy ``dead_time >= 0``. Default is ``0.0 * u.ms``.
    n_proc : ArrayLike, optional
        Scalar integer number of independent component processes per output
        train, shape ``()`` after conversion. Parsed by nearest-integer check
        with absolute tolerance ``1e-12``. Must satisfy ``n_proc >= 1``.
        Default is ``1``.
    frequency : ArrayLike, optional
        Scalar sinusoidal modulation frequency in Hz, shape ``()`` after
        conversion. ``frequency == 0`` disables sinusoidal variation even when
        ``relative_amplitude > 0``. Default is ``0.0 * u.Hz``.
    relative_amplitude : ArrayLike, optional
        Scalar dimensionless modulation amplitude :math:`A`, shape ``()``
        after conversion. Must satisfy ``0 <= relative_amplitude <= 1``.
        Default is ``0.0``.
    start : ArrayLike, optional
        Scalar relative activation time in ms, shape ``()`` after conversion.
        Effective lower activity bound is ``origin + start`` and is exclusive.
        Must be grid-representable when ``dt`` is available.
        Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Scalar relative deactivation time in ms, shape ``()`` after
        conversion. Effective upper activity bound is ``origin + stop`` and is
        inclusive. ``None`` maps to ``+inf``. Must satisfy ``stop >= start``
        and be grid-representable when finite and ``dt`` is available.
        Default is ``None``.
    origin : ArrayLike, optional
        Scalar time-origin offset in ms, shape ``()`` after conversion.
        Added to ``start`` and ``stop`` to compute absolute active bounds.
        Must be grid-representable when finite and ``dt`` is available.
        Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Seed used to initialize ``numpy.random.default_rng`` in
        :meth:`init_state`. Default is ``0``.
    name : str or None, optional
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
       * - ``rate``
         - ``0.0 * u.Hz``
         - :math:`r`
         - Component-process rate in spikes/s.
       * - ``dead_time``
         - ``0.0 * u.ms``
         - :math:`\tau_d`
         - Absolute refractory duration in ms.
       * - ``n_proc``
         - ``1``
         - :math:`n_{\mathrm{proc}}`
         - Number of component processes per output train.
       * - ``frequency``
         - ``0.0 * u.Hz``
         - :math:`f`
         - Modulation frequency in Hz.
       * - ``relative_amplitude``
         - ``0.0``
         - :math:`A`
         - Relative sinusoidal modulation amplitude.
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative exclusive lower activity bound.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative inclusive upper bound; ``None`` maps to ``+\infty``.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global offset added to ``start`` and ``stop``.
       * - ``in_size``
         - ``1``
         - -
         - Defines ``self.varshape`` for independent output trains.
       * - ``rng_seed``
         - ``0``
         - -
         - Seed for NumPy RNG used by stochastic transition draws.

    Raises
    ------
    ValueError
        If scalar conversion fails due to non-scalar inputs; if ``dead_time``
        is negative; if ``n_proc < 1``; if ``relative_amplitude`` is outside
        ``[0, 1]``; if ``stop < start``; if ``1000 / rate <= dead_time`` for
        nonzero ``rate``; if integer-valued inputs are non-integral beyond
        tolerance; or if finite ``origin``/``start``/``stop`` are not
        multiples of simulation resolution when ``dt`` is available.
    TypeError
        If conversion to ``u.Hz``/``u.ms`` or numeric casting fails for
        provided parameter types.
    KeyError
        At runtime, if required simulation-context fields (for example ``dt``
        used by ``brainstate.environ.get_dt()``) are unavailable.

    Notes
    -----
    - Initial occupancy matches NEST ``pre_run_hook()``:
      ``floor(rate / 1000 * n_proc * dt)`` in each refractory bin and the
      remainder in ``occ_active``.
    - NEST does not initialize to sinusoidal equilibrium, so modulation can
      show startup transients.
    - Stimulation-backend parameter order in NEST is
      ``[dead_time, rate, n_proc, frequency, relative_amplitude]``.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.ppd_sup_generator(
       ...         in_size=(2, 2),
       ...         rate=20.0 * u.Hz,
       ...         dead_time=2.0 * u.ms,
       ...         n_proc=80,
       ...         frequency=8.0 * u.Hz,
       ...         relative_amplitude=0.25,
       ...         start=5.0 * u.ms,
       ...         stop=50.0 * u.ms,
       ...         rng_seed=3,
       ...     )
       ...     with brainstate.environ.context(t=12.0 * u.ms):
       ...         counts = gen.update()
       ...     _ = counts.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> gen = brainpy.state.ppd_sup_generator(rate=15.0 * u.Hz, n_proc=30)
       >>> gen.set(dead_time=1.5 * u.ms, stop=None, origin=2.0 * u.ms)
       >>> params = gen.get()
       >>> _ = params['dead_time'], params['stop']

    See Also
    --------
    gamma_sup_generator : Superposition of gamma-process component trains.
    sinusoidal_gamma_generator : Inhomogeneous gamma generator with sinusoidal rate modulation.
    poisson_generator : Independent Poisson trains without dead time.

    References
    ----------
    .. [1] NEST source: ``models/ppd_sup_generator.h`` and
           ``models/ppd_sup_generator.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/ppd_sup_generator.html
    .. [3] Deger M, Helias M, Boucsein C, Rotter S (2011).
           Statistical properties of superimposed stationary spike trains.
           Journal of Computational Neuroscience.
           https://doi.org/10.1007/s10827-011-0362-8
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        rate: ArrayLike = 0. * u.Hz,
        dead_time: ArrayLike = 0. * u.ms,
        n_proc: ArrayLike = 1,
        frequency: ArrayLike = 0. * u.Hz,
        relative_amplitude: ArrayLike = 0.0,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.rate = self._to_scalar_rate_hz(rate)
        self.dead_time = self._to_scalar_time_ms(dead_time)
        self.n_proc = self._to_scalar_int(n_proc, name='n_proc')
        self.frequency = self._to_scalar_rate_hz(frequency)
        self.relative_amplitude = self._to_scalar_float(
            relative_amplitude,
            name='relative_amplitude',
        )
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            rate=self.rate,
            dead_time=self.dead_time,
            n_proc=self.n_proc,
            relative_amplitude=self.relative_amplitude,
            start=self.start,
            stop=self.stop,
        )

        self._num_targets = int(np.prod(self.varshape))
        self._hazard_step = 0.0
        self._omega_rad_per_ms = 0.0
        self._num_age_bins = 0
        self._dt_cache_ms = np.nan
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

    @staticmethod
    def _to_scalar_time_ms(value: ArrayLike) -> float:
        dftype = brainstate.environ.dftype()
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.ms), dtype=dftype)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError('Time parameters must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_rate_hz(value: ArrayLike) -> float:
        dftype = brainstate.environ.dftype()
        if isinstance(value, u.Quantity):
            arr = np.asarray(value.to_decimal(u.Hz), dtype=dftype)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError('rate must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

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
        rate: float,
        dead_time: float,
        n_proc: int,
        relative_amplitude: float,
        start: float,
        stop: float,
    ):
        if dead_time < 0.0:
            raise ValueError('The dead time cannot be negative.')

        inv_rate = np.inf if rate == 0.0 else (1000.0 / rate)
        if inv_rate <= dead_time:
            raise ValueError('The inverse rate has to be larger than the dead time.')

        if n_proc < 1:
            raise ValueError('The number of component processes cannot be smaller than one')

        if relative_amplitude < 0.0 or relative_amplitude > 1.0:
            raise ValueError('The relative amplitude of the rate modulation must be in [0,1].')

        if stop < start:
            raise ValueError('stop >= start required.')

    @staticmethod
    def _time_to_step(time_ms: float, dt_ms: float) -> int:
        return int(np.rint(time_ms / dt_ms))

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
        t = brainstate.environ.get('t', default=None)
        if t is None:
            return 0.0
        # Fast path for scalar Quantity (avoids np.asarray round-trip).
        if isinstance(t, u.Quantity):
            return float(t.to_decimal(u.ms))
        return float(t)

    def _refresh_runtime_cache(self, dt_ms: float):
        self._assert_grid_time('origin', self.origin, dt_ms)
        self._assert_grid_time('start', self.start, dt_ms)
        self._assert_grid_time('stop', self.stop, dt_ms)

        self._t_min_step = self._time_to_step(self.origin + self.start, dt_ms)
        if np.isfinite(self.stop):
            self._t_max_step = self._time_to_step(self.origin + self.stop, dt_ms)
        else:
            self._t_max_step = np.iinfo(np.int64).max

        self._num_age_bins = int(self.dead_time / dt_ms)
        self._omega_rad_per_ms = 2.0 * math.pi * self.frequency / 1000.0
        if self.rate > 0.0:
            self._hazard_step = dt_ms / (1000.0 / self.rate - self.dead_time)
        else:
            self._hazard_step = 0.0
        self._dt_cache_ms = float(dt_ms)

    def _is_active(self, curr_step: int) -> bool:
        return (self._t_min_step < curr_step) and (curr_step <= self._t_max_step)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize occupancy arrays and NumPy RNG for all output trains.

        Allocates three :class:`brainstate.ShortTermState` arrays representing
        the age-discretized occupation model and seeds the NumPy random
        generator.  The initial occupancy follows NEST's ``pre_run_hook()``
        logic: ``floor(rate / 1000 * n_proc * dt)`` processes are placed in
        each refractory age bin, and the remainder fills ``occ_active``.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused API placeholder for compatibility with the
            :class:`brainstate.nn.Dynamics` interface. Ignored.
        **kwargs
            Additional unused keyword arguments accepted for API compatibility.
            Ignored.


        Notes
        -----
        If ``dt`` is not available in the simulation environment at call time,
        ``dt_ms`` defaults to ``0.0`` so that ``ini_occ_ref == 0`` and all
        ``n_proc`` processes start in ``occ_active``.  The refractory array is
        still allocated with the correct number of age bins computed from any
        previously cached ``_num_age_bins`` value, which may also be zero if no
        ``dt`` context was ever set.
        """
        del batch_size, kwargs

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)
        else:
            dt_ms = 0.0

        ini_occ_ref = int(self.rate / 1000.0 * self.n_proc * dt_ms)
        ini_occ_act = int(self.n_proc - ini_occ_ref * self._num_age_bins)

        ditype = brainstate.environ.ditype()
        self.occ_refractory = brainstate.ShortTermState(
            np.full(
                (self._num_targets, self._num_age_bins),
                ini_occ_ref,
                dtype=ditype,
            )
        )
        self.occ_active = brainstate.ShortTermState(
            np.full(self._num_targets, ini_occ_act, dtype=ditype)
        )
        self.activate = brainstate.ShortTermState(
            np.zeros(self._num_targets, dtype=ditype)
        )
        self._rng = np.random.default_rng(self.rng_seed)

    def set(
        self,
        *,
        rate: ArrayLike | object = _UNSET,
        dead_time: ArrayLike | object = _UNSET,
        n_proc: ArrayLike | object = _UNSET,
        frequency: ArrayLike | object = _UNSET,
        relative_amplitude: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        r"""Update public generator parameters with NEST-compatible semantics.

        Any parameter left at the internal sentinel ``_UNSET`` retains its
        current value.  All supplied values are converted and cross-validated
        before any attribute is mutated, so the generator state remains
        self-consistent on failure.  If ``dt`` is currently available in
        ``brainstate.environ``, the cached hazard step, angular frequency,
        number of age bins, and timing step bounds are recomputed immediately
        after mutation.

        Parameters
        ----------
        rate : ArrayLike or object, optional
            New scalar component-process rate in Hz.  If omitted, keep the
            current value.  Must satisfy ``1000 / rate > dead_time`` for
            ``rate > 0`` after scalar conversion.
        dead_time : ArrayLike or object, optional
            New scalar absolute refractory duration in ms.  If omitted, keep
            current value.  Must be ``>= 0`` and satisfy
            ``1000 / rate > dead_time`` for nonzero ``rate``.
        n_proc : ArrayLike or object, optional
            New scalar integer number of component processes ``>= 1``.  If
            omitted, keep current value.
        frequency : ArrayLike or object, optional
            New scalar sinusoidal modulation frequency in Hz.  ``0`` disables
            modulation even when ``relative_amplitude > 0``.  If omitted,
            keep current value.
        relative_amplitude : ArrayLike or object, optional
            New scalar dimensionless modulation amplitude in ``[0, 1]``.  If
            omitted, keep current value.
        start : ArrayLike or object, optional
            New scalar relative start time in ms (exclusive lower bound).  If
            omitted, keep current value.
        stop : ArrayLike, None, or object, optional
            New scalar relative stop time in ms (inclusive upper bound).
            ``None`` maps to ``+inf`` (unbounded).  If omitted, keep current
            value.
        origin : ArrayLike or object, optional
            New scalar time-origin offset in ms.  If omitted, keep current
            value.


        Raises
        ------
        ValueError
            If any provided value is non-scalar; if ``dead_time < 0``; if
            ``n_proc < 1``; if ``relative_amplitude`` is outside ``[0, 1]``;
            if ``stop < start``; if ``1000 / rate <= dead_time`` for nonzero
            ``rate``; if integer inputs are non-integral beyond tolerance; or
            if finite timing values are off the simulation grid when ``dt`` is
            available.
        TypeError
            If unit conversion to ``u.Hz`` or ``u.ms`` fails for supplied
            inputs.
        """
        new_dead_time = (
            self.dead_time if dead_time is _UNSET else self._to_scalar_time_ms(dead_time)
        )
        new_rate = self.rate if rate is _UNSET else self._to_scalar_rate_hz(rate)
        new_n_proc = (
            self.n_proc if n_proc is _UNSET else self._to_scalar_int(n_proc, name='n_proc')
        )
        new_frequency = (
            self.frequency if frequency is _UNSET else self._to_scalar_rate_hz(frequency)
        )
        new_relative_amplitude = (
            self.relative_amplitude
            if relative_amplitude is _UNSET
            else self._to_scalar_float(relative_amplitude, name='relative_amplitude')
        )
        new_start = self.start if start is _UNSET else self._to_scalar_time_ms(start)
        if stop is _UNSET:
            new_stop = self.stop
        elif stop is None:
            new_stop = np.inf
        else:
            new_stop = self._to_scalar_time_ms(stop)
        new_origin = self.origin if origin is _UNSET else self._to_scalar_time_ms(origin)

        self._validate_parameters(
            rate=new_rate,
            dead_time=new_dead_time,
            n_proc=new_n_proc,
            relative_amplitude=new_relative_amplitude,
            start=new_start,
            stop=new_stop,
        )

        self.dead_time = new_dead_time
        self.rate = new_rate
        self.n_proc = new_n_proc
        self.frequency = new_frequency
        self.relative_amplitude = new_relative_amplitude
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_runtime_cache(dt_ms)

    def get(self) -> dict:
        r"""Return current public parameters as plain Python scalars.

        Returns
        -------
        out : dict
            ``dict`` with the following keys and value types:

            - ``'rate'`` — ``float``, component-process rate in Hz.
            - ``'dead_time'`` — ``float``, absolute refractory duration in ms.
            - ``'n_proc'`` — ``int``, number of component processes.
            - ``'frequency'`` — ``float``, sinusoidal modulation frequency in Hz.
            - ``'relative_amplitude'`` — ``float``, modulation depth in
              ``[0, 1]``.
            - ``'start'`` — ``float``, relative exclusive lower activity bound
              in ms.
            - ``'stop'`` — ``float``, relative inclusive upper activity bound
              in ms; ``math.inf`` when the generator was constructed or set
              with ``stop=None``.
            - ``'origin'`` — ``float``, time-origin offset in ms.
        """
        return {
            'rate': float(self.rate),
            'dead_time': float(self.dead_time),
            'n_proc': int(self.n_proc),
            'frequency': float(self.frequency),
            'relative_amplitude': float(self.relative_amplitude),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_poisson(self, lam: float) -> int:
        return int(self._rng.poisson(lam))

    def _sample_binomial(self, n: int, p: float) -> int:
        # Clamp only for numerical safety around invalid domain boundaries.
        if p <= 0.0:
            return 0
        if p >= 1.0:
            return int(n)
        return int(self._rng.binomial(n, p))

    def _update_age_distribution_single(
        self,
        occ_ref_row: np.ndarray,
        occ_active: int,
        activate_idx: int,
        hazard_step_t: float,
    ) -> tuple[int, int, int]:
        if occ_active > 0:
            use_poisson_approx = (
                (occ_active >= 100 and hazard_step_t <= 0.01)
                or (occ_active >= 500 and hazard_step_t * occ_active <= 0.1)
            )
            if use_poisson_approx:
                n_spikes = self._sample_poisson(hazard_step_t * occ_active)
                if n_spikes > occ_active:
                    n_spikes = occ_active
            else:
                n_spikes = self._sample_binomial(occ_active, hazard_step_t)
        else:
            n_spikes = 0

        if occ_ref_row.size > 0:
            occ_active = int(occ_active + occ_ref_row[activate_idx] - n_spikes)
            occ_ref_row[activate_idx] = n_spikes
            activate_idx = int((activate_idx + 1) % occ_ref_row.size)

        return int(n_spikes), int(occ_active), int(activate_idx)

    def update(self):
        r"""Advance one simulation step and return per-train spike multiplicity.

        Lazily initializes state on the first call, refreshes the runtime
        cache when ``dt`` changes, applies the active-window test, computes an
        instantaneous hazard (with optional sinusoidal modulation), and updates
        each output train's age-discretized occupation model using NEST's
        branch logic (binomial or Poisson approximation).

        The method mirrors NEST's ``ppd_sup_generator::update`` procedure:

        1. Ensure internal state is initialized; refresh cache if ``dt``
           changed since the last call.
        2. Return zeros immediately when ``rate <= 0`` or no output trains
           exist.
        3. Evaluate the active-window guard:
           :math:`t_{\min} < t \le t_{\max}`.
        4. Compute the per-step hazard:

           .. math::

              h_t = h_{\mathrm{step}}
              \left(1 + A \sin(2\pi f\, t / 1000)\right),

           skipping the sinusoidal term when ``relative_amplitude == 0`` or
           ``frequency == 0``.

        5. For each output train, call
           :meth:`_update_age_distribution_single` which draws
           ``n_spikes`` from the active pool and rotates the refractory ring
           buffer.

        Returns
        -------
        out : jax.Array
            JAX array of dtype ``int64`` and shape ``self.varshape``.  Each
            element is the number of spikes emitted by the corresponding
            output train during the current step.  Returns all-zeros when
            inactive, when ``rate <= 0``, or when no targets are defined.

        Raises
        ------
        KeyError
            If required simulation-context fields (for example ``dt`` via
            ``brainstate.environ.get_dt()``) are unavailable.
        ValueError
            If finite timing parameters are not on the simulation grid after a
            ``dt`` change triggers cache refresh.
        TypeError
            If simulation-time values in the environment cannot be converted
            to scalar milliseconds.
        """
        if not hasattr(self, 'occ_refractory'):
            self.init_state()

        if not np.isfinite(self._dt_cache_ms):
            self._refresh_runtime_cache(self._dt_ms())
        dt_ms = self._dt_cache_ms

        ditype = brainstate.environ.ditype()
        if self.rate <= 0.0 or self._num_targets == 0:
            return np.zeros(self.varshape, dtype=ditype)

        curr_t_ms = self._current_time_ms()
        curr_step = self._time_to_step(curr_t_ms, dt_ms)
        if not self._is_active(curr_step):
            return np.zeros(self.varshape, dtype=ditype)

        if self.relative_amplitude > 0.0 and self.frequency != 0.0:
            hazard_step_t = self._hazard_step * (
                1.0 + self.relative_amplitude * math.sin(self._omega_rad_per_ms * curr_t_ms)
            )
            if hazard_step_t < 0.0 and hazard_step_t > -1e-15:
                hazard_step_t = 0.0
        else:
            hazard_step_t = self._hazard_step

        occ_ref = np.asarray(self.occ_refractory.value, dtype=ditype).copy()
        occ_active = np.asarray(self.occ_active.value, dtype=ditype).copy()
        activate = np.asarray(self.activate.value, dtype=ditype).copy()
        counts = np.zeros(self._num_targets, dtype=ditype)

        for idx in range(self._num_targets):
            n_spikes, occ_act_i, activate_i = self._update_age_distribution_single(
                occ_ref[idx],
                int(occ_active[idx]),
                int(activate[idx]),
                hazard_step_t,
            )
            counts[idx] = n_spikes
            occ_active[idx] = occ_act_i
            activate[idx] = activate_i

        self.occ_refractory.value = occ_ref
        self.occ_active.value = occ_active
        self.activate.value = activate
        return counts.reshape(self.varshape)
