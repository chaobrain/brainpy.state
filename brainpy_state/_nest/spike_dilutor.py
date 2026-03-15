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
import saiunit as u
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTDevice

__all__ = [
    'spike_dilutor',
]

_UNSET = object()


class spike_dilutor(NESTDevice):
    r"""NEST-compatible ``spike_dilutor`` device.

    Short description
    -----------------

    Dilute an incoming mother spike multiplicity into independent child
    multiplicities by Bernoulli copying, one trial per
    ``(target, mother-spike)`` pair.

    Description
    -----------

    ``spike_dilutor`` mirrors NEST ``models/spike_dilutor.cpp``.
    On each active simulation step, the device reads one scalar mother
    multiplicity, then independently assigns each of the :math:`M` child
    targets a copy count drawn from a Binomial distribution with that
    multiplicity and copy probability :math:`p_{\mathrm{copy}}`.

    Outputs are integer multiplicities (``0, 1, 2, ...``), matching NEST
    ``SpikeEvent`` multiplicity semantics rather than binary spikes.

    **1. Model equations and distributional properties**

    Let :math:`N_m(n)` be the incoming mother multiplicity at simulation step
    :math:`n`, and let :math:`p = p_{\mathrm{copy}} \in [0, 1]`. For target
    :math:`j \in \{1, \dots, M\}` with :math:`M=\prod\mathrm{varshape}`:

    .. math::

       N_j(n)=\sum_{k=1}^{N_m(n)} \mathbf{1}[U_{j,k}<p], \quad
       U_{j,k}\sim\mathrm{Uniform}(0,1),

    so conditionally:

    .. math::

       N_j(n)\mid N_m(n)\sim\mathrm{Binomial}(N_m(n),\, p).

    Hence the per-target moments are:

    .. math::

       \mathbb{E}[N_j\mid N_m]=N_m p,\quad
       \mathrm{Var}[N_j\mid N_m]=N_m p(1-p).

    **2. NEST-equivalent update ordering**

    NEST ``models/spike_dilutor.cpp`` evaluates activity, reads one mother
    multiplicity, then in ``event_hook()`` performs explicit Bernoulli loops
    independently for each receiver. This implementation preserves that behavior
    by generating one copied multiplicity per element of ``self.varshape`` from
    the same mother multiplicity in the current step.

    **3. Timing semantics, assumptions, and constraints**

    Activity uses the NEST stimulation-device interval:

    .. math::

       t_{\min} < t \le t_{\max},

    with :math:`t_{\min}=\mathrm{origin}+\mathrm{start}` and
    :math:`t_{\max}=\mathrm{origin}+\mathrm{stop}`. Therefore ``start`` is
    exclusive and ``stop`` is inclusive.

    Grid constraints are enforced when the timing cache is refreshed:

    - Finite ``origin``, ``start``, and ``stop`` must be integer multiples of
      ``dt`` (checked with tight absolute tolerance of ``1e-12``).
    - ``stop >= start`` must hold.
    - Cached step indices are recomputed if the runtime ``dt`` changes between
      calls.

    Mother multiplicity is taken as the sum of the direct ``mother_spikes``
    argument plus any values registered via :meth:`add_current_input` and
    :meth:`add_delta_input`, then truncated toward zero to a non-negative
    integer count.

    **4. Computational implications**

    For ``0 < p_copy < 1``, one update draws a random array of shape
    ``(prod(varshape), n_mother_spikes)`` and counts successes per target.
    Time and temporary-memory complexity are both
    :math:`O(\prod\mathrm{varshape}\cdot n_{\mathrm{mother}})`. Fast paths for
    ``p_copy`` equal to ``0`` or ``1`` avoid random sampling entirely.

    Parameters
    ----------
    in_size : Size, optional
        Output shape specification passed to :class:`Dynamics`. The emitted
        child multiplicity array has shape ``self.varshape`` derived from
        ``in_size``. Default is ``1``.
    p_copy : ArrayLike, optional
        Scalar Bernoulli copy probability :math:`p_{\mathrm{copy}}`.
        Accepted as a scalar-like numeric array or value; converted internally
        to Python ``float`` and validated in ``[0, 1]``. Unitless.
        Default is ``1.0``.
    start : ArrayLike, optional
        Relative start time (ms). Scalar-convertible; active window lower bound
        is ``origin + start`` and is **exclusive**. Must be an integer multiple
        of ``dt`` when finite. Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Relative stop time (ms). ``None`` maps to ``+inf`` (no upper bound).
        When finite, upper bound ``origin + stop`` is **inclusive** and must
        satisfy ``stop >= start``. Must be an integer multiple of ``dt`` when
        finite. Default is ``None``.
    origin : ArrayLike, optional
        Global time offset (ms) added to both ``start`` and ``stop``.
        Scalar-convertible. Must be an integer multiple of ``dt`` when finite.
        Default is ``0.0 * u.ms``.
    rng_seed : int, optional
        Integer seed for NumPy ``default_rng`` used for Bernoulli copy draws.
        The RNG is re-initialised in :meth:`init_state`. Default is ``0``.
    name : str or None, optional
        Optional node name passed to :class:`Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 20 16 24 40

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``p_copy``
         - ``1.0``
         - :math:`p_{\mathrm{copy}}`
         - Bernoulli copy probability used in each mother-spike trial.
       * - ``start``
         - ``0.0 * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative lower time bound; active only for ``t > origin + start``.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative upper time bound; finite value is active for
           ``t <= origin + stop``.
       * - ``origin``
         - ``0.0 * u.ms``
         - :math:`t_0`
         - Global offset added to both relative bounds.
       * - ``in_size``
         - ``1``
         - :math:`M`
         - Number/shape of child targets; ``M = prod(varshape)``.

    Raises
    ------
    ValueError
        If ``p_copy`` is outside ``[0, 1]``, if ``stop < start``, if any time
        parameter is non-scalar or not an integer multiple of ``dt``, or if the
        effective mother multiplicity for a step is negative.
    TypeError
        If parameters cannot be converted to the required numeric scalar types.
    KeyError
        At update time, if the simulation context does not provide the required
        ``dt`` value via ``brainstate.environ.get_dt()``.

    See Also
    --------
    bernoulli_synapse : Per-spike Bernoulli transmission at the synapse level.
    spike_generator : Deterministic spike injection device.

    Notes
    -----
    - Incoming mother spikes are provided through the ``mother_spikes``
      argument of :meth:`update`, and can also be accumulated via
      :meth:`add_delta_input` / :meth:`add_current_input`.
    - Like NEST, this model is deprecated in favour of probabilistic synapses
      (e.g., ``bernoulli_synapse``), which operate at the connection level.
    - NEST restricts ``spike_dilutor`` to single-threaded simulations. This
      backend does not expose NEST thread kernels, so that restriction is not
      modelled here.

    Examples
    --------
    Dilute a mother multiplicity of 3 into 4 independent child channels:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     sd = brainpy.state.spike_dilutor(
       ...         in_size=4,
       ...         p_copy=0.25,
       ...         start=0.0 * u.ms,
       ...         stop=5.0 * u.ms,
       ...         rng_seed=123,
       ...     )
       ...     sd.init_state()
       ...     with brainstate.environ.context(t=1.0 * u.ms):
       ...         y = sd.update(mother_spikes=3)
       ...     _ = (y.shape, y.dtype)  # ((4,), int64)

    Pass-through mode (``p_copy=1.0``) with a 2-D output shape:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     sd = brainpy.state.spike_dilutor(p_copy=1.0, in_size=(2, 2))
       ...     sd.init_state()
       ...     with brainstate.environ.context(t=2.0 * u.ms):
       ...         y = sd.update(mother_spikes=5)
       ...     _ = y.sum()  # == 20  (4 targets × 5 spikes)

    References
    ----------
    .. [1] NEST source: ``models/spike_dilutor.h`` and
           ``models/spike_dilutor.cpp``.
    .. [2] NEST docs:
           https://nest-simulator.readthedocs.io/en/stable/models/spike_dilutor.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        p_copy: ArrayLike = 1.0,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        rng_seed: int = 0,
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.p_copy = self._to_scalar_float(p_copy, name='p_copy')
        self.start = self._to_scalar_time_ms(start)
        self.stop = np.inf if stop is None else self._to_scalar_time_ms(stop)
        self.origin = self._to_scalar_time_ms(origin)
        self.rng_seed = int(rng_seed)

        self._validate_parameters(
            p_copy=self.p_copy,
            start=self.start,
            stop=self.stop,
        )

        self._num_targets = int(np.prod(self.varshape))
        self._dt_cache_ms = np.nan
        self._t_min_step = 0
        self._t_max_step = np.iinfo(np.int64).max
        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

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
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _validate_parameters(
        *,
        p_copy: float,
        start: float,
        stop: float,
    ):
        if p_copy < 0.0 or p_copy > 1.0:
            raise ValueError('Copy probability must be in [0, 1].')
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

    @staticmethod
    def _to_nonnegative_count(value: ArrayLike) -> int:
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        total = float(arr.sum())
        if total < 0.0:
            raise ValueError('mother_spikes must be non-negative.')
        return int(np.trunc(total))

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

    def _refresh_timing_cache(self, dt_ms: float):
        self._assert_grid_time('origin', self.origin, dt_ms)
        self._assert_grid_time('start', self.start, dt_ms)
        self._assert_grid_time('stop', self.stop, dt_ms)

        self._t_min_step = self._time_to_step(self.origin + self.start, dt_ms)
        if np.isfinite(self.stop):
            self._t_max_step = self._time_to_step(self.origin + self.stop, dt_ms)
        else:
            self._t_max_step = np.iinfo(np.int64).max
        self._dt_cache_ms = float(dt_ms)

    def _is_active(self, curr_step: int) -> bool:
        return (self._t_min_step < curr_step) and (curr_step <= self._t_max_step)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialise the per-instance NumPy random number generator.

        Constructs a ``numpy.random.default_rng`` seeded with
        :attr:`rng_seed`.  Must be called before the first :meth:`update`
        call; :meth:`update` calls it automatically when the RNG is absent,
        but explicit initialisation is preferred for reproducibility.

        Parameters
        ----------
        batch_size : int or None, optional
            Unused; accepted for API compatibility with
            :class:`brainstate.nn.Dynamics`. Default is ``None``.
        **kwargs
            Unused; accepted for forward compatibility.

        Notes
        -----
        Calling :meth:`init_state` a second time resets the RNG to the
        initial seed, making simulation runs reproducible when seeded.
        """
        del batch_size, kwargs
        self._rng = np.random.default_rng(self.rng_seed)

    def set(
        self,
        *,
        p_copy: ArrayLike | object = _UNSET,
        start: ArrayLike | object = _UNSET,
        stop: ArrayLike | object = _UNSET,
        origin: ArrayLike | object = _UNSET,
    ):
        r"""Update public parameters with NEST-style validation.

        Updates one or more device parameters.  All arguments are keyword-only
        and optional; unspecified parameters retain their current values.
        If a timing parameter is provided and ``dt`` is available in the
        environment, the step-index cache is refreshed immediately.

        Parameters
        ----------
        p_copy : ArrayLike, optional
            New Bernoulli copy probability. Scalar-convertible; must lie in
            ``[0, 1]``. Raises ``ValueError`` if the constraint is violated.
        start : ArrayLike, optional
            New relative start time (ms). Scalar-convertible; must be an
            integer multiple of ``dt`` when finite.
        stop : ArrayLike or None, optional
            New relative stop time (ms). ``None`` maps to ``+inf``.
            Must satisfy ``stop >= start``.
        origin : ArrayLike, optional
            New global time offset (ms). Scalar-convertible.

        Raises
        ------
        ValueError
            If ``p_copy`` is outside ``[0, 1]``, if ``stop < start`` after
            the update, if any time value is non-scalar, or if a finite time
            value is not an integer multiple of the current ``dt``.
        TypeError
            If a parameter cannot be converted to the required numeric type.

        Examples
        --------
        .. code-block:: python

           >>> import brainpy
           >>> import brainstate
           >>> import saiunit as u
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     sd = brainpy.state.spike_dilutor(p_copy=0.5)
           ...     sd.set(p_copy=0.8, stop=10.0 * u.ms)
           ...     _ = sd.p_copy  # 0.8
        """
        new_p_copy = (
            self.p_copy
            if p_copy is _UNSET
            else self._to_scalar_float(p_copy, name='p_copy')
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
            p_copy=new_p_copy,
            start=new_start,
            stop=new_stop,
        )

        self.p_copy = new_p_copy
        self.start = new_start
        self.stop = new_stop
        self.origin = new_origin

        dt_ms = self._maybe_dt_ms()
        if dt_ms is not None:
            self._refresh_timing_cache(dt_ms)

    def get(self) -> dict:
        r"""Return current public parameters as plain Python scalars.

        Returns
        -------
        dict
            Dictionary containing all device parameters:

            - ``'p_copy'`` : float — Bernoulli copy probability in ``[0, 1]``.
            - ``'start'`` : float — relative start time in ms (exclusive bound).
            - ``'stop'`` : float — relative stop time in ms (inclusive bound);
              ``+inf`` when no upper bound is configured.
            - ``'origin'`` : float — global time offset in ms.

        Examples
        --------
        .. code-block:: python

           >>> import brainpy
           >>> import brainstate
           >>> import saiunit as u
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     sd = brainpy.state.spike_dilutor(p_copy=0.3, stop=5.0 * u.ms)
           ...     params = sd.get()
           ...     _ = params['p_copy']   # 0.3
           ...     _ = params['stop']     # 5.0
        """
        return {
            'p_copy': float(self.p_copy),
            'start': float(self.start),
            'stop': float(self.stop),
            'origin': float(self.origin),
        }

    def _sample_child_spikes(self, n_mother_spikes: int) -> np.ndarray:
        r"""Draw Binomial child multiplicities for all targets.

        For each of the :math:`M = \prod\mathrm{varshape}` targets, sums
        ``n_mother_spikes`` independent Bernoulli trials with success
        probability :math:`p_{\mathrm{copy}}`.  Three fast paths avoid
        random sampling when the result is deterministic:

        - ``n_mother_spikes <= 0`` or ``_num_targets == 0``: return zeros.
        - ``p_copy <= 0.0``: return zeros (no copies ever succeed).
        - ``p_copy >= 1.0``: return ``n_mother_spikes`` for every target.

        Parameters
        ----------
        n_mother_spikes : int
            Non-negative integer mother multiplicity for the current step.

        Returns
        -------
        out : numpy.ndarray
            1-D array of dtype ``int64`` with length ``self._num_targets``.
            Element ``j`` gives the number of mother spikes copied to child
            target ``j``.  Caller is responsible for reshaping to
            ``self.varshape``.

        Notes
        -----
        Random draws are taken from ``self._rng`` (a ``numpy.random.Generator``
        initialised in :meth:`init_state`).  The draw allocates a temporary
        array of shape ``(self._num_targets, n_mother_spikes)``, which may be
        large for high-multiplicity inputs.
        """
        ditype = brainstate.environ.ditype()
        out = np.zeros(self._num_targets, dtype=ditype)

        if n_mother_spikes <= 0 or self._num_targets == 0:
            return out
        if self.p_copy <= 0.0:
            return out
        if self.p_copy >= 1.0:
            out.fill(int(n_mother_spikes))
            return out

        # One Bernoulli trial per (target, mother spike), matching NEST's
        # explicit event_hook loop semantics.
        draws = self._rng.random((self._num_targets, int(n_mother_spikes)))
        out[:] = np.count_nonzero(draws < self.p_copy, axis=1).astype(np.int64)
        return out

    def update(self, mother_spikes: ArrayLike = 0.0):
        r"""Advance one simulation step and emit child spike multiplicities.

        Reads the current simulation time from ``brainstate.environ``,
        checks device activity, and—when active—delegates to
        :meth:`_sample_child_spikes` to draw independent Binomial counts for
        each target.  If the timing cache is stale (``dt`` changed since last
        call), the cache is refreshed before activity is evaluated.

        Parameters
        ----------
        mother_spikes : ArrayLike, optional
            Mother-process multiplicity contribution for the current step.
            Values are summed element-wise over all array elements, then
            combined with any values registered via :meth:`add_current_input`
            and :meth:`add_delta_input`.  The resulting total is truncated
            toward zero to a non-negative integer count.  Unitless count
            semantics. Default is ``0.0``.

        Returns
        -------
        out : numpy.ndarray
            Integer array with dtype ``int64`` and shape ``self.varshape``.
            Each element gives the copied child multiplicity for the
            corresponding target in the current simulation step.  Returns all
            zeros when the device is inactive or when the effective mother
            multiplicity is zero.

        Raises
        ------
        ValueError
            If the effective mother multiplicity is negative after combining
            all inputs, or if a finite timing parameter is not an integer
            multiple of the current ``dt``.
        TypeError
            If ``mother_spikes`` cannot be converted to a numeric array.
        KeyError
            If required simulation context (e.g. ``dt``) is unavailable
            depending on ``brainstate.environ`` behaviour.

        See Also
        --------
        _sample_child_spikes : Low-level Binomial sampling routine.
        init_state : Initialise the RNG before calling update.
        """
        if not hasattr(self, '_rng'):
            self.init_state()

        dt_ms = self._dt_ms()
        if (not np.isfinite(self._dt_cache_ms)) or (
            not math.isclose(dt_ms, self._dt_cache_ms, rel_tol=0.0, abs_tol=1e-15)
        ):
            self._refresh_timing_cache(dt_ms)

        # Mother multiplicity for the current step: direct input argument plus
        # optional registered current/delta inputs.
        total_spikes = self.sum_current_inputs(mother_spikes)
        total_spikes = self.sum_delta_inputs(total_spikes)
        n_mother_spikes = self._to_nonnegative_count(total_spikes)

        curr_step = self._time_to_step(self._current_time_ms(), dt_ms)
        if not self._is_active(curr_step) or n_mother_spikes <= 0:
            ditype = brainstate.environ.ditype()
            return np.zeros(self.varshape, dtype=ditype)

        child_counts = self._sample_child_spikes(n_mother_spikes)
        return child_counts.reshape(self.varshape)
