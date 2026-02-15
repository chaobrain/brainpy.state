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

from __future__ import annotations

import math
from collections.abc import Mapping

import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike

from .static_synapse import _UNSET, static_synapse

__all__ = [
    'stdp_facetshw_synapse_hom',
]


_STDP_EPS = 1.0e-6
_LUT_ENTRY_MIN = 0
_LUT_ENTRY_MAX = 15
_DEFAULT_LUT_0 = (2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15)
_DEFAULT_LUT_1 = (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13)
_DEFAULT_LUT_2 = tuple(range(16))
_DEFAULT_CONFIG_0 = (0, 0, 1, 0)
_DEFAULT_CONFIG_1 = (0, 1, 0, 0)
_DEFAULT_RESET_PATTERN = (1, 1, 1, 1, 1, 1)


class stdp_facetshw_synapse_hom(static_synapse):
    r"""NEST-compatible ``stdp_facetshw_synapse_hom`` connection model.

    Short description
    -----------------

    FACETS hardware-constrained spike-timing dependent plasticity with
    homogeneous (model-level) plasticity parameters.

    Description
    -----------

    ``stdp_facetshw_synapse_hom`` mirrors NEST
    ``models/stdp_facetshw_synapse_hom.h`` and
    ``models/stdp_facetshw_synapse_hom_impl.h``. It is a modified STDP rule
    designed for hardware constraints (BrainScaleS / FACETS), including:

    - 4-bit discrete weight representation through look-up tables (LUTs),
    - periodic controller-driven readout/update cycles,
    - reduced symmetric nearest-neighbor spike pairing,
    - configurable capacitor reset patterns after LUT updates.

    Per-connection mutable state is:

    - ``weight``: current continuous weight,
    - ``a_causal`` and ``a_acausal``: causal/acausal accumulator charges,
    - ``a_thresh_th`` and ``a_thresh_tl``: comparator thresholds,
    - ``init_flag`` / ``synapse_id`` / ``next_readout_time``: controller state,
    - ``t_lastspike``: previous presynaptic spike timestamp.

    Common model-level properties are:

    - ``tau_plus`` and ``tau_minus_stdp`` (causal/acausal time constants),
    - ``Wmax`` and ``weight_per_lut_entry``,
    - driver parameters (``no_synapses``, ``synapses_per_driver``,
      ``driver_readout_time``, ``readout_cycle_duration``),
    - LUT/configuration vectors (``lookuptable_0/1/2``, ``configbit_0/1``,
      ``reset_pattern``).

    Update order (NEST source equivalent)
    -------------------------------------

    For a presynaptic spike at stamp :math:`t_{pre}` with dendritic delay
    :math:`d`, NEST ``stdp_facetshw_synapse_hom::send`` performs:

    1. On first presynaptic activity, assign ``synapse_id`` from
       ``no_synapses``, increment ``no_synapses``, recalculate readout cycle,
       and initialize ``next_readout_time``.
    2. If ``t_pre > next_readout_time``:
       1. Convert continuous weight to 4-bit LUT entry.
       2. Evaluate two comparator bits from
          ``(a_causal, a_acausal, a_thresh_th, a_thresh_tl)`` and config bits.
       3. Select LUT (0/1/2) according to bit pair
          ``(eval_0, eval_1) = (1,0), (0,1), (1,1)`` and update discrete
          weight.
       4. Apply reset bits (causal/acausal) for selected LUT.
       5. Advance ``next_readout_time`` in steps of ``readout_cycle_duration``
          until ``t_pre <= next_readout_time``.
       6. Convert updated discrete weight back to continuous value.
    3. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\; t_{pre}-d]`.
    4. If history is non-empty:
       1. Use the first postsynaptic spike in that interval to update
          ``a_causal``.
       2. Use the last postsynaptic spike in that interval to update
          ``a_acausal``.
    5. Send event with updated ``weight``.
    6. Set ``t_lastspike = t_pre``.

    This implementation preserves the same ordering.

    Event timing semantics
    ----------------------

    As in NEST, this model uses on-grid spike stamps and ignores precise
    sub-step offsets for plasticity calculations.

    Parameters
    ----------
    weight : ArrayLike, optional
        Initial synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    tau_plus : ArrayLike, optional
        Time constant of the causal branch in ms. Must be ``> 0``.
        Default: ``20.0 * u.ms``.
    tau_minus_stdp : ArrayLike, optional
        Time constant of the anti-causal branch in ms. Must be ``> 0``.
        Default: ``20.0 * u.ms``.
    Wmax : ArrayLike, optional
        Maximum biological weight used for LUT conversion.
        Default: ``100.0``.
    weight_per_lut_entry : ArrayLike, optional
        Conversion factor between LUT index and continuous weight. If omitted,
        computed as ``Wmax / (len(lookuptable_0)-1)``.
    no_synapses : ArrayLike, optional
        Total synapse count used by the simulated controller.
        Default: ``0``.
    synapses_per_driver : ArrayLike, optional
        Number of synapses updated in one driver readout.
        Must be positive. Default: ``50``.
    driver_readout_time : ArrayLike, optional
        Processing time per row (ms). Must be positive.
        Default: ``15.0``.
    readout_cycle_duration : ArrayLike, optional
        Readout cycle duration. If omitted, computed from
        ``no_synapses``, ``synapses_per_driver`` and ``driver_readout_time``.
    lookuptable_0 : ArrayLike, optional
        LUT for evaluation bits ``(1,0)``. Entries must be integers in
        ``[0, 15]``. Default follows NEST.
    lookuptable_1 : ArrayLike, optional
        LUT for evaluation bits ``(0,1)``. Entries must be integers in
        ``[0, 15]``. Default follows NEST.
    lookuptable_2 : ArrayLike, optional
        LUT for evaluation bits ``(1,1)``. Entries must be integers in
        ``[0, 15]``. Default is identity LUT.
    configbit_0 : ArrayLike, optional
        First 4-bit evaluation configuration vector. Default: ``[0,0,1,0]``.
    configbit_1 : ArrayLike, optional
        Second 4-bit evaluation configuration vector. Default: ``[0,1,0,0]``.
    reset_pattern : ArrayLike, optional
        Six reset bits (causal/acausal pair per LUT selection). Default: all 1.
    a_causal : ArrayLike, optional
        Initial causal accumulator value. Default: ``0.0``.
    a_acausal : ArrayLike, optional
        Initial acausal accumulator value. Default: ``0.0``.
    a_thresh_th : ArrayLike, optional
        Upper comparator threshold. Default: ``21.835``.
    a_thresh_tl : ArrayLike, optional
        Lower comparator threshold. Default: ``21.835``.
    init_flag : ArrayLike, optional
        Initialized flag for controller state. Default: ``False``.
    synapse_id : ArrayLike, optional
        Initial synapse id. Default: ``0``.
    next_readout_time : ArrayLike, optional
        Initial scheduled readout time in ms. Default: ``0.0``.
    post : object, optional
        Default receiver object.
    name : str, optional
        Object name.

    Notes
    -----
    - This model transmits spike-like events only.
    - ``update(pre_spike=..., post_spike=...)`` supports standalone simulation
      with explicit pre/post spike multiplicities.
    - The model intentionally ignores precise spike timing offsets, as in NEST.
    - Common-property keys are rejected in connect-time synapse specs via
      :meth:`check_synapse_params`.

    References
    ----------
    .. [1] NEST source: ``models/stdp_facetshw_synapse_hom.h``,
           ``models/stdp_facetshw_synapse_hom_impl.h`` and
           ``models/stdp_facetshw_synapse_hom.cpp``.
    .. [2] Morrison A, Diesmann M, Gerstner W (2008). Phenomenological models
           of synaptic plasticity based on spike timing.
           Biological Cybernetics, 98:459-478.
    .. [3] Pfeil T et al. (2012). Is a 4-bit synaptic weight resolution enough?
           Frontiers in Neuroscience, 6:90.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus_stdp: ArrayLike = 20.0 * u.ms,
        Wmax: ArrayLike = 100.0,
        weight_per_lut_entry: ArrayLike | object = _UNSET,
        no_synapses: ArrayLike = 0,
        synapses_per_driver: ArrayLike = 50,
        driver_readout_time: ArrayLike = 15.0,
        readout_cycle_duration: ArrayLike | object = _UNSET,
        lookuptable_0: ArrayLike = _DEFAULT_LUT_0,
        lookuptable_1: ArrayLike = _DEFAULT_LUT_1,
        lookuptable_2: ArrayLike = _DEFAULT_LUT_2,
        configbit_0: ArrayLike = _DEFAULT_CONFIG_0,
        configbit_1: ArrayLike = _DEFAULT_CONFIG_1,
        reset_pattern: ArrayLike = _DEFAULT_RESET_PATTERN,
        a_causal: ArrayLike = 0.0,
        a_acausal: ArrayLike = 0.0,
        a_thresh_th: ArrayLike = 21.835,
        a_thresh_tl: ArrayLike = 21.835,
        init_flag: ArrayLike = False,
        synapse_id: ArrayLike = 0,
        next_readout_time: ArrayLike = 0.0,
        post=None,
        name: str | None = None,
    ):
        weight_value = self._to_scalar_float(weight, name='weight')
        super().__init__(
            weight=weight_value,
            delay=delay,
            receptor_type=receptor_type,
            post=post,
            event_type='spike',
            name=name,
        )

        self.tau_plus = self._to_scalar_time_ms(tau_plus, name='tau_plus')
        self.tau_minus_stdp = self._to_scalar_time_ms(tau_minus_stdp, name='tau_minus_stdp')
        self.Wmax = self._to_scalar_float(Wmax, name='Wmax')

        self._validate_positive(self.tau_plus, name='tau_plus')
        self._validate_positive(self.tau_minus_stdp, name='tau_minus_stdp')

        self.no_synapses = self._to_int_scalar(no_synapses, name='no_synapses')
        self.synapses_per_driver = self._to_int_scalar(synapses_per_driver, name='synapses_per_driver')
        self.driver_readout_time = self._to_scalar_float(driver_readout_time, name='driver_readout_time')

        self._validate_synapses_per_driver(self.synapses_per_driver)
        self._validate_positive(self.driver_readout_time, name='driver_readout_time')

        self.lookuptable_0 = self._to_int_vector(
            lookuptable_0,
            name='lookuptable_0',
            exact_size=16,
            min_value=_LUT_ENTRY_MIN,
            max_value=_LUT_ENTRY_MAX,
        )
        self.lookuptable_1 = self._to_int_vector(
            lookuptable_1,
            name='lookuptable_1',
            exact_size=16,
            min_value=_LUT_ENTRY_MIN,
            max_value=_LUT_ENTRY_MAX,
        )
        self.lookuptable_2 = self._to_int_vector(
            lookuptable_2,
            name='lookuptable_2',
            exact_size=16,
            min_value=_LUT_ENTRY_MIN,
            max_value=_LUT_ENTRY_MAX,
        )

        self._validate_lut_size_match(self.lookuptable_0, self.lookuptable_1)
        self._validate_lut_size_match(self.lookuptable_0, self.lookuptable_2)

        self.configbit_0 = self._to_int_vector(configbit_0, name='configbit_0', exact_size=4)
        self.configbit_1 = self._to_int_vector(configbit_1, name='configbit_1', exact_size=4)
        self.reset_pattern = self._to_int_vector(reset_pattern, name='reset_pattern', exact_size=6)

        if weight_per_lut_entry is _UNSET:
            self.weight_per_lut_entry = float(self.Wmax / (len(self.lookuptable_0) - 1))
        else:
            self.weight_per_lut_entry = self._to_scalar_float(weight_per_lut_entry, name='weight_per_lut_entry')

        if readout_cycle_duration is _UNSET:
            self.readout_cycle_duration = 0.0
            self._calc_readout_cycle_duration()
        else:
            self.readout_cycle_duration = self._to_scalar_float(readout_cycle_duration, name='readout_cycle_duration')

        self.a_causal = self._to_scalar_float(a_causal, name='a_causal')
        self.a_acausal = self._to_scalar_float(a_acausal, name='a_acausal')
        self.a_thresh_th = self._to_scalar_float(a_thresh_th, name='a_thresh_th')
        self.a_thresh_tl = self._to_scalar_float(a_thresh_tl, name='a_thresh_tl')
        self.init_flag = self._to_bool_scalar(init_flag, name='init_flag')
        self.synapse_id = self._to_int_scalar(synapse_id, name='synapse_id')
        self.next_readout_time = self._to_scalar_float(next_readout_time, name='next_readout_time')

        self.discrete_weight = 0
        self.t_lastspike = 0.0
        self._post_hist_t: list[float] = []

        self._a_causal0 = float(self.a_causal)
        self._a_acausal0 = float(self.a_acausal)
        self._a_thresh_th0 = float(self.a_thresh_th)
        self._a_thresh_tl0 = float(self.a_thresh_tl)
        self._init_flag0 = bool(self.init_flag)
        self._synapse_id0 = int(self.synapse_id)
        self._next_readout_time0 = float(self.next_readout_time)
        self._no_synapses0 = int(self.no_synapses)
        self._readout_cycle_duration0 = float(self.readout_cycle_duration)

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        if isinstance(value, u.Quantity):
            unit = u.get_unit(value)
            arr = np.asarray(value.to_decimal(unit), dtype=np.float64)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr.reshape(()))
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        return v

    @classmethod
    def _to_int_scalar(cls, value: ArrayLike, *, name: str) -> int:
        v = cls._to_scalar_float(value, name=name)
        rounded = int(round(v))
        if not math.isclose(v, float(rounded), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be an integer.')
        return rounded

    @classmethod
    def _to_bool_scalar(cls, value: ArrayLike, *, name: str) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        ivalue = cls._to_int_scalar(value, name=name)
        if ivalue not in (0, 1):
            raise ValueError(f'{name} must be boolean-like (0/1/False/True).')
        return bool(ivalue)

    @classmethod
    def _to_non_negative_int_count(cls, value: ArrayLike, *, name: str) -> int:
        count = cls._to_int_scalar(value, name=name)
        if count < 0:
            raise ValueError(f'{name} must be non-negative.')
        return count

    @classmethod
    def _to_int_vector(
        cls,
        value: ArrayLike,
        *,
        name: str,
        exact_size: int | None = None,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> list[int]:
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        flat = np.asarray(arr.reshape(-1), dtype=np.float64)
        if exact_size is not None and flat.size != exact_size:
            raise ValueError(f'{name} must contain exactly {exact_size} entries.')
        values: list[int] = []
        for raw in flat:
            if not np.isfinite(raw):
                raise ValueError(f'{name} entries must be finite.')
            i = int(round(float(raw)))
            if not math.isclose(float(raw), float(i), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f'{name} entries must be integers.')
            if min_value is not None and i < min_value:
                raise ValueError(f'{name} entries must be in [{min_value},{max_value}].')
            if max_value is not None and i > max_value:
                raise ValueError(f'{name} entries must be in [{min_value},{max_value}].')
            values.append(i)
        return values

    @staticmethod
    def _validate_positive(value: float, *, name: str):
        if value <= 0.0:
            raise ValueError(f'{name} must be > 0.')

    @staticmethod
    def _validate_synapses_per_driver(value: int):
        if value <= 0:
            raise ValueError('synapses_per_driver must be > 0.')

    @staticmethod
    def _validate_lut_size_match(left: list[int], right: list[int]):
        if len(left) != len(right):
            raise ValueError('Look-up table has not 2^4 entries.')

    @staticmethod
    def _round_half_away_from_zero(x: float) -> int:
        if x >= 0.0:
            return int(math.floor(x + 0.5))
        return int(math.ceil(x - 0.5))

    def _calc_readout_cycle_duration(self):
        self.readout_cycle_duration = (
            int((self.no_synapses - 1.0) / self.synapses_per_driver + 1.0) * self.driver_readout_time
        )

    @staticmethod
    def _eval_function(
        a_causal: float,
        a_acausal: float,
        a_thresh_th: float,
        a_thresh_tl: float,
        configbit: list[int],
    ) -> bool:
        return (
            (a_thresh_tl + configbit[2] * a_causal + configbit[1] * a_acausal)
            / (1 + configbit[2] + configbit[1])
            > (a_thresh_th + configbit[0] * a_causal + configbit[3] * a_acausal)
            / (1 + configbit[0] + configbit[3])
        )

    @classmethod
    def _weight_to_entry(cls, weight: float, weight_per_lut_entry: float) -> int:
        return cls._round_half_away_from_zero(weight / weight_per_lut_entry)

    @staticmethod
    def _entry_to_weight(discrete_weight: int, weight_per_lut_entry: float) -> float:
        return float(discrete_weight * weight_per_lut_entry)

    @staticmethod
    def _lookup(discrete_weight: int, table: list[int]) -> int:
        if discrete_weight < 0 or discrete_weight >= len(table):
            raise ValueError(
                f'Discrete weight index {discrete_weight} is out of LUT bounds [0, {len(table) - 1}].'
            )
        return int(table[discrete_weight])

    def _record_post_spike_at(self, t_spike_ms: float):
        self._post_hist_t.append(float(t_spike_ms))

    def record_post_spike(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        t_spike_ms: ArrayLike | None = None,
    ) -> int:
        """Record postsynaptic spikes into internal history."""
        count = self._to_non_negative_int_count(multiplicity, name='post_spike')
        if count == 0:
            return 0

        if t_spike_ms is None:
            dt_ms = self._refresh_delay_if_needed()
            t_value = self._current_time_ms() + dt_ms
        else:
            t_value = self._to_scalar_float(t_spike_ms, name='t_spike_ms')

        for _ in range(count):
            self._record_post_spike_at(float(t_value))
        return count

    def _get_post_history_times(self, t1_ms: float, t2_ms: float) -> list[float]:
        t1_lim = float(t1_ms + _STDP_EPS)
        t2_lim = float(t2_ms + _STDP_EPS)
        selected = []
        for t_post in self._post_hist_t:
            if t_post >= t1_lim and t_post < t2_lim:
                selected.append(float(t_post))
        return selected

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        super().init_state()
        self.a_causal = float(self._a_causal0)
        self.a_acausal = float(self._a_acausal0)
        self.a_thresh_th = float(self._a_thresh_th0)
        self.a_thresh_tl = float(self._a_thresh_tl0)
        self.init_flag = bool(self._init_flag0)
        self.synapse_id = int(self._synapse_id0)
        self.next_readout_time = float(self._next_readout_time0)
        self.no_synapses = int(self._no_synapses0)
        self.readout_cycle_duration = float(self._readout_cycle_duration0)
        self.discrete_weight = 0
        self.t_lastspike = 0.0
        self._post_hist_t = []

    def get(self) -> dict:
        """Return current public parameters and mutable state."""
        params = super().get()
        params['tau_plus'] = float(self.tau_plus)
        params['tau_minus_stdp'] = float(self.tau_minus_stdp)
        params['Wmax'] = float(self.Wmax)
        params['weight_per_lut_entry'] = float(self.weight_per_lut_entry)
        params['no_synapses'] = int(self.no_synapses)
        params['synapses_per_driver'] = int(self.synapses_per_driver)
        params['driver_readout_time'] = float(self.driver_readout_time)
        params['readout_cycle_duration'] = float(self.readout_cycle_duration)
        params['lookuptable_0'] = list(self.lookuptable_0)
        params['lookuptable_1'] = list(self.lookuptable_1)
        params['lookuptable_2'] = list(self.lookuptable_2)
        params['configbit_0'] = list(self.configbit_0)
        params['configbit_1'] = list(self.configbit_1)
        params['reset_pattern'] = list(self.reset_pattern)
        params['a_causal'] = float(self.a_causal)
        params['a_acausal'] = float(self.a_acausal)
        params['a_thresh_th'] = float(self.a_thresh_th)
        params['a_thresh_tl'] = float(self.a_thresh_tl)
        params['init_flag'] = bool(self.init_flag)
        params['synapse_id'] = int(self.synapse_id)
        params['next_readout_time'] = float(self.next_readout_time)
        params['synapse_model'] = 'stdp_facetshw_synapse_hom'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        """Reject common-property assignments in connect-time synapse specs."""
        if syn_spec is None:
            return
        disallowed = (
            'tau_plus',
            'tau_minus_stdp',
            'Wmax',
            'weight_per_lut_entry',
            'no_synapses',
            'synapses_per_driver',
            'driver_readout_time',
            'readout_cycle_duration',
            'lookuptable_0',
            'lookuptable_1',
            'lookuptable_2',
            'configbit_0',
            'configbit_1',
            'reset_pattern',
        )
        for key in disallowed:
            if key in syn_spec:
                raise ValueError(
                    f'{key} cannot be specified in connect-time synapse parameters '
                    'for stdp_facetshw_synapse_hom; set common properties on the '
                    'model itself (for example via CopyModel()/SetDefaults()).'
                )

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        tau_plus: ArrayLike | object = _UNSET,
        tau_minus_stdp: ArrayLike | object = _UNSET,
        Wmax: ArrayLike | object = _UNSET,
        weight_per_lut_entry: ArrayLike | object = _UNSET,
        no_synapses: ArrayLike | object = _UNSET,
        synapses_per_driver: ArrayLike | object = _UNSET,
        driver_readout_time: ArrayLike | object = _UNSET,
        readout_cycle_duration: ArrayLike | object = _UNSET,
        lookuptable_0: ArrayLike | object = _UNSET,
        lookuptable_1: ArrayLike | object = _UNSET,
        lookuptable_2: ArrayLike | object = _UNSET,
        configbit_0: ArrayLike | object = _UNSET,
        configbit_1: ArrayLike | object = _UNSET,
        reset_pattern: ArrayLike | object = _UNSET,
        a_causal: ArrayLike | object = _UNSET,
        a_acausal: ArrayLike | object = _UNSET,
        a_thresh_th: ArrayLike | object = _UNSET,
        a_thresh_tl: ArrayLike | object = _UNSET,
        init_flag: ArrayLike | object = _UNSET,
        synapse_id: ArrayLike | object = _UNSET,
        next_readout_time: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        """Set NEST-style public parameters and mutable state."""
        new_tau_plus = self.tau_plus if tau_plus is _UNSET else self._to_scalar_time_ms(tau_plus, name='tau_plus')
        new_tau_minus_stdp = (
            self.tau_minus_stdp
            if tau_minus_stdp is _UNSET
            else self._to_scalar_time_ms(tau_minus_stdp, name='tau_minus_stdp')
        )
        self._validate_positive(float(new_tau_plus), name='tau_plus')
        self._validate_positive(float(new_tau_minus_stdp), name='tau_minus_stdp')

        new_Wmax = self.Wmax if Wmax is _UNSET else self._to_scalar_float(Wmax, name='Wmax')
        new_weight_per_lut_entry = (
            self.weight_per_lut_entry
            if weight_per_lut_entry is _UNSET
            else self._to_scalar_float(weight_per_lut_entry, name='weight_per_lut_entry')
        )

        new_no_synapses = (
            self.no_synapses
            if no_synapses is _UNSET
            else self._to_int_scalar(no_synapses, name='no_synapses')
        )
        new_synapses_per_driver = (
            self.synapses_per_driver
            if synapses_per_driver is _UNSET
            else self._to_int_scalar(synapses_per_driver, name='synapses_per_driver')
        )
        self._validate_synapses_per_driver(int(new_synapses_per_driver))

        new_driver_readout_time = (
            self.driver_readout_time
            if driver_readout_time is _UNSET
            else self._to_scalar_float(driver_readout_time, name='driver_readout_time')
        )
        self._validate_positive(float(new_driver_readout_time), name='driver_readout_time')

        new_readout_cycle_duration = (
            self.readout_cycle_duration
            if readout_cycle_duration is _UNSET
            else self._to_scalar_float(readout_cycle_duration, name='readout_cycle_duration')
        )

        new_a_causal = self.a_causal if a_causal is _UNSET else self._to_scalar_float(a_causal, name='a_causal')
        new_a_acausal = self.a_acausal if a_acausal is _UNSET else self._to_scalar_float(a_acausal, name='a_acausal')
        new_a_thresh_th = (
            self.a_thresh_th
            if a_thresh_th is _UNSET
            else self._to_scalar_float(a_thresh_th, name='a_thresh_th')
        )
        new_a_thresh_tl = (
            self.a_thresh_tl
            if a_thresh_tl is _UNSET
            else self._to_scalar_float(a_thresh_tl, name='a_thresh_tl')
        )
        new_init_flag = self.init_flag if init_flag is _UNSET else self._to_bool_scalar(init_flag, name='init_flag')
        new_synapse_id = self.synapse_id if synapse_id is _UNSET else self._to_int_scalar(synapse_id, name='synapse_id')
        new_next_readout_time = (
            self.next_readout_time
            if next_readout_time is _UNSET
            else self._to_scalar_float(next_readout_time, name='next_readout_time')
        )

        super_kwargs = {}
        if weight is not _UNSET:
            super_kwargs['weight'] = self._to_scalar_float(weight, name='weight')
        if delay is not _UNSET:
            super_kwargs['delay'] = delay
        if receptor_type is not _UNSET:
            super_kwargs['receptor_type'] = receptor_type
        if post is not _UNSET:
            super_kwargs['post'] = post
        if super_kwargs:
            super().set(**super_kwargs)

        self.tau_plus = float(new_tau_plus)
        self.tau_minus_stdp = float(new_tau_minus_stdp)

        self.Wmax = float(new_Wmax)
        if Wmax is not _UNSET:
            self.weight_per_lut_entry = float(self.Wmax / (len(self.lookuptable_0) - 1))
        if weight_per_lut_entry is not _UNSET:
            self.weight_per_lut_entry = float(new_weight_per_lut_entry)

        if readout_cycle_duration is not _UNSET:
            self.readout_cycle_duration = float(new_readout_cycle_duration)
        if no_synapses is not _UNSET:
            self.no_synapses = int(new_no_synapses)
            self._calc_readout_cycle_duration()
        if synapses_per_driver is not _UNSET:
            self.synapses_per_driver = int(new_synapses_per_driver)
            self._calc_readout_cycle_duration()
        if driver_readout_time is not _UNSET:
            self.driver_readout_time = float(new_driver_readout_time)
            self._calc_readout_cycle_duration()

        if lookuptable_0 is not _UNSET:
            lut0 = self._to_int_vector(
                lookuptable_0,
                name='lookuptable_0',
                exact_size=16,
                min_value=_LUT_ENTRY_MIN,
                max_value=_LUT_ENTRY_MAX,
            )
            if len(lut0) != len(self.lookuptable_1):
                raise ValueError('Look-up table has not 2^4 entries.')
            self.lookuptable_0 = lut0
        if lookuptable_1 is not _UNSET:
            lut1 = self._to_int_vector(
                lookuptable_1,
                name='lookuptable_1',
                exact_size=16,
                min_value=_LUT_ENTRY_MIN,
                max_value=_LUT_ENTRY_MAX,
            )
            if len(lut1) != len(self.lookuptable_0):
                raise ValueError('Look-up table has not 2^4 entries.')
            self.lookuptable_1 = lut1
        if lookuptable_2 is not _UNSET:
            lut2 = self._to_int_vector(
                lookuptable_2,
                name='lookuptable_2',
                exact_size=16,
                min_value=_LUT_ENTRY_MIN,
                max_value=_LUT_ENTRY_MAX,
            )
            if len(lut2) != len(self.lookuptable_0):
                raise ValueError('Look-up table has not 2^4 entries.')
            self.lookuptable_2 = lut2

        if configbit_0 is not _UNSET:
            self.configbit_0 = self._to_int_vector(configbit_0, name='configbit_0', exact_size=4)
        if configbit_1 is not _UNSET:
            self.configbit_1 = self._to_int_vector(configbit_1, name='configbit_1', exact_size=4)
        if reset_pattern is not _UNSET:
            self.reset_pattern = self._to_int_vector(reset_pattern, name='reset_pattern', exact_size=6)

        self.a_causal = float(new_a_causal)
        self.a_acausal = float(new_a_acausal)
        self.a_thresh_th = float(new_a_thresh_th)
        self.a_thresh_tl = float(new_a_thresh_tl)
        self.init_flag = bool(new_init_flag)
        self.synapse_id = int(new_synapse_id)
        self.next_readout_time = float(new_next_readout_time)

        self._a_causal0 = float(self.a_causal)
        self._a_acausal0 = float(self.a_acausal)
        self._a_thresh_th0 = float(self.a_thresh_th)
        self._a_thresh_tl0 = float(self.a_thresh_tl)
        self._init_flag0 = bool(self.init_flag)
        self._synapse_id0 = int(self.synapse_id)
        self._next_readout_time0 = float(self.next_readout_time)
        self._no_synapses0 = int(self.no_synapses)
        self._readout_cycle_duration0 = float(self.readout_cycle_duration)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        """Schedule one outgoing event with NEST ``stdp_facetshw_synapse_hom`` dynamics."""
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)
        t_spike = self._current_time_ms() + dt_ms

        if not self.init_flag:
            self.synapse_id = int(self.no_synapses)
            self.no_synapses += 1
            self._calc_readout_cycle_duration()
            self.next_readout_time = int(self.synapse_id / self.synapses_per_driver) * self.driver_readout_time
            self.init_flag = True

        if t_spike > self.next_readout_time:
            self.discrete_weight = self._weight_to_entry(float(self.weight), float(self.weight_per_lut_entry))

            eval_0 = self._eval_function(
                float(self.a_causal),
                float(self.a_acausal),
                float(self.a_thresh_th),
                float(self.a_thresh_tl),
                self.configbit_0,
            )
            eval_1 = self._eval_function(
                float(self.a_causal),
                float(self.a_acausal),
                float(self.a_thresh_th),
                float(self.a_thresh_tl),
                self.configbit_1,
            )

            if eval_0 and not eval_1:
                self.discrete_weight = self._lookup(self.discrete_weight, self.lookuptable_0)
                if self.reset_pattern[0]:
                    self.a_causal = 0.0
                if self.reset_pattern[1]:
                    self.a_acausal = 0.0
            elif (not eval_0) and eval_1:
                self.discrete_weight = self._lookup(self.discrete_weight, self.lookuptable_1)
                if self.reset_pattern[2]:
                    self.a_causal = 0.0
                if self.reset_pattern[3]:
                    self.a_acausal = 0.0
            elif eval_0 and eval_1:
                self.discrete_weight = self._lookup(self.discrete_weight, self.lookuptable_2)
                if self.reset_pattern[4]:
                    self.a_causal = 0.0
                if self.reset_pattern[5]:
                    self.a_acausal = 0.0

            if self.readout_cycle_duration <= 0.0:
                raise ValueError('readout_cycle_duration must be > 0 during active readout scheduling.')
            while t_spike > self.next_readout_time:
                self.next_readout_time += self.readout_cycle_duration

            self.weight = float(self._entry_to_weight(self.discrete_weight, self.weight_per_lut_entry))

        dendritic_delay = float(self.delay)
        hist = self._get_post_history_times(self.t_lastspike - dendritic_delay, t_spike - dendritic_delay)
        if hist:
            minus_dt_causal = self.t_lastspike - (hist[0] + dendritic_delay)
            assert minus_dt_causal < (-1.0 * _STDP_EPS)
            self.a_causal += math.exp(minus_dt_causal / self.tau_plus)

            minus_dt_acausal = (hist[-1] + dendritic_delay) - t_spike
            self.a_acausal += math.exp(minus_dt_acausal / self.tau_minus_stdp)

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True

    def update(
        self,
        pre_spike: ArrayLike = 0.0,
        *,
        post_spike: ArrayLike = 0.0,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> int:
        """Deliver due events, update post history, then process presynaptic spikes."""
        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        post_count = self._to_non_negative_int_count(post_spike, name='post_spike')
        if post_count > 0:
            t_post = self._current_time_ms() + dt_ms
            for _ in range(post_count):
                self._record_post_spike_at(float(t_post))

        total_pre = self.sum_current_inputs(pre_spike)
        total_pre = self.sum_delta_inputs(total_pre)
        if self._is_nonzero(total_pre):
            self.send(total_pre, post=post, receptor_type=receptor_type)

        return delivered
