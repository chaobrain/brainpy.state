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
    'stdp_dopamine_synapse',
]


_STDP_EPS = 1.0e-6


class stdp_dopamine_synapse(static_synapse):
    r"""NEST-compatible ``stdp_dopamine_synapse`` connection model.

    Short description
    -----------------

    Synapse type for dopamine-modulated spike-timing dependent plasticity.

    Description
    -----------

    ``stdp_dopamine_synapse`` mirrors NEST
    ``models/stdp_dopamine_synapse.h`` and implements dopamine-modulated STDP
    with per-connection state:

    - ``weight``: synaptic efficacy,
    - ``Kplus``: presynaptic facilitation trace,
    - ``c``: eligibility trace,
    - ``n``: dopamine trace,
    - ``t_last_update``: timestamp of last propagated state update,
    - ``t_lastspike``: timestamp of previous presynaptic spike.

    In NEST, postsynaptic depression trace ``Kminus`` is read from the
    postsynaptic archiving neuron. For standalone compatibility, this backend
    reproduces it with an internal post-spike history buffer parameterized by
    ``tau_minus`` (not a synapse parameter in NEST).

    Dopaminergic spikes are provided by a NEST ``volume_transmitter``.
    In this backend, dopamine spikes can be fed through ``update(...,
    dopa_spike=...)`` or :meth:`record_dopa_spike`, while still requiring a
    non-``None`` ``volume_transmitter`` handle to preserve NEST connection
    semantics.

    Update order (NEST source equivalent)
    -------------------------------------

    For a presynaptic spike at stamp :math:`t_{pre}` and dendritic delay
    :math:`d`, NEST ``stdp_dopamine_synapse::send`` performs:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last\_update}}-d,\; t_{pre}-d]`.
    2. For each postsynaptic spike :math:`t_{post}` in that range:
       1. Propagate dopamine/eligibility/weight in
          :math:`(t_0,\; t_{post}+d]`.
       2. Facilitate eligibility if :math:`t_{pre} - t_{post} > \epsilon`:
          :math:`c \leftarrow c + A_+ K_+ \exp((t_{\mathrm{last\_update}}-(t_{post}+d))/\tau_+)`.
    3. Propagate dopamine/eligibility/weight up to :math:`t_{pre}`.
    4. Depress eligibility:
       :math:`c \leftarrow c - A_- K^-(t_{pre}-d)`.
    5. Send event using updated ``weight``.
    6. Update presynaptic trace:
       :math:`K_+ \leftarrow K_+ \exp((t_{\mathrm{last\_update}}-t_{pre})/\tau_+) + 1`.
    7. Set ``t_last_update = t_lastspike = t_pre``.

    This implementation preserves the same ordering.

    Weight integration
    ------------------

    Between event times, NEST integrates:

    .. math::
       \dot w = c(t)\,(n(t)-b),
       \quad
       \dot c = -c/\tau_c,
       \quad
       \dot n = -n/\tau_n

    and updates weight in closed form using ``expm1`` over each interval:

    .. math::
       w \leftarrow w - c_0 \Big(
       \frac{n_0}{\tau_s}\,\mathrm{expm1}(\tau_s \Delta^-)
       - b\,\tau_c\,\mathrm{expm1}(\Delta^-/\tau_c)
       \Big),
       \qquad
       \tau_s = \frac{\tau_c+\tau_n}{\tau_c\tau_n}

    with :math:`\Delta^- = t_0 - t_1 \le 0`, and clipping to
    :math:`[W_{\min}, W_{\max}]`.

    Event timing semantics
    ----------------------

    As in NEST, this model uses on-grid spike stamps and ignores precise
    sub-step offsets for plasticity updates.

    Parameters
    ----------
    weight : ArrayLike, optional
        Initial synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    volume_transmitter : object, optional
        Placeholder handle for NEST compatibility. Must be non-``None`` before
        running updates, matching NEST requirement that a volume transmitter is
        assigned before simulation. Default: ``None``.
    A_plus : ArrayLike, optional
        Facilitation coefficient. Default: ``1.0``.
    A_minus : ArrayLike, optional
        Depression coefficient. Default: ``1.5``.
    tau_plus : ArrayLike, optional
        Presynaptic trace time constant in ms. Default: ``20.0 * u.ms``.
    tau_minus : ArrayLike, optional
        Postsynaptic trace time constant in ms.
        In NEST this belongs to postsynaptic archiving neurons; here it is
        stored on the synapse for standalone compatibility.
        Default: ``20.0 * u.ms``.
    tau_c : ArrayLike, optional
        Eligibility trace time constant in ms. Default: ``1000.0 * u.ms``.
    tau_n : ArrayLike, optional
        Dopamine trace time constant in ms. Default: ``200.0 * u.ms``.
    b : ArrayLike, optional
        Dopamine baseline concentration. Default: ``0.0``.
    Wmin : ArrayLike, optional
        Minimum weight bound. Default: ``0.0``.
    Wmax : ArrayLike, optional
        Maximum weight bound. Default: ``200.0``.
    Kplus : ArrayLike, optional
        Initial presynaptic trace value. Must be non-negative.
        Default: ``0.0``.
    c : ArrayLike, optional
        Initial eligibility trace value. Default: ``0.0``.
    n : ArrayLike, optional
        Initial dopamine trace value. Default: ``0.0``.
    post : object, optional
        Default receiver object.
    name : str, optional
        Object name.

    Notes
    -----
    - The model transmits spike-like events only.
    - ``update(pre_spike=..., post_spike=..., dopa_spike=...)`` supports
      explicit per-step dopamine multiplicity for standalone simulations.
    - This backend performs one ``trigger_update_weight`` propagation per
      simulation step in :meth:`update`, corresponding to NEST default
      ``volume_transmitter`` delivery interval.

    References
    ----------
    .. [1] NEST source: ``models/stdp_dopamine_synapse.h``,
           ``models/stdp_dopamine_synapse.cpp``,
           ``models/volume_transmitter.h`` and
           ``models/volume_transmitter.cpp``.
    .. [2] Potjans W, Morrison A, Diesmann M (2010). Enabling functional
           neural circuit simulations with distributed computing of
           neuromodulated plasticity. Frontiers in Computational Neuroscience,
           4:141. https://doi.org/10.3389/fncom.2010.00141
    .. [3] Izhikevich EM (2007). Solving the distal reward problem through
           linkage of STDP and dopamine signaling. Cerebral Cortex,
           17(10):2443-2452. https://doi.org/10.1093/cercor/bhl152
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        volume_transmitter=None,
        A_plus: ArrayLike = 1.0,
        A_minus: ArrayLike = 1.5,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        tau_c: ArrayLike = 1000.0 * u.ms,
        tau_n: ArrayLike = 200.0 * u.ms,
        b: ArrayLike = 0.0,
        Wmin: ArrayLike = 0.0,
        Wmax: ArrayLike = 200.0,
        Kplus: ArrayLike = 0.0,
        c: ArrayLike = 0.0,
        n: ArrayLike = 0.0,
        post=None,
        name: str | None = None,
    ):
        super().__init__(
            weight=weight,
            delay=delay,
            receptor_type=receptor_type,
            post=post,
            event_type='spike',
            name=name,
        )

        self.volume_transmitter = volume_transmitter
        self.A_plus = self._to_scalar_float(A_plus, name='A_plus')
        self.A_minus = self._to_scalar_float(A_minus, name='A_minus')
        self.tau_plus = self._to_scalar_time_ms(tau_plus, name='tau_plus')
        self.tau_minus = self._to_scalar_time_ms(tau_minus, name='tau_minus')
        self.tau_c = self._to_scalar_time_ms(tau_c, name='tau_c')
        self.tau_n = self._to_scalar_time_ms(tau_n, name='tau_n')
        self.b = self._to_scalar_float(b, name='b')
        self.Wmin = self._to_scalar_float(Wmin, name='Wmin')
        self.Wmax = self._to_scalar_float(Wmax, name='Wmax')
        self.Kplus = self._to_scalar_float(Kplus, name='Kplus')
        self.c = self._to_scalar_float(c, name='c')
        self.n = self._to_scalar_float(n, name='n')

        self._validate_tau_positive(self.tau_plus, name='tau_plus')
        self._validate_tau_positive(self.tau_minus, name='tau_minus')
        self._validate_tau_positive(self.tau_c, name='tau_c')
        self._validate_tau_positive(self.tau_n, name='tau_n')
        self._validate_non_negative(self.Kplus, name='Kplus')

        self._Kplus0 = float(self.Kplus)
        self._c0 = float(self.c)
        self._n0 = float(self.n)
        self._t_last_update0 = 0.0
        self._t_lastspike0 = 0.0

        self.t_last_update = float(self._t_last_update0)
        self.t_lastspike = float(self._t_lastspike0)
        self.dopa_spikes_idx = 0

        self._post_kminus = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t: list[float] = []
        self._post_hist_kminus: list[float] = []
        self._dopa_spikes: list[tuple[float, float]] = [(0.0, 0.0)]

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

    @staticmethod
    def _validate_tau_positive(value: float, *, name: str):
        if value <= 0.0:
            raise ValueError(f'{name} must be > 0.')

    @staticmethod
    def _validate_non_negative(value: float, *, name: str):
        if value < 0.0:
            raise ValueError(f'{name} must be non-negative.')

    @staticmethod
    def _to_non_negative_int_count(value: ArrayLike, *, name: str) -> int:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr.reshape(()))
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        if v < 0.0:
            raise ValueError(f'{name} must be non-negative.')
        rounded = int(round(v))
        if not math.isclose(v, float(rounded), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be an integer spike count.')
        return rounded

    @staticmethod
    def _to_non_negative_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr.reshape(()))
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        if v < 0.0:
            raise ValueError(f'{name} must be non-negative.')
        return v

    def _ensure_volume_transmitter(self):
        if self.volume_transmitter is None:
            raise ValueError('No volume transmitter has been assigned to the dopamine synapse.')

    def _update_dopamine(self):
        minus_dt = self._dopa_spikes[self.dopa_spikes_idx][0] - self._dopa_spikes[self.dopa_spikes_idx + 1][0]
        self.dopa_spikes_idx += 1
        self.n = (
            self.n * math.exp(minus_dt / self.tau_n)
            + self._dopa_spikes[self.dopa_spikes_idx][1] / self.tau_n
        )

    def _update_weight(self, c0: float, n0: float, minus_dt: float):
        taus = (self.tau_c + self.tau_n) / (self.tau_c * self.tau_n)
        self.weight = (
            float(self.weight)
            - c0 * (n0 / taus * math.expm1(taus * minus_dt) - self.b * self.tau_c * math.expm1(minus_dt / self.tau_c))
        )
        if self.weight < self.Wmin:
            self.weight = float(self.Wmin)
        if self.weight > self.Wmax:
            self.weight = float(self.Wmax)

    def _process_dopa_spikes(self, t0: float, t1: float):
        # Process dopamine spikes in (t0, t1], reproducing NEST
        # stdp_dopamine_synapse::process_dopa_spikes_.
        if t1 < (t0 - _STDP_EPS):
            raise ValueError('process_dopa_spikes requires t1 >= t0.')
        if not self._dopa_spikes:
            self._dopa_spikes = [(t0, 0.0)]
            self.dopa_spikes_idx = 0

        if (
            len(self._dopa_spikes) > self.dopa_spikes_idx + 1
            and (t1 - self._dopa_spikes[self.dopa_spikes_idx + 1][0] > -1.0 * _STDP_EPS)
        ):
            n0 = self.n * math.exp((self._dopa_spikes[self.dopa_spikes_idx][0] - t0) / self.tau_n)
            self._update_weight(self.c, n0, t0 - self._dopa_spikes[self.dopa_spikes_idx + 1][0])
            self._update_dopamine()

            while (
                len(self._dopa_spikes) > self.dopa_spikes_idx + 1
                and (t1 - self._dopa_spikes[self.dopa_spikes_idx + 1][0] > -1.0 * _STDP_EPS)
            ):
                cd = self.c * math.exp((t0 - self._dopa_spikes[self.dopa_spikes_idx][0]) / self.tau_c)
                self._update_weight(
                    cd,
                    self.n,
                    self._dopa_spikes[self.dopa_spikes_idx][0] - self._dopa_spikes[self.dopa_spikes_idx + 1][0],
                )
                self._update_dopamine()

            cd = self.c * math.exp((t0 - self._dopa_spikes[self.dopa_spikes_idx][0]) / self.tau_c)
            self._update_weight(cd, self.n, self._dopa_spikes[self.dopa_spikes_idx][0] - t1)
        else:
            n0 = self.n * math.exp((self._dopa_spikes[self.dopa_spikes_idx][0] - t0) / self.tau_n)
            self._update_weight(self.c, n0, t0 - t1)

        self.c = self.c * math.exp((t0 - t1) / self.tau_c)

    def _facilitate(self, kplus: float):
        self.c += self.A_plus * kplus

    def _depress(self, kminus: float):
        self.c -= self.A_minus * kminus

    def clear_post_history(self):
        """Clear internal postsynaptic STDP history state."""
        self._post_kminus = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t = []
        self._post_hist_kminus = []

    def clear_dopamine_history(self):
        """Reset internal dopamine spike history with pseudo spike at current update time."""
        anchor_t = float(self.t_last_update)
        self._dopa_spikes = [(anchor_t, 0.0)]
        self.dopa_spikes_idx = 0

    def _record_post_spike_at(self, t_spike_ms: float):
        self._post_kminus = (
            self._post_kminus * math.exp((self._last_post_spike - t_spike_ms) / self.tau_minus) + 1.0
        )
        self._last_post_spike = float(t_spike_ms)
        self._post_hist_t.append(float(t_spike_ms))
        self._post_hist_kminus.append(float(self._post_kminus))

    def record_post_spike(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        t_spike_ms: ArrayLike | None = None,
    ) -> int:
        """Record postsynaptic spikes into internal STDP history."""
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

    def record_dopa_spike(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        t_spike_ms: ArrayLike | None = None,
    ) -> float:
        """Record dopamine spikes into internal volume-transmitter history."""
        mult = self._to_non_negative_float(multiplicity, name='dopa_spike')
        if mult == 0.0:
            return 0.0

        if t_spike_ms is None:
            dt_ms = self._refresh_delay_if_needed()
            t_value = self._current_time_ms() + dt_ms
        else:
            t_value = self._to_scalar_float(t_spike_ms, name='t_spike_ms')

        if not self._dopa_spikes:
            self._dopa_spikes = [(float(t_value), float(mult))]
            self.dopa_spikes_idx = 0
            return mult

        t_last = self._dopa_spikes[-1][0]
        if t_value < (t_last - _STDP_EPS):
            raise ValueError('Dopamine spikes must be recorded in non-decreasing time order.')

        if abs(t_value - t_last) <= _STDP_EPS:
            t_prev, mult_prev = self._dopa_spikes[-1]
            self._dopa_spikes[-1] = (float(t_prev), float(mult_prev + mult))
        else:
            self._dopa_spikes.append((float(t_value), float(mult)))
        return mult

    def _get_post_history_times(self, t1_ms: float, t2_ms: float) -> list[float]:
        t1_lim = float(t1_ms + _STDP_EPS)
        t2_lim = float(t2_ms + _STDP_EPS)
        selected = []
        for t_post in self._post_hist_t:
            if t_post >= t1_lim and t_post < t2_lim:
                selected.append(float(t_post))
        return selected

    def _get_K_value(self, t_ms: float) -> float:
        # Return trace strictly before t, matching ArchivingNode::get_K_value.
        for idx in range(len(self._post_hist_t) - 1, -1, -1):
            t_post = self._post_hist_t[idx]
            if (t_ms - t_post) > _STDP_EPS:
                return self._post_hist_kminus[idx] * math.exp((t_post - t_ms) / self.tau_minus)
        return 0.0

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        super().init_state()
        self.Kplus = float(self._Kplus0)
        self.c = float(self._c0)
        self.n = float(self._n0)
        self.t_last_update = float(self._t_last_update0)
        self.t_lastspike = float(self._t_lastspike0)
        self.clear_post_history()
        self._dopa_spikes = [(float(self.t_last_update), 0.0)]
        self.dopa_spikes_idx = 0

    def get(self) -> dict:
        """Return current public parameters and mutable state."""
        params = super().get()
        params['volume_transmitter'] = self.volume_transmitter
        params['A_plus'] = float(self.A_plus)
        params['A_minus'] = float(self.A_minus)
        params['tau_plus'] = float(self.tau_plus)
        params['tau_minus'] = float(self.tau_minus)
        params['tau_c'] = float(self.tau_c)
        params['tau_n'] = float(self.tau_n)
        params['b'] = float(self.b)
        params['Wmin'] = float(self.Wmin)
        params['Wmax'] = float(self.Wmax)
        params['Kplus'] = float(self.Kplus)
        params['c'] = float(self.c)
        params['n'] = float(self.n)
        params['synapse_model'] = 'stdp_dopamine_synapse'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        """Reject common-property assignments in connect-time synapse specs."""
        if syn_spec is None:
            return
        disallowed = ('vt', 'volume_transmitter', 'A_minus', 'A_plus', 'Wmax', 'Wmin', 'b', 'tau_c', 'tau_n', 'tau_plus')
        for key in disallowed:
            if key in syn_spec:
                raise ValueError(
                    f'{key} cannot be specified in connect-time synapse parameters '
                    'for stdp_dopamine_synapse; set common properties on the model '
                    'itself (for example via CopyModel()/SetDefaults()).'
                )

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        volume_transmitter: object = _UNSET,
        A_plus: ArrayLike | object = _UNSET,
        A_minus: ArrayLike | object = _UNSET,
        tau_plus: ArrayLike | object = _UNSET,
        tau_minus: ArrayLike | object = _UNSET,
        tau_c: ArrayLike | object = _UNSET,
        tau_n: ArrayLike | object = _UNSET,
        b: ArrayLike | object = _UNSET,
        Wmin: ArrayLike | object = _UNSET,
        Wmax: ArrayLike | object = _UNSET,
        Kplus: ArrayLike | object = _UNSET,
        c: ArrayLike | object = _UNSET,
        n: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        """Set NEST-style public parameters and mutable state."""
        new_A_plus = self.A_plus if A_plus is _UNSET else self._to_scalar_float(A_plus, name='A_plus')
        new_A_minus = self.A_minus if A_minus is _UNSET else self._to_scalar_float(A_minus, name='A_minus')
        new_tau_plus = self.tau_plus if tau_plus is _UNSET else self._to_scalar_time_ms(tau_plus, name='tau_plus')
        new_tau_minus = self.tau_minus if tau_minus is _UNSET else self._to_scalar_time_ms(tau_minus, name='tau_minus')
        new_tau_c = self.tau_c if tau_c is _UNSET else self._to_scalar_time_ms(tau_c, name='tau_c')
        new_tau_n = self.tau_n if tau_n is _UNSET else self._to_scalar_time_ms(tau_n, name='tau_n')
        new_b = self.b if b is _UNSET else self._to_scalar_float(b, name='b')
        new_Wmin = self.Wmin if Wmin is _UNSET else self._to_scalar_float(Wmin, name='Wmin')
        new_Wmax = self.Wmax if Wmax is _UNSET else self._to_scalar_float(Wmax, name='Wmax')
        new_Kplus = self.Kplus if Kplus is _UNSET else self._to_scalar_float(Kplus, name='Kplus')
        new_c = self.c if c is _UNSET else self._to_scalar_float(c, name='c')
        new_n = self.n if n is _UNSET else self._to_scalar_float(n, name='n')

        self._validate_tau_positive(float(new_tau_plus), name='tau_plus')
        self._validate_tau_positive(float(new_tau_minus), name='tau_minus')
        self._validate_tau_positive(float(new_tau_c), name='tau_c')
        self._validate_tau_positive(float(new_tau_n), name='tau_n')
        self._validate_non_negative(float(new_Kplus), name='Kplus')

        super_kwargs = {}
        if weight is not _UNSET:
            super_kwargs['weight'] = self._normalize_scalar_weight(weight)
        if delay is not _UNSET:
            super_kwargs['delay'] = delay
        if receptor_type is not _UNSET:
            super_kwargs['receptor_type'] = receptor_type
        if post is not _UNSET:
            super_kwargs['post'] = post
        if super_kwargs:
            super().set(**super_kwargs)

        if volume_transmitter is not _UNSET:
            self.volume_transmitter = volume_transmitter

        self.A_plus = float(new_A_plus)
        self.A_minus = float(new_A_minus)
        self.tau_plus = float(new_tau_plus)
        self.tau_minus = float(new_tau_minus)
        self.tau_c = float(new_tau_c)
        self.tau_n = float(new_tau_n)
        self.b = float(new_b)
        self.Wmin = float(new_Wmin)
        self.Wmax = float(new_Wmax)
        self.Kplus = float(new_Kplus)
        self.c = float(new_c)
        self.n = float(new_n)

        self._Kplus0 = float(self.Kplus)
        self._c0 = float(self.c)
        self._n0 = float(self.n)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        """Schedule one outgoing event with NEST ``stdp_dopamine_synapse`` dynamics."""
        self._ensure_volume_transmitter()
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)
        t_spike = self._current_time_ms() + dt_ms
        dendritic_delay = float(self.delay)

        t0 = self.t_last_update
        for t_post in self._get_post_history_times(self.t_last_update - dendritic_delay, t_spike - dendritic_delay):
            self._process_dopa_spikes(t0, t_post + dendritic_delay)
            t0 = t_post + dendritic_delay
            minus_dt = self.t_last_update - t0
            if (t_spike - t_post) > _STDP_EPS:
                self._facilitate(self.Kplus * math.exp(minus_dt / self.tau_plus))

        self._process_dopa_spikes(t0, t_spike)
        self._depress(self._get_K_value(t_spike - dendritic_delay))

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.Kplus = float(self.Kplus * math.exp((self.t_last_update - t_spike) / self.tau_plus) + 1.0)
        self.t_last_update = float(t_spike)
        self.t_lastspike = float(t_spike)
        return True

    def trigger_update_weight(self, *, t_trig_ms: ArrayLike | None = None):
        """Propagate state to trigger time as NEST ``trigger_update_weight`` does."""
        self._ensure_volume_transmitter()
        if t_trig_ms is None:
            dt_ms = self._refresh_delay_if_needed()
            t_trig = self._current_time_ms() + dt_ms
        else:
            t_trig = self._to_scalar_float(t_trig_ms, name='t_trig_ms')

        if t_trig < (self.t_last_update - _STDP_EPS):
            raise ValueError('t_trig_ms must be greater than or equal to t_last_update.')

        dendritic_delay = float(self.delay)
        t0 = self.t_last_update
        for t_post in self._get_post_history_times(self.t_last_update - dendritic_delay, t_trig - dendritic_delay):
            self._process_dopa_spikes(t0, t_post + dendritic_delay)
            t0 = t_post + dendritic_delay
            minus_dt = self.t_last_update - t0
            self._facilitate(self.Kplus * math.exp(minus_dt / self.tau_plus))

        self._process_dopa_spikes(t0, t_trig)
        self.n = self.n * math.exp((self._dopa_spikes[self.dopa_spikes_idx][0] - t_trig) / self.tau_n)
        self.Kplus = self.Kplus * math.exp((self.t_last_update - t_trig) / self.tau_plus)
        self.t_last_update = float(t_trig)
        self._dopa_spikes = [(float(t_trig), 0.0)]
        self.dopa_spikes_idx = 0

    def update(
        self,
        pre_spike: ArrayLike = 0.0,
        *,
        post_spike: ArrayLike = 0.0,
        dopa_spike: ArrayLike = 0.0,
        post=None,
        receptor_type: ArrayLike | None = None,
        trigger_dopa_update: bool = True,
    ) -> int:
        """Deliver events, update traces, process pre spikes, then trigger update.

        Step order is:

        1. Deliver due delayed events.
        2. Record postsynaptic spikes at current on-grid spike stamp ``t + dt``.
        3. Record dopamine multiplicity at the same stamp.
        4. Aggregate presynaptic multiplicity and run :meth:`send`.
        5. Optionally call :meth:`trigger_update_weight` at ``t + dt``.
        """
        self._ensure_volume_transmitter()

        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        t_spike = self._current_time_ms() + dt_ms

        post_count = self._to_non_negative_int_count(post_spike, name='post_spike')
        for _ in range(post_count):
            self._record_post_spike_at(float(t_spike))

        dopa_mult = self._to_non_negative_float(dopa_spike, name='dopa_spike')
        if dopa_mult > 0.0:
            self.record_dopa_spike(dopa_mult, t_spike_ms=t_spike)

        total_pre = self.sum_current_inputs(pre_spike)
        total_pre = self.sum_delta_inputs(total_pre)
        if self._is_nonzero(total_pre):
            self.send(total_pre, post=post, receptor_type=receptor_type)

        if trigger_dopa_update:
            self.trigger_update_weight(t_trig_ms=t_spike)

        return delivered
