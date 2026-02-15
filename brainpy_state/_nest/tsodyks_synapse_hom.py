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

from .static_synapse_hom_w import static_synapse_hom_w

__all__ = [
    'tsodyks_synapse_hom',
]


_UNSET = object()


class tsodyks_synapse_hom(static_synapse_hom_w):
    r"""NEST-compatible ``tsodyks_synapse_hom`` connection model.

    Short description
    -----------------

    Synapse type with short-term plasticity using homogeneous parameters.

    Description
    -----------

    ``tsodyks_synapse_hom`` mirrors NEST ``models/tsodyks_synapse_hom.h``.
    It implements the short-term plasticity model of Tsodyks, Uziel and
    Markram (2000), with dynamic per-connection state:

    - ``x``: resources in recovered state,
    - ``y``: resources in active state,
    - ``u``: utilization (release probability),
    - ``z = 1 - x - y``: resources in inactive state.

    For a spike at time :math:`t_s` and :math:`h = t_s - t_{\mathrm{last}}`,
    NEST updates in this exact order:

    1. Propagation from :math:`t_{\mathrm{last}}` to :math:`t_s`:

       .. math::
          P_{uu} &= \begin{cases}
          0, & \tau_{fac}=0 \\
          e^{-h/\tau_{fac}}, & \tau_{fac}>0
          \end{cases} \\
          P_{yy} &= e^{-h/\tau_{psc}} \\
          P_{zz} &= e^{-h/\tau_{rec}} \\
          P_{xy} &= \frac{(P_{zz}-1)\tau_{rec} - (P_{yy}-1)\tau_{psc}}
                         {\tau_{psc}-\tau_{rec}} \\
          P_{xz} &= 1 - P_{zz}

       Then:

       .. math::
          u \leftarrow u \cdot P_{uu}

          x \leftarrow x + P_{xy} y + P_{xz} z

          y \leftarrow y \cdot P_{yy}

    2. Spike-triggered jump in ``u``:

       .. math::
          u \leftarrow u + U(1-u)

    3. Released amount:

       .. math::
          \Delta y = u \cdot x

    4. Spike-triggered jumps in ``x`` and ``y``:

       .. math::
          x \leftarrow x - \Delta y

          y \leftarrow y + \Delta y

    5. Event amplitude:

       .. math::
          w_{\mathrm{eff}} = \Delta y \cdot w

    This implementation preserves that ordering in :meth:`send`. Delay
    scheduling and delivery follow :class:`static_synapse_hom_w`.

    Homogeneous-property semantics
    ------------------------------

    In NEST, ``weight``, ``U``, ``tau_psc``, ``tau_fac`` and ``tau_rec`` are
    common synapse-model properties (``TsodyksHomCommonProperties``), while
    ``x``, ``y`` and ``u`` are per-connection state.

    This implementation mirrors that behavior by:

    - forbidding ``set_weight(...)`` (same as NEST),
    - rejecting these common properties in connect-time synapse specs via
      :meth:`check_synapse_params`,
    - allowing model-level updates via :meth:`set(...)`.

    Event timing semantics
    ----------------------

    As in NEST, this model uses spike stamps and ignores precise sub-step
    offsets when computing plasticity updates.

    Parameters
    ----------
    weight : ArrayLike, optional
        Common synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    tau_psc : ArrayLike, optional
        Common postsynaptic-current time constant in ms. Must be ``> 0``.
        Default: ``3.0 * u.ms``.
    tau_fac : ArrayLike, optional
        Common facilitation time constant in ms. Must be ``>= 0``.
        Default: ``0.0 * u.ms``.
    tau_rec : ArrayLike, optional
        Common recovery time constant in ms. Must be ``> 0``.
        Default: ``800.0 * u.ms``.
    U : ArrayLike, optional
        Common utilization increment parameter in ``[0, 1]``.
        Default: ``0.5``.
    x : ArrayLike, optional
        Initial recovered resources. Together with ``y`` must satisfy
        ``x + y <= 1``. Default: ``1.0``.
    y : ArrayLike, optional
        Initial active resources. Together with ``x`` must satisfy
        ``x + y <= 1``. Default: ``0.0``.
    u : ArrayLike, optional
        Initial utilization state. NEST stores this as mutable per-connection
        state and does not enforce bounds on assignment.
        Default: ``0.0``.
    post : object, optional
        Default receiver object.
    name : str, optional
        Object name.

    Notes
    -----
    - The model transmits spike-like events only.
    - ``init_state()`` resets queue state and restores ``x``, ``y``, ``u``.
    - ``t_lastspike`` starts at ``0.0`` as in NEST. If ``x != 1`` initially,
      the first-spike dynamics depend on this initial timestamp.

    References
    ----------
    .. [1] NEST source: ``models/tsodyks_synapse_hom.h`` and
           ``models/tsodyks_synapse_hom.cpp``.
    .. [2] Tsodyks M, Uziel A, Markram H (2000). Synchrony generation in
           recurrent networks with frequency-dependent synapses.
           Journal of Neuroscience, 20:RC50.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_psc: ArrayLike = 3.0 * u.ms,
        tau_fac: ArrayLike = 0.0 * u.ms,
        tau_rec: ArrayLike = 800.0 * u.ms,
        U: ArrayLike = 0.5,
        x: ArrayLike = 1.0,
        y: ArrayLike = 0.0,
        u: ArrayLike = 0.0,
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

        self.tau_psc = self._to_scalar_time_ms(tau_psc, name='tau_psc')
        self.tau_fac = self._to_scalar_time_ms(tau_fac, name='tau_fac')
        self.tau_rec = self._to_scalar_time_ms(tau_rec, name='tau_rec')
        self.U = self._to_scalar_unit_interval_no_quotes(U, name='U')

        x0 = self._to_scalar_float(x, name='x')
        y0 = self._to_scalar_float(y, name='y')
        u0 = self._to_scalar_float(u, name='u')

        self._validate_tau_psc(self.tau_psc)
        self._validate_tau_fac(self.tau_fac)
        self._validate_tau_rec(self.tau_rec)
        self._validate_xy_sum(x0, y0)

        self._x0 = float(x0)
        self._y0 = float(y0)
        self._u0 = float(u0)

        self.x = float(self._x0)
        self.y = float(self._y0)
        self.u = float(self._u0)
        self.t_lastspike = 0.0

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_unit_interval_no_quotes(value: ArrayLike, *, name: str) -> float:
        v = tsodyks_synapse_hom._to_scalar_float(value, name=name)
        if v < 0.0 or v > 1.0:
            raise ValueError(f'{name} must be in [0,1].')
        return float(v)

    @staticmethod
    def _validate_tau_psc(value: float):
        if value <= 0.0:
            raise ValueError('tau_psc must be > 0.')

    @staticmethod
    def _validate_tau_fac(value: float):
        if value < 0.0:
            raise ValueError('tau_fac must be >= 0.')

    @staticmethod
    def _validate_tau_rec(value: float):
        if value <= 0.0:
            raise ValueError('tau_rec must be > 0.')

    @staticmethod
    def _validate_xy_sum(x: float, y: float):
        if x + y > 1.0:
            raise ValueError('x + y must be <= 1.0.')

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        super().init_state()
        self.x = float(self._x0)
        self.y = float(self._y0)
        self.u = float(self._u0)
        self.t_lastspike = 0.0

    def get(self) -> dict:
        """Return current public parameters and mutable state."""
        params = super().get()
        params['tau_psc'] = float(self.tau_psc)
        params['tau_fac'] = float(self.tau_fac)
        params['tau_rec'] = float(self.tau_rec)
        params['U'] = float(self.U)
        params['x'] = float(self.x)
        params['y'] = float(self.y)
        params['u'] = float(self.u)
        params['synapse_model'] = 'tsodyks_synapse_hom'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        """Reject common-property assignments in connect-time synapse specs."""
        if syn_spec is None:
            return
        disallowed = ('weight', 'U', 'tau_psc', 'tau_rec', 'tau_fac')
        for key in disallowed:
            if key in syn_spec:
                raise ValueError(
                    f'{key} cannot be specified in connect-time synapse parameters '
                    'for tsodyks_synapse_hom; set common properties on the model '
                    'itself (for example via CopyModel()/SetDefaults()).'
                )

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        tau_psc: ArrayLike | object = _UNSET,
        tau_fac: ArrayLike | object = _UNSET,
        tau_rec: ArrayLike | object = _UNSET,
        U: ArrayLike | object = _UNSET,
        x: ArrayLike | object = _UNSET,
        y: ArrayLike | object = _UNSET,
        u: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        """Set public parameters following NEST-style validation semantics."""
        new_tau_psc = (
            self.tau_psc
            if tau_psc is _UNSET
            else self._to_scalar_time_ms(tau_psc, name='tau_psc')
        )
        new_tau_fac = (
            self.tau_fac
            if tau_fac is _UNSET
            else self._to_scalar_time_ms(tau_fac, name='tau_fac')
        )
        new_tau_rec = (
            self.tau_rec
            if tau_rec is _UNSET
            else self._to_scalar_time_ms(tau_rec, name='tau_rec')
        )
        new_U = (
            self.U
            if U is _UNSET
            else self._to_scalar_unit_interval_no_quotes(U, name='U')
        )
        new_x = self.x if x is _UNSET else self._to_scalar_float(x, name='x')
        new_y = self.y if y is _UNSET else self._to_scalar_float(y, name='y')
        new_u = self.u if u is _UNSET else self._to_scalar_float(u, name='u')

        self._validate_tau_psc(float(new_tau_psc))
        self._validate_tau_fac(float(new_tau_fac))
        self._validate_tau_rec(float(new_tau_rec))
        self._validate_xy_sum(float(new_x), float(new_y))

        super_kwargs = {}
        if weight is not _UNSET:
            super_kwargs['weight'] = weight
        if delay is not _UNSET:
            super_kwargs['delay'] = delay
        if receptor_type is not _UNSET:
            super_kwargs['receptor_type'] = receptor_type
        if post is not _UNSET:
            super_kwargs['post'] = post
        if super_kwargs:
            super().set(**super_kwargs)

        self.tau_psc = float(new_tau_psc)
        self.tau_fac = float(new_tau_fac)
        self.tau_rec = float(new_tau_rec)
        self.U = float(new_U)

        self.x = float(new_x)
        self.y = float(new_y)
        self.u = float(new_u)

        self._x0 = float(self.x)
        self._y0 = float(self.y)
        self._u0 = float(self.u)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        """Schedule one outgoing event with NEST ``tsodyks_synapse_hom`` dynamics."""
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        t_spike = self._current_time_ms() + dt_ms
        h = float(t_spike - self.t_lastspike)

        puu = 0.0 if self.tau_fac == 0.0 else math.exp(-h / self.tau_fac)
        pyy = math.exp(-h / self.tau_psc)
        pzz = math.exp(-h / self.tau_rec)
        pxy = ((pzz - 1.0) * self.tau_rec - (pyy - 1.0) * self.tau_psc) / (
            self.tau_psc - self.tau_rec
        )
        pxz = 1.0 - pzz

        z = 1.0 - self.x - self.y

        # Keep ordering identical to NEST models/tsodyks_synapse_hom.h::send.
        self.u *= puu
        self.x += pxy * self.y + pxz * z
        self.y *= pyy

        self.u += self.U * (1.0 - self.u)

        delta_y_tsp = self.u * self.x
        self.x -= delta_y_tsp
        self.y += delta_y_tsp

        # NEST code sets receiver before weight and delay assignment.
        receiver = self._resolve_receiver(post)
        weighted_payload = multiplicity * (delta_y_tsp * self.weight)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True

    def update(
        self,
        pre_spike: ArrayLike = 0.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> int:
        """Deliver due events, then schedule current-step presynaptic input."""
        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        total = self.sum_current_inputs(pre_spike)
        total = self.sum_delta_inputs(total)
        if self._is_nonzero(total):
            self.send(total, post=post, receptor_type=receptor_type)

        return delivered
