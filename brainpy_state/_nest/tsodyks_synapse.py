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

import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike

from .static_synapse import static_synapse

__all__ = [
    'tsodyks_synapse',
]


_UNSET = object()


class tsodyks_synapse(static_synapse):
    r"""NEST-compatible ``tsodyks_synapse`` connection model.

    Short description
    -----------------

    Synapse type with short-term depression and facilitation.

    Description
    -----------

    ``tsodyks_synapse`` mirrors NEST ``models/tsodyks_synapse.h`` and
    implements the short-term plasticity model from Tsodyks, Uziel and
    Markram (2000). The state variables are:

    - ``x``: resources in recovered state,
    - ``y``: resources in active state,
    - ``u``: utilization (release probability),
    - ``z = 1 - x - y``: resources in inactive state.

    For an incoming spike at time :math:`t_s`, with
    :math:`h = t_s - t_{\mathrm{last}}`, NEST updates in this exact order:

    1. Propagation from :math:`t_{\mathrm{last}}` to :math:`t_s`:

       .. math::
          u \leftarrow u \cdot P_{uu}

          x \leftarrow x + P_{xy} y - P_{zz} z

          y \leftarrow y \cdot P_{yy}

       where

       .. math::
          P_{uu} = \begin{cases}
            0, & \tau_{fac}=0 \\
            e^{-h/\tau_{fac}}, & \tau_{fac}>0
          \end{cases}

          P_{yy} = e^{-h/\tau_{psc}}

          P_{zz} = e^{-h/\tau_{rec}} - 1

          P_{xy} = \frac{P_{zz}\tau_{rec} - (P_{yy}-1)\tau_{psc}}
                         {\tau_{psc}-\tau_{rec}}

    2. Spike-triggered jump in ``u``:

       .. math::
          u \leftarrow u + U(1-u)

    3. Released amount at spike time:

       .. math::
          \Delta y = u \cdot x

    4. Spike-triggered jumps in ``x`` and ``y``:

       .. math::
          x \leftarrow x - \Delta y

          y \leftarrow y + \Delta y

    5. Event weight delivered to target:

       .. math::
          w_{\mathrm{eff}} = \Delta y \cdot w

    This implementation preserves that ordering exactly in :meth:`send`.
    Delay scheduling and receiver delivery follow :class:`static_synapse`.

    Event timing semantics
    ----------------------

    NEST evaluates this model on spike stamps and ignores precise sub-step
    spike offsets. This implementation follows the same behavior by using
    the on-grid spike stamp ``t + dt`` for each step where presynaptic
    multiplicity is non-zero.

    Parameters
    ----------
    weight : ArrayLike, optional
        Baseline synaptic weight ``w``. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    tau_psc : ArrayLike, optional
        Time constant of synaptic current in ms. Must be ``> 0``.
        Default: ``3.0 * u.ms``.
    tau_fac : ArrayLike, optional
        Facilitation time constant in ms. Must be ``>= 0``.
        Default: ``0.0 * u.ms``.
    tau_rec : ArrayLike, optional
        Recovery (depression) time constant in ms. Must be ``> 0``.
        Default: ``800.0 * u.ms``.
    U : ArrayLike, optional
        Utilization increment parameter in ``[0, 1]``.
        Default: ``0.5``.
    x : ArrayLike, optional
        Initial recovered resources. Together with ``y`` must satisfy
        ``x + y <= 1``. Default: ``1.0``.
    y : ArrayLike, optional
        Initial active resources. Together with ``x`` must satisfy
        ``x + y <= 1``. Default: ``0.0``.
    u : ArrayLike, optional
        Initial utilization value in ``[0, 1]``. Default: ``0.0``.
    post : object, optional
        Default receiver object.
    name : str, optional
        Object name.

    Notes
    -----
    - This model transmits spike-like events only.
    - The state variables ``x``, ``y`` and ``u`` are mutable connection
      states and are returned by :meth:`get`.
    - ``init_state()`` resets queue state and restores ``x``, ``y``, ``u``
      to the configured initial values.

    References
    ----------
    .. [1] NEST source: ``models/tsodyks_synapse.h`` and
           ``models/tsodyks_synapse.cpp``.
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
        self.U = self._to_scalar_unit_interval(U, name='U')

        x0 = self._to_scalar_float(x, name='x')
        y0 = self._to_scalar_float(y, name='y')
        u0 = self._to_scalar_unit_interval(u, name='u')

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
    def _to_scalar_unit_interval(value: ArrayLike, *, name: str) -> float:
        v = tsodyks_synapse._to_scalar_float(value, name=name)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"'{name}' must be in [0,1].")
        return float(v)

    @staticmethod
    def _validate_tau_psc(value: float):
        if value <= 0.0:
            raise ValueError("'tau_psc' must be > 0.")

    @staticmethod
    def _validate_tau_fac(value: float):
        if value < 0.0:
            raise ValueError("'tau_fac' must be >= 0.")

    @staticmethod
    def _validate_tau_rec(value: float):
        if value <= 0.0:
            raise ValueError("'tau_rec' must be > 0.")

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
        params['synapse_model'] = 'tsodyks_synapse'
        return params

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
        """Set NEST-style public parameters."""
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
        new_U = self.U if U is _UNSET else self._to_scalar_unit_interval(U, name='U')
        new_x = self.x if x is _UNSET else self._to_scalar_float(x, name='x')
        new_y = self.y if y is _UNSET else self._to_scalar_float(y, name='y')
        new_u = self.u if u is _UNSET else self._to_scalar_unit_interval(u, name='u')

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
        """Schedule one outgoing event with NEST ``tsodyks_synapse`` dynamics."""
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST uses the spike stamp and ignores precise sub-step offsets.
        t_spike = self._current_time_ms() + dt_ms
        h = float(t_spike - self.t_lastspike)

        puu = 0.0 if self.tau_fac == 0.0 else math.exp(-h / self.tau_fac)
        pyy = math.exp(-h / self.tau_psc)
        pzz = math.expm1(-h / self.tau_rec)
        pxy = (pzz * self.tau_rec - (pyy - 1.0) * self.tau_psc) / (self.tau_psc - self.tau_rec)

        z = 1.0 - self.x - self.y

        # Keep ordering identical to NEST models/tsodyks_synapse.h::send.
        self.u *= puu
        self.x += pxy * self.y - pzz * z
        self.y *= pyy

        self.u += self.U * (1.0 - self.u)

        delta_y_tsp = self.u * self.x
        self.x -= delta_y_tsp
        self.y += delta_y_tsp

        weighted_payload = multiplicity * (delta_y_tsp * self.weight)
        receiver = self._resolve_receiver(post)
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
