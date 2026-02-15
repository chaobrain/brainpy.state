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
    'tsodyks2_synapse',
]


_UNSET = object()


class tsodyks2_synapse(static_synapse):
    r"""NEST-compatible ``tsodyks2_synapse`` connection model.

    Short description
    -----------------

    Synapse type with short-term depression and facilitation.

    Description
    -----------

    ``tsodyks2_synapse`` mirrors NEST ``models/tsodyks2_synapse.h``.
    The model stores per-connection state variables:

    - ``x``: current scaling factor of synaptic efficacy,
    - ``u``: current release probability,
    - ``t_lastspike``: last presynaptic spike stamp.

    Together with fixed model parameters ``U``, ``tau_rec`` and ``tau_fac``,
    incoming spikes are processed in the same order as NEST:

    1. For ``h = t_spike - t_lastspike``, if this is not the first spike
       (i.e. ``t_lastspike >= 0``), propagate state to the current spike:

       .. math::
          x \leftarrow 1 + (x - xu - 1)e^{-h/\tau_{rec}}

          u \leftarrow U + u(1-U)e^{-h/\tau_{fac}}

       with the NEST special case ``tau_fac == 0``:

       .. math::
          e^{-h/\tau_{fac}} \equiv 0

    2. Compute effective synaptic weight for this spike:

       .. math::
          w_{eff} = x u w

    3. Schedule event delivery with inherited static-synapse delay semantics.

    4. Set ``t_lastspike = t_spike``.

    This model scales the baseline synaptic weight only and is suitable for
    current- or conductance-based postsynaptic dynamics.

    Event timing semantics
    ----------------------

    As in NEST, updates use spike stamps and ignore precise sub-step offsets.
    In this backend, each presynaptic event at simulation step ``t`` is
    evaluated at on-grid stamp ``t + dt``.

    Parameters
    ----------
    weight : ArrayLike, optional
        Baseline synaptic weight ``w``. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    U : ArrayLike, optional
        Utilization increment parameter in ``[0, 1]``. Default: ``0.5``.
    u : ArrayLike, optional
        Initial release probability in ``[0, 1]``. Default: ``U``.
    x : ArrayLike, optional
        Initial scaling factor of synaptic efficacy. Default: ``1.0``.
    tau_rec : ArrayLike, optional
        Recovery time constant in ms. Must be ``> 0``.
        Default: ``800.0 * u.ms``.
    tau_fac : ArrayLike, optional
        Facilitation time constant in ms. Must be ``>= 0``.
        Default: ``0.0 * u.ms``.
    post : object, optional
        Default receiver object.
    name : str, optional
        Object name.

    Notes
    -----
    - ``tau_fac == 0`` disables facilitation exactly as in NEST.
    - ``init_state()`` restores ``x``, ``u`` and ``t_lastspike`` to
      configured initial values.

    References
    ----------
    .. [1] NEST source: ``models/tsodyks2_synapse.h`` and
           ``models/tsodyks2_synapse.cpp``.
    .. [2] Tsodyks MV, Markram H (1997). The neural code between neocortical
           pyramidal neurons depends on neurotransmitter release probability.
           PNAS, 94(2):719-723.
    .. [3] Fuhrmann G, Segev I, Markram H, Tsodyks MV (2002).
           Coding of temporal information by activity-dependent synapses.
           Journal of Neurophysiology, 87(1):140-148.
    .. [4] Maass W, Markram H (2002). Synapses as dynamic memory buffers.
           Neural Networks, 15(2):155-161.
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        U: ArrayLike = 0.5,
        u: ArrayLike | object = _UNSET,
        x: ArrayLike = 1.0,
        tau_rec: ArrayLike = 800.0 * u.ms,
        tau_fac: ArrayLike = 0.0 * u.ms,
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

        self.U = self._to_scalar_unit_interval(U, name='U')
        u0 = self.U if u is _UNSET else self._to_scalar_unit_interval(u, name='u')

        self.x = self._to_scalar_float(x, name='x')
        self.u = float(u0)
        self.tau_rec = self._to_scalar_time_ms(tau_rec, name='tau_rec')
        self.tau_fac = self._to_scalar_time_ms(tau_fac, name='tau_fac')

        self._validate_tau_rec(self.tau_rec)
        self._validate_tau_fac(self.tau_fac)

        self._x0 = float(self.x)
        self._u0 = float(self.u)

        self.t_lastspike = -1.0

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_unit_interval(value: ArrayLike, *, name: str) -> float:
        v = tsodyks2_synapse._to_scalar_float(value, name=name)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"'{name}' must be in [0,1].")
        return float(v)

    @staticmethod
    def _validate_tau_rec(value: float):
        if value <= 0.0:
            raise ValueError("'tau_rec' must be > 0.")

    @staticmethod
    def _validate_tau_fac(value: float):
        if value < 0.0:
            raise ValueError("'tau_fac' must be >= 0.")

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        super().init_state()
        self.x = float(self._x0)
        self.u = float(self._u0)
        self.t_lastspike = -1.0

    def get(self) -> dict:
        """Return current public parameters and mutable state."""
        params = super().get()
        params['U'] = float(self.U)
        params['u'] = float(self.u)
        params['x'] = float(self.x)
        params['tau_rec'] = float(self.tau_rec)
        params['tau_fac'] = float(self.tau_fac)
        params['synapse_model'] = 'tsodyks2_synapse'
        return params

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        U: ArrayLike | object = _UNSET,
        u: ArrayLike | object = _UNSET,
        x: ArrayLike | object = _UNSET,
        tau_rec: ArrayLike | object = _UNSET,
        tau_fac: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        """Set NEST-style public parameters."""
        new_U = self.U if U is _UNSET else self._to_scalar_unit_interval(U, name='U')
        new_u = self.u if u is _UNSET else self._to_scalar_unit_interval(u, name='u')
        new_x = self.x if x is _UNSET else self._to_scalar_float(x, name='x')
        new_tau_rec = (
            self.tau_rec
            if tau_rec is _UNSET
            else self._to_scalar_time_ms(tau_rec, name='tau_rec')
        )
        new_tau_fac = (
            self.tau_fac
            if tau_fac is _UNSET
            else self._to_scalar_time_ms(tau_fac, name='tau_fac')
        )

        self._validate_tau_rec(float(new_tau_rec))
        self._validate_tau_fac(float(new_tau_fac))

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

        self.U = float(new_U)
        self.u = float(new_u)
        self.x = float(new_x)
        self.tau_rec = float(new_tau_rec)
        self.tau_fac = float(new_tau_fac)

        self._x0 = float(self.x)
        self._u0 = float(self.u)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        """Schedule one outgoing event with NEST ``tsodyks2_synapse`` dynamics."""
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST evaluates this model on the spike stamp.
        t_spike = self._current_time_ms() + dt_ms

        if self.t_lastspike >= 0.0:
            h = float(t_spike - self.t_lastspike)
            x_decay = math.exp(-h / self.tau_rec)
            u_decay = 0.0 if self.tau_fac == 0.0 else math.exp(-h / self.tau_fac)

            # Keep ordering identical to NEST models/tsodyks2_synapse.h::send.
            self.x = 1.0 + (self.x - self.x * self.u - 1.0) * x_decay
            self.u = self.U + self.u * (1.0 - self.U) * u_decay

        receiver = self._resolve_receiver(post)
        weighted_payload = multiplicity * (self.x * self.u * self.weight)
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
