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
    'quantal_stp_synapse',
]


_UNSET = object()


class quantal_stp_synapse(static_synapse):
    r"""NEST-compatible ``quantal_stp_synapse`` connection model.

    Short description
    -----------------

    Probabilistic synapse model with short-term plasticity.

    Description
    -----------

    ``quantal_stp_synapse`` mirrors NEST ``models/quantal_stp_synapse.h``.
    It is the stochastic quantal release variant of short-term plasticity,
    where each connection stores:

    - ``u``: dynamic release probability per site,
    - ``n``: total number of release sites,
    - ``a``: currently available release sites,
    - ``t_lastspike``: last presynaptic spike stamp.

    On each incoming spike at stamp :math:`t_s`:

    1. If this is not the first spike (``t_lastspike >= 0``), propagate state:

       .. math::
          h = t_s - t_{\mathrm{last}}

          p_{\mathrm{decay}} = e^{-h/\tau_{rec}}

          u_{\mathrm{decay}} =
            \begin{cases}
              0, & \tau_{fac} < 10^{-10} \\
              e^{-h/\tau_{fac}}, & \text{otherwise}
            \end{cases}

          u \leftarrow U + u(1-U)u_{\mathrm{decay}}

       Then recover depleted sites stochastically. For each depleted site
       (``n - a`` independent trials), recover with probability
       :math:`1 - p_{\mathrm{decay}}`.

    2. Draw released sites from currently available sites: for each of the
       ``a`` available sites, release with probability ``u``.

    3. If at least one site releases (``n_release > 0``), send one event with
       weight :math:`w_{\mathrm{eff}} = n_{\mathrm{release}} \cdot w` and set
       ``a <- a - n_release``.

    4. Set ``t_lastspike = t_s`` irrespective of whether a spike was sent.

    This ordering is identical to NEST source
    ``models/quantal_stp_synapse.h::send``.

    Event timing semantics
    ----------------------

    As in NEST, update uses spike stamps and ignores precise sub-step offsets.
    In this backend each event processed at step ``t`` uses on-grid stamp
    ``t + dt``.

    Parameters
    ----------
    weight : ArrayLike, optional
        Baseline per-site synaptic weight ``w``. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    U : ArrayLike, optional
        Maximal release probability in ``[0, 1]``. Default: ``0.5``.
    u : ArrayLike, optional
        Initial release probability in ``[0, 1]``. Default: ``U``.
    n : ArrayLike, optional
        Total number of release sites. Default: ``1``.
    a : ArrayLike, optional
        Initial number of available release sites. Default: ``n``.
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
    - Random draws use NumPy global RNG (``np.random``).
    - ``init_state()`` restores mutable connection state ``u``, ``a`` and
      ``t_lastspike`` to configured initial values.

    References
    ----------
    .. [1] NEST source: ``models/quantal_stp_synapse.h`` and
           ``models/quantal_stp_synapse_impl.h``.
    .. [2] Fuhrmann G, Segev I, Markram H, Tsodyks MV (2002).
           Coding of temporal information by activity-dependent synapses.
           Journal of Neurophysiology, 87(1):140-148.
    .. [3] Loebel A, Silberberg G, Helbig D, Markram H, Tsodyks MV,
           Richardson MJE (2009). Multiquantal release underlies the
           distribution of synaptic efficacies in the neocortex.
           Frontiers in Computational Neuroscience, 3:27.
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
        n: ArrayLike = 1,
        a: ArrayLike | object = _UNSET,
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
        self.u = self.U if u is _UNSET else self._to_scalar_unit_interval(u, name='u')
        self.n = self._to_scalar_int(n, name='n')
        self.a = self.n if a is _UNSET else self._to_scalar_int(a, name='a')
        self.tau_rec = self._to_scalar_time_ms(tau_rec, name='tau_rec')
        self.tau_fac = self._to_scalar_time_ms(tau_fac, name='tau_fac')

        self._validate_tau_rec(float(self.tau_rec))
        self._validate_tau_fac(float(self.tau_fac))

        self._u0 = float(self.u)
        self._a0 = int(self.a)

        self.t_lastspike = -1.0

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_unit_interval(value: ArrayLike, *, name: str) -> float:
        v = quantal_stp_synapse._to_scalar_float(value, name=name)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"'{name}' must be in [0,1].")
        return float(v)

    @staticmethod
    def _to_scalar_int(value: ArrayLike, *, name: str) -> int:
        v = quantal_stp_synapse._to_scalar_float(value, name=name)
        if not float(v).is_integer():
            raise ValueError(f"'{name}' must be an integer.")
        return int(v)

    @staticmethod
    def _validate_tau_rec(value: float):
        if value <= 0.0:
            raise ValueError("'tau_rec' must be > 0.")

    @staticmethod
    def _validate_tau_fac(value: float):
        if value < 0.0:
            raise ValueError("'tau_fac' must be >= 0.")

    @staticmethod
    def _sample_uniform() -> float:
        return float(np.random.random())

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        super().init_state()
        self.u = float(self._u0)
        self.a = int(self._a0)
        self.t_lastspike = -1.0

    def get(self) -> dict:
        """Return current public parameters and mutable state."""
        params = super().get()
        params['U'] = float(self.U)
        params['u'] = float(self.u)
        params['tau_rec'] = float(self.tau_rec)
        params['tau_fac'] = float(self.tau_fac)
        params['n'] = int(self.n)
        params['a'] = int(self.a)
        params['synapse_model'] = 'quantal_stp_synapse'
        return params

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        U: ArrayLike | object = _UNSET,
        u: ArrayLike | object = _UNSET,
        n: ArrayLike | object = _UNSET,
        a: ArrayLike | object = _UNSET,
        tau_rec: ArrayLike | object = _UNSET,
        tau_fac: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        """Set NEST-style public parameters."""
        new_U = self.U if U is _UNSET else self._to_scalar_unit_interval(U, name='U')
        new_u = self.u if u is _UNSET else self._to_scalar_unit_interval(u, name='u')
        new_n = self.n if n is _UNSET else self._to_scalar_int(n, name='n')
        new_a = self.a if a is _UNSET else self._to_scalar_int(a, name='a')
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
        self.n = int(new_n)
        self.a = int(new_a)
        self.tau_rec = float(new_tau_rec)
        self.tau_fac = float(new_tau_fac)

        self._u0 = float(self.u)
        self._a0 = int(self.a)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        """Schedule one outgoing event with NEST ``quantal_stp_synapse`` dynamics."""
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST evaluates this model on spike stamps.
        t_spike = self._current_time_ms() + dt_ms

        if self.t_lastspike >= 0.0:
            h = float(t_spike - self.t_lastspike)
            p_decay = math.exp(-h / self.tau_rec)
            u_decay = 0.0 if self.tau_fac < 1.0e-10 else math.exp(-h / self.tau_fac)

            # Keep ordering identical to NEST models/quantal_stp_synapse.h::send.
            self.u = self.U + self.u * (1.0 - self.U) * u_decay

            recovery_prob = 1.0 - p_decay
            depleted_sites = int(self.n - self.a)
            for _ in range(depleted_sites if depleted_sites > 0 else 0):
                if self._sample_uniform() < recovery_prob:
                    self.a += 1

        n_release = 0
        available_sites = int(self.a)
        for _ in range(available_sites if available_sites > 0 else 0):
            if self._sample_uniform() < self.u:
                n_release += 1

        send_spike = n_release > 0
        if send_spike:
            receiver = self._resolve_receiver(post)
            weighted_payload = multiplicity * (float(n_release) * self.weight)
            rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)

            delivery_step = int(current_step + int(self._delay_steps))
            self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))
            self.a -= int(n_release)

        self.t_lastspike = float(t_spike)
        return send_spike

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
