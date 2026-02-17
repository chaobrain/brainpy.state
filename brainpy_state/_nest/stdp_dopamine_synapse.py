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

    Synapse type for dopamine-modulated spike-timing dependent plasticity following
    the NEST ``models/stdp_dopamine_synapse`` implementation. This model combines
    classical STDP with reward modulation through an eligibility trace mechanism,
    enabling reinforcement learning in spiking neural networks.

    **1. Model Overview**

    ``stdp_dopamine_synapse`` implements dopamine-modulated STDP with per-connection
    state variables:

    - ``weight``: synaptic efficacy (modulated by dopamine)
    - ``Kplus``: presynaptic facilitation trace
    - ``c``: eligibility trace (records recent spike timing correlations)
    - ``n``: dopamine trace (reward signal)
    - ``t_last_update``: timestamp of last propagated state update
    - ``t_lastspike``: timestamp of previous presynaptic spike

    In NEST, the postsynaptic depression trace ``Kminus`` is read from the postsynaptic
    archiving neuron. For standalone compatibility, this implementation maintains an
    internal post-spike history buffer parameterized by ``tau_minus`` (not a synapse
    parameter in NEST but a neuron property).

    Dopaminergic spikes are provided by a NEST ``volume_transmitter`` device. In this
    implementation, dopamine spikes can be fed through ``update(..., dopa_spike=...)``
    or :meth:`record_dopa_spike`, while still requiring a non-``None``
    ``volume_transmitter`` handle to preserve NEST connection semantics.

    **2. Mathematical Formulation**

    Between spike events, the model integrates three coupled differential equations:

    .. math::
       \frac{dw}{dt} &= c(t) \cdot (n(t) - b) \\
       \frac{dc}{dt} &= -\frac{c}{\tau_c} \\
       \frac{dn}{dt} &= -\frac{n}{\tau_n}

    where:

    - :math:`w`: synaptic weight
    - :math:`c`: eligibility trace
    - :math:`n`: dopamine concentration
    - :math:`b`: dopamine baseline
    - :math:`\tau_c`: eligibility trace decay time constant
    - :math:`\tau_n`: dopamine trace decay time constant

    Weight updates are computed analytically using ``expm1`` over each time interval:

    .. math::
       w \leftarrow w - c_0 \left(
       \frac{n_0}{\tau_s} \cdot \mathrm{expm1}(\tau_s \Delta t)
       - b \tau_c \cdot \mathrm{expm1}\left(\frac{\Delta t}{\tau_c}\right)
       \right)

    where :math:`\tau_s = \frac{\tau_c + \tau_n}{\tau_c \tau_n}` and :math:`\Delta t = t_0 - t_1 \le 0`.
    The weight is then clipped to :math:`[W_{\min}, W_{\max}]`.

    **3. Eligibility Trace Updates**

    At spike events, the eligibility trace is modified by:

    - **Facilitation** (pre-before-post): :math:`c \leftarrow c + A_+ K_+(t)`
    - **Depression** (post-before-pre): :math:`c \leftarrow c - A_- K_-(t)`

    where :math:`K_+` and :math:`K_-` are exponentially decaying spike traces:

    .. math::
       K_+(t) &= \sum_{t_i^{\mathrm{pre}} < t} \exp\left(-\frac{t - t_i^{\mathrm{pre}}}{\tau_+}\right) \\
       K_-(t) &= \sum_{t_j^{\mathrm{post}} < t} \exp\left(-\frac{t - t_j^{\mathrm{post}}}{\tau_-}\right)

    **4. Update Order for Presynaptic Spikes**

    For a presynaptic spike at time :math:`t_{\mathrm{pre}}` with dendritic delay :math:`d`,
    NEST ``stdp_dopamine_synapse::send`` performs:

    1. Read postsynaptic history in :math:`(t_{\mathrm{last\_update}} - d,\; t_{\mathrm{pre}} - d]`
    2. For each postsynaptic spike :math:`t_{\mathrm{post}}` in that range:

       a. Propagate dopamine/eligibility/weight to :math:`t_{\mathrm{post}} + d`
       b. If :math:`t_{\mathrm{pre}} - t_{\mathrm{post}} > \epsilon`, facilitate:
          :math:`c \leftarrow c + A_+ K_+ \exp((t_{\mathrm{last\_update}} - (t_{\mathrm{post}} + d)) / \tau_+)`

    3. Propagate dopamine/eligibility/weight to :math:`t_{\mathrm{pre}}`
    4. Depress eligibility: :math:`c \leftarrow c - A_- K_-(t_{\mathrm{pre}} - d)`
    5. Send event with updated ``weight``
    6. Update presynaptic trace:
       :math:`K_+ \leftarrow K_+ \exp((t_{\mathrm{last\_update}} - t_{\mathrm{pre}}) / \tau_+) + 1`
    7. Set :math:`t_{\mathrm{last\_update}} = t_{\mathrm{lastspike}} = t_{\mathrm{pre}}`

    This implementation preserves the exact NEST update ordering.

    **5. Event Timing Semantics**

    As in NEST, this model uses on-grid spike timestamps and ignores precise sub-step
    offsets for plasticity updates. All spike times are rounded to simulation time steps.

    Parameters
    ----------
    weight : float or array-like, optional
        Initial synaptic weight (unitless). Default: ``1.0``.
    delay : Quantity or array-like, optional
        Synaptic delay. Must have time units. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor identifier on the postsynaptic neuron. Default: ``0``.
    volume_transmitter : object or None, optional
        Handle to a dopamine volume transmitter (NEST compatibility placeholder).
        Must be set to a non-``None`` value before simulation starts. Default: ``None``.
    A_plus : float or array-like, optional
        Facilitation amplitude for pre-before-post spike pairs (unitless). Default: ``1.0``.
    A_minus : float or array-like, optional
        Depression amplitude for post-before-pre spike pairs (unitless). Default: ``1.5``.
    tau_plus : Quantity or array-like, optional
        Presynaptic trace decay time constant. Must have time units. Default: ``20.0 * u.ms``.
    tau_minus : Quantity or array-like, optional
        Postsynaptic trace decay time constant. Must have time units.
        In NEST, this is a neuron property; here it is stored on the synapse for
        standalone operation. Default: ``20.0 * u.ms``.
    tau_c : Quantity or array-like, optional
        Eligibility trace decay time constant. Must have time units. Default: ``1000.0 * u.ms``.
    tau_n : Quantity or array-like, optional
        Dopamine trace decay time constant. Must have time units. Default: ``200.0 * u.ms``.
    b : float or array-like, optional
        Dopamine baseline concentration (unitless). Weight changes occur proportionally
        to :math:`(n - b)`. Default: ``0.0``.
    Wmin : float or array-like, optional
        Minimum allowed weight (hard lower bound). Default: ``0.0``.
    Wmax : float or array-like, optional
        Maximum allowed weight (hard upper bound). Default: ``200.0``.
    Kplus : float or array-like, optional
        Initial presynaptic trace value. Must be non-negative. Default: ``0.0``.
    c : float or array-like, optional
        Initial eligibility trace value. Default: ``0.0``.
    n : float or array-like, optional
        Initial dopamine trace value. Default: ``0.0``.
    post : Dynamics or None, optional
        Default postsynaptic receiver object. Can be overridden in :meth:`send`.
        Default: ``None``.
    name : str or None, optional
        Object name for identification. Default: ``None``.

    Parameter Mapping
    -----------------

    NEST-to-BrainPy parameter correspondence:

    +----------------------+----------------------+------------------+----------------------------+
    | NEST Parameter       | BrainPy Parameter    | Units            | Description                |
    +======================+======================+==================+============================+
    | ``weight``           | ``weight``           | unitless         | Synaptic efficacy          |
    +----------------------+----------------------+------------------+----------------------------+
    | ``delay``            | ``delay``            | ms               | Synaptic delay             |
    +----------------------+----------------------+------------------+----------------------------+
    | ``receptor_type``    | ``receptor_type``    | integer          | Postsynaptic receptor port |
    +----------------------+----------------------+------------------+----------------------------+
    | ``vt``               | ``volume_transmitter``| handle          | Dopamine volume transmitter|
    +----------------------+----------------------+------------------+----------------------------+
    | ``A_plus``           | ``A_plus``           | unitless         | Facilitation amplitude     |
    +----------------------+----------------------+------------------+----------------------------+
    | ``A_minus``          | ``A_minus``          | unitless         | Depression amplitude       |
    +----------------------+----------------------+------------------+----------------------------+
    | ``tau_plus``         | ``tau_plus``         | ms               | Pre-trace time constant    |
    +----------------------+----------------------+------------------+----------------------------+
    | (neuron property)    | ``tau_minus``        | ms               | Post-trace time constant   |
    +----------------------+----------------------+------------------+----------------------------+
    | ``tau_c``            | ``tau_c``            | ms               | Eligibility decay constant |
    +----------------------+----------------------+------------------+----------------------------+
    | ``tau_n``            | ``tau_n``            | ms               | Dopamine decay constant    |
    +----------------------+----------------------+------------------+----------------------------+
    | ``b``                | ``b``                | unitless         | Dopamine baseline          |
    +----------------------+----------------------+------------------+----------------------------+
    | ``Wmin``             | ``Wmin``             | unitless         | Minimum weight bound       |
    +----------------------+----------------------+------------------+----------------------------+
    | ``Wmax``             | ``Wmax``             | unitless         | Maximum weight bound       |
    +----------------------+----------------------+------------------+----------------------------+
    | ``Kplus``            | ``Kplus``            | unitless         | Presynaptic trace state    |
    +----------------------+----------------------+------------------+----------------------------+
    | ``c``                | ``c``                | unitless         | Eligibility trace state    |
    +----------------------+----------------------+------------------+----------------------------+
    | ``n``                | ``n``                | unitless         | Dopamine trace state       |
    +----------------------+----------------------+------------------+----------------------------+

    Raises
    ------
    ValueError
        - If any time constant (``tau_plus``, ``tau_minus``, ``tau_c``, ``tau_n``) is non-positive.
        - If ``Kplus`` is negative.
        - If ``volume_transmitter`` is ``None`` when ``send`` or ``update`` is called.
        - If dopamine spikes are recorded out of temporal order.
        - If ``trigger_update_weight`` is called with a time earlier than ``t_last_update``.
        - If any scalar parameter has non-scalar shape or non-finite value.

    Notes
    -----
    - This model transmits spike-like events only (no continuous current injection).
    - ``update(pre_spike=..., post_spike=..., dopa_spike=...)`` supports explicit
      per-step dopamine multiplicity for standalone simulations without a separate
      volume transmitter object.
    - One ``trigger_update_weight`` propagation is performed per simulation step in
      :meth:`update`, corresponding to NEST's default ``volume_transmitter`` delivery
      interval.
    - Common properties (``A_plus``, ``A_minus``, ``tau_plus``, ``tau_c``, ``tau_n``,
      ``Wmin``, ``Wmax``, ``b``, ``volume_transmitter``) cannot be specified in
      connect-time synapse specs; set them on the model template (via
      ``CopyModel``/``SetDefaults`` in NEST parlance).
    - For large-scale simulations, consider memory overhead: each synapse maintains
      internal postsynaptic and dopamine spike histories.

    Examples
    --------
    Basic usage with explicit dopamine signaling:

    .. code-block:: python

       >>> import brainpy.state as bp
       >>> import brainunit as u
       >>> # Create synapse with volume transmitter placeholder
       >>> syn = bp.stdp_dopamine_synapse(
       ...     weight=1.0,
       ...     delay=1.0 * u.ms,
       ...     A_plus=1.0,
       ...     A_minus=1.5,
       ...     tau_plus=20.0 * u.ms,
       ...     tau_minus=20.0 * u.ms,
       ...     tau_c=1000.0 * u.ms,
       ...     tau_n=200.0 * u.ms,
       ...     b=0.0,
       ...     Wmin=0.0,
       ...     Wmax=10.0,
       ...     volume_transmitter=object()  # dummy handle
       ... )
       >>> syn.init_state()
       >>> # Simulate pre spike followed by post spike (causal pairing)
       >>> syn.update(pre_spike=1.0, post_spike=0.0)  # pre spike at t=dt
       >>> syn.update(pre_spike=0.0, post_spike=1.0)  # post spike at t=2*dt
       >>> # Deliver reward signal
       >>> syn.update(pre_spike=0.0, post_spike=0.0, dopa_spike=1.0)  # dopamine at t=3*dt
       >>> print(f"Weight after reward: {syn.weight:.3f}")
       Weight after reward: ...

    Recording dopamine spikes explicitly:

    .. code-block:: python

       >>> syn = bp.stdp_dopamine_synapse(
       ...     weight=5.0,
       ...     volume_transmitter=object(),
       ...     tau_c=1000.0 * u.ms,
       ...     tau_n=200.0 * u.ms
       ... )
       >>> syn.init_state()
       >>> # Record multiple dopamine spikes
       >>> syn.record_dopa_spike(1.0, t_spike_ms=10.0)
       1.0
       >>> syn.record_dopa_spike(0.5, t_spike_ms=15.0)
       0.5
       >>> # Trigger weight update to specified time
       >>> syn.trigger_update_weight(t_trig_ms=20.0)
       >>> print(f"Weight: {syn.weight:.3f}")
       Weight: ...

    See Also
    --------
    static_synapse : Base class for static synaptic connections
    stdp_synapse : Classical spike-timing dependent plasticity without dopamine modulation
    tsodyks_synapse : Short-term synaptic plasticity (depression and facilitation)

    References
    ----------
    .. [1] NEST source: ``models/stdp_dopamine_synapse.h``,
           ``models/stdp_dopamine_synapse.cpp``,
           ``models/volume_transmitter.h``,
           ``models/volume_transmitter.cpp``.
    .. [2] Potjans W, Morrison A, Diesmann M (2010). Enabling functional neural circuit
           simulations with distributed computing of neuromodulated plasticity.
           Frontiers in Computational Neuroscience, 4:141.
           https://doi.org/10.3389/fncom.2010.00141
    .. [3] Izhikevich EM (2007). Solving the distal reward problem through linkage of
           STDP and dopamine signaling. Cerebral Cortex, 17(10):2443-2452.
           https://doi.org/10.1093/cercor/bhl152
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
        """Clear internal postsynaptic STDP history state.

        Resets the postsynaptic spike history buffer and depression trace to initial
        conditions. This is useful when reinitializing the synapse or starting a new
        trial in an experiment.

        Notes
        -----
        Does not affect presynaptic traces (``Kplus``), eligibility trace (``c``),
        dopamine trace (``n``), or weight. To fully reset the synapse, use
        :meth:`init_state` instead.
        """
        self._post_kminus = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t = []
        self._post_hist_kminus = []

    def clear_dopamine_history(self):
        """Reset internal dopamine spike history.

        Clears the dopamine spike buffer and reinitializes it with a single pseudo-spike
        at the current ``t_last_update`` time with zero multiplicity. This effectively
        resets the dopamine delivery history while preserving temporal continuity.

        Notes
        -----
        Does not modify the current dopamine trace value (``n``). To reset ``n`` as well,
        use :meth:`set` with ``n=0.0`` or call :meth:`init_state`.
        """
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
        """Record postsynaptic spikes into internal STDP history.

        Adds postsynaptic spike events to the internal history buffer and updates the
        postsynaptic depression trace (``Kminus``). These spikes are used during
        subsequent presynaptic spike processing to compute eligibility trace updates.

        Parameters
        ----------
        multiplicity : float or array-like, optional
            Number of spikes to record (must be non-negative integer or convertible
            to integer). Fractional values will be rounded. Default: ``1.0``.
        t_spike_ms : float, array-like, or None, optional
            Explicit spike timestamp in milliseconds. If ``None``, uses the current
            simulation time plus one timestep (``t + dt``). Default: ``None``.

        Returns
        -------
        int
            Number of spikes actually recorded (rounded ``multiplicity``).

        Raises
        ------
        ValueError
            If ``multiplicity`` is not a non-negative integer (after rounding).
        ValueError
            If ``t_spike_ms`` is not a scalar or not finite.

        Notes
        -----
        - Multiple spikes at the same timestamp accumulate exponentially in the ``Kminus``
          trace: :math:`K_- \leftarrow (K_- + 1)^{\text{multiplicity}}`.
        - This method does not trigger plasticity updates; it only records spike times
          for future processing during :meth:`send` or :meth:`trigger_update_weight`.

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.stdp_dopamine_synapse(volume_transmitter=object())
           >>> syn.init_state()
           >>> # Record single spike at current time
           >>> n = syn.record_post_spike(1.0)
           >>> print(n)
           1
           >>> # Record multiple spikes at explicit time
           >>> n = syn.record_post_spike(3.0, t_spike_ms=25.0)
           >>> print(n)
           3
        """
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
        """Record dopamine spikes into internal volume-transmitter history.

        Adds dopamine delivery events to the internal dopamine spike buffer. These
        events are processed during subsequent :meth:`trigger_update_weight` calls
        to update the dopamine trace (``n``) and modulate synaptic weight changes.

        Parameters
        ----------
        multiplicity : float or array-like, optional
            Dopamine spike magnitude (must be non-negative float). Unlike postsynaptic
            spikes, dopamine "spikes" can have arbitrary positive real values representing
            dopamine concentration increments. Default: ``1.0``.
        t_spike_ms : float, array-like, or None, optional
            Explicit timestamp in milliseconds for dopamine delivery. If ``None``, uses
            current simulation time plus one timestep (``t + dt``). Default: ``None``.

        Returns
        -------
        float
            Dopamine multiplicity actually recorded.

        Raises
        ------
        ValueError
            If ``multiplicity`` is negative.
        ValueError
            If ``t_spike_ms`` is earlier than the last recorded dopamine spike time
            (dopamine spikes must be recorded in non-decreasing temporal order).
        ValueError
            If ``multiplicity`` or ``t_spike_ms`` is not scalar or not finite.

        Notes
        -----
        - Multiple dopamine deliveries at the same timestamp (within tolerance
          :math:`\epsilon = 10^{-6}` ms) are accumulated additively.
        - Dopamine spikes are not processed immediately; they are stored in a queue
          and integrated during the next :meth:`trigger_update_weight` call.
        - To simulate volume transmission delays, record dopamine spikes at future
          timestamps relative to the triggering event.

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.stdp_dopamine_synapse(
           ...     volume_transmitter=object(),
           ...     tau_n=200.0 * u.ms
           ... )
           >>> syn.init_state()
           >>> # Record reward signal at t=100 ms
           >>> mult = syn.record_dopa_spike(1.5, t_spike_ms=100.0)
           >>> print(f"Recorded dopamine: {mult}")
           Recorded dopamine: 1.5
           >>> # Accumulate additional dopamine at same time
           >>> mult = syn.record_dopa_spike(0.5, t_spike_ms=100.0)
           >>> print(f"Total at t=100: {syn._dopa_spikes[-1][1]}")
           Total at t=100: 2.0
           >>> # Trigger update to integrate dopamine
           >>> syn.trigger_update_weight(t_trig_ms=150.0)
        """
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
        """Initialize or reset all synapse state variables to their default values.

        Resets all dynamic state (traces, timestamps, spike histories) to initial
        conditions as specified during construction or via :meth:`set`. This is typically
        called at the start of a simulation.

        Parameters
        ----------
        batch_size : int or None, optional
            Ignored (included for API compatibility). This synapse operates in scalar mode.
        **kwargs
            Additional keyword arguments (ignored).

        Notes
        -----
        Resets the following state variables:

        - ``Kplus``: presynaptic trace
        - ``c``: eligibility trace
        - ``n``: dopamine trace
        - ``t_last_update``: last update timestamp
        - ``t_lastspike``: last presynaptic spike timestamp
        - Postsynaptic spike history (cleared)
        - Dopamine spike history (initialized with zero-multiplicity pseudo-spike)
        - Event delivery queue (inherited from parent)

        Does not reset parameters (``A_plus``, ``tau_c``, etc.) or initial weight.
        """
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
        """Return current public parameters and mutable state.

        Retrieves all NEST-style synapse parameters and state variables as a dictionary,
        suitable for inspection, serialization, or comparison with NEST ``GetStatus``
        output.

        Returns
        -------
        dict
            Dictionary containing:

            - All parent class parameters (``weight``, ``delay``, ``receptor_type``, etc.)
            - ``volume_transmitter``: dopamine volume transmitter handle
            - ``A_plus``: facilitation amplitude
            - ``A_minus``: depression amplitude
            - ``tau_plus``: presynaptic trace time constant (ms)
            - ``tau_minus``: postsynaptic trace time constant (ms)
            - ``tau_c``: eligibility trace time constant (ms)
            - ``tau_n``: dopamine trace time constant (ms)
            - ``b``: dopamine baseline
            - ``Wmin``: minimum weight bound
            - ``Wmax``: maximum weight bound
            - ``Kplus``: current presynaptic trace value
            - ``c``: current eligibility trace value
            - ``n``: current dopamine trace value
            - ``synapse_model``: model identifier (``'stdp_dopamine_synapse'``)

        Notes
        -----
        Time constants are returned in milliseconds (internal representation), not as
        ``brainunit.Quantity`` objects.

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.stdp_dopamine_synapse(
           ...     weight=5.0,
           ...     A_plus=1.0,
           ...     volume_transmitter=object()
           ... )
           >>> syn.init_state()
           >>> params = syn.get()
           >>> print(params['weight'])
           5.0
           >>> print(params['A_plus'])
           1.0
           >>> print(params['synapse_model'])
           stdp_dopamine_synapse
        """
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
        """Validate connect-time synapse specification for disallowed common properties.

        Enforces NEST convention that certain parameters (STDP learning rules, dopamine
        modulation parameters) must be set on the synapse model template rather than
        per-connection at connect time.

        Parameters
        ----------
        syn_spec : dict or None
            Synapse specification dictionary to validate. If ``None``, no validation
            is performed.

        Raises
        ------
        ValueError
            If ``syn_spec`` contains any of the following disallowed keys:
            ``'vt'``, ``'volume_transmitter'``, ``'A_plus'``, ``'A_minus'``,
            ``'tau_plus'``, ``'tau_c'``, ``'tau_n'``, ``'Wmin'``, ``'Wmax'``, ``'b'``.

        Notes
        -----
        In NEST, dopamine synapse common properties are set globally on the synapse
        model (via ``CopyModel`` or ``SetDefaults``), not per-connection. This method
        enforces that convention to prevent user confusion and maintain NEST compatibility.

        Allowed per-connection parameters: ``weight``, ``delay``, ``receptor_type``.

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.stdp_dopamine_synapse(volume_transmitter=object())
           >>> # Valid: per-connection weight/delay
           >>> syn.check_synapse_params({'weight': 2.0, 'delay': 1.5})
           >>> # Invalid: common property at connect time
           >>> syn.check_synapse_params({'A_plus': 1.0})
           Traceback (most recent call last):
               ...
           ValueError: A_plus cannot be specified in connect-time synapse parameters ...
        """
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
        """Set NEST-style public parameters and mutable state.

        Updates synapse parameters and/or state variables. Validates all new values
        before applying changes. Only parameters explicitly provided are modified.

        Parameters
        ----------
        weight : float, array-like, or sentinel, optional
            New synaptic weight. If ``_UNSET``, weight is not changed.
        delay : Quantity, array-like, or sentinel, optional
            New synaptic delay. If ``_UNSET``, delay is not changed.
        receptor_type : int or sentinel, optional
            New receptor port identifier. If ``_UNSET``, receptor type is not changed.
        volume_transmitter : object or sentinel, optional
            New volume transmitter handle. If ``_UNSET``, handle is not changed.
        A_plus : float, array-like, or sentinel, optional
            New facilitation amplitude. If ``_UNSET``, not changed.
        A_minus : float, array-like, or sentinel, optional
            New depression amplitude. If ``_UNSET``, not changed.
        tau_plus : Quantity, array-like, or sentinel, optional
            New presynaptic trace time constant. If ``_UNSET``, not changed.
        tau_minus : Quantity, array-like, or sentinel, optional
            New postsynaptic trace time constant. If ``_UNSET``, not changed.
        tau_c : Quantity, array-like, or sentinel, optional
            New eligibility trace time constant. If ``_UNSET``, not changed.
        tau_n : Quantity, array-like, or sentinel, optional
            New dopamine trace time constant. If ``_UNSET``, not changed.
        b : float, array-like, or sentinel, optional
            New dopamine baseline. If ``_UNSET``, not changed.
        Wmin : float, array-like, or sentinel, optional
            New minimum weight bound. If ``_UNSET``, not changed.
        Wmax : float, array-like, or sentinel, optional
            New maximum weight bound. If ``_UNSET``, not changed.
        Kplus : float, array-like, or sentinel, optional
            New presynaptic trace value. Must be non-negative. If ``_UNSET``, not changed.
        c : float, array-like, or sentinel, optional
            New eligibility trace value. If ``_UNSET``, not changed.
        n : float, array-like, or sentinel, optional
            New dopamine trace value. If ``_UNSET``, not changed.
        post : object or sentinel, optional
            New default postsynaptic receiver. If ``_UNSET``, not changed.

        Raises
        ------
        ValueError
            - If any time constant is non-positive.
            - If ``Kplus`` is negative.
            - If any parameter has non-scalar shape or non-finite value.

        Notes
        -----
        - Changing parameters during a simulation may produce non-physical discontinuities.
          For clean state resets, use :meth:`init_state` instead.
        - Updating initial state values (``Kplus``, ``c``, ``n``) also updates the
          stored initial conditions used by :meth:`init_state`.

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.stdp_dopamine_synapse(
           ...     weight=1.0,
           ...     A_plus=1.0,
           ...     volume_transmitter=object()
           ... )
           >>> syn.init_state()
           >>> # Modify learning rate mid-simulation
           >>> syn.set(A_plus=2.0, A_minus=3.0)
           >>> print(syn.A_plus)
           2.0
           >>> # Reset dopamine trace
           >>> syn.set(n=0.5)
           >>> print(syn.n)
           0.5
        """
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
        """Schedule one outgoing spike event with dopamine-modulated STDP updates.

        Implements the NEST ``stdp_dopamine_synapse::send`` method: processes postsynaptic
        spike history, updates eligibility trace, propagates dopamine and weight dynamics,
        then schedules a delayed synaptic event to the postsynaptic target.

        Parameters
        ----------
        multiplicity : float or array-like, optional
            Presynaptic spike multiplicity (typically ``1.0`` for single spikes, or
            higher for burst events). If zero, no event is sent. Default: ``1.0``.
        post : Dynamics or None, optional
            Postsynaptic receiver object. If ``None``, uses the synapse's default ``post``
            attribute. Default: ``None``.
        receptor_type : int, array-like, or None, optional
            Target receptor port on postsynaptic neuron. If ``None``, uses the synapse's
            default ``receptor_type``. Default: ``None``.

        Returns
        -------
        bool
            ``True`` if an event was scheduled, ``False`` if ``multiplicity`` was zero.

        Raises
        ------
        ValueError
            If ``volume_transmitter`` is ``None`` (must be assigned before sending events).

        Notes
        -----
        **Update sequence (following NEST ``stdp_dopamine_synapse::send``):**

        1. Retrieve postsynaptic spike history in the window
           :math:`(t_{\mathrm{last\_update}} - d,\; t_{\mathrm{pre}} - d]`
           where :math:`d` is dendritic delay.
        2. For each postsynaptic spike :math:`t_{\mathrm{post}}` in that window:

           a. Propagate dopamine/eligibility/weight to :math:`t_{\mathrm{post}} + d`
           b. If :math:`t_{\mathrm{pre}} - t_{\mathrm{post}} > \epsilon`, facilitate
              eligibility trace: :math:`c \leftarrow c + A_+ K_+(t)`

        3. Propagate dopamine/eligibility/weight to :math:`t_{\mathrm{pre}}`
        4. Depress eligibility trace: :math:`c \leftarrow c - A_- K_-(t_{\mathrm{pre}} - d)`
        5. Schedule spike event with updated ``weight`` for delivery at
           :math:`t_{\mathrm{pre}} + \mathrm{delay}`
        6. Update presynaptic trace:
           :math:`K_+ \leftarrow K_+ \exp((t_{\mathrm{last\_update}} - t_{\mathrm{pre}}) / \tau_+) + 1`
        7. Set :math:`t_{\mathrm{last\_update}} = t_{\mathrm{lastspike}} = t_{\mathrm{pre}}`

        The event payload is ``multiplicity * weight``, modulated by dopamine-driven
        weight changes up to the current spike time.

        **Timing:** Uses current simulation time plus one timestep (``t + dt``) as the
        spike timestamp (on-grid timing).

        Examples
        --------
        .. code-block:: python

           >>> import brainpy.state as bp
           >>> import brainunit as u
           >>> # Create postsynaptic neuron and synapse
           >>> post_neuron = bp.LIF(1)
           >>> syn = bp.stdp_dopamine_synapse(
           ...     weight=5.0,
           ...     delay=1.0 * u.ms,
           ...     post=post_neuron,
           ...     volume_transmitter=object()
           ... )
           >>> syn.init_state()
           >>> # Send presynaptic spike
           >>> sent = syn.send(1.0)
           >>> print(sent)
           True
           >>> # Check updated trace
           >>> print(f"Kplus after spike: {syn.Kplus:.3f}")
           Kplus after spike: 1.000

        See Also
        --------
        trigger_update_weight : Propagate state without sending spike event
        update : High-level method combining spike delivery, recording, and sending
        """
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
        """Propagate dopamine, eligibility, and weight to specified trigger time.

        Implements NEST ``stdp_dopamine_synapse::trigger_update_weight``: processes
        postsynaptic spikes, integrates dopamine spike history, updates weight via
        eligibility trace modulation, and advances all state traces to the trigger time.
        This is typically called once per simulation timestep to integrate volume
        transmitter dopamine deliveries.

        Parameters
        ----------
        t_trig_ms : float, array-like, or None, optional
            Target timestamp in milliseconds to which state should be propagated.
            If ``None``, uses current simulation time plus one timestep (``t + dt``).
            Default: ``None``.

        Raises
        ------
        ValueError
            If ``volume_transmitter`` is ``None``.
        ValueError
            If ``t_trig_ms`` is earlier than ``t_last_update`` (backward time propagation
            is not allowed).
        ValueError
            If ``t_trig_ms`` is not scalar or not finite.

        Notes
        -----
        **Update sequence:**

        1. Retrieve postsynaptic spike history in the window
           :math:`(t_{\mathrm{last\_update}} - d,\; t_{\mathrm{trig}} - d]`
           where :math:`d` is dendritic delay.
        2. For each postsynaptic spike :math:`t_{\mathrm{post}}` in that window:

           a. Propagate dopamine/eligibility/weight to :math:`t_{\mathrm{post}} + d`
           b. Facilitate eligibility trace:
              :math:`c \leftarrow c + A_+ K_+(t) \exp((t_{\mathrm{last\_update}} - (t_{\mathrm{post}} + d)) / \tau_+)`

        3. Propagate dopamine/eligibility/weight to :math:`t_{\mathrm{trig}}`
        4. Advance dopamine trace: :math:`n \leftarrow n \exp((t_{\mathrm{last\_dopa}} - t_{\mathrm{trig}}) / \tau_n)`
        5. Advance presynaptic trace:
           :math:`K_+ \leftarrow K_+ \exp((t_{\mathrm{last\_update}} - t_{\mathrm{trig}}) / \tau_+)`
        6. Set :math:`t_{\mathrm{last\_update}} = t_{\mathrm{trig}}`
        7. Reset dopamine spike buffer to :math:`[(t_{\mathrm{trig}}, 0)]`

        Weight integration uses the analytical solution:

        .. math::
           w \leftarrow w - c(t_0) \left(
           \frac{n(t_0)}{\tau_s} \mathrm{expm1}(\tau_s \Delta t)
           - b \tau_c \mathrm{expm1}(\Delta t / \tau_c)
           \right)

        where :math:`\tau_s = (\tau_c + \tau_n) / (\tau_c \tau_n)` and clipped to
        :math:`[W_{\min}, W_{\max}]`.

        **Typical usage:** Called once per timestep in :meth:`update` with
        ``trigger_dopa_update=True`` (default) to synchronize with volume transmitter
        delivery intervals.

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.stdp_dopamine_synapse(
           ...     weight=5.0,
           ...     tau_c=1000.0 * u.ms,
           ...     tau_n=200.0 * u.ms,
           ...     b=0.1,
           ...     volume_transmitter=object()
           ... )
           >>> syn.init_state()
           >>> # Record some eligibility via spike pairing (not shown)
           >>> syn.c = 0.5  # artificial eligibility
           >>> # Record dopamine spike
           >>> syn.record_dopa_spike(2.0, t_spike_ms=10.0)
           >>> # Trigger update to integrate dopamine effect
           >>> syn.trigger_update_weight(t_trig_ms=50.0)
           >>> print(f"Weight after dopamine: {syn.weight:.3f}")
           Weight after dopamine: ...

        See Also
        --------
        send : Send presynaptic spike with plasticity updates
        record_dopa_spike : Record dopamine delivery event
        update : High-level simulation step including trigger_update_weight
        """
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
        """Execute one simulation timestep with spike delivery, plasticity, and dopamine updates.

        This is the primary high-level interface for advancing synapse dynamics. It integrates
        spike event delivery, postsynaptic/dopamine spike recording, presynaptic spike
        processing with STDP updates, and dopamine-modulated weight integration.

        Parameters
        ----------
        pre_spike : float or array-like, optional
            Presynaptic spike multiplicity for the current timestep (typically ``0.0`` or
            ``1.0``). Aggregated with any registered input sources via :meth:`sum_current_inputs`
            and :meth:`sum_delta_inputs`. Default: ``0.0``.
        post_spike : float or array-like, optional
            Postsynaptic spike multiplicity (integer count, will be rounded). Recorded into
            internal postsynaptic history for STDP processing. Default: ``0.0``.
        dopa_spike : float or array-like, optional
            Dopamine delivery multiplicity (arbitrary non-negative float representing
            concentration increment). Recorded into internal dopamine history. Default: ``0.0``.
        post : Dynamics or None, optional
            Postsynaptic receiver for spike events. If ``None``, uses synapse's default
            ``post`` attribute. Default: ``None``.
        receptor_type : int, array-like, or None, optional
            Target receptor port on postsynaptic neuron. If ``None``, uses synapse's
            default ``receptor_type``. Default: ``None``.
        trigger_dopa_update : bool, optional
            If ``True``, calls :meth:`trigger_update_weight` at the end of the timestep
            to integrate dopamine effects (standard NEST behavior). If ``False``, skips
            weight propagation (useful for debugging or custom control). Default: ``True``.

        Returns
        -------
        int
            Number of delayed synaptic events delivered to postsynaptic targets during
            this timestep.

        Raises
        ------
        ValueError
            If ``volume_transmitter`` is ``None``.
        ValueError
            If ``post_spike`` or ``dopa_spike`` cannot be converted to valid non-negative
            scalar counts/multiplicities.

        Notes
        -----
        **Update sequence:**

        1. **Deliver delayed events**: Process all events in the delivery queue scheduled
           for the current simulation step (``t``). Events are sent to their target neurons.
        2. **Record postsynaptic spikes**: If ``post_spike > 0``, record spike(s) at
           timestamp ``t + dt`` into internal postsynaptic history. Updates ``Kminus`` trace.
        3. **Record dopamine spikes**: If ``dopa_spike > 0``, record dopamine delivery
           at timestamp ``t + dt`` into internal dopamine history.
        4. **Process presynaptic spikes**: If ``pre_spike > 0`` (after aggregating input
           sources), call :meth:`send` to:

           - Process postsynaptic history for STDP eligibility updates
           - Integrate dopamine/weight dynamics
           - Schedule delayed synaptic event
           - Update presynaptic trace ``Kplus``

        5. **Trigger dopamine update** (if ``trigger_dopa_update=True``): Call
           :meth:`trigger_update_weight` to propagate all traces and weight to ``t + dt``,
           integrating queued dopamine spikes.

        **Timing convention:** All spikes (pre, post, dopamine) are timestamped at
        ``t + dt`` (on-grid, next simulation step).

        **Typical usage:** Called once per simulation timestep in a training loop.

        Examples
        --------
        Basic simulation loop with pre/post spike pairing:

        .. code-block:: python

           >>> import brainpy.state as bp
           >>> import brainunit as u
           >>> import brainstate as bs
           >>> # Setup
           >>> post_neuron = bp.LIF(1)
           >>> syn = bp.stdp_dopamine_synapse(
           ...     weight=5.0,
           ...     delay=1.0 * u.ms,
           ...     post=post_neuron,
           ...     volume_transmitter=object(),
           ...     A_plus=0.01,
           ...     A_minus=0.012
           ... )
           >>> syn.init_state()
           >>> post_neuron.init_all_states()
           >>> # Simulate causal spike pairing
           >>> with bs.environ.context(dt=0.1 * u.ms):
           ...     delivered = syn.update(pre_spike=1.0, post_spike=0.0)  # pre at t=0.1ms
           ...     delivered = syn.update(pre_spike=0.0, post_spike=1.0)  # post at t=0.2ms
           ...     # Deliver reward
           ...     delivered = syn.update(pre_spike=0.0, post_spike=0.0, dopa_spike=1.0)
           >>> print(f"Final weight: {syn.weight:.4f}")
           Final weight: ...

        Disabling dopamine updates for testing:

        .. code-block:: python

           >>> syn = bp.stdp_dopamine_synapse(
           ...     weight=1.0,
           ...     volume_transmitter=object()
           ... )
           >>> syn.init_state()
           >>> # Update without weight propagation
           >>> delivered = syn.update(
           ...     pre_spike=1.0,
           ...     trigger_dopa_update=False
           ... )
           >>> # Weight remains unchanged (no dopamine integration)
           >>> print(syn.weight)
           1.0

        See Also
        --------
        send : Process presynaptic spike with STDP updates
        trigger_update_weight : Propagate state with dopamine integration
        record_post_spike : Manually record postsynaptic spike
        record_dopa_spike : Manually record dopamine delivery
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
