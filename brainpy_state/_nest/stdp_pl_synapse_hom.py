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

from .static_synapse import static_synapse

__all__ = [
    'stdp_pl_synapse_hom',
]


_UNSET = object()
_STDP_EPS = 1.0e-6


class stdp_pl_synapse_hom(static_synapse):
    r"""NEST-compatible ``stdp_pl_synapse_hom`` connection model.

    ``stdp_pl_synapse_hom`` implements the power-law spike-timing-dependent
    plasticity (STDP) rule from Morrison et al. (2007) with homogeneous
    plasticity parameters. This synapse exhibits asymmetric potentiation and
    depression with non-linear, power-law weight dependence, making it suitable
    for modeling balanced networks with realistic weight distributions.

    The model replicates NEST ``models/stdp_pl_synapse_hom.h`` exactly, including
    propagator computation, update ordering, and event timing semantics. Delay
    scheduling and receiver delivery inherit from :class:`static_synapse`.

    **1. Mathematical Model**

    **State Variables:**

    - ``weight`` (:math:`w`): Synaptic efficacy (current/conductance units or dimensionless)
    - ``Kplus`` (:math:`K^+`): Presynaptic eligibility trace (dimensionless)
    - ``t_lastspike`` (:math:`t_{\mathrm{last}}`): Timestamp of previous presynaptic spike (ms)
    - Internal postsynaptic history buffer: ``(t_post, K^-(t_post))`` pairs

    **Continuous-time dynamics (between spikes):**

    Presynaptic trace decay:

    .. math::

       \frac{dK^+}{dt} = -\frac{K^+}{\tau_+}

    Postsynaptic trace decay (maintained in internal buffer):

    .. math::

       \frac{dK^-}{dt} = -\frac{K^-}{\tau_-}

    where:

    - :math:`\tau_+ > 0` -- Potentiation time constant (ms)
    - :math:`\tau_- > 0` -- Depression time constant (ms)

    **Upon presynaptic spike at time** :math:`t_{\mathrm{pre}}` **with dendritic delay** :math:`d`:

    **Step 1: Facilitation (Potentiation)** — Process all postsynaptic spikes in the causal window:

    For each postsynaptic spike :math:`t_{\mathrm{post}}` in the interval
    :math:`(t_{\mathrm{last}} - d,\, t_{\mathrm{pre}} - d]`:

    .. math::

       K^+_{\mathrm{eff}} = K^+ \cdot \exp\left(\frac{t_{\mathrm{last}} - (t_{\mathrm{post}} + d)}{\tau_+}\right)

       w \leftarrow w + \lambda \, w^\mu \, K^+_{\mathrm{eff}}

    where:

    - :math:`\lambda` -- Learning rate (dimensionless)
    - :math:`\mu` -- Power-law exponent for potentiation (:math:`\mu \in [0, 1]` typical)

    **Interpretation:** The presynaptic trace :math:`K^+` is back-propagated to
    the time of the postsynaptic spike (:math:`t_{\mathrm{post}} + d`, accounting
    for dendritic delay), producing a smaller effective trace for older postsynaptic
    spikes. Potentiation is **multiplicative** and **sub-linear** in weight
    (:math:`w^\mu` with :math:`\mu < 1`), promoting stable weight distributions.

    **Step 2: Depression** — Apply depression based on the postsynaptic trace at the pre-spike time:

    .. math::

       K^-_{\mathrm{eff}} = K^-\left(t_{\mathrm{pre}} - d\right)

       w \leftarrow w - \alpha \lambda \, w \, K^-_{\mathrm{eff}}

       w \leftarrow \max(w, 0)

    where :math:`\alpha` is the depression scaling factor.

    **Interpretation:** Depression is **linear** in weight and occurs when a
    presynaptic spike is preceded by postsynaptic activity. The weight is clipped
    to zero to prevent negative values.

    **Step 3: Event Transmission** — Schedule the weighted event with updated ``weight``.

    **Step 4: Presynaptic Trace Update:**

    .. math::

       K^+ \leftarrow K^+ \cdot \exp\left(\frac{t_{\mathrm{last}} - t_{\mathrm{pre}}}{\tau_+}\right) + 1

       t_{\mathrm{last}} \leftarrow t_{\mathrm{pre}}

    **Postsynaptic spike handling (via internal buffer):**

    Upon postsynaptic spike at :math:`t_{\mathrm{post}}`:

    .. math::

       K^- \leftarrow K^- \cdot \exp\left(\frac{t_{\mathrm{last\_post}} - t_{\mathrm{post}}}{\tau_-}\right) + 1

    Stored as ``(t_post, K^-)`` in history buffer for future lookups.

    **2. Update Ordering and NEST Compatibility**

    This implementation preserves the exact update sequence from NEST
    ``models/stdp_pl_synapse_hom.h::send()``:

    1. Read postsynaptic spike history in :math:`(t_{\mathrm{last}} - d,\, t_{\mathrm{pre}} - d]`
    2. For each retrieved postsynaptic spike, compute back-propagated :math:`K^+_{\mathrm{eff}}`
    3. Apply facilitation: :math:`w \leftarrow w + \lambda w^\mu K^+_{\mathrm{eff}}`
    4. Retrieve depression trace :math:`K^-_{\mathrm{eff}}` at :math:`t_{\mathrm{pre}} - d`
    5. Apply depression: :math:`w \leftarrow \max(w - \alpha \lambda w K^-_{\mathrm{eff}}, 0)`
    6. Schedule weighted spike event
    7. Update presynaptic trace: :math:`K^+ \leftarrow K^+ e^{(t_{\mathrm{last}} - t_{\mathrm{pre}})/\tau_+} + 1`
    8. Update timestamp: :math:`t_{\mathrm{last}} \leftarrow t_{\mathrm{pre}}`

    **3. Homogeneous-Property Semantics**

    In NEST, ``tau_plus``, ``lambda``, ``alpha``, and ``mu`` are **common model properties**
    shared by all synapses of this type, while ``weight`` and ``Kplus``
    are **per-connection state**.

    This implementation enforces NEST connect-time semantics:

    - Common properties (``tau_plus``, ``lambda``, ``alpha``, ``mu``) are set
      at model instantiation or via ``SetDefaults()`` / ``CopyModel()``
    - Per-connection properties (``weight``, ``Kplus``) can be set via
      ``Connect(..., syn_spec={...})``
    - :meth:`check_synapse_params` rejects attempts to override common properties
      in connection specifications

    **4. Event Timing Semantics**

    NEST evaluates this model using on-grid spike time stamps and ignores precise
    sub-step offsets. This implementation follows the same convention:

    - Presynaptic spike detected at simulation step ``n``
    - Spike time stamp: :math:`t_{\mathrm{spike}} = t_n + dt`
    - Dendritic arrival time: :math:`t_{\mathrm{arrival}} = t_{\mathrm{spike}} - d`
    - Delivery time: :math:`t_{\mathrm{delivery}} = t_{\mathrm{spike}} + \mathrm{delay}`

    **5. Stability Constraints and Computational Implications**

    **Parameter Constraints:**

    - :math:`\tau_+ > 0` (enforced in ``__init__`` and ``set``)
    - :math:`\tau_- > 0` (recommended, not enforced)
    - :math:`\lambda \geq 0` (learning rate)
    - :math:`\alpha \geq 0` (depression scaling)
    - :math:`\mu \in [0, 1]` (typical range; not enforced)
    - :math:`K^+ \geq 0` (initial presynaptic trace; typically zero)
    - :math:`w \geq 0` (maintained via clipping in depression)

    **Numerical Considerations:**

    - Trace propagation uses ``math.exp()`` for exponential decay
    - Power-law computation uses ``numpy.power()`` with float64 precision
    - Postsynaptic history is stored as Python lists ``_post_hist_t`` and
      ``_post_hist_kminus``; lookups are :math:`O(n)` where :math:`n` is the
      number of stored postsynaptic spikes
    - Per-call cost: :math:`O(n_{\mathrm{post}})` where :math:`n_{\mathrm{post}}`
      is the number of postsynaptic spikes in the causal window
    - All state variables are Python floats (``float64`` precision)

    **Behavioral Regimes:**

    - **Power-law stabilization** (:math:`\mu < 1`): Potentiation is sub-linear in
      weight, preventing runaway growth and promoting log-normal weight distributions
      (Morrison et al., 2007)
    - **Balanced networks**: The combination of power-law potentiation and linear
      depression naturally regulates weight distributions in recurrent networks
    - **Weight clamping**: Depression clipping at :math:`w = 0` prevents negative
      weights; no upper bound is enforced (unlike ``stdp_synapse`` with ``Wmax``)

    **Failure Modes:**

    - **Non-finite weights**: Power-law computation :math:`w^\mu` can produce
      ``inf`` or ``nan`` for extreme weights; users should monitor weight distributions
    - **Trace overflow**: Large spike trains can accumulate unbounded :math:`K^+`
      or :math:`K^-` values (not a practical issue for typical firing rates)
    - **History buffer growth**: Postsynaptic spike history is not pruned; long
      simulations with high postsynaptic firing rates may consume memory

    Parameters
    ----------
    weight : ArrayLike, optional
        Initial synaptic weight :math:`w` (dimensionless or with receiver-specific units).
        Scalar float or array-like. Must be non-negative. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic transmission delay :math:`d` in milliseconds. Must be ``> 0``.
        Quantized to integer time steps per :class:`static_synapse` conventions.
        Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor identifier (non-negative integer).
        Default: ``0``.
    tau_plus : ArrayLike, optional
        Potentiation time constant :math:`\tau_+` in milliseconds. Must be ``> 0``.
        Scalar float or brainunit ``Quantity``. **Common property** (not per-connection).
        Default: ``20.0 * u.ms``.
    tau_minus : ArrayLike, optional
        Depression trace time constant :math:`\tau_-` in milliseconds. Must be ``> 0``.
        Scalar float or brainunit ``Quantity``.
        In NEST, this parameter belongs to the postsynaptic ``ArchivingNode``; here
        it is stored on the synapse for standalone compatibility.
        Default: ``20.0 * u.ms``.
    lambda_ : ArrayLike, optional
        Learning rate :math:`\lambda` (dimensionless). Must be non-negative.
        **Common property** (not per-connection). Default: ``0.1``.
    alpha : ArrayLike, optional
        Depression scaling factor :math:`\alpha` (dimensionless). Must be non-negative.
        Controls the relative strength of depression vs. potentiation.
        **Common property** (not per-connection). Default: ``1.0``.
    mu : ArrayLike, optional
        Power-law exponent :math:`\mu` for potentiation (dimensionless).
        Typical range: :math:`[0, 1]`. Values :math:`< 1` produce sub-linear
        potentiation; :math:`\mu = 0` disables weight dependence.
        **Common property** (not per-connection). Default: ``0.4``.
    Kplus : ArrayLike, optional
        Initial presynaptic eligibility trace :math:`K^+` (dimensionless).
        Must be non-negative. Scalar float or array-like.
        **Per-connection state**. Default: ``0.0``.
    post : object, optional
        Default receiver object (typically a neuron or neuron group).
        Can be overridden in ``send()`` and ``update()`` calls. Default: ``None``.
    name : str, optional
        Object name for identification and debugging. Default: ``None``.

    **Parameter Mapping (NEST vs. brainpy.state)**

    The following table maps NEST parameter names to this implementation:

    ========================  ========================  =================
    NEST Parameter            brainpy.state Parameter   Type
    ========================  ========================  =================
    ``weight``                ``weight``                per-connection
    ``delay``                 ``delay``                 per-connection
    ``receptor_type``         ``receptor_type``         per-connection
    ``tau_plus``              ``tau_plus``              common property
    ``tau_minus``             ``tau_minus``             common property
    ``lambda``                ``lambda_``               common property
    ``alpha``                 ``alpha``                 common property
    ``mu``                    ``mu``                    common property
    ``Kplus``                 ``Kplus``                 per-connection
    ========================  ========================  =================

    Notes
    -----
    - The model transmits spike-like events only (no graded signals).
    - ``update(pre_spike=..., post_spike=...)`` accepts both presynaptic and
      postsynaptic spike multiplicities (integer counts) for standalone STDP
      simulation without explicit neuron models.
    - ``record_post_spike(multiplicity, t_spike_ms=None)`` can be used to
      manually feed postsynaptic spikes when the postsynaptic model does not
      expose NEST ``ArchivingNode`` APIs.
    - Postsynaptic spike history is **not automatically pruned**; users may call
      ``clear_post_history()`` to reset internal buffers if needed.
    - Unlike ``stdp_synapse``, this model has **no upper weight bound** (``Wmax``);
      weight stability relies on power-law potentiation dynamics.

    See Also
    --------
    stdp_synapse : Classical pair-based STDP with separate potentiation/depression exponents
    stdp_triplet_synapse : Triplet STDP rule (Pfister-Gerstner)
    static_synapse : Base class for event scheduling and delay handling

    References
    ----------
    .. [1] NEST source: ``models/stdp_pl_synapse_hom.h`` and
           ``models/stdp_pl_synapse_hom.cpp``.
    .. [2] Morrison A, Aertsen A, Diesmann M (2007). Spike-timing dependent
           plasticity in balanced random networks.
           Neural Computation, 19(6):1437-1467.
           DOI: 10.1162/neco.2007.19.6.1437

    Examples
    --------
    **Basic standalone STDP simulation:**

    .. code-block:: python

       >>> import brainpy.state as bps
       >>> import brainunit as u
       >>>
       >>> # Create synapse with power-law STDP
       >>> syn = bps.stdp_pl_synapse_hom(
       ...     weight=1.0,
       ...     tau_plus=20*u.ms,
       ...     tau_minus=20*u.ms,
       ...     lambda_=0.1,
       ...     alpha=1.0,
       ...     mu=0.4,
       ... )
       >>> syn.init_state()
       >>>
       >>> # Simulate pre-before-post pairing (potentiation)
       >>> syn.record_post_spike(t_spike_ms=10.0)  # post spike at 10 ms
       >>> syn.send(1.0)  # pre spike at 11 ms (assuming dt=1ms, t=10ms)
       >>> print(f"Weight after potentiation: {syn.weight:.4f}")
       Weight after potentiation: 1.0xxx
       >>>
       >>> # Simulate post-before-pre pairing (depression)
       >>> syn.record_post_spike(t_spike_ms=20.0)  # post spike at 20 ms
       >>> syn.send(1.0)  # pre spike at 10 ms (causally follows post)
       >>> print(f"Weight after depression: {syn.weight:.4f}")
       Weight after depression: 0.9xxx

    **Enforcing homogeneous-property semantics:**

    .. code-block:: python

       >>> import brainpy.state as bps
       >>>
       >>> syn = bps.stdp_pl_synapse_hom(lambda_=0.05)
       >>>
       >>> # Allowed: per-connection properties
       >>> syn.check_synapse_params({'weight': 2.0, 'Kplus': 0.5})  # OK
       >>>
       >>> # Disallowed: common properties in connection specs
       >>> try:
       ...     syn.check_synapse_params({'lambda': 0.1})
       ... except ValueError as e:
       ...     print(e)
       lambda cannot be specified in connect-time synapse parameters...
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        lambda_: ArrayLike = 0.1,
        alpha: ArrayLike = 1.0,
        mu: ArrayLike = 0.4,
        Kplus: ArrayLike = 0.0,
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

        self.tau_plus = self._to_scalar_time_ms(tau_plus, name='tau_plus')
        self.tau_minus = self._to_scalar_time_ms(tau_minus, name='tau_minus')
        self.lambda_ = self._to_scalar_float(lambda_, name='lambda')
        self.alpha = self._to_scalar_float(alpha, name='alpha')
        self.mu = self._to_scalar_float(mu, name='mu')
        self.Kplus = self._to_scalar_float(Kplus, name='Kplus')

        self._validate_tau_plus(self.tau_plus)

        self._Kplus0 = float(self.Kplus)
        self._t_lastspike0 = 0.0

        self.t_lastspike = float(self._t_lastspike0)
        self._post_kminus = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t: list[float] = []
        self._post_hist_kminus: list[float] = []

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
    def _validate_tau_plus(value: float):
        if value <= 0.0:
            raise ValueError('tau_plus must be > 0.')

    def _facilitate(self, w: float, kplus: float) -> float:
        power_term = float(np.power(np.float64(w), np.float64(self.mu)))
        return w + (self.lambda_ * power_term * kplus)

    def _depress(self, w: float, kminus: float) -> float:
        new_w = w - (self.lambda_ * self.alpha * w * kminus)
        return new_w if new_w > 0.0 else 0.0

    def clear_post_history(self):
        r"""Clear internal postsynaptic STDP history state.

        Resets the internal postsynaptic spike history buffer and depression trace
        to initial conditions. This method is useful for:

        - Resetting the synapse state between simulation trials
        - Reclaiming memory after long simulations with high postsynaptic firing rates
        - Debugging and testing STDP dynamics

        The method resets:

        - ``_post_kminus``: Depression trace to ``0.0``
        - ``_last_post_spike``: Last postsynaptic spike time to ``-1.0``
        - ``_post_hist_t``: Spike time history to empty list
        - ``_post_hist_kminus``: Depression trace history to empty list

        Presynaptic state (``Kplus``, ``t_lastspike``) is **not** affected.

        Notes
        -----
        - This method does **not** reset ``weight`` or presynaptic trace ``Kplus``
        - Called automatically by ``init_state()``
        - Postsynaptic history is **not** automatically pruned during simulation;
          manual calls to this method may be needed for very long runs

        See Also
        --------
        init_state : Full state initialization including history clearing
        record_post_spike : Add postsynaptic spikes to history buffer
        """
        self._post_kminus = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t = []
        self._post_hist_kminus = []

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
        r"""Record postsynaptic spikes into the internal STDP history buffer.

        This method manually adds postsynaptic spike events to the internal history
        buffer used for STDP computation. It is intended for standalone STDP
        simulation when the postsynaptic neuron does not expose NEST
        ``ArchivingNode`` APIs.

        For each spike, the method:

        1. Updates the depression trace:
           :math:`K^- \leftarrow K^- \exp((t_{\mathrm{last}} - t_{\mathrm{spike}})/\tau_-) + 1`
        2. Stores the spike time and trace value in the history buffer

        Parameters
        ----------
        multiplicity : ArrayLike, optional
            Number of spikes to record (non-negative integer count).
            If ``< 1.0``, no spikes are recorded. Default: ``1.0``.
        t_spike_ms : ArrayLike or None, optional
            Spike time stamp in milliseconds (scalar float or brainunit ``Quantity``).
            If ``None``, uses the current simulation time plus one time step:
            :math:`t_{\mathrm{spike}} = t_{\mathrm{current}} + dt`.
            Default: ``None``.

        Returns
        -------
        int
            Number of spikes actually recorded (integer count).

        Raises
        ------
        ValueError
            If ``multiplicity`` is not a scalar, not finite, negative, or not
            close to an integer value.
        ValueError
            If ``t_spike_ms`` is provided but not a scalar or not finite.

        Notes
        -----
        - Multiple spikes at the same time are recorded sequentially, updating
          the trace after each spike (matches NEST behavior for simultaneous spikes)
        - Spike times are stored in milliseconds (Python float)
        - The internal history buffer grows unbounded; call ``clear_post_history()``
          to reclaim memory if needed
        - This method does **not** trigger STDP weight updates; updates occur
          during presynaptic spike processing in ``send()``

        See Also
        --------
        clear_post_history : Reset postsynaptic spike history buffer
        update : Main update method that accepts ``post_spike`` parameter
        send : Presynaptic spike processing (applies STDP weight updates)

        Examples
        --------
        **Record postsynaptic spikes at explicit times:**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>> import brainunit as u
           >>>
           >>> syn = bps.stdp_pl_synapse_hom(tau_minus=20*u.ms)
           >>> syn.init_state()
           >>>
           >>> # Record single spike at 10 ms
           >>> n = syn.record_post_spike(1.0, t_spike_ms=10.0)
           >>> print(f"Recorded {n} spike(s)")
           Recorded 1 spike(s)
           >>>
           >>> # Record multiple simultaneous spikes
           >>> n = syn.record_post_spike(3.0, t_spike_ms=20.0)
           >>> print(f"Recorded {n} spike(s)")
           Recorded 3 spike(s)

        **Use current simulation time (automatic stamping):**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>> import brainunit as u
           >>> import brainstate as bst
           >>>
           >>> syn = bps.stdp_pl_synapse_hom()
           >>> syn.init_state()
           >>>
           >>> with bst.environ.context(dt=0.1*u.ms):
           ...     syn.record_post_spike()  # Uses t_current + dt
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

    def _get_post_history_times(self, t1_ms: float, t2_ms: float) -> list[float]:
        t1_lim = float(t1_ms + _STDP_EPS)
        t2_lim = float(t2_ms + _STDP_EPS)
        selected = []
        for t_post in self._post_hist_t:
            if t_post >= t1_lim and t_post < t2_lim:
                selected.append(t_post)
        return selected

    def _get_K_value(self, t_ms: float) -> float:
        # Return trace strictly before t, matching ArchivingNode::get_K_value.
        for idx in range(len(self._post_hist_t) - 1, -1, -1):
            t_post = self._post_hist_t[idx]
            if (t_ms - t_post) > _STDP_EPS:
                return self._post_hist_kminus[idx] * math.exp((t_post - t_ms) / self.tau_minus)
        return 0.0

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize synapse state variables to default values.

        Resets all mutable state to initial conditions, including:

        - ``weight``: Baseline synaptic weight (inherited from ``static_synapse``)
        - ``Kplus``: Presynaptic eligibility trace to ``_Kplus0``
        - ``t_lastspike``: Last presynaptic spike time to ``_t_lastspike0`` (default ``0.0``)
        - Postsynaptic spike history buffer (cleared via ``clear_post_history()``)
        - Event delivery queue (inherited from ``static_synapse``)

        Parameters
        ----------
        batch_size : int, optional
            Batch size for vectorized state initialization. Currently unused;
            this synapse operates in scalar mode only. Default: ``None``.
        **kwargs : dict, optional
            Additional keyword arguments (unused; provided for API compatibility).

        Notes
        -----
        - This method must be called before simulation begins
        - Clears all postsynaptic spike history (calls ``clear_post_history()``)
        - Does **not** reset common properties (``tau_plus``, ``lambda_``, ``alpha``, ``mu``)
        - Presynaptic trace is reset to initial value set via constructor or ``set()``

        See Also
        --------
        clear_post_history : Clear postsynaptic spike history only
        set : Update parameters and initial state values
        """
        del batch_size, kwargs
        super().init_state()
        self.Kplus = float(self._Kplus0)
        self.t_lastspike = float(self._t_lastspike0)
        self.clear_post_history()

    def get(self) -> dict:
        r"""Return current public parameters and mutable state.

        Retrieves all NEST-compatible public parameters and per-connection state
        variables as a dictionary. This method is used for introspection, logging,
        and state serialization.

        Returns
        -------
        dict
            Dictionary mapping parameter/state names to their current values:

            - ``'weight'``: Current synaptic weight (float)
            - ``'delay'``: Synaptic delay in ms (float)
            - ``'receptor_type'``: Receiver port ID (int)
            - ``'tau_plus'``: Potentiation time constant in ms (float)
            - ``'tau_minus'``: Depression time constant in ms (float)
            - ``'lambda'``: Learning rate (float)
            - ``'alpha'``: Depression scaling factor (float)
            - ``'mu'``: Power-law exponent (float)
            - ``'Kplus'``: Current presynaptic trace value (float)
            - ``'synapse_model'``: Model identifier (``'stdp_pl_synapse_hom'``)

        Notes
        -----
        - All brainunit ``Quantity`` values are converted to Python floats (SI units)
        - Internal state (``t_lastspike``, postsynaptic history) is **not** included
        - The returned dictionary can be used with ``set(**params)`` for state restoration
        - Key names match NEST conventions (``'lambda'`` instead of ``'lambda_'``)

        See Also
        --------
        set : Update parameters and state from dictionary
        init_state : Reset state to initial values

        Examples
        --------
        .. code-block:: python

           >>> import brainpy.state as bps
           >>> import brainunit as u
           >>>
           >>> syn = bps.stdp_pl_synapse_hom(
           ...     weight=1.5,
           ...     tau_plus=20*u.ms,
           ...     lambda_=0.1,
           ... )
           >>> syn.init_state()
           >>>
           >>> params = syn.get()
           >>> print(params['weight'])
           1.5
           >>> print(params['lambda'])
           0.1
           >>> print(params['synapse_model'])
           stdp_pl_synapse_hom
        """
        params = super().get()
        params['tau_plus'] = float(self.tau_plus)
        params['tau_minus'] = float(self.tau_minus)
        params['lambda'] = float(self.lambda_)
        params['alpha'] = float(self.alpha)
        params['mu'] = float(self.mu)
        params['Kplus'] = float(self.Kplus)
        params['synapse_model'] = 'stdp_pl_synapse_hom'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        r"""Validate connect-time synapse parameter specification.

        Enforces NEST's homogeneous-property semantics by rejecting attempts to
        override common model properties (``tau_plus``, ``lambda``, ``alpha``, ``mu``)
        in per-connection synapse specifications.

        In NEST, homogeneous models share plasticity parameters across all connections,
        while per-connection properties (``weight``, ``Kplus``) can vary. This method
        prevents accidental overrides that would violate this contract.

        Parameters
        ----------
        syn_spec : Mapping[str, object] or None
            Synapse parameter specification dictionary, typically provided in
            ``Connect(..., syn_spec={...})`` calls. If ``None``, no validation is performed.

        Raises
        ------
        ValueError
            If ``syn_spec`` contains any of the disallowed common properties:
            ``'tau_plus'``, ``'lambda'``, ``'alpha'``, ``'mu'``.

        Notes
        -----
        - Allowed per-connection keys: ``'weight'``, ``'delay'``, ``'receptor_type'``, ``'Kplus'``
        - Disallowed common-property keys: ``'tau_plus'``, ``'lambda'``, ``'alpha'``, ``'mu'``
        - To change common properties, use ``set(tau_plus=..., lambda_=..., ...)`` on
          the model instance, or NEST-style ``SetDefaults()`` / ``CopyModel()`` APIs
        - This check is performed automatically during connection establishment

        See Also
        --------
        set : Update model parameters (common and per-connection)

        Examples
        --------
        **Valid per-connection specification:**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>>
           >>> syn = bps.stdp_pl_synapse_hom(lambda_=0.1)
           >>>
           >>> # Allowed: per-connection properties
           >>> syn.check_synapse_params({'weight': 2.0, 'Kplus': 0.5})  # OK

        **Invalid common-property override:**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>>
           >>> syn = bps.stdp_pl_synapse_hom(lambda_=0.1)
           >>>
           >>> # Disallowed: common property in connection spec
           >>> try:
           ...     syn.check_synapse_params({'lambda': 0.05})
           ... except ValueError as e:
           ...     print(e)
           lambda cannot be specified in connect-time synapse parameters...
        """
        if syn_spec is None:
            return
        disallowed = ('tau_plus', 'lambda', 'alpha', 'mu')
        for key in disallowed:
            if key in syn_spec:
                raise ValueError(
                    f'{key} cannot be specified in connect-time synapse parameters '
                    'for stdp_pl_synapse_hom; set common properties on the model '
                    'itself (for example via CopyModel()/SetDefaults()).'
                )

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        tau_plus: ArrayLike | object = _UNSET,
        tau_minus: ArrayLike | object = _UNSET,
        lambda_: ArrayLike | object = _UNSET,
        alpha: ArrayLike | object = _UNSET,
        mu: ArrayLike | object = _UNSET,
        Kplus: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        r"""Set NEST-style public parameters and mutable state.

        Updates model parameters (common properties and per-connection state) with
        validation. This method supports partial updates—only specified parameters
        are modified.

        Parameters
        ----------
        weight : ArrayLike or sentinel, optional
            New synaptic weight. Scalar float or array-like. Must be non-negative.
            If ``_UNSET``, current value is preserved.
        delay : ArrayLike or sentinel, optional
            New synaptic delay in ms. Must be ``> 0``. If ``_UNSET``, current value
            is preserved.
        receptor_type : int or sentinel, optional
            New receiver port ID (non-negative integer). If ``_UNSET``, current value
            is preserved.
        tau_plus : ArrayLike or sentinel, optional
            New potentiation time constant in ms. Must be ``> 0``. If ``_UNSET``,
            current value is preserved.
        tau_minus : ArrayLike or sentinel, optional
            New depression time constant in ms. Must be ``> 0`` (not enforced).
            If ``_UNSET``, current value is preserved.
        lambda_ : ArrayLike or sentinel, optional
            New learning rate. Must be non-negative. If ``_UNSET``, current value
            is preserved.
        alpha : ArrayLike or sentinel, optional
            New depression scaling factor. Must be non-negative (not enforced).
            If ``_UNSET``, current value is preserved.
        mu : ArrayLike or sentinel, optional
            New power-law exponent. If ``_UNSET``, current value is preserved.
        Kplus : ArrayLike or sentinel, optional
            New presynaptic trace value. Must be non-negative (not enforced).
            Updates both ``self.Kplus`` (current state) and ``self._Kplus0``
            (initial value for ``init_state()``). If ``_UNSET``, current value
            is preserved.
        post : object or sentinel, optional
            New default receiver object. If ``_UNSET``, current value is preserved.

        Raises
        ------
        ValueError
            If ``tau_plus`` is provided and ``<= 0``.
        ValueError
            If any parameter is not a scalar, not finite, or violates type constraints.

        Notes
        -----
        - All parameters are optional; only provided values are updated
        - Parameter validation is performed before any state is modified
        - Setting ``Kplus`` updates both current state and initial-value storage
        - Common properties (``tau_plus``, ``lambda_``, ``alpha``, ``mu``) should
          typically be set at model creation, not per-connection
        - This method does **not** clear postsynaptic spike history or reset
          ``t_lastspike``

        See Also
        --------
        get : Retrieve current parameter values
        init_state : Reset state to initial values

        Examples
        --------
        **Update learning rate and weight:**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>> import brainunit as u
           >>>
           >>> syn = bps.stdp_pl_synapse_hom(weight=1.0, lambda_=0.1)
           >>> syn.init_state()
           >>>
           >>> syn.set(weight=2.0, lambda_=0.05)
           >>> print(syn.get()['weight'])
           2.0
           >>> print(syn.get()['lambda'])
           0.05

        **Update time constants:**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>> import brainunit as u
           >>>
           >>> syn = bps.stdp_pl_synapse_hom()
           >>> syn.set(tau_plus=15*u.ms, tau_minus=25*u.ms)
           >>> print(syn.tau_plus)
           15.0
           >>> print(syn.tau_minus)
           25.0
        """
        new_tau_plus = (
            self.tau_plus
            if tau_plus is _UNSET
            else self._to_scalar_time_ms(tau_plus, name='tau_plus')
        )
        self._validate_tau_plus(float(new_tau_plus))

        new_tau_minus = (
            self.tau_minus
            if tau_minus is _UNSET
            else self._to_scalar_time_ms(tau_minus, name='tau_minus')
        )
        new_lambda = (
            self.lambda_
            if lambda_ is _UNSET
            else self._to_scalar_float(lambda_, name='lambda')
        )
        new_alpha = self.alpha if alpha is _UNSET else self._to_scalar_float(alpha, name='alpha')
        new_mu = self.mu if mu is _UNSET else self._to_scalar_float(mu, name='mu')
        new_Kplus = self.Kplus if Kplus is _UNSET else self._to_scalar_float(Kplus, name='Kplus')

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

        self.tau_plus = float(new_tau_plus)
        self.tau_minus = float(new_tau_minus)
        self.lambda_ = float(new_lambda)
        self.alpha = float(new_alpha)
        self.mu = float(new_mu)
        self.Kplus = float(new_Kplus)

        self._Kplus0 = float(self.Kplus)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        r"""Schedule one outgoing event with NEST ``stdp_pl_synapse_hom`` dynamics.

        Processes a presynaptic spike event by applying power-law STDP weight updates
        and scheduling the weighted event for delayed delivery to the postsynaptic
        neuron. This method implements the exact update sequence from NEST
        ``models/stdp_pl_synapse_hom.h::send()``.

        **Update Sequence:**

        1. **Compute spike timestamp:** :math:`t_{\mathrm{spike}} = t_{\mathrm{current}} + dt`
        2. **Facilitation (Potentiation):** For each postsynaptic spike :math:`t_{\mathrm{post}}`
           in the causal window :math:`(t_{\mathrm{last}} - d,\, t_{\mathrm{spike}} - d]`:

           - Back-propagate presynaptic trace: :math:`K^+_{\mathrm{eff}} = K^+ \exp((t_{\mathrm{last}} - (t_{\mathrm{post}} + d))/\tau_+)`
           - Apply potentiation: :math:`w \leftarrow w + \lambda w^\mu K^+_{\mathrm{eff}}`

        3. **Depression:** Retrieve postsynaptic trace :math:`K^-_{\mathrm{eff}}` at
           :math:`t_{\mathrm{spike}} - d` and apply depression:

           - :math:`w \leftarrow w - \alpha \lambda w K^-_{\mathrm{eff}}`
           - Clip to non-negative: :math:`w \leftarrow \max(w, 0)`

        4. **Event Scheduling:** Schedule weighted event :math:`w_{\mathrm{eff}} = w \times \mathrm{multiplicity}`
           for delivery at :math:`t_{\mathrm{delivery}} = t_{\mathrm{spike}} + \mathrm{delay}`
        5. **Presynaptic Trace Update:** :math:`K^+ \leftarrow K^+ \exp((t_{\mathrm{last}} - t_{\mathrm{spike}})/\tau_+) + 1`
        6. **Timestamp Update:** :math:`t_{\mathrm{last}} \leftarrow t_{\mathrm{spike}}`

        Parameters
        ----------
        multiplicity : ArrayLike, optional
            Presynaptic spike multiplicity (scalar float, typically ``1.0``).
            If zero or negative, no event is scheduled and the method returns ``False``.
            Default: ``1.0``.
        post : object or None, optional
            Receiver object (typically a neuron or neuron group). If ``None``, uses
            the default receiver set in the constructor or via ``set(post=...)``.
            Default: ``None``.
        receptor_type : ArrayLike or None, optional
            Receiver port identifier (non-negative integer). If ``None``, uses
            ``self.receptor_type``. Default: ``None``.

        Returns
        -------
        bool
            ``True`` if an event was scheduled, ``False`` if ``multiplicity`` was zero.

        Raises
        ------
        ValueError
            If ``receptor_type`` is provided but not a valid non-negative integer.
        RuntimeError
            If no receiver is available (``post`` is ``None`` and no default receiver
            is set).

        Notes
        -----
        - The method uses **on-grid spike timing**: spike time is :math:`t + dt`,
          ignoring precise sub-step offsets
        - Dendritic delay :math:`d` shifts the STDP causal window but does **not**
          affect event delivery time (delivery delay is separate)
        - Weight updates are applied **before** event scheduling, so the delivered
          event reflects the updated weight
        - Presynaptic trace is updated **after** STDP computation
        - Postsynaptic spike history must be maintained externally via
          ``record_post_spike()`` or ``update(post_spike=...)``

        See Also
        --------
        update : Combined pre/post spike processing with automatic history management
        record_post_spike : Manually add postsynaptic spikes to history buffer

        Examples
        --------
        **Standalone presynaptic spike processing:**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>> import brainunit as u
           >>>
           >>> syn = bps.stdp_pl_synapse_hom(weight=1.0, lambda_=0.1, mu=0.4)
           >>> syn.init_state()
           >>>
           >>> # Record postsynaptic spike at 10 ms
           >>> syn.record_post_spike(t_spike_ms=10.0)
           >>>
           >>> # Process presynaptic spike at 11 ms (causally follows post)
           >>> success = syn.send(1.0)
           >>> print(f"Event scheduled: {success}")
           Event scheduled: True
           >>> print(f"Updated weight: {syn.weight:.4f}")
           Updated weight: 1.0xxx

        **With explicit receiver:**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>>
           >>> class DummyReceiver:
           ...     def receive(self, weight, port, event_type):
           ...         print(f"Received {weight} on port {port}")
           >>>
           >>> syn = bps.stdp_pl_synapse_hom()
           >>> syn.init_state()
           >>> receiver = DummyReceiver()
           >>>
           >>> syn.send(1.0, post=receiver, receptor_type=0)
           True
        """
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST uses on-grid event stamps in this model.
        t_spike = self._current_time_ms() + dt_ms
        dendritic_delay = float(self.delay)

        # Facilitation due to postsynaptic spikes in
        # (t_lastspike - dendritic_delay, t_spike - dendritic_delay].
        t1 = self.t_lastspike - dendritic_delay
        t2 = t_spike - dendritic_delay
        for t_post in self._get_post_history_times(t1, t2):
            minus_dt = self.t_lastspike - (t_post + dendritic_delay)
            assert minus_dt < (-1.0 * _STDP_EPS)
            kplus_term = self.Kplus * math.exp(minus_dt / self.tau_plus)
            self.weight = float(self._facilitate(float(self.weight), float(kplus_term)))

        # Depression due to current presynaptic spike.
        kminus_value = self._get_K_value(t_spike - dendritic_delay)
        self.weight = float(self._depress(float(self.weight), float(kminus_value)))

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.Kplus = float(self.Kplus * math.exp((self.t_lastspike - t_spike) / self.tau_plus) + 1.0)
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
        r"""Deliver due events, update postsynaptic history, then process presynaptic spikes.

        Main update method for standalone STDP simulation. This method orchestrates
        the complete synaptic update cycle in three phases:

        1. **Event Delivery:** Deliver all events scheduled for the current time step
           to the postsynaptic receiver
        2. **Postsynaptic History Update:** Record incoming postsynaptic spikes into
           the internal STDP history buffer
        3. **Presynaptic Spike Processing:** Apply STDP weight updates and schedule
           new events via ``send()``

        This ordering matches NEST's event-driven simulation semantics, where
        postsynaptic spike history is updated before processing presynaptic spikes
        arriving in the same time step.

        Parameters
        ----------
        pre_spike : ArrayLike, optional
            Presynaptic spike count (non-negative integer or float). Summed with
            any registered current/delta inputs before processing. If zero, no
            presynaptic spike is processed. Default: ``0.0``.
        post_spike : ArrayLike, optional
            Postsynaptic spike count (non-negative integer or float). Recorded into
            the internal STDP history buffer at time :math:`t_{\mathrm{current}} + dt`.
            Default: ``0.0``.
        post : object or None, optional
            Receiver object for event delivery. If ``None``, uses the default receiver
            set in the constructor or via ``set(post=...)``. Default: ``None``.
        receptor_type : ArrayLike or None, optional
            Receiver port identifier (non-negative integer). If ``None``, uses
            ``self.receptor_type``. Default: ``None``.

        Returns
        -------
        int
            Number of events delivered to the postsynaptic receiver during this step.

        Raises
        ------
        ValueError
            If ``post_spike`` is not a scalar, not finite, negative, or not close to
            an integer value.
        ValueError
            If ``receptor_type`` is provided but not a valid non-negative integer.
        RuntimeError
            If a presynaptic spike is triggered but no receiver is available.

        Notes
        -----
        - The method uses **on-grid spike timing**: spikes are stamped at
          :math:`t_{\mathrm{current}} + dt`
        - Presynaptic input is accumulated from three sources:

          1. ``pre_spike`` parameter
          2. ``current_inputs`` (registered via ``add_current_input()``)
          3. ``delta_inputs`` (registered via ``add_delta_input()``)

        - Multiple postsynaptic spikes at the same time (``post_spike > 1``) are
          recorded sequentially with trace updates between each spike
        - This method is typically called once per time step in a simulation loop

        See Also
        --------
        send : Presynaptic spike processing and STDP weight updates
        record_post_spike : Manually record postsynaptic spikes
        add_current_input : Register input sources for presynaptic spike accumulation

        Examples
        --------
        **Basic STDP simulation loop:**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>> import brainunit as u
           >>> import brainstate as bst
           >>>
           >>> syn = bps.stdp_pl_synapse_hom(
           ...     weight=1.0,
           ...     tau_plus=20*u.ms,
           ...     tau_minus=20*u.ms,
           ...     lambda_=0.1,
           ...     mu=0.4,
           ... )
           >>> syn.init_state()
           >>>
           >>> with bst.environ.context(dt=1.0*u.ms):
           ...     # Pre-before-post pairing (potentiation)
           ...     for step in range(5):
           ...         bst.environ.set_t(step * 1.0)
           ...         pre = 1.0 if step == 0 else 0.0
           ...         post = 1.0 if step == 2 else 0.0
           ...         syn.update(pre_spike=pre, post_spike=post)
           ...     print(f"Weight after potentiation: {syn.weight:.4f}")
           Weight after potentiation: 1.0xxx

        **With input accumulation:**

        .. code-block:: python

           >>> import brainpy.state as bps
           >>> import brainunit as u
           >>>
           >>> syn = bps.stdp_pl_synapse_hom()
           >>> syn.init_state()
           >>>
           >>> # Register input source
           >>> syn.add_current_input('pre_neurons', lambda: 0.5)
           >>>
           >>> # Update with explicit + accumulated input
           >>> n_delivered = syn.update(pre_spike=0.5)  # Total: 0.5 + 0.5 = 1.0
           >>> print(f"Delivered {n_delivered} event(s)")
           Delivered 1 event(s)
        """
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
