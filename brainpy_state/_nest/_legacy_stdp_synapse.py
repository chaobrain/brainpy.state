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
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike

from ._legacy_imperative import ImperativeSynapseBase

__all__ = [
    'stdp_synapse',
]

_UNSET = object()
_STDP_EPS = 1.0e-6


class stdp_synapse(ImperativeSynapseBase):
    r"""NEST-compatible ``stdp_synapse`` connection model.

    ``stdp_synapse`` implements pair-based spike-timing dependent plasticity (STDP)
    following Guetig et al. (2003) and the NEST reference implementation from
    ``models/stdp_synapse.h``. The model supports asymmetric Hebbian learning with
    configurable weight-dependent potentiation and depression exponents.

    The synapse maintains three dynamic state variables per connection:

    - ``weight``: current synaptic efficacy (plastic, updated on each presynaptic spike)
    - ``Kplus``: presynaptic eligibility trace (exponentially decays with time constant ``tau_plus``)
    - ``t_lastspike``: timestamp of the most recent presynaptic spike

    Postsynaptic spike history is stored internally with time constant ``tau_minus``.
    In NEST, ``tau_minus`` is a postsynaptic neuron parameter (``ArchivingNode``);
    here it is stored on the synapse for standalone compatibility, enabling STDP
    simulation without requiring postsynaptic neurons to implement archiving APIs.

    **1. Mathematical Model**

    State Variables
    ---------------

    - ``w``: Synaptic weight (plastic, bounded to :math:`[0, W_{\max}]` or :math:`[W_{\max}, 0]`)
    - ``K^+``: Presynaptic eligibility trace (decays with :math:`\tau_+`)
    - ``K^-``: Postsynaptic eligibility trace (decays with :math:`\tau_-`)

    **Continuous-time dynamics (between spikes):**

    .. math::

       \frac{dK^+}{dt} = -\frac{K^+}{\tau_+}

       \frac{dK^-}{dt} = -\frac{dK^-}{\tau_-}

    **Upon presynaptic spike at time** :math:`t_{\text{pre}}`:

    Let :math:`d` denote the dendritic (synaptic) delay. The NEST ``stdp_synapse::send``
    method performs the following sequence:

    **Step 1: Facilitation (potentiation) from past postsynaptic spikes**

    For each postsynaptic spike :math:`t_{\text{post}}` in the window
    :math:`(t_{\text{last}} - d,\, t_{\text{pre}} - d]`, where
    :math:`t_{\text{last}}` is the timestamp of the previous presynaptic spike:

    .. math::

       \Delta t = (t_{\text{post}} + d) - t_{\text{last}}

       K^+_{\text{eff}} = K^+ \cdot e^{(t_{\text{last}} - (t_{\text{post}} + d)) / \tau_+}

       \hat{w} \leftarrow \hat{w} + \lambda (1 - \hat{w})^{\mu_+} K^+_{\text{eff}}

    where :math:`\hat{w} = w / W_{\max}` is the normalized weight.

    **Step 2: Depression from current presynaptic spike**

    Retrieve postsynaptic trace :math:`K^-` at time :math:`t_{\text{pre}} - d`:

    .. math::

       K^-_{\text{eff}} = K^-(t_{\text{pre}} - d)

       \hat{w} \leftarrow \hat{w} - \alpha \lambda \hat{w}^{\mu_-} K^-_{\text{eff}}

    **Step 3: Deliver spike event**

    Send event with updated weight ``w`` to the postsynaptic receiver.

    **Step 4: Update presynaptic trace**

    .. math::

       K^+ \leftarrow K^+ \cdot e^{(t_{\text{last}} - t_{\text{pre}}) / \tau_+} + 1

       t_{\text{last}} \leftarrow t_{\text{pre}}

    **Upon postsynaptic spike at time** :math:`t_{\text{post}}`:

    .. math::

       K^- \\leftarrow K^- \\cdot e^{(t_{\\text{last\\_post}} - t_{\\text{post}}) / \\tau_-} + 1

       t_{\\text{last\\_post}} \\leftarrow t_{\\text{post}}

    **Weight Update Functions:**

    Potentiation (post-before-pre, :math:`\Delta t > 0`):

    .. math::

       \hat{w} \leftarrow \hat{w} + \lambda (1 - \hat{w})^{\mu_+} K^+

    Depression (pre-before-post, :math:`\Delta t < 0`):

    .. math::

       \hat{w} \leftarrow \hat{w} - \alpha \lambda \hat{w}^{\mu_-} K^-

    Final weight is clipped to :math:`[0, W_{\max}]` for positive weights, or
    :math:`[W_{\max}, 0]` for negative weights (inhibitory synapses).

    **2. Update Ordering and NEST Compatibility**

    This implementation replicates the exact update sequence from NEST
    ``models/stdp_synapse.h::send()``:

    1. Query postsynaptic spike history in window :math:`(t_{\text{last}} - d,\, t_{\text{pre}} - d]`
    2. Apply facilitation for each retrieved postsynaptic spike
    3. Compute postsynaptic trace :math:`K^-` at :math:`t_{\text{pre}} - d`
    4. Apply depression based on :math:`K^-`
    5. Schedule weighted spike event for delivery after delay :math:`d`
    6. Update presynaptic trace :math:`K^+` and timestamp :math:`t_{\text{last}}`

    **3. Event Timing Semantics**

    As in NEST, this model uses **on-grid spike timestamps** and ignores precise
    sub-step offsets. Spike times are discretized to simulation time steps:

    - Presynaptic spike detected at step ``n`` → stamped at :math:`t_{\text{spike}} = t + dt`
    - Postsynaptic spike recorded at step ``n`` → stamped at :math:`t_{\text{spike}} = t + dt`
    - Inter-spike intervals computed from discrete timestamps

    This differs from continuous-time STDP but matches NEST's default behavior.

    **4. Stability Constraints and Computational Implications**

    **Parameter Constraints:**

    - :math:`\tau_+ > 0`, :math:`\tau_- > 0` (time constants must be positive)
    - :math:`\lambda > 0` (learning rate; negative values would invert plasticity)
    - :math:`\alpha \geq 0` (depression scaling; typically 1.0)
    - :math:`\mu_+ \geq 0`, :math:`\mu_- \geq 0` (exponents; 1.0 for linear, 0.0 for additive)
    - :math:`W_{\max} \neq 0` and :math:`\text{sign}(w) = \text{sign}(W_{\max})`
    - :math:`K^+ \geq 0` (trace must be non-negative)

    Numerical Considerations
    ------------------------

    - All state variables stored as Python ``float`` (``float64`` precision)
    - Exponential decays computed using ``math.exp()`` for numerical stability
    - Power functions use ``math.pow()`` (may degrade for large exponents)
    - Per-spike cost: :math:`O(N_{\text{post}})` where :math:`N_{\text{post}}` is
      the number of postsynaptic spikes in the facilitation window

    **Behavioral Regimes:**

    - **Symmetric STDP** (:math:`\alpha = 1`, :math:`\mu_+ = \mu_- = 1`):
      Classical pair-based rule (Song et al., 2000)
    - **Additive STDP** (:math:`\mu_+ = \mu_- = 0`):
      Weight-independent updates (van Rossum et al., 2000)
    - **Multiplicative STDP** (:math:`\mu_+ = \mu_- = 1`):
      Soft bounds stabilize weight distributions (Guetig et al., 2003)
    - **Asymmetric depression** (:math:`\alpha > 1`):
      Stronger depression relative to potentiation

    Failure Modes
    -------------

    - ``weight`` and ``Wmax`` must have the same sign; otherwise ``ValueError`` on init or set
    - ``Kplus`` must be non-negative; otherwise ``ValueError`` on init or set
    - Postsynaptic spike history grows unbounded if not cleared; use
      ``clear_post_history()`` periodically for long simulations

    Parameters
    ----------
    weight : float, array-like, or Quantity, optional
        Initial synaptic weight :math:`w`. Scalar float, dimensionless or with
        receiver-specific units (e.g., pA, nS). Must have the same sign as ``Wmax``.
        Default: ``1.0`` (dimensionless).
    delay : float, array-like, or Quantity, optional
        Synaptic transmission delay :math:`d` in milliseconds. Must be ``> 0``.
        Quantized to integer time steps per :class:`static_synapse` conventions.
        Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receptor port identifier on the postsynaptic neuron. Non-negative integer
        specifying which input channel receives the event. Default: ``0``.
    tau_plus : float, array-like, or Quantity, optional
        Presynaptic trace time constant :math:`\tau_+` in milliseconds. Must be ``> 0``.
        Controls the width of the potentiation (post-before-pre) window.
        Default: ``20.0 * u.ms``.
    tau_minus : float, array-like, or Quantity, optional
        Postsynaptic trace time constant :math:`\tau_-` in milliseconds. Must be ``> 0``.
        In NEST, this is a postsynaptic neuron parameter; here it is stored on the
        synapse for standalone compatibility. Controls the width of the depression
        (pre-before-post) window. Default: ``20.0 * u.ms``.
    lambda_ : float, array-like, or Quantity, optional
        Learning rate parameter :math:`\lambda` (dimensionless). Scales both
        potentiation and depression updates. Typical values: 0.001–0.1.
        Default: ``0.01``.
    alpha : float, array-like, or Quantity, optional
        Asymmetry parameter :math:`\alpha` (dimensionless). Scales depression
        relative to potentiation. :math:`\alpha = 1.0` yields symmetric STDP;
        :math:`\alpha > 1.0` strengthens depression. Default: ``1.0``.
    mu_plus : float, array-like, or Quantity, optional
        Potentiation exponent :math:`\mu_+` (dimensionless). Controls weight
        dependence of potentiation. :math:`\mu_+ = 0`: additive; :math:`\mu_+ = 1`:
        multiplicative (soft upper bound). Default: ``1.0``.
    mu_minus : float, array-like, or Quantity, optional
        Depression exponent :math:`\mu_-` (dimensionless). Controls weight
        dependence of depression. :math:`\mu_- = 0`: additive; :math:`\mu_- = 1`:
        multiplicative (soft lower bound). Default: ``1.0``.
    Wmax : float, array-like, or Quantity, optional
        Maximum weight bound :math:`W_{\max}` (same units as ``weight``). Weights are
        clipped to :math:`[0, W_{\max}]` for excitatory synapses or :math:`[W_{\max}, 0]`
        for inhibitory synapses. Must have the same sign as ``weight``.
        Default: ``100.0`` (dimensionless).
    Kplus : float, array-like, or Quantity, optional
        Initial presynaptic trace value :math:`K^+` (dimensionless). Must be
        non-negative. Typically initialized to ``0.0`` (no presynaptic history).
        Default: ``0.0``.
    post : Dynamics, optional
        Default postsynaptic receiver object. If provided, :meth:`send` and
        :meth:`update` will target this receiver unless overridden. Must implement
        either ``add_delta_input`` or ``add_current_input`` methods.
        Default: ``None`` (must provide receiver explicitly in method calls).
    name : str, optional
        Unique identifier for this synapse instance. Default: auto-generated.

    Parameter Mapping
    -----------------

    NEST ``stdp_synapse`` parameters map to this implementation as follows:

    ==================  ====================  ========================================
    NEST Parameter      brainpy.state Param   Notes
    ==================  ====================  ========================================
    ``weight``          ``weight``            Plastic, updated on each pre-spike
    ``delay``           ``delay``             Converted to ms, discretized to steps
    ``receptor_type``   ``receptor_type``     Integer ≥ 0
    ``tau_plus``        ``tau_plus``          Pre-synaptic trace decay (ms)
    ``tau_minus``       (neuron param)        Here: synapse param ``tau_minus`` (ms)
    ``lambda``          ``lambda_``           Learning rate (underscore to avoid keyword)
    ``alpha``           ``alpha``             Depression asymmetry factor
    ``mu_plus``         ``mu_plus``           Potentiation exponent
    ``mu_minus``        ``mu_minus``          Depression exponent
    ``Wmax``            ``Wmax``              Weight upper bound (or lower for inhib.)
    ``Kplus``           ``Kplus``             Pre-synaptic trace state variable
    ==================  ====================  ========================================

    Attributes
    ----------
    weight : float
        Current synaptic weight (plastic, updated during simulation).
    Kplus : float
        Current presynaptic trace value.
    t_lastspike : float
        Timestamp (ms) of the most recent presynaptic spike.
    tau_plus : float
        Presynaptic trace time constant (ms).
    tau_minus : float
        Postsynaptic trace time constant (ms).
    lambda_ : float
        Learning rate.
    alpha : float
        Depression asymmetry factor.
    mu_plus : float
        Potentiation exponent.
    mu_minus : float
        Depression exponent.
    Wmax : float
        Maximum weight bound.

    See Also
    --------
    static_synapse : Base class for non-plastic synapses
    tsodyks_synapse : Short-term plasticity (depression/facilitation)
    stdp_synapse_hom : Homogeneous-weight variant with shared weight across connections

    Notes
    -----
    - The model transmits spike-like events only (``event_type='spike'``).
    - ``update(pre_spike=..., post_spike=...)`` accepts both presynaptic and
      postsynaptic spike multiplicities for standalone STDP simulation.
    - ``record_post_spike(...)`` can be used to manually feed postsynaptic spikes
      when the postsynaptic model does not expose NEST archiving APIs.
    - Postsynaptic spike history grows unbounded; call ``clear_post_history()``
      periodically in long simulations to prevent memory issues.

    References
    ----------
    .. [1] NEST source code: ``models/stdp_synapse.h`` and ``models/stdp_synapse.cpp``.
           https://github.com/nest/nest-simulator
    .. [2] Guetig R, Aharonov R, Rotter S, Sompolinsky H (2003).
           Learning input correlations through nonlinear temporally asymmetric
           Hebbian plasticity. *Journal of Neuroscience*, 23(9):3697-3714.
           DOI: `10.1523/JNEUROSCI.23-09-03697.2003 <https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003>`_
    .. [3] Song S, Miller KD, Abbott LF (2000). Competitive Hebbian learning
           through spike-timing-dependent synaptic plasticity.
           *Nature Neuroscience*, 3(9):919-926.
           DOI: `10.1038/78829 <https://doi.org/10.1038/78829>`_
    .. [4] van Rossum MCW, Bi G-Q, Turrigiano GG (2000). Stable Hebbian learning
           from spike timing-dependent plasticity.
           *Journal of Neuroscience*, 20(23):8812-8821.
           DOI: `10.1523/JNEUROSCI.20-23-08812.2000 <https://doi.org/10.1523/JNEUROSCI.20-23-08812.2000>`_

    Examples
    --------
    **Basic STDP synapse with default parameters:**

    .. code-block:: python

       >>> from brainpy import state as bst
       >>> import saiunit as u
       >>> syn = bst.stdp_synapse(weight=0.5, delay=1.0 * u.ms)
       >>> syn.get()
       {'weight': 0.5, 'delay': 1.0, 'receptor_type': 0, 'tau_plus': 20.0,
        'tau_minus': 20.0, 'lambda': 0.01, 'alpha': 1.0, 'mu_plus': 1.0,
        'mu_minus': 1.0, 'Wmax': 100.0, 'Kplus': 0.0, 'synapse_model': 'stdp_synapse'}

    **Asymmetric STDP (stronger depression):**

    .. code-block:: python

       >>> syn = bst.stdp_synapse(
       ...     weight=1.0,
       ...     tau_plus=16.8 * u.ms,
       ...     tau_minus=33.7 * u.ms,
       ...     lambda_=0.005,
       ...     alpha=1.05,  # 5% stronger depression
       ...     Wmax=2.0,
       ... )

    **Additive STDP (weight-independent updates):**

    .. code-block:: python

       >>> syn = bst.stdp_synapse(
       ...     weight=0.5,
       ...     mu_plus=0.0,  # additive potentiation
       ...     mu_minus=0.0,  # additive depression
       ...     lambda_=0.001,
       ...     Wmax=1.0,
       ... )

    **Manual postsynaptic spike recording:**

    .. code-block:: python

       >>> import brainstate
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     syn = bst.stdp_synapse(weight=1.0)
       ...     syn.init_state()
       ...     # Simulate postsynaptic spike at t=5.0 ms
       ...     syn.record_post_spike(multiplicity=1, t_spike_ms=5.0)
       ...     # Simulate presynaptic spike at t=10.0 ms (after post-spike)
       ...     # This should potentiate the weight
       ...     syn.send(multiplicity=1)  # uses on-grid stamp t + dt
       ...     print(f"Updated weight: {syn.weight:.6f}")  # > 1.0 (potentiated)
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        lambda_: ArrayLike = 0.01,
        alpha: ArrayLike = 1.0,
        mu_plus: ArrayLike = 1.0,
        mu_minus: ArrayLike = 1.0,
        Wmax: ArrayLike = 100.0,
        Kplus: ArrayLike = 0.0,
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
        self.tau_minus = self._to_scalar_time_ms(tau_minus, name='tau_minus')
        self.lambda_ = self._to_scalar_float(lambda_, name='lambda')
        self.alpha = self._to_scalar_float(alpha, name='alpha')
        self.mu_plus = self._to_scalar_float(mu_plus, name='mu_plus')
        self.mu_minus = self._to_scalar_float(mu_minus, name='mu_minus')
        self.Wmax = self._to_scalar_float(Wmax, name='Wmax')
        self.Kplus = self._to_scalar_float(Kplus, name='Kplus')

        self._validate_weight_wmax_sign(weight_value, self.Wmax)
        self._validate_non_negative(self.Kplus, name='Kplus')

        self._Kplus0 = float(self.Kplus)
        self._t_lastspike0 = 0.0

        self.t_lastspike = float(self._t_lastspike0)
        self._post_kminus = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t: list[float] = []
        self._post_hist_kminus: list[float] = []

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        dftype = brainstate.environ.dftype()
        if isinstance(value, u.Quantity):
            unit = u.get_unit(value)
            arr = np.asarray(value.to_decimal(unit), dtype=dftype)
        else:
            arr = np.asarray(u.math.asarray(value), dtype=dftype)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr.reshape(()))
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        return v

    @staticmethod
    def _nest_sign(value: float) -> int:
        # Matches NEST set_status sign check:
        # ((x >= 0) - (x < 0)), so zero counts as positive.
        return int(value >= 0.0) - int(value < 0.0)

    @staticmethod
    def _validate_non_negative(value: float, *, name: str):
        if value < 0.0:
            raise ValueError(f'{name} must be non-negative.')

    @classmethod
    def _validate_weight_wmax_sign(cls, weight: float, Wmax: float):
        if cls._nest_sign(weight) != cls._nest_sign(Wmax):
            raise ValueError('Weight and Wmax must have same sign.')

    @staticmethod
    def _to_non_negative_int_count(value: ArrayLike, *, name: str) -> int:
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
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

    def _facilitate(self, w: float, kplus: float) -> float:
        norm_w = (w / self.Wmax) + (self.lambda_ * math.pow(1.0 - (w / self.Wmax), self.mu_plus) * kplus)
        return norm_w * self.Wmax if norm_w < 1.0 else self.Wmax

    def _depress(self, w: float, kminus: float) -> float:
        norm_w = (w / self.Wmax) - (self.alpha * self.lambda_ * math.pow(w / self.Wmax, self.mu_minus) * kminus)
        return norm_w * self.Wmax if norm_w > 0.0 else 0.0

    def clear_post_history(self):
        r"""Clear internal postsynaptic spike history and reset trace state.

        Resets all postsynaptic STDP state to initial conditions:

        - Clears spike history buffer (timestamps and trace values)
        - Resets postsynaptic trace ``K^-`` to zero
        - Resets last postsynaptic spike timestamp to ``-1.0``

        This method should be called periodically in long simulations to prevent
        unbounded growth of the spike history buffer. Typical usage: clear history
        at the start of each trial or after weight convergence phases.

        See Also
        --------
        init_state : Reinitialize all synapse state including weights and traces
        record_post_spike : Record postsynaptic spikes into the history buffer
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

        This method updates the postsynaptic eligibility trace :math:`K^-` and stores
        the spike timestamp for later use by :meth:`send` when processing presynaptic
        spikes. Each recorded spike increments :math:`K^-` by 1.0 after exponential
        decay from the previous postsynaptic spike.

        The trace update follows:

        .. math::

           K^- \\leftarrow K^- \\cdot e^{(t_{\\text{last\\_post}} - t_{\\text{spike}}) / \\tau_-} + 1

        where :math:`t_{\\text{last\\_post}}` is the timestamp of the previous postsynaptic
        spike and :math:`\\tau_-` is the postsynaptic trace time constant.

        Multiple spikes can be recorded by setting ``multiplicity > 1``. This is
        equivalent to calling the method ``multiplicity`` times at the same timestamp.

        Parameters
        ----------
        multiplicity : float, array-like, or Quantity, optional
            Number of postsynaptic spikes to record at this timestamp. Must be a
            non-negative integer-valued scalar (fractional values will be rejected).
            Use ``multiplicity=0`` to skip recording (returns immediately).
            Default: ``1`` (single spike).
        t_spike_ms : float, array-like, or Quantity, optional
            Spike timestamp in milliseconds. Must be a scalar float with or without
            time units. If ``None``, uses the current on-grid spike stamp
            :math:`t + dt` where :math:`t` is the current simulation time and
            :math:`dt` is the simulation time step. Default: ``None`` (on-grid).

        Returns
        -------
        int
            Number of spikes successfully recorded (equal to ``multiplicity``).

        Raises
        ------
        ValueError
            - If ``multiplicity`` is negative or not integer-valued.
            - If ``t_spike_ms`` is not a finite scalar.

        See Also
        --------
        clear_post_history : Clear all postsynaptic spike history
        send : Process presynaptic spike and apply STDP weight updates

        Examples
        --------
        **Record single postsynaptic spike at current time:**

        .. code-block:: python

           >>> import brainstate
           >>> import saiunit as u
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     syn = bst.stdp_synapse(weight=1.0)
           ...     syn.init_state()
           ...     count = syn.record_post_spike()  # uses t + dt
           ...     print(count)
           1

        **Record postsynaptic spike at explicit timestamp:**

        .. code-block:: python

           >>> syn.record_post_spike(multiplicity=1, t_spike_ms=5.0)
           1

        **Record burst of 3 postsynaptic spikes:**

        .. code-block:: python

           >>> syn.record_post_spike(multiplicity=3)
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
        r"""Initialize synapse state for simulation.

        Resets all dynamic state variables to their initial values:

        - ``Kplus``: presynaptic trace → initial value (default ``0.0``)
        - ``t_lastspike``: last presynaptic spike time → ``0.0`` ms
        - Postsynaptic spike history and trace → cleared

        This method should be called before starting a new simulation or trial.
        Inherits delay queue initialization from :class:`static_synapse`.

        Parameters
        ----------
        batch_size : int, optional
            Ignored (provided for API compatibility with batched models).
        **kwargs
            Ignored (provided for API compatibility).

        See Also
        --------
        clear_post_history : Clear only postsynaptic history without resetting other state
        set : Update parameters without reinitializing state
        """
        del batch_size, kwargs
        super().init_state()
        self.Kplus = float(self._Kplus0)
        self.t_lastspike = float(self._t_lastspike0)
        self.clear_post_history()

    def get(self) -> dict:
        r"""Return current public parameters and mutable state.

        Retrieves all NEST-compatible parameters and dynamic state variables in a
        dictionary format suitable for inspection, logging, or serialization. Includes
        both inherited parameters from :class:`static_synapse` (``weight``, ``delay``,
        ``receptor_type``) and STDP-specific parameters.

        Returns
        -------
        dict
            Dictionary containing:

            - ``'weight'`` : float – Current synaptic weight (plastic)
            - ``'delay'`` : float – Transmission delay (ms)
            - ``'receptor_type'`` : int – Postsynaptic receptor port
            - ``'tau_plus'`` : float – Presynaptic trace time constant (ms)
            - ``'tau_minus'`` : float – Postsynaptic trace time constant (ms)
            - ``'lambda'`` : float – Learning rate (key name without underscore)
            - ``'alpha'`` : float – Depression asymmetry factor
            - ``'mu_plus'`` : float – Potentiation exponent
            - ``'mu_minus'`` : float – Depression exponent
            - ``'Wmax'`` : float – Maximum weight bound
            - ``'Kplus'`` : float – Current presynaptic trace value
            - ``'synapse_model'`` : str – Always ``'stdp_synapse'`` (NEST identifier)

        See Also
        --------
        set : Update parameters and state
        init_state : Reinitialize state to defaults

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.stdp_synapse(weight=0.5, lambda_=0.01)
           >>> params = syn.get()
           >>> params['weight']
           0.5
           >>> params['lambda']
           0.01
           >>> params['synapse_model']
           'stdp_synapse'
        """
        params = super().get()
        params['tau_plus'] = float(self.tau_plus)
        params['tau_minus'] = float(self.tau_minus)
        params['lambda'] = float(self.lambda_)
        params['alpha'] = float(self.alpha)
        params['mu_plus'] = float(self.mu_plus)
        params['mu_minus'] = float(self.mu_minus)
        params['Wmax'] = float(self.Wmax)
        params['Kplus'] = float(self.Kplus)
        params['synapse_model'] = 'stdp_synapse'
        return params

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
        mu_plus: ArrayLike | object = _UNSET,
        mu_minus: ArrayLike | object = _UNSET,
        Wmax: ArrayLike | object = _UNSET,
        Kplus: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        r"""Set NEST-style public parameters and mutable state.

        Updates one or more synapse parameters and state variables without reinitializing
        the full simulation state. Mimics NEST's ``SetStatus`` API. All parameters are
        validated before assignment to ensure consistency (e.g., ``weight`` and ``Wmax``
        must have the same sign).

        Only specified parameters are updated; unspecified parameters retain their
        current values. To reset all state to initial conditions, use :meth:`init_state`
        instead.

        Parameters
        ----------
        weight : float, array-like, or Quantity, optional
            New synaptic weight. Must have the same sign as ``Wmax`` (or new ``Wmax``
            if both are specified). Validated before assignment. Default: unchanged.
        delay : float, array-like, or Quantity, optional
            New transmission delay (ms). Must be positive. Will be discretized to
            integer time steps on next usage. Default: unchanged.
        receptor_type : int, optional
            New receptor port identifier. Must be non-negative. Default: unchanged.
        tau_plus : float, array-like, or Quantity, optional
            New presynaptic trace time constant (ms). Must be positive.
            Default: unchanged.
        tau_minus : float, array-like, or Quantity, optional
            New postsynaptic trace time constant (ms). Must be positive.
            Default: unchanged.
        lambda_ : float, array-like, or Quantity, optional
            New learning rate. Typically positive. Default: unchanged.
        alpha : float, array-like, or Quantity, optional
            New depression asymmetry factor. Typically non-negative. Default: unchanged.
        mu_plus : float, array-like, or Quantity, optional
            New potentiation exponent. Must be non-negative. Default: unchanged.
        mu_minus : float, array-like, or Quantity, optional
            New depression exponent. Must be non-negative. Default: unchanged.
        Wmax : float, array-like, or Quantity, optional
            New maximum weight bound. Must have the same sign as ``weight`` (or new
            ``weight`` if both are specified). Default: unchanged.
        Kplus : float, array-like, or Quantity, optional
            New presynaptic trace value. Must be non-negative. Typically used to
            restore saved state rather than manipulate during simulation.
            Default: unchanged.
        post : Dynamics, optional
            New default postsynaptic receiver. Default: unchanged.

        Raises
        ------
        ValueError
            - If ``weight`` and ``Wmax`` have different signs.
            - If ``Kplus`` is negative.
            - If any parameter has non-finite values or incorrect shape.

        See Also
        --------
        get : Retrieve current parameters and state
        init_state : Reinitialize all state to defaults

        Examples
        --------
        **Update learning rate during simulation:**

        .. code-block:: python

           >>> syn = bst.stdp_synapse(weight=1.0, lambda_=0.01)
           >>> syn.set(lambda_=0.001)  # reduce learning rate
           >>> syn.get()['lambda']
           0.001

        **Update multiple parameters atomically:**

        .. code-block:: python

           >>> syn.set(
           ...     weight=0.5,
           ...     Wmax=2.0,
           ...     alpha=1.1,
           ... )

        **Restore saved state:**

        .. code-block:: python

           >>> saved_params = syn.get()
           >>> # ... simulation ...
           >>> syn.set(**{k: v for k, v in saved_params.items()
           ...            if k != 'synapse_model'})  # restore all except model name
        """
        new_weight = self.weight if weight is _UNSET else self._to_scalar_float(weight, name='weight')
        new_tau_plus = (
            self.tau_plus
            if tau_plus is _UNSET
            else self._to_scalar_time_ms(tau_plus, name='tau_plus')
        )
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
        new_mu_plus = (
            self.mu_plus
            if mu_plus is _UNSET
            else self._to_scalar_float(mu_plus, name='mu_plus')
        )
        new_mu_minus = (
            self.mu_minus
            if mu_minus is _UNSET
            else self._to_scalar_float(mu_minus, name='mu_minus')
        )
        new_Wmax = self.Wmax if Wmax is _UNSET else self._to_scalar_float(Wmax, name='Wmax')
        new_Kplus = self.Kplus if Kplus is _UNSET else self._to_scalar_float(Kplus, name='Kplus')

        self._validate_weight_wmax_sign(float(new_weight), float(new_Wmax))
        self._validate_non_negative(float(new_Kplus), name='Kplus')

        super_kwargs = {}
        if weight is not _UNSET:
            super_kwargs['weight'] = float(new_weight)
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
        self.mu_plus = float(new_mu_plus)
        self.mu_minus = float(new_mu_minus)
        self.Wmax = float(new_Wmax)
        self.Kplus = float(new_Kplus)

        self._Kplus0 = float(self.Kplus)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        r"""Schedule one outgoing event with NEST ``stdp_synapse`` dynamics.

        Processes a presynaptic spike, applies STDP weight updates (facilitation from
        past postsynaptic spikes and depression from current spike), then schedules
        the event for delayed delivery to the postsynaptic receiver. This method
        replicates the exact update sequence from NEST ``models/stdp_synapse.h::send()``.

        **Update sequence:**

        1. Compute inter-spike interval :math:`h = t_{\text{spike}} - t_{\text{last}}`
        2. Retrieve postsynaptic spikes in window :math:`(t_{\\text{last}} - d,\\, t_{\\text{spike}} - d]`
        3. Apply facilitation for each retrieved postsynaptic spike (post-before-pre)
        4. Compute postsynaptic trace :math:`K^-` at :math:`t_{\text{spike}} - d`
        5. Apply depression based on :math:`K^-` (pre-before-post)
        6. Schedule weighted event for delivery at :math:`t_{\text{spike}} + \text{delay}`
        7. Update presynaptic trace: :math:`K^+ \\leftarrow K^+ \\cdot e^{-h/\\tau_+} + 1`
        8. Update last spike timestamp: :math:`t_{\\text{last}} \\leftarrow t_{\\text{spike}}`

        The final delivered weight is :math:`w_{\\text{eff}} = w \\cdot \\text{multiplicity}`
        where :math:`w` is the plasticity-updated weight.

        Parameters
        ----------
        multiplicity : float, array-like, or Quantity, optional
            Presynaptic spike multiplicity (event magnitude). Scalar value, typically
            ``1.0`` for a single spike or ``0.0`` to skip transmission. The delivered
            payload is scaled by this factor. Default: ``1.0``.
        post : Dynamics, optional
            Postsynaptic receiver object for this event. If ``None``, uses the default
            receiver specified during initialization. Must implement ``add_delta_input``
            or handle spike events. Default: ``None`` (use default receiver).
        receptor_type : int, optional
            Receptor port override for this event. If ``None``, uses the synapse's
            default ``receptor_type``. Default: ``None`` (use default receptor).

        Returns
        -------
        bool
            ``True`` if an event was scheduled (``multiplicity != 0``), ``False`` otherwise.

        Raises
        ------
        ValueError
            - If ``multiplicity`` is not a finite scalar.
            - If ``receptor_type`` is negative or not an integer.
            - If no postsynaptic receiver is available (neither ``post`` argument nor
              default receiver specified).

        See Also
        --------
        update : High-level method combining event delivery, post-spike recording, and sending
        record_post_spike : Manually record postsynaptic spikes for STDP

        Notes
        -----
        - Spike timestamp uses on-grid time :math:`t + dt` (NEST convention).
        - Dendritic delay :math:`d` shifts the STDP causality window backward in time.
        - Postsynaptic spike history is never cleared by this method; call
          :meth:`clear_post_history` periodically to prevent memory growth.

        Examples
        --------
        **Send single presynaptic spike:**

        .. code-block:: python

           >>> import brainstate
           >>> import saiunit as u
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     post_neuron = bst.LIF(1)
           ...     syn = bst.stdp_synapse(weight=1.0, post=post_neuron)
           ...     syn.init_state()
           ...     post_neuron.init_state()
           ...     success = syn.send(multiplicity=1.0)
           ...     print(success)
           True

        **Send presynaptic spike with receptor override:**

        .. code-block:: python

           >>> syn.send(multiplicity=1.0, receptor_type=1)  # target receptor port 1
           True

        **Skip transmission (zero multiplicity):**

        .. code-block:: python

           >>> syn.send(multiplicity=0.0)
           False
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
        r"""Deliver due events, update post history, then process pre spikes.

        High-level update method combining all STDP synapse operations for a single
        simulation time step. This method is typically called once per time step in
        network simulations and handles:

        1. Delivery of delayed events from previous time steps
        2. Recording of postsynaptic spikes into the STDP history buffer
        3. Aggregation of presynaptic inputs (from ``current_inputs`` and ``delta_inputs``)
        4. STDP weight update and event scheduling via :meth:`send`

        The update order ensures correct causality: delayed events are delivered before
        processing new spikes, and postsynaptic spikes are recorded before presynaptic
        spikes are processed (allowing immediate STDP updates if delay is minimal).

        **Update sequence:**

        **Step 1: Deliver due events**
            Check the internal delay queue and deliver all events scheduled for the
            current simulation step to their target receivers.

        **Step 2: Record postsynaptic spikes**
            If ``post_spike > 0``, record ``post_spike`` postsynaptic spikes at
            timestamp :math:`t + dt` into the STDP history buffer. This updates the
            postsynaptic trace :math:`K^-`.

        **Step 3: Aggregate presynaptic inputs**
            Sum inputs from:
            - ``pre_spike`` argument (explicit input)
            - ``current_inputs`` dict (accumulated continuous inputs)
            - ``delta_inputs`` dict (accumulated spike inputs)

        **Step 4: Process presynaptic spike**
            If aggregated input is non-zero, call :meth:`send` to apply STDP weight
            updates and schedule a new delayed event.

        Parameters
        ----------
        pre_spike : float, array-like, or Quantity, optional
            Presynaptic spike multiplicity (explicit input). Added to accumulated
            inputs from ``current_inputs`` and ``delta_inputs``. Typically ``0.0``
            (no explicit input) or ``1.0`` (single spike). Default: ``0.0``.
        post_spike : float, array-like, or Quantity, optional
            Postsynaptic spike multiplicity to record. Must be a non-negative
            integer-valued scalar. If ``> 0``, records the specified number of
            postsynaptic spikes at the current on-grid timestamp :math:`t + dt`.
            Default: ``0.0`` (no postsynaptic spikes).
        post : Dynamics, optional
            Postsynaptic receiver object for event delivery. If ``None``, uses the
            default receiver specified during initialization. Default: ``None``.
        receptor_type : int, optional
            Receptor port override for event delivery. If ``None``, uses the synapse's
            default ``receptor_type``. Default: ``None``.

        Returns
        -------
        int
            Number of events delivered during this time step (from the delay queue).
            Does **not** include the newly scheduled event from this time step's
            presynaptic spike (that event will be counted in a future time step).

        Raises
        ------
        ValueError
            - If ``post_spike`` is negative or not integer-valued.
            - If ``pre_spike`` or aggregated inputs are not finite scalars.

        See Also
        --------
        send : Low-level method for processing a single presynaptic spike
        record_post_spike : Record postsynaptic spikes without other update operations

        Notes
        -----
        - This method modifies synapse state (``weight``, ``Kplus``, ``t_lastspike``,
          postsynaptic history) and should be called exactly once per time step.
        - The returned delivery count reflects past events, not the current time step's
          transmission.
        - For standalone STDP testing without a network, manually call :meth:`record_post_spike`
          and :meth:`send` instead of relying on :meth:`update`.

        Examples
        --------
        **Typical usage in network simulation loop:**

        .. code-block:: python

           >>> import brainstate
           >>> import saiunit as u
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     pre = bst.LIF(1)
           ...     post = bst.LIF(1)
           ...     syn = bst.stdp_synapse(weight=1.0, post=post)
           ...     pre.init_state()
           ...     post.init_state()
           ...     syn.init_state()
           ...     # Simulation step: presynaptic spike, no postsynaptic spike
           ...     delivered = syn.update(pre_spike=1.0, post_spike=0.0)
           ...     # Simulation step: no presynaptic spike, postsynaptic spike
           ...     delivered = syn.update(pre_spike=0.0, post_spike=1.0)

        **Standalone STDP test with explicit spike times:**

        .. code-block:: python

           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     syn = bst.stdp_synapse(weight=1.0, tau_plus=20.0*u.ms, tau_minus=20.0*u.ms)
           ...     syn.init_state()
           ...     # Post-before-pre: potentiation expected
           ...     syn.record_post_spike(multiplicity=1, t_spike_ms=5.0)
           ...     syn.send(multiplicity=1)  # pre-spike at t + dt (uses on-grid time)
           ...     print(f"Weight after potentiation: {syn.weight:.6f}")  # > 1.0
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
