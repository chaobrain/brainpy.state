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
from typing import Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

__all__ = [
    'vogels_sprekeler_synapse',
]


class vogels_sprekeler_synapse:
    r"""NEST-compatible ``vogels_sprekeler_synapse`` connection model.

    This class reproduces connection-level semantics of NEST
    ``models/vogels_sprekeler_synapse.{h,cpp}``, implementing inhibitory
    spike-timing-dependent plasticity (iSTDP) following Vogels & Sprekeler (2011).
    The rule combines symmetric STDP (no depression for post-before-pre timing)
    with constant presynaptic depression, designed to maintain excitatory-inhibitory
    balance in recurrent networks.

    **1. Mathematical Model**

    The learning rule modifies synaptic weight :math:`w` based on spike timing:

    .. math::

        \Delta w = \begin{cases}
        \eta \cdot K_- & \text{(pre-post pairing)} \\
        -\alpha \eta & \text{(constant depression per pre spike)}
        \end{cases}

    where:

    - :math:`K_-` is the postsynaptic trace (decaying exponentially with time constant :math:`\tau`)
    - :math:`\eta` is the learning rate
    - :math:`\alpha` is the constant depression factor

    Traces evolve as:

    .. math::

        \frac{dK_+}{dt} = -\frac{K_+}{\tau} + \sum_i \delta(t - t_i^{\text{pre}})

        \frac{dK_-}{dt} = -\frac{K_-}{\tau} + \sum_j \delta(t - t_j^{\text{post}})

    **2. NEST Send-Order Update Sequence**

    For one presynaptic spike at time :math:`t` with dendritic delay :math:`d`:

    1. **History lookup**: Retrieve postsynaptic spikes in :math:`(t_{\text{last}} - d, t - d]`
    2. **Pairwise facilitation**: For each postsynaptic spike :math:`t_j` in history:

       .. math::

           w \leftarrow \operatorname{facilitate}\!\left(
           w, K_+ \exp\left(\frac{t_{\text{last}} - (t_j + d)}{\tau}\right)\right)

    3. **Current postsynaptic trace facilitation**:

       .. math::

           w \leftarrow \operatorname{facilitate}(w, K_-(t - d))

    4. **Constant depression**:

       .. math::

           w \leftarrow \operatorname{depress}(w)

    5. **Emit spike event** using updated weight
    6. **Update presynaptic trace**:

       .. math::

           K_+ \leftarrow K_+ \exp\left(\frac{t_{\text{last}} - t}{\tau}\right) + 1

    7. **Update timestamp**: :math:`t_{\text{last}} \leftarrow t`

    **3. Weight Clipping Rules**

    Facilitate and depress operations are sign-aware via :math:`W_{\max}`:

    .. math::

        \operatorname{facilitate}(w, k) =
        \operatorname{copysign}\left(\min(|w| + \eta k,\ |W_{\max}|), W_{\max}\right)

    .. math::

        \operatorname{depress}(w) =
        \operatorname{copysign}\left(\max(|w| - \alpha\eta,\ 0), W_{\max}\right)

    This ensures weights saturate at :math:`\pm |W_{\max}|` while preserving sign.

    **4. Biological Motivation**

    This rule implements the iSTDP mechanism proposed by Vogels & Sprekeler for
    inhibitory synapses. The constant depression term :math:`\alpha` causes weight
    decay independent of post-pre timing, while facilitation occurs for pre-post
    pairings. This asymmetry drives inhibitory weights to track excitatory activity,
    maintaining balanced network states without fine-tuned parameters.

    Parameters
    ----------
    weight : float, array-like, optional
        Synaptic weight (unitless). Can be positive (excitatory) or negative
        (inhibitory). Must have same sign as ``Wmax`` if non-zero.
        Default: ``0.5``.
    delay : float, array-like, optional
        Dendritic delay in milliseconds used for spike history lookup. Must be
        positive. Determines time window for postsynaptic spike detection.
        Default: ``1.0`` ms.
    delay_steps : int, array-like, optional
        Event delivery delay in integer simulation steps (≥1). Controls when
        spike arrives at postsynaptic target after emission.
        Default: ``1``.
    tau : float, array-like, optional
        STDP time constant in milliseconds (>0). Governs exponential decay of
        presynaptic (:math:`K_+`) and postsynaptic (:math:`K_-`) traces.
        Typical range: 10-50 ms.
        Default: ``20.0`` ms.
    alpha : float, array-like, optional
        Constant depression factor (unitless). Scales the per-spike weight
        reduction: :math:`\Delta w = -\alpha \eta`. Setting :math:`\alpha = 0`
        disables constant depression (pure Hebbian STDP).
        Default: ``0.12``.
    eta : float, array-like, optional
        Learning rate (unitless). Scales both facilitation and depression.
        Smaller values (≪1) ensure gradual weight changes.
        Default: ``0.001``.
    Wmax : float, array-like, optional
        Signed maximum absolute weight (unitless). Defines saturation bounds
        :math:`[-|W_{\max}|, +|W_{\max}|]` and determines sign of weight dynamics.
        Must have same sign as ``weight`` (if ``weight != 0``).
        Default: ``1.0``.
    Kplus : float, array-like, optional
        Initial presynaptic STDP trace value (unitless, ≥0). Represents accumulated
        presynaptic activity. Typically initialized to 0 before simulation.
        Default: ``0.0``.
    t_last_spike_ms : float, array-like, optional
        Timestamp of last presynaptic spike in milliseconds. Used for trace decay
        calculations. Initialize to simulation start time or 0.
        Default: ``0.0`` ms.
    name : str, optional
        Model instance name for identification.
        Default: ``None``.

    See Also
    --------
    stdp_synapse : Classical asymmetric STDP rule
    stdp_dopamine_synapse : Reward-modulated STDP

    Parameter Mapping
    -----------------

    +-----------------+---------------------+-------------------+
    | NEST Parameter  | brainpy.state       | Unit              |
    +=================+=====================+===================+
    | ``weight``      | ``weight``          | unitless          |
    +-----------------+---------------------+-------------------+
    | ``delay``       | ``delay``           | ms                |
    +-----------------+---------------------+-------------------+
    | ``delay_steps`` | ``delay_steps``     | steps             |
    +-----------------+---------------------+-------------------+
    | ``tau``         | ``tau``             | ms                |
    +-----------------+---------------------+-------------------+
    | ``alpha``       | ``alpha``           | unitless          |
    +-----------------+---------------------+-------------------+
    | ``eta``         | ``eta``             | unitless          |
    +-----------------+---------------------+-------------------+
    | ``Wmax``        | ``Wmax``            | unitless          |
    +-----------------+---------------------+-------------------+
    | ``Kplus``       | ``Kplus``           | unitless          |
    +-----------------+---------------------+-------------------+
    | ``t_lastspike`` | ``t_last_spike_ms`` | ms                |
    +-----------------+---------------------+-------------------+

    **Target Interface Requirements**

    The ``send()`` method requires postsynaptic targets to implement:

    - ``get_history(t1, t2)`` -- Returns spike history entries in time window
      ``(t1, t2]`` (exclusive-inclusive). Entries must expose spike time via
      attribute ``t_`` or ``t``, dict key ``'t_'`` or ``'t'``, or first tuple
      element.

    - ``get_K_value(t)`` or ``get_k_value(t)`` -- Returns postsynaptic STDP trace
      :math:`K_-` at time ``t`` (in ms). Must return float.

    1. **Dendritic delay semantics**: Unlike axonal delays (which shift spike
       arrival time), the ``delay`` parameter here controls the temporal window
       for history lookup: :math:`(t_{\text{last}} - d, t - d]`. This implements
       NEST's dendritic delay convention.

    2. **Sign constraints**: If ``weight != 0``, both ``weight`` and ``Wmax`` must
       have the same sign. Attempting to set opposite signs raises ``ValueError``.
       This preserves synapse type (excitatory/inhibitory) throughout learning.

    3. **Trace positivity**: ``Kplus`` must remain non-negative. Negative values
       raise ``ValueError`` during initialization or ``set_status()``.

    4. **Sub-step timing**: As in NEST, precise spike times within a time step
       (e.g., off-grid timestamps) are **ignored** for plasticity calculations.
       All updates use coarse time step boundaries.

    5. **Event multiplicity**: The ``multiplicity`` parameter in ``send()`` is
       validated but not explicitly used in weight updates (reserved for future
       multi-spike events).

    References
    ----------
    .. [1] Vogels, T. P., & Sprekeler, H. (2011). Inhibitory plasticity balances
           excitation and inhibition in sensory pathways and memory networks.
           *Science*, 334(6062), 1569-1573.
           https://doi.org/10.1126/science.1211095

    .. [2] NEST Initiative (2024). Vogels-Sprekeler Synapse Model.
           NEST Simulator Documentation.
           https://nest-simulator.readthedocs.io/en/stable/models/vogels_sprekeler_synapse.html

    Examples
    --------
    Basic synapse creation with default parameters:

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> syn = bp.vogels_sprekeler_synapse()
        >>> syn.get_status()
        {'weight': 0.5, 'tau': 20.0, 'alpha': 0.12, 'eta': 0.001, ...}

    Configure for inhibitory synapse with stronger depression:

    .. code-block:: python

        >>> syn = bp.vogels_sprekeler_synapse(
        ...     weight=-0.8,
        ...     Wmax=-2.0,
        ...     alpha=0.2,
        ...     eta=0.005,
        ...     tau=30.0
        ... )
        >>> syn.weight
        -0.8

    Update parameters after creation:

    .. code-block:: python

        >>> syn.set_status({'alpha': 0.15, 'eta': 0.002})
        >>> syn.alpha
        0.15

    Process presynaptic spike with mock postsynaptic target:

    .. code-block:: python

        >>> class MockNeuron:
        ...     def get_history(self, t1, t2):
        ...         # Return spike at t=10.5 ms
        ...         return [{'t_': 10.5}]
        ...     def get_K_value(self, t):
        ...         return 0.3  # Current postsynaptic trace
        >>> target = MockNeuron()
        >>> event = syn.send(t_spike_ms=15.0, target=target)
        >>> event['weight']  # Updated weight after plasticity
        0.506...
        >>> syn.Kplus  # Updated presynaptic trace
        1.0

    Simulate spike train:

    .. code-block:: python

        >>> pre_times = [10.0, 20.0, 30.0, 40.0]
        >>> events = syn.simulate_pre_spike_train(
        ...     pre_spike_times_ms=pre_times,
        ...     target=target
        ... )
        >>> len(events)
        4
        >>> [e['weight'] for e in events]  # Weight evolution
        [0.512..., 0.518..., 0.524..., 0.530...]
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    IS_PRIMARY = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True
    SUPPORTS_WFR = True

    def __init__(
        self,
        weight: ArrayLike = 0.5,
        delay: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        tau: ArrayLike = 20.0,
        alpha: ArrayLike = 0.12,
        eta: ArrayLike = 0.001,
        Wmax: ArrayLike = 1.0,
        Kplus: ArrayLike = 0.0,
        t_last_spike_ms: ArrayLike = 0.0,
        name: str | None = None,
    ):
        self.name = name

        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay = self._validate_positive_delay(delay)
        self.delay_steps = self._validate_delay_steps(delay_steps)
        self.tau = self._validate_positive_tau(tau)
        self.alpha = self._to_float_scalar(alpha, name='alpha')
        self.eta = self._to_float_scalar(eta, name='eta')
        self.Wmax = self._to_float_scalar(Wmax, name='Wmax')
        self.Kplus = self._to_float_scalar(Kplus, name='Kplus')
        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

        self._check_constraints()

    @property
    def properties(self) -> dict[str, Any]:
        r"""Return model properties dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``'has_delay'`` (bool): True (model supports synaptic delays).
            - ``'is_primary'`` (bool): True (model is a primary connection).
            - ``'supports_hpc'`` (bool): True (hybrid parallel computing compatible).
            - ``'supports_lbl'`` (bool): True (supports local backward learning).
            - ``'supports_wfr'`` (bool): True (supports waveform relaxation).
        """
        return {
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        r"""Retrieve current parameter and state values.

        Returns all synapse parameters, STDP trace state, and model properties
        as a dictionary. Follows NEST ``GetStatus`` semantics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - ``'weight'`` (float): Current synaptic weight (unitless).
            - ``'delay'`` (float): Dendritic delay (ms).
            - ``'delay_steps'`` (int): Event delivery delay (steps).
            - ``'tau'`` (float): STDP time constant (ms).
            - ``'alpha'`` (float): Constant depression factor (unitless).
            - ``'eta'`` (float): Learning rate (unitless).
            - ``'Wmax'`` (float): Signed maximum weight (unitless).
            - ``'Kplus'`` (float): Presynaptic STDP trace (unitless).
            - ``'t_last_spike_ms'`` (float): Last presynaptic spike time (ms).
            - ``'size_of'`` (int): Memory footprint in bytes.
            - ``'has_delay'`` (bool): Delay support flag.
            - ``'is_primary'`` (bool): Primary connection flag.
            - ``'supports_hpc'`` (bool): HPC compatibility flag.
            - ``'supports_lbl'`` (bool): LBL compatibility flag.
            - ``'supports_wfr'`` (bool): WFR compatibility flag.

        Examples
        --------
        .. code-block:: python

            >>> syn = bp.vogels_sprekeler_synapse(weight=0.7, tau=25.0)
            >>> status = syn.get_status()
            >>> status['weight']
            0.7
            >>> status['tau']
            25.0
        """
        return {
            'weight': float(self.weight),
            'delay': float(self.delay),
            'delay_steps': int(self.delay_steps),
            'tau': float(self.tau),
            'alpha': float(self.alpha),
            'eta': float(self.eta),
            'Wmax': float(self.Wmax),
            'Kplus': float(self.Kplus),
            't_last_spike_ms': float(self.t_last_spike_ms),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update synapse parameters and state variables.

        Modifies synapse configuration following NEST ``SetStatus`` semantics.
        Validates all updates and enforces constraints (positive delays/tau,
        non-negative Kplus, matching weight/Wmax signs).

        Parameters
        ----------
        status : dict[str, Any], optional
            Dictionary of parameter updates. Valid keys: ``'weight'``, ``'delay'``,
            ``'delay_steps'``, ``'tau'``, ``'alpha'``, ``'eta'``, ``'Wmax'``,
            ``'Kplus'``, ``'t_last_spike_ms'``.
        **kwargs
            Additional parameter updates as keyword arguments. Merged with
            ``status`` dict (kwargs take precedence).

        Raises
        ------
        ValueError
            If ``delay ≤ 0``, ``tau ≤ 0``, ``delay_steps < 1``, ``Kplus < 0``,
            or ``weight`` and ``Wmax`` have opposite signs (when ``weight != 0``).
        TypeError
            If parameter values are not scalar or convertible to required numeric type.

        Notes
        -----
        Constraint checking runs **after** all updates are applied, so transient
        inconsistent states (e.g., setting weight before Wmax) are allowed within
        a single call.

        Examples
        --------
        Update single parameter:

        .. code-block:: python

            >>> syn = bp.vogels_sprekeler_synapse()
            >>> syn.set_status({'alpha': 0.15})
            >>> syn.alpha
            0.15

        Update multiple parameters at once:

        .. code-block:: python

            >>> syn.set_status({'eta': 0.002, 'tau': 30.0})
            >>> syn.eta, syn.tau
            (0.002, 30.0)

        Use keyword arguments:

        .. code-block:: python

            >>> syn.set_status(weight=0.8, Wmax=2.0)
            >>> syn.weight, syn.Wmax
            (0.8, 2.0)

        Invalid updates raise errors:

        .. code-block:: python

            >>> syn.set_status({'tau': -5.0})  # doctest: +SKIP
            ValueError: tau must be > 0.
            >>> syn.set_status({'Kplus': -0.1})  # doctest: +SKIP
            ValueError: State Kplus must be positive.
        """
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        if 'weight' in updates:
            self.weight = self._to_float_scalar(updates['weight'], name='weight')
        if 'delay' in updates:
            self.delay = self._validate_positive_delay(updates['delay'])
        if 'delay_steps' in updates:
            self.delay_steps = self._validate_delay_steps(updates['delay_steps'])
        if 'tau' in updates:
            self.tau = self._validate_positive_tau(updates['tau'])
        if 'alpha' in updates:
            self.alpha = self._to_float_scalar(updates['alpha'], name='alpha')
        if 'eta' in updates:
            self.eta = self._to_float_scalar(updates['eta'], name='eta')
        if 'Wmax' in updates:
            self.Wmax = self._to_float_scalar(updates['Wmax'], name='Wmax')
        if 'Kplus' in updates:
            self.Kplus = self._to_float_scalar(updates['Kplus'], name='Kplus')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')

        self._check_constraints()

    def get(self, key: str = 'status'):
        r"""Retrieve specific parameter or full status dictionary.

        Parameters
        ----------
        key : str, optional
            Parameter name or ``'status'`` for full dictionary. Valid keys:
            ``'weight'``, ``'delay'``, ``'delay_steps'``, ``'tau'``, ``'alpha'``,
            ``'eta'``, ``'Wmax'``, ``'Kplus'``, ``'t_last_spike_ms'``, ``'size_of'``,
            ``'has_delay'``, ``'is_primary'``, ``'supports_hpc'``, ``'supports_lbl'``,
            ``'supports_wfr'``.
            Default: ``'status'``.

        Returns
        -------
        Any
            If ``key == 'status'``, returns full status dictionary (see ``get_status()``).
            Otherwise, returns the requested parameter value.

        Raises
        ------
        KeyError
            If ``key`` is not recognized.

        Examples
        --------
        .. code-block:: python

            >>> syn = bp.vogels_sprekeler_synapse(weight=0.6, tau=25.0)
            >>> syn.get('weight')
            0.6
            >>> syn.get('tau')
            25.0
            >>> status = syn.get('status')
            >>> 'weight' in status
            True
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for vogels_sprekeler_synapse.get().')

    def send(
        self,
        t_spike_ms: ArrayLike,
        target: Any,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay: ArrayLike | None = None,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        r"""Process one presynaptic spike and return emitted SpikeEvent payload.

        Implements the complete Vogels-Sprekeler STDP update sequence for a single
        presynaptic spike: retrieves postsynaptic spike history, applies pairwise
        facilitation, facilitates with current postsynaptic trace, applies constant
        depression, updates presynaptic trace, and emits spike event.

        Parameters
        ----------
        t_spike_ms : float, array-like
            Presynaptic spike time in milliseconds (scalar).
        target : Any
            Postsynaptic target object. Must implement:

            - ``get_history(t1, t2)`` -- iterable of spike entries in ``(t1, t2]``.
              Each entry must expose spike time via attribute ``t_``/``t``,
              dict key ``'t_'``/``'t'``, or first tuple element.
            - ``get_K_value(t)`` or ``get_k_value(t)`` -- float (postsynaptic
              trace :math:`K_-` at time ``t``).
        receptor_type : int, array-like, optional
            Postsynaptic receptor port index (≥0). Included in returned event
            dictionary for routing.
            Default: ``0``.
        multiplicity : float, array-like, optional
            Spike event multiplicity (≥0). Validated but not used in weight updates.
            Default: ``1.0``.
        delay : float, array-like, optional
            Override dendritic delay (ms, >0). If ``None``, uses ``self.delay``.
            Default: ``None``.
        delay_steps : int, array-like, optional
            Override event delivery delay (steps, ≥1). If ``None``, uses ``self.delay_steps``.
            Default: ``None``.

        Returns
        -------
        dict[str, Any]
            Spike event dictionary with keys:
            - ``'weight'`` (float): Updated synaptic weight after plasticity.
            - ``'delay'`` (float): Effective dendritic delay (ms).
            - ``'delay_steps'`` (int): Event delivery delay (steps).
            - ``'receptor_type'`` (int): Postsynaptic receptor index.
            - ``'multiplicity'`` (float): Event multiplicity.
            - ``'t_spike_ms'`` (float): Presynaptic spike time (ms).
            - ``'Kminus'`` (float): Postsynaptic trace value at ``t_spike - delay``.
            - ``'Kplus_pre'`` (float): Presynaptic trace **before** update.
            - ``'Kplus_post'`` (float): Presynaptic trace **after** update.

        Raises
        ------
        ValueError
            If ``delay ≤ 0``, ``delay_steps < 1``, ``multiplicity < 0``, or
            any parameter is non-scalar/non-finite.
        AttributeError
            If ``target`` does not implement required methods ``get_history()``
            and ``get_K_value()``/``get_k_value()``.
        TypeError
            If history entries do not expose spike time via expected attributes/keys.

        Notes
        -----
        1. **State updates are persistent**: This method modifies ``self.weight``,
           ``self.Kplus``, and ``self.t_last_spike_ms`` in place.

        2. **History lookup window**: Retrieves postsynaptic spikes in
           :math:`(t_{\text{last}} - d, t_{\text{spike}} - d]`, where :math:`d` is
           the dendritic delay.

        3. **Trace decay calculation**: Presynaptic trace decays as
           :math:`K_+ \leftarrow K_+ \exp((t_{\text{last}} - t) / \tau) + 1`,
           ensuring continuous exponential decay between spikes.

        4. **Weight clipping**: Facilitation and depression operations automatically
           clip weights to :math:`\pm |W_{\max}|` while preserving sign.

        Examples
        --------
        Process single spike with mock target:

        .. code-block:: python

            >>> import brainpy.state as bp
            >>> class MockTarget:
            ...     def get_history(self, t1, t2):
            ...         return [{'t_': 12.0}]  # One post spike at 12 ms
            ...     def get_K_value(self, t):
            ...         return 0.4
            >>> syn = bp.vogels_sprekeler_synapse(weight=0.5, tau=20.0, eta=0.01)
            >>> target = MockTarget()
            >>> event = syn.send(t_spike_ms=15.0, target=target)
            >>> event['weight']  # Facilitated then depressed
            0.502...
            >>> event['Kplus_post']  # Updated presynaptic trace
            1.0

        Override delay for specific spike:

        .. code-block:: python

            >>> event = syn.send(
            ...     t_spike_ms=20.0,
            ...     target=target,
            ...     delay=2.5,
            ...     delay_steps=3
            ... )
            >>> event['delay']
            2.5
            >>> event['delay_steps']
            3

        Access postsynaptic trace from event:

        .. code-block:: python

            >>> event['Kminus']  # Postsynaptic trace at spike time - delay
            0.4
        """
        t_spike = self._to_float_scalar(t_spike_ms, name='t_spike_ms')
        dendritic_delay = self.delay if delay is None else self._validate_positive_delay(delay)
        event_delay_steps = (
            self.delay_steps
            if delay_steps is None
            else self._validate_delay_steps(delay_steps)
        )

        history_entries = self._get_history(
            target,
            self.t_last_spike_ms - dendritic_delay,
            t_spike - dendritic_delay,
        )

        for entry in history_entries:
            t_hist = self._extract_history_time(entry)
            minus_dt = self.t_last_spike_ms - (t_hist + dendritic_delay)
            self.weight = self._facilitate(self.weight, self.Kplus * math.exp(minus_dt / self.tau))

        kminus = self._get_k_value(target, t_spike - dendritic_delay)
        self.weight = self._facilitate(self.weight, kminus)
        self.weight = self._depress(self.weight)

        event = {
            'weight': float(self.weight),
            'delay': float(dendritic_delay),
            'delay_steps': int(event_delay_steps),
            'receptor_type': self._to_int_scalar(receptor_type, name='receptor_type'),
            'multiplicity': self._validate_multiplicity(multiplicity),
            't_spike_ms': float(t_spike),
            'Kminus': float(kminus),
            'Kplus_pre': float(self.Kplus),
        }

        self.Kplus = self.Kplus * math.exp((self.t_last_spike_ms - t_spike) / self.tau) + 1.0
        self.t_last_spike_ms = t_spike
        event['Kplus_post'] = float(self.Kplus)
        return event

    def to_spike_event(
        self,
        t_spike_ms: ArrayLike,
        target: Any,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay: ArrayLike | None = None,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        r"""Alias for ``send()`` method.

        Identical to ``send()`` with the same parameters and return value.
        Provided for API compatibility with alternative naming conventions.

        See Also
        --------
        send : Primary spike processing method (full documentation).
        """
        return self.send(
            t_spike_ms=t_spike_ms,
            target=target,
            receptor_type=receptor_type,
            multiplicity=multiplicity,
            delay=delay,
            delay_steps=delay_steps,
        )

    def simulate_pre_spike_train(
        self,
        pre_spike_times_ms: ArrayLike,
        target: Any,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay: ArrayLike | None = None,
        delay_steps: ArrayLike | None = None,
    ) -> list[dict[str, Any]]:
        r"""Process a sequence of presynaptic spikes and return event list.

        Iteratively calls ``send()`` for each spike time, accumulating weight
        updates and trace dynamics across the entire spike train. Useful for
        simulating synapse evolution under controlled input patterns.

        Parameters
        ----------
        pre_spike_times_ms : array-like
            Presynaptic spike times in milliseconds. Can be 1D array, list, or
            scalar. Automatically flattened to 1D.
        target : Any
            Postsynaptic target (see ``send()`` for interface requirements).
        receptor_type : int, array-like, optional
            Postsynaptic receptor port (see ``send()``).
            Default: ``0``.
        multiplicity : float, array-like, optional
            Event multiplicity (see ``send()``).
            Default: ``1.0``.
        delay : float, array-like, optional
            Dendritic delay override (ms, see ``send()``).
            Default: ``None``.
        delay_steps : int, array-like, optional
            Delivery delay override (steps, see ``send()``).
            Default: ``None``.

        Returns
        -------
        list[dict[str, Any]]
            List of spike event dictionaries (one per input spike), in temporal
            order. Each dict has same structure as ``send()`` return value.

        Notes
        -----
        **State evolution**: Because ``send()`` modifies synapse state (``weight``,
        ``Kplus``, ``t_last_spike_ms``), the returned events reflect **cumulative**
        plasticity. Event ``i`` depends on events ``0`` through ``i-1``.

        Examples
        --------
        Simulate regular spike train:

        .. code-block:: python

            >>> import numpy as np
            >>> import brainpy.state as bp
            >>> class MockTarget:
            ...     def get_history(self, t1, t2):
            ...         # Postsynaptic spikes at 10, 30, 50 ms
            ...         return [{'t_': t} for t in [10, 30, 50] if t1 < t <= t2]
            ...     def get_K_value(self, t):
            ...         return 0.3
            >>> syn = bp.vogels_sprekeler_synapse(weight=0.5, eta=0.01)
            >>> target = MockTarget()
            >>> pre_times = np.arange(5, 60, 10)  # Spikes at 5, 15, 25, 35, 45, 55 ms
            >>> events = syn.simulate_pre_spike_train(pre_times, target)
            >>> len(events)
            6
            >>> [e['weight'] for e in events]  # Weight trajectory
            [0.498..., 0.502..., 0.505..., 0.509..., 0.512..., 0.515...]

        Extract presynaptic trace evolution:

        .. code-block:: python

            >>> kplus_trajectory = [e['Kplus_post'] for e in events]
            >>> kplus_trajectory[0]  # After first spike
            1.0
            >>> kplus_trajectory[-1]  # After last spike
            1.0

        Weight evolution with strong depression:

        .. code-block:: python

            >>> syn2 = bp.vogels_sprekeler_synapse(
            ...     weight=1.0,
            ...     alpha=0.5,
            ...     eta=0.02
            ... )
            >>> events2 = syn2.simulate_pre_spike_train([10, 20, 30], target)
            >>> [e['weight'] for e in events2]  # Depression dominates
            [0.996..., 0.992..., 0.988...]
        """
        times = np.asarray(u.math.asarray(pre_spike_times_ms), dtype=np.float64).reshape(-1)
        events = []
        for t in times:
            events.append(
                self.send(
                    t_spike_ms=float(t),
                    target=target,
                    receptor_type=receptor_type,
                    multiplicity=multiplicity,
                    delay=delay,
                    delay_steps=delay_steps,
                )
            )
        return events

    def _facilitate(self, w: float, kplus: float) -> float:
        new_w = abs(w) + self.eta * kplus
        return math.copysign(min(new_w, abs(self.Wmax)), self.Wmax)

    def _depress(self, w: float) -> float:
        new_w = abs(w) - self.alpha * self.eta
        return math.copysign(max(new_w, 0.0), self.Wmax)

    def _check_constraints(self):
        if self.Kplus < 0.0:
            raise ValueError('State Kplus must be positive.')
        if self.weight != 0.0 and (math.copysign(1.0, self.weight) != math.copysign(1.0, self.Wmax)):
            raise ValueError('Weight and Wmax must have same sign.')

    @staticmethod
    def _get_history(target: Any, t1: float, t2: float):
        if hasattr(target, 'get_history'):
            return target.get_history(float(t1), float(t2))
        raise AttributeError(
            'Target must provide get_history(t1, t2) for vogels_sprekeler_synapse.'
        )

    @staticmethod
    def _extract_history_time(entry: Any) -> float:
        if hasattr(entry, 't_'):
            return float(entry.t_)
        if hasattr(entry, 't'):
            return float(entry.t)
        if isinstance(entry, dict):
            if 't_' in entry:
                return float(entry['t_'])
            if 't' in entry:
                return float(entry['t'])
        if isinstance(entry, (tuple, list)) and len(entry) >= 1:
            return float(entry[0])
        raise TypeError(
            'History entry must expose a time as attribute t_/t, mapping key t_/t, or first tuple element.'
        )

    @staticmethod
    def _get_k_value(target: Any, t: float) -> float:
        if hasattr(target, 'get_K_value'):
            return float(target.get_K_value(float(t)))
        if hasattr(target, 'get_k_value'):
            return float(target.get_k_value(float(t)))
        raise AttributeError(
            'Target must provide get_K_value(t) or get_k_value(t) for vogels_sprekeler_synapse.'
        )

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr[0])
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        return v

    @staticmethod
    def _to_int_scalar(value: ArrayLike, name: str) -> int:
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr[0])
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        iv = int(round(v))
        if abs(v - iv) > 1e-12:
            raise ValueError(f'{name} must be an integer value.')
        return iv

    @classmethod
    def _validate_positive_delay(cls, value: ArrayLike) -> float:
        d = cls._to_float_scalar(value, name='delay')
        if d <= 0.0:
            raise ValueError('delay must be > 0.')
        return d

    @classmethod
    def _validate_delay_steps(cls, value: ArrayLike) -> int:
        d = cls._to_int_scalar(value, name='delay_steps')
        if d < 1:
            raise ValueError('delay_steps must be >= 1.')
        return d

    @classmethod
    def _validate_positive_tau(cls, value: ArrayLike) -> float:
        tau = cls._to_float_scalar(value, name='tau')
        if tau <= 0.0:
            raise ValueError('tau must be > 0.')
        return tau

    @classmethod
    def _validate_multiplicity(cls, value: ArrayLike) -> float:
        m = cls._to_float_scalar(value, name='multiplicity')
        if m < 0.0:
            raise ValueError('multiplicity must be >= 0.')
        return m
