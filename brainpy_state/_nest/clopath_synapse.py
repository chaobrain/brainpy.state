import math
from typing import Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

from ._base import NESTSynapse

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

__all__ = [
    'clopath_synapse',
]


class clopath_synapse(NESTSynapse):
    r"""NEST-compatible voltage-based STDP synapse following the Clopath plasticity rule.

    This synapse model implements the voltage-based spike-timing-dependent plasticity (STDP)
    rule described by Clopath et al. (2010). Unlike traditional pair-based STDP, weight updates
    depend on postsynaptic membrane voltage traces archived by the target neuron, enabling
    voltage-dependent learning rules that capture homeostatic regulation and triplet interactions.

    **1. Model Overview**

    The Clopath synapse is a connection-level model that modulates synaptic weights based on:

    - Presynaptic spike times and a presynaptic trace :math:`\bar{x}(t)`
    - Postsynaptic voltage-derived traces for long-term potentiation (LTP) and depression (LTD)
    - Hard bounds :math:`[W_\text{min}, W_\text{max}]` on synaptic weight

    The model requires the postsynaptic neuron to maintain voltage-dependent history buffers
    (e.g., ``aeif_psc_delta_clopath`` in NEST).

    **2. State Variables**

    Each connection maintains:

    - :math:`w` -- Current synaptic weight
    - :math:`\bar{x}` -- Presynaptic trace (low-pass filtered spike train)
    - :math:`t_\text{last}` -- Timestamp of most recent presynaptic spike (milliseconds)
    - :math:`\tau_x` -- Time constant for presynaptic trace decay (milliseconds)
    - :math:`W_\text{min}, W_\text{max}` -- Hard lower/upper weight bounds

    **3. Plasticity Update Sequence**

    On each presynaptic spike at time :math:`t` (with dendritic delay :math:`d` and previous
    spike time :math:`t_\text{last}`), the following steps are executed in order:

    **(a) Long-Term Potentiation (LTP):**

    Retrieve all LTP history entries from the postsynaptic neuron in the interval
    :math:`(t_\text{last} - d,\, t - d]`. For each entry with timestamp :math:`t_i` and
    amplitude :math:`\text{dw}_i`:

    .. math::

        w \leftarrow \min\left(W_\text{max},\, w + \text{dw}_i \cdot \bar{x}
        \exp\left(\frac{t_\text{last} - (t_i + d)}{\tau_x}\right)\right)

    This facilitates the weight proportional to the presynaptic trace at the effective time
    :math:`t_i + d`, accounting for exponential decay from :math:`t_\text{last}`.

    **(b) Long-Term Depression (LTD):**

    Query the postsynaptic LTD value at the effective time :math:`t - d`:

    .. math::

        w \leftarrow \max(W_\text{min},\, w - \text{dw}_\text{LTD}(t - d))

    This depresses the weight by the current LTD trace magnitude, clamped to :math:`W_\text{min}`.

    **(c) Event Emission:**

    A spike event is generated with the updated weight and delay parameters.

    **(d) Presynaptic Trace Update:**

    The presynaptic trace is updated to account for the new spike:

    .. math::

        \bar{x} \leftarrow \bar{x} \exp\left(\frac{t_\text{last} - t}{\tau_x}\right)
        + \frac{1}{\tau_x}

    **(e) Timestamp Update:**

    The last spike time is updated: :math:`t_\text{last} \leftarrow t`.

    **4. Mathematical Foundations**

    The presynaptic trace :math:`\bar{x}(t)` is a low-pass filter of the presynaptic spike
    train :math:`S_\text{pre}(t) = \sum_k \delta(t - t_k)`:

    .. math::

        \tau_x \frac{d\bar{x}}{dt} = -\bar{x} + S_\text{pre}(t)

    At each spike time :math:`t_k`, the exact solution yields the jump condition:

    .. math::

        \bar{x}(t_k^+) = \bar{x}(t_k^-) e^{-(t_k - t_{k-1})/\tau_x} + \frac{1}{\tau_x}

    This exact event-driven update is implemented in step (d).

    **5. Postsynaptic Interface Requirements**

    The target neuron must provide:

    - ``get_LTP_history(t1, t2)`` or ``get_ltp_history(t1, t2)``:
      Returns iterable of LTP events in :math:`(t_1, t_2]`
    - ``get_LTD_value(t)`` or ``get_ltd_value(t)``:
      Returns scalar LTD amplitude at time :math:`t`

    Each LTP history entry must support extraction of:

    - Time field: ``t_``, ``t``, ``time_ms``, or ``time``
    - Weight change field: ``dw_``, ``dw``, ``delta_w``, or ``weight_change``

    Supported entry formats:

    - Object with attributes ``t_`` and ``dw_``
    - Object with attributes ``t`` and ``dw``
    - Dictionary with keys ``'t'``/``'t_'`` and ``'dw'``/``'dw_'``
    - 2-tuple ``(t, dw)``

    Parameters
    ----------
    weight : float, optional
        Initial synaptic weight (dimensionless). Must satisfy sign consistency with ``Wmin``
        and ``Wmax``. Default: ``1.0``.
    delay : float, optional
        Dendritic propagation delay in milliseconds. Must be positive. Default: ``1.0``.
    delay_steps : int, optional
        Integer delay in simulation time steps for event delivery. Must be >= 1. Default: ``1``.
    x_bar : float, optional
        Initial presynaptic trace value (dimensionless). Typically initialized to ``0.0`` before
        any spikes. Default: ``0.0``.
    tau_x : float, optional
        Time constant for presynaptic trace exponential decay (milliseconds). Must be positive
        and non-zero. Controls the temporal window of LTP. Typical values: 10-20 ms. Default: ``15.0``.
    Wmin : float, optional
        Hard lower bound on synaptic weight (dimensionless). Must have same sign as ``weight``
        according to NEST's internal sign checks. Default: ``0.0``.
    Wmax : float, optional
        Hard upper bound on synaptic weight (dimensionless). Must have same sign as ``weight``
        according to NEST's internal sign checks. Default: ``100.0``.
    t_last_spike_ms : float, optional
        Timestamp of the most recent presynaptic spike (milliseconds). Initialized to ``0.0``
        before the first spike. Default: ``0.0``.
    name : str or None, optional
        Optional identifier for this connection instance. Default: ``None``.

    Parameter Mapping
    -----------------

    This table shows the correspondence between brainpy.state and NEST parameter names:

    =====================  =====================  =============  ===================
    brainpy.state          NEST                   Unit           Description
    =====================  =====================  =============  ===================
    ``weight``             ``weight``             (unitless)     Synaptic weight
    ``delay``              ``delay``              ms             Dendritic delay
    ``delay_steps``        (runtime)              steps          Event delivery delay
    ``x_bar``              ``x_bar``              (unitless)     Presynaptic trace
    ``tau_x``              ``tau_x``              ms             Presynaptic time constant
    ``Wmin``               ``Wmin``               (unitless)     Minimum weight
    ``Wmax``               ``Wmax``               (unitless)     Maximum weight
    ``t_last_spike_ms``    (internal state)       ms             Last spike timestamp
    =====================  =====================  =============  ===================

    Attributes
    ----------
    HAS_DELAY : bool
        Connection supports propagation delay. Always ``True``.
    IS_PRIMARY : bool
        Connection is a primary connection type. Always ``True``.
    REQUIRES_CLOPATH_ARCHIVING : bool
        Connection requires voltage trace archiving from postsynaptic neuron. Always ``True``.
    SUPPORTS_HPC : bool
        Model supports high-performance computing infrastructure. Always ``True``.
    SUPPORTS_LBL : bool
        Model supports label-based lookup. Always ``True``.
    SUPPORTS_WFR : bool
        Model supports waveform relaxation iteration. Always ``True``.

    Raises
    ------
    ValueError
        If ``weight``, ``Wmin``, and ``Wmax`` do not satisfy sign consistency constraints.
        NEST enforces: ``sign(weight) == sign(Wmin)`` and ``sign(weight) == sign(Wmax)``,
        where sign tests use different comparison operators for min vs max bounds.
    ValueError
        If ``delay`` <= 0 or ``delay_steps`` < 1.
    ValueError
        If any parameter is non-finite (NaN or Inf).
    ValueError
        If ``tau_x`` is zero (division by zero in trace updates).
    AttributeError
        If target neuron does not provide required ``get_LTP_history`` and ``get_LTD_value``
        methods during ``send()`` call.

    See Also
    --------
    aeif_psc_delta_clopath : Adaptive exponential IF neuron with Clopath archiving (NEST)
    hh_psc_alpha_clopath : Hodgkin-Huxley neuron with Clopath archiving (NEST)
    stdp_synapse : Traditional pair-based STDP synapse

    Notes
    -----
    **Implementation Details:**

    - All internal computations use 64-bit floating point (``float64``) to match NEST precision.
    - Precise sub-grid spike timing offsets are ignored; all spike times are treated as exact
      multiples of the simulation time step.
    - The update sequence strictly follows ``clopath_synapse::send()`` in NEST to ensure
      numerical equivalence.
    - Sign constraints use NEST's asymmetric comparison operators: ``Wmin`` uses ``>=`` vs ``<``
      while ``Wmax`` uses ``>`` vs ``<=``.

    **Biological Interpretation:**

    The Clopath rule captures key experimental observations:

    - LTP depends on presynaptic activity (spike trace :math:`\bar{x}`) and postsynaptic
      depolarization (voltage-derived LTP trace).
    - LTD depends on presynaptic spikes paired with postsynaptic voltage without strong
      depolarization.
    - The voltage dependence enables homeostatic regulation: neurons with high baseline firing
      rates have reduced LTP, preventing runaway excitation.
    - The model reproduces triplet STDP effects without explicit triplet terms.

    **Computational Considerations:**

    - Memory overhead scales with the number of LTP history entries archived by the postsynaptic
      neuron (typically bounded by a sliding time window).
    - For large fan-in networks, the LTP history query in step (a) may become a bottleneck.
      Consider using sparse indexing or binned histograms for postsynaptic traces.
    - The exponential decay calculations use ``math.exp`` for scalar operations. For vectorized
      implementations, replace with ``jax.numpy.exp`` or equivalent.

    References
    ----------
    .. [1] Clopath, C., Büsing, L., Vasilaki, E., & Gerstner, W. (2010). Connectivity reflects
           coding: a model of voltage-based STDP with homeostasis. *Nature Neuroscience*, 13(3),
           344-352. DOI: 10.1038/nn.2479
    .. [2] NEST Initiative (2024). NEST Simulator Documentation: clopath_synapse.
           https://nest-simulator.readthedocs.io/
    .. [3] NEST source code: ``models/clopath_synapse.h`` and ``models/clopath_synapse.cpp``.
           https://github.com/nest/nest-simulator

    Examples
    --------
    **Basic Usage:**

    Create a Clopath synapse with default parameters:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> synapse = bst.clopath_synapse(weight=0.5, tau_x=15.0, Wmin=0.0, Wmax=1.0)
        >>> synapse.get_status()
        {'weight': 0.5, 'tau_x': 15.0, 'Wmin': 0.0, 'Wmax': 1.0, ...}

    **Simulating Presynaptic Spike Train:**

    Assuming a postsynaptic neuron with Clopath archiving:

    .. code-block:: python

        >>> # Mock target neuron with required interface
        >>> class ClopathNeuron:
        ...     def get_ltp_history(self, t1, t2):
        ...         # Return LTP events in (t1, t2]
        ...         return [(10.5, 0.05), (12.3, 0.08)]  # (time_ms, dw)
        ...     def get_ltd_value(self, t):
        ...         # Return LTD amplitude at time t
        ...         return 0.02
        >>> target = ClopathNeuron()
        >>> synapse = bst.clopath_synapse(weight=1.0, tau_x=10.0, Wmin=0.0, Wmax=5.0)
        >>> # Process spike train
        >>> spike_times = [10.0, 20.0, 30.0]
        >>> events = synapse.simulate_pre_spike_train(spike_times, target)
        >>> print(f"Final weight: {synapse.weight:.3f}")
        Final weight: 1.123

    **Weight Evolution with Voltage-Dependent Plasticity:**

    .. code-block:: python

        >>> synapse = bst.clopath_synapse(weight=1.0, tau_x=15.0, Wmin=-2.0, Wmax=2.0)
        >>> # Simulate pairing protocol: pre before post (LTP)
        >>> for t in [10, 20, 30]:
        ...     event = synapse.send(t_spike_ms=t, target=target)
        >>> print(f"Weight after LTP protocol: {synapse.weight:.3f}")
        Weight after LTP protocol: 1.450

    **Sign Constraint Validation:**

    .. code-block:: python

        >>> # Valid: all same sign (positive)
        >>> synapse = bst.clopath_synapse(weight=1.0, Wmin=0.0, Wmax=5.0)
        >>> # Invalid: mixed signs
        >>> try:
        ...     synapse = bst.clopath_synapse(weight=1.0, Wmin=-1.0, Wmax=5.0)
        ... except ValueError as e:
        ...     print(e)
        Weight and Wmin must have same sign.
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    IS_PRIMARY = True
    REQUIRES_CLOPATH_ARCHIVING = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True
    SUPPORTS_WFR = True

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        x_bar: ArrayLike = 0.0,
        tau_x: ArrayLike = 15.0,
        Wmin: ArrayLike = 0.0,
        Wmax: ArrayLike = 100.0,
        t_last_spike_ms: ArrayLike = 0.0,
        name: str | None = None,
    ):
        self.name = name

        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay = self._validate_positive_delay(delay)
        self.delay_steps = self._validate_delay_steps(delay_steps)

        self.x_bar = self._to_float_scalar(x_bar, name='x_bar')
        self.tau_x = self._to_float_scalar(tau_x, name='tau_x')
        self.Wmin = self._to_float_scalar(Wmin, name='Wmin')
        self.Wmax = self._to_float_scalar(Wmax, name='Wmax')
        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

        self._check_weight_sign_constraints()

    @property
    def properties(self) -> dict[str, Any]:
        r"""Return dictionary of connection model properties and capabilities.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``'has_delay'`` (bool): Connection supports propagation delay
            - ``'is_primary'`` (bool): Connection is primary type
            - ``'requires_clopath_archiving'`` (bool): Requires voltage trace archiving
            - ``'supports_hpc'`` (bool): High-performance computing support
            - ``'supports_lbl'`` (bool): Label-based lookup support
            - ``'supports_wfr'`` (bool): Waveform relaxation support
        """
        return {
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'requires_clopath_archiving': self.REQUIRES_CLOPATH_ARCHIVING,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        r"""Retrieve current connection state and parameter values.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all connection parameters, state variables, and properties:

            - ``'weight'`` (float): Current synaptic weight
            - ``'delay'`` (float): Dendritic delay in milliseconds
            - ``'delay_steps'`` (int): Integer delay in simulation steps
            - ``'x_bar'`` (float): Current presynaptic trace value
            - ``'tau_x'`` (float): Presynaptic trace time constant (ms)
            - ``'Wmin'`` (float): Minimum weight bound
            - ``'Wmax'`` (float): Maximum weight bound
            - ``'t_last_spike_ms'`` (float): Last presynaptic spike time (ms)
            - ``'size_of'`` (int): Memory footprint in bytes
            - ``'has_delay'`` (bool): Delay support flag
            - ``'is_primary'`` (bool): Primary connection flag
            - ``'requires_clopath_archiving'`` (bool): Archiving requirement flag
            - ``'supports_hpc'`` (bool): HPC support flag
            - ``'supports_lbl'`` (bool): Label-based lookup flag
            - ``'supports_wfr'`` (bool): Waveform relaxation flag

        Notes
        -----
        This method provides NEST-compatible status retrieval. All values are returned as
        Python native types (float, int, bool) rather than NumPy arrays.
        """
        return {
            'weight': float(self.weight),
            'delay': float(self.delay),
            'delay_steps': int(self.delay_steps),
            'x_bar': float(self.x_bar),
            'tau_x': float(self.tau_x),
            'Wmin': float(self.Wmin),
            'Wmax': float(self.Wmax),
            't_last_spike_ms': float(self.t_last_spike_ms),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'requires_clopath_archiving': self.REQUIRES_CLOPATH_ARCHIVING,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update connection parameters and state variables.

        Parameters
        ----------
        status : dict[str, Any] or None, optional
            Dictionary of parameter name-value pairs to update. Supported keys: ``'weight'``,
            ``'delay'``, ``'delay_steps'``, ``'x_bar'``, ``'tau_x'``, ``'Wmin'``, ``'Wmax'``,
            ``'t_last_spike_ms'``. Default: ``None``.
        **kwargs
            Additional parameter updates as keyword arguments. These are merged with ``status``
            dictionary; keyword arguments take precedence.

        Raises
        ------
        ValueError
            If updated parameters violate sign consistency constraints (``weight``, ``Wmin``,
            ``Wmax`` must all have compatible signs).
        ValueError
            If ``delay`` <= 0 or ``delay_steps`` < 1.
        ValueError
            If any parameter value is non-finite (NaN or Inf).

        Notes
        -----
        This method provides NEST-compatible parameter setting. Sign constraints are re-checked
        after all updates are applied. If multiple parameters are updated together, validation
        occurs atomically after all changes.

        Examples
        --------
        Update single parameter:

        .. code-block:: python

            >>> synapse = bst.clopath_synapse(weight=1.0)
            >>> synapse.set_status(weight=0.5)
            >>> synapse.get_status()['weight']
            0.5

        Update multiple parameters:

        .. code-block:: python

            >>> synapse.set_status({'weight': 2.0, 'tau_x': 20.0})
            >>> synapse.set_status(Wmin=0.0, Wmax=5.0)
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
        if 'x_bar' in updates:
            self.x_bar = self._to_float_scalar(updates['x_bar'], name='x_bar')
        if 'tau_x' in updates:
            self.tau_x = self._to_float_scalar(updates['tau_x'], name='tau_x')
        if 'Wmin' in updates:
            self.Wmin = self._to_float_scalar(updates['Wmin'], name='Wmin')
        if 'Wmax' in updates:
            self.Wmax = self._to_float_scalar(updates['Wmax'], name='Wmax')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')

        self._check_weight_sign_constraints()

    def get(self, key: str = 'status'):
        r"""Retrieve connection status or specific parameter value.

        Parameters
        ----------
        key : str, optional
            Key to retrieve. Use ``'status'`` for full status dictionary, or specify a parameter
            name (e.g., ``'weight'``, ``'tau_x'``, ``'Wmin'``). Default: ``'status'``.

        Returns
        -------
        dict[str, Any] or float or int or bool
            If ``key == 'status'``, returns full status dictionary. Otherwise returns the
            requested parameter value with type matching the parameter (float, int, or bool).

        Raises
        ------
        KeyError
            If ``key`` is not ``'status'`` and does not match any parameter or property name
            in the status dictionary.

        Examples
        --------
        .. code-block:: python

            >>> synapse = bst.clopath_synapse(weight=1.5, tau_x=12.0)
            >>> synapse.get('weight')
            1.5
            >>> synapse.get('tau_x')
            12.0
            >>> status = synapse.get('status')
            >>> status['Wmax']
            100.0
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for clopath_synapse.get().')

    def set_weight(self, weight: ArrayLike):
        r"""Set synaptic weight value.

        Parameters
        ----------
        weight : float or array-like
            New synaptic weight value (scalar). Must be finite and satisfy sign consistency
            with ``Wmin`` and ``Wmax``.

        Raises
        ------
        ValueError
            If ``weight`` is non-scalar, non-finite, or violates sign constraints.

        Notes
        -----
        This is a convenience method equivalent to ``set_status(weight=...)``, but does not
        re-check sign constraints (assumes they were satisfied during initialization).
        """
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, delay: ArrayLike):
        r"""Set dendritic propagation delay.

        Parameters
        ----------
        delay : float or array-like
            New delay in milliseconds (scalar). Must be positive.

        Raises
        ------
        ValueError
            If ``delay`` is non-scalar, non-finite, or <= 0.
        """
        self.delay = self._validate_positive_delay(delay)

    def set_delay_steps(self, delay_steps: ArrayLike):
        r"""Set integer delay in simulation time steps.

        Parameters
        ----------
        delay_steps : int or array-like
            New delay in time steps (scalar integer). Must be >= 1.

        Raises
        ------
        ValueError
            If ``delay_steps`` is non-scalar, non-finite, non-integer, or < 1.
        """
        self.delay_steps = self._validate_delay_steps(delay_steps)

    def send(
        self,
        t_spike_ms: ArrayLike,
        target: Any,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay: ArrayLike | None = None,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        r"""Process one presynaptic spike with Clopath plasticity and return spike event payload.

        This method implements the core plasticity update sequence:

        1. Query postsynaptic LTP history in the interval since last presynaptic spike
        2. Apply facilitation (LTP) for each history entry using decayed presynaptic trace
        3. Apply depression (LTD) at the current effective spike time
        4. Update presynaptic trace and last spike timestamp
        5. Return spike event with updated weight

        Parameters
        ----------
        t_spike_ms : float or array-like
            Current presynaptic spike time in milliseconds (scalar). Must be finite.
        target : object
            Postsynaptic neuron or target object. Must provide ``get_LTP_history(t1, t2)``
            (or ``get_ltp_history``) and ``get_LTD_value(t)`` (or ``get_ltd_value``) methods.
        receptor_type : int or array-like, optional
            Receptor port index on target neuron (scalar integer). Default: ``0``.
        multiplicity : float or array-like, optional
            Spike event multiplicity (scalar, >= 0). Used for batch spike processing.
            Default: ``1.0``.
        delay : float or array-like or None, optional
            Override dendritic delay for this spike (milliseconds, scalar). If ``None``, uses
            connection's stored ``self.delay``. Default: ``None``.
        delay_steps : int or array-like or None, optional
            Override integer delay in time steps for this event (scalar). If ``None``, uses
            connection's stored ``self.delay_steps``. Default: ``None``.

        Returns
        -------
        dict[str, Any]
            Spike event payload dictionary with keys:

            - ``'weight'`` (float): Updated synaptic weight after plasticity
            - ``'delay'`` (float): Dendritic delay used (milliseconds)
            - ``'delay_steps'`` (int): Integer delay in time steps
            - ``'receptor_type'`` (int): Target receptor port index
            - ``'multiplicity'`` (float): Spike multiplicity
            - ``'t_spike_ms'`` (float): Presynaptic spike timestamp

        Raises
        ------
        ValueError
            If ``tau_x`` is zero (division by zero in exponential decay calculations).
        ValueError
            If any parameter is non-scalar or non-finite.
        ValueError
            If ``delay`` <= 0, ``delay_steps`` < 1, or ``multiplicity`` < 0.
        AttributeError
            If ``target`` does not provide required ``get_LTP_history`` and ``get_LTD_value``
            methods (or their lowercase variants).
        ValueError
            If any LTP history entry does not provide extractable time and weight change fields.

        Notes
        -----
        **Update Sequence Details:**

        Let :math:`t` = ``t_spike_ms``, :math:`d` = effective delay, :math:`t_\text{last}` =
        ``self.t_last_spike_ms``.

        *Step 1: LTP Application*

        For each LTP history entry :math:`(t_i, \text{dw}_i)` in :math:`(t_\text{last} - d, t - d]`:

        .. math::

            w \leftarrow \min(W_\text{max},\, w + \text{dw}_i \cdot \bar{x}
            \exp((t_\text{last} - (t_i + d)) / \tau_x))

        *Step 2: LTD Application*

        .. math::

            w \leftarrow \max(W_\text{min},\, w - \text{dw}_\text{LTD}(t - d))

        *Step 3: Presynaptic Trace Update*

        .. math::

            \bar{x} \leftarrow \bar{x} \exp((t_\text{last} - t) / \tau_x) + 1 / \tau_x

        *Step 4: Timestamp Update*

        .. math::

            t_\text{last} \leftarrow t

        **Side Effects**


        This method modifies connection state:

        - ``self.weight``: Updated by LTP and LTD
        - ``self.x_bar``: Updated with new spike contribution
        - ``self.t_last_spike_ms``: Set to current spike time

        **Performance Considerations:**

        The LTP history query dominates runtime for neurons with many incoming connections.
        Consider using bounded history buffers (sliding window) in the target neuron to limit
        the number of entries returned.

        Examples
        --------
        Process single spike:

        .. code-block:: python

            >>> class MockTarget:
            ...     def get_ltp_history(self, t1, t2):
            ...         return [(5.5, 0.1)]  # One LTP event
            ...     def get_ltd_value(self, t):
            ...         return 0.02
            >>> target = MockTarget()
            >>> synapse = bst.clopath_synapse(weight=1.0, tau_x=10.0, Wmin=0.0, Wmax=5.0)
            >>> event = synapse.send(t_spike_ms=10.0, target=target)
            >>> print(f"Updated weight: {event['weight']:.3f}")
            Updated weight: 1.023

        Process spike with custom delay:

        .. code-block:: python

            >>> event = synapse.send(t_spike_ms=20.0, target=target, delay=2.5)
            >>> print(f"Delay used: {event['delay']} ms")
            Delay used: 2.5 ms
        """
        t_spike = self._to_float_scalar(t_spike_ms, name='t_spike_ms')
        if self.tau_x == 0.0:
            raise ValueError('tau_x must be non-zero.')

        dendritic_delay = self.delay if delay is None else self._validate_positive_delay(delay)
        event_delay_steps = (
            self.delay_steps
            if delay_steps is None
            else self._validate_delay_steps(delay_steps)
        )

        ltp_entries = self._get_ltp_history(
            target,
            self.t_last_spike_ms - dendritic_delay,
            t_spike - dendritic_delay,
        )

        for entry in ltp_entries:
            t_hist, dw = self._extract_history_entry(entry)
            minus_dt = self.t_last_spike_ms - (t_hist + dendritic_delay)

            self.weight = self._facilitate(
                self.weight,
                dw,
                self.x_bar * math.exp(minus_dt / self.tau_x),
            )

        ltd_dw = self._get_ltd_value(target, t_spike - dendritic_delay)
        self.weight = self._depress(self.weight, ltd_dw)

        event = {
            'weight': float(self.weight),
            'delay': float(dendritic_delay),
            'delay_steps': int(event_delay_steps),
            'receptor_type': self._to_int_scalar(receptor_type, name='receptor_type'),
            'multiplicity': self._validate_multiplicity(multiplicity),
            't_spike_ms': float(t_spike),
        }

        self.x_bar = self.x_bar * math.exp((self.t_last_spike_ms - t_spike) / self.tau_x) + 1.0 / self.tau_x
        self.t_last_spike_ms = t_spike

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
        r"""Alias for ``send()`` method with identical semantics.

        This method provides an alternative name for spike event generation, maintaining
        compatibility with different naming conventions. All parameters and return values
        are identical to ``send()``.

        See Also
        --------
        send : Primary spike processing method with full documentation
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
        spike_times_ms: ArrayLike,
        target: Any,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay: ArrayLike | None = None,
        delay_steps: ArrayLike | None = None,
    ) -> list[dict[str, Any]]:
        r"""Process a sequence of presynaptic spikes and return event payloads for each.

        This method sequentially processes multiple presynaptic spikes, updating connection
        state (weight, presynaptic trace, last spike time) after each spike. The plasticity
        updates are cumulative: each spike's LTP/LTD application affects the weight seen by
        subsequent spikes.

        Parameters
        ----------
        spike_times_ms : array-like
            Array of presynaptic spike times in milliseconds. Values are converted to 1-D float64
            array and processed in order. Must contain finite values.
        target : object
            Postsynaptic neuron providing ``get_LTP_history`` and ``get_LTD_value`` methods.
        receptor_type : int or array-like, optional
            Receptor port index (scalar). Default: ``0``.
        multiplicity : float or array-like, optional
            Spike multiplicity for all events (scalar). Default: ``1.0``.
        delay : float or array-like or None, optional
            Dendritic delay override (milliseconds). If ``None``, uses ``self.delay``.
            Default: ``None``.
        delay_steps : int or array-like or None, optional
            Integer delay override (time steps). If ``None``, uses ``self.delay_steps``.
            Default: ``None``.

        Returns
        -------
        list[dict[str, Any]]
            List of spike event payloads, one per input spike time. Each event dictionary
            contains the same keys as returned by ``send()``: ``'weight'``, ``'delay'``,
            ``'delay_steps'``, ``'receptor_type'``, ``'multiplicity'``, ``'t_spike_ms'``.

        Raises
        ------
        ValueError
            If any spike time is non-finite or if ``tau_x`` is zero.
        AttributeError
            If ``target`` does not provide required methods.

        Notes
        -----
        **Ordering Effects:**

        Spike times are processed in array order (not necessarily sorted by time). For
        biologically realistic simulations, ensure ``spike_times_ms`` is sorted in ascending
        order. Out-of-order spikes may produce unphysical weight trajectories due to incorrect
        exponential decay calculations.

        **State Evolution:**

        After processing spike train ``[t1, t2, ..., tn]``, the connection state reflects:

        - ``self.weight``: Cumulative effect of all LTP/LTD updates
        - ``self.x_bar``: Presynaptic trace at time ``tn``
        - ``self.t_last_spike_ms``: Set to ``tn``

        **Memory Considerations:**

        The returned event list stores a separate dictionary for each spike. For very long
        spike trains (>10^6 spikes), consider processing in batches to reduce memory overhead.

        Examples
        --------
        Process spike train and track weight evolution:

        .. code-block:: python

            >>> class MockTarget:
            ...     def get_ltp_history(self, t1, t2):
            ...         # Return one LTP event per query interval
            ...         if t2 > t1:
            ...             return [((t1 + t2) / 2, 0.05)]
            ...         return []
            ...     def get_ltd_value(self, t):
            ...         return 0.01
            >>> target = MockTarget()
            >>> synapse = bst.clopath_synapse(weight=1.0, tau_x=10.0, Wmin=0.0, Wmax=3.0)
            >>> spike_times = [10.0, 20.0, 30.0, 40.0, 50.0]
            >>> events = synapse.simulate_pre_spike_train(spike_times, target)
            >>> weights = [evt['weight'] for evt in events]
            >>> print(f"Weight trajectory: {weights}")
            Weight trajectory: [1.045, 1.083, 1.115, 1.142, 1.165]

        Verify presynaptic trace evolution:

        .. code-block:: python

            >>> synapse = bst.clopath_synapse(weight=1.0, tau_x=15.0)
            >>> events = synapse.simulate_pre_spike_train([0.0, 15.0, 30.0], target)
            >>> print(f"Final x_bar: {synapse.x_bar:.3f}")
            Final x_bar: 0.091
        """
        dftype = brainstate.environ.dftype()
        times = np.asarray(u.math.asarray(spike_times_ms), dtype=dftype).reshape(-1)
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

    def _check_weight_sign_constraints(self):
        r"""Validate sign consistency between weight and bounds using NEST's exact sign tests.

        Raises
        ------
        ValueError
            If ``weight`` and ``Wmin`` have incompatible signs (using >= vs < comparison).
        ValueError
            If ``weight`` and ``Wmax`` have incompatible signs (using > vs <= comparison).

        Notes
        -----
        NEST uses asymmetric sign tests:

        - ``Wmin`` test: ``(weight >= 0) - (weight < 0)`` must equal ``(Wmin >= 0) - (Wmin < 0)``
        - ``Wmax`` test: ``(weight > 0) - (weight <= 0)`` must equal ``(Wmax > 0) - (Wmax <= 0)``

        This asymmetry means zero is treated differently for min vs max bounds. For example,
        ``weight=0.0``, ``Wmin=0.0``, ``Wmax=0.0`` is valid, but ``weight=0.0``, ``Wmin=-1.0``
        is invalid.
        """
        # Keep sign checks exactly as in NEST clopath_synapse::set_status.
        if self._sign_like_wmin(self.weight) != self._sign_like_wmin(self.Wmin):
            raise ValueError('Weight and Wmin must have same sign.')

        if self._sign_like_wmax(self.weight) != self._sign_like_wmax(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')

    @staticmethod
    def _sign_like_wmin(x: float) -> int:
        r"""Compute NEST-compatible sign test for Wmin comparison (>= 0 vs < 0).

        Returns -1 for negative, +1 for non-negative.
        """
        return int((x >= 0.0) - (x < 0.0))

    @staticmethod
    def _sign_like_wmax(x: float) -> int:
        r"""Compute NEST-compatible sign test for Wmax comparison (> 0 vs <= 0).

        Returns -1 for non-positive, +1 for positive.
        """
        return int((x > 0.0) - (x <= 0.0))

    def _depress(self, w: float, dw: float) -> float:
        r"""Apply LTD weight depression with hard lower bound clipping.

        Parameters
        ----------
        w : float
            Current weight.
        dw : float
            Depression magnitude (positive value reduces weight).

        Returns
        -------
        float
            Updated weight: ``max(Wmin, w - dw)``.
        """
        w_new = w - float(dw)
        return w_new if w_new > self.Wmin else self.Wmin

    def _facilitate(self, w: float, dw: float, x_trace: float) -> float:
        r"""Apply LTP weight facilitation with hard upper bound clipping.

        Parameters
        ----------
        w : float
            Current weight.
        dw : float
            Facilitation magnitude from postsynaptic LTP trace.
        x_trace : float
            Decayed presynaptic trace value at effective time.

        Returns
        -------
        float
            Updated weight: ``min(Wmax, w + dw * x_trace)``.
        """
        w_new = w + float(dw) * float(x_trace)
        return w_new if w_new < self.Wmax else self.Wmax

    def _get_ltp_history(self, target: Any, t1: float, t2: float):
        r"""Query postsynaptic LTP history entries in time interval (t1, t2].

        Parameters
        ----------
        target : object
            Postsynaptic neuron with ``get_LTP_history`` or ``get_ltp_history`` method.
        t1 : float
            Start time (exclusive) in milliseconds.
        t2 : float
            End time (inclusive) in milliseconds.

        Returns
        -------
        list or iterable
            LTP history entries in the interval. Returns empty list if method returns ``None``.

        Raises
        ------
        AttributeError
            If target does not provide ``get_LTP_history`` or ``get_ltp_history`` callable method.
        """
        fn = getattr(target, 'get_LTP_history', None)
        if fn is None:
            fn = getattr(target, 'get_ltp_history', None)
        if fn is None or not callable(fn):
            raise AttributeError(
                'Target must provide get_LTP_history(t1, t2) or get_ltp_history(t1, t2).'
            )
        history = fn(float(t1), float(t2))
        if history is None:
            return []
        return history

    def _get_ltd_value(self, target: Any, t: float) -> float:
        r"""Query postsynaptic LTD depression amplitude at given time.

        Parameters
        ----------
        target : object
            Postsynaptic neuron with ``get_LTD_value`` or ``get_ltd_value`` method.
        t : float
            Query time in milliseconds.

        Returns
        -------
        float
            LTD depression magnitude at time ``t``.

        Raises
        ------
        AttributeError
            If target does not provide ``get_LTD_value`` or ``get_ltd_value`` callable method.
        """
        fn = getattr(target, 'get_LTD_value', None)
        if fn is None:
            fn = getattr(target, 'get_ltd_value', None)
        if fn is None or not callable(fn):
            raise AttributeError(
                'Target must provide get_LTD_value(t) or get_ltd_value(t).'
            )
        return float(fn(float(t)))

    @staticmethod
    def _extract_history_entry(entry: Any) -> tuple[float, float]:
        r"""Extract time and weight change from LTP history entry.

        Supports multiple entry formats:

        - Dictionary: keys ``'t'``/``'t_'``/``'time'``/``'time_ms'`` and
          ``'dw'``/``'dw_'``/``'delta_w'``/``'weight_change'``
        - 2-tuple or list: ``(time, dw)``
        - Object: attributes ``t``/``t_`` and ``dw``/``dw_``

        Parameters
        ----------
        entry : dict or tuple or object
            LTP history entry from postsynaptic neuron.

        Returns
        -------
        tuple[float, float]
            Extracted ``(time_ms, dw)`` as floats.

        Raises
        ------
        ValueError
            If entry does not provide extractable time and weight change values.
        """
        t = None
        dw = None

        if isinstance(entry, dict):
            t = entry.get('t_', entry.get('t', entry.get('time_ms', entry.get('time', None))))
            dw = entry.get('dw_', entry.get('dw', entry.get('delta_w', entry.get('weight_change', None))))
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            t, dw = entry[0], entry[1]
        else:
            t = getattr(entry, 't_', getattr(entry, 't', None))
            dw = getattr(entry, 'dw_', getattr(entry, 'dw', None))

        if t is None or dw is None:
            raise ValueError('Each LTP history entry must provide both time and dw values.')

        return float(t), float(dw)

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        r"""Convert array-like value to scalar float with validation.

        Strips brainunit Quantity wrapper if present, converts to float64 array, and validates
        scalar shape and finite value.

        Parameters
        ----------
        value : array-like
            Input value (may be brainunit Quantity, NumPy array, JAX array, or Python scalar).
        name : str
            Parameter name for error messages.

        Returns
        -------
        float
            Validated scalar float value.

        Raises
        ------
        ValueError
            If value is not scalar (size != 1) or not finite (NaN or Inf).
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr[0])
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        return v

    @staticmethod
    def _to_int_scalar(value: ArrayLike, name: str) -> int:
        r"""Convert array-like value to scalar integer with validation.

        Strips brainunit Quantity wrapper if present, converts to float64, validates scalar shape
        and finite value, then rounds to nearest integer and checks for integer-valued input.

        Parameters
        ----------
        value : array-like
            Input value (may be brainunit Quantity, array, or scalar).
        name : str
            Parameter name for error messages.

        Returns
        -------
        int
            Validated integer value.

        Raises
        ------
        ValueError
            If value is not scalar, not finite, or not integer-valued (|value - round(value)| > 1e-12).
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr[0])
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        vr = int(round(v))
        if abs(v - vr) > 1e-12:
            raise ValueError(f'{name} must be integer-valued.')
        return vr

    def _validate_positive_delay(self, value: ArrayLike) -> float:
        r"""Validate and convert delay to positive float scalar.

        Parameters
        ----------
        value : array-like
            Delay value in milliseconds.

        Returns
        -------
        float
            Validated delay (must be > 0).

        Raises
        ------
        ValueError
            If delay <= 0, not scalar, or not finite.
        """
        d = self._to_float_scalar(value, name='delay')
        if d <= 0.0:
            raise ValueError('delay must be > 0.')
        return d

    def _validate_delay_steps(self, value: ArrayLike) -> int:
        r"""Validate and convert delay_steps to positive integer scalar.

        Parameters
        ----------
        value : array-like
            Integer delay in simulation time steps.

        Returns
        -------
        int
            Validated delay_steps (must be >= 1).

        Raises
        ------
        ValueError
            If delay_steps < 1, not scalar, not integer-valued, or not finite.
        """
        d = self._to_int_scalar(value, name='delay_steps')
        if d < 1:
            raise ValueError('delay_steps must be >= 1.')
        return d

    def _validate_multiplicity(self, value: ArrayLike) -> float:
        r"""Validate and convert multiplicity to non-negative float scalar.

        Parameters
        ----------
        value : array-like
            Spike multiplicity value.

        Returns
        -------
        float
            Validated multiplicity (must be >= 0).

        Raises
        ------
        ValueError
            If multiplicity < 0, not scalar, or not finite.
        """
        m = self._to_float_scalar(value, name='multiplicity')
        if m < 0.0:
            raise ValueError('multiplicity must be >= 0.')
        return m
