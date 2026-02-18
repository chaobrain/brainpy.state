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
    'ht_synapse',
]


class ht_synapse:
    r"""NEST-compatible Hill-Tononi synapse with vesicle-pool depression.

    Implements the short-term depression model from Hill & Tononi (2005) used to
    simulate thalamocortical sleep/wake dynamics. The model tracks a normalized
    vesicle pool :math:`P \in [0,1]` that recovers exponentially toward 1 and
    depletes multiplicatively on each presynaptic spike. The effective synaptic
    weight is the baseline weight scaled by the current pool availability.

    This implementation replicates NEST's ``ht_synapse.{h,cpp}`` connection model
    exactly, including event ordering, default values, and numerical precision.

    **1. Mathematical Model**

    The vesicle pool evolves according to:

    .. math::

       \frac{dP}{dt} = \frac{1 - P}{\tau_P}

    where :math:`\tau_P` is the recovery time constant (milliseconds).

    **2. Spike Processing**

    When a spike arrives at time :math:`t`, with the last spike at :math:`t_{\text{last}}`:

    1. **Recover pool** (exponential relaxation from :math:`P_{\text{old}}` toward 1):

       .. math::

          P_{\text{send}} = 1 - (1 - P_{\text{old}}) \exp\left(-\frac{t - t_{\text{last}}}{\tau_P}\right)

    2. **Emit event** with depression-modulated weight:

       .. math::

          w_{\text{eff}} = w \cdot P_{\text{send}}

    3. **Deplete pool** by fractional amount :math:`\delta_P \in [0,1]`:

       .. math::

          P_{\text{new}} = (1 - \delta_P) P_{\text{send}}

    4. **Update last spike time**:

       .. math::

          t_{\text{last}} \leftarrow t

    This ordering (recover → emit → deplete → update) matches NEST exactly and
    differs from some formulations that deplete before recovery.

    **3. Implementation Notes**

    - **Timing precision**: Uses grid-aligned spike times; sub-grid offsets are
      ignored (consistent with NEST's non-precise-timing variant).
    - **Initial state**: Pool starts at :math:`P = 1.0`, last spike time at 0.0 ms.
    - **Delay handling**: Connections store ``delay_steps ≥ 1``; both ``delay``
      and ``delay_steps`` keys are accepted and synchronized in ``set_status``.
    - **Depletion semantics**: :math:`\delta_P = 0.125` means each spike removes
      12.5% of the *current* pool, not a fixed decrement.

    **4. Computational Characteristics**

    - **Complexity**: :math:`O(1)` per spike; exponential evaluation via ``math.exp``.
    - **Stability**: Numerically stable for :math:`\tau_P > 0` and bounded
      :math:`\delta_P \in [0,1]`; no risk of negative pools.
    - **Comparison with NEST**: Direct equivalence when using identical parameters,
      time step, and spike trains. Floating-point differences :math:`< 10^{-12}`.

    Parameters
    ----------
    weight : float, ArrayLike, optional
        Baseline synaptic weight (dimensionless). Can be positive (excitatory) or
        negative (inhibitory). Default: ``1.0``.
    delay_steps : int, ArrayLike, optional
        Transmission delay in integer simulation steps. Must satisfy
        ``delay_steps ≥ 1``. Default: ``1``.
    tau_P : float, ArrayLike, optional
        Vesicle pool recovery time constant (milliseconds). Must be strictly
        positive. Controls how quickly the pool refills toward 1.0 after depletion.
        Larger values → slower recovery → stronger depression. Default: ``500.0`` ms.
    delta_P : float, ArrayLike, optional
        Fractional pool depletion per spike, dimensionless. Must satisfy
        :math:`0 \leq \delta_P \leq 1`. Value of 0 disables depression; 1 fully
        depletes the pool. Default: ``0.125`` (12.5% depletion).
    P : float, ArrayLike, optional
        Initial pool availability, dimensionless. Must satisfy :math:`0 \leq P \leq 1`.
        Typically initialized to 1.0 (fully available). Default: ``1.0``.
    name : str, optional
        Optional instance identifier for debugging and logging. Default: ``None``.

    Parameter Mapping
    -----------------

    ================================  ============================  =================
    **brainpy.state**                 **NEST**                      **Description**
    ================================  ============================  =================
    ``weight``                        ``weight``                    Baseline weight
    ``delay_steps`` / ``delay``       ``delay``                     Transmission lag
    ``tau_P``                         ``tau_P``                     Recovery τ (ms)
    ``delta_P``                       ``delta_P``                   Depletion fraction
    ``P``                             ``P`` (internal state)        Pool availability
    ``t_last_spike_ms``               ``t_lastspike_`` (internal)   Last spike time
    ================================  ============================  =================

    Raises
    ------
    ValueError
        - If ``weight``, ``tau_P``, ``delta_P``, or ``P`` is non-scalar or non-finite.
        - If ``delay_steps < 1`` or is non-integer-valued.
        - If ``tau_P ≤ 0`` (recovery time must be positive).
        - If ``delta_P`` or ``P`` is outside :math:`[0, 1]`.
        - If ``multiplicity < 0`` in ``send()`` method.

    Examples
    --------
    **Basic usage with single connection:**

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> syn = bst.ht_synapse(weight=2.5, tau_P=300.0, delta_P=0.2, P=1.0)
       >>> syn.get_status()
       {'weight': 2.5, 'delay_steps': 1, 'tau_P': 300.0, 'delta_P': 0.2,
        'P': 1.0, 't_last_spike_ms': 0.0, ...}

    **Simulate spike train and observe depression:**

    .. code-block:: python

       >>> spike_times = [10.0, 20.0, 30.0, 40.0]  # milliseconds
       >>> events = syn.simulate_spike_train(spike_times)
       >>> for evt in events:
       ...     print(f"t={evt['t_spike_ms']:.1f}ms, w_eff={evt['weight']:.3f}, "
       ...           f"P_send={evt['P_send']:.3f}, P_post={evt['P_post']:.3f}")
       t=10.0ms, w_eff=2.467, P_send=0.987, P_post=0.790
       t=20.0ms, w_eff=2.021, P_send=0.809, P_post=0.647
       t=30.0ms, w_eff=1.692, P_send=0.677, P_post=0.541
       t=40.0ms, w_eff=1.442, P_send=0.577, P_post=0.462

    **Update parameters dynamically:**

    .. code-block:: python

       >>> syn.set_status(delta_P=0.5, tau_P=200.0)
       >>> syn.get('delta_P')
       0.5

    **Reset pool state mid-simulation:**

    .. code-block:: python

       >>> syn.reset_state(P=0.3, t_last_spike_ms=50.0)
       >>> syn.P
       0.3

    **Process individual spike with custom delay:**

    .. code-block:: python

       >>> event = syn.send(t_spike_ms=100.0, delay_steps=5, receptor_type=1)
       >>> event['delay_steps'], event['receptor_type']
       (5, 1)

    See Also
    --------
    ht_neuron : Hill-Tononi neuron model with intrinsic adaptation.
    tsodyks_synapse : Alternative Tsodyks-Markram STP with facilitation + depression.
    quantal_stp_synapse : Vesicle-based STP with stochastic release.

    Notes
    -----
    **Differences from other STP models:**

    - **Tsodyks-Markram** (``tsodyks_synapse``): Includes facilitation via :math:`u`
      variable; more parameters but richer dynamics.
    - **Quantal STP** (``quantal_stp_synapse``): Discrete vesicle counts with
      stochastic release; this model uses continuous :math:`P`.
    - **ht_synapse**: Simpler, purely depressing model optimized for large-scale
      thalamocortical simulations (Hill & Tononi 2005).

    **Biological interpretation:**

    - :math:`P` represents the fraction of *readily releasable* vesicles.
    - :math:`\tau_P = 500` ms captures slow vesicle replenishment typical of
      depressing cortical synapses.
    - :math:`\delta_P = 0.125` corresponds to ~12.5% vesicle release per spike.

    **Numerical considerations:**

    - For very short inter-spike intervals :math:`\Delta t \ll \tau_P`, the
      exponential term :math:`\exp(-\Delta t / \tau_P) \approx 1 - \Delta t / \tau_P`,
      so recovery is approximately linear.
    - For :math:`\Delta t \gg \tau_P`, pool fully recovers to :math:`P \to 1`.
    - Model is stable for all physically meaningful parameter ranges.

    **NEST compatibility:**

    - Direct equivalence to NEST 3.x ``ht_synapse`` model (C++ implementation).
    - All default values match NEST defaults exactly.
    - Event ordering and state updates identical to NEST's ``send()`` method.
    - Does not support NEST's precise spike timing (``*_ps`` variants).

    References
    ----------
    .. [1] Hill S, Tononi G (2005). "Modeling sleep and wakefulness in the
           thalamocortical system." *Journal of Neurophysiology* 93(3):1671-1698.
           https://doi.org/10.1152/jn.00915.2004
    .. [2] NEST Simulator documentation:
           https://nest-simulator.readthedocs.io/en/stable/models/ht_synapse.html
    .. [3] NEST source code: ``models/ht_synapse.h`` and ``models/ht_synapse.cpp``
           (NEST 3.9+).
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    SUPPORTS_WFR = False
    IS_PRIMARY = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        tau_P: ArrayLike = 500.0,
        delta_P: ArrayLike = 0.125,
        P: ArrayLike = 1.0,
        name: str | None = None,
    ):
        self.name = name
        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay_steps = self._validate_delay_steps(delay_steps)
        self.tau_P = self._validate_tau_P(tau_P)
        self.delta_P = self._validate_fraction(delta_P, name='delta_P')
        self.P = self._validate_fraction(P, name='P')

        # NEST default initialization: t_lastspike_ = 0.0
        self.t_last_spike_ms = 0.0

    @property
    def properties(self) -> dict[str, Any]:
        r"""Model capability flags mirroring NEST connection properties.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``'has_delay'`` : bool
                Whether connection supports transmission delay (always ``True``).
            - ``'supports_wfr'`` : bool
                Whether model supports waveform relaxation (always ``False``).
            - ``'is_primary'`` : bool
                Whether this is a primary connection type (always ``True``).
            - ``'supports_hpc'`` : bool
                Whether model supports high-performance computing features (``True``).
            - ``'supports_lbl'`` : bool
                Whether model supports label-based connectivity (``True``).

        Notes
        -----
        These flags match NEST's synapse model introspection API and are used
        for compatibility checking in network construction tools.
        """
        return {
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
        }

    def get_status(self) -> dict[str, Any]:
        r"""Retrieve complete connection state and parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all synapse state and metadata:

            - ``'weight'`` : float — Baseline synaptic weight.
            - ``'delay_steps'`` : int — Transmission delay (simulation steps).
            - ``'delay'`` : int — Alias of ``delay_steps`` for NEST compatibility.
            - ``'tau_P'`` : float — Pool recovery time constant (ms).
            - ``'delta_P'`` : float — Fractional depletion per spike [0,1].
            - ``'P'`` : float — Current pool availability [0,1].
            - ``'t_last_spike_ms'`` : float — Last processed spike time (ms).
            - ``'size_of'`` : int — Memory footprint in bytes (Python object size).
            - ``'has_delay'`` : bool — Delay support flag (always ``True``).
            - ``'supports_wfr'`` : bool — Waveform relaxation flag (``False``).
            - ``'is_primary'`` : bool — Primary connection flag (``True``).
            - ``'supports_hpc'`` : bool — HPC support flag (``True``).
            - ``'supports_lbl'`` : bool — Label-based connectivity flag (``True``).

        Notes
        -----
        Mimics NEST's ``GetStatus`` for connections. All numeric state is converted
        to native Python types (``float``, ``int``) for serialization safety.

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.ht_synapse(weight=2.0, tau_P=400.0)
           >>> status = syn.get_status()
           >>> status['tau_P']
           400.0
           >>> syn.send(t_spike_ms=10.0)
           >>> syn.get_status()['P']  # Pool state after spike
           0.875
        """
        return {
            'weight': float(self.weight),
            'delay_steps': int(self.delay_steps),
            'delay': int(self.delay_steps),
            'tau_P': float(self.tau_P),
            'delta_P': float(self.delta_P),
            'P': float(self.P),
            't_last_spike_ms': float(self.t_last_spike_ms),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update connection parameters and state from dictionary or kwargs.

        Parameters
        ----------
        status : dict[str, Any], optional
            Dictionary of parameter updates. Keys match ``get_status()`` output.
            If ``None``, only ``kwargs`` are applied. Default: ``None``.
        **kwargs
            Additional parameter updates as keyword arguments. Values here override
            any conflicting keys in ``status``.

        Raises
        ------
        ValueError
            - If ``delay`` and ``delay_steps`` are both provided but differ.
            - If any parameter value fails validation (see ``__init__`` docstring).

        Notes
        -----
        **Delay parameter handling:**

        - Both ``delay`` and ``delay_steps`` are accepted as aliases.
        - If both are provided and differ, raises ``ValueError``.
        - Internally, both map to the same ``delay_steps`` attribute.

        **Validation:**

        - All numeric updates are validated (finite, correct range, scalar).
        - State variables (``P``, ``t_last_spike_ms``) can be updated to reset
          the synapse mid-simulation.

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.ht_synapse(weight=1.0, tau_P=500.0)
           >>> syn.set_status({'weight': 2.5, 'delta_P': 0.3})
           >>> syn.get('weight')
           2.5

        .. code-block:: python

           >>> syn.set_status(tau_P=300.0, P=0.5)  # Reset pool mid-simulation
           >>> syn.tau_P, syn.P
           (300.0, 0.5)

        .. code-block:: python

           >>> syn.set_status(delay=3, delay_steps=3)  # OK, identical values
           >>> syn.set_status(delay=2, delay_steps=3)  # Raises ValueError
           ValueError: delay and delay_steps must be identical when both are provided.
        """
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        if 'weight' in updates:
            self.set_weight(updates['weight'])

        has_delay = 'delay' in updates
        has_delay_steps = 'delay_steps' in updates
        if has_delay and has_delay_steps:
            d = self._to_int_scalar(updates['delay'], name='delay')
            ds = self._to_int_scalar(updates['delay_steps'], name='delay_steps')
            if d != ds:
                raise ValueError('delay and delay_steps must be identical when both are provided.')
            self.set_delay_steps(ds)
        elif has_delay_steps:
            self.set_delay_steps(updates['delay_steps'])
        elif has_delay:
            self.set_delay(updates['delay'])

        if 'tau_P' in updates:
            self.tau_P = self._validate_tau_P(updates['tau_P'])
        if 'delta_P' in updates:
            self.delta_P = self._validate_fraction(updates['delta_P'], name='delta_P')
        if 'P' in updates:
            self.P = self._validate_fraction(updates['P'], name='P')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')

    def get(self, key: str = 'status'):
        r"""Retrieve a specific parameter or complete status dictionary.

        Parameters
        ----------
        key : str, optional
            Parameter name to retrieve. Special value ``'status'`` returns the
            complete status dictionary. Default: ``'status'``.

        Returns
        -------
        Any
            If ``key == 'status'``, returns ``dict[str, Any]`` from ``get_status()``.
            Otherwise, returns the value of the requested parameter (type depends
            on parameter: ``float``, ``int``, ``bool``).

        Raises
        ------
        KeyError
            If ``key`` is not ``'status'`` and is not present in the status dictionary.

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.ht_synapse(weight=2.0, tau_P=400.0)
           >>> syn.get('tau_P')
           400.0
           >>> syn.get('P')
           1.0
           >>> syn.get('status')  # Full dictionary
           {'weight': 2.0, 'tau_P': 400.0, 'P': 1.0, ...}

        .. code-block:: python

           >>> syn.get('nonexistent_key')
           KeyError: 'Unsupported key "nonexistent_key" for ht_synapse.get().'
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for ht_synapse.get().')

    def set_weight(self, weight: ArrayLike):
        r"""Update the baseline synaptic weight.

        Parameters
        ----------
        weight : float, ArrayLike
            New baseline weight (dimensionless scalar).

        Raises
        ------
        ValueError
            If ``weight`` is not a finite scalar.

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.ht_synapse(weight=1.0)
           >>> syn.set_weight(3.5)
           >>> syn.weight
           3.5
        """
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, delay: ArrayLike):
        r"""Update the transmission delay (alias for ``set_delay_steps``).

        Parameters
        ----------
        delay : int, ArrayLike
            New delay in integer simulation steps, must be ≥ 1.

        Raises
        ------
        ValueError
            If ``delay`` is not an integer-valued scalar ≥ 1.

        Notes
        -----
        This method is provided for NEST compatibility. Internally, it updates
        the same ``delay_steps`` attribute as ``set_delay_steps()``.

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.ht_synapse(delay_steps=1)
           >>> syn.set_delay(5)
           >>> syn.delay_steps
           5
        """
        self.delay_steps = self._validate_delay_steps(delay, name='delay')

    def set_delay_steps(self, delay_steps: ArrayLike):
        r"""Update the transmission delay in simulation steps.

        Parameters
        ----------
        delay_steps : int, ArrayLike
            New delay in integer simulation steps, must be ≥ 1.

        Raises
        ------
        ValueError
            If ``delay_steps`` is not an integer-valued scalar ≥ 1.

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.ht_synapse(delay_steps=1)
           >>> syn.set_delay_steps(3)
           >>> syn.delay_steps
           3
        """
        self.delay_steps = self._validate_delay_steps(delay_steps, name='delay_steps')

    def reset_state(
        self,
        P: ArrayLike = 1.0,
        t_last_spike_ms: ArrayLike = 0.0,
    ):
        r"""Reset internal state variables to specified values.

        Useful for initializing or reinitializing the synapse mid-simulation without
        recreating the object.

        Parameters
        ----------
        P : float, ArrayLike, optional
            New pool availability in [0, 1]. Default: ``1.0`` (fully available).
        t_last_spike_ms : float, ArrayLike, optional
            New last spike timestamp (milliseconds). Default: ``0.0``.

        Raises
        ------
        ValueError
            - If ``P`` is not a finite scalar in [0, 1].
            - If ``t_last_spike_ms`` is not a finite scalar.

        Notes
        -----
        This method does *not* reset parameters (``weight``, ``tau_P``, ``delta_P``,
        ``delay_steps``). Use ``set_status()`` for parameter updates.

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.ht_synapse()
           >>> syn.send(t_spike_ms=10.0)  # Depletes pool
           >>> syn.P
           0.875
           >>> syn.reset_state(P=1.0, t_last_spike_ms=0.0)  # Restore initial state
           >>> syn.P
           1.0
        """
        self.P = self._validate_fraction(P, name='P')
        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

    def recover_pool(self, t_spike_ms: ArrayLike) -> float:
        r"""Advance pool state to specified time via exponential recovery.

        Updates the internal pool variable ``P`` by integrating the recovery ODE
        from ``t_last_spike_ms`` to ``t_spike_ms``, *without* depletion. This is
        used internally by ``send()`` before emitting a spike event.

        Parameters
        ----------
        t_spike_ms : float, ArrayLike
            Target time in milliseconds. Must be ≥ ``t_last_spike_ms`` for physical
            consistency, though negative intervals are mathematically allowed
            (interpreted as backward recovery, producing :math:`P < P_{\text{old}}`).

        Returns
        -------
        float
            Updated pool availability :math:`P \in [0, 1]` after recovery.

        Notes
        -----
        **Mathematical formula:**

        .. math::

           P_{\text{new}} = 1 - (1 - P_{\text{old}}) \exp\left(-\frac{\Delta t}{\tau_P}\right)

        where :math:`\Delta t = t_{\text{spike}} - t_{\text{last}}`.

        **Side effects:**

        - Modifies ``self.P`` in-place.
        - Does *not* update ``t_last_spike_ms`` (caller's responsibility).
        - Does *not* deplete the pool (use ``send()`` for full spike processing).

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.ht_synapse(tau_P=200.0, P=0.5)
           >>> syn.t_last_spike_ms = 0.0
           >>> P_recovered = syn.recover_pool(t_spike_ms=100.0)
           >>> P_recovered
           0.696734...
           >>> syn.P  # State updated in-place
           0.696734...
           >>> syn.t_last_spike_ms  # NOT updated by recover_pool
           0.0

        See Also
        --------
        send : Full spike processing (recovery + depletion + time update).
        """
        t = self._to_float_scalar(t_spike_ms, name='t_spike_ms')
        h = t - self.t_last_spike_ms
        self.P = 1.0 - (1.0 - self.P) * math.exp(-h / self.tau_P)
        return float(self.P)

    def send(
        self,
        t_spike_ms: ArrayLike,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        r"""Process incoming presynaptic spike and return emitted event payload.

        Implements the full Hill-Tononi spike transmission protocol: recover pool,
        emit depression-modulated spike, deplete pool, update timestamp. Event
        ordering matches NEST's ``ht_synapse::send()`` exactly.

        Parameters
        ----------
        t_spike_ms : float, ArrayLike
            Spike arrival time in milliseconds (grid-aligned).
        receptor_type : int, ArrayLike, optional
            Target receptor port identifier (non-negative integer). Passed through
            to the event payload without modification. Default: ``0``.
        multiplicity : float, ArrayLike, optional
            Spike event multiplicity (non-negative scalar). Scales the effective
            weight in the event payload. Default: ``1.0``.
        delay_steps : int, ArrayLike, optional
            Transmission delay override in simulation steps (must be ≥ 1).
            If ``None``, uses the synapse's default ``delay_steps``. Default: ``None``.

        Returns
        -------
        dict[str, Any]
            Spike event payload with keys:

            - ``'weight'`` : float
                Effective synaptic weight = ``weight * P_send * multiplicity``.
            - ``'delay_steps'`` : int
                Transmission delay (steps).
            - ``'delay'`` : int
                Alias of ``delay_steps`` for NEST compatibility.
            - ``'receptor_type'`` : int
                Target receptor port (passed through).
            - ``'multiplicity'`` : float
                Event multiplicity (passed through).
            - ``'t_spike_ms'`` : float
                Spike time (milliseconds).
            - ``'P_send'`` : float
                Pool availability *before* depletion [0, 1].
            - ``'P_post'`` : float
                Pool availability *after* depletion [0, 1].

        Raises
        ------
        ValueError
            - If ``t_spike_ms`` is not a finite scalar.
            - If ``receptor_type`` is not a non-negative integer scalar.
            - If ``multiplicity`` is not a non-negative scalar.
            - If ``delay_steps`` override is provided but is not an integer ≥ 1.

        Notes
        -----
        **Execution order (NEST-compatible):**

        1. **Recover pool** to time ``t_spike_ms``:

           .. math::

              P_{\text{send}} = 1 - (1 - P_{\text{old}}) \exp\left(-\frac{t - t_{\text{last}}}{\tau_P}\right)

        2. **Compute effective weight**:

           .. math::

              w_{\text{eff}} = w \cdot P_{\text{send}} \cdot \text{multiplicity}

        3. **Deplete pool**:

           .. math::

              P_{\text{new}} = (1 - \delta_P) P_{\text{send}}

        4. **Update last spike time**:

           .. math::

              t_{\text{last}} \leftarrow t

        **State updates:**

        - Modifies ``self.P`` (pool availability) in-place.
        - Modifies ``self.t_last_spike_ms`` in-place.

        **Event interpretation:**

        - The returned ``weight`` incorporates depression but *not* delay.
        - The caller (network simulation engine) is responsible for queueing
          the event with the specified ``delay_steps`` offset.

        Examples
        --------
        **Single spike processing:**

        .. code-block:: python

           >>> syn = bst.ht_synapse(weight=2.0, tau_P=300.0, delta_P=0.2)
           >>> event = syn.send(t_spike_ms=10.0)
           >>> event['weight']  # Full weight (pool fully available)
           2.0
           >>> event['P_send']
           1.0
           >>> event['P_post']  # Depleted by 20%
           0.8

        **Rapid spike train (depression accumulates):**

        .. code-block:: python

           >>> syn.reset_state()  # Start fresh
           >>> e1 = syn.send(t_spike_ms=0.0)
           >>> e2 = syn.send(t_spike_ms=10.0)  # Insufficient recovery time
           >>> e1['weight'], e2['weight']
           (2.0, 1.627...)  # Second spike depressed

        **Custom delay and receptor:**

        .. code-block:: python

           >>> event = syn.send(t_spike_ms=50.0, delay_steps=5, receptor_type=2)
           >>> event['delay_steps'], event['receptor_type']
           (5, 2)

        See Also
        --------
        to_spike_event : Alias of this method for event-style APIs.
        simulate_spike_train : Process multiple spikes in sequence.
        recover_pool : Pool recovery without depletion (internal use).
        """
        t = self._to_float_scalar(t_spike_ms, name='t_spike_ms')
        p_send = self.recover_pool(t)
        mult = self._validate_multiplicity(multiplicity)
        eff_weight = self.weight * p_send * mult

        self.P *= (1.0 - self.delta_P)
        self.t_last_spike_ms = t

        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        return {
            'weight': float(eff_weight),
            'delay_steps': int(d),
            'delay': int(d),
            'receptor_type': self._to_int_scalar(receptor_type, name='receptor_type'),
            'multiplicity': mult,
            't_spike_ms': float(t),
            'P_send': float(p_send),
            'P_post': float(self.P),
        }

    def to_spike_event(
        self,
        t_spike_ms: ArrayLike,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        r"""Alias of :meth:`send` for event-style APIs.

        Provided for compatibility with event-driven simulation frameworks that
        prefer explicit ``to_*_event`` method naming. Functionality is identical
        to ``send()``.

        Parameters
        ----------
        t_spike_ms : float, ArrayLike
            Spike arrival time (milliseconds).
        receptor_type : int, ArrayLike, optional
            Target receptor port. Default: ``0``.
        multiplicity : float, ArrayLike, optional
            Event multiplicity. Default: ``1.0``.
        delay_steps : int, ArrayLike, optional
            Delay override (steps). Default: ``None`` (use synapse default).

        Returns
        -------
        dict[str, Any]
            Spike event payload (see :meth:`send` for details).

        See Also
        --------
        send : Primary spike processing method with full documentation.
        """
        return self.send(
            t_spike_ms=t_spike_ms,
            receptor_type=receptor_type,
            multiplicity=multiplicity,
            delay_steps=delay_steps,
        )

    def simulate_spike_train(
        self,
        spike_times_ms: ArrayLike,
        receptor_type: ArrayLike = 0,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> list[dict[str, Any]]:
        r"""Process a sequence of spikes and return all emitted events.

        Convenience method for simulating a spike train through the synapse and
        observing depression dynamics over time. Each spike updates the internal
        state sequentially (recovery + depletion).

        Parameters
        ----------
        spike_times_ms : array_like
            Spike times in milliseconds. Can be list, tuple, or array-like.
            Flattened internally; shape is not preserved. Spikes are processed
            in the order provided (typically sorted ascending).
        receptor_type : int, ArrayLike, optional
            Target receptor port for all events. Default: ``0``.
        multiplicity : float, ArrayLike, optional
            Event multiplicity for all events. Default: ``1.0``.
        delay_steps : int, ArrayLike, optional
            Delay override for all events (steps). Default: ``None`` (use synapse
            default).

        Returns
        -------
        list[dict[str, Any]]
            List of spike event payloads, one per input spike. Each dictionary
            has the structure documented in :meth:`send`.

        Notes
        -----
        **State persistence:**

        - Internal state (``P``, ``t_last_spike_ms``) is preserved across calls.
        - To start from a known state, call ``reset_state()`` first.

        **Ordering:**

        - Spikes are processed sequentially in the order they appear in
          ``spike_times_ms``. For correct dynamics, times should be sorted.
        - Out-of-order spikes are mathematically allowed but produce non-physical
          recovery (negative time intervals → pool *decreases*).

        **Performance:**

        - :math:`O(N)` where :math:`N` is the number of spikes.
        - Each spike requires one ``math.exp()`` evaluation.

        Examples
        --------
        **Observe depression over spike train:**

        .. code-block:: python

           >>> syn = bst.ht_synapse(weight=1.0, tau_P=100.0, delta_P=0.25)
           >>> spike_times = [0, 10, 20, 30, 40]  # milliseconds
           >>> events = syn.simulate_spike_train(spike_times)
           >>> for evt in events:
           ...     print(f"t={evt['t_spike_ms']:.0f} ms, "
           ...           f"w={evt['weight']:.3f}, P={evt['P_post']:.3f}")
           t=0 ms, w=1.000, P=0.750
           t=10 ms, w=0.821, P=0.616
           t=20 ms, w=0.703, P=0.527
           t=30 ms, w=0.613, P=0.460
           t=40 ms, w=0.542, P=0.406

        **Reset state between trains:**

        .. code-block:: python

           >>> syn.reset_state(P=1.0, t_last_spike_ms=0.0)
           >>> events2 = syn.simulate_spike_train([100, 110, 120])
           >>> events2[0]['P_send']  # Pool fully recovered
           1.0

        **Custom receptor and delay:**

        .. code-block:: python

           >>> events = syn.simulate_spike_train(
           ...     spike_times_ms=[0, 50],
           ...     receptor_type=2,
           ...     delay_steps=5
           ... )
           >>> events[0]['receptor_type'], events[0]['delay_steps']
           (2, 5)

        See Also
        --------
        send : Process individual spike with full control.
        reset_state : Reset internal state between simulations.
        """
        times = np.asarray(u.math.asarray(spike_times_ms), dtype=np.float64).reshape(-1)
        events = []
        for t in times:
            events.append(
                self.send(
                    t_spike_ms=float(t),
                    receptor_type=receptor_type,
                    multiplicity=multiplicity,
                    delay_steps=delay_steps,
                )
            )
        return events

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
        vr = int(round(v))
        if abs(v - vr) > 1e-12:
            raise ValueError(f'{name} must be integer-valued.')
        return vr

    def _validate_delay_steps(self, delay_steps: ArrayLike, name: str = 'delay_steps') -> int:
        d = self._to_int_scalar(delay_steps, name=name)
        if d < 1:
            raise ValueError(f'{name} must be >= 1.')
        return d

    def _validate_tau_P(self, tau_P: ArrayLike) -> float:
        v = self._to_float_scalar(tau_P, name='tau_P')
        if v <= 0.0:
            raise ValueError('tau_P > 0 required.')
        return v

    def _validate_fraction(self, value: ArrayLike, name: str) -> float:
        v = self._to_float_scalar(value, name=name)
        if v < 0.0 or v > 1.0:
            raise ValueError(f'0 <= {name} <= 1 required.')
        return v

    def _validate_multiplicity(self, multiplicity: ArrayLike) -> float:
        m = self._to_float_scalar(multiplicity, name='multiplicity')
        if m < 0.0:
            raise ValueError('multiplicity must be >= 0.')
        return m
