"""Legacy imperative ``urbanczik_synapse`` (NEST event-driven ``send()`` port).

Preserved as the no-NEST reference oracle for the Urbanczik-Senn rule. The active
public :class:`brainpy.state.urbanczik_synapse` is now the frozen spec + pure
``update(state, ctx)`` rule kernel on
:class:`~brainpy_state._network._event_plastic.VoltageCoupledPlasticProj`
(see :mod:`brainpy_state._nest.urbanczik_synapse`); this module is retired from
the active path but kept (mirroring ``_legacy_clopath_synapse`` /
``_legacy_stdp_synapse``) so the faithful event-driven host-side trace stays
available and tested.
"""
import math
from typing import Any

import brainstate
import saiunit as u
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
    'urbanczik_synapse',
]


class urbanczik_synapse(NESTSynapse):
    r"""NEST-compatible ``urbanczik_synapse`` connection model.

    Plastic synapse implementing Urbanczik-Senn dendritic prediction error learning rule for
    supervised learning in multi-compartment neurons. This synapse requires target neurons
    that archive dendritic prediction errors (e.g., ``pp_cond_exp_mc_urbanczik``).

    **1. Mathematical Model**

    This implementation reproduces the connection-level semantics of NEST
    ``models/urbanczik_synapse.{h,cpp}``. The learning rule combines presynaptic spike traces
    with postsynaptic dendritic prediction errors to update synaptic weights through a
    low-pass filtered plasticity signal.

    **1.1. Presynaptic Traces**

    Two exponential traces track presynaptic spiking activity with different time constants:

    .. math::
       \tau_L^\mathrm{tr}(t) = \tau_L^\mathrm{tr}(t_{last}) \exp\left(\frac{t_{last}-t}{\tau_L}\right) + 1

    .. math::
       \tau_s^\mathrm{tr}(t) = \tau_s^\mathrm{tr}(t_{last}) \exp\left(\frac{t_{last}-t}{\tau_s}\right) + 1

    where :math:`t` is current spike time, :math:`t_{last}` is previous spike time,
    :math:`\tau_L = C_m / g_L` is membrane time constant, and :math:`\tau_s` is synaptic
    time constant (``tau_syn_ex`` for excitatory weights, ``tau_syn_in`` for inhibitory).

    **1.2. Plasticity Signal**

    For each postsynaptic dendritic prediction error entry :math:`(t_i, \Delta w_i)` in the
    history window :math:`(t_{last} - d, t - d]` (where :math:`d` is dendritic delay):

    .. math::
       \Pi_i = \left[\tau_L^\mathrm{tr}\exp\left(\frac{t_{last}-(t_i+d)}{\tau_L}\right)
               - \tau_s^\mathrm{tr}\exp\left(\frac{t_{last}-(t_i+d)}{\tau_s}\right)\right] \Delta w_i

    Two integrals accumulate plasticity contributions:

    .. math::
       \Pi_\mathrm{int} \leftarrow \Pi_\mathrm{int} + \sum_i \Pi_i

    .. math::
       \Pi_\mathrm{exp} \leftarrow \exp\left(\frac{t_{last}-t}{\tau_\Delta}\right)\Pi_\mathrm{exp}
                                    + \sum_i \exp\left(\frac{(t_i+d)-t}{\tau_\Delta}\right)\Pi_i

    where :math:`\tau_\Delta` is the low-pass filter time constant for weight changes.

    **1.3. Weight Update**

    The synaptic weight is updated using the filtered difference of integrals:

    .. math::
       w \leftarrow \mathrm{clip}\left(w_0 + \frac{15\,C_m\,\tau_s\,\eta}{g_L(\tau_L-\tau_s)}
                                        (\Pi_\mathrm{int} - \Pi_\mathrm{exp}), W_{min}, W_{max}\right)

    where :math:`w_0` is ``init_weight``, :math:`\eta` is learning rate, and :math:`C_m`, :math:`g_L`
    are membrane capacitance and leak conductance of the dendritic compartment.

    **2. NEST Implementation Fidelity**

    This class preserves NEST's exact send-ordering in ``urbanczik_synapse::send(...)``:

    1. Read archived history in :math:`(t_{last} - d, t - d]`
    2. Update :math:`\Pi_\mathrm{int}` and :math:`\Pi_\mathrm{exp}` integrals
    3. Compute new weight with clipping
    4. Emit spike event with updated weight
    5. Update :math:`\tau_L^\mathrm{tr}` and :math:`\tau_s^\mathrm{tr}` traces
    6. Set :math:`t_{last} = t`

    **3. Computational Considerations**

    - **Synaptic time constant selection**: Uses ``tau_syn_ex`` when ``weight > 0``, otherwise
      ``tau_syn_in``, matching NEST's current-weight-dependent branching
    - **Numerical precision**: Sub-grid timestamp offsets are ignored as in NEST
    - **Weight bounds**: Hard clipping to [Wmin, Wmax] after each update
    - **Sign constraints**: Weight, Wmin, and Wmax must all share the same sign (enforced at init
      and status updates)

    **4. Target Neuron Requirements**

    The target neuron must implement the Urbanczik archiving interface:

    - ``get_urbanczik_history(t1, t2, comp)``: Returns prediction error entries in :math:`(t1, t2]`
      for compartment ``comp`` (default 1 for dendritic)
    - ``get_g_L(comp)``, ``get_tau_L(comp)`` or ``get_C_m(comp)/get_g_L(comp)``
    - ``get_C_m(comp)``, ``get_tau_syn_ex(comp)``, ``get_tau_syn_in(comp)``

    History entries support multiple formats: objects with ``t_``/``dw_`` attributes (NEST-style),
    objects with ``t``/``dw`` attributes, dicts with those keys, or 2-tuples ``(t, dw)``.

    Parameters
    ----------
    weight : float or ArrayLike, optional
        Initial synaptic weight (dimensionless). Must share sign with Wmin and Wmax.
        Default: ``1.0``
    delay : float or ArrayLike, optional
        Dendritic delay in milliseconds used for history lookup. Must be ``> 0``.
        Default: ``1.0`` ms
    delay_steps : int or ArrayLike, optional
        Event delivery delay in simulation time steps. Must be ``>= 1``.
        Default: ``1``
    tau_Delta : float or ArrayLike, optional
        Time constant in milliseconds for low-pass filtering of weight changes. Controls
        the temporal smoothing of plasticity signals. Larger values produce slower, more
        stable learning. Default: ``100.0`` ms
    eta : float or ArrayLike, optional
        Learning rate (dimensionless). Scales the magnitude of weight updates. Typical range
        0.01–0.1 for cortical models. Default: ``0.07``
    Wmin : float or ArrayLike, optional
        Lower bound of synaptic weight (hard clipping). Must share sign with weight and Wmax.
        Default: ``0.0``
    Wmax : float or ArrayLike, optional
        Upper bound of synaptic weight (hard clipping). Must share sign with weight and Wmin.
        Default: ``100.0``
    PI_integral : float or ArrayLike, optional
        Initial value of unfiltered accumulated plasticity integral :math:`\Pi_\mathrm{int}`.
        Default: ``0.0``
    PI_exp_integral : float or ArrayLike, optional
        Initial value of exponentially filtered plasticity integral :math:`\Pi_\mathrm{exp}`.
        Default: ``0.0``
    tau_L_trace : float or ArrayLike, optional
        Initial state of :math:`\tau_L` presynaptic trace. Default: ``0.0``
    tau_s_trace : float or ArrayLike, optional
        Initial state of :math:`\tau_s` presynaptic trace. Default: ``0.0``
    t_last_spike_ms : float or ArrayLike, optional
        Last presynaptic spike time in milliseconds. Default: ``-1.0`` (no previous spike)
    name : str, optional
        Instance name for debugging and logging. Default: ``None``

    Parameter Mapping
    -----------------
    NEST parameter mappings to this implementation:

    =============================  =========================================================
    NEST Parameter                 brainpy.state Attribute
    =============================  =========================================================
    ``weight``                     ``weight`` (current synaptic weight)
    ``delay``                      ``delay`` (dendritic delay for history)
    ``tau_Delta``                  ``tau_Delta`` (low-pass time constant)
    ``eta``                        ``eta`` (learning rate)
    ``Wmin``                       ``Wmin`` (lower weight bound)
    ``Wmax``                       ``Wmax`` (upper weight bound)
    ``init_weight``                ``init_weight`` (baseline weight for updates)
    ``receptor_type``              passed per spike event
    ``t_lastspike``                ``t_last_spike_ms``
    =============================  =========================================================

    Raises
    ------
    ValueError
        If ``delay`` is not positive, ``delay_steps < 1``, or weight/Wmin/Wmax have
        inconsistent signs
    AttributeError
        If target neuron does not implement required Urbanczik archiving interface methods

    Notes
    -----
    - **set_status() behavior**: Following NEST, ``set_status()`` always resets ``init_weight``
      to the current ``weight`` unless explicitly provided in the status dict
    - **Multiplicity**: Spike multiplicity is validated but not used in plasticity computation
      (NEST compatibility)
    - **Sub-grid timing**: Precise spike time offsets within a time step are ignored in this
      plasticity rule (consistent with NEST implementation)

    Examples
    --------
    Basic synapse creation and spike processing:

    .. code-block:: python

       >>> from brainpy import state as bp
       >>> # Create synapse with moderate learning rate
       >>> syn = bp.urbanczik_synapse(
       ...     weight=0.5,
       ...     delay=1.0,
       ...     tau_Delta=80.0,
       ...     eta=0.05,
       ...     Wmin=0.0,
       ...     Wmax=10.0
       ... )
       >>>
       >>> # Check initial status
       >>> status = syn.get_status()
       >>> print(f"Initial weight: {status['weight']}")
       Initial weight: 0.5
       >>> print(f"Learning rate: {status['eta']}")
       Learning rate: 0.05

    Processing spike trains with mock target neuron:

    .. code-block:: python

       >>> class MockUrbanczikNeuron:
       ...     def __init__(self):
       ...         self.history = []
       ...
       ...     def get_urbanczik_history(self, t1, t2, comp):
       ...         # Return prediction errors in (t1, t2]
       ...         return [(t, dw) for t, dw in self.history if t1 < t <= t2]
       ...
       ...     def get_g_L(self, comp): return 10.0  # nS
       ...     def get_tau_L(self, comp): return 20.0  # ms
       ...     def get_C_m(self, comp): return 200.0  # pF
       ...     def get_tau_syn_ex(self, comp): return 2.0  # ms
       ...     def get_tau_syn_in(self, comp): return 5.0  # ms
       >>>
       >>> target = MockUrbanczikNeuron()
       >>>
       >>> # Simulate presynaptic spike at t=10 ms
       >>> event = syn.send(t_spike_ms=10.0, target=target)
       >>> print(f"Weight after spike: {event['weight']:.3f}")
       Weight after spike: 0.500
       >>>
       >>> # Add dendritic prediction error and process another spike
       >>> target.history.append((8.0, 0.1))  # (time_ms, delta_w)
       >>> event = syn.send(t_spike_ms=20.0, target=target)
       >>> print(f"Weight after learning: {event['weight']:.3f}")
       Weight after learning: 0.503

    Weight bound enforcement:

    .. code-block:: python

       >>> syn = bp.urbanczik_synapse(weight=5.0, Wmin=0.0, Wmax=10.0)
       >>> syn.set_status({'weight': 12.0})  # Exceeds Wmax
       >>> print(syn.get('weight'))
       10.0
       >>> syn.set_status({'weight': -1.0})  # Violates sign constraint
       Traceback (most recent call last):
       ValueError: Weight and Wmax must have same sign.

    References
    ----------
    .. [1] Urbanczik R, Senn W (2014). Learning by the dendritic prediction of somatic spiking.
           *Neuron* 81(3):521-528. DOI: 10.1016/j.neuron.2013.11.030
    .. [2] Jordan J, Sacramento J, Wybo WAM, et al. (2021). Conductance-based dendrites perform
           reliability-weighted opinion pooling. *arXiv* 2109.02040.
    .. [3] NEST source: ``models/urbanczik_synapse.h`` and ``models/urbanczik_synapse.cpp``
           (NEST Simulator version 3.9+)

    See Also
    --------
    pp_cond_exp_mc_urbanczik : Multi-compartment neuron supporting Urbanczik archiving
    stdp_synapse : Classical spike-timing dependent plasticity synapse
    """

    # Retired from the public path; the spec+rule rebuild now owns
    # ``brainpy.state.urbanczik_synapse``.
    __module__ = 'brainpy_state._nest._legacy_urbanczik_synapse'

    HAS_DELAY = True
    IS_PRIMARY = True
    REQUIRES_URBANCZIK_ARCHIVING = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True
    SUPPORTS_WFR = True

    DENDRITIC_COMPARTMENT = 1

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        tau_Delta: ArrayLike = 100.0,
        eta: ArrayLike = 0.07,
        Wmin: ArrayLike = 0.0,
        Wmax: ArrayLike = 100.0,
        PI_integral: ArrayLike = 0.0,
        PI_exp_integral: ArrayLike = 0.0,
        tau_L_trace: ArrayLike = 0.0,
        tau_s_trace: ArrayLike = 0.0,
        t_last_spike_ms: ArrayLike = -1.0,
        name: str | None = None,
    ):
        super().__init__(in_size=1, name=name)

        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay = self._validate_positive_delay(delay)
        self.delay_steps = self._validate_delay_steps(delay_steps)

        self.tau_Delta = self._to_float_scalar(tau_Delta, name='tau_Delta')
        self.eta = self._to_float_scalar(eta, name='eta')
        self.Wmin = self._to_float_scalar(Wmin, name='Wmin')
        self.Wmax = self._to_float_scalar(Wmax, name='Wmax')

        self.PI_integral = self._to_float_scalar(PI_integral, name='PI_integral')
        self.PI_exp_integral = self._to_float_scalar(PI_exp_integral, name='PI_exp_integral')
        self.tau_L_trace = self._to_float_scalar(tau_L_trace, name='tau_L_trace')
        self.tau_s_trace = self._to_float_scalar(tau_s_trace, name='tau_s_trace')
        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

        # NEST initializes init_weight_ from weight_.
        self.init_weight = float(self.weight)

        self._check_weight_sign_constraints()

    @property
    def properties(self) -> dict[str, Any]:
        r"""Return synapse capability flags.

        Returns
        -------
        dict[str, Any]
            Dictionary with boolean flags:
            - ``has_delay``: Connection supports transmission delays (always True)
            - ``is_primary``: This is a primary connection type (always True)
            - ``requires_urbanczik_archiving``: Target must implement Urbanczik history (always True)
            - ``supports_hpc``: Compatible with high-performance computing features (always True)
            - ``supports_lbl``: Supports local branching levels in dendritic trees (always True)
            - ``supports_wfr``: Supports waveform relaxation methods (always True)

        Notes
        -----
        These flags match NEST synapse property conventions for integration with NEST-compatible
        simulation infrastructure.
        """
        return {
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'requires_urbanczik_archiving': self.REQUIRES_URBANCZIK_ARCHIVING,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        r"""Return current synapse state and parameters.

        Returns
        -------
        dict[str, Any]
            Complete synapse state dictionary. Keys include:
            ``weight`` (float), ``delay`` (float), ``delay_steps`` (int),
            ``tau_Delta`` (float), ``eta`` (float), ``Wmin`` (float),
            ``Wmax`` (float), ``init_weight`` (float), ``PI_integral`` (float),
            ``PI_exp_integral`` (float), ``tau_L_trace`` (float),
            ``tau_s_trace`` (float), ``t_last_spike_ms`` (float),
            ``size_of`` (int), and capability flags
            (``has_delay``, ``is_primary``, etc.).

        Notes
        -----
        Compatible with NEST ``GetStatus()`` semantics. All floating-point values are
        guaranteed finite (no NaN or infinity).

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.urbanczik_synapse(weight=2.5, eta=0.08)
           >>> status = syn.get_status()
           >>> print(f"Weight: {status['weight']}, Learning rate: {status['eta']}")
           Weight: 2.5, Learning rate: 0.08
        """
        return {
            'weight': float(self.weight),
            'delay': float(self.delay),
            'delay_steps': int(self.delay_steps),
            'tau_Delta': float(self.tau_Delta),
            'eta': float(self.eta),
            'Wmin': float(self.Wmin),
            'Wmax': float(self.Wmax),
            'init_weight': float(self.init_weight),
            'PI_integral': float(self.PI_integral),
            'PI_exp_integral': float(self.PI_exp_integral),
            'tau_L_trace': float(self.tau_L_trace),
            'tau_s_trace': float(self.tau_s_trace),
            't_last_spike_ms': float(self.t_last_spike_ms),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'requires_urbanczik_archiving': self.REQUIRES_URBANCZIK_ARCHIVING,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update synapse parameters and state.

        Parameters
        ----------
        status : dict[str, Any], optional
            Dictionary of parameter updates. Keys match those returned by ``get_status()``.
        **kwargs
            Additional parameter updates as keyword arguments. These override any values
            in ``status`` dict if both are provided.

        Raises
        ------
        ValueError
            If updated parameters violate sign constraints (weight, Wmin, Wmax must share sign),
            if delay is not positive, if delay_steps < 1, or if any value is non-finite.

        Notes
        -----
        - **NEST compatibility**: Following NEST ``SetStatus()`` semantics, ``init_weight``
          is automatically reset to the current ``weight`` after updates unless explicitly
          provided in the update dict
        - All updatable parameters: ``weight``, ``delay``, ``delay_steps``, ``tau_Delta``,
          ``eta``, ``Wmin``, ``Wmax``, ``PI_integral``, ``PI_exp_integral``, ``tau_L_trace``,
          ``tau_s_trace``, ``t_last_spike_ms``, ``init_weight``
        - Sign constraint validation occurs after all updates are applied

        Examples
        --------
        Update single parameter:

        .. code-block:: python

           >>> syn = bp.urbanczik_synapse(weight=1.0)
           >>> syn.set_status(eta=0.1)
           >>> print(syn.get('eta'))
           0.1

        Batch update with dict:

        .. code-block:: python

           >>> syn.set_status({'weight': 2.0, 'tau_Delta': 120.0})
           >>> status = syn.get_status()
           >>> print(f"Weight: {status['weight']}, tau_Delta: {status['tau_Delta']}")
           Weight: 2.0, tau_Delta: 120.0

        Keyword arguments override dict values:

        .. code-block:: python

           >>> syn.set_status({'eta': 0.05}, eta=0.08)
           >>> print(syn.get('eta'))
           0.08
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
        if 'tau_Delta' in updates:
            self.tau_Delta = self._to_float_scalar(updates['tau_Delta'], name='tau_Delta')
        if 'eta' in updates:
            self.eta = self._to_float_scalar(updates['eta'], name='eta')
        if 'Wmin' in updates:
            self.Wmin = self._to_float_scalar(updates['Wmin'], name='Wmin')
        if 'Wmax' in updates:
            self.Wmax = self._to_float_scalar(updates['Wmax'], name='Wmax')
        if 'PI_integral' in updates:
            self.PI_integral = self._to_float_scalar(updates['PI_integral'], name='PI_integral')
        if 'PI_exp_integral' in updates:
            self.PI_exp_integral = self._to_float_scalar(updates['PI_exp_integral'], name='PI_exp_integral')
        if 'tau_L_trace' in updates:
            self.tau_L_trace = self._to_float_scalar(updates['tau_L_trace'], name='tau_L_trace')
        if 'tau_s_trace' in updates:
            self.tau_s_trace = self._to_float_scalar(updates['tau_s_trace'], name='tau_s_trace')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')

        if 'init_weight' in updates:
            self.init_weight = self._to_float_scalar(updates['init_weight'], name='init_weight')
        else:
            # NEST set_status() always syncs init_weight_ to current weight_.
            self.init_weight = float(self.weight)

        self._check_weight_sign_constraints()

    def get(self, key: str = 'status'):
        r"""Retrieve synapse parameter or full status.

        Parameters
        ----------
        key : str, optional
            Parameter name or ``'status'`` for complete state dict. Valid keys match
            those in ``get_status()`` return dict. Default: ``'status'``

        Returns
        -------
        Any
            If ``key='status'``, returns full status dict. Otherwise returns scalar value
            of requested parameter.

        Raises
        ------
        KeyError
            If ``key`` is not a recognized parameter name.

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.urbanczik_synapse(weight=1.5, eta=0.06)
           >>> syn.get('weight')
           1.5
           >>> syn.get('eta')
           0.06
           >>> full_status = syn.get('status')
           >>> print(type(full_status))
           <class 'dict'>
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for urbanczik_synapse.get().')

    def set_weight(self, weight: ArrayLike):
        r"""Update synaptic weight with sign constraint validation.

        Parameters
        ----------
        weight : float or ArrayLike
            New synaptic weight. Must be finite scalar and share sign with Wmin/Wmax.

        Raises
        ------
        ValueError
            If weight violates sign constraints or is non-finite/non-scalar.
        """
        self.weight = self._to_float_scalar(weight, name='weight')
        self._check_weight_sign_constraints()

    def set_delay(self, delay: ArrayLike):
        r"""Update dendritic delay.

        Parameters
        ----------
        delay : float or ArrayLike
            New dendritic delay in milliseconds. Must be ``> 0``.

        Raises
        ------
        ValueError
            If delay is not positive, non-finite, or non-scalar.
        """
        self.delay = self._validate_positive_delay(delay)

    def set_delay_steps(self, delay_steps: ArrayLike):
        r"""Update event delivery delay in time steps.

        Parameters
        ----------
        delay_steps : int or ArrayLike
            New event delivery delay. Must be ``>= 1``.

        Raises
        ------
        ValueError
            If delay_steps is less than 1 or not an integer value.
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
        r"""Process presynaptic spike, update weight via dendritic prediction errors, and emit event.

        This method implements the core Urbanczik-Senn plasticity computation. It retrieves
        postsynaptic prediction error history from the target neuron, updates presynaptic traces
        and plasticity integrals, computes new synaptic weight, and returns spike event payload.

        **Computation Order (NEST-exact)**:

        1. Query target's ``get_urbanczik_history()`` for entries in :math:`(t_{last} - d, t - d]`
        2. For each history entry, compute :math:`\Pi_i` using current trace states
        3. Update :math:`\Pi_\mathrm{int}` and :math:`\Pi_\mathrm{exp}` accumulators
        4. Compute new weight with clipping to [Wmin, Wmax]
        5. Create spike event dict with updated weight
        6. Update :math:`\tau_L^\mathrm{tr}` and :math:`\tau_s^\mathrm{tr}` traces
        7. Set :math:`t_{last} = t`

        Parameters
        ----------
        t_spike_ms : float or ArrayLike
            Presynaptic spike time in milliseconds. Must be finite scalar.
        target : Any
            Postsynaptic neuron implementing Urbanczik archiving interface. Must provide:
            ``get_urbanczik_history(t1, t2, comp)``, ``get_g_L(comp)``, ``get_tau_L(comp)``
            (or ``get_C_m(comp)``), ``get_tau_syn_ex(comp)``, ``get_tau_syn_in(comp)``.
        receptor_type : int or ArrayLike, optional
            Receptor channel index on target neuron. Default: ``0``
        multiplicity : float or ArrayLike, optional
            Spike event multiplicity (validated but not used in plasticity). Must be ``>= 0``.
            Default: ``1.0``
        delay : float or ArrayLike, optional
            Override dendritic delay for this spike (milliseconds, must be ``> 0``).
            If ``None``, uses ``self.delay``. Default: ``None``
        delay_steps : int or ArrayLike, optional
            Override event delivery delay for this spike (steps, must be ``>= 1``).
            If ``None``, uses ``self.delay_steps``. Default: ``None``

        Returns
        -------
        dict[str, Any]
            Spike event dictionary. Keys: ``weight`` (float, updated synaptic weight),
            ``delay`` (float, dendritic delay in ms), ``delay_steps`` (int),
            ``receptor_type`` (int), ``multiplicity`` (float),
            ``t_spike_ms`` (float, spike time in ms),
            ``tau_s_ms`` (float, synaptic time constant used),
            ``PI_integral`` (float), ``PI_exp_integral`` (float),
            ``tau_L_trace_post`` (float), ``tau_s_trace_post`` (float).

        Raises
        ------
        AttributeError
            If target does not implement required Urbanczik archiving interface methods.
        ValueError
            If any parameter is non-finite, delay is not positive, delay_steps < 1,
            or multiplicity < 0.

        Notes
        -----
        - **Synaptic time constant selection**: Uses ``tau_syn_ex`` if current ``weight > 0``,
          otherwise ``tau_syn_in``. This branching matches NEST's current-weight-dependent logic.
        - **Sub-grid precision**: Sub-timestep spike offsets are ignored in plasticity computation
          (NEST compatibility).
        - **History window**: Query interval :math:`(t_{last} - d, t - d]` is open on left, closed
          on right, matching NEST's ``get_history()`` semantics.
        - **Multiplicity**: Validated but not incorporated into weight update (NEST behavior).

        Examples
        --------
        Process single spike:

        .. code-block:: python

           >>> syn = bp.urbanczik_synapse(weight=1.0, eta=0.05)
           >>> event = syn.send(t_spike_ms=10.0, target=mock_neuron)
           >>> print(f"Updated weight: {event['weight']:.3f}")
           Updated weight: 1.003

        Override delay for specific spike:

        .. code-block:: python

           >>> event = syn.send(t_spike_ms=20.0, target=mock_neuron, delay=2.0)
           >>> print(f"Delay used: {event['delay']} ms")
           Delay used: 2.0 ms

        Access trace states post-update:

        .. code-block:: python

           >>> event = syn.send(t_spike_ms=30.0, target=mock_neuron)
           >>> print(f"tau_L trace: {event['tau_L_trace_post']:.3f}")
           tau_L trace: 1.105
        """
        t_spike = self._to_float_scalar(t_spike_ms, name='t_spike_ms')

        dendritic_delay = self.delay if delay is None else self._validate_positive_delay(delay)
        event_delay_steps = (
            self.delay_steps
            if delay_steps is None
            else self._validate_delay_steps(delay_steps)
        )

        comp = self.DENDRITIC_COMPARTMENT
        history_entries = self._get_urbanczik_history(
            target,
            self.t_last_spike_ms - dendritic_delay,
            t_spike - dendritic_delay,
            comp=comp,
        )

        g_L = self._get_compartment_value(target, ['get_g_L', 'get_g_l'], comp=comp, field='g_L')
        tau_L = self._get_tau_L(target, comp=comp)
        C_m = self._get_compartment_value(target, ['get_C_m', 'get_c_m'], comp=comp, field='C_m')
        tau_syn_ex = self._get_compartment_value(
            target,
            ['get_tau_syn_ex', 'get_tau_syn_exc'],
            comp=comp,
            field='tau_syn_ex',
        )
        tau_syn_in = self._get_compartment_value(
            target,
            ['get_tau_syn_in', 'get_tau_syn_inh'],
            comp=comp,
            field='tau_syn_in',
        )
        tau_s = tau_syn_ex if self.weight > 0.0 else tau_syn_in

        dPI_exp_integral = 0.0
        for entry in history_entries:
            t_hist, dw = self._extract_history_entry(entry)

            t_up = t_hist + dendritic_delay
            minus_delta_t_up = self.t_last_spike_ms - t_up
            minus_t_down = t_up - t_spike

            PI = (
                     self.tau_L_trace * math.exp(minus_delta_t_up / tau_L)
                     - self.tau_s_trace * math.exp(minus_delta_t_up / tau_s)
                 ) * dw
            self.PI_integral += PI
            dPI_exp_integral += math.exp(minus_t_down / self.tau_Delta) * PI

        self.PI_exp_integral = (
            math.exp((self.t_last_spike_ms - t_spike) / self.tau_Delta) * self.PI_exp_integral
            + dPI_exp_integral
        )

        self.weight = self.PI_integral - self.PI_exp_integral
        self.weight = self.init_weight + (
            self.weight * 15.0 * C_m * tau_s * self.eta / (g_L * (tau_L - tau_s))
        )
        if self.weight > self.Wmax:
            self.weight = self.Wmax
        elif self.weight < self.Wmin:
            self.weight = self.Wmin

        event = {
            'weight': float(self.weight),
            'delay': float(dendritic_delay),
            'delay_steps': int(event_delay_steps),
            'receptor_type': self._to_int_scalar(receptor_type, name='receptor_type'),
            'multiplicity': self._validate_multiplicity(multiplicity),
            't_spike_ms': float(t_spike),
            'tau_s_ms': float(tau_s),
            'PI_integral': float(self.PI_integral),
            'PI_exp_integral': float(self.PI_exp_integral),
        }

        self.tau_L_trace = self.tau_L_trace * math.exp((self.t_last_spike_ms - t_spike) / tau_L) + 1.0
        self.tau_s_trace = self.tau_s_trace * math.exp((self.t_last_spike_ms - t_spike) / tau_s) + 1.0
        self.t_last_spike_ms = t_spike

        event['tau_L_trace_post'] = float(self.tau_L_trace)
        event['tau_s_trace_post'] = float(self.tau_s_trace)
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

        This method provides an alternative API name for spike event generation, matching
        common naming conventions in event-based simulators.

        Parameters
        ----------
        t_spike_ms : float or ArrayLike
            Presynaptic spike time in milliseconds.
        target : Any
            Postsynaptic neuron with Urbanczik archiving interface.
        receptor_type : int or ArrayLike, optional
            Receptor channel index. Default: ``0``
        multiplicity : float or ArrayLike, optional
            Spike event multiplicity. Default: ``1.0``
        delay : float or ArrayLike, optional
            Override dendritic delay (milliseconds). Default: ``None`` (use ``self.delay``)
        delay_steps : int or ArrayLike, optional
            Override event delivery delay (steps). Default: ``None`` (use ``self.delay_steps``)

        Returns
        -------
        dict[str, Any]
            Spike event dictionary identical to ``send()`` return value.

        See Also
        --------
        send : Primary spike processing method with full documentation.
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
        r"""Process sequence of presynaptic spikes and return event history.

        Convenience method for batch processing of spike trains. Each spike is processed
        sequentially using ``send()``, with synapse state (weight, traces, integrals)
        updating after each spike.

        Parameters
        ----------
        pre_spike_times_ms : array_like
            1D array or sequence of presynaptic spike times in milliseconds. Will be
            flattened if multidimensional. Order matters: earlier spikes affect later ones
            through trace dynamics.
        target : Any
            Postsynaptic neuron with Urbanczik archiving interface.
        receptor_type : int or ArrayLike, optional
            Receptor channel index applied to all spikes. Default: ``0``
        multiplicity : float or ArrayLike, optional
            Spike multiplicity applied to all spikes. Default: ``1.0``
        delay : float or ArrayLike, optional
            Override dendritic delay for all spikes (milliseconds). Default: ``None``
        delay_steps : int or ArrayLike, optional
            Override event delivery delay for all spikes (steps). Default: ``None``

        Returns
        -------
        list[dict[str, Any]]
            List of spike event dictionaries, one per input spike, in chronological order.
            Each dict has same structure as ``send()`` return value.

        Notes
        -----
        - **Stateful processing**: Synapse internal state (weight, traces) persists across
          spikes in the train. Final state reflects cumulative plasticity effects.
        - **Performance**: For large spike trains (>10000 spikes), consider batching or
          vectorization depending on target neuron implementation.
        - **Temporal ordering**: Input spikes should typically be sorted in ascending time
          order for biologically realistic plasticity dynamics.

        Examples
        --------
        Process spike train:

        .. code-block:: python

           >>> syn = bp.urbanczik_synapse(weight=1.0, eta=0.05)
           >>> spike_times = [10.0, 15.0, 20.0, 25.0]
           >>> events = syn.simulate_pre_spike_train(spike_times, target=mock_neuron)
           >>> weights = [e['weight'] for e in events]
           >>> print(f"Weight trajectory: {weights}")
           Weight trajectory: [1.002, 1.005, 1.008, 1.011]

        Extract trace evolution:

        .. code-block:: python

           >>> import numpy as np
           >>> spike_times = np.arange(0, 100, 5.0)
           >>> events = syn.simulate_pre_spike_train(spike_times, target=mock_neuron)
           >>> tau_L_traces = [e['tau_L_trace_post'] for e in events]
           >>> print(f"Final tau_L trace: {tau_L_traces[-1]:.3f}")
           Final tau_L trace: 2.456

        See Also
        --------
        send : Process single spike with full control over parameters.
        """
        dftype = brainstate.environ.dftype()
        times = np.asarray(u.math.asarray(pre_spike_times_ms), dtype=dftype).reshape(-1)
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
        r"""Validate that weight, Wmin, and Wmax share consistent signs.

        Raises
        ------
        ValueError
            If weight and Wmin have different signs, or if weight and Wmax have different signs.

        Notes
        -----
        Sign check logic and error messages exactly match NEST ``urbanczik_synapse::set_status()``.
        Uses NEST's specific sign comparison semantics via ``_sign_like_wmax()``.
        """
        # Keep sign checks/message text aligned with NEST urbanczik_synapse::set_status.
        if bool(np.signbit(self.weight)) != bool(np.signbit(self.Wmax)):
            raise ValueError('Weight and Wmin must have same sign.')

        if self._sign_like_wmax(self.weight) != self._sign_like_wmax(self.Wmax):
            raise ValueError('Weight and Wmax must have same sign.')

    @staticmethod
    def _sign_like_wmax(x: float) -> int:
        r"""NEST-compatible sign function for weight bound validation.

        Parameters
        ----------
        x : float
            Value to check sign of.

        Returns
        -------
        int
            dftype = brainstate.environ.dftype()
            ``1`` if x > 0, ``-1`` if x <= 0 (matching NEST's sign semantics for Wmax).

        Notes
        -----
        This function differs from standard sign() by treating zero as negative, consistent
        with NEST's weight bound validation logic.
        """
        return int((x > 0.0) - (x <= 0.0))

    @staticmethod
    def _get_urbanczik_history(target: Any, t1: float, t2: float, comp: int):
        r"""Retrieve dendritic prediction error history from target neuron.

        Parameters
        ----------
        target : Any
            Postsynaptic neuron object implementing ``get_urbanczik_history()`` method.
        t1 : float
            Start time in milliseconds (exclusive).
        t2 : float
            End time in milliseconds (inclusive).
        comp : int
            Compartment index (typically 1 for dendritic compartment).

        Returns
        -------
        list
            List of history entries in interval (t1, t2]. Each entry can be:
            - Object with ``t_``/``dw_`` attributes (NEST-style)
            - Object with ``t``/``dw`` attributes
            - Dict with ``'t'``/``'dw'`` keys
            - 2-tuple ``(time, delta_w)``
            Returns empty list if no entries or if target returns None.

        Raises
        ------
        AttributeError
            If target does not provide ``get_urbanczik_history()`` method.

        Notes
        -----
        First attempts to call with ``(t1, t2, comp)`` signature. If that raises TypeError
        (target doesn't accept comp argument), falls back to ``(t1, t2)`` signature.
        """
        fn = getattr(target, 'get_urbanczik_history', None)
        if fn is None or not callable(fn):
            raise AttributeError(
                'Target must provide get_urbanczik_history(t1, t2, comp) for urbanczik_synapse.'
            )

        try:
            history = fn(float(t1), float(t2), int(comp))
        except TypeError:
            history = fn(float(t1), float(t2))

        if history is None:
            return []
        return history

    @classmethod
    def _get_tau_L(cls, target: Any, comp: int) -> float:
        r"""Retrieve membrane time constant tau_L from target neuron.

        Parameters
        ----------
        target : Any
            Postsynaptic neuron object.
        comp : int
            Compartment index.

        Returns
        -------
        float
            Membrane time constant in milliseconds (tau_L = C_m / g_L).

        Raises
        ------
        AttributeError
            If target provides neither ``get_tau_L()`` nor both ``get_C_m()`` and ``get_g_L()``.

        Notes
        -----
        Tries the following in order:
        1. Direct ``get_tau_L(comp)`` or ``get_tau_l(comp)`` method
        2. If method doesn't accept comp argument, tries ``get_tau_L()`` without argument
        3. Falls back to computing tau_L = C_m / g_L from compartment values

        Handles both uppercase (``get_tau_L``, ``get_C_m``, ``get_g_L``) and lowercase
        (``get_tau_l``, ``get_c_m``, ``get_g_l``) method naming conventions.
        """
        fn = getattr(target, 'get_tau_L', None)
        if fn is None:
            fn = getattr(target, 'get_tau_l', None)
        if fn is not None and callable(fn):
            try:
                return float(fn(int(comp)))
            except TypeError:
                return float(fn())

        c_m = cls._get_compartment_value(target, ['get_C_m', 'get_c_m'], comp=comp, field='C_m')
        g_l = cls._get_compartment_value(target, ['get_g_L', 'get_g_l'], comp=comp, field='g_L')
        return float(c_m / g_l)

    @staticmethod
    def _get_compartment_value(target: Any, names: list[str], comp: int, field: str) -> float:
        r"""Retrieve compartment-specific parameter from target neuron.

        Parameters
        ----------
        target : Any
            Postsynaptic neuron object.
        names : list[str]
            List of method names to try in order (e.g., ``['get_C_m', 'get_c_m']``).
        comp : int
            Compartment index.
        field : str
            Field name for error message (e.g., ``'C_m'``, ``'g_L'``).

        Returns
        -------
        float
            Requested parameter value.

        Raises
        ------
        AttributeError
            If target does not provide any of the specified methods.

        Notes
        -----
        Tries each method name in order. For each callable method found, first attempts
        to call with compartment index ``comp``, then falls back to calling without arguments
        if TypeError is raised (for neurons without multi-compartment support).
        """
        for name in names:
            fn = getattr(target, name, None)
            if fn is None or not callable(fn):
                continue
            try:
                return float(fn(int(comp)))
            except TypeError:
                return float(fn())
        raise AttributeError(
            f'Target must provide {"/".join(names)}(comp) for urbanczik_synapse ({field}).'
        )

    @staticmethod
    def _extract_history_entry(entry: Any) -> tuple[float, float]:
        r"""Parse history entry into (time, delta_w) tuple.

        Parameters
        ----------
        entry : Any
            History entry in one of the supported formats:
            - Dict with keys ``'t_'``/``'dw_'`` or ``'t'``/``'dw'`` (or variants)
            - 2-tuple or list ``[time, delta_w]``
            - Object with attributes ``t_``/``dw_`` (NEST-style) or ``t``/``dw``

        Returns
        -------
        tuple[float, float]
            Extracted ``(time_ms, delta_w)`` pair.

        Raises
        ------
        ValueError
            If entry does not provide both time and delta_w values in any recognized format.

        Notes
        -----
        Supported key/attribute name variants for time: ``'t_'``, ``'t'``, ``'time_ms'``, ``'time'``
        Supported key/attribute name variants for delta_w: ``'dw_'``, ``'dw'``, ``'delta_w'``,
        ``'weight_change'``

        Prioritizes NEST-style naming (``t_``, ``dw_``) over simpler alternatives.
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
            raise ValueError('Each Urbanczik history entry must provide both time and dw values.')

        return float(t), float(dw)

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        r"""Convert value to finite float scalar with validation.

        Parameters
        ----------
        value : ArrayLike
            Input value (scalar, array, or saiunit Quantity).
        name : str
            Parameter name for error messages.

        Returns
        -------
        float
            Validated finite float scalar.

        Raises
        ------
        ValueError
            If value is not scalar, non-finite (NaN/infinity), or cannot be converted to float.

        Notes
        -----
        Handles saiunit Quantity objects by extracting mantissa. Flattens arrays to check
        for single-element constraint.
        """
        dftype = brainstate.environ.dftype()
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr[0])
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        return v

    @staticmethod
    def _to_int_scalar(value: ArrayLike, name: str) -> int:
        r"""Convert value to integer scalar with validation.

        Parameters
        ----------
        value : ArrayLike
            Input value (scalar, array, or saiunit Quantity).
        name : str
            Parameter name for error messages.

        Returns
        -------
        int
            Validated integer scalar.

        Raises
        ------
        ValueError
            If value is not scalar, non-finite, not sufficiently close to integer
            (tolerance 1e-12), or cannot be converted.

        Notes
        -----
        Converts to float first, validates finiteness, then rounds and checks integer constraint
        with 1e-12 absolute tolerance. Handles saiunit Quantity objects.
        """
        dftype = brainstate.environ.dftype()
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
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
        r"""Validate dendritic delay is positive.

        Parameters
        ----------
        value : ArrayLike
            Delay value to validate.

        Returns
        -------
        float
            Validated delay in milliseconds.

        Raises
        ------
        ValueError
            If delay is not positive (must be > 0).
        """
        d = cls._to_float_scalar(value, name='delay')
        if d <= 0.0:
            raise ValueError('delay must be > 0.')
        return d

    @classmethod
    def _validate_delay_steps(cls, value: ArrayLike) -> int:
        r"""Validate event delivery delay steps.

        Parameters
        ----------
        value : ArrayLike
            Delay steps value to validate.

        Returns
        -------
        int
            Validated delay steps (must be >= 1).

        Raises
        ------
        ValueError
            If delay_steps is less than 1.
        """
        d = cls._to_int_scalar(value, name='delay_steps')
        if d < 1:
            raise ValueError('delay_steps must be >= 1.')
        return d

    @classmethod
    def _validate_multiplicity(cls, value: ArrayLike) -> float:
        r"""Validate spike event multiplicity is non-negative.

        Parameters
        ----------
        value : ArrayLike
            Multiplicity value to validate.

        Returns
        -------
        float
            Validated multiplicity (must be >= 0).

        Raises
        ------
        ValueError
            If multiplicity is negative.

        Notes
        -----
        Multiplicity is validated but not used in plasticity computation (NEST compatibility).
        """
        m = cls._to_float_scalar(value, name='multiplicity')
        if m < 0.0:
            raise ValueError('multiplicity must be >= 0.')
        return m
