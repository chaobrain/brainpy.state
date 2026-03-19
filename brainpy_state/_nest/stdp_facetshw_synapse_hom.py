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
from collections.abc import Mapping

import brainstate
import saiunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike

from .static_synapse import _UNSET, static_synapse

__all__ = [
    'stdp_facetshw_synapse_hom',
]

_STDP_EPS = 1.0e-6
_LUT_ENTRY_MIN = 0
_LUT_ENTRY_MAX = 15
_DEFAULT_LUT_0 = (2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15)
_DEFAULT_LUT_1 = (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13)
_DEFAULT_LUT_2 = tuple(range(16))
_DEFAULT_CONFIG_0 = (0, 0, 1, 0)
_DEFAULT_CONFIG_1 = (0, 1, 0, 0)
_DEFAULT_RESET_PATTERN = (1, 1, 1, 1, 1, 1)


class stdp_facetshw_synapse_hom(static_synapse):
    r"""NEST-compatible ``stdp_facetshw_synapse_hom`` connection model.

    Implements hardware-constrained spike-timing dependent plasticity (STDP) designed for the
    BrainScaleS / FACETS neuromorphic hardware platform. This model features 4-bit discrete weight
    representation, periodic controller-driven weight updates via look-up tables, reduced symmetric
    nearest-neighbor spike pairing, and configurable capacitor reset patterns. All plasticity
    parameters are model-level (homogeneous across all synapses).

    **1. Mathematical Formulation**

    The model maintains two exponentially decaying accumulator traces that encode pre-post spike
    timing relationships:

    **Causal accumulator** (potentiation):

    .. math::

        a_{\text{causal}}(t) = \sum_{t_{\text{post}} < t_{\text{pre,last}}}
            \exp\left(\frac{t_{\text{post}} - t_{\text{pre,last}}}{\tau_+}\right)

    Updated when a postsynaptic spike occurs before the last presynaptic spike (only the *first*
    post-spike in each interval contributes).

    **Acausal accumulator** (depression):

    .. math::

        a_{\text{acausal}}(t) = \sum_{t_{\text{post}} > t_{\text{pre}}}
            \exp\left(\frac{t_{\text{post}} - t_{\text{pre}}}{\tau_-}\right)

    Updated when a postsynaptic spike occurs after the current presynaptic spike (only the *last*
    post-spike in each interval contributes).

    **2. Hardware-Constrained Weight Update**

    Weight updates are *not* instantaneous but occur at periodic readout cycles mimicking hardware
    controller limitations:

    a. **Continuous to discrete conversion**:

       .. math::

           w_{\text{discrete}} = \text{round}\left(\frac{w_{\text{continuous}}}{w_{\text{per\_entry}}}\right)

       where :math:`w_{\text{per\_entry}} = W_{\max} / 15` (4-bit representation: 0–15).

    b. **Comparator evaluation**: Two evaluation functions produce binary decisions:

       .. math::

           E_k(a_c, a_a) = \left(\frac{a_{\text{tl}} + c_{k,2} a_c + c_{k,1} a_a}{1 + c_{k,2} + c_{k,1}} >
                                   \frac{a_{\text{th}} + c_{k,0} a_c + c_{k,3} a_a}{1 + c_{k,0} + c_{k,3}}\right)

       where :math:`k \in \{0, 1\}`, :math:`c_{k,\cdot}` are configuration bits, :math:`a_c =
       a_{\text{causal}}`, :math:`a_a = a_{\text{acausal}}`, :math:`a_{\text{th}} =
       a_{\text{thresh\_th}}`, and :math:`a_{\text{tl}} = a_{\text{thresh\_tl}}`.

    c. **LUT selection and weight update**:

       - If :math:`(E_0, E_1) = (1, 0)` -- apply ``lookuptable_0[w_discrete]`` to get :math:`w_{\text{discrete}}'`
       - If :math:`(E_0, E_1) = (0, 1)` -- apply ``lookuptable_1[w_discrete]`` to get :math:`w_{\text{discrete}}'`
       - If :math:`(E_0, E_1) = (1, 1)` -- apply ``lookuptable_2[w_discrete]`` to get :math:`w_{\text{discrete}}'`
       - If :math:`(E_0, E_1) = (0, 0)` -- no weight change

    d. **Capacitor reset**: After LUT application, accumulators may be reset to zero based on
       6-bit ``reset_pattern`` (pairs of causal/acausal reset bits for each LUT).

    e. **Discrete to continuous conversion**:

       .. math::

           w_{\text{continuous}}' = w_{\text{discrete}}' \times w_{\text{per\_entry}}

    **3. Readout Cycle Scheduling**

    Weight updates occur only when :math:`t_{\text{spike}} > t_{\text{next\_readout}}`, where
    :math:`t_{\text{next\_readout}}` advances in fixed intervals:

    .. math::

        T_{\text{cycle}} = \left\lceil\frac{N_{\text{synapses}}}{N_{\text{synapses/driver}}}\right\rceil
                           \times T_{\text{driver\_readout}}

    Each synapse is assigned a unique ``synapse_id`` upon first activation, determining its initial
    readout time offset within the cycle.

    **4. Event Ordering and Timing**

    For a presynaptic spike at time :math:`t_{\text{pre}}` with dendritic delay :math:`d`:

    1. **Initialize controller state** (on first spike):
       - Assign ``synapse_id`` from global counter ``no_synapses``
       - Increment ``no_synapses``, recalculate cycle duration
       - Set initial ``next_readout_time`` based on synapse_id

    2. **Check readout window**: if :math:`t_{\text{pre}} > t_{\text{next\_readout}}`:
       - Convert :math:`w` to 4-bit discrete representation
       - Evaluate comparator functions :math:`E_0` and :math:`E_1`
       - Apply selected LUT and reset pattern
       - Advance :math:`t_{\text{next\_readout}}` until :math:`t_{\text{pre}} \leq t_{\text{next\_readout}}`
       - Convert updated discrete weight back to continuous value

    3. **Spike pairing**: query postsynaptic history in :math:`(t_{\text{last}} - d, t_{\text{pre}} - d]`.
       If history non-empty, update :math:`a_{\text{causal}}` using *first* post-spike timestamp
       and update :math:`a_{\text{acausal}}` using *last* post-spike timestamp.

    4. **Event transmission**: schedule spike event with current weight at :math:`t_{\text{pre}} + d`

    5. **Update state**: set :math:`t_{\text{lastspike}} = t_{\text{pre}}`

    **5. Computational Constraints**

    - Uses on-grid spike timestamps (ignores sub-step offsets)
    - Reduced spike pairing (first/last only) minimizes computational cost
    - Discrete 4-bit weight representation matches hardware constraints
    - Periodic updates simulate asynchronous hardware readout cycles

    Parameters
    ----------
    weight : float or array-like, default: 1.0
        Initial continuous synaptic weight (dimensionless). Converted to 4-bit discrete
        representation during readout cycles.
    delay : Quantity[time], default: 1.0 * u.ms
        Synaptic transmission delay. Affects timing of post-synaptic event delivery and the
        temporal window for spike pairing (dendritic delay :math:`d` in equations above).
    receptor_type : int, default: 0
        Target receptor port identifier on the postsynaptic neuron. Used for routing events to
        specific synaptic channels (e.g., AMPA vs GABA receptors).
    tau_plus : Quantity[time], default: 20.0 * u.ms
        Time constant :math:`\tau_+` for the causal (potentiation) accumulator. Must be positive.
        Controls decay rate of timing information for pre-before-post pairings.
    tau_minus_stdp : Quantity[time], default: 20.0 * u.ms
        Time constant :math:`\tau_-` for the acausal (depression) accumulator. Must be positive.
        Controls decay rate of timing information for post-before-pre pairings.
    Wmax : float, default: 100.0
        Maximum biological weight. Defines the upper bound of the continuous weight range and is
        used to compute ``weight_per_lut_entry`` if not explicitly provided. Typically set to
        match the biological range of synaptic efficacy.
    weight_per_lut_entry : float, optional
        Conversion factor between LUT index (0–15) and continuous weight. If not provided,
        automatically computed as :math:`W_{\max} / 15`. This quantization step determines the
        granularity of weight representation.
    no_synapses : int, default: 0
        Global synapse counter used by the simulated hardware controller. Automatically incremented
        when new synapses are initialized. Affects readout cycle scheduling.
    synapses_per_driver : int, default: 50
        Number of synapses updated per driver readout cycle. Must be positive. Models hardware
        bandwidth constraints: larger values reduce cycle duration but may be hardware-infeasible.
    driver_readout_time : float, default: 15.0
        Processing time (ms) required for one driver row readout. Must be positive. Combined with
        ``synapses_per_driver``, determines the total readout cycle duration.
    readout_cycle_duration : float, optional
        Total duration (ms) of one complete readout cycle. If not provided, automatically computed
        as :math:`\lceil N_{\text{synapses}} / N_{\text{synapses/driver}} \rceil \times
        T_{\text{driver\_readout}}`. Explicitly setting this overrides the automatic calculation.
    lookuptable_0 : array-like of 16 ints, default: (2,3,4,4,5,6,7,8,9,10,11,12,13,14,14,15)
        Look-up table applied when comparator evaluation yields :math:`(E_0, E_1) = (1, 0)`.
        Each entry must be an integer in [0, 15]. Index by current discrete weight, returns new
        discrete weight. Default implements potentiation (weight increase).
    lookuptable_1 : array-like of 16 ints, default: (0,0,1,2,3,4,5,6,7,8,9,10,10,11,12,13)
        Look-up table applied when comparator evaluation yields :math:`(E_0, E_1) = (0, 1)`.
        Default implements depression (weight decrease).
    lookuptable_2 : array-like of 16 ints, default: (0,1,2,...,15)
        Look-up table applied when comparator evaluation yields :math:`(E_0, E_1) = (1, 1)`.
        Default is identity (no change). Can be configured for combined potentiation/depression.
    configbit_0 : array-like of 4 ints, default: (0,0,1,0)
        Configuration bits :math:`[c_{0,0}, c_{0,1}, c_{0,2}, c_{0,3}]` for comparator function
        :math:`E_0`. Determines which accumulator values influence the evaluation threshold.
    configbit_1 : array-like of 4 ints, default: (0,1,0,0)
        Configuration bits :math:`[c_{1,0}, c_{1,1}, c_{1,2}, c_{1,3}]` for comparator function
        :math:`E_1`. Default configuration creates asymmetry between :math:`E_0` and :math:`E_1`.
    reset_pattern : array-like of 6 ints, default: (1,1,1,1,1,1)
        Six reset bits controlling accumulator resets after LUT application:
        ``[causal_reset_0, acausal_reset_0, causal_reset_1, acausal_reset_1, causal_reset_2,
        acausal_reset_2]``. Bit value 1 means reset to zero, 0 means preserve accumulator value.
    a_causal : float, default: 0.0
        Initial value of the causal (potentiation) accumulator. Typically starts at zero and
        accumulates over simulation time based on spike timing.
    a_acausal : float, default: 0.0
        Initial value of the acausal (depression) accumulator.
    a_thresh_th : float, default: 21.835
        Upper comparator threshold :math:`a_{\text{th}}` used in evaluation functions. Controls
        sensitivity to accumulator values. Default matches NEST hardware parameters.
    a_thresh_tl : float, default: 21.835
        Lower comparator threshold :math:`a_{\text{tl}}` used in evaluation functions.
    init_flag : bool, default: False
        Internal flag tracking whether the synapse has been initialized (assigned a synapse_id).
        Set to True automatically upon first presynaptic spike.
    synapse_id : int, default: 0
        Unique identifier assigned to this synapse by the controller. Determines initial readout
        time offset. Automatically assigned on first spike if init_flag is False.
    next_readout_time : float, default: 0.0
        Timestamp (ms) of the next scheduled readout cycle for this synapse. Advances in steps of
        ``readout_cycle_duration`` after each readout window.
    post : Dynamics, optional
        Default postsynaptic target object. Can be overridden per-call in ``send()`` and
        ``update()`` methods.
    name : str, optional
        Unique identifier for this synapse instance. Used for debugging and logging.

    Parameter Mapping
    -----------------

    Correspondence between NEST C++ implementation and this Python implementation:

    +----------------------------+-------------------------------+------------------------------------+
    | NEST Parameter             | brainpy.state Parameter       | Notes                              |
    +============================+===============================+====================================+
    | ``weight``                 | ``weight``                    | Continuous synaptic weight         |
    +----------------------------+-------------------------------+------------------------------------+
    | ``delay``                  | ``delay``                     | Transmission delay (ms)            |
    +----------------------------+-------------------------------+------------------------------------+
    | ``tau_plus``               | ``tau_plus``                  | Causal time constant (ms)          |
    +----------------------------+-------------------------------+------------------------------------+
    | ``tau_minus_stdp``         | ``tau_minus_stdp``            | Acausal time constant (ms)         |
    +----------------------------+-------------------------------+------------------------------------+
    | ``Wmax``                   | ``Wmax``                      | Maximum weight                     |
    +----------------------------+-------------------------------+------------------------------------+
    | ``no_synapses``            | ``no_synapses``               | Global synapse counter             |
    +----------------------------+-------------------------------+------------------------------------+
    | ``synapses_per_driver``    | ``synapses_per_driver``       | Synapses per readout cycle         |
    +----------------------------+-------------------------------+------------------------------------+
    | ``driver_readout_time``    | ``driver_readout_time``       | Driver processing time (ms)        |
    +----------------------------+-------------------------------+------------------------------------+
    | ``readout_cycle_duration`` | ``readout_cycle_duration``    | Full cycle duration (ms)           |
    +----------------------------+-------------------------------+------------------------------------+
    | ``lookuptable_0``          | ``lookuptable_0``             | LUT for (1,0) evaluation           |
    +----------------------------+-------------------------------+------------------------------------+
    | ``lookuptable_1``          | ``lookuptable_1``             | LUT for (0,1) evaluation           |
    +----------------------------+-------------------------------+------------------------------------+
    | ``lookuptable_2``          | ``lookuptable_2``             | LUT for (1,1) evaluation           |
    +----------------------------+-------------------------------+------------------------------------+
    | ``configbit_0``            | ``configbit_0``               | 4-bit config for :math:`E_0`       |
    +----------------------------+-------------------------------+------------------------------------+
    | ``configbit_1``            | ``configbit_1``               | 4-bit config for :math:`E_1`       |
    +----------------------------+-------------------------------+------------------------------------+
    | ``reset_pattern``          | ``reset_pattern``             | 6-bit accumulator reset pattern    |
    +----------------------------+-------------------------------+------------------------------------+
    | ``a_causal``               | ``a_causal``                  | Causal accumulator state           |
    +----------------------------+-------------------------------+------------------------------------+
    | ``a_acausal``              | ``a_acausal``                 | Acausal accumulator state          |
    +----------------------------+-------------------------------+------------------------------------+
    | ``a_thresh_th``            | ``a_thresh_th``               | Upper comparator threshold         |
    +----------------------------+-------------------------------+------------------------------------+
    | ``a_thresh_tl``            | ``a_thresh_tl``               | Lower comparator threshold         |
    +----------------------------+-------------------------------+------------------------------------+

    Attributes
    ----------
    discrete_weight : int
        Current 4-bit discrete weight representation (0–15). Updated during readout cycles by
        applying the selected look-up table.
    t_lastspike : float
        Timestamp (ms) of the most recent presynaptic spike. Used to determine the temporal window
        for querying postsynaptic spike history.

    Raises
    ------
    ValueError
        - If ``tau_plus`` or ``tau_minus_stdp`` is non-positive
        - If ``synapses_per_driver`` is non-positive
        - If ``driver_readout_time`` is non-positive
        - If look-up table entries are outside [0, 15] range
        - If look-up tables have mismatched sizes (must all be length 16)
        - If ``configbit_0`` or ``configbit_1`` do not have exactly 4 entries
        - If ``reset_pattern`` does not have exactly 6 entries
        - If ``readout_cycle_duration`` is zero or negative during active readout scheduling
        - If common properties (e.g., ``tau_plus``, ``Wmax``, LUTs) are included in per-synapse
          connection specifications

    Notes
    -----
    **Design Rationale**:

    This model replicates the behavior of NEST's ``stdp_facetshw_synapse_hom`` implementation,
    which was designed to simulate the BrainScaleS neuromorphic hardware platform. Key constraints:

    - **4-bit weight quantization**: Hardware synapses use 4-bit weight storage (16 discrete
      levels), requiring explicit continuous↔discrete conversion.
    - **Periodic controller updates**: Hardware cannot update all synapses simultaneously;
      instead, a controller sequentially reads/writes synapse rows at fixed intervals.
    - **Reduced spike pairing**: Full all-to-all spike pairing is computationally expensive;
      using only first/last post-spikes per interval provides sufficient plasticity information.
    - **Homogeneous parameters**: All plasticity parameters (time constants, LUTs, thresholds)
      are model-level, not per-synapse, matching hardware constraints.

    **Differences from Standard STDP**:

    - Standard STDP uses immediate weight updates on every spike pair. This model defers updates
      to periodic readout cycles.
    - Standard STDP uses continuous weight values. This model quantizes weights to 4-bit discrete
      values during updates.
    - Standard STDP uses all-to-all spike pairing. This model uses only first/last post-spikes.

    **Common-Property Restrictions**:

    Unlike NEST's per-synapse models (e.g., ``stdp_synapse``), this homogeneous variant does not
    allow time constants, LUTs, or configuration bits to be set per-connection. Attempting to pass
    these in ``syn_spec`` dictionaries will raise ``ValueError``. To customize these parameters:

    - Set them at model creation: ``stdp_facetshw_synapse_hom(tau_plus=15*u.ms, ...)``
    - Or use ``set()`` method on the model instance: ``model.set(tau_plus=15*u.ms)``

    **Simulation Performance**:

    This model is more computationally efficient than full all-to-all STDP due to:

    1. Reduced spike pairing (O(1) instead of O(n_post_spikes) per pre-spike)
    2. Infrequent weight updates (only at readout cycles, not every spike)
    3. Simplified LUT-based weight changes (no exponential calculations)

    However, tracking postsynaptic spike history requires memory proportional to post-spike count.

    Examples
    --------
    **1. Basic Usage with Default Hardware Parameters**

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import saiunit as u
        >>>
        >>> # Create synapse with default FACETS hardware parameters
        >>> syn = bp.stdp_facetshw_synapse_hom(
        ...     weight=1.0,
        ...     delay=1.0 * u.ms,
        ...     tau_plus=20.0 * u.ms,
        ...     tau_minus_stdp=20.0 * u.ms,
        ...     Wmax=100.0,
        ... )
        >>>
        >>> # Initialize state
        >>> syn.init_state()
        >>>
        >>> # Simulate presynaptic spike at t=10 ms
        >>> # (post-spike must be recorded separately)
        >>> syn.record_post_spike(1.0, t_spike_ms=8.0)  # post-before-pre
        >>> syn.send(1.0)  # process pre-spike
        >>>
        >>> # Check updated state
        >>> params = syn.get()
        >>> print(f"Weight: {params['weight']:.3f}")
        >>> print(f"Causal accumulator: {params['a_causal']:.3f}")

    **2. Custom Look-Up Tables for Asymmetric Learning**

    .. code-block:: python

        >>> # Create LUTs with strong potentiation, weak depression
        >>> lut_pot = [3, 5, 7, 9, 11, 12, 13, 14, 14, 15, 15, 15, 15, 15, 15, 15]  # strong LTP
        >>> lut_dep = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # weak LTD
        >>>
        >>> syn = bp.stdp_facetshw_synapse_hom(
        ...     weight=50.0,
        ...     Wmax=100.0,
        ...     lookuptable_0=lut_pot,
        ...     lookuptable_1=lut_dep,
        ...     configbit_0=(0, 0, 1, 0),  # causal-dominant for potentiation
        ...     configbit_1=(0, 1, 0, 0),  # acausal-dominant for depression
        ... )

    **3. Controller Timing Configuration**

    .. code-block:: python

        >>> # Simulate hardware with 100 synapses, 25 per driver, 10ms readout time
        >>> syn = bp.stdp_facetshw_synapse_hom(
        ...     weight=10.0,
        ...     no_synapses=100,
        ...     synapses_per_driver=25,
        ...     driver_readout_time=10.0,
        ... )
        >>>
        >>> # Readout cycle duration computed automatically:
        >>> # ceil(100 / 25) * 10ms = 4 * 10ms = 40ms
        >>> print(f"Cycle duration: {syn.readout_cycle_duration} ms")

    **4. Selective Accumulator Resets**

    .. code-block:: python

        >>> # Only reset causal accumulator after potentiation (LUT 0)
        >>> # Preserve both accumulators for depression (LUT 1)
        >>> syn = bp.stdp_facetshw_synapse_hom(
        ...     weight=50.0,
        ...     reset_pattern=(1, 0, 0, 0, 1, 1),  # reset pattern for [LUT0_causal, LUT0_acausal,
        ...                                         #                      LUT1_causal, LUT1_acausal,
        ...                                         #                      LUT2_causal, LUT2_acausal]
        ... )

    See Also
    --------
    stdp_synapse : Standard all-to-all pair-based STDP without hardware constraints
    stdp_triplet_synapse : Triplet STDP rule with additional pre-post-pre and post-pre-post terms
    static_synapse : Base class for static (non-plastic) synaptic connections

    References
    ----------
    .. [1] NEST source code: ``models/stdp_facetshw_synapse_hom.h``,
           ``models/stdp_facetshw_synapse_hom_impl.h``, and
           ``models/stdp_facetshw_synapse_hom.cpp``
           https://github.com/nest/nest-simulator
    .. [2] Morrison A, Diesmann M, Gerstner W (2008). Phenomenological models of synaptic
           plasticity based on spike timing. Biological Cybernetics, 98(6):459-478.
           https://doi.org/10.1007/s00422-008-0233-1
    .. [3] Pfeil T, Grübl A, Jeltsch S, et al. (2012). Is a 4-bit synaptic weight resolution
           enough? Constraints on enabling spike-timing dependent plasticity in neuromorphic
           hardware. Frontiers in Neuroscience, 6:90.
           https://doi.org/10.3389/fnins.2012.00090
    .. [4] Schemmel J, Brüderle D, Grübl A, Hock M, Meier K, Millner S (2010). A wafer-scale
           neuromorphic hardware system for large-scale neural modeling. Proceedings of 2010 IEEE
           International Symposium on Circuits and Systems, 1947-1950.
           https://doi.org/10.1109/ISCAS.2010.5536970
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus_stdp: ArrayLike = 20.0 * u.ms,
        Wmax: ArrayLike = 100.0,
        weight_per_lut_entry: ArrayLike | object = _UNSET,
        no_synapses: ArrayLike = 0,
        synapses_per_driver: ArrayLike = 50,
        driver_readout_time: ArrayLike = 15.0,
        readout_cycle_duration: ArrayLike | object = _UNSET,
        lookuptable_0: ArrayLike = _DEFAULT_LUT_0,
        lookuptable_1: ArrayLike = _DEFAULT_LUT_1,
        lookuptable_2: ArrayLike = _DEFAULT_LUT_2,
        configbit_0: ArrayLike = _DEFAULT_CONFIG_0,
        configbit_1: ArrayLike = _DEFAULT_CONFIG_1,
        reset_pattern: ArrayLike = _DEFAULT_RESET_PATTERN,
        a_causal: ArrayLike = 0.0,
        a_acausal: ArrayLike = 0.0,
        a_thresh_th: ArrayLike = 21.835,
        a_thresh_tl: ArrayLike = 21.835,
        init_flag: ArrayLike = False,
        synapse_id: ArrayLike = 0,
        next_readout_time: ArrayLike = 0.0,
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
        self.tau_minus_stdp = self._to_scalar_time_ms(tau_minus_stdp, name='tau_minus_stdp')
        self.Wmax = self._to_scalar_float(Wmax, name='Wmax')

        self._validate_positive(self.tau_plus, name='tau_plus')
        self._validate_positive(self.tau_minus_stdp, name='tau_minus_stdp')

        self.no_synapses = self._to_int_scalar(no_synapses, name='no_synapses')
        self.synapses_per_driver = self._to_int_scalar(synapses_per_driver, name='synapses_per_driver')
        self.driver_readout_time = self._to_scalar_float(driver_readout_time, name='driver_readout_time')

        self._validate_synapses_per_driver(self.synapses_per_driver)
        self._validate_positive(self.driver_readout_time, name='driver_readout_time')

        self.lookuptable_0 = self._to_int_vector(
            lookuptable_0,
            name='lookuptable_0',
            exact_size=16,
            min_value=_LUT_ENTRY_MIN,
            max_value=_LUT_ENTRY_MAX,
        )
        self.lookuptable_1 = self._to_int_vector(
            lookuptable_1,
            name='lookuptable_1',
            exact_size=16,
            min_value=_LUT_ENTRY_MIN,
            max_value=_LUT_ENTRY_MAX,
        )
        self.lookuptable_2 = self._to_int_vector(
            lookuptable_2,
            name='lookuptable_2',
            exact_size=16,
            min_value=_LUT_ENTRY_MIN,
            max_value=_LUT_ENTRY_MAX,
        )

        self._validate_lut_size_match(self.lookuptable_0, self.lookuptable_1)
        self._validate_lut_size_match(self.lookuptable_0, self.lookuptable_2)

        self.configbit_0 = self._to_int_vector(configbit_0, name='configbit_0', exact_size=4)
        self.configbit_1 = self._to_int_vector(configbit_1, name='configbit_1', exact_size=4)
        self.reset_pattern = self._to_int_vector(reset_pattern, name='reset_pattern', exact_size=6)

        if weight_per_lut_entry is _UNSET:
            self.weight_per_lut_entry = float(self.Wmax / (len(self.lookuptable_0) - 1))
        else:
            self.weight_per_lut_entry = self._to_scalar_float(weight_per_lut_entry, name='weight_per_lut_entry')

        if readout_cycle_duration is _UNSET:
            self.readout_cycle_duration = 0.0
            self._calc_readout_cycle_duration()
        else:
            self.readout_cycle_duration = self._to_scalar_float(readout_cycle_duration, name='readout_cycle_duration')

        self.a_causal = self._to_scalar_float(a_causal, name='a_causal')
        self.a_acausal = self._to_scalar_float(a_acausal, name='a_acausal')
        self.a_thresh_th = self._to_scalar_float(a_thresh_th, name='a_thresh_th')
        self.a_thresh_tl = self._to_scalar_float(a_thresh_tl, name='a_thresh_tl')
        self.init_flag = self._to_bool_scalar(init_flag, name='init_flag')
        self.synapse_id = self._to_int_scalar(synapse_id, name='synapse_id')
        self.next_readout_time = self._to_scalar_float(next_readout_time, name='next_readout_time')

        self.discrete_weight = 0
        self.t_lastspike = 0.0
        self._post_hist_t: list[float] = []

        self._a_causal0 = float(self.a_causal)
        self._a_acausal0 = float(self.a_acausal)
        self._a_thresh_th0 = float(self.a_thresh_th)
        self._a_thresh_tl0 = float(self.a_thresh_tl)
        self._init_flag0 = bool(self.init_flag)
        self._synapse_id0 = int(self.synapse_id)
        self._next_readout_time0 = float(self.next_readout_time)
        self._no_synapses0 = int(self.no_synapses)
        self._readout_cycle_duration0 = float(self.readout_cycle_duration)

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        dftype = brainstate.environ.dftype()
        if isinstance(value, u.Quantity):
            unit = u.get_unit(value)
            arr = np.asarray(value.to_decimal(unit), dtype=dftype)
        else:
            arr = np.asarray(u.math.asarray(value, dtype=dftype), dtype=dftype)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        v = float(arr.reshape(()))
        if not np.isfinite(v):
            raise ValueError(f'{name} must be finite.')
        return v

    @classmethod
    def _to_int_scalar(cls, value: ArrayLike, *, name: str) -> int:
        v = cls._to_scalar_float(value, name=name)
        rounded = int(round(v))
        if not math.isclose(v, float(rounded), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'{name} must be an integer.')
        return rounded

    @classmethod
    def _to_bool_scalar(cls, value: ArrayLike, *, name: str) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        ivalue = cls._to_int_scalar(value, name=name)
        if ivalue not in (0, 1):
            raise ValueError(f'{name} must be boolean-like (0/1/False/True).')
        return bool(ivalue)

    @classmethod
    def _to_non_negative_int_count(cls, value: ArrayLike, *, name: str) -> int:
        count = cls._to_int_scalar(value, name=name)
        if count < 0:
            raise ValueError(f'{name} must be non-negative.')
        return count

    @classmethod
    def _to_int_vector(
        cls,
        value: ArrayLike,
        *,
        name: str,
        exact_size: int | None = None,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> list[int]:
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        dftype = brainstate.environ.dftype()
        flat = np.asarray(arr.reshape(-1), dtype=dftype)
        if exact_size is not None and flat.size != exact_size:
            raise ValueError(f'{name} must contain exactly {exact_size} entries.')
        values: list[int] = []
        for raw in flat:
            if not np.isfinite(raw):
                raise ValueError(f'{name} entries must be finite.')
            i = int(round(float(raw)))
            if not math.isclose(float(raw), float(i), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f'{name} entries must be integers.')
            if min_value is not None and i < min_value:
                raise ValueError(f'{name} entries must be in [{min_value},{max_value}].')
            if max_value is not None and i > max_value:
                raise ValueError(f'{name} entries must be in [{min_value},{max_value}].')
            values.append(i)
        return values

    @staticmethod
    def _validate_positive(value: float, *, name: str):
        if value <= 0.0:
            raise ValueError(f'{name} must be > 0.')

    @staticmethod
    def _validate_synapses_per_driver(value: int):
        if value <= 0:
            raise ValueError('synapses_per_driver must be > 0.')

    @staticmethod
    def _validate_lut_size_match(left: list[int], right: list[int]):
        if len(left) != len(right):
            raise ValueError('Look-up table has not 2^4 entries.')

    @staticmethod
    def _round_half_away_from_zero(x: float) -> int:
        if x >= 0.0:
            return int(math.floor(x + 0.5))
        return int(math.ceil(x - 0.5))

    def _calc_readout_cycle_duration(self):
        self.readout_cycle_duration = (
            int((self.no_synapses - 1.0) / self.synapses_per_driver + 1.0) * self.driver_readout_time
        )

    @staticmethod
    def _eval_function(
        a_causal: float,
        a_acausal: float,
        a_thresh_th: float,
        a_thresh_tl: float,
        configbit: list[int],
    ) -> bool:
        return (
            (a_thresh_tl + configbit[2] * a_causal + configbit[1] * a_acausal)
            / (1 + configbit[2] + configbit[1])
            > (a_thresh_th + configbit[0] * a_causal + configbit[3] * a_acausal)
            / (1 + configbit[0] + configbit[3])
        )

    @classmethod
    def _weight_to_entry(cls, weight: float, weight_per_lut_entry: float) -> int:
        return cls._round_half_away_from_zero(weight / weight_per_lut_entry)

    @staticmethod
    def _entry_to_weight(discrete_weight: int, weight_per_lut_entry: float) -> float:
        return float(discrete_weight * weight_per_lut_entry)

    @staticmethod
    def _lookup(discrete_weight: int, table: list[int]) -> int:
        if discrete_weight < 0 or discrete_weight >= len(table):
            raise ValueError(
                f'Discrete weight index {discrete_weight} is out of LUT bounds [0, {len(table) - 1}].'
            )
        return int(table[discrete_weight])

    def _record_post_spike_at(self, t_spike_ms: float):
        self._post_hist_t.append(float(t_spike_ms))

    def record_post_spike(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        t_spike_ms: ArrayLike | None = None,
    ) -> int:
        r"""Record postsynaptic spikes into internal history buffer.

        Registers postsynaptic spike events for subsequent spike-timing calculations during
        presynaptic spike processing. Recorded spikes are stored in an internal buffer and queried
        when ``send()`` is called to compute accumulator updates based on pre-post spike timing.

        Parameters
        ----------
        multiplicity : float or array-like, default: 1.0
            Number of postsynaptic spikes to record at the given timestamp. Must be non-negative.
            For ``multiplicity=0``, no spikes are recorded (no-op). For ``multiplicity>1``,
            multiple identical spike timestamps are added to the history buffer.
        t_spike_ms : float or array-like, optional
            Explicit spike timestamp in milliseconds. If not provided, defaults to current
            simulation time plus delay: ``current_time_ms + delay``. This allows recording spikes
            at arbitrary past or future times for offline analysis or replay scenarios.

        Returns
        -------
        count : int
            Number of spike events actually recorded (equals ``multiplicity`` if non-negative,
            otherwise 0).

        Raises
        ------
        ValueError
            If ``multiplicity`` is negative.

        Notes
        -----
        - The history buffer is unbounded and grows with every recorded spike. For long simulations
          with high-frequency postsynaptic activity, consider periodically clearing old spikes that
          fall outside the relevant temporal window (not implemented in this version).
        - Spike timestamps are stored as floating-point values with millisecond precision. Spikes
          are matched to presynaptic events using a small tolerance (``_STDP_EPS = 1e-6 ms``) to
          account for floating-point rounding errors.
        - This method is typically called automatically by the postsynaptic neuron's spike
          generation mechanism, but can also be invoked manually for testing or replay purposes.

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bp
            >>> import saiunit as u
            >>>
            >>> syn = bp.stdp_facetshw_synapse_hom(weight=10.0, delay=1.0 * u.ms)
            >>> syn.init_state()
            >>>
            >>> # Record single post-spike at t=5.0 ms
            >>> count = syn.record_post_spike(1.0, t_spike_ms=5.0)
            >>> print(f"Recorded {count} spike(s)")
            Recorded 1 spike(s)
            >>>
            >>> # Record burst of 3 post-spikes at t=10.0 ms
            >>> count = syn.record_post_spike(3.0, t_spike_ms=10.0)
            >>> print(f"Recorded {count} spike(s)")
            Recorded 3 spike(s)
            >>>
            >>> # No-op: record zero spikes
            >>> count = syn.record_post_spike(0.0, t_spike_ms=15.0)
            >>> print(f"Recorded {count} spike(s)")
            Recorded 0 spike(s)
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
                selected.append(float(t_post))
        return selected

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        super().init_state()
        self.a_causal = float(self._a_causal0)
        self.a_acausal = float(self._a_acausal0)
        self.a_thresh_th = float(self._a_thresh_th0)
        self.a_thresh_tl = float(self._a_thresh_tl0)
        self.init_flag = bool(self._init_flag0)
        self.synapse_id = int(self._synapse_id0)
        self.next_readout_time = float(self._next_readout_time0)
        self.no_synapses = int(self._no_synapses0)
        self.readout_cycle_duration = float(self._readout_cycle_duration0)
        self.discrete_weight = 0
        self.t_lastspike = 0.0
        self._post_hist_t = []

    def get(self) -> dict:
        r"""Return current public parameters and mutable state.

        Retrieves all user-accessible parameters and state variables in NEST-compatible format.
        Includes inherited static synapse properties (``weight``, ``delay``, ``receptor_type``),
        plasticity parameters (time constants, LUTs, thresholds), and controller state (synapse_id,
        readout timing, accumulators).

        Returns
        -------
        params : dict
            Dictionary mapping parameter names (str) to current values. Keys include:

            - ``'weight'`` (float): Current continuous synaptic weight
            - ``'delay'`` (float): Transmission delay in ms
            - ``'receptor_type'`` (int): Target receptor port
            - ``'tau_plus'`` (float): Causal time constant (ms)
            - ``'tau_minus_stdp'`` (float): Acausal time constant (ms)
            - ``'Wmax'`` (float): Maximum weight for LUT conversion
            - ``'weight_per_lut_entry'`` (float): Weight quantization step
            - ``'no_synapses'`` (int): Global synapse counter
            - ``'synapses_per_driver'`` (int): Synapses per readout cycle
            - ``'driver_readout_time'`` (float): Driver processing time (ms)
            - ``'readout_cycle_duration'`` (float): Full cycle duration (ms)
            - ``'lookuptable_0'`` (list[int]): LUT for (1,0) evaluation (16 entries)
            - ``'lookuptable_1'`` (list[int]): LUT for (0,1) evaluation (16 entries)
            - ``'lookuptable_2'`` (list[int]): LUT for (1,1) evaluation (16 entries)
            - ``'configbit_0'`` (list[int]): Comparator config for E_0 (4 entries)
            - ``'configbit_1'`` (list[int]): Comparator config for E_1 (4 entries)
            - ``'reset_pattern'`` (list[int]): Accumulator reset bits (6 entries)
            - ``'a_causal'`` (float): Current causal accumulator value
            - ``'a_acausal'`` (float): Current acausal accumulator value
            - ``'a_thresh_th'`` (float): Upper comparator threshold
            - ``'a_thresh_tl'`` (float): Lower comparator threshold
            - ``'init_flag'`` (bool): Initialization status
            - ``'synapse_id'`` (int): Assigned synapse identifier
            - ``'next_readout_time'`` (float): Next scheduled readout (ms)
            - ``'synapse_model'`` (str): Model name ('stdp_facetshw_synapse_hom')

        Notes
        -----
        This method provides a snapshot of the current state at the time of invocation. State
        variables like ``a_causal``, ``a_acausal``, and ``weight`` may change during subsequent
        spike processing.

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bp
            >>> import saiunit as u
            >>>
            >>> syn = bp.stdp_facetshw_synapse_hom(
            ...     weight=50.0,
            ...     tau_plus=15.0 * u.ms,
            ...     Wmax=100.0,
            ... )
            >>> syn.init_state()
            >>>
            >>> # Get initial state
            >>> params = syn.get()
            >>> print(f"Weight: {params['weight']}")
            Weight: 50.0
            >>> print(f"Tau plus: {params['tau_plus']} ms")
            Tau plus: 15.0 ms
            >>> print(f"Synapse model: {params['synapse_model']}")
            Synapse model: stdp_facetshw_synapse_hom
        """
        params = super().get()
        params['tau_plus'] = float(self.tau_plus)
        params['tau_minus_stdp'] = float(self.tau_minus_stdp)
        params['Wmax'] = float(self.Wmax)
        params['weight_per_lut_entry'] = float(self.weight_per_lut_entry)
        params['no_synapses'] = int(self.no_synapses)
        params['synapses_per_driver'] = int(self.synapses_per_driver)
        params['driver_readout_time'] = float(self.driver_readout_time)
        params['readout_cycle_duration'] = float(self.readout_cycle_duration)
        params['lookuptable_0'] = list(self.lookuptable_0)
        params['lookuptable_1'] = list(self.lookuptable_1)
        params['lookuptable_2'] = list(self.lookuptable_2)
        params['configbit_0'] = list(self.configbit_0)
        params['configbit_1'] = list(self.configbit_1)
        params['reset_pattern'] = list(self.reset_pattern)
        params['a_causal'] = float(self.a_causal)
        params['a_acausal'] = float(self.a_acausal)
        params['a_thresh_th'] = float(self.a_thresh_th)
        params['a_thresh_tl'] = float(self.a_thresh_tl)
        params['init_flag'] = bool(self.init_flag)
        params['synapse_id'] = int(self.synapse_id)
        params['next_readout_time'] = float(self.next_readout_time)
        params['synapse_model'] = 'stdp_facetshw_synapse_hom'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        r"""Validate synapse specification for connection-time parameter assignment.

        Ensures that model-level (common) properties are not included in per-synapse connection
        specifications. The ``_hom`` (homogeneous) variant requires all plasticity parameters to be
        set at the model level, not per-connection, matching NEST's hardware-constrained design.

        Parameters
        ----------
        syn_spec : dict or None
            Synapse specification dictionary passed during connection setup. Should only contain
            per-synapse properties like ``weight``, ``delay``, and ``receptor_type``. Must NOT
            contain plasticity parameters (time constants, LUTs, configuration bits, etc.).

        Raises
        ------
        ValueError
            If any of the following model-level parameters appear in ``syn_spec``:
            ``'tau_plus'``, ``'tau_minus_stdp'``, ``'Wmax'``, ``'weight_per_lut_entry'``,
            ``'no_synapses'``, ``'synapses_per_driver'``, ``'driver_readout_time'``,
            ``'readout_cycle_duration'``, ``'lookuptable_0'``, ``'lookuptable_1'``,
            ``'lookuptable_2'``, ``'configbit_0'``, ``'configbit_1'``, ``'reset_pattern'``.

        Notes
        -----
        This restriction enforces the homogeneous design constraint: all synapses using this model
        share the same plasticity parameters, matching the BrainScaleS hardware architecture where
        plasticity settings are global, not per-synapse.

        To customize plasticity parameters:

        - Set them at model instantiation: ``stdp_facetshw_synapse_hom(tau_plus=15*u.ms, ...)``
        - Or update via ``set()`` method: ``model.set(tau_plus=15*u.ms)``

        Per-synapse properties (``weight``, ``delay``, ``receptor_type``) CAN be specified in
        ``syn_spec`` as they represent individual connection strengths, not shared plasticity rules.

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bp
            >>>
            >>> syn = bp.stdp_facetshw_synapse_hom(tau_plus=20.0)
            >>>
            >>> # Valid: per-synapse weight/delay specification
            >>> syn.check_synapse_params({'weight': 50.0, 'delay': 1.5})
            >>> # No error raised
            >>>
            >>> # Invalid: attempting to set model-level parameter per-synapse
            >>> try:
            ...     syn.check_synapse_params({'tau_plus': 15.0})
            ... except ValueError as e:
            ...     print(e)
            tau_plus cannot be specified in connect-time synapse parameters for
            stdp_facetshw_synapse_hom; set common properties on the model itself
            (for example via CopyModel()/SetDefaults()).
        """
        if syn_spec is None:
            return
        disallowed = (
            'tau_plus',
            'tau_minus_stdp',
            'Wmax',
            'weight_per_lut_entry',
            'no_synapses',
            'synapses_per_driver',
            'driver_readout_time',
            'readout_cycle_duration',
            'lookuptable_0',
            'lookuptable_1',
            'lookuptable_2',
            'configbit_0',
            'configbit_1',
            'reset_pattern',
        )
        for key in disallowed:
            if key in syn_spec:
                raise ValueError(
                    f'{key} cannot be specified in connect-time synapse parameters '
                    'for stdp_facetshw_synapse_hom; set common properties on the '
                    'model itself (for example via CopyModel()/SetDefaults()).'
                )

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        tau_plus: ArrayLike | object = _UNSET,
        tau_minus_stdp: ArrayLike | object = _UNSET,
        Wmax: ArrayLike | object = _UNSET,
        weight_per_lut_entry: ArrayLike | object = _UNSET,
        no_synapses: ArrayLike | object = _UNSET,
        synapses_per_driver: ArrayLike | object = _UNSET,
        driver_readout_time: ArrayLike | object = _UNSET,
        readout_cycle_duration: ArrayLike | object = _UNSET,
        lookuptable_0: ArrayLike | object = _UNSET,
        lookuptable_1: ArrayLike | object = _UNSET,
        lookuptable_2: ArrayLike | object = _UNSET,
        configbit_0: ArrayLike | object = _UNSET,
        configbit_1: ArrayLike | object = _UNSET,
        reset_pattern: ArrayLike | object = _UNSET,
        a_causal: ArrayLike | object = _UNSET,
        a_acausal: ArrayLike | object = _UNSET,
        a_thresh_th: ArrayLike | object = _UNSET,
        a_thresh_tl: ArrayLike | object = _UNSET,
        init_flag: ArrayLike | object = _UNSET,
        synapse_id: ArrayLike | object = _UNSET,
        next_readout_time: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        r"""Update public parameters and mutable state variables.

        Provides NEST-compatible interface for modifying synapse properties and state after
        instantiation. All parameters are optional; only provided arguments are updated. Performs
        validation and automatically recomputes dependent values (e.g., ``readout_cycle_duration``
        when driver parameters change, ``weight_per_lut_entry`` when ``Wmax`` changes).

        Parameters
        ----------
        weight : float or array-like, optional
            New continuous synaptic weight (dimensionless).
        delay : Quantity[time] or array-like, optional
            New transmission delay. Must include units (e.g., ``1.5 * u.ms``).
        receptor_type : int, optional
            New receptor port identifier.
        tau_plus : Quantity[time] or array-like, optional
            New causal time constant. Must be positive.
        tau_minus_stdp : Quantity[time] or array-like, optional
            New acausal time constant. Must be positive.
        Wmax : float or array-like, optional
            New maximum weight. Automatically recomputes ``weight_per_lut_entry`` unless
            ``weight_per_lut_entry`` is also explicitly provided.
        weight_per_lut_entry : float or array-like, optional
            New weight quantization step. Overrides automatic computation from ``Wmax``.
        no_synapses : int or array-like, optional
            New global synapse count. Automatically recomputes ``readout_cycle_duration``.
        synapses_per_driver : int or array-like, optional
            New synapses-per-driver count. Must be positive. Automatically recomputes
            ``readout_cycle_duration``.
        driver_readout_time : float or array-like, optional
            New driver processing time (ms). Must be positive. Automatically recomputes
            ``readout_cycle_duration``.
        readout_cycle_duration : float or array-like, optional
            New cycle duration (ms). If provided, overrides automatic computation.
        lookuptable_0 : array-like of 16 ints, optional
            New LUT for (1,0) evaluation. All entries must be in [0, 15].
        lookuptable_1 : array-like of 16 ints, optional
            New LUT for (0,1) evaluation. All entries must be in [0, 15].
        lookuptable_2 : array-like of 16 ints, optional
            New LUT for (1,1) evaluation. All entries must be in [0, 15].
        configbit_0 : array-like of 4 ints, optional
            New comparator configuration for E_0.
        configbit_1 : array-like of 4 ints, optional
            New comparator configuration for E_1.
        reset_pattern : array-like of 6 ints, optional
            New accumulator reset pattern (6 binary flags).
        a_causal : float or array-like, optional
            New causal accumulator value.
        a_acausal : float or array-like, optional
            New acausal accumulator value.
        a_thresh_th : float or array-like, optional
            New upper comparator threshold.
        a_thresh_tl : float or array-like, optional
            New lower comparator threshold.
        init_flag : bool or array-like, optional
            New initialization status (True = initialized, False = uninitialized).
        synapse_id : int or array-like, optional
            New synapse identifier.
        next_readout_time : float or array-like, optional
            New next scheduled readout time (ms).
        post : Dynamics, optional
            New default postsynaptic target object.

        Raises
        ------
        ValueError
            - If ``tau_plus`` or ``tau_minus_stdp`` is non-positive
            - If ``synapses_per_driver`` is non-positive
            - If ``driver_readout_time`` is non-positive
            - If LUT entries are outside [0, 15] range
            - If LUT sizes are mismatched (must all be 16 entries)
            - If ``configbit_0`` or ``configbit_1`` do not have exactly 4 entries
            - If ``reset_pattern`` does not have exactly 6 entries
            - If any scalar parameter is non-finite (NaN or ±inf)

        Notes
        -----
        **Automatic Recomputation Logic**:

        1. When ``Wmax`` is updated (and ``weight_per_lut_entry`` is not):
           ``weight_per_lut_entry = Wmax / 15``

        2. When any of ``no_synapses``, ``synapses_per_driver``, or ``driver_readout_time`` is
           updated (and ``readout_cycle_duration`` is not):
           ``readout_cycle_duration = ceil(no_synapses / synapses_per_driver) * driver_readout_time``

        3. Explicit ``weight_per_lut_entry`` or ``readout_cycle_duration`` arguments override
           automatic computation.

        **State Persistence**:

        All updated values are also saved to internal ``_*0`` attributes (e.g., ``_a_causal0``,
        ``_tau_plus0``) to ensure consistent state restoration when ``init_state()`` is called.

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bp
            >>> import saiunit as u
            >>>
            >>> syn = bp.stdp_facetshw_synapse_hom(
            ...     weight=10.0,
            ...     tau_plus=20.0 * u.ms,
            ...     Wmax=100.0,
            ... )
            >>> syn.init_state()
            >>>
            >>> # Update time constants
            >>> syn.set(tau_plus=15.0 * u.ms, tau_minus_stdp=25.0 * u.ms)
            >>>
            >>> # Update weight (triggers discrete conversion on next readout)
            >>> syn.set(weight=50.0)
            >>>
            >>> # Reset accumulators to zero
            >>> syn.set(a_causal=0.0, a_acausal=0.0)
            >>>
            >>> # Change maximum weight (automatically updates weight_per_lut_entry)
            >>> syn.set(Wmax=200.0)
            >>> print(syn.weight_per_lut_entry)  # Now 200.0 / 15 = 13.333...
            13.333333333333334
        """
        new_tau_plus = self.tau_plus if tau_plus is _UNSET else self._to_scalar_time_ms(tau_plus, name='tau_plus')
        new_tau_minus_stdp = (
            self.tau_minus_stdp
            if tau_minus_stdp is _UNSET
            else self._to_scalar_time_ms(tau_minus_stdp, name='tau_minus_stdp')
        )
        self._validate_positive(float(new_tau_plus), name='tau_plus')
        self._validate_positive(float(new_tau_minus_stdp), name='tau_minus_stdp')

        new_Wmax = self.Wmax if Wmax is _UNSET else self._to_scalar_float(Wmax, name='Wmax')
        new_weight_per_lut_entry = (
            self.weight_per_lut_entry
            if weight_per_lut_entry is _UNSET
            else self._to_scalar_float(weight_per_lut_entry, name='weight_per_lut_entry')
        )

        new_no_synapses = (
            self.no_synapses
            if no_synapses is _UNSET
            else self._to_int_scalar(no_synapses, name='no_synapses')
        )
        new_synapses_per_driver = (
            self.synapses_per_driver
            if synapses_per_driver is _UNSET
            else self._to_int_scalar(synapses_per_driver, name='synapses_per_driver')
        )
        self._validate_synapses_per_driver(int(new_synapses_per_driver))

        new_driver_readout_time = (
            self.driver_readout_time
            if driver_readout_time is _UNSET
            else self._to_scalar_float(driver_readout_time, name='driver_readout_time')
        )
        self._validate_positive(float(new_driver_readout_time), name='driver_readout_time')

        new_readout_cycle_duration = (
            self.readout_cycle_duration
            if readout_cycle_duration is _UNSET
            else self._to_scalar_float(readout_cycle_duration, name='readout_cycle_duration')
        )

        new_a_causal = self.a_causal if a_causal is _UNSET else self._to_scalar_float(a_causal, name='a_causal')
        new_a_acausal = self.a_acausal if a_acausal is _UNSET else self._to_scalar_float(a_acausal, name='a_acausal')
        new_a_thresh_th = (
            self.a_thresh_th
            if a_thresh_th is _UNSET
            else self._to_scalar_float(a_thresh_th, name='a_thresh_th')
        )
        new_a_thresh_tl = (
            self.a_thresh_tl
            if a_thresh_tl is _UNSET
            else self._to_scalar_float(a_thresh_tl, name='a_thresh_tl')
        )
        new_init_flag = self.init_flag if init_flag is _UNSET else self._to_bool_scalar(init_flag, name='init_flag')
        new_synapse_id = self.synapse_id if synapse_id is _UNSET else self._to_int_scalar(synapse_id, name='synapse_id')
        new_next_readout_time = (
            self.next_readout_time
            if next_readout_time is _UNSET
            else self._to_scalar_float(next_readout_time, name='next_readout_time')
        )

        super_kwargs = {}
        if weight is not _UNSET:
            super_kwargs['weight'] = self._to_scalar_float(weight, name='weight')
        if delay is not _UNSET:
            super_kwargs['delay'] = delay
        if receptor_type is not _UNSET:
            super_kwargs['receptor_type'] = receptor_type
        if post is not _UNSET:
            super_kwargs['post'] = post
        if super_kwargs:
            super().set(**super_kwargs)

        self.tau_plus = float(new_tau_plus)
        self.tau_minus_stdp = float(new_tau_minus_stdp)

        self.Wmax = float(new_Wmax)
        if Wmax is not _UNSET:
            self.weight_per_lut_entry = float(self.Wmax / (len(self.lookuptable_0) - 1))
        if weight_per_lut_entry is not _UNSET:
            self.weight_per_lut_entry = float(new_weight_per_lut_entry)

        if readout_cycle_duration is not _UNSET:
            self.readout_cycle_duration = float(new_readout_cycle_duration)
        if no_synapses is not _UNSET:
            self.no_synapses = int(new_no_synapses)
            self._calc_readout_cycle_duration()
        if synapses_per_driver is not _UNSET:
            self.synapses_per_driver = int(new_synapses_per_driver)
            self._calc_readout_cycle_duration()
        if driver_readout_time is not _UNSET:
            self.driver_readout_time = float(new_driver_readout_time)
            self._calc_readout_cycle_duration()

        if lookuptable_0 is not _UNSET:
            lut0 = self._to_int_vector(
                lookuptable_0,
                name='lookuptable_0',
                exact_size=16,
                min_value=_LUT_ENTRY_MIN,
                max_value=_LUT_ENTRY_MAX,
            )
            if len(lut0) != len(self.lookuptable_1):
                raise ValueError('Look-up table has not 2^4 entries.')
            self.lookuptable_0 = lut0
        if lookuptable_1 is not _UNSET:
            lut1 = self._to_int_vector(
                lookuptable_1,
                name='lookuptable_1',
                exact_size=16,
                min_value=_LUT_ENTRY_MIN,
                max_value=_LUT_ENTRY_MAX,
            )
            if len(lut1) != len(self.lookuptable_0):
                raise ValueError('Look-up table has not 2^4 entries.')
            self.lookuptable_1 = lut1
        if lookuptable_2 is not _UNSET:
            lut2 = self._to_int_vector(
                lookuptable_2,
                name='lookuptable_2',
                exact_size=16,
                min_value=_LUT_ENTRY_MIN,
                max_value=_LUT_ENTRY_MAX,
            )
            if len(lut2) != len(self.lookuptable_0):
                raise ValueError('Look-up table has not 2^4 entries.')
            self.lookuptable_2 = lut2

        if configbit_0 is not _UNSET:
            self.configbit_0 = self._to_int_vector(configbit_0, name='configbit_0', exact_size=4)
        if configbit_1 is not _UNSET:
            self.configbit_1 = self._to_int_vector(configbit_1, name='configbit_1', exact_size=4)
        if reset_pattern is not _UNSET:
            self.reset_pattern = self._to_int_vector(reset_pattern, name='reset_pattern', exact_size=6)

        self.a_causal = float(new_a_causal)
        self.a_acausal = float(new_a_acausal)
        self.a_thresh_th = float(new_a_thresh_th)
        self.a_thresh_tl = float(new_a_thresh_tl)
        self.init_flag = bool(new_init_flag)
        self.synapse_id = int(new_synapse_id)
        self.next_readout_time = float(new_next_readout_time)

        self._a_causal0 = float(self.a_causal)
        self._a_acausal0 = float(self.a_acausal)
        self._a_thresh_th0 = float(self.a_thresh_th)
        self._a_thresh_tl0 = float(self.a_thresh_tl)
        self._init_flag0 = bool(self.init_flag)
        self._synapse_id0 = int(self.synapse_id)
        self._next_readout_time0 = float(self.next_readout_time)
        self._no_synapses0 = int(self.no_synapses)
        self._readout_cycle_duration0 = float(self.readout_cycle_duration)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        r"""Process presynaptic spike and schedule outgoing synaptic event.

        Implements the full NEST ``stdp_facetshw_synapse_hom::send`` protocol: controller
        initialization, readout-cycle-based weight update via look-up tables, nearest-neighbor
        spike pairing with postsynaptic history, and event scheduling with current weight. This is
        the core plasticity update method triggered by each presynaptic spike.

        **Processing Steps** (matching NEST order):

        1. **Controller initialization** (first spike only):
           - Assign unique ``synapse_id`` from global counter ``no_synapses``
           - Increment ``no_synapses`` and recompute ``readout_cycle_duration``
           - Calculate initial ``next_readout_time`` based on synapse ID and driver readout schedule
           - Set ``init_flag = True``

        2. **Readout-based weight update** (if current time exceeds ``next_readout_time``):
           - Convert continuous ``weight`` to 4-bit discrete representation via rounding
           - Evaluate two comparator functions ``E_0`` and ``E_1`` from accumulators and thresholds
           - Select LUT based on evaluation bits:

             * ``(1, 0)`` -- apply ``lookuptable_0`` (typically potentiation)
             * ``(0, 1)`` -- apply ``lookuptable_1`` (typically depression)
             * ``(1, 1)`` -- apply ``lookuptable_2`` (typically no change or combined rule)
             * ``(0, 0)`` -- no weight update

           - Reset accumulators to zero according to selected LUT's reset bits in ``reset_pattern``
           - Advance ``next_readout_time`` by ``readout_cycle_duration`` until it exceeds current time
           - Convert updated discrete weight back to continuous value

        3. **Spike pairing** (if postsynaptic history exists):
           - Query postsynaptic spikes in interval :math:`(t_{\text{last}} - d, t_{\text{pre}} - d]` where :math:`d` is dendritic delay.
           - Update ``a_causal`` using *first* post-spike in interval (pre-before-post timing).
           - Update ``a_acausal`` using *last* post-spike in interval (post-before-pre timing).

        4. **Event scheduling**:
           - Schedule spike event with weighted payload ``multiplicity * weight``
           - Deliver to target ``post`` at receptor port ``receptor_type``
           - Delivery time: current time + delay

        5. **State update**:
           - Record current spike timestamp in ``t_lastspike`` for next iteration

        Parameters
        ----------
        multiplicity : float or array-like, default: 1.0
            Presynaptic spike count or weight. Typical value is 1.0 for single spike. For
            ``multiplicity=0``, no event is generated (early return). The weighted payload sent to
            the postsynaptic target is ``multiplicity * weight``.
        post : Dynamics, optional
            Target postsynaptic object. If not provided, uses the default ``self.post`` set at
            initialization or via ``set()``. Must implement ``add_current_input()`` or equivalent
            input reception method.
        receptor_type : int, optional
            Target receptor port on postsynaptic neuron. If not provided, uses ``self.receptor_type``.
            Allows routing to different synaptic channels (e.g., 0 for AMPA, 1 for GABA).

        Returns
        -------
        sent : bool
            ``True`` if an event was scheduled (``multiplicity != 0``), ``False`` otherwise.

        Raises
        ------
        ValueError
            - If ``readout_cycle_duration`` is zero or negative when attempting to advance
              ``next_readout_time`` (indicates invalid controller configuration)
            - If discrete weight index falls outside LUT bounds [0, 15] during LUT lookup
              (indicates weight or Wmax misconfiguration)

        Notes
        -----
        **Timing Semantics**:

        - All spike timestamps are on-grid (fixed time steps). Sub-step precise timing is not used.
        - Dendritic delay :math:`d` affects both event delivery and the temporal window for
          querying postsynaptic history. The history query interval is *delay-adjusted*:
          spikes are matched at times :math:`(t_{\text{last}} - d, t_{\text{pre}} - d]` rather
          than :math:`(t_{\text{last}}, t_{\text{pre}}]`, accounting for the fact that
          postsynaptic spikes reach the synapse earlier than they occur at the soma.

        **Accumulator Updates**:

        - ``a_causal`` grows when post-spikes occur *before* the last pre-spike (pre-before-post
          pairing → potentiation signal)
        - ``a_acausal`` grows when post-spikes occur *after* the current pre-spike (post-before-pre
          pairing → depression signal)
        - Only the *first* post-spike updates ``a_causal`` (oldest timing)
        - Only the *last* post-spike updates ``a_acausal`` (most recent timing)
        - This reduced pairing strategy minimizes computational cost while preserving essential
          timing information

        **Readout Cycle Behavior**:

        - Weight updates are *not* immediate but deferred to periodic readout cycles
        - Multiple presynaptic spikes can occur between readouts; only the last spike triggers
          the update check
        - If simulation time jumps significantly (e.g., after long pauses), ``next_readout_time``
          advances in multiple steps to catch up
        - This models hardware constraints where a controller sequentially updates synapse arrays

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bp
            >>> import saiunit as u
            >>>
            >>> # Create synapse and postsynaptic target
            >>> syn = bp.stdp_facetshw_synapse_hom(
            ...     weight=10.0,
            ...     delay=1.0 * u.ms,
            ...     tau_plus=20.0 * u.ms,
            ...     Wmax=100.0,
            ... )
            >>> syn.init_state()
            >>>
            >>> # Record post-spike before pre-spike (depression scenario)
            >>> syn.record_post_spike(1.0, t_spike_ms=8.0)
            >>>
            >>> # Process pre-spike at t=10 ms
            >>> sent = syn.send(1.0)  # multiplicity=1.0
            >>> print(f"Event sent: {sent}")
            Event sent: True
            >>>
            >>> # Check accumulator update
            >>> params = syn.get()
            >>> print(f"Acausal accumulator: {params['a_acausal']:.4f}")
            Acausal accumulator: 0.9048
            >>>
            >>> # No event for zero multiplicity
            >>> sent = syn.send(0.0)
            >>> print(f"Event sent: {sent}")
            Event sent: False
        """
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)
        t_spike = self._current_time_ms() + dt_ms

        if not self.init_flag:
            self.synapse_id = int(self.no_synapses)
            self.no_synapses += 1
            self._calc_readout_cycle_duration()
            self.next_readout_time = int(self.synapse_id / self.synapses_per_driver) * self.driver_readout_time
            self.init_flag = True

        if t_spike > self.next_readout_time:
            self.discrete_weight = self._weight_to_entry(float(self.weight), float(self.weight_per_lut_entry))

            eval_0 = self._eval_function(
                float(self.a_causal),
                float(self.a_acausal),
                float(self.a_thresh_th),
                float(self.a_thresh_tl),
                self.configbit_0,
            )
            eval_1 = self._eval_function(
                float(self.a_causal),
                float(self.a_acausal),
                float(self.a_thresh_th),
                float(self.a_thresh_tl),
                self.configbit_1,
            )

            if eval_0 and not eval_1:
                self.discrete_weight = self._lookup(self.discrete_weight, self.lookuptable_0)
                if self.reset_pattern[0]:
                    self.a_causal = 0.0
                if self.reset_pattern[1]:
                    self.a_acausal = 0.0
            elif (not eval_0) and eval_1:
                self.discrete_weight = self._lookup(self.discrete_weight, self.lookuptable_1)
                if self.reset_pattern[2]:
                    self.a_causal = 0.0
                if self.reset_pattern[3]:
                    self.a_acausal = 0.0
            elif eval_0 and eval_1:
                self.discrete_weight = self._lookup(self.discrete_weight, self.lookuptable_2)
                if self.reset_pattern[4]:
                    self.a_causal = 0.0
                if self.reset_pattern[5]:
                    self.a_acausal = 0.0

            if self.readout_cycle_duration <= 0.0:
                raise ValueError('readout_cycle_duration must be > 0 during active readout scheduling.')
            while t_spike > self.next_readout_time:
                self.next_readout_time += self.readout_cycle_duration

            self.weight = float(self._entry_to_weight(self.discrete_weight, self.weight_per_lut_entry))

        dendritic_delay = float(self.delay)
        hist = self._get_post_history_times(self.t_lastspike - dendritic_delay, t_spike - dendritic_delay)
        if hist:
            minus_dt_causal = self.t_lastspike - (hist[0] + dendritic_delay)
            assert minus_dt_causal < (-1.0 * _STDP_EPS)
            self.a_causal += math.exp(minus_dt_causal / self.tau_plus)

            minus_dt_acausal = (hist[-1] + dendritic_delay) - t_spike
            self.a_acausal += math.exp(minus_dt_acausal / self.tau_minus_stdp)

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

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
        r"""Advance synapse state by one time step.

        Orchestrates the full update cycle: delivers due events from the internal queue, records
        postsynaptic spikes into history, aggregates presynaptic input from all sources, and
        triggers plasticity updates via ``send()``. This method should be called once per simulation
        time step to maintain consistent event timing.

        **Update Sequence**:

        1. **Event delivery**: Pop and deliver all events scheduled for the current time step from
           the internal event queue. Delivered events invoke ``add_current_input()`` or equivalent
           methods on target postsynaptic objects.

        2. **Post-spike recording**: If ``post_spike > 0``, record the specified number of
           postsynaptic spikes at the current time (plus delay) into the internal history buffer.
           These spikes will be queried by subsequent presynaptic spikes for plasticity calculations.

        3. **Pre-spike aggregation**: Sum presynaptic input from:
           - Explicit ``pre_spike`` argument
           - Current inputs registered via ``add_current_input()`` (from projections/other sources)
           - Delta inputs registered via ``add_delta_input()`` (spike-like events)

        4. **Plasticity processing**: If aggregated pre-spike count is non-zero, invoke ``send()``
           to perform readout-based weight update, spike pairing, and event scheduling.

        Parameters
        ----------
        pre_spike : float or array-like, default: 0.0
            Explicit presynaptic spike count for this time step. Typically 0.0 or 1.0. Can be a
            fractional value for rate-based approximations. This value is *added* to any inputs
            accumulated via ``add_current_input()`` / ``add_delta_input()``.
        post_spike : float or array-like, default: 0.0
            Postsynaptic spike count for this time step. Must be non-negative. Used for recording
            spike history for plasticity calculations. Common values: 0.0 (no spike) or 1.0 (spike).
        post : Dynamics, optional
            Target postsynaptic object for event delivery. If not provided, uses ``self.post``.
        receptor_type : int, optional
            Target receptor port. If not provided, uses ``self.receptor_type``.

        Returns
        -------
        delivered : int
            Number of events delivered to postsynaptic targets during this time step. Equals the
            number of events that were scheduled for delivery at the current time. Can be 0 if no
            events were due, or ≥1 if delayed events arrived.

        Notes
        -----
        **Integration with BrainPy Projections**:

        When used within a ``Projection`` or similar container, presynaptic spikes from multiple
        sources are typically accumulated via ``add_current_input()`` calls. The ``pre_spike``
        argument provides an *additional* explicit input, useful for:

        - Manual spike injection in testing/debugging
        - Direct neuron-to-synapse connections without projections
        - Feedforward external stimulation

        The total presynaptic drive is:
        ``total_pre = pre_spike + sum(current_inputs) + sum(delta_inputs)``

        **Timing Precision**:

        - All spike times are on-grid (rounded to simulation time steps)
        - ``post_spike`` timestamp is ``current_time + delay`` (same convention as ``send()``)
        - Event delivery respects the internal event queue's scheduling, accounting for synaptic
          delays set via ``delay`` parameter

        **State Consistency**:

        This method does *not* clear the postsynaptic history buffer. For long simulations, history
        grows unbounded unless manually cleared (not implemented in current version). Future
        enhancements may add automatic history pruning for spikes older than ``tau_plus + tau_minus``.

        **Performance Considerations**:

        - Event delivery is O(n_events_due) per call
        - Post-spike recording is O(post_spike) per call
        - Pre-spike processing triggers full ``send()`` logic, including LUT lookups and history
          queries, which is O(1) per call (reduced pairing strategy)

        Examples
        --------
        .. code-block:: python

            >>> import brainpy.state as bp
            >>> import saiunit as u
            >>> import brainstate as bst
            >>>
            >>> # Setup simulation context
            >>> with bst.environ.context(dt=0.1 * u.ms):
            ...     syn = bp.stdp_facetshw_synapse_hom(
            ...         weight=10.0,
            ...         delay=1.0 * u.ms,
            ...         tau_plus=20.0 * u.ms,
            ...     )
            ...     syn.init_state()
            ...
            ...     # Time step 0: no activity
            ...     delivered = syn.update(pre_spike=0.0, post_spike=0.0)
            ...     print(f"Step 0: delivered {delivered} events")
            ...
            ...     # Time step 1: post-spike only
            ...     delivered = syn.update(pre_spike=0.0, post_spike=1.0)
            ...     print(f"Step 1: delivered {delivered} events")
            ...
            ...     # Time step 2: pre-spike (triggers plasticity)
            ...     delivered = syn.update(pre_spike=1.0, post_spike=0.0)
            ...     print(f"Step 2: delivered {delivered} events")
            ...
            ...     # Check accumulator update from spike pairing
            ...     params = syn.get()
            ...     print(f"Causal accumulator: {params['a_causal']:.4f}")
            Step 0: delivered 0 events
            Step 1: delivered 0 events
            Step 2: delivered 0 events
            Causal accumulator: 0.9048
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
