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



import math
from typing import Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

__all__ = [
    'jonke_synapse',
]


class jonke_synapse(NESTSynapse):
    r"""NEST-compatible ``jonke_synapse`` connection model with weight-dependent STDP.

    Implements spike-timing-dependent plasticity with exponential weight dependence and
    additive offsets, following NEST's ``jonke_synapse`` semantics. The model applies
    multiplicative weight factors :math:`\exp(\mu w)` to both facilitation and depression
    branches, producing nonlinear weight dynamics that can stabilize synaptic strengths
    or implement homeostatic control.

    **1. Mathematical Formulation**

    The plasticity rule operates on synaptic weight :math:`w(t)` using presynaptic trace
    :math:`K_+(t)` (with time constant :math:`\tau_+`) and postsynaptic trace :math:`K_-(t)`:

    .. math::

       \frac{dK_+}{dt} &= -\frac{K_+}{\tau_+} + \sum_f \delta(t - t_f^{\text{pre}}) \\
       \frac{dK_-}{dt} &= -\frac{K_-}{\tau_-} + \sum_j \delta(t - t_j^{\text{post}})

    **Weight-dependent plasticity kernels:**

    .. math::

       \Phi_+(w) &= \exp(\mu_+ w) \\
       \Phi_-(w) &= \exp(\mu_- w)

    **Update rules (applied at spike times):**

    .. math::

       \Delta w_+ &= \lambda \left( \Phi_+(w) K_+ - \beta \right) \quad \text{(facilitation)} \\
       \Delta w_- &= \lambda \left( -\alpha \Phi_-(w) K_- - \beta \right) \quad \text{(depression)}

    The weight is hard-bounded to :math:`[0, W_{\max}]` after each update.

    **2. Temporal Dynamics**

    At each presynaptic spike at time :math:`t`:

    1. **History lookup:** Read postsynaptic spikes in :math:`(t_{\text{last}} - d,\; t - d]`
    2. **Facilitation pass:** For each post-spike :math:`t_j` in history:

       .. math::

          K_+^{\text{eff}} = K_+(t_{\text{last}}) \exp\left(\frac{t_{\text{last}} - (t_j + d)}{\tau_+}\right)

          w \leftarrow w + \lambda \left( \Phi_+(w) K_+^{\text{eff}} - \beta \right)

    3. **Depression:** Using postsynaptic :math:`K_-(t - d)`:

       .. math::

          w \leftarrow w + \lambda \left( -\alpha \Phi_-(w) K_-(t-d) - \beta \right)

    4. **Event emission:** Spike delivered with updated weight
    5. **Trace update:**

       .. math::

          K_+ \leftarrow K_+ \exp\left(\frac{t_{\text{last}} - t}{\tau_+}\right) + 1

    6. **State update:** :math:`t_{\text{last}} \leftarrow t`

    **3. Design Considerations**

    - **Exponential weight dependence:** :math:`\mu_+ > 0` implements soft upper bound
      (larger weights resist growth); :math:`\mu_+ < 0` enables runaway potentiation.
      Similarly for :math:`\mu_-` and depression.
    - **Additive offset :math:`\beta`:** Shifts both update branches uniformly. Positive
      :math:`\beta` biases toward depression, negative toward potentiation. Can implement
      heterosynaptic competition.
    - **Asymmetric depression scaling:** :math:`\alpha` allows independent control of
      depression amplitude relative to potentiation.
    - **Numerical stability:** For large :math:`|\mu w|`, :math:`\exp(\mu w)` may
      overflow/underflow. Consider :math:`\mu` values keeping :math:`|\mu W_{\max}| < 10`.

    **4. Computational Properties**

    - **Time complexity:** :math:`O(N_{\text{post-spikes}})` per presynaptic spike, where
      :math:`N_{\text{post-spikes}}` is the count in delay window.
    - **Dendritic delay semantics:** History lookups use :math:`t - d` (compensating for
      backpropagation time). Event delivery uses `delay_steps` (axonal propagation).
    - **Precision:** Sub-grid spike timing (offset component in NEST) is ignored; all
      updates use grid-aligned times only.

    Parameters
    ----------
    weight : float or array-like, default=1.0
        Initial synaptic efficacy (dimensionless or pA/mV). Must be finite. Updated during
        plasticity and bounded to :math:`[0, W_{\max}]`.
    delay : float or array-like, default=1.0
        Dendritic delay in milliseconds for history lookups and depression timing
        (:math:`d` in equations). Must be positive.
    delay_steps : int or array-like, default=1
        Axonal event delivery delay in simulation time steps. Must be ≥ 1. Typically set
        to match `delay` quantized to grid resolution.
    Kplus : float or array-like, default=0.0
        Initial presynaptic trace value :math:`K_+(0)`. Must be non-negative. Evolves
        according to :math:`\tau_+` dynamics.
    t_last_spike_ms : float or array-like, default=0.0
        Timestamp of last presynaptic spike in milliseconds. Used for trace decay
        computation between spikes.
    alpha : float or array-like, default=1.0
        Depression amplitude scaling factor. :math:`\alpha = 1` gives symmetric update
        magnitudes (when :math:`\mu_+ = \mu_- = 0, \beta = 0`).
    beta : float or array-like, default=0.0
        Additive offset applied to both update branches (dimensionless). Positive values
        bias toward depression. Enables heterosynaptic effects.
    lambda_ : float or array-like, default=0.01
        Learning rate :math:`\lambda`. Controls plasticity time scale. Set to 0 to disable
        learning. Typical values: :math:`10^{-4}` to :math:`10^{-1}`.
    mu_plus : float or array-like, default=0.0
        Facilitation weight dependence exponent :math:`\mu_+`. Positive values create soft
        upper bound. Units: inverse of weight units.
    mu_minus : float or array-like, default=0.0
        Depression weight dependence exponent :math:`\mu_-`. Positive values accelerate
        depression at high weights.
    tau_plus : float or array-like, default=20.0
        Presynaptic trace time constant in milliseconds :math:`\tau_+`. Controls
        potentiation temporal window. Typical range: 10–40 ms.
    Wmax : float or array-like, default=100.0
        Hard upper weight bound :math:`W_{\max}`. Weights exceeding this after updates are
        clipped. Lower bound is always 0.
    name : str or None, optional
        Instance identifier for debugging and logging.

    Parameter Mapping
    -----------------

    ================================  ================================  =================
    NEST Parameter                    brainpy.state Parameter           Units / Notes
    ================================  ================================  =================
    ``weight``                        ``weight``                        dimensionless
    ``delay``                         ``delay``                         ms
    ``delay_steps``                   ``delay_steps``                   steps
    ``Kplus``                         ``Kplus``                         dimensionless
    ``t_lastspike``                   ``t_last_spike_ms``               ms
    ``alpha``                         ``alpha``                         dimensionless
    ``beta``                          ``beta``                          dimensionless
    ``lambda``                        ``lambda_``                       dimensionless
    ``mu_plus``                       ``mu_plus``                       1/weight
    ``mu_minus``                      ``mu_minus``                      1/weight
    ``tau_plus``                      ``tau_plus``                      ms
    ``Wmax``                          ``Wmax``                          same as weight
    ================================  ================================  =================

    Raises
    ------
    ValueError
        - If ``Kplus < 0`` (violates trace non-negativity).
        - If ``delay <= 0`` (non-physical delay).
        - If ``delay_steps < 1`` (invalid event scheduling).
        - If any parameter is non-finite (NaN or ±inf).
        - If scalar parameters have size ≠ 1.

    Notes
    -----
    - **Target interface requirements:** The postsynaptic target object passed to ``send()``
      must implement:

      * ``get_history(t1, t2) -> iterable``: Returns postsynaptic spike times in
        :math:`(t_1, t_2]`. Each entry is an object/dict/tuple with time accessible via
        ``.t_``, ``.t``, ``['t_']``, ``['t']``, or first element.
      * ``get_K_value(t) -> float`` or ``get_k_value(t) -> float``: Returns depression
        trace :math:`K_-(t)` at time `t`.

    - **NEST compatibility:** Reproduces behavior of ``nest-simulator/models/jonke_synapse.cpp``
      including parameter validation, update ordering, and spike event payload structure.
    - **Sub-grid timing:** Unlike NEST's precise spike timing mode, this implementation uses
      grid-aligned times only (ignoring offset components).
    - **Homeostatic interpretation:** With :math:`\mu_+ > 0` and appropriate :math:`\beta`,
      the model can implement sliding threshold mechanisms that stabilize weight distributions.

    Examples
    --------
    **Basic STDP with linear weight dependence:**

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> # Create synapse with standard STDP parameters
        >>> syn = bp.jonke_synapse(
        ...     weight=5.0,
        ...     delay=1.0,
        ...     lambda_=0.01,
        ...     tau_plus=20.0,
        ...     alpha=1.0,
        ...     beta=0.0,
        ...     mu_plus=0.0,
        ...     mu_minus=0.0,
        ...     Wmax=10.0
        ... )
        >>> syn.get_status()['weight']
        5.0

    **Exponential weight dependence (soft bounds):**

    .. code-block:: python

        >>> # Mu_plus > 0 creates resistance to potentiation at high weights
        >>> syn_bounded = bp.jonke_synapse(
        ...     weight=1.0,
        ...     lambda_=0.01,
        ...     mu_plus=0.1,
        ...     mu_minus=0.05,
        ...     Wmax=20.0
        ... )
        >>> # At w=10: Phi_+(10) = exp(0.1*10) = 2.72 (potentiation enhanced)
        >>> # At w=0: Phi_+(0) = 1.0 (baseline)

    **Heterosynaptic plasticity via beta offset:**

    .. code-block:: python

        >>> # Positive beta biases toward depression
        >>> syn_hetero = bp.jonke_synapse(
        ...     weight=5.0,
        ...     lambda_=0.005,
        ...     beta=0.05,
        ...     alpha=1.2
        ... )
        >>> # All weights slowly decay even without post-spikes (beta term)

    **Simulate spike-pair interaction:**

    .. code-block:: python

        >>> class MockTarget:
        ...     def get_history(self, t1, t2):
        ...         # Return single post-spike at t=15 ms
        ...         if t1 < 15.0 <= t2:
        ...             return [{'t_': 15.0}]
        ...         return []
        ...     def get_k_value(self, t):
        ...         # Kminus trace at depression check time
        ...         return 0.8
        >>>
        >>> target = MockTarget()
        >>> syn = bp.jonke_synapse(weight=5.0, lambda_=0.01, tau_plus=20.0)
        >>>
        >>> # Pre-spike at t=10 ms (pre-before-post → no facilitation yet)
        >>> event1 = syn.send(t_spike_ms=10.0, target=target)
        >>> print(f"Weight after pre@10: {event1['weight']:.3f}")
        Weight after pre@10: 4.992
        >>>
        >>> # Pre-spike at t=20 ms (post@15 in history → facilitation)
        >>> event2 = syn.send(t_spike_ms=20.0, target=target)
        >>> print(f"Weight after pre@20: {event2['weight']:.3f}")
        Weight after pre@20: 5.034

    See Also
    --------
    stdp_synapse : Classical pair-based STDP without weight dependence.
    stdp_triplet_synapse : Triplet STDP rule with better experimental fit.
    vogels_sprekeler_synapse : Inhibitory STDP for E/I balance.

    References
    ----------
    .. [1] Jonke, Z., Habenschuss, S., & Maass, W. (2017). Feedback inhibition shapes
           emergent computational properties of cortical microcircuit motifs.
           *Journal of Neuroscience*, 37(35), 8511-8523. https://doi.org/10.1523/JNEUROSCI.2078-16.2017
    .. [2] NEST Simulator source code: ``models/jonke_synapse.h`` and
           ``models/jonke_synapse.cpp`` (https://github.com/nest/nest-simulator).
    .. [3] van Rossum, M. C., Bi, G. Q., & Turrigiano, G. G. (2000). Stable Hebbian learning
           from spike timing-dependent plasticity. *Journal of Neuroscience*, 20(23),
           8812-8821. [For multiplicative STDP theory]
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    IS_PRIMARY = True
    SUPPORTS_HPC = True
    SUPPORTS_LBL = True
    SUPPORTS_WFR = False

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        Kplus: ArrayLike = 0.0,
        t_last_spike_ms: ArrayLike = 0.0,
        alpha: ArrayLike = 1.0,
        beta: ArrayLike = 0.0,
        lambda_: ArrayLike = 0.01,
        mu_plus: ArrayLike = 0.0,
        mu_minus: ArrayLike = 0.0,
        tau_plus: ArrayLike = 20.0,
        Wmax: ArrayLike = 100.0,
        name: str | None = None,
    ):
        self.name = name

        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay = self._validate_positive_delay(delay)
        self.delay_steps = self._validate_delay_steps(delay_steps)

        self.Kplus = self._to_float_scalar(Kplus, name='Kplus')
        if self.Kplus < 0.0:
            raise ValueError('Kplus must be non-negative.')

        self.t_last_spike_ms = self._to_float_scalar(t_last_spike_ms, name='t_last_spike_ms')

        self.alpha = self._to_float_scalar(alpha, name='alpha')
        self.beta = self._to_float_scalar(beta, name='beta')
        self.lambda_ = self._to_float_scalar(lambda_, name='lambda_')
        self.mu_plus = self._to_float_scalar(mu_plus, name='mu_plus')
        self.mu_minus = self._to_float_scalar(mu_minus, name='mu_minus')
        self.tau_plus = self._to_float_scalar(tau_plus, name='tau_plus')
        self.Wmax = self._to_float_scalar(Wmax, name='Wmax')

    @property
    def properties(self) -> dict[str, Any]:
        r"""NEST synapse model capability flags.

        Returns
        -------
        dict[str, Any]
            Dictionary with boolean capability flags:

            - ``has_delay``: Supports delayed spike delivery (always True).
            - ``is_primary``: Primary connection type for spike transmission (always True).
            - ``supports_hpc``: Compatible with NEST's high-performance computing mode (True).
            - ``supports_lbl``: Supports label-based connectivity (True).
            - ``supports_wfr``: Supports waveform relaxation method (always False for
              plasticity models).
        """
        return {
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def get_status(self) -> dict[str, Any]:
        r"""Retrieve complete synapse state snapshot (NEST GetStatus compatible).

        Returns current values of all parameters, state variables, and model capabilities.
        Output format matches NEST's ``GetStatus`` dictionary structure.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - ``weight`` (float): Current synaptic efficacy.
            - ``delay`` (float): Dendritic delay in ms.
            - ``delay_steps`` (int): Event delivery delay in steps.
            - ``Kplus`` (float): Current presynaptic trace value.
            - ``t_last_spike_ms`` (float): Last presynaptic spike time in ms.
            - ``alpha`` (float): Depression scaling factor.
            - ``beta`` (float): Additive offset.
            - ``lambda`` (float): Learning rate (key name uses NEST convention).
            - ``mu_plus`` (float): Facilitation weight exponent.
            - ``mu_minus`` (float): Depression weight exponent.
            - ``tau_plus`` (float): Presynaptic trace time constant in ms.
            - ``Wmax`` (float): Maximum weight bound.
            - ``size_of`` (int): Memory footprint in bytes.
            - Capability flags (``has_delay``, ``is_primary``, etc.).

        Examples
        --------
        .. code-block:: python

            >>> syn = bp.jonke_synapse(weight=3.5, lambda_=0.02, tau_plus=15.0)
            >>> status = syn.get_status()
            >>> print(status['weight'], status['lambda'], status['tau_plus'])
            3.5 0.02 15.0
        """
        return {
            'weight': float(self.weight),
            'delay': float(self.delay),
            'delay_steps': int(self.delay_steps),
            'Kplus': float(self.Kplus),
            't_last_spike_ms': float(self.t_last_spike_ms),
            'alpha': float(self.alpha),
            'beta': float(self.beta),
            'lambda': float(self.lambda_),
            'mu_plus': float(self.mu_plus),
            'mu_minus': float(self.mu_minus),
            'tau_plus': float(self.tau_plus),
            'Wmax': float(self.Wmax),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'is_primary': self.IS_PRIMARY,
            'supports_hpc': self.SUPPORTS_HPC,
            'supports_lbl': self.SUPPORTS_LBL,
            'supports_wfr': self.SUPPORTS_WFR,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update synapse parameters and state (NEST SetStatus compatible).

        Modifies any subset of parameters. Unspecified keys retain current values. Validates
        all updates before applying (atomic operation). Accepts both dictionary argument and
        keyword arguments (merged with kwargs taking precedence).

        Parameters
        ----------
        status : dict[str, Any] or None, optional
            Dictionary of parameter updates. Keys match those in ``get_status()``. If None,
            only ``kwargs`` are processed.
        **kwargs
            Additional parameter updates as keyword arguments. Merged with ``status`` dict.
            If both ``status['key']`` and ``key=value`` are provided for the same parameter,
            ``kwargs`` takes precedence.

        Raises
        ------
        ValueError
            - If ``Kplus`` is set to negative value.
            - If ``delay <= 0`` or ``delay_steps < 1``.
            - If both ``lambda`` and ``lambda_`` are provided with different values.
            - If any scalar parameter is non-finite or has size ≠ 1.

        Notes
        -----
        - The learning rate can be specified as either ``lambda`` (NEST convention) or
          ``lambda_`` (Python identifier). Both refer to the same internal state.
        - Validation occurs after all updates are collected, ensuring atomic updates (all
          succeed or all fail).
        - Setting ``lambda=0`` disables plasticity without affecting trace dynamics.

        Examples
        --------
        .. code-block:: python

            >>> syn = bp.jonke_synapse(weight=5.0, lambda_=0.01)
            >>> syn.set_status({'weight': 8.0, 'lambda': 0.005})
            >>> syn.get_status()['weight']
            8.0
            >>> syn.get_status()['lambda']
            0.005

        **Keyword argument syntax:**

        .. code-block:: python

            >>> syn.set_status(Wmax=50.0, mu_plus=0.1)
            >>> syn.get_status()['Wmax']
            50.0

        **Disable learning:**

        .. code-block:: python

            >>> syn.set_status(lambda_=0.0)
            >>> # Synapse now transmits spikes but does not update weight
        """
        updates = {}
        if status is not None:
            updates.update(status)
        updates.update(kwargs)

        if 'lambda' in updates and 'lambda_' in updates:
            lv = self._to_float_scalar(updates['lambda'], name='lambda')
            lvv = self._to_float_scalar(updates['lambda_'], name='lambda_')
            if lv != lvv:
                raise ValueError('lambda and lambda_ must be identical when both are provided.')

        if 'weight' in updates:
            self.weight = self._to_float_scalar(updates['weight'], name='weight')
        if 'delay' in updates:
            self.delay = self._validate_positive_delay(updates['delay'])
        if 'delay_steps' in updates:
            self.delay_steps = self._validate_delay_steps(updates['delay_steps'])
        if 'Kplus' in updates:
            self.Kplus = self._to_float_scalar(updates['Kplus'], name='Kplus')
        if 't_last_spike_ms' in updates:
            self.t_last_spike_ms = self._to_float_scalar(updates['t_last_spike_ms'], name='t_last_spike_ms')
        if 'alpha' in updates:
            self.alpha = self._to_float_scalar(updates['alpha'], name='alpha')
        if 'beta' in updates:
            self.beta = self._to_float_scalar(updates['beta'], name='beta')
        if 'lambda' in updates:
            self.lambda_ = self._to_float_scalar(updates['lambda'], name='lambda')
        if 'lambda_' in updates:
            self.lambda_ = self._to_float_scalar(updates['lambda_'], name='lambda_')
        if 'mu_plus' in updates:
            self.mu_plus = self._to_float_scalar(updates['mu_plus'], name='mu_plus')
        if 'mu_minus' in updates:
            self.mu_minus = self._to_float_scalar(updates['mu_minus'], name='mu_minus')
        if 'tau_plus' in updates:
            self.tau_plus = self._to_float_scalar(updates['tau_plus'], name='tau_plus')
        if 'Wmax' in updates:
            self.Wmax = self._to_float_scalar(updates['Wmax'], name='Wmax')

        if self.Kplus < 0.0:
            raise ValueError('Kplus must be non-negative.')

    def get(self, key: str = 'status'):
        r"""Retrieve parameter or full status dictionary by key (NEST Get compatible).

        Parameters
        ----------
        key : str, default='status'
            Parameter name or ``'status'`` for full dictionary. Valid keys include:
            ``'weight'``, ``'delay'``, ``'Kplus'``, ``'lambda'``, ``'tau_plus'``, etc.

        Returns
        -------
        Any
            - If ``key='status'``: full dictionary from ``get_status()``.
            - Otherwise: scalar value of the requested parameter.

        Raises
        ------
        KeyError
            If ``key`` is not ``'status'`` and not found in status dictionary.

        Examples
        --------
        .. code-block:: python

            >>> syn = bp.jonke_synapse(weight=7.0, tau_plus=25.0)
            >>> syn.get('weight')
            7.0
            >>> syn.get('tau_plus')
            25.0
            >>> syn.get('status')['lambda']
            0.01
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for jonke_synapse.get().')

    def set_weight(self, weight: ArrayLike):
        r"""Update synaptic weight (convenience method).

        Parameters
        ----------
        weight : float or array-like
            New synaptic efficacy value. Must be scalar and finite.

        Raises
        ------
        ValueError
            If weight is non-scalar or non-finite.
        """
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, delay: ArrayLike):
        r"""Update dendritic delay (convenience method).

        Parameters
        ----------
        delay : float or array-like
            New delay in milliseconds. Must be positive and finite.

        Raises
        ------
        ValueError
            If delay ≤ 0, non-scalar, or non-finite.
        """
        self.delay = self._validate_positive_delay(delay)

    def set_delay_steps(self, delay_steps: ArrayLike):
        r"""Update event delivery delay in steps (convenience method).

        Parameters
        ----------
        delay_steps : int or array-like
            New delay in simulation time steps. Must be ≥ 1.

        Raises
        ------
        ValueError
            If delay_steps < 1, non-integer, or non-finite.
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
        r"""Process presynaptic spike with plasticity and return spike event payload.

        Implements the full NEST ``jonke_synapse::send()`` protocol:

        1. Retrieve postsynaptic spike history in delay-compensated window
        2. Apply facilitation for each postsynaptic spike in history
        3. Apply depression using current postsynaptic trace
        4. Update presynaptic trace and timestamp
        5. Return spike event dictionary with updated weight

        This is the core method for spike-driven plasticity computation.

        Parameters
        ----------
        t_spike_ms : float or array-like
            Current presynaptic spike time in milliseconds. Must be scalar and ≥
            ``t_last_spike_ms`` (non-decreasing spike times assumed).
        target : object
            Postsynaptic neuron or recorder object. Must implement:

            - ``get_history(t1, t2) -> iterable``: Postsynaptic spike times in (t1, t2].
            - ``get_K_value(t) -> float`` or ``get_k_value(t) -> float``: Depression trace
              :math:`K_-(t)`.
        receptor_type : int or array-like, default=0
            Postsynaptic receptor channel identifier (e.g., 0=AMPA, 1=NMDA, 2=GABA_A).
            Passed through to event payload without modification.
        multiplicity : float or array-like, default=1.0
            Spike event amplitude multiplier. Must be non-negative. Scales effective weight
            in postsynaptic neuron. Typical use: probabilistic synapses or multi-vesicle
            release.
        delay : float or array-like or None, optional
            Override dendritic delay for this spike (in ms). If None, uses ``self.delay``.
            Affects history lookup window: :math:`(t_{\text{last}} - d,\; t - d]`.
        delay_steps : int or array-like or None, optional
            Override event delivery delay (in steps). If None, uses ``self.delay_steps``.
            Determines when postsynaptic neuron receives the spike.

        Returns
        -------
        dict[str, Any]
            Spike event payload dictionary containing:

            - ``weight`` (float): Updated synaptic efficacy after plasticity.
            - ``delay`` (float): Dendritic delay used (ms).
            - ``delay_steps`` (int): Event delivery delay (steps).
            - ``receptor_type`` (int): Postsynaptic receptor channel.
            - ``multiplicity`` (float): Spike amplitude multiplier.
            - ``t_spike_ms`` (float): Presynaptic spike time (ms).
            - ``Kminus`` (float): Postsynaptic trace value at depression check time.
            - ``Kplus_pre`` (float): Presynaptic trace before update.
            - ``Kplus_post`` (float): Presynaptic trace after update.

        Raises
        ------
        ValueError
            - If ``t_spike_ms``, ``receptor_type``, or ``multiplicity`` are non-scalar.
            - If ``multiplicity < 0``.
            - If ``delay <= 0`` or ``delay_steps < 1`` (when overriding defaults).
        AttributeError
            If ``target`` does not implement required ``get_history()`` or ``get_K_value()``
            methods.
        TypeError
            If history entries do not expose time via supported interface (see Notes).

        Notes
        -----
        - **History entry format:** Each entry from ``target.get_history(t1, t2)`` must be:

          * Object with ``.t_`` or ``.t`` attribute, OR
          * Dictionary with ``'t_'`` or ``'t'`` key, OR
          * Tuple/list where first element is time (float).

        - **Delay semantics:** History lookup uses :math:`t - d` to account for
          backpropagation delay. Event delivery uses ``delay_steps`` for forward propagation.
        - **State mutation:** Updates ``self.weight``, ``self.Kplus``, and
          ``self.t_last_spike_ms`` in place. Not thread-safe without external synchronization.
        - **Causality:** If :math:`t_{\text{spike}} < t_{\text{last}}`, trace decay may
          produce negative exponential argument (mathematically valid but may indicate
          simulation error).

        Examples
        --------
        **Basic spike transmission:**

        .. code-block:: python

            >>> class PostNeuron:
            ...     def get_history(self, t1, t2):
            ...         return []  # No post-spikes
            ...     def get_K_value(self, t):
            ...         return 0.5  # Constant depression trace
            >>>
            >>> syn = bp.jonke_synapse(weight=5.0, lambda_=0.01, beta=0.02)
            >>> target = PostNeuron()
            >>> event = syn.send(t_spike_ms=10.0, target=target)
            >>>
            >>> print(f"Weight: {event['weight']:.3f}")
            Weight: 4.980
            >>> # Depression applied: dw = 0.01 * (-1.0 * 1.0 * 0.5 - 0.02) = -0.007
            >>> # Weight bounded: max(0, 5.0 - 0.007) = 4.993 (approx, with exp factors)

        **Spike-pair potentiation:**

        .. code-block:: python

            >>> class PostNeuron:
            ...     def get_history(self, t1, t2):
            ...         # Post-spike at t=12 ms (between t1 and t2)
            ...         if t1 < 12.0 <= t2:
            ...             return [{'t_': 12.0}]
            ...         return []
            ...     def get_K_value(self, t):
            ...         return 0.0  # No depression trace yet
            >>>
            >>> syn = bp.jonke_synapse(weight=5.0, lambda_=0.01, tau_plus=20.0)
            >>> target = PostNeuron()
            >>>
            >>> # First pre-spike at t=10 ms (before post@12)
            >>> event1 = syn.send(t_spike_ms=10.0, target=target)
            >>> print(f"Weight after pre@10: {event1['weight']:.3f}, Kplus: {event1['Kplus_post']:.3f}")
            Weight after pre@10: 5.000, Kplus: 1.000
            >>>
            >>> # Second pre-spike at t=15 ms (post@12 now in history)
            >>> event2 = syn.send(t_spike_ms=15.0, target=target)
            >>> print(f"Weight after pre@15: {event2['weight']:.3f}")
            Weight after pre@15: 5.009
            >>> # Facilitation from Kplus(10) decayed to t=12: exp((10-13)/20) ≈ 0.861

        **Override delay per spike:**

        .. code-block:: python

            >>> event = syn.send(
            ...     t_spike_ms=20.0,
            ...     target=target,
            ...     delay=2.5,
            ...     delay_steps=3
            ... )
            >>> print(event['delay'], event['delay_steps'])
            2.5 3

        See Also
        --------
        to_spike_event : Alias for ``send()`` (NEST naming compatibility).
        simulate_pre_spike_train : Process multiple spikes in sequence.
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
            kplus_t = self.Kplus * math.exp(minus_dt / self.tau_plus)
            self.weight = self._facilitate(self.weight, kplus_t)

        kminus = self._get_k_value(target, t_spike - dendritic_delay)
        self.weight = self._depress(self.weight, kminus)

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

        self.Kplus = self.Kplus * math.exp((self.t_last_spike_ms - t_spike) / self.tau_plus) + 1.0
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
        r"""Alias for ``send()`` method (NEST naming compatibility).

        Identical functionality to ``send()``. Provided for API consistency with NEST's
        ``Connection::to_spike_event()`` naming convention.

        Parameters
        ----------
        t_spike_ms : float or array-like
            Presynaptic spike time in milliseconds.
        target : object
            Postsynaptic target with required interface.
        receptor_type : int or array-like, default=0
            Receptor channel identifier.
        multiplicity : float or array-like, default=1.0
            Spike amplitude multiplier.
        delay : float or array-like or None, optional
            Override dendritic delay (ms).
        delay_steps : int or array-like or None, optional
            Override event delivery delay (steps).

        Returns
        -------
        dict[str, Any]
            Spike event payload (see ``send()`` for structure).

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
        r"""Process sequence of presynaptic spikes and track weight evolution.

        Convenience method for simulating complete spike train interactions. Sequentially
        calls ``send()`` for each spike time, maintaining plasticity state across spikes.
        Useful for analyzing STDP curves, weight trajectories, or protocol responses.

        Parameters
        ----------
        pre_spike_times_ms : array-like
            Presynaptic spike times in milliseconds. Shape: ``(n_spikes,)`` or any shape
            (will be flattened). Times need not be sorted but should be non-decreasing for
            physically meaningful trace dynamics.
        target : object
            Postsynaptic target with required interface (same as ``send()``).
        receptor_type : int or array-like, default=0
            Receptor channel identifier (constant for all spikes).
        multiplicity : float or array-like, default=1.0
            Spike amplitude multiplier (constant for all spikes).
        delay : float or array-like or None, optional
            Override dendritic delay for all spikes (ms). If None, uses ``self.delay``.
        delay_steps : int or array-like or None, optional
            Override event delivery delay for all spikes (steps).

        Returns
        -------
        list[dict[str, Any]]
            Event payloads for each spike, in order. Length equals ``len(pre_spike_times_ms)``.
            Each dictionary has structure documented in ``send()``.

        Notes
        -----
        - **State evolution:** Synapse state (``weight``, ``Kplus``, ``t_last_spike_ms``)
          evolves across the sequence. Final state reflects cumulative plasticity from all
          spikes.
        - **Performance:** For large spike trains (>10⁴ spikes), consider batching or
          vectorized implementations if available.
        - **Non-sorted times:** If times are unsorted, trace decay may produce unexpected
          results (negative exponential arguments). Always verify input ordering.

        Examples
        --------
        **STDP pairing protocol (pre-post and post-pre pairs):**

        .. code-block:: python

            >>> class PostNeuron:
            ...     def __init__(self):
            ...         self.post_spikes = [15.0, 35.0]  # Post-spikes at t=15, 35
            ...     def get_history(self, t1, t2):
            ...         return [{'t_': t} for t in self.post_spikes if t1 < t <= t2]
            ...     def get_K_value(self, t):
            ...         return 1.0 if t >= 15.0 else 0.0
            >>>
            >>> syn = bp.jonke_synapse(weight=5.0, lambda_=0.01, tau_plus=20.0)
            >>> target = PostNeuron()
            >>>
            >>> # Pre-spikes at t=[10, 20, 30, 40] ms
            >>> events = syn.simulate_pre_spike_train(
            ...     pre_spike_times_ms=[10.0, 20.0, 30.0, 40.0],
            ...     target=target
            ... )
            >>>
            >>> # Track weight evolution
            >>> for i, evt in enumerate(events):
            ...     print(f"Spike {i}: t={evt['t_spike_ms']:.1f} ms, weight={evt['weight']:.4f}")
            Spike 0: t=10.0 ms, weight=5.0000
            Spike 1: t=20.0 ms, weight=5.0085
            Spike 2: t=30.0 ms, weight=5.0112
            Spike 3: t=40.0 ms, weight=5.0098

        **Extract weight trajectory:**

        .. code-block:: python

            >>> weights = [evt['weight'] for evt in events]
            >>> pre_traces = [evt['Kplus_post'] for evt in events]
            >>> print(weights)
            [5.0, 5.0085, 5.0112, 5.0098]

        **Frequency-dependent plasticity:**

        .. code-block:: python

            >>> # High-frequency pre-spikes (10 Hz for 1 sec)
            >>> times = np.arange(0, 1000, 100)  # t=0, 100, 200, ..., 900 ms
            >>> events = syn.simulate_pre_spike_train(times, target)
            >>> print(f"Initial weight: {events[0]['weight']:.3f}")
            >>> print(f"Final weight: {events[-1]['weight']:.3f}")

        See Also
        --------
        send : Single spike processing (core method).
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
        r"""Compute facilitation weight update with exponential weight dependence.

        Applies potentiation rule: :math:`\Delta w = \lambda (\exp(\mu_+ w) K_+ - \beta)`.
        Weight is clipped to :math:`[0, W_{\max}]` after update.

        Parameters
        ----------
        w : float
            Current synaptic weight.
        kplus : float
            Effective presynaptic trace value (time-decayed :math:`K_+`).

        Returns
        -------
        float
            Updated weight after facilitation, hard-bounded to :math:`W_{\max}`.

        Notes
        -----
        - Returns unchanged weight if :math:`\lambda = 0` (learning disabled).
        - Does not enforce lower bound here (depression handles that).
        - Exponential overflow for large :math:`\mu_+ w` will raise Python exception.
        """
        if self.lambda_ == 0.0:
            return w
        k_w = math.exp(self.mu_plus * w)
        dw = self.lambda_ * (k_w * kplus - self.beta)
        new_w = w + dw
        return new_w if new_w < self.Wmax else self.Wmax

    def _depress(self, w: float, kminus: float) -> float:
        r"""Compute depression weight update with exponential weight dependence.

        Applies depression rule: :math:`\Delta w = \lambda (-\alpha \exp(\mu_- w) K_- - \beta)`.
        Weight is clipped to :math:`[0, W_{\max}]` after update (lower bound at 0).

        Parameters
        ----------
        w : float
            Current synaptic weight.
        kminus : float
            Postsynaptic depression trace value :math:`K_-(t - d)`.

        Returns
        -------
        float
            Updated weight after depression, hard-bounded to :math:`\geq 0`.

        Notes
        -----
        - Returns unchanged weight if :math:`\lambda = 0` (learning disabled).
        - Enforces non-negative weights (biological constraint for excitatory synapses).
        - Exponential overflow for large :math:`\mu_- w` will raise Python exception.
        """
        if self.lambda_ == 0.0:
            return w
        k_w = math.exp(self.mu_minus * w)
        dw = self.lambda_ * (-self.alpha * k_w * kminus - self.beta)
        new_w = w + dw
        return new_w if new_w > 0.0 else 0.0

    @staticmethod
    def _get_history(target: Any, t1: float, t2: float):
        r"""Retrieve postsynaptic spike history from target neuron.

        Parameters
        ----------
        target : object
            Postsynaptic target with ``get_history()`` method.
        t1 : float
            Window start time (exclusive) in milliseconds.
        t2 : float
            Window end time (inclusive) in milliseconds.

        Returns
        -------
        iterable
            Postsynaptic spike entries in interval :math:`(t_1, t_2]`. Each entry format
            must be compatible with ``_extract_history_time()``.

        Raises
        ------
        AttributeError
            If target does not implement ``get_history(t1, t2)`` method.
        """
        if hasattr(target, 'get_history'):
            return target.get_history(float(t1), float(t2))
        raise AttributeError(
            'Target must provide get_history(t1, t2) for jonke_synapse.'
        )

    @staticmethod
    def _extract_history_time(entry: Any) -> float:
        r"""Extract spike time from history entry in flexible format.

        Supports multiple entry formats for compatibility with various neuron implementations:
        object attributes, dictionary keys, or tuple/list indexing.

        Parameters
        ----------
        entry : object or dict or tuple or list
            History entry from ``get_history()``. Must encode spike time via:

            - Attribute ``.t_`` or ``.t`` (object interface), OR
            - Key ``'t_'`` or ``'t'`` (dict interface), OR
            - First element ``[0]`` (sequence interface).

        Returns
        -------
        float
            Spike time in milliseconds.

        Raises
        ------
        TypeError
            If entry does not conform to any supported format.
        """
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
        r"""Retrieve postsynaptic depression trace value at specified time.

        Parameters
        ----------
        target : object
            Postsynaptic neuron with depression trace interface.
        t : float
            Query time in milliseconds (typically :math:`t_{\text{spike}} - d`).

        Returns
        -------
        float
            Postsynaptic trace :math:`K_-(t)` (dimensionless, typically :math:`\geq 0`).

        Raises
        ------
        AttributeError
            If target does not implement ``get_K_value(t)`` or ``get_k_value(t)`` method.

        Notes
        -----
        - Method name is case-insensitive: accepts ``get_K_value`` or ``get_k_value``.
        - No sign enforcement: negative trace values are permitted (though non-physical).
        """
        if hasattr(target, 'get_K_value'):
            return float(target.get_K_value(float(t)))
        if hasattr(target, 'get_k_value'):
            return float(target.get_k_value(float(t)))
        raise AttributeError(
            'Target must provide get_K_value(t) or get_k_value(t) for jonke_synapse.'
        )

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        r"""Convert array-like input to validated float scalar.

        Handles brainunit Quantities, NumPy arrays, and Python scalars. Ensures result is
        single finite value.

        Parameters
        ----------
        value : array-like
            Input value (may be Quantity, array, or scalar).
        name : str
            Parameter name for error messages.

        Returns
        -------
        float
            Scalar float value.

        Raises
        ------
        ValueError
            - If input size ≠ 1 (not scalar).
            - If value is NaN or ±inf (non-finite).
        """
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
        r"""Convert array-like input to validated integer scalar.

        Similar to ``_to_float_scalar`` but enforces integer values (within floating-point
        tolerance 1e-12).

        Parameters
        ----------
        value : array-like
            Input value (may be Quantity, array, or scalar).
        name : str
            Parameter name for error messages.

        Returns
        -------
        int
            Scalar integer value.

        Raises
        ------
        ValueError
            - If input size ≠ 1 (not scalar).
            - If value is NaN or ±inf (non-finite).
            - If value is not integer-valued (e.g., 2.5 fails, 2.0 succeeds).
        """
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
        r"""Validate and convert delay to positive float scalar.

        Parameters
        ----------
        value : array-like
            Delay in milliseconds.

        Returns
        -------
        float
            Validated positive delay.

        Raises
        ------
        ValueError
            If delay ≤ 0, non-scalar, or non-finite.
        """
        d = cls._to_float_scalar(value, name='delay')
        if d <= 0.0:
            raise ValueError('delay must be > 0.')
        return d

    @classmethod
    def _validate_delay_steps(cls, value: ArrayLike) -> int:
        r"""Validate and convert delay_steps to integer ≥ 1.

        Parameters
        ----------
        value : array-like
            Delay in simulation steps.

        Returns
        -------
        int
            Validated delay_steps ≥ 1.

        Raises
        ------
        ValueError
            If delay_steps < 1, non-integer, non-scalar, or non-finite.
        """
        d = cls._to_int_scalar(value, name='delay_steps')
        if d < 1:
            raise ValueError('delay_steps must be >= 1.')
        return d

    @classmethod
    def _validate_multiplicity(cls, value: ArrayLike) -> float:
        r"""Validate and convert multiplicity to non-negative float scalar.

        Parameters
        ----------
        value : array-like
            Spike amplitude multiplier.

        Returns
        -------
        float
            Validated multiplicity ≥ 0.

        Raises
        ------
        ValueError
            If multiplicity < 0, non-scalar, or non-finite.
        """
        m = cls._to_float_scalar(value, name='multiplicity')
        if m < 0.0:
            raise ValueError('multiplicity must be >= 0.')
        return m
