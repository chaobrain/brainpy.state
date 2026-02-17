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

    Probabilistic synapse model with short-term plasticity based on stochastic
    quantal vesicle release. Implements the Tsodyks-Markram model with
    discrete release sites, where each presynaptic spike triggers
    probabilistic release from a finite pool of vesicle docking sites.

    Mathematical Model
    ------------------

    Each synapse maintains four state variables:

    - ``u`` -- dynamic release probability per site (evolves with facilitation)
    - ``n`` -- total number of release sites (fixed)
    - ``a`` -- currently available (recovered) release sites
    - ``t_lastspike`` -- timestamp of last presynaptic spike

    **1. State Evolution Between Spikes**

    Upon arrival of a presynaptic spike at time :math:`t_s`, if a previous
    spike occurred at :math:`t_{\mathrm{last}}` (i.e., ``t_lastspike >= 0``):

    .. math::
        h = t_s - t_{\mathrm{last}}

        p_{\mathrm{decay}} = e^{-h/\tau_{\mathrm{rec}}}

        u_{\mathrm{decay}} =
          \begin{cases}
            0, & \tau_{\mathrm{fac}} < 10^{-10} \\
            e^{-h/\tau_{\mathrm{fac}}}, & \text{otherwise}
          \end{cases}

        u \leftarrow U + u(1 - U) \, u_{\mathrm{decay}}

    The release probability ``u`` undergoes:

    - Depression baseline: decays toward baseline ``U``
    - Facilitation: modulated by exponential decay with time constant
      :math:`\tau_{\mathrm{fac}}`
    - Special case: if :math:`\tau_{\mathrm{fac}} < 10^{-10}`, no
      facilitation occurs (:math:`u_{\mathrm{decay}} = 0`), and ``u`` is
      reset to ``U``

    **2. Stochastic Recovery of Release Sites**

    Depleted sites recover probabilistically. For each of the :math:`n - a`
    depleted sites, an independent Bernoulli trial with success probability
    :math:`1 - p_{\mathrm{decay}}` determines recovery. This implements
    exponential-distributed recovery times in discrete form.

    **3. Stochastic Vesicle Release**

    For each of the ``a`` currently available sites, an independent Bernoulli
    trial with success probability ``u`` determines whether that site releases
    a vesicle. Let :math:`n_{\mathrm{release}}` denote the total number of
    sites that released.

    **4. Synaptic Transmission**

    If :math:`n_{\mathrm{release}} > 0`:

    .. math::
        w_{\mathrm{eff}} = n_{\mathrm{release}} \cdot w

    where ``w`` is the baseline per-site weight. The effective weight
    :math:`w_{\mathrm{eff}}` is delivered to the postsynaptic neuron after
    the synaptic delay. The number of available sites is updated:

    .. math::
        a \leftarrow a - n_{\mathrm{release}}

    If :math:`n_{\mathrm{release}} = 0`, no event is sent (synaptic failure).

    **5. Timestamp Update**

    Regardless of whether an event was transmitted, the last spike time is
    updated:

    .. math::
        t_{\mathrm{lastspike}} \leftarrow t_s

    **Implementation Fidelity**

    The update ordering matches NEST ``models/quantal_stp_synapse.h::send``
    exactly:

    1. Update ``u`` (facilitation)
    2. Recover depleted sites
    3. Draw released sites
    4. Send event if any release occurred
    5. Update ``t_lastspike``

    **Probabilistic Properties**

    - **Expected release**: :math:`\mathbb{E}[n_{\mathrm{release}}] = a \cdot u`
    - **Variance**: :math:`\mathrm{Var}[n_{\mathrm{release}}] = a \cdot u \cdot (1 - u)`
    - **Failure probability**: :math:`P(\text{no release}) = (1 - u)^a`

    For small ``u`` and moderate ``a``, the distribution of released vesicles
    is approximately Poisson with rate :math:`\lambda = a \cdot u`.

    **Event Timing Semantics**

    As in NEST, the model operates on discrete spike timestamps and ignores
    sub-step temporal offsets. Each event processed at simulation step ``t``
    uses the on-grid timestamp ``t + dt`` for all decay calculations.

    Parameters
    ----------
    weight : float or array_like, optional
        Baseline per-site synaptic weight ``w`` (dimensionless or with units).
        The effective transmitted weight is :math:`w_{\mathrm{eff}} =
        n_{\mathrm{release}} \cdot w`. Default: ``1.0``.
    delay : float or array_like, optional
        Synaptic transmission delay. Must be positive and include time units.
        Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Target receptor port identifier on the postsynaptic neuron. Used to
        route synaptic input to specific receptor channels. Default: ``0``.
    U : float or array_like, optional
        Baseline release probability per site. Must be in ``[0, 1]``. Controls
        the equilibrium facilitation level. Default: ``0.5``.
    u : float or array_like, optional
        Initial release probability per site. Must be in ``[0, 1]``. If not
        specified, defaults to ``U``. Default: ``U``.
    n : int or array_like, optional
        Total number of release sites (vesicle docking sites). Must be a
        non-negative integer. Default: ``1``.
    a : int or array_like, optional
        Initial number of available (recovered) release sites. Must be a
        non-negative integer not exceeding ``n``. If not specified, defaults
        to ``n`` (all sites initially available). Default: ``n``.
    tau_rec : float or array_like, optional
        Recovery time constant for depleted sites. Must be positive and include
        time units. Governs the rate of site replenishment. Default:
        ``800.0 * u.ms``.
    tau_fac : float or array_like, optional
        Facilitation time constant. Must be non-negative and include time units.
        If zero or very small (:math:`< 10^{-10}` ms), facilitation is disabled
        and ``u`` is reset to ``U`` on each spike. Default: ``0.0 * u.ms``
        (no facilitation).
    post : object, optional
        Default postsynaptic receiver object. Can be overridden in ``send()``
        and ``update()`` calls.
    name : str, optional
        Unique identifier for this synapse instance.

    Parameter Mapping
    -----------------

    Direct correspondence with NEST ``quantal_stp_synapse``:

    +-------------------+--------------------+-----------------------------------+
    | brainpy.state     | NEST               | Description                       |
    +===================+====================+===================================+
    | ``weight``        | ``weight``         | Per-site baseline weight          |
    +-------------------+--------------------+-----------------------------------+
    | ``delay``         | ``delay``          | Transmission delay (ms)           |
    +-------------------+--------------------+-----------------------------------+
    | ``receptor_type`` | ``receptor_type``  | Receptor port identifier          |
    +-------------------+--------------------+-----------------------------------+
    | ``U``             | ``U``              | Baseline release probability      |
    +-------------------+--------------------+-----------------------------------+
    | ``u``             | ``u``              | Current release probability       |
    +-------------------+--------------------+-----------------------------------+
    | ``n``             | ``n``              | Total release sites               |
    +-------------------+--------------------+-----------------------------------+
    | ``a``             | ``a``              | Available release sites           |
    +-------------------+--------------------+-----------------------------------+
    | ``tau_rec``       | ``tau_rec``        | Recovery time constant (ms)       |
    +-------------------+--------------------+-----------------------------------+
    | ``tau_fac``       | ``tau_fac``        | Facilitation time constant (ms)   |
    +-------------------+--------------------+-----------------------------------+

    Notes
    -----
    **Random Number Generation**

    All probabilistic operations use the NumPy global random number generator
    (``np.random.random()``). For reproducible simulations, seed the generator
    with ``np.random.seed()`` before initialization.

    **State Initialization**

    Calling ``init_state()`` resets the mutable state variables (``u``, ``a``,
    ``t_lastspike``) to their configured initial values. The reset values are
    stored internally as ``_u0`` and ``_a0`` when parameters are set.

    **Computational Cost**

    The per-spike cost scales as :math:`O(n)` due to the independent Bernoulli
    trials for each site. For large ``n`` (e.g., :math:`n > 100`), consider
    using the continuous Tsodyks-Markram model (``tsodyks_synapse``) as an
    efficient approximation.

    **Biophysical Interpretation**

    - ``n``: number of active zones or docking sites
    - ``u``: vesicle release probability (calcium-dependent)
    - ``U``: baseline release probability (intrinsic to synapse type)
    - ``tau_rec``: vesicle recycling / replenishment time
    - ``tau_fac``: residual calcium decay time (controls facilitation)

    **Comparison with tsodyks_synapse**


    ``tsodyks_synapse`` uses continuous-valued resource variables and
    deterministic updates. ``quantal_stp_synapse`` uses discrete site counts
    and stochastic release, producing trial-to-trial variability consistent
    with experimental observations of synaptic unreliability.

    See Also
    --------
    tsodyks_synapse : Deterministic continuous Tsodyks-Markram model
    tsodyks2_synapse : Alternative Tsodyks-Markram formulation
    static_synapse : Static synaptic connection without plasticity

    References
    ----------
    .. [1] NEST source: ``models/quantal_stp_synapse.h`` and
           ``models/quantal_stp_synapse_impl.h``.
    .. [2] Fuhrmann G, Segev I, Markram H, Tsodyks MV (2002).
           Coding of temporal information by activity-dependent synapses.
           *Journal of Neurophysiology*, 87(1):140-148.
           DOI: `10.1152/jn.00258.2001 <https://doi.org/10.1152/jn.00258.2001>`_
    .. [3] Loebel A, Silberberg G, Helbig D, Markram H, Tsodyks MV,
           Richardson MJE (2009). Multiquantal release underlies the
           distribution of synaptic efficacies in the neocortex.
           *Frontiers in Computational Neuroscience*, 3:27.
           DOI: `10.3389/neuro.10.027.2009 <https://doi.org/10.3389/neuro.10.027.2009>`_
    .. [4] Maass W, Markram H (2002). Synapses as dynamic memory buffers.
           *Neural Networks*, 15(2):155-161.
           DOI: `10.1016/S0893-6080(01)00144-7 <https://doi.org/10.1016/S0893-6080(01)00144-7>`_

    Examples
    --------
    **Basic usage with default parameters**

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import brainunit as u
        >>> # Single release site, 50% release probability
        >>> syn = bp.quantal_stp_synapse(weight=1.0, U=0.5, n=1)
        >>> syn.init_state()

    **Multi-site synapse with facilitation**

    .. code-block:: python

        >>> # 10 release sites, facilitating synapse
        >>> syn = bp.quantal_stp_synapse(
        ...     weight=0.1,
        ...     U=0.3,
        ...     n=10,
        ...     tau_rec=800.0 * u.ms,
        ...     tau_fac=50.0 * u.ms
        ... )
        >>> syn.init_state()

    **Depressing synapse (no facilitation)**

    .. code-block:: python

        >>> # High initial release probability, slow recovery
        >>> syn = bp.quantal_stp_synapse(
        ...     weight=0.5,
        ...     U=0.8,
        ...     n=5,
        ...     tau_rec=1000.0 * u.ms,
        ...     tau_fac=0.0 * u.ms  # No facilitation
        ... )
        >>> syn.init_state()

    **Simulating paired-pulse response**

    .. code-block:: python

        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducibility
        >>> syn = bp.quantal_stp_synapse(
        ...     weight=1.0, U=0.5, n=10, tau_rec=500.0 * u.ms,
        ...     tau_fac=50.0 * u.ms
        ... )
        >>> syn.init_state()
        >>> # First pulse
        >>> response1 = syn.send(multiplicity=1.0)
        >>> # Second pulse 50 ms later (simulate time advance)
        >>> syn.t_lastspike = 0.0
        >>> # ... (advance simulation time by 50 ms)
        >>> response2 = syn.send(multiplicity=1.0)

    **Runtime parameter modification**

    .. code-block:: python

        >>> syn.set(U=0.7, tau_rec=300.0 * u.ms)
        >>> status = syn.get()
        >>> print(status['U'], status['tau_rec'])
        0.7 300.0
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
        r"""Initialize or reset synapse state to configured initial values.

        Resets the three mutable state variables to their initial configuration:

        - ``u``: reset to ``_u0`` (initial release probability)
        - ``a``: reset to ``_a0`` (initial available sites)
        - ``t_lastspike``: reset to ``-1.0`` (no prior spike)

        This method should be called before simulation or when resetting
        between trials.

        Parameters
        ----------
        batch_size : int, optional
            Ignored (retained for API compatibility with batch-aware models).
        **kwargs
            Additional keyword arguments (ignored).

        Notes
        -----
        The initial values ``_u0`` and ``_a0`` are cached internally when
        parameters are set via ``__init__()`` or ``set()``. Modifying ``u``
        or ``a`` directly during simulation does not affect these cached
        values.
        """
        del batch_size, kwargs
        super().init_state()
        self.u = float(self._u0)
        self.a = int(self._a0)
        self.t_lastspike = -1.0

    def get(self) -> dict:
        r"""Return current public parameters and mutable state.

        Retrieves all NEST-compatible parameters and the current values of
        mutable state variables. This method is useful for inspecting synapse
        configuration and monitoring state evolution during simulation.

        Returns
        -------
        dict
            Dictionary containing:

            - Inherited static synapse parameters (``weight``, ``delay``,
              ``receptor_type``)
            - ``U`` : float
                Baseline release probability
            - ``u`` : float
                Current release probability
            - ``tau_rec`` : float
                Recovery time constant (ms, unitless value)
            - ``tau_fac`` : float
                Facilitation time constant (ms, unitless value)
            - ``n`` : int
                Total number of release sites
            - ``a`` : int
                Current number of available sites
            - ``synapse_model`` : str
                Fixed string ``'quantal_stp_synapse'`` for NEST compatibility

        Notes
        -----
        The returned dictionary is a snapshot of the current state. Modifying
        the dictionary does not affect the synapse. Use ``set()`` to update
        parameters.

        Examples
        --------
        .. code-block:: python

            >>> syn = bp.quantal_stp_synapse(U=0.5, n=10)
            >>> syn.init_state()
            >>> status = syn.get()
            >>> print(status['u'], status['a'])
            0.5 10
        """
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
        r"""Set NEST-style public parameters and state variables.

        Updates synapse parameters and optionally resets mutable state. All
        parameters are validated before being applied. If validation fails,
        the synapse state remains unchanged.

        Parameters
        ----------
        weight : float or array_like, optional
            New baseline per-site weight. If not provided, current value is
            retained.
        delay : float or array_like, optional
            New synaptic delay (must include time units). If not provided,
            current value is retained.
        receptor_type : int, optional
            New receptor port identifier. If not provided, current value is
            retained.
        U : float or array_like, optional
            New baseline release probability. Must be in ``[0, 1]``. If not
            provided, current value is retained.
        u : float or array_like, optional
            New current release probability. Must be in ``[0, 1]``. If not
            provided, current value is retained. Also updates ``_u0`` (the
            reset value used by ``init_state()``).
        n : int or array_like, optional
            New total number of release sites. Must be a non-negative integer.
            If not provided, current value is retained.
        a : int or array_like, optional
            New number of available sites. Must be a non-negative integer not
            exceeding ``n``. If not provided, current value is retained. Also
            updates ``_a0`` (the reset value used by ``init_state()``).
        tau_rec : float or array_like, optional
            New recovery time constant (must include time units and be
            positive). If not provided, current value is retained.
        tau_fac : float or array_like, optional
            New facilitation time constant (must include time units and be
            non-negative). If not provided, current value is retained.
        post : object, optional
            New default postsynaptic receiver. If not provided, current value
            is retained.

        Raises
        ------
        ValueError
            If any parameter fails validation:

            - ``U`` or ``u`` not in ``[0, 1]``
            - ``n`` or ``a`` not an integer
            - ``tau_rec <= 0``
            - ``tau_fac < 0``

        Notes
        -----
        **State Reset Values**

        Setting ``u`` or ``a`` updates both the current state and the cached
        initial values (``_u0``, ``_a0``). This means subsequent calls to
        ``init_state()`` will use the new values as the reset target.

        **Atomic Update**

        All validations are performed before any state is modified. If any
        parameter is invalid, the entire operation is aborted without
        side effects.

        Examples
        --------
        .. code-block:: python

            >>> syn = bp.quantal_stp_synapse(U=0.5, n=10)
            >>> syn.set(U=0.7, tau_rec=300.0 * u.ms)
            >>> syn.set(a=5)  # Partially deplete available sites
            >>> syn.init_state()  # Now resets to a=5 (new initial value)
        """
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
        r"""Schedule one outgoing synaptic event with quantal STP dynamics.

        Processes a presynaptic spike by updating facilitation state, recovering
        depleted sites, drawing stochastic vesicle release, and scheduling
        transmission to the postsynaptic target. This method implements the
        NEST ``quantal_stp_synapse::send`` algorithm exactly.

        **Processing Steps**

        1. **Facilitation Update**: If a previous spike exists
           (``t_lastspike >= 0``), compute inter-spike interval and update
           release probability ``u`` according to decay and facilitation rules.

        2. **Site Recovery**: For each depleted site, perform a Bernoulli
           trial with recovery probability :math:`1 - e^{-h/\tau_{\mathrm{rec}}}`.

        3. **Vesicle Release**: For each available site, perform a Bernoulli
           trial with release probability ``u``. Count total released sites
           as ``n_release``.

        4. **Event Transmission**: If ``n_release > 0``, schedule a delayed
           event with effective weight :math:`w_{\mathrm{eff}} =
           n_{\mathrm{release}} \cdot w \cdot \text{multiplicity}` and update
           available sites: ``a <- a - n_release``.

        5. **Timestamp Update**: Store current spike time in ``t_lastspike``,
           regardless of whether transmission occurred.

        Parameters
        ----------
        multiplicity : float or array_like, optional
            Presynaptic spike count or scaling factor. The effective transmitted
            weight is :math:`w_{\mathrm{eff}} = n_{\mathrm{release}} \cdot w
            \cdot \text{multiplicity}`. If zero or negligible, processing is
            skipped. Default: ``1.0``.
        post : object, optional
            Target postsynaptic receiver. If ``None``, uses the default
            receiver set during initialization.
        receptor_type : int or array_like, optional
            Target receptor port. If ``None``, uses the default receptor type
            set during initialization.

        Returns
        -------
        bool
            ``True`` if at least one vesicle was released and an event was
            transmitted; ``False`` if synaptic failure occurred (no release)
            or if ``multiplicity`` was zero.

        Notes
        -----
        **Synaptic Failure**

        Due to stochastic release, transmission may fail even with positive
        ``multiplicity``. The failure probability is :math:`(1 - u)^a`. For
        small ``u`` or ``a``, failures are common.

        **Timestamp Semantics**

        The spike timestamp is computed as ``current_time + dt``, representing
        the end of the current simulation step. This matches NEST's on-grid
        spike timing convention.

        **Random Number Generation**

        Each call may invoke the RNG up to :math:`n` times (worst case: all
        sites depleted, all recover, all release). For reproducibility, seed
        ``np.random`` before simulation.

        Examples
        --------
        .. code-block:: python

            >>> import numpy as np
            >>> np.random.seed(42)
            >>> syn = bp.quantal_stp_synapse(weight=1.0, U=0.5, n=10)
            >>> syn.init_state()
            >>> transmitted = syn.send(multiplicity=1.0)
            >>> if transmitted:
            ...     print(f"Released vesicles, available sites now: {syn.a}")
            ... else:
            ...     print("Synaptic failure")
        """
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
        r"""Deliver due events and process current presynaptic input.

        Performs two operations in sequence:

        1. **Event Delivery**: Delivers all queued synaptic events scheduled
           for the current simulation step to their target postsynaptic
           receivers.

        2. **Input Processing**: Aggregates all current-step presynaptic input
           (from ``pre_spike`` argument and registered input sources via
           ``current_inputs`` and ``delta_inputs``), then calls ``send()`` if
           the total input is non-zero.

        This method is typically called once per simulation step as part of
        the projection's update loop.

        Parameters
        ----------
        pre_spike : float or array_like, optional
            Presynaptic spike input for the current step (dimensionless or
            with units). This value is summed with inputs from registered
            sources before processing. Default: ``0.0``.
        post : object, optional
            Target postsynaptic receiver for the outgoing event (if any). If
            ``None``, uses the default receiver.
        receptor_type : int or array_like, optional
            Target receptor port for the outgoing event (if any). If ``None``,
            uses the default receptor type.

        Returns
        -------
        int
            Number of synaptic events delivered to postsynaptic targets during
            this step. This count reflects events scheduled in prior steps,
            not the current ``send()`` call.

        Notes
        -----
        **Update Order**

        Event delivery precedes input processing. This means events generated
        in the current step are delivered ``delay`` steps later, consistent
        with NEST's event-driven semantics.

        **Input Aggregation**

        The method sums three input sources:

        - Direct ``pre_spike`` argument
        - Registered ``current_inputs`` (via ``add_current_input()``)
        - Registered ``delta_inputs`` (via ``add_delta_input()``)

        If the total is zero (within numerical tolerance), ``send()`` is
        skipped.

        Examples
        --------
        .. code-block:: python

            >>> syn = bp.quantal_stp_synapse(weight=1.0, U=0.5, n=10)
            >>> syn.init_state()
            >>> # Process spike input
            >>> delivered = syn.update(pre_spike=1.0)
            >>> # Check how many events were delivered this step
            >>> print(f"Delivered {delivered} events")
        """
        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        total = self.sum_current_inputs(pre_spike)
        total = self.sum_delta_inputs(total)
        if self._is_nonzero(total):
            self.send(total, post=post, receptor_type=receptor_type)

        return delivered
