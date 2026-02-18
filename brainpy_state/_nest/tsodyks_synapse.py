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

import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike

from .static_synapse import static_synapse

__all__ = [
    'tsodyks_synapse',
]


_UNSET = object()


class tsodyks_synapse(static_synapse):
    r"""NEST-compatible ``tsodyks_synapse`` connection model.

    ``tsodyks_synapse`` implements the short-term plasticity (STP) model of
    Tsodyks, Uziel, and Markram (2000), exhibiting both synaptic depression
    and facilitation. This synapse tracks three dynamic state variables—recovered
    resources ``x``, active resources ``y``, and utilization ``u``—that evolve
    between presynaptic spikes and are updated upon spike arrival.

    The model replicates NEST ``models/tsodyks_synapse.h`` exactly, including
    propagator computation, update ordering, and event timing semantics. Delay
    scheduling and receiver delivery inherit from :class:`static_synapse`.

    **1. Mathematical Model**

    State Variables
    ---------------

    - ``x``: Fraction of resources in the recovered (available) state
    - ``y``: Fraction of resources in the active (released) state
    - ``u``: Utilization (instantaneous release probability)
    - ``z = 1 - x - y``: Fraction of resources in the inactive (unavailable) state

    **Constraint:** :math:`x + y + z = 1` at all times; ``x, y, z`` are non-negative.

    **Continuous-time dynamics (between spikes):**

    .. math::

       \frac{du}{dt} = -\frac{u}{\tau_{\mathrm{fac}}}

       \frac{dy}{dt} = -\frac{y}{\tau_{\mathrm{psc}}}

       \frac{dz}{dt} = -\frac{z}{\tau_{\mathrm{rec}}}

       x = 1 - y - z

    where:

    - :math:`\tau_{\mathrm{fac}}` -- Facilitation time constant (ms). If :math:`\tau_{\mathrm{fac}} = 0`, facilitation is disabled and :math:`u` decays instantly to zero.
    - :math:`\tau_{\mathrm{psc}}` -- Time constant of synaptic current decay (ms).
    - :math:`\tau_{\mathrm{rec}}` -- Recovery time constant for inactive resources (ms).

    **Upon presynaptic spike at time** :math:`t_s`:

    Let :math:`h = t_s - t_{\mathrm{last}}` be the inter-spike interval since the last presynaptic spike.

    **Step 1: Propagate state from** :math:`t_{\mathrm{last}}` **to** :math:`t_s`:

    .. math::

       u \leftarrow u \cdot P_{uu}

       x \leftarrow x + P_{xy} \, y - P_{zz} \, z

       y \leftarrow y \cdot P_{yy}

    where the exact propagators are:

    .. math::

       P_{uu} = \begin{cases}
         0, & \tau_{\mathrm{fac}} = 0 \\
         e^{-h/\tau_{\mathrm{fac}}}, & \tau_{\mathrm{fac}} > 0
       \end{cases}

       P_{yy} = e^{-h/\tau_{\mathrm{psc}}}

       P_{zz} = e^{-h/\tau_{\mathrm{rec}}} - 1

       P_{xy} = \frac{P_{zz} \, \tau_{\mathrm{rec}} - (P_{yy} - 1) \, \tau_{\mathrm{psc}}}
                     {\tau_{\mathrm{psc}} - \tau_{\mathrm{rec}}}

    **Step 2: Spike-triggered facilitation**:

    .. math::

       u \leftarrow u + U (1 - u)

    where :math:`U` is the baseline utilization increment parameter.

    **Step 3: Resource release**:

    .. math::

       \Delta y = u \cdot x

    **Step 4: Update resource fractions**:

    .. math::

       x \leftarrow x - \Delta y

       y \leftarrow y + \Delta y

    **Step 5: Effective weight delivered to postsynaptic neuron**:

    .. math::

       w_{\mathrm{eff}} = \Delta y \cdot w

    where :math:`w` is the baseline synaptic weight.

    **2. Update Ordering and NEST Compatibility**

    This implementation preserves the exact update sequence from NEST
    ``models/tsodyks_synapse.h::send()``:

    1. Compute propagators :math:`P_{uu}, P_{yy}, P_{zz}, P_{xy}` from inter-spike interval :math:`h`
    2. Propagate utilization: ``u *= P_uu``
    3. Propagate recovered resources: ``x += P_xy * y - P_zz * z``
    4. Propagate active resources: ``y *= P_yy``
    5. Facilitation jump: ``u += U * (1 - u)``
    6. Compute release: ``delta_y = u * x``
    7. Update resources: ``x -= delta_y``, ``y += delta_y``
    8. Schedule weighted event: ``w_eff = delta_y * weight``

    **3. Event Timing Semantics**

    NEST evaluates this model using spike time stamps (on-grid times) and ignores
    precise sub-step offsets. This implementation follows the same convention:

    - Presynaptic spike detected at simulation step ``n``
    - Spike time stamp: :math:`t_{\mathrm{spike}} = t_n + dt`
    - Inter-spike interval: :math:`h = t_{\mathrm{spike}} - t_{\mathrm{lastspike}}`
    - Delivery time: :math:`t_{\mathrm{delivery}} = t_{\mathrm{spike}} + \mathrm{delay}`

    **4. Stability Constraints and Computational Implications**

    **Parameter Constraints:**

    - :math:`\tau_{\mathrm{psc}} > 0` (strictly positive)
    - :math:`\tau_{\mathrm{fac}} \geq 0` (zero disables facilitation)
    - :math:`\tau_{\mathrm{rec}} > 0` (strictly positive)
    - :math:`U \in [0, 1]`
    - :math:`x, y, u \in [0, 1]`
    - :math:`x + y \leq 1` (ensures :math:`z \geq 0`)

    Numerical Considerations
    ------------------------

    - Propagators :math:`P_{uu}, P_{yy}, P_{zz}` are computed using ``math.exp()``
      and ``math.expm1()`` for numerical stability.
    - The cross-propagator :math:`P_{xy}` involves division by :math:`\tau_{\mathrm{psc}} - \tau_{\mathrm{rec}}`.
      If these time constants are nearly equal, numerical precision may degrade.
      NEST does not provide a singular fallback; users should avoid
      :math:`\tau_{\mathrm{psc}} \approx \tau_{\mathrm{rec}}`.
    - All state variables are stored as Python floats (``float64`` precision).
    - Per-call cost is :math:`O(1)` (scalar operations only).

    **Behavioral Regimes:**

    - **Depression-dominated** (:math:`\tau_{\mathrm{fac}} = 0`, :math:`U > 0`):
      Repeated spikes deplete ``x``, reducing ``delta_y`` over time.
    - **Facilitation-dominated** (:math:`\tau_{\mathrm{fac}} > 0`, large :math:`U`):
      Utilization ``u`` grows with repeated spikes, increasing release.
    - **Mixed dynamics**: Both effects coexist, yielding complex short-term plasticity.

    Parameters
    ----------
    weight : ArrayLike, optional
        Baseline synaptic weight :math:`w` (dimensionless or with receiver-specific units).
        Scalar float or array-like. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic transmission delay :math:`d` in milliseconds. Must be ``> 0``.
        Quantized to integer time steps per :class:`static_synapse` conventions.
        Scalar with ``brainunit`` time dimension or dimensionless value interpreted
        as milliseconds. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Postsynaptic receptor port identifier (non-negative integer). Routes events
        to labeled input channels on the receiver neuron. Default: ``0``.
    tau_psc : ArrayLike, optional
        Time constant of synaptic current decay :math:`\tau_{\mathrm{psc}}` in milliseconds.
        Must be ``> 0``. Scalar with ``brainunit`` time dimension or dimensionless
        value interpreted as milliseconds. Default: ``3.0 * u.ms``.
    tau_fac : ArrayLike, optional
        Facilitation time constant :math:`\tau_{\mathrm{fac}}` in milliseconds.
        Must be ``>= 0``. Set to ``0.0 * u.ms`` to disable facilitation.
        Scalar with ``brainunit`` time dimension or dimensionless value interpreted
        as milliseconds. Default: ``0.0 * u.ms``.
    tau_rec : ArrayLike, optional
        Recovery (depression) time constant :math:`\tau_{\mathrm{rec}}` in milliseconds.
        Must be ``> 0``. Scalar with ``brainunit`` time dimension or dimensionless
        value interpreted as milliseconds. Default: ``800.0 * u.ms``.
    U : ArrayLike, optional
        Baseline utilization increment parameter :math:`U` (dimensionless).
        Must be in ``[0, 1]``. Determines the magnitude of facilitation per spike.
        Scalar float. Default: ``0.5``.
    x : ArrayLike, optional
        Initial fraction of recovered resources (dimensionless). Must be in ``[0, 1]``.
        Together with ``y``, must satisfy ``x + y <= 1``. Scalar float. Default: ``1.0``.
    y : ArrayLike, optional
        Initial fraction of active resources (dimensionless). Must be in ``[0, 1]``.
        Together with ``x``, must satisfy ``x + y <= 1``. Scalar float. Default: ``0.0``.
    u : ArrayLike, optional
        Initial utilization value (dimensionless). Must be in ``[0, 1]``.
        Scalar float. Default: ``0.0``.
    post : object, optional
        Default postsynaptic receiver neuron. If provided, this neuron will be the
        target for all :meth:`send` calls unless overridden by the ``post`` argument
        in :meth:`send` or :meth:`update`. Default: ``None``.
    name : str, optional
        Unique identifier for this synapse instance. Used for debugging and logging.
        Default: ``None`` (auto-generated).

    Parameter Mapping
    -----------------

    =========================  ===================  =========================================
    NEST Parameter             brainpy.state        Description
    =========================  ===================  =========================================
    ``weight``                 ``weight``           Baseline synaptic weight
    ``delay``                  ``delay``            Synaptic delay (ms)
    ``receptor_type``          ``receptor_type``    Postsynaptic receptor port
    ``tau_psc``                ``tau_psc``          Synaptic current time constant (ms)
    ``tau_fac``                ``tau_fac``          Facilitation time constant (ms)
    ``tau_rec``                ``tau_rec``          Recovery time constant (ms)
    ``U``                      ``U``                Utilization increment parameter
    ``x``                      ``x``                Recovered resources state variable
    ``y``                      ``y``                Active resources state variable
    ``u``                      ``u``                Utilization state variable
    =========================  ===================  =========================================

    Attributes
    ----------
    x : float
        Current fraction of recovered resources. Mutable state variable.
    y : float
        Current fraction of active resources. Mutable state variable.
    u : float
        Current utilization value. Mutable state variable.
    t_lastspike : float
        Time stamp of the last presynaptic spike (ms). Used to compute inter-spike
        intervals for propagator calculations.

    Notes
    -----
    - **Event Type**: This model transmits ``'spike'`` events only. Other event
      types (``'rate'``, ``'current'``, ``'conductance'``) are not supported.
    - **State Variables**: ``x``, ``y``, and ``u`` are mutable per-connection states.
      They can be inspected via :meth:`get` and modified via :meth:`set`.
    - **Initialization**: Calling ``init_state()`` resets the internal event queue
      and restores ``x``, ``y``, ``u`` to their initial values (``self._x0``,
      ``self._y0``, ``self._u0``). It also resets ``t_lastspike`` to ``0.0``.
    - **Scalar-Only**: All parameters and state variables are scalar floats. This
      model does not support vectorized per-connection parameters.
    - **No Precise Timing**: Unlike some NEST models with ``_ps`` variants, this
      implementation uses on-grid spike stamps and does not track sub-step offsets.

    See Also
    --------
    tsodyks_synapse_hom : Homogeneous variant with shared state across all connections.
    tsodyks2_synapse : Alternative Tsodyks model with different parameterization.
    static_synapse : Base class for non-plastic synaptic connections.

    References
    ----------
    .. [1] Tsodyks M, Uziel A, Markram H (2000). Synchrony generation in recurrent
           networks with frequency-dependent synapses. Journal of Neuroscience,
           20(RC50):1-5.
    .. [2] NEST source code: ``models/tsodyks_synapse.h`` and ``models/tsodyks_synapse.cpp``
           (https://github.com/nest/nest-simulator)
    .. [3] Tsodyks M, Markram H (1997). The neural code between neocortical pyramidal
           neurons depends on neurotransmitter release probability. PNAS, 94(2):719-723.

    Examples
    --------
    **1. Depression-dominated synapse (excitatory with depletion):**

    .. code-block:: python

       >>> import brainpy.state as bp
       >>> import brainunit as u
       >>> syn = bp.nest.tsodyks_synapse(
       ...     weight=1.0,
       ...     delay=1.5 * u.ms,
       ...     tau_psc=5.0 * u.ms,
       ...     tau_fac=0.0 * u.ms,   # no facilitation
       ...     tau_rec=800.0 * u.ms,
       ...     U=0.5,
       ...     x=1.0,
       ...     y=0.0,
       ...     u=0.0
       ... )

    **2. Facilitation-dominated synapse (inhibitory with strengthening):**

    .. code-block:: python

       >>> syn = bp.nest.tsodyks_synapse(
       ...     weight=-2.0,          # inhibitory
       ...     delay=1.0 * u.ms,
       ...     tau_psc=3.0 * u.ms,
       ...     tau_fac=200.0 * u.ms,  # strong facilitation
       ...     tau_rec=800.0 * u.ms,
       ...     U=0.15,
       ...     x=1.0,
       ...     y=0.0,
       ...     u=0.0
       ... )

    **3. Simulating short-term plasticity:**

    .. code-block:: python

       >>> import brainstate as bst
       >>> with bst.environ.context(dt=0.1 * u.ms):
       ...     syn.init_all_states()
       ...     # Simulate spike train at 50 Hz
       ...     spike_times = [0.0, 20.0, 40.0, 60.0, 80.0]  # ms
       ...     for t_spike in spike_times:
       ...         # Advance simulation to spike time
       ...         # ... (step simulation forward)
       ...         syn.send(multiplicity=1.0)
       ...         print(f"t={t_spike:.1f} ms: u={syn.u:.3f}, x={syn.x:.3f}, y={syn.y:.3f}")

    **4. Inspecting and modifying state:**

    .. code-block:: python

       >>> params = syn.get()
       >>> print(params['x'], params['y'], params['u'])
       1.0 0.0 0.0
       >>> syn.set(x=0.8, y=0.1, u=0.3)
       >>> print(syn.x, syn.y, syn.u)
       0.8 0.1 0.3

    **5. Multi-receptor connection:**

    .. code-block:: python

       >>> syn_ex = bp.nest.tsodyks_synapse(
       ...     weight=1.0, delay=1.0 * u.ms, receptor_type=0, U=0.5
       ... )
       >>> syn_in = bp.nest.tsodyks_synapse(
       ...     weight=-1.0, delay=1.0 * u.ms, receptor_type=1, U=0.25
       ... )
       >>> # Excitatory and inhibitory inputs routed to different receptor ports
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_psc: ArrayLike = 3.0 * u.ms,
        tau_fac: ArrayLike = 0.0 * u.ms,
        tau_rec: ArrayLike = 800.0 * u.ms,
        U: ArrayLike = 0.5,
        x: ArrayLike = 1.0,
        y: ArrayLike = 0.0,
        u: ArrayLike = 0.0,
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

        self.tau_psc = self._to_scalar_time_ms(tau_psc, name='tau_psc')
        self.tau_fac = self._to_scalar_time_ms(tau_fac, name='tau_fac')
        self.tau_rec = self._to_scalar_time_ms(tau_rec, name='tau_rec')
        self.U = self._to_scalar_unit_interval(U, name='U')

        x0 = self._to_scalar_float(x, name='x')
        y0 = self._to_scalar_float(y, name='y')
        u0 = self._to_scalar_unit_interval(u, name='u')

        self._validate_tau_psc(self.tau_psc)
        self._validate_tau_fac(self.tau_fac)
        self._validate_tau_rec(self.tau_rec)
        self._validate_xy_sum(x0, y0)

        self._x0 = float(x0)
        self._y0 = float(y0)
        self._u0 = float(u0)

        self.x = float(self._x0)
        self.y = float(self._y0)
        self.u = float(self._u0)
        self.t_lastspike = 0.0

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_unit_interval(value: ArrayLike, *, name: str) -> float:
        v = tsodyks_synapse._to_scalar_float(value, name=name)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"'{name}' must be in [0,1].")
        return float(v)

    @staticmethod
    def _validate_tau_psc(value: float):
        if value <= 0.0:
            raise ValueError("'tau_psc' must be > 0.")

    @staticmethod
    def _validate_tau_fac(value: float):
        if value < 0.0:
            raise ValueError("'tau_fac' must be >= 0.")

    @staticmethod
    def _validate_tau_rec(value: float):
        if value <= 0.0:
            raise ValueError("'tau_rec' must be > 0.")

    @staticmethod
    def _validate_xy_sum(x: float, y: float):
        if x + y > 1.0:
            raise ValueError('x + y must be <= 1.0.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize or reset all state variables to their configured initial values.

        Resets the internal event delivery queue (via ``super().init_state()``) and
        restores the short-term plasticity state variables ``x``, ``y``, ``u`` to
        their initial values (``self._x0``, ``self._y0``, ``self._u0``). Also resets
        the last spike time stamp to ``0.0``.

        Parameters
        ----------
        batch_size : int, optional
            Ignored. This scalar synapse model does not support batching.
        **kwargs
            Additional keyword arguments. Ignored.

        Notes
        -----
        - This method is typically called once at the start of a simulation or when
          resetting the network state.
        - After calling this method, the synapse behaves as if no presynaptic spikes
          have occurred yet (``t_lastspike = 0.0``).
        """
        del batch_size, kwargs
        super().init_state()
        self.x = float(self._x0)
        self.y = float(self._y0)
        self.u = float(self._u0)
        self.t_lastspike = 0.0

    def get(self) -> dict:
        r"""Return current public parameters and mutable state variables.

        Retrieves all NEST-visible synapse parameters, including the baseline weight,
        delay, receptor type (from ``super().get()``), time constants, utilization
        parameter, and current state variables ``x``, ``y``, ``u``.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``'weight'`` (float): Baseline synaptic weight
            - ``'delay'`` (float): Synaptic delay in ms
            - ``'receptor_type'`` (int): Postsynaptic receptor port
            - ``'tau_psc'`` (float): Synaptic current time constant in ms
            - ``'tau_fac'`` (float): Facilitation time constant in ms
            - ``'tau_rec'`` (float): Recovery time constant in ms
            - ``'U'`` (float): Utilization increment parameter
            - ``'x'`` (float): Current recovered resources
            - ``'y'`` (float): Current active resources
            - ``'u'`` (float): Current utilization value
            - ``'synapse_model'`` (str): Model identifier (``'tsodyks_synapse'``)

        Notes
        -----
        - The returned dictionary reflects the *current* state at the time of the call.
          State variables ``x``, ``y``, ``u`` evolve during simulation.
        - This method is compatible with NEST's ``GetStatus()`` semantics.

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.nest.tsodyks_synapse(U=0.5, tau_rec=800.0 * u.ms)
           >>> params = syn.get()
           >>> print(params['U'], params['tau_rec'], params['x'])
           0.5 800.0 1.0
        """
        params = super().get()
        params['tau_psc'] = float(self.tau_psc)
        params['tau_fac'] = float(self.tau_fac)
        params['tau_rec'] = float(self.tau_rec)
        params['U'] = float(self.U)
        params['x'] = float(self.x)
        params['y'] = float(self.y)
        params['u'] = float(self.u)
        params['synapse_model'] = 'tsodyks_synapse'
        return params

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        tau_psc: ArrayLike | object = _UNSET,
        tau_fac: ArrayLike | object = _UNSET,
        tau_rec: ArrayLike | object = _UNSET,
        U: ArrayLike | object = _UNSET,
        x: ArrayLike | object = _UNSET,
        y: ArrayLike | object = _UNSET,
        u: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        r"""Set NEST-style public parameters and state variables.

        Updates synapse parameters and/or state variables. Only parameters explicitly
        provided (not ``_UNSET``) are modified. All changes are validated before
        application. If ``x`` or ``y`` are updated, their sum is checked to ensure
        ``x + y <= 1``.

        Updating state variables ``x``, ``y``, or ``u`` also updates their initial
        values (``self._x0``, ``self._y0``, ``self._u0``), so subsequent calls to
        ``init_state()`` will restore to the newly set values.

        Parameters
        ----------
        weight : ArrayLike, optional
            New baseline synaptic weight. If not provided, weight is unchanged.
        delay : ArrayLike, optional
            New synaptic delay in ms. Must be ``> 0``. If not provided, delay is unchanged.
        receptor_type : int, optional
            New postsynaptic receptor port. Must be a non-negative integer. If not
            provided, receptor type is unchanged.
        tau_psc : ArrayLike, optional
            New synaptic current time constant in ms. Must be ``> 0``. If not provided,
            ``tau_psc`` is unchanged.
        tau_fac : ArrayLike, optional
            New facilitation time constant in ms. Must be ``>= 0``. If not provided,
            ``tau_fac`` is unchanged.
        tau_rec : ArrayLike, optional
            New recovery time constant in ms. Must be ``> 0``. If not provided,
            ``tau_rec`` is unchanged.
        U : ArrayLike, optional
            New utilization increment parameter. Must be in ``[0, 1]``. If not provided,
            ``U`` is unchanged.
        x : ArrayLike, optional
            New recovered resources value. Must be in ``[0, 1]``. Together with ``y``,
            must satisfy ``x + y <= 1``. If not provided, ``x`` is unchanged.
        y : ArrayLike, optional
            New active resources value. Must be in ``[0, 1]``. Together with ``x``,
            must satisfy ``x + y <= 1``. If not provided, ``y`` is unchanged.
        u : ArrayLike, optional
            New utilization value. Must be in ``[0, 1]``. If not provided, ``u`` is unchanged.
        post : object, optional
            New default postsynaptic receiver neuron. If not provided, receiver is unchanged.

        Raises
        ------
        ValueError
            If any parameter violates its constraint (e.g., ``tau_psc <= 0``, ``U`` out
            of range, ``x + y > 1``).

        Notes
        -----
        - This method is compatible with NEST's ``SetStatus()`` semantics.
        - Parameter validation occurs *before* any state is modified. If validation
          fails, no changes are applied.
        - Updating state variables mid-simulation can produce non-physical dynamics.
          Use with caution outside of initialization or testing contexts.

        Examples
        --------
        .. code-block:: python

           >>> syn = bp.nest.tsodyks_synapse(U=0.5)
           >>> syn.set(U=0.8, tau_rec=600.0 * u.ms)
           >>> print(syn.U, syn.tau_rec)
           0.8 600.0

           >>> syn.set(x=0.7, y=0.2)
           >>> print(syn.x, syn.y)
           0.7 0.2

           >>> syn.set(x=0.8, y=0.3)  # doctest: +SKIP
           ValueError: x + y must be <= 1.0.
        """
        new_tau_psc = (
            self.tau_psc
            if tau_psc is _UNSET
            else self._to_scalar_time_ms(tau_psc, name='tau_psc')
        )
        new_tau_fac = (
            self.tau_fac
            if tau_fac is _UNSET
            else self._to_scalar_time_ms(tau_fac, name='tau_fac')
        )
        new_tau_rec = (
            self.tau_rec
            if tau_rec is _UNSET
            else self._to_scalar_time_ms(tau_rec, name='tau_rec')
        )
        new_U = self.U if U is _UNSET else self._to_scalar_unit_interval(U, name='U')
        new_x = self.x if x is _UNSET else self._to_scalar_float(x, name='x')
        new_y = self.y if y is _UNSET else self._to_scalar_float(y, name='y')
        new_u = self.u if u is _UNSET else self._to_scalar_unit_interval(u, name='u')

        self._validate_tau_psc(float(new_tau_psc))
        self._validate_tau_fac(float(new_tau_fac))
        self._validate_tau_rec(float(new_tau_rec))
        self._validate_xy_sum(float(new_x), float(new_y))

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

        self.tau_psc = float(new_tau_psc)
        self.tau_fac = float(new_tau_fac)
        self.tau_rec = float(new_tau_rec)
        self.U = float(new_U)

        self.x = float(new_x)
        self.y = float(new_y)
        self.u = float(new_u)

        self._x0 = float(self.x)
        self._y0 = float(self.y)
        self._u0 = float(self.u)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        r"""Schedule one outgoing event with NEST ``tsodyks_synapse`` short-term plasticity dynamics.

        Processes a presynaptic spike event by:

        1. Propagating state variables ``u``, ``x``, ``y`` from the last spike time to the current spike time.
        2. Applying spike-triggered facilitation to ``u``.
        3. Computing the released resource fraction ``delta_y = u * x``.
        4. Updating resource fractions ``x`` and ``y``.
        5. Scheduling a weighted event (``delta_y * weight * multiplicity``) for delivery to the postsynaptic neuron.

        The update ordering exactly matches NEST ``models/tsodyks_synapse.h::send()``.

        Parameters
        ----------
        multiplicity : ArrayLike, optional
            Presynaptic spike count (typically ``1.0`` for a single spike or ``0.0`` for
            no spike). Can be a float representing spike rate or an integer spike count.
            Default: ``1.0``.
        post : object, optional
            Postsynaptic receiver neuron. If ``None``, uses the default receiver
            specified at construction (``self.post``). Default: ``None``.
        receptor_type : ArrayLike, optional
            Receptor port to target on the postsynaptic neuron. If ``None``, uses
            ``self.receptor_type``. Default: ``None``.

        Returns
        -------
        bool
            ``True`` if an event was scheduled (i.e., ``multiplicity`` is non-zero),
            ``False`` otherwise.

        Notes
        -----
        - **Event Timing**: The spike time stamp is computed as ``current_time + dt``
          (on-grid time). Inter-spike interval ``h`` is the difference between the
          current spike stamp and ``self.t_lastspike``.
        - **State Update**: State variables ``u``, ``x``, ``y`` are updated *in place*
          and persist across calls. ``t_lastspike`` is updated to the current spike stamp.
        - **Zero Multiplicity**: If ``multiplicity`` is zero or negligible, no event
          is scheduled and state variables are *not* updated. Returns ``False``.
        - **Effective Weight**: The delivered payload is ``delta_y * weight * multiplicity``,
          where ``delta_y`` is the released resource fraction computed from current state.

        Warnings
        --------
        - If ``tau_psc`` and ``tau_rec`` are numerically close, the propagator ``P_xy``
          may suffer from floating-point cancellation. Users should avoid configurations
          where ``abs(tau_psc - tau_rec) < 1e-6``.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate as bst
           >>> with bst.environ.context(dt=0.1 * u.ms):
           ...     syn = bp.nest.tsodyks_synapse(weight=1.0, U=0.5, tau_rec=800.0 * u.ms)
           ...     syn.init_all_states()
           ...     # First spike
           ...     success = syn.send(multiplicity=1.0)
           ...     print(f"Scheduled: {success}, u={syn.u:.3f}, x={syn.x:.3f}")
           Scheduled: True, u=0.500, x=0.500
           ...     # Second spike 20 ms later (simulate time advancement)
           ...     success = syn.send(multiplicity=1.0)
           ...     print(f"Scheduled: {success}, u={syn.u:.3f}, x={syn.x:.3f}")
           Scheduled: True, u=0.750, x=0.125
        """
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST uses the spike stamp and ignores precise sub-step offsets.
        t_spike = self._current_time_ms() + dt_ms
        h = float(t_spike - self.t_lastspike)

        puu = 0.0 if self.tau_fac == 0.0 else math.exp(-h / self.tau_fac)
        pyy = math.exp(-h / self.tau_psc)
        pzz = math.expm1(-h / self.tau_rec)
        pxy = (pzz * self.tau_rec - (pyy - 1.0) * self.tau_psc) / (self.tau_psc - self.tau_rec)

        z = 1.0 - self.x - self.y

        # Keep ordering identical to NEST models/tsodyks_synapse.h::send.
        self.u *= puu
        self.x += pxy * self.y - pzz * z
        self.y *= pyy

        self.u += self.U * (1.0 - self.u)

        delta_y_tsp = self.u * self.x
        self.x -= delta_y_tsp
        self.y += delta_y_tsp

        weighted_payload = multiplicity * (delta_y_tsp * self.weight)
        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True

    def update(
        self,
        pre_spike: ArrayLike = 0.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> int:
        r"""Deliver due events and process current-step presynaptic input.

        This method performs two tasks per simulation time step:

        1. **Deliver pending events**: Dequeue and dispatch all events scheduled for
           delivery at the current simulation time step (via ``_deliver_due_events()``).
        2. **Process presynaptic input**: Collect current-step and delta inputs
           (via ``sum_current_inputs()`` and ``sum_delta_inputs()``), then schedule
           a new event if total input is non-zero (via ``send()``).

        This is the main per-step entry point for synapse dynamics when integrated
        into a simulation loop.

        Parameters
        ----------
        pre_spike : ArrayLike, optional
            Presynaptic spike count or rate for the current simulation step. Typically
            ``0.0`` (no spike) or ``1.0`` (spike). This value is accumulated with any
            other inputs registered via ``add_current_input()`` or ``add_delta_input()``.
            Default: ``0.0``.
        post : object, optional
            Postsynaptic receiver neuron. If ``None``, uses the default receiver
            specified at construction. Default: ``None``.
        receptor_type : ArrayLike, optional
            Receptor port to target. If ``None``, uses ``self.receptor_type``. Default: ``None``.

        Returns
        -------
        int
            Number of events delivered during this step (from the delivery queue).

        Notes
        -----
        - **Execution Order**: Delivery precedes scheduling. Events scheduled at step
          ``n`` are delivered at step ``n + delay_steps``.
        - **Input Accumulation**: ``pre_spike`` is summed with any inputs registered
          via the ``current_inputs`` and ``delta_inputs`` dictionaries (inherited from
          :class:`Dynamics`). The total determines whether a new event is scheduled.
        - **Typical Usage**: Call this method once per simulation time step in a network
          update loop, after presynaptic neuron spike detection.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate as bst
           >>> with bst.environ.context(dt=0.1 * u.ms):
           ...     syn = bp.nest.tsodyks_synapse(weight=1.0, delay=1.0 * u.ms, U=0.5)
           ...     syn.init_all_states()
           ...     # Simulate one step with a presynaptic spike
           ...     delivered_count = syn.update(pre_spike=1.0)
           ...     print(f"Delivered: {delivered_count}, State: u={syn.u:.3f}, x={syn.x:.3f}")
           Delivered: 0, State: u=0.500, x=0.500
           ...     # Advance simulation by delay steps (typically via outer loop)
           ...     # ... (advance time by 1.0 ms)
           ...     delivered_count = syn.update(pre_spike=0.0)
           ...     print(f"Delivered: {delivered_count}")
           Delivered: 1
        """
        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        total = self.sum_current_inputs(pre_spike)
        total = self.sum_delta_inputs(total)
        if self._is_nonzero(total):
            self.send(total, post=post, receptor_type=receptor_type)

        return delivered
