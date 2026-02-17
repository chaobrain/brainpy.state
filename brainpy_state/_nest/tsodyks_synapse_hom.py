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

from .static_synapse_hom_w import static_synapse_hom_w

__all__ = [
    'tsodyks_synapse_hom',
]


_UNSET = object()


class tsodyks_synapse_hom(static_synapse_hom_w):
    r"""NEST-compatible ``tsodyks_synapse_hom`` connection model with homogeneous short-term plasticity.

    **1. Short Description**

    Synapse type with short-term plasticity using homogeneous parameters across all connections,
    implementing the Tsodyks-Markram model with depressing and facilitating dynamics.

    **2. Mathematical Description**

    ``tsodyks_synapse_hom`` mirrors NEST ``models/tsodyks_synapse_hom.h`` and implements the
    short-term plasticity model of Tsodyks, Uziel and Markram (2000) [2]_, with dynamic
    per-connection state variables:

    - ``x``: fraction of resources in recovered state (available for release)
    - ``y``: fraction of resources in active state (released but not yet decayed)
    - ``u``: utilization (release probability upon spike arrival)
    - ``z = 1 - x - y``: fraction of resources in inactive state (released and decayed, recovering)

    For a presynaptic spike at time :math:`t_s` with inter-spike interval
    :math:`h = t_s - t_{\mathrm{last}}`, NEST updates states in this exact order:

    **Step 1: Continuous-time propagation from** :math:`t_{\mathrm{last}}` **to** :math:`t_s`

    First compute propagation factors:

    .. math::
        P_{uu} &= \begin{cases}
        0, & \tau_{\mathrm{fac}}=0 \\
        e^{-h/\tau_{\mathrm{fac}}}, & \tau_{\mathrm{fac}}>0
        \end{cases} \\
        P_{yy} &= e^{-h/\tau_{\mathrm{psc}}} \\
        P_{zz} &= e^{-h/\tau_{\mathrm{rec}}} \\
        P_{xy} &= \frac{(P_{zz}-1)\tau_{\mathrm{rec}} - (P_{yy}-1)\tau_{\mathrm{psc}}}
                       {\tau_{\mathrm{psc}}-\tau_{\mathrm{rec}}} \\
        P_{xz} &= 1 - P_{zz}

    Then update state variables:

    .. math::
        u &\leftarrow u \cdot P_{uu} \\
        x &\leftarrow x + P_{xy} y + P_{xz} z \\
        y &\leftarrow y \cdot P_{yy}

    **Step 2: Spike-triggered facilitation**

    .. math::
        u \leftarrow u + U(1-u)

    where :math:`U` is the utilization increment parameter.

    **Step 3: Resource release calculation**

    .. math::
        \Delta y = u \cdot x

    This is the fraction of resources released by the current spike.

    **Step 4: State updates for resource depletion**

    .. math::
        x &\leftarrow x - \Delta y \\
        y &\leftarrow y + \Delta y

    **Step 5: Event amplitude computation**

    .. math::
        w_{\mathrm{eff}} = \Delta y \cdot w

    where :math:`w` is the common synaptic weight.

    This implementation preserves the exact NEST ordering in :meth:`send`. Delay
    scheduling and delivery follow :class:`static_synapse_hom_w`.

    **3. Homogeneous-Property Semantics**

    In NEST, ``weight``, ``U``, ``tau_psc``, ``tau_fac`` and ``tau_rec`` are common
    synapse-model properties (stored in ``TsodyksHomCommonProperties``), shared across
    all connections using this synapse type. In contrast, ``x``, ``y`` and ``u`` are
    per-connection state variables that evolve independently for each synapse.

    This implementation mirrors that behavior by:

    - Forbidding ``set_weight(...)`` calls (same as NEST)
    - Rejecting these common properties in connect-time synapse specs via
      :meth:`check_synapse_params`
    - Allowing model-level parameter updates via :meth:`set(...)`

    **4. Event Timing Semantics**

    As in NEST, this model uses spike timestamps (grid-aligned) and ignores precise
    sub-step offsets when computing plasticity updates. All spikes are treated as
    occurring at the beginning of their respective time step.

    Parameters
    ----------
    weight : float or array-like, optional
        Common synaptic weight (dimensionless). Shared across all connections. Must be scalar.
        Default: ``1.0``.
    delay : Quantity (time) or array-like, optional
        Synaptic transmission delay. Must be positive. Scalar or broadcastable to connection count.
        Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Target receptor port ID on the postsynaptic neuron. Default: ``0``.
    tau_psc : Quantity (time) or array-like, optional
        Time constant for postsynaptic current (active resource decay). Must be ``> 0`` and scalar.
        Corresponds to NEST's ``tau_psc`` parameter. Default: ``3.0 * u.ms``.
    tau_fac : Quantity (time) or array-like, optional
        Facilitation time constant (utilization decay). Must be ``>= 0`` and scalar.
        When ``tau_fac = 0``, facilitation is disabled (:math:`P_{uu} = 0`).
        Corresponds to NEST's ``tau_fac`` parameter. Default: ``0.0 * u.ms``.
    tau_rec : Quantity (time) or array-like, optional
        Recovery time constant (inactive → recovered transition). Must be ``> 0`` and scalar.
        Corresponds to NEST's ``tau_rec`` parameter. Default: ``800.0 * u.ms``.
    U : float or array-like, optional
        Utilization increment parameter. Must be in ``[0, 1]`` and scalar.
        Determines the spike-triggered increase in release probability.
        Corresponds to NEST's ``U`` parameter. Default: ``0.5``.
    x : float or array-like, optional
        Initial fraction of recovered resources. Must satisfy ``x + y <= 1`` and be scalar.
        Default: ``1.0`` (all resources initially available).
    y : float or array-like, optional
        Initial fraction of active resources. Must satisfy ``x + y <= 1`` and be scalar.
        Default: ``0.0`` (no resources initially active).
    u : float or array-like, optional
        Initial utilization state (release probability). Must be scalar.
        NEST stores this as mutable per-connection state without enforcing bounds on assignment.
        Default: ``0.0``.
    post : object, optional
        Default postsynaptic target object. Can be overridden in :meth:`send` and :meth:`update`.
    name : str, optional
        Descriptive name for this synapse model instance.

    **Parameter Mapping (NEST → brainpy.state)**

    ================  ==========================  ==========  ===========
    NEST Parameter    brainpy.state Parameter     Type        Scope
    ================  ==========================  ==========  ===========
    weight            weight                      float       Common
    delay             delay                       Quantity    Common
    receptor_type     receptor_type               int         Common
    tau_psc           tau_psc                     Quantity    Common
    tau_fac           tau_fac                     Quantity    Common
    tau_rec           tau_rec                     Quantity    Common
    U                 U                           float       Common
    x                 x                           float       State
    y                 y                           float       State
    u                 u                           float       State
    ================  ==========================  ==========  ===========

    Notes
    -----
    - **Event type**: This model transmits spike-like events only, not continuous currents.
    - **State initialization**: ``init_state()`` resets queue state and restores ``x``, ``y``, ``u``
      to their constructor-specified initial values.
    - **Last-spike timestamp**: ``t_lastspike`` starts at ``0.0`` ms (as in NEST). If ``x != 1``
      initially, the first-spike dynamics depend on this initial timestamp.
    - **Homogeneous constraints**: Common properties (``weight``, ``U``, ``tau_psc``, ``tau_fac``,
      ``tau_rec``) cannot be set per-connection via connect-time synapse specs. Use ``set()`` to
      modify them at the model level.
    - **Validation**: Constructor validates ``tau_psc > 0``, ``tau_fac >= 0``, ``tau_rec > 0``,
      ``U ∈ [0,1]``, and ``x + y <= 1``. ``set()`` re-validates on updates.
    - **Numerical stability**: When ``tau_psc ≈ tau_rec``, the :math:`P_{xy}` calculation may
      become numerically unstable. NEST uses the same formula without special handling.

    See Also
    --------
    tsodyks_synapse : Heterogeneous variant with per-connection parameters
    static_synapse_hom_w : Parent class for homogeneous static connections

    References
    ----------
    .. [1] NEST source: ``models/tsodyks_synapse_hom.h`` and
           ``models/tsodyks_synapse_hom.cpp``.
    .. [2] Tsodyks M, Uziel A, Markram H (2000). Synchrony generation in
           recurrent networks with frequency-dependent synapses.
           Journal of Neuroscience, 20:RC50.
           https://www.jneurosci.org/content/20/1/RC50

    Examples
    --------
    Create a depressing synapse (default parameters):

    .. code-block:: python

        >>> import brainpy.state as bs
        >>> import brainunit as u
        >>> syn = bs.tsodyks_synapse_hom(
        ...     weight=2.0,
        ...     delay=1.5 * u.ms,
        ...     U=0.5,
        ...     tau_rec=800.0 * u.ms,
        ...     tau_fac=0.0 * u.ms,  # no facilitation
        ... )

    Create a facilitating synapse:

    .. code-block:: python

        >>> syn_fac = bs.tsodyks_synapse_hom(
        ...     weight=1.5,
        ...     U=0.15,
        ...     tau_rec=200.0 * u.ms,
        ...     tau_fac=750.0 * u.ms,  # strong facilitation
        ...     tau_psc=5.0 * u.ms,
        ... )

    Update common properties (e.g., increase facilitation):

    .. code-block:: python

        >>> syn_fac.set(tau_fac=1000.0 * u.ms, U=0.2)

    Initialize state for simulation:

    .. code-block:: python

        >>> syn.init_state()
        >>> print(syn.x, syn.y, syn.u)  # (1.0, 0.0, 0.0)

    Retrieve current parameters and state:

    .. code-block:: python

        >>> params = syn.get()
        >>> print(params['synapse_model'])  # 'tsodyks_synapse_hom'
        >>> print(params['U'], params['tau_rec'])  # (0.5, 800.0)
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
        self.U = self._to_scalar_unit_interval_no_quotes(U, name='U')

        x0 = self._to_scalar_float(x, name='x')
        y0 = self._to_scalar_float(y, name='y')
        u0 = self._to_scalar_float(u, name='u')

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
    def _to_scalar_unit_interval_no_quotes(value: ArrayLike, *, name: str) -> float:
        v = tsodyks_synapse_hom._to_scalar_float(value, name=name)
        if v < 0.0 or v > 1.0:
            raise ValueError(f'{name} must be in [0,1].')
        return float(v)

    @staticmethod
    def _validate_tau_psc(value: float):
        if value <= 0.0:
            raise ValueError('tau_psc must be > 0.')

    @staticmethod
    def _validate_tau_fac(value: float):
        if value < 0.0:
            raise ValueError('tau_fac must be >= 0.')

    @staticmethod
    def _validate_tau_rec(value: float):
        if value <= 0.0:
            raise ValueError('tau_rec must be > 0.')

    @staticmethod
    def _validate_xy_sum(x: float, y: float):
        if x + y > 1.0:
            raise ValueError('x + y must be <= 1.0.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Reset queue state and restore initial resource/utilization values.

        Resets the event delivery queue (inherited from parent) and restores the
        short-term plasticity state variables (``x``, ``y``, ``u``) to their
        constructor-specified initial values (``_x0``, ``_y0``, ``_u0``). Also resets
        the last-spike timestamp to ``0.0`` ms.

        Parameters
        ----------
        batch_size : int, optional
            Ignored for scalar synapse models (retained for API compatibility).
        **kwargs : dict, optional
            Ignored (retained for API compatibility).

        Notes
        -----
        - This method should be called before starting a new simulation to ensure
          consistent initial conditions.
        - After calling ``init_state()``, the synapse is in a "naive" state with
          ``t_lastspike = 0.0``, meaning the first spike will have inter-spike
          interval :math:`h = t_{\mathrm{spike}} - 0.0`.
        """
        del batch_size, kwargs
        super().init_state()
        self.x = float(self._x0)
        self.y = float(self._y0)
        self.u = float(self._u0)
        self.t_lastspike = 0.0

    def get(self) -> dict:
        r"""Return current public parameters and mutable state.

        Retrieves all public parameters (``weight``, ``delay``, ``receptor_type``, ``tau_psc``,
        ``tau_fac``, ``tau_rec``, ``U``) and current state variables (``x``, ``y``, ``u``)
        as a dictionary. Inherited parameters are retrieved via ``super().get()``.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``'weight'`` (float): Common synaptic weight.
            - ``'delay'`` (float): Synaptic delay in ms.
            - ``'receptor_type'`` (int): Receiver port ID.
            - ``'tau_psc'`` (float): PSC time constant in ms.
            - ``'tau_fac'`` (float): Facilitation time constant in ms.
            - ``'tau_rec'`` (float): Recovery time constant in ms.
            - ``'U'`` (float): Utilization increment parameter.
            - ``'x'`` (float): Current recovered resource fraction.
            - ``'y'`` (float): Current active resource fraction.
            - ``'u'`` (float): Current utilization state.
            - ``'synapse_model'`` (str): Always ``'tsodyks_synapse_hom'``.

        Notes
        -----
        - All time constants are returned in milliseconds (converted from internal storage).
        - State variables (``x``, ``y``, ``u``) reflect the current simulation state, not
          initial values.
        - This method is compatible with NEST's ``GetStatus()`` semantics.
        """
        params = super().get()
        params['tau_psc'] = float(self.tau_psc)
        params['tau_fac'] = float(self.tau_fac)
        params['tau_rec'] = float(self.tau_rec)
        params['U'] = float(self.U)
        params['x'] = float(self.x)
        params['y'] = float(self.y)
        params['u'] = float(self.u)
        params['synapse_model'] = 'tsodyks_synapse_hom'
        return params

    def check_synapse_params(self, syn_spec: Mapping[str, object] | None):
        r"""Reject common-property assignments in connect-time synapse specs.

        Validates that connect-time synapse specifications do not attempt to override
        common (homogeneous) properties that must be shared across all connections.
        This enforces NEST's homogeneous synapse semantics where ``weight``, ``U``,
        ``tau_psc``, ``tau_fac``, and ``tau_rec`` are model-level properties.

        Parameters
        ----------
        syn_spec : dict or None
            Synapse specification dictionary passed during connection creation.
            If ``None``, validation is skipped (no parameters to check).

        Raises
        ------
        ValueError
            If ``syn_spec`` contains any of the disallowed keys: ``'weight'``, ``'U'``,
            ``'tau_psc'``, ``'tau_rec'``, or ``'tau_fac'``. These must be set at the
            model level using :meth:`set` (or NEST's ``CopyModel``/``SetDefaults``).

        Notes
        -----
        - This method is called internally during connection setup to prevent misuse.
        - In NEST, ``weight`` is a common property for ``tsodyks_synapse_hom`` but can
          vary per-connection in ``tsodyks_synapse``. This model follows the ``_hom``
          variant's restrictions.
        - To modify common properties, use ``model.set(weight=..., U=..., ...)`` before
          or during simulation, not during connection creation.
        """
        if syn_spec is None:
            return
        disallowed = ('weight', 'U', 'tau_psc', 'tau_rec', 'tau_fac')
        for key in disallowed:
            if key in syn_spec:
                raise ValueError(
                    f'{key} cannot be specified in connect-time synapse parameters '
                    'for tsodyks_synapse_hom; set common properties on the model '
                    'itself (for example via CopyModel()/SetDefaults()).'
                )

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
        r"""Set public parameters following NEST-style validation semantics.

        Updates model parameters and/or state variables with comprehensive validation.
        All parameters are optional; only specified values are updated. This method
        mirrors NEST's ``SetStatus()`` for homogeneous synapse models.

        Parameters
        ----------
        weight : float or array-like, optional
            New common synaptic weight (dimensionless). Must be scalar.
        delay : Quantity (time) or array-like, optional
            New synaptic transmission delay. Must be positive.
        receptor_type : int, optional
            New target receptor port ID.
        tau_psc : Quantity (time) or array-like, optional
            New PSC time constant. Must be ``> 0`` and scalar.
        tau_fac : Quantity (time) or array-like, optional
            New facilitation time constant. Must be ``>= 0`` and scalar.
        tau_rec : Quantity (time) or array-like, optional
            New recovery time constant. Must be ``> 0`` and scalar.
        U : float or array-like, optional
            New utilization increment parameter. Must be in ``[0, 1]`` and scalar.
        x : float or array-like, optional
            New recovered resource fraction. Must satisfy ``x + y <= 1`` and be scalar.
        y : float or array-like, optional
            New active resource fraction. Must satisfy ``x + y <= 1`` and be scalar.
        u : float or array-like, optional
            New utilization state. Must be scalar.
        post : object, optional
            New default postsynaptic target object.

        Raises
        ------
        ValueError
            If any parameter fails validation:
            - ``tau_psc <= 0``
            - ``tau_fac < 0``
            - ``tau_rec <= 0``
            - ``U ∉ [0, 1]``
            - ``x + y > 1``
            - Non-scalar values for common properties

        Notes
        -----
        - All validation is performed **before** any state is modified (atomic update).
        - When updating ``x`` or ``y``, the sum constraint is checked with the new values.
        - Updating ``x``, ``y``, or ``u`` also updates the internal initial-value cache
          (``_x0``, ``_y0``, ``_u0``), so subsequent ``init_state()`` calls will use
          the new values.
        - Time constants are validated after conversion to milliseconds.
        - This method does **not** reset ``t_lastspike`` or clear the event queue.

        Examples
        --------
        Update facilitation parameters:

        .. code-block:: python

            >>> syn.set(tau_fac=500.0 * u.ms, U=0.3)

        Update state variables (e.g., after external perturbation):

        .. code-block:: python

            >>> syn.set(x=0.8, y=0.1, u=0.2)

        Update multiple parameters atomically:

        .. code-block:: python

            >>> syn.set(
            ...     weight=2.5,
            ...     delay=2.0 * u.ms,
            ...     tau_rec=600.0 * u.ms,
            ...     U=0.6,
            ... )
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
        new_U = (
            self.U
            if U is _UNSET
            else self._to_scalar_unit_interval_no_quotes(U, name='U')
        )
        new_x = self.x if x is _UNSET else self._to_scalar_float(x, name='x')
        new_y = self.y if y is _UNSET else self._to_scalar_float(y, name='y')
        new_u = self.u if u is _UNSET else self._to_scalar_float(u, name='u')

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
        r"""Schedule one outgoing event with NEST ``tsodyks_synapse_hom`` dynamics.

        Computes short-term plasticity state updates following the Tsodyks-Markram model,
        calculates the effective synaptic weight, and schedules a delayed spike event
        for delivery to the postsynaptic target.

        **Algorithm:**

        1. Compute inter-spike interval :math:`h = t_{\mathrm{spike}} - t_{\mathrm{lastspike}}`
        2. Apply continuous-time propagation (exponential decay) to ``u``, ``x``, ``y``
        3. Apply spike-triggered facilitation: :math:`u \leftarrow u + U(1-u)`
        4. Compute released resources: :math:`\Delta y = u \cdot x`
        5. Update ``x`` and ``y`` for depletion/release
        6. Compute effective weight: :math:`w_{\mathrm{eff}} = \Delta y \cdot w \cdot \text{multiplicity}`
        7. Schedule delayed event in delivery queue
        8. Update ``t_lastspike`` to current spike time

        Parameters
        ----------
        multiplicity : float or array-like, optional
            Spike count or scaling factor. If zero, no event is scheduled and ``False``
            is returned immediately. Default: ``1.0``.
        post : object, optional
            Target postsynaptic object. If ``None``, uses the default receiver stored
            in ``self.post``.
        receptor_type : int or array-like, optional
            Target receptor port ID. If ``None``, uses ``self.receptor_type``.

        Returns
        -------
        bool
            ``True`` if an event was successfully scheduled, ``False`` if ``multiplicity``
            was zero (no event sent).

        Notes
        -----
        - **State updates**: This method mutates ``self.x``, ``self.y``, ``self.u``, and
          ``self.t_lastspike`` in-place.
        - **Ordering**: The update order exactly matches NEST's ``tsodyks_synapse_hom.h::send()``.
        - **Delay handling**: The event is scheduled for delivery at
          :math:`t_{\mathrm{current}} + \text{dt} + \text{delay}`, where ``dt`` is the
          simulation timestep.
        - **Queue structure**: Events are stored as ``(receiver, payload, port, 'spike')``
          tuples in the internal delay queue.
        - **Numerical stability**: When ``tau_psc ≈ tau_rec``, the :math:`P_{xy}` calculation
          may lose precision (same limitation as NEST).

        See Also
        --------
        update : Delivers queued events and schedules new ones from presynaptic input
        """
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        t_spike = self._current_time_ms() + dt_ms
        h = float(t_spike - self.t_lastspike)

        puu = 0.0 if self.tau_fac == 0.0 else math.exp(-h / self.tau_fac)
        pyy = math.exp(-h / self.tau_psc)
        pzz = math.exp(-h / self.tau_rec)
        pxy = ((pzz - 1.0) * self.tau_rec - (pyy - 1.0) * self.tau_psc) / (
            self.tau_psc - self.tau_rec
        )
        pxz = 1.0 - pzz

        z = 1.0 - self.x - self.y

        # Keep ordering identical to NEST models/tsodyks_synapse_hom.h::send.
        self.u *= puu
        self.x += pxy * self.y + pxz * z
        self.y *= pyy

        self.u += self.U * (1.0 - self.u)

        delta_y_tsp = self.u * self.x
        self.x -= delta_y_tsp
        self.y += delta_y_tsp

        # NEST code sets receiver before weight and delay assignment.
        receiver = self._resolve_receiver(post)
        weighted_payload = multiplicity * (delta_y_tsp * self.weight)
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
        r"""Deliver due events, then schedule current-step presynaptic input.

        This is the main simulation-step method, combining event delivery and spike
        transmission. It first delivers all queued events that are due for the current
        time step, then processes new presynaptic input (if any) by calling :meth:`send`.

        **Workflow:**

        1. Refresh delay computation if needed (inherited from parent)
        2. Deliver all events scheduled for the current simulation step
        3. Sum presynaptic inputs from ``pre_spike``, ``current_inputs``, and ``delta_inputs``
        4. If total input is non-zero, call :meth:`send` to schedule a new event with
           short-term plasticity

        Parameters
        ----------
        pre_spike : float or array-like, optional
            Presynaptic spike input for the current time step (dimensionless spike count
            or activation). Default: ``0.0`` (no input).
        post : object, optional
            Target postsynaptic object for new events. If ``None``, uses ``self.post``.
        receptor_type : int or array-like, optional
            Target receptor port ID for new events. If ``None``, uses ``self.receptor_type``.

        Returns
        -------
        int
            Number of events delivered to postsynaptic targets during this time step.

        Notes
        -----
        - **Event delivery**: Delivered events are removed from the internal queue after
          processing.
        - **Input summation**: The method sums:
          1. Explicit ``pre_spike`` argument
          2. Accumulated ``current_inputs`` (from projections)
          3. Accumulated ``delta_inputs`` (from delta-function sources)
        - **Plasticity application**: Short-term plasticity state updates occur only when
          ``send()`` is called (i.e., when total input is non-zero).
        - **Typical usage**: Called once per simulation time step for each synapse model
          instance.

        Examples
        --------
        Simulation loop with explicit presynaptic spike:

        .. code-block:: python

            >>> for t in range(1000):
            ...     spike = 1.0 if t % 100 == 0 else 0.0  # spike every 100 steps
            ...     delivered = syn.update(pre_spike=spike, post=target_neuron)

        Simulation with projection-driven input (no explicit spike):

        .. code-block:: python

            >>> # Projections accumulate input in syn.current_inputs
            >>> for t in range(1000):
            ...     delivered = syn.update(post=target_neuron)
        """
        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        total = self.sum_current_inputs(pre_spike)
        total = self.sum_delta_inputs(total)
        if self._is_nonzero(total):
            self.send(total, post=post, receptor_type=receptor_type)

        return delivered
