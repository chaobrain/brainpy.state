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
    'tsodyks2_synapse',
]


_UNSET = object()


class tsodyks2_synapse(static_synapse):
    r"""NEST-compatible ``tsodyks2_synapse`` connection model.

    Short description
    -----------------

    Synapse type with short-term depression and facilitation.

    Description
    -----------

    ``tsodyks2_synapse`` mirrors NEST ``models/tsodyks2_synapse.h``.
    The model stores per-connection state variables:

    - ``x``: current scaling factor of synaptic efficacy,
    - ``u``: current release probability,
    - ``t_lastspike``: last presynaptic spike stamp.

    Together with fixed model parameters ``U``, ``tau_rec`` and ``tau_fac``,
    incoming spikes are processed in the same order as NEST:

    **1. State propagation**

    For ``h = t_spike - t_lastspike``, if this is not the first spike
    (i.e. ``t_lastspike >= 0``), propagate state to the current spike:

    .. math::
       x \leftarrow 1 + (x - xu - 1)e^{-h/\tau_{rec}}

       u \leftarrow U + u(1-U)e^{-h/\tau_{fac}}

    with the NEST special case ``tau_fac == 0``:

    .. math::
       e^{-h/\tau_{fac}} \equiv 0

    **2. Effective weight computation**

    Compute effective synaptic weight for this spike:

    .. math::
       w_{eff} = x u w

    **3. Event scheduling**

    Schedule event delivery with inherited static-synapse delay semantics.

    **4. State update**

    Set ``t_lastspike = t_spike``.

    This model scales the baseline synaptic weight only and is suitable for
    current- or conductance-based postsynaptic dynamics.

    Event timing semantics
    ----------------------

    As in NEST, updates use spike stamps and ignore precise sub-step offsets.
    In this backend, each presynaptic event at simulation step ``t`` is
    evaluated at on-grid stamp ``t + dt``.

    Mathematical formulation
    ------------------------

    The tsodyks2_synapse model implements a simplified version of the
    Tsodyks-Markram short-term plasticity mechanism that tracks only two
    key dynamic variables rather than the full resource state model.

    **State variables:**

    - :math:`x(t)`: scaling factor representing available resources
    - :math:`u(t)`: utilization (release probability)
    - :math:`t_{\mathrm{last}}`: timestamp of last presynaptic spike

    **Parameters:**

    - :math:`U`: baseline utilization increment (facilitation magnitude)
    - :math:`\tau_{rec}`: recovery time constant (depression timescale)
    - :math:`\tau_{fac}`: facilitation time constant
    - :math:`w`: baseline synaptic weight

    **Dynamics:**

    Upon arrival of a presynaptic spike at time :math:`t_s`, with inter-spike
    interval :math:`h = t_s - t_{\mathrm{last}}`:

    1. If :math:`t_{\mathrm{last}} \geq 0` (not the first spike), propagate
       state variables:

       .. math::
          x(t_s) = 1 + [x(t_{\mathrm{last}}) - x(t_{\mathrm{last}})u(t_{\mathrm{last}}) - 1]
                   \exp\left(-\frac{h}{\tau_{rec}}\right)

          u(t_s) = U + u(t_{\mathrm{last}})(1-U)\exp\left(-\frac{h}{\tau_{fac}}\right)

       with the special case when :math:`\tau_{fac} = 0`:

       .. math::
          \exp\left(-\frac{h}{\tau_{fac}}\right) \equiv 0

    2. Compute effective synaptic strength:

       .. math::
          w_{\mathrm{eff}} = x(t_s) \cdot u(t_s) \cdot w

    3. Update last spike time:

       .. math::
          t_{\mathrm{last}} \leftarrow t_s

    **Biological interpretation:**

    - **Depression** (:math:`x` dynamics): After each spike, :math:`x`
      decreases by factor :math:`xu`, representing depletion of available
      resources. It recovers exponentially with time constant :math:`\tau_{rec}`.

    - **Facilitation** (:math:`u` dynamics): The utilization :math:`u`
      increases toward :math:`U` after each spike, representing
      calcium-dependent facilitation of release probability. It decays
      exponentially with time constant :math:`\tau_{fac}`.

    - **Combined effect**: The effective synaptic weight :math:`w_{\mathrm{eff}}`
      is the product of available resources (:math:`x`), release probability
      (:math:`u`), and baseline weight (:math:`w`).

    **Parameter regimes:**

    - **Depression-dominated**: :math:`\tau_{rec} \gg \tau_{fac}`, :math:`U`
      moderate. Resources deplete faster than facilitation builds up.

    - **Facilitation-dominated**: :math:`\tau_{fac} \gg \tau_{rec}`, :math:`U`
      small. Facilitation accumulates across multiple spikes.

    - **Pure depression**: :math:`\tau_{fac} = 0`, fixed :math:`u = U`.
      No facilitation dynamics.

    Computational considerations
    ----------------------------

    - **Event-driven updates**: State propagation occurs only at spike times,
      making the model computationally efficient for sparse spike trains.

    - **Exponential approximation**: For very small :math:`h`, numerical
      precision may be limited. NEST uses the convention that
      :math:`\tau_{fac} = 0` exactly eliminates facilitation rather than
      using a threshold.

    - **First spike handling**: The condition :math:`t_{\mathrm{last}} < 0`
      distinguishes the first spike, which uses initial :math:`x` and :math:`u`
      values directly without propagation.

    - **Multiplicative weight scaling**: Unlike ``tsodyks_synapse``, this model
      does not track postsynaptic current dynamics separately. The effective
      weight :math:`w_{\mathrm{eff}}` directly modulates the delivered event.

    Comparison with tsodyks_synapse
    --------------------------------

    The tsodyks2_synapse model differs from tsodyks_synapse in several ways:

    **State representation:**

    - tsodyks2: Two variables (:math:`x`, :math:`u`)
    - tsodyks: Three variables (:math:`x`, :math:`y`, :math:`u`) plus
      postsynaptic current :math:`\tau_{psc}`

    **Update equations:**

    - tsodyks2: Simplified model where :math:`x` represents effective resources
      after accounting for utilization
    - tsodyks: Full resource model with explicit recovered (:math:`x`), active
      (:math:`y`), and inactive (:math:`z = 1-x-y`) resource fractions

    **Postsynaptic dynamics:**

    - tsodyks2: No built-in postsynaptic filtering; :math:`w_{\mathrm{eff}}`
      directly scales the event
    - tsodyks: Includes :math:`\tau_{psc}` for postsynaptic current dynamics

    **Use cases:**

    - tsodyks2: Simpler, faster, suitable when postsynaptic filtering is
      handled separately (e.g., by neuron model)
    - tsodyks: More detailed, includes postsynaptic current kinetics in the
      synapse model

    Parameters
    ----------
    weight : float or array-like, optional
        Baseline synaptic weight :math:`w`. Dimensionless scalar or array.
        Default: ``1.0``.
    delay : float or Quantity, optional
        Synaptic transmission delay. Must be a positive time value in
        milliseconds. Accepts ``brainunit.Quantity`` with time units or
        float (interpreted as ms). Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor identifier for multi-receptor postsynaptic
        neurons. Non-negative integer. Default: ``0``.
    U : float or array-like, optional
        Utilization increment parameter :math:`U`. Must be in ``[0, 1]``.
        Controls the magnitude of facilitation per spike. Dimensionless scalar.
        Default: ``0.5``.
    u : float or array-like or UNSET, optional
        Initial release probability :math:`u(0)`. Must be in ``[0, 1]``.
        If not provided, defaults to ``U``. Dimensionless scalar.
        Default: ``U``.
    x : float or array-like, optional
        Initial scaling factor of synaptic efficacy :math:`x(0)`. Represents
        initial available resources. Dimensionless scalar. Typical range
        ``[0, 1]``, though values ``> 1`` are allowed. Default: ``1.0``.
    tau_rec : float or Quantity, optional
        Recovery (depression) time constant :math:`\tau_{rec}`. Must be
        strictly positive (``> 0``). Controls the timescale of resource
        replenishment. Accepts ``brainunit.Quantity`` with time units or
        float (interpreted as ms). Default: ``800.0 * u.ms``.
    tau_fac : float or Quantity, optional
        Facilitation time constant :math:`\tau_{fac}`. Must be non-negative
        (``>= 0``). When zero, facilitation is disabled and :math:`u` remains
        constant at :math:`U`. Accepts ``brainunit.Quantity`` with time units
        or float (interpreted as ms). Default: ``0.0 * u.ms``.
    post : object, optional
        Default postsynaptic receiver object. If provided, this receiver will
        be used by default in :meth:`send` and :meth:`update` calls when no
        explicit receiver is specified. Default: ``None``.
    name : str, optional
        Optional name identifier for this connection object. Used for debugging
        and introspection. Default: ``None``.

    Parameter Mapping
    -----------------

    The following table maps NEST parameter names to brainpy.state
    equivalents and describes their interpretation:

    +-----------------+------------------+-------------+---------------------+
    | NEST Parameter  | brainpy.state    | Type        | Description         |
    +=================+==================+=============+=====================+
    | ``weight``      | ``weight``       | float       | Baseline synaptic   |
    |                 |                  |             | weight :math:`w`    |
    +-----------------+------------------+-------------+---------------------+
    | ``delay``       | ``delay``        | Quantity    | Transmission delay  |
    |                 |                  | (ms)        | in milliseconds     |
    +-----------------+------------------+-------------+---------------------+
    | ``receptor_     | ``receptor_      | int         | Target receptor     |
    | type``          | type``           |             | port identifier     |
    +-----------------+------------------+-------------+---------------------+
    | ``U``           | ``U``            | float       | Utilization         |
    |                 |                  | [0,1]       | increment           |
    +-----------------+------------------+-------------+---------------------+
    | ``u``           | ``u``            | float       | Initial/current     |
    |                 |                  | [0,1]       | release probability |
    +-----------------+------------------+-------------+---------------------+
    | ``x``           | ``x``            | float       | Initial/current     |
    |                 |                  |             | efficacy scaling    |
    +-----------------+------------------+-------------+---------------------+
    | ``tau_rec``     | ``tau_rec``      | Quantity    | Recovery time       |
    |                 |                  | (ms)        | constant            |
    +-----------------+------------------+-------------+---------------------+
    | ``tau_fac``     | ``tau_fac``      | Quantity    | Facilitation time   |
    |                 |                  | (ms)        | constant            |
    +-----------------+------------------+-------------+---------------------+

    Notes
    -----
    - This model transmits spike-like events only. The ``event_type`` is
      fixed to ``'spike'``.

    - The state variables ``x``, ``u``, and ``t_lastspike`` are mutable
      connection states. Current values can be retrieved via :meth:`get`
      and modified via :meth:`set`.

    - When ``tau_fac == 0`` exactly, facilitation is disabled and
      :math:`u` remains constant. This matches NEST's behavior and avoids
      numerical issues with :math:`\exp(-h/0)`.

    - Calling :meth:`init_state` resets the event queue and restores
      ``x``, ``u``, and ``t_lastspike`` to their configured initial values.

    - The model uses event-driven updates: state propagation occurs only
      at spike times, not at every simulation timestep.

    Raises
    ------
    ValueError
        If ``U`` or ``u`` is not in the range ``[0, 1]``.
    ValueError
        If ``tau_rec`` is not strictly positive (``<= 0``).
    ValueError
        If ``tau_fac`` is negative (``< 0``).
    ValueError
        If any scalar parameter has size ``!= 1`` when converted to array.

    See Also
    --------
    tsodyks_synapse : Full resource-based STP model with postsynaptic current.
    static_synapse : Base class providing delay and weight handling.
    tsodyks_synapse_hom : Homogeneous-weight variant of tsodyks_synapse.

    References
    ----------
    .. [1] NEST source: ``models/tsodyks2_synapse.h`` and
           ``models/tsodyks2_synapse.cpp``.
    .. [2] Tsodyks MV, Markram H (1997). The neural code between neocortical
           pyramidal neurons depends on neurotransmitter release probability.
           PNAS, 94(2):719-723. DOI: 10.1073/pnas.94.2.719
    .. [3] Fuhrmann G, Segev I, Markram H, Tsodyks MV (2002).
           Coding of temporal information by activity-dependent synapses.
           Journal of Neurophysiology, 87(1):140-148. DOI: 10.1152/jn.00258.2001
    .. [4] Maass W, Markram H (2002). Synapses as dynamic memory buffers.
           Neural Networks, 15(2):155-161. DOI: 10.1016/S0893-6080(01)00144-7

    Examples
    --------
    Create a depression-dominated synapse (fast recovery, no facilitation):

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import brainunit as u
       >>> syn = bst.tsodyks2_synapse(
       ...     weight=1.0,
       ...     delay=1.5 * u.ms,
       ...     U=0.5,
       ...     tau_rec=100.0 * u.ms,
       ...     tau_fac=0.0 * u.ms
       ... )
       >>> syn.init_state()

    Create a facilitation-dominated synapse (slow recovery, long facilitation):

    .. code-block:: python

       >>> syn_fac = bst.tsodyks2_synapse(
       ...     weight=0.5,
       ...     U=0.15,
       ...     tau_rec=800.0 * u.ms,
       ...     tau_fac=1000.0 * u.ms
       ... )

    Inspect current synapse state:

    .. code-block:: python

       >>> state = syn.get()
       >>> print(f"x={state['x']}, u={state['u']}")
       x=1.0, u=0.5

    Simulate a spike train and observe depression:

    .. code-block:: python

       >>> import brainstate as bst
       >>> import jax.numpy as jnp
       >>> # Create postsynaptic target
       >>> class SimpleTarget(bst.nn.Dynamics):
       ...     def __init__(self):
       ...         super().__init__()
       ...         self.input = 0.0
       ...     def update(self, inp=0.0):
       ...         self.input = inp
       >>> target = SimpleTarget()
       >>> syn = bst.tsodyks2_synapse(
       ...     weight=1.0,
       ...     delay=0.1 * u.ms,
       ...     U=0.5,
       ...     tau_rec=200.0 * u.ms,
       ...     tau_fac=0.0 * u.ms,
       ...     post=target
       ... )
       >>> syn.init_state()
       >>> # Simulate high-frequency spike train
       >>> with bst.environ.context(dt=0.1 * u.ms):
       ...     for i in range(10):
       ...         delivered = syn.update(pre_spike=1.0)
       ...         state = syn.get()
       ...         print(f"Step {i}: x={state['x']:.3f}, w_eff={state['x']*state['u']:.3f}")
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        U: ArrayLike = 0.5,
        u: ArrayLike | object = _UNSET,
        x: ArrayLike = 1.0,
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
        u0 = self.U if u is _UNSET else self._to_scalar_unit_interval(u, name='u')

        self.x = self._to_scalar_float(x, name='x')
        self.u = float(u0)
        self.tau_rec = self._to_scalar_time_ms(tau_rec, name='tau_rec')
        self.tau_fac = self._to_scalar_time_ms(tau_fac, name='tau_fac')

        self._validate_tau_rec(self.tau_rec)
        self._validate_tau_fac(self.tau_fac)

        self._x0 = float(self.x)
        self._u0 = float(self.u)

        self.t_lastspike = -1.0

    @staticmethod
    def _to_scalar_float(value: ArrayLike, *, name: str) -> float:
        arr = np.asarray(u.math.asarray(value, dtype=jnp.float64), dtype=np.float64)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr.reshape(()))

    @staticmethod
    def _to_scalar_unit_interval(value: ArrayLike, *, name: str) -> float:
        v = tsodyks2_synapse._to_scalar_float(value, name=name)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"'{name}' must be in [0,1].")
        return float(v)

    @staticmethod
    def _validate_tau_rec(value: float):
        if value <= 0.0:
            raise ValueError("'tau_rec' must be > 0.")

    @staticmethod
    def _validate_tau_fac(value: float):
        if value < 0.0:
            raise ValueError("'tau_fac' must be >= 0.")

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize or reset synapse state variables.

        Resets the event delivery queue and restores dynamic state variables
        (``x``, ``u``, ``t_lastspike``) to their configured initial values.
        This method should be called before starting a simulation or when
        reinitializing the model.

        Parameters
        ----------
        batch_size : int, optional
            Ignored. Included for API compatibility with batched models.
        **kwargs : dict, optional
            Additional keyword arguments. Ignored.

        Returns
        -------
        None

        Notes
        -----
        - The initial values of ``x`` and ``u`` are determined by the values
          provided during object construction (or last :meth:`set` call).
        - ``t_lastspike`` is always reset to ``-1.0`` to indicate no prior
          spike has occurred.
        - The inherited event queue from :class:`static_synapse` is also
          cleared.
        """
        del batch_size, kwargs
        super().init_state()
        self.x = float(self._x0)
        self.u = float(self._u0)
        self.t_lastspike = -1.0

    def get(self) -> dict:
        """Return current public parameters and mutable state.

        Retrieves a dictionary containing all NEST-compatible parameters and
        current dynamic state variables. This is useful for introspection,
        checkpointing, and parameter queries.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``'weight'`` : float
              Baseline synaptic weight.
            - ``'delay'`` : float
              Synaptic delay in milliseconds.
            - ``'receptor_type'`` : int
              Target receptor port identifier.
            - ``'U'`` : float
              Utilization increment parameter (fixed).
            - ``'u'`` : float
              Current release probability (mutable state).
            - ``'x'`` : float
              Current efficacy scaling factor (mutable state).
            - ``'tau_rec'`` : float
              Recovery time constant in milliseconds.
            - ``'tau_fac'`` : float
              Facilitation time constant in milliseconds.
            - ``'synapse_model'`` : str
              Model identifier string ``'tsodyks2_synapse'``.

        Notes
        -----
        - The returned ``u`` and ``x`` values reflect the current synapse
          state, which changes after each spike.
        - All time constants are returned as float values in milliseconds,
          with units stripped.
        - The ``synapse_model`` key is included for NEST compatibility and
          model identification.

        Examples
        --------
        .. code-block:: python

           >>> import brainpy.state as bst
           >>> import brainunit as u
           >>> syn = bst.tsodyks2_synapse(weight=2.0, U=0.4, tau_rec=500*u.ms)
           >>> syn.init_state()
           >>> params = syn.get()
           >>> print(params['U'], params['x'])
           0.4 1.0
        """
        params = super().get()
        params['U'] = float(self.U)
        params['u'] = float(self.u)
        params['x'] = float(self.x)
        params['tau_rec'] = float(self.tau_rec)
        params['tau_fac'] = float(self.tau_fac)
        params['synapse_model'] = 'tsodyks2_synapse'
        return params

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        U: ArrayLike | object = _UNSET,
        u: ArrayLike | object = _UNSET,
        x: ArrayLike | object = _UNSET,
        tau_rec: ArrayLike | object = _UNSET,
        tau_fac: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        """Set NEST-style public parameters and mutable state.

        Updates one or more synapse parameters or state variables. All
        arguments are keyword-only and optional. Only provided parameters
        are modified; others retain their current values. New values are
        validated before assignment.

        Parameters
        ----------
        weight : float or array-like, optional
            New baseline synaptic weight. If not provided, unchanged.
        delay : float or Quantity, optional
            New synaptic delay in milliseconds. Must be positive. If not
            provided, unchanged.
        receptor_type : int, optional
            New target receptor port identifier. If not provided, unchanged.
        U : float or array-like, optional
            New utilization increment parameter. Must be in ``[0, 1]``.
            If not provided, unchanged.
        u : float or array-like, optional
            New current release probability. Must be in ``[0, 1]``.
            If not provided, unchanged.
        x : float or array-like, optional
            New current efficacy scaling factor. If not provided, unchanged.
        tau_rec : float or Quantity, optional
            New recovery time constant in milliseconds. Must be ``> 0``.
            If not provided, unchanged.
        tau_fac : float or Quantity, optional
            New facilitation time constant in milliseconds. Must be ``>= 0``.
            If not provided, unchanged.
        post : object, optional
            New default postsynaptic receiver. If not provided, unchanged.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``U`` or ``u`` is not in ``[0, 1]``.
        ValueError
            If ``tau_rec`` is not strictly positive.
        ValueError
            If ``tau_fac`` is negative.
        ValueError
            If any scalar parameter has incorrect shape.

        Notes
        -----
        - Setting ``u`` or ``x`` modifies the current synapse state, affecting
          subsequent spike processing.
        - The initial values (``_u0``, ``_x0``) are updated to match new
          ``u`` and ``x`` values, so :meth:`init_state` will restore these
          new values.
        - All validation is performed before any state modification, ensuring
          atomicity (either all changes succeed or none do).

        Examples
        --------
        .. code-block:: python

           >>> import brainpy.state as bst
           >>> import brainunit as u
           >>> syn = bst.tsodyks2_synapse(weight=1.0, U=0.5)
           >>> syn.init_state()
           >>> # Modify parameters
           >>> syn.set(U=0.3, tau_rec=300.0 * u.ms)
           >>> # Modify current state
           >>> syn.set(u=0.8, x=0.5)
           >>> params = syn.get()
           >>> print(params['U'], params['u'], params['x'])
           0.3 0.8 0.5
        """
        new_U = self.U if U is _UNSET else self._to_scalar_unit_interval(U, name='U')
        new_u = self.u if u is _UNSET else self._to_scalar_unit_interval(u, name='u')
        new_x = self.x if x is _UNSET else self._to_scalar_float(x, name='x')
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
        self.x = float(new_x)
        self.tau_rec = float(new_tau_rec)
        self.tau_fac = float(new_tau_fac)

        self._x0 = float(self.x)
        self._u0 = float(self.u)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        r"""Schedule one outgoing event with NEST ``tsodyks2_synapse`` dynamics.

        Processes a presynaptic spike event by:

        1. Propagating synapse state (``x``, ``u``) from the last spike time
           to the current spike time using exponential decay equations.
        2. Computing the effective synaptic weight as
           :math:`w_{\mathrm{eff}} = x \cdot u \cdot w \cdot \text{multiplicity}`.
        3. Scheduling delayed delivery of the weighted event to the
           postsynaptic target.
        4. Updating ``t_lastspike`` to the current spike timestamp.

        This method implements the exact NEST ``tsodyks2_synapse`` update
        semantics, including the special case for ``tau_fac == 0``.

        Parameters
        ----------
        multiplicity : float or array-like, optional
            Presynaptic spike multiplicity. Scales the effective weight
            linearly. For standard point-neuron spikes, this is ``1.0``.
            For population models or weighted spike counts, can be > 1.
            Default: ``1.0``.
        post : object, optional
            Target postsynaptic receiver for this event. If ``None``, uses
            the default receiver set during construction. Default: ``None``.
        receptor_type : int or array-like, optional
            Target receptor port for this event. If ``None``, uses the
            default ``receptor_type`` set during construction.
            Default: ``None``.

        Returns
        -------
        bool
            ``True`` if an event was scheduled (multiplicity non-zero),
            ``False`` otherwise.

        Notes
        -----
        - If ``multiplicity`` is zero or evaluates to zero after
          conversion, no event is scheduled and state is not updated.
        - The spike timestamp is computed as ``current_time + dt``, where
          ``dt`` is the simulation timestep. This matches NEST's on-grid
          spike timing convention.
        - For the first spike (``t_lastspike < 0``), state propagation is
          skipped and initial ``x`` and ``u`` values are used directly.
        - State variables ``x`` and ``u`` are updated in place during
          this call, affecting all subsequent spikes.
        - The effective weight calculation includes the multiplicity factor,
          so multiple coincident spikes can be represented as a single call
          with ``multiplicity > 1``.

        Examples
        --------
        .. code-block:: python

           >>> import brainpy.state as bst
           >>> import brainunit as u
           >>> # Create synapse with target
           >>> class Target(bst.nn.Dynamics):
           ...     def __init__(self):
           ...         super().__init__()
           ...         self.current = 0.0
           >>> target = Target()
           >>> syn = bst.tsodyks2_synapse(
           ...     weight=1.0,
           ...     delay=1.0 * u.ms,
           ...     U=0.5,
           ...     tau_rec=100.0 * u.ms,
           ...     post=target
           ... )
           >>> syn.init_state()
           >>> # Send a spike
           >>> with bst.environ.context(dt=0.1 * u.ms):
           ...     success = syn.send(multiplicity=1.0)
           >>> print(success)
           True
           >>> # Send multiple spikes in quick succession
           >>> for i in range(5):
           ...     syn.send(multiplicity=1.0)
           ...     state = syn.get()
           ...     print(f"Spike {i+1}: x={state['x']:.3f}, u={state['u']:.3f}")
        """
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST evaluates this model on the spike stamp.
        t_spike = self._current_time_ms() + dt_ms

        if self.t_lastspike >= 0.0:
            h = float(t_spike - self.t_lastspike)
            x_decay = math.exp(-h / self.tau_rec)
            u_decay = 0.0 if self.tau_fac == 0.0 else math.exp(-h / self.tau_fac)

            # Keep ordering identical to NEST models/tsodyks2_synapse.h::send.
            self.x = 1.0 + (self.x - self.x * self.u - 1.0) * x_decay
            self.u = self.U + self.u * (1.0 - self.U) * u_decay

        receiver = self._resolve_receiver(post)
        weighted_payload = multiplicity * (self.x * self.u * self.weight)
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
        """Deliver due events, then schedule current-step presynaptic input.

        Primary update method called at each simulation timestep. Performs
        two operations in sequence:

        1. **Delivery phase**: Delivers all events scheduled for the current
           timestep to their target receivers (accumulated from past spikes
           with appropriate delays).
        2. **Reception phase**: Processes current presynaptic input by
           aggregating spike counts and calling :meth:`send` to schedule
           new delayed events.

        This method integrates with the event queue system inherited from
        :class:`static_synapse` and handles both current and delta input
        collection via :meth:`sum_current_inputs` and :meth:`sum_delta_inputs`.

        Parameters
        ----------
        pre_spike : float or array-like, optional
            Presynaptic spike input at the current timestep. Represents
            spike multiplicity (typically ``0.0`` for no spike, ``1.0`` for
            a single spike). Can be > 1 for population models.
            Default: ``0.0``.
        post : object, optional
            Target postsynaptic receiver for new events. If ``None``, uses
            the default receiver set during construction. Default: ``None``.
        receptor_type : int or array-like, optional
            Target receptor port for new events. If ``None``, uses the
            default ``receptor_type`` set during construction.
            Default: ``None``.

        Returns
        -------
        int
            Number of events delivered to postsynaptic targets during the
            delivery phase of this timestep.

        Notes
        -----
        - The method first delivers events from the queue, then processes
          new presynaptic input. This ordering ensures causality: events
          generated at timestep :math:`t` are delivered at :math:`t + d`,
          where :math:`d` is the delay in timesteps.
        - Current inputs are accumulated from multiple sources via
          :meth:`sum_current_inputs`, allowing multiple projections to
          target the same synapse.
        - Delta inputs (instantaneous inputs delivered in the same timestep)
          are also accumulated via :meth:`sum_delta_inputs`.
        - If the total aggregated input is zero, no new event is scheduled
          and synapse state remains unchanged.

        Examples
        --------
        .. code-block:: python

           >>> import brainpy.state as bst
           >>> import brainunit as u
           >>> # Create synapse with target
           >>> class Target(bst.nn.Dynamics):
           ...     def __init__(self):
           ...         super().__init__()
           ...         self.current = 0.0
           ...     def update(self, inp=0.0):
           ...         self.current += inp
           >>> target = Target()
           >>> syn = bst.tsodyks2_synapse(
           ...     weight=1.0,
           ...     delay=1.0 * u.ms,
           ...     U=0.5,
           ...     tau_rec=200.0 * u.ms,
           ...     post=target
           ... )
           >>> syn.init_state()
           >>> target.init_state()
           >>> # Simulate spike train
           >>> with bst.environ.context(dt=0.1 * u.ms):
           ...     for t in range(20):
           ...         spike = 1.0 if t in [0, 5, 10] else 0.0
           ...         n_delivered = syn.update(pre_spike=spike)
           ...         if n_delivered > 0:
           ...             print(f"Step {t}: delivered {n_delivered} events")
        """
        dt_ms = self._refresh_delay_if_needed()
        step = self._curr_step(dt_ms)
        delivered = self._deliver_due_events(step)

        total = self.sum_current_inputs(pre_spike)
        total = self.sum_delta_inputs(total)
        if self._is_nonzero(total):
            self.send(total, post=post, receptor_type=receptor_type)

        return delivered
