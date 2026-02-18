from brainpy_state._nest._base import NESTSynapse
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



from typing import Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

__all__ = [
    'sic_connection',
]


class sic_connection(NESTSynapse):
    r"""NEST-compatible ``sic_connection`` synapse model for astrocyte-to-neuron slow inward current (SIC) coupling.

    This class implements connection-level semantics of NEST's ``sic_connection`` model
    (``models/sic_connection.{h,cpp}``), which transmits slow inward current coefficients
    from astrocytes to neurons via ``SICEvent`` secondary events. The connection stores
    a scalar weight that multiplies the SIC coefficient stream and supports delayed delivery.

    **1. Biological Context**

    Astrocytes can modulate neuronal excitability through release of gliotransmitters,
    triggering slow inward currents (SICs) in nearby neurons. These currents typically
    arise from extrasynaptic NMDA receptors activated by glutamate released from astrocytes,
    producing depolarizing currents that persist for hundreds of milliseconds to seconds.
    This connection model represents the functional coupling between astrocyte dynamics
    and neuronal membrane conductance.

    **2. Supported Model Pairings**

    In standard NEST model sets:

    - **Source models** (emit ``SICEvent``): ``astrocyte_lr_1994``
    - **Target models** (handle ``SICEvent``): ``aeif_cond_alpha_astro``

    Methods :meth:`supports_connection` and :meth:`check_connection` validate model
    compatibility at the model-name level.

    **3. Mathematical Formulation**

    The connection transmits a coefficient array :math:`c = [c_0, c_1, \ldots, c_{n-1}]`
    representing SIC amplitude values scheduled for future time steps. Each coefficient
    is scaled by the connection weight :math:`w` and delivered with delay :math:`d`:

    .. math::

       I_{\text{SIC}}(t + d + i) = w \cdot c_i, \qquad i = 0, 1, \ldots, n-1

    where :math:`t` is the current simulation time, :math:`d` is the connection delay,
    and :math:`i` indexes the coefficient array.

    **4. Delay Mapping and Buffer Indexing**

    NEST's target-side SIC handling in ``aeif_cond_alpha_astro::handle(SICEvent&)`` uses:

    .. math::

       \text{offset} = d - d_{\text{min}}, \qquad \text{slot}_i = \text{offset} + i

    where :math:`d` is the event delay in steps, :math:`d_{\text{min}}` is the minimum
    network delay, and :math:`i` is the coefficient index.

    Local step-based APIs in this package interpret ``delay_steps`` as a zero-indexed
    offset (``delay_steps - 1``). Therefore, for an absolute NEST delay :math:`d` and
    minimum delay :math:`d_{\text{min}}`, this class maps to local delay:

    .. math::

       d_{\text{local}} = (d - d_{\text{min}}) + 1

    This ensures proper alignment with NEST's buffer indexing. Methods
    :meth:`to_aeif_sic_event` and :meth:`coeffarray_to_step_events` apply this mapping.

    **5. Event Structure**

    The connection produces event dictionaries with:

    - ``coeffs`` (array or scalar): SIC coefficient value(s)
    - ``weight`` (float): Connection weight × multiplicity
    - ``delay_steps`` (int): Delay offset for buffer indexing
    - ``multiplicity`` (float): Event count (always 1.0 after weight multiplication)

    Parameters
    ----------
    weight : float, array-like, optional
        Synaptic weight multiplying SIC coefficients. Must be scalar. Default: ``1.0``.
    delay_steps : int, array-like, optional
        Absolute event delay in simulation steps (NEST-style). Must be scalar integer
        ``>= 1``. Default: ``1``.
    name : str, optional
        Model instance name for identification. Default: ``None``.

    Parameter Mapping
    -----------------

    ================  =============  ========  ===========================================
    Parameter         NEST Name      Unit      Description
    ================  =============  ========  ===========================================
    weight            weight         unitless  Coefficient scaling factor
    delay_steps       delay          steps     Absolute transmission delay
    ================  =============  ========  ===========================================

    Attributes
    ----------
    weight : float
        Validated scalar weight.
    delay_steps : int
        Validated delay in steps (≥ 1).
    name : str or None
        Optional instance identifier.
    HAS_DELAY : bool
        Class attribute, always ``True`` (supports delayed transmission).
    SUPPORTS_WFR : bool
        Class attribute, always ``False`` (waveform relaxation not supported).
    SUPPORTED_SOURCES : tuple of str
        Valid source model names (``'astrocyte_lr_1994'``).
    SUPPORTED_TARGETS : tuple of str
        Valid target model names (``'aeif_cond_alpha_astro'``).

    Raises
    ------
    ValueError
        If ``weight`` or ``delay_steps`` are not scalar, or if ``delay_steps < 1``.

    Notes
    -----
    - This class represents connection semantics only. SIC coefficient generation
      is handled by the source model (e.g., ``astrocyte_lr_1994``).
    - Connection objects are stateless; all state (membrane conductances, SIC time
      courses) is maintained by source and target neuron models.
    - The ``multiplicity`` parameter in event methods allows scaling weights for
      multi-synapse connections but is always normalized to 1.0 in final events.

    References
    ----------
    .. [1] NEST source code: ``models/sic_connection.h`` and ``models/sic_connection.cpp``.
    .. [2] NEST receiver logic: ``models/aeif_cond_alpha_astro.cpp``, ``handle(SICEvent&)`` method.
    .. [3] NEST test suite: ``testsuite/pytests/test_sic_connection.py``.
    .. [4] Li, Y. X., & Rinzel, J. (1994). Equations for InsP3 receptor-mediated
           [Ca2+]i oscillations derived from a detailed kinetic model: a Hodgkin-Huxley
           like formalism. *Journal of Theoretical Biology*, 166(4), 461-473.

    Examples
    --------
    **Basic connection setup:**

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> conn = bp.sic_connection(weight=0.5, delay_steps=2)
        >>> conn.get_status()
        {'weight': 0.5, 'delay_steps': 2, 'delay': 2, ...}

    **Create SIC event for target neuron:**

    .. code-block:: python

        >>> coeffs = [0.1, 0.3, 0.5, 0.3, 0.1]  # Time-varying SIC profile
        >>> event = conn.to_aeif_sic_event(
        ...     coeffarray=coeffs,
        ...     min_delay_steps=1,
        ...     multiplicity=1.0
        ... )
        >>> event['weight']  # Scaled by connection weight
        0.5
        >>> event['delay_steps']  # Local delay: (2 - 1) + 1 = 2
        2

    **Decompose coefficient array into per-step events:**

    .. code-block:: python

        >>> events = conn.coeffarray_to_step_events(
        ...     coeffarray=[0.1, 0.2, 0.3],
        ...     min_delay_steps=1
        ... )
        >>> len(events)
        3
        >>> events[0]['delay_steps']  # First coefficient at delay 2
        2
        >>> events[2]['delay_steps']  # Third coefficient at delay 4
        4

    **Validate model compatibility:**

    .. code-block:: python

        >>> bp.sic_connection.supports_connection(
        ...     'astrocyte_lr_1994', 'aeif_cond_alpha_astro'
        ... )
        True
        >>> bp.sic_connection.check_connection(
        ...     'astrocyte_lr_1994', 'iaf_psc_alpha'
        ... )  # doctest: +SKIP
        ValueError: Unsupported sic_connection pair...
    """

    __module__ = 'brainpy.state'

    HAS_DELAY = True
    SUPPORTS_WFR = False
    SUPPORTED_SOURCES = ('astrocyte_lr_1994',)
    SUPPORTED_TARGETS = ('aeif_cond_alpha_astro',)

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay_steps: ArrayLike = 1,
        name: str | None = None,
    ):
        self.name = name
        self.weight = self._to_float_scalar(weight, name='weight')
        self.delay_steps = self._validate_delay_steps(delay_steps)

    @property
    def properties(self) -> dict[str, Any]:
        r"""Return connection capability flags and supported model types.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``'has_delay'`` (bool): Always ``True`` for ``sic_connection``.
            - ``'supports_wfr'`` (bool): Always ``False`` (waveform relaxation not implemented).
            - ``'supported_sources'`` (tuple[str, ...]): Model names that can emit ``SICEvent``.
            - ``'supported_targets'`` (tuple[str, ...]): Model names that can receive ``SICEvent``.

        Notes
        -----
        This property provides introspection for connection compatibility checking and
        simulation infrastructure setup (e.g., delay buffer allocation).
        """
        return {
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
            'supported_sources': self.SUPPORTED_SOURCES,
            'supported_targets': self.SUPPORTED_TARGETS,
        }

    def get_status(self) -> dict[str, Any]:
        r"""Return complete connection state and metadata.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

            - ``'weight'`` (float): Connection weight.
            - ``'delay_steps'`` (int): Delay in simulation steps.
            - ``'delay'`` (int): Alias for ``delay_steps`` (NEST compatibility).
            - ``'size_of'`` (int): Memory footprint in bytes (Python object overhead).
            - ``'has_delay'``, ``'supports_wfr'``, ``'supported_sources'``,
              ``'supported_targets'``: Connection properties (see :meth:`properties`).

        Notes
        -----
        This method mirrors NEST's ``GetStatus`` API, providing a snapshot of all
        connection parameters and capabilities. The ``delay`` key is an alias for
        ``delay_steps`` to maintain NEST naming conventions.
        """
        return {
            'weight': float(self.weight),
            'delay_steps': int(self.delay_steps),
            'delay': int(self.delay_steps),
            'size_of': int(self.__sizeof__()),
            'has_delay': self.HAS_DELAY,
            'supports_wfr': self.SUPPORTS_WFR,
            'supported_sources': self.SUPPORTED_SOURCES,
            'supported_targets': self.SUPPORTED_TARGETS,
        }

    def set_status(self, status: dict[str, Any] | None = None, **kwargs):
        r"""Update connection parameters (NEST ``SetStatus`` API).

        Parameters
        ----------
        status : dict[str, Any], optional
            Dictionary of parameters to update. Valid keys: ``'weight'``, ``'delay'``,
            ``'delay_steps'``. Default: ``None``.
        **kwargs
            Additional keyword arguments merged with ``status`` dict.

        Raises
        ------
        ValueError
            If both ``'delay'`` and ``'delay_steps'`` are provided with different values,
            or if parameter values fail validation (non-scalar, delay < 1, etc.).

        Notes
        -----
        - If both ``'delay'`` and ``'delay_steps'`` are present, they must be identical
          (they are aliases in NEST).
        - Validation is delegated to :meth:`set_weight` and :meth:`set_delay_steps`.
        - Unknown keys are silently ignored (NEST-compatible behavior).

        Examples
        --------
        .. code-block:: python

            >>> conn = bp.sic_connection(weight=1.0, delay_steps=1)
            >>> conn.set_status(weight=2.5, delay=3)
            >>> conn.get_status()['weight']
            2.5
            >>> conn.get_status()['delay_steps']
            3
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

    def get(self, key: str = 'status'):
        r"""Retrieve connection parameter or full status dictionary.

        Parameters
        ----------
        key : str, optional
            Parameter name to retrieve. If ``'status'`` (default), returns full status
            dictionary from :meth:`get_status`. Otherwise, must be a valid status key
            (e.g., ``'weight'``, ``'delay'``, ``'delay_steps'``).

        Returns
        -------
        Any
            If ``key == 'status'``: full status dictionary.
            Otherwise: value of the requested parameter.

        Raises
        ------
        KeyError
            If ``key`` is not ``'status'`` and not present in the status dictionary.

        Examples
        --------
        .. code-block:: python

            >>> conn = bp.sic_connection(weight=0.8, delay_steps=2)
            >>> conn.get('weight')
            0.8
            >>> conn.get('status')['delay']
            2
        """
        if key == 'status':
            return self.get_status()
        status = self.get_status()
        if key in status:
            return status[key]
        raise KeyError(f'Unsupported key "{key}" for sic_connection.get().')

    def set_weight(self, weight: ArrayLike):
        r"""Update connection weight.

        Parameters
        ----------
        weight : float or array-like
            New synaptic weight. Must be scalar after conversion.

        Raises
        ------
        ValueError
            If ``weight`` is not scalar.

        Examples
        --------
        .. code-block:: python

            >>> conn = bp.sic_connection()
            >>> conn.set_weight(2.5)
            >>> conn.weight
            2.5
        """
        self.weight = self._to_float_scalar(weight, name='weight')

    def set_delay(self, delay: ArrayLike):
        r"""Update connection delay (alias for :meth:`set_delay_steps`).

        Parameters
        ----------
        delay : int or array-like
            New delay in simulation steps. Must be scalar integer ``>= 1``.

        Raises
        ------
        ValueError
            If ``delay`` is not scalar integer or ``< 1``.
        """
        self.delay_steps = self._validate_delay_steps(delay, name='delay')

    def set_delay_steps(self, delay_steps: ArrayLike):
        r"""Update connection delay in simulation steps.

        Parameters
        ----------
        delay_steps : int or array-like
            New delay in steps. Must be scalar integer ``>= 1``.

        Raises
        ------
        ValueError
            If ``delay_steps`` is not scalar integer or ``< 1``.

        Examples
        --------
        .. code-block:: python

            >>> conn = bp.sic_connection(delay_steps=1)
            >>> conn.set_delay_steps(5)
            >>> conn.delay_steps
            5
        """
        self.delay_steps = self._validate_delay_steps(delay_steps, name='delay_steps')

    @classmethod
    def _model_name(cls, model: Any) -> str:
        r"""Extract model name from string, class, or instance.

        Parameters
        ----------
        model : Any
            Model identifier (string name, class, or instance).

        Returns
        -------
        str
            Normalized model name for compatibility checking.

        Notes
        -----
        Extraction order:
        1. If ``model`` is string, return as-is.
        2. If ``model`` has ``__name__`` attribute (class), return it.
        3. If ``model`` has ``__class__.__name__`` (instance), return class name.
        4. Fallback to ``str(model)``.
        """
        if isinstance(model, str):
            return model
        if hasattr(model, '__name__'):
            return str(model.__name__)
        if hasattr(model, '__class__') and hasattr(model.__class__, '__name__'):
            return str(model.__class__.__name__)
        return str(model)

    @classmethod
    def supports_connection(cls, source_model: Any, target_model: Any) -> bool:
        r"""Check if source-target model pair is compatible with ``sic_connection``.

        Parameters
        ----------
        source_model : Any
            Source model (string name, class, or instance). Must be in
            ``SUPPORTED_SOURCES`` (``'astrocyte_lr_1994'``).
        target_model : Any
            Target model (string name, class, or instance). Must be in
            ``SUPPORTED_TARGETS`` (``'aeif_cond_alpha_astro'``).

        Returns
        -------
        bool
            ``True`` if both models are in their respective supported lists, ``False`` otherwise.

        Examples
        --------
        .. code-block:: python

            >>> bp.sic_connection.supports_connection(
            ...     'astrocyte_lr_1994', 'aeif_cond_alpha_astro'
            ... )
            True
            >>> bp.sic_connection.supports_connection(
            ...     'iaf_psc_alpha', 'aeif_cond_alpha_astro'
            ... )
            False
        """
        src = cls._model_name(source_model)
        tgt = cls._model_name(target_model)
        return src in cls.SUPPORTED_SOURCES and tgt in cls.SUPPORTED_TARGETS

    @classmethod
    def check_connection(cls, source_model: Any, target_model: Any) -> bool:
        r"""Validate source-target model pair and raise error if incompatible.

        Parameters
        ----------
        source_model : Any
            Source model (string name, class, or instance).
        target_model : Any
            Target model (string name, class, or instance).

        Returns
        -------
        bool
            Always returns ``True`` if validation passes.

        Raises
        ------
        ValueError
            If the model pair is not supported (see :meth:`supports_connection`).
            Error message includes actual and expected model names.

        Examples
        --------
        .. code-block:: python

            >>> bp.sic_connection.check_connection(
            ...     'astrocyte_lr_1994', 'aeif_cond_alpha_astro'
            ... )
            True
            >>> bp.sic_connection.check_connection(
            ...     'iaf_psc_alpha', 'aeif_cond_alpha_astro'
            ... )  # doctest: +SKIP
            ValueError: Unsupported sic_connection pair...
        """
        ok = cls.supports_connection(source_model, target_model)
        if not ok:
            src = cls._model_name(source_model)
            tgt = cls._model_name(target_model)
            raise ValueError(
                f'Unsupported sic_connection pair: source={src}, target={tgt}. '
                f'Expected source in {cls.SUPPORTED_SOURCES} and target in {cls.SUPPORTED_TARGETS}.'
            )
        return True

    def prepare_secondary_event(
        self,
        coeffarray: ArrayLike,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        r"""Create a NEST-style ``SICEvent`` payload with coefficient array.

        Parameters
        ----------
        coeffarray : array-like
            1D array of SIC coefficients for future time steps. Must be non-empty.
            Shape: ``(n_steps,)``.
        delay_steps : int, array-like, optional
            Override connection delay. If ``None``, uses ``self.delay_steps``. Default: ``None``.

        Returns
        -------
        dict[str, Any]
            NEST-compatible event dictionary:

            - ``'coeffarray'`` (ndarray): Validated float64 coefficient array, shape ``(n_steps,)``.
            - ``'weight'`` (float): Connection weight.
            - ``'delay_steps'`` (int): Absolute delay in simulation steps.

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty, or if ``delay_steps`` validation fails.

        Notes
        -----
        This method produces raw NEST-style events. For local step-based consumption,
        use :meth:`to_aeif_sic_event` or :meth:`coeffarray_to_step_events` which apply
        delay mapping.

        Examples
        --------
        .. code-block:: python

            >>> conn = bp.sic_connection(weight=0.5, delay_steps=2)
            >>> event = conn.prepare_secondary_event([0.1, 0.3, 0.5])
            >>> event['coeffarray']
            array([0.1, 0.3, 0.5])
            >>> event['delay_steps']
            2
        """
        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        return {
            'coeffarray': self._to_coeff_array(coeffarray),
            'weight': float(self.weight),
            'delay_steps': int(d),
        }

    def to_aeif_sic_event(
        self,
        coeffarray: ArrayLike,
        min_delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        r"""Create SIC event payload consumable by ``aeif_cond_alpha_astro.update``.

        Applies delay mapping from NEST absolute delay to local step-based offset:
        ``local_delay = (delay - min_delay) + 1``.

        Parameters
        ----------
        coeffarray : array-like
            1D array of SIC coefficients. Shape: ``(n_steps,)``.
        min_delay_steps : int, array-like, optional
            Minimum network delay (NEST-style). Used for delay offset calculation.
            Default: ``1``.
        multiplicity : float, array-like, optional
            Scaling factor for connection weight (e.g., for multi-synapse connections).
            Default: ``1.0``.
        delay_steps : int, array-like, optional
            Override connection delay. If ``None``, uses ``self.delay_steps``. Default: ``None``.

        Returns
        -------
        dict[str, Any]
            Local event dictionary:

            - ``'coeffs'`` (ndarray): Coefficient array, shape ``(n_steps,)``.
            - ``'weight'`` (float): Connection weight × multiplicity.
            - ``'delay_steps'`` (int): Local delay offset = ``(delay - min_delay) + 1``.
            - ``'multiplicity'`` (float): Always ``1.0`` (absorbed into weight).

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty, if ``delay_steps < min_delay_steps``, or if
            any parameter is not scalar where required.

        Notes
        -----
        This method is the primary interface for integrating ``sic_connection`` events
        into step-based neuron update loops. The delay mapping ensures compatibility
        with NEST's buffer indexing semantics.

        Examples
        --------
        .. code-block:: python

            >>> conn = bp.sic_connection(weight=0.5, delay_steps=3)
            >>> event = conn.to_aeif_sic_event(
            ...     coeffarray=[0.1, 0.2],
            ...     min_delay_steps=1,
            ...     multiplicity=2.0
            ... )
            >>> event['weight']  # 0.5 * 2.0
            1.0
            >>> event['delay_steps']  # (3 - 1) + 1
            3
        """
        coeff = self._to_coeff_array(coeffarray)
        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        local_delay = self._to_local_delay_steps(d, min_delay_steps=min_delay_steps)
        mult = self._to_float_scalar(multiplicity, name='multiplicity')
        return {
            'coeffs': coeff,
            'weight': float(self.weight * mult),
            'delay_steps': int(local_delay),
            'multiplicity': 1.0,
        }

    def to_sic_event(
        self,
        coeff: ArrayLike,
        min_delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> dict[str, Any]:
        r"""Create single-coefficient SIC event for local step-based APIs.

        This is an alias for :meth:`to_aeif_sic_event` supporting both single-value
        and array-valued coefficients.

        Parameters
        ----------
        coeff : float, array-like
            SIC coefficient value(s). Can be scalar or array.
        min_delay_steps : int, array-like, optional
            Minimum network delay. Default: ``1``.
        multiplicity : float, array-like, optional
            Weight scaling factor. Default: ``1.0``.
        delay_steps : int, array-like, optional
            Override delay. Default: ``None`` (use ``self.delay_steps``).

        Returns
        -------
        dict[str, Any]
            See :meth:`to_aeif_sic_event` return value.

        Examples
        --------
        .. code-block:: python

            >>> conn = bp.sic_connection(weight=0.8, delay_steps=2)
            >>> event = conn.to_sic_event(coeff=0.5, min_delay_steps=1)
            >>> event['coeffs']
            array([0.5])
            >>> event['delay_steps']
            2
        """
        return self.to_aeif_sic_event(
            coeffarray=coeff,
            min_delay_steps=min_delay_steps,
            multiplicity=multiplicity,
            delay_steps=delay_steps,
        )

    def coeffarray_to_step_events(
        self,
        coeffarray: ArrayLike,
        min_delay_steps: ArrayLike = 1,
        multiplicity: ArrayLike = 1.0,
        delay_steps: ArrayLike | None = None,
    ) -> list[dict[str, Any]]:
        r"""Map lag-indexed SIC coefficients to one event per future time step.

        Decomposes an ``n``-element coefficient array into ``n`` separate events,
        each scheduled for a distinct future step. Coefficient ``i`` is delivered
        at ``local_delay + i`` steps in the future.

        Parameters
        ----------
        coeffarray : array-like
            1D array of SIC coefficients, length ``n``. Coefficient ``i`` corresponds
            to time step ``current_time + delay + i``.
        min_delay_steps : int, array-like, optional
            Minimum network delay for delay mapping. Default: ``1``.
        multiplicity : float, array-like, optional
            Weight scaling factor. Default: ``1.0``.
        delay_steps : int, array-like, optional
            Override connection delay. Default: ``None`` (use ``self.delay_steps``).

        Returns
        -------
        list[dict[str, Any]]
            List of ``n`` event dictionaries, one per coefficient. Each contains:

            - ``'coeffs'`` (float): Single coefficient value.
            - ``'weight'`` (float): Connection weight × multiplicity.
            - ``'delay_steps'`` (int): Local delay = ``base_delay + i`` for index ``i``.
            - ``'multiplicity'`` (float): Always ``1.0``.

        Raises
        ------
        ValueError
            If ``coeffarray`` is empty or parameter validation fails.

        Notes
        -----
        This method is useful for event-driven simulations where SIC coefficients
        need to be delivered as discrete time-stamped events rather than as a
        bundled array. The base delay is computed as ``(delay - min_delay) + 1``,
        and each coefficient adds its index to this base.

        Examples
        --------
        .. code-block:: python

            >>> conn = bp.sic_connection(weight=0.5, delay_steps=2)
            >>> events = conn.coeffarray_to_step_events(
            ...     coeffarray=[0.1, 0.3, 0.5],
            ...     min_delay_steps=1
            ... )
            >>> len(events)
            3
            >>> events[0]['coeffs']
            0.1
            >>> events[0]['delay_steps']  # (2 - 1) + 1 + 0
            2
            >>> events[2]['delay_steps']  # (2 - 1) + 1 + 2
            4
        """
        coeff = self._to_coeff_array(coeffarray)
        d = self.delay_steps if delay_steps is None else self._validate_delay_steps(delay_steps, name='delay_steps')
        local_delay = self._to_local_delay_steps(d, min_delay_steps=min_delay_steps)
        mult = self._to_float_scalar(multiplicity, name='multiplicity')
        weight = float(self.weight * mult)

        events = []
        for i, c in enumerate(coeff):
            events.append(
                {
                    'coeffs': float(c),
                    'weight': weight,
                    'delay_steps': int(local_delay + i),
                    'multiplicity': 1.0,
                }
            )
        return events

    @classmethod
    def _to_local_delay_steps(
        cls,
        delay_steps: ArrayLike,
        min_delay_steps: ArrayLike = 1,
    ) -> int:
        r"""Convert NEST absolute delay to local step-based offset.

        Implements the mapping: ``local_delay = (delay - min_delay) + 1``.

        Parameters
        ----------
        delay_steps : int, array-like
            Absolute NEST-style delay in steps.
        min_delay_steps : int, array-like, optional
            Minimum network delay. Default: ``1``.

        Returns
        -------
        int
            Local delay offset for buffer indexing.

        Raises
        ------
        ValueError
            If ``delay_steps < min_delay_steps`` (violates causality).

        Notes
        -----
        NEST uses absolute delays and computes buffer offsets as ``delay - min_delay``.
        Local step-based APIs use zero-indexed delays, so we add 1 to align with
        NEST's buffer indexing semantics where delay=1 corresponds to the next step.
        """
        delay = cls._validate_delay_steps(delay_steps, name='delay_steps')
        min_delay = cls._validate_delay_steps(min_delay_steps, name='min_delay_steps')
        if delay < min_delay:
            raise ValueError('delay_steps must be >= min_delay_steps.')
        return int(delay - min_delay + 1)

    @staticmethod
    def _to_coeff_array(value: ArrayLike) -> np.ndarray:
        r"""Convert input to validated 1D float64 coefficient array.

        Parameters
        ----------
        value : array-like
            Coefficient value(s). May be scalar, array, or brainunit Quantity.

        Returns
        -------
        ndarray
            Validated 1D float64 array, shape ``(n,)`` where ``n >= 1``.

        Raises
        ------
        ValueError
            If the result is empty (size 0).

        Notes
        -----
        - Strips units from brainunit Quantity objects.
        - Flattens multi-dimensional inputs to 1D.
        - Scalars are converted to shape ``(1,)`` arrays.
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError('Coefficient array must not be empty.')
        return arr

    @staticmethod
    def _to_float_scalar(value: ArrayLike, name: str) -> float:
        r"""Convert input to validated scalar float.

        Parameters
        ----------
        value : array-like
            Input value (may be scalar, array, or brainunit Quantity).
        name : str
            Parameter name for error messages.

        Returns
        -------
        float
            Validated scalar float64 value.

        Raises
        ------
        ValueError
            If the input is not scalar (size != 1 after flattening).

        Notes
        -----
        - Strips units from brainunit Quantity objects.
        - Flattens multi-dimensional inputs before checking size.
        """
        if isinstance(value, u.Quantity):
            value = u.get_mantissa(value)
        arr = np.asarray(u.math.asarray(value), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return float(arr[0])

    @staticmethod
    def _to_int_scalar(value: ArrayLike, name: str) -> int:
        r"""Convert input to validated scalar integer.

        Parameters
        ----------
        value : array-like
            Input value (may be scalar, array, or brainunit Quantity).
        name : str
            Parameter name for error messages.

        Returns
        -------
        int
            Validated integer value (rounded if float is within tolerance).

        Raises
        ------
        ValueError
            If the input is:
            - Not scalar (size != 1 after flattening).
            - Not finite (NaN or ±inf).
            - Not integer-valued (differs from nearest int by > 1e-12).

        Notes
        -----
        - Strips units from brainunit Quantity objects.
        - Allows float inputs if they are integer-valued within 1e-12 tolerance.
        """
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

    @classmethod
    def _validate_delay_steps(cls, delay_steps: ArrayLike, name: str = 'delay_steps') -> int:
        r"""Validate delay parameter and ensure it is a positive integer.

        Parameters
        ----------
        delay_steps : array-like
            Delay value in simulation steps.
        name : str, optional
            Parameter name for error messages. Default: ``'delay_steps'``.

        Returns
        -------
        int
            Validated delay value (≥ 1).

        Raises
        ------
        ValueError
            If ``delay_steps`` is not scalar, not integer-valued, or ``< 1``.

        Notes
        -----
        Minimum delay of 1 step is required for causal event delivery in NEST semantics.
        """
        d = cls._to_int_scalar(delay_steps, name=name)
        if d < 1:
            raise ValueError(f'{name} must be >= 1.')
        return d
