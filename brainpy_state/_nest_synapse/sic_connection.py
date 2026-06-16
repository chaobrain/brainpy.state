from typing import Any

import brainstate
import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

from brainpy_state._nest_base.base import NESTSynapse

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
    'sic_connection',
]


class sic_connection(NESTSynapse):
    r"""NEST-compatible ``sic_connection`` synapse model for astrocyte-to-neuron slow inward current (SIC) coupling.

    This class is the NEST-parity connection *spec* for ``sic_connection``
    (``models/sic_connection.{h,cpp}``): a one-way astrocyte->neuron edge carrying the
    slow inward current. It stores the scalar ``weight`` (a unitless multiplier of the
    astrocyte's per-step ``SIC``) and the integer ``delay_steps``, validates the
    sender/receiver model pair, and is consumed by :class:`~brainpy_state.Simulator`,
    which builds the routing — an ``as_current``
    :class:`~brainpy_state._nest_network.event_proj.EventProjection` that reads the
    astrocyte's emission holder and deposits ``weight·SIC`` into the neuron's labelled
    ``'I_SIC'`` current channel each step.

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

    **3. Substrate delivery**

    On the :class:`~brainpy_state.Simulator` substrate the astrocyte emits its per-step
    graded ``SIC`` continuously (seam-(H) emission). The Simulator reads that emission
    holder and deposits

    .. math::

       I_{\text{SIC}} = w \cdot \mathrm{SIC}[n-1]

    into the neuron's labelled ``'I_SIC'`` current channel, which the neuron reads (and
    pops) **before** its ``I_stim`` read so the device current and the SIC current never
    collide. The edge is one-way (a NEST ``SICEvent`` has no back-channel).
    ``delay_steps=1`` (the NEST default / minimum delay) rides the substrate's intrinsic
    one-step pipeline latency; a larger ``delay_steps`` adds ``(delay_steps - 1)``
    buffered steps. There is **no** host-side event queue: the SIC current is a
    State-backed channel, so the whole simulation lowers into one compiled ``for_loop``.

    Parameters
    ----------
    weight : float, array-like, optional
        Unitless weight multiplying the astrocyte's per-step ``SIC``. Must be scalar.
        Default: ``1.0``.
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
    - This class is the connection *spec* only. The per-step ``SIC`` is generated by
      the source model (``astrocyte_lr_1994``) and the membrane integration by the
      target (``aeif_cond_alpha_astro``); the :class:`~brainpy_state.Simulator` builds
      the routing between them.
    - Connection objects are stateless; all state (membrane conductances, SIC time
      courses) is maintained by source and target neuron models.

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

        >>> from brainpy import state as bp
        >>> conn = bp.sic_connection(weight=0.5, delay_steps=2)
        >>> conn.get_status()
        {'weight': 0.5, 'delay_steps': 2, 'delay': 2, ...}

    **Wire it on a Simulator (astrocyte -> neuron):**

    .. code-block:: python

        >>> import brainunit as u
        >>> sim = bp.Simulator(dt=0.1 * u.ms)
        >>> astro = sim.create(bp.astrocyte_lr_1994, 1)
        >>> neuron = sim.create(bp.aeif_cond_alpha_astro, 1)
        >>> proj = sim.connect(astro, neuron, synapse=bp.sic_connection(weight=0.5))

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
        super().__init__(in_size=1, name=name)
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
        dftype = brainstate.environ.dftype()
        arr = np.asarray(u.math.asarray(value), dtype=dftype).reshape(-1)
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
