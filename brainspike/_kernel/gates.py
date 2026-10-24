# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Voltage gates of ``conductance`` ports (K0 spec §4.2, task T8).

A gate is a callable ``gate(V) -> dimensionless`` stored on a
:class:`~brainspike._kernel.ports.Conductance` port. The port current
:math:`g\,\mathrm{gate}(V)\,(E_{rev} - V)` is computed by
:func:`~brainspike._kernel.model.conductance_current` inside the model's
``derivatives``, so the gate sees the membrane potential of every RK substep.

Gates are frozen value objects: their constants carry units when given and are
stored as canonical Python floats, so a gate is hashable, comparable and adds no
dtype of its own (the result follows the dtype of ``V``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import brainunit as u
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.units import check_unit, to_mantissa

__all__ = ['MgBlock']

_PER_MV = u.get_unit(1.0 / u.mV)


def _canonical(value: Any, unit: Any, what: str) -> float:
    check_unit(value, unit, f'MgBlock {what}')
    # convert on the host in float64: ``to_mantissa`` would round Python-float
    # constants to float32 when x64 is off at construction time
    m = np.asarray(value.to_decimal(unit) if isinstance(value, u.Quantity) else value,
                   dtype=np.float64)
    if m.ndim != 0:
        raise ValueError(f'MgBlock {what} must be a scalar, got shape {m.shape}')
    return float(m)


@dataclass(frozen=True, init=False)
class MgBlock:
    r"""Voltage-dependent Mg\ :sup:`2+` block of NMDA receptors (Jahr & Stevens 1990).

    .. math::

       B(V) = \left(1 + \frac{[\mathrm{Mg}^{2+}]_o}{\beta}\,
       e^{-\alpha\,(V - V_{\mathrm{offset}})}\right)^{-1}

    Parameters
    ----------
    Mg : brainunit.Quantity, default ``1.2 * u.mM``
        Extracellular magnesium concentration (a concentration, >= 0).
    alpha : brainunit.Quantity, default ``0.062 / u.mV``
        Voltage sensitivity (inverse voltage).
    beta : brainunit.Quantity, default ``3.57 * u.mM``
        Mg\ :sup:`2+` dissociation constant (a concentration, > 0).
    V_offset : brainunit.Quantity, default ``0 * u.mV``
        Voltage offset.

    Raises
    ------
    UnitMismatchError
        If a constant has the wrong dimension (e.g. a bare ``0.062`` for ``alpha``).
    ValueError
        If ``Mg < 0``, ``beta <= 0`` or a constant is not a scalar.

    Notes
    -----
    Fixes goal-01 bug 2 by construction: the old ``_brainpy`` ``MgBlock`` had
    unitless defaults and failed with a unit-carrying ``V``. Here ``V`` **must**
    carry a voltage unit (a bare array raises), so the curve cannot silently
    depend on the voltage unit.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainspike._kernel.gates import MgBlock
        >>> B = MgBlock()
        >>> round(float(B(0.0 * u.mV)), 6)
        0.748428
        >>> round(float(B(-0.065 * u.volt)), 6)
        0.050223
        >>> float(MgBlock(Mg=0.0 * u.mM)(-80.0 * u.mV))
        1.0
    """

    Mg_mM: float
    alpha_per_mV: float
    beta_mM: float
    V_offset_mV: float

    def __init__(self, Mg: Any = 1.2 * u.mM, alpha: Any = 0.062 / u.mV,
                 beta: Any = 3.57 * u.mM, V_offset: Any = 0.0 * u.mV) -> None:
        Mg = _canonical(Mg, u.mM, 'Mg')
        alpha = _canonical(alpha, _PER_MV, 'alpha')
        beta = _canonical(beta, u.mM, 'beta')
        V_offset = _canonical(V_offset, u.mV, 'V_offset')
        if Mg < 0:
            raise ValueError(f'MgBlock Mg must be >= 0, got {Mg} mM')
        if beta <= 0:
            raise ValueError(f'MgBlock beta must be > 0, got {beta} mM')
        object.__setattr__(self, 'Mg_mM', Mg)
        object.__setattr__(self, 'alpha_per_mV', alpha)
        object.__setattr__(self, 'beta_mM', beta)
        object.__setattr__(self, 'V_offset_mV', V_offset)

    @property
    def Mg(self) -> u.Quantity:
        """Extracellular Mg\\ :sup:`2+` concentration."""
        return self.Mg_mM * u.mM

    @property
    def alpha(self) -> u.Quantity:
        """Voltage sensitivity."""
        return self.alpha_per_mV * _PER_MV

    @property
    def beta(self) -> u.Quantity:
        """Dissociation constant."""
        return self.beta_mM * u.mM

    @property
    def V_offset(self) -> u.Quantity:
        """Voltage offset."""
        return self.V_offset_mV * u.mV

    def __call__(self, V: Any) -> Any:
        """Unblocked fraction ``B(V)`` (dimensionless array, dtype of ``V``).

        Parameters
        ----------
        V : brainunit.Quantity
            Membrane potential (any voltage unit, any shape; may be traced).

        Raises
        ------
        UnitMismatchError
            If ``V`` is not a voltage.
        """
        check_unit(V, u.mV, 'MgBlock V')
        v = to_mantissa(V, u.mV)
        dtype = v.dtype if jnp.issubdtype(v.dtype, jnp.floating) else jnp.result_type(float)
        v = v.astype(dtype)
        ratio = jnp.asarray(self.Mg_mM / self.beta_mM, dtype)
        return 1.0 / (1.0 + ratio * jnp.exp(-jnp.asarray(self.alpha_per_mV, dtype)
                                            * (v - jnp.asarray(self.V_offset_mV, dtype))))
