# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Parameter and state declarations, distributions, and their materialisation.

A model *declares* its parameters with :class:`Params` / :class:`Param` and its
dynamical states with :class:`States` / :class:`Init`. Declarations are static,
unit-carrying descriptions; :meth:`Params.materialize` and :meth:`States.materialize`
turn them into brainstate ``State`` objects for a population of ``n`` neurons:

* trainable parameters become :class:`brainstate.ParamState`;
* all other parameters become :class:`ConstParamState` -- still a ``State`` holding a
  **traced** array, so parameter sweeps and ``vmap`` never recompile;
* float states become :class:`brainstate.HiddenState`, non-float states (integer
  counters, flags) become :class:`brainstate.ShortTermState`.

Parameters are per-neuron (shape ``(n,)``) by default, so heterogeneity needs no
opt-in. Values may be given as scalars, ``(n,)`` arrays or :class:`Distribution`
objects; distributions are unit-aware and support unit-aware clipping.

Inside ``step`` a model sees plain values gathered in a :class:`Record` (attribute
access, immutable, a JAX pytree): see :func:`read_states` / :func:`write_states`.
"""
from __future__ import annotations

import zlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.units import (
    as_unit,
    check_unit,
    from_mantissa,
    resolve_dtype,
    to_mantissa,
    unit_of,
)

__all__ = [
    'Record',
    'Distribution',
    'Normal',
    'Uniform',
    'LogNormal',
    'ConstParamState',
    'Param',
    'Params',
    'Init',
    'States',
    'read_states',
    'write_states',
    'name_key',
]


# ---------------------------------------------------------------------------
# Record: immutable attribute-access pytree of named values
# ---------------------------------------------------------------------------

class Record(Mapping):
    """Immutable mapping of named values with attribute access (a JAX pytree).

    Used for the parameter values ``p`` and state values ``s`` a model sees inside
    ``step``.

    Parameters
    ----------
    **fields
        Name/value pairs. Insertion order is preserved and part of the pytree
        structure.

    Notes
    -----
    ``r.name`` and ``r['name']`` are equivalent. :meth:`replace` returns a new
    record and refuses unknown names, so a typo in a reset map fails loudly.
    """

    __slots__ = ('_data',)

    def __init__(self, **fields: Any) -> None:
        object.__setattr__(self, '_data', dict(fields))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f'Record has no field {name!r}; fields are {list(self._data)}'
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError('Record is immutable; use .replace(...)')

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        inner = ', '.join(f'{k}={v!r}' for k, v in self._data.items())
        return f'Record({inner})'

    def replace(self, **changes: Any) -> 'Record':
        """Return a copy with ``changes`` applied.

        Raises
        ------
        KeyError
            If a name in ``changes`` is not a field of the record.
        """
        unknown = set(changes) - set(self._data)
        if unknown:
            raise KeyError(f'unknown field(s) {sorted(unknown)}; fields are {list(self._data)}')
        return Record(**{**self._data, **changes})


jax.tree_util.register_pytree_node(
    Record,
    lambda r: (tuple(r._data.values()), tuple(r._data.keys())),
    lambda keys, vals: Record(**dict(zip(keys, vals))),
)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def _opt_mantissa(x: Any, unit: u.Unit, name: str) -> float | None:
    if x is None:
        return None
    check_unit(x, unit, name)
    return float(np.asarray(to_mantissa(x, unit)))


class Distribution:
    """Base class of unit-aware random value specifications.

    Subclasses implement :meth:`_sample_mantissa` and :meth:`_support` in the unit
    :attr:`unit`. Optional clipping bounds ``low`` / ``high`` are given with units
    (any unit of the same dimension) and are applied after sampling.
    """

    unit: u.Unit
    low: float | None
    high: float | None

    def _init_clip(self, low: Any, high: Any) -> None:
        lo = _opt_mantissa(low, self.unit, 'low')
        hi = _opt_mantissa(high, self.unit, 'high')
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f'clip bounds are inverted: low={low} > high={high}')
        object.__setattr__(self, 'low', lo)
        object.__setattr__(self, 'high', hi)

    def _sample_mantissa(self, key: jax.Array, shape: tuple, dtype: Any) -> jax.Array:  # pragma: no cover
        raise NotImplementedError

    def _support(self) -> tuple[float, float]:  # pragma: no cover
        raise NotImplementedError

    def bounds(self) -> tuple[float, float]:
        """Closed bounds ``(lo, hi)`` of the samples, as mantissas in :attr:`unit`.

        Combines the distribution's support with the clipping bounds; ``-inf`` /
        ``inf`` mean unbounded. Used by connect-time sign checks.
        """
        lo, hi = self._support()
        if self.low is not None:
            lo = max(lo, self.low)
        if self.high is not None:
            hi = min(hi, self.high)
        return lo, hi

    def sample(self, key: jax.Array, shape: tuple = (), dtype: Any = None) -> Any:
        """Draw samples.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        shape : tuple of int, default ()
            Output shape (``(0,)`` is allowed).
        dtype : dtype, optional
            Float dtype of the mantissa; defaults to JAX's default float.

        Returns
        -------
        brainunit.Quantity or jax.Array
            Samples in :attr:`unit` (plain array when dimensionless), clipped to
            ``[low, high]``.
        """
        if dtype is None:
            dtype = jnp.zeros((), dtype=float).dtype
        m = self._sample_mantissa(key, tuple(shape), dtype)
        if self.low is not None or self.high is not None:
            m = jnp.clip(m, self.low, self.high)
        return from_mantissa(m.astype(dtype), self.unit)


@dataclass(frozen=True, init=False)
class Normal(Distribution):
    """Normal distribution ``mean + std * N(0, 1)``, optionally clipped.

    Parameters
    ----------
    mean : float or brainunit.Quantity
        Mean; its unit becomes the distribution's unit.
    std : float or brainunit.Quantity
        Standard deviation, same dimension as ``mean`` (converted to its unit).
    low, high : float or brainunit.Quantity, optional
        Clipping bounds with the same dimension as ``mean``.

    Raises
    ------
    UnitMismatchError
        If ``std`` / ``low`` / ``high`` do not match the dimension of ``mean``.
    ValueError
        If ``std < 0`` or ``low > high``.
    """

    mean: float
    std: float
    unit: u.Unit
    low: float | None
    high: float | None

    def __init__(self, mean: Any, std: Any, *, low: Any = None, high: Any = None) -> None:
        unit = unit_of(mean)
        check_unit(std, unit, 'std')
        object.__setattr__(self, 'unit', unit)
        object.__setattr__(self, 'mean', float(np.asarray(to_mantissa(mean, unit))))
        object.__setattr__(self, 'std', float(np.asarray(to_mantissa(std, unit))))
        if self.std < 0:
            raise ValueError(f'std must be non-negative, got {std}')
        self._init_clip(low, high)

    def _sample_mantissa(self, key, shape, dtype):
        return self.mean + self.std * jax.random.normal(key, shape, dtype)

    def _support(self):
        if self.std == 0:
            return self.mean, self.mean
        return -np.inf, np.inf


@dataclass(frozen=True, init=False)
class Uniform(Distribution):
    """Uniform distribution on ``[low, high)``.

    Parameters
    ----------
    low, high : float or brainunit.Quantity
        Bounds with the same dimension; the unit of ``low`` is the distribution's
        unit.

    Raises
    ------
    UnitMismatchError
        If ``high`` does not match the dimension of ``low``.
    ValueError
        If ``low > high``.
    """

    unit: u.Unit
    low: float | None
    high: float | None

    def __init__(self, low: Any, high: Any) -> None:
        object.__setattr__(self, 'unit', unit_of(low))
        self._init_clip(low, high)

    def _sample_mantissa(self, key, shape, dtype):
        return jax.random.uniform(key, shape, dtype, minval=self.low, maxval=self.high)

    def _support(self):
        return self.low, self.high


@dataclass(frozen=True, init=False)
class LogNormal(Distribution):
    """Log-normal distribution ``exp(mu + sigma * N(0, 1))`` in ``unit``.

    Parameters
    ----------
    mu, sigma : float
        Mean and standard deviation of the underlying normal (dimensionless,
        log-space of the mantissa in ``unit``).
    unit : brainunit.Unit, optional
        Unit of the samples; ``None`` means dimensionless.
    low, high : float or brainunit.Quantity, optional
        Clipping bounds with the dimension of ``unit``.

    Raises
    ------
    UnitMismatchError
        If ``mu`` / ``sigma`` carry units, or the clip bounds do not match ``unit``.
    ValueError
        If ``sigma < 0`` or ``low > high``.
    """

    mu: float
    sigma: float
    unit: u.Unit
    low: float | None
    high: float | None

    def __init__(self, mu: Any, sigma: Any, *, unit: Any = None,
                 low: Any = None, high: Any = None) -> None:
        check_unit(mu, None, 'mu (log-space, must be dimensionless)')
        check_unit(sigma, None, 'sigma (log-space, must be dimensionless)')
        object.__setattr__(self, 'unit', as_unit(unit))
        object.__setattr__(self, 'mu', float(np.asarray(mu)))
        object.__setattr__(self, 'sigma', float(np.asarray(sigma)))
        if self.sigma < 0:
            raise ValueError(f'sigma must be non-negative, got {sigma}')
        self._init_clip(low, high)

    def _sample_mantissa(self, key, shape, dtype):
        return jnp.exp(self.mu + self.sigma * jax.random.normal(key, shape, dtype))

    def _support(self):
        return 0.0, np.inf


# ---------------------------------------------------------------------------
# States holding parameters
# ---------------------------------------------------------------------------

class ConstParamState(brainstate.State):
    """A non-trainable model parameter held as a traced ``State``.

    Distinct from :class:`brainstate.ParamState` so that gradient collection
    (``module.states(brainstate.ParamState)``) skips it, while jit/``for_loop``
    still treat its value as a traced input: changing ``.value`` between calls
    does not trigger a recompile.
    """


def name_key(key: jax.Array, name: str) -> jax.Array:
    """Derive an independent key for ``name`` from ``key``.

    Uses ``fold_in`` with a CRC32 of the name, so adding or reordering other
    declarations does not change the stream of ``name``.
    """
    return jax.random.fold_in(key, zlib.crc32(name.encode()) & 0x7FFFFFFF)


def _resolve_value(value: Any, unit: u.Unit, shape: tuple, key: jax.Array | None,
                   dtype: Any, what: str) -> Any:
    """Turn a value or distribution into an array/Quantity of ``shape`` in ``unit``."""
    if isinstance(value, Distribution):
        check_unit(value.unit, unit, what)
        if key is None:
            raise ValueError(f'{what} is a distribution; an RNG key is required to sample it')
        floating = jnp.issubdtype(np.dtype(dtype), jnp.inexact)
        value = value.sample(key, shape, dtype if floating else None)
    check_unit(value, unit, what)
    m = to_mantissa(value, unit, dtype)
    if m.shape != shape:
        if m.ndim == 0:
            m = jnp.broadcast_to(m, shape)
        else:
            raise ValueError(f'{what}: expected a scalar or shape {shape}, got shape {m.shape}')
    return from_mantissa(m, unit)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

_MISSING = object()


@dataclass(frozen=True, init=False)
class Param:
    """Declaration of one model parameter.

    Parameters
    ----------
    default : scalar, array, brainunit.Quantity or Distribution, optional
        Default value. ``None`` makes the parameter required (it must then be
        given at materialisation, unless ``dist`` is set).
    unit : brainunit.Unit, optional
        Unit the value is stored in. Inferred from ``default`` (or ``dist``) when
        omitted; ``None`` everywhere means dimensionless.
    trainable : bool, default False
        Store as :class:`brainstate.ParamState` (trainable) instead of
        :class:`ConstParamState`.
    per_neuron : bool, default True
        Shape ``(n,)`` per neuron (heterogeneity is the default). If False the
        value is shared and keeps its own shape (scalar for distributions).
    dist : Distribution, optional
        Default distribution; when set it is used instead of ``default`` if no
        override is given (``default`` then documents the nominal value).

    Raises
    ------
    UnitMismatchError
        If ``default`` / ``dist`` do not match ``unit``.
    ValueError
        If both ``default`` and ``dist`` are ``None`` and ``unit`` is not given
        (the unit could not be inferred).
    """

    default: Any
    unit: u.Unit
    trainable: bool
    per_neuron: bool
    dist: Distribution | None

    def __init__(self, default: Any = None, unit: Any = None, *, trainable: bool = False,
                 per_neuron: bool = True, dist: Distribution | None = None) -> None:
        if isinstance(default, Distribution):
            default, dist = None, (dist if dist is not None else default)
        if unit is None:
            if default is not None:
                unit = unit_of(default)
            elif dist is not None:
                unit = dist.unit
        unit = as_unit(unit)
        if default is not None:
            check_unit(default, unit, 'Param default')
        if dist is not None:
            if not isinstance(dist, Distribution):
                raise TypeError(f'dist must be a Distribution, got {type(dist).__name__}')
            check_unit(dist.unit, unit, 'Param dist')
        object.__setattr__(self, 'default', default)
        object.__setattr__(self, 'unit', unit)
        object.__setattr__(self, 'trainable', bool(trainable))
        object.__setattr__(self, 'per_neuron', bool(per_neuron))
        object.__setattr__(self, 'dist', dist)

    def materialize(self, n: int, key: jax.Array | None = None, value: Any = _MISSING,
                    dtype: Any = np.float32, name: str = 'param') -> Any:
        """Concrete value for a population of ``n`` neurons.

        Parameters
        ----------
        n : int
            Population size (``0`` allowed).
        key : jax.Array, optional
            Key used when the value is a distribution.
        value : optional
            Override (scalar, array, Quantity or Distribution). Defaults to
            ``dist`` if set, else ``default``.
        dtype : dtype, default float32
            Float dtype of the stored mantissa.
        name : str
            Used in error messages.

        Returns
        -------
        brainunit.Quantity or jax.Array
            Shape ``(n,)`` when ``per_neuron``; in :attr:`unit`.
        """
        if value is _MISSING:
            value = self.dist if self.dist is not None else self.default
        if value is None:
            raise ValueError(f'parameter {name!r} is required but no value was given')
        if self.per_neuron:
            shape = (int(n),)
        elif isinstance(value, Distribution):
            shape = ()
        else:
            shape = tuple(np.shape(u.get_mantissa(value)))
        return _resolve_value(value, self.unit, shape, key, dtype, f'parameter {name!r}')


class Params(Mapping):
    """Ordered collection of :class:`Param` declarations.

    Parameters
    ----------
    **params : Param or value
        Declarations. A bare value ``v`` is shorthand for ``Param(v)``.
    """

    def __init__(self, **params: Any) -> None:
        self._decl = {k: v if isinstance(v, Param) else Param(v) for k, v in params.items()}

    def __getitem__(self, name: str) -> Param:
        return self._decl[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decl)

    def __len__(self) -> int:
        return len(self._decl)

    def __repr__(self) -> str:
        return f'Params({", ".join(self._decl)})'

    def extend(self, **params: Any) -> 'Params':
        """New :class:`Params` with ``params`` added or replacing existing ones."""
        return Params(**{**self._decl, **params})

    def materialize(self, n: int, key: jax.Array | None = None,
                    overrides: Mapping[str, Any] | None = None,
                    precision: Any = 'float32',
                    trainable: Iterable[str] = ()) -> dict[str, brainstate.State]:
        """Create one ``State`` per parameter for ``n`` neurons.

        Parameters
        ----------
        n : int
            Population size (``0`` allowed).
        key : jax.Array, optional
            Root key; parameter ``name`` uses ``name_key(key, name)``.
        overrides : mapping, optional
            Per-parameter values replacing the declared defaults.
        precision : {'float32', 'float64'}, default 'float32'
            Float tier of the stored values.
        trainable : iterable of str, default ()
            Extra names to store as ``ParamState`` regardless of the declaration.

        Returns
        -------
        dict of str to brainstate.State
            ``ParamState`` for trainable parameters, ``ConstParamState`` otherwise,
            in declaration order.

        Raises
        ------
        KeyError
            If ``overrides`` or ``trainable`` name an undeclared parameter.
        """
        overrides = dict(overrides or {})
        trainable = set(trainable)
        unknown = (set(overrides) | trainable) - set(self._decl)
        if unknown:
            raise KeyError(f'unknown parameter(s) {sorted(unknown)}; declared: {list(self._decl)}')
        dtype = resolve_dtype(precision)
        out = {}
        for name, decl in self._decl.items():
            k = None if key is None else name_key(key, name)
            val = decl.materialize(n, k, overrides.get(name, _MISSING), dtype, name)
            cls = brainstate.ParamState if (decl.trainable or name in trainable) else ConstParamState
            out[name] = cls(val)
        return out


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class Init:
    """Declaration of the initial value of one per-neuron state.

    Parameters
    ----------
    value : scalar, array, Quantity, Distribution or callable
        Initial value. A callable receives the parameter :class:`Record` and
        returns the value (e.g. ``Init(lambda p: p.E_L)``).
    unit : brainunit.Unit, optional
        Storage unit. Inferred from ``value`` when omitted; for callables it is
        inferred from the returned value at materialisation.
    dtype : dtype, optional
        Storage dtype. ``None`` means the float tier of the model; integer/bool
        dtypes make a non-differentiable ``ShortTermState``.
    """

    value: Any
    unit: u.Unit | None
    dtype: Any

    def __init__(self, value: Any, unit: Any = None, dtype: Any = None) -> None:
        if unit is not None:
            unit = as_unit(unit)
        elif isinstance(value, Distribution):
            unit = value.unit
        elif not callable(value):
            unit = unit_of(value)
        if unit is not None and not callable(value):
            check_unit(value.unit if isinstance(value, Distribution) else value, unit, 'Init value')
        object.__setattr__(self, 'value', value)
        object.__setattr__(self, 'unit', unit)
        object.__setattr__(self, 'dtype', None if dtype is None else np.dtype(dtype))

    def materialize(self, n: int, key: jax.Array | None = None, params: Record | None = None,
                    value: Any = _MISSING, dtype: Any = np.float32, name: str = 'state') -> Any:
        """Concrete ``(n,)`` initial value.

        Parameters
        ----------
        n : int
            Population size (``0`` allowed).
        key : jax.Array, optional
            Key used when the value is a distribution.
        params : Record, optional
            Parameter values passed to callable initialisers.
        value : optional
            Override for the declared value (same forms as ``Init.value``).
        dtype : dtype, default float32
            Float tier; ignored when :attr:`dtype` is set.
        name : str
            Used in error messages.

        Returns
        -------
        brainunit.Quantity or jax.Array
            Shape ``(n,)``.
        """
        if value is _MISSING:
            value = self.value
        if callable(value) and not isinstance(value, Distribution):
            if params is None:
                raise ValueError(f'state {name!r} has a callable initialiser; params are required')
            value = value(params)
        unit = self.unit if self.unit is not None else (
            value.unit if isinstance(value, Distribution) else unit_of(value))
        dtype = self.dtype if self.dtype is not None else dtype
        return _resolve_value(value, unit, (int(n),), key, dtype, f'state {name!r}')


class States(Mapping):
    """Ordered collection of :class:`Init` declarations.

    Parameters
    ----------
    **inits : Init or value
        Declarations. A bare value ``v`` is shorthand for ``Init(v)``.
    """

    def __init__(self, **inits: Any) -> None:
        self._decl = {k: v if isinstance(v, Init) else Init(v) for k, v in inits.items()}

    def __getitem__(self, name: str) -> Init:
        return self._decl[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decl)

    def __len__(self) -> int:
        return len(self._decl)

    def __repr__(self) -> str:
        return f'States({", ".join(self._decl)})'

    def extend(self, **inits: Any) -> 'States':
        """New :class:`States` with ``inits`` added or replacing existing ones."""
        return States(**{**self._decl, **inits})

    def materialize(self, n: int, key: jax.Array | None = None, params: Record | None = None,
                    overrides: Mapping[str, Any] | None = None,
                    precision: Any = 'float32') -> dict[str, brainstate.State]:
        """Create one ``State`` per declared state for ``n`` neurons.

        Parameters
        ----------
        n : int
            Population size (``0`` allowed).
        key : jax.Array, optional
            Root key; state ``name`` uses ``name_key(key, name)``.
        params : Record, optional
            Parameter values for callable initialisers (see :func:`read_states`).
        overrides : mapping, optional
            Per-state initial values replacing the declarations.
        precision : {'float32', 'float64'}, default 'float32'
            Float tier.

        Returns
        -------
        dict of str to brainstate.State
            ``HiddenState`` for float states, ``ShortTermState`` otherwise.

        Raises
        ------
        KeyError
            If ``overrides`` names an undeclared state.
        """
        overrides = dict(overrides or {})
        unknown = set(overrides) - set(self._decl)
        if unknown:
            raise KeyError(f'unknown state(s) {sorted(unknown)}; declared: {list(self._decl)}')
        dtype = resolve_dtype(precision)
        out = {}
        for name, decl in self._decl.items():
            k = None if key is None else name_key(key, name)
            val = decl.materialize(n, k, params, overrides.get(name, _MISSING), dtype, name)
            floating = jnp.issubdtype(u.get_mantissa(val).dtype, jnp.floating)
            out[name] = (brainstate.HiddenState if floating else brainstate.ShortTermState)(val)
        return out


# ---------------------------------------------------------------------------
# Reading / writing State dicts as Records
# ---------------------------------------------------------------------------

def read_states(states: Mapping[str, brainstate.State]) -> Record:
    """Gather ``State`` values into a :class:`Record` (same names and order)."""
    return Record(**{k: s.value for k, s in states.items()})


def write_states(states: Mapping[str, brainstate.State], values: Mapping[str, Any]) -> None:
    """Write ``values`` (e.g. a :class:`Record`) back into ``states``.

    Each value is converted to the unit and dtype of the current state value and
    must have the same shape (or be a scalar, which is broadcast). A subset of the
    names may be written.

    Raises
    ------
    KeyError
        If ``values`` contains a name not in ``states``.
    UnitMismatchError
        If a value's dimension differs from the state's.
    ValueError
        If a non-scalar value has the wrong shape.
    """
    unknown = set(values) - set(states)
    if unknown:
        raise KeyError(f'unknown state(s) {sorted(unknown)}; known: {list(states)}')
    for name, v in values.items():
        st = states[name]
        old = st.value
        unit = unit_of(old)
        dtype = u.get_mantissa(old).dtype
        st.value = _resolve_value(v, unit, tuple(np.shape(u.get_mantissa(old))), None, dtype,
                                  f'state {name!r}')

