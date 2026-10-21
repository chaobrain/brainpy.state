# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""The ``NeuronModel`` contract, its generated ``step``, and the ``Device`` base.

A model *declares* (class attributes, K0 spec §4.3):

``params``
    :class:`~brainpy_state._kernel.params.Params`.
``states``
    :class:`~brainpy_state._kernel.params.States`. Because
    ``brainstate.nn.Module.states()`` is a method, the declaration written as
    ``states = States(...)`` in a class body is moved to :attr:`NeuronModel.state_decl`
    at class creation, keeping ``model.states(...)`` usable.
``inputs`` / ``outputs``
    :class:`~brainpy_state._kernel.ports.Ports`.
``spiking``
    :class:`~brainpy_state._kernel.spiking.SpikeRule` or ``None``.
``integrator``
    :class:`~brainpy_state._kernel.integrators.ExpEuler`, ``RK(order, substeps)``
    or ``Exact()`` (then ``step`` must be overridden).
``capabilities``
    :class:`Capabilities`.

and implements ``derivatives(s, p, inp, ctx) -> dict`` (continuous part) and
optionally ``on_input(s, p, inp, ctx) -> s`` (event jumps).

Generated step (architecture doc §7.1)
--------------------------------------
1. **latch**: ``current`` / ``graded`` ports act with the value latched at the
   previous step (state ``latch_<port>``); this step's value is stored for the
   next one (NEST ring-buffer / ``y0`` semantics, a one-step delay).
2. **integrate** ``derivatives`` with the model's integrator. Port inputs are
   computed once, outside the unrolled substeps, and are constant across them;
   ``conductance`` ports are passed as-is, so a gate ``gate(V)`` evaluated inside
   ``derivatives`` sees the substep's ``V``.
3. after **every substep**: refractory clamp, spike condition, jump over all
   states, refractory counter, spike **count** (:mod:`.spiking`).
4. **event** jumps (``on_input``) after the continuous propagation.
5. outputs are read from the **new** state values (the spike output is a float
   ``HiddenState``), which braintrace requires.

Additional states are created automatically: ``latch_<port>`` (float,
``HiddenState``) per current/graded input, and the spike rule's output
(``HiddenState``) and refractory counter (int32 ``ShortTermState``).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.integrators import Exact, Integrator, RK, integrate
from brainpy_state._kernel.params import (
    Init,
    Params,
    Record,
    States,
    name_key,
    read_states,
    write_states,
)
from brainpy_state._kernel.ports import (
    Conductance,
    Current,
    Graded,
    PortError,
    Ports,
    TimedEvent,
)
from brainpy_state._kernel.spiking import SpikeRule
from brainpy_state._kernel.units import (
    as_unit,
    check_unit,
    from_mantissa,
    resolve_dtype,
    to_mantissa,
    unit_of,
)

__all__ = [
    'Capabilities',
    'NeuronModel',
    'Device',
    'conductance_current',
    'latch_name',
]

_LATCHED = (Current, Graded)


def latch_name(port: str) -> str:
    """Name of the state latching the ``current`` / ``graded`` input ``port``."""
    return f'latch_{port}'


def _keep_dependency(new: Any, old: Any) -> Any:
    # braintrace 0.2.6 fails (KeyError in the hidden-group transition) when a
    # HiddenState's new value depends only on inputs; ``+ 0 * old`` keeps the
    # hidden->hidden edge in the jaxpr without changing the value.
    return new + 0 * old


def conductance_current(port: Conductance, g: Any, V: Any, p: Record) -> Any:
    r"""Synaptic current of a conductance port: :math:`g\,\mathrm{gate}(V)\,(E_{rev} - V)`.

    Call it inside ``derivatives`` so the gate and the driving force use the
    substep's ``V``.

    Parameters
    ----------
    port : Conductance
        Port declaration (``E_rev`` as a Quantity or a parameter name, optional
        ``gate``).
    g : brainunit.Quantity
        Conductance payload of the port.
    V : brainunit.Quantity
        Membrane potential.
    p : Record
        Parameter values (to resolve ``E_rev`` given by name).

    Returns
    -------
    brainunit.Quantity
        Current (positive = depolarising).

    Raises
    ------
    PortError
        If the port declares no ``E_rev``.
    """
    if port.E_rev is None:
        raise PortError('conductance port has no E_rev; compute the driving force in derivatives')
    E = p[port.E_rev] if isinstance(port.E_rev, str) else port.E_rev
    if port.gate is not None:
        g = g * port.gate(V)
    return g * (E - V)


@dataclass(frozen=True)
class Capabilities:
    """What a model supports; checked by compilers and learners.

    Parameters
    ----------
    jit_safe : bool, default True
        Runs under ``jit`` (host-only models set False -> compile error).
    differentiable : bool, default True
        Reverse-mode differentiable step (surrogate BPTT).
    etrace : bool, default True
        Compatible with braintrace online learning (weights through ETP
        primitives, outputs read from ``HiddenState``).
    precise_spike_times : bool, default False
        Emits/consumes in-step spike offsets (``timed_event``).
    """

    jit_safe: bool = True
    differentiable: bool = True
    etrace: bool = True
    precise_spike_times: bool = False


class NeuronModel(brainstate.nn.Module):
    """Base class of neuron models (and, via :class:`Device`, of devices).

    Parameters
    ----------
    n : int
        Population size (``0`` allowed).
    params : mapping, optional
        Parameter overrides (scalars, ``(n,)`` arrays, Quantities or
        distributions).
    init : mapping, optional
        Initial-value overrides of the declared states.
    precision : {'float32', 'float64'}, default 'float32'
        Float tier.
    key : jax.Array, optional
        Root key for distributions (parameters and initial values).
    trainable : iterable of str, default ()
        Parameters to store as ``brainstate.ParamState`` regardless of the
        declaration.
    integrator : Integrator, optional
        Override of the class integrator (e.g. more substeps).
    detach_reset : bool, optional
        Default for the spike rule's ``detach_reset`` (set by learners).
    name : str, optional
        Module name.

    Attributes
    ----------
    param_states : dict of str to brainstate.State
        One ``ParamState`` / ``ConstParamState`` per parameter.
    dyn_states : dict of str to brainstate.State
        Declared states plus latches, spike output and refractory counter.

    Raises
    ------
    TypeError
        At class creation, if the declaration is inconsistent (``Exact()``
        without an overridden ``step``, spike output not an output port, ...).
    """

    params: ClassVar[Params] = Params()
    state_decl: ClassVar[States] = States()
    inputs: ClassVar[Ports] = Ports()
    outputs: ClassVar[Ports] = Ports()
    spiking: ClassVar[SpikeRule | None] = None
    integrator: ClassVar[Integrator] = RK(order=4, substeps=1)
    capabilities: ClassVar[Capabilities] = Capabilities()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        decl = cls.__dict__.get('states')
        if isinstance(decl, States):
            cls.state_decl = decl
            delattr(cls, 'states')  # restore brainstate's Module.states() method
        elif decl is not None and not callable(decl):
            raise TypeError(f'{cls.__name__}.states must be a States declaration')
        cls._validate_declaration()

    @classmethod
    def _validate_declaration(cls, integrator: Integrator | None = None) -> None:
        name = cls.__name__
        for attr, typ in (('params', Params), ('state_decl', States), ('inputs', Ports),
                          ('outputs', Ports)):
            if not isinstance(getattr(cls, attr), typ):
                raise TypeError(f'{name}.{attr} must be a {typ.__name__}')
        integ = cls.integrator if integrator is None else integrator
        if not isinstance(integ, Integrator):
            raise TypeError(f'{name}.integrator must be an Integrator, got {type(integ).__name__}')
        if integ.exact and cls.step is NeuronModel.step:
            raise TypeError(f'{name} declares Exact() but does not override step(); exact '
                            f'integration needs hand-written propagators')
        rule = cls.spiking
        if rule is not None:
            if not isinstance(rule, SpikeRule):
                raise TypeError(f'{name}.spiking must be a SpikeRule or None')
            if rule.output not in cls.outputs:
                raise TypeError(f'{name}: spike output {rule.output!r} is not a declared output '
                                f'port (outputs: {list(cls.outputs)})')
            extra = set(rule.state_decl()) & set(cls.state_decl)
            if extra:
                raise TypeError(f'{name}: states {sorted(extra)} clash with the spike rule')
            if rule.clamp_var is not None and rule.clamp_var not in cls.state_decl:
                raise TypeError(f'{name}: Clamp var {rule.clamp_var!r} is not a declared state')
        latches = {latch_name(k) for k, v in cls.inputs.items() if isinstance(v, _LATCHED)}
        clash = latches & set(cls.state_decl)
        if clash:
            raise TypeError(f'{name}: states {sorted(clash)} clash with input latches')

    # ---------------------------------------------------------------- construction

    def __init__(self, n: int, params: Mapping[str, Any] | None = None,
                 init: Mapping[str, Any] | None = None, *, precision: Any = 'float32',
                 key: jax.Array | None = None, trainable: Any = (),
                 integrator: Integrator | None = None, detach_reset: bool | None = None,
                 name: str | None = None) -> None:
        super().__init__(name=name)
        if integrator is not None:
            type(self)._validate_declaration(integrator)
            self.integrator = integrator
        self.n = int(n)
        if self.n < 0:
            raise ValueError(f'n must be non-negative, got {n}')
        self.precision = resolve_dtype(precision).name
        self.detach_reset = detach_reset
        self._key = key
        self._init_overrides = dict(init or {})
        self.param_states = self.params.materialize(
            self.n, None if key is None else name_key(key, 'params'), overrides=params,
            precision=self.precision, trainable=trainable)
        self.init_state()

    @classmethod
    def all_state_decl(cls) -> States:
        """Declared states plus input latches and the spike rule's states."""
        extra = {latch_name(k): Init(from_mantissa(0.0, v.unit))
                 for k, v in cls.inputs.items() if isinstance(v, _LATCHED)}
        if cls.spiking is not None:
            extra.update(dict(cls.spiking.state_decl()))
        return cls.state_decl.extend(**extra)

    def init_state(self, batch_size: int | None = None, **kwargs: Any) -> None:
        """(Re)create the dynamical states from their declarations.

        Batching is done with ``vmap`` over the state pytree, so ``batch_size``
        must be ``None``.
        """
        if batch_size is not None:
            raise NotImplementedError('NeuronModel states are unbatched; batch with vmap')
        key = None if self._key is None else name_key(self._key, 'states')
        self.dyn_states = self.all_state_decl().materialize(
            self.n, key, params=read_states(self.param_states),
            overrides=self._init_overrides, precision=self.precision)

    def reset_state(self, batch_size: int | None = None, **kwargs: Any) -> None:
        """Reset the dynamical states to their initial values."""
        self.init_state(batch_size, **kwargs)

    # ---------------------------------------------------------------- contract

    def derivatives(self, s: Record, p: Record, inp: Record, ctx: StepContext) -> Mapping[str, Any]:
        """Time derivatives of the continuous states.

        Parameters
        ----------
        s, p : Record
            State and parameter values (Quantities).
        inp : Record
            Port inputs: latched values for ``current``/``graded`` ports, this
            step's payload for the others.
        ctx : StepContext
            Context with ``t`` set to the evaluation time inside the step.

        Returns
        -------
        mapping of str to Quantity
            ``d state / dt`` per float state (unit ``state unit / time``);
            states omitted are constant.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement derivatives()')

    def on_input(self, s: Record, p: Record, inp: Record, ctx: StepContext) -> Record:
        """Event jumps applied after the step's propagation (default: none)."""
        return s

    def output_states(self) -> dict[str, str]:
        """Output port -> state name it is read from (default: the spike output)."""
        if self.spiking is None:
            return {}
        return {self.spiking.output: self.spiking.output}

    def float_state_names(self) -> list[str]:
        """Declared float states integrated by the generated step."""
        decl = self.state_decl
        return [k for k in decl
                if jnp.issubdtype(u.get_mantissa(self.dyn_states[k].value).dtype, jnp.floating)]

    def prepare_inputs(self, inputs: Mapping[str, Any] | None = None) -> Record:
        """Complete port inputs: zeros for missing ports, scalars broadcast to ``(n,)``.

        Raises
        ------
        PortError
            On an undeclared port name.
        UnitMismatchError
            On a payload with the wrong dimension.
        """
        inputs = dict(inputs or {})
        unknown = set(inputs) - set(self.inputs)
        if unknown:
            raise PortError(f'{type(self).__name__}: unknown input port(s) {sorted(unknown)}; '
                            f'declared: {list(self.inputs)}')
        dtype = resolve_dtype(self.precision)
        out = {}
        for k, kind in self.inputs.items():
            if k not in inputs:
                out[k] = kind.zeros(self.n, dtype)
            elif isinstance(kind, TimedEvent):
                out[k] = inputs[k]
            else:
                check_unit(inputs[k], kind.unit, f'input {k!r}')
                m = jnp.broadcast_to(to_mantissa(inputs[k], kind.unit, dtype), (self.n,))
                out[k] = from_mantissa(m, kind.unit)
        return Record(**out)

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Advance one step (generated from the declaration; see module docstring).

        Parameters
        ----------
        s, p : Record
            State and parameter values.
        inp : mapping
            Complete port inputs (see :meth:`prepare_inputs`).
        ctx : StepContext
            Step context (``t`` at the step start, ``dt``).

        Returns
        -------
        s : Record
            New state values.
        outputs : dict
            Output port values, read from the new states.
        """
        inp = inp if isinstance(inp, Record) else Record(**inp)
        rule = self.spiking
        integ = self.integrator

        # 1. latch: derivatives see the previous step's current/graded values
        latched = {k: s[latch_name(k)] for k, v in self.inputs.items() if isinstance(v, _LATCHED)}
        inp_ode = Record(**{**inp, **latched})
        new_latches = {latch_name(k): _keep_dependency(inp[k], s[latch_name(k)]) for k in latched}

        # 2. integrate the float states on mantissas
        names = self.float_state_names()
        units = {k: unit_of(s[k]) for k in names}
        dt_ms = to_mantissa(ctx.dt, u.ms)
        t0_ms = to_mantissa(ctx.t, u.ms)
        detach = rule.resolve_detach(self.detach_reset) if rule is not None else False
        carry = rule.begin(s, p, dt_ms, integ.substeps) if rule is not None else None
        clamp_var = rule.clamp_var if rule is not None else None
        n_sub = integ.substeps

        def to_record(y, base):
            return base.replace(**{k: from_mantissa(y[k], units[k]) for k in names})

        def f(t, y, c):
            st = to_record(y, s)
            d = self.derivatives(st, p, inp_ode, ctx.replace(t=from_mantissa(t, u.ms)))
            unknown = set(d) - set(names)
            if unknown:
                raise KeyError(f'{type(self).__name__}.derivatives returned non-integrated '
                               f'states {sorted(unknown)}; integrated: {names}')
            dy = {}
            for k in names:
                if k in d:
                    dy[k] = jnp.broadcast_to(
                        to_mantissa(d[k], as_unit(units[k]) / u.ms, y[k].dtype), y[k].shape)
                else:
                    dy[k] = jnp.zeros_like(y[k])
            if clamp_var is not None:
                dy[clamp_var] = jnp.where(rule.refractory_mask(c), 0, dy[clamp_var])
            return dy

        def hook(k, t, y, c):
            if rule is None:
                return y, c
            st = to_record(y, s_holder[0])
            st = rule.clamp(st, p, c)
            c = rule.tick(c, 'substep')
            if rule.when == 'substep' or k == n_sub - 1:
                st, c = rule.check(st, p, c, detach_reset=detach)
            s_holder[0] = st  # non-float states changed by jumps
            return {kk: to_mantissa(st[kk], units[kk]) for kk in names}, c

        s_holder = [s]
        y0 = {k: to_mantissa(s[k], units[k]) for k in names}
        y, carry = integrate(integ, f, y0, t0_ms, dt_ms, hook=hook, aux=carry)
        s = to_record(y, s_holder[0])

        # 3. close the step: spike output and refractory counter
        if rule is not None:
            spike, counter = rule.end(carry)
            changes = {rule.output: spike}
            if counter is not None:
                changes[rule.counter] = counter
            s = s.replace(**changes)

        # 4. event jumps after propagation
        s = self.on_input(s, p, inp, ctx)
        s = s.replace(**new_latches)

        # 5. outputs from the new state values
        return s, {port: s[name] for port, name in self.output_states().items()}

    # ---------------------------------------------------------------- stateful API

    def read(self) -> tuple[Record, Record]:
        """Current ``(state, params)`` values as Records."""
        return read_states(self.dyn_states), read_states(self.param_states)

    def update(self, ctx: StepContext, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Stateful step: read states, :meth:`step`, write states, return outputs.

        Outputs listed by :meth:`output_states` are read back from the written
        ``State`` objects (braintrace needs outputs read from a ``HiddenState``).

        Parameters
        ----------
        ctx : StepContext
            Step context.
        inputs : mapping, optional
            Port payloads (missing ports receive zeros).

        Returns
        -------
        dict
            Output port values.
        """
        s, p = self.read()
        s_new, out = self.step(s, p, self.prepare_inputs(inputs), ctx)
        write_states(self.dyn_states, s_new)
        out = dict(out)
        for port, name in self.output_states().items():
            out[port] = self.dyn_states[name].value
        return out


class Device(NeuronModel):
    """Base of stimulation devices: populations with outputs and no inputs.

    Subclasses declare ``outputs`` (``event``, ``current``, ``graded``) and
    override :meth:`step` (``Exact()`` integrator); devices consume randomness
    only through ``ctx.key``. Default capabilities: not differentiable.

    Raises
    ------
    TypeError
        At class creation, if ``inputs`` is not empty.
    """

    integrator: ClassVar[Integrator] = Exact()
    capabilities: ClassVar[Capabilities] = Capabilities(differentiable=False, etrace=False)

    @classmethod
    def _validate_declaration(cls, integrator: Integrator | None = None) -> None:
        if len(cls.inputs):
            raise TypeError(f'{cls.__name__}: devices have no inputs, got {list(cls.inputs)}')
        if getattr(cls.step, '_device_placeholder', False) and 'step' not in cls.__dict__:
            raise TypeError(f'{cls.__name__}: devices must override step()')
        if cls.spiking is not None and cls.spiking.output not in cls.outputs:
            raise TypeError(f'{cls.__name__}: spike output {cls.spiking.output!r} is not an output port')

    def step(self, s: Record, p: Record, inp: Mapping[str, Any], ctx: StepContext
             ) -> tuple[Record, dict[str, Any]]:
        """Emit this step's outputs (to be implemented by subclasses)."""
        raise NotImplementedError(f'{type(self).__name__} must implement step()')

    step._device_placeholder = True
