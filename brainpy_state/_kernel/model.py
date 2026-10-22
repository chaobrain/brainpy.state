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

Receptor ports (decision D-b): an input ``Event(unit=..., into='g_ex')`` names
the declared state the neuron filters it into. The state must exist and have the
port unit's dimension (checked at class creation). The default ``on_input`` adds
each receptor port's payload to its state (``g_ex += w``); models whose kinetics
take the weight differently (alpha kernels: ``dg_ex += e / tau * w``) override
``on_input`` (or ``step``).

Generated step (architecture doc §7.1)
--------------------------------------
1. **latch** (per port, decision D-a): ``current`` / ``graded`` ports declared
   ``latch=True`` (the default) act with the value latched at the previous step
   (state ``latch_<port>``); this step's value is stored for the next one (NEST
   ring-buffer / ``y0`` semantics, a one-step delay). Ports declared
   ``latch=False`` act with this step's value (constant over the step's
   substeps). A latched port's payload of step ``n`` must not influence step
   ``n``'s update -- only the stored latch; the runner relies on it to deliver
   latched inputs late (zero-delay cycles through latched ports).
2. **integrate** ``derivatives`` with the model's integrator. Port inputs are
   computed once, outside the substep ``scan``, and are constant across them;
   ``conductance`` ports are passed as-is, so a gate ``gate(V)`` evaluated inside
   ``derivatives`` sees the substep's ``V``.
3. after **every substep**: refractory clamp, spike condition, jump over all
   states, refractory counter, spike **count** (:mod:`.spiking`).
4. **event** jumps (``on_input``; by default the receptor-port jumps) after the
   continuous propagation.
5. outputs are read from the **new** state values (the spike output is a float
   ``HiddenState``), which braintrace requires.

Additional states are created automatically: ``latch_<port>`` (float,
``HiddenState``) per latched current/graded input, and the spike rule's output
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
    Event,
    PortError,
    Ports,
    TimedEvent,
    is_latched,
    is_receptor,
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
    'latch_inputs',
    'latch_name',
    'latched_ports',
    'with_conductance_ports',
]

def latch_name(port: str) -> str:
    """Name of the state latching the ``current`` / ``graded`` input ``port``."""
    return f'latch_{port}'


def latched_ports(inputs: Ports) -> tuple[str, ...]:
    """Names of the latched input ports (``Current`` / ``Graded`` with ``latch=True``)."""
    return tuple(k for k, v in inputs.items() if is_latched(v))


def _keep_dependency(new: Any, old: Any) -> Any:
    # upstream: braintrace#184
    # braintrace 0.2.6 fails (KeyError in the hidden-group transition) when a
    # HiddenState's new value depends only on inputs; ``+ 0 * old`` keeps the
    # hidden->hidden edge in the jaxpr without changing the value.
    return new + 0 * old


def latch_inputs(inputs: Ports, s: Record, inp: Record) -> tuple[Record, dict]:
    """Port values acting this step and the new latch states (decision D-a).

    Parameters
    ----------
    inputs : Ports
        The model's input port declarations.
    s : Record
        State values (holding ``latch_<port>`` for every latched port).
    inp : Record
        This step's port payloads.

    Returns
    -------
    acting : Record
        ``inp`` with every latched port replaced by the value latched at the
        previous step; unlatched ports keep this step's payload.
    new : dict
        ``{latch_<port>: this step's payload}`` for the latched ports, to store
        (with the braintrace dependency workaround ``new + 0 * old``).
    """
    ports = latched_ports(inputs)
    acting = Record(**{**inp, **{k: s[latch_name(k)] for k in ports}})
    # upstream: braintrace#184
    new = {latch_name(k): _keep_dependency(inp[k], s[latch_name(k)]) for k in ports}
    return acting, new


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


def with_conductance_ports(model: type, name: str | None = None,
                           params: Mapping[str, Any] | None = None,
                           **ports: Conductance) -> type:
    r"""Derive a model class with extra (optionally voltage-gated) conductance ports.

    The generic receiving end of conductance delivery (K0 T8): the new ports are
    ``Conductance`` ports -- the payload is the conductance itself,
    delivered once per step and constant over the substeps -- and the model adds
    :math:`\sum_k g_k\,\mathrm{gate}_k(V)\,(E_k - V)` to its membrane equation through
    :meth:`NeuronModel.conductance_port_current`, so every gate is evaluated at the
    ``V`` of each RK substep.

    Parameters
    ----------
    model : type
        A ``NeuronModel`` subclass with ``accepts_conductance_ports = True``
        (``iaf_cond_exp``, ``aeif_cond_alpha``).
    name : str, optional
        Name of the new class (default ``'<model>_' + '_'.join(ports)``).
    params : mapping, optional
        Extra parameter declarations (e.g. ``E_nmda=0 * u.mV`` referenced by a
        port's ``E_rev``).
    **ports : Conductance
        New input ports; each must declare ``E_rev``.

    Returns
    -------
    type
        Subclass of ``model`` with the extended ``inputs`` (and ``params``).

    Raises
    ------
    TypeError
        If ``model`` does not opt in, a port is not a ``Conductance`` or no port
        is given.
    PortError
        If a port has no ``E_rev``, names an undeclared parameter or
        clashes with an existing port.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.gates import MgBlock
        >>> from brainpy_state._kernel.model import with_conductance_ports
        >>> from brainpy_state._kernel.models import iaf_cond_exp
        >>> from brainpy_state._kernel.ports import Conductance
        >>> Cell = with_conductance_ports(
        ...     iaf_cond_exp, nmda=Conductance(u.nS, E_rev='E_nmda', gate=MgBlock()),
        ...     params=dict(E_nmda=0.0 * u.mV))
        >>> list(Cell.inputs)
        ['ex', 'in_', 'I', 'nmda']
    """
    if not (isinstance(model, type) and issubclass(model, NeuronModel)):
        raise TypeError(f'model must be a NeuronModel subclass, got {model!r}')
    if not getattr(model, 'accepts_conductance_ports', False):
        raise TypeError(f'{model.__name__} does not add conductance-port currents to its membrane '
                        f'equation (accepts_conductance_ports is False)')
    if not ports:
        raise TypeError('give at least one conductance port (name=Conductance(...))')
    new_params = model.params.extend(**dict(params or {}))
    for k, kind in ports.items():
        if not isinstance(kind, Conductance):
            raise TypeError(f'port {k!r} must be a Conductance, got {type(kind).__name__}')
        if kind.E_rev is None:
            raise PortError(f'port {k!r} needs E_rev (a voltage or a parameter name)')
        if isinstance(kind.E_rev, str) and kind.E_rev not in new_params:
            raise PortError(f'port {k!r}: E_rev parameter {kind.E_rev!r} is not declared '
                            f'(declared: {list(new_params)}); add it with params=')
        if k in model.inputs:
            raise PortError(f'port {k!r} already exists on {model.__name__}')
    attrs = {'inputs': Ports(**dict(model.inputs), **ports), 'params': new_params,
             '__module__': model.__module__,
             '__doc__': f'{model.__name__} with extra conductance ports {list(ports)}.'}
    return type(name or f'{model.__name__}_{"_".join(ports)}', (model,), attrs)


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
        Default for the spike rule's ``detach_reset``; a running ``Program``'s
        ``detach_reset`` option (``StepContext.detach_reset``) takes precedence.
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
    #: The model adds :meth:`conductance_port_current` to its membrane equation, so
    #: :func:`with_conductance_ports` may extend it with (gated) conductance ports.
    accepts_conductance_ports: ClassVar[bool] = False

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
        cls._validate_receptors()
        latches = {latch_name(k) for k in latched_ports(cls.inputs)}
        clash = latches & set(cls.state_decl)
        if clash:
            raise TypeError(f'{name}: states {sorted(clash)} clash with input latches')

    @classmethod
    def _validate_receptors(cls) -> None:
        """Receptor ports (``into=``) must name a declared float state of the port's dimension."""
        for k, kind in cls.inputs.items():
            if not is_receptor(kind):
                continue
            where = f'{cls.__name__}.inputs[{k!r}]'
            if kind.into not in cls.state_decl:
                raise PortError(f'{where}: receptor port filters into {kind.into!r}, which is not a '
                                f'declared state (declared: {list(cls.state_decl)})')
            init = cls.state_decl[kind.into]
            if init.dtype is not None and not jnp.issubdtype(init.dtype, jnp.floating):
                raise PortError(f'{where}: receptor state {kind.into!r} must be a float state, '
                                f'not {init.dtype}')
            if init.unit is not None and u.get_dim(init.unit) != u.get_dim(kind.unit):
                raise PortError(f'{where}: weight unit {kind.unit} does not match the dimension '
                                f'of receptor state {kind.into!r} ({init.unit})')

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
        extra = {latch_name(k): Init(from_mantissa(0.0, cls.inputs[k].unit))
                 for k in latched_ports(cls.inputs)}
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
            Port inputs: the previous step's values for latched
            ``current``/``graded`` ports, this step's payload for the others.
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
        """Event jumps applied after the step's propagation.

        The default applies the receptor-port jumps (decision D-b): every
        ``Event(..., into=<state>)`` input adds its payload to ``<state>``
        (several ports into one state add up, in port order). Models without
        receptor ports are left unchanged. ``TimedEvent`` receptor payloads are
        not handled here: a model with one integrates them in its own ``step``.

        Parameters
        ----------
        s, p : Record
            State values after the propagation, and parameter values.
        inp : Record
            Port payloads of the step.
        ctx : StepContext
            Step context.

        Returns
        -------
        Record
            The jumped states.
        """
        changes: dict[str, Any] = {}
        for k, kind in self.inputs.items():
            if isinstance(kind, Event) and kind.into is not None:
                changes[kind.into] = changes.get(kind.into, s[kind.into]) + inp[k]
        return s.replace(**changes) if changes else s

    def conductance_port_current(self, inp: Record, V: Any, p: Record) -> Any:
        r"""Summed current of the ``conductance`` input ports.

        :math:`\sum_k g_k\,\mathrm{gate}_k(V)\,(E_k - V)` over every
        :class:`~brainpy_state._kernel.ports.Conductance` port (receptor ports,
        ``Event(..., into=...)``, are integrated by the model's own states). Call it inside ``derivatives`` with
        the substep's ``V`` so each gate is re-evaluated at every substep.

        Parameters
        ----------
        inp : Record
            Port inputs of the step (conductance payloads).
        V : brainunit.Quantity
            Membrane potential used for the gate and the driving force.
        p : Record
            Parameter values (``E_rev`` given by name).

        Returns
        -------
        brainunit.Quantity
            Current in pA (zeros shaped like ``V`` when there is no such port).
        """
        total = u.Quantity(jnp.zeros_like(to_mantissa(V, u.mV)), unit=u.pA)
        for k, kind in self.inputs.items():
            if isinstance(kind, Conductance):
                total = total + conductance_current(kind, inp[k], V, p)
        return total

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

        # 1. latch: derivatives see the previous step's values of latched ports and
        #    this step's values of unlatched ones (decision D-a)
        inp_ode, new_latches = latch_inputs(self.inputs, s, inp)

        # 2. integrate the float states on mantissas. The substeps run as one scan
        #    (integrators.integrate); every state the hook can change is carried
        #    explicitly: float states in ``y``, all other states (spike output,
        #    int/bool states, latches) as mantissas in ``x``, plus the SpikeCarry.
        names = self.float_state_names()
        units = {k: unit_of(s[k]) for k in s}
        dt_ms = to_mantissa(ctx.dt, u.ms)
        t0_ms = to_mantissa(ctx.t, u.ms)
        detach = rule.resolve_detach(ctx.detach_default(self.detach_reset)) if rule is not None else False
        carry = rule.begin(s, p, dt_ms, integ.substeps) if rule is not None else None
        clamp_var = rule.clamp_var if rule is not None else None
        others = [k for k in s if k not in names]
        quantity = {k: isinstance(s[k], u.Quantity) for k in others}

        def to_record(y, base):
            return base.replace(**{k: from_mantissa(y[k], units[k]) for k in names})

        def pack_others(st):
            return {k: (to_mantissa(st[k], units[k]) if quantity[k] else jnp.asarray(st[k]))
                    for k in others}

        def unpack_others(x, base):
            return base.replace(**{k: (from_mantissa(x[k], units[k]) if quantity[k] else x[k])
                                   for k in others})

        def f(t, y, aux):
            # non-integrated states come from the step start (K0 semantics)
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
                dy[clamp_var] = jnp.where(rule.refractory_mask(aux[1]), 0, dy[clamp_var])
            return dy

        def hook(k, t, y, aux, final):
            x, c = aux
            st = unpack_others(x, to_record(y, s))
            st = rule.clamp(st, p, c)
            c = rule.tick(c, 'substep')
            if final or rule.when == 'substep':
                st, c = rule.check(st, p, c, detach_reset=detach, relaxed=ctx.relaxed)
            return {kk: to_mantissa(st[kk], units[kk]) for kk in names}, (pack_others(st), c)

        y0 = {k: to_mantissa(s[k], units[k]) for k in names}
        aux0 = (pack_others(s), carry)
        y, (x, carry) = integrate(integ, f, y0, t0_ms, dt_ms,
                                  hook=None if rule is None else hook, aux=aux0)
        s = unpack_others(x, to_record(y, s))

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
