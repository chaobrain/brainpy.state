# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""braintrace (online learning) compatibility of K0 networks (K0 spec §4.7, decision L1).

Online-learning algorithms (e-prop, D-RTRL, ...) live in braintrace;
``brainpy.state`` guarantees that its networks are *etrace-compatible* and says
clearly when one is not. braintrace 0.2.6 silently returns zero or wrong gradients
in several situations (spec §1), so the rules are checked before
``braintrace.compile``:

1. **Weights are ``ParamState``\\ s applied through braintrace primitives**
   (``braintrace.matmul`` for Dense, ``braintrace.sparse_matmul`` for CSR -- the
   storages' ``forward_etrace``). A trainable ``ParamState`` that braintrace does not
   relate to a hidden state (raw matmul, neuron parameter used element-wise) would
   get a zero gradient.
2. **No braintrace primitive inside a ``while`` body**: braintrace 0.2.6 has no
   fixed per-iteration Jacobian for a data-dependent trip count and raises
   ``NotImplementedError``. Static ``scan`` / ``fori_loop`` bodies are fine
   (braintrace unrolls them; gradients equal the Python-unrolled loop), so port
   inputs may be applied per substep in a scanned integrator.
3. **Outputs are read from ``HiddenState`` values**: an output depending on a
   weight through a value that is not a new hidden-state value gives zero gradients.
4. **No ``ParamState`` writes inside the traced step**: weights are constant within
   an online step, so local plasticity on a weight that online learning trains is
   rejected (decision L4; braintrace does not support time-varying parameters).
5. **Every new ``HiddenState`` value depends on a hidden state** (braintrace 0.2.6
   raises ``KeyError`` when one depends only on inputs; the fix is
   ``new + 0 * old``, which :class:`EtraceNetwork` applies to every hidden state).

:func:`check_plan` applies the plan-level part (plasticity, capabilities,
trainable neuron parameters) at network-construction time; :func:`check_etrace`
traces one step of a module and applies all rules. :class:`EtraceNetwork` is the
stateful one-step module of a plan that braintrace compiles, and :class:`Online`
is the learner built from a ``Program`` (decision D-c) that runs online-learning
episodes from any state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import brainstate
import braintrace
import jax
import jax.extend.core as jex
import jax.numpy as jnp

from brainpy_state._kernel.params import Record
from brainpy_state._kernel.runner import Program
from brainpy_state._kernel.units import to_mantissa, unit_of

__all__ = [
    'EtraceError',
    'EtraceReport',
    'check_plan',
    'check_etrace',
    'EtraceNetwork',
    'Online',
]

# upstream: braintrace#186 -- braintrace 0.2.6 handles ETP primitives in static scan
# bodies but not in while bodies; rule 2 rejects only the latter.
_LOOP_PRIMITIVES = ('while',)


class EtraceError(ValueError):
    """A network or module violates a braintrace compatibility rule."""


# ---------------------------------------------------------------------------
# plan-level checks
# ---------------------------------------------------------------------------

def check_plan(plan: Any) -> None:
    """Plan-level braintrace rules (run when an :class:`EtraceNetwork` is built).

    Parameters
    ----------
    plan : Plan
        Compiled network.

    Raises
    ------
    EtraceError
        If a projection has a plasticity rule (local plasticity + online learning,
        decision L4), a neuron parameter is trainable (braintrace only learns
        weights applied by its primitives: it would get a zero gradient), a
        population that is not etrace-capable receives trainable weights, or there
        is no trainable weight at all.
    """
    plastic = [pp.name for pp in plan.projections if getattr(pp.proj, 'plasticity', None) is not None]
    if plastic:
        raise EtraceError(f'projections {plastic} have plasticity rules: local plasticity '
                          f'cannot be combined with online learning (weights must stay constant '
                          f'within a braintrace step; decision L4). Use BPTT, or drop the rule.')
    params = [f'{p.name}.{k}' for p in plan.populations for k, s in p.model.param_states.items()
              if isinstance(s, brainstate.ParamState)]
    if params:
        raise EtraceError(f'trainable neuron parameters {params}: braintrace online learning only '
                          f'trains weights applied through its primitives and would give them a '
                          f'zero gradient. Make them non-trainable or use BPTT.')
    trainable = [pp for pp in plan.projections if pp.proj.trainable and pp.groups]
    if not trainable:
        raise EtraceError('no trainable weights: connect(..., trainable=True) at least one projection')
    for pp in trainable:
        for seg in pp.proj.post.segments:
            pplan = plan.populations[seg.pop]
            if not pplan.pop.model.capabilities.etrace:
                raise EtraceError(f'{pp.name}: target population {pplan.name!r} '
                                  f'({pplan.pop.model.__name__}) is not etrace-compatible '
                                  f'(capabilities.etrace=False)')


# ---------------------------------------------------------------------------
# trace-level checks
# ---------------------------------------------------------------------------

@dataclass
class EtraceReport:
    """Summary of a passed :func:`check_etrace`.

    Attributes
    ----------
    weights : list of str
        Paths of the trainable ``ParamState``\\ s (all related to hidden states).
    hidden : list of str
        Paths of the ``HiddenState``\\ s written by the step.
    relations : list of tuple
        ``(primitive name, weight path, hidden paths)`` per braintrace relation.
    """

    weights: list = field(default_factory=list)
    hidden: list = field(default_factory=list)
    relations: list = field(default_factory=list)


def _sub_jaxprs(eqn: Any) -> list:
    out = []

    def visit(v):
        if isinstance(v, (jex.ClosedJaxpr, jex.Jaxpr)):
            out.append(getattr(v, 'jaxpr', v))  # ClosedJaxpr -> Jaxpr (an alias in newer jax)
        elif isinstance(v, (tuple, list)):
            for x in v:
                visit(x)

    for v in eqn.params.values():
        visit(v)
    return out


def _etp_in_loops(jaxpr: Any, in_loop: bool = False) -> list:
    """Names of ETP primitives found inside ``while`` bodies (recursively)."""
    found = []
    for eqn in jaxpr.eqns:
        if isinstance(eqn.primitive, braintrace.ETPPrimitive) and in_loop:
            found.append(eqn.primitive.name)
        loop = in_loop or eqn.primitive.name in _LOOP_PRIMITIVES
        for sub in _sub_jaxprs(eqn):
            found.extend(_etp_in_loops(sub, loop))
    return found


def _is_var(v: Any) -> bool:
    return not isinstance(v, jex.Literal)


def _flow(jaxpr: Any, seed: dict, reset: dict | None = None) -> dict:
    """Forward data-flow of ``int`` bit sets: ``out = OR(inputs)``; ``reset`` overrides."""
    val = dict(seed)
    reset = reset or {}
    for eqn in jaxpr.eqns:
        m = 0
        for v in eqn.invars:
            if _is_var(v):
                m |= val.get(v, 0)
        for o in eqn.outvars:
            val[o] = reset.get(o, m)
    return val


def _state_paths(model: Any) -> dict[int, str]:
    paths = {}
    for path, st in model.states().items():
        paths[id(st)] = '.'.join(str(p) for p in path)
    return paths


def check_etrace(model: Any, *example_inputs: Any) -> EtraceReport:
    """Check the braintrace rules (module docstring) on one traced step of ``model``.

    Parameters
    ----------
    model : brainstate.nn.Module
        Module whose ``update(*example_inputs)`` is one online step (e.g. an
        :class:`EtraceNetwork`); its states must be initialised.
    *example_inputs
        Inputs of one step.

    Returns
    -------
    EtraceReport

    Raises
    ------
    EtraceError
        With a message naming the offending state / primitive and the fix.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> from brainpy_state._kernel.learn.etrace import EtraceError, check_etrace
        >>> class Raw(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.W = brainstate.ParamState(jnp.ones((2, 3)))
        ...         self.h = brainstate.HiddenState(jnp.zeros(3))
        ...     def update(self, x):
        ...         self.h.value = 0.9 * self.h.value + x @ self.W.value  # raw matmul
        ...         return self.h.value
        >>> try:
        ...     check_etrace(Raw(), jnp.ones(2))
        ... except EtraceError as e:
        ...     print(str(e).split(':')[0])
        trainable weights not applied through a braintrace primitive (or not reaching a HiddenState)
    """
    paths = _state_paths(model)
    name = lambda s: paths.get(id(s), repr(s))
    sf = brainstate.transform.StatefulFunction(model.update)
    sf.make_jaxpr(*example_inputs)
    closed = sf.get_jaxpr(*example_inputs)
    jaxpr = getattr(closed, 'jaxpr', closed)
    states = list(sf.get_states(*example_inputs))
    writes = list(sf.get_write_states(*example_inputs))

    # rule 4: no ParamState writes
    # upstream: braintrace#187 -- intended behaviour; shorten this message to a pointer
    # once braintrace documents the rule and makes NotSupportedError actionable.
    bad = [name(s) for s in writes if isinstance(s, brainstate.ParamState)]
    if bad:
        raise EtraceError(f'ParamState(s) {bad} are written inside the step: online learning '
                          f'needs weights that are constant within a step (local plasticity + '
                          f'online learning is not supported, decision L4)')

    # rule 2: no ETP primitives inside while bodies (static scan / fori_loop are fine)
    # upstream: braintrace#186
    looped = _etp_in_loops(jaxpr)
    if looped:
        raise EtraceError(f'braintrace primitives {sorted(set(looped))} are called inside a '
                          f'while_loop body: braintrace cannot build eligibility traces for a '
                          f'data-dependent trip count. Use a static scan / fori_loop (fixed '
                          f'number of iterations), or apply the weights outside the loop')

    # map states <-> jaxpr variables
    n_args = len(jax.tree.leaves(example_inputs))
    in_vars = list(jaxpr.invars)
    state_in: dict[int, list] = {}
    pos = n_args
    for s in states:
        k = len(jax.tree.leaves(s.value))
        state_in[id(s)] = in_vars[pos:pos + k]
        pos += k
    out_vars = list(jaxpr.outvars)
    n_write = sum(len(jax.tree.leaves(s.value)) for s in writes)
    n_out = len(out_vars) - n_write
    state_out: dict[int, list] = {}
    pos = n_out
    for s in writes:
        k = len(jax.tree.leaves(s.value))
        state_out[id(s)] = out_vars[pos:pos + k]
        pos += k
    hidden_r = [s for s in states if isinstance(s, brainstate.HiddenState)]
    hidden_w = [s for s in writes if isinstance(s, brainstate.HiddenState)]
    weights = [s for s in states if isinstance(s, brainstate.ParamState)]

    # rule 5: every new hidden value depends on a hidden state
    seed = {}
    for s in hidden_r:
        for v in state_in[id(s)]:
            seed[v] = 1
    dep = _flow(jaxpr, seed)
    orphan = []
    for s in hidden_w:
        if not any(_is_var(v) and dep.get(v, 0) for v in state_out[id(s)]):
            orphan.append(name(s))
    if orphan:
        raise EtraceError(f'HiddenState(s) {orphan} get a new value that depends on no hidden '
                          f'state (only on inputs/parameters); braintrace 0.2.6 fails on them. '
                          f'Keep the dependency with `new + 0 * old`, or store them in a '
                          f'ShortTermState')

    # rule 3: outputs depend on weights only through new hidden values
    # upstream: braintrace#185 -- braintrace silently returns zero gradients otherwise
    taint_seed = {v: 1 for s in weights for v in state_in[id(s)]}
    clean = {v: 0 for s in hidden_w for v in state_out[id(s)] if _is_var(v)}
    taint = _flow(jaxpr, taint_seed, reset=clean)
    for v in clean:
        taint[v] = 0
    leaked = [i for i, v in enumerate(out_vars[:n_out]) if _is_var(v) and taint.get(v, 0)]
    if leaked:
        raise EtraceError(f'outputs {leaked} depend on trainable weights through values that '
                          f'are not new HiddenState values; braintrace would return zero '
                          f'gradients. Return values read from HiddenStates after writing them')

    # rule 1: every trainable weight goes through a braintrace primitive
    relations = braintrace.find_hidden_param_op_relations_from_module(model, *example_inputs)
    related = {id(st) for r in relations for st in r.trainable_param_states.values()}
    missing = [name(s) for s in weights if id(s) not in related]
    if missing:
        raise EtraceError(f'trainable weights not applied through a braintrace primitive '
                          f'(or not reaching a HiddenState): {missing}; they would get zero '
                          f'online gradients. Route them through braintrace.matmul / '
                          f'sparse_matmul (Storage.forward_etrace)')
    rel = [(r.primitive.name, name(next(iter(r.trainable_param_states.values()))),
            ['.'.join(map(str, h)) for h in r.connected_hidden_paths]) for r in relations]
    return EtraceReport(weights=[name(s) for s in weights], hidden=[name(s) for s in hidden_w],
                        relations=rel)


# ---------------------------------------------------------------------------
# one-step module of a plan
# ---------------------------------------------------------------------------

def _keep_dependency(new: Any, old: Any) -> Any:
    return new + 0 * old


def _names(plan: Any, spec: Any) -> list:
    if spec is None:
        return []
    if isinstance(spec, (str, tuple)):
        spec = [spec]
    out = []
    for item in spec:
        pop, sub = (item if isinstance(item, tuple) else (item, None))
        out.append((plan.population(pop), sub))
    return out


class EtraceNetwork(brainstate.nn.Module):
    """Stateful one-step module of a plan, compiled by ``braintrace.compile``.

    ``update(x)`` runs one network step (the runner's step contract) with
    trainable weights applied through braintrace primitives, writes the new
    states back into the models' ``State`` objects and returns the outputs read
    from their ``HiddenState``\\ s. Delay rings are ``ShortTermState``\\ s (the
    online algorithms treat delayed inputs as given).

    Parameters
    ----------
    plan : Plan or Program
        Compiled network (checked with :func:`check_plan`), or a program whose
        ``spikes`` / ``detach_reset`` options are kept (``etrace=True`` is set).
    inputs : str, tuple or list, optional
        Where ``x`` enters: a population name (its single input port), a
        ``(population, port)`` tuple, or a list of those. One input: ``x`` is an
        ``(n,)`` array; several: a tuple in the list's order.
    outputs : str, tuple or list
        Returned states: a population name (its spike output), a
        ``(population, state)`` tuple, or a list of those; concatenated mantissas.
    trial : int, default 0
        Trial index of the RNG streams (spec §4.5).

    Attributes
    ----------
    program : Program
        Pure step functions with ``etrace=True``.
    n : brainstate.ShortTermState
        Global index of the next step.
    """

    def __init__(self, plan: Any, inputs: Any = None, outputs: Any = None, trial: int = 0) -> None:
        super().__init__()
        program = plan if isinstance(plan, Program) else Program(plan)
        plan = program.plan
        check_plan(plan)
        self.plan = plan
        self.program = program.replace(etrace=True)
        self.trial = int(trial)
        self._inputs = []
        for pplan, port in _names(plan, inputs):
            ports = list(pplan.model.inputs)
            if port is None:
                if len(ports) != 1:
                    raise ValueError(f'input population {pplan.name!r} has ports {ports}; give '
                                     f'({pplan.name!r}, port)')
                port = ports[0]
            pplan.model.inputs.require(port, pplan.name)
            self._inputs.append((pplan, port))
        self._outputs = []
        for pplan, st in _names(plan, outputs):
            if st is None:
                if pplan.model.spiking is None:
                    raise ValueError(f'output population {pplan.name!r} has no spike rule; give '
                                     f'({pplan.name!r}, state)')
                st = pplan.model.spiking.output
            if st not in pplan.model.dyn_states:
                raise ValueError(f'{pplan.name!r} has no state {st!r}')
            if not isinstance(pplan.model.dyn_states[st], brainstate.HiddenState):
                raise EtraceError(f'output {pplan.name}.{st} is not a HiddenState')
            self._outputs.append((pplan, st))
        if not self._outputs:
            raise ValueError('EtraceNetwork needs at least one output')
        self.init_state()

    def init_state(self, *args: Any, **kwargs: Any) -> None:
        """Fresh model states, zero delay rings, step counter 0."""
        for m in self.plan.models:
            m.init_state()
        st0 = self.program.init_state()
        self.rings = {k: brainstate.ShortTermState(v) for k, v in st0['rings'].items()}
        self.arrivals = {k: brainstate.ShortTermState(v) for k, v in st0['arrivals'].items()}
        # synapse-model states (K0 T6) are carried dynamics: HiddenStates for braintrace
        self.syn = {f'{g}/{k}': brainstate.HiddenState(v) for g, d in st0['syn'].items()
                    for k, v in d.items()}
        self.n = brainstate.ShortTermState(jnp.zeros((), jnp.int32))

    def example_input(self) -> Any:
        """Zero input of one step (``None`` without inputs)."""
        xs = tuple(jnp.zeros((p.pop.n,), self.program.dtype) for p, _ in self._inputs)
        if not xs:
            return None
        return xs[0] if len(xs) == 1 else xs

    def read(self) -> dict:
        """Current network state as a program state pytree (``Program.init_state`` layout)."""
        state = {'pops': {p.name: Record(**{k: s.value for k, s in p.model.dyn_states.items()})
                          for p in self.plan.populations},
                 'rings': {k: s.value for k, s in self.rings.items()},
                 'arrivals': {k: s.value for k, s in self.arrivals.items()},
                 'syn': {}, 'plastic': {}}
        for k, s in self.syn.items():
            g, name = k.rsplit('/', 1)
            state['syn'].setdefault(g, {})[name] = s.value
        return state

    def load(self, state: dict, n: Any = 0) -> None:
        """Set the network state from a program state pytree (unbatched) and the step index.

        Parameters
        ----------
        state : dict
            ``Program.init_state()`` layout, e.g. a runner's ``state``.
        n : int, default 0
            Global index of the next step.

        Raises
        ------
        ValueError
            If ``state`` is not an unbatched state of this network's program.
        """
        ref = self.read()
        if jax.tree.structure(state) != jax.tree.structure(ref):
            raise ValueError('state is not a state of this program (Program.init_state layout)')
        for a, r in zip(jax.tree.leaves(state), jax.tree.leaves(ref)):
            if jnp.shape(a) != jnp.shape(r):
                raise ValueError(f'state leaf has shape {jnp.shape(a)}, expected {jnp.shape(r)} '
                                 f'(online learning runs unbatched states)')
        for p in self.plan.populations:
            for k, s in p.model.dyn_states.items():
                s.value = state['pops'][p.name][k]
        for k, s in self.rings.items():
            s.value = state['rings'][k]
        for k, s in self.arrivals.items():
            s.value = state['arrivals'][k]
        for k, s in self.syn.items():
            g, name = k.rsplit('/', 1)
            s.value = state['syn'][g][name]
        self.n.value = jnp.asarray(n, jnp.int32)

    def update(self, x: Any = None) -> jax.Array:
        """One network step; returns the concatenated output mantissas."""
        ext = None
        if self._inputs:
            xs = (x,) if len(self._inputs) == 1 else tuple(x)
            if len(xs) != len(self._inputs):
                raise ValueError(f'expected {len(self._inputs)} inputs, got {len(xs)}')
            ext = {}
            for (pplan, port), v in zip(self._inputs, xs):
                ext.setdefault(pplan.name, {})[port] = v
        plan = self.plan
        prog = self.program
        state = self.read()
        n = self.n.value
        new, _ = prog.step(prog.read_params(), state, n, jnp.uint32(self.trial), ext=ext)
        for p in plan.populations:
            for k, s in p.model.dyn_states.items():
                v = new['pops'][p.name][k]
                s.value = _keep_dependency(v, s.value) if isinstance(s, brainstate.HiddenState) else v
        for k, s in self.rings.items():
            s.value = new['rings'][k]
        for k, s in self.arrivals.items():
            s.value = new['arrivals'][k]
        for k, s in self.syn.items():
            g, name = k.rsplit('/', 1)
            s.value = _keep_dependency(new['syn'][g][name], s.value)
        self.n.value = n + 1
        outs = []
        for pplan, st in self._outputs:
            v = pplan.model.dyn_states[st].value
            outs.append(to_mantissa(v, unit_of(v)))
        return outs[0] if len(outs) == 1 else jnp.concatenate(outs)


# ---------------------------------------------------------------------------
# learner
# ---------------------------------------------------------------------------

class Online:
    """braintrace online learner over a :class:`Program` (decision D-c).

    Builds an :class:`EtraceNetwork` of the program (with ``etrace=True``), runs
    :func:`check_etrace` and compiles it with ``braintrace.compile``. Episodes start
    from an explicit ``state0`` (unbatched) at global step ``n0``: the network's
    states are loaded from it and the eligibility traces are reset.

    Parameters
    ----------
    program : Program
        The program to learn (``plan.program()``); its ``spikes`` / ``detach_reset``
        options are kept.
    algorithm : str or braintrace algorithm, default 'eprop'
        Passed to ``braintrace.compile`` (``'eprop'``, ``'d_rtrl'``, ...).
    inputs, outputs
        See :class:`EtraceNetwork`.
    loss : callable, optional
        Per-step loss ``loss(outputs, target) -> scalar`` (``outputs``: the
        network's concatenated output mantissas of the step); needed by
        :meth:`grad` / :meth:`run`.
    optimizer : braintools.optim.Optimizer, optional
        Needed by :meth:`run`; registered on the trainable weights here.
    **options
        Extra ``braintrace.compile`` options.

    Attributes
    ----------
    program : Program
        The program the network runs (``etrace=True``).
    network : EtraceNetwork
        The one-step module.
    model : braintrace.ETraceAlgorithm
        The compiled online learner (``model(x)`` steps;
        ``model.etrace_grad(xs, ys, step_fn=...)`` computes gradients).
    report : EtraceReport
        Result of :func:`check_etrace`.
    params : dict of str to brainstate.ParamState
        Trainable weights (``'weights/<projection>/<group>'``).

    Raises
    ------
    TypeError
        If ``program`` is not a Program.
    EtraceError
        If a braintrace rule is violated (including: no trainable weight).
    """

    def __init__(self, program: Any, algorithm: Any = 'eprop', *, inputs: Any = None,
                 outputs: Any = None, loss: Any = None, optimizer: Any = None,
                 **options: Any) -> None:
        if not isinstance(program, Program):
            raise TypeError(f'Online needs a Program (plan.program()), got '
                            f'{type(program).__name__}')
        if loss is not None and not callable(loss):
            raise TypeError(f'loss must be callable (outputs, target) -> scalar, got {loss!r}')
        self.algorithm = algorithm
        self.loss_fn = loss
        self.optimizer = optimizer
        self.options = options
        net = EtraceNetwork(program, inputs, outputs)
        self.program = net.program
        x0 = net.example_input()
        args = () if x0 is None else (x0,)
        self.report = check_etrace(net, *args)
        self.model = braintrace.compile(net, algorithm, *args, **options)
        self.network = net
        self.params = {f'weights/{pp.name}/{g}': grp.weight
                       for pp in self.program.plan.projections for g, grp in enumerate(pp.groups)
                       if isinstance(grp.weight, brainstate.ParamState)}
        if optimizer is not None:
            optimizer.register_trainable_weights(self.params)
        self._jit_grad = brainstate.transform.jit(self._grad)
        self._jit_run = brainstate.transform.jit(self._run)

    @property
    def plan(self) -> Any:
        """The compiled network of :attr:`program`."""
        return self.program.plan

    def load(self, state0: Any = None, n0: Any = 0) -> None:
        """Start an episode: load ``state0`` at step ``n0`` and reset the traces.

        Parameters
        ----------
        state0 : dict, optional
            Unbatched state pytree of the program (``program.init_state()``, a
            runner's ``state``, or the state :meth:`run` returned); ``None`` for
            the initial state.
        n0 : int, default 0
            Global index of the next step.
        """
        self.network.load(self.program.init_state() if state0 is None else state0, n0)
        reset = getattr(self.model, 'reset_state', None)
        if callable(reset):
            reset()

    def state(self) -> tuple[dict, jax.Array]:
        """Current ``(state, n)`` of the network (a program state pytree, next step)."""
        return self.network.read(), self.network.n.value

    def _sequence_loss(self, xs: Any, ys: Any) -> tuple:
        if self.loss_fn is None:
            raise RuntimeError('Online was created without a loss')
        loss_fn, model = self.loss_fn, self.model
        grads, loss = model.etrace_grad(
            xs, ys, step_fn=lambda x, y: jnp.asarray(loss_fn(model(x), y)),
            weights=self.params, return_value=True, loss_output='scalar')
        return loss, grads

    def _grad(self, xs: Any, ys: Any) -> tuple:
        loss, grads = self._sequence_loss(xs, ys)
        return loss, grads, self.network.read()

    def _run(self, xs: Any, ys: Any) -> tuple:
        loss, grads, state = self._grad(xs, ys)
        self.optimizer.update(grads)
        return loss, state

    def grad(self, state0: Any, data: Any, targets: Any, *, n0: Any = 0) -> tuple[jax.Array, dict]:
        """Online (eligibility-trace) gradients of one episode from ``state0``.

        Parameters
        ----------
        state0 : dict or None
            See :meth:`load`.
        data : array or tuple of arrays
            Per-step inputs, ``(T, n_in)`` (a tuple for several inputs, in the
            order of ``inputs``).
        targets : array
            Per-step targets ``(T, ...)`` passed to the loss with the step's
            outputs.
        n0 : int, default 0
            Global index of the episode's first step.

        Returns
        -------
        loss : jax.Array
            Mean per-step loss.
        grads : dict of str to Quantity
            Accumulated online gradients per trainable weight (keys of
            :attr:`params`).
        """
        self.load(state0, n0)
        loss, grads, _ = self._jit_grad(data, targets)
        return loss, grads

    def run(self, state0: Any, data: Any, targets: Any, *, n0: Any = 0) -> tuple[jax.Array, dict]:
        """One online-learning episode: gradients from ``state0`` and one optimizer step.

        Arguments as :meth:`grad`.

        Returns
        -------
        loss : jax.Array
            Mean per-step loss (before the update).
        state : dict
            The episode's final state (the next episode starts here at ``n0 + T``).
        """
        if self.optimizer is None:
            raise RuntimeError('Online was created without an optimizer')
        self.load(state0, n0)
        return self._jit_run(data, targets)
