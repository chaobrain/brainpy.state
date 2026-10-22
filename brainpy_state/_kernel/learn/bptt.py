# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Surrogate-gradient BPTT over a :class:`Program` (K0 spec §4.7, K1a-5).

:class:`BPTT` is built from a :class:`~brainpy_state._kernel.runner.Program` (decision
D-c) and differentiates a loss of an episode -- a rollout of the program from an
explicit ``state0`` at global step ``n0`` (the initial state, a checkpoint, or a
runner's state after a warm-up) -- with respect to every trainable parameter of the
plan:

* weights of projections connected with ``trainable=True`` (one ``ParamState``
  per edge group, in the storage's native layout), and
* neuron parameters declared trainable (``net.add(..., trainable=('tau',))``).

The rollout is one ``brainstate.transform.scan`` (or
``brainstate.transform.checkpointed_scan`` with ``checkpoint=base``) inside one
``brainstate.transform.jit``; the step carry is the explicit state pytree, hence
``scan`` rather than ``for_loop`` (AGENTS.md #10). ``truncation=K`` stops the
gradient of the carried state every ``K`` steps (truncated BPTT; the forward pass is
unchanged). ``detach_reset`` and ``spikes`` are the learner's program options
(architecture doc §8.2-2): the learner runs ``program.replace(detach_reset=...,
spikes=...)``, so the caller's program is never mutated.

The loss sees **every step** of every observer: ``loss(recordings, targets)`` with
``recordings[observer name][recordable]`` of shape ``(T, n)`` (``(batch, T, n)``
for a batched episode), mantissas in the observer's units; observer intervals do not
apply here. :func:`spike_count_mse`, :func:`rate_mse` and :func:`membrane_mse` are
small readouts for the gates.

Per-step data enters through ``data``: ``{population: {port: (T, n) values}}`` (or
``{population: (T, n)}`` for single-input populations such as
:class:`~brainpy_state._kernel.learn.inputs.ExternalInput`), added to the port's
input at every step (``Program.step(..., ext=...)``).

Gradients are taken with respect to the parameter **mantissas** and returned as
Quantities in the parameter's unit, which is what ``braintools.optim`` optimizers
expect.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel.runner import Program, duration_steps
from brainpy_state._kernel.units import from_mantissa, unit_of
from brainpy_state._kernel.xla import kernel_jit

__all__ = [
    'BPTT',
    'trainable_states',
    'spike_count_mse',
    'rate_mse',
    'membrane_mse',
]


# ---------------------------------------------------------------------------
# trainable parameters of a plan
# ---------------------------------------------------------------------------

def trainable_states(plan: Any) -> dict[str, brainstate.ParamState]:
    """Every trainable ``ParamState`` of a plan, by a readable key.

    Parameters
    ----------
    plan : Plan
        Compiled network.

    Returns
    -------
    dict of str to brainstate.ParamState
        ``'pops/<population>/<parameter>'`` for neuron parameters and
        ``'weights/<projection>/<group>'`` for projection weights (one group per
        distinct delay of a post-side projection, else group 0), and
        ``'plasticity/<projection>/<rule parameter>'`` for learned plasticity rule
        parameters.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.network import Network
        >>> from brainpy_state._kernel.models import LIF
        >>> from brainpy_state._kernel.learn.bptt import trainable_states
        >>> net = Network(dt=0.1 * u.ms)
        >>> A = net.add(LIF, 3, name='A', trainable=('tau',))
        >>> _ = net.connect(A, A, port='delta', weight=0.1 * u.mV, trainable=True, name='rec')
        >>> sorted(trainable_states(net.compile()))
        ['pops/A/tau', 'weights/rec/0']
    """
    out: dict[str, brainstate.ParamState] = {}
    for p in plan.populations:
        for k, st in p.model.param_states.items():
            if isinstance(st, brainstate.ParamState):
                out[f'pops/{p.name}/{k}'] = st
    for pp in plan.projections:
        for g, grp in enumerate(pp.groups):
            if isinstance(grp.weight, brainstate.ParamState):
                out[f'weights/{pp.name}/{g}'] = grp.weight
    for name, states in getattr(plan, 'plastic_params', {}).items():
        for k, st in states.items():  # learned plasticity (K0 T6, reference path)
            if isinstance(st, brainstate.ParamState):
                out[f'plasticity/{name}/{k}'] = st
    return out


def _substitute(params: dict, values: Mapping[str, Any]) -> dict:
    """``params`` (``Program.read_params`` pytree) with trainable entries replaced."""
    pops = dict(params['pops'])
    weights = {k: list(v) for k, v in params['weights'].items()}
    plasticity = {k: dict(v) for k, v in params.get('plasticity', {}).items()}
    changes: dict[str, dict] = {}
    for key, v in values.items():
        kind, name, leaf = key.split('/', 2)
        if kind == 'pops':
            changes.setdefault(name, {})[leaf] = v
        elif kind == 'plasticity':
            plasticity[name][leaf] = v
        else:
            weights[name][int(leaf)] = v
    for name, ch in changes.items():
        pops[name] = pops[name].replace(**ch)
    return {**params, 'pops': pops, 'weights': weights, 'plasticity': plasticity}


# ---------------------------------------------------------------------------
# loss readouts (gates)
# ---------------------------------------------------------------------------

def _plain(x: Any, unit: Any = None) -> jax.Array:
    if isinstance(x, u.Quantity):
        return x.to_decimal(unit) if unit is not None else x.mantissa
    return jnp.asarray(x)


def spike_count_mse(spikes: Any, target: Any) -> jax.Array:
    """Mean squared error of per-neuron spike counts.

    Parameters
    ----------
    spikes : array, shape (..., T, n)
        Per-step spike output (time on axis -2).
    target : array_like, shape (n,) or broadcastable to (..., n)
        Target counts over the ``T`` steps.

    Returns
    -------
    jax.Array
        Scalar ``mean((sum_t spikes - target) ** 2)``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainpy_state._kernel.learn.bptt import spike_count_mse
        >>> float(spike_count_mse(jnp.array([[1.0, 0.0], [1.0, 1.0]]), jnp.array([1.0, 1.0])))
        0.5
    """
    counts = jnp.sum(_plain(spikes), axis=-2)
    return jnp.mean((counts - _plain(target)) ** 2)


def rate_mse(spikes: Any, target: Any, dt: Any) -> jax.Array:
    """Mean squared error of per-neuron firing rates in Hz.

    Parameters
    ----------
    spikes : array, shape (..., T, n)
        Per-step spike output.
    target : Quantity or array_like
        Target rates (a frequency Quantity, or plain numbers in Hz).
    dt : Quantity
        Step size.

    Returns
    -------
    jax.Array
        Scalar ``mean((rate_Hz - target_Hz) ** 2)``.
    """
    x = _plain(spikes)
    T = x.shape[-2]
    duration_s = T * float(u.Quantity(dt).to_decimal(u.second))
    rate = jnp.sum(x, axis=-2) / duration_s
    return jnp.mean((rate - _plain(target, u.Hz)) ** 2)


def membrane_mse(v: Any, target: Any) -> jax.Array:
    """Mean squared error of a recorded trace (e.g. membrane potential).

    Parameters
    ----------
    v : array, shape (..., T, n)
        Recorded mantissas (observer units).
    target : array_like
        Target in the same units, broadcastable to ``v``.

    Returns
    -------
    jax.Array
        Scalar ``mean((v - target) ** 2)``.
    """
    return jnp.mean((_plain(v) - _plain(target)) ** 2)


# ---------------------------------------------------------------------------
# BPTT
# ---------------------------------------------------------------------------

def _cut(tree: Any, cut: Any) -> Any:
    """Stop the gradient of every float leaf where ``cut`` is True."""
    def f(a):
        if jnp.issubdtype(jnp.asarray(a).dtype, jnp.inexact):
            return jnp.where(cut, jax.lax.stop_gradient(a), a)
        return a
    return jax.tree.map(f, tree)


def _check_int(name: str, v: Any, low: int) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < low:
        raise ValueError(f'{name} must be None or an int >= {low}, got {v!r}')
    return int(v)


class BPTT:
    """Surrogate-gradient BPTT learner over a :class:`Program` (see the module docstring).

    Built from a program (decision D-c); every episode starts from an explicit
    ``state0`` at global step ``n0`` -- the initial state, a checkpoint, or the
    state a :class:`~brainpy_state._kernel.runner.Runner` reached after a warm-up
    (``runner.state``, ``runner.n``, ``runner.trials``). The episode's forward pass
    is the runner's step contract, so it equals ``runner.cont()`` from the same
    state.

    Parameters
    ----------
    program : Program
        The program to differentiate (``plan.program()``).
    loss : callable
        ``loss(recordings, targets) -> scalar``; ``recordings`` maps observer names
        to ``{recordable: (T, n) mantissas}`` (leading trial axis when batched).
    optimizer : braintools.optim.Optimizer, optional
        Needed by :meth:`run` / :meth:`fit`; registered on the trainable
        parameters here.
    checkpoint : int, optional
        Use ``brainstate.transform.checkpointed_scan`` with this ``base`` (>= 2):
        activations are rematerialised on the backward pass (same gradients, less
        memory).
    truncation : int, optional
        Truncated BPTT: the gradient through the carried state is cut at every
        multiple of ``truncation`` steps after the episode start.
    detach_reset : bool or None, default True
        The learner's ``detach_reset`` program option (a spike rule's own explicit
        setting wins); ``None`` keeps the program's option.
    spikes : {'hard', 'relaxed'} or None, default None
        The learner's spike-mode program option; ``None`` keeps the program's.

    Attributes
    ----------
    program : Program
        The program the episodes run (``program`` with the learner's options).
    params : dict of str to brainstate.ParamState
        Trainable parameters (:func:`trainable_states`).

    Raises
    ------
    TypeError
        If ``program`` is not a Program or ``loss`` is not callable.
    ValueError
        If the plan has no trainable parameter or no observer, or a trainable
        parameter belongs to (or a trainable projection targets) a population whose
        model is not differentiable.

    Examples
    --------
    .. code-block:: python

        >>> import braintools
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from brainpy_state._kernel.network import Network
        >>> from brainpy_state._kernel.models import LIF
        >>> from brainpy_state._kernel.learn import BPTT, ExternalInput, spike_count_mse
        >>> net = Network(dt=0.1 * u.ms)
        >>> x = net.add(ExternalInput, 2, name='x')
        >>> out = net.add(LIF, 3, name='out', surrogate=braintools.surrogate.Sigmoid())
        >>> _ = net.connect(x, out, port='delta', weight=0.6 * u.mV, trainable=True, name='w')
        >>> rec = net.observe(out, 'spike', name='rec')
        >>> prog = net.compile().program()
        >>> bptt = BPTT(prog, lambda r, y: spike_count_mse(r['rec']['spike'], y),
        ...             braintools.optim.Adam(lr=0.05))
        >>> data = {'x': jnp.ones((20, 2))}
        >>> loss, grads = bptt.grad(prog.init_state(), data, jnp.full(3, 5.0))
        >>> sorted(grads), grads['weights/w/0'].shape
        (['weights/w/0'], (2, 3))
        >>> loss, state = bptt.run(prog.init_state(), data, jnp.full(3, 5.0))
        >>> loss, state = bptt.run(state, data, jnp.full(3, 5.0), n0=20)  # continue
    """

    def __init__(self, program: Any, loss: Callable, optimizer: Any = None, *,
                 checkpoint: int | None = None, truncation: int | None = None,
                 detach_reset: bool | None = True, spikes: str | None = None) -> None:
        if not isinstance(program, Program):
            raise TypeError(f'BPTT needs a Program (plan.program()), got {type(program).__name__}')
        if not callable(loss):
            raise TypeError(f'loss must be callable (recordings, targets) -> scalar, got {loss!r}')
        options: dict[str, Any] = {}
        if detach_reset is not None:
            options['detach_reset'] = detach_reset
        if spikes is not None:
            options['spikes'] = spikes
        self.program = program.replace(**options)
        self.loss_fn = loss
        self.optimizer = optimizer
        self.checkpoint = _check_int('checkpoint', checkpoint, 2)
        self.truncation = _check_int('truncation', truncation, 1)
        plan = self.program.plan
        params = trainable_states(plan)
        if not params:
            raise ValueError('BPTT: the plan has no trainable parameter; use '
                             "net.connect(..., trainable=True) or net.add(..., trainable=('name',))")
        if not plan.observers:
            raise ValueError('BPTT: the loss is built from observers; add one with net.observe(...)')
        for p in plan.populations:
            cap = p.pop.model.capabilities
            own = any(isinstance(s, brainstate.ParamState) for s in p.model.param_states.values())
            fed = any(plan.projections[j].proj.trainable for j in p.incoming)
            if (own or fed) and not cap.differentiable:
                raise ValueError(f'BPTT: population {p.name!r} ({p.pop.model.__name__}) is not '
                                 f'differentiable but has trainable parameters or trainable '
                                 f'incoming weights')
        self.params: dict[str, brainstate.ParamState] = params
        self._units = {k: unit_of(s.value) for k, s in params.items()}
        if self.optimizer is not None:
            self.optimizer.register_trainable_weights(params)
        self._jit_record = kernel_jit(self._record, static_argnums=(0,))
        self._jit_loss = kernel_jit(self._loss_only, static_argnums=(0,))
        self._jit_grad = kernel_jit(self._value_and_grad, static_argnums=(0,))
        self._jit_run = kernel_jit(self._train_step, static_argnums=(0,))
        self._jit_fit = kernel_jit(self._fit, static_argnums=(0, 1))

    @property
    def plan(self) -> Any:
        """The compiled network of :attr:`program`."""
        return self.program.plan

    def _require_optimizer(self) -> None:
        if self.optimizer is None:
            raise RuntimeError('BPTT was created without an optimizer')

    # ------------------------------------------------------------- inputs
    def _normalize_xs(self, xs: Any, batch: int | None) -> tuple[Any, int | None]:
        """Validated ``{population: {port: values}}`` and the number of steps it implies."""
        if xs is None:
            return None, None
        if not isinstance(xs, Mapping):
            raise TypeError(f'data must be a mapping {{population: {{port: values}}}}, got '
                            f'{type(xs).__name__}')
        plan = self.plan
        lead = 2 if batch is not None else 1
        out, n_steps = {}, None
        for name, v in xs.items():
            pplan = plan.population(name)
            if not isinstance(v, Mapping):
                ports = list(pplan.model.inputs)
                if len(ports) != 1:
                    raise ValueError(f'data[{name!r}]: population has input ports {ports}; '
                                     f'give {{port: values}}')
                v = {ports[0]: v}
            for port, a in v.items():
                pplan.model.inputs.require(port, name)
                shape = np.shape(u.get_mantissa(a))
                if len(shape) < lead or (batch is not None and shape[0] != batch):
                    want = '(trials, steps, ...)' if batch is not None else '(steps, ...)'
                    raise ValueError(f'data[{name!r}][{port!r}] must have leading shape {want} '
                                     f'(trials = {batch}), got {shape}')
                t = shape[lead - 1]
                if n_steps is not None and t != n_steps:
                    raise ValueError(f'data[{name!r}][{port!r}] has {t} steps, other inputs '
                                     f'{n_steps}')
                n_steps = t
            out[name] = dict(v)
        return out, n_steps

    def _prepare(self, state0: Any, data: Any, duration: Any, n0: Any, trials: Any) -> tuple:
        if trials is not None:
            trials = jnp.asarray(trials, jnp.uint32)
            if trials.ndim != 1 or trials.shape[0] < 1:
                raise ValueError(f'trials must be a non-empty 1-D array of trial indices '
                                 f'(Runner.trials of a batched runner), got shape {trials.shape}')
            batch = int(trials.shape[0])
        else:
            batch = None
        xs, t_data = self._normalize_xs(data, batch)
        if duration is not None:
            n_steps = duration_steps(duration, self.program.dt)
            if t_data is not None and t_data != n_steps:
                raise ValueError(f'duration is {n_steps} steps but data has {t_data}')
        elif t_data is not None:
            n_steps = t_data
        else:
            raise ValueError('give per-step data or a duration')
        if n_steps < 1:
            raise ValueError('BPTT needs an episode of at least one step')
        if state0 is None:
            state0 = self.program.init_state(batch)
        else:
            self._check_state(state0, batch)
        n0 = jnp.asarray(n0, jnp.int32)
        if n0.ndim != 0:
            raise ValueError(f'n0 must be a scalar step index, got shape {n0.shape}')
        if trials is None:
            trials = jnp.zeros((), jnp.uint32)
        return n_steps, state0, n0, trials, xs

    def _check_state(self, state0: Any, batch: int | None) -> None:
        ref = jax.eval_shape(self.program.init_state)
        if jax.tree.structure(state0) != jax.tree.structure(ref):
            raise ValueError('state0 is not a state of this program (Program.init_state, '
                             'Runner.state, or a state returned by a learner)')
        for a, r in zip(jax.tree.leaves(state0), jax.tree.leaves(ref)):
            want = r.shape if batch is None else (batch,) + r.shape
            if np.shape(a) != want:
                hint = ('pass trials=runner.trials for a batched state' if batch is None
                        else 'the leading axis must match trials')
                raise ValueError(f'state0 leaf has shape {np.shape(a)}, expected {want} ({hint})')

    # ------------------------------------------------------------- rollout
    def _episode(self, params: dict, state: dict, n0: Any, trial: Any, xs: Any,
                 n_steps: int) -> tuple[dict, dict]:
        prog = self.program
        observers = prog.plan.observers
        K = self.truncation

        def body(st, inp):
            i, x = inp
            if K is not None:
                st = _cut(st, (i > 0) & (i % K == 0))
            st, _ = prog.step(params, st, n0 + i, trial, ext=x)
            return st, {o.name: prog.observe(o, st['pops']) for o in observers}

        scan_xs = (jnp.arange(n_steps, dtype=jnp.int32), xs)
        if self.checkpoint is None:
            return brainstate.transform.scan(body, state, scan_xs)
        return brainstate.transform.checkpointed_scan(body, state, scan_xs, base=self.checkpoint)

    def _rollout(self, params: dict, state0: Any, n0: Any, trials: Any, xs: Any,
                 n_steps: int) -> tuple[dict, dict]:
        f = lambda st, tr, x: self._episode(params, st, n0, tr, x, n_steps)
        if trials.ndim == 1:
            return jax.vmap(f, in_axes=(0, 0, None if xs is None else 0))(state0, trials, xs)
        return f(state0, trials, xs)

    def _record(self, n_steps: int, state0: Any, n0: Any, trials: Any, xs: Any) -> dict:
        return self._rollout(self.program.read_params(), state0, n0, trials, xs, n_steps)[1]

    def _objective(self, train: dict, base: dict, n_steps: int, state0: Any, n0: Any, trials: Any,
                   xs: Any, targets: Any) -> tuple[jax.Array, dict]:
        values = {k: from_mantissa(m, self._units[k]) for k, m in train.items()}
        final, recs = self._rollout(_substitute(base, values), state0, n0, trials, xs, n_steps)
        out = self.loss_fn(recs, targets)
        if isinstance(out, u.Quantity):
            out = out.to_decimal(u.UNITLESS) if out.is_unitless else out.mantissa
        out = jnp.asarray(out)
        if out.shape != ():
            raise ValueError(f'loss must return a scalar, got shape {out.shape}')
        return out, final

    def _train_values(self) -> dict:
        return {k: u.get_mantissa(s.value) for k, s in self.params.items()}

    def _loss_only(self, n_steps: int, state0: Any, n0: Any, trials: Any, xs: Any,
                   targets: Any) -> jax.Array:
        base = self.program.read_params()
        return self._objective(self._train_values(), base, n_steps, state0, n0, trials, xs,
                               targets)[0]

    def _value_and_grad(self, n_steps: int, state0: Any, n0: Any, trials: Any, xs: Any,
                        targets: Any) -> tuple:
        base = self.program.read_params()
        (loss, final), g = jax.value_and_grad(self._objective, has_aux=True)(
            self._train_values(), base, n_steps, state0, n0, trials, xs, targets)
        return loss, {k: from_mantissa(v, self._units[k]) for k, v in g.items()}, final

    def _train_step(self, n_steps: int, state0: Any, n0: Any, trials: Any, xs: Any,
                    targets: Any) -> tuple:
        loss, grads, final = self._value_and_grad(n_steps, state0, n0, trials, xs, targets)
        self.optimizer.update(grads)
        return loss, final

    def _fit(self, n_steps: int, n_epochs: int, state0: Any, n0: Any, trials: Any, xs: Any,
             targets: Any) -> jax.Array:
        return brainstate.transform.for_loop(
            lambda i: self._train_step(n_steps, state0, n0, trials, xs, targets)[0],
            jnp.arange(n_epochs))

    # -------------------------------------------------------------- public
    def record(self, state0: Any = None, data: Any = None, *, duration: Any = None, n0: Any = 0,
               trials: Any = None) -> dict:
        """Recordings of one episode exactly as the loss sees them (forward only).

        Parameters
        ----------
        state0, data, duration, n0, trials
            See :meth:`grad`.

        Returns
        -------
        dict
            ``{observer: {recordable: (T, n)}}`` mantissas of every step (leading
            trial axis when batched); observer intervals do not apply.
        """
        n_steps, state0, n0, trials, xs = self._prepare(state0, data, duration, n0, trials)
        return self._jit_record(n_steps, state0, n0, trials, xs)

    def loss(self, state0: Any = None, data: Any = None, targets: Any = None, *,
             duration: Any = None, n0: Any = 0, trials: Any = None) -> jax.Array:
        """Loss of one episode (forward only, jitted); arguments as :meth:`grad`.

        Returns
        -------
        jax.Array
            Scalar loss.
        """
        n_steps, state0, n0, trials, xs = self._prepare(state0, data, duration, n0, trials)
        return self._jit_loss(n_steps, state0, n0, trials, xs, targets)

    def grad(self, state0: Any = None, data: Any = None, targets: Any = None, *,
             duration: Any = None, n0: Any = 0, trials: Any = None) -> tuple[jax.Array, dict]:
        """Loss and gradients of one episode from ``state0``.

        Parameters
        ----------
        state0 : dict, optional
            State pytree the episode starts from (``program.init_state()``,
            ``runner.state``, or the state :meth:`run` returned); ``None`` for the
            program's initial state.
        data : mapping, optional
            Per-step external inputs ``{population: {port: (T, n)}}`` (or
            ``{population: (T, n)}`` for single-input populations; leading trial
            axis when batched). The episode length ``T`` comes from the data.
        targets : pytree, optional
            Passed to the loss function.
        duration : Quantity, optional
            Episode length (whole multiple of ``dt``); required without ``data``.
        n0 : int, default 0
            Global step index of the episode's first step (time ``n0 * dt``; RNG
            streams), e.g. ``runner.n``.
        trials : array_like, optional
            Trial indices of a batched ``state0`` (``runner.trials``); ``None`` for
            an unbatched episode (trial 0).

        Returns
        -------
        loss : jax.Array
            Scalar loss.
        grads : dict of str to Quantity
            ``d loss / d mantissa`` per trainable parameter (keys of :attr:`params`),
            in the parameter's unit.
        """
        n_steps, state0, n0, trials, xs = self._prepare(state0, data, duration, n0, trials)
        loss, grads, _ = self._jit_grad(n_steps, state0, n0, trials, xs, targets)
        return loss, grads

    def run(self, state0: Any = None, data: Any = None, targets: Any = None, *,
            duration: Any = None, n0: Any = 0, trials: Any = None) -> tuple[jax.Array, dict]:
        """One training episode: gradients from ``state0`` and one optimizer step.

        Arguments as :meth:`grad`.

        Returns
        -------
        loss : jax.Array
            Loss of the episode (before the update).
        state : dict
            The episode's final state (computed with the parameters before the
            update); the next episode of a continuing / truncated run starts here
            at ``n0 + T``.
        """
        self._require_optimizer()
        n_steps, state0, n0, trials, xs = self._prepare(state0, data, duration, n0, trials)
        return self._jit_run(n_steps, state0, n0, trials, xs, targets)

    def fit(self, state0: Any = None, data: Any = None, targets: Any = None, *, n_epochs: int,
            duration: Any = None, n0: Any = 0, trials: Any = None) -> jax.Array:
        """``n_epochs`` optimizer steps on the same episode, in one compiled ``for_loop``.

        Arguments as :meth:`grad`; every epoch starts from ``state0``.

        Returns
        -------
        jax.Array, shape (n_epochs,)
            Loss of every epoch (before its update).
        """
        self._require_optimizer()
        epochs = _check_int('n_epochs', n_epochs, 1)
        if epochs is None:
            raise ValueError('n_epochs must be an int >= 1')
        n_steps, state0, n0, trials, xs = self._prepare(state0, data, duration, n0, trials)
        return self._jit_fit(n_steps, epochs, state0, n0, trials, xs, targets)
