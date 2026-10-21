# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Surrogate-gradient backpropagation through time over a compiled plan (K0 spec §4.7).

:class:`BPTT` differentiates a loss of an episode -- a rollout of the plan's pure
:class:`~brainpy_state._kernel.runner.Program` from the runner's initial state at
step 0 -- with respect to every trainable parameter of the plan:

* weights of projections connected with ``trainable=True`` (one ``ParamState``
  per edge group, in the storage's native layout), and
* neuron parameters declared trainable (``net.add(..., trainable=('tau',))``).

The rollout is one ``brainstate.transform.scan`` (or
``brainstate.transform.checkpointed_scan`` with ``checkpoint=base``) inside one
``brainstate.transform.jit``; the step carry is the explicit state pytree, hence
``scan`` rather than ``for_loop`` (AGENTS.md #10). ``truncation=K`` stops the
gradient of the carried state every ``K`` steps (truncated BPTT; the forward pass is
unchanged). ``detach_reset`` is the learner-side default of every spike rule's
``detach_reset`` (architecture doc §8.2-2).

The loss sees **every step** of every observer: ``loss(recordings, targets)`` with
``recordings[observer name][recordable]`` of shape ``(T, n)`` (``(batch, T, n)``
for a batched runner), mantissas in the observer's units; observer intervals do not
apply here. :func:`spike_count_mse`, :func:`rate_mse` and :func:`membrane_mse` are
small readouts for the gates.

Per-step data enters through ``xs``: ``{population: {port: (T, n) values}}`` (or
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

from brainpy_state._kernel.runner import duration_steps
from brainpy_state._kernel.spiking import relaxed_enabled, relaxed_spikes
from brainpy_state._kernel.units import from_mantissa, unit_of

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
        distinct delay of a post-side projection, else group 0).

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
    return out


def _substitute(params: dict, values: Mapping[str, Any]) -> dict:
    """``params`` (``Program.read_params`` pytree) with trainable entries replaced."""
    pops = dict(params['pops'])
    weights = {k: list(v) for k, v in params['weights'].items()}
    changes: dict[str, dict] = {}
    for key, v in values.items():
        kind, name, leaf = key.split('/', 2)
        if kind == 'pops':
            changes.setdefault(name, {})[leaf] = v
        else:
            weights[name][int(leaf)] = v
    for name, ch in changes.items():
        pops[name] = pops[name].replace(**ch)
    return {'pops': pops, 'weights': weights}


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
    """Surrogate-gradient BPTT learner (see the module docstring).

    Parameters
    ----------
    loss : callable
        ``loss(recordings, targets) -> scalar``; ``recordings`` maps observer names
        to ``{recordable: (T, n) mantissas}`` (leading trial axis when batched).
    optimizer : braintools.optim.Optimizer, optional
        Needed by :meth:`step` / :meth:`fit`; registered on the trainable
        parameters at :meth:`attach`.
    checkpoint : int, optional
        Use ``brainstate.transform.checkpointed_scan`` with this ``base`` (>= 2):
        activations are rematerialised on the backward pass (same gradients, less
        memory).
    truncation : int, optional
        Truncated BPTT: the gradient through the carried state is cut at every
        multiple of ``truncation`` steps.
    detach_reset : bool or None, default True
        Default ``detach_reset`` of every population's spike rule (a rule's own
        explicit setting wins); ``None`` leaves the models unchanged.

    Attributes
    ----------
    runner : Runner or None
        The runner the learner is attached to.
    params : dict of str to brainstate.ParamState
        Trainable parameters (:func:`trainable_states`).

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
        >>> bptt = BPTT(lambda r, y: spike_count_mse(r['rec']['spike'], y),
        ...             braintools.optim.Adam(lr=0.05))
        >>> run = net.compile().runner(learner=bptt)
        >>> xs = {'x': jnp.ones((20, 2))}
        >>> loss, grads = bptt.grad(2.0 * u.ms, xs=xs, targets=jnp.full(3, 5.0))
        >>> sorted(grads), grads['weights/w/0'].shape
        (['weights/w/0'], (2, 3))
    """

    def __init__(self, loss: Callable, optimizer: Any = None, checkpoint: int | None = None,
                 truncation: int | None = None, detach_reset: bool | None = True) -> None:
        if not callable(loss):
            raise TypeError(f'loss must be callable (recordings, targets) -> scalar, got {loss!r}')
        if detach_reset is not None and not isinstance(detach_reset, bool):
            raise TypeError(f'detach_reset must be a bool or None, got {detach_reset!r}')
        self.loss_fn = loss
        self.optimizer = optimizer
        self.checkpoint = _check_int('checkpoint', checkpoint, 2)
        self.truncation = _check_int('truncation', truncation, 1)
        self.detach_reset = detach_reset
        self.runner = None
        self.params: dict[str, brainstate.ParamState] = {}

    # ---------------------------------------------------------------- binding
    def attach(self, runner: Any) -> None:
        """Bind to a runner (called by ``plan.runner(learner=...)``).

        Raises
        ------
        RuntimeError
            If already attached.
        ValueError
            If the plan has no trainable parameter or no observer, or a trainable
            parameter belongs to (or a trainable projection targets) a population
            whose model is not differentiable.
        """
        if self.runner is not None:
            raise RuntimeError('this BPTT learner is already attached to a runner')
        plan = runner.plan
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
        if self.detach_reset is not None:
            for m in plan.models:
                m.detach_reset = self.detach_reset
        self.runner = runner
        self.params = params
        self._units = {k: unit_of(s.value) for k, s in params.items()}
        if self.optimizer is not None:
            self.optimizer.register_trainable_weights(params)
        self._jit_record = brainstate.transform.jit(self._record, static_argnums=(0, 1))
        self._jit_loss = brainstate.transform.jit(self._loss_only, static_argnums=(0, 1))
        self._jit_grad = brainstate.transform.jit(self._value_and_grad, static_argnums=(0, 1))
        self._jit_step = brainstate.transform.jit(self._train_step, static_argnums=(0, 1))
        self._jit_fit = brainstate.transform.jit(self._fit, static_argnums=(0, 1, 2))

    def _require(self, optimizer: bool = False) -> None:
        if self.runner is None:
            raise RuntimeError('attach the learner first: plan.runner(learner=bptt)')
        if optimizer and self.optimizer is None:
            raise RuntimeError('BPTT was created without an optimizer')

    # ------------------------------------------------------------- inputs
    def _normalize_xs(self, xs: Any, n_steps: int) -> Any:
        if xs is None:
            return None
        if not isinstance(xs, Mapping):
            raise TypeError(f'xs must be a mapping {{population: {{port: values}}}}, got '
                            f'{type(xs).__name__}')
        plan = self.runner.plan
        batch = self.runner.batch
        out = {}
        for name, v in xs.items():
            pplan = plan.population(name)
            if not isinstance(v, Mapping):
                ports = list(pplan.model.inputs)
                if len(ports) != 1:
                    raise ValueError(f'xs[{name!r}]: population has input ports {ports}; '
                                     f'give {{port: values}}')
                v = {ports[0]: v}
            for port, a in v.items():
                pplan.model.inputs.require(port, name)
                shape = np.shape(u.get_mantissa(a))
                lead = 2 if batch is not None else 1
                want = (batch, n_steps) if batch is not None else (n_steps,)
                if len(shape) < lead or tuple(shape[:lead]) != want:
                    raise ValueError(f'xs[{name!r}][{port!r}] must have leading shape {want} '
                                     f'(trials, steps) / (steps,), got {shape}')
            out[name] = dict(v)
        return out

    # ------------------------------------------------------------- rollout
    def _episode(self, params: dict, state: dict, trial: Any, xs: Any, n_steps: int) -> dict:
        prog = self.runner.program
        observers = prog.plan.observers
        K = self.truncation

        def body(st, inp):
            i, x = inp
            if K is not None:
                st = _cut(st, (i > 0) & (i % K == 0))
            st, _ = prog.step(params, st, i, trial, ext=x)
            return st, {o.name: prog.observe(o, st['pops']) for o in observers}

        scan_xs = (jnp.arange(n_steps, dtype=jnp.int32), xs)
        if self.checkpoint is None:
            _, ys = brainstate.transform.scan(body, state, scan_xs)
        else:
            _, ys = brainstate.transform.checkpointed_scan(body, state, scan_xs, base=self.checkpoint)
        return ys

    def _recordings(self, params: dict, n_steps: int, xs: Any) -> dict:
        run = self.runner
        f = lambda st, tr, x: self._episode(params, st, tr, x, n_steps)
        if run.batch is not None:
            return jax.vmap(f, in_axes=(0, 0, None if xs is None else 0))(run._init, run._trials, xs)
        return f(run._init, run._trials, xs)

    def _record(self, n_steps: int, relaxed: bool, xs: Any) -> dict:
        with relaxed_spikes(relaxed):
            return self._recordings(self.runner.program.read_params(), n_steps, xs)

    def _objective(self, train: dict, base: dict, n_steps: int, xs: Any, targets: Any) -> jax.Array:
        values = {k: from_mantissa(m, self._units[k]) for k, m in train.items()}
        recs = self._recordings(_substitute(base, values), n_steps, xs)
        out = self.loss_fn(recs, targets)
        if isinstance(out, u.Quantity):
            out = out.to_decimal(u.UNITLESS) if out.is_unitless else out.mantissa
        out = jnp.asarray(out)
        if out.shape != ():
            raise ValueError(f'loss must return a scalar, got shape {out.shape}')
        return out

    def _train_values(self) -> dict:
        return {k: u.get_mantissa(s.value) for k, s in self.params.items()}

    def _loss_only(self, n_steps: int, relaxed: bool, xs: Any, targets: Any) -> jax.Array:
        with relaxed_spikes(relaxed):
            base = self.runner.program.read_params()
            return self._objective(self._train_values(), base, n_steps, xs, targets)

    def _value_and_grad(self, n_steps: int, relaxed: bool, xs: Any, targets: Any) -> tuple:
        with relaxed_spikes(relaxed):
            base = self.runner.program.read_params()
            loss, g = jax.value_and_grad(self._objective)(self._train_values(), base, n_steps, xs,
                                                          targets)
        return loss, {k: from_mantissa(v, self._units[k]) for k, v in g.items()}

    def _train_step(self, n_steps: int, relaxed: bool, xs: Any, targets: Any) -> jax.Array:
        loss, grads = self._value_and_grad(n_steps, relaxed, xs, targets)
        self.optimizer.update(grads)
        return loss

    def _fit(self, n_steps: int, relaxed: bool, n_epochs: int, xs: Any, targets: Any) -> jax.Array:
        return brainstate.transform.for_loop(
            lambda i: self._train_step(n_steps, relaxed, xs, targets),
            jnp.arange(n_epochs))

    # -------------------------------------------------------------- public
    def _prepare(self, duration: Any, xs: Any) -> tuple[int, Any]:
        n_steps = duration_steps(duration, self.runner.program.dt)
        if n_steps < 1:
            raise ValueError('BPTT needs a duration of at least one step')
        return n_steps, self._normalize_xs(xs, n_steps)

    def record(self, duration: Any, xs: Any = None) -> dict:
        """Recordings of one episode exactly as the loss sees them (forward only).

        Returns
        -------
        dict
            ``{observer: {recordable: (T, n)}}`` mantissas (leading trial axis when
            batched).
        """
        self._require()
        n_steps, xs = self._prepare(duration, xs)
        return self._jit_record(n_steps, relaxed_enabled(), xs)

    def loss(self, duration: Any, xs: Any = None, targets: Any = None) -> jax.Array:
        """Loss of one episode of ``duration`` (forward only, jitted).

        Parameters
        ----------
        duration : Quantity
            Episode length (whole multiple of ``dt``).
        xs : mapping, optional
            Per-step external inputs (see the module docstring).
        targets : pytree, optional
            Passed to the loss function.

        Returns
        -------
        jax.Array
            Scalar loss.
        """
        self._require()
        n_steps, xs = self._prepare(duration, xs)
        return self._jit_loss(n_steps, relaxed_enabled(), xs, targets)

    def grad(self, duration: Any, xs: Any = None, targets: Any = None) -> tuple[jax.Array, dict]:
        """Loss and gradients of one episode (see :meth:`loss` for the arguments).

        Returns
        -------
        loss : jax.Array
            Scalar loss.
        grads : dict of str to Quantity
            ``d loss / d mantissa`` per trainable parameter (keys of :attr:`params`),
            in the parameter's unit.
        """
        self._require()
        n_steps, xs = self._prepare(duration, xs)
        return self._jit_grad(n_steps, relaxed_enabled(), xs, targets)

    def step(self, duration: Any, xs: Any = None, targets: Any = None) -> jax.Array:
        """One optimizer step on one episode; returns the loss before the update."""
        self._require(optimizer=True)
        n_steps, xs = self._prepare(duration, xs)
        return self._jit_step(n_steps, relaxed_enabled(), xs, targets)

    def fit(self, duration: Any, n_epochs: int, xs: Any = None, targets: Any = None) -> jax.Array:
        """``n_epochs`` optimizer steps in one compiled ``for_loop``.

        Returns
        -------
        jax.Array, shape (n_epochs,)
            Loss of every epoch (before its update).
        """
        self._require(optimizer=True)
        n_epochs = _check_int('n_epochs', n_epochs, 1)
        if n_epochs is None:
            raise ValueError('n_epochs must be an int >= 1')
        n_steps, xs = self._prepare(duration, xs)
        return self._jit_fit(n_steps, relaxed_enabled(), n_epochs, xs, targets)
