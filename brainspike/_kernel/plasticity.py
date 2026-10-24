# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Plasticity rules over per-edge state, NEST lazy timing (K0 spec §4.6, T6).

A :class:`PlasticityRule` is attached with ``net.connect(..., plasticity=rule)``. It
keeps the protocol of today's ``_nest_network/event_plastic.py`` substrate -- a pure,
vectorised ``update(state, ctx) -> (state, w_eff)`` over the ``(E,)`` edge arrays,
plus declared traces / reads / modes -- with three changes:

* rule parameters are **traced** (``ctx.params``), held in ``ConstParamState`` or,
  when listed in ``trainable``, ``ParamState`` (learned plasticity);
* the traces handed to the rule are the **strictly prior** values (NEST
  ``get_K_value``), so rules no longer subtract the current spike;
* the dendritic / axonal **delay split** is part of the runtime (no faked shift).

Timing (architecture doc §7.1)
------------------------------
With a projection delay of ``d = d_ax + d_den`` steps (``d_den >= 1``), a pre spike of
step ``n`` reaches the synapse at step ``n + d_ax`` and a post spike of step ``m`` at
``m + d_den``. At synapse step ``s`` the runtime decays the traces, applies the rule to
the post arrivals and then to the pre arrivals (NEST ``stdp_synapse::send``:
facilitation by the post history, then depression with ``K-(t_pre - d)``), adds the
events to the traces, and deposits ``W_eff^T x`` into a post-side arrival ring
``d_den`` steps ahead -- so each event is delivered with the weight of its own send.
NEST's ``stdp_synapse`` has ``d_ax = 0``.

``timing='lazy'`` (default, NEST): the observable weight changes only on pre sends
(an extra committed ``(E,)`` copy); ``timing='eager'``: the working weight is the
observable. Delivered amplitudes are identical in both modes.

Two paths
---------
``'fast'``: ``brainevent.update_csr_on_binary_post`` then
``update_csr_on_binary_pre`` ("add trace, clip") for rules with a
:meth:`PlasticityRule.fast_form`; forward only. ``'reference'``: the rule's ``update``
over gathered ``(E,)`` arrays; differentiable w.r.t. the rule parameters.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

import brainevent
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainspike._kernel.params import ConstParamState
from brainspike._kernel.units import check_unit, to_mantissa

__all__ = [
    'PLASTICITY_PATHS',
    'TIMINGS',
    'PlasticContext',
    'FastForm',
    'PlasticityRule',
    'static_synapse',
    'stdp_synapse',
    'PlasticPlan',
]

#: Paths accepted by :class:`PlasticityRule` (``path=``).
PLASTICITY_PATHS = ('auto', 'fast', 'reference')

#: Timing semantics accepted by :class:`PlasticityRule` (``timing=``).
TIMINGS = ('lazy', 'eager')


class PlasticContext(NamedTuple):
    """Per-step inputs of :meth:`PlasticityRule.update` (all unit-free mantissas).

    Per-edge arrays have shape ``(E,)`` in the projection's CSR (pre-major) order.

    Attributes
    ----------
    pre_spike : jax.Array
        ``1`` where the edge's pre spike reaches the synapse this step, else ``0``.
    post_spike : jax.Array
        ``1`` where the edge's post spike reaches the synapse this step (after the
        dendritic delay), else ``0``.
    pre_trace : jax.Array
        Pre trace (``traces['pre']`` declared), **strictly prior** to this step's own
        pre event; zeros when not declared.
    post_trace : jax.Array
        Post trace, strictly prior to this step's post arrival; zeros when not
        declared.
    params : dict
        Rule parameters ``{name: scalar}`` (traced; differentiable on the
        reference path).
    t : jax.Array
        Time at the start of the synapse step, ms.
    dt : jax.Array
        Step, ms.
    key : jax.Array or None
        RNG key (stochastic rules; ``None`` in K0).
    """

    pre_spike: jax.Array
    post_spike: jax.Array
    pre_trace: jax.Array
    post_trace: jax.Array
    params: dict
    t: jax.Array
    dt: jax.Array
    key: Any = None


@dataclass(frozen=True)
class FastForm:
    """"Add trace, clip" form of a rule (brainevent fast path).

    On a post arrival at neuron ``j``: ``w_e += on_post * pre_trace[pre(e)]`` for the
    edges into ``j``; then on a pre arrival of ``i``: ``w_e += on_pre *
    post_trace[post(e)]`` for the edges out of ``i``; every update clips **all**
    weights to ``[w_min, w_max]``.

    Attributes
    ----------
    on_post, on_pre : jax.Array
        Scalar coefficients of the pre trace (post-triggered) and of the post trace
        (pre-triggered).
    w_min, w_max : jax.Array
        Clip interval.
    """

    on_post: Any
    on_pre: Any
    w_min: Any
    w_max: Any


class PlasticityRule:
    """Base class / protocol of plasticity rules.

    Subclasses set the class attributes, call ``super().__init__`` with their
    parameter values, and implement :meth:`update` (and optionally
    :meth:`fast_form`, :meth:`weight_interval`, :meth:`check_weights`).

    Parameters
    ----------
    params : mapping of str to float
        Rule parameter values (mantissas: times in ms, weights in the projection's
        weight unit).
    trainable : sequence of str, default ()
        Parameters stored as ``brainstate.ParamState`` (learned plasticity); forces
        the differentiable reference path.
    axonal_delay : brainunit.Quantity, default 0 ms
        Axonal part of the projection delay (rounded to steps); the rest is
        dendritic and must be at least one step.
    timing : {'lazy', 'eager'}, default 'lazy'
        Observable-weight semantics (see the module docstring).
    path : {'auto', 'fast', 'reference'}, default 'auto'
        Update path; ``'auto'`` picks ``'fast'`` when eligible.

    Attributes
    ----------
    traces : dict
        ``{'pre' | 'post': <name of the tau parameter (ms)>}``; all-to-all traces
        (decay, then ``+1`` per event).
    reads : tuple of str
        Post-neuron state reads (none in K0).
    post_port : str
        Output port of the post population that feeds the post trace.
    is_static : bool
        The rule never changes the weights (compiled as an ordinary projection).
    """

    traces: ClassVar[dict] = {}
    reads: ClassVar[tuple] = ()
    post_port: ClassVar[str] = 'spike'
    is_static: ClassVar[bool] = False
    stochastic: ClassVar[bool] = False

    def __init__(self, params: Mapping[str, Any] | None = None, *, trainable: Any = (),
                 axonal_delay: Any = 0.0 * u.ms, timing: str = 'lazy', path: str = 'auto') -> None:
        self.params = {k: float(v) for k, v in (params or {}).items()}
        if isinstance(trainable, str):
            trainable = (trainable,)
        trainable = tuple(trainable)
        unknown = set(trainable) - set(self.params)
        if unknown:
            raise ValueError(f'{type(self).__name__}: unknown trainable parameter(s) {sorted(unknown)}; '
                             f'parameters: {sorted(self.params)}')
        self.trainable = trainable
        check_unit(axonal_delay, u.ms, 'axonal_delay')
        ax = np.asarray(u.Quantity(axonal_delay).to_decimal(u.ms), np.float64)
        if ax.ndim != 0 or not np.isfinite(ax) or ax < 0:
            raise ValueError(f'axonal_delay must be a non-negative scalar, got {axonal_delay}')
        self.axonal_delay = float(ax) * u.ms
        if timing not in TIMINGS:
            raise ValueError(f'timing must be one of {TIMINGS}, got {timing!r}')
        self.timing = timing
        if path not in PLASTICITY_PATHS:
            raise ValueError(f'path must be one of {PLASTICITY_PATHS}, got {path!r}')
        self.path = path
        for side, name in self.traces.items():
            if side not in ('pre', 'post'):  # pragma: no cover - subclass declaration error
                raise ValueError(f'trace side must be pre or post, got {side!r}')
            if not self.params.get(name, 0.0) > 0:
                raise ValueError(f'{name} must be positive, got {self.params.get(name)}')

    def __repr__(self) -> str:
        inner = ', '.join(f'{k}={v:g}' for k, v in self.params.items())
        return f'{type(self).__name__}({inner})'

    # ------------------------------------------------------------ protocol
    def edge_state_init(self) -> dict:
        """Extra per-edge state ``{name: initial scalar}`` (``weight`` is implicit)."""
        return {}

    def update(self, state: dict, ctx: PlasticContext) -> tuple[dict, jax.Array]:  # pragma: no cover
        """Pure rule step over ``(E,)`` arrays; returns ``(state, w_eff)``."""
        raise NotImplementedError

    def fast_form(self, p: Mapping[str, Any]) -> FastForm | None:
        """The "add trace, clip" form for parameter values ``p``, or ``None``."""
        return None

    def weight_interval(self, p: Mapping[str, float]) -> tuple[float, float] | None:
        """Interval initial weights must lie in for the fast path (``None``: any)."""
        return None

    def check_weights(self, w0: np.ndarray, p: Mapping[str, float]) -> None:
        """Validate the realised initial weights (mantissas, concrete) against ``p``."""

    def resolve(self, weight_unit: Any) -> dict:
        """Parameter values (mantissas) for a projection with ``weight_unit``."""
        return dict(self.params)


class static_synapse(PlasticityRule):
    """NEST ``static_synapse``: weights never change.

    A projection with ``plasticity=static_synapse()`` compiles to an ordinary
    projection (the weight stays in the plan's weight state; any storage).

    Examples
    --------
    .. code-block:: python

        >>> from brainspike._kernel.plasticity import static_synapse
        >>> r = static_synapse()
        >>> r.is_static, r.edge_state_init()
        (True, {})
    """

    is_static: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__({})

    def update(self, state: dict, ctx: PlasticContext) -> tuple[dict, jax.Array]:
        return state, state['weight']


def _pow(x: jax.Array, mu: float) -> jax.Array:
    # mu is static: avoid 0 ** 0 (and its NaN gradient) for the additive rule
    if mu == 0.0:
        return jnp.ones_like(x)
    if mu == 1.0:
        return x
    return x ** mu


class stdp_synapse(PlasticityRule):
    r"""NEST ``stdp_synapse``: pair-based all-to-all STDP (Guetig et al. 2003).

    With :math:`\hat w = w / W_{\max}`, on a post arrival at the synapse

    .. math:: \hat w \leftarrow \min\big(1,\ \hat w + \lambda (1-\hat w)^{\mu_+} K^+\big)

    and on a pre arrival

    .. math:: \hat w \leftarrow \max\big(0,\ \hat w - \alpha\lambda \hat w^{\mu_-} K^-\big),

    facilitation first when both happen in one step (NEST order). :math:`K^+` is the
    pre trace (``tau_plus``), :math:`K^-` the post trace (``tau_minus``, a property of
    the post neuron in NEST), both strictly prior to the step's own event. The whole
    NEST delay is dendritic (``axonal_delay=0``).

    Parameters
    ----------
    tau_plus, tau_minus : brainunit.Quantity, default 20 ms
        Trace time constants (> 0).
    lambda_ : float, default 0.01
        Learning rate.
    alpha : float, default 1.0
        Depression asymmetry.
    mu_plus, mu_minus : float, default 1.0
        Weight-dependence exponents (static: they select the rule's form; ``0`` is
        additive, ``1`` multiplicative).
    Wmax : float or brainunit.Quantity, default 100.0
        Weight bound, same sign as the weights (a float is in the projection's
        weight unit).
    trainable, axonal_delay, timing, path
        See :class:`PlasticityRule`. Trainable: ``'lambda_'``, ``'alpha'``,
        ``'tau_plus'``, ``'tau_minus'``, ``'Wmax'``.

    Notes
    -----
    The fast path applies when ``mu_plus == mu_minus == 0``: facilitation adds
    :math:`\lambda W_{\max} K^+` and depression :math:`-\alpha\lambda W_{\max} K^-`,
    clipped to :math:`[\min(0, W_{\max}), \max(0, W_{\max})]`. It needs the initial
    weights inside that interval (brainevent clips all weights; NEST only clips the
    side it moves to). Pre spikes of multiplicity > 1 count once for the trace (NEST
    ``Kplus_ += 1`` per send) and ``count x w`` for the delivery.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainspike._kernel.plasticity import stdp_synapse
        >>> r = stdp_synapse(tau_plus=16.8 * u.ms, lambda_=0.005, mu_plus=0.0, mu_minus=0.0)
        >>> r.traces
        {'pre': 'tau_plus', 'post': 'tau_minus'}
        >>> r.fast_form(r.params) is not None, stdp_synapse().fast_form({}) is None
        (True, True)
    """

    traces: ClassVar[dict] = {'pre': 'tau_plus', 'post': 'tau_minus'}

    def __init__(self, tau_plus: Any = 20.0 * u.ms, tau_minus: Any = 20.0 * u.ms,
                 lambda_: float = 0.01, alpha: float = 1.0, mu_plus: float = 1.0,
                 mu_minus: float = 1.0, Wmax: Any = 100.0, *, trainable: Any = (),
                 axonal_delay: Any = 0.0 * u.ms, timing: str = 'lazy', path: str = 'auto') -> None:
        check_unit(tau_plus, u.ms, 'tau_plus')
        check_unit(tau_minus, u.ms, 'tau_minus')
        self.mu_plus = float(mu_plus)
        self.mu_minus = float(mu_minus)
        if isinstance(Wmax, u.Quantity) and not Wmax.unit.is_unitless:
            self.Wmax_q = Wmax
            wmax = float(np.sign(float(u.get_mantissa(Wmax))))  # resolved per projection
        else:
            self.Wmax_q = None
            wmax = float(u.get_mantissa(Wmax))
        params = dict(tau_plus=float(u.Quantity(tau_plus).to_decimal(u.ms)),
                      tau_minus=float(u.Quantity(tau_minus).to_decimal(u.ms)),
                      lambda_=float(lambda_), alpha=float(alpha), Wmax=wmax)
        if params['Wmax'] == 0.0:
            raise ValueError('Wmax must be non-zero')
        super().__init__(params, trainable=trainable, axonal_delay=axonal_delay, timing=timing,
                         path=path)

    def resolve(self, weight_unit: Any) -> dict:
        p = dict(self.params)
        if self.Wmax_q is not None:
            check_unit(self.Wmax_q, weight_unit, 'Wmax')
            p['Wmax'] = float(to_mantissa(self.Wmax_q, weight_unit))
        return p

    def check_weights(self, w0: np.ndarray, p: Mapping[str, float]) -> None:
        # NEST set_status: sign(x) = (x >= 0) - (x < 0), zero counts as positive
        sign_w = np.where(w0 >= 0, 1, -1)
        if w0.size and np.any(sign_w != (1 if p['Wmax'] >= 0 else -1)):
            raise ValueError('stdp_synapse: weights and Wmax must have the same sign (NEST)')

    def weight_interval(self, p: Mapping[str, float]) -> tuple[float, float]:
        w = float(p['Wmax'])
        return (min(0.0, w), max(0.0, w))

    # ------------------------------------------------------------- kernel
    def _facilitate(self, w: jax.Array, k: jax.Array, p: Mapping) -> jax.Array:
        wmax = p['Wmax']
        norm = w / wmax + p['lambda_'] * _pow(1.0 - w / wmax, self.mu_plus) * k
        return jnp.where(norm < 1.0, norm * wmax, wmax)

    def _depress(self, w: jax.Array, k: jax.Array, p: Mapping) -> jax.Array:
        wmax = p['Wmax']
        norm = w / wmax - p['alpha'] * p['lambda_'] * _pow(w / wmax, self.mu_minus) * k
        return jnp.where(norm > 0.0, norm * wmax, jnp.zeros_like(norm))

    def update(self, state: dict, ctx: PlasticContext) -> tuple[dict, jax.Array]:
        p = ctx.params
        w = state['weight']
        w = jnp.where(ctx.post_spike > 0, self._facilitate(w, ctx.pre_trace, p), w)
        w = jnp.where(ctx.pre_spike > 0, self._depress(w, ctx.post_trace, p), w)
        return {**state, 'weight': w}, w

    def fast_form(self, p: Mapping[str, Any]) -> FastForm | None:
        if self.mu_plus != 0.0 or self.mu_minus != 0.0:
            return None
        wmax = p['Wmax']
        zero = jnp.zeros_like(jnp.asarray(wmax))
        return FastForm(on_post=p['lambda_'] * wmax,
                        on_pre=-p['alpha'] * p['lambda_'] * wmax,
                        w_min=jnp.minimum(zero, wmax), w_max=jnp.maximum(zero, wmax))


def _param_state(value: Any, trainable: bool) -> brainstate.State:
    return brainstate.ParamState(value) if trainable else ConstParamState(value)


class PlasticPlan:
    """Compiled plasticity of one projection (both paths, traces, lazy commit).

    Built by the compiler; used by the runner's emit phase (pure methods).

    Parameters
    ----------
    rule : PlasticityRule
        The rule (``is_static`` False).
    storage : CSR
        The projection's CSR storage (edge structure; weights live in the state).
    w0 : jax.Array
        Initial weights in CSR order (mantissas, ``(E,)``).
    weight_unit : brainunit.Unit
        Unit of the weights.
    d_ax, d_den : int
        Axonal and dendritic delays in steps (``d_den >= 1``).
    dtype : numpy.dtype
        Float dtype.

    Attributes
    ----------
    path : {'fast', 'reference'}
        Resolved update path.
    states : dict of str to brainstate.State
        Rule parameter states (``ParamState`` for trainable ones).
    """

    def __init__(self, rule: PlasticityRule, storage: Any, w0: jax.Array, weight_unit: Any,
                 d_ax: int, d_den: int, dtype: Any) -> None:
        self.rule = rule
        self.storage = storage
        self.dtype = np.dtype(dtype)
        self.weight_unit = weight_unit
        self.d_ax = int(d_ax)
        self.d_den = int(d_den)
        self.n_pre, self.n_post, self.n_edges = storage.n_pre, storage.n_post, storage.n_edges
        self.w0 = jnp.asarray(w0, self.dtype)
        p = rule.resolve(weight_unit)
        w0_np = np.asarray(self.w0)
        rule.check_weights(w0_np, p)
        self.states = {k: _param_state(jnp.asarray(v, self.dtype), k in rule.trainable)
                       for k, v in p.items()}
        self.pre_e = jnp.asarray(storage._pre[storage.order].astype(np.int32))
        self.post_e = jnp.asarray(np.asarray(storage.indices).astype(np.int32))
        self.path = self._resolve_path(p, w0_np)
        if self.path == 'fast':
            # upstream: brainevent#200 -- int64 -> int32 scatter FutureWarning under x64
            ptr, ind, perm = brainevent.csr_to_csc_index(storage.indptr, storage.indices,
                                                         shape=(self.n_pre, self.n_post))
            self.csc: tuple[Any, Any, Any] | None = (jnp.asarray(ind), jnp.asarray(ptr),
                                                     jnp.asarray(perm))
        else:
            self.csc = None

    # upstream: brainevent#202 -- the fast update kernels are forward-only, so
    # differentiable runs fall back to the generic path chosen here.
    def _resolve_path(self, p: dict, w0: np.ndarray) -> str:
        rule = self.rule
        has_fast = rule.fast_form(p) is not None
        why = None
        if not has_fast:
            why = f'{type(rule).__name__} with these parameters has no "add trace, clip" form'
        elif rule.trainable:
            why = f'trainable rule parameters {list(rule.trainable)} need the differentiable path'
        else:
            lo_hi = rule.weight_interval(p)
            if lo_hi is not None and w0.size and (w0.min() < lo_hi[0] or w0.max() > lo_hi[1]):
                why = (f'initial weights lie outside the clip interval {lo_hi} (brainevent '
                       f'clips all weights)')
        if rule.path == 'fast':
            if why is not None:
                raise ValueError(f"path='fast' is not available: {why}")
            return 'fast'
        if rule.path == 'reference' or why is not None:
            return 'reference'
        return 'fast'

    # ---------------------------------------------------------------- state
    @property
    def lazy(self) -> bool:
        """Whether a committed (observable) weight copy is kept."""
        return self.rule.timing == 'lazy'

    def init_state(self) -> dict:
        """Initial state pytree (one trial)."""
        aux = {k: jnp.full((self.n_edges,), v, self.dtype)
               for k, v in self.rule.edge_state_init().items()}
        st = {'edge': {'weight': self.w0, **aux}}
        if self.lazy:
            st['committed'] = self.w0
        traces = self.rule.traces
        st['pre_trace'] = jnp.zeros((self.n_pre if 'pre' in traces else 0,), self.dtype)
        st['post_trace'] = jnp.zeros((self.n_post if 'post' in traces else 0,), self.dtype)
        return st

    def read_params(self) -> dict:
        """Current rule parameter values."""
        return {k: s.value for k, s in self.states.items()}

    def weight(self, state: dict) -> jax.Array:
        """Observable weights (CSR order) of a state: committed (lazy) or working."""
        return state['committed'] if self.lazy else state['edge']['weight']

    def nbytes(self) -> int:
        """Device memory of the per-edge and per-neuron state (one trial)."""
        n_edge = 1 + len(self.rule.edge_state_init()) + int(self.lazy)
        n_tr = (self.n_pre if 'pre' in self.rule.traces else 0) + \
               (self.n_post if 'post' in self.rule.traces else 0)
        return (n_edge * self.n_edges + n_tr) * self.dtype.itemsize

    # ----------------------------------------------------------------- step
    def _decay(self, trace: jax.Array, side: str, p: dict, dt: Any) -> jax.Array:
        name = self.rule.traces.get(side)
        if name is None:
            return trace
        return trace * jnp.exp(-dt / p[name])

    def step(self, state: dict, p: dict, x: jax.Array, q: jax.Array, t: Any, dt: Any
             ) -> tuple[dict, jax.Array]:
        """One synapse step (pure).

        Parameters
        ----------
        state : dict
            :meth:`init_state` pytree.
        p : dict
            Rule parameter values (:meth:`read_params`).
        x : jax.Array
            ``(n_pre,)`` pre events reaching the synapse this step (counts).
        q : jax.Array
            ``(n_post,)`` post events reaching the synapse this step.
        t, dt : jax.Array
            Synapse-step start time and step (ms).

        Returns
        -------
        state : dict
            New state.
        w_eff : jax.Array
            ``(E,)`` weights to deliver this step's pre events with (CSR order).
        """
        dt = jnp.asarray(dt, self.dtype)
        xs = (x > 0).astype(self.dtype)
        qs = (q > 0).astype(self.dtype)
        kp = self._decay(state['pre_trace'], 'pre', p, dt)
        km = self._decay(state['post_trace'], 'post', p, dt)
        has_pre, has_post = 'pre' in self.rule.traces, 'post' in self.rule.traces
        if self.path == 'fast':
            ff = self.rule.fast_form(p)
            # the fast path is chosen only with a fast form and the CSC index (_resolve_path)
            assert ff is not None and self.csc is not None
            shape = (self.n_pre, self.n_post)
            ind, ptr, perm = self.csc
            w = state['edge']['weight']
            w = brainevent.update_csr_on_binary_post(
                w, ind, ptr, perm, (ff.on_post * kp).astype(self.dtype), qs,
                ff.w_min, ff.w_max, shape=shape)
            w = brainevent.update_csr_on_binary_pre(
                w, self.storage.indices, self.storage.indptr, xs,
                (ff.on_pre * km).astype(self.dtype), ff.w_min, ff.w_max, shape=shape)
            w = jnp.asarray(w, self.dtype)
            edge, w_eff = {**state['edge'], 'weight': w}, w
        else:
            zero = jnp.zeros((self.n_edges,), self.dtype)
            ctx = PlasticContext(
                pre_spike=xs[self.pre_e], post_spike=qs[self.post_e],
                pre_trace=kp[self.pre_e] if has_pre else zero,
                post_trace=km[self.post_e] if has_post else zero,
                params=p, t=jnp.asarray(t, self.dtype), dt=dt)
            edge, w_eff = self.rule.update(state['edge'], ctx)
            edge = {k: jnp.asarray(v, self.dtype) for k, v in edge.items()}
            w_eff = jnp.broadcast_to(jnp.asarray(w_eff, self.dtype), (self.n_edges,))
        new = dict(state)
        new['edge'] = edge
        if self.lazy:
            new['committed'] = jnp.where(xs[self.pre_e] > 0, edge['weight'], state['committed'])
        if has_pre:
            new['pre_trace'] = kp + xs
        if has_post:
            new['post_trace'] = km + qs
        return new, w_eff

    def describe(self) -> str:
        """Short label for ``Plan.describe()``."""
        return (f'{type(self.rule).__name__}(path={self.path}, timing={self.rule.timing}, '
                f'd_ax={self.d_ax}, d_den={self.d_den})')
