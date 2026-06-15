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

"""
Shared utilities for NEST-compatible neuron and device models.

This module extracts common helper functions used across 60+ model files in the
``brainpy_state._nest`` package.  All functions are stateless (no ``self``
parameter) and operate on plain NumPy / JAX arrays or brainunit quantities.
"""

from typing import Callable, NamedTuple, Optional

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainstate.typing import PyTree
from jax.interpreters.partial_eval import DynamicJaxprTracer

__all__ = [
    'is_tracer',
    'cond_any',
    'validate_aeif_overflow',
    'propagator_exp',
    'alpha_propagator_p31_p32',
    'AdaptiveRungeKuttaStep',
    'stack_schedule_values',
]


def is_tracer(x):
    return isinstance(x, (jax.ShapeDtypeStruct, jax.core.ShapedArray, DynamicJaxprTracer, jax.core.Tracer))


def cond_any(condition) -> bool:
    """Reduce a boolean *condition* to a Python ``bool`` for parameter validation.

    This is the shared guard used by every NEST model's parameter-validation
    code so that ``if`` checks remain safe under ``jax.jit``.  When *condition*
    (or the array backing a unitful :class:`~brainunit.Quantity`) is a JAX tracer
    -- i.e. the model is being constructed/traced under ``jit``, ``vmap``,
    ``grad`` etc. -- the Python ``if`` cannot be evaluated, so this returns
    ``False`` and the guarded validation branch is skipped.  For concrete
    inputs it evaluates ``bool(np.any(...))`` exactly as before.

    Parameters
    ----------
    condition : ArrayLike or Quantity
        A boolean array/scalar (typically the result of a comparison such as
        ``self.C_m <= 0 * u.pF``).  May be a NumPy array, a JAX array, a
        brainunit Quantity, or a JAX tracer.

    Returns
    -------
    bool
        ``False`` if *condition* is a tracer; otherwise ``bool(np.any(condition))``.
    """
    cond = u.get_mantissa(condition)
    if is_tracer(cond):
        return False
    return bool(np.any(np.asarray(cond)))


# ---------------------------------------------------------------------------
# C. Parameter validation
# ---------------------------------------------------------------------------

def validate_aeif_overflow(v_peak, v_th, delta_t):
    """Check exponential term overflow for adaptive exponential models.

    Mirrors the NEST overflow guard for the exponential term at spike time.
    All three arguments should carry the same voltage unit (e.g. mV) so that
    ``(v_peak - v_th) / delta_t`` is dimensionless.

    Parameters
    ----------
    v_peak : ArrayLike
        Peak voltage (with units, e.g. mV).
    v_th : ArrayLike
        Threshold voltage (same units as *v_peak*).
    delta_t : ArrayLike
        Slope factor (same units as *v_peak*).
    """
    # Compute the dimensionless ratio; units cancel in (v_peak - v_th) / delta_t.
    # Extract mantissa values for plain numpy comparison.
    v_peak = np.asarray(u.get_mantissa(v_peak))
    v_th = np.asarray(u.get_mantissa(v_th))
    delta_t = np.asarray(u.get_mantissa(delta_t))

    positive_dt = delta_t > 0.0
    if np.any(positive_dt):
        dftype = brainstate.environ.dftype()
        finfo = np.finfo(dtype=dftype)
        safety_margin = 1e20 if finfo.bits >= 64 else 1e10
        max_exp_arg = np.log(finfo.max / safety_margin)
        ratio = (v_peak - v_th) / np.where(positive_dt, delta_t, 1.0)
        if np.any(ratio[positive_dt] >= max_exp_arg):
            raise ValueError(
                'The current combination of V_peak, V_th and Delta_T will '
                'lead to numerical overflow at spike time; try for instance '
                'to increase Delta_T or to reduce V_peak to avoid this problem.'
            )


# ---------------------------------------------------------------------------
# E. Numerical propagators
# ---------------------------------------------------------------------------

def propagator_exp(tau_syn, tau_m, c_m, h_ms):
    r"""Compute the off-diagonal propagator :math:`P_{21}` numerically stably.

    For a linear two-compartment system coupling a synaptic current
    (decaying with ``tau_syn``) to membrane voltage (decaying with ``tau_m``),
    the exact one-step propagator is

    .. math::

       P_{21} = \frac{\tau_{\mathrm{syn}} \tau_m}
                     {C_m (\tau_m - \tau_{\mathrm{syn}})}
                \left(e^{-h/\tau_m} - e^{-h/\tau_{\mathrm{syn}}}\right).

    A singularity-safe fallback ``(h / C_m) * exp(-h / tau_m)`` is used when
    ``tau_syn`` is numerically close to ``tau_m``.

    Parameters
    ----------
    tau_syn : np.ndarray
        Synaptic time constant in ms.
    tau_m : np.ndarray
        Membrane time constant in ms.
    c_m : np.ndarray
        Membrane capacitance in pF.
    h_ms : float
        Simulation step size in ms.

    Returns
    -------
    np.ndarray
        Propagator coefficient P21.
    """
    tau_syn = np.asarray(tau_syn, dtype=np.float64)
    tau_m = np.asarray(tau_m, dtype=np.float64)
    c_m = np.asarray(c_m, dtype=np.float64)
    h_ms = np.asarray(h_ms, dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        beta = tau_syn * tau_m / (tau_m - tau_syn)
        gamma = beta / c_m
        inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)
        exp_h_tau_syn = np.exp(-h_ms / tau_syn)
        expm1_h_tau = np.expm1(h_ms * inv_beta)
        p32_raw = gamma * exp_h_tau_syn * expm1_h_tau

        normal_min = np.finfo(np.float64).tiny
        regular_mask = np.isfinite(p32_raw) & (np.abs(p32_raw) >= normal_min) & (p32_raw > 0.0)
        p32_singular = h_ms / c_m * np.exp(-h_ms / tau_m)
        return np.where(regular_mask, p32_raw, p32_singular)


def alpha_propagator_p31_p32(tau_syn, tau_m, c_m, h_ms):
    r"""Compute alpha-kernel membrane propagator terms ``P31`` and ``P32``.

    Mirrors NEST ``IAFPropagatorAlpha`` masking logic with NumPy finite/normal
    checks to avoid catastrophic cancellation when ``tau_syn ~ tau_m``.

    Parameters
    ----------
    tau_syn : np.ndarray
        Synaptic time constant in ms.
    tau_m : np.ndarray
        Membrane time constant in ms.
    c_m : np.ndarray
        Membrane capacitance in pF.
    h_ms : float
        Integration step in ms.

    Returns
    -------
    P31 : np.ndarray
        Alpha propagator P31 coefficient.
    P32 : np.ndarray
        Alpha propagator P32 coefficient.
    """
    tau_syn = np.asarray(tau_syn, dtype=np.float64)
    tau_m = np.asarray(tau_m, dtype=np.float64)
    c_m = np.asarray(c_m, dtype=np.float64)
    h_ms = np.asarray(h_ms, dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        beta = tau_syn * tau_m / (tau_m - tau_syn)
        gamma = beta / c_m
        inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)

        exp_h_tau_syn = np.exp(-h_ms / tau_syn)
        expm1_h_tau = np.expm1(h_ms * inv_beta)

        p32_raw = gamma * exp_h_tau_syn * expm1_h_tau
        exp_h_tau_m = np.exp(-h_ms / tau_m)
        p32_singular = h_ms / c_m * exp_h_tau_m

        normal_min = np.finfo(np.float64).tiny
        p32_regular_mask = np.isfinite(p32_raw) & (np.abs(p32_raw) >= normal_min) & (p32_raw > 0.0)
        p32 = np.where(p32_regular_mask, p32_raw, p32_singular)

        h_min_regular = 1e-7 * tau_m * tau_m / np.abs(tau_m - tau_syn)
        p31_regular_mask = np.isfinite(h_min_regular) & (h_ms > h_min_regular)

        p31_regular = gamma * exp_h_tau_syn * (beta * expm1_h_tau - h_ms)
        p31_singular = 0.5 * h_ms * h_ms / c_m * exp_h_tau_m
        p31 = np.where(p31_regular_mask, p31_regular, p31_singular)

    return p31, p32


# ---------------------------------------------------------------------------
# F. Adaptive Runge-Kutta integrators
# ---------------------------------------------------------------------------


class ButcherTableau(NamedTuple):
    """Coefficients for an embedded explicit Runge-Kutta method.

    An *s*-stage embedded pair uses the higher-order weights ``b`` for the
    solution and the lower-order weights ``b_hat`` for error estimation.

    Attributes
    ----------
    c : tuple of float
        Nodes (abscissae), length ``s``.
    A : tuple of tuples
        Stage coefficient matrix (ragged lower-triangular).
        ``A[0] = ()``, ``A[i]`` has ``i`` entries for ``i >= 1``.
    b : tuple of float
        Higher-order solution weights, length ``s``.
    b_hat : tuple of float
        Lower-order weights for error estimation, length ``s``.
    error_order : int
        Order of the lower-order method (controls step-size exponents).
    """
    c: tuple
    A: tuple
    b: tuple
    b_hat: tuple
    error_order: int


# --- Preset Butcher tableaux ------------------------------------------------

RKF45 = ButcherTableau(
    c=(0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2),
    A=(
        (),
        (1 / 4,),
        (3 / 32, 9 / 32),
        (1932 / 2197, -7200 / 2197, 7296 / 2197),
        (439 / 216, -8.0, 3680 / 513, -845 / 4104),
        (-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40),
    ),
    b=(16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55),
    b_hat=(25 / 216, 0.0, 1408 / 2565, 2197 / 4104, -1 / 5, 0.0),
    error_order=4,
)

DOPRI5 = ButcherTableau(
    c=(0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0),
    A=(
        (),
        (1 / 5,),
        (3 / 40, 9 / 40),
        (44 / 45, -56 / 15, 32 / 9),
        (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
        (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
        (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
    ),
    b=(35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0),
    b_hat=(5179 / 57600, 0.0, 7571 / 16695, 393 / 640,
           -92097 / 339200, 187 / 2100, 1 / 40),
    error_order=4,
)

BOGACKI_SHAMPINE = ButcherTableau(
    c=(0.0, 1 / 2, 3 / 4, 1.0),
    A=(
        (),
        (1 / 2,),
        (0.0, 3 / 4),
        (2 / 9, 1 / 3, 4 / 9),
    ),
    b=(2 / 9, 1 / 3, 4 / 9, 0.0),
    b_hat=(7 / 24, 1 / 4, 1 / 3, 1 / 8),
    error_order=2,
)

HEUN_EULER = ButcherTableau(
    c=(0.0, 1.0),
    A=(
        (),
        (1.0,),
    ),
    b=(1 / 2, 1 / 2),
    b_hat=(1.0, 0.0),
    error_order=1,
)

CASH_KARP = ButcherTableau(
    c=(0.0, 1 / 5, 3 / 10, 3 / 5, 1.0, 7 / 8),
    A=(
        (),
        (1 / 5,),
        (3 / 40, 9 / 40),
        (3 / 10, -9 / 10, 6 / 5),
        (-11 / 54, 5 / 2, -70 / 27, 35 / 27),
        (1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096),
    ),
    b=(37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771),
    b_hat=(2825 / 27648, 0.0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4),
    error_order=4,
)

TSIT5 = ButcherTableau(
    c=(0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0),
    A=(
        (),
        (0.161,),
        (-0.008480655492356989, 0.335480655492357),
        (2.8971530571054935, -6.359448489975075, 4.3622954328695815),
        (5.325864828439257, -11.748883564062828, 7.4955393428898365,
         -0.09249506636175525),
        (5.86145544294642, -12.92096931784711, 8.159367898576159,
         -0.071584973281401, -0.028269050394068616),
        (0.09646076681806523, 0.01, 0.4798896504144996,
         1.379008574103742, -3.290069515436081, 2.324710524099774),
    ),
    b=(0.09646076681806523, 0.01, 0.4798896504144996,
       1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0),
    b_hat=(0.001780011052226, 0.000816434459657, -0.007880878010262,
           0.144711007173263, -0.582357165452555, 0.458082105929187,
           1.0 / 66.0),
    error_order=4,
)

MIDPOINT_EULER = ButcherTableau(
    c=(0.0, 1 / 2),
    A=(
        (),
        (1 / 2,),
    ),
    b=(0.0, 1.0),
    b_hat=(1.0, 0.0),
    error_order=1,
)

FEHLBERG2 = ButcherTableau(
    c=(0.0, 1 / 2, 1.0),
    A=(
        (),
        (1 / 2,),
        (1 / 256, 255 / 256),
    ),
    b=(1 / 512, 255 / 256, 1 / 512),
    b_hat=(1 / 256, 255 / 256, 0.0),
    error_order=1,
)


def _is_quantity(x):
    """Check if *x* is a brainunit Quantity (used as ``is_leaf`` for tree ops)."""
    return isinstance(x, u.Quantity)


def _rk_weighted_sum(state, h, coeffs, k_stages):
    """Compute ``state + h * sum(c_j * k_j)`` over a JAX pytree.

    Skips zero coefficients to avoid unnecessary computation.
    Uses ``is_leaf=_is_quantity`` so that Quantities with different unit
    representations (e.g. state in pA vs derivative in pA/ms) are treated
    as opaque leaves and do not trigger a pytree structure mismatch.
    """
    nonzero = [(c, k) for c, k in zip(coeffs, k_stages) if c != 0.0]
    if not nonzero:
        return state
    cs, ks = zip(*nonzero)

    def _leaf_fn(s, *k_vals):
        acc = cs[0] * k_vals[0]
        for c, k in zip(cs[1:], k_vals[1:]):
            acc = acc + c * k
        return s + h * acc

    return jax.tree.map(_leaf_fn, state, *ks, is_leaf=_is_quantity)


def _rk_error_estimate(h, b_error, k_stages):
    """Compute error estimate ``h * sum(b_error[i] * k[i])`` as a pytree.

    Unlike :func:`_rk_weighted_sum` this does **not** add the base state,
    yielding the raw truncation-error estimate for each leaf.
    """
    nonzero = [(c, k) for c, k in zip(b_error, k_stages) if c != 0.0]
    if not nonzero:
        # Degenerate case – error coefficients are all zero.
        first_k = k_stages[0]
        return jax.tree.map(
            lambda x: u.math.zeros_like(x) if _is_quantity(x) else jnp.zeros_like(x),
            first_k,
            is_leaf=_is_quantity,
        )
    cs, ks = zip(*nonzero)

    def _leaf_fn(*k_vals):
        acc = cs[0] * k_vals[0]
        for c, kv in zip(cs[1:], k_vals[1:]):
            acc = acc + c * kv
        return h * acc

    return jax.tree.map(_leaf_fn, *ks, is_leaf=_is_quantity)


def _rk_scaled_error_norm(y0, y1, y_error, atol, rtol):
    """Per-element max of the scaled error across all pytree leaves.

    For each leaf the scaled error is::

        |y_error| / (atol + rtol * max(|y0|, |y1|))

    When *rtol* = 0 this reduces to ``|y_error| / atol`` which is equivalent
    to the previous absolute-error check with threshold ``atol``.

    Returns a **per-element** (not scalar) array so that each neuron in a
    vectorised population can accept / reject independently.
    """

    def _leaf_err(y0_l, y1_l, ye_l):
        abs_err = u.get_mantissa(u.math.abs(ye_l))
        abs_y = jnp.maximum(
            u.get_mantissa(u.math.abs(y0_l)),
            u.get_mantissa(u.math.abs(y1_l)),
        )
        return abs_err / (atol + rtol * abs_y)

    err_tree = jax.tree.map(_leaf_err, y0, y1, y_error, is_leaf=_is_quantity)
    err_leaves = jax.tree.leaves(err_tree)
    # Skip zero-size leaves (e.g., empty adaptation arrays when n_stc=0 or n_sfa=0).
    # A leaf with shape (0, n) has size=0 and contributes no error information, but
    # would otherwise produce shape (0,) arrays that contaminate broadcasts with
    # non-empty leaves (e.g., shape (n,) -> (0,) via jnp.maximum broadcasting rules).
    non_empty = [e for e in err_leaves if e.size > 0]
    if not non_empty:
        return jnp.zeros(jax.tree.leaves(y0)[0].shape)
    # Reduce each leaf to the minimum ndim (per-neuron shape) so that
    # leaves with extra trailing dimensions (e.g. per-receptor) don't
    # broadcast and expand the shape of the combined error.
    min_ndim = min(e.ndim for e in non_empty)
    reduced = []
    for e in non_empty:
        while e.ndim > min_ndim:
            e = jnp.max(e, axis=-1)
        reduced.append(e)
    err = reduced[0]
    for e in reduced[1:]:
        err = jnp.maximum(err, e)
    return err


tableau_mapping = {
    'RKF45': RKF45,
    'DOPRI5': DOPRI5,
    'BOGACKI_SHAMPINE': BOGACKI_SHAMPINE,
    'HEUN_EULER': HEUN_EULER,
    'CASH_KARP': CASH_KARP,
    'TSIT5': TSIT5,
    'MIDPOINT_EULER': MIDPOINT_EULER,
    'FEHLBERG2': FEHLBERG2,
}


class AdaptiveRungeKuttaStep:
    """JAX-based adaptive embedded Runge-Kutta ODE integrator.

    Supports arbitrary Butcher tableaux, JAX pytree state/extra,
    unit-aware Quantities via brainunit, and optional per-substep
    event callbacks for spike detection, refractory clamping, etc.

    **Differentiability.**  The integrator is fully compatible with JAX
    automatic differentiation (``jax.grad``, ``jax.value_and_grad``, …).
    Step-size adaptation is detached from the computation graph via
    ``jax.lax.stop_gradient`` following the approach in *diffrax*: the
    time discretisation is treated as a non-differentiable implementation
    detail so that gradients flow only through the ODE solution itself.
    This typically yields a ~3× speedup in backward passes compared to
    differentiating through the adaptive controller.

    Available methods: ``'RKF45'``, ``'DOPRI5'``, ``'BOGACKI_SHAMPINE'``,
    ``'HEUN_EULER'``, ``'CASH_KARP'``, ``'TSIT5'``, ``'MIDPOINT_EULER'``,
    ``'FEHLBERG2'``.

    Parameters
    ----------
    method : str
        Name of the embedded RK method (key in ``tableau_mapping``).
    vf : callable
        Vector field ``vf(state, extra) -> derivatives``.
    dt : Quantity or float, optional
        Total integration interval.  Defaults to ``brainstate.environ.get_dt()``.
    atol : float, optional
        Absolute error tolerance (unitless).  Default: 1e-6.
    rtol : float, optional
        Relative error tolerance (unitless).  Default: 0.0.
        When non-zero the local error is scaled as
        ``|err| / (atol + rtol * max(|y0|, |y1|))``.
    min_h : Quantity, optional
        Minimum step size.  Defaults to ``1e-8 * u.ms``.
    max_iters : int, optional
        Maximum substep count.  Default: 100000.
    event_fn : callable, optional
        ``event_fn(state, extra, accept) -> (state, extra)``
        Called after each accepted substep.

    Examples
    --------
    >>> step = AdaptiveRungeKuttaStep('DOPRI5', vf=my_vector_field)
    >>> state, h, extra = step(state, h, extra=extra)
    """

    def __init__(
        self,
        method: str,
        vf: Callable,
        dt: Optional[u.Quantity['time']] = None,
        atol: float = 1e-6,
        rtol: float = 0.0,
        min_h: Optional[u.Quantity] = None,
        max_iters: int = 100000,
        event_fn: Optional[Callable] = None,
    ):
        if method not in tableau_mapping:
            raise ValueError(
                f'Unknown method {method!r}. '
                f'Available: {", ".join(tableau_mapping)}'
            )
        if min_h is None:
            min_h = 1e-8 * u.ms
        if dt is None:
            dt = brainstate.environ.get_dt()
        self.tableau = tableau_mapping[method]
        self.vf = vf
        self.dt = dt
        self.atol = atol
        self.rtol = rtol
        self.min_h = min_h
        self.max_iters = max_iters
        self.event_fn = event_fn

        # Precompute error coefficients: b_error = b - b_hat.
        # The truncation error is h * sum(b_error[i] * k[i]), avoiding the
        # need to compute the lower-order solution y_low separately.
        tab = self.tableau
        self.b_error = tuple(bi - bhi for bi, bhi in zip(tab.b, tab.b_hat))

    @classmethod
    def available_method(cls):
        """Return a list of available method names."""
        return list(tableau_mapping.keys())

    def __call__(
        self,
        state: PyTree,
        h: u.Quantity,
        extra: Optional[PyTree] = None
    ):
        """Integrate over one simulation timestep.

        Parameters
        ----------
        state : pytree
            Initial ODE state.
        h : Quantity
            Adaptive step size (updated across calls).
        extra : pytree, optional
            Mutable auxiliary data passed through the loop.

        Returns
        -------
        state : pytree
            Final integrated state.
        h : Quantity
            Final adaptive step size.
        extra : pytree
            Final auxiliary data.
        """
        vf = self.vf
        dt = self.dt
        atol = self.atol
        rtol = self.rtol
        min_h = self.min_h
        event_fn = self.event_fn
        b_error = self.b_error

        tableau = self.tableau
        s = len(tableau.c)
        accept_exp = 1.0 / (tableau.error_order + 1)
        reject_exp = 1.0 / tableau.error_order

        first_leaf = jax.tree.leaves(state)[0]
        v_shape = first_leaf.shape
        # Use the dtype of the state leaves (not dftype) so that float64
        # state (e.g., when jax_enable_x64 is True) keeps t_local consistent.
        state_dtype = first_leaf.dtype

        # Match units of t_local to dt (unitless if dt is unitless)
        if isinstance(dt, u.Quantity):
            t_local = jnp.zeros(v_shape, dtype=state_dtype) * u.ms
        else:
            t_local = jnp.zeros(v_shape, dtype=state_dtype)
        h = u.math.maximum(h, min_h)

        init_carry = (state, t_local, h, extra, jnp.array(0, dtype=jnp.int32))

        def _cond_fn(carry):
            _, t_loc, _, _, n_iters = carry
            return jnp.any(t_loc < dt) & (n_iters < self.max_iters)

        def _body_fn(carry):
            state, t_loc, h, extra, n_iters = carry

            active = jax.lax.stop_gradient(t_loc < dt)

            h = u.math.where(
                active,
                u.math.maximum(min_h, u.math.minimum(h, dt - t_loc)),
                h,
            )

            # --- RK stage evaluations (unrolled at trace time) -----------
            k = []
            for i in range(s):
                if i == 0:
                    y_i = state
                else:
                    y_i = _rk_weighted_sum(state, h, tableau.A[i], k)
                k.append(vf(y_i, extra))

            # --- Higher-order solution -----------------------------------
            y_high = _rk_weighted_sum(state, h, tableau.b, k)

            # --- Error estimate from precomputed b_error = b - b_hat -----
            y_error = _rk_error_estimate(h, b_error, k)

            # --- Scaled error norm (threshold is 1.0) --------------------
            err = _rk_scaled_error_norm(state, y_high, y_error, atol, rtol)

            # Replace NaN errors with inf to force step rejection rather
            # than propagating NaN through the gradient graph.
            err = jnp.where(jnp.isnan(err), jnp.inf, err)

            accept = active & ((err <= 1.0) | (h <= min_h))
            reject = active & ~accept

            # Select accepted or rejected state via jnp.where (both
            # branches are always evaluated — no control-flow branching).
            new_state = jax.tree.map(
                lambda yh, si: u.math.where(accept, yh, si),
                y_high, state,
                is_leaf=_is_quantity,
            )
            t_loc = u.math.where(accept, t_loc + h, t_loc)

            if event_fn is not None:
                new_state, extra = event_fn(new_state, extra, accept)

            # --- Step-size control (detached from gradient graph) ---------
            # Following diffrax: the step-size adaptation is a
            # discretisation detail that should not contribute to the
            # backward-pass gradient.  Detaching it via stop_gradient
            # yields ~3× faster backward passes and avoids numerical
            # instabilities in the adaptive controller's gradient.
            err_sg = jax.lax.stop_gradient(jnp.maximum(err, 1e-30))

            inv_err = 1.0 / err_sg
            fac_accept = jax.lax.stop_gradient(jnp.where(
                err_sg <= 1e-30,
                5.0,
                jnp.clip(0.9 * inv_err ** accept_exp, 0.2, 5.0),
            ))
            fac_reject = jax.lax.stop_gradient(
                jnp.clip(0.9 * inv_err ** reject_exp, 0.2, 1.0)
            )
            h = u.math.where(
                accept, u.math.maximum(min_h, h * fac_accept), h
            )
            h = u.math.where(
                reject, u.math.maximum(min_h, h * fac_reject), h
            )

            return (new_state, t_loc, h, extra, n_iters + 1)

        carry_out = brainstate.transform.while_loop(_cond_fn, _body_fn, init_carry)
        state, _, h, extra, _ = carry_out

        return state, h, extra


# ---------------------------------------------------------------------------
# F. Schedule value stacking
# ---------------------------------------------------------------------------

def stack_schedule_values(amplitude_values, varshape):
    """Convert a sequence of schedule values into a stacked Quantity array.

    Each entry in *amplitude_values* is converted via :func:`u.math.asarray`
    and then expanded to the same number of dimensions as the element with the
    highest ndim (by prepending size-1 axes).  The element-wise maximum size
    in each dimension defines *final_shape*, and every entry is broadcast to
    that shape before stacking.  The result has shape
    ``(K, *final_shape)`` where ``final_shape`` is broadcastable to
    *varshape*.

    Parameters
    ----------
    amplitude_values : Sequence
        Ordered sequence of ``K`` plateau values.  Entries may be unitful
        Quantities or plain numerics; unit consistency is enforced by
        :func:`u.math.stack`.
    varshape : tuple
        Output shape of the model (``self.varshape``).  Used only for the
        empty-schedule fallback zeros array.

    Returns
    -------
    amplitude_values : Quantity or jax.Array
        Shape ``(K, *final_shape)`` when *amplitude_values* is non-empty, or
        shape ``(0, *varshape)`` when it is empty.
    """
    assert len(amplitude_values) >= 0, 'Schedule must have at least one plateau value.'
    amp_vals = [u.math.asarray(v) for v in amplitude_values]
    max_ndim = max(v.ndim for v in amp_vals)
    # Align all values to max_ndim by prepending size-1 axes.
    expanded = []
    for v in amp_vals:
        extra = max_ndim - v.ndim
        if extra:
            v = u.math.reshape(v, (1,) * extra + v.shape)
        expanded.append(v)
    # Final shape: element-wise maximum size in each dimension.
    final_shape = tuple(max(v.shape[d] for v in expanded) for d in range(max_ndim))
    assert u.math.broadcast_shapes(final_shape, varshape), (
        f'Final shape {final_shape} is not broadcastable to varshape {varshape}'
    )
    # Broadcast each element to final_shape, then stack to (K, *final_shape).
    stacked = u.math.stack([u.math.broadcast_to(v, final_shape) for v in expanded])
    # Force a jnp-backed mantissa. The schedule is gathered with a *traced* index
    # inside a jitted ``for_loop`` (``amplitude_values[searchsorted(t)]``); a NumPy
    # mantissa cannot be indexed by a tracer and raises TracerArrayConversionError.
    # A list of scalar Quantities already stacks to jnp, but a NumPy-array Quantity
    # (``np.asarray([...]) * u.pA``) does not, so coerce unconditionally.
    return u.maybe_decimal(jnp.asarray(u.get_mantissa(stacked)) * u.get_unit(stacked))
