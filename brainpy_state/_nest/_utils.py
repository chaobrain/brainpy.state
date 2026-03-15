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
parameter) and operate on plain NumPy / JAX arrays or saiunit quantities.
"""

from typing import Callable, NamedTuple, Optional

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u
from brainstate.typing import PyTree
from jax.interpreters.partial_eval import DynamicJaxprTracer

__all__ = [
    'to_numpy',
    'to_numpy_unitless',
    'broadcast_to_state',
    'refractory_counts',
    'get_spike_scaled',
    'check_positive',
    'check_non_negative',
    'check_reset_below_threshold',
    'validate_aeif_overflow',
    'propagator_exp',
    'alpha_propagator_p31_p32',
    'rkf45_integrate',
    'ButcherTableau',
    'AdaptiveRungeKuttaStep',
    'sum_signed_delta_inputs',
    'time_window_gate',
    'stack_schedule_values',
]


def is_tracer(x):
    return isinstance(x, (jax.ShapeDtypeStruct, jax.core.ShapedArray, DynamicJaxprTracer, jax.core.Tracer))


# ---------------------------------------------------------------------------
# A. Conversion helpers
# ---------------------------------------------------------------------------

def to_numpy(x, unit):
    """Convert a saiunit quantity to a unitless float64 NumPy array.

    Parameters
    ----------
    x : ArrayLike
        Quantity with units (e.g. ``V_th`` in mV).
    unit : saiunit.Unit
        Unit to divide by before stripping (e.g. ``u.mV``).

    Returns
    -------
    np.ndarray
        Unitless float64 array.
    """
    dftype = brainstate.environ.dftype()
    return np.asarray(u.math.asarray(x / unit), dtype=dftype)


def to_numpy_unitless(x):
    """Convert an array-like to a unitless float64 NumPy array (no unit division).

    Parameters
    ----------
    x : ArrayLike
        Value to convert (already unitless or with units stripped).

    Returns
    -------
    np.ndarray
        Float64 array.
    """
    dftype = brainstate.environ.dftype()
    return np.asarray(u.math.asarray(x), dtype=dftype)


def broadcast_to_state(x_np, shape):
    """Broadcast a NumPy array to the target state shape.

    Parameters
    ----------
    x_np : np.ndarray
        Source array.
    shape : tuple
        Target shape (typically ``self.V.value.shape``).

    Returns
    -------
    np.ndarray
        Broadcast view with the given shape.
    """
    return np.broadcast_to(x_np, shape)


# ---------------------------------------------------------------------------
# B. Refractory computation
# ---------------------------------------------------------------------------

def refractory_counts(t_ref):
    """Convert refractory duration to integer simulation-step counts.

    Computes ``ceil(t_ref / dt)`` using the current environment ``dt``.

    Parameters
    ----------
    t_ref : ArrayLike
        Refractory period with time units.

    Returns
    -------
    jnp.ndarray
        Integer step count (int32).
    """
    dt = brainstate.environ.get_dt()
    ditype = brainstate.environ.ditype()
    return u.math.asarray(u.math.ceil(t_ref / dt), dtype=ditype)


# ---------------------------------------------------------------------------
# C. Spike detection
# ---------------------------------------------------------------------------

def get_spike_scaled(V, V_th, V_reset, spk_fun):
    """Compute differentiable spike output using surrogate gradient.

    Scales the voltage as ``(V - V_th) / (V_th - V_reset)`` and passes
    the result through the surrogate function.

    Parameters
    ----------
    V : ArrayLike
        Membrane potential (with voltage units).
    V_th : ArrayLike
        Spike threshold.
    V_reset : ArrayLike
        Reset potential.
    spk_fun : Callable
        Surrogate gradient function.

    Returns
    -------
    ArrayLike
        Surrogate spike output.
    """
    v_scaled = (V - V_th) / (V_th - V_reset)
    return spk_fun(v_scaled)


# ---------------------------------------------------------------------------
# D. Parameter validation
# ---------------------------------------------------------------------------

def check_positive(value, unit, name):
    """Raise ``ValueError`` if any element of *value* is not strictly positive.

    Parameters
    ----------
    value : ArrayLike
        Parameter value with units (compared directly via ``value <= 0 * unit``).
    unit : saiunit.Unit
        Unit for the zero comparison threshold.
    name : str
        Human-readable parameter name for the error message.
    """
    if np.any(value <= 0.0 * unit):
        raise ValueError(f'{name} must be strictly positive.')


def check_non_negative(value, unit, name):
    """Raise ``ValueError`` if any element of *value* is negative.

    Parameters
    ----------
    value : ArrayLike
        Parameter value with units (compared directly via ``value < 0 * unit``).
    unit : saiunit.Unit
        Unit for the zero comparison threshold.
    name : str
        Human-readable parameter name for the error message.
    """
    if np.any(value < 0.0 * unit):
        raise ValueError(f'{name} must not be negative.')


def check_reset_below_threshold(V_reset, V_th):
    """Raise ``ValueError`` if ``V_reset >= V_th`` for any element.

    Parameters
    ----------
    V_reset : ArrayLike
        Reset potential (with units).
    V_th : ArrayLike
        Threshold potential (same units as V_reset).
    """
    if np.any(V_reset >= V_th):
        raise ValueError('Reset potential must be smaller than threshold.')


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


def rkf45_integrate(dynamics_fn, y0, dt, h0, atol=1e-3, min_h=1e-8, max_iters=10000):
    r"""Integrate an ODE system for one simulation timestep using RKF45.

    Implements Runge-Kutta-Fehlberg 4(5) with embedded error estimation and
    automatic step size control.

    Parameters
    ----------
    dynamics_fn : callable
        Function ``dynamics_fn(*y) -> tuple`` returning time derivatives.
        Each element of *y* and the return tuple must be a scalar float.
    y0 : tuple of float
        Initial state values (e.g. ``(v, ge, gi)``).
    dt : float
        Target integration interval in ms.
    h0 : float
        Initial / previous adaptive step size in ms.
    atol : float, optional
        Absolute error tolerance. Default: 1e-3.
    min_h : float, optional
        Minimum step size in ms. Default: 1e-8.
    max_iters : int, optional
        Maximum iteration count. Default: 10000.

    Returns
    -------
    y_final : tuple of float
        Final state values after integrating over ``[0, dt]``.
    h_final : float
        Final adaptive step size for the next call.
    """
    n = len(y0)
    t = 0.0
    h = max(h0, min_h)
    y = list(y0)
    iters = 0

    while t < dt and iters < max_iters:
        iters += 1
        h = min(h, dt - t)
        h = max(h, min_h)

        k1 = dynamics_fn(*y)
        y2 = tuple(y[i] + h * k1[i] / 4.0 for i in range(n))
        k2 = dynamics_fn(*y2)
        y3 = tuple(y[i] + h * (3.0 * k1[i] / 32.0 + 9.0 * k2[i] / 32.0) for i in range(n))
        k3 = dynamics_fn(*y3)
        y4 = tuple(
            y[i] + h * (1932.0 * k1[i] / 2197.0 - 7200.0 * k2[i] / 2197.0 + 7296.0 * k3[i] / 2197.0)
            for i in range(n)
        )
        k4 = dynamics_fn(*y4)
        y5 = tuple(
            y[i] + h * (439.0 * k1[i] / 216.0 - 8.0 * k2[i] + 3680.0 * k3[i] / 513.0 - 845.0 * k4[i] / 4104.0)
            for i in range(n)
        )
        k5 = dynamics_fn(*y5)
        y6 = tuple(
            y[i] + h * (
                -8.0 * k1[i] / 27.0 + 2.0 * k2[i] - 3544.0 * k3[i] / 2565.0
                + 1859.0 * k4[i] / 4104.0 - 11.0 * k5[i] / 40.0
            )
            for i in range(n)
        )
        k6 = dynamics_fn(*y6)

        y4_sol = tuple(
            y[i] + h * (25.0 * k1[i] / 216.0 + 1408.0 * k3[i] / 2565.0 + 2197.0 * k4[i] / 4104.0 - k5[i] / 5.0)
            for i in range(n)
        )
        y5_sol = tuple(
            y[i] + h * (
                16.0 * k1[i] / 135.0 + 6656.0 * k3[i] / 12825.0 + 28561.0 * k4[i] / 56430.0
                - 9.0 * k5[i] / 50.0 + 2.0 * k6[i] / 55.0
            )
            for i in range(n)
        )

        err = max(abs(y5_sol[i] - y4_sol[i]) for i in range(n))

        if err <= atol or h <= min_h:
            y = list(y5_sol)
            t += h
            if err == 0.0:
                fac = 5.0
            else:
                fac = 0.9 * (atol / err) ** 0.2
                fac = min(5.0, max(0.2, fac))
            h = max(min_h, h * fac)
        else:
            fac = 0.9 * (atol / err) ** 0.25
            fac = min(1.0, max(0.2, fac))
            h = max(min_h, h * fac)

    return tuple(y), h


def _is_quantity(x):
    """Check if *x* is a saiunit Quantity (used as ``is_leaf`` for tree ops)."""
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


def _rk_max_error(y_high, y_low):
    """Max absolute error across all pytree leaves (unitless)."""

    def _leaf_err(yh, yl):
        return u.get_mantissa(u.math.abs(yh - yl))

    err_tree = jax.tree.map(_leaf_err, y_high, y_low, is_leaf=_is_quantity)
    err_leaves = jax.tree.leaves(err_tree)
    err = err_leaves[0]
    for e in err_leaves[1:]:
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
    unit-aware Quantities via saiunit, and optional per-substep
    event callbacks for spike detection, refractory clamping, etc.

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
        dt=None,
        atol: float = 1e-6,
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
        self.min_h = min_h
        self.max_iters = max_iters
        self.event_fn = event_fn

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
        min_h = self.min_h
        event_fn = self.event_fn

        tableau = self.tableau
        s = len(tableau.c)
        accept_exp = 1.0 / (tableau.error_order + 1)
        reject_exp = 1.0 / tableau.error_order

        first_leaf = jax.tree.leaves(state)[0]
        v_shape = first_leaf.shape
        dftype = brainstate.environ.dftype()

        t_local = jnp.zeros(v_shape, dtype=dftype) * u.ms
        h = u.math.maximum(h, min_h)

        init_carry = (state, t_local, h, extra, jnp.array(0, dtype=jnp.int32))

        def _cond_fn(carry):
            _, t_loc, _, _, n_iters = carry
            return jnp.any(t_loc < dt) & (n_iters < self.max_iters)

        def _body_fn(carry):
            state, t_loc, h, extra, n_iters = carry

            active = t_loc < dt

            h = u.math.where(
                active,
                u.math.maximum(min_h, u.math.minimum(h, dt - t_loc)),
                h,
            )

            k = []
            for i in range(s):
                if i == 0:
                    y_i = state
                else:
                    y_i = _rk_weighted_sum(state, h, tableau.A[i], k)
                k.append(vf(y_i, extra))

            y_high = _rk_weighted_sum(state, h, tableau.b, k)
            y_low = _rk_weighted_sum(state, h, tableau.b_hat, k)

            err = _rk_max_error(y_high, y_low)

            accept = active & ((err <= atol) | (h <= min_h))
            reject = active & ~accept

            new_state = jax.tree.map(
                lambda yh, si: u.math.where(accept, yh, si),
                y_high, state,
                is_leaf=_is_quantity,
            )
            t_loc = u.math.where(accept, t_loc + h, t_loc)

            if event_fn is not None:
                new_state, extra = event_fn(new_state, extra, accept)

            err_safe = jnp.maximum(err, 1e-30)
            fac_accept = jnp.where(
                err == 0.0,
                5.0,
                jnp.minimum(5.0, jnp.maximum(
                    0.2, 0.9 * (atol / err_safe) ** accept_exp
                )),
            )
            fac_reject = jnp.minimum(
                1.0, jnp.maximum(0.2, 0.9 * (atol / err_safe) ** reject_exp)
            )
            h = u.math.where(
                accept, u.math.maximum(min_h, h * fac_accept), h
            )
            h = u.math.where(
                reject, u.math.maximum(min_h, h * fac_reject), h
            )

            return (new_state, t_loc, h, extra, n_iters + 1)

        carry_out = jax.lax.while_loop(_cond_fn, _body_fn, init_carry)
        state, _, h, extra, _ = carry_out

        return state, h, extra


# ---------------------------------------------------------------------------
# G. Conductance input splitting
# ---------------------------------------------------------------------------

def sum_signed_delta_inputs(delta_inputs, zero_ex, zero_in):
    """Split delta inputs by sign into excitatory and inhibitory components.

    Positive values go to excitatory, negative (absolute) values go to
    inhibitory.

    Parameters
    ----------
    delta_inputs : dict or None
        Dictionary of delta input labels to values or callables.
    zero_ex : ArrayLike
        Zero-valued array matching excitatory conductance shape/units.
    zero_in : ArrayLike
        Zero-valued array matching inhibitory conductance shape/units.

    Returns
    -------
    g_ex : ArrayLike
        Sum of positive delta inputs.
    g_in : ArrayLike
        Sum of absolute negative delta inputs.
    """
    g_ex = zero_ex
    g_in = zero_in
    if delta_inputs is None:
        return g_ex, g_in

    for key in tuple(delta_inputs.keys()):
        out = delta_inputs[key]
        if callable(out):
            out = out()
        else:
            delta_inputs.pop(key)

        zero = u.math.zeros_like(out)
        g_ex = g_ex + u.math.maximum(out, zero)
        g_in = g_in + u.math.maximum(-out, zero)
    return g_ex, g_in


# ---------------------------------------------------------------------------
# H. Generator time-window gating
# ---------------------------------------------------------------------------

def time_window_gate(value, origin, start, stop):
    """Apply NEST-style half-open ``[start, stop)`` window gating to a value.

    Parameters
    ----------
    value : ArrayLike
        The signal to gate (e.g. current amplitude).
    origin : ArrayLike
        Time origin added to start/stop.
    start : ArrayLike
        Relative start time (inclusive).
    stop : ArrayLike or None
        Relative stop time (exclusive). None means no upper bound.

    Returns
    -------
    ArrayLike
        ``value`` where the window is active, zero elsewhere.
    """
    t = brainstate.environ.get('t')
    t_start = origin + start
    if stop is not None:
        t_stop = origin + stop
        active = u.math.logical_and(t >= t_start, t < t_stop)
    else:
        active = t >= t_start
    return u.math.where(active, value, u.math.zeros_like(value))


# ---------------------------------------------------------------------------
# I. Schedule value stacking
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
    return u.math.stack([u.math.broadcast_to(v, final_shape) for v in expanded])
