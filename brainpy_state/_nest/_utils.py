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

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u
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
    'propagator_exp',
    'alpha_propagator_p31_p32',
    'rkf45_integrate',
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
        Parameter value with units.
    unit : saiunit.Unit
        Unit for conversion.
    name : str
        Human-readable parameter name for the error message.
    """
    if np.any(to_numpy(value, unit) <= 0.0):
        raise ValueError(f'{name} must be strictly positive.')


def check_non_negative(value, unit, name):
    """Raise ``ValueError`` if any element of *value* is negative.

    Parameters
    ----------
    value : ArrayLike
        Parameter value with units.
    unit : saiunit.Unit
        Unit for conversion.
    name : str
        Human-readable parameter name for the error message.
    """
    if np.any(to_numpy(value, unit) < 0.0):
        raise ValueError(f'{name} must not be negative.')


def check_reset_below_threshold(V_reset, V_th):
    """Raise ``ValueError`` if ``V_reset >= V_th`` for any element.

    Parameters
    ----------
    V_reset : ArrayLike
        Reset potential (mV).
    V_th : ArrayLike
        Threshold potential (mV).
    """
    if np.any(to_numpy(V_reset, u.mV) >= to_numpy(V_th, u.mV)):
        raise ValueError('Reset potential must be smaller than threshold.')


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
# F. RKF45 adaptive integrator
# ---------------------------------------------------------------------------

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
