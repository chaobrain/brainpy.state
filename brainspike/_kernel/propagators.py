# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Exact-integration propagators in JAX, float32-safe and differentiable.

JAX ports of the NumPy (float64-only) propagators in
``brainspike/_nest_base/utils.py`` (``propagator_exp``,
``alpha_propagator_p31_p32``) and of the coefficient set built by
``iaf_psc_alpha._precompute_propagators``. All functions take **traced**
parameters and step size, so they are evaluated inside ``step`` (spec §4.1) and
can be differentiated with respect to every argument.

Formulation
-----------
With :math:`a = h/\tau_{syn}`, :math:`b = h/\tau_m` and :math:`x = a - b`,
the membrane coefficients of the linear PSC systems are

.. math::

   P_{32} = \frac{h}{C_m}\, e^{-a}\, \varphi_1(x), \qquad
   P_{31} = \frac{h^2}{C_m}\, e^{-a}\, \varphi_2(x),

with :math:`\varphi_1(x) = (e^x - 1)/x` and
:math:`\varphi_2(x) = (e^x - 1 - x)/x^2`. This is algebraically identical to the
NEST expressions :math:`\gamma e^{-a}\,\mathrm{expm1}(x)` and
:math:`\gamma e^{-a}(\beta\,\mathrm{expm1}(x) - h)` with
:math:`\beta = \tau_{syn}\tau_m/(\tau_m-\tau_{syn})`, :math:`\gamma=\beta/C_m`,
but has no :math:`1/(\tau_m-\tau_{syn})` factor: at :math:`\tau_m = \tau_{syn}`
it reduces exactly to NEST's singular fallbacks :math:`(h/C_m)e^{-h/\tau_m}` and
:math:`\tfrac12 (h^2/C_m) e^{-h/\tau_m}`, and near it it is accurate to a few ulp
(the NumPy original loses up to :math:`\epsilon/x^2` relative accuracy in
:math:`P_{31}` there, see the tests).

:math:`\varphi_1, \varphi_2` use a Taylor series for :math:`|x| < 1/2` and the
closed form elsewhere, selected with the *double-where* trick (the closed form is
evaluated at a safe point inside the series region), so values **and gradients**
are finite at the singularity. For :math:`|x| \ge 1/2` the products
:math:`e^{-a}\varphi_k(x)` are evaluated as differences of
:math:`e^{-a}, e^{-b}` so that they neither overflow nor produce ``0 * inf``
when :math:`e^{-a}` underflows (very small :math:`\tau_{syn}`).

Units
-----
Inputs may be :class:`brainunit.Quantity` (converted to ms / pF) or plain
mantissas already in ms / pF. Outputs are plain arrays: :math:`P_{32}` in
mV/pA (= ms/pF), :math:`P_{31}` in mV/(pA/ms) (= ms²/pF), :math:`P_{30}` in
mV/pA, :math:`P_{21}` in ms, the rest dimensionless.

Known deviation from NEST
-------------------------
When :math:`e^{-h/\tau_{syn}}` underflows in the NumPy original (e.g.
:math:`\tau_{syn} \lesssim h/700` in float64), its raw :math:`P_{32}` becomes
``0 * inf`` and the singular fallback :math:`(h/C_m)e^{-h/\tau_m}` is returned,
which is not the limit of the exact propagator. This module returns the
mathematically correct value there (the regime is far outside NEST's
documented use).
"""
from __future__ import annotations

import math
from typing import Any

import brainunit as u
import jax.numpy as jnp

from brainspike._kernel.units import to_mantissa

__all__ = [
    'phi1',
    'phi2',
    'exp_decay',
    'propagator_exp',
    'alpha_propagator_p31_p32',
    'iaf_psc_alpha_propagators',
]

#: |x| below which the Taylor series of phi1 / phi2 is used.
_SERIES_CUT = 0.5
#: Number of Taylor terms: truncation error < 0.5**N / (N + 2)! ~ 1e-19 at N = 14.
_N_TERMS = 14


def _as_ms(x: Any) -> Any:
    return to_mantissa(x, u.ms) if isinstance(x, u.Quantity) else jnp.asarray(x)


def _as_pF(x: Any) -> Any:
    return to_mantissa(x, u.pF) if isinstance(x, u.Quantity) else jnp.asarray(x)


def _float_args(*xs: Any) -> list:
    """Promote all arguments to a common floating dtype (float32 stays float32)."""
    dtype = jnp.promote_types(jnp.result_type(*xs), jnp.float32)
    return [jnp.asarray(x, dtype) for x in xs]


def _series(x: Any, offset: int) -> Any:
    """``sum_{n>=0} x**n / (n + offset)!`` by Horner's rule (``_N_TERMS`` terms)."""
    coef = [1.0 / math.factorial(n + offset) for n in range(_N_TERMS)]
    acc = jnp.full_like(x, coef[-1])
    for c in reversed(coef[:-1]):
        acc = acc * x + c
    return acc


def phi1(x: Any) -> Any:
    r"""Stable :math:`\varphi_1(x) = (e^x - 1)/x` with :math:`\varphi_1(0) = 1`.

    Parameters
    ----------
    x : array_like
        Argument (any float dtype).

    Returns
    -------
    jax.Array
        :math:`\varphi_1(x)`; finite value and gradient at ``x == 0``.
    """
    (x,) = _float_args(x)
    small = jnp.abs(x) < _SERIES_CUT
    x_safe = jnp.where(small, 1.0, x)  # double-where: closed form never sees x ~ 0
    return jnp.where(small, _series(jnp.where(small, x, 0.0), 1), jnp.expm1(x_safe) / x_safe)


def phi2(x: Any) -> Any:
    r"""Stable :math:`\varphi_2(x) = (e^x - 1 - x)/x^2` with :math:`\varphi_2(0) = 1/2`.

    Parameters
    ----------
    x : array_like
        Argument (any float dtype).

    Returns
    -------
    jax.Array
        :math:`\varphi_2(x)`; finite value and gradient at ``x == 0``.
    """
    (x,) = _float_args(x)
    small = jnp.abs(x) < _SERIES_CUT
    x_safe = jnp.where(small, 1.0, x)
    closed = (jnp.expm1(x_safe) - x_safe) / (x_safe * x_safe)
    return jnp.where(small, _series(jnp.where(small, x, 0.0), 2), closed)


def _exp_phi1(a: Any, b: Any) -> Any:
    r""":math:`e^{-a}\varphi_1(a - b) = (e^{-b} - e^{-a})/(a - b)`, overflow-free."""
    x = a - b
    small = jnp.abs(x) < _SERIES_CUT
    x_safe = jnp.where(small, 1.0, x)
    near = jnp.exp(-a) * _series(jnp.where(small, x, 0.0), 1)
    far = (jnp.exp(-b) - jnp.exp(-a)) / x_safe
    return jnp.where(small, near, far)


def _exp_phi2(a: Any, b: Any) -> Any:
    r""":math:`e^{-a}\varphi_2(a - b) = (e^{-b} - e^{-a}(1 + x))/x^2`, overflow-free."""
    x = a - b
    small = jnp.abs(x) < _SERIES_CUT
    x_safe = jnp.where(small, 1.0, x)
    near = jnp.exp(-a) * _series(jnp.where(small, x, 0.0), 2)
    far = (jnp.exp(-b) - jnp.exp(-a) * (1.0 + x_safe)) / (x_safe * x_safe)
    return jnp.where(small, near, far)


def exp_decay(tau: Any, h: Any) -> Any:
    r"""Decay factor :math:`e^{-h/\tau}`.

    Parameters
    ----------
    tau, h : array_like or brainunit.Quantity
        Time constant and step (ms when given as mantissas).

    Returns
    -------
    jax.Array
    """
    tau, h = _float_args(_as_ms(tau), _as_ms(h))
    return jnp.exp(-h / tau)


def propagator_exp(tau_syn: Any, tau_m: Any, c_m: Any, h: Any) -> Any:
    r"""Membrane response to an exponentially decaying current, :math:`P_{21}`/:math:`P_{32}`.

    .. math::

       P = \frac{\tau_{syn}\tau_m}{C_m(\tau_m-\tau_{syn})}
           \left(e^{-h/\tau_m} - e^{-h/\tau_{syn}}\right)
         = \frac{h}{C_m} e^{-h/\tau_{syn}} \varphi_1\!\left(\frac{h}{\tau_{syn}}
           - \frac{h}{\tau_m}\right),

    i.e. the voltage change (mV) after one step of length ``h`` caused by a unit
    (1 pA) synaptic current decaying with ``tau_syn``. Equals NEST's singular
    fallback :math:`(h/C_m)e^{-h/\tau_m}` at :math:`\tau_m = \tau_{syn}`.

    Parameters
    ----------
    tau_syn, tau_m : array_like or brainunit.Quantity
        Synaptic and membrane time constants (ms), broadcastable.
    c_m : array_like or brainunit.Quantity
        Membrane capacitance (pF).
    h : array_like or brainunit.Quantity
        Step size (ms); may be traced.

    Returns
    -------
    jax.Array
        Coefficient in mV/pA, broadcast shape, float32 unless an input is float64.
    """
    tau_syn, tau_m, c_m, h = _float_args(_as_ms(tau_syn), _as_ms(tau_m), _as_pF(c_m), _as_ms(h))
    return h / c_m * _exp_phi1(h / tau_syn, h / tau_m)


def alpha_propagator_p31_p32(tau_syn: Any, tau_m: Any, c_m: Any, h: Any) -> tuple[Any, Any]:
    r"""Membrane coefficients :math:`P_{31}, P_{32}` of the alpha-shaped PSC system.

    For the state :math:`(dI, I, V)` of ``iaf_psc_alpha``, :math:`P_{31}` maps
    :math:`dI` (pA/ms) and :math:`P_{32}` maps :math:`I` (pA) to the voltage
    change over one step (see the module docstring for the formulas). Mirrors
    ``brainspike._nest_base.utils.alpha_propagator_p31_p32`` (NEST
    ``IAFPropagatorAlpha``) including its limits at :math:`\tau_m = \tau_{syn}`.

    Parameters
    ----------
    tau_syn, tau_m : array_like or brainunit.Quantity
        Synaptic and membrane time constants (ms), broadcastable.
    c_m : array_like or brainunit.Quantity
        Membrane capacitance (pF).
    h : array_like or brainunit.Quantity
        Step size (ms); may be traced.

    Returns
    -------
    P31 : jax.Array
        In mV/(pA/ms).
    P32 : jax.Array
        In mV/pA.
    """
    tau_syn, tau_m, c_m, h = _float_args(_as_ms(tau_syn), _as_ms(tau_m), _as_pF(c_m), _as_ms(h))
    a, b = h / tau_syn, h / tau_m
    p32 = h / c_m * _exp_phi1(a, b)
    p31 = h * h / c_m * _exp_phi2(a, b)
    return p31, p32


def iaf_psc_alpha_propagators(tau_m: Any, tau_syn_ex: Any, tau_syn_in: Any, c_m: Any,
                              h: Any) -> dict[str, Any]:
    r"""All step coefficients of ``iaf_psc_alpha`` (NEST exact integration).

    JAX counterpart of ``iaf_psc_alpha._precompute_propagators``
    (``_nest_neuron/iaf_psc_alpha.py``), computed from traced values.

    Parameters
    ----------
    tau_m, tau_syn_ex, tau_syn_in : array_like or brainunit.Quantity
        Membrane and synaptic time constants (ms).
    c_m : array_like or brainunit.Quantity
        Membrane capacitance (pF).
    h : array_like or brainunit.Quantity
        Step size (ms).

    Returns
    -------
    dict of str to jax.Array
        ``P11_ex, P22_ex, P21_ex, P11_in, P22_in, P21_in`` (synaptic alpha
        kernels; ``P21`` in ms), ``expm1_tau_m`` (:math:`e^{-h/\tau_m}-1`),
        ``P30`` (mV/pA), ``P31_ex, P32_ex, P31_in, P32_in`` (see
        :func:`alpha_propagator_p31_p32`) and ``psc_init_ex, psc_init_in``
        (:math:`e/\tau_{syn}`, 1/ms) that scale a weight (pA) into the jump of
        :math:`dI`.
    """
    tau_m, tau_ex, tau_in, c_m, h = _float_args(
        _as_ms(tau_m), _as_ms(tau_syn_ex), _as_ms(tau_syn_in), _as_pF(c_m), _as_ms(h))
    out = {}
    for tag, tau in (('ex', tau_ex), ('in', tau_in)):
        p11 = jnp.exp(-h / tau)
        out[f'P11_{tag}'] = p11
        out[f'P22_{tag}'] = p11
        out[f'P21_{tag}'] = h * p11
    expm1_tau_m = jnp.expm1(-h / tau_m)
    out['expm1_tau_m'] = expm1_tau_m
    out['P30'] = -tau_m / c_m * expm1_tau_m
    out['P31_ex'], out['P32_ex'] = alpha_propagator_p31_p32(tau_ex, tau_m, c_m, h)
    out['P31_in'], out['P32_in'] = alpha_propagator_p31_p32(tau_in, tau_m, c_m, h)
    out['psc_init_ex'] = math.e / tau_ex
    out['psc_init_in'] = math.e / tau_in
    return out
