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

from __future__ import annotations

import math
from typing import Callable

import numpy as np

import brainstate
import braintools
import brainunit as u
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Dynamics

__all__ = [
    'siegert_neuron',
]

try:
    from scipy import integrate as _sp_integrate
    from scipy import special as _sp_special

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - fallback path when SciPy is unavailable.
    _HAVE_SCIPY = False


# Gauss-Legendre nodes used by the scalar quadrature helpers.
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(64)


class siegert_neuron(Dynamics):
    r"""NEST-compatible ``siegert_neuron`` mean-field rate model.

    Short description
    -----------------

    Mean-field rate model using the Siegert gain function of a noisy LIF neuron.

    Description
    -----------

    ``siegert_neuron`` follows NEST's ``siegert_neuron`` model [1]_ with
    first-order rate dynamics

    .. math::

       \tau\,\frac{dr(t)}{dt} = -r(t) + \text{mean} + \Phi(\mu, \sigma^2),

    where :math:`\Phi` is the Siegert firing-rate function of a leaky
    integrate-and-fire neuron with refractory period and optional synaptic-time
    correction [2]_, [3]_.

    Incoming diffusion connections contribute per-step drift and diffusion terms
    :math:`\mu` and :math:`\sigma^2`.

    NEST-compatible update ordering (non-WFR path)
    ..............................................

    For each simulation step:

    1. Collect delayed and instantaneous diffusion-event buffers.
    2. Build total drift/diffusion input for this step.
    3. Evaluate Siegert drive ``Phi(mu, sigma^2)``.
    4. Update rate by exact exponential propagators:
       ``rate <- P1 * rate + P2 * (mean + drive)``.
    5. Publish outgoing diffusion coefficient as the updated rate.

    This mirrors NEST's non-waveform-relaxation ``update_`` semantics where the
    emitted coefficient array for the next delay window is overwritten with the
    post-update rate.

    Parameters
    ----------
    in_size : Size
        Population shape.
    tau : Quantity[ms], optional
        Time constant of the first-order rate dynamics. Default ``1 ms``.
    tau_m : Quantity[ms], optional
        Membrane time constant in the Siegert gain function. Default ``5 ms``.
    tau_syn : Quantity[ms], optional
        Synaptic time constant used in the colored-noise threshold shift.
        Default ``0 ms``.
    t_ref : Quantity[ms], optional
        Refractory period in the gain function. Default ``2 ms``.
    mean : float, optional
        Constant additive drive term in the rate ODE. Default ``0.0``.
    theta : float, optional
        Threshold relative to resting potential (mV in NEST docs).
        Default ``15.0``.
    V_reset : float, optional
        Reset value relative to resting potential. Default ``0.0``.
    rate_initializer : Callable, optional
        Initializer for ``rate``. Default ``Constant(0.0)``.
    name : str, optional
        Module name.

    Notes
    -----
    Runtime diffusion events can be supplied in two channels:

    - ``instant_diffusion_events``: applied in the current step.
    - ``delayed_diffusion_events``: scheduled by integer ``delay_steps``
      (default ``1``).

    Event format supports dicts or tuples/lists. Dict keys:

    - ``coeff`` (or ``rate``/``value``),
    - ``drift_factor``,
    - ``diffusion_factor``,
    - ``weight`` (optional, default ``1``),
    - ``multiplicity`` (optional, default ``1``),
    - ``delay_steps`` (or ``delay``).

    References
    ----------
    .. [1] Hahne J, Dahmen D, Schuecker J, Frommer A, Bolten M, Helias M,
       Diesmann M (2017). Integration of continuous-time dynamics in a spiking
       neural network simulator. Frontiers in Neuroinformatics, 11:34.
       DOI: ``10.3389/fninf.2017.00034``.
    .. [2] Fourcaud N, Brunel N (2002). Dynamics of the firing probability of
       noisy integrate-and-fire neurons. Neural Computation, 14(9):2057-2110.
       DOI: ``10.1162/089976602320264015``.
    .. [3] Schuecker J, Diesmann M, Helias M (2015). Modulated escape from a
       metastable state driven by colored noise. Physical Review E, 92:052119.
       DOI: ``10.1103/PhysRevE.92.052119``.
    """

    __module__ = 'brainpy.state'

    # NEST value: alpha = |zeta(1/2)| * sqrt(2)
    _ALPHA = 2.0652531522312172

    def __init__(
        self,
        in_size: Size,
        tau: ArrayLike = 1.0 * u.ms,
        tau_m: ArrayLike = 5.0 * u.ms,
        tau_syn: ArrayLike = 0.0 * u.ms,
        t_ref: ArrayLike = 2.0 * u.ms,
        mean: ArrayLike = 0.0,
        theta: ArrayLike = 15.0,
        V_reset: ArrayLike = 0.0,
        rate_initializer: Callable = braintools.init.Constant(0.0),
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.tau = braintools.init.param(tau, self.varshape)
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.tau_syn = braintools.init.param(tau_syn, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.mean = braintools.init.param(mean, self.varshape)
        self.theta = braintools.init.param(theta, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)

        self.rate_initializer = rate_initializer

        self._delayed_drift_queue = {}
        self._delayed_diffusion_queue = {}

        self._validate_parameters()

    @property
    def recordables(self):
        return ['rate']

    @property
    def receptor_types(self):
        # NEST handles DiffusionConnectionEvent via receptor type 1.
        return {'DIFFUSION': 1}

    @staticmethod
    def _to_numpy(x):
        return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _to_numpy_ms(x):
        return np.asarray(u.math.asarray(x / u.ms), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @staticmethod
    def _to_int_scalar(x, name: str):
        arr = np.asarray(u.math.asarray(x), dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return int(arr[0])

    @staticmethod
    def _coerce_events(events):
        if events is None:
            return []
        if isinstance(events, dict):
            return [events]
        if isinstance(events, tuple):
            if len(events) == 0:
                return []
            if isinstance(events[0], (dict, tuple, list)):
                return list(events)
            return [events]
        if isinstance(events, list):
            if len(events) == 0:
                return []
            if isinstance(events[0], (dict, tuple, list)):
                return events
            return [tuple(events)]
        return [events]

    @staticmethod
    def _queue_add(queue: dict, step_idx: int, value: np.ndarray):
        if step_idx in queue:
            queue[step_idx] = queue[step_idx] + value
        else:
            queue[step_idx] = np.array(value, dtype=np.float64, copy=True)

    def _drain_delayed_queue(self, step_idx: int, state_shape):
        drift = self._delayed_drift_queue.pop(step_idx, None)
        diffusion = self._delayed_diffusion_queue.pop(step_idx, None)

        if drift is None:
            drift = np.zeros(state_shape, dtype=np.float64)
        else:
            drift = np.array(self._broadcast_to_state(np.asarray(drift, dtype=np.float64), state_shape), copy=True)

        if diffusion is None:
            diffusion = np.zeros(state_shape, dtype=np.float64)
        else:
            diffusion = np.array(
                self._broadcast_to_state(np.asarray(diffusion, dtype=np.float64), state_shape),
                copy=True,
            )

        return drift, diffusion

    def _extract_event_fields(self, ev, default_delay_steps: int):
        if isinstance(ev, dict):
            coeff = ev.get('coeff', ev.get('rate', ev.get('value', 0.0)))
            drift_factor = ev.get('drift_factor', 1.0)
            diffusion_factor = ev.get('diffusion_factor', 0.0)
            weight = ev.get('weight', 1.0)
            multiplicity = ev.get('multiplicity', 1.0)
            delay_steps = ev.get('delay_steps', ev.get('delay', default_delay_steps))
        elif isinstance(ev, (tuple, list)):
            if len(ev) == 1:
                coeff = ev[0]
                drift_factor = 1.0
                diffusion_factor = 0.0
                weight = 1.0
                multiplicity = 1.0
                delay_steps = default_delay_steps
            elif len(ev) == 2:
                coeff, drift_factor = ev
                diffusion_factor = 0.0
                weight = 1.0
                multiplicity = 1.0
                delay_steps = default_delay_steps
            elif len(ev) == 3:
                coeff, drift_factor, diffusion_factor = ev
                weight = 1.0
                multiplicity = 1.0
                delay_steps = default_delay_steps
            elif len(ev) == 4:
                coeff, drift_factor, diffusion_factor, delay_steps = ev
                weight = 1.0
                multiplicity = 1.0
            elif len(ev) == 5:
                coeff, drift_factor, diffusion_factor, delay_steps, weight = ev
                multiplicity = 1.0
            elif len(ev) == 6:
                coeff, drift_factor, diffusion_factor, delay_steps, weight, multiplicity = ev
            else:
                raise ValueError('Diffusion event tuples must have length 1 to 6.')
        else:
            coeff = ev
            drift_factor = 1.0
            diffusion_factor = 0.0
            weight = 1.0
            multiplicity = 1.0
            delay_steps = default_delay_steps

        delay_steps = self._to_int_scalar(delay_steps, name='delay_steps')
        return coeff, drift_factor, diffusion_factor, weight, multiplicity, delay_steps

    def _event_to_drift_diffusion(self, ev, default_delay_steps: int, state_shape):
        coeff, drift_factor, diffusion_factor, weight, multiplicity, delay_steps = self._extract_event_fields(
            ev,
            default_delay_steps,
        )

        coeff_np = self._broadcast_to_state(self._to_numpy(coeff), state_shape)
        drift_factor_np = self._broadcast_to_state(self._to_numpy(drift_factor), state_shape)
        diffusion_factor_np = self._broadcast_to_state(self._to_numpy(diffusion_factor), state_shape)
        weight_np = self._broadcast_to_state(self._to_numpy(weight), state_shape)
        multiplicity_np = self._broadcast_to_state(self._to_numpy(multiplicity), state_shape)

        weighted_coeff = coeff_np * weight_np * multiplicity_np
        drift = drift_factor_np * weighted_coeff
        diffusion = diffusion_factor_np * weighted_coeff

        return drift, diffusion, delay_steps

    def _accumulate_instant_events(self, events, state_shape):
        drift = np.zeros(state_shape, dtype=np.float64)
        diffusion = np.zeros(state_shape, dtype=np.float64)
        for ev in self._coerce_events(events):
            d_i, s_i, delay_steps = self._event_to_drift_diffusion(
                ev,
                default_delay_steps=0,
                state_shape=state_shape,
            )
            if delay_steps != 0:
                raise ValueError('instant_diffusion_events must not specify non-zero delay_steps.')
            drift += d_i
            diffusion += s_i
        return drift, diffusion

    def _schedule_delayed_events(self, events, step_idx: int, state_shape):
        drift_now = np.zeros(state_shape, dtype=np.float64)
        diffusion_now = np.zeros(state_shape, dtype=np.float64)

        for ev in self._coerce_events(events):
            d_i, s_i, delay_steps = self._event_to_drift_diffusion(
                ev,
                default_delay_steps=1,
                state_shape=state_shape,
            )
            if delay_steps < 0:
                raise ValueError('delay_steps for delayed_diffusion_events must be >= 0.')
            if delay_steps == 0:
                drift_now += d_i
                diffusion_now += s_i
            else:
                target_step = step_idx + delay_steps
                self._queue_add(self._delayed_drift_queue, target_step, d_i)
                self._queue_add(self._delayed_diffusion_queue, target_step, s_i)

        return drift_now, diffusion_now

    def _validate_parameters(self):
        if np.any(self._to_numpy_ms(self.tau) <= 0.0):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self._to_numpy_ms(self.tau_m) <= 0.0):
            raise ValueError('Membrane time constant tau_m must be > 0.')
        if np.any(self._to_numpy_ms(self.tau_syn) < 0.0):
            raise ValueError('Synaptic time constant tau_syn must be >= 0.')
        if np.any(self._to_numpy_ms(self.t_ref) < 0.0):
            raise ValueError('Refractory period t_ref must be >= 0.')
        if np.any(self._to_numpy(self.V_reset) >= self._to_numpy(self.theta)):
            raise ValueError('Reset potential V_reset must be smaller than threshold theta.')

    def init_state(self, batch_size: int = None, **kwargs):
        rate = braintools.init.param(self.rate_initializer, self.varshape, batch_size)
        rate_np = self._to_numpy(rate)

        self.rate = brainstate.ShortTermState(rate_np)
        self.instant_rate = brainstate.ShortTermState(np.array(rate_np, dtype=np.float64, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(rate_np, dtype=np.float64, copy=True))
        self._step_count = brainstate.ShortTermState(np.asarray(0, dtype=np.int64))

        self._delayed_drift_queue = {}
        self._delayed_diffusion_queue = {}

    @staticmethod
    def _gauss_legendre_scalar_integral(func, a: float, b: float):
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        pts = mid + half * _GAUSS_NODES
        vals = np.asarray([func(float(x)) for x in pts], dtype=np.float64)
        return float(half * np.sum(_GAUSS_WEIGHTS * vals))

    @staticmethod
    def _erfcx_pos_scalar(x: float):
        if _HAVE_SCIPY:
            return float(_sp_special.erfcx(x))

        if x < 25.0:
            return math.exp(x * x) * math.erfc(x)

        inv = 1.0 / x
        inv2 = inv * inv
        poly = 1.0 + 0.5 * inv2 + 0.75 * inv2 * inv2 + 1.875 * inv2**3 + 6.5625 * inv2**4
        return (inv / math.sqrt(math.pi)) * poly

    @staticmethod
    def _integral_erfcx_asympt(a: float, b: float):
        inv_a2 = 1.0 / (a * a)
        inv_b2 = 1.0 / (b * b)

        term0 = math.log(b / a)
        term1 = -0.25 * (inv_b2 - inv_a2)
        term2 = -(3.0 / 16.0) * (inv_b2 * inv_b2 - inv_a2 * inv_a2)
        term3 = -(5.0 / 16.0) * (inv_b2**3 - inv_a2**3)
        term4 = -(105.0 / 128.0) * (inv_b2**4 - inv_a2**4)

        return (term0 + term1 + term2 + term3 + term4) / math.sqrt(math.pi)

    @classmethod
    def _integral_erfcx_pos(cls, a: float, b: float):
        if a == b:
            return 0.0

        sign = 1.0
        lo = float(a)
        hi = float(b)
        if lo > hi:
            sign = -1.0
            lo, hi = hi, lo

        if _HAVE_SCIPY:
            result, _ = _sp_integrate.quad(
                lambda s: float(_sp_special.erfcx(s)),
                lo,
                hi,
                epsabs=0.0,
                epsrel=1.49e-8,
                limit=1000,
            )
            return sign * float(result)

        split = 8.0
        total = 0.0

        if lo < split:
            hi_num = min(hi, split)
            width = hi_num - lo
            nseg = max(1, int(math.ceil(width / 2.0)))
            seg_w = width / nseg
            left = lo
            for _ in range(nseg):
                right = left + seg_w
                total += cls._gauss_legendre_scalar_integral(cls._erfcx_pos_scalar, left, right)
                left = right

        if hi > split:
            lo_as = max(lo, split)
            total += cls._integral_erfcx_asympt(lo_as, hi)

        return sign * total

    @classmethod
    def _dawsn_pos_scalar(cls, x: float):
        if _HAVE_SCIPY:
            return float(_sp_special.dawsn(x))

        if x == 0.0:
            return 0.0

        if x < 0.2:
            x2 = x * x
            return x * (
                1.0
                - (2.0 / 3.0) * x2
                + (4.0 / 15.0) * x2 * x2
                - (8.0 / 105.0) * x2**3
                + (16.0 / 945.0) * x2**4
            )

        if x >= 8.0:
            inv = 1.0 / x
            inv2 = inv * inv
            return (
                0.5 * inv
                + 0.25 * inv * inv2
                + (3.0 / 8.0) * inv * inv2**2
                + (15.0 / 16.0) * inv * inv2**3
                + (105.0 / 32.0) * inv * inv2**4
            )

        # Dawson(x) = exp(-x^2) * integral_0^x exp(t^2) dt
        nseg = max(1, int(math.ceil(x / 1.0)))
        seg_w = x / nseg
        left = 0.0
        integral = 0.0
        for _ in range(nseg):
            right = left + seg_w
            integral += cls._gauss_legendre_scalar_integral(lambda t: math.exp(t * t), left, right)
            left = right

        return math.exp(-x * x) * integral

    @classmethod
    def _siegert_scalar(
        cls,
        mu: float,
        sigma_square: float,
        tau_m_ms: float,
        tau_syn_ms: float,
        t_ref_ms: float,
        theta: float,
        v_reset: float,
    ):
        if sigma_square <= 0.0:
            if mu > theta:
                return 1e3 / (t_ref_ms + tau_m_ms * math.log((mu - v_reset) / (mu - theta)))
            return 0.0

        sigma = math.sqrt(sigma_square)

        # NEST fast path for very subthreshold input (Brunel 2000, eq. 22 estimate).
        if (theta - mu) > 6.0 * sigma:
            return 0.0

        threshold_shift = (cls._ALPHA / 2.0) * math.sqrt(tau_syn_ms / tau_m_ms)

        y_th = (theta - mu) / sigma + threshold_shift
        y_r = (v_reset - mu) / sigma + threshold_shift

        sqrt_pi = math.sqrt(math.pi)

        if y_r > 0.0:
            result = cls._integral_erfcx_pos(y_r, y_th)
            integral = (
                2.0 * cls._dawsn_pos_scalar(y_th)
                - 2.0 * math.exp(y_r * y_r - y_th * y_th) * cls._dawsn_pos_scalar(y_r)
                - math.exp(-y_th * y_th) * result
            )
            e = math.exp(-y_th * y_th)
            return 1e3 * e / (e * t_ref_ms + tau_m_ms * sqrt_pi * integral)

        if y_th < 0.0:
            integral = cls._integral_erfcx_pos(-y_th, -y_r)
            return 1e3 / (t_ref_ms + tau_m_ms * sqrt_pi * integral)

        result = cls._integral_erfcx_pos(y_th, -y_r)
        integral = 2.0 * cls._dawsn_pos_scalar(y_th) + math.exp(-y_th * y_th) * result
        e = math.exp(-y_th * y_th)
        return 1e3 * e / (e * t_ref_ms + tau_m_ms * sqrt_pi * integral)

    @classmethod
    def _siegert_array(
        cls,
        mu: np.ndarray,
        sigma_square: np.ndarray,
        tau_m_ms: np.ndarray,
        tau_syn_ms: np.ndarray,
        t_ref_ms: np.ndarray,
        theta: np.ndarray,
        v_reset: np.ndarray,
    ):
        out = np.empty_like(mu, dtype=np.float64)
        for idx in np.ndindex(mu.shape):
            out[idx] = cls._siegert_scalar(
                float(mu[idx]),
                float(sigma_square[idx]),
                float(tau_m_ms[idx]),
                float(tau_syn_ms[idx]),
                float(t_ref_ms[idx]),
                float(theta[idx]),
                float(v_reset[idx]),
            )
        return out

    def siegert_rate(self, mu: ArrayLike, sigma_square: ArrayLike):
        """Evaluate the NEST-compatible Siegert gain function ``Phi(mu, sigma^2)``.

        Inputs are broadcast together with model parameters and the model shape.
        Returned values are in Hz, matching NEST.
        """
        mu_np = self._to_numpy(mu)
        sigma_np = self._to_numpy(sigma_square)
        state_shape = np.broadcast(
            mu_np,
            sigma_np,
            self._to_numpy(self.theta),
        ).shape

        mu_b = self._broadcast_to_state(mu_np, state_shape)
        sigma_b = self._broadcast_to_state(sigma_np, state_shape)
        tau_m_b = self._broadcast_to_state(self._to_numpy_ms(self.tau_m), state_shape)
        tau_syn_b = self._broadcast_to_state(self._to_numpy_ms(self.tau_syn), state_shape)
        t_ref_b = self._broadcast_to_state(self._to_numpy_ms(self.t_ref), state_shape)
        theta_b = self._broadcast_to_state(self._to_numpy(self.theta), state_shape)
        v_reset_b = self._broadcast_to_state(self._to_numpy(self.V_reset), state_shape)

        return self._siegert_array(
            mu_b,
            sigma_b,
            tau_m_b,
            tau_syn_b,
            t_ref_b,
            theta_b,
            v_reset_b,
        )

    def update(
        self,
        x=0.0,
        drift_input: ArrayLike = 0.0,
        diffusion_input: ArrayLike = 0.0,
        instant_diffusion_events=None,
        delayed_diffusion_events=None,
    ):
        h = float(u.math.asarray(brainstate.environ.get_dt() / u.ms))

        state_shape = self.rate.value.shape
        step_idx = int(np.asarray(self._step_count.value, dtype=np.int64).reshape(-1)[0])

        drift_delayed, diffusion_delayed = self._drain_delayed_queue(step_idx, state_shape)
        d_now, s_now = self._schedule_delayed_events(
            delayed_diffusion_events,
            step_idx=step_idx,
            state_shape=state_shape,
        )
        drift_delayed += d_now
        diffusion_delayed += s_now

        drift_instant, diffusion_instant = self._accumulate_instant_events(
            instant_diffusion_events,
            state_shape=state_shape,
        )

        # Keep compatibility with the standard Dynamics input hooks.
        drift_direct = self._broadcast_to_state(
            self._to_numpy(self.sum_current_inputs(x, self.rate.value) + drift_input + self.sum_delta_inputs(0.0)),
            state_shape,
        )
        diffusion_direct = self._broadcast_to_state(self._to_numpy(diffusion_input), state_shape)

        mu_total = drift_delayed + drift_instant + drift_direct
        sigma_square_total = diffusion_delayed + diffusion_instant + diffusion_direct

        rate_prev = self._broadcast_to_state(self._to_numpy(self.rate.value), state_shape)
        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        mean = self._broadcast_to_state(self._to_numpy(self.mean), state_shape)

        tau_m = self._broadcast_to_state(self._to_numpy_ms(self.tau_m), state_shape)
        tau_syn = self._broadcast_to_state(self._to_numpy_ms(self.tau_syn), state_shape)
        t_ref = self._broadcast_to_state(self._to_numpy_ms(self.t_ref), state_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.theta), state_shape)
        v_reset = self._broadcast_to_state(self._to_numpy(self.V_reset), state_shape)

        drive = self._siegert_array(mu_total, sigma_square_total, tau_m, tau_syn, t_ref, theta, v_reset)

        p1 = np.exp(-h / tau)
        p2 = -np.expm1(-h / tau)
        rate_new = p1 * rate_prev + p2 * (mean + drive)

        self.rate.value = rate_new

        # NEST non-WFR update emits coefficient arrays overwritten by final rate.
        self.delayed_rate.value = np.array(rate_new, dtype=np.float64, copy=True)
        self.instant_rate.value = np.array(rate_new, dtype=np.float64, copy=True)

        self._step_count.value = np.asarray(step_idx + 1, dtype=np.int64)
        return rate_new
