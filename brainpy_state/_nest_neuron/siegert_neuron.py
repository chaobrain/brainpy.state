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


import math
from typing import Callable

import brainstate
import braintools
import brainunit as u
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jax_special
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base._base import NESTNeuron
from brainpy_state._nest_base._utils import is_tracer, cond_any

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


class siegert_neuron(NESTNeuron):
    r"""NEST-compatible ``siegert_neuron`` mean-field rate model.

    **1. Overview**

    Mean-field rate model using the Siegert gain function of a noisy LIF neuron.
    This model computes the population-averaged firing rate from drift-diffusion
    input statistics via an analytic transfer function, enabling efficient
    large-scale network simulation without explicit spike generation.

    **2. Mathematical Description**

    The rate dynamics follow a first-order ODE:

    .. math::

       \tau\,\frac{dr(t)}{dt} = -r(t) + \text{mean} + \Phi(\mu, \sigma^2),

    where:

    - :math:`r(t)` is the population firing rate (Hz)
    - :math:`\tau` is the rate time constant
    - :math:`\text{mean}` is a constant baseline drive
    - :math:`\Phi(\mu, \sigma^2)` is the Siegert transfer function
    - :math:`\mu` is the total drift input (mean membrane potential shift)
    - :math:`\sigma^2` is the total diffusion input (variance)

    The Siegert function :math:`\Phi` analytically computes the steady-state
    firing rate of a leaky integrate-and-fire neuron receiving white noise with
    drift :math:`\mu` and diffusion :math:`\sigma^2`, subject to threshold
    :math:`\theta`, reset :math:`V_{\text{reset}}`, refractory period
    :math:`t_{\text{ref}}`, and membrane time constant :math:`\tau_m` [2]_.

    For colored noise (finite :math:`\tau_{\text{syn}} > 0`), a threshold shift
    correction is applied [3]_:

    .. math::

       \Delta_{\text{th}} = \frac{\alpha}{2} \sqrt{\frac{\tau_{\text{syn}}}{\tau_m}},

    where :math:`\alpha = |\zeta(1/2)| \sqrt{2} \approx 2.0653`.

    The integration is performed via exact exponential propagators:

    .. math::

       r(t + \Delta t) = e^{-\Delta t / \tau} r(t) + \left(1 - e^{-\Delta t / \tau}\right)
       \left(\text{mean} + \Phi(\mu, \sigma^2)\right).

    **3. NEST-Compatible Update Ordering (Non-WFR Path)**

    For each simulation step:

    1. Collect delayed and instantaneous diffusion-event buffers from queues.
    2. Sum all drift and diffusion contributions (delayed, instant, direct inputs).
    3. Evaluate Siegert transfer function :math:`\Phi(\mu_{\text{total}}, \sigma^2_{\text{total}})`.
    4. Update rate via exact exponential step: :math:`r \leftarrow P_1 r + P_2 (\text{mean} + \Phi)`.
    5. Publish updated rate to ``delayed_rate`` and ``instant_rate`` buffers for outgoing connections.

    This mirrors NEST's non-waveform-relaxation ``update_`` semantics where
    emitted diffusion coefficients are overwritten with the post-update rate.

    **4. Diffusion Event Handling**

    Runtime diffusion events modulate drift and diffusion inputs. Events can be
    supplied via two channels:

    - ``instant_diffusion_events``: applied in the current step (delay = 0)
    - ``delayed_diffusion_events``: scheduled by integer ``delay_steps`` (default 1)

    Event format supports dicts, tuples, or lists. Dict keys:

    - ``coeff`` (or ``rate``/``value``): base coefficient
    - ``drift_factor``: multiplier for drift contribution
    - ``diffusion_factor``: multiplier for diffusion contribution
    - ``weight``: connection weight (default 1)
    - ``multiplicity``: event count (default 1)
    - ``delay_steps`` (or ``delay``): integer delay in steps

    Tuple/list format: ``(coeff, drift_factor, diffusion_factor, delay_steps, weight, multiplicity)``.
    Shorter tuples use default values for trailing fields.

    Drift and diffusion contributions are computed as:

    .. math::

       \mu &= \text{coeff} \times \text{weight} \times \text{multiplicity} \times \text{drift\_factor}, \\
       \sigma^2 &= \text{coeff} \times \text{weight} \times \text{multiplicity} \times \text{diffusion\_factor}.

    **5. Siegert Transfer Function Computation**

    The Siegert function is evaluated element-wise for array inputs. For each
    population element, the computation handles three regimes:

    - **Deterministic (σ² ≤ 0)**: If μ > θ, returns LIF firing rate; else 0.
    - **Very subthreshold (θ - μ > 6σ)**: Returns 0 (Brunel 2000 fast path).
    - **General diffusive**: Computes via integral of scaled complementary error
      function (erfcx) and Dawson's integral, using either SciPy (if available)
      or custom Gauss-Legendre quadrature with asymptotic expansions.

    Numerical integration uses 64-point Gauss-Legendre quadrature for erfcx and
    adaptive segmentation for Dawson's integral, ensuring relative accuracy
    ~1.5e-8.

    Parameters
    ----------
    in_size : Size
        Population shape. Tuple of ints or single int for 1D populations.
        Determines the spatial structure of the rate model. For example,
        ``(10, 10)`` creates a 10×10 grid of mean-field neurons.
    tau : Quantity[ms], optional
        Time constant of the first-order rate dynamics (must be > 0). Controls
        the rate of convergence to the steady-state Siegert value. Smaller
        values produce faster tracking of input changes. Default: ``1 ms``.
    tau_m : Quantity[ms], optional
        Membrane time constant used in the Siegert gain function (must be > 0).
        Represents the passive membrane time constant of the modeled LIF neurons.
        Default: ``5 ms``.
    tau_syn : Quantity[ms], optional
        Synaptic time constant for colored-noise threshold correction (must be ≥ 0).
        When ``tau_syn > 0``, applies a threshold shift to account for finite
        synaptic rise time [3]_. Use ``0 ms`` for white noise (no correction).
        Default: ``0 ms``.
    t_ref : Quantity[ms], optional
        Refractory period in the Siegert gain function (must be ≥ 0). Represents
        the absolute refractory period during which the neuron cannot spike.
        Increases the interspike interval and reduces firing rates. Default: ``2 ms``.
    mean : float, optional
        Constant additive baseline drive in the rate ODE (dimensionless). Shifts
        the firing rate upward without affecting dynamics. Can be scalar or
        array matching ``in_size``. Default: ``0.0``.
    theta : float, optional
        Spike threshold relative to resting potential (dimensionless, corresponds
        to mV in NEST). Must be > ``V_reset``. Defines the firing threshold in
        the Siegert transfer function. Default: ``15.0``.
    V_reset : float, optional
        Reset potential relative to resting potential (dimensionless, corresponds
        to mV in NEST). Must be < ``theta``. Neuron is reset to this value after
        spiking in the underlying LIF model. Default: ``0.0``.
    rate_initializer : Callable, optional
        Initializer function for the ``rate`` state variable. Called as
        ``rate_initializer(shape, batch_size)`` during ``init_state()``. Default:
        ``braintools.init.Constant(0.0)`` (all neurons start at 0 Hz).
    name : str, optional
        Unique identifier for this module. If ``None``, auto-generated. Used for
        logging and debugging.

    Parameter Mapping
    -----------------

    =========================  ==========================  ==================================
    NEST Parameter             brainpy.state Parameter     Description
    =========================  ==========================  ==================================
    ``tau``                    ``tau``                     Rate dynamics time constant
    ``tau_m``                  ``tau_m``                   Membrane time constant (Siegert)
    ``tau_syn``                ``tau_syn``                 Synaptic time constant (threshold shift)
    ``t_ref``                  ``t_ref``                   Refractory period (Siegert)
    ``mean``                   ``mean``                    Constant baseline drive
    ``theta``                  ``theta``                   Spike threshold (relative to rest)
    ``V_reset``                ``V_reset``                 Reset potential (relative to rest)
    ``rate``                   ``rate``                    Current firing rate (Hz)
    =========================  ==========================  ==================================

    Raises
    ------
    ValueError
        If ``tau`` ≤ 0, ``tau_m`` ≤ 0, ``tau_syn`` < 0, ``t_ref`` < 0, or ``V_reset`` ≥ ``theta``.
    ValueError
        If ``instant_diffusion_events`` contains non-zero ``delay_steps``.
    ValueError
        If ``delayed_diffusion_events`` contains negative ``delay_steps``.
    ValueError
        If event tuples have length > 6 or < 1.

    Notes
    -----
    **Computational Complexity**


    - Siegert evaluation is the primary bottleneck (O(N) per neuron).
    - Without SciPy, custom quadrature adds ~10× overhead.
    - Delayed event queues are sparse dicts (O(1) insertion, O(K) retrieval
      for K active delays).

    **Numerical Stability:**

    - Uses ``erfcx(x) = exp(x²) erfc(x)`` to avoid overflow for large x.
    - Asymptotic expansions for erfcx and Dawson's integral when x > 8.
    - Exact exponential propagators (``exp`` and ``expm1``) prevent drift accumulation.

    **Batch Dimensions:**

    States support an optional leading batch dimension for parallelizing multiple
    network realizations. Initialize with ``init_state(batch_size=B)`` to create
    shape ``(B, *in_size)``.

    **Integration with NEST:**

    This implementation reproduces NEST 3.9+ behavior for ``siegert_neuron`` in
    non-waveform-relaxation mode. Key differences:

    - NEST uses precise spike times; brainpy.state uses fixed-step integration.
    - NEST's WFR mode (iterative delay resolution) is not implemented.
    - Event formats are compatible but may differ in edge cases (consult NEST docs).

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

    Examples
    --------
    **Basic usage with constant input:**

    .. code-block:: python

        >>> from brainpy import state as bp
        >>> import brainunit as u
        >>> import brainstate
        >>> model = bp.siegert_neuron(in_size=10, tau=2*u.ms, tau_m=10*u.ms)
        >>> model.init_all_states()
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     for _ in range(100):
        ...         rate = model.update(drift_input=12.0, diffusion_input=4.0)
        >>> print(rate)  # Steady-state firing rate in Hz

    **Using diffusion events for network coupling:**

    .. code-block:: python

        >>> model.init_all_states()
        >>> event = {'coeff': 50.0, 'drift_factor': 0.1, 'diffusion_factor': 0.05, 'delay_steps': 1}
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     rate = model.update(delayed_diffusion_events=event)
        >>> print(model.rate.value)  # Rate after delayed event delivery

    **Mean-field network with recurrent connections:**

    .. code-block:: python

        >>> exc = bp.siegert_neuron(in_size=800, tau=1*u.ms, theta=15.0)
        >>> inh = bp.siegert_neuron(in_size=200, tau=1*u.ms, theta=15.0)
        >>> exc.init_all_states()
        >>> inh.init_all_states()
        >>> # Simulate recurrent network (conceptual; requires projection setup)
        >>> with brainstate.environ.context(dt=0.1*u.ms):
        ...     for t in range(1000):
        ...         exc_drive = exc.rate.value.sum() * 0.01
        ...         inh_drive = inh.rate.value.sum() * -0.02
        ...         exc.update(drift_input=exc_drive + inh_drive, diffusion_input=2.0)
        ...         inh.update(drift_input=exc_drive, diffusion_input=1.0)
    """

    __module__ = 'brainpy.state'

    # NEST value: alpha = |zeta(1/2)| * sqrt(2)
    _ALPHA = 2.0652531522312172

    # Seam-(H) continuous-rate emitter: the Simulator allocates an emission holder
    # and captures ``rate`` each step so an outgoing diffusion_connection can read
    # the previous step's rate (NEST min_delay=1). See _network/_simulator.py.
    _emission_continuous = True
    _emission_attr = 'rate'

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
        dftype = brainstate.environ.dftype()
        return np.asarray(u.get_mantissa(x), dtype=dftype)

    @staticmethod
    def _to_numpy_ms(x):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.get_mantissa(x / u.ms), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.tau, self.tau_m, self.tau_syn, self.t_ref, self.V_reset, self.theta)):
            return

        if cond_any(self.tau <= 0.0 * u.ms):
            raise ValueError('Time constant tau must be > 0.')
        if cond_any(self.tau_m <= 0.0 * u.ms):
            raise ValueError('Membrane time constant tau_m must be > 0.')
        if cond_any(self.tau_syn < 0.0 * u.ms):
            raise ValueError('Synaptic time constant tau_syn must be >= 0.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory period t_ref must be >= 0.')
        if cond_any(self.V_reset >= self.theta):
            raise ValueError('Reset potential V_reset must be smaller than threshold theta.')

    def init_state(self, **kwargs):
        rate = braintools.init.param(self.rate_initializer, self.varshape)
        rate_np = self._to_numpy(rate)

        self.rate = brainstate.ShortTermState(rate_np)
        dftype = brainstate.environ.dftype()
        self.instant_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))

    @staticmethod
    def _gauss_legendre_scalar_integral(func, a: float, b: float):
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        pts = mid + half * _GAUSS_NODES
        dftype = brainstate.environ.dftype()
        vals = np.asarray([func(float(x)) for x in pts], dtype=dftype)
        return float(half * np.sum(_GAUSS_WEIGHTS * vals))

    @staticmethod
    def _erfcx_pos_scalar(x: float):
        if _HAVE_SCIPY:
            return float(_sp_special.erfcx(x))

        if x < 25.0:
            return math.exp(x * x) * math.erfc(x)

        inv = 1.0 / x
        inv2 = inv * inv
        poly = 1.0 + 0.5 * inv2 + 0.75 * inv2 * inv2 + 1.875 * inv2 ** 3 + 6.5625 * inv2 ** 4
        return (inv / math.sqrt(math.pi)) * poly

    @staticmethod
    def _integral_erfcx_asympt(a: float, b: float):
        inv_a2 = 1.0 / (a * a)
        inv_b2 = 1.0 / (b * b)

        term0 = math.log(b / a)
        term1 = -0.25 * (inv_b2 - inv_a2)
        term2 = -(3.0 / 16.0) * (inv_b2 * inv_b2 - inv_a2 * inv_a2)
        term3 = -(5.0 / 16.0) * (inv_b2 ** 3 - inv_a2 ** 3)
        term4 = -(105.0 / 128.0) * (inv_b2 ** 4 - inv_a2 ** 4)

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
                - (8.0 / 105.0) * x2 ** 3
                + (16.0 / 945.0) * x2 ** 4
            )

        if x >= 8.0:
            inv = 1.0 / x
            inv2 = inv * inv
            return (
                0.5 * inv
                + 0.25 * inv * inv2
                + (3.0 / 8.0) * inv * inv2 ** 2
                + (15.0 / 16.0) * inv * inv2 ** 3
                + (105.0 / 32.0) * inv * inv2 ** 4
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
        dftype = brainstate.environ.dftype()
        out = np.empty_like(mu, dtype=dftype)
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

    # ------------------------------------------------------------------
    # JAX-native Siegert transfer (goal 15c, design B).
    #
    # The host ``_siegert_scalar`` path above (SciPy / numpy Gauss-Legendre)
    # is the quadrature *oracle*; it stays eager and drives ``siegert_rate``.
    # The ``*_jax`` methods below re-express the same three-branch algorithm in
    # ``jax.numpy`` so ``update`` lowers under ``brainstate.transform.for_loop``
    # / ``jit``. They are validated against the oracle in
    # ``_validation/siegert_diffusion_test.py``.
    # ------------------------------------------------------------------

    @staticmethod
    def _erfcx_jax(x):
        r"""Scaled complementary error function ``erfcx(x) = exp(x^2) erfc(x)``.

        Direct ``exp(x^2) erfc(x)`` for ``x < 8`` (clipped to avoid overflow in the
        unused branch); a seven-term ``1/x`` asymptotic series for ``x >= 8``.
        """
        x = jnp.asarray(x)
        inv = 1.0 / jnp.where(x != 0.0, x, 1.0)
        inv2 = inv * inv
        # erfcx(x) ~ 1/(x sqrt(pi)) * sum_k (-1)^k (2k-1)!!/(2x^2)^k  (alternating).
        poly = (1.0 - 0.5 * inv2 + 0.75 * inv2 ** 2 - 1.875 * inv2 ** 3
                + 6.5625 * inv2 ** 4 - 29.53125 * inv2 ** 5 + 162.421875 * inv2 ** 6)
        asympt = inv / jnp.sqrt(jnp.pi) * poly
        x_safe = jnp.minimum(x, 8.0)
        direct = jnp.exp(x_safe * x_safe) * jax_special.erfc(x_safe)
        return jnp.where(x < 8.0, direct, asympt)

    @classmethod
    def _dawsn_jax(cls, x):
        r"""Dawson's integral ``D(x) = exp(-x^2) \int_0^x exp(t^2) dt`` (odd in x).

        Taylor series for ``|x| < 0.2``; a five-term ``1/x`` asymptotic series for
        ``|x| >= 8``; an 8-segment 64-point Gauss-Legendre quadrature of
        ``\int_0^{|x|} exp(t^2) dt`` (segments of width ``<= 1``) in between. The
        argument to ``exp`` is clipped to ``[., 8]`` so the (unused) mid branch
        stays finite for large ``|x|``.
        """
        x = jnp.asarray(x)
        ax = jnp.abs(x)
        sgn = jnp.sign(x)

        x2 = ax * ax
        taylor = ax * (1.0 - (2.0 / 3.0) * x2 + (4.0 / 15.0) * x2 ** 2
                       - (8.0 / 105.0) * x2 ** 3 + (16.0 / 945.0) * x2 ** 4)

        inv = 1.0 / jnp.where(ax > 0.0, ax, 1.0)
        inv2 = inv * inv
        asympt = (0.5 * inv + 0.25 * inv * inv2 + (3.0 / 8.0) * inv * inv2 ** 2
                  + (15.0 / 16.0) * inv * inv2 ** 3 + (105.0 / 32.0) * inv * inv2 ** 4
                  + (945.0 / 64.0) * inv * inv2 ** 5 + (10395.0 / 128.0) * inv * inv2 ** 6)

        nseg = 8
        nodes = jnp.asarray(_GAUSS_NODES)
        weights = jnp.asarray(_GAUSS_WEIGHTS)
        seg_w = ax / nseg
        k = jnp.arange(nseg)
        left = seg_w[..., None] * k
        mid = left + 0.5 * seg_w[..., None]
        half_seg = 0.5 * seg_w
        pts = mid[..., None] + half_seg[..., None, None] * nodes
        integrand = jnp.exp(jnp.minimum(pts, 8.0) ** 2)
        integ_per_seg = jnp.sum(weights * integrand, axis=-1) * half_seg[..., None]
        integral = jnp.sum(integ_per_seg, axis=-1)
        mid_val = jnp.exp(-ax * ax) * integral

        out = jnp.where(ax < 0.2, taylor, jnp.where(ax >= 8.0, asympt, mid_val))
        return sgn * out

    @staticmethod
    def _integral_erfcx_asympt_jax(a, b):
        r"""Closed-form ``\int_a^b erfcx(s) ds`` via the ``1/s`` asymptotic series."""
        inv_a2 = 1.0 / (a * a)
        inv_b2 = 1.0 / (b * b)
        # Antiderivative of the alternating erfcx asymptotic series; the odd-order
        # terms are +, the even-order terms - (integral of (-1)^k (2k-1)!!/(2s^2)^k).
        term0 = jnp.log(b / a)
        term1 = 0.25 * (inv_b2 - inv_a2)
        term2 = -(3.0 / 16.0) * (inv_b2 ** 2 - inv_a2 ** 2)
        term3 = (5.0 / 16.0) * (inv_b2 ** 3 - inv_a2 ** 3)
        term4 = -(105.0 / 128.0) * (inv_b2 ** 4 - inv_a2 ** 4)
        term5 = (945.0 / 320.0) * (inv_b2 ** 5 - inv_a2 ** 5)
        return (term0 + term1 + term2 + term3 + term4 + term5) / jnp.sqrt(jnp.pi)

    @classmethod
    def _integral_erfcx_jax(cls, a, b):
        r"""``\int_a^b erfcx(s) ds`` for non-negative bounds (signed in ``b - a``).

        ``[lo, min(hi, 8)]`` is integrated with a fixed 64-point Gauss-Legendre rule
        (``erfcx`` is smooth and bounded there); ``[max(lo, 8), hi]`` uses the
        closed-form asymptotic antiderivative.
        """
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        lo = jnp.minimum(a, b)
        hi = jnp.maximum(a, b)
        sign = jnp.sign(b - a)
        split = 8.0

        c = jnp.minimum(hi, split)
        nodes = jnp.asarray(_GAUSS_NODES)
        weights = jnp.asarray(_GAUSS_WEIGHTS)
        mid = 0.5 * (lo + c)
        half = 0.5 * (c - lo)
        pts = mid[..., None] + half[..., None] * nodes
        gl = half * jnp.sum(weights * cls._erfcx_jax(pts), axis=-1)
        gl = jnp.where(lo < split, gl, 0.0)

        d = jnp.maximum(lo, split)
        asy = cls._integral_erfcx_asympt_jax(d, hi)
        asy = jnp.where(hi > split, asy, 0.0)
        return sign * (gl + asy)

    @classmethod
    def _siegert_phi_core(cls, mu, sigma_square, tau_m_ms, tau_syn_ms, t_ref_ms, theta, v_reset):
        r"""JAX three-branch Siegert transfer on broadcast arrays (Hz)."""
        mu = jnp.asarray(mu)
        sig2 = jnp.asarray(sigma_square)
        sqrt_pi = jnp.sqrt(jnp.pi)

        # Deterministic LIF (sigma^2 <= 0): guard the log argument to stay finite.
        gap = jnp.where(mu > theta, mu - theta, 1.0)
        ratio = jnp.where(mu > theta, (mu - v_reset) / gap, 2.0)
        det = jnp.where(mu > theta, 1e3 / (t_ref_ms + tau_m_ms * jnp.log(ratio)), 0.0)

        sigma = jnp.sqrt(jnp.maximum(sig2, 1e-12))
        shift = (cls._ALPHA / 2.0) * jnp.sqrt(tau_syn_ms / tau_m_ms)
        y_th = (theta - mu) / sigma + shift
        y_r = (v_reset - mu) / sigma + shift
        e_th = jnp.exp(-y_th * y_th)

        # Clamp heavy-function arguments to their valid (non-negative) ranges; this
        # is a no-op in each branch's *selected* region and keeps the unused branches
        # finite (value-safe jnp.where).
        yth_p = jnp.maximum(y_th, 0.0)
        yr_p = jnp.maximum(y_r, 0.0)
        myth_p = jnp.maximum(-y_th, 0.0)
        myr_p = jnp.maximum(-y_r, 0.0)

        # Branch A: y_r > 0.
        iA = cls._integral_erfcx_jax(yr_p, yth_p)
        expd = jnp.exp(jnp.minimum(y_r * y_r - y_th * y_th, 0.0))
        integ_A = 2.0 * cls._dawsn_jax(yth_p) - 2.0 * expd * cls._dawsn_jax(yr_p) - e_th * iA
        rate_A = 1e3 * e_th / (e_th * t_ref_ms + tau_m_ms * sqrt_pi * integ_A)

        # Branch B: y_th < 0.
        iB = cls._integral_erfcx_jax(myth_p, myr_p)
        rate_B = 1e3 / (t_ref_ms + tau_m_ms * sqrt_pi * iB)

        # Branch C: y_r <= 0 <= y_th.
        iC = cls._integral_erfcx_jax(yth_p, myr_p)
        integ_C = 2.0 * cls._dawsn_jax(yth_p) + e_th * iC
        rate_C = 1e3 * e_th / (e_th * t_ref_ms + tau_m_ms * sqrt_pi * integ_C)

        rate = jnp.where(y_r > 0.0, rate_A, jnp.where(y_th < 0.0, rate_B, rate_C))
        # Brunel (2000) deep-subthreshold fast path.
        rate = jnp.where((theta - mu) > 6.0 * sigma, 0.0, rate)
        rate = jnp.where(sig2 <= 0.0, det, rate)
        return jnp.maximum(rate, 0.0)

    def _siegert_phi_jax(self, mu: ArrayLike, sigma_square: ArrayLike):
        r"""Evaluate the JAX Siegert transfer with this model's parameters (Hz).

        JAX-lowering counterpart of :meth:`siegert_rate`: ``mu`` / ``sigma_square``
        may be tracers, the model parameters are folded in as static constants, and
        the result is a ``jax.numpy`` array that composes under
        ``brainstate.transform`` primitives.

        Parameters
        ----------
        mu : ArrayLike
            Drift input (mean membrane potential shift), broadcastable with
            ``sigma_square`` and the model parameters.
        sigma_square : ArrayLike
            Diffusion input (membrane potential variance); non-negative.

        Returns
        -------
        rate : jax.Array
            Firing rate in Hz (broadcast shape of inputs and parameters).

        See Also
        --------
        siegert_rate : Eager SciPy / Gauss-Legendre quadrature oracle.
        """
        theta = jnp.asarray(self._to_numpy(self.theta))
        v_reset = jnp.asarray(self._to_numpy(self.V_reset))
        tau_m_ms = jnp.asarray(self._to_numpy_ms(self.tau_m))
        tau_syn_ms = jnp.asarray(self._to_numpy_ms(self.tau_syn))
        t_ref_ms = jnp.asarray(self._to_numpy_ms(self.t_ref))
        return self._siegert_phi_core(
            jnp.asarray(mu), jnp.asarray(sigma_square),
            tau_m_ms, tau_syn_ms, t_ref_ms, theta, v_reset,
        )

    def siegert_rate(self, mu: ArrayLike, sigma_square: ArrayLike):
        r"""Evaluate the NEST-compatible Siegert transfer function.

        Computes the steady-state firing rate :math:`\Phi(\mu, \sigma^2)` of a
        noisy LIF neuron with drift ``mu`` and diffusion ``sigma_square``, using
        the analytic Siegert formula [2]_ with optional colored-noise correction [3]_.

        The computation is vectorized over population elements. Inputs are broadcast
        with model parameters (``theta``, ``tau_m``, etc.) to produce an output
        array matching the broadcast shape.

        Parameters
        ----------
        mu : ArrayLike
            Drift input (mean membrane potential shift, dimensionless). Scalar or
            array broadcastable with ``sigma_square`` and model parameters. Positive
            values depolarize the neuron. Typically in the range [0, 30] for
            physiological parameters.
        sigma_square : ArrayLike
            Diffusion input (membrane potential variance, dimensionless squared).
            Scalar or array broadcastable with ``mu`` and model parameters. Must be
            non-negative. Typical values: 0.1–10 for moderate noise. Zero produces
            deterministic LIF behavior.

        Returns
        -------
        rate : ndarray
            Firing rate in Hz (shape matches broadcast of inputs and model parameters).
            Values ≥ 0. Returns 0 for subthreshold inputs (μ < θ with low noise).
            Maximum rate is approximately ``1000 / t_ref`` Hz (refractory limit).

        Notes
        -----
        **Special Cases:**

        - If ``sigma_square`` ≤ 0: deterministic LIF (returns 0 if μ ≤ θ, else
          fires at constant-input rate).
        - If (θ - μ) > 6σ: deep subthreshold (returns 0, Brunel 2000 fast path).
        - If ``t_ref`` = 0: no refractory limit (rate can diverge for μ >> θ).

        **Performance:**

        - Without SciPy: uses 64-point Gauss-Legendre quadrature (~10× slower).
        - With SciPy: uses ``scipy.integrate.quad`` and ``scipy.special`` (faster).

        **Broadcasting Rules:**

        Output shape is ``np.broadcast(mu, sigma_square, theta).shape``. For
        example, if model has ``in_size=(10,)``, ``mu`` is scalar, and
        ``sigma_square`` has shape ``(10,)``, output shape is ``(10,)``.

        Examples
        --------
        **Single neuron with varying drift:**

        .. code-block:: python

            >>> from brainpy import state as bp
            >>> import numpy as np
            >>> import brainunit as u
            >>> model = bp.siegert_neuron(in_size=1, tau_m=10*u.ms, t_ref=2*u.ms, theta=15.0)
            >>> mu_vals = np.linspace(0, 25, 50)
            >>> rates = model.siegert_rate(mu=mu_vals, sigma_square=2.0)
            >>> print(rates.shape)  # (50,)
            >>> print(rates.max())  # Maximum firing rate in Hz

        **Population with heterogeneous noise:**

        .. code-block:: python

            >>> model = bp.siegert_neuron(in_size=100, tau_m=10*u.ms)
            >>> sigma_sq = np.linspace(0.1, 5.0, 100)
            >>> rates = model.siegert_rate(mu=15.0, sigma_square=sigma_sq)
            >>> print(rates.shape)  # (100,)

        **2D grid with spatially varying input:**

        .. code-block:: python

            >>> model = bp.siegert_neuron(in_size=(10, 10), tau_m=10*u.ms)
            >>> mu_grid = np.random.uniform(10, 20, size=(10, 10))
            >>> rates = model.siegert_rate(mu=mu_grid, sigma_square=3.0)
            >>> print(rates.shape)  # (10, 10)
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

    def update(self, x=0.0, drift_input: ArrayLike = 0.0, diffusion_input: ArrayLike = 0.0):
        r"""Advance the Siegert rate by one step (NEST non-WFR semantics).

        Drift and diffusion are read from the dual-channel substrate seam that a
        :class:`~brainpy_state.diffusion_connection` deposits into (goal 15c,
        design A):

        - drift  :math:`\mu = \mathrm{sum\_current\_inputs}(x, r)
          + \mathrm{drift\_input} + \mathrm{sum\_delta\_inputs}(0,\ \text{label}=`
          ``'diffusion_mu'`` :math:`)`,
        - diffusion :math:`\sigma^2 = \mathrm{diffusion\_input}
          + \mathrm{sum\_delta\_inputs}(0,\ \text{label}=` ``'diffusion_sigma2'``
          :math:`)`.

        The two channels carry distinct labels so a single ``diffusion_connection``
        making two seam deposits (``drift_factor * rate`` and
        ``diffusion_factor * rate``) never cross-contaminates :math:`\mu` and
        :math:`\sigma^2`. The rate then relaxes by the exact exponential propagator
        :math:`r \leftarrow P_1 r + P_2(\mathrm{mean} + \Phi)`.

        The Siegert transfer is evaluated with the JAX port :meth:`_siegert_phi_jax`,
        so the whole step lowers under ``brainstate.transform.for_loop`` / ``jit``
        (drive it with those, not a bare Python loop).

        Parameters
        ----------
        x : ArrayLike, optional
            External drive forwarded to ``sum_current_inputs`` (dimensionless).
        drift_input : ArrayLike, optional
            Direct drift contribution added to the total drift before the Siegert
            evaluation.
        diffusion_input : ArrayLike, optional
            Direct diffusion (variance) contribution; must be non-negative.

        Returns
        -------
        rate_new : jax.Array
            Updated firing rate in Hz (shape ``self.rate.value.shape``).
        """
        h = float(u.get_mantissa(brainstate.environ.get_dt() / u.ms))
        state_shape = self.rate.value.shape
        rate_prev = jnp.asarray(self.rate.value)

        # Drift: current-input seam + direct arg + labeled diffusion-drift channel.
        mu_total = (self.sum_current_inputs(x, rate_prev) + drift_input
                    + self.sum_delta_inputs(0.0, label='diffusion_mu'))
        # Diffusion (variance): direct arg + labeled diffusion-variance channel.
        sigma_square_total = (diffusion_input
                              + self.sum_delta_inputs(0.0, label='diffusion_sigma2'))

        drive = self._siegert_phi_jax(mu_total, sigma_square_total)

        tau = jnp.asarray(self._to_numpy_ms(self.tau))
        mean = jnp.asarray(self._to_numpy(self.mean))
        p1 = jnp.exp(-h / tau)
        p2 = -jnp.expm1(-h / tau)
        rate_new = jnp.broadcast_to(p1 * rate_prev + p2 * (mean + drive), state_shape)

        self.rate.value = rate_new
        # NEST non-WFR: outgoing delayed/instant buffers carry the final rate.
        self.delayed_rate.value = rate_new
        self.instant_rate.value = rate_new
        return rate_new
