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

import brainstate

from brainpy_state._nest._base import NESTDevice
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

__all__ = [
    'noise_generator',
]


class noise_generator(NESTDevice):
    r"""Gaussian white-noise current generator compatible with NEST.

    Generate a piecewise-constant Gaussian current with optional sinusoidal
    modulation of the noise standard deviation and a NEST-style activity window.

    **1. Stochastic process and update rule**

    Let :math:`\delta` be the configured noise update period. For each channel
    and noise interval index :math:`j`, this implementation samples

    .. math::

        A_j = \mu + \xi_j \sigma_{\mathrm{eff}}(t_j), \qquad
        \xi_j \sim \mathcal{N}(0, 1),

    then emits :math:`I(t)=A_j` for :math:`t_j \le t < t_j + \delta` while the
    generator is active. The effective standard deviation is

    .. math::

        \sigma_{\mathrm{eff}}(t)
        = \sqrt{\max\!\left(\sigma^2 + \sigma_{\mathrm{mod}}^2
          \sin(\omega t + \phi),\, 0\right)},
        \qquad \omega = \frac{2\pi f}{1000}.

    The non-negativity clamp follows the implementation exactly:
    ``maximum(., 0)`` is applied before ``sqrt`` so modulation never yields
    invalid real values.

    **2. Variance approximation and assumptions**

    For an LIF membrane receiving the unmodulated process
    (:math:`\sigma_{\mathrm{mod}}=0`) with :math:`\delta \ll \tau_m`, the
    asymptotic membrane potential variance is approximated by

    .. math::

        \Sigma^2 = \frac{\delta \tau_m \sigma^2}{2 C_m^2}.

    This approximation assumes linear subthreshold dynamics, stationary
    statistics, and sufficiently small update period relative to membrane time
    constant. Increasing :math:`\delta` increases drive variance linearly and
    shifts the spectrum away from ideal white-noise behavior.

    **3. Timing semantics and computational implications**

    The activity window is half-open:
    :math:`[t_0 + t_{\mathrm{start,rel}},\ t_0 + t_{\mathrm{stop,rel}})`.
    Therefore, ``start`` is inclusive and ``stop`` is exclusive.

    Noise amplitudes are refreshed when
    ``step_counter % round(noise_dt / dt) == 0``. If ``noise_dt is None``, then
    ``noise_dt = dt`` and updates occur every simulation step.

    This implementation is vectorized over ``self.varshape`` and performs one
    PRNG split and one Gaussian draw per :meth:`update` call, followed by a
    mask that either accepts the new sample or retains the previous amplitude.
    Work per call is :math:`O(\prod \mathrm{varshape})`.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification for :class:`brainstate.nn.Dynamics`.
        The generated current shape is ``self.varshape`` derived from
        ``in_size``. Default is ``1``.
    mean : ArrayLike, optional
        Mean current :math:`\mu` (typically pA). Scalars or arrays are accepted
        and broadcast to ``self.varshape`` by :func:`braintools.init.param`.
        Default is ``0. * u.pA``.
    std : ArrayLike, optional
        Baseline standard deviation :math:`\sigma` (typically pA), broadcast to
        ``self.varshape``. Default is ``0. * u.pA``.
    noise_dt : ArrayLike or None, optional
        Noise refresh interval :math:`\delta` (typically ms). ``None`` means
        use simulation ``dt`` at runtime. Values are converted to integer steps
        by ``round(noise_dt / dt)``; valid execution requires this rounded
        value to be at least ``1`` for every channel. Default is ``None``.
    std_mod : ArrayLike, optional
        Modulation amplitude :math:`\sigma_{\mathrm{mod}}` (typically pA) for
        the sinusoidal term in :math:`\sigma_{\mathrm{eff}}`. Broadcast to
        ``self.varshape``. Default is ``0. * u.pA``.
    frequency : ArrayLike, optional
        Modulation frequency :math:`f` in Hz (or unitless values interpreted as
        Hz), broadcast to ``self.varshape``. Converted internally to rad/ms
        using :math:`\omega = 2\pi f/1000`. Default is ``0. * u.Hz``.
    phase : ArrayLike, optional
        Modulation phase in degrees, broadcast to ``self.varshape``.
        Converted internally as
        :math:`\phi = \mathrm{phase}\cdot 2\pi/360`. Default is ``0.``.
    start : ArrayLike, optional
        Relative activation time :math:`t_{\mathrm{start,rel}}` (typically ms),
        broadcast to ``self.varshape``. Effective lower bound is
        ``origin + start``. Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative deactivation time :math:`t_{\mathrm{stop,rel}}` (typically ms),
        broadcast to ``self.varshape`` when provided. Effective upper bound is
        ``origin + stop`` and is exclusive. ``None`` means no upper bound.
        Default is ``None``.
    origin : ArrayLike, optional
        Time origin :math:`t_0` (typically ms), broadcast to ``self.varshape``
        and added to ``start``/``stop``. Default is ``0. * u.ms``.
    seed : int or None, optional
        PRNG seed used by :func:`jax.random.PRNGKey` in :meth:`init_state`.
        ``None`` selects deterministic fallback seed ``0``. Default is
        ``None``.
    name : str or None, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 18 17 22 43

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``mean``
         - ``0. * u.pA``
         - :math:`\mu`
         - Mean of the Gaussian current samples.
       * - ``std``
         - ``0. * u.pA``
         - :math:`\sigma`
         - Baseline standard deviation of the noise process.
       * - ``noise_dt``
         - ``None``
         - :math:`\delta`
         - Interval between sample refreshes; defaults to simulation ``dt``.
       * - ``std_mod``
         - ``0. * u.pA``
         - :math:`\sigma_{\mathrm{mod}}`
         - Amplitude of sinusoidal modulation in variance term.
       * - ``frequency``
         - ``0. * u.Hz``
         - :math:`f`
         - Modulation frequency converted to :math:`\omega=2\pi f/1000`.
       * - ``phase``
         - ``0.``
         - :math:`\phi_{\mathrm{deg}}`
         - Modulation phase in degrees, converted to radians in update.
       * - ``start``
         - ``0. * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative lower activity bound added to ``origin``.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative upper activity bound added to ``origin``.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global time offset for both activity boundaries.

    Raises
    ------
    ValueError
        If ``in_size`` is invalid or if array-like parameters cannot be
        broadcast to ``self.varshape`` by :func:`braintools.init.param`.
    KeyError
        If runtime environment keys such as ``'t'`` or ``'dt'`` are missing
        when :meth:`update` is called.
    TypeError
        If unitful/unitless arithmetic is incompatible (for example invalid
        combinations among time, frequency, and current parameters).
    ZeroDivisionError
        If ``round(noise_dt / dt)`` evaluates to ``0`` so modulo scheduling in
        :meth:`update` attempts division by zero.

    Notes
    -----
    NEST describes independent random currents per target neuron. In this
    implementation, one generator instance emits one current vector per call;
    downstream targets reading the same channel receive the same value for that
    step. Use separate generator instances to guarantee independent streams.

    See Also
    --------
    dc_generator : Constant current stimulation device.
    ac_generator : Sinusoidal current stimulation device.
    step_current_generator : Piecewise-constant current stimulation device.

    References
    ----------
    .. [1] NEST Simulator documentation for ``noise_generator``:
           https://nest-simulator.readthedocs.io/en/stable/models/noise_generator.html

    Examples
    --------
    Basic usage: unmodulated white-noise drive injected into a single neuron.

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     stim = brainpy.state.noise_generator(
       ...         in_size=1,
       ...         mean=0.0 * u.pA,
       ...         std=100.0 * u.pA,
       ...         noise_dt=0.2 * u.ms,
       ...         seed=42,
       ...     )
       ...     neuron = brainpy.state.iaf_psc_delta(1)
       ...     neuron.init_state()
       ...     with brainstate.environ.context(t=1.0 * u.ms):
       ...         current = stim.update()
       ...         _ = neuron.update(x=current)

    Sinusoidally modulated noise: variance oscillates at gamma frequency (40 Hz)
    within a restricted activity window.

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> gen = brainpy.state.noise_generator(
       ...     in_size=4,
       ...     mean=50.0 * u.pA,
       ...     std=80.0 * u.pA,
       ...     noise_dt=1.0 * u.ms,
       ...     std_mod=40.0 * u.pA,
       ...     frequency=40.0 * u.Hz,
       ...     phase=0.0,
       ...     start=10.0 * u.ms,
       ...     stop=110.0 * u.ms,
       ...     seed=0,
       ... )
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        mean: ArrayLike = 0. * u.pA,
        std: ArrayLike = 0. * u.pA,
        noise_dt: ArrayLike = None,
        std_mod: ArrayLike = 0. * u.pA,
        frequency: ArrayLike = 0. * u.Hz,
        phase: ArrayLike = 0.,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        seed: int = None,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        # parameters
        self.mean = braintools.init.param(mean, self.varshape)
        self.std = braintools.init.param(std, self.varshape)
        self.noise_dt = noise_dt
        self.std_mod = braintools.init.param(std_mod, self.varshape)
        self.frequency = braintools.init.param(frequency, self.varshape)
        self.phase = braintools.init.param(phase, self.varshape)
        self.start = braintools.init.param(start, self.varshape)
        if stop is not None:
            self.stop = braintools.init.param(stop, self.varshape)
        else:
            self.stop = None
        self.origin = braintools.init.param(origin, self.varshape)
        self.seed = seed
        self.rng = brainstate.random.default_rng(self.seed)

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize RNG and internal state buffers for piecewise noise updates.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional batch dimension forwarded to :func:`braintools.init.param`
            when allocating ``current_amp``. ``None`` keeps unbatched state.
            Default is ``None``.
        **kwargs : Any
            Extra keyword arguments accepted for API compatibility with
            :class:`brainstate.nn.Dynamics`. They are currently unused.


        Raises
        ------
        TypeError
            If ``seed`` cannot be interpreted by :func:`jax.random.PRNGKey`.
        ValueError
            If ``batch_size`` or shape metadata is incompatible with
            :func:`braintools.init.param`.

        Notes
        -----
        The PRNG key is stored as a plain Python/JAX attribute rather than a
        :class:`brainstate.ShortTermState`, meaning it is **not** managed by
        the brainstate state-management system and will not be checkpointed
        automatically. Reproducible runs therefore require re-calling
        ``init_state`` with the same ``seed`` before each simulation.

        See Also
        --------
        noise_generator.update : Uses ``_rng_key``, ``current_amp``, and
            ``_step_counter`` populated by this method.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate
           >>> import brainunit as u
           >>> from brainpy.state import noise_generator
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     gen = noise_generator(
           ...         in_size=2,
           ...         std=50.0 * u.pA,
           ...         seed=7,
           ...     )
           ...     gen.init_state()
        """
        # Current noise amplitude (piecewise constant)
        amp = braintools.init.param(braintools.init.Constant(0. * u.pA), self.varshape, batch_size)
        self.current_amp = brainstate.ShortTermState(amp)

        # Step counter for noise update interval tracking
        self._step_counter = brainstate.ShortTermState(jnp.array(0, dtype=jnp.int32))

    def update(self):
        r"""Advance the generator one simulation step and return current output.

        Returns
        -------
        out : jax.Array
            Current-like quantity with shape ``self.varshape``. If active,
            values equal the cached piecewise-constant amplitude sampled from
            ``mean + N(0,1) * effective_std``; otherwise values are zero.

        Raises
        ------
        KeyError
            If environment keys ``'t'`` or ``'dt'`` are missing.
        TypeError
            If unit conversions/comparisons are invalid (for example
            incompatible units in ``noise_dt``, ``dt``, or time bounds).
        ZeroDivisionError
            If ``round(noise_dt / dt)`` is ``0`` and modulo scheduling is
            evaluated with zero divisor.

        Notes
        -----
        The update proceeds in four phases each call:

        1. **Step scheduling** -- ``noise_dt`` is resolved to a whole number of
           simulation steps ``dt_steps = round(noise_dt / dt)``.  A boolean
           flag ``need_update = (step_counter % dt_steps) == 0`` gates whether
           a new amplitude is drawn.
        2. **Effective standard deviation** -- computed as

           .. math::

               \sigma_{\mathrm{eff}} =
               \sqrt{\max\!\left(\sigma^2 +
               \sigma_{\mathrm{mod}}^2 \sin(\omega t + \phi),\, 0\right)}

           using :func:`u.math.maximum` before :func:`u.math.sqrt` so the
           radicand is always non-negative.

        3. **Sample draw** -- ``noise = jax.random.normal(subkey, varshape)``;
           the PRNG key is advanced every call regardless of ``need_update``.
        4. **Masked update** -- ``current_amp`` retains its previous value on
           steps where ``need_update`` is ``False``, avoiding redundant draws
           while keeping the sample schedule deterministic.

        The activity window is ``origin + start <= t < origin + stop``
        (lower-bounded only when ``stop is None``). While inactive the output
        is exactly zero regardless of ``current_amp``.

        See Also
        --------
        noise_generator.init_state : Must be called before the first update.
        noise_generator : Class-level parameter definitions and model equations.
        ac_generator.update : Windowed sinusoidal-current update rule.

        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()

        # Determine noise update interval
        if self.noise_dt is not None:
            noise_dt = self.noise_dt
        else:
            noise_dt = dt

        # Determine noise update interval in steps
        dt_steps = jnp.int32(jnp.round(noise_dt / dt))

        # Check if we need to draw a new noise sample
        step_count = self._step_counter.value
        need_update = (step_count % dt_steps) == 0

        phi_rad = self.phase * 2.0 * jnp.pi / 360.0
        sin_val = jnp.sin(2.0 * jnp.pi * self.frequency * t + phi_rad)

        # std_eff = sqrt(std^2 + std_mod^2 * sin(omega*t + phi))
        std_sq = self.std * self.std
        std_mod_sq = self.std_mod * self.std_mod

        effective_std_sq = std_sq + std_mod_sq * sin_val
        effective_std = u.math.sqrt(u.math.maximum(effective_std_sq, 0. * u.get_unit(effective_std_sq)))

        # Draw noise: mean + N * effective_std
        noise = self.rng.randn(*self.varshape)
        new_amp = self.mean + noise * effective_std

        # Update current amplitude only when needed
        old_amp = self.current_amp.value
        self.current_amp.value = u.math.where(jnp.broadcast_to(need_update, self.varshape), new_amp, old_amp)

        # Increment step counter
        self._step_counter.value = step_count + 1

        # Check if device is active
        t_start = self.origin + self.start
        if self.stop is not None:
            t_stop = self.origin + self.stop
            active = u.math.logical_and(t >= t_start, t < t_stop)
        else:
            active = t >= t_start

        amp_out = self.current_amp.value * jnp.ones(self.varshape)
        return u.math.where(active, amp_out, u.math.zeros_like(amp_out))
