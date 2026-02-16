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
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

__all__ = [
    'noise_generator',
]


class noise_generator(brainstate.nn.Dynamics):
    r"""Gaussian white noise current generator -- NEST-compatible stimulation device.

    Description
    -----------

    ``noise_generator`` produces a piecewise-constant Gaussian "white" noise
    current. The current changes at a user-defined interval :math:`\delta` and
    is given by

    .. math::

        I(t) = \mu + N_j \sigma \quad \text{for } t_0 + j\delta < t \le t_0 + (j+1)\delta

    where :math:`N_j` are Gaussian random numbers with unit standard deviation,
    :math:`\mu` is the mean current, :math:`\sigma` is the standard deviation,
    and :math:`t_0` is the device onset time.

    Additionally, a sinusoidally modulated term can be added to the standard
    deviation of the noise:

    .. math::

        I(t) = \mu + N_j \sqrt{\sigma^2 + \sigma_{\text{mod}}^2 \sin(\omega t + \phi)}

    The effect of the noise current on a leaky integrate-and-fire neuron with
    time constant :math:`\tau_m` and capacitance :math:`C_m` produces a membrane
    potential variance of

    .. math::

        \Sigma^2 = \frac{\delta \tau_m \sigma^2}{2 C_m^2}

    for :math:`\delta \ll \tau_m`.

    This is a brainpy.state re-implementation of the NEST simulator device of
    the same name.

    .. note::

       Unlike NEST, where each target neuron receives a different random current,
       this implementation generates a single random current per call to
       ``update()``. To provide independent noise to multiple neurons, create
       separate ``noise_generator`` instances.

    Parameters
    ----------

    The following parameters can be set. Default values match the NEST simulator.

    =============== ================== =============================== ============================================
    **Parameter**   **Default**        **Math equivalent**             **Description**
    =============== ================== =============================== ============================================
    ``in_size``     1                                                  Output size of the generator
    ``mean``        0 pA               :math:`\mu`                     Mean current
    ``std``         0 pA               :math:`\sigma`                  Standard deviation of current
    ``noise_dt``    ``None``                                           Interval between noise updates (ms).
                                                                       Defaults to simulation dt.
    ``std_mod``     0 pA               :math:`\sigma_{\text{mod}}`     Modulation amplitude of std
    ``frequency``   0 Hz               :math:`f`                       Frequency of sine modulation
    ``phase``       0 deg              :math:`\phi_{\text{deg}}`       Phase of sine modulation (0--360 deg)
    ``start``       0 ms               :math:`t_{\text{start,rel}}`    Activation time relative to ``origin``
    ``stop``        ``None`` (inf)     :math:`t_{\text{stop,rel}}`     Deactivation time relative to ``origin``
    ``origin``      0 ms               :math:`t_0`                     Global time offset
    ``seed``        ``None``                                           Random seed for reproducibility
    =============== ================== =============================== ============================================

    Examples
    --------

    Basic usage:

    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    ...     ng = brainpy.state.noise_generator(mean=0. * u.pA,
    ...                              std=100. * u.pA,
    ...                              seed=42)
    ...     neuron = brainpy.state.iaf_psc_delta(1)
    ...     neuron.init_state()
    ...
    ...     for step in range(1000):
    ...         with brainstate.environ.context(t=step * 0.1 * u.ms):
    ...             current = ng.update()
    ...             spk = neuron.update(x=current)

    References
    ----------
    .. [1] NEST Simulator, ``noise_generator`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/noise_generator.html

    See Also
    --------
    dc_generator : Constant current generator
    ac_generator : Sinusoidal current generator
    step_current_generator : Piecewise constant current generator
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

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize the RNG key and current amplitude state."""
        if self.seed is not None:
            self._rng_key = jax.random.PRNGKey(self.seed)
        else:
            self._rng_key = jax.random.PRNGKey(0)

        # Current noise amplitude (piecewise constant)
        amp = braintools.init.param(
            braintools.init.Constant(0. * u.pA), self.varshape, batch_size
        )
        self.current_amp = brainstate.ShortTermState(amp)

        # Step counter for noise update interval tracking
        self._step_counter = brainstate.ShortTermState(jnp.array(0, dtype=jnp.int32))

    def update(self):
        """Return the noise current at the current simulation time.

        The noise current is piecewise constant, changing at intervals of
        ``noise_dt`` (defaults to simulation ``dt``). At each change point, a
        new Gaussian random number is drawn.

        Returns
        -------
        current : Quantity[pA]
            The output noise current, shaped ``(in_size,)``.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()

        # Determine noise update interval
        if self.noise_dt is not None:
            noise_dt = self.noise_dt
        else:
            noise_dt = dt

        # Determine noise update interval in steps
        if u.is_unitless(noise_dt) and u.is_unitless(dt):
            dt_steps = jnp.int32(jnp.round(noise_dt / dt))
        else:
            dt_steps = jnp.int32(jnp.round(
                (noise_dt / u.ms) / (dt / u.ms)
            ))

        # Check if we need to draw a new noise sample
        step_count = self._step_counter.value
        need_update = (step_count % dt_steps) == 0

        # Advance RNG key
        self._rng_key, subkey = jax.random.split(self._rng_key)

        # Compute the effective standard deviation (with optional modulation)
        if u.is_unitless(t):
            t_ms = t
        else:
            t_ms = t / u.ms

        freq_val = self.frequency
        if u.is_unitless(freq_val):
            freq_ms = freq_val
        else:
            freq_ms = freq_val / u.Hz

        omega = 2.0 * jnp.pi * freq_ms / 1000.0
        phi_rad = self.phase * 2.0 * jnp.pi / 360.0

        sin_val = jnp.sin(omega * t_ms + phi_rad)

        # std_eff = sqrt(std^2 + std_mod^2 * sin(omega*t + phi))
        std_sq = self.std * self.std
        std_mod_sq = self.std_mod * self.std_mod

        if u.is_unitless(std_sq):
            effective_std_sq = std_sq + std_mod_sq * sin_val
            effective_std = u.math.sqrt(u.math.maximum(effective_std_sq, 0.))
        else:
            effective_std_sq = std_sq + std_mod_sq * sin_val
            # Ensure non-negative before sqrt
            zero = u.math.zeros_like(effective_std_sq)
            effective_std_sq = u.math.maximum(effective_std_sq, zero)
            effective_std = u.math.sqrt(effective_std_sq)

        # Draw noise: mean + N * effective_std
        noise = jax.random.normal(subkey, shape=self.varshape)
        new_amp = self.mean + noise * effective_std

        # Update current amplitude only when needed
        old_amp = self.current_amp.value
        self.current_amp.value = u.math.where(
            jnp.broadcast_to(need_update, self.varshape),
            new_amp, old_amp
        )

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
