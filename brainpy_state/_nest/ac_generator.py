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
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

__all__ = [
    'ac_generator',
]


class ac_generator(brainstate.nn.Dynamics):
    r"""AC current generator -- NEST-compatible stimulation device.

    Description
    -----------

    ``ac_generator`` produces a sinusoidal alternating current (AC) which is
    sent to all connected neurons. The current is given by

    .. math::

        I(t) = \mathrm{offset} + \mathrm{amplitude} \cdot \sin(\omega t + \phi)

    where

    .. math::

        \omega  = 2 \pi \cdot \mathrm{frequency} / 1000 \quad (\text{converting Hz to 1/ms}) \\
        \phi = \frac{\mathrm{phase}}{180} \cdot \pi

    This is a brainpy.state re-implementation of the NEST simulator device of
    the same name, using NEST-standard parameterization.

    Implementation
    ..............

    Internally, the AC signal is generated using an exact matrix rotation
    method (Rotter & Diesmann, 1999) rather than evaluating the sine function
    at each step. Two state variables ``y_0`` and ``y_1`` are maintained:

    .. math::

        \begin{pmatrix} y_0^{n+1} \\ y_1^{n+1} \end{pmatrix}
        = \begin{pmatrix}
            \cos(\omega h) & -\sin(\omega h) \\
            \sin(\omega h) &  \cos(\omega h)
          \end{pmatrix}
        \begin{pmatrix} y_0^n \\ y_1^n \end{pmatrix}

    where :math:`h` is the time step. The initial conditions are:

    .. math::

        y_0(0) = \mathrm{amplitude} \cdot \cos(\phi) \\
        y_1(0) = \mathrm{amplitude} \cdot \sin(\phi)

    The output current is :math:`I = y_1 + \mathrm{offset}` when the device
    is active.

    However, this re-implementation uses the direct sinusoidal formula
    for simplicity and JAX compatibility, as JAX's ``jnp.sin`` is efficient
    and differentiable. The oscillator runs continuously; setting ``start``
    and ``stop`` only windows the output current -- it does not shift
    the time axis.

    Timing convention
    .................

    The active window is the half-open interval
    :math:`[t_{\text{start}},\; t_{\text{stop}})` in terms of the simulation
    time ``t``. Setting ``start`` and ``stop`` only windows the current as
    defined above. It does not shift the time axis.

    Parameters
    ----------

    The following parameters can be set. Default values match the NEST simulator.

    =============== ================== =============================== ============================================
    **Parameter**   **Default**        **Math equivalent**             **Description**
    =============== ================== =============================== ============================================
    ``in_size``     1                                                  Output size of the generator
    ``amplitude``   0 pA               :math:`A`                       Amplitude of the sine current
    ``offset``      0 pA               :math:`I_0`                     Constant amplitude offset (DC component)
    ``frequency``   0 Hz               :math:`f`                       Frequency of the AC signal
    ``phase``       0 deg              :math:`\phi_{\text{deg}}`       Phase of sine current (0--360 deg)
    ``start``       0 ms               :math:`t_{\text{start,rel}}`    Activation time relative to ``origin``
    ``stop``        ``None`` (inf)     :math:`t_{\text{stop,rel}}`     Deactivation time relative to ``origin``
    ``origin``      0 ms               :math:`t_0`                     Global time offset
    =============== ================== =============================== ============================================

    Examples
    --------

    Basic usage with a neuron:

    >>> import brainpy.state as bps
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    ...     ac = bps.ac_generator(amplitude=500. * u.pA,
    ...                           offset=100. * u.pA,
    ...                           frequency=100. * u.Hz,
    ...                           phase=0.,
    ...                           start=5. * u.ms,
    ...                           stop=50. * u.ms)
    ...     neuron = bps.iaf_psc_delta(1)
    ...     neuron.init_state()
    ...
    ...     for step in range(1000):
    ...         with brainstate.environ.context(t=step * 0.1 * u.ms):
    ...             current = ac.update()
    ...             spk = neuron.update(x=current)

    References
    ----------
    .. [1] Rotter S and Diesmann M (1999). Exact digital simulation of time-
           invariant linear systems with applications to neuronal modeling,
           Biol. Cybern. 81, 381-402. DOI: https://doi.org/10.1007/s004220050570
    .. [2] NEST Simulator, ``ac_generator`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/ac_generator.html

    See Also
    --------
    dc_generator : Constant current generator
    step_current_generator : Piecewise constant current generator
    noise_generator : Gaussian white noise current generator
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        amplitude: ArrayLike = 0. * u.pA,
        offset: ArrayLike = 0. * u.pA,
        frequency: ArrayLike = 0. * u.Hz,
        phase: ArrayLike = 0.,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        # parameters
        self.amplitude = braintools.init.param(amplitude, self.varshape)
        self.offset = braintools.init.param(offset, self.varshape)
        self.frequency = braintools.init.param(frequency, self.varshape)
        # phase in degrees -- store as-is (NEST convention)
        self.phase = braintools.init.param(phase, self.varshape)
        self.start = braintools.init.param(start, self.varshape)
        if stop is not None:
            self.stop = braintools.init.param(stop, self.varshape)
        else:
            self.stop = None
        self.origin = braintools.init.param(origin, self.varshape)

    def update(self):
        """Return the AC current at the current simulation time.

        The current is computed as:

        .. math::

            I(t) = \\text{offset} + \\text{amplitude} \\cdot \\sin(\\omega t + \\phi)

        The device is active when ``origin + start <= t < origin + stop``.
        When inactive, the output is zero.

        Returns
        -------
        current : Quantity[pA]
            The output current, shaped ``(in_size,)``.
        """
        t = brainstate.environ.get('t')

        # Convert frequency from Hz to angular frequency in rad/ms
        # omega = 2 * pi * freq / 1000 (since t is in ms, freq is in Hz)
        freq_val = self.frequency
        if u.is_unitless(freq_val):
            omega = 2.0 * jnp.pi * freq_val / 1000.0
        else:
            # frequency in Hz -> convert to 1/ms
            freq_ms = freq_val / u.Hz  # dimensionless number in Hz
            omega = 2.0 * jnp.pi * freq_ms / 1000.0

        # Convert phase from degrees to radians
        phi_rad = self.phase * 2.0 * jnp.pi / 360.0

        # Get t in ms (dimensionless)
        if u.is_unitless(t):
            t_ms = t
        else:
            t_ms = t / u.ms

        # Compute sine current: amplitude * sin(omega * t + phi) + offset
        I_ac = self.amplitude * jnp.sin(omega * t_ms + phi_rad) + self.offset

        # Check if device is active
        t_start = self.origin + self.start
        if self.stop is not None:
            t_stop = self.origin + self.stop
            active = u.math.logical_and(t >= t_start, t < t_stop)
        else:
            active = t >= t_start

        # Broadcast to varshape
        I_ac_full = I_ac * jnp.ones(self.varshape)
        return u.math.where(active, I_ac_full, u.math.zeros_like(I_ac_full))
