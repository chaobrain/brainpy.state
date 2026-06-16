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

from brainpy_state._nest_base.base import NESTDevice

__all__ = [
    'ac_generator',
]


class ac_generator(NESTDevice):
    r"""AC current generator -- NEST-compatible stimulation device.

    Generate a sinusoidal current with a constant DC offset and gate the output
    with a half-open activity window using NEST-compatible parameter semantics.

    **1. Model equations**

    For each output channel, the emitted current is

    .. math::

        I(t) = \begin{cases}
            I_0 + A\sin(\omega t + \phi) & \text{if } t_{\mathrm{start}} \le t
                < t_{\mathrm{stop}}, \\
            0 & \text{otherwise},
        \end{cases}

    where :math:`\omega = 2\pi f / 1000` (rad/ms) when :math:`f` is given in
    Hz and simulation time :math:`t` is in ms, and

    .. math::

        t_{\mathrm{start}} = t_0 + t_{\mathrm{start,rel}}, \qquad
        t_{\mathrm{stop}}  = t_0 + t_{\mathrm{stop,rel}}.

    If ``stop is None``, then :math:`t_{\mathrm{stop}} = +\infty`.

    **2. Rotation-matrix interpretation**

    The NEST reference implementation propagates the oscillator state with an
    exact rotation matrix (Rotter and Diesmann, 1999):

    .. math::

        \begin{pmatrix} y_0^{n+1} \\ y_1^{n+1} \end{pmatrix}
        =
        \begin{pmatrix}
            \cos(\omega h) & -\sin(\omega h) \\
            \sin(\omega h) &  \cos(\omega h)
        \end{pmatrix}
        \begin{pmatrix} y_0^n \\ y_1^n \end{pmatrix},

    with initial state :math:`y_0(0) = A\cos\phi`, :math:`y_1(0) = A\sin\phi`
    and output :math:`I(t) = y_1(t) + I_0`. This implementation instead
    evaluates the equivalent closed-form expression :math:`A\sin(\omega t +
    \phi)` directly via :func:`jax.numpy.sin`, which is numerically identical
    but stateless.

    **3. Timing semantics and computational implications**

    The active interval is the half-open set
    :math:`[t_{\mathrm{start}},\, t_{\mathrm{stop}})`. Since neuron states are
    advanced from ``t`` to ``t + dt`` in each step, a current enabled at
    :math:`t_{\mathrm{start}}` first affects the membrane trajectory after that
    update (observable at :math:`t_{\mathrm{start}} + dt`); the last active
    update starts at :math:`t_{\mathrm{stop}} - dt`. Because the phase is tied
    to absolute simulation time ``t``, windowing by ``start``/``stop`` does
    *not* reset the oscillator phase. Per-call complexity is
    :math:`O(\prod \mathrm{varshape})`, dominated by one sine evaluation and
    one conditional mask.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification understood by
        :class:`brainstate.nn.Dynamics`. The emitted current shape is
        ``self.varshape`` derived from ``in_size``. Default is ``1``.
    amplitude : ArrayLike, optional
        Sinusoidal amplitude :math:`A` (typically pA). Scalars or arrays are
        accepted and broadcast to ``self.varshape`` via
        :func:`braintools.init.param`. Default is ``0. * u.pA``.
    offset : ArrayLike, optional
        Constant DC offset :math:`I_0` added to the sinusoid (typically pA),
        broadcast to ``self.varshape``. Default is ``0. * u.pA``.
    frequency : ArrayLike, optional
        Oscillation frequency :math:`f` in Hz (or a unitless numeric
        interpreted as Hz). Converted internally to
        :math:`\omega = 2\pi f / 1000` (rad/ms). Broadcast to
        ``self.varshape``. Default is ``0. * u.Hz``.
    phase : ArrayLike, optional
        Initial phase :math:`\phi_{\mathrm{deg}}` in degrees (NEST convention).
        Converted internally as :math:`\phi = \phi_{\mathrm{deg}} \cdot 2\pi /
        360`. Stored as a dimensionless scalar or array broadcast to
        ``self.varshape``. Default is ``0.``.
    start : ArrayLike, optional
        Relative activation time :math:`t_{\mathrm{start,rel}}` (typically ms),
        broadcast to ``self.varshape``. Effective start is
        ``origin + start``. Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative deactivation time :math:`t_{\mathrm{stop,rel}}` (typically
        ms), broadcast to ``self.varshape`` when provided. Effective stop is
        ``origin + stop`` and the upper bound is exclusive. ``None`` means the
        sinusoid is never deactivated. Default is ``None``.
    origin : ArrayLike, optional
        Global time origin :math:`t_0` (typically ms) added to both ``start``
        and ``stop``, broadcast to ``self.varshape``. Default is ``0. * u.ms``.
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
       * - ``amplitude``
         - ``0. * u.pA``
         - :math:`A`
         - Peak sinusoidal excursion in current units (typically pA).
       * - ``offset``
         - ``0. * u.pA``
         - :math:`I_0`
         - Constant baseline current added to the sinusoid.
       * - ``frequency``
         - ``0. * u.Hz``
         - :math:`f`
         - Frequency in Hz; converted to :math:`\omega = 2\pi f/1000` rad/ms.
       * - ``phase``
         - ``0.``
         - :math:`\phi_{\mathrm{deg}}`
         - Input phase in degrees; converted to radians each update step.
       * - ``start``
         - ``0. * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative start time; effective lower bound is ``origin + start``.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative stop time; effective upper bound is ``origin + stop``.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global offset applied to both window boundaries.

    Raises
    ------
    ValueError
        If ``in_size`` is invalid or any parameter cannot be broadcast to
        ``self.varshape`` by :func:`braintools.init.param`.
    TypeError
        If unitful/unitless arithmetic is invalid during parameter
        initialization (e.g., incompatible units in ``amplitude`` or
        ``offset``).

    See Also
    --------
    dc_generator : Constant current stimulation device.
    step_current_generator : Piecewise-constant current stimulation.
    noise_generator : Gaussian white-noise current stimulation.

    References
    ----------
    .. [1] Rotter S., Diesmann M. (1999). Exact digital simulation of
           time-invariant linear systems with applications to neuronal
           modeling. *Biol. Cybern.*, 81, 381-402.
           https://doi.org/10.1007/s004220050570
    .. [2] NEST Simulator documentation for ``ac_generator``:
           https://nest-simulator.readthedocs.io/en/stable/models/ac_generator.html

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     stim = brainpy.state.ac_generator(
       ...         in_size=1,
       ...         amplitude=500.0 * u.pA,
       ...         offset=100.0 * u.pA,
       ...         frequency=100.0 * u.Hz,
       ...         phase=30.0,
       ...         start=5.0 * u.ms,
       ...         stop=50.0 * u.ms,
       ...     )
       ...     with brainstate.environ.context(t=10.0 * u.ms):
       ...         current = stim.update()
       ...     _ = current.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> ac1 = brainpy.state.ac_generator(
       ...     amplitude=200.0 * u.pA,
       ...     offset=50.0 * u.pA,
       ...     frequency=40.0 * u.Hz,
       ...     phase=0.0,
       ... )
       >>> ac2 = brainpy.state.ac_generator(
       ...     amplitude=100.0 * u.pA,
       ...     offset=0.0 * u.pA,
       ...     frequency=80.0 * u.Hz,
       ...     phase=90.0,
       ...     start=10.0 * u.ms,
       ...     stop=60.0 * u.ms,
       ... )
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
        r"""Compute the window-gated sinusoidal current at environment time ``t``.

        Returns
        -------
        current : jax.Array
            Current-like quantity with shape ``self.varshape``. For channels
            where ``origin + start <= t < origin + stop`` (or
            ``t >= origin + start`` when ``stop is None``), values equal
            :math:`I_0 + A\sin(\omega t + \phi)` where
            :math:`\omega = 2\pi f / 1000` (rad/ms) and
            :math:`\phi = \phi_{\mathrm{deg}} \cdot 2\pi / 360` (rad).
            Inactive channels are exactly zero.

        Raises
        ------
        KeyError
            If the environment time key ``'t'`` is not available in
            ``brainstate.environ``.
        TypeError
            If ``t``, ``frequency``, ``amplitude``, or ``offset`` carry
            incompatible units preventing valid arithmetic.

        Notes
        -----
        Frequency and phase conversions are performed per call:

        .. math::

            \omega = \frac{2\pi f}{1000} \, (\text{rad/ms}), \qquad
            \phi   = \frac{\phi_{\mathrm{deg}} \cdot 2\pi}{360} \, (\text{rad}).

        The waveform depends only on absolute ``t``; the oscillator carries no
        internal state. Entering and leaving the activity window therefore does
        not reset or shift the phase. Start is inclusive and stop is exclusive,
        matching NEST semantics. If ``stop <= start`` (after adding ``origin``),
        the active set is empty and the output is always zero.

        See Also
        --------
        ac_generator : Class-level parameter definitions and model equations.
        dc_generator.update : Windowed constant-current update rule.
        """
        t = brainstate.environ.get('t')

        # Convert phase from degrees to radians
        phi_rad = self.phase * 2.0 * jnp.pi / 360.0

        # Compute sine current: amplitude * sin(omega * t + phi) + offset
        I_ac = self.amplitude * jnp.sin(2.0 * jnp.pi * self.frequency * t + phi_rad) + self.offset

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
