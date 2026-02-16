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

    Generate a NEST-compatible sinusoidal current and gate it with a
    half-open activity window.

    The emitted current for each output channel is

    .. math::

        I(t) = I_0 + A \sin(\omega t + \phi),

    with :math:`\omega = 2\pi f / 1000` (rad/ms) when :math:`f` is provided in
    Hz and simulation time :math:`t` is in ms.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape of the generator. The produced current has shape
        ``self.varshape`` derived from ``in_size``. Default is ``1``.
    amplitude : ArrayLike, optional
        Sinusoidal amplitude :math:`A` (typically pA). Scalars or arrays are
        accepted and broadcast to ``self.varshape`` by
        :func:`braintools.init.param`. Default is ``0. * u.pA``.
    offset : ArrayLike, optional
        Constant DC offset :math:`I_0` (typically pA), broadcast to
        ``self.varshape``. Default is ``0. * u.pA``.
    frequency : ArrayLike, optional
        Oscillation frequency :math:`f` in Hz (or unitless numeric interpreted
        as Hz). Broadcast to ``self.varshape``. Default is ``0. * u.Hz``.
    phase : ArrayLike, optional
        Phase in degrees using NEST convention. Converted internally as
        :math:`\phi = \text{phase} \cdot 2\pi / 360`. Broadcast to
        ``self.varshape``. Default is ``0.``.
    start : ArrayLike, optional
        Relative activation time in ms. Effective start is
        :math:`t_\mathrm{start} = \mathrm{origin} + \mathrm{start}`.
        Broadcast to ``self.varshape``. Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative deactivation time in ms. Effective stop is
        :math:`t_\mathrm{stop} = \mathrm{origin} + \mathrm{stop}` and the
        device is active for :math:`t < t_\mathrm{stop}`. ``None`` means no
        upper time bound. Default is ``None``.
    origin : ArrayLike, optional
        Global time origin in ms used to shift ``start``/``stop``.
        Broadcast to ``self.varshape``. Default is ``0. * u.ms``.
    name : str or None, optional
        Optional object name passed to :class:`brainstate.nn.Dynamics`.

    Returns
    -------
    out : Any
        A dynamics node whose :meth:`update` method returns a current quantity
        with shape ``self.varshape`` and units inherited from ``amplitude`` and
        ``offset``.

    Raises
    ------
    ValueError
        If a parameter cannot be broadcast to ``in_size`` or violates internal
        shape constraints enforced by :func:`braintools.init.param`.
    TypeError
        If arithmetic with provided unitful/unitless values is invalid during
        parameter initialization or later updates.

    See Also
    --------
    dc_generator : Constant current stimulation device.
    step_current_generator : Piecewise-constant current stimulation.
    noise_generator : Gaussian white-noise current stimulation.

    Notes
    -----
    The NEST reference implementation can be written as a linear oscillator
    rotated exactly each step (Rotter and Diesmann, 1999):

    .. math::

        \begin{pmatrix} y_0^{n+1} \\ y_1^{n+1} \end{pmatrix}
        =
        \begin{pmatrix}
            \cos(\omega h) & -\sin(\omega h) \\
            \sin(\omega h) &  \cos(\omega h)
        \end{pmatrix}
        \begin{pmatrix} y_0^n \\ y_1^n \end{pmatrix},

    with initial state :math:`y_0(0)=A\cos\phi`, :math:`y_1(0)=A\sin\phi`,
    and output :math:`I(t)=y_1(t)+I_0`.

    This implementation computes the equivalent closed-form sinusoid directly
    using :func:`jax.numpy.sin`. Computational implications:

    - The oscillator phase is tied to absolute simulation time ``t``.
      Windowing by ``start``/``stop`` does not reset phase.
    - Per call, work is vectorized and dominated by one sine evaluation and
      one conditional mask over ``self.varshape``.
    - Activity uses the half-open interval
      :math:`[\mathrm{origin}+\mathrm{start},\ \mathrm{origin}+\mathrm{stop})`.

    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 18 17 20 45

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
         - Frequency converted to :math:`\omega = 2\pi f/1000` (rad/ms).
       * - ``phase``
         - ``0.``
         - :math:`\phi_{\mathrm{deg}}`
         - Input phase in degrees converted to radians in the update step.
       * - ``start``, ``stop``
         - ``0. * u.ms``, ``None``
         - :math:`t_{\mathrm{start,rel}}, t_{\mathrm{stop,rel}}`
         - Relative window limits added to ``origin``.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global time offset applied to both window boundaries.

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
        r"""Summary
        -------
        Compute the current at the environment time ``t`` and apply
        ``[origin + start, origin + stop)`` gating.

        Parameters
        ----------
        None
            This method reads simulation time from ``brainstate.environ``
            key ``'t'`` and uses instance parameters set at construction.

        Returns
        -------
        current : brainunit.Quantity
            Current array with shape ``self.varshape``. For active channels,
            values equal
            :math:`\mathrm{offset} + \mathrm{amplitude}\sin(\omega t + \phi)`;
            inactive channels are exactly zero.

        Raises
        ------
        KeyError
            If ``brainstate.environ`` does not define current time ``'t'``.
        TypeError
            If unitful arithmetic is invalid (for example incompatible units
            in ``t``, ``frequency``, ``amplitude``, or ``offset``).

        See Also
        --------
        ac_generator : Parameter definitions and model assumptions.
        dc_generator.update : Windowed constant-current update rule.

        Notes
        -----
        Frequency conversion is performed per call as
        :math:`\omega = 2\pi f/1000` so that ``f`` in Hz and ``t`` in ms
        produce a radian phase argument.

        Phase conversion uses NEST's degree convention:
        :math:`\phi = \mathrm{phase}\cdot 2\pi/360`.

        The oscillator itself is not stateful in this implementation; the
        waveform depends only on absolute ``t``. Consequently, changing the
        activity window does not alter phase continuity. Complexity is
        :math:`O(\prod \text{varshape})` due to broadcasting and masking.

        References
        ----------
        .. [1] NEST Simulator documentation for ``ac_generator``:
               https://nest-simulator.readthedocs.io/en/stable/models/ac_generator.html
        .. [2] Rotter S., Diesmann M. (1999). Exact digital simulation of
               time-invariant linear systems with applications to neuronal
               modeling. *Biol. Cybern.*, 81, 381-402.
               https://doi.org/10.1007/s004220050570

        Examples
        --------
        .. code-block:: python

           >>> import brainstate
           >>> import brainunit as u
           >>> from brainpy.state import ac_generator
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     gen = ac_generator(
           ...         in_size=3,
           ...         amplitude=200.0 * u.pA,
           ...         offset=20.0 * u.pA,
           ...         frequency=250.0 * u.Hz,
           ...         phase=90.0,
           ...         start=1.0 * u.ms,
           ...         stop=3.0 * u.ms,
           ...     )
           ...     with brainstate.environ.context(t=1.0 * u.ms):
           ...         current = gen.update()
           ...     _ = current.shape
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
