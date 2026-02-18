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
    'dc_generator',
]


class dc_generator(brainstate.nn.Dynamics):
    r"""DC current generator -- NEST-compatible stimulation device.

    Generate a constant current pulse and gate it with a half-open activity
    window using NEST-compatible parameter semantics.

    **1. Model equations**

    For each output channel, the generated current is

    .. math::

        I(t) = \begin{cases}
            A & \text{if } t_{\mathrm{start}} \le t < t_{\mathrm{stop}}, \\
            0 & \text{otherwise},
        \end{cases}

    where :math:`A` is ``amplitude`` and

    .. math::

        t_{\mathrm{start}} = t_0 + t_{\mathrm{start,rel}}, \qquad
        t_{\mathrm{stop}}  = t_0 + t_{\mathrm{stop,rel}}.

    If ``stop is None``, then :math:`t_{\mathrm{stop}} = +\infty` and the
    generator runs indefinitely from :math:`t_{\mathrm{start}}` onward.

    **2. Timing semantics, assumptions, and constraints**

    The active interval is the half-open set
    :math:`[t_{\mathrm{start}},\, t_{\mathrm{stop}})`.  Since neuron states are
    advanced from ``t`` to ``t + dt`` in each step, a current enabled at
    :math:`t_{\mathrm{start}}` first affects the membrane trajectory after that
    update (observable at :math:`t_{\mathrm{start}} + dt`); the last active
    update starts at :math:`t_{\mathrm{stop}} - dt`.

    This implementation is stateless: :meth:`update` recomputes a boolean mask
    at each call using the environment time, then applies :func:`u.math.where`.
    Assumptions and constraints:

    - If ``stop <= start`` (after adding ``origin``), the active set is empty
      and the output is identically zero for all ``t``.
    - ``amplitude``, ``start``, ``stop``, and ``origin`` must each be
      broadcastable to ``self.varshape``; the shape check is performed by
      :func:`braintools.init.param` during :meth:`__init__`.
    - Unitless numerics in ``start``, ``stop``, and ``origin`` are treated as
      milliseconds; unitless numerics in ``amplitude`` are treated as pA.

    **3. Computational implications**

    Per-call complexity is :math:`O(\prod \mathrm{varshape})`, dominated by one
    broadcast allocation ``amplitude * ones(varshape)`` and one masked
    selection. No recurrent state is maintained, so the model is fully
    replayable given the same environment time sequence.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification understood by
        :class:`brainstate.nn.Dynamics`. The emitted current shape is
        ``self.varshape`` derived from ``in_size``. Default is ``1``.
    amplitude : ArrayLike, optional
        Constant current amplitude :math:`A` (typically pA). Scalars or arrays
        are accepted and broadcast to ``self.varshape`` via
        :func:`braintools.init.param`. Default is ``0. * u.pA``.
    start : ArrayLike, optional
        Relative start time :math:`t_{\mathrm{start,rel}}` (typically ms),
        broadcast to ``self.varshape``. Effective start is
        ``origin + start`` (inclusive). Default is ``0. * u.ms``.
    stop : ArrayLike or None, optional
        Relative stop time :math:`t_{\mathrm{stop,rel}}` (typically ms),
        broadcast to ``self.varshape`` when provided. Effective stop is
        ``origin + stop`` (exclusive). ``None`` means the pulse never
        deactivates. Default is ``None``.
    origin : ArrayLike, optional
        Time origin :math:`t_0` (typically ms) added to ``start`` and ``stop``,
        broadcast to ``self.varshape``. Default is ``0. * u.ms``.
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
         - Constant current value emitted during the active window.
       * - ``start``
         - ``0. * u.ms``
         - :math:`t_{\mathrm{start,rel}}`
         - Relative start time; effective inclusive lower bound is ``origin + start``.
       * - ``stop``
         - ``None``
         - :math:`t_{\mathrm{stop,rel}}`
         - Relative stop time; effective exclusive upper bound is ``origin + stop``.
       * - ``origin``
         - ``0. * u.ms``
         - :math:`t_0`
         - Global offset applied to both window boundaries.

    Raises
    ------
    ValueError
        If ``in_size`` is invalid or if any array-like parameter cannot be
        broadcast to ``self.varshape`` by :func:`braintools.init.param`.
    TypeError
        If invalid unitful/unitless arithmetic is provided (for example, values
        with incompatible units in current or time comparisons).

    Notes
    -----
    NEST recommends using neuron parameter ``I_e`` when a constant bias current
    is needed throughout the full simulation. Use ``dc_generator`` when the
    current must be switched on/off at specific simulation times.

    See Also
    --------
    ac_generator : Sinusoidal current stimulation device.
    step_current_generator : Piecewise-constant current stimulation.
    noise_generator : Gaussian white-noise current stimulation.

    References
    ----------
    .. [1] NEST Simulator documentation for ``dc_generator``:
           https://nest-simulator.readthedocs.io/en/stable/models/dc_generator.html

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     gen = brainpy.state.dc_generator(
       ...         in_size=1,
       ...         amplitude=500.0 * u.pA,
       ...         start=10.0 * u.ms,
       ...         stop=50.0 * u.ms,
       ...     )
       ...     with brainstate.environ.context(t=10.0 * u.ms):
       ...         current = gen.update()
       ...     _ = current.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainunit as u
       >>> dc1 = brainpy.state.dc_generator(
       ...     amplitude=300.0 * u.pA,
       ...     start=0.0 * u.ms,
       ...     stop=100.0 * u.ms,
       ... )
       >>> dc2 = brainpy.state.dc_generator(
       ...     amplitude=-200.0 * u.pA,
       ...     start=50.0 * u.ms,
       ...     stop=150.0 * u.ms,
       ... )
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        amplitude: ArrayLike = 0. * u.pA,
        start: ArrayLike = 0. * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0. * u.ms,
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        # parameters
        self.amplitude = braintools.init.param(amplitude, self.varshape)
        self.start = braintools.init.param(start, self.varshape)
        if stop is not None:
            self.stop = braintools.init.param(stop, self.varshape)
        else:
            self.stop = None
        self.origin = braintools.init.param(origin, self.varshape)

    def update(self):
        r"""Compute the window-gated constant current at environment time ``t``.

        Returns
        -------
        current : jax.Array
            Current-like quantity with shape ``self.varshape`` and units
            inherited from ``amplitude``. Values equal ``amplitude`` on channels
            where ``origin + start <= t < origin + stop`` (or
            ``t >= origin + start`` when ``stop is None``), and zero elsewhere.

        Raises
        ------
        KeyError
            If the environment time key ``'t'`` is not available in
            ``brainstate.environ``.
        TypeError
            If ``t``, ``start``, ``stop``, or ``origin`` cannot be compared due
            to incompatible units/dtypes.

        Notes
        -----
        Start is inclusive and stop is exclusive, matching NEST semantics.
        If ``stop <= start`` (after adding ``origin``), the active set is empty
        and the output is identically zero for all ``t``. The model carries no
        internal state, so repeated calls with the same environment time produce
        identical results.

        See Also
        --------
        dc_generator : Class-level parameter definitions and model equations.
        ac_generator.update : Windowed sinusoidal-current update rule.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate
           >>> import brainunit as u
           >>> from brainpy.state import dc_generator
           >>> with brainstate.environ.context(dt=0.1 * u.ms):
           ...     gen = dc_generator(amplitude=250.0 * u.pA, start=2.0 * u.ms, stop=4.0 * u.ms)
           ...     with brainstate.environ.context(t=3.0 * u.ms):
           ...         current = gen.update()
           ...     _ = current.shape
        """
        t = brainstate.environ.get('t')
        t_start = self.origin + self.start
        if self.stop is not None:
            t_stop = self.origin + self.stop
            active = u.math.logical_and(t >= t_start, t < t_stop)
        else:
            active = t >= t_start
        # Broadcast amplitude to varshape so the output always has the
        # correct shape, even when amplitude was given as a scalar.
        amplitude = self.amplitude * jnp.ones(self.varshape)
        return u.math.where(active, amplitude, u.math.zeros_like(amplitude))
