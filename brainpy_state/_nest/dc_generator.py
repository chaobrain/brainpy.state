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
    r"""DC current generator — NEST-compatible stimulation device.

    Description
    -----------

    ``dc_generator`` produces a constant direct current (DC) which is sent to
    all connected neurons. The current is activated at time ``start`` and
    deactivated at time ``stop``, both relative to ``origin``.

    This is a brainpy.state re-implementation of the NEST simulator device of
    the same name, using NEST-standard parameterization.

    Current output
    ..............

    The device produces a current

    .. math::

        I(t) = \begin{cases}
            \text{amplitude} & \text{if } t_{\text{start}} \leq t < t_{\text{stop}} \\
            0 & \text{otherwise}
        \end{cases}

    where :math:`t_{\text{start}} = \text{origin} + \text{start}` and
    :math:`t_{\text{stop}} = \text{origin} + \text{stop}`.

    The current is constant throughout the active window and is identical for
    all connected post-synaptic neurons. The current directly enters the
    neuron's input current equation, equivalent to the ``I_e`` parameter in
    NEST neuron models.

    Timing convention
    .................

    The active window is the half-open interval
    :math:`[t_{\text{start}},\; t_{\text{stop}})` in terms of the simulation
    time ``t`` (the start of each integration step). Because a step beginning
    at time ``t`` advances the membrane state to ``t + dt``, the first
    observable effect of the current on the neuron's membrane potential appears
    at time :math:`t_{\text{start}} + dt`, and the last effect appears at time
    :math:`t_{\text{stop}}`. This matches the observable behavior of the NEST
    ``dc_generator`` device.

    .. note::

       NEST's ``dc_generator`` documentation notes that it is more efficient
       to use a neuron's built-in ``I_e`` parameter when a constant bias
       current is needed for the entire simulation. The ``dc_generator`` is
       most useful when the current needs to be switched on or off at specific
       times.

    Parameters
    ----------

    The following parameters can be set. Default values match the NEST simulator.

    =============== ================== =============================== ============================================
    **Parameter**   **Default**        **Math equivalent**             **Description**
    =============== ================== =============================== ============================================
    ``in_size``     1                                                  Output size of the generator
    ``amplitude``   0 pA               :math:`I`                       Amplitude of the generated current
    ``start``       0 ms               :math:`t_{\text{start,rel}}`    Activation time relative to ``origin``
    ``stop``        ``None`` (∞)       :math:`t_{\text{stop,rel}}`     Deactivation time relative to ``origin``
    ``origin``      0 ms               :math:`t_0`                     Global time offset
    =============== ================== =============================== ============================================

    Examples
    --------

    Basic usage with an iaf_psc_delta neuron:

    >>> import brainpy.state as bps
    >>> import brainstate
    >>> import brainunit as u
    >>>
    >>> with brainstate.environ.context(dt=0.1 * u.ms):
    ...     dc = bps.dc_generator(amplitude=500. * u.pA,
    ...                           start=10. * u.ms,
    ...                           stop=50. * u.ms)
    ...     neuron = bps.iaf_psc_delta(1)
    ...     neuron.init_state()
    ...
    ...     # In simulation loop:
    ...     for step in range(1000):
    ...         with brainstate.environ.context(t=step * 0.1 * u.ms):
    ...             current = dc.update()
    ...             spk = neuron.update(x=current)

    Multiple generators with different time windows:

    >>> dc1 = bps.dc_generator(amplitude=300. * u.pA,
    ...                        start=0. * u.ms, stop=100. * u.ms)
    >>> dc2 = bps.dc_generator(amplitude=-200. * u.pA,
    ...                        start=50. * u.ms, stop=150. * u.ms)

    References
    ----------
    .. [1] NEST Simulator, ``dc_generator`` device.
           https://nest-simulator.readthedocs.io/en/stable/models/dc_generator.html

    See Also
    --------
    iaf_psc_delta : Leaky integrate-and-fire neuron with delta-shaped PSCs
    SpikeTime : Input neuron group with pre-specified spike times
    PoissonSpike : Poisson spike generator
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
        """Return the current amplitude if the device is active, else zero.

        The device is active when ``origin + start <= t < origin + stop``,
        where ``t`` is the current simulation time read from
        ``brainstate.environ``.

        Returns
        -------
        current : Quantity[pA]
            The output current, shaped ``(in_size,)``.
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
