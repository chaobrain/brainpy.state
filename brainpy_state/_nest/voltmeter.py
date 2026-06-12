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

from typing import Optional

import saiunit as u
from brainstate.typing import ArrayLike, Size

from .multimeter import multimeter

__all__ = [
    'voltmeter',
]


class voltmeter(multimeter):
    r"""NEST-compatible ``voltmeter`` recording device.

    A ``voltmeter`` is a :class:`multimeter` preset to record only the membrane
    potential ``V_m``. It exists purely so that NEST scripts that create a
    ``voltmeter`` port verbatim; it carries no behaviour beyond fixing
    ``record_from`` to ``('V_m',)``.

    On the explicit :class:`~brainpy_state.network.Simulator` API the voltmeter
    is connected in NEST's reversed direction --- ``sim.connect(voltmeter,
    neuron)`` --- and the simulator records ``V_m`` per step by tapping the
    neuron's membrane :class:`~brainstate.HiddenState` inside its ``for_loop``,
    rather than driving this device's imperative recording path. The trace is
    then read with ``res.trace(voltmeter, 'V_m')``.

    Parameters
    ----------
    in_size : Size, optional
        Output size/shape specification. Default is ``1``.
    interval : ArrayLike, optional
        Recording interval (ms) forwarded to :class:`multimeter`. Default is
        ``1.0 * u.ms``.
    offset : ArrayLike, optional
        Interval offset (ms). Default is ``0.0 * u.ms``.
    start : ArrayLike, optional
        Relative activation time (ms). Default is ``0.0 * u.ms``.
    stop : ArrayLike or None, optional
        Relative deactivation time (ms). ``None`` means no upper bound. Default
        is ``None``.
    origin : ArrayLike, optional
        Time origin (ms) added to ``start``/``stop``. Default is ``0.0 * u.ms``.
    time_in_steps : bool, optional
        Stamp recorded events in integer steps rather than ms. Default is
        ``False``.
    frozen : bool, optional
        Must be ``False`` --- recorders cannot be frozen. Default is ``False``.
    name : str or None, optional
        Optional node name.

    See Also
    --------
    multimeter : The general analog recorder this device specialises.

    References
    ----------
    .. [1] NEST Simulator documentation for ``voltmeter``:
           https://nest-simulator.readthedocs.io/en/stable/models/voltmeter.html

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> vm = brainpy.state.voltmeter()
       >>> vm.record_from
       ('V_m',)
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size = 1,
        interval: ArrayLike = 1.0 * u.ms,
        offset: ArrayLike = 0.0 * u.ms,
        start: ArrayLike = 0.0 * u.ms,
        stop: ArrayLike = None,
        origin: ArrayLike = 0.0 * u.ms,
        time_in_steps: bool = False,
        frozen: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            in_size=in_size,
            record_from=('V_m',),
            interval=interval,
            offset=offset,
            start=start,
            stop=stop,
            origin=origin,
            time_in_steps=time_in_steps,
            frozen=frozen,
            name=name,
        )
