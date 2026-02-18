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
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

__all__ = [
    'ignore_and_fire',
]


class ignore_and_fire(NESTNeuron):
    r"""Ignore-and-fire neuron model for generating spikes at fixed intervals.

    Description
    -----------

    The ``ignore_and_fire`` neuron is a neuron model that generates spikes at
    a predefined ``rate`` with a constant inter-spike interval ("fire"),
    irrespective of its inputs ("ignore"). In this simplest version of the
    ``ignore_and_fire`` neuron, the inputs from other neurons or devices are
    not processed at all.

    This is a brainpy.state re-implementation of the NEST simulator model of the
    same name, using NEST-standard parameterization.

    **1. Model equations and dynamics**

    The model's internal state variable, the ``phase``, describes the time to
    the next spike relative to the firing period (the inverse of the ``rate``).

    The firing period (in simulation time steps) is computed as:

    .. math::

        T_{\text{fire}} = \text{round}\!\left(\frac{1}{\text{rate}} \times 1000\right) / dt

    where rate is in spikes/s and the result is expressed in simulation time steps
    (NEST rounds this to the simulation grid via ``Time::get_steps()``).

    The initial phase countdown (in simulation time steps) is computed as:

    .. math::

        N_{\text{phase}} = \text{round}\!\left(\frac{\text{phase}}{\text{rate}}
        \times 1000\right) / dt

    In each update step, the model checks whether the countdown has reached zero:

    - If ``phase_steps == 0``: a spike is emitted and the countdown is reset to
      ``firing_period_steps - 1``.
    - Otherwise: the countdown is decremented by 1.

    To create asynchronous activity for a population of ``ignore_and_fire``
    neurons, the firing phases can be randomly initialized.

    **2. Assumptions, constraints, and computational implications**

    - The model assumes unit-compatible parameters and broadcast-compatible
      shapes against ``self.varshape``.
    - The ``phase`` parameter must satisfy :math:`0 < \text{phase} \le 1`,
      representing fractional position within the firing period.
    - The ``rate`` parameter must be positive. Extremely low rates (< 1/1000 Hz)
      may cause integer overflow when converting to time steps.
    - Per-step compute is :math:`O(\prod \mathrm{varshape})` with vectorized
      elementwise operations (phase countdown and spike emission).
    - All inputs to :meth:`update` are completely ignored; the model fires
      deterministically based solely on its internal clock.

    .. note::

        The ``ignore_and_fire`` neuron is primarily used for neuronal-network
        model verification and validation purposes ("benchmarking"), in
        particular, to evaluate the correctness and performance of connectivity
        generation and inter-neuron communication. It permits an easy scaling
        of the network size and/or connectivity without affecting the output
        spike statistics. The amount of network traffic is predefined by the
        user, and therefore fully controllable and predictable, irrespective
        of the network size and structure.

    .. note::

        This model inherits from :class:`Dynamics` rather than :class:`Neuron`
        because it has no membrane potential, no threshold-based spike
        generation, and no subthreshold dynamics. Surrogate gradients and
        spike reset mechanisms are therefore not applicable.

    Parameters
    ----------
    in_size : Size
        Population shape specification. All neuron parameters are broadcast to
        ``self.varshape`` derived from ``in_size``.
    phase : ArrayLike, optional
        Initial fractional position within the firing period, where 0 means
        immediate firing and 1 means firing after a full period. Must satisfy
        :math:`0 < \text{phase} \le 1`. Scalar or array broadcast-compatible
        with ``varshape``. Unitless. Default is ``1.0``.
    rate : ArrayLike, optional
        Firing rate in spikes per second. Must be positive. Scalar or array
        broadcast-compatible with ``varshape``. Default is ``10. * u.Hz``.
    name : str or None, optional
        Optional node name passed to :class:`brainstate.nn.Dynamics`.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to NEST ``ignore_and_fire``
       :header-rows: 1
       :widths: 22 18 22 38

       * - Parameter
         - Default
         - Math symbol
         - Semantics
       * - ``phase``
         - ``1.0``
         - :math:`\phi \in (0, 1]`
         - Fractional position in firing period; 1.0 = full period delay.
       * - ``rate``
         - ``10.0`` Hz
         - :math:`f` (Hz)
         - Spike rate; firing period :math:`T = 1/f`.

    Attributes
    ----------
    phase_steps : ShortTermState
        Integer countdown to next spike (in simulation time steps). Decrements
        each step; fires when reaching zero.
    firing_period_steps : ShortTermState
        Integer duration of firing period (in simulation time steps). Constant
        after initialization.

    Examples
    --------
    Create an ``ignore_and_fire`` neuron with 10 Hz firing rate:

    .. code-block:: python

        >>> import brainpy
        >>> import brainstate
        >>> import brainunit as u
        >>>
        >>> # Create an ignore_and_fire neuron with 10 Hz firing rate
        >>> neuron = brainpy.state.ignore_and_fire(1, rate=10.0 * u.Hz)
        >>>
        >>> # Initialize the state
        >>> neuron.init_state()
        >>>
        >>> # Step the neuron and check for spikes
        >>> with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
        ...     spike = neuron.update()
        ...     print(f"Spike: {spike}")

    Create a population with random phases for asynchronous activity:

    .. code-block:: python

        >>> import numpy as np
        >>> # Create 100 neurons with random initial phases
        >>> phases = np.random.uniform(0.01, 1.0, size=100)
        >>> neurons = brainpy.state.ignore_and_fire(100, phase=phases, rate=20.0 * u.Hz)

    References
    ----------
    .. [1] NEST Simulator, ``ignore_and_fire`` model.
           https://nest-simulator.readthedocs.io/en/stable/models/ignore_and_fire.html
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        phase: ArrayLike = 1.0,
        rate: ArrayLike = 10. * u.Hz,
        name: str = None,
    ):
        r"""Initialize the ignore_and_fire neuron model.

        Stores parameters, initializes base :class:`Dynamics` state, and
        validates parameter constraints. Does not initialize internal state
        variables (``phase_steps``, ``firing_period_steps``); call
        :meth:`init_state` before simulation.

        Parameters
        ----------
        in_size : Size
            Population shape specification passed to
            :class:`brainstate.nn.Dynamics`. Determines ``self.varshape``.
        phase : ArrayLike, optional
            Initial fractional position within the firing period. Must satisfy
            :math:`0 < \text{phase} \le 1`. Broadcast to ``varshape`` via
            :func:`braintools.init.param`. Default is ``1.0``.
        rate : ArrayLike, optional
            Firing rate in spikes per second. Must be positive. Broadcast to
            ``varshape`` via :func:`braintools.init.param`. Default is
            ``10. * u.Hz``.
        name : str or None, optional
            Optional node name passed to :class:`brainstate.nn.Dynamics`.

        Raises
        ------
        ValueError
            If ``phase`` violates :math:`0 < \text{phase} \le 1` or ``rate``
            is not strictly positive, raised by :meth:`_validate_parameters`.

        See Also
        --------
        init_state : Initialize state variables before simulation.
        update : Perform one simulation time step.
        """
        super().__init__(in_size, name=name)

        # Store parameters
        self.phase = braintools.init.param(phase, self.varshape)
        self.rate = braintools.init.param(rate, self.varshape)

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        r"""Validate ``phase`` and ``rate`` parameters after initialization.

        Ensures that:

        - ``phase`` satisfies :math:`0 < \text{phase} \le 1` element-wise.
        - ``rate`` is strictly positive element-wise.

        Called automatically from :meth:`__init__` after parameter storage.

        Raises
        ------
        ValueError
            If any element of ``phase`` is :math:`\le 0` or :math:`> 1`, with
            message "Phase must be > 0 and <= 1."
        ValueError
            If any element of ``rate`` (after unit conversion to Hz) is
            :math:`\le 0`, with message "Firing rate must be > 0."

        Notes
        -----
        Unit conversion is applied to ``rate`` via
        :func:`brainunit.get_magnitude` before validation. ``phase`` is
        validated as a unitless scalar or array.
        """
        phase = self.phase
        rate = self.rate

        # Convert to raw values for comparison
        phase_val = np.asarray(phase)
        if np.any(phase_val <= 0.0) or np.any(phase_val > 1.0):
            raise ValueError("Phase must be > 0 and <= 1.")

        rate_val = np.asarray(u.get_magnitude(rate))
        if np.any(rate_val <= 0.0):
            raise ValueError("Firing rate must be > 0.")

    def _calc_initial_variables(self, batch_size=None):
        r"""Compute firing period and phase countdown in simulation time steps.

        This method replicates NEST's ``calc_initial_variables_`` semantics,
        which converts rate and phase parameters to integer step counts using
        ``Time::get_steps()`` (nearest-integer rounding to the simulation grid).

        The conversion formulas are:

        .. math::

            T_{\text{fire}} &= \text{round}\!\left(\frac{1000}{\text{rate}} / dt\right) \\
            N_{\text{phase}} &= \text{round}\!\left(\frac{1000 \cdot \text{phase}}{\text{rate}} / dt\right)

        where ``rate`` is in Hz, ``dt`` is in ms, and both results are integer
        step counts.

        Parameters
        ----------
        batch_size : int or None, optional
            Batch dimension size; not currently used by the implementation but
            accepted for API compatibility with :meth:`init_state`.

        Returns
        -------
        firing_period_steps : ndarray
            Integer array with shape ``varshape`` containing firing period
            durations in simulation time steps. Dtype is ``int32``.
        phase_steps : ndarray
            Integer array with shape ``varshape`` containing initial countdown
            values in simulation time steps. Dtype is ``int32``.

        Notes
        -----
        NEST computes these as:

        .. code-block:: cpp

            firing_period_steps = Time(Time::ms(1.0 / rate * 1000.0)).get_steps()
            phase_steps = Time(Time::ms(phase / rate * 1000.0)).get_steps()

        ``Time::get_steps()`` rounds to the nearest simulation time step. We
        replicate this by computing the period/phase in ms, dividing by ``dt``,
        then rounding to the nearest integer via ``np.rint``.
        """
        dt = brainstate.environ.get_dt()
        dt_ms = u.get_magnitude(u.maybe_decimal(dt / u.ms))

        rate_hz = u.get_magnitude(u.maybe_decimal(self.rate / u.Hz))
        phase_val = np.asarray(self.phase)

        # period in ms = 1/rate * 1000
        period_ms = 1.0 / rate_hz * 1000.0
        # NEST uses Time(Time::ms(...)).get_steps() which rounds to nearest step
        firing_period_steps = np.rint(period_ms / dt_ms).astype(np.int32)

        # phase time in ms = phase/rate * 1000
        phase_ms = phase_val / rate_hz * 1000.0
        phase_steps = np.rint(phase_ms / dt_ms).astype(np.int32)

        return firing_period_steps, phase_steps

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize internal state variables for simulation.

        Computes and stores ``firing_period_steps`` and ``phase_steps`` as
        :class:`brainstate.ShortTermState` arrays. Both are derived from
        the ``rate`` and ``phase`` parameters via
        :meth:`_calc_initial_variables`, then broadcast to batch shape if
        ``batch_size`` is provided.

        Parameters
        ----------
        batch_size : int or None, optional
            Batch dimension size. When provided, state arrays are broadcast to
            shape ``(batch_size, *varshape)``. When ``None``, state shape is
            ``varshape``. Default is ``None``.
        **kwargs : dict
            Additional keyword arguments passed to parent
            :meth:`brainstate.nn.Dynamics.init_state` (currently unused).

        Raises
        ------
        ValueError
            If ``phase`` is not in :math:`(0, 1]` or ``rate`` is not positive,
            raised during :meth:`_validate_parameters` called in
            :meth:`__init__`.

        Side Effects
        ------------
        Creates or overwrites the following instance attributes:

        - ``self.firing_period_steps`` : :class:`brainstate.ShortTermState`
        - ``self.phase_steps`` : :class:`brainstate.ShortTermState`

        Notes
        -----
        This method must be called before the first :meth:`update` call,
        typically via ``neuron.init_state()`` or automatically through
        higher-level APIs like ``brainstate.nn.Module.init_all_states()``.
        """
        firing_period_steps, phase_steps = self._calc_initial_variables(batch_size)

        if batch_size is not None:
            firing_period_steps = np.broadcast_to(firing_period_steps, (batch_size,) + self.varshape)
            phase_steps = np.broadcast_to(phase_steps, (batch_size,) + self.varshape)

        self.firing_period_steps = brainstate.ShortTermState(
            jnp.asarray(firing_period_steps, dtype=jnp.int32)
        )
        self.phase_steps = brainstate.ShortTermState(
            jnp.asarray(phase_steps, dtype=jnp.int32)
        )

    def update(self, x=None):
        r"""Update the ignore_and_fire neuron for one simulation time step.

        Decrements the internal phase countdown and emits spikes when the
        countdown reaches zero. All external inputs are completely ignored.

        The update logic follows NEST's deterministic firing schedule:

        1. Check if ``phase_steps == 0``:

           - If yes: emit spike (output 1.0), reset countdown to
             ``firing_period_steps - 1``.
           - If no: emit no spike (output 0.0), decrement countdown by 1.

        2. Update ``self.phase_steps`` with the new countdown value.

        Parameters
        ----------
        x : ArrayLike or None, optional
            Input signal (ignored). Accepted for API compatibility with other
            neuron models but has no effect on the dynamics. Any value or shape
            is permitted; the parameter is never accessed.

        Returns
        -------
        spike : jnp.ndarray
            Float array with shape ``varshape`` (or ``(batch_size, *varshape)``
            if initialized with batching). Contains ``1.0`` at positions where
            a spike is emitted this step and ``0.0`` elsewhere. Dtype is
            ``float32`` or the default JAX float dtype.

        Notes
        -----
        This method must be called after :meth:`init_state`. Calling
        :meth:`update` before initialization will raise an
        :class:`AttributeError` due to missing ``phase_steps`` state.

        The spike output is suitable for direct use as delta-synapse input
        (units of spikes/step) or as a binary event indicator for recording.
        """
        phase_steps = self.phase_steps.value
        firing_period_steps = self.firing_period_steps.value

        # Threshold crossing: phase_steps == 0 means fire
        spike = jnp.where(phase_steps == 0, 1.0, 0.0)

        # Update phase_steps:
        # if fired (phase_steps == 0): reset to firing_period_steps - 1
        # else: decrement by 1
        new_phase_steps = jnp.where(
            phase_steps == 0,
            firing_period_steps - 1,
            phase_steps - 1,
        )

        self.phase_steps.value = new_phase_steps

        return spike
