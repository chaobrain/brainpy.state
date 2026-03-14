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
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

__all__ = [
    'iaf_psc_delta_ps',
]


class iaf_psc_delta_ps(NESTNeuron):
    r"""NEST-compatible ``iaf_psc_delta_ps`` with precise spike timing.

    Description
    -----------

    ``iaf_psc_delta_ps`` is a current-based leaky integrate-and-fire neuron
    with delta-shaped synaptic jumps (weights in mV), exact linear
    subthreshold integration, and precise off-grid spike timing inside each
    global simulation step. The implementation follows NEST
    ``models/iaf_psc_delta_ps.{h,cpp}`` semantics, including event ordering by
    within-step offsets, analytic threshold-crossing localization for
    current-driven spikes, and optional accumulation of refractory-time inputs.

    **1. Linear Membrane Dynamics and Exact Closed-Form Propagator**

    The subthreshold membrane potential dynamics are

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m}
       + \frac{I_\mathrm{ext}(t) + I_e}{C_m},

    with piecewise-constant :math:`I_\mathrm{ext}` over each simulation step.

    Defining :math:`U = V_m - E_L`, :math:`R = \tau_m / C_m`, and a constant
    current over an interval :math:`\Delta t`, exact integration gives

    .. math::

       U(t + \Delta t) = U(t)e^{-\Delta t/\tau_m}
       + R(I_\mathrm{ext}+I_e)\left(1 - e^{-\Delta t/\tau_m}\right).

    The code evaluates this update with ``expm1``-based algebra for numerical
    stability when :math:`\Delta t/\tau_m` is small, which reduces
    cancellation error in fine-step simulations.

    **2. Spike Generation Mechanisms and Precise Spike-Time Derivation**

    Two spike mechanisms are implemented:

    - **Instantaneous event-driven spikes**: if an incoming delta event at
      offset :math:`\delta` pushes :math:`U \ge U_{th}`, spike time is the
      event time exactly.
    - **Current-driven spikes**: if propagation yields :math:`U \ge U_{th}`,
      spike offset is solved analytically from the exact trajectory:

      .. math::

         \Delta t_\mathrm{cross}
         = -\tau_m \log\frac{V_\infty - U}{V_\infty - U_{th}},
         \quad
         V_\infty = R(I_\mathrm{ext}+I_e).

    The model stores:

    - ``last_spike_time``: absolute spike time in ms,
    - ``last_spike_offset``: off-grid offset relative to the right border of
      the current grid step (NEST semantics),
    - ``last_spike_step``: on-grid step index used internally for refractory logic.

    **3. Refractory Handling and Deferred Refractory-Input Accumulation**

    After a spike, membrane potential is reset to ``V_reset`` and clamped during
    the absolute refractory period.

    In NEST ``iaf_psc_delta_ps``, refractory duration in steps is derived as
    ``floor(t_ref / dt)`` (via ``Time(...).get_steps()``) and must be at least one
    simulation step. This implementation enforces the same runtime constraint.

    By default, spikes arriving during refractory are discarded. If
    ``refractory_input=True``, they are accumulated and exponentially damped until
    end of refractoriness, then applied once at refractory release, matching NEST.

    **4. Event Ordering, Assumptions, Constraints, and Computational Implications**

    For each simulation step the update proceeds as follows:

    1. Optional immediate spike if state starts super-threshold.
    2. Process within-step events in offset order (start to end of step):

       - propagate to event time (if non-refractory),
       - check current-driven crossing,
       - apply event jump and check instant crossing.

    3. Propagate remaining interval (if any).
    4. Store new external current input buffer for next step.

    Assumptions and constraints used by the implementation:

    - Parameter tensors are scalar or broadcastable to ``self.varshape``.
    - Required physical inequalities are validated at construction:
      ``V_reset < V_th``, ``C_m > 0``, ``tau_m > 0``, ``t_ref >= 0``, and if
      ``V_min`` is provided then ``V_reset >= V_min``.
    - Runtime requires ``floor(t_ref / dt) >= 1`` and ``dt > 0``.
    - Every precise event offset must satisfy ``0 <= offset <= dt``.

    Computationally, the update iterates scalar-wise over ``np.ndindex``
    across the full state shape and processes all local events in each cell,
    so cost is :math:`O(|\mathrm{state}| \cdot K)` per step for ``K`` events
    (excluding input aggregation).

    Parameters
    ----------
    in_size : Size
        Population shape specification used to derive ``self.varshape``.
        Scalar integer for 1D populations or tuple for multi-dimensional.
    E_L : ArrayLike, optional
        Resting membrane potential :math:`E_L` in mV. Scalar or array-like
        broadcastable to ``self.varshape``. Default is ``-70. * u.mV``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF, broadcastable to
        ``self.varshape``. Must be strictly positive elementwise.
        Default is ``250. * u.pF``.
    tau_m : ArrayLike, optional
        Membrane time constant :math:`\tau_m` in ms, broadcastable to
        ``self.varshape``. Must be strictly positive elementwise.
        Default is ``10. * u.ms``.
    t_ref : ArrayLike, optional
        Absolute refractory duration :math:`t_{ref}` in ms, broadcastable to
        ``self.varshape``. At runtime converted to steps by
        ``floor(t_ref / dt)`` and must produce at least one step.
        Default is ``2. * u.ms``.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_{th}` in mV, broadcastable to ``self.varshape``.
        Default is ``-55. * u.mV``.
    V_reset : ArrayLike, optional
        Reset potential :math:`V_{reset}` in mV, broadcastable to
        ``self.varshape``. Must satisfy ``V_reset < V_th`` elementwise.
        Default is ``-70. * u.mV``.
    I_e : ArrayLike, optional
        Constant external current :math:`I_e` in pA, broadcastable to
        ``self.varshape``. Added to buffered current each propagation segment.
        Default is ``0. * u.pA``.
    V_min : ArrayLike or None, optional
        Optional lower membrane bound :math:`V_{min}` in mV, broadcastable to
        ``self.varshape``. ``None`` disables lower clipping (uses ``-inf``).
        Default is ``None``.
    V_initializer : Callable, optional
        Initializer used by :meth:`init_state` to create membrane state ``V``.
        Must return values unit-compatible with mV and shape-compatible with
        ``self.varshape`` (and optional batch prefix). Default is
        ``braintools.init.Constant(-70. * u.mV)``.
    spk_fun : Callable, optional
        Surrogate spike nonlinearity used by :meth:`get_spike` and returned by
        :meth:`update`. Receives normalized threshold distance tensor.
        Default is ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset mode forwarded to :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` matches NEST hard-reset behavior. Default is ``'hard'``.
    refractory_input : bool, optional
        If ``False``, delta events received during refractory are ignored.
        If ``True``, they are exponentially weighted into
        ``refractory_spike_buffer`` and applied at refractory release.
        Default is ``False``.
    ref_var : bool, optional
        If ``True``, exposes additional state ``self.refractory`` mirroring
        ``self.is_refractory`` for introspection. Default is ``False``.
    name : str or None, optional
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 17 28 14 16 35

       * - Parameter
         - Type / shape / unit
         - Default
         - Math symbol
         - Semantics
       * - ``in_size``
         - :class:`~brainstate.typing.Size`; scalar/tuple
         - required
         - --
         - Defines ``self.varshape`` for all state/parameter broadcasts.
       * - ``E_L``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``-70. * u.mV``
         - :math:`E_L`
         - Resting membrane potential and origin of transformed state ``U``.
       * - ``C_m``
         - ArrayLike, broadcastable (pF), ``> 0``
         - ``250. * u.pF``
         - :math:`C_m`
         - Converts current to membrane-rate contribution.
       * - ``tau_m``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``10. * u.ms``
         - :math:`\tau_m`
         - Leak/relaxation time constant in exact propagator.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), runtime ``floor(t_ref/dt) >= 1``
         - ``2. * u.ms``
         - :math:`t_{ref}`
         - Absolute refractory duration.
       * - ``V_th`` and ``V_reset``
         - ArrayLike, broadcastable (mV), with ``V_reset < V_th``
         - ``-55. * u.mV``, ``-70. * u.mV``
         - :math:`V_{th}`, :math:`V_{reset}`
         - Threshold and post-spike reset levels.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant injected current term.
       * - ``V_min``
         - ArrayLike broadcastable (mV) or ``None``
         - ``None``
         - :math:`V_{min}`
         - Optional lower clip on membrane potential.
       * - ``V_initializer``
         - Callable returning mV-compatible values
         - ``Constant(-70. * u.mV)``
         - --
         - Initializes membrane state ``V``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate spike output function.
       * - ``spk_reset``
         - str
         - ``'hard'``
         - --
         - Reset policy inherited from base neuron class.
       * - ``refractory_input``
         - bool
         - ``False``
         - --
         - Controls treatment of refractory-time delta events.
       * - ``ref_var``
         - bool
         - ``False``
         - --
         - Exposes persistent refractory state variable.
       * - ``name``
         - str | None
         - ``None``
         - --
         - Optional node identifier.

    Raises
    ------
    ValueError
        If validated construction/runtime constraints fail, including invalid
        parameter inequalities (for example ``V_reset >= V_th``), non-positive
        time constants/capacitance, ``dt <= 0``, invalid event offsets, or
        ``floor(t_ref / dt) < 1``.
    TypeError
        If provided arguments are incompatible with expected unit arithmetic
        (mV, pA, pF, ms) or callable interfaces.
    KeyError
        If required simulation context entries (``t`` and/or ``dt``) are
        missing when :meth:`update` is called.
    AttributeError
        If :meth:`update` is called before :meth:`init_state` creates required
        state variables.

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential state in mV, shape ``self.varshape`` (or with
        leading batch dimension when ``batch_size`` is specified).
    I_stim : brainstate.ShortTermState
        One-step buffered continuous current input in pA. Applied in the
        *next* update call (NEST ring-buffer semantics).
    last_spike_time : brainstate.ShortTermState
        Absolute precise spike time (ms) for the latest emitted spike.
        Initialized to ``-1e7 * u.ms`` (far past) to indicate no prior spike.
    last_spike_step : brainstate.ShortTermState
        Integer (``jnp.int32``) step index associated with the latest emitted
        spike. Initialized to ``-1``.
    last_spike_offset : brainstate.ShortTermState
        Precise within-step offset (ms) measured from the step right boundary
        (NEST convention: ``0`` at step end, ``dt`` at step start).
    is_refractory : brainstate.ShortTermState
        Boolean mask indicating which neurons are currently in the absolute
        refractory period.
    refractory_spike_buffer : brainstate.ShortTermState
        Deferred refractory-time delta contribution (mV). Non-zero only when
        ``refractory_input=True``; accumulates exponentially decayed delta
        events and is released at end of refractoriness.
    refractory : brainstate.ShortTermState
        Mirror of ``is_refractory`` exposed for external inspection. Present
        only when ``ref_var=True``.

    Notes
    -----
    - ``x`` passed to ``update(x=...)`` is buffered into ``I_stim`` and applied
      on the *next* step, mirroring NEST ring-buffer semantics for current events.
    - Delta inputs from ``add_delta_input`` are interpreted as on-grid events at
      step end (offset ``0``).
    - Additional within-step precise events can be supplied through
      ``update(spike_events=...)`` where each event is ``(offset, weight)``
      or ``{'offset': ..., 'weight': ...}`` in units of ms and mV.
    - This model uses ``floor(t_ref / dt)`` for refractory step conversion
      (matching NEST ``iaf_psc_delta_ps``), whereas ``iaf_psc_delta`` uses
      ``ceil(t_ref / dt)``.

    Examples
    --------
    Basic usage with constant current drive:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_delta_ps(in_size=2, t_ref=2.0 * u.ms)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         spk = neu.update(x=200.0 * u.pA)
       ...     _ = spk.shape

    Precise within-step spike events with ``refractory_input=True``:

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_delta_ps(in_size=1, refractory_input=True)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = neu.update(spike_events=[(0.04 * u.ms, 2.5 * u.mV)])

    References
    ----------
    .. [1] Rotter S, Diesmann M (1999). Exact simulation of time-invariant linear
           systems with applications to neuronal modeling. Biological Cybernetics
           81:381-402. DOI: https://doi.org/10.1007/s004220050570
    .. [2] Diesmann M, Gewaltig M-O, Rotter S, Aertsen A (2001). State space
           analysis of synchronous spiking in cortical neural networks.
           Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X
    .. [3] Morrison A, Straube S, Plesser HE, Diesmann M (2007).
           Exact subthreshold integration with continuous spike times in
           discrete-time neural network simulations. Neural Computation
           19(1):47-79. DOI: https://doi.org/10.1162/neco.2007.19.1.47
    .. [4] Hanuschkin A, Kunkel S, Helias M, Morrison A, Diesmann M (2010).
           A general and efficient method for incorporating exact spike times in
           globally time-driven simulations. Frontiers in Neuroinformatics 4:113.
           DOI: https://doi.org/10.3389/fninf.2010.00113

    See Also
    --------
    iaf_psc_delta : Current-based LIF with delta synapses (on-grid spike times)
    iaf_cond_exp : Conductance-based LIF with exponential synapses
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 250. * u.pF,
        tau_m: ArrayLike = 10. * u.ms,
        t_ref: ArrayLike = 2. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -70. * u.mV,
        I_e: ArrayLike = 0. * u.pA,
        V_min: Optional[ArrayLike] = None,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        refractory_input: bool = False,
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_min = None if V_min is None else braintools.init.param(V_min, self.varshape)

        self.V_initializer = V_initializer
        self.refractory_input = refractory_input
        self.ref_var = ref_var

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        v_reset = self._to_numpy(self.V_reset, u.mV)
        v_th = self._to_numpy(self.V_th, u.mV)
        c_m = self._to_numpy(self.C_m, u.pF)
        tau_m = self._to_numpy(self.tau_m, u.ms)

        if np.any(v_reset >= v_th):
            raise ValueError('Reset potential must be smaller than threshold.')
        if self.V_min is not None and np.any(v_reset < self._to_numpy(self.V_min, u.mV)):
            raise ValueError('Reset potential must be greater or equal to minimum potential.')
        if np.any(c_m <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(tau_m <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize membrane, timing, and refractory runtime states.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, states use
            ``self.varshape``; otherwise they use
            ``(batch_size,) + self.varshape``.
        **kwargs : Any
            Unused compatibility arguments.


        Raises
        ------
        ValueError
            If initializer outputs cannot be broadcast to target state shape.
        TypeError
            If initializer values are not unit-compatible with mV/pA/ms states.
        """
        V0 = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V0 / u.mV))

        self.V = brainstate.HiddenState(V0)
        self.I_stim = brainstate.ShortTermState(zeros * u.pA)
        self.last_spike_time = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        )
        last_step = braintools.init.param(braintools.init.Constant(-1), self.varshape, batch_size)
        ditype = brainstate.environ.ditype()
        self.last_spike_step = brainstate.ShortTermState(u.math.asarray(last_step, dtype=ditype))
        self.last_spike_offset = brainstate.ShortTermState(zeros * u.ms)
        is_refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
        self.is_refractory = brainstate.ShortTermState(is_refractory)
        self.refractory_spike_buffer = brainstate.ShortTermState(u.math.zeros_like(V0))

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(is_refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Evaluate surrogate spike activation for a voltage tensor.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Voltage values in mV, broadcast-compatible with ``self.varshape``
            (or current state shape when batched). If ``None``, uses
            ``self.V.value``.

        Returns
        -------
        out : dict
            Output of ``self.spk_fun`` evaluated on normalized threshold
            distance ``(V - V_th) / (V_th - V_reset)`` with same shape as ``V``.

        Raises
        ------
        TypeError
            If ``V`` cannot participate in unit-compatible arithmetic.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    @staticmethod
    def _canonicalize_spike_events(
        spike_events: Optional[Union[Sequence, dict, Tuple]],
    ) -> Sequence:
        r"""Normalize accepted spike-event container variants.

        Parameters
        ----------
        spike_events : Sequence or dict or tuple or None
            Event specification accepted by :meth:`update`:
            ``None``, a single ``{'offset', 'weight'}`` dict, one
            ``(offset, weight)`` tuple, or a sequence of these entries.

        Returns
        -------
        out : Sequence
            Sequence-like iterable of event records. Single dict/tuple inputs
            are wrapped into a one-element list; ``None`` returns ``[]``.
        """
        if spike_events is None:
            return []
        if isinstance(spike_events, dict):
            return [spike_events]
        if isinstance(spike_events, tuple) and len(spike_events) == 2:
            return [spike_events]
        return spike_events

    def _parse_spike_events(
        self,
        spike_events: Optional[Union[Sequence, dict, Tuple]],
        shape,
    ) -> Sequence[Tuple[float, np.ndarray]]:
        r"""Parse precise spike events into numeric offsets and broadcast weights.

        Parameters
        ----------
        spike_events : Sequence or dict or tuple or None
            Event specification in one of these forms:
            ``(offset, weight)``, ``{'offset': ..., 'weight': ...}``, or
            sequence of such entries. ``offset`` is interpreted in ms and
            ``weight`` in mV; plain numeric values are promoted to these units.
        shape : tuple of int
            Target state shape used to broadcast each event weight.

        Returns
        -------
        out : Sequence[Tuple[float, np.ndarray]]
            List of parsed events ``[(offset_ms, weight_np), ...]`` where
            ``offset_ms`` is ``float`` and ``weight_np`` is a ``float64``
            ``numpy.ndarray`` broadcast to ``shape`` (unit: mV).

        Raises
        ------
        ValueError
            If an event dictionary does not contain both ``offset`` and
            ``weight``, or if an event record has unsupported structure.
        TypeError
            If offsets/weights are not convertible to ms/mV-compatible arrays.
        """
        parsed = []
        for ev in self._canonicalize_spike_events(spike_events):
            if isinstance(ev, dict):
                if 'offset' not in ev or 'weight' not in ev:
                    raise ValueError('Each spike event dict must contain "offset" and "weight".')
                offset, weight = ev['offset'], ev['weight']
            else:
                if not isinstance(ev, Iterable):
                    raise ValueError(f'Unsupported spike event format: {ev}.')
                offset, weight = ev

            offset_ms = float(
                u.math.asarray((offset if not isinstance(offset, (int, float)) else offset * u.ms) / u.ms))
            weight_q = weight if not isinstance(weight, (int, float)) else weight * u.mV
            weight_np = self._broadcast_to_state(self._to_numpy(weight_q, u.mV), shape)
            parsed.append((offset_ms, weight_np))
        return parsed

    def update(self, x=0. * u.pA, spike_events: Optional[Union[Sequence, dict, Tuple]] = None):
        r"""Advance one simulation step with optional precise within-step events.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input in pA. This value is buffered into
            ``self.I_stim`` and applied in the *next* update call, matching
            NEST ring-buffer current semantics.
        spike_events : Sequence or dict or tuple or None, optional
            Optional precise delta events applied in the current step.
            Accepted formats are ``(offset, weight)``,
            ``{'offset': ..., 'weight': ...}``, or a sequence of such events.
            ``offset`` is in ms measured from the step right boundary with
            NEST convention (``0`` at step end, ``dt`` at step start).
            ``weight`` is a voltage jump in mV and may be scalar or
            broadcastable to neuron state shape.

        Returns
        -------
        out : jax.Array
            Surrogate spike output from :meth:`get_spike` with shape
            ``self.V.value.shape``. Elements corresponding to neurons that
            spiked in this step are forced slightly above threshold before
            surrogate evaluation to encode emitted spikes after hard reset.

        Raises
        ------
        ValueError
            If ``dt <= 0``, if ``floor(t_ref / dt) < 1``, if event offsets are
            outside ``[0, dt]``, or if event structures are invalid.
        KeyError
            If simulation context does not provide required ``t``/``dt``.
        AttributeError
            If state variables are unavailable because :meth:`init_state` was
            not called before :meth:`update`.
        TypeError
            If inputs or internal values are not unit-compatible with expected
            pA/mV/ms arithmetic.
        """
        t_q = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt_ms = float(u.math.asarray(dt_q / u.ms))
        t_ms = float(u.math.asarray(t_q / u.ms))

        if dt_ms <= 0.0:
            raise ValueError('Simulation time step must be positive.')

        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        U = V - E_L
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        t_ref = self._broadcast_to_state(self._to_numpy(self.t_ref, u.ms), v_shape)
        U_th = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        U_reset = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)
        dftype = brainstate.environ.dftype()
        U_min = -np.inf * np.ones(v_shape, dtype=dftype)
        if self.V_min is not None:
            U_min = self._broadcast_to_state(self._to_numpy(self.V_min - self.E_L, u.mV), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)

        I_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        ditype = brainstate.environ.ditype()
        last_step = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.last_spike_step.value), dtype=ditype),
            v_shape,
        )
        last_offset = self._broadcast_to_state(self._to_numpy(self.last_spike_offset.value, u.ms), v_shape)
        is_refractory = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.is_refractory.value), dtype=bool),
            v_shape,
        )
        refr_buffer = self._broadcast_to_state(self._to_numpy(self.refractory_spike_buffer.value, u.mV), v_shape)
        last_spike_time_prev = self._broadcast_to_state(self._to_numpy(self.last_spike_time.value, u.ms), v_shape)

        refr_steps = np.floor(t_ref / dt_ms).astype(np.int64)
        if np.any(refr_steps < 1):
            raise ValueError('Refractory time must be at least one time step.')

        on_grid_delta = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.mV), u.mV), v_shape)
        new_i_stim = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)
        parsed_events = self._parse_spike_events(spike_events, v_shape)
        parsed_events.append((0.0, on_grid_delta))
        parsed_events = sorted(parsed_events, key=lambda z: z[0], reverse=True)
        if any((ev_off < 0.0 or ev_off > dt_ms) for ev_off, _ in parsed_events):
            raise ValueError('All spike event offsets must satisfy 0 <= offset <= dt.')

        step_idx = int(round(t_ms / dt_ms))
        eps = np.finfo(np.float64).eps

        V_next = np.empty_like(V)
        last_step_next = np.empty_like(last_step)
        last_offset_next = np.empty_like(last_offset)
        is_refractory_next = np.empty_like(is_refractory)
        refr_buffer_next = np.empty_like(refr_buffer)
        last_spike_time_next = np.empty_like(last_spike_time_prev)
        spike_mask = np.zeros_like(V, dtype=bool)
        V_for_spike = np.empty_like(V)

        for idx in np.ndindex(v_shape):
            u_i = U[idx]
            i_i = I_stim[idx]
            tau_i = tau_m[idx]
            t_ref_i = t_ref[idx]
            c_m_i = C_m[idx]
            i_e_i = I_e[idx]
            u_th_i = U_th[idx]
            u_reset_i = U_reset[idx]
            u_min_i = U_min[idx]
            refr_steps_i = int(refr_steps[idx])

            last_step_i = int(last_step[idx])
            last_offset_i = float(last_offset[idx])
            is_refr_i = bool(is_refractory[idx])
            refr_buf_i = float(refr_buffer[idx])
            spike_time_i = float(last_spike_time_prev[idx])

            r_mem = tau_i / c_m_i
            did_spike = False

            def _propagate(delta_t_ms: float):
                nonlocal u_i
                if delta_t_ms <= 0.0:
                    return
                expm1_dt = math.expm1(-delta_t_ms / tau_i)
                v_inf = r_mem * (i_i + i_e_i)
                # Numerically stable arrangement used in NEST.
                u_i = -v_inf * expm1_dt + u_i * expm1_dt + u_i

            def _emit_spike(offset_u_ms: float):
                nonlocal did_spike, last_step_i, last_offset_i, is_refr_i, u_i, spike_time_i
                v_inf = r_mem * (i_i + i_e_i)
                ratio = (v_inf - u_i) / (v_inf - u_th_i)
                ratio = min(1.0, max(np.finfo(np.float64).tiny, ratio))
                dt_cross = -tau_i * math.log(ratio)
                spike_off = offset_u_ms + dt_cross
                spike_off = min(dt_ms, max(0.0, spike_off))

                did_spike = True
                last_step_i = step_idx + 1
                last_offset_i = spike_off
                is_refr_i = True
                u_i = u_reset_i
                spike_time_i = t_ms + dt_ms - spike_off

            def _emit_instant_spike(spike_off_ms: float):
                nonlocal did_spike, last_step_i, last_offset_i, is_refr_i, u_i, spike_time_i
                spike_off = min(dt_ms, max(0.0, spike_off_ms))

                did_spike = True
                last_step_i = step_idx + 1
                last_offset_i = spike_off
                is_refr_i = True
                u_i = u_reset_i
                spike_time_i = t_ms + dt_ms - spike_off

            # Super-threshold at step start: spike at t + epsilon.
            if (not is_refr_i) and (u_i >= u_th_i):
                _emit_instant_spike(dt_ms * (1.0 - eps))

            local_events = [(off, w[idx], False) for off, w in parsed_events]
            if is_refr_i and (step_idx + 1 - last_step_i == refr_steps_i):
                local_events.append((last_offset_i, 0.0, True))
            local_events.sort(key=lambda z: z[0], reverse=True)

            if len(local_events) == 0:
                if not is_refr_i:
                    _propagate(dt_ms)
                    if u_i < u_min_i:
                        u_i = u_min_i
                    if u_i >= u_th_i:
                        _emit_spike(0.0)
            else:
                t_cursor = dt_ms
                for ev_offset, ev_weight, end_of_refract in local_events:
                    if is_refr_i:
                        t_cursor = ev_offset
                        if not end_of_refract:
                            if self.refractory_input:
                                expo = -(
                                    ((last_step_i - step_idx - 1) * dt_ms)
                                    - (last_offset_i - ev_offset)
                                    + t_ref_i
                                ) / tau_i
                                refr_buf_i += ev_weight * math.exp(expo)
                        else:
                            is_refr_i = False
                            if self.refractory_input:
                                u_i += refr_buf_i
                                refr_buf_i = 0.0
                            if u_i >= u_th_i:
                                _emit_instant_spike(t_cursor)
                        continue

                    _propagate(t_cursor - ev_offset)
                    t_cursor = ev_offset

                    if u_i >= u_th_i:
                        _emit_spike(t_cursor)
                        continue

                    u_i += ev_weight
                    if u_i >= u_th_i:
                        _emit_instant_spike(t_cursor)

                if (not is_refr_i) and (t_cursor > 0.0):
                    _propagate(t_cursor)
                    if u_i >= u_th_i:
                        _emit_spike(0.0)

            v_i = u_i + E_L[idx]
            V_next[idx] = v_i
            last_step_next[idx] = last_step_i
            last_offset_next[idx] = last_offset_i
            is_refractory_next[idx] = is_refr_i
            refr_buffer_next[idx] = refr_buf_i
            last_spike_time_next[idx] = spike_time_i

            spike_mask[idx] = did_spike
            V_for_spike[idx] = (E_L[idx] + u_th_i + 1e-12) if did_spike else v_i

        self.V.value = V_next * u.mV
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_step.value = jnp.asarray(last_step_next, dtype=ditype)
        self.last_spike_offset.value = last_offset_next * u.ms
        self.is_refractory.value = jnp.asarray(is_refractory_next, dtype=bool)
        self.refractory_spike_buffer.value = refr_buffer_next * u.mV
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time_next * u.ms)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.is_refractory.value)

        return self.get_spike(V_for_spike * u.mV)
