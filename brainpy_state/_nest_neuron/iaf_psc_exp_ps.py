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

from typing import Callable, Iterable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, propagator_exp, cond_any

__all__ = [
    'iaf_psc_exp_ps',
]


class iaf_psc_exp_ps(NESTNeuron):
    r"""NEST-compatible ``iaf_psc_exp_ps`` with precise spike times.

    Description
    -----------

    ``iaf_psc_exp_ps`` is a current-based leaky integrate-and-fire neuron with
    exponential excitatory/inhibitory PSC states and off-grid event/spike
    timing. The implementation follows NEST
    ``models/iaf_psc_exp_ps.{h,cpp}`` semantics: within-step event ordering by
    precise offsets, exact closed-form mini-step propagation, sub-step
    threshold localization by root search, and refractory release modeled as an
    explicit pseudo-event.

    **1. Continuous-time dynamics and exact integration**

    Let :math:`U = V_m - E_L`, :math:`I_{ex}` and :math:`I_{in}` be
    excitatory/inhibitory PSC states (pA), and :math:`y_0` the one-step
    buffered continuous input current (pA). Subthreshold dynamics are

    .. math::

       \frac{dU}{dt} = -\frac{U}{\tau_m}
       + \frac{I_e + y_0 + I_{ex} + I_{in}}{C_m},

    .. math::

       \frac{dI_{ex}}{dt} = -\frac{I_{ex}}{\tau_{syn,ex}}, \qquad
       \frac{dI_{in}}{dt} = -\frac{I_{in}}{\tau_{syn,in}}.

    Over a mini-interval :math:`\Delta t`, exact integration gives

    .. math::

       U(t+\Delta t) = P_{20}(\Delta t)\,(I_e+y_0)
       + P_{21,ex}(\Delta t)\,I_{ex}(t)
       + P_{21,in}(\Delta t)\,I_{in}(t)
       + U(t)e^{-\Delta t/\tau_m},

    where
    :math:`P_{20}=-\frac{\tau_m}{C_m}\left(e^{-\Delta t/\tau_m}-1\right)` and
    :math:`P_{21,X}` are evaluated by
    :func:`propagator_exp` (from ``_utils``). PSC states decay exactly via
    :math:`I_X(t+\Delta t)=I_X(t)e^{-\Delta t/\tau_{syn,X}}`.

    **2. Precise-time event processing**

    Event offsets use NEST convention: ``offset=dt`` at step start and
    ``offset=0`` at step end. For each global step:

    1. Build local event list from ``spike_events`` and on-grid delta input
       (always added at ``offset=0``).
    2. Sort events in descending offset and split the step into mini-intervals.
    3. Propagate exactly on each mini-interval.
    4. If :math:`U` reaches threshold, solve
       :math:`f(\delta)=U(\delta)-U_{th}=0` with bounded bisection
       (64 iterations) to obtain off-grid spike time.
    5. Reset to ``V_reset`` and enter refractory state; release from refractory
       occurs through a pseudo-event when
       ``step_idx + 1 - last_spike_step == ceil(t_ref / dt)``.

    **3. Assumptions, constraints, and computational complexity**

    - Parameters are scalar or broadcastable to ``self.varshape``.
    - Construction-time constraints enforce
      ``V_reset < V_th``, ``C_m > 0``, ``tau_m > 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, and when ``V_min`` is provided:
      ``V_reset >= V_min``.
    - Runtime requires ``ceil(t_ref / dt) >= 1``.
    - All precise offsets must satisfy ``0 <= offset <= dt``.
    - Continuous input ``x`` is buffered (stored into ``y0`` for the next
      global step), matching NEST current-event timing.
    - Per-step complexity is
      :math:`O(|\mathrm{state}| \cdot K)` for ``K`` local events, plus root
      search cost on threshold-crossing mini-intervals.

    Parameters
    ----------
    in_size : Size
        Population shape specification. Model parameters and states are
        broadcast to ``self.varshape`` derived from ``in_size``.
    E_L : ArrayLike, optional
        Resting potential :math:`E_L` in mV, broadcastable to ``self.varshape``.
        Default is ``-70. * u.mV``.
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
        ``self.varshape``. Converted at runtime to steps using
        ``ceil(t_ref / dt)`` and must produce at least one step.
        Default is ``2. * u.ms``.
    V_th : ArrayLike, optional
        Threshold voltage :math:`V_{th}` in mV, broadcastable to
        ``self.varshape``. Default is ``-55. * u.mV``.
    V_reset : ArrayLike, optional
        Reset voltage :math:`V_{reset}` in mV, broadcastable to
        ``self.varshape``. Must satisfy ``V_reset < V_th`` elementwise.
        Default is ``-70. * u.mV``.
    tau_syn_ex : ArrayLike, optional
        Excitatory PSC decay constant :math:`\tau_{syn,ex}` in ms,
        broadcastable to ``self.varshape`` and strictly positive.
        Default is ``2. * u.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory PSC decay constant :math:`\tau_{syn,in}` in ms,
        broadcastable to ``self.varshape`` and strictly positive.
        Default is ``2. * u.ms``.
    I_e : ArrayLike, optional
        Constant external current :math:`I_e` in pA, broadcastable to
        ``self.varshape``. Added in each mini-step propagation.
        Default is ``0. * u.pA``.
    V_min : ArrayLike or None, optional
        Optional lower bound :math:`V_{min}` in mV, broadcastable to
        ``self.varshape``. If ``None``, no lower clip is applied.
        Default is ``None``.
    V_initializer : Callable, optional
        Initializer used by :meth:`init_state` for membrane state ``V``.
        Must return mV-compatible values with shape compatible with
        ``self.varshape`` (and optional batch prefix). Default is
        ``braintools.init.Constant(-70. * u.mV)``.
    spk_fun : Callable, optional
        Surrogate spike function used by :meth:`get_spike` and
        :meth:`update`. Receives normalized threshold distance tensor.
        Default is ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy forwarded to :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` matches NEST hard reset. Default is ``'hard'``.
    ref_var : bool, optional
        If ``True``, creates exposed ``self.refractory`` mirroring
        ``self.is_refractory`` for inspection. Default is ``False``.
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
         - Defines ``self.varshape`` for parameter/state broadcasting.
       * - ``E_L``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``-70. * u.mV``
         - :math:`E_L`
         - Resting potential and voltage-offset origin.
       * - ``C_m``
         - ArrayLike, broadcastable (pF), ``> 0``
         - ``250. * u.pF``
         - :math:`C_m`
         - Converts current terms to membrane-rate contribution.
       * - ``tau_m``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``10. * u.ms``
         - :math:`\tau_m`
         - Leak time constant in exact subthreshold propagation.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), runtime ``ceil(t_ref/dt) >= 1``
         - ``2. * u.ms``
         - :math:`t_{ref}`
         - Absolute refractory duration.
       * - ``V_th`` and ``V_reset``
         - ArrayLike, broadcastable (mV), with ``V_reset < V_th``
         - ``-55. * u.mV``, ``-70. * u.mV``
         - :math:`V_{th}`, :math:`V_{reset}`
         - Threshold and post-spike reset levels.
       * - ``tau_syn_ex`` and ``tau_syn_in``
         - ArrayLike, broadcastable (ms), each ``> 0``
         - ``2. * u.ms``
         - :math:`\tau_{syn,ex}`, :math:`\tau_{syn,in}`
         - Exponential PSC decay constants.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant injected current added every mini-step.
       * - ``V_min``
         - ArrayLike broadcastable (mV) or ``None``
         - ``None``
         - :math:`V_{min}`
         - Optional lower clamp applied after membrane propagation.
       * - ``V_initializer``
         - Callable returning mV-compatible values
         - ``Constant(-70. * u.mV)``
         - --
         - Initializes membrane state ``V``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate spike output nonlinearity.
       * - ``spk_reset``
         - str
         - ``'hard'``
         - --
         - Reset mode inherited from base ``Neuron``.
       * - ``ref_var``
         - bool
         - ``False``
         - --
         - Allocate exposed ``refractory`` mirror state.
       * - ``name``
         - str | None
         - ``None``
         - --
         - Optional node name.

    Raises
    ------
    ValueError
        If validated constraints fail (for example ``V_reset >= V_th``,
        non-positive capacitance/time constants, ``V_reset < V_min``,
        ``ceil(t_ref / dt) < 1``, or event offsets outside ``[0, dt]``).
    TypeError
        If provided arguments are incompatible with expected units/callables
        (mV, pA, pF, ms).
    KeyError
        If simulation context values ``t`` and/or ``dt`` are missing when
        :meth:`update` is called.
    AttributeError
        If :meth:`update` is called before :meth:`init_state` creates required
        runtime states.

    Attributes
    ----------
    V : HiddenState
        Membrane potential state in mV.
    I_syn_ex : ShortTermState
        Excitatory PSC state in pA.
    I_syn_in : ShortTermState
        Inhibitory PSC state in pA.
    y0 : ShortTermState
        One-step buffered continuous current in pA.
    is_refractory : ShortTermState
        Boolean refractory mask.
    last_spike_step : ShortTermState
        Step index of latest emitted spike.
    last_spike_offset : ShortTermState
        Precise offset (ms) from right step boundary for latest spike.
    last_spike_time : ShortTermState
        Absolute precise spike time in ms.
    refractory : ShortTermState
        Optional mirror of ``is_refractory`` when ``ref_var=True``.

    Notes
    -----
    - ``spike_events`` accepts ``(offset, weight)`` tuples or
      ``{'offset': ..., 'weight': ...}`` dicts.
    - Offsets are in ms and measured from the right edge of the current step.
    - Positive event weights contribute to excitatory PSC state; negative
      weights contribute to inhibitory PSC state.
    - Internal propagation and root finding are evaluated in NumPy float64 and
      written back into BrainUnit states at end of step.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_exp_ps(in_size=2, I_e=200.0 * u.pA)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         spk = neu.update()
       ...     _ = spk.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_exp_ps(in_size=1)
       ...     neu.init_state()
       ...     ev = [{'offset': 0.08 * u.ms, 'weight': 120.0 * u.pA}]
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = neu.update(spike_events=ev)

    References
    ----------
    .. [1] NEST source: ``models/iaf_psc_exp_ps.h`` and
           ``models/iaf_psc_exp_ps.cpp``.
    .. [2] Rotter S, Diesmann M (1999). Exact simulation of time-invariant
           linear systems with applications to neuronal modeling.
           Biological Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    .. [3] Morrison A, Straube S, Plesser HE, Diesmann M (2007). Exact
           subthreshold integration with continuous spike times in discrete
           time neural network simulations. Neural Computation 19(1):47-79.
           DOI: https://doi.org/10.1162/neco.2007.19.1.47
    .. [4] Hanuschkin A, Kunkel S, Helias M, Morrison A, Diesmann M (2010).
           A general and efficient method for incorporating exact spike times
           in globally time-driven simulations. Frontiers in Neuroinformatics.
           DOI: https://doi.org/10.3389/fninf.2010.00113
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
        tau_syn_ex: ArrayLike = 2. * u.ms,
        tau_syn_in: ArrayLike = 2. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        V_min: ArrayLike = None,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
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
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_min = None if V_min is None else braintools.init.param(V_min, self.varshape)
        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

        # Precompute refractory step count.
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    def _validate_parameters(self):
        r"""Validate model parameters against constraints.

        Raises
        ------
        ValueError
            If parameter inequalities or positivity constraints are violated.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_reset, self.C_m, self.tau_m)):
            return

        if cond_any(self.V_reset >= self.V_th):
            raise ValueError('Reset potential must be smaller than threshold.')
        if self.V_min is not None and cond_any(self.V_reset < self.V_min):
            raise ValueError('Reset potential must be greater equal minimum potential.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.tau_m <= 0.0 * u.ms):
            raise ValueError('Membrane time constant must be strictly positive.')
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms) or cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize membrane, synaptic, and precise-timing runtime states.

        This method allocates all internal state variables required for precise
        spike-time simulation. Membrane potential ``V`` is initialized using
        ``self.V_initializer``, synaptic currents and buffered inputs are
        initialized to zero, and spike-tracking states are initialized to
        sentinel values (``last_spike_step = -1``, ``last_spike_time = -1e7 ms``)
        indicating no prior spike events.

        Parameters
        ----------
        **kwargs : Any
            Unused compatibility arguments for subclass extension.

        Raises
        ------
        ValueError
            If initializer outputs cannot be broadcast to state shape
            ``self.varshape`` or if shapes are incompatible.
        TypeError
            If initializer outputs are not unit-compatible with expected state
            units (mV for voltage, pA for currents, ms for time, bool for flags).
        AttributeError
            If ``self.V_initializer`` is not callable or does not produce valid
            output for the requested shape.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()

        V = braintools.init.param(self.V_initializer, self.varshape)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.y0 = brainstate.ShortTermState(zeros * u.pA)
        self.is_refractory = brainstate.ShortTermState(np.zeros(self.varshape, dtype=bool))
        self.last_spike_step = brainstate.ShortTermState(
            u.math.full(self.varshape, -1, dtype=ditype)
        )
        self.last_spike_offset = brainstate.ShortTermState(zeros * u.ms)
        self.last_spike_time = brainstate.ShortTermState(
            u.math.full(self.varshape, -1e7 * u.ms)
        )

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(np.zeros(self.varshape, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        r"""Evaluate surrogate spike output from membrane potential.

        Applies the surrogate spike function (typically
        ``braintools.surrogate.ReluGrad`` or similar) to a normalized
        threshold-distance metric. This enables differentiable spike generation
        for gradient-based learning while maintaining biological spike semantics.

        The normalized threshold distance is computed as
        :math:`(V - V_{th}) / (V_{th} - V_{reset})`, which maps the voltage
        range between reset and threshold to ``[0, 1]``, with values above
        threshold producing positive outputs through the surrogate function.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Voltage tensor in mV, broadcast-compatible with ``self.varshape``
            (or current batched state shape). If ``None``, uses
            ``self.V.value``. Default is ``None``.

        Returns
        -------
        out : dict
            Output of ``self.spk_fun`` applied to normalized threshold distance
            ``(V - V_th) / (V_th - V_reset)`` with same shape as input ``V``.
            Typically float values in ``[0, 1]`` or similar range depending on
            the surrogate function's output characteristics.

        Raises
        ------
        TypeError
            If ``V`` is not compatible with unit arithmetic in mV or if unit
            conversion operations fail.
        AttributeError
            If ``self.spk_fun`` is not callable or if required parameters
            (``V_th``, ``V_reset``) are not available.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        r"""Parse spike events into normalized (offset_ms, weight_array) tuples.

        Converts mixed-format spike events (tuples or dicts) into a uniform
        internal representation suitable for event processing. Offsets are
        extracted in ms units, weights are extracted in pA units and broadcast
        to match the neuron population shape.

        Parameters
        ----------
        spike_events : Iterable or None
            User-provided spike events as ``(offset, weight)`` tuples or
            ``{'offset': ..., 'weight': ...}`` dicts.
        v_shape : tuple
            Target shape for broadcasting weight arrays (typically
            ``self.V.value.shape``).

        Returns
        -------
        list of tuple[float, np.ndarray]
            List of ``(offset_ms, weight_np)`` pairs where ``offset_ms`` is a
            float scalar in ms and ``weight_np`` is a float64 array broadcast
            to ``v_shape``.
        """
        events = []
        if spike_events is None:
            return events
        dftype = brainstate.environ.dftype()
        for ev in spike_events:
            if isinstance(ev, dict):
                offs = ev.get('offset', 0.0 * u.ms)
                w = ev.get('weight', 0.0 * u.pA)
            else:
                offs, w = ev
            off_ms = float(u.math.asarray(offs / u.ms))
            w_np = np.asarray(u.math.asarray(w / u.pA), dtype=dftype)
            events.append((off_ms, np.broadcast_to(w_np, v_shape)))
        return events

    @staticmethod
    def _bisect_root(f, t_hi: float):
        r"""Find root of scalar function using bounded bisection.

        Locates the point where ``f(t)`` crosses zero within the interval
        ``[0, t_hi]`` using bisection with 64 iterations. Assumes ``f`` is
        continuous and monotonically increasing within the search interval.

        This method is used to find the precise sub-step time at which the
        membrane potential crosses the spike threshold during exact integration.

        Parameters
        ----------
        f : Callable[[float], float]
            Scalar function representing threshold distance as a function of
            time offset within a mini-interval. Expected to be negative at
            ``t=0`` and positive or zero at ``t=t_hi`` for a valid crossing.
        t_hi : float
            Upper bound of search interval in ms (mini-interval duration).

        Returns
        -------
        float
            Estimated root location in ``[0, t_hi]`` in ms. If no crossing is
            detected (``f(0) > 0`` or ``f(t_hi) <= 0``), returns boundary
            values ``0.0`` or ``t_hi`` respectively. If ``f(t_hi)`` is
            non-finite, returns ``t_hi``.

        Notes
        -----
        The bisection uses 64 iterations, providing approximately
        :math:`2^{-64}` relative precision on the root location within the
        search interval. This is sufficient for neuroscience simulation
        time scales (ms resolution).
        """
        lo = 0.0
        hi = float(t_hi)
        f_lo = f(lo)
        f_hi = f(hi)
        if not np.isfinite(f_hi):
            return hi
        if f_lo > 0.0:
            return 0.0
        if f_hi <= 0.0:
            return hi
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if f_mid > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def update(self, x=0. * u.pA, spike_events=None):
        r"""Advance one global step with precise within-step event handling.

        This method implements the complete NEST-compatible precise-spike-time
        algorithm for ``iaf_psc_exp_ps``. Each global time step is subdivided
        into mini-intervals determined by spike event offsets. Within each
        mini-interval, membrane potential and synaptic currents are propagated
        exactly using closed-form exponential solutions. When the membrane
        potential crosses threshold, bisection root-finding (64 iterations)
        localizes the precise sub-step spike time.

        **Update sequence:**

        1. Parse and validate ``spike_events`` and on-grid delta inputs.
        2. Sort events in descending offset (from step start to step end).
        3. For each neuron, process events sequentially:

           a. Propagate states exactly over each mini-interval.
           b. Apply event weights to PSC states (ex/in channels by sign).
           c. Check for threshold crossing and localize spike time if needed.
           d. Apply hard reset and enter refractory state on spike.
           e. Release from refractory via pseudo-event at calculated step.

        4. Buffer incoming current ``x`` into ``y0`` for next step.
        5. Compute surrogate spike output for gradient-based learning.

        **Implementation notes:**

        - All propagation uses NumPy float64 for numerical stability.
        - Event offsets follow NEST convention: ``offset=dt`` at step start,
          ``offset=0`` at step end.
        - Refractory neurons clamp membrane potential but allow PSC decay.
        - Root finding uses bounded bisection over ``[0, dt]`` with 64 iterations.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous current input in pA for the current global step.
            Aggregated through :meth:`sum_current_inputs` and stored in ``y0``
            for use in the next step (one-step buffering). Scalar or array-like
            broadcastable to ``self.V.value.shape``. Default is ``0. * u.pA``.
        spike_events : Iterable[tuple[Any, Any] | dict[str, Any]] or None, optional
            Optional off-grid events inside the current step. Each entry is
            ``(offset, weight)`` or ``{'offset': ..., 'weight': ...}``, where
            ``offset`` is in ms measured from the right step boundary and
            ``weight`` is in pA. Offsets must satisfy ``0 <= offset <= dt``.
            Positive weights update excitatory PSC; negative weights update
            inhibitory PSC. ``None`` means no extra precise events. On-grid
            delta inputs are automatically included at ``offset=0``.
            Default is ``None``.

        Returns
        -------
        out : jax.Array
            Surrogate spike output from :meth:`get_spike`, shape
            ``self.V.value.shape``. Values correspond to
            ``self.spk_fun((V - V_th) / (V_th - V_reset))`` after exact
            piecewise propagation, event application, refractory logic, and
            precise spike-time localization. For neurons that spiked, the
            voltage is clamped slightly above threshold to ensure differentiable
            spike detection; for non-spiking neurons, voltage is clamped below
            threshold.

        Raises
        ------
        ValueError
            If ``ceil(t_ref / dt) < 1`` (refractory period too short for time
            step), or if any event offset lies outside ``[0, dt]``, or if
            parameter constraints are violated at runtime.
        KeyError
            If simulation context values ``t`` (current time) or ``dt`` (time
            step) are unavailable from ``brainstate.environ``.
        TypeError
            If ``x`` or ``spike_events`` entries are not unit-compatible with
            pA/ms conversions, or if type conversions fail during numerical
            computation.
        AttributeError
            If required runtime states (``V``, ``I_syn_ex``, ``I_syn_in``,
            ``y0``, ``is_refractory``, etc.) are missing because
            :meth:`init_state` has not been called.
        """
        import math

        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))
        t_ms = float(u.math.asarray(t / u.ms))
        step_idx = int(round(t_ms / h))
        eps = np.finfo(np.float64).eps
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        v_shape = self.V.value.shape

        E_L = np.broadcast_to(np.asarray(u.math.asarray(self.E_L / u.mV), dtype=dftype), v_shape)
        y2 = np.broadcast_to(np.asarray(u.math.asarray(self.V.value / u.mV), dtype=dftype), v_shape) - E_L
        y1_ex = np.broadcast_to(np.asarray(u.math.asarray(self.I_syn_ex.value / u.pA), dtype=dftype), v_shape)
        y1_in = np.broadcast_to(np.asarray(u.math.asarray(self.I_syn_in.value / u.pA), dtype=dftype), v_shape)
        y0 = np.broadcast_to(np.asarray(u.math.asarray(self.y0.value / u.pA), dtype=dftype), v_shape)

        is_refractory = np.broadcast_to(
            np.asarray(u.math.asarray(self.is_refractory.value), dtype=bool), v_shape
        )
        last_spike_step = np.broadcast_to(
            np.asarray(u.math.asarray(self.last_spike_step.value), dtype=ditype), v_shape
        )
        last_spike_offset = np.broadcast_to(
            np.asarray(u.math.asarray(self.last_spike_offset.value / u.ms), dtype=dftype), v_shape
        )
        last_spike_time_prev = np.broadcast_to(
            np.asarray(u.math.asarray(self.last_spike_time.value / u.ms), dtype=dftype), v_shape
        )

        tau_m = np.broadcast_to(np.asarray(u.math.asarray(self.tau_m / u.ms), dtype=dftype), v_shape)
        tau_ex = np.broadcast_to(np.asarray(u.math.asarray(self.tau_syn_ex / u.ms), dtype=dftype), v_shape)
        tau_in = np.broadcast_to(np.asarray(u.math.asarray(self.tau_syn_in / u.ms), dtype=dftype), v_shape)
        c_m = np.broadcast_to(np.asarray(u.math.asarray(self.C_m / u.pF), dtype=dftype), v_shape)
        i_e = np.broadcast_to(np.asarray(u.math.asarray(self.I_e / u.pA), dtype=dftype), v_shape)
        u_th = np.broadcast_to(
            np.asarray(u.math.asarray((self.V_th - self.E_L) / u.mV), dtype=dftype), v_shape
        )
        u_reset = np.broadcast_to(
            np.asarray(u.math.asarray((self.V_reset - self.E_L) / u.mV), dtype=dftype), v_shape
        )
        u_min = -np.inf * np.ones(v_shape, dtype=dftype)
        if self.V_min is not None:
            u_min = np.broadcast_to(
                np.asarray(u.math.asarray((self.V_min - self.E_L) / u.mV), dtype=dftype), v_shape
            )

        refr_steps = np.broadcast_to(
            np.asarray(u.math.asarray(self.ref_count), dtype=ditype), v_shape
        )
        if cond_any(refr_steps < 1):
            raise ValueError('Refractory time must be at least one time step.')

        # Events in a step, sorted from step start (offset=dt) to step end (offset=0).
        events = self._parse_spike_events(spike_events, v_shape)
        on_grid = np.broadcast_to(
            np.asarray(u.math.asarray(self.sum_delta_inputs(0. * u.pA) / u.pA), dtype=dftype), v_shape
        )
        events.append((0.0, on_grid))
        events.sort(key=lambda z: z[0], reverse=True)
        for off, _ in events:
            if off < 0.0 or off > h:
                raise ValueError('All spike event offsets must satisfy 0 <= offset <= dt.')

        y0_next = np.broadcast_to(
            np.asarray(u.math.asarray(self.sum_current_inputs(x, self.V.value) / u.pA), dtype=dftype), v_shape
        )

        y0_new = np.empty_like(y0)
        y1_ex_new = np.empty_like(y1_ex)
        y1_in_new = np.empty_like(y1_in)
        y2_new = np.empty_like(y2)
        refr_new = np.empty_like(is_refractory)
        last_step_new = np.empty_like(last_spike_step)
        last_offset_new = np.empty_like(last_spike_offset)
        last_time_new = np.empty_like(last_spike_time_prev)
        spike_mask = np.zeros(v_shape, dtype=bool)
        v_for_spike = np.empty_like(y2)

        for idx in np.ndindex(v_shape):
            y0_i = float(y0[idx])
            y1e_i = float(y1_ex[idx])
            y1i_i = float(y1_in[idx])
            y2_i = float(y2[idx])
            refr_i = bool(is_refractory[idx])
            last_step_i = int(last_spike_step[idx])
            last_off_i = float(last_spike_offset[idx])
            spike_time_i = float(last_spike_time_prev[idx])

            tau_m_i = float(tau_m[idx])
            tau_ex_i = float(tau_ex[idx])
            tau_in_i = float(tau_in[idx])
            c_m_i = float(c_m[idx])
            i_e_i = float(i_e[idx])
            u_th_i = float(u_th[idx])
            u_reset_i = float(u_reset[idx])
            u_min_i = float(u_min[idx])
            refr_steps_i = int(refr_steps[idx])

            did_spike = False
            before = [y0_i, y1e_i, y1i_i, y2_i]

            def set_before():
                before[0] = y0_i
                before[1] = y1e_i
                before[2] = y1i_i
                before[3] = y2_i

            def threshold_distance(dt_local):
                P20 = -tau_m_i / c_m_i * math.expm1(-dt_local / tau_m_i)
                P21e = propagator_exp(np.asarray(tau_ex_i), np.asarray(tau_m_i), np.asarray(c_m_i),
                                                   dt_local)
                P21i = propagator_exp(np.asarray(tau_in_i), np.asarray(tau_m_i), np.asarray(c_m_i),
                                                   dt_local)
                y2_r = P20 * (i_e_i + before[0]) + P21e * before[1] + P21i * before[2] + before[3] * math.exp(
                    -dt_local / tau_m_i)
                return y2_r - u_th_i

            def propagate(dt_local):
                nonlocal y1e_i, y1i_i, y2_i
                if dt_local <= 0.0:
                    return
                if not refr_i:
                    P20 = -tau_m_i / c_m_i * math.expm1(-dt_local / tau_m_i)
                    P21e = propagator_exp(np.asarray(tau_ex_i), np.asarray(tau_m_i), np.asarray(c_m_i),
                                                       dt_local)
                    P21i = propagator_exp(np.asarray(tau_in_i), np.asarray(tau_m_i), np.asarray(c_m_i),
                                                       dt_local)
                    y2_i = P20 * (i_e_i + y0_i) + P21e * y1e_i + P21i * y1i_i + y2_i * math.exp(-dt_local / tau_m_i)
                    y2_i = max(y2_i, u_min_i)
                y1e_i = y1e_i * math.exp(-dt_local / tau_ex_i)
                y1i_i = y1i_i * math.exp(-dt_local / tau_in_i)

            def emit_spike(t0, dt_local):
                nonlocal y2_i, refr_i, last_step_i, last_off_i, spike_time_i, did_spike
                root = self._bisect_root(threshold_distance, dt_local)
                spike_off = h - (t0 + root)
                spike_off = min(h, max(0.0, spike_off))
                last_step_i = step_idx + 1
                last_off_i = spike_off
                y2_i = u_reset_i
                refr_i = True
                spike_time_i = t_ms + h - spike_off
                did_spike = True

            def emit_instant_spike(spike_off):
                nonlocal y2_i, refr_i, last_step_i, last_off_i, spike_time_i, did_spike
                so = min(h, max(0.0, spike_off))
                last_step_i = step_idx + 1
                last_off_i = so
                y2_i = u_reset_i
                refr_i = True
                spike_time_i = t_ms + h - so
                did_spike = True

            if (not refr_i) and (y2_i >= u_th_i):
                emit_instant_spike(h * (1.0 - eps))

            local_events = [(off, w[idx], False) for off, w in events]
            if refr_i and (step_idx + 1 - last_step_i == refr_steps_i):
                local_events.append((last_off_i, 0.0, True))
            local_events.sort(key=lambda z: z[0], reverse=True)

            last_off = h
            if len(local_events) == 0:
                propagate(h)
                if y2_i >= u_th_i:
                    set_before()
                    emit_spike(0.0, h)
            else:
                for ev_off, ev_w, end_of_refract in local_events:
                    ministep = last_off - ev_off
                    if ministep > 0.0:
                        set_before()
                        propagate(ministep)
                        if y2_i >= u_th_i:
                            emit_spike(h - last_off, ministep)
                    if end_of_refract:
                        refr_i = False
                    else:
                        if ev_w >= 0.0:
                            y1e_i += ev_w
                        else:
                            y1i_i += ev_w
                    set_before()
                    last_off = ev_off
                if last_off > 0.0:
                    set_before()
                    propagate(last_off)
                    if y2_i >= u_th_i:
                        emit_spike(h - last_off, last_off)

            y0_i = float(y0_next[idx])
            y0_new[idx] = y0_i
            y1_ex_new[idx] = y1e_i
            y1_in_new[idx] = y1i_i
            y2_new[idx] = y2_i
            refr_new[idx] = refr_i
            last_step_new[idx] = last_step_i
            last_offset_new[idx] = last_off_i
            last_time_new[idx] = spike_time_i
            spike_mask[idx] = did_spike
            v_for_spike[idx] = (u_th_i + 1e-12) if did_spike else min(y2_i, u_th_i - 1e-12)

        self.y0.value = y0_new * u.pA
        self.I_syn_ex.value = y1_ex_new * u.pA
        self.I_syn_in.value = y1_in_new * u.pA
        self.V.value = (y2_new + E_L) * u.mV
        self.is_refractory.value = jnp.asarray(refr_new, dtype=bool)
        self.last_spike_step.value = jnp.asarray(last_step_new, dtype=ditype)
        self.last_spike_offset.value = last_offset_new * u.ms
        self.last_spike_time.value = jax.lax.stop_gradient(last_time_new * u.ms)
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.is_refractory.value)

        return self.get_spike((v_for_spike + E_L) * u.mV)
