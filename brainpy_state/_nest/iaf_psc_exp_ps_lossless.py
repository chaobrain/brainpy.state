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
from typing import Callable, Iterable

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron
from .iaf_psc_exp import iaf_psc_exp

__all__ = [
    'iaf_psc_exp_ps_lossless',
]


class iaf_psc_exp_ps_lossless(Neuron):
    r"""NEST-compatible ``iaf_psc_exp_ps_lossless`` with lossless spike detection.

    Description
    -----------

    ``iaf_psc_exp_ps_lossless`` is a current-based leaky integrate-and-fire
    neuron with exponential excitatory/inhibitory PSC states and off-grid
    event/spike timing. The implementation follows NEST
    ``models/iaf_psc_exp_ps_lossless.{h,cpp}`` semantics: within-step event
    ordering by precise offsets, exact closed-form mini-step propagation,
    **lossless spike detection** via state-space envelope analysis (Krishnan et
    al., 2018), sub-step threshold localization by root search, and refractory
    release modeled as an explicit pseudo-event.

    Compared with :class:`iaf_psc_exp_ps`, this model adds a state-space spike
    detector that can detect spikes hidden between sampled endpoints of a
    mini-interval, preventing numerical spike loss in scenarios with fast
    subthreshold oscillations or large discrete input jumps.

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
    :meth:`iaf_psc_exp._propagator_exp`. PSC states decay exactly via
    :math:`I_X(t+\Delta t)=I_X(t)e^{-\Delta t/\tau_{syn,X}}`.

    **2. Lossless spike detection criterion**

    Before propagating each mini-interval :math:`[t, t+\Delta t]`, the
    algorithm checks whether the trajectory :math:`U(s)` crosses threshold at
    any :math:`s \in [t, t+\Delta t]` using an analytical envelope test
    (Krishnan et al., 2018). This requires :math:`\tau_{syn,ex} = \tau_{syn,in}
    = \tau_s` so that the combined synaptic current
    :math:`I(s) = I_{ex}(s) + I_{in}(s)` decays as a single exponential.

    The membrane trajectory within the mini-interval is

    .. math::

       U(s) = \frac{\tau_m}{C_m} I_e + \frac{\tau_m \tau_s}{C_m(\tau_m - \tau_s)}
       \left[ I(t) \left(e^{-(s-t)/\tau_m} - e^{-(s-t)/\tau_s}\right) \right]
       + U(t) e^{-(s-t)/\tau_m}.

    The lossless criterion computes the maximum of :math:`U(s)` over
    :math:`s \in [t, t+\Delta t]` analytically and checks if it exceeds
    :math:`U_{th}`. If so, bisection root finding localizes the precise
    crossing time. This guarantees no spike is missed due to discrete sampling,
    even when the trajectory peaks between grid points.

    **Implementation note:** The envelope test uses algebraic bounds on the
    trajectory extremum. See :meth:`is_spike_lossless` internal function and
    Krishnan et al. (2018) for derivation details.

    **3. Precise-time event processing**

    Event offsets use NEST convention: ``offset=dt`` at step start and
    ``offset=0`` at step end. For each global step:

    1. Build local event list from ``spike_events`` and on-grid delta input
       (always added at ``offset=0``).
    2. Sort events in descending offset and split the step into mini-intervals.
    3. **Apply lossless spike test** to each mini-interval before propagation.
    4. If lossless test indicates a spike, solve
       :math:`f(\delta)=U(\delta)-U_{th}=0` with bounded bisection
       (64 iterations) to obtain off-grid spike time.
    5. Reset to ``V_reset`` and enter refractory state; release from refractory
       occurs through a pseudo-event when
       ``step_idx + 1 - last_spike_step == ceil(t_ref / dt)``.

    **4. Assumptions, constraints, and computational complexity**

    - Parameters are scalar or broadcastable to ``self.varshape``.
    - Construction-time constraints enforce
      ``V_reset < V_th``, ``C_m > 0``, ``tau_m > 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, ``t_ref >= 0``, and
      **``tau_syn_ex == tau_syn_in``** (required for lossless envelope test),
      and ``tau_m != tau_syn_ex`` (to avoid singular propagator).
    - When ``V_min`` is provided: ``V_reset >= V_min``.
    - All precise offsets must satisfy ``0 <= offset <= dt``.
    - Continuous input ``x`` is buffered (stored into ``y0`` for the next
      global step), matching NEST current-event timing.
    - Per-step complexity is
      :math:`O(|\mathrm{state}| \cdot K)` for ``K`` local events, plus
      envelope test and root search cost on each mini-interval.

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
        ``self.varshape``. Must be strictly positive elementwise and must
        differ from ``tau_syn_ex`` to avoid singular propagator.
        Default is ``10. * u.ms``.
    t_ref : ArrayLike, optional
        Absolute refractory duration :math:`t_{ref}` in ms, broadcastable to
        ``self.varshape``. Can be zero (no refractory period).
        Converted at runtime to steps using ``ceil(t_ref / dt)``.
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
        **Must equal ``tau_syn_in``** for lossless spike detection.
        Default is ``2. * u.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory PSC decay constant :math:`\tau_{syn,in}` in ms,
        broadcastable to ``self.varshape`` and strictly positive.
        **Must equal ``tau_syn_ex``** for lossless spike detection.
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
         - ArrayLike, broadcastable (ms), ``> 0``, ``!= tau_syn_ex``
         - ``10. * u.ms``
         - :math:`\tau_m`
         - Leak time constant in exact subthreshold propagation.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), ``>= 0``, runtime ``ceil(t_ref/dt)``
         - ``2. * u.ms``
         - :math:`t_{ref}`
         - Absolute refractory duration (can be zero).
       * - ``V_th`` and ``V_reset``
         - ArrayLike, broadcastable (mV), with ``V_reset < V_th``
         - ``-55. * u.mV``, ``-70. * u.mV``
         - :math:`V_{th}`, :math:`V_{reset}`
         - Threshold and post-spike reset levels.
       * - ``tau_syn_ex`` and ``tau_syn_in``
         - ArrayLike, broadcastable (ms), each ``> 0``, **must be equal**
         - ``2. * u.ms``
         - :math:`\tau_{syn,ex}`, :math:`\tau_{syn,in}`
         - Exponential PSC decay constants (equality required).
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

    Returns
    -------
    out : Any
        Configured neuron node. Each :meth:`update` call returns surrogate
        spike output with shape ``self.V.value.shape``.

    Raises
    ------
    ValueError
        If validated constraints fail (for example ``V_reset >= V_th``,
        non-positive capacitance/time constants, ``V_reset < V_min``,
        ``tau_syn_ex != tau_syn_in``, ``tau_m == tau_syn_ex``, ``t_ref < 0``,
        or event offsets outside ``[0, dt]``).
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
    - The lossless spike test prevents numerical spike loss but adds
      computational overhead compared to :class:`iaf_psc_exp_ps`.
    - **Constraint ``tau_syn_ex == tau_syn_in``** is necessary because the
      analytical envelope test requires a single combined synaptic time
      constant. This differs from standard ``iaf_psc_exp_ps`` which allows
      independent excitatory/inhibitory time constants.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_exp_ps_lossless(
       ...         in_size=2, I_e=200.0 * u.pA
       ...     )
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         spk = neu.update()
       ...     _ = spk.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_exp_ps_lossless(in_size=1)
       ...     neu.init_state()
       ...     ev = [{'offset': 0.08 * u.ms, 'weight': 120.0 * u.pA}]
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = neu.update(spike_events=ev)

    References
    ----------
    .. [1] NEST source: ``models/iaf_psc_exp_ps_lossless.h`` and
           ``models/iaf_psc_exp_ps_lossless.cpp``.
    .. [2] Krishnan J, Porta Mana P, Helias M, Diesmann M, Di Napoli E (2018).
           Perfect detection of spikes in the linear sub-threshold dynamics of
           point neurons. Frontiers in Neuroinformatics 11:75.
           DOI: https://doi.org/10.3389/fninf.2017.00075
    .. [3] Rotter S, Diesmann M (1999). Exact simulation of time-invariant
           linear systems with applications to neuronal modeling.
           Biological Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    .. [4] Morrison A, Straube S, Plesser HE, Diesmann M (2007). Exact
           subthreshold integration with continuous spike times in discrete
           time neural network simulations. Neural Computation 19(1):47-79.
           DOI: https://doi.org/10.1162/neco.2007.19.1.47
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

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if self.V_min is not None and np.any(self._to_numpy(self.V_reset, u.mV) < self._to_numpy(self.V_min, u.mV)):
            raise ValueError('Reset potential must be greater than or equal to minimum potential.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')
        tau_ex = self._to_numpy(self.tau_syn_ex, u.ms)
        tau_in = self._to_numpy(self.tau_syn_in, u.ms)
        tau_m = self._to_numpy(self.tau_m, u.ms)
        if np.any(tau_m <= 0.0) or np.any(tau_ex <= 0.0) or np.any(tau_in <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(np.abs(tau_ex - tau_in) > 0.0):
            raise ValueError('tau_syn_ex == tau_syn_in is required in this implementation.')
        if np.any(np.isclose(tau_m, tau_ex)) or np.any(np.isclose(tau_m, tau_in)):
            raise ValueError('Membrane and synapse time constants must differ.')

    def init_state(self, batch_size: int = None, **kwargs):
        """Initialize membrane, synaptic, and precise-timing runtime states.

        This method allocates all internal state variables required for
        lossless precise spike-time simulation. Membrane potential ``V`` is
        initialized using ``self.V_initializer``, synaptic currents and
        buffered inputs are initialized to zero, and spike-tracking states are
        initialized to sentinel values (``last_spike_step = -1``,
        ``last_spike_time = -1e7 ms``) indicating no prior spike events.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, states use
            ``self.varshape``; otherwise states use
            ``(batch_size,) + self.varshape``. Default is ``None``.
        **kwargs : Any
            Unused compatibility arguments for subclass extension.

        Returns
        -------
        out : Any
            ``None``. The method mutates the object in-place by creating
            ``V`` (HiddenState, mV), ``I_syn_ex`` (ShortTermState, pA),
            ``I_syn_in`` (ShortTermState, pA), ``y0`` (ShortTermState, pA),
            ``is_refractory`` (ShortTermState, bool),
            ``last_spike_step`` (ShortTermState, int32),
            ``last_spike_offset`` (ShortTermState, ms),
            ``last_spike_time`` (ShortTermState, ms), and optionally
            ``refractory`` (ShortTermState, bool) when ``ref_var=True``.

        Raises
        ------
        ValueError
            If initializer outputs cannot be broadcast to state shape
            ``(batch_size,) + self.varshape`` or if shapes are incompatible.
        TypeError
            If initializer outputs are not unit-compatible with expected state
            units (mV for voltage, pA for currents, ms for time, bool for flags).
        AttributeError
            If ``self.V_initializer`` is not callable or does not produce valid
            output for the requested shape.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        last_step = braintools.init.param(braintools.init.Constant(-1), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.y0 = brainstate.ShortTermState(zeros * u.pA)
        self.is_refractory = brainstate.ShortTermState(np.zeros(V.shape, dtype=bool))
        self.last_spike_step = brainstate.ShortTermState(u.math.asarray(last_step, dtype=jnp.int32))
        self.last_spike_offset = brainstate.ShortTermState(zeros * u.ms)
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(np.zeros(V.shape, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        """Evaluate surrogate spike output from membrane potential.

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
        out : Any
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
        """Parse spike events into normalized (offset_ms, weight_array) tuples.

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
        for ev in spike_events:
            if isinstance(ev, dict):
                offs = ev.get('offset', 0.0 * u.ms)
                w = ev.get('weight', 0.0 * u.pA)
            else:
                offs, w = ev
            off_ms = float(u.math.asarray(offs / u.ms))
            w_np = np.asarray(u.math.asarray(w / u.pA), dtype=np.float64)
            events.append((off_ms, np.broadcast_to(w_np, v_shape)))
        return events

    @staticmethod
    def _bisect_root(f, t_hi: float):
        """Find root of scalar function using bounded bisection.

        Locates the point where ``f(t)`` crosses zero within the interval
        ``[0, t_hi]`` using bisection with 64 iterations. Assumes ``f`` is
        continuous and monotonically increasing within the search interval.

        This method is used to find the precise sub-step time at which the
        membrane potential crosses the spike threshold during exact integration,
        after the lossless spike detection criterion has indicated a crossing
        exists within the mini-interval.

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
        r"""Advance one global step with lossless precise spike detection.

        This method implements the complete NEST-compatible lossless
        precise-spike-time algorithm for ``iaf_psc_exp_ps_lossless``. Each
        global time step is subdivided into mini-intervals determined by spike
        event offsets. Before propagating each mini-interval, the **lossless
        spike detection criterion** (Krishnan et al., 2018) checks whether the
        membrane trajectory crosses threshold anywhere within the interval, even
        between sampled grid points. If a crossing is detected, bisection
        root-finding (64 iterations) localizes the precise sub-step spike time.

        **Update sequence:**

        1. Parse and validate ``spike_events`` and on-grid delta inputs.
        2. Sort events in descending offset (from step start to step end).
        3. For each neuron, process events sequentially:

           a. **Apply lossless spike test** to each mini-interval using
              analytical envelope bounds (see internal ``is_spike_lossless``).
           b. If test indicates spike, use bisection to find precise crossing.
           c. Propagate states exactly over the mini-interval (or up to spike).
           d. Apply event weights to PSC states (ex/in channels by sign).
           e. Apply hard reset and enter refractory state on spike.
           f. Release from refractory via pseudo-event at calculated step.

        4. Buffer incoming current ``x`` into ``y0`` for next step.
        5. Compute surrogate spike output for gradient-based learning.

        **Lossless spike detection:**

        The internal ``is_spike_lossless`` function computes analytical bounds
        on the maximum of the membrane trajectory :math:`U(s)` over the
        mini-interval :math:`s \in [t, t+\Delta t]`. This requires
        :math:`\tau_{syn,ex} = \tau_{syn,in}` so that the combined synaptic
        current decays exponentially. The test returns:

        - ``np.nan`` if no spike is detected (trajectory stays below threshold).
        - ``dt_local`` if the trajectory is above threshold at interval end.
        - A positive time value if the envelope analysis indicates a spike
          occurs within the interval (followed by bisection refinement).

        **Implementation notes:**

        - All propagation uses NumPy float64 for numerical stability.
        - Event offsets follow NEST convention: ``offset=dt`` at step start,
          ``offset=0`` at step end.
        - Refractory neurons clamp membrane potential but allow PSC decay.
        - The lossless test adds computational overhead but guarantees no spikes
          are missed due to discrete time sampling.

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
        out : Any
            Surrogate spike output from :meth:`get_spike`, shape
            ``self.V.value.shape``. Values correspond to
            ``self.spk_fun((V - V_th) / (V_th - V_reset))`` after lossless
            spike detection, exact piecewise propagation, event application,
            refractory logic, and precise spike-time localization. For neurons
            that spiked, the voltage is clamped slightly above threshold to
            ensure differentiable spike detection; for non-spiking neurons,
            voltage is clamped below threshold.

        Raises
        ------
        ValueError
            If ``t_ref < 0`` (refractory time is negative), or if any event
            offset lies outside ``[0, dt]``, or if parameter constraints are
            violated at runtime.
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

        See Also
        --------
        iaf_psc_exp_ps : Standard precise-spike model without lossless detection.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))
        t_ms = float(u.math.asarray(t / u.ms))
        step_idx = int(round(t_ms / h))
        eps = np.finfo(np.float64).eps

        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        y2 = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        I_syn_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape)
        I_syn_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape)
        y0 = self._broadcast_to_state(self._to_numpy(self.y0.value, u.pA), v_shape)

        is_refractory = self._broadcast_to_state(np.asarray(u.math.asarray(self.is_refractory.value), dtype=bool), v_shape)
        last_spike_step = self._broadcast_to_state(np.asarray(u.math.asarray(self.last_spike_step.value), dtype=np.int32), v_shape)
        last_spike_offset = self._broadcast_to_state(self._to_numpy(self.last_spike_offset.value, u.ms), v_shape)
        last_spike_time_prev = self._broadcast_to_state(self._to_numpy(self.last_spike_time.value, u.ms), v_shape)

        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        c_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        i_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        u_th = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        u_reset = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)
        u_min = -np.inf * np.ones(v_shape, dtype=np.float64)
        if self.V_min is not None:
            u_min = self._broadcast_to_state(self._to_numpy(self.V_min - self.E_L, u.mV), v_shape)

        refr_steps = self._broadcast_to_state(
            np.asarray(np.ceil(self._to_numpy(self.t_ref, u.ms) / h), dtype=np.int64),
            v_shape,
        )
        if np.any(refr_steps < 0):
            raise ValueError('Refractory time must not be negative.')

        events = self._parse_spike_events(spike_events, v_shape)
        on_grid = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        events.append((0.0, on_grid))
        events.sort(key=lambda z: z[0], reverse=True)
        for off, _ in events:
            if off < 0.0 or off > h:
                raise ValueError('All spike event offsets must satisfy 0 <= offset <= dt.')

        y0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        y0_new = np.empty_like(y0)
        y1_ex_new = np.empty_like(I_syn_ex)
        y1_in_new = np.empty_like(I_syn_in)
        y2_new = np.empty_like(y2)
        refr_new = np.empty_like(is_refractory)
        last_step_new = np.empty_like(last_spike_step)
        last_offset_new = np.empty_like(last_spike_offset)
        last_time_new = np.empty_like(last_spike_time_prev)
        spike_mask = np.zeros(v_shape, dtype=bool)
        v_for_spike = np.empty_like(y2)

        for idx in np.ndindex(v_shape):
            y0_i = float(y0[idx])
            y1e_i = float(I_syn_ex[idx])
            y1i_i = float(I_syn_in[idx])
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
                P21e = iaf_psc_exp._propagator_exp(np.asarray(tau_ex_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local)
                P21i = iaf_psc_exp._propagator_exp(np.asarray(tau_in_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local)
                y2_r = P20 * (i_e_i + before[0]) + P21e * before[1] + P21i * before[2] + before[3] * math.exp(-dt_local / tau_m_i)
                return y2_r - u_th_i

            def propagate(dt_local):
                nonlocal y1e_i, y1i_i, y2_i
                if dt_local <= 0.0:
                    return
                if not refr_i:
                    P20 = -tau_m_i / c_m_i * math.expm1(-dt_local / tau_m_i)
                    P21e = iaf_psc_exp._propagator_exp(np.asarray(tau_ex_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local)
                    P21i = iaf_psc_exp._propagator_exp(np.asarray(tau_in_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local)
                    y2_i = P20 * (i_e_i + y0_i) + P21e * y1e_i + P21i * y1i_i + y2_i * math.exp(-dt_local / tau_m_i)
                    y2_i = max(y2_i, u_min_i)
                y1e_i = y1e_i * math.exp(-dt_local / tau_ex_i)
                y1i_i = y1i_i * math.exp(-dt_local / tau_in_i)

            def is_spike_lossless(dt_local):
                if dt_local <= 0.0:
                    return np.nan
                I0 = before[1] + before[2]
                V0 = before[3]
                exp_tau_s = math.expm1(dt_local / tau_ex_i)
                exp_tau_m = math.expm1(dt_local / tau_m_i)
                exp_tau_m_s = math.expm1(dt_local / tau_m_i - dt_local / tau_ex_i)
                Ie_tot = before[0] + i_e_i

                a1 = tau_m_i * tau_ex_i
                a2 = tau_m_i * (tau_m_i - tau_ex_i)
                a3 = c_m_i * u_th_i * (tau_m_i - tau_ex_i)
                a4 = c_m_i * (tau_m_i - tau_ex_i)

                b1 = -tau_m_i * tau_m_i
                b2 = tau_m_i * tau_ex_i
                b3 = tau_m_i * c_m_i * u_th_i
                b4 = -c_m_i * (tau_m_i - tau_ex_i)

                c1 = tau_m_i / c_m_i
                c2 = (-tau_m_i * tau_ex_i) / (c_m_i * (tau_m_i - tau_ex_i))
                c3 = (tau_m_i * tau_m_i) / (c_m_i * (tau_m_i - tau_ex_i))
                c4 = tau_ex_i / tau_m_i
                c5 = (c_m_i * u_th_i) / tau_m_i
                c6 = 1.0 - (tau_ex_i / tau_m_i)

                f = (a1 * I0 * exp_tau_m_s + exp_tau_m * (a3 - Ie_tot * a2) + a3) / a4
                g = ((I0 + Ie_tot) * (b1 * exp_tau_m + b2 * exp_tau_s) + b3 * (exp_tau_m - exp_tau_s)) / (b4 * exp_tau_s)
                b_env = c1 * Ie_tot + c2 * I0 + c3 * (I0 ** c4) * ((c5 - Ie_tot) ** c6)

                if (V0 < g) and (V0 <= f):
                    return np.nan
                if V0 >= f:
                    return dt_local
                if V0 < b_env:
                    return np.nan
                try:
                    return (a1 / (tau_m_i * tau_ex_i)) * math.log(b1 * I0 / (a2 * Ie_tot - a1 * I0 - a4 * V0))
                except (ValueError, ZeroDivisionError):
                    return np.nan

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
                set_before()
                st = is_spike_lossless(h)
                propagate(h)
                if np.isfinite(st):
                    emit_spike(0.0, st)
            else:
                for ev_off, ev_w, end_of_refract in local_events:
                    ministep = last_off - ev_off
                    if ministep > 0.0:
                        set_before()
                        st = is_spike_lossless(ministep)
                        propagate(ministep)
                        if np.isfinite(st):
                            emit_spike(h - last_off, st)
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
                    st = is_spike_lossless(last_off)
                    propagate(last_off)
                    if np.isfinite(st):
                        emit_spike(h - last_off, st)

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
        self.last_spike_step.value = jnp.asarray(last_step_new, dtype=jnp.int32)
        self.last_spike_offset.value = last_offset_new * u.ms
        self.last_spike_time.value = jax.lax.stop_gradient(last_time_new * u.ms)
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.is_refractory.value)

        return self.get_spike((v_for_spike + E_L) * u.mV)
