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

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron
from .iaf_psc_alpha import iaf_psc_alpha

__all__ = [
    'iaf_psc_alpha_ps',
]


class iaf_psc_alpha_ps(NESTNeuron):
    r"""NEST-compatible ``iaf_psc_alpha_ps`` with precise spike timing.

    Description
    -----------

    ``iaf_psc_alpha_ps`` is a current-based leaky integrate-and-fire neuron
    with alpha-shaped excitatory/inhibitory postsynaptic currents (PSCs),
    fixed absolute refractoriness, and off-grid spike/event timing. The
    implementation matches NEST ``models/iaf_psc_alpha_ps.{h,cpp}`` semantics:
    event-driven mini-step splitting inside each global ``dt`` interval,
    exact linear propagators for alpha states, and bisection-based sub-step
    threshold-time localization.

    **1. Continuous-Time Model and Alpha Current State-Space**


    Define :math:`U = V_m - E_L` and :math:`I_\mathrm{syn}=I_\mathrm{ex}+I_\mathrm{in}`.
    Subthreshold dynamics are

    .. math::

       \frac{dU}{dt} = -\frac{U}{\tau_m} + \frac{I_\mathrm{syn} + I_e + y_\mathrm{in}}{C_m}.

    For each channel :math:`X\in\{\mathrm{ex},\mathrm{in}\}`, alpha PSCs use
    a two-state system:

    .. math::

       \frac{d\,dI_X}{dt} = -\frac{dI_X}{\tau_{\mathrm{syn},X}}, \qquad
       \frac{dI_X}{dt} = dI_X - \frac{I_X}{\tau_{\mathrm{syn},X}}.

    This realizes normalized kernel

    .. math::

       i_X(t) = \frac{e}{\tau_{\mathrm{syn},X}} t e^{-t/\tau_{\mathrm{syn},X}} \Theta(t),

    so a spike weight :math:`w` (pA) is injected into derivative states as
    :math:`dI_\mathrm{ex}\leftarrow dI_\mathrm{ex}+\frac{e}{\tau_{\mathrm{syn,ex}}}w`
    for :math:`w\ge 0` and
    :math:`dI_\mathrm{in}\leftarrow dI_\mathrm{in}+\frac{e}{\tau_{\mathrm{syn,in}}}w`
    for :math:`w<0` (inhibitory channel stays negative by sign convention).

    **2. Exact Mini-Step Propagation and Precise Threshold Crossing**


    For each local interval :math:`\Delta t` between two ordered event offsets,
    the code uses exact closed-form updates:

    .. math::

       dI_X(t+\Delta t) = e^{-\Delta t/\tau_{\mathrm{syn},X}} dI_X(t),

    .. math::

       I_X(t+\Delta t) = e^{-\Delta t/\tau_{\mathrm{syn},X}}
       \big(I_X(t) + \Delta t\, dI_X(t)\big),

    .. math::

       U(t+\Delta t) = U(t) + \left(e^{-\Delta t/\tau_m}-1\right)U(t)
       + P_{30}(I_e+y_\mathrm{in})
       + \sum_X \left(P_{31,X} dI_X(t) + P_{32,X} I_X(t)\right),

    with :math:`P_{30}=\tau_m(1-e^{-\Delta t/\tau_m})/C_m` and
    :math:`P_{31,X}, P_{32,X}` evaluated by
    :meth:`iaf_psc_alpha._alpha_propagator_p31_p32` (including stable handling
    near :math:`\tau_m\approx\tau_{\mathrm{syn},X}`).

    If :math:`U` crosses :math:`U_{th}=V_{th}-E_L` inside a mini-step, the
    crossing time solves :math:`f(\delta)=U(\delta)-U_{th}=0` using bounded
    bisection (64 iterations), producing off-grid spike offset
    ``spike_off = dt - (local_time + delta)``.

    **3. Event Ordering, Refractory Pseudo-Event, and Timing Convention**


    Off-grid events are sorted by ``offset`` in descending order, where
    ``offset`` is measured from the right boundary of the current step
    (:math:`0` at step end, :math:`dt` at step start). Each neuron can also
    insert a refractory-release pseudo-event at stored ``last_spike_offset``
    when ``step_idx + 1 - last_spike_step == ceil(t_ref / dt)``.

    On spike emission:

    - membrane state is reset to ``V_reset``,
    - refractory flag is set,
    - ``last_spike_step``, ``last_spike_offset``, ``last_spike_time`` are
      updated with precise sub-step timing.

    **4. Assumptions, Constraints, and Computational Implications**


    - Construction constraints enforce ``C_m > 0``, ``tau_m > 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, and ``V_reset < V_th``.
    - If ``V_min`` is set, ``V_reset >= V_min`` is required.
    - Runtime requires ``ceil(t_ref / dt) >= 1``; otherwise update fails.
    - ``x`` is ring-buffered current input: values supplied at step ``n`` are
      consumed as ``y_input`` in step ``n+1``.
    - Update is per-neuron scalarized and iterates with ``np.ndindex`` over
      ``self.varshape``; with ``K`` within-step events, cost is
      :math:`O(|\mathrm{varshape}| \cdot K)`, plus root-search work when
      threshold is crossed.

    Parameters
    ----------
    in_size : Size
        Population shape specification. All model parameters are broadcast to
        ``self.varshape`` derived from ``in_size``.
    E_L : ArrayLike, optional
        Resting potential :math:`E_L` in mV. Scalar or array-like broadcastable
        to ``self.varshape``. Default is ``-70. * u.mV``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF. Must be strictly positive after
        broadcasting to ``self.varshape``. Default is ``250. * u.pF``.
    tau_m : ArrayLike, optional
        Membrane time constant :math:`\tau_m` in ms. Must be strictly positive.
        Default is ``10. * u.ms``.
    t_ref : ArrayLike, optional
        Absolute refractory time :math:`t_{ref}` in ms. Converted at runtime to
        grid steps via ``ceil(t_ref / dt)``. Must yield at least one step.
        Default is ``2. * u.ms``.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_{th}` in mV, broadcastable to ``self.varshape``.
        Default is ``-55. * u.mV``.
    V_reset : ArrayLike, optional
        Post-spike reset potential :math:`V_{reset}` in mV. Must satisfy
        ``V_reset < V_th`` elementwise. Default is ``-70. * u.mV``.
    tau_syn_ex : ArrayLike, optional
        Excitatory alpha time constant :math:`\tau_{\mathrm{syn,ex}}` in ms.
        Strictly positive. Default is ``2. * u.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory alpha time constant :math:`\tau_{\mathrm{syn,in}}` in ms.
        Strictly positive. Default is ``2. * u.ms``.
    I_e : ArrayLike, optional
        Constant external current :math:`I_e` in pA, broadcastable to
        ``self.varshape``. Added in each mini-step membrane update.
        Default is ``0. * u.pA``.
    V_min : ArrayLike or None, optional
        Optional lower voltage clamp :math:`V_{min}` in mV. When provided,
        membrane candidates are clipped by ``max(V, V_min)`` before threshold
        tests. ``None`` disables clipping. Default is ``None``.
    V_initializer : Callable, optional
        Initializer for membrane state ``V`` used in :meth:`init_state`.
        Must return values unit-compatible with mV. Default is
        ``braintools.init.Constant(-70. * u.mV)``.
    spk_fun : Callable, optional
        Surrogate spike function used by :meth:`get_spike` and returned by
        :meth:`update`. It receives normalized threshold distance and returns a
        spike-like array broadcastable to neuron shape.
        Default is ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy passed to :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` matches NEST hard-reset behavior. Default is ``'hard'``.
    ref_var : bool, optional
        If ``True``, creates exposed state ``self.refractory`` mirroring
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
         - Defines population shape ``self.varshape``.
       * - ``E_L``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``-70. * u.mV``
         - :math:`E_L`
         - Resting potential; membrane offset origin.
       * - ``C_m``
         - ArrayLike, broadcastable (pF), ``> 0``
         - ``250. * u.pF``
         - :math:`C_m`
         - Membrane capacitance in all propagators.
       * - ``tau_m``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``10. * u.ms``
         - :math:`\tau_m`
         - Membrane leak time constant.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), runtime ``ceil(t_ref/dt) >= 1``
         - ``2. * u.ms``
         - :math:`t_{ref}`
         - Absolute refractory duration.
       * - ``V_th`` and ``V_reset``
         - ArrayLike, broadcastable (mV), with ``V_reset < V_th``
         - ``-55. * u.mV``, ``-70. * u.mV``
         - :math:`V_{th}`, :math:`V_{reset}`
         - Threshold and reset levels.
       * - ``tau_syn_ex`` and ``tau_syn_in``
         - ArrayLike, broadcastable (ms), each ``> 0``
         - ``2. * u.ms``
         - :math:`\tau_{\mathrm{syn,ex}}`, :math:`\tau_{\mathrm{syn,in}}`
         - Alpha PSC decay constants.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant injected current.
       * - ``V_min``
         - ArrayLike broadcastable (mV) or ``None``
         - ``None``
         - :math:`V_{min}`
         - Optional lower membrane bound.
       * - ``V_initializer``
         - Callable returning mV-compatible values
         - ``Constant(-70. * u.mV)``
         - --
         - Initial membrane state initializer.
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
        If parameter constraints are violated (for example ``C_m <= 0``,
        ``tau_m <= 0``, ``tau_syn_ex <= 0``, ``tau_syn_in <= 0``,
        ``V_reset >= V_th``, ``V_reset < V_min``), if refractory duration in
        steps is below one, or if any ``spike_events`` offset is outside
        ``[0, dt]``.
    TypeError
        If supplied quantities are not unit-compatible with expected units
        (mV, ms, pA, pF) during conversion.
    KeyError
        If simulation context keys such as ``t`` or ``dt`` are missing when
        :meth:`update` is called.
    AttributeError
        If :meth:`update` is called before :meth:`init_state` creates required
        states (for example ``V`` or synaptic buffers).

    Notes
    -----
    - ``spike_events`` accepts ``(offset, weight)`` tuples or
      ``{'offset': ..., 'weight': ...}`` dicts. Offsets are in ms and measured
      from the right step boundary (NEST convention).
    - Positive event weights update the excitatory derivative state; negative
      event weights update inhibitory derivative state.
    - The implementation computes all internal propagators in ``float64`` NumPy
      space and writes back BrainUnit states afterward.
    - ``last_spike_time`` stores precise absolute spike time in ms and is
      stop-gradient wrapped.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_alpha_ps(in_size=(2,), I_e=220.0 * u.pA)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=1.0 * u.ms):
       ...         spk = neu.update()
       ...     _ = spk.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import saiunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_alpha_ps(in_size=1)
       ...     neu.init_state()
       ...     ev = [{'offset': 0.08 * u.ms, 'weight': 120.0 * u.pA}]
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = neu.update(spike_events=ev)

    References
    ----------
    .. [1] NEST source: ``models/iaf_psc_alpha_ps.h`` and
           ``models/iaf_psc_alpha_ps.cpp``.
    .. [2] Rotter S, Diesmann M (1999). Exact simulation of time-invariant linear
           systems with applications to neuronal modeling. Biological Cybernetics
           81:381-402. DOI: https://doi.org/10.1007/s004220050570
    .. [3] Morrison A, Straube S, Plesser HE, Diesmann M (2007). Exact
           subthreshold integration with continuous spike times in discrete time
           neural network simulations. Neural Computation 19(1):47-79.
           DOI: https://doi.org/10.1162/neco.2007.19.1.47
    .. [4] Hanuschkin A, Kunkel S, Helias M, Morrison A, Diesmann M (2010).
           A general and efficient method for incorporating exact spike times in
           globally time-driven simulations. Frontiers in Neuroinformatics.
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

    @staticmethod
    def _to_numpy(x, unit):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if self.V_min is not None and np.any(self._to_numpy(self.V_reset, u.mV) < self._to_numpy(self.V_min, u.mV)):
            raise ValueError('Reset potential must be greater equal minimum potential.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        last_step = braintools.init.param(braintools.init.Constant(-1), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        dftype = brainstate.environ.dftype()
        self.dI_syn_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=dftype))
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.dI_syn_in = brainstate.ShortTermState(np.asarray(zeros, dtype=dftype))
        self.y_input = brainstate.ShortTermState(zeros * u.pA)
        self.is_refractory = brainstate.ShortTermState(np.zeros(V.shape, dtype=bool))
        ditype = brainstate.environ.ditype()
        self.last_spike_step = brainstate.ShortTermState(u.math.asarray(last_step, dtype=ditype))
        self.last_spike_offset = brainstate.ShortTermState(zeros * u.ms)
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(np.zeros(V.shape, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
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
            dftype = brainstate.environ.dftype()
            w_np = np.asarray(u.math.asarray(w / u.pA), dtype=dftype)
            events.append((off_ms, np.broadcast_to(w_np, v_shape)))
        return events

    @staticmethod
    def _bisect_root(f, t_hi: float):
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
        r"""Advance one simulation step with optional precise within-step events.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous external current in pA for the current global step.
            Value is accumulated through :meth:`sum_current_inputs` and written
            to ``self.y_input`` for use in the next step (one-step buffering).
            Scalar or array-like broadcastable to ``self.V.value.shape``.
            Default is ``0. * u.pA``.
        spike_events : Iterable[tuple[Any, Any] | dict[str, Any]] or None, optional
            Optional off-grid spike events within this ``dt`` step. Each item is
            either ``(offset, weight)`` or ``{'offset': ..., 'weight': ...}``,
            where ``offset`` is in ms from the right step edge and ``weight`` is
            in pA. ``offset`` must satisfy ``0 <= offset <= dt``.
            Positive weights target excitatory alpha derivative state; negative
            weights target inhibitory alpha derivative state. ``None`` means no
            extra within-step events. On-grid delta inputs collected from
            :meth:`sum_delta_inputs` are still included at ``offset=0``.

        Returns
        -------
        out : jax.Array
            Spike output from :meth:`get_spike` with shape
            ``self.V.value.shape``. Values are surrogate spikes from
            ``self.spk_fun`` evaluated on threshold-scaled membrane potential
            after precise-time integration and event handling.

        Raises
        ------
        ValueError
            If computed refractory steps satisfy ``ceil(t_ref / dt) < 1`` or if
            any event offset is outside ``[0, dt]``.
        KeyError
            If simulation context values ``t`` or ``dt`` are missing.
        TypeError
            If provided quantities are not unit-compatible with ms/pA during
            conversion of ``x`` or ``spike_events``.
        AttributeError
            If called before required states are initialized via
            :meth:`init_state`.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))
        t_ms = float(u.math.asarray(t / u.ms))
        step_idx = int(round(t_ms / h))
        eps = np.finfo(np.float64).eps

        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        V_m = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        I_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape)
        dftype = brainstate.environ.dftype()
        dI_ex = self._broadcast_to_state(np.asarray(self.dI_syn_ex.value, dtype=dftype), v_shape)
        I_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape)
        dI_in = self._broadcast_to_state(np.asarray(self.dI_syn_in.value, dtype=dftype), v_shape)
        y_input = self._broadcast_to_state(self._to_numpy(self.y_input.value, u.pA), v_shape)

        is_refractory = self._broadcast_to_state(np.asarray(u.math.asarray(self.is_refractory.value), dtype=bool),
                                                 v_shape)
        ditype = brainstate.environ.ditype()
        last_spike_step = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.last_spike_step.value), dtype=ditype), v_shape)
        last_spike_offset = self._broadcast_to_state(self._to_numpy(self.last_spike_offset.value, u.ms), v_shape)
        last_spike_time_prev = self._broadcast_to_state(self._to_numpy(self.last_spike_time.value, u.ms), v_shape)

        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        c_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        i_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        u_th = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        u_reset = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)
        u_min = -np.inf * np.ones(v_shape, dtype=dftype)
        if self.V_min is not None:
            u_min = self._broadcast_to_state(self._to_numpy(self.V_min - self.E_L, u.mV), v_shape)

        refr_steps = self._broadcast_to_state(
            np.asarray(np.ceil(self._to_numpy(self.t_ref, u.ms) / h), dtype=ditype),
            v_shape,
        )
        if np.any(refr_steps < 1):
            raise ValueError('Refractory time must be at least one time step.')

        psc_norm_ex = math.e / tau_ex
        psc_norm_in = math.e / tau_in

        events = self._parse_spike_events(spike_events, v_shape)
        on_grid = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        events.append((0.0, on_grid))
        events.sort(key=lambda z: z[0], reverse=True)
        for off, _ in events:
            if off < 0.0 or off > h:
                raise ValueError('All spike event offsets must satisfy 0 <= offset <= dt.')

        y_input_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        y_input_new = np.empty_like(y_input)
        I_ex_new = np.empty_like(I_ex)
        dI_ex_new = np.empty_like(dI_ex)
        I_in_new = np.empty_like(I_in)
        dI_in_new = np.empty_like(dI_in)
        V_new = np.empty_like(V_m)
        refr_new = np.empty_like(is_refractory)
        last_step_new = np.empty_like(last_spike_step)
        last_offset_new = np.empty_like(last_spike_offset)
        last_time_new = np.empty_like(last_spike_time_prev)
        spike_mask = np.zeros(v_shape, dtype=bool)
        v_for_spike = np.empty_like(V_m)

        for idx in np.ndindex(v_shape):
            y0_i = float(y_input[idx])
            Ie_i = float(I_ex[idx])
            dIe_i = float(dI_ex[idx])
            Ii_i = float(I_in[idx])
            dIi_i = float(dI_in[idx])
            V_i = float(V_m[idx])
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
            pnorm_ex = float(psc_norm_ex[idx])
            pnorm_in = float(psc_norm_in[idx])

            did_spike = False
            before = [y0_i, Ie_i, dIe_i, Ii_i, dIi_i, V_i]

            def set_before():
                before[0] = y0_i
                before[1] = Ie_i
                before[2] = dIe_i
                before[3] = Ii_i
                before[4] = dIi_i
                before[5] = V_i

            def threshold_distance(dt_local):
                expm1_tm = math.expm1(-dt_local / tau_m_i)
                P30 = -tau_m_i / c_m_i * expm1_tm
                P31e, P32e = iaf_psc_alpha._alpha_propagator_p31_p32(
                    np.asarray(tau_ex_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local
                )
                P31i, P32i = iaf_psc_alpha._alpha_propagator_p31_p32(
                    np.asarray(tau_in_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local
                )
                V_r = (
                    P30 * (i_e_i + before[0])
                    + P31e * before[2]
                    + P32e * before[1]
                    + P31i * before[4]
                    + P32i * before[3]
                    + before[5] * expm1_tm
                    + before[5]
                )
                return V_r - u_th_i

            def propagate(dt_local):
                nonlocal Ie_i, dIe_i, Ii_i, dIi_i, V_i
                if dt_local <= 0.0:
                    return
                if not refr_i:
                    expm1_tm = math.expm1(-dt_local / tau_m_i)
                    P30 = -tau_m_i / c_m_i * expm1_tm
                    P31e, P32e = iaf_psc_alpha._alpha_propagator_p31_p32(
                        np.asarray(tau_ex_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local
                    )
                    P31i, P32i = iaf_psc_alpha._alpha_propagator_p31_p32(
                        np.asarray(tau_in_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local
                    )
                    V_i = (
                        P30 * (i_e_i + y0_i)
                        + P31e * dIe_i
                        + P32e * Ie_i
                        + P31i * dIi_i
                        + P32i * Ii_i
                        + V_i * expm1_tm
                        + V_i
                    )
                    V_i = max(V_i, u_min_i)

                exp_ex = math.exp(-dt_local / tau_ex_i)
                exp_in = math.exp(-dt_local / tau_in_i)
                Ie_i = exp_ex * dt_local * dIe_i + exp_ex * Ie_i
                dIe_i = exp_ex * dIe_i
                Ii_i = exp_in * dt_local * dIi_i + exp_in * Ii_i
                dIi_i = exp_in * dIi_i

            def emit_spike(t0, dt_local):
                nonlocal V_i, refr_i, last_step_i, last_off_i, spike_time_i, did_spike
                root = self._bisect_root(threshold_distance, dt_local)
                spike_off = h - (t0 + root)
                spike_off = min(h, max(0.0, spike_off))
                last_step_i = step_idx + 1
                last_off_i = spike_off
                V_i = u_reset_i
                refr_i = True
                spike_time_i = t_ms + h - spike_off
                did_spike = True

            def emit_instant_spike(spike_off):
                nonlocal V_i, refr_i, last_step_i, last_off_i, spike_time_i, did_spike
                so = min(h, max(0.0, spike_off))
                last_step_i = step_idx + 1
                last_off_i = so
                V_i = u_reset_i
                refr_i = True
                spike_time_i = t_ms + h - so
                did_spike = True

            if (not refr_i) and (V_i >= u_th_i):
                emit_instant_spike(h * (1.0 - eps))

            local_events = [(off, w[idx], False) for off, w in events]
            if refr_i and (step_idx + 1 - last_step_i == refr_steps_i):
                local_events.append((last_off_i, 0.0, True))
            local_events.sort(key=lambda z: z[0], reverse=True)

            last_off = h
            if len(local_events) == 0:
                set_before()
                propagate(h)
                if V_i >= u_th_i:
                    emit_spike(0.0, h)
            else:
                for ev_off, ev_w, end_of_refract in local_events:
                    ministep = last_off - ev_off
                    if ministep > 0.0:
                        set_before()
                        propagate(ministep)
                        if V_i >= u_th_i:
                            emit_spike(h - last_off, ministep)
                    if end_of_refract:
                        refr_i = False
                    else:
                        if ev_w >= 0.0:
                            dIe_i += pnorm_ex * ev_w
                        else:
                            dIi_i += pnorm_in * ev_w
                    set_before()
                    last_off = ev_off
                if last_off > 0.0:
                    set_before()
                    propagate(last_off)
                    if V_i >= u_th_i:
                        emit_spike(h - last_off, last_off)

            y0_i = float(y_input_next[idx])
            y_input_new[idx] = y0_i
            I_ex_new[idx] = Ie_i
            dI_ex_new[idx] = dIe_i
            I_in_new[idx] = Ii_i
            dI_in_new[idx] = dIi_i
            V_new[idx] = V_i
            refr_new[idx] = refr_i
            last_step_new[idx] = last_step_i
            last_offset_new[idx] = last_off_i
            last_time_new[idx] = spike_time_i
            spike_mask[idx] = did_spike
            v_for_spike[idx] = (u_th_i + 1e-12) if did_spike else min(V_i, u_th_i - 1e-12)

        self.y_input.value = y_input_new * u.pA
        self.I_syn_ex.value = I_ex_new * u.pA
        self.dI_syn_ex.value = dI_ex_new
        self.I_syn_in.value = I_in_new * u.pA
        self.dI_syn_in.value = dI_in_new
        self.V.value = (V_new + E_L) * u.mV
        self.is_refractory.value = jnp.asarray(refr_new, dtype=bool)
        self.last_spike_step.value = jnp.asarray(last_step_new, dtype=ditype)
        self.last_spike_offset.value = last_offset_new * u.ms
        self.last_spike_time.value = jax.lax.stop_gradient(last_time_new * u.ms)
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.is_refractory.value)

        return self.get_spike((v_for_spike + E_L) * u.mV)
