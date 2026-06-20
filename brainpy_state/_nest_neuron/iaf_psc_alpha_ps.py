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

from collections.abc import Callable, Iterable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, cond_any
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
    - Update is vectorized over ``self.varshape`` using array operations.
      With ``K`` within-step events, cost is
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
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_alpha_ps(in_size=(2,), I_e=220.0 * u.pA)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=1.0 * u.ms):
       ...         spk = neu.update()
       ...     _ = spk.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
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
        name: str | None = None,
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

    def _validate_parameters(self):
        r"""Validate model parameters against NEST constraints.

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
        r"""Initialize persistent and short-term state variables.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Raises
        ------
        ValueError
            If an initializer cannot be broadcast to requested shape.
        TypeError
            If initializer outputs have incompatible units/dtypes for the
            corresponding state variables.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()

        V = braintools.init.param(self.V_initializer, self.varshape)
        zeros = u.math.zeros(self.varshape, dtype=V.dtype)

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.dI_syn_ex = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=dftype))
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.dI_syn_in = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=dftype))
        self.y_input = brainstate.ShortTermState(zeros * u.pA)
        self.is_refractory = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(False), self.varshape)
        )
        self.last_spike_step = brainstate.ShortTermState(
            u.math.full(self.varshape, -1, dtype=ditype)
        )
        self.last_spike_offset = brainstate.ShortTermState(
            u.math.zeros(self.varshape, dtype=dftype) * u.ms
        )
        self.last_spike_time = brainstate.ShortTermState(
            u.math.full(self.varshape, -1e7 * u.ms)
        )

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(
                braintools.init.param(braintools.init.Constant(False), self.varshape)
            )

    def get_spike(self, V: ArrayLike = None):
        r"""Evaluate surrogate spike output from membrane voltage.

        Parameters
        ----------
        V : ArrayLike, optional
            Voltage values with shape broadcastable to ``self.varshape`` and
            units compatible with mV. If ``None``, uses current state
            ``self.V.value``.

        Returns
        -------
        ArrayLike
            Surrogate spike activation produced by
            ``spk_fun((V - V_th) / (V_th - V_reset))``.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    @staticmethod
    def _parse_spike_events(spike_events: Iterable, v_shape):
        """Parse spike events into a list of (offset_ms, weight_np) tuples.

        Parameters
        ----------
        spike_events : Iterable or None
            Off-grid spike events within this step.
        v_shape : tuple
            Target state shape for broadcasting weights.

        Returns
        -------
        list of (float, np.ndarray)
            Parsed events as (offset_in_ms, weight_array) pairs.
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
            off_ms = float(u.get_mantissa(offs / u.ms))
            w_np = np.broadcast_to(
                np.asarray(u.get_mantissa(w / u.pA), dtype=dftype),
                v_shape,
            )
            events.append((off_ms, w_np))
        return events

    def _precompute_constants(self, h, v_shape, dftype, ditype):
        """Pre-compute constant numpy parameter arrays for use in update().

        Caches all parameter-derived arrays that are invariant across
        simulation steps (fixed dt, shape, and dtypes).  Subsequent
        calls to update() reuse the cached arrays, eliminating per-step
        JAX dispatch overhead for parameter conversions.

        Parameters
        ----------
        h : float
            Step size in ms.
        v_shape : tuple
            State array shape.
        dftype : dtype
            Float dtype for computations.
        ditype : dtype
            Integer dtype for step counters.
        """
        _tnp = lambda x, unit: np.broadcast_to(
            np.asarray(u.get_mantissa(x / unit), dtype=dftype), v_shape
        )

        E_L = _tnp(self.E_L, u.mV)
        refr_steps = np.broadcast_to(
            np.asarray(np.ceil(_tnp(self.t_ref, u.ms) / h), dtype=ditype), v_shape
        )
        if cond_any(refr_steps < 1):
            raise ValueError('Refractory time must be at least one time step.')

        self._c_E_L = E_L
        self._c_tau_m = _tnp(self.tau_m, u.ms)
        self._c_tau_ex = _tnp(self.tau_syn_ex, u.ms)
        self._c_tau_in = _tnp(self.tau_syn_in, u.ms)
        self._c_c_m = _tnp(self.C_m, u.pF)
        self._c_i_e = _tnp(self.I_e, u.pA)
        self._c_u_th = _tnp(self.V_th - self.E_L, u.mV)
        self._c_u_reset = _tnp(self.V_reset - self.E_L, u.mV)
        self._c_u_min = -np.inf * np.ones(v_shape, dtype=dftype)
        if self.V_min is not None:
            self._c_u_min = _tnp(self.V_min - self.E_L, u.mV)
        self._c_refr_steps = refr_steps
        self._c_psc_norm_ex = np.e / self._c_tau_ex
        self._c_psc_norm_in = np.e / self._c_tau_in
        # Cache key
        self._c_key = (h, v_shape, dftype, ditype)

    @staticmethod
    def _bisect_root(f, t_hi: float):
        """Find root of f in [0, t_hi] using bisection (64 iterations).

        Parameters
        ----------
        f : callable
            Scalar function to find root of.
        t_hi : float
            Upper bound of search interval.

        Returns
        -------
        float
            Approximate root location.
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

    def _propagate_vectorized(self, dt_local, V_m, I_ex, dI_ex, I_in, dI_in,
                              y0, tau_m, tau_ex, tau_in, c_m, i_e, u_min,
                              is_refractory):
        """Propagate all state variables forward by dt_local (vectorized).

        Parameters
        ----------
        dt_local : np.ndarray
            Local time step for each neuron.
        V_m, I_ex, dI_ex, I_in, dI_in : np.ndarray
            State variables.
        y0 : np.ndarray
            Buffered input current.
        tau_m, tau_ex, tau_in, c_m, i_e : np.ndarray
            Model parameters.
        u_min : np.ndarray
            Lower voltage clamp.
        is_refractory : np.ndarray
            Boolean refractory mask.

        Returns
        -------
        tuple of np.ndarray
            Updated (V_m, I_ex, dI_ex, I_in, dI_in).
        """
        active = dt_local > 0.0

        # Membrane propagation (only for non-refractory neurons).
        expm1_tm = np.where(active, np.expm1(-dt_local / tau_m), 0.0)
        P30 = np.where(active, -tau_m / c_m * expm1_tm, 0.0)
        P31e, P32e = iaf_psc_alpha._alpha_propagator_p31_p32(tau_ex, tau_m, c_m, dt_local)
        P31i, P32i = iaf_psc_alpha._alpha_propagator_p31_p32(tau_in, tau_m, c_m, dt_local)

        V_candidate = (
            P30 * (i_e + y0)
            + P31e * dI_ex
            + P32e * I_ex
            + P31i * dI_in
            + P32i * I_in
            + V_m * expm1_tm
            + V_m
        )
        V_candidate = np.maximum(V_candidate, u_min)
        V_new = np.where(active & ~is_refractory, V_candidate, V_m)

        # Synaptic state propagation (always, regardless of refractory).
        exp_ex = np.where(active, np.exp(-dt_local / tau_ex), 1.0)
        exp_in = np.where(active, np.exp(-dt_local / tau_in), 1.0)
        I_ex_new = np.where(active, exp_ex * dt_local * dI_ex + exp_ex * I_ex, I_ex)
        dI_ex_new = np.where(active, exp_ex * dI_ex, dI_ex)
        I_in_new = np.where(active, exp_in * dt_local * dI_in + exp_in * I_in, I_in)
        dI_in_new = np.where(active, exp_in * dI_in, dI_in)

        return V_new, I_ex_new, dI_ex_new, I_in_new, dI_in_new

    def _threshold_distance_vectorized(self, dt_local, V_before, I_ex_before,
                                       dI_ex_before, I_in_before, dI_in_before,
                                       y0, tau_m, tau_ex, tau_in, c_m, i_e, u_th):
        """Compute threshold distance after propagation by dt_local (vectorized).

        Parameters
        ----------
        dt_local : np.ndarray or float
            Local time step.
        V_before, I_ex_before, dI_ex_before, I_in_before, dI_in_before : np.ndarray
            State variables before propagation.
        y0 : np.ndarray
            Buffered input current.
        tau_m, tau_ex, tau_in, c_m, i_e : np.ndarray
            Model parameters.
        u_th : np.ndarray
            Threshold in relative coordinates.

        Returns
        -------
        np.ndarray
            V(t + dt_local) - u_th for each neuron.
        """
        expm1_tm = np.expm1(-dt_local / tau_m)
        P30 = -tau_m / c_m * expm1_tm
        P31e, P32e = iaf_psc_alpha._alpha_propagator_p31_p32(tau_ex, tau_m, c_m, dt_local)
        P31i, P32i = iaf_psc_alpha._alpha_propagator_p31_p32(tau_in, tau_m, c_m, dt_local)
        V_r = (
            P30 * (i_e + y0)
            + P31e * dI_ex_before
            + P32e * I_ex_before
            + P31i * dI_in_before
            + P32i * I_in_before
            + V_before * expm1_tm
            + V_before
        )
        return V_r - u_th

    def _bisect_vectorized(self, t_hi, V_before, I_ex_before, dI_ex_before,
                           I_in_before, dI_in_before, y0, tau_m, tau_ex,
                           tau_in, c_m, i_e, u_th, mask):
        """Vectorized bisection to find threshold crossing time.

        Parameters
        ----------
        t_hi : np.ndarray
            Upper bound of search interval for each neuron.
        V_before, I_ex_before, dI_ex_before, I_in_before, dI_in_before : np.ndarray
            State variables before the ministep.
        y0 : np.ndarray
            Buffered input current.
        tau_m, tau_ex, tau_in, c_m, i_e : np.ndarray
            Model parameters.
        u_th : np.ndarray
            Threshold in relative coordinates.
        mask : np.ndarray
            Boolean mask of neurons to perform bisection on.

        Returns
        -------
        np.ndarray
            Approximate crossing times for each neuron (only valid where mask is True).
        """
        lo = np.zeros_like(t_hi)
        hi = t_hi.copy()

        for _ in range(64):
            mid = 0.5 * (lo + hi)
            f_mid = self._threshold_distance_vectorized(
                mid, V_before, I_ex_before, dI_ex_before,
                I_in_before, dI_in_before, y0, tau_m, tau_ex, tau_in, c_m, i_e, u_th,
            )
            crossed = f_mid > 0.0
            hi = np.where(mask & crossed, mid, hi)
            lo = np.where(mask & ~crossed, mid, lo)

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
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        h = float(u.get_mantissa(dt_q / u.ms))
        t_ms = float(u.get_mantissa(t / u.ms))
        step_idx = int(round(t_ms / h))
        eps = np.finfo(np.float64).eps

        v_shape = self.V.value.shape

        # Use cached constant parameter arrays; recompute only when key changes.
        if not hasattr(self, '_c_key') or self._c_key != (h, v_shape, dftype, ditype):
            self._precompute_constants(h, v_shape, dftype, ditype)

        E_L = self._c_E_L
        tau_m = self._c_tau_m
        tau_ex = self._c_tau_ex
        tau_in = self._c_tau_in
        c_m = self._c_c_m
        i_e = self._c_i_e
        u_th = self._c_u_th
        u_reset = self._c_u_reset
        u_min = self._c_u_min
        refr_steps = self._c_refr_steps
        psc_norm_ex = self._c_psc_norm_ex
        psc_norm_in = self._c_psc_norm_in

        # Convert per-step state arrays to unitless numpy.
        _tn_state = lambda x, unit: np.broadcast_to(
            np.asarray(u.get_mantissa(x / unit), dtype=dftype), v_shape
        )

        V_m = _tn_state(self.V.value, u.mV) - E_L
        I_ex = _tn_state(self.I_syn_ex.value, u.pA)
        dI_ex = np.broadcast_to(np.asarray(self.dI_syn_ex.value, dtype=dftype), v_shape)
        I_in = _tn_state(self.I_syn_in.value, u.pA)
        dI_in = np.broadcast_to(np.asarray(self.dI_syn_in.value, dtype=dftype), v_shape)
        y_input = _tn_state(self.y_input.value, u.pA)

        is_refractory = np.broadcast_to(
            np.asarray(self.is_refractory.value, dtype=bool), v_shape
        ).copy()
        last_spike_step = np.broadcast_to(
            np.asarray(self.last_spike_step.value, dtype=ditype), v_shape
        ).copy()
        last_spike_offset = _tn_state(self.last_spike_offset.value, u.ms).copy()
        last_spike_time_prev = _tn_state(self.last_spike_time.value, u.ms).copy()

        # Parse spike events and add on-grid delta inputs.
        events = self._parse_spike_events(spike_events, v_shape)
        on_grid = np.broadcast_to(
            np.asarray(u.get_mantissa(self.sum_delta_inputs(0. * u.pA) / u.pA), dtype=dftype),
            v_shape,
        )
        events.append((0.0, on_grid))
        events.sort(key=lambda z: z[0], reverse=True)
        for off, _ in events:
            if off < 0.0 or off > h:
                raise ValueError('All spike event offsets must satisfy 0 <= offset <= dt.')

        # Current input for next step (one-step delay).
        y_input_next = np.broadcast_to(
            np.asarray(u.get_mantissa(self.sum_current_inputs(x, self.V.value) / u.pA), dtype=dftype),
            v_shape,
        )

        # Working copies for mutation during event processing.
        V_m = V_m.copy()
        I_ex = I_ex.copy()
        dI_ex = dI_ex.copy()
        I_in = I_in.copy()
        dI_in = dI_in.copy()

        spike_mask = np.zeros(v_shape, dtype=bool)

        # --- Handle neurons already above threshold at start of step ---
        instant_spike = (~is_refractory) & (V_m >= u_th)
        if cond_any(instant_spike):
            spike_off = h * (1.0 - eps)
            last_spike_step = np.where(instant_spike, step_idx + 1, last_spike_step)
            last_spike_offset = np.where(instant_spike, spike_off, last_spike_offset)
            V_m = np.where(instant_spike, u_reset, V_m)
            is_refractory = is_refractory | instant_spike
            last_spike_time_prev = np.where(instant_spike, t_ms + h - spike_off, last_spike_time_prev)
            spike_mask = spike_mask | instant_spike

        # --- Build local events including refractory-release pseudo-event ---
        # Determine which neurons need a refractory-release event.
        refr_release = is_refractory & ((step_idx + 1 - last_spike_step) == refr_steps)
        refr_release_offset = np.where(refr_release, last_spike_offset, -1.0)

        # Combine external events with refractory-release events and sort.
        # Process events from largest offset (step start) to smallest (step end).
        all_offsets = [off for off, _ in events]
        if cond_any(refr_release):
            unique_refr_offsets = np.unique(refr_release_offset[refr_release])
            all_offsets = sorted(set(all_offsets) | set(unique_refr_offsets.tolist()), reverse=True)
        else:
            all_offsets = sorted(set(all_offsets), reverse=True)

        # Build event lookup: for each offset, get the weight array (if any).
        event_weight_map = {}
        for off, w in events:
            if off in event_weight_map:
                event_weight_map[off] = event_weight_map[off] + w
            else:
                event_weight_map[off] = w.copy()

        # Process all events in descending offset order.
        last_off = np.full(v_shape, h, dtype=dftype)

        for ev_off in all_offsets:
            ministep = last_off - ev_off

            # Propagate where ministep > 0.
            propagate_mask = ministep > 0.0
            if cond_any(propagate_mask):
                dt_local = np.where(propagate_mask, ministep, 0.0)
                V_before = V_m.copy()
                I_ex_before = I_ex.copy()
                dI_ex_before = dI_ex.copy()
                I_in_before = I_in.copy()
                dI_in_before = dI_in.copy()

                V_m, I_ex, dI_ex, I_in, dI_in = self._propagate_vectorized(
                    dt_local, V_m, I_ex, dI_ex, I_in, dI_in,
                    y_input, tau_m, tau_ex, tau_in, c_m, i_e, u_min, is_refractory,
                )

                # Check for threshold crossing.
                crossed = propagate_mask & (~is_refractory) & (V_m >= u_th)
                if cond_any(crossed):
                    root = self._bisect_vectorized(
                        dt_local, V_before, I_ex_before, dI_ex_before,
                        I_in_before, dI_in_before, y_input, tau_m, tau_ex,
                        tau_in, c_m, i_e, u_th, crossed,
                    )
                    spike_off = h - ((h - last_off) + root)
                    spike_off = np.clip(spike_off, 0.0, h)
                    last_spike_step = np.where(crossed, step_idx + 1, last_spike_step)
                    last_spike_offset = np.where(crossed, spike_off, last_spike_offset)
                    V_m = np.where(crossed, u_reset, V_m)
                    is_refractory = is_refractory | crossed
                    last_spike_time_prev = np.where(crossed, t_ms + h - spike_off, last_spike_time_prev)
                    spike_mask = spike_mask | crossed

            # Apply event: refractory release or synaptic weight.
            is_refr_release_here = refr_release & (np.abs(refr_release_offset - ev_off) < 1e-15)
            is_refractory = np.where(is_refr_release_here, False, is_refractory)

            if ev_off in event_weight_map:
                ev_w = event_weight_map[ev_off]
                # Non-refractory-release neurons get synaptic input.
                apply_weight = ~is_refr_release_here
                dI_ex = np.where(apply_weight & (ev_w >= 0.0), dI_ex + psc_norm_ex * ev_w, dI_ex)
                dI_in = np.where(apply_weight & (ev_w < 0.0), dI_in + psc_norm_in * ev_w, dI_in)

            last_off = np.where(propagate_mask | is_refr_release_here, ev_off, last_off)

        # --- Final propagation from last event to step end ---
        final_ministep = last_off
        propagate_final = final_ministep > 0.0
        if cond_any(propagate_final):
            dt_local = np.where(propagate_final, final_ministep, 0.0)
            V_before = V_m.copy()
            I_ex_before = I_ex.copy()
            dI_ex_before = dI_ex.copy()
            I_in_before = I_in.copy()
            dI_in_before = dI_in.copy()

            V_m, I_ex, dI_ex, I_in, dI_in = self._propagate_vectorized(
                dt_local, V_m, I_ex, dI_ex, I_in, dI_in,
                y_input, tau_m, tau_ex, tau_in, c_m, i_e, u_min, is_refractory,
            )

            # Check for threshold crossing in final segment.
            crossed = propagate_final & (~is_refractory) & (V_m >= u_th)
            if cond_any(crossed):
                root = self._bisect_vectorized(
                    dt_local, V_before, I_ex_before, dI_ex_before,
                    I_in_before, dI_in_before, y_input, tau_m, tau_ex,
                    tau_in, c_m, i_e, u_th, crossed,
                )
                spike_off = h - ((h - last_off) + root)
                spike_off = np.clip(spike_off, 0.0, h)
                last_spike_step = np.where(crossed, step_idx + 1, last_spike_step)
                last_spike_offset = np.where(crossed, spike_off, last_spike_offset)
                V_m = np.where(crossed, u_reset, V_m)
                is_refractory = is_refractory | crossed
                last_spike_time_prev = np.where(crossed, t_ms + h - spike_off, last_spike_time_prev)
                spike_mask = spike_mask | crossed

        # Construct spike output voltage for surrogate gradient.
        v_for_spike = np.where(spike_mask, u_th + 1e-12, np.minimum(V_m, u_th - 1e-12))

        # Write back state.
        self.y_input.value = y_input_next * u.pA
        self.I_syn_ex.value = I_ex * u.pA
        self.dI_syn_ex.value = dI_ex
        self.I_syn_in.value = I_in * u.pA
        self.dI_syn_in.value = dI_in
        self.V.value = (V_m + E_L) * u.mV
        self.is_refractory.value = jnp.asarray(is_refractory, dtype=bool)
        self.last_spike_step.value = jnp.asarray(last_spike_step, dtype=ditype)
        self.last_spike_offset.value = last_spike_offset * u.ms
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time_prev * u.ms)
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.is_refractory.value)

        return self.get_spike((v_for_spike + E_L) * u.mV)
