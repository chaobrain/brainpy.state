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

from typing import Callable

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'iaf_chxk_2008',
]


class iaf_chxk_2008(NESTNeuron):
    r"""NEST-compatible ``iaf_chxk_2008`` with alpha synapses and precise AHP timing.

    Description
    -----------

    ``iaf_chxk_2008`` is a conductance-based leaky integrate-and-fire neuron
    with alpha-function excitatory/inhibitory synaptic conductances and a
    spike-triggered after-hyperpolarization (AHP) conductance, developed for
    modeling retina-LGN transmission (Casti et al., 2008). The implementation
    follows NEST ``models/iaf_chxk_2008.{h,cpp}`` semantics: adaptive RKF45
    integration, threshold crossing from below, precise in-step spike timing via
    linear interpolation, spike-triggered AHP kicks with exact sub-step decay,
    and optional ``ahp_bug`` mode that reproduces the historical single-AHP
    behavior from the original Fortran code.

    **1. Membrane and conductance dynamics**

    Let :math:`V_m` be membrane potential (mV), :math:`g_\mathrm{ex}`,
    :math:`g_\mathrm{in}`, :math:`g_\mathrm{ahp,state}` be conductance states
    (nS), and :math:`I_\mathrm{stim}` be the one-step buffered external current
    (pA). Subthreshold dynamics are

    .. math::

       \frac{dV_m}{dt} =
       \frac{-I_\mathrm{leak} - I_{\mathrm{syn,ex}} - I_{\mathrm{syn,in}}
             - I_\mathrm{ahp} + I_e + I_\mathrm{stim}}{C_m},

    where

    .. math::

       I_\mathrm{leak} = g_L (V_m - E_L),
       \quad
       I_{\mathrm{syn,ex}} = g_\mathrm{ex}(V_m - E_\mathrm{ex}),

    .. math::

       I_{\mathrm{syn,in}} = g_\mathrm{in}(V_m - E_\mathrm{in}),
       \quad
       I_\mathrm{ahp} = g_\mathrm{ahp,state}(V_m - E_\mathrm{ahp}).

    Each conductance channel (excitatory, inhibitory, AHP) evolves as an
    alpha-function state pair :math:`(dg, g_\mathrm{state})`:

    .. math::

       \frac{d\,dg}{dt} = -\frac{dg}{\tau},
       \qquad
       \frac{dg_\mathrm{state}}{dt} = dg - \frac{g_\mathrm{state}}{\tau}.

    Incoming spike weights (nS) are interpreted with sign convention: positive
    weights drive excitatory channel, negative weights (absolute value) drive
    inhibitory channel. Jumps are applied to :math:`dg` with NEST normalization:

    .. math::

       dg_\mathrm{ex} \leftarrow dg_\mathrm{ex} + \frac{e}{\tau_\mathrm{ex}} w_+,
       \qquad
       dg_\mathrm{in} \leftarrow dg_\mathrm{in} + \frac{e}{\tau_\mathrm{in}} |w_-|.

    **2. Precise output spike timing and AHP kick**

    A spike is emitted only on threshold crossing from below:

    .. math::

       V_m(t_k^-) < V_{th} \;\wedge\; V_m(t_k^+) \ge V_{th}.

    When a crossing is detected, the precise in-step spike time is computed by
    linear interpolation. Let :math:`dt_\mathrm{spike}` be the duration from
    spike time to step end:

    .. math::

       dt_\mathrm{spike}
       = h \frac{V_m(t_k^+) - V_{th}}{V_m(t_k^+) - V_m(t_k^-)},

    where :math:`h` is the step size. The AHP alpha is initialized at spike
    time and decayed forward to step end:

    .. math::

       \Delta dg_\mathrm{ahp}
       = \frac{g_\mathrm{ahp} e}{\tau_\mathrm{ahp}}
         \exp\!\left(-\frac{dt_\mathrm{spike}}{\tau_\mathrm{ahp}}\right),

    .. math::

       \Delta g_\mathrm{ahp,state}
       = \Delta dg_\mathrm{ahp}\, dt_\mathrm{spike}.

    If ``ahp_bug=True``, these values **replace** the current AHP state (single
    AHP mode); otherwise they are **added** (multiple AHP accumulation).

    **3. Numerical integration via RKF45**

    The seven coupled ODEs (:math:`V_m`, three :math:`dg` states, three
    :math:`g_\mathrm{state}` variables) are integrated using Runge-Kutta-Fehlberg
    4(5) with adaptive step size control. Local truncation error is estimated
    by comparing 4th and 5th order solutions and step size is adjusted to keep
    error below ``gsl_error_tol``. Minimum step size is ``_MIN_H = 1e-8`` ms and
    iteration limit is ``_MAX_ITERS = 10000`` per global step.

    **4. Update order matching NEST semantics**

    Each simulation step follows NEST ordering:

    1. Integrate all ODE states over :math:`[t, t+dt]` via RKF45.
    2. Check threshold crossing from below; if crossed, compute precise spike
       time and apply AHP kick at that time (with ``ahp_bug`` mode if enabled).
    3. Apply arriving signed spike weights to :math:`dg_\mathrm{ex}` and
       :math:`dg_\mathrm{in}` after integration completes.
    4. Store incoming continuous current ``x`` into buffered ``I_stim`` for
       next step (NEST current-event timing convention).

    **5. Assumptions, constraints, and failure modes**

    - Parameters are scalar or broadcastable to ``self.varshape``.
    - Construction-time constraints enforce ``C_m > 0``, ``tau_syn_ex > 0``,
      ``tau_syn_in > 0``, ``tau_ahp > 0``.
    - No explicit reset or refractory period: neuron can spike repeatedly if
      voltage remains above threshold.
    - Adaptive integration can fail if ``_MAX_ITERS`` is exceeded; in practice
      this is rare with reasonable parameter values.
    - Continuous input ``x`` passed to :meth:`update` is delayed by one step
      via ``I_stim`` (ring-buffer semantics), while spike events are applied
      after ODE integration.
    - Per-step complexity is :math:`O(|\mathrm{state}| \cdot K_\mathrm{iter})`
      where :math:`K_\mathrm{iter}` is the number of RKF45 substeps (typically
      1-5 per global step).

    Parameters
    ----------
    in_size : Size
        Population shape specification. Model parameters and states are
        broadcast to ``self.varshape`` derived from ``in_size``.
    V_th : ArrayLike, optional
        Spike threshold voltage :math:`V_{th}` in mV, broadcastable to
        ``self.varshape``. Default is ``-45. * u.mV``.
    g_L : ArrayLike, optional
        Leak conductance :math:`g_L` in nS, broadcastable to ``self.varshape``.
        Default is ``100. * u.nS``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF, broadcastable to
        ``self.varshape``. Must be strictly positive elementwise.
        Default is ``1000. * u.pF``.
    E_ex : ArrayLike, optional
        Excitatory reversal potential :math:`E_\mathrm{ex}` in mV,
        broadcastable to ``self.varshape``. Default is ``20. * u.mV``.
    E_in : ArrayLike, optional
        Inhibitory reversal potential :math:`E_\mathrm{in}` in mV,
        broadcastable to ``self.varshape``. Default is ``-90. * u.mV``.
    E_L : ArrayLike, optional
        Resting potential :math:`E_L` in mV, broadcastable to
        ``self.varshape``. Default is ``-60. * u.mV``.
    tau_syn_ex : ArrayLike, optional
        Excitatory alpha time constant :math:`\tau_\mathrm{ex}` in ms,
        broadcastable to ``self.varshape``. Must be strictly positive
        elementwise. Default is ``1. * u.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory alpha time constant :math:`\tau_\mathrm{in}` in ms,
        broadcastable to ``self.varshape``. Must be strictly positive
        elementwise. Default is ``1. * u.ms``.
    I_e : ArrayLike, optional
        Constant external current :math:`I_e` in pA, broadcastable to
        ``self.varshape``. Added in each integration substep.
        Default is ``0. * u.pA``.
    tau_ahp : ArrayLike, optional
        AHP alpha time constant :math:`\tau_\mathrm{ahp}` in ms,
        broadcastable to ``self.varshape``. Must be strictly positive
        elementwise. Default is ``0.5 * u.ms``.
    E_ahp : ArrayLike, optional
        AHP reversal potential :math:`E_\mathrm{ahp}` in mV, broadcastable to
        ``self.varshape``. Default is ``-95. * u.mV``.
    g_ahp : ArrayLike, optional
        AHP kick conductance scale :math:`g_\mathrm{ahp}` in nS,
        broadcastable to ``self.varshape``. Controls magnitude of AHP alpha
        initialized at each spike. Default is ``443.8 * u.nS``.
    ahp_bug : ArrayLike, optional
        Boolean flag (broadcastable to ``self.varshape``) enabling historical
        single-AHP bug mode. If ``True``, each spike replaces existing AHP
        state with new AHP kick. If ``False``, AHP kicks accumulate.
        Default is ``False``.
    gsl_error_tol : ArrayLike, optional
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
        Default is ``1e-3``.
    V_initializer : Callable, optional
        Initializer used by :meth:`init_state` for membrane potential ``V``.
        Must return mV-compatible values with shape compatible with
        ``self.varshape``.
        Default is ``braintools.init.Constant(-60. * u.mV)``.
    g_ex_initializer : Callable, optional
        Initializer for excitatory conductance state ``g_ex`` (nS).
        Default is ``braintools.init.Constant(0. * u.nS)``.
    g_in_initializer : Callable, optional
        Initializer for inhibitory conductance state ``g_in`` (nS).
        Default is ``braintools.init.Constant(0. * u.nS)``.
    g_ahp_initializer : Callable, optional
        Initializer for AHP conductance state ``g_ahp_state`` (nS).
        Default is ``braintools.init.Constant(0. * u.nS)``.
    spk_fun : Callable, optional
        Surrogate spike function used by :meth:`get_spike` and
        :meth:`update`. Receives normalized threshold distance tensor.
        Default is ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy forwarded to :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` matches NEST behavior. Default is ``'hard'``.
    name : str or None, optional
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 18 28 14 15 35

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
       * - ``V_th``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``-45. * u.mV``
         - :math:`V_{th}`
         - Spike threshold voltage.
       * - ``g_L``
         - ArrayLike, broadcastable (nS)
         - ``100. * u.nS``
         - :math:`g_L`
         - Leak conductance.
       * - ``C_m``
         - ArrayLike, broadcastable (pF), ``> 0``
         - ``1000. * u.pF``
         - :math:`C_m`
         - Membrane capacitance.
       * - ``E_ex``
         - ArrayLike, broadcastable (mV)
         - ``20. * u.mV``
         - :math:`E_\mathrm{ex}`
         - Excitatory reversal potential.
       * - ``E_in``
         - ArrayLike, broadcastable (mV)
         - ``-90. * u.mV``
         - :math:`E_\mathrm{in}`
         - Inhibitory reversal potential.
       * - ``E_L``
         - ArrayLike, broadcastable (mV)
         - ``-60. * u.mV``
         - :math:`E_L`
         - Resting potential.
       * - ``tau_syn_ex``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``1. * u.ms``
         - :math:`\tau_\mathrm{ex}`
         - Excitatory alpha time constant.
       * - ``tau_syn_in``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``1. * u.ms``
         - :math:`\tau_\mathrm{in}`
         - Inhibitory alpha time constant.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant external current.
       * - ``tau_ahp``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``0.5 * u.ms``
         - :math:`\tau_\mathrm{ahp}`
         - AHP alpha time constant.
       * - ``E_ahp``
         - ArrayLike, broadcastable (mV)
         - ``-95. * u.mV``
         - :math:`E_\mathrm{ahp}`
         - AHP reversal potential.
       * - ``g_ahp``
         - ArrayLike, broadcastable (nS)
         - ``443.8 * u.nS``
         - :math:`g_\mathrm{ahp}`
         - AHP kick conductance scale.
       * - ``ahp_bug``
         - ArrayLike broadcastable bool
         - ``False``
         - --
         - Enable single-AHP historical bug mode.
       * - ``gsl_error_tol``
         - ArrayLike, broadcastable, unitless, ``> 0``
         - ``1e-3``
         - --
         - Local absolute tolerance for the embedded RKF45 error estimate.
       * - ``V_initializer``
         - Callable returning mV-compatible values
         - ``Constant(-60. * u.mV)``
         - --
         - Initializes membrane state ``V``.
       * - ``g_ex_initializer``
         - Callable returning nS-compatible values
         - ``Constant(0. * u.nS)``
         - --
         - Initializes excitatory conductance.
       * - ``g_in_initializer``
         - Callable returning nS-compatible values
         - ``Constant(0. * u.nS)``
         - --
         - Initializes inhibitory conductance.
       * - ``g_ahp_initializer``
         - Callable returning nS-compatible values
         - ``Constant(0. * u.nS)``
         - --
         - Initializes AHP conductance state.
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
       * - ``name``
         - str | None
         - ``None``
         - --
         - Optional node name.

    Raises
    ------
    ValueError
        If validated constraints fail (non-positive capacitance, non-positive
        time constants, non-positive gsl_error_tol).
    TypeError
        If provided arguments are incompatible with expected units/callables
        (mV, pA, pF, ms, nS).
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
    dg_ex : ShortTermState
        Excitatory conductance rate-of-change state (nS/ms).
    g_ex : HiddenState
        Excitatory conductance state in nS.
    dg_in : ShortTermState
        Inhibitory conductance rate-of-change state (nS/ms).
    g_in : HiddenState
        Inhibitory conductance state in nS.
    dg_ahp : ShortTermState
        AHP conductance rate-of-change state (nS/ms).
    g_ahp_state : HiddenState
        AHP conductance state in nS.
    I_stim : ShortTermState
        One-step buffered external current in pA.
    integration_step : ShortTermState
        Adaptive RKF45 step size hint in ms.
    last_spike_time : ShortTermState
        Absolute precise spike time in ms.

    Notes
    -----
    - The model has no explicit membrane reset or refractory state: after
      crossing threshold, voltage continues to evolve and can spike again.
    - Continuous input ``x`` passed to :meth:`update` is **buffered** and
      affects the **next** step (NEST current-event timing).
    - Like NEST, this model provides precise output spike timing via linear
      interpolation but does not process off-grid spike-input offsets.
    - RKF45 integration is performed via the adaptive integrator and
      written back into BrainUnit states at step end.
    - ``ahp_bug=True`` reproduces the original Fortran behavior where only one
      AHP is tracked; this is primarily for validation against legacy code.

    Examples
    --------
    Create a single neuron with default parameters and simulate:

    .. code-block:: python

        >>> import brainstate as bs
        >>> import saiunit as u
        >>> from brainpy import state as bps
        >>> neuron = bps.iaf_chxk_2008(1)
        >>> with bs.environ.context(dt=0.1 * u.ms):
        ...     neuron.init_state()
        ...     spike = neuron.update(100. * u.pA)  # buffered to next step

    Inspect AHP kick behavior after spike:

    .. code-block:: python

        >>> neuron.V.value  # check membrane potential
        >>> neuron.g_ahp_state.value  # check AHP conductance state

    Recordables
    -----------

    ``V_m``, ``g_ex``, ``g_in``, ``g_ahp``, ``I_syn_ex``, ``I_syn_in``, ``I_ahp``

    References
    ----------
    .. [1] Casti A, Hayot F, Xiao Y, Kaplan E (2008). A simple model of
           retina-LGN transmission. Journal of Computational Neuroscience
           24:235-252. DOI: https://doi.org/10.1007/s10827-007-0053-7
    .. [2] NEST source: ``models/iaf_chxk_2008.h`` and
           ``models/iaf_chxk_2008.cpp``.
    """

    __module__ = 'brainpy.state'

    RECORDABLES = (
        'V_m',
        'g_ex',
        'g_in',
        'g_ahp',
        'I_syn_ex',
        'I_syn_in',
        'I_ahp',
    )

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 10000

    def __init__(
        self,
        in_size: Size,
        V_th: ArrayLike = -45.0 * u.mV,
        g_L: ArrayLike = 100.0 * u.nS,
        C_m: ArrayLike = 1000.0 * u.pF,
        E_ex: ArrayLike = 20.0 * u.mV,
        E_in: ArrayLike = -90.0 * u.mV,
        E_L: ArrayLike = -60.0 * u.mV,
        tau_syn_ex: ArrayLike = 1.0 * u.ms,
        tau_syn_in: ArrayLike = 1.0 * u.ms,
        I_e: ArrayLike = 0.0 * u.pA,
        tau_ahp: ArrayLike = 0.5 * u.ms,
        E_ahp: ArrayLike = -95.0 * u.mV,
        g_ahp: ArrayLike = 443.8 * u.nS,
        ahp_bug: ArrayLike = False,
        gsl_error_tol: ArrayLike = 1e-3,
        V_initializer: Callable = braintools.init.Constant(-60.0 * u.mV),
        g_ex_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        g_in_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        g_ahp_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.V_th = braintools.init.param(V_th, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.tau_ahp = braintools.init.param(tau_ahp, self.varshape)
        self.E_ahp = braintools.init.param(E_ahp, self.varshape)
        self.g_ahp = braintools.init.param(g_ahp, self.varshape)
        self.ahp_bug = braintools.init.param(ahp_bug, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
        self.g_ahp_initializer = g_ahp_initializer

        self._validate_parameters()

        self.integrator = AdaptiveRungeKuttaStep(
            method='RKF45',
            vf=self._vector_field,
            event_fn=None,
            min_h=self._MIN_H,
            max_iters=self._MAX_ITERS,
            atol=self.gsl_error_tol,
            dt=brainstate.environ.get_dt()
        )

        # other variable
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(0)

    @property
    def recordables(self):
        return list(self.RECORDABLES)

    def _validate_parameters(self):
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.tau_syn_ex, self.tau_ahp)):
            return

        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.tau_ahp <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

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
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape)
        g_ahp_init = braintools.init.param(self.g_ahp_initializer, self.varshape)
        zeros = u.math.zeros(self.varshape, dtype=V.dtype) * (u.nS / u.ms)

        zeros_pA = u.math.zeros(self.varshape, dtype=dftype) * u.pA

        self.V = brainstate.HiddenState(V)
        self.dg_ex = brainstate.ShortTermState(zeros)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.dg_in = brainstate.ShortTermState(zeros)
        self.g_in = brainstate.HiddenState(g_in)
        self.dg_ahp = brainstate.ShortTermState(zeros)
        self.g_ahp_state = brainstate.HiddenState(g_ahp_init)

        self.I_syn_ex = brainstate.ShortTermState(zeros_pA)
        self.I_syn_in = brainstate.ShortTermState(zeros_pA)
        self.I_ahp = brainstate.ShortTermState(zeros_pA)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms, dtype=dftype))
        self.last_spike_offset = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.ms, dtype=dftype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

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
            ``spk_fun((V - V_th) / |V_th - E_L|)``.
        """
        V = self.V.value if V is None else V
        denom = u.math.abs(self.V_th - self.E_L) + 1e-12 * u.mV
        v_scaled = (V - self.V_th) / denom
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, dg_ex, g_ex, dg_in, g_in, dg_ahp, g_ahp_state — ODE state variables.
        extra : DotDict
            Keys: i_stim — buffered external current for this step.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        i_leak = self.g_L * (state.V - self.E_L)
        i_syn_exc = state.g_ex * (state.V - self.E_ex)
        i_syn_inh = state.g_in * (state.V - self.E_in)
        i_ahp = state.g_ahp_state * (state.V - self.E_ahp)

        dV = (-i_leak - i_syn_exc - i_syn_inh - i_ahp + self.I_e + extra.i_stim) / self.C_m

        ddg_ex = -state.dg_ex / self.tau_syn_ex
        dg_ex_dt = state.dg_ex - state.g_ex / self.tau_syn_ex
        ddg_in = -state.dg_in / self.tau_syn_in
        dg_in_dt = state.dg_in - state.g_in / self.tau_syn_in
        ddg_ahp = -state.dg_ahp / self.tau_ahp
        dg_ahp_dt = state.dg_ahp - state.g_ahp_state / self.tau_ahp

        return DotDict(
            V=dV, dg_ex=ddg_ex, g_ex=dg_ex_dt,
            dg_in=ddg_in, g_in=dg_in_dt,
            dg_ahp=ddg_ahp, g_ahp_state=dg_ahp_dt,
        )


    def update(self, x=0.0 * u.pA, w_ex=None, w_in=None):
        r"""Advance the neuron by one simulation step.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous external current input in pA, broadcastable to
            ``self.varshape``. This value is stored into ``I_stim`` and applied
            at the next simulation step (one-step delay).
        w_ex : ArrayLike or None, optional
            Excitatory synaptic weight increment (nS) to add to ``dg_ex`` after
            integration, scaled by ``e/tau_syn_ex``. When ``None`` (default),
            the value is read from registered delta inputs with label ``'w_ex'``.
            Provide an explicit array for JIT-compatible (for_loop) usage.
        w_in : ArrayLike or None, optional
            Inhibitory synaptic weight increment (nS), analogous to ``w_ex``
            but for ``dg_in`` with label ``'w_in'``.

        Returns
        -------
        jax.Array
            Binary spike tensor with dtype ``jnp.float64`` and shape
            ``self.V.value.shape``. A value of ``1.0`` indicates a threshold
            crossing from below during the integrated interval :math:`(t, t+dt]`.

        Notes
        -----
        Integration uses an adaptive RKF45 loop.  Spike detection and AHP kicks
        follow NEST semantics: crossing is checked at the *global* step level
        (comparing V before and after the full integration), and the AHP state
        is updated post-integration using linear interpolation of the spike time.
        Synaptic inputs (``w_ex``, ``w_in``) are applied after integration.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        V_start = V       # saved for global-step spike offset computation
        dg_ex = self.dg_ex.value  # nS/ms
        g_ex = self.g_ex.value    # nS
        dg_in = self.dg_in.value  # nS/ms
        g_in = self.g_in.value    # nS
        dg_ahp = self.dg_ahp.value      # nS/ms
        g_ahp_state = self.g_ahp_state.value  # nS
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration (no per-substep event callback).
        ode_state = DotDict(
            V=V, dg_ex=dg_ex, g_ex=g_ex,
            dg_in=dg_in, g_in=g_in,
            dg_ahp=dg_ahp, g_ahp_state=g_ahp_state,
        )
        extra = DotDict(i_stim=i_stim)

        ode_state, h, _ = self.integrator(state=ode_state, h=h, extra=extra)
        V = ode_state.V
        dg_ex, g_ex = ode_state.dg_ex, ode_state.g_ex
        dg_in, g_in = ode_state.dg_in, ode_state.g_in
        dg_ahp, g_ahp_state = ode_state.dg_ahp, ode_state.g_ahp_state

        # Global-step spike detection: threshold crossing from below only.
        crossed = (V_start < self.V_th) & (V >= self.V_th)

        # Global-step spike-offset interpolation: time from spike to step end.
        denom = V - V_start
        safe_denom = u.math.where(u.math.abs(denom) < 1e-30 * u.mV, 1.0 * u.mV, denom)
        spike_offset = dt * (V - self.V_th) / safe_denom
        spike_offset = u.math.clip(spike_offset, 0.0 * u.ms, dt)
        spike_offset = u.math.where(crossed, spike_offset, 0.0 * u.ms)

        # Apply AHP kick post-integration (matches NEST reference semantics).
        pscon_ahp = self.g_ahp * np.e / self.tau_ahp  # nS/ms
        delta_dg_ahp = pscon_ahp * u.math.exp(-spike_offset / self.tau_ahp)
        delta_g_ahp = delta_dg_ahp * spike_offset

        ahp_bug_on = crossed & jnp.asarray(self.ahp_bug)
        ahp_bug_off = crossed & jnp.logical_not(jnp.asarray(self.ahp_bug))
        new_dg_ahp = u.math.where(ahp_bug_on, delta_dg_ahp, dg_ahp)
        new_dg_ahp = u.math.where(ahp_bug_off, new_dg_ahp + delta_dg_ahp, new_dg_ahp)
        new_g_ahp = u.math.where(ahp_bug_on, delta_g_ahp, g_ahp_state)
        new_g_ahp = u.math.where(ahp_bug_off, new_g_ahp + delta_g_ahp, new_g_ahp)
        dg_ahp = new_dg_ahp
        g_ahp_state = new_g_ahp

        # Compute recordable synaptic currents (post-integration, pre-weight-update).
        I_syn_ex = g_ex * (V - self.E_ex)    # nS * mV = pA
        I_syn_in = g_in * (V - self.E_in)    # nS * mV = pA
        I_ahp_cur = g_ahp_state * (V - self.E_ahp)  # nS * mV = pA

        # Synaptic spike inputs (applied after integration).
        if w_ex is None:
            w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
        if w_in is None:
            w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')
        pscon_ex = np.e / self.tau_syn_ex  # 1/ms
        pscon_in = np.e / self.tau_syn_in  # 1/ms
        dg_ex = dg_ex + pscon_ex * w_ex   # nS/ms
        dg_in = dg_in + pscon_in * w_in   # nS/ms

        # Update spike-time and spike-offset states.
        new_spike_offset = u.math.where(crossed, spike_offset, self.last_spike_offset.value)
        new_spike_time = u.math.where(crossed, t + dt - spike_offset, self.last_spike_time.value)

        # Write back state.
        self.V.value = V
        self.dg_ex.value = dg_ex
        self.g_ex.value = g_ex
        self.dg_in.value = dg_in
        self.g_in.value = g_in
        self.dg_ahp.value = dg_ahp
        self.g_ahp_state.value = g_ahp_state
        self.I_syn_ex.value = I_syn_ex
        self.I_syn_in.value = I_syn_in
        self.I_ahp.value = I_ahp_cur
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        self.last_spike_offset.value = jax.lax.stop_gradient(new_spike_offset)
        self.last_spike_time.value = jax.lax.stop_gradient(new_spike_time)

        return u.math.asarray(crossed, dtype=dftype)
