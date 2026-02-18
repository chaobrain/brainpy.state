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
from typing import Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

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
    error below ``_ATOL = 1e-3``. Minimum step size is ``_MIN_H = 1e-8`` ms and
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
    - Continuous input ``x`` passed to :meth:`update` affects the **next** step
      due to one-step buffering.
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
    V_initializer : Callable, optional
        Initializer used by :meth:`init_state` for membrane potential ``V``.
        Must return mV-compatible values with shape compatible with
        ``self.varshape`` (and optional batch prefix).
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
        time constants).
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
        Excitatory conductance rate-of-change state (dimensionless).
    g_ex : HiddenState
        Excitatory conductance state in nS.
    dg_in : ShortTermState
        Inhibitory conductance rate-of-change state (dimensionless).
    g_in : HiddenState
        Inhibitory conductance state in nS.
    dg_ahp : ShortTermState
        AHP conductance rate-of-change state (dimensionless).
    g_ahp_state : HiddenState
        AHP conductance state in nS.
    I_syn_ex : ShortTermState
        Excitatory synaptic current in pA.
    I_syn_in : ShortTermState
        Inhibitory synaptic current in pA.
    I_ahp : ShortTermState
        AHP current in pA.
    I_stim : ShortTermState
        One-step buffered external current in pA.
    integration_step : ShortTermState
        Adaptive RKF45 step size hint in ms.
    last_spike_time : ShortTermState
        Absolute precise spike time in ms.
    last_spike_offset : ShortTermState
        Precise offset (ms) from right step boundary for latest spike.

    Notes
    -----
    - The model has no explicit membrane reset or refractory state: after
      crossing threshold, voltage continues to evolve and can spike again.
    - Continuous input ``x`` passed to :meth:`update` is **buffered** and
      affects the **next** step (NEST current-event timing).
    - Like NEST, this model provides precise output spike timing via linear
      interpolation but does not process off-grid spike-input offsets.
    - RKF45 integration is performed in float64 for numerical stability and
      written back into BrainUnit states at step end.
    - ``ahp_bug=True`` reproduces the original Fortran behavior where only one
      AHP is tracked; this is primarily for validation against legacy code.

    Examples
    --------
    Create a single neuron with default parameters and simulate:

    .. code-block:: python

        >>> import brainstate as bs
        >>> import brainunit as u
        >>> import brainpy.state as bps
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

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
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

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
        self.g_ahp_initializer = g_ahp_initializer

        self._validate_parameters()

    @property
    def recordables(self):
        return list(self.RECORDABLES)

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _to_bool_numpy(x):
        return np.asarray(u.math.asarray(x), dtype=bool)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_ahp, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        g_ahp = braintools.init.param(self.g_ahp_initializer, self.varshape, batch_size)

        zeros = np.zeros_like(np.asarray(u.math.asarray(V / u.mV), dtype=np.float64))

        self.V = brainstate.HiddenState(V)
        self.dg_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.g_ex = brainstate.HiddenState(g_ex)
        self.dg_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.g_in = brainstate.HiddenState(g_in)
        self.dg_ahp = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.g_ahp_state = brainstate.HiddenState(g_ahp)

        self.I_syn_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64) * u.pA)
        self.I_syn_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64) * u.pA)
        self.I_ahp = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64) * u.pA)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        self.last_spike_offset = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64) * u.ms)

        dt = brainstate.environ.get_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.g_ex.value = braintools.init.param(self.g_ex_initializer, self.varshape, batch_size)
        self.g_in.value = braintools.init.param(self.g_in_initializer, self.varshape, batch_size)
        self.g_ahp_state.value = braintools.init.param(self.g_ahp_initializer, self.varshape, batch_size)

        zeros = np.zeros_like(np.asarray(u.math.asarray(self.V.value / u.mV), dtype=np.float64))
        self.dg_ex.value = np.asarray(zeros, dtype=np.float64)
        self.dg_in.value = np.asarray(zeros, dtype=np.float64)
        self.dg_ahp.value = np.asarray(zeros, dtype=np.float64)

        self.I_syn_ex.value = np.asarray(zeros, dtype=np.float64) * u.pA
        self.I_syn_in.value = np.asarray(zeros, dtype=np.float64) * u.pA
        self.I_ahp.value = np.asarray(zeros, dtype=np.float64) * u.pA

        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        self.last_spike_offset.value = np.asarray(zeros, dtype=np.float64) * u.ms

        dt = brainstate.environ.get_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        denom = u.math.abs(self.V_th - self.E_L) + 1e-12 * u.mV
        v_scaled = (V - self.V_th) / denom
        return self.spk_fun(v_scaled)

    def _sum_signed_delta_inputs(self):
        w_ex = u.math.zeros_like(self.g_ex.value)
        w_in = u.math.zeros_like(self.g_in.value)
        if self.delta_inputs is None:
            return w_ex, w_in

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            zero = u.math.zeros_like(out)
            w_ex = w_ex + u.math.maximum(out, zero)
            w_in = w_in + u.math.maximum(-out, zero)
        return w_ex, w_in

    @staticmethod
    def _dynamics_scalar(v, dg_ex, g_ex, dg_in, g_in, dg_ahp, g_ahp_state, i_stim, p):
        i_syn_exc = g_ex * (v - p['E_ex'])
        i_syn_inh = g_in * (v - p['E_in'])
        i_ahp = g_ahp_state * (v - p['E_ahp'])
        i_leak = p['g_L'] * (v - p['E_L'])

        dv = (-i_leak - i_syn_exc - i_syn_inh - i_ahp + i_stim + p['I_e']) / p['C_m']
        ddg_ex = -dg_ex / p['tau_syn_ex']
        dg_ex_dt = dg_ex - g_ex / p['tau_syn_ex']
        ddg_in = -dg_in / p['tau_syn_in']
        dg_in_dt = dg_in - g_in / p['tau_syn_in']
        ddg_ahp = -dg_ahp / p['tau_ahp']
        dg_ahp_dt = dg_ahp - g_ahp_state / p['tau_ahp']
        return dv, ddg_ex, dg_ex_dt, ddg_in, dg_in_dt, ddg_ahp, dg_ahp_dt

    def _rkf45_integrate_scalar(self, v0, dg_ex0, g_ex0, dg_in0, g_in0, dg_ahp0, g_ahp0, i_stim, h0, dt, p):
        t = 0.0
        h = max(h0, self._MIN_H)
        y = np.asarray([v0, dg_ex0, g_ex0, dg_in0, g_in0, dg_ahp0, g_ahp0], dtype=np.float64)
        iters = 0

        def f(y_):
            return np.asarray(
                self._dynamics_scalar(
                    y_[0], y_[1], y_[2], y_[3], y_[4], y_[5], y_[6], i_stim, p
                ),
                dtype=np.float64
            )

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = max(self._MIN_H, min(h, dt - t))

            k1 = f(y)
            k2 = f(y + h * (1.0 / 4.0) * k1)
            k3 = f(y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
            k4 = f(y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0))
            k5 = f(y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0))
            k6 = f(
                y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0))

            y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
            y5 = y + h * (
                    16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0)
            err = float(np.max(np.abs(y5 - y4)))

            if err <= self._ATOL or h <= self._MIN_H:
                y = y5
                t += h
                fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (self._ATOL / err) ** 0.2))
                h = max(self._MIN_H, h * fac)
            else:
                fac = min(1.0, max(0.2, 0.9 * (self._ATOL / err) ** 0.25))
                h = max(self._MIN_H, h * fac)

        return y[0], y[1], y[2], y[3], y[4], y[5], y[6], h

    def update(self, x=0.0 * u.pA):
        t_q = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        t_ms = float(u.math.asarray(t_q / u.ms))
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        dg_ex = self._broadcast_to_state(np.asarray(self.dg_ex.value, dtype=np.float64), v_shape)
        g_ex = self._broadcast_to_state(self._to_numpy(self.g_ex.value, u.nS), v_shape)
        dg_in = self._broadcast_to_state(np.asarray(self.dg_in.value, dtype=np.float64), v_shape)
        g_in = self._broadcast_to_state(self._to_numpy(self.g_in.value, u.nS), v_shape)
        dg_ahp = self._broadcast_to_state(np.asarray(self.dg_ahp.value, dtype=np.float64), v_shape)
        g_ahp_state = self._broadcast_to_state(self._to_numpy(self.g_ahp_state.value, u.nS), v_shape)

        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape)
        last_spike_time = self._broadcast_to_state(self._to_numpy(self.last_spike_time.value, u.ms), v_shape)
        last_spike_offset = self._broadcast_to_state(self._to_numpy(self.last_spike_offset.value, u.ms), v_shape)

        p = {
            'V_th': self._broadcast_to_state(self._to_numpy(self.V_th, u.mV), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'E_ex': self._broadcast_to_state(self._to_numpy(self.E_ex, u.mV), v_shape),
            'E_in': self._broadcast_to_state(self._to_numpy(self.E_in, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'tau_syn_ex': self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape),
            'tau_syn_in': self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
            'tau_ahp': self._broadcast_to_state(self._to_numpy(self.tau_ahp, u.ms), v_shape),
            'E_ahp': self._broadcast_to_state(self._to_numpy(self.E_ahp, u.mV), v_shape),
            'PSConInit_E': self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_ex, u.ms), v_shape),
            'PSConInit_I': self._broadcast_to_state(np.e / self._to_numpy(self.tau_syn_in, u.ms), v_shape),
            'PSConInit_AHP': self._broadcast_to_state(
                self._to_numpy(self.g_ahp, u.nS) * np.e / self._to_numpy(self.tau_ahp, u.ms),
                v_shape,
            ),
            'ahp_bug': self._broadcast_to_state(self._to_bool_numpy(self.ahp_bug), v_shape),
        }

        w_ex_q, w_in_q = self._sum_signed_delta_inputs()
        w_ex = self._broadcast_to_state(self._to_numpy(w_ex_q, u.nS), v_shape)
        w_in = self._broadcast_to_state(self._to_numpy(w_in_q, u.nS), v_shape)

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        spike_mask = np.zeros(v_shape, dtype=bool)
        v_for_spike = np.empty_like(V)

        V_next = np.empty_like(V)
        dg_ex_next = np.empty_like(dg_ex)
        g_ex_next = np.empty_like(g_ex)
        dg_in_next = np.empty_like(dg_in)
        g_in_next = np.empty_like(g_in)
        dg_ahp_next = np.empty_like(dg_ahp)
        g_ahp_next = np.empty_like(g_ahp_state)
        h_next = np.empty_like(h_int)
        last_spike_time_next = np.empty_like(last_spike_time)
        last_spike_offset_next = np.empty_like(last_spike_offset)

        i_syn_ex_next = np.empty_like(V)
        i_syn_in_next = np.empty_like(V)
        i_ahp_next = np.empty_like(V)

        tiny = np.finfo(np.float64).tiny
        eps = np.finfo(np.float64).eps

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            vm_prev = V[idx]

            v_i, dg_ex_i, g_ex_i, dg_in_i, g_in_i, dg_ahp_i, g_ahp_i, h_i = self._rkf45_integrate_scalar(
                V[idx],
                dg_ex[idx],
                g_ex[idx],
                dg_in[idx],
                g_in[idx],
                dg_ahp[idx],
                g_ahp_state[idx],
                i_stim[idx],
                h_int[idx],
                dt,
                local_p,
            )

            crossed = (vm_prev < local_p['V_th']) and (v_i >= local_p['V_th'])
            if crossed:
                denom = v_i - vm_prev
                if abs(denom) < tiny:
                    dt_from_spike_to_end = 0.0
                else:
                    dt_from_spike_to_end = dt * (v_i - local_p['V_th']) / denom
                dt_from_spike_to_end = min(dt, max(0.0, float(dt_from_spike_to_end)))

                delta_dg = local_p['PSConInit_AHP'] * math.exp(-dt_from_spike_to_end / local_p['tau_ahp'])
                delta_g = delta_dg * dt_from_spike_to_end

                if bool(local_p['ahp_bug']):
                    g_ahp_i = delta_g
                    dg_ahp_i = delta_dg
                else:
                    g_ahp_i = g_ahp_i + delta_g
                    dg_ahp_i = dg_ahp_i + delta_dg

                spike_mask[idx] = True
                last_spike_time_i = t_ms + dt - dt_from_spike_to_end
                last_spike_offset_i = dt_from_spike_to_end
                v_for_spike[idx] = max(v_i, local_p['V_th'] + eps)
            else:
                last_spike_time_i = last_spike_time[idx]
                last_spike_offset_i = last_spike_offset[idx]
                if v_i >= local_p['V_th']:
                    # Crossing from below is required for spiking.
                    v_for_spike[idx] = np.nextafter(local_p['V_th'], -np.inf)
                else:
                    v_for_spike[idx] = v_i

            dg_ex_i = dg_ex_i + local_p['PSConInit_E'] * w_ex[idx]
            dg_in_i = dg_in_i + local_p['PSConInit_I'] * w_in[idx]

            V_next[idx] = v_i
            dg_ex_next[idx] = dg_ex_i
            g_ex_next[idx] = g_ex_i
            dg_in_next[idx] = dg_in_i
            g_in_next[idx] = g_in_i
            dg_ahp_next[idx] = dg_ahp_i
            g_ahp_next[idx] = g_ahp_i
            h_next[idx] = h_i
            last_spike_time_next[idx] = last_spike_time_i
            last_spike_offset_next[idx] = last_spike_offset_i

            i_syn_ex_next[idx] = g_ex_i * (v_i - local_p['E_ex'])
            i_syn_in_next[idx] = g_in_i * (v_i - local_p['E_in'])
            i_ahp_next[idx] = g_ahp_i * (v_i - local_p['E_ahp'])

        self.V.value = V_next * u.mV
        self.dg_ex.value = dg_ex_next
        self.g_ex.value = g_ex_next * u.nS
        self.dg_in.value = dg_in_next
        self.g_in.value = g_in_next * u.nS
        self.dg_ahp.value = dg_ahp_next
        self.g_ahp_state.value = g_ahp_next * u.nS

        self.I_syn_ex.value = i_syn_ex_next * u.pA
        self.I_syn_in.value = i_syn_in_next * u.pA
        self.I_ahp.value = i_ahp_next * u.pA

        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time_next * u.ms)
        self.last_spike_offset.value = jax.lax.stop_gradient(last_spike_offset_next * u.ms)

        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float64) * u.mV)
