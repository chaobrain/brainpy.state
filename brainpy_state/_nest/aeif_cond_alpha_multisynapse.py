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
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, validate_aeif_overflow, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'aeif_cond_alpha_multisynapse',
]


class aeif_cond_alpha_multisynapse(NESTNeuron):
    r"""NEST-compatible ``aeif_cond_alpha_multisynapse`` neuron model.

    Conductance-based adaptive exponential integrate-and-fire neuron with
    alpha-shaped synapses and an arbitrary number of receptor ports.

    Parameters
    ----------
    in_size : Size
        Population shape. All per-neuron states use ``self.varshape`` derived
        from ``in_size``.
    V_peak : ArrayLike
        Spike detection voltage in mV, broadcastable to ``self.varshape``.
        Used as detection threshold only when ``Delta_T > 0``.
    V_reset : ArrayLike
        Reset and refractory clamp voltage in mV, broadcastable to
        ``self.varshape``.
    t_ref : ArrayLike
        Absolute refractory duration in ms, broadcastable to ``self.varshape``.
        Converted to integer grid counts by ``ceil(t_ref / dt)``.
    g_L : ArrayLike
        Leak conductance in nS, broadcastable to ``self.varshape``.
    C_m : ArrayLike
        Membrane capacitance in pF, broadcastable to ``self.varshape``.
    E_L : ArrayLike
        Leak reversal potential in mV, broadcastable to ``self.varshape``.
    Delta_T : ArrayLike
        Exponential slope factor in mV, broadcastable to ``self.varshape``.
        ``Delta_T == 0`` disables the exponential current and switches
        detection to ``V_th``.
    tau_w : ArrayLike
        Adaptation time constant in ms, broadcastable to ``self.varshape``.
    a : ArrayLike
        Subthreshold adaptation conductance in nS, broadcastable to
        ``self.varshape``.
    b : ArrayLike
        Spike-triggered adaptation jump in pA, broadcastable to
        ``self.varshape``.
    V_th : ArrayLike
        Exponential soft-threshold voltage in mV, broadcastable to
        ``self.varshape``.
    tau_syn : ArrayLike
        Receptor time constants in ms. Values are flattened to shape
        ``(n_receptors,)``; each entry must be strictly positive.
    E_rev : ArrayLike
        Receptor reversal potentials in mV. Values are flattened to shape
        ``(n_receptors,)`` and must have the same length as ``tau_syn``.
    I_e : ArrayLike
        Constant external current in pA, broadcastable to ``self.varshape``.
    gsl_error_tol : ArrayLike
        Unitless local absolute tolerance for RKF45, broadcastable to
        ``self.varshape`` and strictly positive.
    V_initializer : Callable
        Initializer for membrane voltage ``V`` (mV domain). Must support shape
        ``self.varshape`` (and optional batch axis via framework init helpers).
    g_initializer : Callable
        Initializer for conductance state ``g`` (nS domain). Must support
        shape ``self.varshape + (n_receptors,)``.
    w_initializer : Callable
        Initializer for adaptation current ``w`` (pA domain), shape
        ``self.varshape``.
    spk_fun : Callable
        Surrogate spike function used by :meth:`get_spike`.
    spk_reset : str
        Reset policy from :class:`~brainpy_state._base.Neuron`; ``'hard'``
        matches NEST semantics.
    ref_var : bool
        If ``True``, allocates a boolean refractory state variable
        ``self.refractory``.
    name : str | None
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 17 25 15 20 43

       * - Parameter
         - Type / shape / unit
         - Default
         - Math symbol
         - Semantics
       * - ``in_size``
         - :class:`~brainstate.typing.Size`; scalar or tuple
         - required
         - --
         - Population shape defining ``self.varshape``.
       * - ``V_peak``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``0.0 * u.mV``
         - :math:`V_\mathrm{peak}`
         - Spike detection threshold when ``Delta_T > 0`` and membrane RHS
           clamp bound via :math:`\min(V, V_{\mathrm{peak}})`.
       * - ``V_reset``
         - ArrayLike, broadcastable (mV)
         - ``-60.0 * u.mV``
         - :math:`V_\mathrm{reset}`
         - Reset value and refractory clamp voltage.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms)
         - ``0.0 * u.ms``
         - :math:`t_\mathrm{ref}`
         - Absolute refractory duration converted to integer step counts.
       * - ``g_L`` and ``C_m``
         - ArrayLike, broadcastable (nS, pF)
         - ``30.0 * u.nS``, ``281.0 * u.pF``
         - :math:`g_L`, :math:`C_m`
         - Leak conductance and membrane capacitance in AdEx membrane
           dynamics.
       * - ``E_L``, ``Delta_T``, and ``V_th``
         - ArrayLike, broadcastable (mV)
         - ``-70.6 * u.mV``, ``2.0 * u.mV``, ``-50.4 * u.mV``
         - :math:`E_L`, :math:`\Delta_T`, :math:`V_\mathrm{th}`
         - Leak reversal, exponential slope, and soft threshold.
       * - ``tau_w``, ``a``, and ``b``
         - ArrayLike, broadcastable (ms, nS, pA)
         - ``144.0 * u.ms``, ``4.0 * u.nS``, ``80.5 * u.pA``
         - :math:`\tau_w`, :math:`a`, :math:`b`
         - Adaptation time scale, coupling, and spike-triggered jump.
       * - ``tau_syn``
         - ArrayLike, flattened to ``(n_receptors,)`` (ms)
         - ``(2.0,) * u.ms``
         - :math:`\tau_{\mathrm{syn},k}`
         - Receptor-specific alpha time constants; each ``> 0``.
       * - ``E_rev``
         - ArrayLike, flattened to ``(n_receptors,)`` (mV)
         - ``(0.0,) * u.mV``
         - :math:`E_{\mathrm{rev},k}`
         - Receptor-specific reversal potentials, same length as ``tau_syn``.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0.0 * u.pA``
         - :math:`I_e`
         - Constant current added every RKF45 substep.
       * - ``gsl_error_tol``
         - ArrayLike, broadcastable, unitless, ``> 0``
         - ``1e-6``
         - --
         - Embedded RKF45 local error tolerance.
       * - ``V_initializer``
         - Callable
         - ``Constant(-70.6 * u.mV)``
         - --
         - Initializer for ``V``.
       * - ``g_initializer``
         - Callable
         - ``Constant(0.0 * u.nS)``
         - --
         - Initializer for ``g`` with shape ``[..., n_receptors]``.
       * - ``w_initializer``
         - Callable
         - ``Constant(0.0 * u.pA)``
         - --
         - Initializer for adaptation current ``w``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate nonlinearity used by :meth:`get_spike`.
       * - ``spk_reset``
         - str
         - ``'hard'``
         - --
         - Reset mode inherited from :class:`~brainpy_state._base.Neuron`.
       * - ``ref_var``
         - bool
         - ``False``
         - --
         - If ``True``, exposes boolean ``self.refractory``.
       * - ``name``
         - str | None
         - ``None``
         - --
         - Optional node name.

    Raises
    ------
    ValueError
        Raised at initialization or update time if any of the following holds:

        - ``E_rev`` and ``tau_syn`` lengths differ.
        - Any ``tau_syn <= 0``, ``tau_w <= 0``, ``C_m <= 0``,
          ``gsl_error_tol <= 0``, or ``t_ref < 0``.
        - Any ``V_peak < V_th`` or ``V_reset >= V_peak``.
        - Any ``Delta_T < 0`` or overflow guard on
          ``(V_peak - V_th) / Delta_T`` is violated for ``Delta_T > 0``.
        - Incoming spike event receptor index is outside
          ``[1, n_receptors]``.
        - Incoming conductance weights are negative (both explicit
          ``spike_events`` and default ``add_delta_input`` path).
        - Nonzero default delta-conductance input is provided when
          ``n_receptors == 0``.
        - Runtime instability guard is triggered during integration
          (``V < -1e3`` mV or ``w`` outside ``[-1e6, 1e6]`` pA).

    Description
    -----------

    ``aeif_cond_alpha_multisynapse`` follows NEST
    ``models/aeif_cond_alpha_multisynapse.{h,cpp}``.
    It extends ``aeif_cond_alpha`` by replacing fixed excitatory/inhibitory
    channels with receptor-indexed alpha conductances.

    Each receptor ``k`` has:

    - synaptic time constant ``tau_syn[k]``,
    - reversal potential ``E_rev[k]``,
    - alpha states ``dg[k]`` and ``g[k]``.

    Receptor ports are 1-based (NEST convention): ``1..n_receptors``.

    **1. Continuous dynamics**

    Let :math:`V` be membrane voltage, :math:`w` adaptation current, and
    :math:`g_k` receptor conductances.

    .. math::

       C_m \frac{dV}{dt}
       =
       -g_L (V - E_L)
       + g_L \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)
       + \sum_k g_k (E_{\mathrm{rev},k} - V)
       - w + I_e + I_{stim}.

    Adaptation dynamics:

    .. math::

       \tau_w \frac{dw}{dt} = a (V - E_L) - w.

    Receptor alpha states:

    .. math::

       \frac{d\,dg_k}{dt} = -\frac{dg_k}{\tau_{\mathrm{syn},k}},
       \qquad
       \frac{d g_k}{dt} = dg_k - \frac{g_k}{\tau_{\mathrm{syn},k}}.

    Incoming spike weights ``w_k`` (in nS) are applied as:

    .. math::

       dg_k \leftarrow dg_k + \frac{e}{\tau_{\mathrm{syn},k}} w_k.

    The :math:`e/\tau_{\mathrm{syn},k}` factor is the alpha-kernel
    normalization from NEST. For a single spike with weight :math:`w_k`,
    the resulting conductance is:

    .. math::

       g_k(t) = w_k \frac{t}{\tau_{\mathrm{syn},k}}
       \exp\!\left(1 - \frac{t}{\tau_{\mathrm{syn},k}}\right), \quad t \ge 0,

    which peaks at :math:`t=\tau_{\mathrm{syn},k}` with peak value :math:`w_k`.

    **2. Spike and refractory semantics**

    - During refractory integration, effective voltage is clamped to
      ``V_reset`` and :math:`dV/dt = 0`.
    - Outside refractory period, the RHS uses :math:`\min(V, V_{peak})`.
    - Spike detection threshold is:
      - ``V_peak`` if ``Delta_T > 0``,
      - ``V_th`` if ``Delta_T == 0``.
    - On each detected spike (inside RKF45 substeps):
      - ``V <- V_reset``
      - ``w <- w + b``
      - refractory counter ``r <- refractory_counts + 1`` if refractory is enabled.

    **3. Update order per simulation step (NEST semantics)**

    1. Integrate ODEs on :math:`(t, t+dt]` using adaptive RKF45.
    2. Inside integration loop: refractory clamp and spike/reset/adaptation.
    3. Decrement refractory counter once.
    4. Apply incoming receptor-specific spike events to ``dg``.
    5. Store continuous current input ``x`` into one-step delayed ``I_stim``.

    **4. Event semantics**

    ``spike_events`` passed to :meth:`update` must be an iterable of
    ``(receptor_type, weight)`` or dictionaries with keys
    ``receptor_type``/``receptor`` and ``weight``.

    - Receptor types are 1-based and must satisfy ``1 <= receptor_type <= n_receptors``.
    - Weights are conductances (nS) and must be non-negative, matching NEST
      conductance multisynapse constraints.
    - ``add_delta_input`` stream is mapped to receptor 1 by default; those
      values must also be non-negative.

    State variables
    ---------------

    - ``V``: membrane potential :math:`V_m`.
    - ``w``: adaptation current.
    - ``dg``: alpha auxiliary states per receptor ``[..., n_receptors]``.
    - ``g``: receptor conductances ``[..., n_receptors]``.
    - ``refractory_step_count``: remaining refractory grid steps.
    - ``integration_step``: persistent RKF45 internal step size.
    - ``I_stim``: one-step delayed current buffer.
    - ``last_spike_time``: last emitted spike time (:math:`t+dt` on spike).
    - ``refractory``: optional boolean refractory indicator.

    Recordables
    -----------

    Dynamic recordables follow NEST naming:

    - ``V_m``
    - ``w``
    - ``g_1``, ``g_2``, ..., ``g_n``

    Notes
    -----

    - Default ``t_ref = 0`` matches NEST and can allow multiple spikes inside
      one simulation step.
    - This implementation targets source-level parity with NEST update ordering
      rather than high-performance vectorization.
    - Computational cost scales with
      ``prod(self.V.value.shape) * n_receptors`` and is dominated by
      scalar RKF45 substepping in Python loops.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state._nest.aeif_cond_alpha_multisynapse import (
       ...     aeif_cond_alpha_multisynapse,
       ... )
       >>> _ = brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms)
       >>> neu = aeif_cond_alpha_multisynapse(
       ...     in_size=4,
       ...     tau_syn=(0.5, 2.0) * u.ms,
       ...     E_rev=(0.0, -75.0) * u.mV,
       ... )
       >>> neu.init_state()
       >>> spikes = neu.update(
       ...     x=120.0 * u.pA,
       ...     spike_events=[{'receptor_type': 1, 'weight': 0.8 * u.nS}],
       ... )
       >>> spikes.shape
       (4,)

    References
    ----------
    .. [1] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
           model as an effective description of neuronal activity.
           Journal of Neurophysiology, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [2] NEST source: ``models/aeif_cond_alpha_multisynapse.h`` and
           ``models/aeif_cond_alpha_multisynapse.cpp``.
    """

    __module__ = 'brainpy.state'

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        V_peak: ArrayLike = 0.0 * u.mV,
        V_reset: ArrayLike = -60.0 * u.mV,
        t_ref: ArrayLike = 0.0 * u.ms,
        g_L: ArrayLike = 30.0 * u.nS,
        C_m: ArrayLike = 281.0 * u.pF,
        E_L: ArrayLike = -70.6 * u.mV,
        Delta_T: ArrayLike = 2.0 * u.mV,
        tau_w: ArrayLike = 144.0 * u.ms,
        a: ArrayLike = 4.0 * u.nS,
        b: ArrayLike = 80.5 * u.pA,
        V_th: ArrayLike = -50.4 * u.mV,
        tau_syn: ArrayLike = (2.0,) * u.ms,
        E_rev: ArrayLike = (0.0,) * u.mV,
        I_e: ArrayLike = 0.0 * u.pA,
        gsl_error_tol: ArrayLike = 1e-6,
        V_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        g_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        w_initializer: Callable = braintools.init.Constant(0.0 * u.pA),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.V_peak = braintools.init.param(V_peak, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.Delta_T = braintools.init.param(Delta_T, self.varshape)
        self.tau_w = braintools.init.param(tau_w, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        dftype = brainstate.environ.dftype()
        self.tau_syn = np.asarray(u.math.asarray(tau_syn / u.ms), dtype=dftype).reshape(-1)
        self.E_rev = np.asarray(u.math.asarray(E_rev / u.mV), dtype=dftype).reshape(-1)

        self.V_initializer = V_initializer
        self.g_initializer = g_initializer
        self.w_initializer = w_initializer
        self.ref_var = ref_var

        self._validate_parameters()

        self.integrator = AdaptiveRungeKuttaStep(
            method='RKF45',
            vf=self._vector_field,
            event_fn=self._event_fn,
            min_h=self._MIN_H,
            max_iters=self._MAX_ITERS,
            atol=self.gsl_error_tol,
            dt=brainstate.environ.get_dt()
        )

        # other variable
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    @property
    def n_receptors(self):
        return int(self.tau_syn.size)

    @property
    def recordables(self):
        return ['V_m', 'w', *[f'g_{i + 1}' for i in range(self.n_receptors)]]

    def _validate_parameters(self):
        v_reset = self.V_reset
        v_peak = self.V_peak
        v_th = self.V_th
        delta_t = self.Delta_T / u.mV

        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (v_reset, v_peak, v_th, delta_t)):
            return

        if self.E_rev.size != self.tau_syn.size:
            raise ValueError('The E_rev and tau_syn arrays must have the same size.')
        if cond_any(self.tau_syn <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive.')
        if cond_any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if cond_any(v_reset >= v_peak):
            raise ValueError('Ensure that: V_reset < V_peak .')
        if cond_any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if cond_any(self.tau_w <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        # Mirror NEST overflow guard for exponential term at spike time.
        validate_aeif_overflow(v_peak, v_th, delta_t)

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
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        w = braintools.init.param(self.w_initializer, self.varshape)
        g = braintools.init.param(self.g_initializer, self.varshape + (self.n_receptors,))

        self.V = brainstate.HiddenState(V)
        self.w = brainstate.HiddenState(w)

        # dg has shape varshape + (n_receptors,), stored unitless (mantissa in nS/ms)
        zeros_dg = u.math.zeros(self.varshape + (self.n_receptors,), dtype=V.dtype)
        self.dg = brainstate.ShortTermState(zeros_dg)
        self.g = brainstate.HiddenState(g)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        dftype = brainstate.environ.dftype()
        out = np.zeros(v_shape + (self.n_receptors,), dtype=dftype)
        if spike_events is None:
            return out

        if isinstance(spike_events, dict):
            spike_events = [spike_events]

        for ev in spike_events:
            if isinstance(ev, dict):
                receptor = int(ev.get('receptor_type', ev.get('receptor', 1)))
                weight = ev.get('weight', 0.0)
            else:
                receptor, weight = ev
                receptor = int(receptor)

            if receptor <= 0 or receptor > self.n_receptors:
                raise ValueError(f'Receptor type {receptor} out of range [1, {self.n_receptors}].')

            w_np = np.asarray(u.math.asarray(weight / u.nS), dtype=dftype)
            if cond_any(w_np < 0.0):
                raise ValueError('Synaptic weights for conductance-based multisynapse models must be non-negative.')
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, w, and per-receptor dg_0..dg_{n-1}, g_0..g_{n-1}.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_peak_detect, tau_syn, E_rev.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, u.math.minimum(state.V, self.V_peak))

        # Synaptic current: sum over all receptors g_k * (E_rev_k - V)
        i_syn = 0.0 * u.pA
        for k in range(self.n_receptors):
            g_k = state[f'g_{k}']
            E_rev_k = extra.E_rev[k] * u.mV
            i_syn = i_syn + g_k * (E_rev_k - v_eff)

        delta_t_safe = u.math.where(self.Delta_T == 0.0 * u.mV, 1.0 * u.mV, self.Delta_T)
        exp_arg = u.math.clip((v_eff - self.V_th) / delta_t_safe, -500.0, 500.0)
        i_spike = self.g_L * self.Delta_T * u.math.exp(exp_arg)

        dV_raw = (
                     -self.g_L * (v_eff - self.E_L) + i_spike
                     + i_syn - state.w + self.I_e + extra.i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        dw = (self.a * (v_eff - self.E_L) - state.w) / self.tau_w

        result = DotDict(V=dV, w=dw)
        for k in range(self.n_receptors):
            tau_k = extra.tau_syn[k] * u.ms
            dg_k = state[f'dg_{k}']
            g_k = state[f'g_{k}']
            result[f'dg_{k}'] = -dg_k / tau_k
            result[f'g_{k}'] = dg_k - g_k / tau_k

        return result

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, w, and per-receptor dg_k, g_k.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_peak_detect, tau_syn, E_rev.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & ((state.V < -1e3 * u.mV) | (state.w < -1e6 * u.pA) | (state.w > 1e6 * u.pA))
        )

        refr_accept = accept & (extra.r > 0)
        new_V = u.math.where(refr_accept, self.V_reset, state.V)

        spike_now = accept & (extra.r <= 0) & (new_V >= extra.v_peak_detect)
        spike_mask = extra.spike_mask | spike_now
        new_V = u.math.where(spike_now, self.V_reset, new_V)
        new_w = u.math.where(spike_now, state.w + self.b, state.w)
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict({**state, 'V': new_V, 'w': new_w})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

    def update(self, x=0.0 * u.pA, spike_events=None):
        r"""Advance the neuron by one simulation step.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous external current input in pA, broadcastable to
            ``self.varshape``. This value is stored into ``I_stim`` and applied
            at the next simulation step (one-step delay).
        spike_events : Iterable, optional
            Receptor-specific spike events. Each element should be a dict with
            keys ``receptor_type`` (or ``receptor``) and ``weight``, or a
            ``(receptor_type, weight)`` tuple.

        Returns
        -------
        jax.Array
            Binary spike tensor with dtype ``jnp.float64`` and shape
            ``self.V.value.shape``. A value of ``1.0`` indicates at least one
            internal spike event occurred during the integrated interval
            :math:`(t, t+dt]`.

        Raises
        ------
        ValueError
            If RKF45 integration enters a guarded unstable regime
            (``V < -1e3 mV`` or ``|w| > 1e6 pA``), indicating divergent
            dynamics for the current parameter/input regime.

        Notes
        -----
        Integration is performed with an adaptive vectorized RKF45 loop,
        including in-loop spike/reset/adaptation events and optional
        multiple spikes per step. All arithmetic is unit-aware via
        ``brainunit.math``.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        n_receptors = self.n_receptors

        # Read state variables with their natural units.
        V = self.V.value  # mV
        w = self.w.value  # pA
        dg = self.dg.value  # nS/ms, shape varshape + (n_receptors,)
        g = self.g.value  # nS, shape varshape + (n_receptors,)
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Spike detection threshold: V_peak if Delta_T > 0, else V_th.
        v_peak_detect = u.math.where(self.Delta_T > 0.0 * u.mV, self.V_peak, self.V_th)

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Build ODE state DotDict with per-receptor dg_k and g_k.
        ode_state = DotDict(V=V, w=w)
        for k in range(n_receptors):
            ode_state[f'dg_{k}'] = dg[..., k] * (u.nS / u.ms)  # dg stored unitless, restore nS/ms
            ode_state[f'g_{k}'] = g[..., k]  # g stored with nS units

        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            v_peak_detect=v_peak_detect,
            tau_syn=self.tau_syn,
            E_rev=self.E_rev,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V = ode_state.V
        w = ode_state.w
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Extract per-receptor states back.
        dg_list = []
        g_list = []
        for k in range(n_receptors):
            dg_list.append(ode_state[f'dg_{k}'])
            g_list.append(ode_state[f'g_{k}'])

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in aeif_cond_alpha_multisynapse dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Parse spike events and apply default delta inputs.
        v_shape = self.V.value.shape
        w_by_rec = self._parse_spike_events(spike_events, v_shape)
        w_default_raw = self.sum_delta_inputs(0.0 * u.nS)
        w_default_val = w_default_raw / u.nS
        if isinstance(w_default_val, u.Quantity):
            w_default_val = u.get_mantissa(w_default_val)
        w_default = np.asarray(w_default_val, dtype=dftype)
        w_default = np.broadcast_to(w_default, v_shape)
        if n_receptors > 0:
            if cond_any(w_default < 0.0):
                raise ValueError('Synaptic weights for conductance-based multisynapse models must be non-negative.')
            w_by_rec[..., 0] += w_default
        elif cond_any(w_default != 0.0):
            raise ValueError('No receptor ports available for incoming spike conductance.')

        # Apply synaptic spike inputs to dg per receptor.
        pscon = np.e / self.tau_syn  # shape (n_receptors,), unitless 1/ms
        for k in range(n_receptors):
            dg_list[k] = dg_list[k] + (pscon[k] / u.ms) * (w_by_rec[..., k] * u.nS)  # nS/ms

        # Write back state.
        self.V.value = V
        self.w.value = w

        # Stack dg and g back to arrays with shape varshape + (n_receptors,).
        if n_receptors > 0:
            dg_stacked = u.math.stack(dg_list, axis=-1)
            g_stacked = u.math.stack(g_list, axis=-1)
            self.dg.value = u.get_mantissa(dg_stacked)  # stored unitless (nS/ms mantissa)
            self.g.value = g_stacked
        else:
            self.dg.value = u.math.zeros(self.varshape + (0,), dtype=dftype)
            self.g.value = u.math.zeros(self.varshape + (0,), dtype=dftype) * u.nS

        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
