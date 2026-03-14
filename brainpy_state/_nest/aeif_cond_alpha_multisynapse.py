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

from ._utils import is_tracer, validate_aeif_overflow
from ._base import NESTNeuron

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
       >>> import saiunit as u
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

    _MIN_H = 1e-8  # ms
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
        self.gsl_error_tol = braintools.init.param(gsl_error_tol, self.varshape)

        dftype = brainstate.environ.dftype()
        self.tau_syn = np.asarray(u.math.asarray(tau_syn / u.ms), dtype=dftype).reshape(-1)
        self.E_rev = np.asarray(u.math.asarray(E_rev / u.mV), dtype=dftype).reshape(-1)

        self.V_initializer = V_initializer
        self.g_initializer = g_initializer
        self.w_initializer = w_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @property
    def n_receptors(self):
        return int(self.tau_syn.size)

    @property
    def recordables(self):
        return ['V_m', 'w', *[f'g_{i + 1}' for i in range(self.n_receptors)]]

    @staticmethod
    def _to_numpy(x, unit):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _to_numpy_unitless(x):
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @staticmethod
    def _broadcast_to_receptors(x_np: np.ndarray, shape, n_receptors: int):
        return np.broadcast_to(x_np, shape + (n_receptors,))

    def _validate_parameters(self):
        v_reset = self.V_reset
        v_peak = self.V_peak
        v_th = self.V_th
        delta_t = self.Delta_T / u.ms

        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (v_reset, v_peak, v_th, delta_t)):
            return

        if self.E_rev.size != self.tau_syn.size:
            raise ValueError('The E_rev and tau_syn arrays must have the same size.')
        if np.any(self.tau_syn <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive.')
        if np.any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if np.any(v_reset >= v_peak):
            raise ValueError('Ensure that: V_reset < V_peak .')
        if np.any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if np.any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self.tau_w <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        # Mirror NEST overflow guard for exponential term at spike time.
        validate_aeif_overflow(v_peak, v_th, delta_t)

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        w = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        g = braintools.init.param(self.g_initializer, self.varshape + (self.n_receptors,), batch_size)

        self.V = brainstate.HiddenState(V)
        self.w = brainstate.HiddenState(w)
        dftype = brainstate.environ.dftype()
        self.dg = brainstate.ShortTermState(np.zeros(g.shape, dtype=dftype))
        self.g = brainstate.HiddenState(g)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        ditype = brainstate.environ.ditype()
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=ditype))

        dt = brainstate.environ.get_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        self.g.value = braintools.init.param(self.g_initializer, self.varshape + (self.n_receptors,), batch_size)
        dftype = brainstate.environ.dftype()
        self.dg.value = np.zeros(np.asarray(self.g.value).shape, dtype=dftype)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        ditype = brainstate.environ.ditype()
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=ditype)
        dt = brainstate.environ.get_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )
        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        ditype = brainstate.environ.ditype()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

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
            if np.any(w_np < 0.0):
                raise ValueError('Synaptic weights for conductance-based multisynapse models must be non-negative.')
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    @staticmethod
    def _dynamics_scalar(y, is_refractory, i_stim, p):
        v_eff = p['V_reset'] if is_refractory else min(y[0], p['V_peak_rhs'])
        w = y[1]
        dg = y[2::2]
        g = y[3::2]

        i_syn = float(np.sum(g * (p['E_rev'] - v_eff)))
        i_spike = 0.0 if p['Delta_T'] == 0.0 else (
            p['Delta_T'] * p['g_L'] * math.exp((v_eff - p['V_th']) / p['Delta_T'])
        )
        dv = 0.0 if is_refractory else (
                                           -p['g_L'] * (v_eff - p['E_L']) + i_spike + i_syn - w + p['I_e'] + i_stim
                                       ) / p['C_m']
        dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']

        dy = np.empty_like(y)
        dy[0] = dv
        dy[1] = dw
        dy[2::2] = -dg / p['tau_syn']
        dy[3::2] = dg - g / p['tau_syn']
        return dy

    def update(self, x=0.0 * u.pA, spike_events=None):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape
        n_receptors = self.n_receptors

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        w = self._broadcast_to_state(self._to_numpy(self.w.value, u.pA), v_shape)
        dftype = brainstate.environ.dftype()
        dg = self._broadcast_to_receptors(np.asarray(self.dg.value, dtype=dftype), v_shape, n_receptors)
        g = self._broadcast_to_receptors(self._to_numpy(self.g.value, u.nS), v_shape, n_receptors)
        ditype = brainstate.environ.ditype()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=ditype),
            v_shape,
        )
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape)

        p = {
            'V_peak_rhs': self._broadcast_to_state(self._to_numpy(self.V_peak, u.mV), v_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'Delta_T': self._broadcast_to_state(self._to_numpy(self.Delta_T, u.mV), v_shape),
            'tau_w': self._broadcast_to_state(self._to_numpy(self.tau_w, u.ms), v_shape),
            'a': self._broadcast_to_state(self._to_numpy(self.a, u.nS), v_shape),
            'b': self._broadcast_to_state(self._to_numpy(self.b, u.pA), v_shape),
            'V_th': self._broadcast_to_state(self._to_numpy(self.V_th, u.mV), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
            'atol': self._broadcast_to_state(self._to_numpy_unitless(self.gsl_error_tol), v_shape),
            'tau_syn': self._broadcast_to_receptors(self.tau_syn, v_shape, n_receptors),
            'E_rev': self._broadcast_to_receptors(self.E_rev, v_shape, n_receptors),
        }

        v_peak_detect = np.where(p['Delta_T'] > 0.0, p['V_peak_rhs'], p['V_th'])
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=ditype),
            v_shape,
        )

        w_by_rec = self._parse_spike_events(spike_events, v_shape)
        w_default = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0.0 * u.nS), u.nS), v_shape)
        if n_receptors > 0:
            if np.any(w_default < 0.0):
                raise ValueError('Synaptic weights for conductance-based multisynapse models must be non-negative.')
            w_by_rec[..., 0] += w_default
        elif np.any(w_default != 0.0):
            raise ValueError('No receptor ports available for incoming spike conductance.')

        g0 = self._broadcast_to_receptors(np.e / self.tau_syn, v_shape, n_receptors)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        spike_mask = np.zeros(v_shape, dtype=bool)
        V_next = np.empty_like(V)
        w_next = np.empty_like(w)
        dg_next = np.empty_like(dg)
        g_next = np.empty_like(g)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            y = np.empty(2 + 2 * n_receptors, dtype=dftype)
            y[0] = V[idx]
            y[1] = w[idx]
            y[2::2] = dg[idx]
            y[3::2] = g[idx]

            r_i = int(r[idx])
            h_i = float(max(h_int[idx], self._MIN_H))
            t_local = 0.0
            iters = 0
            local_spike = False

            while t_local < dt and iters < self._MAX_ITERS:
                iters += 1
                h_i = max(self._MIN_H, min(h_i, dt - t_local))
                is_refractory = r_i > 0

                def f(y_):
                    dftype = brainstate.environ.dftype()
                    return np.asarray(self._dynamics_scalar(y_, is_refractory, i_stim[idx], local_p), dtype=dftype)

                k1 = f(y)
                k2 = f(y + h_i * (1.0 / 4.0) * k1)
                k3 = f(y + h_i * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
                k4 = f(y + h_i * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0))
                k5 = f(y + h_i * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0))
                k6 = f(
                    y
                    + h_i
                    * (
                        -8.0 * k1 / 27.0
                        + 2.0 * k2
                        - 3544.0 * k3 / 2565.0
                        + 1859.0 * k4 / 4104.0
                        - 11.0 * k5 / 40.0
                    )
                )

                y4 = y + h_i * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
                y5 = y + h_i * (
                    16.0 * k1 / 135.0
                    + 6656.0 * k3 / 12825.0
                    + 28561.0 * k4 / 56430.0
                    - 9.0 * k5 / 50.0
                    + 2.0 * k6 / 55.0
                )
                err = float(np.max(np.abs(y5 - y4)))
                atol = float(local_p['atol'])

                if err <= atol or h_i <= self._MIN_H:
                    y = y5
                    t_local += h_i
                    fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
                    h_i = max(self._MIN_H, h_i * fac)

                    if y[0] < -1e3 or y[1] < -1e6 or y[1] > 1e6:
                        raise ValueError('Numerical instability in aeif_cond_alpha_multisynapse dynamics.')

                    if r_i > 0:
                        y[0] = local_p['V_reset']
                    elif y[0] >= v_peak_detect[idx]:
                        local_spike = True
                        y[0] = local_p['V_reset']
                        y[1] += local_p['b']
                        r_i = int(refr_counts[idx]) + 1 if int(refr_counts[idx]) > 0 else 0
                else:
                    fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
                    h_i = max(self._MIN_H, h_i * fac)

            if r_i > 0:
                r_i -= 1

            y[2::2] += g0[idx] * w_by_rec[idx]

            spike_mask[idx] = local_spike
            V_next[idx] = y[0]
            w_next[idx] = y[1]
            dg_next[idx] = y[2::2]
            g_next[idx] = y[3::2]
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.w.value = w_next * u.pA
        self.dg.value = dg_next
        self.g.value = g_next * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=ditype)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
