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

__all__ = [
    'aeif_cond_alpha_multisynapse',
]


class aeif_cond_alpha_multisynapse(Neuron):
    r"""NEST-compatible ``aeif_cond_alpha_multisynapse`` neuron model.

    Short description
    -----------------

    Conductance-based adaptive exponential integrate-and-fire neuron with
    alpha-shaped synapses and an arbitrary number of receptor ports.

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

    Continuous dynamics
    ...................

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

    Spike and refractory semantics
    ..............................

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

    Update order per simulation step (NEST semantics)
    ..................................................

    1. Integrate ODEs on :math:`(t, t+dt]` using adaptive RKF45.
    2. Inside integration loop: refractory clamp and spike/reset/adaptation.
    3. Decrement refractory counter once.
    4. Apply incoming receptor-specific spike events to ``dg``.
    5. Store continuous current input ``x`` into one-step delayed ``I_stim``.

    Event semantics
    ...............

    ``spike_events`` passed to :meth:`update` must be an iterable of
    ``(receptor_type, weight)`` or dictionaries with keys
    ``receptor_type``/``receptor`` and ``weight``.

    - Receptor types are 1-based and must satisfy ``1 <= receptor_type <= n_receptors``.
    - Weights are conductances (nS) and must be non-negative, matching NEST
      conductance multisynapse constraints.
    - ``add_delta_input`` stream is mapped to receptor 1 by default; those
      values must also be non-negative.

    Parameters
    ----------

    ==================== =================== ============================================== =====================================================
    **Parameter**        **Default**         **Math equivalent**                             **Description**
    ==================== =================== ============================================== =====================================================
    ``in_size``          (required)                                                         Population shape
    ``V_peak``           0 mV                :math:`V_\mathrm{peak}`                         Spike detection threshold (if ``Delta_T > 0``)
    ``V_reset``          -60 mV              :math:`V_\mathrm{reset}`                        Reset potential
    ``t_ref``            0 ms                :math:`t_\mathrm{ref}`                          Absolute refractory duration
    ``g_L``              30 nS               :math:`g_\mathrm{L}`                            Leak conductance
    ``C_m``              281 pF              :math:`C_\mathrm{m}`                            Membrane capacitance
    ``E_L``              -70.6 mV            :math:`E_\mathrm{L}`                            Leak reversal potential
    ``Delta_T``          2 mV                :math:`\Delta_T`                                Exponential slope factor
    ``tau_w``            144 ms              :math:`\tau_w`                                  Adaptation time constant
    ``a``                4 nS                :math:`a`                                       Subthreshold adaptation
    ``b``                80.5 pA             :math:`b`                                       Spike-triggered adaptation increment
    ``V_th``             -50.4 mV            :math:`V_\mathrm{th}`                           Spike initiation threshold (exponential term)
    ``tau_syn``          ``[2.0]`` ms        :math:`\tau_{\mathrm{syn},k}`                   Synaptic time constants per receptor
    ``E_rev``            ``[0.0]`` mV        :math:`E_{\mathrm{rev},k}`                      Reversal potentials per receptor
    ``I_e``              0 pA                :math:`I_\mathrm{e}`                            Constant external current
    ``gsl_error_tol``    1e-6                (solver tolerance)                              RKF45 local error tolerance
    ``V_initializer``    Constant(-70.6 mV)                                                  Membrane initializer
    ``g_initializer``    Constant(0 nS)                                                      Conductance initializer ``[..., n_receptors]``
    ``w_initializer``    Constant(0 pA)                                                      Adaptation current initializer
    ``spk_fun``          ReluGrad()                                                          Surrogate spike function
    ``spk_reset``        ``'hard'``                                                          Reset mode; hard reset matches NEST behavior
    ``ref_var``          ``False``                                                           If True, expose boolean refractory indicator
    ==================== =================== ============================================== =====================================================

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

        self.tau_syn = np.asarray(u.math.asarray(tau_syn / u.ms), dtype=np.float64).reshape(-1)
        self.E_rev = np.asarray(u.math.asarray(E_rev / u.mV), dtype=np.float64).reshape(-1)

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
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _to_numpy_unitless(x):
        return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @staticmethod
    def _broadcast_to_receptors(x_np: np.ndarray, shape, n_receptors: int):
        return np.broadcast_to(x_np, shape + (n_receptors,))

    def _validate_parameters(self):
        v_reset = self._to_numpy(self.V_reset, u.mV)
        v_peak = self._to_numpy(self.V_peak, u.mV)
        v_th = self._to_numpy(self.V_th, u.mV)
        delta_t = self._to_numpy(self.Delta_T, u.mV)

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
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_w, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy_unitless(self.gsl_error_tol) <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        # Mirror NEST overflow guard for exponential term at spike time.
        positive_dt = delta_t > 0.0
        if np.any(positive_dt):
            max_exp_arg = np.log(np.finfo(np.float64).max / 1e20)
            ratio = (v_peak - v_th) / np.where(positive_dt, delta_t, 1.0)
            if np.any(ratio[positive_dt] >= max_exp_arg):
                raise ValueError(
                    'The current combination of V_peak, V_th and Delta_T will lead to numerical overflow at spike '
                    'time; try for instance to increase Delta_T or to reduce V_peak to avoid this problem.'
                )

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        w = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        g = braintools.init.param(self.g_initializer, self.varshape + (self.n_receptors,), batch_size)

        self.V = brainstate.HiddenState(V)
        self.w = brainstate.HiddenState(w)
        self.dg = brainstate.ShortTermState(np.zeros(g.shape, dtype=np.float64))
        self.g = brainstate.HiddenState(g)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = self._safe_dt()
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
        self.dg.value = np.zeros(np.asarray(self.g.value).shape, dtype=np.float64)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        dt = self._safe_dt()
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
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        out = np.zeros(v_shape + (self.n_receptors,), dtype=np.float64)
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

            w_np = np.asarray(u.math.asarray(weight / u.nS), dtype=np.float64)
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
        dg = self._broadcast_to_receptors(np.asarray(self.dg.value, dtype=np.float64), v_shape, n_receptors)
        g = self._broadcast_to_receptors(self._to_numpy(self.g.value, u.nS), v_shape, n_receptors)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
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
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
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
            y = np.empty(2 + 2 * n_receptors, dtype=np.float64)
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
                    return np.asarray(self._dynamics_scalar(y_, is_refractory, i_stim[idx], local_p), dtype=np.float64)

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
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=jnp.float64)
