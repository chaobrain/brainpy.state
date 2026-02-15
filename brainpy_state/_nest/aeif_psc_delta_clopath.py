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

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'aeif_psc_delta_clopath',
]


class aeif_psc_delta_clopath(Neuron):
    r"""NEST-compatible ``aeif_psc_delta_clopath`` neuron model.

    Short description
    -----------------

    Adaptive exponential integrate-and-fire neuron with delta-shaped
    synaptic input and Clopath low-pass voltage traces.

    Description
    -----------

    ``aeif_psc_delta_clopath`` follows NEST
    ``models/aeif_psc_delta_clopath.{h,cpp}`` and extends
    ``aeif_psc_delta`` with:

    - spike afterpotential current ``z``,
    - adaptive threshold ``V_th`` with post-spike jump,
    - post-spike voltage clamping (``t_clamp``, ``V_clamp``),
    - Clopath trace variables ``u_bar_plus``, ``u_bar_minus``,
      ``u_bar_bar``.

    The model is intended for compatibility with voltage-based Clopath
    plasticity workflows.

    Membrane and adaptation dynamics
    .................................

    Let :math:`V` be membrane voltage, :math:`w` adaptation current,
    :math:`z` depolarizing spike afterpotential current,
    :math:`V_{th}` adaptive threshold.

    .. math::

       C_m \frac{dV}{dt}
       =
       -g_L (V - E_L)
       + g_L \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)
       - w + z + I_e + I_{stim}.

    .. math::

       \tau_w \frac{dw}{dt} = a (V - E_L) - w,
       \qquad
       \tau_z \frac{dz}{dt} = -z,
       \qquad
       \tau_{V_{th}} \frac{dV_{th}}{dt} = -(V_{th} - V_{th,rest}).

    Clopath low-pass states
    ........................

    .. math::

       \tau_{u+} \frac{du_{bar+}}{dt} = -u_{bar+} + V,

    .. math::

       \tau_{u-} \frac{du_{bar-}}{dt} = -u_{bar-} + V,

    .. math::

       \tau_{u\bar{}} \frac{du_{bar\bar{}}}{dt} = -u_{bar\bar{}} + u_{bar-}.

    Incoming delta spikes are voltage jumps (mV):

    .. math::

       V \leftarrow V + J \sum_k \delta(t - t_k).

    Refractory and clamping semantics (NEST order)
    ...............................................

    During integration, effective membrane voltage is:

    - ``V_clamp`` while ``clamp_step_count > 0``,
    - ``V_reset`` while ``refractory_step_count > 0``,
    - otherwise ``min(V, V_peak)``.

    Spike handling is performed inside every accepted RKF45 substep:

    1. Apply current-step delta jump only if neither refractory nor clamped.
    2. Detect threshold crossing:
       - if ``Delta_T > 0``, threshold is ``V_peak``;
       - if ``Delta_T == 0``, threshold is dynamic ``V_th``.
    3. On spike:
       - set ``V <- V_clamp``,
       - ``w <- w + b``,
       - ``z <- I_sp``,
       - ``V_th <- V_th_max``,
       - initialize clamping counter ``clamp_counts + 1``.
    4. When clamping counter reaches 1 inside substep loop:
       - set ``V <- V_reset``,
       - clear clamp counter,
       - initialize refractory counter ``refractory_counts + 1``.

    After finishing full ``dt`` integration:

    1. write Clopath delayed-buffer bookkeeping,
    2. decrement clamp counter,
    3. decrement refractory counter,
    4. store one-step delayed current ``I_stim <- x``.

    This ordering reproduces NEST implementation details, including
    in-loop multiple spikes when parameters allow them.

    Parameters
    ----------

    ==================== ================== ====================================== ================================================================
    **Parameter**        **Default**        **Math equivalent**                    **Description**
    ==================== ================== ====================================== ================================================================
    ``in_size``          (required)                                                Population shape
    ``V_peak``           33 mV              :math:`V_\mathrm{peak}`               Spike detection threshold for ``Delta_T > 0``
    ``V_reset``          -60 mV             :math:`V_\mathrm{reset}`              Reset potential
    ``t_ref``            0 ms               :math:`t_\mathrm{ref}`                Absolute refractory duration
    ``g_L``              30 nS              :math:`g_\mathrm{L}`                  Leak conductance
    ``C_m``              281 pF             :math:`C_\mathrm{m}`                  Membrane capacitance
    ``E_L``              -70.6 mV           :math:`E_\mathrm{L}`                  Leak reversal potential
    ``Delta_T``          2 mV               :math:`\Delta_T`                      Exponential slope factor
    ``tau_w``            144 ms             :math:`\tau_w`                        Adaptation time constant
    ``tau_z``            40 ms              :math:`\tau_z`                        Spike afterpotential time constant
    ``tau_V_th``         50 ms              :math:`\tau_{V_{th}}`                 Adaptive threshold time constant
    ``V_th_max``         30.4 mV            :math:`V_{th,max}`                     Threshold value immediately after spike
    ``V_th_rest``        -50.4 mV           :math:`V_{th,rest}`                    Resting threshold value
    ``tau_u_bar_plus``   7 ms               :math:`\tau_{u+}`                     Time constant of ``u_bar_plus``
    ``tau_u_bar_minus``  10 ms              :math:`\tau_{u-}`                     Time constant of ``u_bar_minus``
    ``tau_u_bar_bar``    500 ms             :math:`\tau_{u\bar{}}`               Time constant of ``u_bar_bar``
    ``a``                4 nS               :math:`a`                              Subthreshold adaptation strength
    ``b``                80.5 pA            :math:`b`                              Spike-triggered adaptation increment
    ``I_sp``             400 pA             :math:`I_{sp}`                         Spike afterpotential current reset value
    ``I_e``              0 pA               :math:`I_\mathrm{e}`                  Constant external current
    ``A_LTD``            1.4e-4             :math:`A_\mathrm{LTD}`                Clopath depression amplitude
    ``A_LTP``            8.0e-5             :math:`A_\mathrm{LTP}`                Clopath facilitation amplitude
    ``theta_plus``       -45.3 mV           :math:`\theta_+`                      Clopath potentiation threshold
    ``theta_minus``      -70.6 mV           :math:`\theta_-`                      Clopath depression threshold
    ``A_LTD_const``      ``True``                                                   If False, LTD scales with ``u_bar_bar**2 / u_ref_squared``
    ``delay_u_bars``     5 ms               (delay)                                Delay used by Clopath u-bar buffer bookkeeping
    ``u_ref_squared``    60                 :math:`u_\mathrm{ref}^2`              Clopath LTD homeostatic reference
    ``gsl_error_tol``    1e-6               (solver tolerance)                     RKF45 local error tolerance
    ``t_clamp``          2 ms               :math:`t_\mathrm{clamp}`              Spike clamping duration
    ``V_clamp``          33 mV              :math:`V_\mathrm{clamp}`              Clamped voltage after spike
    ``V_initializer``    Constant(E_L)                                              Membrane initializer
    ``w_initializer``    Constant(0 pA)                                             Adaptation initializer
    ``z_initializer``    Constant(0 pA)                                             Spike-current initializer
    ``V_th_initializer`` Constant(-50.4 mV)                                         Adaptive-threshold initializer
    ``u_bar_plus_initializer``  Constant(-70.6 mV)                                  ``u_bar_plus`` initializer
    ``u_bar_minus_initializer`` Constant(-70.6 mV)                                  ``u_bar_minus`` initializer
    ``u_bar_bar_initializer``   Constant(-70.6 mV)                                  ``u_bar_bar`` initializer
    ``spk_fun``          ReluGrad()                                                 Surrogate spike function
    ``spk_reset``        ``'hard'``                                                 Reset mode; hard reset matches NEST behavior
    ``ref_var``          ``False``                                                  If True, expose refractory/clamped indicator
    ==================== ================== ====================================== ================================================================

    State variables
    ---------------

    - ``V``: membrane potential :math:`V_m`.
    - ``w``: adaptation current.
    - ``z``: spike afterpotential current.
    - ``V_th``: adaptive threshold.
    - ``u_bar_plus``, ``u_bar_minus``, ``u_bar_bar``: Clopath low-pass traces.
    - ``refractory_step_count``: remaining refractory grid steps.
    - ``clamp_step_count``: remaining clamp grid steps.
    - ``integration_step``: persistent RKF45 internal step size.
    - ``I_stim``: one-step delayed current buffer.
    - ``delayed_u_bar_plus_buffer`` / ``delayed_u_bar_minus_buffer``:
      delay buffers used in Clopath bookkeeping.
    - ``delayed_u_bars_idx``: delay-buffer pointer index.
    - ``last_spike_time``: last emitted spike time (:math:`t+dt` on spike).

    Notes
    -----

    - Default ``t_ref=0`` matches NEST model defaults.
    - This implementation keeps Clopath delayed u-bar bookkeeping state,
      matching NEST update ordering even without a dedicated Clopath synapse
      implementation in this repository.

    References
    ----------
    .. [1] Clopath C, Busing L, Vasilaki E, Gerstner W (2010).
           Connectivity reflects coding: a model of voltage-based STDP with
           homeostasis. Nature Neuroscience, 13(3):344-352.
           DOI: https://doi.org/10.1038/nn.2479
    .. [2] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
           model as an effective description of neuronal activity.
           Journal of Neurophysiology, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [3] NEST source: ``models/aeif_psc_delta_clopath.h`` and
           ``models/aeif_psc_delta_clopath.cpp``.
    """

    __module__ = 'brainpy.state'

    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        V_peak: ArrayLike = 33.0 * u.mV,
        V_reset: ArrayLike = -60.0 * u.mV,
        t_ref: ArrayLike = 0.0 * u.ms,
        g_L: ArrayLike = 30.0 * u.nS,
        C_m: ArrayLike = 281.0 * u.pF,
        E_L: ArrayLike = -70.6 * u.mV,
        Delta_T: ArrayLike = 2.0 * u.mV,
        tau_w: ArrayLike = 144.0 * u.ms,
        tau_z: ArrayLike = 40.0 * u.ms,
        tau_V_th: ArrayLike = 50.0 * u.ms,
        V_th_max: ArrayLike = 30.4 * u.mV,
        V_th_rest: ArrayLike = -50.4 * u.mV,
        tau_u_bar_plus: ArrayLike = 7.0 * u.ms,
        tau_u_bar_minus: ArrayLike = 10.0 * u.ms,
        tau_u_bar_bar: ArrayLike = 500.0 * u.ms,
        a: ArrayLike = 4.0 * u.nS,
        b: ArrayLike = 80.5 * u.pA,
        I_sp: ArrayLike = 400.0 * u.pA,
        I_e: ArrayLike = 0.0 * u.pA,
        A_LTD: ArrayLike = 14.0e-5,
        A_LTP: ArrayLike = 8.0e-5,
        theta_plus: ArrayLike = -45.3 * u.mV,
        theta_minus: ArrayLike = -70.6 * u.mV,
        A_LTD_const: bool = True,
        delay_u_bars: ArrayLike = 5.0 * u.ms,
        u_ref_squared: ArrayLike = 60.0,
        gsl_error_tol: ArrayLike = 1e-6,
        t_clamp: ArrayLike = 2.0 * u.ms,
        V_clamp: ArrayLike = 33.0 * u.mV,
        V_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        w_initializer: Callable = braintools.init.Constant(0.0 * u.pA),
        z_initializer: Callable = braintools.init.Constant(0.0 * u.pA),
        V_th_initializer: Callable = braintools.init.Constant(-50.4 * u.mV),
        u_bar_plus_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        u_bar_minus_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        u_bar_bar_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
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
        self.tau_z = braintools.init.param(tau_z, self.varshape)
        self.tau_V_th = braintools.init.param(tau_V_th, self.varshape)
        self.V_th_max = braintools.init.param(V_th_max, self.varshape)
        self.V_th_rest = braintools.init.param(V_th_rest, self.varshape)
        self.tau_u_bar_plus = braintools.init.param(tau_u_bar_plus, self.varshape)
        self.tau_u_bar_minus = braintools.init.param(tau_u_bar_minus, self.varshape)
        self.tau_u_bar_bar = braintools.init.param(tau_u_bar_bar, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.I_sp = braintools.init.param(I_sp, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        # Clopath-related parameters kept for source-level compatibility.
        self.A_LTD = braintools.init.param(A_LTD, self.varshape)
        self.A_LTP = braintools.init.param(A_LTP, self.varshape)
        self.theta_plus = braintools.init.param(theta_plus, self.varshape)
        self.theta_minus = braintools.init.param(theta_minus, self.varshape)
        self.A_LTD_const = bool(A_LTD_const)
        self.delay_u_bars = braintools.init.param(delay_u_bars, self.varshape)
        self.u_ref_squared = braintools.init.param(u_ref_squared, self.varshape)

        self.gsl_error_tol = braintools.init.param(gsl_error_tol, self.varshape)
        self.t_clamp = braintools.init.param(t_clamp, self.varshape)
        self.V_clamp = braintools.init.param(V_clamp, self.varshape)

        self.V_initializer = V_initializer
        self.w_initializer = w_initializer
        self.z_initializer = z_initializer
        self.V_th_initializer = V_th_initializer
        self.u_bar_plus_initializer = u_bar_plus_initializer
        self.u_bar_minus_initializer = u_bar_minus_initializer
        self.u_bar_bar_initializer = u_bar_bar_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _to_numpy_unitless(x):
        return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _to_numpy_time_ms(x):
        try:
            return np.asarray(u.math.asarray(x / u.ms), dtype=np.float64)
        except Exception:
            return np.asarray(u.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        v_reset = self._to_numpy(self.V_reset, u.mV)
        v_peak = self._to_numpy(self.V_peak, u.mV)
        v_th_rest = self._to_numpy(self.V_th_rest, u.mV)
        v_th_max = self._to_numpy(self.V_th_max, u.mV)
        delta_t = self._to_numpy(self.Delta_T, u.mV)

        if np.any(v_reset >= v_peak):
            raise ValueError('Ensure that V_reset < V_peak .')
        if np.any(delta_t < 0.0):
            raise ValueError('Delta_T must be greater than or equal to zero.')
        if np.any(v_th_max < v_th_rest):
            raise ValueError('V_th_max >= V_th_rest required.')
        if np.any(v_peak < v_th_rest):
            raise ValueError('V_peak >= V_th_rest required.')

        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Ensure that C_m > 0')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.t_clamp, u.ms) < 0.0):
            raise ValueError('Ensure that t_clamp >= 0')

        if np.any(self._to_numpy(self.tau_w, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_z, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_V_th, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_u_bar_plus, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_u_bar_minus, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_u_bar_bar, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')

        if np.any(self._to_numpy_unitless(self.u_ref_squared) <= 0.0):
            raise ValueError('Ensure that u_ref_squared > 0')
        if np.any(self._to_numpy_unitless(self.gsl_error_tol) <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        # Mirror NEST overflow guard for exponential term at spike time.
        positive_dt = delta_t > 0.0
        if np.any(positive_dt):
            max_exp_arg = np.log(np.finfo(np.float64).max / 1e20)
            ratio = (v_peak - v_th_rest) / np.where(positive_dt, delta_t, 1.0)
            if np.any(ratio[positive_dt] >= max_exp_arg):
                raise ValueError(
                    'The current combination of V_peak, V_th_rest and Delta_T will lead to numerical overflow at '
                    'spike time; try for instance to increase Delta_T or to reduce V_peak to avoid this problem.'
                )

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def _delay_u_bars_steps(self, dt_q):
        dt_ms = float(u.math.asarray(dt_q / u.ms))
        delay_ms = self._to_numpy_time_ms(self.delay_u_bars)
        delay_steps = np.asarray(np.rint(delay_ms / dt_ms), dtype=np.int64) + 1

        if np.any(delay_steps < 1):
            raise ValueError('delay_u_bars must map to at least one delay-buffer entry.')
        if np.any(delay_steps != delay_steps.flat[0]):
            raise ValueError(
                'delay_u_bars must map to a uniform number of delay steps across the neuron state shape.'
            )
        return int(delay_steps.flat[0])

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _clamp_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_clamp / dt), dtype=jnp.int32)

    def _allocate_clopath_delay_buffers(self, state_shape, dt_q):
        delay_steps = self._delay_u_bars_steps(dt_q)
        self.delayed_u_bars_steps = brainstate.ShortTermState(np.asarray(delay_steps, dtype=np.int32))
        self.delayed_u_bars_idx = brainstate.ShortTermState(np.asarray(0, dtype=np.int32))

        buf_shape = (delay_steps,) + tuple(state_shape)
        self.delayed_u_bar_plus_buffer = brainstate.ShortTermState(np.zeros(buf_shape, dtype=np.float64))
        self.delayed_u_bar_minus_buffer = brainstate.ShortTermState(np.zeros(buf_shape, dtype=np.float64))

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        w = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        z = braintools.init.param(self.z_initializer, self.varshape, batch_size)

        v_th = braintools.init.param(self.V_th_initializer, self.varshape, batch_size)
        u_plus = braintools.init.param(self.u_bar_plus_initializer, self.varshape, batch_size)
        u_minus = braintools.init.param(self.u_bar_minus_initializer, self.varshape, batch_size)
        u_bar = braintools.init.param(self.u_bar_bar_initializer, self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.w = brainstate.HiddenState(w)
        self.z = brainstate.HiddenState(z)
        self.V_th = brainstate.HiddenState(v_th)
        self.u_bar_plus = brainstate.HiddenState(u_plus)
        self.u_bar_minus = brainstate.HiddenState(u_minus)
        self.u_bar_bar = brainstate.HiddenState(u_bar)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.clamp_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = self._safe_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        v_shape = np.asarray(u.math.asarray(V / u.mV), dtype=np.float64).shape
        self._allocate_clopath_delay_buffers(v_shape, dt)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        self.z.value = braintools.init.param(self.z_initializer, self.varshape, batch_size)
        self.V_th.value = braintools.init.param(self.V_th_initializer, self.varshape, batch_size)
        self.u_bar_plus.value = braintools.init.param(self.u_bar_plus_initializer, self.varshape, batch_size)
        self.u_bar_minus.value = braintools.init.param(self.u_bar_minus_initializer, self.varshape, batch_size)
        self.u_bar_bar.value = braintools.init.param(self.u_bar_bar_initializer, self.varshape, batch_size)

        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )

        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        self.clamp_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)

        dt = self._safe_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

        v_shape = np.asarray(u.math.asarray(self.V.value / u.mV), dtype=np.float64).shape
        self._allocate_clopath_delay_buffers(v_shape, dt)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        if hasattr(self, 'V_th'):
            v_th = self.V_th.value
        else:
            v_th = self.V_th_rest
        v_scaled = (V - v_th) / (v_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _sum_delta_inputs(self):
        delta_v = u.math.zeros_like(self.V.value)
        if self.delta_inputs is None:
            return delta_v

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)
            delta_v = delta_v + out

        return delta_v

    @staticmethod
    def _dynamics_scalar(v, w, z, v_th, u_plus, u_minus, u_bar, is_refractory, is_clamped, i_stim, p):
        if is_refractory or is_clamped:
            v_eff = p['V_clamp'] if is_clamped else p['V_reset']
        else:
            v_eff = min(v, p['V_peak_rhs'])

        i_spike = 0.0 if p['Delta_T'] == 0.0 else (
            p['g_L'] * p['Delta_T'] * math.exp((v_eff - v_th) / p['Delta_T'])
        )

        dv = 0.0 if (is_refractory or is_clamped) else (
            -p['g_L'] * (v_eff - p['E_L']) + i_spike - w + z + p['I_e'] + i_stim
        ) / p['C_m']

        # NEST sets dw/dt = 0 while clamped, but not during pure refractory.
        dw = 0.0 if is_clamped else (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']
        dz = -z / p['tau_z']
        dv_th = -(v_th - p['V_th_rest']) / p['tau_V_th']

        du_plus = (-u_plus + v_eff) / p['tau_u_bar_plus']
        du_minus = (-u_minus + v_eff) / p['tau_u_bar_minus']
        du_bar = (-u_bar + u_minus) / p['tau_u_bar_bar']

        return dv, dw, dz, dv_th, du_plus, du_minus, du_bar

    def _write_clopath_history(self, V_m, u_plus, u_minus, u_bar, p):
        plus_buf = np.asarray(self.delayed_u_bar_plus_buffer.value, dtype=np.float64)
        minus_buf = np.asarray(self.delayed_u_bar_minus_buffer.value, dtype=np.float64)

        delay_steps = int(np.asarray(self.delayed_u_bars_steps.value, dtype=np.int32))
        idx = int(np.asarray(self.delayed_u_bars_idx.value, dtype=np.int32))

        plus_buf[idx] = u_plus
        minus_buf[idx] = u_minus

        idx = (idx + 1) % delay_steps

        del_u_plus = plus_buf[idx]
        del_u_minus = minus_buf[idx]

        # Keep same delayed-buffer and threshold gating behavior as NEST.
        # The resulting dw traces are used by Clopath synapses in NEST.
        if self.A_LTD_const:
            _ = np.where(
                del_u_minus > p['theta_minus'],
                p['A_LTD'] * (del_u_minus - p['theta_minus']),
                0.0,
            )
        else:
            _ = np.where(
                del_u_minus > p['theta_minus'],
                p['A_LTD'] * (u_bar * u_bar) * (del_u_minus - p['theta_minus']) / p['u_ref_squared'],
                0.0,
            )

        _ = np.where(
            (V_m > p['theta_plus']) & (del_u_plus > p['theta_minus']),
            p['A_LTP'] * (V_m - p['theta_plus']) * (del_u_plus - p['theta_minus']) * p['dt'],
            0.0,
        )

        self.delayed_u_bar_plus_buffer.value = plus_buf
        self.delayed_u_bar_minus_buffer.value = minus_buf
        self.delayed_u_bars_idx.value = np.asarray(idx, dtype=np.int32)

    def update(self, x=0.0 * u.pA):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        w = self._broadcast_to_state(self._to_numpy(self.w.value, u.pA), v_shape)
        z = self._broadcast_to_state(self._to_numpy(self.z.value, u.pA), v_shape)
        v_th = self._broadcast_to_state(self._to_numpy(self.V_th.value, u.mV), v_shape)
        u_plus = self._broadcast_to_state(self._to_numpy(self.u_bar_plus.value, u.mV), v_shape)
        u_minus = self._broadcast_to_state(self._to_numpy(self.u_bar_minus.value, u.mV), v_shape)
        u_bar = self._broadcast_to_state(self._to_numpy(self.u_bar_bar.value, u.mV), v_shape)

        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            v_shape,
        )
        clamp_r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.clamp_step_count.value), dtype=np.int32),
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
            'tau_z': self._broadcast_to_state(self._to_numpy(self.tau_z, u.ms), v_shape),
            'tau_V_th': self._broadcast_to_state(self._to_numpy(self.tau_V_th, u.ms), v_shape),
            'V_th_max': self._broadcast_to_state(self._to_numpy(self.V_th_max, u.mV), v_shape),
            'V_th_rest': self._broadcast_to_state(self._to_numpy(self.V_th_rest, u.mV), v_shape),
            'tau_u_bar_plus': self._broadcast_to_state(self._to_numpy(self.tau_u_bar_plus, u.ms), v_shape),
            'tau_u_bar_minus': self._broadcast_to_state(self._to_numpy(self.tau_u_bar_minus, u.ms), v_shape),
            'tau_u_bar_bar': self._broadcast_to_state(self._to_numpy(self.tau_u_bar_bar, u.ms), v_shape),
            'a': self._broadcast_to_state(self._to_numpy(self.a, u.nS), v_shape),
            'b': self._broadcast_to_state(self._to_numpy(self.b, u.pA), v_shape),
            'I_sp': self._broadcast_to_state(self._to_numpy(self.I_sp, u.pA), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
            'V_clamp': self._broadcast_to_state(self._to_numpy(self.V_clamp, u.mV), v_shape),
            'theta_plus': self._broadcast_to_state(self._to_numpy(self.theta_plus, u.mV), v_shape),
            'theta_minus': self._broadcast_to_state(self._to_numpy(self.theta_minus, u.mV), v_shape),
            'A_LTD': self._broadcast_to_state(self._to_numpy_unitless(self.A_LTD), v_shape),
            'A_LTP': self._broadcast_to_state(self._to_numpy_unitless(self.A_LTP), v_shape),
            'u_ref_squared': self._broadcast_to_state(self._to_numpy_unitless(self.u_ref_squared), v_shape),
            'atol': self._broadcast_to_state(self._to_numpy_unitless(self.gsl_error_tol), v_shape),
            'dt': self._broadcast_to_state(np.asarray(dt, dtype=np.float64), v_shape),
        }

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )
        clamp_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._clamp_counts()), dtype=np.int32),
            v_shape,
        )

        delta_v_q = self._sum_delta_inputs()
        delta_v = self._broadcast_to_state(self._to_numpy(delta_v_q, u.mV), v_shape)

        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        spike_mask = np.zeros(v_shape, dtype=bool)

        V_next = np.empty_like(V)
        w_next = np.empty_like(w)
        z_next = np.empty_like(z)
        v_th_next = np.empty_like(v_th)
        u_plus_next = np.empty_like(u_plus)
        u_minus_next = np.empty_like(u_minus)
        u_bar_next = np.empty_like(u_bar)

        r_next = np.empty_like(r)
        clamp_r_next = np.empty_like(clamp_r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            y = np.asarray([
                V[idx], w[idx], z[idx], v_th[idx], u_plus[idx], u_minus[idx], u_bar[idx]
            ], dtype=np.float64)

            r_i = int(r[idx])
            clamp_i = int(clamp_r[idx])
            h_i = float(max(h_int[idx], self._MIN_H))
            t_local = 0.0
            iters = 0
            local_spike = False

            pending_delta = float(delta_v[idx])

            while t_local < dt and iters < self._MAX_ITERS:
                iters += 1
                h_i = max(self._MIN_H, min(h_i, dt - t_local))

                is_refractory = r_i > 0
                is_clamped = clamp_i > 0

                def f(y_):
                    return np.asarray(
                        self._dynamics_scalar(
                            y_[0],
                            y_[1],
                            y_[2],
                            y_[3],
                            y_[4],
                            y_[5],
                            y_[6],
                            is_refractory,
                            is_clamped,
                            i_stim[idx],
                            local_p,
                        ),
                        dtype=np.float64,
                    )

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

                y4 = y + h_i * (
                    25.0 * k1 / 216.0
                    + 1408.0 * k3 / 2565.0
                    + 2197.0 * k4 / 4104.0
                    - k5 / 5.0
                )
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
                        raise ValueError('Numerical instability in aeif_psc_delta_clopath dynamics.')

                    if r_i == 0 and clamp_i == 0:
                        y[0] += pending_delta
                    pending_delta = 0.0

                    v_peak_detect_i = local_p['V_peak_rhs'] if local_p['Delta_T'] > 0.0 else y[3]

                    if y[0] >= v_peak_detect_i and clamp_i == 0:
                        local_spike = True

                        y[0] = local_p['V_clamp']
                        y[1] += local_p['b']
                        y[2] = local_p['I_sp']
                        y[3] = local_p['V_th_max']

                        clamp_i = int(clamp_counts[idx]) + 1 if int(clamp_counts[idx]) > 0 else 0
                    elif clamp_i == 1:
                        y[0] = local_p['V_reset']
                        clamp_i = 0
                        r_i = int(refr_counts[idx]) + 1 if int(refr_counts[idx]) > 0 else 0

                    if r_i > 0:
                        y[0] = local_p['V_reset']
                else:
                    fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
                    h_i = max(self._MIN_H, h_i * fac)

            # Decrement counters after full dt integration, matching NEST.
            if clamp_i > 0:
                clamp_i -= 1
            if r_i > 0:
                r_i -= 1

            spike_mask[idx] = local_spike

            V_next[idx] = y[0]
            w_next[idx] = y[1]
            z_next[idx] = y[2]
            v_th_next[idx] = y[3]
            u_plus_next[idx] = y[4]
            u_minus_next[idx] = y[5]
            u_bar_next[idx] = y[6]

            r_next[idx] = r_i
            clamp_r_next[idx] = clamp_i
            h_next[idx] = h_i

        self._write_clopath_history(
            V_next,
            u_plus_next,
            u_minus_next,
            u_bar_next,
            p,
        )

        self.V.value = V_next * u.mV
        self.w.value = w_next * u.pA
        self.z.value = z_next * u.pA
        self.V_th.value = v_th_next * u.mV
        self.u_bar_plus.value = u_plus_next * u.mV
        self.u_bar_minus.value = u_minus_next * u.mV
        self.u_bar_bar.value = u_bar_next * u.mV

        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.clamp_step_count.value = jnp.asarray(clamp_r_next, dtype=jnp.int32)

        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA

        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(
                (self.refractory_step_count.value > 0) | (self.clamp_step_count.value > 0)
            )

        return u.math.asarray(spike_mask, dtype=jnp.float64)
