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
    'iaf_psc_exp',
]


class iaf_psc_exp(Neuron):
    r"""NEST-compatible ``iaf_psc_exp`` neuron model.

    Description
    -----------

    ``iaf_psc_exp`` is a current-based leaky integrate-and-fire neuron with
    hard reset, fixed absolute refractory period, and exponential excitatory
    and inhibitory postsynaptic currents. The implementation follows NEST
    ``models/iaf_psc_exp.{h,cpp}`` update order, including one-step buffered
    current input and receptor-1 filtered current handling.

    **1. Continuous-time dynamics**

    The subthreshold membrane equation is

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m}
       + \frac{I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}} + I_e + I_0}{C_m}

    where ``I_0`` is the buffered current from the previous simulation step.
    Synaptic currents decay exponentially:

    .. math::

       \frac{dI_{\mathrm{syn,ex}}}{dt} = -\frac{I_{\mathrm{syn,ex}}}{\tau_{\mathrm{syn,ex}}}
       \qquad
       \frac{dI_{\mathrm{syn,in}}}{dt} = -\frac{I_{\mathrm{syn,in}}}{\tau_{\mathrm{syn,in}}}.

    NEST also defines a second current receptor ``I_1`` that is filtered
    through the excitatory kernel; this is exposed by ``update(x_filtered=...)``.

    **2. Exact step propagator and NEST update ordering**

    For time step :math:`h=dt` (in ms), this implementation uses exact
    exponentials for all linear sub-systems:

    .. math::

       P_{11,\mathrm{ex}} = e^{-h/\tau_{\mathrm{syn,ex}}}, \quad
       P_{11,\mathrm{in}} = e^{-h/\tau_{\mathrm{syn,in}}}, \quad
       P_{22} = e^{-h/\tau_m},

    .. math::

       P_{20} = \frac{\tau_m}{C_m}(1 - P_{22}),

    .. math::

       P_{21}(\tau_{\mathrm{syn}}) =
       \frac{\tau_{\mathrm{syn}}\tau_m}{C_m(\tau_m - \tau_{\mathrm{syn}})}
       \left(e^{-h/\tau_m} - e^{-h/\tau_{\mathrm{syn}}}\right),

    where ``P21`` is evaluated numerically stably by :meth:`_propagator_exp`.
    Let :math:`V_\mathrm{rel}=V_m-E_L`. The candidate membrane update is

    .. math::

       V_{\mathrm{rel},n+1} =
       P_{22}V_{\mathrm{rel},n}
       + P_{21,\mathrm{ex}} I_{\mathrm{syn,ex},n}
       + P_{21,\mathrm{in}} I_{\mathrm{syn,in},n}
       + P_{20}(I_e + I_{0,n}).

    Per-step update order is:

    1. Update membrane potential if not refractory.
    2. Decay synaptic currents.
    3. Add filtered-current contribution to excitatory synaptic current.
    4. Add arriving spikes (positive -> excitatory, negative -> inhibitory).
    5. Threshold test, reset and refractory assignment.
    6. Store buffered currents for next step.

    **3. Escape-noise threshold dynamics**

    Deterministic thresholding is used when :math:`\delta < 10^{-10}`:
    :math:`V_{\mathrm{rel}} \ge \theta`, where
    :math:`\theta = V_{th} - E_L`.

    For :math:`\delta > 0`, the model uses an exponential hazard:

    .. math::

       \phi(V) = \rho \exp\left(\frac{V_{\mathrm{rel}} - \theta}{\delta}\right),

    and spikes with step probability :math:`p=\phi(V)\,h\,10^{-3}` because
    :math:`\phi` is in ``1/s`` while ``h`` is in ms. In this implementation,
    stochastic decisions use ``numpy.random.random``.

    **4. Stability constraints and computational implications**

    - Construction enforces ``V_reset < V_th``, ``C_m > 0``, ``tau_m > 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, ``t_ref >= 0``, ``rho >= 0``,
      and ``delta >= 0``.
    - ``_propagator_exp`` uses a singular fallback
      :math:`(h/C_m)\exp(-h/\tau_m)` when ``tau_syn`` is numerically close to
      ``tau_m``, avoiding cancellation in
      :math:`(e^{-h/\tau_m} - e^{-h/\tau_{syn}})/(\tau_m-\tau_{syn})`.
    - Per-call cost is :math:`O(\prod \mathrm{varshape})` with vectorized
      NumPy operations in ``float64`` for coefficient evaluation.
    - Buffered current semantics match NEST ring-buffer timing:
      ``x``/``x_filtered`` supplied at step ``n`` are stored and consumed at
      step ``n+1``.

    Parameters
    ----------
    in_size : Size
        Population shape specification. All per-neuron parameters are
        broadcast to ``self.varshape``.
    E_L : ArrayLike, optional
        Resting potential :math:`E_L` in mV; scalar or array broadcastable to
        ``self.varshape``. Default is ``-70. * u.mV``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF; broadcastable and strictly
        positive. Default is ``250. * u.pF``.
    tau_m : ArrayLike, optional
        Membrane time constant :math:`\tau_m` in ms; broadcastable and
        strictly positive. Default is ``10. * u.ms``.
    t_ref : ArrayLike, optional
        Absolute refractory period :math:`t_{ref}` in ms; broadcastable and
        nonnegative. Converted to integer steps by ``ceil(t_ref / dt)``.
        Default is ``2. * u.ms``.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_{th}` in mV; broadcastable to
        ``self.varshape``. Default is ``-55. * u.mV``.
    V_reset : ArrayLike, optional
        Post-spike reset potential :math:`V_{reset}` in mV; broadcastable and
        must satisfy ``V_reset < V_th`` elementwise. Default is
        ``-70. * u.mV``.
    tau_syn_ex : ArrayLike, optional
        Excitatory synaptic decay constant :math:`\tau_{\mathrm{syn,ex}}` in
        ms; broadcastable and strictly positive. Default is ``2. * u.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory synaptic decay constant :math:`\tau_{\mathrm{syn,in}}` in
        ms; broadcastable and strictly positive. Default is ``2. * u.ms``.
    I_e : ArrayLike, optional
        Constant external injected current :math:`I_e` in pA; scalar or array
        broadcastable to ``self.varshape``. Default is ``0. * u.pA``.
    rho : ArrayLike, optional
        Escape-noise base firing intensity :math:`\rho` in ``1/s``;
        broadcastable and nonnegative. Used only in stochastic mode
        (``delta > 0``). Default is ``0.01 / u.second``.
    delta : ArrayLike, optional
        Escape-noise soft-threshold width :math:`\delta` in mV; broadcastable
        and nonnegative. ``delta == 0`` reproduces deterministic thresholding.
        Default is ``0. * u.mV``.
    V_initializer : Callable, optional
        Initializer for membrane state ``V`` used by :meth:`init_state`.
        Default is ``braintools.init.Constant(-70. * u.mV)``.
    spk_fun : Callable, optional
        Surrogate spike nonlinearity used by :meth:`get_spike`. Default is
        ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy inherited from :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` matches NEST reset behavior. Default is ``'hard'``.
    ref_var : bool, optional
        If ``True``, allocates ``self.refractory`` (boolean) for external
        inspection of refractory state. Default is ``False``.
    name : str or None, optional
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 16 28 14 16 36

       * - Parameter
         - Type / shape / unit
         - Default
         - Math symbol
         - Semantics
       * - ``in_size``
         - :class:`~brainstate.typing.Size`; scalar/tuple
         - required
         - --
         - Defines neuron population shape ``self.varshape``.
       * - ``E_L``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``-70. * u.mV``
         - :math:`E_L`
         - Resting membrane potential.
       * - ``C_m``
         - ArrayLike, broadcastable (pF), ``> 0``
         - ``250. * u.pF``
         - :math:`C_m`
         - Membrane capacitance in voltage integration.
       * - ``tau_m``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``10. * u.ms``
         - :math:`\tau_m`
         - Membrane leak time constant.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), ``>= 0``
         - ``2. * u.ms``
         - :math:`t_{ref}`
         - Absolute refractory duration.
       * - ``V_th`` and ``V_reset``
         - ArrayLike, broadcastable (mV), with ``V_reset < V_th``
         - ``-55. * u.mV``, ``-70. * u.mV``
         - :math:`V_{th}`, :math:`V_{reset}`
         - Threshold and post-spike reset voltages.
       * - ``tau_syn_ex`` and ``tau_syn_in``
         - ArrayLike, broadcastable (ms), each ``> 0``
         - ``2. * u.ms``
         - :math:`\tau_{\mathrm{syn,ex}}`, :math:`\tau_{\mathrm{syn,in}}`
         - Exponential PSC decay constants.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant current injected every step.
       * - ``rho`` and ``delta``
         - ArrayLike, broadcastable; ``rho`` in ``1/s`` and ``delta`` in mV,
           both ``>= 0``
         - ``0.01 / u.second``, ``0. * u.mV``
         - :math:`\rho`, :math:`\delta`
         - Escape-noise hazard parameters.
       * - ``V_initializer``
         - Callable
         - ``Constant(-70. * u.mV)``
         - --
         - Initializer for membrane state ``V``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate function used for output spikes.
       * - ``spk_reset``
         - ``str`` (typically ``'hard'``)
         - ``'hard'``
         - --
         - Reset behavior selection in base class.
       * - ``ref_var``
         - ``bool``
         - ``False``
         - --
         - Enables explicit boolean refractory state variable.
       * - ``name``
         - ``str`` or ``None``
         - ``None``
         - --
         - Optional instance name.

    Raises
    ------
    ValueError
        Raised at construction when any validated constraint is violated:
        ``V_reset >= V_th``, nonpositive ``C_m``/``tau_m``/synaptic time
        constants, negative ``t_ref``, negative ``rho``, or negative ``delta``.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state._nest.iaf_psc_exp import iaf_psc_exp
       >>> brainstate.environ.set(dt=0.1 * u.ms, t=0.0 * u.ms)
       >>> neu = iaf_psc_exp(in_size=(3,), I_e=250. * u.pA, delta=0. * u.mV)
       >>> neu.init_state()
       >>> out = neu.update(x=0. * u.pA, x_filtered=0. * u.pA)
       >>> out.shape
       (3,)
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
        rho: ArrayLike = 0.01 / u.second,
        delta: ArrayLike = 0. * u.mV,
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
        self.rho = braintools.init.param(rho, self.varshape)
        self.delta = braintools.init.param(delta, self.varshape)

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
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('Synaptic time constants must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')
        if np.any(self._to_numpy(self.rho, 1 / u.second) < 0.0):
            raise ValueError('Stochastic firing intensity rho must not be negative.')
        if np.any(self._to_numpy(self.delta, u.mV) < 0.0):
            raise ValueError('Threshold width delta must not be negative.')

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.i_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.i_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.i_0 = brainstate.ShortTermState(zeros * u.pA)
        self.i_1 = brainstate.ShortTermState(zeros * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(u.math.asarray(ref_steps > 0, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def update(self, x=0. * u.pA, x_filtered=0. * u.pA):
        """Advance the neuron state by one simulation step.

        Parameters
        ----------
        x : ArrayLike, optional
            Current input in pA for receptor-0 (standard current port). Scalar
            or array broadcastable to ``self.varshape``. The value is buffered
            and applied in the next step (NEST ring-buffer semantics). Default
            is ``0. * u.pA``.
        x_filtered : ArrayLike, optional
            Current input in pA for receptor-1. It is buffered to ``self.i_1``
            and injected through excitatory exponential filtering at the next
            update step via ``(1 - P11_ex) * i_1``. Scalar or array
            broadcastable to ``self.varshape``. Default is ``0. * u.pA``.

        Returns
        -------
        out : Any
            Surrogate spike output returned by :meth:`get_spike`. The output is
            elementwise over the neuron state shape (and batch axis, if
            initialized). For emitted spikes, the voltage argument to
            :meth:`get_spike` is nudged above threshold by ``1e-12`` mV to
            preserve positive spike activation under hard reset.

        Raises
        ------
        ValueError
            If provided inputs cannot be broadcast to the internal state shape.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        V_reset_rel = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)
        rho = self._broadcast_to_state(self._to_numpy(self.rho, 1 / u.second), v_shape)
        delta = self._broadcast_to_state(self._to_numpy(self.delta, u.mV), v_shape)

        i_0 = self._broadcast_to_state(self._to_numpy(self.i_0.value, u.pA), v_shape)
        i_1 = self._broadcast_to_state(self._to_numpy(self.i_1.value, u.pA), v_shape)
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value, u.pA), v_shape)
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.i_syn_in.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )

        P11_ex = np.exp(-h / tau_ex)
        P11_in = np.exp(-h / tau_in)
        P22 = np.exp(-h / tau_m)
        P21_ex = self._propagator_exp(tau_ex, tau_m, C_m, h)
        P21_in = self._propagator_exp(tau_in, tau_m, C_m, h)
        P20 = tau_m / C_m * (1.0 - P22)

        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all > 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)
        i_0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)
        i_1_next = self._broadcast_to_state(self._to_numpy(x_filtered, u.pA), v_shape)

        not_refractory = r == 0
        V_candidate = V_rel * P22 + i_syn_ex * P21_ex + i_syn_in * P21_in + (I_e + i_0) * P20
        V_rel = np.where(not_refractory, V_candidate, V_rel)
        r = np.where(not_refractory, r, r - 1)

        i_syn_ex = i_syn_ex * P11_ex
        i_syn_in = i_syn_in * P11_in

        # receptor type 1 current filtered through excitatory synapse.
        i_syn_ex = i_syn_ex + (1.0 - P11_ex) * i_1

        i_syn_ex = i_syn_ex + w_ex
        i_syn_in = i_syn_in + w_in

        deterministic = delta < 1e-10
        det_spike = V_rel >= theta
        # Probability is phi * h * 1e-3 since phi is in 1/s and h in ms.
        phi = rho * np.exp((V_rel - theta) / np.where(delta < 1e-10, 1.0, delta))
        stoch_spike = np.random.random(size=v_shape) < phi * h * 1e-3
        spike_cond = np.where(deterministic, det_spike, stoch_spike)

        r = np.where(spike_cond, self._broadcast_to_state(np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape), r)
        V_before_reset = V_rel
        V_rel = np.where(spike_cond, V_reset_rel, V_rel)

        self.V.value = (V_rel + E_L) * u.mV
        self.i_syn_ex.value = i_syn_ex * u.pA
        self.i_syn_in.value = i_syn_in * u.pA
        self.i_0.value = i_0_next * u.pA
        self.i_1.value = i_1_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        V_out = np.where(spike_cond, theta + E_L + 1e-12, V_before_reset + E_L)
        return self.get_spike(V_out * u.mV)

    @staticmethod
    def _propagator_exp(tau_syn: np.ndarray, tau_m: np.ndarray, c_m: np.ndarray, h_ms: float):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            beta = tau_syn * tau_m / (tau_m - tau_syn)
            gamma = beta / c_m
            inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)
            exp_h_tau_syn = np.exp(-h_ms / tau_syn)
            expm1_h_tau = np.expm1(h_ms * inv_beta)
            p32_raw = gamma * exp_h_tau_syn * expm1_h_tau

            normal_min = np.finfo(np.float64).tiny
            regular_mask = np.isfinite(p32_raw) & (np.abs(p32_raw) >= normal_min) & (p32_raw > 0.0)
            p32_singular = h_ms / c_m * np.exp(-h_ms / tau_m)
            return np.where(regular_mask, p32_raw, p32_singular)
