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
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron
from .iaf_psc_exp import iaf_psc_exp

__all__ = [
    'iaf_psc_exp_htum',
]


class iaf_psc_exp_htum(NESTNeuron):
    r"""NEST-compatible ``iaf_psc_exp_htum`` neuron model.

    Description
    -----------

    ``iaf_psc_exp_htum`` is a current-based leaky integrate-and-fire neuron
    with exponential excitatory/inhibitory PSCs and two refractory clocks:
    an absolute refractory period and a longer (or equal) total refractory
    period. The implementation matches NEST ``iaf_psc_exp_htum`` semantics:
    membrane integration is disabled during absolute refractory, while
    threshold crossing is disabled during total refractory.

    **1. Continuous-Time Dynamics**


    For membrane potential :math:`V_m` and resting potential :math:`E_L`,
    subthreshold dynamics are:

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m}
       + \frac{I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}} + I_e + I_0}{C_m},

    where :math:`I_0` is a one-step delayed buffered continuous current.
    Synaptic currents follow first-order exponential decay:

    .. math::

       \frac{dI_{\mathrm{syn,ex}}}{dt} =
       -\frac{I_{\mathrm{syn,ex}}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{dI_{\mathrm{syn,in}}}{dt} =
       -\frac{I_{\mathrm{syn,in}}}{\tau_{\mathrm{syn,in}}}.

    **2. Exact Discrete-Time Propagation and Dual Refractory Gating**


    With :math:`h=dt` (ms), the implementation uses exact linear propagators:

    .. math::

       P_{11,\mathrm{ex}} = e^{-h/\tau_{\mathrm{syn,ex}}}, \quad
       P_{11,\mathrm{in}} = e^{-h/\tau_{\mathrm{syn,in}}}, \quad
       P_{22} = e^{-h/\tau_m},

    .. math::

       P_{20} = \frac{\tau_m}{C_m}(1 - P_{22}),

    .. math::

       P_{21}(\tau_{\mathrm{syn}})=
       \frac{\tau_{\mathrm{syn}}\tau_m}{C_m(\tau_m-\tau_{\mathrm{syn}})}
       \left(e^{-h/\tau_m}-e^{-h/\tau_{\mathrm{syn}}}\right),

    where :meth:`iaf_psc_exp._propagator_exp` handles the numerically delicate
    :math:`\tau_{\mathrm{syn}} \approx \tau_m` case.

    Let :math:`V_{\mathrm{rel}} = V_m - E_L` and
    :math:`\theta = V_{th} - E_L`. The candidate voltage update is:

    .. math::

       V_{\mathrm{rel},n+1} = P_{22}V_{\mathrm{rel},n}
       + P_{21,\mathrm{ex}} I_{\mathrm{syn,ex},n}
       + P_{21,\mathrm{in}} I_{\mathrm{syn,in},n}
       + P_{20}(I_e + I_{0,n}).

    Dual refractory counters are stored as integer step counts:

    - :math:`r_{\mathrm{abs}} = \lceil t_{\mathrm{ref,abs}} / dt \rceil`
    - :math:`r_{\mathrm{tot}} = \lceil t_{\mathrm{ref,tot}} / dt \rceil`

    Integration is applied only where ``r_abs == 0``. Spiking is allowed only
    where ``r_tot == 0`` and :math:`V_{\mathrm{rel}} \ge \theta`.
    On spike: voltage resets to ``V_reset``, ``r_abs`` and ``r_tot`` are
    reloaded from their ceiling-converted refractory durations, and
    ``last_spike_time`` is set to ``t + dt`` (NEST-aligned grid timing).

    **3. Step Ordering and NEST Timing Equivalence**


    Per simulation step:

    1. Compute membrane candidate only for neurons outside absolute refractory.
    2. Decrement absolute refractory counters for clamped neurons.
    3. Decay synaptic currents and add arriving delta spikes:
       positive weights to excitatory channel, negative to inhibitory channel.
    4. Apply threshold test only for neurons outside total refractory.
    5. On spike, apply reset and set both refractory counters.
    6. Decrement total refractory counters for non-spiking refractory neurons.
    7. Buffer continuous current input ``i_0`` for use at the next step.

    This ordering preserves NEST's one-step delayed current-event handling and
    supports mixed per-neuron parameterization via broadcasted arrays.

    **4. Stability Constraints and Computational Implications**


    - Construction enforces ``V_reset < V_th``, ``C_m > 0``, ``tau_m > 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, ``t_ref_abs > 0``,
      ``t_ref_tot > 0``, and ``t_ref_tot >= t_ref_abs``.
    - Refractory durations are discretized by ``ceil``; effective refractory
      lengths are therefore quantized in units of ``dt``.
    - Coefficient evaluation is vectorized in ``float64`` NumPy and scales as
      :math:`O(\prod \mathrm{varshape})` per step.
    - ``i_0`` buffering implies current passed at step ``n`` contributes to
      membrane integration at step ``n+1``, not immediately.

    Parameters
    ----------
    in_size : Size
        Population shape specification. Per-neuron parameters are initialized
        and broadcast to ``self.varshape``.
    E_L : ArrayLike, optional
        Resting potential :math:`E_L` in mV; scalar or array broadcastable to
        ``self.varshape``. Default is ``-70. * u.mV``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF; broadcastable and strictly
        positive. Default is ``250. * u.pF``.
    tau_m : ArrayLike, optional
        Membrane time constant :math:`\tau_m` in ms; broadcastable and
        strictly positive. Default is ``10. * u.ms``.
    t_ref_abs : ArrayLike, optional
        Absolute refractory duration in ms; broadcastable and strictly
        positive. Converted to integer step count by
        ``ceil(t_ref_abs / dt)``. Default is ``2. * u.ms``.
    t_ref_tot : ArrayLike, optional
        Total refractory duration in ms; broadcastable and strictly positive,
        with ``t_ref_tot >= t_ref_abs`` elementwise. Converted by
        ``ceil(t_ref_tot / dt)``. Default is ``2. * u.ms``.
    V_th : ArrayLike, optional
        Threshold potential :math:`V_{th}` in mV; scalar or array
        broadcastable to ``self.varshape``. Default is ``-55. * u.mV``.
    V_reset : ArrayLike, optional
        Reset potential :math:`V_{reset}` in mV; broadcastable and must satisfy
        ``V_reset < V_th`` elementwise. Default is ``-70. * u.mV``.
    tau_syn_ex : ArrayLike, optional
        Excitatory PSC decay constant :math:`\tau_{\mathrm{syn,ex}}` in ms;
        broadcastable and strictly positive. Default is ``2. * u.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory PSC decay constant :math:`\tau_{\mathrm{syn,in}}` in ms;
        broadcastable and strictly positive. Default is ``2. * u.ms``.
    I_e : ArrayLike, optional
        Constant external injected current :math:`I_e` in pA; scalar or array
        broadcastable to ``self.varshape``. Default is ``0. * u.pA``.
    V_initializer : Callable, optional
        Initializer callable consumed by :meth:`init_state` for ``self.V``.
        It must return values compatible with mV units and neuron shape.
        Default is ``braintools.init.Constant(-70. * u.mV)``.
    spk_fun : Callable, optional
        Surrogate spike nonlinearity applied by :meth:`get_spike` to a scaled
        voltage distance from threshold. Default is
        ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset mode used by :class:`~brainpy_state._base.Neuron`; ``'hard'``
        matches reset semantics used by this model. Default is ``'hard'``.
    ref_var : bool, optional
        If ``True``, allocates ``self.refractory`` (boolean short-term state)
        mirroring ``refractory_tot_step_count > 0``. Default is ``False``.
    name : str or None, optional
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 18 30 15 14 33

       * - Parameter
         - Type / shape / unit
         - Default
         - Math symbol
         - Semantics
       * - ``in_size``
         - :class:`~brainstate.typing.Size`; scalar or tuple-like
         - required
         - --
         - Defines population shape ``self.varshape``.
       * - ``E_L``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``-70. * u.mV``
         - :math:`E_L`
         - Resting membrane potential.
       * - ``C_m``
         - ArrayLike, broadcastable (pF), ``> 0``
         - ``250. * u.pF``
         - :math:`C_m`
         - Membrane capacitance.
       * - ``tau_m``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``10. * u.ms``
         - :math:`\tau_m`
         - Leak time constant.
       * - ``t_ref_abs``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``2. * u.ms``
         - :math:`t_{\mathrm{ref,abs}}`
         - Absolute membrane clamp duration; quantized to
           ``ceil(t_ref_abs / dt)`` steps.
       * - ``t_ref_tot``
         - ArrayLike, broadcastable (ms), ``> 0``,
           ``>= t_ref_abs`` elementwise
         - ``2. * u.ms``
         - :math:`t_{\mathrm{ref,tot}}`
         - Total spike-suppression duration; quantized to
           ``ceil(t_ref_tot / dt)`` steps.
       * - ``V_th``
         - ArrayLike, broadcastable (mV)
         - ``-55. * u.mV``
         - :math:`V_{th}`
         - Spike threshold potential.
       * - ``V_reset``
         - ArrayLike, broadcastable (mV), ``< V_th`` elementwise
         - ``-70. * u.mV``
         - :math:`V_{reset}`
         - Post-spike reset potential.
       * - ``tau_syn_ex``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``2. * u.ms``
         - :math:`\tau_{\mathrm{syn,ex}}`
         - Excitatory exponential synaptic decay constant.
       * - ``tau_syn_in``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``2. * u.ms``
         - :math:`\tau_{\mathrm{syn,in}}`
         - Inhibitory exponential synaptic decay constant.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant injected current.
       * - ``V_initializer``
         - Callable returning values compatible with neuron shape (mV)
         - ``Constant(-70. * u.mV)``
         - --
         - Initial value generator for membrane potential state.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate spike output transform.
       * - ``spk_reset``
         - ``str`` (typically ``'hard'``)
         - ``'hard'``
         - --
         - Reset mode passed to base class.
       * - ``ref_var``
         - ``bool``
         - ``False``
         - --
         - Enables explicit boolean refractory flag state.
       * - ``name``
         - ``str`` or ``None``
         - ``None``
         - --
         - Optional instance name.

    Raises
    ------
    ValueError
        Raised during construction if any hard model constraint is violated:
        ``V_reset >= V_th``, nonpositive ``C_m``/``tau_m``/synaptic time
        constants, nonpositive ``t_ref_abs``/``t_ref_tot``, or
        ``t_ref_abs > t_ref_tot``.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state._nest.iaf_psc_exp_htum import iaf_psc_exp_htum
       >>> brainstate.environ.set(dt=0.1 * u.ms, t=0.0 * u.ms)
       >>> neu = iaf_psc_exp_htum(
       ...     in_size=(2,),
       ...     I_e=200. * u.pA,
       ...     t_ref_abs=1.0 * u.ms,
       ...     t_ref_tot=2.0 * u.ms,
       ... )
       >>> neu.init_state()
       >>> out = neu.update(x=0. * u.pA)
       >>> out.shape
       (2,)
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 250. * u.pF,
        tau_m: ArrayLike = 10. * u.ms,
        t_ref_abs: ArrayLike = 2. * u.ms,
        t_ref_tot: ArrayLike = 2. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -70. * u.mV,
        tau_syn_ex: ArrayLike = 2. * u.ms,
        tau_syn_in: ArrayLike = 2. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
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
        self.t_ref_abs = braintools.init.param(t_ref_abs, self.varshape)
        self.t_ref_tot = braintools.init.param(t_ref_tot, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
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
        if np.any(self._to_numpy(self.t_ref_abs, u.ms) <= 0.0) or np.any(self._to_numpy(self.t_ref_tot, u.ms) <= 0.0):
            raise ValueError('All refractory time constants must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref_abs, u.ms) > self._to_numpy(self.t_ref_tot, u.ms)):
            raise ValueError('Total refractory period must be >= absolute refractory period.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('Synaptic time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Allocate and initialise all dynamic state variables.

        Creates ``HiddenState`` and ``ShortTermState`` fields required by
        :meth:`update`. All counters are zero-initialised; ``last_spike_time``
        is set to ``-1e7 * u.ms`` (effectively never spiked).

        Parameters
        ----------
        batch_size : int or None, optional
            If not ``None``, state arrays gain a leading batch dimension of
            this size. Pass ``None`` for single-trial (non-batched) simulation.
            Default is ``None``.
        **kwargs
            Absorbed for API compatibility with the base-class signature.

        Notes
        -----
        After this call the following state attributes exist:

        - ``self.V`` (:class:`~brainstate.HiddenState`, mV) — membrane
          potential, shape ``(batch_size, *varshape)`` or ``varshape``.
        - ``self.i_syn_ex`` / ``self.i_syn_in``
          (:class:`~brainstate.ShortTermState`, pA) — exponential excitatory
          and inhibitory synaptic current buffers.
        - ``self.i_0`` (:class:`~brainstate.ShortTermState`, pA) — one-step
          delayed continuous current buffer.
        - ``self.refractory_abs_step_count`` /
          ``self.refractory_tot_step_count``
          (:class:`~brainstate.ShortTermState`, ``int32``) — remaining
          absolute and total refractory step counters.
        - ``self.last_spike_time`` (:class:`~brainstate.ShortTermState`, ms) —
          time of most recent spike per neuron; initialised to ``-1e7 ms``.
        - ``self.refractory`` (:class:`~brainstate.ShortTermState`, ``bool``)
          — present only when ``ref_var=True``; mirrors
          ``refractory_tot_step_count > 0``.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.i_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.i_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.i_0 = brainstate.ShortTermState(zeros * u.pA)
        ditype = brainstate.environ.ditype()
        self.refractory_abs_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=ditype))
        self.refractory_tot_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=ditype))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(u.math.asarray(ref_steps > 0, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        r"""Compute surrogate spike output from membrane voltage.

        Scales the membrane potential relative to the threshold/reset gap and
        passes the result through the configured surrogate nonlinearity
        ``self.spk_fun``, enabling gradient-based optimisation through
        spike-generation events.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Membrane potential in mV. If ``None``, uses the current value of
            ``self.V.value``. Shape must be broadcastable to the neuron state
            shape. Default is ``None``.

        Returns
        -------
        spike : ArrayLike
            Surrogate spike values with the same shape as ``V`` (or
            ``self.V.value`` if ``V`` is ``None``). For hard thresholding,
            non-zero values indicate a spike event at this step.

        Notes
        -----
        The scaling is

        .. math::

           v_{\mathrm{scaled}} = \frac{V - V_{th}}{V_{th} - V_{reset}},

        which places the threshold at :math:`v_{\mathrm{scaled}} = 0` and the
        reset at :math:`v_{\mathrm{scaled}} = -1`.  The surrogate function
        ``spk_fun`` (e.g., ``ReluGrad``) is then applied element-wise.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_abs_counts(self):
        dt = brainstate.environ.get_dt()
        ditype = brainstate.environ.ditype()
        return u.math.asarray(u.math.ceil(self.t_ref_abs / dt), dtype=ditype)

    def _refractory_tot_counts(self):
        dt = brainstate.environ.get_dt()
        ditype = brainstate.environ.ditype()
        return u.math.asarray(u.math.ceil(self.t_ref_tot / dt), dtype=ditype)

    def update(self, x=0. * u.pA):
        r"""Advance the neuron state by one simulation step.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous current input at the current step in pA. Accepted as a
            scalar or array broadcastable to the membrane state shape. This
            value is buffered into ``i_0`` and used for membrane integration on
            the next call (one-step delay), matching NEST current-event timing.
            Default is ``0. * u.pA``.

        Returns
        -------
        out : jax.Array
            Surrogate spike output from :meth:`get_spike`, with shape equal to
            the neuron state shape (or batched state shape after
            :meth:`init_state`). Values correspond to threshold events detected
            at this step under total refractory gating.

        Raises
        ------
        AttributeError
            If called before :meth:`init_state` has created required state
            fields (``V``, synaptic buffers, refractory counters).
        ValueError
            If input ``x`` or internal arrays cannot be broadcast to the state
            shape, or if upstream state/input values carry incompatible units
            for conversion to mV/pA/ms.
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

        i_0 = self._broadcast_to_state(self._to_numpy(self.i_0.value, u.pA), v_shape)
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value, u.pA), v_shape)
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.i_syn_in.value, u.pA), v_shape)
        ditype = brainstate.environ.ditype()
        r_abs = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_abs_step_count.value), dtype=ditype), v_shape
        )
        r_tot = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_tot_step_count.value), dtype=ditype), v_shape
        )

        P11_ex = np.exp(-h / tau_ex)
        P11_in = np.exp(-h / tau_in)
        P22 = np.exp(-h / tau_m)
        P21_ex = iaf_psc_exp._propagator_exp(tau_ex, tau_m, C_m, h)
        P21_in = iaf_psc_exp._propagator_exp(tau_in, tau_m, C_m, h)
        P20 = tau_m / C_m * (1.0 - P22)

        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all >= 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)
        i_0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        if np.any(r_abs == 0):
            V_candidate = V_rel * P22 + i_syn_ex * P21_ex + i_syn_in * P21_in + (I_e + i_0) * P20
            V_rel = np.where(r_abs == 0, V_candidate, V_rel)
        r_abs = np.where(r_abs == 0, r_abs, r_abs - 1)

        i_syn_ex = i_syn_ex * P11_ex + w_ex
        i_syn_in = i_syn_in * P11_in + w_in

        can_spike = r_tot == 0
        spike_cond = can_spike & (V_rel >= theta)
        refr_abs = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_abs_counts()), dtype=ditype), v_shape
        )
        refr_tot = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_tot_counts()), dtype=ditype), v_shape
        )
        r_abs = np.where(spike_cond, refr_abs, r_abs)
        r_tot = np.where(spike_cond, refr_tot, np.where(r_tot > 0, r_tot - 1, r_tot))
        V_before_reset = V_rel
        V_rel = np.where(spike_cond, V_reset_rel, V_rel)

        self.V.value = (V_rel + E_L) * u.mV
        self.i_syn_ex.value = i_syn_ex * u.pA
        self.i_syn_in.value = i_syn_in * u.pA
        self.i_0.value = i_0_next * u.pA
        self.refractory_abs_step_count.value = jnp.asarray(r_abs, dtype=ditype)
        self.refractory_tot_step_count.value = jnp.asarray(r_tot, dtype=ditype)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_tot_step_count.value > 0)

        # Emit spikes only on actual threshold events (respecting total refractory).
        V_nospike = np.minimum(V_before_reset, theta - 1e-12)
        V_out = np.where(spike_cond, theta + E_L + 1e-12, V_nospike + E_L)
        return self.get_spike(V_out * u.mV)
