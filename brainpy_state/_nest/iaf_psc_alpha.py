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
    'iaf_psc_alpha',
]


class iaf_psc_alpha(Neuron):
    r"""NEST-compatible ``iaf_psc_alpha`` neuron model.

    Description
    -----------
    ``iaf_psc_alpha`` is a current-based leaky integrate-and-fire neuron with
    hard threshold/reset, fixed absolute refractory period, and alpha-shaped
    excitatory/inhibitory current kernels. The implementation mirrors NEST
    ``models/iaf_psc_alpha.{h,cpp}`` update order and propagator formulas.

    **1. Continuous-Time Dynamics**

    The membrane dynamics are

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m} + \frac{I_\text{syn} + I_e}{C_m}

    with :math:`I_\text{syn} = I_{\text{syn,ex}} + I_{\text{syn,in}}`.

    Each alpha current channel is represented by a two-state linear system:

    .. math::

       \frac{d\,dI_X}{dt} = -\frac{dI_X}{\tau_{\text{syn},X}}, \qquad
       \frac{dI_X}{dt} = dI_X - \frac{I_X}{\tau_{\text{syn},X}},
       \quad X \in \{\mathrm{ex}, \mathrm{in}\}.

    This is equivalent to the normalized alpha kernel

    .. math::

       i_X(t) = \frac{e}{\tau_{\text{syn},X}}\, t\, e^{-t/\tau_{\text{syn},X}} \Theta(t),

    which peaks at 1 when :math:`t=\tau_{\text{syn},X}`. Incoming spike weight
    :math:`w` (pA) is split by sign so :math:`w_+=\max(w,0)` drives excitatory
    state and :math:`w_-=\min(w,0)` drives inhibitory state.

    **2. Exact Discrete Propagator and NEST Update Order**

    For fixed step :math:`h=dt`, exact linear propagation is applied to
    :math:`y_3=V_m-E_L`, synaptic states, and a one-step delayed current buffer
    :math:`y_0`:

    .. math::

       dI_{X,n+1} = P_{11,X}\, dI_{X,n} + \frac{e}{\tau_{\text{syn},X}} w_{X,n},

    .. math::

       I_{X,n+1} = P_{21,X}\, dI_{X,n} + P_{22,X}\, I_{X,n},

    .. math::

       y_{3,n+1} = y_{3,n} + \big(e^{-h/\tau_m}-1\big) y_{3,n}
       + P_{30}(y_{0,n} + I_e)
       + \sum_{X \in \{\mathrm{ex},\mathrm{in}\}}
       \left(P_{31,X} dI_{X,n} + P_{32,X} I_{X,n}\right).

    Here :math:`P_{11,X}=P_{22,X}=e^{-h/\tau_{\text{syn},X}}`,
    :math:`P_{21,X}=h\,e^{-h/\tau_{\text{syn},X}}`, and
    :math:`P_{30}=\tau_m(1-e^{-h/\tau_m})/C_m`.

    Internal state (NEST notation):

    - :math:`y_0` -- buffered external current for next step,
    - :math:`dI_{ex}, I_{ex}` and :math:`dI_{in}, I_{in}` -- alpha-kernel states,
    - :math:`y_3 = V_m - E_L`,
    - :math:`r` -- refractory countdown in grid steps.

    Per-step update order:

    1. Update membrane potential if not refractory.
    2. Update synaptic alpha states.
    3. Add arriving spike input to :math:`dI_{ex}` / :math:`dI_{in}`.
    4. Perform threshold test, reset, refractory assignment, spike emission.
    5. Store buffered external current for the next step.

    **3. Near-Singular Regime :math:`\tau_m \approx \tau_{\text{syn}}`**

    Direct formulas for :math:`P_{31}` and :math:`P_{32}` contain divisions by
    :math:`(\tau_m-\tau_{\text{syn}})`, which are ill-conditioned near
    equality. The helper :meth:`_alpha_propagator_p31_p32` follows NEST's
    ``IAFPropagatorAlpha`` behavior and switches to stable limits:

    .. math::

       P_{32}^{\mathrm{sing}} = \frac{h}{C_m} e^{-h/\tau_m}, \qquad
       P_{31}^{\mathrm{sing}} = \frac{h^2}{2C_m} e^{-h/\tau_m},

    preventing cancellation/underflow artifacts around
    :math:`\tau_m=\tau_{\text{syn}}`.

    **4. Assumptions, Constraints, and Computational Implications**

    - ``C_m > 0``, ``tau_m > 0``, ``tau_syn_ex > 0``, ``tau_syn_in > 0``,
      ``t_ref >= 0``, and ``V_reset < V_th`` are enforced at construction.
    - ``update(x=...)`` uses one-step delayed current buffering (NEST
      ring-buffer semantics): current provided at step ``n`` contributes at
      step ``n+1`` through ``y0``.
    - The update path is vectorized over ``self.varshape`` and performs
      :math:`O(\prod \mathrm{varshape})` floating-point work per call.
    - Internal coefficient math is in ``float64`` via NumPy conversion, while
      exposed states remain in BrainUnit quantities.

    Parameters
    ----------
    in_size : Size
        Population shape specification. All per-neuron parameters are
        broadcast to ``self.varshape`` derived from ``in_size``.
    E_L : ArrayLike, optional
        Resting potential :math:`E_L` in mV; scalar or array broadcastable to
        ``self.varshape``. Default is ``-70. * u.mV``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF; broadcastable to
        ``self.varshape`` and strictly positive. Default is ``250. * u.pF``.
    tau_m : ArrayLike, optional
        Membrane time constant :math:`\tau_m` in ms; broadcastable and strictly
        positive. Default is ``10. * u.ms``.
    t_ref : ArrayLike, optional
        Absolute refractory period :math:`t_{ref}` in ms; broadcastable and
        nonnegative. Converted to integer step counts by ``ceil(t_ref / dt)``.
        Default is ``2. * u.ms``.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_{th}` in mV; broadcastable to ``self.varshape``.
        Default is ``-55. * u.mV``.
    V_reset : ArrayLike, optional
        Reset potential :math:`V_{reset}` in mV; broadcastable and must satisfy
        ``V_reset < V_th`` elementwise. Default is ``-70. * u.mV``.
    tau_syn_ex : ArrayLike, optional
        Excitatory alpha time constant :math:`\tau_{\text{syn,ex}}` in ms;
        broadcastable and strictly positive. Default is ``2. * u.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory alpha time constant :math:`\tau_{\text{syn,in}}` in ms;
        broadcastable and strictly positive. Default is ``2. * u.ms``.
    I_e : ArrayLike, optional
        Constant injected current :math:`I_e` in pA; scalar or array
        broadcastable to ``self.varshape``. Default is ``0. * u.pA``.
    V_min : ArrayLike or None, optional
        Optional lower voltage clamp :math:`V_{min}` in mV. When provided,
        membrane candidate update is clipped with ``max(V, V_min)`` before
        thresholding. ``None`` disables clipping. Default is ``None``.
    V_initializer : Callable, optional
        Initializer for membrane state ``V``. Called by :meth:`init_state`.
        Default is ``braintools.init.Constant(-70. * u.mV)``.
    spk_fun : Callable, optional
        Surrogate spike nonlinearity used by :meth:`get_spike`. Default is
        ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy inherited from :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` matches NEST hard reset semantics. Default is ``'hard'``.
    ref_var : bool, optional
        If ``True``, allocates boolean state ``self.refractory`` for external
        inspection of refractory condition. Default is ``False``.
    name : str or None, optional
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 17 27 14 16 36

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
         - Membrane capacitance used in subthreshold integration.
       * - ``tau_m``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``10. * u.ms``
         - :math:`\tau_m`
         - Leak time constant in membrane propagator.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), ``>= 0``
         - ``2. * u.ms``
         - :math:`t_{ref}`
         - Absolute refractory duration.
       * - ``V_th`` and ``V_reset``
         - ArrayLike, broadcastable (mV), with ``V_reset < V_th``
         - ``-55. * u.mV``, ``-70. * u.mV``
         - :math:`V_{th}`, :math:`V_{reset}`
         - Threshold and post-spike reset levels.
       * - ``tau_syn_ex`` and ``tau_syn_in``
         - ArrayLike, broadcastable (ms), each ``> 0``
         - ``2. * u.ms``
         - :math:`\tau_{\text{syn,ex}}`, :math:`\tau_{\text{syn,in}}`
         - Alpha kernel time constants for excitatory/inhibitory currents.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant external current added every step.
       * - ``V_min``
         - ArrayLike broadcastable (mV) or ``None``
         - ``None``
         - :math:`V_{min}`
         - Optional lower bound on membrane candidate update.
       * - ``V_initializer``
         - Callable
         - ``Constant(-70. * u.mV)``
         - --
         - Initializer used for membrane state ``V``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate spike function returned by :meth:`update`.
       * - ``spk_reset``
         - str
         - ``'hard'``
         - --
         - Reset mode inherited from :class:`~brainpy_state._base.Neuron`.
       * - ``ref_var``
         - bool
         - ``False``
         - --
         - Allocate boolean state ``self.refractory`` when enabled.
       * - ``name``
         - str | None
         - ``None``
         - --
         - Optional node name.

    Returns
    -------
    out : Any
        Configured neuron node. Each :meth:`update` call returns spike output
        with shape ``self.V.value.shape`` produced by ``spk_fun``.

    Raises
    ------
    ValueError
        If parameter constraints are violated: ``C_m <= 0``, ``tau_m <= 0``,
        ``tau_syn_ex <= 0``, ``tau_syn_in <= 0``, ``t_ref < 0``, or
        ``V_reset >= V_th``.
    TypeError
        If provided quantities are not unit-compatible with expected units
        (mV, ms, pF, pA) during conversion/broadcasting.
    KeyError
        At runtime, if required simulation context entries (for example ``t``
        or ``dt``) are missing when :meth:`update` is called.
    AttributeError
        If :meth:`update` is called before :meth:`init_state` creates required
        state holders.

    Notes
    -----

    - State variables are ``V``, ``I_syn_ex``, ``I_syn_in``, ``dI_syn_ex``,
      ``dI_syn_in``, ``y0``, ``refractory_step_count``, and ``last_spike_time``.
      ``refractory`` is added only when ``ref_var=True``.
    - Spike weights from ``sum_delta_inputs`` are interpreted in pA:
      positive values are excitatory and negative values are inhibitory.
    - ``update(x=...)`` stores ``x`` into ``y0`` for the next step, matching
      NEST current-event buffering semantics.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_alpha(in_size=(2,), I_e=220.0 * u.pA)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=1.0 * u.ms):
       ...         spk = neu.update()
       ...     _ = spk.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_alpha(in_size=1, tau_syn_ex=1.5 * u.ms)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = neu.update(x=200.0 * u.pA)
       ...     with brainstate.environ.context(t=0.1 * u.ms):
       ...         spk_next = neu.update()
       ...     _ = spk_next

    References
    ----------
    .. [1] NEST source: ``models/iaf_psc_alpha.h`` and
           ``models/iaf_psc_alpha.cpp``.
    .. [2] Rotter S, Diesmann M (1999). Exact simulation of time-invariant linear
           systems with applications to neuronal modeling. Biological Cybernetics
           81:381-402. DOI: https://doi.org/10.1007/s004220050570
    .. [3] Diesmann M, Gewaltig M-O, Rotter S, Aertsen A (2001). State space
           analysis of synchronous spiking in cortical neural networks.
           Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X
    .. [4] Morrison A, Straube S, Plesser HE, Diesmann M (2007). Exact
           subthreshold integration with continuous spike times in discrete time
           neural network simulations. Neural Computation 19(1):47-79.
           DOI: https://doi.org/10.1162/neco.2007.19.1.47
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
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be > 0.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be > 0.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('All synaptic time constants must be > 0.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError("The refractory time t_ref can't be negative.")
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize runtime states for membrane, synapses, and refractoriness.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension used by
            :func:`braintools.init.param` when creating all state arrays.
            ``None`` creates unbatched states with shape ``self.varshape``.
        **kwargs : Any
            Unused compatibility arguments. Accepted for interface consistency
            with other nodes.

        Returns
        -------
        out : Any
            ``None``. This method mutates the instance in-place by creating
            ``V``, synaptic states, refractory counters, and spike-time state.

        Raises
        ------
        ValueError
            If initializers cannot broadcast to ``self.varshape`` (or to
            ``(batch_size,) + self.varshape`` when batching is requested).
        TypeError
            If initializer outputs are incompatible with expected unit/array
            conversions for voltage, current, or integer refractory states.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.dI_syn_ex = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.dI_syn_in = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.y0 = brainstate.ShortTermState(np.asarray(zeros, dtype=np.float64))
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(u.math.asarray(ref_steps > 0, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        r"""Evaluate surrogate spike output for a voltage tensor.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Voltage input in mV, broadcast-compatible with ``self.varshape``.
            If ``None``, uses current membrane state ``self.V.value``.

        Returns
        -------
        out : Any
            Surrogate spike output from ``self.spk_fun`` with the same shape as
            ``V`` (or ``self.V.value`` when ``V is None``).
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    @staticmethod
    def _alpha_propagator_p31_p32(tau_syn: np.ndarray, tau_m: np.ndarray, c_m: np.ndarray, h_ms: float):
        r"""Compute alpha-kernel membrane propagator terms ``P31`` and ``P32``.

        Parameters
        ----------
        tau_syn : numpy.ndarray
            Synaptic time constants in ms. Shape must be broadcast-compatible
            with state tensors.
        tau_m : numpy.ndarray
            Membrane time constants in ms, broadcast-compatible with
            ``tau_syn`` and positive.
        c_m : numpy.ndarray
            Membrane capacitances in pF, broadcast-compatible with ``tau_syn``
            and positive.
        h_ms : float
            Integration step in ms.

        Returns
        -------
        out : Any
            Tuple ``(P31, P32)`` of ``float64`` NumPy arrays, each broadcast to
            the input shapes. Singular fallback limits are applied when regular
            formulas become numerically unreliable near
            ``tau_m ~= tau_syn``.

        Notes
        -----
        This helper reproduces NEST ``IAFPropagatorAlpha`` masking logic with
        NumPy finite/normal checks to avoid catastrophic cancellation.
        """
        # Mirrors NEST IAFPropagatorAlpha and singular fallback behavior.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            beta = tau_syn * tau_m / (tau_m - tau_syn)
            gamma = beta / c_m
            inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)

            exp_h_tau_syn = np.exp(-h_ms / tau_syn)
            expm1_h_tau = np.expm1(h_ms * inv_beta)

            p32_raw = gamma * exp_h_tau_syn * expm1_h_tau
            exp_h_tau_m = np.exp(-h_ms / tau_m)
            p32_singular = h_ms / c_m * exp_h_tau_m

            # NEST checks "isnormal && > 0". Approximate isnormal in NumPy.
            normal_min = np.finfo(np.float64).tiny
            p32_regular_mask = np.isfinite(p32_raw) & (np.abs(p32_raw) >= normal_min) & (p32_raw > 0.0)
            p32 = np.where(p32_regular_mask, p32_raw, p32_singular)

            h_min_regular = 1e-7 * tau_m * tau_m / np.abs(tau_m - tau_syn)
            p31_regular_mask = np.isfinite(h_min_regular) & (h_ms > h_min_regular)

            p31_regular = gamma * exp_h_tau_syn * (beta * expm1_h_tau - h_ms)
            p31_singular = 0.5 * h_ms * h_ms / c_m * exp_h_tau_m
            p31 = np.where(p31_regular_mask, p31_regular, p31_singular)

        return p31, p32

    def update(self, x=0. * u.pA):
        r"""Advance the neuron by one simulation step.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous current input in pA for this step. ``x`` is accumulated
            through :meth:`sum_current_inputs` and stored in ``y0`` for use on
            the next call (one-step delayed buffering).

        Returns
        -------
        out : Any
            Spike output tensor from :meth:`get_spike`, shape
            ``self.V.value.shape``. On threshold crossings, ``v_out`` is nudged
            above threshold by ``1e-12`` mV-equivalent to preserve positive
            surrogate activation.

        Raises
        ------
        KeyError
            If simulation context does not provide ``t`` or ``dt``.
        AttributeError
            If required states are missing because :meth:`init_state` was not
            called.
        TypeError
            If ``x`` or stored states are not unit-compatible with expected pA
            / mV conversions.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        v_abs = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        y3 = v_abs - E_L

        c_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        i_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)

        theta_rel = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        v_reset_rel = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)
        lower_rel = -np.inf * np.ones(v_shape, dtype=np.float64)
        if self.V_min is not None:
            lower_rel = self._broadcast_to_state(self._to_numpy(self.V_min - self.E_L, u.mV), v_shape)

        y0 = self._broadcast_to_state(np.asarray(self.y0.value, dtype=np.float64), v_shape)
        dI_ex = self._broadcast_to_state(np.asarray(self.dI_syn_ex.value, dtype=np.float64), v_shape)
        I_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape)
        dI_in = self._broadcast_to_state(np.asarray(self.dI_syn_in.value, dtype=np.float64), v_shape)
        I_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            v_shape,
        )

        # Coefficients from NEST pre_run_hook.
        P11_ex = np.exp(-h / tau_ex)
        P22_ex = P11_ex
        P21_ex = h * P11_ex

        P11_in = np.exp(-h / tau_in)
        P22_in = P11_in
        P21_in = h * P11_in

        expm1_tau_m = np.expm1(-h / tau_m)
        P30 = -tau_m / c_m * expm1_tau_m
        P31_ex, P32_ex = self._alpha_propagator_p31_p32(tau_ex, tau_m, c_m, h)
        P31_in, P32_in = self._alpha_propagator_p31_p32(tau_in, tau_m, c_m, h)

        epsc_init = math.e / tau_ex
        ipsc_init = math.e / tau_in

        # Spike/current buffers for next step.
        w_all = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        w_ex = np.where(w_all > 0.0, w_all, 0.0)
        w_in = np.where(w_all < 0.0, w_all, 0.0)
        y0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        # 1) membrane update
        not_refractory = r == 0
        y3_candidate = (
            P30 * (y0 + i_e)
            + P31_ex * dI_ex
            + P32_ex * I_ex
            + P31_in * dI_in
            + P32_in * I_in
            + expm1_tau_m * y3
            + y3
        )
        y3_candidate = np.maximum(y3_candidate, lower_rel)
        y3 = np.where(not_refractory, y3_candidate, y3)
        r = np.where(not_refractory, r, r - 1)

        # 2) synaptic alpha updates
        I_ex = P21_ex * dI_ex + P22_ex * I_ex
        dI_ex = dI_ex * P11_ex
        dI_ex = dI_ex + epsc_init * w_ex

        I_in = P21_in * dI_in + P22_in * I_in
        dI_in = dI_in * P11_in
        dI_in = dI_in + ipsc_init * w_in

        # 3) threshold + reset
        spike_cond = y3 >= theta_rel
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            v_shape,
        )
        r = np.where(spike_cond, refr_counts, r)
        y3_for_spike = y3
        y3 = np.where(spike_cond, v_reset_rel, y3)

        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )

        # write-back
        self.V.value = (y3 + E_L) * u.mV
        self.I_syn_ex.value = I_ex * u.pA
        self.I_syn_in.value = I_in * u.pA
        self.dI_syn_ex.value = dI_ex
        self.dI_syn_in.value = dI_in
        self.y0.value = y0_next
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        v_out = np.where(spike_cond, theta_rel + E_L + 1e-12, y3_for_spike + E_L)
        return self.get_spike(v_out * u.mV)
