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
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron
from .iaf_psc_alpha import iaf_psc_alpha

__all__ = [
    'iaf_psc_alpha_multisynapse',
]


class iaf_psc_alpha_multisynapse(NESTNeuron):
    r"""NEST-compatible ``iaf_psc_alpha_multisynapse`` neuron model.

    Current-based leaky integrate-and-fire neuron with an arbitrary number of
    receptor-indexed alpha-shaped synaptic current channels.

    Description
    -----------
    ``iaf_psc_alpha_multisynapse`` mirrors NEST
    ``models/iaf_psc_alpha_multisynapse.{h,cpp}`` and generalizes
    :class:`iaf_psc_alpha` from two fixed excitatory/inhibitory channels to
    ``n_receptors`` independently parameterized current ports.

    Each receptor ``k`` (1-based, NEST convention) carries its own alpha time
    constant ``tau_syn[k-1]``. Synaptic weights are signed currents in pA;
    positive values are depolarizing and negative values are hyperpolarizing.

    **1. Continuous-Time Dynamics and Receptor States**

    Membrane dynamics are

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m}
       + \frac{\sum_k I_k + I_e + I_0}{C_m},

    where :math:`I_0` is the one-step delayed continuous current buffer
    (NEST ring-buffer semantics) and :math:`I_k` are the per-receptor alpha
    currents.

    For each receptor :math:`k`, the alpha current kernel is represented by a
    two-state linear system (``y1[k]``, ``y2[k]``):

    .. math::

       \frac{d\,y1_k}{dt} = -\frac{y1_k}{\tau_{\mathrm{syn},k}}, \qquad
       \frac{d\,y2_k}{dt} = y1_k - \frac{y2_k}{\tau_{\mathrm{syn},k}}.

    The effective synaptic current for receptor :math:`k` is :math:`I_k = y2_k`.
    An incoming spike with weight :math:`w_k` (pA) is injected into ``y1[k]``
    with the NEST alpha normalization factor:

    .. math::

       y1_k \leftarrow y1_k + \frac{e}{\tau_{\mathrm{syn},k}} w_k.

    This normalization ensures that a single spike with weight :math:`w_k`
    produces a current kernel that peaks exactly at :math:`w_k` when
    :math:`t = \tau_{\mathrm{syn},k}`:

    .. math::

       I_k(t) = w_k \frac{t}{\tau_{\mathrm{syn},k}}
       \exp\!\left(1 - \frac{t}{\tau_{\mathrm{syn},k}}\right), \quad t \ge 0.

    **2. Exact Discrete Propagator, Derivation Constraints, and Stability**

    With fixed step :math:`h = dt`, exact matrix propagation of the linear
    subsystem is used. For each receptor :math:`k`:

    .. math::

       y1_{k,n+1} = P_{11,k}\,y1_{k,n} + \frac{e}{\tau_{\mathrm{syn},k}} w_{k,n},

    .. math::

       y2_{k,n+1} = P_{21,k}\,y1_{k,n} + P_{22,k}\,y2_{k,n},

    where :math:`P_{11,k} = P_{22,k} = e^{-h/\tau_{\mathrm{syn},k}}` and
    :math:`P_{21,k} = h\,e^{-h/\tau_{\mathrm{syn},k}}`.

    Membrane relative voltage :math:`y_3 = V_m - E_L` is updated as

    .. math::

       y_{3,n+1} = P_{33}\,y_{3,n} + P_{30}(I_{0,n} + I_e)
       + \sum_k \left(P_{31,k}\,y1_{k,n} + P_{32,k}\,y2_{k,n}\right),

    with :math:`P_{33} = e^{-h/\tau_m}` and
    :math:`P_{30} = \tau_m(1 - e^{-h/\tau_m})/C_m`.
    Coefficients :math:`P_{31,k}`, :math:`P_{32,k}` are computed via
    :meth:`iaf_psc_alpha._alpha_propagator_p31_p32`, which applies the stable
    near-singular limit for :math:`\tau_m \approx \tau_{\mathrm{syn},k}`:

    .. math::

       P_{32}^{\mathrm{sing}} = \frac{h}{C_m} e^{-h/\tau_m}, \qquad
       P_{31}^{\mathrm{sing}} = \frac{h^2}{2C_m} e^{-h/\tau_m},

    preventing catastrophic cancellation when :math:`\tau_m = \tau_{\mathrm{syn},k}`.

    **3. Update Order per Simulation Step (NEST Semantics)**

    Per-step execution order:

    1. Integrate membrane with exact propagator for neurons not refractory
       (:math:`r = 0`).
    2. Decrement refractory counters for neurons currently refractory
       (:math:`r > 0`).
    3. Propagate all receptor alpha states ``y1``, ``y2`` forward by one step.
    4. Inject receptor-specific spike weights into ``y1``, including default
       delta input mapped to receptor 1 when ``n_receptors > 0``.
    5. Apply threshold test, hard reset, refractory assignment, and spike
       emission.
    6. Store buffered continuous current for the next step.

    **4. Assumptions, Constraints, and Computational Implications**

    - ``C_m > 0``, ``tau_m > 0``, all ``tau_syn > 0``, ``t_ref >= 0``, and
      ``V_reset < V_th`` are enforced at construction.
    - ``update(x=...)`` uses one-step delayed current buffering: current
      provided at step ``n`` contributes through ``i_const`` at step ``n+1``,
      matching NEST ring-buffer event semantics.
    - The update path is fully vectorized over ``self.varshape`` and scales as
      :math:`O(\prod \mathrm{varshape} \times n\_receptors)` per call.
    - Internal propagator arithmetic is performed in NumPy ``float64`` before
      writing back to BrainUnit-typed states.
    - When ``n_receptors == 0``, all spike event inputs are silently ignored.

    Parameters
    ----------
    in_size : Size
        Population shape specification. Per-neuron parameters and state
        variables are broadcast/initialized over ``self.varshape`` derived
        from ``in_size``.
    E_L : ArrayLike, optional
        Resting potential :math:`E_L` in mV; scalar or array broadcastable to
        ``self.varshape``. Default is ``-70. * u.mV``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF; broadcastable to
        ``self.varshape`` and strictly positive. Default is ``250. * u.pF``.
    tau_m : ArrayLike, optional
        Membrane time constant :math:`\tau_m` in ms; broadcastable and
        strictly positive. Default is ``10. * u.ms``.
    t_ref : ArrayLike, optional
        Absolute refractory period :math:`t_\mathrm{ref}` in ms; broadcastable
        and nonnegative. Converted to integer grid steps by
        ``ceil(t_ref / dt)``. Default is ``2. * u.ms``.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_\mathrm{th}` in mV; broadcastable to
        ``self.varshape``. Default is ``-55. * u.mV``.
    V_reset : ArrayLike, optional
        Post-spike reset potential :math:`V_\mathrm{reset}` in mV;
        broadcastable and constrained by ``V_reset < V_th`` elementwise.
        Default is ``-70. * u.mV``.
    tau_syn : ArrayLike, optional
        Receptor alpha time constants in ms. Values are converted to a
        1-D ``float64`` array with shape ``(n_receptors,)``; every entry must
        be strictly positive. The number of entries defines ``n_receptors``.
        Default is ``(2.0,) * u.ms`` (one receptor).
    I_e : ArrayLike, optional
        Constant injected current :math:`I_e` in pA; scalar or array
        broadcastable to ``self.varshape``. Default is ``0. * u.pA``.
    V_min : ArrayLike or None, optional
        Optional lower clamp :math:`V_\mathrm{min}` in mV applied to the
        membrane candidate update before thresholding. ``None`` disables
        clamping. Default is ``None``.
    V_initializer : Callable, optional
        Initializer for membrane state ``V`` used by :meth:`init_state`.
        Default is ``braintools.init.Constant(-70. * u.mV)``.
    spk_fun : Callable, optional
        Surrogate spike function used by :meth:`get_spike` and returned by
        :meth:`update`. Default is ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy inherited from :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` reproduces NEST hard reset behavior. Default is ``'hard'``.
    ref_var : bool, optional
        If ``True``, allocates optional boolean state ``self.refractory`` for
        external refractory inspection. Default is ``False``.
    name : str or None, optional
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
         - Defines population/state shape ``self.varshape``.
       * - ``E_L``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``-70. * u.mV``
         - :math:`E_L`
         - Leak reversal (resting) potential.
       * - ``C_m``
         - ArrayLike, broadcastable (pF), ``> 0``
         - ``250. * u.pF``
         - :math:`C_m`
         - Membrane capacitance in subthreshold integration.
       * - ``tau_m``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``10. * u.ms``
         - :math:`\tau_m`
         - Membrane leak time constant.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), ``>= 0``
         - ``2. * u.ms``
         - :math:`t_\mathrm{ref}`
         - Absolute refractory duration in physical time.
       * - ``V_th`` and ``V_reset``
         - ArrayLike, broadcastable (mV), with ``V_reset < V_th``
         - ``-55. * u.mV``, ``-70. * u.mV``
         - :math:`V_\mathrm{th}`, :math:`V_\mathrm{reset}`
         - Threshold and post-spike reset levels.
       * - ``tau_syn``
         - ArrayLike, flattened to ``(n_receptors,)`` (ms), each ``> 0``
         - ``(2.0,) * u.ms``
         - :math:`\tau_{\mathrm{syn},k}`
         - Receptor-specific alpha time constants; length defines
           ``n_receptors``.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant current added each update step.
       * - ``V_min``
         - ArrayLike broadcastable (mV) or ``None``
         - ``None``
         - :math:`V_\mathrm{min}`
         - Optional lower clamp on candidate membrane voltage.
       * - ``V_initializer``
         - Callable
         - ``Constant(-70. * u.mV)``
         - --
         - Initializer for membrane state ``V``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate nonlinearity used for spike output.
       * - ``spk_reset``
         - str
         - ``'hard'``
         - --
         - Reset mode from :class:`~brainpy_state._base.Neuron`.
       * - ``ref_var``
         - bool
         - ``False``
         - --
         - If ``True``, exposes boolean state ``self.refractory``.
       * - ``name``
         - str | None
         - ``None``
         - --
         - Optional node name.

    Raises
    ------
    ValueError
        Raised at initialization or update time if any of the following holds:

        - ``C_m <= 0``, ``tau_m <= 0``, any ``tau_syn <= 0``, ``t_ref < 0``,
          or ``V_reset >= V_th``.
        - A spike event receptor index is outside ``[1, n_receptors]``.
    TypeError
        If parameters or inputs are not unit-compatible with the expected
        conversions (mV, ms, pF, pA).
    KeyError
        If simulation context entries (for example ``t`` or ``dt``) are
        missing when :meth:`update` is called.
    AttributeError
        If :meth:`update` is called before :meth:`init_state` creates required
        state holders.

    Notes
    -----
    - State variables are ``V``, ``y1_syn``, ``y2_syn``, ``i_const``,
      ``refractory_step_count``, and ``last_spike_time``; ``refractory`` is
      added only when ``ref_var=True``.
    - Spike weights from ``spike_events`` and ``sum_delta_inputs`` are signed
      currents in pA: positive for depolarizing, negative for hyperpolarizing
      receptors. This differs from conductance-based multisynapse models where
      weights must be non-negative.
    - ``update(x=...)`` stores ``x`` into ``i_const`` for use on the next
      step, matching NEST current-event buffering semantics.
    - If ``n_receptors == 0``, all spike event inputs are silently ignored and
      ``sum_delta_inputs`` is discarded.
    - Default delta input from ``sum_delta_inputs`` is routed to receptor 1
      when ``n_receptors > 0``, replicating NEST default port behavior.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state._nest.iaf_psc_alpha_multisynapse import (
       ...     iaf_psc_alpha_multisynapse,
       ... )
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = iaf_psc_alpha_multisynapse(
       ...         in_size=3,
       ...         tau_syn=(1.5, 3.0) * u.ms,
       ...         I_e=180.0 * u.pA,
       ...     )
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         spk = neu.update(
       ...             spike_events=[{'receptor_type': 2, 'weight': 40.0 * u.pA}]
       ...         )
       ...     _ = spk.shape

    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state._nest.iaf_psc_alpha_multisynapse import (
       ...     iaf_psc_alpha_multisynapse,
       ... )
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = iaf_psc_alpha_multisynapse(in_size=1, tau_syn=(2.0,) * u.ms)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = neu.update(x=250.0 * u.pA)
       ...     with brainstate.environ.context(t=0.1 * u.ms):
       ...         spk_next = neu.update()
       ...     _ = spk_next

    References
    ----------
    .. [1] NEST source: ``models/iaf_psc_alpha_multisynapse.h`` and
           ``models/iaf_psc_alpha_multisynapse.cpp``.
    .. [2] Rotter S, Diesmann M (1999). Exact simulation of time-invariant
           linear systems with applications to neuronal modeling. Biological
           Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    .. [3] Diesmann M, Gewaltig M-O, Rotter S, Aertsen A (2001). State space
           analysis of synchronous spiking in cortical neural networks.
           Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X
    .. [4] Morrison A, Straube S, Plesser HE, Diesmann M (2007). Exact
           subthreshold integration with continuous spike times in discrete
           time neural network simulations. Neural Computation 19(1):47-79.
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
        tau_syn: ArrayLike = (2.0,) * u.ms,
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
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_min = None if V_min is None else braintools.init.param(V_min, self.varshape)
        self.tau_syn = np.asarray(u.math.asarray(tau_syn / u.ms), dtype=np.float64).reshape(-1)
        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @property
    def n_receptors(self):
        return int(self.tau_syn.size)

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self.tau_syn <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize runtime states for membrane, synaptic, and refractory variables.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension used by
            :func:`braintools.init.param` when creating all state arrays.
            ``None`` creates unbatched states with shape ``self.varshape``.
        **kwargs : Any
            Unused compatibility arguments; accepted for interface consistency
            with other nodes.


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
        zeros = np.zeros(V.shape + (self.n_receptors,), dtype=np.float64)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.y1_syn = brainstate.ShortTermState(zeros)
        self.y2_syn = brainstate.ShortTermState(zeros * u.pA)
        self.i_const = brainstate.ShortTermState(np.zeros(V.shape, dtype=np.float64) * u.pA)
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
        out : dict
            Surrogate spike output from ``self.spk_fun`` with the same shape
            as ``V`` (or ``self.V.value`` when ``V is None``). The input to
            ``spk_fun`` is scaled as ``(V - V_th) / (V_th - V_reset)`` so
            the surrogate activates positively for suprathreshold voltages.
        """
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
        for ev in spike_events:
            if isinstance(ev, dict):
                receptor = int(ev.get('receptor_type', ev.get('receptor', 1)))
                weight = ev.get('weight', 0.0)
            else:
                receptor, weight = ev
                receptor = int(receptor)
            if receptor < 1 or receptor > self.n_receptors:
                raise ValueError(f'Receptor type {receptor} out of range [1, {self.n_receptors}].')
            w_np = np.asarray(u.math.asarray(weight / u.pA), dtype=np.float64)
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    def update(self, x=0. * u.pA, spike_events=None):
        r"""Advance the neuron by one simulation step.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous current input in pA for this step. ``x`` is accumulated
            through :meth:`sum_current_inputs` and stored in ``i_const`` for
            use on the next call (one-step delayed buffering matching NEST
            ring-buffer semantics). Default is ``0. * u.pA``.
        spike_events : iterable or None, optional
            Receptor-indexed spike weight events to inject this step. Each
            entry must be either:

            - A ``(receptor_type, weight)`` tuple where ``receptor_type`` is
              a 1-based integer in ``[1, n_receptors]`` and ``weight`` is a
              scalar or array in pA (broadcastable to ``self.varshape``).
            - A ``dict`` with keys ``'receptor_type'`` (or ``'receptor'``)
              and ``'weight'``.

            Multiple events for the same receptor are accumulated additively.
            ``None`` injects no receptor spike events. Default is ``None``.

        Returns
        -------
        out : jax.Array
            Spike output tensor from :meth:`get_spike`, shape
            ``self.V.value.shape``. On threshold crossings, the voltage
            presented to ``spk_fun`` is nudged above threshold by ``1e-12``
            mV-equivalent to preserve positive surrogate activation.

        Raises
        ------
        ValueError
            If any receptor index in ``spike_events`` is outside
            ``[1, n_receptors]``.
        KeyError
            If simulation context does not provide ``t`` or ``dt``.
        AttributeError
            If required states are missing because :meth:`init_state` was not
            called.
        TypeError
            If ``x`` or stored states are not unit-compatible with expected
            pA / mV conversions.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))
        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        V_reset_rel = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)
        lower = -np.inf * np.ones(v_shape, dtype=np.float64)
        if self.V_min is not None:
            lower = self._broadcast_to_state(self._to_numpy(self.V_min - self.E_L, u.mV), v_shape)

        y1_syn = np.asarray(self.y1_syn.value, dtype=np.float64)
        y2_syn = np.asarray(u.math.asarray(self.y2_syn.value / u.pA), dtype=np.float64)
        i_const = self._broadcast_to_state(self._to_numpy(self.i_const.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )

        P33 = np.exp(-h / tau_m)
        P30 = (1.0 - P33) * tau_m / C_m
        P11 = np.exp(-h / self.tau_syn)
        P22 = P11
        P21 = h * P11
        P31 = np.stack([
            iaf_psc_alpha._alpha_propagator_p31_p32(tau_s * np.ones(v_shape), tau_m, C_m, h)[0]
            for tau_s in self.tau_syn
        ], axis=-1)
        P32 = np.stack([
            iaf_psc_alpha._alpha_propagator_p31_p32(tau_s * np.ones(v_shape), tau_m, C_m, h)[1]
            for tau_s in self.tau_syn
        ], axis=-1)
        psc_init = math.e / self.tau_syn

        w_by_rec = self._parse_spike_events(spike_events, v_shape)
        w_default = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        if self.n_receptors > 0:
            w_by_rec[..., 0] += w_default
        i_const_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        if np.any(r == 0):
            V_candidate = P30 * (i_const + I_e) + P33 * V_rel + np.sum(P31 * y1_syn + P32 * y2_syn, axis=-1)
            V_candidate = np.maximum(V_candidate, lower)
            V_rel = np.where(r == 0, V_candidate, V_rel)
        r = np.where(r == 0, r, r - 1)

        y2_syn = P21 * y1_syn + P22 * y2_syn
        y1_syn = y1_syn * P11
        y1_syn = y1_syn + psc_init * w_by_rec

        spike_cond = V_rel >= theta
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )
        r = np.where(spike_cond, refr_counts, r)
        V_before_reset = V_rel
        V_rel = np.where(spike_cond, V_reset_rel, V_rel)

        self.V.value = (V_rel + E_L) * u.mV
        self.y1_syn.value = y1_syn
        self.y2_syn.value = y2_syn * u.pA
        self.i_const.value = i_const_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        V_out = np.where(spike_cond, theta + E_L + 1e-12, V_before_reset + E_L)
        return self.get_spike(V_out * u.mV)
