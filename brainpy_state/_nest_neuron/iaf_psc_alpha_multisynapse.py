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
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, cond_any
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
       >>> from brainpy.state import iaf_psc_alpha_multisynapse
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
       >>> from brainpy.state import iaf_psc_alpha_multisynapse
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
        dftype = brainstate.environ.dftype()
        self.tau_syn = np.asarray(u.math.asarray(tau_syn / u.ms), dtype=dftype).reshape(-1)
        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

        # Precompute refractory step count.
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    @property
    def n_receptors(self):
        return int(self.tau_syn.size)

    def _validate_parameters(self):
        """Validate model parameters against NEST constraints.

        Raises
        ------
        ValueError
            If parameter inequalities or positivity constraints are violated.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_reset, self.C_m)):
            return
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.tau_m <= 0.0 * u.ms):
            raise ValueError('Membrane time constant must be strictly positive.')
        if cond_any(self.tau_syn <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time must not be negative.')
        if cond_any(self.V_reset >= self.V_th):
            raise ValueError('Reset potential must be smaller than threshold.')

    def init_state(self, **kwargs):
        r"""Initialize runtime states for membrane, synaptic, and refractory variables.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Raises
        ------
        ValueError
            If initializers cannot broadcast to ``self.varshape``.
        TypeError
            If initializer outputs are incompatible with expected unit/array
            conversions for voltage, current, or integer refractory states.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()

        V = braintools.init.param(self.V_initializer, self.varshape)
        syn_shape = self.varshape + (self.n_receptors,)

        self.V = brainstate.HiddenState(V)
        self.y1_syn = brainstate.ShortTermState(u.math.zeros(syn_shape, dtype=dftype))
        self.y2_syn = brainstate.ShortTermState(u.math.zeros(syn_shape, dtype=dftype) * u.pA)
        self.i_const = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(
                braintools.init.param(braintools.init.Constant(False), self.varshape)
            )

        # Pre-compute propagator coefficients (constant for a given dt).
        self._precompute_propagators()

    def _precompute_propagators(self):
        """Pre-compute NEST propagator coefficients from dt and model parameters.

        Called once during ``init_state`` so that ``update`` never needs to
        recompute exponentials each step and remains JIT-compatible.
        """
        dt = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt / u.ms))
        dftype = brainstate.environ.dftype()

        tau_m = np.asarray(u.get_mantissa(self.tau_m / u.ms), dtype=np.float64)
        C_m = np.asarray(u.get_mantissa(self.C_m / u.pF), dtype=np.float64)

        self._P11 = np.exp(-h / self.tau_syn).astype(dftype)   # (n_receptors,)
        self._P22 = self._P11
        self._P21 = (h * self._P11).astype(dftype)

        P33 = np.exp(-h / tau_m)
        self._P33 = P33.astype(dftype)                         # varshape
        self._P30 = ((1.0 - P33) * tau_m / C_m).astype(dftype)

        P31_list = []
        P32_list = []
        for tau_s in self.tau_syn:
            p31, p32 = iaf_psc_alpha._alpha_propagator_p31_p32(
                tau_s * np.ones(self.varshape),
                tau_m,
                C_m,
                h,
            )
            P31_list.append(p31.astype(dftype))
            P32_list.append(p32.astype(dftype))
        self._P31 = np.stack(P31_list, axis=-1)               # varshape + (n_receptors,)
        self._P32 = np.stack(P32_list, axis=-1)

        self._psc_init = (np.e / self.tau_syn).astype(dftype)  # (n_receptors,)

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
        ditype = brainstate.environ.ditype()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        dftype = brainstate.environ.dftype()
        out = np.zeros(v_shape + (self.n_receptors,), dtype=dftype)
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
            w_np = np.asarray(u.math.asarray(weight / u.pA), dtype=dftype)
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    def update(self, x=0. * u.pA, spike_events=None, w_by_rec=None):
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
            Ignored when ``w_by_rec`` is provided.
        w_by_rec : array-like or None, optional
            Pre-computed per-receptor spike weights in pA (dimensionless),
            shape broadcastable to ``self.varshape + (n_receptors,)``. When
            provided, bypasses ``spike_events`` parsing and
            ``sum_delta_inputs``, making the update JIT-compatible for use
            inside ``brainstate.transform.for_loop``. Default is ``None``.

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
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        y1_syn = self.y1_syn.value          # unitless JAX array, varshape + (n_rec,)
        y2_syn = self.y2_syn.value          # pA
        i_const = self.i_const.value        # pA
        r = self.refractory_step_count.value  # int

        # Current input for next step (one-step delay).
        new_i_const = self.sum_current_inputs(x, V)  # pA

        # Build per-receptor spike weight array (pA values, dimensionless).
        if w_by_rec is None:
            # Python-level path: parses spike_events dicts/tuples (not JIT-compatible).
            dftype = brainstate.environ.dftype()
            v_shape = self.varshape
            w_val = self._parse_spike_events(spike_events, v_shape)  # numpy
            w_delta = np.asarray(
                u.get_mantissa(self.sum_delta_inputs(0. * u.pA) / u.pA),
                dtype=dftype,
            )
            w_delta = np.broadcast_to(w_delta, v_shape)
            if self.n_receptors > 0:
                w_val = w_val.copy()
                w_val[..., 0] += w_delta
        else:
            # JAX-array path: caller supplies pre-computed weights, JIT-compatible.
            w_val = w_by_rec  # shape broadcastable to varshape + (n_receptors,)

        # Strip units from state values using u.get_mantissa (JAX-compatible).
        y1_val = y1_syn                              # already unitless
        y2_val = u.get_mantissa(y2_syn / u.pA)      # JAX array
        i_const_val = u.get_mantissa(i_const / u.pA)
        I_e_val = u.get_mantissa(self.I_e / u.pA)
        V_rel = u.get_mantissa((V - self.E_L) / u.mV)

        # Use pre-computed propagator coefficients.
        P11 = self._P11
        P21 = self._P21
        P22 = self._P22
        P30 = self._P30
        P31 = self._P31
        P32 = self._P32
        P33 = self._P33
        psc_init = self._psc_init

        # 1) Membrane update for non-refractory neurons.
        V_candidate = (
            P30 * (i_const_val + I_e_val)
            + P33 * V_rel
            + jnp.sum(P31 * y1_val + P32 * y2_val, axis=-1)
        )
        if self.V_min is not None:
            lower = u.get_mantissa((self.V_min - self.E_L) / u.mV)
            V_candidate = u.math.maximum(V_candidate, lower)
        not_refractory = r == 0
        V_rel = jnp.where(not_refractory, V_candidate, V_rel)
        r = jnp.where(not_refractory, r, r - 1)

        # 2) Synaptic alpha state propagation.
        y2_val = P21 * y1_val + P22 * y2_val
        y1_val = y1_val * P11 + psc_init * w_val

        # 3) Threshold test, reset, and refractory assignment.
        theta_val = u.get_mantissa((self.V_th - self.E_L) / u.mV)
        V_reset_val = u.get_mantissa((self.V_reset - self.E_L) / u.mV)
        spike_cond = V_rel >= theta_val
        r = jnp.where(
            spike_cond,
            jnp.asarray(u.get_mantissa(self.ref_count), dtype=ditype),
            r,
        )
        V_before_reset = V_rel
        V_rel = jnp.where(spike_cond, V_reset_val, V_rel)

        # Write back state.
        E_L_val = u.get_mantissa(self.E_L / u.mV)
        self.V.value = (V_rel + E_L_val) * u.mV
        self.y1_syn.value = y1_val
        self.y2_syn.value = y2_val * u.pA
        self.i_const.value = new_i_const + u.math.zeros(self.varshape) * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=ditype)
        last_spike_time = u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        V_out = jnp.where(spike_cond, theta_val + E_L_val + 1e-12, V_before_reset + E_L_val)
        return self.get_spike(V_out * u.mV)
