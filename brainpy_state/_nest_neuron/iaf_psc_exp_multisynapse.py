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

from brainpy_state._nest_base._base import NESTNeuron
from brainpy_state._nest_base._utils import is_tracer, propagator_exp, cond_any

__all__ = [
    'iaf_psc_exp_multisynapse',
]


class iaf_psc_exp_multisynapse(NESTNeuron):
    r"""NEST-compatible ``iaf_psc_exp_multisynapse`` neuron model.

    Current-based leaky integrate-and-fire neuron with an arbitrary number of
    receptor-indexed exponential postsynaptic current channels.

    Description
    -----------
    ``iaf_psc_exp_multisynapse`` mirrors NEST
    ``models/iaf_psc_exp_multisynapse.{h,cpp}`` and generalizes
    :class:`iaf_psc_exp` from two fixed excitatory/inhibitory channels to
    ``n_receptors`` independently parameterized current ports, each with its
    own exponential decay time constant.

    Each receptor ``k`` (1-based, NEST convention) carries its own decay
    constant ``tau_syn[k-1]``. Synaptic weights are signed currents in pA;
    positive values are depolarizing and negative values are hyperpolarizing.

    **1. Continuous-Time Dynamics and Receptor States**

    Define :math:`V_{\mathrm{rel}} = V_m - E_L`. For receptor :math:`k`, the
    synaptic current decays exponentially:

    .. math::

       \frac{dI_k}{dt} = -\frac{I_k}{\tau_{\mathrm{syn},k}}.

    The membrane equation couples all receptor currents additively:

    .. math::

       \frac{dV_{\mathrm{rel}}}{dt}
       = -\frac{V_{\mathrm{rel}}}{\tau_m}
       + \frac{\sum_k I_k + I_e + I_0}{C_m},

    where :math:`I_0` is the one-step delayed continuous-current buffer (NEST
    ring-buffer semantics). Assumptions match NEST's current-based model:
    additive receptor currents, constant parameters within one simulation step,
    and fixed ``dt`` for exact propagator coefficients.

    **2. Exact Discrete Propagator, Derivation Constraints, and Stability**

    For step size :math:`h = dt` (ms), receptor currents are integrated
    exactly:

    .. math::

       I_{k,n+1} = P_{11,k}\, I_{k,n} + w_{k,n},
       \qquad P_{11,k} = e^{-h/\tau_{\mathrm{syn},k}},

    where :math:`w_{k,n}` is the total weight arriving at receptor :math:`k`
    during step :math:`n`.

    The membrane update uses the exact propagator:

    .. math::

       V_{\mathrm{rel},n+1}
       = P_{22}\, V_{\mathrm{rel},n}
       + P_{20}(I_e + I_{0,n})
       + \sum_k P_{21,k}\, I_{k,n},

    with propagator coefficients

    .. math::

       P_{22} = e^{-h/\tau_m}, \qquad
       P_{20} = \frac{\tau_m}{C_m}(1 - P_{22}),

    .. math::

       P_{21,k}
       = \frac{\tau_{\mathrm{syn},k}\,\tau_m}
         {C_m\,(\tau_m - \tau_{\mathrm{syn},k})}
         \left(e^{-h/\tau_m} - e^{-h/\tau_{\mathrm{syn},k}}\right).

    :func:`propagator_exp` (from ``_utils``) evaluates :math:`P_{21,k}` with a
    singular-limit fallback :math:`(h / C_m)\,e^{-h/\tau_m}` when
    :math:`\tau_{\mathrm{syn},k} \approx \tau_m`, preventing catastrophic
    cancellation in the denominator :math:`(\tau_m - \tau_{\mathrm{syn},k})`.
    Construction additionally rejects ``np.isclose(tau_syn, tau_m)`` to
    preserve robust conditioning and avoid near-degenerate parameterizations.

    **3. Update Order per Simulation Step (NEST Semantics)**

    Per-step execution order:

    1. Integrate membrane with exact propagator for neurons not refractory
       (:math:`r = 0`).
    2. Decrement refractory counters for refractory neurons (:math:`r > 0`).
    3. Decay all receptor currents :math:`I_k` by :math:`P_{11,k}`.
    4. Inject receptor-specific spike weights :math:`w_{k,n}`, including
       default delta input mapped to receptor 1 when ``n_receptors > 0``.
    5. Apply threshold test, hard reset, refractory assignment, record
       spike time, and store buffered continuous current ``x`` for step
       :math:`n+1`.

    **4. Assumptions, Constraints, and Computational Implications**

    - ``C_m > 0``, ``tau_m > 0``, all ``tau_syn > 0``,
      ``not isclose(tau_syn, tau_m)``, ``t_ref >= 0``, and
      ``V_reset < V_th`` are enforced at construction.
    - ``update(x=...)`` uses one-step delayed current buffering: current
      provided at step ``n`` contributes through ``i_const`` at step ``n+1``,
      matching NEST ring-buffer event semantics.
    - The update path is fully vectorized over ``self.varshape`` and scales
      as :math:`O(\prod \mathrm{varshape} \times n\_receptors)` per call.
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
        and nonnegative. Converted to integer grid steps via
        ``ceil(t_ref / dt)``. Default is ``2. * u.ms``.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_\mathrm{th}` in mV; broadcastable to
        ``self.varshape``. Default is ``-55. * u.mV``.
    V_reset : ArrayLike, optional
        Post-spike reset potential :math:`V_\mathrm{reset}` in mV;
        broadcastable and constrained by ``V_reset < V_th`` elementwise.
        Default is ``-70. * u.mV``.
    tau_syn : ArrayLike, optional
        Synaptic decay constants in ms for all receptor ports. Converted to a
        1-D ``float64`` array of shape ``(n_receptors,)`` via
        ``np.asarray(...).reshape(-1)``. Every entry must be strictly
        positive and must not be numerically equal to ``tau_m`` under
        ``np.isclose``. The number of entries defines ``n_receptors``.
        Default is ``(2.0,) * u.ms`` (one receptor).
    I_e : ArrayLike, optional
        Constant injected current :math:`I_e` in pA; scalar or array
        broadcastable to ``self.varshape``. Default is ``0. * u.pA``.
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
        Optional node name passed to the parent module. Default is ``None``.

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
         - ArrayLike, flattened to ``(n_receptors,)`` (ms), each ``> 0`` and
           not ``isclose`` to ``tau_m``
         - ``(2.0,) * u.ms``
         - :math:`\tau_{\mathrm{syn},k}`
         - Receptor-specific exponential PSC decay constants; number of
           entries defines ``n_receptors``.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant current added each update step.
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

        - ``V_reset >= V_th``.
        - ``C_m <= 0``, ``tau_m <= 0``, any ``tau_syn <= 0``, or ``t_ref < 0``.
        - Any ``tau_syn`` is numerically equal to ``tau_m`` under
          ``np.isclose``.
        - A spike event receptor index is outside ``[1, n_receptors]``.
    TypeError
        If parameters or inputs are not unit-compatible with expected
        conversions (mV, ms, pF, pA).
    KeyError
        If simulation context entries (for example ``t`` or ``dt``) are
        missing when :meth:`update` is called.
    AttributeError
        If :meth:`update` is called before :meth:`init_state` creates required
        state holders.

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential in mV; shape ``self.varshape``.
    i_syn : brainstate.ShortTermState
        Per-receptor synaptic currents in pA; shape
        ``self.varshape + (n_receptors,)``.
    i_const : brainstate.ShortTermState
        Buffered continuous current (pA) applied on the next simulation step.
        Shape ``self.varshape``.
    refractory_step_count : brainstate.ShortTermState
        Integer countdown of remaining refractory steps (``jnp.int32``).
        Shape ``self.varshape``.
    last_spike_time : brainstate.ShortTermState
        Simulation time of the most recent spike (ms). Shape
        ``self.varshape``.
    refractory : brainstate.ShortTermState
        Boolean refractory mask; only present when ``ref_var=True``.

    Notes
    -----
    - This implementation uses exact (analytical) integration of the linear
      subthreshold ODE via pre-computed propagator coefficients, matching
      NEST's update precision for fixed-step simulation.
    - Continuous current input ``x`` is combined with ``I_e`` and any
      additional current sources registered via :meth:`sum_current_inputs`;
      the combined value is buffered one step (NEST ring-buffer semantics).
    - Spike weights from ``spike_events`` and ``sum_delta_inputs`` are signed
      currents in pA: positive for depolarizing, negative for hyperpolarizing
      receptors.
    - Default delta input from ``sum_delta_inputs`` is routed to receptor 1
      when ``n_receptors > 0``, replicating NEST default port behavior.
    - If ``n_receptors == 0``, all spike event inputs are silently ignored and
      ``sum_delta_inputs`` is discarded.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state._nest_neuron.iaf_psc_exp_multisynapse import (
       ...     iaf_psc_exp_multisynapse,
       ... )
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = iaf_psc_exp_multisynapse(
       ...         in_size=2,
       ...         tau_syn=(2.0, 8.0) * u.ms,
       ...         I_e=180.0 * u.pA,
       ...     )
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         spk = neu.update(
       ...             spike_events=[{'receptor_type': 2, 'weight': 35.0 * u.pA}]
       ...         )
       ...     _ = spk.shape

    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state._nest_neuron.iaf_psc_exp_multisynapse import (
       ...     iaf_psc_exp_multisynapse,
       ... )
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = iaf_psc_exp_multisynapse(in_size=1, tau_syn=(2.0,) * u.ms)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = neu.update(x=250.0 * u.pA)
       ...     with brainstate.environ.context(t=0.1 * u.ms):
       ...         spk_next = neu.update()
       ...     _ = spk_next

    References
    ----------
    .. [1] Rotter S, Diesmann M (1999). Exact simulation of time-invariant
           linear systems with applications to neuronal modeling. Biological
           Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570
    .. [2] Diesmann M, Gewaltig M-O, Rotter S, Aertsen A (2001). State space
           analysis of synchronous spiking in cortical neural networks.
           Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X
    .. [3] Morrison A, Straube S, Plesser HE, Diesmann M (2007). Exact
           subthreshold integration with continuous spike times in discrete
           time neural network simulations. Neural Computation 19(1):47-79.
           DOI: https://doi.org/10.1162/neco.2007.19.1.47

    See Also
    --------
    iaf_psc_exp : LIF with two fixed exponential PSC channels (exc/inh)
    iaf_psc_alpha_multisynapse : Multisynapse variant with alpha-shaped PSCs
    iaf_psc_delta : LIF neuron with delta-function PSCs (voltage-jump synapses)
    LIF : Leaky integrate-and-fire (brainpy parameterization)
    """

    __module__ = 'brainpy.state'

    #: Unit the multi-receptor ``connect(receptor_type=k)`` bridge uses for the
    #: gathered ``w_by_rec`` mantissa (current-based -> pA).
    receptor_input_unit = u.pA

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
        dftype = brainstate.environ.dftype()
        self.tau_syn = np.asarray(u.math.asarray(tau_syn / u.ms), dtype=dftype).reshape(-1)
        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

        # Pre-compute refractory step count (matches aeif_cond_alpha pattern).
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    @property
    def n_receptors(self):
        r"""Number of independent synaptic receptor ports.

        Returns
        -------
        out : int
            Length of ``self.tau_syn``; equals ``len(tau_syn)`` as supplied
            at construction.
        """
        return int(self.tau_syn.size)

    def _validate_parameters(self):
        r"""Check parameter constraints and raise ``ValueError`` on violation.

        Validates the following conditions (all checked at construction time):

        - ``V_reset < V_th`` elementwise.
        - ``C_m > 0`` elementwise.
        - ``tau_m > 0`` elementwise.
        - All entries in ``tau_syn > 0``.
        - No entry in ``tau_syn`` is numerically equal to ``tau_m`` under
          ``np.isclose`` (prevents near-singular propagator evaluation).
        - ``t_ref >= 0`` elementwise.

        Raises
        ------
        ValueError
            On the first violated constraint, with a descriptive message.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_reset, self.C_m)):
            return
        if cond_any(self.V_reset >= self.V_th):
            raise ValueError('Reset potential must be smaller than threshold.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be > 0.')
        if cond_any(self.tau_m <= 0.0 * u.ms):
            raise ValueError('Membrane time constant must be strictly positive.')
        if cond_any(self.tau_syn <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive.')
        tau_m_ms = self.tau_m / u.ms
        if cond_any(np.isclose(self.tau_syn, tau_m_ms)):
            raise ValueError('Membrane and synapse time constants must differ.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time must not be negative.')

    def init_state(self, **kwargs):
        r"""Initialize membrane potential and all synaptic/refractory states.

        Parameters
        ----------
        **kwargs : Any
            Unused compatibility arguments; accepted for interface consistency
            with other nodes.

        Raises
        ------
        ValueError
            If ``V_initializer`` output cannot be broadcast to the target
            state shape.
        TypeError
            If initializer values are incompatible with required
            numeric/unit conversions.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()

        V = braintools.init.param(self.V_initializer, self.varshape)

        self.V = brainstate.HiddenState(V)
        self.i_syn = brainstate.ShortTermState(
            u.math.zeros(self.varshape + (self.n_receptors,), dtype=dftype) * u.pA
        )
        self.i_const = brainstate.ShortTermState(
            u.math.zeros(self.varshape, dtype=dftype) * u.pA
        )
        self.last_spike_time = brainstate.ShortTermState(
            u.math.full(self.varshape, -1e7 * u.ms)
        )
        self.refractory_step_count = brainstate.ShortTermState(
            u.math.full(self.varshape, 0, dtype=ditype)
        )

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

        self._precompute_propagators()

    def _precompute_propagators(self):
        """Pre-compute NEST propagator coefficients from dt and model parameters.

        Called once during ``init_state`` so that ``update`` never needs to
        recompute exponentials each step and remains JIT-compatible.
        """
        dt = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt / u.ms))
        dftype = brainstate.environ.dftype()

        tau_m_ms = np.asarray(u.get_mantissa(self.tau_m / u.ms), dtype=np.float64)
        C_m_pF = np.asarray(u.get_mantissa(self.C_m / u.pF), dtype=np.float64)

        # Membrane propagators.
        P22 = np.exp(-h / tau_m_ms)
        self._P22 = P22.astype(dftype)
        self._P20 = (tau_m_ms / C_m_pF * (1.0 - P22)).astype(dftype)

        # Synaptic decay.
        self._P11_syn = np.exp(-h / self.tau_syn).astype(dftype)

        # Per-receptor membrane coupling.
        P21_list = []
        for tau_s in self.tau_syn:
            P21_list.append(
                propagator_exp(
                    tau_s * np.ones(self.varshape), tau_m_ms, C_m_pF, h
                ).astype(dftype)
            )
        self._P21_syn = np.stack(P21_list, axis=-1)

        # Pre-compute constant voltage and current values.
        self._E_L_mV = np.asarray(u.get_mantissa(self.E_L / u.mV), dtype=dftype)
        self._theta_mV = np.asarray(u.get_mantissa((self.V_th - self.E_L) / u.mV), dtype=dftype)
        self._V_reset_rel_mV = np.asarray(u.get_mantissa((self.V_reset - self.E_L) / u.mV), dtype=dftype)
        self._I_e_pA = np.asarray(u.get_mantissa(self.I_e / u.pA), dtype=dftype)

    def get_spike(self, V: ArrayLike = None):
        r"""Evaluate surrogate spike activation for a voltage tensor.

        Scales the voltage relative to threshold and reset to compute a
        dimensionless argument passed to the surrogate nonlinearity
        ``self.spk_fun``:

        .. math::

           \text{out} = \mathrm{spk\_fun}\!\left(
               \frac{V - V_\mathrm{th}}{V_\mathrm{th} - V_\mathrm{reset}}
           \right).

        Parameters
        ----------
        V : ArrayLike or None, optional
            Membrane voltage in mV, broadcast-compatible with
            ``self.varshape``. If ``None``, ``self.V.value`` is used.

        Returns
        -------
        out : dict
            Surrogate spike output from ``self.spk_fun`` with the same shape
            as ``V`` (or ``self.V.value`` when ``V is None``). Positive values
            indicate a spike; the argument to ``spk_fun`` is positive when
            :math:`V > V_\mathrm{th}`.

        Raises
        ------
        TypeError
            If ``V`` cannot participate in arithmetic with membrane
            parameters due to incompatible dtype or unit.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        r"""Parse spike event descriptors into a per-receptor weight array.

        Converts a heterogeneous iterable of spike events into a contiguous
        ``float64`` NumPy array that can be added directly to ``i_syn``.

        Parameters
        ----------
        spike_events : iterable or None
            Events to parse. Each entry must be one of:

            - A ``(receptor_type, weight)`` tuple where ``receptor_type`` is
              a 1-based integer in ``[1, n_receptors]`` and ``weight`` is a
              scalar or array in pA broadcastable to ``v_shape``.
            - A ``dict`` with keys ``'receptor_type'`` (or ``'receptor'``)
              and ``'weight'``.

            Multiple events for the same receptor are accumulated additively.
            ``None`` is treated as an empty sequence.
        v_shape : tuple of int
            Shape of the neuron population state (``self.V.value.shape``).

        Returns
        -------
        out : np.ndarray
            Array of shape ``v_shape + (n_receptors,)`` with dtype
            ``float64``. Entry ``[..., k]`` is the total weight (in pA)
            arriving at receptor ``k+1`` this step.

        Raises
        ------
        ValueError
            If any ``receptor_type`` is outside ``[1, n_receptors]``.
        TypeError
            If a weight value is not unit-compatible with pA.
        """
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
        r"""Advance the neuron state by one simulation step.

        Executes the full NEST-compatible per-step update: exact membrane
        propagation for non-refractory neurons, receptor current decay and
        spike injection, threshold/reset/refractory logic, and buffered
        current storage.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous current input in pA for this step. ``x`` is accumulated
            through :meth:`sum_current_inputs` (which also adds any registered
            projection currents) and stored in ``i_const`` for use on the
            **next** step, matching NEST ring-buffer semantics. Scalar or
            array broadcastable to ``self.varshape``. Default is
            ``0. * u.pA``.
        spike_events : iterable or None, optional
            Receptor-indexed spike weight events to inject this step. Each
            entry must be either:

            - A ``(receptor_type, weight)`` tuple where ``receptor_type`` is
              a 1-based integer in ``[1, n_receptors]`` and ``weight`` is a
              scalar or array in pA broadcastable to ``self.varshape``.
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
            Surrogate spike output from :meth:`get_spike` with shape
            ``self.V.value.shape``. For neurons that fire this step, the
            voltage argument to :meth:`get_spike` is nudged
            :math:`\theta + E_L + 10^{-12}` mV (above threshold) to ensure a
            positive surrogate activation is returned even after the hard
            voltage reset.

        Raises
        ------
        ValueError
            If any receptor index in ``spike_events`` is outside
            ``[1, n_receptors]``.
        KeyError
            If the simulation environment context does not supply ``t`` or
            ``dt``.
        AttributeError
            If state variables are missing because :meth:`init_state` has not
            been called before ``update``.
        TypeError
            If ``x`` or stored states are not unit-compatible with expected
            pA / mV arithmetic.
        ValueError
            If provided inputs cannot be broadcast to the internal state
            shape.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        i_syn = self.i_syn.value  # pA, shape varshape + (n_receptors,)
        i_const = self.i_const.value  # pA
        r = self.refractory_step_count.value  # int

        # Use pre-computed constants (avoids recomputing exponentials each step).
        I_e_pA = self._I_e_pA
        E_L_mV = self._E_L_mV
        theta_mV = self._theta_mV
        V_reset_rel_mV = self._V_reset_rel_mV
        P22 = self._P22
        P20 = self._P20
        P11_syn = self._P11_syn
        P21_syn = self._P21_syn

        # Strip units (JAX-compatible via u.get_mantissa).
        i_const_pA = u.get_mantissa(i_const / u.pA)
        V_rel_mV = u.get_mantissa((V - self.E_L) / u.mV)
        i_syn_pA = u.get_mantissa(i_syn / u.pA)

        # Build per-receptor spike weight array.
        if w_by_rec is None:
            # Python-level path: parses spike_events dicts/tuples (not JIT-compatible).
            dftype = brainstate.environ.dftype()
            v_shape = self.V.value.shape
            w_val = self._parse_spike_events(spike_events, v_shape)
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
            w_val = w_by_rec

        # Current input for next step (one-step delay).
        new_i_const = self.sum_current_inputs(x, self.V.value)  # pA

        # 1. Membrane integration for non-refractory neurons.
        not_refractory = r == 0
        V_candidate = (
            V_rel_mV * P22
            + (I_e_pA + i_const_pA) * P20
            + jnp.sum(P21_syn * i_syn_pA, axis=-1)
        )
        V_rel_mV = jnp.where(not_refractory, V_candidate, V_rel_mV)

        # 2. Decrement refractory counters.
        r = jnp.where(not_refractory, r, r - 1)

        # 3. Decay receptor currents and inject spike weights.
        i_syn_pA = i_syn_pA * P11_syn
        i_syn_pA = i_syn_pA + w_val

        # 4. Threshold test, reset, refractory assignment.
        spike_cond = V_rel_mV >= theta_mV
        r = jnp.where(spike_cond, jnp.asarray(u.get_mantissa(self.ref_count), dtype=ditype), r)
        V_before_reset = V_rel_mV
        V_rel_mV = jnp.where(spike_cond, V_reset_rel_mV, V_rel_mV)

        # Write back state.
        self.V.value = (V_rel_mV + E_L_mV) * u.mV
        self.i_syn.value = i_syn_pA * u.pA
        self.i_const.value = new_i_const + u.math.zeros(self.varshape) * u.pA
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        last_spike_time = u.math.where(spike_cond, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        V_out = jnp.where(spike_cond, theta_mV + E_L_mV + 1e-12, V_before_reset + E_L_mV)
        return self.get_spike(V_out * u.mV)
