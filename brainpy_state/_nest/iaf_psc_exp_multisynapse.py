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

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest._base import NESTNeuron
from .iaf_psc_exp import iaf_psc_exp

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

    :meth:`iaf_psc_exp._propagator_exp` evaluates :math:`P_{21,k}` with a
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
        Membrane potential in mV; shape ``self.varshape`` (or
        ``(batch_size,) + self.varshape`` when batched).
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
       >>> from brainpy_state._nest.iaf_psc_exp_multisynapse import (
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
       >>> from brainpy_state._nest.iaf_psc_exp_multisynapse import (
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
        self.tau_syn = np.asarray(u.math.asarray(tau_syn / u.ms), dtype=np.float64).reshape(-1)
        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

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

    @staticmethod
    def _to_numpy(x, unit):
        r"""Strip a BrainUnit quantity and return a ``float64`` NumPy array.

        Parameters
        ----------
        x : ArrayLike
            Input value with units compatible with ``unit``.
        unit : brainunit.Unit
            Physical unit to divide out (e.g. ``u.mV``, ``u.ms``, ``u.pA``).

        Returns
        -------
        out : np.ndarray
            Dimensionless ``float64`` array with the same shape as ``x``.

        Raises
        ------
        TypeError
            If ``x`` is not unit-compatible with ``unit``.
        """
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast a NumPy array to the neuron state shape.

        Parameters
        ----------
        x_np : np.ndarray
            Array to broadcast; must be compatible with ``shape``.
        shape : tuple of int
            Target shape (typically ``self.V.value.shape``).

        Returns
        -------
        out : np.ndarray
            View of ``x_np`` broadcast to ``shape`` without copying data.

        Raises
        ------
        ValueError
            If ``x_np`` cannot be broadcast to ``shape`` following NumPy
            broadcasting rules.
        """
        return np.broadcast_to(x_np, shape)

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
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be > 0.')
        if np.any(self._to_numpy(self.tau_m, u.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self.tau_syn <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive.')
        if np.any(np.isclose(self.tau_syn, self._to_numpy(self.tau_m, u.ms))):
            raise ValueError('Membrane and synapse time constants must differ.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize membrane potential and all synaptic/refractory states.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, states have shape
            ``self.varshape``; otherwise ``(batch_size,) + self.varshape``.
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
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = np.zeros(V.shape + (self.n_receptors,), dtype=np.float64)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.i_syn = brainstate.ShortTermState(zeros * u.pA)
        self.i_const = brainstate.ShortTermState(np.zeros(V.shape, dtype=np.float64) * u.pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(u.math.asarray(ref_steps > 0, dtype=bool))

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

    def _refractory_counts(self):
        r"""Convert refractory duration to integer simulation-step counts.

        Computes :math:`\lceil t_\mathrm{ref} / dt \rceil` using the current
        simulation step size from the environment context, matching NEST's
        grid-step rounding convention.

        Returns
        -------
        out : jnp.ndarray
            Integer array (``jnp.int32``) broadcast-compatible with
            ``self.varshape``; value is ``ceil(self.t_ref / dt)``.

        Raises
        ------
        KeyError
            If simulation context does not provide ``dt``.
        TypeError
            If ``t_ref`` and ``dt`` are not unit-compatible for division.
        """
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

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

        i_syn = np.asarray(u.math.asarray(self.i_syn.value / u.pA), dtype=np.float64)
        i_const = self._broadcast_to_state(self._to_numpy(self.i_const.value, u.pA), v_shape)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        )

        P22 = np.exp(-h / tau_m)
        P20 = tau_m / C_m * (1.0 - P22)
        P11_syn = np.exp(-h / self.tau_syn)
        P21_syn = np.stack([
            iaf_psc_exp._propagator_exp(tau_s * np.ones(v_shape), tau_m, C_m, h) for tau_s in self.tau_syn
        ], axis=-1)

        w_by_rec = self._parse_spike_events(spike_events, v_shape)
        w_default = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        if self.n_receptors > 0:
            w_by_rec[..., 0] += w_default
        i_const_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        if np.any(r == 0):
            V_candidate = V_rel * P22 + (I_e + i_const) * P20 + np.sum(P21_syn * i_syn, axis=-1)
            V_rel = np.where(r == 0, V_candidate, V_rel)
        r = np.where(r == 0, r, r - 1)

        i_syn = i_syn * P11_syn
        i_syn = i_syn + w_by_rec

        spike_cond = V_rel >= theta
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )
        r = np.where(spike_cond, refr_counts, r)
        V_before_reset = V_rel
        V_rel = np.where(spike_cond, V_reset_rel, V_rel)

        self.V.value = (V_rel + E_L) * u.mV
        self.i_syn.value = i_syn * u.pA
        self.i_const.value = i_const_next * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        )
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        V_out = np.where(spike_cond, theta + E_L + 1e-12, V_before_reset + E_L)
        return self.get_spike(V_out * u.mV)
