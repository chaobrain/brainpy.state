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
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, propagator_exp, cond_any

__all__ = [
    'iaf_psc_exp',
]


class iaf_psc_exp(NESTNeuron):
    r"""NEST-compatible ``iaf_psc_exp`` neuron model.

    Description
    -----------

    ``iaf_psc_exp`` is a current-based leaky integrate-and-fire neuron with
    hard reset, fixed absolute refractory period, and exponential excitatory
    and inhibitory postsynaptic currents. The implementation follows NEST
    ``models/iaf_psc_exp.{h,cpp}`` update order, including one-step buffered
    current input and receptor-1 filtered current handling.

    **1. Continuous-Time Dynamics**


    The subthreshold membrane equation is

    .. math::

       \frac{dV_m}{dt} = -\frac{V_m - E_L}{\tau_m}
       + \frac{I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}} + I_e + I_0}{C_m}

    where :math:`I_0` is the buffered current from the previous simulation
    step. Synaptic currents decay exponentially:

    .. math::

       \frac{dI_{\mathrm{syn,ex}}}{dt} = -\frac{I_{\mathrm{syn,ex}}}{\tau_{\mathrm{syn,ex}}}
       \qquad
       \frac{dI_{\mathrm{syn,in}}}{dt} = -\frac{I_{\mathrm{syn,in}}}{\tau_{\mathrm{syn,in}}}.

    NEST also defines a second current receptor :math:`I_1` that is filtered
    through the excitatory kernel; this is exposed via
    ``update(x_filtered=...)``.

    **2. Exact Step Propagator and NEST Update Ordering**


    For time step :math:`h = dt` (in ms), exact exponentials are used for
    all linear sub-systems:

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

    where :math:`P_{21}` is evaluated numerically stably by
    :func:`~brainpy_state._nest_base.utils.propagator_exp`. Let :math:`V_\mathrm{rel} = V_m - E_L`.
    The candidate membrane update is

    .. math::

       V_{\mathrm{rel},n+1} =
       P_{22} V_{\mathrm{rel},n}
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

    **3. Escape-Noise Threshold Dynamics**


    Deterministic thresholding is used when :math:`\delta < 10^{-10}`:
    :math:`V_{\mathrm{rel}} \ge \theta`, where
    :math:`\theta = V_{th} - E_L`.

    For :math:`\delta > 0`, the model uses an exponential hazard:

    .. math::

       \phi(V) = \rho \exp\!\left(\frac{V_{\mathrm{rel}} - \theta}{\delta}\right),

    and spikes with step probability :math:`p = \phi(V)\,h\times10^{-3}`
    because :math:`\phi` is in ``1/s`` while ``h`` is in ms. Stochastic
    decisions use ``numpy.random.random``.

    **4. Stability Constraints and Computational Implications**


    - Construction enforces ``V_reset < V_th``, ``C_m > 0``, ``tau_m > 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, ``t_ref >= 0``, ``rho >= 0``,
      and ``delta >= 0``.
    - :func:`~brainpy_state._nest_base.utils.propagator_exp` uses a singular fallback
      :math:`(h/C_m)\exp(-h/\tau_m)` when ``tau_syn`` is numerically close
      to ``tau_m``, avoiding cancellation in
      :math:`(e^{-h/\tau_m} - e^{-h/\tau_{\mathrm{syn}}})/(\tau_m - \tau_{\mathrm{syn}})`.
    - Per-call cost is :math:`O(\prod \mathrm{varshape})` with vectorized
      NumPy operations in ``float64`` for coefficient evaluation.
    - Buffered current semantics match NEST ring-buffer timing:
      ``x``/``x_filtered`` supplied at step ``n`` are stored and consumed at
      step ``n+1``.

    Parameters
    ----------
    in_size : Size
        Population shape specification. All per-neuron parameters are
        broadcast to ``self.varshape`` derived from ``in_size``.
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
        If ``True``, allocates ``self.refractory`` (boolean array) for
        external inspection of the refractory state. Default is ``False``.
    name : str or None, optional
        Optional node name passed to the parent module. Default is ``None``.

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
         - ArrayLike, broadcastable; ``rho`` in ``1/s``, ``delta`` in mV,
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
        constants, negative ``t_ref``, negative ``rho``, or negative
        ``delta``.

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential in mV; shape ``self.varshape``.
    i_syn_ex : brainstate.ShortTermState
        Excitatory synaptic current in pA.
    i_syn_in : brainstate.ShortTermState
        Inhibitory synaptic current in pA.
    i_0 : brainstate.ShortTermState
        Buffered receptor-0 current (pA) applied on the next simulation step.
    i_1 : brainstate.ShortTermState
        Buffered receptor-1 current (pA) filtered through the excitatory
        exponential kernel on the next simulation step.
    refractory_step_count : brainstate.ShortTermState
        Integer countdown of remaining refractory steps (``jnp.int32``).
    last_spike_time : brainstate.ShortTermState
        Simulation time of the most recent spike (ms).
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
    - Delta spike inputs from :meth:`sum_delta_inputs` are split by sign:
      positive weights increment ``i_syn_ex``; negative weights increment
      ``i_syn_in``.
    - The stochastic escape-noise mode (``delta > 0``) uses
      ``numpy.random.random`` and is therefore **not** JIT-compilable via
      JAX. Use ``delta=0`` for fully differentiable, JIT-compatible runs.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state._nest_neuron.iaf_psc_exp import iaf_psc_exp
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = iaf_psc_exp(in_size=(3,), I_e=250. * u.pA, delta=0. * u.mV)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         out = neu.update(x=0. * u.pA, x_filtered=0. * u.pA)
       ...     _ = out.shape

    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as u
       >>> from brainpy_state._nest_neuron.iaf_psc_exp import iaf_psc_exp
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = iaf_psc_exp(
       ...         in_size=10,
       ...         tau_syn_ex=2.0 * u.ms,
       ...         tau_syn_in=5.0 * u.ms,
       ...         ref_var=True,
       ...     )
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         spk = neu.update(x=300.0 * u.pA)
       ...     _ = spk.shape

    References
    ----------
    .. [1] Rotter S, Diesmann M (1999). Exact simulation of time-invariant
           linear systems with applications to neuronal modeling. Biological
           Cybernetics 81:381-402. DOI: https://doi.org/10.1007/s004220050570
    .. [2] Diesmann M, Gewaltig M-O, Rotter S, & Aertsen A (2001). State
           space analysis of synchronous spiking in cortical neural networks.
           Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X
    .. [3] Brette R, Rudolph M, Carnevale T, et al. (2007). Simulation of
           networks of spiking neurons: a review of tools and strategies.
           Journal of Computational Neuroscience 23:349-398.
           DOI: https://doi.org/10.1007/s10827-007-0038-6

    See Also
    --------
    iaf_psc_delta : LIF neuron with delta-function PSCs (voltage-jump synapses)
    iaf_cond_exp : LIF neuron with exponential conductance synapses
    LIF : Leaky integrate-and-fire (brainpy parameterization)
    LIFRef : Leaky integrate-and-fire with explicit refractory tracking
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

        # Precompute refractory step count.
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    def _validate_parameters(self):
        r"""Validate model parameters against NEST constraints.

        Raises
        ------
        ValueError
            If parameter inequalities or positivity constraints are violated.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_reset, self.C_m)):
            return
        if cond_any(self.V_reset >= self.V_th):
            raise ValueError('Reset potential must be smaller than threshold.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.tau_m <= 0.0 * u.ms):
            raise ValueError('Membrane time constant must be strictly positive.')
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms) or cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('Synaptic time constants must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time must not be negative.')
        if cond_any(self.rho < 0.0 / u.second):
            raise ValueError('Stochastic firing intensity rho must not be negative.')
        if cond_any(self.delta < 0.0 * u.mV):
            raise ValueError('Threshold width delta must not be negative.')

    def init_state(self, **kwargs):
        r"""Initialize membrane potential and all synaptic/refractory states.

        Parameters
        ----------
        **kwargs : Any
            Unused compatibility arguments.

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
        zeros_pA = u.math.zeros(self.varshape, dtype=dftype) * u.pA

        self.V = brainstate.HiddenState(V)
        self.i_syn_ex = brainstate.ShortTermState(zeros_pA)
        self.i_syn_in = brainstate.ShortTermState(zeros_pA)
        self.i_0 = brainstate.ShortTermState(zeros_pA)
        self.i_1 = brainstate.ShortTermState(zeros_pA)
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

        # Pre-compute propagator coefficients (constant for a given dt).
        self._precompute_propagators()

    def _precompute_propagators(self):
        """Pre-compute NEST propagator coefficients from dt and model parameters."""
        dt_q = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        h = float(u.math.asarray(dt_q / u.ms))

        tau_ex_np = np.asarray(u.math.asarray(self.tau_syn_ex / u.ms), dtype=dftype)
        tau_in_np = np.asarray(u.math.asarray(self.tau_syn_in / u.ms), dtype=dftype)
        tau_m_np = np.asarray(u.math.asarray(self.tau_m / u.ms), dtype=dftype)
        C_m_np = np.asarray(u.math.asarray(self.C_m / u.pF), dtype=dftype)

        self._P11_ex = jnp.asarray(np.exp(-h / tau_ex_np))
        self._P11_in = jnp.asarray(np.exp(-h / tau_in_np))
        self._P22 = jnp.asarray(np.exp(-h / tau_m_np))
        self._P21_ex = jnp.asarray(propagator_exp(tau_ex_np, tau_m_np, C_m_np, h))
        self._P21_in = jnp.asarray(propagator_exp(tau_in_np, tau_m_np, C_m_np, h))
        self._P20 = jnp.asarray(tau_m_np / C_m_np * (1.0 - np.exp(-h / tau_m_np)))
        self._h = h

        # Pre-compute stochastic threshold parameters.
        self._delta_np = jnp.asarray(np.asarray(u.math.asarray(self.delta / u.mV), dtype=dftype))
        self._rho_np = jnp.asarray(np.asarray(u.math.asarray(self.rho * u.second), dtype=dftype))
        self._deterministic = self._delta_np < 1e-10
        self._delta_safe = jnp.where(self._deterministic, 1.0, self._delta_np)

    def get_spike(self, V: ArrayLike = None):
        r"""Evaluate surrogate spike activation for a voltage tensor.

        Scales the voltage relative to threshold and reset to compute a
        dimensionless argument passed to the surrogate nonlinearity
        ``self.spk_fun``:

        .. math::

           \text{out} = \mathrm{spk\_fun}\!\left(
               \frac{V - V_{th}}{V_{th} - V_{reset}}
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
            as ``V`` (or ``self.V.value`` when ``V`` is ``None``).

        Raises
        ------
        TypeError
            If ``V`` cannot participate in arithmetic with membrane
            parameters due to incompatible dtype or unit.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.pA, x_filtered=0. * u.pA):
        r"""Advance the neuron state by one simulation step.

        Parameters
        ----------
        x : ArrayLike, optional
            Current input in pA for receptor-0 (standard current port). Scalar
            or array broadcastable to ``self.varshape``. The value is buffered
            (stored in ``self.i_0``) and applied in the **next** step, matching
            NEST ring-buffer semantics. Default is ``0. * u.pA``.
        x_filtered : ArrayLike, optional
            Current input in pA for receptor-1. Buffered in ``self.i_1`` and
            injected through excitatory exponential filtering at the next
            update step via ``(1 - P_{11,\mathrm{ex}}) \times i_1``. Scalar
            or array broadcastable to ``self.varshape``.
            Default is ``0. * u.pA``.

        Returns
        -------
        out : jax.Array
            Surrogate spike output from :meth:`get_spike` with shape
            ``self.V.value.shape``. For neurons that fire this step, the
            voltage argument to :meth:`get_spike` is nudged
            :math:`\theta + E_L + 10^{-12}\,\text{mV}` (above threshold) to
            ensure a positive surrogate activation is returned even after the
            hard voltage reset.

        Raises
        ------
        KeyError
            If the simulation environment context does not supply ``t`` or
            ``dt``.
        AttributeError
            If state variables are missing because :meth:`init_state` has not
            been called before ``update``.
        TypeError
            If input/state values are not unit-compatible with expected pA/mV
            arithmetic.
        ValueError
            If provided inputs cannot be broadcast to the internal state shape.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        h = self._h

        # Read state variables with their natural units.
        V = self.V.value  # mV
        i_syn_ex = self.i_syn_ex.value  # pA
        i_syn_in = self.i_syn_in.value  # pA
        i_0 = self.i_0.value  # pA
        i_1 = self.i_1.value  # pA
        r = self.refractory_step_count.value  # int

        # Use pre-computed propagator coefficients.
        P11_ex = self._P11_ex
        P11_in = self._P11_in
        P22 = self._P22
        P21_ex = self._P21_ex
        P21_in = self._P21_in
        P20 = self._P20

        # Relative voltages and thresholds (unit-aware).
        V_rel = V - self.E_L  # mV
        theta = self.V_th - self.E_L  # mV
        V_reset_rel = self.V_reset - self.E_L  # mV

        # 1. Update membrane potential if not refractory.
        not_refractory = r == 0
        # P21 coefficients have units ms/pF which, multiplied by pA, yield mV.
        # P22 is unitless, P20 has units ms/pF * pA = mV.
        V_candidate = (
            V_rel * P22
            + i_syn_ex * (P21_ex * (u.mV / u.pA))
            + i_syn_in * (P21_in * (u.mV / u.pA))
            + (self.I_e + i_0) * (P20 * (u.mV / u.pA))
        )
        V_rel = u.math.where(not_refractory, V_candidate, V_rel)
        r = u.math.where(not_refractory, r, r - 1)

        # 2. Decay synaptic currents.
        i_syn_ex = i_syn_ex * P11_ex
        i_syn_in = i_syn_in * P11_in

        # 3. Receptor type 1 current filtered through excitatory synapse.
        i_syn_ex = i_syn_ex + (1.0 - P11_ex) * i_1

        # 4. Add arriving spikes (positive -> excitatory, negative -> inhibitory).
        w_all = self.sum_delta_inputs(u.math.zeros_like(self.i_syn_ex.value))
        w_ex = u.math.where(w_all > 0.0 * u.pA, w_all, 0.0 * u.pA)
        w_in = u.math.where(w_all < 0.0 * u.pA, w_all, 0.0 * u.pA)
        i_syn_ex = i_syn_ex + w_ex
        i_syn_in = i_syn_in + w_in

        # Buffered current inputs for next step (one-step delay).
        new_i_0 = self.sum_current_inputs(x, self.V.value)
        new_i_1 = u.math.asarray(x_filtered) + u.math.zeros(self.varshape) * u.pA

        # 5. Threshold test, reset and refractory assignment.
        # Deterministic thresholding when delta < 1e-10 mV.
        det_spike = V_rel >= theta

        # Stochastic escape-noise: phi * h * 1e-3 (phi in 1/s, h in ms).
        V_rel_np_val = u.math.asarray(V_rel / u.mV)
        theta_np_val = u.math.asarray(theta / u.mV)
        phi = self._rho_np * jnp.exp((V_rel_np_val - theta_np_val) / self._delta_safe)
        stoch_spike = jnp.asarray(np.random.random(size=self.varshape)) < phi * h * 1e-3

        spike_cond = jnp.where(self._deterministic, det_spike, stoch_spike)

        r = u.math.where(spike_cond, self.ref_count, r)
        V_before_reset = V_rel
        V_rel = u.math.where(spike_cond, V_reset_rel, V_rel)

        # 6. Write back state.
        self.V.value = V_rel + self.E_L
        self.i_syn_ex.value = i_syn_ex
        self.i_syn_in.value = i_syn_in
        self.i_0.value = new_i_0 + u.math.zeros(self.varshape) * u.pA
        self.i_1.value = new_i_1
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        last_spike_time = u.math.where(spike_cond, t + dt_q, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        # For surrogate spike output, nudge voltage above threshold on spike.
        V_out = u.math.where(spike_cond, theta + self.E_L + 1e-12 * u.mV, V_before_reset + self.E_L)
        return self.get_spike(V_out)
