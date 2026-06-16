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
from brainstate.util import DotDict

from brainpy_state._nest_base._base import NESTNeuron
from brainpy_state._nest_base._utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'iaf_cond_alpha',
]


class iaf_cond_alpha(NESTNeuron):
    r"""Leaky integrate-and-fire model with alpha-shaped conductance synapses.

    Description
    -----------

    ``iaf_cond_alpha`` is a conductance-based leaky integrate-and-fire neuron with

    * hard threshold,
    * fixed absolute refractory period,
    * alpha-shaped excitatory and inhibitory synaptic conductances (second-order kinetics),
    * no adaptation variables.

    This implementation follows NEST ``iaf_cond_alpha`` dynamics and update order,
    using NEST C++ model behavior as the source of truth.

    **1. Membrane Potential and Synaptic Currents**

    The membrane potential evolves according to

    .. math::

       \frac{dV_\mathrm{m}}{dt} =
       \frac{-g_\mathrm{L}(V_\mathrm{m}-E_\mathrm{L})
             - I_\mathrm{syn}
             + I_\mathrm{e}
             + I_\mathrm{stim}}
            {C_\mathrm{m}}

    with

    .. math::

       I_\mathrm{syn}
       = I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}}
       = g_\mathrm{ex}(V_\mathrm{m}-E_\mathrm{ex})
       + g_\mathrm{in}(V_\mathrm{m}-E_\mathrm{in}) .

    **2. Alpha-Shaped Conductance Kinetics**

    Alpha conductances use two coupled state variables per channel:

    .. math::

       \frac{d\,dg_\mathrm{ex}}{dt} = -\frac{dg_\mathrm{ex}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{d g_\mathrm{ex}}{dt}
       = dg_\mathrm{ex} - \frac{g_\mathrm{ex}}{\tau_{\mathrm{syn,ex}}},

    .. math::

       \frac{d\,dg_\mathrm{in}}{dt} = -\frac{dg_\mathrm{in}}{\tau_{\mathrm{syn,in}}},
       \qquad
       \frac{d g_\mathrm{in}}{dt}
       = dg_\mathrm{in} - \frac{g_\mathrm{in}}{\tau_{\mathrm{syn,in}}}.

    A presynaptic spike with weight :math:`w` causes an instantaneous jump at
    the end of the simulation step. Positive/negative weights map to
    excitatory/inhibitory channels:

    .. math::

       w > 0 \Rightarrow dg_\mathrm{ex} \leftarrow dg_\mathrm{ex} + \frac{e}{\tau_{\mathrm{syn,ex}}} w,

    .. math::

       w < 0 \Rightarrow dg_\mathrm{in} \leftarrow dg_\mathrm{in} + \frac{e}{\tau_{\mathrm{syn,in}}} |w|.

    The normalization factor :math:`e/\tau` ensures the conductance peak matches
    the weight magnitude (in nS).

    **3. Spike Emission and Refractory Mechanism**

    A spike is emitted when :math:`V_\mathrm{m} \ge V_\mathrm{th}` at the end of
    a simulation step. On spike:

    * :math:`V_\mathrm{m}` is reset to :math:`V_\mathrm{reset}`,
    * refractory counter is set to :math:`\lceil t_\mathrm{ref}/dt \rceil`,
    * spike time is recorded as :math:`t + dt`.

    During absolute refractory period:

    * effective membrane potential in current computation is clamped to :math:`V_\mathrm{reset}`,
    * :math:`dV_\mathrm{m}/dt = 0`,
    * conductances continue to decay.

    **4. Numerical Integration and Update Order**

    NEST integrates this model with adaptive RKF45. This implementation mirrors
    that behavior with an RKF45(4,5) integrator and persistent internal step size.
    The discrete-time update order is:

    1. Integrate continuous dynamics on :math:`(t, t+dt]` using RKF45 with adaptive substeps.
    2. Apply refractory countdown / threshold test / reset and spike emission.
    3. Add synaptic conductance jumps from spike inputs arriving this step.
    4. Store external current input as :math:`I_\mathrm{stim}` for the next step.

    The one-step delayed application of current input (``I_stim`` buffer) is
    intentional and matches NEST's ring-buffer update semantics.

    Parameters
    ----------
    in_size : tuple of int or int
        Shape of the neuron population. Can be an integer for 1D populations or
        a tuple for multi-dimensional populations.
    E_L : ArrayLike, optional
        Leak reversal potential :math:`E_\mathrm{L}`. Default: -70 mV.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_\mathrm{m}`. Must be strictly positive.
        Default: 250 pF.
    t_ref : ArrayLike, optional
        Absolute refractory period :math:`t_\mathrm{ref}`. Must be non-negative.
        Default: 2 ms.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_\mathrm{th}`. Must be larger than ``V_reset``.
        Default: -55 mV.
    V_reset : ArrayLike, optional
        Reset potential :math:`V_\mathrm{reset}`. Must be smaller than ``V_th``.
        Default: -60 mV.
    E_ex : ArrayLike, optional
        Excitatory reversal potential :math:`E_\mathrm{ex}`. Default: 0 mV.
    E_in : ArrayLike, optional
        Inhibitory reversal potential :math:`E_\mathrm{in}`. Default: -85 mV.
    g_L : ArrayLike, optional
        Leak conductance :math:`g_\mathrm{L}`. Must be strictly positive.
        Default: 16.6667 nS (yields :math:`\tau_\mathrm{m} = 15` ms with default ``C_m``).
    tau_syn_ex : ArrayLike, optional
        Excitatory alpha time constant :math:`\tau_{\mathrm{syn,ex}}`. Must be
        strictly positive. Default: 0.2 ms.
    tau_syn_in : ArrayLike, optional
        Inhibitory alpha time constant :math:`\tau_{\mathrm{syn,in}}`. Must be
        strictly positive. Default: 2.0 ms.
    I_e : ArrayLike, optional
        Constant external current :math:`I_\mathrm{e}`. Default: 0 pA.
    gsl_error_tol : ArrayLike, optional
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
        Default: 1e-3.
    V_initializer : Callable, optional
        Initializer for membrane potential. Default: Constant(-70 mV).
    g_ex_initializer : Callable, optional
        Initializer for excitatory conductance. Default: Constant(0 nS).
    g_in_initializer : Callable, optional
        Initializer for inhibitory conductance. Default: Constant(0 nS).
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation (differentiable approximation).
        Default: ReluGrad().
    spk_reset : str, optional
        Spike reset mode. ``'hard'`` uses stop_gradient (matches NEST behavior),
        ``'soft'`` allows gradients through reset. Default: ``'hard'``.
    ref_var : bool, optional
        If True, expose ``refractory`` state variable as boolean indicator.
        Default: False.
    name : str, optional
        Name of the neuron group.

    Parameter Mapping
    -----------------

    ==================== ================== ========================================
    **Parameter**        **Default**        **Math equivalent**
    ==================== ================== ========================================
    ``in_size``          (required)         --
    ``E_L``              -70 mV             :math:`E_\mathrm{L}`
    ``C_m``              250 pF             :math:`C_\mathrm{m}`
    ``t_ref``            2 ms               :math:`t_\mathrm{ref}`
    ``V_th``             -55 mV             :math:`V_\mathrm{th}`
    ``V_reset``          -60 mV             :math:`V_\mathrm{reset}`
    ``E_ex``             0 mV               :math:`E_\mathrm{ex}`
    ``E_in``             -85 mV             :math:`E_\mathrm{in}`
    ``g_L``              16.6667 nS         :math:`g_\mathrm{L}`
    ``tau_syn_ex``       0.2 ms             :math:`\tau_{\mathrm{syn,ex}}`
    ``tau_syn_in``       2.0 ms             :math:`\tau_{\mathrm{syn,in}}`
    ``I_e``              0 pA               :math:`I_\mathrm{e}`
    ``gsl_error_tol``    1e-3               --
    ``V_initializer``    Constant(-70 mV)   --
    ``g_ex_initializer`` Constant(0 nS)     --
    ``g_in_initializer`` Constant(0 nS)     --
    ``spk_fun``          ReluGrad()         --
    ``spk_reset``        ``'hard'``         --
    ``ref_var``          ``False``          --
    ==================== ================== ========================================

    State Variables
    ---------------

    ========================= ================================================================
    **State variable**        **Description**
    ========================= ================================================================
    ``V``                     Membrane potential :math:`V_\mathrm{m}`
    ``dg_ex``                 Excitatory alpha auxiliary state
    ``g_ex``                  Excitatory conductance :math:`g_\mathrm{ex}`
    ``dg_in``                 Inhibitory alpha auxiliary state
    ``g_in``                  Inhibitory conductance :math:`g_\mathrm{in}`
    ``last_spike_time``       Last spike time (recorded at :math:`t+dt`)
    ``refractory_step_count`` Remaining refractory grid steps
    ``integration_step``      Internal RKF45 step-size state (persistent)
    ``I_stim``                Buffered current applied in next step
    ``refractory``            Optional boolean refractory indicator (if ``ref_var=True``)
    ========================= ================================================================

    **Sends:**
    ``SpikeEvent`` (conceptually; represented as returned spike tensor from ``update``).

    **Receives:**
    Signed spike-weight conductance increments through ``add_delta_input``.
    - External current input through ``x`` in :meth:`update` (one-step delayed).

    Raises
    ------
    ValueError
        If ``V_reset >= V_th``, ``C_m <= 0``, ``t_ref < 0``, or any time constants
        are non-positive.

    Notes
    -----

    - Defaults follow NEST C++ source for ``iaf_cond_alpha`` (``models/iaf_cond_alpha.h/.cpp``).
    - Synaptic spike weights are interpreted in conductance units (nS), with
      positive/negative sign selecting excitatory/inhibitory channel.
    - The alpha shape produces a smoother conductance transient than single exponentials,
      peaking at :math:`t = \tau` after a spike.
    - During refractory period, the effective voltage used for current computation is
      clamped, but the actual ``V`` state continues to be updated (remains at reset value).

    References
    ----------
    .. [1] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. Journal of Computational Neuroscience,
           16:159-175. DOI: https://doi.org/10.1023/B:JCNS.0000014108.03012.81
    .. [2] Bernander O, Douglas RJ, Martin KAC, Koch C (1991). Synaptic
           background activity influences spatiotemporal integration in single
           pyramidal cells. PNAS, 88(24):11569-11573.
           DOI: https://doi.org/10.1073/pnas.88.24.11569
    .. [3] Kuhn A, Rotter S (2004). Neuronal integration of synaptic input in
           the fluctuation-driven regime. Journal of Neuroscience, 24(10):2345-2356.
           DOI: https://doi.org/10.1523/JNEUROSCI.3349-03.2004
    .. [4] NEST Simulator ``iaf_cond_alpha`` model documentation and C++ source:
           ``models/iaf_cond_alpha.h`` and ``models/iaf_cond_alpha.cpp``.

    See Also
    --------
    iaf_cond_exp : Conductance-based LIF with exponential synapses
    iaf_psc_alpha : Current-based LIF with alpha synapses
    iaf_psc_delta : Current-based LIF with delta synapses

    Examples
    --------
    Create a population of 100 conductance-based neurons:

    .. code-block:: python

        >>> from brainpy import state as bst
        >>> import brainunit as u
        >>> neurons = bst.iaf_cond_alpha(
        ...     in_size=100,
        ...     V_th=-50. * u.mV,
        ...     tau_syn_ex=0.5 * u.ms,
        ...     tau_syn_in=2.0 * u.ms
        ... )
    """
    __module__ = 'brainpy.state'

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    #: Unit the multi-receptor ``connect(receptor_type=k)`` bridge uses to scale the
    #: gathered per-port deposit into the ``w_by_rec`` mantissa (conductance -> nS).
    receptor_input_unit = u.nS

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 250. * u.pF,
        t_ref: ArrayLike = 2. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -60. * u.mV,
        E_ex: ArrayLike = 0. * u.mV,
        E_in: ArrayLike = -85. * u.mV,
        g_L: ArrayLike = 16.6667 * u.nS,
        tau_syn_ex: ArrayLike = 0.2 * u.ms,
        tau_syn_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        gsl_error_tol: ArrayLike = 1e-3,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        g_ex_initializer: Callable = braintools.init.Constant(0. * u.nS),
        g_in_initializer: Callable = braintools.init.Constant(0. * u.nS),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        self.V_initializer = V_initializer
        self.g_ex_initializer = g_ex_initializer
        self.g_in_initializer = g_in_initializer
        self.ref_var = ref_var

        # Two delivery ports for the Simulator's multi-receptor bridge: receptor 1 ->
        # g_ex (excitatory), receptor 2 -> g_in (inhibitory). NEST routes excitation /
        # inhibition by weight *sign* (iaf_cond_alpha.cpp handle(): weight>0 ->
        # spike_exc_, else spike_inh_); the Simulator expresses that as receptor_type=1/2
        # with positive nS weights.
        self.n_receptors = 2

        self._validate_parameters()

        self.integrator = AdaptiveRungeKuttaStep(
            method='RKF45',
            vf=self._vector_field,
            event_fn=self._event_fn,
            min_h=self._MIN_H,
            max_iters=self._MAX_ITERS,
            atol=self.gsl_error_tol,
            dt=brainstate.environ.get_dt()
        )

        # other variable
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    def _validate_parameters(self):
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_reset, self.C_m)):
            return
        if cond_any(self.V_reset >= self.V_th):
            raise ValueError('Reset potential must be smaller than threshold.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms) or cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables.

        Creates and initializes membrane potential, conductance states, refractory
        counters, integration step size, and optional refractory indicator.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments (unused, for API compatibility).

        Notes
        -----
        - ``V``, ``g_ex``, ``g_in`` are initialized using their respective initializers.
        - ``dg_ex``, ``dg_in`` (alpha auxiliary states) are initialized to zero.
        - ``last_spike_time`` is set to large negative value (-1e7 ms).
        - ``refractory_step_count`` starts at 0 (not in refractory period).
        - ``integration_step`` is initialized to the global timestep ``dt``.
        - ``I_stim`` buffer starts at 0 pA.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        g_ex = braintools.init.param(self.g_ex_initializer, self.varshape)
        g_in = braintools.init.param(self.g_in_initializer, self.varshape)
        zeros = u.math.zeros(self.varshape, dtype=V.dtype) * (u.nS / u.ms)

        self.dg_ex = brainstate.ShortTermState(zeros)
        self.dg_in = brainstate.ShortTermState(zeros)
        self.g_ex = brainstate.HiddenState(g_ex)
        self.g_in = brainstate.HiddenState(g_in)
        self.V = brainstate.HiddenState(V)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output using surrogate gradient.

        Applies the surrogate spike function to a normalized voltage to produce
        a continuous approximation of spike events suitable for gradient-based learning.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential to evaluate. If None, uses current ``self.V.value``.
            Shape must match neuron population shape.

        Returns
        -------
        ArrayLike
            Spike output in [0, 1], where values close to 1 indicate spike events.
            Shape matches input voltage shape.

        Notes
        -----
        The voltage is normalized to :math:`(V - V_\mathrm{th}) / (V_\mathrm{th} - V_\mathrm{reset})`
        before applying the surrogate function. This makes the surrogate function
        operate in a standardized range regardless of absolute voltage values.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, dg_ex, g_ex, dg_in, g_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, i_stim -- mutable auxiliary data carried
            through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        # During refractory: v_eff = V_reset. Otherwise: v_eff = min(V, V_th).
        v_eff = u.math.where(is_refractory, self.V_reset, u.math.minimum(state.V, self.V_th))

        i_syn_exc = state.g_ex * (v_eff - self.E_ex)
        i_syn_inh = state.g_in * (v_eff - self.E_in)
        i_leak = self.g_L * (v_eff - self.E_L)

        dV_raw = (-i_leak - i_syn_exc - i_syn_inh + self.I_e + extra.i_stim) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        ddg_ex = -state.dg_ex / self.tau_syn_ex
        dg_ex_dt = state.dg_ex - state.g_ex / self.tau_syn_ex
        ddg_in = -state.dg_in / self.tau_syn_in
        dg_in_dt = state.dg_in - state.g_in / self.tau_syn_in

        return DotDict(V=dV, dg_ex=ddg_ex, g_ex=dg_ex_dt, dg_in=ddg_in, g_in=dg_in_dt)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, dg_ex, g_ex, dg_in, g_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, i_stim.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        # Clamp voltage during refractory period.
        refr_accept = accept & (extra.r > 0)
        new_V = u.math.where(refr_accept, self.V_reset, state.V)

        # Spike detection: not refractory and V >= V_th.
        spike_now = accept & (extra.r <= 0) & (new_V >= self.V_th)
        spike_mask = extra.spike_mask | spike_now
        new_V = u.math.where(spike_now, self.V_reset, new_V)
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict({**state, 'V': new_V})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r})
        return new_state, new_extra

    def update(self, x=0. * u.pA, w_by_rec=None):
        r"""Advance neuron state by one simulation timestep.

        Integrates ODEs, handles refractory period and spike emission, applies
        synaptic conductance jumps, and buffers external current for next step.
        This method implements the full NEST update semantics.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input for this timestep (pA). Broadcasted to population
            shape. This input is buffered and applied in the *next* timestep (one-step
            delay) to match NEST ring-buffer semantics. Default: 0 pA.
        w_by_rec : ArrayLike or None, optional
            Per-receptor synaptic conductance jumps supplied by the Simulator's
            multi-receptor bridge, shape ``(*varshape, n_receptors)`` as a dimensionless
            ``nS`` mantissa. Column 0 (``receptor_type=1``) is added to the excitatory
            conductance ``g_ex`` and column 1 (``receptor_type=2``) to the inhibitory
            conductance ``g_in``. When ``None`` (the legacy path), the jumps are
            self-pulled from the ``label='w_ex'/'w_in'`` delta inputs instead.

        Returns
        -------
        ArrayLike
            Differentiable spike output (values in [0, 1], shape matching population).
            Computed using surrogate gradient on pre-reset membrane potential.

        Notes
        -----
        **Update order** (matching NEST):

        1. **Integrate ODEs**: Use RKF45 to advance ``V``, ``dg_ex``, ``g_ex``,
           ``dg_in``, ``g_in`` over ``(t, t+dt]`` with ``I_stim`` from previous step.
        2. **Refractory/spike handling**:

           - If in refractory period: clamp ``V`` to ``V_reset``, decrement counter.
           - Else if ``V >= V_th``: emit spike, reset ``V`` to ``V_reset``, set
             refractory counter.
        3. **Apply synaptic inputs**: Add conductance jumps from incoming spikes
           (via ``add_delta_input``) to ``dg_ex`` / ``dg_in`` with alpha normalization.
        4. **Buffer current input**: Store ``x`` into ``I_stim`` for next timestep.

        The surrogate spike is computed from the *pre-reset* voltage to allow gradient
        flow through spike events during training.

        **Failure modes**: If integration does not converge within ``_MAX_ITERS``
        iterations, the final state may be inaccurate. Reduce global ``dt`` or check
        for extreme parameter values if this occurs.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        dg_ex = self.dg_ex.value  # nS/ms
        g_ex = self.g_ex.value  # nS
        dg_in = self.dg_in.value  # nS/ms
        g_in = self.g_in.value  # nS
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, dg_ex=dg_ex, g_ex=g_ex, dg_in=dg_in, g_in=g_in)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            i_stim=i_stim,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, dg_ex, g_ex = ode_state.V, ode_state.dg_ex, ode_state.g_ex
        dg_in, g_in = ode_state.dg_in, ode_state.g_in
        spike_mask, r = extra.spike_mask, extra.r

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration). Two delivery paths: the
        # Simulator's multi-receptor bridge pre-gathers per-receptor conductance jumps
        # (nS mantissa, shape ``(*varshape, n_receptors)``) and passes them via
        # ``w_by_rec`` -- column 0 -> g_ex, column 1 -> g_in; the legacy BrainPy-style
        # path self-pulls the ``label='w_ex'/'w_in'`` delta inputs. Only the *source* of
        # ``w_ex``/``w_in`` changes; the alpha kinetics below consume them unchanged.
        if w_by_rec is not None:
            w_by_rec = jnp.asarray(w_by_rec, dtype=dftype)
            w_ex = w_by_rec[..., 0] * u.nS
            w_in = w_by_rec[..., 1] * u.nS
        else:
            w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
            w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')
        pscon_ex = np.e / self.tau_syn_ex  # 1/ms
        pscon_in = np.e / self.tau_syn_in  # 1/ms

        # Apply synaptic spike inputs.
        dg_ex = dg_ex + pscon_ex * w_ex  # nS/ms + 1/ms * nS = nS/ms
        dg_in = dg_in + pscon_in * w_in  # nS/ms + 1/ms * nS = nS/ms

        # Write back state.
        self.V.value = V
        self.dg_ex.value = dg_ex
        self.g_ex.value = g_ex
        self.dg_in.value = dg_in
        self.g_in.value = g_in
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
