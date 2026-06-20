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

from collections.abc import Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'hh_cond_exp_traub',
]


class hh_cond_exp_traub(NESTNeuron):
    r"""NEST-compatible ``hh_cond_exp_traub`` neuron model.

    Hodgkin-Huxley model for Brette et al. (2007) review, based on Traub and
    Miles (1991) hippocampal pyramidal cell model.

    This is a modified Hodgkin-Huxley neuron model specifically developed for
    the Brette et al. (2007) simulator review, based on a model of hippocampal
    pyramidal cells by Traub and Miles (1991). Key differences from the original
    Traub-Miles model:

    - This is a point neuron, not a compartmental model.
    - Only ``I_Na`` and ``I_K`` ionic currents are included (no calcium dynamics),
      with simplified ``I_K`` dynamics giving three gating variables instead of
      eight.
    - Incoming spikes induce an instantaneous conductance change followed by
      exponential decay (conductance-based synapses), not activation over time.

    Parameters
    ----------
    in_size : int, tuple of int
        Population shape (number of neurons or spatial dimensions).
    E_L : ArrayLike, default -60 mV
        Leak reversal potential. Must be finite.
    C_m : ArrayLike, default 200 pF
        Membrane capacitance. Must be strictly positive.
    g_Na : ArrayLike, default 20000 nS
        Sodium peak conductance. Must be non-negative.
    g_K : ArrayLike, default 6000 nS
        Potassium peak conductance. Must be non-negative.
    g_L : ArrayLike, default 10 nS
        Leak conductance. Must be non-negative.
    E_Na : ArrayLike, default 50 mV
        Sodium reversal potential. Must be finite.
    E_K : ArrayLike, default -90 mV
        Potassium reversal potential. Must be finite.
    V_T : ArrayLike, default -63 mV
        Voltage offset for gating dynamics. Shifts the effective threshold
        to approximately V_T + 30 mV.
    E_ex : ArrayLike, default 0 mV
        Excitatory synaptic reversal potential. Must be finite.
    E_in : ArrayLike, default -80 mV
        Inhibitory synaptic reversal potential. Must be finite.
    t_ref : ArrayLike, default 2 ms
        Duration of refractory period. Must be non-negative. Traub and Miles
        used 3 ms; NEST default is 2 ms.
    tau_syn_ex : ArrayLike, default 5 ms
        Excitatory synaptic time constant. Must be strictly positive.
    tau_syn_in : ArrayLike, default 10 ms
        Inhibitory synaptic time constant. Must be strictly positive.
    I_e : ArrayLike, default 0 pA
        Constant external input current. Can be positive or negative.
    V_m_init : ArrayLike, optional
        Initial membrane potential. If None, defaults to E_L.
    Act_m_init : ArrayLike, optional
        Initial sodium activation gating variable (0 <= m <= 1). If None,
        computed from equilibrium at V_m_init.
    Inact_h_init : ArrayLike, optional
        Initial sodium inactivation gating variable (0 <= h <= 1). If None,
        computed from equilibrium at V_m_init.
    Act_n_init : ArrayLike, optional
        Initial potassium activation gating variable (0 <= n <= 1). If None,
        computed from equilibrium at V_m_init.
    gsl_error_tol : ArrayLike
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
    spk_fun : Callable, default braintools.surrogate.ReluGrad()
        Surrogate spike function for differentiable spike generation.
    spk_reset : str, default 'hard'
        Reset mode ('hard' or 'soft'). Note: HH models do not reset voltage
        after spikes; this parameter affects gradient computation only.
    name : str, optional
        Name of the neuron population.

    Attributes
    ----------
    V : brainstate.HiddenState
        Membrane potential with shape (\*in_size,) in mV.
    m : brainstate.HiddenState
        Sodium activation gating variable (0 <= m <= 1), shape (\*in_size,).
    h : brainstate.HiddenState
        Sodium inactivation gating variable (0 <= h <= 1), shape (\*in_size,).
    n : brainstate.HiddenState
        Potassium activation gating variable (0 <= n <= 1), shape (\*in_size,).
    g_ex : brainstate.HiddenState
        Excitatory synaptic conductance in nS, shape (\*in_size,).
    g_in : brainstate.HiddenState
        Inhibitory synaptic conductance in nS, shape (\*in_size,).
    I_stim : brainstate.ShortTermState
        Stimulation current buffer in pA, shape (\*in_size,).
    refractory_step_count : brainstate.ShortTermState
        Refractory countdown in grid steps, shape (\*in_size,), dtype int32.
    integration_step : brainstate.ShortTermState
        Persistent RKF45 substep size estimate (ms).
    last_spike_time : brainstate.ShortTermState
        Time of most recent spike in ms, shape (\*in_size,).

    Raises
    ------
    ValueError
        If C_m <= 0, t_ref < 0, tau_syn_ex <= 0, or tau_syn_in <= 0.

    Notes
    -----
    - Unlike IAF models, the HH model does **not** reset the membrane
      potential after a spike. Repolarization occurs naturally through
      the potassium current.
    - During the refractory period, subthreshold dynamics continue to
      evolve freely; only spike emission is suppressed.
    - Synaptic spike weights are interpreted in conductance units (nS).
      Positive weights drive excitatory synapses; negative weights drive
      inhibitory synapses (sign is flipped, i.e. ``g_in += |w|``).
    - The numerical integration uses an adaptive RKF45 (Runge-Kutta-Fehlberg)
      integrator implemented in JAX with unit-aware arithmetic via brainunit.
      This is equivalent to NEST's GSL RKF45 implementation for numerical
      correspondence.

    Mathematical Formulation
    -------------------------

    **1. Membrane and Ionic Current Dynamics**

    The membrane potential evolves as:

    .. math::

       C_m \frac{dV_m}{dt} = -(I_{Na} + I_K + I_L + I_{syn,ex} + I_{syn,in})
                              + I_{stim} + I_e

    where the currents are:

    .. math::

       I_{Na}     &= g_{Na}\, m^3\, h\, (V_m - E_{Na})  \\
       I_K        &= g_K\,   n^4\,     (V_m - E_K)       \\
       I_L        &= g_L\,             (V_m - E_L)        \\
       I_{syn,ex} &= g_{ex}\,          (V_m - E_{ex})     \\
       I_{syn,in} &= g_{in}\,          (V_m - E_{in})

    **2. Channel Gating Variables**

    Gating variables :math:`m`, :math:`h`, :math:`n` obey first-order kinetics:

    .. math::

       \frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)\,x
                     = \alpha_x - (\alpha_x + \beta_x)\, x

    with Traub-Miles rate functions using shifted voltage
    :math:`V = V_m - V_T` (voltage in mV, rates in 1/ms):

    .. math::

       \alpha_n &= \frac{0.032\,(15 - V)}{e^{(15 - V)/5} - 1}, \quad
       \beta_n  = 0.5\,e^{(10 - V)/40}                                \\
       \alpha_m &= \frac{0.32\,(13 - V)}{e^{(13 - V)/4} - 1}, \quad
       \beta_m  = \frac{0.28\,(V - 40)}{e^{(V - 40)/5} - 1}          \\
       \alpha_h &= 0.128\,e^{(17 - V)/18}, \quad
       \beta_h  = \frac{4}{1 + e^{(40 - V)/5}}

    The voltage offset :math:`V_T` (default -63 mV) shifts the effective
    threshold to approximately -50 mV.

    **3. Exponential Conductance Synapses**

    Synaptic conductances decay exponentially:

    .. math::

       \frac{dg_{ex}}{dt} &= -g_{ex} / \tau_{syn,ex} \\
       \frac{dg_{in}}{dt} &= -g_{in} / \tau_{syn,in}

    A presynaptic spike with weight :math:`w` causes an instantaneous
    conductance jump:

    - :math:`w > 0` -- :math:`g_{ex} \leftarrow g_{ex} + w`
    - :math:`w < 0` -- :math:`g_{in} \leftarrow g_{in} + |w|`

    **4. Spike Detection**

    A spike is emitted when all three conditions are satisfied:

    1. ``r == 0`` (not in refractory period), **and**
    2. ``V_m >= V_T + 30`` mV (threshold crossing), **and**
    3. ``V_old > V_m`` (local maximum, the potential is now falling).

    Unlike integrate-and-fire models, no voltage reset occurs -- the
    potassium current naturally repolarizes the membrane.

    .. warning::

       To avoid multiple spikes during the falling flank of a spike, it is
       essential to choose a sufficiently long refractory period.
       Traub and Miles used :math:`t_{ref} = 3` ms, while the default here
       is :math:`t_{ref} = 2` ms (matching NEST).

    **5. Numerical Integration**

    NEST uses GSL RKF45 (Runge-Kutta-Fehlberg 4/5) with adaptive step-size
    control. This implementation uses an adaptive RKF45 integrator implemented
    in JAX with unit-aware arithmetic via brainunit, matching NEST's integration
    approach for numerical correspondence.

    The ODE system is 6-dimensional per neuron:
    :math:`[V_m, m, h, n, g_{ex}, g_{in}]`.

    Parameter Mapping
    -----------------

    The following table shows the correspondence between brainpy.state parameters
    and NEST/mathematical notation:

    ==================== ================== =============================== ====================================================
    **Parameter**        **Default**        **Math equivalent**             **Description**
    ==================== ================== =============================== ====================================================
    ``in_size``          (required)         --                              Population shape
    ``E_L``              -60 mV             :math:`E_L`                     Leak reversal potential
    ``C_m``              200 pF             :math:`C_m`                     Membrane capacitance
    ``g_Na``             20000 nS           :math:`g_{Na}`                  Sodium peak conductance
    ``g_K``              6000 nS            :math:`g_K`                     Potassium peak conductance
    ``g_L``              10 nS              :math:`g_L`                     Leak conductance
    ``E_Na``             50 mV              :math:`E_{Na}`                  Sodium reversal potential
    ``E_K``              -90 mV             :math:`E_K`                     Potassium reversal potential
    ``V_T``              -63 mV             :math:`V_T`                     Voltage offset for gating dynamics
    ``E_ex``             0 mV               :math:`E_{ex}`                  Excitatory synaptic reversal potential
    ``E_in``             -80 mV             :math:`E_{in}`                  Inhibitory synaptic reversal potential
    ``t_ref``            2 ms               :math:`t_{ref}`                 Duration of refractory period
    ``tau_syn_ex``       5 ms               :math:`\tau_{syn,ex}`           Excitatory synaptic time constant
    ``tau_syn_in``       10 ms              :math:`\tau_{syn,in}`           Inhibitory synaptic time constant
    ``I_e``              0 pA               :math:`I_e`                     Constant external input current
    ``V_m_init``         None               --                              Initial V_m (None -> E_L)
    ``Act_m_init``       None               --                              Initial Na activation (None -> equilibrium)
    ``Inact_h_init``     None               --                              Initial Na inactivation (None -> equilibrium)
    ``Act_n_init``       None               --                              Initial K activation (None -> equilibrium)
    ``gsl_error_tol``    1e-3               --                              Local RKF45 error tolerance
    ``spk_fun``          ReluGrad()         --                              Surrogate spike function
    ``spk_reset``        ``'hard'``         --                              Reset mode
    ==================== ================== =============================== ====================================================

    Examples
    --------
    .. code-block:: python

       >>> import brainstate as bst
       >>> import brainunit as u
       >>> from brainpy.state import hh_cond_exp_traub
       >>>
       >>> # Create a population of 100 Traub HH neurons
       >>> neurons = hh_cond_exp_traub(100)
       >>> neurons.init_all_states()
       >>>
       >>> # Run a simulation with constant current injection
       >>> with bst.environ.context(dt=0.1*u.ms):
       ...     for i in range(1000):
       ...         spikes = neurons.update(I_e=200*u.pA)

    .. code-block:: python

       >>> # Compare with NEST default parameters
       >>> import nest
       >>> nest_neuron = nest.Create('hh_cond_exp_traub')
       >>> nest.GetStatus(nest_neuron, ['V_m', 'E_L', 'C_m', 'g_Na', 'g_K'])
       [(-60.0, -60.0, 200.0, 20000.0, 6000.0)]
       >>>
       >>> # Match in brainpy.state
       >>> bp_neuron = hh_cond_exp_traub(1, E_L=-60*u.mV, C_m=200*u.pF,
       ...                               g_Na=20000*u.nS, g_K=6000*u.nS)

    References
    ----------
    .. [1] Brette R et al. (2007). Simulation of networks of spiking neurons:
           A review of tools and strategies. Journal of Computational
           Neuroscience 23:349-98.
           DOI: https://doi.org/10.1007/s10827-007-0038-6
    .. [2] Traub RD and Miles R (1991). Neuronal networks of the hippocampus.
           Cambridge University Press, Cambridge UK.
    .. [3] ModelDB entry: http://modeldb.yale.edu/83319

    See Also
    --------
    hh_psc_alpha : Hodgkin-Huxley with alpha-shaped postsynaptic currents.
    iaf_cond_exp : Leaky integrate-and-fire with conductance-based synapses.
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
        E_L: ArrayLike = -60. * u.mV,
        C_m: ArrayLike = 200. * u.pF,
        g_Na: ArrayLike = 20000. * u.nS,
        g_K: ArrayLike = 6000. * u.nS,
        g_L: ArrayLike = 10. * u.nS,
        E_Na: ArrayLike = 50. * u.mV,
        E_K: ArrayLike = -90. * u.mV,
        V_T: ArrayLike = -63. * u.mV,
        E_ex: ArrayLike = 0. * u.mV,
        E_in: ArrayLike = -80. * u.mV,
        t_ref: ArrayLike = 2. * u.ms,
        tau_syn_ex: ArrayLike = 5. * u.ms,
        tau_syn_in: ArrayLike = 10. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        V_m_init: ArrayLike = None,
        Act_m_init: ArrayLike = None,
        Inact_h_init: ArrayLike = None,
        Act_n_init: ArrayLike = None,
        gsl_error_tol: ArrayLike = 1e-3,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str | None = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.g_Na = braintools.init.param(g_Na, self.varshape)
        self.g_K = braintools.init.param(g_K, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.E_Na = braintools.init.param(E_Na, self.varshape)
        self.E_K = braintools.init.param(E_K, self.varshape)
        self.V_T = braintools.init.param(V_T, self.varshape)
        self.E_ex = braintools.init.param(E_ex, self.varshape)
        self.E_in = braintools.init.param(E_in, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_m_init = V_m_init
        self.Act_m_init = Act_m_init
        self.Inact_h_init = Inact_h_init
        self.Act_n_init = Act_n_init
        self.gsl_error_tol = gsl_error_tol

        # Two synaptic receptors (g_ex / g_in) addressable by ``connect(receptor_type=k)``:
        # the Simulator's multi-receptor bridge gathers per-port conductance jumps into a
        # ``(*varshape, n_receptors)`` blob and passes it to ``update(w_by_rec=...)``;
        # column 0 -> g_ex (receptor 1), column 1 -> g_in (receptor 2).
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
        r"""Validate parameter constraints.

        Raises
        ------
        ValueError
            If capacitance C_m <= 0, refractory time t_ref < 0, or any synaptic
            time constant (tau_syn_ex, tau_syn_in) <= 0.

        Notes
        -----
        This is called during __init__ to ensure physical validity of parameters.
        Conductances (g_L, g_Na, g_K) are not validated for positivity since
        zero conductance is physically meaningful (though unusual).
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.t_ref, self.tau_syn_ex)):
            return
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms) or cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Initializes membrane potential, gating variables, synaptic conductances,
        stimulation current buffer, refractory counter, and last spike time. If
        initial values are not explicitly provided, they are computed as follows:

        - ``V``: defaults to ``E_L``
        - ``m, h, n``: computed from equilibrium at initial ``V`` using Traub-Miles
          rate equations (without V_T offset, matching NEST initialization)
        - ``g_ex, g_in``: initialized to zero
        - ``I_stim``: initialized to zero
        - ``refractory_step_count``: initialized to zero (not refractory)
        - ``last_spike_time``: initialized to -1e7 ms (far in the past)

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        The equilibrium gating variable computation uses the raw voltage V (not
        V - V_T) to match NEST's initialization procedure. During dynamics, the
        rate equations use the shifted voltage V - V_T, but initialization uses
        the unshifted value for consistency with NEST's ``State_::State_``
        constructor.

        This initialization ensures the neuron starts in a stable resting state
        when V_m_init = E_L (default). For custom initial voltages, gating
        variables are automatically adjusted to the corresponding equilibrium.

        Examples
        --------
        .. code-block:: python

           >>> import brainstate as bst
           >>> import brainunit as u
           >>> from brainpy.state import hh_cond_exp_traub
           >>>
           >>> # Initialize with default rest state
           >>> neurons = hh_cond_exp_traub(100)
           >>> neurons.init_state()
           >>> print(neurons.V.value[0])  # Should be E_L = -60 mV
           -60.0 mV
           >>>
           >>> # Initialize with custom voltage
           >>> neurons = hh_cond_exp_traub(100, V_m_init=-65*u.mV)
           >>> neurons.init_state()
           >>> print(neurons.V.value[0])
           -65.0 mV

        Raises
        ------
        ValueError
            If an initializer cannot be broadcast to requested shape.
        TypeError
            If initializer outputs have incompatible units/dtypes for the
            corresponding state variables.

        See Also
        --------
        _hh_cond_exp_traub_equilibrium : Computes equilibrium gating values.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        # Default V_m_init to E_L (matching NEST: y_[0] = p.E_L)
        if self.V_m_init is not None:
            V_init_val = self.V_m_init
        else:
            V_init_val = self.E_L

        V = braintools.init.param(braintools.init.Constant(V_init_val), self.varshape)

        # Compute equilibrium gating variables at initial V.
        # NEST uses raw V_m (not V_m - V_T) for equilibrium initialization.
        V_init_mV = float(np.asarray(u.math.asarray(V_init_val / u.mV)).flat[0])
        m_eq, h_eq, n_eq = _hh_cond_exp_traub_equilibrium(V_init_mV)

        if self.Act_m_init is not None:
            m_init = float(np.asarray(u.math.asarray(self.Act_m_init / u.UNITLESS)).flat[0])
        else:
            m_init = m_eq
        if self.Inact_h_init is not None:
            h_init = float(np.asarray(u.math.asarray(self.Inact_h_init / u.UNITLESS)).flat[0])
        else:
            h_init = h_eq
        if self.Act_n_init is not None:
            n_init = float(np.asarray(u.math.asarray(self.Act_n_init / u.UNITLESS)).flat[0])
        else:
            n_init = n_eq

        self.V = brainstate.HiddenState(V)
        self.m = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(m_init), self.varshape)
        )
        self.h = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(h_init), self.varshape)
        )
        self.n = brainstate.HiddenState(
            braintools.init.param(braintools.init.Constant(n_init), self.varshape)
        )
        self.g_ex = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=V.dtype) * u.nS)
        self.g_in = brainstate.HiddenState(u.math.zeros(self.varshape, dtype=V.dtype) * u.nS)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

    def get_spike(self, V: ArrayLike = None):
        r"""Compute differentiable spike output using surrogate gradient function.

        Applies the surrogate spike function to the membrane potential. This is
        used for gradient-based learning; actual spike detection in the update
        method uses discrete threshold crossing logic (V >= V_T + 30 and local
        maximum).

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential in mV, shape (\*in_size,) or (batch_size, \*in_size).
            If None, uses the current state ``self.V.value``.

        Returns
        -------
        ArrayLike
            Differentiable spike output with the same shape as input V. Values are
            approximately 0 (no spike) or 1 (spike) with smooth gradients for
            backpropagation.

        Notes
        -----
        The voltage is scaled to unitless values (mV) before applying the surrogate
        function. For Hodgkin-Huxley neurons, the actual spike threshold is
        V_T + 30 mV (default: -33 mV), but the surrogate function operates on
        the raw scaled voltage for gradient computation.

        This method is primarily used for surrogate gradient learning. The discrete
        spike detection logic in the update method is independent and uses the
        three-condition test (refractory, threshold, local maximum).

        Examples
        --------
        .. code-block:: python

           >>> import brainunit as u
           >>> import jax.numpy as jnp
           >>> from brainpy.state import hh_cond_exp_traub
           >>>
           >>> neurons = hh_cond_exp_traub(10)
           >>> neurons.init_state()
           >>>
           >>> # Get spike output for current state
           >>> spikes = neurons.get_spike()
           >>> print(spikes.shape)
           (10,)
           >>>
           >>> # Get spike output for custom voltage
           >>> V_custom = jnp.array([-60., -50., -40.]) * u.mV
           >>> neurons_3 = hh_cond_exp_traub(3)
           >>> neurons_3.init_state()
           >>> spikes_custom = neurons_3.get_spike(V_custom)

        See Also
        --------
        update : Main update method with discrete spike detection logic.
        """
        V = self.V.value if V is None else V
        # For HH neurons with Traub threshold: spike at V_T + 30.
        # Scale relative to 0 mV for the surrogate function.
        v_scaled = V / (1. * u.mV)
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, g_ex, g_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, V_old -- mutable
            auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        V_m = state.V

        # Ionic currents
        I_Na = self.g_Na * state.m ** 3 * state.h * (V_m - self.E_Na)
        I_K = self.g_K * state.n ** 4 * (V_m - self.E_K)
        I_L = self.g_L * (V_m - self.E_L)

        # Synaptic currents (conductance-based)
        I_syn_exc = state.g_ex * (V_m - self.E_ex)
        I_syn_inh = state.g_in * (V_m - self.E_in)

        # Membrane voltage derivative
        dV = (-I_Na - I_K - I_L - I_syn_exc - I_syn_inh + extra.i_stim + self.I_e) / self.C_m

        # Shifted voltage for gating variable rate equations
        V_shifted = (V_m - self.V_T) / u.mV  # unitless

        # Traub-Miles rate functions
        alpha_n = 0.032 * (15.0 - V_shifted) / (u.math.exp((15.0 - V_shifted) / 5.0) - 1.0) / u.ms
        beta_n = 0.5 * u.math.exp((10.0 - V_shifted) / 40.0) / u.ms
        alpha_m = 0.32 * (13.0 - V_shifted) / (u.math.exp((13.0 - V_shifted) / 4.0) - 1.0) / u.ms
        beta_m = 0.28 * (V_shifted - 40.0) / (u.math.exp((V_shifted - 40.0) / 5.0) - 1.0) / u.ms
        alpha_h = 0.128 * u.math.exp((17.0 - V_shifted) / 18.0) / u.ms
        beta_h = 4.0 / (1.0 + u.math.exp((40.0 - V_shifted) / 5.0)) / u.ms

        # Gating variable derivatives
        dm = alpha_m - (alpha_m + beta_m) * state.m
        dh = alpha_h - (alpha_h + beta_h) * state.h
        dn = alpha_n - (alpha_n + beta_n) * state.n

        # Synaptic conductance derivatives
        dg_ex = -state.g_ex / self.tau_syn_ex
        dg_in = -state.g_in / self.tau_syn_in

        return DotDict(V=dV, m=dm, h=dh, n=dn, g_ex=dg_ex, g_in=dg_in)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection and refractory handling.

        Detects spikes using threshold crossing and local maximum conditions,
        and manages refractory state. Unlike IAF models, no voltage reset is
        applied -- repolarization occurs naturally through potassium currents.

        Parameters
        ----------
        state : DotDict
            Keys: V, m, h, n, g_ex, g_in -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, V_old.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & ((state.V < -1e3 * u.mV) | (state.V > 1e3 * u.mV))
        )

        # Spike detection threshold: V_T + 30 mV
        v_threshold = self.V_T + 30.0 * u.mV

        # Spike conditions: not refractory, threshold crossed, and local maximum (V_old > V)
        not_refractory = extra.r <= 0
        crossed_threshold = state.V >= v_threshold
        local_max = extra.V_old > state.V
        spike_now = accept & not_refractory & crossed_threshold & local_max

        spike_mask = extra.spike_mask | spike_now

        # Update V_old to track the previous voltage for local-max detection
        new_V_old = u.math.where(accept, state.V, extra.V_old)

        # Set refractory counter on spike
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count, extra.r)

        new_state = DotDict({**state})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable, 'V_old': new_V_old})
        return new_state, new_extra

    def update(self, x=0. * u.pA, w_by_rec=None):
        r"""Update neuron state for one simulation step.

        Integrates the 6-dimensional ODE system for one time step using adaptive
        RKF45 solver, processes incoming synaptic inputs, detects spikes based on
        threshold crossing and local maximum, and updates refractory state.

        The update follows the NEST ``hh_cond_exp_traub`` update order:

        1. Record pre-integration membrane potential (``V_old``).
        2. Integrate the full 6-dimensional ODE system over one time step
           using an adaptive RKF45 solver.
        3. Add arriving synaptic conductance jumps to ``g_ex`` / ``g_in``.
        4. Check spike condition: ``V_m >= V_T + 30 and V_old > V_m``
           (threshold + local maximum).
        5. Update refractory counter and record spike time.
        6. Store buffered stimulation current for the next step.

        Parameters
        ----------
        x : ArrayLike, default 0 pA
            External stimulation current input (in addition to ``I_e``), shape
            () or (\*in_size,). This current is added
            to the constant ``I_e`` parameter and any registered current inputs
            via ``add_current_input()``.
        w_by_rec : ArrayLike or None, optional
            Per-receptor conductance jumps gathered by the Simulator's multi-receptor
            bridge, shape ``(*varshape, n_receptors)`` in nanosiemens -- column 0 is the
            excitatory port (``g_ex``), column 1 the inhibitory port (``g_in``). ``None``
            (the default) selects the legacy self-pull path (``label='w_ex'/'w_in'`` delta
            inputs), so direct calls and the BrainPy-style API are unchanged.

        Returns
        -------
        ArrayLike
            Spike output with shape (\*in_size,). Values are computed using the
            surrogate spike function for differentiability. Spikes occur only
            when the discrete spike condition is satisfied (not refractory,
            threshold crossed, and local maximum detected).

        Notes
        -----
        **Integration Details:**

        Each neuron's state is integrated using an adaptive RKF45 integrator
        implemented in JAX with unit-aware arithmetic. This matches NEST's
        GSL RKF45 solver. The ODE system is:

        .. math::

           \frac{d}{dt}\begin{bmatrix} V_m \\ m \\ h \\ n \\ g_{ex} \\ g_{in} \end{bmatrix}
           = \begin{bmatrix}
               (-I_{Na} - I_K - I_L - I_{syn,ex} - I_{syn,in} + I_{stim} + I_e) / C_m \\
               \alpha_m - (\alpha_m + \beta_m) m \\
               \alpha_h - (\alpha_h + \beta_h) h \\
               \alpha_n - (\alpha_n + \beta_n) n \\
               -g_{ex} / \tau_{syn,ex} \\
               -g_{in} / \tau_{syn,in}
           \end{bmatrix}

        **Spike Detection Logic:**

        A spike is detected when all three conditions are met:

        1. ``refractory_step_count == 0`` (not in refractory period)
        2. ``V_m >= V_T + 30`` (threshold crossing)
        3. ``V_old > V_m`` (local maximum - voltage falling)

        No voltage reset occurs; repolarization is handled by intrinsic currents.

        **Synaptic Input Processing:**

        Delta inputs (spike events) are collected and split by sign:

        - Positive weights -> excitatory conductance (``g_ex += w``)
        - Negative weights -> inhibitory conductance (``g_in += |w|``)

        Conductance jumps are applied **after** ODE integration, matching NEST's
        update sequence.

        **Computational Complexity**

        Integration is performed with an adaptive vectorized RKF45 loop,
        including in-loop spike detection and refractory handling. All
        arithmetic is unit-aware via ``brainunit.math``.

        **Failure Modes**

        - If the integrator detects numerical instability (``V < -1e3 mV``
          or ``V > 1e3 mV``), a runtime error is raised.
        - Extreme parameter values (very large conductances, very small time
          constants) may cause numerical instability.

        See Also
        --------
        init_state : Initialize state variables.
        get_spike : Compute surrogate spike output.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        m = self.m.value  # unitless
        h_val = self.h.value  # unitless
        n = self.n.value  # unitless
        g_ex = self.g_ex.value  # nS
        g_in = self.g_in.value  # nS
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h_step = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, m=m, h=h_val, n=n, g_ex=g_ex, g_in=g_in)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            V_old=V,  # Track previous V for local-max spike detection
        )

        ode_state, h_step, extra = self.integrator(state=ode_state, h=h_step, extra=extra)
        V, m, h_val = ode_state.V, ode_state.m, ode_state.h
        n, g_ex, g_in = ode_state.n, ode_state.g_ex, ode_state.g_in
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in hh_cond_exp_traub dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration). Two delivery paths: the
        # Simulator's multi-receptor bridge pre-gathers per-receptor conductance jumps
        # (nS mantissa, shape ``(*varshape, n_receptors)``) and passes them via
        # ``w_by_rec`` -- column 0 -> g_ex, column 1 -> g_in; the legacy BrainPy-style path
        # self-pulls the ``label='w_ex'/'w_in'`` delta inputs. Only the *source* of
        # ``w_ex``/``w_in`` changes; the exponential kinetics below consume them unchanged.
        if w_by_rec is not None:
            w_by_rec = jnp.asarray(w_by_rec, dtype=dftype)
            w_ex = w_by_rec[..., 0] * u.nS
            w_in = w_by_rec[..., 1] * u.nS
        else:
            w_ex = self.sum_delta_inputs(u.math.zeros_like(self.g_ex.value), label='w_ex')
            w_in = self.sum_delta_inputs(u.math.zeros_like(self.g_in.value), label='w_in')

        # Apply synaptic spike inputs (instantaneous conductance jump).
        g_ex = g_ex + w_ex
        g_in = g_in + w_in

        # Write back state.
        self.V.value = V
        self.m.value = m
        self.h.value = h_val
        self.n.value = n
        self.g_ex.value = g_ex
        self.g_in.value = g_in
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h_step
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return u.math.asarray(spike_mask, dtype=dftype)


def _hh_cond_exp_traub_equilibrium(V):
    r"""Compute Traub HH gating variable equilibrium values at voltage V (mV).

    This matches NEST's ``State_::State_(const Parameters_&)`` initialization,
    which applies the Traub rate equations **without** the V_T offset.  The
    dynamics function uses ``V - V_T`` in its rate equations, but the
    equilibrium initialization in NEST uses the raw voltage ``y_[0]`` (= E_L).

    Parameters
    ----------
    V : float
        Membrane potential in mV (unitless scalar). Must be a finite value.

    Returns
    -------
    m_inf : float
        Sodium activation equilibrium value (0 <= m_inf <= 1).
    h_inf : float
        Sodium inactivation equilibrium value (0 <= h_inf <= 1).
    n_inf : float
        Potassium activation equilibrium value (0 <= n_inf <= 1).

    Notes
    -----
    The function evaluates the steady-state gating variables using Traub-Miles
    rate equations at the unshifted voltage V (not V - V_T). This is the
    correct initialization procedure for NEST compatibility, where initial
    gating states are computed at the raw E_L value, not shifted by V_T.

    The alpha/beta rate functions may encounter division by zero at special
    voltage values (e.g., V = 15 mV for alpha_n). These singularities are
    removable via L'Hospital's rule but may cause numerical issues if V is
    exactly at these points.

    Mathematical Formulation
    ------------------------
    Equilibrium values are computed from the Traub-Miles rate equations:

    .. math::

       x_{\infty}(V) = \frac{\alpha_x(V)}{\alpha_x(V) + \beta_x(V)}

    **1. Potassium Activation (n)**

    .. math::

       \alpha_n &= \frac{0.032(15 - V)}{e^{(15-V)/5} - 1} \\
       \beta_n  &= 0.5 \, e^{(10-V)/40}

    **2. Sodium Activation (m)**

    .. math::

       \alpha_m &= \frac{0.32(13 - V)}{e^{(13-V)/4} - 1} \\
       \beta_m  &= \frac{0.28(V - 40)}{e^{(V-40)/5} - 1}

    **3. Sodium Inactivation (h)**

    .. math::

       \alpha_h &= 0.128 \, e^{(17-V)/18} \\
       \beta_h  &= \frac{4}{1 + e^{(40-V)/5}}

    Examples
    --------
    .. code-block:: python

       >>> from brainpy_state._nest_neuron.hh_cond_exp_traub import _hh_cond_exp_traub_equilibrium
       >>> m_inf, h_inf, n_inf = _hh_cond_exp_traub_equilibrium(-60.0)
       >>> print(f"m={m_inf:.4f}, h={h_inf:.4f}, n={n_inf:.4f}")
       m=0.0529, h=0.5961, n=0.3177

    See Also
    --------
    hh_cond_exp_traub : The neuron model class that uses these equilibrium values.
    """
    import math
    alpha_n = 0.032 * (15.0 - V) / (math.exp((15.0 - V) / 5.0) - 1.0)
    beta_n = 0.5 * math.exp((10.0 - V) / 40.0)
    alpha_m = 0.32 * (13.0 - V) / (math.exp((13.0 - V) / 4.0) - 1.0)
    beta_m = 0.28 * (V - 40.0) / (math.exp((V - 40.0) / 5.0) - 1.0)
    alpha_h = 0.128 * math.exp((17.0 - V) / 18.0)
    beta_h = 4.0 / (1.0 + math.exp((40.0 - V) / 5.0))
    m_inf = alpha_m / (alpha_m + beta_m)
    h_inf = alpha_h / (alpha_h + beta_h)
    n_inf = alpha_n / (alpha_n + beta_n)
    return m_inf, h_inf, n_inf
