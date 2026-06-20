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

from collections.abc import Callable, Iterable

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, validate_aeif_overflow, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'aeif_cond_beta_multisynapse',
]


class aeif_cond_beta_multisynapse(NESTNeuron):
    r"""NEST-compatible ``aeif_cond_beta_multisynapse`` neuron model.

    Conductance-based adaptive exponential integrate-and-fire neuron with
    beta-shaped synaptic conductances and an arbitrary number of receptor ports.
    Implements NEST's ``aeif_cond_beta_multisynapse`` model with source-level
    parity in update ordering, refractory handling, and spike detection.

    This model extends the adaptive exponential integrate-and-fire (AdEx) framework
    [1]_ with beta-function synaptic conductances instead of exponential or alpha
    shapes. Each receptor port maintains independent rise/decay time constants and
    reversal potentials, enabling multi-receptor networks (e.g., AMPA + GABA_A).

    Parameters
    ----------
    in_size : Size
        Population shape as integer, tuple, or Size object. Required.
    V_peak : ArrayLike, optional
        Spike detection threshold (mV). Used when ``Delta_T > 0``; otherwise
        ``V_th`` is used. Must satisfy ``V_peak >= V_th``. Default: 0.0 mV.
    V_reset : ArrayLike, optional
        Post-spike reset potential (mV). Must satisfy ``V_reset < V_peak``.
        Default: -60.0 mV.
    t_ref : ArrayLike, optional
        Absolute refractory period (ms). During refractory, ``dV/dt = 0`` and
        voltage is clamped to ``V_reset``. Default: 0.0 ms (no refractory).
    g_L : ArrayLike, optional
        Leak conductance (nS). Must be strictly positive. Default: 30.0 nS.
    C_m : ArrayLike, optional
        Membrane capacitance (pF). Must be strictly positive. Default: 281.0 pF.
    E_L : ArrayLike, optional
        Leak reversal potential (mV). Default: -70.6 mV.
    Delta_T : ArrayLike, optional
        Exponential slope factor (mV). Must be non-negative. When ``Delta_T = 0``,
        reduces to LIF-like dynamics. Default: 2.0 mV.
    tau_w : ArrayLike, optional
        Adaptation time constant (ms). Must be strictly positive. Default: 144.0 ms.
    a : ArrayLike, optional
        Subthreshold adaptation coupling (nS). Default: 4.0 nS.
    b : ArrayLike, optional
        Spike-triggered adaptation increment (pA). Added to ``w`` on each spike.
        Default: 80.5 pA.
    V_th : ArrayLike, optional
        Spike initiation threshold (mV) for exponential term. Must satisfy
        ``V_th <= V_peak``. Default: -50.4 mV.
    tau_rise : ArrayLike, optional
        Synaptic rise time constants (ms) per receptor, shape ``(n_receptors,)``.
        Must be strictly positive and satisfy ``tau_rise <= tau_decay`` element-wise.
        Default: (2.0,) ms (single receptor).
    tau_decay : ArrayLike, optional
        Synaptic decay time constants (ms) per receptor, shape ``(n_receptors,)``.
        Must be strictly positive and satisfy ``tau_decay >= tau_rise`` element-wise.
        Default: (20.0,) ms (single receptor).
    E_rev : ArrayLike, optional
        Reversal potentials (mV) per receptor, shape ``(n_receptors,)``.
        Default: (0.0,) mV (excitatory-like).
    I_e : ArrayLike, optional
        Constant external current (pA). Default: 0.0 pA.
    gsl_error_tol : ArrayLike, optional
        RKF45 local error tolerance (unitless). Smaller values improve accuracy
        but increase computational cost. Must be strictly positive. Default: 1e-6.
    V_initializer : Callable, optional
        Membrane potential initializer. Default: Constant(-70.6 mV).
    g_initializer : Callable, optional
        Conductance state initializer with shape ``[..., n_receptors]``.
        Default: Constant(0.0 nS).
    w_initializer : Callable, optional
        Adaptation current initializer. Default: Constant(0.0 pA).
    spk_fun : Callable, optional
        Surrogate gradient function for differentiable spike generation.
        Default: ReluGrad().
    spk_reset : str, optional
        Spike reset mode. ``'hard'`` (stop_gradient, matches NEST) or ``'soft'``
        (subtract threshold). Default: ``'hard'``.
    ref_var : bool, optional
        If True, expose ``refractory`` state variable (boolean indicator).
        Default: False.
    name : str, optional
        Instance name. If None, auto-generated.

    Parameter Mapping
    -----------------

    ======================== ===================== ===============================================
    **BrainPy Parameter**    **NEST Parameter**    **Description**
    ======================== ===================== ===============================================
    ``in_size``              (model count)         Population shape
    ``V_peak``               ``V_peak``            Spike detection threshold
    ``V_reset``              ``V_reset``           Reset potential
    ``t_ref``                ``t_ref``             Refractory period
    ``g_L``                  ``g_L``               Leak conductance
    ``C_m``                  ``C_m``               Membrane capacitance
    ``E_L``                  ``E_L``               Leak reversal
    ``Delta_T``              ``Delta_T``           Slope factor
    ``tau_w``                ``tau_w``             Adaptation time constant
    ``a``                    ``a``                 Subthreshold adaptation
    ``b``                    ``b``                 Spike-triggered adaptation
    ``V_th``                 ``V_th``              Exponential threshold
    ``tau_rise``             ``tau_rise``          Rise time per receptor
    ``tau_decay``            ``tau_decay``         Decay time per receptor
    ``E_rev``                ``E_rev``             Reversal potential per receptor
    ``I_e``                  ``I_e``               Constant external current
    ``gsl_error_tol``        ``gsl_error_tol``     RKF45 tolerance
    ======================== ===================== ===============================================

    Mathematical Model
    ------------------

    **1. Membrane Dynamics**

    The membrane voltage :math:`V` evolves according to:

    .. math::

       C_m \frac{dV}{dt} = -g_L (V - E_L) + g_L \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)
                           + \sum_{k=1}^{n_{\text{rec}}} g_k (E_{\text{rev},k} - V)
                           - w + I_e + I_{\text{stim}},

    where:

    - :math:`C_m` -- membrane capacitance (pF)
    - :math:`g_L` -- leak conductance (nS)
    - :math:`E_L` -- leak reversal potential (mV)
    - :math:`\Delta_T` -- exponential slope factor (mV)
    - :math:`V_{th}` -- spike initiation threshold (mV)
    - :math:`g_k` -- conductance of receptor :math:`k` (nS)
    - :math:`E_{\text{rev},k}` -- reversal potential of receptor :math:`k` (mV)
    - :math:`w` -- adaptation current (pA)
    - :math:`I_e` -- constant external current (pA)
    - :math:`I_{\text{stim}}` -- delayed injected current (pA)

    During refractory period, :math:`dV/dt = 0` and :math:`V` is clamped to
    :math:`V_{\text{reset}}`. Outside refractory, the exponential term uses
    :math:`\min(V, V_{\text{peak}})` to prevent numerical overflow.

    **2. Adaptation Dynamics**

    The adaptation current :math:`w` follows:

    .. math::

       \tau_w \frac{dw}{dt} = a (V - E_L) - w,

    where :math:`a` (nS) couples subthreshold membrane voltage fluctuations to
    adaptation. On each spike, :math:`w \leftarrow w + b` implements spike-
    triggered adaptation.

    **3. Beta-Function Synaptic Conductances**

    Each receptor :math:`k` maintains two state variables:

    .. math::

       \frac{d\,dg_k}{dt} = -\frac{dg_k}{\tau_{\text{rise},k}}, \quad
       \frac{dg_k}{dt} = dg_k - \frac{g_k}{\tau_{\text{decay},k}}.

    Incoming spikes with weight :math:`w_k` (nS) increment the auxiliary state:

    .. math::

       dg_k \leftarrow dg_k + g_{0,k} w_k,

    where :math:`g_{0,k}` is the beta normalization factor ensuring unit weight
    produces unit peak conductance:

    .. math::

       g_{0,k} = \frac{1/\tau_{\text{rise},k} - 1/\tau_{\text{decay},k}}{\exp(-t_{\text{peak}}/\tau_{\text{decay},k}) - \exp(-t_{\text{peak}}/\tau_{\text{rise},k})},

    with :math:`t_{\text{peak}} = \tau_{\text{decay},k} \tau_{\text{rise},k} \log(\tau_{\text{decay},k}/\tau_{\text{rise},k}) / (\tau_{\text{decay},k} - \tau_{\text{rise},k})`.
    In the equal-time-constant limit, this reduces to the alpha normalization
    :math:`e / \tau`.

    **4. Spike Detection and Reset**

    A spike is detected when:

    - :math:`V \geq V_{\text{peak}}` if :math:`\Delta_T > 0`
    - :math:`V \geq V_{th}` if :math:`\Delta_T = 0`

    Upon spike detection (within RKF45 substeps):

    1. :math:`V \leftarrow V_{\text{reset}}`
    2. :math:`w \leftarrow w + b`
    3. Refractory counter :math:`r \leftarrow \lceil t_{\text{ref}} / dt \rceil + 1` (if ``t_ref > 0``)

    **5. Update Order (NEST Semantics)**

    Each simulation step :math:`(t, t+dt]` proceeds as:

    1. Integrate ODEs using adaptive RKF45 with internal substeps
    2. Inside integration: apply refractory clamp and spike/reset logic
    3. Decrement refractory counter once (outside integration)
    4. Apply incoming spike events to ``dg`` states with beta normalization
    5. Store continuous current input for next step (one-step delay)

    **Computational Notes**

    - **Numerical integration**: Runge-Kutta-Fehlberg (RKF45) adaptive solver
      with local error tolerance ``gsl_error_tol``. Internal step size adapts
      dynamically and persists across simulation steps.
    - **Refractory handling**: During refractory, effective voltage is clamped
      to ``V_reset`` for all ODE terms, including adaptation coupling.
    - **Overflow protection**: Exponential term uses :math:`\min(V, V_{\text{peak}})`
      outside refractory to prevent :math:`\exp(\cdot)` overflow. Validation
      ensures :math:`(V_{\text{peak}} - V_{th}) / \Delta_T` stays below overflow
      threshold when :math:`\Delta_T > 0`.
    - **Spike event format**: ``spike_events`` must be an iterable of
      ``(receptor_type, weight)`` tuples or dicts with keys ``receptor_type``/
      ``receptor`` and ``weight``. Receptor types are 1-based (NEST convention):
      ``1 <= receptor_type <= n_receptors``. Weights (nS) must be non-negative.
    - **Default input mapping**: ``add_delta_input`` stream is mapped to receptor 1;
      weights must be non-negative.
    - **Instability detection**: Integration raises ``ValueError`` if
      :math:`V < -1000` mV or :math:`|w| > 10^6` pA, indicating numerical collapse.

    Attributes
    ----------
    V : HiddenState
        Membrane potential (mV), shape ``(*in_size,)``.
    w : HiddenState
        Adaptation current (pA), shape ``(*in_size,)``.
    dg : ShortTermState
        Beta auxiliary states (nS/ms), shape ``(*in_size, n_receptors)``.
    g : HiddenState
        Receptor conductances (nS), shape ``(*in_size, n_receptors)``.
    refractory_step_count : ShortTermState
        Remaining refractory steps (int32), shape ``(*in_size,)``.
    integration_step : ShortTermState
        Persistent RKF45 step size (ms), shape ``(*in_size,)``.
    I_stim : ShortTermState
        One-step delayed current buffer (pA), shape ``(*in_size,)``.
    last_spike_time : ShortTermState
        Last spike time (ms), shape ``(*in_size,)``. Initialized to -1e7 ms.
    refractory : ShortTermState, optional
        Boolean refractory indicator, shape ``(*in_size,)``. Only present if
        ``ref_var=True``.

    Raises
    ------
    ValueError
        If ``tau_rise.size != tau_decay.size != E_rev.size``.
    ValueError
        If any ``tau_rise <= 0`` or ``tau_decay <= 0``.
    ValueError
        If any ``tau_decay < tau_rise``.
    ValueError
        If any ``V_peak < V_th`` or ``V_reset >= V_peak``.
    ValueError
        If ``Delta_T < 0`` or ``C_m <= 0`` or ``t_ref < 0`` or ``tau_w <= 0``.
    ValueError
        If ``gsl_error_tol <= 0``.
    ValueError
        If :math:`(V_{\text{peak}} - V_{th}) / \Delta_T` exceeds overflow threshold
        (when :math:`\Delta_T > 0`).
    ValueError
        During ``update``, if receptor type out of range ``[1, n_receptors]``.
    ValueError
        During ``update``, if synaptic weight is negative (conductance constraint).
    ValueError
        During ``update``, if numerical instability detected (:math:`V < -1000` mV
        or :math:`|w| > 10^6` pA).

    See Also
    --------
    aeif_cond_alpha_multisynapse : Alpha-function variant
    aeif_cond_exp : Single exponential synapse
    aeif_psc_exp : Current-based AdEx

    Notes
    -----
    - Default ``t_ref = 0`` matches NEST and allows multiple spikes per timestep.
      Set ``t_ref > 0`` to enforce physiological refractory periods.
    - Beta conductances provide more realistic synaptic shapes than single
      exponentials but require two state variables per receptor (``dg`` and ``g``).
    - When ``tau_rise = tau_decay``, normalization degenerates to alpha-function
      limit :math:`e / \tau`.

    References
    ----------
    .. [1] Brette R, Gerstner W (2005). Adaptive exponential integrate-and-fire
           model as an effective description of neuronal activity.
           Journal of Neurophysiology, 94:3637-3642.
           DOI: https://doi.org/10.1152/jn.00686.2005
    .. [2] Roth A, van Rossum M (2013). Modeling synapses.
           In *Computational Modeling Methods for Neuroscientists*.
           MIT Press, Cambridge, MA.
    .. [3] NEST 3.9+ source: ``models/aeif_cond_beta_multisynapse.h`` and
           ``models/aeif_cond_beta_multisynapse.cpp``.

    Examples
    --------
    Create a two-receptor neuron (excitatory + inhibitory):

    .. code-block:: python

       >>> from brainpy import state as bp
       >>> import brainunit as u
       >>> neuron = bp.aeif_cond_beta_multisynapse(
       ...     in_size=10,
       ...     tau_rise=(2.0, 0.5) * u.ms,
       ...     tau_decay=(20.0, 8.0) * u.ms,
       ...     E_rev=(0.0, -80.0) * u.mV,  # excitatory, inhibitory
       ... )
       >>> neuron.n_receptors
       2

    Simulate with receptor-specific spike events:

    .. code-block:: python

       >>> import brainstate as bst
       >>> with bst.environ.context(dt=0.1 * u.ms):
       ...     neuron.init_all_states()
       ...     # Excitatory spike to receptor 1
       ...     events = [(1, 5.0 * u.nS)]
       ...     spk = neuron.update(x=100.0 * u.pA, spike_events=events)
       ...     print(neuron.V.value)  # doctest: +SKIP

    Multi-receptor dictionary format:

    .. code-block:: python

       >>> events = [
       ...     {'receptor_type': 1, 'weight': 3.0 * u.nS},
       ...     {'receptor_type': 2, 'weight': 2.0 * u.nS},
       ... ]
       >>> spk = neuron.update(spike_events=events)  # doctest: +SKIP
    """

    __module__ = 'brainpy.state'

    #: Unit the multi-receptor ``connect(receptor_type=k)`` bridge uses to scale the
    #: gathered per-port deposit into the ``w_by_rec`` mantissa (conductance -> nS).
    receptor_input_unit = u.nS

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000
    _EPS = np.finfo(np.float64).eps

    def __init__(
        self,
        in_size: Size,
        V_peak: ArrayLike = 0.0 * u.mV,
        V_reset: ArrayLike = -60.0 * u.mV,
        t_ref: ArrayLike = 0.0 * u.ms,
        g_L: ArrayLike = 30.0 * u.nS,
        C_m: ArrayLike = 281.0 * u.pF,
        E_L: ArrayLike = -70.6 * u.mV,
        Delta_T: ArrayLike = 2.0 * u.mV,
        tau_w: ArrayLike = 144.0 * u.ms,
        a: ArrayLike = 4.0 * u.nS,
        b: ArrayLike = 80.5 * u.pA,
        V_th: ArrayLike = -50.4 * u.mV,
        tau_rise: ArrayLike = (2.0,) * u.ms,
        tau_decay: ArrayLike = (20.0,) * u.ms,
        E_rev: ArrayLike = (0.0,) * u.mV,
        I_e: ArrayLike = 0.0 * u.pA,
        gsl_error_tol: ArrayLike = 1e-6,
        V_initializer: Callable = braintools.init.Constant(-70.6 * u.mV),
        g_initializer: Callable = braintools.init.Constant(0.0 * u.nS),
        w_initializer: Callable = braintools.init.Constant(0.0 * u.pA),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str | None = None,
    ):
        r"""Initialize aeif_cond_beta_multisynapse neuron.

        All parameters are documented in the class docstring.
        """
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.V_peak = braintools.init.param(V_peak, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.Delta_T = braintools.init.param(Delta_T, self.varshape)
        self.tau_w = braintools.init.param(tau_w, self.varshape)
        self.a = braintools.init.param(a, self.varshape)
        self.b = braintools.init.param(b, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        dftype = brainstate.environ.dftype()
        self.tau_rise = np.asarray(u.math.asarray(tau_rise / u.ms), dtype=dftype).reshape(-1)
        self.tau_decay = np.asarray(u.math.asarray(tau_decay / u.ms), dtype=dftype).reshape(-1)
        self.E_rev = np.asarray(u.math.asarray(E_rev / u.mV), dtype=dftype).reshape(-1)

        self.V_initializer = V_initializer
        self.g_initializer = g_initializer
        self.w_initializer = w_initializer
        self.ref_var = ref_var

        self._validate_parameters()
        self._g0 = np.asarray(
            [self._beta_normalization_factor_scalar(tr, td) for tr, td in zip(self.tau_rise, self.tau_decay)],
            dtype=dftype,
        )

        # Per-receptor unit-aware time constants for the vector field.
        self._tau_rise_ms = jnp.asarray(self.tau_rise) * u.ms
        self._tau_decay_ms = jnp.asarray(self.tau_decay) * u.ms
        self._E_rev_mV = jnp.asarray(self.E_rev) * u.mV

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

    @property
    def n_receptors(self):
        r"""Number of receptor ports.

        Returns
        -------
        int
            Number of receptor types, inferred from ``tau_rise.size``.
        """
        return int(self.tau_rise.size)

    @property
    def recordables(self):
        r"""List of recordable state variable names.

        Returns
        -------
        list of str
            Dynamic recordables following NEST naming: ``['V_m', 'w', 'g_1', 'g_2', ..., 'g_n']``.
        """
        return ['V_m', 'w', *[f'g_{i + 1}' for i in range(self.n_receptors)]]

    @classmethod
    def _beta_normalization_factor_scalar(cls, tau_rise: float, tau_decay: float):
        r"""Compute beta normalization factor for single receptor.

        Ensures unit weight produces unit peak conductance. Implements NEST's
        beta normalization formula, degenerating to alpha normalization
        :math:`e / \tau` when :math:`\tau_{\text{rise}} = \tau_{\text{decay}}`.

        Parameters
        ----------
        tau_rise : float
            Synaptic rise time constant (ms, unitless).
        tau_decay : float
            Synaptic decay time constant (ms, unitless).

        Returns
        -------
        float
            Normalization factor :math:`g_0` such that unit weight produces unit peak.
            If :math:`\tau_{\text{rise}} \approx \tau_{\text{decay}}`, returns :math:`e / \tau_{\text{decay}}`.

        Notes
        -----
        The normalization factor is:

        .. math::

           g_0 = \frac{1/\tau_{\text{rise}} - 1/\tau_{\text{decay}}}{\exp(-t_{\text{peak}}/\tau_{\text{decay}}) - \exp(-t_{\text{peak}}/\tau_{\text{rise}})},

        where :math:`t_{\text{peak}} = \tau_{\text{decay}} \tau_{\text{rise}} \log(\tau_{\text{decay}}/\tau_{\text{rise}}) / (\tau_{\text{decay}} - \tau_{\text{rise}})`.
        """
        tau_difference = tau_decay - tau_rise
        peak_value = 0.0
        if abs(tau_difference) > cls._EPS:
            t_peak = tau_decay * tau_rise * np.log(tau_decay / tau_rise) / tau_difference
            peak_value = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)

        if abs(peak_value) < cls._EPS:
            return np.e / tau_decay

        return (1.0 / tau_rise - 1.0 / tau_decay) / peak_value

    def _validate_parameters(self):
        r"""Validate model parameters at initialization.

        Raises
        ------
        ValueError
            If parameter constraints are violated (see class docstring for details).
            Specific checks include:
            - Receptor array size consistency (``tau_rise``, ``tau_decay``, ``E_rev``)
            - Strict positivity (``tau_rise``, ``tau_decay``, ``C_m``, ``tau_w``, ``gsl_error_tol``)
            - Ordering constraints (``tau_decay >= tau_rise``, ``V_peak >= V_th``, ``V_reset < V_peak``)
            - Non-negativity (``Delta_T``, ``t_ref``)
            - Overflow prevention (exponential term when ``Delta_T > 0``)
        """
        v_reset = self.V_reset
        v_peak = self.V_peak
        v_th = self.V_th
        delta_t = self.Delta_T / u.mV

        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (v_reset, v_peak, v_th, delta_t)):
            return

        if self.E_rev.size != self.tau_rise.size or self.E_rev.size != self.tau_decay.size:
            raise ValueError(
                'The reversal potential, synaptic rise time and synaptic decay time arrays must have the same size.')
        if cond_any(self.tau_rise <= 0.0) or cond_any(self.tau_decay <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive')
        if cond_any(self.tau_decay < self.tau_rise):
            raise ValueError('Synaptic rise time must be smaller than or equal to decay time.')
        if cond_any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if cond_any(v_reset >= v_peak):
            raise ValueError('Ensure that: V_reset < V_peak .')
        if cond_any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')
        if cond_any(self.tau_w <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')
        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        # Mirror NEST overflow guard for exponential term at spike time.
        validate_aeif_overflow(v_peak, v_th, delta_t)

    def init_state(self, **kwargs):
        r"""Initialize all state variables.

        Creates ``HiddenState`` and ``ShortTermState`` attributes for membrane
        potential, adaptation current, receptor conductances, refractory counters,
        RKF45 step size, and delayed current buffer.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        Initializes:
        - ``V`` (HiddenState): membrane potential from ``V_initializer``
        - ``w`` (HiddenState): adaptation current from ``w_initializer``
        - ``dg`` (ShortTermState): beta auxiliary states, initialized to zero
        - ``g`` (HiddenState): receptor conductances from ``g_initializer``
        - ``last_spike_time`` (ShortTermState): initialized to -1e7 ms
        - ``refractory_step_count`` (ShortTermState): initialized to 0
        - ``integration_step`` (ShortTermState): RKF45 step size, initialized to ``dt``
        - ``I_stim`` (ShortTermState): delayed current buffer, initialized to 0 pA
        - ``refractory`` (ShortTermState, optional): boolean indicator if ``ref_var=True``
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        w = braintools.init.param(self.w_initializer, self.varshape)
        g = braintools.init.param(self.g_initializer, self.varshape + (self.n_receptors,))

        # dg stored unitless (mantissa in nS/ms)
        zeros_dg = u.math.zeros(self.varshape + (self.n_receptors,), dtype=V.dtype)

        self.V = brainstate.HiddenState(V)
        self.w = brainstate.HiddenState(w)
        self.dg = brainstate.ShortTermState(zeros_dg)
        self.g = brainstate.HiddenState(g)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Compute surrogate spike output for gradient-based learning.

        Applies surrogate gradient function to scaled membrane potential for
        differentiable spike generation. Does not modify state variables.

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (mV). If None, uses current ``self.V.value``.

        Returns
        -------
        ArrayLike
            Surrogate spike output in [0, 1], shape ``(*in_size,)``. Produced by
            ``spk_fun`` applied to ``(V - V_th) / (V_th - V_reset)``.

        Notes
        -----
        This method is primarily used for gradient computation in training contexts.
        Actual spike detection during forward simulation uses hard thresholds in
        ``update`` method.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        r"""Parse incoming spike events into receptor-specific weight array.

        Converts event list/dict format into NumPy array with receptor-specific
        conductance increments, validating receptor types and weight non-negativity.

        Parameters
        ----------
        spike_events : Iterable or None
            Spike events as:
            - List of ``(receptor_type, weight)`` tuples
            - List of dicts with keys ``'receptor_type'``/``'receptor'`` and ``'weight'``
            - Single dict (auto-wrapped to list)
            - None (returns zero array)
        v_shape : tuple
            Neuron population shape for broadcasting.

        Returns
        -------
        np.ndarray
            Weight array (nS, unitless) with shape ``(*v_shape, n_receptors)``.
            Element ``[..., k]`` contains total conductance increment for receptor ``k+1``.

        Raises
        ------
        ValueError
            If receptor type out of range ``[1, n_receptors]``.
        ValueError
            If any weight is negative (conductance constraint).

        Notes
        -----
        Receptor types are 1-based (NEST convention). Internal indexing is 0-based.
        Multiple events for the same receptor are summed.
        """
        dftype = brainstate.environ.dftype()
        out = np.zeros(v_shape + (self.n_receptors,), dtype=dftype)
        if spike_events is None:
            return out

        if isinstance(spike_events, dict):
            spike_events = [spike_events]

        for ev in spike_events:
            if isinstance(ev, dict):
                raw_receptor = ev.get('receptor_type', ev.get('receptor', 1))
                receptor = int(raw_receptor if raw_receptor is not None else 1)
                weight = ev.get('weight', 0.0)
            else:
                receptor, weight = ev
                receptor = int(receptor)

            if receptor <= 0 or receptor > self.n_receptors:
                raise ValueError(f'Receptor type {receptor} out of range [1, {self.n_receptors}].')

            w_np = np.asarray(u.math.asarray(weight / u.nS), dtype=dftype)
            if cond_any(w_np < 0.0):
                raise ValueError('Synaptic weights for conductance-based multisynapse models must be non-negative.')
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, dg, g, w -- ODE state variables.
            ``dg`` and ``g`` have an extra trailing receptor dimension.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_peak_detect -- mutable
            auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, u.math.minimum(state.V, self.V_peak))

        # Synaptic current: sum over receptors g_k * (E_rev_k - V)
        # v_eff has shape (*varshape,), E_rev has shape (n_receptors,)
        # g has shape (*varshape, n_receptors)
        # We need to expand v_eff for broadcasting: (*varshape, 1)
        v_eff_expanded = u.math.expand_dims(v_eff, axis=-1)
        i_syn = u.math.sum(state.g * (self._E_rev_mV - v_eff_expanded), axis=-1)

        delta_t_safe = u.math.where(self.Delta_T == 0.0 * u.mV, 1.0 * u.mV, self.Delta_T)
        exp_arg = u.math.clip((v_eff - self.V_th) / delta_t_safe, -500.0, 500.0)
        i_spike = self.g_L * self.Delta_T * u.math.exp(exp_arg)

        dV_raw = (
                     -self.g_L * (v_eff - self.E_L) + i_spike
                     + i_syn - state.w + self.I_e + extra.i_stim
                 ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        # Beta synapse dynamics per receptor:
        # ddg_k = -dg_k / tau_rise_k
        # dg_k_dt = dg_k - g_k / tau_decay_k
        ddg = -state.dg / self._tau_rise_ms
        dg_dt = state.dg - state.g / self._tau_decay_ms

        dw = (self.a * (v_eff - self.E_L) - state.w) / self.tau_w

        return DotDict(V=dV, dg=ddg, g=dg_dt, w=dw)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection, reset, and refractory handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, dg, g, w -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, v_peak_detect.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & ((state.V < -1e3 * u.mV) | (state.w < -1e6 * u.pA) | (state.w > 1e6 * u.pA))
        )

        refr_accept = accept & (extra.r > 0)
        new_V = u.math.where(refr_accept, self.V_reset, state.V)

        spike_now = accept & (extra.r <= 0) & (new_V >= extra.v_peak_detect)
        spike_mask = extra.spike_mask | spike_now
        new_V = u.math.where(spike_now, self.V_reset, new_V)
        new_w = u.math.where(spike_now, state.w + self.b, state.w)
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict({**state, 'V': new_V, 'w': new_w})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

    def update(self, x=0.0 * u.pA, spike_events=None, w_by_rec=None):
        r"""Advance model by one simulation timestep (NEST-compatible update).

        Integrates ODEs over :math:`(t, t+dt]` using adaptive RKF45 with
        vectorized integration, spike detection, refractory handling, and
        receptor-specific spike event application. Follows NEST's update
        ordering exactly.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous current input (pA), shape broadcastable to ``(*in_size,)``.
            Summed with ``current_inputs`` and ``I_e``, then delayed by one timestep
            (NEST semantics). Default: 0.0 pA.
        spike_events : Iterable or None, optional
            Incoming spike events as:
            - List of ``(receptor_type, weight)`` tuples
            - List of dicts with keys ``'receptor_type'``/``'receptor'`` and ``'weight'``
            - Single dict (auto-wrapped to list)
            - None (no spike input)
            Receptor types are 1-based: ``1 <= receptor_type <= n_receptors``.
            Weights (nS) must be non-negative. Default: None.
        w_by_rec : ArrayLike or None, optional
            Pre-gathered per-receptor conductance jumps (nS mantissa), shape
            ``(*in_size, n_receptors)``. When given, it is used directly and the
            eager ``spike_events``/``add_delta_input`` parsing is bypassed. This is
            the JIT-safe path the multi-receptor ``connect(receptor_type=k)`` seam
            drives via the Simulator bridge. Default: None.

        Returns
        -------
        ArrayLike
            Binary spike indicator (0 or 1), shape ``(*in_size,)``. Float64 for
            gradient compatibility. Value is 1.0 if spike occurred during
            :math:`(t, t+dt]`, else 0.0.

        Raises
        ------
        ValueError
            If receptor type out of range ``[1, n_receptors]``.
        ValueError
            If any spike event weight is negative (conductance constraint).
        ValueError
            If ``add_delta_input`` stream contains negative values (mapped to receptor 1).
        ValueError
            If no receptor ports exist but ``delta_inputs`` or ``spike_events`` are non-zero.
        ValueError
            If numerical instability detected during integration (:math:`V < -1000` mV
            or :math:`|w| > 10^6` pA).

        Notes
        -----
        Integration is performed with an adaptive vectorized RKF45 loop,
        including in-loop spike/reset/adaptation events and optional
        multiple spikes per step. All arithmetic is unit-aware via
        ``brainunit.math``.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        n_receptors = self.n_receptors
        v_shape = self.V.value.shape

        # Read state variables with their natural units.
        V = self.V.value  # mV
        dg = self.dg.value * (u.nS / u.ms)  # stored unitless, restore nS/ms
        g = self.g.value  # nS
        w = self.w.value  # pA
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Spike detection threshold: V_peak if Delta_T > 0, else V_th.
        v_peak_detect = u.math.where(self.Delta_T > 0.0 * u.mV, self.V_peak, self.V_th)

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        if w_by_rec is not None:
            # Bridged blob path (``connect(receptor_type=k)`` seam): per-receptor
            # nS-mantissa conductance jumps, shape ``(*v_shape, n_receptors)``,
            # already gathered by the Simulator. Used directly — no eager parse.
            w_by_rec = jnp.asarray(w_by_rec, dtype=dftype)
        else:
            # Parse spike events into per-receptor weight array.
            w_by_rec = self._parse_spike_events(spike_events, v_shape)

            # Default delta input mapped to receptor 1.
            # Use jnp.asarray (not np.asarray) so this path is JIT-compatible inside
            # brainstate.transform.for_loop, where sum_delta_inputs may return a tracer.
            w_default = u.get_mantissa(u.math.asarray(self.sum_delta_inputs(0.0 * u.nS) / u.nS))
            w_default = jnp.broadcast_to(jnp.asarray(w_default, dtype=dftype), v_shape)
            if n_receptors > 0:
                # Guard with is_tracer: concrete values support Python-level ValueError;
                # traced values (inside JIT) skip the eager check safely.
                if not is_tracer(w_default) and cond_any(np.asarray(w_default) < 0.0):
                    raise ValueError('Synaptic weights for conductance-based multisynapse models must be non-negative.')
                # Use JAX immutable update so w_by_rec stays JIT-compatible.
                w_by_rec = jnp.asarray(w_by_rec).at[..., 0].add(w_default)
            elif not is_tracer(w_default) and cond_any(np.asarray(w_default) != 0.0):
                raise ValueError('No receptor ports available for incoming spike conductance.')

        # Beta normalization factors (unitless, per receptor).
        g0 = np.broadcast_to(self._g0, v_shape + (n_receptors,))

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, dg=dg, g=g, w=w)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            v_peak_detect=v_peak_detect,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V, dg, g, w = ode_state.V, ode_state.dg, ode_state.g, ode_state.w
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in aeif_cond_beta_multisynapse dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Apply incoming spike events to dg states with beta normalization.
        # g0 has shape (*v_shape, n_receptors), w_by_rec has shape (*v_shape, n_receptors)
        # g0 * w_by_rec gives nS (unitless), need to convert to nS/ms for dg units
        # In NEST beta multisynapse: dg_k += g0_k * w_k where g0 has units 1/ms
        # so g0 * w (nS) gives nS/ms
        dg_increment = jnp.asarray(g0 * w_by_rec) * (u.nS / u.ms)
        dg = dg + dg_increment

        # Write back state.
        self.V.value = V
        self.dg.value = u.get_mantissa(dg)  # store unitless mantissa
        self.g.value = g
        self.w.value = w
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
