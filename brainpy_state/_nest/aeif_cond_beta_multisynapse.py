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
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

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
        Beta auxiliary states (unitless), shape ``(*in_size, n_receptors)``.
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
    - This implementation prioritizes NEST compatibility over vectorization.
      Per-neuron scalar loops ensure identical update ordering and spike semantics.
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

       >>> import brainpy.state as bp
       >>> import saiunit as u
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

    _MIN_H = 1e-8  # ms
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
        name: str = None,
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
        self.gsl_error_tol = braintools.init.param(gsl_error_tol, self.varshape)

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

    @staticmethod
    def _to_numpy(x, unit):
        r"""Convert quantity to NumPy array in specified unit.

        Parameters
        ----------
        x : ArrayLike
            Quantity with units.
        unit : saiunit.Unit
            Target unit for conversion.

        Returns
        -------
        np.ndarray
            Float64 array in target unit (unitless).
        """
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x / unit), dtype=dftype)

    @staticmethod
    def _to_numpy_unitless(x):
        r"""Convert unitless quantity to NumPy array.

        Parameters
        ----------
        x : ArrayLike
            Unitless quantity.

        Returns
        -------
        np.ndarray
            Float64 array.
        """
        dftype = brainstate.environ.dftype()
        return np.asarray(u.math.asarray(x), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast parameter to state shape.

        Parameters
        ----------
        x_np : np.ndarray
            Parameter array to broadcast.
        shape : tuple
            Target shape (neuron population shape).

        Returns
        -------
        np.ndarray
            Broadcast array with shape ``shape``.
        """
        return np.broadcast_to(x_np, shape)

    @staticmethod
    def _broadcast_to_receptors(x_np: np.ndarray, shape, n_receptors: int):
        r"""Broadcast parameter to receptor array shape.

        Parameters
        ----------
        x_np : np.ndarray
            Parameter array to broadcast.
        shape : tuple
            Neuron population shape.
        n_receptors : int
            Number of receptor ports.

        Returns
        -------
        np.ndarray
            Broadcast array with shape ``(*shape, n_receptors)``.
        """
        return np.broadcast_to(x_np, shape + (n_receptors,))

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
        v_reset = self._to_numpy(self.V_reset, u.mV)
        v_peak = self._to_numpy(self.V_peak, u.mV)
        v_th = self._to_numpy(self.V_th, u.mV)
        delta_t = self._to_numpy(self.Delta_T, u.mV)

        if self.E_rev.size != self.tau_rise.size or self.E_rev.size != self.tau_decay.size:
            raise ValueError(
                'The reversal potential, synaptic rise time and synaptic decay time arrays must have the same size.')
        if np.any(self.tau_rise <= 0.0) or np.any(self.tau_decay <= 0.0):
            raise ValueError('All synaptic time constants must be strictly positive')
        if np.any(self.tau_decay < self.tau_rise):
            raise ValueError('Synaptic rise time must be smaller than or equal to decay time.')
        if np.any(v_peak < v_th):
            raise ValueError('V_peak >= V_th required.')
        if np.any(v_reset >= v_peak):
            raise ValueError('Ensure that: V_reset < V_peak .')
        if np.any(delta_t < 0.0):
            raise ValueError('Delta_T must be positive.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')
        if np.any(self._to_numpy(self.tau_w, u.ms) <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(self._to_numpy_unitless(self.gsl_error_tol) <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

        positive_dt = delta_t > 0.0
        if np.any(positive_dt):
            max_exp_arg = np.log(np.finfo(np.float64).max / 1e20)
            ratio = (v_peak - v_th) / np.where(positive_dt, delta_t, 1.0)
            if np.any(ratio[positive_dt] >= max_exp_arg):
                raise ValueError(
                    'The current combination of V_peak, V_th and Delta_T will lead to numerical overflow at spike '
                    'time; try for instance to increase Delta_T or to reduce V_peak to avoid this problem.'
                )

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Creates ``HiddenState`` and ``ShortTermState`` attributes for membrane
        potential, adaptation current, receptor conductances, refractory counters,
        RKF45 step size, and delayed current buffer.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension prepended to ``in_size``. If None, states have shape
            ``(*in_size,)`` or ``(*in_size, n_receptors)`` for receptor arrays.
        **kwargs
            Additional initialization arguments (unused).

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
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        w = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        g = braintools.init.param(self.g_initializer, self.varshape + (self.n_receptors,), batch_size)

        self.V = brainstate.HiddenState(V)
        self.w = brainstate.HiddenState(w)
        dftype = brainstate.environ.dftype()
        self.dg = brainstate.ShortTermState(np.zeros(g.shape, dtype=dftype))
        self.g = brainstate.HiddenState(g)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        ditype = brainstate.environ.ditype()
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=ditype))

        dt = brainstate.environ.get_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to initial values.

        Reapplies initializers to all state variables without recreating state
        objects. Useful for resetting between simulation trials.

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension for reinitialization. If None, uses existing batch size.
        **kwargs
            Additional reset arguments (unused).
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.w.value = braintools.init.param(self.w_initializer, self.varshape, batch_size)
        self.g.value = braintools.init.param(self.g_initializer, self.varshape + (self.n_receptors,), batch_size)
        dftype = brainstate.environ.dftype()
        self.dg.value = np.zeros(np.asarray(self.g.value).shape, dtype=dftype)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        ditype = brainstate.environ.ditype()
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=ditype)
        dt = brainstate.environ.get_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )
        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

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

    def _refractory_counts(self):
        r"""Compute refractory period in simulation timesteps.

        Returns
        -------
        ArrayLike
            Number of timesteps for refractory period, shape ``(*in_size,)``.
            Computed as :math:`\lceil t_{\text{ref}} / dt \rceil` (int32).

        Notes
        -----
        Returns 0 when ``t_ref = 0``. Used to initialize refractory counter on
        spike detection.
        """
        dt = brainstate.environ.get_dt()
        ditype = brainstate.environ.ditype()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

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
                receptor = int(ev.get('receptor_type', ev.get('receptor', 1)))
                weight = ev.get('weight', 0.0)
            else:
                receptor, weight = ev
                receptor = int(receptor)

            if receptor <= 0 or receptor > self.n_receptors:
                raise ValueError(f'Receptor type {receptor} out of range [1, {self.n_receptors}].')

            w_np = np.asarray(u.math.asarray(weight / u.nS), dtype=dftype)
            if np.any(w_np < 0.0):
                raise ValueError('Synaptic weights for conductance-based multisynapse models must be non-negative.')
            out[..., receptor - 1] += np.broadcast_to(w_np, v_shape)
        return out

    @staticmethod
    def _dynamics_scalar(y, is_refractory, i_stim, p):
        r"""Compute ODE right-hand side for single neuron (RKF45 substep).

        Parameters
        ----------
        y : np.ndarray
            State vector with shape ``(2 + 2*n_receptors,)``::
                y[0]: V (mV, unitless)
                y[1]: w (pA, unitless)
                y[2::2]: dg[k] (unitless)
                y[3::2]: g[k] (nS, unitless)
        is_refractory : bool
            Whether neuron is in refractory period.
        i_stim : float
            Delayed injected current (pA, unitless).
        p : dict
            Parameter dict with keys: ``'V_reset'``, ``'V_peak_rhs'``, ``'E_L'``,
            ``'C_m'``, ``'g_L'``, ``'Delta_T'``, ``'tau_w'``, ``'a'``, ``'V_th'``,
            ``'I_e'``, ``'tau_rise'``, ``'tau_decay'``, ``'E_rev'`` (all unitless
            scalars or receptor arrays).

        Returns
        -------
        np.ndarray
            Derivative vector ``dy/dt`` with same shape as ``y`` (unitless).

        Notes
        -----
        - During refractory, effective voltage is clamped to ``V_reset`` and ``dV/dt = 0``.
        - Outside refractory, exponential term uses ``min(V, V_peak_rhs)`` to prevent overflow.
        - Adaptation coupling ``a * (V_eff - E_L)`` persists during refractory.
        - Beta states evolve independently of refractory status.
        """
        v_eff = p['V_reset'] if is_refractory else min(y[0], p['V_peak_rhs'])
        w = y[1]
        dg = y[2::2]
        g = y[3::2]

        i_syn = float(np.sum(g * (p['E_rev'] - v_eff)))
        i_spike = 0.0 if p['Delta_T'] == 0.0 else (
            p['Delta_T'] * p['g_L'] * math.exp((v_eff - p['V_th']) / p['Delta_T'])
        )
        dv = 0.0 if is_refractory else (
                                           -p['g_L'] * (v_eff - p['E_L']) + i_spike + i_syn - w + p['I_e'] + i_stim
                                       ) / p['C_m']
        dw = (p['a'] * (v_eff - p['E_L']) - w) / p['tau_w']

        dy = np.empty_like(y)
        dy[0] = dv
        dy[1] = dw
        dy[2::2] = -dg / p['tau_rise']
        dy[3::2] = dg - g / p['tau_decay']
        return dy

    def update(self, x=0.0 * u.pA, spike_events=None):
        r"""Advance model by one simulation timestep (NEST-compatible update).

        Integrates ODEs over :math:`(t, t+dt]` using adaptive RKF45 with per-neuron
        scalar loops, spike detection, refractory handling, and receptor-specific
        spike event application. Follows NEST's update ordering exactly.

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
        **Update Order (NEST Semantics)**

        1. **ODE Integration**: Adaptive RKF45 integrates :math:`(V, w, dg, g)` from
           :math:`t` to :math:`t+dt` using internal substeps. Integration step size
           persists across simulation steps.

        2. **Refractory Handling**: During refractory integration, :math:`V` is clamped
           to :math:`V_{\text{reset}}` and :math:`dV/dt = 0`. Adaptation :math:`w`
           continues to evolve.

        3. **Spike Detection**: Within each RKF45 substep, if :math:`V \geq V_{\text{peak}}`
           (or :math:`V \geq V_{th}` if :math:`\Delta_T = 0`):

           - :math:`V \leftarrow V_{\text{reset}}`
           - :math:`w \leftarrow w + b`
           - Refractory counter :math:`r \leftarrow \lceil t_{\text{ref}} / dt \rceil + 1`
             (if ``t_ref > 0``)

        4. **Refractory Decrement**: After integration completes, refractory counter
           is decremented once: :math:`r \leftarrow \max(0, r - 1)`.

        5. **Spike Event Application**: Incoming weights are applied to ``dg`` states:
           :math:`dg_k \leftarrow dg_k + g_{0,k} w_k`, where :math:`g_{0,k}` is the
           beta normalization factor.

        6. **Current Delay**: Current input :math:`x + \sum \text{current\_inputs}`
           dftype = brainstate.environ.dftype()
           ditype = brainstate.environ.ditype()
           is stored in ``I_stim`` for use in the **next** timestep (one-step delay,
           matching NEST's event delivery semantics).

        **Computational Details**

        - **Per-neuron scalar loops**: Each neuron integrates independently to ensure
          NEST-compatible update ordering and spike semantics.
        - **Adaptive step size**: RKF45 local error tolerance ``gsl_error_tol``
          controls accuracy. Step size adapts dynamically (5th-order error estimate).
        - **Instability detection**: Raises ``ValueError`` if :math:`V < -1000` mV
          or :math:`|w| > 10^6` pA, indicating parameter issues or numerical collapse.
        - **Default input mapping**: ``add_delta_input`` stream maps to receptor 1.
          If no receptors exist, non-zero delta inputs raise ``ValueError``.
        - **State updates**: All state variables (``V``, ``w``, ``dg``, ``g``,
          ``refractory_step_count``, ``integration_step``, ``I_stim``,
          ``last_spike_time``, optionally ``refractory``) are updated in-place.

        **Performance Notes**

        This implementation prioritizes NEST compatibility over vectorization.
        Per-neuron loops with Python scalar operations are ~10-100x slower than
        fully vectorized implementations but ensure identical spike timing and
        update ordering. For large-scale simulations, consider NEST-GPU or
        vectorized BrainPy models.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        v_shape = self.V.value.shape
        n_receptors = self.n_receptors

        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape)
        w = self._broadcast_to_state(self._to_numpy(self.w.value, u.pA), v_shape)
        dg = self._broadcast_to_receptors(np.asarray(self.dg.value, dtype=dftype), v_shape, n_receptors)
        g = self._broadcast_to_receptors(self._to_numpy(self.g.value, u.nS), v_shape, n_receptors)
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=ditype),
            v_shape,
        )
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), v_shape)

        p = {
            'V_peak_rhs': self._broadcast_to_state(self._to_numpy(self.V_peak, u.mV), v_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape),
            'E_L': self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape),
            'C_m': self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape),
            'g_L': self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape),
            'Delta_T': self._broadcast_to_state(self._to_numpy(self.Delta_T, u.mV), v_shape),
            'tau_w': self._broadcast_to_state(self._to_numpy(self.tau_w, u.ms), v_shape),
            'a': self._broadcast_to_state(self._to_numpy(self.a, u.nS), v_shape),
            'b': self._broadcast_to_state(self._to_numpy(self.b, u.pA), v_shape),
            'V_th': self._broadcast_to_state(self._to_numpy(self.V_th, u.mV), v_shape),
            'I_e': self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape),
            'atol': self._broadcast_to_state(self._to_numpy_unitless(self.gsl_error_tol), v_shape),
            'tau_rise': self._broadcast_to_receptors(self.tau_rise, v_shape, n_receptors),
            'tau_decay': self._broadcast_to_receptors(self.tau_decay, v_shape, n_receptors),
            'E_rev': self._broadcast_to_receptors(self.E_rev, v_shape, n_receptors),
        }

        v_peak_detect = np.where(p['Delta_T'] > 0.0, p['V_peak_rhs'], p['V_th'])
        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=ditype),
            v_shape,
        )

        w_by_rec = self._parse_spike_events(spike_events, v_shape)
        w_default = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0.0 * u.nS), u.nS), v_shape)
        if n_receptors > 0:
            if np.any(w_default < 0.0):
                raise ValueError('Synaptic weights for conductance-based multisynapse models must be non-negative.')
            w_by_rec[..., 0] += w_default
        elif np.any(w_default != 0.0):
            raise ValueError('No receptor ports available for incoming spike conductance.')

        g0 = self._broadcast_to_receptors(self._g0, v_shape, n_receptors)
        new_i_stim_q = self.sum_current_inputs(x, self.V.value)
        new_i_stim = self._broadcast_to_state(self._to_numpy(new_i_stim_q, u.pA), v_shape)

        spike_mask = np.zeros(v_shape, dtype=bool)
        V_next = np.empty_like(V)
        w_next = np.empty_like(w)
        dg_next = np.empty_like(dg)
        g_next = np.empty_like(g)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(v_shape):
            local_p = {k: p[k][idx] for k in p}
            y = np.empty(2 + 2 * n_receptors, dtype=dftype)
            y[0] = V[idx]
            y[1] = w[idx]
            y[2::2] = dg[idx]
            y[3::2] = g[idx]

            r_i = int(r[idx])
            h_i = float(max(h_int[idx], self._MIN_H))
            t_local = 0.0
            iters = 0
            local_spike = False

            while t_local < dt and iters < self._MAX_ITERS:
                iters += 1
                h_i = max(self._MIN_H, min(h_i, dt - t_local))
                is_refractory = r_i > 0

                def f(y_):
                    return np.asarray(self._dynamics_scalar(y_, is_refractory, i_stim[idx], local_p), dtype=dftype)

                k1 = f(y)
                k2 = f(y + h_i * (1.0 / 4.0) * k1)
                k3 = f(y + h_i * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
                k4 = f(y + h_i * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0))
                k5 = f(y + h_i * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0))
                k6 = f(
                    y
                    + h_i
                    * (
                        -8.0 * k1 / 27.0
                        + 2.0 * k2
                        - 3544.0 * k3 / 2565.0
                        + 1859.0 * k4 / 4104.0
                        - 11.0 * k5 / 40.0
                    )
                )

                y4 = y + h_i * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
                y5 = y + h_i * (
                    16.0 * k1 / 135.0
                    + 6656.0 * k3 / 12825.0
                    + 28561.0 * k4 / 56430.0
                    - 9.0 * k5 / 50.0
                    + 2.0 * k6 / 55.0
                )
                err = float(np.max(np.abs(y5 - y4)))
                atol = float(local_p['atol'])

                if err <= atol or h_i <= self._MIN_H:
                    y = y5
                    t_local += h_i
                    fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2))
                    h_i = max(self._MIN_H, h_i * fac)

                    if y[0] < -1e3 or y[1] < -1e6 or y[1] > 1e6:
                        raise ValueError('Numerical instability in aeif_cond_beta_multisynapse dynamics.')

                    if r_i > 0:
                        y[0] = local_p['V_reset']
                    elif y[0] >= v_peak_detect[idx]:
                        local_spike = True
                        y[0] = local_p['V_reset']
                        y[1] += local_p['b']
                        r_i = int(refr_counts[idx]) + 1 if int(refr_counts[idx]) > 0 else 0
                else:
                    fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
                    h_i = max(self._MIN_H, h_i * fac)

            if r_i > 0:
                r_i -= 1

            y[2::2] += g0[idx] * w_by_rec[idx]

            spike_mask[idx] = local_spike
            V_next[idx] = y[0]
            w_next[idx] = y[1]
            dg_next[idx] = y[2::2]
            g_next[idx] = y[3::2]
            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.w.value = w_next * u.pA
        self.dg.value = dg_next
        self.g.value = g_next * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=ditype)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
