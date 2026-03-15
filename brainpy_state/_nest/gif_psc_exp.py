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

from typing import Callable, Optional, Sequence

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, AdaptiveRungeKuttaStep

__all__ = [
    'gif_psc_exp',
]


class gif_psc_exp(NESTNeuron):
    r"""Current-based generalized integrate-and-fire neuron (GIF) model.

    This is a brainpy.state re-implementation of the NEST simulator's ``gif_psc_exp``
    model according to Mensi et al. (2012) [1]_ and Pozzorini et al. (2015) [2]_, using
    NEST-standard parameterization and exact integration.

    The GIF model features both spike-triggered adaptation currents and a dynamic
    firing threshold for spike-frequency adaptation. It generates spikes stochastically
    based on a point process with intensity that depends on the distance between the
    membrane potential and the adaptive threshold.

    **1. Mathematical Model**

    **1.1 Membrane Dynamics**

    The membrane potential :math:`V` is governed by:

    .. math::

       C_\mathrm{m} \frac{dV(t)}{dt} = -g_\mathrm{L}(V(t) - E_\mathrm{L})
           - \eta_1(t) - \eta_2(t) - \ldots - \eta_n(t) + I(t)

    where:

    - :math:`C_\mathrm{m}` is the membrane capacitance
    - :math:`g_\mathrm{L}` is the leak conductance
    - :math:`E_\mathrm{L}` is the leak reversal potential
    - :math:`\eta_i(t)` are spike-triggered currents (stc)
    - :math:`I(t) = I_\mathrm{syn,ex}(t) + I_\mathrm{syn,in}(t) + I_\mathrm{e} + I_\mathrm{stim}(t)`

    **1.2 Synaptic Currents**

    Synaptic currents decay exponentially:

    .. math::

       \frac{dI_{\mathrm{syn,ex}}}{dt} = -\frac{I_{\mathrm{syn,ex}}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{dI_{\mathrm{syn,in}}}{dt} = -\frac{I_{\mathrm{syn,in}}}{\tau_{\mathrm{syn,in}}}

    Incoming spike weights (in pA) are routed by sign: positive weights to
    :math:`I_{\mathrm{syn,ex}}`, negative to :math:`I_{\mathrm{syn,in}}`.

    **1.3 Spike-Triggered Currents (STC)**

    Each spike-triggered current element :math:`\eta_i` evolves as:

    .. math::

       \tau_{\eta_i} \frac{d\eta_i}{dt} = -\eta_i

    On spike emission:

    .. math::

       \eta_i \leftarrow \eta_i + q_{\eta_i}

    **1.4 Spike-Frequency Adaptation (SFA)**

    The neuron fires stochastically with intensity:

    .. math::

       \lambda(t) = \lambda_0 \cdot \exp\left(\frac{V(t) - V_T(t)}{\Delta_V}\right)

    where the dynamic threshold :math:`V_T(t)` is:

    .. math::

       V_T(t) = V_{T^*} + \gamma_1(t) + \gamma_2(t) + \ldots + \gamma_m(t)

    Each adaptation element :math:`\gamma_i` evolves as:

    .. math::

       \tau_{\gamma_i} \frac{d\gamma_i}{dt} = -\gamma_i

    On spike emission:

    .. math::

       \gamma_i \leftarrow \gamma_i + q_{\gamma_i}

    **1.5 Stochastic Spiking**

    The probability of firing within a time step :math:`dt` is:

    .. math::

       P(\text{spike}) = 1 - \exp(-\lambda(t) \cdot dt)

    A uniformly distributed random number is drawn each (non-refractory) time step
    and compared to this probability to determine spike emission.

    **1.6 Refractory Period**

    After a spike, the neuron enters an absolute refractory period of duration
    :math:`t_\mathrm{ref}`. During this period:

    - The refractory counter decrements each step
    - :math:`V_\mathrm{m}` is clamped to :math:`V_\mathrm{reset}`
    - Synaptic currents continue to decay and receive inputs
    - No spike checks are performed

    **2. Numerical Integration**

    The model uses adaptive Runge-Kutta-Fehlberg (RKF45) integration for the
    continuous dynamics, with per-substep event callbacks for spike detection,
    refractory clamping, and adaptation jumps.

    The discrete-time update order per simulation step is:

    1. **ODE integration**: Integrate all continuous dynamics (membrane, synaptic
       currents, STC elements, SFA elements) via adaptive RKF45. Inside the
       integration loop: apply refractory clamp and stochastic spike/reset/adaptation.
    2. **Post-loop**: Decrement refractory counter once.
    3. **Synaptic input**: Apply arriving spike weights to ``I_syn_ex``/``I_syn_in``.
    4. **Buffer external input**: Store ``I_stim`` for the next step
       (NEST ring buffer semantics).

    Parameters
    ----------
    in_size : int, tuple of int
        Shape of the neuron population. Can be an integer for 1D or a tuple for
        multi-dimensional populations.
    g_L : ArrayLike, optional
        Leak conductance :math:`g_\mathrm{L}`. Scalar or array matching ``in_size``.
        Must be strictly positive. Default: 4.0 nS.
    E_L : ArrayLike, optional
        Leak reversal potential :math:`E_\mathrm{L}`. Scalar or array matching
        ``in_size``. Default: -70.0 mV.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_\mathrm{m}`. Scalar or array matching ``in_size``.
        Must be strictly positive. Default: 80.0 pF.
    V_reset : ArrayLike, optional
        Reset potential :math:`V_\mathrm{reset}` after spike. Scalar or array matching
        ``in_size``. Default: -55.0 mV.
    Delta_V : ArrayLike, optional
        Stochasticity level :math:`\Delta_V` (noise intensity). Scalar or array matching
        ``in_size``. Must be strictly positive. Default: 0.5 mV.
    V_T_star : ArrayLike, optional
        Base firing threshold :math:`V_{T^*}` (before adaptation). Scalar or array
        matching ``in_size``. Default: -35.0 mV.
    lambda_0 : float, optional
        Stochastic intensity at threshold :math:`\lambda_0` in 1/s. Must be non-negative.
        Default: 1.0 /s (converted internally to 1/ms).
    t_ref : ArrayLike, optional
        Absolute refractory period :math:`t_\mathrm{ref}`. Scalar or array matching
        ``in_size``. Must be non-negative. Default: 4.0 ms.
    tau_syn_ex : ArrayLike, optional
        Excitatory synaptic time constant :math:`\tau_{\mathrm{syn,ex}}`. Scalar or
        array matching ``in_size``. Must be strictly positive. Default: 2.0 ms.
    tau_syn_in : ArrayLike, optional
        Inhibitory synaptic time constant :math:`\tau_{\mathrm{syn,in}}`. Scalar or
        array matching ``in_size``. Must be strictly positive. Default: 2.0 ms.
    I_e : ArrayLike, optional
        Constant external current :math:`I_\mathrm{e}`. Scalar or array matching
        ``in_size``. Default: 0.0 pA.
    tau_sfa : Sequence[float], optional
        SFA time constants :math:`\tau_{\gamma_i}` in ms. Each element must be strictly
        positive. Length must match ``q_sfa``. Default: () (no SFA).
    q_sfa : Sequence[float], optional
        SFA jump values :math:`q_{\gamma_i}` in mV (added to :math:`\gamma_i` on spike).
        Length must match ``tau_sfa``. Default: () (no SFA).
    tau_stc : Sequence[float], optional
        STC time constants :math:`\tau_{\eta_i}` in ms. Each element must be strictly
        positive. Length must match ``q_stc``. Default: () (no STC).
    q_stc : Sequence[float], optional
        STC jump values :math:`q_{\eta_i}` in nA (added to :math:`\eta_i` on spike).
        Length must match ``tau_stc``. Default: () (no STC).
    gsl_error_tol : ArrayLike, optional
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
        Default: 1e-6.
    rng_key : jax.Array, optional
        JAX PRNG key for stochastic spiking. If None, uses a default key (seed 0).
        For reproducible results, provide an explicit key. Default: None.
    V_initializer : Callable, optional
        Initializer for membrane potential. Must accept shape arguments.
        Default: ``braintools.init.Constant(-70.0 * u.mV)``.
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation. Used for gradient-based
        learning. Default: ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Spike reset mode. ``'hard'`` (stop gradient) matches NEST behavior; ``'soft'``
        subtracts threshold. Default: ``'hard'``.
    ref_var : bool, optional
        If ``True``, allocate and expose ``self.refractory`` state. Default: False.
    name : str, optional
        Name of the neuron group. Default: None (auto-generated).

    Parameter Mapping
    -----------------

    ==================== =================== =================================== =====================================================
    **Parameter**        **Default**         **Math equivalent**                 **Description**
    ==================== =================== =================================== =====================================================
    ``in_size``          (required)          —                                   Population shape
    ``g_L``              4.0 nS              :math:`g_\mathrm{L}`                Leak conductance
    ``E_L``              -70.0 mV            :math:`E_\mathrm{L}`                Leak reversal potential
    ``C_m``              80.0 pF             :math:`C_\mathrm{m}`                Membrane capacitance
    ``V_reset``          -55.0 mV            :math:`V_\mathrm{reset}`            Reset potential
    ``Delta_V``          0.5 mV              :math:`\Delta_V`                    Stochasticity level
    ``V_T_star``         -35.0 mV            :math:`V_{T^*}`                     Base firing threshold
    ``lambda_0``         1.0 /s              :math:`\lambda_0`                   Stochastic intensity at threshold
    ``t_ref``            4.0 ms              :math:`t_\mathrm{ref}`              Absolute refractory period
    ``tau_syn_ex``       2.0 ms              :math:`\tau_{\mathrm{syn,ex}}`      Excitatory synaptic time constant
    ``tau_syn_in``       2.0 ms              :math:`\tau_{\mathrm{syn,in}}`      Inhibitory synaptic time constant
    ``I_e``              0.0 pA              :math:`I_\mathrm{e}`                Constant external current
    ``tau_sfa``          () ms               :math:`\tau_{\gamma_i}`             SFA time constants (tuple/list)
    ``q_sfa``            () mV               :math:`q_{\gamma_i}`                SFA jump values (tuple/list)
    ``tau_stc``          () ms               :math:`\tau_{\eta_i}`               STC time constants (tuple/list)
    ``q_stc``            () nA               :math:`q_{\eta_i}`                  STC jump values (tuple/list)
    ``gsl_error_tol``    1e-6                —                                   RKF45 absolute error tolerance
    ``rng_key``          None                —                                   JAX PRNG key for stochastic spiking
    ``V_initializer``    Constant(-70 mV)    —                                   Initializer for membrane potential
    ``spk_fun``          ReluGrad()          —                                   Surrogate spike function
    ``spk_reset``        ``'hard'``          —                                   Reset mode; hard reset matches NEST
    ``ref_var``          False               —                                   If True, expose boolean refractory state
    ==================== =================== =================================== =====================================================

    Raises
    ------
    ValueError
        - If ``C_m <= 0`` (capacitance must be strictly positive).
        - If ``g_L <= 0`` (conductance must be strictly positive).
        - If ``Delta_V <= 0`` (stochasticity level must be strictly positive).
        - If ``t_ref < 0`` (refractory time cannot be negative).
        - If ``lambda_0 < 0`` (intensity cannot be negative).
        - If ``tau_syn_ex <= 0`` or ``tau_syn_in <= 0`` (synaptic time constants must
          be strictly positive).
        - If lengths of ``tau_sfa`` and ``q_sfa`` do not match.
        - If lengths of ``tau_stc`` and ``q_stc`` do not match.
        - If any element of ``tau_sfa`` or ``tau_stc`` is non-positive.
        - If ``gsl_error_tol <= 0``.

    Warnings
    --------
    - Stochastic spiking: Because spiking is stochastic (random number drawn each
      step), exact spike-time reproducibility requires matching the random number
      generator state. For deterministic testing, set ``rng_key`` explicitly.
    - GIF toolbox compatibility: In the NEST implementation, stc and sfa element
      jumps occur immediately after spike emission. The GIF toolbox uses a different
      convention where jumps occur after the refractory period. Conversion:

      .. math::

         q_{\eta,\text{toolbox}} = q_{\eta,\text{NEST}} \cdot
             (1 - \exp(-t_\mathrm{ref} / \tau_\eta))

    Notes
    -----
    - Defaults follow NEST C++ source for ``gif_psc_exp`` (``models/gif_psc_exp.cpp``).
    - ``lambda_0`` is specified in 1/s (as in NEST's Python interface) and is
      internally converted to 1/ms for computation.
    - Synaptic spike weights are interpreted in current units (pA), with positive/
      negative sign selecting excitatory/inhibitory channel.
    - The dynamics use adaptive RKF45 integration with per-substep event callbacks
      for spike detection, refractory clamping, and adaptation jumps.
    - State variables (``V``, ``I_syn_ex``, ``I_syn_in``, etc.) are accessible as
      attributes after calling ``init_state()``.

    Examples
    --------
    Create a single GIF neuron with default parameters:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import saiunit as u
        >>> import brainstate as bs
        >>> bs.environ.set(dt=0.1 * u.ms)
        >>> neuron = bst.gif_psc_exp(in_size=1)
        >>> neuron.init_state()
        >>> spike = neuron.update(x=100 * u.pA)

    Create a population with spike-triggered current adaptation:

    .. code-block:: python

        >>> neuron = bst.gif_psc_exp(
        ...     in_size=100,
        ...     tau_stc=[5.0, 50.0],  # two stc elements
        ...     q_stc=[10.0, 50.0],   # jump values in nA
        ... )
        >>> neuron.init_state()

    Create a neuron with spike-frequency adaptation (dynamic threshold):

    .. code-block:: python

        >>> neuron = bst.gif_psc_exp(
        ...     in_size=1,
        ...     tau_sfa=[10.0, 100.0],  # two sfa elements
        ...     q_sfa=[5.0, 10.0],      # jump values in mV
        ... )
        >>> neuron.init_state()

    Use a custom RNG key for reproducible stochastic spiking:

    .. code-block:: python

        >>> import jax
        >>> key = jax.random.PRNGKey(42)
        >>> neuron = bst.gif_psc_exp(in_size=1, rng_key=key)
        >>> neuron.init_state()

    References
    ----------
    .. [1] Mensi S, Naud R, Pozzorini C, Avermann M, Petersen CC, Gerstner W
           (2012). Parameter extraction and classification of three cortical
           neuron types reveals two distinct adaptation mechanisms. Journal of
           Neurophysiology, 107(6):1756-1775.
           DOI: https://doi.org/10.1152/jn.00408.2011
    .. [2] Pozzorini C, Mensi S, Hagens O, Naud R, Koch C, Gerstner W (2015).
           Automated high-throughput characterization of single neurons by means
           of simplified spiking models. PLoS Computational Biology, 11(6),
           e1004275.
           DOI: https://doi.org/10.1371/journal.pcbi.1004275
    .. [3] NEST Simulator ``gif_psc_exp`` model documentation and C++ source:
           ``models/gif_psc_exp.h`` and ``models/gif_psc_exp.cpp``.

    See Also
    --------
    gif_cond_exp : Conductance-based GIF model with adaptive integration.
    iaf_psc_exp : Simple integrate-and-fire neuron with exponential synapses.
    gif_psc_exp_multisynapse : GIF model with multiple receptor ports.
    """
    __module__ = 'brainpy.state'

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        g_L: ArrayLike = 4.0 * u.nS,
        E_L: ArrayLike = -70.0 * u.mV,
        C_m: ArrayLike = 80.0 * u.pF,
        V_reset: ArrayLike = -55.0 * u.mV,
        Delta_V: ArrayLike = 0.5 * u.mV,
        V_T_star: ArrayLike = -35.0 * u.mV,
        lambda_0: float = 1.0,  # 1/s, as in NEST Python interface
        t_ref: ArrayLike = 4.0 * u.ms,
        tau_syn_ex: ArrayLike = 2.0 * u.ms,
        tau_syn_in: ArrayLike = 2.0 * u.ms,
        I_e: ArrayLike = 0.0 * u.pA,
        tau_sfa: Sequence[float] = (),  # ms values
        q_sfa: Sequence[float] = (),  # mV values
        tau_stc: Sequence[float] = (),  # ms values
        q_stc: Sequence[float] = (),  # nA values
        gsl_error_tol: ArrayLike = 1e-6,
        rng_key: Optional[jax.Array] = None,
        V_initializer: Callable = braintools.init.Constant(-70.0 * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # Membrane parameters
        self.g_L = braintools.init.param(g_L, self.varshape)
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.Delta_V = braintools.init.param(Delta_V, self.varshape)
        self.V_T_star = braintools.init.param(V_T_star, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)

        # Synaptic parameters
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)

        # Stochastic spiking: lambda_0 in 1/s, store as 1/ms internally
        self.lambda_0 = lambda_0 / 1000.0  # convert from 1/s to 1/ms

        # Adaptation parameters (stored as plain Python tuples of floats in ms/mV/nA)
        self.tau_sfa = tuple(float(x) for x in tau_sfa)
        self.q_sfa = tuple(float(x) for x in q_sfa)
        self.tau_stc = tuple(float(x) for x in tau_stc)
        self.q_stc = tuple(float(x) for x in q_stc)

        if len(self.tau_sfa) != len(self.q_sfa):
            raise ValueError(
                f"'tau_sfa' and 'q_sfa' must have the same length. "
                f"Got {len(self.tau_sfa)} and {len(self.q_sfa)}."
            )
        if len(self.tau_stc) != len(self.q_stc):
            raise ValueError(
                f"'tau_stc' and 'q_stc' must have the same length. "
                f"Got {len(self.tau_stc)} and {len(self.q_stc)}."
            )

        # Store tau arrays as unitful quantities for use in vector field
        self._tau_sfa_ms = jnp.array([t for t in self.tau_sfa]) * u.ms if self.tau_sfa else None
        self._tau_stc_ms = jnp.array([t for t in self.tau_stc]) * u.ms if self.tau_stc else None
        self._q_sfa_mV = jnp.array([q for q in self.q_sfa]) * u.mV if self.q_sfa else None
        self._q_stc_pA = jnp.array([q for q in self.q_stc]) * u.pA if self.q_stc else None

        self.gsl_error_tol = gsl_error_tol

        # RNG key for stochastic spiking
        self._rng_key = rng_key

        # Initializers
        self.V_initializer = V_initializer
        self.ref_var = ref_var

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

    def _sum_signed_delta_inputs(self):
        r"""Route delta inputs by sign: positive -> excitatory, negative -> inhibitory.

        This matches NEST's spike routing where each spike event is individually
        directed to the excitatory or inhibitory buffer based on weight sign.

        Returns
        -------
        w_ex : ArrayLike
            Total excitatory synaptic weight jumps (pA) for this time step. Array
            matching neuron population shape.
        w_in : ArrayLike
            Total inhibitory synaptic weight jumps (pA, as negative values) for this
            time step. Array matching neuron population shape.

        Notes
        -----
        Consumes callable delta inputs from ``self.delta_inputs`` dict after evaluation.
        Non-callable entries remain until next call.
        """
        w_ex = u.math.zeros_like(self.I_syn_ex.value)
        w_in = u.math.zeros_like(self.I_syn_in.value)
        if self.delta_inputs is None:
            return w_ex, w_in

        for key in tuple(self.delta_inputs.keys()):
            out = self.delta_inputs[key]
            if callable(out):
                out = out()
            else:
                self.delta_inputs.pop(key)

            zero = u.math.zeros_like(out)
            w_ex = w_ex + u.math.maximum(out, zero)
            w_in = w_in + u.math.minimum(out, zero)
        return w_ex, w_in

    def _validate_parameters(self):
        r"""Validate model parameters against NEST constraints.

        Raises
        ------
        ValueError
            If parameter inequalities or positivity constraints are violated.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.C_m, self.g_L, self.Delta_V)):
            return
        if np.any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self.g_L <= 0.0 * u.nS):
            raise ValueError('Membrane conductance must be strictly positive.')
        if np.any(self.Delta_V <= 0.0 * u.mV):
            raise ValueError('Delta_V must be strictly positive.')
        if np.any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time must not be negative.')
        if self.lambda_0 < 0.0:
            raise ValueError('lambda_0 must not be negative.')
        if np.any(self.tau_syn_ex <= 0.0 * u.ms) or \
            np.any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('Synapse time constants must be strictly positive.')
        if np.any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')
        for tau in self.tau_sfa:
            if tau <= 0.0:
                raise ValueError('All SFA time constants must be strictly positive.')
        for tau in self.tau_stc:
            if tau <= 0.0:
                raise ValueError('All STC time constants must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize persistent and short-term state variables.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Raises
        ------
        ValueError
            If an initializer cannot be broadcast to requested shape.
        TypeError
            If initializer outputs have incompatible units/dtypes for the
            corresponding state variables.
        """
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=V.dtype) * u.pA)
        self.I_syn_in = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=V.dtype) * u.pA)

        # STC elements: shape (n_stc, *varshape) in pA
        n_stc = len(self.tau_stc)
        if n_stc > 0:
            self.stc_elems = brainstate.HiddenState(
                u.math.zeros((n_stc, *self.varshape), dtype=dftype) * u.pA
            )
        else:
            self.stc_elems = brainstate.HiddenState(
                u.math.zeros((0, *self.varshape), dtype=dftype) * u.pA
            )

        # SFA elements: shape (n_sfa, *varshape) in mV
        n_sfa = len(self.tau_sfa)
        if n_sfa > 0:
            self.sfa_elems = brainstate.HiddenState(
                u.math.zeros((n_sfa, *self.varshape), dtype=dftype) * u.mV
            )
        else:
            self.sfa_elems = brainstate.HiddenState(
                u.math.zeros((0, *self.varshape), dtype=dftype) * u.mV
            )

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.full(self.varshape, 0.0 * u.pA, dtype=dftype))

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Generate surrogate spike output for gradient-based learning.

        Applies the surrogate gradient function to a scaled membrane potential to
        produce a differentiable spike signal. This is used for gradient-based
        learning, not for the actual stochastic spike generation (which happens in
        ``update()``).

        Parameters
        ----------
        V : ArrayLike, optional
            Membrane potential (mV). If None, uses ``self.V.value``. Default: None.

        Returns
        -------
        spike : array
            Differentiable spike output (float). Shape matches ``V``. Values are
            in [0, 1] or similar range depending on ``spk_fun``.

        Notes
        -----
        The scaling used is ``(V - V_reset) / Delta_V`` before applying ``spk_fun``.
        This function is typically used internally by the framework for gradient
        computation, not called directly by users.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_reset) / (self.Delta_V)
        return self.spk_fun(v_scaled)

    @staticmethod
    def _propagator_exp(tau_syn: np.ndarray, tau_m: np.ndarray, c_m: np.ndarray, h_ms: float):
        r"""Compute the propagator coefficient P21 (I_syn -> V_m) for exact integration.

        This matches NEST's ``IAFPropagatorExp::evaluate()`` with singularity handling.
        The propagator describes how a synaptic current at step :math:`n` contributes
        to the membrane potential at step :math:`n+1` under exact exponential integration.

        When :math:`\tau_\mathrm{syn} \approx \tau_\mathrm{m}`, the formula develops
        a singularity. The implementation detects this numerically and falls back to
        the limit form.

        Parameters
        ----------
        tau_syn : float or ndarray
            Synaptic time constant in ms. Must be strictly positive. Scalar or array.
        tau_m : float or ndarray
            Membrane time constant in ms. Must be strictly positive. Scalar or array.
        c_m : float or ndarray
            Membrane capacitance in pF. Must be strictly positive. Scalar or array.
        h_ms : float
            Time step in ms. Must be strictly positive.

        Returns
        -------
        P21 : float or ndarray
            Propagator coefficient (unitless, but effectively in mV/pA when applied).
            Shape matches broadcasted shape of inputs. Falls back to singularity-safe
            limit when :math:`\tau_\mathrm{syn} \approx \tau_\mathrm{m}`.

        Notes
        -----
        The regular formula is:

        .. math::

           P_{21} = \frac{\tau_\mathrm{syn} \tau_\mathrm{m}}{C_\mathrm{m}(\tau_\mathrm{m} - \tau_\mathrm{syn})}
               \exp(-h/\tau_\mathrm{syn})
               \left[\exp\left(h \frac{\tau_\mathrm{m} - \tau_\mathrm{syn}}{\tau_\mathrm{syn} \tau_\mathrm{m}}\right) - 1\right]

        The singular limit (:math:`\tau_\mathrm{syn} \to \tau_\mathrm{m}`) is:

        .. math::

           P_{21} = \frac{h}{C_\mathrm{m}} \exp(-h/\tau_\mathrm{m})

        Singularity detection checks for non-finite results, subnormal floats, or
        non-positive values.
        """
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            beta = tau_syn * tau_m / (tau_m - tau_syn)
            gamma = beta / c_m
            inv_beta = (tau_m - tau_syn) / (tau_syn * tau_m)
            exp_h_tau_syn = np.exp(-h_ms / tau_syn)
            expm1_h_tau = np.expm1(h_ms * inv_beta)
            p32_raw = gamma * exp_h_tau_syn * expm1_h_tau

            normal_min = np.finfo(np.float64).tiny
            regular_mask = np.isfinite(p32_raw) & (np.abs(p32_raw) >= normal_min) & (p32_raw > 0.0)
            p32_singular = h_ms / c_m * np.exp(-h_ms / tau_m)
            return np.where(regular_mask, p32_raw, p32_singular)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all neurons simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V, I_syn_ex, I_syn_in, stc_elems, sfa_elems — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim — mutable
            auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        v_eff = u.math.where(is_refractory, self.V_reset, state.V)

        # Compute total STC current
        stc_total = u.math.sum(state.stc_elems, axis=0) if len(self.tau_stc) > 0 else 0.0 * u.pA

        # Membrane dynamics: C_m * dV/dt = -g_L*(V - E_L) - stc_total + I_syn_ex + I_syn_in + I_e + I_stim
        dV_raw = (
            -self.g_L * (v_eff - self.E_L)
            - stc_total
            + state.I_syn_ex + state.I_syn_in
            + self.I_e + extra.i_stim
        ) / self.C_m
        dV = u.math.where(is_refractory, u.math.zeros_like(dV_raw), dV_raw)

        # Synaptic current dynamics: dI_syn/dt = -I_syn / tau_syn
        dI_syn_ex = -state.I_syn_ex / self.tau_syn_ex
        dI_syn_in = -state.I_syn_in / self.tau_syn_in

        # STC dynamics: tau_eta_i * d(eta_i)/dt = -eta_i  =>  d(eta_i)/dt = -eta_i / tau_eta_i
        if len(self.tau_stc) > 0:
            # _tau_stc_ms shape: (n_stc,), stc_elems shape: (n_stc, *varshape)
            # Reshape tau for broadcasting
            tau_shape = (-1,) + (1,) * len(self.varshape)
            d_stc = -state.stc_elems / u.math.reshape(self._tau_stc_ms, tau_shape)
        else:
            d_stc = state.stc_elems * 0.0 / u.ms

        # SFA dynamics: tau_gamma_i * d(gamma_i)/dt = -gamma_i  =>  d(gamma_i)/dt = -gamma_i / tau_gamma_i
        if len(self.tau_sfa) > 0:
            tau_shape = (-1,) + (1,) * len(self.varshape)
            d_sfa = -state.sfa_elems / u.math.reshape(self._tau_sfa_ms, tau_shape)
        else:
            d_sfa = state.sfa_elems * 0.0 / u.ms

        return DotDict(V=dV, I_syn_ex=dI_syn_ex, I_syn_in=dI_syn_in, stc_elems=d_stc, sfa_elems=d_sfa)

    def _event_fn(self, state, extra, accept):
        """In-loop stochastic spike detection, reset, and adaptation handling.

        Parameters
        ----------
        state : DotDict
            Keys: V, I_syn_ex, I_syn_in, stc_elems, sfa_elems — ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim, rand_vals.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset/refractory info.
        """
        unstable = extra.unstable | jnp.any(
            accept & ((state.V < -1e3 * u.mV) | (state.I_syn_ex > 1e6 * u.pA) | (state.I_syn_in < -1e6 * u.pA))
        )

        # Refractory clamping: if refractory and accepted, clamp V to V_reset
        refr_accept = accept & (extra.r > 0)
        new_V = u.math.where(refr_accept, self.V_reset, state.V)

        # Compute dynamic threshold V_T = V_T_star + sum(sfa_elems)
        sfa_total = u.math.sum(state.sfa_elems, axis=0) if len(self.tau_sfa) > 0 else 0.0 * u.mV
        V_T = self.V_T_star + sfa_total

        # Stochastic spike check for non-refractory accepted neurons
        # lambda = lambda_0 * exp((V - V_T) / Delta_V)
        exp_arg = u.math.clip((new_V - V_T) / self.Delta_V, -500.0, 500.0)
        lam = self.lambda_0 * u.math.exp(exp_arg)  # 1/ms

        # Spike probability over the current substep duration
        # Using dt from the integration step is not directly available here,
        # so we use a fixed dt from the environment for the probability check
        dt_q = brainstate.environ.get_dt()
        h_ms = u.get_mantissa(dt_q / u.ms)
        spike_prob = -jnp.expm1(-lam * h_ms)
        spike_prob = jnp.clip(spike_prob, 0.0, 1.0)

        spike_now = accept & (extra.r <= 0) & (extra.rand_vals < spike_prob)
        spike_mask = extra.spike_mask | spike_now

        # Reset V on spike
        new_V = u.math.where(spike_now, self.V_reset, new_V)

        # Jump STC elements on spike: eta_i += q_eta_i
        new_stc = state.stc_elems
        if len(self.tau_stc) > 0:
            # _q_stc_pA shape: (n_stc,), need to broadcast to (n_stc, *varshape)
            q_shape = (-1,) + (1,) * len(self.varshape)
            stc_jump = u.math.reshape(self._q_stc_pA, q_shape) * u.math.ones_like(state.stc_elems)
            new_stc = u.math.where(spike_now, new_stc + stc_jump, new_stc)

        # Jump SFA elements on spike: gamma_i += q_gamma_i
        new_sfa = state.sfa_elems
        if len(self.tau_sfa) > 0:
            q_shape = (-1,) + (1,) * len(self.varshape)
            sfa_jump = u.math.reshape(self._q_sfa_mV, q_shape) * u.math.ones_like(state.sfa_elems)
            new_sfa = u.math.where(spike_now, new_sfa + sfa_jump, new_sfa)

        # Set refractory counter on spike
        r = u.math.where(spike_now & (self.ref_count > 0), self.ref_count + 1, extra.r)

        new_state = DotDict({**state, 'V': new_V, 'stc_elems': new_stc, 'sfa_elems': new_sfa})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'r': r, 'unstable': unstable})
        return new_state, new_extra

    def update(self, x=0.0 * u.pA):
        r"""Advance the neuron by one simulation step.

        Performs a complete simulation step following NEST's ``gif_psc_exp`` update
        order: integrate ODE dynamics via adaptive RKF45 (with in-loop stochastic
        spike detection, refractory clamping, and adaptation jumps), then apply
        synaptic spike weights and buffer external input.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input (pA). Scalar or array matching population shape.
            This input is ``buffered`` and applied in the ``next`` time step, matching
            NEST's ring buffer semantics. Default: 0.0 pA.

        Returns
        -------
        spike : jax.Array
            Binary spike output (0 or 1) as float array. Shape matches neuron
            population. Spikes are generated stochastically based on firing
            intensity :math:`\lambda(t)`.

        Raises
        ------
        ValueError
            If RKF45 integration enters a guarded unstable regime, indicating
            divergent dynamics for the current parameter/input regime.

        Notes
        -----
        Integration is performed with an adaptive vectorized RKF45 loop,
        including in-loop stochastic spike/reset/adaptation events. All
        arithmetic is unit-aware via ``saiunit.math``.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV
        I_syn_ex = self.I_syn_ex.value  # pA
        I_syn_in = self.I_syn_in.value  # pA
        stc_elems = self.stc_elems.value  # pA, shape (n_stc, *varshape)
        sfa_elems = self.sfa_elems.value  # mV, shape (n_sfa, *varshape)
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA
        h = self.integration_step.value  # ms

        # Current input for next step (one-step delay).
        new_i_stim = self.sum_current_inputs(x, self.V.value)  # pA

        # Advance RNG state for this step
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = jax.random.uniform(subkey, shape=self.varshape)

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(V=V, I_syn_ex=I_syn_ex, I_syn_in=I_syn_in, stc_elems=stc_elems, sfa_elems=sfa_elems)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
            rand_vals=rand_vals,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        V = ode_state.V
        I_syn_ex, I_syn_in = ode_state.I_syn_ex, ode_state.I_syn_in
        stc_elems, sfa_elems = ode_state.stc_elems, ode_state.sfa_elems
        spike_mask, r, unstable = extra.spike_mask, extra.r, extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in gif_psc_exp dynamics.'
        )

        # Decrement refractory counter.
        r = u.math.where(r > 0, r - 1, r)

        # Synaptic spike inputs (applied after integration).
        w_ex, w_in = self._sum_signed_delta_inputs()
        I_syn_ex = I_syn_ex + w_ex
        I_syn_in = I_syn_in + w_in

        # Write back state.
        self.V.value = V
        self.I_syn_ex.value = I_syn_ex
        self.I_syn_in.value = I_syn_in
        self.stc_elems.value = stc_elems
        self.sfa_elems.value = sfa_elems
        self.refractory_step_count.value = jnp.asarray(u.get_mantissa(r), dtype=ditype)
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
