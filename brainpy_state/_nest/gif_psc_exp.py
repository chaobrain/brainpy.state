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
from typing import Callable, Optional, Sequence

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

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

    The model uses exact (analytic) propagators for the linear subthreshold dynamics,
    matching NEST's integration scheme. The discrete-time update order per simulation
    step is:

    1. **Adaptation decay**: Compute total stc (sum of :math:`\eta_i`) and sfa threshold
       (:math:`V_{T^*} + \sum \gamma_i`), then decay all elements by :math:`\exp(-dt/\tau_i)`.
    2. **Synaptic decay**: :math:`I_{\mathrm{syn}} \leftarrow I_{\mathrm{syn}} \cdot \exp(-dt/\tau_\mathrm{syn})`.
    3. **Synaptic input**: Add spike weight jumps from spikes arriving this step.
    4. **Membrane update / spike check**:

       - If not refractory: update :math:`V` via exact propagator:

         .. math::

            V \leftarrow P_{30}(I_\mathrm{stim} + I_e - \sum\eta_i) + P_{33} V + P_{31} E_L
                + I_{\mathrm{syn,ex}} P_{21,\mathrm{ex}} + I_{\mathrm{syn,in}} P_{21,\mathrm{in}}

         Compute :math:`\lambda(t)`, draw random number, potentially emit spike (increment
         stc/sfa elements by :math:`q_i`, set refractory counter).

       - If refractory: decrement counter, clamp :math:`V` to :math:`V_\mathrm{reset}`.

    5. **Buffer external input**: Store :math:`I_\mathrm{stim}` for the next step
       (NEST ring buffer semantics).

    **3. Propagator Coefficients**

    The exact integration uses the following propagators (see NEST ``gif_psc_exp.cpp``):

    - :math:`\tau_\mathrm{m} = C_\mathrm{m} / g_\mathrm{L}` (membrane time constant)
    - :math:`P_{33} = \exp(-dt/\tau_\mathrm{m})` (membrane potential decay)
    - :math:`P_{31} = 1 - \exp(-dt/\tau_\mathrm{m})` (:math:`E_L` contribution)
    - :math:`P_{30} = \tau_\mathrm{m}/C_\mathrm{m} \cdot (1 - \exp(-dt/\tau_\mathrm{m}))` (current contribution)
    - :math:`P_{21}` (synaptic current → membrane): computed via ``_propagator_exp()``
      with singularity handling when :math:`\tau_\mathrm{syn} \approx \tau_\mathrm{m}`.

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
    rng_key : jax.Array, optional
        JAX PRNG key for stochastic spiking. If None, uses a default key (seed 0).
        For reproducible results, provide an explicit key. Default: None.
    V_initializer : Callable, optional
        Initializer for membrane potential. Must accept shape and batch_size arguments.
        Default: ``braintools.init.Constant(-70.0 * u.mV)``.
    spk_fun : Callable, optional
        Surrogate gradient function for spike generation. Used for gradient-based
        learning. Default: ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Spike reset mode. ``'hard'`` (stop gradient) matches NEST behavior; ``'soft'``
        subtracts threshold. Default: ``'hard'``.
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
    ``rng_key``          None                —                                   JAX PRNG key for stochastic spiking
    ``V_initializer``    Constant(-70 mV)    —                                   Initializer for membrane potential
    ``spk_fun``          ReluGrad()          —                                   Surrogate spike function
    ``spk_reset``        ``'hard'``          —                                   Reset mode; hard reset matches NEST
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

    - Numerical stability: If ``tau_m`` is very close to ``tau_syn_ex`` or
      ``tau_syn_in``, the model numerically behaves as if ``tau_m`` equals the
      synaptic time constant to avoid singularities in the propagator computation.

    Notes
    -----
    - Defaults follow NEST C++ source for ``gif_psc_exp`` (``models/gif_psc_exp.cpp``).
    - ``lambda_0`` is specified in 1/s (as in NEST's Python interface) and is
      internally converted to 1/ms for computation.
    - Synaptic spike weights are interpreted in current units (pA), with positive/
      negative sign selecting excitatory/inhibitory channel.
    - The subthreshold dynamics use exact (analytic) integration via propagator
      coefficients, matching NEST's integration scheme. This differs from
      ``gif_cond_exp``, which uses RKF45 adaptive integration.
    - State variables (``V``, ``I_syn_ex``, ``I_syn_in``, etc.) are accessible as
      attributes after calling ``init_state()`` or ``reset_state()``.

    Examples
    --------
    Create a single GIF neuron with default parameters:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
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
        rng_key: Optional[jax.Array] = None,
        V_initializer: Callable = braintools.init.Constant(-70.0 * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
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

        # RNG key for stochastic spiking
        self._rng_key = rng_key

        # Initializers
        self.V_initializer = V_initializer

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

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
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.g_L, u.nS) <= 0.0):
            raise ValueError('Membrane conductance must be strictly positive.')
        if np.any(self._to_numpy(self.Delta_V, u.mV) <= 0.0):
            raise ValueError('Delta_V must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')
        if self.lambda_0 < 0.0:
            raise ValueError('lambda_0 must not be negative.')
        if np.any(self._to_numpy(self.tau_syn_ex, u.ms) <= 0.0) or \
            np.any(self._to_numpy(self.tau_syn_in, u.ms) <= 0.0):
            raise ValueError('Synapse time constants must be strictly positive.')
        for tau in self.tau_sfa:
            if tau <= 0.0:
                raise ValueError('All SFA time constants must be strictly positive.')
        for tau in self.tau_stc:
            if tau <= 0.0:
                raise ValueError('All STC time constants must be strictly positive.')

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables.

        Creates and initializes:
        - Membrane potential ``V`` (via ``V_initializer``)
        - Synaptic currents ``I_syn_ex``, ``I_syn_in`` (zero)
        - Refractory counter ``refractory_step_count`` (zero)
        - Buffered stimulus current ``I_stim`` (zero)
        - Last spike time ``last_spike_time`` (-1e7 ms, i.e., far past)
        - Adaptation elements ``_stc_elems``, ``_sfa_elems`` (zero arrays)
        - RNG state ``_rng_state`` (from ``rng_key`` or default)

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If None, no batch dimension is added. If provided,
            state arrays have shape ``(batch_size, *in_size)``. Default: None.
        **kwargs
            Additional keyword arguments (ignored).

        Notes
        -----
        Must be called before ``update()`` or ``reset_state()``. State variables are
        stored as ``brainstate.HiddenState`` or ``brainstate.ShortTermState`` and are
        accessible as instance attributes.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        self.I_stim = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size)
        )

        # Adaptation state: stc and sfa element arrays (unitless floats in nA and mV respectively)
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)  # total stc current (nA)
        self._sfa_val = np.full(v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=np.float64)

        # RNG state
        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to initial values.

        Resets all state variables to the same initial values as ``init_state()``,
        but preserves the state objects (only updates their ``.value`` attributes).

        Parameters
        ----------
        batch_size : int, optional
            Batch dimension size. If None, no batch dimension is added. If provided,
            state arrays have shape ``(batch_size, *in_size)``. Default: None.
        **kwargs
            Additional keyword arguments (ignored).

        Notes
        -----
        Use this to reset the neuron to initial conditions without recreating state
        objects. Useful for running multiple trials or resetting after a simulation.
        """
        self.V.value = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(self.V.value / u.mV))
        self.I_syn_ex.value = zeros * u.pA
        self.I_syn_in.value = zeros * u.pA
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)
        self.I_stim.value = braintools.init.param(
            braintools.init.Constant(0.0 * u.pA), self.varshape, batch_size
        )

        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)
        v_shape = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self._stc_elems = np.zeros((n_stc, *v_shape), dtype=np.float64) if n_stc > 0 else None
        self._sfa_elems = np.zeros((n_sfa, *v_shape), dtype=np.float64) if n_sfa > 0 else None
        self._stc_val = np.zeros(v_shape, dtype=np.float64)
        self._sfa_val = np.full(v_shape, float(self._to_numpy(self.V_T_star, u.mV)), dtype=np.float64)

        if self._rng_key is not None:
            self._rng_state = self._rng_key
        else:
            self._rng_state = jax.random.PRNGKey(0)

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

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

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

    def update(self, x=0.0 * u.pA):
        r"""Update neuron state for one simulation step.

        Performs a complete simulation step following NEST's ``gif_psc_exp`` update
        order: decay adaptation elements, decay and update synaptic currents, update
        membrane potential via exact propagator (if not refractory), perform stochastic
        spike check, and buffer external input for the next step.

        The update order matches NEST's implementation:

        1. Decay stc/sfa elements, compute totals
        2. Decay synaptic currents exponentially
        3. Add spike weight jumps from incoming spikes
        4. If not refractory: update membrane potential via exact propagator, perform
           stochastic spike check (draw random number, compare to firing probability),
           potentially emit spike (increment adaptation elements, set refractory counter).
           If refractory: decrement counter, clamp V to V_reset.
        5. Buffer external current for next step (NEST ring buffer semantics)

        Parameters
        ----------
        x : ArrayLike, optional
            External current input (pA). Scalar or array matching population shape.
            This input is ``buffered`` and applied in the ``next`` time step, matching
            NEST's ring buffer semantics. Default: 0.0 pA.

        Returns
        -------
        spike : jax.Array
            Binary spike output (0 or 1) as float32 array. Shape matches neuron
            population (batch_size, \*in_size) if batched, or (\*in_size) otherwise.
            Spikes are generated stochastically based on firing intensity :math:`\lambda(t)`.

        Notes
        -----
        - The returned spike array is binary (0 or 1) from the stochastic spike check,
          not the differentiable surrogate output from ``get_spike()``.
        - Random number generation advances the internal RNG state each call. For
          reproducible results, set ``rng_key`` explicitly at construction.
        - Synaptic spike inputs are summed from ``self.delta_inputs`` (registered by
          projections) and routed by sign: positive weights → excitatory channel,
          negative weights → inhibitory channel.
        - Current inputs (including ``x``) are summed from ``self.current_inputs`` and
          buffered as ``I_stim`` for application in the *next* step.
        - State variables (``V``, ``I_syn_ex``, ``I_syn_in``, etc.) are updated in-place.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))  # dt in ms as float

        v_shape = self.V.value.shape

        # Extract state variables as numpy arrays
        V = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape).copy()
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape).copy()
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape).copy()
        r = self._broadcast_to_state(
            np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32), v_shape
        ).copy()
        i_stim = self._broadcast_to_state(self._to_numpy(self.I_stim.value, u.pA), v_shape).copy()

        # Extract parameters as numpy arrays
        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        g_L = self._broadcast_to_state(self._to_numpy(self.g_L, u.nS), v_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        V_reset = self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), v_shape)
        tau_syn_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_syn_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        V_T_star = float(self._to_numpy(self.V_T_star, u.mV))
        Delta_V = float(self._to_numpy(self.Delta_V, u.mV))
        lambda_0 = self.lambda_0  # 1/ms

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32), v_shape
        )

        # Compute propagator coefficients (exact integration)
        tau_m = C_m / g_L  # membrane time constant in ms
        P33 = np.exp(-h / tau_m)
        P30 = -1.0 / C_m * np.expm1(-h / tau_m) * tau_m  # = tau_m/C_m * (1 - exp(-h/tau_m))
        P31 = -np.expm1(-h / tau_m)  # = 1 - exp(-h/tau_m)
        P11_ex = np.exp(-h / tau_syn_ex)
        P11_in = np.exp(-h / tau_syn_in)
        P21_ex = self._propagator_exp(tau_syn_ex, tau_m, C_m, h)
        P21_in = self._propagator_exp(tau_syn_in, tau_m, C_m, h)

        # Compute exponential decay factors for adaptation
        P_stc = [math.exp(-h / tau) for tau in self.tau_stc]
        P_sfa = [math.exp(-h / tau) for tau in self.tau_sfa]

        # Get synaptic inputs (spike weights: positive -> excitatory, negative -> inhibitory)
        w_ex_q, w_in_q = self._sum_signed_delta_inputs()
        w_ex = self._broadcast_to_state(self._to_numpy(w_ex_q, u.pA), v_shape)
        w_in = self._broadcast_to_state(self._to_numpy(w_in_q, u.pA), v_shape)

        # Get external current for NEXT step (NEST ring buffer semantics)
        new_i_stim = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        # Advance RNG state for this step
        self._rng_state, subkey = jax.random.split(self._rng_state)
        rand_vals = np.asarray(jax.random.uniform(subkey, shape=v_shape), dtype=np.float64)

        spike_mask = np.zeros_like(V, dtype=bool)

        for idx in np.ndindex(v_shape):
            # ---- Step 1: Decay stc/sfa elements and compute totals ----
            stc_total = 0.0
            if self._stc_elems is not None:
                for i in range(len(self.tau_stc)):
                    stc_total += self._stc_elems[i][idx]
                    self._stc_elems[i][idx] *= P_stc[i]

            sfa_total = V_T_star
            if self._sfa_elems is not None:
                for i in range(len(self.tau_sfa)):
                    sfa_total += self._sfa_elems[i][idx]
                    self._sfa_elems[i][idx] *= P_sfa[i]

            self._stc_val[idx] = stc_total
            self._sfa_val[idx] = sfa_total

            # ---- Step 2: Decay synaptic currents ----
            i_syn_ex[idx] *= P11_ex[idx]
            i_syn_in[idx] *= P11_in[idx]

            # ---- Step 3: Add synaptic weight jumps ----
            i_syn_ex[idx] += w_ex[idx]
            i_syn_in[idx] += w_in[idx]

            # ---- Step 4: Refractory / membrane update / spike check ----
            if r[idx] == 0:
                # Not refractory: update membrane potential via exact propagator
                V[idx] = (P30[idx] * (i_stim[idx] + I_e[idx] - stc_total)
                          + P33[idx] * V[idx]
                          + P31[idx] * E_L[idx]
                          + i_syn_ex[idx] * P21_ex[idx]
                          + i_syn_in[idx] * P21_in[idx])

                # Stochastic spike check
                lam = lambda_0 * math.exp((V[idx] - sfa_total) / Delta_V)
                if lam > 0.0:
                    spike_prob = -math.expm1(-lam * h)
                    if rand_vals[idx] < spike_prob:
                        # Spike!
                        spike_mask[idx] = True

                        # Jump stc elements
                        if self._stc_elems is not None:
                            for i in range(len(self.q_stc)):
                                self._stc_elems[i][idx] += self.q_stc[i]

                        # Jump sfa elements
                        if self._sfa_elems is not None:
                            for i in range(len(self.q_sfa)):
                                self._sfa_elems[i][idx] += self.q_sfa[i]

                        r[idx] = refr_counts[idx]
            else:
                # Refractory: decrement counter, clamp V to V_reset
                r[idx] -= 1
                V[idx] = V_reset[idx]

        # ---- Step 5: Store new I_stim for next step, update state ----
        self.V.value = V * u.mV
        self.I_syn_ex.value = i_syn_ex * u.pA
        self.I_syn_in.value = i_syn_in * u.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        return jnp.asarray(spike_mask, dtype=jnp.float32)
