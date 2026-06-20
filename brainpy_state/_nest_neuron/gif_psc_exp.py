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

from collections.abc import Callable, Sequence

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest_base.base import NESTNeuron
from brainpy_state._nest_base.utils import is_tracer, cond_any

__all__ = [
    'gif_psc_exp',
]


class _AdaptElemsRow:
    """Mutable row-view into an adaptation element ShortTermState array.

    Returned by ``_AdaptElems.__getitem__``. Supports both read
    (``row[j]``) and write (``row[j] = val``) which updates the
    underlying :class:`brainstate.ShortTermState` in-place via JAX's
    functional ``.at[i, j].set(val)`` API.
    """

    def __init__(self, state, row_idx):
        self._state = state
        self._idx = row_idx

    def __getitem__(self, idx):
        return self._state.value[self._idx][idx]

    def __setitem__(self, idx, value):
        self._state.value = self._state.value.at[self._idx, idx].set(value)


class _AdaptElems:
    """Mutable wrapper around a ShortTermState holding adaptation elements.

    Shape of the underlying array is ``(n_elems, *varshape)``.
    Supports indexing (``elems[i]``) which returns an :class:`_AdaptElemsRow`
    supporting further item-read/write.
    """

    def __init__(self, state):
        self._state = state

    def __getitem__(self, idx):
        return _AdaptElemsRow(self._state, idx)

    def __setitem__(self, idx, value):
        self._state.value = self._state.value.at[idx].set(value)

    def __len__(self):
        return self._state.value.shape[0]


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

    The model uses exact matrix-exponential integration, matching NEST's update
    order precisely. The discrete-time update per simulation step is:

    1. **STC/SFA totals**: Sum adaptation elements (before decay), then decay.
    2. **Synaptic decay**: :math:`I_\mathrm{syn} \leftarrow I_\mathrm{syn} \cdot e^{-dt/\tau}`.
    3. **Spike weights**: Add arriving spike weights to :math:`I_\mathrm{syn,ex}` / :math:`I_\mathrm{syn,in}`.
    4. **V update**: If not refractory, apply exact propagator using post-weight synaptic currents.
       If refractory, clamp :math:`V` to :math:`V_\mathrm{reset}` and decrement counter.
    5. **Store I_stim**: Buffer external input for next step (NEST ring buffer semantics).

    Parameters
    ----------
    in_size : int, tuple of int
        Shape of the neuron population.
    g_L : ArrayLike, optional
        Leak conductance. Default: 4.0 nS.
    E_L : ArrayLike, optional
        Leak reversal potential. Default: -70.0 mV.
    C_m : ArrayLike, optional
        Membrane capacitance. Default: 80.0 pF.
    V_reset : ArrayLike, optional
        Reset potential after spike. Default: -55.0 mV.
    Delta_V : ArrayLike, optional
        Stochasticity level. Default: 0.5 mV.
    V_T_star : ArrayLike, optional
        Base firing threshold. Default: -35.0 mV.
    lambda_0 : float, optional
        Stochastic intensity at threshold in 1/s. Default: 1.0 /s.
    t_ref : ArrayLike, optional
        Absolute refractory period. Default: 4.0 ms.
    tau_syn_ex : ArrayLike, optional
        Excitatory synaptic time constant. Default: 2.0 ms.
    tau_syn_in : ArrayLike, optional
        Inhibitory synaptic time constant. Default: 2.0 ms.
    I_e : ArrayLike, optional
        Constant external current. Default: 0.0 pA.
    tau_sfa : Sequence[float], optional
        SFA time constants in ms. Default: () (no SFA).
    q_sfa : Sequence[float], optional
        SFA jump values in mV. Default: () (no SFA).
    tau_stc : Sequence[float], optional
        STC time constants in ms. Default: () (no STC).
    q_stc : Sequence[float], optional
        STC jump values in pA. Default: () (no STC).
    rng_key : jax.Array, optional
        JAX PRNG key for stochastic spiking. Default: None (seed 0).
    V_initializer : Callable, optional
        Initializer for membrane potential. Default: Constant(-70 mV).
    spk_fun : Callable, optional
        Surrogate gradient function. Default: ReluGrad().
    spk_reset : str, optional
        Spike reset mode. Default: 'hard'.
    ref_var : bool, optional
        If True, expose boolean refractory state. Default: False.
    name : str, optional
        Name of the neuron group. Default: None.

    References
    ----------
    .. [1] Mensi S et al. (2012). Parameter extraction and classification of three
           cortical neuron types. Journal of Neurophysiology, 107(6):1756-1775.
    .. [2] Pozzorini C et al. (2015). Automated high-throughput characterization of
           single neurons. PLoS Computational Biology, 11(6), e1004275.
    .. [3] NEST Simulator ``gif_psc_exp`` model: ``models/gif_psc_exp.h``.

    See Also
    --------
    gif_cond_exp : Conductance-based GIF model.
    iaf_psc_exp : Simple IAF neuron with exponential synapses.
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
        q_stc: Sequence[float] = (),  # pA values
        rng_key: jax.Array | None = None,
        V_initializer: Callable = braintools.init.Constant(-70.0 * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str | None = None,
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

        # Adaptation parameters (stored as plain Python tuples of floats in ms/mV/pA)
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
        self.ref_var = ref_var

        self._validate_parameters()

        # Refractory counter (integer steps)
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

    def _sum_signed_delta_inputs(self):
        r"""Route delta inputs by sign: positive -> excitatory, negative -> inhibitory."""
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
        r"""Validate model parameters against NEST constraints."""
        if any(is_tracer(v) for v in (self.C_m, self.g_L, self.Delta_V)):
            return
        if cond_any(self.C_m <= 0.0 * u.pF):
            raise ValueError('Capacitance must be strictly positive.')
        if cond_any(self.g_L <= 0.0 * u.nS):
            raise ValueError('Membrane conductance must be strictly positive.')
        if cond_any(self.Delta_V <= 0.0 * u.mV):
            raise ValueError('Delta_V must be strictly positive.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time must not be negative.')
        if self.lambda_0 < 0.0:
            raise ValueError('lambda_0 must not be negative.')
        if cond_any(self.tau_syn_ex <= 0.0 * u.ms) or \
                cond_any(self.tau_syn_in <= 0.0 * u.ms):
            raise ValueError('Synapse time constants must be strictly positive.')
        for tau in self.tau_sfa:
            if tau <= 0.0:
                raise ValueError('All SFA time constants must be strictly positive.')
        for tau in self.tau_stc:
            if tau <= 0.0:
                raise ValueError('All STC time constants must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize persistent and short-term state variables."""
        ditype = brainstate.environ.ditype()

        v_shape = self.varshape
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)

        V = braintools.init.param(self.V_initializer, v_shape)
        # Force float64 precision for V and synaptic currents
        V_f64 = jnp.asarray(u.get_mantissa(V / u.mV), dtype=jnp.float64) * u.mV

        self.V = brainstate.HiddenState(V_f64)
        self.I_syn_ex = brainstate.ShortTermState(jnp.zeros(v_shape, dtype=jnp.float64) * u.pA)
        self.I_syn_in = brainstate.ShortTermState(jnp.zeros(v_shape, dtype=jnp.float64) * u.pA)

        # Adaptation state: stc/sfa elements stored as float64 for exact decay.
        # Shape (n_stc, *v_shape) for stc, (n_sfa, *v_shape) for sfa.
        self._stc_elems_state = (
            brainstate.ShortTermState(jnp.zeros((n_stc, *v_shape), dtype=jnp.float64))
            if n_stc > 0 else None
        )
        self._sfa_elems_state = (
            brainstate.ShortTermState(jnp.zeros((n_sfa, *v_shape), dtype=jnp.float64))
            if n_sfa > 0 else None
        )

        # Extract V_T_star as float64 numpy for initializing sfa_val
        V_T_star_np = np.asarray(u.get_mantissa(self.V_T_star / u.mV), dtype=np.float64)

        # Total STC current and effective threshold (updated at start of each step)
        self._stc_val_state = brainstate.ShortTermState(jnp.zeros(v_shape, dtype=jnp.float64))
        self._sfa_val_state = brainstate.ShortTermState(
            jnp.zeros(v_shape, dtype=jnp.float64) + jnp.asarray(V_T_star_np, dtype=jnp.float64)
        )

        self.last_spike_time = brainstate.ShortTermState(u.math.full(v_shape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(v_shape, 0, dtype=ditype))
        self.I_stim = brainstate.ShortTermState(jnp.zeros(v_shape, dtype=jnp.float64) * u.pA)

        # RNG state as ShortTermState for JIT compatibility.
        rng_init = self._rng_key if self._rng_key is not None else jax.random.PRNGKey(0)
        self._rng_state = brainstate.ShortTermState(rng_init)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), v_shape)
            self.refractory = brainstate.ShortTermState(refractory)

        # Pre-compute exact propagator coefficients (float64, numpy)
        self._precompute_propagators()

    def _precompute_propagators(self):
        """Pre-compute exact matrix-exponential propagator coefficients.

        Matches NEST's IAFPropagatorExp approach. All values are float64 numpy
        arrays computed once at init and reused every update step.
        """
        dt = brainstate.environ.get_dt()
        dt_ms = float(np.asarray(u.get_mantissa(dt / u.ms)))

        tau_syn_ex_ms = np.asarray(u.get_mantissa(self.tau_syn_ex / u.ms), dtype=np.float64)
        tau_syn_in_ms = np.asarray(u.get_mantissa(self.tau_syn_in / u.ms), dtype=np.float64)
        C_m_pF = np.asarray(u.get_mantissa(self.C_m / u.pF), dtype=np.float64)
        g_L_nS = np.asarray(u.get_mantissa(self.g_L / u.nS), dtype=np.float64)
        tau_m_ms = C_m_pF / g_L_nS

        self._dt_ms = dt_ms
        # Membrane propagators
        self._P33 = np.exp(-dt_ms / tau_m_ms)
        self._P30 = -1.0 / C_m_pF * np.expm1(-dt_ms / tau_m_ms) * tau_m_ms
        self._P31 = -np.expm1(-dt_ms / tau_m_ms)
        # Synaptic decay coefficients
        self._P11_ex = np.exp(-dt_ms / tau_syn_ex_ms)
        self._P11_in = np.exp(-dt_ms / tau_syn_in_ms)
        # Synaptic-to-membrane coupling propagators
        self._P21_ex = self._propagator_exp(tau_syn_ex_ms, tau_m_ms, C_m_pF, dt_ms)
        self._P21_in = self._propagator_exp(tau_syn_in_ms, tau_m_ms, C_m_pF, dt_ms)

        # Pre-extracted parameter values as float64 numpy
        self._E_L_mV_np = np.asarray(u.get_mantissa(self.E_L / u.mV), dtype=np.float64)
        self._V_reset_mV_np = np.asarray(u.get_mantissa(self.V_reset / u.mV), dtype=np.float64)
        self._V_T_star_mV_np = np.asarray(u.get_mantissa(self.V_T_star / u.mV), dtype=np.float64)
        self._Delta_V_mV_np = np.asarray(u.get_mantissa(self.Delta_V / u.mV), dtype=np.float64)
        self._I_e_pA_np = np.asarray(u.get_mantissa(self.I_e / u.pA), dtype=np.float64)

    @property
    def _stc_elems(self):
        """Spike-triggered current elements (n_stc, *varshape), plain float (pA)."""
        return _AdaptElems(self._stc_elems_state)

    @property
    def _sfa_elems(self):
        """Spike-frequency adaptation elements (n_sfa, *varshape), plain float (mV)."""
        return _AdaptElems(self._sfa_elems_state)

    @property
    def _stc_val(self):
        """Total STC current at start of last update step (*varshape), plain float (pA)."""
        return self._stc_val_state.value

    @property
    def _sfa_val(self):
        """Effective firing threshold (V_T_star + sum of sfa) at start of last step (mV)."""
        return self._sfa_val_state.value

    def get_spike(self, V: ArrayLike = None):
        r"""Generate surrogate spike output for gradient-based learning."""
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_reset) / (self.Delta_V)
        return self.spk_fun(v_scaled)

    @staticmethod
    def _propagator_exp(tau_syn: np.ndarray, tau_m: np.ndarray, c_m: np.ndarray, h_ms: float):
        r"""Compute the propagator coefficient P21 (I_syn -> V_m) for exact integration.

        Matches NEST's ``IAFPropagatorExp::evaluate()`` with singularity handling.

        Parameters
        ----------
        tau_syn : float or ndarray
            Synaptic time constant in ms.
        tau_m : float or ndarray
            Membrane time constant in ms.
        c_m : float or ndarray
            Membrane capacitance in pF.
        h_ms : float
            Time step in ms.

        Returns
        -------
        P21 : float or ndarray
            Propagator coefficient (mV/pA when applied to pA current).
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
        r"""Advance the neuron by one simulation step.

        Follows NEST's ``gif_psc_exp`` update order exactly:

        1. STC/SFA totals (before decay) + decay elements.
        2. Decay synaptic currents.
        3. Add arriving spike weights.
        4. Update V via exact propagator (non-refractory) or clamp to V_reset (refractory).
        5. Stochastic spike check; on spike: STC/SFA jumps, set refractory counter.
        6. Buffer external current for next step.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input (pA). Buffered for the NEXT time step (NEST ring
            buffer semantics). Default: 0.0 pA.

        Returns
        -------
        spike : jax.Array
            Binary spike output as float array matching population shape.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        dt_ms = self._dt_ms

        v_shape = self.varshape
        n_dims = len(v_shape)
        n_stc = len(self.tau_stc)
        n_sfa = len(self.tau_sfa)

        # Read state as plain float64
        V_mV = jnp.asarray(u.get_mantissa(self.V.value / u.mV), dtype=jnp.float64)
        I_syn_ex_pA = jnp.asarray(u.get_mantissa(self.I_syn_ex.value / u.pA), dtype=jnp.float64)
        I_syn_in_pA = jnp.asarray(u.get_mantissa(self.I_syn_in.value / u.pA), dtype=jnp.float64)
        r = self.refractory_step_count.value
        i_stim_pA = jnp.asarray(u.get_mantissa(self.I_stim.value / u.pA), dtype=jnp.float64)

        # Buffer current input for next step (NEST ring-buffer semantics).
        new_i_stim = self.sum_current_inputs(x, self.V.value)
        new_i_stim_pA = jnp.asarray(u.get_mantissa(new_i_stim / u.pA), dtype=jnp.float64)

        # ---- Step 1: stc/sfa totals (before decay) + decay elements ----
        if n_stc > 0:
            stc_elems = self._stc_elems_state.value          # (n_stc, *v_shape) float64
            stc_total_pA = jnp.sum(stc_elems, axis=0)        # (*v_shape) float64
            P_stc = jnp.array(
                [np.exp(-dt_ms / tau) for tau in self.tau_stc], dtype=jnp.float64
            ).reshape(n_stc, *([1] * n_dims))
            stc_elems_decayed = stc_elems * P_stc
        else:
            stc_total_pA = jnp.zeros(v_shape, dtype=jnp.float64)
            stc_elems_decayed = None

        if n_sfa > 0:
            sfa_elems = self._sfa_elems_state.value           # (n_sfa, *v_shape) float64
            V_T_star_f64 = jnp.asarray(self._V_T_star_mV_np, dtype=jnp.float64)
            sfa_total_mV = V_T_star_f64 + jnp.sum(sfa_elems, axis=0)
            P_sfa = jnp.array(
                [np.exp(-dt_ms / tau) for tau in self.tau_sfa], dtype=jnp.float64
            ).reshape(n_sfa, *([1] * n_dims))
            sfa_elems_decayed = sfa_elems * P_sfa
        else:
            sfa_total_mV = (
                jnp.asarray(self._V_T_star_mV_np, dtype=jnp.float64)
                + jnp.zeros(v_shape, dtype=jnp.float64)
            )
            sfa_elems_decayed = None

        # Store totals for property access
        self._stc_val_state.value = stc_total_pA
        self._sfa_val_state.value = sfa_total_mV

        # ---- Step 2: Decay synaptic currents ----
        I_syn_ex_pA = I_syn_ex_pA * jnp.asarray(self._P11_ex, dtype=jnp.float64)
        I_syn_in_pA = I_syn_in_pA * jnp.asarray(self._P11_in, dtype=jnp.float64)

        # ---- Step 3: Add arriving spike weights ----
        w_ex, w_in = self._sum_signed_delta_inputs()
        w_ex_pA = jnp.asarray(u.get_mantissa(w_ex / u.pA), dtype=jnp.float64)
        w_in_pA = jnp.asarray(u.get_mantissa(w_in / u.pA), dtype=jnp.float64)
        I_syn_ex_pA = I_syn_ex_pA + w_ex_pA
        I_syn_in_pA = I_syn_in_pA + w_in_pA

        # ---- Step 4: RNG for stochastic spike check ----
        new_rng, subkey = jax.random.split(self._rng_state.value)
        self._rng_state.value = new_rng
        rand_vals = jax.random.uniform(subkey, shape=v_shape)

        # ---- Step 5: V update via exact propagator + stochastic spike ----
        is_refractory = r > 0

        # Pre-load float64 propagator constants
        P33 = jnp.asarray(self._P33, dtype=jnp.float64)
        P30 = jnp.asarray(self._P30, dtype=jnp.float64)
        P31 = jnp.asarray(self._P31, dtype=jnp.float64)
        P21_ex = jnp.asarray(self._P21_ex, dtype=jnp.float64)
        P21_in = jnp.asarray(self._P21_in, dtype=jnp.float64)
        E_L_f64 = jnp.asarray(self._E_L_mV_np, dtype=jnp.float64)
        V_reset_f64 = jnp.asarray(self._V_reset_mV_np, dtype=jnp.float64)
        I_e_f64 = jnp.asarray(self._I_e_pA_np, dtype=jnp.float64)
        Delta_V_f64 = jnp.asarray(self._Delta_V_mV_np, dtype=jnp.float64)

        # Exact propagator: V_new = P33*V + P30*(I_stim+I_e-stc) + P31*E_L + I_syn*P21
        V_propagated = (
            P33 * V_mV
            + P30 * (i_stim_pA + I_e_f64 - stc_total_pA)
            + P31 * E_L_f64
            + I_syn_ex_pA * P21_ex
            + I_syn_in_pA * P21_in
        )

        # Stochastic spike check for non-refractory neurons
        exp_arg = jnp.clip((V_propagated - sfa_total_mV) / Delta_V_f64, -500.0, 500.0)
        lam = jnp.float64(self.lambda_0) * jnp.exp(exp_arg)  # 1/ms
        spike_prob = jnp.clip(-jnp.expm1(-lam * jnp.float64(dt_ms)), 0.0, 1.0)
        rand_f64 = rand_vals.astype(jnp.float64)
        spike_mask = (~is_refractory) & (rand_f64 < spike_prob)

        # V after step: propagated if non-refractory, V_reset if refractory
        # Note: on spike step V = V_propagated (not V_reset), matching NEST reference.
        V_mV = jnp.where(is_refractory, V_reset_f64, V_propagated)

        # Update refractory counter
        ref_count = jnp.asarray(self.ref_count, dtype=ditype)
        new_r = jnp.where(
            is_refractory,
            r - 1,
            jnp.where(spike_mask & (ref_count > 0), ref_count, r)
        )

        # ---- Step 6: stc/sfa jumps on spike ----
        spike_mask_f64 = spike_mask.astype(jnp.float64)
        if n_stc > 0:
            q_stc_arr = jnp.array(self.q_stc, dtype=jnp.float64).reshape(
                n_stc, *([1] * n_dims)
            )
            self._stc_elems_state.value = stc_elems_decayed + q_stc_arr * spike_mask_f64
        if n_sfa > 0:
            q_sfa_arr = jnp.array(self.q_sfa, dtype=jnp.float64).reshape(
                n_sfa, *([1] * n_dims)
            )
            self._sfa_elems_state.value = sfa_elems_decayed + q_sfa_arr * spike_mask_f64

        # ---- Step 7: Write back state ----
        self.V.value = V_mV * u.mV
        self.I_syn_ex.value = I_syn_ex_pA * u.pA
        self.I_syn_in.value = I_syn_in_pA * u.pA
        self.refractory_step_count.value = jnp.asarray(new_r, dtype=ditype)
        # Keep the population shape even when no current input is registered (then
        # ``sum_current_inputs`` returns the scalar ``x``): a scalar I_stim would
        # change the for_loop/scan carry type between iterations. (See the mc model
        # for the same broadcast-on-write idiom.)
        self.I_stim.value = (new_i_stim_pA + jnp.zeros(v_shape, dtype=jnp.float64)) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(new_r > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
