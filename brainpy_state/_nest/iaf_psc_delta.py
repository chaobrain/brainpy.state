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
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron

__all__ = [
    'iaf_psc_delta',
]


class iaf_psc_delta(NESTNeuron):
    r"""NEST-compatible ``iaf_psc_delta`` neuron model.

    Description
    -----------

    ``iaf_psc_delta`` is a current-based leaky integrate-and-fire neuron with
    hard threshold/reset, absolute refractory period, and delta-shaped
    synaptic events represented as instantaneous membrane-voltage jumps
    (weights in mV). The implementation follows the NEST model
    ``iaf_psc_delta`` update semantics, including refractory handling and
    step-wise exact subthreshold propagation.

    **1. Continuous-Time Dynamics and Exact Per-Step Propagator**

    The membrane dynamics are

    .. math::

       \frac{dV_\text{m}}{dt} = -\frac{V_{\text{m}} - E_\text{L}}{\tau_{\text{m}}}
       + \dot{\Delta}_{\text{syn}}
       + \frac{I_{\text{syn}} + I_\text{e}}{C_{\text{m}}}

    where :math:`I_\text{syn}` is the sum of continuous current inputs and
    :math:`\dot{\Delta}_{\text{syn}}` captures impulse-like jumps from
    delta synapses.

    For fixed simulation step :math:`h=dt` and piecewise-constant current
    :math:`I_k`, exact integration of the linear subthreshold ODE yields

    .. math::

       V_{k+1}^{\mathrm{cand}}
       = E_L + (V_k - E_L)e^{-h/\tau_m}
       + \frac{\tau_m}{C_m}\left(I_k + I_e\right)\left(1 - e^{-h/\tau_m}\right),

    which is implemented directly in :meth:`update`. This is equivalent to
    the propagator formulation used in NEST for this linear system.

    **2. Spike Condition, Reset, and Refractory Countdown**

    After adding delta-input jump :math:`\Delta_{\text{syn},k}`, a spike is
    emitted at step end if the post-update potential crosses threshold:

    .. math::

       V_k^{\mathrm{post}} \ge V_{th}.

    On spike:

    .. math::

       V \leftarrow V_{reset}, \qquad
       r \leftarrow \left\lceil \frac{t_{ref}}{dt} \right\rceil,

    where :math:`r` is the integer refractory-step counter. While
    :math:`r > 0`, the membrane is clamped (no subthreshold integration is
    committed), then :math:`r` decrements by one each simulation step.

    **3. Delta Synapses, Voltage Jumps, and Charge Interpretation**

    The change in membrane potential due to synaptic inputs can be formulated as:

    .. math::

       \dot{\Delta}_{\text{syn}}(t) = \sum_{j} w_j \sum_k \delta(t-t_j^k-d_j) \;,

    where :math:`j` indexes either excitatory (:math:`w_j > 0`) or inhibitory
    (:math:`w_j < 0`) presynaptic neurons, :math:`k` indexes the spike times of
    neuron :math:`j`, :math:`d_j` is the delay from neuron :math:`j`, and
    :math:`\delta` is the Dirac delta distribution. This implies that the jump in
    voltage upon a single synaptic input spike is

    .. math::

       \Delta_{\text{syn}} = w \;,

    where :math:`w` is synaptic weight in mV. Positive weights are excitatory
    and negative weights are inhibitory.

    The change in voltage caused by the synaptic input can be interpreted as being
    caused by individual post-synaptic currents (PSCs) given by

    .. math::

       i_{\text{syn}}(t) = C_{\text{m}} \cdot w \cdot \delta(t) \;.

    As a consequence, the total charge :math:`q` transferred by a single PSC is

    .. math::

       q = \int_0^{\infty}  i_{\text{syn}}(t)\, dt = C_{\text{m}} \cdot w \;.

    **4. Assumptions, Constraints, and Computational Implications**

    - The model assumes unit-compatible parameters and broadcast-compatible
      shapes against ``self.varshape``.
    - ``V_min`` is optional; when provided, candidate voltage is clipped with
      ``max(V, V_min)`` before threshold evaluation.
    - Per-step compute is :math:`O(\prod \mathrm{varshape})` with vectorized
      elementwise operations.
    - ``refractory_input=False`` discards delta events that arrive during
      refractory clamping, while ``refractory_input=True`` stores a decayed
      contribution that is released when refractoriness ends.

    .. note::

       This implementation uses exact integration for subthreshold dynamics
       and NEST-compatible conversion of refractory duration to grid steps via
       ``ceil(t_ref / dt)``.

    Parameters
    ----------

    in_size : Size
        Population shape specification. All neuron parameters are broadcast to
        ``self.varshape`` derived from ``in_size``.
    E_L : ArrayLike, optional
        Resting potential :math:`E_L` in mV; scalar or array broadcastable to
        ``self.varshape``. Default is ``-70. * u.mV``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF; broadcastable to
        ``self.varshape``. Expected positive for physical behavior. Default is
        ``250. * u.pF``.
    tau_m : ArrayLike, optional
        Membrane time constant :math:`\tau_m` in ms; broadcastable to
        ``self.varshape``. Expected positive. Default is ``10. * u.ms``.
    t_ref : ArrayLike, optional
        Absolute refractory duration :math:`t_{ref}` in ms. Converted to
        integer simulation steps using ``ceil(t_ref / dt)``. Default is
        ``2. * u.ms``.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_{th}` in mV; broadcastable to ``self.varshape``.
        Default is ``-55. * u.mV``.
    V_reset : ArrayLike, optional
        Post-spike reset potential :math:`V_{reset}` in mV; broadcastable to
        ``self.varshape``. Default is ``-70. * u.mV``.
    I_e : ArrayLike, optional
        Constant external current :math:`I_e` in pA; scalar or array
        broadcastable to ``self.varshape``. Default is ``0. * u.pA``.
    V_min : ArrayLike or None, optional
        Optional lower membrane bound :math:`V_{min}` in mV. If ``None``,
        no lower clipping is applied. Default is ``None``.
    V_initializer : Callable, optional
        Initializer for membrane state ``V`` in :meth:`init_state`. Output
        must be shape-compatible with ``self.varshape`` (and optional batch
        prefix). Default is ``braintools.init.Constant(-70. * u.mV)``.
    spk_fun : Callable, optional
        Surrogate spike function used by :meth:`get_spike`. Default is
        ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy inherited from :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` reproduces NEST hard reset behavior. Default is ``'hard'``.
    refractory_input : bool, optional
        If ``False``, delta inputs during refractory are ignored. If ``True``,
        refractory-arriving delta jumps are accumulated in
        ``refractory_spike_buffer`` and applied after refractory release.
        Default is ``False``.
    ref_var : bool, optional
        If ``True``, allocate boolean refractory state ``self.refractory`` for
        inspection. Default is ``False``.
    name : str or None, optional
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table:: Parameter mapping to model symbols
       :header-rows: 1
       :widths: 17 28 14 16 35

       * - Parameter
         - Type / shape / unit
         - Default
         - Math symbol
         - Semantics
       * - ``in_size``
         - :class:`~brainstate.typing.Size`; scalar/tuple
         - required
         - --
         - Defines population/state shape ``self.varshape``.
       * - ``E_L``
         - ArrayLike, broadcastable to ``self.varshape`` (mV)
         - ``-70. * u.mV``
         - :math:`E_L`
         - Resting membrane potential.
       * - ``C_m``
         - ArrayLike, broadcastable (pF)
         - ``250. * u.pF``
         - :math:`C_m`
         - Membrane capacitance in subthreshold propagator.
       * - ``tau_m``
         - ArrayLike, broadcastable (ms)
         - ``10. * u.ms``
         - :math:`\tau_m`
         - Membrane leak time constant.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), step-converted by ``ceil``
         - ``2. * u.ms``
         - :math:`t_{ref}`
         - Absolute refractory duration.
       * - ``V_th`` and ``V_reset``
         - ArrayLike, broadcastable (mV)
         - ``-55. * u.mV``, ``-70. * u.mV``
         - :math:`V_{th}`, :math:`V_{reset}`
         - Threshold and reset voltages.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * u.pA``
         - :math:`I_e`
         - Constant injected current.
       * - ``V_min``
         - ArrayLike broadcastable (mV) or ``None``
         - ``None``
         - :math:`V_{min}`
         - Optional lower clamp before threshold test.
       * - ``V_initializer``
         - Callable
         - ``Constant(-70. * u.mV)``
         - --
         - Initializes membrane state ``V``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate spike output function.
       * - ``spk_reset``
         - str
         - ``'hard'``
         - --
         - Reset mode inherited from base neuron class.
       * - ``refractory_input``
         - bool
         - ``False``
         - --
         - Controls refractory-period treatment of delta inputs.
       * - ``ref_var``
         - bool
         - ``False``
         - --
         - Enables persistent refractory boolean state.
       * - ``name``
         - str | None
         - ``None``
         - --
         - Optional node identifier.

    Raises
    ------
    ValueError
        If parameter initialization or broadcasting fails (for example due to
        incompatible array shape in ``braintools.init.param``).
    TypeError
        If provided values are not compatible with expected units/types
        (mV, ms, pF, pA, or callable initializers/spike functions).
    KeyError
        At runtime, if required simulation context entries (for example ``t``
        or ``dt``) are missing when :meth:`update` is called.
    AttributeError
        If :meth:`update` is called before :meth:`init_state` creates required
        state variables.

    Attributes
    ----------
    V : HiddenState
        Membrane potential.
    last_spike_time : ShortTermState
        Time of the last spike, used to implement the refractory period.
    refractory : HiddenState
        Neuron refractory state (only present if ``ref_var=True``).

    Notes
    -----
    - State variables are ``V``, ``last_spike_time``,
      ``refractory_step_count``, and ``refractory_spike_buffer``. ``refractory``
      exists only when ``ref_var=True``.
    - Continuous current input ``x`` is combined with ``I_e`` through
      :meth:`sum_current_inputs` in the same simulation step.
    - Delta events from :meth:`sum_delta_inputs` are interpreted in mV and
      added as instantaneous voltage jumps.

    Examples
    --------
    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_delta(in_size=10, t_ref=2.0 * u.ms)
       ...     neu.init_state(batch_size=1)
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         spk = neu.update(x=500.0 * u.pA)
       ...     _ = spk.shape

    .. code-block:: python

       >>> import brainpy
       >>> import brainstate
       >>> import brainunit as u
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     neu = brainpy.state.iaf_psc_delta(in_size=(2,), V_min=-80.0 * u.mV)
       ...     neu.init_state()
       ...     with brainstate.environ.context(t=0.0 * u.ms):
       ...         _ = neu.update(x=120.0 * u.pA)

    References
    ----------
    .. [1] Rotter S, Diesmann M (1999). Exact simulation of time-invariant linear
           systems with applications to neuronal modeling. Biological Cybernetics
           81:381-402. DOI: https://doi.org/10.1007/s004220050570
    .. [2] Diesmann M, Gewaltig M-O, Rotter S, & Aertsen A (2001). State space
           analysis of synchronous spiking in cortical neural networks.
           Neurocomputing 38-40:565-571.
           DOI: https://doi.org/10.1016/S0925-2312(01)00409-X

    See Also
    --------
    LIF : Leaky integrate-and-fire with current-based synapses
    LIFRef : Leaky integrate-and-fire with refractory period (brainpy parameterization)
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
        I_e: ArrayLike = 0. * u.pA,
        V_min: ArrayLike = None,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        refractory_input: bool = False,
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.V_min = V_min
        self.V_initializer = V_initializer
        self.refractory_input = refractory_input
        self.ref_var = ref_var

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize membrane and refractory runtime states.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, states are created
            with shape ``self.varshape``; otherwise with
            ``(batch_size,) + self.varshape``.
        **kwargs : Any
            Unused compatibility arguments.


        Raises
        ------
        ValueError
            If initializer outputs cannot be broadcast to target state shape.
        TypeError
            If initializer values are incompatible with required numeric/unit
            conversions.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        self.V = brainstate.HiddenState(V)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))
        self.refractory_spike_buffer = brainstate.ShortTermState(u.math.zeros_like(V))
        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def get_spike(self, V: ArrayLike = None):
        r"""Evaluate surrogate spike activation for a voltage tensor.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Membrane voltage input in mV, broadcast-compatible with
            ``self.varshape``. If ``None``, the method uses ``self.V.value``.

        Returns
        -------
        out : dict
            Surrogate spike output from ``self.spk_fun`` with the same shape as
            ``V`` (or ``self.V.value`` when ``V`` is ``None``).

        Raises
        ------
        TypeError
            If ``V`` cannot participate in arithmetic with membrane parameters
            due to incompatible dtype/unit.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        r"""Convert refractory duration to integer simulation-step counts.

        Returns
        -------
        out : Any
            Integer array (``jnp.int32``) with shape broadcast-compatible to
            ``self.varshape``; computed as ``ceil(self.t_ref / dt)``.

        Raises
        ------
        KeyError
            If simulation context does not provide ``dt``.
        TypeError
            If ``t_ref`` and ``dt`` are not unit-compatible for division.
        """
        dt = brainstate.environ.get_dt()
        # NEST converts refractory duration to grid steps by rounding up to the
        # next simulation step.
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def update(self, x=0. * u.pA):
        r"""Advance the neuron by one simulation step.

        Parameters
        ----------
        x : ArrayLike, optional
            External current input in pA for this step. It is combined with
            ``I_e`` and additional current sources from
            :meth:`sum_current_inputs`.

        Returns
        -------
        out : jax.Array
            Surrogate spike output from :meth:`get_spike` with shape
            ``self.V.value.shape``. The returned spike signal is computed from
            pre-reset post-threshold voltage ``v_post``.

        Raises
        ------
        KeyError
            If simulation context does not provide required entries ``t`` or
            ``dt``.
        AttributeError
            If required states are missing because :meth:`init_state` has not
            been called.
        TypeError
            If input/state values are not unit-compatible with expected pA/mV
            arithmetic.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        last_v = self.V.value
        ref_steps = self.refractory_step_count.value

        # Exact subthreshold propagation for one fixed simulation step.
        decay = u.math.exp(-dt / self.tau_m)
        i_total = self.sum_current_inputs(self.I_e + x, last_v)
        v_candidate = self.E_L + (last_v - self.E_L) * decay + (i_total / self.C_m) * self.tau_m * (1. - decay)
        delta_v = self.sum_delta_inputs(u.math.zeros_like(last_v))
        v_candidate = v_candidate + delta_v

        if self.refractory_input:
            v_candidate = v_candidate + self.refractory_spike_buffer.value

        if self.V_min is not None:
            v_candidate = u.math.maximum(v_candidate, self.V_min)

        not_refractory = ref_steps == 0
        v_post = u.math.where(not_refractory, v_candidate, last_v)

        if self.refractory_input:
            refr_decay = u.math.exp(-ref_steps * dt / self.tau_m)
            self.refractory_spike_buffer.value = u.math.where(
                not_refractory,
                u.math.zeros_like(self.refractory_spike_buffer.value),
                self.refractory_spike_buffer.value + delta_v * refr_decay
            )

        ref_steps = u.math.where(not_refractory, ref_steps, ref_steps - 1)

        spike_cond = v_post >= self.V_th
        self.refractory_step_count.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, self._refractory_counts(), ref_steps)
        )
        self.V.value = u.math.where(spike_cond, self.V_reset, v_post)
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_cond, t + dt, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return self.get_spike(v_post)
