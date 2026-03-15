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

from typing import Callable, Sequence

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
    'iaf_chs_2007',
]


class iaf_chs_2007(NESTNeuron):
    r"""NEST-compatible ``iaf_chs_2007`` spike-response neuron model.

    Description
    -----------

    ``iaf_chs_2007`` is a discrete-time linear spike-response neuron model
    developed for analyzing thalamic filtering of retinal spike trains
    (Carandini, Horton, and Sincich, 2007). The normalized membrane potential
    is the sum of three components:

    - An alpha-shaped postsynaptic potential (PSP) waveform ``V_syn`` driven
      by excitatory spike inputs,
    - A post-spike reset/after-hyperpolarization (AHP) component ``V_spike``
      that decays exponentially after each spike emission,
    - An optional externally prepared noise trace scaled by ``V_noise``.

    This implementation mirrors NEST's C++ model
    ``models/iaf_chs_2007.{h,cpp}`` to ensure semantic equivalence: exact
    discrete-time update order, normalized voltage conventions
    (:math:`U_{th} = 1`, :math:`E_L = 0`), positive-weight-only PSP
    accumulation, and external noise injection without refractory state.

    **1. Model equations and exact integration**

    Let :math:`h = dt` (in ms) denote the global integration step, and
    :math:`k` index discrete time steps. NEST precomputes four propagators at
    initialization:

    .. math::

       P_{11} = e^{-h / \tau_{\mathrm{epsp}}}, \qquad
       P_{22} = P_{11}, \qquad
       P_{30} = e^{-h / \tau_{\mathrm{reset}}},

    .. math::

       P_{21} = U_{\mathrm{epsp}} \, e \, P_{11} \, \frac{h}{\tau_{\mathrm{epsp}}}.

    Here :math:`U_{\mathrm{epsp}}` sets the EPSP peak amplitude and
    :math:`P_{21}` is derived from the alpha-function kernel integral.

    Each simulation step updates state in the following order:

    .. math::

       V_{\mathrm{syn}}^{k+1} = P_{22} \, V_{\mathrm{syn}}^k
       + P_{21} \, i_{\mathrm{syn}}^k,

    .. math::

       i_{\mathrm{syn}}^{k+1} = P_{11} \, i_{\mathrm{syn}}^k
       + \max(w_k, 0),

    .. math::

       V_{\mathrm{spike}}^{k+1} = P_{30} \, V_{\mathrm{spike}}^k,

    .. math::

       V_m^{k+1} = V_{\mathrm{syn}}^{k+1} + V_{\mathrm{spike}}^{k+1}
       + U_{\mathrm{noise}} \, \eta_k,

    where :math:`w_k = \sum_{j} w_j \delta(t - t_j^{spike})` collects incoming
    excitatory spike weights delivered at step :math:`k`, and
    :math:`\eta_k` is the externally provided noise sample (if configured).

    Spike emission uses the hard threshold :math:`U_{\mathrm{th}} = 1`:

    .. math::

       V_m^{k+1} \ge 1 \quad \Longrightarrow \quad
       V_{\mathrm{spike}}^{k+1} \leftarrow V_{\mathrm{spike}}^{k+1} - U_{\mathrm{reset}},
       \quad
       V_m^{k+1} \leftarrow V_m^{k+1} - U_{\mathrm{reset}}.

    Both the reset/AHP component and the total membrane potential are
    decremented by :math:`U_{\mathrm{reset}}` upon spike emission. No
    refractory clamping occurs; the neuron can spike again immediately if
    :math:`V_m` remains above threshold.

    **2. Update ordering and NEST semantics**

    The per-step operation sequence is identical to NEST's C++
    ``update()`` routine:

    1. Update ``V_syn`` from the previous step's ``i_syn_ex``.
    2. Decay ``i_syn_ex`` by ``P11``.
    3. Add arriving excitatory spike weights (non-negative values only;
       negative inputs are clamped to zero).
    4. Decay ``V_spike`` by ``P30``.
    5. Sample and add noise term if ``V_noise > 0`` and noise buffer is
       non-empty.
    6. Compute total ``V_m`` and apply threshold/reset/spike emission logic.

    A critical consequence of this ordering is that a spike arriving in
    the current step immediately increments ``i_syn_ex``, but the resulting
    ``V_syn`` contribution appears in ``V_m`` only from the next step onward
    (one-step synaptic delay).

    **3. Noise semantics**

    NEST expects noise to be externally prepared (e.g., from a Gaussian
    distribution with specified variance) and supplied as a pre-generated
    sequence. This implementation follows the same convention:

    - Noise is active only if ``V_noise > 0.`` and the ``noise`` buffer is
      non-empty.
    - One sample per neuron per step is consumed from the flat noise array
      using a ``position`` index.
    - If the noise buffer is exhausted before the end of the simulation, an
      ``IndexError`` is raised.
      **Users must provide a noise array of length at least equal to the total number of simulation steps.**

    **4. Assumptions, constraints, and computational complexity**

    - All model parameters are scalar or broadcastable to ``self.varshape``.
    - Construction-time constraints enforce ``V_epsp >= 0``,
      ``V_reset >= 0``, ``tau_epsp > 0``, ``tau_reset > 0`` elementwise.
    - The model operates in normalized voltage units where
      :math:`E_L = 0` (rest), :math:`U_{th} = 1` (threshold).
    - Negative input weights are silently clamped to zero (matching NEST's
      positive-weight-only convention for this model).
    - Unlike standard LIF models, there is no refractory state or explicit
      continuous current input handler; the ``update(x=...)`` argument is
      unused by design.
    - Per-step complexity is :math:`O(|\mathrm{state}|)` for state
      propagation, plus :math:`O(K)` for collecting ``K`` delta inputs.

    Parameters
    ----------
    in_size : Size
        Population shape specification. Model parameters and states are
        broadcast to ``self.varshape`` derived from ``in_size``.
    tau_epsp : ArrayLike, optional
        EPSP time constant :math:`\tau_{\mathrm{epsp}}` in ms, broadcastable
        to ``self.varshape``. Must be strictly positive elementwise.
        Controls the rise/decay timescale of the alpha-shaped PSP kernel.
        Default is ``8.5 * u.ms``.
    tau_reset : ArrayLike, optional
        Post-spike reset/AHP time constant :math:`\tau_{\mathrm{reset}}` in
        ms, broadcastable to ``self.varshape``. Must be strictly positive
        elementwise. Governs exponential decay of the ``V_spike`` component
        after each spike. Default is ``15.4 * u.ms``.
    V_epsp : ArrayLike, optional
        Normalized maximal EPSP amplitude :math:`U_{\mathrm{epsp}}`
        (dimensionless), broadcastable to ``self.varshape``. Must be
        non-negative elementwise. Sets the peak height of the PSP waveform
        per unit weight. Default is ``0.77``.
    V_reset : ArrayLike, optional
        Normalized reset/AHP magnitude :math:`U_{\mathrm{reset}}`
        (dimensionless), broadcastable to ``self.varshape``. Must be
        non-negative elementwise. Both ``V_spike`` and ``V_m`` are decremented
        by this value upon threshold crossing. Default is ``2.31``.
    V_noise : ArrayLike, optional
        Noise scale factor :math:`U_{\mathrm{noise}}` (dimensionless),
        broadcastable to ``self.varshape``. Multiplies externally provided
        noise samples. No noise is added if ``V_noise == 0`` or if the
        ``noise`` buffer is empty. Default is ``0.0``.
    noise : Sequence[float] | np.ndarray | None, optional
        Externally prepared noise samples (dimensionless). If provided,
        must be a flat 1D sequence of at least ``num_steps`` values, where
        ``num_steps`` is the total simulation duration in steps. One sample
        per neuron per step is consumed sequentially. If ``None``, no noise
        is applied. Default is ``None``.
    gsl_error_tol : ArrayLike, optional
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
        Default is ``1e-6``.
    V_initializer : Callable, optional
        Initializer used by :meth:`init_state` for membrane potential ``V``.
        Must return dimensionless values with shape compatible with
        ``self.varshape`` (and optional batch prefix). Default is
        ``braintools.init.Constant(0.0)``.
    spk_fun : Callable, optional
        Surrogate spike function used by :meth:`get_spike` and
        :meth:`update`. Receives normalized threshold distance tensor.
        Default is ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy forwarded to :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` matches NEST's hard subtraction reset. Default is ``'hard'``.
    name : str or None, optional
        Optional node name.

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
         - Defines ``self.varshape`` for parameter/state broadcasting.
       * - ``tau_epsp``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``8.5 * u.ms``
         - :math:`\tau_{\mathrm{epsp}}`
         - Alpha-kernel time constant for EPSP waveform.
       * - ``tau_reset``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``15.4 * u.ms``
         - :math:`\tau_{\mathrm{reset}}`
         - Exponential decay time constant for post-spike AHP.
       * - ``V_epsp``
         - ArrayLike, broadcastable (dimensionless), ``>= 0``
         - ``0.77``
         - :math:`U_{\mathrm{epsp}}`
         - Normalized peak EPSP amplitude per unit weight.
       * - ``V_reset``
         - ArrayLike, broadcastable (dimensionless), ``>= 0``
         - ``2.31``
         - :math:`U_{\mathrm{reset}}`
         - Normalized reset/AHP decrement applied on spike.
       * - ``V_noise``
         - ArrayLike, broadcastable (dimensionless)
         - ``0.0``
         - :math:`U_{\mathrm{noise}}`
         - Noise scale factor; zero disables noise injection.
       * - ``noise``
         - Sequence[float] | np.ndarray | None (dimensionless)
         - ``None``
         - :math:`\eta_k`
         - Externally prepared noise samples; must have length >= ``num_steps``.
       * - ``gsl_error_tol``
         - ArrayLike, broadcastable, unitless, ``> 0``
         - ``1e-6``
         - --
         - Local absolute tolerance for the embedded RKF45 error estimate.
       * - ``V_initializer``
         - Callable returning (dimensionless)
         - ``Constant(0.0)``
         - :math:`V_m(0)`
         - Initial membrane potential at ``t=0``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Differentiable surrogate for spike emission.
       * - ``spk_reset``
         - str (``'hard'`` or ``'soft'``)
         - ``'hard'``
         - --
         - Reset mode; ``'hard'`` matches NEST semantics.

    Raises
    ------
    ValueError
        - If ``V_epsp < 0`` or ``V_reset < 0`` elementwise.
        - If ``tau_epsp <= 0`` or ``tau_reset <= 0`` elementwise.
    IndexError
        If the noise buffer is exhausted before the end of the simulation.
        Ensure the ``noise`` array has length at least equal to the total
        number of simulation steps.

    Notes
    -----

    **Key differences from standard LIF models:**

    - Operates in **normalized voltage units** where :math:`E_L = 0` and
      :math:`U_{th} = 1`. There is no explicit leak conductance or external
      current integration beyond the PSP and AHP components.
    - **No refractory state**: The neuron can spike immediately again if
      :math:`V_m` remains above threshold after reset.
    - **Positive-weight-only synapses**: Negative incoming spike weights are
      clamped to zero, matching NEST's convention for this model.
    - **External noise injection**: Noise is not generated internally but must
      be pre-prepared and supplied as a flat array. The model consumes one
      sample per neuron per step from the ``noise`` buffer using a sequential
      index.
    - **No continuous current input**: Unlike ``iaf_psc_exp`` and similar
      models, ``iaf_chs_2007`` in NEST has no ``CurrentEvent`` handler. The
      ``update(x=...)`` argument is present for API compatibility but is
      intentionally unused.

    **Spike-response interpretation:**

    The model is a discrete-time linear spike-response model (SRM) where each
    incoming spike triggers an alpha-shaped PSP and each output spike triggers
    an exponential AHP. The total membrane potential is the linear sum of
    these components plus optional noise. This differs from integrate-and-fire
    models that compute a continuous leak current; here, the "leak" is
    implicit in the exponential decay of the PSP and AHP kernels.

    **Computational considerations**


    - State propagation uses exact exponential integration with precomputed
      propagators ``P11``, ``P22``, ``P30``, and ``P21``, ensuring
      machine-precision accuracy regardless of ``dt`` (within numerical
      precision of ``exp`` and floating-point arithmetic).
    - The model performs all computations in NumPy on the host before
      transferring results to JAX, which may limit GPU acceleration
      efficiency. This design choice ensures exact NEST compatibility,
      including identical floating-point rounding behavior.
    - For large-scale simulations, consider using the standard ``LIF`` or
      ``ExpIF`` models, which are fully JIT-compatible and GPU-accelerated.

    References
    ----------
    .. [1] Carandini M, Horton JC, Sincich LC (2007). Thalamic filtering of
           retinal spike trains by postsynaptic summation. Journal of Vision
           7(14):20, 1-11. DOI: https://doi.org/10.1167/7.14.20
    .. [2] Rotter S, Diesmann M (1999). Exact simulation of time-invariant
           linear systems with applications to neuronal modeling.
           Biological Cybernetics 81:381-402.
           DOI: https://doi.org/10.1007/s004220050570

    Examples
    --------
    Create a population of 100 neurons and drive them with Poisson spike input:

    .. code-block:: python

        >>> import brainpy.state as bp
        >>> import saiunit as u
        >>> import brainstate as bs
        >>> import brainevent as be
        >>> import numpy as np
        >>>
        >>> # Prepare external noise (Gaussian)
        >>> num_steps = 1000
        >>> noise = np.random.randn(num_steps)
        >>>
        >>> # Create model
        >>> model = bp.iaf_chs_2007(
        ...     in_size=100,
        ...     tau_epsp=8.5 * u.ms,
        ...     tau_reset=15.4 * u.ms,
        ...     V_epsp=0.77,
        ...     V_reset=2.31,
        ...     V_noise=0.1,
        ...     noise=noise
        ... )
        >>>
        >>> # Poisson spike source
        >>> poisson = be.nn.PoissonEncoder(in_size=100)
        >>>
        >>> # Projection
        >>> proj = bp.AlignPostProj(
        ...     comm=be.nn.AllToAll(pre_size=100, post_size=100, w_init=0.05),
        ...     out=bp.CUBA(),
        ...     post=model
        ... )
        >>>
        >>> # Simulate
        >>> with bs.environ.context(dt=0.1 * u.ms):
        ...     model.init_state()
        ...     for _ in range(num_steps):
        ...         inp = poisson.generate()
        ...         proj(inp)
        ...         spk = model.update()
        ...
    """

    __module__ = 'brainpy.state'

    _U_TH = 1.0  # NEST hard-coded normalized threshold.
    _E_L = 0.0  # NEST hard-coded normalized rest potential.
    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        tau_epsp: ArrayLike = 8.5 * u.ms,
        tau_reset: ArrayLike = 15.4 * u.ms,
        V_epsp: ArrayLike = 0.77,
        V_reset: ArrayLike = 2.31,
        V_noise: ArrayLike = 0.0,
        noise: Sequence[float] | np.ndarray | None = None,
        gsl_error_tol: ArrayLike = 1e-6,
        V_initializer: Callable = braintools.init.Constant(0.0),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.tau_epsp = braintools.init.param(tau_epsp, self.varshape)
        self.tau_reset = braintools.init.param(tau_reset, self.varshape)
        self.V_epsp = braintools.init.param(V_epsp, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.V_noise = braintools.init.param(V_noise, self.varshape)
        self.gsl_error_tol = gsl_error_tol

        dftype = brainstate.environ.dftype()
        self.noise = np.asarray([] if noise is None else u.math.asarray(noise), dtype=dftype).reshape(-1)
        self.V_initializer = V_initializer

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

    def _validate_parameters(self):
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_epsp, self.V_reset, self.tau_epsp)):
            return

        if np.any(self.V_epsp < 0.0):
            raise ValueError('EPSP amplitude V_epsp cannot be negative.')
        if np.any(self.V_reset < 0.0):
            raise ValueError('Reset magnitude V_reset cannot be negative.')
        if np.any(self.tau_epsp <= 0.0 * u.ms) or np.any(self.tau_reset <= 0.0 * u.ms):
            raise ValueError('All time constants must be strictly positive.')

    def init_state(self, **kwargs):
        r"""Initialize all state variables for the neuron population.

        Creates and registers the following states with brainstate:

        - ``i_syn_ex``: Excitatory synaptic current state (ShortTermState,
          initialized to zeros).
        - ``V_syn``: EPSP waveform state (ShortTermState, initialized to zeros).
        - ``V_spike``: Post-spike reset/AHP state (ShortTermState, initialized
          to zeros).
        - ``V``: Normalized membrane potential (HiddenState, initialized via
          ``V_initializer``).
        - ``position``: Current index into the ``noise`` buffer (ShortTermState,
          initialized to zeros as int64).
        - ``last_spike_time``: Last spike time in ms (ShortTermState,
          initialized to ``-1e7 * u.ms`` to indicate no previous spike).

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        All states except ``V`` are initialized to zero. The membrane potential
        ``V`` is initialized using ``self.V_initializer``, which defaults to
        ``Constant(0.0)`` (normalized resting potential).
        """
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()

        V = braintools.init.param(self.V_initializer, self.varshape)
        zeros = u.math.zeros(self.varshape, dtype=dftype)
        zeros_per_ms = u.math.zeros(self.varshape, dtype=dftype) / u.ms

        self.i_syn_ex = brainstate.ShortTermState(zeros_per_ms)
        self.V_syn = brainstate.ShortTermState(zeros)
        self.V_spike = brainstate.ShortTermState(zeros)
        self.V = brainstate.HiddenState(V)
        self.position = brainstate.ShortTermState(u.math.zeros(self.varshape, dtype=ditype))

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)

    def reset_state(self, batch_size: int = None, **kwargs):
        r"""Reset all state variables to their initial values.

        Resets all state variables to the same values as :meth:`init_state`,
        but operates on existing state instances rather than creating new ones.
        This is useful for re-initializing a model between simulation runs
        without destroying the computational graph.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional batch dimension size. If provided, states will have shape
            ``(batch_size, *self.varshape)``. If ``None``, states have shape
            ``self.varshape``.
        **kwargs
            Additional keyword arguments (currently unused).

        Notes
        -----
        The ``position`` index into the noise buffer is reset to zero. If you
        want to continue consuming the noise sequence from where it left off,
        manually preserve and restore ``self.position.value`` before/after
        calling this method.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        dftype = brainstate.environ.dftype()
        zeros = np.zeros_like(np.asarray(u.math.asarray(V), dtype=dftype))
        ditype = brainstate.environ.ditype()
        idx0 = np.zeros_like(zeros, dtype=ditype)

        self.i_syn_ex.value = jnp.asarray(zeros, dtype=dftype)
        self.V_syn.value = jnp.asarray(zeros, dtype=dftype)
        self.V_spike.value = jnp.asarray(zeros, dtype=dftype)
        self.V.value = jnp.asarray(u.math.asarray(V), dtype=dftype)
        self.position.value = jnp.asarray(idx0, dtype=ditype)
        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )

    def get_spike(self, V: ArrayLike = None):
        r"""Compute spike output using the surrogate spike function.

        Applies the surrogate gradient function ``spk_fun`` to the normalized
        threshold distance :math:`V_m - U_{th}`, where :math:`U_{th} = 1`. This
        produces a differentiable spike output suitable for gradient-based
        training.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Membrane potential tensor (dimensionless), with shape matching
            ``self.varshape`` (plus optional batch prefix). If ``None``, uses
            ``self.V.value`` (the current membrane state). Default is ``None``.

        Returns
        -------
        ArrayLike
            Spike output tensor with the same shape as ``V``. Values are
            typically in [0, 1] or continuous approximations depending on the
            chosen ``spk_fun``.

        Notes
        -----
        Unlike typical LIF models where :math:`U_{reset} < U_{th}`, this model
        has :math:`U_{reset} > U_{th}` by default (2.31 vs 1.0). The spike
        function operates on :math:`V_m - U_{th}` to detect threshold crossings;
        the surrogate gradient enables backpropagation through spike events.
        """
        V = self.V.value if V is None else V
        # Unlike typical LIF models, here U_reset > U_th by default.
        # Therefore, we scale directly with threshold crossing sign.
        v_scaled = V - self._U_TH
        return self.spk_fun(v_scaled)

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for the iaf_chs_2007 ODE system.

        The underlying continuous-time ODEs are:

        - ``di_syn_ex/dt = -i_syn_ex / tau_epsp``
        - ``dV_syn/dt = V_epsp * e / tau_epsp * i_syn_ex - V_syn / tau_epsp``
        - ``dV_spike/dt = -V_spike / tau_reset``

        Parameters
        ----------
        state : DotDict
            Keys: i_syn_ex, V_syn, V_spike -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, unstable -- mutable auxiliary data carried
            through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        di_syn_ex = -state.i_syn_ex / self.tau_epsp
        dV_syn = self.V_epsp * np.e / self.tau_epsp * state.i_syn_ex - state.V_syn / self.tau_epsp
        dV_spike = -state.V_spike / self.tau_reset

        return DotDict(i_syn_ex=di_syn_ex, V_syn=dV_syn, V_spike=dV_spike)

    def _event_fn(self, state, extra, accept):
        """In-loop spike detection and reset handling.

        After each accepted substep, computes the total membrane potential
        ``V_m = V_syn + V_spike + noise`` and applies the threshold/reset
        logic: if ``V_m >= U_th``, decrements ``V_spike`` by ``U_reset``.

        Parameters
        ----------
        state : DotDict
            Keys: i_syn_ex, V_syn, V_spike -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, unstable, noise_term.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (new_state, new_extra) DotDicts with updated spike/reset info.
        """
        unstable = extra.unstable | jnp.any(
            accept & (
                (u.get_mantissa(u.math.abs(state.V_syn)) > 1e6) |
                (u.get_mantissa(u.math.abs(state.V_spike)) > 1e6)
            )
        )

        V_m = state.V_syn + state.V_spike + extra.noise_term
        spike_now = accept & (u.get_mantissa(V_m) >= self._U_TH)
        spike_mask = extra.spike_mask | spike_now

        new_V_spike = u.math.where(spike_now, state.V_spike - self.V_reset, state.V_spike)

        new_state = DotDict({**state, 'V_spike': new_V_spike})
        new_extra = DotDict({**extra, 'spike_mask': spike_mask, 'unstable': unstable})
        return new_state, new_extra

    def update(self, x=0.0):
        r"""Advance the neuron state by one simulation time step.

        Implements the discrete-time update rule following NEST's exact
        sequence:

        1. Update ``V_syn`` from previous ``i_syn_ex`` using propagator ``P21``.
        2. Decay ``i_syn_ex`` by ``P11``.
        3. Add arriving excitatory delta inputs (non-negative weights only).
        4. Decay ``V_spike`` by ``P30``.
        5. Sample and add noise term if ``V_noise > 0`` and noise is available.
        6. Compute total membrane potential ``V_m`` and apply threshold/reset
           logic: if :math:`V_m \ge U_{th}`, decrement both ``V_spike`` and
           ``V_m`` by ``U_{reset}``, and update ``last_spike_time``.
        7. Return spike output via surrogate function.

        Parameters
        ----------
        x : ArrayLike, optional
            Continuous input current (unused by design). NEST's C++
            ``iaf_chs_2007`` has no ``CurrentEvent`` handler; this argument is
            present for API compatibility with other neuron models but is
            intentionally ignored. Default is ``0.0``.

        Returns
        -------
        ArrayLike
            Spike output tensor with shape matching ``self.V.value``. Computed
            by :meth:`get_spike` using the normalized threshold distance
            :math:`V_m - U_{th}` and the surrogate function ``spk_fun``.

        Raises
        ------
        IndexError
            If the noise buffer is exhausted (``position >= len(noise)``) before
            the end of the simulation step. Ensure the ``noise`` array has
            length at least equal to the total number of simulation steps.

        Notes
        -----
        **Update ordering semantics:**

        The key consequence of NEST's update order is that a spike arriving in
        the current step immediately increments ``i_syn_ex``, but the resulting
        ``V_syn`` contribution appears in the total ``V_m`` only from the next
        step onward. This introduces an effective one-step synaptic delay for
        all spike inputs.

        **Propagator precomputation:**

        Each step recomputes the propagators ``P11``, ``P22``, ``P30``, and
        ``P21`` from the current ``dt`` and model parameters. This ensures
        correctness even if ``dt`` changes between steps, but incurs a small
        computational cost. In practice, ``dt`` is typically fixed for the
        entire simulation.

        **Noise consumption:**

        One noise sample per neuron per step is consumed from the flat
        ``self.noise`` array using the ``position`` index. For vectorized or
        batched states, each element independently increments its ``position``
        index if ``V_noise > 0`` for that element.
        **The noise buffer must be pre-allocated with at least ``num_steps`` samples.**

        **No refractory state:**

        Unlike standard LIF models, this model has no refractory clamping. If
        :math:`V_m` remains above threshold after reset (possible if
        :math:`U_{reset}` is small relative to the PSP amplitude), the neuron
        can spike again immediately in the next step.

        **Computational complexity:**

        Per-step cost is :math:`O(|\mathrm{state}|)` for state propagation,
        plus :math:`O(K)` for collecting ``K`` delta inputs, plus
        :math:`O(|\mathrm{state}|)` for noise sampling (if active). All
        operations are vectorized via JAX and the adaptive RKF45 integrator.
        """
        # NEST iaf_chs_2007 has no CurrentEvent handler; x is intentionally unused.
        del x

        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables.
        i_syn_ex = self.i_syn_ex.value
        V_syn = self.V_syn.value
        V_spike = self.V_spike.value
        pos = self.position.value
        h = self.integration_step.value

        # Noise term computation (pre-integration, stays constant during substeps).
        noise_term = u.math.zeros(self.varshape, dtype=dftype)
        if self.noise.size > 0:
            U_noise = self.V_noise
            use_noise = U_noise > 0.0
            if np.any(u.get_mantissa(use_noise)):
                pos_np = np.asarray(u.math.asarray(pos), dtype=int)
                if np.any(pos_np[np.asarray(u.get_mantissa(use_noise), dtype=bool)] >= self.noise.size):
                    raise IndexError(
                        'Noise signal exhausted before end of simulation. '
                        'Provide a noise vector at least as long as all simulated steps.'
                    )
                sampled_noise = u.math.zeros(self.varshape, dtype=dftype)
                use_mask = np.asarray(u.get_mantissa(use_noise), dtype=bool)
                noise_vals = jnp.asarray(self.noise[pos_np.clip(0, self.noise.size - 1)], dtype=dftype)
                sampled_noise = u.math.where(use_noise, noise_vals, sampled_noise)
                noise_term = U_noise * sampled_noise
                pos = u.math.where(use_noise, pos + 1, pos)

        # Synaptic spike inputs (positive-weight-only, applied before integration).
        w_ex = self.sum_delta_inputs(u.math.zeros(self.varshape, dtype=dftype), label='w_ex')
        w_ex = u.math.maximum(w_ex, 0.0)
        i_syn_ex = i_syn_ex + w_ex / u.ms

        # Adaptive RKF45 integration via generic integrator.
        ode_state = DotDict(i_syn_ex=i_syn_ex, V_syn=V_syn, V_spike=V_spike)
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            unstable=jnp.array(False),
            noise_term=noise_term,
        )

        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        i_syn_ex = ode_state.i_syn_ex
        V_syn = ode_state.V_syn
        V_spike = ode_state.V_spike
        spike_mask = extra.spike_mask
        unstable = extra.unstable

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in iaf_chs_2007 dynamics.'
        )

        # Compute total membrane potential.
        V_m = V_syn + V_spike + noise_term

        # Write back state.
        self.i_syn_ex.value = i_syn_ex
        self.V_syn.value = V_syn
        self.V_spike.value = V_spike
        self.V.value = V_m
        self.position.value = jnp.asarray(u.get_mantissa(pos), dtype=ditype)
        self.integration_step.value = h
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        return u.math.asarray(spike_mask, dtype=dftype)
