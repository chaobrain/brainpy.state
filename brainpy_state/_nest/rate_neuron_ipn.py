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
import saiunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest.lin_rate import _lin_rate_base
from ._utils import is_tracer

__all__ = [
    'rate_neuron_ipn',
]


class rate_neuron_ipn(_lin_rate_base):
    r"""NEST-compatible input-noise rate-neuron template with stochastic dynamics.

    Implements the NEST ``rate_neuron_ipn<TNonlinearities>`` template model, a
    continuous-time rate neuron with additive Gaussian input noise. With default
    settings, this is equivalent to NEST's ``lin_rate_ipn``. The model supports
    custom input nonlinearities, multiplicative coupling (rate-dependent synaptic
    efficacy), and flexible input summation modes.

    Mathematical Description
    ========================

    **1. Continuous-Time Stochastic Dynamics**

    The rate state :math:`X(t)` evolves according to the Langevin equation:

    .. math::

       \tau\,dX(t) = [-\lambda X(t) + \mu + I_\mathrm{net}(t)]\,dt
       + \sqrt{\tau}\,\sigma\,dW(t),

    where:

    - :math:`\tau > 0` is the time constant (ms).
    - :math:`\lambda \ge 0` is the passive decay rate (dimensionless). Controls
      exponential relaxation; :math:`\lambda=0` yields driftless diffusion.
    - :math:`\mu` is the mean drive (dimensionless, external constant input).
    - :math:`\sigma \ge 0` is the input-noise strength (dimensionless).
    - :math:`W(t)` is a standard Wiener process.
    - :math:`I_\mathrm{net}(t)` is the network input (see below).

    The stationary distribution variance (without external input) is
    :math:`\sigma^2/(2\lambda)` for :math:`\lambda > 0`; for :math:`\lambda=0`,
    the model is non-stationary.

    **2. Network Input Structure**

    The network input :math:`I_\mathrm{net}(t)` decomposes into excitatory and
    inhibitory branches:

    .. math::

       I_\mathrm{net}(t) = H_\mathrm{ex}(X) \cdot g(I_\mathrm{ex}(t))
                         + H_\mathrm{in}(X) \cdot g(I_\mathrm{in}(t)),

    where:

    - :math:`I_\mathrm{ex}(t)`, :math:`I_\mathrm{in}(t)` are synaptic input
      branches (sign-separated by event weight).
    - :math:`g(\cdot)` is the input nonlinearity. Default: :math:`g(h)=g\,h`
      (linear gain).
    - :math:`H_\mathrm{ex}(X)`, :math:`H_\mathrm{in}(X)` are optional
      multiplicative coupling factors (rate-dependent synaptic efficacy).
      Default: :math:`H_\mathrm{ex}(X)=g_\mathrm{ex}(\theta_\mathrm{ex}-X)`,
      :math:`H_\mathrm{in}(X)=g_\mathrm{in}(\theta_\mathrm{in}+X)`.
      Only active if ``mult_coupling=True``.

    The ``linear_summation`` switch controls nonlinearity application:

    - ``linear_summation=True``:
      :math:`I_\mathrm{net}(t) = H\cdot g(I_\mathrm{ex}+I_\mathrm{in})`.
    - ``linear_summation=False``:
      :math:`I_\mathrm{net}(t) = H_\mathrm{ex}\cdot g(I_\mathrm{ex})
      + H_\mathrm{in}\cdot g(I_\mathrm{in})`.

    **3. Discrete-Time Integration (Stochastic Exponential Euler)**

    For time step :math:`h=dt` (in ms), the model uses an exact Ornstein-Uhlenbeck
    integration scheme for the linear part, with Euler-Maruyama for the forcing:

    .. math::

       X_{n+1} = P_1 X_n + P_2 (\mu + I_\mathrm{net,n}) + N\,\xi_n,

    where :math:`\xi_n\sim\mathcal{N}(0,1)` is standard Gaussian noise.

    **For** :math:`\lambda > 0`:

    .. math::

       P_1 = \exp\left(-\frac{\lambda h}{\tau}\right), \quad
       P_2 = \frac{1-P_1}{\lambda}, \quad
       N = \sigma\sqrt{\frac{1-P_1^2}{2\lambda}}.

    **For** :math:`\lambda = 0` (Euler-Maruyama):

    .. math::

       P_1=1, \quad P_2=\frac{h}{\tau}, \quad N=\sigma\sqrt{\frac{h}{\tau}}.

    The noise factor :math:`N` is derived from exact OU process integration over
    :math:`[0, h]`, ensuring correct fluctuation amplitude as :math:`h\to 0`.

    **4. Update Ordering (Matching NEST ``rate_neuron_ipn_impl.h``)**

    Per simulation step:

    1. **Store outgoing delayed value**: current ``rate`` is recorded as
       ``delayed_rate``.
    2. **Draw noise**: sample :math:`\xi_n\sim\mathcal{N}(0,1)`, compute
       :math:`\mathrm{noise}_n=\sigma\,\xi_n`.
    3. **Propagate intrinsic dynamics**: apply stochastic exponential Euler to
       :math:`X_n` with external drive and noise.
    4. **Read event buffers**: drain delayed events arriving at current step;
       accumulate instantaneous events.
    5. **Apply network input**: according to ``linear_summation`` and
       ``mult_coupling`` settings.

       - ``linear_summation=True``: nonlinearity applied to summed branch input
         during update.
       - ``linear_summation=False``: nonlinearity applied per event during
         buffering (matching NEST event handlers).

    6. **Rectification** (optional): if ``rectify_output=True``, clamp
       :math:`X_{n+1}\gets\max(X_{n+1},\,\mathrm{rectify\_rate})`.
    7. **Update state variables**: ``rate``, ``noise``, ``delayed_rate``,
       ``instant_rate``, ``_step_count``.

    **5. Numerical Stability and Computational Complexity**

    - Construction enforces :math:`\tau>0`, :math:`\lambda\ge 0`,
      :math:`\sigma\ge 0`, :math:`\mathrm{rectify\_rate}\ge 0`.
    - The exponential Euler scheme is numerically stable for all :math:`h>0`.
    - Stochastic dynamics may violate deterministic stability bounds; use
      ``rectify_output=True`` to enforce rate constraints.
    - Per-call cost is :math:`O(\prod\mathrm{varshape})` with vectorized NumPy
      operations in float64 for coefficient evaluation and state updates.

    Parameters
    ----------
    in_size : Size
        Population shape (tuple or int). All per-neuron parameters are broadcast
        to ``self.varshape``.
    tau : ArrayLike, optional
        Time constant :math:`\tau` (ms). Scalar or array broadcastable to
        ``self.varshape``. Must be :math:`>0`. Default: ``10.0 * u.ms``.
    lambda_ : ArrayLike, optional
        Passive decay rate :math:`\lambda` (dimensionless). Scalar or array
        broadcastable to ``self.varshape``. Must be :math:`\ge 0`. Controls
        exponential relaxation (:math:`\lambda=0` yields driftless diffusion).
        Default: ``1.0``.
    sigma : ArrayLike, optional
        Input-noise scale :math:`\sigma` (dimensionless). Scalar or array
        broadcastable to ``self.varshape``. Must be :math:`\ge 0`. Default:
        ``1.0``.
    mu : ArrayLike, optional
        Mean drive :math:`\mu` (dimensionless). Scalar or array broadcastable to
        ``self.varshape``. External constant input to the rate dynamics. Default:
        ``0.0``.
    g : ArrayLike, optional
        Linear gain parameter :math:`g` (dimensionless) used by the default
        input nonlinearity :math:`g(h)=g\,h`. Scalar or array broadcastable to
        ``self.varshape``. Default: ``1.0``.
    mult_coupling : bool, optional
        Enable multiplicative coupling (rate-dependent synaptic efficacy). If
        ``True``, applies :math:`H_\mathrm{ex}(X)` and :math:`H_\mathrm{in}(X)`
        to synaptic inputs. Default: ``False``.
    g_ex : ArrayLike, optional
        Excitatory multiplicative coupling gain :math:`g_\mathrm{ex}`
        (dimensionless). Scalar or array broadcastable to ``self.varshape``.
        Only used if ``mult_coupling=True``. Default: ``1.0``.
    g_in : ArrayLike, optional
        Inhibitory multiplicative coupling gain :math:`g_\mathrm{in}`
        (dimensionless). Scalar or array broadcastable to ``self.varshape``.
        Only used if ``mult_coupling=True``. Default: ``1.0``.
    theta_ex : ArrayLike, optional
        Excitatory coupling reference rate :math:`\theta_\mathrm{ex}`
        (dimensionless). Scalar or array broadcastable to ``self.varshape``.
        Only used if ``mult_coupling=True``. Default: ``0.0``.
    theta_in : ArrayLike, optional
        Inhibitory coupling reference rate :math:`\theta_\mathrm{in}`
        (dimensionless). Scalar or array broadcastable to ``self.varshape``.
        Only used if ``mult_coupling=True``. Default: ``0.0``.
    linear_summation : bool, optional
        Controls where the input nonlinearity is applied. If ``True``, the
        nonlinearity is applied to the sum of excitatory and inhibitory inputs.
        If ``False``, the nonlinearity is applied separately to each input branch
        (matching NEST event semantics). Default: ``True``.
    rectify_rate : ArrayLike, optional
        Lower bound :math:`X_\mathrm{min}` for the rate when
        ``rectify_output=True`` (dimensionless). Scalar or array broadcastable to
        ``self.varshape``. Must be :math:`\ge 0`. Default: ``0.0``.
    rectify_output : bool, optional
        If ``True``, clamp the rate output to
        :math:`X\ge\mathrm{rectify\_rate}` after each update step. Default:
        ``False``.
    input_nonlinearity : Callable or None, optional
        Custom input nonlinearity :math:`g(\cdot)` replacing the default
        :math:`g(h)=g\,h`. Callable signature: ``f(h)`` (receives NumPy array) or
        ``f(model, h)`` (receives model instance and array). Must return array of
        same shape as input. If ``None``, uses default linear gain. Default:
        ``None``.
    mult_coupling_ex_fn : Callable or None, optional
        Custom excitatory multiplicative coupling function
        :math:`H_\mathrm{ex}(X)`. Callable signature: ``f(rate)`` or
        ``f(model, rate)``. Must return array of same shape as input. If ``None``,
        uses default :math:`g_\mathrm{ex}(\theta_\mathrm{ex}-X)`. Default:
        ``None``.
    mult_coupling_in_fn : Callable or None, optional
        Custom inhibitory multiplicative coupling function
        :math:`H_\mathrm{in}(X)`. Callable signature: ``f(rate)`` or
        ``f(model, rate)``. Must return array of same shape as input. If ``None``,
        uses default :math:`g_\mathrm{in}(\theta_\mathrm{in}+X)`. Default:
        ``None``.
    rate_initializer : Callable, optional
        Initializer for the ``rate`` state variable :math:`X_0`. Callable
        compatible with ``braintools.init`` API. Default:
        ``braintools.init.Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for the ``noise`` state variable (records last noise sample
        :math:`\sigma\,\xi_{n-1}`). Callable compatible with ``braintools.init``
        API. Default: ``braintools.init.Constant(0.0)``.
    name : str or None, optional
        Module name for identification in hierarchies. If ``None``, an
        auto-generated name is used. Default: ``None``.

    Parameter Mapping
    -----------------

    The following table maps NEST ``rate_neuron_ipn`` / ``lin_rate_ipn``
    parameters to brainpy.state equivalents:

    =============================== ===================== =========
    NEST Parameter                  brainpy.state         Default
    =============================== ===================== =========
    ``tau``                         ``tau``               10 ms
    ``lambda``                      ``lambda_``           1.0
    ``sigma``                       ``sigma``             1.0
    ``mu``                          ``mu``                0.0
    ``g`` (nonlinearity gain)       ``g``                 1.0
    ``mult_coupling``               ``mult_coupling``     False
    ``g_ex``, ``g_in``              ``g_ex``, ``g_in``    1.0
    ``theta_ex``, ``theta_in``      ``theta_ex``,         0.0
                                    ``theta_in``
    ``linear_summation``            ``linear_summation``  True
    ``rectify_rate``                ``rectify_rate``      0.0
    ``rectify_output``              ``rectify_output``    False
    =============================== ===================== =========

    Attributes
    ----------
    rate : brainstate.ShortTermState
        Current rate state :math:`X_n` (float64 array of shape ``self.varshape``
        or ``(batch_size,) + self.varshape``).
    noise : brainstate.ShortTermState
        Last noise sample :math:`\sigma\,\xi_{n-1}` (float64 array, same shape as
        ``rate``).
    instant_rate : brainstate.ShortTermState
        Rate value after instantaneous event application (float64 array, same
        shape as ``rate``).
    delayed_rate : brainstate.ShortTermState
        Rate value before current update, used for delayed projections (float64
        array, same shape as ``rate``).
    _step_count : brainstate.ShortTermState
        Internal step counter for delayed event scheduling (int64 scalar).
    _delayed_ex_queue : dict
        Internal queue mapping ``step_idx`` to accumulated excitatory delayed
        events.
    _delayed_in_queue : dict
        Internal queue mapping ``step_idx`` to accumulated inhibitory delayed
        events.

    Raises
    ------
    ValueError
        If ``tau <= 0``, ``lambda_ < 0``, ``sigma < 0``, or
        ``rectify_rate < 0``.
    ValueError
        If ``instant_rate_events`` contain non-zero ``delay_steps``.
    ValueError
        If ``delayed_rate_events`` contain negative ``delay_steps``.
    ValueError
        If event tuples have length other than 2, 3, or 4.

    Notes
    -----
    **Runtime Event Semantics**

    - ``instant_rate_events``: Applied in the current step without delay. Each
      event can be:

      - A scalar (treated as ``rate`` value with ``weight=1.0``).
      - A tuple ``(rate, weight)`` or ``(rate, weight, delay_steps)`` or
        ``(rate, weight, delay_steps, multiplicity)``.
      - A dict with keys ``'rate'``/``'coeff'``/``'value'``, ``'weight'``,
        ``'delay_steps'``/``'delay'``, ``'multiplicity'``.

    - ``delayed_rate_events``: Scheduled with integer ``delay_steps`` (units of
      simulation time step). Same format as ``instant_rate_events``.

    - Sign convention: events with ``weight >= 0`` contribute to the excitatory
      branch; events with ``weight < 0`` contribute to the inhibitory branch.

    - For ``linear_summation=False``, event values are transformed by the input
      nonlinearity during buffering (matching NEST event handlers).

    **Comparison to Output-Noise Variant**

    The ``rate_neuron_opn`` model uses output noise (applied after nonlinearity),
    while ``rate_neuron_ipn`` uses input noise (applied before dynamics
    propagation). This leads to different stationary distributions and noise
    scaling behavior. Input noise typically results in stronger fluctuations at
    high rates.

    **Failure Modes**


    - No automatic failure handling. Negative time constants, decay rates, or
      noise parameters are caught at construction by ``_validate_parameters``.
    - Invalid event formats raise ``ValueError`` during update.
    - Numerical instability is unlikely due to exact OU integration, but
      extreme parameter combinations (very large :math:`\sigma`, very small
      :math:`\tau`) may lead to rate explosions without ``rectify_output=True``.

    Examples
    --------
    **Example 1**: Minimal stochastic rate neuron with external drive.

    .. code-block:: python

       >>> from brainpy import state as bst
       >>> import saiunit as u
       >>> model = bst.rate_neuron_ipn(in_size=10, tau=20*u.ms, sigma=0.5)
       >>> model.init_all_states(batch_size=1)
       >>> rate = model(x=0.1)  # external drive
       >>> print(rate.shape)
       (1, 10)

    **Example 2**: Multiplicative coupling with custom nonlinearity.

    .. code-block:: python

       >>> import numpy as np
       >>> def tanh_nonlin(h):
       ...     return np.tanh(h)
       >>> model = bst.rate_neuron_ipn(
       ...     in_size=5,
       ...     tau=10*u.ms,
       ...     lambda_=2.0,
       ...     mult_coupling=True,
       ...     g_ex=1.5, theta_ex=1.0,
       ...     input_nonlinearity=tanh_nonlin
       ... )
       >>> model.init_all_states()

    **Example 3**: Update with instantaneous and delayed events.

    .. code-block:: python

       >>> model = bst.rate_neuron_ipn(in_size=3, tau=10*u.ms, sigma=0.1)
       >>> model.init_all_states()
       >>> instant_event = {'rate': 1.0, 'weight': 0.1}
       >>> delayed_event = {'rate': 0.5, 'weight': -0.05, 'delay_steps': 3}
       >>> rate = model.update(
       ...     x=0.2,
       ...     instant_rate_events=instant_event,
       ...     delayed_rate_events=delayed_event
       ... )

    References
    ----------
    .. [1] NEST Simulator Documentation: ``rate_neuron_ipn``
           https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_ipn.html
    .. [2] Hahne, J., Dahmen, D., Schuecker, J., Frommer, A., Bolten, M.,
           Helias, M., & Diesmann, M. (2017). Integration of continuous-time
           dynamics in a spiking neural network simulator.
           *Frontiers in Neuroinformatics*, 11, 34.
           https://doi.org/10.3389/fninf.2017.00034

    See Also
    --------
    rate_neuron_opn : Output-noise variant of the rate neuron template.
    lin_rate : Deterministic linear rate neuron (``sigma=0``).
    """

    __module__ = 'brainpy.state'

    #: The rate-neuron template carries genuine ``(H_ex, H_in)`` factors, so
    #: ``mult_coupling`` splits the deposit into the ``'rate_ex'``/``'rate_in'``
    #: channels (spec §3.2).
    _supports_mult_coupling = True

    @property
    def _phi_signature(self):
        """Extend the base φ identity with the user ``input_nonlinearity`` callable.

        The template's φ is the user-supplied ``input_nonlinearity`` (or the linear
        gain ``g·h`` when ``None``); two templates share a φ only when they reference
        the *same* callable object — functions are compared by identity, since two
        arbitrary callables cannot be proven equal.
        """
        return super()._phi_signature + (('input_nonlinearity', self.input_nonlinearity),)

    def __init__(
        self,
        in_size: Size,
        tau: ArrayLike = 10.0 * u.ms,
        lambda_: ArrayLike = 1.0,
        sigma: ArrayLike = 1.0,
        mu: ArrayLike = 0.0,
        g: ArrayLike = 1.0,
        mult_coupling: bool = False,
        g_ex: ArrayLike = 1.0,
        g_in: ArrayLike = 1.0,
        theta_ex: ArrayLike = 0.0,
        theta_in: ArrayLike = 0.0,
        linear_summation: bool = True,
        rectify_rate: ArrayLike = 0.0,
        rectify_output: bool = False,
        input_nonlinearity: Callable | None = None,
        mult_coupling_ex_fn: Callable | None = None,
        mult_coupling_in_fn: Callable | None = None,
        rate_initializer: Callable = braintools.init.Constant(0.0),
        noise_initializer: Callable = braintools.init.Constant(0.0),
        name: str = None,
    ):
        super().__init__(
            in_size=in_size,
            tau=tau,
            sigma=sigma,
            mu=mu,
            g=g,
            mult_coupling=mult_coupling,
            g_ex=g_ex,
            g_in=g_in,
            theta_ex=theta_ex,
            theta_in=theta_in,
            linear_summation=linear_summation,
            rate_initializer=rate_initializer,
            noise_initializer=noise_initializer,
            name=name,
        )
        self.lambda_ = braintools.init.param(lambda_, self.varshape)
        self.rectify_rate = braintools.init.param(rectify_rate, self.varshape)
        self.rectify_output = bool(rectify_output)

        self.input_nonlinearity = input_nonlinearity
        self.mult_coupling_ex_fn = mult_coupling_ex_fn
        self.mult_coupling_in_fn = mult_coupling_in_fn

        self._validate_parameters()

    @property
    def recordables(self):
        r"""List of state variable names that can be recorded during simulation.

        Returns
        -------
        list of str
            ``['rate', 'noise']``. The ``rate`` variable records the current rate
            state :math:`X_n`, and ``noise`` records the last noise sample
            :math:`\sigma\,\xi_{n-1}`.

        Notes
        -----
        These variables can be accessed via recording tools in BrainPy for
        post-simulation analysis of rate dynamics and noise contributions.
        """
        return ['rate', 'noise']

    @property
    def receptor_types(self):
        r"""Receptor type dictionary for projection compatibility.

        Returns
        -------
        dict[str, int]
            ``{'RATE': 0}``. Rate neurons have a single unified receptor port
            for all rate-based inputs. Excitatory vs. inhibitory separation is
            handled internally via event weight signs.

        Notes
        -----
        This property is used by projection objects to validate connection targets.
        Unlike spiking neurons with separate AMPA/GABA receptor ports, rate neurons
        use sign-based branch routing (``weight >= 0`` → excitatory branch,
        ``weight < 0`` → inhibitory branch).
        """
        return {'RATE': 0}

    def _validate_parameters(self):
        r"""Validate model parameters at construction time.

        Raises
        ------
        ValueError
            If ``tau <= 0``, ``lambda_ < 0``, ``sigma < 0``, or
            ``rectify_rate < 0``.

        Notes
        -----
        This method is called automatically during ``__init__``.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.tau, self.sigma)):
            return
        if np.any(self.tau <= 0.0 * u.ms):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self.lambda_ < 0.0):
            raise ValueError('Passive decay rate lambda must be >= 0.')
        if np.any(self.sigma < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')
        if np.any(self.rectify_rate < 0.0):
            raise ValueError('Rectifying rate must be >= 0.')

    def _call_nl(self, fn: Callable, x: np.ndarray):
        r"""Call user-provided nonlinearity with flexible signature.

        Parameters
        ----------
        fn : Callable
            User-provided function with signature ``f(x)`` or ``f(model, x)``.
        x : np.ndarray
            Input array (float64).

        Returns
        -------
        np.ndarray
            Output of ``fn``, coerced to float64 NumPy array.

        Notes
        -----
        Tries ``fn(self, x)`` first (passing model instance), then falls back
        to ``fn(x)`` if signature mismatch occurs.
        """
        try:
            return fn(self, x)
        except TypeError as first_error:
            try:
                return fn(x)
            except TypeError:
                raise first_error

    def _activation(self, h):
        r"""Input nonlinearity :math:`g(h)` (JAX; reads ``self``).

        Uses the user-supplied ``input_nonlinearity`` when provided (invoked as
        ``fn(self, h)`` then ``fn(h)``), otherwise the default linear gain
        :math:`g(h)=g\,h`. Must be JAX-expressible so the step lowers under
        ``brainstate.transform.for_loop`` / ``jit``.
        """
        if self.input_nonlinearity is None:
            return u.get_mantissa(self.g) * h
        return self._call_nl(self.input_nonlinearity, h)

    def _mult_factors(self, rate):
        r"""Multiplicative coupling factors :math:`(H_\mathrm{ex}, H_\mathrm{in})` (JAX).

        Defaults to :math:`H_\mathrm{ex}=g_\mathrm{ex}(\theta_\mathrm{ex}-X)` and
        :math:`H_\mathrm{in}=g_\mathrm{in}(\theta_\mathrm{in}+X)`; the user callables
        ``mult_coupling_ex_fn`` / ``mult_coupling_in_fn`` override each branch
        independently.
        """
        if self.mult_coupling_ex_fn is None:
            H_ex = u.get_mantissa(self.g_ex) * (u.get_mantissa(self.theta_ex) - rate)
        else:
            H_ex = self._call_nl(self.mult_coupling_ex_fn, rate)
        if self.mult_coupling_in_fn is None:
            H_in = u.get_mantissa(self.g_in) * (u.get_mantissa(self.theta_in) + rate)
        else:
            H_in = self._call_nl(self.mult_coupling_in_fn, rate)
        return H_ex, H_in

    def init_state(self, **kwargs):
        r"""Initialize all state variables for simulation.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        This method initializes:

        - ``rate``: Current rate state :math:`X_n`.
        - ``noise``: Last noise sample :math:`\sigma\,\xi_{n-1}`.
        - ``instant_rate``: Rate after instantaneous event application.
        - ``delayed_rate``: Rate before current update (for delayed projections).
        - ``_step_count``: Internal step counter for delay scheduling.
        - ``_delayed_ex_queue``, ``_delayed_in_queue``: Delay queues.

        All state arrays are initialized as float64 NumPy arrays using the
        provided initializers.
        """
        rate = braintools.init.param(self.rate_initializer, self.varshape)
        noise = braintools.init.param(self.noise_initializer, self.varshape)
        rate_np = self._to_numpy(rate)
        noise_np = self._to_numpy(noise)

        self.rate = brainstate.ShortTermState(rate_np)
        self.noise = brainstate.ShortTermState(noise_np)
        dftype = brainstate.environ.dftype()
        self.instant_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))
        self._alloc_phi_rate(rate_np)

    def update(self, x=0.0, noise=None):
        r"""Advance the input-noise rate dynamics by one step.

        Network coupling arrives continuously through the substrate's delta
        channel (seam-(H)): :math:`h=\sum_\mathrm{delta} w\,r_\mathrm{pre}` is read
        from ``sum_delta_inputs(0.0)`` and the external drive from
        ``sum_current_inputs(x, rate)``. The whole step is JAX-expressible so it
        lowers under ``brainstate.transform.for_loop`` / ``jit``.

        Parameters
        ----------
        x : ArrayLike, optional
            External drive added to ``mu`` (broadcast to ``self.varshape``).
        noise : ArrayLike, optional
            Externally supplied :math:`\xi_n`; drawn from :math:`\mathcal{N}(0,1)`
            when ``None``.

        Returns
        -------
        rate_new : ArrayLike
            Updated rate :math:`X_{n+1}` (shape ``self.rate.value.shape``).
        """
        h = float(u.get_mantissa(brainstate.environ.get_dt() / u.ms))
        dftype = brainstate.environ.dftype()
        state_shape = self.rate.value.shape

        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        sigma = self._broadcast_to_state(self._to_numpy(self.sigma), state_shape)
        mu = self._broadcast_to_state(self._to_numpy(self.mu), state_shape)
        lambda_ = self._broadcast_to_state(self._to_numpy(self.lambda_), state_shape)
        rectify_rate = self._broadcast_to_state(self._to_numpy(self.rectify_rate), state_shape)

        rate_prev = jnp.broadcast_to(jnp.asarray(self.rate.value, dtype=dftype), state_shape)

        mu_ext, h_a, h_b = self._read_coupling(x)

        if noise is None:
            xi = brainstate.random.randn(*state_shape)
        else:
            xi = jnp.broadcast_to(jnp.asarray(noise, dtype=dftype), state_shape)
        noise_now = sigma * xi

        if np.any(lambda_ > 0.0):
            P1 = np.exp(-lambda_ * h / tau)
            P2 = -np.expm1(-lambda_ * h / tau) / np.where(lambda_ == 0.0, 1.0, lambda_)
            input_noise_factor = np.sqrt(
                -0.5 * np.expm1(-2.0 * lambda_ * h / tau) / np.where(lambda_ == 0.0, 1.0, lambda_)
            )
            zero_lambda = lambda_ == 0.0
            if np.any(zero_lambda):
                P1 = np.where(zero_lambda, 1.0, P1)
                P2 = np.where(zero_lambda, h / tau, P2)
                input_noise_factor = np.where(zero_lambda, np.sqrt(h / tau), input_noise_factor)
        else:
            P1 = np.ones_like(lambda_)
            P2 = h / tau
            input_noise_factor = np.sqrt(h / tau)

        mu_total = mu + mu_ext
        rate_new = P1 * rate_prev + P2 * mu_total + input_noise_factor * noise_now
        rate_new = rate_new + P2 * self._coupling_increment(rate_prev, h_a, h_b)

        if self.rectify_output:
            rate_new = jnp.where(rate_new < rectify_rate, rectify_rate, rate_new)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.delayed_rate.value = rate_prev
        self.instant_rate.value = rate_new
        self._store_phi_rate(rate_new)
        return rate_new
