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
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from brainpy_state._nest.lin_rate import _lin_rate_base
from ._utils import is_tracer

__all__ = [
    'rate_neuron_opn',
]


class rate_neuron_opn(_lin_rate_base):
    r"""NEST-compatible ``rate_neuron_opn`` output-noise rate-neuron template.

    ``rate_neuron_opn`` implements the NEST template model
    ``rate_neuron_opn<TNonlinearities>`` with the deterministic dynamics

    .. math::

       \tau \frac{dX(t)}{dt}
       = -X(t) + \mu + I_\mathrm{net}(t),

    and output noise applied after the nonlinearity:

    .. math::

       X_\mathrm{noisy}(t)
       = X(t) + \sqrt{\frac{\tau}{h}}\,\sigma\,\xi(t),

    where :math:`X(t)` is the deterministic rate state, :math:`\tau` is the
    time constant, :math:`\mu` is the mean drive, :math:`\sigma\ge 0` is the
    output-noise strength, :math:`h` is the simulation time step, and
    :math:`\xi(t)\sim\mathcal{N}(0,1)` is standard Gaussian white noise
    approximated as piecewise constant over :math:`h`.

    With default callables this is equivalent to NEST ``lin_rate_opn``:

    - ``input(h) = g * h``
    - ``mult_coupling_ex(rate) = g_ex * (theta_ex - rate)``
    - ``mult_coupling_in(rate) = g_in * (theta_in + rate)``

    Mathematical Description
    ------------------------

    **1. Continuous-Time Deterministic Dynamics**

    The deterministic rate state :math:`X(t)` evolves according to

    .. math::

       \tau \frac{dX(t)}{dt} = -X(t) + \mu + I_\mathrm{net}(t),

    where :math:`\tau>0` is the time constant and :math:`I_\mathrm{net}(t)` is
    the network input decomposed as

    .. math::

       I_\mathrm{net}(t) = H_\mathrm{ex}(X_\mathrm{noisy}) \cdot g(I_\mathrm{ex}(t))
                         + H_\mathrm{in}(X_\mathrm{noisy}) \cdot g(I_\mathrm{in}(t)),

    where:

    - :math:`I_\mathrm{ex}(t)` and :math:`I_\mathrm{in}(t)` are excitatory and
      inhibitory synaptic input branches.
    - :math:`g(\cdot)` is the input nonlinearity. Default: :math:`g(h)=g\,h`.
    - :math:`H_\mathrm{ex}(X_\mathrm{noisy})` and
      :math:`H_\mathrm{in}(X_\mathrm{noisy})` are optional multiplicative
      coupling factors dependent on the *noisy* rate. Default:
      :math:`H_\mathrm{ex}=g_\mathrm{ex}(\theta_\mathrm{ex}-X_\mathrm{noisy})`,
      :math:`H_\mathrm{in}=g_\mathrm{in}(\theta_\mathrm{in}+X_\mathrm{noisy})`.
      Only active if ``mult_coupling=True``.

    The ``linear_summation`` switch controls whether the nonlinearity is
    applied to the summed input or to individual synaptic branches:

    - ``linear_summation=True``:
      :math:`I_\mathrm{net}(t) = H\cdot g(I_\mathrm{ex}+I_\mathrm{in})`.
    - ``linear_summation=False``:
      :math:`I_\mathrm{net}(t) = H_\mathrm{ex}\cdot g(I_\mathrm{ex})
      + H_\mathrm{in}\cdot g(I_\mathrm{in})`.

    **2. Output Noise (Postsynaptic Noise Model)**

    Output noise is added *after* the deterministic dynamics, creating a noisy
    observation of the rate:

    .. math::

       X_\mathrm{noisy}(t) = X(t) + \sqrt{\frac{\tau}{h}}\,\sigma\,\xi(t),

    where :math:`\xi(t)\sim\mathcal{N}(0,1)` is standard Gaussian white noise.
    The scaling factor :math:`\sqrt{\tau/h}` ensures that the noise amplitude
    is independent of the discretization time step :math:`h` in the limit
    :math:`h\to 0`.

    **Critical difference from input-noise model**: The noisy rate
    :math:`X_\mathrm{noisy}` is used for *multiplicative coupling* evaluation
    (if ``mult_coupling=True``) and as the *outgoing signal* to downstream
    neurons, but the noise does *not* feed back into the deterministic
    dynamics. This contrasts with the input-noise variant (``rate_neuron_ipn``)
    where noise enters the differential equation directly.

    **3. Discrete-Time Integration**

    For time step :math:`h=dt` (in ms), the deterministic part uses exponential
    Euler integration (exact for the linear ODE):

    .. math::

       X_{n+1} = P_1 X_n + P_2 (\mu + I_\mathrm{net,n}),

    where

    .. math::

       P_1 = \exp(-h/\tau), \quad P_2 = 1 - P_1 = -\mathrm{expm1}(-h/\tau).

    Output noise is added independently at each step:

    .. math::

       X_\mathrm{noisy,n} = X_n + \sqrt{\frac{\tau}{h}}\,\sigma\,\xi_n,

    where :math:`\xi_n\sim\mathcal{N}(0,1)` is drawn at each step.

    **4. Update Ordering (Matching NEST ``rate_neuron_opn_impl.h``)**

    Per simulation step:

    1. Draw noise sample :math:`\xi_n\sim\mathcal{N}(0,1)`, compute
       :math:`\mathrm{noise}_n = \sigma\,\xi_n`.
    2. Compute noisy rate:
       :math:`X_\mathrm{noisy,n} = X_n + \sqrt{\tau/h}\,\mathrm{noise}_n`.
    3. Propagate deterministic intrinsic dynamics:
       :math:`X' = P_1 X_n + P_2 (\mu + \mu_\mathrm{ext})`.
    4. Read delayed and instantaneous event buffers.
    5. Apply network input according to NEST semantics:

       - ``linear_summation=True``: nonlinearity applied to summed branch input
         during update.
       - ``linear_summation=False``: nonlinearity applied per incoming event
         while buffering (handled in event processing).

    6. If ``mult_coupling=True``, multiplicative coupling factors
       :math:`H_\mathrm{ex}(X_\mathrm{noisy,n})` and
       :math:`H_\mathrm{in}(X_\mathrm{noisy,n})` are evaluated at the *noisy*
       rate (matching NEST ``rate_neuron_opn_impl.h``).
    7. Store updated ``rate``, ``noise``, and expose ``noisy_rate`` as
       outgoing delayed/instantaneous event value.

    **5. Stability Constraints and Computational Implications**

    - Construction enforces :math:`\tau>0`, :math:`\sigma\ge 0`.
    - The deterministic dynamics are unconditionally stable (exponential
      relaxation to :math:`\mu + I_\mathrm{net}` with time constant :math:`\tau`).
    - Output noise does not affect stability but may violate rate bounds; no
      automatic rectification is provided (unlike ``rate_neuron_ipn``).
    - Noise variance scales as :math:`\tau\sigma^2/h` per step. For fixed
      :math:`\tau` and :math:`\sigma`, this diverges as :math:`h\to 0`,
      reflecting the white-noise nature of :math:`\xi(t)`.
    - The exponential Euler scheme is numerically stable for all :math:`h>0`.
    - Per-call cost is :math:`O(\prod\mathrm{varshape})` with vectorized NumPy
      operations in ``float64`` for coefficient evaluation and state update.

    Parameters
    ----------
    in_size : Size
        Population shape specification (tuple of int or single int). All
        per-neuron parameters are broadcast to ``self.varshape``. For example,
        ``in_size=10`` creates 10 neurons, ``in_size=(4, 5)`` creates a 4×5
        grid.
    tau : ArrayLike, optional
        Time constant :math:`\tau` (brainunit quantity with ms dimension).
        Scalar or array broadcastable to ``self.varshape``. Must be :math:`>0`.
        Controls the exponential relaxation rate of the deterministic dynamics.
        Default: ``10.0 * u.ms``.
    sigma : ArrayLike, optional
        Output-noise scale :math:`\sigma` (dimensionless scalar or array).
        Broadcastable to ``self.varshape``. Must be :math:`\ge 0`. Determines
        the standard deviation of the Gaussian noise added to the output rate.
        Default: ``1.0``.
    mu : ArrayLike, optional
        Mean drive :math:`\mu` (dimensionless scalar or array). Broadcastable
        to ``self.varshape``. External constant input to the rate dynamics,
        added to the network input. Default: ``0.0``.
    g : ArrayLike, optional
        Linear gain parameter :math:`g` (dimensionless scalar or array).
        Broadcastable to ``self.varshape``. Used by the default input
        nonlinearity :math:`g(h)=g\,h`. Ignored if ``input_nonlinearity`` is
        provided. Default: ``1.0``.
    mult_coupling : bool, optional
        Enable multiplicative coupling (rate-dependent synaptic efficacy). If
        ``True``, applies :math:`H_\mathrm{ex}(X_\mathrm{noisy})` and
        :math:`H_\mathrm{in}(X_\mathrm{noisy})` to synaptic inputs, evaluated
        at the *noisy* rate. If ``False``, :math:`H_\mathrm{ex}=H_\mathrm{in}=1`.
        Default: ``False``.
    g_ex : ArrayLike, optional
        Excitatory multiplicative coupling gain :math:`g_\mathrm{ex}`
        (dimensionless scalar or array). Broadcastable to ``self.varshape``.
        Only used if ``mult_coupling=True``. Default: ``1.0``.
    g_in : ArrayLike, optional
        Inhibitory multiplicative coupling gain :math:`g_\mathrm{in}`
        (dimensionless scalar or array). Broadcastable to ``self.varshape``.
        Only used if ``mult_coupling=True``. Default: ``1.0``.
    theta_ex : ArrayLike, optional
        Excitatory coupling reference rate :math:`\theta_\mathrm{ex}`
        (dimensionless scalar or array). Broadcastable to ``self.varshape``.
        Only used if ``mult_coupling=True``. Default: ``0.0``.
    theta_in : ArrayLike, optional
        Inhibitory coupling reference rate :math:`\theta_\mathrm{in}`
        (dimensionless scalar or array). Broadcastable to ``self.varshape``.
        Only used if ``mult_coupling=True``. Default: ``0.0``.
    linear_summation : bool, optional
        NEST switch controlling where the input nonlinearity is applied. If
        ``True``, the nonlinearity is applied to the sum of excitatory and
        inhibitory inputs (post-summation). If ``False``, the nonlinearity is
        applied separately to each input branch before summation (per-branch).
        Default: ``True``.
    input_nonlinearity : Callable[[ArrayLike], ArrayLike] or Callable[[rate_neuron_opn, ArrayLike], ArrayLike] or None, optional
        Custom input nonlinearity :math:`g(\cdot)` replacing the default
        :math:`g(h)=g\,h`. Callable signature can be ``f(h)`` (receives float64
        NumPy array of shape ``state_shape``, returns array of same shape) or
        ``f(model, h)`` (receives model instance and array, returns array).
        Must be vectorized and compatible with NumPy broadcasting. If ``None``,
        uses default linear gain. Default: ``None``.
    mult_coupling_ex_fn : Callable[[ArrayLike], ArrayLike] or Callable[[rate_neuron_opn, ArrayLike], ArrayLike] or None, optional
        Custom excitatory multiplicative coupling function
        :math:`H_\mathrm{ex}(X_\mathrm{noisy})`. Callable signature can be
        ``f(rate)`` or ``f(model, rate)``. Must return array of same shape as
        input. Evaluated at the *noisy* rate. If ``None``, uses default
        :math:`g_\mathrm{ex}(\theta_\mathrm{ex}-X_\mathrm{noisy})`. Default:
        ``None``.
    mult_coupling_in_fn : Callable[[ArrayLike], ArrayLike] or Callable[[rate_neuron_opn, ArrayLike], ArrayLike] or None, optional
        Custom inhibitory multiplicative coupling function
        :math:`H_\mathrm{in}(X_\mathrm{noisy})`. Callable signature can be
        ``f(rate)`` or ``f(model, rate)``. Must return array of same shape as
        input. Evaluated at the *noisy* rate. If ``None``, uses default
        :math:`g_\mathrm{in}(\theta_\mathrm{in}+X_\mathrm{noisy})`. Default:
        ``None``.
    rate_initializer : Callable, optional
        Initializer for the deterministic ``rate`` state variable :math:`X_0`.
        Callable compatible with ``braintools.init`` API (signature:
        ``(shape, batch_size) -> ArrayLike``). Default:
        ``braintools.init.Constant(0.0)``.
    noise_initializer : Callable, optional
        Initializer for the ``noise`` state variable (records last noise sample
        :math:`\sigma\,\xi_{n-1}`). Callable compatible with ``braintools.init``
        API. Default: ``braintools.init.Constant(0.0)``.
    noisy_rate_initializer : Callable, optional
        Initializer for the ``noisy_rate`` state variable
        :math:`X_\mathrm{noisy,0}` and outgoing event values. Callable
        compatible with ``braintools.init`` API. Default:
        ``braintools.init.Constant(0.0)``.
    name : str or None, optional
        Module name for identification in hierarchies. If ``None``,
        auto-generates a unique name. Default: ``None``.

    Parameter Mapping
    -----------------

    The following table maps NEST ``rate_neuron_opn`` / ``lin_rate_opn``
    parameters to brainpy.state equivalents:

    =============================== ========================== ===========
    NEST Parameter                  brainpy.state              Default
    =============================== ========================== ===========
    ``tau``                         ``tau``                    10 ms
    ``sigma``                       ``sigma``                  1.0
    ``mu``                          ``mu``                     0.0
    ``g`` (nonlinearity gain)       ``g``                      1.0
    ``mult_coupling``               ``mult_coupling``          False
    ``g_ex``, ``g_in``              ``g_ex``, ``g_in``         1.0
    ``theta_ex``, ``theta_in``      ``theta_ex``, ``theta_in`` 0.0
    ``linear_summation``            ``linear_summation``       True
    =============================== ========================== ===========

    Attributes
    ----------
    rate : brainstate.ShortTermState
        Deterministic rate state :math:`X_n` (float64 array of shape
        ``self.varshape`` or ``(batch_size,) + self.varshape``). This is the
        noise-free rate variable.
    noise : brainstate.ShortTermState
        Last noise sample :math:`\sigma\,\xi_{n-1}` (float64 array, same shape
        as ``rate``). Records the noise term used in the previous step.
    noisy_rate : brainstate.ShortTermState
        Noisy rate :math:`X_\mathrm{noisy,n} = X_n + \sqrt{\tau/h}\,\mathrm{noise}_n`
        (float64 array, same shape as ``rate``). This is the outgoing signal
        sent to downstream neurons and used for multiplicative coupling
        evaluation.
    instant_rate : brainstate.ShortTermState
        Noisy rate value for instantaneous event propagation (float64 array,
        same shape as ``rate``). Set to ``noisy_rate`` after each update.
    delayed_rate : brainstate.ShortTermState
        Noisy rate value for delayed projections (float64 array, same shape as
        ``rate``). Set to ``noisy_rate`` after each update.
    _step_count : brainstate.ShortTermState
        Internal step counter for delayed event scheduling (int64 scalar).
        Incremented by 1 after each ``update`` call.
    _delayed_ex_queue : dict
        Internal queue mapping ``step_idx`` (int) to accumulated excitatory
        delayed events (float64 array of shape ``state_shape``).
    _delayed_in_queue : dict
        Internal queue mapping ``step_idx`` (int) to accumulated inhibitory
        delayed events (float64 array of shape ``state_shape``).

    Raises
    ------
    ValueError
        If ``tau <= 0`` (checked during ``__init__`` via
        ``_validate_parameters``).
    ValueError
        If ``sigma < 0`` (checked during ``__init__`` via
        ``_validate_parameters``).
    ValueError
        If ``instant_rate_events`` contain non-zero ``delay_steps`` (checked
        during ``update`` via ``_accumulate_instant_events``).
    ValueError
        If ``delayed_rate_events`` contain negative ``delay_steps`` (checked
        during ``update`` via ``_schedule_delayed_events``).
    ValueError
        If event tuples have length other than 2, 3, or 4 (checked during
        ``update`` via ``_extract_event_fields``).

    Notes
    -----
    **Runtime Events**

    Events can be provided to ``update()`` via ``instant_rate_events`` and
    ``delayed_rate_events`` parameters. Each event can be specified as:

    - **Scalar**: Treated as ``rate`` value with ``weight=1.0``.
    - **Tuple**: ``(rate, weight)`` or ``(rate, weight, delay_steps)`` or
      ``(rate, weight, delay_steps, multiplicity)``.
    - **Dict**: Keys ``'rate'``/``'coeff'``/``'value'`` (event value),
      ``'weight'`` (synaptic weight), ``'delay_steps'``/``'delay'`` (integer
      delay in time steps), ``'multiplicity'`` (event count).

    **Sign Convention**: Events with ``weight >= 0`` contribute to the
    excitatory branch; events with ``weight < 0`` contribute to the inhibitory
    branch.

    **Linear Summation Semantics**: For ``linear_summation=False``, event
    values are transformed by the input nonlinearity during buffering (matching
    NEST event handlers). For ``linear_summation=True``, the nonlinearity is
    applied to the summed input during the update step.

    **Comparison to ``rate_neuron_ipn``**

    The ``_opn`` variant uses output noise (applied after nonlinearity and
    transmitted to downstream neurons), while ``_ipn`` uses input noise (applied
    before dynamics propagation, directly affecting the state evolution). This
    leads to different stationary distributions, noise scaling, and stability
    properties. In ``_opn``, noise does not feed back into the deterministic
    dynamics.

    Examples
    --------
    Minimal output-noise rate neuron:

    .. code-block:: python

       >>> from brainpy import state as bst
       >>> import brainunit as u
       >>> model = bst.rate_neuron_opn(in_size=10, tau=20*u.ms, sigma=0.5)
       >>> model.init_all_states(batch_size=1)
       >>> rate = model(x=0.1)  # external drive
       >>> print(rate.shape)
       (1, 10)

    Multiplicative coupling with custom nonlinearity:

    .. code-block:: python

       >>> import numpy as np
       >>> def tanh_nonlin(h):
       ...     return np.tanh(h)
       >>> model = bst.rate_neuron_opn(
       ...     in_size=5,
       ...     tau=10*u.ms,
       ...     sigma=0.3,
       ...     mult_coupling=True,
       ...     g_ex=1.5, theta_ex=1.0,
       ...     input_nonlinearity=tanh_nonlin
       ... )

    Accessing noisy rate output:

    .. code-block:: python

       >>> model = bst.rate_neuron_opn(in_size=3, tau=10*u.ms, sigma=0.2)
       >>> model.init_all_states()
       >>> rate_deterministic = model.update(x=0.5)  # propagates deterministic dynamics
       >>> rate_noisy = model.noisy_rate.value        # includes output noise
       >>> print(rate_noisy.shape)
       (3,)

    References
    ----------
    .. [1] NEST Simulator Documentation: ``rate_neuron_opn``
           https://nest-simulator.readthedocs.io/en/stable/models/rate_neuron_opn.html
    .. [2] Hahne, J., Dahmen, D., Schuecker, J., Frommer, A., Bolten, M.,
           Helias, M., & Diesmann, M. (2017). Integration of continuous-time
           dynamics in a spiking neural network simulator.
           *Frontiers in Neuroinformatics*, 11, 34.

    See Also
    --------
    rate_neuron_ipn : Input-noise variant of the rate neuron template.
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
        sigma: ArrayLike = 1.0,
        mu: ArrayLike = 0.0,
        g: ArrayLike = 1.0,
        mult_coupling: bool = False,
        g_ex: ArrayLike = 1.0,
        g_in: ArrayLike = 1.0,
        theta_ex: ArrayLike = 0.0,
        theta_in: ArrayLike = 0.0,
        linear_summation: bool = True,
        input_nonlinearity: Callable | None = None,
        mult_coupling_ex_fn: Callable | None = None,
        mult_coupling_in_fn: Callable | None = None,
        rate_initializer: Callable = braintools.init.Constant(0.0),
        noise_initializer: Callable = braintools.init.Constant(0.0),
        noisy_rate_initializer: Callable = braintools.init.Constant(0.0),
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

        self.input_nonlinearity = input_nonlinearity
        self.mult_coupling_ex_fn = mult_coupling_ex_fn
        self.mult_coupling_in_fn = mult_coupling_in_fn
        self.noisy_rate_initializer = noisy_rate_initializer

        self._validate_parameters()

    @property
    def recordables(self):
        r"""List of state variable names that can be recorded.

        Returns
        -------
        list of str
            ``['rate', 'noise', 'noisy_rate']``.
        """
        return ['rate', 'noise', 'noisy_rate']

    @property
    def receptor_types(self):
        r"""Receptor type dictionary for projection compatibility.

        Returns
        -------
        dict[str, int]
            ``{'RATE': 0}``. Rate neurons have a single receptor type.
        """
        return {'RATE': 0}

    def _validate_parameters(self):
        r"""Validate model parameters at construction time.

        Raises
        ------
        ValueError
            If ``tau <= 0`` or ``sigma < 0``.

        Notes
        -----
        This method is called automatically during ``__init__``.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.tau, self.sigma)):
            return
        if np.any(self.tau <= 0.0 * u.ms):
            raise ValueError('Time constant tau must be > 0.')
        if np.any(self.sigma < 0.0):
            raise ValueError('Noise parameter sigma must be >= 0.')

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

        This method must be called before the first ``update()`` call. It
        creates all internal state variables (``rate``, ``noise``,
        ``noisy_rate``, ``instant_rate``, ``delayed_rate``, ``_step_count``)
        and resets the delayed event queues.

        Parameters
        ----------
        **kwargs
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        **Initialized State Variables**

        This method initializes the following state variables:

        - **rate** (``brainstate.ShortTermState``): Deterministic rate state
          :math:`X_n` (float64 array). Initialized using ``rate_initializer``.
        - **noise** (``brainstate.ShortTermState``): Last noise sample
          :math:`\sigma\,\xi_{n-1}` (float64 array). Initialized using
          ``noise_initializer``.
        - **noisy_rate** (``brainstate.ShortTermState``): Noisy rate
          :math:`X_\mathrm{noisy,n} = X_n + \sqrt{\tau/h}\,\mathrm{noise}_n`
          (float64 array). Initialized using ``noisy_rate_initializer``.
        - **instant_rate** (``brainstate.ShortTermState``): Noisy rate value for
          instantaneous event propagation (float64 array). Initialized as a copy
          of ``noisy_rate``.
        - **delayed_rate** (``brainstate.ShortTermState``): Noisy rate value for
          delayed projections (float64 array). Initialized as a copy of
          ``noisy_rate``.
        - **_step_count** (``brainstate.ShortTermState``): Internal step counter
          for delayed event scheduling (int64 scalar). Initialized to ``0``.
        - **_delayed_ex_queue** (dict): Internal queue mapping ``step_idx``
          (int) to accumulated excitatory delayed events (float64 array).
          Initialized as empty dict.
        - **_delayed_in_queue** (dict): Internal queue mapping ``step_idx``
          (int) to accumulated inhibitory delayed events (float64 array).
          Initialized as empty dict.

        **Array Precision**

        All state arrays are float64 NumPy arrays. All parameters (``tau``,
        ``sigma``, ``mu``, etc.) are coerced to float64 during initialization.

        **Repeated Calls**

        Calling ``init_state()`` multiple times will overwrite existing state
        variables and clear the delayed event queues. This can be used to reset
        the model to initial conditions.

        Examples
        --------
        Initialize a single population:

        .. code-block:: python

           >>> from brainpy import state as bst
           >>> import brainunit as u
           >>> model = bst.rate_neuron_opn(in_size=10, tau=20*u.ms)
           >>> model.init_state()
           >>> print(model.rate.value.shape)
           (10,)

        Custom initializers:

        .. code-block:: python

           >>> import braintools
           >>> model = bst.rate_neuron_opn(
           ...     in_size=5,
           ...     tau=10*u.ms,
           ...     rate_initializer=braintools.init.Normal(0.5, 0.1),
           ...     noisy_rate_initializer=braintools.init.Normal(0.5, 0.1)
           ... )
           >>> model.init_state()
           >>> print(model.rate.value.mean())  # approximately 0.5

        See Also
        --------
        update : Perform one simulation step after initialization.
        """
        rate = braintools.init.param(self.rate_initializer, self.varshape)
        noise = braintools.init.param(self.noise_initializer, self.varshape)
        noisy_rate = braintools.init.param(self.noisy_rate_initializer, self.varshape)
        rate_np = self._to_numpy(rate)
        noise_np = self._to_numpy(noise)
        noisy_rate_np = self._to_numpy(noisy_rate)

        self.rate = brainstate.ShortTermState(rate_np)
        self.noise = brainstate.ShortTermState(noise_np)
        self.noisy_rate = brainstate.ShortTermState(noisy_rate_np)
        dftype = brainstate.environ.dftype()
        self.instant_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=dftype, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=dftype, copy=True))
        self._alloc_phi_rate(rate_np)

    def update(self, x=0.0, noise=None):
        r"""Advance the output-noise rate dynamics by one step.

        Network coupling arrives continuously through the substrate's delta
        channel (seam-(H)): :math:`h=\sum_\mathrm{delta} w\,r_\mathrm{pre}` is read
        from ``sum_delta_inputs(0.0)`` and the external drive from
        ``sum_current_inputs(x, rate)``. Output noise is added to form the noisy
        rate :math:`X_\mathrm{noisy}` *before* the multiplicative coupling factors
        are evaluated. The whole step lowers under
        ``brainstate.transform.for_loop`` / ``jit``.

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

        rate_prev = jnp.broadcast_to(jnp.asarray(self.rate.value, dtype=dftype), state_shape)

        mu_ext, h_a, h_b = self._read_coupling(x)

        if noise is None:
            xi = brainstate.random.randn(*state_shape)
        else:
            xi = jnp.broadcast_to(jnp.asarray(noise, dtype=dftype), state_shape)
        noise_now = sigma * xi

        P1 = np.exp(-h / tau)
        P2 = -np.expm1(-h / tau)
        output_noise_factor = np.sqrt(tau / h)
        noisy_rate = rate_prev + output_noise_factor * noise_now

        mu_total = mu + mu_ext
        rate_new = P1 * rate_prev + P2 * mu_total
        rate_new = rate_new + P2 * self._coupling_increment(noisy_rate, h_a, h_b)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.noisy_rate.value = noisy_rate
        self.delayed_rate.value = noisy_rate
        self.instant_rate.value = noisy_rate
        self._store_phi_rate(rate_new)
        return rate_new
