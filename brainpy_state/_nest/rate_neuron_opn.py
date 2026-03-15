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
        Time constant :math:`\tau` (saiunit quantity with ms dimension).
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

       >>> import brainpy.state as bst
       >>> import saiunit as u
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

    def _input_transform(self, h: np.ndarray, state_shape):
        r"""Apply input nonlinearity :math:`g(h)`.

        Parameters
        ----------
        h : np.ndarray
            Input value (pre-nonlinearity, float64).
        state_shape : tuple
            Target broadcast shape for output.

        Returns
        -------
        np.ndarray
            Transformed input :math:`g(h)` broadcast to ``state_shape``.

        Notes
        -----
        If ``input_nonlinearity`` is ``None``, uses default :math:`g(h)=g\,h`.
        Otherwise calls user-provided callable.
        r"""
        h_np = self._broadcast_to_state(self._to_numpy(h), state_shape)
        if self.input_nonlinearity is None:
            g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
            return g * h_np
        y = self._call_nl(self.input_nonlinearity, h_np)
        return self._broadcast_to_state(self._to_numpy(y), state_shape)

    def _mult_ex_transform(self, rate: np.ndarray, state_shape):
        r"""Compute excitatory multiplicative coupling factor :math:`H_\mathrm{ex}(X_\mathrm{noisy})`.

        Parameters
        ----------
        rate : np.ndarray
            Current noisy rate state :math:`X_\mathrm{noisy}` (float64).
        state_shape : tuple
            Target broadcast shape for output.

        Returns
        -------
        np.ndarray
            Coupling factor :math:`H_\mathrm{ex}(X_\mathrm{noisy})` broadcast to
            ``state_shape``.

        Notes
        -----
        If ``mult_coupling_ex_fn`` is ``None``, uses default
        :math:`g_\mathrm{ex}(\theta_\mathrm{ex}-X_\mathrm{noisy})`. Otherwise
        calls user-provided callable. Evaluated at the *noisy* rate (matching
        NEST ``rate_neuron_opn_impl.h``).
        r"""
        rate_np = self._broadcast_to_state(self._to_numpy(rate), state_shape)
        if self.mult_coupling_ex_fn is None:
            g_ex = self._broadcast_to_state(self._to_numpy(self.g_ex), state_shape)
            theta_ex = self._broadcast_to_state(self._to_numpy(self.theta_ex), state_shape)
            return g_ex * (theta_ex - rate_np)
        y = self._call_nl(self.mult_coupling_ex_fn, rate_np)
        return self._broadcast_to_state(self._to_numpy(y), state_shape)

    def _mult_in_transform(self, rate: np.ndarray, state_shape):
        r"""Compute inhibitory multiplicative coupling factor :math:`H_\mathrm{in}(X_\mathrm{noisy})`.

        Parameters
        ----------
        rate : np.ndarray
            Current noisy rate state :math:`X_\mathrm{noisy}` (float64).
        state_shape : tuple
            Target broadcast shape for output.

        Returns
        -------
        np.ndarray
            Coupling factor :math:`H_\mathrm{in}(X_\mathrm{noisy})` broadcast to
            ``state_shape``.

        Notes
        -----
        If ``mult_coupling_in_fn`` is ``None``, uses default
        :math:`g_\mathrm{in}(\theta_\mathrm{in}+X_\mathrm{noisy})`. Otherwise
        calls user-provided callable. Evaluated at the *noisy* rate (matching
        NEST ``rate_neuron_opn_impl.h``).
        """
        rate_np = self._broadcast_to_state(self._to_numpy(rate), state_shape)
        if self.mult_coupling_in_fn is None:
            g_in = self._broadcast_to_state(self._to_numpy(self.g_in), state_shape)
            theta_in = self._broadcast_to_state(self._to_numpy(self.theta_in), state_shape)
            return g_in * (theta_in + rate_np)
        y = self._call_nl(self.mult_coupling_in_fn, rate_np)
        return self._broadcast_to_state(self._to_numpy(y), state_shape)

    def _extract_event_fields(self, ev, default_delay_steps: int):
        r"""Extract ``(rate, weight, multiplicity, delay_steps)`` from event.

        Parameters
        ----------
        ev : scalar, dict, tuple, or list
            Event specification. See class docstring for format.
        default_delay_steps : int
            Default delay if not specified in event.

        Returns
        -------
        rate : ArrayLike
            Event rate value.
        weight : ArrayLike
            Event weight (sign determines excitatory/inhibitory branch).
        multiplicity : ArrayLike
            Event multiplicity factor.
        delay_steps : int
            Integer delay in simulation time steps.

        Raises
        ------
        ValueError
            If tuple/list event has length other than 2, 3, or 4.
        """
        if isinstance(ev, dict):
            rate = ev.get('rate', ev.get('coeff', ev.get('value', 0.0)))
            weight = ev.get('weight', 1.0)
            multiplicity = ev.get('multiplicity', 1.0)
            delay_steps = ev.get('delay_steps', ev.get('delay', default_delay_steps))
        elif isinstance(ev, (tuple, list)):
            if len(ev) == 2:
                rate, weight = ev
                delay_steps = default_delay_steps
                multiplicity = 1.0
            elif len(ev) == 3:
                rate, weight, delay_steps = ev
                multiplicity = 1.0
            elif len(ev) == 4:
                rate, weight, delay_steps, multiplicity = ev
            else:
                raise ValueError('Rate event tuples must have length 2, 3, or 4.')
        else:
            rate = ev
            weight = 1.0
            multiplicity = 1.0
            delay_steps = default_delay_steps

        delay_steps = self._to_int_scalar(delay_steps, name='delay_steps')
        return rate, weight, multiplicity, delay_steps

    def _event_to_ex_in(self, ev, default_delay_steps: int, state_shape):
        r"""Convert event to excitatory and inhibitory contributions.

        Parameters
        ----------
        ev : scalar, dict, tuple, or list
            Event specification.
        default_delay_steps : int
            Default delay if not specified in event.
        state_shape : tuple
            Target shape for broadcast.

        Returns
        -------
        ex : np.ndarray
            Excitatory contribution (float64 array of shape ``state_shape``).
        inh : np.ndarray
            Inhibitory contribution (float64 array of shape ``state_shape``).
        delay_steps : int
            Integer delay in simulation time steps.

        Notes
        -----
        Sign convention: events with ``weight >= 0`` contribute to ``ex``,
        events with ``weight < 0`` contribute to ``inh``. For
        ``linear_summation=False``, the input nonlinearity is applied during
        this conversion (matching NEST event handling).
        """
        rate, weight, multiplicity, delay_steps = self._extract_event_fields(ev, default_delay_steps)

        rate_np = self._broadcast_to_state(self._to_numpy(rate), state_shape)
        weight_np = self._broadcast_to_state(self._to_numpy(weight), state_shape)
        multiplicity_np = self._broadcast_to_state(self._to_numpy(multiplicity), state_shape)
        dftype = brainstate.environ.dftype()
        weight_sign = self._broadcast_to_state(
            np.asarray(u.math.asarray(weight), dtype=dftype) >= 0.0,
            state_shape,
        )

        if self.linear_summation:
            weighted_value = rate_np * weight_np * multiplicity_np
        else:
            weighted_value = self._input_transform(rate_np, state_shape) * weight_np * multiplicity_np

        ex = np.where(weight_sign, weighted_value, 0.0)
        inh = np.where(weight_sign, 0.0, weighted_value)
        return ex, inh, delay_steps

    def _accumulate_instant_events(self, events, state_shape):
        r"""Accumulate instantaneous events (no delay).

        Parameters
        ----------
        events : None, dict, tuple, list, or iterable
            Instantaneous event specification(s).
        state_shape : tuple
            Target shape for broadcast.

        Returns
        -------
        ex : np.ndarray
            Total excitatory contribution (float64 array of shape ``state_shape``).
        inh : np.ndarray
            Total inhibitory contribution (float64 array of shape ``state_shape``).

        Raises
        ------
        ValueError
            If any event specifies non-zero ``delay_steps``.
        """
        dftype = brainstate.environ.dftype()
        ex = np.zeros(state_shape, dtype=dftype)
        inh = np.zeros(state_shape, dtype=dftype)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._event_to_ex_in(
                ev,
                default_delay_steps=0,
                state_shape=state_shape,
            )
            if delay_steps != 0:
                raise ValueError('instant_rate_events must not specify non-zero delay_steps.')
            ex += ex_i
            inh += inh_i
        return ex, inh

    def _schedule_delayed_events(self, events, step_idx: int, state_shape):
        r"""Schedule delayed events and return zero-delay contributions.

        Parameters
        ----------
        events : None, dict, tuple, list, or iterable
            Delayed event specification(s).
        step_idx : int
            Current simulation step index.
        state_shape : tuple
            Target shape for broadcast.

        Returns
        -------
        ex_now : np.ndarray
            Excitatory events with ``delay_steps=0`` (float64 array of shape
            ``state_shape``).
        inh_now : np.ndarray
            Inhibitory events with ``delay_steps=0`` (float64 array of shape
            ``state_shape``).

        Raises
        ------
        ValueError
            If any event has negative ``delay_steps``.

        Notes
        -----
        Events with ``delay_steps > 0`` are added to internal delay queues
        ``_delayed_ex_queue`` and ``_delayed_in_queue`` at target step
        ``step_idx + delay_steps``.
        """
        dftype = brainstate.environ.dftype()
        ex_now = np.zeros(state_shape, dtype=dftype)
        inh_now = np.zeros(state_shape, dtype=dftype)
        for ev in self._coerce_events(events):
            ex_i, inh_i, delay_steps = self._event_to_ex_in(
                ev,
                default_delay_steps=1,
                state_shape=state_shape,
            )
            if delay_steps < 0:
                raise ValueError('delay_steps for delayed_rate_events must be >= 0.')
            if delay_steps == 0:
                ex_now += ex_i
                inh_now += inh_i
            else:
                target_step = step_idx + delay_steps
                self._queue_add(self._delayed_ex_queue, target_step, ex_i)
                self._queue_add(self._delayed_in_queue, target_step, inh_i)
        return ex_now, inh_now

    def _common_inputs_template(self, x, instant_rate_events, delayed_rate_events):
        r"""Collect all input contributions for the current update step.

        Parameters
        ----------
        x : ArrayLike
            External drive passed to ``update``.
        instant_rate_events : None, dict, tuple, list, or iterable
            Instantaneous events.
        delayed_rate_events : None, dict, tuple, list, or iterable
            Delayed events.

        Returns
        -------
        state_shape : tuple
            Current state shape (with batch dimension if present).
        step_idx : int
            Current simulation step index.
        delayed_ex : np.ndarray
            Delayed excitatory input arriving at current step (float64 array).
        delayed_in : np.ndarray
            Delayed inhibitory input arriving at current step (float64 array).
        instant_ex : np.ndarray
            Instantaneous excitatory input (float64 array).
        instant_in : np.ndarray
            Instantaneous inhibitory input (float64 array).
        mu_ext : np.ndarray
            External drive from ``x`` and current inputs (float64 array).

        Notes
        -----
        This method combines:

        1. Delayed events arriving at current step (drained from queues).
        2. Newly scheduled delayed events with ``delay_steps=0``.
        3. Instantaneous events.
        4. Delta inputs (sign-separated into excitatory/inhibitory).
        5. Current inputs via ``sum_current_inputs``.
        """
        state_shape = self.rate.value.shape
        ditype = brainstate.environ.ditype()
        step_idx = int(np.asarray(self._step_count.value, dtype=ditype).reshape(-1)[0])

        delayed_ex, delayed_in = self._drain_delayed_queue(step_idx, state_shape)
        delayed_ex_now, delayed_in_now = self._schedule_delayed_events(
            delayed_rate_events,
            step_idx=step_idx,
            state_shape=state_shape,
        )
        delayed_ex = delayed_ex + delayed_ex_now
        delayed_in = delayed_in + delayed_in_now

        instant_ex, instant_in = self._accumulate_instant_events(
            instant_rate_events,
            state_shape=state_shape,
        )

        delta_input = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0.0)), state_shape)
        instant_ex += np.where(delta_input > 0.0, delta_input, 0.0)
        instant_in += np.where(delta_input < 0.0, delta_input, 0.0)

        mu_ext = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.rate.value)), state_shape)

        return state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables for simulation.

        This method must be called before the first ``update()`` call. It
        creates all internal state variables (``rate``, ``noise``,
        ``noisy_rate``, ``instant_rate``, ``delayed_rate``, ``_step_count``)
        and resets the delayed event queues.

        Parameters
        ----------
        batch_size : int or None, optional
            Batch dimension size for parallel simulation of multiple instances.
            If ``None`` (default), state shape is ``self.varshape`` (single
            instance). If ``int > 0``, state shape is
            ``(batch_size,) + self.varshape`` (batched instances). For example,
            if ``self.varshape=(10,)`` and ``batch_size=32``, the state shape
            will be ``(32, 10)``. Default: ``None``.
        **kwargs
            Additional keyword arguments (reserved for future use, currently
            ignored).

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
        Initialize without batch dimension (single instance):

        .. code-block:: python

           >>> import brainpy.state as bst
           >>> import saiunit as u
           >>> model = bst.rate_neuron_opn(in_size=10, tau=20*u.ms)
           >>> model.init_state()
           >>> print(model.rate.value.shape)
           (10,)

        Initialize with batch dimension (32 parallel instances):

        .. code-block:: python

           >>> model = bst.rate_neuron_opn(in_size=10, tau=20*u.ms)
           >>> model.init_state(batch_size=32)
           >>> print(model.rate.value.shape)
           (32, 10)

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
        rate = braintools.init.param(self.rate_initializer, self.varshape, batch_size)
        noise = braintools.init.param(self.noise_initializer, self.varshape, batch_size)
        noisy_rate = braintools.init.param(self.noisy_rate_initializer, self.varshape, batch_size)
        rate_np = self._to_numpy(rate)
        noise_np = self._to_numpy(noise)
        noisy_rate_np = self._to_numpy(noisy_rate)

        self.rate = brainstate.ShortTermState(rate_np)
        self.noise = brainstate.ShortTermState(noise_np)
        self.noisy_rate = brainstate.ShortTermState(noisy_rate_np)
        dftype = brainstate.environ.dftype()
        self.instant_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=dftype, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(noisy_rate_np, dtype=dftype, copy=True))
        ditype = brainstate.environ.ditype()
        self._step_count = brainstate.ShortTermState(np.asarray(0, dtype=ditype))

        self._delayed_ex_queue = {}
        self._delayed_in_queue = {}

    def update(self, x=0.0, instant_rate_events=None, delayed_rate_events=None, noise=None):
        r"""Perform one simulation step of output-noise rate dynamics.

        This method implements the core update algorithm for the output-noise
        rate neuron model. It propagates the deterministic rate dynamics,
        applies output noise, processes delayed and instantaneous synaptic
        events, and evaluates optional multiplicative coupling factors.

        Parameters
        ----------
        x : ArrayLike, optional
            External drive (dimensionless scalar or array). Broadcastable to
            ``self.varshape`` or current batch shape. Added to ``mu`` as
            constant forcing term. Default: ``0.0``.
        instant_rate_events : None or dict or tuple or list or iterable, optional
            Instantaneous rate events applied in the current step without delay.
            Each event can be:

            - Scalar (treated as ``rate`` value with ``weight=1.0``).
            - Tuple: ``(rate, weight)`` or ``(rate, weight, delay_steps)`` or
              ``(rate, weight, delay_steps, multiplicity)``.
            - Dict with keys ``'rate'``/``'coeff'``/``'value'``, ``'weight'``,
              ``'delay_steps'``/``'delay'``, ``'multiplicity'``.

            Events with non-zero ``delay_steps`` will raise ``ValueError``.
            Default: ``None`` (no instantaneous events).
        delayed_rate_events : None or dict or tuple or list or iterable, optional
            Delayed rate events scheduled with integer ``delay_steps`` (units of
            simulation time step :math:`h`). Same format as
            ``instant_rate_events``. Events with ``delay_steps=0`` are applied
            immediately. Events with ``delay_steps>0`` are queued and applied
            after the specified delay. Negative ``delay_steps`` raise
            ``ValueError``. Default: ``None`` (no delayed events).
        noise : ArrayLike or None, optional
            Externally supplied noise sample :math:`\xi_n` (dimensionless scalar
            or array). Broadcastable to current batch shape. If ``None``
            (default), draws :math:`\xi_n\sim\mathcal{N}(0,1)` internally using
            ``np.random.normal``. If provided, must have zero mean and unit
            variance for correct noise amplitude. Default: ``None``.

        Returns
        -------
        rate_new : np.ndarray
            Updated deterministic rate state :math:`X_{n+1}` (float64 array of
            shape ``self.rate.value.shape``). This is the noise-free rate after
            one simulation step. To access the noisy rate, use
            ``self.noisy_rate.value``.

        Raises
        ------
        ValueError
            If ``instant_rate_events`` contain non-zero ``delay_steps``.
        ValueError
            If ``delayed_rate_events`` contain negative ``delay_steps``.
        ValueError
            If event tuples have length other than 2, 3, or 4.

        Notes
        -----
        **Update Algorithm**

        The method performs the following steps in order:

        **1. Input Collection**

        Collect all input contributions for the current step:

        - Delayed events arriving at current step (drained from internal queues
          ``_delayed_ex_queue`` and ``_delayed_in_queue``).
        - Newly scheduled delayed events with ``delay_steps=0`` (from
          ``delayed_rate_events``).
        - Instantaneous events (from ``instant_rate_events``).
        - Delta inputs via ``sum_delta_inputs(0.0)`` (sign-separated into
          excitatory/inhibitory branches).
        - Current inputs via ``sum_current_inputs(x, rate)`` (external drive and
          synaptic inputs).

        **2. Propagator Coefficients**

        Compute exponential Euler integration coefficients:

        .. math::

           P_1 = \exp(-h/\tau), \quad P_2 = 1 - P_1 = -\mathrm{expm1}(-h/\tau),

        where :math:`h` is the simulation time step (in ms) and :math:`\tau` is
        the time constant. Uses ``np.expm1`` for numerically stable evaluation
        of :math:`1-e^{-x}`.

        **3. Output Noise**

        Draw noise sample :math:`\xi_n\sim\mathcal{N}(0,1)` (or use external
        ``noise`` parameter) and compute noisy rate:

        .. math::

           X_\mathrm{noisy,n} = X_n + \sqrt{\frac{\tau}{h}}\,\sigma\,\xi_n.

        The scaling factor :math:`\sqrt{\tau/h}` ensures correct amplitude
        scaling in the :math:`h\to 0` limit.

        **4. Deterministic Dynamics Propagation**

        Propagate the deterministic part of the dynamics:

        .. math::

           X' = P_1 X_n + P_2(\mu + \mu_\mathrm{ext}),

        where :math:`\mu_\mathrm{ext}` is the external drive from ``x`` and
        current inputs.

        **5. Multiplicative Coupling**

        If ``mult_coupling=True``, evaluate multiplicative coupling factors at
        the *noisy* rate:

        .. math::

           H_\mathrm{ex}(X_\mathrm{noisy,n}), \quad
           H_\mathrm{in}(X_\mathrm{noisy,n}).

        If ``mult_coupling=False``, :math:`H_\mathrm{ex}=H_\mathrm{in}=1`.

        **6. Network Input Application**

        Apply network input according to ``linear_summation`` mode:

        - **linear_summation=True**: Nonlinearity applied to summed input:

          .. math::

             X_{n+1} = X' + P_2 [H_\mathrm{ex}\cdot g(I_\mathrm{ex})
             + H_\mathrm{in}\cdot g(I_\mathrm{in})].

          If ``mult_coupling=False``, simplifies to:

          .. math::

             X_{n+1} = X' + P_2 g(I_\mathrm{ex} + I_\mathrm{in}).

        - **linear_summation=False**: Nonlinearity applied per branch during
          event processing. Network input is already transformed:

          .. math::

             X_{n+1} = X' + P_2 [H_\mathrm{ex}\cdot I_\mathrm{ex}
             + H_\mathrm{in}\cdot I_\mathrm{in}].

        **7. State Updates**

        Update all state variables:

        - ``rate``: Deterministic rate :math:`X_{n+1}`.
        - ``noise``: Noise sample :math:`\sigma\,\xi_n`.
        - ``noisy_rate``: Noisy rate :math:`X_\mathrm{noisy,n}`.
        - ``delayed_rate``: Noisy rate for delayed projections.
        - ``instant_rate``: Noisy rate for instantaneous projections.
        - ``_step_count``: Incremented by 1.

        **Key Distinction from ``rate_neuron_ipn``**

        The noisy rate :math:`X_\mathrm{noisy,n}` is used for multiplicative
        coupling and as the outgoing signal to downstream neurons, but the noise
        does *not* feed back into the deterministic dynamics (i.e., :math:`X'`
        depends only on the noise-free rate :math:`X_n`). This contrasts with
        the input-noise variant (``rate_neuron_ipn``) where noise enters the
        differential equation directly.

        **Numerical Stability**

        - The exponential Euler scheme is unconditionally stable for all
          :math:`h>0`.
        - Uses ``np.expm1(-h/tau)`` to avoid catastrophic cancellation for
          small :math:`h/\tau`.
        - Noise scaling :math:`\sqrt{\tau/h}` ensures correct amplitude in the
          :math:`h\to 0` limit, but per-step noise variance diverges (reflecting
          white-noise nature).

        **Failure Modes**


        - **Invalid time constants**: Caught at construction by
          ``_validate_parameters`` (enforces :math:`\tau>0`, :math:`\sigma\ge 0`).
        - **Invalid events**: Raises ``ValueError`` for events with incorrect
          ``delay_steps`` or tuple length.
        - **Unbounded rates**: No automatic rectification or clipping. Noisy
          rate can exceed any bounds.
        - **NaN propagation**: If input contains NaN, all downstream states will
          be NaN. No automatic detection or recovery.

        See Also
        --------
        init_state : Initialize all state variables before first update.
        rate_neuron_ipn.update : Input-noise variant update method.
        """
        h = float(u.math.asarray(brainstate.environ.get_dt() / u.ms))

        state_shape, step_idx, delayed_ex, delayed_in, instant_ex, instant_in, mu_ext = self._common_inputs_template(
            x=x,
            instant_rate_events=instant_rate_events,
            delayed_rate_events=delayed_rate_events,
        )

        tau = self._broadcast_to_state(self._to_numpy_ms(self.tau), state_shape)
        sigma = self._broadcast_to_state(self._to_numpy(self.sigma), state_shape)
        mu = self._broadcast_to_state(self._to_numpy(self.mu), state_shape)

        rate_prev = self._broadcast_to_state(self._to_numpy(self.rate.value), state_shape)

        if noise is None:
            xi = np.random.normal(size=state_shape)
        else:
            xi = self._broadcast_to_state(self._to_numpy(noise), state_shape)
        noise_now = sigma * xi

        P1 = np.exp(-h / tau)
        P2 = -np.expm1(-h / tau)
        output_noise_factor = np.sqrt(tau / h)
        noisy_rate = rate_prev + output_noise_factor * noise_now

        mu_total = mu + mu_ext
        rate_new = P1 * rate_prev + P2 * mu_total

        H_ex = np.ones_like(rate_prev)
        H_in = np.ones_like(rate_prev)
        if self.mult_coupling:
            H_ex = self._mult_ex_transform(noisy_rate, state_shape)
            H_in = self._mult_in_transform(noisy_rate, state_shape)

        if self.linear_summation:
            if self.mult_coupling:
                rate_new += P2 * H_ex * self._input_transform(delayed_ex + instant_ex, state_shape)
                rate_new += P2 * H_in * self._input_transform(delayed_in + instant_in, state_shape)
            else:
                rate_new += P2 * self._input_transform(
                    delayed_ex + instant_ex + delayed_in + instant_in,
                    state_shape,
                )
        else:
            rate_new += P2 * H_ex * (delayed_ex + instant_ex)
            rate_new += P2 * H_in * (delayed_in + instant_in)

        self.rate.value = rate_new
        self.noise.value = noise_now
        self.noisy_rate.value = noisy_rate
        self.delayed_rate.value = noisy_rate
        self.instant_rate.value = noisy_rate
        ditype = brainstate.environ.ditype()
        self._step_count.value = np.asarray(step_idx + 1, dtype=ditype)
        return rate_new
