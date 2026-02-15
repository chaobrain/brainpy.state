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
import jax.scipy.special as jspecial
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Dynamics

__all__ = [
    'erfc_neuron',
]


class erfc_neuron(Dynamics):
    r"""Binary stochastic neuron with complementary error-function gain.

    Description
    -----------

    ``erfc_neuron`` re-implements NEST's binary neuron model of the same name.
    The neuron keeps a persistent synaptic input state :math:`h` and updates
    its binary output :math:`y \in \{0, 1\}` at Poisson-distributed update
    times with mean interval :math:`\tau_m`.

    At each scheduled update, the new binary state is sampled as

    .. math::

       y \leftarrow \mathbf{1}[U < g(h + c)], \quad U \sim \mathrm{Uniform}(0, 1),

    with gain function

    .. math::

       g(h) = \frac{1}{2}\,\mathrm{erfc}\!\left(-\frac{h - \theta}{\sqrt{2}\,\sigma}\right).

    This matches the NEST implementation in ``gainfunction_erfc::operator()``.
    The model corresponds to a McCulloch-Pitts threshold unit with additive
    Gaussian noise of standard deviation :math:`\sigma`.

    Update order (NEST semantics)
    -----------------------------

    Each simulation step follows the same ordering as NEST's
    ``binary_neuron::update()``:

    1. Accumulate delta inputs into persistent :math:`h`.
    2. Read current input :math:`c` for the present step.
    3. If ``t + dt > t_next`` (strict inequality), sample a new binary state
       from :math:`g(h+c)`.
    4. If an update happened, advance ``t_next`` by ``Exp(1) * tau_m``.

    As in NEST, probabilities are not explicitly clipped before comparing
    against uniform random numbers.

    Parameters
    ----------
    in_size : Size
        Number/shape of neurons.
    tau_m : ArrayLike, optional
        Mean inter-update interval :math:`\tau_m`. Default: ``10.0 * u.ms``.
    theta : ArrayLike, optional
        Threshold :math:`\theta`. Default: ``0.0 * u.mV``.
    sigma : ArrayLike, optional
        Gain/noise parameter :math:`\sigma`. Default: ``1.0 * u.mV``.
    y_initializer : Callable, optional
        Initializer for initial binary state ``y``. Default:
        ``braintools.init.Constant(0.0)``.
    stochastic_update : bool, optional
        If ``True`` (default), use Poisson update scheduling as in NEST.
        If ``False``, update each time step while retaining stochastic
        state sampling from the same gain function.
    rng_seed : int, optional
        Seed for internal random sampling. Default: ``0``.
    name : str, optional
        Object name.

    Attributes
    ----------
    y : ShortTermState
        Binary output state (float64 values 0.0 or 1.0).
    h : ShortTermState
        Persistent summed synaptic input.
    t_next : ShortTermState
        Next stochastic update time (only if ``stochastic_update=True``).

    Notes
    -----
    In NEST, binary-neuron communication encodes state transitions using spike
    multiplicity (double spike for up-transition, single spike for
    down-transition). Here, equivalent effects are represented through delta
    inputs added to :math:`h`.

    References
    ----------
    .. [1] Ginzburg I, Sompolinsky H (1994). Theory of correlations in
           stochastic neural networks. PRE 50(4):3171.
           DOI: https://doi.org/10.1103/PhysRevE.50.3171
    .. [2] McCulloch W, Pitts W (1943). A logical calculus of the ideas
           immanent in nervous activity. Bulletin of Mathematical Biophysics,
           5:115-133. DOI: https://doi.org/10.1007/BF02478259
    .. [3] Morrison A, Diesmann M (2007). Maintaining causality in discrete
           time neuronal simulations. Lectures in Supercomputational
           Neuroscience. DOI: https://doi.org/10.1007/978-3-540-73159-7_10
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        tau_m: ArrayLike = 10. * u.ms,
        theta: ArrayLike = 0. * u.mV,
        sigma: ArrayLike = 1. * u.mV,
        y_initializer: Callable = braintools.init.Constant(0.0),
        stochastic_update: bool = True,
        rng_seed: int = 0,
        name: str = None,
    ):
        super().__init__(in_size, name=name)

        self.tau_m = braintools.init.param(tau_m, self.varshape)
        if u.math.any(self.tau_m <= 0. * u.ms):
            raise ValueError('tau_m must be strictly positive.')

        self.theta = braintools.init.param(theta, self.varshape)
        self.sigma = braintools.init.param(sigma, self.varshape)
        self.y_initializer = y_initializer
        self.stochastic_update = stochastic_update
        self.rng_seed = int(rng_seed)

    def init_state(self, batch_size: int = None, **kwargs):
        shape = self.varshape if batch_size is None else (batch_size, *self.varshape)

        y = braintools.init.param(self.y_initializer, self.varshape, batch_size)
        self.y = brainstate.ShortTermState(u.math.asarray(y, dtype=jnp.float64))
        self.h = brainstate.ShortTermState(u.math.zeros(shape, dtype=jnp.float64) * u.mV)
        self.rng_key = brainstate.ShortTermState(jax.random.PRNGKey(self.rng_seed))

        if self.stochastic_update:
            exp0 = self._sample_exponential(self.y.value.shape)
            next_interval = exp0 * u.math.asarray(self.tau_m / u.ms, dtype=jnp.float64) * u.ms
            self.t_next = brainstate.ShortTermState(next_interval)

    def _sample_uniform(self, shape):
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.uniform(subkey, shape=shape, dtype=jnp.float64)

    def _sample_exponential(self, shape):
        key, subkey = jax.random.split(self.rng_key.value)
        self.rng_key.value = key
        return jax.random.exponential(subkey, shape=shape, dtype=jnp.float64)

    def _gain_probability(self, h):
        arg = -(h - self.theta) / (u.math.asarray(jnp.sqrt(2.0), dtype=jnp.float64) * self.sigma)
        return 0.5 * jspecial.erfc(u.math.asarray(arg, dtype=jnp.float64))

    def update(self, x=0. * u.mV):
        # NEST ordering: first integrate binary-event deltas into persistent h.
        delta_h = self.sum_delta_inputs(u.math.zeros_like(self.h.value))
        self.h.value = self.h.value + delta_h

        # Then include current input for this step in gain evaluation.
        c = self.sum_current_inputs(x, self.h.value)
        p = u.math.asarray(self._gain_probability(self.h.value + c), dtype=jnp.float64)

        if self.stochastic_update:
            t = brainstate.environ.get('t')
            dt = brainstate.environ.get_dt()
            current_time = t + dt
            should_update = current_time > self.t_next.value

            if bool(u.math.asarray(u.math.any(should_update))):
                u_rand = self._sample_uniform(self.y.value.shape)
                new_y = u.math.asarray(u_rand < p, dtype=jnp.float64)
                self.y.value = jax.lax.stop_gradient(u.math.where(should_update, new_y, self.y.value))

                next_interval = (
                    self._sample_exponential(self.y.value.shape)
                    * u.math.asarray(self.tau_m / u.ms, dtype=jnp.float64)
                    * u.ms
                )
                self.t_next.value = u.math.where(
                    should_update,
                    self.t_next.value + next_interval,
                    self.t_next.value
                )
        else:
            u_rand = self._sample_uniform(self.y.value.shape)
            self.y.value = jax.lax.stop_gradient(
                u.math.asarray(u_rand < p, dtype=jnp.float64)
            )

        return self.y.value
