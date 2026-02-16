# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Union, Optional, Sequence, Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size, DTypeLike

from brainpy_state._misc import set_module_as

__all__ = [
    'SpikeTime',
    'PoissonSpike',
    'PoissonEncoder',
    'PoissonInput',
    'poisson_input',
]


class SpikeTime(brainstate.nn.Dynamics):
    """The input neuron group characterized by spikes emitting at given times.

    Internally builds a ``brainevent.CSR`` matrix of shape
    ``[n_max_time_step, n_neuron]`` so that ``update()`` reduces to a
    single CSR row slice.

    >>> # Get 2 neurons, firing spikes at 10 ms and 20 ms.
    >>> SpikeTime(2, times=[10, 20])
    >>> # or
    >>> # Get 2 neurons, the neuron 0 fires spikes at 10 ms and 20 ms.
    >>> SpikeTime(2, times=[10, 20], indices=[0, 0])
    >>> # or
    >>> # Get 2 neurons, neuron 0 fires at 10 ms and 30 ms, neuron 1 fires at 20 ms.
    >>> SpikeTime(2, times=[10, 20, 30], indices=[0, 1, 0])
    >>> # or
    >>> # Get 2 neurons; at 10 ms, neuron 0 fires; at 20 ms, neuron 0 and 1 fire;
    >>> # at 30 ms, neuron 1 fires.
    >>> SpikeTime(2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])
    >>> # or
    >>> # Get 2 neurons with a uniform float weight for all spikes.
    >>> SpikeTime(2, times=[10, 20], indices=[0, 1], weights=2.5)
    >>> # or
    >>> # Get 2 neurons with per-event weights.
    >>> SpikeTime(2, times=[10, 20], indices=[0, 1], weights=[0.5, 0.8])

    Parameters
    ----------
    in_size : int, tuple, list
        The neuron group geometry.
    indices : list, tuple, ArrayType
        The neuron indices at each time point to emit spikes.
    times : list, tuple, ArrayType
        The time points which generate the spikes.
    weights : bool, float, Sequence, or ArrayLike, default=True
        Spike weights. ``True`` produces boolean output (backward-compatible).
        A float scalar applies the same weight to every spike. A sequence of
        the same length as ``indices``/``times`` assigns per-event weights.
        The output dtype is inferred from this parameter.
    time_as_step : callable, str, default='round'
        Rounding method for converting spike times to integer step indices.
        Options: ``'floor'``, ``'round'``, ``'ceil'``.
    name : str, optional
        The name of the dynamic system.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        indices: Union[Sequence, ArrayLike],
        times: Union[Sequence, ArrayLike],
        weights: Union[float, Sequence, ArrayLike] = 1.0,
        time_as_step: str | Callable = 'round',
        name: Optional[str] = None,
    ):
        super().__init__(in_size=in_size, name=name)

        # parameters
        if len(indices) != len(times):
            raise ValueError(f'The length of "indices" and "times" must be the same. '
                             f'However, we got {len(indices)} != {len(times)}.')
        if callable(time_as_step):
            self.time_as_step = time_as_step
        else:
            if time_as_step not in ('floor', 'round', 'ceil'):
                raise ValueError(f'"time_as_step" must be one of "floor", "round", "ceil". '
                                 f'Got {time_as_step!r}.')
        self.num_times = len(times)
        self.time_as_step = time_as_step

        # weights
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        weights = u.math.asarray(weights, dtype=dftype)
        self._scalar_weight = (weights.ndim == 0)
        self._spk_dtype = weights.dtype
        if not self._scalar_weight and len(weights) != self.num_times:
            raise ValueError(
                f'Length of "weights" must match "indices"/"times". '
                f'Got {len(weights)} != {self.num_times}.'
            )

        # data about times and indices
        times = u.math.asarray(times, dtype=dftype)
        indices = u.math.asarray(indices, dtype=ditype)
        if self._scalar_weight:
            self.weights = weights
            self.times, self.indices = u.lax.sort((times, indices), dimension=0)
        else:
            sorted_arrays = u.lax.sort((times, indices, weights), dimension=0)
            self.times = sorted_arrays[0]
            self.indices = sorted_arrays[1]
            self.weights = sorted_arrays[2]

        # Build CSR matrix [n_max_time_step, n_neuron]
        import brainevent

        assert len(self.varshape) == 1, 'SpikeTime only supports 1D neuron groups for now.'
        n_cols = self.varshape[-1]

        dt = brainstate.environ.get_dt()
        ratio = self.times / dt
        if isinstance(ratio, u.Quantity):
            ratio = ratio.to_decimal()

        steps = jnp.asarray(self.rounding(ratio), dtype=ditype)
        indices = jnp.asarray(self.indices, dtype=ditype)

        if self.num_times == 0:
            return

        # Build CSR indptr
        n_rows = int(steps.max()) + 1
        counts = np.bincount(np.asarray(steps), minlength=n_rows)
        indptr_np = np.zeros(n_rows + 1, dtype=ditype)
        np.cumsum(counts, out=indptr_np[1:])

        self._csr = brainevent.CSR(
            self.weights,
            jax.numpy.asarray(indices, dtype=ditype),
            jax.numpy.asarray(indptr_np, dtype=ditype),
            shape=(n_rows, n_cols)
        )

    @property
    def rounding(self):
        if callable(self.time_as_step):
            return self.time_as_step
        elif self.time_as_step == 'floor':
            return u.math.floor
        elif self.time_as_step == 'round':
            return u.math.round
        else:
            return u.math.ceil

    def _to_int(self, ratio):
        if isinstance(ratio, u.Quantity):
            ratio = ratio.to_decimal()
        return u.math.asarray(self.rounding(ratio), dtype=brainstate.environ.ditype())

    def update(self, index=None, time=None):
        if self.num_times == 0:
            return jax.numpy.zeros(self.varshape, dtype=self._spk_dtype)

        if index is not None:
            i = jax.numpy.asarray(index, dtype=brainstate.environ.ditype())
        elif time is not None:
            i = self._to_int(time / brainstate.environ.get_dt())
        else:
            # Both None — read from brainstate.environ
            i = brainstate.environ.get('i', default=None)
            if i is None:
                t = brainstate.environ.get('t')
                dt = brainstate.environ.get_dt()
                i = self._to_int(t / dt)
            else:
                i = jax.numpy.asarray(i, dtype=brainstate.environ.ditype())

        # Slice CSR row → dense spike vector of shape (n_cols,)
        # Reshape to 1D for csr_slice_rows, then squeeze back
        return self._csr[jnp.asarray(i, dtype=jnp.int32).reshape(1)][0]


class PoissonSpike(brainstate.nn.Dynamics):
    """
    Poisson Neuron Group.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        freqs: Union[ArrayLike, Callable],
        spk_type: DTypeLike = bool,
        name: Optional[str] = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.spk_type = spk_type

        # parameters
        self.freqs = braintools.init.param(freqs, self.varshape, allow_none=False)

    def update(self):
        spikes = brainstate.random.rand(*self.varshape) <= (self.freqs * brainstate.environ.get_dt())
        spikes = u.math.asarray(spikes, dtype=self.spk_type)
        return spikes


class PoissonEncoder(brainstate.nn.Dynamics):
    r"""Poisson spike encoder for converting firing rates to spike trains.

    This class implements a Poisson process to generate spikes based on provided
    firing rates. Unlike the PoissonSpike class, this encoder accepts firing rates
    as input during the update step rather than having them fixed at initialization.

    The spike generation follows a Poisson process where the probability of a spike
    in each time step is proportional to the firing rate and the simulation time step:

    $$
    P(\text{spike}) = \text{rate} \cdot \text{dt}
    $$

    For each neuron and time step, the encoder draws a random number from a uniform
    distribution [0,1] and generates a spike if the number is less than or equal to
    the spiking probability.

    Parameters
    ----------
    in_size : Size
        Size of the input to the encoder, defining the shape of the output spike train.
    spk_type : DTypeLike, default=bool
        Data type for the generated spikes. Typically boolean for binary spikes.
    name : str, optional
        Name of the encoder brainstate.nn.Module.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>> import numpy as np
    >>>
    >>> # Create a Poisson encoder for 10 neurons
    >>> encoder = brainpy.state.PoissonEncoder(10)
    >>>
    >>> # Generate spikes with varying firing rates
    >>> rates = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) * u.Hz
    >>> spikes = encoder.update(rates)
    >>>
    >>> # Use in a more complex processing pipeline
    >>> # First, generate rate-coded output from an analog signal
    >>> analog_values = np.random.rand(10) * 100  # values between 0 and 100
    >>> firing_rates = analog_values * u.Hz  # convert to firing rates
    >>> spike_train = encoder.update(firing_rates)
    >>>
    >>> # Feed the spikes into a spiking neural network
    >>> neuron_layer = brainpy.state.LIF(10)
    >>> neuron_layer.init_state(batch_size=1)
    >>> output_spikes = neuron_layer.update(spike_train)

    Notes
    -----
    - This encoder is particularly useful for rate-to-spike conversion in neuromorphic
      computing applications and sensory encoding tasks.
    - The statistical properties of the generated spike trains follow a Poisson process,
      where the inter-spike intervals are exponentially distributed.
    - For small time steps (dt), the number of spikes in a time window T approximately
      follows a Poisson distribution with parameter λ = rate * T.
    - Unlike PoissonSpike which has fixed rates, this encoder allows dynamic rate changes
      with every update call, making it suitable for encoding time-varying signals.
    - The independence of spike generation between time steps results in renewal process
      statistics without memory of previous spiking history.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        spk_type: DTypeLike = bool,
        name: Optional[str] = None,
    ):
        super().__init__(in_size=in_size, name=name)
        self.spk_type = spk_type

    def update(self, freqs: ArrayLike):
        spikes = brainstate.random.rand(*self.varshape) <= (freqs * brainstate.environ.get_dt())
        spikes = u.math.asarray(spikes, dtype=self.spk_type)
        return spikes


class PoissonInput(brainstate.nn.Module):
    r"""Poisson Input to the given state variable.

    This class provides a way to add independent Poisson-distributed spiking input
    to a target state variable. For large numbers of inputs, this implementation is
    computationally more efficient than creating separate Poisson spike generators.

    The synaptic events are generated randomly during simulation runtime and are not
    preloaded or stored in memory, which improves memory efficiency for large-scale
    simulations. All inputs target the same variable with the same frequency and
    synaptic weight.

    The Poisson process generates spikes with probability based on the frequency and
    simulation time step:

    $$
    P(\text{spike}) = \text{freq} \cdot \text{dt}
    $$

    For computational efficiency, two different methods are used for spike generation:

    1. For large numbers of inputs, a normal approximation:
       $$
       \text{inputs} \sim \mathcal{N}(\mu, \sigma^2)
       $$
       where $\mu = \text{num\_input} \cdot p$ and $\sigma^2 = \text{num\_input} \cdot p \cdot (1-p)$

    2. For smaller numbers, a direct binomial sampling:
       $$
       \text{inputs} \sim \text{Binomial}(\text{num\_input}, p)
       $$

    where $p = \text{freq} \cdot \text{dt}$ in both cases.

    Parameters
    ----------
    target : brainstate.nn.Prefetch
        The variable that is targeted by this input. Should be an instance of
        :py:class:`brainstate.State` that's brainstate.nn.Prefetched via the target mechanism.
    indices : Union[np.ndarray, jax.Array]
        Indices of the target to receive input. If None, input is applied to the entire target.
    num_input : int
        The number of independent Poisson input sources.
    freq : Union[int, float]
        The firing frequency of each input source in Hz.
    weight :  ndarray, float, or brainunit.Quantity
        The synaptic weight of each input spike.
    name : Optional[str], optional
        The name of this brainstate.nn.Module.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>> import numpy as np
    >>>
    >>> # Create a neuron group with membrane potential
    >>> neuron = brainpy.state.LIF(100)
    >>> neuron.init_state(batch_size=1)
    >>>
    >>> # Add Poisson input to all neurons
    >>> poisson_in = brainpy.state.PoissonInput(
    ...     target=neuron.V,
    ...     indices=None,
    ...     num_input=200,
    ...     freq=50 * u.Hz,
    ...     weight=0.1 * u.mV
    ... )
    >>>
    >>> # Add Poisson input only to specific neurons
    >>> indices = np.array([0, 10, 20, 30])
    >>> specific_input = brainpy.state.PoissonInput(
    ...     target=neuron.V,
    ...     indices=indices,
    ...     num_input=50,
    ...     freq=100 * u.Hz,
    ...     weight=0.2 * u.mV
    ... )
    >>>
    >>> # Run simulation with the inputs
    >>> for t in range(100):
    ...     poisson_in.update()
    ...     specific_input.update()
    ...     neuron.update()

    Notes
    -----
    - The Poisson inputs are statistically independent between update steps and across
      target neurons.
    - This implementation is particularly efficient for large numbers of inputs or targets.
    - For very sparse connectivity patterns, consider using individual PoissonSpike neurons
      with specific connectivity patterns instead.
    - The update method internally calls the poisson_input function which handles the
      spike generation and target state updates.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        target: brainstate.nn.Prefetch,
        indices: Union[np.ndarray, jax.Array],
        num_input: int,
        freq: u.Quantity[u.Hz],
        weight: Union[jax.typing.ArrayLike, u.Quantity],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.target = target
        self.indices = indices
        self.num_input = num_input
        self.freq = freq
        self.weight = weight

    def update(self):
        target_state = getattr(self.target.module, self.target.item)

        # generate Poisson input
        poisson_input(
            self.freq,
            self.num_input,
            self.weight,
            target_state,
            self.indices,
        )


@set_module_as('brainpy.state')
def poisson_input(
    freq: u.Quantity[u.Hz],
    num_input: int,
    weight: Union[jax.typing.ArrayLike, u.Quantity],
    target: brainstate.State,
    indices: Optional[Union[np.ndarray, jax.Array]] = None,
    refractory: Optional[Union[jax.Array]] = None,
):
    r"""Generates Poisson-distributed input spikes to a target state variable.

    This function simulates Poisson input to a given state, updating the target
    variable with generated spikes based on the specified frequency, number of inputs,
    and synaptic weight. The input can be applied to specific indices of the target
    or to the entire target if indices are not provided.

    The function uses two different methods to generate the Poisson-distributed input:
    1. For large numbers of inputs (a > 5 and b > 5), a normal approximation is used
    2. For smaller numbers, a direct binomial sampling approach is used

    Mathematical model for Poisson input:
    $$
    P(\text{spike}) = \text{freq} \cdot \text{dt}
    $$

    For the normal approximation (when a > 5 and b > 5):
    $$
    \text{inputs} \sim \mathcal{N}(a, b \cdot p)
    $$
    where:
    $$
    a = \text{num\_input} \cdot p
    $$
    $$
    b = \text{num\_input} \cdot (1 - p)
    $$
    $$
    p = \text{freq} \cdot \text{dt}
    $$

    For direct binomial sampling (when a ≤ 5 or b ≤ 5):
    $$
    \text{inputs} \sim \text{Binomial}(\text{num\_input}, p)
    $$

    Parameters
    ----------
    freq : u.Quantity[u.Hz]
        The frequency of the Poisson input in Hertz.
    num_input : int
        The number of input channels or neurons generating the Poisson spikes.
    weight : u.Quantity
        The synaptic weight applied to each spike.
    target : State
        The target state variable to which the Poisson input is applied.
    indices : Optional[Union[np.ndarray, jax.Array]], optional
        Specific indices of the target to apply the input. If None, the input is applied
        to the entire target.
    refractory : Optional[Union[jax.Array]], optional
        A boolean array indicating which parts of the target are in a refractory state
        and should not be updated. Should be the same length as the target.

    Examples
    --------
    >>> import brainpy
    >>> import brainstate
    >>> import brainunit as u
    >>> import numpy as np
    >>>
    >>> # Create a membrane potential state
    >>> V = brainstate.HiddenState(np.zeros(100) * u.mV)
    >>>
    >>> # Add Poisson input to all neurons at 50 Hz
    >>> brainpy.state.poisson_input(
    ...     freq=50 * u.Hz,
    ...     num_input=200,
    ...     weight=0.1 * u.mV,
    ...     target=V
    ... )
    >>>
    >>> # Apply Poisson input only to a subset of neurons
    >>> indices = np.array([0, 10, 20, 30])
    >>> brainpy.state.poisson_input(
    ...     freq=100 * u.Hz,
    ...     num_input=50,
    ...     weight=0.2 * u.mV,
    ...     target=V,
    ...     indices=indices
    ... )
    >>>
    >>> # Apply input with refractory mask
    >>> refractory = np.zeros(100, dtype=bool)
    >>> refractory[40:60] = True  # neurons 40-59 are in refractory period
    >>> brainpy.state.poisson_input(
    ...     freq=75 * u.Hz,
    ...     num_input=100,
    ...     weight=0.15 * u.mV,
    ...     target=V,
    ...     refractory=refractory
    ... )

    Notes
    -----
    - The function automatically switches between normal approximation and binomial
      sampling based on the input parameters to optimize computation efficiency.
    - For large numbers of inputs, the normal approximation provides significant
      performance improvements.
    - The weight parameter is applied uniformly to all generated spikes.
    - When refractory is provided, the corresponding target elements are not updated.
    """
    freq = brainstate.maybe_state(freq)
    weight = brainstate.maybe_state(weight)

    assert isinstance(target, brainstate.State), 'The target must be a State.'
    p = freq * brainstate.environ.get_dt()
    p = p.to_decimal() if isinstance(p, u.Quantity) else p
    a = num_input * p
    b = num_input * (1 - p)
    tar_val = target.value
    cond = u.math.logical_and(a > 5, b > 5)

    if indices is None:
        # generate Poisson input
        branch1 = jax.tree.map(
            lambda tar: brainstate.random.normal(
                a,
                b * p,
                tar.shape,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity
        )
        branch2 = jax.tree.map(
            lambda tar: brainstate.random.binomial(
                num_input,
                p,
                tar.shape,
                check_valid=False,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity,
        )

        inp = jax.tree.map(
            lambda b1, b2: u.math.where(cond, b1, b2),
            branch1,
            branch2,
            is_leaf=u.math.is_quantity,
        )

        # update target variable
        data = jax.tree.map(
            lambda tar, x: tar + x * weight,
            target.value,
            inp,
            is_leaf=u.math.is_quantity
        )

    else:
        # generate Poisson input
        branch1 = jax.tree.map(
            lambda tar: brainstate.random.normal(
                a,
                b * p,
                tar[indices].shape,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity
        )
        branch2 = jax.tree.map(
            lambda tar: brainstate.random.binomial(
                num_input,
                p,
                tar[indices].shape,
                check_valid=False,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity
        )

        inp = jax.tree.map(
            lambda b1, b2: u.math.where(cond, b1, b2),
            branch1,
            branch2,
            is_leaf=u.math.is_quantity,
        )

        # update target variable
        data = jax.tree.map(
            lambda x, tar: tar.at[indices].add(x * weight),
            inp,
            tar_val,
            is_leaf=u.math.is_quantity
        )

    if refractory is not None:
        target.value = jax.tree.map(
            lambda x, tar: u.math.where(refractory, tar, x),
            data,
            tar_val,
            is_leaf=u.math.is_quantity
        )
    else:
        target.value = data
