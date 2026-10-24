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

from collections.abc import Sequence, Callable

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

__all__ = [
    'SpikeTime',
]


class SpikeTime(brainstate.nn.Dynamics):
    r"""The input neuron group characterized by spikes emitting at given times.

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

    See Also
    --------
    PoissonSpike : Stochastic Poisson spike generator.
    PoissonEncoder : Poisson encoder with dynamic rates.

    Notes
    -----
    - Internally, a ``brainevent.CSR`` sparse matrix of shape
      ``(n_max_time_step, n_neuron)`` is pre-built at construction, so
      :meth:`update` reduces to a single CSR row slice.
    - Only 1-D neuron groups are supported.
    - Spike times are converted to integer step indices using the rounding
      method specified by ``time_as_step``.
    """
    __module__ = 'brainspike'

    def __init__(
        self,
        in_size: Size,
        indices: Sequence | ArrayLike,
        times: Sequence | ArrayLike,
        weights: float | Sequence | ArrayLike = 1.0,
        time_as_step: str | Callable = 'round',
        name: str | None = None,
    ):
        super().__init__(in_size=in_size, name=name)

        # parameters
        if len(indices) != len(times):
            raise ValueError(f'The length of "indices" and "times" must be the same. '
                             f'However, we got {len(indices)} != {len(times)}.')
        if not callable(time_as_step):
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
        # Reshape to 1D for csr_slice_rows, then squeeze back.
        # Steps outside [0, n_rows) carry no events: clamp the gather index to a
        # valid row, then mask the result back to zeros. This keeps behaviour
        # identical in eager mode (no IndexError) and under JIT (no silent clamp).
        ditype = brainstate.environ.ditype()
        i = jnp.asarray(i, dtype=ditype)
        n_rows = self._csr.shape[0]
        row = self._csr[jnp.clip(i, 0, n_rows - 1).reshape(1)][0]
        return jnp.where(
            (i >= 0) & (i < n_rows),
            row,
            jnp.zeros(self.varshape, dtype=self._spk_dtype),
        )
