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
from brainstate.typing import Size

from ._base import NESTNeuron

__all__ = [
    'rate_transformer_node',
]


class rate_transformer_node(NESTNeuron):
    r"""NEST-compatible ``rate_transformer_node`` template model.

    A stateless rate-based processing node that aggregates weighted incoming rate signals
    and applies a configurable input nonlinearity. Serves as an intermediary transformation
    stage in rate-based neural networks, mimicking NEST's ``rate_transformer_node<TNonlinearities>``
    template.

    **1. Mathematical Model**

    The model implements the transformation:

    .. math::

       X_i(t) = \phi\!\left(\sum_j w_{ij}\,\psi\!\left(X_j(t-d_{ij})\right)\right)

    where:

    - :math:`X_i(t)` is the output rate of node :math:`i` at time :math:`t`
    - :math:`X_j(t-d_{ij})` are incoming rates from presynaptic nodes :math:`j` with delay :math:`d_{ij}`
    - :math:`w_{ij}` are connection weights
    - :math:`\phi` is the input nonlinearity (applied to summed input)
    - :math:`\psi` is the output nonlinearity (applied per-event before summation)

    The model has **no intrinsic dynamics**: no differential equations, no noise, no membrane
    time constant. It acts purely as a feedforward transformation stage.

    **2. Nonlinearity Application Modes**

    The ``linear_summation`` parameter controls where the nonlinearity is applied, matching
    NEST's event handler semantics:

    **Mode A: ``linear_summation=True`` (default, recommended)**
      - Event handlers store weighted rates **without** applying nonlinearity
      - Nonlinearity :math:`\phi` is applied **once** to the summed input during ``update()``
      - Efficient when many inputs converge to one node
      - Math: :math:`X_i = \phi\!\left(\sum_j w_{ij} X_j(t-d_{ij})\right)`

    **Mode B: ``linear_summation=False``**
      - Nonlinearity :math:`\psi` is applied **per-event** before summation
      - Event handlers pre-transform each incoming rate
      - Useful for models where nonlinearity operates on individual inputs
      - Math: :math:`X_i = \sum_j w_{ij}\,\psi\!\left(X_j(t-d_{ij})\right)`

    **3. Default Nonlinearity**

    If no custom ``input_nonlinearity`` is provided, the model uses a simple gain function:

    .. math::

       \phi(h) = g \cdot h

    where :math:`g` is the ``g`` parameter (default 1.0).

    **4. Update Algorithm**

    dftype = brainstate.environ.dftype()
    ditype = brainstate.environ.ditype()
    The update sequence (matching NEST's ``rate_transformer_node_impl.h``) is:

    1. **Store delayed output**: Copy current ``rate`` → ``delayed_rate`` for outgoing delayed connections
    2. **Drain delayed queue**: Retrieve events scheduled for current timestep
    3. **Process delayed events**: Handle ``delayed_rate_events`` with explicit delay specifications
    4. **Process instant events**: Handle ``instant_rate_events`` (zero-delay)
    5. **Sum contributions**: Aggregate all weighted inputs into a single value
    6. **Apply nonlinearity** (if ``linear_summation=True``): Transform summed input
    7. **Update state**: Store new ``rate`` and ``instant_rate``
    8. **Increment step counter**: Advance internal timestep

    **5. Event Handling**

    The model accepts two types of runtime events in ``update()``:

    **Instant events (``instant_rate_events``)**:
      - Applied in the **current** timestep (zero delay)
      - Must not specify non-zero ``delay_steps`` (raises ``ValueError``)
      - Typical use: direct feedforward connections

    **Delayed events (``delayed_rate_events``)**:
      - Stored in internal queue and applied after ``delay_steps`` timesteps
      - Default delay is 1 step if not specified
      - Negative delays raise ``ValueError``

    **Event format** (flexible tuple/dict):
      - 2-tuple: ``(rate, weight)`` → uses default delay
      - 3-tuple: ``(rate, weight, delay_steps)``
      - 4-tuple: ``(rate, weight, delay_steps, multiplicity)``
      - dict: ``{'rate': ..., 'weight': ..., 'delay_steps': ..., 'multiplicity': ...}``
      - Scalar: interpreted as ``rate`` with ``weight=1.0, delay=default, multiplicity=1.0``

    **6. Latency Considerations**

    As in NEST, inserting a transformer node introduces **one simulation step of latency**
    compared to direct instantaneous connections. This is because:

    - Output ``instant_rate`` is updated at the **end** of the current timestep
    - Downstream nodes read this value in the **next** timestep
    - Even "instantaneous" connections have this inherent discrete-time delay

    **7. Computational Complexity**

    - **Time complexity**: :math:`O(E)` per timestep, where :math:`E` is the number of events
    - **Space complexity**: :math:`O(D \times N)` for delayed queue, where :math:`D` is max delay and :math:`N` is population size
    - **Nonlinearity cost**: :math:`O(N)` per call (applied once if ``linear_summation=True``, or :math:`E` times otherwise)

    Parameters
    ----------
    in_size : int, tuple of int, or Size
        Shape of the node population. Can be a scalar (1D population), tuple (multi-dimensional),
        or ``brainstate.Size`` object. Determines the shape of ``rate`` state variable.
    linear_summation : bool, optional
        Controls where the input nonlinearity is applied:

        - ``True`` (default): Apply nonlinearity **once** to summed input (efficient, recommended)
        - ``False``: Apply nonlinearity **per-event** before summation (mathematically different)

        See "Nonlinearity Application Modes" above for detailed semantics.
    g : float, array_like, optional
        Gain parameter for the default linear nonlinearity :math:`\phi(h) = g \cdot h`.
        Can be a scalar (shared across population) or array with shape matching ``in_size``.
        Ignored if custom ``input_nonlinearity`` is provided. Default: ``1.0`` (identity transform).
    input_nonlinearity : callable, optional
        Custom input nonlinearity function replacing the default :math:`g \cdot h`. Must accept:

        - Signature 1: ``f(h)`` where ``h`` is ndarray of summed inputs (shape ``in_size``)
        - Signature 2: ``f(model, h)`` where ``model`` is ``self`` (for accessing parameters)

        The function is automatically inspected to determine which signature to use.
        If ``None`` (default), uses ``g * h``. Return value must broadcast to ``in_size``.
    rate_initializer : callable, optional
        Initializer for the ``rate`` state variable. Must be a callable accepting ``(shape, batch_size)``
        and returning an array. Common choices:

        - ``braintools.init.Constant(0.0)`` (default): Initialize to zero
        - ``braintools.init.Normal(0.0, 0.1)``: Gaussian initialization
        - ``braintools.init.Uniform(0.0, 1.0)``: Uniform random

        Default: ``Constant(0.0)`` (all rates start at zero).
    name : str, optional
        Unique identifier for this module instance. Used for logging, debugging, and visualization.
        If ``None``, an auto-generated name is assigned. Default: ``None``.

    Parameter Mapping
    -----------------

    This table maps brainpy.state parameters to NEST's template instantiation:

    ================================  ================================  ================================
    brainpy.state parameter           NEST equivalent                   Notes
    ================================  ================================  ================================
    ``in_size``                       (implicit in node creation)       Population size
    ``linear_summation=True``         ``linear_summation=true``         Default mode in NEST
    ``linear_summation=False``        ``linear_summation=false``        Per-event nonlinearity
    ``g``                             (template parameter)              Gain in default nonlinearity
    ``input_nonlinearity``            ``TNonlinearities::input``        Custom :math:`\phi` or :math:`\psi`
    ``rate_initializer``              (no direct equiv)                 NEST defaults to 0.0
    ================================  ================================  ================================

    State Variables
    ---------------
    rate : ndarray, shape (in_size,)
        **Primary output rate** of the node. Updated at the end of each timestep after applying
        nonlinearity. This is the value read by downstream connections in the **next** timestep.
        Type: ``ShortTermState`` (persists between timesteps but not saved in checkpoints).
    instant_rate : ndarray, shape (in_size,)
        **Instantaneous output rate**, identical to ``rate`` after update. Provided for clarity
        in models that distinguish between instant and delayed outputs. Equals ``rate`` at the
        end of each timestep.
    delayed_rate : ndarray, shape (in_size,)
        **Previous timestep's output rate**, used for delayed outgoing connections. Equals
        ``rate`` from the **previous** timestep. Allows modeling connection delays explicitly.
    _step_count : int64 scalar
        Internal timestep counter used for delayed event scheduling. Increments by 1 each update.
        Not intended for external access.

    Receptor Types
    --------------
    The model exposes a single receptor type:

    - ``'RATE': 0`` — accepts rate-valued inputs (both instant and delayed)

    Use this in connection specifications to route rate signals to the transformer node.

    Recordables
    -----------
    The following state variables can be monitored during simulation:

    - ``'rate'`` — primary output rate (most commonly recorded)

    Notes
    -----
    **Comparison with NEST:**
      - Fully replicates NEST's ``rate_transformer_node_impl.h`` event handling logic
      - Supports all NEST template instantiations via custom ``input_nonlinearity``
      - Uses NumPy-based event queue instead of NEST's C++ ring buffer (functionally identical)
      - Batch processing is supported via ``init_state(batch_size=...)``

    **When to use this model:**
      - **Layered rate networks**: Inserting nonlinear transformations between rate-neuron populations
      - **Gain modulation**: Implementing attention or gating via spatially varying ``g``
      - **Modular architectures**: Separating rate dynamics (in rate neurons) from transformations (in transformer nodes)

    **Failure modes:**
      - Passing non-zero ``delay_steps`` in ``instant_rate_events`` raises ``ValueError``
      - Negative ``delay_steps`` in ``delayed_rate_events`` raises ``ValueError``
      - Custom ``input_nonlinearity`` returning wrong shape causes broadcasting errors

    **Performance tips:**
      - Use ``linear_summation=True`` (default) for better efficiency when many inputs converge
      - Avoid unnecessary delayed events if connections are instantaneous
      - For very long delays, consider pruning old queue entries manually if memory is constrained

    References
    ----------
    .. [1] Hahne, J., Dahmen, D., Schuecker, J., Frommer, A., Bolten, M., Helias, M.,
           & Diesmann, M. (2017). Integration of continuous-time dynamics in a spiking
           neural network simulator. *Frontiers in Neuroinformatics*, 11, 34.
           https://doi.org/10.3389/fninf.2017.00034

    .. [2] NEST Simulator. ``rate_transformer_node`` documentation.
           https://nest-simulator.readthedocs.io/en/stable/models/rate_transformer_node.html

    .. [3] NEST source code: ``rate_transformer_node_impl.h``
           https://github.com/nest/nest-simulator

    Examples
    --------
    **Example 1: Basic usage with default linear gain**

    .. code-block:: python

        >>> import brainpy_state as bst
        >>> import saiunit as u
        >>> import brainstate
        >>> # Create a transformer node with 10 units
        >>> transformer = bst.rate_transformer_node(in_size=10, g=2.0)
        >>> # Initialize state
        >>> transformer.init_state()
        >>> # Send instant rate events (rate=0.5, weight=1.0)
        >>> with brainstate.environ.context(dt=0.1 * u.ms):
        ...     output = transformer.update(instant_rate_events=(0.5, 1.0))
        >>> print(output)  # doctest: +SKIP
        [1.0, 1.0, ..., 1.0]  # g * rate * weight = 2.0 * 0.5 * 1.0

    **Example 2: Custom sigmoid nonlinearity**

    .. code-block:: python

        >>> import numpy as np
        >>> import brainpy_state as bst
        >>> # Define sigmoid activation
        >>> def sigmoid(h):
        ...     return 1.0 / (1.0 + np.exp(-h))
        >>> transformer = bst.rate_transformer_node(
        ...     in_size=5,
        ...     linear_summation=True,
        ...     input_nonlinearity=sigmoid
        ... )
        >>> transformer.init_state()
        >>> # High input drives to saturation
        >>> with brainstate.environ.context(dt=0.1 * u.ms):
        ...     output = transformer.update(instant_rate_events=(10.0, 1.0))
        >>> print(output)  # doctest: +SKIP
        [0.9999..., 0.9999..., ...]  # sigmoid(10) ≈ 1.0

    **Example 3: Delayed event scheduling**

    .. code-block:: python

        >>> import brainpy_state as bst
        >>> import saiunit as u
        >>> import brainstate
        >>> transformer = bst.rate_transformer_node(in_size=3)
        >>> transformer.init_state()
        >>> # Send delayed event (rate=1.0, weight=0.5, delay=2 steps)
        >>> with brainstate.environ.context(dt=1.0 * u.ms):
        ...     out_t0 = transformer.update(delayed_rate_events=(1.0, 0.5, 2))
        ...     out_t1 = transformer.update()  # Delayed event still in queue
        ...     out_t2 = transformer.update()  # Delayed event arrives now
        >>> print(out_t0, out_t1, out_t2)  # doctest: +SKIP
        [0., 0., 0.] [0., 0., 0.] [0.5, 0.5, 0.5]  # Event arrives at t+2

    **Example 4: Per-event nonlinearity (linear_summation=False)**

    .. code-block:: python

        >>> import brainpy_state as bst
        >>> import numpy as np
        >>> # ReLU applied per-event before summation
        >>> relu = lambda h: np.maximum(0, h)
        >>> transformer = bst.rate_transformer_node(
        ...     in_size=1,
        ...     linear_summation=False,
        ...     input_nonlinearity=relu
        ... )
        >>> transformer.init_state()
        >>> # Two events: one positive, one negative
        >>> events = [
        ...     (2.0, 1.0),   # ReLU(2.0) * 1.0 = 2.0
        ...     (-1.0, 1.0)   # ReLU(-1.0) * 1.0 = 0.0
        ... ]
        >>> with brainstate.environ.context(dt=0.1 * u.ms):
        ...     output = transformer.update(instant_rate_events=events)
        >>> print(output)  # doctest: +SKIP
        [2.0]  # Sum of per-event transformed values

    See Also
    --------
    rate_neuron_ipn : Rate neuron with intrinsic dynamics and input nonlinearity
    rate_neuron_opn : Rate neuron with output nonlinearity
    lin_rate : Linear rate neuron with dynamics
    sigmoid_rate : Sigmoid rate neuron
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        linear_summation: bool = True,
        g: float = 1.0,
        input_nonlinearity: Callable | None = None,
        rate_initializer: Callable = braintools.init.Constant(0.0),
        name: str = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.linear_summation = bool(linear_summation)
        self.g = braintools.init.param(g, self.varshape)
        self.input_nonlinearity = input_nonlinearity
        self.rate_initializer = rate_initializer

        self._delayed_queue = {}

    @property
    def recordables(self):
        return ['rate']

    @property
    def receptor_types(self):
        return {'RATE': 0}

    @staticmethod
    def _to_numpy(x):
        return np.asarray(u.math.asarray(x), dtype=dftype)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @staticmethod
    def _to_int_scalar(x, name: str):
        arr = np.asarray(u.math.asarray(x), dtype=dftype).reshape(-1)
        if arr.size != 1:
            raise ValueError(f'{name} must be scalar.')
        return int(arr[0])

    @staticmethod
    def _coerce_events(events):
        if events is None:
            return []
        if isinstance(events, dict):
            return [events]
        if isinstance(events, tuple):
            if len(events) == 0:
                return []
            if isinstance(events[0], (dict, tuple, list)):
                return list(events)
            if len(events) in (2, 3, 4):
                return [events]
        if isinstance(events, list):
            if len(events) == 0:
                return []
            if isinstance(events[0], (dict, tuple, list)):
                return events
            if len(events) in (2, 3, 4):
                return [tuple(events)]
        return [events]

    def _call_nl(self, fn: Callable, x: np.ndarray):
        try:
            return fn(self, x)
        except TypeError as first_error:
            try:
                return fn(x)
            except TypeError:
                raise first_error

    def _input_transform(self, h: np.ndarray, state_shape):
        h_np = self._broadcast_to_state(self._to_numpy(h), state_shape)
        if self.input_nonlinearity is None:
            g = self._broadcast_to_state(self._to_numpy(self.g), state_shape)
            return g * h_np
        y = self._call_nl(self.input_nonlinearity, h_np)
        return self._broadcast_to_state(self._to_numpy(y), state_shape)

    def _extract_event_fields(self, ev, default_delay_steps: int):
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

    def _event_to_weighted_value(self, ev, default_delay_steps: int, state_shape):
        rate, weight, multiplicity, delay_steps = self._extract_event_fields(ev, default_delay_steps)

        rate_np = self._broadcast_to_state(self._to_numpy(rate), state_shape)
        weight_np = self._broadcast_to_state(self._to_numpy(weight), state_shape)
        multiplicity_np = self._broadcast_to_state(self._to_numpy(multiplicity), state_shape)

        if self.linear_summation:
            weighted_value = rate_np * weight_np * multiplicity_np
        else:
            weighted_value = self._input_transform(rate_np, state_shape) * weight_np * multiplicity_np

        return weighted_value, delay_steps

    @staticmethod
    def _queue_add(queue: dict, step_idx: int, value: np.ndarray):
        if step_idx in queue:
            queue[step_idx] = queue[step_idx] + value
        else:
            queue[step_idx] = np.array(value, dtype=dftype, copy=True)

    def _drain_delayed_queue(self, step_idx: int, state_shape):
        value = self._delayed_queue.pop(step_idx, None)
        if value is None:
            return np.zeros(state_shape, dtype=dftype)
        return np.array(self._broadcast_to_state(np.asarray(value, dtype=dftype), state_shape), copy=True)

    def _accumulate_instant_events(self, events, state_shape):
        total = np.zeros(state_shape, dtype=dftype)
        for ev in self._coerce_events(events):
            value, delay_steps = self._event_to_weighted_value(
                ev,
                default_delay_steps=0,
                state_shape=state_shape,
            )
            if delay_steps != 0:
                raise ValueError('instant_rate_events must not specify non-zero delay_steps.')
            total += value
        return total

    def _schedule_delayed_events(self, events, step_idx: int, state_shape):
        total_now = np.zeros(state_shape, dtype=dftype)
        for ev in self._coerce_events(events):
            value, delay_steps = self._event_to_weighted_value(
                ev,
                default_delay_steps=1,
                state_shape=state_shape,
            )
            if delay_steps < 0:
                raise ValueError('delay_steps for delayed_rate_events must be >= 0.')
            if delay_steps == 0:
                total_now += value
            else:
                self._queue_add(self._delayed_queue, step_idx + delay_steps, value)
        return total_now

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all state variables and reset the delayed event queue.

        Allocates ``rate``, ``instant_rate``, ``delayed_rate``, and internal timestep counter.
        Clears any pre-existing delayed events from the internal queue. Must be called before
        the first ``update()`` invocation.

        Parameters
        ----------
        batch_size : int, optional
            Number of parallel simulation batches. If provided, state variables will have shape
            ``(batch_size, *in_size)``. If ``None`` (default), shape is ``in_size``.
        **kwargs
            Additional keyword arguments (ignored, present for API consistency).

        Notes
        -----
        All state variables are initialized using the ``rate_initializer`` provided during
        construction. The delayed event queue (``_delayed_queue``) is reset to an empty dict,
        discarding any events scheduled in previous simulations.

        This method is typically called automatically by ``brainstate`` infrastructure, but can
        be invoked manually to reset the model to initial conditions.
        """
        rate = braintools.init.param(self.rate_initializer, self.varshape, batch_size)
        rate_np = self._to_numpy(rate)

        self.rate = brainstate.ShortTermState(rate_np)
        self.instant_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))
        self.delayed_rate = brainstate.ShortTermState(np.array(rate_np, dtype=dftype, copy=True))
        self._step_count = brainstate.ShortTermState(np.asarray(0, dtype=ditype))

        self._delayed_queue = {}

    def update(self, x=0.0, instant_rate_events=None, delayed_rate_events=None):
        r"""Execute one timestep of the rate transformation algorithm.

        Processes incoming rate events (both instant and delayed), applies the configured
        nonlinearity, and updates the output ``rate`` state variable. Implements the NEST
        ``rate_transformer_node_impl.h`` update sequence.

        Parameters
        ----------
        x : float, array_like, optional
            External input current (ignored). Present for API compatibility with ``Dynamics``
            base class, but rate transformers have no intrinsic current-driven dynamics.
            Default: ``0.0``.
        instant_rate_events : None, tuple, list of tuples, or dict, optional
            **Zero-delay rate events** applied in the current timestep. Event format:

            - **2-tuple**: ``(rate, weight)`` — uses ``delay_steps=0`` (enforced)
            - **3-tuple**: ``(rate, weight, delay_steps)`` — ``delay_steps`` **must** be 0
            - **4-tuple**: ``(rate, weight, delay_steps, multiplicity)`` — ``delay_steps`` **must** be 0
            - **dict**: ``{'rate': r, 'weight': w, 'delay_steps': d, 'multiplicity': m}`` — ``d`` **must** be 0
            - **list/tuple of above**: multiple events processed sequentially
            - **None** (default): no instant events

            Raises ``ValueError`` if any event specifies non-zero ``delay_steps``.
        delayed_rate_events : None, tuple, list of tuples, or dict, optional
            **Delayed rate events** stored in internal queue and applied after specified delay.
            Event format:

            - **2-tuple**: ``(rate, weight)`` — uses default ``delay_steps=1``
            - **3-tuple**: ``(rate, weight, delay_steps)`` — custom delay
            - **4-tuple**: ``(rate, weight, delay_steps, multiplicity)``
            - **dict**: ``{'rate': r, 'weight': w, 'delay_steps': d, 'multiplicity': m}``
            - **list/tuple of above**: multiple events
            - **None** (default): no delayed events

            - ``delay_steps=0``: event applied immediately (equivalent to instant event)
            - ``delay_steps=d > 0``: event applied after ``d`` timesteps
            - Negative ``delay_steps`` raises ``ValueError``

        Returns
        -------
        rate_new : ndarray, shape ``(in_size,)`` or ``(batch_size, *in_size)``
            **Updated output rate** after applying nonlinearity to aggregated inputs. This
            is the new value of the ``rate`` state variable. Shape matches ``in_size``
            (or ``(batch_size, *in_size)`` if batch mode was used in ``init_state()``).

        Raises
        ------
        ValueError
            If ``instant_rate_events`` contains any event with non-zero ``delay_steps``.
        ValueError
            If ``delayed_rate_events`` contains any event with negative ``delay_steps``.
        ValueError
            If event tuples have invalid length (must be 2, 3, or 4).
        ValueError
            If ``delay_steps`` is not a scalar value.

        Notes
        -----
        **Update algorithm (step-by-step)**:
          1. Store current ``rate`` → ``delayed_rate`` (for outgoing delayed connections)
          2. Increment internal ``_step_count``
          3. Drain events scheduled for current timestep from ``_delayed_queue``
          4. Process new ``delayed_rate_events``:

             - Events with ``delay_steps=0`` are applied immediately
             - Events with ``delay_steps>0`` are added to ``_delayed_queue``

          5. Process ``instant_rate_events`` (all applied immediately)
          6. Sum all contributions:

             - If ``linear_summation=True``: sum weighted rates, then apply nonlinearity once
             - If ``linear_summation=False``: apply nonlinearity per-event, then sum

          7. Update ``rate`` and ``instant_rate`` with the new value

        **Event semantics**:
          - **rate**: Input rate value (can be scalar or array matching ``in_size``)
          - **weight**: Connection weight (scalar or array matching ``in_size``)
          - **delay_steps**: Integer delay in timesteps (0 = immediate, >0 = delayed)
          - **multiplicity**: Event count multiplier (default 1.0, rarely used)
          - Effective contribution: ``rate * weight * multiplicity`` (before nonlinearity)

        **Broadcasting rules**:
          - All event fields (``rate``, ``weight``, ``multiplicity``) broadcast to ``in_size``
          - Scalars are replicated across all nodes
          - Arrays must have compatible shapes (standard NumPy broadcasting)

        **Memory management**:
          - Delayed events are stored in ``_delayed_queue`` until their scheduled timestep
          - Queue entries are automatically removed after retrieval (no memory leak)
          - Queue persists across ``update()`` calls but is cleared by ``init_state()``
        """
        del x  # NEST rate transformer has no intrinsic current input.

        state_shape = self.rate.value.shape
        step_idx = int(np.asarray(self._step_count.value, dtype=ditype).reshape(-1)[0])

        delayed_total = self._drain_delayed_queue(step_idx, state_shape)
        delayed_total += self._schedule_delayed_events(
            delayed_rate_events,
            step_idx=step_idx,
            state_shape=state_shape,
        )
        instant_total = self._accumulate_instant_events(
            instant_rate_events,
            state_shape=state_shape,
        )

        rate_prev = self._broadcast_to_state(self._to_numpy(self.rate.value), state_shape)
        if self.linear_summation:
            rate_new = self._input_transform(delayed_total + instant_total, state_shape)
        else:
            rate_new = delayed_total + instant_total

        self.rate.value = rate_new
        self.delayed_rate.value = rate_prev
        self.instant_rate.value = rate_new
        self._step_count.value = np.asarray(step_idx + 1, dtype=ditype)
        return rate_new
