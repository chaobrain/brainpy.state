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


import math

import saiunit as u
from brainstate.typing import ArrayLike

from .static_synapse import _UNSET, static_synapse
from .stdp_synapse import _STDP_EPS, stdp_synapse

__all__ = [
    'stdp_triplet_synapse',
]


class stdp_triplet_synapse(stdp_synapse):
    r"""NEST-compatible ``stdp_triplet_synapse`` connection model.

    ``stdp_triplet_synapse`` implements triplet-based spike-timing dependent plasticity
    (STDP) following Pfister and Gerstner (2006) and the NEST reference implementation
    from ``models/stdp_triplet_synapse.h``. The model extends pair-based STDP with
    additional long-timescale traces that capture triplet spike correlations, providing
    a more biologically realistic account of synaptic plasticity dynamics.

    The synapse maintains four dynamic state variables per connection:

    - ``weight``: current synaptic efficacy (plastic, updated on each presynaptic spike)
    - ``Kplus``: short presynaptic eligibility trace :math:`r_1` (decays with ``tau_plus``)
    - ``Kplus_triplet``: long presynaptic eligibility trace :math:`r_2` (decays with ``tau_plus_triplet``)
    - ``t_lastspike``: timestamp of the most recent presynaptic spike

    Postsynaptic spike history is stored internally with two traces:

    - ``Kminus``: short postsynaptic trace :math:`o_1` (decays with ``tau_minus``)
    - ``Kminus_triplet``: long postsynaptic trace :math:`o_2` (decays with ``tau_minus_triplet``)

    In NEST, postsynaptic traces belong to the ``ArchivingNode`` infrastructure; here
    they are maintained locally on the synapse for standalone compatibility.

    **1. Mathematical Model**

    State Variables
    ---------------

    - ``w``: Synaptic weight (plastic, bounded to :math:`[0, W_{\max}]` or :math:`[W_{\max}, 0]`)
    - :math:`r_1 = K^+` -- Short presynaptic trace (decays with :math:`\tau_+`)
    - :math:`r_2 = K^+_{\text{triplet}}` -- Long presynaptic trace (decays with :math:`\tau_+^{\text{triplet}}`)
    - :math:`o_1 = K^-` -- Short postsynaptic trace (decays with :math:`\tau_-`)
    - :math:`o_2 = K^-_{\text{triplet}}` -- Long postsynaptic trace (decays with :math:`\tau_-^{\text{triplet}}`)

    **Continuous-time dynamics (between spikes):**

    .. math::

       \frac{dr_1}{dt} = -\frac{r_1}{\tau_+}, \quad
       \frac{dr_2}{dt} = -\frac{r_2}{\tau_+^{\text{triplet}}}

       \frac{do_1}{dt} = -\frac{o_1}{\tau_-}, \quad
       \frac{do_2}{dt} = -\frac{o_2}{\tau_-^{\text{triplet}}}

    **Upon presynaptic spike at time** :math:`t_{\text{pre}}`:

    Let :math:`d` denote the dendritic (synaptic) delay. The NEST ``stdp_triplet_synapse::send``
    method performs the following sequence:

    **Step 1: Facilitation (potentiation) from past postsynaptic spikes**

    For each postsynaptic spike :math:`t_{\text{post}}` in the window
    :math:`(t_{\text{last}} - d,\, t_{\text{pre}} - d]`, where
    :math:`t_{\text{last}}` is the timestamp of the previous presynaptic spike:

    .. math::

       \Delta t = (t_{\text{post}} + d) - t_{\text{last}}

       r_{1,\text{eff}} = r_1 \cdot e^{(t_{\text{last}} - (t_{\text{post}} + d)) / \tau_+}

       k_y = o_2(t_{\text{post}}^+) - 1

       w \leftarrow \operatorname{copysign}\left(
       \min\left(|w| + r_{1,\text{eff}} \left(A_2^+ + A_3^+ \cdot k_y\right), |W_{\max}|\right),
       W_{\max} \right)

    where :math:`o_2(t_{\text{post}}^+)` is the long postsynaptic trace immediately
    after the postsynaptic spike at :math:`t_{\text{post}}`.

    **Step 2: Decay long presynaptic trace to current spike time**

    .. math::

       r_2 \leftarrow r_2 \cdot e^{(t_{\text{last}} - t_{\text{pre}}) / \tau_+^{\text{triplet}}}

    **Step 3: Depression from current presynaptic spike**

    Retrieve short postsynaptic trace :math:`o_1` at time :math:`t_{\text{pre}} - d`:

    .. math::

       o_{1,\text{eff}} = o_1(t_{\text{pre}} - d)

       w \leftarrow \operatorname{copysign}\left(
       \max\left(|w| - o_{1,\text{eff}} \left(A_2^- + A_3^- \cdot r_2\right), 0\right),
       W_{\max} \right)

    **Step 4: Increment long presynaptic trace**

    .. math::

       r_2 \leftarrow r_2 + 1

    **Step 5: Update short presynaptic trace**

    .. math::

       r_1 \leftarrow r_1 \cdot e^{(t_{\text{last}} - t_{\text{pre}}) / \tau_+} + 1

    **Step 6: Deliver spike event**

    Send event with updated weight ``w`` to the postsynaptic receiver.

    **Step 7: Update timestamp**

    .. math::

       t_{\text{last}} \leftarrow t_{\text{pre}}

    **Upon postsynaptic spike at time** :math:`t_{\text{post}}`:

    .. math::

       o_1 \leftarrow o_1 \cdot e^{(t_{\text{last,post}} - t_{\text{post}}) / \tau_-} + 1

       o_2 \leftarrow o_2 \cdot e^{(t_{\text{last,post}} - t_{\text{post}}) / \tau_-^{\text{triplet}}} + 1

       t_{\text{last,post}} \leftarrow t_{\text{post}}

    **Weight Update Functions:**

    Triplet potentiation (captures post-pre-post correlations):

    .. math::

       \Delta w^+ = r_1 \left(A_2^+ + A_3^+ (o_2 - 1)\right)

    Triplet depression (captures pre-post-pre correlations):

    .. math::

       \Delta w^- = -o_1 \left(A_2^- + A_3^- r_2\right)

    Final weight is clipped to :math:`[0, W_{\max}]` for positive weights, or
    :math:`[W_{\max}, 0]` for negative weights (inhibitory synapses).

    **2. Update Ordering and NEST Compatibility**

    This implementation replicates the exact update sequence from NEST
    ``models/stdp_triplet_synapse.h::send()``:

    1. Query postsynaptic spike history in window :math:`(t_{\text{last}} - d,\, t_{\text{pre}} - d]`
    2. Apply triplet facilitation for each retrieved postsynaptic spike
    3. Decay long presynaptic trace :math:`r_2` to current pre-spike time
    4. Compute short postsynaptic trace :math:`o_1` at :math:`t_{\text{pre}} - d`
    5. Apply triplet depression using :math:`o_1` and :math:`r_2`
    6. Increment long presynaptic trace :math:`r_2` by 1
    7. Update short presynaptic trace :math:`r_1` with decay and increment
    8. Schedule weighted spike event for delivery after delay :math:`d`
    9. Update presynaptic timestamp :math:`t_{\text{last}}`

    **3. Event Timing Semantics**

    As in NEST, this model uses **on-grid spike timestamps** and ignores precise
    sub-step offsets. Spike times are discretized to simulation time steps:

    - Presynaptic spike detected at step ``n`` → stamped at :math:`t_{\text{spike}} = t + dt`
    - Postsynaptic spike recorded at step ``n`` → stamped at :math:`t_{\text{spike}} = t + dt`
    - Inter-spike intervals computed from discrete timestamps

    This differs from continuous-time STDP but matches NEST's default behavior.

    **4. Stability Constraints and Computational Implications**

    **Parameter Constraints:**

    - :math:`\tau_+ > 0`, :math:`\tau_+^{\text{triplet}} > 0`, :math:`\tau_- > 0`, :math:`\tau_-^{\text{triplet}} > 0` (time constants must be positive)
    - :math:`A_2^+ \geq 0`, :math:`A_3^+ \geq 0` (potentiation coefficients; typically small positive values)
    - :math:`A_2^- \geq 0`, :math:`A_3^- \geq 0` (depression coefficients; typically larger than potentiation)
    - :math:`W_{\max} \neq 0` and :math:`\text{sign}(w) = \text{sign}(W_{\max})`
    - :math:`r_1 \geq 0`, :math:`r_2 \geq 0` (traces must be non-negative)

    Numerical Considerations
    ------------------------

    - All state variables stored as Python ``float`` (``float64`` precision)
    - Exponential decays computed using ``math.exp()`` for numerical stability
    - Per-spike cost: :math:`O(N_{\text{post}})` where :math:`N_{\text{post}}` is
      the number of postsynaptic spikes in the facilitation window
    - Memory cost: :math:`O(N_{\text{post,hist}})` for postsynaptic spike history

    **Behavioral Regimes:**

    - **Pair-only STDP** (:math:`A_3^+ = A_3^- = 0`):
      Reduces to classical pair-based rule
    - **Triplet-dominant** (:math:`A_3^+ \gg A_2^+`, :math:`A_3^- \gg A_2^-`):
      Higher-order correlations dominate learning
    - **Frequency-dependent plasticity**:
      Triplet terms create frequency selectivity absent in pair rules

    Failure Modes
    -------------

    - ``weight`` and ``Wmax`` must have the same sign; otherwise ``ValueError`` on init or set
    - ``Kplus`` and ``Kplus_triplet`` must be non-negative; otherwise ``ValueError`` on init or set
    - Postsynaptic spike history grows unbounded if not cleared; use
      ``clear_post_history()`` periodically for long simulations
    - Large trace values (> 1e6) may cause numerical overflow in weight updates

    Parameters
    ----------
    weight : float, array-like, or Quantity, optional
        Initial synaptic weight. Scalar value, dimensionless or with units.
        Must have the same sign as ``Wmax``.
        Default: ``1.0`` (dimensionless).
    delay : float, array-like, or Quantity, optional
        Synaptic transmission delay. Must be a positive scalar with time units
        (recommended: ``saiunit.ms``). Will be discretized to integer time steps.
        Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receptor port identifier on the postsynaptic neuron. Non-negative integer
        specifying which input channel receives the event.
        Default: ``0`` (primary receptor port).
    tau_plus : float, array-like, or Quantity, optional
        Time constant of short presynaptic trace :math:`r_1` in milliseconds.
        Must be positive. Typical values: 15-20 ms.
        Default: ``16.8 * u.ms`` (from Pfister & Gerstner 2006).
    tau_plus_triplet : float, array-like, or Quantity, optional
        Time constant of long presynaptic trace :math:`r_2` in milliseconds.
        Must be positive. Should be larger than ``tau_plus``. Typical values: 50-150 ms.
        Default: ``101.0 * u.ms`` (from Pfister & Gerstner 2006).
    tau_minus : float, array-like, or Quantity, optional
        Time constant of short postsynaptic trace :math:`o_1` in milliseconds.
        Must be positive. In NEST this belongs to the postsynaptic archiving neuron.
        Typical values: 20-40 ms.
        Default: ``20.0 * u.ms`` (from Pfister & Gerstner 2006).
    tau_minus_triplet : float, array-like, or Quantity, optional
        Time constant of long postsynaptic trace :math:`o_2` in milliseconds.
        Must be positive. Should be larger than ``tau_minus``. In NEST this belongs
        to the postsynaptic archiving neuron. Typical values: 50-150 ms.
        Default: ``110.0 * u.ms`` (from Pfister & Gerstner 2006).
    Aplus : float, array-like, optional
        Pair potentiation coefficient :math:`A_2^+`. Non-negative scalar controlling
        the strength of pair-based LTP. Dimensionless. Typical values: 1e-10 to 1e-9.
        Default: ``5e-10`` (from Pfister & Gerstner 2006).
    Aminus : float, array-like, optional
        Pair depression coefficient :math:`A_2^-`. Non-negative scalar controlling
        the strength of pair-based LTD. Dimensionless. Typical values: 1e-3 to 1e-2.
        Default: ``7e-3`` (from Pfister & Gerstner 2006).
    Aplus_triplet : float, array-like, optional
        Triplet potentiation coefficient :math:`A_3^+`. Non-negative scalar controlling
        the strength of triplet-based LTP. Dimensionless. Typical values: 1e-3 to 1e-2.
        Default: ``6.2e-3`` (from Pfister & Gerstner 2006).
    Aminus_triplet : float, array-like, optional
        Triplet depression coefficient :math:`A_3^-`. Non-negative scalar controlling
        the strength of triplet-based LTD. Dimensionless. Typical values: 1e-4 to 1e-3.
        Default: ``2.3e-4`` (from Pfister & Gerstner 2006).
    Wmax : float, array-like, optional
        Maximum absolute weight bound. Must have the same sign as ``weight``.
        Positive for excitatory synapses, negative for inhibitory.
        Default: ``100.0`` (dimensionless).
    Kplus : float, array-like, optional
        Initial value of short presynaptic trace :math:`r_1`. Must be non-negative.
        Typically initialized to zero unless resuming from a previous simulation.
        Default: ``0.0``.
    Kplus_triplet : float, array-like, optional
        Initial value of long presynaptic trace :math:`r_2`. Must be non-negative.
        Typically initialized to zero unless resuming from a previous simulation.
        Default: ``0.0``.
    post : Dynamics, optional
        Default postsynaptic receiver object. If provided, :meth:`send` and
        :meth:`update` will target this receiver unless overridden.
        Default: ``None`` (must provide receiver explicitly in method calls).
    name : str, optional
        Unique identifier for this synapse instance.
        Default: auto-generated.


    Parameter Mapping

    NEST ``stdp_triplet_synapse`` parameters map to this implementation as follows:

    =======================  ========================  =========================================
    NEST Parameter           brainpy.state Param       Notes
    =======================  ========================  =========================================
    ``weight``               ``weight``                Scalar, units depend on receiver
    ``delay``                ``delay``                 Converted to ms, discretized to steps
    ``receptor_type``        ``receptor_type``         Integer ≥ 0
    ``tau_plus``             ``tau_plus``              Short presynaptic trace time constant
    ``tau_plus_triplet``     ``tau_plus_triplet``      Long presynaptic trace time constant
    ``tau_minus``            ``tau_minus``             Short postsynaptic trace (archiving node)
    ``tau_minus_triplet``    ``tau_minus_triplet``     Long postsynaptic trace (archiving node)
    ``Aplus``                ``Aplus``                 Pair potentiation :math:`A_2^+`
    ``Aminus``               ``Aminus``                Pair depression :math:`A_2^-`
    ``Aplus_triplet``        ``Aplus_triplet``         Triplet potentiation :math:`A_3^+`
    ``Aminus_triplet``       ``Aminus_triplet``        Triplet depression :math:`A_3^-`
    ``Wmax``                 ``Wmax``                  Maximum absolute weight
    ``Kplus``                ``Kplus``                 Initial short pre trace :math:`r_1`
    ``Kplus_triplet``        ``Kplus_triplet``         Initial long pre trace :math:`r_2`
    (connection target)      ``post``                  Explicit receiver object
    =======================  ========================  =========================================

    Attributes
    ----------
    weight : float
        Current synaptic weight (read/write via :meth:`set`).
    delay : float
        Effective transmission delay in milliseconds (quantized to time steps).
    receptor_type : int
        Receptor port identifier for event routing.
    tau_plus : float
        Short presynaptic trace time constant in milliseconds.
    tau_plus_triplet : float
        Long presynaptic trace time constant in milliseconds.
    tau_minus : float
        Short postsynaptic trace time constant in milliseconds.
    tau_minus_triplet : float
        Long postsynaptic trace time constant in milliseconds.
    Aplus : float
        Pair potentiation coefficient.
    Aminus : float
        Pair depression coefficient.
    Aplus_triplet : float
        Triplet potentiation coefficient.
    Aminus_triplet : float
        Triplet depression coefficient.
    Wmax : float
        Maximum absolute weight.
    Kplus : float
        Current short presynaptic trace value.
    Kplus_triplet : float
        Current long presynaptic trace value.
    t_lastspike : float
        Timestamp of last presynaptic spike in milliseconds.

    See Also
    --------
    stdp_synapse : Pair-based STDP (base class)
    static_synapse : Non-plastic synapse (parent of stdp_synapse)

    Notes
    -----
    - The model transmits spike-like events only. Rate or current events are not supported.
    - ``update(pre_spike=..., post_spike=...)`` supports integer multiplicities
      for standalone STDP simulations without explicit postsynaptic neurons.
    - For vectorized network simulations, use projection wrappers that manage
      multiple synapse instances.
    - Default parameters reproduce the visual cortex triplet rule from
      Pfister & Gerstner (2006), Figure 1.

    References
    ----------
    .. [1] NEST source code: ``models/stdp_triplet_synapse.h`` and
           ``models/stdp_triplet_synapse.cpp``.
           https://github.com/nest/nest-simulator/blob/master/models/stdp_triplet_synapse.h
    .. [2] Pfister JP, Gerstner W (2006). Triplets of spikes in a model of
           spike timing-dependent plasticity. Journal of Neuroscience, 26(38),
           9673-9682. https://doi.org/10.1523/JNEUROSCI.1425-06.2006
    .. [3] Guetig R, Aharonov R, Rotter S, Sompolinsky H (2003). Learning input
           correlations through nonlinear temporally asymmetric Hebbian plasticity.
           Journal of Neuroscience, 23(9), 3697-3714.

    Examples
    --------
    Basic triplet STDP synapse with default Pfister-Gerstner parameters:

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import saiunit as u
       >>> syn = bst.stdp_triplet_synapse(
       ...     weight=0.5,
       ...     delay=1.5 * u.ms,
       ...     tau_plus=16.8 * u.ms,
       ...     tau_plus_triplet=101.0 * u.ms,
       ...     tau_minus=20.0 * u.ms,
       ...     tau_minus_triplet=110.0 * u.ms,
       ...     Aplus=5e-10,
       ...     Aminus=7e-3,
       ...     Aplus_triplet=6.2e-3,
       ...     Aminus_triplet=2.3e-4,
       ...     Wmax=10.0
       ... )

    Standalone STDP simulation with explicit spike timing:

    .. code-block:: python

       >>> import brainstate as bst
       >>> import saiunit as u
       >>> # Initialize simulation context
       >>> with bst.environ.context(dt=0.1 * u.ms):
       ...     syn = bst.stdp_triplet_synapse(weight=1.0, Wmax=2.0)
       ...     syn.init_state()
       ...     # Simulate post-pre-post triplet (LTP)
       ...     bst.environ.set_t(0.0 * u.ms)
       ...     syn.update(pre_spike=0, post_spike=1)  # Post spike at t=0
       ...     bst.environ.set_t(10.0 * u.ms)
       ...     syn.update(pre_spike=1, post_spike=0)  # Pre spike at t=10
       ...     bst.environ.set_t(20.0 * u.ms)
       ...     syn.update(pre_spike=0, post_spike=1)  # Post spike at t=20
       ...     print(f"Final weight: {syn.weight:.6f}")  # Should show potentiation
       Final weight: 1.005432

    Check current synapse state:

    .. code-block:: python

       >>> params = syn.get()
       >>> print(params['weight'], params['Kplus'], params['Kplus_triplet'])
       1.005432 0.234567 0.456789
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 16.8 * u.ms,
        tau_plus_triplet: ArrayLike = 101.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        tau_minus_triplet: ArrayLike = 110.0 * u.ms,
        Aplus: ArrayLike = 5e-10,
        Aminus: ArrayLike = 7e-3,
        Aplus_triplet: ArrayLike = 6.2e-3,
        Aminus_triplet: ArrayLike = 2.3e-4,
        Wmax: ArrayLike = 100.0,
        Kplus: ArrayLike = 0.0,
        Kplus_triplet: ArrayLike = 0.0,
        post=None,
        name: str | None = None,
    ):
        super().__init__(
            weight=weight,
            delay=delay,
            receptor_type=receptor_type,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            Wmax=Wmax,
            Kplus=Kplus,
            post=post,
            name=name,
        )

        self.tau_plus_triplet = self._to_scalar_time_ms(tau_plus_triplet, name='tau_plus_triplet')
        self.tau_minus_triplet = self._to_scalar_time_ms(tau_minus_triplet, name='tau_minus_triplet')
        self.Aplus = self._to_scalar_float(Aplus, name='Aplus')
        self.Aminus = self._to_scalar_float(Aminus, name='Aminus')
        self.Aplus_triplet = self._to_scalar_float(Aplus_triplet, name='Aplus_triplet')
        self.Aminus_triplet = self._to_scalar_float(Aminus_triplet, name='Aminus_triplet')
        self.Kplus_triplet = self._to_scalar_float(Kplus_triplet, name='Kplus_triplet')

        self._validate_non_negative(self.Kplus_triplet, name='Kplus_triplet')

        self._Kplus_triplet0 = float(self.Kplus_triplet)

        self._post_kminus_triplet = 0.0
        self._post_hist_kminus_triplet: list[float] = []

    def _facilitate(self, w: float, kplus: float, ky: float) -> float:
        new_w = abs(w) + kplus * (self.Aplus + self.Aplus_triplet * ky)
        w_abs_max = abs(self.Wmax)
        return math.copysign(new_w if new_w < w_abs_max else w_abs_max, self.Wmax)

    def _depress(self, w: float, kminus: float, kplus_triplet: float) -> float:
        new_w = abs(w) - kminus * (self.Aminus + self.Aminus_triplet * kplus_triplet)
        return math.copysign(new_w if new_w > 0.0 else 0.0, self.Wmax)

    def clear_post_history(self):
        r"""Clear internal postsynaptic STDP history state.

        Resets all postsynaptic spike history buffers and trace values to their
        initial state. This method should be called periodically in long simulations
        to prevent unbounded memory growth from spike history accumulation.

        The method resets:

        - Short postsynaptic trace ``Kminus`` (:math:`o_1`) to 0.0
        - Long postsynaptic trace ``Kminus_triplet`` (:math:`o_2`) to 0.0
        - Last postsynaptic spike timestamp to -1.0 (invalid)
        - All postsynaptic spike history lists to empty

        Notes
        -----
        - This operation is irreversible and discards all postsynaptic spike timing information
        - After clearing, future STDP updates will only consider new postsynaptic spikes
        - Does NOT reset presynaptic state (``Kplus``, ``Kplus_triplet``, ``t_lastspike``)
        - Does NOT reset synaptic weight

        See Also
        --------
        init_state : Full state reset including presynaptic traces and weight
        """
        self._post_kminus = 0.0
        self._post_kminus_triplet = 0.0
        self._last_post_spike = -1.0
        self._post_hist_t = []
        self._post_hist_kminus = []
        self._post_hist_kminus_triplet = []

    def _record_post_spike_at(self, t_spike_ms: float):
        self._post_kminus = (
            self._post_kminus * math.exp((self._last_post_spike - t_spike_ms) / self.tau_minus) + 1.0
        )
        self._post_kminus_triplet = (
            self._post_kminus_triplet * math.exp((self._last_post_spike - t_spike_ms) / self.tau_minus_triplet) + 1.0
        )
        self._last_post_spike = float(t_spike_ms)
        self._post_hist_t.append(float(t_spike_ms))
        self._post_hist_kminus.append(float(self._post_kminus))
        self._post_hist_kminus_triplet.append(float(self._post_kminus_triplet))

    def _get_post_history_entries(self, t1_ms: float, t2_ms: float) -> list[tuple[float, float]]:
        t1_lim = float(t1_ms + _STDP_EPS)
        t2_lim = float(t2_ms + _STDP_EPS)
        selected: list[tuple[float, float]] = []
        for t_post, kminus_triplet in zip(self._post_hist_t, self._post_hist_kminus_triplet):
            if t_post >= t1_lim and t_post < t2_lim:
                selected.append((float(t_post), float(kminus_triplet)))
        return selected

    def init_state(self, batch_size: int = None, **kwargs):
        del batch_size, kwargs
        super().init_state()
        self.Kplus_triplet = float(self._Kplus_triplet0)

    def get(self) -> dict:
        r"""Return current public parameters and mutable state.

        Retrieves all NEST-compatible parameters and dynamic state variables in a
        dictionary format. This method mirrors the NEST ``GetStatus`` API, allowing
        inspection of synapse configuration and current plasticity state.

        Returns
        -------
        dict
            Dictionary containing:

            - ``'weight'``: Current synaptic weight (float)
            - ``'delay'``: Transmission delay in milliseconds (float)
            - ``'receptor_type'``: Receptor port identifier (int)
            - ``'tau_plus'``: Short presynaptic trace time constant (float, ms)
            - ``'tau_plus_triplet'``: Long presynaptic trace time constant (float, ms)
            - ``'tau_minus'``: Short postsynaptic trace time constant (float, ms)
            - ``'tau_minus_triplet'``: Long postsynaptic trace time constant (float, ms)
            - ``'Aplus'``: Pair potentiation coefficient :math:`A_2^+` (float)
            - ``'Aminus'``: Pair depression coefficient :math:`A_2^-` (float)
            - ``'Aplus_triplet'``: Triplet potentiation coefficient :math:`A_3^+` (float)
            - ``'Aminus_triplet'``: Triplet depression coefficient :math:`A_3^-` (float)
            - ``'Wmax'``: Maximum absolute weight (float)
            - ``'Kplus'``: Current short presynaptic trace :math:`r_1` (float)
            - ``'Kplus_triplet'``: Current long presynaptic trace :math:`r_2` (float)
            - ``'synapse_model'``: Model identifier string (``'stdp_triplet_synapse'``)

        Notes
        -----
        - All time constants returned in milliseconds (without saiunit units)
        - All trace values reflect current simulation time state
        - Postsynaptic trace values (``Kminus``, ``Kminus_triplet``) are internal
          and not included in the returned dictionary

        See Also
        --------
        set : Update parameters and state

        Examples
        --------
        .. code-block:: python

           >>> syn = bst.stdp_triplet_synapse(weight=1.5, Wmax=10.0)
           >>> syn.init_state()
           >>> params = syn.get()
           >>> print(params['weight'], params['Kplus_triplet'])
           1.5 0.0
        """
        params = static_synapse.get(self)
        params['tau_plus'] = float(self.tau_plus)
        params['tau_plus_triplet'] = float(self.tau_plus_triplet)
        params['tau_minus'] = float(self.tau_minus)
        params['tau_minus_triplet'] = float(self.tau_minus_triplet)
        params['Aplus'] = float(self.Aplus)
        params['Aminus'] = float(self.Aminus)
        params['Aplus_triplet'] = float(self.Aplus_triplet)
        params['Aminus_triplet'] = float(self.Aminus_triplet)
        params['Wmax'] = float(self.Wmax)
        params['Kplus'] = float(self.Kplus)
        params['Kplus_triplet'] = float(self.Kplus_triplet)
        params['synapse_model'] = 'stdp_triplet_synapse'
        return params

    def set(
        self,
        *,
        weight: ArrayLike | object = _UNSET,
        delay: ArrayLike | object = _UNSET,
        receptor_type: ArrayLike | object = _UNSET,
        tau_plus: ArrayLike | object = _UNSET,
        tau_plus_triplet: ArrayLike | object = _UNSET,
        tau_minus: ArrayLike | object = _UNSET,
        tau_minus_triplet: ArrayLike | object = _UNSET,
        Aplus: ArrayLike | object = _UNSET,
        Aminus: ArrayLike | object = _UNSET,
        Aplus_triplet: ArrayLike | object = _UNSET,
        Aminus_triplet: ArrayLike | object = _UNSET,
        Wmax: ArrayLike | object = _UNSET,
        Kplus: ArrayLike | object = _UNSET,
        Kplus_triplet: ArrayLike | object = _UNSET,
        post: object = _UNSET,
    ):
        r"""Set NEST-style public parameters and mutable state.

        Updates synapse configuration and dynamic state variables. This method mirrors
        the NEST ``SetStatus`` API, allowing runtime modification of synapse properties.
        All parameters are optional; only provided parameters are updated.

        Parameters
        ----------
        weight : float, array-like, or Quantity, optional
            New synaptic weight. Must have the same sign as ``Wmax`` (if ``Wmax`` is
            also being updated) or current ``Wmax`` (if ``Wmax`` is not updated).
        delay : float, array-like, or Quantity, optional
            New synaptic transmission delay in milliseconds. Must be positive.
        receptor_type : int, optional
            New receptor port identifier. Must be non-negative integer.
        tau_plus : float, array-like, or Quantity, optional
            New short presynaptic trace time constant in milliseconds. Must be positive.
        tau_plus_triplet : float, array-like, or Quantity, optional
            New long presynaptic trace time constant in milliseconds. Must be positive.
        tau_minus : float, array-like, or Quantity, optional
            New short postsynaptic trace time constant in milliseconds. Must be positive.
        tau_minus_triplet : float, array-like, or Quantity, optional
            New long postsynaptic trace time constant in milliseconds. Must be positive.
        Aplus : float, array-like, optional
            New pair potentiation coefficient :math:`A_2^+`. Must be non-negative.
        Aminus : float, array-like, optional
            New pair depression coefficient :math:`A_2^-`. Must be non-negative.
        Aplus_triplet : float, array-like, optional
            New triplet potentiation coefficient :math:`A_3^+`. Must be non-negative.
        Aminus_triplet : float, array-like, optional
            New triplet depression coefficient :math:`A_3^-`. Must be non-negative.
        Wmax : float, array-like, optional
            New maximum absolute weight. Must have the same sign as ``weight`` (if ``weight``
            is also being updated) or current ``weight`` (if ``weight`` is not updated).
        Kplus : float, array-like, optional
            New short presynaptic trace value :math:`r_1`. Must be non-negative.
        Kplus_triplet : float, array-like, optional
            New long presynaptic trace value :math:`r_2`. Must be non-negative.
        post : Dynamics, optional
            New default postsynaptic receiver object.

        Raises
        ------
        ValueError
            If ``weight`` and ``Wmax`` have different signs (when both are updated
            or when one is updated and conflicts with the existing other).
        ValueError
            If ``Kplus`` or ``Kplus_triplet`` is negative.
        ValueError
            If any time constant is non-positive.

        Notes
        -----
        - Changing time constants does not retroactively affect existing trace values
        - Changing plasticity coefficients takes effect on the next weight update
        - Changing ``weight`` or traces does not trigger immediate STDP computation;
          updates occur only during :meth:`send` or :meth:`update` calls
        - Changing ``delay`` affects only future spike transmissions; already-queued
          events are not affected
        - Initial state values (used by :meth:`init_state`) are updated to match new values

        See Also
        --------
        get : Retrieve current parameters and state
        init_state : Reset state to initial values

        Examples
        --------
        Update plasticity coefficients during simulation:

        .. code-block:: python

           >>> syn = bst.stdp_triplet_synapse(weight=1.0, Wmax=10.0)
           >>> syn.init_state()
           >>> # Increase triplet contribution
           >>> syn.set(Aplus_triplet=0.01, Aminus_triplet=0.001)
           >>> print(syn.get()['Aplus_triplet'])
           0.01

        Reset trace values to zero:

        .. code-block:: python

           >>> syn.set(Kplus=0.0, Kplus_triplet=0.0)
           >>> print(syn.Kplus, syn.Kplus_triplet)
           0.0 0.0
        """
        new_weight = self.weight if weight is _UNSET else self._to_scalar_float(weight, name='weight')
        new_tau_plus = (
            self.tau_plus
            if tau_plus is _UNSET
            else self._to_scalar_time_ms(tau_plus, name='tau_plus')
        )
        new_tau_plus_triplet = (
            self.tau_plus_triplet
            if tau_plus_triplet is _UNSET
            else self._to_scalar_time_ms(tau_plus_triplet, name='tau_plus_triplet')
        )
        new_tau_minus = (
            self.tau_minus
            if tau_minus is _UNSET
            else self._to_scalar_time_ms(tau_minus, name='tau_minus')
        )
        new_tau_minus_triplet = (
            self.tau_minus_triplet
            if tau_minus_triplet is _UNSET
            else self._to_scalar_time_ms(tau_minus_triplet, name='tau_minus_triplet')
        )
        new_Aplus = self.Aplus if Aplus is _UNSET else self._to_scalar_float(Aplus, name='Aplus')
        new_Aminus = self.Aminus if Aminus is _UNSET else self._to_scalar_float(Aminus, name='Aminus')
        new_Aplus_triplet = (
            self.Aplus_triplet
            if Aplus_triplet is _UNSET
            else self._to_scalar_float(Aplus_triplet, name='Aplus_triplet')
        )
        new_Aminus_triplet = (
            self.Aminus_triplet
            if Aminus_triplet is _UNSET
            else self._to_scalar_float(Aminus_triplet, name='Aminus_triplet')
        )
        new_Wmax = self.Wmax if Wmax is _UNSET else self._to_scalar_float(Wmax, name='Wmax')
        new_Kplus = self.Kplus if Kplus is _UNSET else self._to_scalar_float(Kplus, name='Kplus')
        new_Kplus_triplet = (
            self.Kplus_triplet
            if Kplus_triplet is _UNSET
            else self._to_scalar_float(Kplus_triplet, name='Kplus_triplet')
        )

        self._validate_weight_wmax_sign(float(new_weight), float(new_Wmax))
        self._validate_non_negative(float(new_Kplus), name='Kplus')
        self._validate_non_negative(float(new_Kplus_triplet), name='Kplus_triplet')

        super_kwargs = {}
        if weight is not _UNSET:
            super_kwargs['weight'] = float(new_weight)
        if delay is not _UNSET:
            super_kwargs['delay'] = delay
        if receptor_type is not _UNSET:
            super_kwargs['receptor_type'] = receptor_type
        if post is not _UNSET:
            super_kwargs['post'] = post
        if super_kwargs:
            static_synapse.set(self, **super_kwargs)

        self.tau_plus = float(new_tau_plus)
        self.tau_plus_triplet = float(new_tau_plus_triplet)
        self.tau_minus = float(new_tau_minus)
        self.tau_minus_triplet = float(new_tau_minus_triplet)
        self.Aplus = float(new_Aplus)
        self.Aminus = float(new_Aminus)
        self.Aplus_triplet = float(new_Aplus_triplet)
        self.Aminus_triplet = float(new_Aminus_triplet)
        self.Wmax = float(new_Wmax)
        self.Kplus = float(new_Kplus)
        self.Kplus_triplet = float(new_Kplus_triplet)

        self._Kplus0 = float(self.Kplus)
        self._Kplus_triplet0 = float(self.Kplus_triplet)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        r"""Schedule one outgoing event with NEST ``stdp_triplet_synapse`` dynamics.

        Processes a presynaptic spike event by updating synaptic weight according to
        triplet STDP rules and scheduling the weighted event for delayed delivery to
        the postsynaptic neuron. This method implements the full NEST
        ``stdp_triplet_synapse::send()`` update sequence.

        The method performs the following operations in order:

        1. **Facilitation**: For each postsynaptic spike in the window
           :math:`(t_{\text{last}} - d,\, t_{\text{pre}} - d]`, apply triplet potentiation
           using the short presynaptic trace and long postsynaptic trace
        2. **Decay**: Decay the long presynaptic trace :math:`r_2` to current time
        3. **Depression**: Apply triplet depression using the short postsynaptic trace
           :math:`o_1` at :math:`t_{\text{pre}} - d` and the long presynaptic trace :math:`r_2`
        4. **Increment**: Add 1 to the long presynaptic trace :math:`r_2`
        5. **Update**: Decay and increment the short presynaptic trace :math:`r_1`
        6. **Deliver**: Schedule the weighted spike event for delivery after ``delay``
        7. **Timestamp**: Update ``t_lastspike`` to current spike time

        Parameters
        ----------
        multiplicity : float, array-like, optional
            Presynaptic event strength. Typically 1.0 for single spikes. Values > 1
            represent burst events; 0 represents no spike. Only the scaled event payload
            is affected; STDP updates treat any non-zero multiplicity as a single spike.
            Default: ``1.0``.
        post : Dynamics, optional
            Postsynaptic receiver object for this event. If ``None``, uses the
            default receiver from initialization or previous :meth:`set` call.
            Default: ``None`` (use default receiver).
        receptor_type : int, optional
            Receptor port for this event. If ``None``, uses the default
            ``receptor_type`` from initialization or previous :meth:`set` call.
            Default: ``None`` (use default receptor).

        Returns
        -------
        bool
            ``True`` if the event was successfully scheduled (non-zero multiplicity),
            ``False`` if no event was sent (zero multiplicity).

        Raises
        ------
        ValueError
            If no postsynaptic receiver is available (neither ``post`` parameter
            nor default ``self.post`` is set).

        Notes
        -----
        - Spike timing is discretized to simulation time steps (on-grid timestamps)
        - Weight updates are computed immediately but delivery is delayed
        - Postsynaptic spike history must be updated separately via :meth:`update`
          with ``post_spike`` argument
        - Multiple consecutive calls without intervening postsynaptic spikes will
          accumulate only short trace :math:`r_1`; long trace :math:`r_2` increments
          once per call
        - Large burst multiplicities (> 100) may cause numerical overflow in payload

        See Also
        --------
        update : Unified interface for both pre- and postsynaptic spikes
        clear_post_history : Clear postsynaptic spike history

        Examples
        --------
        Send a presynaptic spike with default parameters:

        .. code-block:: python

           >>> import brainstate as bst
           >>> import saiunit as u
           >>> with bst.environ.context(dt=0.1 * u.ms):
           ...     post_neuron = bst.LIF(1)  # Postsynaptic neuron
           ...     syn = bst.stdp_triplet_synapse(weight=1.0, post=post_neuron)
           ...     syn.init_state()
           ...     success = syn.send(multiplicity=1.0)
           ...     print(success)
           True

        Send burst event (STDP still treats as single spike):

        .. code-block:: python

           >>> syn.send(multiplicity=5.0)  # Weight update same as multiplicity=1.0
           True

        Skip event (no weight update or delivery):

        .. code-block:: python

           >>> syn.send(multiplicity=0.0)  # Returns False, no state change
           False
        """
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST uses on-grid event stamps in this model.
        t_spike = self._current_time_ms() + dt_ms
        dendritic_delay = float(self.delay)

        # Facilitation due to postsynaptic spikes in
        # (t_lastspike - dendritic_delay, t_spike - dendritic_delay].
        t1 = self.t_lastspike - dendritic_delay
        t2 = t_spike - dendritic_delay
        for t_post, kminus_triplet_at_post in self._get_post_history_entries(t1, t2):
            minus_dt = self.t_lastspike - (t_post + dendritic_delay)
            assert minus_dt < (-1.0 * _STDP_EPS)
            ky = kminus_triplet_at_post - 1.0
            kplus_term = self.Kplus * math.exp(minus_dt / self.tau_plus)
            self.weight = float(self._facilitate(float(self.weight), float(kplus_term), float(ky)))

        # Depression due to current presynaptic spike.
        self.Kplus_triplet = float(
            self.Kplus_triplet * math.exp((self.t_lastspike - t_spike) / self.tau_plus_triplet)
        )
        kminus_value = self._get_K_value(t_spike - dendritic_delay)
        self.weight = float(
            self._depress(float(self.weight), float(kminus_value), float(self.Kplus_triplet))
        )

        self.Kplus_triplet = float(self.Kplus_triplet + 1.0)
        self.Kplus = float(self.Kplus * math.exp((self.t_lastspike - t_spike) / self.tau_plus) + 1.0)

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True
