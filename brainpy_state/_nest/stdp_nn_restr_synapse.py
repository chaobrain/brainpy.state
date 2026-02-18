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

import brainunit as u
from brainstate.typing import ArrayLike

from .stdp_synapse import _STDP_EPS, stdp_synapse

__all__ = [
    'stdp_nn_restr_synapse',
]


class stdp_nn_restr_synapse(stdp_synapse):
    r"""NEST-compatible ``stdp_nn_restr_synapse`` connection model.

    Short description
    -----------------

    Synapse type for spike-timing dependent plasticity with restricted
    symmetric nearest-neighbour spike pairing.

    Description
    -----------

    ``stdp_nn_restr_synapse`` mirrors NEST
    ``models/stdp_nn_restr_synapse.h`` and implements the restricted
    nearest-neighbor pairing scheme from Morrison et al. (2008, fig. 7C):

    - On a presynaptic spike, depression uses the nearest preceding
      postsynaptic spike only if that postsynaptic spike occurred after the
      previous presynaptic spike,
    - On postsynaptic spikes, facilitation pairs only with the nearest
      preceding presynaptic spike that has not already been used for
      facilitation.

    A spike therefore participates in at most one depression pair and at most
    one facilitation pair.

    Compared with :class:`stdp_synapse`, this model changes two core STDP
    mechanisms:

    - No running presynaptic ``Kplus`` trace is used,
    - Depression is nearest-neighbor and restricted to intervals where at
      least one postsynaptic spike occurred since the last presynaptic spike.

    **1. Mathematical Formulation**

    With normalized weight :math:`\hat w = w/W_{max}`:

    .. math::
       \hat w \leftarrow \hat w
       + \lambda (1-\hat w)^{\mu_+} k_+

    .. math::
       \hat w \leftarrow \hat w
       - \alpha \lambda \hat w^{\mu_-} k_-

    with clipping to ``[0, Wmax]`` in normalized coordinates, as in NEST.

    The potentiation trace factor for the restricted rule is:

    .. math::
       k_+ = \exp\left(\frac{t_{\mathrm{last}} - (t_{post} + d)}{\tau_+}\right)

    where :math:`t_{\mathrm{last}}` is the previous presynaptic spike time,
    :math:`t_{post}` is the first postsynaptic spike time in the readout
    window, and :math:`d` is the dendritic delay.

    The depression trace factor is:

    .. math::
       k_- = \exp\left(\frac{t_{post}^{\mathrm{nn}} - (t_{pre} - d)}{\tau_-}\right)

    where :math:`t_{post}^{\mathrm{nn}}` is the nearest postsynaptic spike
    strictly before :math:`t_{pre} - d`.

    **2. Update Order (NEST Source Equivalent)**

    For a presynaptic spike at :math:`t_{pre}` with dendritic delay :math:`d`,
    NEST ``stdp_nn_restr_synapse::send`` performs:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\, t_{pre}-d]`.
    2. If history is non-empty, facilitate once using the first postsynaptic
       spike in that interval:
       :math:`\exp((t_{\mathrm{last}}-(t_{post}+d))/\tau_+)`.
    3. If history is non-empty, depress once using nearest-neighbor
       postsynaptic trace at :math:`t_{pre}-d`:
       :math:`\exp((t_{post}^{\mathrm{nn}}-(t_{pre}-d))/\tau_-)`.
    4. Send event with updated ``weight``.
    5. Set ``t_lastspike = t_pre``.

    This implementation preserves that exact ordering.

    **3. Coincidence Semantics**

    Pairs with exact coincidence are discarded by strict time comparisons
    (NEST ``stdp_eps`` behavior). If
    ``presynaptic_spike == postsynaptic_spike + dendritic_delay``,
    :math:`\Delta t = 0` is not used; the nearest strictly earlier valid
    post-spike is used instead.

    **4. Restricted Pairing Constraint**

    The "restricted" property ensures that a postsynaptic spike contributes to
    plasticity only if it occurred in the inter-spike interval between two
    consecutive presynaptic spikes. This prevents accumulation of plasticity
    from postsynaptic spikes that precede the synapse's activation history.

    Parameters
    ----------
    weight : ArrayLike, optional
        Initial synaptic weight (scalar, float). Must have same sign as
        ``Wmax``. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay (scalar, brainunit time). Dendritic delay for spike
        transmission. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id (non-negative integer). Identifies which
        input channel of the postsynaptic neuron this synapse targets.
        Default: ``0``.
    tau_plus : ArrayLike, optional
        Potentiation time constant (scalar, brainunit time, positive).
        Controls the temporal window for LTP. Default: ``20.0 * u.ms``.
    tau_minus : ArrayLike, optional
        Depression trace time constant (scalar, brainunit time, positive).
        Controls the temporal window for LTD. In NEST this belongs to the
        postsynaptic archiving neuron; here it is stored on the synapse for
        standalone compatibility. Default: ``20.0 * u.ms``.
    lambda_ : ArrayLike, optional
        Learning-rate parameter (scalar, float, positive). Global scaling
        factor for weight updates. Default: ``0.01``.
    alpha : ArrayLike, optional
        Depression scaling parameter (scalar, float, positive). Relative
        strength of LTD versus LTP. Default: ``1.0``.
    mu_plus : ArrayLike, optional
        Potentiation exponent (scalar, float, non-negative). Controls the
        weight-dependence of LTP. ``mu_plus=0`` gives additive LTP,
        ``mu_plus=1`` gives multiplicative LTP. Default: ``1.0``.
    mu_minus : ArrayLike, optional
        Depression exponent (scalar, float, non-negative). Controls the
        weight-dependence of LTD. ``mu_minus=0`` gives additive LTD,
        ``mu_minus=1`` gives multiplicative LTD. Default: ``1.0``.
    Wmax : ArrayLike, optional
        Maximum weight bound (scalar, float). Must have same sign as
        ``weight``. Weights are clipped to the range ``[0, Wmax]`` or
        ``[Wmax, 0]`` depending on sign. Default: ``100.0``.
    post : object, optional
        Default receiver object. Must implement postsynaptic input methods.
        If ``None``, must be specified in ``send()`` calls. Default: ``None``.
    name : str, optional
        Object name for identification. Default: ``None``.

    Parameter Mapping
    -----------------

    The following table maps NEST parameter names to their brainpy.state
    equivalents:

    ================== ==================== ===================================
    NEST Parameter     brainpy.state        Notes
    ================== ==================== ===================================
    ``weight``         ``weight``           Synaptic efficacy
    ``delay``          ``delay``            Dendritic delay (ms)
    ``receptor_type``  ``receptor_type``    Target receptor port
    ``tau_plus``       ``tau_plus``         LTP time constant (ms)
    ``tau_minus``      ``tau_minus``        LTD time constant (ms)
    ``lambda``         ``lambda_``          Learning rate
    ``alpha``          ``alpha``            LTD scaling factor
    ``mu_plus``        ``mu_plus``          LTP weight-dependence exponent
    ``mu_minus``       ``mu_minus``         LTD weight-dependence exponent
    ``Wmax``           ``Wmax``             Maximum weight bound
    ``t_lastspike``    ``t_lastspike``      Previous presynaptic spike time
    ================== ==================== ===================================

    Notes
    -----
    - In NEST, ``tau_minus`` is a postsynaptic-neuron parameter.
    - As in NEST, STDP updates are based on on-grid spike stamps and ignore
      sub-step precise offsets.
    - ``Kplus`` is not a parameter of this model (unlike ``stdp_synapse``).
    - The restriction mechanism ensures each spike participates in at most one
      pair of each type (facilitation and depression).

    Examples
    --------
    Create a restricted nearest-neighbor STDP synapse:

    .. code-block:: python

       >>> import brainpy.state as bst
       >>> import brainunit as u
       >>> syn = bst.nn.stdp_nn_restr_synapse(
       ...     weight=0.5,
       ...     delay=1.5 * u.ms,
       ...     tau_plus=20.0 * u.ms,
       ...     tau_minus=20.0 * u.ms,
       ...     lambda_=0.01,
       ...     alpha=1.0,
       ...     mu_plus=1.0,
       ...     mu_minus=1.0,
       ...     Wmax=100.0
       ... )

    Configure asymmetric learning rates:

    .. code-block:: python

       >>> syn = bst.nn.stdp_nn_restr_synapse(
       ...     weight=1.0,
       ...     lambda_=0.005,
       ...     alpha=1.05,  # Slightly stronger depression
       ...     mu_plus=0.0,  # Additive LTP
       ...     mu_minus=1.0  # Multiplicative LTD
       ... )

    References
    ----------
    .. [1] NEST source: ``models/stdp_nn_restr_synapse.h`` and
           ``models/stdp_nn_restr_synapse.cpp``.
    .. [2] Morrison A, Diesmann M, Gerstner W (2008).
           Phenomenological models of synaptic plasticity based on spike timing.
           Biological Cybernetics, 98:459-478.
           DOI: 10.1007/s00422-008-0233-1
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        lambda_: ArrayLike = 0.01,
        alpha: ArrayLike = 1.0,
        mu_plus: ArrayLike = 1.0,
        mu_minus: ArrayLike = 1.0,
        Wmax: ArrayLike = 100.0,
        post=None,
        name: str | None = None,
    ):
        super().__init__(
            weight=weight,
            delay=delay,
            receptor_type=receptor_type,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            lambda_=lambda_,
            alpha=alpha,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            Wmax=Wmax,
            Kplus=0.0,
            post=post,
            name=name,
        )

    def _get_nearest_neighbor_K_value(self, t_ms: float) -> float:
        r"""Compute nearest-neighbor depression trace value at given time.

        Matches NEST ``ArchivingNode::get_K_values`` nearest-neighbor behavior:
        searches backward through the postsynaptic spike history to find the
        latest postsynaptic spike strictly before ``t_ms``, then returns the
        exponentially decayed trace value.

        Parameters
        ----------
        t_ms : float
            Query time in milliseconds (on-grid spike stamp).

        Returns
        -------
        float
            Decayed depression trace value :math:`\exp((t_{post}^{\mathrm{nn}} - t_{ms})/\tau_-)`,
            where :math:`t_{post}^{\mathrm{nn}}` is the nearest postsynaptic spike
            strictly before ``t_ms``. Returns ``0.0`` if no valid postsynaptic
            spike is found.

        Notes
        -----
        - Uses strict inequality ``(t_ms - t_post) > _STDP_EPS`` to exclude
          exact coincidences (NEST ``stdp_eps`` semantics).
        - Iterates backward through ``_post_hist_t`` for efficiency (most
          recent spikes are typically nearest).
        """
        # Match ArchivingNode::get_K_values nearest-neighbor behavior:
        # use latest post spike strictly before t and decay a unit trace.
        for idx in range(len(self._post_hist_t) - 1, -1, -1):
            t_post = self._post_hist_t[idx]
            if (t_ms - t_post) > _STDP_EPS:
                return math.exp((t_post - t_ms) / self.tau_minus)
        return 0.0

    def get(self) -> dict:
        r"""Return current public parameters and mutable state.

        Returns the NEST-compatible parameter dictionary for this synapse,
        excluding ``Kplus`` (which is not used in this model).

        Returns
        -------
        dict
            Dictionary containing all public parameters and mutable state:
            ``weight``, ``delay``, ``receptor_type``, ``tau_plus``,
            ``tau_minus``, ``lambda``, ``alpha``, ``mu_plus``, ``mu_minus``,
            ``Wmax``, ``t_lastspike``, and ``synapse_model``.

        Notes
        -----
        - The returned dictionary has ``synapse_model='stdp_nn_restr_synapse'``
          for NEST compatibility.
        - ``Kplus`` is removed from the parent class's output since it is not
          part of the restricted nearest-neighbor pairing scheme.
        """
        params = super().get()
        params.pop('Kplus', None)
        params['synapse_model'] = 'stdp_nn_restr_synapse'
        return params

    def set(self, **kwargs):
        r"""Set NEST-style public parameters and mutable state.

        Accepts keyword arguments matching NEST parameter names for this
        synapse model. Raises an error if ``Kplus`` is provided (not used in
        this model).

        Parameters
        ----------
        **kwargs
            Keyword arguments for parameters to update. Valid keys: ``weight``,
            ``delay``, ``receptor_type``, ``tau_plus``, ``tau_minus``,
            ``lambda``, ``alpha``, ``mu_plus``, ``mu_minus``, ``Wmax``,
            ``t_lastspike``.

        Raises
        ------
        ValueError
            If ``Kplus`` is provided (not a valid parameter for this model).
        ValueError
            If any provided value fails validation (e.g., negative time
            constant, incompatible weight/Wmax signs).

        Notes
        -----
        - Setting ``weight`` or ``Wmax`` will re-validate the sign consistency
          constraint (both must have the same sign).
        """
        if 'Kplus' in kwargs:
            raise ValueError('Kplus is not a parameter of stdp_nn_restr_synapse.')
        super().set(**kwargs)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        r"""Schedule one outgoing event with NEST ``stdp_nn_restr_synapse`` dynamics.

        Implements the restricted nearest-neighbor STDP pairing rule. On each
        presynaptic spike:

        1. Reads postsynaptic spike history in the interval
           :math:`(t_{\mathrm{last}}-d,\, t_{pre}-d]`.
        2. If the history is non-empty:

           - Applies facilitation (LTP) using the first postsynaptic spike in
             the interval.
           - Applies depression (LTD) using the nearest-neighbor postsynaptic
             trace.
        3. Schedules the spike event with updated weight for delivery after
           the dendritic delay.
        4. Updates ``t_lastspike`` to the current spike time.

        Parameters
        ----------
        multiplicity : ArrayLike, optional
            Scalar event weight multiplier (e.g., spike count). If zero or
            very small, no event is sent. Default: ``1.0``.
        post : object, optional
            Target receiver object. If ``None``, uses the default receiver set
            during initialization. Must implement postsynaptic input methods.
        receptor_type : ArrayLike, optional
            Target receptor port (non-negative integer). If ``None``, uses
            ``self.receptor_type``. Default: ``None``.

        Returns
        -------
        bool
            ``True`` if an event was scheduled, ``False`` if ``multiplicity``
            was zero and no event was sent.

        Notes
        -----
        - **Restricted pairing**: Both LTP and LTD are applied only if at
          least one postsynaptic spike occurred between the previous and
          current presynaptic spikes.
        - **Facilitation**: Uses the first postsynaptic spike in the readout
          window with trace factor
          :math:`\exp((t_{\mathrm{last}} - (t_{post} + d))/\tau_+)`.
        - **Depression**: Uses the nearest-neighbor postsynaptic trace at
          :math:`t_{pre} - d`.
        - **Event timing**: Uses on-grid spike stamps (ignores sub-step
          precise offsets).
        - **Weight bounds**: Updated weight is clipped to ``[0, Wmax]`` or
          ``[Wmax, 0]`` depending on sign.

        Examples
        --------
        Send a presynaptic spike with default multiplicity:

        .. code-block:: python

           >>> sent = syn.send()

        Send with custom multiplicity and receptor type:

        .. code-block:: python

           >>> sent = syn.send(multiplicity=2.0, receptor_type=1)
        """
        if not self._is_nonzero(multiplicity):
            return False

        dt_ms = self._refresh_delay_if_needed()
        current_step = self._curr_step(dt_ms)

        # NEST uses on-grid event stamps in this model.
        t_spike = self._current_time_ms() + dt_ms
        dendritic_delay = float(self.delay)

        # Read postsynaptic history in (t_lastspike - d, t_spike - d].
        t1 = self.t_lastspike - dendritic_delay
        t2 = t_spike - dendritic_delay
        history = self._get_post_history_times(t1, t2)

        # Restricted nearest-neighbor rule: both facilitation and depression
        # are applied only if there was at least one post spike between the
        # previous and current pre spike.
        if history:
            minus_dt = self.t_lastspike - (history[0] + dendritic_delay)
            assert minus_dt < (-1.0 * _STDP_EPS)
            self.weight = float(self._facilitate(float(self.weight), math.exp(minus_dt / self.tau_plus)))

            kminus_value = self._get_nearest_neighbor_K_value(t_spike - dendritic_delay)
            self.weight = float(self._depress(float(self.weight), float(kminus_value)))

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True
