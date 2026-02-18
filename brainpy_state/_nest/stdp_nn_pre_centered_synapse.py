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

from brainstate.typing import ArrayLike

from .stdp_synapse import stdp_synapse

__all__ = [
    'stdp_nn_pre_centered_synapse',
]


_STDP_EPS = 1.0e-6


class stdp_nn_pre_centered_synapse(stdp_synapse):
    r"""NEST-compatible ``stdp_nn_pre_centered_synapse`` connection model.

    Short description
    -----------------

    Synapse type for spike-timing dependent plasticity with presynaptic-
    centered nearest-neighbour spike pairing.

    Description
    -----------

    ``stdp_nn_pre_centered_synapse`` mirrors NEST
    ``models/stdp_nn_pre_centered_synapse.h`` and implements the pairing
    scheme described by Izhikevich and Desai (2003) and Morrison et al. (2008):

    - Each presynaptic spike is depressed by the nearest preceding
      postsynaptic spike,
    - Each postsynaptic spike facilitates all presynaptic spikes that occurred
      after the previous postsynaptic spike.

    Compared with :class:`stdp_synapse`, this model introduces nearest-neighbor
    postsynaptic depression and a presynaptic trace reset behavior:

    - ``Kplus`` decays with ``tau_plus``, increments by ``1`` per pre-spike,
      and is reset to ``0`` when any postsynaptic spike occurred in
      :math:`(t_{\mathrm{last}}-d,\, t_{pre}-d]`.
    - The depression trace term is nearest-neighbor only:
      :math:`\exp((t_{post}^{\mathrm{last}}-t)/\tau_{-})`, where
      :math:`t_{post}^{\mathrm{last}} < t`.

    **1. Mathematical Model**

    The weight update follows the same functional forms as :class:`stdp_synapse`,
    but with nearest-neighbor pairing constraints:

    .. math::
       \hat{w} \leftarrow \hat{w}
       + \lambda (1-\hat{w})^{\mu_+} k_+^{\mathrm{NN}}

    .. math::
       \hat{w} \leftarrow \hat{w}
       - \alpha \lambda \hat{w}^{\mu_-} k_-^{\mathrm{NN}}

    where :math:`\hat{w} = w / W_{\mathrm{max}}` is the normalized weight and

    .. math::
       k_+^{\mathrm{NN}} = \begin{cases}
       K_+ \exp((t_{\mathrm{last}} - (t_{\mathrm{post}}^{(1)} + d))/\tau_+)
       & \text{if } \exists t_{\mathrm{post}}^{(1)} \in
         (t_{\mathrm{last}}-d,\, t_{\mathrm{pre}}-d] \\
       0 & \text{otherwise}
       \end{cases}

    .. math::
       k_-^{\mathrm{NN}} = \begin{cases}
       \exp((t_{\mathrm{post}}^{\mathrm{last}} - (t_{\mathrm{pre}}-d))/\tau_-)
       & \text{if } \exists t_{\mathrm{post}}^{\mathrm{last}} < t_{\mathrm{pre}}-d \\
       0 & \text{otherwise}
       \end{cases}

    Here :math:`t_{\mathrm{post}}^{(1)}` denotes the **first** postsynaptic
    spike in the interval :math:`(t_{\mathrm{last}}-d,\, t_{\mathrm{pre}}-d]`,
    and :math:`t_{\mathrm{post}}^{\mathrm{last}}` denotes the **nearest preceding**
    postsynaptic spike before :math:`t_{\mathrm{pre}}-d`.

    After processing a presynaptic spike that finds a postsynaptic spike in the
    facilitation window, the presynaptic trace is reset:

    .. math::
       K_+ \leftarrow 0

    **2. Update Order (NEST Source Equivalent)**

    For a presynaptic spike at :math:`t_{\mathrm{pre}}` with dendritic delay
    :math:`d`, NEST ``stdp_nn_pre_centered_synapse::send`` performs:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\, t_{\mathrm{pre}}-d]`.
    2. If non-empty, use only the first postsynaptic spike in this interval for
       facilitation with
       :math:`K_+ \exp((t_{\mathrm{last}}-(t_{\mathrm{post}}+d))/\tau_+)`.
    3. If step 2 happened, reset ``Kplus = 0``.
    4. Apply depression from nearest-neighbor postsynaptic trace at
       :math:`t_{\mathrm{pre}}-d`.
    5. Update ``Kplus`` as
       :math:`K_+ \leftarrow K_+ \exp((t_{\mathrm{last}}-t_{\mathrm{pre}})/\tau_+) + 1`.
    6. Send event with updated ``weight``.
    7. Set ``t_lastspike = t_pre``.

    This implementation preserves the same ordering.

    **3. Coincidence Semantics**

    Pairs with exact coincidence are discarded by strict time comparisons
    (NEST ``stdp_eps`` behavior). If
    :math:`t_{\mathrm{pre}} = t_{\mathrm{post}} + d` (within ``1e-6`` ms),
    the coincident postsynaptic spike is not used for depression/facilitation;
    earlier valid nearest neighbors are used instead.

    Parameters
    ----------
    weight : ArrayLike, optional
        Initial synaptic weight. Default: ``1.0``.
    delay : ArrayLike, optional
        Synaptic delay in ms. Default: ``1.0 * u.ms``.
    receptor_type : int, optional
        Receiver port/receptor id. Default: ``0``.
    tau_plus : ArrayLike, optional
        Potentiation time constant :math:`\tau_+` in ms. Default: ``20.0 * u.ms``.
    tau_minus : ArrayLike, optional
        Depression trace time constant :math:`\tau_-` in ms.
        In NEST this is a postsynaptic-neuron parameter; here it is stored on
        the synapse for standalone compatibility. Default: ``20.0 * u.ms``.
    lambda_ : ArrayLike, optional
        Learning-rate parameter :math:`\lambda`. Default: ``0.01``.
    alpha : ArrayLike, optional
        Depression scaling parameter :math:`\alpha`. Default: ``1.0``.
    mu_plus : ArrayLike, optional
        Potentiation exponent :math:`\mu_+`. Default: ``1.0``.
    mu_minus : ArrayLike, optional
        Depression exponent :math:`\mu_-`. Default: ``1.0``.
    Wmax : ArrayLike, optional
        Maximum weight bound :math:`W_{\mathrm{max}}`. Must have same sign as
        ``weight``. Default: ``100.0``.
    Kplus : ArrayLike, optional
        Initial presynaptic trace value :math:`K_+`. Must be non-negative.
        Default: ``0.0``.
    post : object, optional
        Default receiver object for spike transmission.
    name : str, optional
        Object name for debugging and serialization.

    Notes
    -----
    - In NEST, ``tau_minus`` belongs to the postsynaptic archiving neuron.
      This backend stores equivalent state locally for standalone
      compatibility, while preserving update semantics.
    - As in NEST, the model uses on-grid spike stamps and ignores sub-step
      precise spike offsets for STDP updates.
    - The presynaptic trace reset when a postsynaptic spike is found
      distinguishes this model from :class:`stdp_synapse`, which accumulates
      potentiation from all postsynaptic spikes without forgetting prior
      presynaptic activity.

    Examples
    --------
    Nearest-neighbor pre-centered STDP with custom parameters:

    .. code-block:: python

       >>> import brainpy.state as bp
       >>> import brainunit as u
       >>> syn = bp.stdp_nn_pre_centered_synapse(
       ...     weight=0.5,
       ...     delay=1.5 * u.ms,
       ...     tau_plus=16.8 * u.ms,
       ...     tau_minus=33.7 * u.ms,
       ...     lambda_=0.005,
       ...     alpha=0.85,
       ...     Wmax=5.0,
       ... )
       >>> syn.weight
       0.5

    References
    ----------
    .. [1] NEST source: ``models/stdp_nn_pre_centered_synapse.h`` and
           ``models/stdp_nn_pre_centered_synapse.cpp``.
    .. [2] Izhikevich EM, Desai NS (2003). Relating STDP to BCM.
           Neural Computation, 15:1511-1523.
           DOI: 10.1162/089976603321891783
    .. [3] Morrison A, Diesmann M, Gerstner W (2008).
           Phenomenological models of synaptic plasticity based on spike timing.
           Biological Cybernetics, 98:459-478.
           DOI: 10.1007/s00422-008-0233-1
    """

    __module__ = 'brainpy.state'

    def _get_nearest_neighbor_K_value(self, t_ms: float) -> float:
        r"""Compute nearest-neighbor depression trace value at time ``t_ms``.

        Matches NEST ``ArchivingNode::get_K_values`` nearest-neighbor behavior:
        find the latest postsynaptic spike strictly before ``t_ms`` and return
        :math:`\exp((t_{\mathrm{post}}^{\mathrm{last}} - t_{\mathrm{ms}})/\tau_-)`.

        Parameters
        ----------
        t_ms : float
            Query time in milliseconds. Must be positive.

        Returns
        -------
        float
            Depression trace value :math:`k_-^{\mathrm{NN}}` computed from
            the nearest preceding postsynaptic spike. Returns ``0.0`` if no
            valid postsynaptic spike exists in history or if the nearest spike
            is not strictly before ``t_ms`` (within ``1e-6`` ms tolerance).

        Notes
        -----
        - This method iterates backward through ``self._post_hist_t`` to find
          the most recent postsynaptic spike :math:`t_{\mathrm{post}}` such
          that :math:`t_{\mathrm{ms}} - t_{\mathrm{post}} > 10^{-6}` ms.
        - If no such spike exists, depression is zero (no LTD applied).
        - The exponential decay uses ``self.tau_minus``, which in NEST belongs
          to the postsynaptic neuron but is stored locally here.
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

        Returns a dictionary containing all synapse parameters and internal state
        variables, including ``weight``, ``delay``, ``receptor_type``, plasticity
        parameters (``tau_plus``, ``tau_minus``, ``lambda``, ``alpha``, ``mu_plus``,
        ``mu_minus``, ``Wmax``), and the presynaptic trace ``Kplus``.

        Returns
        -------
        dict
            Dictionary with keys ``'synapse_model'`` (str, set to
            ``'stdp_nn_pre_centered_synapse'``), ``'weight'`` (float),
            ``'delay'`` (float in ms), ``'receptor_type'`` (int),
            ``'tau_plus'`` (float in ms), ``'tau_minus'`` (float in ms),
            ``'lambda'`` (float), ``'alpha'`` (float), ``'mu_plus'`` (float),
            ``'mu_minus'`` (float), ``'Wmax'`` (float), ``'Kplus'`` (float),
            ``'t_lastspike'`` (float in ms), and internal history state.

        Notes
        -----
        - The returned dictionary is a snapshot and does not dynamically reflect
          subsequent state changes.
        - This method is used for serialization, debugging, and NEST-API
          compatibility (``GetStatus``).
        """
        params = super().get()
        params['synapse_model'] = 'stdp_nn_pre_centered_synapse'
        return params

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        r"""Schedule one outgoing spike event with nearest-neighbor pre-centered STDP.

        This method implements the complete NEST
        ``stdp_nn_pre_centered_synapse::send`` update sequence:

        1. Query postsynaptic spike history in the interval
           :math:`(t_{\mathrm{last}}-d,\, t_{\mathrm{spike}}-d]`.
        2. If at least one postsynaptic spike exists in that interval, apply
           facilitation using the **first** such spike:

           .. math::
              w \leftarrow w + \lambda (1-w/W_{\mathrm{max}})^{\mu_+}
              K_+ \exp((t_{\mathrm{last}} - (t_{\mathrm{post}}^{(1)} + d))/\tau_+)

           and reset the presynaptic trace :math:`K_+ \leftarrow 0`.

        3. Apply depression from the **nearest preceding** postsynaptic spike:

           .. math::
              w \leftarrow w - \alpha \lambda (w/W_{\mathrm{max}})^{\mu_-}
              \exp((t_{\mathrm{post}}^{\mathrm{last}} - (t_{\mathrm{spike}}-d))/\tau_-)

        4. Update the presynaptic trace:

           .. math::
              K_+ \leftarrow K_+ \exp((t_{\mathrm{last}} - t_{\mathrm{spike}})/\tau_+) + 1

        5. Enqueue a spike event with the updated weight for delivery at step
           :math:`\mathrm{current\_step} + \mathrm{delay\_steps}`.
        6. Update ``t_lastspike`` to the current spike time.

        Parameters
        ----------
        multiplicity : ArrayLike, optional
            Spike multiplicity (weight scaling factor). If zero, no event is sent.
            Default: ``1.0``.
        post : object, optional
            Target receiver object. If ``None``, uses the default receiver set
            at construction.
        receptor_type : ArrayLike or None, optional
            Receptor port id for the event. If ``None``, uses
            ``self.receptor_type``. Must be a non-negative integer.

        Returns
        -------
        bool
            ``True`` if the event was scheduled, ``False`` if ``multiplicity``
            was zero and no event was sent.

        Notes
        -----
        - The weight update occurs **before** the event is enqueued, so the
          transmitted spike carries the plasticity-modified weight.
        - If no postsynaptic spike exists in the facilitation window, facilitation
          is skipped and ``Kplus`` is not reset.
        - Depression uses a strict nearest-neighbor rule: only the most recent
          postsynaptic spike before :math:`t_{\mathrm{spike}}-d` contributes.
        - Coincident spikes (within ``1e-6`` ms tolerance) are excluded from
          both facilitation and depression windows.
        - This method is typically called by the presynaptic neuron's spike
          transmission logic; it can also be invoked manually for testing or
          standalone STDP simulation.

        Examples
        --------
        Manually trigger a presynaptic spike event:

        .. code-block:: python

           >>> import brainpy.state as bp
           >>> import brainunit as u
           >>> syn = bp.stdp_nn_pre_centered_synapse(
           ...     weight=1.0, delay=1.0 * u.ms, tau_plus=20.0 * u.ms
           ... )
           >>> # Assume postsynaptic spikes have been recorded...
           >>> success = syn.send(multiplicity=1.0)
           >>> print(success)
           True
           >>> print(syn.weight)  # Weight has been updated by STDP
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

        # Facilitation from the first postsynaptic spike in the interval.
        if history:
            minus_dt = self.t_lastspike - (history[0] + dendritic_delay)
            assert minus_dt < (-1.0 * _STDP_EPS)
            kplus_term = self.Kplus * math.exp(minus_dt / self.tau_plus)
            self.weight = float(self._facilitate(float(self.weight), float(kplus_term)))

            # Pre-centered nearest-neighbor scheme forgets previous pre spikes
            # once a post spike happened between current and previous pre spike.
            self.Kplus = 0.0

        # Depression from nearest preceding postsynaptic spike.
        kminus_value = self._get_nearest_neighbor_K_value(t_spike - dendritic_delay)
        self.weight = float(self._depress(float(self.weight), float(kminus_value)))

        self.Kplus = float(self.Kplus * math.exp((self.t_lastspike - t_spike) / self.tau_plus) + 1.0)

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True
