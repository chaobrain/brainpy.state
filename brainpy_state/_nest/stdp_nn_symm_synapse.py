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

from __future__ import annotations

import math

import brainunit as u
from brainstate.typing import ArrayLike

from .stdp_synapse import _STDP_EPS, stdp_synapse

__all__ = [
    'stdp_nn_symm_synapse',
]


class stdp_nn_symm_synapse(stdp_synapse):
    r"""NEST-compatible ``stdp_nn_symm_synapse`` connection model.

    Short description
    -----------------

    Synapse type for spike-timing dependent plasticity with symmetric
    nearest-neighbour spike pairing.

    Description
    -----------

    ``stdp_nn_symm_synapse`` mirrors NEST
    ``models/stdp_nn_symm_synapse.h`` and implements the symmetric nearest-
    neighbour pairing scheme from Morrison et al. (2007, 2008):

    - on a presynaptic spike, depression uses the nearest preceding
      postsynaptic spike,
    - postsynaptic spikes since the previous presynaptic spike contribute
      facilitation with nearest-neighbour trace factors.

    Compared with :class:`stdp_synapse`, this model removes the running
    presynaptic ``Kplus`` trace. Facilitation for each postsynaptic spike in
    the readout window uses
    :math:`\exp((t_{\mathrm{last}}-(t_{post}+d))/\tau_+)` directly.

    **1. Mathematical Model**

    The weight update follows the same functional forms as :class:`stdp_synapse`,
    but with symmetric nearest-neighbor pairing:

    .. math::
       \hat{w} \leftarrow \hat{w}
       + \lambda (1-\hat{w})^{\mu_+}
       \sum_{i} \exp((t_{\mathrm{last}}-(t_{\mathrm{post}}^{(i)}+d))/\tau_+)

    .. math::
       \hat{w} \leftarrow \hat{w}
       - \alpha \lambda \hat{w}^{\mu_-} k_-^{\mathrm{NN}}

    where :math:`\hat{w} = w / W_{\mathrm{max}}` is the normalized weight,
    :math:`t_{\mathrm{post}}^{(i)}` are **all** postsynaptic spikes in the
    interval :math:`(t_{\mathrm{last}}-d,\, t_{\mathrm{pre}}-d]`, and

    .. math::
       k_-^{\mathrm{NN}} = \begin{cases}
       \exp((t_{\mathrm{post}}^{\mathrm{last}} - (t_{\mathrm{pre}}-d))/\tau_-)
       & \text{if } \exists t_{\mathrm{post}}^{\mathrm{last}} < t_{\mathrm{pre}}-d \\
       0 & \text{otherwise}
       \end{cases}

    Here :math:`t_{\mathrm{post}}^{\mathrm{last}}` denotes the **nearest
    preceding** postsynaptic spike before :math:`t_{\mathrm{pre}}-d`.

    The symmetric scheme differs from both the all-to-all :class:`stdp_synapse`
    (which accumulates a running ``Kplus`` trace) and the pre-centered
    :class:`stdp_nn_pre_centered_synapse` (which uses only the first postsynaptic
    spike and resets ``Kplus``). Here, **all** postsynaptic spikes in the window
    contribute to potentiation, but each uses an exponential factor computed
    directly from :math:`t_{\mathrm{last}}` without a presynaptic trace variable.

    **2. Update Order (NEST Source Equivalent)**

    For a presynaptic spike at :math:`t_{\mathrm{pre}}` with dendritic delay
    :math:`d`, NEST ``stdp_nn_symm_synapse::send`` performs:

    1. Read postsynaptic history in
       :math:`(t_{\mathrm{last}}-d,\, t_{\mathrm{pre}}-d]`.
    2. For each postsynaptic spike in that interval, apply facilitation with
       :math:`\exp((t_{\mathrm{last}}-(t_{\mathrm{post}}+d))/\tau_+)`.
    3. Apply depression from nearest-neighbor postsynaptic trace at
       :math:`t_{\mathrm{pre}}-d`:
       :math:`\exp((t_{\mathrm{post}}^{\mathrm{nn}}-(t_{\mathrm{pre}}-d))/\tau_-)`.
    4. Send event with updated ``weight``.
    5. Set ``t_lastspike = t_pre``.

    This implementation preserves that exact ordering.

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
        Synaptic delay :math:`d` in ms. Default: ``1.0 * u.ms``.
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
    - ``Kplus`` is not a public parameter for this model because it is not used
      in the symmetric nearest-neighbor scheme. The constructor internally sets
      ``Kplus=0.0`` in the parent class, but it plays no role in weight updates.
    - The symmetric scheme produces different weight dynamics than all-to-all
      STDP: each postsynaptic spike contributes independently to facilitation,
      weighted by its temporal distance from the last presynaptic spike, rather
      than being accumulated into a running trace.

    Examples
    --------
    Symmetric nearest-neighbor STDP with custom parameters:

    .. code-block:: python

       >>> import brainpy.state as bp
       >>> import brainunit as u
       >>> syn = bp.stdp_nn_symm_synapse(
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
    .. [1] NEST source: ``models/stdp_nn_symm_synapse.h`` and
           ``models/stdp_nn_symm_synapse.cpp``.
    .. [2] Morrison A, Aertsen A, Diesmann M (2007).
           Spike-timing dependent plasticity in balanced random networks.
           Neural Computation, 19:1437-1467.
           DOI: 10.1162/089976607808742029
    .. [3] Morrison A, Diesmann M, Gerstner W (2008).
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
        - Unlike the presynaptic trace ``Kplus`` in other STDP models, this
          computes a unit-amplitude trace decayed from a single postsynaptic
          spike time, not an accumulated trace.
        """
        # Match ArchivingNode::get_K_values nearest-neighbor behavior:
        # use latest post spike strictly before t and decay a unit trace.
        for idx in range(len(self._post_hist_t) - 1, -1, -1):
            t_post = self._post_hist_t[idx]
            if (t_ms - t_post) > _STDP_EPS:
                return math.exp((t_post - t_ms) / self.tau_minus)
        return 0.0

    def get(self) -> dict:
        """Return current public parameters and mutable state.

        Returns a dictionary containing all synapse parameters and internal state
        variables, excluding the unused ``Kplus`` parameter (which is not part of
        the symmetric nearest-neighbor scheme).

        Returns
        -------
        dict
            Dictionary with keys ``'synapse_model'`` (str, set to
            ``'stdp_nn_symm_synapse'``), ``'weight'`` (float), ``'delay'``
            (float in ms), ``'receptor_type'`` (int), ``'tau_plus'`` (float in ms),
            ``'tau_minus'`` (float in ms), ``'lambda'`` (float), ``'alpha'`` (float),
            ``'mu_plus'`` (float), ``'mu_minus'`` (float), ``'Wmax'`` (float),
            ``'t_lastspike'`` (float in ms), and internal history state.
            The ``'Kplus'`` key is explicitly removed because it is not used.

        Notes
        -----
        - The returned dictionary is a snapshot and does not dynamically reflect
          subsequent state changes.
        - This method is used for serialization, debugging, and NEST-API
          compatibility (``GetStatus``).
        - Unlike :class:`stdp_synapse` and :class:`stdp_nn_pre_centered_synapse`,
          this model does not maintain a presynaptic trace ``Kplus``, so it is
          excluded from the returned state.
        """
        params = super().get()
        params.pop('Kplus', None)
        params['synapse_model'] = 'stdp_nn_symm_synapse'
        return params

    def set(self, **kwargs):
        """Set NEST-style public parameters and mutable state.

        Updates synapse parameters dynamically. Rejects attempts to set ``Kplus``
        because it is not part of the symmetric nearest-neighbor STDP model.

        Parameters
        ----------
        **kwargs : dict
            Parameter names and values to update. Valid keys include ``'weight'``,
            ``'delay'``, ``'receptor_type'``, ``'tau_plus'``, ``'tau_minus'``,
            ``'lambda'``, ``'alpha'``, ``'mu_plus'``, ``'mu_minus'``, ``'Wmax'``,
            and ``'t_lastspike'``.

        Raises
        ------
        ValueError
            If ``'Kplus'`` is present in ``kwargs``. The symmetric nearest-neighbor
            scheme does not use a presynaptic trace, so setting ``Kplus`` is invalid.

        Notes
        -----
        - This method provides NEST-API compatibility (``SetStatus``).
        - Parameter updates take effect immediately and apply to subsequent
          plasticity updates.
        - Unlike models with ``Kplus``, this model computes facilitation traces
          directly from postsynaptic spike times without maintaining a running
          presynaptic trace variable.

        Examples
        --------
        Update learning rate and potentiation time constant:

        .. code-block:: python

           >>> import brainpy.state as bp
           >>> import brainunit as u
           >>> syn = bp.stdp_nn_symm_synapse(weight=1.0)
           >>> syn.set(lambda_=0.02, tau_plus=15.0 * u.ms)
           >>> syn.get()['lambda']
           0.02
        """
        if 'Kplus' in kwargs:
            raise ValueError('Kplus is not a parameter of stdp_nn_symm_synapse.')
        super().set(**kwargs)

    def send(
        self,
        multiplicity: ArrayLike = 1.0,
        *,
        post=None,
        receptor_type: ArrayLike | None = None,
    ) -> bool:
        r"""Schedule one outgoing spike event with symmetric nearest-neighbor STDP.

        This method implements the complete NEST ``stdp_nn_symm_synapse::send``
        update sequence:

        1. Query postsynaptic spike history in the interval
           :math:`(t_{\mathrm{last}}-d,\, t_{\mathrm{spike}}-d]`.
        2. For **each** postsynaptic spike :math:`t_{\mathrm{post}}^{(i)}` in
           that interval, apply facilitation with:

           .. math::
              w \leftarrow w + \lambda (1-w/W_{\mathrm{max}})^{\mu_+}
              \exp((t_{\mathrm{last}} - (t_{\mathrm{post}}^{(i)} + d))/\tau_+)

           Unlike :class:`stdp_nn_pre_centered_synapse`, this uses **all**
           postsynaptic spikes in the window, not just the first.

        3. Apply depression from the **nearest preceding** postsynaptic spike:

           .. math::
              w \leftarrow w - \alpha \lambda (w/W_{\mathrm{max}})^{\mu_-}
              \exp((t_{\mathrm{post}}^{\mathrm{last}} - (t_{\mathrm{spike}}-d))/\tau_-)

        4. Enqueue a spike event with the updated weight for delivery at step
           :math:`\mathrm{current\_step} + \mathrm{delay\_steps}`.
        5. Update ``t_lastspike`` to the current spike time.

        No presynaptic trace ``Kplus`` is updated because this model does not
        use one.

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
        - **All** postsynaptic spikes in the facilitation window contribute
          independently to potentiation, weighted by their temporal distance from
          the last presynaptic spike. This is the "symmetric" aspect of the model.
        - Depression uses a strict nearest-neighbor rule: only the most recent
          postsynaptic spike before :math:`t_{\mathrm{spike}}-d` contributes.
        - Coincident spikes (within ``1e-6`` ms tolerance) are excluded from
          both facilitation and depression windows.
        - Unlike :class:`stdp_synapse`, no presynaptic trace is maintained; unlike
          :class:`stdp_nn_pre_centered_synapse`, the presynaptic trace is not reset
          after facilitation (because it does not exist).
        - This method is typically called by the presynaptic neuron's spike
          transmission logic; it can also be invoked manually for testing or
          standalone STDP simulation.

        Examples
        --------
        Manually trigger a presynaptic spike event:

        .. code-block:: python

           >>> import brainpy.state as bp
           >>> import brainunit as u
           >>> syn = bp.stdp_nn_symm_synapse(
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

        # Facilitation due to postsynaptic spikes in
        # (t_lastspike - dendritic_delay, t_spike - dendritic_delay].
        t1 = self.t_lastspike - dendritic_delay
        t2 = t_spike - dendritic_delay
        history = self._get_post_history_times(t1, t2)
        for t_post in history:
            minus_dt = self.t_lastspike - (t_post + dendritic_delay)
            assert minus_dt < (-1.0 * _STDP_EPS)
            self.weight = float(self._facilitate(float(self.weight), math.exp(minus_dt / self.tau_plus)))

        # Depression from nearest preceding postsynaptic spike.
        kminus_value = self._get_nearest_neighbor_K_value(t_spike - dendritic_delay)
        self.weight = float(self._depress(float(self.weight), float(kminus_value)))

        receiver = self._resolve_receiver(post)
        rport = self.receptor_type if receptor_type is None else self._to_receptor_type(receptor_type)
        weighted_payload = multiplicity * float(self.weight)

        delivery_step = int(current_step + int(self._delay_steps))
        self._queue[delivery_step].append((receiver, weighted_payload, int(rport), 'spike'))

        self.t_lastspike = float(t_spike)
        return True
