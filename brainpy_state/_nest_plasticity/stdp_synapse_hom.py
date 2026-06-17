# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-faithful ``stdp_synapse_hom`` — homogeneous-parameter pair STDP spec.

Rebuilt as a thin reuse of :class:`~brainpy_state._nest_plasticity.stdp_synapse.stdp_synapse`
on the frozen :class:`~brainpy_state._nest_network.event_plastic.EventPlasticProj`
substrate. In NEST ``stdp_synapse_hom`` stores the plasticity parameters
(``lambda``, ``alpha``, ``mu_plus``, ``mu_minus``, ``tau_plus``, ``Wmax``) as
*common* properties shared by every synapse of the model (a memory optimisation,
set via ``SetDefaults`` / ``CopyModel``) while the weight stays per-connection.
In this spec parameters are already rule-level — shared by every edge of the
projection by construction — so the rule kernel, defaults, traces, simultaneous
exclusion and ``[0, Wmax]`` clamp are *exactly* those of ``stdp_synapse``. This
class exists to mirror NEST's model name and homogeneous-parameter semantics.
"""
from __future__ import annotations

from .stdp_synapse import stdp_synapse

__all__ = ['stdp_synapse_hom']


class stdp_synapse_hom(stdp_synapse):
    r"""Homogeneous (shared-parameter) pair-based STDP synapse spec (NEST ``stdp_synapse_hom``).

    Functionally identical to :class:`~brainpy_state._nest_plasticity.stdp_synapse.stdp_synapse`:
    potentiation on the post spike (using ``K+``), depression on the pre spike
    (using ``K-``), the Guetig (2003) soft-bounded ``facilitate_``/``depress_``
    forms with the weight clamped to :math:`[0, W_{\max}]` inside each update, and
    the same NEST defaults (``lambda=0.01``, ``alpha=1.0``, ``mu_plus=mu_minus=1``,
    ``tau_plus=tau_minus=20`` ms, ``Wmax=100``). In NEST those parameters are
    *common* (homogeneous across the model); here every projection edge already
    shares one rule instance, so the distinction collapses and the kernel is
    inherited verbatim.

    Parameters
    ----------
    weight, delay, receptor_type, tau_plus, tau_minus, lambda_, alpha, mu_plus, mu_minus, Wmax, Kplus
        See :class:`~brainpy_state._nest_plasticity.stdp_synapse.stdp_synapse`.

    Notes
    -----
    **NEST divergence — ``tau_minus`` location.** As for ``stdp_synapse``,
    ``tau_minus`` is a parameter of the postsynaptic neuron (``ArchivingNode``) in
    NEST, not the synapse; here it is a synapse-spec attribute driving the
    substrate's per-post ``K-`` trace so STDP runs standalone.

    **Parity note.** The consolidated NEST vs. brainpy.state divergence reference
    — trace-storage move, the family parameter-location map, and the parity-test
    links — is in :doc:`/nest-style/divergences/stdp` (:ref:`stdp-tau-minus`).

    References
    ----------
    .. [1] NEST ``models/stdp_synapse_hom.h``; Guetig et al. (2003); Morrison et al. (2008).

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy.state import stdp_synapse, stdp_synapse_hom
       >>> s = stdp_synapse_hom(weight=5.0, lambda_=0.01)
       >>> isinstance(s, stdp_synapse)          # thin reuse of the pair kernel
       True
       >>> s.is_homogeneous_weight, s.edge_state_init()
       (False, {})
       >>> float(u.Quantity(s.post_trace_tau).to_decimal(u.ms))
       20.0
    """
    __module__ = 'brainpy.state'
