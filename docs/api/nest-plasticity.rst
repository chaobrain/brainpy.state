.. warning::

   **Experimental — In Development.** The NEST-compatible model family is
   under active development. Parameter names, defaults, numerical behavior,
   and the set of available models may change without notice across 0.0.x
   releases. See the :doc:`NEST-style status page </nest-status/index>` for
   current scope and limitations.


NEST-Compatible Plasticity Models
==================================

Activity-dependent plasticity synapse models compatible with the
`NEST simulator <https://nest-simulator.readthedocs.io/>`_.
All classes inherit from :class:`NESTPlasticity`.
For static and structural synapse models see :doc:`nest-synapses`.

.. currentmodule:: brainpy.state


Short-Term Plasticity (STP)
----------------------------

Transient, reversible weight changes on the timescale of individual spikes,
following the Tsodyks-Markram formalism.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    tsodyks_synapse
    tsodyks_synapse_hom
    tsodyks2_synapse
    quantal_stp_synapse


Spike-Timing Dependent Plasticity (STDP)
-----------------------------------------

Long-term weight changes driven by the relative timing of pre- and
post-synaptic spikes.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    stdp_synapse
    stdp_synapse_hom
    stdp_pl_synapse_hom
    stdp_facetshw_synapse_hom
    stdp_nn_pre_centered_synapse
    stdp_nn_restr_synapse
    stdp_nn_symm_synapse
    stdp_triplet_synapse
    stdp_dopamine_synapse


Voltage-Based / Specialised Learning Rules
-------------------------------------------

Weight updates that depend on membrane voltage trajectories or
supervised teaching signals.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    clopath_synapse
    jonke_synapse
    urbanczik_synapse
    vogels_sprekeler_synapse
    ht_synapse
