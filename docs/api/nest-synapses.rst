NEST-Compatible Synapse & Plasticity Models
============================================

Synapse and plasticity models compatible with the `NEST simulator <https://nest-simulator.readthedocs.io/>`_.

.. currentmodule:: brainpy.state


Static Synapses
---------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    static_synapse
    static_synapse_hom_w
    bernoulli_synapse
    cont_delay_synapse


Short-Term Plasticity (STP)
----------------------------

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


Voltage-Based / Specialized Synapses
--------------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    clopath_synapse
    jonke_synapse
    urbanczik_synapse
    vogels_sprekeler_synapse
    ht_synapse


Gap Junctions & Special Connections
------------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    gap_junction
    diffusion_connection
    rate_connection_instantaneous
    rate_connection_delayed
    sic_connection
