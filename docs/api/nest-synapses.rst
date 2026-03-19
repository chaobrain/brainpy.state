NEST-Compatible Synapse Models
==============================

Static and structural synapse models compatible with the
`NEST simulator <https://nest-simulator.readthedocs.io/>`_.
For activity-dependent plasticity models see :doc:`nest-plasticity`.

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
