NEST-Compatible Simulator
=========================

The NEST-style ``Simulator`` and its helpers: create populations and devices,
connect them under NEST-like connection rules, and simulate
(``create`` / ``connect`` / ``simulate``).

.. currentmodule:: brainspike


Simulation
----------

Create, connect, and run a network.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    network.Simulator
    network.SimulationResult


Population Views & Synapse Collections
--------------------------------------

Handles returned when slicing populations or referring to groups of synapses.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    network.NodeView
    network.SynapseCollection


Recording
---------

Read recorded synaptic weights after simulation.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    network.weight_recorder_events


Event Projections
-----------------

The projections ``Simulator.connect`` builds between populations.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    network.EventProjection
    network.EventPlasticProj
    network.VoltageCoupledPlasticProj


Connection Rules
----------------

Rule objects and helpers used to specify how populations are wired together.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    network.ConnRule
    network.all_to_all
    network.one_to_one
    network.fixed_indegree
    network.fixed_total_number
    network.pairwise_bernoulli
    network.third_factor_bernoulli_with_pool
    network.explicit_edges
    network.send_steps_from_pre
