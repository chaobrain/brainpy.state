NEST-Compatible Network Builder
===============================

High-level builders, projections, connection rules, and simulation utilities
for assembling and running NEST-compatible networks in ``brainpy.state``.
This is the declarative, NEST-like network layer: describe populations and
connections with a :class:`~brainpy.state.network.Builder`, materialize a
:class:`~brainpy.state.network.Network`, and drive it with a
:class:`~brainpy.state.network.Simulator`.

.. currentmodule:: brainpy.state


Network Construction & Execution
--------------------------------

Assemble a network from a declarative description and run it.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    network.Network
    network.Builder
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

Collect spikes, state variables, and synaptic weights during simulation.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    network.Recorder
    network.weight_recorder_events


Projection Classes
------------------

Connect populations under the various NEST-like connection rules.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    network.OneToOneProj
    network.AllToAllProj
    network.PairwiseBernoulliProj
    network.SymmetricPairwiseBernoulliProj
    network.FixedIndegreeProj
    network.FixedOutdegreeProj
    network.FixedTotalNumberProj
    network.PairwisePoissonProj
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
