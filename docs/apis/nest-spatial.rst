NEST-Compatible Spatial Networks
================================

Spatially-structured network primitives compatible with the
`NEST simulator <https://nest-simulator.readthedocs.io/>`_ ``nest.spatial`` API.
These build layers of nodes embedded in 2-D or 3-D space, define
distance-dependent connection kernels and masks, and provide tools to inspect
and visualize the resulting connectivity.

.. currentmodule:: brainpy.state


Layers & Positions
------------------

Construct populations whose nodes carry spatial coordinates.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    spatial.Layer
    spatial.grid
    spatial.free
    spatial.pos
    spatial.source_pos
    spatial.target_pos


Distance & Displacement
-----------------------

Compute spatial relationships between source and target nodes.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    spatial.displacement
    spatial.pairwise_distance
    spatial.Distance


Connection Kernels
------------------

Distance-dependent connection-probability profiles.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    spatial.gaussian
    spatial.exponential
    spatial.gamma
    spatial.gabor
    spatial.gaussian2D


Masks
-----

Spatial regions that restrict the set of candidate targets.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    spatial.circular
    spatial.spherical
    spatial.box
    spatial.rectangular
    spatial.doughnut
    spatial.elliptical
    spatial.ellipsoidal


Spatial Connection Rules
------------------------

Distance-dependent pairwise-Bernoulli connectivity.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    spatial.SpatialConnRule
    spatial.spatial_pairwise_bernoulli


Node Selection & Introspection
------------------------------

Query nodes by position or mask and dump the resulting layer connectivity.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    spatial.center_element
    spatial.nearest_element
    spatial.select_nodes_by_mask
    spatial.target_nodes
    spatial.target_positions
    spatial.dump_layer_nodes
    spatial.dump_layer_connections


Visualization
-------------

Plot layers, connection targets/sources, and probability kernels.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    spatial.plot_layer
    spatial.plot_targets
    spatial.plot_sources
    spatial.plot_probability_parameter
