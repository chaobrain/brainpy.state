.. _nest-spatial:

=============================
Spatially-structured networks
=============================

.. currentmodule:: brainpy.state

When neurons sit at positions in space and the probability (or weight) of a
connection depends on the source→target distance, build the layout and the
distance-dependent rule with the :mod:`brainpy.state.spatial` toolkit. It mirrors
NEST's spatial / topology vocabulary and rides the ordinary
:meth:`Simulator.connect` call — the same call any non-spatial rule uses.


Placing neurons in space
========================

Create a population on a layout by passing ``positions=`` to
:meth:`Simulator.create`:

.. code-block:: python

   import brainunit as u
   from brainpy import state as bp

   sim = bp.Simulator(dt=0.1 * u.ms)
   pos = bp.spatial.grid([4, 3], extent=[2.0, 1.5])     # 4 columns × 3 rows
   pop = sim.create(bp.iaf_psc_alpha, positions=pos)

   coords = sim.get_position(pop)                        # (12, 2) node coordinates

``bp.spatial.grid([nx, ny], extent=[Lx, Ly])`` lays neurons on a regular grid
exactly as NEST does (column is the slow axis, row the fast axis; ``x`` increases
left→right and ``y`` decreases top→bottom). :meth:`Simulator.get_position`
(NEST's ``GetPosition``) reads the node coordinates back.


Distance-dependent connectivity
===============================

A spatial connection rule combines a **probability law** (a function of distance)
with an optional **mask** (a spatial cutoff), and is passed as ``rule=`` to
``connect``:

.. code-block:: python

   sim.connect(
       a, b,
       rule=bp.spatial.spatial_pairwise_bernoulli(
           p=bp.spatial.gaussian(bp.spatial.distance, std=0.5),
           mask=bp.spatial.circular(3.0)),
       weight=1.0 * u.pA, delay=1.0 * u.ms, seed=0)

The building blocks:

- ``bp.spatial.distance`` — the source→target distance, the argument to a kernel.
- ``bp.spatial.gaussian(distance, std=...)`` — a Gaussian connection-probability
  kernel :math:`p(d) = \exp(-d^2 / (2\,\mathrm{std}^2))` (other kernels follow
  the same shape).
- ``bp.spatial.circular(radius)`` — a circular mask clipping connections beyond
  ``radius``.
- ``bp.spatial.spatial_pairwise_bernoulli(p=, mask=)`` — draw each
  source→target edge with probability ``p`` inside ``mask``.

Because the per-distance probability is a fixed law (only the PRNG draw diverges
between NEST and JAX), spatial connectivity is validated **distributionally**: the
empirical :math:`p(d)` curve matches NEST within a band even though individual
draws differ. See :doc:`validation-status`.


Querying the realized footprint
===============================

After wiring, read back which targets a source actually connected to:

.. code-block:: python

   ctr = bp.spatial.center_element(pos)                  # the central node
   targets = bp.spatial.target_positions(sim, a[ctr], b)[0]

``bp.spatial.target_positions`` (NEST's ``GetTargetPositions``) returns the
coordinates of a source's realized targets; ``bp.spatial.center_element`` picks the
node nearest the layout centre — handy for plotting a single neuron's connection
footprint.


See also
========

- :doc:`connectivity` — the non-spatial connection rules and the synapse spec.
- :doc:`/apis/nest-spatial` — the full spatial API.
- :doc:`/examples/nest-gallery` — the ``spatial_*`` example scripts (grid,
  Gaussian, Gabor, 3-D, CSA).
- `NEST spatial documentation <https://nest-simulator.readthedocs.io/>`_ — the
  authoritative reference for upstream spatial / topology semantics.
