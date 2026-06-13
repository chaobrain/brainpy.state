.. _nest-guide:

NEST Porting Guide
==================

.. warning::

   **Experimental — In Development.** The NEST-compatible model family is under
   active development. Parameter names, defaults, and numerical behavior may
   change without notice across 0.0.x releases. See the
   :doc:`NEST-style status page </nest-status/index>` for current scope.

Practical guidance for researchers porting `NEST simulator
<https://nest-simulator.readthedocs.io/>`_ models to ``brainpy.state``. Where the
:doc:`API reference </api/index>` lists *what* exists and the
:doc:`status page </nest-status/index>` tracks *how mature* it is, these pages
explain the **semantic differences** a port must account for — divergences that
are already discovered, frozen, and backed by live-NEST parity tests, but which
the source code alone does not surface at a glance.

.. toctree::
   :maxdepth: 2

   stdp-divergences

Currently this guide covers the discrete spike-timing-dependent plasticity (STDP)
family. The :doc:`stdp-divergences` page documents where STDP learning state lives
in NEST versus ``brainpy.state`` (the ``tau_minus`` trace-storage move and the
wider parameter-location map), the small documented numerical bands, and the exact
nearest-neighbour pairing convention of each ``stdp_nn_*`` variant.
