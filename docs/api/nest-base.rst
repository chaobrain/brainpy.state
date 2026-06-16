.. warning::

   **Experimental — In Development.** The NEST-compatible model family is
   under active development. Parameter names, defaults, numerical behavior,
   and the set of available models may change without notice across 0.0.x
   releases. See the :doc:`NEST-style status page </nest-status/index>` for
   current scope and limitations.


NEST Base Classes
=================

Abstract marker base classes for all NEST-compatible models in
``brainpy.state``.  Every NEST-compatible class inherits from exactly one
of these four bases, which makes it easy to introspect model type at
runtime (e.g. ``isinstance(model, NESTPlasticity)``).

.. currentmodule:: brainpy.state

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   NESTNeuron
   NESTSynapse
   NESTPlasticity
   NESTDevice


Class Hierarchy
---------------

.. code-block:: text

   brainstate.nn.Dynamics
     └─ Dynamics                 (_base.py)
          ├─ Neuron              (_base.py)
          │    └─ NESTNeuron     (_nest_base/_base.py)
          ├─ NESTSynapse         (_nest_base/_base.py)
          │    └─ NESTPlasticity (_nest_base/_base.py)
          └─ NESTDevice          (_nest_base/_base.py)
