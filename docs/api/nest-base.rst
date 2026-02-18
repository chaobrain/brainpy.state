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
          │    └─ NESTNeuron     (_nest/_base.py)
          ├─ NESTSynapse         (_nest/_base.py)
          │    └─ NESTPlasticity (_nest/_base.py)
          └─ NESTDevice          (_nest/_base.py)
