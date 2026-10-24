NEST Base Classes
=================

Abstract marker base classes for all NEST-compatible models in
``brainspike``.  Every NEST-compatible class inherits from exactly one
of these four bases, which makes it easy to introspect model type at
runtime (e.g. ``isinstance(model, NESTPlasticity)``).

.. currentmodule:: brainspike

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   NESTNeuron
   NESTSynapse
   NESTPlasticity
   NESTDevice

