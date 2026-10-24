Relationship between ``brainpy`` and ``brainspike``
======================================================

``brainspike`` is the state-based modeling layer of `BrainPy
<https://brainpy.readthedocs.io/>`_. It is developed and released as the standalone
``brainspike`` package and surfaced through the ``brainspike`` namespace, so the
same code is reachable as ``brainspike`` whether you install classic ``brainpy`` or
the layer on its own.

Why a separate package
----------------------

- **Decoupled release cycle.** ``brainspike`` is versioned and released independently
  of classic ``brainpy``, so its models, fixes, and features ship on their own cadence.
- **Surfaced through the ``brainspike`` namespace.** The standalone ``brainspike``
  distribution is re-exported by classic ``brainpy`` as ``brainspike`` — there is no
  separate import path to learn.
- **Built on the BrainX substrate.** It is built on ``brainstate`` (State management),
  ``brainunit`` (physical units), and the rest of the `BrainX ecosystem
  <https://brainx.chaobrain.com/>`_, and compiles through JAX to CPU, GPU, and TPU.

Which paradigm to use
---------------------

- Reach for **``brainspike``** for new work: ``State``-based models,
  surrogate-gradient / differentiable training, online learning, and JAX-native
  pipelines. This is the recommended starting point.
- Keep using **classic ``brainpy``** for existing ``DynamicalSystem`` models and for the
  array/operator and integrator API (``brainpy.math``, ``brainpy.integrators``,
  ``brainpy.dyn``). It is unchanged and fully supported.

The two paradigms coexist; adopting ``brainspike`` does not deprecate or remove any
part of classic ``brainpy``.

Installation
------------

``brainspike`` is bundled with classic ``brainpy`` (``brainpy >= 2.7.6``), so if you
installed ``brainpy`` you already have it — no code changes are required:

.. code-block:: python

   import brainpy

   import brainspike
   neuron = brainspike.LIF(...)

To install or upgrade the layer on its own release cycle:

.. code-block:: bash

   pip install -U brainspike

See Also
--------

- :doc:`/project/ecosystem` — the wider BrainX ecosystem ``brainspike`` builds on.
- `Classic BrainPy documentation <https://brainpy.readthedocs.io/>`_ — the
  ``DynamicalSystem``-based API.
