Relationship between ``brainpy`` and ``brainpy.state``
======================================================

``brainpy.state`` is the state-based modeling layer of `BrainPy
<https://brainpy.readthedocs.io/>`_. It is developed and released as the standalone
``brainpy_state`` package and surfaced through the ``brainpy.state`` namespace, so the
same code is reachable as ``brainpy.state`` whether you install classic ``brainpy`` or
the layer on its own.

Why a separate package
----------------------

- **Decoupled release cycle.** ``brainpy.state`` is versioned and released independently
  of classic ``brainpy``, so its models, fixes, and features ship on their own cadence.
- **Surfaced through the ``brainpy.state`` namespace.** The standalone ``brainpy_state``
  distribution is re-exported by classic ``brainpy`` as ``brainpy.state`` — there is no
  separate import path to learn.
- **Built on the BrainX substrate.** It is built on ``brainstate`` (State management),
  ``brainunit`` (physical units), and the rest of the `BrainX ecosystem
  <https://brainx.chaobrain.com/>`_, and compiles through JAX to CPU, GPU, and TPU.

Which paradigm to use
---------------------

- Reach for **``brainpy.state``** for new work: ``State``-based models,
  surrogate-gradient / differentiable training, online learning, and JAX-native
  pipelines. This is the recommended starting point.
- Keep using **classic ``brainpy``** for existing ``DynamicalSystem`` models and for the
  array/operator and integrator API (``brainpy.math``, ``brainpy.integrators``,
  ``brainpy.dyn``). It is unchanged and fully supported.

The two paradigms coexist; adopting ``brainpy.state`` does not deprecate or remove any
part of classic ``brainpy``.

Installation
------------

``brainpy.state`` is bundled with classic ``brainpy`` (``brainpy >= 2.7.6``), so if you
installed ``brainpy`` you already have it — no code changes are required:

.. code-block:: python

   import brainpy

   neuron = brainpy.state.LIF(...)

To install or upgrade the layer on its own release cycle:

.. code-block:: bash

   pip install -U brainpy.state

See Also
--------

- :doc:`/project/ecosystem` — the wider BrainX ecosystem ``brainpy.state`` builds on.
- `Classic BrainPy documentation <https://brainpy.readthedocs.io/>`_ — the
  ``DynamicalSystem``-based API.
