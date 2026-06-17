============
Installation
============

``brainpy.state`` is a flexible, efficient, and extensible framework for
computational neuroscience and brain-inspired computation. This page gets you a
working install fast.

Requirements
============

- **Python 3.10 or later**
- The ``pip`` package manager
- Linux (Ubuntu 16.04+), macOS (10.12+), or Windows

``brainpy.state`` runs on top of JAX, so the only thing that changes between
hardware backends is which JAX build gets pulled in. The extras below
(``[cpu]``, ``[cuda12]``, ``[cuda13]``, ``[tpu]``) select that for you.

Basic installation
==================

Install the latest release with CPU support:

.. code-block:: bash

   pip install brainpy.state -U

This is enough for development, testing, and small-to-medium simulations.

Choose a backend
================

Pick the tab that matches your hardware. Each extra installs ``brainpy.state``
together with the matching JAX build.

.. tab-set::

   .. tab-item:: CPU

      .. code-block:: bash

         pip install brainpy.state[cpu] -U

      The default, portable choice. Great for development, testing, and
      small-scale simulations on any machine.

   .. tab-item:: CUDA 12

      .. code-block:: bash

         pip install brainpy.state[cuda12] -U

      NVIDIA GPU acceleration for the CUDA 12.x toolkit. Make sure your driver
      and CUDA toolkit match before installing.

   .. tab-item:: CUDA 13

      .. code-block:: bash

         pip install brainpy.state[cuda13] -U

      NVIDIA GPU acceleration for the CUDA 13.x toolkit.

   .. tab-item:: TPU

      .. code-block:: bash

         pip install brainpy.state[tpu] -U

      Google Cloud TPU support — typically used on Google Cloud Platform or in
      a Colab TPU runtime.

.. note::

   The backend extra only controls the JAX build. The ``brainpy.state`` API and
   all example code are identical across CPU, GPU, and TPU — write once, run
   anywhere.

Install the full BrainX ecosystem
=================================

To install ``brainpy.state`` alongside the rest of the ecosystem in one go:

.. code-block:: bash

   pip install BrainX -U

This pulls in:

- ``brainpy.state`` — the point-neuron modeling layer (this package)
- ``brainstate`` — state management and the compilation/``transform`` backend
- ``brainunit`` — the physical-units system
- ``braintools`` — initializers, optimizers, metrics, surrogate gradients, and
  visualization helpers
- ``braintrace`` — the linear-memory online-learning engine
- additional ecosystem packages

Verifying the install
=====================

Confirm everything imports and a model can be constructed:

.. code-block:: python

   import brainpy
   import brainstate
   import brainunit as u

   print(f"brainpy.state version: {brainpy.state.__version__}")
   print(f"brainstate version:    {brainstate.__version__}")

   # Build a small population of LIF neurons to check the API is live.
   neuron = brainpy.state.LIF(10)
   print("Installation successful!")

Backend notes
=============

.. dropdown:: CPU

   The CPU build is the default and needs no extra system libraries. If JAX
   reports falling back to CPU on a GPU machine, you installed the CPU extra (or
   a CPU-only JAX) — reinstall with the ``[cuda12]`` / ``[cuda13]`` extra.

.. dropdown:: GPU (CUDA)

   For NVIDIA GPU acceleration, ensure that:

   1. Your NVIDIA drivers are up to date.
   2. The installed CUDA toolkit matches the extra you chose (12.x or 13.x).
   3. Your GPU is CUDA-capable.

   If you installed a GPU extra but still see CUDA errors, the most common cause
   is a driver/toolkit version mismatch.

.. dropdown:: TPU

   Use the ``[tpu]`` extra inside a TPU-enabled environment (Google Cloud TPU
   VMs or a Colab TPU runtime). JAX discovers the TPU automatically once the
   matching build is installed.

Installing from source
======================

For development against a local checkout:

.. code-block:: bash

   git clone https://github.com/chaobrain/brainpy.state.git
   cd brainpy.state
   pip install -e .

This creates an editable install that reflects your local changes.

Next steps
==========

- Take the :doc:`5-minute tour </get-started/5-minute-tour>` to build and run
  your first network.
- Read the :doc:`mental model </get-started/mental-model>` to understand the
  four ideas the framework is built on.
- Dive into the :doc:`Core Concepts spine </concepts/index>` when you want the
  full story behind the design.
