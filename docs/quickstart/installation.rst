Installation Guide
==================

``brainpy.state`` is a flexible, efficient, and extensible framework for computational neuroscience and
brain-inspired computation. This guide will help you install ``brainpy.state`` on your system.

Requirements
------------

- Python 3.10 or later
- pip package manager
- Supported platforms: Linux (Ubuntu 16.04+), macOS (10.12+), Windows

Basic Installation
------------------

Install the latest version of ``brainpy.state``:

.. code-block:: bash

   pip install brainpy-state -U

This will install brainpy.state with CPU support by default.

Hardware-Specific Installation
-------------------------------

Depending on your hardware, you can install ``brainpy.state`` with optimized support:

CPU Only
~~~~~~~~

For CPU-only installations:

.. code-block:: bash

   pip install brainpy-state[cpu] -U

This is suitable for development, testing, and small-scale simulations.

GPU Support (CUDA)
~~~~~~~~~~~~~~~~~~

For NVIDIA GPU acceleration:

**CUDA 12.x:**

.. code-block:: bash

   pip install brainpy-state[cuda12] -U

**CUDA 13.x:**

.. code-block:: bash

   pip install brainpy-state[cuda13] -U

.. note::
   Make sure you have the appropriate CUDA toolkit installed on your system before installing the GPU version.

TPU Support
~~~~~~~~~~~

For Google Cloud TPU support:

.. code-block:: bash

   pip install brainpy-state[tpu] -U

This is typically used when running on Google Cloud Platform or Colab with TPU runtime.

Ecosystem Installation
----------------------

To install ``brainpy.state`` along with the entire ecosystem of tools:

.. code-block:: bash

   pip install BrainX -U

This includes:

- ``brainpy-state``: Main framework
- ``brainstate``: State management and compilation backend
- ``saiunit``: Physical units system
- ``braintools``: Utilities and tools
- Additional ecosystem packages

Verifying Installation
----------------------

To verify that ``brainpy.state`` is installed correctly:

.. code-block:: python

   import brainpy.state
   import brainstate
   import saiunit as u

   print(f"brainpy.state version: {brainpy.__version__}")
   print(f"BrainState version: {brainstate.__version__}")

   # Test basic functionality
   neuron = brainpy.state.LIF(10)
   print("Installation successful!")

Development Installation
------------------------

If you want to install brainpy.state from source for development:

.. code-block:: bash

   git clone https://github.com/chaobrain/brainpy.state.git
   cd brainpy.state
   pip install -e .

This creates an editable installation that reflects your local changes.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'brainpy.state'**

Make sure you've activated the correct Python environment and that the installation completed successfully.

**CUDA not found**

If you installed the GPU version but get CUDA errors, ensure that:

1. Your NVIDIA drivers are up to date
2. CUDA toolkit is installed and matches the version (12.x or 13.x)
3. Your GPU is CUDA-capable

**Version Conflicts**

If you're upgrading from a previous version, you might need to uninstall the old version first:

.. code-block:: bash

   pip uninstall brainpy-state
   pip install brainpy-state -U

Getting Help
~~~~~~~~~~~~

If you encounter issues:

- Check the `GitHub Issues <https://github.com/chaobrain/brainpy.state/issues>`_
- Read the documentation at `https://brainpy-state.readthedocs.io/ <https://brainpy-state.readthedocs.io/>`_
- Join our community discussions

Next Steps
----------

Now that you have brainpy.state installed, you can:

- Follow the :doc:`5-minute tutorial <5min-tutorial>` for a quick introduction
- Read about :doc:`core concepts <../core-concepts/index>` to understand brainpy.state's architecture
- Explore the `examples <https://github.com/chaobrain/brainpy.state/tree/main/examples>`_ for detailed guides
