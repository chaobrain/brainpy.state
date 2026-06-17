===========
Get Started
===========

``brainpy.state`` is the point-neuron modeling layer of the BrainX ecosystem:
**one differentiable substrate** that serves two audiences from the same code.

- If you come from **computational neuroscience**, you build biophysical
  spiking networks, run them efficiently, and analyze their dynamics.
- If you come from **brain-inspired computing / SNN-ML**, the very same models
  are differentiable — you train them with surrogate gradients and scale them
  with linear-memory online learning.

These three pages take you from a clean install to a running network to the
mental model that makes the rest of the documentation click.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: 1 · Install
      :link: /get-started/installation
      :link-type: doc

      Get a working install for CPU, GPU (CUDA), or TPU — or pull the whole
      BrainX ecosystem in one command.

   .. grid-item-card:: 2 · 5-minute tour
      :link: /get-started/5-minute-tour
      :link-type: doc

      Build and run an excitatory–inhibitory balanced network, then plot a
      spike raster. The fastest way to feel what ``brainpy.state`` does.

   .. grid-item-card:: 3 · Mental model
      :link: /get-started/mental-model
      :link-type: doc

      The four big ideas — state, units, composition, and ``transform`` — plus
      the "two worlds, one substrate" framing that runs through the whole site.

Where to go next
================

After Get Started, the :doc:`/concepts/index` spine explains *why* the pieces
fit together — most importantly the keystone
:doc:`/concepts/alignpre-alignpost` design. From there, pick your track:
:doc:`/brainpy-style/index` for native modeling and training, or the
:doc:`/apis/index` to look up a specific model.

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   5-minute-tour
   mental-model
