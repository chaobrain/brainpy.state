BrainPy-style Examples
======================

Complete, runnable examples built with the native ``brainpy.state`` API:
composable neurons, synapses, AlignPre/AlignPost projections, and
surrogate-gradient training. Every script lives in the `examples/brainpy_like/
<https://github.com/chaobrain/brainpy.state/tree/main/examples/brainpy_like>`_
directory of the repository.

Each card is tagged along two axes:

.. admonition:: How to read the tags
   :class: note

   - **World** — **Simulation** reproduces neuroscience dynamics; **Training**
     learns weights by gradient descent through the spiking model.
   - **Difficulty** — *beginner* (single mechanism, short), *intermediate*
     (a full network or training loop), *advanced* (multi-mechanism models or
     paper-scale reproductions).

   These tags describe scope, not stability — every example is production-ready.

If you are new here, start with :doc:`/get-started/5-minute-tour` and the
:doc:`/brainpy-style/tutorials/01-first-neuron` tutorial track, then come back to
adapt one of these.

Balanced & E/I networks
-----------------------

The canonical excitatory-inhibitory balanced networks — the natural home of the
AlignPost projection (see :doc:`/concepts/alignpre-alignpost`).

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Brunel random network (Builder)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brunel.py

      Two populations, random fixed-indegree COBA wiring, built with the
      imperative ``Builder`` API — the flagship Network-API example.

      **Simulation** · beginner

   .. grid-item-card:: COBA balanced network (2005)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/103_COBA_2005.py

      Vogels-Abbott conductance-based E/I network with ``AlignPostProj`` +
      ``Expon`` + ``COBA``. The reference simulation pattern.

      **Simulation** · beginner

   .. grid-item-card:: CUBA balanced network (2005)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/104_CUBA_2005.py

      The current-based variant of Vogels-Abbott — simpler and faster than COBA.

      **Simulation** · beginner

   .. grid-item-card:: CUBA balanced network (variant)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/104_CUBA_2005_version2.py

      A second construction of the CUBA network showing an alternative wiring
      style.

      **Simulation** · intermediate

   .. grid-item-card:: EI balanced network (1996)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/102_EI_net_1996.py

      Van Vreeswijk & Sompolinsky balanced network exhibiting chaotic
      asynchronous-irregular firing.

      **Simulation** · intermediate

   .. grid-item-card:: COBA with Hodgkin-Huxley neurons (2007)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/106_COBA_HH_2007.py

      The COBA benchmark rebuilt on biophysically detailed Hodgkin-Huxley cells.

      **Simulation** · intermediate

Oscillations & rhythms
----------------------

Network rhythms — gamma generation, synchrony, and fast oscillations.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Interneuron gamma (1996)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/107_gamma_oscillation_1996.py

      Wang & Buzsaki inhibition-based gamma (30-80 Hz) in a hippocampal
      interneuron network.

      **Simulation** · intermediate

   .. grid-item-card:: Synfire chains (1999)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/108_synfire_chains_199.py

      Diesmann et al. reliable propagation of synchronous spike volleys through a
      feedforward chain.

      **Simulation** · intermediate

   .. grid-item-card:: Fast global oscillation
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/109_fast_global_oscillation.py

      Brunel & Hakim fast (>100 Hz) global oscillations in sparsely-connected
      inhibitory networks with low firing rates.

      **Simulation** · intermediate

   .. grid-item-card:: Gamma mechanisms — AI state (2021)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/110_Susin_Destexhe_2021_gamma_oscillation_AI.py

      Susin & Destexhe asynchronous-irregular background state (no rhythm).

      **Simulation** · advanced

   .. grid-item-card:: Gamma mechanisms — CHING (2021)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/111_Susin_Destexhe_2021_gamma_oscillation_CHING.py

      Susin & Destexhe coherent high-frequency inhibition-based gamma.

      **Simulation** · advanced

   .. grid-item-card:: Gamma mechanisms — ING (2021)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/112_Susin_Destexhe_2021_gamma_oscillation_ING.py

      Susin & Destexhe pure-inhibitory (ING) gamma generation.

      **Simulation** · advanced

   .. grid-item-card:: Gamma mechanisms — PING (2021)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/113_Susin_Destexhe_2021_gamma_oscillation_PING.py

      Susin & Destexhe pyramidal-interneuron (PING) gamma — the E-I loop.

      **Simulation** · advanced

   .. grid-item-card:: Gamma mechanisms — combined driver (2021)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/Susin_Destexhe_2021_gamma_oscillation.py

      The combined Susin & Destexhe model exploring all four gamma regimes in one
      script.

      **Simulation** · advanced

Reproductions & large models
----------------------------

Paper-scale simulations and signal-propagation studies.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Cortical microcircuit model
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/203_cortical_model.py

      A multi-layer cortical column model with layer-specific LIF populations and
      connectivity.

      **Simulation** · advanced

   .. grid-item-card:: Inter-areal propagation (Joglekar 2018)
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/204_joglekar_2018_propagation.py

      Joglekar et al. graded vs all-or-none signal propagation across a
      connectome-constrained network of cortical areas.

      **Simulation** · advanced

Training spiking networks
-------------------------

Gradient-based training through the spiking model with surrogate gradients (see
:doc:`/concepts/differentiability`).

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Surrogate-gradient LIF
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/200_surrogate_grad_lif.py

      Train a small spiking network on a synthetic task with surrogate gradients
      (the spytorch tutorial 1 reproduction).

      **Training** · beginner

   .. grid-item-card:: Fashion-MNIST SNN
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/201_surrogate_grad_lif_fashion_mnist.py

      A multi-layer spiking network classifying Fashion-MNIST end to end
      (spytorch tutorials 2 & 3).

      **Training** · intermediate

   .. grid-item-card:: MNIST with a rate readout
      :link: https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/202_mnist_lif_readout.py

      MNIST classification with a ``LeakyRateReadout`` decoding spikes into class
      scores.

      **Training** · intermediate

See Also
--------

- :doc:`/examples/nest-gallery` — the NEST-compatible example gallery.
- :doc:`/brainpy-style/index` — tutorials and task-oriented how-to guides.
- :doc:`/concepts/alignpre-alignpost` — the projection design behind the network
  examples.
