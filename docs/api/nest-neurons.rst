.. warning::

   **Experimental — In Development.** The NEST-compatible model family is
   under active development. Parameter names, defaults, numerical behavior,
   and the set of available models may change without notice across 0.0.x
   releases. See the :doc:`NEST-style status page </nest-status/index>` for
   current scope and limitations.


NEST-Compatible Neuron Models
=============================

Neuron models compatible with the `NEST simulator <https://nest-simulator.readthedocs.io/>`_.

.. currentmodule:: brainpy.state


IAF Neurons — Current-Based (psc)
----------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   iaf_psc_delta
   iaf_psc_delta_ps
   iaf_psc_alpha
   iaf_psc_alpha_multisynapse
   iaf_psc_alpha_ps
   iaf_psc_exp
   iaf_psc_exp_multisynapse
   iaf_psc_exp_htum
   iaf_psc_exp_ps
   iaf_psc_exp_ps_lossless


IAF Neurons — Conductance-Based (cond)
---------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   iaf_cond_alpha
   iaf_cond_alpha_mc
   iaf_cond_beta
   iaf_cond_exp
   iaf_cond_exp_sfa_rr


IAF Neurons — Specialized Variants
------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   iaf_bw_2001
   iaf_bw_2001_exact
   iaf_chs_2007
   iaf_chxk_2008
   iaf_tum_2000


Adaptive Exponential IF (AdEx) Neurons
--------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   aeif_cond_alpha
   aeif_cond_alpha_astro
   aeif_cond_alpha_multisynapse
   aeif_cond_beta_multisynapse
   aeif_cond_exp
   aeif_psc_alpha
   aeif_psc_delta
   aeif_psc_delta_clopath
   aeif_psc_exp


Generalized IF (GIF) Neurons
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   gif_cond_exp
   gif_cond_exp_multisynapse
   gif_pop_psc_exp
   gif_psc_exp
   gif_psc_exp_multisynapse


Multi-Timescale Adaptive Threshold (MAT) Neurons
-------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   mat2_psc_exp
   amat2_psc_exp


Generalized LIF (GLIF) Neurons — Allen Institute
-------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   glif_cond
   glif_psc
   glif_psc_double_alpha


Hodgkin-Huxley Family
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   hh_psc_alpha
   hh_psc_alpha_clopath
   hh_psc_alpha_gap
   hh_cond_exp_traub
   hh_cond_beta_gap_traub
   ht_neuron


Izhikevich Neuron
-----------------

.. autosummary::
   :toctree: generated/nest/
   :nosignatures:
   :template: classtemplate.rst

   izhikevich


Point Process Neurons
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   pp_psc_delta
   pp_cond_exp_mc_urbanczik


Binary Neurons
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   mcculloch_pitts_neuron
   ginzburg_neuron
   erfc_neuron


Rate Neurons
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   lin_rate_ipn
   lin_rate_opn
   tanh_rate_ipn
   tanh_rate_opn
   sigmoid_rate_ipn
   sigmoid_rate_gg_1998_ipn
   gauss_rate_ipn
   threshold_lin_rate_ipn
   threshold_lin_rate_opn
   rate_neuron_ipn
   rate_neuron_opn
   rate_transformer_node
   siegert_neuron


Other Spiking Neurons
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ignore_and_fire
