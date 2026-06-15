.. _stdp-divergences:

==================================================================
STDP parity: where state lives and how spikes pair
==================================================================

.. warning::

   **Experimental — In Development.** The NEST-compatible model family is
   under active development. Parameter names, defaults, and numerical behavior
   may change without notice across 0.0.x releases. See the
   :doc:`NEST-style status page </nest-status/index>` for current scope and
   limitations.

.. currentmodule:: brainpy.state

This page is for users **porting NEST spike-timing-dependent plasticity (STDP)
code** to ``brainpy.state``. It documents two things the source alone will not
tell you at a glance:

1. **Where the learning state lives** — NEST keeps the post-synaptic eligibility
   trace (and several rule parameters) on the *neuron*; ``brainpy.state`` moves
   them onto the *synapse* (or a broadcast node) so a rule runs standalone, with
   no archiving-capable neuron required. Setting a parameter on the wrong object
   relative to NEST is the single most common porting mistake.
2. **How each ``stdp_nn_*`` variant pairs spikes** — the three nearest-neighbour
   schemes differ in subtle, load-bearing ways; this page states each pairing
   convention exactly and cites the NEST source line that defines it.

Every divergence below is backed by a live-NEST parity test (named per section);
this page consolidates what those tests proved.

.. contents:: On this page
   :local:
   :depth: 2


The online-substrate model in one paragraph
============================================

``brainpy.state`` STDP rules are pure ``update(state, ctx)`` kernels evaluated
**every time step** on a JAX event-driven substrate. The substrate maintains the
pre-synaptic ``K+`` and post-synaptic ``K-`` traces (decay-then-add per neuron,
gathered per edge) and the kernel applies **potentiation on the post spike** and
**depression on the pre spike**. NEST instead *defers* the weight integral to the
synapse's ``send()`` (the next pre spike), where its ``weight_recorder`` samples.
With no depression between pre spikes the two schemes apply the same operations in
the same order, so the weight **coincides at every pre-spike (send) time** — which
is exactly where parity is asserted. Keep this online-vs-deferred equivalence in
mind: it is why a few rules carry a small documented band rather than
bit-for-bit agreement (see :ref:`stdp-numerical-divergences`).


.. _stdp-tau-minus:

Trace storage: ``tau_minus`` is a synapse parameter here, a neuron parameter in NEST
====================================================================================

This is the canonical STDP trace-storage divergence.

In **NEST**, the post-synaptic trace ``K-`` is owned by the *postsynaptic
neuron*: ``tau_minus`` is an ``archiving_node`` parameter and the synapse reads
the trace back through ``get_K_value()`` at ``send()`` time. Setting
``tau_minus`` on the synapse in NEST has **no effect**:

.. code-block:: python

   # NEST: tau_minus is a parameter of the POSTSYNAPTIC NEURON (ArchivingNode).
   neuron = nest.Create("iaf_psc_alpha")
   nest.SetStatus(neuron, {"tau_minus": 20.0})        # ms — drives the K- trace
   nest.CopyModel("stdp_synapse", "syn", {"tau_plus": 20.0})
   # the synapse carries tau_plus; the K- time constant lives on the neuron.

In **brainpy.state**, ``tau_minus`` is a **synapse-spec attribute**. It drives the
substrate's per-post-neuron ``K-`` trace directly, so STDP runs without requiring
the post-synaptic neuron to implement an archiving API:

.. code-block:: python

   >>> import brainunit as u
   >>> import brainpy.state as bp
   >>> s = bp.stdp_synapse(tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms)
   >>> float(u.Quantity(s.tau_minus).to_decimal(u.ms))
   20.0

**Porting rule.** A NEST script that sets ``tau_minus`` on the neuron must set the
**same value on the brainpy.state synapse spec**. A live-NEST parity run sets both
the post neuron's ``tau_minus`` and the synapse's ``tau_minus`` to identical values
so the two ``K-`` traces match. The same move applies to every trace-based STDP
rule in the family (``stdp_synapse``, ``stdp_synapse_hom``, ``stdp_pl_synapse_hom``,
``stdp_triplet_synapse`` and its second post trace ``tau_minus_triplet``, the three
``stdp_nn_*`` variants, ``stdp_facetshw_synapse_hom``, and ``stdp_dopamine_synapse``).
``ht_synapse`` is trace-free and is therefore exempt.


.. _stdp-param-location:

Parameter-location map for the STDP family
==========================================

The trace-storage move generalises: wherever NEST precomputes a weight change on
an archiving neuron, ``brainpy.state`` relocates the governing parameters onto the
synapse spec (or, for the dopamine modulator, onto the ``volume_transmitter``) so
the rule is self-contained. Parity always sets identical values on both sides.

.. list-table::
   :header-rows: 1
   :widths: 24 30 30 30

   * - Parameter(s)
     - NEST home
     - brainpy.state home
     - Why it moves
   * - ``tau_minus``
       (``stdp_synapse``, ``_hom``, ``_pl_hom``, ``stdp_nn_*``,
       ``stdp_facetshw_synapse_hom``, ``stdp_dopamine_synapse``)
     - Post neuron — ``archiving_node`` (read via ``get_K_value``)
     - Synapse spec attribute (drives the substrate's per-post ``K-`` trace)
     - Run STDP without an archiving-capable postsynaptic neuron.
   * - ``tau_minus_triplet`` (``stdp_triplet_synapse``)
     - Post neuron — ``archiving_node`` (slow post trace)
     - Synapse spec attribute (second per-post trace)
     - Same as ``tau_minus``.
   * - ``A_LTP``, ``A_LTD``, ``theta_plus``, ``theta_minus``
       (``clopath_synapse``)
     - Post neuron — ``ClopathArchivingNode`` (precomputes Δw)
     - Synapse spec attributes
     - Self-containment; mirrors the ``tau_minus`` move.
   * - ``tau_u_bar_plus``, ``tau_u_bar_minus`` (Clopath voltage filters)
     - Post neuron (``aeif_psc_delta_clopath`` / ``hh_psc_alpha_clopath``)
     - **Stays on the neuron** (read per edge each step)
     - The low-pass voltage filters *are* neuron state; recorded once, reused.
   * - ``delay_u_bars`` (``clopath_synapse``)
     - Post neuron — archiving ring buffer (default ``4.0 ms``)
     - Synapse spec attribute; the online reader uses a one-step lag
     - No analog ring buffer online (see :ref:`stdp-clopath`).
   * - ``n`` (dopamine level, ``stdp_dopamine_synapse``)
     - Per-synapse state
     - On the ``volume_transmitter`` (a broadcast ``HiddenState``)
     - ``n`` depends only on ``tau_n`` + the shared dopamine train, so it is a
       node-level (not per-edge) quantity.
   * - ``tau_n`` (``stdp_dopamine_synapse``)
     - Common property, bound to the ``volume_transmitter``
     - On the ``volume_transmitter``; the spec's ``tau_n`` must match it
     - ``n`` lives on the transmitter, so its time constant does too.
   * - ``A_plus``, ``A_minus``, ``tau_plus``, ``tau_c``, ``b``, ``Wmin``,
       ``Wmax`` (``stdp_dopamine_synapse``)
     - Common properties (``CopyModel``)
     - Synapse spec attributes
     - Self-containment.


.. _stdp-numerical-divergences:

Documented numerical divergences
=================================

Three rules do not reproduce NEST bit-for-bit. In each case the **direction and
ordering of the weight change are exact**; only a small magnitude band remains,
and each band is asserted by a live-NEST parity test. Pull the numbers from here,
not from a fresh measurement.

.. _stdp-clopath:

Clopath ``delay_u_bars`` and online LTP (≤ 5 % band)
----------------------------------------------------

:What diverges: The stored weight after spike-pairing.
:Why: NEST evaluates LTP/LTD against post voltages held in a ring buffer delayed
   by ``delay_u_bars`` (default ``4.0 ms``); the online reader has no analog ring
   buffer and reads the post state with the substrate's intrinsic **one-step** lag,
   so parity sets ``delay_u_bars = 0.1 ms``. The ``4.0 ms`` default is **not
   reproducible** online. The residual is the online-instantaneous read versus
   NEST's deferred decayed-history sum; it grows with pairing frequency.
:Tolerance asserted: A **documented 5 % band** plus exact direction and
   frequency-ordering (realised ≤ 3.31 %; a pure-LTD train matches to 0.002 %).
:Parity test: ``brainpy_state/_nest/_validation/clopath_synapse_parity_test.py``.

.. _stdp-dopamine:

Dopamine online-vs-deferred integration (~0.2 % band)
-----------------------------------------------------

:What diverges: The weight trajectory under dopamine modulation.
:Why: ``brainpy.state`` integrates the weight **every step** against the broadcast
   ``n`` (with a one-step lag), whereas NEST integrates lazily at ``send()`` /
   ``trigger_update_weight``. Because ``n`` is a clean scalar carrying no analog
   history, the residual is far tighter than Clopath's.
:Tolerance asserted: A **~0.2 % band** (realised ``max|Δw| < 8e-3 pA`` over LTP /
   LTD / clamp / window sweeps spanning 50–200 pA), with direction and ordering
   exact.
:Parity test: ``brainpy_state/_nest/_validation/stdp_dopamine_synapse_parity_test.py``.

.. _stdp-phantom-pre:

Nearest-neighbour "phantom pre at 0" (``stdp_nn_symm`` / ``stdp_nn_restr`` only)
--------------------------------------------------------------------------------

:What diverges: A single facilitation against a *virtual* pre spike at ``t = 0``.
:Why: NEST's first ``send()`` initialises ``t_lastspike_ = 0``, so a post spike
   that precedes the first real pre facilitates against a phantom pre at the origin.
   The ``brainpy.state`` substrate seeds its traces and eligibility flags at 0,
   modelling the physically-correct "no pre has occurred" — so it does **not**
   reproduce the phantom pairing. ``stdp_nn_pre_centered_synapse`` and
   ``stdp_facetshw_synapse_hom`` are immune (their facilitation scales by a
   trace/flag that starts at 0).
:Tolerance asserted: Not reproduced (documented). Parity sidesteps it with a
   leading pre spike, or by placing the first pre late enough that the
   ``exp(-(Δ)/tau_plus)`` phantom term sits below the test's absolute tolerance.
:Parity tests: ``brainpy_state/_nest/_validation/stdp_nn_symm_synapse_parity_test.py``
   and ``…/stdp_nn_restr_synapse_parity_test.py``.


.. _stdp-nn-pairing:

Nearest-neighbour pairing conventions
=====================================

The pair-based weight maps are identical to :class:`stdp_synapse`
(potentiation on a post spike, depression on a pre spike):

.. math::

   \hat w \leftarrow \hat w + \lambda\,(1-\hat w)^{\mu_+}\, K^{+}
   \qquad\text{(post spike)}

   \hat w \leftarrow \hat w - \alpha\,\lambda\,\hat w^{\mu_-}\, K^{-}
   \qquad\text{(pre spike)}

with :math:`\hat w = w / W_{\max}`. What differs between the variants is **which
spikes contribute to** :math:`K^{+}` **and** :math:`K^{-}` — the pairing
convention. All three reproduce NEST to machine precision on single spike pairs
(realised abs. error ``~1e-13``–``1e-15``); each is named with its single-pair
regression test below.

The substrate supplies nearest-neighbour traces via a per-side
``pre_trace_mode`` / ``post_trace_mode`` of ``'nearest'`` (reset-to-1 on the
trace's own spike, decay otherwise), so the value gathered at a partner spike is
the **single nearest preceding pairing**, not the all-to-all sum:

.. code-block:: python

   >>> import brainpy.state as bp
   >>> sym = bp.stdp_nn_symm_synapse()
   >>> sym.pre_trace_mode, sym.post_trace_mode
   ('nearest', 'nearest')


.. _stdp-nn-symm:

``stdp_nn_symm_synapse`` — symmetric nearest-neighbour
------------------------------------------------------

Each spike pairs **only with its nearest partner on the other side**: a post
spike facilitates with the nearest *preceding* pre spike (:math:`K^{+}` from the
substrate's ``'nearest'`` pre trace), and a pre spike depresses with the nearest
*preceding* post spike (:math:`K^{-}` from the ``'nearest'`` post trace). The
kernel is byte-identical to :class:`stdp_synapse`; the nearest-ness lives entirely
in the trace mode.

* **NEST source:** ``models/stdp_nn_symm_synapse.h`` — ``send()`` lines 246–297
  (``facilitate_`` at 280; nearest ``Kminus`` depression at 286–287).
* **References:** Morrison, Aertsen & Diesmann (2007) *Neural Comput.*
  19:1437–1467; Morrison, Diesmann & Gerstner (2008) *Biol. Cybern.* 98:459–478,
  fig. 7A.
* **Divergence:** see :ref:`stdp-phantom-pre`.
* **Parity test:** ``brainpy_state/_nest/_validation/stdp_nn_symm_synapse_parity_test.py``.


.. _stdp-nn-restr:

``stdp_nn_restr_synapse`` — restricted symmetric nearest-neighbour
------------------------------------------------------------------

The symmetric scheme **plus a one-pairing-per-spike restriction**: a post spike
facilitates with the nearest preceding pre **only if a pre has occurred since the
previous post**, and a pre spike depresses with the nearest preceding post **only
if a post has occurred since the previous pre**. ``brainpy.state`` carries this
with two per-edge eligibility flags (``pre_avail`` / ``post_avail``): each spike
makes its own side available and consumes the opposite, reproducing NEST's
``start != finish`` gate without scanning history.

* **NEST source:** ``models/stdp_nn_restr_synapse.h`` — ``send()`` lines 244–307
  (facilitation gated on ``start != finish`` at 270–283; nearest ``Kminus``
  depression at 287–297).
* **References:** Morrison, Diesmann & Gerstner (2008) *Biol. Cybern.*
  98:459–478, fig. 7C.
* **Divergence:** see :ref:`stdp-phantom-pre`.
* **Parity test:** ``brainpy_state/_nest/_validation/stdp_nn_restr_synapse_parity_test.py``.


.. _stdp-nn-pre-centered:

``stdp_nn_pre_centered_synapse`` — presynaptic-centered nearest-neighbour
-------------------------------------------------------------------------

Asymmetric. A post spike facilitates with **all pre spikes since the last post**
(an *accumulated* :math:`K^{+}`), and that accumulator is **reset to 0 on every
post spike**; a pre spike depresses with the nearest preceding post (substrate
``'nearest'`` :math:`K^{-}`). Because the reset is triggered by the post spike,
two edges from one pre reset at different times, so :math:`K^{+}` is a **per-edge,
in-kernel** accumulate-then-reset trace rather than a substrate per-neuron trace —
hence ``pre_trace_tau is None``:

.. code-block:: python

   >>> import brainpy.state as bp
   >>> pc = bp.stdp_nn_pre_centered_synapse()
   >>> pc.pre_trace_tau is None, pc.post_trace_mode
   (True, 'nearest')

This variant is immune to :ref:`stdp-phantom-pre` (its facilitation scales by an
accumulator that starts at 0).

* **NEST source:** ``models/stdp_nn_pre_centered_synapse.h`` — ``send()`` lines
  249–317 (accumulated ``Kplus`` facilitation + reset at 287–294; nearest
  ``Kminus`` depression at 299–302; ``Kplus`` decay/+1 at 304).
* **References:** Morrison, Diesmann & Gerstner (2008) *Biol. Cybern.*
  98:459–478, fig. 7B; Izhikevich & Desai (2003) *Neural Comput.* 15:1511–1523.
* **Parity test:** ``brainpy_state/_nest/_validation/stdp_nn_pre_centered_synapse_parity_test.py``.


.. _stdp-facetshw:

``stdp_facetshw_synapse_hom`` — hardware (FACETS) charge accumulation + LUT readout
-----------------------------------------------------------------------------------

A hardware-emulating rule whose weight is **quantised to a 4-bit index**
(``wple = Wmax / 15``). Between periodic *readouts* it accumulates two charges in
the substrate's ``'nearest'`` mode:

* ``a_causal`` — the nearest pre before this post, paired with the post
  (potentiation evidence);
* ``a_acausal`` — the nearest post before this pre, paired with the pre
  (depression evidence).

A readout (the first pre past each ``readout_cycle_duration``) does
quantise → two ``eval_function`` evaluations → LUT lookup → reset → advance clock.
The readout runs **before** folding its triggering pair, so a readout never sees
its own pair's charge — an observable ordering that the parity test pins.

* **NEST keys / scope:** the common ``tau_minus`` is exposed as ``tau_minus_stdp``;
  ``a_thresh_th`` / ``a_thresh_tl`` are per-synapse; ``no_synapses`` and
  ``readout_cycle_duration`` auto-compute (do not set them). Single-driver scope
  (``E ≤ synapses_per_driver``). On-grid weights (e.g. ``5 * wple``) avoid the
  ``weight = 1.0`` footgun, which quantises to index 0 and is zeroed by the first
  readout.
* **NEST source:** ``models/stdp_facetshw_synapse_hom.h`` and ``…_impl.h``
  (``send()``: readout, quantisation, LUT update).
* **Parity test:** ``brainpy_state/_nest/_validation/stdp_facetshw_synapse_hom_parity_test.py``.


See also
========

- :doc:`/api/nest-plasticity` — the full NEST-compatible plasticity model list.
- :doc:`/nest-status/index` — overall NEST-compatibility scope and limitations.
- `NEST simulator documentation <https://nest-simulator.readthedocs.io/>`_ — the
  authoritative reference for upstream model semantics.
